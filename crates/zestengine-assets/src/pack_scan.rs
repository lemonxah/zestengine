//! Recognise what a downloaded asset pack holds: a folder (or an extracted
//! zip) of models and textures from Quixel Megascans, Fab or anywhere else.
//!
//! Two layouts are understood:
//!
//! - **Megascans**: files named `<id>[_VarX][_LODn].fbx` and
//!   `<id>_<res>[_Billboard]_<Role>.jpg`, with an optional `<id>.json`
//!   (checked against the layout of the Fab downloads on the author's machine:
//!   one folder per id, no JSON, `Albedo` or `BaseColor`, both `Roughness` and
//!   `Gloss`, `Billboard_*` impostor maps beside a plant).
//! - **Generic** (Fab, Sketchfab, Unreal / Substance / Unity exports): models
//!   anywhere, textures classified by name suffix ([`crate::tex_roles`]) and
//!   grouped into material sets by prefix.
//!
//! Scanning only reads names and sizes; nothing is decoded or copied.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

use crate::megascans::{AssetKind, MegascansMeta, lod_of_name, read_megascans_json};
use crate::tex_roles::{Confidence, TexRole, UNREADABLE_IMAGE_EXTS, classify, is_image_ext, match_key};

/// Model extensions the engine loads, most preferred first.
pub const MODEL_EXTS: &[&str] = &["fbx", "glb", "gltf", "vrm"];

/// Model-like files the engine cannot read (listed in the report).
const FOREIGN_MODEL_EXTS: &[&str] = &["obj", "abc", "usd", "usda", "usdc", "usdz", "blend", "max", "ma", "mb", "c4d", "3ds", "dae", "stl", "ply", "uasset", "umap", "zpr"];

const MAX_FILES: usize = 100_000;
const MAX_DEPTH: usize = 12;

#[derive(Clone, Debug)]
pub struct ModelLod {
    pub lod: u32,
    pub path: PathBuf,
    pub bytes: u64,
}

/// One mesh with its LOD chain (or one standalone model file).
#[derive(Clone, Debug)]
pub struct ModelVariant {
    /// `VarA`, a file stem, or empty for the only mesh of an asset.
    pub name: String,
    /// Sorted by LOD number, finest first.
    pub lods: Vec<ModelLod>,
}

#[derive(Clone, Debug)]
pub struct SetTexture {
    pub path: PathBuf,
    /// Editable in the import dialog.
    pub role: TexRole,
    pub confidence: Confidence,
    pub resolution: Option<u32>,
    pub bytes: u64,
}

/// Textures that make one material.
#[derive(Clone, Debug)]
pub struct TextureSet {
    pub name: String,
    /// Megascans billboard impostor maps (the engine builds its own impostor).
    pub billboard: bool,
    pub textures: Vec<SetTexture>,
}

#[derive(Clone, Debug)]
pub struct PackAsset {
    /// Megascans id, or the asset's file / folder name.
    pub id: String,
    pub name: String,
    pub kind: AssetKind,
    pub megascans: bool,
    /// Folder the asset's main files sit in.
    pub dir: PathBuf,
    pub variants: Vec<ModelVariant>,
    pub sets: Vec<TextureSet>,
    /// Real-world size in metres of the tiling area (surfaces, decals, atlases).
    /// Editable; `None` = tile at 1 UV unit per metre.
    pub size_m: Option<[f32; 2]>,
    pub warnings: Vec<String>,
    pub json: Option<MegascansMeta>,
    /// The user's choice in the dialog.
    pub include: bool,
}

impl PackAsset {
    pub fn texture_count(&self) -> usize {
        self.sets.iter().map(|s| s.textures.len()).sum()
    }

    pub fn model_bytes(&self) -> u64 {
        self.variants.iter().flat_map(|v| &v.lods).map(|l| l.bytes).sum()
    }

    pub fn texture_bytes(&self) -> u64 {
        self.sets.iter().flat_map(|s| &s.textures).map(|t| t.bytes).sum()
    }
}

#[derive(Clone, Debug, Default)]
pub struct PackScan {
    pub root: PathBuf,
    pub assets: Vec<PackAsset>,
    /// Pack-level notes: unsupported files, images the engine cannot read.
    pub warnings: Vec<String>,
}

#[derive(Default)]
struct Found {
    models: Vec<(PathBuf, u64)>,
    images: Vec<(PathBuf, u64)>,
    jsons: Vec<PathBuf>,
    foreign: Vec<PathBuf>,
    unreadable_images: Vec<PathBuf>,
}

fn walk(dir: &Path, depth: usize, out: &mut Found, count: &mut usize) -> Result<()> {
    if depth > MAX_DEPTH {
        return Ok(());
    }
    let mut entries: Vec<_> = std::fs::read_dir(dir)?.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        *count += 1;
        if *count > MAX_FILES {
            bail!("more than {MAX_FILES} files: not an asset pack?");
        }
        let name = e.file_name().to_string_lossy().into_owned();
        let Ok(ft) = e.file_type() else { continue };
        // Never follow links out of the pack.
        if ft.is_symlink() || name.starts_with('.') || name == "__MACOSX" {
            continue;
        }
        let path = e.path();
        if ft.is_dir() {
            walk(&path, depth + 1, out, count)?;
            continue;
        }
        let ext = path.extension().map(|x| x.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
        let bytes = e.metadata().map(|m| m.len()).unwrap_or(0);
        if MODEL_EXTS.contains(&ext.as_str()) {
            out.models.push((path, bytes));
        } else if is_image_ext(&ext) {
            out.images.push((path, bytes));
        } else if ext == "json" {
            out.jsons.push(path);
        } else if FOREIGN_MODEL_EXTS.contains(&ext.as_str()) {
            out.foreign.push(path);
        } else if UNREADABLE_IMAGE_EXTS.contains(&ext.as_str()) {
            out.unreadable_images.push(path);
        }
    }
    Ok(())
}

/// A Megascans asset id: 5 to 12 lowercase letters / digits.
pub fn is_megascans_id(s: &str) -> bool {
    (5..=12).contains(&s.len()) && s.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
}

fn format_rank(path: &Path) -> usize {
    let ext = path.extension().map(|x| x.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    MODEL_EXTS.iter().position(|e| *e == ext).unwrap_or(usize::MAX)
}

/// Split a model stem into `(base, variant, lod)`: `id_VarB_LOD2` ->
/// (`id`, `VarB`, Some(2)). The base is what is left before the markers.
pub fn split_model_stem(stem: &str) -> (String, String, Option<u32>) {
    let lod = lod_of_name(stem);
    let mut base: Vec<&str> = Vec::new();
    let mut variant = String::new();
    for tok in stem.split('_') {
        let low = tok.to_ascii_lowercase();
        let is_lod = low.starts_with("lod") && low[3..].chars().all(|c| c.is_ascii_digit());
        let is_var = low.starts_with("var") && low.len() > 3 && low[3..].chars().all(|c| c.is_ascii_alphanumeric());
        if is_lod {
            continue;
        }
        if is_var && variant.is_empty() {
            variant = tok.to_owned();
        } else {
            base.push(tok);
        }
    }
    (base.join("_"), variant, lod)
}

/// One model-variant's LOD chain: `lod -> candidates`.
type LodGroup = BTreeMap<u32, Vec<(PathBuf, u64)>>;

/// Group model files into variants with sorted LOD chains. Several formats of
/// the same mesh keep the most preferred one; the rest are returned in the
/// second list (for the report).
fn group_models(files: &[(PathBuf, u64)]) -> (Vec<ModelVariant>, Vec<PathBuf>) {
    // (base, variant) -> lod -> candidates
    let mut groups: BTreeMap<(String, String), LodGroup> = BTreeMap::new();
    for (p, bytes) in files {
        let stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let (base, variant, lod) = split_model_stem(&stem);
        groups.entry((base, variant)).or_default().entry(lod.unwrap_or(0)).or_default().push((p.clone(), *bytes));
    }
    let mut ignored = Vec::new();
    let variants = groups
        .into_iter()
        .map(|((base, variant), lods)| {
            let lods = lods
                .into_iter()
                .map(|(lod, mut c)| {
                    c.sort_by_key(|(p, _)| format_rank(p));
                    let (path, bytes) = c.remove(0);
                    ignored.extend(c.into_iter().map(|x| x.0));
                    ModelLod { lod, path, bytes }
                })
                .collect();
            // A lone untagged mesh has no variant name; keep the base for generic packs.
            let name = if variant.is_empty() { base } else { variant };
            ModelVariant { name, lods }
        })
        .collect();
    (variants, ignored)
}

/// One classified image: (path, bytes, classification).
type ClassifiedImage = (PathBuf, u64, crate::tex_roles::Classified);

/// One owned path entry: (directory, models under it, images under it).
type OwnedEntry = (PathBuf, Vec<(PathBuf, u64)>, Vec<(PathBuf, u64)>);

fn classify_images(images: &[(PathBuf, u64)]) -> (Vec<ClassifiedImage>, Vec<(PathBuf, u64)>) {
    let mut ok = Vec::new();
    let mut unknown = Vec::new();
    for (p, b) in images {
        let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        match classify(&name) {
            Some(c) => ok.push((p.clone(), *b, c)),
            None => unknown.push((p.clone(), *b)),
        }
    }
    (ok, unknown)
}

/// Build sets from classified textures: one per (prefix, billboard).
fn build_sets(classified: Vec<(PathBuf, u64, crate::tex_roles::Classified)>, unknown: Vec<(PathBuf, u64)>, fallback_name: &str) -> Vec<TextureSet> {
    let mut sets: BTreeMap<(String, bool), TextureSet> = BTreeMap::new();
    for (path, bytes, c) in classified {
        let name = if c.prefix.is_empty() { fallback_name.to_owned() } else { c.prefix.clone() };
        let set = sets.entry((name.to_ascii_lowercase(), c.billboard)).or_insert_with(|| TextureSet { name, billboard: c.billboard, textures: Vec::new() });
        set.textures.push(SetTexture { path, role: c.role, confidence: c.confidence, resolution: c.resolution, bytes });
    }
    if !unknown.is_empty() {
        // Names with no readable role: kept (so the user can assign one), unused until then.
        let set = sets.entry(("unsorted".into(), false)).or_insert_with(|| TextureSet { name: "Unsorted".into(), billboard: false, textures: Vec::new() });
        for (path, bytes) in unknown {
            set.textures.push(SetTexture { path, role: TexRole::Ignore, confidence: Confidence::Weak, resolution: None, bytes });
        }
    }
    sets.into_values().collect()
}

fn kind_without_json(has_models: bool, sets: &[TextureSet], name: &str) -> AssetKind {
    let main: Vec<&TextureSet> = sets.iter().filter(|s| !s.billboard).collect();
    let has = |r: TexRole| main.iter().any(|s| s.textures.iter().any(|t| t.role == r));
    let low = name.to_ascii_lowercase();
    if low.contains("decal") {
        return AssetKind::Decal;
    }
    if has_models {
        if sets.iter().any(|s| s.billboard) || (has(TexRole::Opacity) && has(TexRole::Translucency)) {
            AssetKind::Plant
        } else {
            AssetKind::Model3d
        }
    } else if has(TexRole::Albedo) || has(TexRole::Normal) || has(TexRole::Height) {
        AssetKind::Surface
    } else {
        AssetKind::Textures
    }
}

fn megascans_assets(found: &mut Found, root: &Path, warnings: &mut Vec<String>) -> Vec<PackAsset> {
    // id -> (dir, images, models, json)
    #[derive(Default)]
    struct Parts {
        dir: PathBuf,
        images: Vec<(PathBuf, u64)>,
        models: Vec<(PathBuf, u64)>,
        json: Option<MegascansMeta>,
    }
    let mut parts: BTreeMap<(PathBuf, String), Parts> = BTreeMap::new();
    let dir_of = |p: &Path| p.parent().unwrap_or(root).to_owned();

    // Ids come from `<id>_<res>_<Role>` texture names and `<id>.json` files.
    for (p, _) in &found.images {
        let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if let Some(c) = classify(&name)
            && c.resolution.is_some()
            && is_megascans_id(&c.prefix)
        {
            parts.entry((dir_of(p), c.prefix)).or_default();
        }
    }
    let mut json_left = Vec::new();
    for j in std::mem::take(&mut found.jsons) {
        let stem = j.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        match read_megascans_json(&j) {
            Some(meta) => {
                let id = if meta.id.is_empty() { stem } else { meta.id.clone() };
                parts.entry((dir_of(&j), id)).or_default().json = Some(meta);
            }
            None => json_left.push(j),
        }
    }
    found.jsons = json_left;
    if parts.is_empty() {
        return Vec::new();
    }
    for (key, part) in parts.iter_mut() {
        part.dir = key.0.clone();
    }
    // Claim the files each id owns (same folder, name starts with the id).
    let claims = |p: &Path, parts: &BTreeMap<(PathBuf, String), Parts>| -> Option<(PathBuf, String)> {
        let name = p.file_name()?.to_string_lossy().into_owned();
        let d = dir_of(p);
        parts.keys().filter(|(pd, id)| *pd == d && name.starts_with(id.as_str())).max_by_key(|(_, id)| id.len()).cloned()
    };
    let images = std::mem::take(&mut found.images);
    for (p, b) in images {
        match claims(&p, &parts) {
            Some(k) => parts.get_mut(&k).unwrap().images.push((p, b)),
            None => found.images.push((p, b)),
        }
    }
    let models = std::mem::take(&mut found.models);
    for (p, b) in models {
        match claims(&p, &parts) {
            Some(k) => parts.get_mut(&k).unwrap().models.push((p, b)),
            None => found.models.push((p, b)),
        }
    }

    let mut assets = Vec::new();
    for ((_, id), part) in parts {
        if part.images.is_empty() && part.models.is_empty() {
            continue;
        }
        let (variants, ignored) = group_models(&part.models);
        let (classified, unknown) = classify_images(&part.images);
        let sets = build_sets(classified, unknown, &id);
        let mut asset_warnings = Vec::new();
        if !ignored.is_empty() {
            asset_warnings.push(format!("{} model file(s) in other formats ignored (one format per mesh and LOD is used)", ignored.len()));
        }
        let json = part.json;
        let name = json.as_ref().map(|m| m.name.clone()).filter(|n| !n.is_empty()).unwrap_or_else(|| id.clone());
        let kind = json.as_ref().and_then(|m| m.kind).unwrap_or_else(|| kind_without_json(!variants.is_empty(), &sets, &name));
        let size_m = json.as_ref().and_then(|m| m.size_m);
        if json.is_none() {
            asset_warnings.push("no export JSON: the asset type and real-world size are guessed from the files".into());
        }
        assets.push(PackAsset { id, name, kind, megascans: true, dir: part.dir, variants, sets, size_m, warnings: asset_warnings, json, include: true });
    }
    if assets.len() > 1 {
        warnings.push(format!("{} Megascans assets found", assets.len()));
    }
    assets
}

/// Is `dir` equal to or inside `ancestor`?
fn within(dir: &Path, ancestor: &Path) -> bool {
    dir.starts_with(ancestor)
}

fn generic_assets(found: Found, root: &Path, warnings: &mut Vec<String>) -> Vec<PackAsset> {
    let root_name = root.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "pack".into());
    let root_name = root_name.strip_suffix(".zip").map(str::to_owned).unwrap_or(root_name);
    // Asset roots: folders that directly hold models.
    let mut roots: Vec<PathBuf> = Vec::new();
    for (p, _) in &found.models {
        let d = p.parent().unwrap_or(root).to_owned();
        if !roots.contains(&d) {
            roots.push(d);
        }
    }
    if roots.is_empty() {
        // Textures only: one asset per folder of images.
        for (p, _) in &found.images {
            let d = p.parent().unwrap_or(root).to_owned();
            if !roots.contains(&d) {
                roots.push(d);
            }
        }
    }
    let mut owned: Vec<OwnedEntry> = roots.iter().map(|r| (r.clone(), Vec::new(), Vec::new())).collect();
    for (p, b) in found.models {
        let d = p.parent().unwrap_or(root).to_owned();
        if let Some(o) = owned.iter_mut().find(|o| o.0 == d) {
            o.1.push((p, b));
        }
    }
    let model_assets = owned.iter().any(|o| !o.1.is_empty());
    for (p, b) in found.images {
        let d = p.parent().unwrap_or(root).to_owned();
        let stem_key = p.file_stem().map(|s| match_key(&s.to_string_lossy())).unwrap_or_default();
        // Nearest enclosing asset root; else the only root; else the best name match.
        let idx = owned
            .iter()
            .enumerate()
            .filter(|(_, o)| within(&d, &o.0))
            .max_by_key(|(_, o)| o.0.components().count())
            .map(|(i, _)| i)
            .or_else(|| (owned.len() == 1).then_some(0))
            .or_else(|| {
                owned.iter().position(|o| {
                    o.1.iter().any(|(m, _)| {
                        let mk = m.file_stem().map(|s| match_key(&s.to_string_lossy())).unwrap_or_default();
                        !mk.is_empty() && stem_key.contains(&mk)
                    })
                })
            })
            .or_else(|| {
                // Same parent folder as an asset's folder.
                owned.iter().position(|o| o.0.parent() == d.parent() && model_assets)
            })
            .unwrap_or(0);
        if owned.is_empty() {
            owned.push((root.to_owned(), Vec::new(), Vec::new()));
        }
        let k = idx.min(owned.len() - 1);
        owned[k].2.push((p, b));
    }
    let mut assets = Vec::new();
    for (dir, models, images) in owned {
        if models.is_empty() && images.is_empty() {
            continue;
        }
        let (variants, ignored) = group_models(&models);
        let (classified, unknown) = classify_images(&images);
        let single_name = (variants.len() == 1).then(|| variants[0].name.clone()).filter(|n| !n.is_empty());
        let folder_name = if dir == root { root_name.clone() } else { dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| root_name.clone()) };
        let name = single_name.unwrap_or(folder_name);
        let sets = build_sets(classified, unknown, &name);
        let kind = kind_without_json(!variants.is_empty(), &sets, &name);
        let mut asset_warnings = Vec::new();
        if !ignored.is_empty() {
            asset_warnings.push(format!("{} model file(s) in other formats ignored", ignored.len()));
        }
        assets.push(PackAsset { id: name.clone(), name, kind, megascans: false, dir, variants, sets, size_m: None, warnings: asset_warnings, json: None, include: true });
    }
    let _ = warnings;
    assets
}

/// Scan `root` (a folder, or a zip already extracted) for importable assets.
pub fn scan_pack(root: &Path) -> Result<PackScan> {
    if !root.is_dir() {
        bail!("{} is not a folder", root.display());
    }
    let mut found = Found::default();
    let mut count = 0usize;
    walk(root, 0, &mut found, &mut count)?;
    let mut scan = PackScan { root: root.to_owned(), ..Default::default() };
    if !found.foreign.is_empty() {
        let mut exts: Vec<String> = found.foreign.iter().filter_map(|p| p.extension().map(|e| e.to_string_lossy().to_ascii_lowercase())).collect();
        exts.sort();
        exts.dedup();
        scan.warnings.push(format!("{} file(s) in formats the engine does not read ({}): export FBX or glTF from the source application", found.foreign.len(), exts.join(", ")));
    }
    if !found.unreadable_images.is_empty() {
        scan.warnings.push(format!("{} texture(s) in formats the engine does not read (tif, psd, dds, ...): convert them to png / jpg", found.unreadable_images.len()));
    }
    let foreign = std::mem::take(&mut found.foreign);
    let mut assets = megascans_assets(&mut found, root, &mut scan.warnings);
    let leftover = !found.models.is_empty() || !found.images.is_empty();
    if leftover {
        assets.extend(generic_assets(found, root, &mut scan.warnings));
    }
    // Unique names (duplicate asset handling): the same name twice gets a numeric suffix.
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    for a in &mut assets {
        let n = seen.entry(a.name.to_ascii_lowercase()).or_insert(0);
        *n += 1;
        if *n > 1 {
            a.name = format!("{}_{}", a.name, *n);
        }
    }
    if assets.is_empty() && !foreign.is_empty() {
        scan.warnings.push("no model or texture the engine can read was found".into());
    }
    scan.assets = assets;
    Ok(scan)
}

/// What a zip archive looks like, from its central directory alone.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArchiveKind {
    /// Models and / or textures: an asset pack.
    Pack,
    /// A MetaHuman package (`Manifest.json` + `.uasset` files).
    MetaHuman,
    Unknown,
}

pub fn peek_archive(path: &Path) -> ArchiveKind {
    let Ok(list) = crate::pack_zip::list_zip(path, &crate::pack_zip::ZipLimits::default()) else { return ArchiveKind::Unknown };
    let ext = |n: &str| n.rsplit_once('.').map(|x| x.1.to_ascii_lowercase()).unwrap_or_default();
    let names: Vec<&str> = list.entries.iter().filter(|e| !e.is_dir).map(|e| e.name.as_str()).collect();
    let has_uasset = names.iter().any(|n| ext(n) == "uasset");
    let has_manifest = names.iter().any(|n| n.eq_ignore_ascii_case("manifest.json"));
    if has_uasset && has_manifest || path.extension().is_some_and(|e| e.eq_ignore_ascii_case("mhpkg")) {
        return ArchiveKind::MetaHuman;
    }
    if names.iter().any(|n| MODEL_EXTS.contains(&ext(n).as_str()) || (is_image_ext(&ext(n)) && classify(n).is_some())) {
        return ArchiveKind::Pack;
    }
    ArchiveKind::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack_zip::tests::{make_zip, scratch};

    fn touch(root: &Path, rel: &str) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, b"x").unwrap();
    }

    #[test]
    fn model_stems_split_into_variant_and_lod() {
        assert_eq!(split_model_stem("uddmcgbia_VarA_LOD0"), ("uddmcgbia".into(), "VarA".into(), Some(0)));
        assert_eq!(split_model_stem("xb5ebhf"), ("xb5ebhf".into(), "".into(), None));
        assert_eq!(split_model_stem("rock_LOD12"), ("rock".into(), "".into(), Some(12)));
        assert_eq!(split_model_stem("Stylized_Paladin"), ("Stylized_Paladin".into(), "".into(), None));
    }

    #[test]
    fn lod_chain_is_numerically_ordered_and_formats_deduped() {
        let files: Vec<(PathBuf, u64)> = ["a_LOD10.fbx", "a_LOD2.fbx", "a_LOD0.fbx", "a_LOD0.gltf", "a_LOD1.glb"].iter().map(|n| (PathBuf::from(n), 1)).collect();
        let (v, ignored) = group_models(&files);
        assert_eq!(v.len(), 1);
        let order: Vec<u32> = v[0].lods.iter().map(|l| l.lod).collect();
        assert_eq!(order, vec![0, 1, 2, 10]);
        assert_eq!(v[0].lods[0].path, PathBuf::from("a_LOD0.fbx"), "fbx beats gltf");
        assert_eq!(ignored, vec![PathBuf::from("a_LOD0.gltf")]);
    }

    #[test]
    fn megascans_folder_without_json_and_with_variants() {
        let dir = scratch("ms");
        for n in ["xb5ebhf/xb5ebhf.fbx", "xb5ebhf/xb5ebhf_8K_BaseColor.jpg", "xb5ebhf/xb5ebhf_8K_Normal.jpg", "xb5ebhf/xb5ebhf_8K_Roughness.jpg", "xb5ebhf/xb5ebhf_8K_Gloss.jpg", "xb5ebhf/xb5ebhf_8K_AO.jpg"] {
            touch(&dir, n);
        }
        for n in ["uddmcgbia/uddmcgbia_VarA_LOD0.fbx", "uddmcgbia/uddmcgbia_VarB_LOD0.fbx", "uddmcgbia/uddmcgbia_VarB_LOD1.fbx", "uddmcgbia/uddmcgbia_4K_Basecolor.jpg", "uddmcgbia/uddmcgbia_4K_Opacity.jpg", "uddmcgbia/uddmcgbia_4K_Translucency.jpg", "uddmcgbia/uddmcgbia_8K_Billboard_Basecolor.jpg"] {
            touch(&dir, n);
        }
        for n in ["udlladln/udlladln_8K_Albedo.jpg", "udlladln/udlladln_8K_Normal.jpg", "udlladln/udlladln_8K_Displacement.jpg"] {
            touch(&dir, n);
        }
        let scan = scan_pack(&dir).unwrap();
        assert_eq!(scan.assets.len(), 3, "{:#?}", scan.assets.iter().map(|a| (&a.id, a.kind)).collect::<Vec<_>>());
        let by = |id: &str| scan.assets.iter().find(|a| a.id == id).unwrap();
        let rock = by("xb5ebhf");
        assert!(rock.megascans);
        assert_eq!(rock.kind, AssetKind::Model3d);
        assert_eq!(rock.variants.len(), 1);
        assert_eq!(rock.sets.len(), 1);
        assert_eq!(rock.texture_count(), 5);
        let plant = by("uddmcgbia");
        assert_eq!(plant.kind, AssetKind::Plant, "opacity + translucency + billboard");
        assert_eq!(plant.variants.iter().map(|v| (v.name.as_str(), v.lods.len())).collect::<Vec<_>>(), vec![("VarA", 1), ("VarB", 2)]);
        assert_eq!(plant.sets.iter().filter(|s| s.billboard).count(), 1);
        assert_eq!(by("udlladln").kind, AssetKind::Surface);
        assert!(by("udlladln").variants.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn megascans_json_sets_type_name_and_size() {
        let dir = scratch("msjson");
        touch(&dir, "abc12345/abc12345_4K_Albedo.jpg");
        touch(&dir, "abc12345/abc12345_4K_Normal.jpg");
        std::fs::write(dir.join("abc12345/abc12345.json"), r#"{"id":"abc12345","semanticTags":{"asset_type":"decal","name":"Crack"},"meta":[{"key":"scanArea","value":"0.5x0.5"}]}"#).unwrap();
        let scan = scan_pack(&dir).unwrap();
        let a = &scan.assets[0];
        assert_eq!((a.kind, a.name.as_str(), a.size_m), (AssetKind::Decal, "Crack", Some([0.5, 0.5])));
        assert!(a.json.is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn generic_fab_layout_with_textures_folder_and_materials_by_prefix() {
        let dir = scratch("generic");
        for n in ["Stylized_Paladin.fbx", "Textures/Armor_Base_color.png", "Textures/Armor_Normal_OpenGL.png", "Textures/Armor_Metallic.png", "Textures/Armor_Roughness.png", "Textures/Body_Base_color.png", "Textures/Body_Normal_OpenGL.png", "Textures/notes.png", "readme.txt"] {
            touch(&dir, n);
        }
        let scan = scan_pack(&dir).unwrap();
        assert_eq!(scan.assets.len(), 1);
        let a = &scan.assets[0];
        assert!(!a.megascans);
        assert_eq!(a.name, "Stylized_Paladin");
        assert_eq!(a.kind, AssetKind::Model3d);
        let names: Vec<_> = a.sets.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, vec!["Armor", "Body", "Unsorted"]);
        let unsorted = a.sets.iter().find(|s| s.name == "Unsorted").unwrap();
        assert_eq!(unsorted.textures[0].role, TexRole::Ignore);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn foreign_formats_are_reported_and_duplicates_renamed() {
        let dir = scratch("foreign");
        for n in ["a/Rock.fbx", "a/Rock_Albedo.png", "b/Rock.fbx", "b/Rock_Albedo.png", "c/scene.blend", "c/Wall.tif"] {
            touch(&dir, n);
        }
        let scan = scan_pack(&dir).unwrap();
        assert_eq!(scan.assets.len(), 2);
        let names: Vec<_> = scan.assets.iter().map(|a| a.name.clone()).collect();
        assert_eq!(names, vec!["Rock", "Rock_2"]);
        assert_eq!(scan.warnings.len(), 2, "{:?}", scan.warnings);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn archives_are_peeked_without_extracting() {
        let dir = scratch("peek");
        make_zip(&dir.join("pack.zip"), &[("m.fbx", b"x", false), ("m_Albedo.png", b"x", false)]);
        make_zip(&dir.join("mh.mhpkg"), &[("Manifest.json", b"{}", false), ("a.uasset", b"x", false)]);
        make_zip(&dir.join("other.zip"), &[("readme.txt", b"x", false)]);
        assert_eq!(peek_archive(&dir.join("pack.zip")), ArchiveKind::Pack);
        assert_eq!(peek_archive(&dir.join("mh.mhpkg")), ArchiveKind::MetaHuman);
        assert_eq!(peek_archive(&dir.join("other.zip")), ArchiveKind::Unknown);
        assert_eq!(peek_archive(&dir.join("missing.zip")), ArchiveKind::Unknown);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The author's real Fab downloads, when present on this machine.
    #[test]
    fn real_fab_megascans_export_scans() {
        let root = Path::new("/home/lemonxah/Downloads/fab/south_african_slate_quarry_raw");
        if !root.is_dir() {
            return;
        }
        let scan = scan_pack(root).unwrap();
        assert!(scan.assets.len() >= 80, "{}", scan.assets.len());
        assert!(scan.assets.iter().all(|a| a.megascans));
        let plant = scan.assets.iter().find(|a| a.id == "uddmcgbia").unwrap();
        assert_eq!(plant.kind, AssetKind::Plant);
        assert_eq!(plant.variants.len(), 11);
        assert!(scan.assets.iter().any(|a| a.kind == AssetKind::Surface));
        assert!(scan.assets.iter().any(|a| a.kind == AssetKind::Model3d));
    }
}
