//! The model importer's material remap table: "source material name -> a
//! `.material` asset", plus the three ways to fill it (Search and Remap,
//! Extract Textures, Extract Materials). The table lives in the model's `.meta`
//! ([`ModelImport::material_remap`] / [`ModelImport::texture_remap`]) and is
//! applied after the import, so editing it never invalidates the import cache.
//!
//! Every function here edits a `ModelImport` the caller owns (the inspector's
//! edited copy), so a fill is an ordinary settings edit that Apply or Revert
//! handles like any other. Files written by an extract land on disk
//! immediately, as in Unity.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};

use crate::{MaterialFile, MaterialNaming, MaterialSearch, MaterialTextures, ModelImport, Scene, TEXTURE_REMAP_SLOTS, save_material_file, texture_remap_key};

/// Project-relative path with forward slashes, the form the tables store.
pub fn project_relative(project_root: &Path, path: &Path) -> String {
    let rel = path.strip_prefix(project_root).unwrap_or(path);
    rel.to_string_lossy().replace('\\', "/")
}

/// A name made safe to use as a file stem.
pub fn file_stem_of(name: &str, fallback: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ' ') { c } else { '_' }).collect();
    let s = s.trim_matches(|c| c == '_' || c == '.' || c == ' ');
    if s.is_empty() { fallback.to_owned() } else { s.to_owned() }
}

/// `dir/stem.ext`, or `dir/stem_1.ext`, `stem_2.ext`, ... if taken.
pub fn unique_file(dir: &Path, stem: &str, ext: &str) -> PathBuf {
    let mut p = dir.join(format!("{stem}.{ext}"));
    let mut n = 0;
    while p.exists() {
        n += 1;
        p = dir.join(format!("{stem}_{n}.{ext}"));
    }
    p
}

/// The file name `Search and Remap` looks for.
pub fn expected_name(naming: MaterialNaming, model: &Path, material: &str) -> String {
    let stem = file_stem_of(material, "material");
    match naming {
        // The texture rule has no file name to look for; the name rules' fallback.
        MaterialNaming::BasedOnMaterialName | MaterialNaming::BasedOnTextureName => format!("{stem}.material"),
        MaterialNaming::BasedOnModelNameAndMaterialName => {
            let m = model.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            format!("{}-{stem}.material", file_stem_of(&m, "model"))
        }
    }
}

/// `.material` files directly in `dir` (and in its `Materials` subfolder), by lowercase name.
fn materials_in(dir: &Path, out: &mut HashMap<String, PathBuf>) {
    let mut scan = |d: &Path| {
        if let Ok(rd) = std::fs::read_dir(d) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("material"))
                    && let Some(n) = p.file_name()
                {
                    out.entry(n.to_string_lossy().to_lowercase()).or_insert(p);
                }
            }
        }
    };
    scan(dir);
    if let Ok(rd) = std::fs::read_dir(dir) {
        let mut subs: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.is_dir() && p.file_name().is_some_and(|n| n.to_string_lossy().eq_ignore_ascii_case("materials"))).collect();
        subs.sort();
        for s in subs {
            scan(&s);
        }
    }
}

fn walk_materials(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = rd.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let name = e.file_name().to_string_lossy().into_owned();
        let p = e.path();
        if p.is_dir() {
            if !name.starts_with('.') && name != "target" && name != "node_modules" && depth < 24 {
                walk_materials(&p, depth + 1, out);
            }
        } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("material")) {
            out.push(p);
        }
    }
}

/// Find a `.material` for each of `sources`, by the naming rule within the
/// search scope. `Local` looks in the model's folder (and its `Materials`
/// folder), `RecursiveUp` then in each parent up to the project root, and
/// `Everywhere` anywhere in the project, nearest folder first. Returns
/// `(source name, project-relative path)` for the ones found.
pub fn search_and_remap(project_root: &Path, model: &Path, sources: &[String], naming: MaterialNaming, search: MaterialSearch) -> Vec<(String, String)> {
    let model_dir = model.parent().unwrap_or(project_root);
    // Candidate folders in priority order.
    let mut layers: Vec<HashMap<String, PathBuf>> = Vec::new();
    match search {
        MaterialSearch::Local => {
            let mut l = HashMap::new();
            materials_in(model_dir, &mut l);
            layers.push(l);
        }
        MaterialSearch::RecursiveUp => {
            let mut dir = Some(model_dir);
            while let Some(d) = dir {
                let mut l = HashMap::new();
                materials_in(d, &mut l);
                layers.push(l);
                if d == project_root || !d.starts_with(project_root) {
                    break;
                }
                dir = d.parent();
            }
        }
        MaterialSearch::Everywhere => {
            let mut all = Vec::new();
            walk_materials(project_root, 0, &mut all);
            // Nearest to the model first: fewest path components apart.
            let dist = |p: &Path| -> usize {
                let a: Vec<_> = model_dir.components().collect();
                let b: Vec<_> = p.parent().unwrap_or(project_root).components().collect();
                let common = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
                (a.len() - common) + (b.len() - common)
            };
            all.sort_by_key(|p| dist(p));
            let mut l = HashMap::new();
            for p in all {
                if let Some(n) = p.file_name() {
                    l.entry(n.to_string_lossy().to_lowercase()).or_insert(p);
                }
            }
            layers.push(l);
        }
    }
    sources
        .iter()
        .filter_map(|name| {
            let want = expected_name(naming, model, name).to_lowercase();
            layers.iter().find_map(|l| l.get(&want)).map(|p| (name.clone(), project_relative(project_root, p)))
        })
        .collect()
}

/// A model material as the texture-name rule sees it: its name and the
/// (slot, file stem) of each texture it uses, in slot order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SourceMaterial {
    pub name: String,
    pub textures: Vec<(String, String)>,
}

/// One Search and Remap hit, with the rule that made it (shown in the table row).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RemapMatch {
    pub source: String,
    /// Project-relative `.material`.
    pub path: String,
    pub reason: String,
}

/// The file stem of a texture name or path (either slash, extension dropped).
pub fn texture_stem(name: &str) -> String {
    let file = name.rsplit(['/', '\\']).next().unwrap_or(name);
    match file.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() && ext.len() <= 5 && ext.chars().all(|c| c.is_ascii_alphanumeric()) => stem.to_owned(),
        _ => file.to_owned(),
    }
}

/// Suffixes that name a texture's role rather than its subject.
const ROLE_SUFFIXES: &[&str] = &[
    "albedo", "basecolor", "base_color", "diffuse", "color", "colour", "col", "diff", "normal", "normalmap", "norm", "nrm", "nor", "n", "orm", "rma", "arm", "roughness", "rough", "metallic",
    "metalness", "metal", "ao", "occlusion", "emission", "emissive", "emit", "opacity", "alpha", "height", "disp", "displacement", "mask", "d", "c", "a",
];

/// A texture stem without its role suffix and separators, lowercased:
/// `Wood_Albedo` and `wood-normal` both give `wood`. A stem that is only a
/// role (`normal`) is returned whole.
pub fn texture_base_name(stem: &str) -> String {
    let lower = stem.to_lowercase();
    let mut base = lower.as_str();
    // Strip role suffixes repeatedly (`wood_albedo_4k` keeps `wood_albedo_4k`; `wood_4k_albedo` -> `wood_4k`).
    loop {
        let trimmed = base.trim_end_matches(['_', '-', '.', ' ']);
        let Some(cut) = trimmed.rfind(['_', '-', '.', ' ']) else { break };
        if ROLE_SUFFIXES.contains(&&trimmed[cut + 1..]) {
            base = &trimmed[..cut];
        } else {
            break;
        }
    }
    let base = base.trim_matches(['_', '-', '.', ' ']);
    if base.is_empty() { lower } else { base.to_owned() }
}

/// Shortest shared prefix (in characters) the texture-name rule accepts.
pub const MIN_TEXTURE_PREFIX: usize = 4;

fn common_prefix_len(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

/// How well a candidate material's textures match a source material's.
/// Compared lexicographically; `None` = no match. Fields, in priority order:
/// slots whose texture stems are equal, source textures whose base names are
/// found among the candidate's, and the longest shared base-name prefix.
fn texture_score(source: &SourceMaterial, cand: &[(String, String)]) -> Option<(usize, usize, usize)> {
    let low = |s: &str| s.to_lowercase();
    let exact = source.textures.iter().filter(|(slot, stem)| cand.iter().any(|(cs, cst)| cs == slot && low(cst) == low(stem))).count();
    let cand_bases: Vec<String> = cand.iter().map(|(_, s)| texture_base_name(s)).collect();
    let src_bases: Vec<String> = source.textures.iter().map(|(_, s)| texture_base_name(s)).collect();
    let base_equal = src_bases.iter().filter(|b| cand_bases.contains(b)).count();
    let prefix = src_bases.iter().flat_map(|a| cand_bases.iter().map(move |b| common_prefix_len(a, b))).max().unwrap_or(0);
    (exact > 0 || base_equal > 0 || prefix >= MIN_TEXTURE_PREFIX).then_some((exact, base_equal, prefix))
}

fn texture_reason(score: (usize, usize, usize), source: &SourceMaterial, cand: &[(String, String)]) -> String {
    let (exact, base_equal, _) = score;
    if exact > 0 {
        let (slot, stem) = source.textures.iter().find(|(slot, stem)| cand.iter().any(|(cs, cst)| cs == slot && cst.eq_ignore_ascii_case(stem))).expect("exact > 0");
        return format!("same texture '{stem}' in the {slot} slot{}", if exact > 1 { format!(" (+{} more)", exact - 1) } else { String::new() });
    }
    let cand_bases: Vec<String> = cand.iter().map(|(_, s)| texture_base_name(s)).collect();
    if base_equal > 0 {
        let b = source.textures.iter().map(|(_, s)| texture_base_name(s)).find(|b| cand_bases.contains(b)).expect("base_equal > 0");
        return format!("textures share the name '{b}'");
    }
    let src_bases: Vec<String> = source.textures.iter().map(|(_, s)| texture_base_name(s)).collect();
    let best = src_bases.iter().flat_map(|a| cand_bases.iter().map(move |b| (common_prefix_len(a, b), a))).max_by_key(|(n, _)| *n).map(|(n, a)| a.chars().take(n).collect::<String>());
    format!("textures share the prefix '{}'", best.unwrap_or_default())
}

/// `.material` files in `dir` then in its `Materials` subfolder, sorted by name.
fn materials_listing(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut scan = |d: &Path| {
        let mut here: Vec<PathBuf> = std::fs::read_dir(d).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("material"))).collect();
        here.sort();
        out.extend(here);
    };
    scan(dir);
    let mut subs: Vec<PathBuf> = std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.is_dir() && p.file_name().is_some_and(|n| n.to_string_lossy().eq_ignore_ascii_case("materials"))).collect();
    subs.sort();
    for s in subs {
        scan(&s);
    }
    out
}

/// Every `.material` in the search scope with its closeness rank (0 = the
/// model's own folder; smaller is nearer).
fn scope_materials(project_root: &Path, model: &Path, search: MaterialSearch) -> Vec<(usize, PathBuf)> {
    let model_dir = model.parent().unwrap_or(project_root);
    match search {
        MaterialSearch::Local => materials_listing(model_dir).into_iter().map(|p| (0, p)).collect(),
        MaterialSearch::RecursiveUp => {
            let mut out = Vec::new();
            let mut dir = Some(model_dir);
            let mut rank = 0;
            while let Some(d) = dir {
                out.extend(materials_listing(d).into_iter().map(|p| (rank, p)));
                rank += 1;
                if d == project_root || !d.starts_with(project_root) {
                    break;
                }
                dir = d.parent();
            }
            out
        }
        MaterialSearch::Everywhere => {
            let mut all = Vec::new();
            walk_materials(project_root, 0, &mut all);
            let a: Vec<_> = model_dir.components().collect();
            all.into_iter()
                .map(|p| {
                    let b: Vec<_> = p.parent().unwrap_or(project_root).components().collect();
                    let common = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
                    ((a.len() - common) + (b.len() - common), p)
                })
                .collect()
        }
    }
}

/// The (slot, stem) list of a loaded `.material`, in the importer's slot order.
fn material_texture_stems(path: &Path) -> Vec<(String, String)> {
    let Ok(f) = crate::load_material_file(path) else { return Vec::new() };
    let t = &f.textures;
    TEXTURE_REMAP_SLOTS.iter().zip([&t.albedo, &t.normal, &t.orm, &t.emission]).filter_map(|(slot, p)| p.as_ref().map(|p| ((*slot).to_owned(), texture_stem(&p.to_string_lossy())))).collect()
}

/// One candidate material: (rank, project-relative path, texture stems).
type MaterialCandidate = (usize, String, Vec<(String, String)>);

/// Search and Remap by texture names. For each source material that uses
/// textures, every `.material` in the scope is scored by [`texture_score`];
/// the best score wins. Ties go to the nearer folder (the model's own folder,
/// then each parent / the fewest path steps away), then to the
/// lexicographically smallest project-relative path, so the answer never
/// depends on directory order. Materials without textures never match.
pub fn search_by_texture_names(project_root: &Path, model: &Path, sources: &[SourceMaterial], search: MaterialSearch) -> Vec<RemapMatch> {
    let wanted: Vec<&SourceMaterial> = sources.iter().filter(|s| !s.textures.is_empty()).collect();
    if wanted.is_empty() {
        return Vec::new();
    }
    let cands: Vec<MaterialCandidate> = scope_materials(project_root, model, search)
        .into_iter()
        .map(|(rank, p)| (rank, project_relative(project_root, &p), material_texture_stems(&p)))
        .filter(|(_, _, stems)| !stems.is_empty())
        .collect();
    wanted
        .into_iter()
        .filter_map(|src| {
            // Best = highest score, then lowest rank, then smallest path.
            let (score, _, path, stems) = cands
                .iter()
                .filter_map(|(rank, path, stems)| texture_score(src, stems).map(|sc| (sc, *rank, path, stems)))
                .min_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then_with(|| a.2.to_lowercase().cmp(&b.2.to_lowercase())).then_with(|| a.2.cmp(b.2)))?;
            Some(RemapMatch { source: src.name.clone(), path: path.clone(), reason: texture_reason(score, src, stems) })
        })
        .collect()
}

/// Search and Remap by any of the three naming rules, with the reason for each hit.
pub fn search_matches(project_root: &Path, model: &Path, sources: &[SourceMaterial], naming: MaterialNaming, search: MaterialSearch) -> Vec<RemapMatch> {
    if naming == MaterialNaming::BasedOnTextureName {
        return search_by_texture_names(project_root, model, sources, search);
    }
    let names: Vec<String> = sources.iter().map(|s| s.name.clone()).collect();
    search_and_remap(project_root, model, &names, naming, search)
        .into_iter()
        .map(|(source, path)| {
            let file = Path::new(&path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            RemapMatch { source, reason: format!("named {file}"), path }
        })
        .collect()
}

/// The texture-name rule's view of a scene's materials: each material's name
/// and the source file stem of every texture slot it uses (names come from
/// the importer's `texture_names`; slots without a recorded name are skipped).
pub fn source_materials_of(scene: &Scene) -> Vec<SourceMaterial> {
    (0..scene.materials.len())
        .map(|mi| SourceMaterial {
            name: key_name(scene, mi),
            textures: TEXTURE_REMAP_SLOTS
                .iter()
                .enumerate()
                .filter_map(|(si, slot)| {
                    let ti = slot_texture(&scene.materials[mi], si)?;
                    let stem = scene.identity.texture_names.get(ti).filter(|n| !n.is_empty())?;
                    Some(((*slot).to_owned(), stem.clone()))
                })
                .collect(),
        })
        .collect()
}

/// A project-relative folder setting resolved to an absolute path, or why not.
/// `setting` empty = `<model dir>/<default_name>`. The folder may not exist
/// yet. Rejected: absolute paths, `..`, and (when part of it exists) a path
/// that leaves the project through a symlink.
pub fn resolve_extract_dir(project_root: &Path, model: &Path, setting: &str, default_name: &str) -> Result<PathBuf> {
    let setting = setting.trim().replace('\\', "/");
    if setting.is_empty() {
        return Ok(model.parent().unwrap_or(project_root).join(default_name));
    }
    let rel = Path::new(&setting);
    if rel.is_absolute() || setting.starts_with('/') || setting.contains(':') {
        anyhow::bail!("'{setting}' is not a project-relative folder");
    }
    if rel.components().any(|c| !matches!(c, std::path::Component::Normal(_) | std::path::Component::CurDir)) {
        anyhow::bail!("'{setting}' leaves the project");
    }
    let full = project_root.join(rel);
    // The deepest existing ancestor must still resolve inside the project.
    if let (Ok(root), Some(real)) = (project_root.canonicalize(), full.ancestors().find_map(|a| a.canonicalize().ok()))
        && !real.starts_with(&root)
    {
        anyhow::bail!("'{setting}' leaves the project");
    }
    Ok(full)
}

/// The sub-folder names of project-relative `rel` (`""` = the root), sorted,
/// for the folder chooser. Hidden folders and build output are left out; a
/// `rel` that leaves the project lists nothing.
pub fn list_subfolders(project_root: &Path, rel: &str) -> Vec<String> {
    let dir = if rel.trim().is_empty() {
        project_root.to_path_buf()
    } else {
        let Ok(d) = resolve_extract_dir(project_root, project_root, rel, "") else { return Vec::new() };
        d
    };
    let mut out: Vec<String> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| !n.starts_with('.') && n != "target" && n != "node_modules")
        .collect();
    out.sort_by_key(|n| n.to_lowercase());
    out
}

/// What an extract wrote.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExtractReport {
    pub textures: usize,
    pub materials: usize,
    /// `.meta` files written so the new images import with the kind their slot expects.
    pub metas: usize,
    /// Entries skipped because the table already had them.
    pub already_mapped: usize,
}

fn slot_texture(m: &crate::SceneMaterial, slot: usize) -> Option<usize> {
    [m.albedo, m.normal, m.orm, m.emission][slot]
}

/// The name a material is keyed by in the tables.
fn key_name(scene: &Scene, i: usize) -> String {
    scene.identity.material_names.get(i).cloned().unwrap_or_else(|| scene.materials[i].name.clone())
}

/// Write the model's embedded textures as PNGs into `dir` and add them to
/// `model.texture_remap`, skipping slots already in the table. One image per
/// embedded texture, even when several slots share it (the first slot that
/// uses it decides its `.meta`; an existing `.meta` is never overwritten).
pub fn extract_textures(scene: &Scene, dir: &Path, project_root: &Path, model: &mut ModelImport) -> Result<ExtractReport> {
    let mut report = ExtractReport::default();
    let mut written: HashMap<usize, String> = HashMap::new();
    for (mi, m) in scene.materials.iter().enumerate() {
        let name = key_name(scene, mi);
        for (si, slot) in TEXTURE_REMAP_SLOTS.iter().enumerate() {
            let Some(ti) = slot_texture(m, si) else { continue };
            let key = texture_remap_key(&name, slot);
            if model.texture_remap.contains_key(&key) {
                report.already_mapped += 1;
                continue;
            }
            let rel = match written.get(&ti) {
                Some(rel) => rel.clone(),
                None => {
                    let t = &scene.textures[ti];
                    if t.hdr {
                        tracing::warn!("extract: skipping HDR texture of '{name}' ({slot})");
                        continue;
                    }
                    let img = image::RgbaImage::from_raw(t.width, t.height, t.pixels.clone()).with_context(|| format!("texture {ti} has a bad pixel buffer"))?;
                    std::fs::create_dir_all(dir)?;
                    let path = unique_file(dir, &format!("{}_{slot}", file_stem_of(&name, "material")), "png");
                    img.save(&path).with_context(|| format!("writing {}", path.display()))?;
                    report.textures += 1;
                    // The slot decides the import kind (normal map, linear data, sRGB colour);
                    // slots 0..4 here are the binding slots `slot_expectation` numbers.
                    match crate::ensure_first_import_for_slot(&path, si) {
                        Ok(true) => report.metas += 1,
                        Ok(false) => {}
                        Err(e) => tracing::warn!("extract: writing the .meta of {}: {e:#}", path.display()),
                    }
                    let rel = project_relative(project_root, &path);
                    written.insert(ti, rel.clone());
                    rel
                }
            };
            model.texture_remap.insert(key, rel);
        }
    }
    Ok(report)
}

/// Write each not-yet-remapped material as a `.material` into `mat_dir` and add
/// it to `model.material_remap`. Textures a material needs are extracted into
/// `tex_dir` first when the texture table lacks them, so the file references
/// real images.
pub fn extract_materials(scene: &Scene, mat_dir: &Path, tex_dir: &Path, project_root: &Path, model: &mut ModelImport) -> Result<ExtractReport> {
    let mut report = extract_textures(scene, tex_dir, project_root, model)?;
    report.already_mapped = 0;
    for (mi, m) in scene.materials.iter().enumerate() {
        let name = key_name(scene, mi);
        if model.material_remap.contains_key(&name) {
            report.already_mapped += 1;
            continue;
        }
        let tex = |slot: &str| model.texture_remap.get(&texture_remap_key(&name, slot)).map(PathBuf::from);
        let file = MaterialFile {
            name: m.name.clone(),
            shader: "standard".into(),
            params: m.params,
            features: m.features,
            render_queue: None,
            textures: MaterialTextures { albedo: tex("albedo"), normal: tex("normal"), orm: tex("orm"), emission: tex("emission"), ..Default::default() },
            custom: Default::default(),
        };
        std::fs::create_dir_all(mat_dir)?;
        let path = unique_file(mat_dir, &file_stem_of(&name, "material"), "material");
        save_material_file(&path, &file)?;
        model.material_remap.insert(name, project_relative(project_root, &path));
        report.materials += 1;
    }
    Ok(report)
}

/// Why `path` cannot be a remap target, if it cannot.
pub fn reject_target(model: &Path, target: &Path) -> Option<&'static str> {
    if target == model {
        return Some("a model cannot be remapped to itself");
    }
    if !target.extension().is_some_and(|e| e.eq_ignore_ascii_case("material")) {
        return Some("pick a .material file");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AssetMeta, ImporterSettings, SceneMaterial, load_asset_meta, save_asset_meta};
    use zestengine_render::{MaterialFeatures, MaterialParams, TextureData};

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("zest_remap_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn scene_with(materials: &[(&str, [Option<usize>; 4])], textures: usize) -> Scene {
        let mut s = Scene {
            meshes: Vec::new(),
            morphs: Vec::new(),
            springs: Default::default(),
            textures: (0..textures).map(|i| TextureData { width: 2, height: 2, pixels: vec![i as u8 * 40; 16], srgb: true, hdr: false }).collect(),
            materials: materials
                .iter()
                .map(|(n, t)| SceneMaterial { name: (*n).into(), params: MaterialParams::default(), features: MaterialFeatures::default(), albedo: t[0], normal: t[1], orm: t[2], emission: t[3] })
                .collect(),
            instances: Vec::new(),
            skeletons: Vec::new(),
            animations: Vec::new(),
            collision: Vec::new(),
            identity: Default::default(),
        };
        s.identity.material_names = materials.iter().map(|m| m.0.to_owned()).collect();
        s
    }

    #[test]
    fn the_remap_table_round_trips_through_the_meta() {
        let dir = tmp("meta");
        let asset = dir.join("hero.fbx");
        let mut meta = AssetMeta::new_for(&asset);
        let m = meta.importer.as_model_mut().unwrap();
        m.material_remap.insert("Skin".into(), "mats/skin.material".into());
        m.texture_remap.insert(texture_remap_key("Skin", "albedo"), "tex/skin.png".into());
        m.source_materials = vec!["Skin".into(), "Hair".into()];
        save_asset_meta(&asset, &meta).unwrap();
        let back = load_asset_meta(&asset).unwrap().unwrap();
        let m = back.importer.as_model().unwrap();
        assert_eq!(m.material_remap.get("Skin").map(String::as_str), Some("mats/skin.material"));
        assert_eq!(m.texture_remap.get("Skin/albedo").map(String::as_str), Some("tex/skin.png"));
        assert_eq!(m.source_materials, ["Skin", "Hair"]);
        assert!(matches!(back.importer, ImporterSettings::Model(_)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_old_meta_without_the_tables_loads_empty() {
        let dir = tmp("old");
        let asset = dir.join("m.fbx");
        std::fs::write(crate::meta_path(&asset), "(id:\"00000000-0000-4000-8000-000000000001\",importer:Model((scale:1.0)))").unwrap();
        let m = load_asset_meta(&asset).unwrap().unwrap();
        let m = m.importer.as_model().unwrap();
        assert!(m.material_remap.is_empty() && m.texture_remap.is_empty() && m.source_materials.is_empty());
        assert!(m.weld_vertices && m.optimize_mesh && m.import_blend_shapes, "new settings default on");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extracting_textures_fills_the_table_and_a_second_run_adds_nothing() {
        let dir = tmp("tex");
        let scene = scene_with(&[("Skin", [Some(0), Some(1), None, None]), ("Hair", [Some(0), None, None, None])], 2);
        let mut model = ModelImport::default();
        let r = extract_textures(&scene, &dir.join("textures"), &dir, &mut model).unwrap();
        assert_eq!(r.textures, 2, "texture 0 is shared by two slots and written once");
        assert_eq!(model.texture_remap.len(), 3);
        assert_eq!(model.texture_remap["Skin/albedo"], model.texture_remap["Hair/albedo"]);
        assert!(dir.join(&model.texture_remap["Skin/normal"]).is_file());
        let again = extract_textures(&scene, &dir.join("textures"), &dir, &mut model).unwrap();
        assert_eq!((again.textures, again.already_mapped), (0, 3));
        let pngs = std::fs::read_dir(dir.join("textures")).unwrap().flatten().filter(|e| e.path().extension().is_some_and(|x| x == "png")).count();
        assert_eq!(pngs, 2, "no duplicates on disk");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extracting_materials_fills_the_table_with_loadable_files_and_does_not_duplicate() {
        let dir = tmp("mat");
        let scene = scene_with(&[("Skin", [Some(0), None, None, None]), ("Hair", [None; 4])], 1);
        let mut model = ModelImport::default();
        let r = extract_materials(&scene, &dir.join("materials"), &dir.join("textures"), &dir, &mut model).unwrap();
        assert_eq!((r.materials, r.textures), (2, 1));
        let skin = crate::load_material_file(dir.join(&model.material_remap["Skin"])).unwrap();
        assert_eq!(skin.textures.albedo.as_deref(), Some(Path::new(model.texture_remap["Skin/albedo"].as_str())), "the file references the extracted image");
        let again = extract_materials(&scene, &dir.join("materials"), &dir.join("textures"), &dir, &mut model).unwrap();
        assert_eq!((again.materials, again.textures, again.already_mapped), (0, 0, 2));
        assert_eq!(std::fs::read_dir(dir.join("materials")).unwrap().count(), 2);
        // A cleared row is extracted again, as a new file rather than over the old one.
        model.material_remap.remove("Hair");
        let third = extract_materials(&scene, &dir.join("materials"), &dir.join("textures"), &dir, &mut model).unwrap();
        assert_eq!(third.materials, 1);
        assert_eq!(std::fs::read_dir(dir.join("materials")).unwrap().count(), 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn search_and_remap_honours_naming_and_scope() {
        let dir = tmp("search");
        let (models, far) = (dir.join("assets/models"), dir.join("other/deep"));
        std::fs::create_dir_all(models.join("Materials")).unwrap();
        std::fs::create_dir_all(&far).unwrap();
        let write = |p: PathBuf| save_material_file(p, &MaterialFile { name: "x".into(), shader: "standard".into(), params: MaterialParams::default(), features: MaterialFeatures::default(), render_queue: None, textures: Default::default(), custom: Default::default() }).unwrap();
        write(models.join("Materials/Skin.material"));
        write(dir.join("assets/Hair.material"));
        write(far.join("Cloth.material"));
        write(models.join("hero-Eyes.material"));
        let model = models.join("hero.fbx");
        let names: Vec<String> = ["Skin", "Hair", "Cloth", "Eyes"].map(String::from).to_vec();
        let found = |naming, search| -> Vec<(String, String)> { search_and_remap(&dir, &model, &names, naming, search) };

        let local = found(MaterialNaming::BasedOnMaterialName, MaterialSearch::Local);
        assert_eq!(local, [("Skin".to_string(), "assets/models/Materials/Skin.material".to_string())], "local sees the folder and its Materials subfolder");
        let up = found(MaterialNaming::BasedOnMaterialName, MaterialSearch::RecursiveUp);
        assert_eq!(up.iter().map(|p| p.0.as_str()).collect::<Vec<_>>(), ["Skin", "Hair"], "up finds parents but not the sibling tree");
        let all = found(MaterialNaming::BasedOnMaterialName, MaterialSearch::Everywhere);
        assert_eq!(all.len(), 3, "everywhere also finds Cloth");
        assert!(all.iter().any(|p| p.0 == "Cloth" && p.1 == "other/deep/Cloth.material"));
        let by_model = found(MaterialNaming::BasedOnModelNameAndMaterialName, MaterialSearch::Local);
        assert_eq!(by_model, [("Eyes".to_string(), "assets/models/hero-Eyes.material".to_string())]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn meta_texture(path: &Path) -> crate::TextureImport {
        load_asset_meta(path).unwrap().unwrap().importer.as_texture().cloned().unwrap()
    }

    #[test]
    fn extracted_textures_get_the_import_kind_of_their_slot_and_existing_metas_stay() {
        let dir = tmp("kinds");
        // albedo, normal, orm, emission: four different images.
        let scene = scene_with(&[("Skin", [Some(0), Some(1), Some(2), Some(3)])], 4);
        // A leftover .meta where the normal map is about to be written must survive.
        let tex_dir = dir.join("textures");
        std::fs::create_dir_all(&tex_dir).unwrap();
        let keep = tex_dir.join("Skin_orm.png");
        let mut custom = AssetMeta::new_for(&keep);
        custom.importer.as_texture_mut().unwrap().srgb = true;
        save_asset_meta(&keep, &custom).unwrap();
        let mut model = ModelImport::default();
        let r = extract_textures(&scene, &tex_dir, &dir, &mut model).unwrap();
        assert_eq!((r.textures, r.metas), (4, 3), "the orm image had a .meta already");
        let at = |slot: &str| dir.join(&model.texture_remap[&format!("Skin/{slot}")]);
        let n = meta_texture(&at("normal"));
        assert!(n.configured && n.kind == crate::TextureKind::NormalMap && !n.srgb, "normal slot -> NormalMap, linear");
        let a = meta_texture(&at("albedo"));
        assert!(a.configured && a.srgb && a.kind == crate::TextureKind::Default, "colour slots stay sRGB");
        assert!(meta_texture(&at("emission")).srgb);
        assert!(meta_texture(&at("orm")).srgb, "the existing .meta was not touched");
        // A second run writes nothing.
        let again = extract_textures(&scene, &tex_dir, &dir, &mut model).unwrap();
        assert_eq!((again.textures, again.metas), (0, 0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_folder_settings_resolve_inside_the_project() {
        let dir = tmp("dirs");
        let model = dir.join("assets/hero.fbx");
        std::fs::create_dir_all(dir.join("assets")).unwrap();
        let ok = |s: &str| resolve_extract_dir(&dir, &model, s, "Textures");
        assert_eq!(ok("").unwrap(), dir.join("assets/Textures"), "empty = the default beside the model");
        assert_eq!(ok("art/tex").unwrap(), dir.join("art/tex"));
        assert_eq!(ok("art\\tex").unwrap(), dir.join("art/tex"), "backslashes are accepted");
        for bad in ["../out", "a/../../out", "/etc", "C:/x", "a/.."] {
            assert!(ok(bad).is_err(), "{bad} must be rejected");
        }
        #[cfg(unix)]
        {
            let outside = tmp("dirs_outside");
            std::os::unix::fs::symlink(&outside, dir.join("link")).unwrap();
            assert!(ok("link/sub").is_err(), "a symlink out of the project is rejected");
            let _ = std::fs::remove_dir_all(&outside);
        }
        std::fs::create_dir_all(dir.join("assets/b")).unwrap();
        std::fs::create_dir_all(dir.join("assets/A")).unwrap();
        std::fs::create_dir_all(dir.join("assets/.hidden")).unwrap();
        assert_eq!(list_subfolders(&dir, "assets"), ["A", "b"]);
        assert_eq!(list_subfolders(&dir, ""), ["assets"], "the dangling symlink is not a folder");
        assert!(list_subfolders(&dir, "../x").is_empty());
        // An old meta without the settings keeps today's folders.
        let m: ModelImport = ron::from_str("(scale:1.0)").unwrap();
        assert!(m.extract_textures_dir.is_empty() && m.extract_materials_dir.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn texture_names_reduce_to_a_base_name() {
        assert_eq!(texture_stem("textures\\Wood_Albedo.PNG"), "Wood_Albedo");
        assert_eq!(texture_stem("a/b/wood.tar.gz"), "wood.tar");
        assert_eq!(texture_base_name("Wood_Albedo"), "wood");
        assert_eq!(texture_base_name("wood-normal"), "wood");
        assert_eq!(texture_base_name("wood_4k_albedo"), "wood_4k");
        assert_eq!(texture_base_name("brick_diffuse_n"), "brick", "stacked roles");
        assert_eq!(texture_base_name("normal"), "normal", "a bare role is kept");
        assert_eq!(texture_base_name("planks"), "planks");
    }

    fn mat_with(path: PathBuf, albedo: Option<&str>, normal: Option<&str>) {
        let textures = MaterialTextures { albedo: albedo.map(PathBuf::from), normal: normal.map(PathBuf::from), ..Default::default() };
        save_material_file(path, &MaterialFile { name: "x".into(), shader: "standard".into(), params: MaterialParams::default(), features: MaterialFeatures::default(), render_queue: None, textures, custom: Default::default() }).unwrap();
    }

    fn src(name: &str, tex: &[(&str, &str)]) -> SourceMaterial {
        SourceMaterial { name: name.into(), textures: tex.iter().map(|(a, b)| ((*a).into(), (*b).into())).collect() }
    }

    #[test]
    fn search_by_texture_name_ranks_exact_then_base_then_prefix_with_a_stable_tie_break() {
        let dir = tmp("texname");
        let models = dir.join("assets/models");
        std::fs::create_dir_all(models.join("Materials")).unwrap();
        std::fs::create_dir_all(dir.join("other")).unwrap();
        // Exact: same albedo file stem.
        mat_with(dir.join("other/exact.material"), Some("tex/wood_albedo.png"), None);
        // Same base name through other roles.
        mat_with(models.join("Materials/base.material"), Some("t/wood_diffuse.png"), Some("t/wood_nrm.png"));
        // Shared prefix only.
        mat_with(models.join("prefix.material"), Some("t/woodland_color.png"), None);
        // Unrelated, and one without textures.
        mat_with(models.join("stone.material"), Some("t/stone_albedo.png"), None);
        mat_with(models.join("empty.material"), None, None);
        let model = models.join("hero.fbx");
        let by = |sources: &[SourceMaterial], scope| search_matches(&dir, &model, sources, MaterialNaming::BasedOnTextureName, scope);

        let wood = src("Wood", &[("albedo", "wood_albedo")]);
        let all = by(std::slice::from_ref(&wood), MaterialSearch::Everywhere);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].path, "other/exact.material", "an exact stem beats a nearer base-name match");
        assert_eq!(all[0].reason, "same texture 'wood_albedo' in the albedo slot");

        // Without the exact one in scope, the base-name match wins over the prefix match.
        let up = by(std::slice::from_ref(&wood), MaterialSearch::RecursiveUp);
        assert_eq!(up[0].path, "assets/models/Materials/base.material");
        assert_eq!(up[0].reason, "textures share the name 'wood'");
        std::fs::remove_file(models.join("Materials/base.material")).unwrap();
        let pre = by(std::slice::from_ref(&wood), MaterialSearch::Local);
        assert_eq!(pre[0].path, "assets/models/prefix.material");
        assert_eq!(pre[0].reason, "textures share the prefix 'wood'");

        // Nothing for unrelated textures; a source without textures never matches.
        assert!(by(&[src("Glass", &[("albedo", "glass_albedo")]), src("Bare", &[])], MaterialSearch::Everywhere).is_empty());

        // Ties: two equally good materials -> the nearer folder, then the smaller path.
        mat_with(dir.join("other/exact2.material"), Some("x/wood_albedo.png"), None);
        mat_with(models.join("z_near.material"), Some("x/wood_albedo.png"), None);
        let tie = by(std::slice::from_ref(&wood), MaterialSearch::Everywhere);
        assert_eq!(tie[0].path, "assets/models/z_near.material", "nearer folder wins");
        std::fs::remove_file(models.join("z_near.material")).unwrap();
        let tie = by(std::slice::from_ref(&wood), MaterialSearch::Everywhere);
        assert_eq!(tie[0].path, "other/exact.material", "equal distance: smaller path wins");
        let again = by(std::slice::from_ref(&wood), MaterialSearch::Everywhere);
        assert_eq!(tie, again, "deterministic");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn source_materials_use_the_recorded_texture_names_and_the_name_rules_report_a_reason() {
        let mut scene = scene_with(&[("Skin", [Some(0), Some(1), None, None]), ("Hair", [None; 4])], 2);
        scene.identity.texture_names = vec!["skin_albedo".into()];
        let s = source_materials_of(&scene);
        assert_eq!(s[0].textures, [("albedo".to_string(), "skin_albedo".to_string())], "a texture without a recorded name is skipped");
        assert!(s[1].textures.is_empty());
        let dir = tmp("reason");
        std::fs::create_dir_all(dir.join("m")).unwrap();
        mat_with(dir.join("m/Skin.material"), None, None);
        let hits = search_matches(&dir, &dir.join("m/hero.fbx"), &s, MaterialNaming::BasedOnMaterialName, MaterialSearch::Local);
        assert_eq!(hits, [RemapMatch { source: "Skin".into(), path: "m/Skin.material".into(), reason: "named Skin.material".into() }]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_remap_target_must_be_a_material_and_not_the_model() {
        let model = Path::new("a/hero.fbx");
        assert!(reject_target(model, Path::new("a/hero.fbx")).is_some());
        assert!(reject_target(model, Path::new("a/skin.png")).is_some());
        assert!(reject_target(model, Path::new("a/skin.material")).is_none());
    }
}
