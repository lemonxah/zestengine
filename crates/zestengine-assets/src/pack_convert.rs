//! Convert a scanned asset pack ([`crate::pack_scan`]) into project assets.
//!
//! Per asset the importer writes, under `<target>/<asset>/`:
//! `models/` (copied model files, each with a `.meta`; the provided LODs stay
//! separate files), `textures/` (copied, or channel-repacked, each with a
//! `.meta` carrying its colour space / compression / mips), one `.material`
//! per texture set, one `.prefab` per mesh variant (the model + materials,
//! with a LOD Group wired to the provided LOD files) and an
//! `import_report.txt` listing what was mapped and what was not.
//!
//! Planning is pure ([`plan_material`]); file work is in [`import_asset`].
//! Nothing is downloaded and nothing is bundled: it only reads the user's
//! own pack and writes into their project.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use image::RgbaImage;
use zestengine_render::{AlphaMode, MaterialFeatures, MaterialParams};

use crate::megascans::AssetKind;
use crate::pack_scan::{ModelLod, ModelVariant, PackAsset, SetTexture, TextureSet};
use crate::tex_roles::{Chan, Pick, TexRole, match_key, orm_layout};
use crate::unity_convert::{Level, ReportLine};
use crate::{MaterialFile, MaterialTextures, save_material_file};

/// What to do when the target folder of an asset already exists.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DuplicatePolicy {
    /// Import next to it as `name_2`, `name_3`, ... (never touches existing files).
    #[default]
    KeepBoth,
    /// Leave the existing asset alone and report it.
    Skip,
    /// Write over files with the same name; existing `.meta` ids are kept so
    /// references to them hold.
    Overwrite,
}

impl DuplicatePolicy {
    pub const ALL: [DuplicatePolicy; 3] = [DuplicatePolicy::KeepBoth, DuplicatePolicy::Skip, DuplicatePolicy::Overwrite];
    pub fn label(self) -> &'static str {
        match self {
            DuplicatePolicy::KeepBoth => "Keep both (rename the new one)",
            DuplicatePolicy::Skip => "Skip existing",
            DuplicatePolicy::Overwrite => "Overwrite same-named files",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ImportOptions {
    /// Project-relative folder the asset folders go into.
    pub target_dir: String,
    /// Largest texture edge kept (0 = no limit). Applied as the textures'
    /// import `max_size` and when channels are repacked. Also decides which
    /// resolution is used when a pack carries several.
    pub max_texture_size: u32,
    pub duplicates: DuplicatePolicy,
    /// Parallax strength used for displacement maps (uv shift at full height).
    pub displacement_scale: f32,
    /// Write `.prefab` files (otherwise models + materials only).
    pub write_prefabs: bool,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self { target_dir: "megascans".into(), max_texture_size: 4096, duplicates: DuplicatePolicy::KeepBoth, displacement_scale: 0.03, write_prefabs: true }
    }
}

// ---------------------------------------------------------------------------
// Channel maths
// ---------------------------------------------------------------------------

/// One output channel of a repack: read `pick.src` of `img`.
#[derive(Clone, Copy)]
pub struct ChanFrom<'a> {
    pub img: &'a RgbaImage,
    pub pick: Pick,
}

fn chan_index(c: Chan) -> usize {
    match c {
        Chan::R => 0,
        Chan::G => 1,
        Chan::B => 2,
        Chan::A => 3,
    }
}

/// Build an ORM image (R occlusion, G roughness, B metallic, A opaque) from
/// per-channel sources. Missing channels get 255 (no occlusion), 255 (fully
/// rough, the material scalar scales it) and 0 (dielectric). Inputs are
/// sampled with nearest-neighbour scaling when their size differs from `size`.
pub fn pack_channels(sources: [Option<ChanFrom<'_>>; 3], size: (u32, u32)) -> RgbaImage {
    let (w, h) = (size.0.max(1), size.1.max(1));
    let defaults = [255u8, 255, 0];
    let mut out = RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut px = [0u8, 0, 0, 255];
            for (c, src) in sources.iter().enumerate() {
                px[c] = match src {
                    None => defaults[c],
                    Some(s) => {
                        let (iw, ih) = s.img.dimensions();
                        let v = s.img.get_pixel(x * iw / w, y * ih / h).0[chan_index(s.pick.src)];
                        if s.pick.invert { 255 - v } else { v }
                    }
                };
            }
            out.put_pixel(x, y, image::Rgba(px));
        }
    }
    out
}

/// Colour with an opacity map as alpha (decals). `opacity` is read from its R
/// channel and resampled to the colour's size; no opacity keeps the colour's alpha.
pub fn compose_rgba(color: &RgbaImage, opacity: Option<&RgbaImage>) -> RgbaImage {
    let mut out = color.clone();
    if let Some(o) = opacity {
        let (w, h) = out.dimensions();
        let (ow, oh) = o.dimensions();
        for y in 0..h {
            for x in 0..w {
                out.get_pixel_mut(x, y).0[3] = o.get_pixel(x * ow / w.max(1), y * oh / h.max(1)).0[0];
            }
        }
    }
    out
}

/// Flip the green channel of a tangent-space normal (DirectX <-> OpenGL).
pub fn flip_normal_green(img: &mut RgbaImage) {
    for p in img.pixels_mut() {
        p.0[1] = 255 - p.0[1];
    }
}

/// Edge limit for a repack: `size` shrunk (keeping the aspect) so its longer
/// edge is at most `cap` (0 = unlimited).
pub fn capped_size(size: (u32, u32), cap: u32) -> (u32, u32) {
    let long = size.0.max(size.1);
    if cap == 0 || long <= cap {
        return size;
    }
    let s = cap as f32 / long as f32;
    (((size.0 as f32 * s).round() as u32).max(1), ((size.1 as f32 * s).round() as u32).max(1))
}

/// Tiling for a surface that covers `size_m` metres per texture repeat, with
/// one UV unit per metre: `1 / size`. Unknown size keeps the texture at 1.
pub fn tiling_for_size(size_m: Option<[f32; 2]>) -> [f32; 2] {
    match size_m {
        Some([w, h]) if w > 0.0 && h > 0.0 => [1.0 / w, 1.0 / h],
        _ => [1.0, 1.0],
    }
}

/// Screen-coverage thresholds for `n` LOD levels, descending, last 0 so the
/// final level serves everything beyond it: 0.30, 0.12, 0.048, ...
pub fn lod_screen_sizes(n: usize) -> Vec<f32> {
    (0..n).map(|i| if i + 1 == n { 0.0 } else { 0.3 * 0.4f32.powi(i as i32) }).collect()
}

// ---------------------------------------------------------------------------
// Material planning
// ---------------------------------------------------------------------------

/// Where the ORM texture of a material comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum OrmPlan {
    None,
    /// An ORM map already in the engine's layout, used as it is.
    Direct(PathBuf),
    /// Channels taken from other maps: `[occlusion, roughness, metallic]`.
    Repack([Option<(PathBuf, Pick)>; 3]),
}

#[derive(Clone, Debug)]
pub struct MatPlan {
    pub name: String,
    pub params: MaterialParams,
    pub features: MaterialFeatures,
    pub albedo: Option<PathBuf>,
    /// Normal map and whether its green channel needs flipping (DirectX source).
    pub normal: Option<(PathBuf, bool)>,
    pub orm: OrmPlan,
    pub ao: Option<PathBuf>,
    pub roughness: Option<PathBuf>,
    pub metallic: Option<PathBuf>,
    pub opacity: Option<PathBuf>,
    pub displacement: Option<PathBuf>,
    pub emission: Option<PathBuf>,
    /// Translucency map (foliage, cloth): drives the standard shader's back-lit translucency.
    pub translucency: Option<PathBuf>,
    pub notes: Vec<ReportLine>,
}

impl MatPlan {
    /// Anything to write a material for.
    pub fn is_empty(&self) -> bool {
        self.albedo.is_none()
            && self.normal.is_none()
            && self.orm == OrmPlan::None
            && self.ao.is_none()
            && self.roughness.is_none()
            && self.metallic.is_none()
            && self.opacity.is_none()
            && self.displacement.is_none()
            && self.emission.is_none()
            && self.translucency.is_none()
    }
}

/// What planning needs to know about the asset.
#[derive(Clone, Copy, Debug)]
pub struct PlanCtx {
    pub kind: AssetKind,
    pub size_m: Option<[f32; 2]>,
    pub max_texture_size: u32,
    pub displacement_scale: f32,
}

/// The best texture among `roles` (first role with any texture wins): the
/// largest resolution within `cap`, else the smallest above it.
fn pick_best<'a>(set: &'a TextureSet, roles: &[TexRole], cap: u32) -> Option<&'a SetTexture> {
    roles.iter().find_map(|role| {
        set.textures.iter().filter(|t| t.role == *role).min_by(|a, b| {
            let key = |t: &SetTexture| match t.resolution {
                Some(r) if cap == 0 || r <= cap => (0u8, u32::MAX - r),
                Some(r) => (1, r),
                None => (2, 0),
            };
            key(a).cmp(&key(b)).then_with(|| a.path.cmp(&b.path))
        })
    })
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

fn note(level: Level, property: &str, text: impl Into<String>) -> ReportLine {
    ReportLine { level, property: property.to_owned(), note: text.into() }
}

/// Map one texture set to a material: roles to slots, inversions, packed-map
/// splitting, alpha / two-sided / tiling decisions, and a report line for every
/// map that is mapped, approximated or dropped.
pub fn plan_material(set: &TextureSet, ctx: &PlanCtx) -> MatPlan {
    let cap = ctx.max_texture_size;
    let mut p = MatPlan {
        name: set.name.clone(),
        params: MaterialParams::default(),
        features: MaterialFeatures::default(),
        albedo: None,
        normal: None,
        orm: OrmPlan::None,
        ao: None,
        roughness: None,
        metallic: None,
        opacity: None,
        displacement: None,
        emission: None,
        translucency: None,
        notes: Vec::new(),
    };
    let best = |roles: &[TexRole]| pick_best(set, roles, cap);

    // Surplus textures of a role are named so the choice is visible.
    for role in TexRole::ALL {
        if role == TexRole::Ignore {
            continue;
        }
        let n = set.textures.iter().filter(|t| t.role == role).count();
        if n > 1
            && let Some(b) = best(&[role])
        {
            p.notes.push(note(Level::Info, role.label(), format!("{n} files with this role; using {}", file_name(&b.path))));
        }
    }

    if let Some(t) = best(&[TexRole::Albedo]) {
        p.albedo = Some(t.path.clone());
        p.notes.push(note(Level::Mapped, "albedo", file_name(&t.path)));
    }

    // Normal map. The engine reads OpenGL (Y+); a DirectX map is flipped on import.
    if let Some(t) = best(&[TexRole::Normal, TexRole::NormalDx]) {
        let flip = t.role == TexRole::NormalDx;
        p.normal = Some((t.path.clone(), flip));
        p.features.normal_map = true;
        p.notes.push(if flip {
            note(Level::Mapped, "normal", format!("{} (DirectX convention: green channel flipped on import)", file_name(&t.path)))
        } else {
            note(Level::Mapped, "normal", format!("{} (OpenGL convention, used as is)", file_name(&t.path)))
        });
        if set.textures.iter().any(|x| x.role == TexRole::Normal) && set.textures.iter().any(|x| x.role == TexRole::NormalDx) {
            p.notes.push(note(Level::Info, "normal", "both OpenGL and DirectX normal maps found; using the OpenGL one"));
        }
    }

    // Occlusion / roughness / metallic: a packed map wins over split maps.
    let split_ao = best(&[TexRole::Ao]);
    let split_rough = best(&[TexRole::Roughness]);
    let split_gloss = best(&[TexRole::Gloss]);
    let split_metal = best(&[TexRole::Metallic]);
    if let Some(t) = best(&[TexRole::Orm]) {
        p.orm = OrmPlan::Direct(t.path.clone());
        p.params.roughness = 1.0;
        p.params.metallic = 1.0;
        p.notes.push(note(Level::Mapped, "orm", format!("{} (R occlusion, G roughness, B metallic, used as is)", file_name(&t.path))));
        for (t, what) in [(split_ao, "ambient occlusion"), (split_rough.or(split_gloss), "roughness"), (split_metal, "metallic")] {
            if let Some(t) = t {
                p.notes.push(note(Level::Info, what, format!("{} ignored: the ORM map already carries it", file_name(&t.path))));
            }
        }
    } else if let Some(packed) = best(&[TexRole::Rma, TexRole::MetalRough, TexRole::MetalGloss, TexRole::MaskMap]) {
        let layout = orm_layout(packed.role).expect("packed role has a layout");
        let from_split = |t: Option<&SetTexture>, invert: bool| t.map(|t| (t.path.clone(), Pick { src: Chan::R, invert }));
        let packed_pick = |c: usize| layout[c].map(|pk| (packed.path.clone(), pk));
        let ao = packed_pick(0).or_else(|| from_split(split_ao, false));
        let rough = packed_pick(1).or_else(|| from_split(split_rough, false)).or_else(|| from_split(split_gloss, true));
        let metal = packed_pick(2).or_else(|| from_split(split_metal, false));
        p.params.roughness = 1.0;
        p.params.metallic = 1.0;
        p.notes.push(note(
            Level::Approx,
            "orm",
            format!("{} is a {} map: repacked into the engine's ORM layout (R occlusion, G roughness, B metallic)", file_name(&packed.path), packed.role.label()),
        ));
        p.orm = OrmPlan::Repack([ao, rough, metal]);
    } else {
        if let Some(t) = split_ao {
            p.ao = Some(t.path.clone());
            p.notes.push(note(Level::Mapped, "ambient occlusion", file_name(&t.path)));
        }
        if let Some(t) = split_rough {
            p.roughness = Some(t.path.clone());
            p.params.roughness = 1.0;
            p.notes.push(note(Level::Mapped, "roughness", file_name(&t.path)));
            if let Some(g) = split_gloss {
                p.notes.push(note(Level::Info, "gloss", format!("{} ignored: a roughness map is present", file_name(&g.path))));
            }
        } else if let Some(g) = split_gloss {
            p.roughness = Some(g.path.clone());
            p.params.roughness = 1.0;
            p.params.roughness_invert = true;
            p.notes.push(note(Level::Mapped, "roughness", format!("{} (gloss: the roughness slot is inverted)", file_name(&g.path))));
        } else {
            p.params.roughness = 0.8;
        }
        if let Some(t) = split_metal {
            p.metallic = Some(t.path.clone());
            p.params.metallic = 1.0;
            p.notes.push(note(Level::Mapped, "metallic", file_name(&t.path)));
        } else {
            p.params.metallic = 0.0;
        }
    }

    // Opacity: alpha-tested. Foliage (opacity with translucency) and plants are two-sided.
    let has_translucency = best(&[TexRole::Translucency]);
    if let Some(t) = best(&[TexRole::Opacity]) {
        p.opacity = Some(t.path.clone());
        p.features.alpha_mode = AlphaMode::Cutout;
        p.params.alpha_cutoff = 0.5;
        p.notes.push(note(Level::Mapped, "opacity", format!("{} (alpha clip at 0.5)", file_name(&t.path))));
    }
    if matches!(ctx.kind, AssetKind::Plant | AssetKind::Atlas) || (p.opacity.is_some() && has_translucency.is_some()) {
        p.features.double_sided = true;
        p.notes.push(note(Level::Mapped, "two-sided", "foliage and atlas cards are drawn on both faces"));
    }
    if let Some(t) = has_translucency {
        // The Megascans map holds the colour of the light that passes (bright = translucent); the
        // standard shader reads its red channel as the amount and tints by the albedo.
        p.translucency = Some(t.path.clone());
        p.params.transmission.translucency = 1.0;
        p.params.transmission.map_is_thickness = false;
        p.notes.push(note(Level::Approx, "translucency", format!("{} -> Translucency Map (red channel as the amount, tinted by the albedo; the map's own colour is not carried)", file_name(&t.path))));
    }

    // Height: displacement as parallax; bump only as a stand-in on tiling surfaces.
    if let Some(t) = best(&[TexRole::Height]) {
        p.displacement = Some(t.path.clone());
        p.params.displacement_scale = ctx.displacement_scale;
        p.notes.push(note(Level::Approx, "displacement", format!("{} as parallax occlusion height, scale {}", file_name(&t.path), ctx.displacement_scale)));
        if let Some(b) = best(&[TexRole::Bump]) {
            p.notes.push(note(Level::Info, "bump", format!("{} ignored: a displacement map is present", file_name(&b.path))));
        }
    } else if let Some(b) = best(&[TexRole::Bump]) {
        if matches!(ctx.kind, AssetKind::Surface | AssetKind::Imperfection) {
            p.displacement = Some(b.path.clone());
            p.params.displacement_scale = ctx.displacement_scale * 0.5;
            p.notes.push(note(Level::Approx, "bump", format!("{} used as a low-detail height for parallax", file_name(&b.path))));
        } else {
            p.notes.push(note(Level::Info, "bump", format!("{} ignored (the normal map carries the detail)", file_name(&b.path))));
        }
    }

    if let Some(t) = best(&[TexRole::Emission]) {
        p.emission = Some(t.path.clone());
        p.features.emission = true;
        p.params.emission_color = [1.0, 1.0, 1.0];
        p.notes.push(note(Level::Mapped, "emission", file_name(&t.path)));
    }
    if let Some(t) = best(&[TexRole::Specular]) {
        p.notes.push(note(Level::Info, "specular", format!("{} not used: the shader derives reflectance from albedo and metallic", file_name(&t.path))));
    }
    if let Some(t) = best(&[TexRole::Cavity]) {
        p.notes.push(note(Level::Info, "cavity", format!("{} not used (baked cavity shading; occlusion covers it)", file_name(&t.path))));
    }
    if let Some(t) = best(&[TexRole::Fuzz]) {
        p.notes.push(note(Level::Unsupported, "fuzz", format!("{} not used: no sheen term in the standard shader", file_name(&t.path))));
    }

    // Tiling surfaces: one texture repeat covers the scanned area.
    if matches!(ctx.kind, AssetKind::Surface | AssetKind::Imperfection) {
        let t = tiling_for_size(ctx.size_m);
        p.params.albedo_tiling = t;
        p.params.normal_tiling = t;
        p.params.orm_tiling = t;
        match ctx.size_m {
            Some([w, h]) => p.notes.push(note(Level::Mapped, "tiling", format!("{w} x {h} m per repeat: tiling {:.3} x {:.3} per metre of UV", t[0], t[1]))),
            None => p.notes.push(note(Level::Info, "tiling", "real-world size unknown: tiling 1 (set the tile size in the import dialog or the material)")),
        }
    }
    p
}

// ---------------------------------------------------------------------------
// Assigning sets to a model's materials
// ---------------------------------------------------------------------------

/// For each of a model's material names, the index of the texture set (in
/// `set_names`) that belongs to it: an equal normalised name, then a name that
/// contains the other, then the only set. `None` = no match.
pub fn assign_sets(material_names: &[String], set_names: &[String]) -> Vec<Option<usize>> {
    let sets: Vec<String> = set_names.iter().map(|s| match_key(s)).collect();
    material_names
        .iter()
        .map(|m| {
            let key = match_key(m);
            sets.iter()
                .position(|s| !s.is_empty() && *s == key)
                .or_else(|| sets.iter().position(|s| s.len() >= 3 && !key.is_empty() && (key.contains(s.as_str()) || s.contains(key.as_str()))))
                .or_else(|| (sets.len() == 1).then_some(0))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct AssetResult {
    pub name: String,
    /// Absolute folder written.
    pub dir: PathBuf,
    pub materials: Vec<PathBuf>,
    pub models: Vec<PathBuf>,
    pub prefabs: Vec<PathBuf>,
    pub textures: usize,
    pub skipped: bool,
    pub error: Option<String>,
    pub lines: Vec<ReportLine>,
}

impl AssetResult {
    pub fn count(&self, level: Level) -> usize {
        self.lines.iter().filter(|l| l.level == level).count()
    }

    pub fn to_text(&self) -> String {
        let mut s = format!("Import report for '{}'\nfolder: {}\n", self.name, self.dir.display());
        if let Some(e) = &self.error {
            s.push_str(&format!("\nFAILED: {e}\n"));
        }
        let sections = [
            (Level::Mapped, "Mapped"),
            (Level::Approx, "Approximated (check visually)"),
            (Level::Unsupported, "Not supported yet"),
            (Level::Unmapped, "Not mapped"),
            (Level::Info, "Notes"),
        ];
        for (level, title) in sections {
            let rows: Vec<&ReportLine> = self.lines.iter().filter(|l| l.level == level).collect();
            if rows.is_empty() {
                continue;
            }
            s.push_str(&format!("\n{title} ({}):\n", rows.len()));
            for l in rows {
                s.push_str(&format!("  {}: {}\n", l.property, l.note));
            }
        }
        s
    }
}

#[derive(Clone, Debug, Default)]
pub struct ImportOutcome {
    pub assets: Vec<AssetResult>,
    pub cancelled: bool,
}

impl ImportOutcome {
    pub fn message(&self) -> String {
        let done = self.assets.iter().filter(|a| !a.skipped && a.error.is_none()).count();
        let skipped = self.assets.iter().filter(|a| a.skipped).count();
        let failed = self.assets.iter().filter(|a| a.error.is_some()).count();
        let unsupported: usize = self.assets.iter().map(|a| a.count(Level::Unsupported)).sum();
        let mut s = format!("Imported {done} asset(s)");
        if skipped > 0 {
            s.push_str(&format!(", {skipped} skipped"));
        }
        if failed > 0 {
            s.push_str(&format!(", {failed} failed"));
        }
        if unsupported > 0 {
            s.push_str(&format!("; {unsupported} unsupported map(s), see each import_report.txt"));
        }
        if self.cancelled {
            s.push_str(" (cancelled)");
        }
        s
    }
}

// ---------------------------------------------------------------------------
// File work
// ---------------------------------------------------------------------------

fn rel(project_root: &Path, p: &Path) -> String {
    crate::material_remap::project_relative(project_root, p)
}

/// `dir/name`, or `dir/stem_2.ext`, ... when taken by something else.
fn unique_in(dir: &Path, name: &str, taken: &BTreeMap<PathBuf, PathBuf>, overwrite: bool) -> PathBuf {
    let first = dir.join(name);
    if overwrite || (!first.exists() && !taken.values().any(|v| *v == first)) {
        return first;
    }
    let (stem, ext) = match name.rsplit_once('.') {
        Some((s, e)) => (s.to_owned(), format!(".{e}")),
        None => (name.to_owned(), String::new()),
    };
    (2..)
        .map(|n| dir.join(format!("{stem}_{n}{ext}")))
        .find(|p| !p.exists() && !taken.values().any(|v| v == p))
        .expect("unbounded range")
}

/// Write `meta` for `abs`, keeping the id of an existing `.meta`.
fn save_meta_keep_id(abs: &Path, mut importer: crate::ImporterSettings) -> Result<()> {
    let id = crate::load_asset_meta(abs).ok().flatten().map(|m| m.id).unwrap_or_default();
    // Keep settings the user may have made on an existing model meta.
    if let (Some(old), Some(new)) = (crate::load_asset_meta(abs).ok().flatten().and_then(|m| m.importer.as_model().cloned()), importer.as_model_mut()) {
        *new = old;
    }
    crate::save_asset_meta(abs, &crate::AssetMeta { id, importer })
}

/// Which slot a texture is written for (decides its import settings).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TexSlot {
    Albedo,
    Normal { flip_green: bool },
    Orm,
    /// AO / roughness / metallic / brush.
    Single,
    Opacity,
    Height,
    Emission,
    /// Colour with alpha (decals).
    ColorAlpha,
}

/// The import settings for a texture in `slot`: colour space and kind per map
/// (sRGB albedo / emission, BC5 normal, BC4 single channels, BC7 data), mips
/// on, alpha-coverage kept for cut-outs, the edge cap from the options.
pub fn texture_import_for(slot: TexSlot, max_size: u32) -> crate::TextureImport {
    use crate::{Compression, SlotExpect};
    let expect = match slot {
        TexSlot::Albedo | TexSlot::Emission | TexSlot::ColorAlpha => SlotExpect::Color,
        TexSlot::Normal { .. } => SlotExpect::Normal,
        TexSlot::Orm => SlotExpect::Data,
        TexSlot::Single | TexSlot::Opacity | TexSlot::Height => SlotExpect::Single,
    };
    let mut t = crate::import_for_expectation(expect);
    t.max_size = max_size;
    t.mips.enabled = true;
    match slot {
        TexSlot::Normal { flip_green } => t.normal.flip_green = flip_green,
        TexSlot::Opacity => {
            t.mips.preserve_coverage = true;
            t.mips.cutoff = 0.5;
        }
        TexSlot::Height => t.compression = Compression::High,
        TexSlot::ColorAlpha => t.alpha_is_transparency = true,
        _ => {}
    }
    t
}

struct Writer<'a> {
    project_root: &'a Path,
    asset_dir: PathBuf,
    opts: &'a ImportOptions,
    overwrite: bool,
    /// source texture -> written copy (one copy per source, whatever slots use it)
    copied: BTreeMap<PathBuf, PathBuf>,
    textures: usize,
}

impl Writer<'_> {
    fn copy_texture(&mut self, src: &Path, slot: TexSlot) -> Result<PathBuf> {
        if let Some(done) = self.copied.get(src) {
            return Ok(done.clone());
        }
        let dir = self.asset_dir.join("textures");
        std::fs::create_dir_all(&dir)?;
        let dest = unique_in(&dir, &file_name(src), &self.copied, self.overwrite);
        std::fs::copy(src, &dest).with_context(|| format!("copying {}", src.display()))?;
        save_meta_keep_id(&dest, crate::ImporterSettings::Texture(texture_import_for(slot, self.opts.max_texture_size)))?;
        self.copied.insert(src.to_owned(), dest.clone());
        self.textures += 1;
        Ok(dest)
    }

    fn write_png(&mut self, stem: &str, img: &RgbaImage, slot: TexSlot) -> Result<PathBuf> {
        let dir = self.asset_dir.join("textures");
        std::fs::create_dir_all(&dir)?;
        let dest = unique_in(&dir, &format!("{}.png", crate::material_remap::file_stem_of(stem, "texture")), &self.copied, self.overwrite);
        img.save(&dest).with_context(|| format!("writing {}", dest.display()))?;
        save_meta_keep_id(&dest, crate::ImporterSettings::Texture(texture_import_for(slot, self.opts.max_texture_size)))?;
        self.copied.insert(dest.clone(), dest.clone());
        self.textures += 1;
        Ok(dest)
    }

    fn rel(&self, p: &Path) -> PathBuf {
        PathBuf::from(rel(self.project_root, p))
    }

    /// [`Self::copy_texture`], returning the project-relative path.
    fn copy_rel(&mut self, src: &Path, slot: TexSlot) -> Result<PathBuf> {
        let dest = self.copy_texture(src, slot)?;
        Ok(self.rel(&dest))
    }

    fn png_rel(&mut self, stem: &str, img: &RgbaImage, slot: TexSlot) -> Result<PathBuf> {
        let dest = self.write_png(stem, img, slot)?;
        Ok(self.rel(&dest))
    }
}

fn load_rgba(path: &Path) -> Result<RgbaImage> {
    Ok(crate::material_file::open_image_compat(path)?.into_rgba8())
}

/// Resize to `size` when it differs (Triangle filter, good for downscaling).
fn fit(img: RgbaImage, size: (u32, u32)) -> RgbaImage {
    if img.dimensions() == size { img } else { image::imageops::resize(&img, size.0, size.1, image::imageops::FilterType::Triangle) }
}

/// Write a planned material: copy / repack its textures, save the `.material`.
fn write_material(plan: &MatPlan, w: &mut Writer<'_>, mats_dir: &Path, lines: &mut Vec<ReportLine>) -> Result<PathBuf> {
    let mut tex = MaterialTextures::default();
    if let Some(src) = &plan.albedo {
        tex.albedo = Some(w.copy_rel(src, TexSlot::Albedo)?);
    }
    if let Some((src, flip)) = &plan.normal {
        tex.normal = Some(w.copy_rel(src, TexSlot::Normal { flip_green: *flip })?);
    }
    match &plan.orm {
        OrmPlan::None => {}
        OrmPlan::Direct(src) => tex.orm = Some(w.copy_rel(src, TexSlot::Orm)?),
        OrmPlan::Repack(chans) => {
            // Load each distinct source once, at the capped size of the largest.
            let mut images: BTreeMap<PathBuf, RgbaImage> = BTreeMap::new();
            let mut size = (1u32, 1u32);
            for (path, _) in chans.iter().flatten() {
                if !images.contains_key(path) {
                    let img = load_rgba(path)?;
                    size = (size.0.max(img.width()), size.1.max(img.height()));
                    images.insert(path.clone(), img);
                }
            }
            let size = capped_size(size, w.opts.max_texture_size);
            let fitted: BTreeMap<PathBuf, RgbaImage> = images.into_iter().map(|(p, i)| (p, fit(i, size))).collect();
            let sources = [0, 1, 2].map(|c| chans[c].as_ref().map(|(p, pick)| ChanFrom { img: &fitted[p], pick: *pick }));
            let img = pack_channels(sources, size);
            let stem = format!("{}_orm", plan.name);
            tex.orm = Some(w.png_rel(&stem, &img, TexSlot::Orm)?);
            lines.push(note(Level::Info, "orm", format!("wrote a repacked ORM texture ({}x{})", size.0, size.1)));
        }
    }
    if let Some(src) = &plan.ao {
        tex.ao = Some(w.copy_rel(src, TexSlot::Single)?);
    }
    if let Some(src) = &plan.roughness {
        tex.roughness = Some(w.copy_rel(src, TexSlot::Single)?);
    }
    if let Some(src) = &plan.metallic {
        tex.metallic = Some(w.copy_rel(src, TexSlot::Single)?);
    }
    if let Some(src) = &plan.opacity {
        tex.opacity = Some(w.copy_rel(src, TexSlot::Opacity)?);
    }
    if let Some(src) = &plan.displacement {
        tex.displacement = Some(w.copy_rel(src, TexSlot::Height)?);
    }
    if let Some(src) = &plan.emission {
        tex.emission = Some(w.copy_rel(src, TexSlot::Emission)?);
    }
    if let Some(src) = &plan.translucency {
        tex.translucency = Some(w.copy_rel(src, TexSlot::Single)?);
    }
    let file = MaterialFile {
        name: plan.name.clone(),
        shader: "standard".into(),
        params: plan.params,
        features: plan.features,
        render_queue: None,
        textures: tex,
        custom: BTreeMap::new(),
    };
    std::fs::create_dir_all(mats_dir)?;
    let stem = crate::material_remap::file_stem_of(&plan.name, "material");
    let path = if w.overwrite { mats_dir.join(format!("{stem}.material")) } else { crate::material_remap::unique_file(mats_dir, &stem, "material") };
    save_material_file(&path, &file)?;
    Ok(path)
}

/// Copy a model file (and what it references) into `<asset>/models/`.
fn copy_model(src: &Path, models_dir: &Path, taken: &BTreeMap<PathBuf, PathBuf>, overwrite: bool) -> Result<PathBuf> {
    let ext = src.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    let name = file_name(src);
    let stem = src.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    // glTF brings external buffers / images: each model gets a folder so they cannot collide.
    let dir = if ext == "gltf" { models_dir.join(crate::material_remap::file_stem_of(&stem, "model")) } else { models_dir.to_owned() };
    std::fs::create_dir_all(&dir)?;
    let dest = unique_in(&dir, &name, taken, overwrite);
    std::fs::copy(src, &dest).with_context(|| format!("copying {}", src.display()))?;
    let src_dir = src.parent().unwrap_or(Path::new("."));
    if ext == "gltf"
        && let Ok(text) = std::fs::read_to_string(src)
        && let Ok(json) = serde_json::from_str::<serde_json::Value>(&text)
    {
        for key in ["buffers", "images"] {
            for item in json[key].as_array().into_iter().flatten() {
                let Some(uri) = item["uri"].as_str() else { continue };
                if uri.starts_with("data:") || uri.contains("://") {
                    continue;
                }
                let decoded = uri.replace("%20", " ");
                let Some(rel_path) = crate::pack_zip::safe_rel(&decoded) else { continue };
                let from = src_dir.join(&rel_path);
                let to = dest.parent().unwrap_or(&dir).join(&rel_path);
                if from.is_file() {
                    if let Some(parent) = to.parent() {
                        std::fs::create_dir_all(parent)?;
                    }
                    std::fs::copy(&from, &to).with_context(|| format!("copying {}", from.display()))?;
                }
            }
        }
    }
    Ok(dest)
}

/// A node name safe to use as a file stem (the variant, or the asset name).
fn variant_label(asset: &PackAsset, v: &ModelVariant) -> String {
    if v.name.is_empty() || v.name == asset.name { asset.name.clone() } else { format!("{}_{}", asset.name, v.name) }
}

/// Everything per-asset the importer needs while it runs.
struct Job<'a> {
    asset: &'a PackAsset,
    opts: &'a ImportOptions,
    project_root: &'a Path,
}

fn import_models(job: &Job<'_>, w: &mut Writer<'_>, sets: &[(String, PathBuf)], result: &mut AssetResult) -> Result<()> {
    let models_dir = w.asset_dir.join("models");
    let set_names: Vec<String> = sets.iter().map(|s| s.0.clone()).collect();
    let mut taken: BTreeMap<PathBuf, PathBuf> = BTreeMap::new();
    for variant in &job.asset.variants {
        let mut copied: Vec<(u32, PathBuf)> = Vec::new();
        for ModelLod { lod, path, .. } in &variant.lods {
            let dest = copy_model(path, &models_dir, &taken, w.overwrite)?;
            taken.insert(path.clone(), dest.clone());
            save_meta_keep_id(&dest, crate::ImporterSettings::for_extension(&dest))?;
            result.models.push(dest.clone());
            copied.push((*lod, dest));
        }
        let Some((_, lod0)) = copied.first().cloned() else { continue };

        // Load each LOD to learn its material names (remap) and parts (prefab).
        let mut part0: Option<(crate::Scene, Vec<usize>)> = None;
        for (lod, dest) in &copied {
            let scene = match crate::load_model_with_meta(dest) {
                Ok(s) => s,
                Err(e) => {
                    result.lines.push(note(Level::Unsupported, "model", format!("{} could not be loaded ({e:#}); copied but not wired", file_name(dest))));
                    continue;
                }
            };
            let names: Vec<String> = (0..scene.materials.len()).map(|i| scene.identity.material_names.get(i).cloned().unwrap_or_else(|| scene.materials[i].name.clone())).collect();
            let assigned = assign_sets(&names, &set_names);
            if assigned.iter().any(Option::is_none) && sets.len() > 1 {
                result.lines.push(note(Level::Approx, "materials", format!("{}: material(s) matched no texture set by name; the first set is used", file_name(dest))));
            }
            let picks: Vec<usize> = assigned.iter().map(|a| a.unwrap_or(0)).collect();
            // Remap source material names to the new materials (applied by the model importer).
            if !sets.is_empty()
                && let Ok(Some(mut meta)) = crate::load_asset_meta(dest)
                && let Some(model) = meta.importer.as_model_mut()
            {
                for (name, set) in names.iter().zip(&picks) {
                    model.material_remap.insert(name.clone(), rel(job.project_root, &sets[*set].1));
                }
                let _ = crate::save_asset_meta(dest, &meta);
            }
            if *lod == copied[0].0 {
                part0 = Some((scene, picks));
            }
        }

        if !job.opts.write_prefabs {
            continue;
        }
        let Some((scene, picks)) = part0 else { continue };
        let prefab = build_prefab(job, variant, &copied, &scene, &picks, sets, lod0.as_path(), result)?;
        let path = w.asset_dir.join(format!("{}.prefab", crate::material_remap::file_stem_of(&variant_label(job.asset, variant), "asset")));
        crate::save_scene_file(&path, &prefab)?;
        result.prefabs.push(path);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_prefab(job: &Job<'_>, variant: &ModelVariant, copied: &[(u32, PathBuf)], scene: &crate::Scene, picks: &[usize], sets: &[(String, PathBuf)], lod0: &Path, result: &mut AssetResult) -> Result<crate::SceneFile> {
    use crate::scene_file::{ComponentData, MaterialRef, ModelInstanceRef, ModelPart, ObjectSource, SceneEntry, SceneFile};
    let model_rel = rel(job.project_root, lod0);
    let parts = ModelPart::of(scene);
    let root_id = zestengine_core::ObjectId::new().to_string();
    let blank = |name: String, source: ObjectSource| SceneEntry {
        id: String::new(),
        name,
        source,
        enabled: true,
        static_geometry: false,
        lightmap_scale: 1.0,
        cast_shadows: zestengine_core::ShadowCasting::On,
        layer: 0,
        material: MaterialRef::Model,
        extra_materials: Vec::new(),
        renderer_split: false,
        parent: None,
        components: Vec::new(),
        translation: [0.0; 3],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0; 3],
        model: None,
        model_part: None,
        prefab: None,
        prefab_parent: None,
        net_role: Default::default(),
        component_roles: Default::default(),
    };
    let mut root = blank(variant_label(job.asset, variant), ObjectSource::Empty);
    root.id = root_id;
    root.model = Some(ModelInstanceRef { path: model_rel, model: scene.identity.guid, removed: Vec::new() });
    let mut file = SceneFile { entries: vec![root], ..Default::default() };
    let parts_for = parts.clone();
    file.expand_models(&move |_| Some(parts_for.clone()));

    // Materials by the slot's material index.
    let mat_of = |slot_material: usize| -> Option<MaterialRef> {
        let set = *picks.get(slot_material)?;
        sets.get(set).map(|s| MaterialRef::File(rel(job.project_root, &s.1)))
    };
    for (entry, part_inst) in file.entries.iter_mut().skip(1).zip(scene.instances.iter().filter(|i| !i.slots.is_empty())) {
        let refs: Vec<MaterialRef> = part_inst.slots.iter().map(|s| mat_of(s.material).unwrap_or(MaterialRef::Model)).collect();
        entry.set_renderer_materials(&refs);
    }

    // The provided LODs: only for a single-part model (the LOD Group swaps the whole model).
    if copied.len() > 1 {
        if parts.len() == 1 && file.entries.len() == 2 {
            let sizes = lod_screen_sizes(copied.len());
            let levels = copied
                .iter()
                .enumerate()
                .map(|(i, (_, p))| zestengine_core::LodLevel { max_distance: 15.0 * 2f32.powi(i as i32), model: if i == 0 { String::new() } else { rel(job.project_root, p) }, keep: 0.0, screen_size: sizes[i] })
                .collect();
            let lod = zestengine_core::LodComponent { levels, cull_distance: 0.0, by_screen_size: true };
            file.entries[1].components.push(ComponentData::of(&lod));
            result.lines.push(note(Level::Mapped, "LODs", format!("{} provided LOD levels wired to a LOD Group by screen size", copied.len())));
        } else {
            result.lines.push(note(Level::Approx, "LODs", "the model has several parts, so the provided LOD files are copied but not wired; the engine's automatic LOD chain applies"));
        }
    } else if matches!(job.asset.kind, AssetKind::Plant) {
        result.lines.push(note(Level::Info, "LODs", "one LOD only: the engine's automatic LOD chain and octahedral impostor handle distance"));
    }
    Ok(file)
}

/// Import one asset into the project. Errors are returned in the result, never
/// thrown, so one broken asset does not stop a batch.
pub fn import_asset(asset: &PackAsset, project_root: &Path, opts: &ImportOptions, progress: &mut dyn FnMut(&str) -> bool) -> AssetResult {
    let mut result = AssetResult { name: asset.name.clone(), ..Default::default() };
    let name_stem = crate::material_remap::file_stem_of(&asset.name, "asset");
    let target = project_root.join(opts.target_dir.trim_matches('/'));
    let mut dir = target.join(&name_stem);
    let overwrite = opts.duplicates == DuplicatePolicy::Overwrite;
    if dir.exists() {
        match opts.duplicates {
            DuplicatePolicy::Skip => {
                result.dir = dir;
                result.skipped = true;
                result.lines.push(note(Level::Info, "skipped", "the target folder already exists"));
                return result;
            }
            DuplicatePolicy::KeepBoth => dir = (2..).map(|n| target.join(format!("{name_stem}_{n}"))).find(|p| !p.exists()).expect("unbounded range"),
            DuplicatePolicy::Overwrite => {}
        }
    }
    result.dir = dir.clone();
    if let Err(e) = run_import(asset, project_root, opts, &dir, overwrite, progress, &mut result) {
        result.error = Some(format!("{e:#}"));
    }
    // The report goes next to the result either way.
    if dir.is_dir() {
        let _ = std::fs::write(dir.join("import_report.txt"), result.to_text());
    }
    result
}

fn run_import(asset: &PackAsset, project_root: &Path, opts: &ImportOptions, dir: &Path, overwrite: bool, progress: &mut dyn FnMut(&str) -> bool, result: &mut AssetResult) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    result.lines.extend(asset.warnings.iter().map(|w| note(Level::Info, "scan", w.clone())));
    let mut w = Writer { project_root, asset_dir: dir.to_owned(), opts, overwrite, copied: BTreeMap::new(), textures: 0 };
    let ctx = PlanCtx { kind: asset.kind, size_m: asset.size_m, max_texture_size: opts.max_texture_size, displacement_scale: opts.displacement_scale };
    let job = Job { asset, opts, project_root };

    if asset.kind == AssetKind::Brush {
        for t in asset.sets.iter().flat_map(|s| &s.textures).filter(|t| t.role != TexRole::Ignore || true) {
            if !progress(&file_name(&t.path)) {
                anyhow::bail!("cancelled");
            }
            w.copy_texture(&t.path, TexSlot::Single)?;
        }
        result.textures = w.textures;
        result.lines.push(note(Level::Info, "brush", "brush alphas copied as single-channel textures; ZestEngine has no brush asset type to wire them to"));
        return Ok(());
    }

    // Materials, one per usable non-billboard set.
    let mut sets: Vec<(String, PathBuf)> = Vec::new();
    for set in &asset.sets {
        if set.billboard {
            result.lines.push(note(Level::Info, "billboard", format!("{} billboard texture(s) skipped: the engine builds its own octahedral impostor for distant meshes", set.textures.len())));
            continue;
        }
        if !progress(&set.name) {
            anyhow::bail!("cancelled");
        }
        let plan = plan_material(set, &ctx);
        if plan.is_empty() {
            if set.textures.iter().any(|t| t.role != TexRole::Ignore) || set.name != "Unsorted" {
                result.lines.push(note(Level::Info, &set.name, "no usable texture role; no material written (assign roles in the dialog)"));
            }
            continue;
        }
        result.lines.extend(plan.notes.iter().map(|l| ReportLine { property: format!("{}: {}", set.name, l.property), ..l.clone() }));
        let mats_dir = dir.to_owned();
        let path = write_material(&plan, &mut w, &mats_dir, &mut result.lines)?;
        result.materials.push(path.clone());
        sets.push((set.name.clone(), path));
    }
    for t in asset.sets.iter().flat_map(|s| &s.textures).filter(|t| t.role == TexRole::Ignore && !t.path.as_os_str().is_empty()) {
        result.lines.push(note(Level::Unmapped, &file_name(&t.path), "no texture role was recognised for this file (it was not copied); assign one in the dialog to use it"));
    }

    if asset.kind == AssetKind::Decal && !asset.sets.is_empty() {
        import_decal(&job, &mut w, result)?;
    }

    import_models(&job, &mut w, &sets, result)?;
    if asset.variants.is_empty() {
        match asset.kind {
            AssetKind::Surface | AssetKind::Imperfection => result.lines.push(note(Level::Info, "material", "tiling surface: assign the material to any mesh; the tiling already matches the scanned size")),
            AssetKind::Atlas => result.lines.push(note(Level::Info, "atlas", "imported as a cut-out two-sided material; slice the atlas into elements with the sprite tools if you need individual cards")),
            _ => {}
        }
    }
    result.textures = w.textures;
    Ok(())
}

/// Decals: albedo + opacity composed into one RGBA texture on a Decal Projector prefab.
fn import_decal(job: &Job<'_>, w: &mut Writer<'_>, result: &mut AssetResult) -> Result<()> {
    use crate::scene_file::{ComponentData, MaterialRef, ObjectSource, SceneEntry, SceneFile};
    let Some(set) = job.asset.sets.iter().find(|s| !s.billboard && s.textures.iter().any(|t| t.role == TexRole::Albedo)) else {
        result.lines.push(note(Level::Unsupported, "decal", "no albedo texture: no Decal Projector written"));
        return Ok(());
    };
    let cap = job.opts.max_texture_size;
    let albedo = pick_best(set, &[TexRole::Albedo], cap).expect("checked above");
    let opacity = pick_best(set, &[TexRole::Opacity], cap);
    let color = load_rgba(&albedo.path)?;
    let size = capped_size(color.dimensions(), cap);
    let color = fit(color, size);
    let alpha = opacity.map(|o| load_rgba(&o.path).map(|i| fit(i, size))).transpose()?;
    let img = compose_rgba(&color, alpha.as_ref());
    let tex = w.write_png(&format!("{}_decal", job.asset.name), &img, TexSlot::ColorAlpha)?;
    let dims = job.asset.size_m.unwrap_or([1.0, 1.0]);
    let projector = zestengine_core::DecalProjector { texture: rel(job.project_root, &tex), size: dims, depth: (dims[0].min(dims[1]) * 0.5).max(0.1), ..Default::default() };
    let entry = SceneEntry {
        id: zestengine_core::ObjectId::new().to_string(),
        name: job.asset.name.clone(),
        source: ObjectSource::Empty,
        enabled: true,
        static_geometry: false,
        lightmap_scale: 1.0,
        cast_shadows: zestengine_core::ShadowCasting::On,
        layer: 0,
        material: MaterialRef::Model,
        extra_materials: Vec::new(),
        renderer_split: false,
        parent: None,
        components: vec![ComponentData::of(&projector)],
        translation: [0.0; 3],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0; 3],
        model: None,
        model_part: None,
        prefab: None,
        prefab_parent: None,
        net_role: Default::default(),
        component_roles: Default::default(),
    };
    let path = w.asset_dir.join(format!("{}.prefab", crate::material_remap::file_stem_of(&job.asset.name, "decal")));
    crate::save_scene_file(&path, &SceneFile { entries: vec![entry], ..Default::default() })?;
    result.prefabs.push(path);
    result.lines.push(note(Level::Mapped, "decal", format!("albedo{} composed into one RGBA texture on a Decal Projector ({} x {} m)", if opacity.is_some() { " + opacity" } else { "" }, dims[0], dims[1])));
    let dropped: Vec<&str> = set.textures.iter().filter(|t| matches!(t.role, TexRole::Normal | TexRole::NormalDx | TexRole::Roughness | TexRole::Gloss | TexRole::Ao | TexRole::Height)).map(|_| "map").collect();
    if !dropped.is_empty() {
        result.lines.push(note(Level::Unsupported, "decal maps", "the Decal Projector projects one colour texture: normal, roughness, occlusion and height maps of the decal are not projected"));
    }
    Ok(())
}

/// Import every included asset of a scan. `progress(done, total, current)`
/// returns false to cancel (finished assets stay).
pub fn import_assets(assets: &[PackAsset], project_root: &Path, opts: &ImportOptions, progress: &mut dyn FnMut(usize, usize, &str) -> bool) -> ImportOutcome {
    let chosen: Vec<&PackAsset> = assets.iter().filter(|a| a.include).collect();
    let mut out = ImportOutcome::default();
    // Two assets with one name in one run get distinct folders even under Overwrite.
    let mut used: BTreeMap<String, usize> = BTreeMap::new();
    for (i, a) in chosen.iter().enumerate() {
        if !progress(i, chosen.len(), &a.name) {
            out.cancelled = true;
            break;
        }
        let mut asset = (*a).clone();
        let n = used.entry(asset.name.to_ascii_lowercase()).or_insert(0);
        *n += 1;
        if *n > 1 {
            asset.name = format!("{}_{}", asset.name, n);
        }
        let name = asset.name.clone();
        let mut inner = |_: &str| progress(i, chosen.len(), &name);
        let r = import_asset(&asset, project_root, opts, &mut inner);
        let was_cancelled = r.error.as_deref().is_some_and(|e| e.contains("cancelled"));
        out.assets.push(r);
        if was_cancelled {
            out.cancelled = true;
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack_scan::scan_pack;
    use crate::pack_zip::tests::scratch;
    use crate::tex_roles::Confidence;

    fn tex(name: &str, role: TexRole, res: Option<u32>) -> SetTexture {
        SetTexture { path: PathBuf::from(name), role, confidence: Confidence::Known, resolution: res, bytes: 1 }
    }

    fn set(textures: Vec<SetTexture>) -> TextureSet {
        TextureSet { name: "rock".into(), billboard: false, textures }
    }

    fn ctx(kind: AssetKind) -> PlanCtx {
        PlanCtx { kind, size_m: None, max_texture_size: 4096, displacement_scale: 0.03 }
    }

    fn has(p: &MatPlan, level: Level, prop: &str) -> bool {
        p.notes.iter().any(|n| n.level == level && n.property == prop)
    }

    #[test]
    fn channel_remap_maths() {
        // 1x1 images make the channel picks exact.
        let px = |v: [u8; 4]| RgbaImage::from_pixel(1, 1, image::Rgba(v));
        let mr = px([10, 120, 200, 30]); // glTF metal-rough: G rough, B metal
        let ao = px([77, 0, 0, 255]);
        let out = pack_channels(
            [Some(ChanFrom { img: &ao, pick: Pick { src: Chan::R, invert: false } }), Some(ChanFrom { img: &mr, pick: Pick { src: Chan::G, invert: false } }), Some(ChanFrom { img: &mr, pick: Pick { src: Chan::B, invert: false } })],
            (1, 1),
        );
        assert_eq!(out.get_pixel(0, 0).0, [77, 120, 200, 255]);
        // Unity metal-gloss: A smoothness inverted to roughness, R metallic; no occlusion -> 255.
        let mg = px([200, 0, 0, 55]);
        let l = orm_layout(TexRole::MetalGloss).unwrap();
        let out = pack_channels([None, l[1].map(|pick| ChanFrom { img: &mg, pick }), l[2].map(|pick| ChanFrom { img: &mg, pick })], (1, 1));
        assert_eq!(out.get_pixel(0, 0).0, [255, 200, 200, 255]);
        // Nothing at all: no occlusion, rough, dielectric.
        assert_eq!(pack_channels([None, None, None], (1, 1)).get_pixel(0, 0).0, [255, 255, 0, 255]);
        // Mismatched sizes sample by scaled coordinates.
        let big = RgbaImage::from_fn(4, 4, |x, _| image::Rgba([x as u8 * 10, 0, 0, 255]));
        let out = pack_channels([Some(ChanFrom { img: &big, pick: Pick { src: Chan::R, invert: false } }), None, None], (2, 2));
        assert_eq!((out.get_pixel(0, 0).0[0], out.get_pixel(1, 0).0[0]), (0, 20));
    }

    #[test]
    fn normal_flip_and_decal_compose() {
        let mut n = RgbaImage::from_pixel(1, 1, image::Rgba([128, 200, 255, 255]));
        flip_normal_green(&mut n);
        assert_eq!(n.get_pixel(0, 0).0, [128, 55, 255, 255]);
        let color = RgbaImage::from_pixel(2, 2, image::Rgba([9, 9, 9, 255]));
        let op = RgbaImage::from_pixel(1, 1, image::Rgba([64, 0, 0, 255]));
        assert_eq!(compose_rgba(&color, Some(&op)).get_pixel(1, 1).0, [9, 9, 9, 64]);
        assert_eq!(compose_rgba(&color, None).get_pixel(0, 0).0[3], 255);
    }

    #[test]
    fn sizes_tiling_and_lod_thresholds() {
        assert_eq!(capped_size((8192, 4096), 4096), (4096, 2048));
        assert_eq!(capped_size((1000, 500), 4096), (1000, 500));
        assert_eq!(capped_size((8192, 8192), 0), (8192, 8192));
        assert_eq!(tiling_for_size(Some([2.0, 4.0])), [0.5, 0.25]);
        assert_eq!(tiling_for_size(None), [1.0, 1.0]);
        let s = lod_screen_sizes(4);
        assert_eq!(s.len(), 4);
        assert!(s.windows(2).all(|w| w[0] > w[1]), "descending: {s:?}");
        assert_eq!(*s.last().unwrap(), 0.0);
        assert_eq!(lod_screen_sizes(1), vec![0.0]);
    }

    #[test]
    fn megascans_gloss_and_normal_and_cutout_foliage() {
        let s = set(vec![
            tex("a_8K_Albedo.jpg", TexRole::Albedo, Some(8192)),
            tex("a_8K_Normal.jpg", TexRole::Normal, Some(8192)),
            tex("a_8K_Gloss.jpg", TexRole::Gloss, Some(8192)),
            tex("a_8K_Opacity.jpg", TexRole::Opacity, Some(8192)),
            tex("a_8K_Translucency.jpg", TexRole::Translucency, Some(8192)),
            tex("a_8K_Specular.jpg", TexRole::Specular, Some(8192)),
            tex("a_8K_Cavity.jpg", TexRole::Cavity, Some(8192)),
        ]);
        let p = plan_material(&s, &ctx(AssetKind::Plant));
        assert_eq!(p.normal.as_ref().map(|n| n.1), Some(false), "Megascans normals are OpenGL: not flipped");
        assert!(p.params.roughness_invert, "gloss alone is the roughness slot inverted");
        assert_eq!(p.roughness, Some(PathBuf::from("a_8K_Gloss.jpg")));
        assert_eq!(p.features.alpha_mode, AlphaMode::Cutout);
        assert!(p.features.double_sided && p.features.normal_map);
        assert!(has(&p, Level::Approx, "translucency"), "the Megascans translucency map now maps");
        assert_eq!(p.translucency, Some(PathBuf::from("a_8K_Translucency.jpg")));
        assert_eq!(p.params.transmission.translucency, 1.0);
        assert!(has(&p, Level::Info, "specular") && has(&p, Level::Info, "cavity"));
    }

    #[test]
    fn roughness_beats_gloss_and_dx_normals_flip() {
        let s = set(vec![
            tex("r.png", TexRole::Roughness, None),
            tex("g.png", TexRole::Gloss, None),
            tex("n.png", TexRole::NormalDx, None),
            tex("m.png", TexRole::Metallic, None),
            tex("o.png", TexRole::Ao, None),
        ]);
        let p = plan_material(&s, &ctx(AssetKind::Model3d));
        assert_eq!(p.roughness, Some(PathBuf::from("r.png")));
        assert!(!p.params.roughness_invert);
        assert!(has(&p, Level::Info, "gloss"));
        assert_eq!(p.normal, Some((PathBuf::from("n.png"), true)));
        assert_eq!((p.params.metallic, p.params.roughness), (1.0, 1.0));
        assert_eq!(p.metallic, Some(PathBuf::from("m.png")));
        assert_eq!(p.ao, Some(PathBuf::from("o.png")));
        // No maps at all: dielectric, mid-rough.
        let bare = plan_material(&set(vec![tex("a.png", TexRole::Albedo, None)]), &ctx(AssetKind::Model3d));
        assert_eq!((bare.params.metallic, bare.params.roughness), (0.0, 0.8));
    }

    #[test]
    fn packed_orm_is_direct_and_other_layouts_repack() {
        let direct = plan_material(&set(vec![tex("x_ORM.png", TexRole::Orm, None), tex("x_AO.png", TexRole::Ao, None)]), &ctx(AssetKind::Model3d));
        assert_eq!(direct.orm, OrmPlan::Direct(PathBuf::from("x_ORM.png")));
        assert!(direct.ao.is_none(), "the separate AO is ignored");
        let mr = plan_material(&set(vec![tex("x_mr.png", TexRole::MetalRough, None), tex("x_ao.png", TexRole::Ao, None)]), &ctx(AssetKind::Model3d));
        let OrmPlan::Repack(ch) = &mr.orm else { panic!("expected repack, got {:?}", mr.orm) };
        assert_eq!(ch[0], Some((PathBuf::from("x_ao.png"), Pick { src: Chan::R, invert: false })), "glTF has no occlusion: the separate map fills it");
        assert_eq!(ch[1], Some((PathBuf::from("x_mr.png"), Pick { src: Chan::G, invert: false })));
        assert_eq!(ch[2], Some((PathBuf::from("x_mr.png"), Pick { src: Chan::B, invert: false })));
        assert!(has(&mr, Level::Approx, "orm"));
    }

    #[test]
    fn surface_tiling_displacement_and_bump_fallback() {
        let s = set(vec![tex("a.jpg", TexRole::Albedo, None), tex("d.jpg", TexRole::Height, None), tex("b.jpg", TexRole::Bump, None)]);
        let mut c = ctx(AssetKind::Surface);
        c.size_m = Some([2.0, 2.0]);
        let p = plan_material(&s, &c);
        assert_eq!(p.params.albedo_tiling, [0.5, 0.5]);
        assert_eq!(p.params.normal_tiling, [0.5, 0.5]);
        assert_eq!(p.params.orm_tiling, [0.5, 0.5]);
        assert_eq!(p.displacement, Some(PathBuf::from("d.jpg")));
        assert_eq!(p.params.displacement_scale, 0.03);
        assert!(has(&p, Level::Info, "bump"));
        // Bump alone stands in on a surface, not on a 3D asset.
        let b = set(vec![tex("a.jpg", TexRole::Albedo, None), tex("b.jpg", TexRole::Bump, None)]);
        assert_eq!(plan_material(&b, &c).displacement, Some(PathBuf::from("b.jpg")));
        assert_eq!(plan_material(&b, &ctx(AssetKind::Model3d)).displacement, None);
        // Unknown size: tiling 1 and a note.
        let u = plan_material(&s, &ctx(AssetKind::Surface));
        assert_eq!(u.params.albedo_tiling, [1.0, 1.0]);
        assert!(has(&u, Level::Info, "tiling"));
        // 3D assets never tile.
        let mut m = ctx(AssetKind::Model3d);
        m.size_m = Some([2.0, 2.0]);
        assert_eq!(plan_material(&s, &m).params.albedo_tiling, [1.0, 1.0]);
    }

    #[test]
    fn resolution_choice_respects_the_cap() {
        let s = set(vec![tex("a_1k.png", TexRole::Albedo, Some(1024)), tex("a_4k.png", TexRole::Albedo, Some(4096)), tex("a_8k.png", TexRole::Albedo, Some(8192))]);
        let mut c = ctx(AssetKind::Model3d);
        assert_eq!(plan_material(&s, &c).albedo, Some(PathBuf::from("a_4k.png")));
        c.max_texture_size = 2048;
        assert_eq!(plan_material(&s, &c).albedo, Some(PathBuf::from("a_1k.png")));
        c.max_texture_size = 0;
        assert_eq!(plan_material(&s, &c).albedo, Some(PathBuf::from("a_8k.png")));
        let only_big = set(vec![tex("a_8k.png", TexRole::Albedo, Some(8192)), tex("a_16k.png", TexRole::Albedo, Some(16384))]);
        c.max_texture_size = 4096;
        assert_eq!(plan_material(&only_big, &c).albedo, Some(PathBuf::from("a_8k.png")), "smallest above the cap");
    }

    #[test]
    fn material_names_find_their_sets() {
        let sets = vec!["Armor".to_owned(), "Body".to_owned()];
        let names = vec!["M_Armor".to_owned(), "body_mat".to_owned(), "Mystery".to_owned()];
        assert_eq!(assign_sets(&names, &sets), vec![Some(0), Some(1), None]);
        assert_eq!(assign_sets(&names, &["only".to_owned()]), vec![Some(0), Some(0), Some(0)]);
        assert!(assign_sets(&names, &[]).iter().all(Option::is_none));
    }

    #[test]
    fn texture_import_settings_per_slot() {
        use crate::{Compression, TextureKind};
        let a = texture_import_for(TexSlot::Albedo, 4096);
        assert!(a.srgb && a.kind == TextureKind::Default && a.mips.enabled && a.max_size == 4096);
        let n = texture_import_for(TexSlot::Normal { flip_green: true }, 2048);
        assert!(n.kind == TextureKind::NormalMap && n.normal.flip_green && !n.srgb && n.max_size == 2048);
        assert!(!texture_import_for(TexSlot::Normal { flip_green: false }, 0).normal.flip_green);
        let s = texture_import_for(TexSlot::Single, 4096);
        assert!(s.kind == TextureKind::SingleChannel && !s.srgb);
        let o = texture_import_for(TexSlot::Opacity, 4096);
        assert!(o.mips.preserve_coverage && o.kind == TextureKind::SingleChannel);
        assert_eq!(texture_import_for(TexSlot::Height, 4096).compression, Compression::High);
        let d = texture_import_for(TexSlot::Orm, 4096);
        assert!(!d.srgb && d.kind == TextureKind::Default);
        assert!(texture_import_for(TexSlot::ColorAlpha, 4096).alpha_is_transparency);
        assert!(texture_import_for(TexSlot::Emission, 4096).srgb);
    }

    fn write_png(path: &Path, w: u32, h: u32, px: [u8; 4]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        RgbaImage::from_pixel(w, h, image::Rgba(px)).save(path).unwrap();
    }

    #[test]
    fn surface_round_trip_writes_a_loadable_material() {
        let src = scratch("rt_src");
        let proj = scratch("rt_proj");
        write_png(&src.join("udlladln/udlladln_4K_Albedo.png"), 8, 8, [200, 100, 50, 255]);
        write_png(&src.join("udlladln/udlladln_4K_Normal.png"), 8, 8, [128, 128, 255, 255]);
        write_png(&src.join("udlladln/udlladln_4K_Gloss.png"), 8, 8, [60, 60, 60, 255]);
        write_png(&src.join("udlladln/udlladln_4K_Displacement.png"), 8, 8, [90, 90, 90, 255]);
        std::fs::write(src.join("udlladln/udlladln.json"), r#"{"id":"udlladln","semanticTags":{"asset_type":"surface","name":"Slate"},"meta":[{"key":"scanArea","value":"2x2"}]}"#).unwrap();
        let scan = scan_pack(&src).unwrap();
        assert_eq!(scan.assets.len(), 1);
        let opts = ImportOptions::default();
        let out = import_assets(&scan.assets, &proj, &opts, &mut |_, _, _| true);
        let r = &out.assets[0];
        assert!(r.error.is_none(), "{:?}", r.error);
        assert_eq!(r.materials.len(), 1);
        let m = crate::load_material_file(&r.materials[0]).unwrap();
        assert_eq!(m.params.albedo_tiling, [0.5, 0.5]);
        assert!(m.params.roughness_invert && m.features.normal_map);
        assert_eq!(m.params.displacement_scale, 0.03);
        let normal = proj.join(m.textures.normal.clone().unwrap());
        assert!(normal.is_file());
        // The texture metas carry the per-map settings and the materials resolve inside the project.
        let nm = crate::load_asset_meta(&normal).unwrap().unwrap();
        assert_eq!(nm.importer.as_texture().unwrap().kind, crate::TextureKind::NormalMap);
        let am = crate::load_asset_meta(proj.join(m.textures.albedo.clone().unwrap())).unwrap().unwrap();
        assert!(am.importer.as_texture().unwrap().srgb);
        assert!(r.dir.join("import_report.txt").is_file());
        assert!(r.dir.starts_with(proj.join("megascans")));
        let _ = std::fs::remove_dir_all(&src);
        let _ = std::fs::remove_dir_all(&proj);
    }

    #[test]
    fn duplicate_policies() {
        let src = scratch("dup_src");
        let proj = scratch("dup_proj");
        write_png(&src.join("rock/rock_Albedo.png"), 4, 4, [1, 2, 3, 255]);
        std::fs::write(src.join("rock/rock.fbx"), "not a real fbx").unwrap();
        let scan = scan_pack(&src).unwrap();
        let mut opts = ImportOptions { write_prefabs: false, ..Default::default() };
        let first = import_assets(&scan.assets, &proj, &opts, &mut |_, _, _| true);
        let dir1 = first.assets[0].dir.clone();
        let tex1 = std::fs::read_dir(dir1.join("textures")).unwrap().count();
        // Keep both: a second folder, the first untouched.
        let second = import_assets(&scan.assets, &proj, &opts, &mut |_, _, _| true);
        assert_ne!(second.assets[0].dir, dir1);
        assert!(second.assets[0].dir.is_dir());
        // Skip: nothing written.
        opts.duplicates = DuplicatePolicy::Skip;
        let third = import_assets(&scan.assets, &proj, &opts, &mut |_, _, _| true);
        assert!(third.assets[0].skipped);
        // Overwrite: same folder, texture meta id preserved.
        opts.duplicates = DuplicatePolicy::Overwrite;
        let tex_path = std::fs::read_dir(dir1.join("textures")).unwrap().flatten().map(|e| e.path()).find(|p| p.extension().is_some_and(|e| e == "png")).unwrap();
        let id_before = crate::load_asset_meta(&tex_path).unwrap().unwrap().id;
        let fourth = import_assets(&scan.assets, &proj, &opts, &mut |_, _, _| true);
        assert_eq!(fourth.assets[0].dir, dir1);
        assert_eq!(crate::load_asset_meta(&tex_path).unwrap().unwrap().id, id_before);
        assert_eq!(std::fs::read_dir(dir1.join("textures")).unwrap().count(), tex1, "no renamed copies");
        // Two assets of one name in one run: distinct folders even when overwriting.
        let mut twice = scan.assets.clone();
        twice.push(scan.assets[0].clone());
        let both = import_assets(&twice, &proj, &opts, &mut |_, _, _| true);
        assert_ne!(both.assets[0].dir, both.assets[1].dir);
        let _ = std::fs::remove_dir_all(&src);
        let _ = std::fs::remove_dir_all(&proj);
    }

    #[test]
    fn cancel_stops_the_batch() {
        let src = scratch("cancel_src");
        let proj = scratch("cancel_proj");
        write_png(&src.join("aaaaaa/aaaaaa_2K_Albedo.png"), 2, 2, [1, 1, 1, 255]);
        write_png(&src.join("bbbbbb/bbbbbb_2K_Albedo.png"), 2, 2, [1, 1, 1, 255]);
        let scan = scan_pack(&src).unwrap();
        let out = import_assets(&scan.assets, &proj, &ImportOptions::default(), &mut |done, _, _| done == 0);
        assert!(out.cancelled);
        assert_eq!(out.assets.len(), 1);
        let _ = std::fs::remove_dir_all(&src);
        let _ = std::fs::remove_dir_all(&proj);
    }

    #[test]
    fn decal_imports_one_rgba_texture_and_a_projector_prefab() {
        let src = scratch("decal_src");
        let proj = scratch("decal_proj");
        write_png(&src.join("dec12345/dec12345_2K_Albedo.png"), 4, 4, [10, 20, 30, 255]);
        write_png(&src.join("dec12345/dec12345_2K_Opacity.png"), 4, 4, [128, 0, 0, 255]);
        write_png(&src.join("dec12345/dec12345_2K_Normal.png"), 4, 4, [128, 128, 255, 255]);
        std::fs::write(src.join("dec12345/dec12345.json"), r#"{"id":"dec12345","semanticTags":{"asset_type":"decal","name":"Crack"},"meta":[{"key":"scanArea","value":"0.5x0.5"}]}"#).unwrap();
        let scan = scan_pack(&src).unwrap();
        let out = import_assets(&scan.assets, &proj, &ImportOptions::default(), &mut |_, _, _| true);
        let r = &out.assets[0];
        assert!(r.error.is_none(), "{:?}", r.error);
        assert_eq!(r.prefabs.len(), 1);
        let scene = crate::load_scene_file(&r.prefabs[0]).unwrap();
        let reg = {
            let mut reg = zestengine_core::ComponentRegistry::default();
            reg.register::<zestengine_core::DecalProjector>();
            reg
        };
        let comp = scene.entries[0].components[0].load(&reg).unwrap();
        let d = comp.as_any().downcast_ref::<zestengine_core::DecalProjector>().unwrap();
        assert_eq!(d.size, [0.5, 0.5]);
        let img = image::open(proj.join(&d.texture)).unwrap().into_rgba8();
        assert_eq!(img.get_pixel(0, 0).0, [10, 20, 30, 128]);
        assert!(r.lines.iter().any(|l| l.level == Level::Unsupported && l.property == "decal maps"));
        let _ = std::fs::remove_dir_all(&src);
        let _ = std::fs::remove_dir_all(&proj);
    }

    #[test]
    fn provided_lods_become_a_lod_group_in_the_prefab() {
        let src = scratch("lod_src");
        let proj = scratch("lod_proj");
        // One triangle with an embedded buffer.
        let gltf = r#"{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0,"name":"tri"}],"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]}],"bufferViews":[{"buffer":0,"byteLength":36}],"buffers":[{"byteLength":36,"uri":"data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAA"}]}"#;
        for lod in 0..3 {
            std::fs::create_dir_all(src.join("tri")).unwrap();
            std::fs::write(src.join(format!("tri/tri_LOD{lod}.gltf")), gltf).unwrap();
        }
        write_png(&src.join("tri/tri_Albedo.png"), 2, 2, [200, 0, 0, 255]);
        let scan = scan_pack(&src).unwrap();
        assert_eq!(scan.assets[0].variants[0].lods.len(), 3);
        let out = import_assets(&scan.assets, &proj, &ImportOptions::default(), &mut |_, _, _| true);
        let r = &out.assets[0];
        assert!(r.error.is_none(), "{:?}", r.error);
        assert_eq!(r.models.len(), 3);
        assert_eq!(r.prefabs.len(), 1, "{}", r.to_text());
        let prefab = crate::load_scene_file(&r.prefabs[0]).unwrap();
        let reg = {
            let mut reg = zestengine_core::ComponentRegistry::default();
            reg.register::<zestengine_core::LodComponent>();
            reg
        };
        let part = prefab.entries.iter().find(|e| e.components.iter().any(|c| c.kind == "LOD Group")).expect("part with a LOD Group");
        let comp = part.components.iter().find(|c| c.kind == "LOD Group").unwrap().load(&reg).unwrap();
        let lod = comp.as_any().downcast_ref::<zestengine_core::LodComponent>().unwrap();
        assert_eq!(lod.levels.len(), 3);
        assert!(lod.by_screen_size);
        assert!(lod.levels[0].model.is_empty(), "level 0 is the base mesh");
        assert!(lod.levels[1].model.ends_with("tri_LOD1.gltf") && lod.levels[2].model.ends_with("tri_LOD2.gltf"), "{:?}", lod.levels.iter().map(|l| &l.model).collect::<Vec<_>>());
        assert!(lod.levels[0].screen_size > lod.levels[1].screen_size && lod.levels[2].screen_size == 0.0);
        assert!(matches!(crate::renderer_migration::material_refs(part).first(), Some(crate::scene_file::MaterialRef::File(p)) if p.ends_with(".material")));
        // The model metas carry the remap for the source material.
        let m0 = crate::load_asset_meta(&r.models[0]).unwrap().unwrap();
        assert!(m0.importer.as_model().is_some());
        let _ = std::fs::remove_dir_all(&src);
        let _ = std::fs::remove_dir_all(&proj);
    }

    /// Real Megascans FBX from the author's Fab downloads: the LOD chain, materials and prefab.
    #[test]
    fn real_megascans_model_imports_with_prefab() {
        let root = Path::new("/home/lemonxah/Downloads/fab/south_african_slate_quarry_raw/xb5ebhf");
        if !root.is_dir() {
            return;
        }
        // Copy only the (small) textures' stand-ins: the 61 MB FBX is read in place by the scan,
        // so stage a copy of just the model with tiny textures beside it.
        let src = scratch("real_src");
        let proj = scratch("real_proj");
        std::fs::create_dir_all(src.join("xb5ebhf")).unwrap();
        std::fs::copy(root.join("xb5ebhf.fbx"), src.join("xb5ebhf/xb5ebhf.fbx")).unwrap();
        for role in ["BaseColor", "Normal", "Roughness", "AO"] {
            write_png(&src.join(format!("xb5ebhf/xb5ebhf_8K_{role}.png")), 4, 4, [120, 120, 120, 255]);
        }
        let scan = scan_pack(&src).unwrap();
        let out = import_assets(&scan.assets, &proj, &ImportOptions::default(), &mut |_, _, _| true);
        let r = &out.assets[0];
        assert!(r.error.is_none(), "{:?}\n{}", r.error, r.to_text());
        assert_eq!(r.prefabs.len(), 1, "{}", r.to_text());
        let prefab = crate::load_scene_file(&r.prefabs[0]).unwrap();
        assert!(prefab.entries.len() >= 2);
        assert!(prefab.entries.iter().skip(1).all(|e| matches!(crate::renderer_migration::material_refs(e).first(), Some(crate::scene_file::MaterialRef::File(p)) if p.ends_with(".material"))), "{:?}", prefab.entries.iter().map(crate::renderer_migration::material_refs).collect::<Vec<_>>());
        let _ = std::fs::remove_dir_all(&src);
        let _ = std::fs::remove_dir_all(&proj);
    }
}
