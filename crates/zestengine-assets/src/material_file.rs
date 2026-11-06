//! `.material` asset files (RON).
//!
//! A material file stores shader parameters, feature toggles, and
//! project-relative texture paths. The `shader` field selects which shader
//! the material targets: "standard", or the project-relative path of a
//! custom `.frag` shader (see shader_file.rs); custom property values are
//! stored by name in `custom`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use zestengine_render::{MaterialFeatures, MaterialParams, TextureData};
use serde::{Deserialize, Serialize};

pub const MATERIAL_EXTENSION: &str = "material";

fn default_shader() -> String {
    "standard".into()
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MaterialTextures {
    pub albedo: Option<PathBuf>,
    pub normal: Option<PathBuf>,
    pub orm: Option<PathBuf>,
    pub emission: Option<PathBuf>,
    // Extended slots (Poiyomi-style). `#[serde(default)]` keeps old files loading.
    pub opacity: Option<PathBuf>,
    pub emission_mask: Option<PathBuf>,
    pub matcap: [Option<PathBuf>; 3],
    pub matcap_mask: [Option<PathBuf>; 3],
    /// Split AO / Roughness / Metallic maps (alternative to the packed ORM).
    pub ao: Option<PathBuf>,
    pub roughness: Option<PathBuf>,
    pub metallic: Option<PathBuf>,
    /// Height / displacement map (parallax occlusion mapping).
    pub displacement: Option<PathBuf>,
    /// Detail layer maps (sRGB albedo, linear normal / ORM / mask).
    pub detail_albedo: Option<PathBuf>,
    pub detail_normal: Option<PathBuf>,
    pub detail_orm: Option<PathBuf>,
    pub detail_mask: Option<PathBuf>,
    /// Translucency (or thickness, see `TransmissionParams::map_is_thickness`) map.
    pub translucency: Option<PathBuf>,
    /// Custom-shader texture properties by property name.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub shader: BTreeMap<String, PathBuf>,
    /// Texture-slot import warnings the user chose to Ignore: (slot index,
    /// project-relative texture path). Tied to the texture so assigning a
    /// different one brings the warning back.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ignored_warnings: Vec<(usize, PathBuf)>,
}

impl MaterialTextures {
    /// True when no slot is assigned (used to skip serializing empty sets).
    pub fn is_empty(&self) -> bool {
        self.albedo.is_none()
            && self.normal.is_none()
            && self.orm.is_none()
            && self.emission.is_none()
            && self.opacity.is_none()
            && self.emission_mask.is_none()
            && self.matcap.iter().all(Option::is_none)
            && self.matcap_mask.iter().all(Option::is_none)
            && self.ao.is_none()
            && self.roughness.is_none()
            && self.metallic.is_none()
            && self.displacement.is_none()
            && self.detail_albedo.is_none()
            && self.detail_normal.is_none()
            && self.detail_orm.is_none()
            && self.detail_mask.is_none()
            && self.translucency.is_none()
            && self.shader.is_empty()
            && self.ignored_warnings.is_empty()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaterialFile {
    pub name: String,
    /// "standard" or a project-relative custom `.frag` shader path.
    #[serde(default = "default_shader")]
    pub shader: String,
    pub params: MaterialParams,
    pub features: MaterialFeatures,
    /// Draw-order priority (Unity render queue). `None` = derive from the
    /// alpha mode on load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub render_queue: Option<i32>,
    #[serde(default)]
    pub textures: MaterialTextures,
    /// Custom-shader property values by property name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub custom: BTreeMap<String, Vec<f32>>,
}

pub fn load_material_file(path: impl AsRef<Path>) -> Result<MaterialFile> {
    let path = path.as_ref();
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let material: MaterialFile = ron::from_str(&text)
        .with_context(|| format!("parsing material file {}", path.display()))?;
    Ok(material)
}

pub fn save_material_file(path: impl AsRef<Path>, material: &MaterialFile) -> Result<()> {
    let path = path.as_ref();
    let text = ron::ser::to_string_pretty(material, ron::ser::PrettyConfig::default())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Load an image file from disk as texture data. EXR/HDR sources load NATIVELY
/// as linear RGBA f16 (no LDR clamp, no PNG conversion); everything else loads
/// as RGBA8.
pub fn load_texture_file(path: impl AsRef<Path>, srgb: bool) -> Result<TextureData> {
    let path = path.as_ref();
    // EXR and Radiance HDR both carry linear float radiance.
    let is_float = path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("exr") || e.eq_ignore_ascii_case("hdr"));
    let dynimg =
        open_image_compat(path).with_context(|| format!("loading texture {}", path.display()))?;
    if is_float {
        // Native HDR float: keep the full linear float range as RGBA f16 — no
        // clamping to LDR, no PNG round-trip.
        let img = dynimg.into_rgba32f();
        let (w, h) = (img.width(), img.height());
        let mut pixels = Vec::with_capacity((w as usize) * (h as usize) * 8);
        for p in img.pixels() {
            for c in 0..4 {
                pixels.extend_from_slice(&half::f16::from_f32(p.0[c]).to_le_bytes());
            }
        }
        Ok(TextureData { width: w, height: h, pixels, srgb: false, hdr: true })
    } else {
        let img = dynimg.into_rgba8();
        Ok(TextureData {
            width: img.width(),
            height: img.height(),
            pixels: img.into_raw(),
            srgb,
            hdr: false,
        })
    }
}

/// Decode an image, with a fallback for EXR files using compressions the pure-
/// Rust `exr` decoder doesn't implement (notably DWAA/DWAB). When the native
/// decode fails on an `.exr`, transcode it to a temporary lossless-ZIP EXR via
/// `oiiotool` (OpenImageIO) if present, then decode that. Other formats and
/// supported EXR compressions take the fast direct path.
pub fn open_image_compat(path: &Path) -> Result<image::DynamicImage> {
    match image::open(path) {
        Ok(img) => Ok(img),
        Err(err) => {
            let is_exr = path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("exr"));
            if is_exr {
                if let Some(img) = transcode_exr(path) {
                    return Ok(img);
                }
                tracing::warn!(
                    "EXR '{}' couldn't be decoded (unsupported compression like DWAA, or a \
                     single-channel/deep layout); install OpenImageIO (`oiiotool`) for a \
                     fallback, or re-export as RGB with ZIP/PIZ compression",
                    path.display()
                );
            }
            Err(err.into())
        }
    }
}

/// Transcode an EXR the Rust `exr` decoder can't read (unsupported compression
/// like DWAA/DWAB, or a single-channel/non-RGB layout) to a temp lossless **ZIP
/// EXR** via `oiiotool`, then decode that. Stays FLOAT end-to-end (no PNG/LDR
/// round-trip): the source's real channels are queried first, then a `--ch`
/// expression maps them to RGBA referencing ONLY existing channels and filling
/// the rest with constants — so an RGB source (no A) or a single-channel "Y"
/// source transcode WITHOUT oiiotool's "Unknown channel … filling with 0"
/// warnings. `--compression zip` is what the Rust decoder supports. Returns
/// `None` if `oiiotool` is missing or the conversion/decode fails.
fn transcode_exr(path: &Path) -> Option<image::DynamicImage> {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    path.hash(&mut hasher);
    let mut tmp = std::env::temp_dir();
    tmp.push(format!("zestengine_exr_{:016x}.exr", hasher.finish()));

    // Query the source's actual channel names (e.g. "R,G,B", "R,G,B,A", "Y") by
    // parsing oiiotool's `--info -v` "channel list:" line (the `{TOP.channelnames}`
    // echo expression isn't evaluated by all oiiotool builds).
    let names: Vec<String> = std::process::Command::new("oiiotool")
        .args(["--info", "-v"])
        .arg(path)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| {
            let text = String::from_utf8_lossy(&o.stdout).into_owned();
            text.lines()
                .find_map(|l| l.split_once("channel list:"))
                .map(|(_, list)| {
                    list.split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect()
                })
        })
        .unwrap_or_default();
    let has = |n: &str| names.iter().any(|c| c == n);
    // Build a channel expression that references only channels that exist:
    //   RGB(A): keep R,G,B; A from source or constant 1.0.
    //   single channel (Y / luminance / lone R): replicate it to RGB, A = 1.0.
    let ch = if has("R") && has("G") && has("B") {
        if has("A") {
            "R,G,B,A".to_string()
        } else {
            "R,G,B,A=1.0".to_string()
        }
    } else if let Some(first) = names.first() {
        format!("R={first},G={first},B={first},A=1.0")
    } else {
        // Channel query failed (oiiotool missing/old): fall back to the plain
        // RGBA request (may warn, but still fills missing channels).
        "R=R,G=G,B=B,A=A".to_string()
    };

    let ok = std::process::Command::new("oiiotool")
        .arg(path)
        .args(["--ch", &ch, "--compression", "zip", "-o"])
        .arg(&tmp)
        .status()
        .ok()
        .is_some_and(|s| s.success())
        // Last-resort retry: replicate channel 0 to RGB (handles odd layouts).
        || std::process::Command::new("oiiotool")
            .arg(path)
            .args(["--ch", "0,0,0", "--compression", "zip", "-o"])
            .arg(&tmp)
            .status()
            .ok()
            .is_some_and(|s| s.success());
    let img = if ok { image::open(&tmp).ok() } else { None };
    let _ = std::fs::remove_file(&tmp);
    img
}

#[cfg(test)]
mod stencil_tests {
    use super::*;
    use zestengine_render::{StencilCompare, StencilOp, StencilState};

    /// A material written before `StencilState` existed (no `stencil` key in the
    /// features) loads with the inactive default.
    #[test]
    fn old_material_files_load_without_stencil() {
        let old = r#"(name: "Old", params: (), features: (toon: false, normal_map: false, emission: false, alpha_mode: Opaque, double_sided: false))"#;
        let m: MaterialFile = ron::from_str(old).expect("old file loads");
        assert_eq!(m.features.stencil, StencilState::default());
        assert!(!m.features.stencil.is_active());
    }

    #[test]
    fn stencil_state_round_trips_through_ron() {
        let mut f = MaterialFeatures::default();
        f.stencil.reference = 5;
        f.stencil.front.compare = StencilCompare::NotEqual;
        f.stencil.front.pass_op = StencilOp::IncrementWrap;
        f.stencil.back.zfail_op = StencilOp::Invert;
        f.stencil.color_mask = 0;
        let file = MaterialFile {
            name: "S".into(),
            shader: "standard".into(),
            params: MaterialParams::default(),
            features: f,
            render_queue: Some(1999),
            textures: MaterialTextures::default(),
            custom: Default::default(),
        };
        let text = ron::ser::to_string_pretty(&file, ron::ser::PrettyConfig::default()).unwrap();
        let back: MaterialFile = ron::from_str(&text).unwrap();
        assert_eq!(back.features.stencil, f.stencil);
        assert_eq!(back.render_queue, Some(1999));
        // A partial stencil block fills the rest from defaults.
        let partial = r#"(name: "P", params: (), features: (toon: false, normal_map: false, emission: false, alpha_mode: Opaque, double_sided: false, stencil: (reference: 3)))"#;
        let m: MaterialFile = ron::from_str(partial).unwrap();
        assert_eq!((m.features.stencil.reference, m.features.stencil.read_mask, m.features.stencil.color_mask), (3, 255, 15));
    }
}

#[cfg(test)]
mod detail_tests {
    use super::*;

    /// Files written before the detail layer / transmission existed carry neither
    /// block (nor the five texture paths) and load with both layers off.
    #[test]
    fn old_material_files_load_with_the_detail_layer_and_transmission_off() {
        let old = r#"(name: "Old", params: (metallic: 0.5), features: (toon: false, normal_map: false, emission: false, alpha_mode: Opaque, double_sided: false), textures: (albedo: Some("a.png")))"#;
        let m: MaterialFile = ron::from_str(old).expect("old file loads");
        assert_eq!(m.params.metallic, 0.5);
        assert!(!m.params.detail.is_on());
        assert_eq!(m.params.detail.tiling, [1.0, 1.0]);
        assert_eq!(m.params.transmission.translucency, 0.0);
        assert_eq!(m.params.transmission.transmission, 0.0);
        assert!(m.textures.detail_albedo.is_none() && m.textures.translucency.is_none());
    }

    #[test]
    fn detail_and_transmission_round_trip_and_partial_blocks_fill_defaults() {
        let mut p = MaterialParams::default();
        p.detail.albedo_strength = 0.8;
        p.detail.tiling = [6.0, 6.0];
        p.transmission.translucency = 1.5;
        p.transmission.tint = [1.0, 0.5, 0.25];
        let tex = MaterialTextures {
            detail_normal: Some("d_n.png".into()),
            translucency: Some("t.png".into()),
            ..Default::default()
        };
        let file = MaterialFile { name: "D".into(), shader: "standard".into(), params: p, features: Default::default(), render_queue: None, textures: tex, custom: Default::default() };
        let text = ron::ser::to_string_pretty(&file, ron::ser::PrettyConfig::default()).unwrap();
        let back: MaterialFile = ron::from_str(&text).unwrap();
        assert_eq!(back.params.detail, p.detail);
        assert_eq!(back.params.transmission, p.transmission);
        assert_eq!(back.textures.detail_normal.as_deref(), Some(Path::new("d_n.png")));
        assert!(!back.textures.is_empty());
        let partial = r#"(name: "P", params: (detail: (normal_strength: 0.5), transmission: (transmission: 0.7)), features: (toon: false, normal_map: false, emission: false, alpha_mode: Opaque, double_sided: false))"#;
        let m: MaterialFile = ron::from_str(partial).unwrap();
        assert_eq!((m.params.detail.normal_strength, m.params.detail.tiling), (0.5, [1.0, 1.0]));
        assert_eq!((m.params.transmission.transmission, m.params.transmission.power), (0.7, 4.0));
    }
}

#[cfg(test)]
mod tess_tests {
    use super::*;

    /// Files written before tessellated displacement existed load with it off; a file that sets
    /// it round-trips, and a partial block fills the rest from the defaults.
    #[test]
    fn tessellation_is_off_in_old_files_and_round_trips() {
        let old = r#"(name: "Old", params: (metallic: 0.5), features: (toon: false, normal_map: false, emission: false, alpha_mode: Opaque, double_sided: false))"#;
        let m: MaterialFile = ron::from_str(old).expect("old file loads");
        assert!(!m.params.tess.enabled);
        let mut p = MaterialParams::default();
        p.tess.enabled = true;
        p.tess.scale = 0.2;
        p.tess.edge_px = 20.0;
        let file = MaterialFile { name: "T".into(), shader: "standard".into(), params: p, features: Default::default(), render_queue: None, textures: Default::default(), custom: Default::default() };
        let text = ron::ser::to_string_pretty(&file, ron::ser::PrettyConfig::default()).unwrap();
        let back: MaterialFile = ron::from_str(&text).unwrap();
        assert_eq!(back.params.tess, p.tess);
        let partial = r#"(name: "P", params: (tess: (enabled: true, scale: 0.3)), features: (toon: false, normal_map: false, emission: false, alpha_mode: Opaque, double_sided: false))"#;
        let m: MaterialFile = ron::from_str(partial).unwrap();
        assert!(m.params.tess.enabled);
        assert_eq!((m.params.tess.scale, m.params.tess.midlevel, m.params.tess.max_factor), (0.3, 0.5, 32.0));
    }
}
