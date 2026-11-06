//! Unity material conversion: a parsed Unity `.mat` (see `unity_mat.rs`) to a
//! ZestEngine `.material` using property mappings for ZestEngine shaders.
//!
//! Recognised shaders (Poiyomi, lilToon, MToon -> ZestEngine Toon; Mochie
//! Standard, Unity Standard, URP Lit -> the PBR standard shader) are mapped
//! property by property through the alias tables below. Those tables hold
//! property *names* (the identifiers Unity serialises into every `.mat`) and
//! the documented meaning of their values; no shader source is used. Everything
//! the converter does not map is listed in the [`Report`], never dropped
//! silently. See `docs/shaders-unity-compat.md` for the property table, the
//! gaps and the sources.
//!
//! Two stages so the mapping is testable without a file system:
//! [`convert_material`] is pure (a [`Plan`]: values + pending texture GUIDs +
//! report); [`finish_plan`] resolves GUIDs through the Unity project's `.meta`
//! files, copies the textures into the project and writes the `.material`
//! and `<name>.convert.txt`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use zestengine_render::{AlphaMode, BlendMode, MaterialFeatures, MaterialParams, StencilCompare, StencilFace, StencilOp, StencilState};

use crate::unity_mat::{
    UnityMaterial, UnityShader, UnityTexEnv, identify_shader, looks_binary, parse_meta_guid, parse_unity_materials, shader_declared_name,
};
use crate::{MaterialFile, MaterialTextures, save_material_file};

/// Where the converter puts things; project-relative with forward slashes.
#[derive(Clone, Debug)]
pub struct ConvertOptions {
    /// The ZestEngine Toon shader as the project references it.
    pub toon_shader: String,
    /// The ZestEngine Screen FX shader (Mochie Screen FX converts to it).
    pub screenfx_shader: String,
    /// Folder (under the project root) converted textures are copied to.
    pub texture_dir: String,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self { toon_shader: "shaders/zestengine_toon.frag".into(), screenfx_shader: "shaders/zestengine_screenfx.frag".into(), texture_dir: "textures/unity".into() }
    }
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// Mapped one to one.
    Mapped,
    /// Mapped with a different formula / meaning; check it visually.
    Approx,
    /// A feature with no ZestEngine equivalent yet (see the gap list).
    Unsupported,
    /// A property the converter has no rule for.
    Unmapped,
    /// Something happened that is worth knowing (file copied, ...).
    Info,
}

#[derive(Clone, Debug)]
pub struct ReportLine {
    pub level: Level,
    pub property: String,
    pub note: String,
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub material: String,
    pub shader: String,
    pub target: String,
    pub lines: Vec<ReportLine>,
}

impl Report {
    pub fn count(&self, level: Level) -> usize {
        self.lines.iter().filter(|l| l.level == level).count()
    }

    /// Plain text for `<name>.convert.txt`.
    pub fn to_text(&self) -> String {
        let mut s = format!("Unity material '{}'\nshader: {}\nconverted to: {}\n", self.material, self.shader, self.target);
        let sections = [
            (Level::Mapped, "Mapped"),
            (Level::Approx, "Approximated (check visually)"),
            (Level::Unsupported, "Not supported yet (see docs/shaders-unity-compat.md)"),
            (Level::Unmapped, "No conversion rule (properties with a non-default value)"),
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

    /// One line for a notification.
    pub fn summary(&self) -> String {
        format!(
            "{}: {} mapped, {} approximate, {} unsupported, {} unmapped",
            self.material,
            self.count(Level::Mapped),
            self.count(Level::Approx),
            self.count(Level::Unsupported),
            self.count(Level::Unmapped)
        )
    }
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum TexTarget {
    Albedo,
    Normal,
    Emission,
    Displacement,
    /// Standard-shader detail layer maps and the translucency / thickness map.
    DetailAlbedo,
    DetailNormal,
    DetailOrm,
    DetailMask,
    Translucency,
    /// A Toon-shader texture property by name.
    Custom(String),
}

/// A texture that must be rewritten (channels moved) before the engine can use it.
#[derive(Clone, Debug, PartialEq)]
pub enum RepackKind {
    /// Unity Standard `_DetailMask`: the mask is the alpha channel; ours is red.
    MaskFromAlpha,
    /// HDRP `_DetailMap` red: detail albedo (linear 0.5 = neutral) -> sRGB grey for the x2 blend.
    HdrpAlbedo,
    /// HDRP `_DetailMap` alpha / green: normal X / Y -> a three-channel tangent normal map.
    HdrpNormal,
    /// HDRP `_DetailMap` blue: smoothness -> detail ORM (green = 1 - smoothness).
    HdrpSmoothness,
}

#[derive(Clone, Debug)]
pub struct RepackJob {
    pub kind: RepackKind,
    pub guid: String,
    pub unity_prop: String,
}

#[derive(Clone, Debug)]
pub struct TexUse {
    pub target: TexTarget,
    pub guid: String,
    pub unity_prop: String,
}

/// Where the roughness channel of a packed ORM comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum RoughSource {
    None,
    /// Unity smoothness stored in the alpha of the metallic map (x scale).
    MetallicAlpha(f32),
    /// Unity smoothness stored in the albedo alpha (x scale).
    AlbedoAlpha(f32),
    /// A roughness map's red channel (Mochie style), used as is.
    RoughnessMap(String),
}

/// Textures the standard shader reads packed (R = AO, G = roughness, B = metallic).
#[derive(Clone, Debug, Default)]
pub struct OrmPlan {
    pub metallic_guid: Option<String>,
    pub occlusion_guid: Option<String>,
    pub albedo_guid: Option<String>,
    pub rough: Option<RoughSource>,
}

impl OrmPlan {
    fn is_empty(&self) -> bool {
        self.metallic_guid.is_none() && self.occlusion_guid.is_none() && self.rough.as_ref().is_none_or(|r| *r == RoughSource::None)
    }
}

/// The result of the pure stage.
#[derive(Clone, Debug)]
pub struct Plan {
    pub name: String,
    pub shader: UnityShader,
    pub file: MaterialFile,
    pub textures: Vec<TexUse>,
    pub orm: OrmPlan,
    /// Textures written out as new files with their channels rearranged.
    pub repacks: Vec<RepackJob>,
    pub report: Report,
}

/// sRGB (gamma) component to linear.
pub fn srgb_to_linear(c: f32) -> f32 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

struct Ctx<'a> {
    m: &'a UnityMaterial,
    used: BTreeSet<String>,
    lines: Vec<ReportLine>,
    tex: Vec<TexUse>,
    repacks: Vec<RepackJob>,
}

impl<'a> Ctx<'a> {
    fn new(m: &'a UnityMaterial) -> Self {
        Self { m, used: BTreeSet::new(), lines: Vec::new(), tex: Vec::new(), repacks: Vec::new() }
    }

    fn mark(&mut self, keys: &[&str]) {
        for k in keys {
            if self.m.all_property_names().any(|n| n == *k) {
                self.used.insert((*k).to_owned());
            }
        }
    }

    fn f(&mut self, keys: &[&str]) -> Option<f32> {
        self.mark(keys);
        self.m.float_any(keys)
    }

    /// A toggle: `Some(on)` if any alias is present.
    fn on(&mut self, keys: &[&str]) -> Option<bool> {
        self.f(keys).map(|v| v > 0.5)
    }

    fn color(&mut self, keys: &[&str]) -> Option<[f32; 4]> {
        self.mark(keys);
        self.m.color_any(keys)
    }

    /// A slot with a texture assigned.
    fn tex(&mut self, keys: &[&str]) -> Option<(String, UnityTexEnv)> {
        self.mark(keys);
        self.m.tex_any(keys).map(|(k, t)| (k.to_owned(), t.clone()))
    }

    fn note(&mut self, level: Level, prop: &str, note: impl Into<String>) {
        self.lines.push(ReportLine { level, property: prop.to_owned(), note: note.into() });
    }

    fn use_tex(&mut self, target: TexTarget, prop: &str, env: &UnityTexEnv, level: Level, dest: &str) {
        if let Some(g) = &env.guid {
            self.tex.push(TexUse { target, guid: g.clone(), unity_prop: prop.to_owned() });
            self.note(level, prop, format!("texture -> {dest}"));
        }
    }

    /// Unmapped properties: non-default ones listed one by one, inert ones
    /// (0 / empty texture) in a single line.
    fn finish_unmapped(&mut self) {
        let m = self.m;
        let mut inert: Vec<&str> = Vec::new();
        for (k, t) in &m.textures {
            if self.used.contains(k) {
                continue;
            }
            match &t.guid {
                Some(g) => self.lines.push(ReportLine { level: Level::Unmapped, property: k.clone(), note: format!("texture {g} not used") }),
                None => inert.push(k),
            }
        }
        for (k, v) in &m.floats {
            if self.used.contains(k) {
                continue;
            }
            if *v == 0.0 {
                inert.push(k);
            } else {
                self.lines.push(ReportLine { level: Level::Unmapped, property: k.clone(), note: format!("{v}") });
            }
        }
        for (k, c) in &m.colors {
            if self.used.contains(k) {
                continue;
            }
            if c[..3] == [0.0; 3] {
                inert.push(k);
            } else {
                self.lines.push(ReportLine { level: Level::Unmapped, property: k.clone(), note: format!("({}, {}, {}, {})", c[0], c[1], c[2], c[3]) });
            }
        }
        if !inert.is_empty() {
            self.lines.push(ReportLine {
                level: Level::Info,
                property: format!("{} unused properties", inert.len()),
                note: format!("zero / empty, ignored: {}", inert.join(", ")),
            });
        }
    }
}

/// Property alias tables for the Toon-shader family. Each concept lists the
/// serialised names the shaders use for it (first present wins).
struct ToonNames {
    main_tex: &'static [&'static str],
    color: &'static [&'static str],
    cutoff: &'static [&'static str],
    bump: &'static [&'static str],
    bump_scale: &'static [&'static str],
    shadow_enable: &'static [&'static str],
    shadow_color: &'static [&'static str],
    shadow_strength: &'static [&'static str],
    shadow_border: &'static [&'static str],
    shadow_blur: &'static [&'static str],
    shadow2_color: &'static [&'static str],
    shadow2_border: &'static [&'static str],
    shadow2_blur: &'static [&'static str],
    shadow_mask: &'static [&'static str],
    ramp: &'static [&'static str],
    ramp_offset: &'static [&'static str],
    light_min: &'static [&'static str],
    light_max: &'static [&'static str],
    rim_enable: &'static [&'static str],
    rim_color: &'static [&'static str],
    rim_width: &'static [&'static str],
    rim_sharp: &'static [&'static str],
    rim_strength: &'static [&'static str],
    rim_mask: &'static [&'static str],
    matcap_enable: &'static [&'static str],
    matcap_tex: &'static [&'static str],
    matcap_color: &'static [&'static str],
    matcap_mask: &'static [&'static str],
    matcap_strength: &'static [&'static str],
    emission_enable: &'static [&'static str],
    emission_tex: &'static [&'static str],
    emission_color: &'static [&'static str],
    emission_strength: &'static [&'static str],
    emission_mask: &'static [&'static str],
    emission_scroll: &'static [&'static str],
    outline_enable: &'static [&'static str],
    outline_width: &'static [&'static str],
    outline_color: &'static [&'static str],
    alpha_mask: &'static [&'static str],
    alpha_mod: &'static [&'static str],
    metallic: &'static [&'static str],
    smoothness: &'static [&'static str],
    saturation: &'static [&'static str],
    cull: &'static [&'static str],
}

const POIYOMI: ToonNames = ToonNames {
    main_tex: &["_MainTex"],
    color: &["_Color"],
    cutoff: &["_Cutoff"],
    bump: &["_BumpMap"],
    bump_scale: &["_BumpScale"],
    shadow_enable: &["_ShadingEnabled"],
    shadow_color: &["_ShadowColor", "_Shadow1Color", "_LightingShadowColor", "_ShadowTint"],
    shadow_strength: &["_ShadowStrength", "_LightingShadowStrength"],
    shadow_border: &["_ShadowBorder", "_Shadow1Border", "_LightingShadowBorder"],
    shadow_blur: &["_ShadowBlur", "_Shadow1Blur"],
    shadow2_color: &["_Shadow2ndColor", "_Shadow2Color"],
    shadow2_border: &["_Shadow2ndBorder", "_Shadow2Border"],
    shadow2_blur: &["_Shadow2ndBlur", "_Shadow2Blur"],
    shadow_mask: &["_LightingShadowMask", "_ShadowMask"],
    ramp: &["_ToonRamp"],
    ramp_offset: &["_ShadowOffset"],
    light_min: &["_LightingMinLightBrightness"],
    light_max: &["_LightingCap"],
    rim_enable: &["_EnableRimLighting", "_RimLightingEnabled"],
    rim_color: &["_RimLightColor"],
    rim_width: &["_RimWidth"],
    rim_sharp: &["_RimSharpness"],
    rim_strength: &["_RimStrength"],
    rim_mask: &["_RimMask"],
    matcap_enable: &["_MatcapEnable"],
    matcap_tex: &["_Matcap"],
    matcap_color: &["_MatcapColor"],
    matcap_mask: &["_MatcapMask"],
    matcap_strength: &["_MatcapIntensity"],
    emission_enable: &["_EnableEmission"],
    emission_tex: &["_EmissionMap"],
    emission_color: &["_EmissionColor"],
    emission_strength: &["_EmissionStrength"],
    emission_mask: &["_EmissionMask"],
    emission_scroll: &["_EmissionScrollingVector"],
    outline_enable: &["_EnableOutlines", "_EnableOutline"],
    outline_width: &["_LineWidth", "_OutlineWidth"],
    outline_color: &["_OutlineColor"],
    alpha_mask: &["_AlphaMask"],
    alpha_mod: &["_AlphaMod"],
    metallic: &["_Metallic"],
    smoothness: &["_Smoothness"],
    saturation: &["_Saturation"],
    cull: &["_Cull"],
};

const LILTOON: ToonNames = ToonNames {
    main_tex: &["_MainTex", "_BaseMap"],
    color: &["_Color", "_BaseColor"],
    cutoff: &["_Cutoff"],
    bump: &["_BumpMap"],
    bump_scale: &["_BumpScale"],
    shadow_enable: &["_UseShadow"],
    shadow_color: &["_ShadowColor"],
    shadow_strength: &["_ShadowStrength"],
    shadow_border: &["_ShadowBorder"],
    shadow_blur: &["_ShadowBlur"],
    shadow2_color: &["_Shadow2ndColor"],
    shadow2_border: &["_Shadow2ndBorder"],
    shadow2_blur: &["_Shadow2ndBlur"],
    shadow_mask: &["_ShadowStrengthMask"],
    ramp: &[],
    ramp_offset: &[],
    light_min: &["_LightMinLimit"],
    light_max: &["_LightMaxLimit"],
    rim_enable: &["_UseRim"],
    rim_color: &["_RimColor"],
    rim_width: &["_RimBorder"],
    rim_sharp: &["_RimBlur"],
    rim_strength: &[],
    rim_mask: &["_RimColorTex"],
    matcap_enable: &["_UseMatCap"],
    matcap_tex: &["_MatCapTex"],
    matcap_color: &["_MatCapColor"],
    matcap_mask: &["_MatCapBlendMask"],
    matcap_strength: &["_MatCapBlend"],
    emission_enable: &["_UseEmission"],
    emission_tex: &["_EmissionMap"],
    emission_color: &["_EmissionColor"],
    emission_strength: &[],
    emission_mask: &["_EmissionBlendMask"],
    emission_scroll: &[],
    outline_enable: &[],
    outline_width: &["_OutlineWidth"],
    outline_color: &["_OutlineColor"],
    alpha_mask: &["_AlphaMask"],
    alpha_mod: &[],
    metallic: &["_Metallic"],
    smoothness: &["_Smoothness"],
    saturation: &[],
    cull: &["_Cull"],
};

const MTOON: ToonNames = ToonNames {
    main_tex: &["_MainTex"],
    color: &["_Color"],
    cutoff: &["_Cutoff"],
    bump: &["_BumpMap"],
    bump_scale: &["_BumpScale"],
    shadow_enable: &[],
    shadow_color: &["_ShadeColor"],
    shadow_strength: &["_ReceiveShadowRate"],
    shadow_border: &["_ShadeShift"],
    shadow_blur: &["_ShadeToony"],
    shadow2_color: &[],
    shadow2_border: &[],
    shadow2_blur: &[],
    shadow_mask: &["_ReceiveShadowTexture"],
    ramp: &[],
    ramp_offset: &[],
    light_min: &[],
    light_max: &[],
    rim_enable: &[],
    rim_color: &["_RimColor"],
    rim_width: &["_RimFresnelPower"],
    rim_sharp: &["_RimLift"],
    rim_strength: &["_RimLightingMix"],
    rim_mask: &["_RimTexture"],
    matcap_enable: &[],
    matcap_tex: &["_SphereAdd"],
    matcap_color: &[],
    matcap_mask: &[],
    matcap_strength: &[],
    emission_enable: &[],
    emission_tex: &["_EmissionMap"],
    emission_color: &["_EmissionColor"],
    emission_strength: &[],
    emission_mask: &[],
    emission_scroll: &[],
    outline_enable: &[],
    outline_width: &["_OutlineWidth"],
    outline_color: &["_OutlineColor"],
    alpha_mask: &[],
    alpha_mod: &[],
    metallic: &[],
    smoothness: &[],
    saturation: &[],
    cull: &["_CullMode"],
};

/// Keyword features of ZestEngine Toon, all written explicitly so a converted
/// material never depends on the shader's defaults.
const TOON_FEATURES: [&str; 25] = [
    "RECEIVE_SHADOWS", "NORMAL_MAP", "ALPHA", "RIM", "MATCAP", "MATCAP2", "SPECULAR", "REFLECTIONS", "EMISSION", "EMISSION2", "DETAIL", "DECAL", "DISSOLVE", "GLITTER", "UV_DISCARD", "PARALLAX", "VERTEX_COLOR", "OUTLINE", "COLOR_MASK", "DISTANCE_FADE", "PANOSPHERE", "FLIPBOOK", "GRAB", "AUDIO", "FUR",
];

/// ZestEngine Toon `lighting_mode` enum indices.
const LM_RAMP: f32 = 0.0;
const LM_MULTILAYER: f32 = 1.0;
const LM_REALISTIC: f32 = 2.0;
const LM_FLAT: f32 = 3.0;
const LM_WRAPPED: f32 = 4.0;

fn dominant_alpha(ctx: &mut Ctx, fam: &UnityShader) -> AlphaMode {
    let m = ctx.m;
    let mode = match fam {
        UnityShader::PoiyomiToon => ctx.f(&["_Mode"]).map(|v| v.round() as i32),
        UnityShader::LilToon => ctx.f(&["_TransparentMode"]).map(|v| v.round() as i32),
        _ => ctx.f(&["_BlendMode", "_Mode"]).map(|v| v.round() as i32),
    };
    let mapped = match (fam, mode) {
        // Poiyomi: 0 Opaque, 1 Cutout, 2 TransClipping, 3 Fade, 4 Transparent, 5+ additive / multiplicative.
        (UnityShader::PoiyomiToon, Some(0)) => Some(AlphaMode::Opaque),
        (UnityShader::PoiyomiToon, Some(1)) => Some(AlphaMode::Cutout),
        (UnityShader::PoiyomiToon, Some(2..=4)) => Some(AlphaMode::Blend),
        (UnityShader::PoiyomiToon, Some(5..=8)) => Some(AlphaMode::Blend),
        // lilToon: 0 Normal, 1 Cutout, 2 Transparent, 3 Refraction, 4 Fur, 5 Gem.
        (UnityShader::LilToon, Some(0)) => Some(AlphaMode::Opaque),
        (UnityShader::LilToon, Some(1)) => Some(AlphaMode::Cutout),
        (UnityShader::LilToon, Some(2)) => Some(AlphaMode::Blend),
        (UnityShader::LilToon, Some(v @ 3..=5)) => {
            let what = ["refraction", "fur", "gem"][(v - 3) as usize];
            let note = if v == 3 { "refraction mode -> Grab Pass (Refraction) feature".to_owned() } else { format!("{what} mode has no equivalent; alpha blend used") };
            ctx.note(if v == 3 { Level::Approx } else { Level::Unsupported }, "_TransparentMode", note);
            Some(AlphaMode::Blend)
        }
        // MToon / Standard / Mochie: 0 Opaque, 1 Cutout, 2 Fade, 3 Transparent.
        (_, Some(0)) => Some(AlphaMode::Opaque),
        (_, Some(1)) => Some(AlphaMode::Cutout),
        (_, Some(2 | 3)) => Some(AlphaMode::Blend),
        _ => None,
    };
    mapped.unwrap_or_else(|| {
        if m.has_keyword("_ALPHATEST_ON") {
            AlphaMode::Cutout
        } else if m.has_keyword("_ALPHABLEND_ON") || m.has_keyword("_ALPHAPREMULTIPLY_ON") || m.render_queue.is_some_and(|q| q >= 3000) {
            AlphaMode::Blend
        } else if m.render_queue.is_some_and(|q| (2450..3000).contains(&q)) {
            AlphaMode::Cutout
        } else {
            AlphaMode::Opaque
        }
    })
}

/// Unity cull enum (0 off, 1 front, 2 back) to (double sided, cull front faces).
fn cull_state(ctx: &mut Ctx, keys: &[&str]) -> (bool, bool) {
    match ctx.f(keys).map(|v| v.round() as i32) {
        Some(0) => (true, false),
        Some(1) => {
            ctx.note(Level::Mapped, keys.first().copied().unwrap_or("_Cull"), "front-face culling -> Cull Front Faces (shows the inside of the mesh)");
            (false, true)
        }
        _ => (false, false),
    }
}

/// Unity blend factor pair (`UnityEngine.Rendering.BlendMode`) to a ZestEngine
/// transparent blend equation, when it is one of the common ones.
fn blend_from_factors(src: i32, dst: i32) -> Option<BlendMode> {
    match (src, dst) {
        (5, 10) => Some(BlendMode::Alpha),
        (1, 10) => Some(BlendMode::Premultiplied),
        (5 | 1, 1) => Some(BlendMode::Additive),
        (2 | 0, 0 | 3) => Some(BlendMode::Multiply),
        _ => None,
    }
}

/// Unity `CompareFunction` enum value to a stencil compare (`Disabled` = Always).
fn stencil_compare_from(v: i32) -> StencilCompare {
    match v {
        1 => StencilCompare::Never,
        2 => StencilCompare::Less,
        3 => StencilCompare::Equal,
        4 => StencilCompare::LessEqual,
        5 => StencilCompare::Greater,
        6 => StencilCompare::NotEqual,
        7 => StencilCompare::GreaterEqual,
        _ => StencilCompare::Always,
    }
}

/// Unity `StencilOp` enum value to a stencil op.
fn stencil_op_from(v: i32) -> StencilOp {
    match v {
        1 => StencilOp::Zero,
        2 => StencilOp::Replace,
        3 => StencilOp::IncrementClamp,
        4 => StencilOp::DecrementClamp,
        5 => StencilOp::Invert,
        6 => StencilOp::IncrementWrap,
        7 => StencilOp::DecrementWrap,
        _ => StencilOp::Keep,
    }
}

/// Stencil + colour mask: Poiyomi (`_StencilReference`, `_StencilCompareFunction`,
/// `_StencilPassOp` ..., `_StencilType` preset), lilToon (`_StencilRef`, `_StencilComp`,
/// `_StencilPass` ...) and the Unity UI / Standard names (`_Stencil`, `_StencilOp`).
/// Per-face `...Front` / `...Back` variants (`_StencilCompBack`, `_StencilPassBack`...)
/// fill the two faces separately. `_StencilType` only fills what the explicit
/// properties left out: 1 = writer (Always + Replace), 2 = reader (Equal + Keep).
/// The draw-order rule (writers before readers) rides on the render queue, which is
/// converted separately: it is reported when the state writes the stencil.
fn convert_stencil(ctx: &mut Ctx, file: &mut MaterialFile) {
    let int = |ctx: &mut Ctx, keys: &[&str]| ctx.f(keys).map(|v| v.round() as i32);
    let reference = int(ctx, &["_StencilReference", "_StencilRef", "_Stencil"]);
    let read_mask = int(ctx, &["_StencilReadMask"]);
    let write_mask = int(ctx, &["_StencilWriteMask"]);
    let kind = int(ctx, &["_StencilType"]).unwrap_or(0);
    let comp = int(ctx, &["_StencilCompareFunction", "_StencilComp"]);
    let pass = int(ctx, &["_StencilPassOp", "_StencilPass", "_StencilOp"]);
    let fail = int(ctx, &["_StencilFailOp", "_StencilFail"]);
    let zfail = int(ctx, &["_StencilZFailOp", "_StencilZFail"]);
    let mut face = StencilFace {
        compare: comp.map_or(StencilCompare::Always, stencil_compare_from),
        pass_op: pass.map_or(StencilOp::Keep, stencil_op_from),
        fail_op: fail.map_or(StencilOp::Keep, stencil_op_from),
        zfail_op: zfail.map_or(StencilOp::Keep, stencil_op_from),
    };
    let mut preset = false;
    if kind == 1 && comp.is_none_or(|c| c == 8 || c == 0) && pass.is_none_or(|p| p == 0) {
        (face.compare, face.pass_op) = (StencilCompare::Always, StencilOp::Replace);
        preset = true;
    } else if kind == 2 && comp.is_none_or(|c| c == 8 || c == 0) {
        face.compare = StencilCompare::Equal;
        preset = true;
    }
    let mut state = StencilState::both(face, reference.unwrap_or(0).clamp(0, 255) as u8, read_mask.unwrap_or(255).clamp(0, 255) as u8, write_mask.unwrap_or(255).clamp(0, 255) as u8);
    // Per-face overrides.
    let face_keys: [(&str, &str, &str, &str); 2] = [
        ("Front", "_StencilCompFront", "_StencilPassFront", "_StencilFailFront"),
        ("Back", "_StencilCompBack", "_StencilPassBack", "_StencilFailBack"),
    ];
    for (i, (name, c, p, f)) in face_keys.into_iter().enumerate() {
        let z = if i == 0 { "_StencilZFailFront" } else { "_StencilZFailBack" };
        let (c, p, f, z) = (int(ctx, &[c]), int(ctx, &[p]), int(ctx, &[f]), int(ctx, &[z]));
        if c.is_none() && p.is_none() && f.is_none() && z.is_none() {
            continue;
        }
        let target = if i == 0 { &mut state.front } else { &mut state.back };
        if let Some(c) = c {
            target.compare = stencil_compare_from(c);
        }
        if let Some(p) = p {
            target.pass_op = stencil_op_from(p);
        }
        if let Some(f) = f {
            target.fail_op = stencil_op_from(f);
        }
        if let Some(z) = z {
            target.zfail_op = stencil_op_from(z);
        }
        ctx.note(Level::Mapped, "_Stencil*", format!("{name}-face stencil state"));
    }
    if let Some(mask) = int(ctx, &["_ColorMask"]) {
        // Unity's ColorMask enum is the RGBA bit set (R = 8 ... A = 1 in the UI enum,
        // serialised as the 0..15 value with A = bit 0); ZestEngine keeps R = 1.
        let rgba = mask.clamp(0, 15) as u8;
        state.color_mask = ((rgba >> 3) & 1) | (((rgba >> 2) & 1) << 1) | (((rgba >> 1) & 1) << 2) | ((rgba & 1) << 3);
        if state.color_mask != 15 {
            ctx.note(Level::Mapped, "_ColorMask", format!("colour mask {rgba} -> Stencil: Write Colour {}", if state.color_mask == 0 { "off (pure stencil writer)" } else { "partial" }));
        }
    }
    // `_ZWrite` only matters next to stencil / colour masking (a plain material keeps the default).
    let zwrite = ctx.on(&["_ZWrite"]);
    if let Some(false) = zwrite && (state.is_active() || state.color_mask != 15) {
        state.depth_write = false;
        ctx.note(Level::Mapped, "_ZWrite", "ZWrite off -> Stencil: Write Depth off (the stencil writer does not occlude later draws)");
    }
    if state.is_active() {
        let how = if preset { " (from the _StencilType preset)" } else { "" };
        ctx.note(
            Level::Mapped,
            "_Stencil*",
            format!("stencil ref {} read {:#04x} write {:#04x}: {:?} / pass {:?}{how}", state.reference, state.read_mask, state.write_mask, state.front.compare, state.front.pass_op),
        );
        let writes = [state.front, state.back].iter().any(|f| f.pass_op != StencilOp::Keep || f.fail_op != StencilOp::Keep || f.zfail_op != StencilOp::Keep);
        if writes {
            ctx.note(Level::Info, "_Stencil*", "this material writes the stencil: it draws before its readers only if its render queue is lower (check the queue of both)");
        }
    }
    if state.is_active() || state.color_mask != 15 || !state.depth_write {
        file.features.stencil = state;
    }
}

/// Convert one parsed material. `shader` comes from [`identify_shader`].
pub fn convert_material(m: &UnityMaterial, shader: UnityShader, opts: &ConvertOptions) -> Plan {
    let mut ctx = Ctx::new(m);
    let name = if m.name.is_empty() { "material".to_owned() } else { m.name.clone() };
    let mut file = MaterialFile {
        name: name.clone(),
        shader: "standard".into(),
        params: MaterialParams::default(),
        features: MaterialFeatures::default(),
        render_queue: None,
        textures: MaterialTextures::default(),
        custom: BTreeMap::new(),
    };
    let mut orm = OrmPlan::default();
    let target;
    match &shader {
        UnityShader::PoiyomiToon => {
            target = "ZestEngine Toon";
            convert_toon(&mut ctx, &mut file, &shader, &POIYOMI, opts);
        }
        UnityShader::LilToon => {
            target = "ZestEngine Toon";
            convert_toon(&mut ctx, &mut file, &shader, &LILTOON, opts);
        }
        UnityShader::MToon => {
            target = "ZestEngine Toon";
            convert_toon(&mut ctx, &mut file, &shader, &MTOON, opts);
        }
        UnityShader::MochieOther(n) if n.to_ascii_lowercase().starts_with("screen fx") => {
            target = "ZestEngine Screen FX";
            convert_mochie_screenfx(&mut ctx, &mut file, opts);
        }
        UnityShader::MochieOther(n) => {
            target = "PBR standard (placeholder)";
            ctx.note(Level::Unsupported, "shader", format!("Mochie '{n}' has no ZestEngine counterpart; only colour, texture and alpha were carried over"));
            orm = convert_standard(&mut ctx, &mut file, &shader);
        }
        _ => {
            target = "PBR standard";
            if let UnityShader::Unknown(n) = &shader {
                ctx.note(Level::Info, "shader", format!("{} - generic Standard-style mapping used{}", shader.label(), if n.is_empty() { " (the shader's name is not known: pass the folder with its .shader file or its guid)" } else { "" }));
            }
            orm = convert_standard(&mut ctx, &mut file, &shader);
        }
    }
    file.render_queue = m.render_queue.filter(|q| *q > 0);
    ctx.finish_unmapped();
    let mut lines = std::mem::take(&mut ctx.lines);
    lines.sort_by_key(|l| l.level);
    Plan {
        name: name.clone(),
        report: Report { material: name, shader: shader.label(), target: target.into(), lines },
        shader,
        file,
        textures: ctx.tex,
        orm,
        repacks: ctx.repacks,
    }
}

// ---------------------------------------------------------------------------
// Toon family
// ---------------------------------------------------------------------------

struct Toon<'a> {
    custom: &'a mut BTreeMap<String, Vec<f32>>,
}

impl Toon<'_> {
    fn v(&mut self, k: &str, v: &[f32]) {
        self.custom.insert(k.to_owned(), v.to_vec());
    }
    fn on(&mut self, k: &str, on: bool) {
        self.v(k, &[f32::from(on)]);
    }
}

fn convert_toon(ctx: &mut Ctx, file: &mut MaterialFile, fam: &UnityShader, n: &ToonNames, opts: &ConvertOptions) {
    file.shader = opts.toon_shader.clone();
    file.features.toon = true;
    let m = ctx.m;
    let mut custom = BTreeMap::new();
    {
        let mut t = Toon { custom: &mut custom };
        for k in TOON_FEATURES {
            t.on(k, false);
        }
    }
    let is_poi = *fam == UnityShader::PoiyomiToon;
    let is_lil = *fam == UnityShader::LilToon;
    let is_mtoon = *fam == UnityShader::MToon;

    // ---- base colour, albedo, tiling ----
    let mut main_color = [1.0f32; 4];
    if let Some(c) = ctx.color(n.color) {
        main_color = [srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2]), c[3]];
        ctx.note(Level::Mapped, n.color[0], "colour (sRGB -> linear) -> main_color");
    }
    let mut st = [1.0, 1.0, 0.0, 0.0];
    if let Some((k, env)) = ctx.tex(n.main_tex) {
        st = [env.scale[0], env.scale[1], env.offset[0], env.offset[1]];
        ctx.use_tex(TexTarget::Albedo, &k, &env, Level::Mapped, "Albedo slot (sRGB), tiling/offset -> main_st");
        file.params.albedo_tiling = env.scale;
        file.params.albedo_offset = env.offset;
    } else {
        ctx.mark(n.main_tex);
    }
    if let Some(s) = ctx.f(n.saturation) {
        // Poiyomi saturation is -1..1 around 0; ZestEngine's is 0..2 around 1.
        let sat = (1.0 + s).clamp(0.0, 2.0);
        custom.insert("saturation".into(), vec![sat]);
        ctx.note(Level::Approx, n.saturation[0], format!("{s} -> saturation {sat}"));
    }

    // ---- alpha / cull / queue ----
    file.features.alpha_mode = dominant_alpha(ctx, fam);
    if let Some(c) = ctx.f(n.cutoff) {
        file.params.alpha_cutoff = c.clamp(0.0, 1.0);
        ctx.note(Level::Mapped, n.cutoff[0], "alpha cutoff");
    }
    (file.features.double_sided, file.features.cull_front) = cull_state(ctx, n.cull);
    convert_stencil(ctx, file);
    file.features.blend = toon_blend(ctx, fam, file.features.alpha_mode);
    let mut alpha_kw = false;
    if let Some((k, env)) = ctx.tex(n.alpha_mask) {
        ctx.use_tex(TexTarget::Custom("alpha_mask".into()), &k, &env, Level::Mapped, "alpha_mask");
        alpha_kw = true;
    }
    if let Some(a) = ctx.f(n.alpha_mod).filter(|a| *a != 0.0) {
        custom.insert("alpha_mod".into(), vec![a.clamp(-1.0, 1.0)]);
        ctx.note(Level::Mapped, n.alpha_mod[0], "alpha offset");
        alpha_kw = true;
    }
    custom.insert("ALPHA".into(), vec![f32::from(alpha_kw)]);

    // ---- normal map ----
    if let Some((k, env)) = ctx.tex(n.bump) {
        ctx.use_tex(TexTarget::Normal, &k, &env, Level::Mapped, "Normal slot (normal-map import)");
        file.features.normal_map = true;
        file.params.normal_tiling = env.scale;
        file.params.normal_offset = env.offset;
        custom.insert("NORMAL_MAP".into(), vec![1.0]);
        let s = ctx.f(n.bump_scale).unwrap_or(1.0);
        custom.insert("normal_strength".into(), vec![s]);
        file.params.normal_strength = s;
    } else {
        ctx.mark(n.bump_scale);
    }

    // ---- lighting / shadows ----
    let mut lighting = LM_MULTILAYER;
    if is_poi {
        let enabled = ctx.on(n.shadow_enable).unwrap_or(true);
        if let Some(mode) = ctx.f(&["_LightingMode"]).map(|v| v.round() as i32) {
            // Poiyomi order: Ramp, Multilayer Math, Wrapped, Skin, ShadeMap, Flat, Realistic, Cloth, SDF, PBR.
            lighting = match mode {
                0 => LM_RAMP,
                1 => LM_MULTILAYER,
                2 => LM_WRAPPED,
                5 => LM_FLAT,
                6 | 7 | 9 => LM_REALISTIC,
                _ => LM_MULTILAYER,
            };
            let names = ["Texture Ramp", "Multilayer Math", "Wrapped", "Skin", "ShadeMap", "Flat", "Realistic", "Cloth", "SDF", "PBR"];
            let nm = names.get(mode.max(0) as usize).copied().unwrap_or("?");
            let lvl = if matches!(mode, 0 | 1 | 2 | 5 | 6) { Level::Mapped } else { Level::Approx };
            ctx.note(lvl, "_LightingMode", format!("{nm} -> lighting_mode {}", ["Ramp", "Multilayer", "Realistic", "Flat", "Wrapped"][lighting as usize]));
        }
        if !enabled {
            lighting = LM_FLAT;
            ctx.note(Level::Mapped, "_ShadingEnabled", "off -> Flat lighting");
        }
    }
    custom.insert("lighting_mode".into(), vec![lighting]);
    if lighting == LM_WRAPPED && let Some(w) = ctx.f(&["_LightingWrappedWrap", "_WrappedWrap"]) {
        custom.insert("wrap_amount".into(), vec![w.clamp(0.0, 1.0)]);
    }
    let shadows = if is_lil { ctx.on(n.shadow_enable).unwrap_or(true) } else { true };
    custom.insert("RECEIVE_SHADOWS".into(), vec![1.0]);
    if shadows {
        if let Some(c) = ctx.color(n.shadow_color) {
            let a = if is_lil { c[3] } else { 1.0 };
            custom.insert("shadow_color".into(), vec![srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2])]);
            if is_lil {
                custom.insert("shadow_strength".into(), vec![a.clamp(0.0, 1.0)]);
            }
            ctx.note(Level::Mapped, n.shadow_color[0], "shadow colour");
        }
        if is_mtoon {
            // MToon: shade shift (-1..1) and toony (0..1) define the terminator.
            let shift = ctx.f(&["_ShadeShift"]).unwrap_or(0.0);
            let toony = ctx.f(&["_ShadeToony"]).unwrap_or(0.9);
            custom.insert("layer1_border".into(), vec![(0.5 - shift * 0.5).clamp(0.0, 1.0)]);
            custom.insert("layer1_blur".into(), vec![((1.0 - toony) * 0.5).clamp(0.0, 0.5)]);
            ctx.note(Level::Approx, "_ShadeShift/_ShadeToony", "-> layer1_border / layer1_blur");
            if let Some(r) = ctx.f(n.shadow_strength) {
                custom.insert("shadow_strength".into(), vec![r.clamp(0.0, 1.0)]);
            }
            if let Some((k, _)) = ctx.tex(&["_ShadeTexture"]) {
                ctx.note(Level::Unsupported, &k, "per-pixel shade-colour texture (ZestEngine uses one shadow colour times the albedo)");
            }
        } else {
            if let Some(v) = ctx.f(n.shadow_strength).filter(|_| !is_lil) {
                custom.insert("shadow_strength".into(), vec![v.clamp(0.0, 1.0)]);
                ctx.note(Level::Mapped, n.shadow_strength[0], "shadow strength");
            } else {
                ctx.mark(n.shadow_strength);
            }
            if let Some(b) = ctx.f(n.shadow_border) {
                custom.insert("layer1_border".into(), vec![b.clamp(0.0, 1.0)]);
                ctx.note(Level::Mapped, n.shadow_border[0], "layer 1 border");
            }
            if let Some(b) = ctx.f(n.shadow_blur) {
                custom.insert("layer1_blur".into(), vec![b.clamp(0.0, 0.5)]);
                ctx.note(Level::Mapped, n.shadow_blur[0], "layer 1 blur");
            }
        }
        if let Some(c) = ctx.color(n.shadow2_color) {
            custom.insert("layer2_color".into(), vec![srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2])]);
            custom.insert("layer2_strength".into(), vec![if is_lil { c[3].clamp(0.0, 1.0) } else { 1.0 }]);
            ctx.note(Level::Mapped, n.shadow2_color[0], "layer 2 colour/strength");
            if let Some(b) = ctx.f(n.shadow2_border) {
                custom.insert("layer2_border".into(), vec![b.clamp(0.0, 1.0)]);
            }
            if let Some(b) = ctx.f(n.shadow2_blur) {
                custom.insert("layer2_blur".into(), vec![b.clamp(0.0, 0.5)]);
            }
        } else {
            ctx.mark(n.shadow2_border);
            ctx.mark(n.shadow2_blur);
        }
        if let Some((k, env)) = ctx.tex(n.shadow_mask) {
            ctx.use_tex(TexTarget::Custom("shadow_mask".into()), &k, &env, Level::Mapped, "shadow_mask");
        }
        if let Some((k, env)) = ctx.tex(n.ramp) {
            ctx.use_tex(TexTarget::Custom("ramp_tex".into()), &k, &env, Level::Mapped, "ramp_tex");
            if let Some(o) = ctx.f(n.ramp_offset) {
                custom.insert("ramp_offset".into(), vec![o.clamp(-1.0, 1.0)]);
            }
        }
    } else {
        custom.insert("shadow_strength".into(), vec![0.0]);
        ctx.note(Level::Mapped, n.shadow_enable[0], "off -> shadow strength 0");
    }
    if let Some(v) = ctx.f(n.light_min) {
        custom.insert("light_min".into(), vec![v.clamp(0.0, 1.0)]);
    }
    if let Some(v) = ctx.f(n.light_max) {
        custom.insert("light_max".into(), vec![v.clamp(0.0, 10.0)]);
    }
    if lighting == LM_REALISTIC {
        if let Some(v) = ctx.f(n.metallic) {
            custom.insert("metallic".into(), vec![v.clamp(0.0, 1.0)]);
        }
        if let Some(v) = ctx.f(n.smoothness) {
            custom.insert("roughness".into(), vec![(1.0 - v).clamp(0.0, 1.0)]);
            ctx.note(Level::Approx, n.smoothness[0], "smoothness -> roughness (1 - s)");
        }
    }

    // ---- rim ----
    let rim_on = match ctx.on(n.rim_enable) {
        Some(b) => b,
        None => ctx.color(n.rim_color).is_some_and(|c| c[..3] != [0.0; 3]),
    };
    if rim_on {
        custom.insert("RIM".into(), vec![1.0]);
        if let Some(c) = ctx.color(n.rim_color) {
            custom.insert("rim_color".into(), vec![srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2])]);
            ctx.note(Level::Mapped, n.rim_color[0], "rim colour");
        }
        let w = ctx.f(n.rim_width);
        let sharp = ctx.f(n.rim_sharp);
        if is_poi {
            if let Some(v) = w {
                custom.insert("rim_width".into(), vec![v.clamp(0.0, 1.0)]);
                ctx.note(Level::Mapped, n.rim_width[0], "rim width");
            }
            if let Some(v) = sharp {
                custom.insert("rim_sharpness".into(), vec![v.clamp(0.0, 1.0)]);
            }
            if let Some(v) = ctx.f(n.rim_strength) {
                custom.insert("rim_strength".into(), vec![v.clamp(0.0, 4.0)]);
            }
        } else if is_lil {
            if let Some(v) = w {
                custom.insert("rim_width".into(), vec![(1.0 - v).clamp(0.0, 1.0)]);
                ctx.note(Level::Approx, n.rim_width[0], "rim border -> rim_width (1 - border)");
            }
            if let Some(v) = sharp {
                custom.insert("rim_sharpness".into(), vec![(1.0 - v).clamp(0.0, 1.0)]);
                ctx.note(Level::Approx, n.rim_sharp[0], "rim blur -> rim_sharpness (1 - blur)");
            }
        } else {
            if let Some(p) = w {
                custom.insert("rim_width".into(), vec![(1.0 / (1.0 + p.max(0.0))).clamp(0.05, 1.0)]);
                ctx.note(Level::Approx, n.rim_width[0], "fresnel power -> rim_width (1 / (1 + p))");
            }
            if let Some(v) = ctx.f(n.rim_strength) {
                custom.insert("rim_strength".into(), vec![v.clamp(0.0, 4.0)]);
            }
            ctx.mark(n.rim_sharp);
        }
        if let Some((k, env)) = ctx.tex(n.rim_mask) {
            ctx.use_tex(TexTarget::Custom("rim_mask".into()), &k, &env, Level::Approx, "rim_mask");
        }
    } else {
        for k in [n.rim_width, n.rim_sharp, n.rim_strength, n.rim_color] {
            ctx.mark(k);
        }
    }

    // ---- matcap ----
    let has_matcap = ctx.tex(n.matcap_tex);
    let matcap_on = ctx.on(n.matcap_enable).unwrap_or(has_matcap.is_some() && !is_mtoon) && has_matcap.is_some()
        || (is_mtoon && has_matcap.is_some());
    if let (true, Some((k, env))) = (matcap_on, has_matcap) {
        custom.insert("MATCAP".into(), vec![1.0]);
        ctx.use_tex(TexTarget::Custom("matcap_tex".into()), &k, &env, Level::Mapped, "matcap_tex");
        if let Some((mk, menv)) = ctx.tex(n.matcap_mask) {
            ctx.use_tex(TexTarget::Custom("matcap_mask".into()), &mk, &menv, Level::Mapped, "matcap_mask");
        }
        // Blend: ZestEngine Add 0, Multiply 1, Replace 2, Screen 3.
        let (blend, strength) = if is_poi {
            let add = ctx.f(&["_MatcapAdd"]).unwrap_or(0.0);
            let mul = ctx.f(&["_MatcapMultiply"]).unwrap_or(0.0);
            let rep = ctx.f(&["_MatcapReplace"]).unwrap_or(0.0);
            let inten = ctx.f(n.matcap_strength).unwrap_or(1.0);
            if rep >= add && rep >= mul && rep > 0.0 {
                (2.0, rep * inten)
            } else if mul > add {
                (1.0, mul * inten)
            } else {
                (0.0, if add > 0.0 { add * inten } else { inten })
            }
        } else if is_lil {
            let mode = ctx.f(&["_MatCapBlendMode"]).map(|v| v.round() as i32).unwrap_or(1);
            // lil: 0 Normal, 1 Add, 2 Screen, 3 Multiply.
            let blend = match mode {
                0 => 2.0,
                2 => 3.0,
                3 => 1.0,
                _ => 0.0,
            };
            (blend, ctx.f(n.matcap_strength).unwrap_or(1.0))
        } else {
            (0.0, 1.0)
        };
        custom.insert("matcap_blend".into(), vec![blend]);
        custom.insert("matcap_strength".into(), vec![strength.clamp(0.0, 2.0)]);
        ctx.note(Level::Approx, "matcap blend/strength", format!("blend {} strength {strength:.2}", ["Add", "Multiply", "Replace", "Screen"][blend as usize]));
        let _ = ctx.color(n.matcap_color);
        if ctx.m.color_any(n.matcap_color).is_some_and(|c| c[..3] != [1.0; 3]) {
            ctx.note(Level::Unsupported, n.matcap_color.first().copied().unwrap_or("_MatcapColor"), "matcap tint (bake into the matcap texture)");
        }
        if is_mtoon {
            ctx.note(Level::Approx, "_SphereAdd", "MToon additive sphere texture -> matcap (Add)");
        }
    } else {
        ctx.mark(n.matcap_color);
        ctx.mark(n.matcap_mask);
        ctx.mark(n.matcap_strength);
    }

    // ---- emission ----
    let ecol = ctx.color(n.emission_color);
    let etex = ctx.tex(n.emission_tex);
    let emission_on = match ctx.on(n.emission_enable) {
        Some(b) => b,
        None => ecol.is_some_and(|c| c[..3] != [0.0; 3]) || m.has_keyword("_EMISSION") && etex.is_some(),
    } && (ecol.is_some_and(|c| c[..3] != [0.0; 3]) || etex.is_some());
    if emission_on {
        custom.insert("EMISSION".into(), vec![1.0]);
        file.features.emission = true;
        // HDR colours are stored linear by Unity.
        let c = ecol.unwrap_or([1.0; 4]);
        custom.insert("emission_color".into(), vec![c[0].max(0.0), c[1].max(0.0), c[2].max(0.0)]);
        file.params.emission_color = [c[0].max(0.0), c[1].max(0.0), c[2].max(0.0)];
        let strength = ctx.f(n.emission_strength).unwrap_or(1.0);
        custom.insert("emission_strength".into(), vec![strength.clamp(0.0, 20.0)]);
        file.params.emission_intensity = strength;
        ctx.note(Level::Mapped, n.emission_color[0], "emission colour (HDR, linear) and strength");
        if let Some((k, env)) = etex {
            ctx.use_tex(TexTarget::Emission, &k, &env, Level::Mapped, "Emission slot (sRGB)");
            file.params.emission_tiling = env.scale;
            file.params.emission_offset = env.offset;
        }
        if let Some((k, env)) = ctx.tex(n.emission_mask) {
            ctx.use_tex(TexTarget::Custom("emission_mask".into()), &k, &env, Level::Mapped, "emission_mask");
        }
        if let Some(c) = ctx.color(n.emission_scroll) {
            custom.insert("emission_scroll".into(), vec![c[0], c[1]]);
            ctx.note(Level::Mapped, n.emission_scroll[0], "emission scroll");
        }
    } else {
        for k in [n.emission_strength, n.emission_mask, n.emission_scroll] {
            ctx.mark(k);
        }
    }

    // ---- outline ----
    let width = ctx.f(n.outline_width).unwrap_or(0.0);
    let outline_on = match ctx.on(n.outline_enable) {
        Some(b) => b && width > 0.0,
        None if is_lil => {
            // lilToon draws outlines only through its Outline shader variants.
            let named = m.shader_name.as_deref().is_some_and(|s| s.to_ascii_lowercase().contains("outline"));
            let unknown_name = m.shader_name.is_none();
            if width > 0.0 && unknown_name {
                ctx.note(Level::Info, "_OutlineWidth", "outline enabled from the width alone (the shader name is not known)");
            }
            width > 0.0 && (named || unknown_name)
        }
        None => width > 0.0 && m.shader_name.as_deref().is_some_and(|s| s.to_ascii_lowercase().contains("outline")),
    };
    if outline_on {
        custom.insert("OUTLINE".into(), vec![1.0]);
        let factor = if is_poi { 5.0 } else { 1.0 };
        custom.insert("outline_width".into(), vec![(width * factor).clamp(0.0, 10.0)]);
        ctx.note(Level::Approx, n.outline_width[0], format!("width {width} x {factor} -> outline_width (units differ; check visually)"));
        let c = ctx.color(n.outline_color).unwrap_or([0.0, 0.0, 0.0, 1.0]);
        custom.insert("outline_color".into(), vec![srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2]), c[3]]);
    } else {
        ctx.mark(n.outline_color);
    }
    if is_poi && let Some(t) = ctx.on(&["_EnableOutlines"]) && !t {
        ctx.mark(&["_OutlineTexture", "_OutlineMask"]);
    }

    convert_toon_extras(ctx, file, &mut custom, fam);
    custom.insert("main_color".into(), main_color.to_vec());
    custom.insert("main_st".into(), st.to_vec());
    file.params.alpha_cutoff = file.params.alpha_cutoff.clamp(0.0, 1.0);

    // Features that have no mapping at all in this family but are on in the source.
    if is_poi {
        let flags: [(&str, &str); 5] = [
            ("_EnableGlitter", "glitter (set the ZestEngine glitter fields by hand)"),
            ("_GlitterEnable", "glitter (set the ZestEngine glitter fields by hand)"),
            ("_VertexManipulationLocalTranslation", "vertex manipulation"),
            ("_TPSPenetratorEnabled", "TPS"),
            ("_ParallaxHeightMap", "parallax"),
        ];
        for (k, what) in flags {
            let present = m.floats.get(k).is_some_and(|v| *v != 0.0) || m.textures.get(k).is_some_and(|t| t.guid.is_some());
            if present {
                ctx.note(Level::Unsupported, k, format!("{what}: see the gap list"));
            }
        }
    }
    file.custom = custom;
}

/// Transparent blend equation of a Toon-family material: Poiyomi's mode enum
/// (5, 6 additive; 7, 8 multiplicative), else the serialised blend factors.
fn toon_blend(ctx: &mut Ctx, fam: &UnityShader, alpha: AlphaMode) -> BlendMode {
    if alpha != AlphaMode::Blend {
        ctx.mark(&["_SrcBlend", "_DstBlend", "_SrcBlendFA", "_DstBlendFA"]);
        return BlendMode::Alpha;
    }
    if *fam == UnityShader::PoiyomiToon {
        match ctx.m.float("_Mode").map(|v| v.round() as i32) {
            Some(5 | 6) => {
                ctx.note(Level::Mapped, "_Mode", "additive -> Blend: Additive");
                return BlendMode::Additive;
            }
            Some(7 | 8) => {
                ctx.note(Level::Mapped, "_Mode", "multiplicative -> Blend: Multiply (2x multiplicative is plain multiply)");
                return BlendMode::Multiply;
            }
            _ => {}
        }
    }
    let src = ctx.f(&["_SrcBlend", "_BlendSrc"]).map(|v| v.round() as i32);
    let dst = ctx.f(&["_DstBlend", "_BlendDst"]).map(|v| v.round() as i32);
    if let (Some(src), Some(dst)) = (src, dst) {
        match blend_from_factors(src, dst) {
            Some(BlendMode::Alpha) | None => {}
            Some(b) => {
                ctx.note(Level::Mapped, "_SrcBlend/_DstBlend", format!("blend factors {src}/{dst} -> Blend: {b:?}"));
                return b;
            }
        }
    }
    BlendMode::Alpha
}

/// HSV adjust, second matcap / emission / shadow layer / main layer, dissolve,
/// decal, detail, reflections, refraction (grab), flipbook, panosphere,
/// AudioLink: the Toon features added for the Unity gaps (UNITY-3 .. 10). Names
/// are the serialised property names of the Unity shaders (see
/// docs/shaders-unity-compat.md; unverified against real exports).
fn convert_toon_extras(ctx: &mut Ctx, file: &mut MaterialFile, custom: &mut BTreeMap<String, Vec<f32>>, fam: &UnityShader) {
    let is_poi = *fam == UnityShader::PoiyomiToon;
    let is_lil = *fam == UnityShader::LilToon;
    let lin3 = |c: [f32; 4]| vec![srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2])];
    let on = |custom: &mut BTreeMap<String, Vec<f32>>, k: &str| {
        custom.insert(k.to_owned(), vec![1.0]);
    };

    // ---- hue / brightness ----
    if is_lil {
        // lilToon `_MainTexHSVG`: hue shift (-1..1 wraps), saturation, value, gamma.
        if let Some(v) = ctx.color(&["_MainTexHSVG"]) {
            if v[0] != 0.0 {
                custom.insert("hue_shift".into(), vec![v[0].rem_euclid(1.0)]);
                ctx.note(Level::Approx, "_MainTexHSVG", format!("hue {} -> hue_shift", v[0]));
            }
            if v[2] != 1.0 && v[2] > 0.0 {
                custom.insert("brightness".into(), vec![v[2].clamp(0.0, 2.0)]);
            }
            if v[1] != 1.0 && v[1] >= 0.0 {
                custom.insert("saturation".into(), vec![v[1].clamp(0.0, 2.0)]);
            }
        }
    } else if is_poi {
        if let Some(h) = ctx.f(&["_MainHueShift", "_Hue", "_HueShift"]).filter(|h| *h != 0.0) {
            custom.insert("hue_shift".into(), vec![h.rem_euclid(1.0)]);
            ctx.note(Level::Approx, "_MainHueShift", format!("{h} -> hue_shift (turns)"));
        }
        if let Some(b) = ctx.f(&["_MainBrightness", "_Brightness"]).filter(|b| *b != 1.0 && *b > 0.0) {
            custom.insert("brightness".into(), vec![b.clamp(0.0, 2.0)]);
            ctx.note(Level::Approx, "_MainBrightness", format!("{b} -> brightness"));
        }
    }

    // ---- shadow layer 3 (lilToon has three) ----
    if is_lil && let Some(c) = ctx.color(&["_Shadow3rdColor"]).filter(|c| c[3] > 0.0) {
        custom.insert("layer3_color".into(), lin3(c));
        custom.insert("layer3_strength".into(), vec![c[3].clamp(0.0, 1.0)]);
        if let Some(b) = ctx.f(&["_Shadow3rdBorder"]) {
            custom.insert("layer3_border".into(), vec![b.clamp(0.0, 1.0)]);
        }
        if let Some(b) = ctx.f(&["_Shadow3rdBlur"]) {
            custom.insert("layer3_blur".into(), vec![b.clamp(0.0, 0.5)]);
        }
        ctx.note(Level::Mapped, "_Shadow3rdColor", "shadow layer 3 colour / strength / border / blur");
    } else {
        ctx.mark(&["_Shadow3rdBorder", "_Shadow3rdBlur", "_Shadow3rdColor"]);
    }

    // ---- matcap 2 ----
    let (m2_on, m2_tex, m2_mask, m2_blend, m2_mode) = if is_lil {
        (&["_UseMatCap2nd"][..], &["_MatCap2ndTex"][..], &["_MatCap2ndBlendMask"][..], &["_MatCap2ndBlend"][..], &["_MatCap2ndBlendMode"][..])
    } else {
        (&["_Matcap2Enable"][..], &["_Matcap2"][..], &["_Matcap2Mask"][..], &["_Matcap2Intensity"][..], &["_Matcap2Blend"][..])
    };
    if let Some((k, env)) = ctx.tex(m2_tex)
        && ctx.on(m2_on).unwrap_or(true)
    {
        on(custom, "MATCAP2");
        ctx.use_tex(TexTarget::Custom("matcap2_tex".into()), &k, &env, Level::Mapped, "matcap2_tex");
        if let Some((mk, menv)) = ctx.tex(m2_mask) {
            ctx.use_tex(TexTarget::Custom("matcap2_mask".into()), &mk, &menv, Level::Mapped, "matcap2_mask");
        }
        let strength = ctx.f(m2_blend).unwrap_or(1.0);
        custom.insert("matcap2_strength".into(), vec![strength.clamp(0.0, 2.0)]);
        // ZestEngine Add 0, Multiply 1, Replace 2, Screen 3; lil: 0 normal, 1 add, 2 screen, 3 multiply.
        let blend = match ctx.f(m2_mode).map(|v| v.round() as i32) {
            Some(0) => 2.0,
            Some(2) => 3.0,
            Some(3) => 1.0,
            _ => 0.0,
        };
        custom.insert("matcap2_blend".into(), vec![blend]);
        ctx.note(Level::Approx, m2_tex[0], format!("second matcap, strength {strength:.2}, blend {}", ["Add", "Multiply", "Replace", "Screen"][blend as usize]));
    } else {
        ctx.mark(m2_mask);
        ctx.mark(m2_blend);
        ctx.mark(m2_mode);
    }

    // ---- emission 2 ----
    let (e2_on, e2_tex, e2_col, e2_mask) = if is_lil {
        (&["_UseEmission2nd"][..], &["_Emission2ndMap"][..], &["_Emission2ndColor"][..], &["_Emission2ndBlendMask"][..])
    } else {
        (&["_EnableEmission1"][..], &["_EmissionMap1"][..], &["_EmissionColor1"][..], &["_EmissionMask1"][..])
    };
    let e2c = ctx.color(e2_col);
    let e2t = ctx.tex(e2_tex);
    if ctx.on(e2_on).unwrap_or(false) && (e2t.is_some() || e2c.is_some_and(|c| c[..3] != [0.0; 3])) {
        on(custom, "EMISSION2");
        let c = e2c.unwrap_or([1.0; 4]);
        custom.insert("emission2_color".into(), vec![c[0].max(0.0), c[1].max(0.0), c[2].max(0.0)]);
        custom.insert("emission2_pulse_amount".into(), vec![0.0]);
        if let Some((k, env)) = e2t {
            ctx.use_tex(TexTarget::Custom("emission2_tex".into()), &k, &env, Level::Mapped, "emission2_tex");
        }
        ctx.mark(e2_mask);
        ctx.note(Level::Mapped, e2_col[0], "second emission colour (HDR, linear)");
        if ctx.m.tex_any(e2_mask).is_some() {
            ctx.note(Level::Unsupported, e2_mask[0], "second emission mask (bake it into the emission 2 texture alpha)");
        }
    } else {
        ctx.mark(e2_mask);
    }

    // ---- main 2nd layer (lilToon) as a full-UV decal ----
    if is_lil && ctx.on(&["_UseMain2ndTex"]).unwrap_or(false)
        && let Some((k, env)) = ctx.tex(&["_Main2ndTex"])
    {
        on(custom, "DECAL");
        ctx.use_tex(TexTarget::Custom("decal_tex".into()), &k, &env, Level::Approx, "decal_tex (full-UV second layer)");
        custom.insert("decal_position".into(), vec![0.5, 0.5]);
        custom.insert("decal_scale".into(), vec![1.0, 1.0]);
        if let Some(c) = ctx.color(&["_Color2nd"]) {
            custom.insert("decal_color".into(), vec![srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2]), c[3]]);
        }
        // lil: 0 normal, 1 add, 2 screen, 3 multiply; ZestEngine decal: Replace, Multiply, Add.
        let blend = match ctx.f(&["_Main2ndTexBlendMode"]).map(|v| v.round() as i32) {
            Some(1 | 2) => 2.0,
            Some(3) => 1.0,
            _ => 0.0,
        };
        custom.insert("decal_blend".into(), vec![blend]);
        ctx.note(Level::Approx, "_Main2ndTex", "second main layer -> full-UV decal (angle and masks are not carried)");
    } else {
        ctx.mark(&["_Color2nd", "_Main2ndTexBlendMode", "_Main2ndTexAngle"]);
    }

    // ---- detail normal (lil `_Bump2nd`) ----
    if is_lil && ctx.on(&["_UseBump2ndMap"]).unwrap_or(false)
        && let Some((k, env)) = ctx.tex(&["_Bump2ndMap"])
    {
        on(custom, "DETAIL");
        ctx.use_tex(TexTarget::Custom("detail_normal".into()), &k, &env, Level::Approx, "detail_normal");
        if let Some(sc) = ctx.f(&["_Bump2ndScale"]) {
            custom.insert("detail_normal_strength".into(), vec![sc.clamp(0.0, 2.0)]);
        }
        custom.insert("detail_albedo_strength".into(), vec![0.0]);
    } else {
        ctx.mark(&["_Bump2ndScale"]);
    }
    // ---- detail (Poiyomi) ----
    if is_poi && let (Some((k, env)), true) = (ctx.tex(&["_DetailTex"]), ctx.on(&["_DetailEnabled", "_EnableDetail"]).unwrap_or(true)) {
        on(custom, "DETAIL");
        ctx.use_tex(TexTarget::Custom("detail_albedo".into()), &k, &env, Level::Approx, "detail_albedo (x2 modulate)");
        if let Some((nk, nenv)) = ctx.tex(&["_DetailNormalMap"]) {
            ctx.use_tex(TexTarget::Custom("detail_normal".into()), &nk, &nenv, Level::Approx, "detail_normal");
        }
        if let Some((mk, menv)) = ctx.tex(&["_DetailMask"]) {
            ctx.use_tex(TexTarget::Custom("detail_mask".into()), &mk, &menv, Level::Mapped, "detail_mask");
        }
        if let Some(v) = ctx.f(&["_DetailTexIntensity"]) {
            custom.insert("detail_albedo_strength".into(), vec![v.clamp(0.0, 1.0)]);
        }
        if let Some(v) = ctx.f(&["_DetailNormalMapScale"]) {
            custom.insert("detail_normal_strength".into(), vec![v.clamp(0.0, 2.0)]);
        }
    }

    // ---- dissolve ----
    // Names checked against Poiyomi 8-10 `.mat` files: `_EnableDissolve`, `_DissolveAlpha`
    // (the amount), `_DissolveNoiseTexture`, `_DissolveEdgeWidth`, `_DissolveEdgeColor`,
    // `_DissolveEdgeEmission`.
    if is_poi && ctx.on(&["_EnableDissolve"]).unwrap_or(false) {
        on(custom, "DISSOLVE");
        if let Some((k, env)) = ctx.tex(&["_DissolveNoiseTexture"]) {
            ctx.use_tex(TexTarget::Custom("dissolve_noise".into()), &k, &env, Level::Mapped, "dissolve_noise");
        }
        if let Some(a) = ctx.f(&["_DissolveAlpha"]) {
            custom.insert("dissolve_amount".into(), vec![a.clamp(0.0, 1.0)]);
        }
        if let Some(w) = ctx.f(&["_DissolveEdgeWidth"]) {
            custom.insert("dissolve_edge_width".into(), vec![w.clamp(0.0, 0.25)]);
        }
        let glow = 1.0 + ctx.f(&["_DissolveEdgeEmission"]).unwrap_or(0.0).max(0.0);
        if let Some(c) = ctx.color(&["_DissolveEdgeColor"]) {
            custom.insert("dissolve_edge_color".into(), vec![c[0].max(0.0) * glow, c[1].max(0.0) * glow, c[2].max(0.0) * glow]);
        }
        ctx.note(Level::Approx, "_EnableDissolve", "dissolve: noise, amount, edge width / colour (+ edge emission as brightness); dissolve types, masks and detail noise are not carried");
    } else {
        ctx.mark(&["_DissolveAlpha", "_DissolveEdgeWidth", "_DissolveEdgeColor", "_DissolveEdgeEmission", "_DissolveNoiseTexture"]);
    }

    // ---- decal (Poiyomi) ----
    // `_DecalEnabled`, `_DecalTexture`, `_DecalColor`, `_DecalRotation` (degrees),
    // `_DecalPosition`, `_DecalScale` (vectors) match real Poiyomi 8-10 materials.
    if is_poi && ctx.on(&["_DecalEnabled"]).unwrap_or(false)
        && let Some((k, env)) = ctx.tex(&["_DecalTexture"])
    {
        on(custom, "DECAL");
        ctx.use_tex(TexTarget::Custom("decal_tex".into()), &k, &env, Level::Approx, "decal_tex");
        if let Some(c) = ctx.color(&["_DecalColor"]) {
            custom.insert("decal_color".into(), vec![srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2]), c[3]]);
        }
        if let Some(r) = ctx.f(&["_DecalRotation"]) {
            custom.insert("decal_rotation".into(), vec![r.rem_euclid(360.0)]);
        }
        if let Some(c) = ctx.color(&["_DecalPosition"]) {
            custom.insert("decal_position".into(), vec![c[0], c[1]]);
        }
        if let Some(c) = ctx.color(&["_DecalScale"]) {
            custom.insert("decal_scale".into(), vec![c[0], c[1]]);
        }
    }

    // ---- reflections / specular (lilToon `_UseReflection`) ----
    if is_lil && ctx.on(&["_UseReflection"]).unwrap_or(false) {
        on(custom, "REFLECTIONS");
        let smooth = ctx.f(&["_Smoothness"]).unwrap_or(1.0);
        custom.insert("reflection_roughness".into(), vec![(1.0 - smooth).clamp(0.0, 1.0)]);
        let metallic = ctx.f(&["_Metallic"]).unwrap_or(0.0);
        custom.insert("reflection_strength".into(), vec![(0.4 + metallic * 0.6).clamp(0.0, 2.0)]);
        ctx.note(Level::Approx, "_UseReflection", "reflection roughness = 1 - smoothness; strength follows metallic");
        if ctx.on(&["_ApplySpecular"]).unwrap_or(false) {
            on(custom, "SPECULAR");
            custom.insert("specular_mode".into(), vec![2.0]);
            custom.insert("roughness".into(), vec![(1.0 - smooth).clamp(0.0, 1.0)]);
            ctx.note(Level::Approx, "_ApplySpecular", "specular -> Realistic specular mode");
        }
    }

    // ---- refraction / grab pass ----
    let lil_refraction = is_lil && ctx.m.float("_TransparentMode").is_some_and(|v| v.round() as i32 == 3);
    // Poiyomi 9/10 serialise `_EnableGrabpass`, `_GrabpassColor`, `_GrabBlurDistance`,
    // `_GrabBlurPower`, `_GrabPassBrightness`, `_GrabPassBlendFactor`, `_GrabMode`.
    let poi_grab = is_poi && ctx.on(&["_EnableGrabpass"]).unwrap_or(false);
    if lil_refraction || poi_grab {
        on(custom, "GRAB");
        file.features.grab = true;
        if lil_refraction {
            let strength = ctx.f(&["_RefractionStrength"]).unwrap_or(0.1);
            custom.insert("grab_distortion".into(), vec![strength.abs().clamp(0.0, 1.0)]);
            let from_main = ctx.on(&["_RefractionColorFromMain"]).unwrap_or(false);
            if let Some(c) = ctx.color(&["_RefractionColor"]).filter(|_| !from_main) {
                custom.insert("grab_tint".into(), lin3(c));
            }
            ctx.mark(&["_RefractionFresnelPower"]);
            ctx.note(Level::Approx, "_RefractionStrength", format!("{strength} -> grab_distortion (screen-space offset by the surface normal; no fresnel curve)"));
        } else {
            let bright = ctx.f(&["_GrabPassBrightness"]).filter(|b| *b > 0.0).unwrap_or(1.0);
            if let Some(c) = ctx.color(&["_GrabpassColor"]) {
                let t = lin3(c);
                custom.insert("grab_tint".into(), vec![t[0] * bright, t[1] * bright, t[2] * bright]);
            }
            if let Some(b) = ctx.f(&["_GrabBlurDistance"]).filter(|b| *b > 0.0) {
                custom.insert("grab_blur".into(), vec![(b * 200.0).clamp(0.0, 32.0)]);
            }
            ctx.mark(&["_GrabBlurPower", "_GrabPassBlendFactor", "_GrabMode", "_GrabPassUseAlpha", "_GrabBlurQuality", "_GrabBlurDirections", "_GrabSrcBlend", "_GrabDstBlend"]);
            ctx.note(Level::Approx, "_EnableGrabpass", "grab pass -> Grab Pass (Refraction): tint x brightness and a blur radius of 200 px per unit of _GrabBlurDistance; the blend factor, blur quality and Poiyomi's own blend equations are not carried");
        }
        // The surface blends over the grabbed scene: a transparent draw.
        if file.features.alpha_mode == AlphaMode::Opaque {
            file.features.alpha_mode = AlphaMode::Blend;
        }
        file.features.blend = BlendMode::Alpha;
    } else {
        ctx.mark(&["_GrabpassColor", "_GrabBlurDistance", "_RefractionStrength", "_RefractionColor", "_RefractionColorFromMain", "_RefractionFresnelPower"]);
    }

    // ---- flipbook (Poiyomi) ----
    // The enable flag when serialised; else a flipbook texture alone means it is in use.
    // Real Poiyomi materials carry `_EnableFlipbook`, `_FlipbookTexArray` (a Texture2DArray
    // asset), `_FlipbookTotalFrames`, `_FlipbookFPS`, `_FlipbookCurrentFrame`,
    // `_FlipbookManualFrameControl`, `_FlipbookCrossfadeEnabled`, `_FlipbookColor` and the
    // `_FlipbookReplace` / `_FlipbookMultiply` / `_FlipbookAdd` blend flags; there is no
    // columns / rows property because the frames are array slices.
    let flipbook_on = ctx.on(&["_EnableFlipbook"]);
    if is_poi && flipbook_on.unwrap_or(false) {
        match ctx.tex(&["_FlipbookTexArray"]) {
            Some((k, env)) if env.guid.is_some() => {
                on(custom, "FLIPBOOK");
                ctx.use_tex(TexTarget::Custom("flip_tex".into()), &k, &env, Level::Approx, "flip_tex (a sprite sheet; a Texture2DArray asset cannot be read)");
                let frames = ctx.f(&["_FlipbookTotalFrames"]).unwrap_or(1.0).round().clamp(1.0, 256.0);
                // No grid in the material: a sheet is assumed to be one row of frames.
                let (cols, rows) = (frames.clamp(1.0, 16.0), 1.0);
                custom.insert("flip_frames".into(), vec![frames]);
                custom.insert("flip_cols".into(), vec![cols]);
                custom.insert("flip_rows".into(), vec![rows]);
                if let Some(fps) = ctx.f(&["_FlipbookFPS"]) {
                    custom.insert("flip_fps".into(), vec![fps.clamp(0.0, 60.0)]);
                }
                let manual = ctx.on(&["_FlipbookManualFrameControl"]).unwrap_or(false);
                if let Some(cur) = ctx.f(&["_FlipbookCurrentFrame"]).filter(|c| manual && *c >= 0.0) {
                    custom.insert("flip_manual".into(), vec![cur]);
                    custom.insert("flip_fps".into(), vec![0.0]);
                }
                custom.insert("flip_crossfade".into(), vec![f32::from(ctx.on(&["_FlipbookCrossfadeEnabled"]).unwrap_or(false))]);
                if let Some(c) = ctx.color(&["_FlipbookColor"]) {
                    custom.insert("flip_color".into(), vec![srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2]), c[3]]);
                }
                ctx.mark(&["_FlipbookScaleOffset", "_FlipbookReplace", "_FlipbookMultiply", "_FlipbookAdd", "_FlipbookBlendType", "_FlipbookEmissionStrength", "_FlipbookTiled", "_FlipbookStartFrame", "_FlipbookEndFrame"]);
            }
            _ => ctx.note(Level::Unsupported, "_EnableFlipbook", "flipbook without a texture (Poiyomi stores a Texture2DArray asset in _FlipbookTexArray, which cannot be read; export the frames as a sprite sheet)"),
        }
    }

    // ---- panosphere (Poiyomi) ----
    if is_poi && let Some((k, env)) = ctx.tex(&["_PanosphereTexture"]) && env.guid.is_some() && ctx.on(&["_PanoToggle"]).unwrap_or(true) {
        on(custom, "PANOSPHERE");
        ctx.use_tex(TexTarget::Custom("pano_tex".into()), &k, &env, Level::Mapped, "pano_tex (equirectangular)");
        if let Some(c) = ctx.color(&["_PanosphereColor"]) {
            custom.insert("pano_color".into(), vec![c[0].max(0.0), c[1].max(0.0), c[2].max(0.0)]);
        }
        // `_PanospherePan` is a pan vector (x = horizontal turns per second).
        if let Some(pan) = ctx.color(&["_PanospherePan"]).filter(|p| p[0] != 0.0) {
            custom.insert("pano_spin".into(), vec![pan[0].clamp(-1.0, 1.0)]);
        }
        ctx.mark(&["_PanoBlend", "_PanoEmission", "_PanoCubeMap", "_PanoCubeMapToggle", "_PanoMask", "_PanoUseBothEyes", "_PanoInfiniteStereoToggle"]);
        ctx.note(Level::Approx, "_PanosphereTexture", "panosphere, unlit (put it on an inside-out / Cull Front sphere)");
    }

    // ---- AudioLink (Poiyomi) ----
    // Real names (Poiyomi 8-10): master `_EnableAudioLink`; the "add" rows are vectors whose
    // green channel is the maximum added value (`_AudioLinkRimEmissionAdd`,
    // `_AudioLinkAddEmission`, `_AudioLinkEmission0CenterOut`) with a band index 0-3
    // (`..Band`: bass, low mid, high mid, treble).
    if is_poi && ctx.on(&["_EnableAudioLink"]).unwrap_or(false) {
        let mut any = false;
        let gain = |ctx: &mut Ctx, key: &str| ctx.color(&[key]).map_or(0.0, |c| c[1].max(c[0]));
        let band = |ctx: &mut Ctx, key: &str| ctx.f(&[key]).unwrap_or(0.0).round().clamp(0.0, 3.0);
        let e = gain(ctx, "_AudioLinkAddEmission").max(gain(ctx, "_AudioLinkEmission0CenterOut"));
        let eb = if ctx.m.float_any(&["_AudioLinkAddEmissionBand"]).is_some() && gain(ctx, "_AudioLinkAddEmission") > 0.0 {
            band(ctx, "_AudioLinkAddEmissionBand")
        } else {
            band(ctx, "_AudioLinkEmission0CenterOutBand")
        };
        if e > 0.0 {
            custom.insert("audio_emission".into(), vec![e.clamp(0.0, 20.0)]);
            custom.insert("audio_emission_band".into(), vec![eb]);
            any = true;
        }
        let r = gain(ctx, "_AudioLinkRimEmissionAdd").max(gain(ctx, "_AudioLinkRimBrightnessAdd"));
        if r > 0.0 {
            custom.insert("audio_rim".into(), vec![r.clamp(0.0, 4.0)]);
            custom.insert("audio_rim_band".into(), vec![band(ctx, "_AudioLinkRimEmissionBand")]);
            any = true;
        }
        if any {
            on(custom, "AUDIO");
            ctx.note(Level::Approx, "_EnableAudioLink", "AudioLink -> Audio Reactive: emission and rim gains by band (the engine's audio analysis drives the bands, not the AudioLink texture)");
        } else {
            ctx.note(Level::Info, "_EnableAudioLink", "AudioLink is on but no emission or rim gain is set; Audio Reactive left off");
        }
        ctx.mark(&["_AudioLinkEmission0CenterOutBand", "_AudioLinkAddEmissionBand", "_AudioLinkRimEmissionBand", "_AudioLinkRimBrightnessBand", "_AudioLinkRimWidthAdd", "_AudioLinkRimWidthBand", "_AudioLinkAnimToggle"]);
    }

    // ---- outline width from vertex colour (lilToon) ----
    if is_lil && custom.get("OUTLINE").is_some_and(|v| v[0] != 0.0) && ctx.f(&["_OutlineVertexR2Width"]).is_some_and(|v| v != 0.0) {
        custom.insert("outline_vertex_width".into(), vec![1.0]);
        ctx.note(Level::Mapped, "_OutlineVertexR2Width", "outline width from the red vertex colour -> outline_vertex_width");
    }
    if let Some((k, _)) = ctx.tex(&["_OutlineWidthMask"]) {
        ctx.note(Level::Unsupported, &k, "outline width mask texture (the outline hull is drawn in the vertex stage; use vertex colour width)");
    }
}

/// Mochie Screen FX (and Screen FX X) onto ZestEngine Screen FX. The property names are the
/// ones in the Screen FX prefab material shipped with Mochie's shaders (v1.65 - 1.75); the
/// meaning of each `*Model` enum beyond "0 = off" and the value ranges are not documented
/// anywhere the converter can read, so the amounts are approximations and say so.
fn convert_mochie_screenfx(ctx: &mut Ctx, file: &mut MaterialFile, opts: &ConvertOptions) {
    file.shader = opts.screenfx_shader.clone();
    let mut custom: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let on = |custom: &mut BTreeMap<String, Vec<f32>>, k: &str| {
        custom.insert(k.to_owned(), vec![1.0]);
    };
    for k in ["FILTER", "SHAKE", "DISTORT", "BLUR", "PIXELATE", "CHROMATIC", "NOISE"] {
        custom.insert(k.into(), vec![0.0]);
    }
    // The whole screen: Mochie's volume is the mesh around the camera.
    custom.insert("volume_shape".into(), vec![0.0]);
    custom.insert("GRAB".into(), vec![1.0]);
    file.features.grab = true;
    file.features.alpha_mode = AlphaMode::Blend;
    file.features.blend = BlendMode::Alpha;
    file.features.double_sided = false;
    file.features.cull_front = true;
    if let Some(o) = ctx.f(&["_Opacity"]) {
        custom.insert("fx_amount".into(), vec![o.clamp(0.0, 1.0)]);
    }
    // Filtering: toggled by `_FilterModel`; or by any value away from identity.
    let (hue, sat, con, bri) = (ctx.f(&["_Hue"]).unwrap_or(0.0), ctx.f(&["_Saturation"]).unwrap_or(1.0), ctx.f(&["_Contrast"]).unwrap_or(1.0), ctx.f(&["_Brightness"]).unwrap_or(1.0));
    let tint = ctx.color(&["_Color"]).filter(|c| c[..3] != [1.0; 3]);
    let invert = ctx.f(&["_Invert"]).unwrap_or(0.0);
    ctx.mark(&["_FilterModel", "_FilterStrength", "_HueMode", "_AutoShift", "_AutoShiftSpeed", "_InvertR", "_InvertG", "_InvertB", "_InvertX", "_InvertY", "_SaturationR", "_SaturationG", "_SaturationB", "_RGB", "_HDR", "_Value", "_Flavor", "_ClampToggle", "_ClampMax", "_ColorUseGlobal", "_ColorMinRange", "_ColorMaxRange"]);
    if hue != 0.0 || sat != 1.0 || con != 1.0 || bri != 1.0 || tint.is_some() || invert != 0.0 || ctx.m.float("_FilterModel").is_some_and(|v| v != 0.0) {
        on(&mut custom, "FILTER");
        custom.insert("filt_hue".into(), vec![hue.rem_euclid(1.0)]);
        custom.insert("filt_saturation".into(), vec![sat.clamp(0.0, 2.0)]);
        custom.insert("filt_contrast".into(), vec![con.clamp(0.0, 2.0)]);
        custom.insert("filt_brightness".into(), vec![bri.clamp(0.0, 2.0)]);
        if let Some(c) = tint {
            custom.insert("filt_tint".into(), vec![srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2])]);
        }
        custom.insert("filt_invert".into(), vec![invert.clamp(0.0, 1.0)]);
        ctx.note(Level::Mapped, "_Saturation/_Contrast/_Brightness/_Hue/_Color/_Invert", "filtering");
    }
    // Effects: each has a `*Model` selector (0 = off) and a strength.
    let model = |ctx: &mut Ctx, k: &str| ctx.f(&[k]).is_some_and(|v| v.round() as i32 != 0);
    if model(ctx, "_ShakeModel") {
        on(&mut custom, "SHAKE");
        let amp = ctx.f(&["_Amplitude"]).unwrap_or(0.5);
        custom.insert("shake_amount".into(), vec![(amp * 16.0).clamp(0.0, 64.0)]);
        let speed = ctx.f(&["_ShakeSpeedXY", "_ShakeSpeedX"]).unwrap_or(1.0);
        custom.insert("shake_speed".into(), vec![(speed * 5.0).clamp(0.0, 60.0)]);
        ctx.note(Level::Approx, "_ShakeModel", "shake: 16 px per unit of _Amplitude, speed x5");
    }
    ctx.mark(&["_ShakeModel", "_Amplitude", "_ShakeSpeedX", "_ShakeSpeedY", "_ShakeSpeedXY", "_ShakeNoiseTex", "_ShakeUseGlobal", "_ShakeMinRange", "_ShakeMaxRange"]);
    if model(ctx, "_DistortionModel") {
        on(&mut custom, "DISTORT");
        let st = ctx.f(&["_DistortionStr"]).unwrap_or(0.5);
        custom.insert("distort_amount".into(), vec![(st * 28.0).clamp(0.0, 100.0)]);
        if let Some(sp) = ctx.f(&["_DistortionSpeed"]) {
            custom.insert("distort_speed".into(), vec![sp.clamp(0.0, 10.0)]);
        }
        ctx.note(Level::Approx, "_DistortionModel", "distortion: 28 px per unit of _DistortionStr");
    }
    ctx.mark(&["_DistortionModel", "_DistortionStr", "_DistortionSpeed", "_DistortionRadius", "_DistortionFade", "_DistortionP2O", "_DistortionUseGlobal", "_DistortionMinRange", "_DistortionMaxRange"]);
    if model(ctx, "_BlurModel") {
        on(&mut custom, "BLUR");
        let r = ctx.f(&["_BlurRadius"]).unwrap_or(1.0);
        custom.insert("blur_radius".into(), vec![(r * 8.0).clamp(0.0, 32.0)]);
        ctx.note(Level::Approx, "_BlurModel", "blur: 8 px per unit of _BlurRadius; the blur kernel models are not carried");
    }
    ctx.mark(&["_BlurModel", "_BlurRadius", "_BlurStr", "_BlurSamples", "_BlurOpacity", "_BlurY", "_CrushBlur", "_BlurUseGlobal", "_BlurMinRange", "_BlurMaxRange"]);
    if let Some(px) = ctx.f(&["_PixelationStr"]).filter(|v| *v > 0.0) {
        on(&mut custom, "PIXELATE");
        custom.insert("pixel_size".into(), vec![(px * 32.0).clamp(1.0, 64.0)]);
        ctx.note(Level::Approx, "_PixelationStr", "pixelation: 32 px blocks per unit of strength");
    }
    ctx.mark(&["_PixelBlurSamples", "_PixelationStr"]);
    if let Some(n) = ctx.f(&["_Noise"]).filter(|v| v.round() as i32 != 0) {
        let _ = n;
        on(&mut custom, "NOISE");
        custom.insert("noise_amount".into(), vec![ctx.f(&["_NoiseStrength"]).unwrap_or(0.12).clamp(0.0, 1.0)]);
        custom.insert("noise_colour".into(), vec![f32::from(ctx.on(&["_NoiseRGB"]).unwrap_or(false))]);
        ctx.note(Level::Approx, "_Noise", "noise");
    }
    ctx.mark(&["_Noise", "_NoiseStrength", "_NoiseRGB", "_NoiseMode", "_NoiseUseGlobal", "_NoiseMinRange", "_NoiseMaxRange"]);
    if let Some(a) = ctx.f(&["_RGBSplit", "_ChromaticAberration"]).filter(|v| *v > 0.0) {
        on(&mut custom, "CHROMATIC");
        custom.insert("ca_amount".into(), vec![(a * 16.0).clamp(0.0, 32.0)]);
    }
    for (k, what) in [("_Fog", "fog"), ("_DoF", "depth of field"), ("_Zoom", "zoom"), ("_SST", "sprite sheet overlay"), ("_Letterbox", "letterbox"), ("_DepthBufferToggle", "depth-buffer visualisation"), ("_SobelFilterToggle", "Sobel outline filter"), ("_NMFToggle", "normal-map filter"), ("_TPToggle", "triplanar overlay")] {
        if ctx.m.floats.get(k).is_some_and(|v| *v != 0.0) {
            ctx.note(Level::Unsupported, k, format!("Screen FX {what}: not in ZestEngine Screen FX"));
            ctx.mark(&[k]);
        }
    }
    ctx.mark(&["_SrcBlend", "_DstBlend", "_BlendMode", "_CameraRenderMode", "_MirrorRenderMode", "_MaterialResetCheck", "_DisplayGlobalGizmo", "_AudioLinkToggle"]);
    ctx.note(Level::Info, "volume", "set to Everywhere (Mochie draws a mesh around the camera); switch it to Box / Sphere to confine the effect");
    file.custom = custom;
}

// ---------------------------------------------------------------------------
// PBR standard family (Unity Standard, URP Lit, Mochie Standard, generic)
// ---------------------------------------------------------------------------

fn convert_standard(ctx: &mut Ctx, file: &mut MaterialFile, fam: &UnityShader) -> OrmPlan {
    let m = ctx.m;
    file.shader = "standard".into();
    let mut orm = OrmPlan::default();

    let mut base = [1.0f32; 4];
    if let Some(c) = ctx.color(&["_Color", "_BaseColor", "_TintColor"]) {
        base = [srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2]), c[3]];
        ctx.note(Level::Mapped, "_Color", "colour (sRGB -> linear) -> base colour");
    }
    file.params.base_color = base;
    let mut albedo_guid = None;
    if let Some((k, env)) = ctx.tex(&["_MainTex", "_BaseMap"]) {
        file.params.albedo_tiling = env.scale;
        file.params.albedo_offset = env.offset;
        file.params.orm_tiling = env.scale;
        file.params.orm_offset = env.offset;
        albedo_guid = env.guid.clone();
        ctx.use_tex(TexTarget::Albedo, &k, &env, Level::Mapped, "Albedo slot (sRGB), tiling/offset");
    }
    let fam_mochie = matches!(fam, UnityShader::MochieStandard | UnityShader::MochieOther(_));
    file.features.alpha_mode = dominant_alpha(ctx, &UnityShader::UnityStandard);
    if let Some(c) = ctx.f(&["_Cutoff", "_AlphaCutoff"]) {
        file.params.alpha_cutoff = c.clamp(0.0, 1.0);
        ctx.note(Level::Mapped, "_Cutoff", "alpha cutoff");
    }
    (file.features.double_sided, file.features.cull_front) = cull_state(ctx, &["_Cull", "_CullMode", "_Culling"]);
    convert_stencil(ctx, file);
    ctx.mark(&["_SrcBlend", "_DstBlend", "_ZWrite", "_BlendOp"]);

    if let Some((k, env)) = ctx.tex(&["_BumpMap", "_NormalMap"]) {
        ctx.use_tex(TexTarget::Normal, &k, &env, Level::Mapped, "Normal slot (normal-map import)");
        file.features.normal_map = true;
        file.params.normal_tiling = env.scale;
        file.params.normal_offset = env.offset;
        file.params.normal_strength = ctx.f(&["_BumpScale", "_NormalStrength"]).unwrap_or(1.0);
    } else {
        ctx.mark(&["_BumpScale", "_NormalStrength"]);
    }

    // ---- metallic / smoothness / occlusion ----
    let metal_tex = ctx.tex(&["_MetallicGlossMap", "_MetallicMap", "_MetallicGlossMap"]);
    let rough_tex = ctx.tex(&["_RoughnessMap"]);
    let occ_tex = ctx.tex(&["_OcclusionMap"]);
    let scalar_metal = ctx.f(&["_Metallic"]);
    let smooth = ctx.f(&["_Glossiness", "_Smoothness"]);
    let gloss_scale = ctx.f(&["_GlossMapScale"]);
    let rough = ctx.f(&["_Roughness"]);
    if let Some(s) = ctx.f(&["_OcclusionStrength"]) {
        file.params.occlusion_strength = s.clamp(0.0, 1.0);
    }
    let smooth_src = ctx.f(&["_SmoothnessTextureChannel"]).map(|v| v.round() as i32).unwrap_or(0);

    if metal_tex.is_some() || rough_tex.is_some() || occ_tex.is_some() {
        orm.metallic_guid = metal_tex.as_ref().and_then(|(_, e)| e.guid.clone());
        orm.occlusion_guid = occ_tex.as_ref().and_then(|(_, e)| e.guid.clone());
        orm.albedo_guid = albedo_guid;
        orm.rough = Some(if let Some((_, e)) = &rough_tex {
            RoughSource::RoughnessMap(e.guid.clone().unwrap_or_default())
        } else if metal_tex.is_some() && smooth_src == 0 {
            RoughSource::MetallicAlpha(gloss_scale.or(smooth).unwrap_or(1.0))
        } else if smooth_src == 1 {
            RoughSource::AlbedoAlpha(gloss_scale.or(smooth).unwrap_or(1.0))
        } else {
            RoughSource::None
        });
        // The packed map carries the full values; scalars are 1 (map-driven) or the fallback.
        file.params.metallic = if metal_tex.is_some() { 1.0 } else { scalar_metal.unwrap_or(0.0) };
        file.params.roughness = match (&orm.rough, rough, smooth) {
            (Some(RoughSource::None), Some(r), _) => r,
            (Some(RoughSource::None), None, Some(s)) => 1.0 - s,
            (Some(RoughSource::None), None, None) => 0.7,
            _ => 1.0,
        };
        let which: Vec<&str> = [metal_tex.as_ref().map(|_| "metallic (R)"), rough_tex.as_ref().map(|_| "roughness (R)"), occ_tex.as_ref().map(|_| "occlusion (G)")].into_iter().flatten().collect();
        ctx.note(Level::Approx, "metallic/smoothness/occlusion maps", format!("repacked into one ORM texture (R occlusion, G roughness = 1 - smoothness, B metallic) from {}", which.join(", ")));
    } else {
        file.params.metallic = scalar_metal.unwrap_or(0.0);
        file.params.roughness = match (rough, smooth) {
            (Some(r), _) => r,
            (None, Some(s)) => 1.0 - s,
            _ => 0.7,
        };
        if scalar_metal.is_some() || smooth.is_some() || rough.is_some() {
            ctx.note(Level::Mapped, "_Metallic/_Glossiness", "scalar metallic / roughness (1 - smoothness)");
        }
    }
    file.params.metallic = file.params.metallic.clamp(0.0, 1.0);
    file.params.roughness = file.params.roughness.clamp(0.0, 1.0);
    if matches!(fam, UnityShader::UnityStandardSpecular) {
        ctx.note(Level::Approx, "_SpecColor", "specular setup treated as a dielectric/metallic material; check reflections");
        ctx.mark(&["_SpecColor", "_SpecGlossMap"]);
    }

    // ---- emission ----
    let ecol = ctx.color(&["_EmissionColor"]);
    let etex = ctx.tex(&["_EmissionMap"]);
    let lit = ecol.is_some_and(|c| c[..3] != [0.0; 3]);
    if lit || etex.is_some() && m.has_keyword("_EMISSION") {
        file.features.emission = true;
        let c = ecol.unwrap_or([1.0; 4]);
        file.params.emission_color = [c[0].max(0.0), c[1].max(0.0), c[2].max(0.0)];
        file.params.emission_intensity = 1.0;
        ctx.note(Level::Mapped, "_EmissionColor", "emission colour (HDR, linear)");
        if let Some((k, env)) = etex {
            file.params.emission_tiling = env.scale;
            file.params.emission_offset = env.offset;
            ctx.use_tex(TexTarget::Emission, &k, &env, Level::Mapped, "Emission slot (sRGB)");
        }
    }

    // ---- height ----
    if let Some((k, env)) = ctx.tex(&["_ParallaxMap", "_HeightMap"]) {
        let scale = ctx.f(&["_Parallax", "_HeightScale"]).unwrap_or(0.02);
        file.params.displacement_scale = scale.clamp(0.0, 0.15);
        ctx.use_tex(TexTarget::Displacement, &k, &env, Level::Approx, "Displacement slot (parallax)");
    } else {
        ctx.mark(&["_Parallax", "_HeightScale"]);
    }
    convert_standard_detail(ctx, file);
    convert_standard_translucency(ctx, file);
    if fam_mochie {
        for (k, what) in [("_SSR", "screen-space reflections"), ("_Reflections", "reflection options"), ("_SubsurfaceScattering", "subsurface scattering"), ("_RainMode", "rain / puddles"), ("_Wind", "wind")] {
            if m.floats.get(k).is_some_and(|v| *v != 0.0) {
                ctx.note(Level::Unsupported, k, format!("{what}: see the gap list"));
            }
        }
    }
    orm
}

/// Detail layer of the Unity Standard / URP Lit (`_DetailAlbedoMap` x2, `_DetailNormalMap`,
/// `_DetailMask` alpha) and HDRP Lit (`_DetailMap`: R albedo, AG normal, B smoothness).
fn convert_standard_detail(ctx: &mut Ctx, file: &mut MaterialFile) {
    let alb = ctx.tex(&["_DetailAlbedoMap"]).filter(|(_, e)| e.guid.is_some());
    let nrm = ctx.tex(&["_DetailNormalMap"]).filter(|(_, e)| e.guid.is_some());
    let hdrp = ctx.tex(&["_DetailMap"]).filter(|(_, e)| e.guid.is_some());
    let mask = ctx.tex(&["_DetailMask"]).filter(|(_, e)| e.guid.is_some());
    let albedo_scale = ctx.f(&["_DetailAlbedoMapScale", "_DetailAlbedoScale"]);
    let normal_scale = ctx.f(&["_DetailNormalMapScale", "_DetailNormalScale"]);
    let smooth_scale = ctx.f(&["_DetailSmoothnessScale"]);
    ctx.mark(&["_UVSec", "_UVDetail", "_LinkDetailsWithBase"]);
    if alb.is_none() && nrm.is_none() && hdrp.is_none() {
        return;
    }
    let d = &mut file.params.detail;
    let first = alb.as_ref().or(nrm.as_ref()).or(hdrp.as_ref()).map(|(_, e)| e.clone());
    if let Some(e) = &first {
        d.tiling = e.scale;
        d.offset = e.offset;
    }
    d.blend = 0.0; // Multiply x2, Unity's behaviour
    if let Some((k, env)) = alb {
        d.albedo_strength = albedo_scale.unwrap_or(1.0).clamp(0.0, 1.0);
        ctx.use_tex(TexTarget::DetailAlbedo, &k, &env, Level::Mapped, "Detail Albedo (x2 blend)");
    }
    if let Some((k, env)) = nrm {
        d.normal_strength = normal_scale.unwrap_or(1.0).clamp(0.0, 2.0);
        ctx.use_tex(TexTarget::DetailNormal, &k, &env, Level::Mapped, "Detail Normal (normal-map import)");
    }
    if let Some((k, env)) = hdrp
        && let Some(g) = env.guid.clone()
    {
        d.albedo_strength = albedo_scale.unwrap_or(1.0).clamp(0.0, 1.0);
        d.normal_strength = normal_scale.unwrap_or(1.0).clamp(0.0, 2.0);
        d.orm_strength = smooth_scale.unwrap_or(1.0).clamp(0.0, 1.0);
        ctx.repacks.push(RepackJob { kind: RepackKind::HdrpAlbedo, guid: g.clone(), unity_prop: k.clone() });
        ctx.repacks.push(RepackJob { kind: RepackKind::HdrpNormal, guid: g.clone(), unity_prop: k.clone() });
        ctx.repacks.push(RepackJob { kind: RepackKind::HdrpSmoothness, guid: g, unity_prop: k.clone() });
        ctx.note(Level::Approx, &k, "HDRP packed detail map split into detail albedo (R), normal (A, G) and ORM (B smoothness); smoothness adds as a multiplier on roughness");
    }
    if let Some((k, env)) = mask
        && let Some(g) = env.guid
    {
        ctx.repacks.push(RepackJob { kind: RepackKind::MaskFromAlpha, guid: g, unity_prop: k.clone() });
        ctx.note(Level::Approx, &k, "detail mask (alpha channel) -> Detail Mask (red)");
    }
}

/// HDRP Lit translucent / subsurface materials: `_ThicknessMap` (R, bright = thick),
/// `_TransmissionTint`, `_MaterialID` 0 (subsurface) or 5 (translucent). The HDRP
/// names are from its material serialisation as documented; unverified here.
fn convert_standard_translucency(ctx: &mut Ctx, file: &mut MaterialFile) {
    let id = ctx.f(&["_MaterialID"]).map(|v| v.round() as i32);
    let thick = ctx.tex(&["_ThicknessMap"]);
    let tint = ctx.color(&["_TransmissionTint"]);
    ctx.mark(&["_ThicknessRemap", "_Thickness", "_DiffusionProfile", "_DiffusionProfileHash", "_DiffusionProfileAsset"]);
    let on = matches!(id, Some(0 | 5)) || thick.as_ref().is_some_and(|(_, e)| e.guid.is_some());
    if !on {
        return;
    }
    let t = &mut file.params.transmission;
    t.translucency = 1.0;
    t.map_is_thickness = true;
    if let Some(c) = tint {
        t.translucency_color = [srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2])];
    }
    if let Some((k, env)) = thick {
        ctx.use_tex(TexTarget::Translucency, &k, &env, Level::Approx, "Translucency Map (thickness: bright = thick)");
    }
    ctx.note(Level::Approx, "_MaterialID", "HDRP subsurface / translucent -> thin back-lit translucency (no diffusion profile; thickness remap ignored)");
}

// ---------------------------------------------------------------------------
// Unity project: GUID map
// ---------------------------------------------------------------------------

/// A folder of Unity assets indexed by GUID through the `.meta` files.
#[derive(Clone, Debug, Default)]
pub struct UnityAssets {
    pub root: PathBuf,
    pub guid_path: BTreeMap<String, PathBuf>,
    /// Shader GUID -> declared name (from `.shader` files in the folder).
    pub shader_names: BTreeMap<String, String>,
}

impl UnityAssets {
    /// Index `root` recursively (depth-limited). Never fails: unreadable
    /// entries are skipped.
    pub fn scan(root: &Path) -> Self {
        let mut a = Self { root: root.to_owned(), ..Self::default() };
        a.walk(root, 0);
        a
    }

    fn walk(&mut self, dir: &Path, depth: u32) {
        if depth > 24 {
            return;
        }
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                self.walk(&p, depth + 1);
                continue;
            }
            let Some(name) = p.file_name().and_then(|n| n.to_str()) else { continue };
            let Some(asset_name) = name.strip_suffix(".meta") else { continue };
            let asset = p.with_file_name(asset_name);
            if !asset.is_file() {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&p) else { continue };
            let Some(guid) = parse_meta_guid(&text) else { continue };
            if asset_name.to_ascii_lowercase().ends_with(".shader")
                && let Ok(src) = std::fs::read_to_string(&asset)
                && let Some(n) = shader_declared_name(&src)
            {
                self.shader_names.insert(guid.clone(), n);
            }
            self.guid_path.insert(guid, asset);
        }
    }

    pub fn path_of(&self, guid: &str) -> Option<&Path> {
        self.guid_path.get(&guid.to_ascii_lowercase()).map(PathBuf::as_path)
    }

    /// Attach the shader name the material's shader GUID resolves to.
    pub fn name_shader(&self, m: UnityMaterial) -> UnityMaterial {
        let name = m.shader_guid.as_ref().and_then(|g| self.shader_names.get(g)).cloned();
        m.with_shader_name(name)
    }
}

// ---------------------------------------------------------------------------
// Finishing: textures, ORM packing, writing
// ---------------------------------------------------------------------------

/// Image extensions the engine's loader reads (Unity's PSD / TIFF / DDS are not
/// among them and are reported).
fn loadable(p: &Path) -> bool {
    p.extension().and_then(|e| e.to_str()).is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg" | "tga" | "bmp" | "exr" | "hdr" | "gif" | "webp"))
}

/// Pack Unity metallic / occlusion / smoothness inputs into one ORM image
/// (R occlusion, G roughness, B metallic). Missing inputs give 1 (no occlusion),
/// the fallback roughness 1 and metallic 0 (the material scalars scale it).
pub fn pack_orm(metallic: Option<&image::RgbaImage>, occlusion: Option<&image::RgbaImage>, rough: RoughPixels<'_>) -> image::RgbaImage {
    let dims = [metallic.map(|i| i.dimensions()), occlusion.map(|i| i.dimensions()), rough.dims()];
    let (w, h) = dims.iter().flatten().fold((1u32, 1u32), |a, d| (a.0.max(d.0), a.1.max(d.1)));
    let sample = |img: &image::RgbaImage, x: u32, y: u32| -> [u8; 4] {
        let (iw, ih) = img.dimensions();
        img.get_pixel(x * iw / w, y * ih / h).0
    };
    let mut out = image::RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let ao = occlusion.map_or(255, |i| sample(i, x, y)[1]);
            let r = match &rough {
                RoughPixels::None => 255,
                RoughPixels::SmoothAlpha(img, s) => 255 - ((sample(img, x, y)[3] as f32 * s).clamp(0.0, 255.0)) as u8,
                RoughPixels::Roughness(img) => sample(img, x, y)[0],
            };
            let mt = metallic.map_or(0, |i| sample(i, x, y)[0]);
            out.put_pixel(x, y, image::Rgba([ao, r, mt, 255]));
        }
    }
    out
}

pub enum RoughPixels<'a> {
    None,
    /// Smoothness in this image's alpha, times the scale.
    SmoothAlpha(&'a image::RgbaImage, f32),
    Roughness(&'a image::RgbaImage),
}

impl RoughPixels<'_> {
    fn dims(&self) -> Option<(u32, u32)> {
        match self {
            Self::None => None,
            Self::SmoothAlpha(i, _) | Self::Roughness(i) => Some(i.dimensions()),
        }
    }
}

/// Result of finishing one material.
#[derive(Clone, Debug)]
pub struct Converted {
    pub name: String,
    pub material_path: PathBuf,
    pub report_path: PathBuf,
    pub report: Report,
}

fn copy_texture(src: &Path, project_root: &Path, tex_dir: &str, taken: &mut BTreeMap<PathBuf, PathBuf>) -> Result<PathBuf> {
    if let Some(done) = taken.get(src) {
        return Ok(done.clone());
    }
    let dir = project_root.join(tex_dir);
    std::fs::create_dir_all(&dir)?;
    let name = src.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "texture.png".into());
    let (stem, ext) = match name.rsplit_once('.') {
        Some((s, e)) => (s.to_owned(), e.to_owned()),
        None => (name.clone(), "png".into()),
    };
    let src_len = std::fs::metadata(src).map(|m| m.len()).ok();
    let mut dest = dir.join(&name);
    let mut n = 0;
    // Reuse an identical file already there; otherwise find a free name.
    while dest.exists() && std::fs::metadata(&dest).map(|m| m.len()).ok() != src_len {
        n += 1;
        dest = dir.join(format!("{stem}_{n}.{ext}"));
    }
    if !dest.exists() {
        std::fs::copy(src, &dest).with_context(|| format!("copying {}", src.display()))?;
    }
    taken.insert(src.to_owned(), dest.clone());
    Ok(dest)
}

fn rel(project_root: &Path, p: &Path) -> PathBuf {
    PathBuf::from(crate::material_remap::project_relative(project_root, p))
}

/// Resolve the plan's textures through `assets`, copy them into the project,
/// write `<out_dir>/<name>.material` and `<name>.convert.txt`.
fn srgb_encode(l: f32) -> u8 {
    let l = l.clamp(0.0, 1.0);
    let c = if l <= 0.003_130_8 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 };
    (c * 255.0 + 0.5) as u8
}

/// Unity Standard's detail mask is its alpha channel; the engine reads red.
pub fn repack_mask_from_alpha(img: &image::RgbaImage) -> image::RgbaImage {
    let mut out = img.clone();
    for p in out.pixels_mut() {
        let a = p[3];
        *p = image::Rgba([a, a, a, 255]);
    }
    out
}

/// HDRP's detail albedo is the red channel read as linear with 0.5 neutral
/// (`albedo *= 2 * detail`). The engine's x2 blend decodes sRGB and multiplies by
/// 4.5948 (neutral = grey 128), so the value is scaled by 2 / 4.5948 and sRGB-encoded.
pub fn repack_hdrp_detail_albedo(img: &image::RgbaImage) -> image::RgbaImage {
    let mut out = img.clone();
    for p in out.pixels_mut() {
        let v = srgb_encode(p[0] as f32 / 255.0 * (2.0 / 4.594_794));
        *p = image::Rgba([v, v, v, 255]);
    }
    out
}

/// HDRP packs the detail normal in alpha (x) and green (y); rebuild z.
pub fn repack_hdrp_detail_normal(img: &image::RgbaImage) -> image::RgbaImage {
    let mut out = img.clone();
    for p in out.pixels_mut() {
        let (x, y) = (p[3] as f32 / 255.0 * 2.0 - 1.0, p[1] as f32 / 255.0 * 2.0 - 1.0);
        let z = (1.0 - x * x - y * y).max(0.0).sqrt();
        *p = image::Rgba([p[3], p[1], ((z * 0.5 + 0.5) * 255.0 + 0.5) as u8, 255]);
    }
    out
}

/// HDRP's detail smoothness (blue) as a detail ORM that multiplies the base
/// roughness: green = 1 - smoothness, occlusion and metallic left alone.
pub fn repack_hdrp_detail_orm(img: &image::RgbaImage) -> image::RgbaImage {
    let mut out = img.clone();
    for p in out.pixels_mut() {
        *p = image::Rgba([255, 255 - p[2], 255, 255]);
    }
    out
}

pub fn finish_plan(mut plan: Plan, assets: &UnityAssets, project_root: &Path, out_dir: &Path, opts: &ConvertOptions) -> Result<Converted> {
    let mut copied: BTreeMap<PathBuf, PathBuf> = BTreeMap::new();
    let mut notes: Vec<ReportLine> = Vec::new();
    let resolve = |guid: &str, prop: &str, notes: &mut Vec<ReportLine>| -> Option<PathBuf> {
        let Some(src) = assets.path_of(guid) else {
            notes.push(ReportLine { level: Level::Unsupported, property: prop.to_owned(), note: format!("texture guid {guid} not found in the Unity folder") });
            return None;
        };
        if !loadable(src) {
            notes.push(ReportLine {
                level: Level::Unsupported,
                property: prop.to_owned(),
                note: format!("{} is not a format the engine loads (convert it to png)", src.display()),
            });
            return None;
        }
        Some(src.to_owned())
    };
    for t in std::mem::take(&mut plan.textures) {
        let Some(src) = resolve(&t.guid, &t.unity_prop, &mut notes) else { continue };
        let dest = copy_texture(&src, project_root, &opts.texture_dir, &mut copied)?;
        let slot = match &t.target {
            TexTarget::Albedo => Some(0),
            TexTarget::Normal => Some(1),
            TexTarget::Emission => Some(3),
            TexTarget::Displacement => Some(15),
            TexTarget::DetailAlbedo => Some(0),
            TexTarget::DetailNormal => Some(1),
            TexTarget::DetailOrm => Some(2),
            TexTarget::DetailMask | TexTarget::Translucency => Some(4),
            TexTarget::Custom(_) => None,
        };
        if let Some(s) = slot {
            let _ = crate::ensure_first_import_for_slot(&dest, s);
        }
        let r = rel(project_root, &dest);
        let tx = &mut plan.file.textures;
        match t.target {
            TexTarget::Albedo => tx.albedo = Some(r),
            TexTarget::Normal => tx.normal = Some(r),
            TexTarget::Emission => tx.emission = Some(r),
            TexTarget::Displacement => tx.displacement = Some(r),
            TexTarget::DetailAlbedo => tx.detail_albedo = Some(r),
            TexTarget::DetailNormal => tx.detail_normal = Some(r),
            TexTarget::DetailOrm => tx.detail_orm = Some(r),
            TexTarget::DetailMask => tx.detail_mask = Some(r),
            TexTarget::Translucency => tx.translucency = Some(r),
            TexTarget::Custom(name) => {
                tx.shader.insert(name, r);
            }
        }
    }
    for job in std::mem::take(&mut plan.repacks) {
        let Some(src) = resolve(&job.guid, &job.unity_prop, &mut notes) else { continue };
        let Ok(img) = crate::material_file::open_image_compat(&src).map(|i| i.into_rgba8()) else {
            notes.push(ReportLine { level: Level::Unsupported, property: job.unity_prop.clone(), note: format!("could not read {}", src.display()) });
            continue;
        };
        let (out, suffix, slot) = match job.kind {
            RepackKind::MaskFromAlpha => (repack_mask_from_alpha(&img), "detail_mask", 4),
            RepackKind::HdrpAlbedo => (repack_hdrp_detail_albedo(&img), "detail_albedo", 0),
            RepackKind::HdrpNormal => (repack_hdrp_detail_normal(&img), "detail_normal", 1),
            RepackKind::HdrpSmoothness => (repack_hdrp_detail_orm(&img), "detail_orm", 2),
        };
        let dir = project_root.join(&opts.texture_dir);
        std::fs::create_dir_all(&dir)?;
        let path = crate::material_remap::unique_file(&dir, &format!("{}_{suffix}", crate::material_remap::file_stem_of(&plan.name, "material")), "png");
        out.save(&path).with_context(|| format!("writing {}", path.display()))?;
        let _ = crate::ensure_first_import_for_slot(&path, slot);
        let r = rel(project_root, &path);
        let tx = &mut plan.file.textures;
        match job.kind {
            RepackKind::MaskFromAlpha => tx.detail_mask = Some(r),
            RepackKind::HdrpAlbedo => tx.detail_albedo = Some(r),
            RepackKind::HdrpNormal => tx.detail_normal = Some(r),
            RepackKind::HdrpSmoothness => tx.detail_orm = Some(r),
        }
        notes.push(ReportLine { level: Level::Info, property: job.unity_prop.clone(), note: format!("rewrote the channels into {}", path.display()) });
    }
    if !plan.orm.is_empty() {
        let load = |guid: &Option<String>, prop: &str, notes: &mut Vec<ReportLine>| -> Option<image::RgbaImage> {
            let src = resolve(guid.as_deref()?, prop, notes)?;
            crate::material_file::open_image_compat(&src).ok().map(|i| i.into_rgba8())
        };
        let metal = load(&plan.orm.metallic_guid, "metallic map", &mut notes);
        let occ = load(&plan.orm.occlusion_guid, "occlusion map", &mut notes);
        let alb_img;
        let rough_img;
        let rough = match plan.orm.rough.clone() {
            Some(RoughSource::MetallicAlpha(s)) => metal.as_ref().map_or(RoughPixels::None, |i| RoughPixels::SmoothAlpha(i, s)),
            Some(RoughSource::AlbedoAlpha(s)) => {
                alb_img = load(&plan.orm.albedo_guid, "albedo (smoothness alpha)", &mut notes);
                alb_img.as_ref().map_or(RoughPixels::None, |i| RoughPixels::SmoothAlpha(i, s))
            }
            Some(RoughSource::RoughnessMap(g)) => {
                rough_img = load(&Some(g), "roughness map", &mut notes);
                rough_img.as_ref().map_or(RoughPixels::None, RoughPixels::Roughness)
            }
            _ => RoughPixels::None,
        };
        if metal.is_some() || occ.is_some() || !matches!(rough, RoughPixels::None) {
            let img = pack_orm(metal.as_ref(), occ.as_ref(), rough);
            let dir = project_root.join(&opts.texture_dir);
            std::fs::create_dir_all(&dir)?;
            let path = crate::material_remap::unique_file(&dir, &format!("{}_orm", crate::material_remap::file_stem_of(&plan.name, "material")), "png");
            img.save(&path).with_context(|| format!("writing {}", path.display()))?;
            let _ = crate::ensure_first_import_for_slot(&path, 2);
            plan.file.textures.orm = Some(rel(project_root, &path));
            notes.push(ReportLine { level: Level::Info, property: "orm".into(), note: format!("wrote {}", path.display()) });
        }
    }
    plan.report.lines.extend(notes);
    plan.report.lines.sort_by_key(|l| l.level);

    let stem = crate::material_remap::file_stem_of(&plan.name, "material");
    let material_path = crate::material_remap::unique_file(out_dir, &stem, "material");
    save_material_file(&material_path, &plan.file)?;
    let report_path = material_path.with_extension("convert.txt");
    std::fs::write(&report_path, plan.report.to_text()).with_context(|| format!("writing {}", report_path.display()))?;
    Ok(Converted { name: plan.name, material_path, report_path, report: plan.report })
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Summary of a conversion run (one file or a folder).
#[derive(Clone, Debug, Default)]
pub struct ConvertSummary {
    pub converted: Vec<Converted>,
    /// (file, why) for files that could not be read / had no Material.
    pub failed: Vec<(PathBuf, String)>,
}

impl ConvertSummary {
    pub fn message(&self) -> String {
        let unsupported: usize = self.converted.iter().map(|c| c.report.count(Level::Unsupported)).sum();
        let unmapped: usize = self.converted.iter().map(|c| c.report.count(Level::Unmapped)).sum();
        format!(
            "Converted {} Unity material(s), {} failed; {unsupported} unsupported and {unmapped} unmapped properties (see the .convert.txt files)",
            self.converted.len(),
            self.failed.len()
        )
    }
}

/// Convert every Material in one `.mat` file. `assets` indexes the Unity
/// folder the file lives in (so GUIDs resolve).
pub fn convert_mat_file(mat: &Path, assets: &UnityAssets, project_root: &Path, out_dir: &Path, opts: &ConvertOptions, summary: &mut ConvertSummary) {
    let bytes = match std::fs::read(mat) {
        Ok(b) => b,
        Err(e) => {
            summary.failed.push((mat.to_owned(), e.to_string()));
            return;
        }
    };
    if looks_binary(&bytes) {
        summary.failed.push((mat.to_owned(), "binary-serialised .mat (set Unity's Asset Serialization to Force Text and re-export)".into()));
        return;
    }
    let mats = parse_unity_materials(&String::from_utf8_lossy(&bytes));
    if mats.is_empty() {
        summary.failed.push((mat.to_owned(), "no Material object found".into()));
        return;
    }
    for mut m in mats {
        if m.name.is_empty() {
            m.name = mat.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        }
        let m = assets.name_shader(m);
        let shader = identify_shader(&m);
        let plan = convert_material(&m, shader, opts);
        match finish_plan(plan, assets, project_root, out_dir, opts) {
            Ok(c) => summary.converted.push(c),
            Err(e) => summary.failed.push((mat.to_owned(), format!("{e:#}"))),
        }
    }
}

/// Convert one `.mat` (indexing its own folder tree for GUIDs; pass a wider
/// `unity_root` when textures live elsewhere in the Unity project).
pub fn convert_unity_material(mat: &Path, unity_root: Option<&Path>, project_root: &Path, out_dir: &Path, opts: &ConvertOptions) -> ConvertSummary {
    let root = unity_root.or_else(|| mat.parent()).unwrap_or(Path::new("."));
    let assets = UnityAssets::scan(root);
    let mut summary = ConvertSummary::default();
    convert_mat_file(mat, &assets, project_root, out_dir, opts, &mut summary);
    summary
}

/// Convert every `.mat` under `folder`. `progress(done, total, current)` is
/// called before each file; return false to cancel.
pub fn convert_unity_folder(folder: &Path, project_root: &Path, out_dir: &Path, opts: &ConvertOptions, progress: &mut dyn FnMut(usize, usize, &str) -> bool) -> ConvertSummary {
    let assets = UnityAssets::scan(folder);
    let mut mats = Vec::new();
    collect_mats(folder, &mut mats, 0);
    mats.sort();
    let mut summary = ConvertSummary::default();
    let total = mats.len();
    for (i, mat) in mats.iter().enumerate() {
        let label = mat.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if !progress(i, total, &label) {
            break;
        }
        convert_mat_file(mat, &assets, project_root, out_dir, opts, &mut summary);
    }
    summary
}

fn collect_mats(dir: &Path, out: &mut Vec<PathBuf>, depth: u32) {
    if depth > 24 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_mats(&p, out, depth + 1);
        } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("mat")) {
            out.push(p);
        }
    }
}

/// Record converted materials in a model import's remap table (by Unity
/// material name == the model's source material name), so an imported FBX /
/// glTF picks them up. Existing entries are kept.
pub fn add_to_model_remap(model: &mut crate::ModelImport, project_root: &Path, converted: &[Converted]) -> usize {
    let mut added = 0;
    for c in converted {
        if !model.material_remap.contains_key(&c.name) {
            model.material_remap.insert(c.name.clone(), crate::material_remap::project_relative(project_root, &c.material_path));
            added += 1;
        }
    }
    added
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unity_mat::fixtures::{MOCHIE, POIYOMI};

    fn plan_of(text: &str, idx: usize, name: Option<&str>) -> Plan {
        let m = parse_unity_materials(text)[idx].clone().with_shader_name(name.map(str::to_owned));
        let s = identify_shader(&m);
        convert_material(&m, s, &ConvertOptions::default())
    }

    fn c(p: &Plan, k: &str) -> Vec<f32> {
        p.file.custom.get(k).cloned().unwrap_or_default()
    }

    /// Every value / texture name the converter writes must be a real property of
    /// shaders/zestengine_toon.frag, with the right component count.
    #[test]
    fn toon_output_matches_the_shader_declarations() {
        let body = include_str!("../../../shaders/zestengine_toon.frag");
        let src = crate::parse_shader_source(body, "zestengine_toon").unwrap();
        let rich = "Material:\n  m_Name: R\n  m_SavedProperties:\n    m_Floats:\n    - _UseShadow: 1\n    - _UseRim: 1\n    - _UseMatCap: 1\n    - _UseEmission: 1\n    - _OutlineWidth: 1\n    - _AlphaMask: 0\n    m_Colors:\n    - _Shadow2ndColor: {r: 1, g: 1, b: 1, a: 1}\n    - _EmissionColor: {r: 1, g: 1, b: 1, a: 1}\n    - _RimColor: {r: 1, g: 1, b: 1, a: 1}\n";
        let mut plans = vec![plan_of(POIYOMI, 0, Some("Poiyomi Toon")), plan_of(rich, 0, Some("Hidden/lilToonOutline")), plan_of(rich, 0, Some("VRM/MToon"))];
        // Textures in the custom slots too.
        plans[0].textures.push(TexUse { target: TexTarget::Custom("x".into()), guid: String::new(), unity_prop: String::new() });
        for p in &plans {
            for (k, v) in &p.file.custom {
                let prop = src.props.iter().find(|q| q.name == *k).unwrap_or_else(|| panic!("converter writes unknown toon property {k}"));
                assert_eq!(v.len(), prop.default.len(), "component count of {k}");
            }
            for t in &p.textures {
                if let TexTarget::Custom(n) = &t.target {
                    if n == "x" {
                        continue;
                    }
                    let prop = src.props.iter().find(|q| q.name == *n).unwrap_or_else(|| panic!("converter targets unknown texture property {n}"));
                    assert!(matches!(prop.kind, crate::ShaderPropKind::Texture { .. }), "{n} is a texture property");
                }
            }
        }
    }

    #[test]
    fn colour_space_conversion() {
        assert!((srgb_to_linear(0.5) - 0.2140).abs() < 1e-3);
        assert_eq!(srgb_to_linear(0.0), 0.0);
        assert_eq!(srgb_to_linear(1.0), 1.0);
        assert_eq!(srgb_to_linear(2.0), 1.0, "clamped");
    }

    #[test]
    fn poiyomi_values_map_to_toon() {
        let p = plan_of(POIYOMI, 0, Some(".poiyomi/Poiyomi Toon"));
        assert_eq!(p.shader, UnityShader::PoiyomiToon);
        assert_eq!(p.file.shader, "shaders/zestengine_toon.frag");
        assert_eq!(p.file.features.alpha_mode, AlphaMode::Cutout);
        assert!(p.file.features.double_sided, "_Cull 0");
        assert_eq!(p.file.params.alpha_cutoff, 0.35);
        assert_eq!(p.file.render_queue, Some(2450));
        // colour 0.5 sRGB -> ~0.214 linear
        assert!((c(&p, "main_color")[0] - 0.2140).abs() < 1e-3);
        assert_eq!(c(&p, "main_st"), vec![2.0, 3.0, 0.5, 0.25]);
        assert_eq!(c(&p, "lighting_mode"), vec![LM_MULTILAYER]);
        assert_eq!(c(&p, "RIM"), vec![1.0]);
        assert_eq!(c(&p, "rim_width"), vec![0.4]);
        assert_eq!(c(&p, "MATCAP"), vec![1.0]);
        assert_eq!(c(&p, "matcap_blend"), vec![0.0], "_MatcapAdd dominant -> Add");
        assert_eq!(c(&p, "matcap_strength"), vec![1.4]);
        assert_eq!(c(&p, "EMISSION"), vec![1.0]);
        assert_eq!(c(&p, "emission_strength"), vec![3.0]);
        assert_eq!(c(&p, "emission_color"), vec![1.0, 0.5, 0.0], "HDR emission stays linear");
        assert_eq!(c(&p, "OUTLINE"), vec![0.0]);
        assert_eq!(c(&p, "NORMAL_MAP"), vec![1.0]);
        let targets: Vec<_> = p.textures.iter().map(|t| t.target.clone()).collect();
        assert!(targets.contains(&TexTarget::Albedo) && targets.contains(&TexTarget::Normal));
        assert!(targets.contains(&TexTarget::Custom("matcap_tex".into())));
    }

    #[test]
    fn report_lists_every_unmapped_property() {
        let p = plan_of(POIYOMI, 0, Some(".poiyomi/Poiyomi Toon"));
        let text = p.report.to_text();
        assert!(text.contains("Poiyomi Toon") && text.contains("ZestEngine Toon"));
        // Unsupported feature with a value, called out by name.
        assert!(text.contains("_FlipbookTexArray"), "the flipbook texture is called out: {text}");
        assert!(!text.contains("stencil effects"), "stencil is no longer reported as unsupported: {text}");
        // `_StencilType: 1` is the writer preset (Always + Replace).
        assert!(text.contains("_StencilType") || text.contains("_Stencil*"), "stencil is reported as mapped: {text}");
        assert_eq!(p.file.features.stencil.front.pass_op, StencilOp::Replace);
        assert!(text.contains("_AudioLinkLightingStrength"), "unmapped float with a value is listed: {text}");
        // Zero-valued ones are summarised, not lost.
        assert!(text.contains("_ShaderOptimizerEnabled"), "{text}");
        // Stencil was the only Unsupported line of this fixture; it is mapped now.
        assert_eq!(p.report.count(Level::Unsupported), 0, "{text}");
    }

    #[test]
    fn stencil_properties_map_for_poiyomi_liltoon_and_unity() {
        let mat = |floats: &str| format!("Material:\n  m_Name: S\n  m_SavedProperties:\n    m_Floats:\n{floats}");
        // Poiyomi explicit reader: ref 2, Equal(3), Keep; masks.
        let p = plan_of(
            &mat("    - _StencilReference: 2\n    - _StencilCompareFunction: 3\n    - _StencilPassOp: 0\n    - _StencilReadMask: 15\n    - _StencilWriteMask: 240\n"),
            0, Some(".poiyomi/Poiyomi Toon"),
        );
        let st = p.file.features.stencil;
        assert_eq!((st.reference, st.read_mask, st.write_mask), (2, 15, 240));
        assert_eq!((st.front.compare, st.front.pass_op), (StencilCompare::Equal, StencilOp::Keep));
        assert_eq!(st.front, st.back);
        // lilToon writer: Always + Replace + fail/zfail ops, ColorMask 0.
        let p = plan_of(
            &mat("    - _StencilRef: 7\n    - _StencilComp: 8\n    - _StencilPass: 2\n    - _StencilFail: 1\n    - _StencilZFail: 5\n    - _ColorMask: 0\n"),
            0, Some("lilToon"),
        );
        let st = p.file.features.stencil;
        assert_eq!(st.reference, 7);
        assert_eq!((st.front.pass_op, st.front.fail_op, st.front.zfail_op), (StencilOp::Replace, StencilOp::Zero, StencilOp::Invert));
        assert_eq!(st.color_mask, 0, "ColorMask 0 -> a pure writer");
        assert!(p.report.to_text().contains("this material writes the stencil"), "{}", p.report.to_text());
        // A ColorMask 0 + ZWrite 0 writer keeps both switches even with the default compare.
        let p = plan_of(&mat("    - _ColorMask: 0\n    - _ZWrite: 0\n    - _StencilReference: 4\n    - _StencilPass: 2\n"), 0, Some("lilToon"));
        assert!(!p.file.features.stencil.depth_write && p.file.features.stencil.color_mask == 0);
        // ZWrite 0 alone (a plain transparent material) is not a stencil state.
        let p = plan_of(&mat("    - _ZWrite: 0\n"), 0, Some("lilToon"));
        assert_eq!(p.file.features.stencil, StencilState::default());
        // Poiyomi reader preset from `_StencilType: 2` alone.
        let p = plan_of(&mat("    - _StencilReference: 1\n    - _StencilType: 2\n"), 0, Some(".poiyomi/Poiyomi Toon"));
        assert_eq!(p.file.features.stencil.front.compare, StencilCompare::Equal);
        // Unity Standard / UI names + per-face overrides.
        let p = plan_of(&mat("    - _Stencil: 3\n    - _StencilComp: 5\n    - _StencilCompBack: 6\n    - _StencilPassBack: 3\n"), 0, Some("Standard"));
        let st = p.file.features.stencil;
        assert_eq!(st.front.compare, StencilCompare::Greater);
        assert_eq!((st.back.compare, st.back.pass_op), (StencilCompare::NotEqual, StencilOp::IncrementClamp));
        // Defaults stay inactive and no "unsupported" line is raised.
        let p = plan_of(&mat("    - _StencilReference: 0\n    - _StencilComp: 8\n"), 0, Some("lilToon"));
        assert!(!p.file.features.stencil.is_active());
        assert_eq!(p.file.features.stencil, StencilState::default());
    }

    #[test]
    fn mochie_standard_maps_to_pbr_with_orm() {
        let p = plan_of(MOCHIE, 0, Some("Mochie/Standard"));
        assert_eq!(p.shader, UnityShader::MochieStandard);
        assert_eq!(p.file.shader, "standard");
        assert!(p.file.features.normal_map);
        assert_eq!(p.file.params.normal_strength, 1.5);
        assert_eq!(p.file.params.base_color, [1.0; 4]);
        assert!(!p.file.features.double_sided, "_Culling 2 = back");
        assert!(p.orm.metallic_guid.is_some());
        assert!(matches!(p.orm.rough, Some(RoughSource::RoughnessMap(_))));
        assert_eq!(p.file.params.metallic, 1.0, "map drives metallic");
        assert!(p.report.to_text().contains("_SSR"), "SSR is reported as unsupported");
    }

    #[test]
    fn unity_standard_transparent_and_scalars() {
        let p = plan_of(MOCHIE, 1, None);
        assert_eq!(p.shader, UnityShader::UnityStandard);
        assert_eq!(p.file.features.alpha_mode, AlphaMode::Blend, "_Mode 3");
        assert!((p.file.params.roughness - 0.1).abs() < 1e-6, "1 - 0.9");
        assert_eq!(p.file.params.base_color[3], 0.3);
    }

    #[test]
    fn alpha_modes_follow_keywords_and_queue() {
        let mk = |kw: &str, q: i32| {
            let m = UnityMaterial { keywords: vec![kw.into()], render_queue: Some(q), ..Default::default() };
            convert_material(&m, UnityShader::UnityStandard, &ConvertOptions::default()).file.features.alpha_mode
        };
        assert_eq!(mk("_ALPHATEST_ON", 2450), AlphaMode::Cutout);
        assert_eq!(mk("_ALPHABLEND_ON", 3000), AlphaMode::Blend);
        assert_eq!(mk("X", 2000), AlphaMode::Opaque);
    }

    #[test]
    fn lil_and_mtoon_mappings() {
        let lil = "Material:\n  m_Name: L\n  m_SavedProperties:\n    m_Floats:\n    - _UseRim: 1\n    - _RimBorder: 0.8\n    - _TransparentMode: 2\n    - _UseShadow: 1\n    - _OutlineWidth: 0.5\n    m_Colors:\n    - _Shadow2ndColor: {r: 0.5, g: 0.5, b: 0.5, a: 0.4}\n    - _RimColor: {r: 1, g: 1, b: 1, a: 1}\n";
        let p = plan_of(lil, 0, Some("lilToon"));
        assert_eq!(p.file.features.alpha_mode, AlphaMode::Blend);
        assert!((c(&p, "rim_width")[0] - 0.2).abs() < 1e-6);
        assert_eq!(c(&p, "layer2_strength"), vec![0.4]);
        assert_eq!(c(&p, "OUTLINE"), vec![0.0], "lilToon without an Outline variant draws none");
        let p = plan_of(lil, 0, Some("Hidden/lilToonOutline"));
        assert_eq!(c(&p, "OUTLINE"), vec![1.0]);
        let mt = "Material:\n  m_Name: M\n  m_SavedProperties:\n    m_Floats:\n    - _ShadeToony: 1\n    - _ShadeShift: 0\n    - _BlendMode: 1\n    - _CullMode: 0\n    m_Colors:\n    - _ShadeColor: {r: 0.9, g: 0.8, b: 0.8, a: 1}\n";
        let p = plan_of(mt, 0, Some("VRM/MToon"));
        assert_eq!(p.shader, UnityShader::MToon);
        assert_eq!(p.file.features.alpha_mode, AlphaMode::Cutout);
        assert!(p.file.features.double_sided);
        assert_eq!(c(&p, "layer1_blur"), vec![0.0]);
        assert_eq!(c(&p, "layer1_border"), vec![0.5]);
    }

    fn tex_entry(prop: &str, guid: &str) -> String {
        format!("    - {prop}:\n        m_Texture: {{fileID: 2800000, guid: {guid}, type: 3}}\n        m_Scale: {{x: 1, y: 1}}\n        m_Offset: {{x: 0, y: 0}}\n")
    }

    fn mat_text(floats: &[(&str, f32)], colors: &[(&str, [f32; 4])], texs: &[(&str, &str)]) -> String {
        let mut s = String::from("Material:\n  m_Name: T\n  m_SavedProperties:\n    m_TexEnvs:\n");
        for (k, g) in texs {
            s.push_str(&tex_entry(k, g));
        }
        s.push_str("    m_Floats:\n");
        for (k, v) in floats {
            s.push_str(&format!("    - {k}: {v}\n"));
        }
        s.push_str("    m_Colors:\n");
        for (k, c) in colors {
            s.push_str(&format!("    - {k}: {{r: {}, g: {}, b: {}, a: {}}}\n", c[0], c[1], c[2], c[3]));
        }
        s
    }

    #[test]
    fn cull_front_and_blend_modes_map_to_render_state() {
        let front = plan_of(&mat_text(&[("_Cull", 1.0)], &[], &[]), 0, Some("lilToon"));
        assert!(front.file.features.cull_front && !front.file.features.double_sided, "Cull Front");
        assert_eq!(front.report.count(Level::Unsupported), 0, "{}", front.report.to_text());
        let back = plan_of(&mat_text(&[("_Cull", 2.0)], &[], &[]), 0, Some("lilToon"));
        assert!(!back.file.features.cull_front && !back.file.features.double_sided);
        // Poiyomi modes 5 / 7 and generic blend factors.
        let add = plan_of(&mat_text(&[("_Mode", 5.0)], &[], &[]), 0, Some(".poiyomi/Poiyomi Toon"));
        assert_eq!((add.file.features.alpha_mode, add.file.features.blend), (AlphaMode::Blend, BlendMode::Additive));
        let mul = plan_of(&mat_text(&[("_Mode", 7.0)], &[], &[]), 0, Some(".poiyomi/Poiyomi Toon"));
        assert_eq!(mul.file.features.blend, BlendMode::Multiply);
        let fade = plan_of(&mat_text(&[("_Mode", 3.0), ("_SrcBlend", 5.0), ("_DstBlend", 10.0)], &[], &[]), 0, Some(".poiyomi/Poiyomi Toon"));
        assert_eq!(fade.file.features.blend, BlendMode::Alpha);
        let lil_add = plan_of(&mat_text(&[("_TransparentMode", 2.0), ("_SrcBlend", 5.0), ("_DstBlend", 1.0)], &[], &[]), 0, Some("lilToon"));
        assert_eq!(lil_add.file.features.blend, BlendMode::Additive);
        assert_eq!(blend_from_factors(2, 0), Some(BlendMode::Multiply));
        assert_eq!(blend_from_factors(1, 10), Some(BlendMode::Premultiplied));
        // An opaque material ignores stray blend factors.
        let opaque = plan_of(&mat_text(&[("_TransparentMode", 0.0), ("_SrcBlend", 1.0), ("_DstBlend", 1.0)], &[], &[]), 0, Some("lilToon"));
        assert_eq!(opaque.file.features.blend, BlendMode::Alpha);
    }

    #[test]
    fn lil_extras_map_to_the_new_toon_features() {
        let t = mat_text(
            &[
                ("_TransparentMode", 3.0),
                ("_RefractionStrength", 0.25),
                ("_UseMatCap2nd", 1.0),
                ("_MatCap2ndBlend", 0.6),
                ("_MatCap2ndBlendMode", 3.0),
                ("_UseEmission2nd", 1.0),
                ("_UseMain2ndTex", 1.0),
                ("_Main2ndTexBlendMode", 3.0),
                ("_UseReflection", 1.0),
                ("_Smoothness", 0.75),
                ("_ApplySpecular", 1.0),
            ],
            &[
                ("_Shadow3rdColor", [0.4, 0.4, 0.5, 0.5]),
                ("_Shadow3rdBorder", [0.0; 4]),
                ("_MainTexHSVG", [0.25, 1.0, 1.5, 1.0]),
                ("_RefractionColor", [1.0, 0.5, 0.5, 1.0]),
                ("_Emission2ndColor", [2.0, 1.0, 0.0, 1.0]),
            ],
            &[("_MatCap2ndTex", "aa"), ("_Main2ndTex", "bb"), ("_Bump2ndMap", "cc")],
        );
        let p = plan_of(&t, 0, Some("lilToon"));
        assert_eq!(c(&p, "GRAB"), vec![1.0]);
        assert!(p.file.features.grab, "the material asks for the opaque copy");
        assert_eq!(c(&p, "grab_distortion"), vec![0.25]);
        assert_eq!(c(&p, "grab_tint").len(), 3);
        assert_eq!(c(&p, "MATCAP2"), vec![1.0]);
        assert_eq!(c(&p, "matcap2_blend"), vec![1.0], "lil multiply -> Multiply");
        assert_eq!(c(&p, "matcap2_strength"), vec![0.6]);
        assert_eq!(c(&p, "EMISSION2"), vec![1.0]);
        assert_eq!(c(&p, "emission2_color"), vec![2.0, 1.0, 0.0]);
        assert_eq!(c(&p, "layer3_strength"), vec![0.5]);
        assert_eq!(c(&p, "DECAL"), vec![1.0]);
        assert_eq!(c(&p, "decal_blend"), vec![1.0]);
        assert_eq!(c(&p, "REFLECTIONS"), vec![1.0]);
        assert_eq!(c(&p, "specular_mode"), vec![2.0]);
        assert_eq!(c(&p, "hue_shift"), vec![0.25]);
        assert_eq!(c(&p, "brightness"), vec![1.5]);
        let targets: Vec<_> = p.textures.iter().map(|t| t.target.clone()).collect();
        assert!(targets.contains(&TexTarget::Custom("matcap2_tex".into())) && targets.contains(&TexTarget::Custom("decal_tex".into())));
        // The Detail normal is off without `_UseBump2ndMap`.
        assert_eq!(c(&p, "DETAIL"), vec![0.0]);
    }

    #[test]
    fn poiyomi_extras_map_to_the_new_toon_features() {
        let t = mat_text(
            &[
                ("_LightingMode", 2.0),
                ("_Mode", 4.0),
                ("_EnableDissolve", 1.0),
                ("_DissolveAlpha", 0.4),
                ("_EnableFlipbook", 1.0),
                ("_FlipbookTotalFrames", 16.0),
                ("_FlipbookFPS", 24.0),
                ("_EnableAudioLink", 1.0),
                ("_AudioLinkRimEmissionBand", 2.0),
                ("_EnableGrabpass", 1.0),
                ("_GrabBlurDistance", 0.04),
                ("_MainHueShift", 0.5),
            ],
            &[("_GrabpassColor", [1.0, 1.0, 1.0, 1.0]), ("_AudioLinkRimEmissionAdd", [0.0, 2.0, 0.0, 0.0]), ("_PanospherePan", [0.25, 0.0, 0.0, 0.0])],
            &[("_DissolveNoiseTexture", "dd"), ("_FlipbookTexArray", "ee"), ("_PanosphereTexture", "ff"), ("_DetailTex", "gg")],
        );
        let p = plan_of(&t, 0, Some(".poiyomi/Poiyomi Toon"));
        assert_eq!(c(&p, "lighting_mode"), vec![LM_WRAPPED], "Poiyomi Wrapped");
        assert_eq!(c(&p, "DISSOLVE"), vec![1.0]);
        assert_eq!(c(&p, "dissolve_amount"), vec![0.4]);
        assert_eq!(c(&p, "FLIPBOOK"), vec![1.0]);
        assert_eq!(c(&p, "flip_frames"), vec![16.0]);
        assert_eq!((c(&p, "flip_cols"), c(&p, "flip_rows")), (vec![16.0], vec![1.0]), "no grid in a Poiyomi material: one row of frames");
        assert_eq!(c(&p, "flip_fps"), vec![24.0]);
        assert_eq!(c(&p, "PANOSPHERE"), vec![1.0]);
        assert_eq!(c(&p, "AUDIO"), vec![1.0]);
        assert_eq!(c(&p, "audio_rim"), vec![2.0], "the green channel of the add row is the gain");
        assert_eq!(c(&p, "audio_rim_band"), vec![2.0]);
        assert_eq!(c(&p, "pano_spin"), vec![0.25], "_PanospherePan x");
        assert_eq!(c(&p, "GRAB"), vec![1.0]);
        assert_eq!(c(&p, "grab_blur"), vec![8.0]);
        assert_eq!(c(&p, "DETAIL"), vec![1.0]);
        assert_eq!(c(&p, "hue_shift"), vec![0.5]);
        assert_eq!(p.file.features.alpha_mode, AlphaMode::Blend);
    }

    #[test]
    fn every_extra_key_exists_in_the_toon_shader() {
        // Everything the extras write must be declared by the shader (a typo would
        // silently do nothing).
        let body = include_str!("../../../shaders/zestengine_toon.frag");
        let source = crate::parse_shader_source(body, "zestengine_toon").unwrap();
        let t = mat_text(
            &[("_TransparentMode", 3.0), ("_UseMatCap2nd", 1.0), ("_UseEmission2nd", 1.0), ("_UseMain2ndTex", 1.0), ("_UseReflection", 1.0), ("_ApplySpecular", 1.0), ("_OutlineWidth", 1.0), ("_OutlineVertexR2Width", 1.0)],
            &[("_Shadow3rdColor", [0.5, 0.5, 0.5, 1.0]), ("_MainTexHSVG", [0.2, 1.1, 1.2, 1.0]), ("_Emission2ndColor", [1.0, 1.0, 1.0, 1.0])],
            &[("_MatCap2ndTex", "aa"), ("_Main2ndTex", "bb")],
        );
        let p = plan_of(&t, 0, Some("Hidden/lilToonOutline"));
        for k in p.file.custom.keys() {
            assert!(source.prop(k).is_some(), "the converter writes '{k}', which ZestEngine Toon does not declare");
        }
        assert_eq!(c(&p, "outline_vertex_width"), vec![1.0]);
    }

    #[test]
    fn mochie_glass_still_has_no_converter_and_says_so() {
        // No real Mochie Glass material exists on this machine, so there is no verified name list.
        let m = UnityMaterial { name: "g".into(), shader_name: Some("Mochie/Glass".into()), ..Default::default() };
        let p = convert_material(&m, identify_shader(&m), &ConvertOptions::default());
        assert!(p.report.to_text().contains("no ZestEngine counterpart"));
    }

    /// The property names are those of the Screen FX prefab material in Mochie's v1.65-1.75
    /// packages (read from the `.unitypackage` on this machine); the effect settings are the
    /// converter's choices.
    #[test]
    fn mochie_screen_fx_converts_to_screen_fx() {
        let t = mat_text(
            &[("_Opacity", 0.8), ("_Saturation", 1.5), ("_Contrast", 1.0), ("_Brightness", 1.0), ("_BlurModel", 1.0), ("_BlurRadius", 2.0), ("_PixelationStr", 0.25), ("_Zoom", 1.0)],
            &[("_Color", [1.0, 0.5, 0.5, 1.0])],
            &[],
        );
        let m = parse_unity_materials(&t)[0].clone().with_shader_name(Some("Mochie/Screen FX X".into()));
        let p = convert_material(&m, identify_shader(&m), &ConvertOptions::default());
        assert_eq!(p.report.target, "ZestEngine Screen FX");
        assert_eq!(p.file.shader, "shaders/zestengine_screenfx.frag");
        assert!(p.file.features.grab && p.file.features.cull_front && p.file.features.alpha_mode == AlphaMode::Blend);
        let cv = |k: &str| p.file.custom.get(k).cloned().unwrap_or_default();
        assert_eq!(cv("fx_amount"), vec![0.8]);
        assert_eq!((cv("FILTER"), cv("filt_saturation")), (vec![1.0], vec![1.5]));
        assert_eq!((cv("BLUR"), cv("blur_radius")), (vec![1.0], vec![16.0]));
        assert_eq!((cv("PIXELATE"), cv("pixel_size")), (vec![1.0], vec![8.0]));
        assert_eq!(cv("SHAKE"), vec![0.0]);
        assert!(p.report.lines.iter().any(|l| l.level == Level::Unsupported && l.property == "_Zoom"), "{}", p.report.to_text());
        // Everything written is declared by the Screen FX shader.
        let source = crate::parse_shader_source(include_str!("../../../shaders/zestengine_screenfx.frag"), "zestengine_screenfx").unwrap();
        for k in p.file.custom.keys() {
            assert!(source.prop(k).is_some(), "the converter writes '{k}', which ZestEngine Screen FX does not declare");
        }
    }

    #[test]
    fn pack_orm_channels() {
        let mut metal = image::RgbaImage::new(2, 2);
        for p in metal.pixels_mut() {
            *p = image::Rgba([200, 0, 0, 51]); // metallic 200, smoothness 0.2
        }
        let occ = image::RgbaImage::from_pixel(1, 1, image::Rgba([0, 100, 0, 255]));
        let out = pack_orm(Some(&metal), Some(&occ), RoughPixels::SmoothAlpha(&metal, 1.0));
        assert_eq!(out.dimensions(), (2, 2));
        let px = out.get_pixel(0, 0).0;
        assert_eq!(px, [100, 255 - 51, 200, 255]);
        let flat = pack_orm(None, None, RoughPixels::None);
        assert_eq!(flat.get_pixel(0, 0).0, [255, 255, 0, 255]);
    }

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("zest_unity_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_png(path: &Path, rgba: [u8; 4]) {
        image::RgbaImage::from_pixel(4, 4, image::Rgba(rgba)).save(path).unwrap();
    }

    fn meta(dir: &Path, file: &str, guid: &str) {
        std::fs::write(dir.join(format!("{file}.meta")), format!("fileFormatVersion: 2\nguid: {guid}\n")).unwrap();
    }

    #[test]
    fn guid_scan_and_shader_names() {
        let root = tmp("scan");
        std::fs::create_dir_all(root.join("Textures")).unwrap();
        write_png(&root.join("Textures/a.png"), [1, 2, 3, 255]);
        meta(&root.join("Textures"), "a.png", "11111111111111111111111111111111");
        std::fs::write(root.join("x.shader"), "Shader \".poiyomi/Poiyomi Toon\"\n{ }\n").unwrap();
        meta(&root, "x.shader", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        std::fs::write(root.join("orphan.png.meta"), "guid: 99999999999999999999999999999999\n").unwrap();
        let a = UnityAssets::scan(&root);
        assert!(a.path_of("11111111111111111111111111111111").unwrap().ends_with("a.png"));
        assert!(a.path_of("99999999999999999999999999999999").is_none(), "a .meta without its asset is skipped");
        assert_eq!(a.shader_names["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"], ".poiyomi/Poiyomi Toon");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn full_round_trip_writes_a_loadable_material() {
        let unity = tmp("rt_unity");
        let project = tmp("rt_project");
        let out = project.join("materials");
        // Poiyomi material referencing textures by guid.
        std::fs::write(unity.join("Body.mat"), POIYOMI).unwrap();
        for (file, guid) in [("main.png", "11111111111111111111111111111111"), ("bump.png", "22222222222222222222222222222222"), ("matcap.png", "33333333333333333333333333333333")] {
            write_png(&unity.join(file), [128, 128, 255, 255]);
            meta(&unity, file, guid);
        }
        std::fs::write(unity.join("p.shader"), "Shader \"Hidden/Locked/.poiyomi/Poiyomi Toon/abc\"\n").unwrap();
        meta(&unity, "p.shader", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");

        let mut seen = Vec::new();
        let s = convert_unity_folder(&unity, &project, &out, &ConvertOptions::default(), &mut |d, t, n| {
            seen.push((d, t, n.to_owned()));
            true
        });
        assert_eq!(s.failed.len(), 0, "{:?}", s.failed);
        assert_eq!(s.converted.len(), 1);
        assert_eq!(seen, vec![(0, 1, "Body.mat".to_owned())]);
        assert!(s.message().contains("Converted 1"));
        let c = &s.converted[0];
        let mf = crate::load_material_file(&c.material_path).expect("loadable .material");
        assert_eq!(mf.shader, "shaders/zestengine_toon.frag");
        assert_eq!(mf.textures.albedo.as_deref(), Some(Path::new("textures/unity/main.png")));
        assert_eq!(mf.textures.normal.as_deref(), Some(Path::new("textures/unity/bump.png")));
        assert_eq!(mf.textures.shader["matcap_tex"], PathBuf::from("textures/unity/matcap.png"));
        assert!(project.join("textures/unity/main.png").is_file());
        // Slot-aware meta: the normal texture was imported as a normal map.
        assert!(crate::asset_meta::meta_path(project.join("textures/unity/bump.png")).exists());
        let report = std::fs::read_to_string(&c.report_path).unwrap();
        assert!(report.contains("Poiyomi Toon") && report.contains("_FlipbookTexArray"), "{report}");
        // The flipbook texture's guid has no file: that is reported, not dropped.
        let _ = std::fs::remove_dir_all(&unity);
        let _ = std::fs::remove_dir_all(&project);
    }

    #[test]
    fn missing_texture_and_unsupported_format_are_reported() {
        let unity = tmp("miss_unity");
        let project = tmp("miss_project");
        std::fs::write(unity.join("m.mat"), MOCHIE).unwrap();
        std::fs::write(unity.join("main.psd"), b"8BPS").unwrap();
        meta(&unity, "main.psd", "11111111111111111111111111111111");
        let s = convert_unity_material(&unity.join("m.mat"), None, &project, &project.join("m"), &ConvertOptions::default());
        assert_eq!(s.converted.len(), 2);
        let text = s.converted[0].report.to_text();
        assert!(text.contains("not a format the engine loads"), "{text}");
        assert!(text.contains("not found in the Unity folder"), "{text}");
        let _ = std::fs::remove_dir_all(&unity);
        let _ = std::fs::remove_dir_all(&project);
    }

    #[test]
    fn standard_round_trip_packs_orm() {
        let unity = tmp("orm_unity");
        let project = tmp("orm_project");
        std::fs::write(unity.join("m.mat"), MOCHIE).unwrap();
        for (file, guid) in [("metal.png", "55555555555555555555555555555555"), ("rough.png", "66666666666666666666666666666666")] {
            write_png(&unity.join(file), [255, 255, 255, 255]);
            meta(&unity, file, guid);
        }
        let s = convert_unity_material(&unity.join("m.mat"), None, &project, &project.join("m"), &ConvertOptions::default());
        let mf = crate::load_material_file(&s.converted[0].material_path).unwrap();
        let orm = mf.textures.orm.expect("packed orm");
        let img = image::open(project.join(&orm)).unwrap().into_rgba8();
        assert_eq!(img.get_pixel(0, 0).0, [255, 255, 255, 255]);
        let _ = std::fs::remove_dir_all(&unity);
        let _ = std::fs::remove_dir_all(&project);
    }

    #[test]
    fn garbage_input_never_panics() {
        let unity = tmp("junk_unity");
        let project = tmp("junk_project");
        std::fs::write(unity.join("bin.mat"), [0u8, 1, 2, 3, 0xff, 0xfe]).unwrap();
        std::fs::write(unity.join("txt.mat"), "not yaml at all {{{ ]]]").unwrap();
        std::fs::write(unity.join("empty.mat"), "").unwrap();
        std::fs::write(unity.join("weird.mat"), "Material:\n  m_SavedProperties:\n    m_TexEnvs:\n    - _MainTex:\n        m_Texture: {guid: ../../etc/passwd}\n").unwrap();
        let s = convert_unity_folder(&unity, &project, &project.join("m"), &ConvertOptions::default(), &mut |_, _, _| true);
        assert_eq!(s.failed.len(), 3);
        assert_eq!(s.converted.len(), 1);
        let _ = std::fs::remove_dir_all(&unity);
        let _ = std::fs::remove_dir_all(&project);
    }

    #[test]
    fn standard_detail_maps_become_the_detail_layer() {
        let t = mat_text(&[("_DetailNormalMapScale", 0.7)], &[], &[("_DetailAlbedoMap", "aa"), ("_DetailNormalMap", "bb"), ("_DetailMask", "cc")]);
        let p = plan_of(&t, 0, Some("Standard"));
        let d = p.file.params.detail;
        assert_eq!((d.albedo_strength, d.normal_strength, d.blend), (1.0, 0.7, 0.0), "x2 blend, scale carried");
        let targets: Vec<_> = p.textures.iter().map(|t| t.target.clone()).collect();
        assert!(targets.contains(&TexTarget::DetailAlbedo) && targets.contains(&TexTarget::DetailNormal), "{targets:?}");
        assert!(p.repacks.iter().any(|r| r.kind == RepackKind::MaskFromAlpha), "the mask is the alpha channel in Unity");
        assert_eq!(p.report.count(Level::Unmapped), 0, "{}", p.report.to_text());
        // No detail maps: layer off.
        let none = plan_of(&mat_text(&[("_DetailNormalMapScale", 1.0)], &[], &[]), 0, Some("Standard"));
        assert!(!none.file.params.detail.is_on());
    }

    #[test]
    fn hdrp_packed_detail_map_is_split_into_three_textures() {
        let t = mat_text(&[("_DetailAlbedoScale", 0.5), ("_DetailNormalScale", 1.5), ("_DetailSmoothnessScale", 0.25)], &[], &[("_DetailMap", "dd")]);
        let p = plan_of(&t, 0, Some("HDRP/Lit"));
        let d = p.file.params.detail;
        assert_eq!((d.albedo_strength, d.normal_strength, d.orm_strength), (0.5, 1.5, 0.25));
        let kinds: Vec<_> = p.repacks.iter().map(|r| r.kind.clone()).collect();
        assert_eq!(kinds, vec![RepackKind::HdrpAlbedo, RepackKind::HdrpNormal, RepackKind::HdrpSmoothness]);
        // The repack maths: HDRP linear 0.5 is neutral, which must land on the engine's neutral grey.
        let neutral = image::RgbaImage::from_pixel(1, 1, image::Rgba([128, 128, 128, 255]));
        let a = repack_hdrp_detail_albedo(&neutral).get_pixel(0, 0)[0];
        let linear = srgb_to_linear(a as f32 / 255.0) * 4.594_794;
        assert!((linear - 128.0 / 255.0 * 2.0).abs() < 0.03, "x2 of the detail value: {linear}");
        let flat = image::RgbaImage::from_pixel(1, 1, image::Rgba([0, 128, 200, 128]));
        let n = repack_hdrp_detail_normal(&flat).get_pixel(0, 0).0;
        assert!(n[2] > 250 && (n[0] as i32 - 128).abs() <= 1, "flat normal rebuilt: {n:?}");
        assert_eq!(repack_hdrp_detail_orm(&flat).get_pixel(0, 0).0, [255, 55, 255, 255], "roughness = 1 - smoothness");
        assert_eq!(repack_mask_from_alpha(&flat).get_pixel(0, 0).0, [128, 128, 128, 255]);
    }

    #[test]
    fn detail_repacks_are_written_by_the_round_trip() {
        let unity = tmp("detail_unity");
        let project = tmp("detail_project");
        let mat = "Material:\n  m_Name: D\n  m_Shader: {fileID: 46, guid: 0000000000000000f000000000000000, type: 0}\n  m_SavedProperties:\n    m_TexEnvs:\n".to_owned()
            + &tex_entry("_DetailMask", "77777777777777777777777777777777")
            + &tex_entry("_DetailAlbedoMap", "88888888888888888888888888888888")
            + "    m_Floats: []\n    m_Colors: []\n";
        std::fs::write(unity.join("d.mat"), mat).unwrap();
        write_png(&unity.join("mask.png"), [10, 20, 30, 200]);
        meta(&unity, "mask.png", "77777777777777777777777777777777");
        write_png(&unity.join("det.png"), [128, 128, 128, 255]);
        meta(&unity, "det.png", "88888888888888888888888888888888");
        let s = convert_unity_material(&unity.join("d.mat"), None, &project, &project.join("m"), &ConvertOptions::default());
        assert_eq!(s.converted.len(), 1, "{:?}", s.failed);
        let mf = crate::load_material_file(&s.converted[0].material_path).unwrap();
        assert!(mf.textures.detail_albedo.is_some(), "{:?}", mf.textures);
        let mask = image::open(project.join(mf.textures.detail_mask.expect("mask written"))).unwrap().into_rgba8();
        assert_eq!(mask.get_pixel(0, 0).0, [200, 200, 200, 255], "alpha moved to red");
        let _ = std::fs::remove_dir_all(&unity);
        let _ = std::fs::remove_dir_all(&project);
    }

    #[test]
    fn hdrp_translucent_material_maps_to_translucency() {
        let t = mat_text(&[("_MaterialID", 5.0)], &[("_TransmissionTint", [1.0, 0.5, 0.25, 1.0])], &[("_ThicknessMap", "tt")]);
        let p = plan_of(&t, 0, Some("HDRP/Lit"));
        let tr = p.file.params.transmission;
        assert_eq!(tr.translucency, 1.0);
        assert!(tr.map_is_thickness, "the HDRP map is thickness");
        assert!(tr.translucency_color[0] > tr.translucency_color[1] && tr.translucency_color[1] > tr.translucency_color[2], "tint kept, in linear: {:?}", tr.translucency_color);
        assert!(p.textures.iter().any(|t| t.target == TexTarget::Translucency));
        // A plain Standard material has none.
        let plain = plan_of(&mat_text(&[("_Metallic", 0.5)], &[], &[]), 0, Some("Standard"));
        assert_eq!(plain.file.params.transmission.translucency, 0.0);
    }

    /// Real Poiyomi materials from a Unity project on this machine (skipped when absent):
    /// the property names checked in this pass must produce the features they claim to.
    #[test]
    fn real_poiyomi_materials_map_the_verified_names() {
        let root = Path::new("/home/lemonxah/storage/avis/lemon_meringue/Assets");
        if !root.is_dir() {
            eprintln!("[real data] no Unity project at {}: skipped", root.display());
            return;
        }
        let assets = UnityAssets::scan(root);
        let mut mats = Vec::new();
        collect_mats(root, &mut mats, 0);
        let (mut grab, mut flip, mut dissolve, mut decal, mut poi, mut total) = (0, 0, 0, 0, 0, 0);
        for path in mats {
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            for mut m in parse_unity_materials(&text) {
                total += 1;
                if m.name.is_empty() {
                    m.name = "m".into();
                }
                let m = assets.name_shader(m);
                let shader = identify_shader(&m);
                let plan = convert_material(&m, shader.clone(), &ConvertOptions::default());
                if shader != UnityShader::PoiyomiToon {
                    continue;
                }
                poi += 1;
                let on = |k: &str| plan.file.custom.get(k).is_some_and(|v| v[0] != 0.0);
                if m.float("_EnableGrabpass").is_some_and(|v| v > 0.5) {
                    assert!(on("GRAB") && plan.file.features.grab, "{}: _EnableGrabpass maps to Grab Pass", plan.name);
                    grab += 1;
                }
                if m.float("_EnableFlipbook").is_some_and(|v| v > 0.5) {
                    // The frames are a Texture2DArray: the report must say what happened, not drop it.
                    let text = plan.report.to_text();
                    assert!(on("FLIPBOOK") || text.contains("Texture2DArray"), "{}: {text}", plan.name);
                    flip += 1;
                }
                if m.float("_EnableDissolve").is_some_and(|v| v > 0.5) {
                    assert!(on("DISSOLVE"), "{}: _EnableDissolve maps to Dissolve", plan.name);
                    assert!(plan.file.custom.contains_key("dissolve_amount"), "{}: _DissolveAlpha is the amount", plan.name);
                    dissolve += 1;
                }
                if m.float("_DecalEnabled").is_some_and(|v| v > 0.5) && m.tex_any(&["_DecalTexture"]).is_some_and(|(_, e)| e.guid.is_some()) {
                    assert!(on("DECAL"), "{}: _DecalEnabled maps to Decal", plan.name);
                    decal += 1;
                }
            }
        }
        eprintln!("[real data] {total} materials, {poi} Poiyomi: grab {grab}, flipbook {flip}, dissolve {dissolve}, decal {decal}");
        assert!(poi > 0, "the project holds Poiyomi materials");
    }
}
