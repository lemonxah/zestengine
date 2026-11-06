//! Texture role detection from file names, for downloaded asset packs.
//!
//! A file name is split into tokens (separators and camelCase), a trailing
//! resolution token (`4K`, `2048`) and the Megascans `Billboard` marker are
//! set aside, and the last one to four tokens are looked up in the suffix
//! tables below. The tables hold the *naming conventions* of the common
//! sources: Megascans (`<id>_8K_Albedo`), Unreal (`T_Rock_BaseColor`, `_N`,
//! `_ORM`), Substance / Sketchfab (`_Base_color`, `_Normal_OpenGL`,
//! `_Height`), glTF (`_baseColor`, `_metallicRoughness`) and Unity (`_MainTex`,
//! `_BumpMap`, `_MetallicGlossMap`). What is left of the name is the material
//! or asset prefix textures are grouped by.
//!
//! Single-letter and other short aliases (`_N`, `_D`, `_M`) are accepted as
//! [`Confidence::Weak`] guesses so the import dialog can flag them.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// What a texture holds. The packed roles name a channel layout; the engine's
/// own packed layout is [`TexRole::Orm`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TexRole {
    Albedo,
    /// Tangent-space normal, OpenGL convention (Y+, green up): the engine's own
    /// and Megascans' / glTF's / Substance's default.
    Normal,
    /// Tangent-space normal, DirectX convention (Y-): Unreal, `_DX`. The green
    /// channel is flipped on import.
    NormalDx,
    Roughness,
    /// Smoothness: roughness inverted.
    Gloss,
    Metallic,
    Ao,
    /// R occlusion, G roughness, B metallic (also `ARM`): the engine's layout.
    Orm,
    /// R roughness, G metallic, B occlusion.
    Rma,
    /// glTF: G roughness, B metallic (R unused).
    MetalRough,
    /// Unity `_MetallicGlossMap`: R metallic, A smoothness.
    MetalGloss,
    /// Unity HDRP mask map: R metallic, G occlusion, B detail mask, A smoothness.
    MaskMap,
    /// Height / displacement (white = high).
    Height,
    /// Megascans `Bump`: a lower-quality height, used when nothing better exists.
    Bump,
    Opacity,
    Translucency,
    Specular,
    Cavity,
    Emission,
    Fuzz,
    /// Present in the pack but not used (previews, thumbnails).
    Ignore,
}

impl TexRole {
    /// Every role, in dropdown order.
    pub const ALL: [TexRole; 21] = [
        TexRole::Albedo,
        TexRole::Normal,
        TexRole::NormalDx,
        TexRole::Roughness,
        TexRole::Gloss,
        TexRole::Metallic,
        TexRole::Ao,
        TexRole::Orm,
        TexRole::Rma,
        TexRole::MetalRough,
        TexRole::MetalGloss,
        TexRole::MaskMap,
        TexRole::Height,
        TexRole::Bump,
        TexRole::Opacity,
        TexRole::Translucency,
        TexRole::Specular,
        TexRole::Cavity,
        TexRole::Emission,
        TexRole::Fuzz,
        TexRole::Ignore,
    ];

    pub fn label(self) -> &'static str {
        match self {
            TexRole::Albedo => "Albedo / base colour",
            TexRole::Normal => "Normal (OpenGL, Y+)",
            TexRole::NormalDx => "Normal (DirectX, Y-)",
            TexRole::Roughness => "Roughness",
            TexRole::Gloss => "Gloss / smoothness",
            TexRole::Metallic => "Metallic",
            TexRole::Ao => "Ambient occlusion",
            TexRole::Orm => "ORM (R occlusion, G rough, B metal)",
            TexRole::Rma => "RMA (R rough, G metal, B occlusion)",
            TexRole::MetalRough => "Metal-rough (G rough, B metal)",
            TexRole::MetalGloss => "Metal-gloss (R metal, A smooth)",
            TexRole::MaskMap => "Mask map (R metal, G occl., A smooth)",
            TexRole::Height => "Height / displacement",
            TexRole::Bump => "Bump",
            TexRole::Opacity => "Opacity",
            TexRole::Translucency => "Translucency",
            TexRole::Specular => "Specular",
            TexRole::Cavity => "Cavity",
            TexRole::Emission => "Emission",
            TexRole::Fuzz => "Fuzz / sheen",
            TexRole::Ignore => "Ignore",
        }
    }

    /// A packed map carrying several of occlusion / roughness / metallic.
    pub fn is_packed(self) -> bool {
        matches!(self, TexRole::Orm | TexRole::Rma | TexRole::MetalRough | TexRole::MetalGloss | TexRole::MaskMap)
    }

    /// Holds colour (decoded from sRGB); everything else is linear data.
    pub fn is_srgb(self) -> bool {
        matches!(self, TexRole::Albedo | TexRole::Emission)
    }
}

/// How sure a name match is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Confidence {
    /// A short or ambiguous alias (`_N`, `_D`, `_M`): worth checking.
    Weak,
    /// A name from one of the known conventions.
    Known,
}

/// A source channel of an input image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chan {
    R,
    G,
    B,
    A,
}

/// One output channel read from one input channel, optionally inverted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pick {
    pub src: Chan,
    pub invert: bool,
}

const fn pick(src: Chan) -> Option<Pick> {
    Some(Pick { src, invert: false })
}

const fn pick_inv(src: Chan) -> Option<Pick> {
    Some(Pick { src, invert: true })
}

/// Where each of the engine's ORM channels (`[occlusion, roughness, metallic]`)
/// comes from in a packed map of `role`. `None` = the map does not carry it.
/// Non-packed roles return `None` for the whole layout.
pub fn orm_layout(role: TexRole) -> Option<[Option<Pick>; 3]> {
    Some(match role {
        TexRole::Orm => [pick(Chan::R), pick(Chan::G), pick(Chan::B)],
        TexRole::Rma => [pick(Chan::B), pick(Chan::R), pick(Chan::G)],
        TexRole::MetalRough => [None, pick(Chan::G), pick(Chan::B)],
        TexRole::MetalGloss => [None, pick_inv(Chan::A), pick(Chan::R)],
        TexRole::MaskMap => [pick(Chan::G), pick_inv(Chan::A), pick(Chan::R)],
        _ => return None,
    })
}

/// Result of classifying one texture file name.
#[derive(Clone, Debug, PartialEq)]
pub struct Classified {
    /// The name before the role: the material / asset prefix (resolution and
    /// `Billboard` markers removed).
    pub prefix: String,
    pub role: TexRole,
    pub confidence: Confidence,
    /// Edge length in texels from a `4K` / `2048` token.
    pub resolution: Option<u32>,
    /// A Megascans billboard-impostor texture (`..._Billboard_...`).
    pub billboard: bool,
    /// UDIM tile number (`.1001`), when the name carries one.
    pub udim: Option<u32>,
}

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

/// `(joined lowercase tokens, role)`. Looked up for the last 1..=4 tokens of
/// a name, longest first. Entries of 1-2 letters are weak guesses.
const ROLE_TABLE: &[(&str, TexRole)] = &[
    // Albedo
    ("albedo", TexRole::Albedo),
    ("albedomap", TexRole::Albedo),
    ("albedotransparency", TexRole::Albedo),
    ("basecolor", TexRole::Albedo),
    ("basecolour", TexRole::Albedo),
    ("basecolormap", TexRole::Albedo),
    ("basemap", TexRole::Albedo),
    ("diffuse", TexRole::Albedo),
    ("diffusemap", TexRole::Albedo),
    ("diffusecolor", TexRole::Albedo),
    ("diff", TexRole::Albedo),
    ("color", TexRole::Albedo),
    ("colour", TexRole::Albedo),
    ("colormap", TexRole::Albedo),
    ("col", TexRole::Albedo),
    ("alb", TexRole::Albedo),
    ("maintex", TexRole::Albedo),
    ("bc", TexRole::Albedo),
    ("d", TexRole::Albedo),
    ("c", TexRole::Albedo),
    // Normal (OpenGL unless named otherwise)
    ("normal", TexRole::Normal),
    ("normalmap", TexRole::Normal),
    ("normalgl", TexRole::Normal),
    ("normalogl", TexRole::Normal),
    ("normalopengl", TexRole::Normal),
    ("norgl", TexRole::Normal),
    ("nor", TexRole::Normal),
    ("nrm", TexRole::Normal),
    ("nrml", TexRole::Normal),
    ("bumpmap", TexRole::Normal),
    ("normaldx", TexRole::NormalDx),
    ("normaldirectx", TexRole::NormalDx),
    ("nordx", TexRole::NormalDx),
    ("n", TexRole::Normal),
    // Roughness / gloss / metallic / occlusion
    ("roughness", TexRole::Roughness),
    ("roughnessmap", TexRole::Roughness),
    ("rough", TexRole::Roughness),
    ("rgh", TexRole::Roughness),
    ("r", TexRole::Roughness),
    ("gloss", TexRole::Gloss),
    ("glossiness", TexRole::Gloss),
    ("glossmap", TexRole::Gloss),
    ("smoothness", TexRole::Gloss),
    ("smooth", TexRole::Gloss),
    ("metallic", TexRole::Metallic),
    ("metallicmap", TexRole::Metallic),
    ("metalness", TexRole::Metallic),
    ("metal", TexRole::Metallic),
    ("met", TexRole::Metallic),
    ("mtl", TexRole::Metallic),
    ("m", TexRole::Metallic),
    ("ao", TexRole::Ao),
    ("ambientocclusion", TexRole::Ao),
    ("ambientocc", TexRole::Ao),
    ("occlusion", TexRole::Ao),
    ("occlusionmap", TexRole::Ao),
    ("occ", TexRole::Ao),
    // Packed
    ("orm", TexRole::Orm),
    ("arm", TexRole::Orm),
    ("aorm", TexRole::Orm),
    ("ormmap", TexRole::Orm),
    ("occlusionroughnessmetallic", TexRole::Orm),
    ("occlusionroughnessmetalness", TexRole::Orm),
    ("aoroughnessmetallic", TexRole::Orm),
    ("rma", TexRole::Rma),
    ("rmao", TexRole::Rma),
    ("roughnessmetallicao", TexRole::Rma),
    ("roughnessmetalnessao", TexRole::Rma),
    ("roughnessmetallicocclusion", TexRole::Rma),
    ("metallicroughness", TexRole::MetalRough),
    ("metalroughness", TexRole::MetalRough),
    ("metalrough", TexRole::MetalRough),
    ("metallicgloss", TexRole::MetalGloss),
    ("metallicglossmap", TexRole::MetalGloss),
    ("metallicsmoothness", TexRole::MetalGloss),
    ("metalsmoothness", TexRole::MetalGloss),
    ("maskmap", TexRole::MaskMap),
    // Height / bump
    ("height", TexRole::Height),
    ("heightmap", TexRole::Height),
    ("displacement", TexRole::Height),
    ("displacementmap", TexRole::Height),
    ("displace", TexRole::Height),
    ("disp", TexRole::Height),
    ("parallax", TexRole::Height),
    ("parallaxmap", TexRole::Height),
    ("h", TexRole::Height),
    ("bump", TexRole::Bump),
    ("bumpiness", TexRole::Bump),
    // Transparency
    ("opacity", TexRole::Opacity),
    ("opacitymask", TexRole::Opacity),
    ("alpha", TexRole::Opacity),
    ("transparency", TexRole::Opacity),
    ("cutout", TexRole::Opacity),
    ("translucency", TexRole::Translucency),
    ("transmission", TexRole::Translucency),
    ("transmissive", TexRole::Translucency),
    ("subsurface", TexRole::Translucency),
    ("thickness", TexRole::Translucency),
    // Other
    ("specular", TexRole::Specular),
    ("specularcolor", TexRole::Specular),
    ("specularlevel", TexRole::Specular),
    ("spec", TexRole::Specular),
    ("cavity", TexRole::Cavity),
    ("cav", TexRole::Cavity),
    ("emissive", TexRole::Emission),
    ("emissivecolor", TexRole::Emission),
    ("emission", TexRole::Emission),
    ("emissionmap", TexRole::Emission),
    ("emit", TexRole::Emission),
    ("glow", TexRole::Emission),
    ("selfillum", TexRole::Emission),
    ("e", TexRole::Emission),
    ("fuzz", TexRole::Fuzz),
    ("sheen", TexRole::Fuzz),
    ("preview", TexRole::Ignore),
    ("thumb", TexRole::Ignore),
    ("thumbnail", TexRole::Ignore),
    ("icon", TexRole::Ignore),
];

fn lookup(joined: &str) -> Option<TexRole> {
    ROLE_TABLE.iter().find(|(k, _)| *k == joined).map(|(_, r)| *r)
}

/// Image extensions the engine decodes.
pub const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "tga", "bmp", "exr", "hdr", "gif", "webp"];

/// Image formats found in packs that the engine cannot read.
pub const UNREADABLE_IMAGE_EXTS: &[&str] = &["tif", "tiff", "psd", "dds", "ktx", "ktx2", "tx", "heic", "avif"];

pub fn is_image_ext(ext: &str) -> bool {
    IMAGE_EXTS.contains(&ext.to_ascii_lowercase().as_str())
}

// ---------------------------------------------------------------------------
// Tokenising
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Token {
    start: usize,
    end: usize,
    text: String,
}

fn is_sep(c: char) -> bool {
    matches!(c, '_' | '-' | '.' | ' ' | '+' | '(' | ')')
}

/// Split on separators and on camelCase boundaries (`baseColor` -> `base`,
/// `Color`; `NormalGL` -> `Normal`, `GL`), keeping byte offsets.
fn tokenize(stem: &str) -> Vec<Token> {
    let mut out: Vec<Token> = Vec::new();
    let mut start: Option<usize> = None;
    let chars: Vec<(usize, char)> = stem.char_indices().collect();
    let flush = |out: &mut Vec<Token>, start: &mut Option<usize>, end: usize| {
        if let Some(s) = start.take() {
            out.push(Token { start: s, end, text: stem[s..end].to_owned() });
        }
    };
    for (k, &(i, c)) in chars.iter().enumerate() {
        if is_sep(c) {
            flush(&mut out, &mut start, i);
            continue;
        }
        if let Some(s) = start {
            let prev = chars[k - 1].1;
            let next = chars.get(k + 1).map(|x| x.1);
            // lower -> Upper starts a word; so does the last capital of an
            // acronym run followed by lowercase (`GLNormal` -> GL, Normal).
            let camel = (prev.is_lowercase() && c.is_uppercase()) || (prev.is_uppercase() && c.is_uppercase() && next.is_some_and(|n| n.is_lowercase()));
            // letters <-> digits split too (`Normal4K`), except a trailing K/k unit.
            let digit_edge = (prev.is_ascii_digit() != c.is_ascii_digit()) && !(prev.is_ascii_digit() && matches!(c, 'k' | 'K') && next.is_none_or(|n| is_sep(n) || !n.is_alphanumeric()));
            if camel || digit_edge {
                flush(&mut out, &mut start, i);
                start = Some(i);
            }
            let _ = s;
        } else {
            start = Some(i);
        }
    }
    flush(&mut out, &mut start, stem.len());
    out
}

/// `4K` / `8k` / `2048` as texels. Plain numbers count only from 256 up.
pub fn parse_resolution(token: &str) -> Option<u32> {
    let t = token.to_ascii_lowercase();
    if let Some(n) = t.strip_suffix('k')
        && let Ok(k) = n.parse::<u32>()
        && (1..=32).contains(&k)
    {
        return Some(k * 1024);
    }
    let n: u32 = t.parse().ok()?;
    ((256..=32768).contains(&n) && n.is_power_of_two()).then_some(n)
}

/// A UDIM tile suffix (`1001`..`1099`..) split off the stem: `Body_Color.1001`.
fn split_udim(stem: &str) -> (&str, Option<u32>) {
    if let Some(pos) = stem.rfind(['.', '_'])
        && let tail = &stem[pos + 1..]
        && tail.len() == 4
        && let Ok(n) = tail.parse::<u32>()
        && (1001..=1200).contains(&n)
    {
        return (&stem[..pos], Some(n));
    }
    (stem, None)
}

/// Classify a texture file name (with or without an extension; directories are
/// ignored). `None` = no role could be read from the name.
pub fn classify(file_name: &str) -> Option<Classified> {
    let base = Path::new(file_name).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| file_name.to_owned());
    let stem = match base.rsplit_once('.') {
        Some((s, e)) if is_image_ext(e) || UNREADABLE_IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str()) => s.to_owned(),
        _ => base.clone(),
    };
    let (stem, udim) = split_udim(&stem);
    let mut tokens = tokenize(stem);
    if tokens.is_empty() {
        return None;
    }
    let mut resolution = None;
    let mut billboard = false;
    // Trailing resolution / marker tokens: `Rock_Albedo_4K`.
    while let Some(last) = tokens.last() {
        if let Some(r) = parse_resolution(&last.text) {
            resolution = resolution.or(Some(r));
            tokens.pop();
        } else {
            break;
        }
    }
    // Longest suffix first, with and without a trailing `map` / `tex` word.
    let n = tokens.len();
    let mut found: Option<(usize, TexRole)> = None; // (tokens consumed, role)
    'outer: for take in (1..=4.min(n)).rev() {
        let joined: String = tokens[n - take..].iter().map(|t| t.text.to_ascii_lowercase()).collect();
        if let Some(r) = lookup(&joined) {
            found = Some((take, r));
            break 'outer;
        }
        // `Albedo_Map`, `Normal_Tex`: drop the filler and retry on the rest.
        if take >= 2 && matches!(tokens[n - 1].text.to_ascii_lowercase().as_str(), "map" | "tex" | "texture") {
            let rest: String = tokens[n - take..n - 1].iter().map(|t| t.text.to_ascii_lowercase()).collect();
            if let Some(r) = lookup(&rest) {
                found = Some((take, r));
                break 'outer;
            }
        }
    }
    let (take, mut role) = found?;
    let role_start = tokens[n - take].start;
    let joined: String = tokens[n - take..].iter().map(|t| t.text.to_ascii_lowercase()).collect();
    // A short alias with nothing before it is just a word, not a suffix.
    let weak = joined.len() <= 2 && !matches!(joined.as_str(), "ao" | "bc");
    if weak && n == take {
        return None;
    }
    // Everything before the role is the prefix; strip resolution / billboard markers.
    let mut pre: Vec<Token> = tokens[..n - take].to_vec();
    while let Some(last) = pre.last() {
        if let Some(r) = parse_resolution(&last.text) {
            resolution = resolution.or(Some(r));
            pre.pop();
        } else if last.text.eq_ignore_ascii_case("billboard") || last.text.eq_ignore_ascii_case("billboards") {
            billboard = true;
            pre.pop();
        } else {
            break;
        }
    }
    let prefix = match (pre.first(), pre.last()) {
        (Some(a), Some(b)) => stem[a.start..b.end].to_owned(),
        _ => String::new(),
    };
    // Unreal's `T_` prefix means DirectX normals unless the name says otherwise.
    if role == TexRole::Normal && matches!(joined.as_str(), "n" | "normal" | "nrm" | "nor") && stem.starts_with("T_") {
        role = TexRole::NormalDx;
    }
    let _ = role_start;
    Some(Classified { prefix, role, confidence: if weak { Confidence::Weak } else { Confidence::Known }, resolution, billboard, udim })
}

/// Normalise a name for loose matching (material name vs texture prefix):
/// lowercase alphanumerics only, a leading `T_` / `M_` / `MI_` dropped.
pub fn match_key(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    let trimmed = ["t_", "m_", "mi_", "mat_", "material_"].iter().find_map(|p| lower.strip_prefix(p)).unwrap_or(&lower);
    trimmed.chars().filter(|c| c.is_ascii_alphanumeric()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn role(name: &str) -> (String, TexRole) {
        let c = classify(name).unwrap_or_else(|| panic!("no role in {name}"));
        (c.prefix, c.role)
    }

    #[test]
    fn megascans_names() {
        let c = classify("uddmcgbia_4K_Basecolor.jpg").unwrap();
        assert_eq!((c.prefix.as_str(), c.role, c.resolution, c.billboard), ("uddmcgbia", TexRole::Albedo, Some(4096), false));
        let c = classify("uddmcgbia_8K_Billboard_Translucency.jpg").unwrap();
        assert_eq!((c.prefix.as_str(), c.role, c.resolution, c.billboard), ("uddmcgbia", TexRole::Translucency, Some(8192), true));
        for (n, r) in [
            ("x_8K_Albedo.jpg", TexRole::Albedo),
            ("x_8K_AO.jpg", TexRole::Ao),
            ("x_8K_Bump.jpg", TexRole::Bump),
            ("x_8K_Cavity.jpg", TexRole::Cavity),
            ("x_8K_Displacement.jpg", TexRole::Height),
            ("x_8K_Gloss.jpg", TexRole::Gloss),
            ("x_8K_Normal.jpg", TexRole::Normal),
            ("x_8K_Opacity.jpg", TexRole::Opacity),
            ("x_8K_Roughness.jpg", TexRole::Roughness),
            ("x_8K_Specular.jpg", TexRole::Specular),
            ("x_2K_Metalness.jpg", TexRole::Metallic),
            ("x_2K_Fuzz.jpg", TexRole::Fuzz),
            ("x_4K_Occlusion-Roughness-Metallic.png", TexRole::Orm),
            ("x_4K_ORM.png", TexRole::Orm),
        ] {
            assert_eq!(role(n), ("x".into(), r), "{n}");
        }
    }

    #[test]
    fn unreal_substance_gltf_unity_names() {
        assert_eq!(role("T_Rock_01_BaseColor.png"), ("T_Rock_01".into(), TexRole::Albedo));
        assert_eq!(role("T_Rock_01_ORM.png"), ("T_Rock_01".into(), TexRole::Orm));
        assert_eq!(role("T_Rock_01_N.png"), ("T_Rock_01".into(), TexRole::NormalDx), "Unreal normals are DirectX");
        assert_eq!(role("Armor_Base_color.png"), ("Armor".into(), TexRole::Albedo));
        assert_eq!(role("Armor_Normal_OpenGL.png"), ("Armor".into(), TexRole::Normal));
        assert_eq!(role("Armor_Normal_DirectX.png"), ("Armor".into(), TexRole::NormalDx));
        assert_eq!(role("Armor_normal_dx.png"), ("Armor".into(), TexRole::NormalDx));
        assert_eq!(role("Armor_Height.png"), ("Armor".into(), TexRole::Height));
        assert_eq!(role("Armor_Emissive.png"), ("Armor".into(), TexRole::Emission));
        assert_eq!(role("zzzDefault_Material_baseColor.png"), ("zzzDefault_Material".into(), TexRole::Albedo));
        assert_eq!(role("zzzDefault_Material_metallicRoughness.jpeg"), ("zzzDefault_Material".into(), TexRole::MetalRough));
        assert_eq!(role("M_MP5_Interior_ORM.jpeg"), ("M_MP5_Interior".into(), TexRole::Orm));
        assert_eq!(role("Wood_MainTex.png"), ("Wood".into(), TexRole::Albedo));
        assert_eq!(role("Wood_BumpMap.png"), ("Wood".into(), TexRole::Normal), "Unity _BumpMap is a normal map");
        assert_eq!(role("Wood_MetallicGlossMap.png"), ("Wood".into(), TexRole::MetalGloss));
        assert_eq!(role("Wood_OcclusionMap.png"), ("Wood".into(), TexRole::Ao));
        assert_eq!(role("Wood_MaskMap.png"), ("Wood".into(), TexRole::MaskMap));
        assert_eq!(role("Wood_Albedo_Map.png"), ("Wood".into(), TexRole::Albedo));
        assert_eq!(role("Hair_Alpha.png"), ("Hair".into(), TexRole::Opacity));
        assert_eq!(role("boxA_normal.jpg"), ("boxA".into(), TexRole::Normal));
    }

    #[test]
    fn resolution_and_udim_and_weak_aliases() {
        let c = classify("Rock_Albedo_2048.png").unwrap();
        assert_eq!((c.prefix.as_str(), c.resolution), ("Rock", Some(2048)));
        let c = classify("Body_BaseColor.1001.png").unwrap();
        assert_eq!((c.prefix.as_str(), c.udim), ("Body", Some(1001)));
        let c = classify("Rock_D.png").unwrap();
        assert_eq!((c.role, c.confidence), (TexRole::Albedo, Confidence::Weak));
        // A lone short word is not a suffix.
        assert!(classify("n.png").is_none());
        assert!(classify("holiday_photo.jpg").is_none());
        assert_eq!(parse_resolution("4K"), Some(4096));
        assert_eq!(parse_resolution("8k"), Some(8192));
        assert_eq!(parse_resolution("1000"), None);
        assert_eq!(parse_resolution("128"), None, "small numbers are variant numbers, not sizes");
    }

    #[test]
    fn match_key_ignores_decoration() {
        assert_eq!(match_key("T_Rock_01"), match_key("rock-01"));
        assert_eq!(match_key("M_MP5"), "mp5");
    }

    #[test]
    fn orm_layouts() {
        let l = orm_layout(TexRole::Rma).unwrap();
        assert_eq!(l[0], pick(Chan::B));
        assert_eq!(l[1], pick(Chan::R));
        assert_eq!(l[2], pick(Chan::G));
        assert!(orm_layout(TexRole::MetalRough).unwrap()[0].is_none());
        assert_eq!(orm_layout(TexRole::MetalGloss).unwrap()[1], pick_inv(Chan::A));
        assert!(orm_layout(TexRole::Albedo).is_none());
        for r in TexRole::ALL {
            assert_eq!(r.is_packed(), orm_layout(r).is_some(), "{r:?}");
        }
    }

    #[test]
    fn real_world_names_classify() {
        for (name, expect) in [("Eye_Iris_Roughness.png", TexRole::Roughness), ("Hair_Color.png", TexRole::Albedo), ("Eye_Iris_Normal.png", TexRole::Normal), ("Body_Metallic.png", TexRole::Metallic)] {
            assert_eq!(role(name).1, expect, "{name}");
        }
    }
}
