//! Unity `.mat` (text YAML) reader and shader identification.
//!
//! Reads only the *data* of a Material object (shader reference, keywords,
//! saved texture / float / colour properties); no Unity shader code is read or
//! run. Unity's YAML is a restricted dialect (`%TAG`, `--- !u!21 &id` document
//! headers, flow maps like `{r: 1, g: 0}`), so this is a tolerant line reader
//! rather than a general YAML parser: it accepts the current layout, the
//! older `first:` / `second:` property layout and multi-document files, skips
//! anything it does not understand, and never panics on garbage.

use std::collections::BTreeMap;

/// One `m_TexEnvs` entry.
#[derive(Clone, Debug, PartialEq)]
pub struct UnityTexEnv {
    /// Asset GUID of the texture (`None` when the slot is empty).
    pub guid: Option<String>,
    pub file_id: i64,
    pub scale: [f32; 2],
    pub offset: [f32; 2],
}

impl Default for UnityTexEnv {
    fn default() -> Self {
        Self { guid: None, file_id: 0, scale: [1.0, 1.0], offset: [0.0, 0.0] }
    }
}

/// A parsed Material object.
#[derive(Clone, Debug, Default)]
pub struct UnityMaterial {
    pub name: String,
    /// `m_Shader` GUID (Unity stores no shader name in the material).
    pub shader_guid: Option<String>,
    pub shader_file_id: i64,
    /// Shader name when known (filled by [`UnityMaterial::with_shader_name`],
    /// from the `.shader` file the GUID points at, or supplied by the caller).
    pub shader_name: Option<String>,
    pub keywords: Vec<String>,
    pub render_queue: Option<i32>,
    pub textures: BTreeMap<String, UnityTexEnv>,
    pub floats: BTreeMap<String, f32>,
    pub colors: BTreeMap<String, [f32; 4]>,
}

impl UnityMaterial {
    pub fn with_shader_name(mut self, name: Option<String>) -> Self {
        if name.is_some() {
            self.shader_name = name;
        }
        self
    }

    pub fn float(&self, key: &str) -> Option<f32> {
        self.floats.get(key).copied()
    }

    /// First present float among `keys`.
    pub fn float_any(&self, keys: &[&str]) -> Option<f32> {
        keys.iter().find_map(|k| self.float(k))
    }

    pub fn color_any(&self, keys: &[&str]) -> Option<[f32; 4]> {
        keys.iter().find_map(|k| self.colors.get(*k).copied())
    }

    /// First texture slot among `keys` that has a texture assigned.
    pub fn tex_any<'a>(&'a self, keys: &[&'a str]) -> Option<(&'a str, &'a UnityTexEnv)> {
        keys.iter().find_map(|k| self.textures.get(*k).filter(|t| t.guid.is_some()).map(|t| (*k, t)))
    }

    pub fn has_keyword(&self, kw: &str) -> bool {
        self.keywords.iter().any(|k| k == kw)
    }

    /// Every property name in the material (textures, floats, colours).
    pub fn all_property_names(&self) -> impl Iterator<Item = &str> {
        self.textures.keys().chain(self.floats.keys()).chain(self.colors.keys()).map(String::as_str)
    }
}

/// The all-zero-ish GUID Unity uses for built-in resources (`Standard` etc.).
pub const BUILTIN_GUID: &str = "0000000000000000f000000000000000";
/// `fileID` of the built-in Standard shader within [`BUILTIN_GUID`].
pub const BUILTIN_STANDARD_FILE_ID: i64 = 46;
pub const BUILTIN_STANDARD_SPECULAR_FILE_ID: i64 = 45;

/// Parse every Material object in `text`. Returns an empty list for text with
/// no Material (a prefab, a scene, a binary `.mat`).
pub fn parse_unity_materials(text: &str) -> Vec<UnityMaterial> {
    let mut docs: Vec<Vec<&str>> = vec![Vec::new()];
    for line in text.lines() {
        if line.starts_with("---") {
            docs.push(Vec::new());
        } else if !line.starts_with('%')
            && let Some(d) = docs.last_mut()
        {
            d.push(line);
        }
    }
    docs.iter().filter_map(|d| parse_doc(d)).collect()
}

/// True when the bytes are not Unity text YAML (Unity's "force binary"
/// serialisation); the caller reports it instead of parsing garbage.
pub fn looks_binary(bytes: &[u8]) -> bool {
    bytes.iter().take(512).any(|&b| b == 0)
}

#[derive(Clone, Copy, PartialEq)]
enum Sec {
    None,
    Tex,
    Float,
    Color,
}

fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start_matches(' ').len()
}

/// `{a: 1, b: x}` -> value of key `k` as text.
fn flow_get<'a>(s: &'a str, k: &str) -> Option<&'a str> {
    let inner = s.trim().strip_prefix('{')?.trim_end().strip_suffix('}')?;
    for part in inner.split(',') {
        let (key, val) = part.split_once(':')?;
        if key.trim() == k {
            return Some(val.trim());
        }
    }
    None
}

fn num(s: &str) -> Option<f32> {
    let v: f32 = s.trim().parse().ok()?;
    v.is_finite().then_some(v)
}

fn flow_f(s: &str, k: &str, d: f32) -> f32 {
    flow_get(s, k).and_then(num).unwrap_or(d)
}

fn parse_doc(lines: &[&str]) -> Option<UnityMaterial> {
    // A Material document has a top-level `Material:` key.
    if !lines.iter().any(|l| l.trim_end() == "Material:") {
        return None;
    }
    let mut m = UnityMaterial::default();
    let mut sec = Sec::None;
    let mut sec_indent = 0usize;
    // Current property (new layout: the `- name:` item; old layout: `name:`).
    let mut cur: Option<String> = None;
    let mut in_valid_kw = false;
    for raw in lines {
        let line = raw.trim_end();
        let t = line.trim_start();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        let ind = indent_of(line);
        let is_item = t.starts_with("- ") || t == "-";
        // Section changes: a key at or left of the section key's indent.
        if sec != Sec::None && ind <= sec_indent && !is_item {
            sec = Sec::None;
            cur = None;
        }
        if in_valid_kw {
            if let Some(kw) = t.strip_prefix("- ") {
                m.keywords.push(kw.trim().to_owned());
                continue;
            }
            in_valid_kw = false;
        }
        let body = if is_item { t.trim_start_matches('-').trim_start() } else { t };
        let Some((key, val)) = body.split_once(':') else { continue };
        let (key, val) = (key.trim(), val.trim());

        match key {
            "m_Name" => m.name = val.trim_matches(|c| c == '"' || c == '\'').to_owned(),
            "m_Shader" => {
                m.shader_guid = flow_get(val, "guid").map(str::to_owned).filter(|g| !g.is_empty() && g.chars().any(|c| c != '0'));
                m.shader_file_id = flow_get(val, "fileID").and_then(|f| f.parse().ok()).unwrap_or(0);
            }
            "m_ShaderKeywords" => m.keywords.extend(val.split_whitespace().map(str::to_owned)),
            "m_ValidKeywords" => {
                if val.starts_with('[') {
                    let inner = val.trim_start_matches('[').trim_end_matches(']');
                    m.keywords.extend(inner.split(',').map(|s| s.trim().to_owned()).filter(|s| !s.is_empty()));
                } else if val.is_empty() {
                    in_valid_kw = true;
                }
            }
            "m_CustomRenderQueue" => m.render_queue = val.parse().ok().filter(|q: &i32| *q >= 0),
            "m_TexEnvs" => (sec, sec_indent, cur) = (Sec::Tex, ind, None),
            "m_Floats" | "m_Ints" => (sec, sec_indent, cur) = (Sec::Float, ind, None),
            "m_Colors" => (sec, sec_indent, cur) = (Sec::Color, ind, None),
            _ if sec == Sec::None => {}
            // ---- inside a property section ----
            "name" => cur = Some(val.to_owned()), // old `first:` layout
            "first" | "second" | "data" if val.is_empty() => {}
            "second" => match sec {
                Sec::Float => {
                    if let (Some(n), Some(v)) = (&cur, num(val)) {
                        m.floats.insert(n.clone(), v);
                    }
                }
                Sec::Color => {
                    if let Some(n) = &cur {
                        m.colors.insert(n.clone(), color_of(val));
                    }
                }
                _ => {}
            },
            "m_Texture" if sec == Sec::Tex => {
                if let Some(n) = &cur {
                    let e = m.textures.entry(n.clone()).or_default();
                    e.guid = flow_get(val, "guid").map(str::to_owned).filter(|g| !g.is_empty() && g.chars().any(|c| c != '0'));
                    e.file_id = flow_get(val, "fileID").and_then(|f| f.parse().ok()).unwrap_or(0);
                }
            }
            "m_Scale" | "m_Offset" if sec == Sec::Tex => {
                if let Some(n) = &cur {
                    let e = m.textures.entry(n.clone()).or_default();
                    let v = [flow_f(val, "x", if key == "m_Scale" { 1.0 } else { 0.0 }), flow_f(val, "y", if key == "m_Scale" { 1.0 } else { 0.0 })];
                    if key == "m_Scale" {
                        e.scale = v;
                    } else {
                        e.offset = v;
                    }
                }
            }
            // Old block-style colour components under `second:`.
            "r" | "g" | "b" | "a" if sec == Sec::Color && !is_item => {
                if let (Some(n), Some(v)) = (&cur, num(val)) {
                    let c = m.colors.entry(n.clone()).or_insert([0.0; 4]);
                    c[match key { "r" => 0, "g" => 1, "b" => 2, _ => 3 }] = v;
                }
            }
            // `- _Name: value` item of the current layout.
            _ if is_item => {
                cur = Some(key.to_owned());
                match sec {
                    Sec::Float => {
                        if let Some(v) = num(val) {
                            m.floats.insert(key.to_owned(), v);
                        }
                    }
                    Sec::Color => {
                        m.colors.insert(key.to_owned(), color_of(val));
                    }
                    Sec::Tex => {
                        m.textures.entry(key.to_owned()).or_default();
                    }
                    Sec::None => {}
                }
            }
            _ => {}
        }
    }
    m.keywords.sort();
    m.keywords.dedup();
    Some(m)
}

fn color_of(val: &str) -> [f32; 4] {
    [flow_f(val, "r", 0.0), flow_f(val, "g", 0.0), flow_f(val, "b", 0.0), flow_f(val, "a", 1.0)]
}

/// Parse a Unity `.meta` file's `guid:` line.
pub fn parse_meta_guid(text: &str) -> Option<String> {
    text.lines().find_map(|l| {
        let g = l.trim().strip_prefix("guid:")?.trim();
        (g.len() == 32 && g.chars().all(|c| c.is_ascii_hexdigit())).then(|| g.to_ascii_lowercase())
    })
}

/// The name a `.shader` source declares in its leading `Shader "Name"` line
/// (only the declared name is read; nothing else of the file is used).
pub fn shader_declared_name(text: &str) -> Option<String> {
    for line in text.lines().take(80) {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("Shader")
            && let Some(start) = rest.find('"')
            && let Some(len) = rest[start + 1..].find('"')
        {
            return Some(rest[start + 1..start + 1 + len].to_owned());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Shader identification
// ---------------------------------------------------------------------------

/// The shader families the converter understands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnityShader {
    PoiyomiToon,
    LilToon,
    MToon,
    MochieStandard,
    /// A Mochie shader with no converter (Water, Screen FX, ...); the string is the name.
    MochieOther(String),
    UnityStandard,
    UnityStandardSpecular,
    /// URP/HDRP Lit / SimpleLit / Unlit-style (`_BaseMap` / `_BaseColor`).
    UrpLit,
    /// Recognised by nothing; converted with the generic Standard-like mapping.
    Unknown(String),
}

impl UnityShader {
    pub fn label(&self) -> String {
        match self {
            Self::PoiyomiToon => "Poiyomi Toon".into(),
            Self::LilToon => "lilToon".into(),
            Self::MToon => "VRM/MToon".into(),
            Self::MochieStandard => "Mochie Standard".into(),
            Self::MochieOther(n) => format!("Mochie ({n})"),
            Self::UnityStandard => "Unity Standard".into(),
            Self::UnityStandardSpecular => "Unity Standard (Specular)".into(),
            Self::UrpLit => "URP/Lit".into(),
            Self::Unknown(n) if n.is_empty() => "unknown shader".into(),
            Self::Unknown(n) => format!("unknown shader '{n}'"),
        }
    }

    /// True when the result is a Toon-shader material (else PBR standard).
    pub fn is_toon(&self) -> bool {
        matches!(self, Self::PoiyomiToon | Self::LilToon | Self::MToon)
    }
}

/// Identify a shader from its declared name. Handles locked / optimised
/// variants (`Hidden/Locked/.poiyomi/...`, `Hidden/lilToonOutline`, ...).
pub fn identify_by_name(name: &str) -> Option<UnityShader> {
    let n = name.to_ascii_lowercase();
    if n.contains("poiyomi") {
        return Some(UnityShader::PoiyomiToon);
    }
    if n.contains("liltoon") || n.contains("lilcustomshader") {
        return Some(UnityShader::LilToon);
    }
    if n.contains("mtoon") {
        return Some(UnityShader::MToon);
    }
    if n.contains("mochie") {
        let tail = name.rsplit('/').next().unwrap_or(name).trim();
        let t = tail.to_ascii_lowercase();
        // `Mochie/Standard`, `Mochie/Standard Lite`, `Mochie/Standard Mobile`,
        // `Locked/...` copies of them.
        return Some(if t.starts_with("standard") { UnityShader::MochieStandard } else { UnityShader::MochieOther(tail.to_owned()) });
    }
    match n.as_str() {
        "standard" => return Some(UnityShader::UnityStandard),
        "standard (specular setup)" => return Some(UnityShader::UnityStandardSpecular),
        _ => {}
    }
    if n.contains("universal render pipeline") || n.contains("hdrp/lit") || n.contains("shader graphs") {
        return Some(UnityShader::UrpLit);
    }
    None
}

/// Identify by property signature, for locked / renamed / GUID-only shaders.
pub fn identify_by_properties(m: &UnityMaterial) -> Option<UnityShader> {
    let has = |k: &str| m.textures.contains_key(k) || m.floats.contains_key(k) || m.colors.contains_key(k);
    let score = |keys: &[&str]| keys.iter().filter(|k| has(k)).count();
    // Distinctive property names, one family each.
    let lil = score(&["_lilToonVersion", "_UseShadow", "_Shadow2ndColor", "_UseRim", "_UseMatCap", "_UseEmission", "_AsUnlit", "_TransparentMode", "_RimBorder", "_ShadowBorder"]);
    let mtoon = score(&["_ShadeColor", "_ShadeToony", "_ShadeShift", "_RimFresnelPower", "_LightColorAttenuation", "_ReceiveShadowRate", "_OutlineWidthMode", "_SphereAdd", "_IndirectLightIntensity"]);
    let poi = score(&["_ShaderOptimizerEnabled", "_LightingMode", "_ShadingEnabled", "_LightingShadowStrength", "_ToonRamp", "_MainColorAdjustToggle", "_EnableRimLighting", "_RimLightColor", "_MatcapEnable", "_LightingMinLightBrightness", "_EnableAudioLink", "_StencilRef", "_AlphaMod", "_EnableOutlines", "_MatcapIntensity"]) + usize::from(m.shader_name.as_deref().is_some_and(|n| n.to_ascii_lowercase().contains("poiyomi")));
    let mochie = score(&["_RoughnessMap", "_PackedMap", "_SSRStrength", "_SSR", "_Reflections", "_Specularity", "_ReflShadowStrength"]);
    let best = [(lil, 1), (mtoon, 2), (poi, 3), (mochie, 4)].into_iter().max_by_key(|(s, _)| *s)?;
    if best.0 >= 2 {
        return Some(match best.1 {
            1 => UnityShader::LilToon,
            2 => UnityShader::MToon,
            3 => UnityShader::PoiyomiToon,
            _ => UnityShader::MochieStandard,
        });
    }
    if has("_BaseMap") && has("_BaseColor") {
        return Some(UnityShader::UrpLit);
    }
    if has("_MetallicGlossMap") || has("_Glossiness") || has("_GlossMapScale") {
        return Some(UnityShader::UnityStandard);
    }
    if has("_SpecGlossMap") || has("_SpecColor") && has("_Glossiness") {
        return Some(UnityShader::UnityStandardSpecular);
    }
    None
}

/// Full identification: built-in GUID, then shader name, then properties.
pub fn identify_shader(m: &UnityMaterial) -> UnityShader {
    if m.shader_guid.as_deref() == Some(BUILTIN_GUID) {
        match m.shader_file_id {
            BUILTIN_STANDARD_FILE_ID => return UnityShader::UnityStandard,
            BUILTIN_STANDARD_SPECULAR_FILE_ID => return UnityShader::UnityStandardSpecular,
            _ => {}
        }
    }
    if let Some(name) = m.shader_name.as_deref()
        && let Some(s) = identify_by_name(name)
    {
        return s;
    }
    if let Some(s) = identify_by_properties(m) {
        return s;
    }
    UnityShader::Unknown(m.shader_name.clone().unwrap_or_default())
}

#[cfg(test)]
pub(crate) mod fixtures {
    /// A Poiyomi-like material: property names as they appear in serialised mats.
    pub const POIYOMI: &str = "%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!21 &2100000
Material:
  serializedVersion: 8
  m_ObjectHideFlags: 0
  m_Name: Body
  m_Shader: {fileID: 4800000, guid: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, type: 3}
  m_ShaderKeywords: _ALPHATEST_ON _EMISSION
  m_LightmapFlags: 4
  m_CustomRenderQueue: 2450
  stringTagMap:
    RenderType: TransparentCutout
  disabledShaderPasses: []
  m_SavedProperties:
    serializedVersion: 3
    m_TexEnvs:
    - _MainTex:
        m_Texture: {fileID: 2800000, guid: 11111111111111111111111111111111, type: 3}
        m_Scale: {x: 2, y: 3}
        m_Offset: {x: 0.5, y: 0.25}
    - _BumpMap:
        m_Texture: {fileID: 2800000, guid: 22222222222222222222222222222222, type: 3}
        m_Scale: {x: 1, y: 1}
        m_Offset: {x: 0, y: 0}
    - _EmissionMap:
        m_Texture: {fileID: 0}
        m_Scale: {x: 1, y: 1}
        m_Offset: {x: 0, y: 0}
    - _Matcap:
        m_Texture: {fileID: 2800000, guid: 33333333333333333333333333333333, type: 3}
        m_Scale: {x: 1, y: 1}
        m_Offset: {x: 0, y: 0}
    - _FlipbookTexArray:
        m_Texture: {fileID: 2800000, guid: 44444444444444444444444444444444, type: 3}
        m_Scale: {x: 1, y: 1}
        m_Offset: {x: 0, y: 0}
    m_Ints: []
    m_Floats:
    - _Cull: 0
    - _Cutoff: 0.35
    - _EnableRimLighting: 1
    - _LightingMode: 1
    - _Mode: 1
    - _RimWidth: 0.4
    - _MatcapEnable: 1
    - _MatcapAdd: 0.7
    - _MatcapIntensity: 2
    - _EnableEmission: 1
    - _EmissionStrength: 3
    - _FlipbookTotalFrames: 4
    - _StencilType: 1
    - _AudioLinkLightingStrength: 0.5
    - _ShaderOptimizerEnabled: 0
    m_Colors:
    - _Color: {r: 0.5, g: 0.5, b: 0.5, a: 1}
    - _EmissionColor: {r: 1, g: 0.5, b: 0, a: 1}
    - _RimLightColor: {r: 1, g: 1, b: 1, a: 1}
    - _ShadowColor: {r: 0.7, g: 0.6, b: 0.8, a: 1}
";

    /// A Mochie-Standard-like material plus a second document (multi-doc).
    pub const MOCHIE: &str = "%YAML 1.1
%TAG !u! tag:unity3d.com,2011:
--- !u!21 &2100000
Material:
  m_Name: Metal
  m_Shader: {fileID: 4800000, guid: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb, type: 3}
  m_ShaderKeywords:
  m_ValidKeywords:
  - _NORMALMAP
  - _EMISSION
  m_SavedProperties:
    m_TexEnvs:
    - _MainTex:
        m_Texture: {fileID: 2800000, guid: 11111111111111111111111111111111, type: 3}
        m_Scale: {x: 1, y: 1}
        m_Offset: {x: 0, y: 0}
    - _MetallicMap:
        m_Texture: {fileID: 2800000, guid: 55555555555555555555555555555555, type: 3}
        m_Scale: {x: 1, y: 1}
        m_Offset: {x: 0, y: 0}
    - _RoughnessMap:
        m_Texture: {fileID: 2800000, guid: 66666666666666666666666666666666, type: 3}
        m_Scale: {x: 1, y: 1}
        m_Offset: {x: 0, y: 0}
    - _BumpMap:
        m_Texture: {fileID: 2800000, guid: 22222222222222222222222222222222, type: 3}
        m_Scale: {x: 1, y: 1}
        m_Offset: {x: 0, y: 0}
    m_Floats:
    - _Metallic: 0.9
    - _Roughness: 0.25
    - _BumpScale: 1.5
    - _SSR: 1
    - _Culling: 2
    m_Colors:
    - _Color: {r: 1, g: 1, b: 1, a: 1}
--- !u!21 &2100002
Material:
  m_Name: Glass
  m_Shader: {fileID: 46, guid: 0000000000000000f000000000000000, type: 0}
  m_SavedProperties:
    m_Floats:
    - _Mode: 3
    - _Glossiness: 0.9
    m_Colors:
    - _Color: {r: 1, g: 1, b: 1, a: 0.3}
--- !u!1 &100
GameObject:
  m_Name: not a material
";
}

#[cfg(test)]
mod tests {
    use super::fixtures::*;
    use super::*;

    #[test]
    fn parses_poiyomi_like_material() {
        let ms = parse_unity_materials(POIYOMI);
        assert_eq!(ms.len(), 1);
        let m = &ms[0];
        assert_eq!(m.name, "Body");
        assert_eq!(m.shader_guid.as_deref(), Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
        assert_eq!(m.keywords, vec!["_ALPHATEST_ON", "_EMISSION"]);
        assert_eq!(m.render_queue, Some(2450));
        let main = &m.textures["_MainTex"];
        assert_eq!(main.guid.as_deref(), Some("11111111111111111111111111111111"));
        assert_eq!(main.scale, [2.0, 3.0]);
        assert_eq!(main.offset, [0.5, 0.25]);
        assert!(m.textures["_EmissionMap"].guid.is_none(), "an empty slot has no guid");
        assert_eq!(m.float("_Cutoff"), Some(0.35));
        assert_eq!(m.colors["_Color"], [0.5, 0.5, 0.5, 1.0]);
    }

    #[test]
    fn parses_multi_document_and_valid_keywords() {
        let ms = parse_unity_materials(MOCHIE);
        assert_eq!(ms.len(), 2, "the GameObject document is skipped");
        assert_eq!(ms[0].keywords, vec!["_EMISSION", "_NORMALMAP"]);
        assert_eq!(ms[0].float("_Roughness"), Some(0.25));
        assert_eq!(ms[1].name, "Glass");
        assert_eq!(ms[1].shader_guid.as_deref(), Some(BUILTIN_GUID));
        assert_eq!(ms[1].shader_file_id, 46);
    }

    #[test]
    fn parses_the_old_first_second_layout() {
        let old = "--- !u!21 &1
Material:
  m_Name: Old
  m_SavedProperties:
    m_TexEnvs:
      data:
        first:
          name: _MainTex
        second:
          m_Texture: {fileID: 2800000, guid: 99999999999999999999999999999999, type: 3}
          m_Scale: {x: 4, y: 4}
          m_Offset: {x: 0, y: 0}
    m_Floats:
      data:
        first:
          name: _Cutoff
        second: 0.25
    m_Colors:
      data:
        first:
          name: _Color
        second:
          r: 0.1
          g: 0.2
          b: 0.3
          a: 1
";
        let m = &parse_unity_materials(old)[0];
        assert_eq!(m.textures["_MainTex"].scale, [4.0, 4.0]);
        assert_eq!(m.textures["_MainTex"].guid.as_deref(), Some("99999999999999999999999999999999"));
        assert_eq!(m.float("_Cutoff"), Some(0.25));
        assert_eq!(m.colors["_Color"], [0.1, 0.2, 0.3, 1.0]);
    }

    #[test]
    fn tolerates_garbage() {
        for junk in ["", "Material:", "Material:\n  m_Shader: {{{{", "--- \n--- \n---", "\u{0}\u{1}\u{2}", "Material:\n  m_SavedProperties:\n    m_Floats:\n    - : :\n    - _A: nan\n    - _B: 1e999\n    - -\n", &"Material:\n  m_Name: x\n".repeat(100)] {
            let _ = parse_unity_materials(junk);
        }
        let m = &parse_unity_materials("Material:\n  m_SavedProperties:\n    m_Floats:\n    - _A: nan\n    - _B: 2\n")[0];
        assert!(!m.floats.contains_key("_A"));
        assert_eq!(m.float("_B"), Some(2.0));
        assert!(looks_binary(&[0, 1, 2]));
    }

    #[test]
    fn identifies_by_name() {
        use UnityShader::*;
        assert_eq!(identify_by_name(".poiyomi/Poiyomi Toon"), Some(PoiyomiToon));
        assert_eq!(identify_by_name("Hidden/Locked/.poiyomi/Poiyomi Pro/abc123"), Some(PoiyomiToon));
        assert_eq!(identify_by_name("Mochie/Standard"), Some(MochieStandard));
        assert_eq!(identify_by_name("Mochie/Standard Lite"), Some(MochieStandard));
        assert_eq!(identify_by_name("Mochie/Screen FX"), Some(MochieOther("Screen FX".into())));
        assert_eq!(identify_by_name("lilToon"), Some(LilToon));
        assert_eq!(identify_by_name("Hidden/lilToonOutline"), Some(LilToon));
        assert_eq!(identify_by_name("VRM/MToon"), Some(MToon));
        assert_eq!(identify_by_name("Standard"), Some(UnityStandard));
        assert_eq!(identify_by_name("Universal Render Pipeline/Lit"), Some(UrpLit));
        assert_eq!(identify_by_name("Custom/Mine"), None);
    }

    #[test]
    fn identifies_by_guid_name_and_properties() {
        let ms = parse_unity_materials(MOCHIE);
        assert_eq!(identify_shader(&ms[1]), UnityShader::UnityStandard, "built-in guid");
        // Locked / renamed: only the property signature remains.
        let poi = parse_unity_materials(POIYOMI)[0].clone();
        assert_eq!(identify_shader(&poi), UnityShader::PoiyomiToon);
        let named = parse_unity_materials(MOCHIE)[0].clone().with_shader_name(Some("Mochie/Standard".into()));
        assert_eq!(identify_shader(&named), UnityShader::MochieStandard);
        let unknown = UnityMaterial::default();
        assert!(matches!(identify_shader(&unknown), UnityShader::Unknown(_)));
    }

    #[test]
    fn reads_meta_guid_and_shader_name() {
        assert_eq!(parse_meta_guid("fileFormatVersion: 2\nguid: ABCDEFabcdef0123456789abcdef0123\n"), Some("abcdefabcdef0123456789abcdef0123".into()));
        assert_eq!(parse_meta_guid("guid: short"), None);
        assert_eq!(shader_declared_name("// c\nShader \".poiyomi/Poiyomi Toon\"\n{"), Some(".poiyomi/Poiyomi Toon".into()));
        assert_eq!(shader_declared_name("nothing here"), None);
    }
}
