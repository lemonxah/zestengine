//! Custom-shader property declarations (v1 pragmas and v2 `@property` blocks),
//! their storage layout (push constants, std140 material block, keyword
//! specialization constants, texture bindings) and value packing.
//!
//! Values live in a flat "value space" (`MaterialModel::custom_values`):
//! lanes 0..15 mirror the v1 push block, v2 properties follow from
//! [`SHADER_VALUE_BASE`]. Saved materials store values by property name.

use std::collections::BTreeMap;

use anyhow::{Context as _, Result, bail};

/// Push-constant floats available to v1 properties (the 16th lane carries the
/// baked lightmap layer).
pub const SHADER_PROP_FLOATS: usize = 15;
/// First value-space lane used by v2 properties.
pub const SHADER_VALUE_BASE: usize = 16;
/// Keyword (specialization constant) slots per shader; constant ids 0..N.
pub const MAX_SHADER_KEYWORDS: usize = 32;
/// Texture properties per shader (set 2, bindings 1..=N).
pub const MAX_SHADER_TEXTURES: usize = 24;
/// Size cap of the generated std140 material block (set 2, binding 0).
pub const SHADER_UBO_MAX_BYTES: usize = 4096;
/// Specialization constant id carrying the material's alpha mode.
pub const ALPHA_MODE_CONSTANT_ID: u32 = 63;
/// `MaterialTexturePaths` slot index of the first custom texture property.
pub const SHADER_TEXTURE_SLOT_BASE: usize = 16;

/// Keyword values for one material: index = constant id.
pub type KeywordSet = [u8; MAX_SHADER_KEYWORDS];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderLang {
    Glsl,
    Hlsl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureDefault {
    White,
    Black,
    Normal,
    Gray,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ShaderPropKind {
    /// Slider (v1 `float`, v2 `Range(min, max)`).
    Float { min: f32, max: f32 },
    /// Unbounded float (v2 `Float`).
    Drag,
    /// Integer; `min == max` means unbounded.
    Int { min: i32, max: i32 },
    Toggle,
    Enum(Vec<String>),
    /// RGBA.
    Color { hdr: bool },
    /// RGB.
    Color3 { hdr: bool },
    /// 2..=4 floats.
    Vector(u8),
    /// Sampler + a `<name>_st` vec4 (tiling xy, offset zw) in the value space.
    Texture { default: TextureDefault, srgb: bool, binding: u32 },
}

impl ShaderPropKind {
    /// Value-space floats.
    pub fn size(&self) -> usize {
        match self {
            Self::Float { .. } | Self::Drag | Self::Int { .. } | Self::Toggle | Self::Enum(_) => 1,
            Self::Color3 { .. } => 3,
            Self::Color { .. } | Self::Texture { .. } => 4,
            Self::Vector(n) => *n as usize,
        }
    }

    fn is_integer(&self) -> bool {
        matches!(self, Self::Int { .. } | Self::Enum(_))
    }

    /// std140 (size, alignment) in bytes of the block member.
    fn std140(&self) -> (u32, u32) {
        match self.size() {
            1 => (4, 4),
            2 => (8, 8),
            3 => (12, 16),
            _ => (16, 16),
        }
    }

    pub fn glsl_type(&self) -> &'static str {
        match self {
            Self::Int { .. } | Self::Enum(_) => "int",
            _ => ["float", "vec2", "vec3", "vec4"][self.size() - 1],
        }
    }

    pub fn hlsl_type(&self) -> &'static str {
        match self {
            Self::Int { .. } | Self::Enum(_) => "int",
            _ => ["float", "float2", "float3", "float4"][self.size() - 1],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PropStorage {
    /// v1: lanes of the 16-float push block.
    Push { offset: usize },
    /// v2: byte offset in the std140 material block.
    Uniform { offset: u32 },
    /// v2 `@keyword`: specialization constant id.
    Keyword { id: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderCondOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl ShaderCondOp {
    pub fn test(self, a: f32, b: f32) -> bool {
        match self {
            Self::Eq => (a - b).abs() < 0.5,
            Self::Ne => (a - b).abs() >= 0.5,
            Self::Lt => a < b,
            Self::Le => a <= b,
            Self::Gt => a > b,
            Self::Ge => a >= b,
        }
    }
}

/// One comparison against another property's first value lane.
#[derive(Clone, Debug, PartialEq)]
pub struct ShaderCondTerm {
    /// Value-space offset of the tested property.
    pub offset: usize,
    pub op: ShaderCondOp,
    pub value: f32,
}

/// OR of AND-groups; empty = always true.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ShaderCondition(pub Vec<Vec<ShaderCondTerm>>);

impl ShaderCondition {
    pub fn eval(&self, values: &[f32]) -> bool {
        self.0.is_empty()
            || self.0.iter().any(|all| {
                all.iter()
                    .all(|t| t.op.test(values.get(t.offset).copied().unwrap_or(0.0), t.value))
            })
    }
}

#[derive(Clone, Debug)]
pub struct ShaderProp {
    /// Shader identifier (also the key in saved material files).
    pub name: String,
    pub label: String,
    pub kind: ShaderPropKind,
    pub default: Vec<f32>,
    /// Value-space offset.
    pub offset: usize,
    pub storage: PropStorage,
    /// Index into `ShaderSource::groups`.
    pub group: Option<usize>,
    /// Sub-heading drawn above this property.
    pub header: Option<String>,
    pub show_if: ShaderCondition,
    pub tooltip: Option<String>,
}

impl ShaderProp {
    pub fn is_keyword(&self) -> bool {
        matches!(self.storage, PropStorage::Keyword { .. })
    }
}

#[derive(Clone, Debug)]
pub struct ShaderGroup {
    pub label: String,
    /// Header checkbox: index into `props` (a Toggle, usually a keyword).
    pub toggle: Option<usize>,
    pub show_if: ShaderCondition,
}

#[derive(Clone, Debug)]
pub struct ShaderPreset {
    pub name: String,
    /// (prop index, values).
    pub values: Vec<(usize, Vec<f32>)>,
}

#[derive(Clone, Debug)]
pub struct ShaderSource {
    pub display_name: String,
    pub lang: ShaderLang,
    pub props: Vec<ShaderProp>,
    pub groups: Vec<ShaderGroup>,
    pub presets: Vec<ShaderPreset>,
    pub body: String,
    /// The `// @vertex` ... `// @endvertex` section (a vertex hook), cut out of `body`.
    /// None for the shaders that have no hook, which compile exactly as before.
    pub vertex: Option<VertexSection>,
    /// std140 block size in bytes (0 = no material block).
    pub ubo_size: u32,
    /// Value-space length (>= 16).
    pub value_len: usize,
}

/// A custom shader's vertex-hook section: GLSL that defines
/// `void zestengine_vertex(inout VertexData v)` (see docs/shaders.md, "Vertex hook").
#[derive(Clone, Debug, PartialEq)]
pub struct VertexSection {
    /// The section's text (without the marker lines).
    pub body: String,
    /// 1-based line of the file the first line of `body` is on (for compiler messages).
    pub first_line: u32,
}

/// Split the `// @vertex` ... `// @endvertex` section out of `text`. Returns the text with the
/// section's lines blanked (so the fragment stage's line numbers stay those of the file) and the
/// section. An unterminated section runs to the end of the file; a second one is an error.
pub fn split_vertex_section(text: &str) -> Result<(String, Option<VertexSection>)> {
    let mut out = String::with_capacity(text.len());
    let mut section: Option<VertexSection> = None;
    let mut open: Option<usize> = None;
    for (i, line) in text.split_inclusive('\n').enumerate() {
        let directive = line.trim().strip_prefix("//").map(|r| r.trim_start()).and_then(|r| r.strip_prefix('@')).map(|d| d.trim());
        match (directive, open) {
            (Some("vertex"), None) => {
                if section.is_some() {
                    bail!("shader has two `// @vertex` sections (line {})", i + 1);
                }
                section = Some(VertexSection { body: String::new(), first_line: i as u32 + 2 });
                open = Some(i);
                out.push('\n');
            }
            (Some("endvertex"), Some(_)) => {
                open = None;
                out.push('\n');
            }
            (_, Some(_)) => {
                if let Some(sec) = section.as_mut() {
                    sec.body.push_str(line);
                }
                out.push('\n');
            }
            _ => out.push_str(line),
        }
    }
    Ok((out, section.filter(|s| !s.body.trim().is_empty())))
}

impl ShaderSource {
    pub fn defaults(&self) -> Vec<f32> {
        let mut out = vec![0.0; self.value_len];
        for prop in &self.props {
            out[prop.offset..prop.offset + prop.kind.size()].copy_from_slice(&prop.default);
        }
        out
    }

    /// Defaults overlaid with named values (saved material data).
    pub fn pack(&self, named: &BTreeMap<String, Vec<f32>>) -> Vec<f32> {
        let mut out = self.defaults();
        for prop in &self.props {
            if let Some(values) = named.get(&prop.name)
                && values.len() == prop.kind.size()
            {
                out[prop.offset..prop.offset + values.len()].copy_from_slice(values);
            }
        }
        out
    }

    /// Value space -> named map for saving.
    pub fn unpack(&self, values: &[f32]) -> BTreeMap<String, Vec<f32>> {
        let mut out = BTreeMap::new();
        for prop in &self.props {
            let end = prop.offset + prop.kind.size();
            if end <= values.len() {
                out.insert(prop.name.clone(), values[prop.offset..end].to_vec());
            }
        }
        out
    }

    /// Carry values across a layout change (hot reload), matched by name.
    pub fn remap_from(&self, old: &ShaderSource, values: &[f32]) -> Vec<f32> {
        self.pack(&old.unpack(values))
    }

    /// The 16 push-constant floats (v1 lanes; lane 15 is engine-owned).
    pub fn push_data(&self, values: &[f32]) -> [f32; 16] {
        let mut out = [0.0; 16];
        let n = values.len().min(SHADER_PROP_FLOATS);
        out[..n].copy_from_slice(&values[..n]);
        out
    }

    /// The std140 material block bytes.
    pub fn uniform_bytes(&self, values: &[f32]) -> Vec<u8> {
        let mut out = vec![0u8; self.ubo_size as usize];
        for prop in &self.props {
            let PropStorage::Uniform { offset } = prop.storage else {
                continue;
            };
            for i in 0..prop.kind.size() {
                let v = values.get(prop.offset + i).copied().unwrap_or(0.0);
                let bytes = if prop.kind.is_integer() {
                    (v.round() as i32).to_le_bytes()
                } else {
                    v.to_le_bytes()
                };
                let at = offset as usize + i * 4;
                out[at..at + 4].copy_from_slice(&bytes);
            }
        }
        out
    }

    /// Keyword values (index = constant id) for the pipeline key.
    pub fn keywords(&self, values: &[f32]) -> KeywordSet {
        let mut out = [0u8; MAX_SHADER_KEYWORDS];
        for prop in &self.props {
            if let PropStorage::Keyword { id } = prop.storage {
                let v = values.get(prop.offset).copied().unwrap_or(0.0);
                out[id as usize] = match prop.kind {
                    ShaderPropKind::Toggle => (v != 0.0) as u8,
                    _ => v.round().clamp(0.0, 255.0) as u8,
                };
            }
        }
        out
    }

    pub fn keyword_count(&self) -> usize {
        self.props.iter().filter(|p| p.is_keyword()).count()
    }

    /// Texture properties in binding order.
    pub fn textures(&self) -> impl Iterator<Item = &ShaderProp> {
        self.props.iter().filter(|p| matches!(p.kind, ShaderPropKind::Texture { .. }))
    }

    /// Whether materials need the set-2 material block / textures.
    pub fn has_material_set(&self) -> bool {
        self.ubo_size > 0 || self.textures().next().is_some()
    }

    pub fn prop(&self, name: &str) -> Option<&ShaderProp> {
        self.props.iter().find(|p| p.name == name)
    }

    /// Apply a preset's assignments to a value space.
    pub fn apply_preset(&self, preset: usize, values: &mut [f32]) {
        let Some(preset) = self.presets.get(preset) else {
            return;
        };
        for (pi, v) in &preset.values {
            let p = &self.props[*pi];
            if p.offset + v.len() <= values.len() {
                values[p.offset..p.offset + v.len()].copy_from_slice(v);
            }
        }
    }
}

struct Parser {
    lang: ShaderLang,
    display_name: String,
    props: Vec<ShaderProp>,
    raw_conds: Vec<(usize, String, usize)>,
    groups: Vec<ShaderGroup>,
    raw_group_conds: Vec<(usize, String, usize)>,
    raw_presets: Vec<(String, String, usize)>,
    push_offset: usize,
    value_offset: usize,
    ubo_offset: u32,
    next_keyword: u32,
    next_binding: u32,
    group: Option<usize>,
    pending_header: Option<String>,
    block_show_if: Option<String>,
}

/// Parse property declarations and allocate storage.
pub fn parse_shader_source(text: &str, fallback_name: &str) -> Result<ShaderSource> {
    parse_shader_source_lang(text, fallback_name, ShaderLang::Glsl)
}

pub fn parse_shader_source_lang(text: &str, fallback_name: &str, lang: ShaderLang) -> Result<ShaderSource> {
    let mut p = Parser {
        lang,
        display_name: fallback_name.to_owned(),
        props: Vec::new(),
        raw_conds: Vec::new(),
        groups: Vec::new(),
        raw_group_conds: Vec::new(),
        raw_presets: Vec::new(),
        push_offset: 0,
        value_offset: SHADER_VALUE_BASE,
        ubo_offset: 0,
        next_keyword: 0,
        next_binding: 1,
        group: None,
        pending_header: None,
        block_show_if: None,
    };

    for (line_no, line) in text.lines().enumerate() {
        let line_no = line_no + 1;
        let t = line.trim();
        if let Some(pragma) = t.strip_prefix("//!") {
            p.v1_pragma(pragma.trim(), line_no)?;
        } else if let Some(rest) = t.strip_prefix("//") {
            let rest = rest.trim_start();
            if let Some(directive) = rest.strip_prefix('@') {
                p.v2_directive(directive.trim(), line_no)
                    .with_context(|| format!("shader directive on line {line_no}"))?;
            }
        }
    }
    let mut source = p.finish()?;
    // HLSL has no vertex hook (the hook is GLSL and spliced into the engine's GLSL vertex shaders).
    if lang == ShaderLang::Glsl {
        let (body, vertex) = split_vertex_section(text)?;
        source.body = body;
        source.vertex = vertex;
    } else {
        source.body = text.to_owned();
    }
    Ok(source)
}

impl Parser {
    fn v1_pragma(&mut self, pragma: &str, line_no: usize) -> Result<()> {
        if let Some(name) = pragma.strip_prefix("shader") {
            self.display_name = name.trim().trim_matches('"').to_owned();
        } else if let Some(decl) = pragma.strip_prefix("prop") {
            let prop = parse_v1_prop(decl.trim(), &mut self.push_offset)
                .with_context(|| format!("shader pragma on line {line_no}"))?;
            self.push(prop, line_no)?;
        }
        Ok(())
    }

    fn push(&mut self, prop: ShaderProp, line_no: usize) -> Result<usize> {
        if self.props.iter().any(|p| p.name == prop.name) {
            bail!("duplicate property {:?} (line {line_no})", prop.name);
        }
        self.props.push(prop);
        Ok(self.props.len() - 1)
    }

    fn v2_directive(&mut self, d: &str, line_no: usize) -> Result<()> {
        let (word, rest) = split_word(d);
        match word {
            "shader" => self.display_name = unquote(rest.trim()).0,
            "property" => {
                self.v2_property(rest, line_no)?;
            }
            "feature" => {
                // `@feature NAME "Label" [= 0|1] [@show_if(..)]`: a keyword toggle
                // that opens a foldout whose header carries it.
                let (name, rest) = split_word(rest.trim());
                let (label, rest) = if rest.trim_start().starts_with('"') {
                    unquote(rest.trim())
                } else {
                    (title_case(name), rest.to_owned())
                };
                let (rest, show_if) = take_attr(&rest, "show_if");
                let default = rest
                    .trim()
                    .strip_prefix('=')
                    .map(|v| v.trim().parse::<f32>())
                    .transpose()
                    .context("bad @feature default")?
                    .unwrap_or(0.0);
                self.group = None;
                self.block_show_if = None;
                let decl = format!("{name} \"{label}\" Toggle = {default} @keyword");
                let idx = self.v2_property(&decl, line_no)?;
                self.groups.push(ShaderGroup { label, toggle: Some(idx), show_if: ShaderCondition::default() });
                let gi = self.groups.len() - 1;
                self.props[idx].group = Some(gi);
                if let Some(cond) = show_if {
                    self.raw_group_conds.push((gi, cond, line_no));
                }
                self.group = Some(gi);
            }
            "foldout" => {
                let (label, rest) = unquote(rest.trim());
                let (_, show_if) = take_attr(&rest, "show_if");
                self.groups.push(ShaderGroup { label, toggle: None, show_if: ShaderCondition::default() });
                let gi = self.groups.len() - 1;
                if let Some(cond) = show_if {
                    self.raw_group_conds.push((gi, cond, line_no));
                }
                self.group = Some(gi);
                self.block_show_if = None;
            }
            "endfoldout" | "end" => {
                self.group = None;
                self.block_show_if = None;
            }
            "header" => self.pending_header = Some(unquote(rest.trim()).0),
            "show_if" => self.block_show_if = Some(rest.trim().to_owned()),
            "end_show_if" => self.block_show_if = None,
            "preset" => {
                let (name, rest) = unquote(rest.trim());
                self.raw_presets.push((name, rest, line_no));
            }
            // Unknown directives are ignored (forward compatibility).
            _ => {}
        }
        Ok(())
    }

    /// `name ["Label"] Type[(args)] [= default] [@keyword] [@linear|@srgb]
    /// [@show_if(expr)] [@tooltip("text")]`
    fn v2_property(&mut self, decl: &str, line_no: usize) -> Result<usize> {
        let (decl, show_if) = take_attr(decl, "show_if");
        let (decl, tooltip) = take_attr(&decl, "tooltip");
        let mut flags = Vec::new();
        let mut core = String::new();
        for tok in decl.split_whitespace() {
            if let Some(f) = tok.strip_prefix('@') {
                flags.push(f.to_ascii_lowercase());
            } else {
                core.push_str(tok);
                core.push(' ');
            }
        }
        let keyword = flags.iter().any(|f| f == "keyword");
        let linear = flags.iter().any(|f| f == "linear");

        let (name, rest) = split_word(core.trim());
        let name = name.to_owned();
        validate_ident(&name)?;
        let (label, rest) = if rest.trim_start().starts_with('"') {
            unquote(rest.trim())
        } else {
            (title_case(&name), rest.to_owned())
        };
        let (type_part, default_part) = match rest.split_once('=') {
            Some((t, d)) => (t.trim().to_owned(), Some(d.trim().to_owned())),
            None => (rest.trim().to_owned(), None),
        };
        let (type_name, args) = match type_part.split_once('(') {
            Some((t, a)) => (
                t.trim().to_owned(),
                Some(a.trim_end().strip_suffix(')').context("unclosed type arguments")?.to_owned()),
            ),
            None => (type_part.trim().to_owned(), None),
        };
        let args: Vec<String> = args
            .map(|a| a.split(',').map(|s| s.trim().to_owned()).filter(|s| !s.is_empty()).collect())
            .unwrap_or_default();
        let num = |s: &str| s.parse::<f32>().with_context(|| format!("bad number {s:?}"));
        let hdr = args.iter().any(|a| a.eq_ignore_ascii_case("hdr"));

        let kind = match type_name.to_ascii_lowercase().as_str() {
            "range" => {
                let [a, b] = args.as_slice() else { bail!("Range takes (min, max)") };
                ShaderPropKind::Float { min: num(a)?, max: num(b)? }
            }
            "float" => ShaderPropKind::Drag,
            "int" => ShaderPropKind::Int { min: 0, max: 0 },
            "intrange" => {
                let [a, b] = args.as_slice() else { bail!("IntRange takes (min, max)") };
                ShaderPropKind::Int { min: num(a)? as i32, max: num(b)? as i32 }
            }
            "toggle" => ShaderPropKind::Toggle,
            "enum" => {
                if args.is_empty() {
                    bail!("Enum needs at least one option");
                }
                for a in &args {
                    validate_ident(a).with_context(|| format!("enum option {a:?}"))?;
                }
                ShaderPropKind::Enum(args.clone())
            }
            "color" => ShaderPropKind::Color { hdr },
            "color3" => ShaderPropKind::Color3 { hdr },
            "vector" | "vector4" => ShaderPropKind::Vector(4),
            "vector3" => ShaderPropKind::Vector(3),
            "vector2" => ShaderPropKind::Vector(2),
            "texture2d" | "texture" => {
                let default = match default_part.as_deref().map(str::to_ascii_lowercase).as_deref() {
                    None | Some("white") => TextureDefault::White,
                    Some("black") => TextureDefault::Black,
                    Some("normal" | "bump") => TextureDefault::Normal,
                    Some("gray" | "grey") => TextureDefault::Gray,
                    Some(other) => bail!("unknown texture default {other:?} (white, black, normal, gray)"),
                };
                let srgb = !linear && default != TextureDefault::Normal && default != TextureDefault::Gray
                    || flags.iter().any(|f| f == "srgb");
                if self.next_binding as usize > MAX_SHADER_TEXTURES {
                    bail!("too many texture properties (max {MAX_SHADER_TEXTURES})");
                }
                let binding = self.next_binding;
                self.next_binding += 1;
                ShaderPropKind::Texture { default, srgb, binding }
            }
            other => bail!(
                "unknown property type {other:?} (Range, Float, Int, IntRange, Toggle, Enum, Color, Color3, Vector, Vector2, Vector3, Texture2D)"
            ),
        };

        let size = kind.size();
        let default = match (&kind, default_part.as_deref()) {
            (ShaderPropKind::Texture { .. }, _) => vec![1.0, 1.0, 0.0, 0.0],
            (ShaderPropKind::Enum(opts), Some(d)) => match opts.iter().position(|o| o == d) {
                Some(i) => vec![i as f32],
                None => vec![num(d).with_context(|| format!("{d:?} is not an option of {name:?}"))?],
            },
            (_, Some(d)) => {
                let d = d.trim().trim_start_matches('(').trim_end_matches(')');
                let mut vals = d.split(',').map(|v| num(v.trim())).collect::<Result<Vec<f32>>>()?;
                if vals.len() == 1 && size > 1 {
                    vals = vec![vals[0]; size];
                    if matches!(kind, ShaderPropKind::Color { .. }) {
                        vals[3] = 1.0;
                    }
                }
                if vals.len() == 3 && size == 4 && matches!(kind, ShaderPropKind::Color { .. }) {
                    vals.push(1.0);
                }
                if vals.len() != size {
                    bail!("default for {name:?} needs {size} value(s), got {}", vals.len());
                }
                vals
            }
            (ShaderPropKind::Float { min, .. }, None) => vec![*min],
            (ShaderPropKind::Int { min, .. }, None) => vec![*min as f32],
            (ShaderPropKind::Color { .. }, None) => vec![1.0; 4],
            (ShaderPropKind::Color3 { .. }, None) => vec![1.0; 3],
            (_, None) => vec![0.0; size],
        };

        let storage = if keyword {
            if !matches!(kind, ShaderPropKind::Toggle | ShaderPropKind::Enum(_) | ShaderPropKind::Int { .. }) {
                bail!("@keyword needs a Toggle, Enum, Int or IntRange property ({name:?})");
            }
            if self.next_keyword as usize >= MAX_SHADER_KEYWORDS {
                bail!("too many keywords (max {MAX_SHADER_KEYWORDS})");
            }
            let id = self.next_keyword;
            self.next_keyword += 1;
            PropStorage::Keyword { id }
        } else {
            let (bytes, align) = kind.std140();
            let at = self.ubo_offset.div_ceil(align) * align;
            self.ubo_offset = at + bytes;
            if self.ubo_offset as usize > SHADER_UBO_MAX_BYTES {
                bail!("material block exceeds {SHADER_UBO_MAX_BYTES} bytes");
            }
            PropStorage::Uniform { offset: at }
        };

        let offset = self.value_offset;
        self.value_offset += size;
        let prop = ShaderProp {
            name,
            label,
            kind,
            default,
            offset,
            storage,
            group: self.group,
            header: self.pending_header.take(),
            show_if: ShaderCondition::default(),
            tooltip: tooltip.map(|t| unquote(&t).0),
        };
        let idx = self.push(prop, line_no)?;
        let mut conds: Vec<String> = Vec::new();
        if let Some(c) = &self.block_show_if {
            conds.push(c.clone());
        }
        if let Some(c) = show_if {
            conds.push(c);
        }
        if !conds.is_empty() {
            // Block + inline conditions both apply (AND).
            let joined = if conds.len() == 2 && !conds.iter().any(|c| c.contains("||")) {
                conds.join(" && ")
            } else if conds.len() == 2 {
                conds[1].clone()
            } else {
                conds.remove(0)
            };
            self.raw_conds.push((idx, joined, line_no));
        }
        Ok(idx)
    }

    fn finish(mut self) -> Result<ShaderSource> {
        for (idx, raw, line) in std::mem::take(&mut self.raw_conds) {
            let cond = parse_condition(&raw, &self.props).with_context(|| format!("@show_if on line {line}"))?;
            self.props[idx].show_if = cond;
        }
        for (gi, raw, line) in std::mem::take(&mut self.raw_group_conds) {
            let cond = parse_condition(&raw, &self.props).with_context(|| format!("@show_if on line {line}"))?;
            self.groups[gi].show_if = cond;
        }
        let mut presets = Vec::new();
        for (name, raw, line) in std::mem::take(&mut self.raw_presets) {
            let values = parse_preset(&raw, &self.props).with_context(|| format!("@preset on line {line}"))?;
            presets.push(ShaderPreset { name, values });
        }
        let ubo_size = self.ubo_offset.div_ceil(16) * 16;
        Ok(ShaderSource {
            display_name: self.display_name,
            lang: self.lang,
            props: self.props,
            groups: self.groups,
            presets,
            body: String::new(),
            vertex: None,
            ubo_size,
            value_len: self.value_offset,
        })
    }
}

fn parse_v1_prop(decl: &str, offset: &mut usize) -> Result<ShaderProp> {
    let mut parts = decl.split_whitespace();
    let name = parts.next().context("missing property name")?.to_owned();
    validate_ident(&name)?;
    let kind_word = parts.next().context("missing property kind")?;
    let rest: String = parts.collect::<Vec<_>>().join(" ");

    let range = parse_call(&rest, "range")?;
    let default = parse_call(&rest, "default")?;

    let kind = match kind_word {
        "float" => {
            let (min, max) = match range.as_deref() {
                Some([min, max]) => (*min, *max),
                Some(_) => bail!("range() takes two values"),
                None => (0.0, 1.0),
            };
            ShaderPropKind::Float { min, max }
        }
        "toggle" => ShaderPropKind::Toggle,
        "color" => ShaderPropKind::Color { hdr: false },
        "color3" => ShaderPropKind::Color3 { hdr: false },
        other => bail!("unknown property kind {other:?} (float, toggle, color, color3)"),
    };

    let size = kind.size();
    let default = match default {
        Some(values) if values.len() == size => values,
        Some(values) => bail!("default() for {name:?} needs {size} value(s), got {}", values.len()),
        None => match kind {
            ShaderPropKind::Float { min, .. } => vec![min],
            ShaderPropKind::Toggle => vec![0.0],
            ShaderPropKind::Color { .. } => vec![1.0; 4],
            _ => vec![1.0; 3],
        },
    };

    // Vec-valued properties must not straddle a vec4 (their define is a
    // swizzle of one push vec4); color3 from component 2+ has no swizzle.
    let mut at = *offset;
    if (at % 4) + size > 4 || (size == 3 && at % 4 > 1) {
        at = at.div_ceil(4) * 4;
    }
    if at + size > SHADER_PROP_FLOATS {
        bail!("too many properties: the push block holds {SHADER_PROP_FLOATS} floats");
    }
    *offset = at + size;

    Ok(ShaderProp {
        label: title_case(&name),
        name,
        kind,
        default,
        offset: at,
        storage: PropStorage::Push { offset: at },
        group: None,
        header: None,
        show_if: ShaderCondition::default(),
        tooltip: None,
    })
}

/// `a && b || !c && d == Option && e >= 2`
fn parse_condition(raw: &str, props: &[ShaderProp]) -> Result<ShaderCondition> {
    let raw = raw.trim().trim_start_matches('(').trim_end_matches(')');
    let mut any = Vec::new();
    for alt in raw.split("||") {
        let mut all = Vec::new();
        for term in alt.split("&&") {
            all.push(parse_term(term.trim(), props)?);
        }
        any.push(all);
    }
    Ok(ShaderCondition(any))
}

fn parse_term(term: &str, props: &[ShaderProp]) -> Result<ShaderCondTerm> {
    let find = |name: &str| {
        props
            .iter()
            .find(|p| p.name == name)
            .with_context(|| format!("unknown property {name:?} in condition"))
    };
    for (sym, op) in [
        ("==", ShaderCondOp::Eq),
        ("!=", ShaderCondOp::Ne),
        ("<=", ShaderCondOp::Le),
        (">=", ShaderCondOp::Ge),
        ("<", ShaderCondOp::Lt),
        (">", ShaderCondOp::Gt),
    ] {
        if let Some((l, r)) = term.split_once(sym) {
            let p = find(l.trim())?;
            let r = r.trim();
            let value = match &p.kind {
                ShaderPropKind::Enum(opts) if opts.iter().any(|o| o == r) => {
                    opts.iter().position(|o| o == r).unwrap() as f32
                }
                _ => r.parse::<f32>().with_context(|| format!("bad value {r:?} in condition"))?,
            };
            return Ok(ShaderCondTerm { offset: p.offset, op, value });
        }
    }
    if let Some(name) = term.strip_prefix('!') {
        let p = find(name.trim())?;
        return Ok(ShaderCondTerm { offset: p.offset, op: ShaderCondOp::Eq, value: 0.0 });
    }
    let p = find(term)?;
    Ok(ShaderCondTerm { offset: p.offset, op: ShaderCondOp::Ne, value: 0.0 })
}

/// `a=1 b=Option c=(1,0,0,1)`
fn parse_preset(raw: &str, props: &[ShaderProp]) -> Result<Vec<(usize, Vec<f32>)>> {
    let mut out = Vec::new();
    let mut rest = raw.trim();
    while !rest.is_empty() {
        let (name, after) = rest.split_once('=').context("expected name=value")?;
        let name = name.trim();
        let after = after.trim_start();
        let (value, next) = if let Some(inner) = after.strip_prefix('(') {
            let close = inner.find(')').context("unclosed (")?;
            (&inner[..close], &inner[close + 1..])
        } else {
            match after.find(char::is_whitespace) {
                Some(i) => (&after[..i], &after[i..]),
                None => (after, ""),
            }
        };
        let pi = props
            .iter()
            .position(|p| p.name == name)
            .with_context(|| format!("unknown property {name:?} in preset"))?;
        let p = &props[pi];
        let vals = match &p.kind {
            ShaderPropKind::Enum(opts) if opts.iter().any(|o| o == value.trim()) => {
                vec![opts.iter().position(|o| o == value.trim()).unwrap() as f32]
            }
            _ => value
                .split(',')
                .map(|v| v.trim().parse::<f32>().with_context(|| format!("bad value {v:?} for {name:?}")))
                .collect::<Result<Vec<_>>>()?,
        };
        if vals.len() != p.kind.size() {
            bail!("preset value for {name:?} needs {} value(s)", p.kind.size());
        }
        out.push((pi, vals));
        rest = next.trim_start().trim_start_matches(',').trim_start();
    }
    Ok(out)
}

/// Remove `@name(...)` from a declaration, returning the rest and the args.
fn take_attr(decl: &str, name: &str) -> (String, Option<String>) {
    let pat = format!("@{name}(");
    let Some(start) = decl.find(&pat) else {
        return (decl.to_owned(), None);
    };
    let inner_start = start + pat.len();
    let mut depth = 1;
    let mut in_str = false;
    for (i, c) in decl[inner_start..].char_indices() {
        match c {
            '"' => in_str = !in_str,
            '(' if !in_str => depth += 1,
            ')' if !in_str => {
                depth -= 1;
                if depth == 0 {
                    let arg = decl[inner_start..inner_start + i].to_owned();
                    let rest = format!("{}{}", &decl[..start], &decl[inner_start + i + 1..]);
                    return (rest, Some(arg));
                }
            }
            _ => {}
        }
    }
    (decl.to_owned(), None)
}

fn split_word(s: &str) -> (&str, &str) {
    let s = s.trim_start();
    match s.find(char::is_whitespace) {
        Some(i) => (&s[..i], &s[i..]),
        None => (s, ""),
    }
}

/// Leading `"quoted"` string and the remainder (or the whole word unquoted).
fn unquote(s: &str) -> (String, String) {
    let s = s.trim_start();
    if let Some(inner) = s.strip_prefix('"')
        && let Some(end) = inner.find('"')
    {
        return (inner[..end].to_owned(), inner[end + 1..].to_owned());
    }
    (s.trim_matches('"').to_owned(), String::new())
}

fn validate_ident(name: &str) -> Result<()> {
    if !name.chars().next().is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    {
        bail!("property name {name:?} is not a valid identifier");
    }
    Ok(())
}

/// Extract `name(a, b, ...)` arguments from a v1 pragma tail.
fn parse_call(text: &str, name: &str) -> Result<Option<Vec<f32>>> {
    let Some(start) = text.find(name) else {
        return Ok(None);
    };
    let after = &text[start + name.len()..];
    let Some(open) = after.trim_start().strip_prefix('(') else {
        return Ok(None);
    };
    let close = open.find(')').with_context(|| format!("unclosed {name}("))?;
    open[..close]
        .split(',')
        .map(|v| v.trim().parse::<f32>().with_context(|| format!("bad number {:?} in {name}()", v.trim())))
        .collect::<Result<Vec<f32>>>()
        .map(Some)
}

pub(crate) fn title_case(name: &str) -> String {
    name.split('_')
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut chars = w.chars();
            match chars.next() {
                Some(c) => c.to_uppercase().chain(chars).collect::<String>(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
// @shader "Sample"
// @property tint "Tint" Color = (1, 0.5, 0.25, 1)
// @property glow "Glow" Range(0, 4) = 1
// @property mode "Mode" Enum(Flat, Ramp, Smooth) = Ramp @keyword
// @feature RIM "Rim Lighting"
// @property rim_color "Rim Color" Color3(HDR) = (1, 1, 1)
// @header "Shape"
// @property rim_width Float = 0.5 @show_if(mode != Flat)
// @property rim_mask "Rim Mask" Texture2D = white @linear
// @end
// @foldout "Misc" @show_if(RIM)
// @property flip Toggle = 1
// @property offset2 Vector2 = (0.25, 0.75)
// @property count IntRange(1, 8) = 3
// @property dir Vector3 = (0, 1, 0)
// @property ramp_tex Texture2D = gray
// @preset "Simple" RIM=0 mode=Flat
// @preset "Fancy" RIM=1 mode=Smooth tint=(1,1,1,1)
"#;

    #[test]
    fn parses_v2_properties_and_groups() {
        let s = parse_shader_source(SAMPLE, "x").unwrap();
        assert_eq!(s.display_name, "Sample");
        let names: Vec<_> = s.props.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(
            names,
            ["tint", "glow", "mode", "RIM", "rim_color", "rim_width", "rim_mask", "flip", "offset2", "count", "dir", "ramp_tex"]
        );
        assert_eq!(s.groups.len(), 2);
        assert_eq!(s.groups[0].label, "Rim Lighting");
        assert_eq!(s.groups[0].toggle, Some(3));
        assert_eq!(s.props[4].group, Some(0));
        assert_eq!(s.props[1].group, None);
        assert_eq!(s.props[7].group, Some(1));
        assert_eq!(s.props[5].header.as_deref(), Some("Shape"));
        assert_eq!(s.props[2].default, vec![1.0]);
        assert_eq!(s.props[4].kind, ShaderPropKind::Color3 { hdr: true });
        assert!(matches!(
            s.props[6].kind,
            ShaderPropKind::Texture { default: TextureDefault::White, srgb: false, binding: 1 }
        ));
        assert!(matches!(
            s.props[11].kind,
            ShaderPropKind::Texture { default: TextureDefault::Gray, srgb: false, binding: 2 }
        ));
        assert_eq!(s.props[2].storage, PropStorage::Keyword { id: 0 });
        assert_eq!(s.props[3].storage, PropStorage::Keyword { id: 1 });
        assert_eq!(s.keyword_count(), 2);
        assert_eq!(s.presets.len(), 2);
        assert_eq!(s.presets[0].values, vec![(3, vec![0.0]), (2, vec![0.0])]);
    }

    #[test]
    fn std140_layout() {
        let s = parse_shader_source(SAMPLE, "x").unwrap();
        let off = |n: &str| match s.prop(n).unwrap().storage {
            PropStorage::Uniform { offset } => offset,
            other => panic!("{n}: {other:?}"),
        };
        assert_eq!(off("tint"), 0); // vec4
        assert_eq!(off("glow"), 16); // float
        assert_eq!(off("rim_color"), 32); // vec3 aligns to 16
        assert_eq!(off("rim_width"), 44); // float packs into the vec3's tail
        assert_eq!(off("rim_mask"), 48); // _st vec4
        assert_eq!(off("flip"), 64);
        assert_eq!(off("offset2"), 72); // vec2 aligns to 8
        assert_eq!(off("count"), 80);
        assert_eq!(off("dir"), 96); // vec3 aligns to 16
        assert_eq!(off("ramp_tex"), 112);
        assert_eq!(s.ubo_size, 128);

        let values = s.defaults();
        let bytes = s.uniform_bytes(&values);
        let f = |at: usize| f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let i = |at: usize| i32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        assert_eq!(f(4), 0.5);
        assert_eq!(f(16), 1.0);
        assert_eq!(f(44), 0.5);
        assert_eq!((f(48), f(52), f(56)), (1.0, 1.0, 0.0));
        assert_eq!((f(72), f(76)), (0.25, 0.75));
        assert_eq!(i(80), 3);
        assert_eq!(f(100), 1.0);
    }

    #[test]
    fn conditions_keywords_and_presets() {
        let s = parse_shader_source(SAMPLE, "x").unwrap();
        let mut values = s.defaults();
        let width = s.prop("rim_width").unwrap();
        assert!(width.show_if.eval(&values)); // mode = Ramp
        values[s.prop("mode").unwrap().offset] = 0.0;
        assert!(!width.show_if.eval(&values));
        assert!(!s.groups[1].show_if.eval(&values)); // RIM off
        let kw = s.keywords(&values);
        assert_eq!(&kw[..3], &[0, 0, 0]);
        s.apply_preset(1, &mut values);
        let kw = s.keywords(&values);
        assert_eq!(&kw[..2], &[2, 1]);
        assert!(s.groups[1].show_if.eval(&values));
    }

    #[test]
    fn named_round_trip_and_remap() {
        let s = parse_shader_source(SAMPLE, "x").unwrap();
        let mut values = s.defaults();
        values[s.prop("glow").unwrap().offset] = 3.5;
        let named = s.unpack(&values);
        assert_eq!(named["glow"], vec![3.5]);
        assert_eq!(s.pack(&named), values);
        // A reordered shader keeps values by name.
        let reordered = parse_shader_source(
            "// @property extra Float = 9\n// @property glow Range(0, 4) = 1\n",
            "y",
        )
        .unwrap();
        let moved = reordered.remap_from(&s, &values);
        assert_eq!(moved[reordered.prop("glow").unwrap().offset], 3.5);
        assert_eq!(moved[reordered.prop("extra").unwrap().offset], 9.0);
    }

    #[test]
    fn v1_and_v2_coexist() {
        let s = parse_shader_source(
            "//! prop tint color default(1, 0, 0, 1)\n//! prop speed float range(0, 10) default(2)\n// @property glow Float = 4\n",
            "z",
        )
        .unwrap();
        assert_eq!(s.props[0].storage, PropStorage::Push { offset: 0 });
        assert_eq!(s.props[1].storage, PropStorage::Push { offset: 4 });
        assert_eq!(s.props[2].offset, SHADER_VALUE_BASE);
        let v = s.defaults();
        assert_eq!(s.push_data(&v)[4], 2.0);
        assert_eq!(s.uniform_bytes(&v).len(), 16);
    }

    #[test]
    fn rejects_bad_declarations() {
        assert!(parse_shader_source("// @property 1x Float", "e").is_err());
        assert!(parse_shader_source("// @property a Wat", "e").is_err());
        assert!(parse_shader_source("// @property a Color = (1, 2)", "e").is_err());
        assert!(parse_shader_source("// @property a Float @keyword", "e").is_err());
        assert!(parse_shader_source("// @property a Float @show_if(nope)", "e").is_err());
        assert!(parse_shader_source("// @property a Float\n// @property a Float", "e").is_err());
    }
}
