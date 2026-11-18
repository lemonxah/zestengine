use std::path::Path;

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};

use crate::shader_props::{ShaderLang, ShaderSource, SHADER_VALUE_BASE};

pub const SHADER_GRAPH_EXTENSION: &str = "shadergraph";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphValueType {
    #[default]
    Float,
    Vec2,
    Vec3,
    Vec4,
}

impl GraphValueType {
    pub fn glsl(self) -> &'static str {
        match self {
            Self::Float => "float",
            Self::Vec2 => "vec2",
            Self::Vec3 => "vec3",
            Self::Vec4 => "vec4",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Float => "Float",
            Self::Vec2 => "Vec2",
            Self::Vec3 => "Vec3",
            Self::Vec4 => "Vec4",
        }
    }

    pub fn is_vector(self) -> bool {
        !matches!(self, Self::Float)
    }

    pub fn dims(self) -> usize {
        match self {
            Self::Float => 1,
            Self::Vec2 => 2,
            Self::Vec3 => 3,
            Self::Vec4 => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphTextureSlot {
    Albedo,
    Normal,
    Orm,
    Emission,
}

impl GraphTextureSlot {
    pub const ALL: [GraphTextureSlot; 4] = [Self::Albedo, Self::Normal, Self::Orm, Self::Emission];

    pub fn label(self) -> &'static str {
        match self {
            Self::Albedo => "Albedo",
            Self::Normal => "Normal",
            Self::Orm => "ORM",
            Self::Emission => "Emission",
        }
    }

    pub fn sampler(self) -> &'static str {
        match self {
            Self::Albedo => "t_albedo",
            Self::Normal => "t_normal",
            Self::Orm => "t_orm",
            Self::Emission => "t_emission",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphInput {
    #[default]
    Uv,
    Uv1,
    Time,
    WorldPosition,
    WorldNormal,
    ViewDirection,
    VertexColor,
}

impl GraphInput {
    pub const ALL: [GraphInput; 7] = [Self::Uv, Self::Uv1, Self::Time, Self::WorldPosition, Self::WorldNormal, Self::ViewDirection, Self::VertexColor];

    pub fn label(self) -> &'static str {
        match self {
            Self::Uv => "UV",
            Self::Uv1 => "UV 2",
            Self::Time => "Time",
            Self::WorldPosition => "World Position",
            Self::WorldNormal => "World Normal",
            Self::ViewDirection => "View Direction",
            Self::VertexColor => "Vertex Color",
        }
    }

    pub fn ty(self) -> GraphValueType {
        match self {
            Self::Uv | Self::Uv1 => GraphValueType::Vec2,
            Self::Time => GraphValueType::Float,
            Self::WorldPosition | Self::WorldNormal | Self::ViewDirection => GraphValueType::Vec3,
            Self::VertexColor => GraphValueType::Vec4,
        }
    }

    fn expr(self) -> &'static str {
        match self {
            Self::Uv => "v_uv",
            Self::Uv1 => "v_uv1",
            Self::Time => "u_time",
            Self::WorldPosition => "v_world_pos",
            Self::WorldNormal => "normalize(v_normal)",
            Self::ViewDirection => "normalize(u_camera_pos - v_world_pos)",
            Self::VertexColor => "v_color",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphUnary {
    #[default]
    Abs,
    Fract,
    Floor,
    Saturate,
    Sqrt,
    Negate,
    Length,
}

impl GraphUnary {
    pub const ALL: [GraphUnary; 7] = [Self::Abs, Self::Fract, Self::Floor, Self::Saturate, Self::Sqrt, Self::Negate, Self::Length];

    pub fn label(self) -> &'static str {
        match self {
            Self::Abs => "Absolute",
            Self::Fract => "Fraction",
            Self::Floor => "Floor",
            Self::Saturate => "Saturate",
            Self::Sqrt => "Square Root",
            Self::Negate => "Negate",
            Self::Length => "Length",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphBinary {
    #[default]
    Min,
    Max,
    Pow,
    Step,
    Mod,
    Distance,
    Cross,
}

impl GraphBinary {
    pub const ALL: [GraphBinary; 7] = [Self::Min, Self::Max, Self::Pow, Self::Step, Self::Mod, Self::Distance, Self::Cross];

    pub fn label(self) -> &'static str {
        match self {
            Self::Min => "Minimum",
            Self::Max => "Maximum",
            Self::Pow => "Power",
            Self::Step => "Step",
            Self::Mod => "Modulo",
            Self::Distance => "Distance",
            Self::Cross => "Cross",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum GraphParam {
    Float { default: f32, min: f32, max: f32 },
    Color { default: [f32; 4] },
}

impl GraphParam {
    pub fn ty(&self) -> GraphValueType {
        match self {
            Self::Float { .. } => GraphValueType::Float,
            Self::Color { .. } => GraphValueType::Vec4,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GraphNodeKind {
    Value { value: f32 },
    Color { color: [f32; 4] },
    TextureSample { slot: GraphTextureSlot, uv: Option<usize> },
    Add { a: usize, b: usize },
    Sub { a: usize, b: usize },
    Mul { a: usize, b: usize },
    Div { a: usize, b: usize },
    Mix { a: usize, b: usize, t: usize },
    Sin { x: usize },
    Cos { x: usize },
    Clamp { x: usize, lo: usize, hi: usize },
    OneMinus { x: usize },
    Dot { a: usize, b: usize },
    Normalize { x: usize },
    SwizzleRgb { x: usize },
    Output { color: usize },
    Input { source: GraphInput },
    Vector2 { value: [f32; 2] },
    Vector3 { value: [f32; 3] },
    Unary { op: GraphUnary, x: usize },
    Binary { op: GraphBinary, a: usize, b: usize },
    Smoothstep { lo: usize, hi: usize, x: usize },
    Split { x: usize, channel: u8 },
    Combine { r: usize, g: usize, b: usize, a: Option<usize> },
    Fresnel { power: f32, normal: Option<usize> },
    Panner { uv: Option<usize>, tiling: [f32; 2], speed: [f32; 2] },
    Noise { uv: Option<usize>, scale: f32 },
    NormalMap { uv: Option<usize>, strength: f32 },
    Parameter { name: String, param: GraphParam },
    Surface { albedo: usize, normal: Option<usize>, metallic: Option<usize>, roughness: Option<usize>, emission: Option<usize>, alpha: Option<usize> },
}

enum Slot<'a> {
    Required(&'a mut usize),
    Optional(&'a mut Option<usize>),
}

impl GraphNodeKind {
    pub const PALETTE: &'static [&'static str] = &[
        "Value",
        "Color",
        "Vector 2",
        "Vector 3",
        "Parameter",
        "Color Parameter",
        "Scene Input",
        "Texture Sample",
        "Normal Map",
        "Tiling And Scroll",
        "Noise",
        "Fresnel",
        "Add",
        "Subtract",
        "Multiply",
        "Divide",
        "Mix",
        "Sine",
        "Cosine",
        "Clamp",
        "One Minus",
        "Math",
        "Math 2",
        "Smoothstep",
        "Dot",
        "Normalize",
        "RGB",
        "Split",
        "Combine",
        "Output",
        "Surface Output",
    ];

    pub fn from_title(name: &str, hint: Option<usize>) -> GraphNodeKind {
        let i = hint.unwrap_or(0);
        match name {
            "Value" => Self::Value { value: 0.5 },
            "Color" => Self::Color { color: [0.85, 0.42, 0.2, 1.0] },
            "Vector 2" => Self::Vector2 { value: [1.0, 1.0] },
            "Vector 3" => Self::Vector3 { value: [0.0, 1.0, 0.0] },
            "Parameter" => Self::Parameter { name: "Amount".into(), param: GraphParam::Float { default: 0.5, min: 0.0, max: 1.0 } },
            "Color Parameter" => Self::Parameter { name: "Tint".into(), param: GraphParam::Color { default: [1.0, 1.0, 1.0, 1.0] } },
            "Scene Input" => Self::Input { source: GraphInput::Uv },
            "Texture Sample" => Self::TextureSample { slot: GraphTextureSlot::Albedo, uv: None },
            "Normal Map" => Self::NormalMap { uv: None, strength: 1.0 },
            "Tiling And Scroll" => Self::Panner { uv: None, tiling: [1.0, 1.0], speed: [0.0, 0.0] },
            "Noise" => Self::Noise { uv: None, scale: 8.0 },
            "Fresnel" => Self::Fresnel { power: 5.0, normal: None },
            "Add" => Self::Add { a: i, b: i },
            "Subtract" => Self::Sub { a: i, b: i },
            "Multiply" => Self::Mul { a: i, b: i },
            "Divide" => Self::Div { a: i, b: i },
            "Mix" => Self::Mix { a: i, b: i, t: i },
            "Sine" => Self::Sin { x: i },
            "Cosine" => Self::Cos { x: i },
            "Clamp" => Self::Clamp { x: i, lo: i, hi: i },
            "One Minus" => Self::OneMinus { x: i },
            "Math" => Self::Unary { op: GraphUnary::Abs, x: i },
            "Math 2" => Self::Binary { op: GraphBinary::Min, a: i, b: i },
            "Smoothstep" => Self::Smoothstep { lo: i, hi: i, x: i },
            "Dot" => Self::Dot { a: i, b: i },
            "Normalize" => Self::Normalize { x: i },
            "RGB" => Self::SwizzleRgb { x: i },
            "Split" => Self::Split { x: i, channel: 0 },
            "Combine" => Self::Combine { r: i, g: i, b: i, a: None },
            "Output" => Self::Output { color: i },
            "Surface Output" => Self::Surface { albedo: i, normal: None, metallic: None, roughness: None, emission: None, alpha: None },
            _ => Self::Value { value: 0.0 },
        }
    }

    pub fn title(&self) -> &'static str {
        match self {
            Self::Value { .. } => "Value",
            Self::Color { .. } => "Color",
            Self::TextureSample { .. } => "Texture Sample",
            Self::Add { .. } => "Add",
            Self::Sub { .. } => "Subtract",
            Self::Mul { .. } => "Multiply",
            Self::Div { .. } => "Divide",
            Self::Mix { .. } => "Mix",
            Self::Sin { .. } => "Sine",
            Self::Cos { .. } => "Cosine",
            Self::Clamp { .. } => "Clamp",
            Self::OneMinus { .. } => "One Minus",
            Self::Dot { .. } => "Dot",
            Self::Normalize { .. } => "Normalize",
            Self::SwizzleRgb { .. } => "RGB",
            Self::Output { .. } => "Output",
            Self::Input { source } => source.label(),
            Self::Vector2 { .. } => "Vector 2",
            Self::Vector3 { .. } => "Vector 3",
            Self::Unary { op, .. } => op.label(),
            Self::Binary { op, .. } => op.label(),
            Self::Smoothstep { .. } => "Smoothstep",
            Self::Split { .. } => "Split",
            Self::Combine { .. } => "Combine",
            Self::Fresnel { .. } => "Fresnel",
            Self::Panner { .. } => "Tiling And Scroll",
            Self::Noise { .. } => "Noise",
            Self::NormalMap { .. } => "Normal Map",
            Self::Parameter { param: GraphParam::Float { .. }, .. } => "Parameter",
            Self::Parameter { param: GraphParam::Color { .. }, .. } => "Color Parameter",
            Self::Surface { .. } => "Surface Output",
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Output { .. } | Self::Surface { .. })
    }

    pub fn input_slots(&self) -> &'static [&'static str] {
        match self {
            Self::Value { .. } | Self::Color { .. } | Self::Input { .. } | Self::Vector2 { .. } | Self::Vector3 { .. } | Self::Parameter { .. } => &[],
            Self::TextureSample { .. } | Self::Panner { .. } | Self::Noise { .. } | Self::NormalMap { .. } => &["UV"],
            Self::Add { .. } | Self::Sub { .. } | Self::Mul { .. } | Self::Div { .. } | Self::Dot { .. } | Self::Binary { .. } => &["A", "B"],
            Self::Mix { .. } => &["A", "B", "T"],
            Self::Sin { .. } | Self::Cos { .. } | Self::OneMinus { .. } | Self::Normalize { .. } | Self::SwizzleRgb { .. } | Self::Unary { .. } | Self::Split { .. } => &["X"],
            Self::Clamp { .. } => &["X", "Lo", "Hi"],
            Self::Smoothstep { .. } => &["Lo", "Hi", "X"],
            Self::Combine { .. } => &["R", "G", "B", "A"],
            Self::Fresnel { .. } => &["Normal"],
            Self::Output { .. } => &["Color"],
            Self::Surface { .. } => &["Albedo", "Normal", "Metallic", "Roughness", "Emission", "Alpha"],
        }
    }

    fn slot_mut(&mut self, slot: usize) -> Option<Slot<'_>> {
        use Slot::{Optional, Required};
        Some(match (self, slot) {
            (Self::TextureSample { uv, .. } | Self::Panner { uv, .. } | Self::Noise { uv, .. } | Self::NormalMap { uv, .. }, 0) => Optional(uv),
            (Self::Add { a, .. } | Self::Sub { a, .. } | Self::Mul { a, .. } | Self::Div { a, .. } | Self::Dot { a, .. } | Self::Binary { a, .. } | Self::Mix { a, .. }, 0) => Required(a),
            (Self::Add { b, .. } | Self::Sub { b, .. } | Self::Mul { b, .. } | Self::Div { b, .. } | Self::Dot { b, .. } | Self::Binary { b, .. } | Self::Mix { b, .. }, 1) => Required(b),
            (Self::Mix { t, .. }, 2) => Required(t),
            (Self::Sin { x } | Self::Cos { x } | Self::OneMinus { x } | Self::Normalize { x } | Self::SwizzleRgb { x } | Self::Unary { x, .. } | Self::Split { x, .. } | Self::Clamp { x, .. }, 0) => Required(x),
            (Self::Clamp { lo, .. }, 1) | (Self::Smoothstep { lo, .. }, 0) => Required(lo),
            (Self::Clamp { hi, .. }, 2) | (Self::Smoothstep { hi, .. }, 1) => Required(hi),
            (Self::Smoothstep { x, .. }, 2) => Required(x),
            (Self::Combine { r, .. }, 0) => Required(r),
            (Self::Combine { g, .. }, 1) => Required(g),
            (Self::Combine { b, .. }, 2) => Required(b),
            (Self::Combine { a, .. }, 3) => Optional(a),
            (Self::Fresnel { normal, .. }, 0) => Optional(normal),
            (Self::Output { color }, 0) => Required(color),
            (Self::Surface { albedo, .. }, 0) => Required(albedo),
            (Self::Surface { normal, .. }, 1) => Optional(normal),
            (Self::Surface { metallic, .. }, 2) => Optional(metallic),
            (Self::Surface { roughness, .. }, 3) => Optional(roughness),
            (Self::Surface { emission, .. }, 4) => Optional(emission),
            (Self::Surface { alpha, .. }, 5) => Optional(alpha),
            _ => return None,
        })
    }

    pub fn input(&self, slot: usize) -> Option<usize> {
        match self.clone().slot_mut(slot)? {
            Slot::Required(v) => Some(*v),
            Slot::Optional(v) => *v,
        }
    }

    pub fn is_optional(&self, slot: usize) -> bool {
        matches!(self.clone().slot_mut(slot), Some(Slot::Optional(_)))
    }

    pub fn default_label(&self, slot: usize) -> Option<&'static str> {
        if !self.is_optional(slot) {
            return None;
        }
        Some(match (self, slot) {
            (Self::TextureSample { .. } | Self::Panner { .. } | Self::Noise { .. } | Self::NormalMap { .. }, _) => "v_uv",
            (Self::Fresnel { .. }, _) | (Self::Surface { .. }, 1) => "mesh",
            (Self::Combine { .. }, _) => "none",
            (Self::Surface { .. }, 2) => "0",
            (Self::Surface { .. }, 3) => "0.5",
            (Self::Surface { .. }, 4) => "none",
            (Self::Surface { .. }, _) => "auto",
            _ => "auto",
        })
    }

    pub fn with_input(&self, slot: usize, value: usize) -> Option<Self> {
        let mut k = self.clone();
        match k.slot_mut(slot)? {
            Slot::Required(v) => *v = value,
            Slot::Optional(v) => *v = Some(value),
        }
        Some(k)
    }

    pub fn disconnect(&self, slot: usize) -> Option<Self> {
        let mut k = self.clone();
        match k.slot_mut(slot)? {
            Slot::Optional(v) if v.is_some() => *v = None,
            _ => return None,
        }
        Some(k)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    pub position: [f32; 2],
    pub kind: GraphNodeKind,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ShaderGraph {
    pub nodes: Vec<GraphNode>,
}

pub fn default_graph() -> ShaderGraph {
    ShaderGraph {
        nodes: vec![
            GraphNode { position: [-220.0, -20.0], kind: GraphNodeKind::Color { color: [0.85, 0.42, 0.2, 1.0] } },
            GraphNode { position: [40.0, -20.0], kind: GraphNodeKind::Output { color: 0 } },
        ],
    }
}

pub fn topo_order(graph: &ShaderGraph) -> Result<Vec<usize>, String> {
    let n = graph.nodes.len();
    let mut indegree = vec![0usize; n];
    let mut adj = vec![Vec::new(); n];
    for (id, node) in graph.nodes.iter().enumerate() {
        for slot in 0..node.kind.input_slots().len() {
            if let Some(dep) = node.kind.input(slot) {
                if dep >= n {
                    return Err(format!("node {id} references missing node {dep}"));
                }
                indegree[id] += 1;
                adj[dep].push(id);
            }
        }
    }
    let mut queue: Vec<usize> = (0..n).filter(|&i| indegree[i] == 0).collect();
    let mut out = Vec::with_capacity(n);
    while let Some(id) = queue.pop() {
        out.push(id);
        for &next in &adj[id] {
            indegree[next] -= 1;
            if indegree[next] == 0 {
                queue.push(next);
            }
        }
    }
    if out.len() != n {
        return Err("graph contains a cycle".into());
    }
    Ok(out)
}

pub fn infer_types(graph: &ShaderGraph, order: &[usize]) -> Result<Vec<GraphValueType>, String> {
    use GraphValueType::{Float, Vec2, Vec3, Vec4};
    let mut ty = vec![Float; graph.nodes.len()];
    for &id in order {
        let node = &graph.nodes[id];
        let t = |i: usize| ty.get(i).copied().ok_or_else(|| format!("node {id} references missing node {i}"));
        let optional = |input: &Option<usize>, want: &[GraphValueType], what: &str| -> Result<(), String> {
            let Some(i) = input else { return Ok(()) };
            let got = t(*i)?;
            if want.contains(&got) {
                Ok(())
            } else {
                Err(format!("node {id}: {what} needs {}, got {}", want.iter().map(|w| w.label()).collect::<Vec<_>>().join(" or "), got.label()))
            }
        };
        let scalar_or = |i: usize, like: GraphValueType, what: &str| -> Result<(), String> {
            let got = t(i)?;
            if got == Float || got == like { Ok(()) } else { Err(format!("node {id}: {what} has type {} but the value is {}", got.label(), like.label())) }
        };
        ty[id] = match &node.kind {
            GraphNodeKind::Value { .. } => Float,
            GraphNodeKind::Color { .. } => Vec4,
            GraphNodeKind::TextureSample { uv, .. } => {
                optional(uv, &[Vec2], "UV")?;
                Vec4
            }
            GraphNodeKind::Input { source } => source.ty(),
            GraphNodeKind::Vector2 { .. } => Vec2,
            GraphNodeKind::Vector3 { .. } => Vec3,
            GraphNodeKind::Parameter { param, .. } => param.ty(),
            GraphNodeKind::Add { a, b } | GraphNodeKind::Sub { a, b } | GraphNodeKind::Mul { a, b } | GraphNodeKind::Div { a, b } => {
                common(t(*a)?, t(*b)?).ok_or_else(|| format!("node {id}: {a} and {b} have incompatible types"))?
            }
            GraphNodeKind::Mix { a, b, t: f } => {
                let ab = common(t(*a)?, t(*b)?).ok_or_else(|| format!("node {id}: {a} and {b} have incompatible types"))?;
                let tf = t(*f)?;
                if tf != Float && tf != ab {
                    return Err(format!("node {id}: mix factor has type {} but inputs are {}", tf.label(), ab.label()));
                }
                ab
            }
            GraphNodeKind::Sin { x } | GraphNodeKind::Cos { x } => {
                let xt = t(*x)?;
                if xt != Float {
                    return Err(format!("node {id}: sine/cosine needs a Float input, got {}", xt.label()));
                }
                Float
            }
            GraphNodeKind::Clamp { x, lo, hi } => {
                let xt = t(*x)?;
                scalar_or(*lo, xt, "clamp lo")?;
                scalar_or(*hi, xt, "clamp hi")?;
                xt
            }
            GraphNodeKind::Smoothstep { lo, hi, x } => {
                let xt = t(*x)?;
                scalar_or(*lo, xt, "smoothstep lo")?;
                scalar_or(*hi, xt, "smoothstep hi")?;
                xt
            }
            GraphNodeKind::OneMinus { x } => t(*x)?,
            GraphNodeKind::Unary { op, x } => {
                let xt = t(*x)?;
                match op {
                    GraphUnary::Length if !xt.is_vector() => return Err(format!("node {id}: length needs a vector input, got {}", xt.label())),
                    GraphUnary::Length => Float,
                    _ => xt,
                }
            }
            GraphNodeKind::Binary { op, a, b } => {
                let (at, bt) = (t(*a)?, t(*b)?);
                match op {
                    GraphBinary::Distance => {
                        if !at.is_vector() || at != bt {
                            return Err(format!("node {id}: distance needs two vectors of the same type"));
                        }
                        Float
                    }
                    GraphBinary::Cross => {
                        if at != Vec3 || bt != Vec3 {
                            return Err(format!("node {id}: cross needs two Vec3 inputs"));
                        }
                        Vec3
                    }
                    _ => common(at, bt).ok_or_else(|| format!("node {id}: {a} and {b} have incompatible types"))?,
                }
            }
            GraphNodeKind::Dot { a, b } => {
                let (at, bt) = (t(*a)?, t(*b)?);
                if !at.is_vector() || at != bt {
                    return Err(format!("node {id}: dot needs two vectors of the same type"));
                }
                Float
            }
            GraphNodeKind::Normalize { x } => {
                let xt = t(*x)?;
                if !xt.is_vector() {
                    return Err(format!("node {id}: normalize needs a vector input, got {}", xt.label()));
                }
                xt
            }
            GraphNodeKind::SwizzleRgb { x } => {
                let xt = t(*x)?;
                if !matches!(xt, Vec3 | Vec4) {
                    return Err(format!("node {id}: rgb needs a vec3/vec4 input, got {}", xt.label()));
                }
                Vec3
            }
            GraphNodeKind::Split { x, channel } => {
                let xt = t(*x)?;
                if !xt.is_vector() || *channel as usize >= xt.dims() {
                    return Err(format!("node {id}: a {} has no channel {}", xt.label(), ["R", "G", "B", "A"].get(*channel as usize).copied().unwrap_or("?")));
                }
                Float
            }
            GraphNodeKind::Combine { r, g, b, a } => {
                for (what, i) in [("R", r), ("G", g), ("B", b)] {
                    if t(*i)? != Float {
                        return Err(format!("node {id}: combine {what} needs a Float"));
                    }
                }
                optional(a, &[Float], "combine A")?;
                if a.is_some() { Vec4 } else { Vec3 }
            }
            GraphNodeKind::Fresnel { normal, .. } => {
                optional(normal, &[Vec3], "fresnel normal")?;
                Float
            }
            GraphNodeKind::Panner { uv, .. } => {
                optional(uv, &[Vec2], "UV")?;
                Vec2
            }
            GraphNodeKind::Noise { uv, .. } => {
                optional(uv, &[Vec2], "UV")?;
                Float
            }
            GraphNodeKind::NormalMap { uv, .. } => {
                optional(uv, &[Vec2], "UV")?;
                Vec3
            }
            GraphNodeKind::Output { color } => {
                let ct = t(*color)?;
                if !matches!(ct, Vec3 | Vec4) {
                    return Err(format!("node {id}: output needs a vec3/vec4 colour, got {}", ct.label()));
                }
                Vec4
            }
            GraphNodeKind::Surface { albedo, normal, metallic, roughness, emission, alpha } => {
                let at = t(*albedo)?;
                if !matches!(at, Vec3 | Vec4) {
                    return Err(format!("node {id}: albedo needs a vec3/vec4 colour, got {}", at.label()));
                }
                optional(normal, &[Vec3], "normal")?;
                optional(metallic, &[Float], "metallic")?;
                optional(roughness, &[Float], "roughness")?;
                optional(emission, &[Vec3, Vec4], "emission")?;
                optional(alpha, &[Float], "alpha")?;
                Vec4
            }
        };
    }
    Ok(ty)
}

pub fn validate(graph: &ShaderGraph) -> Result<Vec<GraphValueType>, String> {
    let order = topo_order(graph)?;
    infer_types(graph, &order)
}

fn common(a: GraphValueType, b: GraphValueType) -> Option<GraphValueType> {
    match (a, b) {
        (x, y) if x == y => Some(x),
        (GraphValueType::Float, v) | (v, GraphValueType::Float) => Some(v),
        _ => None,
    }
}

fn lit(x: f32) -> String {
    if x.is_finite() { format!("{x:?}") } else { "0.0".into() }
}

pub fn parameter_ident(name: &str) -> String {
    let mut ident: String = name.trim().chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' }).collect();
    while ident.contains("__") {
        ident = ident.replace("__", "_");
    }
    let ident = ident.trim_matches('_');
    if ident.is_empty() { "p_param".into() } else { format!("p_{ident}") }
}

fn parameter_directives(graph: &ShaderGraph) -> Result<String, String> {
    let mut seen: Vec<(String, GraphValueType)> = Vec::new();
    let mut out = String::new();
    for (id, node) in graph.nodes.iter().enumerate() {
        let GraphNodeKind::Parameter { name, param } = &node.kind else { continue };
        let ident = parameter_ident(name);
        if let Some((_, ty)) = seen.iter().find(|(i, _)| *i == ident) {
            if *ty != param.ty() {
                return Err(format!("node {id}: parameter \"{name}\" is declared as both a number and a colour"));
            }
            continue;
        }
        seen.push((ident.clone(), param.ty()));
        let label = name.replace('"', "");
        match param {
            GraphParam::Float { default, min, max } if max > min => {
                out.push_str(&format!("// @property {ident} \"{label}\" Range({}, {}) = {}\n", lit(*min), lit(*max), lit(default.clamp(*min, *max))));
            }
            GraphParam::Float { default, .. } => out.push_str(&format!("// @property {ident} \"{label}\" Float = {}\n", lit(*default))),
            GraphParam::Color { default } => out.push_str(&format!(
                "// @property {ident} \"{label}\" Color = ({}, {}, {}, {})\n",
                lit(default[0]),
                lit(default[1]),
                lit(default[2]),
                lit(default[3])
            )),
        }
    }
    Ok(out)
}

const NOISE_HELPERS: &str = "float zg_hash(vec2 p) {\n    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);\n}\nfloat zg_noise(vec2 p) {\n    vec2 i = floor(p);\n    vec2 f = fract(p);\n    f = f * f * (3.0 - 2.0 * f);\n    return mix(mix(zg_hash(i), zg_hash(i + vec2(1.0, 0.0)), f.x), mix(zg_hash(i + vec2(0.0, 1.0)), zg_hash(i + vec2(1.0, 1.0)), f.x), f.y);\n}\n";

pub fn generate_glsl(graph: &ShaderGraph) -> Result<String, String> {
    let order = topo_order(graph)?;
    let types = infer_types(graph, &order)?;
    let terminals: Vec<usize> = graph.nodes.iter().enumerate().filter(|(_, n)| n.kind.is_terminal()).map(|(i, _)| i).collect();
    let output = match terminals.as_slice() {
        [] => return Err("graph has no Output node".to_owned()),
        [one] => *one,
        _ => return Err("graph has more than one Output node".to_owned()),
    };

    let mut text = parameter_directives(graph)?;
    if graph.nodes.iter().any(|n| matches!(n.kind, GraphNodeKind::Noise { .. })) {
        text.push_str(NOISE_HELPERS);
    }
    text.push_str("void main() {\n");
    for &id in &order {
        if id == output {
            continue;
        }
        text.push_str("    ");
        text.push_str(&node_expr(graph, id, &types)?);
        text.push('\n');
    }
    let rgb = |i: usize| if types[i] == GraphValueType::Vec4 { format!("v{i}.rgb") } else { format!("v{i}") };
    match &graph.nodes[output].kind {
        GraphNodeKind::Output { color } => {
            let color = match types[*color] {
                GraphValueType::Vec3 => format!("vec4(v{color}, 1.0)"),
                _ => format!("v{color}"),
            };
            text.push_str(&format!("    o_color = vec4(zestengine_postfx({color}.rgb, gl_FragCoord.xy), {color}.a);\n"));
        }
        GraphNodeKind::Surface { albedo, normal, metallic, roughness, emission, alpha } => {
            let normal = normal.map_or_else(|| "v_normal".to_owned(), |n| format!("v{n}"));
            let metallic = metallic.map_or_else(|| "0.0".to_owned(), |m| format!("v{m}"));
            let roughness = roughness.map_or_else(|| "0.5".to_owned(), |r| format!("v{r}"));
            let emission = emission.map_or_else(|| "vec3(0.0)".to_owned(), rgb);
            let alpha = match alpha {
                Some(a) => format!("v{a}"),
                None if types[*albedo] == GraphValueType::Vec4 => format!("v{albedo}.a"),
                None => "1.0".to_owned(),
            };
            for line in [
                format!("vec3 zg_albedo = {};", rgb(*albedo)),
                format!("vec3 zg_n = normalize({normal});"),
                format!("float zg_metallic = clamp({metallic}, 0.0, 1.0);"),
                format!("float zg_rough = clamp({roughness}, 0.04, 1.0);"),
                "vec3 zg_v = normalize(u_camera_pos - v_world_pos);".to_owned(),
                "vec3 zg_diffuse = zg_albedo * (1.0 - zg_metallic) * (zestengine_direct_lambert(v_world_pos, zg_n) + zestengine_baked_gi(v_world_pos, zg_n));".to_owned(),
                "vec3 zg_f0 = mix(vec3(0.04), zg_albedo, zg_metallic);".to_owned(),
                "float zg_fres = pow(1.0 - max(dot(zg_n, zg_v), 0.0), 5.0);".to_owned(),
                "vec3 zg_spec = zestengine_env(v_world_pos, reflect(-zg_v, zg_n), zg_rough) * mix(zg_f0, vec3(1.0), zg_fres * (1.0 - zg_rough));".to_owned(),
                format!("vec3 zg_lit = zg_diffuse + zg_spec + {emission};"),
                format!("o_color = vec4(zestengine_postfx(zg_lit, gl_FragCoord.xy), {alpha});"),
            ] {
                text.push_str("    ");
                text.push_str(&line);
                text.push('\n');
            }
        }
        _ => unreachable!(),
    }
    text.push_str("}\n");
    Ok(text)
}

fn node_expr(graph: &ShaderGraph, id: usize, types: &[GraphValueType]) -> Result<String, String> {
    let node = &graph.nodes[id];
    let ty = types[id];
    let promoted = |i: usize| if types[i] == ty { format!("v{i}") } else { format!("{}(v{i})", ty.glsl()) };
    let uv_of = |uv: &Option<usize>| uv.map_or_else(|| "v_uv".to_owned(), |u| format!("v{u}"));
    let g = ty.glsl();
    let line = match &node.kind {
        GraphNodeKind::Value { value } => format!("float v{id} = {};", lit(*value)),
        GraphNodeKind::Color { color } => format!("vec4 v{id} = vec4({}, {}, {}, {});", lit(color[0]), lit(color[1]), lit(color[2]), lit(color[3])),
        GraphNodeKind::Vector2 { value } => format!("vec2 v{id} = vec2({}, {});", lit(value[0]), lit(value[1])),
        GraphNodeKind::Vector3 { value } => format!("vec3 v{id} = vec3({}, {}, {});", lit(value[0]), lit(value[1]), lit(value[2])),
        GraphNodeKind::Input { source } => format!("{g} v{id} = {};", source.expr()),
        GraphNodeKind::Parameter { name, .. } => format!("{g} v{id} = {};", parameter_ident(name)),
        GraphNodeKind::TextureSample { slot, uv } => format!("vec4 v{id} = texture({}, {});", slot.sampler(), uv_of(uv)),
        GraphNodeKind::Add { a, b } => format!("{g} v{id} = {} + {};", promoted(*a), promoted(*b)),
        GraphNodeKind::Sub { a, b } => format!("{g} v{id} = {} - {};", promoted(*a), promoted(*b)),
        GraphNodeKind::Mul { a, b } => format!("{g} v{id} = {} * {};", promoted(*a), promoted(*b)),
        GraphNodeKind::Div { a, b } => format!("{g} v{id} = {} / {};", promoted(*a), promoted(*b)),
        GraphNodeKind::Mix { a, b, t } => format!("{g} v{id} = mix({}, {}, {});", promoted(*a), promoted(*b), promoted(*t)),
        GraphNodeKind::Sin { x } => format!("float v{id} = sin(v{x});"),
        GraphNodeKind::Cos { x } => format!("float v{id} = cos(v{x});"),
        GraphNodeKind::Clamp { x, lo, hi } => format!("{g} v{id} = clamp(v{x}, {}, {});", promoted(*lo), promoted(*hi)),
        GraphNodeKind::Smoothstep { lo, hi, x } => format!("{g} v{id} = smoothstep({}, {}, v{x});", promoted(*lo), promoted(*hi)),
        GraphNodeKind::OneMinus { x } => format!("{g} v{id} = {g}(1.0) - v{x};"),
        GraphNodeKind::Unary { op, x } => match op {
            GraphUnary::Abs => format!("{g} v{id} = abs(v{x});"),
            GraphUnary::Fract => format!("{g} v{id} = fract(v{x});"),
            GraphUnary::Floor => format!("{g} v{id} = floor(v{x});"),
            GraphUnary::Saturate => format!("{g} v{id} = clamp(v{x}, 0.0, 1.0);"),
            GraphUnary::Sqrt => format!("{g} v{id} = sqrt(max(v{x}, {g}(0.0)));"),
            GraphUnary::Negate => format!("{g} v{id} = -v{x};"),
            GraphUnary::Length => format!("float v{id} = length(v{x});"),
        },
        GraphNodeKind::Binary { op, a, b } => match op {
            GraphBinary::Min => format!("{g} v{id} = min({}, {});", promoted(*a), promoted(*b)),
            GraphBinary::Max => format!("{g} v{id} = max({}, {});", promoted(*a), promoted(*b)),
            GraphBinary::Pow => format!("{g} v{id} = pow(max({}, {g}(0.0)), {});", promoted(*a), promoted(*b)),
            GraphBinary::Step => format!("{g} v{id} = step({}, {});", promoted(*a), promoted(*b)),
            GraphBinary::Mod => format!("{g} v{id} = mod({}, {});", promoted(*a), promoted(*b)),
            GraphBinary::Distance => format!("float v{id} = distance(v{a}, v{b});"),
            GraphBinary::Cross => format!("vec3 v{id} = cross(v{a}, v{b});"),
        },
        GraphNodeKind::Dot { a, b } => format!("float v{id} = dot(v{a}, v{b});"),
        GraphNodeKind::Normalize { x } => format!("{g} v{id} = normalize(v{x});"),
        GraphNodeKind::SwizzleRgb { x } => format!("vec3 v{id} = v{x}.rgb;"),
        GraphNodeKind::Split { x, channel } => format!("float v{id} = v{x}.{};", ["r", "g", "b", "a"][(*channel as usize).min(3)]),
        GraphNodeKind::Combine { r, g: green, b, a } => match a {
            Some(a) => format!("vec4 v{id} = vec4(v{r}, v{green}, v{b}, v{a});"),
            None => format!("vec3 v{id} = vec3(v{r}, v{green}, v{b});"),
        },
        GraphNodeKind::Fresnel { power, normal } => {
            let n = normal.map_or_else(|| "v_normal".to_owned(), |n| format!("v{n}"));
            format!("float v{id} = pow(1.0 - clamp(dot(normalize({n}), normalize(u_camera_pos - v_world_pos)), 0.0, 1.0), {});", lit(power.max(0.0)))
        }
        GraphNodeKind::Panner { uv, tiling, speed } => format!(
            "vec2 v{id} = {} * vec2({}, {}) + vec2({}, {}) * u_time;",
            uv_of(uv),
            lit(tiling[0]),
            lit(tiling[1]),
            lit(speed[0]),
            lit(speed[1])
        ),
        GraphNodeKind::Noise { uv, scale } => format!("float v{id} = zg_noise({} * {});", uv_of(uv), lit(*scale)),
        GraphNodeKind::NormalMap { uv, strength } => format!(
            "vec3 v{id}; {{ vec3 zg_t = texture(t_normal, {}).xyz * 2.0 - 1.0; zg_t.xy *= {}; vec3 zg_ng = normalize(v_normal); vec3 zg_tg = normalize(v_tangent.xyz - zg_ng * dot(v_tangent.xyz, zg_ng)); v{id} = normalize(zg_tg * zg_t.x + cross(zg_ng, zg_tg) * v_tangent.w * zg_t.y + zg_ng * zg_t.z); }}",
            uv_of(uv),
            lit(*strength)
        ),
        GraphNodeKind::Output { .. } | GraphNodeKind::Surface { .. } => return Err("Output nodes are emitted separately".to_owned()),
    };
    Ok(line)
}

pub fn graph_shader_source(graph: &ShaderGraph, fallback_name: &str) -> Result<ShaderSource, String> {
    let body = generate_glsl(graph)?;
    if body.starts_with("// @property") {
        return crate::shader_props::parse_shader_source(&body, fallback_name).map_err(|e| format!("{e:#}"));
    }
    Ok(ShaderSource {
        display_name: fallback_name.to_owned(),
        lang: ShaderLang::Glsl,
        props: Vec::new(),
        groups: Vec::new(),
        presets: Vec::new(),
        body,
        vertex: None,
        ubo_size: 0,
        value_len: SHADER_VALUE_BASE,
    })
}

pub fn is_shader_graph_path(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case(SHADER_GRAPH_EXTENSION))
}

pub fn load_graph(path: impl AsRef<Path>) -> Result<ShaderGraph> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    ron::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

pub fn load_shader_graph(path: impl AsRef<Path>) -> Result<ShaderSource> {
    let path = path.as_ref();
    let graph = load_graph(path)?;
    let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "shader graph".into());
    graph_shader_source(&graph, &name).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
}

pub fn save_shader_graph(path: impl AsRef<Path>, graph: &ShaderGraph) -> Result<()> {
    let path = path.as_ref();
    let text = ron::ser::to_string_pretty(graph, ron::ser::PrettyConfig::default())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph(kinds: Vec<GraphNodeKind>) -> ShaderGraph {
        ShaderGraph {
            nodes: kinds.into_iter().map(|kind| GraphNode { position: [0.0, 0.0], kind }).collect(),
        }
    }

    fn everything_graph() -> ShaderGraph {
        use GraphNodeKind as K;
        graph(vec![
            K::Input { source: GraphInput::Uv },
            K::Panner { uv: Some(0), tiling: [2.0, 2.0], speed: [0.1, 0.0] },
            K::TextureSample { slot: GraphTextureSlot::Albedo, uv: Some(1) },
            K::Parameter { name: "Tint".into(), param: GraphParam::Color { default: [1.0, 0.9, 0.8, 1.0] } },
            K::Mul { a: 2, b: 3 },
            K::Noise { uv: Some(1), scale: 6.0 },
            K::Parameter { name: "Wear Amount".into(), param: GraphParam::Float { default: 0.5, min: 0.0, max: 4.0 } },
            K::Binary { op: GraphBinary::Pow, a: 5, b: 6 },
            K::Unary { op: GraphUnary::Saturate, x: 7 },
            K::Fresnel { power: 3.0, normal: None },
            K::Color { color: [0.2, 0.6, 1.0, 1.0] },
            K::Mul { a: 10, b: 9 },
            K::NormalMap { uv: Some(1), strength: 1.0 },
            K::Split { x: 2, channel: 3 },
            K::Value { value: 0.2 },
            K::Smoothstep { lo: 14, hi: 6, x: 5 },
            K::Input { source: GraphInput::WorldPosition },
            K::Unary { op: GraphUnary::Length, x: 16 },
            K::Binary { op: GraphBinary::Min, a: 17, b: 14 },
            K::Combine { r: 8, g: 15, b: 18, a: None },
            K::Vector3 { value: [0.0, 1.0, 0.0] },
            K::Binary { op: GraphBinary::Cross, a: 19, b: 20 },
            K::Binary { op: GraphBinary::Distance, a: 19, b: 20 },
            K::Input { source: GraphInput::Time },
            K::Unary { op: GraphUnary::Fract, x: 23 },
            K::Binary { op: GraphBinary::Step, a: 24, b: 5 },
            K::Binary { op: GraphBinary::Mod, a: 23, b: 14 },
            K::Binary { op: GraphBinary::Max, a: 25, b: 26 },
            K::Vector2 { value: [3.0, -1.5] },
            K::Unary { op: GraphUnary::Abs, x: 28 },
            K::Unary { op: GraphUnary::Negate, x: 29 },
            K::Unary { op: GraphUnary::Floor, x: 30 },
            K::Unary { op: GraphUnary::Sqrt, x: 31 },
            K::Unary { op: GraphUnary::Length, x: 32 },
            K::Add { a: 27, b: 33 },
            K::Add { a: 34, b: 22 },
            K::Input { source: GraphInput::VertexColor },
            K::Input { source: GraphInput::ViewDirection },
            K::Input { source: GraphInput::WorldNormal },
            K::Dot { a: 37, b: 38 },
            K::Input { source: GraphInput::Uv1 },
            K::Unary { op: GraphUnary::Length, x: 40 },
            K::Add { a: 39, b: 41 },
            K::Add { a: 42, b: 35 },
            K::Mul { a: 21, b: 43 },
            K::Add { a: 11, b: 36 },
            K::Parameter { name: "wear amount".into(), param: GraphParam::Float { default: 0.5, min: 0.0, max: 4.0 } },
            K::Surface { albedo: 4, normal: Some(12), metallic: Some(8), roughness: Some(15), emission: Some(45), alpha: Some(13) },
        ])
    }

    #[test]
    fn a_graph_using_every_node_compiles_to_a_lit_shader_with_material_properties() {
        let g = everything_graph();
        let types = validate(&g).unwrap();
        assert_eq!(types[1], GraphValueType::Vec2);
        assert_eq!(types[19], GraphValueType::Vec3);
        assert_eq!(types[44], GraphValueType::Vec3);
        assert_eq!(types[45], GraphValueType::Vec4);
        let text = generate_glsl(&g).unwrap();
        assert_eq!(text.matches("// @property").count(), 2, "two parameters with one name share a property:\n{text}");
        assert!(text.contains("// @property p_wear_amount \"Wear Amount\" Range(0.0, 4.0) = 0.5"));
        assert!(text.contains("// @property p_tint \"Tint\" Color = (1.0, 0.9, 0.8, 1.0)"));
        assert!(text.contains("float zg_noise(vec2 p)"));
        assert!(text.contains("zestengine_direct_lambert") && text.contains("zestengine_env"));
        let source = graph_shader_source(&g, "everything").unwrap();
        assert_eq!(source.display_name, "everything");
        assert_eq!(source.props.len(), 2, "parameters become inspector properties");
        let spirv = crate::shader_file::compile_shader(&source, "everything.shadergraph").expect("the generated shader compiles");
        assert!(spirv.len() > 1000);
    }

    #[test]
    fn an_unlit_graph_with_new_nodes_compiles_and_old_files_still_load() {
        use GraphNodeKind as K;
        let g = graph(vec![
            K::Input { source: GraphInput::Uv },
            K::Split { x: 0, channel: 0 },
            K::Split { x: 0, channel: 1 },
            K::Noise { uv: None, scale: 12.0 },
            K::Value { value: 1.0 },
            K::Combine { r: 1, g: 2, b: 3, a: Some(4) },
            K::Output { color: 5 },
        ]);
        let source = graph_shader_source(&g, "unlit").unwrap();
        assert!(source.props.is_empty());
        assert!(source.body.starts_with("float zg_hash"));
        crate::shader_file::compile_shader(&source, "unlit.shadergraph").expect("compiles");

        let old = "(nodes: [(position: (0.0, 0.0), kind: Color(color: (1.0, 0.0, 0.0, 1.0))), (position: (1.0, 0.0), kind: TextureSample(slot: Albedo, uv: None)), (position: (2.0, 0.0), kind: Mul(a: 0, b: 1)), (position: (3.0, 0.0), kind: Output(color: 2))])";
        let loaded: ShaderGraph = ron::from_str(old).unwrap();
        assert!(generate_glsl(&loaded).unwrap().contains("vec4 v2 = v0 * v1;"));
        for g in [everything_graph(), g] {
            let text = ron::ser::to_string_pretty(&g, Default::default()).unwrap();
            assert_eq!(ron::from_str::<ShaderGraph>(&text).unwrap(), g);
        }
    }

    #[test]
    fn type_errors_and_bad_structures_are_reported() {
        use GraphNodeKind as K;
        let err = |kinds: Vec<K>| generate_glsl(&graph(kinds)).unwrap_err();
        assert!(err(vec![K::Input { source: GraphInput::Uv }, K::Split { x: 0, channel: 2 }, K::Color { color: [1.0; 4] }, K::Output { color: 2 }]).contains("no channel B"));
        assert!(err(vec![K::Input { source: GraphInput::Uv }, K::Binary { op: GraphBinary::Cross, a: 0, b: 0 }, K::Color { color: [1.0; 4] }, K::Output { color: 2 }]).contains("cross needs two Vec3"));
        assert!(err(vec![K::Value { value: 1.0 }, K::Unary { op: GraphUnary::Length, x: 0 }, K::Color { color: [1.0; 4] }, K::Output { color: 2 }]).contains("length needs a vector"));
        assert!(err(vec![K::Value { value: 1.0 }, K::Noise { uv: Some(0), scale: 1.0 }, K::Color { color: [1.0; 4] }, K::Output { color: 2 }]).contains("UV needs Vec2"));
        assert!(err(vec![K::Value { value: 1.0 }, K::Surface { albedo: 0, normal: None, metallic: None, roughness: None, emission: None, alpha: None }]).contains("albedo needs"));
        assert!(err(vec![K::Color { color: [1.0; 4] }, K::Output { color: 0 }, K::Surface { albedo: 0, normal: None, metallic: None, roughness: None, emission: None, alpha: None }]).contains("more than one Output"));
        assert!(err(vec![K::Color { color: [1.0; 4] }]).contains("no Output"));
        assert!(err(vec![
            K::Parameter { name: "Mix".into(), param: GraphParam::Float { default: 0.0, min: 0.0, max: 1.0 } },
            K::Parameter { name: "mix".into(), param: GraphParam::Color { default: [1.0; 4] } },
            K::Output { color: 1 },
        ])
        .contains("both a number and a colour"));
    }

    #[test]
    fn slots_connect_disconnect_and_every_palette_entry_makes_its_node() {
        use GraphNodeKind as K;
        for name in K::PALETTE {
            let kind = K::from_title(name, Some(3));
            assert!(*name == "Value" || !matches!(kind, K::Value { value } if value == 0.0), "{name} is in the palette but not built");
            for slot in 0..kind.input_slots().len() {
                let wired = kind.with_input(slot, 9).unwrap_or_else(|| panic!("{name} slot {slot} accepts a wire"));
                assert_eq!(wired.input(slot), Some(9));
                assert_eq!(wired.disconnect(slot).is_some(), kind.is_optional(slot), "{name} slot {slot}");
                assert_eq!(kind.default_label(slot).is_some(), kind.is_optional(slot));
            }
            assert!(kind.with_input(kind.input_slots().len(), 1).is_none());
        }
        let surface = K::from_title("Surface Output", Some(0));
        assert!(surface.is_terminal() && K::from_title("Output", None).is_terminal() && !K::from_title("Mix", None).is_terminal());
        assert_eq!(surface.input(1), None);
        assert_eq!(surface.with_input(1, 4).unwrap().disconnect(1).unwrap(), surface);
        assert_eq!(parameter_ident("  Wear -- Amount! "), "p_wear_amount");
        assert_eq!(parameter_ident("??"), "p_param");
    }

    #[test]
    fn topo_order_puts_dependencies_first() {
        let g = graph(vec![
            GraphNodeKind::Value { value: 0.5 },
            GraphNodeKind::Value { value: 2.0 },
            GraphNodeKind::Add { a: 0, b: 1 },
            GraphNodeKind::Mul { a: 2, b: 2 },
            GraphNodeKind::Output { color: 3 },
        ]);
        let order = topo_order(&g).unwrap();
        let pos = |id: usize| order.iter().position(|&i| i == id).unwrap();
        assert!(pos(0) < pos(2) && pos(1) < pos(2));
        assert!(pos(2) < pos(3));
        assert!(pos(3) < pos(4));
    }

    #[test]
    fn topo_order_detects_cycles() {
        let g = graph(vec![GraphNodeKind::Add { a: 1, b: 1 }, GraphNodeKind::Add { a: 0, b: 0 }]);
        assert!(topo_order(&g).is_err());
    }

    #[test]
    fn topo_order_rejects_missing_nodes() {
        let g = graph(vec![GraphNodeKind::Add { a: 0, b: 7 }]);
        assert!(topo_order(&g).is_err());
    }

    #[test]
    fn default_graph_generates_a_surface_body() {
        let body = generate_glsl(&default_graph()).unwrap();
        assert!(body.contains("void main()"));
        assert!(body.contains("zestengine_postfx"));
        assert!(body.contains("o_color"));
        assert!(body.contains("vec4 v0"));
    }

    #[test]
    fn save_load_round_trips() {
        let dir = std::env::temp_dir().join(format!("zestengine-graph-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("s.shadergraph");
        save_shader_graph(&path, &default_graph()).unwrap();
        let loaded = load_shader_graph(&path).unwrap();
        assert_eq!(loaded.display_name, "s");
        assert_eq!(loaded.body, generate_glsl(&default_graph()).unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
