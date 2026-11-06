//! Visual shader graph: a node DAG compiled to a GLSL surface-shader body.
//!
//! A `.shadergraph` file stores the graph (RON). [`load_shader_graph`] reads
//! it, [`graph_shader_source`] turns it into the fragment body the existing
//! shader pipeline compiles (see shader_file.rs), and the editor (in
//! zestengine-engine) draws and edits it. The starter node set is:
//! colour constant, texture sample (fixed material slots), math ops, and an
//! output node.

use std::path::Path;

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};

use crate::shader_props::{ShaderLang, ShaderSource, SHADER_VALUE_BASE};

pub const SHADER_GRAPH_EXTENSION: &str = "shadergraph";

/// The value type a node outputs.
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
}

/// One of the fixed material texture slots the standard pipeline binds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphTextureSlot {
    Albedo,
    Normal,
    Orm,
    Emission,
}

impl GraphTextureSlot {
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

/// One shader-graph node. Inputs are stored as ids of other nodes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GraphNodeKind {
    /// Scalar constant.
    Value { value: f32 },
    /// RGBA colour constant.
    Color { color: [f32; 4] },
    /// Sample a fixed material texture; `uv` defaults to `v_uv`.
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
    /// Drop a vec4 to its rgb (also accepts a vec3).
    SwizzleRgb { x: usize },
    /// The surface colour written to `o_color` (after the post chain).
    Output { color: usize },
}

impl GraphNodeKind {
    /// Human-readable node name for the editor.
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
        }
    }

    /// Input slot labels (parallel to the node's stored inputs).
    pub fn input_slots(&self) -> Vec<&'static str> {
        match self {
            Self::Value { .. } | Self::Color { .. } => Vec::new(),
            Self::TextureSample { .. } => vec!["UV"],
            Self::Add { .. } | Self::Sub { .. } | Self::Mul { .. } | Self::Div { .. } | Self::Dot { .. } => {
                vec!["A", "B"]
            }
            Self::Mix { .. } => vec!["A", "B", "T"],
            Self::Sin { .. } | Self::Cos { .. } | Self::OneMinus { .. } | Self::Normalize { .. } | Self::SwizzleRgb { .. } => {
                vec!["X"]
            }
            Self::Clamp { .. } => vec!["X", "Lo", "Hi"],
            Self::Output { .. } => vec!["Color"],
        }
    }

    /// The connected input id at `slot` (None = unconnected / default).
    pub fn input(&self, slot: usize) -> Option<usize> {
        match self {
            Self::TextureSample { uv, .. } if slot == 0 => *uv,
            Self::Add { a, b } | Self::Sub { a, b } | Self::Mul { a, b } | Self::Div { a, b } | Self::Dot { a, b } => match slot {
                0 => Some(*a),
                _ => Some(*b),
            },
            Self::Mix { a, b, t } => match slot {
                0 => Some(*a),
                1 => Some(*b),
                _ => Some(*t),
            },
            Self::Sin { x } | Self::Cos { x } | Self::OneMinus { x } | Self::Normalize { x } | Self::SwizzleRgb { x } => (slot == 0).then_some(*x),
            Self::Clamp { x, lo, hi } => match slot {
                0 => Some(*x),
                1 => Some(*lo),
                _ => Some(*hi),
            },
            Self::Output { color } => (slot == 0).then_some(*color),
            _ => None,
        }
    }

    /// `self` with slot `slot` connected to node `value`.
    pub fn with_input(&self, slot: usize, value: usize) -> Option<Self> {
        let mut k = self.clone();
        match &mut k {
            Self::TextureSample { uv, .. } if slot == 0 => *uv = Some(value),
            Self::Add { a, b } | Self::Sub { a, b } | Self::Mul { a, b } | Self::Div { a, b } | Self::Dot { a, b } => {
                if slot == 0 { *a = value; } else if slot == 1 { *b = value; } else { return None; }
            }
            Self::Mix { a, b, t } => match slot {
                0 => *a = value,
                1 => *b = value,
                2 => *t = value,
                _ => return None,
            },
            Self::Sin { x } | Self::Cos { x } | Self::OneMinus { x } | Self::Normalize { x } | Self::SwizzleRgb { x } => {
                if slot == 0 { *x = value; } else { return None; }
            }
            Self::Clamp { x, lo, hi } => match slot {
                0 => *x = value,
                1 => *lo = value,
                2 => *hi = value,
                _ => return None,
            },
            Self::Output { color } if slot == 0 => *color = value,
            Self::Output { .. } => return None,
            _ => return None,
        }
        Some(k)
    }

    /// Disconnect an optional input (`TextureSample` UV); required inputs are
    /// left untouched (None for no-op / not optional).
    pub fn disconnect(&self, slot: usize) -> Option<Self> {
        let mut k = self.clone();
        match &mut k {
            Self::TextureSample { uv, .. } if slot == 0 && uv.is_some() => {
                *uv = None;
                Some(k)
            }
            _ => None,
        }
    }
}

/// A node plus its editor position.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphNode {
    pub position: [f32; 2],
    pub kind: GraphNodeKind,
}

/// A shader graph: `nodes[i]` is node id `i`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ShaderGraph {
    pub nodes: Vec<GraphNode>,
}

/// A graph a fresh `.shadergraph` starts from: a colour into the output.
pub fn default_graph() -> ShaderGraph {
    ShaderGraph {
        nodes: vec![
            GraphNode {
                position: [-220.0, -20.0],
                kind: GraphNodeKind::Color { color: [0.85, 0.42, 0.2, 1.0] },
            },
            GraphNode { position: [40.0, -20.0], kind: GraphNodeKind::Output { color: 0 } },
        ],
    }
}

/// Topologically order the nodes (dependencies first). Errors on a cycle or a
/// reference to an out-of-range node.
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

/// Infer each node's output type, in dependency order.
pub fn infer_types(graph: &ShaderGraph, order: &[usize]) -> Result<Vec<GraphValueType>, String> {
    let mut ty = vec![GraphValueType::Float; graph.nodes.len()];
    for &id in order {
        let node = &graph.nodes[id];
        let t = |i: usize, id: usize| ty.get(i).copied().ok_or_else(|| format!("node {id} references missing node {i}"));
        ty[id] = match &node.kind {
            GraphNodeKind::Value { .. } => GraphValueType::Float,
            GraphNodeKind::Color { .. } | GraphNodeKind::TextureSample { .. } => GraphValueType::Vec4,
            GraphNodeKind::Add { a, b }
            | GraphNodeKind::Sub { a, b }
            | GraphNodeKind::Mul { a, b }
            | GraphNodeKind::Div { a, b } => common(t(*a, id)?, t(*b, id)?)
                .ok_or_else(|| format!("node {id}: {a} and {b} have incompatible types"))?,
            GraphNodeKind::Mix { a, b, t: f } => {
                let ab = common(t(*a, id)?, t(*b, id)?)
                    .ok_or_else(|| format!("node {id}: {a} and {b} have incompatible types"))?;
                let tf = t(*f, id)?;
                if tf != GraphValueType::Float && tf != ab {
                    return Err(format!("node {id}: mix factor has type {} but inputs are {}", tf.label(), ab.label()));
                }
                ab
            }
            GraphNodeKind::Sin { x } | GraphNodeKind::Cos { x } => {
                let xt = t(*x, id)?;
                if xt != GraphValueType::Float {
                    return Err(format!("node {id}: sine/cosine needs a Float input, got {}", xt.label()));
                }
                GraphValueType::Float
            }
            GraphNodeKind::Clamp { x, lo, hi } => {
                let xt = t(*x, id)?;
                for (slot, i) in [("lo", lo), ("hi", hi)] {
                    let v = t(*i, id)?;
                    if v != GraphValueType::Float && v != xt {
                        return Err(format!("node {id}: clamp {slot} has type {} but x is {}", v.label(), xt.label()));
                    }
                }
                xt
            }
            GraphNodeKind::OneMinus { x } => t(*x, id)?,
            GraphNodeKind::Dot { a, b } => {
                let (at, bt) = (t(*a, id)?, t(*b, id)?);
                if !at.is_vector() || at != bt {
                    return Err(format!("node {id}: dot needs two vectors of the same type"));
                }
                GraphValueType::Float
            }
            GraphNodeKind::Normalize { x } => {
                let xt = t(*x, id)?;
                if !xt.is_vector() {
                    return Err(format!("node {id}: normalize needs a vector input, got {}", xt.label()));
                }
                xt
            }
            GraphNodeKind::SwizzleRgb { x } => {
                let xt = t(*x, id)?;
                if !matches!(xt, GraphValueType::Vec3 | GraphValueType::Vec4) {
                    return Err(format!("node {id}: rgb needs a vec3/vec4 input, got {}", xt.label()));
                }
                GraphValueType::Vec3
            }
            GraphNodeKind::Output { color } => {
                let ct = t(*color, id)?;
                if !matches!(ct, GraphValueType::Vec3 | GraphValueType::Vec4) {
                    return Err(format!("node {id}: output needs a vec3/vec4 colour, got {}", ct.label()));
                }
                GraphValueType::Vec4
            }
        };
    }
    Ok(ty)
}

/// Validate the graph and return the inferred types (a convenience for the
/// editor's error display).
pub fn validate(graph: &ShaderGraph) -> Result<Vec<GraphValueType>, String> {
    let order = topo_order(graph)?;
    infer_types(graph, &order)
}

/// `a` and `b`'s common arithmetic type; a scalar widens to a vector.
fn common(a: GraphValueType, b: GraphValueType) -> Option<GraphValueType> {
    match (a, b) {
        (x, y) if x == y => Some(x),
        (GraphValueType::Float, v) | (v, GraphValueType::Float) => Some(v),
        _ => None,
    }
}

/// Emit the GLSL body for the graph: one `void main()` defining the local
/// node variables and writing `o_color` from the Output node.
pub fn generate_glsl(graph: &ShaderGraph) -> Result<String, String> {
    let order = topo_order(graph)?;
    let types = infer_types(graph, &order)?;
    let output = graph
        .nodes
        .iter()
        .enumerate()
        .find(|(_, n)| matches!(n.kind, GraphNodeKind::Output { .. }))
        .map(|(i, _)| i)
        .ok_or_else(|| "graph has no Output node".to_owned())?;

    let mut body = String::from("void main() {\n");
    for &id in &order {
        if id == output {
            continue;
        }
        let line = node_expr(graph, id, &types)?;
        body.push_str("    ");
        body.push_str(&line);
        body.push('\n');
    }

    let color = match &graph.nodes[output].kind {
        GraphNodeKind::Output { color } => *color,
        _ => unreachable!(),
    };
    let color = match types[color] {
        GraphValueType::Vec3 => format!("vec4(v{color}, 1.0)"),
        _ => format!("v{color}"),
    };
    let _ = writeln_body(
        &mut body,
        &format!("o_color = vec4(zestengine_postfx({color}.rgb, gl_FragCoord.xy), {color}.a);"),
    );
    body.push_str("}\n");
    Ok(body)
}

fn writeln_body(body: &mut String, line: &str) -> std::fmt::Result {
    body.push_str("    ");
    body.push_str(line);
    body.push('\n');
    Ok(())
}

/// One statement for a non-output node (`types` is the inferred type per node).
fn node_expr(graph: &ShaderGraph, id: usize, types: &[GraphValueType]) -> Result<String, String> {
    let node = &graph.nodes[id];
    let ty = types[id];
    let promoted = |expr: String, from: GraphValueType| {
        if from == ty {
            expr
        } else {
            format!("{}({expr})", ty.glsl())
        }
    };
    let line = match &node.kind {
        GraphNodeKind::Value { value } => format!("float v{id} = {value};"),
        GraphNodeKind::Color { color } => {
            format!("vec4 v{id} = vec4({}, {}, {}, {});", color[0], color[1], color[2], color[3])
        }
        GraphNodeKind::TextureSample { slot, uv } => {
            let uv = uv.map(|u| format!("v{u}")).unwrap_or_else(|| "v_uv".to_owned());
            format!("vec4 v{id} = texture({}, {uv});", slot.sampler())
        }
        GraphNodeKind::Add { a, b } => {
            let a = promoted(format!("v{a}"), types[*a]);
            let b = promoted(format!("v{b}"), types[*b]);
            format!("{} v{id} = {a} + {b};", ty.glsl())
        }
        GraphNodeKind::Sub { a, b } => {
            let a = promoted(format!("v{a}"), types[*a]);
            let b = promoted(format!("v{b}"), types[*b]);
            format!("{} v{id} = {a} - {b};", ty.glsl())
        }
        GraphNodeKind::Mul { a, b } => {
            let a = promoted(format!("v{a}"), types[*a]);
            let b = promoted(format!("v{b}"), types[*b]);
            format!("{} v{id} = {a} * {b};", ty.glsl())
        }
        GraphNodeKind::Div { a, b } => {
            let a = promoted(format!("v{a}"), types[*a]);
            let b = promoted(format!("v{b}"), types[*b]);
            format!("{} v{id} = {a} / {b};", ty.glsl())
        }
        GraphNodeKind::Mix { a, b, t: f } => {
            let a = promoted(format!("v{a}"), types[*a]);
            let b = promoted(format!("v{b}"), types[*b]);
            let f = promoted(format!("v{f}"), types[*f]);
            format!("{} v{id} = mix({a}, {b}, {f});", ty.glsl())
        }
        GraphNodeKind::Sin { x } => format!("float v{id} = sin(v{x});"),
        GraphNodeKind::Cos { x } => format!("float v{id} = cos(v{x});"),
        GraphNodeKind::Clamp { x, lo, hi } => {
            let lo = promoted(format!("v{lo}"), types[*lo]);
            let hi = promoted(format!("v{hi}"), types[*hi]);
            format!("{} v{id} = clamp(v{x}, {lo}, {hi});", ty.glsl())
        }
        GraphNodeKind::OneMinus { x } => {
            let one = if ty == GraphValueType::Float { "1.0".to_owned() } else { format!("{}(1.0)", ty.glsl()) };
            format!("{} v{id} = {one} - v{x};", ty.glsl())
        }
        GraphNodeKind::Dot { a, b } => format!("float v{id} = dot(v{a}, v{b});"),
        GraphNodeKind::Normalize { x } => format!("{} v{id} = normalize(v{x});", ty.glsl()),
        GraphNodeKind::SwizzleRgb { x } => format!("vec3 v{id} = v{x}.rgb;"),
        GraphNodeKind::Output { .. } => return Err("Output nodes are emitted separately".to_owned()),
    };
    Ok(line)
}

/// Turn the graph into the [`ShaderSource`] the shader library compiles.
pub fn graph_shader_source(graph: &ShaderGraph, fallback_name: &str) -> Result<ShaderSource, String> {
    let body = generate_glsl(graph)?;
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

/// True for `.shadergraph` files.
pub fn is_shader_graph_path(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case(SHADER_GRAPH_EXTENSION))
}

/// Read a `.shadergraph` file into its graph.
pub fn load_graph(path: impl AsRef<Path>) -> Result<ShaderGraph> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    ron::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

/// Read a `.shadergraph` file into a compiled [`ShaderSource`].
pub fn load_shader_graph(path: impl AsRef<Path>) -> Result<ShaderSource> {
    let path = path.as_ref();
    let graph = load_graph(path)?;
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "shader graph".into());
    graph_shader_source(&graph, &name).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
}

/// Save a `.shadergraph` file (RON, like the rest of the project assets).
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
