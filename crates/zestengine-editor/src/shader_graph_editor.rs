//! The Shader Graph tab: a node editor for `.shadergraph` assets. It edits
//! [`zestengine_assets::ShaderGraph`], previews the generated GLSL, and saves
//! the graph as RON so the shader library compiles and hot-reloads it.

use std::path::{Path, PathBuf};

use egui::{Color32, Pos2, Vec2, vec2};
use zestengine_assets::{
    GraphNode, GraphNodeKind, GraphTextureSlot, GraphValueType, ShaderGraph, generate_glsl, SHADER_GRAPH_EXTENSION,
};
use crate::node_canvas::{CanvasTarget, Link, NodeCanvasModel, NodeCanvasState, NodeStyle, PortDesc, PortRef};
use crate::egui_phosphor::regular as ph;
use crate::icons::Icon;
use crate::{theme, widgets};

use crate::anim_controller_editor::{rel_path, unique_path};

/// The names the palette lists, in creation order.
const NODE_KINDS: &[&str] = &[
    "Value",
    "Color",
    "Texture Sample",
    "Add",
    "Subtract",
    "Multiply",
    "Divide",
    "Mix",
    "Sine",
    "Cosine",
    "Clamp",
    "One Minus",
    "Dot",
    "Normalize",
    "RGB",
    "Output",
];

pub struct ShaderGraphTab {
    pub open_request: bool,
    last_file: Option<PathBuf>,
    path: Option<String>,
    graph: ShaderGraph,
    dirty: bool,
    canvas: NodeCanvasState,
    context: Option<(CanvasTarget, Pos2)>,
    /// Toolbar search: non-matching nodes dim.
    filter: String,
    left_width: f32,
    right_width: f32,
    spawn_offset: f32,
}

impl Default for ShaderGraphTab {
    fn default() -> Self {
        Self {
            open_request: false,
            last_file: None,
            path: None,
            graph: ShaderGraph::default(),
            dirty: false,
            canvas: NodeCanvasState::default(),
            context: None,
            filter: String::new(),
            left_width: 240.0,
            right_width: 300.0,
            spawn_offset: 0.0,
        }
    }
}

fn file_stem(path: &str) -> String {
    Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_owned())
}

/// A fresh `.shadergraph` asset in `dir`, opened in the tab.
pub fn create_asset(dir: &Path, root: &Path, tab: &mut ShaderGraphTab) {
    let path = unique_path(dir, "New Shader Graph", SHADER_GRAPH_EXTENSION);
    match zestengine_assets::save_shader_graph(&path, &zestengine_assets::default_graph()) {
        Ok(()) => {
            tab.open(root, &rel_path(root, &path));
            tab.open_request = true;
        }
        Err(e) => tracing::error!("creating shader graph: {e:#}"),
    }
}

/// The editor should focus its tab this frame (callers read this, then the
/// dock tab is opened in `editor_app`).
pub fn take_open_request(ctx: &egui::Context, tab: &mut ShaderGraphTab, root: &Path) -> bool {
    let key = egui::Id::new("zestengine-open-shadergraph");
    if let Some(path) = ctx.data_mut(|d| d.remove_temp::<String>(key)) {
        tab.open(root, &path);
        return true;
    }
    std::mem::take(&mut tab.open_request)
}

/// Inspector body for a selected `.shadergraph` file.
pub fn file_inspector(ui: &mut egui::Ui, path: &Path, root: &Path, tab: &mut ShaderGraphTab) {
    let rel = rel_path(root, path);
    ui.label(egui::RichText::new(file_stem(&rel)).font(theme::semibold(theme::SIZE_TITLE)).color(theme::TEXT_STRONG));
    ui.label(egui::RichText::new(&rel).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_MUTED));
    ui.add_space(8.0);
    if tab.last_file.as_deref() != Some(path) {
        tab.last_file = Some(path.to_owned());
        tab.open(root, &rel);
        tab.open_request = true;
    }
    if widgets::secondary_button(ui, Some(Icon::FileDoc), "Open in Shader Graph").clicked() {
        tab.open(root, &rel);
        tab.open_request = true;
    }
    ui.add_space(6.0);
    widgets::info_note(ui, "Assign this .shadergraph to a material's Shader field to use it on objects.");
}

impl ShaderGraphTab {
    pub fn open(&mut self, root: &Path, rel: &str) {
        if self.path.as_deref() == Some(rel) {
            return;
        }
        if self.dirty {
            self.save(root);
        }
        match zestengine_assets::load_graph(root.join(rel)) {
            Ok(graph) => {
                self.graph = graph;
                self.path = Some(rel.to_owned());
                self.dirty = false;
                self.canvas.reset();
                self.canvas.request_frame(false);
            }
            Err(e) => tracing::error!("opening shader graph {rel}: {e:#}"),
        }
    }

    fn save(&mut self, root: &Path) {
        let Some(path) = &self.path else {
            return;
        };
        match zestengine_assets::save_shader_graph(root.join(path), &self.graph) {
            Ok(()) => self.dirty = false,
            Err(e) => tracing::error!("saving shader graph: {e:#}"),
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, root: &Path) {
        egui::TopBottomPanel::top("zestengine-shadergraph-toolbar")
            .frame(egui::Frame::new().fill(theme::BG_PANEL).inner_margin(egui::Margin::symmetric(12, 6)))
            .show_inside(ui, |ui| self.toolbar(ui, root));

        if self.path.is_none() {
            self.empty_state(ui, root);
            return;
        }

        egui::SidePanel::left("zestengine-shadergraph-left")
            .resizable(false)
            .exact_width(self.left_width)
            .frame(egui::Frame::new().fill(theme::BG_PANEL).inner_margin(egui::Margin::same(12)))
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().id_salt("zestengine-shadergraph-left-scroll").show(ui, |ui| {
                    self.palette(ui);
                });
            });
        egui::SidePanel::right("zestengine-shadergraph-right")
            .resizable(false)
            .exact_width(self.right_width)
            .frame(egui::Frame::new().fill(theme::BG_PANEL).inner_margin(egui::Margin::same(12)))
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().id_salt("zestengine-shadergraph-right-scroll").show(ui, |ui| {
                    self.inspector(ui);
                });
            });
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(theme::BG_VIEWPORT))
            .show_inside(ui, |ui| self.canvas(ui));

        if self.dirty && !ui.input(|i| i.pointer.any_down()) && !ui.ctx().wants_keyboard_input() {
            self.save(root);
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui, root: &Path) {
        ui.horizontal(|ui| {
            ui.set_min_height(theme::H_TOOLBAR);
            ui.label(egui::RichText::new(ph::FLOW_ARROW).size(15.0).color(theme::TEXT_MUTED));
            match &self.path {
                Some(p) => {
                    ui.label(egui::RichText::new(file_stem(p)).font(theme::semibold(theme::SIZE_TITLE)).color(theme::TEXT_STRONG))
                        .on_hover_text(p.as_str());
                    widgets::badge(ui, if self.dirty { "Unsaved" } else { "Saved" });
                }
                None => {
                    ui.label(egui::RichText::new("No shader graph").color(theme::TEXT_MUTED));
                }
            }
            ui.add_space(8.0);
            if widgets::secondary_button(ui, Some(Icon::Plus), "New").clicked() {
                create_asset(&root.join("shaders"), root, self);
                self.open_request = false;
            }
            if self.path.is_some()
                && widgets::icon_button_sized(ui, Icon::Target, "Frame graph", false, Vec2::splat(theme::H_FIELD)).clicked()
            {
                self.canvas.request_frame(false);
            }
            if self.path.is_some()
                && widgets::icon_button_sized(ui, Icon::Snap, "Snap nodes to the grid (Shift while dragging inverts)", self.canvas.snap, Vec2::splat(theme::H_FIELD)).clicked()
            {
                self.canvas.snap = !self.canvas.snap;
            }
            if self.path.is_some() {
                widgets::search_field(ui, &mut self.filter, "Filter nodes", 170.0);
            }
            if self.path.is_some()
                && widgets::secondary_button(ui, Some(Icon::FileDoc), "Save").clicked()
            {
                self.save(root);
            }
        });
    }

    fn empty_state(&mut self, ui: &mut egui::Ui, root: &Path) {
        egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::BG_VIEWPORT)).show_inside(ui, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(ui.available_height() * 0.3);
                ui.label(egui::RichText::new("No shader graph open").font(theme::semibold(theme::SIZE_TITLE)).color(theme::TEXT_STRONG));
                ui.label(egui::RichText::new("Select a .shadergraph in Files, or create one.").color(theme::TEXT_MUTED));
                ui.add_space(10.0);
                if widgets::primary_button(ui, Some(Icon::Plus), "Create Shader Graph").clicked() {
                    create_asset(&root.join("shaders"), root, self);
                    self.open_request = false;
                }
            });
        });
    }

    fn palette(&mut self, ui: &mut egui::Ui) {
        widgets::eyebrow(ui, "Add node");
        for name in NODE_KINDS {
            if ui.add_sized([self.left_width - 24.0, theme::H_FIELD], egui::Button::new(*name)).clicked() {
                let kind = kind_from_name(name, self.selected());
                self.spawn_node(kind);
            }
        }
        ui.add_space(8.0);
        widgets::eyebrow(ui, "Graph");
        let ok = zestengine_assets::validate(&self.graph);
        match ok {
            Ok(types) => {
                widgets::badge(ui, &format!("{} nodes", self.graph.nodes.len()));
                ui.label(egui::RichText::new(format!("Output type: {}", types.last().map(|t| t.label()).unwrap_or("?"))).color(theme::TEXT_MUTED));
            }
            Err(e) => {
                ui.label(egui::RichText::new("Graph invalid").color(theme::STATUS_ERROR).strong());
                ui.add(egui::Label::new(egui::RichText::new(e).color(theme::STATUS_ERROR).small()).wrap());
            }
        }
    }

    fn inspector(&mut self, ui: &mut egui::Ui) {
        widgets::eyebrow(ui, "Node");
        let Some(id) = self.selected() else {
            ui.label(egui::RichText::new("Select a node to edit it.").color(theme::TEXT_MUTED));
            widgets::section_gap(ui);
            self.glsl_preview(ui);
            return;
        };
        let kind = self.graph.nodes[id].kind.clone();
        ui.label(egui::RichText::new(kind.title()).font(theme::semibold(theme::SIZE_BODY)).color(theme::TEXT_STRONG));
        ui.label(egui::RichText::new(format!("Node {id}")).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_MUTED));

        let mut changed = false;
        match &kind {
            GraphNodeKind::Value { value } => {
                let mut v = *value;
                if ui.add(egui::DragValue::new(&mut v).speed(0.01)).changed() {
                    changed = true;
                    if let GraphNodeKind::Value { value } = &mut self.graph.nodes[id].kind {
                        *value = v;
                    }
                }
            }
            GraphNodeKind::Color { color } => {
                let mut c = *color;
                if widgets::color_hex(ui, &mut c).changed() {
                    changed = true;
                    if let GraphNodeKind::Color { color } = &mut self.graph.nodes[id].kind {
                        *color = c;
                    }
                }
            }
            GraphNodeKind::TextureSample { slot, uv } => {
                let mut idx = *slot as usize;
                egui::ComboBox::from_id_salt(("shadergraph-tex-slot", id))
                    .selected_text(slot.label())
                    .show_ui(ui, |ui| {
                        for (i, s) in [GraphTextureSlot::Albedo, GraphTextureSlot::Normal, GraphTextureSlot::Orm, GraphTextureSlot::Emission].iter().enumerate() {
                            ui.selectable_value(&mut idx, i, s.label());
                        }
                    });
                if idx != *slot as usize {
                    changed = true;
                    if let GraphNodeKind::TextureSample { slot, .. } = &mut self.graph.nodes[id].kind {
                        *slot = [GraphTextureSlot::Albedo, GraphTextureSlot::Normal, GraphTextureSlot::Orm, GraphTextureSlot::Emission][idx];
                    }
                }
                let uv_text = uv.map(|u| format!("node {u}")).unwrap_or_else(|| "v_uv".into());
                widgets::property_row(ui, "UV", |ui| {
                    ui.label(egui::RichText::new(uv_text).color(theme::TEXT_MUTED));
                });
                if uv.is_some()
                    && ui.button("Disconnect UV").clicked()
                    && let Some(k) = self.graph.nodes[id].kind.disconnect(0)
                {
                    self.graph.nodes[id].kind = k;
                    changed = true;
                }
            }
            GraphNodeKind::Output { .. } => {
                ui.label(egui::RichText::new("Writes the final surface colour.").color(theme::TEXT_MUTED));
            }
            _ => {
                for (slot, label) in kind.input_slots().iter().enumerate() {
                    let text = kind
                        .input(slot)
                        .map(|src| format!("node {src}"))
                        .unwrap_or_else(|| "unconnected".into());
                    widgets::property_row(ui, *label, |ui| {
                        ui.label(egui::RichText::new(text).color(theme::TEXT_MUTED));
                    });
                }
            }
        }

        if ui.button("Delete Node").clicked() {
            self.delete_node(id);
            self.canvas.clear_selection();
            return;
        }

        if changed {
            self.dirty = true;
        }

        widgets::section_gap(ui);
        self.glsl_preview(ui);
    }

    fn glsl_preview(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Generated GLSL")
            .default_open(false)
            .show(ui, |ui| match generate_glsl(&self.graph) {
                Ok(body) => {
                    ui.add(
                        egui::Label::new(egui::RichText::new(body).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_MUTED))
                            .wrap(),
                    );
                }
                Err(e) => {
                    ui.label(egui::RichText::new(e).color(theme::STATUS_ERROR).small());
                }
            });
    }

    // --------------------------------------------------------------- canvas

    fn selected(&self) -> Option<usize> {
        self.canvas.primary.filter(|&i| i < self.graph.nodes.len())
    }

    fn spawn_node(&mut self, kind: GraphNodeKind) -> usize {
        let c = self.canvas.view.center_canvas();
        let at = [c.x - 80.0 + self.spawn_offset, c.y - 50.0 + self.spawn_offset];
        self.spawn_offset = (self.spawn_offset + 28.0) % 140.0;
        self.spawn_node_at(kind, at)
    }

    fn spawn_node_at(&mut self, kind: GraphNodeKind, at: [f32; 2]) -> usize {
        self.graph.nodes.push(GraphNode { position: at, kind });
        let id = self.graph.nodes.len() - 1;
        self.canvas.select_only(id);
        self.dirty = true;
        id
    }

    fn delete_node(&mut self, id: usize) {
        delete_graph_node(&mut self.graph, id);
        self.dirty = true;
    }

    fn canvas(&mut self, ui: &mut egui::Ui) {
        let out = {
            let mut model = GraphModel {
                types: node_types(&self.graph),
                filter: self.filter.trim().to_lowercase(),
                graph: &mut self.graph,
            };
            self.canvas.show(ui, &mut model)
        };
        if out.changed {
            self.dirty = true;
        }
        if out.context.is_some() {
            self.context = out.context;
        }
        out.response.context_menu(|ui| {
            widgets::menu_frame(ui);
            let Some((target, at)) = self.context else {
                ui.close();
                return;
            };
            match target {
                CanvasTarget::Node(_) => {
                    if ui.button("Delete").clicked() {
                        let mut model = GraphModel {
                            types: node_types(&self.graph),
                            filter: String::new(),
                            graph: &mut self.graph,
                        };
                        if self.canvas.delete_selection(&mut model) {
                            self.dirty = true;
                        }
                        ui.close();
                    }
                }
                CanvasTarget::Wire(link) => {
                    let removable = self
                        .graph
                        .nodes
                        .get(link.to.node)
                        .is_some_and(|n| n.kind.disconnect(link.to.port).is_some());
                    if ui.add_enabled(removable, egui::Button::new("Delete Wire")).clicked() {
                        if let Some(k) = self.graph.nodes[link.to.node].kind.disconnect(link.to.port) {
                            self.graph.nodes[link.to.node].kind = k;
                            self.canvas.selected_links.clear();
                            self.dirty = true;
                        }
                        ui.close();
                    }
                    if !removable {
                        ui.label(egui::RichText::new("This input is required: drop another wire on it to replace this one.").color(theme::TEXT_MUTED).small());
                    }
                }
                CanvasTarget::Canvas => {
                    ui.menu_button("Add Node", |ui| {
                        widgets::menu_frame(ui);
                        for name in NODE_KINDS {
                            if ui.button(*name).clicked() {
                                let kind = kind_from_name(name, self.selected());
                                self.spawn_node_at(kind, [at.x, at.y]);
                                ui.close();
                            }
                        }
                    });
                }
            }
        });
    }
}

/// Remove node `id`, reconnecting or clamping the inputs that referenced it.
fn delete_graph_node(graph: &mut ShaderGraph, id: usize) {
    graph.nodes.remove(id);
    let max = graph.nodes.len().saturating_sub(1);
    for node in &mut graph.nodes {
        let mut slot = 0;
        while slot < node.kind.input_slots().len() {
            let Some(src) = node.kind.input(slot) else {
                slot += 1;
                continue;
            };
            if src == id {
                if let Some(k) = node.kind.disconnect(slot) {
                    node.kind = k;
                    slot += 1;
                    continue;
                }
                if let Some(k) = node.kind.with_input(slot, 0) {
                    node.kind = k;
                }
            } else if src > id
                && let Some(k) = node.kind.with_input(slot, (src - 1).min(max))
            {
                node.kind = k;
            }
            slot += 1;
        }
    }
}

// ------------------------------------------------------- canvas model

/// Input slot labels without allocating (mirrors `GraphNodeKind::input_slots`).
fn slot_labels(kind: &GraphNodeKind) -> &'static [&'static str] {
    use GraphNodeKind as K;
    match kind {
        K::Value { .. } | K::Color { .. } => &[],
        K::TextureSample { .. } => &["UV"],
        K::Add { .. } | K::Sub { .. } | K::Mul { .. } | K::Div { .. } | K::Dot { .. } => &["A", "B"],
        K::Mix { .. } => &["A", "B", "T"],
        K::Sin { .. } | K::Cos { .. } | K::OneMinus { .. } | K::Normalize { .. } | K::SwizzleRgb { .. } => &["X"],
        K::Clamp { .. } => &["X", "Lo", "Hi"],
        K::Output { .. } => &["Color"],
    }
}

fn output_label(kind: &GraphNodeKind) -> &'static str {
    match kind {
        GraphNodeKind::Value { .. } => "Value",
        GraphNodeKind::Color { .. } => "Color",
        GraphNodeKind::TextureSample { .. } => "RGBA",
        GraphNodeKind::Dot { .. } => "Result",
        _ => "Out",
    }
}

/// Category chip colour and glyph per node kind.
fn category(kind: &GraphNodeKind) -> (Color32, &'static str) {
    use GraphNodeKind as K;
    match kind {
        K::Value { .. } | K::Color { .. } => (theme::CAT_INPUT, ph::SIGN_IN),
        K::TextureSample { .. } => (theme::CAT_TEXTURE, ph::IMAGE),
        K::Dot { .. } | K::Normalize { .. } | K::SwizzleRgb { .. } => (theme::CAT_VECTOR, ph::ARROWS_OUT_CARDINAL),
        K::Output { .. } => (theme::CAT_OUTPUT, ph::SIGN_OUT),
        _ => (theme::CAT_MATH, ph::PLUS_MINUS),
    }
}

fn type_color(t: Option<GraphValueType>) -> Color32 {
    match t {
        Some(GraphValueType::Float) => theme::WIRE_FLOAT,
        Some(GraphValueType::Vec2) => theme::WIRE_VEC2,
        Some(GraphValueType::Vec3) => theme::WIRE_VEC3,
        Some(GraphValueType::Vec4) => theme::WIRE_VEC4,
        None => theme::TEXT_DIM,
    }
}

fn type_label(t: Option<GraphValueType>) -> &'static str {
    t.map_or("Any", |t| t.label())
}

/// Output type per node: the validated inference, or a per-kind guess when the
/// graph does not validate (so wires keep their colours while it is broken).
fn node_types(graph: &ShaderGraph) -> Vec<Option<GraphValueType>> {
    if let Ok(types) = zestengine_assets::topo_order(graph).and_then(|o| zestengine_assets::infer_types(graph, &o)) {
        return types.into_iter().map(Some).collect();
    }
    graph
        .nodes
        .iter()
        .map(|n| match n.kind {
            GraphNodeKind::Value { .. } | GraphNodeKind::Sin { .. } | GraphNodeKind::Cos { .. } | GraphNodeKind::Dot { .. } => {
                Some(GraphValueType::Float)
            }
            GraphNodeKind::Color { .. } | GraphNodeKind::TextureSample { .. } | GraphNodeKind::Output { .. } => Some(GraphValueType::Vec4),
            GraphNodeKind::SwizzleRgb { .. } => Some(GraphValueType::Vec3),
            _ => None,
        })
        .collect()
}

/// Would wiring `from` into an input of `to` close a loop (`from` already
/// depends on `to`)?
fn depends_on(graph: &ShaderGraph, from: usize, to: usize) -> bool {
    let mut seen = vec![false; graph.nodes.len()];
    let mut stack = vec![from];
    while let Some(n) = stack.pop() {
        if n == to {
            return true;
        }
        if n >= seen.len() || std::mem::replace(&mut seen[n], true) {
            continue;
        }
        for slot in 0..slot_labels(&graph.nodes[n].kind).len() {
            if let Some(src) = graph.nodes[n].kind.input(slot) {
                stack.push(src);
            }
        }
    }
    false
}

/// [`NodeCanvasModel`] over a shader graph for one frame.
struct GraphModel<'a> {
    graph: &'a mut ShaderGraph,
    types: Vec<Option<GraphValueType>>,
    /// Lower-case node-title filter ("" = show all).
    filter: String,
}

impl GraphModel<'_> {
    fn ty(&self, node: usize) -> Option<GraphValueType> {
        self.types.get(node).copied().flatten()
    }
}

impl NodeCanvasModel for GraphModel<'_> {
    fn node_count(&self) -> usize {
        self.graph.nodes.len()
    }

    fn position(&self, node: usize) -> Pos2 {
        Pos2::from(self.graph.nodes[node].position)
    }

    fn set_position(&mut self, node: usize, pos: Pos2) {
        self.graph.nodes[node].position = [pos.x, pos.y];
    }

    fn style(&self, node: usize) -> NodeStyle {
        let kind = &self.graph.nodes[node].kind;
        let (chip, glyph) = category(kind);
        NodeStyle { title: kind.title().into(), chip, glyph }
    }

    fn input_count(&self, node: usize) -> usize {
        slot_labels(&self.graph.nodes[node].kind).len()
    }

    fn output_count(&self, node: usize) -> usize {
        usize::from(!matches!(self.graph.nodes[node].kind, GraphNodeKind::Output { .. }))
    }

    fn input(&self, node: usize, port: usize) -> PortDesc {
        let kind = &self.graph.nodes[node].kind;
        let label = slot_labels(kind)[port];
        // A wired input shows its source's type; a free UV is a Vec2.
        let t = match kind.input(port) {
            Some(src) => self.ty(src),
            None if matches!(kind, GraphNodeKind::TextureSample { .. }) => Some(GraphValueType::Vec2),
            None => None,
        };
        PortDesc { label: label.into(), type_label: type_label(t), color: type_color(t) }
    }

    fn output(&self, node: usize, _port: usize) -> PortDesc {
        let t = self.ty(node);
        PortDesc { label: output_label(&self.graph.nodes[node].kind).into(), type_label: type_label(t), color: type_color(t) }
    }

    fn links(&self, out: &mut Vec<Link>) {
        for (to, node) in self.graph.nodes.iter().enumerate() {
            for port in 0..slot_labels(&node.kind).len() {
                let Some(src) = node.kind.input(port) else { continue };
                if src >= self.graph.nodes.len() || matches!(self.graph.nodes[src].kind, GraphNodeKind::Output { .. }) {
                    continue;
                }
                out.push(Link { from: PortRef { node: src, port: 0 }, to: PortRef { node: to, port }, color: type_color(self.ty(src)) });
            }
        }
    }

    fn can_connect(&self, from: PortRef, to: PortRef) -> bool {
        from.node != to.node
            && from.node < self.graph.nodes.len()
            && to.node < self.graph.nodes.len()
            && !matches!(self.graph.nodes[from.node].kind, GraphNodeKind::Output { .. })
            && self.graph.nodes[to.node].kind.with_input(to.port, from.node).is_some()
            && !depends_on(self.graph, from.node, to.node)
    }

    fn connect(&mut self, from: PortRef, to: PortRef) -> bool {
        let Some(k) = self.graph.nodes[to.node].kind.with_input(to.port, from.node) else { return false };
        self.graph.nodes[to.node].kind = k;
        true
    }

    fn can_disconnect(&self, link: &Link) -> bool {
        self.graph.nodes[link.to.node].kind.disconnect(link.to.port).is_some()
    }

    fn disconnect(&mut self, link: &Link) -> bool {
        let Some(k) = self.graph.nodes[link.to.node].kind.disconnect(link.to.port) else { return false };
        self.graph.nodes[link.to.node].kind = k;
        true
    }

    fn delete_node(&mut self, node: usize) {
        delete_graph_node(self.graph, node);
    }

    fn matches_filter(&self, node: usize) -> bool {
        self.filter.is_empty() || self.graph.nodes[node].kind.title().to_lowercase().contains(&self.filter)
    }

    fn body_height(&self, node: usize) -> f32 {
        match self.graph.nodes[node].kind {
            GraphNodeKind::Value { .. } | GraphNodeKind::Color { .. } | GraphNodeKind::TextureSample { .. } => 32.0,
            _ => 0.0,
        }
    }

    fn draw_body(&mut self, ui: &mut egui::Ui, node: usize) -> bool {
        match &mut self.graph.nodes[node].kind {
            GraphNodeKind::Value { value } => {
                let size = vec2(ui.available_width(), ui.available_height());
                ui.add_sized(size, egui::DragValue::new(value).speed(0.01)).changed()
            }
            GraphNodeKind::Color { color } => {
                let r = ui.color_edit_button_rgba_unmultiplied(color);
                let c = Color32::from_rgb(
                    (color[0].clamp(0.0, 1.0) * 255.0) as u8,
                    (color[1].clamp(0.0, 1.0) * 255.0) as u8,
                    (color[2].clamp(0.0, 1.0) * 255.0) as u8,
                );
                ui.label(egui::RichText::new(format!("#{:02X}{:02X}{:02X}", c.r(), c.g(), c.b())).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_MUTED));
                r.changed()
            }
            GraphNodeKind::TextureSample { slot, .. } => {
                let before = *slot;
                egui::ComboBox::from_id_salt(("shadergraph-tex-slot", node))
                    .selected_text(slot.label())
                    .width(ui.available_width())
                    .show_ui(ui, |ui| {
                        for s in [GraphTextureSlot::Albedo, GraphTextureSlot::Normal, GraphTextureSlot::Orm, GraphTextureSlot::Emission] {
                            ui.selectable_value(slot, s, s.label());
                        }
                    });
                *slot != before
            }
            _ => false,
        }
    }

    fn has_port_default(&self, node: usize, _input: usize) -> bool {
        matches!(self.graph.nodes[node].kind, GraphNodeKind::TextureSample { .. })
    }

    fn draw_port_default(&mut self, ui: &mut egui::Ui, _node: usize, _input: usize) -> bool {
        ui.label(egui::RichText::new("v_uv").font(theme::mono(theme::SIZE_META)).color(theme::TEXT_DIM));
        false
    }
}

fn kind_from_name(name: &str, hint: Option<usize>) -> GraphNodeKind {
    let i = hint.unwrap_or(0);
    match name {
        "Value" => GraphNodeKind::Value { value: 0.5 },
        "Color" => GraphNodeKind::Color { color: [0.85, 0.42, 0.2, 1.0] },
        "Texture Sample" => GraphNodeKind::TextureSample { slot: GraphTextureSlot::Albedo, uv: None },
        "Add" => GraphNodeKind::Add { a: i, b: i },
        "Subtract" => GraphNodeKind::Sub { a: i, b: i },
        "Multiply" => GraphNodeKind::Mul { a: i, b: i },
        "Divide" => GraphNodeKind::Div { a: i, b: i },
        "Mix" => GraphNodeKind::Mix { a: i, b: i, t: i },
        "Sine" => GraphNodeKind::Sin { x: i },
        "Cosine" => GraphNodeKind::Cos { x: i },
        "Clamp" => GraphNodeKind::Clamp { x: i, lo: i, hi: i },
        "One Minus" => GraphNodeKind::OneMinus { x: i },
        "Dot" => GraphNodeKind::Dot { a: i, b: i },
        "Normalize" => GraphNodeKind::Normalize { x: i },
        "RGB" => GraphNodeKind::SwizzleRgb { x: i },
        "Output" => GraphNodeKind::Output { color: i },
        _ => GraphNodeKind::Value { value: 0.0 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn every_kind() -> Vec<GraphNodeKind> {
        NODE_KINDS.iter().map(|n| kind_from_name(n, Some(0))).collect()
    }

    #[test]
    fn slot_labels_match_the_asset_crate() {
        for k in every_kind() {
            assert_eq!(slot_labels(&k), k.input_slots().as_slice(), "{}", k.title());
        }
    }

    #[test]
    fn cycles_are_refused_and_links_are_typed() {
        let mut graph = zestengine_assets::default_graph();
        graph.nodes.push(GraphNode { position: [0.0, 0.0], kind: GraphNodeKind::OneMinus { x: 0 } });
        graph.nodes.push(GraphNode { position: [0.0, 0.0], kind: GraphNodeKind::Normalize { x: 2 } });
        let types = node_types(&graph);
        let model = GraphModel { types, filter: String::new(), graph: &mut graph };
        let p = |node, port| PortRef { node, port };
        // Output (1) feeding anything, or a node feeding itself, is refused.
        assert!(!model.can_connect(p(1, 0), p(2, 0)));
        assert!(!model.can_connect(p(2, 0), p(2, 0)));
        // 3 depends on 2: wiring 3 back into 2 would loop.
        assert!(!model.can_connect(p(3, 0), p(2, 0)));
        assert!(model.can_connect(p(0, 0), p(3, 0)));
        let mut links = Vec::new();
        model.links(&mut links);
        assert_eq!(links.len(), 3);
        assert!(links.iter().all(|l| l.color == theme::WIRE_VEC4 || l.color == theme::TEXT_DIM));
    }
}

#[cfg(test)]
#[path = "shots/shader_graph.rs"]
mod shots;
