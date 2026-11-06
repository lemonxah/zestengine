//! Inspectors and viewport gizmos for bone cloth (`Bone Cloth`) and its
//! colliders (`Cloth Colliders`), including the capsule auto-fit controls,
//! and the Mesh Cloth pin painter's panel.

use zestengine_core::{
    BoneCloth, BrushFalloff, BrushMode, ClothCollider, ClothColliders, ClothConnection, ClothFitPreset, ClothShape,
    ClothShapeWorld, ClothTeleport, ClothWind, MeshCloth, ObjectId, ObjectRef, PaintChannel, PinSource, WindShape,
};
use egui::{DragValue, RichText, Ui};
use glam::{Mat4, Vec3};

use crate::components::{EditorComponents, Gizmo, GizmoEdit, GizmoSpec, Inspect, InspectCtx};
use crate::icons::Icon;
use crate::theme;
use crate::widgets;

pub(crate) fn register(e: &mut EditorComponents) {
    e.register::<BoneCloth>();
    e.register::<ClothColliders>();
    e.register::<ClothWind>();
    e.register::<MeshCloth>();
}

const COLLIDER: egui::Color32 = egui::Color32::from_rgb(90, 200, 250);
const COLLIDER_SELECTED: egui::Color32 = egui::Color32::from_rgb(255, 210, 90);
const CHAIN: egui::Color32 = egui::Color32::from_rgb(250, 150, 210);
const CHAIN_FIXED: egui::Color32 = egui::Color32::from_rgb(255, 90, 90);
const LINK: egui::Color32 = egui::Color32::from_rgb(170, 110, 160);

fn drag(ui: &mut Ui, label: &str, v: &mut f32, speed: f64, range: std::ops::RangeInclusive<f32>, changed: &mut bool) {
    widgets::property_row(ui, label, |ui| *changed |= ui.add(DragValue::new(v).speed(speed).range(range)).changed());
}

fn check(ui: &mut Ui, label: &str, v: &mut bool, changed: &mut bool) {
    widgets::property_row(ui, label, |ui| *changed |= ui.checkbox(v, "").changed());
}

fn vec3(ui: &mut Ui, label: &str, v: &mut [f32; 3], changed: &mut bool) {
    widgets::property_row(ui, label, |ui| *changed |= widgets::vec3_field(ui, v, 3, 0.005));
}

/// Cull distance row, with a note while the cloth is culled.
fn cull_row(ui: &mut Ui, v: &mut f32, culled: bool, changed: &mut bool) {
    drag(ui, "Cull distance (m)", v, 0.5, 0.0..=10000.0, changed);
    if *v > 0.0 {
        note(ui, if culled { "Culled: farther than this from the camera, so it holds the animated pose." } else { "Holds the animated pose beyond this distance from the camera (0 = never)." });
    }
}

/// Root-to-tip pair of values on one row.
fn pair(ui: &mut Ui, label: &str, v: &mut [f32; 2], speed: f64, range: std::ops::RangeInclusive<f32>, changed: &mut bool) {
    widgets::property_row(ui, label, |ui| {
        *changed |= ui.add(DragValue::new(&mut v[0]).speed(speed).range(range.clone()).prefix("root ")).changed();
        *changed |= ui.add(DragValue::new(&mut v[1]).speed(speed).range(range).prefix("tip ")).changed();
    });
}

fn combo<T: Copy + PartialEq>(ui: &mut Ui, id: &str, label: &str, value: &mut T, all: &[T], name: fn(T) -> &'static str, changed: &mut bool) {
    widgets::property_row(ui, label, |ui| {
        egui::ComboBox::from_id_salt(id).width(ui.available_width()).selected_text(name(*value)).show_ui(ui, |ui| {
            for &k in all {
                *changed |= ui.selectable_value(value, k, name(k)).changed();
            }
        });
    });
}

/// The Mesh Cloth pin painter's settings, shared between its inspector panel
/// and the Scene view through egui memory.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClothPaintTool {
    /// The object being painted (the selection when painting started).
    pub object: Option<ObjectId>,
    pub active: bool,
    pub channel: PaintChannel,
    pub mode: BrushMode,
    pub falloff: BrushFalloff,
    /// Brush radius (world units).
    pub radius: f32,
    pub strength: f32,
    /// What Set paints and Fill fills with.
    pub value: f32,
    pub mirror_x: bool,
    /// Run the cloth while painting (otherwise it holds the animated pose).
    pub simulate: bool,
    pub show_distance: bool,
    /// Share of the mesh height the region buttons take.
    pub height: f32,
    /// A stroke is in progress (one undo step).
    pub stroking: bool,
}

impl Default for ClothPaintTool {
    fn default() -> Self {
        Self {
            object: None,
            active: false,
            channel: PaintChannel::Pin,
            mode: BrushMode::Paint,
            falloff: BrushFalloff::Smooth,
            radius: 0.05,
            strength: 0.5,
            value: 1.0,
            mirror_x: false,
            simulate: false,
            show_distance: false,
            height: 0.15,
            stroking: false,
        }
    }
}

impl ClothPaintTool {
    fn id() -> egui::Id {
        egui::Id::new("zestengine-cloth-paint-tool")
    }

    pub fn get(ctx: &egui::Context) -> Self {
        ctx.data(|d| d.get_temp(Self::id())).unwrap_or_default()
    }

    pub fn set(self, ctx: &egui::Context) {
        ctx.data_mut(|d| d.insert_temp(Self::id(), self));
    }
}

pub const CLOTH_PAINT_HINT: &str = "Drag on the cloth to paint. Shift erases, Alt smooths. [ and ] or Ctrl+Scroll resize the brush. \
    P or Esc leaves painting. Each stroke is one undo step.";

fn channel_label(c: PaintChannel) -> &'static str {
    match c {
        PaintChannel::Pin => "Pin weight",
        PaintChannel::MaxDistance => "Max distance",
    }
}

/// The painter's brush settings and whole-mesh operations. Edits the paint
/// in place; true when it changed.
fn paint_panel(ui: &mut Ui, mc: &mut MeshCloth, tool: &mut ClothPaintTool) -> bool {
    let mut changed = false;
    if widgets::primary_button(ui, Some(Icon::Check), "Done Painting").clicked() {
        tool.active = false;
    }
    note(ui, CLOTH_PAINT_HINT);
    widgets::property_row(ui, "Channel", |ui| {
        let opts = [(PaintChannel::Pin, channel_label(PaintChannel::Pin)), (PaintChannel::MaxDistance, channel_label(PaintChannel::MaxDistance))];
        widgets::segmented(ui, &mut tool.channel, &opts);
    });
    widgets::property_row(ui, "Brush", |ui| {
        let opts = BrushMode::ALL.map(|m| (m, m.label()));
        widgets::segmented(ui, &mut tool.mode, &opts);
    });
    if tool.mode == BrushMode::Set {
        widgets::property_row(ui, "Value", |ui| widgets::slider(ui, egui::Slider::new(&mut tool.value, 0.0..=1.0)));
    }
    widgets::property_row(ui, "Radius (m)", |ui| ui.add(DragValue::new(&mut tool.radius).speed(0.002).range(0.002..=5.0)));
    widgets::property_row(ui, "Strength", |ui| widgets::slider(ui, egui::Slider::new(&mut tool.strength, 0.0..=1.0)));
    widgets::property_row(ui, "Falloff", |ui| {
        let opts = BrushFalloff::ALL.map(|f| (f, f.label()));
        widgets::segmented(ui, &mut tool.falloff, &opts);
    });
    widgets::property_row(ui, "Mirror X", |ui| ui.checkbox(&mut tool.mirror_x, ""));
    widgets::property_row(ui, "Simulate", |ui| ui.checkbox(&mut tool.simulate, "")).on_hover_text("Run the cloth while painting");
    if tool.channel == PaintChannel::MaxDistance || !mc.paint.max_distance.is_empty() {
        widgets::property_row(ui, "Show distance", |ui| ui.checkbox(&mut tool.show_distance, ""));
    }

    let n = mc.paint.vertex_count as usize;
    if n == 0 || !mc.paint.matches(mc.runtime.bind.len(), mc.runtime.topology) {
        note(ui, "Waiting for the cloth to build (the object must be skinned and active).");
        return changed;
    }
    let channel = tool.channel;
    let mut values = mc.paint.values(channel);
    let mut edit = false;
    widgets::eyebrow(ui, "Whole mesh");
    ui.horizontal_wrapped(|ui| {
        if widgets::secondary_button(ui, None, "Fill").on_hover_text("Set every vertex to the brush value").clicked() {
            values.iter_mut().for_each(|v| *v = tool.value);
            edit = true;
        }
        if widgets::secondary_button(ui, None, "Clear").clicked() {
            values.iter_mut().for_each(|v| *v = 0.0);
            edit = true;
        }
        if widgets::secondary_button(ui, None, "Invert").clicked() {
            values.iter_mut().for_each(|v| *v = 1.0 - *v);
            edit = true;
        }
    });
    ui.horizontal_wrapped(|ui| {
        let tolerance = (mc.paint.bounds[3] - mc.paint.bounds[0]).abs().max(1e-3) * 0.01;
        for (label, positive) in [("Mirror +X to -X", true), ("Mirror -X to +X", false)] {
            if widgets::secondary_button(ui, None, label).on_hover_text("Copy each vertex's value to its mirror across local X").clicked() {
                mc.paint.set_values(channel, &values);
                mc.paint.symmetrize(channel, &mc.runtime.bind, positive, tolerance);
                values = mc.paint.values(channel);
                edit = true;
            }
        }
    });
    widgets::property_row(ui, "Height share", |ui| widgets::slider(ui, egui::Slider::new(&mut tool.height, 0.0..=1.0)));
    ui.horizontal_wrapped(|ui| {
        for (label, above) in [("Set top", true), ("Set bottom", false)] {
            if widgets::secondary_button(ui, None, label).on_hover_text("Set that share of the mesh height to the brush value").clicked() {
                for (v, take) in zestengine_core::select_by_height(&mc.runtime.bind, tool.height, above).into_iter().enumerate() {
                    if take {
                        values[v] = tool.value;
                    }
                }
                edit = true;
            }
        }
    });
    if edit {
        mc.paint.set_values(channel, &values);
        mc.runtime.retether = true;
        changed = true;
    }
    if !mc.paint.max_distance.is_empty() && widgets::secondary_button(ui, Some(Icon::Trash), "Remove max distance").clicked() {
        mc.paint.max_distance.set(Vec::new());
        mc.runtime.retether = true;
        changed = true;
    }
    changed
}

fn note(ui: &mut Ui, s: &str) {
    ui.label(RichText::new(s).size(theme::SIZE_META).color(theme::TEXT_DIM));
}

/// A bone picker with a search box. Returns the picked bone; `current` is the
/// button text. `none` adds an entry for "no bone" (returned as "").
fn bone_picker(ui: &mut Ui, id: &str, current: &str, bones: &[String], none: Option<&str>) -> Option<String> {
    let mut picked = None;
    let search_id = egui::Id::new((id, "search"));
    egui::ComboBox::from_id_salt(id)
        .width(ui.available_width())
        .height(320.0)
        .selected_text(if current.is_empty() { none.unwrap_or("(pick a bone)") } else { current })
        .show_ui(ui, |ui| {
            let mut filter: String = ui.data(|d| d.get_temp(search_id)).unwrap_or_default();
            if widgets::search_field(ui, &mut filter, "Filter bones", ui.available_width()).changed() {
                ui.data_mut(|d| d.insert_temp(search_id, filter.clone()));
            }
            if let Some(label) = none
                && ui.selectable_label(current.is_empty(), label).clicked()
            {
                picked = Some(String::new());
            }
            let f = filter.to_lowercase();
            for b in bones.iter().filter(|b| f.is_empty() || b.to_lowercase().contains(&f)) {
                if ui.selectable_label(b == current, b).clicked() {
                    picked = Some(b.clone());
                }
            }
            if bones.is_empty() {
                note(ui, "No skeleton on this object yet");
            }
        });
    picked
}

/// Editable list of bone names with remove buttons and an add picker.
fn bone_list(ui: &mut Ui, id: &str, list: &mut Vec<String>, bones: &[String], add_label: &str, changed: &mut bool) {
    let mut remove = None;
    for (k, name) in list.iter().enumerate() {
        ui.horizontal(|ui| {
            let missing = !bones.is_empty() && !bones.contains(name);
            let text = RichText::new(name).color(if missing { theme::STATUS_ERROR } else { theme::TEXT_BODY });
            ui.label(text).on_hover_text(if missing { "Not in this skeleton" } else { "" });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if widgets::icon_button(ui, Icon::Trash, "Remove", false).clicked() {
                    remove = Some(k);
                }
            });
        });
    }
    if let Some(k) = remove {
        list.remove(k);
        *changed = true;
    }
    let available: Vec<String> = bones.iter().filter(|b| !list.contains(b)).cloned().collect();
    widgets::property_row(ui, add_label, |ui| {
        if let Some(b) = bone_picker(ui, id, "", &available, None) {
            list.push(b);
            *changed = true;
        }
    });
}

// ---------------------------------------------------------------- colliders

impl Inspect for ClothColliders {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        let bones = self.runtime.bones.clone();
        if bones.is_empty() {
            widgets::info_note(ui, "Colliders follow joints of the skinned model on this object. With no skeleton they follow the object itself.");
        }
        widgets::eyebrow(ui, "Colliders");
        let mut remove = None;
        for (k, c) in self.colliders.iter_mut().enumerate() {
            let selected = self.runtime.selected == Some(k);
            ui.horizontal(|ui| {
                changed |= ui.checkbox(&mut c.enabled, "").changed();
                let label = format!("{}  {}", c.name, c.shape.label());
                let row = ui.add_sized(
                    [ui.available_width() - 28.0, theme::H_ROW],
                    egui::Button::selectable(selected, RichText::new(label).color(if c.auto { theme::TEXT_SECONDARY } else { theme::TEXT_BODY })),
                );
                let hover = if c.bone.is_empty() { "Follows the object".to_owned() } else { format!("Follows {}", c.bone) };
                if row.on_hover_text(hover).clicked() {
                    self.runtime.selected = if selected { None } else { Some(k) };
                }
                if widgets::icon_button(ui, Icon::Trash, "Remove", false).clicked() {
                    remove = Some(k);
                }
            });
        }
        if let Some(k) = remove {
            self.colliders.remove(k);
            self.runtime.selected = None;
            changed = true;
        }
        if widgets::dashed_button(ui, Some(Icon::Plus), "Add collider").clicked() {
            let bone = self.runtime.selected.and_then(|s| self.colliders.get(s)).map(|c| c.bone.clone()).unwrap_or_default();
            self.colliders.push(ClothCollider { name: format!("Collider {}", self.colliders.len() + 1), bone, ..Default::default() });
            self.runtime.selected = Some(self.colliders.len() - 1);
            changed = true;
        }

        if let Some(c) = self.runtime.selected.and_then(|s| self.colliders.get_mut(s)) {
            widgets::section_gap(ui);
            widgets::eyebrow(ui, &format!("Edit {}", c.name));
            widgets::property_row(ui, "Name", |ui| changed |= ui.text_edit_singleline(&mut c.name).changed());
            widgets::property_row(ui, "Bone", |ui| {
                if let Some(b) = bone_picker(ui, "zestengine-cloth-collider-bone", &c.bone, &bones, Some("(object)")) {
                    c.bone = b;
                    changed = true;
                }
            });
            combo(ui, "zestengine-cloth-collider-shape", "Shape", &mut c.shape, &ClothShape::ALL, ClothShape::label, &mut changed);
            match c.shape {
                ClothShape::Capsule => {
                    vec3(ui, "Start", &mut c.center, &mut changed);
                    drag(ui, "Start radius", &mut c.radius, 0.002, 0.0..=10.0, &mut changed);
                    vec3(ui, "End", &mut c.end, &mut changed);
                    drag(ui, "End radius", &mut c.end_radius, 0.002, 0.0..=10.0, &mut changed);
                }
                ClothShape::Sphere => {
                    vec3(ui, "Center", &mut c.center, &mut changed);
                    drag(ui, "Radius", &mut c.radius, 0.002, 0.0..=10.0, &mut changed);
                }
                ClothShape::Plane => {
                    vec3(ui, "Point", &mut c.center, &mut changed);
                    vec3(ui, "Normal", &mut c.normal, &mut changed);
                }
            }
            if c.auto {
                note(ui, "Auto-fitted: fitting again replaces it. Untick to keep your edits.");
                check(ui, "Auto-fitted", &mut c.auto, &mut changed);
            }
            note(ui, "Offsets are in the bone's frame. Drag the handles in the viewport.");
        }

        widgets::section_gap(ui);
        widgets::eyebrow(ui, "Auto-fit capsules");
        combo(ui, "zestengine-cloth-fit-preset", "Bones", &mut self.fit.preset, &ClothFitPreset::ALL, ClothFitPreset::label, &mut changed);
        if self.fit.preset == ClothFitPreset::Selected {
            bone_list(ui, "zestengine-cloth-fit-bones", &mut self.fit.bones, &bones, "Add bone", &mut changed);
        }
        drag(ui, "Coverage", &mut self.fit.coverage, 0.005, 0.5..=1.0, &mut changed);
        drag(ui, "Inset", &mut self.fit.inset, 0.001, -0.2..=0.2, &mut changed);
        note(ui, "Coverage: share of each bone's vertices inside its capsule. Inset grows (+) or shrinks (-) every radius.");
        ui.horizontal(|ui| {
            if widgets::primary_button(ui, Some(Icon::Target), "Fit capsules").clicked() {
                self.runtime.fit_requested = true;
                changed = true;
            }
            if self.colliders.iter().any(|c| c.auto) && widgets::secondary_button(ui, Some(Icon::Trash), "Clear fitted").clicked() {
                self.colliders.retain(|c| !c.auto);
                self.runtime.selected = None;
                changed = true;
            }
        });
        if !self.runtime.fit_report.is_empty() {
            note(ui, &self.runtime.fit_report);
        }
        changed
    }
}

/// Wire outline of a resolved collider shape.
fn shape_segments(shape: &ClothShapeWorld) -> Vec<(Vec3, Vec3)> {
    use std::f32::consts::{PI, TAU};
    let mut s = Vec::new();
    match *shape {
        ClothShapeWorld::Sphere { c, r } => {
            circle(&mut s, c, Vec3::X, Vec3::Y, r, 0.0, TAU, 32);
            circle(&mut s, c, Vec3::Y, Vec3::Z, r, 0.0, TAU, 32);
            circle(&mut s, c, Vec3::X, Vec3::Z, r, 0.0, TAU, 32);
        }
        ClothShapeWorld::Capsule { a, b, ra, rb } => {
            let axis = (b - a).normalize_or(Vec3::Y);
            let (u, v) = axis.any_orthonormal_pair();
            circle(&mut s, a, u, v, ra, 0.0, TAU, 32);
            circle(&mut s, b, u, v, rb, 0.0, TAU, 32);
            for d in [u, -u, v, -v] {
                s.push((a + d * ra, b + d * rb));
            }
            for d in [u, v] {
                circle(&mut s, a, d, -axis, ra, 0.0, PI, 16);
                circle(&mut s, a, -d, -axis, ra, 0.0, PI * 0.5, 8);
                circle(&mut s, b, d, axis, rb, 0.0, PI, 16);
            }
        }
        ClothShapeWorld::Plane { p, n } => {
            let (u, v) = n.any_orthonormal_pair();
            let h = 0.25;
            let c = [p + (u + v) * h, p + (u - v) * h, p - (u + v) * h, p - (u - v) * h];
            for k in 0..4 {
                s.push((c[k], c[(k + 1) % 4]));
            }
            s.push((c[0], c[2]));
            s.push((c[1], c[3]));
            s.push((p, p + n * 0.15));
        }
    }
    s
}

#[allow(clippy::too_many_arguments)]
fn circle(out: &mut Vec<(Vec3, Vec3)>, c: Vec3, a: Vec3, b: Vec3, r: f32, from: f32, to: f32, seg: usize) {
    let at = |t: f32| c + (a * t.cos() + b * t.sin()) * r;
    let mut prev = at(from);
    for k in 1..=seg {
        let q = at(from + (to - from) * k as f32 / seg as f32);
        out.push((prev, q));
        prev = q;
    }
}

/// The collider being edited and its bone frame.
fn selected_frame(cc: &ClothColliders) -> Option<(usize, Mat4)> {
    let s = cc.runtime.selected?;
    let c = cc.colliders.get(s)?;
    Some((s, cc.runtime.frame(&c.bone)?))
}

// Gizmo list layout: 0 = other colliders, 1 = the selected one, then for the
// selected collider 2 = start handle, 3 = start radius, 4 = end handle,
// 5 = end radius (capsules only for 4 and 5; planes stop at 2).
impl Gizmo for ClothColliders {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let mut others = Vec::new();
        let mut selected = Vec::new();
        for (k, c) in self.colliders.iter().enumerate() {
            let Some(frame) = self.runtime.frame(&c.bone) else { continue };
            if !c.enabled && self.runtime.selected != Some(k) {
                continue;
            }
            let segs = shape_segments(&c.shape_in(frame, 1.0));
            if self.runtime.selected == Some(k) { selected = segs } else { others.extend(segs) }
        }
        let mut g = vec![
            GizmoSpec::Lines { segments: others, color: COLLIDER, world: false },
            GizmoSpec::Lines { segments: selected, color: COLLIDER_SELECTED, world: false },
        ];
        if let Some((s, frame)) = selected_frame(self) {
            let c = &self.colliders[s];
            let start = frame.transform_point3(Vec3::from(c.center));
            let anchor = frame.w_axis.truncate();
            g.push(GizmoSpec::Handle { position: start, anchor, color: COLLIDER_SELECTED });
            if c.shape != ClothShape::Plane {
                g.push(GizmoSpec::Range { center: start, radius: c.radius, color: COLLIDER_SELECTED });
            }
            if c.shape == ClothShape::Capsule {
                let end = frame.transform_point3(Vec3::from(c.end));
                g.push(GizmoSpec::Handle { position: end, anchor: start, color: COLLIDER_SELECTED });
                g.push(GizmoSpec::Range { center: end, radius: c.end_radius, color: COLLIDER_SELECTED });
            }
        }
        g
    }

    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        let Some((s, frame)) = selected_frame(self) else { return };
        let inv = frame.inverse();
        let c = &mut self.colliders[s];
        match edit {
            GizmoEdit::Handle { index: 2, position } => c.center = inv.transform_point3(position).to_array(),
            GizmoEdit::Handle { index: 4, position } => c.end = inv.transform_point3(position).to_array(),
            GizmoEdit::Range { index: 3, radius } => c.radius = radius.max(0.0),
            GizmoEdit::Range { index: 5, radius } => c.end_radius = radius.max(0.0),
            _ => return,
        }
        c.auto = false;
    }
}

// --------------------------------------------------------------- bone cloth

impl Inspect for BoneCloth {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        let bones = self.runtime.bones.clone();
        if bones.is_empty() {
            widgets::info_note(ui, "Needs a skinned model on this object. Pick the joints each chain hangs from.");
        }
        widgets::eyebrow(ui, "Chains");
        bone_list(ui, "zestengine-cloth-roots", &mut self.roots, &bones, "Add root", &mut changed);
        note(ui, "Every joint below a root simulates. A root with several children (a skirt ring) links them when connected.");
        combo(ui, "zestengine-cloth-connection", "Connection", &mut self.connection, &ClothConnection::ALL, ClothConnection::label, &mut changed);
        if self.connection != ClothConnection::Chains {
            drag(ui, "Link stiffness", &mut self.link_stiffness, 0.005, 0.0..=1.0, &mut changed);
        }
        if let Some(sim) = &self.runtime.sim {
            note(ui, &format!("{} particles, {} links", sim.particles.len(), sim.links.len()));
        }

        widgets::eyebrow(ui, "Motion");
        vec3(ui, "Gravity", &mut self.gravity, &mut changed);
        drag(ui, "Damping", &mut self.damping, 0.002, 0.0..=1.0, &mut changed);
        pair(ui, "Stiffness", &mut self.stiffness, 0.002, 0.0..=1.0, &mut changed);
        drag(ui, "Angle limit (deg)", &mut self.angle_limit, 0.5, 0.0..=180.0, &mut changed);
        drag(ui, "Blend", &mut self.blend, 0.005, 0.0..=1.0, &mut changed);
        drag(ui, "Wind", &mut self.wind, 0.01, 0.0..=5.0, &mut changed);

        widgets::eyebrow(ui, "Collision");
        check(ui, "Collide", &mut self.collide, &mut changed);
        if self.collide {
            pair(ui, "Radius", &mut self.radius, 0.001, 0.0..=1.0, &mut changed);
            drag(ui, "Friction", &mut self.friction, 0.005, 0.0..=1.0, &mut changed);
            check(ui, "Edge collision", &mut self.edge_collision, &mut changed);
            let names = self.runtime.colliders.clone();
            let mut all = self.colliders.is_empty();
            widgets::property_row(ui, "All colliders", |ui| {
                if ui.checkbox(&mut all, "").changed() {
                    self.colliders = if all { Vec::new() } else { names.clone() };
                    changed = true;
                }
            });
            if !all {
                for n in &names {
                    let mut on = self.colliders.contains(n);
                    widgets::property_row(ui, n, |ui| {
                        if ui.checkbox(&mut on, "").changed() {
                            if on { self.colliders.push(n.clone()) } else { self.colliders.retain(|c| c != n) }
                            changed = true;
                        }
                    });
                }
                if names.is_empty() {
                    note(ui, "Add a Cloth Colliders component to this object");
                }
            }
            let mut remove = None;
            for (k, r) in self.collider_objects.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    changed |= ctx.object_ref(ui, "Also collide", r);
                    if widgets::icon_button(ui, Icon::Trash, "Remove", false).clicked() {
                        remove = Some(k);
                    }
                });
            }
            if let Some(k) = remove {
                self.collider_objects.remove(k);
                changed = true;
            }
            if widgets::dashed_button(ui, Some(Icon::Plus), "Collide with another object").clicked() {
                self.collider_objects.push(ObjectRef::NONE);
                changed = true;
            }
        }

        widgets::eyebrow(ui, "Inertia");
        drag(ui, "World inertia", &mut self.world_inertia, 0.005, 0.0..=1.0, &mut changed);
        drag(ui, "Max speed (m/s)", &mut self.max_speed, 0.05, 0.0..=100.0, &mut changed);
        drag(ui, "Teleport distance", &mut self.teleport_distance, 0.01, 0.0..=100.0, &mut changed);
        drag(ui, "Teleport angle (deg)", &mut self.teleport_angle, 0.5, 0.0..=180.0, &mut changed);
        combo(ui, "zestengine-cloth-teleport", "On teleport", &mut self.teleport, &ClothTeleport::ALL, ClothTeleport::label, &mut changed);
        note(ui, "Inertia 1 = fully physical, 0 = moves rigidly with the object. Faster movement than Max speed is carried rigidly.");

        widgets::eyebrow(ui, "Simulation");
        drag(ui, "Frequency (Hz)", &mut self.frequency, 1.0, 10.0..=480.0, &mut changed);
        widgets::property_row(ui, "Iterations", |ui| changed |= ui.add(DragValue::new(&mut self.iterations).range(1..=16)).changed());
        cull_row(ui, &mut self.cull_distance, self.runtime.culled, &mut changed);
        check(ui, "End tail", &mut self.end_tail, &mut changed);
        if widgets::secondary_button(ui, Some(Icon::Reset), "Reset to pose").clicked() {
            self.runtime.reset_requested = true;
        }
        changed
    }
}

impl Gizmo for BoneCloth {
    /// The simulated chains, links between chains, and each particle's
    /// collision radius (world space, live while the cloth simulates).
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let Some(sim) = &self.runtime.sim else { return Vec::new() };
        let mut chain = Vec::new();
        let mut fixed = Vec::new();
        let mut links = Vec::new();
        for p in &sim.particles {
            if let Some(pi) = p.parent {
                chain.push((sim.particles[pi].pos, p.pos));
            }
            let out = if p.fixed { &mut fixed } else { &mut chain };
            let r = if p.fixed { 0.01 } else { p.radius.max(0.004) };
            circle(out, p.pos, Vec3::X, Vec3::Y, r, 0.0, std::f32::consts::TAU, 12);
            circle(out, p.pos, Vec3::Y, Vec3::Z, r, 0.0, std::f32::consts::TAU, 12);
            circle(out, p.pos, Vec3::X, Vec3::Z, r, 0.0, std::f32::consts::TAU, 12);
        }
        for &(a, b) in &sim.links {
            links.push((sim.particles[a].pos, sim.particles[b].pos));
        }
        vec![
            GizmoSpec::Lines { segments: links, color: LINK, world: true },
            GizmoSpec::Lines { segments: chain, color: CHAIN, world: true },
            GizmoSpec::Lines { segments: fixed, color: CHAIN_FIXED, world: true },
        ]
    }
}

const WIND: egui::Color32 = egui::Color32::from_rgb(170, 220, 255);

impl Inspect for ClothWind {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        combo(ui, "zestengine-wind-shape", "Shape", &mut self.shape, &WindShape::ALL, WindShape::label, &mut changed);
        drag(ui, "Strength (m/s)", &mut self.strength, 0.05, 0.0..=100.0, &mut changed);
        drag(ui, "Turbulence", &mut self.turbulence, 0.005, 0.0..=1.0, &mut changed);
        drag(ui, "Gust frequency (Hz)", &mut self.frequency, 0.01, 0.0..=10.0, &mut changed);
        match self.shape {
            WindShape::Sphere => drag(ui, "Radius", &mut self.radius, 0.05, 0.01..=1000.0, &mut changed),
            WindShape::Box => vec3(ui, "Size", &mut self.size, &mut changed),
            WindShape::Global => {}
        }
        note(ui, "Blows along the object's forward (-Z). Bone Cloth's Wind scales how much each cloth feels it.");
        changed
    }
}

impl Gizmo for ClothWind {
    /// An arrow along the wind, plus its sphere or box.
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let len = 0.5 + self.strength.min(20.0) * 0.1;
        let tip = Vec3::NEG_Z * len;
        let mut s = vec![(Vec3::ZERO, tip), (tip, tip + Vec3::new(0.12, 0.0, 0.2)), (tip, tip + Vec3::new(-0.12, 0.0, 0.2))];
        for k in [-1.0f32, 1.0] {
            let o = Vec3::X * 0.2 * k;
            s.push((o, o + tip * 0.7));
        }
        let mut g = vec![GizmoSpec::Lines { segments: s, color: WIND, world: false }];
        match self.shape {
            WindShape::Sphere => g.push(GizmoSpec::Range { center: Vec3::ZERO, radius: self.radius, color: WIND }),
            WindShape::Box => g.push(GizmoSpec::Box {
                center: Vec3::ZERO,
                half_extents: Vec3::from(self.size) * 0.5,
                object_anchored: false,
                blend: 0.0,
                color: WIND,
            }),
            WindShape::Global => {}
        }
        g
    }

    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        match edit {
            GizmoEdit::Range { radius, .. } => self.radius = radius.max(0.01),
            GizmoEdit::Box { half_extents, .. } => self.size = (half_extents * 2.0).to_array(),
            _ => {}
        }
    }
}

impl Inspect for MeshCloth {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        widgets::property_row(ui, "Submesh", |ui| changed |= ui.add(DragValue::new(&mut self.slot).range(0..=63)).changed());
        note(ui, "Which of the object's submeshes is the garment. It must be skinned (rigged); a single bone is enough.");
        if let Some(sim) = &self.runtime.sim {
            note(ui, &format!("{} particles, {} stretch links", sim.particles.len(), sim.edges.len()));
        }

        widgets::eyebrow(ui, "Pinning");
        combo(ui, "zestengine-meshcloth-pin", "Pin by", &mut self.pin, &PinSource::ALL, PinSource::label, &mut changed);
        let bones = self.runtime.bones.clone();
        match self.pin {
            PinSource::VertexColor => {
                widgets::property_row(ui, "Channel", |ui| {
                    let opts = [(0u32, "R"), (1, "G"), (2, "B"), (3, "A")];
                    changed |= widgets::segmented(ui, &mut self.pin_channel, &opts).changed();
                });
                note(ui, "Paint 1 where the cloth is attached to the body, 0 where it hangs free.");
            }
            PinSource::Bones => bone_list(ui, "zestengine-meshcloth-bones", &mut self.pin_bones, &bones, "Add bone", &mut changed),
            PinSource::Top => drag(ui, "Pinned share", &mut self.pin_height, 0.005, 0.0..=1.0, &mut changed),
            PinSource::Painted => {
                let pins = self.paint.pins.as_slice();
                if !pins.is_empty() {
                    let pinned = pins.iter().filter(|&&b| b >= 128).count();
                    note(ui, &format!("{pinned} of {} vertices pinned", pins.len()));
                }
                let mut limit = false;
                drag(ui, "Max distance (m)", &mut self.max_distance, 0.002, 0.0..=10.0, &mut limit);
                if limit {
                    self.runtime.pins_dirty = true;
                    changed = true;
                }
                note(ui, if self.paint.max_distance.is_empty() {
                    "Paint the Max distance channel to limit how far vertices stray from their animated position."
                } else {
                    "A painted max distance of 1 lets a vertex stray this far from its animated position."
                });
            }
        }
        if !self.runtime.paint_note.is_empty() {
            ui.label(RichText::new(&self.runtime.paint_note).size(theme::SIZE_META).color(theme::STATUS_WARN_FG));
        }
        let egui_ctx = ui.ctx().clone();
        let mut tool = ClothPaintTool::get(&egui_ctx);
        let before = tool;
        if self.runtime.paint_mode {
            changed |= paint_panel(ui, self, &mut tool);
        } else {
            if widgets::secondary_button(ui, Some(Icon::Target), "Paint Pins")
                .on_hover_text("Paint pin weights and max distances on the cloth in the Scene view (P)")
                .clicked()
            {
                tool.active = true;
            }
            if self.pin != PinSource::Painted {
                note(ui, "Painting starts from the current pins and switches Pin by to Painted.");
            }
        }
        if tool != before {
            tool.set(&egui_ctx);
        }
        if widgets::secondary_button(ui, Some(Icon::Reset), "Rebuild").clicked() {
            self.runtime.reset_requested = true;
        }

        widgets::eyebrow(ui, "Shape");
        drag(ui, "Particle spacing (m)", &mut self.particle_spacing, 0.001, 0.0..=1.0, &mut changed);
        drag(ui, "Stiffness", &mut self.stiffness, 0.005, 0.0..=1.0, &mut changed);
        drag(ui, "Bending", &mut self.bending, 0.005, 0.0..=1.0, &mut changed);
        drag(ui, "Keep shape", &mut self.shape_stiffness, 0.001, 0.0..=1.0, &mut changed);
        drag(ui, "Tether slack", &mut self.tether_slack, 0.005, 0.0..=2.0, &mut changed);
        vec3(ui, "Gravity", &mut self.gravity, &mut changed);
        drag(ui, "Damping", &mut self.damping, 0.002, 0.0..=1.0, &mut changed);
        drag(ui, "Wind", &mut self.wind, 0.01, 0.0..=5.0, &mut changed);

        widgets::eyebrow(ui, "Collision");
        check(ui, "Collide", &mut self.collide, &mut changed);
        drag(ui, "Radius", &mut self.radius, 0.001, 0.0..=0.5, &mut changed);
        drag(ui, "Friction", &mut self.friction, 0.005, 0.0..=1.0, &mut changed);
        check(ui, "Self collision", &mut self.self_collision, &mut changed);
        if self.self_collision {
            drag(ui, "Thickness", &mut self.thickness, 0.001, 0.0..=0.5, &mut changed);
        }
        let names = self.runtime.collider_names.clone();
        let mut all = self.colliders.is_empty();
        widgets::property_row(ui, "All colliders", |ui| {
            if ui.checkbox(&mut all, "").changed() {
                self.colliders = if all { Vec::new() } else { names.clone() };
                changed = true;
            }
        });
        if !all {
            for n in &names {
                let mut on = self.colliders.contains(n);
                widgets::property_row(ui, n, |ui| {
                    if ui.checkbox(&mut on, "").changed() {
                        if on { self.colliders.push(n.clone()) } else { self.colliders.retain(|c| c != n) }
                        changed = true;
                    }
                });
            }
        }
        let mut remove = None;
        for (k, r) in self.collider_objects.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                changed |= ctx.object_ref(ui, "Also collide", r);
                if widgets::icon_button(ui, Icon::Trash, "Remove", false).clicked() {
                    remove = Some(k);
                }
            });
        }
        if let Some(k) = remove {
            self.collider_objects.remove(k);
            changed = true;
        }
        if widgets::dashed_button(ui, Some(Icon::Plus), "Collide with another object").clicked() {
            self.collider_objects.push(ObjectRef::NONE);
            changed = true;
        }

        widgets::eyebrow(ui, "Inertia");
        drag(ui, "World inertia", &mut self.world_inertia, 0.005, 0.0..=1.0, &mut changed);
        drag(ui, "Max speed (m/s)", &mut self.max_speed, 0.05, 0.0..=100.0, &mut changed);
        drag(ui, "Teleport distance", &mut self.teleport_distance, 0.01, 0.0..=100.0, &mut changed);
        drag(ui, "Teleport angle (deg)", &mut self.teleport_angle, 0.5, 0.0..=180.0, &mut changed);
        combo(ui, "zestengine-meshcloth-teleport", "On teleport", &mut self.teleport, &ClothTeleport::ALL, ClothTeleport::label, &mut changed);

        widgets::eyebrow(ui, "Simulation");
        drag(ui, "Frequency (Hz)", &mut self.frequency, 1.0, 10.0..=480.0, &mut changed);
        widgets::property_row(ui, "Iterations", |ui| changed |= ui.add(DragValue::new(&mut self.iterations).range(1..=32)).changed());
        cull_row(ui, &mut self.cull_distance, self.runtime.culled, &mut changed);
        changed
    }
}

impl Gizmo for MeshCloth {
    /// The simulated particles: pinned ones red, free ones as small crosses
    /// (hidden while painting, which draws its own overlay).
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let Some(sim) = self.runtime.sim.as_ref().filter(|_| !self.runtime.paint_mode) else { return Vec::new() };
        let (mut free, mut pinned) = (Vec::new(), Vec::new());
        let step = sim.particles.len().div_ceil(4000).max(1);
        for p in sim.particles.iter().step_by(step) {
            let r = 0.006;
            let out = if p.pin >= 0.5 { &mut pinned } else { &mut free };
            out.push((p.pos - Vec3::X * r, p.pos + Vec3::X * r));
            out.push((p.pos - Vec3::Y * r, p.pos + Vec3::Y * r));
            out.push((p.pos - Vec3::Z * r, p.pos + Vec3::Z * r));
        }
        vec![
            GizmoSpec::Lines { segments: free, color: CHAIN, world: true },
            GizmoSpec::Lines { segments: pinned, color: CHAIN_FIXED, world: true },
        ]
    }
}

