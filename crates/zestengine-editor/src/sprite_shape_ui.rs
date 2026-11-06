//! Inspectors for Sprite Shapes (2D-8): the `SpriteShapeRenderer` component
//! (spline settings, point list and the selected point, collider, presets) and
//! the `.spriteshape` profile editor (angle ranges with a circular dial,
//! corners, caps, fill). The Scene-view spline tool is `engine/src/shape_edit.rs`;
//! it shares [`ShapeEditState`] through egui memory with the Edit Spline button
//! here, the same way river path editing does. GUI not yet seen on screen.
//!
//! The dial maths ([`dial_point`], [`range_color`], [`hit_angle`]) is plain
//! functions so it is unit tested; the egui code only lays it out.

use egui::{Color32, DragValue, Pos2, RichText, Sense, Ui, Vec2};
use zestengine_core::sprite_shape::presets::PresetData;
use zestengine_core::{
    AngleRange, EdgeFit, ShapePoint, ShapeTangentMode, SpriteShapeProfile, SpriteShapeRenderer, Tessellation, select_range,
};
use zestengine_core::{ObjectId, sprite_shape};

use crate::components::{EditorComponents, Gizmo, Inspect, InspectCtx, asset_chip};
use crate::icons::Icon;
use crate::sprite2d_ui::{combo, layer_row, order_row, sprite_slot};
use crate::{theme, widgets};

pub(crate) fn register(e: &mut EditorComponents) {
    e.register::<SpriteShapeRenderer>();
}

/// Scene-view spline editing state, shared through egui memory: which object
/// it belongs to, whether the tool is on, the selected point and the snap grid.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ShapeEditState {
    pub object: Option<ObjectId>,
    pub active: bool,
    pub selected: Option<usize>,
    /// Grid step in local units; 0 = no snapping.
    pub snap: f32,
    /// The Scene view picked `selected`; the inspector scrolls it into view.
    pub reveal: bool,
    /// Draw the tessellated polyline over the shape.
    pub show_tessellation: bool,
}

impl Default for ShapeEditState {
    fn default() -> Self {
        Self { object: None, active: false, selected: None, snap: 0.0, reveal: false, show_tessellation: false }
    }
}

impl ShapeEditState {
    fn id() -> egui::Id {
        egui::Id::new("zestengine-shape-edit")
    }
    pub fn get(ctx: &egui::Context) -> Self {
        ctx.data(|d| d.get_temp(Self::id())).unwrap_or_default()
    }
    pub fn set(self, ctx: &egui::Context) {
        ctx.data_mut(|d| d.insert_temp(Self::id(), self));
    }
}

pub const SHAPE_HINT: &str = "Click add   Shift+Click insert on the line   Drag points and handles   T mode   C corner   Del remove   Esc done";

fn note(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(text).size(theme::SIZE_META).color(theme::TEXT_DIM));
}

fn drag2(ui: &mut Ui, label: &str, v: &mut [f32; 2], speed: f32, changed: &mut bool) {
    widgets::property_row(ui, label, |ui| {
        ui.horizontal(|ui| {
            for x in v.iter_mut() {
                *changed |= ui.add(DragValue::new(x).speed(speed).max_decimals(3)).changed();
            }
        });
    });
}

fn check_row(ui: &mut Ui, label: &str, v: &mut bool, tip: &str, changed: &mut bool) {
    widgets::property_row(ui, label, |ui| *changed |= ui.checkbox(v, "").on_hover_text(tip).changed());
}

fn num_row(ui: &mut Ui, label: &str, v: &mut f32, speed: f32, range: std::ops::RangeInclusive<f32>, tip: &str, changed: &mut bool) {
    widgets::property_row(ui, label, |ui| *changed |= ui.add(DragValue::new(v).speed(speed).range(range).max_decimals(4)).on_hover_text(tip).changed());
}

fn int_row(ui: &mut Ui, label: &str, v: &mut u32, range: std::ops::RangeInclusive<u32>, tip: &str, changed: &mut bool) {
    widgets::property_row(ui, label, |ui| *changed |= ui.add(DragValue::new(v).range(range)).on_hover_text(tip).changed());
}

fn tangent_mode_label(m: ShapeTangentMode) -> &'static str {
    m.label()
}

/// One-line description of a control point for the list.
pub fn point_summary(k: usize, p: &ShapePoint) -> String {
    let mode = match p.mode {
        ShapeTangentMode::Linear => "L",
        ShapeTangentMode::Continuous => "C",
        ShapeTangentMode::Broken => "B",
    };
    format!("{}  ({:.2}, {:.2})  {mode}{}", k + 1, p.position[0], p.position[1], if p.corner { "" } else { "  no corner" })
}

impl Inspect for SpriteShapeRenderer {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = ctx.file_field(ui, "Profile", &mut self.profile, "none", &[sprite_shape::SPRITE_SHAPE_EXTENSION]);
        if self.profile.is_empty() {
            note(ui, "Without a profile the shape draws nothing. Press + to make one (angle ranges, corners, fill).");
        }
        check_row(ui, "Closed", &mut self.closed, "A closed spline is filled and walked clockwise", &mut changed);
        if self.closed {
            check_row(ui, "Fill", &mut self.fill, "Draw the profile's fill texture / sprite inside the outline", &mut changed);
        }
        combo(ui, "Edge Fit", "shape-fit", &mut self.fit, &EdgeFit::ALL, EdgeFit::label, &mut changed);
        check_row(ui, "Flip Edge", &mut self.flip_edge, "Walk the spline the other way: the edge sprites go to its other side", &mut changed);
        num_row(ui, "Border Pivot", &mut self.border_pivot, 0.01, -1.0..=1.0, "Shifts the edge sprites along the normal as a fraction of their height", &mut changed);
        num_row(ui, "Corner Angle", &mut self.corner_angle_threshold, 0.5, 0.0..=180.0, "Points turning more than this (degrees) can use a corner sprite", &mut changed);
        widgets::property_row(ui, "Color", |ui| changed |= ui.color_edit_button_rgba_unmultiplied(&mut self.color).changed());
        layer_row(ui, "Sorting Layer", "shape-sr", &mut self.sorting_layer, &mut changed);
        order_row(ui, "Order in Layer", &mut self.order_in_layer, &mut changed);
        combo(ui, "Mask Interaction", "shape-mask", &mut self.mask_interaction, &zestengine_core::MaskInteraction2D::ALL, zestengine_core::MaskInteraction2D::label, &mut changed);

        ui.add_space(4.0);
        widgets::eyebrow(ui, "Tessellation");
        combo(ui, "Mode", "shape-tess", &mut self.tessellation, &Tessellation::ALL, Tessellation::label, &mut changed);
        match self.tessellation {
            Tessellation::Adaptive => num_row(ui, "Tolerance", &mut self.tolerance, 0.002, 0.0005..=5.0, "Largest gap between the curve and its polyline (local units)", &mut changed),
            Tessellation::Fixed => int_row(ui, "Detail", &mut self.spline_detail, 1..=64, "Pieces per curved segment (16 = high, 8 = mid, 4 = low)", &mut changed),
        }
        int_row(ui, "Max Points", &mut self.max_points, 4..=60_000, "The tolerance / detail is relaxed to stay under this", &mut changed);

        ui.add_space(4.0);
        widgets::eyebrow(ui, "Collider");
        check_row(ui, "Auto Update", &mut self.auto_update_collider, "Write the outline into this object's Polygon (closed) or Edge (open) Collider 2D when the shape changes", &mut changed);
        if self.auto_update_collider {
            int_row(ui, "Detail", &mut self.collider_detail, 1..=64, "Pieces per curved segment of the collider outline", &mut changed);
            num_row(ui, "Optimize", &mut self.collider_optimize, 0.002, 0.0..=1.0, "Douglas-Peucker tolerance; 0 keeps every point", &mut changed);
            if self.closed {
                num_row(ui, "Offset", &mut self.collider_offset, 0.01, -5.0..=5.0, "Grows (positive) or shrinks the polygon outline", &mut changed);
            }
            note(ui, if self.closed { "Needs a Polygon Collider 2D on this object." } else { "Needs an Edge Collider 2D on this object." });
        }

        ui.add_space(4.0);
        widgets::eyebrow(ui, "Spline");
        changed |= spline_ui(ui, self);
        changed
    }
}
impl Gizmo for SpriteShapeRenderer {}

/// The Edit Spline toggle, presets, point list and the selected point's fields.
fn spline_ui(ui: &mut Ui, s: &mut SpriteShapeRenderer) -> bool {
    let mut changed = false;
    let ctx = ui.ctx().clone();
    let mut edit = ShapeEditState::get(&ctx);
    let before = edit;
    if edit.active {
        if widgets::primary_button(ui, Some(Icon::Check), "Finish Editing Spline").clicked() {
            edit.active = false;
        }
        note(ui, SHAPE_HINT);
    } else if widgets::secondary_button(ui, Some(Icon::Target), "Edit Spline").on_hover_text("Edit the spline in the Scene view").clicked() {
        edit.active = true;
    }
    widgets::property_row(ui, "Snap", |ui| {
        ui.add(DragValue::new(&mut edit.snap).speed(0.05).range(0.0..=10.0).max_decimals(3)).on_hover_text("Grid step for dragging and adding points; 0 = off");
    });
    widgets::property_row(ui, "Show Tessellation", |ui| {
        ui.checkbox(&mut edit.show_tessellation, "").on_hover_text("Draw the line pieces the spline is cut into");
    });
    ui.menu_button("Apply Preset", |ui| {
        widgets::menu_frame(ui);
        for p in sprite_shape::ShapePreset::ALL {
            if ui.button(p.label()).on_hover_text(p.hint()).clicked() {
                sprite_shape::apply_preset(s, &sprite_shape::preset_shape(p));
                changed = true;
                edit.selected = None;
                ui.close();
            }
        }
    });
    note(ui, "A preset replaces the points, closed flag and fit; the profile asset is separate (open it to apply a preset to its ranges).");

    let n = s.points.len();
    edit.selected = edit.selected.filter(|&k| k < n);
    ui.label(RichText::new(format!("{n} control points")).size(theme::SIZE_META).color(theme::TEXT_DIM));
    let reveal = std::mem::take(&mut edit.reveal);
    egui::ScrollArea::vertical().max_height(150.0).id_salt("shape-points").show(ui, |ui| {
        for (k, p) in s.points.iter().enumerate() {
            let selected = edit.selected == Some(k);
            let r = ui.selectable_label(selected, point_summary(k, p));
            if r.clicked() {
                edit.selected = Some(k);
            }
            if selected && reveal {
                r.scroll_to_me(Some(egui::Align::Center));
            }
        }
    });
    if let Some(k) = edit.selected {
        let along = {
            let prev = s.points[if k == 0 { if s.closed { n - 1 } else { 0 } } else { k - 1 }].position;
            let next = s.points[if k + 1 < n { k + 1 } else if s.closed { 0 } else { k }].position;
            [next[0] - prev[0], next[1] - prev[1]]
        };
        let mut remove = false;
        ui.add_space(2.0);
        egui::Frame::new().fill(theme::accent_faint()).corner_radius(4).inner_margin(egui::Margin::symmetric(6, 4)).show(ui, |ui| {
            ui.label(RichText::new(format!("Point {}", k + 1)).strong().color(theme::accent()));
            let p = &mut s.points[k];
            drag2(ui, "Position", &mut p.position, 0.02, &mut changed);
            let mut mode = p.mode;
            widgets::property_row(ui, "Tangent Mode", |ui| {
                egui::ComboBox::from_id_salt("shape-point-mode").selected_text(tangent_mode_label(mode)).show_ui(ui, |ui| {
                    for m in ShapeTangentMode::ALL {
                        ui.selectable_value(&mut mode, m, tangent_mode_label(m));
                    }
                });
            });
            if mode != p.mode {
                let reach = 0.5;
                p.set_mode(mode, along, reach);
                changed = true;
            }
            if p.mode != ShapeTangentMode::Linear {
                let mut t_in = p.tangent_in;
                let mut c_in = false;
                drag2(ui, "Tangent In", &mut t_in, 0.02, &mut c_in);
                if c_in {
                    p.set_tangent_in(t_in);
                    changed = true;
                }
                let mut t_out = p.tangent_out;
                let mut c_out = false;
                drag2(ui, "Tangent Out", &mut t_out, 0.02, &mut c_out);
                if c_out {
                    p.set_tangent_out(t_out);
                    changed = true;
                }
            }
            num_row(ui, "Height", &mut p.height, 0.01, 0.0..=20.0, "Scales the edge sprites' thickness at this point", &mut changed);
            check_row(ui, "Corner", &mut p.corner, "May use a corner sprite when it turns past the threshold", &mut changed);
            int_row(ui, "Sprite Variant", &mut p.sprite_index, 0..=64, "Which sprite of the angle range the segment after this point uses", &mut changed);
            if widgets::secondary_button(ui, Some(Icon::Trash), "Remove Point").clicked() {
                remove = true;
            }
        });
        if remove && n > 2 {
            s.points.remove(k);
            edit.selected = Some(k.saturating_sub(1).min(s.points.len() - 1));
            changed = true;
        } else if remove {
            note(ui, "A spline needs at least two points.");
        }
    }
    if edit != before {
        edit.set(&ctx);
    }
    if s.closed {
        changed |= holes_ui(ui, s);
    }
    changed
}

/// Where a new hole goes: the middle of the outline's bounds, a quarter of its smaller side wide.
pub fn new_hole_for(s: &SpriteShapeRenderer) -> sprite_shape::ShapeHole {
    let (mut lo, mut hi) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]);
    for p in &s.points {
        for k in 0..2 {
            lo[k] = lo[k].min(p.position[k]);
            hi[k] = hi[k].max(p.position[k]);
        }
    }
    if lo[0] > hi[0] {
        return sprite_shape::ShapeHole::ngon([0.0; 2], 0.5, 8);
    }
    let r = ((hi[0] - lo[0]).min(hi[1] - lo[1]) * 0.25).max(0.05);
    sprite_shape::ShapeHole::ngon([(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5], r, 8)
}

/// Holes: inner closed splines that cut the fill (inspector editing; positions per point).
fn holes_ui(ui: &mut Ui, s: &mut SpriteShapeRenderer) -> bool {
    let mut changed = false;
    ui.add_space(4.0);
    widgets::eyebrow(ui, "Holes");
    let mut remove: Option<usize> = None;
    for h in 0..s.holes.len() {
        let header = format!("Hole {}  ({} points)", h + 1, s.holes[h].points.len());
        egui::CollapsingHeader::new(header).id_salt(("shape-hole", h)).show(ui, |ui| {
            let hole = &mut s.holes[h];
            let mut drop_pt: Option<usize> = None;
            for (k, p) in hole.points.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{}", k + 1)).size(theme::SIZE_META).color(theme::TEXT_DIM));
                    changed |= ui.add(DragValue::new(&mut p.position[0]).speed(0.02).max_decimals(3).prefix("x ")).changed();
                    changed |= ui.add(DragValue::new(&mut p.position[1]).speed(0.02).max_decimals(3).prefix("y ")).changed();
                    if ui.small_button("x").on_hover_text("Remove this point").clicked() {
                        drop_pt = Some(k);
                    }
                });
            }
            if let Some(k) = drop_pt
                && hole.points.len() > 3
            {
                hole.points.remove(k);
                changed = true;
            }
            if widgets::secondary_button(ui, Some(Icon::Plus), "Add Point").clicked() {
                // Split the longest side.
                let n = hole.points.len();
                if n >= 2 {
                    let side = |i: usize| {
                        let (a, b) = (hole.points[i].position, hole.points[(i + 1) % n].position);
                        (b[0] - a[0]).hypot(b[1] - a[1])
                    };
                    let i = (0..n).max_by(|&a, &b| side(a).total_cmp(&side(b))).unwrap_or(0);
                    let (a, b) = (hole.points[i].position, hole.points[(i + 1) % n].position);
                    hole.points.insert(i + 1, ShapePoint::at((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5));
                    changed = true;
                }
            }
            if widgets::secondary_button(ui, Some(Icon::Trash), "Remove Hole").clicked() {
                remove = Some(h);
            }
        });
    }
    if let Some(h) = remove {
        s.holes.remove(h);
        changed = true;
    }
    if widgets::secondary_button(ui, Some(Icon::Plus), "Add Hole").on_hover_text("Cut an inner closed spline out of the fill").clicked() {
        s.holes.push(new_hole_for(s));
        changed = true;
    }
    note(ui, "A hole cuts the fill, gets edge sprites facing into it and its own collider path. Keep it inside the outline and apart from other holes.");
    changed
}

// ---------------------------------------------------------------------------
// Profile editor
// ---------------------------------------------------------------------------

/// Palette for the dial and the range headers; stable per range index.
pub fn range_color(i: usize) -> Color32 {
    const P: [Color32; 8] = [
        Color32::from_rgb(0x5B, 0xA5, 0x5B),
        Color32::from_rgb(0x4F, 0x8B, 0xD6),
        Color32::from_rgb(0xD6, 0x9A, 0x3F),
        Color32::from_rgb(0xB0, 0x6B, 0xC9),
        Color32::from_rgb(0xD0, 0x5B, 0x5B),
        Color32::from_rgb(0x4F, 0xB8, 0xB0),
        Color32::from_rgb(0xC9, 0xC2, 0x4F),
        Color32::from_rgb(0x8A, 0x8F, 0xA8),
    ];
    P[i % P.len()]
}

/// Screen position on a dial: angle 0 is up and grows counter-clockwise
/// (so 90 is left), y down.
pub fn dial_point(center: Pos2, radius: f32, deg: f32) -> Pos2 {
    let a = deg.to_radians();
    Pos2::new(center.x - a.sin() * radius, center.y - a.cos() * radius)
}

/// The angle (degrees, `[0, 360)`) of a point around the dial centre; the
/// inverse of [`dial_point`].
pub fn hit_angle(center: Pos2, p: Pos2) -> f32 {
    let d = p - center;
    (-d.x).atan2(-d.y).to_degrees().rem_euclid(360.0)
}

/// The id of the range that picks `angle`, shown next to the dial.
pub fn probe_label(profile: &SpriteShapeProfile, angle: f32) -> String {
    match select_range(&profile.angle_ranges, angle) {
        Some(i) => {
            let r = &profile.angle_ranges[i];
            format!("{:.0} deg -> range {} ({})", angle, i + 1, sprite_count(r.sprites.len()))
        }
        None => format!("{angle:.0} deg -> no range (no edge sprite)"),
    }
}

/// A draggable end of an angle range on the dial rim.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DialHandle {
    Start(usize),
    End(usize),
}

/// Radius (px) within which a press grabs a handle.
const HANDLE_GRAB: f32 = 9.0;

/// The handle nearest to `pos` within [`HANDLE_GRAB`] px of its knob on the rim of a dial of
/// radius `rim`. Later ranges win a tie (they draw on top).
pub fn pick_handle(profile: &SpriteShapeProfile, center: Pos2, rim: f32, pos: Pos2) -> Option<DialHandle> {
    let mut best: Option<(f32, DialHandle)> = None;
    for (i, r) in profile.angle_ranges.iter().enumerate() {
        for (h, deg) in [(DialHandle::Start(i), r.start), (DialHandle::End(i), r.end)] {
            let d = dial_point(center, rim, deg).distance(pos);
            if d <= HANDLE_GRAB && best.is_none_or(|(bd, _)| d <= bd) {
                best = Some((d, h));
            }
        }
    }
    best.map(|b| b.1)
}

/// Move a handle to the pointer angle `angle` (`[0, 360)`), keeping the value continuous: the
/// stored degrees change by the shortest way round, so a range written `-40..40` stays in that
/// form while its end is dragged across 360. `step` snaps the result (0 = free). True when the
/// value changed.
pub fn drag_handle(profile: &mut SpriteShapeProfile, h: DialHandle, angle: f32, step: f32) -> bool {
    let (DialHandle::Start(i) | DialHandle::End(i)) = h;
    let Some(r) = profile.angle_ranges.get_mut(i) else { return false };
    let slot = match h {
        DialHandle::Start(_) => &mut r.start,
        DialHandle::End(_) => &mut r.end,
    };
    let delta = (angle - slot.rem_euclid(360.0) + 540.0).rem_euclid(360.0) - 180.0;
    let mut v = *slot + delta;
    if step > 0.0 {
        v = (v / step).round() * step;
    }
    let v = v.clamp(-720.0, 720.0);
    let changed = (v - *slot).abs() > 1e-4;
    *slot = v;
    changed
}

fn draw_dial(ui: &mut Ui, profile: &mut SpriteShapeProfile) -> bool {
    let mut changed = false;
    let size = 150.0;
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(size), Sense::click_and_drag());
    let center = rect.center();
    let (outer, inner) = (size * 0.5 - 4.0, size * 0.5 - 26.0);
    // Handles are the ends of the ranges; a drag that starts on one moves it, any other drag probes.
    let grab_id = ui.id().with("shape-dial-grab");
    if resp.drag_started() {
        let grabbed = resp.interact_pointer_pos().and_then(|p| pick_handle(profile, center, outer, p));
        ui.data_mut(|d| d.insert_temp(grab_id, grabbed));
    }
    let grabbed: Option<DialHandle> = ui.data(|d| d.get_temp::<Option<DialHandle>>(grab_id)).flatten();
    if resp.dragged()
        && let (Some(h), Some(p)) = (grabbed, resp.interact_pointer_pos())
    {
        let step = if ui.input(|i| i.modifiers.shift) { 5.0 } else { 1.0 };
        changed |= drag_handle(profile, h, hit_angle(center, p), step);
    }
    if resp.drag_stopped() {
        ui.data_mut(|d| d.insert_temp::<Option<DialHandle>>(grab_id, None));
    }
    if resp.hovered() && grabbed.is_none() && let Some(p) = resp.hover_pos() && pick_handle(profile, center, outer, p).is_some() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
    }
    let profile: &SpriteShapeProfile = profile;
    let painter = ui.painter_at(rect);
    painter.circle_filled(center, outer + 2.0, theme::BG_PANEL);
    let mut mesh = egui::Mesh::default();
    for (a0, a1, r) in profile.resolved_arcs() {
        let color = match r {
            Some(i) => range_color(i),
            None => Color32::from_gray(0x2A),
        };
        let steps = (((a1 - a0) / 4.0).ceil() as usize).max(1);
        for s in 0..steps {
            let (x0, x1) = (a0 + (a1 - a0) * s as f32 / steps as f32, a0 + (a1 - a0) * (s + 1) as f32 / steps as f32);
            let base = mesh.vertices.len() as u32;
            for (deg, rad) in [(x0, inner), (x0, outer), (x1, outer), (x1, inner)] {
                mesh.colored_vertex(dial_point(center, rad, deg), color);
            }
            mesh.add_triangle(base, base + 1, base + 2);
            mesh.add_triangle(base, base + 2, base + 3);
        }
    }
    painter.add(egui::Shape::mesh(mesh));
    painter.circle_stroke(center, outer, egui::Stroke::new(1.0_f32, theme::LINE_PANEL));
    painter.circle_stroke(center, inner, egui::Stroke::new(1.0_f32, theme::LINE_PANEL));
    for (deg, label) in [(0.0, "0"), (90.0, "90"), (180.0, "180"), (270.0, "270")] {
        painter.text(dial_point(center, inner - 17.0, deg), egui::Align2::CENTER_CENTER, label, egui::FontId::proportional(theme::SIZE_META), theme::TEXT_DIM);
    }
    // Probe: click or drag on the dial.
    let id = ui.id().with("shape-dial-probe");
    let mut probe: f32 = ui.data(|d| d.get_temp(id)).unwrap_or(0.0);
    if (resp.clicked() || resp.dragged())
        && grabbed.is_none()
        && let Some(p) = resp.interact_pointer_pos()
    {
        probe = hit_angle(center, p);
        ui.data_mut(|d| d.insert_temp(id, probe));
    }
    painter.line_segment([dial_point(center, inner, probe), dial_point(center, outer, probe)], egui::Stroke::new(2.0_f32, Color32::WHITE));
    // Handle knobs on the rim, in the range's colour.
    for (i, r) in profile.angle_ranges.iter().enumerate() {
        for deg in [r.start, r.end] {
            let at = dial_point(center, outer, deg);
            let active = matches!(grabbed, Some(DialHandle::Start(k) | DialHandle::End(k)) if k == i);
            painter.circle_filled(at, if active { 5.5 } else { 4.0 }, range_color(i));
            painter.circle_stroke(at, if active { 5.5 } else { 4.0 }, egui::Stroke::new(1.5_f32, Color32::WHITE));
        }
    }
    ui.label(RichText::new(probe_label(profile, probe)).size(theme::SIZE_META).color(theme::TEXT_SECONDARY));
    changed
}

fn apply_profile_preset(profile: &mut SpriteShapeProfile, data: &PresetData) {
    profile.angle_ranges = data.profile.angle_ranges.clone();
    profile.fill_pixels_per_unit = data.profile.fill_pixels_per_unit;
    profile.fill_edge_offset = data.profile.fill_edge_offset;
}

/// The `.spriteshape` editor: preset menu, the angle dial, the ranges with
/// their sprite lists, corners, caps and fill. True when anything changed.
pub fn sprite_shape_profile_ui(ui: &mut Ui, profile: &mut SpriteShapeProfile) -> bool {
    let mut changed = false;
    ui.menu_button("Apply Preset to Ranges", |ui| {
        widgets::menu_frame(ui);
        for p in sprite_shape::ShapePreset::ALL {
            if ui.button(p.label()).on_hover_text(p.hint()).clicked() {
                apply_profile_preset(profile, &sprite_shape::preset_shape(p));
                changed = true;
                ui.close();
            }
        }
    });
    note(ui, "Angle 0 faces up and grows counter-clockwise. A segment uses the range its outward normal falls in; on overlaps the higher Order wins.");
    changed |= draw_dial(ui, profile);
    note(ui, "Drag the knobs on the dial rim to move a range's start and end (Shift = 5 degree steps); drag elsewhere to probe an angle.");
    let gaps = profile.uncovered_arcs();
    if !gaps.is_empty() {
        let text = gaps.iter().map(|(a, b)| format!("{a:.0}-{b:.0}")).collect::<Vec<_>>().join(", ");
        ui.label(RichText::new(format!("No range covers {text} deg: those edges get no sprite.")).size(theme::SIZE_META).color(theme::STATUS_WARN_FG));
    }

    ui.add_space(4.0);
    widgets::eyebrow(ui, "Angle Ranges");
    let mut remove: Option<usize> = None;
    for i in 0..profile.angle_ranges.len() {
        let r = &mut profile.angle_ranges[i];
        let header = format!("Range {}  {:.0}..{:.0}  order {}  ({})", i + 1, r.start, r.end, r.order, sprite_count(r.sprites.len()));
        egui::CollapsingHeader::new(RichText::new(header).color(range_color(i))).id_salt(("shape-range", i)).default_open(profile_range_open(i)).show(ui, |ui| {
            changed |= angle_range_ui(ui, r, i);
            if widgets::secondary_button(ui, Some(Icon::Trash), "Remove Range").clicked() {
                remove = Some(i);
            }
        });
    }
    if let Some(i) = remove {
        profile.angle_ranges.remove(i);
        changed = true;
    }
    if widgets::secondary_button(ui, Some(Icon::Plus), "Add Range").clicked() {
        profile.angle_ranges.push(AngleRange { start: 0.0, end: 90.0, order: 0, sprites: Vec::new() });
        changed = true;
    }

    ui.add_space(6.0);
    widgets::eyebrow(ui, "Corners and Caps");
    sprite_slot(ui, "Outer Corner", &mut profile.outer_corner, &mut changed);
    sprite_slot(ui, "Inner Corner", &mut profile.inner_corner, &mut changed);
    sprite_slot(ui, "Start Cap", &mut profile.start_cap, &mut changed);
    sprite_slot(ui, "End Cap", &mut profile.end_cap, &mut changed);
    note(ui, "Corner sprites are used where a point turns past the shape's Corner Angle; empty = a mitered join. Caps are for open shapes.");

    ui.add_space(6.0);
    widgets::eyebrow(ui, "Fill");
    widgets::property_row(ui, "Fill Texture", |ui| changed |= asset_chip(ui, "shape-fill-tex", &mut profile.fill_texture, "none", &["png", "jpg", "jpeg", "tga", "bmp"]));
    num_row(ui, "Pixels Per Unit", &mut profile.fill_pixels_per_unit, 1.0, 1.0..=4096.0, "The texture repeats every size_px / PPU world units (needs Repeat wrap)", &mut changed);
    drag2(ui, "UV Offset", &mut profile.fill_uv_offset, 0.02, &mut changed);
    sprite_slot(ui, "Fill Sprite", &mut profile.fill_sprite, &mut changed);
    num_row(ui, "Edge Offset", &mut profile.fill_edge_offset, 0.01, -5.0..=5.0, "Moves the fill outline outward (negative tucks it under the edge sprites)", &mut changed);
    note(ui, "Without a texture the fill sprite is stretched once over the shape's bounds.");
    changed
}

fn profile_range_open(i: usize) -> bool {
    i < 2
}

fn sprite_count(n: usize) -> String {
    if n == 1 { "1 sprite".into() } else { format!("{n} sprites") }
}

fn angle_range_ui(ui: &mut Ui, r: &mut AngleRange, i: usize) -> bool {
    let mut changed = false;
    num_row(ui, "Start", &mut r.start, 0.5, -720.0..=720.0, "Degrees, counter-clockwise from up", &mut changed);
    num_row(ui, "End", &mut r.end, 0.5, -720.0..=720.0, "Degrees; may be less than Start to wrap through 360", &mut changed);
    widgets::property_row(ui, "Order", |ui| changed |= ui.add(DragValue::new(&mut r.order).speed(0.2)).on_hover_text("Higher wins where ranges overlap").changed());
    let mut drop: Option<usize> = None;
    for (k, s) in r.sprites.iter_mut().enumerate() {
        let label = format!("Sprite {}", k + 1);
        let had = !s.is_empty();
        sprite_slot(ui, &label, s, &mut changed);
        // The slot's clear button removes the entry (an unfilled new slot stays).
        if had && s.is_empty() {
            drop = Some(k);
        }
    }
    if let Some(k) = drop {
        r.sprites.remove(k);
        changed = true;
    }
    if widgets::secondary_button(ui, Some(Icon::Plus), "Add Sprite").clicked() {
        r.sprites.push(String::new());
        changed = true;
    }
    if r.sprites.is_empty() {
        note(ui, "Add one or more sprites; a point's Sprite Variant picks between them.");
    }
    let _ = i;
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dial_points_and_angles_are_inverse() {
        let c = Pos2::new(100.0, 100.0);
        let up = dial_point(c, 50.0, 0.0);
        assert!((up.x - 100.0).abs() < 1e-4 && (up.y - 50.0).abs() < 1e-4, "0 deg is up (y down on screen)");
        let left = dial_point(c, 50.0, 90.0);
        assert!((left.x - 50.0).abs() < 1e-3 && (left.y - 100.0).abs() < 1e-3, "90 deg is left");
        let down = dial_point(c, 50.0, 180.0);
        assert!((down.y - 150.0).abs() < 1e-3);
        for deg in [0.0, 1.0, 45.0, 90.0, 179.0, 180.0, 270.0, 359.0] {
            let back = hit_angle(c, dial_point(c, 33.0, deg));
            assert!((back - deg).abs() < 1e-2 || (back - deg).abs() > 359.98, "{deg} -> {back}");
        }
    }

    #[test]
    fn probe_names_the_winning_range() {
        let mut p = SpriteShapeProfile {
            angle_ranges: vec![
                AngleRange { start: -40.0, end: 40.0, order: 1, sprites: vec!["a".into()] },
                AngleRange { start: 0.0, end: 360.0, order: 0, sprites: vec![] },
            ],
            ..Default::default()
        };
        assert!(probe_label(&p, 10.0).contains("range 1"));
        assert!(probe_label(&p, 350.0).contains("range 1"), "wraps through 0");
        assert!(probe_label(&p, 180.0).contains("range 2"));
        p.angle_ranges.truncate(1);
        assert!(probe_label(&p, 180.0).contains("no range"));
    }

    #[test]
    fn preset_replaces_only_the_ranges() {
        let mut p = SpriteShapeProfile { fill_texture: "t.png".into(), outer_corner: "c#0000000000000001".into(), ..Default::default() };
        apply_profile_preset(&mut p, &sprite_shape::preset_shape(sprite_shape::ShapePreset::GroundWithFill));
        assert_eq!(p.angle_ranges.len(), 4);
        assert_eq!((p.fill_texture.as_str(), p.outer_corner.as_str()), ("t.png", "c#0000000000000001"));
    }

    #[test]
    fn point_list_labels() {
        let mut p = ShapePoint::at(1.0, 2.5);
        assert_eq!(point_summary(0, &p), "1  (1.00, 2.50)  L");
        p.mode = ShapeTangentMode::Broken;
        p.corner = false;
        assert!(point_summary(2, &p).ends_with("B  no corner"));
    }

    #[test]
    fn dial_handles_are_picked_and_dragged_without_losing_the_form() {
        let c = Pos2::new(100.0, 100.0);
        let mut p = SpriteShapeProfile {
            angle_ranges: vec![
                AngleRange { start: -40.0, end: 40.0, order: 0, sprites: vec![] },
                AngleRange { start: 100.0, end: 170.0, order: 0, sprites: vec![] },
            ],
            ..Default::default()
        };
        // The knob of range 2's end sits at 170 degrees on the rim.
        let at = dial_point(c, 71.0, 170.0);
        assert_eq!(pick_handle(&p, c, 71.0, at + egui::vec2(2.0, 3.0)), Some(DialHandle::End(1)));
        assert_eq!(pick_handle(&p, c, 71.0, dial_point(c, 71.0, -40.0)), Some(DialHandle::Start(0)));
        assert_eq!(pick_handle(&p, c, 71.0, dial_point(c, 71.0, 135.0)), None, "the middle of a range is not a handle");
        assert_eq!(pick_handle(&p, c, 71.0, c), None);
        // Drag the end across the wrap: 40 -> 350 degrees on the dial means -10 (shortest way), not 350.
        assert!(drag_handle(&mut p, DialHandle::End(0), 350.0, 1.0));
        assert_eq!(p.angle_ranges[0].end, -10.0);
        // Continuing the other way past the start keeps the stored form continuous.
        assert!(drag_handle(&mut p, DialHandle::End(0), 20.0, 1.0));
        assert_eq!(p.angle_ranges[0].end, 20.0);
        // Snapping: Shift = 5 degrees.
        assert!(drag_handle(&mut p, DialHandle::Start(1), 93.0, 5.0));
        assert_eq!(p.angle_ranges[1].start, 95.0);
        // A no-op and a bad index change nothing.
        assert!(!drag_handle(&mut p, DialHandle::Start(1), 95.0, 5.0));
        assert!(!drag_handle(&mut p, DialHandle::Start(9), 10.0, 1.0));
        // The edit is real: the dial's resolved arcs follow.
        assert!(p.angle_ranges[0].contains(10.0));
        assert!(!p.angle_ranges[0].contains(30.0));
    }

    #[test]
    fn a_new_hole_sits_inside_the_outline() {
        let s = SpriteShapeRenderer {
            points: vec![ShapePoint::at(-4.0, 3.0), ShapePoint::at(4.0, 3.0), ShapePoint::at(4.0, -3.0), ShapePoint::at(-4.0, -3.0)],
            closed: true,
            ..Default::default()
        };
        let h = new_hole_for(&s);
        assert_eq!(h.points.len(), 8);
        assert!(h.points.iter().all(|p| p.position[0].abs() < 4.0 && p.position[1].abs() < 3.0));
        let mut with = s.clone();
        with.holes.push(h);
        assert_ne!(with.content_hash(), s.content_hash());
        assert_eq!(new_hole_for(&SpriteShapeRenderer { points: vec![], ..Default::default() }).points.len(), 8);
    }

    #[test]
    fn colors_cycle() {
        assert_eq!(range_color(0), range_color(8));
        assert_ne!(range_color(0), range_color(1));
    }

    #[test]
    fn edit_state_defaults_are_inert() {
        let s = ShapeEditState::default();
        assert!(!s.active && s.selected.is_none() && s.snap == 0.0);
    }
}
