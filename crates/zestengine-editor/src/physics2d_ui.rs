//! Inspectors and gizmos for the 2D physics components (Rigidbody 2D, the 2D
//! colliders, Joint 2D, Effector 2D). A child module of `components` so it
//! shares the themed row helpers. Gizmo geometry lies in the object's XY plane.

use egui::{Color32, DragValue, RichText, Ui};
use glam::{Vec2, Vec3};
use zestengine_core::{
    BodyType2D, BoxCollider2D, CapsuleCollider2D, CapsuleDirection2D, CircleCollider2D, CompositeCollider2D, CompositeGeometry2D, EdgeCollider2D, Effector2D,
    EffectorKind2D, Interpolation2D, Joint2D, Joint2DKind, PointForceMode2D, PolygonCollider2D, Rigidbody2D, SleepMode2D,
};

use super::{Gizmo, GizmoEdit, GizmoSpec, Inspect, InspectCtx, enum_row, property_row};
use crate::{theme, widgets};

const SOLID: Color32 = Color32::from_rgb(240, 220, 70);
const TRIGGER: Color32 = Color32::from_rgb(110, 210, 240);
const JOINT: Color32 = Color32::from_rgb(255, 200, 90);
const EFFECTOR: Color32 = Color32::from_rgb(190, 140, 255);

/// Most polygon / edge vertices that get a draggable handle.
const MAX_HANDLES: usize = 64;

fn vec2_row(ui: &mut Ui, label: &str, v: &mut [f32; 2], speed: f64, changed: &mut bool) {
    widgets::property_row(ui, label, |ui| {
        ui.horizontal(|ui| {
            for x in v.iter_mut() {
                *changed |= ui.add(DragValue::new(x).speed(speed).max_decimals(3)).changed();
            }
        });
    });
}

fn check_row(ui: &mut Ui, label: &str, v: &mut bool, tip: &str, changed: &mut bool) {
    property_row(ui, label, changed, |ui| ui.checkbox(v, "").on_hover_text(tip));
}

fn float_row(ui: &mut Ui, label: &str, v: &mut f32, speed: f64, range: std::ops::RangeInclusive<f32>, changed: &mut bool) {
    property_row(ui, label, changed, |ui| ui.add(DragValue::new(v).speed(speed).range(range)));
}

/// Shared collider rows: offset, trigger, effector / composite flags, density, material.
#[allow(clippy::too_many_arguments)]
fn collider_rows(
    ui: &mut Ui,
    ctx: &InspectCtx,
    offset: &mut [f32; 2],
    is_trigger: &mut bool,
    used_by_effector: &mut bool,
    used_by_composite: Option<&mut bool>,
    density: Option<&mut f32>,
    material: &mut String,
    changed: &mut bool,
) {
    vec2_row(ui, "Offset", offset, 0.01, changed);
    check_row(ui, "Is Trigger", is_trigger, "Detect overlaps without blocking (no collision response)", changed);
    check_row(ui, "Used By Effector", used_by_effector, "An Effector 2D on this object acts through this collider", changed);
    if let Some(c) = used_by_composite {
        check_row(ui, "In Composite", c, "Merge into the Composite Collider 2D above (or on) this object", changed);
    }
    if let Some(d) = density {
        float_row(ui, "Density", d, 0.05, 0.0..=1000.0, changed);
    }
    *changed |= ctx.file_field(ui, "Physics Material", material, "body's or project default", &["physmat"]);
    ui.label(RichText::new("Collides by the object's layer (Layers matrix, shared with 3D)").small().weak());
}

fn circle(out: &mut Vec<(Vec3, Vec3)>, c: Vec2, r: f32) {
    let mut prev = c + Vec2::new(r, 0.0);
    for k in 1..=32 {
        let a = k as f32 / 32.0 * std::f32::consts::TAU;
        let q = c + Vec2::from_angle(a) * r;
        out.push((prev.extend(0.0), q.extend(0.0)));
        prev = q;
    }
}

fn ring(out: &mut Vec<(Vec3, Vec3)>, pts: &[[f32; 2]], off: Vec2, closed: bool) {
    let n = pts.len();
    for k in 0..if closed { n } else { n.saturating_sub(1) } {
        out.push(((Vec2::from(pts[k]) + off).extend(0.0), (Vec2::from(pts[(k + 1) % n]) + off).extend(0.0)));
    }
}

fn arrow(out: &mut Vec<(Vec3, Vec3)>, from: Vec2, dir: Vec2, len: f32) {
    let tip = from + dir * len;
    out.push((from.extend(0.0), tip.extend(0.0)));
    let side = dir.perp();
    for s in [-1.0, 1.0] {
        out.push((tip.extend(0.0), (tip - dir * len * 0.2 + side * s * len * 0.12).extend(0.0)));
    }
}

fn colour(trigger: bool) -> Color32 {
    if trigger { TRIGGER } else { SOLID }
}

impl Inspect for Rigidbody2D {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        enum_row(ui, "Body Type", "zestengine-body2d-type", &mut self.body_type, &BodyType2D::ALL, BodyType2D::label, &mut changed);
        check_row(ui, "Simulated", &mut self.simulated, "Off: the body and its colliders leave the simulation", &mut changed);
        if self.body_type == BodyType2D::Dynamic {
            let mut set = self.mass > 0.0;
            widgets::property_row(ui, "Mass", |ui| {
                let r = widgets::segmented(ui, &mut set, &[(false, "Auto"), (true, "Set")])
                    .on_hover_text("Auto: collider area times density. Set: a fixed mass in kg (densities are scaled to match).");
                if r.changed() {
                    self.mass = if set { self.measured.map_or(1.0, |(m, _)| m.max(0.01)) } else { 0.0 };
                    changed = true;
                }
            });
            if set {
                float_row(ui, "Mass (kg)", &mut self.mass, 0.1, 0.001..=1e6, &mut changed);
            }
            if let Some((m, a)) = self.measured.filter(|(_, a)| *a > 1e-6) {
                ui.label(RichText::new(format!("{m:.2} kg · {a:.3} m2 · {:.2} kg/m2", m / a)).small().color(theme::TEXT_DIM));
            }
            float_row(ui, "Gravity Scale", &mut self.gravity_scale, 0.05, -100.0..=100.0, &mut changed);
            float_row(ui, "Linear Drag", &mut self.linear_damping, 0.01, 0.0..=1000.0, &mut changed);
            float_row(ui, "Angular Drag", &mut self.angular_damping, 0.01, 0.0..=1000.0, &mut changed);
            widgets::property_row(ui, "Freeze", |ui| {
                ui.horizontal(|ui| {
                    changed |= ui.checkbox(&mut self.freeze_x, "X").changed();
                    changed |= ui.checkbox(&mut self.freeze_y, "Y").changed();
                    changed |= ui.checkbox(&mut self.freeze_rotation, "Rotation").changed();
                });
            });
            check_row(ui, "Continuous", &mut self.continuous, "Continuous collision detection: fast bodies do not tunnel through thin colliders", &mut changed);
            enum_row(ui, "Sleeping", "zestengine-body2d-sleep", &mut self.sleep_mode, &SleepMode2D::ALL, SleepMode2D::label, &mut changed);
            enum_row(ui, "Interpolate", "zestengine-body2d-interp", &mut self.interpolation, &Interpolation2D::ALL, Interpolation2D::label, &mut changed);
        }
        changed |= ctx.file_field(ui, "Physics Material", &mut self.material, "project default", &["physmat"]);
        check_row(ui, "Debug Draw", &mut self.debug_draw, "Outline this body's colliders in the Play view", &mut changed);
        changed
    }
}
impl Gizmo for Rigidbody2D {
    /// A ring with axes at the body origin, coloured by body type.
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let color = match self.body_type {
            BodyType2D::Dynamic => Color32::from_rgb(255, 170, 80),
            BodyType2D::Kinematic => Color32::from_rgb(120, 190, 255),
            BodyType2D::Static => Color32::from_rgb(170, 170, 180),
        };
        let mut segments = Vec::new();
        circle(&mut segments, Vec2::ZERO, 0.12);
        segments.push((Vec3::new(-0.2, 0.0, 0.0), Vec3::new(0.2, 0.0, 0.0)));
        segments.push((Vec3::new(0.0, -0.2, 0.0), Vec3::new(0.0, 0.2, 0.0)));
        vec![GizmoSpec::Lines { segments, color, world: false }]
    }
}

impl Inspect for BoxCollider2D {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        vec2_row(ui, "Size", &mut self.size, 0.02, &mut changed);
        for s in &mut self.size {
            *s = s.max(0.01);
        }
        collider_rows(ui, ctx, &mut self.offset, &mut self.is_trigger, &mut self.used_by_effector, Some(&mut self.used_by_composite), Some(&mut self.density), &mut self.material, &mut changed);
        changed
    }
}
impl Gizmo for BoxCollider2D {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        vec![GizmoSpec::Box {
            center: Vec3::new(self.offset[0], self.offset[1], 0.0),
            half_extents: Vec3::new(self.size[0] * 0.5, self.size[1] * 0.5, 0.005),
            object_anchored: false,
            blend: 0.0,
            color: colour(self.is_trigger),
        }]
    }
    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Box { half_extents, center, .. } = edit {
            self.size = [(half_extents.x * 2.0).max(0.01), (half_extents.y * 2.0).max(0.01)];
            self.offset = [center.x, center.y];
        }
    }
}

impl Inspect for CircleCollider2D {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        float_row(ui, "Radius", &mut self.radius, 0.01, 0.005..=1000.0, &mut changed);
        collider_rows(ui, ctx, &mut self.offset, &mut self.is_trigger, &mut self.used_by_effector, Some(&mut self.used_by_composite), Some(&mut self.density), &mut self.material, &mut changed);
        changed
    }
}
impl Gizmo for CircleCollider2D {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let mut segments = Vec::new();
        circle(&mut segments, Vec2::from(self.offset), self.radius);
        vec![GizmoSpec::Lines { segments, color: colour(self.is_trigger), world: false }]
    }
}

impl Inspect for CapsuleCollider2D {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        vec2_row(ui, "Size", &mut self.size, 0.02, &mut changed);
        for s in &mut self.size {
            *s = s.max(0.01);
        }
        enum_row(ui, "Direction", "zestengine-capsule2d-dir", &mut self.direction, &CapsuleDirection2D::ALL, CapsuleDirection2D::label, &mut changed);
        collider_rows(ui, ctx, &mut self.offset, &mut self.is_trigger, &mut self.used_by_effector, Some(&mut self.used_by_composite), Some(&mut self.density), &mut self.material, &mut changed);
        changed
    }
}
impl Gizmo for CapsuleCollider2D {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let (w, h) = (self.size[0], self.size[1]);
        let off = Vec2::from(self.offset);
        let (r, half, axis) = match self.direction {
            CapsuleDirection2D::Vertical => (w * 0.5, (h * 0.5 - w * 0.5).max(0.0), Vec2::Y),
            CapsuleDirection2D::Horizontal => (h * 0.5, (w * 0.5 - h * 0.5).max(0.0), Vec2::X),
        };
        let (top, bot) = (off + axis * half, off - axis * half);
        let side = axis.perp() * r;
        let base = axis.to_angle();
        let mut s = Vec::new();
        let mut arc = |c: Vec2, from: f32| {
            let mut prev = c + Vec2::from_angle(from) * r;
            for k in 1..=16 {
                let q = c + Vec2::from_angle(from + k as f32 / 16.0 * std::f32::consts::PI) * r;
                s.push((prev.extend(0.0), q.extend(0.0)));
                prev = q;
            }
        };
        arc(top, base - std::f32::consts::FRAC_PI_2);
        arc(bot, base + std::f32::consts::FRAC_PI_2);
        s.push(((top + side).extend(0.0), (bot + side).extend(0.0)));
        s.push(((top - side).extend(0.0), (bot - side).extend(0.0)));
        vec![GizmoSpec::Lines { segments: s, color: colour(self.is_trigger), world: false }]
    }
}

/// Draggable vertex handles (after the outline at index 0).
fn vertex_handles(paths: impl Iterator<Item = [f32; 2]>, off: Vec2) -> impl Iterator<Item = GizmoSpec> {
    paths.take(MAX_HANDLES).map(move |p| {
        let position = (Vec2::from(p) + off).extend(0.0);
        GizmoSpec::Handle { position, anchor: position, color: SOLID }
    })
}

impl Inspect for PolygonCollider2D {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        let verts: usize = self.paths.iter().map(Vec::len).sum();
        ui.label(RichText::new(format!("{} path(s), {verts} vertices. Concave paths split into convex pieces; nested paths are holes. Drag the vertex handles in the scene view.", self.paths.len())).small().weak());
        ui.horizontal(|ui| {
            if ui.button("Reset to box").clicked() {
                *self = PolygonCollider2D { offset: self.offset, is_trigger: self.is_trigger, material: std::mem::take(&mut self.material), ..Default::default() };
                changed = true;
            }
            if ui.button("Add vertex").on_hover_text("Splits the first path's longest edge").clicked()
                && let Some(p) = self.paths.first_mut()
                && p.len() >= 2
            {
                let n = p.len();
                let k = (0..n).max_by(|&a, &b| (Vec2::from(p[a]) - Vec2::from(p[(a + 1) % n])).length().total_cmp(&(Vec2::from(p[b]) - Vec2::from(p[(b + 1) % n])).length())).unwrap_or(0);
                let mid = (Vec2::from(p[k]) + Vec2::from(p[(k + 1) % n])) * 0.5;
                p.insert(k + 1, mid.to_array());
                changed = true;
            }
        });
        collider_rows(ui, ctx, &mut self.offset, &mut self.is_trigger, &mut self.used_by_effector, Some(&mut self.used_by_composite), Some(&mut self.density), &mut self.material, &mut changed);
        changed
    }
}
impl Gizmo for PolygonCollider2D {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let off = Vec2::from(self.offset);
        let mut s = Vec::new();
        for p in &self.paths {
            ring(&mut s, p, off, true);
        }
        let mut g = vec![GizmoSpec::Lines { segments: s, color: colour(self.is_trigger), world: false }];
        g.extend(vertex_handles(self.paths.iter().flatten().copied(), off));
        g
    }
    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Handle { index, position } = edit
            && index >= 1
        {
            let off = Vec2::from(self.offset);
            if let Some(v) = self.paths.iter_mut().flatten().nth(index - 1) {
                *v = (position.truncate() - off).to_array();
            }
        }
    }
}

impl Inspect for EdgeCollider2D {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        ui.label(RichText::new(format!("{} points. Drag the handles in the scene view.", self.points.len())).small().weak());
        check_row(ui, "Closed", &mut self.closed, "Join the last point back to the first", &mut changed);
        ui.horizontal(|ui| {
            if ui.button("Add point").clicked() {
                let last = self.points.last().copied().unwrap_or([0.0; 2]);
                self.points.push([last[0] + 1.0, last[1]]);
                changed = true;
            }
            if ui.button("Remove point").clicked() && self.points.len() > 2 {
                self.points.pop();
                changed = true;
            }
        });
        collider_rows(ui, ctx, &mut self.offset, &mut self.is_trigger, &mut self.used_by_effector, Some(&mut self.used_by_composite), None, &mut self.material, &mut changed);
        changed
    }
}
impl Gizmo for EdgeCollider2D {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let off = Vec2::from(self.offset);
        let mut s = Vec::new();
        ring(&mut s, &self.points, off, self.closed);
        let mut g = vec![GizmoSpec::Lines { segments: s, color: colour(self.is_trigger), world: false }];
        g.extend(vertex_handles(self.points.iter().copied(), off));
        g
    }
    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Handle { index, position } = edit
            && index >= 1
            && let Some(p) = self.points.get_mut(index - 1)
        {
            *p = (position.truncate() - Vec2::from(self.offset)).to_array();
        }
    }
}

impl Inspect for CompositeCollider2D {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        enum_row(ui, "Geometry", "zestengine-composite2d-geom", &mut self.geometry, &CompositeGeometry2D::ALL, CompositeGeometry2D::label, &mut changed);
        let note = match self.geometry {
            CompositeGeometry2D::Polygons => "Solid: every part's shape in one collider.",
            CompositeGeometry2D::Outlines => "Hollow boundary segments (no union): cheap static terrain.",
        };
        ui.label(RichText::new(note).small().weak());
        vec2_row(ui, "Offset", &mut self.offset, 0.01, &mut changed);
        check_row(ui, "Is Trigger", &mut self.is_trigger, "The merged collider detects overlaps without blocking", &mut changed);
        check_row(ui, "Used By Effector", &mut self.used_by_effector, "An Effector 2D on this object acts through the merged collider", &mut changed);
        changed |= ctx.file_field(ui, "Physics Material", &mut self.material, "body's or project default", &["physmat"]);
        ui.label(RichText::new("Merges colliders flagged Used By Composite on this object and its children").small().weak());
        changed
    }
}
impl Gizmo for CompositeCollider2D {}

impl Inspect for Joint2D {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        enum_row(ui, "Type", "zestengine-joint2d-kind", &mut self.kind, &Joint2DKind::ALL, Joint2DKind::label, &mut changed);
        changed |= ctx.object_ref(ui, "Connected Body", &mut self.connected);
        if !self.connected.is_set() {
            ui.label(RichText::new("Unset: attached to its start point in the world").small().weak());
        }
        vec2_row(ui, "Anchor", &mut self.anchor, 0.01, &mut changed);
        let k = self.kind;
        if matches!(k, Joint2DKind::Slider | Joint2DKind::Wheel) {
            float_row(ui, "Axis Angle", &mut self.axis_angle, 0.5, -360.0..=360.0, &mut changed);
        }
        if matches!(k, Joint2DKind::Hinge | Joint2DKind::Slider | Joint2DKind::Wheel) {
            let unit = if k == Joint2DKind::Hinge { " deg" } else { " m" };
            check_row(ui, "Limits", &mut self.limits_enabled, "", &mut changed);
            if self.limits_enabled {
                widgets::property_row(ui, "Range", |ui| {
                    ui.horizontal(|ui| {
                        changed |= ui.add(DragValue::new(&mut self.limits[0]).speed(0.5).suffix(unit)).changed();
                        changed |= ui.add(DragValue::new(&mut self.limits[1]).speed(0.5).suffix(unit)).changed();
                    });
                });
            }
            check_row(ui, "Motor", &mut self.motor_enabled, "", &mut changed);
            if self.motor_enabled {
                let unit = if k == Joint2DKind::Slider { "m/s" } else { "deg/s" };
                property_row(ui, "Motor Speed", &mut changed, |ui| ui.add(DragValue::new(&mut self.motor_speed).speed(1.0).suffix(format!(" {unit}"))));
                float_row(ui, "Max Force", &mut self.motor_max_force, 1.0, 0.0..=1e7, &mut changed);
            }
        }
        if matches!(k, Joint2DKind::Distance | Joint2DKind::Spring) {
            float_row(ui, "Distance", &mut self.distance, 0.01, 0.0..=1000.0, &mut changed);
            ui.label(RichText::new("0 = distance at start").small().weak());
        }
        if k == Joint2DKind::Distance {
            check_row(ui, "Max Distance Only", &mut self.max_distance_only, "A rope: only stops the bodies drifting further apart", &mut changed);
        }
        if matches!(k, Joint2DKind::Spring | Joint2DKind::Wheel) {
            float_row(ui, "Stiffness", &mut self.stiffness, 0.5, 0.0..=1e6, &mut changed);
            float_row(ui, "Damping", &mut self.damping, 0.05, 0.0..=1e4, &mut changed);
        }
        check_row(ui, "Collide Connected", &mut self.collide_connected, "", &mut changed);
        ui.label(RichText::new("Needs a Rigidbody 2D on this object").small().weak());
        changed
    }
}
impl Gizmo for Joint2D {
    /// Anchor handle (drag to move it), the slider / wheel axis, the hinge's
    /// limit arc, and the spring / distance length.
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let anchor = Vec2::from(self.anchor);
        let mut s = Vec::new();
        match self.kind {
            Joint2DKind::Hinge => {
                let (lo, hi) = if self.limits_enabled { (self.limits[0].to_radians(), self.limits[1].to_radians()) } else { (0.0, std::f32::consts::TAU) };
                let mut prev = anchor + Vec2::from_angle(lo) * 0.35;
                for k in 1..=32 {
                    let q = anchor + Vec2::from_angle(lo + (hi - lo) * k as f32 / 32.0) * 0.35;
                    s.push((prev.extend(0.0), q.extend(0.0)));
                    prev = q;
                }
                if self.limits_enabled {
                    for t in [lo, hi] {
                        s.push((anchor.extend(0.0), (anchor + Vec2::from_angle(t) * 0.35).extend(0.0)));
                    }
                }
            }
            Joint2DKind::Slider | Joint2DKind::Wheel => {
                let axis = Vec2::from_angle(self.axis_angle.to_radians());
                let (lo, hi) = if self.limits_enabled { (self.limits[0], self.limits[1]) } else { (-0.5, 0.5) };
                s.push(((anchor + axis * lo).extend(0.0), (anchor + axis * hi).extend(0.0)));
                for t in [lo, hi] {
                    let p = anchor + axis * t;
                    s.push(((p + axis.perp() * 0.08).extend(0.0), (p - axis.perp() * 0.08).extend(0.0)));
                }
            }
            Joint2DKind::Spring | Joint2DKind::Distance if self.distance > 0.0 => {
                s.push((anchor.extend(0.0), (anchor + Vec2::NEG_Y * self.distance).extend(0.0)));
            }
            _ => {}
        }
        let mut g = vec![GizmoSpec::Handle { position: anchor.extend(0.0), anchor: Vec3::ZERO, color: JOINT }];
        if !s.is_empty() {
            g.push(GizmoSpec::Lines { segments: s, color: JOINT, world: false });
        }
        g
    }
    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Handle { index: 0, position } = edit {
            self.anchor = [position.x, position.y];
        }
    }
}

impl Inspect for Effector2D {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        enum_row(ui, "Type", "zestengine-effector2d-kind", &mut self.kind, &EffectorKind2D::ALL, EffectorKind2D::label, &mut changed);
        let hint = if self.kind.uses_trigger() {
            "Acts on bodies inside this object's trigger collider(s) flagged Used By Effector."
        } else {
            "Acts on contacts with this object's solid collider(s) flagged Used By Effector."
        };
        ui.label(RichText::new(hint).small().weak());
        check_row(ui, "Use Layer Mask", &mut self.use_layer_mask, "Only affect bodies on the chosen layers", &mut changed);
        if self.use_layer_mask {
            property_row(ui, "Layer Mask", &mut changed, |ui| ui.add(DragValue::new(&mut self.layer_mask).hexadecimal(8, false, true)));
        }
        match self.kind {
            EffectorKind2D::Area => {
                float_row(ui, "Force Angle", &mut self.force_angle, 0.5, -360.0..=360.0, &mut changed);
                check_row(ui, "Use Global Angle", &mut self.use_global_angle, "Off: the angle turns with the object", &mut changed);
                float_row(ui, "Force Magnitude", &mut self.force_magnitude, 0.1, -1e6..=1e6, &mut changed);
                float_row(ui, "Force Variation", &mut self.force_variation, 0.1, 0.0..=1e6, &mut changed);
                float_row(ui, "Linear Drag", &mut self.linear_drag, 0.01, 0.0..=1000.0, &mut changed);
                float_row(ui, "Angular Drag", &mut self.angular_drag, 0.01, 0.0..=1000.0, &mut changed);
            }
            EffectorKind2D::Buoyancy => {
                float_row(ui, "Surface Level", &mut self.surface_level, 0.01, -1000.0..=1000.0, &mut changed);
                float_row(ui, "Fluid Density", &mut self.fluid_density, 0.05, 0.0..=1000.0, &mut changed);
                float_row(ui, "Linear Drag", &mut self.linear_drag, 0.01, 0.0..=1000.0, &mut changed);
                float_row(ui, "Angular Drag", &mut self.angular_drag, 0.01, 0.0..=1000.0, &mut changed);
                float_row(ui, "Flow Angle", &mut self.flow_angle, 0.5, -360.0..=360.0, &mut changed);
                float_row(ui, "Flow Magnitude", &mut self.flow_magnitude, 0.05, -1000.0..=1000.0, &mut changed);
                float_row(ui, "Flow Variation", &mut self.flow_variation, 0.05, 0.0..=1000.0, &mut changed);
            }
            EffectorKind2D::Point => {
                float_row(ui, "Force Magnitude", &mut self.force_magnitude, 0.1, -1e6..=1e6, &mut changed);
                ui.label(RichText::new("Positive attracts, negative repels").small().weak());
                float_row(ui, "Force Variation", &mut self.force_variation, 0.1, 0.0..=1e6, &mut changed);
                float_row(ui, "Distance Scale", &mut self.distance_scale, 0.05, 0.0..=1000.0, &mut changed);
                enum_row(ui, "Force Mode", "zestengine-effector2d-mode", &mut self.force_mode, &PointForceMode2D::ALL, PointForceMode2D::label, &mut changed);
                float_row(ui, "Linear Drag", &mut self.linear_drag, 0.01, 0.0..=1000.0, &mut changed);
            }
            EffectorKind2D::Platform => {
                check_row(ui, "Use One Way", &mut self.use_one_way, "Contacts only hold from the top arc; bodies pass through from below", &mut changed);
                float_row(ui, "Surface Arc", &mut self.surface_arc, 1.0, 0.0..=180.0, &mut changed);
                check_row(ui, "Use Side Friction", &mut self.use_side_friction, "Sideways contacts keep friction", &mut changed);
                check_row(ui, "Use Side Bounce", &mut self.use_side_bounce, "Sideways contacts keep bounce", &mut changed);
            }
            EffectorKind2D::Surface => {
                float_row(ui, "Speed", &mut self.speed, 0.05, -1000.0..=1000.0, &mut changed);
                float_row(ui, "Speed Variation", &mut self.speed_variation, 0.05, 0.0..=1000.0, &mut changed);
                ui.label(RichText::new("Positive moves things to the right of the surface's up").small().weak());
            }
        }
        changed
    }
}
impl Gizmo for Effector2D {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let mut s = Vec::new();
        match self.kind {
            EffectorKind2D::Area => {
                for dy in [-0.3, 0.0, 0.3] {
                    arrow(&mut s, Vec2::new(0.0, dy), Vec2::from_angle(self.force_angle.to_radians()), 0.6);
                }
            }
            EffectorKind2D::Buoyancy => {
                s.push((Vec3::new(-2.0, self.surface_level, 0.0), Vec3::new(2.0, self.surface_level, 0.0)));
                arrow(&mut s, Vec2::new(0.0, self.surface_level - 0.6), Vec2::Y, 0.5);
            }
            EffectorKind2D::Point => {
                circle(&mut s, Vec2::ZERO, 0.15);
                let sign = if self.force_magnitude >= 0.0 { -1.0 } else { 1.0 };
                for k in 0..4 {
                    let d = Vec2::from_angle(k as f32 * std::f32::consts::FRAC_PI_2);
                    arrow(&mut s, d * (0.5 - 0.25 * sign), d * sign, 0.3);
                }
            }
            EffectorKind2D::Platform => {
                arrow(&mut s, Vec2::ZERO, Vec2::Y, 0.5);
                if self.use_one_way {
                    let arc = self.surface_arc.clamp(0.0, 180.0).to_radians();
                    for sgn in [-1.0, 1.0] {
                        let d = Vec2::from_angle(std::f32::consts::FRAC_PI_2 + arc * sgn);
                        s.push((Vec3::ZERO, (d * 0.5).extend(0.0)));
                    }
                }
            }
            EffectorKind2D::Surface => {
                let d = if self.speed >= 0.0 { Vec2::X } else { Vec2::NEG_X };
                arrow(&mut s, Vec2::new(-0.3 * d.x, 0.2), d, 0.6);
            }
        }
        vec![GizmoSpec::Lines { segments: s, color: EFFECTOR, world: false }]
    }
}
