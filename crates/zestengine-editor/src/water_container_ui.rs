//! Inspector and gizmo for `Water Container`.

use zestengine_core::{WaterContainer, surface_polygon};
use egui::{DragValue, RichText, Ui};
use glam::Vec3;

use crate::components::{EditorComponents, Gizmo, GizmoEdit, GizmoSpec, Inspect, InspectCtx};
use crate::icons::Icon;
use crate::theme;
use crate::widgets;

pub(crate) fn register(e: &mut EditorComponents) {
    e.register::<WaterContainer>();
}

const WATER: egui::Color32 = egui::Color32::from_rgb(80, 170, 230);
const LEVEL: egui::Color32 = egui::Color32::from_rgb(140, 220, 255);

fn note(ui: &mut Ui, s: &str) {
    ui.label(RichText::new(s).size(theme::SIZE_META).color(theme::TEXT_DIM));
}

fn scale_of(c: &WaterContainer) -> Vec3 {
    let s = Vec3::from(c.runtime.scale);
    if s.abs().min_element() > 1e-6 { s } else { Vec3::ONE }
}

impl Inspect for WaterContainer {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut c = false;
        note(ui, "Holds water inside the blue box; its top face is the opening. Tip, swing or stop it hard and water pours over the rim; dip the opening under water, or pour water in, and it fills. The water's weight is added to the Rigid Body.");
        let capacity = self.capacity_litres(scale_of(self));
        widgets::property_row(ui, "Capacity", |ui| ui.label(format!("{capacity:.1} L")));
        if self.runtime.started {
            let r = &self.runtime;
            let share = r.litres / r.capacity.max(1e-6);
            widgets::property_row(ui, "Holding", |ui| ui.label(format!("{:.1} L ({:.0}%)", r.litres, share * 100.0)));
            if r.spilling > 0.05 || r.filling.abs() > 0.05 {
                note(ui, &format!("Pouring out {:.1} L/s, flowing in {:.1} L/s", r.spilling, r.filling));
            }
        }
        widgets::eyebrow(ui, "Inner volume");
        widgets::property_row(ui, "Centre", |ui| {
            for v in &mut self.center {
                c |= ui.add(DragValue::new(v).speed(0.01)).changed();
            }
        });
        widgets::property_row(ui, "Half size", |ui| {
            for v in &mut self.half_extents {
                c |= ui.add(DragValue::new(v).speed(0.01).range(0.001..=1000.0)).changed();
            }
        });
        if widgets::secondary_button(ui, Some(Icon::Target), "Fit to mesh").on_hover_text("Fit the box just inside the mesh's bounds, open at the top").clicked() {
            self.runtime.fit_requested = true;
            c = true;
        }
        widgets::eyebrow(ui, "Water");
        widgets::property_row(ui, "Start fill", |ui| {
            let mut pct = self.initial_fill * 100.0;
            let r = widgets::slider(ui, egui::Slider::new(&mut pct, 0.0..=100.0).suffix("%").fixed_decimals(0));
            if r.changed() {
                self.initial_fill = (pct / 100.0).clamp(0.0, 1.0);
                c = true;
            }
            r.on_hover_text(format!("{:.1} L at the start of Play", self.initial_fill * capacity));
        });
        widgets::property_row(ui, "Start litres", |ui| {
            let mut litres = self.initial_fill * capacity;
            if ui.add(DragValue::new(&mut litres).speed(0.1).range(0.0..=capacity.max(0.0)).suffix(" L")).changed() {
                self.initial_fill = (litres / capacity.max(1e-6)).clamp(0.0, 1.0);
                c = true;
            }
        });
        widgets::property_row(ui, "Pour rate", |ui| {
            c |= ui.add(DragValue::new(&mut self.pour_rate).speed(0.02).range(0.0..=10.0)).on_hover_text("Scales how fast water pours over the rim").changed();
        });
        widgets::property_row(ui, "Inflow", |ui| {
            c |= ui.add(DragValue::new(&mut self.inflow).speed(0.02).range(0.0..=10.0)).on_hover_text("Scales how fast it fills through an opening under water").changed();
        });
        widgets::property_row(ui, "Slosh", |ui| {
            c |= widgets::slider(ui, egui::Slider::new(&mut self.slosh, 0.0..=1.0)).on_hover_text("0 settles at once, 1 rocks on for a long time").changed();
        });
        widgets::eyebrow(ui, "Look");
        widgets::property_row(ui, "Colour", |ui| c |= ui.color_edit_button_rgb(&mut self.color).changed());
        widgets::property_row(ui, "Opacity", |ui| {
            c |= widgets::slider(ui, egui::Slider::new(&mut self.opacity, 0.0..=1.0)).changed();
        });
        c
    }
}

impl Gizmo for WaterContainer {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let center = Vec3::from(self.center);
        let mut g = vec![GizmoSpec::Box { center, half_extents: Vec3::from(self.half_extents), object_anchored: false, blend: 0.0, color: WATER }];
        // The water's surface while playing (box frame is metres; the gizmo
        // draws in object space before scale).
        if self.runtime.started && self.runtime.litres > 1e-3 {
            let s = scale_of(self);
            let half = self.half_m(s);
            let poly = surface_polygon(half, self.runtime.up, self.runtime.level);
            let local = |p: Vec3| center + p / s;
            let segments: Vec<(Vec3, Vec3)> = (0..poly.len()).map(|k| (local(poly[k]), local(poly[(k + 1) % poly.len()]))).collect();
            if !segments.is_empty() {
                g.push(GizmoSpec::Lines { segments, color: LEVEL, world: false });
            }
        }
        g
    }

    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Box { half_extents, center, .. } = edit {
            self.half_extents = half_extents.max(Vec3::splat(0.001)).to_array();
            self.center = center.to_array();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_box_handle_resizes_the_inner_volume() {
        let mut c = WaterContainer::default();
        c.apply_gizmo_edit(GizmoEdit::Box { index: 0, half_extents: Vec3::new(0.3, 0.4, 0.2), center: Vec3::new(0.0, 0.1, 0.0) });
        assert_eq!(c.half_extents, [0.3, 0.4, 0.2]);
        assert_eq!(c.center, [0.0, 0.1, 0.0]);
        assert!(matches!(c.gizmos()[0], GizmoSpec::Box { object_anchored: false, .. }));
        assert!((c.capacity_litres(Vec3::ONE) - 0.6 * 0.8 * 0.4 * 1000.0).abs() < 0.1);
    }

    #[test]
    fn the_level_shows_while_playing() {
        let mut c = WaterContainer::default();
        assert_eq!(c.gizmos().len(), 1);
        c.runtime.started = true;
        c.runtime.litres = 100.0;
        c.runtime.up = Vec3::Y;
        c.runtime.level = 0.0;
        match &c.gizmos()[1] {
            GizmoSpec::Lines { segments, .. } => assert_eq!(segments.len(), 4),
            _ => panic!("level outline"),
        }
    }
}
