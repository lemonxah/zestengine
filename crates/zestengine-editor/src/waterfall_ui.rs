//! Inspectors and builder gizmos for `Waterfall`, `Whirlpool` and
//! `Wave Rider`.

use zestengine_core::{RideMode, WaveRider, Waterfall, Whirlpool, fall_time};
use egui::{DragValue, RichText, Ui};
use glam::Vec3;

use crate::components::{EditorComponents, Gizmo, GizmoEdit, GizmoSpec, Inspect, InspectCtx};
use crate::theme;
use crate::widgets;

pub(crate) fn register(e: &mut EditorComponents) {
    e.register::<Waterfall>();
    e.register::<Whirlpool>();
    e.register::<WaveRider>();
}

const WATER: egui::Color32 = egui::Color32::from_rgb(80, 170, 230);
const SPEED: egui::Color32 = egui::Color32::from_rgb(250, 200, 90);

fn drag(ui: &mut Ui, label: &str, v: &mut f32, speed: f64, range: std::ops::RangeInclusive<f32>, tip: &str, changed: &mut bool) {
    widgets::property_row(ui, label, |ui| *changed |= ui.add(DragValue::new(v).speed(speed).range(range)).on_hover_text(tip).changed());
}

fn note(ui: &mut Ui, s: &str) {
    ui.label(RichText::new(s).size(theme::SIZE_META).color(theme::TEXT_DIM));
}

impl Inspect for Waterfall {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut c = false;
        note(ui, "Drag the two lip handles to set where the water pours over and the yellow handle to set how fast it leaves. It falls along the object's +Z and lands on water or the ground below.");
        widgets::eyebrow(ui, "Lip");
        for (k, label) in ["Start", "End"].into_iter().enumerate() {
            widgets::property_row(ui, label, |ui| {
                for v in &mut self.lip[k] {
                    c |= ui.add(DragValue::new(v).speed(0.02)).changed();
                }
            });
        }
        widgets::eyebrow(ui, "Flow");
        drag(ui, "Exit speed (m/s)", &mut self.flow_speed, 0.05, 0.0..=20.0, "How far out it throws before falling", &mut c);
        drag(ui, "Volume", &mut self.volume, 0.01, 0.0..=1.0, "How dense the falling sheet is", &mut c);
        drag(ui, "Spread (m/s)", &mut self.spread, 0.01, 0.0..=2.0, "Sideways scatter as it falls", &mut c);
        drag(ui, "Max drop (m)", &mut self.max_drop, 0.1, 0.1..=500.0, "Fall height when nothing below catches it", &mut c);
        widgets::eyebrow(ui, "Plunge");
        drag(ui, "Splash", &mut self.splash, 0.02, 0.0..=4.0, "Spray thrown up where it lands", &mut c);
        drag(ui, "Mist", &mut self.mist, 0.02, 0.0..=4.0, "Mist rising from the plunge", &mut c);
        drag(ui, "Foam", &mut self.foam, 0.02, 0.0..=4.0, "Foam drifting away on the current", &mut c);
        drag(ui, "Swirl (m/s)", &mut self.swirl, 0.05, -10.0..=10.0, "Plunge pool vortex; negative turns clockwise. Floating bodies get carried round", &mut c);
        drag(ui, "Swirl radius", &mut self.swirl_radius, 0.05, 0.1..=10.0, "As a multiple of the lip width", &mut c);
        widgets::eyebrow(ui, "Look");
        widgets::property_row(ui, "Colour", |ui| c |= ui.color_edit_button_rgb(&mut self.color).changed());
        drag(ui, "Brightness", &mut self.brightness, 0.01, 0.0..=4.0, "Lower it in dark scenes", &mut c);
        c
    }
}

fn speed_handle(f: &Waterfall) -> Vec3 {
    let mid = (Vec3::from(f.lip[0]) + Vec3::from(f.lip[1])) * 0.5;
    mid + Vec3::Z * f.flow_speed * 0.5
}

impl Gizmo for Waterfall {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let (a, b) = (Vec3::from(self.lip[0]), Vec3::from(self.lip[1]));
        let mid = (a + b) * 0.5;
        let mut g = vec![
            GizmoSpec::Handle { position: a, anchor: b, color: WATER },
            GizmoSpec::Handle { position: b, anchor: a, color: WATER },
            GizmoSpec::Handle { position: speed_handle(self), anchor: mid, color: SPEED },
        ];
        // Free-fall arcs from the lip's ends and middle, in local space.
        let t_end = fall_time(self.max_drop);
        let mut segments = vec![(a, b)];
        for start in [a, mid, b] {
            let at = |k: usize| {
                let t = t_end * k as f32 / 16.0;
                start + Vec3::Z * self.flow_speed * t - Vec3::Y * (0.5 * zestengine_core::GRAVITY * t * t)
            };
            for k in 0..16 {
                segments.push((at(k), at(k + 1)));
            }
        }
        g.push(GizmoSpec::Lines { segments, color: WATER, world: false });
        g
    }

    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Handle { index, position } = edit {
            match index {
                0 | 1 => self.lip[index] = position.to_array(),
                2 => {
                    let mid = (Vec3::from(self.lip[0]) + Vec3::from(self.lip[1])) * 0.5;
                    self.flow_speed = ((position.z - mid.z) * 2.0).clamp(0.0, 20.0);
                }
                _ => {}
            }
        }
    }
}

impl Inspect for Whirlpool {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut c = false;
        note(ui, "Swirls the water around this object: floating bodies and foam get carried round and drawn in.");
        drag(ui, "Radius (m)", &mut self.radius, 0.05, 0.1..=200.0, "Reach of the swirl", &mut c);
        drag(ui, "Strength (m/s)", &mut self.strength, 0.05, 0.0..=30.0, "Peak swirl speed, halfway out", &mut c);
        drag(ui, "Pull (m/s)", &mut self.pull, 0.02, 0.0..=10.0, "Inward drift at the centre", &mut c);
        widgets::property_row(ui, "Clockwise", |ui| c |= ui.checkbox(&mut self.clockwise, "").changed());
        drag(ui, "Foam", &mut self.foam, 0.02, 0.0..=4.0, "Foam drawn into the swirl", &mut c);
        c
    }
}

impl Gizmo for Whirlpool {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let mut segments = Vec::new();
        let ring = |r: f32, segments: &mut Vec<(Vec3, Vec3)>| {
            for k in 0..48 {
                let a = k as f32 / 48.0 * std::f32::consts::TAU;
                let b = (k + 1) as f32 / 48.0 * std::f32::consts::TAU;
                segments.push((Vec3::new(a.cos(), 0.0, a.sin()) * r, Vec3::new(b.cos(), 0.0, b.sin()) * r));
            }
        };
        ring(self.radius, &mut segments);
        // A spiral showing the turning direction.
        let s = if self.clockwise { -1.0 } else { 1.0 };
        let at = |k: usize| {
            let t = k as f32 / 64.0;
            let a = s * t * std::f32::consts::TAU * 2.0;
            Vec3::new(a.cos(), 0.0, a.sin()) * self.radius * (1.0 - t * 0.85)
        };
        for k in 0..64 {
            segments.push((at(k), at(k + 1)));
        }
        vec![
            GizmoSpec::Range { center: Vec3::ZERO, radius: self.radius, color: WATER },
            GizmoSpec::Lines { segments, color: WATER, world: false },
        ]
    }

    fn apply_gizmo_edit(&mut self, edit: GizmoEdit) {
        if let GizmoEdit::Range { radius, .. } = edit {
            self.radius = radius.max(0.1);
        }
    }
}

impl Inspect for WaveRider {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut c = false;
        note(ui, "Rides the waves without a physics body. The transform is the calm anchor (what the network syncs); each client adds the same wave motion from the shared water clock. Use it instead of a dynamic Rigid Body.");
        widgets::property_row(ui, "Mode", |ui| {
            let opts: Vec<(RideMode, &str)> = RideMode::ALL.iter().map(|m| (*m, m.label())).collect();
            c |= widgets::segmented(ui, &mut self.mode, &opts).changed();
        });
        match self.mode {
            RideMode::Float => {
                drag(ui, "Draft (m)", &mut self.draft, 0.005, 0.0..=50.0, "How deep it sits in the water", &mut c);
                drag(ui, "Reach (m)", &mut self.reach, 0.02, 0.0..=50.0, "Snaps onto the surface when its anchor is this close", &mut c);
            }
            RideMode::Suspended => note(ui, "Keeps its depth; the water's orbits carry it, fading with depth."),
        }
        drag(ui, "Size (m)", &mut self.size, 0.02, 0.0..=200.0, "Across: waves much shorter than this barely move it", &mut c);
        drag(ui, "Weight", &mut self.weight, 0.01, 0.0..=1.0, "0 = light as a cork, 1 = heavy: bobs on the average swell, hardly swings", &mut c);
        drag(ui, "Tilt", &mut self.tilt, 0.01, 0.0..=1.0, "How much it leans with the surface", &mut c);
        widgets::eyebrow(ui, "Drift");
        drag(ui, "Steepness", &mut self.drift_steepness, 0.005, 0.0..=1.0, "Waves steeper than this push it along (0 = never)", &mut c);
        drag(ui, "Strength", &mut self.drift, 0.02, 0.0..=10.0, "How far steep waves push it", &mut c);
        let s = &self.state;
        let status = if !s.in_water {
            "Not over water".to_owned()
        } else if s.on_surface {
            format!("On the surface (anchor {:.2} m under)", s.depth)
        } else {
            format!("Under water, {:.2} m deep", s.depth)
        };
        note(ui, &status);
        c
    }
}

impl Gizmo for WaveRider {
    fn gizmos(&self) -> Vec<GizmoSpec> {
        // The waterline it floats at: a ring at the draft above the origin.
        if self.mode != RideMode::Float {
            return Vec::new();
        }
        let r = (self.size * 0.5).max(0.05);
        let y = self.draft;
        let segments = (0..32)
            .map(|k| {
                let a = k as f32 / 32.0 * std::f32::consts::TAU;
                let b = (k + 1) as f32 / 32.0 * std::f32::consts::TAU;
                (Vec3::new(a.cos() * r, y, a.sin() * r), Vec3::new(b.cos() * r, y, b.sin() * r))
            })
            .collect();
        vec![GizmoSpec::Lines { segments, color: WATER, world: false }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_edit_the_lip_and_speed() {
        let mut f = Waterfall::default();
        f.apply_gizmo_edit(GizmoEdit::Handle { index: 1, position: Vec3::new(3.0, 0.5, 0.0) });
        assert_eq!(f.lip[1], [3.0, 0.5, 0.0]);
        let mid = (Vec3::from(f.lip[0]) + Vec3::from(f.lip[1])) * 0.5;
        f.apply_gizmo_edit(GizmoEdit::Handle { index: 2, position: mid + Vec3::Z * 2.0 });
        assert!((f.flow_speed - 4.0).abs() < 1e-5);
        match &f.gizmos()[2] {
            GizmoSpec::Handle { position, .. } => assert!(position.distance(mid + Vec3::Z * 2.0) < 1e-5),
            _ => panic!("speed handle"),
        }
    }
}
