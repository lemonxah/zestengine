//! Inspector and viewport gizmo for `Humanoid IK` (full-body tracking).

use zestengine_core::{HumanoidIk, IkSource, TRACKER_ROLES, TRACKER_SLOTS};
use egui::{DragValue, RichText, Ui};
use glam::Vec3;

use crate::components::{EditorComponents, Gizmo, GizmoSpec, Inspect, InspectCtx};
use crate::theme;
use crate::widgets;

pub(crate) fn register(e: &mut EditorComponents) {
    e.register::<HumanoidIk>();
}

const TARGET: egui::Color32 = egui::Color32::from_rgb(120, 230, 160);
const BODY: egui::Color32 = egui::Color32::from_rgb(70, 150, 110);

fn drag(ui: &mut Ui, label: &str, v: &mut f32, speed: f64, range: std::ops::RangeInclusive<f32>, changed: &mut bool) {
    widgets::property_row(ui, label, |ui| *changed |= ui.add(DragValue::new(v).speed(speed).range(range)).changed());
}

fn note(ui: &mut Ui, s: &str) {
    ui.label(RichText::new(s).size(theme::SIZE_META).color(theme::TEXT_DIM));
}

impl Inspect for HumanoidIk {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        widgets::property_row(ui, "Source", |ui| {
            egui::ComboBox::from_id_salt("zestengine-humanoid-ik-source")
                .width(ui.available_width())
                .selected_text(self.source.label())
                .show_ui(ui, |ui| {
                    for s in IkSource::ALL {
                        changed |= ui.selectable_value(&mut self.source, s, s.label()).changed();
                    }
                });
        });
        match self.source {
            IkSource::Auto => note(ui, "Remote players' avatars follow their replicated trackers; everything else follows this machine's trackers."),
            IkSource::LocalTrackers => note(ui, "This machine's headset, controllers and trackers."),
            IkSource::Network => note(ui, "Targets from the Network Avatar on this object."),
            IkSource::Objects => note(ui, "Each tracker role follows a scene object."),
        }
        if matches!(self.source, IkSource::Auto | IkSource::LocalTrackers) {
            changed |= ctx.object_ref(ui, "Tracking root", &mut self.tracking_root);
            note(ui, "Unset: the scene's VR Character.");
        }
        if self.source == IkSource::Objects {
            self.objects.resize(TRACKER_SLOTS, zestengine_core::ObjectRef::NONE);
            for (k, r) in self.objects.iter_mut().enumerate() {
                changed |= ctx.object_ref(ui, TRACKER_ROLES[k], r);
            }
        }
        if !self.runtime.humanoid {
            widgets::info_note(ui, "No humanoid skeleton recognised on this object yet (needs hips and head). Add a bone map on the Avatar Descriptor if the names are unusual.");
        }
        let live: Vec<&str> =
            self.runtime.targets.slots().iter().zip(TRACKER_ROLES).filter(|(t, _)| t.is_some()).map(|(_, n)| n).collect();
        note(ui, &if live.is_empty() { "No trackers right now".to_owned() } else { format!("Tracking {}/11: {}", live.len(), live.join(", ")) });

        note(ui, "11-point tracking: headset + hands, hips, chest, elbows, knees and feet. Missing body trackers use the engine IK fallback.");
        widgets::eyebrow(ui, "Body");
        widgets::property_row(ui, "Locomotion", |ui| {
            use zestengine_core::TrackingLocomotion;
            egui::ComboBox::from_id_salt("tracking-locomotion")
                .selected_text(format!("{:?}", self.locomotion)).show_ui(ui, |ui| {
                    for value in [TrackingLocomotion::Auto, TrackingLocomotion::On, TrackingLocomotion::Off] {
                        changed |= ui.selectable_value(&mut self.locomotion, value, format!("{value:?}")).changed();
                    }
                });
        });
        note(ui, "Auto disables canned locomotion while hips and both feet are tracked. IK still drives the body in every mode.");
        drag(ui, "Weight", &mut self.weight, 0.005, 0.0..=1.0, &mut changed);
        drag(ui, "Spine twist", &mut self.spine_twist, 0.005, 0.0..=1.0, &mut changed);
        drag(ui, "Shoulder rotation", &mut self.shoulder_rotation, 0.005, 0.0..=1.0, &mut changed);

        widgets::eyebrow(ui, "Feet without trackers");
        widgets::property_row(ui, "Plant and step", |ui| changed |= ui.checkbox(&mut self.plant_feet, "").changed());
        if self.plant_feet {
            drag(ui, "Step distance (m)", &mut self.step_distance, 0.005, 0.02..=2.0, &mut changed);
            drag(ui, "Step angle (deg)", &mut self.step_angle, 0.5, 5.0..=180.0, &mut changed);
            drag(ui, "Step time (s)", &mut self.step_time, 0.005, 0.05..=2.0, &mut changed);
            drag(ui, "Step height (m)", &mut self.step_height, 0.002, 0.0..=0.5, &mut changed);
        }
        note(ui, "Players calibrate from the VR body menu (the Menu button on a VR Character) or gameplay: ctx.start_body_calibration(), or ctx.calibrate_trackers() while they hold a T-pose. The editor's Windows > Body Tracking sets tracker roles.");
        changed
    }
}

impl Gizmo for HumanoidIk {
    /// The live tracker targets (a ring and axis tick each, in the avatar's
    /// space) joined into a stick body.
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let slots = self.runtime.targets.slots();
        let mut targets = Vec::new();
        for (p, r) in slots.iter().flatten() {
            let n = 16;
            for k in 0..n {
                let a = k as f32 / n as f32 * std::f32::consts::TAU;
                let b = (k + 1) as f32 / n as f32 * std::f32::consts::TAU;
                let ring = |t: f32| *p + *r * Vec3::new(t.cos(), 0.0, t.sin()) * 0.04;
                targets.push((ring(a), ring(b)));
            }
            targets.push((*p, *p + *r * Vec3::Y * 0.08));
        }
        let mut body = Vec::new();
        let links = [(0, 6), (6, 3), (0, 3), (6, 7), (7, 1), (6, 8), (8, 2), (3, 9), (9, 4), (3, 10), (10, 5)];
        for (a, b) in links {
            if let (Some((pa, _)), Some((pb, _))) = (slots[a], slots[b]) {
                body.push((pa, pb));
            }
        }
        vec![
            GizmoSpec::Lines { segments: body, color: BODY, world: false },
            GizmoSpec::Lines { segments: targets, color: TARGET, world: false },
        ]
    }
}
