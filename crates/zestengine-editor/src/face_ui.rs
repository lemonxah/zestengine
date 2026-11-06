//! Inspectors for `Face Tracking` and `Blend Shapes`.

use zestengine_core::{BlendShapes, FaceSource, FaceTracking, LipSync, LipSyncMode, VoiceChat, VoiceRange, VoiceRolloff, VoiceTransmit};
use egui::{DragValue, RichText, Ui};
use glam::Vec3;

use crate::components::{EditorComponents, Gizmo, GizmoSpec, Inspect, InspectCtx};
use crate::icons::Icon;
use crate::theme;
use crate::widgets;

pub(crate) fn register(e: &mut EditorComponents) {
    e.register::<FaceTracking>();
    e.register::<BlendShapes>();
    e.register::<LipSync>();
    e.register::<VoiceChat>();
}

fn note(ui: &mut Ui, s: &str) {
    ui.label(RichText::new(s).size(theme::SIZE_META).color(theme::TEXT_DIM));
}

impl Inspect for FaceTracking {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        widgets::property_row(ui, "Source", |ui| {
            egui::ComboBox::from_id_salt("zestengine-face-source").width(ui.available_width()).selected_text(self.source.label()).show_ui(ui, |ui| {
                for s in FaceSource::ALL {
                    changed |= ui.selectable_value(&mut self.source, s, s.label()).changed();
                }
            });
        });
        note(ui, "Local reads the headset's face tracking (WiVRn, Virtual Desktop, SteamVR) through OpenXR, else External tracking (OSC) when enabled in Project Settings.");
        let status = if self.runtime.active { "Tracking" } else { "No face data" };
        widgets::property_row(ui, "Status", |ui| ui.label(RichText::new(status).color(if self.runtime.active { theme::STATUS_OK } else { theme::TEXT_DIM })));
        widgets::property_row(ui, "Smoothing", |ui| changed |= ui.add(DragValue::new(&mut self.smoothing).speed(0.005).range(0.0..=0.98)).changed());
        widgets::property_row(ui, "Eye bones", |ui| changed |= ui.checkbox(&mut self.eye_bones, "").changed());
        if self.eye_bones {
            widgets::property_row(ui, "Eye yaw / pitch (deg)", |ui| {
                changed |= ui.add(DragValue::new(&mut self.eye_angle[0]).speed(0.5).range(0.0..=60.0)).changed();
                changed |= ui.add(DragValue::new(&mut self.eye_angle[1]).speed(0.5).range(0.0..=60.0)).changed();
            });
        }

        widgets::eyebrow(ui, &format!("Matched blendshapes ({} of {})", self.runtime.matched.len(), self.runtime.shapes.len()));
        if self.runtime.shapes.is_empty() {
            note(ui, "The model on this object has no blendshapes yet (or isn't loaded).");
        }
        egui::ScrollArea::vertical().id_salt("zestengine-face-matched").max_height(160.0).show(ui, |ui| {
            for (shape, expr) in &self.runtime.matched {
                let live = self.runtime.active.then(|| self.source_for(shape).map(|(s, _)| s.read(&self.runtime.frame))).flatten();
                ui.horizontal(|ui| {
                    ui.label(shape);
                    ui.label(RichText::new(format!("<- {expr}")).color(theme::TEXT_DIM));
                    if let Some(v) = live {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.add(egui::ProgressBar::new(v).desired_width(60.0));
                        });
                    }
                });
            }
        });

        widgets::eyebrow(ui, "Mappings");
        note(ui, "Drive a blendshape from a Unified Expressions name it doesn't match on its own.");
        let mut remove = None;
        for (k, (shape, expr)) in self.mappings.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt(("zestengine-face-map-shape", k)).width(140.0).selected_text(shape.as_str()).show_ui(ui, |ui| {
                    for s in &self.runtime.shapes {
                        if ui.selectable_label(s == shape, s).clicked() {
                            *shape = s.clone();
                            changed = true;
                        }
                    }
                });
                changed |= ui.add(egui::TextEdit::singleline(expr).desired_width(140.0).hint_text("JawOpen")).changed();
                if widgets::icon_button(ui, Icon::Trash, "Remove", false).clicked() {
                    remove = Some(k);
                }
            });
        }
        if let Some(k) = remove {
            self.mappings.remove(k);
            changed = true;
        }
        if widgets::dashed_button(ui, Some(Icon::Plus), "Add mapping").clicked() {
            self.mappings.push((self.runtime.shapes.first().cloned().unwrap_or_default(), String::new()));
            changed = true;
        }
        changed
    }
}

impl Gizmo for FaceTracking {
    /// The live gaze of each eye as a line from the object's head height.
    fn gizmos(&self) -> Vec<GizmoSpec> {
        let f = &self.runtime.frame;
        if !self.runtime.active || !f.has_gaze {
            return Vec::new();
        }
        let mut s = Vec::new();
        for (x, g) in [(0.03f32, f.gaze_left), (-0.03, f.gaze_right)] {
            let origin = Vec3::new(x, 1.6, 0.0);
            let dir = Vec3::new(-g.x * 0.5, g.y * 0.5, 1.0).normalize();
            s.push((origin, origin + dir * 0.3));
        }
        vec![GizmoSpec::Lines { segments: s, color: theme::STATUS_INFO, world: false }]
    }
}

impl Inspect for BlendShapes {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        if self.runtime.names.is_empty() {
            note(ui, "The model on this object has no blendshapes yet (or isn't loaded).");
        }
        note(ui, "Face Tracking overrides the shapes it drives while it tracks.");
        let names = std::mem::take(&mut self.runtime.names);
        // Only the rows in view are laid out: a face model has hundreds of shapes.
        egui::ScrollArea::vertical().id_salt("zestengine-blend-shapes").max_height(320.0).show_rows(ui, crate::theme::H_ROW, names.len(), |ui, range| {
            for name in &names[range] {
                let mut w = self.get(name);
                widgets::property_row(ui, name.as_str(), |ui| {
                    if crate::widgets::slider(ui, egui::Slider::new(&mut w, 0.0..=1.0).show_value(true)).changed() {
                        crate::blendshape_ui::set_weight(self, name, w);
                        changed = true;
                    }
                });
            }
        });
        self.runtime.names = names;
        changed
    }
}

impl Gizmo for BlendShapes {}

impl Inspect for LipSync {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        widgets::property_row(ui, "Mode", |ui| {
            egui::ComboBox::from_id_salt("zestengine-lipsync-mode").width(ui.available_width()).selected_text(self.mode.label()).show_ui(ui, |ui| {
                for m in LipSyncMode::ALL {
                    changed |= ui.selectable_value(&mut self.mode, m, m.label()).changed();
                }
            });
        });
        widgets::property_row(ui, "Smoothing", |ui| changed |= ui.add(DragValue::new(&mut self.smoothing).speed(0.005).range(0.0..=0.98)).changed());
        widgets::property_row(ui, "Gain", |ui| changed |= ui.add(DragValue::new(&mut self.gain).speed(0.01).range(0.0..=4.0)).changed());
        widgets::property_row(ui, "Drive face shapes", |ui| changed |= ui.checkbox(&mut self.drive_face_shapes, "").changed());
        note(ui, "Without viseme blendshapes (v_aa, VRM A I U E O), the jaw and lips are moved through the Unified Expressions shapes instead.");
        let status = if self.runtime.active { "Speaking" } else { "Quiet" };
        widgets::property_row(ui, "Status", |ui| ui.label(RichText::new(status).color(if self.runtime.active { theme::STATUS_OK } else { theme::TEXT_DIM })));
        widgets::eyebrow(ui, &format!("Viseme blendshapes ({})", self.runtime.matched.len()));
        for (shape, viseme) in &self.runtime.matched {
            let i = zestengine_core::VISEMES.iter().position(|v| v == viseme).unwrap_or(0);
            ui.horizontal(|ui| {
                ui.label(shape);
                ui.label(RichText::new(format!("<- {viseme}")).color(theme::TEXT_DIM));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add(egui::ProgressBar::new(self.runtime.frame.get(i)).desired_width(60.0));
                });
            });
        }
        changed
    }
}

impl Gizmo for LipSync {}

impl Inspect for VoiceChat {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        widgets::property_row(ui, "Enabled", |ui| changed |= ui.checkbox(&mut self.enabled, "").changed());
        widgets::property_row(ui, "Range", |ui| {
            let opts = [(VoiceRange::Proximity, "Proximity"), (VoiceRange::Global, "Global")];
            changed |= widgets::segmented(ui, &mut self.range, &opts).changed();
        });
        if self.range == VoiceRange::Proximity {
            widgets::property_row(ui, "Rolloff", |ui| {
                let opts = [(VoiceRolloff::Logarithmic, "Natural"), (VoiceRolloff::Linear, "Linear")];
                changed |= widgets::segmented(ui, &mut self.rolloff, &opts).changed();
            });
            widgets::property_row(ui, "Full volume within (m)", |ui| changed |= ui.add(DragValue::new(&mut self.min_distance).speed(0.05).range(0.01..=100.0)).changed());
            widgets::property_row(ui, "Silent past (m)", |ui| changed |= ui.add(DragValue::new(&mut self.max_distance).speed(0.1).range(0.1..=1000.0)).changed());
        }
        widgets::property_row(ui, "Transmit", |ui| {
            let opts = [(VoiceTransmit::VoiceActivity, "Voice activity"), (VoiceTransmit::PushToTalk, "Push to talk"), (VoiceTransmit::AlwaysOn, "Always")];
            changed |= widgets::segmented(ui, &mut self.transmit, &opts).changed();
        });
        match self.transmit {
            VoiceTransmit::PushToTalk => {
                widgets::property_row(ui, "Push-to-talk action", |ui| changed |= ui.text_edit_singleline(&mut self.push_to_talk_action).changed());
            }
            VoiceTransmit::VoiceActivity => {
                widgets::property_row(ui, "Activation (dBFS)", |ui| changed |= ui.add(DragValue::new(&mut self.activation_db).speed(0.5).range(-80.0..=0.0)).changed());
            }
            VoiceTransmit::AlwaysOn => {}
        }
        widgets::property_row(ui, "Mic gain", |ui| changed |= ui.add(DragValue::new(&mut self.mic_gain).speed(0.01).range(0.0..=8.0)).changed());
        widgets::property_row(ui, "Bitrate (bit/s)", |ui| changed |= ui.add(DragValue::new(&mut self.bitrate).speed(100.0).range(6_000..=64_000)).changed());
        note(ui, "Opus with loss recovery when libopus is installed, else uncompressed. Each player's buffer adapts to their connection's jitter (40 to 300 ms).");
        changed
    }
}

impl Gizmo for VoiceChat {
    /// The proximity range: full volume inside the inner sphere, silent past
    /// the outer one.
    fn gizmos(&self) -> Vec<GizmoSpec> {
        if self.range != VoiceRange::Proximity {
            return Vec::new();
        }
        vec![
            GizmoSpec::Range { center: Vec3::ZERO, radius: self.min_distance, color: theme::STATUS_OK },
            GizmoSpec::Range { center: Vec3::ZERO, radius: self.max_distance, color: theme::STATUS_INFO },
        ]
    }

    fn apply_gizmo_edit(&mut self, edit: crate::components::GizmoEdit) {
        if let crate::components::GizmoEdit::Range { index, radius } = edit {
            match index {
                0 => self.min_distance = radius.max(0.01),
                _ => self.max_distance = radius.max(self.min_distance + 0.01),
            }
        }
    }
}

