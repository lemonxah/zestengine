//! Inspectors for `Constraint` and `Custom Properties`.

use egui::{DragValue, RichText, Ui};
use zestengine_core::{AimUp, Constraint, ConstraintKind, ConstraintSource, CustomProperties, CustomValue, ObjectRef};

use crate::components::{EditorComponents, Gizmo, Inspect, InspectCtx};
use crate::theme;
use crate::widgets;

pub(crate) fn register(e: &mut EditorComponents) {
    e.register::<Constraint>();
    e.register::<CustomProperties>();
}

fn note(ui: &mut Ui, s: &str) {
    ui.label(RichText::new(s).size(theme::SIZE_META).color(theme::TEXT_DIM));
}

fn axes(ui: &mut Ui, label: &str, a: &mut [bool; 3]) -> bool {
    let mut changed = false;
    widgets::property_row(ui, label, |ui| {
        for (v, n) in a.iter_mut().zip(["X", "Y", "Z"]) {
            changed |= ui.checkbox(v, n).changed();
        }
    });
    changed
}

fn vec3_row(ui: &mut Ui, label: &str, v: &mut [f32; 3], speed: f32) -> bool {
    let mut changed = false;
    widgets::property_row(ui, label, |ui| {
        for c in v.iter_mut() {
            changed |= ui.add(DragValue::new(c).speed(speed)).changed();
        }
    });
    changed
}

impl Inspect for Constraint {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let mut changed = false;
        widgets::property_row(ui, "Type", |ui| {
            egui::ComboBox::from_id_salt("zestengine-constraint-kind").selected_text(self.kind.label()).show_ui(ui, |ui| {
                for k in ConstraintKind::ALL {
                    changed |= ui.selectable_value(&mut self.kind, k, k.label()).changed();
                }
            });
        });
        widgets::property_row(ui, "Weight", |ui| {
            changed |= crate::widgets::slider(ui, egui::Slider::new(&mut self.weight, 0.0..=1.0)).changed();
        });
        if matches!(self.kind, ConstraintKind::Parent | ConstraintKind::Position) {
            changed |= axes(ui, "Position Axes", &mut self.translation_axes);
        }
        if matches!(self.kind, ConstraintKind::Parent | ConstraintKind::Rotation | ConstraintKind::Aim) {
            changed |= axes(ui, "Rotation Axes", &mut self.rotation_axes);
        }
        if matches!(self.kind, ConstraintKind::Parent | ConstraintKind::Scale) {
            changed |= axes(ui, "Scale Axes", &mut self.scale_axes);
        }
        if self.kind == ConstraintKind::Aim {
            changed |= vec3_row(ui, "Aim Axis", &mut self.aim_axis, 0.01);
            widgets::property_row(ui, "Up", |ui| {
                egui::ComboBox::from_id_salt("zestengine-constraint-up")
                    .selected_text(match self.aim_up {
                        AimUp::Scene => "Scene Up",
                        AimUp::ToObject => "Towards Object",
                        AimUp::AlignObject => "Align With Object",
                        AimUp::Vector => "World Vector",
                        AimUp::None => "None",
                    })
                    .show_ui(ui, |ui| {
                        for (m, n) in [(AimUp::Scene, "Scene Up"), (AimUp::ToObject, "Towards Object"), (AimUp::AlignObject, "Align With Object"), (AimUp::Vector, "World Vector"), (AimUp::None, "None")] {
                            changed |= ui.selectable_value(&mut self.aim_up, m, n).changed();
                        }
                    });
            });
            if self.aim_up == AimUp::Vector {
                changed |= vec3_row(ui, "Up Vector", &mut self.up_vector, 0.01);
            }
            if matches!(self.aim_up, AimUp::ToObject | AimUp::AlignObject) {
                changed |= ctx.object_ref(ui, "Up Object", &mut self.up_object);
            }
        }
        let mut offset_t = self.offset.translation;
        if matches!(self.kind, ConstraintKind::Parent | ConstraintKind::Position) && vec3_row(ui, "Offset", &mut offset_t, 0.01) {
            self.offset.translation = offset_t;
            changed = true;
        }
        ui.add_space(4.0);
        widgets::eyebrow(ui, "Sources");
        let mut remove = None;
        for (i, s) in self.sources.iter_mut().enumerate() {
            changed |= ctx.object_ref(ui, &format!("Source {}", i + 1), &mut s.target);
            widgets::property_row(ui, "  Weight", |ui| {
                changed |= ui.add(DragValue::new(&mut s.weight).speed(0.01).range(0.0..=1.0)).changed();
                if ui.small_button("Remove").clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            self.sources.remove(i);
            changed = true;
        }
        if ui.button("Add Source").clicked() {
            self.sources.push(ConstraintSource { target: ObjectRef::NONE, weight: 1.0, ..Default::default() });
            changed = true;
        }
        if !self.unresolved.is_empty() {
            note(ui, &format!("Not found when the model was placed: {}", self.unresolved.join(", ")));
        }
        note(ui, "The object's own transform is its rest pose; the constraint blends from it by Weight. Sources are read at the start of the frame.");
        changed
    }
}

impl Gizmo for Constraint {}

impl Inspect for CustomProperties {
    fn inspector_ui(&mut self, ui: &mut Ui, _ctx: &InspectCtx) -> bool {
        let mut changed = false;
        let mut remove = None;
        for (i, (name, value)) in self.values.iter_mut().enumerate() {
            widgets::property_row(ui, name.as_str(), |ui| {
                match value {
                    CustomValue::Float(v) => changed |= ui.add(DragValue::new(v).speed(0.01)).changed(),
                    CustomValue::Int(v) => changed |= ui.add(DragValue::new(v)).changed(),
                    CustomValue::Bool(v) => changed |= ui.checkbox(v, "").changed(),
                    CustomValue::Vec3(v) => {
                        for c in v.iter_mut() {
                            changed |= ui.add(DragValue::new(c).speed(0.01)).changed();
                        }
                    }
                }
                if ui.small_button("x").clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            self.values.remove(i);
            changed = true;
        }
        ui.horizontal(|ui| {
            for (label, v) in [("Float", CustomValue::Float(0.0)), ("Int", CustomValue::Int(0)), ("Bool", CustomValue::Bool(false)), ("Vector", CustomValue::Vec3([0.0; 3]))] {
                if ui.small_button(format!("+ {label}")).clicked() {
                    let mut n = 1;
                    while self.get(&format!("{label}{n}")).is_some() {
                        n += 1;
                    }
                    self.set(&format!("{label}{n}"), v);
                    changed = true;
                }
            }
        });
        note(ui, "Imported from a model's user properties; animation clips can drive them by name (a vector's lanes are Name.x, Name.y, Name.z).");
        changed
    }
}

impl Gizmo for CustomProperties {}
