//! The `.physmat` physics material editor and Project Settings > Physics.

use zestengine_core::{CombineMode, PhysicsMaterial, PhysicsSettings};
use egui::{DragValue, RichText, Ui};

use crate::theme;
use crate::widgets;

fn drag(ui: &mut Ui, label: &str, v: &mut f32, speed: f64, range: std::ops::RangeInclusive<f32>, tip: &str) -> bool {
    let mut c = false;
    widgets::property_row(ui, label, |ui| c |= ui.add(DragValue::new(v).speed(speed).range(range)).on_hover_text(tip).changed());
    c
}

fn combine(ui: &mut Ui, label: &str, mode: &mut CombineMode, salt: &str) -> bool {
    let mut c = false;
    widgets::property_row(ui, label, |ui| {
        egui::ComboBox::from_id_salt(salt).selected_text(mode.label()).show_ui(ui, |ui| {
            for m in CombineMode::ALL {
                c |= ui.selectable_value(mode, m, m.label()).changed();
            }
        });
    });
    c
}

fn note(ui: &mut Ui, s: &str) {
    ui.label(RichText::new(s).size(theme::SIZE_META).color(theme::TEXT_DIM));
}

/// Edit a physics material in place; true when anything changed.
pub fn physmat_editor_ui(ui: &mut Ui, m: &mut PhysicsMaterial) -> bool {
    let mut c = false;
    widgets::property_row(ui, "Preset", |ui| {
        egui::ComboBox::from_id_salt("zestengine-physmat-preset").selected_text("Apply...").show_ui(ui, |ui| {
            for (name, preset) in PhysicsMaterial::PRESETS {
                if ui.selectable_label(false, name).clicked() {
                    *m = preset;
                    c = true;
                }
            }
        });
    });
    widgets::eyebrow(ui, "Contact");
    c |= drag(ui, "Friction", &mut m.friction, 0.01, 0.0..=4.0, "While sliding: 0 = ice, ~0.5 wood, 1+ rubber");
    let mut separate = m.static_friction.is_some();
    crate::widgets::property_row(ui, "Static friction", |ui| {
        if ui
            .checkbox(&mut separate, "")
            .on_hover_text("Its own coefficient while at rest: it takes more to start something sliding than to keep it going. Off = the same as Friction")
            .changed()
        {
            m.static_friction = separate.then_some(m.friction * 1.2);
            c = true;
        }
        if let Some(s) = m.static_friction.as_mut() {
            c |= ui.add(egui::DragValue::new(s).speed(0.01).range(0.0..=4.0)).changed();
        }
    });
    c |= combine(ui, "Friction combine", &mut m.friction_combine, "zestengine-physmat-fric");
    c |= drag(ui, "Bounciness", &mut m.bounciness, 0.01, 0.0..=1.0, "Share of speed kept on a bounce");
    c |= combine(ui, "Bounce combine", &mut m.bounce_combine, "zestengine-physmat-bounce");
    c |= drag(ui, "Contact offset (m)", &mut m.contact_offset, 0.001, 0.0..=0.2, "Touch this far before overlapping (thin or fast shapes)");
    note(ui, "When two materials use different combine modes the stronger wins: Maximum, Multiply, Minimum, Average.");
    widgets::eyebrow(ui, "Mass");
    c |= drag(ui, "Density (kg/m3)", &mut m.density, 10.0, 1.0..=25000.0, "Water 1000, wood ~600, steel ~7850");
    note(ui, "Used by Rigid Bodies whose Mass is 0; a set Mass wins.");
    widgets::eyebrow(ui, "Water");
    c |= drag(ui, "Buoyancy", &mut m.buoyancy, 0.02, 0.0..=10.0, "Scales the lift from the water this body displaces");
    c |= drag(ui, "Water drag", &mut m.water_drag, 0.02, 0.0..=10.0, "Scales the water's drag on the body");
    widgets::info_note(ui, "Contact stiffness is a solver setting for every contact, under Project Settings > Physics.");
    c
}

/// Project Settings > Physics. `materials` lists the project's `.physmat`
/// files (project-relative) for the default-material picker.
pub fn physics_settings_ui(ui: &mut Ui, s: &mut PhysicsSettings, materials: impl FnOnce() -> Vec<String>) -> bool {
    let mut c = false;
    widgets::property_row(ui, "Gravity", |ui| {
        for (v, axis) in s.gravity.iter_mut().zip(["X", "Y", "Z"]) {
            c |= ui.add(DragValue::new(v).speed(0.05).prefix(format!("{axis} "))).changed();
        }
    });
    widgets::property_row(ui, "Default material", |ui| {
        let shown = if s.default_material.is_empty() { "None (built-in)" } else { s.default_material.as_str() };
        egui::ComboBox::from_id_salt("zestengine-physics-default-mat").selected_text(shown).show_ui(ui, |ui| {
            c |= ui.selectable_value(&mut s.default_material, String::new(), "None (built-in)").changed();
            for m in materials() {
                let label = m.clone();
                c |= ui.selectable_value(&mut s.default_material, m, label).changed();
            }
        });
    });
    c |= drag(ui, "Contact stiffness (Hz)", &mut s.contact_stiffness, 0.5, 1.0..=1000.0, "Higher pushes overlaps apart harder; lower sinks in and settles softly");
    c |= drag(ui, "Contact damping", &mut s.contact_damping, 0.05, 0.0..=100.0, "Higher absorbs the push-back, lower springs");
    widgets::property_row(ui, "Solver iterations", |ui| {
        c |= ui.add(DragValue::new(&mut s.solver_iterations).range(1..=32)).on_hover_text("More is steadier stacks and joints").changed();
    });
    note(ui, "Applies from the next Play.");
    c
}
