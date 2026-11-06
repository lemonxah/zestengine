//! Editor fields for a controller state's behaviours and motion-time
//! parameter, kept apart from the graph and blend-tree editors.
//!
//! One hook: [`state_behaviour_fields`], called from wherever the state's
//! inspector is drawn. It edits `AnimState::time_parameter`,
//! `AnimState::behaviours` and `AnimState::behaviour_settings` (the Parameter
//! Driver's entries are edited in place; other registered behaviours are
//! listed and removable but have no fields of their own).

use egui::Ui;
use zestengine_assets::{AnimParamKind, AnimParameter, AnimState};
use zestengine_core::{AvatarParamKind, DriverEntry, DriverOp, PARAMETER_DRIVER, ParamDriverSettings, Remap};
use crate::icons::Icon;
use crate::{theme, widgets};

/// Draw the Motion Time and Behaviours fields of `st`. `params` are the
/// controller's parameters. Returns true when anything changed.
pub fn state_behaviour_fields(ui: &mut Ui, id: egui::Id, st: &mut AnimState, params: &[AnimParameter]) -> bool {
    let mut changed = false;
    widgets::property_row(ui, "Motion time", |ui| {
        let shown = if st.time_parameter.is_empty() { "None".to_owned() } else { st.time_parameter.clone() };
        egui::ComboBox::from_id_salt(id.with("time")).width(ui.available_width()).selected_text(shown).show_ui(ui, |ui| {
            changed |= ui.selectable_value(&mut st.time_parameter, String::new(), "None").changed();
            for p in params.iter().filter(|p| p.kind == AnimParamKind::Float) {
                changed |= ui.selectable_value(&mut st.time_parameter, p.name.clone(), &p.name).changed();
            }
        });
    });
    if !st.time_parameter.is_empty() {
        widgets::info_note(ui, "A Float parameter picks the pose (0 to 1 through the motion). The state's speed is ignored.");
    }

    widgets::eyebrow(ui, "Behaviours");
    let mut remove = None;
    for (i, name) in st.behaviours.iter().enumerate() {
        ui.horizontal(|ui| {
            ui.label(name);
            if widgets::icon_button(ui, Icon::Close, "Remove behaviour", false).clicked() {
                remove = Some(i);
            }
        });
    }
    if let Some(i) = remove {
        let name = st.behaviours.remove(i);
        if !st.behaviours.contains(&name) {
            st.behaviour_settings.remove(&name);
        }
        changed = true;
    }
    ui.menu_button("Add behaviour", |ui| {
        widgets::menu_frame(ui);
        for name in zestengine_core::state_behaviour_names() {
            if !st.behaviours.contains(&name) && ui.button(&name).clicked() {
                st.behaviours.push(name);
                changed = true;
                ui.close();
            }
        }
    });
    if st.behaviours.iter().any(|b| b == PARAMETER_DRIVER) {
        let text = st.behaviour_settings.entry(PARAMETER_DRIVER.to_owned()).or_default();
        let mut settings = ParamDriverSettings::from_ron(text).unwrap_or_default();
        if driver_fields(ui, id.with("driver"), &mut settings, params) {
            *text = settings.to_ron();
            changed = true;
        }
    }
    changed
}

fn kind_of(p: &AnimParameter) -> AvatarParamKind {
    match p.kind {
        AnimParamKind::Float => AvatarParamKind::Float,
        AnimParamKind::Int => AvatarParamKind::Int,
        AnimParamKind::Bool | AnimParamKind::Trigger => AvatarParamKind::Bool,
    }
}

/// A combo over the controller's parameters. Picking one reports it through
/// `picked` so the caller can adopt its type.
fn param_pick(ui: &mut Ui, id: egui::Id, value: &mut String, params: &[AnimParameter], picked: &mut Option<AvatarParamKind>) -> bool {
    let mut changed = false;
    let shown = if value.is_empty() { "(choose)".to_owned() } else { value.clone() };
    egui::ComboBox::from_id_salt(id).width(110.0).selected_text(shown).show_ui(ui, |ui| {
        for p in params {
            if ui.selectable_value(value, p.name.clone(), &p.name).changed() {
                *picked = Some(kind_of(p));
                changed = true;
            }
        }
    });
    changed
}

fn driver_fields(ui: &mut Ui, id: egui::Id, d: &mut ParamDriverSettings, params: &[AnimParameter]) -> bool {
    let mut changed = false;
    widgets::eyebrow(ui, "Parameter Driver");
    widgets::info_note(ui, "Runs once each time the state is entered.");
    widgets::property_row(ui, "Local only", |ui| {
        changed |= ui.checkbox(&mut d.local_only, "").on_hover_text("Skip on other players' copies, so Add and Random cannot diverge").changed();
    });
    let mut remove = None;
    for (i, e) in d.entries.iter_mut().enumerate() {
        let eid = id.with(i);
        widgets::divider(ui);
        let mut picked = None;
        ui.horizontal(|ui| {
            changed |= param_pick(ui, eid.with("name"), &mut e.name, params, &mut picked);
            if let Some(k) = picked {
                e.kind = k;
            }
            let mut op = DriverOp::LABELS.iter().position(|l| *l == e.op.label()).unwrap_or(0);
            let before = op;
            egui::ComboBox::from_id_salt(eid.with("op")).width(70.0).selected_text(DriverOp::LABELS[op]).show_ui(ui, |ui| {
                for (k, l) in DriverOp::LABELS.iter().enumerate() {
                    ui.selectable_value(&mut op, k, *l);
                }
            });
            if op != before {
                e.op = DriverOp::default_for(DriverOp::LABELS[op]);
                changed = true;
            }
            if widgets::icon_button(ui, Icon::Close, "Remove entry", false).clicked() {
                remove = Some(i);
            }
        });
        ui.horizontal(|ui| match &mut e.op {
            DriverOp::Set { value } | DriverOp::Add { value } => {
                changed |= ui.add(egui::DragValue::new(value).speed(0.05).prefix("Value ")).changed();
            }
            DriverOp::Random { min, max, chance } => {
                if e.kind == AvatarParamKind::Bool {
                    changed |= ui.add(egui::DragValue::new(chance).speed(0.01).range(0.0..=1.0).prefix("Chance ")).changed();
                } else {
                    changed |= ui.add(egui::DragValue::new(min).speed(0.05).prefix("Min ")).changed();
                    changed |= ui.add(egui::DragValue::new(max).speed(0.05).prefix("Max ")).changed();
                }
            }
            DriverOp::Copy { source, .. } => {
                let mut ignore = None;
                changed |= param_pick(ui, eid.with("src"), source, params, &mut ignore);
            }
        });
        if let DriverOp::Copy { remap, .. } = &mut e.op {
            let mut on = remap.is_some();
            if ui.checkbox(&mut on, "Remap range").changed() {
                *remap = on.then(Remap::default);
                changed = true;
            }
            if let Some(r) = remap {
                ui.horizontal(|ui| {
                    for (v, p) in [(&mut r.src_min, "From "), (&mut r.src_max, "to "), (&mut r.dst_min, "into "), (&mut r.dst_max, "to ")] {
                        changed |= ui.add(egui::DragValue::new(v).speed(0.02).prefix(p)).changed();
                    }
                });
            }
        }
        let kinds = [AvatarParamKind::Bool, AvatarParamKind::Int, AvatarParamKind::Float];
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Type").size(theme::SIZE_META).color(theme::TEXT_DIM));
            for k in kinds {
                changed |= ui.selectable_value(&mut e.kind, k, k.label()).changed();
            }
        });
    }
    if let Some(i) = remove {
        d.entries.remove(i);
        changed = true;
    }
    if widgets::dashed_button(ui, Some(Icon::Plus), "Add entry").clicked() {
        d.entries.push(DriverEntry::default());
        changed = true;
    }
    changed
}
