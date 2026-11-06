//! New Project dialog body: template picker cards with a Desktop / VR filter,
//! then the location fields. Built on the editor theme tokens and widget kit.

use std::path::Path;

use crate::{icons, theme, widgets};
use egui::{Ui, Vec2};

use zestengine_engine::templates::{ProjectTemplate, TEMPLATES, TemplateCategory, TemplateIcon};

pub(crate) struct NewProjectState<'a> {
    pub parent: &'a mut String,
    pub name: &'a mut String,
    pub template: &'a mut String,
    pub filter: &'a mut Option<TemplateCategory>,
}

const CARD_W: f32 = 200.0;
const CARD_H: f32 = 104.0;
const GAP: f32 = 8.0;
const COLUMNS: usize = 3;

/// The dialog contents. Returns true when "Create & Open" was clicked with
/// valid inputs; `cancel` is set when Cancel was clicked.
pub(crate) fn new_project_ui(ui: &mut Ui, s: NewProjectState, cancel: &mut bool) -> bool {
    let width = CARD_W * COLUMNS as f32 + GAP * (COLUMNS as f32 - 1.0);
    ui.set_width(width);

    ui.horizontal(|ui| {
        widgets::eyebrow(ui, "Template");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            widgets::segmented(
                ui,
                s.filter,
                &[(None, "All"), (Some(TemplateCategory::Desktop), "Desktop"), (Some(TemplateCategory::Vr), "VR")],
            );
        });
    });
    ui.add_space(6.0);

    let shown: Vec<&ProjectTemplate> =
        TEMPLATES.iter().filter(|t| s.filter.is_none_or(|f| t.category == f)).collect();
    for row in shown.chunks(COLUMNS) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = GAP;
            for t in row {
                if template_card(ui, t, *s.template == t.id).clicked() {
                    *s.template = t.id.to_string();
                }
            }
        });
        ui.add_space(GAP);
    }

    if let Some(t) = zestengine_engine::templates::find(s.template) {
        widgets::info_note(ui, &format!("{}: {}", t.name, t.features.join("  /  ")));
    }

    ui.add_space(theme::PAD);
    widgets::eyebrow(ui, "Location");
    let field_w = width - theme::LABEL_WIDTH - 8.0;
    widgets::property_row(ui, "Parent folder", |ui| {
        ui.add(egui::TextEdit::singleline(s.parent).desired_width(field_w))
    });
    widgets::property_row(ui, "Project name", |ui| ui.add(egui::TextEdit::singleline(s.name).desired_width(field_w)));
    let target = Path::new(s.parent.as_str()).join(s.name.as_str());
    ui.label(egui::RichText::new(target.display().to_string()).font(theme::mono(theme::SIZE_META)).color(theme::TEXT_DIM));

    let valid = !s.parent.trim().is_empty()
        && !s.name.trim().is_empty()
        && Path::new(s.parent.as_str()).is_dir()
        && zestengine_engine::templates::find(s.template).is_some();
    if !valid {
        ui.label(
            egui::RichText::new("Pick an existing parent folder, a name and a template.")
                .size(theme::SIZE_META)
                .color(theme::STATUS_WARN_FG),
        );
    } else if target.join("project.zestengine").exists() {
        ui.label(
            egui::RichText::new("A project already exists there.").size(theme::SIZE_META).color(theme::STATUS_WARN_FG),
        );
    }

    let mut create = false;
    widgets::dialog_footer(ui, |ui| {
        ui.add_enabled_ui(valid, |ui| {
            if widgets::primary_button_sized(ui, Some(icons::Icon::Plus), "Create & Open", theme::H_PRIMARY).clicked() {
                create = true;
            }
        });
        if widgets::secondary_button_sized(ui, None, "Cancel", theme::H_PRIMARY).clicked() {
            *cancel = true;
        }
    });
    create && valid
}

fn template_card(ui: &mut Ui, t: &ProjectTemplate, selected: bool) -> egui::Response {
    widgets::select_card(ui, Vec2::new(CARD_W, CARD_H), template_icon(t.icon), t.name, t.category.label(), t.description, selected)
        .on_hover_text(t.description)
}

/// The editor icon drawn on a template's card.
fn template_icon(icon: TemplateIcon) -> icons::Icon {
    use icons::Icon;
    match icon {
        TemplateIcon::Empty => Icon::Plus,
        TemplateIcon::Crosshair => Icon::Target,
        TemplateIcon::Camera => Icon::Camera,
        TemplateIcon::Pointer => Icon::ToolSelect,
        TemplateIcon::Grid => Icon::Grid,
        TemplateIcon::Headset => Icon::Gizmo,
        TemplateIcon::Social => Icon::Audio,
    }
}

#[cfg(test)]
#[path = "shots/dialogs.rs"]
mod dialog_shots;
