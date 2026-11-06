//! Project Settings > Appearance rows for the scene view's selection outline.

use crate::{theme, widgets};

/// Outline on/off, colour (the accent unless customised) and width in pixels.
/// Returns true when anything changed.
pub fn selection_outline_settings_ui(ui: &mut egui::Ui, enabled: &mut bool, color: &mut Option<[u8; 3]>, width: &mut f32) -> bool {
    let mut changed = false;
    widgets::property_row(ui, "Selection outline", |ui| {
        changed |= ui.checkbox(enabled, "").on_hover_text("Outline selected objects in the scene view").changed();
    });
    ui.add_enabled_ui(*enabled, |ui| {
        widgets::property_row(ui, "Outline colour", |ui| {
            let accent = theme::accent();
            let following = color.is_none();
            if ui.selectable_label(following, "Accent").on_hover_text("Follow the editor accent colour").clicked() {
                *color = if following { Some([accent.r(), accent.g(), accent.b()]) } else { None };
                changed = true;
            }
            let mut rgb = color.unwrap_or([accent.r(), accent.g(), accent.b()]);
            if ui.color_edit_button_srgb(&mut rgb).on_hover_text("Custom outline colour").changed() {
                *color = Some(rgb);
                changed = true;
            }
        });
        widgets::property_row(ui, "Outline width", |ui| {
            changed |= ui
                .add(egui::DragValue::new(width).range(1.0..=8.0).speed(0.05).suffix(" px"))
                .on_hover_text("Width of the selection outline in pixels (2 is crisp on a 1x display; scaled on HiDPI)")
                .changed();
        });
    });
    changed
}
