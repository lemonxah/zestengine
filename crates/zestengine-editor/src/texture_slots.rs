//! Material texture-slot validation UI: the warning box under a slot whose
//! texture's import settings don't match what the slot reads, with an
//! Auto-fix button or, when the same texture is used by slots that need
//! different settings, a menu (spec §4.1).
//!
//! The editor crate doesn't know about asset metadata. The engine computes
//! [`SlotWarning`]s and publishes them in egui temp data under
//! [`SLOT_WARNINGS_KEY`] before the inspector runs; `texture_row` draws them
//! and leaves the user's choice under [`SLOT_FIX_KEY`] for the engine to take.

use egui::{RichText, Ui};

use crate::{theme, widgets};

/// egui temp-data key holding `Vec<SlotWarning>` for the material on screen.
pub const SLOT_WARNINGS_KEY: &str = "zestengine-texture-slot-warnings";
/// egui temp-data key where a clicked fix is left: `(slot, SlotFix)`.
pub const SLOT_FIX_KEY: &str = "zestengine-texture-slot-fix";

/// One validation result for a material texture slot.
#[derive(Clone, Debug, PartialEq)]
pub struct SlotWarning {
    /// Slot index (binding order, same as `texture_row`'s `idx`).
    pub slot: usize,
    pub text: String,
    /// Informational (a saving) rather than a rendering mismatch.
    pub info: bool,
    /// Label of the single-click fix ("Auto-fix", "Use single channel").
    pub fix_label: String,
    /// Other places the texture is used with incompatible needs, as
    /// "owner (slot name)". Non-empty replaces the button with a menu.
    pub conflicts: Vec<String>,
    /// File name the "Duplicate" option would create.
    pub duplicate_name: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum SlotFix {
    /// Change the texture's import settings to match this slot.
    #[default]
    AutoFix,
    /// Same, accepting that the conflicting uses break.
    FixAnyway,
    /// Copy the file, import the copy for this slot and assign it here.
    Duplicate,
    /// Stop warning about this slot.
    Ignore,
}

/// The warning box for `warning`; returns the chosen fix.
pub fn slot_warning_ui(ui: &mut Ui, warning: &SlotWarning) -> Option<SlotFix> {
    let mut choice = None;
    let (fill, fg) = if warning.info { (theme::BG_APP, theme::TEXT_MUTED) } else { (theme::STATUS_WARN_BG, theme::STATUS_WARN_FG) };
    egui::Frame::new()
        .fill(fill)
        .corner_radius(egui::CornerRadius::same(theme::R_ROW))
        .inner_margin(egui::Margin::symmetric(10, 8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.add(egui::Label::new(RichText::new(&warning.text).size(theme::SIZE_META).color(fg)).wrap());
            ui.add_space(4.0);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if warning.conflicts.is_empty() {
                    if widgets::secondary_button_sized(ui, None, &warning.fix_label, theme::H_BUTTON_SMALL).clicked() {
                        choice = Some(SlotFix::AutoFix);
                    }
                } else {
                    ui.menu_button(RichText::new(format!("{}…", warning.fix_label)).size(theme::SIZE_BODY), |ui| {
                        widgets::menu_frame(ui);
                        if widgets::menu_item(ui, None, &format!("Duplicate as {} (recommended)", warning.duplicate_name), None).clicked() {
                            choice = Some(SlotFix::Duplicate);
                            ui.close();
                        }
                        let breaks = warning.conflicts.join(", ");
                        if widgets::menu_item(ui, None, &format!("Fix for this slot (breaks: {breaks})"), None).clicked() {
                            choice = Some(SlotFix::FixAnyway);
                            ui.close();
                        }
                        widgets::menu_separator(ui);
                        if widgets::menu_item(ui, None, "Ignore", None).clicked() {
                            choice = Some(SlotFix::Ignore);
                            ui.close();
                        }
                    });
                }
            });
        });
    ui.add_space(4.0);
    choice
}

/// Draw the warnings published for `slot`, leaving a clicked fix in temp data.
pub(crate) fn show_for_slot(ui: &mut Ui, slot: usize) {
    let warnings: Vec<SlotWarning> = ui.data(|d| d.get_temp(egui::Id::new(SLOT_WARNINGS_KEY))).unwrap_or_default();
    for w in warnings.iter().filter(|w| w.slot == slot) {
        if let Some(fix) = slot_warning_ui(ui, w) {
            ui.data_mut(|d| d.insert_temp(egui::Id::new(SLOT_FIX_KEY), (slot, fix)));
        }
    }
}
