//! Inspector for the `Wasm Script` component: the module slot, and what the
//! loaded module declares (its actions, properties and capabilities), so the
//! author can see what other objects' bindings will be offered.

use egui::{RichText, Ui};
use zestengine_core::{Component, WasmScript};

use crate::components::{EditorComponents, Gizmo, Inspect, InspectCtx};
use crate::{theme, widgets};

pub(crate) fn register(e: &mut EditorComponents) {
    e.register::<WasmScript>();
}

impl Inspect for WasmScript {
    fn inspector_ui(&mut self, ui: &mut Ui, ctx: &InspectCtx) -> bool {
        let changed = ctx.file_field(ui, "Module", &mut self.module, "logic/door.wasm", &["wasm"]);
        // Reading the status also re-checks the file, so edits show up while the
        // inspector is open (the manifest is re-read on every hot reload).
        let status = self.status();
        if let Some(error) = &status.error {
            ui.label(RichText::new(error).size(theme::SIZE_META).color(theme::STATUS_ERROR));
        } else if !self.module.trim().is_empty() {
            let title = if status.name.is_empty() { "(unnamed)" } else { status.name.as_str() };
            ui.label(RichText::new(title).size(theme::SIZE_META).color(theme::TEXT_DIM));
            if status.untrusted {
                widgets::info_note(ui, "From untrusted content: only ugc_safe actions are available, and the sandbox limits are tighter.");
            }
            let this: &dyn Component = &*self;
            for a in Component::actions(this) {
                let params: Vec<String> = a.params.iter().map(|p| format!("{}: {}", p.name, p.ty.label())).collect();
                let flag = if a.ugc_safe { "" } else { "  (not UGC-safe)" };
                ui.label(RichText::new(format!("{}({}){flag}", a.name, params.join(", "))).size(theme::SIZE_META));
            }
            for p in Component::properties(this) {
                ui.label(RichText::new(format!("property {}: {}", p.name, p.ty.label())).size(theme::SIZE_META).color(theme::TEXT_DIM));
            }
            if !status.capabilities.is_empty() {
                ui.label(RichText::new(format!("capabilities: {}", status.capabilities.join(", "))).size(theme::SIZE_META).color(theme::TEXT_DIM));
            }
        }
        if !self.module.trim().is_empty() && ui.small_button("Reload module").clicked() {
            self.force_reload();
        }
        widgets::info_note(ui, "Bind its actions from an Interactable, Trigger Events or UI Button: pick this component in the function dropdown.");
        changed
    }
}
impl Gizmo for WasmScript {}
