//! Wires the SDF text system to a host: built-in fonts, the project font
//! loader, and the handles UI painting reads.

use std::path::Path;
use std::sync::Arc;

use zestengine_render::Renderer;
use zestengine_text::system::FsLoader;

use crate::game_ui::GameUi;

/// Register the built-in fonts (the same faces egui uses, so text looks the
/// same as the legacy path) and point the font loader at the project. Cheap
/// and idempotent: call every frame.
pub fn init(renderer: &Renderer, assets_root: &Path) {
    let handle = renderer.text_handle();
    let Ok(mut sys) = handle.system.lock() else { return };
    sys.set_fs_root(assets_root, |root| Box::new(FsLoader { root: root.to_path_buf() }));
    if sys.has_fonts() {
        return;
    }
    let defs = egui::FontDefinitions::default();
    let order = defs.families.get(&egui::FontFamily::Proportional).cloned().unwrap_or_default();
    for name in order {
        if let Some(data) = defs.font_data.get(&name) {
            sys.add_builtin_font(&name, data.font.to_vec());
        }
    }
}

/// Point a `GameUi` at the text system (or at egui text when it is
/// unavailable) and at the string tables.
pub fn bind(ui: &mut GameUi, renderer: &Renderer, assets_root: &Path, strings: Option<Arc<zestengine_core::Localization>>) {
    init(renderer, assets_root);
    ui.text = renderer.text_available().then(|| renderer.text_handle());
    ui.strings = strings;
}
