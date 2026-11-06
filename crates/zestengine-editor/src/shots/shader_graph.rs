//! Headless screenshot of the Shader Graph tab with the default graph.

use super::*;
use crate::shot::Shot;

#[test]
fn shader_graph_tab() {
    let root = std::env::temp_dir().join("zestengine_shots").join("sg_project");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let mut tab = ShaderGraphTab::default();
    create_asset(&root, &root, &mut tab);
    let mut shot = Shot::new(1280.0, 720.0, 1.0);
    let img = shot.run(|ctx| {
        egui::CentralPanel::default().frame(egui::Frame::new().fill(crate::theme::BG_APP)).show(ctx, |ui| {
            tab.ui(ui, &root);
        });
    });
    shot.save_named(&img, "shader_graph_tab");
    assert!(img.non_background() > 1000);
}
