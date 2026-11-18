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

#[test]
fn shader_graph_tab_with_a_lit_graph_and_a_selected_parameter() {
    use zestengine_assets::{GraphInput, GraphNode, GraphNodeKind as K, GraphParam, GraphTextureSlot};
    let root = std::env::temp_dir().join("zestengine_shots").join("sg_lit_project");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    let mut tab = ShaderGraphTab::default();
    create_asset(&root, &root, &mut tab);
    let at = |x: f32, y: f32, kind: K| GraphNode { position: [x, y], kind };
    tab.graph.nodes = vec![
        at(-760.0, -160.0, K::Input { source: GraphInput::Uv }),
        at(-560.0, -160.0, K::Panner { uv: Some(0), tiling: [2.0, 2.0], speed: [0.1, 0.0] }),
        at(-340.0, -220.0, K::TextureSample { slot: GraphTextureSlot::Albedo, uv: Some(1) }),
        at(-340.0, -60.0, K::Parameter { name: "Tint".into(), param: GraphParam::Color { default: [1.0, 0.8, 0.6, 1.0] } }),
        at(-120.0, -160.0, K::Mul { a: 2, b: 3 }),
        at(-340.0, 80.0, K::Noise { uv: Some(1), scale: 6.0 }),
        at(-340.0, 200.0, K::Fresnel { power: 3.0, normal: None }),
        at(-120.0, 60.0, K::Parameter { name: "Roughness".into(), param: GraphParam::Float { default: 0.4, min: 0.0, max: 1.0 } }),
        at(140.0, -60.0, K::Surface { albedo: 4, normal: None, metallic: Some(5), roughness: Some(7), emission: None, alpha: None }),
    ];
    tab.canvas.select_only(7);
    let mut shot = Shot::new(1600.0, 900.0, 1.0);
    let img = shot.run(|ctx| {
        egui::CentralPanel::default().frame(egui::Frame::new().fill(crate::theme::BG_APP)).show(ctx, |ui| {
            tab.ui(ui, &root);
        });
    });
    shot.save_named(&img, "shader_graph_lit");
    assert!(img.non_background() > 1000);
    assert!(zestengine_assets::generate_glsl(&tab.graph).is_ok());
}
