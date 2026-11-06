//! Headless screenshots of the material inspector (blend / grab / stencil).

use crate::shot::Shot;
use zestengine_render::{MaterialFeatures, MaterialParams};

#[test]
fn material_inspector() {
    let mut m = zestengine_engine::scene::model_from_material("Glass", &MaterialParams::default(), &MaterialFeatures::default(), false);
    m.alpha_mode = crate::AlphaModeModel::Blend;
    m.grab = true;
    m.stencil.two_sided = true;
    m.stencil.reference = 2;
    let mut search = String::new();
    let mut dropped = None;
    let mut shot = Shot::new(380.0, 4000.0, 1.5);
    let img = shot.run_panel(|ui| {
        crate::material_editor_ui(ui, &mut search, &mut m, &["standard"], None, &mut dropped, "shot");
    });
    shot.save_named(&img, "material_standard");
    assert!(img.non_background() > 500);
}

/// Click the shader-select dropdown and pick a second shader: the material's
/// shader must change (regression: dropdowns in the shader UI dead).
#[test]
fn shader_select_dropdown_switches_shader() {
    let mut m = zestengine_engine::scene::model_from_material("Base", &MaterialParams::default(), &MaterialFeatures::default(), false);
    let mut search = String::new();
    let mut dropped = None;
    let mut shot = Shot::new(380.0, 900.0, 1.5);
    let mut draw = |ui: &mut egui::Ui| {
        crate::material_editor_ui(ui, &mut search, &mut m, &["standard", "toon"], None, &mut dropped, "shot");
    };
    // Pass 1: layout. The shader-select combo is the first row's right half,
    // right of the "Shader" label (label ~55px + spacing).
    shot.run_panel(&mut draw);
    let combo = egui::pos2(8.0 + 55.0 + 8.0 + 40.0, 6.0 + 12.0);
    // Pass 2: click the combo (opens the popup on release).
    shot.clicks = vec![combo];
    shot.run_panel(&mut draw);
    // Pass 3 (gap passes): the popup is now registered; click its second
    // entry ("toon").
    shot.clicks = vec![egui::pos2(combo.x, combo.y + 24.0 + 24.0)];
    shot.run_panel(&mut draw);
    assert_eq!(m.shader, "toon", "clicking a dropdown entry must switch the material's shader");
}

/// Two material editors on screen at once (several materials share one mesh):
/// each dropdown must drive only its own material. The per-instance id salt
/// keeps their popup states apart.
#[test]
fn two_material_editors_keep_their_dropdowns_apart() {
    let mut m1 = zestengine_engine::scene::model_from_material("A", &MaterialParams::default(), &MaterialFeatures::default(), false);
    let mut m2 = zestengine_engine::scene::model_from_material("B", &MaterialParams::default(), &MaterialFeatures::default(), false);
    let mut search = String::new();
    let mut dropped = None;
    let mut shot = Shot::new(380.0, 2000.0, 1.5);
    let mut draw = |ui: &mut egui::Ui| {
        crate::material_editor_ui(ui, &mut search, &mut m1, &["standard", "toon"], None, &mut dropped, "mat-a");
        crate::material_editor_ui(ui, &mut search, &mut m2, &["standard", "toon"], None, &mut dropped, "mat-b");
    };
    shot.run_panel(&mut draw);
    // The first editor's shader-select combo is its first row, right of the label.
    let combo = egui::pos2(8.0 + 55.0 + 8.0 + 40.0, 6.0 + 12.0);
    shot.clicks = vec![combo];
    shot.run_panel(&mut draw);
    shot.clicks = vec![egui::pos2(combo.x, combo.y + 24.0 + 24.0)];
    shot.run_panel(&mut draw);
    assert_eq!(m1.shader, "toon", "the first material's dropdown selection applies");
    assert_eq!(m2.shader, "standard", "the second material's state is untouched");
}
