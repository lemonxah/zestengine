//! Headless screenshot of the Skinning Editor window with a small rig.

use super::*;
use zestengine_assets::sprite::PixelRect;
use zestengine_assets::rig2d::Rig2D;
use crate::shot::Shot;

#[test]
fn skinning_editor_window() {
    let dir = std::env::temp_dir().join("zestengine_shots");
    let _ = std::fs::create_dir_all(&dir);
    let mut img = image::RgbaImage::new(64, 128);
    for y in 8..120u32 {
        for x in 20..44u32 {
            img.put_pixel(x, y, image::Rgba([200, 150, 90, 255]));
        }
    }
    let rgba = img.clone();
    let alpha = AlphaMap::from_rgba8(64, 128, rgba.as_raw());
    let tex = {
        let ctx = egui::Context::default();
        let color = egui::ColorImage::from_rgba_unmultiplied([64, 128], rgba.as_raw());
        ctx.load_texture("shot", color, egui::TextureOptions::NEAREST)
    };
    let mut w = SkinningEditorWindow::default();
    let mut model = SkinningModel::new(Rig2D::default());
    let hip = model.create_bone(None, [0.0, -0.5], [0.0, 0.0]);
    let spine = model.create_bone(Some(hip), [0.0, 0.0], [0.0, 0.5]);
    let _ = spine;
    let sprite = SpriteDef { name: "torso".into(), rect: PixelRect { x: 0, y: 0, w: 64, h: 128 }, ..Default::default() };
    model.generate_geometry(&alpha, sprite.rect, [0.5, 0.5], 100.0, "torso", &w.geo, &w.weights);
    w.model = Some(model);
    w.sprites = vec![sprite];
    w.open = true;
    w.rig_path = Some(dir.join("shot.rig2d"));
    w.sheet_path = dir.join("shot_skin.png").to_string_lossy().into_owned();
    w.status = "Loaded".into();
    let _ = (tex, alpha);
    let mut shot = Shot::new(1000.0, 660.0, 1.25);
    let img = shot.run(|ctx| w.show(ctx));
    shot.save_named(&img, "skinning_editor");
    assert!(img.non_background() > 500);
}

#[test]
fn skinning_editor_weight_sliders() {
    let dir = std::env::temp_dir().join("zestengine_shots");
    let _ = std::fs::create_dir_all(&dir);
    let mut img = image::RgbaImage::new(64, 128);
    for y in 8..120u32 {
        for x in 20..44u32 {
            img.put_pixel(x, y, image::Rgba([200, 150, 90, 255]));
        }
    }
    let alpha = AlphaMap::from_rgba8(64, 128, img.as_raw());
    let mut w = SkinningEditorWindow::default();
    let mut model = SkinningModel::new(Rig2D::default());
    let hip = model.create_bone(None, [0.0, -0.5], [0.0, 0.0]);
    model.create_bone(Some(hip), [0.0, 0.0], [0.0, 0.5]);
    model.rename_bone(0, "Hip");
    model.rename_bone(1, "Spine");
    let sprite = SpriteDef { name: "torso".into(), rect: PixelRect { x: 0, y: 0, w: 64, h: 128 }, ..Default::default() };
    model.generate_geometry(&alpha, sprite.rect, [0.5, 0.5], 100.0, "torso", &w.geo, &w.weights);
    let n = model.part().map_or(0, |p| p.mesh.positions.len() as u32);
    model.select_vertices(&(0..n.min(6)).collect::<Vec<_>>(), false);
    w.model = Some(model);
    w.sprites = vec![sprite];
    w.open = true;
    w.tool = Tool::Weights;
    w.rig_path = Some(dir.join("shot.rig2d"));
    w.sheet_path = dir.join("shot_skin.png").to_string_lossy().into_owned();
    w.status = "Loaded".into();
    let mut shot = Shot::new(1000.0, 700.0, 1.25);
    let img = shot.run(|ctx| w.show(ctx));
    shot.save_named(&img, "skinning_editor_weights");
    assert!(img.non_background() > 500);
}
