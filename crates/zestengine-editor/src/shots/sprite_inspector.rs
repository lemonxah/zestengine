//! Headless screenshot of the texture inspector's Sprite section.

use super::*;
use zestengine_assets::sprite::SpriteMode;
use zestengine_assets::{TextureImport, TextureKind};
use crate::shot::Shot;

#[test]
fn sprite_section() {
    let dir = std::env::temp_dir().join("zestengine_shots");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("shot_sheet2.png");
    let mut img = image::RgbaImage::new(256, 128);
    for y in 10..110u32 {
        for x in 10..100u32 {
            img.put_pixel(x, y, image::Rgba([200, 90, 90, 255]));
        }
    }
    img.save(&path).unwrap();
    let def = |name: &str, x, y, w, h| SpriteDef { name: name.into(), rect: PixelRect { x, y, w, h }, ..Default::default() };
    let mut t = TextureImport {
        kind: TextureKind::Sprite,
        sprite: SpriteSheet {
            mode: SpriteMode::Multiple,
            sprites: vec![def("a", 10, 10, 90, 100), def("b", 120, 10, 60, 60)],
            ..Default::default()
        },
        ..Default::default()
    };
    let mut ui_state = SpriteUi::default();
    for mode in [SpriteMode::Multiple, SpriteMode::Single] {
        t.sprite.mode = mode;
        let mut shot = Shot::new(380.0, 1600.0, 1.5);
        shot.frames = 12;
        shot.frame_delay = std::time::Duration::from_millis(40);
        let img = shot.run_panel(|ui| {
            ui_state.show(ui, &path, &mut t, Some((256, 128)));
        });
        shot.save_named(&img, if mode == SpriteMode::Single { "sprite_section_single" } else { "sprite_section_multiple" });
        assert!(img.non_background() > 300);
    }
}
