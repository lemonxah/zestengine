//! Headless screenshots of the Sprite Editor window.

use super::*;
use crate::shot::Shot;

/// A 256x128 sheet: a few coloured blobs with transparent gaps.
fn sheet_png() -> PathBuf {
    let dir = std::env::temp_dir().join("zestengine_shots");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("shot_sheet.png");
    let mut img = image::RgbaImage::new(256, 128);
    let blobs = [(8u32, 8u32, 56u32, 40u32, [220u8, 90, 70]), (80, 16, 48, 48, [90, 200, 110]), (150, 8, 90, 100, [90, 140, 230]), (20, 70, 100, 50, [230, 200, 80])];
    for (x0, y0, w, h, c) in blobs {
        for y in y0..y0 + h {
            for x in x0..x0 + w {
                let (dx, dy) = ((x - x0) as f32 / w as f32 - 0.5, (y - y0) as f32 / h as f32 - 0.5);
                if dx * dx + dy * dy < 0.25 {
                    img.put_pixel(x, y, image::Rgba([c[0], c[1], c[2], 255]));
                }
            }
        }
    }
    img.save(&path).unwrap();
    path
}

fn import() -> TextureImport {
    let def = |name: &str, x, y, w, h| SpriteDef { name: name.into(), rect: PixelRect { x, y, w, h }, ..Default::default() };
    TextureImport {
        kind: TextureKind::Sprite,
        sprite: SpriteSheet {
            mode: SpriteMode::Multiple,
            sprites: vec![def("hero_idle", 8, 8, 56, 40), def("hero_run", 80, 16, 48, 48), def("big_prop", 150, 8, 90, 100), def("coin", 20, 70, 100, 50)],
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn sprite_editor_window() {
    let path = sheet_png();
    for (name, mode) in [("sprite_editor_sprites", Mode::Sprites), ("sprite_editor_outline", Mode::Outline), ("sprite_editor_physics", Mode::Physics)] {
        let mut w = SpriteEditorWindow::default();
        w.open(&path);
        w.mode = mode;
        let mut t = import();
        let mut shot = Shot::new(1200.0, 720.0, 1.0);
        shot.frames = 12;
        shot.frame_delay = std::time::Duration::from_millis(40);
        let img = shot.run(|ctx| {
            w.show(ctx, &path, &mut t, false);
        });
        shot.save_named(&img, name);
        assert!(img.non_background() > 500);
    }
}
