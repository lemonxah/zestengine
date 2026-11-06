//! Headless screenshots of editor panels (see `zestengine_editor::shot`).
//! `ZESTENGINE_SHOT_DIR=docs/design/screens/raw cargo test -p zestengine-editor --test panel_shots`
//! writes the PNGs; without the variable the tests only check that each panel
//! draws something.

use zestengine_core::*;
use zestengine_editor::shot::{Shot, Shot8};
use zestengine_editor::{EditorComponents, components_ui};

fn comps(shot: &mut Shot, name: &str, w: f32, mut list: Vec<Box<dyn Component>>) -> Shot8 {
    let reg = ComponentRegistry::with_builtins();
    let ed = EditorComponents::with_builtins();
    shot.size = egui::vec2(w, 1400.0);
    let img = shot.run_panel(|ui| {
        components_ui(ui, &mut list, &reg, &ed, &[], &[], None);
    });
    assert!(img.non_background() > 200, "{name} drew nothing");
    shot.save_named(&img, name);
    img
}

#[test]
fn net_manager() {
    let mut shot = Shot::new(380.0, 1400.0, 2.0);
    comps(&mut shot, "net_manager", 380.0, vec![Box::new(NetworkManager::default())]);
}

#[allow(dead_code)] // full-window shot helper kept for panel work
fn window_shot(name: &str, w: f32, h: f32, body: impl FnMut(&egui::Context)) -> Shot8 {
    let mut shot = Shot::new(w, h, 2.0);
    let img = shot.run(body);
    assert!(img.non_background() > 200, "{name} drew nothing");
    shot.save_named(&img, name);
    img
}

fn panel(name: &str, w: f32, ui: impl FnMut(&mut egui::Ui)) -> Shot8 {
    panel_click(name, w, &[], ui)
}

fn panel_click(name: &str, w: f32, clicks: &[egui::Pos2], ui: impl FnMut(&mut egui::Ui)) -> Shot8 {
    let mut shot = Shot::new(w, 1400.0, 2.0);
    shot.clicks = clicks.to_vec();
    let img = shot.run_panel(ui);
    assert!(img.non_background() > 200, "{name} drew nothing");
    shot.save_named(&img, name);
    img
}

#[test]
fn net_manager_full() {
    let m = NetworkManager {
        topology: Topology::Dedicated,
        interest: InterestConfig { enabled: true, ..Default::default() },
        anticheat: AntiCheatConfig { provider: "eos".into(), require: true, ..Default::default() },
        ..Default::default()
    };
    let mut shot = Shot::new(380.0, 1400.0, 2.0);
    comps(&mut shot, "net_manager_full", 380.0, vec![Box::new(m), Box::new(PersistentState { per_player: true, ..Default::default() })]);
}

#[test]
fn blendshapes() {
    let names: Vec<String> = (0..800).map(|i| match i % 4 {
        0 => format!("Face_{}_Smile", i),
        1 => format!("Eye_{}_Blink", i),
        2 => format!("Mouth_{}_Open", i),
        _ => format!("Brow_{}_Up", i),
    }).collect();
    let mut list: Vec<Box<dyn Component>> = vec![Box::new(BlendShapes { weights: vec![(names[0].clone(), 0.6), (names[5].clone(), 1.0)], ..Default::default() })];
    panel_click("blendshapes_800", 380.0, &[egui::pos2(60.0, 22.0)], |ui| {
        zestengine_editor::blendshapes_section(ui, &names, &mut list);
    });
    let short: Vec<String> = ["Blink", "Smile", "Jaw Open", "Mouth_50_Open"].iter().map(|s| s.to_string()).collect();
    panel_click("blendshapes_short", 380.0, &[egui::pos2(60.0, 22.0)], |ui| {
        zestengine_editor::blendshapes_section(ui, &short, &mut list);
    });
}

#[test]
fn sprite_and_tile_components() {
    let mut shot = Shot::new(380.0, 1400.0, 2.0);
    comps(&mut shot, "sprite2d", 380.0, vec![Box::new(SpriteRenderer::default()), Box::new(SortingGroup::default()), Box::new(SpriteMask::default())]);
    comps(&mut shot, "tilemap", 380.0, vec![Box::new(Grid::default()), Box::new(Tilemap::default()), Box::new(TilemapRenderer::default()), Box::new(TilemapCollider2D::default())]);
    comps(&mut shot, "rig2d", 380.0, vec![Box::new(SpriteBone::default()), Box::new(SpriteSkin::default()), Box::new(SpriteLibraryRef::default()), Box::new(SpriteResolver::default()), Box::new(IkSolver2D::default())]);
    comps(&mut shot, "skin_rebound", 380.0, vec![Box::new(SkinRebound::default())]);
    let mut shape = SpriteShapeRenderer::default();
    let data = zestengine_core::sprite_shape::presets::preset_shape(zestengine_core::sprite_shape::presets::ShapePreset::GroundWithFill);
    zestengine_core::sprite_shape::presets::apply_preset(&mut shape, &data);
    shape.closed = true;
    shape.holes.push(zestengine_core::sprite_shape::ShapeHole::ngon([0.0, 0.5], 0.6, 6));
    shape.holes.push(zestengine_core::sprite_shape::ShapeHole::ngon([1.5, 0.5], 0.4, 4));
    comps(&mut shot, "sprite_shape_component", 380.0, vec![Box::new(shape)]);
}

#[test]
fn hierarchy_roles() {
    use zestengine_editor::{SceneObjectRow, ScenePanel, theme::ObjectKind};
    let row = |name: &str, parent: Option<usize>, kind, role, inh| SceneObjectRow { name: name.into(), parent, kind, enabled: true, scene_hidden: false, scene_locked: false, prefab: Default::default(), net_role: role, net_inherited: inh, bone: None };
    let both = NetRole::Both;
    let rows = vec![
        row("World", None, ObjectKind::Group, both, both),
        row("Directional Light", Some(0), ObjectKind::DirectionalLight, both, both),
        row("Player Camera With A Very Long Name Indeed", Some(0), ObjectKind::Camera, NetRole::Client, both),
        row("Server Spawner", Some(0), ObjectKind::Group, NetRole::Server, both),
        row("Spawn Point 1", Some(3), ObjectKind::Mesh, both, NetRole::Server),
        row("HUD", Some(0), ObjectKind::UiCanvas, NetRole::Client, both),
        row("Crate", Some(0), ObjectKind::Mesh, both, both),
    ];
    let mut panel = ScenePanel::new();
    let mut selected = Some(2usize);
    let mut shot = Shot::new(300.0, 260.0, 2.0);
    let img = shot.run(|ctx| {
        egui::CentralPanel::default().frame(egui::Frame::new().fill(zestengine_editor::theme::BG_PANEL)).show(ctx, |ui| {
            panel.ui(ui, &rows, &mut selected);
        });
    });
    shot.save_named(&img, "hierarchy_roles");
    assert!(img.non_background() > 200);
}

#[test]
fn hierarchy_view_state() {
    use zestengine_editor::{SceneObjectRow, ScenePanel, theme::ObjectKind};
    let row = |name: &str, parent: Option<usize>, kind, enabled, hidden, locked| SceneObjectRow { name: name.into(), parent, kind, enabled, scene_hidden: hidden, scene_locked: locked, prefab: Default::default(), net_role: NetRole::Both, net_inherited: NetRole::Both, bone: None };
    let rows = vec![
        row("World", None, ObjectKind::Group, true, false, false),
        row("Hidden Group", Some(0), ObjectKind::Group, true, true, false),
        row("Child of hidden", Some(1), ObjectKind::Mesh, true, false, false),
        row("Locked Props", Some(0), ObjectKind::Group, true, false, true),
        row("Crate (locked via parent)", Some(3), ObjectKind::Mesh, true, false, false),
        row("Disabled Lamp", Some(0), ObjectKind::PointLight, false, false, false),
        row("Disabled AND hidden", Some(0), ObjectKind::Mesh, false, true, false),
        row("Plain Crate", Some(0), ObjectKind::Mesh, true, false, false),
    ];
    let mut panel = ScenePanel::new();
    let mut selected = Some(7usize);
    let mut shot = Shot::new(320.0, 330.0, 2.0);
    let img = shot.run(|ctx| {
        egui::CentralPanel::default().frame(egui::Frame::new().fill(zestengine_editor::theme::BG_PANEL)).show(ctx, |ui| {
            panel.ui(ui, &rows, &mut selected);
        });
    });
    shot.save_named(&img, "hierarchy_view_state");
    assert!(img.non_background() > 200);
}

#[test]
fn tile_inspectors() {
    use zestengine_core::{NeighborReq, RuleOutput, RuleTile, RuleTransform, TileAsset, TilingRule};
    let mut tile = TileAsset::with_sprite("tiles/grass.png#0123456789abcdef");
    panel("tile_asset", 380.0, |ui| {
        zestengine_editor::tile_asset_ui(ui, &mut tile);
    });
    let mut rt = RuleTile { default_sprite: "tiles/wall.png#0123456789abcdef".into(), ..Default::default() };
    let mut r1 = TilingRule { transform: RuleTransform::Rotated, sprites: vec!["tiles/wall.png#aaaaaaaaaaaaaaaa".into()], ..Default::default() };
    r1.neighbors = vec![NeighborReq::This, NeighborReq::Any, NeighborReq::NotThis, NeighborReq::Any, NeighborReq::This, NeighborReq::NotThis, NeighborReq::Any, NeighborReq::This];
    let r2 = TilingRule { output: RuleOutput::Random, sprites: vec!["a#1".into(), "b#2".into()], ..Default::default() };
    rt.rules = vec![r1, r2, TilingRule::default()];
    panel("rule_tile", 380.0, |ui| {
        zestengine_editor::rule_tile_ui(ui, &mut rt);
    });
}

#[test]
fn sprite_shape_profile() {
    let mut profile = zestengine_core::sprite_shape::presets::preset_shape(zestengine_core::sprite_shape::presets::ShapePreset::GroundWithFill).profile;
    for (i, r) in profile.angle_ranges.iter_mut().enumerate() {
        r.sprites = vec![format!("shape/edge{i}.png#0123456789abcdef")];
    }
    panel("sprite_shape_profile", 380.0, |ui| {
        zestengine_editor::sprite_shape_profile_ui(ui, &mut profile);
    });
}

#[test]
fn sorting_layers() {
    let mut layers = zestengine_core::SortingLayers::default();
    layers.add("Background");
    layers.add("Props");
    layers.add("Foreground");
    layers.layers[1].locked = true;
    panel("sorting_layers", 380.0, |ui| {
        zestengine_editor::sorting_layers_ui(ui, &mut layers);
    });
}

#[test]
fn tile_palette_window() {
    use zestengine_editor::{TilePaletteState, palette_from_sprites, tile_palette_window, write_palette};
    let root = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("palette_shot");
    let _ = std::fs::remove_dir_all(&root);
    let sprites: Vec<(String, String)> = (0..14).map(|i| (format!("grass_{i}"), format!("tiles/sheet.png#{i:016x}"))).collect();
    let (files, palette) = palette_from_sprites(&sprites, "Tiles", 5);
    let rel = write_palette(&root, "Tiles", "Terrain", &files, &palette).unwrap();
    let mut shot = Shot::new(420.0, 560.0, 2.0);
    let maps = vec![(zestengine_core::ObjectId::new(), "Ground Tilemap".to_string()), (zestengine_core::ObjectId::new(), "Walls".to_string())];
    let target = maps[0].0;
    let mut st = TilePaletteState::default();
    st.open = true;
    st.project_root = root.clone();
    st.palette_rel = rel;
    st.target = Some(target);
    st.set(&shot.ctx);
    let img = shot.run(|ctx| tile_palette_window(ctx, &maps));
    shot.save_named(&img, "tile_palette");
    assert!(img.non_background() > 200);
}


/// A confirm dialog in an auto-sized window: records (left, right) of each footer button.
fn confirm_dialog(name: &str, title: &str, text: &str, labels: [&str; 3], danger_mid: bool) -> (Shot8, [egui::Rect; 3], egui::Rect) {
    use zestengine_editor::widgets;
    let mut shot = Shot::new(520.0, 260.0, 2.0);
    let rects = std::cell::RefCell::new([egui::Rect::NOTHING; 3]);
    let content = std::cell::Cell::new(egui::Rect::NOTHING);
    let img = shot.run(|ctx| {
        let r = egui::Window::new(title).collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
            ui.label(text);
            widgets::dialog_footer(ui, |ui| {
                rects.borrow_mut()[0] = widgets::primary_button(ui, None, labels[0]).rect;
                rects.borrow_mut()[1] = if danger_mid { widgets::danger_button(ui, None, labels[1]).rect } else { widgets::secondary_button(ui, None, labels[1]).rect };
                rects.borrow_mut()[2] = widgets::secondary_button(ui, None, labels[2]).rect;
            });
            content.set(ui.min_rect());
        });
        assert!(r.is_some());
    });
    shot.save_named(&img, name);
    assert!(img.non_background() > 200);
    let rects = *rects.borrow();
    (img, rects, content.get())
}

#[test]
fn dialog_footer_right_aligned() {
    let (_, r, content) = confirm_dialog("dialog_unsaved", "Unsaved changes", "This scene has unsaved changes. Save before quitting?", ["Save & Quit", "Discard & Quit", "Cancel"], true);
    // Primary rightmost, Cancel leftmost, in order, and the group touches the right edge.
    assert!(r[0].min.x > r[1].max.x && r[1].min.x > r[2].max.x, "{r:?}");
    assert!((r[0].max.x - content.max.x).abs() < 1.5, "primary not flush right: {r:?} {content:?}");
    // The window did not balloon: the content is not much wider than the text + buttons.
    assert!(content.width() < 460.0, "{content:?}");
    assert!(r[2].min.x > content.min.x + 20.0, "buttons should leave a gap on the left: {r:?} {content:?}");
}

#[test]
fn dialog_footer_variants() {
    let (_, r, c) = confirm_dialog("dialog_import_settings", "Unapplied import settings", "Unapplied import settings for 'fbx file.fbx'", ["Apply", "Revert", "Cancel"], false);
    assert!(r[0].min.x > r[1].max.x && r[1].min.x > r[2].max.x);
    assert!((r[0].max.x - c.max.x).abs() < 1.5);
    let (_, r, c) = confirm_dialog("dialog_switch", "Switch project", "This scene has unsaved changes. Save before opening Demo?", ["Save & Switch", "Discard & Switch", "Cancel"], true);
    assert!(r[0].min.x > r[1].max.x && r[1].min.x > r[2].max.x);
    assert!((r[0].max.x - c.max.x).abs() < 1.5);
}

/// Assemble the contact sheets under docs/design/screens from the PNGs the
/// other tests wrote (run last: `ZESTENGINE_SHOT_DIR=<raw dir>
/// ZESTENGINE_SHEET_DIR=docs/design/screens cargo test -p zestengine-editor
/// --test panel_shots contact_sheets -- --ignored`).
#[test]
#[ignore]
fn contact_sheets() {
    let (Some(raw), Some(out)) = (std::env::var_os("ZESTENGINE_SHOT_DIR"), std::env::var_os("ZESTENGINE_SHEET_DIR")) else { return };
    let (raw, out) = (std::path::PathBuf::from(raw), std::path::PathBuf::from(out));
    let sheets: &[(&str, usize, &[&str])] = &[
        ("sheet-2d-inspectors", 2, &["sprite2d", "tilemap", "rig2d", "sprite_shape_component", "tile_asset", "rule_tile", "sorting_layers", "sprite_shape_profile"]),
        ("sheet-net-and-hierarchy", 2, &["net_manager_full", "skin_rebound", "hierarchy_roles", "hierarchy_view_state", "blendshapes_800", "blendshapes_short"]),
        ("sheet-windows", 1, &["tile_palette", "sprite_editor_sprites", "skinning_editor", "pack_import_ready", "pack_import_metahuman"]),
        ("sheet-model-material-sprite", 3, &["model_import_clips", "material_standard", "sprite_section_multiple"]),
        ("sheet-shader-graph", 1, &["shader_graph_tab"]),
        ("sheet-dialogs", 2, &["dialog_unsaved", "dialog_import_settings", "dialog_switch", "dialog_new_project", "pack_import_ready", "pack_import_metahuman"]),
    ];
    for (name, cols, files) in sheets {
        let imgs: Vec<Shot8> = files.iter().filter_map(|f| Shot8::load(&raw.join(format!("{f}.png")))).filter(|i| i.w > 900).collect();
        if imgs.is_empty() {
            continue;
        }
        Shot8::sheet(&imgs, *cols).save(&out.join(format!("{name}.png")));
    }
}

#[test]
fn selection_outline_settings() {
    let (mut on, mut color, mut width) = (true, Some([124u8, 196, 250]), 2.0f32);
    panel("selection_outline_settings", 380.0, |ui| {
        zestengine_editor::widgets::eyebrow(ui, "Appearance");
        zestengine_editor::selection_outline_settings_ui(ui, &mut on, &mut color, &mut width);
    });
}
