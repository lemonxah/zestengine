//! Top-down click-to-move template.

use zestengine_assets::PrimitiveShape;
use zestengine_core::{
    Bob, BoxCollider, Character, ClickToMove, FollowCamera, Interactable, Key, MouseButton, NavAgent,
    NavMeshSurface, Spin, UiAlign, UiCanvas, UiRect, UiText,
};

use super::TemplateContent;
use super::builder::{self, CHARACTER_LAYER, Scene, button, keys, mouse, wasd, wheel};

const LAMP_POST: &str = "models/props/default/lamp_post.glb";
const TREASURE_CHEST: &str = "models/props/default/treasure_chest.glb";

/// A lever on a base that toggles `target` when used.
fn lever(s: &mut Scene, name: &str, t: [f32; 3], target: usize, rebake: bool, prompt: &str) -> usize {
    let root = s.empty(name, None, t);
    s.with(root, BoxCollider { center: [0.0, 0.5, 0.0], size: [0.6, 1.0, 0.6], ..Default::default() });
    s.with(root, Interactable { prompt: prompt.into(), toggle: s.r(target), rebake_navmesh: rebake, ..Default::default() });
    s.prim("Base", PrimitiveShape::Cube, Some(root), [0.0, 0.25, 0.0], [0.6, 0.5, 0.6], "trim");
    let handle = s.prim("Handle", PrimitiveShape::Cube, Some(root), [0.0, 0.75, 0.0], [0.1, 0.6, 0.1], "glow");
    s.rot(handle, glam::Quat::from_rotation_x(0.4));
    root
}

/// Bottom-of-screen help text.
pub(super) fn help_text(s: &mut Scene, text: &str) {
    let canvas = s.empty("Help", None, [0.0; 3]);
    s.with(canvas, UiCanvas::default());
    let label = s.empty("Help Text", Some(canvas), [0.0; 3]);
    s.with(label, UiRect { anchor_min: [0.5, 1.0], anchor_max: [0.5, 1.0], pivot: [0.5, 1.0], position: [0.0, -30.0], size: [1400.0, 50.0] });
    s.with(label, UiText { text: text.into(), size: 26.0, color: [1.0, 1.0, 1.0, 0.85], align_x: UiAlign::Center, ..Default::default() });
}

pub fn build() -> TemplateContent {
    let mut s = Scene::new();
    builder::sun(&mut s, [-0.5, -1.0, -0.2], 3.0);
    s.ground(50.0, "grass");
    let nav = s.empty("NavMesh", None, [0.0, 0.0, 0.0]);
    s.with(nav, NavMeshSurface { size: [50.0, 6.0, 50.0], layer_mask: !(1u32 << CHARACTER_LAYER), ..Default::default() });

    let yard = s.empty("Courtyard", None, [0.0; 3]);
    s.block("Wall Back", Some(yard), [0.0, 1.0, -18.0], [12.0, 2.0, 0.8], "wall");
    s.block("Wall Left", Some(yard), [-6.0, 1.0, -13.0], [0.8, 2.0, 10.0], "wall");
    s.block("Wall Right", Some(yard), [6.0, 1.0, -13.0], [0.8, 2.0, 10.0], "wall");
    s.block("Wall Front L", Some(yard), [-3.8, 1.0, -8.0], [3.6, 2.0, 0.8], "wall");
    s.block("Wall Front R", Some(yard), [3.8, 1.0, -8.0], [3.6, 2.0, 0.8], "wall");
    let gate = s.block("Gate", Some(yard), [0.0, 1.0, -8.0], [4.0, 2.0, 0.5], "metal");
    s.dynamic(gate);
    let treasure = s.prim("Treasure", PrimitiveShape::Sphere, None, [0.0, 0.8, -13.0], [0.6; 3], "glow");
    s.with(treasure, Spin { axis: [0.0, 1.0, 0.0], degrees_per_second: 90.0 });
    let mut bob = Bob::default();
    (bob.axis, bob.amplitude, bob.frequency) = ([0.0, 1.0, 0.0], 0.15, 0.5);
    s.with(treasure, bob);
    let chest = s.model("Treasure Chest", TREASURE_CHEST, None, [2.0, 0.0, -12.0], 1.0);
    s.with(chest, BoxCollider { center: [0.0, 0.32, 0.0], size: [0.54, 0.63, 0.93], ..Default::default() });
    s.entries[chest].static_geometry = true;
    lever(&mut s, "Gate Lever", [4.0, 0.0, -5.5], gate, true, "Open gate");

    let lamp = s.empty("Lamp Post", None, [-9.0, 0.0, 1.0]);
    let lamp_model = s.model("Lamp Model", LAMP_POST, Some(lamp), [0.0, 0.0, 0.0], 1.6);
    s.entries[lamp_model].static_geometry = true;
    let light = s.point_light("Lamp Light", Some(lamp), [0.0, 3.0, 0.0], [1.0, 0.8, 0.5], 6.0, 10.0);
    lever(&mut s, "Lamp Switch", [-7.5, 0.0, 2.5], light, false, "Toggle lamp");

    let rocks = s.empty("Rocks", None, [0.0; 3]);
    for (k, (x, z, w)) in [(8.0, 4.0, 1.6), (-12.0, -6.0, 2.2), (12.0, -10.0, 1.2), (-4.0, 12.0, 1.8), (14.0, 12.0, 2.5)]
        .into_iter()
        .enumerate()
    {
        s.block(&format!("Rock {}", k + 1), Some(rocks), [x, w * 0.4, z], [w, w * 0.8, w * 0.9], "trim");
    }

    let marker = s.prim("Click Marker", PrimitiveShape::Plane, None, [0.0, 0.03, 0.0], [0.35, 1.0, 0.35], "marker");
    s.with(marker, Spin { axis: [0.0, 1.0, 0.0], degrees_per_second: 120.0 });
    s.hidden(marker);

    let player = s.empty("Player", None, [0.0, 0.0, 6.0]);
    s.character_layer(player);
    super::tps::character_body(&mut s, player, "player");
    let marker = s.r(marker);
    s.with(player, Character::default())
        .with(player, NavAgent { speed: 4.5, turn_speed: 720.0, stopping_distance: 0.1, ..Default::default() })
        .with(player, ClickToMove { marker, ..Default::default() });

    let cam = s.camera("Camera", None, [0.0, 13.0, 15.0], 50.0);
    s.with(cam, FollowCamera { target: s.r(player), offset: [0.0, 13.0, 9.0], ..Default::default() });
    s.rot(cam, glam::Quat::from_rotation_x(-55f32.to_radians()));

    help_text(&mut s, "Left click to move  |  Click a lever to use it  |  Mouse wheel zooms");
    builder::pause_menu(&mut s, builder::MAIN_SCENE, false);

    let bindings = builder::bindings(
        vec![
            button("Click", vec![mouse(MouseButton::Left)]),
            wheel("Zoom"),
            wasd("Move"),
            button("Interact", keys(&[Key::E])),
        ],
        vec![
            builder::stick("Move", zestengine_core::PadAxis::LeftStickX, zestengine_core::PadAxis::LeftStickY, 1.0),
            button("Click", vec![builder::pad(zestengine_core::PadButton::South)]),
        ],
    );
    builder::content(s.finish(), Vec::new(), bindings, vec![LAMP_POST, TREASURE_CHEST])
}
