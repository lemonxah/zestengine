//! Strategy (RTS) template.

use zestengine_assets::PrimitiveShape;
use zestengine_core::{
    BoxCollider, Character, HudBinding, HudValue, Key, MouseButton, NavAgent, NavMeshSurface, ResourceAmount, ResourceStore,
    RtsCamera, RtsSelection, Selectable, UiAlign, UiCanvas, UiPanel, UiRect, UiText,
};

use super::TemplateContent;
use super::builder::{self, CHARACTER_LAYER, Scene, axis1, button, keys, mouse, wasd, wheel};

const TREE: &str = "models/props/default/tree.glb";
const BASE_BUILDING: &str = "models/props/default/base_building.glb";

fn prop(s: &mut Scene, name: &str, parent: usize, position: [f32; 3], scale: f32, path: &'static str, meshes: usize) -> usize {
    let i = s.model(name, path, Some(parent), position, scale);
    if let zestengine_assets::ObjectSource::Model { extra_meshes, .. } = &mut s.entries[i].source {
        extra_meshes.extend(1..meshes);
    }
    i
}

/// A selectable unit: capsule body, a selection ring (hidden until selected),
/// a character controller and a nav agent.
fn unit(s: &mut Scene, name: &str, t: [f32; 3]) -> usize {
    let root = s.empty(name, None, t);
    s.character_layer(root);
    super::avatar::visuals(s, root, root, false);
    let ring = s.prim("Selection Ring", PrimitiveShape::Plane, Some(root), [0.0, 0.03, 0.0], [0.6, 1.0, 0.6], "select_ring");
    s.hidden(ring);
    let ring = s.r(ring);
    s.with(root, Character { radius: 0.4, ..Default::default() })
        .with(root, NavAgent { speed: 5.0, turn_speed: 720.0, stopping_distance: 0.15, ..Default::default() })
        .with(root, Selectable { ring, team: 1, ..Default::default() });
    root
}

fn label(s: &mut Scene, parent: usize, name: &str, x: f32, binding: HudBinding) {
    let i = s.empty(name, Some(parent), [0.0; 3]);
    s.with(i, UiRect { anchor_min: [0.0, 0.5], anchor_max: [0.0, 0.5], pivot: [0.0, 0.5], position: [x, 0.0], size: [260.0, 50.0] });
    s.with(i, UiText { text: name.into(), size: 28.0, align_x: UiAlign::Start, ..Default::default() });
    s.with(i, binding);
}

pub fn build() -> TemplateContent {
    let mut s = Scene::new();
    builder::sun(&mut s, [-0.3, -1.0, -0.5], 3.2);
    s.ground(80.0, "grass");
    let nav = s.empty("NavMesh", None, [0.0; 3]);
    s.with(nav, NavMeshSurface { size: [80.0, 6.0, 80.0], cell_size: 0.3, layer_mask: !(1u32 << CHARACTER_LAYER), ..Default::default() });

    let base = s.empty("Base", None, [0.0, 0.0, 16.0]);
    let town_hall = prop(&mut s, "Town Hall", base, [0.0, 0.0, 0.0], 1.0, BASE_BUILDING, 24);
    s.with(town_hall, BoxCollider { center: [0.0, 1.8, 0.6], size: [8.0, 3.6, 9.2], ..Default::default() });
    s.entries[town_hall].static_geometry = true;
    let mine = s.empty("Gold Mine", None, [-16.0, 0.0, -10.0]);
    s.block("Rock", Some(mine), [0.0, 1.2, 0.0], [5.0, 2.4, 4.0], "trim");
    s.prim("Gold", PrimitiveShape::Sphere, Some(mine), [0.0, 2.5, 0.0], [1.2; 3], "glow");
    let forest = s.empty("Forest", None, [0.0; 3]);
    for (k, (x, z)) in [(14.0, -8.0), (17.0, -5.0), (15.0, -2.0), (19.0, -10.0), (12.0, -13.0), (20.0, 0.0)].into_iter().enumerate() {
        let tree = prop(&mut s, &format!("Tree {}", k + 1), forest, [x, 0.0, z], 0.5, TREE, 1);
        s.with(tree, BoxCollider { center: [0.0, 0.9, 0.0], size: [0.6, 1.8, 0.6], ..Default::default() });
        s.entries[tree].static_geometry = true;
    }

    for k in 0..8 {
        let (x, z) = ((k % 4) as f32 * 1.8 - 2.7, (k / 4) as f32 * 1.8 + 6.0);
        unit(&mut s, &format!("Unit {}", k + 1), [x, 0.0, z]);
    }

    let marker = s.prim("Move Marker", PrimitiveShape::Plane, None, [0.0, 0.03, 0.0], [0.5, 1.0, 0.5], "marker");
    s.hidden(marker);

    let game = s.empty("Game", None, [0.0; 3]);
    s.with(
        game,
        ResourceStore {
            resources: vec![
                ResourceAmount { name: "Gold".into(), amount: 200.0, income_per_sec: 5.0, capacity: 0.0 },
                ResourceAmount { name: "Wood".into(), amount: 100.0, income_per_sec: 2.0, capacity: 0.0 },
            ],
        },
    );

    let hud = s.empty("HUD", None, [0.0; 3]);
    s.with(hud, UiCanvas::default());
    let bar = s.empty("Resource Bar", Some(hud), [0.0; 3]);
    s.with(bar, UiRect { anchor_min: [0.0, 0.0], anchor_max: [1.0, 0.0], pivot: [0.5, 0.0], position: [0.0, 0.0], size: [0.0, 64.0] });
    s.with(bar, UiPanel { color: [0.05, 0.06, 0.08, 0.85], corner_radius: 0.0, ..Default::default() });
    let store = s.r(game);
    label(&mut s, bar, "Gold", 30.0, HudBinding { source: store, value: HudValue::Resource("Gold".into()), format: "Gold  {value}".into(), ..Default::default() });
    label(&mut s, bar, "Wood", 300.0, HudBinding { source: store, value: HudValue::Resource("Wood".into()), format: "Wood  {value}".into(), ..Default::default() });
    let selection_box = s.empty("Selection Box", Some(hud), [0.0; 3]);
    s.with(selection_box, UiRect::default());
    s.with(selection_box, UiPanel { color: [0.3, 1.0, 0.4, 0.12], corner_radius: 2.0, border_color: [0.3, 1.0, 0.4, 0.9], border_width: 1.5 });
    s.hidden(selection_box);

    let cam = s.camera("Camera", None, [0.0, 21.0, 22.0], 50.0);
    s.with(cam, RtsCamera { yaw: 0.0, pitch: 55.0, distance: 26.0, ..Default::default() });
    s.with(cam, RtsSelection { box_visual: s.r(selection_box), move_marker: s.r(marker), team: 1, ..Default::default() });
    s.rot(cam, glam::Quat::from_rotation_x(-55f32.to_radians()));
    let cam_ref = s.r(cam);
    label(&mut s, bar, "Selected", 570.0, HudBinding { source: cam_ref, value: HudValue::SelectedCount, format: "Selected  {value}".into(), ..Default::default() });

    super::topdown::help_text(
        &mut s,
        "Drag to box-select  |  Right click to move  |  WASD / screen edges pan  |  Q E rotate  |  Wheel zooms",
    );
    builder::pause_menu(&mut s, builder::MAIN_SCENE, false);

    let mut p = Scene::new();
    let proto = unit(&mut p, "Unit", [0.0; 3]);
    let prefabs = vec![("prefabs/unit.prefab".to_string(), p.prefab(proto))];

    let bindings = builder::bindings(
        vec![
            wasd("Move"),
            wheel("Zoom"),
            axis1("Rotate", keys(&[Key::Q]), keys(&[Key::E])),
            button("Select", vec![mouse(MouseButton::Left)]),
            button("Command", vec![mouse(MouseButton::Right)]),
            button("AddSelect", keys(&[Key::LShift, Key::RShift])),
        ],
        vec![builder::stick("Move", zestengine_core::PadAxis::LeftStickX, zestengine_core::PadAxis::LeftStickY, 1.0)],
    );
    builder::content(s.finish(), prefabs, bindings, vec![TREE, BASE_BUILDING])
}
