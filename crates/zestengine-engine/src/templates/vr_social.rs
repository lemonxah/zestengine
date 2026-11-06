//! VR Social template: a hangout room with avatar, mirror, world-space UI,
//! spawn points and synced grabbables.

use zestengine_assets::PrimitiveShape;
use zestengine_core::{
    BodyKind, BoxCollider, Grabbable, Key, NetworkAvatar, NetworkManager, PlanarReflection, RigidBody, SpawnPoint,
    SphereCollider, Sync, UiAction, UiButton, UiPanel, UiRect, UiText,
};
use glam::Quat;

use super::TemplateContent;
use super::builder::{self, Scene, button, keys};
use super::vr_shooter::{vr_bindings, vr_rig, world_panel};

const HELMET: &str = "models/DamagedHelmet.glb";
const HELMET_META: &str = "models/DamagedHelmet.glb.meta";
const AVATAR_PREFAB: &str = "prefabs/net_avatar.prefab";
const TABLE: &str = "models/props/default/table.glb";
const BALL: &str = "models/props/default/ball.glb";

/// Head + hands + torso, spawned by the Network Manager for every player. The
/// owner's copy mirrors the local VR Character and is hidden for them; everyone
/// else sees it move.
fn net_avatar_prefab() -> crate::prefab::Prefab {
    let mut p = Scene::new();
    let root = p.empty("Net Avatar", None, [0.0; 3]);
    let head = p.empty("Head", Some(root), [0.0, 1.65, 0.0]);
    let left = p.empty("Left Hand", Some(root), [-0.22, 1.35, -0.38]);
    let right = p.empty("Right Hand", Some(root), [0.2, 1.45, -0.42]);
    let visual = super::avatar::visuals(&mut p, root, root, true);
    p.component_mut::<zestengine_core::TemplateAvatar>(visual).unwrap().follow_local_player = true;
    let body = p.child(visual, "Avatar Body").unwrap();
    let hidden = vec![p.r(body)];
    p.with(root, zestengine_core::NetworkVariables::default());
    p.with(
        root,
        NetworkAvatar {
            head: p.r(head),
            left_hand: p.r(left),
            right_hand: p.r(right),
            drive_from_vr: false,
            follow_local_player: true,
            owner_hidden: hidden,
            ..Default::default()
        },
    );
    p.prefab(root)
}

/// Network-replicated transform; ownership follows whoever grabs it (the
/// VR Character requests it), so the Sync component's own grab toggle is off.
fn local_sync() -> Sync {
    let mut sync = Sync::default();
    sync.grabbable = false;
    sync
}

/// A button on a world panel that runs `actions`.
fn panel_button(s: &mut Scene, parent: usize, caption: &str, y: f32, actions: Vec<UiAction>) {
    let b = s.empty(&format!("{caption} Button"), Some(parent), [0.0; 3]);
    s.with(b, UiRect { anchor_min: [0.5, 1.0], anchor_max: [0.5, 1.0], pivot: [0.5, 1.0], position: [0.0, y], size: [520.0, 80.0] });
    s.with(b, UiPanel { color: [0.18, 0.18, 0.22, 0.95], corner_radius: 12.0, ..Default::default() });
    s.with(b, UiButton { on_click: actions, ..Default::default() });
    let label = s.empty("Label", Some(b), [0.0; 3]);
    s.with(label, UiText { text: caption.into(), size: 36.0, ..Default::default() });
}

pub fn build() -> TemplateContent {
    let mut s = Scene::new();
    s.skybox = None;
    // Lit by the room's own lights: no directional light on purpose.
    builder::no_sun(&mut s);
    s.environment.ambient_intensity = 0.6;

    let room = s.empty("Room", None, [0.0; 3]);
    s.ground(20.0, "wood");
    s.block("Wall North", Some(room), [0.0, 1.5, -10.0], [20.0, 3.0, 0.4], "wall");
    s.block("Wall South", Some(room), [0.0, 1.5, 10.0], [20.0, 3.0, 0.4], "wall");
    s.block("Wall East", Some(room), [10.0, 1.5, 0.0], [0.4, 3.0, 20.0], "wall");
    s.block("Wall West", Some(room), [-10.0, 1.5, 0.0], [0.4, 3.0, 20.0], "wall");
    s.block("Ceiling", Some(room), [0.0, 3.1, 0.0], [20.0, 0.2, 20.0], "trim");

    let lights = s.empty("Room Lights", None, [0.0; 3]);
    for (k, (x, z)) in [(-5.0, -5.0), (5.0, -5.0), (-5.0, 5.0), (5.0, 5.0)].into_iter().enumerate() {
        s.point_light(&format!("Light {}", k + 1), Some(lights), [x, 2.7, z], [1.0, 0.92, 0.82], 8.0, 12.0);
    }

    // Mirror on the north wall: a plane whose +Y faces into the room.
    s.block("Mirror Frame", Some(room), [0.0, 1.5, -9.75], [4.3, 2.9, 0.1], "trim");
    let mirror = s.prim("Mirror", PrimitiveShape::Plane, None, [0.0, 1.5, -9.68], [2.0, 1.0, 1.35], "mirror");
    s.rot(mirror, Quat::from_rotation_x(std::f32::consts::FRAC_PI_2));
    s.with(mirror, PlanarReflection { max_roughness: 0.35 });

    // Player spawn points (one per peer, picked by peer id).
    let spawns = s.empty("Spawn Points", None, [0.0; 3]);
    for (k, (x, z, yaw)) in [(0.0, 4.0, 0.0), (3.0, 6.0, 25.0), (-3.0, 6.0, -25.0), (0.0, 8.0, 0.0)].into_iter().enumerate() {
        let p = s.empty(&format!("Spawn {}", k + 1), Some(spawns), [x, 0.0, z]);
        s.yaw(p, yaw);
        s.with(p, SpawnPoint { tag: "player".into(), index: k as u32 });
    }

    vr_rig(&mut s, [0.0, 0.0, 4.0], "player");
    // The shared VR rig supplies both full-body mannequin variants.

    // Host / join from the Network window (or set Auto Start for built games).
    let net = s.empty("Network Manager", None, [0.0, 0.0, 4.0]);
    s.with(
        net,
        NetworkManager {
            session_name: "VR Social".into(),
            player_prefab: AVATAR_PREFAB.into(),
            spawn_tag: "player".into(),
            ..Default::default()
        },
    );

    // Synced grabbables: anyone can pick them up; the holder owns them on the network.
    let table = s.empty("Table", None, [3.5, 0.0, 0.0]);
    let table_model = s.model("Table Model", TABLE, Some(table), [0.0, 0.0, 0.0], 1.0);
    s.with(table_model, BoxCollider { center: [0.0, 0.38, 0.0], size: [1.4, 0.76, 1.4], ..Default::default() });
    s.entries[table_model].static_geometry = true;
    for (k, (x, z, _)) in [(3.1, -0.2, "enemy"), (3.5, 0.2, "ally"), (3.9, -0.1, "glow")].into_iter().enumerate() {
        let ball = s.model(&format!("Ball {}", k + 1), BALL, None, [x, 1.0, z], 1.0);
        if let zestengine_assets::ObjectSource::Model { extra_meshes, .. } = &mut s.entries[ball].source {
            extra_meshes.extend(1..2);
        }
        s.with(ball, SphereCollider { radius: 0.11, ..Default::default() });
        s.with(ball, RigidBody { kind: BodyKind::Dynamic, mass: 0.4, restitution: 0.5, ..Default::default() });
        s.with(ball, Grabbable { radius: 0.12, throwable: true, return_after: 15.0, ..Default::default() });
        s.with(ball, local_sync());
    }
    s.block("Pedestal", None, [-3.5, 0.5, -2.0], [0.6, 1.0, 0.6], "trim");
    let helmet = s.model("Helmet", HELMET, None, [-3.5, 1.3, -2.0], 0.25);
    s.with(helmet, Grabbable { radius: 0.2, throwable: false, return_after: 20.0, ..Default::default() });
    s.with(helmet, local_sync());

    // World-space welcome board with working buttons.
    let board = world_panel(
        &mut s,
        "Welcome Board",
        None,
        [-6.0, 1.6, 2.0],
        Quat::from_rotation_y(60f32.to_radians()),
        [1.4, 1.0],
        1000.0,
        40.0,
        "Welcome!\n\nGrab things with the grip (desktop: right mouse).\nHold V to talk once you're connected.",
    );
    panel_button(&mut s, board, "Toggle Lights", -260.0, vec![UiAction::Toggle("Room Lights".into())]);
    panel_button(&mut s, board, "Hide Mirror", -160.0, vec![UiAction::Toggle("Mirror".into())]);

    builder::pause_menu(&mut s, builder::MAIN_SCENE, true);
    let bindings = vr_bindings(vec![button("Voice", keys(&[Key::V]))]);
    let prefabs = vec![(AVATAR_PREFAB.to_string(), net_avatar_prefab())];
    builder::content(s.finish(), prefabs, bindings, vec![HELMET, HELMET_META, TABLE, BALL])
}
