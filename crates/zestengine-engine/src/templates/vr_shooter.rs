//! VR Shooter template (and the VR rig shared with VR Social).

use zestengine_assets::PrimitiveShape;
use zestengine_core::{
    BodyKind, Grabbable, HitscanWeapon, HudBinding, HudValue, Key, MouseButton, RigidBody,
    SphereCollider, Spin, UiAlign, UiCanvas, UiPanel, UiRect, UiRenderMode, UiScaleMode, UiText,
    VrCharacter, WeaponTrigger,
};
use glam::Quat;

use super::TemplateContent;
use super::builder::{self, Scene, button, mouse, mouse_look, wasd};

#[allow(dead_code)] // rig fields are populated by the template but read later in VR.
pub(super) struct Rig {
    pub root: usize,
    pub head: usize,
    pub left: usize,
    pub right: usize,
}

/// VR player rig: play-space root, head camera, two tracked hands with
/// simple hand models, and a teleport marker.
pub(super) fn vr_rig(s: &mut Scene, t: [f32; 3], spawn_tag: &str) -> Rig {
    let root = s.empty("VR Character", None, t);
    s.character_layer(root);
    s.with(root, zestengine_core::Character::default());
    let head = s.camera("Head", Some(root), [0.0, 1.65, 0.0], 90.0);
    s.hide_own_body_from_camera(head);
    let left = s.empty("Left Hand", Some(root), [-0.22, 1.35, -0.38]);
    let right = s.empty("Right Hand", Some(root), [0.2, 1.45, -0.42]);
    let marker = s.prim(
        "Teleport Marker",
        PrimitiveShape::Plane,
        None,
        [0.0, 0.02, 0.0],
        [0.4, 1.0, 0.4],
        "marker",
    );
    s.with(
        marker,
        Spin {
            axis: [0.0, 1.0, 0.0],
            degrees_per_second: 90.0,
        },
    );
    s.hidden(marker);
    s.with(
        root,
        VrCharacter {
            head: s.r(head),
            left_hand: s.r(left),
            right_hand: s.r(right),
            teleport_marker: s.r(marker),
            spawn_tag: spawn_tag.into(),
            ..Default::default()
        },
    );
    let visual = super::avatar::visuals(s, root, root, true);
    // Desktop simulation has object targets even when no headset is connected.
    for name in ["Male", "Female"] {
        let body = s.child(visual, "Avatar Body").unwrap();
        let model = s.child(body, name).unwrap();
        let targets = vec![s.r(head), s.r(left), s.r(right)];
        s.component_mut::<zestengine_core::HumanoidIk>(model)
            .unwrap()
            .objects = targets;
    }
    Rig {
        root,
        head,
        left,
        right,
    }
}

/// World-space text panel (a 3D UI canvas) facing +Z of `rot`.
#[allow(clippy::too_many_arguments)]
pub(super) fn world_panel(
    s: &mut Scene,
    name: &str,
    parent: Option<usize>,
    t: [f32; 3],
    rot: Quat,
    size_m: [f32; 2],
    px: f32,
    text_size: f32,
    lines: &str,
) -> usize {
    let canvas = s.empty(name, parent, t);
    s.rot(canvas, rot);
    let inset = [size_m[0] * px * 0.12, size_m[1] * px * 0.16];
    s.with(
        canvas,
        UiCanvas {
            render_mode: UiRenderMode::World,
            world_size: size_m,
            pixels_per_meter: px,
            scale_mode: UiScaleMode::ScaleWithScreen,
            reference_size: [size_m[0] * px, size_m[1] * px],
            match_height: 1.0,
            sort_order: 0,
        },
    );
    let bg = s.empty("Background", Some(canvas), [0.0; 3]);
    s.with(
        bg,
        UiPanel {
            color: [0.06, 0.07, 0.09, 0.92],
            corner_radius: 18.0,
            border_color: [1.0, 1.0, 1.0, 0.15],
            border_width: 2.0,
        },
    );
    let text = s.empty("Text", Some(canvas), [0.0; 3]);
    s.with(
        text,
        UiRect {
            anchor_min: [0.0, 0.0],
            anchor_max: [1.0, 1.0],
            pivot: [0.5, 0.5],
            position: [0.0, 0.0],
            size: [-inset[0], -inset[1]],
        },
    );
    s.with(
        text,
        UiText {
            text: lines.into(),
            size: text_size,
            align_x: UiAlign::Start,
            align_y: UiAlign::Start,
            ..Default::default()
        },
    );
    canvas
}

pub(super) fn vr_bindings(extra: Vec<zestengine_core::ActionBinding>) -> zestengine_core::Bindings {
    let mut kb = vec![
        wasd("Move"),
        mouse_look("Look"),
        button("Fire", vec![mouse(MouseButton::Left)]),
        button(
            "Grab",
            vec![
                mouse(MouseButton::Right),
                zestengine_core::InputSource::Key(Key::F),
            ],
        ),
        button(
            "Teleport",
            vec![
                zestengine_core::InputSource::Key(Key::T),
                mouse(MouseButton::Middle),
            ],
        ),
    ];
    kb.extend(extra);
    let mut bindings = builder::bindings(
        kb,
        vec![
            builder::stick(
                "Move",
                zestengine_core::PadAxis::LeftStickX,
                zestengine_core::PadAxis::LeftStickY,
                1.0,
            ),
            builder::stick(
                "Look",
                zestengine_core::PadAxis::RightStickX,
                zestengine_core::PadAxis::RightStickY,
                8.0,
            ),
            button(
                "Fire",
                vec![builder::pad_axis(zestengine_core::PadAxis::RightTrigger)],
            ),
            button(
                "Grab",
                vec![builder::pad(zestengine_core::PadButton::RightBumper)],
            ),
            button(
                "Teleport",
                vec![builder::pad(zestengine_core::PadButton::South)],
            ),
        ],
    );
    // What the VR Character reads in a headset (includes Voice).
    bindings
        .schemes
        .push(zestengine_core::ControlScheme::default_vr());
    bindings
}

/// A grabbable hitscan gun lying at `t`; the held trigger fires it.
fn gun(s: &mut Scene, name: &str, t: [f32; 3], hit: usize, automatic: bool) -> usize {
    let root = s.empty(name, None, t);
    s.yaw(root, -90.0);
    super::weapons::model(
        s,
        "Weapon Model",
        root,
        [0.0; 3],
        if automatic {
            super::weapons::BLASTER
        } else {
            super::weapons::PISTOL
        },
    );
    let muzzle_z = if automatic { -0.165 } else { -0.135 };
    let muzzle = s.empty(
        "Muzzle",
        Some(root),
        [0.0, if automatic { 0.096 } else { 0.059 }, muzzle_z],
    );
    s.with(
        muzzle,
        builder::burst_emitter(
            [1.0, 0.8, 0.35, 1.0],
            [0.4, 1.5],
            [0.06, 0.01],
            [0.04, 0.08],
            25.0,
        ),
    );
    s.with(
        root,
        Grabbable {
            snap_to_hand: true,
            hold_position: [0.0, 0.0, 0.0],
            radius: 0.12,
            throwable: false,
            return_after: 4.0,
            ..Default::default()
        },
    );
    s.with(
        root,
        HitscanWeapon {
            trigger: WeaponTrigger::Grabbed,
            reload_action: String::new(),
            automatic,
            fire_rate: if automatic { 10.0 } else { 5.0 },
            damage: if automatic { 12.0 } else { 25.0 },
            spread_deg: if automatic { 1.2 } else { 0.2 },
            magazine_size: if automatic { 30 } else { 12 },
            ammo: if automatic { 30 } else { 12 },
            infinite_reserve: true,
            reload_time: 1.0,
            muzzle: s.r(muzzle),
            hit_effect: s.r(hit),
            ..Default::default()
        },
    );
    root
}

pub fn build() -> TemplateContent {
    let mut s = Scene::new();
    builder::sun(&mut s, [-0.4, -1.0, -0.4], 3.0);
    s.ground(40.0, "ground");
    let hit = builder::impact_effect(&mut s);
    let rig = vr_rig(&mut s, [0.0, 0.0, 2.0], "");

    let bench = s.empty("Gun Bench", None, [0.0, 0.0, 0.8]);
    s.block(
        "Table",
        Some(bench),
        [0.0, 0.45, 0.0],
        [1.6, 0.9, 0.6],
        "wood",
    );
    let pistol = gun(&mut s, "Pistol", [-0.35, 0.95, 0.8], hit, false);
    gun(&mut s, "Blaster", [0.35, 0.95, 0.8], hit, true);
    for (k, x) in [-0.6f32, 0.6].into_iter().enumerate() {
        let ball = s.prim(
            &format!("Ball {}", k + 1),
            PrimitiveShape::Sphere,
            None,
            [x, 1.0, 0.65],
            [0.14; 3],
            "enemy",
        );
        s.with(ball, SphereCollider::default());
        s.with(
            ball,
            RigidBody {
                kind: BodyKind::Dynamic,
                mass: 0.3,
                restitution: 0.4,
                ..Default::default()
            },
        );
        s.with(
            ball,
            Grabbable {
                radius: 0.1,
                throwable: true,
                return_after: 8.0,
                ..Default::default()
            },
        );
    }

    for (k, (x, z)) in [
        (-4.0, -6.0),
        (-1.5, -9.0),
        (1.5, -7.0),
        (4.0, -11.0),
        (0.0, -14.0),
        (-6.0, -12.0),
    ]
    .into_iter()
    .enumerate()
    {
        builder::target_dummy(&mut s, &format!("Target {}", k + 1), [x, 0.0, z], 0.0);
    }
    let walls = s.empty("Backstop", None, [0.0; 3]);
    s.block(
        "Back Wall",
        Some(walls),
        [0.0, 2.0, -18.0],
        [24.0, 4.0, 1.0],
        "wall",
    );

    // Wrist HUD: a small world-space panel on the left hand, readable when you look at your wrist.
    let wrist = world_panel(
        &mut s,
        "Wrist HUD",
        Some(rig.left),
        [0.0, 0.04, 0.05],
        Quat::from_rotation_x(-60f32.to_radians()),
        [0.14, 0.08],
        2500.0,
        48.0,
        "",
    );
    if let Some(text) = s.child(wrist, "Text") {
        s.with(
            text,
            HudBinding {
                value: HudValue::Score,
                format: "Score {value}".into(),
                ..Default::default()
            },
        );
    }
    let ammo = s.empty("Ammo", Some(wrist), [0.0; 3]);
    s.with(
        ammo,
        UiRect {
            anchor_min: [0.0, 1.0],
            anchor_max: [1.0, 1.0],
            pivot: [0.5, 1.0],
            position: [0.0, -20.0],
            size: [-42.0, 60.0],
        },
    );
    s.with(
        ammo,
        UiText {
            text: "12 / 12".into(),
            size: 44.0,
            align_x: UiAlign::Start,
            color: [1.0, 0.85, 0.4, 1.0],
            ..Default::default()
        },
    );
    s.with(
        ammo,
        HudBinding {
            source: s.r(pistol),
            value: HudValue::Ammo,
            format: "Pistol {value}".into(),
            ..Default::default()
        },
    );

    world_panel(
        &mut s,
        "Instructions",
        None,
        [2.2, 1.6, -0.5],
        Quat::from_rotation_y(-35f32.to_radians()),
        [1.2, 0.7],
        1000.0,
        38.0,
        "VR: grip grabs, trigger fires.\nLeft stick moves, right stick turns;\npush it forward to aim a teleport.\n\nDesktop: WASD + mouse, right mouse\nor F grabs, left mouse fires, T teleports.",
    );
    builder::pause_menu(&mut s, builder::MAIN_SCENE, true);
    builder::content(s.finish(), Vec::new(), vr_bindings(Vec::new()), Vec::new())
}
