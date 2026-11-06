//! Third Person Shooter template.

use zestengine_core::{Character, Health, ThirdPersonCharacter, WeaponHold, WeaponPose};
use glam::{EulerRot, Quat};

use super::TemplateContent;
use super::builder::{self, Scene};

/// A capsule character body (feet at the parent origin) with a visor so
/// facing reads at a glance.
pub(super) fn character_body(s: &mut Scene, parent: usize, material: &str) -> usize {
    let _ = material;
    super::avatar::visuals(s, parent, parent, false)
}

pub fn build() -> TemplateContent {
    let mut s = Scene::new();
    super::fps::arena(&mut s);
    let hit = builder::impact_effect(&mut s);

    let player = s.empty("Player", None, [0.0, 0.0, 12.0]);
    s.character_layer(player);
    let avatar = character_body(&mut s, player, "player");
    let gun = super::weapons::model(&mut s, "Rifle", player, [0.32, 1.2, -0.4], super::weapons::RIFLE);
    let cam = s.camera("Camera", None, [0.55, 1.55, 15.6], 65.0);
    // Where each hand bone goes on the rifle (pistol grip, foregrip). Move these
    // and the rifle with the gizmo outside Play, then Store the pose.
    let hand = |s: &mut Scene, name: &str, position: [f32; 3], [x, y, z]: [f32; 3]| {
        let i = s.empty(name, Some(gun), position);
        s.rot(i, Quat::from_euler(EulerRot::XYZ, x.to_radians(), y.to_radians(), z.to_radians()));
        i
    };
    let right = hand(&mut s, "Right Hand Target", [0.0053, -0.1293, 0.2498], [-83.28, -4.47, -1.68]);
    let left = hand(&mut s, "Left Hand Target", [-0.0589, -0.1039, -0.0972], [-171.61, 22.51, -93.69]);
    s.with(
        gun,
        WeaponHold {
            avatar: s.r(avatar),
            camera: s.r(cam),
            idle: WeaponPose { position: [0.0648, 0.0127, 0.3023], rotation: [-172.06, -46.11, -163.73], ..Default::default() },
            aim: WeaponPose { position: [-0.0352, 0.3022, 0.3127], rotation: [165.96, -13.28, -179.93], ..Default::default() },
            right_hand: s.r(right),
            left_hand: s.r(left),
            ..Default::default()
        },
    );
    let muzzle = s.empty("Muzzle", Some(gun), [0.0, 0.09, -0.316]);
    s.with(muzzle, builder::burst_emitter([1.0, 0.8, 0.35, 1.0], [0.5, 2.0], [0.14, 0.02], [0.04, 0.09], 25.0));

    let mut weapon = super::fps::rifle(s.r(cam), s.r(muzzle), s.r(hit));
    weapon.spread_deg = 0.8;
    s.with(player, ThirdPersonCharacter { camera: s.r(cam), ..Default::default() })
        .with(player, Character::default())
        .with(player, Health { max: 100.0, current: 100.0, regen_per_sec: 5.0, ..Default::default() })
        .with(player, weapon);

    let player_ref = s.r(player);
    builder::shooter_hud(&mut s, player_ref, player_ref, "Hold right mouse to aim. Shoot the targets");
    builder::pause_menu(&mut s, builder::MAIN_SCENE, true);
    builder::content(s.finish(), Vec::new(), builder::shooter_bindings(), vec![super::fps::CRATE, super::fps::METAL_CRATE])
}
