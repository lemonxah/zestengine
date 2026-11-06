//! First Person Shooter template.

use zestengine_core::{
    BodyKind, BoxCollider, Character, FirstPersonCharacter, Health, HitscanWeapon,
    RigidBody,
};
use glam::Quat;

use super::TemplateContent;
use super::builder::{self, Scene};

pub const CRATE: &str = "models/props/default/wooden_crate.glb";
pub const METAL_CRATE: &str = "models/props/default/metal_crate.glb";

fn crate_model(s: &mut Scene, name: &str, parent: usize, position: [f32; 3], path: &'static str, meshes: usize) -> usize {
    let i = s.model(name, path, Some(parent), position, 1.0);
    if let zestengine_assets::ObjectSource::Model { extra_meshes, .. } = &mut s.entries[i].source {
        extra_meshes.extend(1..meshes);
    }
    i
}

/// Walled arena with cover, a ramp up to a platform, a crate stack and a
/// target range at the far end. Shared by the shooter templates.
pub(super) fn arena(s: &mut Scene) {
    builder::sun(s, [-0.4, -1.0, -0.3], 3.0);
    s.ground(60.0, "ground");
    let walls = s.empty("Arena", None, [0.0; 3]);
    s.block(
        "Wall North",
        Some(walls),
        [0.0, 2.0, -30.0],
        [60.0, 4.0, 1.0],
        "wall",
    );
    s.block(
        "Wall South",
        Some(walls),
        [0.0, 2.0, 30.0],
        [60.0, 4.0, 1.0],
        "wall",
    );
    s.block(
        "Wall East",
        Some(walls),
        [30.0, 2.0, 0.0],
        [1.0, 4.0, 60.0],
        "wall",
    );
    s.block(
        "Wall West",
        Some(walls),
        [-30.0, 2.0, 0.0],
        [1.0, 4.0, 60.0],
        "wall",
    );
    let cover = s.empty("Cover", None, [0.0; 3]);
    for (k, (x, z, w)) in [
        (-6.0, 0.0, 3.0),
        (6.0, -2.0, 2.0),
        (-2.0, -8.0, 4.0),
        (9.0, 6.0, 2.5),
        (-10.0, 8.0, 2.0),
    ]
    .into_iter()
    .enumerate()
    {
        s.block(
            &format!("Cover {}", k + 1),
            Some(cover),
            [x, 0.6, z],
            [w, 1.2, 0.8],
            "trim",
        );
    }
    let platform = s.empty("Platform", None, [0.0; 3]);
    s.block(
        "Deck",
        Some(platform),
        [-18.0, 1.5, -12.0],
        [8.0, 3.0, 8.0],
        "wall",
    );
    let ramp = s.block(
        "Ramp",
        Some(platform),
        [-18.0, 0.75, -4.6],
        [3.0, 0.3, 7.6],
        "trim",
    );
    s.rot(ramp, Quat::from_rotation_x(23f32.to_radians()));
    let crates = s.empty("Crates", None, [0.0; 3]);
    for (k, p) in [
        [12.0, 0.5, -10.0],
        [13.1, 0.5, -10.0],
        [12.55, 1.5, -10.0],
        [12.0, 0.5, -11.1],
    ]
    .into_iter()
    .enumerate()
    {
        let (path, meshes) = if k % 2 == 0 { (CRATE, 3) } else { (METAL_CRATE, 5) };
        let c = crate_model(s, &format!("Crate {}", k + 1), crates, p, path, meshes);
        s.with(c, BoxCollider::default());
        s.with(
            c,
            RigidBody {
                kind: BodyKind::Dynamic,
                mass: 8.0,
                ..Default::default()
            },
        );
    }
    for (k, (x, z)) in [
        (-8.0, -20.0),
        (-3.0, -24.0),
        (2.0, -18.0),
        (7.0, -22.0),
        (12.0, -26.0),
        (-14.0, -25.0),
    ]
    .into_iter()
    .enumerate()
    {
        builder::target_dummy(s, &format!("Target {}", k + 1), [x, 0.0, z], 0.0);
    }
}

/// The shooter weapon tuned for a rifle.
pub(super) fn rifle(
    aim_from: zestengine_core::ObjectRef,
    muzzle: zestengine_core::ObjectRef,
    hit: zestengine_core::ObjectRef,
) -> HitscanWeapon {
    HitscanWeapon {
        aim_from,
        muzzle,
        hit_effect: hit,
        damage: 25.0,
        fire_rate: 9.0,
        spread_deg: 0.5,
        magazine_size: 30,
        ammo: 30,
        reserve: 150,
        ..Default::default()
    }
}

pub fn build() -> TemplateContent {
    let mut s = Scene::new();
    arena(&mut s);
    let hit = builder::impact_effect(&mut s);

    let player = s.empty("Player", None, [0.0, 0.0, 12.0]);
    s.character_layer(player);
    super::avatar::visuals(&mut s, player, player, false);
    let cam = s.camera("Player Camera", Some(player), [0.0, 1.65, 0.0], 75.0);
    s.hide_own_body_from_camera(cam);
    let gun = super::weapons::model(
        &mut s,
        "Rifle",
        cam,
        [0.22, -0.2, -0.45],
        super::weapons::RIFLE,
    );
    let muzzle = s.empty("Muzzle", Some(gun), [0.0, 0.09, -0.316]);
    s.with(
        muzzle,
        builder::burst_emitter(
            [1.0, 0.8, 0.35, 1.0],
            [0.5, 2.0],
            [0.12, 0.02],
            [0.04, 0.09],
            25.0,
        ),
    );
    let weapon = rifle(s.r(cam), s.r(muzzle), s.r(hit));
    s.with(
        player,
        FirstPersonCharacter {
            camera: s.r(cam),
            ..Default::default()
        },
    )
    .with(player, Character::default())
    .with(
        player,
        Health {
            max: 100.0,
            current: 100.0,
            regen_per_sec: 5.0,
            ..Default::default()
        },
    )
    .with(player, weapon);

    let player_ref = s.r(player);
    builder::shooter_hud(&mut s, player_ref, player_ref, "Shoot the targets");
    builder::pause_menu(&mut s, builder::MAIN_SCENE, true);

    let mut p = Scene::new();
    let dummy = builder::target_dummy(&mut p, "Target Dummy", [0.0; 3], 0.0);
    let prefabs = vec![("prefabs/target_dummy.prefab".to_string(), p.prefab(dummy))];
    builder::content(s.finish(), prefabs, builder::shooter_bindings(), vec![CRATE, METAL_CRATE])
}
