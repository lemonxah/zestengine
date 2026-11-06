use std::collections::HashSet;
use std::path::{Path, PathBuf};

use zestengine_assets::{MaterialRef, ObjectSource};
use zestengine_core::{
    ActionBinding, Component, ComponentRegistry, ControlScheme, FirstPersonCharacter, Health,
    HitscanWeapon, HudBinding, HudValue, InputSource, InputState, Key, MouseButton, NetView,
    ObjectId, ObjectRef, RawInput, UiText,
};
use glam::{Mat4, Quat, Vec3};

use super::TEMPLATES;
use crate::play::{PlaySession, TickInput, World};
use crate::scene::{LoadedScene, SceneObject};
use crate::shaders::ShaderLibrary;

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("zestengine-template-test-{tag}-{}", ObjectId::new()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Every object id referenced by a component's RON (ObjectRefs serialize as
/// quoted UUID strings).
fn referenced_ids(data: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = data;
    while let Some(start) = rest.find('"') {
        let tail = &rest[start + 1..];
        let Some(end) = tail.find('"') else { break };
        let s = &tail[..end];
        if s.len() == 36 && s.matches('-').count() == 4 && s.parse::<ObjectId>().is_ok() {
            out.push(s.to_string());
        }
        rest = &tail[end + 1..];
    }
    out
}

#[test]
fn every_template_scaffolds_and_loads() {
    let registry = ComponentRegistry::with_builtins();
    let parent = temp_dir("scaffold");
    for template in TEMPLATES {
        let root = crate::bundle::scaffold_project(&parent, template.id, template.id)
            .unwrap_or_else(|e| panic!("{}: scaffold failed: {e:#}", template.id));
        let project = zestengine_assets::load_project_file(&root)
            .unwrap_or_else(|e| panic!("{}: {e:#}", template.id));
        let boot = project.boot_scene.clone().expect("boot scene set");
        assert_eq!(project.last_scene.as_deref(), Some(boot.as_str()));
        assert!(
            !project.bindings.schemes.is_empty(),
            "{}: bindings",
            template.id
        );
        assert!(root.join("Cargo.toml").is_file() && root.join("src/main.rs").is_file());
        let vr = template.category == zestengine_assets::project_templates::TemplateCategory::Vr;
        assert_eq!(project.vr, vr, "{}: only VR templates start in VR", template.id);
        #[cfg(feature = "client")]
        {
            let xr = crate::GameConfig::from_project_dir(&root).expect("game config").xr;
            assert_eq!(xr, if vr { crate::XrMode::Auto } else { crate::XrMode::Off }, "{}", template.id);
        }

        let scene_path = root.join(&boot);
        let scene = zestengine_assets::load_scene_file(&scene_path)
            .unwrap_or_else(|e| panic!("{}: {e:#}", template.id));
        assert!(!scene.entries.is_empty());
        let ids: HashSet<&str> = scene.entries.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(
            ids.len(),
            scene.entries.len(),
            "{}: object ids are unique",
            template.id
        );
        for (i, e) in scene.entries.iter().enumerate() {
            if let Some(p) = e.parent {
                assert!(p < i, "{}: {} parent comes first", template.id, e.name);
            }
            if let MaterialRef::File(m) = &e.material {
                assert!(
                    root.join(m).is_file(),
                    "{}: material {m} written",
                    template.id
                );
            }
            if let ObjectSource::Model { path, .. } = &e.source {
                assert!(
                    root.join(path).is_file(),
                    "{}: model {path} copied",
                    template.id
                );
            }
            for c in &e.components {
                assert!(
                    registry.load(&c.kind, &c.data).is_some(),
                    "{}: {} component {:?} must load: {}",
                    template.id,
                    e.name,
                    c.kind,
                    c.data
                );
                for r in referenced_ids(&c.data) {
                    assert!(
                        ids.contains(r.as_str()),
                        "{}: {} / {} references a missing object",
                        template.id,
                        e.name,
                        c.kind
                    );
                }
            }
        }
        if let Some(sky) = &scene.skybox {
            assert!(root.join(sky).is_file(), "{}: skybox copied", template.id);
        }
        let cameras = scene
            .entries
            .iter()
            .filter(|e| matches!(e.source, ObjectSource::Camera))
            .count();
        assert!(cameras >= 1, "{}: has a camera", template.id);

        for entry in std::fs::read_dir(root.join("prefabs")).unwrap().flatten() {
            let prefab = crate::prefab::Prefab::load(entry.path()).expect("prefab loads");
            assert_eq!(prefab.roots().len(), 1);
            for e in &prefab.entries {
                for c in &e.components {
                    assert!(
                        registry.load(&c.kind, &c.data).is_some(),
                        "prefab component {}",
                        c.kind
                    );
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&parent);
}

#[test]
fn unknown_template_is_rejected() {
    let parent = temp_dir("unknown");
    assert!(crate::bundle::scaffold_project(&parent, "x", "no-such-template").is_err());
    assert!(!parent.join("x").exists());
    let _ = std::fs::remove_dir_all(&parent);
}

#[test]
fn weapon_hold_puts_the_rifle_on_its_pose_and_both_hands_on_it() {
    use zestengine_core::WeaponHold;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model = zestengine_assets::load_gltf(root.join(super::avatar::MALE)).expect("avatar imports");
    let skel = &model.skeletons[0];
    let (scene, _) = load_template("tps");
    let world = scene.world_transforms();
    let find = |name: &str| scene.objects.iter().position(|o| o.name == name).unwrap();
    let (avatar, rifle) = (find("Male"), find("Rifle"));
    let hold = scene.objects[rifle].components.iter().find_map(|c| c.as_any().downcast_ref::<WeaponHold>()).unwrap().clone();
    let mut locals = skel.rest_locals();
    let update = crate::scene::hold_weapon(&scene.objects, &world, avatar, rifle, &hold, true, skel, &mut locals, 0.016).expect("holds");
    let joints = skel.world_matrices(&locals);
    let joint = |name: &str| world[avatar] * joints[skel.joints.iter().position(|j| j.name == name).unwrap()];
    let anchor = joint("spine_03");
    let [x, y, z] = hold.idle.rotation.map(f32::to_radians);
    let idle = anchor * Mat4::from_rotation_translation(Quat::from_euler(glam::EulerRot::XYZ, x, y, z), Vec3::from(hold.idle.position));
    assert!((update.world.w_axis - idle.w_axis).length() < 1e-3, "rifle sits at the idle pose");
    for (bone, target) in [("hand_r", "Right Hand Target"), ("hand_l", "Left Hand Target")] {
        let t = find(target);
        let goal = update.world * scene.objects[t].local_transform();
        let reached = joint(bone);
        assert!((reached.w_axis - goal.w_axis).length() < 0.01, "{bone} reaches its target: {:?} vs {:?}", reached.w_axis, goal.w_axis);
        let (_, rr, _) = reached.to_scale_rotation_translation();
        let (_, gr, _) = goal.to_scale_rotation_translation();
        assert!(rr.angle_between(gr) < 0.05, "{bone} turns to its target");
    }
}

#[test]
fn aiming_turns_the_rifle_toward_the_camera_forward() {
    use zestengine_core::{Animator, WeaponHold};
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model = zestengine_assets::load_gltf(root.join(super::avatar::MALE)).expect("avatar imports");
    let skel = &model.skeletons[0];
    let (mut scene, _) = load_template("tps");
    let find = |name: &str| scene.objects.iter().position(|o| o.name == name).unwrap();
    let (avatar, rifle, cam) = (find("Male"), find("Rifle"), find("Camera"));
    // Aim: the animator's "Aiming" param drives the hold toward the aim pose.
    scene.objects[avatar]
        .components
        .iter_mut()
        .find_map(|c| c.as_any_mut().downcast_mut::<Animator>())
        .unwrap()
        .set_bool("Aiming", true);
    // Tilt the camera up: the rifle must pitch to keep pointing at it.
    scene.objects[cam].rotation = Quat::from_euler(glam::EulerRot::YXZ, 0.0, 40f32.to_radians(), 0.0);
    let world = scene.world_transforms();
    let mut hold = scene.objects[rifle]
        .components
        .iter()
        .find_map(|c| c.as_any().downcast_ref::<WeaponHold>())
        .unwrap()
        .clone();
    hold.blend = 1.0; // fully aimed
    let mut locals = skel.rest_locals();
    let update = crate::scene::hold_weapon(&scene.objects, &world, avatar, rifle, &hold, true, skel, &mut locals, 0.016).expect("holds");
    let (_, wr, _) = update.world.to_scale_rotation_translation();
    let weapon_forward = (wr * Vec3::NEG_Z).normalize_or(Vec3::NEG_Z);
    let cam_forward = world[cam].transform_vector3(Vec3::NEG_Z).normalize_or(Vec3::NEG_Z);
    assert!(
        weapon_forward.dot(cam_forward) > 0.99,
        "rifle points at the camera forward: {weapon_forward:?} vs {cam_forward:?}"
    );
}

#[test]
fn default_avatars_import_and_drive_the_real_animation_controller() {
    use crate::anim_graph::{ClipSet, Rig, advance};
    use zestengine_core::Animator;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let controller = super::avatar::controller();
    for path in [super::avatar::MALE, super::avatar::FEMALE] {
        let model = zestengine_assets::load_gltf(root.join(path)).expect("avatar imports in ZestEngine");
        assert_eq!(model.skeletons.len(), 1);
        assert_eq!(model.meshes.len(), 2, "skin and joint-detail primitives");
        let skeleton = &model.skeletons[0];
        assert!(skeleton.joints.len() < 100, "control rig is stripped");
        let rig = Rig::from_skeleton(skeleton);
        let indices: Vec<usize> = (0..model.animations.len()).collect();
        let assets = Default::default();
        let clips = ClipSet {
            imported: &model.animations,
            indices: &indices,
            assets: &assets,
        };
        let mut animator = Animator {
            controller: super::avatar::CONTROLLER.into(),
            ..Default::default()
        };
        let idle = advance(&mut animator, Some(&controller), &rig, &clips, 0.1).pose;
        assert_eq!(animator.state.layers[0].state_name, "Locomotion");
        animator.set_float("Forward", 4.5);
        let moving = advance(&mut animator, Some(&controller), &rig, &clips, 0.3).pose;
        assert!(
            idle.trs
                .iter()
                .zip(&moving.trs)
                .any(|(a, b)| a.r.angle_between(b.r).abs() > 0.1),
            "movement changes the skeleton"
        );
        animator.set_bool("Crouched", true);
        advance(&mut animator, Some(&controller), &rig, &clips, 0.2);
        assert_eq!(animator.state.layers[0].state_name, "Crouch");
        animator.set_bool("Grounded", false);
        advance(&mut animator, Some(&controller), &rig, &clips, 0.2);
        assert_eq!(animator.state.layers[0].state_name, "Air");
        animator.set_bool("Grounded", true);
        let pose = advance(&mut animator, Some(&controller), &rig, &clips, 0.2).pose;
        assert_eq!(animator.state.layers[0].state_name, "Land");
        assert!(pose.to_locals().iter().all(|m| m.is_finite()));

        // Moving during the landing cuts straight back to locomotion instead
        // of playing out the whole landing clip (no sliding while landing).
        animator.set_float("Speed", 2.0);
        advance(&mut animator, Some(&controller), &rig, &clips, 0.05);
        assert_eq!(animator.state.layers[0].state_name, "Locomotion");
    }
}

#[test]
fn first_person_cameras_exclude_the_avatar_body_layer() {
    use zestengine_core::CameraComponent;
    let character_layer_bit = 1u32 << 8;
    for (id, camera_name, expect_hidden) in [
        ("fps", "Player Camera", true),
        ("tps", "Camera", false),
        ("vr_shooter", "Head", true),
    ] {
        let (scene, _) = load_template(id);
        let camera = scene
            .objects
            .iter()
            .find(|o| o.name == camera_name)
            .unwrap_or_else(|| panic!("{id}: no {camera_name}"));
        let component = camera
            .components
            .iter()
            .find_map(|c| c.as_any().downcast_ref::<CameraComponent>())
            .unwrap();
        let hides_body = component.culling_mask & character_layer_bit == 0;
        assert_eq!(
            hides_body, expect_hidden,
            "{id}: {camera_name} culling mask"
        );
    }
}

#[test]
fn weapon_models_import_in_zestengine() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for (path, expect_meshes) in [
        (super::weapons::RIFLE, 1),
        (super::weapons::PISTOL, 1),
        (super::weapons::BLASTER, 19),
    ] {
        let model =
            zestengine_assets::load_gltf(root.join(path)).unwrap_or_else(|e| panic!("{path}: {e:#}"));
        assert_eq!(
            model.meshes.len(),
            expect_meshes,
            "{path}: mesh primitive count"
        );
        assert!(!model.textures.is_empty(), "{path}: has textures");
        assert!(!model.materials.is_empty(), "{path}: has materials");
    }
}

#[test]
fn crate_model_imports_in_zestengine() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model = zestengine_assets::load_gltf(root.join(super::fps::CRATE))
        .unwrap_or_else(|e| panic!("crate: {e:#}"));
    assert_eq!(model.meshes.len(), 3, "crate mesh primitive count");
    assert!(!model.textures.is_empty(), "crate has textures");
    assert!(!model.materials.is_empty(), "crate has materials");
}

#[test]
fn prop_models_import_in_zestengine() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for (path, expect_meshes) in [
        ("models/props/default/tree.glb", 1),
        ("models/props/default/base_building.glb", 24),
        ("models/props/default/lamp_post.glb", 1),
        ("models/props/default/metal_crate.glb", 5),
        ("models/props/default/treasure_chest.glb", 1),
        ("models/props/default/table.glb", 1),
        ("models/props/default/wall_light.glb", 2),
        ("models/props/default/ball.glb", 2),
    ] {
        let model =
            zestengine_assets::load_gltf(root.join(path)).unwrap_or_else(|e| panic!("{path}: {e:#}"));
        assert_eq!(
            model.meshes.len(),
            expect_meshes,
            "{path}: mesh primitive count"
        );
        assert!(!model.materials.is_empty(), "{path}: has materials");
    }
}

#[test]
fn gameplay_templates_have_switchable_avatars_and_controller_assets() {
    use zestengine_core::{AvatarVariant, TemplateAvatar};
    let parent = temp_dir("avatar-assets");
    for template in TEMPLATES.iter().filter(|t| t.gameplay) {
        let root = crate::bundle::scaffold_project(&parent, template.id, template.id).unwrap();
        let controller =
            zestengine_assets::load_anim_controller(root.join(super::avatar::CONTROLLER)).unwrap();
        assert!(!controller.layers[0].machine.states.is_empty());
        let (mut scene, scheme) = load_template(template.id);
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        run(&mut session, &mut scene, 2, |_| InputState::default());
        let initial = scene
            .objects
            .iter()
            .filter(|o| {
                o.components
                    .iter()
                    .any(|c| c.as_any().is::<TemplateAvatar>())
            })
            .count();
        assert!(initial > 0, "{}: avatars", template.id);
        let mut raw = RawInput::default();
        raw.keys_down.insert(Key::F6);
        let input = zestengine_core::resolve(&scheme, &raw, &InputState::default());
        run(&mut session, &mut scene, 1, |_| input.clone());
        let selectors: Vec<&TemplateAvatar> = scene
            .objects
            .iter()
            .filter_map(|o| {
                o.components
                    .iter()
                    .find_map(|c| c.as_any().downcast_ref::<TemplateAvatar>())
            })
            .collect();
        for avatar in selectors {
            assert_eq!(
                avatar.variant,
                AvatarVariant::Female,
                "{}: F6 switches",
                template.id
            );
            let male = scene
                .objects
                .iter()
                .find(|o| Some(o.id) == avatar.male.id())
                .unwrap();
            let female = scene
                .objects
                .iter()
                .find(|o| Some(o.id) == avatar.female.id())
                .unwrap();
            assert!(!male.enabled && female.enabled, "exactly one model enabled");
        }
    }
    let _ = std::fs::remove_dir_all(parent);
}

#[test]
fn template_ids_are_unique_and_default_exists() {
    let ids: HashSet<&str> = TEMPLATES.iter().map(|t| t.id).collect();
    assert_eq!(ids.len(), TEMPLATES.len());
    assert!(super::find(super::DEFAULT_TEMPLATE).is_some());
    assert!(
        TEMPLATES
            .iter()
            .any(|t| t.category == super::TemplateCategory::Vr)
    );
}

// ------------------------------------------------------- headless gameplay

fn obj(scene: &mut LoadedScene, pos: Vec3, components: Vec<Box<dyn Component>>) -> ObjectId {
    let id = ObjectId::new();
    scene.objects.push(SceneObject {
        net_role: Default::default(),
        component_roles: Default::default(),
        id,
        name: String::new(),
        render: None,
        extra_render: Vec::new(),
        source: ObjectSource::Empty,
        enabled: true,
        static_geometry: false,
        layer: 0,
        parent: None,
        translation: pos,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        components,
    });
    id
}

fn get<T: Component>(scene: &LoadedScene, id: ObjectId) -> &T {
    let i = scene.index_of(id).expect("object");
    scene.objects[i]
        .components
        .iter()
        .find_map(|c| c.as_any().downcast_ref::<T>())
        .expect("component")
}

fn input_with(keys_down: &[Key], mouse_down: &[MouseButton], prev: &InputState) -> InputState {
    let scheme = ControlScheme {
        name: "test".into(),
        actions: vec![
            super::builder::wasd("Move"),
            ActionBinding::button("Fire", [InputSource::Mouse(MouseButton::Left)]),
            ActionBinding::button("Crouch", [InputSource::Key(Key::C)]),
        ],
    };
    let mut raw = RawInput::default();
    raw.keys_down.extend(keys_down.iter().copied());
    raw.mouse_down.extend(mouse_down.iter().copied());
    zestengine_core::resolve(&scheme, &raw, prev)
}

fn run(
    session: &mut PlaySession,
    scene: &mut LoadedScene,
    frames: usize,
    input: impl Fn(usize) -> InputState,
) {
    let registry = ComponentRegistry::with_builtins();
    let mut shaders = ShaderLibrary::default();
    let net = NetView::default();
    let screen = zestengine_core::ScreenInfo::default();
    for f in 0..frames {
        let input = input(f);
        session.tick(
            &mut World {
                scene,
                renderer: None,
                registry: &registry,
                shaders: &mut shaders,
            },
            &TickInput {
                dt: 1.0 / 60.0,
                input: &input,
                net: &net,
                listener_fallback: Mat4::IDENTITY,
                screen: &screen,
            },
        );
    }
}

#[test]
fn hitscan_weapon_damages_health_through_physics() {
    use zestengine_core::BoxCollider;
    let mut scene = LoadedScene::empty();
    let shooter = obj(
        &mut scene,
        Vec3::new(0.0, 1.0, 0.0),
        vec![Box::new(HitscanWeapon {
            spread_deg: 0.0,
            automatic: false,
            damage: 30.0,
            ..Default::default()
        })],
    );
    let target = obj(
        &mut scene,
        Vec3::new(0.0, 1.0, -6.0),
        vec![
            Box::new(BoxCollider {
                size: [2.0, 2.0, 0.5],
                ..Default::default()
            }),
            Box::new(Health::default()),
        ],
    );
    let mut session = PlaySession::new(&scene, Path::new("."), None);
    // Frame 2 presses Fire (frames 0/1 let the physics query structure build).
    let held = input_with(&[], &[MouseButton::Left], &InputState::default());
    run(&mut session, &mut scene, 6, |f| {
        if f == 2 {
            held.clone()
        } else {
            InputState::default()
        }
    });
    assert_eq!(
        get::<Health>(&scene, target).current,
        70.0,
        "one shot landed"
    );
    let w = get::<HitscanWeapon>(&scene, shooter);
    assert_eq!((w.shots_fired, w.ammo), (1, 29));
    assert!(
        w.last_hit.is_some_and(|p| (p.z + 5.75).abs() < 0.05),
        "hit the near face: {:?}",
        w.last_hit
    );
}

#[test]
fn hud_binding_writes_health_text() {
    let mut scene = LoadedScene::empty();
    let player = obj(
        &mut scene,
        Vec3::ZERO,
        vec![Box::new(Health {
            max: 100.0,
            current: 40.0,
            ..Default::default()
        })],
    );
    let label = obj(
        &mut scene,
        Vec3::ZERO,
        vec![
            Box::new(UiText::default()),
            Box::new(HudBinding {
                source: ObjectRef::new(player),
                value: HudValue::Health,
                format: "HP {value}/{max}".into(),
                ..Default::default()
            }),
        ],
    );
    let mut session = PlaySession::new(&scene, Path::new("."), None);
    run(&mut session, &mut scene, 3, |_| InputState::default());
    assert_eq!(get::<UiText>(&scene, label).text, "HP 40/100");
}

#[test]
fn first_person_controller_walks_forward_on_w() {
    let mut scene = LoadedScene::empty();
    let player = obj(
        &mut scene,
        Vec3::ZERO,
        vec![Box::new(FirstPersonCharacter {
            lock_cursor: false,
            ..Default::default()
        })],
    );
    let mut session = PlaySession::new(&scene, Path::new("."), None);
    let w = input_with(&[Key::W], &[], &InputState::default());
    run(&mut session, &mut scene, 60, |_| w.clone());
    let i = scene.index_of(player).unwrap();
    let p = scene.objects[i].translation;
    assert!(p.z < -2.0, "moved forward (-Z): {p:?}");
    assert!(
        p.x.abs() < 1e-3 && p.y.abs() < 1e-3,
        "stayed on the floor line: {p:?}"
    );
}

#[test]
fn crouch_shrinks_the_capsule_and_waits_for_headroom() {
    use zestengine_core::{BoxCollider, Character};
    let build = |ceiling: bool| {
        let mut scene = LoadedScene::empty();
        obj(
            &mut scene,
            Vec3::new(0.0, -0.5, 0.0),
            vec![Box::new(BoxCollider {
                size: [20.0, 1.0, 20.0],
                ..Default::default()
            })],
        );
        if ceiling {
            // Underside at 1.45 m: a crouched (1.2 m) capsule fits, a standing one doesn't.
            obj(
                &mut scene,
                Vec3::new(0.0, 1.7, 0.0),
                vec![Box::new(BoxCollider {
                    size: [6.0, 0.5, 6.0],
                    ..Default::default()
                })],
            );
        }
        let player = obj(
            &mut scene,
            Vec3::ZERO,
            vec![
                Box::new(FirstPersonCharacter {
                    lock_cursor: false,
                    ..Default::default()
                }),
                Box::new(Character::default()),
            ],
        );
        (scene, player)
    };
    let crouch = input_with(&[Key::C], &[], &InputState::default());
    for ceiling in [false, true] {
        let (mut scene, player) = build(ceiling);
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        run(&mut session, &mut scene, 10, |_| crouch.clone());
        let cc = get::<Character>(&scene, player);
        assert!(
            (cc.height - 1.2).abs() < 1e-4 && (cc.center[1] - 0.6).abs() < 1e-4,
            "crouched capsule, feet kept: {} {:?}",
            cc.height,
            cc.center
        );
        run(&mut session, &mut scene, 10, |_| InputState::default());
        let (cc, fpc) = (
            get::<Character>(&scene, player),
            get::<FirstPersonCharacter>(&scene, player),
        );
        if ceiling {
            assert!(
                fpc.crouched && (cc.height - 1.2).abs() < 1e-4,
                "stays crouched under the ceiling"
            );
        } else {
            assert!(
                !fpc.crouched && (cc.height - 1.8).abs() < 1e-4,
                "stands back up in the open"
            );
        }
        let p = scene.objects[scene.index_of(player).unwrap()].translation;
        assert!(p.y.abs() < 0.05, "feet stay on the floor: {p:?}");
    }
}

#[test]
fn character_controllers_rest_on_the_floor_and_walk() {
    use zestengine_core::{BoxCollider, Character, Pawn};
    let w = input_with(&[Key::W], &[], &InputState::default());
    for fps in [false, true] {
        let mut scene = LoadedScene::empty();
        obj(
            &mut scene,
            Vec3::new(0.0, -0.5, 0.0),
            vec![Box::new(BoxCollider {
                size: [40.0, 1.0, 40.0],
                ..Default::default()
            })],
        );
        let mover: Box<dyn Component> = if fps {
            Box::new(FirstPersonCharacter {
                lock_cursor: false,
                ..Default::default()
            })
        } else {
            Box::new(Pawn::default())
        };
        // Spawned with its feet exactly on the floor.
        let player = obj(
            &mut scene,
            Vec3::ZERO,
            vec![mover, Box::new(Character::default())],
        );
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        let y = |scene: &LoadedScene| scene.objects[scene.index_of(player).unwrap()].translation.y;
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for f in 0..180 {
            run(&mut session, &mut scene, 1, |_| {
                if f < 60 {
                    InputState::default()
                } else {
                    w.clone()
                }
            });
            (lo, hi) = (lo.min(y(&scene)), hi.max(y(&scene)));
        }
        let p = scene.objects[scene.index_of(player).unwrap()].translation;
        assert!(
            lo > -1e-3 && hi < 0.03,
            "fps {fps}: never sinks into the floor: {lo}..{hi}"
        );
        assert!(p.z < -6.0, "fps {fps}: walks freely along it: {p}");
    }
}

#[test]
fn vr_character_smooth_movement_collides_with_walls() {
    use zestengine_core::{BoxCollider, Character, VrCharacter};
    let mut scene = LoadedScene::empty();
    obj(&mut scene, Vec3::new(0.0, -0.5, 0.0), vec![Box::new(BoxCollider { size: [40.0, 1.0, 40.0], ..Default::default() })]);
    obj(&mut scene, Vec3::new(0.0, 1.5, -2.0), vec![Box::new(BoxCollider { size: [4.0, 3.0, 0.1], ..Default::default() })]);
    let player = obj(&mut scene, Vec3::ZERO, vec![Box::new(VrCharacter::default()), Box::new(Character::default())]);
    let mut session = PlaySession::new(&scene, Path::new("."), None);
    let walk = input_with(&[Key::W], &[], &InputState::default());
    run(&mut session, &mut scene, 180, |_| walk.clone());
    let p = scene.objects[scene.index_of(player).unwrap()].translation;
    assert!(p.z > -1.7 && p.z < -1.5, "VR movement stops at the wall: {p}");
    assert!(p.y.abs() < 0.03, "grounded: {p}");
}

// ------------------------------------------------------- template playthroughs

/// A template's boot scene as a headless `LoadedScene` (no meshes), plus its bindings.
fn load_template(id: &str) -> (LoadedScene, zestengine_core::ControlScheme) {
    let registry = ComponentRegistry::with_builtins();
    let content = super::build(id).unwrap();
    let file = &content.scenes[0].1;
    let mut scene = LoadedScene::empty();
    scene.layers = file.layers.clone();
    push_entries(&mut scene, &registry, &file.entries);
    (scene, content.bindings.schemes[0].clone())
}

/// Append scene entries (parents local to `entries`) as headless objects.
fn push_entries(
    scene: &mut LoadedScene,
    registry: &ComponentRegistry,
    entries: &[zestengine_assets::SceneEntry],
) {
    let base = scene.objects.len();
    for e in entries {
        scene.objects.push(SceneObject {
            net_role: Default::default(),
            component_roles: Default::default(),
            id: e.id.parse().unwrap(),
            name: e.name.clone(),
            render: None,
            extra_render: Vec::new(),
            source: e.source.clone(),
            enabled: e.enabled,
            static_geometry: e.static_geometry,
            layer: e.layer,
            parent: e.parent.map(|p| p + base),
            translation: Vec3::from(e.translation),
            rotation: Quat::from_array(e.rotation),
            scale: Vec3::from(e.scale),
            components: e
                .components
                .iter()
                .filter_map(|c| registry.load(&c.kind, &c.data))
                .collect(),
        });
    }
}

/// Straight-down camera over `center` (screen top = -Z), 800 x 600.
fn top_down_screen(center: Vec3, height: f32) -> zestengine_core::ScreenInfo {
    zestengine_core::ScreenInfo {
        size: glam::Vec2::new(800.0, 600.0),
        view: Mat4::look_at_rh(center + Vec3::Y * height, center, Vec3::NEG_Z),
        proj: Mat4::perspective_rh(60f32.to_radians(), 800.0 / 600.0, 0.1, 500.0),
        ..Default::default()
    }
}

struct Frame {
    pointer: Option<glam::Vec2>,
    mouse: Vec<MouseButton>,
}

fn play(
    scene: &mut LoadedScene,
    scheme: &ControlScheme,
    base: &zestengine_core::ScreenInfo,
    frames: usize,
    f: impl Fn(usize) -> Frame,
) {
    let registry = ComponentRegistry::with_builtins();
    let mut shaders = ShaderLibrary::default();
    let net = NetView::default();
    let mut session = PlaySession::new(scene, Path::new("."), None);
    let mut state = InputState::default();
    for i in 0..frames {
        let frame = f(i);
        let mut raw = RawInput::default();
        raw.mouse_down.extend(frame.mouse.iter().copied());
        state = zestengine_core::resolve(scheme, &raw, &state);
        let mut screen = base.clone();
        screen.pointer = frame.pointer;
        session.tick(
            &mut World {
                scene,
                renderer: None,
                registry: &registry,
                shaders: &mut shaders,
            },
            &TickInput {
                dt: 1.0 / 60.0,
                input: &state,
                net: &net,
                listener_fallback: Mat4::IDENTITY,
                screen: &screen,
            },
        );
    }
}

fn idle(_: usize) -> Frame {
    Frame {
        pointer: None,
        mouse: Vec::new(),
    }
}

fn position(scene: &LoadedScene, name: &str) -> Vec3 {
    let i = scene
        .objects
        .iter()
        .position(|o| o.name == name)
        .unwrap_or_else(|| panic!("no {name}"));
    scene.world_transform(i).w_axis.truncate()
}

#[test]
fn every_template_runs_headless() {
    for t in TEMPLATES {
        let (mut scene, scheme) = load_template(t.id);
        play(
            &mut scene,
            &scheme,
            &zestengine_core::ScreenInfo::default(),
            90,
            idle,
        );
    }
}

#[test]
fn shooter_players_stand_on_the_ground() {
    for id in ["fps", "tps"] {
        let (mut scene, scheme) = load_template(id);
        play(
            &mut scene,
            &scheme,
            &zestengine_core::ScreenInfo::default(),
            120,
            idle,
        );
        let p = position(&scene, "Player");
        assert!(p.y.abs() < 0.15, "{id}: player rests on the ground: {p:?}");
    }
}

#[test]
fn topdown_click_walks_the_player_over_the_navmesh() {
    let (mut scene, scheme) = load_template("topdown");
    let screen = top_down_screen(Vec3::ZERO, 30.0);
    let start = position(&scene, "Player");
    play(&mut scene, &scheme, &screen, 300, |i| Frame {
        pointer: Some(glam::Vec2::new(600.0, 300.0)),
        mouse: if i == 30 {
            vec![MouseButton::Left]
        } else {
            Vec::new()
        },
    });
    let end = position(&scene, "Player");
    assert!(
        end.x > start.x + 7.0,
        "player walked toward the click: {start:?} -> {end:?}"
    );
}

#[test]
fn strategy_box_select_and_formation_move() {
    let (mut scene, scheme) = load_template("strategy");
    let screen = top_down_screen(Vec3::new(0.0, 0.0, 7.0), 30.0);
    play(&mut scene, &scheme, &screen, 330, |i| match i {
        30 => Frame {
            pointer: Some(glam::Vec2::new(1.0, 1.0)),
            mouse: vec![MouseButton::Left],
        },
        31..=33 => Frame {
            pointer: Some(glam::Vec2::new(799.0, 599.0)),
            mouse: vec![MouseButton::Left],
        },
        34 => Frame {
            pointer: Some(glam::Vec2::new(799.0, 599.0)),
            mouse: Vec::new(),
        },
        36 => Frame {
            pointer: Some(glam::Vec2::new(600.0, 300.0)),
            mouse: vec![MouseButton::Right],
        },
        _ => Frame {
            pointer: Some(glam::Vec2::new(400.0, 300.0)),
            mouse: Vec::new(),
        },
    });
    let cam = scene
        .objects
        .iter()
        .position(|o| o.name == "Camera")
        .unwrap();
    let selection = scene.objects[cam]
        .components
        .iter()
        .find_map(|c| c.as_any().downcast_ref::<zestengine_core::RtsSelection>())
        .unwrap();
    assert_eq!(selection.selected.len(), 8, "box selected every unit");
    let xs: Vec<f32> = (1..=8)
        .map(|k| position(&scene, &format!("Unit {k}")).x)
        .collect();
    assert!(
        xs.iter().all(|x| *x > 6.0),
        "units moved to the order point: {xs:?}"
    );
    let spread =
        xs.iter().cloned().fold(f32::MIN, f32::max) - xs.iter().cloned().fold(f32::MAX, f32::min);
    assert!(
        spread > 1.0,
        "units hold a formation instead of stacking: {xs:?}"
    );
}

#[test]
fn vr_player_desktop_grabs_what_it_looks_at_and_fires_it() {
    use zestengine_core::{Grabbable, VrCharacter, WeaponTrigger};
    let mut scene = LoadedScene::empty();
    let rig = obj(&mut scene, Vec3::ZERO, vec![]);
    let head = obj(&mut scene, Vec3::new(0.0, 1.65, 0.0), vec![]);
    let left = obj(&mut scene, Vec3::ZERO, vec![]);
    let right = obj(&mut scene, Vec3::ZERO, vec![]);
    for child in [1, 2, 3] {
        scene.objects[child].parent = Some(0);
    }
    let i = scene.index_of(rig).unwrap();
    scene.objects[i].components.push(Box::new(VrCharacter {
        head: ObjectRef::new(head),
        left_hand: ObjectRef::new(left),
        right_hand: ObjectRef::new(right),
        ..Default::default()
    }));
    let gun = obj(
        &mut scene,
        Vec3::new(0.0, 1.6, -1.5),
        vec![
            Box::new(Grabbable {
                snap_to_hand: true,
                ..Default::default()
            }),
            Box::new(HitscanWeapon {
                trigger: WeaponTrigger::Grabbed,
                automatic: false,
                spread_deg: 0.0,
                ..Default::default()
            }),
        ],
    );
    let target = obj(
        &mut scene,
        Vec3::new(0.2, 1.4, -8.0),
        vec![
            Box::new(zestengine_core::BoxCollider {
                size: [3.0, 3.0, 0.5],
                ..Default::default()
            }),
            Box::new(Health::default()),
        ],
    );
    let scheme = super::vr_shooter::vr_bindings(Vec::new()).schemes[0].clone();
    play(
        &mut scene,
        &scheme,
        &zestengine_core::ScreenInfo::default(),
        40,
        |i| Frame {
            pointer: None,
            mouse: match i {
                10..=39 => {
                    let mut m = vec![MouseButton::Right];
                    if i == 30 {
                        m.push(MouseButton::Left);
                    }
                    m
                }
                _ => Vec::new(),
            },
        },
    );
    assert!(
        get::<Grabbable>(&scene, gun).held,
        "grabbed by looking at it"
    );
    let hand = scene
        .world_transform(scene.index_of(right).unwrap())
        .w_axis
        .truncate();
    let gun_pos = scene
        .world_transform(scene.index_of(gun).unwrap())
        .w_axis
        .truncate();
    assert!(
        hand.distance(gun_pos) < 0.05,
        "snapped into the right hand: {hand:?} vs {gun_pos:?}"
    );
    assert_eq!(
        get::<HitscanWeapon>(&scene, gun).shots_fired,
        1,
        "the trigger fired it once"
    );
    assert!(
        get::<Health>(&scene, target).current < 100.0,
        "the shot landed"
    );
}

#[test]
fn vr_social_avatar_prefab_mirrors_the_local_player_for_others() {
    let content = super::build("vr_social").unwrap();
    let (_, prefab) = content
        .prefabs
        .iter()
        .find(|(path, _)| path.as_str() == "prefabs/net_avatar.prefab")
        .expect("avatar prefab");
    let (mut scene, scheme) = load_template("vr_social");
    let mut net_manager = scene.objects.iter().filter_map(|o| {
        o.components
            .iter()
            .find_map(|c| c.as_any().downcast_ref::<zestengine_core::NetworkManager>())
    });
    assert_eq!(
        net_manager.next().map(|m| m.player_prefab.as_str()),
        Some("prefabs/net_avatar.prefab")
    );
    let base = scene.objects.len();
    push_entries(
        &mut scene,
        &ComponentRegistry::with_builtins(),
        &prefab.instantiate([0.0, 0.0, 8.0]),
    );
    play(
        &mut scene,
        &scheme,
        &zestengine_core::ScreenInfo::default(),
        30,
        idle,
    );

    let find = |name: &str, from: usize| {
        (from..scene.objects.len())
            .find(|&i| scene.objects[i].name == name)
            .unwrap()
    };
    let player = find("VR Character", 0);
    let local_head = find("Head", 0);
    assert!(local_head < base);
    let avatar = find("Net Avatar", base);
    let avatar_head = find("Head", base);
    let avatar_hand = find("Right Hand", base);
    let local_hand = find("Right Hand", 0);
    let at = |i: usize| scene.world_transform(i).w_axis.truncate();
    assert!(
        at(avatar).distance(at(player)) < 1e-3,
        "avatar root sits on the rig: {:?} vs {:?}",
        at(avatar),
        at(player)
    );
    assert!(
        at(avatar_head).distance(at(local_head)) < 1e-3,
        "head follows: {:?} vs {:?}",
        at(avatar_head),
        at(local_head)
    );
    assert!(
        at(avatar_hand).distance(at(local_hand)) < 1e-3,
        "hand follows"
    );
    {
        let name = "Avatar Body";
        assert!(
            !scene.objects[find(name, base)].enabled,
            "{name} hidden for the owner"
        );
    }
}

#[test]
fn vr_player_reads_remapped_controller_actions() {
    use zestengine_core::{
        BindingSlot, Bindings, ConflictPolicy, ControlsInfo, Grabbable, VrButton, VrFrame, VrCharacter,
    };
    let mut scene = LoadedScene::empty();
    let rig = obj(&mut scene, Vec3::ZERO, vec![]);
    let head = obj(&mut scene, Vec3::ZERO, vec![]);
    let left = obj(&mut scene, Vec3::ZERO, vec![]);
    let right = obj(&mut scene, Vec3::ZERO, vec![]);
    for child in [1, 2, 3] {
        scene.objects[child].parent = Some(0);
    }
    let i = scene.index_of(rig).unwrap();
    scene.objects[i].components.push(Box::new(VrCharacter {
        head: ObjectRef::new(head),
        left_hand: ObjectRef::new(left),
        right_hand: ObjectRef::new(right),
        ..Default::default()
    }));
    let ball = obj(
        &mut scene,
        Vec3::new(0.25, 1.2, -0.3),
        vec![Box::new(Grabbable::default())],
    );
    // The player moved Grab Right from the grip to A.
    let mut bindings = Bindings::from_schemes(vec![ControlScheme::default_vr()]);
    bindings
        .rebind(
            "",
            "Grab Right",
            BindingSlot::Buttons,
            0,
            InputSource::VrButton(VrButton::A),
            ConflictPolicy::Steal,
        )
        .unwrap();
    let scheme = bindings.schemes[0].clone();
    let base = zestengine_core::ScreenInfo {
        controls: ControlsInfo {
            bindings: std::sync::Arc::new(bindings),
            ..Default::default()
        },
        ..Default::default()
    };

    let registry = ComponentRegistry::with_builtins();
    let mut shaders = ShaderLibrary::default();
    let net = NetView::default();
    let mut session = PlaySession::new(&scene, Path::new("."), None);
    let mut state = InputState::default();
    for frame in 0..40 {
        let vr = VrFrame {
            active: true,
            head: Some((Vec3::new(0.0, 1.65, 0.0), Quat::IDENTITY)),
            left_hand: Some((Vec3::new(-0.25, 1.2, -0.3), Quat::IDENTITY)),
            right_hand: Some((Vec3::new(0.25, 1.2, -0.3), Quat::IDENTITY)),
            right_grip: if frame < 20 { 1.0 } else { 0.0 },
            a: frame >= 25,
            ..Default::default()
        };
        let mut raw = RawInput::default();
        raw.set_vr(&vr);
        state = zestengine_core::resolve(&scheme, &raw, &state);
        let screen = zestengine_core::ScreenInfo { vr, ..base.clone() };
        session.tick(
            &mut World {
                scene: &mut scene,
                renderer: None,
                registry: &registry,
                shaders: &mut shaders,
            },
            &TickInput {
                dt: 1.0 / 60.0,
                input: &state,
                net: &net,
                listener_fallback: Mat4::IDENTITY,
                screen: &screen,
            },
        );
        if frame == 19 {
            assert!(
                !get::<Grabbable>(&scene, ball).held,
                "the grip no longer grabs"
            );
        }
    }
    assert!(
        get::<Grabbable>(&scene, ball).held,
        "A grabs after the remap"
    );
}

#[test]
fn invert_look_toggle_tunes_the_look_action_in_every_scheme() {
    use zestengine_core::{ComponentCommand, ControlsInfo, InputCommand, UiToggle};
    let ui = crate::ui_templates::UiTemplate::Settings.build();
    let toggle = ui
        .iter()
        .find(|n| n.name == "Invert Look")
        .and_then(|n| {
            n.components
                .iter()
                .find_map(|c| c.as_any().downcast_ref::<UiToggle>())
        })
        .expect("the Settings template has an Invert Look toggle");
    assert_eq!(toggle.invert_y_action, "Look");

    let mut scene = LoadedScene::empty();
    let id = obj(
        &mut scene,
        Vec3::ZERO,
        vec![Box::new(UiToggle {
            invert_y_action: "Look".into(),
            ..Default::default()
        })],
    );
    // The player inverted the mouse earlier (saved override); the pad isn't.
    let mut bindings = super::builder::shooter_bindings();
    bindings.schemes[0]
        .actions
        .iter_mut()
        .find(|a| a.name == "Look")
        .unwrap()
        .invert_y = true;
    let screen = zestengine_core::ScreenInfo {
        controls: ControlsInfo {
            bindings: std::sync::Arc::new(bindings),
            ..Default::default()
        },
        ..Default::default()
    };
    let registry = ComponentRegistry::with_builtins();
    let mut shaders = ShaderLibrary::default();
    let net = NetView::default();
    let mut session = PlaySession::new(&scene, Path::new("."), None);
    let mut tick = |scene: &mut LoadedScene| {
        session.tick(
            &mut World {
                scene,
                renderer: None,
                registry: &registry,
                shaders: &mut shaders,
            },
            &TickInput {
                dt: 1.0 / 60.0,
                input: &InputState::default(),
                net: &net,
                listener_fallback: Mat4::IDENTITY,
                screen: &screen,
            },
        )
    };
    let tunes = |cmds: Vec<ComponentCommand>| -> Vec<(String, String, Option<bool>)> {
        cmds.into_iter()
            .filter_map(|c| match c {
                ComponentCommand::Input(InputCommand::Tune {
                    scheme,
                    action,
                    tuning,
                }) => Some((scheme, action, tuning.invert_y)),
                _ => None,
            })
            .collect()
    };
    assert!(
        tunes(tick(&mut scene)).is_empty(),
        "showing the saved setting changes nothing"
    );
    assert!(
        get::<UiToggle>(&scene, id).on,
        "the toggle shows the saved inversion"
    );
    // The player unticks it: every scheme with a Look action follows.
    let i = scene.index_of(id).unwrap();
    scene.objects[i].components[0]
        .as_any_mut()
        .downcast_mut::<UiToggle>()
        .unwrap()
        .on = false;
    let got = tunes(tick(&mut scene));
    let schemes: Vec<&str> = got.iter().map(|(s, _, _)| s.as_str()).collect();
    assert_eq!(schemes.len(), 2, "keyboard + mouse and gamepad: {got:?}");
    assert!(
        got.iter()
            .all(|(_, a, inv)| a == "Look" && *inv == Some(false)),
        "{got:?}"
    );
    assert!(tunes(tick(&mut scene)).is_empty(), "applied once");
}

#[test]
fn walking_feeds_the_animator_a_steady_speed() {
    use zestengine_core::{Animator, Key};
    let (mut scene, scheme) = load_template("tps");
    let mut session = PlaySession::new(&scene, Path::new("."), None);
    let mut raw = RawInput::default();
    raw.keys_down.insert(Key::W);
    let walk = zestengine_core::resolve(&scheme, &raw, &InputState::default());
    let mut forward = Vec::new();
    for f in 0..120 {
        run(&mut session, &mut scene, 1, |_| walk.clone());
        let animator = scene.objects.iter().filter(|o| o.enabled).find_map(|o| o.components.iter().find_map(|c| c.as_any().downcast_ref::<Animator>()));
        if f >= 60 {
            forward.push(animator.and_then(|a| a.state.params.get("Forward").copied()).unwrap_or(0.0));
        }
    }
    let (lo, hi) = forward.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    assert!(lo > 3.5, "walking speed never drops out of the walk blend: {lo}..{hi}");
    assert!(hi - lo < 0.2, "steady: {lo}..{hi}");
}

/// No headset: the VR Social template plays flat. The VR Character simulates the
/// rig (spawn point, WASD walk, mouse-look pitch), the head and hands follow,
/// and nothing needs an XR session. Bounded so a hang fails the test.
#[test]
fn vr_social_plays_flat_without_a_headset() {
    use zestengine_core::VrCharacter;
    let started = std::time::Instant::now();
    let (mut scene, scheme) = load_template("vr_social");
    let mut session = PlaySession::new(&scene, Path::new("."), None);
    run(&mut session, &mut scene, 2, |_| InputState::default());
    let rig = |scene: &LoadedScene| {
        let i = scene.objects.iter().position(|o| o.components.iter().any(|c| c.as_any().is::<VrCharacter>())).expect("VR Character");
        scene.world_transform(i).w_axis.truncate()
    };
    let spawn = rig(&scene);
    assert!((spawn - Vec3::new(0.0, 0.0, 4.0)).with_y(0.0).length() < 0.01 && spawn.y.abs() < 0.03, "peer 0 spawns at Spawn 1 with capsule skin clearance: {spawn:?}");
    let mut raw = RawInput::default();
    raw.keys_down.insert(Key::W);
    let walk = zestengine_core::resolve(&scheme, &raw, &InputState::default());
    run(&mut session, &mut scene, 60, |_| walk.clone());
    let moved = rig(&scene);
    assert!(moved.z < spawn.z - 1.0, "W walks the simulated rig forward: {spawn:?} -> {moved:?}");
    let head = position(&scene, "Head");
    assert!((head.y - 1.65).abs() < 0.2 && (head.x - moved.x).abs() < 0.2, "the head rides the rig: {head:?}");
    let hand = position(&scene, "Right Hand");
    assert!(hand.distance(head) < 1.0, "the simulated hand stays in reach: {hand:?}");
    assert!(started.elapsed() < std::time::Duration::from_secs(60), "play frames stay bounded");
}

/// The welcome board's buttons answer a flat-screen click: the crosshair ray
/// (mouse-look locked) lands on the board and the press / release pair
/// reaches the button through a real egui context.
#[test]
fn vr_social_welcome_board_buttons_click_from_the_desktop_crosshair() {
    use crate::game_ui::{GameUi, UiInput, UiPointer, UiScope, desktop_ui_ray, layout_rects, ray_canvas_hit};
    let (mut scene, _) = load_template("vr_social");
    let board = scene.objects.iter().position(|o| o.name == "Welcome Board").expect("board");
    let button = scene.objects.iter().position(|o| o.name == "Toggle Lights Button").expect("button");
    let size = [1.4, 1.0];
    let px = [1400u32, 1000];
    let view_px = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(px[0] as f32, px[1] as f32));
    let rect = layout_rects(&scene, view_px, 1.0, UiScope::Canvas(board), None)
        .into_iter()
        .find(|(i, ..)| *i == button)
        .expect("button laid out")
        .1;
    // Aim a camera 2 m back from the board at the button's centre.
    let world = scene.world_transform(board);
    let local = Vec3::new((rect.center().x / px[0] as f32 - 0.5) * size[0], (0.5 - rect.center().y / px[1] as f32) * size[1], 0.0);
    let target = world.transform_point3(local);
    let eye = world.transform_point3(local + Vec3::new(0.0, 0.0, 2.0));
    let view = Mat4::look_at_rh(eye, target, Vec3::Y);
    let proj = Mat4::perspective_rh(70f32.to_radians(), 16.0 / 9.0, 0.05, 100.0);

    let ctx = egui::Context::default();
    let mut ui = GameUi::default();
    let mut clicks = Vec::new();
    for (down, pressed, released) in [(true, true, false), (false, false, true)] {
        let ray = desktop_ui_ray(view, proj, None, true, [down, pressed, released]).expect("ray");
        let (pos, _) = ray_canvas_hit(&ray, world, size, px).expect("crosshair is on the board");
        assert!(rect.contains(pos), "{pos:?} not in {rect:?}");
        let input = UiInput { pointer: UiPointer { pos: Some(pos), down, pressed, released }, ..Default::default() };
        let raw = egui::RawInput { screen_rect: Some(view_px), ..Default::default() };
        let _ = ctx.run(raw, |ctx| {
            clicks.extend(ui.run(ctx, view_px, &mut scene, Path::new("."), Some(&input), None, UiScope::Canvas(board)).clicks);
        });
    }
    assert_eq!(clicks, vec![scene.objects[button].id], "the button reported one click");
}

/// Every template scene carries exactly one `Directional Light` object (the
/// old environment sun), except those that intentionally have none.
#[test]
fn every_template_scene_has_one_directional_light_unless_intentionally_none() {
    const NO_DIRECTIONAL: &[&str] = &["vr_social"];
    for template in TEMPLATES {
        let content = super::build(template.id).unwrap();
        for (path, scene) in &content.scenes {
            assert!(!scene.environment.sun_enabled, "{}: {path} still enables the legacy sun", template.id);
            let dirs: Vec<_> = scene
                .entries
                .iter()
                .filter(|e| {
                    e.components.iter().any(|c| {
                        c.kind == "Light"
                            && ron::from_str::<zestengine_core::LightComponent>(&c.data).is_ok_and(|l| l.kind == zestengine_core::LightKind::Directional)
                    })
                })
                .collect();
            let want = usize::from(!NO_DIRECTIONAL.contains(&template.id));
            assert_eq!(dirs.len(), want, "{}: {path} directional lights", template.id);
            if let Some(d) = dirs.first() {
                assert_eq!(d.name, "Directional Light", "{}", template.id);
                assert!(matches!(d.source, ObjectSource::Light));
                assert!(d.enabled && d.parent.is_none());
            }
        }
    }
}

#[test]
fn empty_template_default_light_matches_the_old_default_sun() {
    let content = super::build("empty").unwrap();
    let scene = &content.scenes[0].1;
    let light = scene.entries.iter().find(|e| e.name == "Directional Light").expect("light");
    let l: zestengine_core::LightComponent = ron::from_str(&light.components[0].data).unwrap();
    assert_eq!((l.color, l.intensity), ([1.0, 0.98, 0.92], 3.0));
    let travel = Quat::from_array(light.rotation) * Vec3::NEG_Z;
    assert!(travel.distance(Vec3::new(-0.4, -1.0, -0.3).normalize()) < 1e-5);
    let names: Vec<_> = scene.entries.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"Main Camera") && names.contains(&"Ground") && names.contains(&"Cube"));
}

/// What every mesh entry draws with, per slot, plus its shadow / lightmap settings, as text lines.
/// Read from the renderer component when the entry has one, else from the legacy loose fields,
/// so the same function describes a scene before and after the renderer split.
fn look_lines(tag: &str, entries: &[zestengine_assets::SceneEntry]) -> Vec<String> {
    use zestengine_core::renderer_of;
    let registry = ComponentRegistry::with_builtins();
    let mut out = Vec::new();
    for (i, e) in entries.iter().enumerate() {
        let slots = zestengine_assets::slot_count_of(&e.source);
        if slots == 0 {
            continue;
        }
        let component = e.components.iter().find(|c| zestengine_core::is_renderer_kind(&c.kind)).and_then(|c| c.load(&registry));
        let (cast, scale, refs): (_, _, Vec<MaterialRef>) = match component.as_deref().and_then(renderer_of) {
            Some(r) => (r.cast_shadows(), r.lightmap_scale(), (0..slots).map(|k| zestengine_assets::material_ref_of(&r.slot(k))).collect()),
            None => {
                let all: Vec<&MaterialRef> = std::iter::once(&e.material).chain(e.extra_materials.iter()).collect();
                (e.cast_shadows, e.lightmap_scale, (0..slots).map(|k| all.get(k).map_or(MaterialRef::Model, |m| (*m).clone())).collect())
            }
        };
        for (k, m) in refs.iter().enumerate() {
            // Inline snapshots are long; their RON hash keeps the golden readable.
            let text = match m {
                MaterialRef::Inline { .. } => {
                    let ron = ron::to_string(m).unwrap();
                    let h = ron.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3));
                    format!("Inline#{h:016x}")
                }
                other => ron::to_string(other).unwrap(),
            };
            out.push(format!("{tag}|{i}|{}|{k}|{text}|{cast:?}|{scale}", e.name));
        }
    }
    out
}

const GOLDEN_LOOKS: &str = include_str!("golden_looks.txt");

/// The "looks identical" golden: every template's boot scene and prefabs, per object and slot,
/// recorded from the loose-field loader BEFORE the renderer split. The migrated component
/// loader must reproduce it exactly. Re-record with `ZESTENGINE_BLESS=1` only after a template
/// is deliberately changed.
#[test]
fn every_template_draws_with_the_same_materials_and_settings_as_before_the_split() {
    let parent = temp_dir("golden-looks");
    let mut lines = Vec::new();
    for template in TEMPLATES {
        let root = crate::bundle::scaffold_project(&parent, template.id, template.id).unwrap();
        let project = zestengine_assets::load_project_file(&root).unwrap();
        let boot = project.boot_scene.clone().unwrap();
        let scene = zestengine_assets::load_scene_file(root.join(&boot)).unwrap();
        lines.extend(look_lines(&format!("{}:{boot}", template.id), &scene.entries));
        let mut prefabs: Vec<_> = std::fs::read_dir(root.join("prefabs")).unwrap().flatten().map(|e| e.path()).collect();
        prefabs.sort();
        for path in prefabs {
            let prefab = crate::prefab::Prefab::load(&path).unwrap();
            lines.extend(look_lines(&format!("{}:{}", template.id, path.file_name().unwrap().to_string_lossy()), &prefab.entries));
        }
    }
    let _ = std::fs::remove_dir_all(&parent);
    let text = lines.join("\n") + "\n";
    if std::env::var_os("ZESTENGINE_BLESS").is_some() {
        std::fs::write(concat!(env!("CARGO_MANIFEST_DIR"), "/src/templates/golden_looks.txt"), &text).unwrap();
        return;
    }
    assert!(!lines.is_empty());
    let want: Vec<&str> = GOLDEN_LOOKS.lines().collect();
    assert_eq!(lines.len(), want.len(), "the number of drawn slots changed");
    for (got, want) in lines.iter().zip(want) {
        assert_eq!(got, want);
    }
}
