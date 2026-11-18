//! Native systems operate directly on the scene's permanent typed components.
//! Transform values are authoritative ECS data, shared with authoring views.

use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use zestengine_core::{ComponentCommand, ComponentRegistry};
use zestengine_core::ecs::{HierarchyError, Stage, SystemContext, SystemServices};

use super::{LoadedScene, PassEnv};

struct SceneMaterials<'a>(&'a [super::MaterialEntry]);
impl zestengine_core::ecs::MaterialAccess for SceneMaterials<'_> {
    fn material_params(&self, world: &zestengine_core::ecs::SystemWorld<'_>, object: zestengine_core::ObjectId, slot: usize) -> Option<zestengine_core::MaterialParams> {
        let entity = world.entity(object)?;
        let material = if slot == 0 { world.get::<super::metadata::Render>(entity)?.0?.material }
            else { world.get::<super::metadata::ExtraRender>(entity)?.0.get(slot - 1)?.material };
        Some(super::material_from_model(&self.0.get(material)?.model).0)
    }
}

impl LoadedScene {
    pub fn run_ecs_systems(&mut self, registry: &ComponentRegistry, stage: Stage, env: &PassEnv<'_>) -> Result<Vec<ComponentCommand>, HierarchyError> {
        self.run_ecs_event_systems(registry, stage, env, &[])
    }

    pub fn run_ecs_event_systems(&mut self, registry: &ComponentRegistry, stage: Stage, env: &PassEnv<'_>, scene_events: &[zestengine_core::ecs::SceneEvent]) -> Result<Vec<ComponentCommand>, HierarchyError> {
        if !registry.has_runnable_systems(stage, &self.objects.world) { return Ok(Vec::new()); }
        self.objects.propagate_hierarchy()?;
        let mut commands = Vec::new();
        let materials = SceneMaterials(&self.materials);
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            registry.run_systems(stage, &self.objects.world, &mut SystemContext {
                dt: env.dt, unscaled_dt: env.unscaled_dt, time: env.time,
                water_time: env.water_time, input: env.input, commands: &mut commands,
                services: SystemServices { physics: env.physics, nav: env.nav, net: Some(env.net), screen: Some(env.screen), events: env.events, components: Some(&self.objects), scene_events, order: Some(self.objects.indices()), materials: Some(&materials), save: Some(env.save), strings: Some(env.strings), time_scale: Some(env.time_scale) },
            });
        }));
        if let Err(panic) = outcome { resume_unwind(panic); }
        Ok(commands)
    }

    pub fn invoke_native_action(&self, call: zestengine_core::ecs::NativeActionCall<'_>, out: &mut Vec<ComponentCommand>) -> Result<(), String> {
        self.objects.query(|world, _| zestengine_core::ecs::invoke_action_in(world, &self.objects, call, out))
    }

    pub fn fire_native_actions(&self, bindings: &[zestengine_core::EventBinding], source: &zestengine_core::FireSource<'_>, out: &mut Vec<ComponentCommand>) {
        self.objects.query(|world, _| {
            let reader = zestengine_core::ecs::NativeProperties::new(world, &self.objects);
            for binding in bindings { binding.fire_native(&reader, source, out); }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use zestengine_core::{Bob, Disabled, ObjectId, Spin, Transform};
    use zestengine_core::ecs::{Active, Owner, SystemWorld};
    use crate::scene::SceneObject;

    fn with_env(time: f32, run: impl FnOnce(&PassEnv<'_>)) {
        let input = Default::default();
        let net = Default::default();
        let screen = Default::default();
        let save = Default::default();
        let strings = Default::default();
        run(&PassEnv {
            dt: 0.25, unscaled_dt: 0.25, time, water_time: time, time_scale: 1.0,
            input: &input, net: &net, physics: None, nav: None, events: &[],
            screen: &screen, save: &save, strings: &strings,
        });
    }

    fn native_script_file(dir: &std::path::Path, name: &str, wat: &str) -> String {
        let path = dir.join(name);
        std::fs::write(&path, wat::parse_str(wat).unwrap()).unwrap();
        path.to_string_lossy().into_owned()
    }

    fn script_at(path: &str) -> zestengine_core::Script {
        let mut script = zestengine_core::Script::default();
        script.module = path.to_owned();
        script
    }

    fn tick_scripts(scene: &mut LoadedScene, session: &mut crate::play::PlaySession, registry: &ComponentRegistry) {
        use crate::play::{TickInput, World};
        let mut shaders = crate::shaders::ShaderLibrary::default();
        let input = Default::default();
        let net = Default::default();
        let screen = Default::default();
        let output = session.tick(&mut World { scene, renderer: None, registry, shaders: &mut shaders },
            &TickInput { dt: 1.0 / 60.0, input: &input, net: &net, listener_fallback: glam::Mat4::IDENTITY, screen: &screen });
        assert!(output.is_empty(), "guest effects should be applied by the native play session: {output:?}");
    }

    const SCRIPT_COUNTS: &str = r#"(module
        (import "env" "get_prop" (func $get (param i32) (result f32)))
        (import "env" "set_prop" (func $set (param i32 f32)))
        (import "env" "get_position" (func $position (result f32 f32 f32)))
        (import "env" "set_position" (func $move (param f32 f32 f32)))
        (func $advance (param $amount f32) (local $x f32) (local $y f32) (local $z f32)
            (call $position) (local.set $z) (local.set $y) (local.set $x)
            (call $move (f32.add (local.get $x) (local.get $amount)) (local.get $y) (local.get $z)))
        (func (export "start") (call $advance (f32.const 1))
            (call $set (i32.const 0) (f32.add (call $get (i32.const 0)) (f32.const 1))))
        (func (export "update") (call $advance (f32.const 2))
            (call $set (i32.const 1) (f32.add (call $get (i32.const 1)) (f32.const 1)))))"#;

    #[test]
    fn native_script_late_additions_disabled_state_and_duplicate_startup_run_in_play() {
        use crate::play::PlaySession;
        use zestengine_core::Script;
        let dir = std::env::temp_dir().join(format!("zest-native-scripts-{}", ObjectId::new()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = native_script_file(&dir, "counts.wasm", SCRIPT_COUNTS);
        let registry = ComponentRegistry::with_builtins();
        let mut scene = LoadedScene::empty();
        let parent = SceneObject::new_empty(ObjectId::new(), "Parent");
        parent.transform_mut().translation = Vec3::X * 10.0;
        scene.objects.push(parent);
        scene.objects.push(SceneObject::new_empty(ObjectId::new(), "Script owner").with_parent(Some(0)));
        let mut session = PlaySession::new(&scene, std::path::Path::new("."), None);
        tick_scripts(&mut scene, &mut session, &registry);
        scene.objects[1].transform_mut().translation = Vec3::X * 5.0;
        scene.objects[1].components.push(Box::new(script_at(&path)));
        tick_scripts(&mut scene, &mut session, &registry);
        assert_eq!(&scene.objects[1].components.get_typed::<Script>().unwrap().props[..2], &[1.0, 1.0]);
        assert_eq!(scene.world_transform(1).w_axis.x, 18.0);
        scene.objects[1].components.set_enabled(0, false);
        tick_scripts(&mut scene, &mut session, &registry);
        assert_eq!(&scene.objects[1].components.get_including_disabled::<Script>().unwrap().props[..2], &[1.0, 1.0]);
        scene.objects[1].components.set_enabled(0, true);
        tick_scripts(&mut scene, &mut session, &registry);
        assert_eq!(&scene.objects[1].components.get_typed::<Script>().unwrap().props[..2], &[1.0, 2.0]);
        scene.objects[1].components.replace_all(vec![Box::new(script_at(&path)), Box::new(script_at(&path))]);
        scene.objects[1].transform_mut().translation = Vec3::X * 5.0;
        tick_scripts(&mut scene, &mut session, &registry);
        let values: Vec<_> = scene.objects[1].components.iter_typed::<Script>().map(|script| script.props).collect();
        assert!(values.iter().all(|props| props[..2] == [1.0, 1.0]));
        assert_eq!(scene.world_transform(1).w_axis.x, 21.0, "duplicate instances edit the same current native transform");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn native_wasm_reads_script_updated_parented_pose_and_invokes_builtin_actions_in_play() {
        use crate::play::PlaySession;
        use zestengine_core::{Health, WasmScript};
        let dir = std::env::temp_dir().join(format!("zest-native-wasm-play-{}", ObjectId::new()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = native_script_file(&dir, "motion.wasm", SCRIPT_COUNTS);
        let manifest = r#"(capabilities: ["read", "invoke", "state"])"#;
        let encoded: String = manifest.bytes().map(|byte| format!("\\{byte:02x}")).collect();
        let sampler = format!(r#"(module (@custom "zest_manifest" "{encoded}")
            (import "host" "get_property" (func $get (param i32 i32 i32 i32 i32 i32 i32) (result i32)))
            (import "host" "set_float" (func $set (param i32 i32 f32)))
            (import "host" "invoke" (func $invoke (param i32 i32 i32 i32 i32 i32 i32) (result i32)))
            (memory (export "memory") 1)
            (data (i32.const 200) "position")
            (data (i32.const 220) "x")
            (data (i32.const 240) "Health")
            (data (i32.const 260) "Damage")
            (data (i32.const 280) "\01\00\00\00\02\00\00\c8\41")
            (func (export "zest_update") (param f32)
                (drop (call $get (i32.const 100) (i32.const 0) (i32.const 0) (i32.const 200) (i32.const 8) (i32.const 300) (i32.const 32)))
                (call $set (i32.const 220) (i32.const 1) (f32.load offset=1 (i32.const 300)))
                (drop (call $invoke (i32.const 100) (i32.const 240) (i32.const 6) (i32.const 260) (i32.const 6) (i32.const 280) (i32.const 9)))))"#);
        let sampler = native_script_file(&dir, "sampler.wasm", &sampler);
        let registry = ComponentRegistry::with_builtins();
        let mut scene = LoadedScene::empty();
        let parent = SceneObject::new_empty(ObjectId::new(), "Parent");
        parent.transform_mut().translation = Vec3::X * 10.0;
        scene.objects.push(parent);
        let mut actor = SceneObject::new_empty(ObjectId::new(), "Actor").with_parent(Some(0));
        actor.transform_mut().translation = Vec3::X * 5.0;
        actor.components.replace_all(vec![Box::new(script_at(&script)), Box::new(WasmScript::new(sampler)), Box::new(Health::default())]);
        scene.objects.push(actor);
        let mut session = PlaySession::new(&scene, std::path::Path::new("."), None);
        tick_scripts(&mut scene, &mut session, &registry);
        assert_eq!(scene.objects[1].components.get_typed::<WasmScript>().unwrap().float("x"), Some(18.0));
        assert_eq!(scene.objects[1].components.get_typed::<Health>().unwrap().current, 75.0);
        tick_scripts(&mut scene, &mut session, &registry);
        assert_eq!(scene.objects[1].components.get_typed::<WasmScript>().unwrap().float("x"), Some(20.0));
        assert_eq!(scene.objects[1].components.get_typed::<Health>().unwrap().current, 50.0);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn native_motion_retains_permanent_components_and_unsaved_state_between_frames() {
        let registry = ComponentRegistry::with_builtins();
        let mut scene = LoadedScene::empty();
        let mut object = SceneObject::new_empty(ObjectId::new(), "Object");
        object.transform_mut().translation = Vec3::X * 10.0;
        object.components.replace_all(vec![
            Box::new(Bob::default()), Box::new(Disabled(Box::new(Spin::default()))),
            Box::new(Spin { degrees_per_second: 90.0, ..Default::default() }),
            Box::new(Spin { degrees_per_second: 45.0, ..Default::default() }),
        ]);
        scene.objects.push(object);
        with_env(0.25, |env| { scene.run_ecs_systems(&registry, Stage::Update, env).unwrap(); });
        with_env(0.5, |env| { scene.run_ecs_systems(&registry, Stage::Update, env).unwrap(); });
        let object = &scene.objects[0];
        assert_eq!(object.translation().x, 10.0);
        assert!((object.translation().y - 0.5).abs() < 1e-5, "Bob state must survive stages without accumulating its previous offset");
        assert_eq!(object.components.len(), 4);
        assert!(object.components.get(0).unwrap().as_any().is::<Bob>());
        assert!(!zestengine_core::component_enabled(object.components.get(1).unwrap().as_ref()));
        assert!(object.components.get_typed_at::<zestengine_core::ecs::DisabledComponent<Spin>>(1).is_some());
        let expected = glam::Quat::from_rotation_y(67.5_f32.to_radians());
        assert!(object.rotation().abs_diff_eq(expected, 1e-5));
        assert_eq!(scene.objects.world.with_world(|world| world.query::<&Spin>().iter().count()), 2, "both Spin instances remain in permanent ECS storage");
    }

    #[test]
    fn native_ai_play_loop_uses_physics_and_current_motion_before_ticking_trees() {
        use zestengine_core::{Animator, BoxCollider, NavAgent};
        use zestengine_core::ai::{BehaviourTreeAgent, Blackboard, BtAction, BtNode, BtStatus, Perception, PerceptionComponent};
        use crate::play::{PlaySession, TickInput, World};
        let registry = ComponentRegistry::with_builtins();
        let mut scene = LoadedScene::empty();
        let parent = SceneObject::new_empty(ObjectId::new(), "Parent");
        parent.transform_mut().translation = Vec3::X * 10.0;
        scene.objects.push(parent);
        let target_id = ObjectId::new();
        let mut target = SceneObject::new_empty(target_id, "Target");
        target.transform_mut().translation = Vec3::new(10.0, 0.0, -5.0);
        let mut bob = Bob::default();
        bob.axis = Vec3::X.to_array();
        bob.amplitude = 2.0;
        target.components.replace_all(vec![Box::new(bob), Box::new(NavAgent::default())]);
        scene.objects.push(target);
        let mut blackboard = Blackboard::default();
        blackboard.set_string("message", "Spotted");
        let mut actor = SceneObject::new_empty(ObjectId::new(), "Observer").with_parent(Some(0));
        // Perception's update must precede the tree even with this authoring order.
        actor.components.replace_all(vec![
            Box::new(BehaviourTreeAgent { blackboard, root: BtNode::Sequence { children: vec![
                BtNode::Condition { key: "has_target".into(), value: true, invert: false },
                BtNode::Action { action: BtAction::SetAnimatorBool, param: "alert=has_target".into() },
                BtNode::Action { action: BtAction::Emit, param: "message".into() },
                BtNode::Action { action: BtAction::Chase, param: "target_id".into() },
            ] }, ..Default::default() }),
            Box::new(PerceptionComponent { target: target_id.into(), perception: Perception { hearing_range: 0.0, ..Default::default() }, ..Default::default() }),
            Box::new(NavAgent::default()), Box::new(Animator::default()),
            Box::new(Disabled(Box::new(BehaviourTreeAgent::default()))),
        ]);
        let actor_id = actor.id();
        scene.objects.push(actor);
        let actor_entity = scene.objects[2].entity();
        let wall_id = ObjectId::new();
        let mut wall = SceneObject::new_empty(wall_id, "Wall");
        wall.transform_mut().translation = Vec3::new(11.0, 0.0, -2.5);
        wall.components.push(Box::new(BoxCollider { size: [4.0, 4.0, 0.25], ..Default::default() }));
        scene.objects.push(wall);
        let mut session = PlaySession::new(&scene, std::path::Path::new("."), None);
        let mut shaders = crate::shaders::ShaderLibrary::default();
        let input = Default::default();
        let net = Default::default();
        let screen = Default::default();
        let tick = TickInput { dt: 1.0 / 60.0, input: &input, net: &net, listener_fallback: glam::Mat4::IDENTITY, screen: &screen };
        session.tick(&mut World { scene: &mut scene, renderer: None, registry: &registry, shaders: &mut shaders }, &tick);
        let actor = &scene.objects[scene.index_of(actor_id).unwrap()];
        assert_eq!(actor.components.get_typed::<PerceptionComponent>().unwrap().state.last_seen, None, "native stage must receive the real physics query hub");
        assert_eq!(actor.components.get_typed::<BehaviourTreeAgent>().unwrap().last_status, Some(BtStatus::Failure));
        scene.remove_object(scene.index_of(wall_id).unwrap());
        scene.objects.swap(0, 2);
        session.scene_changed(&scene);
        session.tick(&mut World { scene: &mut scene, renderer: None, registry: &registry, shaders: &mut shaders }, &tick);
        let i = scene.index_of(actor_id).unwrap();
        let actor = &scene.objects[i];
        assert_eq!(actor.entity(), actor_entity);
        let perceived = actor.components.get_typed::<PerceptionComponent>().unwrap().state;
        assert_eq!(perceived.last_seen, Some(target_id));
        let target_position = scene.world_transform(scene.index_of(target_id).unwrap()).w_axis.truncate();
        assert!(perceived.last_target.abs_diff_eq(target_position, 1e-5), "perception must see Bob's same-stage native transform write");
        assert!(target_position.x > 10.0);
        let tree = actor.components.get_typed::<BehaviourTreeAgent>().unwrap();
        assert_eq!(tree.last_status, Some(BtStatus::Running));
        assert_eq!(tree.blackboard.get_object("target_id"), Some(target_id));
        assert_eq!(actor.components.get_typed::<NavAgent>().unwrap().state.destination, Some(target_position));
        assert!(actor.components.get_typed::<Animator>().unwrap().get_bool("alert"));
        assert_eq!(scene.objects[scene.index_of(target_id).unwrap()].components.get_typed::<NavAgent>().unwrap().state.destination, None);
        assert!(actor.components.get_typed_at::<zestengine_core::ecs::DisabledComponent<BehaviourTreeAgent>>(4).unwrap().last_status.is_none());
        drop(tree);
        scene.objects[i].components.set_enabled(0, false);
        with_env(0.5, |env| {
            let commands = scene.run_ecs_systems(&registry, Stage::Update, env).unwrap();
            assert!(commands.is_empty(), "disabled trees cannot emit events");
        });
        assert_eq!(scene.objects[i].components.get_including_disabled::<BehaviourTreeAgent>().unwrap().last_status, Some(BtStatus::Running), "disabling retains unsaved tree state");
    }

    #[test]
    fn native_controls_receive_authority_input_and_ui_focus_for_duplicate_and_disabled_data() {
        use zestengine_core::{ActionValue, InputState, NetView, NetworkObject, ScreenInfo, Sync, Vehicle};
        let registry = ComponentRegistry::with_builtins();
        let mut scene = LoadedScene::empty();
        scene.objects.push(SceneObject::new_empty(ObjectId::new(), "Parent"));
        let id = ObjectId::new();
        let mut vehicle = SceneObject::new_empty(id, "Vehicle").with_parent(Some(0));
        vehicle.components.replace_all(vec![Box::new(Vehicle::default()), Box::new(Vehicle::default()),
            Box::new(Disabled(Box::new(Vehicle { throttle: 7.0, ..Default::default() }))), Box::new(Sync::default()), Box::new(NetworkObject::default())]);
        scene.objects.push(vehicle);
        let input = InputState::from_actions(vec![
            ("Move".into(), ActionValue { down: true, axis2: glam::Vec2::new(0.4, 0.8), ..Default::default() }),
            ("Grab".into(), ActionValue { down: true, just_pressed: true, ..Default::default() }),
        ], glam::Vec2::ZERO);
        let mut net = NetView { connected: true, local_peer: 2, ..Default::default() };
        net.owners.insert(id, 3);
        let mut owned = net.clone();
        owned.owners.insert(id, 2);
        with_env(0.0, |env| {
            let mut frame = PassEnv { input: &input, net: &net, ..*env };
            scene.run_ecs_systems(&registry, Stage::FixedUpdate, &frame).unwrap();
            assert_eq!(scene.objects[1].components.get_typed::<Vehicle>().unwrap().throttle, 0.0);
            let commands = scene.run_ecs_systems(&registry, Stage::Update, &frame).unwrap();
            assert!(matches!(commands.as_slice(), [ComponentCommand::RequestOwnership(target)] if *target == id));
            frame.net = &owned;
            scene.run_ecs_systems(&registry, Stage::FixedUpdate, &frame).unwrap();
            for vehicle in scene.objects[1].components.iter_typed::<Vehicle>() {
                assert_eq!((vehicle.throttle, vehicle.steer), (0.8, 0.4));
            }
            let commands = scene.run_ecs_systems(&registry, Stage::Update, &frame).unwrap();
            assert!(matches!(commands.as_slice(), [ComponentCommand::ReleaseOwnership(target)] if *target == id));
            let typing = ScreenInfo { ui_text_editing: true, ..Default::default() };
            frame.screen = &typing;
            scene.run_ecs_systems(&registry, Stage::FixedUpdate, &frame).unwrap();
            for vehicle in scene.objects[1].components.iter_typed::<Vehicle>() { assert_eq!(vehicle.throttle, 0.0); }
            assert_eq!(scene.objects[1].components.get_typed_at::<zestengine_core::ecs::DisabledComponent<Vehicle>>(2).unwrap().throttle, 7.0);
            scene.objects[0].set_enabled(false);
            assert!(scene.run_ecs_systems(&registry, Stage::Update, &frame).unwrap().is_empty());
        });
    }

    #[test]
    fn native_empty_stages_skip_hierarchy_work_until_matching_typed_data_exists() {
        let registry = ComponentRegistry::with_builtins();
        let mut scene = LoadedScene::empty();
        scene.objects.push(SceneObject::new_empty(ObjectId::new(), "Object"));
        scene.objects[0].transform_mut().translation = Vec3::X * 10.0;
        with_env(0.0, |env| {
            scene.run_ecs_systems(&registry, Stage::FixedUpdate, env).unwrap();
            assert_eq!(scene.objects.world.get::<zestengine_core::ecs::GlobalTransform>(scene.objects[0].entity()).unwrap().0.w_axis.x, 0.0);
            scene.objects[0].components.push(Box::new(zestengine_core::Vehicle::default()));
            scene.run_ecs_systems(&registry, Stage::FixedUpdate, env).unwrap();
            assert_eq!(scene.objects.world.get::<zestengine_core::ecs::GlobalTransform>(scene.objects[0].entity()).unwrap().0.w_axis.x, 10.0);
        });
    }

    fn character_tick(scene: &mut LoadedScene, session: &mut crate::play::PlaySession, registry: &ComponentRegistry) -> Vec<ComponentCommand> {
        let mut shaders = crate::shaders::ShaderLibrary::default();
        let input = Default::default();
        let net = Default::default();
        let screen = Default::default();
        session.tick(&mut crate::play::World { scene, renderer: None, registry, shaders: &mut shaders },
            &crate::play::TickInput { dt: 1.0 / 60.0, input: &input, net: &net, listener_fallback: glam::Mat4::IDENTITY, screen: &screen })
    }

    #[test]
    fn native_controller_initialization_tracks_new_instances_on_already_started_objects() {
        use zestengine_core::{CameraComponent, FirstPersonCharacter};
        let registry = ComponentRegistry::with_builtins();
        let mut scene = LoadedScene::empty();
        scene.objects.push(SceneObject::new_empty(ObjectId::new(), "Parent"));
        scene.objects.push(SceneObject::new_empty(ObjectId::new(), "Player").with_parent(Some(0)));
        let camera_id = ObjectId::new();
        let mut camera = SceneObject::new_empty(camera_id, "Camera").with_parent(Some(1));
        camera.components.push(Box::new(CameraComponent::default()));
        scene.objects.push(camera);
        let entity = scene.objects[1].entity();
        let mut session = crate::play::PlaySession::new(&scene, std::path::Path::new("."), None);
        character_tick(&mut scene, &mut session, &registry);
        scene.objects[0].set_enabled(false);
        scene.objects[1].components.push(Box::new(FirstPersonCharacter { camera: camera_id.into(), ..Default::default() }));
        character_tick(&mut scene, &mut session, &registry);
        assert!(!scene.objects[1].components.get_typed::<FirstPersonCharacter>().unwrap().started);
        scene.objects[0].set_enabled(true);
        let commands = character_tick(&mut scene, &mut session, &registry);
        assert!(scene.objects[1].components.get_typed::<FirstPersonCharacter>().unwrap().started);
        assert_eq!(commands.iter().filter(|command| matches!(command, ComponentCommand::SetCursorLocked(true))).count(), 1);
        assert!(!character_tick(&mut scene, &mut session, &registry).iter().any(|command| matches!(command, ComponentCommand::SetCursorLocked(_))));
        scene.objects[1].components.set_enabled(0, false);
        character_tick(&mut scene, &mut session, &registry);
        scene.objects[1].components.set_enabled(0, true);
        assert!(!character_tick(&mut scene, &mut session, &registry).iter().any(|command| matches!(command, ComponentCommand::SetCursorLocked(_))), "reenabling retains initialization state");
        scene.objects[1].components.replace(0, Box::new(FirstPersonCharacter { camera: camera_id.into(), ..Default::default() }));
        let commands = character_tick(&mut scene, &mut session, &registry);
        assert_eq!(commands.iter().filter(|command| matches!(command, ComponentCommand::SetCursorLocked(true))).count(), 1, "replacement data initializes once");
        assert_eq!(scene.objects[1].entity(), entity);
    }

    #[test]
    fn native_pawn_spawn_and_fallback_floor_use_parent_local_coordinates() {
        use zestengine_core::{Pawn, SpawnPoint};
        let registry = ComponentRegistry::with_builtins();
        let mut scene = LoadedScene::empty();
        let parent = SceneObject::new_empty(ObjectId::new(), "Parent");
        parent.transform_mut().translation = Vec3::new(4.0, 4.0, 0.0);
        scene.objects.push(parent);
        let mut pawn = Pawn::default();
        pawn.spawn_tag = "player".into();
        let mut actor = SceneObject::new_empty(ObjectId::new(), "Pawn").with_parent(Some(0));
        actor.components.push(Box::new(pawn));
        scene.objects.push(actor);
        let expected = Vec3::new(12.0, 9.0, 3.0);
        let mut spawn = SceneObject::new_empty(ObjectId::new(), "Spawn");
        spawn.transform_mut().translation = expected;
        spawn.components.push(Box::new(SpawnPoint::default()));
        scene.objects.push(spawn);
        with_env(0.25, |env| {
            scene.run_ecs_systems(&registry, Stage::Start, env).unwrap();
            assert_eq!(scene.world_transform(1).w_axis.truncate(), expected);
            scene.run_ecs_systems(&registry, Stage::Update, env).unwrap();
            assert_eq!(scene.world_transform(1).w_axis.truncate(), expected, "fallback gravity cannot clamp local height to world height");
            assert!(scene.run_ecs_systems(&registry, Stage::Start, env).unwrap().is_empty());
        });
    }

    #[test]
    fn native_constraint_initializes_before_motion_and_preserves_rest_across_disable() {
        use zestengine_core::{Constraint, ConstraintKind, ConstraintSource};
        let registry = ComponentRegistry::with_builtins();
        let mut scene = LoadedScene::empty();
        let source_id = ObjectId::new();
        let source = SceneObject::new_empty(source_id, "Source");
        source.transform_mut().translation = Vec3::X * 10.0;
        scene.objects.push(source);
        let mut value = Constraint::default();
        value.kind = ConstraintKind::Position;
        value.weight = 0.5;
        value.sources.push(ConstraintSource { target: source_id.into(), weight: 1.0, ..Default::default() });
        let mut bob = Bob::default();
        bob.axis = Vec3::X.to_array();
        bob.amplitude = 4.0;
        let mut actor = SceneObject::new_empty(ObjectId::new(), "Constrained");
        actor.transform_mut().translation = Vec3::X * 2.0;
        actor.components.replace_all(vec![Box::new(bob), Box::new(value)]);
        scene.objects.push(actor);
        with_env(0.5, |env| {
            scene.run_ecs_systems(&registry, Stage::Start, env).unwrap();
            scene.run_ecs_systems(&registry, Stage::Update, env).unwrap();
            scene.run_ecs_systems(&registry, Stage::LateUpdate, env).unwrap();
            assert!((scene.objects[1].translation().x - 6.0).abs() < 1e-5, "rest is captured before Bob's first offset");
            scene.objects[1].components.set_enabled(1, false);
            scene.objects[1].transform_mut().translation = Vec3::X * 20.0;
            scene.objects[0].transform_mut().translation = Vec3::X * 14.0;
            scene.objects[1].components.set_enabled(1, true);
            assert!(scene.run_ecs_systems(&registry, Stage::Start, env).unwrap().is_empty());
            scene.run_ecs_systems(&registry, Stage::LateUpdate, env).unwrap();
            assert!((scene.objects[1].translation().x - 8.0).abs() < 1e-5, "reenabling retains the original rest pose");
        });
    }

    #[test]
    fn native_third_person_and_follow_cameras_keep_world_poses_under_parenting_and_reordering() {
        use zestengine_core::{CameraComponent, FollowCamera, ThirdPersonCharacter};
        let registry = ComponentRegistry::with_builtins();
        let mut scene = LoadedScene::empty();
        let group = SceneObject::new_empty(ObjectId::new(), "Camera group");
        *group.transform_mut() = Transform { translation: Vec3::new(-7.0, 3.0, 4.0), rotation: glam::Quat::from_rotation_y(0.7), scale: Vec3::splat(1.5) };
        scene.objects.push(group);
        let actor_id = ObjectId::new();
        let third_id = ObjectId::new();
        let follow_id = ObjectId::new();
        let mut actor = SceneObject::new_empty(actor_id, "Player");
        actor.transform_mut().translation = Vec3::new(8.0, 1.0, -6.0);
        actor.components.push(Box::new(ThirdPersonCharacter { camera: third_id.into(), lock_cursor: false, ..Default::default() }));
        scene.objects.push(actor);
        let mut third = SceneObject::new_empty(third_id, "Third person camera").with_parent(Some(0));
        third.components.push(Box::new(CameraComponent::default()));
        scene.objects.push(third);
        let mut follow = SceneObject::new_empty(follow_id, "Follow camera").with_parent(Some(0));
        follow.components.replace_all(vec![Box::new(CameraComponent::default()), Box::new(FollowCamera { target: actor_id.into(), offset: [2.0, 5.0, 8.0], smoothing: 0.0, ..Default::default() })]);
        scene.objects.push(follow);
        let mut session = crate::play::PlaySession::new(&scene, std::path::Path::new("."), None);
        for frame in 0..2 {
            if frame == 1 { scene.objects.swap(0, 2); }
            character_tick(&mut scene, &mut session, &registry);
            let actor = scene.index_of(actor_id).unwrap();
            let position = scene.world_transform(actor).w_axis.truncate();
            let data = scene.objects[actor].components.get_typed::<ThirdPersonCharacter>().unwrap();
            let rotation = glam::Quat::from_euler(glam::EulerRot::YXZ, data.cam_yaw.to_radians(), data.cam_pitch.to_radians(), 0.0);
            let desired = position + Vec3::Y * data.pivot_height + rotation * Vec3::new(data.cur_shoulder, 0.0, data.cur_distance);
            let actual = scene.world_transform(scene.index_of(third_id).unwrap());
            assert!(actual.w_axis.truncate().abs_diff_eq(desired, 1e-4));
            assert!(actual.to_scale_rotation_translation().1.dot(rotation).abs() > 1.0 - 1e-5);
            let follow = scene.world_transform(scene.index_of(follow_id).unwrap());
            let desired = position + Vec3::new(2.0, 5.0, 8.0);
            assert!(follow.w_axis.truncate().abs_diff_eq(desired, 1e-4));
            let direction = (position + Vec3::Y * 0.5 - desired).normalize();
            assert!((follow.to_scale_rotation_translation().1 * Vec3::NEG_Z).abs_diff_eq(direction, 1e-4));
        }
    }

    #[test]
    fn native_vr_tracking_grabs_and_avatar_follow_keep_world_poses_after_reordering() {
        use zestengine_core::{CameraComponent, Grabbable, VrAvatar, VrCharacter, VrFrame, VrLocomotion, VrTurning};
        use glam::{Mat4, Quat};
        let registry = ComponentRegistry::with_builtins();
        let mut scene = LoadedScene::empty();
        let group = SceneObject::new_empty(ObjectId::new(), "Parent");
        *group.transform_mut() = Transform { translation: Vec3::new(7.0, 2.0, -3.0), rotation: Quat::from_rotation_y(0.7), scale: Vec3::splat(1.4) };
        scene.objects.push(group);
        let rig_id = ObjectId::new();
        let head_id = ObjectId::new();
        let hand_id = ObjectId::new();
        let item_id = ObjectId::new();
        let avatar_id = ObjectId::new();
        let mut rig = SceneObject::new_empty(rig_id, "Rig").with_parent(Some(0));
        rig.transform_mut().translation = Vec3::new(2.0, 0.0, 1.0);
        rig.components.push(Box::new(VrCharacter { head: head_id.into(), right_hand: hand_id.into(), locomotion: VrLocomotion::Disabled, turning: VrTurning::Disabled, ..Default::default() }));
        scene.objects.push(rig);
        let mut head = SceneObject::new_empty(head_id, "Tracked head").with_parent(Some(1));
        head.components.push(Box::new(CameraComponent::default()));
        scene.objects.push(head);
        scene.objects.push(SceneObject::new_empty(hand_id, "Tracked hand").with_parent(Some(1)));
        let hand_pose = (Vec3::new(0.25, 1.3, -0.4), Quat::from_rotation_y(0.3));
        let initial_item = scene.world_transform(1).transform_point3(hand_pose.0);
        let mut item = SceneObject::new_empty(item_id, "Parented item").with_parent(Some(0));
        item.transform_mut().translation = scene.world_transform(0).inverse().transform_point3(initial_item);
        item.components.push(Box::new(Grabbable { snap_to_hand: true, hold_position: [0.0, 0.1, -0.2], return_after: 10.0, ..Default::default() }));
        scene.objects.push(item);
        let mut avatar = SceneObject::new_empty(avatar_id, "Avatar").with_parent(Some(0));
        avatar.components.push(Box::new(VrAvatar { head: head_id.into(), ..Default::default() }));
        scene.objects.push(avatar);
        let home = scene.objects[4].translation();
        let mut session = crate::play::PlaySession::new(&scene, std::path::Path::new("."), None);
        let mut shaders = crate::shaders::ShaderLibrary::default();
        let input = Default::default();
        let net = Default::default();
        for frame in 0..3 {
            if frame == 1 { scene.objects.swap(0, 4); }
            let screen = zestengine_core::ScreenInfo {
                vr: VrFrame { active: true, head: Some((Vec3::new(0.1, 1.7 + frame as f32 * 0.1, 0.0), Quat::from_rotation_y(0.5))),
                    right_hand: Some((hand_pose.0 + Vec3::X * frame as f32 * 0.1, hand_pose.1)), right_grip: if frame < 2 { 1.0 } else { 0.0 }, right_trigger: frame == 0, ..Default::default() },
                ..Default::default()
            };
            session.tick(&mut crate::play::World { scene: &mut scene, renderer: None, registry: &registry, shaders: &mut shaders },
                &crate::play::TickInput { dt: 1.0 / 60.0, input: &input, net: &net, listener_fallback: Mat4::IDENTITY, screen: &screen });
            let head = scene.world_transform(scene.index_of(head_id).unwrap());
            let body = scene.world_transform(scene.index_of(avatar_id).unwrap());
            assert!(body.w_axis.truncate().abs_diff_eq(head.w_axis.truncate() - Vec3::Y * 0.22, 1e-4), "avatar sees this frame's tracked head");
            let item = &scene.objects[scene.index_of(item_id).unwrap()];
            let grab = item.components.get_typed::<Grabbable>().unwrap();
            assert_eq!(grab.held, frame < 2);
            assert_eq!(grab.trigger_down, frame == 0);
            assert_eq!(grab.home.unwrap().0, home);
            if frame < 2 {
                let hand = scene.world_transform(scene.index_of(hand_id).unwrap());
                let expected = hand * Mat4::from_translation(Vec3::new(0.0, 0.1, -0.2));
                let actual = scene.world_transform(scene.index_of(item_id).unwrap());
                assert!(actual.w_axis.truncate().abs_diff_eq(expected.w_axis.truncate(), 1e-4));
                assert!(actual.to_scale_rotation_translation().1.dot(expected.to_scale_rotation_translation().1).abs() > 1.0 - 1e-5);
            }
        }
    }

    #[test]
    fn native_strategy_startup_handles_late_added_and_disabled_instances() {
        use zestengine_core::{ClickToMove, RtsSelection};
        let registry = ComponentRegistry::with_builtins();
        let mut scene = LoadedScene::empty();
        scene.objects.push(SceneObject::new_empty(ObjectId::new(), "Parent"));
        scene.objects.push(SceneObject::new_empty(ObjectId::new(), "Controller").with_parent(Some(0)));
        let rts_marker = ObjectId::new();
        let click_marker = ObjectId::new();
        scene.objects.push(SceneObject::new_empty(rts_marker, "RTS Marker"));
        scene.objects.push(SceneObject::new_empty(click_marker, "Click Marker"));
        let mut session = crate::play::PlaySession::new(&scene, std::path::Path::new("."), None);
        character_tick(&mut scene, &mut session, &registry);
        scene.objects[0].set_enabled(false);
        scene.objects[1].components.push(Box::new(RtsSelection { move_marker: rts_marker.into(), ..Default::default() }));
        scene.objects[1].components.push(Box::new(ClickToMove { marker: click_marker.into(), ..Default::default() }));
        character_tick(&mut scene, &mut session, &registry);
        assert!(!scene.objects[1].components.get_typed::<RtsSelection>().unwrap().started);
        assert!(!scene.objects[1].components.get_typed::<ClickToMove>().unwrap().started);
        scene.objects[0].set_enabled(true);
        character_tick(&mut scene, &mut session, &registry);
        assert!(scene.objects[1].components.get_typed::<RtsSelection>().unwrap().started);
        assert!(scene.objects[1].components.get_typed::<ClickToMove>().unwrap().started);
        assert!(!scene.objects[2].enabled() && !scene.objects[3].enabled());
        scene.objects[2].set_enabled(true);
        scene.objects[1].components.get_typed_mut::<RtsSelection>().unwrap().marker_timer = 0.8;
        scene.objects[1].components.set_enabled(0, false);
        character_tick(&mut scene, &mut session, &registry);
        scene.objects[1].components.set_enabled(0, true);
        character_tick(&mut scene, &mut session, &registry);
        assert!(scene.objects[2].enabled(), "reenabling keeps the initialized marker timer");
        assert!(scene.objects[1].components.get_typed::<RtsSelection>().unwrap().marker_timer > 0.7);
    }

    #[test]
    fn native_trigger_bindings_receive_real_3d_and_2d_contacts() {
        use zestengine_core::{ArgSource, ArgValue, BoxCollider, BoxCollider2D, CircleCollider2D, EventBinding, LightComponent, RigidBody, Rigidbody2D, SphereCollider, TriggerEvents};
        let registry = ComponentRegistry::with_builtins();
        for two_dimensional in [false, true] {
            let mut scene = LoadedScene::empty();
            let light_id = ObjectId::new();
            let set = |value| EventBinding { target: light_id.into(), component: "Light".into(), action: "SetIntensity".into(), args: vec![ArgSource::Const(ArgValue::Float(value))], ..Default::default() };
            let mut zone = SceneObject::new_empty(ObjectId::new(), "Zone");
            zone.components.push(Box::new(TriggerEvents { on_enter: vec![set(5.0)], on_exit: vec![set(0.5)], ..Default::default() }));
            if two_dimensional { zone.components.push(Box::new(BoxCollider2D { size: [6.0, 2.0], is_trigger: true, ..Default::default() })); }
            else { zone.components.push(Box::new(BoxCollider { size: [6.0, 2.0, 6.0], is_trigger: true, ..Default::default() })); }
            scene.objects.push(zone);
            let mut light = SceneObject::new_empty(light_id, "Light");
            light.components.push(Box::new(LightComponent::default()));
            scene.objects.push(light);
            let mut visitor = SceneObject::new_empty(ObjectId::new(), "Visitor");
            visitor.transform_mut().translation = Vec3::Y * 3.0;
            if two_dimensional { visitor.components.replace_all(vec![Box::new(Rigidbody2D::default()), Box::new(CircleCollider2D::default())]); }
            else { visitor.components.replace_all(vec![Box::new(RigidBody::default()), Box::new(SphereCollider::default())]); }
            scene.objects.push(visitor);
            let mut session = crate::play::PlaySession::new(&scene, std::path::Path::new("."), None);
            for _ in 0..120 { character_tick(&mut scene, &mut session, &registry); }
            let trigger = scene.objects[0].components.get_typed::<TriggerEvents>().unwrap();
            assert!(trigger.entered, "native enter dispatched, 2D={two_dimensional}");
            assert_eq!(trigger.inside, 0, "native exit dispatched, 2D={two_dimensional}");
            assert_eq!(scene.objects[1].components.get_typed::<LightComponent>().unwrap().intensity, 0.5);
        }
    }

    #[test]
    fn native_combat_events_reach_huds_next_frame_and_do_not_repeat_after_respawn() {
        use zestengine_core::{Health, HudBinding, HudValue, ScoreEvent, TargetDummy, UiText};
        let registry = ComponentRegistry::with_builtins();
        let mut scene = LoadedScene::empty();
        let mut target = SceneObject::new_empty(ObjectId::new(), "Target");
        target.components.replace_all(vec![Box::new(Health { max: 10.0, current: 10.0, ..Default::default() }),
            Box::new(TargetDummy { points: 7, respawn_delay: 0.04, ..Default::default() })]);
        scene.objects.push(target);
        let mut score = SceneObject::new_empty(ObjectId::new(), "Score");
        score.components.replace_all(vec![Box::new(UiText::default()), Box::new(HudBinding { value: HudValue::Score, ..Default::default() })]);
        scene.objects.push(score);
        let mut session = crate::play::PlaySession::new(&scene, std::path::Path::new("."), None);
        character_tick(&mut scene, &mut session, &registry);
        scene.objects[0].components.get_typed_mut::<Health>().unwrap().apply_damage(10.0);
        character_tick(&mut scene, &mut session, &registry);
        assert!(scene.objects[0].components.get_typed::<TargetDummy>().unwrap().knocked());
        assert_eq!(scene.objects[1].components.get_typed::<UiText>().unwrap().text, "0");
        character_tick(&mut scene, &mut session, &registry);
        assert_eq!(scene.objects[1].components.get_typed::<UiText>().unwrap().text, "7");
        for _ in 0..5 { character_tick(&mut scene, &mut session, &registry); }
        assert!(!scene.objects[0].components.get_typed::<TargetDummy>().unwrap().knocked());
        assert_eq!(scene.objects[0].components.get_typed::<Health>().unwrap().current, 10.0);
        assert_eq!(scene.objects[1].components.get_typed::<HudBinding>().unwrap().score, 7);
        session.post_event(ScoreEvent { points: 3 });
        character_tick(&mut scene, &mut session, &registry);
        assert_eq!(scene.objects[1].components.get_typed::<UiText>().unwrap().text, "10");
    }

    #[test]
    fn native_weapons_fire_after_queued_character_movement_has_been_applied() {
        use zestengine_core::{ActionValue, BoxCollider, Character, Health, HitscanWeapon, InputState};
        fn move_character(world: &SystemWorld<'_>, context: &mut SystemContext<'_>) {
            for (_, (id, _)) in world.query::<(&ObjectId, &Character)>().iter() {
                context.commands.push(ComponentCommand::MoveCharacter { id: *id, delta: Vec3::X });
            }
        }
        let mut registry = ComponentRegistry::with_builtins();
        let mut scene = LoadedScene::empty();
        let camera_id = ObjectId::new();
        let mut shooter = SceneObject::new_empty(ObjectId::new(), "Shooter");
        shooter.components.replace_all(vec![Box::new(Character::default()), Box::new(HitscanWeapon { aim_from: camera_id.into(), automatic: false, spread_deg: 0.0, ..Default::default() })]);
        scene.objects.push(shooter);
        let camera = SceneObject::new_empty(camera_id, "Aim").with_parent(Some(0));
        camera.transform_mut().translation = Vec3::Y * 1.65;
        scene.objects.push(camera);
        let mut target = SceneObject::new_empty(ObjectId::new(), "Target");
        target.transform_mut().translation = Vec3::new(1.0, 1.65, -4.0);
        target.components.replace_all(vec![Box::new(BoxCollider { size: [0.5, 2.0, 0.5], ..Default::default() }), Box::new(Health::default())]);
        scene.objects.push(target);
        let mut session = crate::play::PlaySession::new(&scene, std::path::Path::new("."), None);
        for _ in 0..2 { character_tick(&mut scene, &mut session, &registry); }
        registry.register_system("test::move_before_shooting", Stage::Update, move_character);
        let input = InputState::from_actions(vec![("Fire".into(), ActionValue { down: true, just_pressed: true, ..Default::default() })], glam::Vec2::ZERO);
        let mut shaders = crate::shaders::ShaderLibrary::default();
        session.tick(&mut crate::play::World { scene: &mut scene, renderer: None, registry: &registry, shaders: &mut shaders },
            &crate::play::TickInput { dt: 1.0 / 60.0, input: &input, net: &Default::default(), listener_fallback: glam::Mat4::IDENTITY, screen: &Default::default() });
        assert!((scene.objects[0].translation().x - 1.0).abs() < 0.01);
        assert_eq!(scene.objects[2].components.get_typed::<Health>().unwrap().current, 75.0, "the shot uses the moved position, not last phase's origin");
    }

    #[test]
    fn typed_disabled_payloads_suppress_native_updates_and_contacts_without_resetting_state() {
        use zestengine_core::ecs::{ContactPhase, SceneEvent};
        #[derive(Default, serde::Serialize, serde::Deserialize)]
        struct Probe { #[serde(skip)] calls: u32 }
        impl zestengine_core::TypedComponent for Probe { const NAME: &'static str = "Disabled Native Probe"; }
        let mut registry = ComponentRegistry::with_builtins();
        registry.register_ecs::<Probe>();
        registry.register_system_for::<Probe>("test::disabled_update", Stage::Update, |world, ctx| {
            for (_, (owner, probe)) in world.query::<(&Owner, &mut Probe)>().iter() {
                if ctx.is_active(world, owner.0) { probe.calls += 1; }
            }
        });
        registry.register_system_for::<Probe>("test::disabled_contact", Stage::Events, |world, ctx| {
            for (_, (owner, probe)) in world.query::<(&Owner, &mut Probe)>().iter() {
                if !ctx.is_active(world, owner.0) { continue; }
                for event in ctx.services.scene_events {
                    if let SceneEvent::Contact { object, trigger: false, phase: ContactPhase::Enter, .. } = event
                        && Some(*object) == world.object_id(owner.0) { probe.calls += 10; }
                }
            }
        });
        let mut scene = LoadedScene::empty();
        let mut object = SceneObject::new_empty(ObjectId::new(), "Object");
        let id = object.id();
        object.components.push(Box::new(Disabled(Box::new(Probe { calls: 7 }))));
        scene.objects.push(object);
        let contacts = [SceneEvent::Contact { object: id, other: ObjectId::new(), trigger: false, phase: ContactPhase::Enter, two_dimensional: false }];
        with_env(0.0, |env| {
            scene.run_ecs_systems(&registry, Stage::Update, env).unwrap();
            scene.run_ecs_event_systems(&registry, Stage::Events, env, &contacts).unwrap();
        });
        assert_eq!(scene.objects[0].components.get_including_disabled::<Probe>().unwrap().calls, 7);
        scene.objects[0].components.set_enabled(0, true);
        with_env(0.0, |env| { scene.run_ecs_systems(&registry, Stage::Update, env).unwrap(); });
        assert_eq!(scene.objects[0].components.get_typed::<Probe>().unwrap().calls, 8);
        scene.objects[0].components.set_enabled(0, false);
        with_env(0.0, |env| { scene.run_ecs_event_systems(&registry, Stage::Events, env, &contacts).unwrap(); });
        assert_eq!(scene.objects[0].components.get_including_disabled::<Probe>().unwrap().calls, 8);
        assert_eq!(scene.objects.world.with_world(|world| world.query::<&Disabled>().iter().count()), 0);
    }

    #[test]
    fn systems_can_join_and_edit_registered_builtin_data() {
        use zestengine_core::{LightComponent, RigidBody};
        fn system(world: &SystemWorld<'_>, _: &mut SystemContext<'_>) {
            for (_, (light, _)) in world.query::<(&mut LightComponent, &RigidBody)>().iter() {
                light.intensity = 7.0;
            }
        }
        let mut registry = ComponentRegistry::with_builtins();
        registry.register_system("test::builtins", Stage::Update, system);
        let mut scene = LoadedScene::empty();
        let mut object = SceneObject::new_empty(ObjectId::new(), "Object");
        object.components.replace_all(vec![Box::new(LightComponent::default()), Box::new(RigidBody::default())]);
        scene.objects.push(object);
        with_env(0.25, |env| { scene.run_ecs_systems(&registry, Stage::Update, env).unwrap(); });
        assert_eq!(scene.objects[0].components.get_typed_at::<LightComponent>(0).unwrap().intensity, 7.0);
    }

    #[test]
    fn parent_activation_and_saved_ids_survive_reorder_removal_and_scene_reuse() {
        let registry = ComponentRegistry::with_builtins();
        let mut scene = LoadedScene::empty();
        let parent = SceneObject::new_empty(ObjectId::new(), "Parent");
        *parent.enabled_mut() = false;
        let mut child = SceneObject::new_empty(ObjectId::new(), "Child");
        child.set_parent(Some(0));
        child.components.push(Box::new(Bob::default()));
        let child_id = child.id();
        let parent_id = parent.id();
        scene.objects.extend([parent, child]);
        with_env(0.25, |env| { scene.run_ecs_systems(&registry, Stage::Update, env).unwrap(); });
        assert_eq!(scene.objects[1].translation(), Vec3::ZERO);
        let entity = scene.objects.world.entity(child_id).unwrap();
        assert!(!scene.objects.world.get::<Active>(entity).unwrap().0);
        scene.objects.swap(0, 1);
        scene.objects[0].set_parent(Some(1));
        *scene.objects[1].enabled_mut() = true;
        with_env(0.5, |env| { scene.run_ecs_systems(&registry, Stage::Update, env).unwrap(); });
        assert_eq!(scene.objects.world.entity(child_id), Some(entity));
        assert!((scene.objects[0].translation().y - 0.5).abs() < 1e-5);
        scene.objects.pop();
        scene.objects[0].set_parent(None);
        with_env(0.75, |env| { scene.run_ecs_systems(&registry, Stage::Update, env).unwrap(); });
        assert_eq!(scene.objects.world.entity(parent_id), None);
        assert_eq!(scene.objects.world.entity(child_id), Some(entity));
    }

    #[test]
    fn a_panicking_plugin_does_not_drop_components_or_the_entity_map() {
        fn panic_after_edit(world: &SystemWorld<'_>, _: &mut SystemContext<'_>) {
            for (_, (owner, _)) in world.query::<(&Owner, &Bob)>().iter() {
                world.get_mut::<Transform>(owner.0).unwrap().translation.x = 42.0;
            }
            panic!("plugin failed");
        }
        let mut registry = ComponentRegistry::with_builtins();
        registry.register_system("test::panic", Stage::Update, panic_after_edit);
        let mut scene = LoadedScene::empty();
        let mut object = SceneObject::new_empty(ObjectId::new(), "Object");
        let id = object.id();
        object.components.push(Box::new(Bob::default()));
        scene.objects.push(object);
        let panic = catch_unwind(AssertUnwindSafe(|| with_env(0.25, |env| {
            scene.run_ecs_systems(&registry, Stage::Update, env).unwrap();
        })));
        assert!(panic.is_err());
        assert_eq!(scene.objects[0].translation().x, 42.0);
        assert!(scene.objects[0].components.get(0).unwrap().as_any().is::<Bob>());
        assert!(scene.objects.world.entity(id).is_some());
        assert_eq!(scene.objects.world.with_world(|world| world.query::<&Bob>().iter().count()), 1);
    }
}
