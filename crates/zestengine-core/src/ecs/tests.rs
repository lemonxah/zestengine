use super::*;
use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Default, Serialize, Deserialize)]
struct Velocity(f32);
impl TypedComponent for Velocity { const NAME: &'static str = "test::velocity"; }

#[derive(Default, Serialize, Deserialize)]
struct Health(u32);
impl TypedComponent for Health { const NAME: &'static str = "test::health"; }

#[derive(Default, Serialize, Deserialize)]
struct Stateful {
    value: u32,
    #[serde(skip)]
    live: u32,
}
impl TypedComponent for Stateful { const NAME: &'static str = "test::stateful"; }

#[test]
fn native_event_consumers_borrow_typed_input_and_emit_without_reading_their_own_output() {
    struct Ping(u32);
    struct Pong(u32);
    fn consume(world: &SystemWorld<'_>, context: &mut SystemContext<'_>) {
        for (_, total) in world.query::<&mut Health>().iter() {
            for event in context.events::<Ping>() {
                total.0 += event.0;
                context.emit(Pong(event.0));
            }
        }
        assert_eq!(context.events::<Pong>().count(), 0, "emitted events are next-frame output");
    }
    let mut registry = ComponentRegistry::default();
    registry.register_ecs::<Health>();
    registry.register_system("test::native_events", Stage::Gameplay, consume);
    let mut scene = SceneWorld::new();
    let owner = scene.spawn("Counter");
    scene.insert(owner, Health(0)).unwrap();
    let events = [crate::AnyBox(Box::new(Ping(2))), crate::AnyBox(Box::new("other event")), crate::AnyBox(Box::new(Ping(3)))];
    let input = InputState::default();
    let mut commands = Vec::new();
    let mut context = SystemContext::new(&input, &mut commands);
    context.services.events = &events;
    registry.run_systems(Stage::Gameplay, &scene, &mut context);
    assert_eq!(scene.get::<Health>(owner).unwrap().0, 5);
    let emitted: Vec<_> = commands.iter().filter_map(|command| match command {
        ComponentCommand::Emit(event) => event.0.downcast_ref::<Pong>().map(|event| event.0),
        _ => None,
    }).collect();
    assert_eq!(emitted, [2, 3]);
}

#[test]
fn queries_join_typed_data_and_duplicate_instances_have_an_owner() {
    let mut world = SceneWorld::new();
    let player = world.spawn("Player");
    let first = world.insert(player, Velocity(1.0)).unwrap();
    let second = world.insert(player, Velocity(2.0)).unwrap();
    world.insert(player, Health(100)).unwrap();
    assert_eq!(first, player);
    assert_ne!(second, player);
    assert_eq!(world.get::<Owner>(second).unwrap().0, player);
    world.with_world(|world| {
        for (_, (velocity, health)) in world.query::<(&mut Velocity, &Health)>().iter() {
            velocity.0 *= health.0 as f32;
        }
    });
    assert_eq!(world.get::<Velocity>(first).unwrap().0, 100.0);
    assert_eq!(world.get::<Velocity>(second).unwrap().0, 2.0);
    assert_eq!(world.with_world(|world| world.query::<(&Owner, &Velocity)>().iter().count()), 2);
    world.remove::<Velocity>(second).unwrap();
    assert_eq!(world.with_world(|world| world.query::<&Velocity>().iter().count()), 1);
    assert_eq!(world.len(), 1);
}

#[test]
fn parenting_propagates_rotation_scale_and_inherited_activation() {
    let mut world = SceneWorld::new();
    let parent = world.spawn("Parent");
    let child = world.spawn("Child");
    *world.get_mut::<Transform>(parent).unwrap() = Transform {
        translation: Vec3::new(2.0, 3.0, 4.0), rotation: Quat::from_rotation_y(0.5), scale: Vec3::splat(2.0),
    };
    world.get_mut::<Transform>(child).unwrap().translation = Vec3::X;
    world.set_parent(child, Some(parent), false).unwrap();
    let expected = world.get::<Transform>(parent).unwrap().matrix() * world.get::<Transform>(child).unwrap().matrix();
    assert!(world.get::<GlobalTransform>(child).unwrap().0.abs_diff_eq(expected, 1e-5));
    world.get_mut::<Enabled>(parent).unwrap().0 = false;
    world.propagate().unwrap();
    assert!(!world.get::<Active>(child).unwrap().0);
    assert!(world.get::<Enabled>(child).unwrap().0);
    world.get_mut::<Enabled>(parent).unwrap().0 = true;
    world.propagate().unwrap();
    assert!(world.get::<Active>(child).unwrap().0);
}

#[test]
fn reparenting_can_keep_world_and_rejects_cycles_and_singular_parents() {
    let mut world = SceneWorld::new();
    let a = world.spawn("A");
    let b = world.spawn("B");
    world.get_mut::<Transform>(a).unwrap().translation = Vec3::X * 4.0;
    world.get_mut::<Transform>(b).unwrap().translation = Vec3::Y * 3.0;
    world.propagate().unwrap();
    let before = world.get::<GlobalTransform>(b).unwrap().0;
    world.set_parent(b, Some(a), true).unwrap();
    assert!(world.get::<GlobalTransform>(b).unwrap().0.abs_diff_eq(before, 1e-5));
    assert_eq!(world.set_parent(a, Some(b), false), Err(HierarchyError::Cycle));
    assert_eq!(world.get::<Parent>(a).unwrap().0, None);
    world.set_parent(b, None, true).unwrap();
    world.get_mut::<Transform>(a).unwrap().scale = Vec3::ZERO;
    assert_eq!(world.set_parent(b, Some(a), true), Err(HierarchyError::SingularParent));
    assert_eq!(world.get::<Parent>(b).unwrap().0, None);
}

#[test]
fn deep_hierarchies_resolve_iteratively_and_bulk_parent_validation_is_atomic() {
    let mut world = SceneWorld::new();
    let entities: Vec<_> = (0..4096).map(|_| {
        let entity = world.spawn("Node");
        world.get_mut::<Transform>(entity).unwrap().translation = Vec3::X;
        entity
    }).collect();
    let parents: Vec<_> = entities.windows(2).map(|pair| (pair[1], Some(pair[0]))).collect();
    world.set_parents(&parents).unwrap();
    assert_eq!(world.get::<GlobalTransform>(*entities.last().unwrap()).unwrap().0.w_axis.x, 4096.0);
    let cycle = [(entities[0], Some(entities[4095])), (entities[1], None), (entities[1], Some(entities[0]))];
    assert_eq!(world.set_parents(&cycle), Err(HierarchyError::Cycle));
    assert_eq!(world.get::<Parent>(entities[0]).unwrap().0, None);
    assert_eq!(world.get::<Parent>(entities[1]).unwrap().0, Some(entities[0]));
    world.propagate().unwrap();
}

#[test]
fn deletion_removes_children_and_components_and_stale_saved_references_stay_gone() {
    let mut world = SceneWorld::new();
    let parent = world.spawn("Parent");
    let child = world.spawn("Child");
    let id = world.object_id(child).unwrap();
    world.set_parent(child, Some(parent), false).unwrap();
    world.insert(child, Health(1)).unwrap();
    world.insert(child, Health(2)).unwrap();
    world.despawn_recursive(parent).unwrap();
    assert!(world.is_empty());
    assert_eq!(world.with_world(|world| world.query::<&Owner>().iter().count()), 0);
    let replacement = world.spawn("Replacement");
    assert_ne!(replacement, child);
    assert_eq!(world.entity(id), None);
    assert!(world.get::<Health>(child).is_none());
    assert!(world.spawn_with_id(ObjectId::NIL, "Bad", Transform::default(), true).is_err());
    assert!(world.spawn_with_id(world.object_id(replacement).unwrap(), "Duplicate", Transform::default(), true).is_err());
}

#[test]
fn permanent_components_keep_runtime_state_and_disabled_instances_skip_native_dispatch() {
    let mut registry = ComponentRegistry::default();
    registry.register_ecs::<Stateful>();
    let mut world = SceneWorld::new();
    let owner = world.spawn("Object");
    let mut components = ComponentSet::from_vec(vec![Box::new(Stateful { value: 7, live: 42 })]);
    components.attach(&world, owner);
    assert_eq!(world.get::<Stateful>(owner).unwrap().live, 42);
    world.get_mut::<Stateful>(owner).unwrap().live += 1;
    assert_eq!(components.get_typed::<Stateful>().unwrap().live, 43);
    registry.register_system_for::<Stateful>("test::stateful", Stage::Update, |world, _| {
        for (_, value) in world.query::<&mut Stateful>().iter() { value.live += 1; }
    });
    assert!(registry.has_runnable_systems(Stage::Update, &world));
    components.set_enabled(0, false);
    assert!(!registry.has_runnable_systems(Stage::Update, &world));
    registry.run_systems(Stage::Update, &world, &mut SystemContext::new(&InputState::default(), &mut Vec::new()));
    assert_eq!(components.get_including_disabled::<Stateful>().unwrap().live, 43);
    assert!(world.entity(world.object_id(owner).unwrap()).is_some());
}

#[test]
fn system_reregistration_replaces_code_and_stage_without_double_execution() {
    fn old(_: &SystemWorld<'_>, _: &mut SystemContext<'_>) { panic!("old system must be replaced"); }
    fn new(world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) {
        for (_, velocity) in world.query::<&mut Velocity>().iter() { velocity.0 += ctx.dt; }
    }
    let mut registry = ComponentRegistry::default();
    registry.register_system("plugin::system", Stage::FixedUpdate, old);
    registry.register_system("plugin::system", Stage::Update, new);
    let mut world = SceneWorld::new();
    let owner = world.spawn("Object");
    world.insert(owner, Velocity::default()).unwrap();
    let input = InputState::default();
    let mut commands = Vec::new();
    let mut ctx = SystemContext { dt: 0.25, unscaled_dt: 0.25, time: 1.0, water_time: 1.0, input: &input, commands: &mut commands, services: Default::default() };
    assert!(!registry.has_systems(Stage::FixedUpdate));
    registry.run_systems(Stage::Update, &mut world, &mut ctx);
    assert_eq!(world.get::<Velocity>(owner).unwrap().0, 0.25);
}

#[test]
fn plugin_reload_removes_deleted_systems_and_preserves_other_registrations() {
    fn removed(_: &SystemWorld<'_>, _: &mut SystemContext<'_>) { panic!("removed system ran"); }
    fn add(world: &SystemWorld<'_>, _: &mut SystemContext<'_>) {
        for (_, data) in world.query::<&mut Health>().iter() { data.0 += 1; }
    }
    fn initial(registry: &mut ComponentRegistry) {
        registry.register_system("project::retained", Stage::Update, add);
        registry.register_system("project::removed", Stage::Update, removed);
    }
    fn replacement(registry: &mut ComponentRegistry) {
        registry.register_system("project::retained", Stage::Update, add);
    }
    let mut registry = ComponentRegistry::default();
    registry.register_system("engine::system", Stage::Update, add);
    registry.register_plugin("project", initial);
    registry.register_plugin("project", replacement);
    let mut world = SceneWorld::new();
    let entity = world.spawn("Object");
    world.insert(entity, Health::default()).unwrap();
    let input = InputState::default();
    let mut commands = Vec::new();
    registry.run_systems(Stage::Update, &mut world, &mut SystemContext {
        dt: 0.1, unscaled_dt: 0.1, time: 0.1, water_time: 0.1, input: &input, commands: &mut commands, services: Default::default(),
    });
    assert_eq!(world.get::<Health>(entity).unwrap().0, 2);
    registry.register_plugin("project", |_| {});
    registry.run_systems(Stage::Update, &mut world, &mut SystemContext {
        dt: 0.1, unscaled_dt: 0.1, time: 0.2, water_time: 0.2, input: &input, commands: &mut commands, services: Default::default(),
    });
    assert_eq!(world.get::<Health>(entity).unwrap().0, 3);
}

#[test]
fn entities_build_fresh_components_and_follow_plugin_reloads() {
    fn guard() -> EntityTemplate { EntityTemplate::new(EntityShape::Capsule).with(Health(5)).with(Velocity(2.0)) }
    fn scout() -> EntityTemplate { EntityTemplate::new(EntityShape::Empty).with(Velocity(9.0)) }
    fn initial(registry: &mut ComponentRegistry) {
        registry.register_entity("Guard", guard);
        registry.register_entity("Scout", scout);
    }
    fn replacement(registry: &mut ComponentRegistry) { registry.register_entity("Guard", scout); }
    let mut registry = ComponentRegistry::default();
    registry.register_entity("Engine Entity", guard);
    registry.register_plugin("project", initial);
    assert_eq!(registry.entity_names().collect::<Vec<_>>(), ["Engine Entity", "Guard", "Scout"]);
    let built = registry.create_entity("Guard").unwrap();
    assert_eq!(built.shape, EntityShape::Capsule);
    assert_eq!(built.components.iter().map(|component| component.type_name()).collect::<Vec<_>>(), ["test::health", "test::velocity"]);
    registry.register_plugin("project", replacement);
    assert_eq!(registry.entity_names().collect::<Vec<_>>(), ["Engine Entity", "Guard"], "a reload drops entities the plugin no longer exports");
    assert_eq!(registry.create_entity("Guard").unwrap().components.len(), 1, "re-registering replaces the builder in place");
    assert!(registry.create_entity("Scout").is_none());
    let mut world = SceneWorld::new();
    let owner = world.spawn("Guard");
    let mut components = ComponentSet::from_vec(registry.create_entity("Engine Entity").unwrap().components);
    components.attach(&world, owner);
    assert_eq!(world.get::<Health>(owner).unwrap().0, 5);
    assert_eq!(world.get::<Velocity>(owner).unwrap().0, 2.0);
}

#[test]
fn conditional_systems_follow_typed_membership_and_reregistration() {
    fn run(world: &SystemWorld<'_>, _: &mut SystemContext<'_>) {
        let mut count = 0;
        for (_, velocity) in world.query::<&mut Velocity>().iter() { velocity.0 += 1.0; count += 1; }
        assert!(count > 0, "conditional system cannot run without its data");
    }
    let mut registry = ComponentRegistry::default();
    registry.register_system_for::<Velocity>("test::conditional", Stage::Update, run);
    let mut world = SceneWorld::new();
    let owner = world.spawn("Owner");
    let input = InputState::default();
    let mut commands = Vec::new();
    assert!(!registry.has_runnable_systems(Stage::Update, &world));
    registry.run_systems(Stage::Update, &world, &mut SystemContext::new(&input, &mut commands));
    let mut components = ComponentSet::from_vec(vec![Box::new(crate::Disabled(Box::new(Velocity::default())))]);
    components.attach(&world, owner);
    assert!(!registry.has_runnable_systems(Stage::Update, &world));
    components.set_enabled(0, true);
    assert!(registry.has_runnable_systems(Stage::Update, &world));
    registry.run_systems(Stage::Update, &world, &mut SystemContext::new(&input, &mut commands));
    assert_eq!(components.get_typed::<Velocity>().unwrap().0, 1.0);
    components.set_enabled(0, false);
    assert!(!registry.has_runnable_systems(Stage::Update, &world));
    registry.register_system_for::<Velocity>("test::conditional", Stage::FixedUpdate, run);
    components.set_enabled(0, true);
    assert!(!registry.has_runnable_systems(Stage::Update, &world));
    assert!(registry.has_runnable_systems(Stage::FixedUpdate, &world));
    registry.run_systems(Stage::FixedUpdate, &world, &mut SystemContext::new(&input, &mut commands));
    assert_eq!(components.get_typed::<Velocity>().unwrap().0, 2.0);
}

#[test]
fn native_world_transform_reads_current_parent_writes_without_copyback_or_snapshot_refresh() {
    let mut world = SceneWorld::new();
    let parent = world.spawn("Parent");
    let child = world.spawn("Child");
    world.get_mut::<Transform>(child).unwrap().translation = Vec3::X;
    world.set_parent(child, Some(parent), false).unwrap();
    world.propagate().unwrap();
    world.with_world(|world| {
        world.get_mut::<Transform>(parent).unwrap().translation = Vec3::X * 10.0;
        assert_eq!(world.world_transform(child).unwrap().w_axis.x, 11.0);
        assert_eq!(world.get::<GlobalTransform>(child).unwrap().0.w_axis.x, 1.0, "boundary snapshot remains explicit");
        world.get_mut::<Parent>(parent).unwrap().0 = Some(child);
        assert!(world.world_transform(child).is_none(), "invalid native cycles cannot loop forever");
    });
}

#[test]
fn changing_saved_identity_preserves_entity_components_and_rejects_collisions() {
    use crate::Spin;
    let mut world = SceneWorld::new();
    let entity = world.spawn("Original");
    let other = world.spawn("Other");
    world.insert(entity, Spin { degrees_per_second: 37.0, ..Default::default() }).unwrap();
    let old = world.object_id(entity).unwrap();
    let other_id = world.object_id(other).unwrap();
    let new = ObjectId::new();
    world.set_object_id(entity, new).unwrap();
    assert_eq!(world.entity(old), None);
    assert_eq!(world.entity(new), Some(entity));
    assert_eq!(world.object_id(entity), Some(new));
    assert_eq!(world.get::<Spin>(entity).unwrap().degrees_per_second, 37.0);
    assert_eq!(world.set_object_id(entity, other_id), Err(HierarchyError::DuplicateId));
    assert_eq!(world.set_object_id(entity, ObjectId::from_raw(0)), Err(HierarchyError::InvalidId));
    assert_eq!(world.entity(new), Some(entity));
    assert_eq!(world.entity(other_id), Some(other));
    assert_eq!(world.object_id(entity), Some(new));
    world.despawn(other).unwrap();
    assert_eq!(world.get::<Spin>(entity).unwrap().degrees_per_second, 37.0);
}

#[test]
fn shared_query_scopes_exclude_component_views_and_release_after_unwind() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let mut world = SceneWorld::new();
    let owner = world.spawn("Object");
    world.insert(owner, Stateful { value: 5, live: 9 }).unwrap();
    let view = world.get_mut::<Stateful>(owner).unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| world.with_world(|_| ()))).is_err());
    assert_eq!(view.live, 9);
    drop(view);
    assert!(catch_unwind(AssertUnwindSafe(|| world.with_world(|query| {
        query.get_mut::<Stateful>(owner).unwrap().live = 10;
        panic!("query failed");
    }))).is_err());
    assert_eq!(world.get::<Stateful>(owner).unwrap().live, 10);
    assert_eq!(world.with_world(|query| query.query::<&Stateful>().iter().count()), 1);
}
