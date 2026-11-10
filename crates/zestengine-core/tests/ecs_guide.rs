//! The worked examples of `ECS.md` and `ENGINE_API.md`, compiled and run so the
//! guides cannot drift from the engine. Each module is one example: a component,
//! its system(s), a `register` function, and a test that drives it headlessly the
//! way the guide's "Testing systems" section describes. If you change an example
//! here, change the guide with it (the snippets are copied from these modules).

use glam::{Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};
use zestengine_core::ecs::{
    ContactPhase, ObjectContext, EntityShape, EntityTemplate, Owner, SceneEvent, SceneWorld, Stage, SystemContext, SystemWorld,
    run_ordered,
};
use zestengine_core::{
    ActionCtx, ActionDesc, ActionType, ActionValue, AnyBox, ArgValue, ComponentCommand, ComponentRegistry, InputState,
    MotionContext, ObjectId, ObjectRef, ParamDesc, PhysicsQuery, PropertyDesc, RayFilter, RayHit, Transform, TypedComponent, UiText,
};

/// Everything a system sees for one frame of a headless test.
#[derive(Default)]
struct Frame {
    dt: f32,
    input: InputState,
    events: Vec<AnyBox>,
    scene_events: Vec<SceneEvent>,
}

/// Run every system registered for `stage` once and return the commands they queued.
fn run(registry: &ComponentRegistry, scene: &SceneWorld, stage: Stage, frame: &Frame) -> Vec<ComponentCommand> {
    let mut commands = Vec::new();
    let mut ctx = SystemContext::new(&frame.input, &mut commands);
    ctx.dt = frame.dt;
    ctx.unscaled_dt = frame.dt;
    ctx.services.events = &frame.events;
    ctx.services.scene_events = &frame.scene_events;
    registry.run_systems(stage, scene, &mut ctx);
    commands
}

fn destroyed(commands: &[ComponentCommand]) -> Vec<ObjectId> {
    commands.iter().filter_map(|c| match c { ComponentCommand::Destroy(id) => Some(*id), _ => None }).collect()
}

fn emitted<E: 'static>(commands: &[ComponentCommand]) -> Vec<&E> {
    commands.iter().filter_map(|c| match c { ComponentCommand::Emit(event) => event.0.downcast_ref::<E>(), _ => None }).collect()
}

fn event<E: 'static>(value: E) -> AnyBox { AnyBox(Box::new(value)) }

/// A fake physics world with one object in it, five meters along any ray.
struct Wall(ObjectId);

impl PhysicsQuery for Wall {
    fn raycast(&self, origin: Vec3, dir: Vec3, _max: f32, _filter: &RayFilter) -> Option<RayHit> {
        Some(RayHit { object: self.0, point: origin + dir * 5.0, normal: -dir, distance: 5.0 })
    }
    fn overlap_sphere(&self, _: Vec3, _: f32, _: &RayFilter) -> Vec<ObjectId> { Vec::new() }
    fn velocity(&self, _: ObjectId) -> Option<Vec3> { None }
    fn grounded(&self, _: ObjectId) -> bool { true }
}

// ---------------------------------------------------------------- 1. a first system

mod rotator {
    use super::*;

    /// Spins the object around the Y axis.
    #[derive(Serialize, Deserialize)]
    pub struct Rotator {
        pub degrees_per_second: f32,
    }

    impl Default for Rotator {
        fn default() -> Self { Self { degrees_per_second: 45.0 } }
    }

    impl TypedComponent for Rotator {
        const NAME: &'static str = "Rotator";
    }

    fn rotate(world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) {
        for (_, (owner, rotator)) in world.query::<(&Owner, &Rotator)>().iter() {
            if !ctx.is_active(world, owner.0) { continue; }
            let mut transform = world.get_mut::<Transform>(owner.0).expect("every object has a Transform");
            let step = Quat::from_rotation_y(rotator.degrees_per_second.to_radians() * ctx.dt);
            transform.rotation = (step * transform.rotation).normalize();
        }
    }

    pub fn register(registry: &mut ComponentRegistry) {
        registry.register_ecs::<Rotator>();
        registry.register_system_for::<Rotator>("game::rotate", Stage::Update, rotate);
    }

    #[test]
    fn spins_active_objects_only() {
        let mut registry = ComponentRegistry::default();
        register(&mut registry);
        let mut scene = SceneWorld::new();
        let spinning = scene.spawn("Spinning");
        let parked = scene.spawn("Parked");
        for object in [spinning, parked] {
            scene.insert(object, Rotator { degrees_per_second: 90.0 }).unwrap();
        }
        scene.get_mut::<zestengine_core::ecs::Enabled>(parked).unwrap().0 = false;
        scene.propagate().unwrap();
        run(&registry, &scene, Stage::Update, &Frame { dt: 1.0, ..Default::default() });
        let expected = Quat::from_rotation_y(90f32.to_radians());
        assert!(scene.get::<Transform>(spinning).unwrap().rotation.abs_diff_eq(expected, 1e-5));
        assert_eq!(scene.get::<Transform>(parked).unwrap().rotation, Quat::IDENTITY, "disabled objects do not run");
    }
}

// ---------------------------------------------------------------- 2. input and movement

mod player_mover {
    use super::*;

    /// Walks the object with the `Move` action (WASD / left stick).
    #[derive(Serialize, Deserialize)]
    pub struct PlayerMover {
        pub speed: f32,
    }

    impl Default for PlayerMover {
        fn default() -> Self { Self { speed: 4.0 } }
    }

    impl TypedComponent for PlayerMover {
        const NAME: &'static str = "Player Mover";
    }

    impl PlayerMover {
        fn update(&mut self, obj: &mut ObjectContext<'_, '_>) {
            let direction = obj.input.axis2("Move"); // x = right, y = forward
            let step = Vec3::new(direction.x, 0.0, -direction.y) * self.speed * obj.dt;
            let position = obj.translation() + step;
            obj.set_translation(position);
        }
    }

    pub fn register(registry: &mut ComponentRegistry) {
        registry.register_ecs::<PlayerMover>();
        registry.register_system_for::<PlayerMover>("game::player_mover", Stage::Update, |world, ctx| {
            run_ordered(world, ctx, PlayerMover::update)
        });
    }

    #[test]
    fn moves_with_the_move_action() {
        let mut registry = ComponentRegistry::default();
        register(&mut registry);
        let mut scene = SceneWorld::new();
        let player = scene.spawn("Player");
        scene.insert(player, PlayerMover::default()).unwrap();
        let input = InputState::from_actions(
            [("Move".to_string(), ActionValue { axis2: Vec2::new(1.0, 1.0), ..Default::default() })],
            Vec2::ZERO,
        );
        run(&registry, &scene, Stage::Update, &Frame { dt: 0.5, input, ..Default::default() });
        assert_eq!(scene.get::<Transform>(player).unwrap().translation, Vec3::new(2.0, 0.0, -2.0));
    }
}

// ---------------------------------------------------------------- 3. events: hits and deaths

mod hitpoints {
    use super::*;

    /// Event: something hit `target` for `amount`.
    pub struct Hit {
        pub target: ObjectId,
        pub amount: f32,
    }

    /// Event: the object ran out of hitpoints.
    pub struct Died {
        pub object: ObjectId,
    }

    #[derive(Serialize, Deserialize)]
    pub struct Hitpoints {
        pub current: f32,
        pub max: f32,
    }

    impl Default for Hitpoints {
        fn default() -> Self { Self { current: 100.0, max: 100.0 } }
    }

    impl TypedComponent for Hitpoints {
        const NAME: &'static str = "Hitpoints";
    }

    fn apply_hits(world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) {
        // Events emitted during the previous frame are readable now.
        let hits: Vec<&Hit> = ctx.events::<Hit>().collect();
        if hits.is_empty() { return; }
        for (_, (owner, hp)) in world.query::<(&Owner, &mut Hitpoints)>().iter() {
            let Some(me) = world.object_id(owner.0) else { continue };
            let before = hp.current;
            for hit in hits.iter().filter(|hit| hit.target == me) {
                hp.current = (hp.current - hit.amount).max(0.0);
            }
            if before > 0.0 && hp.current == 0.0 {
                ctx.emit(Died { object: me }); // readable next frame
            }
        }
    }

    fn remove_the_dead(_: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) {
        for died in ctx.events::<Died>() {
            ctx.commands.push(ComponentCommand::Destroy(died.object));
        }
    }

    pub fn register(registry: &mut ComponentRegistry) {
        registry.register_ecs::<Hitpoints>();
        registry.register_system_for::<Hitpoints>("game::apply_hits", Stage::Gameplay, apply_hits);
        registry.register_system("game::remove_the_dead", Stage::Gameplay, remove_the_dead);
    }

    #[test]
    fn hits_arrive_next_frame_and_death_is_announced_once() {
        let mut registry = ComponentRegistry::default();
        register(&mut registry);
        let mut scene = SceneWorld::new();
        let target = scene.spawn("Target");
        scene.insert(target, Hitpoints::default()).unwrap();
        let id = scene.object_id(target).unwrap();

        let first = run(&registry, &scene, Stage::Gameplay, &Frame { dt: 0.016, events: vec![event(Hit { target: id, amount: 30.0 })], ..Default::default() });
        assert_eq!(scene.get::<Hitpoints>(target).unwrap().current, 70.0);
        assert!(emitted::<Died>(&first).is_empty());

        let second = run(&registry, &scene, Stage::Gameplay, &Frame { dt: 0.016, events: vec![event(Hit { target: id, amount: 80.0 })], ..Default::default() });
        assert_eq!(scene.get::<Hitpoints>(target).unwrap().current, 0.0);
        assert_eq!(emitted::<Died>(&second).len(), 1, "death is emitted for the next frame");
        assert!(destroyed(&second).is_empty(), "nobody has received Died yet");

        // The engine hands last frame's emitted events to every system next frame.
        let third = run(&registry, &scene, Stage::Gameplay, &Frame { dt: 0.016, events: vec![event(Died { object: id })], ..Default::default() });
        assert_eq!(destroyed(&third), [id]);
        let again = run(&registry, &scene, Stage::Gameplay, &Frame { dt: 0.016, events: vec![event(Hit { target: id, amount: 5.0 })], ..Default::default() });
        assert!(emitted::<Died>(&again).is_empty(), "already dead: not announced twice");
    }
}

// ---------------------------------------------------------------- 4. timers: lifetime

mod lifetime {
    use super::*;

    /// Destroys its object after `seconds_left`.
    #[derive(Serialize, Deserialize)]
    pub struct Lifetime {
        pub seconds_left: f32,
    }

    impl Default for Lifetime {
        fn default() -> Self { Self { seconds_left: 3.0 } }
    }

    impl TypedComponent for Lifetime {
        const NAME: &'static str = "Lifetime";
    }

    impl Lifetime {
        fn update(&mut self, obj: &mut ObjectContext<'_, '_>) {
            self.seconds_left -= obj.dt;
            if self.seconds_left <= 0.0 {
                obj.destroy_self();
            }
        }
    }

    pub fn register(registry: &mut ComponentRegistry) {
        registry.register_ecs::<Lifetime>();
        registry.register_system_for::<Lifetime>("game::lifetime", Stage::Update, |world, ctx| {
            run_ordered(world, ctx, Lifetime::update)
        });
    }

    #[test]
    fn expires_after_its_time() {
        let mut registry = ComponentRegistry::default();
        register(&mut registry);
        let mut scene = SceneWorld::new();
        let spark = scene.spawn("Spark");
        scene.insert(spark, Lifetime { seconds_left: 1.5 }).unwrap();
        let frame = Frame { dt: 1.0, ..Default::default() };
        assert!(destroyed(&run(&registry, &scene, Stage::Update, &frame)).is_empty());
        assert_eq!(destroyed(&run(&registry, &scene, Stage::Update, &frame)), [scene.object_id(spark).unwrap()]);
    }
}

// ---------------------------------------------------------------- 5. spawning entities

mod crate_spawner {
    use super::*;
    use super::lifetime::Lifetime;

    /// Drops a crate above itself every `every` seconds.
    #[derive(Serialize, Deserialize)]
    pub struct CrateSpawner {
        pub every: f32,
        #[serde(skip)]
        pub timer: f32,
    }

    impl Default for CrateSpawner {
        fn default() -> Self { Self { every: 2.0, timer: 0.0 } }
    }

    impl TypedComponent for CrateSpawner {
        const NAME: &'static str = "Crate Spawner";
    }

    impl CrateSpawner {
        fn update(&mut self, obj: &mut ObjectContext<'_, '_>) {
            self.timer -= obj.dt;
            if self.timer <= 0.0 {
                self.timer = self.every;
                let above = obj.world_position() + Vec3::Y;
                obj.spawn_entity("Crate", above, Quat::IDENTITY);
            }
        }
    }

    /// The entity: a cube that deletes itself after ten seconds.
    fn crate_entity() -> EntityTemplate {
        EntityTemplate::new(EntityShape::Cube).with(Lifetime { seconds_left: 10.0 })
    }

    pub fn register(registry: &mut ComponentRegistry) {
        registry.register_ecs::<CrateSpawner>();
        registry.register_entity("Crate", crate_entity);
        registry.register_system_for::<CrateSpawner>("game::crate_spawner", Stage::Gameplay, |world, ctx| {
            run_ordered(world, ctx, CrateSpawner::update)
        });
    }

    #[test]
    fn spawns_on_a_timer() {
        let mut registry = ComponentRegistry::default();
        lifetime::register(&mut registry);
        register(&mut registry);
        let mut scene = SceneWorld::new();
        let spawner = scene.spawn("Spawner");
        scene.get_mut::<Transform>(spawner).unwrap().translation = Vec3::new(4.0, 0.0, 0.0);
        scene.insert(spawner, CrateSpawner::default()).unwrap();
        let frame = Frame { dt: 0.1, ..Default::default() };

        let first = run(&registry, &scene, Stage::Gameplay, &frame);
        let [ComponentCommand::SpawnEntity { name, translation, .. }] = first.as_slice() else { panic!("one spawn expected: {first:?}") };
        assert_eq!(name, "Crate");
        assert_eq!(*translation, Vec3::new(4.0, 1.0, 0.0));
        assert!(run(&registry, &scene, Stage::Gameplay, &frame).is_empty(), "the timer is cooling down");

        let template = registry.create_entity("Crate").unwrap();
        assert_eq!(template.shape, EntityShape::Cube);
        assert_eq!(template.components.len(), 1);
    }
}

// ---------------------------------------------------------------- 6. other objects: following a target

mod follower {
    use super::*;

    /// Walks toward `target` at `speed`, whatever the hierarchy.
    #[derive(Serialize, Deserialize)]
    pub struct Follower {
        pub target: ObjectRef,
        pub speed: f32,
    }

    impl Default for Follower {
        fn default() -> Self { Self { target: ObjectRef::NONE, speed: 2.0 } }
    }

    impl TypedComponent for Follower {
        const NAME: &'static str = "Follower";
    }

    impl Follower {
        fn update(&mut self, obj: &mut ObjectContext<'_, '_>) {
            let Some(goal) = obj.position_of(self.target) else { return }; // world position
            let here = obj.world_position();
            let to_goal = goal - here;
            let step = self.speed * obj.dt;
            if to_goal.length() <= step {
                obj.set_world_position(goal);
            } else {
                obj.set_world_position(here + to_goal.normalize() * step);
            }
        }
    }

    pub fn register(registry: &mut ComponentRegistry) {
        registry.register_ecs::<Follower>();
        registry.register_system_for::<Follower>("game::follower", Stage::Update, |world, ctx| {
            run_ordered(world, ctx, Follower::update)
        });
    }

    #[test]
    fn closes_the_distance_without_overshooting() {
        let mut registry = ComponentRegistry::default();
        register(&mut registry);
        let mut scene = SceneWorld::new();
        let leader = scene.spawn("Leader");
        let chaser = scene.spawn("Chaser");
        scene.get_mut::<Transform>(leader).unwrap().translation = Vec3::new(10.0, 0.0, 0.0);
        scene.insert(chaser, Follower { target: scene.object_id(leader).unwrap().into(), speed: 2.0 }).unwrap();
        run(&registry, &scene, Stage::Update, &Frame { dt: 1.0, ..Default::default() });
        assert_eq!(scene.get::<Transform>(chaser).unwrap().translation, Vec3::new(2.0, 0.0, 0.0));
        run(&registry, &scene, Stage::Update, &Frame { dt: 10.0, ..Default::default() });
        assert_eq!(scene.get::<Transform>(chaser).unwrap().translation, Vec3::new(10.0, 0.0, 0.0), "stops on the target");
    }
}

// ---------------------------------------------------------------- 7. contact events: pickups and score

mod pickups {
    use super::*;

    /// Marker: objects that can pick coins up.
    #[derive(Default, Serialize, Deserialize)]
    pub struct Collector {}

    impl TypedComponent for Collector {
        const NAME: &'static str = "Collector";
    }

    #[derive(Serialize, Deserialize)]
    pub struct Coin {
        pub value: u32,
    }

    impl Default for Coin {
        fn default() -> Self { Self { value: 1 } }
    }

    impl TypedComponent for Coin {
        const NAME: &'static str = "Coin";
    }

    /// Event: `by` picked up something worth `value`.
    pub struct Collected {
        pub by: ObjectId,
        pub value: u32,
    }

    /// Shows the score in this object's `UiText`.
    #[derive(Default, Serialize, Deserialize)]
    pub struct ScoreBoard {
        #[serde(skip)]
        pub score: u32,
    }

    impl TypedComponent for ScoreBoard {
        const NAME: &'static str = "Score Board";
    }

    impl ScoreBoard {
        fn update(&mut self, obj: &mut ObjectContext<'_, '_>) {
            for collected in obj.events::<Collected>() {
                self.score += collected.value;
            }
            let id = obj.self_id();
            let text = format!("Score {}", self.score);
            obj.edit::<UiText>(id, move |ui| ui.text = text);
        }
    }

    pub fn register(registry: &mut ComponentRegistry) {
        registry.register_ecs::<Collector>();
        registry.register_ecs::<Coin>();
        registry.register_ecs::<ScoreBoard>();
        // Contacts arrive in Stage::Events, through `services.scene_events`.
        registry.register_system_for::<Coin>("game::coin_pickup", Stage::Events, |world, ctx| {
            let contacts = ctx.services.scene_events;
            run_ordered(world, ctx, |coin: &mut Coin, obj| {
                let me = obj.self_id();
                for contact in contacts {
                    if let SceneEvent::Contact { object, other, trigger: true, phase: ContactPhase::Enter, .. } = contact
                        && *object == me
                        && obj.get::<Collector>(*other).is_some()
                    {
                        obj.emit(Collected { by: *other, value: coin.value });
                        obj.destroy_self();
                    }
                }
            })
        });
        registry.register_system_for::<ScoreBoard>("game::score_board", Stage::LateUpdate, |world, ctx| {
            run_ordered(world, ctx, ScoreBoard::update)
        });
    }

    #[test]
    fn collecting_a_coin_scores_next_frame() {
        let mut registry = ComponentRegistry::default();
        register(&mut registry);
        let mut scene = SceneWorld::new();
        let coin = scene.spawn("Coin");
        let player = scene.spawn("Player");
        let board = scene.spawn("Board");
        scene.insert(coin, Coin { value: 5 }).unwrap();
        scene.insert(player, Collector {}).unwrap();
        scene.insert(board, ScoreBoard::default()).unwrap();
        scene.insert(board, UiText::default()).unwrap();
        let (coin_id, player_id) = (scene.object_id(coin).unwrap(), scene.object_id(player).unwrap());

        let touch = |other| SceneEvent::Contact { object: coin_id, other, trigger: true, phase: ContactPhase::Enter, two_dimensional: false };
        let board_id = scene.object_id(board).unwrap();
        let ignored = run(&registry, &scene, Stage::Events, &Frame { scene_events: vec![touch(board_id)], ..Default::default() });
        assert!(ignored.is_empty(), "only a Collector can pick the coin up");
        let commands = run(&registry, &scene, Stage::Events, &Frame { scene_events: vec![touch(player_id)], ..Default::default() });
        assert_eq!(destroyed(&commands), [coin_id]);
        let collected = emitted::<Collected>(&commands);
        assert_eq!((collected[0].by, collected[0].value), (player_id, 5));

        run(&registry, &scene, Stage::LateUpdate, &Frame { events: vec![event(Collected { by: player_id, value: 5 }), event(Collected { by: player_id, value: 10 })], ..Default::default() });
        assert_eq!(scene.get::<UiText>(board).unwrap().text, "Score 15");
    }
}

// ---------------------------------------------------------------- 8. start, state and actions: a door

mod door {
    use super::*;

    /// A door that slides up when opened. Its actions and its `open` property
    /// appear in the editor's event-binding dropdowns.
    #[derive(Serialize, Deserialize)]
    pub struct Door {
        pub lift: f32,
        pub speed: f32,
        #[serde(skip)]
        pub wants_open: bool,
        #[serde(skip)]
        pub closed_y: Option<f32>,
    }

    impl Default for Door {
        fn default() -> Self { Self { lift: 3.0, speed: 2.0, wants_open: false, closed_y: None } }
    }

    static DOOR_ACTIONS: &[ActionDesc] = &[
        ActionDesc::new("Open", "Open", &[], true),
        ActionDesc::new("Close", "Close", &[], true),
        ActionDesc::new("SetSpeed", "Set Speed", &[ParamDesc::float("speed", 2.0)], true),
    ];
    static DOOR_PROPERTIES: &[PropertyDesc] = &[PropertyDesc::new("open", ActionType::Bool)];

    impl TypedComponent for Door {
        const NAME: &'static str = "Door";

        fn actions(&self) -> &'static [ActionDesc] { DOOR_ACTIONS }
        fn properties(&self) -> &'static [PropertyDesc] { DOOR_PROPERTIES }

        fn read_property(&self, property: &str) -> Option<ArgValue> {
            (property == "open").then_some(ArgValue::Bool(self.wants_open))
        }

        fn invoke_action(&mut self, _ctx: &mut ActionCtx<'_>, action: &str, args: &[ArgValue]) -> Result<(), String> {
            match action {
                "Open" => self.wants_open = true,
                "Close" => self.wants_open = false,
                "SetSpeed" => self.speed = args[0].as_f32().max(0.0),
                _ => return Err(zestengine_core::actions::unknown_action(Self::NAME, action)),
            }
            Ok(())
        }
    }

    impl Door {
        /// Runs once for each new instance (Stage::Start): remember where "closed" is.
        fn start(&mut self, obj: &mut ObjectContext<'_, '_>) {
            if self.closed_y.is_none() {
                self.closed_y = Some(obj.translation().y);
            }
        }

        fn update(&mut self, obj: &mut ObjectContext<'_, '_>) {
            let Some(closed) = self.closed_y else { return };
            let goal = if self.wants_open { closed + self.lift } else { closed };
            let mut position = obj.translation();
            let step = self.speed * obj.dt;
            position.y += (goal - position.y).clamp(-step, step);
            obj.set_translation(position);
        }
    }

    pub fn register(registry: &mut ComponentRegistry) {
        registry.register_ecs::<Door>();
        // Only runs while some door has not been initialised yet.
        registry.register_system_if(
            "game::door_start",
            Stage::Start,
            |world, ctx| run_ordered(world, ctx, Door::start),
            |world| world.query::<&Door>().iter().any(|(_, door)| door.closed_y.is_none()),
        );
        registry.register_system_for::<Door>("game::door", Stage::Update, |world, ctx| {
            run_ordered(world, ctx, Door::update)
        });
    }

    #[test]
    fn slides_up_when_opened_and_never_past_its_lift() {
        let mut registry = ComponentRegistry::default();
        register(&mut registry);
        let mut scene = SceneWorld::new();
        let gate = scene.spawn("Gate");
        scene.get_mut::<Transform>(gate).unwrap().translation.y = 1.0;
        scene.insert(gate, Door::default()).unwrap();
        let frame = Frame { dt: 1.0, ..Default::default() };

        assert!(registry.has_runnable_systems(Stage::Start, &scene));
        run(&registry, &scene, Stage::Start, &frame);
        assert_eq!(scene.get::<Door>(gate).unwrap().closed_y, Some(1.0));
        assert!(!registry.has_runnable_systems(Stage::Start, &scene), "started: the Start system sleeps");

        run(&registry, &scene, Stage::Update, &frame);
        assert_eq!(scene.get::<Transform>(gate).unwrap().translation.y, 1.0, "closed doors stay put");

        // Event bindings call actions like this (the engine routes them for you).
        let mut commands = Vec::new();
        let mut ctx = ActionCtx { self_id: scene.object_id(gate).unwrap(), instigator: None, commands: &mut commands, reader: None };
        scene.get_mut::<Door>(gate).unwrap().invoke_action(&mut ctx, "Open", &[]).unwrap();
        assert_eq!(scene.get::<Door>(gate).unwrap().read_property("open"), Some(ArgValue::Bool(true)));
        assert!(scene.get_mut::<Door>(gate).unwrap().invoke_action(&mut ctx, "Explode", &[]).is_err());

        run(&registry, &scene, Stage::Update, &frame);
        assert_eq!(scene.get::<Transform>(gate).unwrap().translation.y, 3.0);
        run(&registry, &scene, Stage::Update, &frame);
        run(&registry, &scene, Stage::Update, &frame);
        assert_eq!(scene.get::<Transform>(gate).unwrap().translation.y, 4.0, "closed + lift");
    }
}

// ---------------------------------------------------------------- 9. two components talking: a poison zone

mod poison_zone {
    use super::*;
    use super::hitpoints::Hitpoints;

    /// Drains `damage_per_second` from every `Hitpoints` object inside `radius`.
    #[derive(Serialize, Deserialize)]
    pub struct PoisonZone {
        pub radius: f32,
        pub damage_per_second: f32,
    }

    impl Default for PoisonZone {
        fn default() -> Self { Self { radius: 3.0, damage_per_second: 10.0 } }
    }

    impl TypedComponent for PoisonZone {
        const NAME: &'static str = "Poison Zone";
    }

    fn poison(world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) {
        for (_, (owner, zone)) in world.query::<(&Owner, &PoisonZone)>().iter() {
            if !ctx.is_active(world, owner.0) { continue; }
            let Some(center) = world.world_transform(owner.0).map(|m| m.w_axis.truncate()) else { continue };
            // A nested query is fine because it borrows different components.
            // Two queries that both borrow the same component mutably would panic.
            for (_, (victim, hp)) in world.query::<(&Owner, &mut Hitpoints)>().iter() {
                let Some(at) = world.world_transform(victim.0).map(|m| m.w_axis.truncate()) else { continue };
                if at.distance(center) <= zone.radius {
                    hp.current = (hp.current - zone.damage_per_second * ctx.dt).max(0.0);
                }
            }
        }
    }

    pub fn register(registry: &mut ComponentRegistry) {
        registry.register_ecs::<PoisonZone>();
        registry.register_system_for::<PoisonZone>("game::poison", Stage::Gameplay, poison);
    }

    #[test]
    fn only_objects_inside_the_radius_are_hurt() {
        let mut registry = ComponentRegistry::default();
        hitpoints::register(&mut registry);
        register(&mut registry);
        let mut scene = SceneWorld::new();
        let zone = scene.spawn("Swamp");
        let inside = scene.spawn("Inside");
        let outside = scene.spawn("Outside");
        scene.insert(zone, PoisonZone::default()).unwrap();
        scene.get_mut::<Transform>(inside).unwrap().translation = Vec3::new(1.0, 0.0, 0.0);
        scene.get_mut::<Transform>(outside).unwrap().translation = Vec3::new(20.0, 0.0, 0.0);
        for victim in [inside, outside] {
            scene.insert(victim, Hitpoints::default()).unwrap();
        }
        run(&registry, &scene, Stage::Gameplay, &Frame { dt: 0.5, ..Default::default() });
        assert_eq!(scene.get::<Hitpoints>(inside).unwrap().current, 95.0);
        assert_eq!(scene.get::<Hitpoints>(outside).unwrap().current, 100.0);
    }
}

// ---------------------------------------------------------------- 10. physics: a ray gun

mod laser {
    use super::*;
    use super::hitpoints::Hitpoints;

    /// Press `Fire`: raycast forward, hurt what is hit, shove it and play a sound.
    #[derive(Serialize, Deserialize)]
    pub struct Laser {
        pub damage: f32,
        pub range: f32,
    }

    impl Default for Laser {
        fn default() -> Self { Self { damage: 25.0, range: 100.0 } }
    }

    impl TypedComponent for Laser {
        const NAME: &'static str = "Laser";
    }

    impl Laser {
        fn update(&mut self, obj: &mut ObjectContext<'_, '_>) {
            if !obj.input.pressed("Fire") { return; }
            let aim = obj.self_transform(); // world position and orientation
            let Some(hit) = obj.raycast(aim.translation, aim.forward(), self.range) else { return };
            let damage = self.damage;
            obj.edit::<Hitpoints>(hit.object, move |hp| hp.current = (hp.current - damage).max(0.0));
            obj.add_impulse(hit.object, aim.forward() * 5.0);
            obj.play_sound_at("audio/zap.ogg", hit.point, 1.0);
        }
    }

    pub fn register(registry: &mut ComponentRegistry) {
        registry.register_ecs::<Laser>();
        registry.register_system_for::<Laser>("game::laser", Stage::Gameplay, |world, ctx| {
            run_ordered(world, ctx, Laser::update)
        });
    }

    #[test]
    fn firing_hurts_pushes_and_sounds_off() {
        let mut registry = ComponentRegistry::default();
        hitpoints::register(&mut registry);
        register(&mut registry);
        let mut scene = SceneWorld::new();
        let gun = scene.spawn("Gun");
        let dummy = scene.spawn("Dummy");
        scene.insert(gun, Laser::default()).unwrap();
        scene.insert(dummy, Hitpoints::default()).unwrap();
        let wall = Wall(scene.object_id(dummy).unwrap());

        let mut commands = Vec::new();
        let fire = InputState::from_actions(
            [("Fire".to_string(), ActionValue { down: true, just_pressed: true, ..Default::default() })],
            Vec2::ZERO,
        );
        let mut ctx = SystemContext::new(&fire, &mut commands);
        ctx.services.physics = Some(&wall);
        registry.run_systems(Stage::Gameplay, &scene, &mut ctx);

        assert_eq!(scene.get::<Hitpoints>(dummy).unwrap().current, 75.0);
        assert!(commands.iter().any(|c| matches!(c, ComponentCommand::AddImpulse { impulse, .. } if *impulse == Vec3::NEG_Z * 5.0)));
        assert!(commands.iter().any(|c| matches!(c, ComponentCommand::PlaySound { clip, .. } if clip == "audio/zap.ogg")));
    }
}

// ---------------------------------------------------------------- 11. a smooth follow camera

mod smooth_follow {
    use super::*;

    /// Glides to `target + offset`, at the same pace on every frame rate.
    #[derive(Serialize, Deserialize)]
    pub struct SmoothFollow {
        pub target: ObjectRef,
        pub offset: [f32; 3],
        pub sharpness: f32,
    }

    impl Default for SmoothFollow {
        fn default() -> Self { Self { target: ObjectRef::NONE, offset: [0.0, 3.0, 6.0], sharpness: 8.0 } }
    }

    impl TypedComponent for SmoothFollow {
        const NAME: &'static str = "Smooth Follow";
    }

    impl SmoothFollow {
        fn update(&mut self, obj: &mut ObjectContext<'_, '_>) {
            let Some(goal) = obj.position_of(self.target) else { return };
            let wanted = goal + Vec3::from(self.offset);
            let here = obj.world_position();
            // 1 - e^(-k·dt) is the share of the gap to close this frame: the result is
            // the same whether the game runs at 30 or 240 frames per second.
            let blend = 1.0 - (-self.sharpness * obj.dt).exp();
            obj.set_world_position(here.lerp(wanted, blend));
        }
    }

    pub fn register(registry: &mut ComponentRegistry) {
        registry.register_ecs::<SmoothFollow>();
        // LateUpdate: after the target has moved this frame.
        registry.register_system_for::<SmoothFollow>("game::smooth_follow", Stage::LateUpdate, |world, ctx| {
            run_ordered(world, ctx, SmoothFollow::update)
        });
    }

    #[test]
    fn closes_the_same_share_of_the_gap_at_any_frame_rate() {
        let mut registry = ComponentRegistry::default();
        register(&mut registry);
        let build = |steps: u32| {
            let mut scene = SceneWorld::new();
            let player = scene.spawn("Player");
            let camera = scene.spawn("Camera");
            scene.get_mut::<Transform>(player).unwrap().translation = Vec3::new(10.0, 0.0, 0.0);
            scene.insert(camera, SmoothFollow { target: scene.object_id(player).unwrap().into(), offset: [0.0, 0.0, 0.0], sharpness: 10.0 }).unwrap();
            for _ in 0..steps {
                run(&registry, &scene, Stage::LateUpdate, &Frame { dt: 0.1 / steps as f32, ..Default::default() });
            }
            let x = scene.get::<Transform>(camera).unwrap().translation.x;
            x
        };
        let expected = 10.0 * (1.0 - (-1.0f32).exp());
        assert!((build(1) - expected).abs() < 1e-4);
        assert!((build(10) - expected).abs() < 0.05, "ten small steps land in the same place as one big one");
    }
}

// ---------------------------------------------------------------- 12. click to select

mod click_select {
    use super::*;

    /// Click with `Fire` to select whatever is under the pointer.
    #[derive(Default, Serialize, Deserialize)]
    pub struct ClickToSelect {
        #[serde(skip)]
        pub selected: Option<ObjectId>,
    }

    impl TypedComponent for ClickToSelect {
        const NAME: &'static str = "Click To Select";
    }

    impl ClickToSelect {
        fn update(&mut self, obj: &mut ObjectContext<'_, '_>) {
            if !obj.input.pressed("Fire") || obj.pointer_over_ui() { return; }
            let Some((origin, direction)) = obj.pointer_ray() else { return };
            self.selected = obj.raycast(origin, direction, 500.0).map(|hit| hit.object);
        }
    }

    pub fn register(registry: &mut ComponentRegistry) {
        registry.register_ecs::<ClickToSelect>();
        registry.register_system_for::<ClickToSelect>("game::click_select", Stage::Update, |world, ctx| {
            run_ordered(world, ctx, ClickToSelect::update)
        });
    }

    #[test]
    fn selects_the_object_under_the_pointer_unless_the_ui_has_it() {
        let mut registry = ComponentRegistry::default();
        register(&mut registry);
        let mut scene = SceneWorld::new();
        let picker = scene.spawn("Picker");
        let crate_ = scene.spawn("Crate");
        scene.insert(picker, ClickToSelect::default()).unwrap();
        let wall = Wall(scene.object_id(crate_).unwrap());
        let click = InputState::from_actions(
            [("Fire".to_string(), ActionValue { down: true, just_pressed: true, ..Default::default() })],
            Vec2::ZERO,
        );
        let mut screen = zestengine_core::ScreenInfo {
            size: Vec2::new(800.0, 600.0),
            pointer: Some(Vec2::new(400.0, 300.0)),
            view: glam::Mat4::IDENTITY,
            proj: glam::Mat4::perspective_rh(1.0, 800.0 / 600.0, 0.1, 100.0),
            ..Default::default()
        };

        screen.pointer_over_ui = true;
        let mut commands = Vec::new();
        let mut ctx = SystemContext::new(&click, &mut commands);
        ctx.services.physics = Some(&wall);
        ctx.services.screen = Some(&screen);
        registry.run_systems(Stage::Update, &scene, &mut ctx);
        assert_eq!(scene.get::<ClickToSelect>(picker).unwrap().selected, None, "a click on a button is not a click on the world");

        screen.pointer_over_ui = false;
        let mut ctx = SystemContext::new(&click, &mut commands);
        ctx.services.physics = Some(&wall);
        ctx.services.screen = Some(&screen);
        registry.run_systems(Stage::Update, &scene, &mut ctx);
        assert_eq!(scene.get::<ClickToSelect>(picker).unwrap().selected, scene.object_id(crate_));
    }
}

// ---------------------------------------------------------------- 13. pausing the game

mod pause {
    use super::*;

    /// Escape pauses and resumes: time stops, the cursor is released.
    #[derive(Default, Serialize, Deserialize)]
    pub struct PauseToggle {
        #[serde(skip)]
        pub paused: bool,
    }

    impl TypedComponent for PauseToggle {
        const NAME: &'static str = "Pause Toggle";
    }

    impl PauseToggle {
        fn update(&mut self, obj: &mut ObjectContext<'_, '_>) {
            if obj.ui_cancel_pressed() {
                self.paused = !self.paused;
                obj.set_time_scale(if self.paused { 0.0 } else { 1.0 });
                obj.set_cursor_locked(!self.paused);
            }
        }
    }

    pub fn register(registry: &mut ComponentRegistry) {
        registry.register_ecs::<PauseToggle>();
        registry.register_system_for::<PauseToggle>("game::pause", Stage::Update, |world, ctx| {
            run_ordered(world, ctx, PauseToggle::update)
        });
    }

    #[test]
    fn escape_toggles_the_time_scale_and_the_cursor() {
        let mut registry = ComponentRegistry::default();
        register(&mut registry);
        let mut scene = SceneWorld::new();
        let menu = scene.spawn("Menu");
        scene.insert(menu, PauseToggle::default()).unwrap();
        let screen = zestengine_core::ScreenInfo { ui_cancel: true, ..Default::default() };
        let input = InputState::default();
        let press = |scene: &SceneWorld| {
            let mut commands = Vec::new();
            let mut ctx = SystemContext::new(&input, &mut commands);
            ctx.services.screen = Some(&screen);
            registry.run_systems(Stage::Update, scene, &mut ctx);
            commands
        };
        let paused = press(&scene);
        assert!(matches!(paused.as_slice(), [ComponentCommand::SetTimeScale(0.0), ComponentCommand::SetCursorLocked(false)]));
        let resumed = press(&scene);
        assert!(matches!(resumed.as_slice(), [ComponentCommand::SetTimeScale(1.0), ComponentCommand::SetCursorLocked(true)]));
    }
}
