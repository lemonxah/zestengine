use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::actions::{ActionCtx, ActionDesc, ActionType, ArgValue, PropertyDesc};
use crate::ecs::{ContactPhase, ObjectContext, Owner, Parent, SceneEvent, Stage, run_ordered};
use crate::{ComponentCommand, MotionContext, ObjectId, ObjectRef, RigidBody, TypedComponent};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrokenEvent {
    pub object: ObjectId,
    pub debris: Option<ObjectId>,
    pub position: Vec3,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Breakable {
    pub broken_prefab: String,
    pub break_with_health: bool,
    pub impact_speed: f32,
    pub keep_object: bool,
    #[serde(skip)]
    pub broken: bool,
    #[serde(skip)]
    pub requested: bool,
    #[serde(skip)]
    pub last_velocity: Vec3,
}

impl Default for Breakable {
    fn default() -> Self {
        Self { broken_prefab: String::new(), break_with_health: true, impact_speed: 0.0, keep_object: false, broken: false, requested: false, last_velocity: Vec3::ZERO }
    }
}

impl Breakable {
    pub fn impact(&self, velocity_now: Vec3, other_velocity: Vec3) -> f32 {
        (velocity_now - self.last_velocity).length().max((self.last_velocity - other_velocity).length())
    }

    pub fn breaks_at(&self, impact: f32) -> bool {
        !self.broken && self.impact_speed > 0.0 && impact >= self.impact_speed
    }

    fn shatter(&mut self, obj: &mut ObjectContext<'_, '_>) {
        if self.broken {
            return;
        }
        self.broken = true;
        self.requested = false;
        let id = obj.self_id();
        let (_, rotation, position) = obj.world.world_transform(obj.owner).map_or((Vec3::ONE, glam::Quat::IDENTITY, Vec3::ZERO), |m| m.to_scale_rotation_translation());
        let debris = (!self.broken_prefab.trim().is_empty()).then(|| obj.spawn_prefab(self.broken_prefab.clone(), position, rotation));
        obj.emit(BrokenEvent { object: id, debris, position });
        if self.keep_object {
            obj.set_active(ObjectRef::new(id), false);
        } else {
            obj.destroy_self();
        }
    }

    fn update_native(&mut self, obj: &mut ObjectContext<'_, '_>) {
        if self.broken {
            return;
        }
        let dead = self.break_with_health && obj.world.get::<crate::Health>(obj.owner).is_some_and(|health| health.is_dead());
        if self.requested || dead {
            self.shatter(obj);
            return;
        }
        let id = obj.self_id();
        self.last_velocity = obj.services.physics.and_then(|physics| physics.velocity(id)).unwrap_or(Vec3::ZERO);
    }

    fn contact_native(&mut self, obj: &mut ObjectContext<'_, '_>, other: ObjectId) {
        if self.broken || self.impact_speed <= 0.0 {
            return;
        }
        let id = obj.self_id();
        let velocity = |object: ObjectId| obj.services.physics.and_then(|physics| physics.velocity(object)).unwrap_or(Vec3::ZERO);
        let impact = self.impact(velocity(id), velocity(other));
        if self.breaks_at(impact) {
            self.shatter(obj);
        }
    }
}

static BREAKABLE_ACTIONS: &[ActionDesc] = &[ActionDesc::new("Break", "Break", &[], false)];
static BREAKABLE_PROPERTIES: &[PropertyDesc] = &[PropertyDesc::new("broken", ActionType::Bool)];

impl TypedComponent for Breakable {
    const NAME: &'static str = "Breakable";

    fn actions(&self) -> &'static [ActionDesc] {
        BREAKABLE_ACTIONS
    }

    fn invoke_action(&mut self, _ctx: &mut ActionCtx, name: &str, _args: &[ArgValue]) -> Result<(), String> {
        match name {
            "Break" => self.requested = !self.broken,
            _ => return Err(crate::actions::unknown_action(Self::NAME, name)),
        }
        Ok(())
    }

    fn properties(&self) -> &'static [PropertyDesc] {
        BREAKABLE_PROPERTIES
    }

    fn read_property(&self, name: &str) -> Option<ArgValue> {
        (name == "broken").then_some(ArgValue::Bool(self.broken))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DebrisBurst {
    pub impulse: f32,
    pub upward: f32,
    pub lifetime: f32,
    #[serde(skip)]
    pub fired: bool,
    #[serde(skip)]
    pub age: f32,
}

impl Default for DebrisBurst {
    fn default() -> Self {
        Self { impulse: 2.0, upward: 0.35, lifetime: 8.0, fired: false, age: 0.0 }
    }
}

pub fn burst_direction(offset: Vec3, upward: f32, index: usize) -> Vec3 {
    let outward = if offset.length_squared() > 1e-6 {
        offset.normalize()
    } else {
        let angle = index as f32 * 2.399_963;
        Vec3::new(angle.cos(), 0.0, angle.sin())
    };
    let lifted = outward + Vec3::Y * upward.max(0.0);
    if lifted.length_squared() > 1e-6 { lifted.normalize() } else { Vec3::Y }
}

impl DebrisBurst {
    fn update_native(&mut self, obj: &mut ObjectContext<'_, '_>) {
        if !self.fired {
            self.fired = true;
            let origin = obj.world.world_transform(obj.owner).map_or(Vec3::ZERO, |m| m.w_axis.truncate());
            let mut pieces = Vec::new();
            for (_, (owner, body)) in obj.world.query::<(&Owner, &RigidBody)>().iter() {
                if body.kind != crate::BodyKind::Dynamic {
                    continue;
                }
                let mut at = Some(owner.0);
                let mut depth = 0;
                while let Some(entity) = at {
                    if entity == obj.owner {
                        pieces.push(owner.0);
                        break;
                    }
                    depth += 1;
                    if depth > 64 {
                        break;
                    }
                    at = obj.world.get::<Parent>(entity).and_then(|parent| parent.0);
                }
            }
            pieces.sort_unstable_by_key(|entity| obj.order_key(obj.world, *entity));
            for (index, entity) in pieces.into_iter().enumerate() {
                let (Some(id), Some(position)) = (obj.world.object_id(entity), obj.world.world_transform(entity).map(|m| m.w_axis.truncate())) else {
                    continue;
                };
                let impulse = burst_direction(position - origin, self.upward, index) * self.impulse;
                obj.commands.push(ComponentCommand::AddImpulse { id, impulse });
            }
        }
        self.age += obj.dt;
        if self.lifetime > 0.0 && self.age >= self.lifetime {
            obj.destroy_self();
        }
    }
}

impl TypedComponent for DebrisBurst {
    const NAME: &'static str = "Debris Burst";
}

pub(crate) fn register_systems(registry: &mut crate::ComponentRegistry) {
    registry.register_ecs::<Breakable>();
    registry.register_ecs::<DebrisBurst>();
    registry.register_system_for::<Breakable>("zestengine::breakable", Stage::Gameplay, |world, ctx| run_ordered(world, ctx, Breakable::update_native));
    registry.register_system_for::<Breakable>("zestengine::breakable_contacts", Stage::Events, |world, ctx| {
        let events = ctx.services.scene_events;
        run_ordered(world, ctx, |breakable: &mut Breakable, obj| {
            for event in events {
                if let SceneEvent::Contact { object, other, trigger: false, phase: ContactPhase::Enter, .. } = event
                    && *object == obj.self_id()
                {
                    breakable.contact_native(obj, *other);
                }
            }
        });
    });
    registry.register_system_for::<DebrisBurst>("zestengine::debris_burst", Stage::Gameplay, |world, ctx| run_ordered(world, ctx, DebrisBurst::update_native));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::{SceneWorld, SystemContext};
    use crate::{AnyBox, ComponentRegistry, Health, InputState};

    fn run(scene: &mut SceneWorld, stage: Stage, dt: f32) -> Vec<ComponentCommand> {
        scene.propagate().unwrap();
        let input = InputState::default();
        let mut commands = Vec::new();
        let mut ctx = SystemContext::new(&input, &mut commands);
        ctx.dt = dt;
        ComponentRegistry::with_builtins().run_systems(stage, scene, &mut ctx);
        commands
    }

    fn broken_events(commands: &[ComponentCommand]) -> Vec<BrokenEvent> {
        commands
            .iter()
            .filter_map(|c| match c {
                ComponentCommand::Emit(AnyBox(event)) => event.downcast_ref::<BrokenEvent>().copied(),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_dead_object_is_swapped_for_its_broken_prefab_once() {
        let mut scene = SceneWorld::new();
        let vase = scene.spawn("Vase");
        scene.get_mut::<crate::Transform>(vase).unwrap().translation = Vec3::new(3.0, 1.0, -2.0);
        scene.insert(vase, Health { current: 0.0, max: 10.0, started: true, ..Default::default() }).unwrap();
        scene.insert(vase, Breakable { broken_prefab: "prefabs/vase_broken.prefab".into(), ..Default::default() }).unwrap();
        let id = scene.object_id(vase).unwrap();
        let commands = run(&mut scene, Stage::Gameplay, 0.016);
        let spawned: Vec<_> = commands
            .iter()
            .filter_map(|c| match c {
                ComponentCommand::SpawnPrefab { path, root_id, translation, .. } => Some((path.as_str(), *root_id, *translation)),
                _ => None,
            })
            .collect();
        assert_eq!(spawned.len(), 1);
        assert_eq!((spawned[0].0, spawned[0].2), ("prefabs/vase_broken.prefab", Vec3::new(3.0, 1.0, -2.0)));
        assert!(commands.iter().any(|c| matches!(c, ComponentCommand::Destroy(target) if *target == id)), "the intact object is removed");
        let events = broken_events(&commands);
        assert_eq!(events, vec![BrokenEvent { object: id, debris: Some(spawned[0].1), position: Vec3::new(3.0, 1.0, -2.0) }]);
        assert!(scene.get::<Breakable>(vase).unwrap().broken);
        let again = run(&mut scene, Stage::Gameplay, 0.016);
        assert!(broken_events(&again).is_empty() && !again.iter().any(|c| matches!(c, ComponentCommand::SpawnPrefab { .. })), "it breaks once");
    }

    #[test]
    fn the_break_action_and_keep_object_disable_instead_of_destroying() {
        let mut scene = SceneWorld::new();
        let crate_object = scene.spawn("Crate");
        scene.insert(crate_object, Breakable { keep_object: true, break_with_health: false, ..Default::default() }).unwrap();
        let id = scene.object_id(crate_object).unwrap();
        assert!(run(&mut scene, Stage::Gameplay, 0.016).iter().all(|c| !matches!(c, ComponentCommand::Emit(_))), "nothing breaks on its own");
        {
            let mut commands = Vec::new();
            let mut ctx = ActionCtx { self_id: id, instigator: None, commands: &mut commands, reader: None };
            let mut breakable = scene.get_mut::<Breakable>(crate_object).unwrap();
            TypedComponent::invoke_action(&mut *breakable, &mut ctx, "Break", &[]).unwrap();
            assert_eq!(TypedComponent::read_property(&*breakable, "broken"), Some(ArgValue::Bool(false)), "it breaks on the next gameplay step");
        }
        let commands = run(&mut scene, Stage::Gameplay, 0.016);
        assert_eq!(broken_events(&commands), vec![BrokenEvent { object: id, debris: None, position: Vec3::ZERO }]);
        assert!(commands.iter().any(|c| matches!(c, ComponentCommand::SetActive { .. })), "kept objects are switched off: {}", commands.len());
        assert!(!commands.iter().any(|c| matches!(c, ComponentCommand::Destroy(_) | ComponentCommand::SpawnPrefab { .. })));
        assert_eq!(TypedComponent::read_property(&*scene.get::<Breakable>(crate_object).unwrap(), "broken"), Some(ArgValue::Bool(true)));
    }

    #[test]
    fn impacts_are_measured_from_the_change_in_speed_and_the_closing_speed() {
        let falling = Breakable { impact_speed: 4.0, last_velocity: Vec3::new(0.0, -6.0, 0.0), ..Default::default() };
        assert!((falling.impact(Vec3::ZERO, Vec3::ZERO) - 6.0).abs() < 1e-5, "a fall that stops dead");
        assert!(falling.breaks_at(falling.impact(Vec3::ZERO, Vec3::ZERO)));
        let resting = Breakable { impact_speed: 4.0, ..Default::default() };
        assert!((resting.impact(Vec3::ZERO, Vec3::new(5.0, 0.0, 0.0)) - 5.0).abs() < 1e-5, "struck by something moving");
        assert!((resting.impact(Vec3::new(3.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)) - 3.0).abs() < 1e-5, "knocked into motion");
        assert!(!resting.breaks_at(3.0) && resting.breaks_at(4.0));
        assert!(!Breakable { impact_speed: 0.0, ..Default::default() }.breaks_at(100.0), "0 turns impact breaking off");
        assert!(!Breakable { impact_speed: 4.0, broken: true, ..Default::default() }.breaks_at(100.0));
    }

    #[test]
    fn a_debris_burst_pushes_its_dynamic_pieces_outward_once_and_cleans_up() {
        let mut scene = SceneWorld::new();
        let root = scene.spawn("Broken vase");
        let outsider = scene.spawn("Unrelated");
        scene.insert(outsider, RigidBody::default()).unwrap();
        let mut ids = Vec::new();
        for (i, offset) in [Vec3::X, -Vec3::X, Vec3::ZERO].into_iter().enumerate() {
            let piece = scene.spawn(&format!("Piece {i}"));
            scene.set_parent(piece, Some(root), false).unwrap();
            scene.get_mut::<crate::Transform>(piece).unwrap().translation = offset;
            scene.insert(piece, RigidBody::default()).unwrap();
            ids.push(scene.object_id(piece).unwrap());
        }
        let fixed = scene.spawn("Base");
        scene.set_parent(fixed, Some(root), false).unwrap();
        scene.insert(fixed, RigidBody { kind: crate::BodyKind::Fixed, ..Default::default() }).unwrap();
        scene.insert(root, DebrisBurst { impulse: 3.0, upward: 0.0, lifetime: 1.0, ..Default::default() }).unwrap();
        let root_id = scene.object_id(root).unwrap();
        let commands = run(&mut scene, Stage::Gameplay, 0.4);
        let pushes: Vec<(ObjectId, Vec3)> = commands
            .iter()
            .filter_map(|c| match c {
                ComponentCommand::AddImpulse { id, impulse } => Some((*id, *impulse)),
                _ => None,
            })
            .collect();
        assert_eq!(pushes.len(), 3, "only the dynamic pieces under the root are pushed");
        let of = |id: ObjectId| pushes.iter().find(|(p, _)| *p == id).unwrap().1;
        assert!((of(ids[0]) - Vec3::X * 3.0).length() < 1e-4 && (of(ids[1]) + Vec3::X * 3.0).length() < 1e-4);
        assert!((of(ids[2]).length() - 3.0).abs() < 1e-4 && of(ids[2]).y.abs() < 1e-4, "a piece at the centre still gets a direction");
        let second = run(&mut scene, Stage::Gameplay, 0.4);
        assert!(second.is_empty(), "the burst happens once");
        let third = run(&mut scene, Stage::Gameplay, 0.4);
        assert!(third.iter().any(|c| matches!(c, ComponentCommand::Destroy(target) if *target == root_id)), "the debris is removed after its lifetime");
        assert!((burst_direction(Vec3::X, 1.0, 0) - Vec3::new(1.0, 1.0, 0.0).normalize()).length() < 1e-5);
    }

    #[test]
    fn only_authored_fields_are_saved() {
        let text = crate::Component::save(&Breakable { broken_prefab: "p.prefab".into(), impact_speed: 5.0, broken: true, requested: true, ..Default::default() });
        assert!(!text.contains("requested") && !text.contains("last_velocity"));
        let back: Breakable = ron::from_str(&text).unwrap();
        assert!(!back.broken && back.impact_speed == 5.0 && back.break_with_health);
        let burst: DebrisBurst = ron::from_str("(impulse: 5.0)").unwrap();
        assert_eq!((burst.impulse, burst.lifetime, burst.fired), (5.0, 8.0, false));
    }
}
