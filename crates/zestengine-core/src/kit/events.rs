//! Event sources that run action bindings: trigger volumes and animation
//! events (see [`crate::actions`]).

use serde::{Deserialize, Serialize};

use crate::{ArgValue, EventActions, FireSource, ObjectId, TypedComponent, queue_actions};
use crate::ecs::{ContactPhase, Entity, Owner, SceneEvent, Stage, SystemContext, SystemWorld};

/// Runs actions when something enters or leaves a trigger collider on this
/// object (a `Box`/`Sphere`/`Capsule Collider` with `Is Trigger` on). The
/// object that entered is the instigator, and `other` in the event payload.
#[derive(Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct TriggerEvents {
    /// Only objects that carry this component (by type name, e.g. "Pawn")
    /// count. Empty = anything.
    pub require_component: String,
    /// Run `on_enter` / `on_exit` for every object, or only when the first
    /// object enters / the last one leaves.
    pub first_and_last_only: bool,
    /// Fire `on_enter` at most once.
    pub once: bool,
    /// Actions run when an object enters.
    pub on_enter: EventActions,
    /// Actions run when an object leaves.
    pub on_exit: EventActions,
    /// Objects currently inside (counted, not saved).
    #[serde(skip)]
    pub inside: u32,
    /// Whether the trigger has entered its active state (runtime).
    #[serde(skip)]
    pub entered: bool,
}


impl TriggerEvents {
    fn accepts(&self, world: &SystemWorld<'_>, context: &SystemContext<'_>, other: ObjectId, instance: Entity) -> bool {
        if self.require_component.is_empty() {
            return true;
        }
        let Some(owner) = world.entity(other) else { return false };
        let Some(index) = context.services.components else { return false };
        (0..index.component_count(owner).unwrap_or(0)).any(|position| index.component_slot(owner, position).is_some_and(|slot|
            slot.name() == self.require_component && !(slot.entity() == instance && slot.type_id() == std::any::TypeId::of::<Self>())))
    }

    fn run(&self, list: &EventActions, context: &mut SystemContext<'_>, me: ObjectId, other: ObjectId) {
        let payload = [("other", ArgValue::Object(other.into())), ("object", ArgValue::Object(me.into()))];
        queue_actions(list, &FireSource { source: me, instigator: Some(other), payload: &payload }, context.commands);
    }
}

impl TypedComponent for TriggerEvents {
    const NAME: &'static str = "Trigger Events";
}

impl TriggerEvents {
    fn enter(&mut self, world: &SystemWorld<'_>, context: &mut SystemContext<'_>, instance: Entity, me: ObjectId, other: ObjectId) {
        if !self.accepts(world, context, other, instance) {
            return;
        }
        self.inside += 1;
        if (self.once && self.entered) || (self.first_and_last_only && self.inside > 1) {
            return;
        }
        self.entered = true;
        self.run(&self.on_enter, context, me, other);
    }

    fn exit(&mut self, world: &SystemWorld<'_>, context: &mut SystemContext<'_>, instance: Entity, me: ObjectId, other: ObjectId) {
        if !self.accepts(world, context, other, instance) || self.inside == 0 {
            return;
        }
        self.inside -= 1;
        if self.first_and_last_only && self.inside > 0 {
            return;
        }
        self.run(&self.on_exit, context, me, other);
    }
}

/// One animation event name and what it runs.
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AnimationEventBinding {
    /// The marker name set on the clip (e.g. "Footstep").
    pub event: String,
    /// The actions this event runs.
    pub actions: EventActions,
}

/// Runs actions when the Animator on this object passes a named animation
/// event marker.
#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AnimationEventBindings {
    /// The bindings, in declaration order.
    pub events: Vec<AnimationEventBinding>,
}

impl TypedComponent for AnimationEventBindings {
    const NAME: &'static str = "Animation Event Bindings";
}

impl AnimationEventBindings {
    fn fire(&self, context: &mut SystemContext<'_>, me: ObjectId, name: &str) {
        let payload = [("name", ArgValue::String(name.to_owned())), ("object", ArgValue::Object(me.into()))];
        for binding in self.events.iter().filter(|b| b.event == name) {
            queue_actions(&binding.actions, &FireSource { source: me, instigator: None, payload: &payload }, context.commands);
        }
    }
}

pub(crate) fn register_systems(registry: &mut crate::ComponentRegistry) {
    registry.register_ecs::<TriggerEvents>();
    registry.register_ecs::<AnimationEventBindings>();
    registry.register_system_for::<TriggerEvents>("zestengine::triggers", Stage::Events, triggers);
    registry.register_system_for::<AnimationEventBindings>("zestengine::animation_bindings", Stage::Events, animation);
}

fn order<T: crate::Component>(world: &SystemWorld<'_>, context: &SystemContext<'_>, owner: Entity, instance: Entity) -> ((usize, u128), usize) {
    let slot = context.services.components.and_then(|index|
        (0..index.component_count(owner)?).find(|&position| index.component_slot(owner, position).is_some_and(|slot|
            slot.entity() == instance && slot.type_id() == std::any::TypeId::of::<T>())))
        .unwrap_or(usize::MAX);
    (context.order_key(world, owner), slot)
}

fn triggers(world: &SystemWorld<'_>, context: &mut SystemContext<'_>) {
    let mut query = world.query::<(&Owner, &mut TriggerEvents)>();
    let mut actors: Vec<_> = query.iter().filter(|(_, (owner, _))| context.is_active(world, owner.0)).collect();
    actors.sort_unstable_by_key(|(instance, (owner, _))| order::<TriggerEvents>(world, context, owner.0, *instance));
    let events = context.services.scene_events;
    for (instance, (owner, trigger)) in actors {
        let Some(me) = world.object_id(owner.0) else { continue };
        for event in events {
            if let SceneEvent::Contact { object, other, trigger: true, phase, .. } = event {
                if *object != me { continue; }
                match phase {
                    ContactPhase::Enter => trigger.enter(world, context, instance, me, *other),
                    ContactPhase::Exit => trigger.exit(world, context, instance, me, *other),
                    ContactPhase::Stay => {},
                }
            }
        }
    }
}

fn animation(world: &SystemWorld<'_>, context: &mut SystemContext<'_>) {
    let mut query = world.query::<(&Owner, &AnimationEventBindings)>();
    let mut actors: Vec<_> = query.iter().filter(|(_, (owner, _))| context.is_active(world, owner.0)).collect();
    actors.sort_unstable_by_key(|(instance, (owner, _))| order::<AnimationEventBindings>(world, context, owner.0, *instance));
    let events = context.services.scene_events;
    for (_, (owner, bindings)) in actors {
        let Some(me) = world.object_id(owner.0) else { continue };
        for event in events {
            if let SceneEvent::Animation { object, name } = event
                && *object == me { bindings.fire(context, me, name); }
        }
    }
}
