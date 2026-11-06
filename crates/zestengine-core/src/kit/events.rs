//! Event sources that run action bindings: trigger volumes and animation
//! events (see [`crate::actions`]).

use serde::{Deserialize, Serialize};

use crate::{ArgValue, ComponentCtx, EventActions, FireSource, InstanceSel, ObjectId, TypedComponent, fire_all};

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
    fn accepts(&self, ctx: &ComponentCtx, other: ObjectId) -> bool {
        if self.require_component.is_empty() {
            return true;
        }
        ctx.index_of(other)
            .is_some_and(|i| ctx.lookup.component_named(i, &self.require_component, 0, InstanceSel::FIRST).is_some())
    }

    fn run(&self, list: &EventActions, ctx: &mut ComponentCtx, other: ObjectId) {
        let me = ctx.self_id();
        let payload = [("other", ArgValue::Object(other.into())), ("object", ArgValue::Object(me.into()))];
        fire_all(list, ctx, &FireSource { source: me, instigator: Some(other), payload: &payload });
    }
}

impl TypedComponent for TriggerEvents {
    const NAME: &'static str = "Trigger Events";

    fn on_trigger_enter(&mut self, ctx: &mut ComponentCtx, other: ObjectId) {
        if !self.accepts(ctx, other) {
            return;
        }
        self.inside += 1;
        if (self.once && self.entered) || (self.first_and_last_only && self.inside > 1) {
            return;
        }
        self.entered = true;
        self.run(&self.on_enter, ctx, other);
    }

    fn on_trigger_exit(&mut self, ctx: &mut ComponentCtx, other: ObjectId) {
        if !self.accepts(ctx, other) || self.inside == 0 {
            return;
        }
        self.inside -= 1;
        if self.first_and_last_only && self.inside > 0 {
            return;
        }
        self.run(&self.on_exit, ctx, other);
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

    fn on_animation_event(&mut self, ctx: &mut ComponentCtx, name: &str) {
        let me = ctx.self_id();
        let payload = [("name", ArgValue::String(name.to_owned())), ("object", ArgValue::Object(me.into()))];
        for binding in self.events.iter().filter(|b| b.event == name) {
            fire_all(&binding.actions, ctx, &FireSource { source: me, instigator: None, payload: &payload });
        }
    }
}
