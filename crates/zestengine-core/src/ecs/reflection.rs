//! Native reflection for event bindings and actions. Slot order is authoring
//! metadata; all values stay in their typed archetypes throughout dispatch.

use std::any::TypeId;
use std::collections::HashMap;
use crate::{ActionCtx, ArgValue, Component, ComponentCommand, InstanceSel, ObjectId, PropertyReader};
use super::{ComponentSlot, Entity, Name, Owner, SystemContext, SystemWorld};

/// Disjoint values lent by a checked query that already owns their column.
trait ComponentLoans {
    fn get(&self, entity: Entity) -> Option<&dyn Component>;
}
impl ComponentLoans for HashMap<Entity, &dyn Component> {
    fn get(&self, entity: Entity) -> Option<&dyn Component> { HashMap::get(self, &entity).copied() }
}

struct QueryLoans<'a, 'q, T> {
    before: &'a [(Entity, (&'q Owner, &'q mut T))],
    after: &'a [(Entity, (&'q Owner, &'q mut T))],
    current: usize,
    positions: &'a HashMap<Entity, usize>,
}
impl<T: Component> ComponentLoans for QueryLoans<'_, '_, T> {
    fn get(&self, entity: Entity) -> Option<&dyn Component> {
        let index = *self.positions.get(&entity)?;
        let (_, (_, value)) = if index < self.current { self.before.get(index)? }
            else { self.after.get(index.checked_sub(self.current + 1)?)? };
        Some(&**value)
    }
}

struct MetadataOnly;
impl ComponentIndex for MetadataOnly {
    fn component_count(&self, _: Entity) -> Option<usize> { Some(0) }
    fn component_slot(&self, _: Entity, _: usize) -> Option<ComponentSlot> { None }
}

/// Supplies authoring handle metadata, without acquiring component payload views.
/// None distinguishes a missing/hidden object from an object with zero components.
pub trait ComponentIndex {
    fn component_count(&self, owner: Entity) -> Option<usize>;
    fn component_slot(&self, owner: Entity, slot: usize) -> Option<ComponentSlot>;
}

#[derive(Clone, Copy)]
pub struct NativeActionCall<'a> {
    pub object: ObjectId,
    pub slot: u32,
    pub component: &'a str,
    pub action: &'a str,
    pub args: &'a [ArgValue],
    pub instigator: Option<ObjectId>,
}

/// Reflection over the actual native world. During an action the executing
/// instance is excluded, and other instances of its type are lent by that
/// action's query so same-archetype property reads remain borrow-safe.
pub struct NativeProperties<'a, 'w> {
    pub world: &'a SystemWorld<'w>,
    pub index: &'a dyn ComponentIndex,
    excluded: Option<(Entity, TypeId)>,
    lent: Option<(TypeId, &'a dyn ComponentLoans)>,
}

impl<'a, 'w> NativeProperties<'a, 'w> {
    pub fn new(world: &'a SystemWorld<'w>, index: &'a dyn ComponentIndex) -> Self {
        Self { world, index, excluded: None, lent: None }
    }
    pub fn object(&self, id: ObjectId) -> Option<Entity> {
        let entity = self.world.entity(id)?;
        self.index.component_count(entity).map(|_| entity)
    }
    pub fn with_component<R>(&self, slot: ComponentSlot, read: impl FnOnce(&dyn Component) -> R) -> Option<R> {
        if self.excluded == Some((slot.entity, slot.type_id())) { return None; }
        if let Some((_, values)) = self.lent.filter(|(ty, _)| *ty == slot.type_id()) {
            return values.get(slot.entity).map(read);
        }
        let value = slot.read_in(self.world)?;
        Some(read(&*value))
    }
    pub fn component(&self, owner: Entity, name: &str, sel: InstanceSel<'_>) -> Option<(usize, ComponentSlot)> {
        let mut seen = 0;
        let mut fallback = None;
        for position in 0..self.index.component_count(owner)? {
            let slot = self.index.component_slot(owner, position)?;
            if slot.name != name || self.excluded == Some((slot.entity, slot.type_id())) { continue; }
            if sel.label.is_empty() && seen == sel.index { return Some((position, slot)); }
            if !sel.label.is_empty() {
                if self.with_component(slot, |value| value.instance_label().as_deref() == Some(sel.label)) == Some(true) { return Some((position, slot)); }
                if seen == sel.index { fallback = Some((position, slot)); }
            }
            seen += 1;
        }
        fallback
    }
}

/// Update typed instances while lending all other instances to reflection.
/// The positions table contains identities only; values remain borrowed in
/// their native columns, with no per-instance copies or unchecked references.
pub(crate) fn run_with_properties<T: Component>(world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>, mut update: impl FnMut(&mut T, Entity, &mut SystemContext<'_>, &NativeProperties<'_, '_>)) {
    let mut query = world.query::<(&Owner, &mut T)>();
    let mut actors: Vec<_> = query.iter().filter(|(_, (owner, _))| ctx.is_visible(world, owner.0)).collect();
    actors.sort_unstable_by_key(|(instance, (owner, _))| ctx.instance_order::<T>(world, owner.0, *instance));
    let positions: HashMap<_, _> = actors.iter().enumerate().map(|(index, (entity, _))| (*entity, index)).collect();
    for current in 0..actors.len() {
        let (before, rest) = actors.split_at_mut(current);
        let ((entity, (owner, value)), after) = rest.split_first_mut().unwrap();
        if !ctx.is_active(world, owner.0) { continue; }
        let loans = QueryLoans { before, after, current, positions: &positions };
        let reader = NativeProperties { world, index: ctx.services.components.unwrap_or(&MetadataOnly),
            excluded: Some((*entity, TypeId::of::<T>())), lent: Some((TypeId::of::<T>(), &loans)) };
        update(value, owner.0, ctx, &reader);
    }
}

impl PropertyReader for NativeProperties<'_, '_> {
    fn read(&self, object: ObjectId, component: &str, property: &str) -> Result<ArgValue, String> {
        let owner = self.object(object).ok_or("object not found")?;
        if component.is_empty() || component == crate::OBJECT_COMPONENT {
            let name = self.world.get::<Name>(owner).ok_or("object has no name")?;
            let position = self.world.world_transform(owner).ok_or("object has no position")?.w_axis.truncate();
            return crate::object_property(object, &name.0, self.world.current_active(owner), position, property);
        }
        let (_, slot) = self.component(owner, component, InstanceSel::FIRST)
            .ok_or_else(|| format!("object has no {component} component (a component cannot read itself during its own action)"))?;
        self.with_component(slot, |value| value.read_property(property)).flatten()
            .ok_or_else(|| format!("{component} has no property '{property}'"))
    }
}

pub fn invoke_action_in(world: &SystemWorld<'_>, index: &dyn ComponentIndex, mut call: NativeActionCall<'_>, out: &mut Vec<ComponentCommand>) -> Result<(), String> {
    let reader = NativeProperties::new(world, index);
    let owner = reader.object(call.object).ok_or("target object is gone")?;
    let slot = match index.component_slot(owner, call.slot as usize) {
        Some(slot) if slot.name == call.component => slot,
        _ => {
            let (position, slot) = reader.component(owner, call.component, InstanceSel::FIRST)
                .ok_or_else(|| format!("target has no {} component", call.component))?;
            call.slot = position as u32;
            slot
        }
    };
    slot.invoke_in(world, index, call, out)
}

pub(super) fn invoke_typed<T: Component>(world: &SystemWorld<'_>, index: &dyn ComponentIndex, slot: ComponentSlot, call: NativeActionCall<'_>, out: &mut Vec<ComponentCommand>) -> Result<(), String> {
    // hecs borrows columns. Lend the disjoint other instances from the same
    // checked query instead of attempting to reborrow that column, copying
    // payloads, or creating unchecked transient component references.
    let mut query = world.query::<&mut T>();
    let mut target = None;
    let mut values: HashMap<Entity, &dyn Component> = HashMap::new();
    for (entity, value) in query.iter() {
        if entity == slot.entity { target = Some(value); }
        else { values.insert(entity, value); }
    }
    let target = target.ok_or("target component is gone")?;
    let reader = NativeProperties { world, index, excluded: Some((slot.entity, slot.type_id())), lent: Some((TypeId::of::<T>(), &values)) };
    target.invoke_action(&mut ActionCtx { self_id: call.object, instigator: call.instigator, commands: out, reader: Some(&reader) }, call.action, call.args)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{ActionDesc, ActionType, ArgSource, EventBinding, FireSource, Health, InputState, LightComponent, ObjectRef, ParamDesc, PropertyDesc, TypedComponent};
    use crate::ecs::{ComponentSet, ContactPhase, Enabled, SceneEvent, SceneWorld, Stage, SystemContext};
    use serde::{Deserialize, Serialize};

    #[derive(Default)]
    struct Index(HashMap<Entity, ComponentSet>);
    impl ComponentIndex for Index {
        fn component_count(&self, owner: Entity) -> Option<usize> { self.0.get(&owner).map(ComponentSet::len) }
        fn component_slot(&self, owner: Entity, slot: usize) -> Option<ComponentSlot> { self.0.get(&owner)?.native_slot(slot) }
    }
    pub(crate) struct Fixture {
        pub(crate) scene: SceneWorld,
        index: Index,
        order: HashMap<ObjectId, usize>,
        registry: crate::ComponentRegistry,
    }
    impl Fixture {
        pub(crate) fn new() -> Self { Self { scene: SceneWorld::new(), index: Index::default(), order: HashMap::new(), registry: crate::ComponentRegistry::with_builtins() } }
        pub(crate) fn add(&mut self, name: &str, values: Vec<Box<dyn Component>>) -> Entity {
            let entity = self.scene.spawn(name);
            let mut values = ComponentSet::from_vec(values);
            values.attach(&self.scene, entity);
            self.index.0.insert(entity, values);
            self.order.insert(self.id(entity), self.order.len());
            entity
        }
        pub(crate) fn id(&self, entity: Entity) -> ObjectId { self.scene.object_id(entity).unwrap() }
        pub(crate) fn components(&self, entity: Entity) -> &ComponentSet { &self.index.0[&entity] }
        pub(crate) fn components_mut(&mut self, entity: Entity) -> &mut ComponentSet { self.index.0.get_mut(&entity).unwrap() }
        pub(crate) fn fire_logged(&self, binding: &crate::EventBinding, source: ObjectId, instigator: Option<ObjectId>, payload: &[(&'static str, ArgValue)]) -> Vec<ComponentCommand> {
            let mut out = Vec::new();
            self.scene.with_world(|world| binding.fire_native(&NativeProperties::new(world, &self.index), &FireSource { source, instigator, payload }, &mut out));
            out
        }
        pub(crate) fn fire(&self, binding: &EventBinding, source: ObjectId, instigator: Option<ObjectId>, payload: &[(&'static str, ArgValue)]) -> Result<Vec<ComponentCommand>, String> {
            let mut out = Vec::new();
            self.scene.with_world(|world| binding.try_fire_native(&NativeProperties::new(world, &self.index), &FireSource { source, instigator, payload }, &mut out))?;
            Ok(out)
        }
        fn apply(&mut self, commands: Vec<ComponentCommand>) -> Result<(), String> {
            self.apply_output(commands).map(|_| ())
        }
        pub(crate) fn apply_output(&mut self, commands: Vec<ComponentCommand>) -> Result<Vec<ComponentCommand>, String> {
            let (errors, output) = self.apply_errors(commands);
            match errors.into_iter().next() { Some(error) => Err(error), None => Ok(output) }
        }
        pub(crate) fn apply_errors(&mut self, commands: Vec<ComponentCommand>) -> (Vec<String>, Vec<ComponentCommand>) {
            let mut resolved = Vec::new();
            self.scene.with_world(|world| {
                let reader = NativeProperties::new(world, &self.index);
                for command in commands {
                    match command {
                        ComponentCommand::FireActions { bindings, source, instigator, payload } => {
                            for binding in bindings { binding.fire_native(&reader, &FireSource { source, instigator, payload: &payload }, &mut resolved); }
                        }
                        command => resolved.push(command),
                    }
                }
            });
            let mut errors = Vec::new();
            let mut output = Vec::new();
            for command in resolved {
                let mut next = Vec::new();
                match command {
                    ComponentCommand::InvokeAction { id, slot, component, action, args, instigator } => {
                        if let Err(error) = self.scene.with_world(|world|
                            invoke_action_in(world, &self.index, NativeActionCall { object: id, slot, component, action, args: &args, instigator }, &mut next)) { errors.push(error); }
                    }
                    ComponentCommand::SetActive(id, active) => {
                        if let Some(entity) = self.scene.entity(id) { self.scene.get_mut::<Enabled>(entity).unwrap().0 = active; }
                        output.push(ComponentCommand::SetActive(id, active));
                    }
                    command => output.push(command),
                }
                let (more_errors, more_output) = self.apply_errors(next);
                errors.extend(more_errors); output.extend(more_output);
            }
            (errors, output)
        }
        pub(crate) fn frame(&mut self, input: &InputState, net: &crate::NetView, screen: &crate::ScreenInfo) -> Vec<ComponentCommand> {
            self.frame_dt(input, net, screen, 0.1)
        }
        pub(crate) fn frame_dt(&mut self, input: &InputState, net: &crate::NetView, screen: &crate::ScreenInfo, dt: f32) -> Vec<ComponentCommand> {
            self.stage(Stage::Gameplay, input, net, screen, dt, 0.0)
        }
        pub(crate) fn stage(&mut self, stage: Stage, input: &InputState, net: &crate::NetView, screen: &crate::ScreenInfo, dt: f32, time: f32) -> Vec<ComponentCommand> {
            self.scene.propagate().unwrap();
            let mut commands = Vec::new();
            let mut context = SystemContext::new(input, &mut commands);
            context.dt = dt;
            context.time = time;
            context.unscaled_dt = 0.1;
            context.services.net = Some(net);
            context.services.screen = Some(screen);
            context.services.components = Some(&self.index);
            context.services.order = Some(&self.order);
            self.registry.run_systems(stage, &self.scene, &mut context);
            commands
        }
        pub(crate) fn events(&mut self, events: &[SceneEvent]) -> Vec<ComponentCommand> {
            self.scene.propagate().unwrap();
            let input = InputState::default();
            let mut commands = Vec::new();
            let mut context = SystemContext::new(&input, &mut commands);
            context.services.components = Some(&self.index);
            context.services.order = Some(&self.order);
            context.services.scene_events = events;
            self.registry.run_systems(Stage::Events, &self.scene, &mut context);
            commands
        }
    }

    fn binding(target: ObjectId, component: &str, action: &str, args: Vec<ArgSource>) -> EventBinding {
        EventBinding { target: target.into(), component: component.into(), action: action.into(), args, ..Default::default() }
    }
    fn constant(value: f32) -> ArgSource { ArgSource::Const(ArgValue::Float(value)) }

    #[derive(Default, Serialize, Deserialize)]
    struct Probe { value: i64, other: ObjectRef, label: String }
    static PROBE_ACTIONS: &[ActionDesc] = &[ActionDesc::new("Read", "Read", &[], true), ActionDesc::new("Set", "Set", &[ParamDesc::float("value", 10.0)], true)];
    static PROBE_PROPERTIES: &[PropertyDesc] = &[PropertyDesc::new("value", ActionType::Int)];
    impl TypedComponent for Probe {
        const NAME: &'static str = "Native Action Probe";
        fn actions(&self) -> &'static [ActionDesc] { PROBE_ACTIONS }
        fn properties(&self) -> &'static [PropertyDesc] { PROBE_PROPERTIES }
        fn read_property(&self, property: &str) -> Option<ArgValue> { (property == "value").then_some(ArgValue::Int(self.value)) }
        fn instance_label(&self) -> Option<String> { Some(self.label.clone()) }
        fn invoke_action(&mut self, context: &mut ActionCtx<'_>, action: &str, args: &[ArgValue]) -> Result<(), String> {
            match action {
                "Read" => self.value = context.reader.ok_or("missing reader")?.read(self.other.id().unwrap_or(context.self_id), Self::NAME, "value")?.as_f32() as i64,
                "Set" => self.value = args[0].as_f32() as i64,
                _ => return Err(crate::actions::unknown_action(Self::NAME, action)),
            }
            Ok(())
        }
    }

    #[test]
    fn native_interaction_instances_follow_authoring_order_after_auxiliary_reordering() {
        let mut f = Fixture::new();
        let light = f.add("Light", vec![Box::new(LightComponent::default())]);
        let source = f.id(light);
        let actors: Vec<Box<dyn Component>> = [1.0, 2.0, 3.0].into_iter().map(|value| Box::new(crate::Interactable {
            requested: Some(source), cooldown: 0.0, on_interact: vec![binding(source, "Light", "SetIntensity", vec![constant(value)])], ..Default::default()
        }) as Box<dyn Component>).collect();
        let lever = f.add("Lever", actors);
        let commands = f.frame(&InputState::default(), &crate::NetView::default(), &crate::ScreenInfo::default());
        f.apply(commands).unwrap();
        assert_eq!(f.scene.get::<LightComponent>(light).unwrap().intensity, 3.0);
        f.index.0.get_mut(&lever).unwrap().swap(1, 2);
        for position in 0..3 { f.index.0[&lever].get_typed_at_mut::<crate::Interactable>(position).unwrap().request(source); }
        let commands = f.frame(&InputState::default(), &crate::NetView::default(), &crate::ScreenInfo::default());
        f.apply(commands).unwrap();
        assert_eq!(f.scene.get::<LightComponent>(light).unwrap().intensity, 2.0, "auxiliary archetype iteration cannot override authored instance order");
    }

    #[test]
    fn native_actions_read_same_type_in_the_same_archetype_and_exclude_only_the_current_instance() {
        let mut f = Fixture::new();
        let other = f.add("Other", vec![Box::new(Probe { value: 17, ..Default::default() })]);
        let actor = f.add("Actor", vec![Box::new(Probe { value: 1, other: f.id(other).into(), ..Default::default() })]);
        let command = f.fire(&binding(f.id(actor), Probe::NAME, "Read", vec![]), f.id(actor), None, &[]).unwrap();
        f.apply(command).unwrap();
        assert_eq!(f.scene.get::<Probe>(actor).unwrap().value, 17, "same-column reads use lent native references");
        f.index.0.get_mut(&actor).unwrap().push(Box::new(Probe { value: 23, label: "second".into(), ..Default::default() }));
        f.scene.get_mut::<Probe>(actor).unwrap().other = ObjectRef::NONE;
        f.apply(f.fire(&binding(f.id(actor), Probe::NAME, "Read", vec![]), f.id(actor), None, &[]).unwrap()).unwrap();
        assert_eq!(f.scene.get::<Probe>(actor).unwrap().value, 23, "own mutable instance is skipped but duplicate siblings are readable");
        let lone = binding(f.id(other), Probe::NAME, "Read", vec![]);
        assert!(f.apply(f.fire(&lone, f.id(other), None, &[]).unwrap()).unwrap_err().contains("cannot read itself"));
    }

    #[test]
    fn native_bindings_preserve_duplicate_labels_reordering_defaults_and_argument_sources() {
        let mut f = Fixture::new();
        let source = f.add("Source", vec![Box::new(Health { current: 37.0, ..Default::default() })]);
        let target = f.add("Target", vec![Box::new(Probe { value: 1, label: "first".into(), ..Default::default() }), Box::new(Probe { value: 2, label: "second".into(), ..Default::default() })]);
        let mut b = binding(f.id(target), Probe::NAME, "Set", vec![ArgSource::Property { object: ObjectRef::NONE, component: "Health".into(), property: "current".into() }]);
        b.instance_label = "second".into();
        for expected in [37, 41] {
            f.scene.get_mut::<Health>(source).unwrap().current = expected as f32;
            f.apply(f.fire(&b, f.id(source), None, &[]).unwrap()).unwrap();
            let values = &f.index.0[&target];
            let slot = if expected == 37 { 1 } else { 0 };
            assert_eq!(values.get_typed_at::<Probe>(slot).unwrap().value, expected);
            if expected == 37 { f.index.0.get_mut(&target).unwrap().swap(0, 1); }
        }
        b.args = vec![ArgSource::Instigator { component: "Health".into(), property: "current".into() }];
        f.apply(f.fire(&b, f.id(target), Some(f.id(source)), &[]).unwrap()).unwrap();
        b.args = vec![ArgSource::Payload("amount".into())];
        f.apply(f.fire(&b, f.id(target), None, &[("amount", ArgValue::Int(9))]).unwrap()).unwrap();
        assert_eq!(f.scene.get::<Probe>(target).unwrap().value, 9);
        b.args.clear();
        f.apply(f.fire(&b, f.id(target), None, &[]).unwrap()).unwrap();
        assert_eq!(f.scene.get::<Probe>(target).unwrap().value, 10, "missing args use declared defaults");
        b.args = vec![ArgSource::Payload("missing".into())];
        assert!(f.fire(&b, f.id(target), None, &[]).unwrap_err().contains("parameter 'value'"));
    }

    #[test]
    fn native_properties_use_live_parent_metadata_and_disabled_payloads_reject_actions() {
        let mut f = Fixture::new();
        let parent = f.add("Parent", vec![]);
        let child = f.add("Child", vec![Box::new(crate::Disabled(Box::new(Health { current: 21.0, ..Default::default() })))]);
        f.scene.set_parent(child, Some(parent), false).unwrap();
        f.scene.get_mut::<Enabled>(parent).unwrap().0 = false;
        f.scene.propagate().unwrap();
        let child_id = f.id(child);
        let toggle = binding(f.id(parent), "Object", "ToggleActive", vec![]);
        f.apply(f.fire(&toggle, f.id(child), None, &[]).unwrap()).unwrap();
        assert!(!f.scene.get::<super::super::Active>(child).unwrap().0, "derived snapshot has not propagated yet");
        f.scene.with_world(|world| {
            let reader = NativeProperties::new(world, &f.index);
            assert_eq!(reader.read(child_id, "", "active"), Ok(ArgValue::Bool(true)));
            assert_eq!(reader.read(child_id, "Health", "current"), Ok(ArgValue::Float(21.0)));
        });
        let damage = binding(f.id(child), "Health", "Damage", vec![constant(10.0)]);
        assert!(f.apply(f.fire(&damage, f.id(child), None, &[]).unwrap()).unwrap_err().contains("disabled"));
        let absent = binding(ObjectId::new(), "Health", "Damage", vec![]);
        assert!(f.fire(&absent, f.id(child), None, &[]).unwrap_err().contains("not found"));
    }

    #[test]
    fn trigger_events_fire_enter_and_exit_with_the_visitor() {
        for two_dimensional in [false, true] {
            let mut f = Fixture::new();
            let light = f.add("Light", vec![Box::new(LightComponent::default())]);
            let light_id = f.id(light);
            let set = |v| binding(f.id(light), "Light", "SetIntensity", vec![constant(v)]);
            let trigger = crate::TriggerEvents { require_component: "Health".into(), on_enter: vec![set(5.0)], on_exit: vec![set(0.5)], ..Default::default() };
            let zone = f.add("Zone", vec![Box::new(trigger)]);
            let player = f.add("Player", vec![Box::new(Health::default())]);
            let krate = f.add("Crate", vec![]);
            let enter = |object, other, phase| SceneEvent::Contact { object, other, trigger: true, phase, two_dimensional };
            assert!(f.events(&[enter(f.id(zone), f.id(krate), ContactPhase::Enter)]).is_empty());
            let commands = f.events(&[enter(f.id(zone), f.id(player), ContactPhase::Enter)]);
            f.apply(commands).unwrap();
            assert_eq!(f.scene.get::<LightComponent>(light).unwrap().intensity, 5.0);
            let commands = f.events(&[enter(f.id(zone), f.id(player), ContactPhase::Exit)]);
            f.apply(commands).unwrap();
            assert_eq!(f.scene.get::<LightComponent>(light).unwrap().intensity, 0.5);
            f.index.0.get_mut(&zone).unwrap().replace_all(vec![Box::new(crate::TriggerEvents { once: true, on_enter: vec![binding(light_id, "Light", "SetIntensity", vec![constant(9.0)])], ..Default::default() })]);
            let commands = f.events(&[enter(f.id(zone), f.id(player), ContactPhase::Enter), enter(f.id(zone), f.id(player), ContactPhase::Enter)]);
            assert_eq!(commands.len(), 1);
            f.apply(commands).unwrap();
            f.index.0.get_mut(&zone).unwrap().set_enabled(0, false);
            assert!(f.events(&[enter(f.id(zone), f.id(player), ContactPhase::Enter)]).is_empty());
            f.index.0.get_mut(&zone).unwrap().set_enabled(0, true);
            assert!(f.events(&[enter(f.id(zone), f.id(player), ContactPhase::Enter)]).is_empty(), "reenabling retains the once-only state");
            f.index.0.get_mut(&zone).unwrap().replace_all(vec![Box::new(crate::TriggerEvents { first_and_last_only: true,
                on_enter: vec![binding(light_id, "Light", "SetIntensity", vec![constant(1.0)])], on_exit: vec![binding(light_id, "Light", "SetIntensity", vec![constant(2.0)])], ..Default::default() })]);
            let mut commands = f.events(&[enter(f.id(zone), f.id(player), ContactPhase::Enter), enter(f.id(zone), f.id(krate), ContactPhase::Enter), enter(f.id(zone), f.id(player), ContactPhase::Exit)]);
            assert_eq!(commands.len(), 1);
            commands.extend(f.events(&[enter(f.id(zone), f.id(krate), ContactPhase::Exit)]));
            assert_eq!(commands.len(), 2);
            f.apply(commands).unwrap();
            assert_eq!(f.scene.get::<LightComponent>(light).unwrap().intensity, 2.0);
        }
    }

    #[test]
    fn animation_event_bindings_match_by_name() {
        let mut f = Fixture::new();
        let light = f.add("Light", vec![Box::new(LightComponent::default())]);
        let rig = f.add("Rig", vec![Box::new(crate::AnimationEventBindings { events: vec![crate::AnimationEventBinding { event: "Footstep".into(),
            actions: vec![binding(f.id(light), "Light", "SetIntensity", vec![constant(2.0)])] }] })]);
        assert!(f.events(&[SceneEvent::Animation { object: f.id(rig), name: "Other".into() }]).is_empty());
        let commands = f.events(&[SceneEvent::Animation { object: f.id(rig), name: "Footstep".into() }]);
        f.apply(commands).unwrap();
        assert_eq!(f.scene.get::<LightComponent>(light).unwrap().intensity, 2.0);
    }
}
