//! Typed disabled payloads. The component value is inline, never a trait-object box.

use std::any::Any;
use std::ops::{Deref, DerefMut};

use crate::{ActionCtx, ActionDesc, ArgValue, Component, PropertyDesc};
use super::ComponentType;

/// A disabled `T`, stored densely in its own typed ECS column. Queries for `T`
/// see enabled instances; query this type explicitly to inspect disabled data.
/// Authoring slots preserve their order and expose `T` through component_inner.
#[repr(transparent)]
pub struct DisabledComponent<T: Component> {
    value: T,
}

impl<T: Component> DisabledComponent<T> {
    pub(crate) fn new(value: T) -> Self { Self { value } }
    pub fn value(&self) -> &T { &self.value }
    pub fn value_mut(&mut self) -> &mut T { &mut self.value }
    pub fn into_value(self) -> T { self.value }
}
impl<T: Component> Deref for DisabledComponent<T> {
    type Target = T;
    fn deref(&self) -> &T { &self.value }
}
impl<T: Component> DerefMut for DisabledComponent<T> {
    fn deref_mut(&mut self) -> &mut T { &mut self.value }
}

impl<T: Component> Component for DisabledComponent<T> {
    fn into_any(self: Box<Self>) -> Box<dyn Any> { self }
    fn into_storage(self: Box<Self>) -> Box<dyn Component> { self }
    fn into_enabled(self: Box<Self>) -> Box<dyn Component> { Box::new(self.value) }
    fn into_disabled(self: Box<Self>) -> Box<dyn Component> { self }
    fn inner_data(&self) -> &dyn Component { self.value.inner_data() }
    fn inner_data_mut(&mut self) -> &mut dyn Component { self.value.inner_data_mut() }
    fn is_enabled(&self) -> bool { false }
    fn storage_type(&self) -> ComponentType { ComponentType::of::<Self>() }
    fn type_name(&self) -> &'static str { self.value.type_name() }
    fn save(&self) -> String { self.value.save() }
    fn actions(&self) -> &'static [ActionDesc] { self.value.actions() }
    fn invoke_action(&mut self, _ctx: &mut ActionCtx, _name: &str, _args: &[ArgValue]) -> Result<(), String> {
        Err(format!("{} is disabled", self.value.type_name()))
    }
    fn properties(&self) -> &'static [PropertyDesc] { self.value.properties() }
    fn read_property(&self, name: &str) -> Option<ArgValue> { self.value.read_property(name) }
    fn instance_label(&self) -> Option<String> { self.value.instance_label() }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Disabled, TypedComponent, component_enabled, component_inner, component_inner_data_mut};
    use crate::ecs::{ComponentSet, Owner, SceneWorld};
    use serde::{Serialize, Deserialize};

    #[derive(Default, Serialize, Deserialize)]
    struct Probe {
        id: u32,
        data: Vec<u8>,
        #[serde(skip)]
        live: u32,
    }
    impl TypedComponent for Probe { const NAME: &'static str = "Disabled Storage Probe"; }
    fn probe(id: u32) -> Box<dyn Component> { Box::new(Probe { id, data: vec![1, 2, 3], live: 40 }) }

    #[test]
    fn disabled_payload_is_inline_and_native_queries_and_editor_share_unsaved_state() {
        let mut scene = SceneWorld::new();
        let owner = scene.spawn("Object");
        let mut components = ComponentSet::new();
        components.push(Box::new(Disabled(probe(1))));
        components.attach(&scene, owner);
        let allocation = components.get_including_disabled::<Probe>().unwrap().data.as_ptr();
        assert_eq!(std::mem::size_of::<DisabledComponent<Probe>>(), std::mem::size_of::<Probe>());
        assert!(components.get_typed::<Probe>().is_none());
        scene.with_world(|world| {
            assert_eq!(world.query::<&Disabled>().iter().count(), 0, "construction boxes cannot remain in ECS");
            assert_eq!(world.query::<&Probe>().iter().count(), 0);
            for (_, (owner_data, value)) in world.query::<(&Owner, &mut DisabledComponent<Probe>)>().iter() {
                assert_eq!(owner_data.0, owner);
                value.live += 2;
                assert_eq!(value.data.as_ptr(), allocation);
            }
        });
        assert_eq!(components.get_including_disabled::<Probe>().unwrap().live, 42);
        components.get_including_disabled_mut::<Probe>().unwrap().id = 7;
        for _ in 0..8 {
            assert!(components.set_enabled(0, true));
            let value = components.get_typed::<Probe>().unwrap();
            assert_eq!((value.id, value.live), (7, 42));
            assert_eq!(value.data.as_ptr(), allocation, "toggle moves the payload without cloning buffers");
            drop(value);
            assert!(components.set_enabled(0, false));
            assert!(!components.set_enabled(0, false));
        }
        let mut next = SceneWorld::new();
        let next_owner = next.spawn("Next");
        components.detach();
        components.attach(&next, next_owner);
        let value = components.get_including_disabled::<Probe>().unwrap();
        assert_eq!((value.id, value.live, value.data.as_ptr()), (7, 42, allocation));
        drop(value);
        next.with_world(|world| assert_eq!(world.get::<DisabledComponent<Probe>>(next_owner).unwrap().live, 42));
    }

    #[test]
    fn duplicate_enable_edits_preserve_order_and_the_primary_enabled_and_disabled_values() {
        let mut scene = SceneWorld::new();
        let owner = scene.spawn("Object");
        let mut components = ComponentSet::from_vec(vec![Box::new(Disabled(probe(1))), probe(2), Box::new(Disabled(probe(3))), probe(4)]);
        components.attach(&scene, owner);
        let values = |components: &ComponentSet| components.iter().map(|value| {
            (component_enabled(value.as_ref()), component_inner(value.as_ref()).as_any().downcast_ref::<Probe>().unwrap().id)
        }).collect::<Vec<_>>();
        assert_eq!(values(&components), [(false, 1), (true, 2), (false, 3), (true, 4)]);
        scene.with_world(|world| {
            assert_eq!(world.query::<&Probe>().iter().count(), 2);
            assert_eq!(world.query::<&DisabledComponent<Probe>>().iter().count(), 2);
            assert_eq!(world.get::<Probe>(owner).unwrap().id, 2);
            assert_eq!(world.get::<DisabledComponent<Probe>>(owner).unwrap().id, 1);
        });
        components.set_enabled(0, true);
        components.set_enabled(1, false);
        assert_eq!(values(&components), [(true, 1), (false, 2), (false, 3), (true, 4)]);
        scene.with_world(|world| {
            assert_eq!(world.get::<Probe>(owner).unwrap().id, 1);
            assert_eq!(world.get::<DisabledComponent<Probe>>(owner).unwrap().id, 2);
        });
        {
            let mut value = components.get_mut(1).unwrap();
            component_inner_data_mut(value.as_mut()).as_any_mut().downcast_mut::<Probe>().unwrap().live = 9;
        }
        components.remove(1);
        assert_eq!(values(&components), [(true, 1), (false, 3), (true, 4)]);
        scene.with_world(|world| assert_eq!(world.get::<DisabledComponent<Probe>>(owner).unwrap().id, 3));
    }

    #[test]
    fn native_disabled_insertion_removal_and_nested_compatibility_inputs_keep_values_typed() {
        let mut scene = SceneWorld::new();
        let owner = scene.spawn("Object");
        scene.insert_disabled(owner, Probe { id: 1, live: 2, ..Default::default() }).unwrap();
        let duplicate = scene.insert_disabled(owner, Probe { id: 3, live: 4, ..Default::default() }).unwrap();
        assert_ne!(duplicate, owner);
        assert_eq!(scene.remove_disabled::<Probe>(duplicate).unwrap().live, 4);
        assert!(scene.get::<Owner>(duplicate).is_none());
        assert_eq!(scene.remove_disabled::<Probe>(owner).unwrap().live, 2);
        assert!(scene.object_id(owner).is_some());
        let mut components = ComponentSet::new();
        components.push(Box::new(Disabled(Box::new(Disabled(probe(8))))));
        components.attach(&scene, owner);
        assert_eq!(components.get_including_disabled::<Probe>().unwrap().id, 8);
        scene.with_world(|world| {
            assert_eq!(world.query::<&Disabled>().iter().count(), 0);
            assert_eq!(world.query::<&DisabledComponent<Probe>>().iter().count(), 1);
        });
        assert!(components.set_enabled(0, true));
        assert_eq!(components.get_typed::<Probe>().unwrap().id, 8);
    }
}
