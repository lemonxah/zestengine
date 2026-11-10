//! Ordered authoring views of typed components that remain in the scene ECS.
//! The order is editor metadata; the component values live only in hecs.

use std::any::TypeId;
use std::sync::Arc;
use std::ops::RangeBounds;

use crate::Component;
use super::{ComponentMut, ComponentRef, Entity, Owner, SceneWorld, DataComponent, SystemWorld, Ref, RefMut};
use super::storage::{Retired, Storage};

/// Type-specific operations retain their originating plugin's code. No cast by
/// display name and no serialization is needed for storage, views or moves.
#[derive(Clone, Copy)]
pub struct ComponentType {
    id: TypeId,
    read: for<'a> fn(&'a Storage, Entity) -> Option<ComponentRef<'a, dyn Component>>,
    write: for<'a> fn(&'a Storage, Entity) -> Option<ComponentMut<'a, dyn Component>>,
    native_read: for<'a, 'w> fn(&'a SystemWorld<'w>, Entity) -> Option<Ref<'a, dyn Component>>,
    native_write: for<'a, 'w> fn(&'a SystemWorld<'w>, Entity) -> Option<RefMut<'a, dyn Component>>,
    native_invoke: fn(&SystemWorld<'_>, &dyn super::ComponentIndex, ComponentSlot, super::NativeActionCall<'_>, &mut Vec<crate::ComponentCommand>) -> Result<(), String>,
    put: fn(&mut hecs::World, Entity, Box<dyn Component>) -> Entity,
    take: fn(&mut hecs::World, Entity) -> Option<Box<dyn Component>>,
    drop: fn(&mut hecs::World, Entity),
    swap: fn(&mut hecs::World, Entity, Entity),
}

impl ComponentType {
    pub fn of<T: Component>() -> Self {
        fn take<T: Component>(world: &mut hecs::World, entity: Entity) -> Option<Box<dyn Component>> {
            let owner = world.get::<&Owner>(entity).ok()?.0;
            let value = world.remove_one::<T>(entity).ok()?;
            if entity != owner { world.despawn(entity).expect("component instance"); }
            Some(Box::new(value))
        }
        Self {
            id: TypeId::of::<T>(),
            read: |store, entity| Some(ComponentRef::map(store.read::<T>(entity)?, |value| value as &dyn Component)),
            write: |store, entity| Some(ComponentMut::map(store.write::<T>(entity)?, |value| value as &mut dyn Component)),
            native_read: |world, entity| Some(Ref::map(world.get::<T>(entity)?, |value| value as &dyn Component)),
            native_write: |world, entity| Some(RefMut::map(world.get_mut::<T>(entity)?, |value| value as &mut dyn Component)),
            native_invoke: super::reflection::invoke_typed::<T>,
            put: |world, owner, value| {
                let value = *value.into_any().downcast::<T>().unwrap_or_else(|_| panic!("component storage type mismatch"));
                if world.satisfies::<&T>(owner).expect("component owner") {
                    world.spawn((Owner(owner), value))
                } else {
                    world.insert_one(owner, value).expect("component owner");
                    owner
                }
            },
            take: take::<T>,
            drop: |world, entity| { drop(take::<T>(world, entity)); },
            swap: |world, a, b| {
                let [Ok(a), Ok(b)] = world.query_many_mut::<&mut T, 2>([a, b]) else { panic!("component instance"); };
                std::mem::swap(a, b);
            },
        }
    }
}

#[derive(Clone, Copy)]
struct Slot {
    name: &'static str,
    entity: Entity,
    ty: ComponentType,
}

/// Copyable authoring handle metadata, with native checked payload access. It
/// owns no component value and remains valid only within its indexed scene.
#[derive(Clone, Copy)]
pub struct ComponentSlot {
    pub(crate) name: &'static str,
    pub(crate) entity: Entity,
    pub(crate) ty: ComponentType,
}

impl ComponentSlot {
    pub fn name(self) -> &'static str { self.name }
    pub fn entity(self) -> Entity { self.entity }
    pub fn type_id(self) -> TypeId { self.ty.id }
    pub fn read_in<'a>(self, world: &'a SystemWorld<'_>) -> Option<Ref<'a, dyn Component>> { (self.ty.native_read)(world, self.entity) }
    /// Borrow the actual typed payload through its originating plugin's vtable.
    /// hecs enforces the same column exclusion as an ordinary mutable query.
    pub fn write_in<'a>(self, world: &'a SystemWorld<'_>) -> Option<RefMut<'a, dyn Component>> { (self.ty.native_write)(world, self.entity) }
    pub(crate) fn invoke_in(self, world: &SystemWorld<'_>, index: &dyn super::ComponentIndex, call: super::NativeActionCall<'_>, out: &mut Vec<crate::ComponentCommand>) -> Result<(), String> {
        (self.ty.native_invoke)(world, index, self, call, out)
    }
}

/// Construction owners are not indexed scene objects, even though their
/// saved identity is already kept in typed metadata.
struct Detached;

/// Engine-owned object data that is not an ordered gameplay component.
/// Values stay typed in hecs and move only when an object changes worlds.
pub trait ObjectMetadata: DataComponent {}

#[derive(Clone, Copy)]
struct MetadataType {
    id: TypeId,
    move_between: fn(&mut hecs::World, Entity, &mut hecs::World, Entity),
    move_within: fn(&mut hecs::World, Entity, Entity),
}

impl MetadataType {
    fn of<T: ObjectMetadata>() -> Self {
        Self {
            id: TypeId::of::<T>(),
            move_between: |source, from, target, to| {
                let value = source.remove_one::<T>(from).expect("source object metadata");
                target.insert_one(to, value).expect("target object metadata");
            },
            move_within: |world, from, to| {
                let value = world.remove_one::<T>(from).expect("source object metadata");
                world.insert_one(to, value).expect("target object metadata");
            },
        }
    }
}

/// An object's ordered component handles. Instances remain in shared typed ECS
/// storage across frames, editor edits and subsystem queries.
pub struct ComponentSet {
    storage: Arc<Storage>,
    owner: Entity,
    slots: Vec<Slot>,
    metadata: Vec<MetadataType>,
}

/// An owned entity/type lookup, allowing a component borrow to coexist with
/// edits to unrelated scene-tree metadata. A borrowed value still prevents
/// structural ECS changes until its view is released.
pub struct ComponentHandle {
    storage: Arc<Storage>,
    slot: Slot,
}

impl ComponentHandle {
    pub fn read(&self) -> Option<ComponentRef<'_, dyn Component>> { (self.slot.ty.read)(&self.storage, self.slot.entity) }
    pub fn write(&self) -> Option<ComponentMut<'_, dyn Component>> { (self.slot.ty.write)(&self.storage, self.slot.entity) }
    pub fn read_inner<T: Component>(&self) -> Option<ComponentRef<'_, T>> {
        ComponentRef::filter_map(self.read()?, |value| crate::component_inner(value).as_any().downcast_ref::<T>())
    }
}

impl Default for ComponentSet {
    fn default() -> Self {
        let storage = Arc::new(Storage::default());
        let owner = {
            let mut world = storage.exclusive();
            let owner = world.spawn((Detached, crate::ObjectId::NIL, crate::Transform::default(), super::Name(String::new()), super::Enabled(true), super::Parent(None)));
            world.insert_one(owner, Owner(owner)).expect("new component owner");
            owner
        };
        Self { storage, owner, slots: Vec::new(), metadata: Vec::new() }
    }
}

impl ComponentSet {
    pub fn new() -> Self { Self::default() }
    pub fn insert_metadata<T: ObjectMetadata>(&mut self, value: T) {
        let mut world = self.storage.exclusive();
        world.insert_one(self.owner, value).expect("object metadata owner");
        if !self.metadata.iter().any(|ty| ty.id == TypeId::of::<T>()) { self.metadata.push(MetadataType::of::<T>()); }
    }

    pub fn metadata<T: ObjectMetadata>(&self) -> Option<ComponentRef<'_, T>> { self.storage.read::<T>(self.owner) }
    pub fn metadata_mut<T: ObjectMetadata>(&self) -> Option<ComponentMut<'_, T>> { self.storage.write::<T>(self.owner) }
    pub fn owner_entity(&self) -> Entity { self.owner }
    pub fn is_detached(&self) -> bool { self.storage.read::<Detached>(self.owner).is_some() }
    pub fn owner_parent(&self) -> Option<Entity> {
        self.storage.read::<super::Parent>(self.owner).expect("object parent").0
    }
    /// Assign a local hierarchy link. SceneWorld validates the complete graph
    /// before propagating derived world transforms and inherited activation.
    pub fn set_owner_parent(&self, parent: Option<Entity>) {
        self.storage.write::<super::Parent>(self.owner).expect("object parent").0 = parent;
    }

    pub fn owner_id(&self) -> crate::ObjectId {
        *self.storage.read::<crate::ObjectId>(self.owner).expect("object saved identity")
    }

    /// Live identities must be edited through SceneWorld so its saved-ID index
    /// changes with the value. Only construction owners may use this setter.
    pub fn set_detached_id(&self, id: crate::ObjectId) {
        assert!(self.storage.read::<Detached>(self.owner).is_some(), "live object identities must be edited through SceneWorld");
        *self.storage.write::<crate::ObjectId>(self.owner).expect("object saved identity") = id;
    }
    pub fn owner_transform(&self) -> crate::Transform {
        *self.storage.read::<crate::Transform>(self.owner).expect("object transform")
    }

    pub fn owner_transform_mut(&self) -> ComponentMut<'_, crate::Transform> {
        self.storage.write::<crate::Transform>(self.owner).expect("object transform")
    }

    pub fn owner_name(&self) -> ComponentRef<'_, String> {
        ComponentRef::map(self.storage.read::<super::Name>(self.owner).expect("object name"), |name| &name.0)
    }

    pub fn owner_name_mut(&self) -> ComponentMut<'_, String> {
        ComponentMut::map(self.storage.write::<super::Name>(self.owner).expect("object name"), |name| &mut name.0)
    }

    pub fn owner_enabled(&self) -> bool {
        self.storage.read::<super::Enabled>(self.owner).expect("object enabled state").0
    }

    pub fn set_owner_enabled(&self, enabled: bool) {
        self.storage.write::<super::Enabled>(self.owner).expect("object enabled state").0 = enabled;
    }

    pub fn owner_enabled_mut(&self) -> ComponentMut<'_, bool> {
        ComponentMut::map(self.storage.write::<super::Enabled>(self.owner).expect("object enabled state"), |enabled| &mut enabled.0)
    }

    /// Bind a complete scene object, copying construction metadata once.
    /// Gameplay-only sets may still use attach without replacing owner data.
    pub fn attach_object(&mut self, scene: &SceneWorld, owner: Entity) {
        if Arc::ptr_eq(&self.storage, &scene.world) && self.owner == owner { return; }
        let transform = self.owner_transform();
        let name = self.owner_name().clone();
        let enabled = self.owner_enabled();
        self.attach(scene, owner);
        *self.owner_transform_mut() = transform;
        *self.owner_name_mut() = name;
        self.set_owner_enabled(enabled);
    }


    pub fn from_vec(values: Vec<Box<dyn Component>>) -> Self { values.into() }
    pub fn len(&self) -> usize { self.slots.len() }
    pub fn is_empty(&self) -> bool { self.slots.is_empty() }
    pub fn slot_type(&self, index: usize) -> Option<TypeId> { self.slots.get(index).map(|slot| slot.ty.id) }
    pub fn slot_name(&self, index: usize) -> Option<&'static str> { self.slots.get(index).map(|slot| slot.name) }
    /// No payload borrow or storage lock; safe while native queries hold the world.
    pub fn native_slot(&self, index: usize) -> Option<ComponentSlot> {
        self.slots.get(index).map(|slot| ComponentSlot { name: slot.name, entity: slot.entity, ty: slot.ty })
    }
    /// Native instance handles in authoring order, without borrowing component
    /// values. Use these with the owning scene's SystemWorld when an operation
    /// must preserve duplicate-instance order (for example collider creation).
    /// Disabled payloads have their own `DisabledComponent<T>` storage type.
    pub fn instance_entities<T: Component>(&self) -> impl Iterator<Item = (usize, Entity)> + '_ {
        self.slots.iter().enumerate().filter_map(|(index, slot)|
            (slot.ty.id == TypeId::of::<T>()).then_some((index, slot.entity)))
    }
    pub fn handle(&self, index: usize) -> Option<ComponentHandle> {
        Some(ComponentHandle { storage: self.storage.clone(), slot: *self.slots.get(index)? })
    }

    pub fn handle_typed<T: Component>(&self) -> Option<ComponentHandle> {
        self.handle(self.slots.iter().position(|slot| slot.ty.id == TypeId::of::<T>())?)
    }

    pub fn get(&self, index: usize) -> Option<ComponentRef<'_, dyn Component>> {
        let slot = self.slots.get(index)?;
        (slot.ty.read)(&self.storage, slot.entity)
    }

    pub fn get_mut(&self, index: usize) -> Option<ComponentMut<'_, dyn Component>> {
        let slot = self.slots.get(index)?;
        (slot.ty.write)(&self.storage, slot.entity)
    }

    pub fn last_mut(&self) -> Option<ComponentMut<'_, dyn Component>> {
        self.get_mut(self.len().checked_sub(1)?)
    }

    pub fn get_typed<T: Component>(&self) -> Option<ComponentRef<'_, T>> {
        let slot = self.slots.iter().find(|slot| slot.ty.id == TypeId::of::<T>())?;
        self.storage.read::<T>(slot.entity)
    }

    pub fn get_typed_mut<T: Component>(&self) -> Option<ComponentMut<'_, T>> {
        let slot = self.slots.iter().find(|slot| slot.ty.id == TypeId::of::<T>())?;
        self.storage.write::<T>(slot.entity)
    }

    pub fn get_typed_at<T: Component>(&self, index: usize) -> Option<ComponentRef<'_, T>> {
        let slot = self.slots.get(index).filter(|slot| slot.ty.id == TypeId::of::<T>())?;
        self.storage.read::<T>(slot.entity)
    }

    pub fn get_typed_at_mut<T: Component>(&self, index: usize) -> Option<ComponentMut<'_, T>> {
        let slot = self.slots.get(index).filter(|slot| slot.ty.id == TypeId::of::<T>())?;
        self.storage.write::<T>(slot.entity)
    }

    pub fn handle_including_disabled<T: Component>(&self) -> Option<ComponentHandle> {
        let index = self.iter().position(|value| crate::component_inner(value.as_ref()).as_any().is::<T>())?;
        self.handle(index)
    }

    pub fn get_including_disabled<T: Component>(&self) -> Option<ComponentRef<'_, T>> {
        self.iter().find_map(|value| ComponentRef::filter_map(value, |component| crate::component_inner(component).as_any().downcast_ref::<T>()))
    }

    pub fn get_including_disabled_mut<T: Component>(&self) -> Option<ComponentMut<'_, T>> {
        self.iter_mut().find_map(|value| ComponentMut::filter_map(value, |component| crate::component_inner_data_mut(component).as_any_mut().downcast_mut::<T>()))
    }

    pub fn set_enabled(&mut self, index: usize, enabled: bool) -> bool {
        if self.get(index).is_none_or(|value| crate::component_enabled(value.as_ref()) == enabled) { return false; }
        let mut value = self.remove(index);
        let changed = crate::set_component_enabled(&mut value, enabled);
        self.insert(index, value);
        changed
    }

    pub fn contains_type<T: 'static>(&self) -> bool {
        self.slots.iter().any(|slot| slot.ty.id == TypeId::of::<T>())
    }

    pub fn iter_typed<T: Component>(&self) -> impl Iterator<Item = ComponentRef<'_, T>> {
        self.slots.iter().filter(|slot| slot.ty.id == TypeId::of::<T>())
            .map(|slot| self.storage.read::<T>(slot.entity).expect("live typed component slot"))
    }

    pub fn iter_typed_mut<T: Component>(&self) -> impl Iterator<Item = ComponentMut<'_, T>> {
        self.slots.iter().filter(|slot| slot.ty.id == TypeId::of::<T>())
            .map(|slot| self.storage.write::<T>(slot.entity).expect("live typed component slot"))
    }

    pub fn iter(&self) -> ComponentIter<'_> {
        ComponentIter { storage: &self.storage, slots: self.slots.iter() }
    }

    pub fn iter_mut(&self) -> ComponentIterMut<'_> {
        ComponentIterMut { storage: &self.storage, slots: self.slots.iter() }
    }

    pub fn push(&mut self, value: Box<dyn Component>) {
        let value = value.into_storage();
        let name = value.type_name();
        let ty = value.storage_type();
        let entity = (ty.put)(&mut self.storage.exclusive(), self.owner, value);
        self.slots.push(Slot { name, entity, ty });
    }

    pub fn insert(&mut self, index: usize, value: Box<dyn Component>) {
        assert!(index <= self.len());
        self.push(value);
        let slot = self.slots.pop().expect("inserted component");
        self.slots.insert(index, slot);
        self.normalize(slot.ty.id);
    }

    pub fn remove(&mut self, index: usize) -> Box<dyn Component> {
        let slot = self.slots[index];
        let value = {
            let mut world = self.storage.exclusive();
            let value = (slot.ty.take)(&mut world, slot.entity).expect("live component slot");
            self.storage.forget_type(slot.entity, slot.ty.id);
            if !world.contains(slot.entity) { self.storage.forget(&[slot.entity]); }
            value
        };
        self.slots.remove(index);
        self.normalize(slot.ty.id);
        value
    }

    pub fn replace(&mut self, index: usize, value: Box<dyn Component>) -> Box<dyn Component> {
        let previous = self.remove(index);
        self.insert(index, value);
        previous
    }

    pub fn clear(&mut self) {
        for slot in self.slots.drain(..) {
            self.storage.retire(Retired { entity: slot.entity, ty: slot.ty.id, drop: slot.ty.drop });
        }
    }

    pub fn replace_all(&mut self, values: Vec<Box<dyn Component>>) {
        self.clear();
        self.extend(values);
    }

    pub fn retain(&mut self, mut keep: impl FnMut(&ComponentRef<'_, dyn Component>) -> bool) {
        let mut removed = Vec::new();
        for (index, component) in self.iter().enumerate() {
            if !keep(&component) { removed.push(index); }
        }
        for index in removed.into_iter().rev() { drop(self.remove(index)); }
    }

    pub fn drain(&mut self, range: impl RangeBounds<usize>) -> std::vec::IntoIter<Box<dyn Component>> {
        let storage = self.storage.clone();
        let mut world = storage.exclusive();
        let mut affected = std::collections::HashSet::new();
        let values: Vec<_> = self.slots.drain(range).map(|slot| {
            affected.insert(slot.ty.id);
            let value = (slot.ty.take)(&mut world, slot.entity).expect("live component slot");
            storage.forget_type(slot.entity, slot.ty.id);
            if !world.contains(slot.entity) { storage.forget(&[slot.entity]); }
            value
        }).collect();
        drop(world);
        for ty in affected { self.normalize(ty); }
        values.into_iter()
    }

    pub fn pop(&mut self) -> Option<Box<dyn Component>> {
        (!self.is_empty()).then(|| self.remove(self.len() - 1))
    }

    pub fn swap(&mut self, a: usize, b: usize) {
        self.slots.swap(a, b);
        self.normalize(self.slots[a].ty.id);
        self.normalize(self.slots[b].ty.id);
    }

    pub fn swap_remove(&mut self, index: usize) -> Box<dyn Component> {
        let value = self.remove(index);
        if index < self.len() {
            let last = self.slots.pop().expect("last component");
            self.slots.insert(index, last);
            self.normalize(last.ty.id);
        }
        value
    }

    /// Bind a constructed object to its scene once. Subsequent system stages
    /// borrow its values in place; they never transfer the components back out.
    pub fn attach(&mut self, scene: &SceneWorld, owner: Entity) {
        assert!(scene.object_id(owner).is_some(), "component owner is a scene object");
        self.attach_storage(scene.world.clone(), owner);
    }

    fn attach_storage(&mut self, storage: Arc<Storage>, owner: Entity) {
        if Arc::ptr_eq(&self.storage, &storage) && self.owner == owner { return; }
        if Arc::ptr_eq(&self.storage, &storage) {
            let mut world = storage.exclusive();
            for ty in &self.metadata {
                (ty.move_within)(&mut world, self.owner, owner);
                storage.forget_type(self.owner, ty.id);
            }
            for slot in &mut self.slots {
                let value = (slot.ty.take)(&mut world, slot.entity).expect("live source component");
                storage.forget_type(slot.entity, slot.ty.id);
                if !world.contains(slot.entity) { storage.forget(&[slot.entity]); }
                slot.entity = (slot.ty.put)(&mut world, owner, value);
            }
            self.owner = owner;
            return;
        }
        let mut source = self.storage.exclusive();
        let mut target = storage.exclusive();
        for ty in &self.metadata {
            (ty.move_between)(&mut source, self.owner, &mut target, owner);
            self.storage.forget_type(self.owner, ty.id);
        }
        for slot in &mut self.slots {
            let value = (slot.ty.take)(&mut source, slot.entity).expect("live source component");
            self.storage.forget_type(slot.entity, slot.ty.id);
            if !source.contains(slot.entity) { self.storage.forget(&[slot.entity]); }
            slot.entity = (slot.ty.put)(&mut target, owner, value);
        }
        if source.get::<&Detached>(self.owner).is_ok() {
            source.despawn(self.owner).expect("detached component owner");
        }
        drop(target);
        drop(source);
        self.storage = storage;
        self.owner = owner;
    }

    /// Move an object out of a scene while keeping its live component values.
    pub fn detach(&mut self) {
        let id = self.owner_id();
        let transform = self.owner_transform();
        let name = self.owner_name().clone();
        let enabled = self.owner_enabled();
        let detached = Self::new();
        self.attach_storage(detached.storage.clone(), detached.owner);
        self.set_detached_id(id);
        *self.owner_transform_mut() = transform;
        *self.owner_name_mut() = name;
        self.set_owner_enabled(enabled);
    }

    fn normalize(&mut self, ty: TypeId) {
        let Some(first) = self.slots.iter().position(|slot| slot.ty.id == ty) else { return };
        if self.slots[first].entity == self.owner { return; }
        let mut world = self.storage.exclusive();
        if let Some(primary) = self.slots.iter().position(|slot| slot.ty.id == ty && slot.entity == self.owner) {
            (self.slots[first].ty.swap)(&mut world, self.slots[first].entity, self.owner);
            self.slots[primary].entity = self.slots[first].entity;
            self.slots[first].entity = self.owner;
        } else {
            let slot = &mut self.slots[first];
            let value = (slot.ty.take)(&mut world, slot.entity).expect("component to promote");
            self.storage.forget(&[slot.entity]);
            slot.entity = (slot.ty.put)(&mut world, self.owner, value);
        }
    }

    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ { self.slots.iter().map(|slot| slot.name) }
}

impl Drop for ComponentSet {
    fn drop(&mut self) { self.clear(); }
}

impl Extend<Box<dyn Component>> for ComponentSet {
    fn extend<I: IntoIterator<Item = Box<dyn Component>>>(&mut self, values: I) {
        for value in values { self.push(value); }
    }
}

impl FromIterator<Box<dyn Component>> for ComponentSet {
    fn from_iter<I: IntoIterator<Item = Box<dyn Component>>>(values: I) -> Self {
        let mut components = Self::new();
        components.extend(values);
        components
    }
}

impl From<Vec<Box<dyn Component>>> for ComponentSet {
    fn from(values: Vec<Box<dyn Component>>) -> Self { values.into_iter().collect() }
}

pub struct ComponentIter<'a> {
    storage: &'a Storage,
    slots: std::slice::Iter<'a, Slot>,
}

impl<'a> Iterator for ComponentIter<'a> {
    type Item = ComponentRef<'a, dyn Component>;
    fn next(&mut self) -> Option<Self::Item> {
        let slot = self.slots.next()?;
        Some((slot.ty.read)(self.storage, slot.entity).expect("live component slot"))
    }
    fn size_hint(&self) -> (usize, Option<usize>) { self.slots.size_hint() }
}
impl DoubleEndedIterator for ComponentIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let slot = self.slots.next_back()?;
        Some((slot.ty.read)(self.storage, slot.entity).expect("live component slot"))
    }
}
impl ExactSizeIterator for ComponentIter<'_> {}

pub struct ComponentIterMut<'a> {
    storage: &'a Storage,
    slots: std::slice::Iter<'a, Slot>,
}

impl<'a> Iterator for ComponentIterMut<'a> {
    type Item = ComponentMut<'a, dyn Component>;
    fn next(&mut self) -> Option<Self::Item> {
        let slot = self.slots.next()?;
        Some((slot.ty.write)(self.storage, slot.entity).expect("live component slot"))
    }
    fn size_hint(&self) -> (usize, Option<usize>) { self.slots.size_hint() }
}
impl DoubleEndedIterator for ComponentIterMut<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let slot = self.slots.next_back()?;
        Some((slot.ty.write)(self.storage, slot.entity).expect("live component slot"))
    }
}
impl ExactSizeIterator for ComponentIterMut<'_> {}

impl<'a> IntoIterator for &'a ComponentSet {
    type Item = ComponentRef<'a, dyn Component>;
    type IntoIter = ComponentIter<'a>;
    fn into_iter(self) -> Self::IntoIter { self.iter() }
}
impl<'a> IntoIterator for &'a mut ComponentSet {
    type Item = ComponentMut<'a, dyn Component>;
    type IntoIter = ComponentIterMut<'a>;
    fn into_iter(self) -> Self::IntoIter { self.iter_mut() }
}
impl IntoIterator for ComponentSet {
    type Item = Box<dyn Component>;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(mut self) -> Self::IntoIter { self.drain(..) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TypedComponent;
    use serde::{Deserialize, Serialize};
    use std::panic::{AssertUnwindSafe, catch_unwind};

    #[derive(Default, Serialize, Deserialize)]
    struct Counter {
        value: u32,
        #[serde(skip)]
        live: u32,
    }
    impl TypedComponent for Counter { const NAME: &'static str = "Storage Test Counter"; }

    fn counter(value: u32) -> Box<dyn Component> { Box::new(Counter { value, live: 42 }) }

    #[test]
    fn scene_systems_and_editor_views_edit_the_same_permanent_values() {
        let mut scene = SceneWorld::new();
        let owner = scene.spawn("Object");
        let mut components = ComponentSet::new();
        components.push(counter(3));
        components.attach(&scene, owner);
        for _ in 0..10 {
            scene.with_world(|world| {
                for (_, value) in world.query::<&mut Counter>().iter() { value.value += 1; }
            });
            assert_eq!(components.get_typed::<Counter>().unwrap().live, 42);
        }
        assert_eq!(components.get_typed::<Counter>().unwrap().value, 13);
        components.get_typed_mut::<Counter>().unwrap().value = 25;
        scene.with_world(|world| assert_eq!(world.get::<Counter>(owner).unwrap().value, 25));
    }

    #[test]
    fn attaching_gameplay_data_preserves_existing_owner_metadata() {
        let mut scene = SceneWorld::new();
        let transform = crate::Transform { translation: glam::Vec3::X * 5.0, ..Default::default() };
        let owner = scene.spawn_with_id(crate::ObjectId::new(), "Existing", transform, false).unwrap();
        let mut components = ComponentSet::new();
        components.push(counter(3));
        components.attach(&scene, owner);
        assert_eq!(components.owner_transform(), transform);
        assert_eq!(&*components.owner_name(), "Existing");
        assert!(!components.owner_enabled());
        components.attach_object(&scene, owner);
        assert_eq!(&*components.owner_name(), "Existing", "rebinding the same object is idempotent");
        assert!(!components.owner_enabled());
    }

    #[test]
    fn saved_identity_is_authoritative_and_live_changes_require_the_world_index() {
        let id = crate::ObjectId::new();
        let mut components = ComponentSet::new();
        components.set_detached_id(id);
        let mut scene = SceneWorld::new();
        let owner = scene.spawn_with_id(id, "Object", Default::default(), true).unwrap();
        components.attach_object(&scene, owner);
        assert_eq!(components.owner_entity(), owner);
        assert_eq!(components.owner_id(), id);
        let next = crate::ObjectId::new();
        assert!(catch_unwind(AssertUnwindSafe(|| components.set_detached_id(next))).is_err());
        assert_eq!(components.owner_id(), id);
        assert_eq!(scene.entity(id), Some(owner));
        assert!(scene.entity(next).is_none());
        scene.set_object_id(owner, next).unwrap();
        assert_eq!(components.owner_id(), next);
        assert!(scene.entity(id).is_none());
        assert_eq!(scene.entity(next), Some(owner));
        components.detach();
        assert_eq!(components.owner_id(), next);
        components.set_detached_id(id);
        assert_eq!(components.owner_id(), id);
        assert_eq!(scene.object_id(owner), Some(next));
    }

    #[test]
    fn native_object_metadata_is_moved_once_without_cloning_or_gameplay_slots() {
        struct AuthoringData(Vec<u8>);
        impl ObjectMetadata for AuthoringData {}
        let mut components = ComponentSet::new();
        components.insert_metadata(AuthoringData(vec![1]));
        components.insert_metadata(AuthoringData(vec![2, 3]));
        let allocation = components.metadata::<AuthoringData>().unwrap().0.as_ptr();
        assert!(components.is_empty());
        let mut scene = SceneWorld::new();
        let owner = scene.spawn("First");
        components.attach_object(&scene, owner);
        assert_eq!(components.metadata::<AuthoringData>().unwrap().0.as_ptr(), allocation);
        scene.with_world(|world| {
            assert_eq!(world.query::<(&crate::ObjectId, &AuthoringData)>().iter().count(), 1);
            world.get_mut::<AuthoringData>(owner).unwrap().0.push(4);
        });
        assert_eq!(components.metadata::<AuthoringData>().unwrap().0, [2, 3, 4]);
        components.clear();
        assert!(components.metadata::<AuthoringData>().is_some());
        let other = scene.spawn("Other");
        components.attach_object(&scene, other);
        assert!(scene.get::<AuthoringData>(owner).is_none());
        assert_eq!(scene.get::<AuthoringData>(other).unwrap().0, [2, 3, 4]);
        components.detach();
        assert!(scene.get::<AuthoringData>(other).is_none());
        let mut next = SceneWorld::new();
        let next_owner = next.spawn("Next");
        components.attach_object(&next, next_owner);
        assert_eq!(next.get::<AuthoringData>(next_owner).unwrap().0, [2, 3, 4]);
    }

    #[test]
    fn independent_objects_can_be_edited_together_and_drop_is_deferred_safely() {
        let mut scene = SceneWorld::new();
        let a = scene.spawn("A");
        let b = scene.spawn("B");
        let mut left = ComponentSet::new();
        let mut right = ComponentSet::new();
        left.push(counter(1)); right.push(counter(2));
        left.attach(&scene, a); right.attach(&scene, b);
        {
            let mut first = left.get_typed_mut::<Counter>().unwrap();
            let mut second = right.get_typed_mut::<Counter>().unwrap();
            std::mem::swap(&mut first.value, &mut second.value);
            assert!(catch_unwind(AssertUnwindSafe(|| left.get_typed::<Counter>())).is_err());
        }
        let first = left.get_typed::<Counter>().unwrap();
        drop(right);
        assert_eq!(first.value, 2);
        drop(first);
        scene.with_world(|world| assert_eq!(world.query::<&Counter>().iter().count(), 1));
    }

    #[test]
    fn duplicate_order_and_primary_entity_queries_stay_consistent() {
        let mut scene = SceneWorld::new();
        let owner = scene.spawn("Object");
        let mut components = ComponentSet::new();
        components.push(counter(10)); components.push(counter(20));
        components.attach(&scene, owner);
        components.insert(0, counter(5));
        scene.with_world(|world| {
            assert_eq!(world.get::<Counter>(owner).unwrap().value, 5);
            assert_eq!(world.query::<&Counter>().iter().count(), 3);
        });
        assert_eq!(components.iter().map(|c| c.as_any().downcast_ref::<Counter>().unwrap().value).collect::<Vec<_>>(), vec![5, 10, 20]);
        assert_eq!(components.remove(0).as_any().downcast_ref::<Counter>().unwrap().value, 5);
        scene.with_world(|world| assert_eq!(world.get::<Counter>(owner).unwrap().value, 10));
        assert_eq!(components.remove(0).as_any().downcast_ref::<Counter>().unwrap().value, 10);
        scene.with_world(|world| assert_eq!(world.get::<Counter>(owner).unwrap().value, 20));
    }

    #[test]
    fn moving_between_scenes_preserves_live_state_without_a_serde_roundtrip() {
        let mut first = SceneWorld::new();
        let first_owner = first.spawn("First");
        let mut second = SceneWorld::new();
        let second_owner = second.spawn("Second");
        let mut components = ComponentSet::new();
        components.push(counter(7));
        components.attach(&first, first_owner);
        components.get_typed_mut::<Counter>().unwrap().live = 123;
        components.detach();
        first.with_world(|world| assert_eq!(world.query::<&Counter>().iter().count(), 0));
        components.attach(&second, second_owner);
        assert_eq!(components.get_typed::<Counter>().unwrap().live, 123);
        second.with_world(|world| assert_eq!(world.get::<Counter>(second_owner).unwrap().value, 7));
    }

    #[test]
    fn native_panic_releases_access_and_keeps_component_values() {
        let mut scene = SceneWorld::new();
        let owner = scene.spawn("Object");
        let mut components = ComponentSet::new();
        components.push(counter(1));
        components.attach(&scene, owner);
        assert!(catch_unwind(AssertUnwindSafe(|| scene.with_world(|world| {
            world.get_mut::<Counter>(owner).unwrap().value = 9;
            panic!("system failure");
        }))).is_err());
        assert_eq!(components.get_typed::<Counter>().unwrap().value, 9);
        scene.with_world(|world| assert_eq!(world.query::<&Counter>().iter().count(), 1));
    }
}
