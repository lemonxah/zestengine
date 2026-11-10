//! Shared ownership of the permanent hecs storage.
//!
//! Component views hold a structural read lock and an entity/type borrow lease.
//! This permits independent components of the same type to be edited through
//! editor views. A native system takes the structural write lock for its whole
//! run and uses hecs' normal column borrows. The two access modes cannot overlap.

use std::any::TypeId;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError};
use std::sync::atomic::{AtomicIsize, Ordering};

use super::{DataComponent, Entity};

#[derive(Default)]
pub(crate) struct Storage {
    world: RwLock<hecs::World>,
    borrows: Mutex<HashMap<Entity, HashMap<TypeId, Arc<AtomicIsize>>>>,
    retired: Mutex<Vec<Retired>>,
}

pub(crate) struct Retired {
    pub entity: Entity,
    pub ty: TypeId,
    pub drop: fn(&mut hecs::World, Entity),
}

impl Storage {
    pub(crate) fn exclusive(&self) -> RwLockWriteGuard<'_, hecs::World> {
        let mut world = match self.world.try_write() {
            Ok(world) => world,
            Err(TryLockError::Poisoned(poison)) => poison.into_inner(),
            Err(TryLockError::WouldBlock) => panic!("structural ECS access while component views are borrowed; release the views or queue the operation"),
        };
        loop {
            let retired = std::mem::take(&mut *self.retired.lock().unwrap_or_else(|poison| poison.into_inner()));
            if retired.is_empty() { break; }
            for value in retired {
                (value.drop)(&mut world, value.entity);
                if world.contains(value.entity) { self.forget_type(value.entity, value.ty); }
                else { self.forget(&[value.entity]); }
            }
        }
        world
    }

    fn structural_read(&self) -> RwLockReadGuard<'_, hecs::World> {
        match self.world.try_read() {
            Ok(world) => world,
            Err(TryLockError::Poisoned(poison)) => poison.into_inner(),
            Err(TryLockError::WouldBlock) => panic!("component-view access during native ECS execution; use the system's SystemWorld view"),
        }
    }

    pub(crate) fn retire(&self, value: Retired) {
        self.retired.lock().unwrap_or_else(|poison| poison.into_inner()).push(value);
    }

    fn lease(&self, entity: Entity, ty: TypeId, mutable: bool) -> Lease {
        let state = self.borrows.lock().unwrap_or_else(|poison| poison.into_inner())
            .entry(entity).or_default().entry(ty).or_default().clone();
        let borrowed = if mutable {
            state.compare_exchange(0, -1, Ordering::Acquire, Ordering::Relaxed).is_ok()
        } else {
            state.fetch_update(Ordering::Acquire, Ordering::Relaxed, |count| {
                (count >= 0 && count < isize::MAX).then(|| count + 1)
            }).is_ok()
        };
        assert!(borrowed, "conflicting ECS component borrow for {entity:?}");
        Lease { state, mutable }
    }

    pub(crate) fn read<T: DataComponent>(&self, entity: Entity) -> Option<ComponentRef<'_, T>> {
        let structural = self.structural_read();
        if !structural.satisfies::<&T>(entity).unwrap_or(false) { return None; }
        let lease = self.lease(entity, TypeId::of::<T>(), false);
        let value = structural.get::<&T>(entity).ok()?;
        let pointer = NonNull::from(&*value);
        drop(value);
        Some(ComponentRef { pointer, lease, structural })
    }

    pub(crate) fn write<T: DataComponent>(&self, entity: Entity) -> Option<ComponentMut<'_, T>> {
        let structural = self.structural_read();
        if !structural.satisfies::<&T>(entity).unwrap_or(false) { return None; }
        let lease = self.lease(entity, TypeId::of::<T>(), true);
        let mut value = structural.get::<&mut T>(entity).ok()?;
        let pointer = NonNull::from(&mut *value);
        drop(value);
        Some(ComponentMut { pointer, lease, structural })
    }

    /// Called only while holding exclusive structural access, when no view can
    /// still refer to any of the removed entities' components.
    pub(crate) fn forget(&self, entities: &[Entity]) {
        let mut borrows = self.borrows.lock().unwrap_or_else(|poison| poison.into_inner());
        for entity in entities { borrows.remove(entity); }
    }

    pub(crate) fn forget_type(&self, entity: Entity, ty: TypeId) {
        let mut borrows = self.borrows.lock().unwrap_or_else(|poison| poison.into_inner());
        if let Some(row) = borrows.get_mut(&entity) {
            row.remove(&ty);
            if row.is_empty() { borrows.remove(&entity); }
        }
    }
}

struct Lease {
    state: Arc<AtomicIsize>,
    mutable: bool,
}

impl Drop for Lease {
    fn drop(&mut self) {
        if self.mutable { self.state.store(0, Ordering::Release); }
        else { self.state.fetch_sub(1, Ordering::Release); }
    }
}

/// A borrowed component in the permanent ECS storage. Dropping the view releases
/// both its component lease and the lock preventing archetype relocation.
pub struct ComponentRef<'a, T: ?Sized> {
    pointer: NonNull<T>,
    lease: Lease,
    structural: RwLockReadGuard<'a, hecs::World>,
}

impl<T: ?Sized> Deref for ComponentRef<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: the structural lock prevents moving/dropping the allocation;
        // the shared lease excludes mutable access to this entity/type slot.
        unsafe { self.pointer.as_ref() }
    }
}

impl<'a, T: ?Sized> ComponentRef<'a, T> {
    pub fn map<U: ?Sized>(self, project: impl FnOnce(&T) -> &U) -> ComponentRef<'a, U> {
        let pointer = NonNull::from(project(&*self));
        ComponentRef { pointer, lease: self.lease, structural: self.structural }
    }

    pub fn filter_map<U: ?Sized>(self, project: impl FnOnce(&T) -> Option<&U>) -> Option<ComponentRef<'a, U>> {
        let pointer = NonNull::from(project(&*self)?);
        Some(ComponentRef { pointer, lease: self.lease, structural: self.structural })
    }

    pub fn as_ref(&self) -> &T { self }
}

/// Exclusive component access, scoped to one entity/type rather than an entire
/// archetype. Independent instances can be edited together by editor tools.
pub struct ComponentMut<'a, T: ?Sized> {
    pointer: NonNull<T>,
    lease: Lease,
    structural: RwLockReadGuard<'a, hecs::World>,
}

impl<T: ?Sized> Deref for ComponentMut<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: the lease exclusively owns this slot and the structural lock
        // keeps the component allocation in place.
        unsafe { self.pointer.as_ref() }
    }
}

impl<T: ?Sized> DerefMut for ComponentMut<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: neither another view of this slot nor a native system can
        // access the component until this lease and structural lock are dropped.
        unsafe { self.pointer.as_mut() }
    }
}

impl<'a, T: ?Sized> ComponentMut<'a, T> {
    pub fn map<U: ?Sized>(mut self, project: impl FnOnce(&mut T) -> &mut U) -> ComponentMut<'a, U> {
        let pointer = NonNull::from(project(&mut *self));
        ComponentMut { pointer, lease: self.lease, structural: self.structural }
    }

    pub fn as_ref(&self) -> &T { self }
    pub fn as_mut(&mut self) -> &mut T { self }

    pub fn filter_map<U: ?Sized>(mut self, project: impl FnOnce(&mut T) -> Option<&mut U>) -> Option<ComponentMut<'a, U>> {
        let pointer = NonNull::from(project(&mut *self)?);
        Some(ComponentMut { pointer, lease: self.lease, structural: self.structural })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    #[test]
    fn independent_instances_and_fields_can_be_borrowed_without_relocating_data() {
        let store = Storage::default();
        let (a, b) = {
            let mut world = store.exclusive();
            (world.spawn((1u32, String::from("a"))), world.spawn((2u32, String::from("b"))))
        };
        {
            let mut a_value = store.write::<u32>(a).unwrap();
            let mut b_value = store.write::<u32>(b).unwrap();
            let name = store.read::<String>(a).unwrap();
            *a_value += *b_value;
            *b_value += 10;
            assert_eq!(&*name, "a");
            assert!(catch_unwind(AssertUnwindSafe(|| store.read::<u32>(a))).is_err());
            assert!(catch_unwind(AssertUnwindSafe(|| store.exclusive())).is_err());
        }
        assert_eq!(*store.read::<u32>(a).unwrap(), 3);
        assert_eq!(*store.read::<u32>(b).unwrap(), 12);
        let mut world = store.exclusive();
        for _ in 0..1024 { world.spawn((3u32, String::new())); }
        drop(world);
        assert_eq!(*store.read::<u32>(a).unwrap(), 3);
    }

    #[test]
    fn projected_views_keep_the_original_slot_borrowed() {
        let store = Storage::default();
        let entity = store.exclusive().spawn((String::from("hello"),));
        let view = ComponentRef::map(store.read::<String>(entity).unwrap(), |s| &s[..2]);
        assert_eq!(&*view, "he");
        assert!(catch_unwind(AssertUnwindSafe(|| store.write::<String>(entity))).is_err());
        drop(view);
        let mut view = ComponentMut::map(store.write::<String>(entity).unwrap(), |s| s.as_mut_str());
        view.make_ascii_uppercase();
        drop(view);
        assert_eq!(&*store.read::<String>(entity).unwrap(), "HELLO");
    }
}
