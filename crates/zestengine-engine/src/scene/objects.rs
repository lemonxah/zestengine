//! Ordered handles for objects whose data and hierarchy live in permanent ECS storage.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock, RwLock};
use std::ops::{Deref, Index, IndexMut, RangeBounds};
use std::slice::SliceIndex;

use zestengine_core::{ObjectId, Transform};
use zestengine_core::ecs::{DataComponent, HierarchyError, Parent, SceneWorld, SystemWorld, Entity};

use super::{SceneObject, metadata::ParentRequest};

/// Derived authoring order only. Parent links are never stored in these tables.
#[derive(Default)]
pub(super) struct SceneOrder {
    tables: RwLock<OrderTables>,
}
#[derive(Default)]
struct OrderTables {
    entities: Vec<Entity>,
    indices: HashMap<Entity, usize>,
    waiting: HashMap<usize, Vec<Entity>>,
    waiting_saved: HashMap<ObjectId, Vec<Entity>>,
}
impl SceneOrder {
    pub(super) fn index(&self, entity: Entity) -> Option<usize> {
        self.tables.read().unwrap_or_else(|p| p.into_inner()).indices.get(&entity).copied()
    }
    pub(super) fn entity(&self, index: usize) -> Option<Entity> {
        self.tables.read().unwrap_or_else(|p| p.into_inner()).entities.get(index).copied()
    }
    pub(super) fn wait_for(&self, child: Entity, target: usize) {
        let mut tables = self.tables.write().unwrap_or_else(|p| p.into_inner());
        let waiting = tables.waiting.entry(target).or_default();
        if !waiting.contains(&child) { waiting.push(child); }
    }
    fn wait_for_saved(&self, child: Entity, target: ObjectId) {
        let mut tables = self.tables.write().unwrap_or_else(|p| p.into_inner());
        let waiting = tables.waiting_saved.entry(target).or_default();
        if !waiting.contains(&child) { waiting.push(child); }
    }
    fn identity_arrived(&self, id: ObjectId) -> Vec<Entity> {
        self.tables.write().unwrap_or_else(|p| p.into_inner()).waiting_saved.remove(&id).unwrap_or_default()
    }
    fn append(&self, entity: Entity, id: ObjectId) -> Vec<Entity> {
        let mut tables = self.tables.write().unwrap_or_else(|p| p.into_inner());
        let index = tables.entities.len();
        tables.entities.push(entity);
        tables.indices.insert(entity, index);
        let mut waiting = tables.waiting.remove(&index).unwrap_or_default();
        waiting.extend(tables.waiting_saved.remove(&id).unwrap_or_default());
        waiting
    }
    fn replace(&self, entities: Vec<Entity>) {
        let indices = entities.iter().enumerate().map(|(i, &entity)| (entity, i)).collect();
        *self.tables.write().unwrap_or_else(|p| p.into_inner()) = OrderTables { entities, indices, waiting: HashMap::new(), waiting_saved: HashMap::new() };
    }
    fn pending(&self) -> Vec<Entity> {
        let tables = self.tables.read().unwrap_or_else(|p| p.into_inner());
        tables.waiting.values().chain(tables.waiting_saved.values()).flatten().copied().collect()
    }
    fn clear_waiting(&self) {
        let mut tables = self.tables.write().unwrap_or_else(|p| p.into_inner());
        tables.waiting.clear();
        tables.waiting_saved.clear();
    }
}

#[derive(Default)]
pub struct SceneObjects {
    ordered: Vec<SceneObject>,
    order: Arc<SceneOrder>,
    index: OnceLock<HashMap<ObjectId, usize>>,
    reserved: HashSet<ObjectId>,
    pub(crate) world: SceneWorld,
}

impl zestengine_core::ecs::ComponentIndex for SceneObjects {
    fn component_count(&self, owner: Entity) -> Option<usize> {
        let row = self.order.index(owner)?;
        self.ordered.get(row).filter(|object| object.entity() == owner).map(|object| object.components.len())
    }
    fn component_slot(&self, owner: Entity, slot: usize) -> Option<zestengine_core::ecs::ComponentSlot> {
        let row = self.order.index(owner)?;
        self.ordered.get(row).filter(|object| object.entity() == owner)?.components.native_slot(slot)
    }
}

impl SceneObjects {
    pub fn new() -> Self { Self::default() }
    pub fn index_of(&self, id: ObjectId) -> Option<usize> { self.indices().get(&id).copied() }
    pub(crate) fn indices(&self) -> &HashMap<ObjectId, usize> {
        self.index.get_or_init(|| self.ordered.iter().enumerate().map(|(i, object)| (object.id(), i)).collect())
    }

    /// Prepare derived identity lookup before entering native checked query access.
    pub(crate) fn query<R>(&self, run: impl FnOnce(&SystemWorld<'_>, &HashMap<ObjectId, usize>) -> R) -> R {
        let indices = self.indices();
        self.world.with_world(|world| run(world, indices))
    }
    pub(crate) fn map_components<T: DataComponent, R>(&self, mut map: impl FnMut(usize, &T, &SystemWorld<'_>) -> Option<R>) -> Vec<R> {
        self.query(|world, indices| {
            let mut values: Vec<_> = world.query::<(&ObjectId, &T)>().iter().filter_map(|(_, (id, component))| {
                let index = *indices.get(id)?;
                map(index, component, world).map(|value| (index, value))
            }).collect();
            values.sort_unstable_by_key(|(index, _)| *index);
            values.into_iter().map(|(_, value)| value).collect()
        })
    }
    pub(crate) fn has_component<T: DataComponent>(&self) -> bool {
        self.query(|world, indices| world.query::<(&ObjectId, &T)>().iter().any(|(_, (id, _))| indices.contains_key(id)))
    }
    pub(crate) fn find_component<T: DataComponent, R>(&self, mut map: impl FnMut(usize, &T, &SystemWorld<'_>) -> Option<R>) -> Option<R> {
        self.query(|world, indices| {
            let mut first = None;
            for (_, (id, component)) in world.query::<(&ObjectId, &T)>().iter() {
                let Some(&index) = indices.get(id) else { continue };
                if first.as_ref().is_some_and(|(before, _)| *before <= index) { continue; }
                if let Some(value) = map(index, component, world) { first = Some((index, value)); }
            }
            first.map(|(_, value)| value)
        })
    }

    fn bind(&mut self, object: &mut SceneObject) {
        let entity = self.world.spawn_with_id(object.id(), object.name(), object.transform(), object.enabled())
            .unwrap_or_else(|error| panic!("inserting scene object {}: {error}", object.id()));
        object.components.attach_object(&self.world, entity);
        object.order = Some(self.order.clone());
    }

    fn resolve_parent(&self, entity: Entity, finish: bool) {
        let Some(request) = self.world.get::<ParentRequest>(entity).map(|request| *request) else { return };
        let parent = if let Some(target) = request.row {
            let parent = target.and_then(|index| self.order.entity(index));
            if let Some(target) = target.filter(|_| parent.is_none() && !finish) { self.order.wait_for(entity, target); return; }
            parent
        } else if let Some(target) = request.saved {
            let parent = self.world.entity(target).filter(|&entity| self.order.index(entity).is_some());
            if parent.is_none() && !finish { self.order.wait_for_saved(entity, target); return; }
            parent
        } else { return };
        self.world.get_mut::<Parent>(entity).expect("object parent").0 = parent;
        *self.world.get_mut::<ParentRequest>(entity).expect("object parent request") = Default::default();
    }

    fn rebuild_order(&mut self) {
        self.order.replace(self.ordered.iter().map(SceneObject::entity).collect());
        for object in &mut self.ordered { object.order = Some(self.order.clone()); }
        for object in &self.ordered { self.resolve_parent(object.entity(), false); }
    }

    pub fn push(&mut self, mut object: SceneObject) {
        self.index.take();
        self.bind(&mut object);
        let entity = object.entity();
        self.ordered.push(object);
        let waiting = self.order.append(entity, self.ordered.last().expect("inserted object").id());
        self.resolve_parent(entity, false);
        for child in waiting { self.resolve_parent(child, false); }
    }
    pub fn set_id(&mut self, index: usize, id: ObjectId) -> Result<(), HierarchyError> {
        self.index.take();
        let entity = self.ordered.get(index).ok_or(HierarchyError::MissingEntity)?.entity();
        self.world.set_object_id(entity, id)?;
        for child in self.order.identity_arrived(id) { self.resolve_parent(child, false); }
        Ok(())
    }
    pub fn insert(&mut self, index: usize, mut object: SceneObject) {
        self.index.take();
        assert!(index <= self.ordered.len());
        self.bind(&mut object);
        self.ordered.insert(index, object);
        self.rebuild_order();
    }
    pub fn remove(&mut self, index: usize) -> SceneObject {
        self.index.take();
        let parent = self.ordered[index].components.owner_parent().and_then(|entity| self.world.object_id(entity));
        let mut object = self.ordered.remove(index);
        let entity = object.entity();
        object.components.detach();
        object.order = None;
        object.stage_detached_parent(parent);
        self.world.despawn(entity).expect("scene object entity");
        self.rebuild_order();
        object
    }
    pub fn pop(&mut self) -> Option<SceneObject> {
        (!self.ordered.is_empty()).then(|| self.remove(self.ordered.len() - 1))
    }
    pub fn clear(&mut self) {
        self.index.take();
        self.ordered.clear();
        self.remove_missing();
        self.rebuild_order();
    }
    pub fn truncate(&mut self, len: usize) {
        self.index.take();
        self.ordered.truncate(len);
        self.remove_missing();
        self.rebuild_order();
    }
    pub fn retain(&mut self, keep: impl FnMut(&SceneObject) -> bool) {
        self.index.take();
        self.ordered.retain(keep);
        self.remove_missing();
        self.rebuild_order();
    }
    fn remove_missing(&mut self) {
        let mut ids = self.reserved.clone();
        ids.extend(self.ordered.iter().map(SceneObject::id));
        self.world.retain_objects(&ids);
    }
    pub fn drain(&mut self, range: impl RangeBounds<usize>) -> std::vec::IntoIter<SceneObject> {
        self.index.take();
        let mut removed: Vec<_> = self.ordered.drain(range).collect();
        for object in &mut removed {
            let parent = object.components.owner_parent().and_then(|entity| self.world.object_id(entity));
            object.components.detach();
            object.order = None;
            object.stage_detached_parent(parent);
        }
        self.remove_missing();
        self.rebuild_order();
        removed.into_iter()
    }

    /// Pooled rows retain their old derived order while temporary resource rows
    /// use a fresh order in the same permanent ECS world.
    pub(crate) fn take_rows(&mut self) -> Vec<SceneObject> {
        self.index.take();
        self.reserved.extend(self.ordered.iter().map(SceneObject::id));
        self.order = Arc::default();
        std::mem::take(&mut self.ordered)
    }
    pub(crate) fn replace_rows(&mut self, rows: Vec<SceneObject>) {
        self.index.take();
        let ids: HashSet<_> = rows.iter().map(|object| object.id()).collect();
        assert_eq!(ids.len(), rows.len(), "duplicate scene object ID");
        assert!(!ids.iter().any(ObjectId::is_nil), "invalid scene object ID");
        drop(std::mem::replace(&mut self.ordered, rows));
        self.order = Arc::default();
        for object in &mut self.ordered {
            let entity = match self.world.entity(object.id()) {
                Some(entity) => entity,
                None => self.world.spawn_with_id(object.id(), object.name(), object.transform(), object.enabled()).expect("replacement object identity"),
            };
            object.components.attach_object(&self.world, entity);
        }
        self.reserved.clear();
        self.remove_missing();
        self.rebuild_order();
        self.propagate_hierarchy().expect("valid replacement scene hierarchy");
    }
    pub fn reorder(&mut self, order: &[usize]) {
        self.index.take();
        assert_eq!(order.len(), self.ordered.len());
        let mut seen = vec![false; order.len()];
        for &index in order {
            assert!(index < seen.len() && !seen[index], "scene order must be a permutation");
            seen[index] = true;
        }
        let mut slots: Vec<_> = std::mem::take(&mut self.ordered).into_iter().map(Some).collect();
        self.ordered = order.iter().map(|&index| slots[index].take().expect("unique index")).collect();
        self.rebuild_order();
    }
    pub fn swap(&mut self, a: usize, b: usize) {
        self.index.take();
        self.ordered.swap(a, b);
        self.rebuild_order();
    }
    pub fn get_mut(&mut self, index: usize) -> Option<&mut SceneObject> {
        self.index.take();
        self.ordered.get_mut(index)
    }
    pub fn last_mut(&mut self) -> Option<&mut SceneObject> { self.index.take(); self.ordered.last_mut() }
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, SceneObject> {
        self.index.take();
        self.ordered.iter_mut()
    }

    /// There is no authoring metadata copy here. Only unfinished import commands
    /// and derived transform/activation propagation remain at a system boundary.
    pub(crate) fn propagate_hierarchy(&mut self) -> Result<(), HierarchyError> {
        for entity in self.order.pending() { self.resolve_parent(entity, true); }
        self.order.clear_waiting();
        self.world.propagate()
    }
}

impl Deref for SceneObjects {
    type Target = [SceneObject];
    fn deref(&self) -> &Self::Target { &self.ordered }
}
impl<I: SliceIndex<[SceneObject]>> Index<I> for SceneObjects {
    type Output = I::Output;
    fn index(&self, index: I) -> &Self::Output { &self.ordered[index] }
}
impl IndexMut<usize> for SceneObjects {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output { self.index.take(); &mut self.ordered[index] }
}
impl Extend<SceneObject> for SceneObjects {
    fn extend<I: IntoIterator<Item = SceneObject>>(&mut self, values: I) { for object in values { self.push(object); } }
}
impl FromIterator<SceneObject> for SceneObjects {
    fn from_iter<I: IntoIterator<Item = SceneObject>>(values: I) -> Self { let mut objects = Self::new(); objects.extend(values); objects }
}
impl From<Vec<SceneObject>> for SceneObjects { fn from(values: Vec<SceneObject>) -> Self { values.into_iter().collect() } }
impl<'a> IntoIterator for &'a SceneObjects {
    type Item = &'a SceneObject;
    type IntoIter = std::slice::Iter<'a, SceneObject>;
    fn into_iter(self) -> Self::IntoIter { self.ordered.iter() }
}
impl<'a> IntoIterator for &'a mut SceneObjects {
    type Item = &'a mut SceneObject;
    type IntoIter = std::slice::IterMut<'a, SceneObject>;
    fn into_iter(self) -> Self::IntoIter { self.iter_mut() }
}
impl IntoIterator for SceneObjects {
    type Item = SceneObject;
    type IntoIter = std::vec::IntoIter<SceneObject>;
    fn into_iter(mut self) -> Self::IntoIter { self.drain(..) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_core::{Bob, ComponentRegistry};

    #[test]
    fn snapshot_restore_keeps_entities_and_replaces_components_in_place() {
        let registry = ComponentRegistry::with_builtins();
        let mut scene = super::super::LoadedScene::empty();
        let id = ObjectId::new();
        let mut object = SceneObject::new_empty(id, "Before");
        object.components.push(Box::new(Bob::default()));
        scene.objects.push(object);
        let entity = scene.objects.world.entity(id).unwrap();
        let snapshot = scene.snapshot_objects();
        *scene.objects[0].name_mut() = "After".into();
        scene.objects[0].components.get_typed_mut::<Bob>().unwrap().amplitude = 9.0;
        scene.restore_objects(&snapshot, &registry);
        assert_eq!(scene.objects.world.entity(id), Some(entity));
        assert_eq!(scene.objects.world.get::<zestengine_core::ecs::Name>(entity).unwrap().0, "Before");
        assert_eq!(scene.objects.world.get::<Bob>(entity).unwrap().amplitude, Bob::default().amplitude);
        assert_eq!(scene.objects[0].components.len(), 1);
    }

    #[test]
    fn taking_and_reordering_rows_keeps_component_addresses_and_entity_handles() {
        let mut objects = SceneObjects::new();
        let mut a = SceneObject::new_empty(ObjectId::new(), "A");
        a.components.push(Box::new(Bob::default()));
        let id = a.id();
        objects.push(a);
        objects.push(SceneObject::new_empty(ObjectId::new(), "B"));
        let entity = objects.world.entity(id).unwrap();
        let address = { let bob = objects.world.get::<Bob>(entity).unwrap(); (&*bob) as *const Bob };
        let mut rows = objects.take_rows();
        rows.reverse();
        objects.replace_rows(rows);
        assert_eq!(objects.world.entity(id), Some(entity));
        assert_eq!((&*objects.world.get::<Bob>(entity).unwrap()) as *const Bob, address);
        assert_eq!(objects[1].id(), id);
        objects.clear();
        assert!(objects.world.entity(id).is_none());
        objects.push(SceneObject::new_empty(id, "Recreated"));
        assert_ne!(objects.world.entity(id), Some(entity));
    }
}

#[cfg(test)]
mod transform_authority_tests {
    use super::*;
    use glam::{Quat, Vec3};
    use zestengine_core::ComponentRegistry;
    use crate::scene::LoadedScene;

    #[test]
    fn editor_and_native_transform_edits_are_immediately_shared_without_a_stage_bridge() {
        let mut scene = LoadedScene::empty();
        let id = ObjectId::new();
        scene.objects.push(SceneObject::new_empty(id, "Object").with_translation(Vec3::X));
        let entity = scene.objects.world.entity(id).unwrap();
        scene.objects[0].transform_mut().translation = Vec3::Y * 2.0;
        assert_eq!(scene.objects.world.get::<Transform>(entity).unwrap().translation, Vec3::Y * 2.0);
        scene.objects.world.with_world(|world| world.get_mut::<Transform>(entity).unwrap().translation = Vec3::Z * 3.0);
        assert_eq!(scene.objects[0].translation(), Vec3::Z * 3.0);
        scene.objects.propagate_hierarchy().unwrap();
        assert_eq!(scene.objects[0].translation(), Vec3::Z * 3.0, "metadata sync must not overwrite native transforms");
    }

    #[test]
    fn names_and_enabled_state_are_shared_with_native_systems_and_queries_without_sync() {
        use zestengine_core::{AudioListener, LightComponent};
        use zestengine_core::ecs::{Enabled, Name};
        let mut scene = LoadedScene::empty();
        let id = ObjectId::new();
        let mut parent = SceneObject::new_empty(id, "Parent");
        parent.components.push(Box::new(LightComponent::default()));
        let mut child = SceneObject::new_empty(ObjectId::new(), "Child");
        child.set_parent(Some(0));
        child.components.push(Box::new(AudioListener::default()));
        scene.objects.extend([parent, child]);
        let entity = scene.objects.world.entity(id).unwrap();
        *scene.objects[0].name_mut() = "Edited".into();
        scene.objects[0].set_enabled(false);
        assert_eq!(scene.objects.world.get::<Name>(entity).unwrap().0, "Edited");
        assert!(!scene.objects.world.get::<Enabled>(entity).unwrap().0);
        assert!(scene.gather_lights_all().is_empty());
        assert!(scene.audio_listener().is_none());
        scene.objects.world.with_world(|world| {
            world.get_mut::<Name>(entity).unwrap().0 = "Native".into();
            world.get_mut::<Enabled>(entity).unwrap().0 = true;
        });
        assert_eq!(scene.objects[0].name(), "Native");
        assert!(scene.objects[0].enabled());
        assert_eq!(scene.gather_lights_all().len(), 1);
        assert_eq!(scene.audio_listener(), Some(Vec3::ZERO));
        scene.objects.propagate_hierarchy().unwrap();
        assert_eq!(scene.objects[0].name(), "Native", "metadata sync must not overwrite native names");
        assert!(scene.objects[0].enabled(), "metadata sync must not overwrite native activation");
    }

    #[test]
    fn snapshot_restore_and_cross_scene_moves_preserve_names_and_enabled_state() {
        let mut scene = LoadedScene::empty();
        let id = ObjectId::new();
        scene.objects.push(SceneObject::new_empty(id, "Saved").with_enabled(false));
        let entity = scene.objects.world.entity(id).unwrap();
        let snapshot = scene.snapshot_objects();
        *scene.objects[0].name_mut() = "Changed".into();
        scene.objects[0].set_enabled(true);
        scene.restore_objects(&snapshot, &ComponentRegistry::with_builtins());
        assert_eq!(scene.objects.world.entity(id), Some(entity));
        assert_eq!(scene.objects[0].name(), "Saved");
        assert!(!scene.objects[0].enabled());
        let object = scene.objects.remove(0);
        assert_eq!(object.name(), "Saved");
        assert!(!object.enabled());
        let mut other = SceneObjects::new();
        other.push(object);
        assert_eq!(other[0].name(), "Saved");
        assert!(!other[0].enabled());
    }

    #[test]
    fn undo_restores_transform_on_the_same_entity_even_without_gameplay_components() {
        let mut scene = LoadedScene::empty();
        let id = ObjectId::new();
        let saved = Transform { translation: Vec3::new(1.0, 2.0, 3.0), rotation: Quat::from_rotation_y(0.5), scale: Vec3::splat(2.0) };
        scene.objects.push(SceneObject::new_empty(id, "Object").with_transform(saved));
        let entity = scene.objects.world.entity(id).unwrap();
        let snapshot = scene.snapshot_objects();
        *scene.objects[0].transform_mut() = Transform::default();
        scene.restore_objects(&snapshot, &ComponentRegistry::with_builtins());
        assert_eq!(scene.objects.world.entity(id), Some(entity));
        assert_eq!(scene.objects[0].transform(), saved);
        assert_eq!(*scene.objects.world.get::<Transform>(entity).unwrap(), saved);
    }

    #[test]
    fn detaching_and_reinserting_an_object_preserves_the_authoritative_transform() {
        let mut first = SceneObjects::new();
        let id = ObjectId::new();
        first.push(SceneObject::new_empty(id, "Object").with_scale(Vec3::splat(3.0)));
        first[0].transform_mut().translation = Vec3::X * 7.0;
        let saved = first[0].transform();
        let object = first.remove(0);
        assert_eq!(object.transform(), saved);
        let mut second = SceneObjects::new();
        second.push(object);
        assert_eq!(second[0].transform(), saved);
        assert_eq!(*second.world.get::<Transform>(second.world.entity(id).unwrap()).unwrap(), saved);
    }
}

#[cfg(test)]
mod metadata_authority_tests {
    use super::*;
    use crate::scene::{LoadedScene, RenderInfo, metadata};
    use zestengine_core::{ComponentRegistry, NetRole};

    fn values(object: &SceneObject) -> (u8, bool, NetRole, Option<(usize, usize)>) {
        (object.layer(), object.static_geometry(), object.net_role(), object.render().map(|info| (info.mesh, info.material)))
    }

    #[test]
    fn native_and_editor_metadata_edits_share_the_same_values_without_sync() {
        let mut scene = LoadedScene::empty();
        scene.objects.push(SceneObject::new_empty(ObjectId::new(), "Object"));
        let entity = scene.objects[0].entity();
        *scene.objects[0].layer_mut() = 3;
        *scene.objects[0].static_geometry_mut() = true;
        *scene.objects[0].net_role_mut() = NetRole::Server;
        *scene.objects[0].render_mut() = Some(RenderInfo { mesh: 1, material: 2 });
        scene.objects.query(|world, _| {
            assert_eq!(world.get::<metadata::Layer>(entity).unwrap().0, 3);
            assert!(world.get::<metadata::StaticGeometry>(entity).unwrap().0);
            assert_eq!(world.get::<metadata::NetRole>(entity).unwrap().0, NetRole::Server);
            assert_eq!(world.get::<metadata::Render>(entity).unwrap().0.unwrap().material, 2);
            world.get_mut::<metadata::Layer>(entity).unwrap().0 = 7;
            world.get_mut::<metadata::StaticGeometry>(entity).unwrap().0 = false;
            world.get_mut::<metadata::NetRole>(entity).unwrap().0 = NetRole::Client;
            world.get_mut::<metadata::Render>(entity).unwrap().0 = Some(RenderInfo { mesh: 4, material: 5 });
        });
        assert_eq!(values(&scene.objects[0]), (7, false, NetRole::Client, Some((4, 5))));
        scene.objects.propagate_hierarchy().unwrap();
        assert_eq!(values(&scene.objects[0]), (7, false, NetRole::Client, Some((4, 5))));
    }

    #[test]
    fn snapshots_and_cross_scene_moves_preserve_typed_metadata_and_entity_identity() {
        let mut scene = LoadedScene::empty();
        let id = ObjectId::new();
        scene.objects.push(SceneObject::new_empty(id, "Object").with_layer(3).with_static_geometry(true)
            .with_net_role(NetRole::Server).with_render(Some(RenderInfo { mesh: 1, material: 2 })));
        let entity = scene.objects[0].entity();
        let saved = values(&scene.objects[0]);
        let snapshot = scene.snapshot_objects();
        *scene.objects[0].layer_mut() = 8;
        *scene.objects[0].render_mut() = None;
        scene.restore_objects(&snapshot, &ComponentRegistry::with_builtins());
        assert_eq!(scene.objects[0].entity(), entity);
        assert_eq!(scene.objects[0].id(), id);
        assert_eq!(values(&scene.objects[0]), saved);
        let object = scene.objects.remove(0);
        assert_eq!(object.id(), id);
        assert_eq!(values(&object), saved);
        let mut other = SceneObjects::new();
        other.push(object);
        assert_eq!(other[0].id(), id);
        assert_eq!(values(&other[0]), saved);
    }

    #[test]
    fn owned_metadata_is_shared_with_native_queries_and_render_slots_keep_order() {
        use zestengine_assets::ObjectSource;
        let mut scene = LoadedScene::empty();
        scene.objects.push(SceneObject::new_empty(ObjectId::new(), "Object")
            .with_source(ObjectSource::Builtin { mesh: 8 })
            .with_render(Some(RenderInfo { mesh: 1, material: 2 }))
            .with_extra_render(vec![RenderInfo { mesh: 3, material: 4 }, RenderInfo { mesh: 5, material: 6 }])
            .with_component_roles(std::collections::BTreeMap::from([("Health".into(), NetRole::Server)])));
        let entity = scene.objects[0].entity();
        scene.objects.query(|world, _| {
            assert!(matches!(world.get::<metadata::Source>(entity).unwrap().0, ObjectSource::Builtin { mesh: 8 }));
            assert_eq!(world.get::<metadata::ExtraRender>(entity).unwrap().0.len(), 2);
            assert_eq!(world.get::<metadata::ComponentRoles>(entity).unwrap().0["Health"], NetRole::Server);
            world.get_mut::<metadata::Source>(entity).unwrap().0 = ObjectSource::Camera;
            world.get_mut::<metadata::ExtraRender>(entity).unwrap().0[1].material = 9;
            world.get_mut::<metadata::ComponentRoles>(entity).unwrap().0.insert("Health".into(), NetRole::Client);
        });
        assert!(matches!(*scene.objects[0].source(), ObjectSource::Camera));
        assert_eq!(scene.objects[0].component_roles()["Health"], NetRole::Client);
        let allocation = scene.objects[0].extra_render().as_ptr();
        let slots = scene.objects[0].render_slots();
        assert_eq!(slots.size_hint(), (3, Some(3)));
        assert_eq!(slots.map(|info| (info.mesh, info.material)).collect::<Vec<_>>(), [(1, 2), (3, 4), (5, 9)]);
        assert_eq!(scene.objects[0].extra_render().as_ptr(), allocation);
        *scene.objects[0].render_mut() = None;
        assert_eq!(scene.objects[0].render_slots().map(|info| info.mesh).collect::<Vec<_>>(), [3, 5]);
        *scene.objects[0].source_mut() = ObjectSource::Light;
        scene.objects[0].extra_render_mut().push(RenderInfo { mesh: 10, material: 11 });
        scene.objects[0].component_roles_mut().remove("Health");
        scene.objects.query(|world, _| {
            assert!(matches!(world.get::<metadata::Source>(entity).unwrap().0, ObjectSource::Light));
            assert_eq!(world.get::<metadata::ExtraRender>(entity).unwrap().0.len(), 3);
            assert!(world.get::<metadata::ComponentRoles>(entity).unwrap().0.is_empty());
        });
    }

    #[test]
    fn owned_metadata_survives_undo_and_moves_between_worlds_without_copying_buffers() {
        use zestengine_assets::ObjectSource;
        let mut scene = LoadedScene::empty();
        let id = ObjectId::new();
        scene.objects.push(SceneObject::new_empty(id, "Object")
            .with_source(ObjectSource::Model { path: "models/mesh.glb".into(), mesh: 1, extra_meshes: vec![2],
                model: None, mesh_id: None, extra_mesh_ids: Vec::new() })
            .with_extra_render(vec![RenderInfo { mesh: 2, material: 3 }])
            .with_component_roles(std::collections::BTreeMap::from([("Health".into(), NetRole::Server)])));
        let entity = scene.objects[0].entity();
        let snapshot = scene.snapshot_objects();
        *scene.objects[0].source_mut() = ObjectSource::Empty;
        scene.objects[0].extra_render_mut().clear();
        scene.objects[0].component_roles_mut().clear();
        scene.restore_objects(&snapshot, &ComponentRegistry::with_builtins());
        assert_eq!(scene.objects[0].entity(), entity);
        assert!(matches!(&*scene.objects[0].source(), ObjectSource::Model { path, .. } if path == "models/mesh.glb"));
        assert_eq!(scene.objects[0].extra_render()[0].material, 3);
        assert_eq!(scene.objects[0].component_roles()["Health"], NetRole::Server);
        let allocation = scene.objects[0].extra_render().as_ptr();
        let object = scene.objects.remove(0);
        assert_eq!(object.extra_render().as_ptr(), allocation);
        let mut other = SceneObjects::new();
        other.push(object);
        assert_eq!(other[0].id(), id);
        assert_eq!(other[0].extra_render().as_ptr(), allocation);
        assert!(matches!(&*other[0].source(), ObjectSource::Model { path, .. } if path == "models/mesh.glb"));
        assert_eq!(other[0].component_roles()["Health"], NetRole::Server);
    }

    #[test]
    fn id_changes_update_views_and_both_indices_atomically_and_reject_collisions() {
        let mut objects = SceneObjects::new();
        let a = ObjectId::new();
        let b = ObjectId::new();
        objects.extend([SceneObject::new_empty(a, "A"), SceneObject::new_empty(b, "B")]);
        let entity = objects[0].entity();
        assert_eq!(objects.index_of(a), Some(0));
        assert_eq!(objects.set_id(0, b), Err(HierarchyError::DuplicateId));
        assert_eq!(objects[0].id(), a);
        assert_eq!(objects.world.entity(a), Some(entity));
        assert_eq!(objects.index_of(a), Some(0));
        let next = ObjectId::new();
        objects.set_id(0, next).unwrap();
        assert_eq!(objects[0].entity(), entity);
        assert_eq!(objects[0].id(), next);
        assert!(objects.index_of(a).is_none());
        assert!(objects.world.entity(a).is_none());
        assert_eq!(objects.index_of(next), Some(0));
        assert_eq!(objects.world.entity(next), Some(entity));
        objects.query(|world, indices| {
            assert_eq!(world.object_id(entity), Some(next));
            assert_eq!(indices.get(&next), Some(&0));
        });
    }
}

#[cfg(test)]
mod hierarchy_authority_tests {
    use super::*;
    use crate::scene::LoadedScene;
    use glam::Vec3;
    use zestengine_core::{AudioListener, Bob, ComponentRegistry};

    #[test]
    fn native_parent_edits_and_tree_reorders_share_links_without_metadata_copyback() {
        let mut scene = LoadedScene::empty();
        let mut child = SceneObject::new_empty(ObjectId::new(), "Child").with_parent(Some(0)).with_translation(Vec3::X);
        child.components.push(Box::new(AudioListener::default()));
        scene.objects.extend([
            SceneObject::new_empty(ObjectId::new(), "A").with_translation(Vec3::X * 10.0),
            SceneObject::new_empty(ObjectId::new(), "B").with_translation(Vec3::X * 20.0),
            child,
        ]);
        let a = scene.objects[0].entity();
        let b = scene.objects[1].entity();
        let child = scene.objects[2].entity();
        assert_eq!(scene.objects.world.get::<Parent>(child).unwrap().0, Some(a));
        assert_eq!(scene.audio_listener(), Some(Vec3::X * 11.0));
        scene.objects.query(|world, _| {
            world.get_mut::<Parent>(child).unwrap().0 = Some(b);
            assert_eq!(scene.world_transform_in(world, 2).w_axis.truncate(), Vec3::X * 21.0);
        });
        assert_eq!(scene.objects[2].parent(), Some(1));
        assert_eq!(scene.world_transform(2).w_axis.truncate(), Vec3::X * 21.0);
        scene.objects.propagate_hierarchy().unwrap();
        assert_eq!(scene.objects.world.get::<Parent>(child).unwrap().0, Some(b));
        scene.objects[1].set_enabled(false);
        assert!(scene.audio_listener().is_none());
        scene.objects[1].set_enabled(true);
        scene.objects.reorder(&[2, 0, 1]);
        assert_eq!(scene.objects[0].entity(), child);
        assert_eq!(scene.objects[0].parent(), Some(2));
        assert_eq!(scene.audio_listener(), Some(Vec3::X * 21.0));
        let renamed = ObjectId::new();
        scene.objects.set_id(2, renamed).unwrap();
        assert_eq!(scene.objects.world.entity(renamed), Some(b));
        assert_eq!(scene.objects[0].parent(), Some(2));
        assert_eq!(scene.objects.world.get::<Parent>(child).unwrap().0, Some(b));
        scene.remove_object(1);
        assert_eq!(scene.objects.len(), 2);
        assert_eq!(scene.objects[0].parent(), Some(1));
        assert_eq!(scene.objects.world.get::<Parent>(child).unwrap().0, Some(b));
        scene.remove_object(1);
        assert!(scene.objects.is_empty(), "deleting a parent removes its native descendants");
    }

    #[test]
    fn forward_parents_and_prefab_resource_rows_preserve_existing_entities_and_state() {
        let mut objects = SceneObjects::new();
        let mut child = SceneObject::new_empty(ObjectId::new(), "Child").with_parent(Some(2));
        child.components.push(Box::new(Bob::default()));
        let id = child.id();
        objects.push(child);
        objects.push(SceneObject::new_empty(ObjectId::new(), "Unrelated"));
        objects.push(SceneObject::new_empty(ObjectId::new(), "Parent"));
        let child = objects[0].entity();
        let parent = objects[2].entity();
        assert_eq!(objects.world.get::<Parent>(child).unwrap().0, Some(parent));
        let mut rows = objects.take_rows();
        assert_eq!(rows[0].parent(), Some(2));
        objects.push(SceneObject::new_empty(ObjectId::new(), "Temporary resource"));
        assert_eq!(rows[0].parent(), Some(2), "pooled rows retain their own old order");
        assert!(!objects.has_component::<Bob>(), "native selection excludes pooled resource owners");
        objects.truncate(0);
        assert_eq!(objects.world.entity(id), Some(child), "resource cleanup must retain pooled entities");
        objects.world.get_mut::<Bob>(child).unwrap().amplitude = 42.0;
        rows.reverse();
        objects.replace_rows(rows);
        assert_eq!(objects[2].entity(), child);
        assert_eq!(objects[2].parent(), Some(0));
        assert_eq!(objects.world.get::<Parent>(child).unwrap().0, Some(parent));
        assert_eq!(objects[2].components.get_typed::<Bob>().unwrap().amplitude, 42.0);
    }

    #[test]
    fn detached_children_follow_saved_parent_identity_after_order_changes_and_cross_world_moves() {
        let mut objects = SceneObjects::new();
        let parent_id = ObjectId::new();
        objects.push(SceneObject::new_empty(parent_id, "Parent"));
        objects.push(SceneObject::new_empty(ObjectId::new(), "Child").with_parent(Some(0)));
        let child = objects.remove(1);
        objects.insert(0, SceneObject::new_empty(ObjectId::new(), "Other"));
        objects.push(child);
        assert_eq!(objects[2].parent(), Some(1));
        let child = objects.remove(2);
        let mut next = SceneObjects::new();
        next.push(child);
        next.push(SceneObject::new_empty(parent_id, "Parent in new world"));
        assert_eq!(next[0].parent(), Some(1));
        assert_eq!(next.world.get::<Parent>(next[0].entity()).unwrap().0, Some(next[1].entity()));
    }

    #[test]
    fn imported_parent_indices_commit_against_new_order_and_undo_restores_native_links() {
        let mut scene = LoadedScene::empty();
        scene.objects.extend([
            SceneObject::new_empty(ObjectId::new(), "A"),
            SceneObject::new_empty(ObjectId::new(), "B").with_parent(Some(0)),
            SceneObject::new_empty(ObjectId::new(), "C"),
        ]);
        let entities: Vec<_> = scene.objects.iter().map(SceneObject::entity).collect();
        let snapshot = scene.snapshot_objects();
        let mut rows = scene.objects.take_rows();
        rows.rotate_right(1);
        rows[0].stage_parent_for_import(None);
        rows[1].stage_parent_for_import(Some(2));
        rows[2].stage_parent_for_import(Some(0));
        scene.objects.replace_rows(rows);
        assert_eq!(scene.objects[1].parent(), Some(2));
        assert_eq!(scene.objects.world.get::<Parent>(entities[0]).unwrap().0, Some(entities[1]));
        assert_eq!(scene.objects.world.get::<Parent>(entities[1]).unwrap().0, Some(entities[2]));
        scene.restore_objects(&snapshot, &ComponentRegistry::with_builtins());
        assert_eq!(scene.objects.iter().map(SceneObject::entity).collect::<Vec<_>>(), entities);
        assert_eq!(scene.objects[0].parent(), None);
        assert_eq!(scene.objects[1].parent(), Some(0));
        assert_eq!(scene.objects[2].parent(), None);
    }
}
