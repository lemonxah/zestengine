//! Typed ECS storage and the native system API shared by games, the editor,
//! headless servers and plugins.
//!
//! **New here? Read the [ECS Guide](https://lemonxah.github.io/zestengine/docs/ECS.html)**, which explains components,
//! systems, stages, commands and events with tested examples, and the
//! [Engine API Guide](https://lemonxah.github.io/zestengine/docs/ENGINE_API.html) for what a system can do (move,
//! spawn, raycast, play sounds, save, network).
//!
//! # The model
//!
//! - A scene **object** is a hecs entity. Its saved identity is an
//!   [`ObjectId`]; the hecs [`Entity`] is only a runtime handle.
//! - A **component** is a [`TypedComponent`] value in a typed column. The first
//!   component of each type lives on the object's own entity, so ordinary joined
//!   queries work. Further instances of the same type live on small auxiliary
//!   entities carrying an [`Owner`] that points back to the object.
//! - A **system** is a [`SystemFn`]: it queries a [`SystemWorld`] and receives a
//!   [`SystemContext`] with time, input, services and a command queue. Systems
//!   are registered on the [`ComponentRegistry`] and
//!   run by [`Stage`].
//! - An **entity template** ([`EntityTemplate`]) is a registered, named, ready-made
//!   object that the editor lists under Create > Entity and systems spawn by name.
//!
//! # Borrowing
//!
//! Native systems use hecs' checked column borrows: two borrows of one component
//! type conflict when either is mutable, and the conflict panics. While a system
//! runs, the owning [`SceneWorld`] holds exclusive structural access, so
//! inserting, removing or despawning must be queued as a
//! [`ComponentCommand`] instead. Editor-style component
//! views ([`ComponentRef`], [`ComponentMut`]) use checked entity/type leases and
//! hold a shared structural lock, which is why the two access modes cannot overlap.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use glam::Mat4;

use crate::{ComponentCommand, ComponentRegistry, InputState, ObjectId, Transform, TypedComponent};

pub use hecs::{Component as DataComponent, Entity, Query, QueryBorrow, QueryMut, Ref, RefMut};
mod storage;
pub use storage::{ComponentMut, ComponentRef};
mod components;
pub use components::{ComponentSet, ComponentType, ComponentHandle, ComponentSlot, ObjectMetadata};mod reflection;
pub use reflection::{ComponentIndex, NativeActionCall, NativeProperties, invoke_action_in};
pub(crate) use reflection::run_with_properties;
#[cfg(test)]
pub(crate) use reflection::tests::Fixture as NativeTestFixture;
mod disabled;
pub use disabled::DisabledComponent;
mod movement;
pub use movement::ObjectContext;

/// The display name of an object (the scene tree label). Cosmetic: names are not
/// unique and may be empty, so refer to objects by [`ObjectId`] or
/// [`ObjectRef`](crate::ObjectRef) instead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Name(pub String);

/// The object this one is parented to (`None` for a root). The parent link is
/// authoritative data; the scene tree's order is derived from it. Change it
/// through [`SceneWorld::set_parent`] so cycles are rejected and the derived
/// transforms are refreshed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Parent(pub Option<Entity>);

/// An object's own on/off switch (the checkbox in the inspector). It says nothing
/// about the ancestors: [`Active`] is the inherited result that systems should
/// respect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Enabled(pub bool);

/// Whether an object is running: its [`Enabled`] flag **and** every ancestor's.
/// Computed by [`SceneWorld::propagate`] at stage boundaries, so it is a snapshot;
/// [`SystemWorld::current_active`] answers for the present moment. Systems should
/// skip objects that are not active (see [`SystemContext::is_active`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Active(pub bool);

/// An object's world matrix as of the last [`SceneWorld::propagate`] (the engine
/// refreshes it at every stage boundary). It does **not** include transform writes
/// made earlier in the current stage; [`SystemWorld::world_transform`] does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlobalTransform(pub Mat4);

/// The object that a component instance belongs to. Object entities own
/// themselves; the auxiliary entity holding a *second* instance of a component
/// type points back at its object.
///
/// This is why systems query `(&Owner, &T)`: `Owner` is the way from a component
/// to its object's [`Transform`], [`Active`] flag and
/// [`ObjectId`].
///
/// ```ignore
/// for (_, (owner, health)) in world.query::<(&Owner, &mut Health)>().iter() {
///     if !ctx.is_active(world, owner.0) { continue; }
///     let id = world.object_id(owner.0);
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Owner(pub Entity);

/// Why a structural scene operation was refused. Nothing changes when one is
/// returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HierarchyError {
    /// The entity is not a scene object in this world (never spawned, or already removed).
    MissingEntity,
    /// Another object already has this saved id.
    DuplicateId,
    /// Scene objects need a non-nil saved id.
    InvalidId,
    /// Parenting would make an object its own ancestor.
    Cycle,
    /// A world transform cannot be preserved under a parent whose matrix cannot be inverted
    /// (for example a zero scale).
    SingularParent,
}

impl fmt::Display for HierarchyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::MissingEntity => "entity is not a scene object in this world",
            Self::DuplicateId => "another entity already has this saved object id",
            Self::InvalidId => "scene objects require a nonzero saved object id",
            Self::Cycle => "parenting would create a hierarchy cycle",
            Self::SingularParent => "cannot preserve a world transform under a singular parent transform",
        })
    }
}

impl std::error::Error for HierarchyError {}

/// Owns a scene's ECS world and the index from saved [`ObjectId`]s to entities.
///
/// All **structural** operations (spawn, parent, insert and remove components,
/// despawn) go through this type, so deleting an object can never leave a saved
/// reference pointing at a reused slot. It is what editors, loaders and tests
/// hold; game systems never see it. They receive a [`SystemWorld`] and queue
/// [`ComponentCommand`]s instead.
///
/// ```ignore
/// let mut scene = SceneWorld::new();
/// let player = scene.spawn("Player");
/// scene.insert(player, Health::default())?;
/// registry.run_systems(Stage::Update, &scene, &mut ctx);
/// assert_eq!(scene.get::<Health>(player).unwrap().current, 100.0);
/// ```
///
/// See [Testing systems](https://lemonxah.github.io/zestengine/docs/ECS.html#testing-systems) in the ECS guide.
#[derive(Default)]
pub struct SceneWorld {
    world: Arc<storage::Storage>,
    objects: HashMap<ObjectId, Entity>,
}

/// The world as a system sees it: queries and checked component access over the
/// scene's permanent typed storage.
///
/// The owning [`SceneWorld`] holds exclusive structural access for the whole run
/// of a system, so nothing can be inserted, removed or despawned underneath you
/// (queue a [`ComponentCommand`] for that) and editor
/// component views are excluded. Within a system, hecs checks column borrows as
/// usual: two borrows of one component type conflict if either is mutable.
///
/// ```ignore
/// fn drain(world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) {
///     for (_, (owner, tank)) in world.query::<(&Owner, &mut FuelTank)>().iter() {
///         if !ctx.is_active(world, owner.0) { continue; }
///         tank.fuel = (tank.fuel - tank.burn_rate * ctx.dt).max(0.0);
///     }
/// }
/// ```
///
/// See [Systems](https://lemonxah.github.io/zestengine/docs/ECS.html#systems) and
/// [Borrowing rules](https://lemonxah.github.io/zestengine/docs/ECS.html#borrowing-rules) in the ECS guide.
pub struct SystemWorld<'a> {
    world: &'a hecs::World,
    objects: &'a HashMap<ObjectId, Entity>,
}

impl SystemWorld<'_> {
    /// Start a hecs query, for example `world.query::<(&Owner, &mut Health)>()`, then
    /// `.iter()` it. Each yielded item is `(Entity, Q::Item)`. Borrows are checked per
    /// component column when the query is created: asking for `&mut T` while another
    /// live borrow of `T` exists panics with "already borrowed".
    ///
    /// Query `(&Owner, &T)` rather than `&T` alone when you need the object a
    /// component belongs to; plain `&T` visits every instance, including duplicates
    /// on auxiliary entities. Iteration order is archetype order, not authoring
    /// order; use [`run_ordered`] when order matters.
    pub fn query<Q: Query>(&self) -> QueryBorrow<'_, Q> { self.world.query::<Q>() }

    /// Borrow component `T` of `entity` for reading. `None` if the entity does not
    /// carry an (enabled) `T`. Panics if `T` of that entity is already mutably borrowed.
    pub fn get<T: DataComponent>(&self, entity: Entity) -> Option<Ref<'_, T>> { self.world.get::<&T>(entity).ok() }

    /// Borrow component `T` of `entity` for writing: the way to edit another
    /// component of the object you are iterating (`world.get_mut::<Transform>(owner.0)`).
    /// `None` if the entity has no `T`. Panics if `T` of that entity is already
    /// borrowed, including by an enclosing query over `T`.
    pub fn get_mut<T: DataComponent>(&self, entity: Entity) -> Option<RefMut<'_, T>> { self.world.get::<&mut T>(entity).ok() }

    /// The runtime entity of the object with this saved id (`None` if it does not exist).
    pub fn entity(&self, id: ObjectId) -> Option<Entity> { self.objects.get(&id).copied() }

    /// The saved id of an object entity (`None` for an entity that is not a scene object).
    pub fn object_id(&self, entity: Entity) -> Option<ObjectId> { self.get::<ObjectId>(entity).map(|id| *id) }

    /// The object's world matrix **right now**: local transforms folded through every
    /// parent, including writes made by earlier systems in this stage (unlike
    /// [`GlobalTransform`], which is a stage-boundary snapshot). `None` if the entity
    /// is not an object or its parent links are malformed (a cycle).
    pub fn world_transform(&self, entity: Entity) -> Option<Mat4> {
        let mut matrix = self.get::<Transform>(entity)?.matrix();
        let mut parent = self.get::<Parent>(entity)?.0;
        for _ in 0..=self.objects.len() {
            let Some(entity) = parent else { return Some(matrix) };
            matrix = self.get::<Transform>(entity)?.matrix() * matrix;
            parent = self.get::<Parent>(entity)?.0;
        }
        None // malformed cyclic native parent links
    }
    /// Whether the object is active **right now**: its [`Enabled`] flag and every
    /// ancestor's, read live rather than from the [`Active`] snapshot. Prefer
    /// [`SystemContext::is_active`] in ordinary systems; this is for code that has
    /// just changed an enabled flag.
    pub fn current_active(&self, entity: Entity) -> bool {
        let mut current = Some(entity);
        for _ in 0..=self.objects.len() {
            let Some(entity) = current else { return true };
            if !self.get::<Enabled>(entity).is_some_and(|value| value.0) { return false; }
            let Some(parent) = self.get::<Parent>(entity) else { return false };
            current = parent.0;
        }
        false
    }
}

impl SceneWorld {
    /// An empty scene.
    pub fn new() -> Self { Self::default() }

    /// The number of scene **objects** (component instances on auxiliary entities do not count).
    pub fn len(&self) -> usize { self.objects.len() }

    /// Whether the scene has no objects.
    pub fn is_empty(&self) -> bool { self.objects.is_empty() }

    /// The runtime entity of the object with this saved id, or `None` if there is
    /// no such object (or it was deleted).
    pub fn entity(&self, id: ObjectId) -> Option<Entity> { self.objects.get(&id).copied() }

    /// The saved id of an object entity, or `None` for a non-object entity (such as
    /// the auxiliary entity of a duplicate component).
    pub fn object_id(&self, entity: Entity) -> Option<ObjectId> {
        self.get::<ObjectId>(entity).map(|id| *id)
    }

    /// Create an enabled root object with a fresh random [`ObjectId`] and an identity
    /// transform. Returns its entity. Add components with [`SceneWorld::insert`].
    pub fn spawn(&mut self, name: impl Into<String>) -> Entity {
        self.spawn_with_id(ObjectId::new(), name, Transform::default(), true).expect("fresh object id")
    }

    /// Create a root object with a given saved id, local transform and enabled state:
    /// what scene loading does. Fails with [`HierarchyError::InvalidId`] for the nil
    /// id and [`HierarchyError::DuplicateId`] if the id is taken.
    pub fn spawn_with_id(
        &mut self, id: ObjectId, name: impl Into<String>, transform: Transform, enabled: bool,
    ) -> Result<Entity, HierarchyError> {
        if id.is_nil() { return Err(HierarchyError::InvalidId); }
        if self.objects.contains_key(&id) { return Err(HierarchyError::DuplicateId); }
        let mut world = self.world.exclusive();
        let entity = world.spawn((id, Name(name.into()), transform, Parent(None), Enabled(enabled), Active(enabled), GlobalTransform(transform.matrix())));
        world.insert_one(entity, Owner(entity)).expect("new entity");
        self.objects.insert(id, entity);
        Ok(entity)
    }

    /// Change a saved identity without moving components or invalidating the
    /// entity. The map and ECS metadata change together, before deletion or
    /// replication can observe the new identity.
    pub fn set_object_id(&mut self, entity: Entity, id: ObjectId) -> Result<(), HierarchyError> {
        if id.is_nil() { return Err(HierarchyError::InvalidId); }
        let old = self.object_id(entity).ok_or(HierarchyError::MissingEntity)?;
        if old == id { return Ok(()); }
        if self.objects.contains_key(&id) { return Err(HierarchyError::DuplicateId); }
        *self.get_mut::<ObjectId>(entity).expect("object identity") = id;
        self.objects.remove(&old);
        self.objects.insert(id, entity);
        Ok(())
    }

    /// Run `run` with a [`SystemWorld`] over this scene: the same view a system
    /// receives. The world is locked exclusively for the duration, so component
    /// views ([`SceneWorld::get`]) must be released first and cannot be taken inside
    /// `run` (both panic). Use it for ad-hoc queries in tools and tests.
    ///
    /// ```ignore
    /// let total: f32 = scene.with_world(|world| {
    ///     world.query::<&Health>().iter().map(|(_, hp)| hp.current).sum()
    /// });
    /// ```
    pub fn with_world<R>(&self, run: impl FnOnce(&SystemWorld<'_>) -> R) -> R {
        let world = self.world.exclusive();
        run(&SystemWorld { world: &world, objects: &self.objects })
    }

    /// Borrow one component of one entity for reading. `None` if the entity has no
    /// `T`. The returned view holds a shared structural lock and a borrow lease on
    /// that slot until it is dropped: a conflicting borrow of the same slot, or any
    /// structural change, panics while it is alive.
    ///
    /// Not available inside a running system (use the [`SystemWorld`] you were given).
    pub fn get<T: DataComponent>(&self, entity: Entity) -> Option<ComponentRef<'_, T>> {
        self.world.read::<T>(entity)
    }

    /// Borrow one component of one entity for writing. Same rules as
    /// [`SceneWorld::get`], with an exclusive lease on the slot. Independent
    /// entities can be borrowed mutably at the same time.
    pub fn get_mut<T: DataComponent>(&self, entity: Entity) -> Option<ComponentMut<'_, T>> {
        self.world.write::<T>(entity)
    }

    /// Attach typed gameplay data. A second instance of the same type gets a
    /// separate entity, with the same Owner. Metadata types are not accepted.
    pub fn insert<T: TypedComponent + Send + std::marker::Sync>(&mut self, owner: Entity, value: T) -> Result<Entity, HierarchyError> {
        if self.object_id(owner).is_none() { return Err(HierarchyError::MissingEntity); }
        let mut world = self.world.exclusive();
        if world.satisfies::<&T>(owner).unwrap_or(false) {
            Ok(world.spawn((Owner(owner), value)))
        } else {
            world.insert_one(owner, value).expect("live owner");
            Ok(owner)
        }
    }

    /// Attach a disabled payload inline. Ordinary T queries continue to select
    /// enabled instances; `DisabledComponent<T>` queries expose this data directly.
    pub fn insert_disabled<T: TypedComponent>(&mut self, owner: Entity, value: T) -> Result<Entity, HierarchyError> {
        if self.object_id(owner).is_none() { return Err(HierarchyError::MissingEntity); }
        let mut world = self.world.exclusive();
        let value = DisabledComponent::new(value);
        if world.satisfies::<&DisabledComponent<T>>(owner).unwrap_or(false) {
            Ok(world.spawn((Owner(owner), value)))
        } else {
            world.insert_one(owner, value).expect("live owner");
            Ok(owner)
        }
    }

    /// Remove one *disabled* component instance (as inserted by
    /// [`SceneWorld::insert_disabled`]) and return its value. The owning object stays.
    pub fn remove_disabled<T: TypedComponent>(&mut self, instance: Entity) -> Option<T> {
        let owner = self.get::<Owner>(instance)?.0;
        let mut world = self.world.exclusive();
        let value = world.remove_one::<DisabledComponent<T>>(instance).ok()?;
        if owner != instance { world.despawn(instance).expect("live component entity"); }
        self.world.forget(&[instance]);
        Some(value.into_value())
    }

    /// Remove one component instance, retaining the owning scene object.
    pub fn remove<T: TypedComponent + Send + std::marker::Sync>(&mut self, instance: Entity) -> Option<T> {
        let owner = self.get::<Owner>(instance)?.0;
        let mut world = self.world.exclusive();
        let value = world.remove_one::<T>(instance).ok()?;
        if owner != instance { world.despawn(instance).expect("live component entity"); }
        self.world.forget(&[instance]);
        Some(value)
    }

    /// Reparent an object, optionally preserving its world-space TRS. Rejects
    /// cycles before changing anything. Children keep following their parent.
    pub fn set_parent(&mut self, child: Entity, parent: Option<Entity>, keep_world: bool) -> Result<(), HierarchyError> {
        if self.object_id(child).is_none() { return Err(HierarchyError::MissingEntity); }
        let mut ancestor = parent;
        let mut visited = HashSet::new();
        while let Some(entity) = ancestor {
            if entity == child || !visited.insert(entity) { return Err(HierarchyError::Cycle); }
            ancestor = self.get::<Parent>(entity).ok_or(HierarchyError::MissingEntity)?.0;
        }
        let local = if keep_world {
            self.propagate()?;
            let global = self.get::<GlobalTransform>(child).expect("object metadata").0;
            let parent_global = parent.and_then(|p| self.get::<GlobalTransform>(p).map(|g| g.0)).unwrap_or(Mat4::IDENTITY);
            let determinant = parent_global.determinant();
            if !parent_global.is_finite() || !determinant.is_finite() || determinant.abs() < 1e-12 {
                return Err(HierarchyError::SingularParent);
            }
            Some(Transform::from_matrix(parent_global.inverse() * global))
        } else { None };
        self.get_mut::<Parent>(child).expect("object metadata").0 = parent;
        if let Some(local) = local {
            *self.get_mut::<Transform>(child).expect("object metadata") = local;
        }
        self.propagate()
    }

    /// Apply an imported hierarchy in one pass. Validation is atomic: on an
    /// invalid graph, all previous parent links remain intact.
    pub fn set_parents(&mut self, parents: &[(Entity, Option<Entity>)]) -> Result<(), HierarchyError> {
        for &(child, parent) in parents {
            if self.object_id(child).is_none() || parent.is_some_and(|p| self.object_id(p).is_none()) {
                return Err(HierarchyError::MissingEntity);
            }
        }
        let previous: Vec<_> = parents.iter().map(|&(child, _)| (child, self.get::<Parent>(child).expect("object metadata").0)).collect();
        for &(child, parent) in parents { self.get_mut::<Parent>(child).expect("object metadata").0 = parent; }
        if let Err(error) = self.propagate() {
            for (child, parent) in previous.into_iter().rev() { self.get_mut::<Parent>(child).expect("object metadata").0 = parent; }
            return Err(error);
        }
        Ok(())
    }

    /// Resolve transforms and inherited activation in O(objects), without
    /// recursion or a hierarchy-depth limit. Bad parent graphs are rejected.
    pub fn propagate(&mut self) -> Result<(), HierarchyError> {
        let mut world = self.world.exclusive();
        let nodes: HashMap<Entity, (Mat4, Option<Entity>, bool)> = world
            .query_mut::<(&Transform, &Parent, &Enabled)>().into_iter()
            .map(|(entity, (local, parent, enabled))| (entity, (local.matrix(), parent.0, enabled.0))).collect();
        let mut resolved: HashMap<Entity, (Mat4, bool)> = HashMap::with_capacity(nodes.len());
        let mut visiting = HashSet::new();
        let mut chain = Vec::new();
        for &start in nodes.keys() {
            if resolved.contains_key(&start) { continue; }
            chain.clear();
            let mut cursor = Some(start);
            while let Some(entity) = cursor {
                if resolved.contains_key(&entity) { break; }
                if !visiting.insert(entity) { return Err(HierarchyError::Cycle); }
                let node = nodes.get(&entity).ok_or(HierarchyError::MissingEntity)?;
                chain.push(entity);
                cursor = node.1;
            }
            while let Some(entity) = chain.pop() {
                let &(local, parent, enabled) = &nodes[&entity];
                let (parent_world, parent_active) = parent.map(|p| resolved[&p]).unwrap_or((Mat4::IDENTITY, true));
                resolved.insert(entity, (parent_world * local, enabled && parent_active));
                visiting.remove(&entity);
            }
        }
        for (entity, (global, active)) in world.query_mut::<(&mut GlobalTransform, &mut Active)>() {
            let (matrix, enabled) = resolved[&entity];
            global.0 = matrix;
            active.0 = enabled;
        }
        Ok(())
    }

    /// Remove a hierarchy, including duplicate component instances. Saved IDs
    /// are removed too; a reused hecs slot cannot satisfy an old ObjectRef.
    pub fn despawn_recursive(&mut self, root: Entity) -> Result<(), HierarchyError> {
        if self.object_id(root).is_none() { return Err(HierarchyError::MissingEntity); }
        let mut children: HashMap<Entity, Vec<Entity>> = HashMap::new();
        self.with_world(|world| {
            for (entity, parent) in world.query::<&Parent>().iter() {
                if let Some(parent) = parent.0 { children.entry(parent).or_default().push(entity); }
            }
        });
        let mut pending = vec![root];
        let mut removed = HashSet::new();
        while let Some(entity) = pending.pop() {
            if !removed.insert(entity) { continue; }
            pending.extend(children.get(&entity).into_iter().flatten().copied());
        }
        self.remove_objects(&removed);
        Ok(())
    }

    /// Remove one object after its component set has been detached or dropped.
    /// Retained children become roots; scene editing applies its own parent map.
    pub fn despawn(&mut self, entity: Entity) -> Result<(), HierarchyError> {
        if self.object_id(entity).is_none() { return Err(HierarchyError::MissingEntity); }
        self.remove_objects(&HashSet::from([entity]));
        Ok(())
    }

    /// Remove objects missing from an authoring scene during migration. Unlike
    /// hierarchy deletion, retained children are detached rather than deleted.
    pub fn retain_objects(&mut self, ids: &HashSet<ObjectId>) {
        let removed = self.objects.iter().filter_map(|(id, &entity)| (!ids.contains(id)).then_some(entity)).collect();
        self.remove_objects(&removed);
    }

    fn remove_objects(&mut self, removed: &HashSet<Entity>) {
        let mut world = self.world.exclusive();
        for (_, parent) in world.query_mut::<&mut Parent>() {
            if parent.0.is_some_and(|p| removed.contains(&p)) { parent.0 = None; }
        }
        let entities: Vec<_> = world.query_mut::<&Owner>().into_iter()
            .filter_map(|(entity, owner)| removed.contains(&owner.0).then_some(entity)).collect();
        for &entity in &entities { world.despawn(entity).expect("queried entity"); }
        self.world.forget(&entities);
        self.objects.retain(|_, entity| !removed.contains(entity));
    }
}

/// When in the frame a system runs. After each stage the engine applies the
/// [`ComponentCommand`]s that stage queued.
///
/// A frame runs: `Start` (repeating while start-up work spawns more objects),
/// animation markers as `Events`, then `FixedUpdate` zero to eight times at 60 Hz
/// (each followed by a physics step), physics contacts as `Events`, `Update`,
/// `Gameplay`, navigation and animation, and finally `LateUpdate`.
///
/// See [Stages](https://lemonxah.github.io/zestengine/docs/ECS.html#stages) in the ECS guide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Before each physics step, at a fixed 60 Hz: `dt` is 1/60 s whatever the frame
    /// rate, and a slow frame runs it up to eight times. Continuous forces belong here.
    FixedUpdate,
    /// Once per frame, after physics: movement, controllers, AI steering, general logic.
    Update,
    /// After gameplay, navigation and animation: cameras following a target, HUD
    /// bindings, anything that mirrors the frame's final state.
    LateUpdate,
    /// Initialises active new instances, before movement. The engine does not call it
    /// "once per instance": it runs each frame, up to eight times while start-up
    /// commands keep spawning objects, so a start-up system tracks completion in its
    /// own component data (and is best registered with
    /// [`ComponentRegistry::register_system_if`](crate::ComponentRegistry::register_system_if)).
    Start,
    /// After `Update`'s movement commands have been applied, so physics movement has
    /// resolved before weapons, damage, pickups and health read poses.
    Gameplay,
    /// Delivers immediate [`SceneEvent`]s (contacts and animation markers) through
    /// [`SystemServices::scene_events`]. Runs only when there are such events.
    Events,
}

/// The phase of a physics contact between two colliders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContactPhase {
    /// The pair started touching (or a trigger was entered) this step.
    Enter,
    /// The pair is still touching (reported by 2D physics once per frame).
    Stay,
    /// The pair stopped touching (or a trigger was left) this step.
    Exit,
}

/// Events that happen *within* the frame and are delivered to systems in
/// [`Stage::Events`] through [`SystemServices::scene_events`]. Generic events one
/// system emits for others are a different channel: they arrive next frame through
/// [`SystemContext::events`].
#[derive(Clone, Debug)]
pub enum SceneEvent {
    /// Two colliders touched. Each object of a pair receives its own event, with
    /// itself as `object` and the other as `other`.
    Contact {
        /// The receiver of this event.
        object: ObjectId,
        /// The object it touched.
        other: ObjectId,
        /// A trigger overlap (a collider with *Is Trigger*) rather than a solid collision.
        trigger: bool,
        /// Whether the contact began, continues or ended.
        phase: ContactPhase,
        /// From the 2D physics world (XY plane) rather than 3D.
        two_dimensional: bool,
    },
    /// An animation clip marker fired on `object`'s animator.
    Animation {
        /// The object whose animator passed the marker.
        object: ObjectId,
        /// The marker name set on the clip.
        name: String,
    },
}

/// Everything a system needs besides the world: time, input, the command queue and
/// the engine services for this frame.
///
/// Systems edit component data directly and ask for everything else by pushing a
/// [`ComponentCommand`] (or calling a helper such as [`SystemContext::emit`]). The
/// engine applies queued commands, in order, when the stage ends.
///
/// Inside [`run_ordered`] you work through an [`ObjectContext`], which dereferences to
/// this type: `obj.dt`, `obj.input` and `obj.emit(..)` all work.
///
/// ```ignore
/// fn tick(world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) {
///     let speed = ctx.input.axis2("Move").length() * ctx.dt;       // time and input
///     ctx.commands.push(ComponentCommand::Quit);                    // a queued command
///     ctx.emit(LevelComplete);                                      // visible next frame
/// }
/// ```
///
/// See [`SystemContext` in the ECS guide](https://lemonxah.github.io/zestengine/docs/ECS.html#what-a-system-can-see).
pub struct SystemContext<'a> {
    /// Seconds this frame, **scaled by the time scale** (0 while paused). In
    /// [`Stage::FixedUpdate`] this is the fixed step, 1/60 s. A single frame is clamped
    /// to 0.25 s so a hitch cannot teleport everything.
    pub dt: f32,
    /// Seconds this frame ignoring the time scale: use it for pause menus and UI that
    /// must keep running while the game is paused.
    pub unscaled_dt: f32,
    /// Scaled seconds since Play started.
    pub time: f32,
    /// The shared water clock. Networked clients agree on it, which is what makes wave
    /// riders and floating bodies deterministic.
    pub water_time: f32,
    /// This frame's snapshot of the player's actions (`down`, `pressed`, `axis2`, …).
    pub input: &'a InputState,
    /// The command queue. Push [`ComponentCommand`]s here; they are applied after the
    /// stage in the order pushed. A command naming an object that no longer exists is ignored.
    pub commands: &'a mut Vec<ComponentCommand>,
    /// Borrowed engine services (physics, navigation, network, screen, …) and the
    /// event streams.
    pub services: SystemServices<'a>,
}

/// Reads an object's material parameters through checked native render metadata.
/// The engine implements it over the scene's materials; implementations must read
/// through the supplied [`SystemWorld`], not through editor component views.
pub trait MaterialAccess {
    /// The parameters of material `slot` (0 is the main material) on `object`, or
    /// `None` if the object or slot does not exist.
    fn material_params(&self, world: &SystemWorld<'_>, object: ObjectId, slot: usize) -> Option<crate::MaterialParams>;
}

/// Engine services lent to systems for one frame, without copying scene data.
///
/// Every service is optional: a headless test, a tool or a server without a
/// renderer simply leaves some of them `None`, and the [`ObjectContext`] helpers
/// built on them degrade gracefully (a raycast returns `None`, the pointer reads as
/// absent). In tests, set the ones your system uses
/// (`ctx.services.physics = Some(&mock)`).
#[derive(Clone, Copy, Default)]
pub struct SystemServices<'a> {
    /// Immediate physics queries: raycasts, overlaps, velocities, grounded checks
    /// (3D and 2D). `None` without a physics world.
    pub physics: Option<&'a dyn crate::PhysicsQuery>,
    /// Navmesh queries: paths and nearest walkable point. `None` until the navmesh is baked.
    pub nav: Option<&'a dyn crate::NavQuery>,
    /// The network session as seen this frame: connection state, peers, owners, RPCs.
    pub net: Option<&'a crate::NetView>,
    /// The game view: size, pointer, camera matrices, UI clicks, VR and control state.
    pub screen: Option<&'a crate::ScreenInfo>,
    /// Material parameter reads (see [`MaterialAccess`]).
    pub materials: Option<&'a dyn MaterialAccess>,
    /// The current save data (read with `int`, `float`, `bool`, `str`, `vec3`).
    pub save: Option<&'a crate::SaveGame>,
    /// Localisation tables for `tr`.
    pub strings: Option<&'a crate::Localization>,
    /// The current time scale (1 normal, 0 paused).
    pub time_scale: Option<f32>,
    /// Events emitted in the **previous** frame, as type-erased values. Read them by
    /// type with [`SystemContext::events`]; emitting an event queues it for the next frame.
    pub events: &'a [crate::AnyBox],
    /// Authoring slot metadata (which components an object has, in inspector order).
    /// Metadata only: payload reads go through [`SystemWorld`].
    pub components: Option<&'a dyn ComponentIndex>,
    /// Contacts and animation markers of this frame, delivered in [`Stage::Events`].
    /// Empty in every other stage.
    pub scene_events: &'a [SceneEvent],
    /// The scene's authoring order (saved id to position), when hosted by an ordered
    /// scene. [`run_ordered`] sorts by it so behaviour is deterministic; standalone
    /// worlds can omit it and fall back to saved ids for ties.
    pub order: Option<&'a HashMap<ObjectId, usize>>,
}

impl<'a> SystemContext<'a> {
    /// A context with zero time and no services, over the given input and command queue.
    /// The starting point for tests and tools; set `dt`, `time` and the `services` you need.
    pub fn new(input: &'a InputState, commands: &'a mut Vec<ComponentCommand>) -> Self {
        Self { dt: 0.0, unscaled_dt: 0.0, time: 0.0, water_time: 0.0, input, commands, services: Default::default() }
    }
    /// The events of type `E` that were emitted during the **previous** frame, in
    /// emission order. Events of other types are skipped. The iterator does not
    /// borrow the context, so you can call [`SystemContext::emit`] while looping.
    ///
    /// ```ignore
    /// for hit in ctx.events::<Hit>() {
    ///     total += hit.amount;
    /// }
    /// ```
    pub fn events<E: std::any::Any>(&self) -> impl Iterator<Item = &'a E> + use<'a, E> {
        self.services.events.iter().filter_map(|event| event.0.downcast_ref::<E>())
    }
    /// Broadcast `event` (any `'static` type) to every system. It becomes readable
    /// through [`SystemContext::events`] in the **next** frame, never in this one, so
    /// a sender cannot react to its own event within the frame.
    ///
    /// ```ignore
    /// ctx.emit(Died { object: me });
    /// ```
    pub fn emit<E: std::any::Any>(&mut self, event: E) {
        self.commands.push(ComponentCommand::Emit(crate::AnyBox(Box::new(event))));
    }
    /// Whether `entity` should run: its [`Active`] flag is set (enabled, and every
    /// ancestor enabled) and it is visible in the scene. Check this before acting
    /// for an object; disabled objects must not update.
    pub fn is_active(&self, world: &SystemWorld<'_>, entity: Entity) -> bool {
        world.get::<Active>(entity).is_some_and(|active| active.0) && self.is_visible(world, entity)
    }
    /// Whether `entity` is an object of the scene being run. Hidden or foreign
    /// objects (for example, ones not in the scene's authoring order) are not visible,
    /// and the object helpers treat them as absent.
    pub fn is_visible(&self, world: &SystemWorld<'_>, entity: Entity) -> bool {
        world.object_id(entity).is_some_and(|id| self.services.order.is_none_or(|order| order.contains_key(&id)))
    }
    /// A sort key that puts objects in scene (authoring) order, ties broken by saved
    /// id. Sort by it whenever the order in which objects act can change the outcome.
    pub fn order_key(&self, world: &SystemWorld<'_>, entity: Entity) -> (usize, u128) {
        let id = world.object_id(entity).unwrap_or(ObjectId::NIL);
        (self.services.order.and_then(|order| order.get(&id).copied()).unwrap_or(usize::MAX), id.raw())
    }
    /// A sort key for one instance of component `T`: object order, then the
    /// instance's position among the object's components, then entity bits. It keeps
    /// duplicate instances of a type in inspector order.
    pub fn instance_order<T: DataComponent>(&self, world: &SystemWorld<'_>, owner: Entity, instance: Entity) -> ((usize, u128), usize, u64) {
        let slot = self.services.components.and_then(|index|
            (0..index.component_count(owner)?).find(|&position| index.component_slot(owner, position).is_some_and(|slot|
                slot.entity() == instance && slot.type_id() == std::any::TypeId::of::<T>())))
            .unwrap_or(usize::MAX);
        (self.order_key(world, owner), slot, instance.to_bits().get())
    }
}

/// The signature of a native system. A plain `fn` pointer, so a system cannot
/// capture variables: keep state in components. A non-capturing closure coerces to it.
///
/// Register one with [`ComponentRegistry::register_system`](crate::ComponentRegistry::register_system),
/// [`register_system_for`](crate::ComponentRegistry::register_system_for) or
/// [`register_system_if`](crate::ComponentRegistry::register_system_if).
/// See [Systems](https://lemonxah.github.io/zestengine/docs/ECS.html#systems).
pub type SystemFn = fn(&SystemWorld<'_>, &mut SystemContext<'_>);

#[derive(Clone)]
pub(crate) struct SystemRegistration {
    name: &'static str,
    stage: Stage,
    run: SystemFn,
    condition: Option<fn(&SystemWorld<'_>) -> bool>,
    plugin: Option<String>,
    generation: u64,
}

/// The built-in mesh a new entity is drawn with. Non-empty shapes also get a mesh
/// renderer with the project's default material (swap it afterwards in the inspector).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EntityShape {
    /// No mesh: a transform that only carries components (a spawner, a manager).
    #[default]
    Empty,
    /// A 1 m cube.
    Cube,
    /// A sphere.
    Sphere,
    /// A capsule.
    Capsule,
    /// A flat plane.
    Plane,
}

impl EntityShape {
    /// Every shape, in declaration order.
    pub const ALL: [EntityShape; 5] = [Self::Empty, Self::Cube, Self::Sphere, Self::Capsule, Self::Plane];
}

/// What a registered entity creates: a mesh shape plus the components a new
/// instance starts with, in inspector order. Build one with
/// [`EntityTemplate::new`] and [`EntityTemplate::with`], and register it with
/// [`ComponentRegistry::register_entity`](crate::ComponentRegistry::register_entity).
///
/// See [Entities](https://lemonxah.github.io/zestengine/docs/ECS.html#entities) in the ECS guide.
///
/// ```ignore
/// fn enemy() -> EntityTemplate {
///     EntityTemplate::new(EntityShape::Capsule).with(Health::default()).with(Chaser::default())
/// }
/// ```
pub struct EntityTemplate {
    /// The built-in mesh the object is drawn with ([`EntityShape::Empty`] for none).
    pub shape: EntityShape,
    /// The components a new instance starts with, in inspector order.
    pub components: Vec<Box<dyn crate::Component>>,
}

impl EntityTemplate {
    /// A template with this shape and no components yet.
    pub fn new(shape: EntityShape) -> Self { Self { shape, components: Vec::new() } }
    /// Add a component to the template (builder style).
    pub fn with<T: TypedComponent>(mut self, component: T) -> Self {
        self.components.push(Box::new(component));
        self
    }
}

/// Builds a fresh [`EntityTemplate`] for every new instance.
pub type EntityFn = fn() -> EntityTemplate;

#[derive(Clone)]
pub(crate) struct EntityRegistration {
    name: &'static str,
    build: EntityFn,
    plugin: Option<String>,
    generation: u64,
}

impl ComponentRegistry {
    /// Register data for native ECS systems, retaining the same scene file,
    /// inspector and gizmo registration as an existing TypedComponent.
    ///
    /// Call this for every component type a plugin adds, next to its systems. It is
    /// the same call as [`ComponentRegistry::register`].
    pub fn register_ecs<T: TypedComponent + Send + std::marker::Sync>(&mut self) { self.register::<T>(); }

    /// Register an **entity**: a named, ready-made object, described by a mesh shape
    /// and the components a new instance starts with. The editor lists it under
    /// Create > Entity, and systems spawn it by name with
    /// [`ObjectContext::spawn_entity`](crate::ecs::ObjectContext::spawn_entity).
    ///
    /// `build` runs for every new instance, so each object gets its own component
    /// values. Names are unique: registering an existing name replaces the builder
    /// in place, so a hot-reloaded plugin supersedes its previous build, and a
    /// reload drops entities the plugin no longer registers.
    ///
    /// ```ignore
    /// fn crate_entity() -> EntityTemplate {
    ///     EntityTemplate::new(EntityShape::Cube).with(Lifetime { seconds_left: 10.0 })
    /// }
    /// registry.register_entity("Crate", crate_entity);
    /// ```
    ///
    /// See [Entities](https://lemonxah.github.io/zestengine/docs/ECS.html#entities) in the ECS guide.
    pub fn register_entity(&mut self, name: &'static str, build: EntityFn) {
        let plugin = self.system_scope.as_ref().map(|(name, _)| name.clone());
        let generation = self.system_scope.as_ref().map_or(0, |(_, generation)| *generation);
        let registration = EntityRegistration { name, build, plugin, generation };
        if let Some(entity) = self.entities.iter_mut().find(|entity| entity.name == name) {
            *entity = registration;
        } else { self.entities.push(registration); }
    }

    /// The names of every registered entity, in registration order (what the
    /// editor's Create > Entity menu lists).
    pub fn entity_names(&self) -> impl Iterator<Item = &'static str> + '_ { self.entities.iter().map(|entity| entity.name) }

    /// A fresh template for a new instance of a registered entity. None for an
    /// unknown name (its plugin is not loaded).
    pub fn create_entity(&self, name: &str) -> Option<EntityTemplate> {
        self.entities.iter().find(|entity| entity.name == name).map(|entity| (entity.build)())
    }

    /// Register a system that runs every time `stage` runs.
    ///
    /// Names are unique: registering an existing name replaces the code and stage in
    /// place, which keeps execution order deterministic across hot reloads. Systems
    /// of one stage run in registration order, each seeing the previous one's edits.
    /// Prefer [`register_system_for`](Self::register_system_for) so unused features cost nothing.
    ///
    /// ```ignore
    /// registry.register_system("game::remove_the_dead", Stage::Gameplay, remove_the_dead);
    /// ```
    pub fn register_system(&mut self, name: &'static str, stage: Stage, run: SystemFn) {
        self.register_system_inner(name, stage, run, None);
    }

    /// Register a system that runs only while at least one **enabled** `T` exists.
    /// Skipping also skips the stage's hierarchy propagation and service set-up when
    /// no other system needs them, so features nobody uses cost nothing. This is
    /// the registration to reach for by default.
    ///
    /// ```ignore
    /// registry.register_system_for::<Health>("game::health", Stage::Gameplay, health);
    /// ```
    pub fn register_system_for<T: DataComponent>(&mut self, name: &'static str, stage: Stage, run: SystemFn) {
        self.register_system_if(name, stage, run, |world| world.query::<&T>().iter().next().is_some());
    }

    /// Register a system that runs only while `condition(world)` is true. The
    /// condition is evaluated every time the stage is considered, so keep it cheap.
    /// This is the tool for start-up work: run while any instance still needs
    /// initialising, then go quiet.
    ///
    /// ```ignore
    /// registry.register_system_if("game::door_start", Stage::Start, start_doors,
    ///     |world| world.query::<&Door>().iter().any(|(_, door)| door.closed_y.is_none()));
    /// ```
    pub fn register_system_if(&mut self, name: &'static str, stage: Stage, run: SystemFn, condition: fn(&SystemWorld<'_>) -> bool) {
        self.register_system_inner(name, stage, run, Some(condition));
    }

    fn register_system_inner(&mut self, name: &'static str, stage: Stage, run: SystemFn, condition: Option<fn(&SystemWorld<'_>) -> bool>) {
        let plugin = self.system_scope.as_ref().map(|(name, _)| name.clone());
        let generation = self.system_scope.as_ref().map_or(0, |(_, generation)| *generation);
        let registration = SystemRegistration { name, stage, run, condition, plugin, generation };
        if let Some(system) = self.systems.iter_mut().find(|system| system.name == name) {
            *system = registration;
        } else { self.systems.push(registration); }
    }

    /// Register one plugin's current systems and entities. Reloading removes
    /// those that disappeared from its export, while retaining the order of the
    /// survivors and leaving the engine's and other plugins' registrations alone.
    pub fn register_plugin(&mut self, name: &str, register: fn(&mut Self)) {
        use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
        self.system_generation += 1;
        let generation = self.system_generation;
        let prior_scope = self.system_scope.replace((name.to_owned(), generation));
        let prior_systems = self.systems.clone();
        let prior_entities = self.entities.clone();
        let outcome = catch_unwind(AssertUnwindSafe(|| register(self)));
        self.system_scope = prior_scope;
        match outcome {
            Ok(()) => {
                self.systems.retain(|system| system.plugin.as_deref() != Some(name) || system.generation == generation);
                self.entities.retain(|entity| entity.plugin.as_deref() != Some(name) || entity.generation == generation);
            }
            Err(panic) => { self.systems = prior_systems; self.entities = prior_entities; resume_unwind(panic); }
        }
    }

    /// Whether any system is registered for `stage`, runnable or not.
    pub fn has_systems(&self, stage: Stage) -> bool { self.systems.iter().any(|system| system.stage == stage) }

    /// Whether any system registered for `stage` would run right now (it has no
    /// condition, or its condition holds). The engine uses it to skip a stage entirely.
    pub fn has_runnable_systems(&self, stage: Stage, world: &SceneWorld) -> bool {
        self.has_systems(stage) && world.with_world(|world| self.systems.iter().filter(|system| system.stage == stage)
            .any(|system| system.condition.is_none_or(|condition| condition(world))))
    }

    /// Run every runnable system registered for `stage`, in registration order, over
    /// `world`. The engine calls this once per stage per frame; tests call it to drive
    /// systems headlessly. Commands the systems queue accumulate in `context.commands`;
    /// applying them is the caller's job (the engine does it at the stage boundary).
    ///
    /// Panics inside a system propagate after the world's locks have been released.
    pub fn run_systems(&self, stage: Stage, world: &SceneWorld, context: &mut SystemContext<'_>) {
        for system in self.systems.iter().filter(|system| system.stage == stage) {
            world.with_world(|view| {
                if system.condition.is_none_or(|condition| condition(view)) { (system.run)(view, context); }
            });
        }
    }
}

/// Run `update` for every active instance of component `T`, in scene (authoring)
/// order, handing it the component and an [`ObjectContext`] bound to the object.
///
/// Compared with a hand-written query loop it skips disabled objects (and anything
/// under a disabled parent), iterates deterministically (so networked prediction and
/// object-to-object interactions behave the same on every machine), and gives you the
/// context helpers for moving, reading other objects, spawning, physics and more.
/// Duplicate instances of `T` on one object each get a call, in inspector order.
///
/// The ordering collects and sorts the instances every call; for thousands of
/// instances where order does not matter, a bare [`SystemWorld::query`] is cheaper.
///
/// ```ignore
/// impl Lifetime {
///     fn update(&mut self, obj: &mut ObjectContext<'_, '_>) {
///         self.seconds_left -= obj.dt;
///         if self.seconds_left <= 0.0 { obj.destroy_self(); }
///     }
/// }
///
/// registry.register_system_for::<Lifetime>("game::lifetime", Stage::Update, |world, ctx| {
///     run_ordered(world, ctx, Lifetime::update)
/// });
/// ```
///
/// See [`run_ordered` in the ECS guide](https://lemonxah.github.io/zestengine/docs/ECS.html#the-comfortable-way-run_ordered-and-ecsmovehost).
pub fn run_ordered<T: DataComponent>(world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>, update: impl Fn(&mut T, &mut ObjectContext<'_, '_>)) {
    let mut query = world.query::<(&Owner, &mut T)>();
    let mut actors: Vec<_> = query.iter().filter(|(_, (owner, _))| ctx.is_active(world, owner.0)).collect();
    actors.sort_unstable_by_key(|(instance, (owner, _))| ctx.instance_order::<T>(world, owner.0, *instance));
    for (_, (owner, value)) in actors { update(value, &mut ObjectContext::new(world, owner.0, ctx)); }
}

pub(crate) fn register_builtins(registry: &mut ComponentRegistry) {
    registry.register_ecs::<crate::Spin>();
    registry.register_ecs::<crate::Bob>();
    registry.register_system_if("zestengine::motion", Stage::Update, motion, |world|
        world.query::<&crate::Spin>().iter().next().is_some() || world.query::<&crate::Bob>().iter().next().is_some());
    registry.register_ecs::<crate::Pawn>();
    registry.register_system_if("zestengine::pawn_start", Stage::Start, crate::pawn_start, |world|
        world.query::<&crate::Pawn>().iter().any(|(_, pawn)| !pawn.started));
    registry.register_system_for::<crate::Pawn>("zestengine::pawn_update", Stage::Update, crate::pawn_update);
    crate::kit::register_character_systems(registry);
    crate::kit::register_vr_systems(registry);
    crate::kit::register_combat_systems(registry);
    crate::destruction::register_systems(registry);
    crate::kit::register_event_systems(registry);
    crate::kit::register_strategy_systems(registry);
    crate::ui::register_systems(registry);
    registry.register_ecs::<crate::ScenePortal>();
    registry.register_system_for::<crate::ScenePortal>("zestengine::portal_preload", Stage::Gameplay,
        |world, ctx| run_ordered(world, ctx, crate::ScenePortal::update_native));
    registry.register_system_for::<crate::ScenePortal>("zestengine::portal_events", Stage::Events, |world, ctx| {
        let events = ctx.services.scene_events;
        run_ordered(world, ctx, |portal: &mut crate::ScenePortal, obj| {
            use crate::MotionContext;
            for event in events {
                if let SceneEvent::Contact { object, other, trigger: true, phase: ContactPhase::Enter, .. } = event
                    && *object == obj.self_id() { portal.trigger_native(obj, *other); }
            }
        });
    });
    crate::net::register_avatar_systems(registry);
    crate::avatar_hands::register_systems(registry);
    crate::kit::register_controls_systems(registry);
    crate::constraint::register_systems(registry);
    registry.register_ecs::<crate::Vehicle>();
    registry.register_ecs::<crate::Sync>();
    registry.register_system_for::<crate::Vehicle>("zestengine::vehicle_controls", Stage::FixedUpdate, crate::vehicle::controls);
    crate::vehicle::register_seat_system(registry);
    registry.register_system_for::<crate::Sync>("zestengine::ownership_controls", Stage::Update, ownership_controls);
    crate::ai::register_systems(registry);
    crate::script::register_systems(registry);
    crate::wasm::register_systems(registry);
}

fn ownership_controls(world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) {
    let mut query = world.query::<(&Owner, &mut crate::Sync)>();
    let mut objects: Vec<_> = query.iter().filter(|(_, (owner, _))| ctx.is_active(world, owner.0)).collect();
    objects.sort_unstable_by_key(|(_, (owner, _))| ctx.order_key(world, owner.0));
    for (_, (owner, sync)) in objects {
        sync.started = true;
        let Some(net) = ctx.services.net.filter(|net| net.connected) else { continue };
        if !sync.grabbable || sync.grab_action.is_empty() || !ctx.input.pressed(&sync.grab_action) { continue; }
        let id = world.object_id(owner.0).expect("component owner");
        ctx.commands.push(if net.owns(id) { ComponentCommand::ReleaseOwnership(id) } else { ComponentCommand::RequestOwnership(id) });
    }
}

fn motion(world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) {
    use glam::{Quat, Vec3};
    for (_, (owner, spin)) in world.query::<(&Owner, &crate::Spin)>().iter() {
        if !world.get::<Active>(owner.0).is_some_and(|active| active.0) { continue; }
        let mut transform = world.get_mut::<Transform>(owner.0).expect("component owner");
        let axis = Vec3::from(spin.axis).normalize_or(Vec3::Y);
        let step = Quat::from_axis_angle(axis, spin.degrees_per_second.to_radians() * ctx.dt);
        transform.rotation = (step * transform.rotation).normalize();
    }
    for (_, (owner, bob)) in world.query::<(&Owner, &mut crate::Bob)>().iter() {
        if !world.get::<Active>(owner.0).is_some_and(|active| active.0) { continue; }
        let mut transform = world.get_mut::<Transform>(owner.0).expect("component owner");
        let phase = ctx.time * bob.frequency * std::f32::consts::TAU;
        let offset = Vec3::from(bob.axis) * (bob.amplitude * phase.sin());
        transform.translation += offset - bob.applied;
        bob.applied = offset;
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod utility_tests;
