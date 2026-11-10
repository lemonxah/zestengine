//! [`ObjectContext`], the per-object helper every native game system uses.
//!
//! Local transforms and sibling component data stay authoritative in the ECS. Reads return checked borrow
//! guards; data edits release their checked borrow before returning. Everything a game developer needs to
//! know (immediate versus queued calls, frame order, targets, local versus world space) is documented on
//! [`ObjectContext`] itself.

use std::ops::{Deref, DerefMut};
use glam::{Mat4, Quat, Vec2, Vec3};
use crate::{Character, ComponentCommand, MotionContext, ObjectId, ObjectRef, RayFilter, RayHit, Swimmer, Transform};
use super::{DataComponent, Entity, Parent, Ref, SystemContext, SystemWorld};

/// Everything a native game system can do to the world, bound to one object (called "self" below).
///
/// A native system is a function the engine runs once per frame stage ([`Stage`](super::Stage)). The usual way
/// to write one is [`run_ordered`](super::run_ordered): it calls your closure once for every active instance of
/// a component, in the scene's authoring order, and passes it the component (`&mut T`) plus an `ObjectContext`
/// bound to the object that owns that instance. Through it you read and edit other objects, move things,
/// spawn and destroy, push physics bodies, play sounds, save, switch scenes and use the network.
///
/// **Guides:** the [Engine API Guide](https://lemonxah.github.io/zestengine/docs/ENGINE_API.html) walks through
/// this type by task (moving things, spawning and destroying, physics, input, audio, networking, with recipes),
/// and the [ECS Guide](https://lemonxah.github.io/zestengine/docs/ECS.html#the-comfortable-way-run_ordered-and-ecsmovehost)
/// explains where it fits in the system model.
///
/// ```ignore
/// use zestengine_core::ecs::{ObjectContext, Stage, SystemContext, SystemWorld, run_ordered};
/// use zestengine_core::{ComponentRegistry, MotionContext, ObjectRef, TypedComponent};
///
/// #[derive(Default, Serialize, Deserialize)]
/// struct Chaser { target: ObjectRef, speed: f32 }
/// impl TypedComponent for Chaser { const NAME: &'static str = "Chaser"; }
///
/// fn chase(world: &SystemWorld<'_>, ctx: &mut SystemContext<'_>) {
///     run_ordered(world, ctx, |chaser: &mut Chaser, obj: &mut ObjectContext<'_, '_>| {
///         // World position of the target, or None when the reference is unset or the object is gone.
///         let Some(goal) = obj.position_of(chaser.target) else { return };
///         let here = obj.world_position();                        // `MotionContext`: world position of self
///         let step = (goal - here).normalize_or_zero() * chaser.speed * obj.dt; // `dt` comes from `SystemContext`
///         obj.set_world_position(here + step);                    // immediate: later systems see the move
///     });
/// }
///
/// fn register(registry: &mut ComponentRegistry) {
///     registry.register_ecs::<Chaser>();
///     registry.register_system_for::<Chaser>("game::chase", Stage::Update, chase);
/// }
/// ```
///
/// # What a call does, and when
///
/// Every method is one of three kinds, and its documentation says which one it is.
///
/// | Kind | Which methods | When you see the result |
/// | --- | --- | --- |
/// | **Immediate** | Reads, and writes that edit native component data in place: [`edit`](Self::edit), [`set_local_transform`](Self::set_local_transform), [`set_world_transform_of`](Self::set_world_transform_of), [`set_world_position_of`](Self::set_world_position_of), [`set_destination`](Self::set_destination), [`stop_agent`](Self::stop_agent), [`emit_particles`](Self::emit_particles), the local copy written by [`set_net_var`](Self::set_net_var), and the [`MotionContext`] setters such as `set_translation` and `set_animator_bool`. | At once: on the next line, and in every system that runs later in the same stage. |
/// | **Queued** | Anything that pushes a [`ComponentCommand`] for the engine: spawning, [`destroy`](Self::destroy), [`set_active`](Self::set_active), [`add_impulse`](Self::add_impulse), [`set_velocity`](Self::set_velocity), [`move_character`](MotionContext::move_character), sounds, saves, [`modify_material`](Self::modify_material), time scale, language, and the local half of networked spawns. | When the current stage is over. The engine applies the queued commands in the order they were pushed, after every system of the stage has run and before the next stage starts. |
/// | **Queued (obj)** | Commands the play session does not handle itself but returns to the application hosting the game (the player or the editor): scene loads, [`quit`](Self::quit), cursor lock, input rebinding, RPCs, ownership requests, and the replication half of networked spawns and variables. | After the frame's last stage. The host applies them at the end of the frame. |
///
/// Spawn methods return the new [`ObjectId`] immediately, but the object only exists once its command has been
/// applied, which is the next stage. Until then reads such as [`get`](Self::get) or
/// [`transform_of`](Self::transform_of) find nothing for that id. Commands you queue *after* a spawn and aim at
/// the new id (for example [`set_active`](Self::set_active)) are applied after it, in order, and do find it.
/// Physics is the exception: a rigid body is created at the next physics sync, so an impulse or velocity aimed
/// at an object spawned (or re-enabled) in the same stage is silently dropped.
///
/// # Frame order
///
/// A frame runs these stages in order. A system sees everything systems of earlier stages did, because queued
/// commands are applied between stages.
///
/// 1. `Start`: objects that just became active run their start-up systems (once per object).
/// 2. `FixedUpdate`: zero or more times per frame at a fixed time step, each followed by a physics step.
/// 3. `Events`: collision, trigger and animation-event callbacks.
/// 4. `Update`, then `Gameplay`, then `LateUpdate`. The commands of `Update` are applied before `Gameplay`
///    runs, so weapons, cameras and the like see positions after movement and physics corrections. Navigation
///    agents and animation advance between `Gameplay` and `LateUpdate`.
///
/// See [`Stage`](super::Stage) for the stage list. Systems of one stage run one after another (never in
/// parallel), so an immediate write by an earlier system is visible to a later one.
///
/// # Targets
///
/// Methods that act on another object take `impl Into<ObjectRef>`: an [`ObjectRef`] (the field type you put on a
/// component and wire up in the inspector) or a bare [`ObjectId`]. A few methods take an [`ObjectRef`] only,
/// because they predate this convention; their docs say so, and an [`ObjectId`] converts with `.into()`.
///
/// * Reads and in-place edits resolve the target with [`entity_of`](Self::entity_of). An unset reference, an id
///   no object has (never spawned, already destroyed, or spawned this stage and not applied yet), or an entity
///   that is not a scene object gives `None`, `false` or a no-op; it is not an error.
/// * Queued methods only check that the reference is set. The object is looked up when the command is applied;
///   if it has gone by then, the command does nothing.
/// * A **disabled** (inactive) object is still a valid target for reads and edits. Use
///   [`is_active`](Self::is_active) to ask whether it is running.
///
/// # Local and world space
///
/// [`translation`](MotionContext::translation), [`rotation`](MotionContext::rotation) and [`scale`](MotionContext::scale) (from
/// [`MotionContext`]) are the object's own **local** values, relative to its parent. [`self_transform`](Self::self_transform),
/// [`transform_of`](Self::transform_of), [`matrix_of`](Self::matrix_of), [`position_of`](Self::position_of) and
/// [`world_position`](MotionContext::world_position) are **world** values. The `set_world_*` methods take world values
/// and convert them through the parent chain; [`set_local_transform`](Self::set_local_transform) writes the local
/// fields directly. World values are computed from the live local transforms, so they include moves made earlier
/// in the same stage.
///
/// # Borrow rules
///
/// Component data is borrowed with hecs' checked borrows, one component column at a time. Asking for a component
/// type that is already mutably borrowed panics (hecs reports `already borrowed`). The usual way to trigger that:
///
/// * requesting the same component type `T` that the running [`run_ordered`](super::run_ordered) system is
///   iterating, for example `ctx.get::<Health>(other)` from inside a `Health` system (the system already holds
///   `&mut Health`);
/// * calling [`edit`](Self::edit) for a type while a [`Ref`] guard of that type from [`get`](Self::get) is still
///   alive.
///
/// Read sibling components of *other* types, copy what you need out of a guard and drop it, or collect ids first
/// and act afterwards.
///
/// # What you get through `Deref`
///
/// `ObjectContext` derefs (also mutably) to [`SystemContext`], so the frame's data is available on the context itself:
///
/// * `dt`: seconds this stage advances game time, scaled by the time scale (the fixed step during `FixedUpdate`);
///   `unscaled_dt`: real seconds, unaffected by [`set_time_scale`](Self::set_time_scale), for UI and pause menus.
/// * `time`: scaled game seconds since play began; `water_time`: the water animation clock (synchronised between
///   peers in a network game).
/// * `input`: this frame's [`InputState`](crate::InputState), for example `ctx.input.down("Fire")`,
///   `pressed`, `released`, `axis`, `axis2` and `mouse_delta`.
/// * `commands`: the command queue itself. Push a [`ComponentCommand`] directly for anything that has no helper
///   here, such as [`ComponentCommand::AddForce`], [`ComponentCommand::PlayAudio`],
///   [`ComponentCommand::AssignMaterial`], [`ComponentCommand::NetDespawn`] or
///   [`ComponentCommand::NetLoadScene`].
/// * `services`: the borrowed frame services ([`SystemServices`](super::SystemServices): `physics`, `nav`, `net`,
///   `screen`, `materials`, `save`, `strings`, `time_scale`, `events`, `scene_events`, ...). The helpers on this
///   type wrap them and fall back to `None`, `false` or a default when a service is missing, as in a headless run.
/// * [`events::<E>()`](SystemContext::events): the events of type `E` emitted during the previous frame, and
///   [`emit`](SystemContext::emit): queue an event that every system reads next frame.
/// * [`is_visible`](SystemContext::is_visible), [`order_key`](SystemContext::order_key) and
///   [`instance_order`](SystemContext::instance_order): lower-level helpers for hand-written queries.
///   (`SystemContext::is_active` takes a world and an entity; the context's own
///   [`is_active`](Self::is_active) takes an object reference and shadows it.)
///
/// # Index
///
/// ## Transforms
///
/// * Self, local: `translation`, `set_translation`, `rotation`, `set_rotation`, `scale` (from [`MotionContext`]).
/// * Self, world: [`self_transform`](Self::self_transform), `world_position` and `set_world_position` (from
///   [`MotionContext`]), [`set_world_rotation`](Self::set_world_rotation), `parent_world`.
/// * Any object, read (world): [`transform_of`](Self::transform_of), [`matrix_of`](Self::matrix_of),
///   [`position_of`](Self::position_of).
/// * Any object, write (immediate): [`set_local_transform`](Self::set_local_transform),
///   [`set_world_transform_of`](Self::set_world_transform_of),
///   [`set_world_position_of`](Self::set_world_position_of).
/// * Placement markers: [`spawn_point`](Self::spawn_point), [`spawn_point_for`](Self::spawn_point_for).
///
/// ## Reading objects
///
/// [`get`](Self::get), [`get_self`](Self::get_self), [`objects_with`](Self::objects_with),
/// [`find_object`](Self::find_object), [`is_active`](Self::is_active), [`entity_of`](Self::entity_of),
/// [`material_params`](Self::material_params), `self_id`, and the public fields [`world`](Self::world) and
/// [`owner`](Self::owner) for direct ECS access.
///
/// ## Editing
///
/// [`edit`](Self::edit) changes another object's component in place (immediate). Queued component changes go
/// through commands: [`modify_material`](Self::modify_material), [`set_active`](Self::set_active).
/// [`emit_particles`](Self::emit_particles), [`set_destination`](Self::set_destination) and
/// [`stop_agent`](Self::stop_agent) are shortcuts for common edits.
///
/// ## Spawning and destroying
///
/// * Local: [`spawn_prefab`](Self::spawn_prefab) (from a `.prefab` file), [`spawn_entity`](Self::spawn_entity)
///   (a registered entity), [`instantiate`](Self::instantiate) (clone a scene object),
///   [`destroy`](Self::destroy), [`destroy_self`](Self::destroy_self), [`set_active`](Self::set_active).
/// * Networked: [`net_spawn`](Self::net_spawn), [`net_spawn_server`](Self::net_spawn_server),
///   [`net_clone`](Self::net_clone), [`net_spawn_with`](Self::net_spawn_with).
///
/// ## Physics
///
/// * Queries (immediate): [`raycast`](Self::raycast), [`raycast_filtered`](Self::raycast_filtered),
///   [`raycast2d`](Self::raycast2d), [`raycast2d_filtered`](Self::raycast2d_filtered),
///   [`water_surface`](MotionContext::water_surface), [`grounded`](MotionContext::grounded).
/// * Forces (queued): [`add_impulse`](Self::add_impulse), [`set_velocity`](Self::set_velocity),
///   [`move_character`](MotionContext::move_character).
/// * Character data: `has_character`, `capsule`, `set_capsule`, `swimmer` (from [`MotionContext`]).
///
/// ## Input and UI
///
/// * Pointer and screen: [`pointer_position`](Self::pointer_position), [`pointer_ray`](Self::pointer_ray),
///   [`screen_ray`](Self::screen_ray), [`world_to_screen`](Self::world_to_screen),
///   [`screen_size`](Self::screen_size), [`pointer_over_ui`](Self::pointer_over_ui),
///   [`ui_text_editing`](Self::ui_text_editing), [`ui_clicked`](Self::ui_clicked),
///   [`ui_cancel_pressed`](Self::ui_cancel_pressed).
/// * Camera and cursor: [`set_active_camera`](Self::set_active_camera),
///   [`set_cursor_locked`](Self::set_cursor_locked).
/// * Bindings and rebinding: [`input_bindings`](Self::input_bindings),
///   [`control_scheme`](Self::control_scheme), [`binding_conflicts`](Self::binding_conflicts),
///   [`rebind_with`](Self::rebind_with), [`tune_action`](Self::tune_action),
///   [`reset_scheme_bindings`](Self::reset_scheme_bindings), [`input_command`](Self::input_command),
///   [`capture_next_input`](Self::capture_next_input), [`captured_input`](Self::captured_input),
///   [`is_capturing_input`](Self::is_capturing_input).
/// * Text: [`tr`](Self::tr), [`language`](Self::language), [`set_language`](Self::set_language).
/// * VR: [`vr_frame`](Self::vr_frame), [`body_calibrating`](Self::body_calibrating).
///
/// ## Audio
///
/// [`play_sound_at`](Self::play_sound_at) (one-shot at a position) and
/// [`set_bus_volume`](Self::set_bus_volume) (mixer bus). Other audio commands
/// ([`ComponentCommand::PlayAudio`], [`ComponentCommand::StopAudio`], [`ComponentCommand::SetBusMuted`]) have no
/// helper; push them onto `commands`.
///
/// ## Animation and navigation
///
/// * Animation: `set_animator_bool` (from [`MotionContext`]), [`emit_particles`](Self::emit_particles).
/// * Navigation: [`nearest_walkable`](Self::nearest_walkable), [`set_destination`](Self::set_destination),
///   [`stop_agent`](Self::stop_agent), [`rebake_navmesh`](Self::rebake_navmesh).
///
/// ## Scenes and saves
///
/// * Scenes: [`load_scene`](Self::load_scene), [`preload_scene`](Self::preload_scene),
///   [`load_scene_additive`](Self::load_scene_additive), [`unload_scene`](Self::unload_scene),
///   [`keep_on_scene_change`](Self::keep_on_scene_change), [`quit`](Self::quit).
/// * Saves: [`save_data`](Self::save_data), [`save_value`](Self::save_value),
///   [`remove_save_value`](Self::remove_save_value), [`write_save`](Self::write_save),
///   [`load_save`](Self::load_save).
/// * Game speed: [`set_time_scale`](Self::set_time_scale), [`time_scale`](Self::time_scale).
///
/// ## Networking
///
/// *Authority* is the right to simulate an object and publish its state. In a session it belongs to the object's
/// owner peer, or to the host for an object nobody owns; see [`has_authority`](Self::has_authority). A copy that
/// another peer simulates is *driven remotely* ([`driven_remotely`](Self::driven_remotely)): local input must not
/// move it. **Offline (no session) every networking method behaves as local**: you have authority over
/// everything, [`connected`](Self::connected), [`is_server`](Self::is_server) and
/// [`is_client`](Self::is_client) are `false`, [`local_peer`](Self::local_peer) is `0`, networked spawns create a
/// plain local object, and [`rpc_self`](Self::rpc_self) loops back to the caller.
///
/// * State: [`connected`](Self::connected), [`is_server`](Self::is_server), [`is_client`](Self::is_client),
///   [`local_peer`](Self::local_peer), [`peers`](Self::peers), [`net_view`](Self::net_view),
///   [`has_authority`](Self::has_authority), [`driven_remotely`](Self::driven_remotely).
/// * Ownership: [`request_ownership_of`](Self::request_ownership_of).
/// * Variables and calls: [`set_net_var`](Self::set_net_var), [`set_net_var_self`](Self::set_net_var_self),
///   [`rpc_self`](Self::rpc_self), [`rpcs_for_self`](Self::rpcs_for_self).
/// * Spawning: [`net_spawn`](Self::net_spawn), [`net_spawn_server`](Self::net_spawn_server),
///   [`net_clone`](Self::net_clone), [`net_spawn_with`](Self::net_spawn_with).
pub struct ObjectContext<'world, 'frame> {
    /// The ECS world this system call runs against.
    ///
    /// Use it for direct component access to the owner (`obj.world.get::<T>(obj.owner)`) or for custom
    /// `world.query` loops. Unlike [`get`](Self::get) and [`edit`](Self::edit) it applies no visibility check. The same
    /// borrow rules apply: see "Borrow rules" on [`ObjectContext`].
    pub world: &'world SystemWorld<'world>,
    /// The object this context acts for ("self"): the object that owns the component instance being updated.
    ///
    /// This is a hecs handle, valid for the current call only. Use [`self_id`](MotionContext::self_id) when you need a
    /// persistent [`ObjectId`] to store or compare.
    pub owner: Entity,
    /// The frame's shared context, reached through [`Deref`].
    context: &'world mut SystemContext<'frame>,
}

impl<'world, 'frame> ObjectContext<'world, 'frame> {
    /// Binds a context to the object `owner` for the duration of one system call.
    ///
    /// Game code rarely calls this: [`run_ordered`](super::run_ordered) builds one context per active component
    /// instance. Build your own in tests, or when a system walks objects by hand with `world.query`, passing the
    /// [`SystemContext`] the engine gave you.
    ///
    /// # Panics
    ///
    /// `owner` must be a scene object (an entity with an [`ObjectId`] and a [`Transform`]). The [`MotionContext`]
    /// methods (`self_id`, `translation`, ...) panic for any other entity.
    pub fn new(world: &'world SystemWorld<'world>, owner: Entity, context: &'world mut SystemContext<'frame>) -> Self {
        Self { world, owner, context }
    }
    /// Whether another peer, not this machine, is currently simulating this object.
    ///
    /// True only when all of these hold: a network session is [`connected`](Self::connected), the object has a
    /// [`NetworkObject`](crate::NetworkObject) or [`Sync`](crate::Sync) component, and the local peer lacks
    /// [authority](Self::has_authority) over it. Input-driven systems (character controllers, weapons) return early
    /// when this is true, so the local keyboard does not move a copy that belongs to someone else.
    ///
    /// **Immediate.** Always `false` offline and for objects that are not networked.
    pub fn driven_remotely(&self) -> bool {
        self.services.net.is_some_and(|net| net.connected
            && (self.world.get::<crate::NetworkObject>(self.owner).is_some() || self.world.get::<crate::Sync>(self.owner).is_some())
            && !net.has_authority(self.self_id()))
    }
    /// The **world** transform of this object (position, rotation and scale decomposed from its world matrix).
    ///
    /// **Immediate.** Computed from the live local transforms, so it includes moves made earlier in this stage by
    /// you or by other systems. For the local fields use [`translation`](MotionContext::translation),
    /// [`rotation`](MotionContext::rotation) and [`scale`](MotionContext::scale); for just the world position,
    /// [`world_position`](MotionContext::world_position).
    pub fn self_transform(&self) -> Transform { Transform::from_matrix(self.world()) }
    /// Resolves an object reference to its ECS [`Entity`], or `None` if the object cannot be used.
    ///
    /// This is the gate in front of [`get`](Self::get), [`edit`](Self::edit), [`transform_of`](Self::transform_of),
    /// [`is_active`](Self::is_active) and the other read and edit methods. It returns `None` when
    ///
    /// * the reference is unset;
    /// * no object has that id (never spawned, already destroyed, or spawned this stage and not applied yet);
    /// * the entity is not *visible*. Visible means the entity is a scene object (it carries an [`ObjectId`]) and,
    ///   when the scene supplies an authoring order, is listed in it. The helper entities that hold the second and
    ///   later copy of a duplicated component are not objects and never resolve.
    ///
    /// Visible does not mean active: a disabled object resolves normally; ask [`is_active`](Self::is_active) whether
    /// it runs. The returned [`Entity`] is a runtime handle for this call only; store [`ObjectId`] or [`ObjectRef`]
    /// values instead.
    pub fn entity_of(&self, target: impl Into<ObjectRef>) -> Option<Entity> {
        let entity = self.world.entity(target.into().id()?)?;
        self.context.is_visible(self.world, entity).then_some(entity)
    }
    /// Reads component `T` of another object.
    ///
    /// Returns a checked borrow guard ([`Ref`], which derefs to `&T`) for the object's primary instance of `T`, or
    /// `None` if the target does not resolve (see [`entity_of`](Self::entity_of)), has no `T`, or has its `T` switched
    /// off. Additional duplicate instances of `T` on the same object are not reachable through `get`.
    ///
    /// **Immediate.** It sees every in-place edit made earlier in this stage. Keep the guard short-lived: it holds a
    /// shared borrow, so editing or mutably borrowing the same component type while it is alive panics (see "Borrow
    /// rules" on [`ObjectContext`]). Copy out what you need and let the guard drop.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Read, then release the borrow before doing anything else with the context.
    /// let hp = ctx.get::<Health>(self.target).map(|health| health.current);
    /// if hp.is_some_and(|hp| hp < 10.0) {
    ///     ctx.play_sound_at("audio/low_health.wav", ctx.self_transform().translation, 1.0);
    /// }
    /// ```
    pub fn get<T: DataComponent>(&self, target: impl Into<ObjectRef>) -> Option<Ref<'_, T>> {
        self.world.get::<T>(self.entity_of(target)?)
    }
    /// Reads component `T` of this object, without the visibility check of [`get`](Self::get).
    ///
    /// Returns `None` if the object has no enabled `T`. **Immediate.**
    ///
    /// Use it for *sibling* components of a different type than the one your system iterates. Never ask for that
    /// type itself: [`run_ordered`](super::run_ordered) already holds `&mut T` for it, and a second borrow of the same
    /// type panics.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // A weapon system reading another component on the same object.
    /// let held = ctx.get_self::<Grabbable>().is_some_and(|grab| grab.held);
    /// ```
    pub fn get_self<T: DataComponent>(&self) -> Option<Ref<'_, T>> { self.world.get::<T>(self.owner) }
    /// Changes component `T` of another object in place by running `edit` on it.
    ///
    /// Returns `true` if the closure ran, `false` if the target does not resolve (see
    /// [`entity_of`](Self::entity_of)) or has no enabled `T`.
    ///
    /// **Immediate.** The closure runs before `edit` returns, with exclusive access to the live data, and later
    /// systems of the same stage see the result. Nothing is queued. The closure only receives the component, so
    /// compute anything that needs the context beforehand. Panics if `T` is already borrowed (see "Borrow rules" on
    /// [`ObjectContext`]), for example while you hold a [`Ref`] of the same type from [`get`](Self::get).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Hurt whatever the ray hit, if it has Health.
    /// let damage = 25.0;
    /// let hurt = ctx.edit::<Health>(hit.object, move |health| { health.apply_damage(damage); });
    /// ```
    pub fn edit<T: DataComponent>(&self, target: impl Into<ObjectRef>, edit: impl FnOnce(&mut T)) -> bool {
        let Some(entity) = self.entity_of(target) else { return false };
        let Some(mut value) = self.world.get_mut::<T>(entity) else { return false };
        edit(&mut value);
        true
    }
    /// The **world** matrix of an object: its own local transform multiplied through all of its ancestors.
    ///
    /// **Immediate.** `None` if the target does not resolve or its parent chain is malformed (a cycle). Computed from
    /// the live local transforms, so it includes moves made earlier in this stage. See
    /// [`transform_of`](Self::transform_of) for the decomposed form and [`position_of`](Self::position_of) for just
    /// the translation.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Where is a point that sits 0.5 m in front of the target, in world space?
    /// if let Some(world) = ctx.matrix_of(self.target) {
    ///     let muzzle = world.transform_point3(Vec3::new(0.0, 0.2, -0.5));
    /// }
    /// ```
    pub fn matrix_of(&self, target: impl Into<ObjectRef>) -> Option<Mat4> {
        self.world.world_transform(self.entity_of(target)?)
    }
    /// The **world** translation, rotation and scale of an object, decomposed from its world matrix.
    ///
    /// **Immediate.** `None` if the target does not resolve (see [`entity_of`](Self::entity_of)). Includes moves made
    /// earlier in this stage. The scale is the accumulated world scale; a parent with non-uniform scale above a
    /// rotated child cannot be represented exactly and loses any shear. For the object's **local** values read the
    /// `Transform` component with [`get`](Self::get).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // How far away, in world space, is the target?
    /// let Some(target) = ctx.transform_of(self.target) else { return };
    /// let distance = (target.translation - ctx.self_transform().translation).length();
    /// ```
    pub fn transform_of(&self, target: impl Into<ObjectRef>) -> Option<Transform> {
        self.matrix_of(target).map(Transform::from_matrix)
    }
    /// Every object that has an enabled component `T`, as ids in authoring order.
    ///
    /// Use it to find "all enemies" or "all spawn markers" without holding a reference to each. Only object roots are
    /// returned: an object appears once even if it carries several copies of `T`. Objects whose copy of `T` is
    /// switched off are not included.
    ///
    /// **Immediate.** The list is a fresh `Vec` built from the world as it is now: objects spawned this stage are not
    /// in it, objects destroyed this stage still are (until the boundary). **Disabled (inactive) objects are
    /// included**; filter with [`is_active`](Self::is_active) to keep only running ones. The order is the scene's
    /// authoring order with the saved id as tie-break, so results are deterministic. It scans every object that has
    /// `T`; cache the result if you need it every frame in a large scene.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Count the enemies still running.
    /// let alive = ctx.objects_with::<Enemy>().into_iter().filter(|id| ctx.is_active(*id)).count();
    /// ```
    pub fn objects_with<T: DataComponent>(&self) -> Vec<ObjectId> {
        let mut objects: Vec<_> = self.world.query::<(&ObjectId, &T)>().iter()
            .filter(|(entity, _)| self.context.is_visible(self.world, *entity))
            .map(|(entity, (id, _))| (self.context.order_key(self.world, entity), *id)).collect();
        objects.sort_unstable_by_key(|(key, _)| *key);
        objects.into_iter().map(|(_, id)| id).collect()
    }
    /// Whether the object is running: enabled itself and under no disabled ancestor.
    ///
    /// `false` if the target does not resolve. **Immediate**, but it reads the activation computed at the *start* of
    /// the stage, so a [`set_active`](Self::set_active) queued during this stage is not reflected until the next
    /// one. A disabled object runs none of its own systems, so enabling it again has to come from another object.
    ///
    /// This shadows [`SystemContext::is_active`], which takes a world and an entity.
    pub fn is_active(&self, target: impl Into<ObjectRef>) -> bool {
        self.entity_of(target).is_some_and(|entity| self.context.is_active(self.world, entity))
    }
    /// Finds an object by name: the first visible object with that name in the scene's authoring order, or `None`.
    ///
    /// The comparison is exact (case-sensitive). Disabled objects are found too. **Immediate**: objects spawned this
    /// stage are not there yet. Names are not unique, so prefer an [`ObjectRef`] field wired up in the inspector for
    /// objects you know at design time, and use `find_object` for lookups by convention (`"Player"`, `"Spawner"`).
    /// It scans every named object, so look an object up once (in a start-up system or on the first frame) and keep
    /// the returned [`ObjectId`] instead of searching every frame.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if !self.player.is_set() {
    ///     self.player = ctx.find_object("Player").map(ObjectRef::new).unwrap_or(ObjectRef::NONE);
    /// }
    /// ```
    pub fn find_object(&self, name: &str) -> Option<ObjectId> {
        self.world.query::<(&ObjectId, &super::Name)>().iter()
            .filter(|(entity, (_, value))| value.0 == name && self.context.is_visible(self.world, *entity))
            .min_by_key(|(entity, _)| self.context.order_key(self.world, *entity))
            .map(|(_, (id, _))| *id)
    }
    /// Whether the UI element (a button, say) was clicked since the previous frame.
    ///
    /// **Immediate.** `false` if the reference is unset, nothing was clicked, or there is no screen service. The
    /// object is not checked for being visible or active. Combine with [`pointer_over_ui`](Self::pointer_over_ui)
    /// to keep UI clicks from also triggering world actions.
    pub fn ui_clicked(&self, target: impl Into<ObjectRef>) -> bool {
        target.into().id().is_some_and(|id| self.services.screen.is_some_and(|screen| screen.ui_clicks.contains(&id)))
    }
    /// Whether the player pressed the UI cancel / back control (Escape, gamepad East) since the previous frame.
    ///
    /// **Immediate.** `false` without a screen service.
    pub fn ui_cancel_pressed(&self) -> bool { self.services.screen.is_some_and(|screen| screen.ui_cancel) }
    /// The live input bindings (every control scheme and its actions, including the player's rebinds), or `None`
    /// without a screen service.
    ///
    /// **Immediate**, read-only. Change bindings with [`rebind_with`](Self::rebind_with),
    /// [`tune_action`](Self::tune_action), [`reset_scheme_bindings`](Self::reset_scheme_bindings) or
    /// [`input_command`](Self::input_command); the change shows here after the host has applied it at the end of the
    /// frame.
    pub fn input_bindings(&self) -> Option<&crate::Bindings> { Some(&self.services.screen?.controls.bindings) }
    /// Queues any [`InputCommand`](crate::InputCommand): rebind, unbind, tune, reset, switch scheme, start or cancel
    /// a capture.
    ///
    /// **Queued (obj).** Wrapped in [`ComponentCommand::Input`] and applied by the host at the end of the frame.
    /// Successful changes are saved to the player's overrides file immediately, so they persist between runs. A
    /// request that names a scheme, action or slot that does not exist is logged and ignored; no error comes back.
    ///
    /// The other input helpers here are shortcuts for common commands. Use this one for the rest: `Unbind`,
    /// `ResetAction`, `ResetAll`, `SetScheme`, `SetAutoSwitch` and `CancelCapture`.
    pub fn input_command(&mut self, command: crate::InputCommand) { self.commands.push(ComponentCommand::Input(command)); }
    /// Changes the analog shaping of an action (deadzone, scale, axis inversion).
    ///
    /// Fields left as `None` in the [`ActionTuning`](crate::ActionTuning) are kept; a deadzone is clamped to 0..1.
    /// `scheme` names a control scheme, and `""` means the active one.
    ///
    /// **Queued (obj).** See [`input_command`](Self::input_command) for timing and error handling.
    pub fn tune_action(&mut self, scheme: &str, action: &str, tuning: crate::ActionTuning) {
        self.input_command(crate::InputCommand::Tune { scheme: scheme.into(), action: action.into(), tuning });
    }
    /// Asks the player to press the next input and reports it back to this object (the "press a key to rebind"
    /// prompt).
    ///
    /// `kind` listens for buttons or for analog axes; `device` limits the accepted inputs to one device (`None` means
    /// any). **Queued (obj).** Listening starts at the end of the frame. Poll [`is_capturing_input`](Self::is_capturing_input)
    /// while waiting and [`captured_input`](Self::captured_input) for the answer. A new request replaces a pending one.
    /// While a capture is pending (and until the pressed input is released) gameplay actions do not see that input.
    pub fn capture_next_input(&mut self, kind: crate::CaptureKind, device: Option<crate::InputDevice>) {
        let owner = self.self_id();
        self.input_command(crate::InputCommand::Capture { owner, request: crate::CaptureRequest { kind, device } });
    }
    /// The outcome of this object's input capture, on the frame it finished; otherwise `None`.
    ///
    /// `Some(CaptureResult::Captured(source))` carries the input the player pressed; `Some(CaptureResult::Cancelled)`
    /// means Escape or a cancel. Only the object that asked sees the result, and only for one frame, so read it
    /// every frame while waiting. Pass the source to [`rebind_with`](Self::rebind_with). **Immediate.**
    /// See [`CaptureResult`](crate::CaptureResult).
    pub fn captured_input(&self) -> Option<crate::CaptureResult> {
        self.services.screen?.controls.captured.filter(|(owner, _)| *owner == self.self_id()).map(|(_, result)| result)
    }
    /// Whether this object's input capture is still waiting for the player to press something.
    ///
    /// **Immediate.** `false` once it resolved or was cancelled, when another object holds the capture, or without a
    /// screen service.
    pub fn is_capturing_input(&self) -> bool { self.services.screen.is_some_and(|screen| screen.controls.capturing == Some(self.self_id())) }
    /// Binds `source` (a key, button or axis, as an [`InputSource`](crate::InputSource)) to an action.
    ///
    /// `scheme` names the control scheme (`""` is the active one) and `action` the action. `slot` picks which list of
    /// the action to edit ([`BindingSlot`](crate::BindingSlot): its buttons, the parts of a composite axis, or the
    /// analog axes) and `index` the position in that list; the input there is replaced. `policy` decides what happens
    /// to other actions that already use this input ([`ConflictPolicy`](crate::ConflictPolicy)): `Swap` (the default)
    /// gives them the input this slot had, `Steal` takes it from them, `Allow` lets both keep it. Binding a stick to
    /// the analog X of a 2D action also binds its Y partner.
    ///
    /// **Queued (obj).** See [`input_command`](Self::input_command). The change is saved to the player's overrides
    /// file. The source normally comes from [`captured_input`](Self::captured_input).
    pub fn rebind_with(&mut self, scheme: &str, action: &str, slot: crate::BindingSlot, index: usize, source: crate::InputSource, policy: crate::ConflictPolicy) {
        self.input_command(crate::InputCommand::Rebind { scheme: scheme.into(), action: action.into(), slot, index, source, policy });
    }
    /// Puts every action of a control scheme back to the project's default bindings (`""` is the active scheme),
    /// discarding the player's rebinds for it.
    ///
    /// **Queued (obj).** See [`input_command`](Self::input_command). An unknown scheme is logged and ignored.
    pub fn reset_scheme_bindings(&mut self, scheme: &str) { self.input_command(crate::InputCommand::ResetScheme(scheme.into())); }
    /// Lists the inputs that are bound to more than one action in a control scheme (`""` is the active scheme).
    ///
    /// **Immediate.** Empty when there are no conflicts, the scheme does not exist, or there is no screen service.
    /// Handy for warning the player in a controls menu after a rebind made with
    /// [`ConflictPolicy::Allow`](crate::ConflictPolicy::Allow).
    pub fn binding_conflicts(&self, scheme: &str) -> Vec<crate::BindingConflict> {
        self.input_bindings().and_then(|bindings| bindings.scheme(scheme)).map(|scheme| scheme.conflicts()).unwrap_or_default()
    }
    /// Sets the volume of an audio mixer bus (for example `"Music"` or `"Effects"`) as a linear gain.
    ///
    /// `1.0` leaves the bus unchanged and `0.0` silences it; negative values are clamped to zero, and values above
    /// `1.0` amplify. The result is multiplied with the volumes of the bus's parent buses.
    ///
    /// **Queued.** Applied at the stage boundary. A bus name that is not in the project's Audio Mixer is silently
    /// ignored.
    pub fn set_bus_volume(&mut self, bus: impl Into<String>, volume: f32) { self.commands.push(ComponentCommand::SetBusVolume { bus: bus.into(), volume }); }
    /// Sets the game speed: `1.0` is normal, `0.5` half speed, `0.0` paused. Negative values count as `0.0`.
    ///
    /// **Queued.** Applied at the stage boundary. The current frame's `dt` was already computed, so the new scale
    /// shows in `dt` from the next frame; [`time_scale`](Self::time_scale) reads the new value from the next stage.
    /// `dt`, `time` and physics follow the scale, while `unscaled_dt` does not: drive pause menus and UI animation
    /// with `unscaled_dt` so they keep working while the game is paused.
    pub fn set_time_scale(&mut self, scale: f32) { self.commands.push(ComponentCommand::SetTimeScale(scale.max(0.0))); }
    /// Switches to another scene (a level change, menu to game).
    ///
    /// `path` is project-relative, for example `"scenes/level2.scene"`. The current scene is replaced: its objects
    /// are removed unless marked with [`keep_on_scene_change`](Self::keep_on_scene_change), and the new scene's
    /// components start on the next frame.
    ///
    /// **Queued (obj).** The player or editor performs the switch after the frame's stages have finished. When
    /// several loads are queued in one frame, the last one wins. In the editor's Play mode the new scene loads in the
    /// background and replaces the current one once it is ready (the old scene keeps running meanwhile), so a scene
    /// you have not [`preload_scene`](Self::preload_scene)d can take several frames. The switch is local: in a
    /// network session every peer would switch on its own; the host uses [`ComponentCommand::NetLoadScene`] to move
    /// everyone.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// if ctx.input.pressed("Confirm") {
    ///     ctx.keep_on_scene_change(self.player, true); // the player survives the switch
    ///     ctx.load_scene("scenes/level2.scene");
    /// }
    /// ```
    pub fn load_scene(&mut self, path: impl Into<String>) { self.commands.push(ComponentCommand::LoadScene(path.into())); }
    /// Starts loading a scene in the background, so that a later [`load_scene`](Self::load_scene) of the same path
    /// switches almost instantly.
    ///
    /// **Queued (obj).** Does not change the current scene. Call it early (when the player nears an exit, say) and
    /// pass the same project-relative path to `load_scene` later.
    pub fn preload_scene(&mut self, path: impl Into<String>) { self.commands.push(ComponentCommand::PreloadScene(path.into())); }
    /// Adds another scene's objects to the current scene without replacing it (streaming rooms or level chunks).
    ///
    /// `path` is project-relative. **Queued (obj).** The objects appear once the host has loaded the scene, which
    /// is not guaranteed to be within the same frame. Remove them again with
    /// [`unload_scene`](Self::unload_scene) using the same path.
    pub fn load_scene_additive(&mut self, path: impl Into<String>) { self.commands.push(ComponentCommand::LoadSceneAdditive(path.into())); }
    /// Removes the objects that an additive load brought in.
    ///
    /// Pass the same project-relative `path` given to [`load_scene_additive`](Self::load_scene_additive).
    /// **Queued (obj).** Does nothing if that scene was not loaded additively.
    pub fn unload_scene(&mut self, path: impl Into<String>) { self.commands.push(ComponentCommand::UnloadScene(path.into())); }
    /// Marks an object (and its children) to survive the next [`load_scene`](Self::load_scene) (`keep = true`), or
    /// clears that mark (`keep = false`). Use it for the player, a music manager or other persistent objects.
    ///
    /// **Queued (obj).** No-op for an unset reference. Set the mark in the same frame as `load_scene` or earlier.
    pub fn keep_on_scene_change(&mut self, target: impl Into<ObjectRef>, keep: bool) {
        if let Some(id) = target.into().id() { self.commands.push(ComponentCommand::KeepOnSceneChange(id, keep)); }
    }
    /// Exits the game. In the editor it ends Play mode instead.
    ///
    /// **Queued (obj).** Applied after the frame's stages have finished; the rest of the frame still runs.
    pub fn quit(&mut self) { self.commands.push(ComponentCommand::Quit); }
    /// The **world** position of an object.
    ///
    /// **Immediate.** `None` if the target does not resolve (see [`entity_of`](Self::entity_of)). Includes moves made
    /// earlier in this stage. Takes an [`ObjectRef`] only; convert an [`ObjectId`] with `.into()`. For rotation and
    /// scale use [`transform_of`](Self::transform_of).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Is the player within 5 metres?
    /// let near = ctx.position_of(self.player)
    ///     .is_some_and(|player| player.distance(ctx.self_transform().translation) < 5.0);
    /// ```
    pub fn position_of(&self, target: ObjectRef) -> Option<Vec3> {
        let entity = self.world.entity(target.id()?)?;
        self.context.is_visible(self.world, entity).then(|| self.world.world_transform(entity)).flatten().map(|matrix| matrix.w_axis.truncate())
    }
    /// The world transform of the first [`SpawnPoint`](crate::SpawnPoint) with this tag, or `None` if there is none.
    ///
    /// Shorthand for [`spawn_point_for`](Self::spawn_point_for) with slot `0`. **Immediate.**
    pub fn spawn_point(&self, tag: &str) -> Option<Transform> {
        self.spawn_point_for(tag, 0)
    }
    /// The world transform of a spawn point with this `tag`, picked by `slot`.
    ///
    /// Slot `0` is the first point in authoring order, `1` the second, and so on. The slot wraps around (modulo the
    /// number of points), so any number of players or enemies can share a few points. `None` when no visible object has a
    /// [`SpawnPoint`](crate::SpawnPoint) with that tag.
    ///
    /// **Immediate.** Disabled points count. The order is the scene's authoring order with the saved id as
    /// tie-break; the point's own `index` field is not consulted here.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Put player number `n` at its spawn point.
    /// if let Some(spawn) = ctx.spawn_point_for("player", n) {
    ///     ctx.set_world_transform_of(player, Some(spawn.translation), Some(spawn.rotation));
    /// }
    /// ```
    pub fn spawn_point_for(&self, tag: &str, slot: usize) -> Option<Transform> {
        let mut points = Vec::new();
        for (entity, (_, point)) in self.world.query::<(&ObjectId, &crate::SpawnPoint)>().iter() {
            if point.tag != tag || !self.context.is_visible(self.world, entity) { continue; }
            let key = self.context.order_key(self.world, entity);
            points.push((key, entity));
        }
        if points.is_empty() { return None; }
        points.sort_unstable_by_key(|(key, _)| *key);
        self.world.world_transform(points[slot % points.len()].1).map(Transform::from_matrix)
    }
    /// Makes a camera object the one the game renders through.
    ///
    /// **Queued.** Takes effect at the stage boundary and stays until another call, or until that camera is
    /// destroyed (which clears it). No-op for an unset reference; the id is not validated when queued. Takes an
    /// [`ObjectRef`] only.
    pub fn set_active_camera(&mut self, target: ObjectRef) {
        if let Some(id) = target.id() { self.commands.push(ComponentCommand::SetActiveCamera(id)); }
    }
    /// Locks and hides the operating system mouse cursor (for mouse-look), or releases it.
    ///
    /// **Queued (obj).** Applied at the end of the frame.
    pub fn set_cursor_locked(&mut self, locked: bool) { self.commands.push(ComponentCommand::SetCursorLocked(locked)); }
    /// Enables or disables an object, and with it its whole subtree.
    ///
    /// A disabled object does not run its systems, render or collide; its children are inactive with it, although
    /// their own enabled flags are untouched. `true` re-enables it.
    ///
    /// **Queued.** Applied at the stage boundary; [`is_active`](Self::is_active) shows it from the next stage. No-op
    /// for an unset reference, for an object that no longer exists, and when the object already has that state. Not
    /// replicated over the network. Because a disabled object runs none of its own systems, enable it again from
    /// another object. A body that is re-enabled gets its physics body at the next physics sync, so an impulse or
    /// velocity aimed at it in the same stage is dropped.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Hide a pickup and show its sparkle effect.
    /// ctx.set_active(self.pickup, false);
    /// ctx.set_active(self.sparkles, true);
    /// ```
    pub fn set_active(&mut self, target: impl Into<ObjectRef>, active: bool) {
        if let Some(id) = target.into().id() { self.commands.push(ComponentCommand::SetActive(id, active)); }
    }
    /// Sets the linear velocity of a dynamic rigid body, in world space, in metres per second.
    ///
    /// The previous linear velocity is replaced (use [`add_impulse`](Self::add_impulse) to add to it); angular
    /// velocity is untouched. Only dynamic bodies are affected, 3D or 2D (a 2D body uses `x` and `y`). A kinematic or
    /// static body, a character-controller object, or an object without a physics body ignores it; move characters
    /// with [`move_character`](MotionContext::move_character).
    ///
    /// **Queued.** Applied at the stage boundary; the body starts moving on the next physics step. No-op for an
    /// unset reference. Silently dropped when the body does not exist yet (spawned or re-enabled in the same stage).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Bounce pad: cancel the fall and fire the body upwards.
    /// ctx.set_velocity(other, Vec3::new(0.0, 12.0, 0.0));
    /// ```
    pub fn set_velocity(&mut self, target: impl Into<ObjectRef>, linear: Vec3) {
        if let Some(id) = target.into().id() { self.commands.push(ComponentCommand::SetVelocity { id, linear }); }
    }
    /// Asks to become the authority over a networked object, so this peer simulates it and publishes its state.
    ///
    /// Whether the request is granted depends on the object's [`ownership mode`](crate::OwnershipMode).
    ///
    /// **Queued (obj).** Forwarded to the network session at the end of the frame and granted or refused
    /// asynchronously, so check [`has_authority`](Self::has_authority) on later frames instead of assuming success.
    /// Does nothing offline, or for an unset reference. Takes an [`ObjectRef`] only.
    pub fn request_ownership_of(&mut self, target: ObjectRef) {
        if let Some(id) = target.id() { self.commands.push(ComponentCommand::RequestOwnership(id)); }
    }
    /// Removes this object, and its children, from the scene.
    ///
    /// **Queued.** The object stays alive until the stage boundary: the rest of your system keeps running, other
    /// systems of this stage still see (and can still act on) it, and your component is not dropped yet. After the
    /// boundary every [`ObjectRef`] to it resolves to nothing. Queueing it twice is harmless. The removal is local:
    /// for a networked object that must disappear on every peer, push [`ComponentCommand::NetDespawn`] instead.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // A projectile that expires.
    /// self.age += ctx.dt;
    /// if self.age > self.lifetime {
    ///     ctx.destroy_self();
    ///     return;
    /// }
    /// ```
    pub fn destroy_self(&mut self) {
        let id = self.self_id();
        self.commands.push(ComponentCommand::Destroy(id));
    }
    /// Removes another object, and its children, from the scene.
    ///
    /// **Queued.** Applied at the stage boundary; until then the object still exists for every system of this stage.
    /// No-op for an unset reference or an object that is already gone; destroying the active camera clears the
    /// camera override. Local only: use [`ComponentCommand::NetDespawn`] to remove a networked object on every peer.
    /// See also [`destroy_self`](Self::destroy_self).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Pick up a coin: remove it and count it.
    /// ctx.destroy(coin);
    /// self.coins += 1;
    /// ```
    pub fn destroy(&mut self, target: impl Into<ObjectRef>) {
        if let Some(id) = target.into().id() { self.commands.push(ComponentCommand::Destroy(id)); }
    }
    /// Clones an existing scene object, with all of its children, as a new root object at a world position and
    /// rotation. Returns the id the clone will have, or `None` if `original` is unset.
    ///
    /// Use it to duplicate a template that already sits in the scene (usually disabled). To create objects from a
    /// file use [`spawn_prefab`](Self::spawn_prefab); for a registered entity use
    /// [`spawn_entity`](Self::spawn_entity).
    ///
    /// **Queued.** The clone is created at the stage boundary, so the returned id resolves from the next stage on;
    /// commands you queue after this call and aim at the id (such as [`set_active`](Self::set_active)) do find it.
    /// The id is returned without checking that `original` exists: if it is gone when the command is applied, nothing
    /// is created. The clone copies the original's components, **enabled state** and scale, so a disabled template
    /// produces a disabled clone. Only `translation` and `rotation` are replaced; the clone's children get fresh ids.
    /// Local only, see [`net_clone`](Self::net_clone) for the networked version.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Fire a bullet cloned from a disabled template object.
    /// if let Some(bullet) = ctx.instantiate(self.bullet_template, muzzle.translation, muzzle.rotation) {
    ///     ctx.set_active(bullet, true); // the template is disabled, so its clone is too
    /// }
    /// ```
    pub fn instantiate(&mut self, original: impl Into<ObjectRef>, translation: Vec3, rotation: Quat) -> Option<ObjectId> {
        let original = original.into().id()?;
        let new_id = ObjectId::new();
        self.commands.push(ComponentCommand::Instantiate { original, new_id, translation, rotation });
        Some(new_id)
    }
    /// Instantiates a prefab file as a new object tree and returns the id its root will have.
    ///
    /// `path` is project-relative, for example `"prefabs/bullet.prefab"`. `translation` (metres) and `rotation` are in
    /// world space and apply to the prefab's root; the first root gets the returned id, its children get ids the
    /// engine picks.
    ///
    /// **Queued.** The objects are created at the stage boundary, so they exist from the next stage and their start-up
    /// systems run at the beginning of the next frame. You can keep the returned id straight away, and commands you
    /// queue after the spawn and aim at it are applied after it, but reads find nothing for it until the next stage.
    /// The id is returned even when the spawn later fails: an unknown or unreadable prefab path is logged as an error
    /// when the command is applied and nothing appears. A prefab is loaded from disk the first time it is used and
    /// cached afterwards, so the first spawn of a given prefab does the file work. The spawn is local; use
    /// [`net_spawn`](Self::net_spawn) to create it on every peer.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Fire a rocket from the muzzle.
    /// let Some(muzzle) = ctx.transform_of(self.muzzle) else { return };
    /// let rocket = ctx.spawn_prefab("prefabs/rocket.prefab", muzzle.translation, muzzle.rotation);
    /// self.last_rocket = ObjectRef::new(rocket); // resolves from the next stage on
    /// ```
    pub fn spawn_prefab(&mut self, path: impl Into<String>, translation: Vec3, rotation: Quat) -> ObjectId {
        let root_id = ObjectId::new();
        self.commands.push(ComponentCommand::SpawnPrefab { path: path.into(), root_id, translation, rotation });
        root_id
    }
    /// Creates a registered entity, by name, as a new root object and returns the id it will have.
    ///
    /// Entities are named templates that a game or plugin registers with
    /// [`ComponentRegistry::register_entity`](crate::ComponentRegistry::register_entity): a built-in mesh shape
    /// ([`EntityShape`](super::EntityShape)) plus the components a new instance starts with
    /// ([`EntityTemplate`](super::EntityTemplate)). The editor lists the same names under Create > Entity. Every spawn
    /// builds a fresh copy of the template.
    ///
    /// `translation` is the world position (metres) and `rotation` the world rotation. The new object is enabled,
    /// has scale `1` and is named after the entity.
    ///
    /// **Queued.** The object is created at the stage boundary (`ComponentCommand::SpawnEntity`), so it exists from
    /// the next stage and its start-up systems run at the beginning of the next frame; commands queued after the
    /// spawn and aimed at the id are applied after it. An unknown name (not registered, or its plugin is not loaded)
    /// is logged as an error when the command is applied and nothing is created, so the returned id then never
    /// resolves. The spawn is local, not replicated.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // "Zombie" was registered with `registry.register_entity("Zombie", zombie)`.
    /// let id = ctx.spawn_entity("Zombie", spawn.translation, spawn.rotation);
    /// self.horde.push(id);
    /// ```
    pub fn spawn_entity(&mut self, name: impl Into<String>, translation: Vec3, rotation: Quat) -> ObjectId {
        let root_id = ObjectId::new();
        self.commands.push(ComponentCommand::SpawnEntity { name: name.into(), root_id, translation, rotation });
        root_id
    }
    /// A copy of the current parameters of one material slot of an object, or `None`.
    ///
    /// `slot` `0` is the main material and `1`, `2`, ... are the object's extra render slots. Returns `None` when the
    /// target does not resolve, the object has no renderer in that slot, or there is no material service (for
    /// example in a headless run).
    ///
    /// **Immediate** read of the scene material. Edits queued with [`modify_material`](Self::modify_material) show
    /// here from the stage after the one that applied them. Changing the returned copy changes nothing; use
    /// `modify_material` to write.
    pub fn material_params(&self, target: impl Into<ObjectRef>, slot: usize) -> Option<crate::MaterialParams> {
        let entity = self.entity_of(target)?;
        self.services.materials?.material_params(self.world, self.world.object_id(entity)?, slot)
    }
    /// Edits one material slot of an object with a closure over its [`MaterialParams`](crate::MaterialParams)
    /// (colour, emission, metallic, roughness and the rest).
    ///
    /// `slot` `0` is the main material and `1`, `2`, ... the extra render slots.
    ///
    /// **Queued.** The closure does not run now. It is boxed (hence `'static`: move what it needs into it) and run at
    /// the stage boundary against the live parameters, the result is written back into the scene material, and the
    /// renderer's copy is refreshed. **The edit goes to the material the slot refers to, so every object that shares
    /// that material changes with it**; materials are not copied per object. No-op for an unset reference, an object
    /// without that slot, or an object that is gone. Not replicated over the network.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Flash the object red when it takes damage.
    /// ctx.modify_material(self.mesh, 0, |material| material.base_color = [1.0, 0.1, 0.1, 1.0]);
    /// ```
    pub fn modify_material(&mut self, target: impl Into<ObjectRef>, slot: usize, edit: impl FnOnce(&mut crate::MaterialParams) + 'static) {
        if let Some(id) = target.into().id() { self.commands.push(ComponentCommand::ModifyMaterialParams { id, slot, edit: crate::MaterialEdit(Box::new(edit)) }); }
    }
    /// A read-only view of the current save data (the key-value store behind
    /// [`save_value`](Self::save_value)), or `None` without a save service.
    ///
    /// **Immediate.** This is the data as it stood when the stage started, so a value queued with `save_value`
    /// during this stage is not visible yet. Use the typed getters of [`SaveGame`](crate::SaveGame): `bool`, `int`,
    /// `float`, `str` and `vec3`.
    pub fn save_data(&self) -> Option<&crate::SaveGame> { self.services.save }
    /// Stores a value under a key in the save data.
    ///
    /// `value` is anything that converts into a [`SaveValue`](crate::SaveValue): `bool`, `i64`, `f32`,
    /// `&str` or `String`, and [`Vec3`]. Other number types do not convert, so write `3i64` and `1.5f32`. Keys are
    /// free-form strings; dotted names such as `"player.hp"` keep related values together.
    ///
    /// **Queued.** Stored in memory at the stage boundary and readable through
    /// [`save_data`](Self::save_data) from the next stage. Nothing touches the disk until
    /// [`write_save`](Self::write_save) runs.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// ctx.save_value("checkpoint", 3i64);
    /// ctx.save_value("player.position", ctx.self_transform().translation);
    /// ctx.write_save("slot1"); // queued after the values above, so it includes them
    ///
    /// // Later, in a following stage or frame:
    /// let checkpoint = ctx.save_data().and_then(|save| save.int("checkpoint")).unwrap_or(0);
    /// ```
    pub fn save_value(&mut self, key: impl Into<String>, value: impl Into<crate::SaveValue>) {
        self.commands.push(ComponentCommand::SaveSet { key: key.into(), value: value.into() });
    }
    /// Deletes a key from the save data.
    ///
    /// **Queued.** Applied at the stage boundary; an unknown key is ignored.
    pub fn remove_save_value(&mut self, key: impl Into<String>) { self.commands.push(ComponentCommand::SaveRemove(key.into())); }
    /// Writes the current save data to a named slot on disk.
    ///
    /// `slot` is a name such as `"slot1"` or `"autosave"`. Characters other than letters, digits, `-` and `_` become
    /// `_`, and the file is `<slot>.save` in the game's per-user save folder; an existing file is replaced.
    ///
    /// **Queued.** Written at the stage boundary, after every command queued before it, so queue your
    /// [`save_value`](Self::save_value) calls first. I/O errors are logged, not reported back to the caller.
    pub fn write_save(&mut self, slot: impl Into<String>) { self.commands.push(ComponentCommand::SaveWrite(slot.into())); }
    /// Replaces the in-memory save data with the contents of a slot file.
    ///
    /// **Queued.** Applied at the stage boundary; [`save_data`](Self::save_data) returns the loaded values from the
    /// next stage. A missing or unreadable slot, or a save made with an incompatible mod set, is logged and leaves
    /// the current data unchanged. Loading only swaps the data: it does not move objects or change scene, so read the
    /// values and apply them yourself (for example with [`load_scene`](Self::load_scene) to the saved level).
    pub fn load_save(&mut self, slot: impl Into<String>) { self.commands.push(ComponentCommand::SaveLoad(slot.into())); }
    /// Translates a localisation key into the active language.
    ///
    /// Falls back to the first (default) language when the active column is empty, and to the key itself when the key
    /// is unknown or no localisation tables are loaded, so text never renders blank and a missing translation is easy
    /// to spot. **Immediate.**
    pub fn tr<'a>(&'a self, key: &'a str) -> &'a str { self.services.strings.map_or(key, |strings| strings.tr(key)) }
    /// The code of the active language (as named in the localisation tables), or `None` without a localisation
    /// service.
    ///
    /// **Immediate.** It is an empty string when the tables define no languages.
    pub fn language(&self) -> Option<&str> { Some(self.services.strings?.current_language()) }
    /// Switches the active language by code.
    ///
    /// **Queued.** Applied at the stage boundary; [`tr`](Self::tr) uses the new language from the next stage. A code
    /// that has no column in the localisation tables is logged as a warning and the language stays as it was.
    pub fn set_language(&mut self, language: impl Into<String>) { self.commands.push(ComponentCommand::SetLanguage(language.into())); }
    /// The current game-speed multiplier (`1.0` is normal), or `1.0` when no service provides one.
    ///
    /// **Immediate.** See [`set_time_scale`](Self::set_time_scale).
    pub fn time_scale(&self) -> f32 { self.services.time_scale.unwrap_or(1.0) }
    /// Gives a rigid body an instant push: its velocity changes by `impulse / mass` at once.
    ///
    /// `impulse` is in world space, in newton-seconds (kg·m/s). It acts at the body's centre of mass, so it never
    /// spins the body, and heavier bodies move less. Only dynamic rigid bodies are affected, 3D or 2D (a 2D body uses
    /// `x` and `y`); anything else ignores it.
    ///
    /// **Queued.** Applied at the stage boundary and effective from the next physics step. No-op for an unset
    /// reference. Silently dropped when the body does not exist yet (spawned or re-enabled in the same stage). For a
    /// continuous force, push [`ComponentCommand::AddForce`] onto `commands` every frame.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Shove whatever the bullet hit along the shot direction.
    /// ctx.add_impulse(hit.object, dir * 4.0);
    /// ```
    pub fn add_impulse(&mut self, target: impl Into<ObjectRef>, impulse: Vec3) {
        if let Some(id) = target.into().id() { self.commands.push(ComponentCommand::AddImpulse { id, impulse }); }
    }
    /// Makes a [`ParticleEmitter`](crate::ParticleEmitter) release `count` extra particles right away (muzzle
    /// flashes, hit sparks).
    ///
    /// **Immediate.** Adds `count` to the emitter's burst counter, which the particle system consumes when it next
    /// updates; repeated calls add up. No-op if the target does not resolve or has no `ParticleEmitter`. To burst
    /// somewhere else, move the emitter first (for example with
    /// [`set_world_transform_of`](Self::set_world_transform_of)).
    pub fn emit_particles(&self, target: impl Into<ObjectRef>, count: u32) {
        self.edit::<crate::ParticleEmitter>(target, |emitter| emitter.control.emit += count);
    }
    /// Plays a one-shot sound from a world position.
    ///
    /// `clip` is a project-relative audio file (for example `"audio/explosion.wav"`), `position` is the world
    /// position it comes from (the sound is spatialised relative to the listener), and `volume` is a linear gain
    /// (`1.0` is the clip's own level; `0.0` to `1.0` is the intended range). It is fire-and-forget: you get no
    /// handle and cannot stop or move the sound. For a non-spatial sound push [`ComponentCommand::PlaySound`] with
    /// `position: None` onto `commands`; to control a sound on an object use its audio source component.
    ///
    /// **Queued.** Starts at the stage boundary. Nothing plays when no audio engine is running (a headless server).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// ctx.play_sound_at("audio/impact.wav", hit.point, 0.8);
    /// ```
    pub fn play_sound_at(&mut self, clip: impl Into<String>, position: Vec3, volume: f32) {
        self.commands.push(ComponentCommand::PlaySound { clip: clip.into(), position: Some(position), volume });
    }
    /// Writes any of another object's **local** (parent-relative) position, rotation and scale. A field passed as
    /// `None` is left alone.
    ///
    /// **Immediate.** The `Transform` component is edited in place before the call returns, so later systems of the
    /// stage, including [`transform_of`](Self::transform_of), see the move (a weapon that follows a camera an earlier
    /// system just aimed, for instance). No-op if the target does not resolve. Takes an [`ObjectRef`] only. For world
    /// values use [`set_world_transform_of`](Self::set_world_transform_of); to move this object use
    /// [`set_translation`](MotionContext::set_translation) and [`set_rotation`](MotionContext::set_rotation).
    ///
    /// Writing a transform is a teleport that does no collision. Move character controllers with
    /// [`move_character`](MotionContext::move_character) and rigid bodies with
    /// [`set_velocity`](Self::set_velocity) or [`add_impulse`](Self::add_impulse). Pass a unit quaternion for
    /// `rotation`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Pitch a child camera without touching its position or scale.
    /// ctx.set_local_transform(self.camera, None, Some(Quat::from_rotation_x(self.pitch)), None);
    /// ```
    pub fn set_local_transform(&mut self, target: ObjectRef, translation: Option<Vec3>, rotation: Option<Quat>, scale: Option<Vec3>) {
        self.edit::<Transform>(target, |transform| {
            if let Some(value) = translation { transform.translation = value; }
            if let Some(value) = rotation { transform.rotation = value; }
            if let Some(value) = scale { transform.scale = value; }
        });
    }
    /// Teleports another object to a **world** position and/or rotation. A part passed as `None` is kept.
    ///
    /// The values are converted into the object's local space through its parent chain, so the object lands where you
    /// ask whether it is a root or nested under a moving parent. **Immediate**: the `Transform` component is edited in
    /// place and later systems of the stage see it. No-op if the target does not resolve. Scale is not changed, and
    /// nothing collides on the way. Takes an [`ObjectRef`] only.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Put an impact effect on the surface, facing out of it, then burst it.
    /// ctx.set_world_transform_of(self.hit_effect, Some(hit.point + hit.normal * 0.02), Some(facing));
    /// ctx.emit_particles(self.hit_effect, 16);
    /// ```
    pub fn set_world_transform_of(&mut self, target: ObjectRef, translation: Option<Vec3>, rotation: Option<Quat>) {
        let Some(entity) = self.entity_of(target) else { return };
        let parent = self.world.get::<Parent>(entity).and_then(|parent| parent.0)
            .and_then(|parent| self.world.world_transform(parent));
        if let Some(mut transform) = self.world.get_mut::<Transform>(entity) {
            if let Some(value) = translation { transform.translation = parent.map_or(value, |parent| parent.inverse().transform_point3(value)); }
            if let Some(value) = rotation { transform.rotation = parent.map_or(value, |parent| parent.to_scale_rotation_translation().1.inverse() * value); }
        }
    }
    /// Teleports another object to a **world** position, keeping its rotation.
    ///
    /// Shorthand for [`set_world_transform_of`](Self::set_world_transform_of) with only a translation, and the same
    /// rules: **immediate**, converted through the parent chain, no-op if the target does not resolve. Unlike that
    /// method it accepts anything that converts into an [`ObjectRef`].
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Move a marker onto the clicked ground point, slightly above it.
    /// ctx.set_world_position_of(self.marker, hit.point + Vec3::Y * 0.03);
    /// ```
    pub fn set_world_position_of(&mut self, target: impl Into<ObjectRef>, translation: Vec3) {
        self.set_world_transform_of(target.into(), Some(translation), None);
    }
    /// Sets **this** object's rotation in world space (converted to local through its parent's rotation).
    ///
    /// **Immediate.** The counterpart of [`world_position`](MotionContext::world_position) and
    /// [`set_world_position`](MotionContext::set_world_position) for rotation. To rotate another object use
    /// [`set_world_transform_of`](Self::set_world_transform_of).
    pub fn set_world_rotation(&mut self, rotation: Quat) {
        let local = self.parent_world().map_or(rotation, |parent| parent.to_scale_rotation_translation().1.inverse() * rotation);
        self.set_rotation(local);
    }
    /// Whether a UI text field currently has the keyboard.
    ///
    /// **Immediate.** While it is `true`, gameplay should ignore keyboard input (typing `w` must not walk the
    /// player). `false` without a screen service.
    pub fn ui_text_editing(&self) -> bool { self.services.screen.is_some_and(|screen| screen.ui_text_editing) }
    /// Whether the pointer is over an interactive UI element.
    ///
    /// **Immediate.** Skip world clicks (picking, shooting) while it is `true`. `false` without a screen service.
    pub fn pointer_over_ui(&self) -> bool { self.services.screen.is_some_and(|screen| screen.pointer_over_ui) }
    /// The size of the game viewport in pixels, or `Vec2::ZERO` without a screen service.
    ///
    /// **Immediate.**
    pub fn screen_size(&self) -> Vec2 { self.services.screen.map_or(Vec2::ZERO, |screen| screen.size) }
    /// The pointer position in pixels from the top-left corner of the game viewport.
    ///
    /// **Immediate.** `None` when the pointer is outside the viewport or there is no screen service.
    pub fn pointer_position(&self) -> Option<Vec2> { self.services.screen?.pointer }
    /// Turns a viewport pixel into a world-space ray `(origin, direction)` through the active camera. The direction
    /// is normalised.
    ///
    /// Pixels are measured from the viewport's top-left corner, like [`pointer_position`](Self::pointer_position).
    /// **Immediate.** `None` without a screen service, when the viewport is smaller than one pixel, or when the camera
    /// matrices cannot be inverted. Feed the ray to [`raycast`](Self::raycast) to pick what is under a pixel.
    pub fn screen_ray(&self, pixel: Vec2) -> Option<(Vec3, Vec3)> {
        let screen = self.services.screen?;
        if screen.size.x < 1.0 || screen.size.y < 1.0 { return None; }
        let ndc = Vec2::new(2.0 * pixel.x / screen.size.x - 1.0, 1.0 - 2.0 * pixel.y / screen.size.y);
        let inverse = (screen.proj * screen.view).inverse();
        let near = inverse.project_point3(ndc.extend(0.0));
        let far = inverse.project_point3(ndc.extend(1.0));
        Some((near, (far - near).try_normalize()?))
    }
    /// The world-space ray `(origin, direction)` through the pointer: [`screen_ray`](Self::screen_ray) applied to
    /// [`pointer_position`](Self::pointer_position).
    ///
    /// **Immediate.** `None` when the pointer is outside the viewport or there is no screen service. Check
    /// [`pointer_over_ui`](Self::pointer_over_ui) first so UI clicks do not also pick world objects.
    pub fn pointer_ray(&self) -> Option<(Vec3, Vec3)> { self.screen_ray(self.pointer_position()?) }
    /// Projects a world position to viewport pixels (origin top-left), for placing UI over objects such as health
    /// bars.
    ///
    /// **Immediate.** `None` without a screen service or when the point is behind the camera. The result is not
    /// clamped to the viewport, so compare it with [`screen_size`](Self::screen_size) to tell whether the point is on
    /// screen.
    pub fn world_to_screen(&self, position: Vec3) -> Option<Vec2> {
        let screen = self.services.screen?;
        let clip = screen.proj * screen.view * position.extend(1.0);
        if clip.w <= 1e-6 { return None; }
        let ndc = clip.truncate() / clip.w;
        Some(Vec2::new((ndc.x + 1.0) * 0.5 * screen.size.x, (1.0 - ndc.y) * 0.5 * screen.size.y))
    }
    /// The closest point on the navigation mesh to `position` (within a few metres), or `None`.
    ///
    /// **Immediate.** `None` when there is no walkable ground nearby, and also when there is no baked navigation mesh
    /// (not baked yet, or stale after [`rebake_navmesh`](Self::rebake_navmesh)). Use it to snap a clicked point
    /// before [`set_destination`](Self::set_destination).
    pub fn nearest_walkable(&self, position: Vec3) -> Option<Vec3> { self.services.nav?.nearest_walkable(position) }
    /// Orders a [`NavAgent`](crate::NavAgent) to walk to a world position.
    ///
    /// `target` is the object that carries the agent (often yourself, `ctx.self_id()`); `position` is in world
    /// space. **Immediate**: the destination is written into the agent component now and flagged for re-planning; the
    /// navigation system moves the agent after the `Gameplay` stage of this frame. No-op if the target does not
    /// resolve or has no `NavAgent`. Poll [`NavAgent::arrived`](crate::NavAgent::arrived) (true when the agent
    /// reached the destination or cannot reach it) to know when it is done.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Click to move.
    /// if ctx.input.pressed("Click") && !ctx.pointer_over_ui() {
    ///     let hit = ctx.pointer_ray().and_then(|(origin, dir)| ctx.raycast(origin, dir, 200.0));
    ///     if let Some(hit) = hit {
    ///         let goal = ctx.nearest_walkable(hit.point).unwrap_or(hit.point);
    ///         ctx.set_destination(ctx.self_id(), goal);
    ///     }
    /// }
    /// ```
    pub fn set_destination(&self, target: impl Into<ObjectRef>, position: Vec3) {
        self.edit::<crate::NavAgent>(target, |agent| agent.set_destination(position));
    }
    /// Cancels a [`NavAgent`](crate::NavAgent)'s destination and path and stops it where it is.
    ///
    /// **Immediate.** No-op if the target does not resolve or has no `NavAgent`.
    pub fn stop_agent(&self, target: impl Into<ObjectRef>) { self.edit::<crate::NavAgent>(target, |agent| agent.stop()); }
    /// Rebuilds the navigation mesh from the current level geometry. Call it after the level changed (a door opened,
    /// a bridge appeared).
    ///
    /// **Queued.** The mesh is marked stale at the stage boundary and baked again after a following physics step;
    /// until the new mesh is ready the navigation queries ([`nearest_walkable`](Self::nearest_walkable)) return
    /// `None`. Baking is expensive: do not call it every frame.
    pub fn rebake_navmesh(&mut self) { self.commands.push(ComponentCommand::RebakeNavMesh); }
    /// Whether this machine is the one that simulates the object, that is, may change its state and have that
    /// replicated.
    ///
    /// In a session an object's authority belongs to its owner peer, or to the host when nobody owns it. **Offline
    /// (no session) it is always `true`.** Takes an [`ObjectId`], not an [`ObjectRef`]. **Immediate.**
    ///
    /// It does not check that the object is networked: on a client, an ordinary scene object with no recorded owner
    /// reports `false`. Only ask about objects that have a [`NetworkObject`](crate::NetworkObject).
    pub fn has_authority(&self, target: ObjectId) -> bool { self.services.net.is_none_or(|net| net.has_authority(target)) }
    /// Shorthand for [`set_net_var`](Self::set_net_var) on this object. Returns what it returns.
    pub fn set_net_var_self(&mut self, key: &str, value: impl Into<crate::NetValue>) -> bool {
        self.set_net_var(self.self_id(), key, value)
    }
    /// Writes a synced variable on a networked object (a key and value on its
    /// [`NetworkVariables`](crate::NetworkVariables)) and replicates it to the other peers.
    ///
    /// Returns `true` when the write was accepted and `false` when it was refused: the target does not resolve, or
    /// the local peer is neither the object's authority ([`has_authority`](Self::has_authority)) nor allowed to write
    /// (`anyone_can_write` on its `NetworkVariables`). Offline you always have authority, so it succeeds.
    ///
    /// **Immediate for the local copy, queued for the network.** The value in the object's `NetworkVariables`
    /// component is updated at once (later systems of this stage read it with [`get`](Self::get)), and a command
    /// carries it to the network session, which replicates it at the end of the frame. Other peers receive it as a
    /// variable-changed event. `value` is anything that converts into a [`NetValue`](crate::NetValue): `bool`, `i32`,
    /// `i64`, `u32`, `f32`, [`Vec3`], [`Quat`], `&str` or `String`, and `Vec<u8>`. It also returns `true` for an object
    /// that has no `NetworkVariables` component, in which case there is no local copy to read.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // The authority publishes its score; every peer can read it from the object's NetworkVariables.
    /// if ctx.set_net_var_self("score", self.score) {
    ///     // accepted: this peer is the authority (always the case offline)
    /// }
    /// let score = ctx.get_self::<NetworkVariables>().and_then(|vars| vars.get("score").cloned());
    /// ```
    pub fn set_net_var(&mut self, target: impl Into<ObjectRef>, key: &str, value: impl Into<crate::NetValue>) -> bool {
        let Some(entity) = self.entity_of(target) else { return false };
        let id = self.world.object_id(entity).expect("network variable owner");
        let anyone = self.world.get::<crate::NetworkVariables>(entity).is_some_and(|vars| vars.anyone_can_write);
        if !anyone && !self.has_authority(id) { return false; }
        let value = value.into();
        if let Some(mut vars) = self.world.get_mut::<crate::NetworkVariables>(entity) { vars.set(key, value.clone()); }
        self.commands.push(ComponentCommand::NetSetVar { id, key: key.to_owned(), value });
        true
    }
    /// Sends a named remote call (RPC) on this object to the peers chosen by `target`.
    ///
    /// `name` identifies the call and `args` carry its values ([`NetValue`](crate::NetValue)). Receivers read the call
    /// with [`rpcs_for_self`](Self::rpcs_for_self) on the same object and match on `name`. `target`
    /// ([`RpcTarget`](crate::RpcTarget)) is `All` (everyone, the sender included), `Others`, `Server` (the host only),
    /// `Owner` (the object's owner, the host when unowned) or `Peer(id)`.
    ///
    /// **Queued (obj).** With a session running, the call is sent reliably at the end of the frame and arrives on
    /// the receivers a frame or more later; the host may refuse calls from clients (per-RPC permissions, rate
    /// limits). **Offline** there are no other peers, so the call is looped back to yourself as an event instead of a
    /// command: it reaches [`rpcs_for_self`](Self::rpcs_for_self) from the next frame, with `from == 0`, for every
    /// target except `Others` (nobody to reach) and `Peer(n)` with `n` other than `0`.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Ask the object's owner to open the door; the owner handles it in its own system.
    /// ctx.rpc_self("open", vec![NetValue::Bool(true)], RpcTarget::Owner);
    /// ```
    pub fn rpc_self(&mut self, name: &str, args: Vec<crate::NetValue>, target: crate::RpcTarget) {
        let object = Some(self.self_id());
        if !self.connected() {
            let local = match target { crate::RpcTarget::Others => false, crate::RpcTarget::Peer(peer) => peer == 0 || peer == self.local_peer(), _ => true };
            if local { self.emit(crate::NetRpc { from: 0, object, name: name.to_owned(), args }); }
        } else {
            self.commands.push(ComponentCommand::NetRpc { object, name: name.to_owned(), args, target, reliable: true });
        }
    }
    /// The RPCs addressed to this object that arrived since the last frame, in arrival order. Iterate it every frame
    /// and match on `rpc.name`.
    ///
    /// It includes calls other peers sent, calls you sent to yourself (for example `RpcTarget::All`), and offline
    /// loopback calls (`from == 0`). Global RPCs sent without an object are not included. Filter on `rpc.from` if you
    /// only want remote calls. **Immediate** read; each call is delivered once.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// for rpc in ctx.rpcs_for_self().filter(|rpc| rpc.name == "open") {
    ///     let open = rpc.arg(0).and_then(NetValue::as_bool).unwrap_or(true);
    ///     self.set_open(ctx, open);
    /// }
    /// ```
    pub fn rpcs_for_self(&self) -> impl Iterator<Item = &'frame crate::NetRpc> + use<'frame> {
        let id = self.self_id();
        let incoming = self.services.net.map_or(&[][..], |net| net.rpcs.as_slice());
        incoming.iter().chain(self.context.events::<crate::NetRpc>()).filter(move |rpc| rpc.object == Some(id))
    }
    /// Whether a network session is running (hosting or joined).
    ///
    /// **Immediate.** `false` offline.
    pub fn connected(&self) -> bool { self.services.net.is_some_and(|net| net.connected) }
    /// A read-only view of the network session: owners, peers, round-trip time, the session clock, events and so on
    /// ([`NetView`](crate::NetView)).
    ///
    /// **Immediate.** Offline (and in standalone worlds with no network service) it is an empty, disconnected view,
    /// so you can read it without checking [`connected`](Self::connected) first. No per-frame copy is made.
    pub fn net_view(&self) -> &crate::NetView {
        static OFFLINE: std::sync::OnceLock<crate::NetView> = std::sync::OnceLock::new();
        self.services.net.unwrap_or_else(|| OFFLINE.get_or_init(crate::NetView::default))
    }
    /// Whether this peer hosts the running session.
    ///
    /// **Immediate.** `false` offline and on clients; offline code should usually ask
    /// [`has_authority`](Self::has_authority) or [`connected`](Self::connected) instead.
    pub fn is_server(&self) -> bool { self.services.net.is_some_and(|net| net.connected && net.is_server) }
    /// Whether this peer has joined a running session hosted by someone else.
    ///
    /// **Immediate.** `false` offline and on the host.
    pub fn is_client(&self) -> bool { self.services.net.is_some_and(|net| net.connected && !net.is_server) }
    /// Everyone in the session, the local peer included ([`PeerInfo`](crate::PeerInfo)).
    ///
    /// **Immediate.** An empty slice offline.
    pub fn peers(&self) -> &[crate::PeerInfo] { self.services.net.map_or(&[], |net| &net.peers) }
    /// Spawns a networked object from any source (a prefab path or a clone of a scene object) with an explicit owner
    /// peer, on every peer. Returns the root id, which is the same on every peer.
    ///
    /// `owner` is the peer id that gets authority; `0` means the host. [`net_spawn`](Self::net_spawn),
    /// [`net_spawn_server`](Self::net_spawn_server) and [`net_clone`](Self::net_clone) are shorthands.
    ///
    /// **Queued.** The spawn is applied locally at the stage boundary (the object exists from the next stage) and the
    /// network session replicates it at the end of the frame; the other peers create the same object under the same
    /// id. A client's request is checked by the host (spawn allow-list, per-player quotas, range) and may be
    /// refused. **Offline** it is a plain local spawn.
    pub fn net_spawn_with(&mut self, source: crate::NetSpawnSource, translation: Vec3, rotation: Quat, owner: u64) -> ObjectId {
        let root_id = ObjectId::new();
        self.commands.push(ComponentCommand::NetSpawn { source, root_id, translation, rotation, owner });
        root_id
    }
    /// Spawns a prefab on every peer, owned by the local peer (so this peer simulates it). Returns the root id.
    ///
    /// `prefab` is a project-relative `.prefab` path, `translation` and `rotation` are in world space. Same timing as
    /// [`net_spawn_with`](Self::net_spawn_with): **queued**, locally from the next stage, replicated at the end of the
    /// frame, refusable by the host when requested by a client, and a plain local spawn offline.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Each player fires a rocket that they own and simulate.
    /// let rocket = ctx.net_spawn("prefabs/rocket.prefab", muzzle.translation, muzzle.rotation);
    /// ```
    pub fn net_spawn(&mut self, prefab: impl Into<String>, translation: Vec3, rotation: Quat) -> ObjectId {
        self.net_spawn_with(crate::NetSpawnSource::Prefab(prefab.into()), translation, rotation, self.local_peer())
    }
    /// Spawns a prefab on every peer, owned by the host (peer `0`), so the server simulates it. Returns the root id.
    ///
    /// Use it for world objects the server controls (pickups, AI enemies) and call it on the host; a client's request
    /// may be refused. Otherwise it works like [`net_spawn`](Self::net_spawn).
    pub fn net_spawn_server(&mut self, prefab: impl Into<String>, translation: Vec3, rotation: Quat) -> ObjectId {
        self.net_spawn_with(crate::NetSpawnSource::Prefab(prefab.into()), translation, rotation, 0)
    }
    /// Clones an existing scene object, with its children, on every peer, owned by the local peer. Returns the root
    /// id, or `None` if `original` is unset.
    ///
    /// Timing and offline behaviour are those of [`net_spawn_with`](Self::net_spawn_with). Unlike
    /// [`instantiate`](Self::instantiate), the clone's root is always enabled, even if the template is disabled.
    pub fn net_clone(&mut self, original: impl Into<ObjectRef>, translation: Vec3, rotation: Quat) -> Option<ObjectId> {
        let original = original.into().id()?;
        Some(self.net_spawn_with(crate::NetSpawnSource::Clone(original), translation, rotation, self.local_peer()))
    }
    /// This machine's peer id.
    ///
    /// **Immediate.** `0` offline.
    pub fn local_peer(&self) -> u64 { self.services.net.map_or(0, |net| net.local_peer) }
    /// The headset and controller state for this frame, or an inactive default without a screen service.
    ///
    /// **Immediate.** Poses are in the tracking (stage) space, not in world space; a VR player object maps them
    /// into the world through its own transform. Check `active` before using them.
    pub fn vr_frame(&self) -> crate::VrFrame { self.services.screen.map_or_else(crate::VrFrame::default, |screen| screen.vr) }
    /// Whether a body calibration step (height measurement, tracker T-pose) is running.
    ///
    /// **Immediate.** `false` without a screen service.
    pub fn body_calibrating(&self) -> bool { self.services.screen.is_some_and(|screen| screen.body.calibrating) }
    /// The active control scheme (keyboard and mouse, gamepad or VR), or `None` without a screen service or schemes.
    ///
    /// **Immediate.** Useful for choosing which button prompts to show.
    pub fn control_scheme(&self) -> Option<&crate::ControlScheme> { self.services.screen?.controls.bindings.active_scheme() }
    /// Casts a ray into the 3D physics world and returns the nearest hit, using a [`RayFilter`] you control.
    ///
    /// It is [`raycast`](Self::raycast) except that nothing is excluded unless you set `filter.exclude`: a ray that
    /// starts inside your own collider hits yourself. `filter.layer_mask` picks the layers (one bit per layer) and
    /// `filter.include_triggers` whether trigger colliders count. **Immediate.**
    pub fn raycast_filtered(&self, origin: Vec3, dir: Vec3, distance: f32, filter: &RayFilter) -> Option<RayHit> {
        MotionContext::raycast(self, origin, dir, distance, filter)
    }
    /// Casts a ray into the 3D physics world and returns the nearest hit, ignoring this object's own colliders.
    ///
    /// `origin` and `dir` are in world space; `dir` is normalised internally and a zero direction gives `None`.
    /// `distance` is the maximum length in metres. Returns a [`RayHit`] (`object`, `point`, `normal`, `distance`), or
    /// `None` when nothing is in range or there is no physics world. It tests every layer and ignores trigger
    /// colliders; use [`raycast_filtered`](Self::raycast_filtered) to change that.
    ///
    /// **Immediate** query. It sees colliders as of the last physics step: a collider that you moved earlier in the
    /// frame is still at its old position until the next step.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Hitscan: damage whatever is in front of the muzzle.
    /// let aim = ctx.self_transform();
    /// if let Some(hit) = ctx.raycast(aim.translation, aim.forward(), 100.0) {
    ///     ctx.edit::<Health>(hit.object, |health| { health.apply_damage(10.0); });
    /// }
    /// ```
    pub fn raycast(&self, origin: Vec3, dir: Vec3, distance: f32) -> Option<RayHit> {
        MotionContext::raycast(self, origin, dir, distance, &RayFilter { exclude: Some(self.self_id()), ..Default::default() })
    }
    /// The 2D counterpart of [`raycast_filtered`](Self::raycast_filtered): a ray in the XY plane against the 2D
    /// physics world, with a [`RayFilter`] you control (nothing is excluded unless you set `filter.exclude`).
    ///
    /// **Immediate.** Returns a [`RayHit2D`](crate::RayHit2D), or `None` when nothing is in range or there is no 2D
    /// physics world.
    pub fn raycast2d_filtered(&self, origin: Vec2, dir: Vec2, distance: f32, filter: &RayFilter) -> Option<crate::RayHit2D> {
        self.services.physics?.raycast2d(origin, dir, distance, filter)
    }
    /// The 2D counterpart of [`raycast`](Self::raycast): the nearest hit of a ray in the XY plane, ignoring this
    /// object's own colliders.
    ///
    /// **Immediate.** Returns a [`RayHit2D`](crate::RayHit2D), or `None` when nothing is in range or there is no 2D
    /// physics world.
    pub fn raycast2d(&self, origin: Vec2, dir: Vec2, distance: f32) -> Option<crate::RayHit2D> {
        self.raycast2d_filtered(origin, dir, distance, &RayFilter { exclude: Some(self.self_id()), ..Default::default() })
    }
}

/// Gives the context the frame data of its [`SystemContext`]: `dt`, `unscaled_dt`, `time`, `water_time`, `input`,
/// `commands` and `services`, plus [`events`](SystemContext::events) and [`emit`](SystemContext::emit). See
/// "What you get through `Deref`" on [`ObjectContext`].
impl<'world, 'frame> Deref for ObjectContext<'world, 'frame> {
    type Target = SystemContext<'frame>;
    /// The frame's [`SystemContext`]. This is what makes `ctx.dt`, `ctx.input`, `ctx.commands` and `ctx.events::<E>()`
    /// work on the context.
    fn deref(&self) -> &Self::Target { self.context }
}
/// Mutable access to the [`SystemContext`], which is what lets `ctx.emit(..)` and `ctx.commands.push(..)` work on
/// the context.
impl DerefMut for ObjectContext<'_, '_> {
    /// The frame's [`SystemContext`], mutably (for `ctx.emit(..)` and `ctx.commands.push(..)`).
    fn deref_mut(&mut self) -> &mut Self::Target { self.context }
}

/// The movement interface of the owner object, shared with the engine's prediction replay (see [`MotionContext`]).
///
/// Reads and in-place writes here are **immediate**; [`move_character`](MotionContext::move_character) is
/// **queued**. Besides the methods below, [`MotionContext`] provides `world`, `world_position` and
/// `set_world_position`, which convert between this object's local values and world space through
/// [`parent_world`](MotionContext::parent_world).
impl MotionContext for ObjectContext<'_, '_> {
    /// The persistent id of this object, stable across frames and saves (unlike the [`owner`](ObjectContext::owner)
    /// entity handle).
    ///
    /// # Panics
    ///
    /// Only if `owner` is not a scene object, which cannot happen for hosts the engine builds.
    fn self_id(&self) -> ObjectId { self.world.object_id(self.owner).expect("movement owner") }
    /// This object's **local** position (its `Transform` translation, relative to the parent).
    ///
    /// **Immediate.** For the world position use [`world_position`](MotionContext::world_position).
    ///
    /// # Panics
    ///
    /// Only if `owner` has no `Transform`, which cannot happen for scene objects.
    fn translation(&self) -> Vec3 { self.world.get::<Transform>(self.owner).expect("movement transform").translation }
    /// Sets this object's **local** position.
    ///
    /// **Immediate.** Written in place; later systems of the stage see it. A transform write is a teleport and does
    /// no collision; move character controllers with [`move_character`](MotionContext::move_character). To set the world
    /// position use [`set_world_position`](MotionContext::set_world_position).
    fn set_translation(&mut self, value: Vec3) { self.world.get_mut::<Transform>(self.owner).expect("movement transform").translation = value; }
    /// This object's **local** rotation (relative to the parent).
    ///
    /// **Immediate.** For another object's rotation use [`transform_of`](ObjectContext::transform_of), which is in world
    /// space.
    fn rotation(&self) -> Quat { self.world.get::<Transform>(self.owner).expect("movement transform").rotation }
    /// Sets this object's **local** rotation. Pass a unit quaternion.
    ///
    /// **Immediate.** Written in place; later systems of the stage see it. To set the world rotation use
    /// [`set_world_rotation`](ObjectContext::set_world_rotation).
    fn set_rotation(&mut self, value: Quat) { self.world.get_mut::<Transform>(self.owner).expect("movement transform").rotation = value; }
    /// This object's **local** scale. For a VR player it includes the calibrated tracking-space scale.
    ///
    /// **Immediate.**
    fn scale(&self) -> Vec3 { self.world.get::<Transform>(self.owner).expect("movement transform").scale }
    /// The world matrix of this object's parent, or `None` for a root object (or a malformed parent chain).
    ///
    /// **Immediate.** Computed from the live local transforms, so it includes moves made earlier in the stage.
    fn parent_world(&self) -> Option<Mat4> {
        self.world.get::<Parent>(self.owner).and_then(|parent| parent.0).and_then(|parent| self.world.world_transform(parent))
    }
    /// Whether the physics world moves this object through a character controller.
    ///
    /// True when a physics world is available and the object has a [`Character`] component that uses the default
    /// capsule. **Immediate.** `false` when there is no physics world.
    fn has_character(&self) -> bool {
        self.services.physics.is_some() && self.world.get::<Character>(self.owner).is_some_and(|character| character.use_default_capsule)
    }
    /// Moves this object's character capsule by `delta` (world space, metres), sliding along obstacles.
    ///
    /// `delta` is the whole displacement for this move (velocity times `dt`), not a velocity.
    ///
    /// **Queued.** Physics performs the move at the stage boundary, and [`grounded`](MotionContext::grounded) then reports
    /// it; do not expect the position to have changed on the next line. Check
    /// [`has_character`](MotionContext::has_character) first. Writing the transform instead would skip collision.
    fn move_character(&mut self, delta: Vec3) {
        let id = self.self_id();
        self.commands.push(ComponentCommand::MoveCharacter { id, delta });
    }
    /// Whether the character touched the ground on its last move.
    ///
    /// **Immediate.** Because [`move_character`](MotionContext::move_character) is queued, this reports the previous move.
    /// `false` when there is no physics world.
    fn grounded(&self) -> bool { self.services.physics.is_some_and(|physics| physics.grounded(self.self_id())) }
    /// The character capsule as `(radius, height, center)`, or `None` if the object has no [`Character`].
    ///
    /// **Immediate.**
    fn capsule(&self) -> Option<(f32, f32, [f32; 3])> {
        self.world.get::<Character>(self.owner).map(|character| (character.radius, character.height, character.center))
    }
    /// Resizes the character capsule's `height` and `center` (the radius is unchanged); used for crouching.
    ///
    /// **Immediate** on the component data; physics reshapes the capsule on the next move. No-op if the object has no
    /// [`Character`].
    fn set_capsule(&mut self, height: f32, center: [f32; 3]) {
        if let Some(mut character) = self.world.get_mut::<Character>(self.owner) { character.height = height; character.center = center; }
    }
    /// This object's swim tuning, or the defaults if it has no `Swimmer` component.
    ///
    /// **Immediate.** Returns a copy.
    fn swimmer(&self) -> Swimmer { self.world.get::<Swimmer>(self.owner).map_or_else(Swimmer::default, |swimmer| *swimmer) }
    /// The world height (Y) of the water surface above or below `position`, or `None` if there is no water there.
    ///
    /// Only active `Water` objects count; where several cover the point, the first in authoring order wins. The height
    /// includes wave displacement and ripples at the current water clock (`water_time`), so it changes every frame.
    /// `None` when `position` is outside every water body horizontally or below its floor. **Immediate.**
    fn water_surface(&self, position: Vec3) -> Option<f32> {
        let mut best = None;
        for (entity, (_, water)) in self.world.query::<(&ObjectId, &crate::Water)>().iter() {
            if !self.context.is_active(self.world, entity) { continue; }
            let Some(matrix) = self.world.world_transform(entity) else { continue };
            let Some((height, _)) = water.height_at(matrix, position, self.water_time) else { continue };
            let key = self.context.order_key(self.world, entity);
            if best.as_ref().is_none_or(|(previous, _)| key < *previous) { best = Some((key, height)); }
        }
        best.map(|(_, height)| height)
    }
    /// Casts a ray into the physics world with an explicit [`RayFilter`] (the [`MotionContext`] form that replay also
    /// uses). [`ObjectContext::raycast_filtered`] is the same call.
    ///
    /// **Immediate.** `None` when nothing is in range or there is no physics world.
    fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32, filter: &RayFilter) -> Option<RayHit> {
        self.services.physics?.raycast(origin, dir, max_dist, filter)
    }
    /// Sets a boolean parameter (for example `"Crouched"`) on this object's `Animator`.
    ///
    /// **Immediate.** Written in place; the animation graph reads it when it next updates. No-op if the object has no
    /// `Animator`.
    fn set_animator_bool(&mut self, name: &str, value: bool) {
        if let Some(mut animator) = self.world.get_mut::<crate::Animator>(self.owner) { animator.set_bool(name, value); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Animator, Character, ComponentRegistry, FirstPersonCharacter, InputState, Pawn, PhysicsQuery, Predictable, ThirdPersonCharacter, TypedComponent};
    use crate::ecs::{SceneWorld, Stage};

    struct Ground;
    impl PhysicsQuery for Ground {
        fn raycast(&self, _: Vec3, _: Vec3, _: f32, _: &RayFilter) -> Option<RayHit> { None }
        fn overlap_sphere(&self, _: Vec3, _: f32, _: &RayFilter) -> Vec<ObjectId> { Vec::new() }
        fn velocity(&self, _: ObjectId) -> Option<Vec3> { None }
        fn grounded(&self, _: ObjectId) -> bool { true }
    }

    #[test]
    fn movement_host_writes_real_sibling_data_and_defers_only_engine_requests() {
        let mut scene = SceneWorld::new();
        let parent = scene.spawn("Parent");
        scene.get_mut::<Transform>(parent).unwrap().translation = Vec3::new(10.0, 4.0, 0.0);
        let owner = scene.spawn("Mover");
        scene.set_parent(owner, Some(parent), false).unwrap();
        scene.insert(owner, Character::default()).unwrap();
        scene.insert(owner, Animator::default()).unwrap();
        let mut water = crate::Water::default();
        water.waves.clear();
        let surface = scene.spawn("Water");
        scene.insert(surface, water).unwrap();
        scene.propagate().unwrap();
        let input = InputState::default();
        let mut commands = Vec::new();
        let mut context = SystemContext::new(&input, &mut commands);
        context.services.physics = Some(&Ground);
        scene.with_world(|world| {
            let mut obj = ObjectContext::new(world, owner, &mut context);
            obj.set_translation(Vec3::new(1.0, 2.0, 3.0));
            assert_eq!(obj.world_position(), Vec3::new(11.0, 6.0, 3.0));
            obj.set_world_position(Vec3::new(15.0, 7.0, 2.0));
            assert_eq!(world.get::<Transform>(owner).unwrap().translation, Vec3::new(5.0, 3.0, 2.0));
            assert!(obj.has_character() && obj.grounded());
            obj.set_capsule(1.0, [0.0, 0.5, 0.0]);
            assert_eq!(world.get::<Character>(owner).unwrap().height, 1.0);
            obj.set_animator_bool("Crouched", true);
            assert!(world.get::<Animator>(owner).unwrap().get_bool("Crouched"));
            assert_eq!(obj.water_surface(Vec3::new(0.0, 0.1, 0.0)), Some(0.0));
            obj.move_character(Vec3::X);
        });
        assert!(matches!(commands.as_slice(), [ComponentCommand::MoveCharacter { id, delta }] if *id == scene.object_id(owner).unwrap() && *delta == Vec3::X));
        scene.get_mut::<super::super::Enabled>(surface).unwrap().0 = false;
        scene.propagate().unwrap();
        let mut context = SystemContext::new(&input, &mut commands);
        scene.with_world(|world| assert_eq!(ObjectContext::new(world, owner, &mut context).water_surface(Vec3::ZERO), None));
    }

    fn replay_matches_live<T: TypedComponent + Predictable>() {
        use crate::{ActionValue, ecs::DisabledComponent};
        let registry = ComponentRegistry::with_builtins();
        let mut scene = SceneWorld::new();
        let live = scene.spawn("Live");
        let replay = scene.spawn("Replay");
        for entity in [live, replay] {
            scene.insert(entity, T::default()).unwrap();
            scene.insert(entity, Character::default()).unwrap();
            scene.insert(entity, Animator::default()).unwrap();
        }
        let empty = InputState::default();
        let mut commands = Vec::new();
        registry.run_systems(Stage::Start, &scene, &mut SystemContext::new(&empty, &mut commands));
        let value = scene.remove::<T>(replay).unwrap();
        scene.insert_disabled(replay, value).unwrap();
        for frame in 0..20 {
            let input = InputState::from_actions(vec![
                ("Move".into(), ActionValue { down: true, axis2: glam::Vec2::new(0.3, 0.8), ..Default::default() }),
                ("Crouch".into(), ActionValue { down: frame < 10, ..Default::default() }),
            ], glam::Vec2::new(0.6, -0.2));
            let mut context = SystemContext::new(&input, &mut commands);
            context.dt = 1.0 / 60.0;
            context.unscaled_dt = context.dt;
            registry.run_systems(Stage::Update, &scene, &mut context);
            scene.with_world(|world| {
                let live_value = world.get::<T>(live).unwrap();
                let extra = live_value.frame_extra();
                let mut replay_value = world.get_mut::<DisabledComponent<T>>(replay).unwrap();
                replay_value.value_mut().replay(&mut ObjectContext::new(world, replay, &mut context), &input, &extra, 1.0 / 60.0);
                assert_eq!(live_value.motion(), replay_value.motion(), "{} motion frame{frame}", T::NAME);
                assert_eq!(*world.get::<Transform>(live).unwrap(), *world.get::<Transform>(replay).unwrap(), "{} transform frame{frame}", T::NAME);
                assert_eq!(world.get::<Character>(live).unwrap().height, world.get::<Character>(replay).unwrap().height);
            });
        }
    }

    #[test]
    fn native_controllers_keep_live_and_predicted_movement_steps_in_agreement() {
        replay_matches_live::<Pawn>();
        replay_matches_live::<FirstPersonCharacter>();
        replay_matches_live::<ThirdPersonCharacter>();
        replay_matches_live::<crate::VrCharacter>();
    }
}
