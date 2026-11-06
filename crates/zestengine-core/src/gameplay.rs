//! The gameplay half of the in-game API: object lifecycle (spawn / destroy /
//! enable), access to other components, events, physics, audio and app control.
//! Everything that mutates the world is deferred through [`ComponentCommand`]
//! and applied by the engine between passes; queries answer immediately.

use std::any::{Any, TypeId};

use glam::{Mat4, Quat, Vec2, Vec3};

use crate::{Component, ComponentCommand, ComponentCtx, InstanceSel, MaterialParams, ObjectId, ObjectRef};

/// A physics ray-cast hit.
#[derive(Clone, Copy, Debug)]
pub struct RayHit {
    /// The object whose collider was hit.
    pub object: ObjectId,
    /// Hit position in world space (meters).
    pub point: Vec3,
    /// Surface normal at the hit point, world space.
    pub normal: Vec3,
    /// Distance along the ray from its origin to the hit, in meters.
    pub distance: f32,
}

/// What a ray cast / overlap test may hit.
#[derive(Clone, Copy, Debug)]
pub struct RayFilter {
    /// Layers to test (bit per layer).
    pub layer_mask: u32,
    /// Also test trigger colliders.
    pub include_triggers: bool,
    /// Ignore this object's colliders (typically the caster itself).
    pub exclude: Option<ObjectId>,
}

impl Default for RayFilter {
    fn default() -> Self {
        Self {
            layer_mask: u32::MAX,
            include_triggers: false,
            exclude: None,
        }
    }
}

/// The game view this frame: size, pointer and camera, plus UI interaction.
#[derive(Clone, Default)]
pub struct ScreenInfo {
    /// Game viewport size in pixels.
    pub size: Vec2,
    /// Pointer position in pixels from the viewport's top-left (None when outside).
    pub pointer: Option<Vec2>,
    /// Camera view matrix (world to camera space).
    pub view: Mat4,
    /// Camera projection matrix (camera to clip space).
    pub proj: Mat4,
    /// The pointer is over an interactive UI element (skip world clicks).
    pub pointer_over_ui: bool,
    /// UI buttons clicked since the last frame.
    pub ui_clicks: Vec<ObjectId>,
    /// Text fields submitted (Enter) since the last frame.
    pub ui_submits: Vec<ObjectId>,
    /// A UI text field is being edited (keyboard goes to it, not gameplay).
    pub ui_text_editing: bool,
    /// UI cancel / back (Escape, gamepad East) was pressed.
    pub ui_cancel: bool,
    /// Headset and controllers (inactive without VR).
    pub vr: crate::VrFrame,
    /// Live input bindings, device in use and rebind capture state.
    pub controls: crate::ControlsInfo,
    /// The player's measured body and how their avatar is fitted to it.
    pub body: crate::BodyInfo,
}

/// Immediate physics queries, implemented by the engine's physics world.
pub trait PhysicsQuery {
    /// Nearest collider along a world-space ray from `origin` toward `dir`,
    /// within `max_dist` meters.
    fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32, filter: &RayFilter) -> Option<RayHit>;
    /// Objects whose colliders overlap a sphere (world-space `center`,
    /// `radius` in meters).
    fn overlap_sphere(&self, center: Vec3, radius: f32, filter: &RayFilter) -> Vec<ObjectId>;
    /// Linear velocity of a rigid body.
    fn velocity(&self, id: ObjectId) -> Option<Vec3>;
    /// Whether a character controller touched the ground on its last move.
    fn grounded(&self, id: ObjectId) -> bool;

    // ---- 2D physics (defaults: no 2D world) ----

    /// Nearest 2D collider along a ray (XY plane).
    fn raycast2d(&self, _origin: Vec2, _dir: Vec2, _max_dist: f32, _filter: &RayFilter) -> Option<crate::RayHit2D> {
        None
    }
    /// Every 2D collider along a segment, nearest first.
    fn raycast_all2d(&self, _origin: Vec2, _dir: Vec2, _max_dist: f32, _filter: &RayFilter) -> Vec<crate::RayHit2D> {
        Vec::new()
    }
    /// 2D colliders containing a point (XY plane).
    fn overlap_point2d(&self, _point: Vec2, _filter: &RayFilter) -> Vec<ObjectId> {
        Vec::new()
    }
    /// 2D colliders overlapping a circle (XY plane, radius in meters).
    fn overlap_circle2d(&self, _center: Vec2, _radius: f32, _filter: &RayFilter) -> Vec<ObjectId> {
        Vec::new()
    }
    /// Objects overlapping a box of `size` at `center`, rotated `angle` radians.
    fn overlap_box2d(&self, _center: Vec2, _size: Vec2, _angle: f32, _filter: &RayFilter) -> Vec<ObjectId> {
        Vec::new()
    }
    /// Sweep a circle along `dir`; the first thing it touches.
    fn circle_cast2d(&self, _center: Vec2, _radius: f32, _dir: Vec2, _max_dist: f32, _filter: &RayFilter) -> Option<crate::RayHit2D> {
        None
    }
    /// Sweep a box of `size` at `center` (rotated `angle` radians) along
    /// `dir`; the first 2D collider it touches.
    fn box_cast2d(&self, _center: Vec2, _size: Vec2, _angle: f32, _dir: Vec2, _max_dist: f32, _filter: &RayFilter) -> Option<crate::RayHit2D> {
        None
    }
    /// The contact points of body `id` with everything it touches.
    fn contacts2d(&self, _id: ObjectId) -> Vec<crate::ContactPoint2D> {
        Vec::new()
    }
}

/// Navmesh queries, implemented by the engine's navigation system.
pub trait NavQuery {
    /// Walkable path (waypoints, start and end included) or None.
    fn find_path(&self, from: Vec3, to: Vec3) -> Option<Vec<Vec3>>;
    /// Closest walkable point (within a few meters).
    fn nearest_walkable(&self, p: Vec3) -> Option<Vec3>;
}

/// Read access to the scene's materials, implemented by the engine's scene.
/// Lets a plugin read the live editable params of any object's material slot
/// (slot 0 = the main material) and, through
/// [`ComponentCtx::modify_material`], edit them.
pub trait MaterialQuery {
    /// The material params of an object's render slot (None when the object
    /// or slot has no material).
    fn material_params(&self, object: usize, slot: usize) -> Option<MaterialParams>;
}

/// Read access to components on other objects, implemented by the engine.
pub trait ComponentLookup {
    /// Read a component on an object by its `TypeId` (None when the object
    /// has no such component).
    fn component(&self, object: usize, type_id: TypeId) -> Option<&dyn Component>;

    /// A component by its type name (`Component::type_name`), for event
    /// bindings that name components as text. `hint` is the slot to try
    /// first (a binding's cached result); returns the slot it was found at.
    fn component_named(&self, _object: usize, _name: &str, _hint: usize, _sel: InstanceSel) -> Option<(usize, &dyn Component)> {
        None
    }
}

/// A type-erased value carried by a command (events).
pub struct AnyBox(pub Box<dyn Any>);

impl std::fmt::Debug for AnyBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AnyBox")
    }
}

type DeferredFn = Box<dyn FnOnce(&mut dyn Any)>;

/// A deferred edit of another component (see [`ComponentCtx::modify`]).
pub struct Deferred(pub DeferredFn);

impl std::fmt::Debug for Deferred {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Deferred")
    }
}

/// A deferred edit of a material slot's params (see
/// [`ComponentCtx::modify_material`]).
pub struct MaterialEdit(pub Box<dyn FnOnce(&mut MaterialParams) + 'static>);

impl std::fmt::Debug for MaterialEdit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MaterialEdit")
    }
}

impl ComponentCtx<'_> {
    // ---- screen / pointer ----

    /// The game viewport size in pixels.
    pub fn screen_size(&self) -> Vec2 {
        self.screen.size
    }

    /// Pointer position in pixels from the game view's top-left.
    pub fn pointer_position(&self) -> Option<Vec2> {
        self.screen.pointer
    }

    /// World-space ray `(origin, direction)` through a pixel of the game view
    /// (click-to-select, click-to-move). Pair with `raycast`.
    pub fn screen_ray(&self, pixel: Vec2) -> Option<(Vec3, Vec3)> {
        let size = self.screen.size;
        if size.x < 1.0 || size.y < 1.0 {
            return None;
        }
        let ndc = Vec2::new(2.0 * pixel.x / size.x - 1.0, 1.0 - 2.0 * pixel.y / size.y);
        let inv = (self.screen.proj * self.screen.view).inverse();
        let near = inv.project_point3(ndc.extend(0.0));
        let far = inv.project_point3(ndc.extend(1.0));
        let dir = (far - near).try_normalize()?;
        Some((near, dir))
    }

    /// Ray under the pointer.
    pub fn pointer_ray(&self) -> Option<(Vec3, Vec3)> {
        self.screen_ray(self.screen.pointer?)
    }

    /// Pixel position of a world point (None when behind the camera).
    pub fn world_to_screen(&self, world: Vec3) -> Option<Vec2> {
        let clip = self.screen.proj * self.screen.view * world.extend(1.0);
        if clip.w <= 1e-6 {
            return None;
        }
        let ndc = clip.truncate() / clip.w;
        Some(Vec2::new(
            (ndc.x + 1.0) * 0.5 * self.screen.size.x,
            (1.0 - ndc.y) * 0.5 * self.screen.size.y,
        ))
    }

    /// Whether the pointer is over interactive UI (ignore world clicks then).
    pub fn pointer_over_ui(&self) -> bool {
        self.screen.pointer_over_ui
    }

    /// Whether a UI button was clicked since the last frame.
    pub fn ui_clicked(&self, button: impl Into<ObjectRef>) -> bool {
        button
            .into()
            .id()
            .is_some_and(|id| self.screen.ui_clicks.contains(&id))
    }

    /// Whether a text field was submitted (Enter) since the last frame.
    pub fn ui_submitted(&self, field: impl Into<ObjectRef>) -> bool {
        field
            .into()
            .id()
            .is_some_and(|id| self.screen.ui_submits.contains(&id))
    }

    /// A UI text field has keyboard focus; ignore keyboard gameplay input.
    pub fn ui_text_editing(&self) -> bool {
        self.screen.ui_text_editing
    }

    /// UI back / cancel pressed (Escape or gamepad East), e.g. to close a menu.
    pub fn ui_cancel_pressed(&self) -> bool {
        self.screen.ui_cancel
    }

    // ---- scene streaming ----

    /// Begin loading a scene in the background (during a transition). A later
    /// `load_scene` / `load_scene_additive` of the same path is then instant.
    pub fn preload_scene(&mut self, path: impl Into<String>) {
        self.commands.push(ComponentCommand::PreloadScene(path.into()));
    }

    /// Stream a scene in next to the current one (rooms, level chunks).
    pub fn load_scene_additive(&mut self, path: impl Into<String>) {
        self.commands.push(ComponentCommand::LoadSceneAdditive(path.into()));
    }

    /// Stream an additively loaded scene back out.
    pub fn unload_scene(&mut self, path: impl Into<String>) {
        self.commands.push(ComponentCommand::UnloadScene(path.into()));
    }

    /// Carry an object (and children) into the next scene on `load_scene`.
    pub fn keep_on_scene_change(&mut self, target: impl Into<ObjectRef>, keep: bool) {
        if let Some(id) = target.into().id() {
            self.commands.push(ComponentCommand::KeepOnSceneChange(id, keep));
        }
    }

    // ---- object lifecycle ----

    /// Instantiate a `.prefab` (project-relative path, e.g. `"prefabs/enemy.prefab"`)
    /// at a world position/rotation. Returns the new root's id right away; the
    /// object exists from the next pass and runs `start` on its first frame.
    pub fn spawn_prefab(&mut self, path: impl Into<String>, translation: Vec3, rotation: Quat) -> ObjectId {
        let root_id = ObjectId::new();
        self.commands.push(ComponentCommand::SpawnPrefab {
            path: path.into(),
            root_id,
            translation,
            rotation,
        });
        root_id
    }

    /// Clone a scene object (with its children and components) as a new root at
    /// a world position/rotation. The clone copies the original's enabled state,
    /// so a disabled template needs `set_active(id, true)` afterwards.
    pub fn instantiate(&mut self, original: impl Into<ObjectRef>, translation: Vec3, rotation: Quat) -> Option<ObjectId> {
        let original = original.into().id()?;
        let new_id = ObjectId::new();
        self.commands.push(ComponentCommand::Instantiate {
            original,
            new_id,
            translation,
            rotation,
        });
        Some(new_id)
    }

    /// Remove an object and its children (end of the current pass).
    pub fn destroy(&mut self, target: impl Into<ObjectRef>) {
        if let Some(id) = target.into().id() {
            self.commands.push(ComponentCommand::Destroy(id));
        }
    }

    /// Destroy the owning object (and its children) at the end of the
    /// current pass.
    pub fn destroy_self(&mut self) {
        let id = self.self_id();
        self.commands.push(ComponentCommand::Destroy(id));
    }

    /// Enable or disable an object. Disabled objects (and their children) stop
    /// updating, rendering and colliding.
    pub fn set_active(&mut self, target: impl Into<ObjectRef>, active: bool) {
        if let Some(id) = target.into().id() {
            self.commands.push(ComponentCommand::SetActive(id, active));
        }
    }

    /// Whether an object is enabled (false for destroyed or unknown objects).
    /// Disabled objects and their children stop updating, rendering and
    /// colliding.
    pub fn is_active(&self, target: impl Into<ObjectRef>) -> bool {
        self.resolve(target.into())
            .and_then(|i| self.active.get(i).copied())
            .unwrap_or(false)
    }

    // ---- other components ----

    /// Read a component of type `T` on another object (or this one).
    pub fn get<T: Component>(&self, target: impl Into<ObjectRef>) -> Option<&T> {
        let index = self.resolve(target.into())?;
        self.lookup
            .component(index, TypeId::of::<T>())?
            .as_any()
            .downcast_ref::<T>()
    }

    /// Read a sibling component on the owning object.
    pub fn get_self<T: Component>(&self) -> Option<&T> {
        self.lookup
            .component(self.self_index, TypeId::of::<T>())?
            .as_any()
            .downcast_ref::<T>()
    }

    /// Edit a component of type `T` on another object, e.g.
    /// `ctx.modify::<Health>(hit.object, |h| h.hp -= 10.0)`. Applied after the
    /// current pass; a no-op if the object or component is gone by then.
    pub fn modify<T: Component>(&mut self, target: impl Into<ObjectRef>, f: impl FnOnce(&mut T) + 'static) {
        let Some(id) = target.into().id() else {
            return;
        };
        self.commands.push(ComponentCommand::Modify {
            id,
            type_id: TypeId::of::<T>(),
            f: Deferred(Box::new(move |any: &mut dyn Any| {
                if let Some(t) = any.downcast_mut::<T>() {
                    f(t);
                }
            })),
        });
    }

    // ---- particles ----

    /// (Re)start a Particle Emitter.
    pub fn play_particles(&mut self, target: impl Into<ObjectRef>) {
        self.modify::<crate::ParticleEmitter>(target, |p| p.control.play = true);
    }

    /// Stop emitting (live particles finish their lifetime).
    pub fn stop_particles(&mut self, target: impl Into<ObjectRef>) {
        self.modify::<crate::ParticleEmitter>(target, |p| p.control.stop = true);
    }

    /// Emit `count` particles immediately (explosions, hits).
    pub fn emit_particles(&mut self, target: impl Into<ObjectRef>, count: u32) {
        self.modify::<crate::ParticleEmitter>(target, move |p| p.control.emit += count);
    }

    // ---- navigation ----

    /// Send a `Nav Agent` to a world point.
    pub fn set_destination(&mut self, agent: impl Into<ObjectRef>, point: Vec3) {
        self.modify::<crate::NavAgent>(agent, move |a| a.set_destination(point));
    }

    /// Stop a `Nav Agent` where it stands.
    pub fn stop_agent(&mut self, agent: impl Into<ObjectRef>) {
        self.modify::<crate::NavAgent>(agent, |a| a.stop());
    }

    /// Re-bake navmeshes (doors opened, obstacles spawned). Takes effect after
    /// the next physics step.
    pub fn rebake_navmesh(&mut self) {
        self.commands.push(ComponentCommand::RebakeNavMesh);
    }

    /// Path over the navmesh (None when unreachable or nothing is baked).
    pub fn find_path(&self, from: Vec3, to: Vec3) -> Option<Vec<Vec3>> {
        self.nav?.find_path(from, to)
    }

    /// The closest walkable point to `p` (None when nothing is baked).
    pub fn nearest_walkable(&self, p: Vec3) -> Option<Vec3> {
        self.nav?.nearest_walkable(p)
    }

    // ---- animation ----

    /// Cross-fade an object's [`crate::Animator`] to `clip` over `fade` seconds.
    pub fn play_animation(&mut self, target: impl Into<ObjectRef>, clip: &str, fade: f32) {
        let clip = clip.to_owned();
        self.modify::<crate::Animator>(target, move |a| a.play(&clip, fade));
    }

    /// Set a blend-tree parameter on an object's Animator.
    pub fn set_animator_float(&mut self, target: impl Into<ObjectRef>, name: &str, value: f32) {
        let name = name.to_owned();
        self.modify::<crate::Animator>(target, move |a| a.set_float(&name, value));
    }

    /// Stop playback of an object's [`crate::Animator`].
    pub fn stop_animation(&mut self, target: impl Into<ObjectRef>) {
        self.modify::<crate::Animator>(target, |a| a.stop());
    }

    /// Set an integer blend-tree parameter on an object's Animator (see
    /// [`Self::set_animator_float`]).
    pub fn set_animator_int(&mut self, target: impl Into<ObjectRef>, name: &str, value: i32) {
        let name = name.to_owned();
        self.modify::<crate::Animator>(target, move |a| a.set_int(&name, value));
    }

    /// Set a boolean blend-tree parameter on an object's Animator (see
    /// [`Self::set_animator_float`]).
    pub fn set_animator_bool(&mut self, target: impl Into<ObjectRef>, name: &str, value: bool) {
        let name = name.to_owned();
        self.modify::<crate::Animator>(target, move |a| a.set_bool(&name, value));
    }

    /// Fire an Animator Controller trigger (consumed by the transition it starts).
    pub fn set_animator_trigger(&mut self, target: impl Into<ObjectRef>, name: &str) {
        let name = name.to_owned();
        self.modify::<crate::Animator>(target, move |a| a.set_trigger(&name));
    }

    /// Clear an Animator Controller trigger so it can fire again.
    pub fn reset_animator_trigger(&mut self, target: impl Into<ObjectRef>, name: &str) {
        let name = name.to_owned();
        self.modify::<crate::Animator>(target, move |a| a.reset_trigger(&name));
    }

    /// Force an Animator Controller state by name, cross-fading over `fade` seconds.
    pub fn play_animator_state(&mut self, target: impl Into<ObjectRef>, state: &str, fade: f32) {
        let state = state.to_owned();
        self.modify::<crate::Animator>(target, move |a| a.play_state(&state, fade));
    }

    /// A parameter of an object's Animator (0 when it has none).
    pub fn animator_float(&self, target: impl Into<ObjectRef>, name: &str) -> f32 {
        self.get::<crate::Animator>(target).map(|a| a.get_float(name)).unwrap_or(0.0)
    }

    /// A boolean parameter of an object's Animator (false when it has none).
    pub fn animator_bool(&self, target: impl Into<ObjectRef>, name: &str) -> bool {
        self.get::<crate::Animator>(target).is_some_and(|a| a.get_bool(name))
    }

    /// Current controller state of an object's Animator on `layer`.
    pub fn animator_state(&self, target: impl Into<ObjectRef>, layer: usize) -> Option<String> {
        self.get::<crate::Animator>(target).and_then(|a| a.current_state(layer).map(str::to_owned))
    }

    // ---- events ----

    /// Broadcast an event of any type; every component can read it next frame
    /// with [`ComponentCtx::events`].
    pub fn emit<E: Any>(&mut self, event: E) {
        self.commands.push(ComponentCommand::Emit(AnyBox(Box::new(event))));
    }

    /// Events of type `E` emitted last frame.
    pub fn events<E: Any>(&self) -> impl Iterator<Item = &E> {
        self.events.iter().filter_map(|e| e.0.downcast_ref::<E>())
    }

    // ---- physics ----

    /// Nearest collider along a ray, ignoring triggers and the owning object.
    pub fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32) -> Option<RayHit> {
        let filter = RayFilter {
            exclude: Some(self.self_id()),
            ..Default::default()
        };
        self.raycast_filtered(origin, dir, max_dist, &filter)
    }

    /// Nearest collider along a world-space ray with an explicit [`RayFilter`]
    /// (layers, triggers, exclusions). `max_dist` is in meters.
    pub fn raycast_filtered(&self, origin: Vec3, dir: Vec3, max_dist: f32, filter: &RayFilter) -> Option<RayHit> {
        self.physics?.raycast(origin, dir, max_dist, filter)
    }

    /// Objects whose colliders intersect a sphere (triggers excluded).
    pub fn overlap_sphere(&self, center: Vec3, radius: f32) -> Vec<ObjectId> {
        self.overlap_sphere_filtered(center, radius, &RayFilter::default())
    }

    /// Objects whose colliders overlap a sphere, with an explicit
    /// [`RayFilter`].
    pub fn overlap_sphere_filtered(&self, center: Vec3, radius: f32, filter: &RayFilter) -> Vec<ObjectId> {
        self.physics
            .map(|p| p.overlap_sphere(center, radius, filter))
            .unwrap_or_default()
    }

    // ---- 2D physics (XY plane) ----

    /// Nearest 2D collider along a ray, ignoring triggers and the owning object.
    pub fn raycast2d(&self, origin: Vec2, dir: Vec2, max_dist: f32) -> Option<crate::RayHit2D> {
        let filter = RayFilter { exclude: Some(self.self_id()), ..Default::default() };
        self.physics?.raycast2d(origin, dir, max_dist, &filter)
    }

    /// Nearest 2D collider along a ray in the XY plane, with an explicit
    /// [`RayFilter`].
    pub fn raycast2d_filtered(&self, origin: Vec2, dir: Vec2, max_dist: f32, filter: &RayFilter) -> Option<crate::RayHit2D> {
        self.physics?.raycast2d(origin, dir, max_dist, filter)
    }

    /// Every hit along the ray, nearest first.
    pub fn raycast_all2d(&self, origin: Vec2, dir: Vec2, max_dist: f32, filter: &RayFilter) -> Vec<crate::RayHit2D> {
        self.physics.map(|p| p.raycast_all2d(origin, dir, max_dist, filter)).unwrap_or_default()
    }

    /// 2D colliders containing a point (XY plane), with an explicit
    /// [`RayFilter`].
    pub fn overlap_point2d(&self, point: Vec2, filter: &RayFilter) -> Vec<ObjectId> {
        self.physics.map(|p| p.overlap_point2d(point, filter)).unwrap_or_default()
    }

    /// 2D colliders overlapping a circle (XY plane), with an explicit
    /// [`RayFilter`].
    pub fn overlap_circle2d(&self, center: Vec2, radius: f32, filter: &RayFilter) -> Vec<ObjectId> {
        self.physics.map(|p| p.overlap_circle2d(center, radius, filter)).unwrap_or_default()
    }

    /// 2D colliders overlapping a box of `size` at `center`, rotated `angle`
    /// radians, with an explicit [`RayFilter`].
    pub fn overlap_box2d(&self, center: Vec2, size: Vec2, angle: f32, filter: &RayFilter) -> Vec<ObjectId> {
        self.physics.map(|p| p.overlap_box2d(center, size, angle, filter)).unwrap_or_default()
    }

    /// Sweep a circle of `radius` along `dir`; the first 2D collider it
    /// touches.
    pub fn circle_cast2d(&self, center: Vec2, radius: f32, dir: Vec2, max_dist: f32, filter: &RayFilter) -> Option<crate::RayHit2D> {
        self.physics?.circle_cast2d(center, radius, dir, max_dist, filter)
    }

    /// Sweep a box of `size` at `center` (rotated `angle` radians) along
    /// `dir`; the first 2D collider it touches.
    pub fn box_cast2d(&self, center: Vec2, size: Vec2, angle: f32, dir: Vec2, max_dist: f32, filter: &RayFilter) -> Option<crate::RayHit2D> {
        self.physics?.box_cast2d(center, size, angle, dir, max_dist, filter)
    }

    /// Contact points of the owning object's 2D body (allocates; call on demand).
    pub fn contacts2d(&self) -> Vec<crate::ContactPoint2D> {
        self.physics.map(|p| p.contacts2d(self.self_id())).unwrap_or_default()
    }

    /// Linear velocity of a rigid body (zero if it has none).
    pub fn velocity(&self, target: impl Into<ObjectRef>) -> Vec3 {
        target
            .into()
            .id()
            .and_then(|id| self.physics?.velocity(id))
            .unwrap_or(Vec3::ZERO)
    }

    /// Set a rigid body's linear velocity (meters per second, world space).
    pub fn set_velocity(&mut self, target: impl Into<ObjectRef>, linear: Vec3) {
        if let Some(id) = target.into().id() {
            self.commands.push(ComponentCommand::SetVelocity { id, linear });
        }
    }

    /// Continuous force (Newtons), applied over the next physics step.
    pub fn add_force(&mut self, target: impl Into<ObjectRef>, force: Vec3) {
        if let Some(id) = target.into().id() {
            self.commands.push(ComponentCommand::AddForce { id, force });
        }
    }

    /// Instant velocity change scaled by mass (jumps, hits, explosions).
    pub fn add_impulse(&mut self, target: impl Into<ObjectRef>, impulse: Vec3) {
        if let Some(id) = target.into().id() {
            self.commands.push(ComponentCommand::AddImpulse { id, impulse });
        }
    }

    /// Rotational force (Newton-meters) on a rigid body, applied over the
    /// next physics step.
    pub fn add_torque(&mut self, target: impl Into<ObjectRef>, torque: Vec3) {
        if let Some(id) = target.into().id() {
            self.commands.push(ComponentCommand::AddTorque { id, torque });
        }
    }

    /// Drive the motor of the `Joint` on an object (deg/s for hinges, m/s for
    /// sliders). Speed 0 with a max force acts as a brake.
    pub fn set_joint_motor(&mut self, target: impl Into<ObjectRef>, speed: f32, max_force: f32) {
        if let Some(id) = target.into().id() {
            self.commands.push(ComponentCommand::SetJointMotor { id, speed, max_force });
        }
    }

    /// Move the owning object's [`crate::CharacterController`] by a world-space
    /// delta (include gravity yourself). Collisions are resolved immediately
    /// after the current pass.
    pub fn move_character(&mut self, delta: Vec3) {
        let id = self.self_id();
        self.commands.push(ComponentCommand::MoveCharacter { id, delta });
    }

    /// Whether the owning character controller was on the ground after its last move.
    pub fn is_grounded(&self) -> bool {
        self.physics.is_some_and(|p| p.grounded(self.self_id()))
    }

    // ---- audio ----

    /// Play a clip once, non-spatially (UI, music stingers).
    pub fn play_sound(&mut self, clip: impl Into<String>, volume: f32) {
        self.commands.push(ComponentCommand::PlaySound {
            clip: clip.into(),
            position: None,
            volume,
        });
    }

    /// Play a clip once at a world position (gunshots, impacts).
    pub fn play_sound_at(&mut self, clip: impl Into<String>, position: Vec3, volume: f32) {
        self.commands.push(ComponentCommand::PlaySound {
            clip: clip.into(),
            position: Some(position),
            volume,
        });
    }

    /// (Re)start the `AudioSource` on an object.
    pub fn play_audio(&mut self, target: impl Into<ObjectRef>) {
        if let Some(id) = target.into().id() {
            self.commands.push(ComponentCommand::PlayAudio(id));
        }
    }

    /// Stop the `AudioSource` on an object.
    pub fn stop_audio(&mut self, target: impl Into<ObjectRef>) {
        if let Some(id) = target.into().id() {
            self.commands.push(ComponentCommand::StopAudio(id));
        }
    }

    // ---- save data ----

    /// The persistent save data (read). Typed getters: `int`, `float`, `bool`,
    /// `str`, `vec3`.
    pub fn save_data(&self) -> &crate::SaveGame {
        self.save
    }

    /// Set a save value (applied after this pass; written to disk by `write_save`).
    pub fn save_value(&mut self, key: impl Into<String>, value: impl Into<crate::SaveValue>) {
        self.commands.push(ComponentCommand::SaveSet { key: key.into(), value: value.into() });
    }

    /// Remove the value stored under `key` (applied after this pass).
    pub fn remove_save_value(&mut self, key: impl Into<String>) {
        self.commands.push(ComponentCommand::SaveRemove(key.into()));
    }

    /// Write the save data to slot `slot` in the user's save folder.
    pub fn write_save(&mut self, slot: impl Into<String>) {
        self.commands.push(ComponentCommand::SaveWrite(slot.into()));
    }

    /// Load slot `slot` (replaces the current save data; readable next pass).
    pub fn load_save(&mut self, slot: impl Into<String>) {
        self.commands.push(ComponentCommand::SaveLoad(slot.into()));
    }

    // ---- localization ----

    /// Translated string for `key` in the current language (the key itself
    /// when missing).
    pub fn tr<'k>(&'k self, key: &'k str) -> &'k str {
        self.strings.tr(key)
    }

    /// The current language code (e.g. `"en"`), used by [`ComponentCtx::tr`].
    pub fn language(&self) -> &str {
        self.strings.current_language()
    }

    /// Switch the language; [`ComponentCtx::tr`] returns the new strings from
    /// the next pass.
    pub fn set_language(&mut self, lang: impl Into<String>) {
        self.commands.push(ComponentCommand::SetLanguage(lang.into()));
    }

    // ---- audio mixer ----

    /// Set a mixer bus volume (0..1), e.g. from a settings menu slider.
    pub fn set_bus_volume(&mut self, bus: impl Into<String>, volume: f32) {
        self.commands.push(ComponentCommand::SetBusVolume { bus: bus.into(), volume });
    }

    /// Mute or unmute an audio mixer bus by name.
    pub fn set_bus_muted(&mut self, bus: impl Into<String>, muted: bool) {
        self.commands.push(ComponentCommand::SetBusMuted { bus: bus.into(), muted });
    }

    // ---- app ----

    /// Lock and hide the cursor for mouse-look, or release it.
    pub fn set_cursor_locked(&mut self, locked: bool) {
        self.commands.push(ComponentCommand::SetCursorLocked(locked));
    }

    /// 0 pauses gameplay (`dt` becomes 0, physics stops), 1 is normal speed.
    pub fn set_time_scale(&mut self, scale: f32) {
        self.commands.push(ComponentCommand::SetTimeScale(scale.max(0.0)));
    }

    /// Quit the application after this pass.
    pub fn quit(&mut self) {
        self.commands.push(ComponentCommand::Quit);
    }

    // ---- water ----

    /// The water surface above or below `p` (world): its height and normal,
    /// or None when `p` isn't over any Water volume. Matches what floats and
    /// what's rendered at the current play time.
    pub fn water_surface(&self, p: glam::Vec3) -> Option<(f32, glam::Vec3)> {
        let ty = std::any::TypeId::of::<crate::Water>();
        (0..self.object_ids.len()).find_map(|i| {
            let water = self.lookup.component(i, ty)?.as_any().downcast_ref::<crate::Water>()?;
            water.height_at(self.object_matrix(i)?, p, self.water_time)
        })
    }

    /// Water in a [`crate::WaterContainer`]: litres and share of its
    /// capacity (as of the last physics step).
    pub fn container_water(&self, target: impl Into<ObjectRef>) -> Option<(f32, f32)> {
        let c = self.get::<crate::WaterContainer>(target)?;
        let cap = c.runtime.capacity.max(1e-6);
        Some((c.runtime.litres, (c.runtime.litres / cap).clamp(0.0, 1.0)))
    }

    /// Add `litres` to a container (what doesn't fit pours over the rim).
    pub fn fill_container(&mut self, target: impl Into<ObjectRef>, litres: f32) {
        let litres = litres.max(0.0);
        self.modify::<crate::WaterContainer>(target, move |c| c.runtime.add += litres);
    }

    /// Pour `litres` out of a container over its lowest rim point, as a
    /// stream (whatever it holds, at most).
    pub fn pour_container(&mut self, target: impl Into<ObjectRef>, litres: f32) {
        let litres = litres.max(0.0);
        self.modify::<crate::WaterContainer>(target, move |c| c.runtime.pour += litres);
    }

    // ---- input bindings ----

    /// The live bindings: project defaults with the player's overrides on top.
    pub fn input_bindings(&self) -> &crate::Bindings {
        &self.screen.controls.bindings
    }

    /// The scheme resolving input this frame.
    pub fn control_scheme(&self) -> Option<&crate::ControlScheme> {
        self.input_bindings().active_scheme()
    }

    /// The device the player used last (VR while a headset runs).
    pub fn input_device(&self) -> crate::InputDevice {
        self.screen.controls.device
    }

    /// Whether the active scheme has an action called `name`.
    pub fn has_action(&self, name: &str) -> bool {
        self.control_scheme().is_some_and(|s| s.action(name).is_some())
    }

    /// Inputs bound to more than one action in `scheme` ("" = active).
    pub fn binding_conflicts(&self, scheme: &str) -> Vec<crate::BindingConflict> {
        self.input_bindings().scheme(scheme).map(|s| s.conflicts()).unwrap_or_default()
    }

    /// Bind `source` at `index` of `slot` ("" scheme = active; an index past
    /// the end adds a binding). Other actions keep the input too; use
    /// [`Self::rebind_with`] to swap or steal it. Saved to the player's
    /// overrides, not the project.
    pub fn rebind(&mut self, scheme: &str, action: &str, slot: crate::BindingSlot, index: usize, source: crate::InputSource) {
        self.rebind_with(scheme, action, slot, index, source, crate::ConflictPolicy::Allow);
    }

    /// Rebind `source` at `index` of `slot` like [`Self::rebind`], with an
    /// explicit [`crate::ConflictPolicy`] deciding what happens to the other
    /// actions already using `source`.
    pub fn rebind_with(
        &mut self,
        scheme: &str,
        action: &str,
        slot: crate::BindingSlot,
        index: usize,
        source: crate::InputSource,
        policy: crate::ConflictPolicy,
    ) {
        self.input_command(crate::InputCommand::Rebind {
            scheme: scheme.into(),
            action: action.into(),
            slot,
            index,
            source,
            policy,
        });
    }

    /// Remove the binding at `index` of `slot` from `action` in `scheme`
    /// ("" = active scheme).
    pub fn unbind(&mut self, scheme: &str, action: &str, slot: crate::BindingSlot, index: usize) {
        self.input_command(crate::InputCommand::Unbind { scheme: scheme.into(), action: action.into(), slot, index });
    }

    /// Deadzone / sensitivity / inversion of an action (e.g. "Invert look").
    pub fn tune_action(&mut self, scheme: &str, action: &str, tuning: crate::ActionTuning) {
        self.input_command(crate::InputCommand::Tune { scheme: scheme.into(), action: action.into(), tuning });
    }

    /// Listen for the next input the player presses (inputs already held are
    /// ignored until released, Escape cancels). Read the result with
    /// [`Self::captured_input`]; gameplay input is held off meanwhile.
    pub fn capture_next_input(&mut self, kind: crate::CaptureKind, device: Option<crate::InputDevice>) {
        let owner = self.self_id();
        self.input_command(crate::InputCommand::Capture { owner, request: crate::CaptureRequest { kind, device } });
    }

    /// Cancel this object's pending input capture (see
    /// [`Self::capture_next_input`]).
    pub fn cancel_input_capture(&mut self) {
        let owner = self.self_id();
        self.input_command(crate::InputCommand::CancelCapture(owner));
    }

    /// This object's capture is still waiting for input.
    pub fn is_capturing_input(&self) -> bool {
        self.screen.controls.capturing == Some(self.self_id())
    }

    /// The result of this object's capture, on the frame it lands.
    pub fn captured_input(&self) -> Option<crate::CaptureResult> {
        self.screen.controls.captured.filter(|(o, _)| *o == self.self_id()).map(|(_, r)| r)
    }

    /// Drop every player override (back to the project's bindings).
    pub fn reset_bindings(&mut self) {
        self.input_command(crate::InputCommand::ResetAll);
    }

    /// Drop every player override of `scheme` (back to the project's
    /// bindings).
    pub fn reset_scheme_bindings(&mut self, scheme: &str) {
        self.input_command(crate::InputCommand::ResetScheme(scheme.into()));
    }

    /// Drop the player's overrides of `action` in `scheme` (back to the
    /// project's bindings).
    pub fn reset_action_binding(&mut self, scheme: &str, action: &str) {
        self.input_command(crate::InputCommand::ResetAction { scheme: scheme.into(), action: action.into() });
    }

    /// Use this scheme from now on (turns automatic switching off).
    pub fn set_control_scheme(&mut self, name: &str) {
        self.input_command(crate::InputCommand::SetScheme(name.into()));
    }

    /// Follow the device in use (VR / gamepad / keyboard + mouse).
    pub fn set_auto_control_scheme(&mut self, on: bool) {
        self.input_command(crate::InputCommand::SetAutoSwitch(on));
    }

    /// Queue a raw [`crate::InputCommand`]; the typed helpers above call this.
    pub fn input_command(&mut self, cmd: crate::InputCommand) {
        self.commands.push(ComponentCommand::Input(cmd));
    }

    // ---- full-body tracking ----

    /// Calibrate full-body tracking: call while the player stands in a T-pose
    /// matching their avatar. Lines the avatar up under the headset and stores
    /// each tracker's offset to its bone.
    pub fn calibrate_trackers(&mut self) {
        self.commands.push(ComponentCommand::CalibrateTrackers);
    }

    /// Forget the stored tracker offsets (back to the default fit).
    pub fn clear_tracker_calibration(&mut self) {
        self.commands.push(ComponentCommand::ClearTrackerCalibration);
    }

    /// The player's standing height (m, to the top of the head): measured in
    /// the headset, entered by the player, or a default until then.
    pub fn player_height(&self) -> f32 {
        self.screen.body.player_height
    }

    /// The local avatar's scale over its authored size (1 unless it is
    /// fitted to the player's height).
    pub fn avatar_scale(&self) -> f32 {
        self.screen.body.avatar_scale
    }

    /// Measured body, fit and calibration state.
    pub fn body(&self) -> &crate::BodyInfo {
        &self.screen.body
    }

    /// Ask the player to stand straight for a moment and measure their height.
    pub fn request_height_calibration(&mut self) {
        self.commands.push(ComponentCommand::Body(crate::BodyCommand::MeasureHeight));
    }

    /// Open the worn-avatar T-pose mirror. Both controller triggers accept alignment.
    pub fn align_avatar(&mut self) {
        self.commands.push(ComponentCommand::Body(crate::BodyCommand::AlignAvatar));
    }

    /// Run the guided full-body calibration (stand straight, T-pose, floor check).
    pub fn start_body_calibration(&mut self) {
        self.commands.push(ComponentCommand::Body(crate::BodyCommand::Calibrate));
    }

    /// Open or close the VR body menu (Calibrate / Height / Done).
    pub fn show_body_menu(&mut self, open: bool) {
        self.commands.push(ComponentCommand::Body(crate::BodyCommand::Menu(Some(open))));
    }

    /// Set the player's height in cm (`None` returns to the measurement).
    pub fn set_player_height_cm(&mut self, cm: Option<f32>) {
        self.commands.push(ComponentCommand::Body(crate::BodyCommand::SetHeight(cm)));
    }

    /// Start or stop the OSC receiver that takes face tracking from external
    /// tools (VRCFaceTracking-style `/avatar/parameters/FT/v2/...` floats) on
    /// `port` (9000 is the usual). Headset face tracking needs no receiver.
    pub fn set_osc_face_tracking(&mut self, enabled: bool, port: u16) {
        self.commands.push(ComponentCommand::SetOscFaceTracking(enabled.then_some(port)));
    }
}
