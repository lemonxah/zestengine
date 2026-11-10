//! The gameplay half of the in-game API: object lifecycle (spawn / destroy /
//! enable), access to other components, events, physics, audio and app control.
//! Native systems edit component data directly. Structural and external-resource
//! changes use [`crate::ComponentCommand`] and apply at stage boundaries.

use std::any::Any;

use glam::{Mat4, Vec2, Vec3};

use crate::{MaterialParams, ObjectId};

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

/// A type-erased value carried by a command (events).
pub struct AnyBox(pub Box<dyn Any>);

impl std::fmt::Debug for AnyBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AnyBox")
    }
}

type DeferredFn = Box<dyn FnOnce(&mut dyn Any)>;

/// A deferred edit carried by a component command.
pub struct Deferred(pub DeferredFn);

impl std::fmt::Debug for Deferred {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Deferred")
    }
}

/// A deferred edit of a material slot's params (see
/// [`crate::ecs::ObjectContext::modify_material`]).
pub struct MaterialEdit(pub Box<dyn FnOnce(&mut MaterialParams) + 'static>);

impl std::fmt::Debug for MaterialEdit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MaterialEdit")
    }
}
