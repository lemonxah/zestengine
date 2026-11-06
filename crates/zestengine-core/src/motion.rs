//! Replayable movement, for networked prediction (Network Object >
//! Predicted): the state a controller's movement rewinds to, the world it
//! moves through ([`MoveHost`]) and the controller side ([`Predictable`]).
//! [`ComponentCtx`] is a `MoveHost`, and so is the engine's rewind / replay
//! simulation, so a live frame and its replay run the same step.

use glam::{Mat4, Quat, Vec3};

use crate::{
    Character, Climb, Component, ComponentCtx, InputState, NetworkObject, ObjectId, ObjectRef, RayFilter, RayHit,
    Swimmer,
};

/// A controller's movement state: what prediction rewinds to and replays
/// inputs from.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MotionState {
    /// Current velocity (m/s).
    pub vel: Vec3,
    /// View yaw / pitch (degrees). They follow the owner's input, so
    /// reconciliation keeps the owner's.
    pub yaw: f32,
    pub pitch: f32,
    /// Facing of a body that turns apart from the view (third person).
    pub body_yaw: f32,
    /// On the ground.
    pub grounded: bool,
    /// Floor height when there is no Character.
    pub floor_y: f32,
    /// Crouching.
    pub crouched: bool,
    /// The Character capsule is shrunk for the crouch.
    pub crouch_capsule: bool,
    /// In water.
    pub swimming: bool,
    /// Seconds of breath left while it has been under (None = full).
    pub breath: Option<f32>,
    /// Current climb, if climbing.
    pub climb: Option<Climb>,
}

/// What a movement step reads and moves: the object, its Character
/// Controller and the world around it.
pub trait MoveHost {
    /// The object's id.
    fn self_id(&self) -> ObjectId;
    /// Local translation / rotation (the object's TRS fields).
    fn translation(&self) -> Vec3;
    /// Set the local translation.
    fn set_translation(&mut self, t: Vec3);
    /// Local rotation.
    fn rotation(&self) -> Quat;
    /// Local scale, including the calibrated VR tracking-space scale.
    fn scale(&self) -> Vec3 { Vec3::ONE }
    /// Set the local rotation.
    fn set_rotation(&mut self, r: Quat);
    /// World matrix of the parent (None for a root object).
    fn parent_world(&self) -> Option<Mat4>;
    /// A Character moves the object through the physics world.
    fn has_character(&self) -> bool;
    /// Move the Character by a world delta, colliding.
    fn move_character(&mut self, delta: Vec3);
    /// The Character touched the ground on its last move.
    fn grounded(&self) -> bool;
    /// The Character's (radius, height, center).
    fn capsule(&self) -> Option<(f32, f32, [f32; 3])>;
    /// Resize the capsule; `capsule` reports the old size until the next move.
    fn set_capsule(&mut self, height: f32, center: [f32; 3]);
    /// The object's swim tuning (defaults without a Swimmer).
    fn swimmer(&self) -> Swimmer;
    /// Height of the water surface above or below `p`.
    fn water_surface(&self, p: Vec3) -> Option<f32>;
    /// Ray-cast against the physics world.
    fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32, filter: &RayFilter) -> Option<RayHit>;
    /// Set an Animator parameter on the object.
    fn set_animator_bool(&mut self, name: &str, value: bool);

    /// World matrix of the object, including calibrated tracking-space scale.
    fn world(&self) -> Mat4 {
        let local = Mat4::from_scale_rotation_translation(self.scale(), self.rotation(), self.translation());
        self.parent_world().map_or(local, |p| p * local)
    }

    /// World position.
    fn world_position(&self) -> Vec3 {
        let t = self.translation();
        self.parent_world().map_or(t, |p| p.transform_point3(t))
    }

    /// Set the world position (converting through the parent).
    fn set_world_position(&mut self, p: Vec3) {
        let local = self.parent_world().map_or(p, |m| m.inverse().transform_point3(p));
        self.set_translation(local);
    }
}

impl MoveHost for ComponentCtx<'_> {
    fn self_id(&self) -> ObjectId {
        ComponentCtx::self_id(self)
    }
    fn translation(&self) -> Vec3 {
        *self.translation
    }
    fn set_translation(&mut self, t: Vec3) {
        *self.translation = t;
    }
    fn rotation(&self) -> Quat {
        *self.rotation
    }
    fn scale(&self) -> Vec3 { *self.scale }
    fn set_rotation(&mut self, r: Quat) {
        *self.rotation = r;
    }
    fn parent_world(&self) -> Option<Mat4> {
        self.parent_world
    }
    fn has_character(&self) -> bool {
        self.physics.is_some() && self.get_self::<Character>().is_some_and(|c| c.use_default_capsule)
    }
    fn move_character(&mut self, delta: Vec3) {
        ComponentCtx::move_character(self, delta);
    }
    fn grounded(&self) -> bool {
        self.is_grounded()
    }
    fn capsule(&self) -> Option<(f32, f32, [f32; 3])> {
        self.get_self::<Character>().map(|c| (c.radius, c.height, c.center))
    }
    fn set_capsule(&mut self, height: f32, center: [f32; 3]) {
        let me = ObjectRef::new(ComponentCtx::self_id(self));
        self.modify::<Character>(me, move |c| {
            c.height = height;
            c.center = center;
        });
    }
    fn swimmer(&self) -> Swimmer {
        self.get_self::<Swimmer>().copied().unwrap_or_default()
    }
    fn water_surface(&self, p: Vec3) -> Option<f32> {
        ComponentCtx::water_surface(self, p).map(|(h, _)| h)
    }
    fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32, filter: &RayFilter) -> Option<RayHit> {
        self.raycast_filtered(origin, dir, max_dist, filter)
    }
    fn set_animator_bool(&mut self, name: &str, value: bool) {
        let me = ComponentCtx::self_id(self);
        ComponentCtx::set_animator_bool(self, me, name, value);
    }
}

/// A controller whose movement networked prediction can rewind and replay.
pub trait Predictable {
    /// The movement state prediction rewinds to.
    fn motion(&self) -> MotionState;
    /// Put the movement back into `m` (resizing the capsule through `host`
    /// to match).
    fn set_motion(&mut self, m: &MotionState, host: &mut dyn MoveHost);
    /// What the frame just simulated took from outside the input actions
    /// (view angles, headset pose, a teleport), carried by its input command.
    fn frame_extra(&self) -> Vec<f32>;
    /// Simulate one recorded input frame. `extra` is the frame's
    /// [`Self::frame_extra`] (empty when it had none).
    fn replay(&mut self, host: &mut dyn MoveHost, input: &InputState, extra: &[f32], dt: f32);
    /// The local player drives it (a Pawn only while possessed).
    fn locally_driven(&self) -> bool {
        true
    }
}

/// The component as a [`Predictable`] controller, if it is one.
pub fn as_predictable(c: &mut dyn Component) -> Option<&mut dyn Predictable> {
    c.predictable_mut()
}

/// True when the component implements [`Predictable`] (has a predictable
/// controller).
pub fn is_predictable(c: &dyn Component) -> bool {
    c.predictable().is_some()
}

/// A networked copy another peer drives (its owner, or the host simulating
/// the owner's predicted inputs): local input mustn't move it.
pub fn driven_remotely(ctx: &ComponentCtx) -> bool {
    ctx.net.connected
        && (ctx.get_self::<NetworkObject>().is_some() || ctx.get_self::<crate::Sync>().is_some())
        && !ctx.has_authority()
}

/// The view (yaw, pitch) a frame's extra input carries, if it does.
pub(crate) fn view_of(extra: &[f32]) -> Option<(f32, f32)> {
    match extra {
        [yaw, pitch, ..] => Some((*yaw, *pitch)),
        _ => None,
    }
}
