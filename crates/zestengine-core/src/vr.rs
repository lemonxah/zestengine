//! VR play-space rig: the transform from the headset's stage/tracking space into
//! world space, plus the locomotion that moves it — fly, grab-drag the world,
//! scale yourself, and snap/smooth turn. Headset-agnostic pure math so the editor
//! and a shipped game share it; the OpenXR layer (`zestengine-xr`) supplies raw
//! stage-space poses, this maps them to world and drives movement.
//!
//! Model: a point in the player's tracking space maps to world as
//!   world = origin + Rot_y(yaw) · (stage · scale)
//! so moving `origin` flies, rotating `yaw` turns, and `scale` resizes the player
//! relative to the world (scale > 1 = you're a giant, the world feels small).

use glam::{Quat, Vec3};

/// The play-space → world transform. Locomotion mutates this; pose mapping reads
/// it to place the headset/controllers/trackers in the world.
#[derive(Clone, Copy, Debug)]
pub struct VrRig {
    /// World position of the tracking-space origin (floor under the player).
    pub origin: Vec3,
    /// Yaw (radians) of the tracking space about world +Y.
    pub yaw: f32,
    /// Player scale. >1 makes the player large (covers ground faster, world looks
    /// small); <1 shrinks the player. Clamped to a sane range by the setters.
    pub scale: f32,
}

impl Default for VrRig {
    fn default() -> Self {
        Self {
            origin: Vec3::ZERO,
            yaw: 0.0,
            scale: 1.0,
        }
    }
}

impl VrRig {
    const MIN_SCALE: f32 = 0.01;
    const MAX_SCALE: f32 = 1000.0;

    fn rot(&self) -> Quat {
        Quat::from_rotation_y(self.yaw)
    }

    /// The stage→world matrix (translate(origin) · rotY(yaw) · scale). Compose its
    /// inverse with an XR eye's stage-space view to get the world-space view:
    /// `world_view = eye.view · rig.stage_to_world_mat().inverse()`.
    pub fn stage_to_world_mat(&self) -> glam::Mat4 {
        glam::Mat4::from_scale_rotation_translation(
            Vec3::splat(self.scale),
            self.rot(),
            self.origin,
        )
    }

    /// Map a stage-space position to world space.
    pub fn stage_to_world(&self, p: Vec3) -> Vec3 {
        self.origin + self.rot() * (p * self.scale)
    }

    /// Map a stage-space orientation to world space (just the yaw rotation).
    pub fn stage_rot_to_world(&self, r: Quat) -> Quat {
        self.rot() * r
    }

    /// Map a full stage-space pose (position + orientation) to world space.
    pub fn stage_pose_to_world(&self, pose: (Vec3, Quat)) -> (Vec3, Quat) {
        (self.stage_to_world(pose.0), self.stage_rot_to_world(pose.1))
    }

    /// Fly: translate the play space by a world-space delta (so the player moves
    /// the opposite way through the world). `dir` need not be normalized.
    pub fn fly(&mut self, world_delta: Vec3) {
        self.origin += world_delta;
    }

    /// Smooth/snap turn about a world pivot (usually the headset position so the
    /// player rotates in place rather than orbiting the origin).
    pub fn turn_about(&mut self, pivot: Vec3, delta_yaw: f32) {
        // Keep `pivot` fixed in world while rotating the play space about it.
        let rel = self.origin - pivot;
        let rot = Quat::from_rotation_y(delta_yaw);
        self.origin = pivot + rot * rel;
        self.yaw += delta_yaw;
    }

    /// Scale the player about a world pivot (e.g. the controller, or the midpoint
    /// between two controllers), keeping that pivot fixed in the world.
    pub fn scale_about(&mut self, pivot: Vec3, factor: f32) {
        let new_scale = (self.scale * factor).clamp(Self::MIN_SCALE, Self::MAX_SCALE);
        let applied = new_scale / self.scale; // re-derive after clamping
        // origin' = pivot - (pivot - origin) * applied keeps `pivot` fixed.
        self.origin = pivot - (pivot - self.origin) * applied;
        self.scale = new_scale;
    }

    /// Grab-drag the world: given the world point that was under the grabbing
    /// controller when the grab started (`anchor_world`) and the controller's
    /// CURRENT stage-space position, move `origin` so the anchor stays glued to
    /// the controller. Call every frame while the grab is held.
    pub fn drag_to(&mut self, anchor_world: Vec3, controller_stage: Vec3) {
        // We need stage_to_world(controller_stage) == anchor_world:
        //   origin = anchor_world - Rot · (controller_stage · scale)
        self.origin = anchor_world - self.rot() * (controller_stage * self.scale);
    }

    /// The world point currently under a controller (its stage position mapped to
    /// world). Capture this at grab start to feed `drag_to`.
    pub fn grab_anchor(&self, controller_stage: Vec3) -> Vec3 {
        self.stage_to_world(controller_stage)
    }
}

/// Headset + controller state for gameplay this frame (`ctx.screen.vr`).
/// Poses are in the tracking (stage) space; a `VrPlayer` maps them to world
/// through its own transform. `active` is false without a running headset.
#[derive(Clone, Copy, Debug, Default)]
pub struct VrFrame {
    /// A headset is running and the poses are live.
    pub active: bool,
    /// Head pose (stage space).
    pub head: Option<(Vec3, Quat)>,
    /// Left-hand pose (stage space).
    pub left_hand: Option<(Vec3, Quat)>,
    /// Right-hand pose (stage space).
    pub right_hand: Option<(Vec3, Quat)>,
    /// Left thumbstick, -1..1 per axis.
    pub left_stick: glam::Vec2,
    /// Right thumbstick, -1..1 per axis.
    pub right_stick: glam::Vec2,
    /// Left grip pull, 0..1.
    pub left_grip: f32,
    /// Right grip pull, 0..1.
    pub right_grip: f32,
    /// Left trigger pressed (thresholded).
    pub left_trigger: bool,
    /// Right trigger pressed (thresholded).
    pub right_trigger: bool,
    /// The menu button is pressed.
    pub menu: bool,
    /// Analog trigger pull (0..1).
    pub left_trigger_value: f32,
    /// Analog trigger pull (0..1).
    pub right_trigger_value: f32,
    /// Face buttons: `X` / `Y` on the left controller, `A` / `B` on the right.
    pub a: bool,
    /// The B button (right controller).
    pub b: bool,
    /// The X button (left controller).
    pub x: bool,
    /// The Y button (left controller).
    pub y: bool,
    /// The left stick is clicked.
    pub left_stick_click: bool,
    /// The right stick is clicked.
    pub right_stick_click: bool,
}

/// A world-space ray (origin + unit direction), e.g. a controller's pointer.
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    /// Ray origin (world space).
    pub origin: Vec3,
    /// Unit direction.
    pub dir: Vec3,
}

impl Ray {
    /// The point `t` units along the ray.
    pub fn point_at(&self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }
}

/// Pointer ray from a controller's WORLD pose, aiming along the controller's
/// forward (-Z, the OpenXR aim convention). Use for selecting / placing /
/// pointing at the hand menu.
pub fn pointer_ray(controller_world_pos: Vec3, controller_world_rot: Quat) -> Ray {
    Ray {
        origin: controller_world_pos,
        dir: (controller_world_rot * Vec3::NEG_Z).normalize_or(Vec3::NEG_Z),
    }
}

/// Intersect a ray with an infinite plane (point + normal). Returns the world hit
/// point if the ray crosses the plane in front of its origin. Handy for laser-
/// pointer interaction with a hand-menu quad.
pub fn ray_plane(ray: Ray, plane_point: Vec3, plane_normal: Vec3) -> Option<Vec3> {
    let denom = ray.dir.dot(plane_normal);
    if denom.abs() < 1e-6 {
        return None;
    }
    let t = (plane_point - ray.origin).dot(plane_normal) / denom;
    (t >= 0.0).then(|| ray.point_at(t))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fly_moves_player_through_world() {
        let mut rig = VrRig::default();
        rig.fly(Vec3::new(0.0, 0.0, -5.0));
        // A stage point at the origin now sits 5 units forward in the world.
        assert!((rig.stage_to_world(Vec3::ZERO) - Vec3::new(0.0, 0.0, -5.0)).length() < 1e-5);
    }

    #[test]
    fn scale_about_keeps_pivot_fixed() {
        let mut rig = VrRig::default();
        let pivot = Vec3::new(2.0, 0.0, 1.0);
        rig.scale_about(pivot, 3.0);
        assert!((rig.scale - 3.0).abs() < 1e-5);
        // The pivot maps to the same world point before/after (it's its own preimage).
        let s = (pivot - rig.origin) / rig.scale; // stage preimage of pivot
        let _ = s;
        assert!((rig.stage_to_world((pivot - rig.origin) / rig.scale * 1.0) - pivot).length() < 1e-3);
    }

    #[test]
    fn drag_glues_anchor_to_controller() {
        let mut rig = VrRig::default();
        let controller_stage_start = Vec3::new(0.3, 1.2, -0.4);
        let anchor = rig.grab_anchor(controller_stage_start);
        // Controller moves; dragging keeps the same world anchor under it.
        let controller_stage_now = Vec3::new(0.6, 1.2, -0.1);
        rig.drag_to(anchor, controller_stage_now);
        assert!((rig.stage_to_world(controller_stage_now) - anchor).length() < 1e-4);
    }

    #[test]
    fn turn_about_keeps_pivot_and_adds_yaw() {
        let mut rig = VrRig::default();
        let pivot = Vec3::new(0.0, 0.0, -2.0);
        rig.turn_about(pivot, std::f32::consts::FRAC_PI_2);
        assert!((rig.yaw - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
        // The pivot is unchanged in world: map its stage preimage back.
        // Stage preimage of pivot: rot⁻¹·(pivot-origin)/scale.
        let pre = rig.rot().inverse() * ((pivot - rig.origin) / rig.scale);
        assert!((rig.stage_to_world(pre) - pivot).length() < 1e-3);
    }

    #[test]
    fn pointer_ray_aims_forward() {
        let r = pointer_ray(Vec3::ZERO, Quat::IDENTITY);
        assert!((r.dir - Vec3::NEG_Z).length() < 1e-5);
        let hit = ray_plane(r, Vec3::new(0.0, 0.0, -3.0), Vec3::Z).unwrap();
        assert!((hit - Vec3::new(0.0, 0.0, -3.0)).length() < 1e-4);
    }
}
