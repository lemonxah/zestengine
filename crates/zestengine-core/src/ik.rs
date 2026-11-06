//! Inverse kinematics for VR full-body tracking: solve a skeleton's joint
//! rotations so end effectors (head, hands, hips, feet) reach tracker targets.
//!
//! Two solvers, matching the VRChat/Unreal IK rig split:
//! - [`solve_two_bone`]: analytic two-bone IK for limbs (upper arm + forearm →
//!   hand, thigh + shin → foot), with a pole/hint for elbow/knee direction.
//!   Exact, single-pass, no iteration — the right tool for a 3-joint limb.
//! - [`solve_fabrik`]: FABRIK for longer chains (spine: hips → chest → neck →
//!   head), iterative, handles arbitrary length.
//!
//! Both work in world space on joint positions; the caller converts the
//! resulting positions back into the skeleton's local joint rotations.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::{ObjectRef, TypedComponent};

/// Full-body IK end-effector targets (world-space pose per limb). The source is
/// open: VR trackers, gameplay (look-at / hand reach / foot planting), a
/// cinematic, or a networked remote avatar — any humanoid avatar can be posed
/// from these. `None` for a limb leaves it at its animated/rest pose.
///
/// (Named `TrackerTargets` for the VR path; [`IkTargets`] is the general alias.)
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrackerTargets {
    /// World pose for the head; None keeps the animated pose.
    pub head: Option<(Vec3, Quat)>,
    /// World pose for the left hand; None keeps the animated pose.
    pub left_hand: Option<(Vec3, Quat)>,
    /// World pose for the right hand; None keeps the animated pose.
    pub right_hand: Option<(Vec3, Quat)>,
    /// World pose for the hips; None keeps the animated pose.
    pub hips: Option<(Vec3, Quat)>,
    /// World pose for the left foot; None keeps the animated pose.
    pub left_foot: Option<(Vec3, Quat)>,
    /// World pose for the right foot; None keeps the animated pose.
    pub right_foot: Option<(Vec3, Quat)>,
    /// World pose for the chest; None keeps the animated pose.
    pub chest: Option<(Vec3, Quat)>,
    /// World pose for the left elbow; None keeps the animated pose.
    pub left_elbow: Option<(Vec3, Quat)>,
    /// World pose for the right elbow; None keeps the animated pose.
    pub right_elbow: Option<(Vec3, Quat)>,
    /// World pose for the left knee; None keeps the animated pose.
    pub left_knee: Option<(Vec3, Quat)>,
    /// World pose for the right knee; None keeps the animated pose.
    pub right_knee: Option<(Vec3, Quat)>,
}

/// Number of tracker roles (the length of [`TrackerTargets::slots`]).
pub const TRACKER_SLOTS: usize = 11;

/// Tracker role names in slot order.
pub const TRACKER_ROLES: [&str; TRACKER_SLOTS] = [
    "Head",
    "Left hand",
    "Right hand",
    "Hips",
    "Left foot",
    "Right foot",
    "Chest",
    "Left elbow",
    "Right elbow",
    "Left knee",
    "Right knee",
];

impl TrackerTargets {
    /// Every role in slot order (head, hands, hips, feet, chest, elbows, knees).
    pub fn slots(&self) -> [Option<(Vec3, Quat)>; TRACKER_SLOTS] {
        [
            self.head,
            self.left_hand,
            self.right_hand,
            self.hips,
            self.left_foot,
            self.right_foot,
            self.chest,
            self.left_elbow,
            self.right_elbow,
            self.left_knee,
            self.right_knee,
        ]
    }

    /// Build from the slot array.
    pub fn from_slots(s: [Option<(Vec3, Quat)>; TRACKER_SLOTS]) -> Self {
        Self {
            head: s[0],
            left_hand: s[1],
            right_hand: s[2],
            hips: s[3],
            left_foot: s[4],
            right_foot: s[5],
            chest: s[6],
            left_elbow: s[7],
            right_elbow: s[8],
            left_knee: s[9],
            right_knee: s[10],
        }
    }

    /// True when any target is set.
    pub fn any(&self) -> bool {
        self.slots().iter().any(Option::is_some)
    }

    /// Every pose moved by the rigid transform `m` (scale is ignored for the
    /// rotations, applied to the positions).
    pub fn transformed(&self, m: glam::Mat4) -> Self {
        let (_, r, _) = m.to_scale_rotation_translation();
        let mut s = self.slots();
        for p in s.iter_mut().flatten() {
            *p = (m.transform_point3(p.0), (r * p.1).normalize());
        }
        Self::from_slots(s)
    }
}

/// General-purpose alias: full-body IK targets for any humanoid avatar, from any
/// source (not VR-specific).
pub type IkTargets = TrackerTargets;

/// Where a [`HumanoidIk`] gets its tracker targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum IkSource {
    /// The network's targets on a remote player's avatar, the local trackers
    /// otherwise.
    #[default]
    Auto,
    /// This machine's VR headset, controllers and trackers.
    LocalTrackers,
    /// The targets a Network Avatar on this object receives.
    Network,
    /// One scene object per tracker role.
    Objects,
}

impl IkSource {
    /// Every source, in declaration order.
    pub const ALL: [IkSource; 4] = [IkSource::Auto, IkSource::LocalTrackers, IkSource::Network, IkSource::Objects];
    /// Inspector label of this source.
    pub fn label(self) -> &'static str {
        match self {
            IkSource::Auto => "Auto",
            IkSource::LocalTrackers => "Local trackers",
            IkSource::Network => "Network",
            IkSource::Objects => "Objects",
        }
    }
}

/// Engine-filled state of a [`HumanoidIk`], not saved.
#[derive(Clone, Debug, Default)]
pub struct HumanoidIkRuntime {
    /// Targets used by the last solve, in the avatar's own space.
    pub targets: TrackerTargets,
    /// The model was recognised as humanoid.
    pub humanoid: bool,
}

/// Full-body IK for a humanoid model: poses it from VR trackers (head, hands,
/// hips, feet, chest, elbows, knees), a remote player's replicated trackers
/// or scene objects, on top of its Animator (or rest) pose.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HumanoidIk {
    /// Whether canned locomotion supplies the base pose under tracking.
    pub locomotion: TrackingLocomotion,
    /// Where the tracker targets come from.
    pub source: IkSource,
    /// Objects source: one object per tracker role, in `TRACKER_ROLES` order.
    pub objects: Vec<ObjectRef>,
    /// Local trackers are relative to this object (unset = the scene's VR
    /// Player, or the world origin without one).
    pub tracking_root: ObjectRef,
    /// 0..1 mix over the incoming pose.
    pub weight: f32,
    /// Share of the head's turn the spine follows.
    pub spine_twist: f32,
    /// 0..1: shoulders turning toward reaching hands.
    pub shoulder_rotation: f32,
    /// Without foot trackers, keep feet planted and step procedurally.
    pub plant_feet: bool,
    /// A planted foot steps once this far from where it belongs (m).
    pub step_distance: f32,
    /// Or once the body turned this far from it (degrees).
    pub step_angle: f32,
    /// Seconds per step.
    pub step_time: f32,
    /// Peak step height (m).
    pub step_height: f32,
    /// Engine-filled state (not saved).
    #[serde(skip)]
    pub runtime: HumanoidIkRuntime,
}

impl Default for HumanoidIk {
    fn default() -> Self {
        Self {
            locomotion: TrackingLocomotion::Auto,
            source: IkSource::Auto,
            objects: vec![ObjectRef::NONE; TRACKER_SLOTS],
            tracking_root: ObjectRef::NONE,
            weight: 1.0,
            spine_twist: 0.5,
            shoulder_rotation: 0.5,
            plant_feet: true,
            step_distance: 0.25,
            step_angle: 45.0,
            step_time: 0.22,
            step_height: 0.08,
            runtime: HumanoidIkRuntime::default(),
        }
    }
}

/// Whether canned locomotion supplies the base pose under tracking.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackingLocomotion {
    /// Suppress locomotion when hips and both feet have tracker targets.
    #[default]
    Auto,
    /// Locomotion always animates.
    On,
    /// Locomotion never animates.
    Off,
}

impl TrackingLocomotion {
    /// Whether canned locomotion should animate given the current targets.
    pub fn animate(self, targets: Option<&TrackerTargets>) -> bool {
        match self {
            Self::On => true,
            Self::Off => false,
            Self::Auto => !targets.is_some_and(|t| t.hips.is_some() && t.left_foot.is_some() && t.right_foot.is_some()),
        }
    }
}

#[cfg(test)]
mod tracking_locomotion_tests {
    use super::*;
    #[test]
    fn full_body_tracking_takes_over_only_in_auto_or_off() {
        let pose = Some((Vec3::ZERO, Quat::IDENTITY));
        let hands_only = TrackerTargets { head: pose, left_hand: pose, right_hand: pose, ..Default::default() };
        let full_body = TrackerTargets { hips: pose, left_foot: pose, right_foot: pose, ..hands_only };
        assert!(TrackingLocomotion::Auto.animate(None));
        assert!(TrackingLocomotion::Auto.animate(Some(&hands_only)));
        assert!(!TrackingLocomotion::Auto.animate(Some(&full_body)));
        assert!(TrackingLocomotion::On.animate(Some(&full_body)));
        assert!(!TrackingLocomotion::Off.animate(None));
    }
}

impl TypedComponent for HumanoidIk {
    const NAME: &'static str = "Humanoid IK";
}

/// One tracker's calibration: the fixed rigid offset from the raw tracker pose to
/// the avatar bone it drives, captured once while the player holds a known
/// reference pose (T-pose). A SteamVR/SlimeVR tracker sits at an arbitrary spot on
/// the limb at an arbitrary orientation; this offset cancels that so the bone
/// follows the tracker naturally.
///
///   offset       = tracker_world⁻¹ · bone_world      (captured at T-pose)
///   bone_target  = tracker_world   · offset           (every frame after)
#[derive(Clone, Copy, Debug)]
pub struct TrackerCalibration {
    /// The rigid offset from tracker space to bone space.
    pub offset: glam::Mat4,
}

impl TrackerCalibration {
    /// Capture the offset from a tracker's world pose to its bone's world pose in
    /// the calibration (T-pose) frame.
    pub fn capture(tracker_world: glam::Mat4, bone_world: glam::Mat4) -> Self {
        Self {
            offset: tracker_world.inverse() * bone_world,
        }
    }

    /// Apply the calibration to a live tracker pose, yielding the bone's target
    /// world pose.
    pub fn apply(&self, tracker_world: glam::Mat4) -> glam::Mat4 {
        tracker_world * self.offset
    }
}

/// Result of a two-bone solve: world-space mid (elbow/knee) and end positions,
/// plus the rotations to apply to the root and mid joints.
#[derive(Clone, Copy, Debug)]
pub struct TwoBoneSolution {
    /// Rotation to apply to the root joint.
    pub root_rotation: Quat,
    /// Rotation to apply to the mid joint.
    pub mid_rotation: Quat,
    /// Solved mid (elbow / knee) position.
    pub mid_pos: Vec3,
    /// Solved end (hand / foot) position.
    pub end_pos: Vec3,
}

/// Analytic two-bone IK. `root`/`mid`/`end` are the current world joint
/// positions (e.g. shoulder/elbow/wrist); `target` is where the end should
/// reach; `pole` biases the bend plane (points roughly where the elbow/knee
/// should aim). Lengths are taken from the current pose so bone lengths are
/// preserved. Returns the rotations to apply to the root and mid joints.
pub fn solve_two_bone(root: Vec3, mid: Vec3, end: Vec3, target: Vec3, pole: Vec3) -> TwoBoneSolution {
    let l_upper = (mid - root).length().max(1e-5);
    let l_lower = (end - mid).length().max(1e-5);
    let to_target = target - root;
    let mut dist = to_target.length().max(1e-5);
    // Clamp the reach so the law-of-cosines stays valid (can't over-extend).
    dist = dist.clamp((l_upper - l_lower).abs() + 1e-4, l_upper + l_lower - 1e-4);
    let dir = to_target.normalize_or(Vec3::Y);

    // Interior angle at the root (between the upper bone and the root→target line).
    let cos_root = ((l_upper * l_upper + dist * dist - l_lower * l_lower)
        / (2.0 * l_upper * dist))
        .clamp(-1.0, 1.0);
    let root_angle = cos_root.acos();

    // Bend axis from the pole: the plane containing root→target and the pole hint.
    let pole_dir = (pole - root).normalize_or(Vec3::Z);
    let mut axis = dir.cross(pole_dir);
    if axis.length_squared() < 1e-8 {
        // Pole colinear with the limb; pick any perpendicular axis.
        axis = dir.cross(Vec3::Y);
        if axis.length_squared() < 1e-8 {
            axis = dir.cross(Vec3::X);
        }
    }
    axis = axis.normalize_or(Vec3::Z);

    // Upper bone direction: rotate the root→target dir by root_angle about axis.
    let upper_dir = Quat::from_axis_angle(axis, root_angle) * dir;
    let mid_pos = root + upper_dir * l_upper;
    let lower_dir = (target - mid_pos).normalize_or(upper_dir);
    let end_pos = mid_pos + lower_dir * l_lower;

    // Rotations that take the current bone directions to the solved ones.
    let cur_upper = (mid - root).normalize_or(Vec3::Y);
    let cur_lower = (end - mid).normalize_or(Vec3::Y);
    let root_rotation = Quat::from_rotation_arc(cur_upper, upper_dir);
    let mid_rotation = Quat::from_rotation_arc(cur_lower, lower_dir);

    TwoBoneSolution {
        root_rotation,
        mid_rotation,
        mid_pos,
        end_pos,
    }
}

/// FABRIK (Forward And Backward Reaching Inverse Kinematics) for a chain of
/// joint world positions. The first joint is treated as the anchored root; the
/// last reaches `target`. Bone lengths are preserved from the input. Runs up to
/// `iterations` passes (8–10 is plenty for a spine). Mutates `joints` in place.
pub fn solve_fabrik(joints: &mut [Vec3], target: Vec3, iterations: usize) {
    let n = joints.len();
    if n < 2 {
        return;
    }
    let lengths: Vec<f32> = (0..n - 1)
        .map(|i| (joints[i + 1] - joints[i]).length().max(1e-5))
        .collect();
    let total: f32 = lengths.iter().sum();
    let root = joints[0];

    // Target unreachable: stretch the chain straight toward it.
    if (target - root).length() >= total {
        let dir = (target - root).normalize_or(Vec3::Y);
        for i in 1..n {
            joints[i] = joints[i - 1] + dir * lengths[i - 1];
        }
        return;
    }

    let tol = 1e-3;
    for _ in 0..iterations {
        if (joints[n - 1] - target).length() < tol {
            break;
        }
        // Backward pass: end → root, pulling the tip to the target.
        joints[n - 1] = target;
        for i in (0..n - 1).rev() {
            let dir = (joints[i] - joints[i + 1]).normalize_or(Vec3::Y);
            joints[i] = joints[i + 1] + dir * lengths[i];
        }
        // Forward pass: root → end, re-anchoring the root.
        joints[0] = root;
        for i in 0..n - 1 {
            let dir = (joints[i + 1] - joints[i]).normalize_or(Vec3::Y);
            joints[i + 1] = joints[i] + dir * lengths[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_bone_reaches_target_in_range() {
        let root = Vec3::ZERO;
        let mid = Vec3::new(0.0, -1.0, 0.0);
        let end = Vec3::new(0.0, -2.0, 0.0);
        let target = Vec3::new(1.0, -1.0, 0.0);
        let s = solve_two_bone(root, mid, end, target, Vec3::new(0.0, -1.0, 1.0));
        // End effector should land very close to the (reachable) target.
        assert!((s.end_pos - target).length() < 1e-2, "end {:?}", s.end_pos);
    }

    #[test]
    fn calibration_roundtrips_then_follows_tracker() {
        use glam::{Mat4, Quat};
        // T-pose: tracker offset from the bone by some rigid transform.
        let bone = Mat4::from_rotation_translation(Quat::IDENTITY, Vec3::new(0.5, 1.0, 0.0));
        let tracker =
            Mat4::from_rotation_translation(Quat::from_rotation_z(0.3), Vec3::new(0.55, 1.1, 0.02));
        let cal = TrackerCalibration::capture(tracker, bone);
        // At the calibration pose, applying recovers the bone exactly.
        let back = cal.apply(tracker);
        assert!((back.w_axis.truncate() - bone.w_axis.truncate()).length() < 1e-4);
        // Move the tracker by a known world translation; the bone target moves with it.
        let moved = Mat4::from_translation(Vec3::new(0.0, 0.2, 0.3)) * tracker;
        let target = cal.apply(moved);
        let expected = bone.w_axis.truncate() + Vec3::new(0.0, 0.2, 0.3);
        assert!((target.w_axis.truncate() - expected).length() < 1e-4);
    }

    #[test]
    fn fabrik_reaches_and_preserves_length() {
        let mut joints = vec![
            Vec3::ZERO,
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 3.0, 0.0),
        ];
        let target = Vec3::new(1.5, 1.5, 0.0);
        solve_fabrik(&mut joints, target, 16);
        assert!((joints[3] - target).length() < 1e-2);
        // Bone lengths preserved (~1.0 each).
        for i in 0..3 {
            assert!(((joints[i + 1] - joints[i]).length() - 1.0).abs() < 1e-2);
        }
    }
}
