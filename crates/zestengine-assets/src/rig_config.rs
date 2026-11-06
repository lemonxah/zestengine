//! Logic behind the Rig section and Configure mode: per-bone states, body-group
//! states for the figure, mapping edits and pose operations. UI-free, so it is
//! tested here; the drawing lives in the engine's `rig_editor`.

use crate::human_rig::{ALL, AnimationType, HUMAN_BONE_COUNT, HumanBone, PoseBone, REQUIRED_BONE_COUNT, RigImport, validate_rig};
use crate::ref_pose::{LocalPose, bones_out_of_pose, sample_bind_pose};
use crate::Skeleton;

/// How one human bone's assignment looks in Configure mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoneState {
    Unassigned,
    Valid,
    /// Assigned but [`validate_rig`] objects (not found, duplicate, wrong parent, zero length ...).
    Invalid,
}

/// Per-bone states and messages for one mapping.
#[derive(Clone, Debug)]
pub struct BoneReport {
    pub state: [BoneState; HUMAN_BONE_COUNT],
    /// Messages per bone (index = `bone as usize`).
    pub messages: Vec<Vec<String>>,
    /// Messages about the whole rig.
    pub general: Vec<String>,
}

impl BoneReport {
    pub fn state(&self, b: HumanBone) -> BoneState {
        self.state[b as usize]
    }

    pub fn messages(&self, b: HumanBone) -> &[String] {
        &self.messages[b as usize]
    }

    pub fn error_count(&self) -> usize {
        self.messages.iter().map(Vec::len).sum::<usize>() + self.general.len()
    }
}

/// Validate `human` against `skel` and sort the messages onto their bones.
pub fn bone_report(skel: &Skeleton, human: &[(HumanBone, String)]) -> BoneReport {
    let mut state = [BoneState::Unassigned; HUMAN_BONE_COUNT];
    let mut messages = vec![Vec::new(); HUMAN_BONE_COUNT];
    let mut general = Vec::new();
    for (b, _) in human {
        state[*b as usize] = BoneState::Valid;
    }
    // A required bone missing is reported by the validator for the bone itself,
    // but it stays Unassigned (grey) rather than turning red.
    for issue in validate_rig(skel, human) {
        match issue.bone {
            Some(b) => {
                if state[b as usize] != BoneState::Unassigned {
                    state[b as usize] = BoneState::Invalid;
                }
                messages[b as usize].push(issue.message);
            }
            None => general.push(issue.message),
        }
    }
    BoneReport { state, messages, general }
}

/// The body groups the figure tints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    Body,
    Head,
    LeftArm,
    RightArm,
    LeftLeg,
    RightLeg,
    LeftHand,
    RightHand,
}

impl Group {
    pub fn bones(self) -> Vec<HumanBone> {
        use HumanBone as B;
        let fingers = |left: bool| {
            let base = if left { 24 } else { 39 };
            ALL[base..base + 15].to_vec()
        };
        match self {
            Group::Body => vec![B::Hips, B::Spine, B::Chest, B::UpperChest],
            Group::Head => vec![B::Neck, B::Head, B::LeftEye, B::RightEye, B::Jaw],
            Group::LeftArm => vec![B::LeftShoulder, B::LeftUpperArm, B::LeftLowerArm, B::LeftHand],
            Group::RightArm => vec![B::RightShoulder, B::RightUpperArm, B::RightLowerArm, B::RightHand],
            Group::LeftLeg => vec![B::LeftUpperLeg, B::LeftLowerLeg, B::LeftFoot, B::LeftToes],
            Group::RightLeg => vec![B::RightUpperLeg, B::RightLowerLeg, B::RightFoot, B::RightToes],
            Group::LeftHand => fingers(true),
            Group::RightHand => fingers(false),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Group::Body => "Body",
            Group::Head => "Head",
            Group::LeftArm => "Left Arm",
            Group::RightArm => "Right Arm",
            Group::LeftLeg => "Left Leg",
            Group::RightLeg => "Right Leg",
            Group::LeftHand => "Left Hand",
            Group::RightHand => "Right Hand",
        }
    }
}

/// Tint of a body group on the figure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupState {
    /// Nothing assigned.
    Grey,
    /// At least one bone valid and no errors.
    Green,
    /// A required bone is unassigned, or a bone has an error.
    Red,
}

/// Tint for `group`: red when a required bone is missing or any bone is
/// invalid (for fingers, when a distal or intermediate bone is assigned
/// without the one before it), green when at least one bone is valid.
pub fn group_state(report: &BoneReport, group: Group) -> GroupState {
    let bones = group.bones();
    let mut any_valid = false;
    let mut bad = false;
    for b in &bones {
        match report.state(*b) {
            BoneState::Valid => any_valid = true,
            BoneState::Invalid => bad = true,
            BoneState::Unassigned => {
                if b.is_required() {
                    bad = true;
                }
            }
        }
    }
    if matches!(group, Group::LeftHand | Group::RightHand) {
        for f in bones.chunks(3) {
            let assigned = |i: usize| report.state(f[i]) != BoneState::Unassigned;
            if (assigned(2) && !assigned(1)) || (assigned(1) && !assigned(0)) {
                bad = true;
            }
        }
    }
    if bad && (any_valid || bones.iter().any(|b| b.is_required())) {
        GroupState::Red
    } else if any_valid {
        GroupState::Green
    } else {
        GroupState::Grey
    }
}

/// Set (`Some`) or clear (`None`) one bone's joint.
pub fn assign_bone(human: &mut Vec<(HumanBone, String)>, bone: HumanBone, joint: Option<&str>) {
    human.retain(|(b, _)| *b != bone);
    if let Some(j) = joint {
        human.push((bone, j.to_string()));
        human.sort_by_key(|(b, _)| *b as usize);
    }
}

/// What the Rig section prints under the mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RigSummary {
    pub required_mapped: usize,
    pub required_total: usize,
    pub mapped: usize,
}

pub fn summary(rig: &RigImport) -> RigSummary {
    RigSummary {
        required_mapped: ALL.iter().filter(|b| b.is_required() && rig.joint_name(**b).is_some()).count(),
        required_total: REQUIRED_BONE_COUNT,
        mapped: rig.human.len(),
    }
}

/// "Humanoid, 15/15 required bones" style line.
pub fn summary_line(rig: &RigImport) -> String {
    let s = summary(rig);
    let kind = match rig.animation_type {
        AnimationType::None => "None",
        AnimationType::Generic => "Generic",
        AnimationType::Humanoid => "Humanoid",
    };
    format!("{kind}, {}/{} required bones, {} mapped", s.required_mapped, s.required_total, s.mapped)
}

/// Pose menu: Enforce T-Pose. Stores the closer of bind and file pose, made valid
/// (the exact canonical T-pose when the rig forces one).
pub fn enforce_tpose(rig: &mut RigImport, skel: &Skeleton) {
    rig.reference_pose = crate::ref_pose::compute_reference_pose_with(skel, &rig.human, rig.force_tpose);
}

/// Pose menu: Reset. Drops the stored reference pose, so the file pose is used.
pub fn reset_pose(rig: &mut RigImport) {
    rig.reference_pose = Vec::<PoseBone>::new();
}

/// Pose menu: Sample Bind-Pose. Stores the pose the skin was bound in; false
/// when the skeleton has no bind data.
pub fn sample_bind(rig: &mut RigImport, skel: &Skeleton) -> bool {
    let Some(p) = sample_bind_pose(skel) else { return false };
    rig.reference_pose = skel
        .joints
        .iter()
        .enumerate()
        .map(|(i, j)| PoseBone { name: j.name.clone(), translation: p.translation[i].to_array(), rotation: p.rotation[i].to_array(), scale: p.scale[i].to_array() })
        .collect();
    true
}

/// Mapped bones that are out of the T-pose tolerance: in the stored reference
/// pose when `reference`, else in the file pose as the model displays it.
pub fn pose_errors(skel: &Skeleton, rig: &RigImport, reference: bool) -> Vec<HumanBone> {
    let mut pose = LocalPose::rest(skel);
    for p in rig.reference_pose.iter().filter(|_| reference) {
        if let Some(i) = skel.joints.iter().position(|j| j.name == p.name) {
            let r = glam::Quat::from_array(p.rotation);
            if r.length_squared() > 1e-6 {
                pose.rotation[i] = r.normalize();
            }
        }
    }
    bones_out_of_pose(skel, &rig.human, &pose)
}

/// Joint names for the transform pickers, in hierarchy order.
pub fn joint_names(skel: &Skeleton) -> Vec<&str> {
    skel.joints.iter().map(|j| j.name.as_str()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Joint;
    use glam::{Mat4, Quat, Vec3};

    fn skel(names: &[(&str, Option<usize>, Vec3)]) -> Skeleton {
        Skeleton {
            joints: names
                .iter()
                .map(|(n, p, w)| Joint {
                    name: (*n).into(),
                    parent: *p,
                    inverse_bind: Mat4::from_translation(-*w),
                    rest_translation: *w,
                    rest_rotation: Quat::IDENTITY,
                    rest_scale: Vec3::ONE,
                })
                .collect(),
            rig: None,
            ..Default::default()
        }
    }

    #[test]
    fn states_follow_the_validator() {
        let s = skel(&[("root", None, Vec3::ZERO), ("hips", Some(0), Vec3::Y), ("spine", Some(1), Vec3::new(0.0, 1.2, 0.0)), ("arm", Some(2), Vec3::new(0.3, 1.2, 0.0))]);
        let human = vec![(HumanBone::Hips, "hips".to_string()), (HumanBone::Spine, "nope".to_string()), (HumanBone::LeftUpperArm, "arm".to_string())];
        let r = bone_report(&s, &human);
        assert_eq!(r.state(HumanBone::Hips), BoneState::Valid);
        assert_eq!(r.state(HumanBone::Spine), BoneState::Invalid);
        assert!(r.messages(HumanBone::Spine)[0].contains("not found"));
        assert_eq!(r.state(HumanBone::Head), BoneState::Unassigned);
        assert!(r.error_count() > 0);
    }

    #[test]
    fn group_tints() {
        let none = bone_report(&skel(&[("a", None, Vec3::ZERO)]), &[]);
        assert_eq!(group_state(&none, Group::LeftHand), GroupState::Grey);
        assert_eq!(group_state(&none, Group::Body), GroupState::Red, "required bones missing");
        assert_eq!(group_state(&none, Group::Head), GroupState::Red, "Head is required");

        // A distal finger bone without its proximal is an error.
        let names: Vec<(String, Option<usize>)> = vec![("h".into(), None), ("d".into(), Some(0))];
        let s = skel(&names.iter().map(|(n, p)| (n.as_str(), *p, Vec3::ZERO)).collect::<Vec<_>>());
        let bad = bone_report(&s, &[(HumanBone::LeftThumbDistal, "d".into())]);
        assert_eq!(group_state(&bad, Group::LeftHand), GroupState::Red);
        let ok = bone_report(&s, &[(HumanBone::LeftThumbProximal, "d".into())]);
        assert_ne!(group_state(&ok, Group::LeftHand), GroupState::Red);
    }

    #[test]
    fn assign_and_summary() {
        let mut rig = RigImport { animation_type: AnimationType::Humanoid, ..Default::default() };
        assign_bone(&mut rig.human, HumanBone::Head, Some("head"));
        assign_bone(&mut rig.human, HumanBone::Hips, Some("hips"));
        assign_bone(&mut rig.human, HumanBone::Jaw, Some("jaw"));
        assert_eq!(rig.human[0].0, HumanBone::Hips, "kept in bone order");
        assert_eq!(summary(&rig), RigSummary { required_mapped: 2, required_total: 15, mapped: 3 });
        assert_eq!(summary_line(&rig), "Humanoid, 2/15 required bones, 3 mapped");
        assign_bone(&mut rig.human, HumanBone::Head, None);
        assert_eq!(summary(&rig).mapped, 2);
        assign_bone(&mut rig.human, HumanBone::Hips, Some("pelvis"));
        assert_eq!(rig.joint_name(HumanBone::Hips), Some("pelvis"));
    }

    #[test]
    fn pose_operations_reflect_the_arm_angle() {
        // Hips/spine/head column plus an arm hanging down: out of T-pose until enforced.
        let mut s = skel(&[
            ("hips", None, Vec3::new(0.0, 1.0, 0.0)),
            ("spine", Some(0), Vec3::new(0.0, 0.2, 0.0)),
            ("chest", Some(1), Vec3::new(0.0, 0.2, 0.0)),
            ("head", Some(2), Vec3::new(0.0, 0.3, 0.0)),
            ("ual", Some(2), Vec3::new(0.1, 0.1, 0.0)),
            ("lal", Some(4), Vec3::new(0.0, -0.3, 0.0)),
            ("hl", Some(5), Vec3::new(0.0, -0.3, 0.0)),
            ("ular", Some(2), Vec3::new(-0.1, 0.1, 0.0)),
            ("lar", Some(7), Vec3::new(0.0, -0.3, 0.0)),
            ("hr", Some(8), Vec3::new(0.0, -0.3, 0.0)),
        ]);
        // Recompute inverse binds as world positions.
        let mut world = vec![Vec3::ZERO; s.joints.len()];
        for i in 0..s.joints.len() {
            world[i] = s.joints[i].parent.map(|p| world[p]).unwrap_or(Vec3::ZERO) + s.joints[i].rest_translation;
            s.joints[i].inverse_bind = Mat4::from_translation(-world[i]);
        }
        let human: Vec<(HumanBone, String)> = [
            (HumanBone::Hips, "hips"),
            (HumanBone::Spine, "spine"),
            (HumanBone::Chest, "chest"),
            (HumanBone::Head, "head"),
            (HumanBone::LeftUpperArm, "ual"),
            (HumanBone::LeftLowerArm, "lal"),
            (HumanBone::LeftHand, "hl"),
            (HumanBone::RightUpperArm, "ular"),
            (HumanBone::RightLowerArm, "lar"),
            (HumanBone::RightHand, "hr"),
        ]
        .iter()
        .map(|(b, n)| (*b, n.to_string()))
        .collect();
        let mut rig = RigImport { animation_type: AnimationType::Humanoid, human, ..Default::default() };
        assert!(pose_errors(&s, &rig, false).contains(&HumanBone::LeftUpperArm), "arm hangs down");
        enforce_tpose(&mut rig, &s);
        assert!(!rig.reference_pose.is_empty());
        assert!(!pose_errors(&s, &rig, true).contains(&HumanBone::LeftUpperArm), "reference pose raises it");
        reset_pose(&mut rig);
        assert!(rig.reference_pose.is_empty());
    }
}
