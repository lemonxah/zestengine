//! Biped Pose: an exact canonical T-pose from the bone mapping.
//!
//! [`crate::ref_pose::make_pose_valid`] only pulls a pose back inside a
//! tolerance cone. The Biped Pose goes all the way: every mapped bone with a
//! mapped child is turned so it points exactly along the canonical axis
//! (spine up, arms along the X axis, legs down, feet forward), the avatar's
//! left on +X, facing +Z, like the reference-pose frame. Translations and
//! scales are kept; only rotations change. Bones without a mapped child (head,
//! toes, finger tips) keep their rotation relative to their parent.

use crate::human_rig::{ALL, HumanBone, PoseBone};
use crate::ref_pose::LocalPose;
use crate::Skeleton;
use glam::{Mat4, Quat, Vec3};

/// Where `bone` should point (towards the first mapped bone of the list) in
/// the canonical pose.
fn rule(bone: HumanBone) -> Option<(&'static [HumanBone], Vec3)> {
    use HumanBone::*;
    const UP: Vec3 = Vec3::Y;
    Some(match bone {
        Hips => (&[Spine, Chest, UpperChest, Neck], UP),
        Spine => (&[Chest, UpperChest, Neck], UP),
        Chest => (&[UpperChest, Neck], UP),
        UpperChest => (&[Neck], UP),
        Neck => (&[Head], UP),
        LeftShoulder => (&[LeftUpperArm], Vec3::X),
        LeftUpperArm => (&[LeftLowerArm], Vec3::X),
        LeftLowerArm => (&[LeftHand], Vec3::X),
        LeftHand => (&[LeftMiddleProximal, LeftIndexProximal, LeftRingProximal, LeftLittleProximal], Vec3::X),
        RightShoulder => (&[RightUpperArm], Vec3::NEG_X),
        RightUpperArm => (&[RightLowerArm], Vec3::NEG_X),
        RightLowerArm => (&[RightHand], Vec3::NEG_X),
        RightHand => (&[RightMiddleProximal, RightIndexProximal, RightRingProximal, RightLittleProximal], Vec3::NEG_X),
        LeftUpperLeg => (&[LeftLowerLeg], Vec3::NEG_Y),
        RightUpperLeg => (&[RightLowerLeg], Vec3::NEG_Y),
        LeftLowerLeg => (&[LeftFoot], Vec3::NEG_Y),
        RightLowerLeg => (&[RightFoot], Vec3::NEG_Y),
        LeftFoot => (&[LeftToes], Vec3::Z),
        RightFoot => (&[RightToes], Vec3::Z),
        LeftThumbProximal => (&[LeftThumbIntermediate], Vec3::new(1.0, 0.0, 1.0).normalize()),
        LeftThumbIntermediate => (&[LeftThumbDistal], Vec3::new(1.0, 0.0, 1.0).normalize()),
        RightThumbProximal => (&[RightThumbIntermediate], Vec3::new(-1.0, 0.0, 1.0).normalize()),
        RightThumbIntermediate => (&[RightThumbDistal], Vec3::new(-1.0, 0.0, 1.0).normalize()),
        LeftIndexProximal => (&[LeftIndexIntermediate], Vec3::X),
        LeftIndexIntermediate => (&[LeftIndexDistal], Vec3::X),
        LeftMiddleProximal => (&[LeftMiddleIntermediate], Vec3::X),
        LeftMiddleIntermediate => (&[LeftMiddleDistal], Vec3::X),
        LeftRingProximal => (&[LeftRingIntermediate], Vec3::X),
        LeftRingIntermediate => (&[LeftRingDistal], Vec3::X),
        LeftLittleProximal => (&[LeftLittleIntermediate], Vec3::X),
        LeftLittleIntermediate => (&[LeftLittleDistal], Vec3::X),
        RightIndexProximal => (&[RightIndexIntermediate], Vec3::NEG_X),
        RightIndexIntermediate => (&[RightIndexDistal], Vec3::NEG_X),
        RightMiddleProximal => (&[RightMiddleIntermediate], Vec3::NEG_X),
        RightMiddleIntermediate => (&[RightMiddleDistal], Vec3::NEG_X),
        RightRingProximal => (&[RightRingIntermediate], Vec3::NEG_X),
        RightRingIntermediate => (&[RightRingDistal], Vec3::NEG_X),
        RightLittleProximal => (&[RightLittleIntermediate], Vec3::NEG_X),
        RightLittleIntermediate => (&[RightLittleDistal], Vec3::NEG_X),
        _ => return None,
    })
}

/// Depth of `bone` in the human tree (Hips = 0), so parents are turned first.
fn depth(bone: HumanBone) -> usize {
    std::iter::successors(bone.parent(), |b| b.parent()).count()
}

fn world_pos(m: &Mat4) -> Vec3 {
    m.w_axis.truncate()
}

fn world_rot(m: &Mat4) -> Quat {
    m.to_scale_rotation_translation().1
}

/// The Biped Pose of `skel` for `human`, as one [`PoseBone`] per joint (the
/// shape of `RigImport::reference_pose`). Empty without a mapped Hips.
pub fn biped_pose(skel: &Skeleton, human: &[(HumanBone, String)]) -> Vec<PoseBone> {
    let joint_of = |b: HumanBone| -> Option<usize> {
        let name = &human.iter().find(|(h, _)| *h == b)?.1;
        skel.joints.iter().position(|j| &j.name == name)
    };
    if joint_of(HumanBone::Hips).is_none() {
        return Vec::new();
    }
    let mut pose = LocalPose::rest(skel);
    let locals = |p: &LocalPose| -> Vec<Mat4> {
        (0..p.rotation.len()).map(|i| Mat4::from_scale_rotation_translation(p.scale[i], p.rotation[i], p.translation[i])).collect()
    };
    let turn = |pose: &mut LocalPose, j: usize, target_dir: Vec3, towards: usize| {
        let world = skel.world_matrices(&locals(pose));
        let d = (world_pos(&world[towards]) - world_pos(&world[j])).normalize_or_zero();
        if d == Vec3::ZERO {
            return;
        }
        let new_world = Quat::from_rotation_arc(d, target_dir) * world_rot(&world[j]);
        let parent_rot = skel.joints[j].parent.map_or(Quat::IDENTITY, |p| world_rot(&world[p]));
        pose.rotation[j] = (parent_rot.inverse() * new_world).normalize();
    };
    // The hips first face the canonical way: left leg on +X.
    if let (Some(h), Some(l), Some(r)) = (joint_of(HumanBone::Hips), joint_of(HumanBone::LeftUpperLeg), joint_of(HumanBone::RightUpperLeg)) {
        let world = skel.world_matrices(&locals(&pose));
        let side = world_pos(&world[l]) - world_pos(&world[r]);
        let side = Vec3::new(side.x, 0.0, side.z).normalize_or_zero();
        if side != Vec3::ZERO {
            let q = Quat::from_rotation_arc(side, Vec3::X);
            let parent_rot = skel.joints[h].parent.map_or(Quat::IDENTITY, |p| world_rot(&world[p]));
            pose.rotation[h] = (parent_rot.inverse() * q * world_rot(&world[h])).normalize();
        }
    }
    let mut bones: Vec<HumanBone> = ALL.to_vec();
    bones.sort_by_key(|b| (depth(*b), *b as usize));
    for b in bones {
        let (Some((children, dir)), Some(j)) = (rule(b), joint_of(b)) else { continue };
        let Some(c) = children.iter().find_map(|c| joint_of(*c)) else { continue };
        turn(&mut pose, j, dir, c);
    }
    skel.joints
        .iter()
        .enumerate()
        .map(|(i, j)| PoseBone { name: j.name.clone(), translation: pose.translation[i].to_array(), rotation: pose.rotation[i].to_array(), scale: pose.scale[i].to_array() })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Joint;

    fn joint(name: &str, parent: Option<usize>, local: Vec3) -> Joint {
        Joint { name: name.into(), parent, inverse_bind: Mat4::IDENTITY, rest_translation: local, rest_rotation: Quat::IDENTITY, rest_scale: Vec3::ONE }
    }

    /// An A-pose: arms slope down and out, hips slightly turned.
    fn a_pose() -> (Skeleton, Vec<(HumanBone, String)>) {
        let skel = Skeleton {
            joints: vec![
                joint("Hips", None, Vec3::new(0.0, 1.0, 0.0)),
                joint("Spine", Some(0), Vec3::new(0.0, 0.2, 0.05)),
                joint("Neck", Some(1), Vec3::new(0.0, 0.4, 0.0)),
                joint("Head", Some(2), Vec3::new(0.0, 0.15, 0.0)),
                joint("UpperArm_L", Some(1), Vec3::new(0.2, 0.25, 0.0)),
                joint("LowerArm_L", Some(4), Vec3::new(0.15, -0.15, 0.0)),
                joint("Hand_L", Some(5), Vec3::new(0.15, -0.15, 0.0)),
                joint("UpperArm_R", Some(1), Vec3::new(-0.2, 0.25, 0.0)),
                joint("LowerArm_R", Some(7), Vec3::new(-0.15, -0.15, 0.0)),
                joint("Hand_R", Some(8), Vec3::new(-0.15, -0.15, 0.0)),
                joint("Leg_L", Some(0), Vec3::new(0.1, -0.05, 0.0)),
                joint("Knee_L", Some(10), Vec3::new(0.0, -0.45, 0.0)),
                joint("Foot_L", Some(11), Vec3::new(0.0, -0.45, 0.0)),
                joint("Leg_R", Some(0), Vec3::new(-0.1, -0.05, 0.0)),
                joint("Knee_R", Some(13), Vec3::new(0.0, -0.45, 0.0)),
                joint("Foot_R", Some(14), Vec3::new(0.0, -0.45, 0.0)),
            ],
            rig: None,
            ..Default::default()
        };
        use HumanBone::*;
        let map = [
            (Hips, "Hips"), (Spine, "Spine"), (Neck, "Neck"), (Head, "Head"),
            (LeftUpperArm, "UpperArm_L"), (LeftLowerArm, "LowerArm_L"), (LeftHand, "Hand_L"),
            (RightUpperArm, "UpperArm_R"), (RightLowerArm, "LowerArm_R"), (RightHand, "Hand_R"),
            (LeftUpperLeg, "Leg_L"), (LeftLowerLeg, "Knee_L"), (LeftFoot, "Foot_L"),
            (RightUpperLeg, "Leg_R"), (RightLowerLeg, "Knee_R"), (RightFoot, "Foot_R"),
        ];
        (skel, map.iter().map(|(b, n)| (*b, (*n).to_owned())).collect())
    }

    fn world_of(skel: &Skeleton, pose: &[PoseBone]) -> Vec<Mat4> {
        let locals: Vec<Mat4> = pose
            .iter()
            .map(|p| Mat4::from_scale_rotation_translation(Vec3::from(p.scale), Quat::from_array(p.rotation), Vec3::from(p.translation)))
            .collect();
        skel.world_matrices(&locals)
    }

    #[test]
    fn arms_legs_and_spine_land_exactly_on_the_canonical_axes() {
        let (skel, human) = a_pose();
        let pose = biped_pose(&skel, &human);
        assert_eq!(pose.len(), skel.joints.len());
        let w = world_of(&skel, &pose);
        let dir = |a: usize, b: usize| (world_pos(&w[b]) - world_pos(&w[a])).normalize();
        assert!((dir(4, 5) - Vec3::X).length() < 1e-4, "left upper arm {:?}", dir(4, 5));
        assert!((dir(5, 6) - Vec3::X).length() < 1e-4);
        assert!((dir(7, 8) - Vec3::NEG_X).length() < 1e-4);
        assert!((dir(10, 11) - Vec3::NEG_Y).length() < 1e-4);
        assert!((dir(11, 12) - Vec3::NEG_Y).length() < 1e-4);
        assert!((dir(0, 1) - Vec3::Y).length() < 1e-4, "hips to spine");
        assert!((dir(2, 3) - Vec3::Y).length() < 1e-4, "neck to head");
        // Translations and scales are untouched.
        assert!(pose.iter().zip(&skel.joints).all(|(p, j)| Vec3::from(p.translation) == j.rest_translation && Vec3::from(p.scale) == j.rest_scale));
    }

    #[test]
    fn the_pose_is_stable_and_needs_a_mapped_hips() {
        let (skel, human) = a_pose();
        let once = biped_pose(&skel, &human);
        // Running it on a skeleton already in that pose changes nothing.
        let mut posed = skel.clone();
        for (j, p) in posed.joints.iter_mut().zip(&once) {
            j.rest_rotation = Quat::from_array(p.rotation);
        }
        let twice = biped_pose(&posed, &human);
        for (a, b) in once.iter().zip(&twice) {
            assert!(Quat::from_array(a.rotation).dot(Quat::from_array(b.rotation)).abs() > 0.99999, "{}", a.name);
        }
        let no_hips: Vec<_> = human.iter().filter(|(b, _)| *b != HumanBone::Hips).cloned().collect();
        assert!(biped_pose(&skel, &no_hips).is_empty());
    }
}
