//! Humanoid pose mirroring for blend-tree children with Mirror set.
//!
//! The pose is mirrored across the sagittal plane (rig-space X = 0, the
//! reference frame faces +Z with the avatar's left on +X). Per node: take the
//! world-space rotation delta from the reference pose, reflect it across the
//! plane, and apply it on top of the *partner* node's reference rotation
//! (Left bone <-> Right bone, centre bones map to themselves). Local
//! rotations are rebuilt from the new world rotations; animated translations
//! are reflected as world vectors. Nodes outside the human map pair by a
//! left/right token in their name when the swapped name exists, else map to
//! themselves. Rigs without a humanoid map have no table and are left alone.

use std::collections::HashMap;

use glam::{Quat, Vec3};
use zestengine_assets::auto_map::mirror_bone;
use zestengine_assets::human_rig::ALL;

use crate::anim_graph::{Pose, Rig};

/// Per-rig mirroring data, built once with the rig.
#[derive(Clone, Debug)]
pub struct RigMirror {
    /// Mirror partner of each node (itself for centre nodes).
    partner: Vec<usize>,
    /// World rotation of each node in the reference pose.
    ref_world: Vec<Quat>,
    /// Parents before children.
    order: Vec<usize>,
}

/// Reflect a vector across the sagittal plane.
pub fn mirror_vec(v: Vec3) -> Vec3 {
    Vec3::new(-v.x, v.y, v.z)
}

/// Reflect a rotation across the sagittal plane (`M R M` for `M = diag(-1, 1, 1)`).
pub fn mirror_quat(q: Quat) -> Quat {
    Quat::from_xyzw(q.x, -q.y, -q.z, q.w)
}

/// Swap the side token of a node name (`LeftHand` <-> `RightHand`, `arm.L` <-> `arm.R`, `hand_l` <-> `hand_r`).
fn swap_side(name: &str) -> Option<String> {
    const WORDS: [(&str, &str); 3] = [("Left", "Right"), ("left", "right"), ("LEFT", "RIGHT")];
    for (l, r) in WORDS {
        if let Some(i) = name.find(l) {
            return Some(format!("{}{}{}", &name[..i], r, &name[i + l.len()..]));
        }
        if let Some(i) = name.find(r) {
            return Some(format!("{}{}{}", &name[..i], l, &name[i + r.len()..]));
        }
    }
    // Single-letter suffix or prefix with a separator.
    let b = name.as_bytes();
    if b.len() >= 3 {
        let (sep, last) = (b[b.len() - 2], b[b.len() - 1]);
        if matches!(sep, b'.' | b'_' | b'-' | b' ') && let Some(o) = flip(last) {
            return Some(format!("{}{}", &name[..name.len() - 1], o as char));
        }
        let (first, sep) = (b[0], b[1]);
        if matches!(sep, b'.' | b'_' | b'-' | b' ') && let Some(o) = flip(first) {
            return Some(format!("{}{}", o as char, &name[1..]));
        }
    }
    None
}

fn flip(c: u8) -> Option<u8> {
    match c {
        b'L' => Some(b'R'),
        b'R' => Some(b'L'),
        b'l' => Some(b'r'),
        b'r' => Some(b'l'),
        _ => None,
    }
}

impl RigMirror {
    /// Build for `rig` (its `human` table must be set) given each node's
    /// world rotation in the reference pose. `None` for non-humanoid rigs.
    pub fn build(rig: &Rig, ref_world: Vec<Quat>) -> Option<Self> {
        let n = rig.len();
        if rig.human.is_empty() || ref_world.len() != n {
            return None;
        }
        let mut partner: Vec<usize> = (0..n).collect();
        let mut human_node = vec![false; n];
        for b in ALL {
            if let Some(i) = rig.human.get(b as usize).copied().flatten() {
                human_node[i] = true;
                // Only Left bones have a distinct mirror; pair both ways.
                let m = mirror_bone(b);
                if m != b && let Some(j) = rig.human.get(m as usize).copied().flatten() {
                    partner[i] = j;
                    partner[j] = i;
                }
            }
        }
        let by_name: HashMap<&str, usize> = rig.names.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
        for i in 0..n {
            if human_node[i] {
                continue;
            }
            if let Some(j) = swap_side(&rig.names[i]).and_then(|s| by_name.get(s.as_str()).copied())
                && !human_node[j]
            {
                partner[i] = j;
            }
        }
        // Keep only symmetric pairs.
        for i in 0..n {
            if partner[partner[i]] != i {
                partner[i] = i;
            }
        }
        // Parents before children, whatever the joint order.
        let mut order = Vec::with_capacity(n);
        let mut done = vec![false; n];
        fn visit(i: usize, parents: &[Option<usize>], done: &mut [bool], order: &mut Vec<usize>) {
            if done[i] {
                return;
            }
            done[i] = true;
            if let Some(p) = parents[i] {
                visit(p, parents, done, order);
            }
            order.push(i);
        }
        for i in 0..n {
            visit(i, &rig.parents, &mut done, &mut order);
        }
        Some(Self { partner, ref_world, order })
    }

    /// Mirror partner of node `i`.
    pub fn partner(&self, i: usize) -> usize {
        self.partner[i]
    }

    /// Mirror `pose` in place.
    pub fn mirror_pose(&self, rig: &Rig, pose: &mut Pose) {
        let n = self.partner.len().min(pose.trs.len());
        if n != self.partner.len() {
            return;
        }
        // World rotation of the source pose and its delta from the reference.
        let mut world = vec![Quat::IDENTITY; n];
        for &i in &self.order {
            let p = rig.parents[i].map(|p| world[p]).unwrap_or(Quat::IDENTITY);
            world[i] = (p * pose.trs[i].r).normalize();
        }
        let delta: Vec<Quat> = (0..n).map(|i| world[i] * self.ref_world[i].inverse()).collect();
        let src = pose.clone();
        let mut new_world = vec![Quat::IDENTITY; n];
        for &i in &self.order {
            let from = self.partner[i];
            new_world[i] = (mirror_quat(delta[from]) * self.ref_world[i]).normalize();
            let parent = rig.parents[i];
            let pw = parent.map(|p| new_world[p]).unwrap_or(Quat::IDENTITY);
            pose.trs[i].r = (pw.inverse() * new_world[i]).normalize();
            pose.trs[i].s = src.trs[from].s;
            pose.touched[i] = src.touched[from];
            pose.scalar[i] = src.scalar[from];
            // Animated translation: reflect the world vector, bring it into the new parent frame.
            if src.touched[from] & crate::anim_graph::CH_T != 0 {
                let src_parent_world = rig.parents[from].map(|p| world[p]).unwrap_or(Quat::IDENTITY);
                let world_t = src_parent_world * src.trs[from].t;
                pose.trs[i].t = pw.inverse() * mirror_vec(world_t);
            } else {
                pose.trs[i].t = src.trs[i].t;
            }
        }
    }
}

/// Whether a missing-table warning has been logged.
static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Log (once) that a Mirror child was ignored because the rig is not humanoid.
pub fn warn_not_humanoid() {
    if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!("blend tree child has Mirror set, but this rig has no humanoid avatar: mirroring is ignored (set the model's Rig to Humanoid)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anim_graph::{CH_R, CH_T, Trs};

    /// hips -> spine -> head, hips -> L/R upper arm. Arms stick out along +-X in the reference.
    pub(crate) fn symmetric() -> (Rig, Vec<Quat>) {
        let names = ["hips", "spine", "head", "arm_L", "arm_R", "prop_L", "prop_R"].map(String::from).to_vec();
        let parents = vec![None, Some(0), Some(1), Some(1), Some(1), Some(3), Some(4)];
        let rest = vec![
            Trs { t: Vec3::new(0.0, 1.0, 0.0), ..Default::default() },
            Trs { t: Vec3::new(0.0, 0.2, 0.0), ..Default::default() },
            Trs { t: Vec3::new(0.0, 0.3, 0.0), ..Default::default() },
            Trs { t: Vec3::new(0.1, 0.0, 0.0), ..Default::default() },
            Trs { t: Vec3::new(-0.1, 0.0, 0.0), ..Default::default() },
            Trs { t: Vec3::new(0.3, 0.0, 0.0), ..Default::default() },
            Trs { t: Vec3::new(-0.3, 0.0, 0.0), ..Default::default() },
        ];
        let mut rig = Rig::new(names, parents, rest, vec![0.0; 7], true);
        use zestengine_assets::human_rig::HumanBone;
        let mut human = vec![None; zestengine_assets::human_rig::HUMAN_BONE_COUNT];
        human[HumanBone::Hips as usize] = Some(0);
        human[HumanBone::Spine as usize] = Some(1);
        human[HumanBone::Head as usize] = Some(2);
        human[HumanBone::LeftUpperArm as usize] = Some(3);
        human[HumanBone::RightUpperArm as usize] = Some(4);
        rig.human = human;
        // Reference: all-identity world rotations.
        let m = RigMirror::build(&rig, vec![Quat::IDENTITY; 7]).unwrap();
        rig.mirror = Some(std::sync::Arc::new(m));
        (rig, vec![Quat::IDENTITY; 7])
    }

    #[test]
    fn swaps_sides_and_reflects_rotation() {
        let (rig, _) = symmetric();
        let m = rig.mirror.clone().unwrap();
        assert_eq!(m.partner(3), 4);
        assert_eq!(m.partner(4), 3);
        assert_eq!(m.partner(5), 6); // prop_L <-> prop_R by name
        assert_eq!(m.partner(1), 1);
        let mut pose = Pose::rest(&rig);
        // Left arm swings forward (+Z) around Y: rotate 40 deg around Y.
        pose.trs[3].r = Quat::from_rotation_y(0.7);
        pose.touched[3] |= CH_R;
        m.mirror_pose(&rig, &mut pose);
        // Right arm takes the reflected rotation (Y rotation reverses), left is at rest.
        assert!(pose.trs[4].r.angle_between(Quat::from_rotation_y(-0.7)) < 1e-4);
        assert!(pose.trs[3].r.angle_between(Quat::IDENTITY) < 1e-4);
        assert_ne!(pose.touched[4] & CH_R, 0);
        assert_eq!(pose.touched[3] & CH_R, 0);
    }

    #[test]
    fn mirroring_twice_restores_the_pose() {
        let (rig, _) = symmetric();
        let m = rig.mirror.clone().unwrap();
        let mut pose = Pose::rest(&rig);
        pose.trs[1].r = Quat::from_euler(glam::EulerRot::XYZ, 0.2, 0.3, -0.4);
        pose.trs[3].r = Quat::from_euler(glam::EulerRot::XYZ, 0.5, -0.1, 0.9);
        pose.trs[6].r = Quat::from_rotation_z(0.3);
        pose.trs[0].t = Vec3::new(0.2, 1.1, 0.4);
        pose.touched[0] |= CH_T;
        let orig = pose.clone();
        m.mirror_pose(&rig, &mut pose);
        m.mirror_pose(&rig, &mut pose);
        for i in 0..7 {
            assert!(pose.trs[i].r.angle_between(orig.trs[i].r) < 1e-3, "node {i}");
            assert!((pose.trs[i].t - orig.trs[i].t).length() < 1e-4, "node {i}");
        }
    }

    #[test]
    fn centre_bones_reflect_and_root_translation_flips_x() {
        let (rig, _) = symmetric();
        let m = rig.mirror.clone().unwrap();
        let mut pose = Pose::rest(&rig);
        // Lean to the avatar's side (roll around Z) and sway the hips.
        pose.trs[1].r = Quat::from_rotation_z(0.3);
        pose.trs[0].t = Vec3::new(0.2, 1.0, 0.1);
        pose.touched[0] |= CH_T;
        m.mirror_pose(&rig, &mut pose);
        assert!(pose.trs[1].r.angle_between(Quat::from_rotation_z(-0.3)) < 1e-4);
        assert!((pose.trs[0].t - Vec3::new(-0.2, 1.0, 0.1)).length() < 1e-5);
    }

    #[test]
    fn world_position_of_a_hand_reflects() {
        let (rig, _) = symmetric();
        let m = rig.mirror.clone().unwrap();
        let fk = |p: &Pose, i: usize| {
            let mut mat = glam::Mat4::IDENTITY;
            let mut chain = vec![i];
            while let Some(par) = rig.parents[*chain.last().unwrap()] {
                chain.push(par);
            }
            for &c in chain.iter().rev() {
                mat *= p.trs[c].to_mat4();
            }
            mat.w_axis.truncate()
        };
        let mut pose = Pose::rest(&rig);
        pose.trs[3].r = Quat::from_euler(glam::EulerRot::XYZ, 0.4, 0.8, 0.3);
        pose.trs[1].r = Quat::from_euler(glam::EulerRot::XYZ, 0.1, 0.2, 0.0);
        let before = pose.clone();
        m.mirror_pose(&rig, &mut pose);
        // The left prop's world position before == the right prop's, reflected, after. Spine pivots too, so compare reflected.
        let l = fk(&before, 5);
        let r = fk(&pose, 6);
        assert!((mirror_vec(l) - r).length() < 1e-4, "{l:?} {r:?}");
    }

    /// A mirrored blend-tree child swings the right arm and walks the root toward -X.
    #[test]
    fn mirrored_tree_child_mirrors_pose_and_root_motion() {
        use crate::anim_graph::{ClipSet, motion_root_delta, sample_motion};
        use zestengine_assets::{AnimChannel, AnimationClip, ChannelPath, Motion};
        use zestengine_core::{BlendMotion, BlendTree};
        let (rig, _) = symmetric();
        let clip = AnimationClip {
            name: "walk".into(),
            duration: 1.0,
            channels: vec![
                AnimChannel {
                    joint: 3,
                    path: ChannelPath::Rotation,
                    times: vec![0.0, 1.0],
                    vec_values: vec![],
                    quat_values: vec![Quat::IDENTITY, Quat::from_rotation_y(1.0)], extra: None
                },
                AnimChannel {
                    joint: 0,
                    path: ChannelPath::Translation,
                    times: vec![0.0, 1.0],
                    vec_values: vec![Vec3::new(0.0, 1.0, 0.0), Vec3::new(2.0, 1.0, 1.0)],
                    quat_values: vec![], extra: None
                },
            ],
            morph_tracks: Vec::new(), playback: None,
        };
        let imported = [clip];
        let idx = [0usize];
        let assets = HashMap::new();
        let set = ClipSet { imported: &imported, indices: &idx, assets: &assets };
        let tree = |mirror| {
            Motion::BlendTree(BlendTree {
                parameter: "P".into(),
                motions: vec![BlendMotion { clip: "walk".into(), mirror, ..Default::default() }],
                ..Default::default()
            })
        };
        let params = HashMap::new();
        let plain = sample_motion(&tree(false), &params, &set, &rig, 1.0, false, false);
        let mir = sample_motion(&tree(true), &params, &set, &rig, 1.0, false, false);
        assert!(plain.trs[3].r.angle_between(Quat::from_rotation_y(1.0)) < 1e-3);
        assert!(mir.trs[4].r.angle_between(Quat::from_rotation_y(-1.0)) < 1e-3);
        assert!(mir.trs[3].r.angle_between(Quat::IDENTITY) < 1e-3);
        let d0 = motion_root_delta(&rig, &tree(false), &params, &set, 0.0, 0.5, false);
        let d1 = motion_root_delta(&rig, &tree(true), &params, &set, 0.0, 0.5, false);
        assert!((d1 - Vec3::new(-d0.x, d0.y, d0.z)).length() < 1e-4 && d0.x > 0.5);
        // A non-humanoid rig leaves a mirrored child untouched.
        let flat = Rig::new(rig.names.clone(), rig.parents.clone(), rig.rest.clone(), vec![0.0; 7], true);
        let p = sample_motion(&tree(true), &params, &set, &flat, 1.0, false, false);
        assert!(p.trs[3].r.angle_between(Quat::from_rotation_y(1.0)) < 1e-3);
    }

    #[test]
    fn non_humanoid_has_no_table() {
        let rig = Rig::new(vec!["a".into()], vec![None], vec![Trs::default()], vec![0.0], true);
        assert!(RigMirror::build(&rig, vec![Quat::IDENTITY]).is_none());
        assert_eq!(swap_side("LeftHand").as_deref(), Some("RightHand"));
        assert_eq!(swap_side("hand.R").as_deref(), Some("hand.L"));
        assert_eq!(swap_side("L_Foot").as_deref(), Some("R_Foot"));
        assert_eq!(swap_side("Spine"), None);
    }
}
