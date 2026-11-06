//! Reference T-pose for humanoid rigs.
//!
//! A rig's rest pose is whatever the artist exported (often an A-pose), so
//! retargeting and tracker calibration need a canonical pose to measure
//! against. [`compute_reference_pose`] picks the closer of the skeleton's bind
//! pose ([`sample_bind_pose`]) and its file pose, then runs [`make_pose_valid`]:
//! a per-bone direction table (global or parent-relative, with a tolerance cone
//! each) that rotates every out-of-tolerance bone by the smallest turn that
//! lands just inside its cone. The result is stored in `RigImport::reference_pose`
//! and read back with [`reference_locals`].
//!
//! The canonical frame faces +Z with the avatar's left on +X and up on +Y.
//! Only rotations are corrected; local translations and scales stay as authored.

use crate::human_rig::{ALL, HUMAN_BONE_COUNT, HumanBone, PoseBone};
use crate::Skeleton;
use glam::{Mat3, Mat4, Quat, Vec3};

/// Hips tolerance for the avatar orientation, degrees.
const HIPS_TOLERANCE_DEG: f32 = 15.0;

/// A pose as per-joint local transforms.
#[derive(Clone, Debug)]
pub struct LocalPose {
    pub translation: Vec<Vec3>,
    pub rotation: Vec<Quat>,
    pub scale: Vec<Vec3>,
}

impl LocalPose {
    /// The skeleton's file (rest) pose.
    pub fn rest(skel: &Skeleton) -> Self {
        Self {
            translation: skel.joints.iter().map(|j| j.rest_translation).collect(),
            rotation: skel.joints.iter().map(|j| j.rest_rotation).collect(),
            scale: skel.joints.iter().map(|j| j.rest_scale).collect(),
        }
    }

    fn locals(&self) -> Vec<Mat4> {
        (0..self.rotation.len())
            .map(|i| Mat4::from_scale_rotation_translation(self.scale[i], self.rotation[i], self.translation[i]))
            .collect()
    }
}

/// The skeleton's local transforms in its reference pose: the stored reference
/// rotations over the authored translations and scales, or the plain rest pose
/// when the rig carries no reference pose.
pub fn reference_locals(skel: &Skeleton) -> Vec<Mat4> {
    let mut out = skel.rest_locals();
    let Some(rig) = skel.rig.as_ref().filter(|r| !r.reference_pose.is_empty()) else { return out };
    let mut apply = |i: usize, p: &PoseBone| {
        let (s, _, t) = out[i].to_scale_rotation_translation();
        let r = Quat::from_array(p.rotation);
        if r.length_squared() > 1e-6 {
            out[i] = Mat4::from_scale_rotation_translation(s, r.normalize(), t);
        }
    };
    if rig.reference_pose.len() == skel.joints.len() && rig.reference_pose.iter().zip(&skel.joints).all(|(p, j)| p.name == j.name) {
        for (i, p) in rig.reference_pose.iter().enumerate() {
            apply(i, p);
        }
    } else {
        let index: std::collections::HashMap<&str, usize> = skel.joints.iter().enumerate().map(|(i, j)| (j.name.as_str(), i)).collect();
        for p in &rig.reference_pose {
            if let Some(&i) = index.get(p.name.as_str()) {
                apply(i, p);
            }
        }
    }
    out
}

fn world_rot(m: &Mat4) -> Quat {
    m.to_scale_rotation_translation().1
}

fn pos(m: &Mat4) -> Vec3 {
    m.w_axis.truncate()
}

/// The pose the skin was bound in: joint rotations from the inverse bind
/// matrices over the file's local translations and scales. `None` when the
/// skeleton has no bind data (every inverse bind is identity).
pub fn sample_bind_pose(skel: &Skeleton) -> Option<LocalPose> {
    if skel.joints.iter().all(|j| j.inverse_bind == Mat4::IDENTITY) {
        return None;
    }
    let world: Vec<Quat> = skel
        .joints
        .iter()
        .map(|j| {
            let m = j.inverse_bind.inverse();
            let (x, y, z) = (m.x_axis.truncate(), m.y_axis.truncate(), m.z_axis.truncate());
            // A mirrored basis (negative determinant) flips Z and Y so the result stays a rotation.
            let flip = if x.cross(y).dot(z) < 0.0 { -1.0 } else { 1.0 };
            let (z, y) = ((z * flip).normalize_or_zero(), (y * flip).normalize_or_zero());
            let x = y.cross(z).normalize_or_zero();
            if x == Vec3::ZERO || y == Vec3::ZERO || z == Vec3::ZERO {
                return j.rest_rotation;
            }
            Quat::from_mat3(&Mat3::from_cols(x, z.cross(x), z)).normalize()
        })
        .collect();
    let mut pose = LocalPose::rest(skel);
    for (i, j) in skel.joints.iter().enumerate() {
        pose.rotation[i] = match j.parent {
            Some(p) => (world[p].inverse() * world[i]).normalize(),
            None => world[i],
        };
    }
    Some(pose)
}

/// Target direction and tolerance of one bone (canonical frame).
struct Row {
    dir: Vec3,
    max: f32,
    /// Measured relative to where the parent's direction actually points.
    relative: bool,
}

fn row(b: HumanBone) -> Option<Row> {
    use HumanBone::*;
    let n = b.name();
    let side = if n.starts_with("Left") {
        1.0
    } else if n.starts_with("Right") {
        -1.0
    } else {
        0.0
    };
    // `x` points outward from the body: toward +X on the left, -X on the right.
    let d = |x: f32, y: f32, z: f32, deg: f32, relative: bool| Some(Row { dir: Vec3::new(side * x, y, z).normalize(), max: deg.to_radians(), relative });
    match b {
        Hips | Head | LeftToes | RightToes | LeftEye | RightEye | Jaw => None,
        LeftUpperLeg | RightUpperLeg => d(0.05, -1.0, 0.0, 15.0, false),
        LeftLowerLeg | RightLowerLeg => d(0.05, -1.0, -0.15, 20.0, false),
        LeftFoot | RightFoot => d(0.05, 0.0, 1.0, 20.0, false),
        Spine | Chest | UpperChest | Neck => d(0.0, 1.0, 0.0, 30.0, false),
        LeftShoulder | RightShoulder => d(1.0, 0.0, 0.0, 20.0, false),
        LeftUpperArm | RightUpperArm | LeftLowerArm | RightLowerArm => d(1.0, 0.0, 0.0, 5.0, false),
        LeftHand | RightHand => d(1.0, 0.0, 0.0, 10.0, true),
        _ => {
            // Fingers: 10 degrees for the proximal phalange, 5 for the rest.
            let deg = if n.ends_with("Proximal") { 10.0 } else { 5.0 };
            if n.contains("Thumb") { d(1.0, 0.0, 1.0, deg, true) } else { d(1.0, 0.0, 0.0, deg, true) }
        }
    }
}

struct Ctx<'a> {
    skel: &'a Skeleton,
    joint: [Option<usize>; HUMAN_BONE_COUNT],
    only_child: Vec<Option<usize>>,
}

/// How far a bone points from its target direction.
struct Eval {
    actual: Vec3,
    target: Vec3,
    angle: f32,
    max: f32,
}

impl<'a> Ctx<'a> {
    fn new(skel: &'a Skeleton, human: &[(HumanBone, String)]) -> Self {
        let mut joint = [None; HUMAN_BONE_COUNT];
        for (b, name) in human {
            joint[*b as usize] = skel.joints.iter().position(|j| &j.name == name);
        }
        let mut kids = vec![Vec::new(); skel.joints.len()];
        for (i, j) in skel.joints.iter().enumerate() {
            if let Some(p) = j.parent {
                kids[p].push(i);
            }
        }
        let only_child = kids.iter().map(|k| (k.len() == 1).then(|| k[0])).collect();
        Self { skel, joint, only_child }
    }

    fn j(&self, b: HumanBone) -> Option<usize> {
        self.joint[b as usize]
    }

    fn mapped(&self, b: HumanBone) -> bool {
        self.joint[b as usize].is_some()
    }

    /// The joint a bone's direction points at: a fixed list for the spine
    /// chain and hands, else the first mapped human child, else the only child transform.
    fn child_joint(&self, b: HumanBone) -> Option<usize> {
        use HumanBone::*;
        let first: &[HumanBone] = match b {
            Spine => &[Chest, UpperChest, Neck, Head],
            Chest => &[UpperChest, Neck, Head],
            UpperChest => &[Neck, Head],
            Neck => &[Head],
            LeftHand => &[LeftMiddleProximal],
            RightHand => &[RightMiddleProximal],
            _ => &[],
        };
        if let Some(j) = first.iter().find_map(|c| self.j(*c)) {
            return Some(j);
        }
        let human_child = ALL.iter().find(|c| self.mapped(**c) && c.mapped_ancestor(|x| self.mapped(x)) == Some(b));
        human_child.and_then(|c| self.j(*c)).or_else(|| self.only_child[self.j(b)?])
    }

    fn world(&self, pose: &LocalPose) -> Vec<Mat4> {
        self.skel.world_matrices(&pose.locals())
    }

    fn actual(&self, b: HumanBone, world: &[Mat4]) -> Option<Vec3> {
        let (j, c) = (self.j(b)?, self.child_joint(b)?);
        let d = pos(&world[c]) - pos(&world[j]);
        (d.length_squared() > 1e-10).then(|| d.normalize())
    }

    fn eval(&self, b: HumanBone, world: &[Mat4]) -> Option<Eval> {
        let r = row(b)?;
        let actual = self.actual(b, world)?;
        let mut target = r.dir;
        if r.relative
            && let Some(p) = b.mapped_ancestor(|x| self.mapped(x))
            && let (Some(pr), Some(pa)) = (row(p), self.actual(p, world))
        {
            target = Quat::from_rotation_arc(pr.dir, pa) * r.dir;
        }
        Some(Eval { actual, target, angle: actual.angle_between(target), max: r.max })
    }

    /// The avatar's orientation relative to the canonical frame, from the leg
    /// and arm lines. `None` when either upper leg or upper arm is missing.
    fn orientation(&self, world: &[Mat4]) -> Option<Quat> {
        use HumanBone::*;
        let p = |b: HumanBone| self.j(b).map(|j| pos(&world[j]));
        let (lul, rul, lua, rua) = (p(LeftUpperLeg)?, p(RightUpperLeg)?, p(LeftUpperArm)?, p(RightUpperArm)?);
        let right = ((rul - lul).normalize_or_zero() + (rua - lua).normalize_or_zero()).normalize_or_zero();
        let mut up = ((lua + rua) * 0.5 - (lul + rul) * 0.5).normalize_or_zero();
        if right == Vec3::ZERO || up == Vec3::ZERO {
            return None;
        }
        // A rig already lined up with an axis gets an exactly axial up.
        if (right.x * right.y).abs() < 0.05 && (right.y * right.z).abs() < 0.05 && (right.x * right.z).abs() < 0.05 {
            let a = up.abs();
            let axis = if a.x >= a.y && a.x >= a.z { Vec3::X } else if a.y >= a.z { Vec3::Y } else { Vec3::Z };
            up = axis * up.dot(axis).signum();
        }
        let fwd = (-right).cross(up).normalize_or_zero();
        if fwd == Vec3::ZERO {
            return None;
        }
        let left = up.cross(fwd);
        Some(Quat::from_mat3(&Mat3::from_cols(left, up, fwd)).normalize())
    }

    /// Set joint `j`'s world rotation to `q`, keeping its local translation.
    fn set_world_rot(&self, pose: &mut LocalPose, world: &[Mat4], j: usize, q: Quat) {
        pose.rotation[j] = match self.skel.joints[j].parent {
            Some(p) => (world_rot(&world[p]).inverse() * q).normalize(),
            None => q.normalize(),
        };
    }

    /// Sum of tolerance overshoots (degrees) over the table and the hips orientation.
    fn error(&self, pose: &LocalPose) -> f32 {
        let world = self.world(pose);
        let mut e = ALL.iter().filter_map(|b| self.eval(*b, &world)).map(|v| (v.angle - v.max).max(0.0).to_degrees()).sum::<f32>();
        if let Some(o) = self.orientation(&world) {
            e += (o.angle_between(Quat::IDENTITY).to_degrees() - HIPS_TOLERANCE_DEG).max(0.0);
        }
        e
    }

    fn make_valid(&self, pose: &mut LocalPose) {
        use HumanBone::*;
        for b in ALL {
            let Some(j) = self.j(b) else { continue };
            let world = self.world(pose);
            if b == Hips {
                if let Some(o) = self.orientation(&world) {
                    let angle = o.angle_between(Quat::IDENTITY);
                    let max = HIPS_TOLERANCE_DEG.to_radians();
                    if angle > 0.99 * max {
                        let t = (1.05 - max / angle).clamp(0.0, 1.0);
                        let corr = Quat::IDENTITY.slerp(o.inverse(), t);
                        self.set_world_rot(pose, &world, j, corr * world_rot(&world[j]));
                    }
                }
                continue;
            }
            let Some(ev) = self.eval(b, &world) else { continue };
            if ev.angle <= 0.99 * ev.max {
                continue;
            }
            let t = (1.05 - ev.max / ev.angle).clamp(0.0, 1.0);
            let corr = Quat::IDENTITY.slerp(Quat::from_rotation_arc(ev.actual, ev.target), t);
            // Legs turn without dragging the foot along.
            let foot = match b {
                LeftUpperLeg | LeftLowerLeg => self.j(LeftFoot),
                RightUpperLeg | RightLowerLeg => self.j(RightFoot),
                _ => None,
            };
            let foot_rot = foot.map(|f| world_rot(&world[f]));
            self.set_world_rot(pose, &world, j, corr * world_rot(&world[j]));
            if let (Some(f), Some(q)) = (foot, foot_rot) {
                let w = self.world(pose);
                self.set_world_rot(pose, &w, f, q);
            }
        }
    }
}

/// Rotate `pose`'s bones until each human bone points within its tolerance
/// cone of the T-pose direction table. Bones missing from `human` are skipped.
pub fn make_pose_valid(skel: &Skeleton, human: &[(HumanBone, String)], pose: &mut LocalPose) {
    Ctx::new(skel, human).make_valid(pose);
}

/// How far `pose` is from a valid T-pose, in degrees summed over the bones
/// that overshoot their tolerance (0 = valid).
pub fn pose_error(skel: &Skeleton, human: &[(HumanBone, String)], pose: &LocalPose) -> f32 {
    Ctx::new(skel, human).error(pose)
}

/// The mapped bones of `human` whose direction in `pose` is outside their
/// T-pose tolerance (the "Character is not in T pose" test), in bone order.
pub fn bones_out_of_pose(skel: &Skeleton, human: &[(HumanBone, String)], pose: &LocalPose) -> Vec<HumanBone> {
    let ctx = Ctx::new(skel, human);
    let world = ctx.world(pose);
    ALL.iter().copied().filter(|b| ctx.eval(*b, &world).is_some_and(|v| v.angle > v.max)).collect()
}

/// The reference T-pose of `skel` for the bone map `human`: the closer of the
/// bind pose and the file pose, made valid. Empty without a mapped Hips.
pub fn compute_reference_pose(skel: &Skeleton, human: &[(HumanBone, String)]) -> Vec<PoseBone> {
    let ctx = Ctx::new(skel, human);
    if !ctx.mapped(HumanBone::Hips) {
        return Vec::new();
    }
    let file = LocalPose::rest(skel);
    let mut pose = match sample_bind_pose(skel) {
        Some(bind) if ctx.error(&bind) < ctx.error(&file) => bind,
        _ => file,
    };
    ctx.make_valid(&mut pose);
    skel.joints
        .iter()
        .enumerate()
        .map(|(i, j)| PoseBone {
            name: j.name.clone(),
            translation: pose.translation[i].to_array(),
            rotation: pose.rotation[i].to_array(),
            scale: pose.scale[i].to_array(),
        })
        .collect()
}

/// [`compute_reference_pose`], or with `force` the exact canonical T-pose of
/// [`crate::rig_biped::biped_pose`] (bone lengths and unmapped bones kept).
pub fn compute_reference_pose_with(skel: &Skeleton, human: &[(HumanBone, String)], force: bool) -> Vec<PoseBone> {
    if force { crate::rig_biped::biped_pose(skel, human) } else { compute_reference_pose(skel, human) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Joint;

    fn joint(name: &str, parent: Option<usize>, t: Vec3) -> Joint {
        Joint { name: name.into(), parent, inverse_bind: Mat4::IDENTITY, rest_translation: t, rest_rotation: Quat::IDENTITY, rest_scale: Vec3::ONE }
    }

    /// A humanoid whose arms hang `drop_deg` below horizontal.
    fn rig(drop_deg: f32) -> (Skeleton, Vec<(HumanBone, String)>) {
        let (c, s) = (drop_deg.to_radians().cos(), drop_deg.to_radians().sin());
        let mut joints = vec![
            joint("Hips", None, Vec3::new(0.0, 1.0, 0.0)),
            joint("Spine", Some(0), Vec3::new(0.0, 0.15, 0.0)),
            joint("Chest", Some(1), Vec3::new(0.0, 0.2, 0.0)),
            joint("Head", Some(2), Vec3::new(0.0, 0.35, 0.0)),
        ];
        let mut human = vec![(HumanBone::Hips, "Hips".to_string()), (HumanBone::Spine, "Spine".into()), (HumanBone::Chest, "Chest".into()), (HumanBone::Head, "Head".into())];
        for (side, x) in [("Left", 1.0f32), ("Right", -1.0)] {
            let base = joints.len();
            joints.push(joint(&format!("{side}Shoulder"), Some(2), Vec3::new(0.05 * x, 0.15, 0.0)));
            joints.push(joint(&format!("{side}UpperArm"), Some(base), Vec3::new(0.1 * x, 0.0, 0.0)));
            joints.push(joint(&format!("{side}LowerArm"), Some(base + 1), Vec3::new(0.25 * c * x, -0.25 * s, 0.0)));
            joints.push(joint(&format!("{side}Hand"), Some(base + 2), Vec3::new(0.25 * c * x, -0.25 * s, 0.0)));
            joints.push(joint(&format!("{side}UpperLeg"), Some(0), Vec3::new(0.1 * x, -0.05, 0.0)));
            joints.push(joint(&format!("{side}LowerLeg"), Some(base + 4), Vec3::new(0.0, -0.45, 0.0)));
            joints.push(joint(&format!("{side}Foot"), Some(base + 5), Vec3::new(0.0, -0.42, 0.0)));
            for (k, n) in ["Shoulder", "UpperArm", "LowerArm", "Hand", "UpperLeg", "LowerLeg", "Foot"].into_iter().enumerate() {
                human.push((HumanBone::from_name(&format!("{side}{n}")).unwrap(), joints[base + k].name.clone()));
            }
        }
        (Skeleton { joints, rig: None, ..Default::default() }, human)
    }

    fn arm_angle(skel: &Skeleton, world: &[Mat4], upper: &str, lower: &str, out: f32) -> f32 {
        let at = |n: &str| pos(&world[skel.joints.iter().position(|j| j.name == n).unwrap()]);
        (at(lower) - at(upper)).normalize().angle_between(Vec3::X * out).to_degrees()
    }

    #[test]
    fn a_pose_is_pulled_into_the_t_pose_cone_and_a_t_pose_is_left_alone() {
        let (a, human) = rig(45.0);
        let mut pose = LocalPose::rest(&a);
        assert!(pose_error(&a, &human, &pose) > 30.0, "the A-pose breaks the 5 degree arm limit");
        make_pose_valid(&a, &human, &mut pose);
        assert!(pose_error(&a, &human, &pose) < 1e-3, "valid after the fix");
        let world = a.world_matrices(&pose.locals());
        for (side, out) in [("Left", 1.0), ("Right", -1.0)] {
            let (u, l, h) = (format!("{side}UpperArm"), format!("{side}LowerArm"), format!("{side}Hand"));
            assert!(arm_angle(&a, &world, &u, &l, out) <= 5.0 + 1e-3);
            assert!(arm_angle(&a, &world, &l, &h, out) <= 5.0 + 1e-3);
        }

        let (t, human) = rig(0.0);
        let mut pose = LocalPose::rest(&t);
        make_pose_valid(&t, &human, &mut pose);
        for (r, j) in pose.rotation.iter().zip(&t.joints) {
            assert!(r.angle_between(j.rest_rotation) < 1e-4, "{} moved", j.name);
        }
    }

    #[test]
    fn reference_pose_round_trips_through_reference_locals() {
        let (mut a, human) = rig(45.0);
        assert!(reference_locals(&a).iter().zip(a.rest_locals()).all(|(x, y)| x.abs_diff_eq(y, 1e-6)), "no reference pose = rest");
        let pose = compute_reference_pose(&a, &human);
        assert_eq!(pose.len(), a.joints.len());
        a.rig = Some(crate::human_rig::RigImport { reference_pose: pose, human, ..Default::default() });
        let world = a.world_matrices(&reference_locals(&a));
        let ang = arm_angle(&a, &world, "LeftUpperArm", "LeftLowerArm", 1.0);
        assert!(ang <= 5.0 + 1e-3, "angle {ang}");
    }

    #[test]
    fn the_bind_pose_wins_when_it_is_the_closer_one() {
        // File pose: arms rotated down 45 degrees. Bind pose (from the inverse binds): a T-pose.
        let (mut skel, human) = rig(0.0);
        let world = skel.world_matrices(&skel.rest_locals());
        for (j, w) in skel.joints.iter_mut().zip(&world) {
            j.inverse_bind = w.inverse();
            if j.name == "LeftUpperArm" || j.name == "RightUpperArm" {
                let sign = if j.name.starts_with("Left") { -1.0 } else { 1.0 };
                j.rest_rotation = Quat::from_rotation_z(sign * 45f32.to_radians());
            }
        }
        let file = LocalPose::rest(&skel);
        let bind = sample_bind_pose(&skel).unwrap();
        assert!(pose_error(&skel, &human, &bind) < pose_error(&skel, &human, &file));
        let pose = compute_reference_pose(&skel, &human);
        let i = skel.joints.iter().position(|j| j.name == "LeftUpperArm").unwrap();
        // The bind-pose rotation is already a T-pose arm: no correction on top.
        assert!(Quat::from_array(pose[i].rotation).angle_between(Quat::IDENTITY) < 1e-3);
    }
}
