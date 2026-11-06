//! Muscle space: a rig-independent description of a humanoid pose.
//!
//! A humanoid pose is 95 normalised muscle values in [-1, 1], a body root
//! (centre of mass and orientation, normalised by the avatar's human scale) and
//! optional limb goals. Any humanoid rig turns the same values into its own
//! joint rotations, so a clip authored once plays on avatars of any
//! proportions and any rest pose.
//!
//! # Muscle model
//!
//! Every muscle drives one angle of one bone: a twist about the bone's aim
//! axis (towards its child) or a swing about one of two axes perpendicular to
//! it. The three axes are built per rig from the reference pose
//! ([`crate::ref_pose`]): the aim comes from the child's position and the two
//! swing axes from canonical directions (up, forward, left; the reference frame
//! faces +Z with the avatar's left on +X), so left and right bones and rigs with
//! different bone orientations share one meaning. Positive values follow the
//! second word of the muscle name (`Down-Up`: up, `Front-Back`: back,
//! `Left-Right`: right, `In-Out`: out, `Stretch`: straightening, `Stretched`:
//! straightening, `Close`: closing). Twist is a right-handed turn about the
//! outward aim on the left side and the centre line, mirrored on the right (equal
//! values on the two sides are mirror images), except on the spine, neck and head
//! where positive turns to the avatar's right.
//!
//! A bone's rotation relative to its reference pose is `neutral * swing * twist`:
//! the *neutral* offset (muscle value 0, a relaxed human: elbows and knees bent,
//! fingers curled) followed by the swing about the parent-attached axes and the
//! twist about the bone. The angle of each axis is divided by `max` when
//! positive and by `|min|` when negative, so values are piecewise linear in
//! [-1, 1] with 0 at the neutral pose. The reference (T-)pose therefore reads
//! as non-zero (a straight forearm is stretch +1).
//!
//! # Body root
//!
//! The root is the mass-weighted centre of mass (default masses per bone, the
//! mass of an unmapped bone goes to its mapped ancestor) in model space divided
//! by the *human scale* (the height of the centre of mass in the reference
//! pose; the model origin is the ground), plus the body orientation (the Hips
//! rotation) relative to the reference. Placing the root for a target
//! rig multiplies by its human scale.
//!
//! # Retargeting
//!
//! [`MuscleRig::muscles_from_pose`] and [`MuscleRig::pose_from_muscles`] are
//! inverses within the limits (values are clamped to [-1, 1] on the way out and
//! in). Muscles alone give each avatar the same joint angles; limb goals then
//! align the endpoints by direction and length ratio: a hand is placed at the
//! source's direction from the shoulder times its fraction of the arm, a foot
//! near the ground at the source's scaled ground position (blended to the ratio
//! rule when airborne), and a two-bone solve bends the limb to reach it. A
//! grounded foot the legs cannot reach lowers the root by the shortfall.
//!
//! Optional bones that the target lacks fold into their parent
//! ([`MuscleRig::fold_missing`]): no Neck adds its muscles to Head, no
//! UpperChest adds half of its muscles to Chest, no Chest adds its muscles to
//! Spine, no Shoulder adds 0.15 of Down-Up and 0.28125 of Front-Back to the arm.
//!
//! Not modelled: translation DoF, the mid-DoF fix that moves knee/elbow
//! rotation without a muscle into the parent, and twist redistribution.

use std::sync::OnceLock;

use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::anim_clip::{AnimClip, AnimTrack, ClipKind, KeyInterp, TrackProperty};
use crate::human_rig::{HUMAN_BONE_COUNT, HumanBone, RigImport};
use crate::ref_pose::reference_locals;
use crate::{AnimationClip, Skeleton};

/// Number of muscles.
pub const MUSCLE_COUNT: usize = 95;

/// The three axes of a muscle bone: twist about the aim, swing Y, swing Z.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    Twist = 0,
    Y = 1,
    Z = 2,
}

/// Muscle groups of the Muscles & Settings tab.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MuscleGroup {
    Body,
    Head,
    LeftArm,
    LeftFingers,
    RightArm,
    RightFingers,
    LeftLeg,
    RightLeg,
}

impl MuscleGroup {
    pub const ALL: [MuscleGroup; 8] = [
        Self::Body,
        Self::Head,
        Self::LeftArm,
        Self::LeftFingers,
        Self::RightArm,
        Self::RightFingers,
        Self::LeftLeg,
        Self::RightLeg,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Body => "Body",
            Self::Head => "Head",
            Self::LeftArm => "Left Arm",
            Self::LeftFingers => "Left Fingers",
            Self::RightArm => "Right Arm",
            Self::RightFingers => "Right Fingers",
            Self::LeftLeg => "Left Leg",
            Self::RightLeg => "Right Leg",
        }
    }
}

/// Static facts about one muscle.
#[derive(Clone, Debug)]
pub struct MuscleInfo {
    pub name: String,
    pub bone: HumanBone,
    pub axis: Axis,
    pub group: MuscleGroup,
    /// Default limits in degrees about the neutral pose (`min <= 0 <= max`).
    pub min: f32,
    pub max: f32,
    /// Mirroring a pose negates this muscle when it has no partner (centre line, lateral axes).
    pub lateral: bool,
    /// The muscle on the other side (itself on the centre line).
    pub partner: usize,
}

// ----------------------------------------------------------------- the table

/// Semantic directions in the canonical frame (left-side / centre meaning;
/// right-side bones mirror them across X).
#[derive(Clone, Copy, Debug)]
enum Dir {
    Up,
    Down,
    Fwd,
    Back,
    /// The avatar's own right.
    Right,
    /// Away from the centre line.
    Out,
}

impl Dir {
    fn vec(self, mirrored: bool) -> Vec3 {
        let v = match self {
            Dir::Up => Vec3::Y,
            Dir::Down => -Vec3::Y,
            Dir::Fwd => Vec3::Z,
            Dir::Back => -Vec3::Z,
            Dir::Right => -Vec3::X,
            Dir::Out => Vec3::X,
        };
        if mirrored { Vec3::new(-v.x, v.y, v.z) } else { v }
    }
}

/// How a bone's aim axis is found.
#[derive(Clone, Debug)]
enum Aim {
    /// Towards the first mapped bone of the list.
    Child(Vec<HumanBone>),
    /// Continue the direction from the mapped parent.
    Extend,
    /// A fixed canonical direction (eyes, jaw).
    Fixed,
}

#[derive(Clone, Debug)]
struct Def {
    name: String,
    min: f32,
    max: f32,
}

#[derive(Clone, Debug)]
struct BoneSpec {
    bone: HumanBone,
    mirrored: bool,
    aim: Aim,
    /// Aim used when the geometry gives none.
    aim_default: Dir,
    y_dir: Dir,
    z_dir: Dir,
    /// +1, or -1 when positive twist turns to the avatar's right.
    twist_sign: f32,
    /// Neutral swing angles (Y, Z) in degrees.
    neutral: [f32; 2],
    /// Twist, Y and Z muscles.
    m: [Option<Def>; 3],
    group: MuscleGroup,
}

use HumanBone as B;

fn def(name: &str, min: f32, max: f32) -> Option<Def> {
    Some(Def { name: name.to_owned(), min, max })
}

#[allow(clippy::too_many_arguments)]
fn spec(
    bone: HumanBone,
    aim: Aim,
    aim_default: Dir,
    y_dir: Dir,
    z_dir: Dir,
    twist_sign: f32,
    neutral: [f32; 2],
    m: [Option<Def>; 3],
    group: MuscleGroup,
) -> BoneSpec {
    BoneSpec { bone, mirrored: false, aim, aim_default, y_dir, z_dir, twist_sign, neutral, m, group }
}

fn child(list: &[HumanBone]) -> Aim {
    Aim::Child(list.to_vec())
}

/// The right-hand twin of a left bone spec: same data, mirrored directions and names.
fn right_of(s: &BoneSpec, group: MuscleGroup) -> BoneSpec {
    let swap = |d: &Option<Def>| d.as_ref().map(|d| Def { name: d.name.replacen("Left", "Right", 1), min: d.min, max: d.max });
    BoneSpec {
        bone: crate::auto_map::mirror_bone(s.bone),
        mirrored: true,
        aim: match &s.aim {
            Aim::Child(list) => Aim::Child(list.iter().map(|b| crate::auto_map::mirror_bone(*b)).collect()),
            a => a.clone(),
        },
        m: [swap(&s.m[0]), swap(&s.m[1]), swap(&s.m[2])],
        group,
        ..s.clone()
    }
}

/// Ranges and offsets per finger: (name, stretched limits per phalanx, stretched
/// neutral per phalanx, spread limit, spread neutral).
type FingerSpec = (&'static str, [(f32, f32); 3], [f32; 3], f32, f32);
const FINGERS: [FingerSpec; 5] = [
    ("Thumb", [(-20.0, 20.0), (-40.0, 35.0), (-40.0, 35.0)], [-14.0, -23.0, -23.0], 25.0, 14.0),
    ("Index", [(-50.0, 50.0), (-45.0, 45.0), (-45.0, 45.0)], [-33.0, -36.5, -36.5], 20.0, 9.0),
    ("Middle", [(-50.0, 50.0), (-45.0, 45.0), (-45.0, 45.0)], [-33.0, -36.5, -36.5], 7.5, 5.0),
    ("Ring", [(-50.0, 50.0), (-45.0, 45.0), (-45.0, 45.0)], [-33.0, -36.5, -36.5], 7.5, -5.0),
    ("Little", [(-50.0, 50.0), (-45.0, 45.0), (-45.0, 45.0)], [-33.0, -36.5, -36.5], 20.0, -9.0),
];

/// Left finger bones per finger (proximal, intermediate, distal).
const LEFT_FINGER_BONES: [[HumanBone; 3]; 5] = [
    [B::LeftThumbProximal, B::LeftThumbIntermediate, B::LeftThumbDistal],
    [B::LeftIndexProximal, B::LeftIndexIntermediate, B::LeftIndexDistal],
    [B::LeftMiddleProximal, B::LeftMiddleIntermediate, B::LeftMiddleDistal],
    [B::LeftRingProximal, B::LeftRingIntermediate, B::LeftRingDistal],
    [B::LeftLittleProximal, B::LeftLittleIntermediate, B::LeftLittleDistal],
];

/// The specs of the centre line and the left side; the right side is derived in [`ordered_specs`].
fn base_specs() -> Vec<BoneSpec> {
    use MuscleGroup as G;
    let mut v = Vec::new();
    // Body: Front-Back (Z, back), Left-Right (Y, right), Twist.
    let chain: [(HumanBone, &'static [HumanBone], f32, &str); 3] = [
        (B::Spine, &[B::Chest, B::UpperChest, B::Neck, B::Head], 40.0, "Spine"),
        (B::Chest, &[B::UpperChest, B::Neck, B::Head], 40.0, "Chest"),
        (B::UpperChest, &[B::Neck, B::Head], 20.0, "UpperChest"),
    ];
    for (bone, kids, r, n) in chain {
        v.push(spec(
            bone, child(kids), Dir::Up, Dir::Right, Dir::Back, -1.0, [0.0; 2],
            [def(&format!("{n} Twist Left-Right"), -r, r), def(&format!("{n} Left-Right"), -r, r), def(&format!("{n} Front-Back"), -r, r)],
            G::Body,
        ));
    }
    // Head.
    v.push(spec(
        B::Neck, child(&[B::Head]), Dir::Up, Dir::Right, Dir::Back, -1.0, [0.0; 2],
        [def("Neck Turn Left-Right", -40.0, 40.0), def("Neck Tilt Left-Right", -40.0, 40.0), def("Neck Nod Down-Up", -40.0, 40.0)],
        G::Head,
    ));
    v.push(spec(
        B::Head, Aim::Extend, Dir::Up, Dir::Right, Dir::Back, -1.0, [0.0; 2],
        [def("Head Turn Left-Right", -40.0, 40.0), def("Head Tilt Left-Right", -40.0, 40.0), def("Head Nod Down-Up", -40.0, 40.0)],
        G::Head,
    ));
    v.push(spec(B::LeftEye, Aim::Fixed, Dir::Fwd, Dir::Out, Dir::Up, 1.0, [0.0; 2], [None, def("Left Eye In-Out", -20.0, 20.0), def("Left Eye Down-Up", -10.0, 15.0)], G::Head));
    v.push(spec(B::RightEye, Aim::Fixed, Dir::Fwd, Dir::Out, Dir::Up, 1.0, [0.0; 2], [None, def("Right Eye In-Out", -20.0, 20.0), def("Right Eye Down-Up", -10.0, 15.0)], G::Head));
    v.push(spec(B::Jaw, Aim::Fixed, Dir::Fwd, Dir::Right, Dir::Up, 1.0, [0.0, -10.0], [None, def("Jaw Left-Right", -10.0, 10.0), def("Jaw Close", -10.0, 10.0)], G::Head));
    // Left leg (the right twin is derived).
    v.push(spec(
        B::LeftUpperLeg, child(&[B::LeftLowerLeg]), Dir::Down, Dir::Out, Dir::Back, 1.0, [0.0, -30.0],
        [def("Left Upper Leg Twist In-Out", -60.0, 60.0), def("Left Upper Leg In-Out", -60.0, 60.0), def("Left Upper Leg Front-Back", -90.0, 50.0)],
        G::LeftLeg,
    ));
    v.push(spec(
        B::LeftLowerLeg, child(&[B::LeftFoot]), Dir::Down, Dir::Out, Dir::Fwd, 1.0, [0.0, -80.0],
        [def("Left Lower Leg Twist In-Out", -90.0, 90.0), None, def("Left Lower Leg Stretch", -80.0, 80.0)],
        G::LeftLeg,
    ));
    v.push(spec(
        B::LeftFoot, child(&[B::LeftToes]), Dir::Fwd, Dir::Out, Dir::Up, 1.0, [0.0; 2],
        [None, def("Left Foot Twist In-Out", -30.0, 30.0), def("Left Foot Up-Down", -50.0, 50.0)],
        G::LeftLeg,
    ));
    v.push(spec(B::LeftToes, Aim::Extend, Dir::Fwd, Dir::Out, Dir::Up, 1.0, [0.0; 2], [None, None, def("Left Toes Up-Down", -50.0, 50.0)], G::LeftLeg));
    // Left arm.
    v.push(spec(
        B::LeftShoulder, child(&[B::LeftUpperArm]), Dir::Out, Dir::Back, Dir::Up, 1.0, [0.0; 2],
        [None, def("Left Shoulder Front-Back", -15.0, 15.0), def("Left Shoulder Down-Up", -15.0, 30.0)],
        G::LeftArm,
    ));
    v.push(spec(
        B::LeftUpperArm, child(&[B::LeftLowerArm]), Dir::Out, Dir::Back, Dir::Up, 1.0, [-30.0, -40.0],
        [def("Left Arm Twist In-Out", -90.0, 90.0), def("Left Arm Front-Back", -100.0, 100.0), def("Left Arm Down-Up", -60.0, 100.0)],
        G::LeftArm,
    ));
    v.push(spec(
        B::LeftLowerArm, child(&[B::LeftHand]), Dir::Out, Dir::Up, Dir::Back, 1.0, [0.0, -80.0],
        [def("Left Forearm Twist In-Out", -90.0, 90.0), None, def("Left Forearm Stretch", -80.0, 80.0)],
        G::LeftArm,
    ));
    v.push(spec(
        B::LeftHand, Aim::Extend, Dir::Out, Dir::Back, Dir::Up, 1.0, [0.0; 2],
        [None, def("Left Hand In-Out", -40.0, 40.0), def("Left Hand Down-Up", -80.0, 80.0)],
        G::LeftArm,
    ));
    // Left fingers.
    for (f, (name, stretch, neutral, spread, spread_n)) in FINGERS.iter().enumerate() {
        for p in 0..3 {
            let bone = LEFT_FINGER_BONES[f][p];
            let aim = if p < 2 { child(&[LEFT_FINGER_BONES[f][p + 1]]) } else { Aim::Extend };
            let stretched = def(&format!("Left {name} {} Stretched", p + 1), stretch[p].0, stretch[p].1);
            let spread_def = (p == 0).then(|| Def { name: format!("Left {name} Spread"), min: -spread, max: *spread });
            v.push(spec(
                bone, aim, if f == 0 { Dir::Fwd } else { Dir::Out }, Dir::Fwd, Dir::Up, 1.0,
                [if p == 0 { *spread_n } else { 0.0 }, neutral[p]],
                [None, spread_def, stretched],
                G::LeftFingers,
            ));
        }
    }
    v
}

/// Every bone spec with its muscles, in muscle-index order.
fn ordered_specs() -> &'static Vec<BoneSpec> {
    static SPECS: OnceLock<Vec<BoneSpec>> = OnceLock::new();
    SPECS.get_or_init(|| {
        let base = base_specs();
        let by = |b: HumanBone| base.iter().find(|s| s.bone == b).unwrap().clone();
        let mut out: Vec<BoneSpec> = Vec::new();
        // Body and head as authored.
        for b in [B::Spine, B::Chest, B::UpperChest, B::Neck, B::Head, B::LeftEye, B::RightEye, B::Jaw] {
            out.push(by(b));
        }
        let leg = [B::LeftUpperLeg, B::LeftLowerLeg, B::LeftFoot, B::LeftToes];
        let arm = [B::LeftShoulder, B::LeftUpperArm, B::LeftLowerArm, B::LeftHand];
        for b in leg {
            out.push(by(b));
        }
        for b in leg {
            out.push(right_of(&by(b), MuscleGroup::RightLeg));
        }
        for b in arm {
            out.push(by(b));
        }
        for b in arm {
            out.push(right_of(&by(b), MuscleGroup::RightArm));
        }
        // Legs, then arms, then the fingers of each hand.
        for f in LEFT_FINGER_BONES {
            for b in f {
                out.push(by(b));
            }
        }
        for f in LEFT_FINGER_BONES {
            for b in f {
                out.push(right_of(&by(b), MuscleGroup::RightFingers));
            }
        }
        out
    })
}

/// The bone on the other side (either way); centre bones map to themselves.
pub fn other_side(b: HumanBone) -> HumanBone {
    let n = b.name();
    let swapped = match (n.strip_prefix("Left"), n.strip_prefix("Right")) {
        (Some(rest), _) => format!("Right{rest}"),
        (_, Some(rest)) => format!("Left{rest}"),
        _ => return b,
    };
    HumanBone::from_name(&swapped).unwrap_or(b)
}

/// All 95 muscles in index order: Body 0-8, Head 9-20, left leg 21-28, right
/// leg 29-36, left arm 37-45, right arm 46-54, left fingers 55-74, right
/// fingers 75-94.
pub fn muscles() -> &'static [MuscleInfo] {
    static TABLE: OnceLock<Vec<MuscleInfo>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut out: Vec<MuscleInfo> = Vec::new();
        for s in ordered_specs() {
            // Unity lists Z, then Y, then twist.
            for axis in [Axis::Z, Axis::Y, Axis::Twist] {
                if let Some(d) = &s.m[axis as usize] {
                    let centre = other_side(s.bone) == s.bone;
                    out.push(MuscleInfo {
                        name: d.name.clone(),
                        bone: s.bone,
                        axis,
                        group: s.group,
                        min: d.min,
                        max: d.max,
                        lateral: centre && axis != Axis::Z,
                        partner: 0,
                    });
                }
            }
        }
        for i in 0..out.len() {
            let mb = other_side(out[i].bone);
            out[i].partner = out.iter().position(|m| m.bone == mb && m.axis == out[i].axis).unwrap_or(i);
        }
        out
    })
}

/// Index of the muscle called `name`.
pub fn muscle_index(name: &str) -> Option<usize> {
    muscles().iter().position(|m| m.name == name)
}

/// Muscle indices of `bone`, per [`Axis`].
pub fn bone_muscles(bone: HumanBone) -> [Option<usize>; 3] {
    let mut out = [None; 3];
    for (i, m) in muscles().iter().enumerate() {
        if m.bone == bone {
            out[m.axis as usize] = Some(i);
        }
    }
    out
}

/// Default masses per bone (kg out of 82.5), used for the centre of mass.
fn default_mass(b: HumanBone) -> f32 {
    match b {
        B::Hips => 12.0,
        B::LeftUpperLeg | B::RightUpperLeg => 10.0,
        B::LeftLowerLeg | B::RightLowerLeg => 4.0,
        B::LeftFoot | B::RightFoot => 0.8,
        B::Spine => 2.5,
        B::Chest | B::UpperChest => 12.0,
        B::Neck => 1.0,
        B::Head => 4.0,
        B::LeftShoulder | B::RightShoulder => 0.5,
        B::LeftUpperArm | B::RightUpperArm => 2.0,
        B::LeftLowerArm | B::RightLowerArm => 1.5,
        B::LeftHand | B::RightHand => 0.5,
        B::LeftToes | B::RightToes => 0.2,
        _ => 0.0,
    }
}

// ------------------------------------------------------------------ settings

/// One edited limit, by muscle name so the file survives table changes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct MuscleLimit {
    pub muscle: String,
    /// Degrees, `-180 <= min <= 0`.
    pub min: f32,
    /// Degrees, `0 <= max <= 180`.
    pub max: f32,
}

/// Per-avatar muscle settings stored in [`RigImport`]: limits that differ from
/// the defaults in [`muscles`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct MuscleSettings {
    pub limits: Vec<MuscleLimit>,
}

impl MuscleSettings {
    /// The limits (degrees) of muscle `i`: the override or the default.
    pub fn limit(&self, i: usize) -> (f32, f32) {
        let m = &muscles()[i];
        self.limits.iter().find(|l| l.muscle == m.name).map(|l| (l.min, l.max)).unwrap_or((m.min, m.max))
    }

    /// Whether muscle `i` has an override.
    pub fn is_modified(&self, i: usize) -> bool {
        let name = &muscles()[i].name;
        self.limits.iter().any(|l| l.muscle == *name)
    }

    /// Set the limits of muscle `i` (clamped to `min <= 0 <= max` within 180 degrees); equal to the default clears the override.
    pub fn set_limit(&mut self, i: usize, min: f32, max: f32) {
        let m = &muscles()[i];
        let (min, max) = (min.clamp(-180.0, 0.0), max.clamp(0.0, 180.0));
        self.limits.retain(|l| l.muscle != m.name);
        if (min - m.min).abs() > 1e-4 || (max - m.max).abs() > 1e-4 {
            self.limits.push(MuscleLimit { muscle: m.name.clone(), min, max });
        }
    }

    /// Back to the default limits of muscle `i`.
    pub fn reset(&mut self, i: usize) {
        let name = &muscles()[i].name;
        self.limits.retain(|l| l.muscle != *name);
    }

    pub fn reset_all(&mut self) {
        self.limits.clear();
    }
}

// --------------------------------------------------------------- pose types

/// Root of a muscle pose: centre of mass divided by the human scale, and the
/// body orientation relative to the reference pose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyRoot {
    pub position: Vec3,
    pub rotation: Quat,
}

impl Default for BodyRoot {
    fn default() -> Self {
        Self { position: Vec3::ZERO, rotation: Quat::IDENTITY }
    }
}

/// Number of limb goals: left hand, right hand, left foot, right foot.
pub const GOAL_COUNT: usize = 4;
pub const GOAL_NAMES: [&str; GOAL_COUNT] = ["LeftHand", "RightHand", "LeftFoot", "RightFoot"];

/// The bones of each limb (upper, middle, end), matching [`GOAL_NAMES`].
const LIMBS: [[HumanBone; 3]; GOAL_COUNT] = [
    [B::LeftUpperArm, B::LeftLowerArm, B::LeftHand],
    [B::RightUpperArm, B::RightLowerArm, B::RightHand],
    [B::LeftUpperLeg, B::LeftLowerLeg, B::LeftFoot],
    [B::RightUpperLeg, B::RightLowerLeg, B::RightFoot],
];

/// Limb endpoints of a pose, for direction-aligned transfer.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct LimbGoals {
    /// End position in model space divided by the human scale, with the height
    /// measured from the end joint's height in the reference pose (a foot
    /// standing on the ground reads 0, whatever the ankle height).
    pub position: [Vec3; GOAL_COUNT],
    /// End position minus the limb root, divided by the limb length (upper plus lower bone).
    pub relative: [Vec3; GOAL_COUNT],
}

/// A humanoid pose in muscle space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MusclePose {
    pub muscles: [f32; MUSCLE_COUNT],
    pub root: BodyRoot,
    pub goals: Option<LimbGoals>,
    /// Bit `i` set when muscle `i` is driven (a clip may drive only some).
    pub driven: u128,
    /// The root is driven (a clip with root tracks); otherwise the Hips are left alone.
    pub root_driven: bool,
}

pub const ALL_DRIVEN: u128 = (1u128 << MUSCLE_COUNT) - 1;

impl Default for MusclePose {
    fn default() -> Self {
        Self { muscles: [0.0; MUSCLE_COUNT], root: BodyRoot::default(), goals: None, driven: ALL_DRIVEN, root_driven: true }
    }
}

impl MusclePose {
    /// Blend towards `other` by `w` (muscles and root position linearly, root rotation by slerp; goals when both have them).
    pub fn lerp(&self, other: &MusclePose, w: f32) -> MusclePose {
        let mut out = *self;
        for i in 0..MUSCLE_COUNT {
            out.muscles[i] += (other.muscles[i] - self.muscles[i]) * w;
        }
        out.root = BodyRoot { position: self.root.position.lerp(other.root.position, w), rotation: self.root.rotation.slerp(other.root.rotation, w) };
        out.goals = match (self.goals, other.goals) {
            (Some(a), Some(b)) => {
                Some(LimbGoals { position: std::array::from_fn(|k| a.position[k].lerp(b.position[k], w)), relative: std::array::from_fn(|k| a.relative[k].lerp(b.relative[k], w)) })
            }
            (a, b) => {
                if w < 0.5 { a } else { b }
            }
        };
        out.driven = self.driven | other.driven;
        out.root_driven = self.root_driven || other.root_driven;
        out
    }

    /// Mirror across the sagittal plane: swap left and right muscles, negate the lateral centre-line ones.
    pub fn mirror(&mut self) {
        let src = self.muscles;
        let info = muscles();
        for (i, m) in info.iter().enumerate() {
            self.muscles[i] = if m.partner == i { if m.lateral { -src[i] } else { src[i] } } else { src[m.partner] };
        }
        let mut driven = 0u128;
        for (i, m) in info.iter().enumerate() {
            if self.driven >> m.partner & 1 != 0 {
                driven |= 1 << i;
            }
        }
        self.driven = driven;
        self.root.position.x = -self.root.position.x;
        let q = self.root.rotation;
        self.root.rotation = Quat::from_xyzw(q.x, -q.y, -q.z, q.w);
        if let Some(g) = self.goals.as_mut() {
            let flip = |v: Vec3| Vec3::new(-v.x, v.y, v.z);
            let old = *g;
            for k in 0..GOAL_COUNT {
                // Hands and feet are listed left then right.
                g.position[k] = flip(old.position[k ^ 1]);
                g.relative[k] = flip(old.relative[k ^ 1]);
            }
        }
    }
}

// ------------------------------------------------------------------ the rig

/// Axes and neutral rotation of one muscle bone, in the bone's local space at the reference pose.
#[derive(Clone, Copy, Debug)]
struct Frame {
    ax: [Vec3; 3],
    neutral: Quat,
}

/// A skeleton's muscle space: everything [`MuscleRig::muscles_from_pose`] and
/// [`MuscleRig::pose_from_muscles`] need, resolved once from the skeleton's
/// bone map and reference pose so evaluation is allocation-free and indexed.
#[derive(Clone, Debug)]
pub struct MuscleRig {
    /// Joint of each human bone.
    joint: [Option<usize>; HUMAN_BONE_COUNT],
    /// Nearest mapped human ancestor.
    parent: [Option<usize>; HUMAN_BONE_COUNT],
    /// Mapped bones, parents first.
    order: Vec<usize>,
    /// The muscles of each bone, as a bit mask.
    bone_bits: [u128; HUMAN_BONE_COUNT],
    /// Reference transform from the human parent's joint frame to this joint's skeleton parent (for Hips: the parent's reference world).
    chain: [Mat4; HUMAN_BONE_COUNT],
    ref_s: Vec<Vec3>,
    ref_r: Vec<Quat>,
    ref_t: Vec<Vec3>,
    parents: Vec<Option<usize>>,
    frame: [Option<Frame>; HUMAN_BONE_COUNT],
    /// Muscle index per bone and axis.
    slot: [[Option<usize>; 3]; HUMAN_BONE_COUNT],
    /// Limits in radians.
    min: [f32; MUSCLE_COUNT],
    max: [f32; MUSCLE_COUNT],
    mass: [f32; HUMAN_BONE_COUNT],
    mass_child: [Option<usize>; HUMAN_BONE_COUNT],
    human_scale: f32,
    body_ref: Quat,
    /// Height of each limb end joint in the reference pose.
    ref_end_y: [f32; GOAL_COUNT],
    rest: MusclePose,
}

fn rot_of(m: &Mat4) -> Quat {
    m.to_scale_rotation_translation().1.normalize()
}

impl MuscleRig {
    /// Build for `skel` from the bone map and muscle settings of `rig`. `None`
    /// when Hips is unmapped or a mapped joint is missing from the skeleton.
    pub fn new(skel: &Skeleton, rig: &RigImport) -> Option<Self> {
        let mut joint = [None; HUMAN_BONE_COUNT];
        for (b, name) in &rig.human {
            joint[*b as usize] = skel.joints.iter().position(|j| &j.name == name);
        }
        let hips = joint[HumanBone::Hips as usize]?;
        let present = |b: usize| joint[b].is_some();
        let n = skel.joints.len();
        let locals = reference_locals(skel);
        let world = skel.world_matrices(&locals);
        let (mut ref_s, mut ref_r, mut ref_t) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        for l in &locals {
            let (s, r, t) = l.to_scale_rotation_translation();
            ref_s.push(s);
            ref_r.push(r.normalize());
            ref_t.push(t);
        }
        let mut parent = [None; HUMAN_BONE_COUNT];
        for b in crate::human_rig::ALL {
            if present(b as usize) {
                parent[b as usize] = b.mapped_ancestor(|x| present(x as usize)).map(|p| p as usize);
            }
        }
        // Parents first.
        let depth = |mut b: usize| {
            let mut d = 0;
            while let Some(p) = parent[b] {
                b = p;
                d += 1;
            }
            d
        };
        let mut order: Vec<usize> = (0..HUMAN_BONE_COUNT).filter(|&b| present(b)).collect();
        order.sort_by_key(|&b| depth(b));
        let mut chain = [Mat4::IDENTITY; HUMAN_BONE_COUNT];
        for &b in &order {
            let j = joint[b].unwrap();
            let skel_parent = skel.joints[j].parent;
            let sp_world = skel_parent.map(|p| world[p]).unwrap_or(Mat4::IDENTITY);
            chain[b] = match parent[b] {
                Some(p) => world[joint[p].unwrap()].inverse() * sp_world,
                None => sp_world,
            };
        }
        let pos = |b: HumanBone| joint[b as usize].map(|j| world[j].w_axis.truncate());
        let mut frame: [Option<Frame>; HUMAN_BONE_COUNT] = [None; HUMAN_BONE_COUNT];
        let mut slot = [[None; 3]; HUMAN_BONE_COUNT];
        let mut mass_child = [None; HUMAN_BONE_COUNT];
        for s in ordered_specs() {
            let b = s.bone as usize;
            let Some(j) = joint[b] else { continue };
            slot[b] = bone_muscles(s.bone);
            let here = world[j].w_axis.truncate();
            let dflt = s.aim_default.vec(s.mirrored);
            let aim = match &s.aim {
                Aim::Child(list) => list.iter().find_map(|&c| pos(c)).map(|p| p - here),
                Aim::Extend => parent[b].and_then(|p| pos(crate::human_rig::ALL[p])).map(|p| here - p),
                Aim::Fixed => None,
            };
            if let Aim::Child(list) = &s.aim {
                mass_child[b] = list.iter().find(|&&c| present(c as usize)).map(|&c| c as usize);
            }
            let a = aim.filter(|v| v.length() > 1e-5).map(|v| v.normalize()).unwrap_or(dflt);
            let (ey, ez) = swing_axes(a, s.y_dir.vec(s.mirrored), s.z_dir.vec(s.mirrored));
            let wr = rot_of(&world[j]);
            let inv = wr.inverse();
            let twist_sign = s.twist_sign * if s.mirrored { -1.0 } else { 1.0 };
            let ax = [inv * (a * twist_sign), inv * ey, inv * ez];
            let neutral = Quat::from_scaled_axis(ax[1] * s.neutral[0].to_radians() + ax[2] * s.neutral[1].to_radians());
            frame[b] = Some(Frame { ax, neutral });
        }
        // Mass of an unmapped bone goes to its mapped ancestor.
        let mut mass = [0.0; HUMAN_BONE_COUNT];
        for b in crate::human_rig::ALL {
            let m = default_mass(b);
            if m == 0.0 {
                continue;
            }
            let mut t = b as usize;
            while !present(t) {
                match b_parent(t) {
                    Some(p) => t = p,
                    None => break,
                }
            }
            if present(t) {
                mass[t] += m;
            }
        }
        let mut min = [0.0; MUSCLE_COUNT];
        let mut max = [0.0; MUSCLE_COUNT];
        for i in 0..MUSCLE_COUNT {
            let (lo, hi) = rig.muscles.limit(i);
            min[i] = lo.clamp(-180.0, 0.0).to_radians();
            max[i] = hi.clamp(0.0, 180.0).to_radians();
        }
        let mut me = Self {
            joint,
            parent,
            order,
            bone_bits: std::array::from_fn(|b| bone_muscles(crate::human_rig::ALL[b]).into_iter().flatten().fold(0u128, |acc, i| acc | 1 << i)),
            chain,
            ref_s,
            ref_r,
            ref_t,
            parents: skel.joints.iter().map(|j| j.parent).collect(),
            frame,
            slot,
            min,
            max,
            mass,
            mass_child,
            human_scale: 1.0,
            body_ref: Quat::IDENTITY,
            ref_end_y: [0.0; GOAL_COUNT],
            rest: MusclePose::default(),
        };
        // Human scale and the reference body frame from the reference pose itself.
        let rots = me.ref_rotations();
        let worlds = me.bone_worlds(&rots, me.ref_t[hips]);
        let (com, _) = me.com_and_frame(&worlds);
        me.human_scale = if com.y > 0.05 { com.y } else { com.length().max(1e-3) };
        me.body_ref = me.body_frame(&worlds);
        for (k, limb) in LIMBS.iter().enumerate() {
            me.ref_end_y[k] = me.joint[limb[2] as usize].map(|_| worlds[limb[2] as usize].w_axis.y).unwrap_or(0.0);
        }
        let rot_all: Vec<Quat> = me.ref_r.clone();
        me.rest = me.muscles_from_pose(&rot_all, me.ref_t[hips]);
        me.rest.goals = None;
        Some(me)
    }

    /// Height of the reference pose's centre of mass (the unit of the root position).
    pub fn human_scale(&self) -> f32 {
        self.human_scale
    }

    /// Whether `bone` is mapped.
    pub fn has(&self, bone: HumanBone) -> bool {
        self.joint[bone as usize].is_some()
    }

    /// The joint of `bone`.
    pub fn joint(&self, bone: HumanBone) -> Option<usize> {
        self.joint[bone as usize]
    }

    /// The muscles of the reference pose (the value of each muscle in a T-pose).
    pub fn rest_pose(&self) -> &MusclePose {
        &self.rest
    }

    /// The reference pose with its limb goals (what a new muscle clip starts from).
    pub fn reference_pose(&self) -> MusclePose {
        let hips = self.joint[HumanBone::Hips as usize].unwrap_or(0);
        self.muscles_from_pose(&self.ref_r, self.ref_t[hips])
    }

    /// The reference pose's root (standing, upright).
    pub fn reference_root(&self) -> BodyRoot {
        self.rest.root
    }

    /// Limits of muscle `i` in degrees on this rig.
    pub fn limit_degrees(&self, i: usize) -> (f32, f32) {
        (self.min[i].to_degrees(), self.max[i].to_degrees())
    }

    /// Mapped bones with the joint of each, parents first, for the bones that
    /// have a muscle in `driven` (all of them for `ALL_DRIVEN`); the Hips are
    /// not included (see [`MusclePose::root_driven`]).
    pub fn driven_joints(&self, driven: u128) -> impl Iterator<Item = (HumanBone, usize)> + '_ {
        self.order.iter().filter(move |&&b| self.bone_bits[b] & driven != 0 && self.parent[b].is_some()).map(|&b| (crate::human_rig::ALL[b], self.joint[b].unwrap()))
    }

    /// Joint of each muscle's bone (for masks); `None` when the bone is unmapped.
    pub fn muscle_joint(&self, i: usize) -> Option<usize> {
        self.joint[muscles()[i].bone as usize]
    }

    fn ref_rotations(&self) -> [Quat; HUMAN_BONE_COUNT] {
        let mut r = [Quat::IDENTITY; HUMAN_BONE_COUNT];
        for &b in &self.order {
            r[b] = self.ref_r[self.joint[b].unwrap()];
        }
        r
    }

    fn angle(&self, i: usize, v: f32) -> f32 {
        let v = v.clamp(-1.0, 1.0);
        if v >= 0.0 { v * self.max[i] } else { -v * self.min[i] }
    }

    fn value(&self, i: usize, a: f32) -> f32 {
        let v = if a >= 0.0 {
            if self.max[i] > 1e-6 { a / self.max[i] } else { 0.0 }
        } else if self.min[i] < -1e-6 {
            a / -self.min[i]
        } else {
            0.0
        };
        v.clamp(-1.0, 1.0)
    }

    /// Local rotation of bone `b` for the muscle values `m`.
    fn bone_rotation(&self, b: usize, m: &[f32; MUSCLE_COUNT]) -> Quat {
        let j = self.joint[b].unwrap();
        let Some(f) = &self.frame[b] else { return self.ref_r[j] };
        let get = |axis: Axis| self.slot[b][axis as usize].map(|i| self.angle(i, m[i])).unwrap_or(0.0);
        let swing = f.ax[1] * get(Axis::Y) + f.ax[2] * get(Axis::Z);
        let r = f.neutral * Quat::from_scaled_axis(swing) * Quat::from_axis_angle(f.ax[0], get(Axis::Twist));
        (self.ref_r[j] * r).normalize()
    }

    /// Write the muscle values of bone `b` from its local rotation.
    fn bone_values(&self, b: usize, rot: Quat, out: &mut [f32; MUSCLE_COUNT]) {
        let j = self.joint[b].unwrap();
        let Some(f) = &self.frame[b] else { return };
        let mut q = (f.neutral.inverse() * self.ref_r[j].inverse() * rot).normalize();
        if q.w < 0.0 {
            q = -q;
        }
        let twist = 2.0 * q.xyz().dot(f.ax[0]).atan2(q.w);
        let mut swing = (q * Quat::from_axis_angle(f.ax[0], twist).inverse()).normalize();
        if swing.w < 0.0 {
            swing = -swing;
        }
        let rv = swing.to_scaled_axis();
        let set = |axis: Axis, a: f32, out: &mut [f32; MUSCLE_COUNT]| {
            if let Some(i) = self.slot[b][axis as usize] {
                out[i] = self.value(i, a);
            }
        };
        set(Axis::Twist, twist, out);
        set(Axis::Y, rv.dot(f.ax[1]), out);
        set(Axis::Z, rv.dot(f.ax[2]), out);
    }

    /// Model-space matrix of every mapped bone for local rotations `rots` (by bone) and the Hips translation.
    fn bone_worlds(&self, rots: &[Quat; HUMAN_BONE_COUNT], hips_t: Vec3) -> [Mat4; HUMAN_BONE_COUNT] {
        let mut w = [Mat4::IDENTITY; HUMAN_BONE_COUNT];
        for &b in &self.order {
            let j = self.joint[b].unwrap();
            let t = if self.parent[b].is_none() { hips_t } else { self.ref_t[j] };
            let local = Mat4::from_scale_rotation_translation(self.ref_s[j], rots[b], t);
            w[b] = match self.parent[b] {
                Some(p) => w[p] * self.chain[b] * local,
                None => self.chain[b] * local,
            };
        }
        w
    }

    /// Centre of mass and the body frame of the pose.
    fn com_and_frame(&self, w: &[Mat4; HUMAN_BONE_COUNT]) -> (Vec3, Quat) {
        let (mut sum, mut total) = (Vec3::ZERO, 0.0);
        for &b in &self.order {
            let m = self.mass[b];
            if m <= 0.0 {
                continue;
            }
            let p = w[b].w_axis.truncate();
            let c = match self.mass_child[b] {
                Some(c) => p.lerp(w[c].w_axis.truncate(), 0.5),
                None => p,
            };
            sum += c * m;
            total += m;
        }
        let com = if total > 0.0 { sum / total } else { w[HumanBone::Hips as usize].w_axis.truncate() };
        (com, self.body_frame(w))
    }

    /// Orientation of the body: the Hips world rotation (the spine's bends stay visible in the muscles).
    fn body_frame(&self, w: &[Mat4; HUMAN_BONE_COUNT]) -> Quat {
        rot_of(&w[HumanBone::Hips as usize])
    }

    /// Muscle values (clamped to [-1, 1]), root and limb goals of a pose given as
    /// local rotations per skeleton joint and the Hips local translation.
    pub fn muscles_from_pose(&self, rot: &[Quat], hips_t: Vec3) -> MusclePose {
        let mut rots = [Quat::IDENTITY; HUMAN_BONE_COUNT];
        let mut m = [0.0; MUSCLE_COUNT];
        for &b in &self.order {
            let j = self.joint[b].unwrap();
            rots[b] = rot.get(j).copied().unwrap_or(self.ref_r[j]).normalize();
            if self.parent[b].is_some() {
                self.bone_values(b, rots[b], &mut m);
            }
        }
        let w = self.bone_worlds(&rots, hips_t);
        let (com, frame) = self.com_and_frame(&w);
        let root = BodyRoot { position: com / self.human_scale, rotation: (frame * self.body_ref.inverse()).normalize() };
        let mut goals = LimbGoals::default();
        let mut any = false;
        for (k, limb) in LIMBS.iter().enumerate() {
            let [u, _, e] = limb.map(|b| b as usize);
            let (Some(_), Some(_)) = (self.joint[u], self.joint[e]) else { continue };
            let Some(len) = self.limb_length(&w, k) else { continue };
            let (pu, pe) = (w[u].w_axis.truncate(), w[e].w_axis.truncate());
            goals.position[k] = Vec3::new(pe.x, pe.y - self.ref_end_y[k], pe.z) / self.human_scale;
            goals.relative[k] = (pe - pu) / len;
            any = true;
        }
        MusclePose { muscles: m, root, goals: any.then_some(goals), driven: ALL_DRIVEN, root_driven: true }
    }

    fn limb_length(&self, w: &[Mat4; HUMAN_BONE_COUNT], k: usize) -> Option<f32> {
        let [u, m, e] = LIMBS[k].map(|b| b as usize);
        self.joint[m]?;
        self.joint[u]?;
        self.joint[e]?;
        let (pu, pm, pe) = (w[u].w_axis.truncate(), w[m].w_axis.truncate(), w[e].w_axis.truncate());
        Some(((pm - pu).length() + (pe - pm).length()).max(1e-6))
    }

    /// Add the muscles of bones this rig lacks to their parents (see the module docs) and zero them.
    pub fn fold_missing(&self, m: &mut [f32; MUSCLE_COUNT]) {
        let has = |b: HumanBone| self.joint[b as usize].is_some();
        let clear = |m: &mut [f32; MUSCLE_COUNT], b: HumanBone| {
            for i in bone_muscles(b).into_iter().flatten() {
                m[i] = 0.0;
            }
        };
        let stat = |m: &mut [f32; MUSCLE_COUNT], from: HumanBone, to: HumanBone, k: f32| {
            let (a, b) = (bone_muscles(from), bone_muscles(to));
            for axis in 0..3 {
                if let (Some(a), Some(b)) = (a[axis], b[axis]) {
                    m[b] += m[a] * k;
                }
            }
        };
        if !has(B::Neck) {
            stat(m, B::Neck, B::Head, 1.0);
            clear(m, B::Neck);
        }
        if !has(B::UpperChest) {
            stat(m, B::UpperChest, B::Chest, 0.5);
            clear(m, B::UpperChest);
        }
        if !has(B::Chest) {
            stat(m, B::Chest, B::Spine, 1.0);
            clear(m, B::Chest);
        }
        for (sh, arm) in [(B::LeftShoulder, B::LeftUpperArm), (B::RightShoulder, B::RightUpperArm)] {
            if !has(sh) {
                let (s, a) = (bone_muscles(sh), bone_muscles(arm));
                if let (Some(s), Some(a)) = (s[Axis::Z as usize], a[Axis::Z as usize]) {
                    m[a] += m[s] * 0.15;
                }
                if let (Some(s), Some(a)) = (s[Axis::Y as usize], a[Axis::Y as usize]) {
                    m[a] += m[s] * 0.28125;
                }
                clear(m, sh);
            }
        }
        for v in m.iter_mut() {
            *v = v.clamp(-1.0, 1.0);
        }
    }

    /// Write the joint rotations (and the Hips translation) of a muscle pose into
    /// `rot` / `trans`, indexed by skeleton joint. Joints that are not human
    /// bones are left as they are. With `pose.goals` set, the limbs are then
    /// aligned to the goals.
    pub fn pose_from_muscles(&self, pose: &MusclePose, rot: &mut [Quat], trans: &mut [Vec3]) {
        let mut m = pose.muscles;
        self.fold_missing(&mut m);
        let mut rots = [Quat::IDENTITY; HUMAN_BONE_COUNT];
        for &b in &self.order {
            rots[b] = self.bone_rotation(b, &m);
        }
        let hips = HumanBone::Hips as usize;
        let hj = self.joint[hips].unwrap();
        // Provisional hips at the reference: measure the body frame and centre of mass.
        let w0 = self.bone_worlds(&rots, self.ref_t[hj]);
        let (com0, frame0) = self.com_and_frame(&w0);
        let desired = (pose.root.rotation * self.body_ref).normalize();
        let delta = (desired * frame0.inverse()).normalize();
        let hips_pos0 = w0[hips].w_axis.truncate();
        let com1 = hips_pos0 + delta * (com0 - hips_pos0);
        let new_pos = hips_pos0 + (pose.root.position * self.human_scale - com1);
        let parent_world = self.chain[hips];
        let hips_world_rot = (delta * rot_of(&w0[hips])).normalize();
        let local_rot = (rot_of(&parent_world).inverse() * hips_world_rot).normalize();
        let local_t = parent_world.inverse().transform_point3(new_pos);
        rots[hips] = local_rot;
        for &b in &self.order {
            let j = self.joint[b].unwrap();
            if let Some(r) = rot.get_mut(j) {
                *r = rots[b];
            }
        }
        if let Some(t) = trans.get_mut(hj) {
            *t = local_t;
        }
        if let Some(g) = &pose.goals {
            self.align_limbs(g, rot, trans);
        }
    }

    // ------------------------------------------------------- limb alignment

    /// World matrix of joint `j` from the given local rotations and translations (scale from the reference).
    fn joint_world(&self, j: usize, rot: &[Quat], trans: &[Vec3]) -> Mat4 {
        let mut chain = [0usize; 64];
        let (mut n, mut cur) = (0, Some(j));
        while let Some(c) = cur {
            if n == chain.len() {
                break;
            }
            chain[n] = c;
            n += 1;
            cur = self.parents[c];
        }
        let mut w = Mat4::IDENTITY;
        for &c in chain[..n].iter().rev() {
            w *= Mat4::from_scale_rotation_translation(self.ref_s[c], rot[c], trans[c]);
        }
        w
    }

    /// The goal position of limb `k` in this rig's model space.
    fn goal_world(&self, g: &LimbGoals, k: usize) -> Vec3 {
        let p = g.position[k] * self.human_scale;
        Vec3::new(p.x, p.y + self.ref_end_y[k], p.z)
    }

    /// Bend each limb so its end meets the goal: see the module docs.
    fn align_limbs(&self, g: &LimbGoals, rot: &mut [Quat], trans: &mut [Vec3]) {
        if rot.len() < self.parents.len() || trans.len() < self.parents.len() {
            return;
        }
        let ground = |h: f32| 1.0 - smoothstep(0.05, 0.15, h);
        // Lower the root by what a grounded foot cannot reach.
        let mut drop = 0.0f32;
        for (k, limb) in LIMBS.iter().enumerate().skip(2) {
            let [u, m, e] = limb.map(|b| self.joint[b as usize]);
            let (Some(u), Some(m), Some(e)) = (u, m, e) else { continue };
            let (wu, wm, we) = (self.joint_world(u, rot, trans), self.joint_world(m, rot, trans), self.joint_world(e, rot, trans));
            let (pu, pm, pe) = (wu.w_axis.truncate(), wm.w_axis.truncate(), we.w_axis.truncate());
            let reach = (pm - pu).length() + (pe - pm).length();
            let target = self.goal_world(g, k);
            let w = ground(g.position[k].y);
            drop = drop.max(((target - pu).length() - reach * 0.999).max(0.0) * w);
        }
        if drop > 1e-5 {
            let hj = self.joint[HumanBone::Hips as usize].unwrap();
            let parent_world = self.parents[hj].map(|p| self.joint_world(p, rot, trans)).unwrap_or(Mat4::IDENTITY);
            trans[hj] += parent_world.inverse().transform_vector3(Vec3::new(0.0, -drop, 0.0));
        }
        for (k, limb) in LIMBS.iter().enumerate() {
            let [u, m, e] = limb.map(|b| self.joint[b as usize]);
            let (Some(u), Some(m), Some(e)) = (u, m, e) else { continue };
            let bm = LIMBS[k][1] as usize;
            let wu = self.joint_world(u, rot, trans);
            let wm = self.joint_world(m, rot, trans);
            let we = self.joint_world(e, rot, trans);
            let (pu, pm, pe) = (wu.w_axis.truncate(), wm.w_axis.truncate(), we.w_axis.truncate());
            let (l1, l2) = ((pm - pu).length(), (pe - pm).length());
            if l1 < 1e-6 || l2 < 1e-6 {
                continue;
            }
            let by_ratio = pu + g.relative[k] * (l1 + l2);
            let w = if k >= 2 { ground(g.position[k].y) } else { 0.0 };
            let target = by_ratio.lerp(self.goal_world(g, k), w);
            let d_tgt = (target - pu).length().clamp((l1 - l2).abs() + 1e-4, (l1 + l2) * 0.9999);
            let cur = (pe - pu).length().max(1e-6);
            let cos_now = ((l1 * l1 + l2 * l2 - cur * cur) / (2.0 * l1 * l2)).clamp(-1.0, 1.0);
            let cos_new = ((l1 * l1 + l2 * l2 - d_tgt * d_tgt) / (2.0 * l1 * l2)).clamp(-1.0, 1.0);
            let bend = cos_new.acos() - cos_now.acos(); // interior angle change (positive = straighter)
            let wm_rot = rot_of(&wm);
            let axis = match self.frame[bm] {
                // The hinge in world space: the middle bone's stretch axis at the reference, carried by its parent's current rotation.
                Some(f) => {
                    let parent_rot = self.parents[m].map(|p| rot_of(&self.joint_world(p, rot, trans))).unwrap_or(Quat::IDENTITY);
                    let a = parent_rot * (self.ref_r[m] * f.ax[2]);
                    a.normalize_or_zero()
                }
                None => Vec3::ZERO,
            };
            let axis = if axis.length() > 0.5 { axis } else { (pm - pu).cross(pe - pm).normalize_or_zero() };
            let axis = if axis.length() > 0.5 { axis } else { (pe - pu).normalize().any_orthonormal_vector() };
            // Positive rotation about the hinge straightens the limb.
            let r1 = Quat::from_axis_angle(axis, bend);
            let pe1 = pm + r1 * (pe - pm);
            let dir_now = pe1 - pu;
            let dir_tgt = (target - pu).normalize_or_zero() * d_tgt;
            if dir_now.length() < 1e-6 || dir_tgt.length() < 1e-6 {
                continue;
            }
            let r2 = Quat::from_rotation_arc(dir_now.normalize(), dir_tgt.normalize());
            let up_parent = self.parents[u].map(|p| rot_of(&self.joint_world(p, rot, trans))).unwrap_or(Quat::IDENTITY);
            let mid_parent = self.parents[m].map(|p| rot_of(&self.joint_world(p, rot, trans))).unwrap_or(Quat::IDENTITY);
            let end_parent = self.parents[e].map(|p| rot_of(&self.joint_world(p, rot, trans))).unwrap_or(Quat::IDENTITY);
            let end_world = rot_of(&we);
            rot[u] = (up_parent.inverse() * r2 * rot_of(&wu)).normalize();
            rot[m] = (mid_parent.inverse() * r1 * wm_rot).normalize();
            if k >= 2 {
                // The sole keeps its world orientation.
                rot[e] = ((r2 * r1 * end_parent).inverse() * end_world).normalize();
            }
        }
    }

    /// Joint positions of a muscle pose, for tests and previews: the world matrices of every joint.
    pub fn world_matrices(&self, skel: &Skeleton, rot: &[Quat], trans: &[Vec3]) -> Vec<Mat4> {
        let locals: Vec<Mat4> = (0..skel.joints.len()).map(|i| Mat4::from_scale_rotation_translation(self.ref_s[i], rot[i], trans[i])).collect();
        skel.world_matrices(&locals)
    }

    /// Local rotations and translations of the reference pose, the starting point for [`Self::pose_from_muscles`].
    pub fn reference_locals(&self) -> (Vec<Quat>, Vec<Vec3>) {
        (self.ref_r.clone(), self.ref_t.clone())
    }
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Human parent index (ignoring what is mapped).
fn b_parent(b: usize) -> Option<usize> {
    crate::human_rig::PARENT[b].map(|p| p as usize)
}

/// The signed swing axes (Y, Z) for aim `a`: positive Y moves the tip towards
/// `y_dir`, positive Z towards `z_dir` (each projected perpendicular to `a`).
fn swing_axes(a: Vec3, y_dir: Vec3, z_dir: Vec3) -> (Vec3, Vec3) {
    let perp = |d: Vec3| {
        let p = d - a * d.dot(a);
        if p.length() > 0.2 {
            p.normalize()
        } else {
            let t = if a.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
            (t - a * t.dot(a)).normalize()
        }
    };
    let ey = a.cross(perp(y_dir)).normalize();
    let ez0 = a.cross(ey);
    let want = a.cross(perp(z_dir));
    let ez = if ez0.dot(want) >= 0.0 { ez0 } else { -ez0 };
    (ey, ez)
}

// ----------------------------------------------------------- muscle clips

/// A muscle clip's tracks resolved to muscle indices once, so sampling does no
/// string work. Resolve again whenever the clip's tracks change.
#[derive(Clone, Debug, Default)]
pub struct MuscleBinding {
    /// (track, muscle).
    curves: Vec<(u32, u8)>,
    root_position: Option<u32>,
    root_rotation: Option<u32>,
    goal_position: [Option<u32>; GOAL_COUNT],
    goal_relative: [Option<u32>; GOAL_COUNT],
}

impl MuscleBinding {
    pub fn new(clip: &AnimClip) -> Self {
        let mut b = Self::default();
        let goal = |name: &str| GOAL_NAMES.iter().position(|g| *g == name);
        for (i, t) in clip.tracks.iter().enumerate() {
            if t.keys.is_empty() {
                continue;
            }
            let i = i as u32;
            match t.property {
                TrackProperty::Muscle => {
                    if let Some(m) = muscle_index(&t.name)
                        && !b.curves.iter().any(|c| c.1 as usize == m)
                    {
                        b.curves.push((i, m as u8));
                    }
                }
                TrackProperty::RootPosition => b.root_position = Some(i),
                TrackProperty::RootRotation => b.root_rotation = Some(i),
                TrackProperty::GoalPosition => {
                    if let Some(g) = goal(&t.name) {
                        b.goal_position[g] = Some(i);
                    }
                }
                TrackProperty::GoalRelative => {
                    if let Some(g) = goal(&t.name) {
                        b.goal_relative[g] = Some(i);
                    }
                }
                _ => {}
            }
        }
        b
    }

    /// Whether the clip drives anything.
    pub fn is_empty(&self) -> bool {
        self.curves.is_empty() && self.root_position.is_none() && self.root_rotation.is_none()
    }

    /// The pose at clip time `t` (already wrapped). Muscles without a curve, and
    /// a missing root, keep their value in `base` and are not marked driven.
    pub fn sample(&self, clip: &AnimClip, t: f32, base: &MusclePose) -> MusclePose {
        let mut p = *base;
        p.driven = 0;
        p.root_driven = self.root_position.is_some() || self.root_rotation.is_some();
        for &(track, m) in &self.curves {
            p.muscles[m as usize] = clip.tracks[track as usize].sample(t)[0];
            p.driven |= 1 << m;
        }
        if let Some(i) = self.root_position {
            let v = clip.tracks[i as usize].sample(t);
            p.root.position = Vec3::new(v[0], v[1], v[2]);
        }
        if let Some(i) = self.root_rotation {
            let q = Quat::from_array(clip.tracks[i as usize].sample(t));
            if q.length_squared() > 1e-6 {
                p.root.rotation = q.normalize();
            }
        }
        if self.goal_position.iter().chain(&self.goal_relative).all(|g| g.is_some()) {
            let at = |i: Option<u32>| {
                let v = clip.tracks[i.unwrap() as usize].sample(t);
                Vec3::new(v[0], v[1], v[2])
            };
            p.goals = Some(LimbGoals { position: std::array::from_fn(|k| at(self.goal_position[k])), relative: std::array::from_fn(|k| at(self.goal_relative[k])) });
        } else {
            p.goals = None;
        }
        p
    }
}

/// Sample a muscle clip at playback time `t` (wrapped when `looping`, else clamped), unbound (resolves names on every call: for tools and tests).
pub fn sample_muscle_clip(clip: &AnimClip, t: f32, base: &MusclePose) -> MusclePose {
    MuscleBinding::new(clip).sample(clip, clip.local_time(t, clip.looping), base)
}

/// Write `pose` as keys at `time` into the tracks of a muscle clip (creating them).
pub fn set_muscle_keys(clip: &mut AnimClip, time: f32, pose: &MusclePose) {
    let put = |clip: &mut AnimClip, prop: TrackProperty, name: &str, v: [f32; 4]| {
        let i = match clip.prop_track_index("", prop, name, 0) {
            Some(i) => i,
            None => {
                clip.tracks.push(AnimTrack::prop("", prop, name, 0));
                clip.tracks.len() - 1
            }
        };
        clip.tracks[i].set_key(time, v);
    };
    for (i, m) in muscles().iter().enumerate() {
        if pose.driven >> i & 1 != 0 {
            put(clip, TrackProperty::Muscle, &m.name, [pose.muscles[i], 0.0, 0.0, 0.0]);
        }
    }
    let p = pose.root.position;
    put(clip, TrackProperty::RootPosition, "", [p.x, p.y, p.z, 0.0]);
    put(clip, TrackProperty::RootRotation, "", pose.root.rotation.to_array());
    if let Some(g) = &pose.goals {
        for (k, name) in GOAL_NAMES.iter().enumerate() {
            put(clip, TrackProperty::GoalPosition, name, [g.position[k].x, g.position[k].y, g.position[k].z, 0.0]);
            put(clip, TrackProperty::GoalRelative, name, [g.relative[k].x, g.relative[k].y, g.relative[k].z, 0.0]);
        }
    }
}

/// Bake an imported humanoid clip into a muscle clip on `skel`: the clip is
/// resampled at `fps`, each frame turned into muscles, root and goals, and the
/// imported clip's name kept as the fallback for non-humanoid rigs. Tracks
/// that never change collapse to one key.
pub fn bake_muscle_clip(skel: &Skeleton, rig: &MuscleRig, clip: &AnimationClip, fps: f32) -> AnimClip {
    let fps = fps.clamp(1.0, 240.0);
    let frames = ((clip.duration * fps).round() as usize).max(1);
    let hips = rig.joint(HumanBone::Hips).unwrap_or(0);
    let mut out = AnimClip { length: clip.duration.max(1.0 / fps), looping: true, frame_rate: fps, kind: ClipKind::Muscle, fallback: clip.name.clone(), ..Default::default() };
    let mut prev_q: Option<Quat> = None;
    for f in 0..=frames {
        let t = (f as f32 / fps).min(clip.duration);
        let locals = clip.sample(skel, t.min((clip.duration - 1e-4).max(0.0)));
        let rot: Vec<Quat> = locals.iter().map(|m| m.to_scale_rotation_translation().1.normalize()).collect();
        let hips_t = locals.get(hips).map(|m| m.w_axis.truncate()).unwrap_or(Vec3::ZERO);
        let mut pose = rig.muscles_from_pose(&rot, hips_t);
        if let Some(q) = prev_q
            && q.dot(pose.root.rotation) < 0.0
        {
            pose.root.rotation = -pose.root.rotation;
        }
        prev_q = Some(pose.root.rotation);
        set_muscle_keys(&mut out, t, &pose);
    }
    for t in &mut out.tracks {
        for k in &mut t.keys {
            k.interp = KeyInterp::Linear;
        }
        let first = t.keys[0].value;
        if t.keys.iter().all(|k| k.value == first) {
            t.keys.truncate(1);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::{AnimChannel, ChannelPath};
    use crate::Joint;

    #[derive(Clone, Copy)]
    struct Opts {
        s: f32,
        arm: f32,
        leg: f32,
        chest: bool,
        upper_chest: bool,
        neck: bool,
        shoulders: bool,
        toes: bool,
    }

    impl Default for Opts {
        fn default() -> Self {
            Self { s: 1.0, arm: 1.0, leg: 1.0, chest: true, upper_chest: true, neck: true, shoulders: true, toes: true }
        }
    }

    /// A T-pose humanoid facing +Z with the avatar's left on +X. Dimensions scale with `s`;
    /// `arm` and `leg` stretch those limbs. Each side has an index finger.
    fn humanoid(o: Opts) -> (Skeleton, RigImport) {
        let s = o.s;
        let mut joints: Vec<Joint> = Vec::new();
        let mut human: Vec<(HumanBone, String)> = Vec::new();
        let add = |b: HumanBone, parent: Option<&str>, t: Vec3, joints: &mut Vec<Joint>, human: &mut Vec<(HumanBone, String)>| {
            let name = b.name().to_string();
            let p = parent.map(|n| joints.iter().position(|j| j.name == n).unwrap());
            joints.push(Joint { name: name.clone(), parent: p, inverse_bind: Mat4::IDENTITY, rest_translation: t, rest_rotation: Quat::IDENTITY, rest_scale: Vec3::ONE });
            human.push((b, name));
        };
        let hips_y = (0.9 * o.leg + 0.08) * s;
        add(B::Hips, None, Vec3::new(0.0, hips_y, 0.0), &mut joints, &mut human);
        add(B::Spine, Some("Hips"), Vec3::new(0.0, 0.12 * s, 0.0), &mut joints, &mut human);
        let mut top = "Spine";
        if o.chest {
            add(B::Chest, Some(top), Vec3::new(0.0, 0.15 * s, 0.0), &mut joints, &mut human);
            top = "Chest";
        }
        if o.upper_chest {
            add(B::UpperChest, Some(top), Vec3::new(0.0, 0.12 * s, 0.0), &mut joints, &mut human);
            top = "UpperChest";
        }
        let mut head_parent = top;
        if o.neck {
            add(B::Neck, Some(top), Vec3::new(0.0, 0.15 * s, 0.0), &mut joints, &mut human);
            head_parent = "Neck";
        }
        add(B::Head, Some(head_parent), Vec3::new(0.0, if o.neck { 0.1 } else { 0.25 } * s, 0.0), &mut joints, &mut human);
        for (side, x) in [("Left", 1.0f32), ("Right", -1.0)] {
            let bone = |n: &str| HumanBone::from_name(&format!("{side}{n}")).unwrap();
            let nm = |n: &str| format!("{side}{n}");
            let mut arm_parent = top.to_string();
            let mut off = Vec3::new(0.12 * x * s, 0.1 * s, 0.0);
            if o.shoulders {
                add(bone("Shoulder"), Some(top), Vec3::new(0.04 * x * s, 0.1 * s, 0.0), &mut joints, &mut human);
                arm_parent = nm("Shoulder");
                off = Vec3::new(0.08 * x * s, 0.0, 0.0);
            }
            add(bone("UpperArm"), Some(&arm_parent), off, &mut joints, &mut human);
            add(bone("LowerArm"), Some(&nm("UpperArm")), Vec3::new(0.28 * o.arm * x * s, 0.0, 0.0), &mut joints, &mut human);
            add(bone("Hand"), Some(&nm("LowerArm")), Vec3::new(0.26 * o.arm * x * s, 0.0, 0.0), &mut joints, &mut human);
            add(bone("IndexProximal"), Some(&nm("Hand")), Vec3::new(0.08 * x * s, 0.0, 0.02 * s), &mut joints, &mut human);
            add(bone("IndexIntermediate"), Some(&nm("IndexProximal")), Vec3::new(0.04 * x * s, 0.0, 0.0), &mut joints, &mut human);
            add(bone("IndexDistal"), Some(&nm("IndexIntermediate")), Vec3::new(0.03 * x * s, 0.0, 0.0), &mut joints, &mut human);
            add(bone("UpperLeg"), Some("Hips"), Vec3::new(0.1 * x * s, 0.0, 0.0), &mut joints, &mut human);
            add(bone("LowerLeg"), Some(&nm("UpperLeg")), Vec3::new(0.0, -0.45 * o.leg * s, 0.0), &mut joints, &mut human);
            add(bone("Foot"), Some(&nm("LowerLeg")), Vec3::new(0.0, -0.45 * o.leg * s, 0.0), &mut joints, &mut human);
            if o.toes {
                add(bone("Toes"), Some(&nm("Foot")), Vec3::new(0.0, -0.04 * s, 0.14 * s), &mut joints, &mut human);
            }
        }
        let rig = RigImport { animation_type: crate::human_rig::AnimationType::Humanoid, human, ..Default::default() };
        (Skeleton { joints, rig: None, ..Default::default() }, rig)
    }

    struct Fixture {
        skel: Skeleton,
        mr: MuscleRig,
    }

    fn fixture(o: Opts) -> Fixture {
        let (skel, rig) = humanoid(o);
        let mr = MuscleRig::new(&skel, &rig).expect("muscle rig");
        Fixture { skel, mr }
    }

    impl Fixture {
        fn fk(&self, p: &MusclePose) -> Vec<Mat4> {
            let (mut r, mut t) = self.mr.reference_locals();
            self.mr.pose_from_muscles(p, &mut r, &mut t);
            self.mr.world_matrices(&self.skel, &r, &t)
        }

        fn pos(&self, w: &[Mat4], name: &str) -> Vec3 {
            w[self.skel.joints.iter().position(|j| j.name == name).unwrap()].w_axis.truncate()
        }

        fn rest(&self) -> MusclePose {
            *self.mr.rest_pose()
        }
    }

    fn mi(name: &str) -> usize {
        muscle_index(name).unwrap_or_else(|| panic!("no muscle {name}"))
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
        fn signed(&mut self) -> f32 {
            self.next() * 2.0 - 1.0
        }
    }

    #[test]
    fn the_table_has_95_muscles_in_unity_order() {
        let t = muscles();
        assert_eq!(t.len(), MUSCLE_COUNT);
        let mut names: Vec<&str> = t.iter().map(|m| m.name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), MUSCLE_COUNT, "names are unique");
        for (name, idx) in [
            ("Spine Front-Back", 0),
            ("UpperChest Twist Left-Right", 8),
            ("Neck Nod Down-Up", 9),
            ("Left Eye Down-Up", 15),
            ("Jaw Close", 19),
            ("Left Upper Leg Front-Back", 21),
            ("Left Lower Leg Stretch", 24),
            ("Left Toes Up-Down", 28),
            ("Right Upper Leg Front-Back", 29),
            ("Left Shoulder Down-Up", 37),
            ("Left Forearm Stretch", 42),
            ("Right Forearm Stretch", 51),
            ("Left Thumb 1 Stretched", 55),
            ("Left Index 2 Stretched", 61),
            ("Right Thumb 1 Stretched", 75),
            ("Right Little 3 Stretched", 94),
        ] {
            assert_eq!(muscle_index(name), Some(idx), "{name}");
        }
        let count = |g| t.iter().filter(|m| m.group == g).count();
        assert_eq!([count(MuscleGroup::Body), count(MuscleGroup::Head), count(MuscleGroup::LeftArm), count(MuscleGroup::RightArm)], [9, 12, 9, 9]);
        assert_eq!([count(MuscleGroup::LeftLeg), count(MuscleGroup::RightLeg), count(MuscleGroup::LeftFingers), count(MuscleGroup::RightFingers)], [8, 8, 20, 20]);
        // Partners are involutions.
        for (i, m) in t.iter().enumerate() {
            assert_eq!(t[m.partner].partner, i, "{}", m.name);
        }
        for m in t {
            assert!(m.min <= 0.0 && m.max >= 0.0, "{} limits straddle zero", m.name);
        }
    }

    #[test]
    fn golden_default_limits() {
        let lim = |n: &str| {
            let m = &muscles()[mi(n)];
            (m.min, m.max)
        };
        assert_eq!(lim("Left Arm Down-Up"), (-60.0, 100.0));
        assert_eq!(lim("Right Arm Down-Up"), (-60.0, 100.0));
        assert_eq!(lim("Left Upper Leg Front-Back"), (-90.0, 50.0));
        assert_eq!(lim("Left Forearm Stretch"), (-80.0, 80.0));
        assert_eq!(lim("Right Ring Spread"), (-7.5, 7.5));
        assert_eq!(lim("Left Thumb 2 Stretched"), (-40.0, 35.0));
        assert_eq!(lim("UpperChest Front-Back"), (-20.0, 20.0));
        assert_eq!(lim("Left Eye Down-Up"), (-10.0, 15.0));
    }

    #[test]
    fn golden_muscle_values_of_the_reference_pose() {
        let f = fixture(Opts::default());
        let r = f.rest();
        let v = |n: &str| r.muscles[mi(n)];
        // Muscle 0 is a relaxed human (bent elbows and knees), so a straight T-pose limb reads at the top of its range.
        assert!((v("Left Forearm Stretch") - 1.0).abs() < 1e-3, "{}", v("Left Forearm Stretch"));
        assert!((v("Right Lower Leg Stretch") - 1.0).abs() < 1e-3);
        assert!((v("Left Arm Down-Up") - 0.4).abs() < 1e-3, "40 of 100 degrees: {}", v("Left Arm Down-Up"));
        assert!((v("Left Arm Front-Back") - 0.3).abs() < 1e-3);
        assert!((v("Right Arm Down-Up") - 0.4).abs() < 1e-3, "left and right agree");
        assert!((v("Left Upper Leg Front-Back") - 0.6).abs() < 1e-3, "30 of 50 degrees: {}", v("Left Upper Leg Front-Back"));
        assert!((v("Left Index 1 Stretched") - 0.66).abs() < 1e-3, "33 of 50 degrees: {}", v("Left Index 1 Stretched"));
        assert!((v("Left Index 2 Stretched") - 36.5 / 45.0).abs() < 1e-3);
        assert!(v("Spine Front-Back").abs() < 1e-4 && v("Head Turn Left-Right").abs() < 1e-4);
        assert!(r.root.rotation.angle_between(Quat::IDENTITY) < 1e-3, "an upright T-pose has no body rotation");
    }

    #[test]
    fn muscle_signs_move_the_body_the_way_the_names_say() {
        let f = fixture(Opts::default());
        let base = f.rest();
        let dir = |w: &[Mat4], a: &str, b: &str| (f.pos(w, b) - f.pos(w, a)).normalize();
        let with = |edit: &dyn Fn(&mut MusclePose)| {
            let mut p = base;
            edit(&mut p);
            f.fk(&p)
        };
        // The neutral arm (muscles 0) hangs 40 degrees down and 30 forward: the swing vector composes as (0, -30, -40) degrees.
        let neutral = |side: &str, edit: &dyn Fn(&mut MusclePose, &str)| {
            let mut p = base;
            for n in ["Arm Down-Up", "Arm Front-Back", "Arm Twist In-Out"] {
                p.muscles[mi(&format!("{side} {n}"))] = 0.0;
            }
            edit(&mut p, side);
            let w = f.fk(&p);
            dir(&w, &format!("{side}UpperArm"), &format!("{side}LowerArm"))
        };
        for side in ["Left", "Right"] {
            let d0 = neutral(side, &|_, _| {});
            assert!(d0.y < -0.3 && d0.z > 0.3, "{side}: down and forward {d0:?}");
            let up = neutral(side, &|p, s| p.muscles[mi(&format!("{s} Arm Down-Up"))] = 1.0);
            let down = neutral(side, &|p, s| p.muscles[mi(&format!("{s} Arm Down-Up"))] = -1.0);
            assert!(up.y > d0.y + 0.3 && down.y < d0.y, "{side}: Down-Up raises the arm");
            let back = neutral(side, &|p, s| p.muscles[mi(&format!("{s} Arm Front-Back"))] = 1.0);
            let front = neutral(side, &|p, s| p.muscles[mi(&format!("{s} Arm Front-Back"))] = -1.0);
            assert!(back.z < d0.z - 0.3 && front.z > d0.z, "{side}: Front-Back swings the arm backwards for +1");
        }
        // Forearm Stretch 0: elbow bent 80 degrees, the hand in front.
        let w = with(&|p| p.muscles[mi("Left Forearm Stretch")] = 0.0);
        let (upper, lower) = (dir(&w, "LeftUpperArm", "LeftLowerArm"), dir(&w, "LeftLowerArm", "LeftHand"));
        assert!((upper.angle_between(lower).to_degrees() - 80.0).abs() < 0.1);
        assert!(f.pos(&w, "LeftHand").z > f.pos(&w, "LeftLowerArm").z + 0.1);
        // Lower Leg Stretch 0: the knee is bent 80 degrees, the foot behind.
        let w = with(&|p| p.muscles[mi("Right Lower Leg Stretch")] = 0.0);
        let (thigh, shin) = (dir(&w, "RightUpperLeg", "RightLowerLeg"), dir(&w, "RightLowerLeg", "RightFoot"));
        assert!((thigh.angle_between(shin).to_degrees() - 80.0).abs() < 0.1);
        assert!(f.pos(&w, "RightFoot").z < f.pos(&w, "RightLowerLeg").z - 0.1);
        // Finger curl.
        let w = with(&|p| p.muscles[mi("Left Index 1 Stretched")] = 0.0);
        let d = dir(&w, "LeftIndexProximal", "LeftIndexIntermediate");
        assert!((d.y + 33f32.to_radians().sin()).abs() < 0.01, "{d:?}");
        // Spine leans: Left-Right +1 to the right, Front-Back +1 backwards.
        let w = with(&|p| p.muscles[mi("Spine Left-Right")] = 1.0);
        assert!(f.pos(&w, "Head").x < -0.05);
        let w = with(&|p| p.muscles[mi("Spine Front-Back")] = 1.0);
        assert!(f.pos(&w, "Head").z - f.pos(&w, "Hips").z < -0.05);
        // Head Turn +1 turns the face to the right.
        let w = with(&|p| p.muscles[mi("Head Turn Left-Right")] = 1.0);
        let head = f.skel.joints.iter().position(|j| j.name == "Head").unwrap();
        let facing = w[head].to_scale_rotation_translation().1 * Vec3::Z;
        assert!((facing.x + 40f32.to_radians().sin()).abs() < 1e-3, "{facing:?}");
    }

    #[test]
    fn random_in_limit_poses_round_trip() {
        let f = fixture(Opts::default());
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let (n, hj) = (f.skel.joints.len(), f.mr.joint(HumanBone::Hips).unwrap());
        for case in 0..300 {
            let mut p = f.rest();
            for i in 0..MUSCLE_COUNT {
                p.muscles[i] = rng.signed() * 0.95;
            }
            p.root.position += Vec3::new(rng.signed(), rng.signed() * 0.3, rng.signed()) * 0.2;
            p.root.rotation = Quat::from_euler(glam::EulerRot::YXZ, rng.signed() * 0.6, rng.signed() * 0.3, rng.signed() * 0.3);
            let (mut r, mut t) = f.mr.reference_locals();
            f.mr.pose_from_muscles(&p, &mut r, &mut t);
            assert_eq!(r.len(), n);
            let back = f.mr.muscles_from_pose(&r, t[hj]);
            for i in 0..MUSCLE_COUNT {
                // Bones the rig lacks (eyes, jaw, thumb, most fingers) have no joint: their muscles stay 0.
                if f.mr.muscle_joint(i).is_none() {
                    continue;
                }
                let m = &muscles()[i];
                let has_axis = f.mr.slot[m.bone as usize][m.axis as usize] == Some(i);
                if has_axis {
                    assert!((back.muscles[i] - p.muscles[i]).abs() < 3e-3, "case {case} {}: {} vs {}", m.name, back.muscles[i], p.muscles[i]);
                }
            }
            assert!((back.root.position - p.root.position).length() < 2e-3, "case {case}: {:?} vs {:?}", back.root.position, p.root.position);
            assert!(back.root.rotation.angle_between(p.root.rotation) < 2e-3, "case {case}");
        }
    }

    #[test]
    fn values_outside_the_limits_clamp() {
        let f = fixture(Opts::default());
        let i = mi("Left Arm Down-Up");
        let mut hi = f.rest();
        hi.muscles[i] = 3.0;
        let mut at = f.rest();
        at.muscles[i] = 1.0;
        let (a, b) = (f.fk(&hi), f.fk(&at));
        assert!((f.pos(&a, "LeftLowerArm") - f.pos(&b, "LeftLowerArm")).length() < 1e-5, "3.0 plays as 1.0");
        let (mut r, mut t) = f.mr.reference_locals();
        let hj = f.mr.joint(HumanBone::Hips).unwrap();
        f.mr.pose_from_muscles(&hi, &mut r, &mut t);
        assert!((f.mr.muscles_from_pose(&r, t[hj]).muscles[i] - 1.0).abs() < 1e-3);
        // A pose beyond the range: forearm twist of 170 degrees against a 90 degree limit.
        let (mut r, t) = f.mr.reference_locals();
        let la = f.mr.joint(HumanBone::LeftLowerArm).unwrap();
        r[la] *= Quat::from_rotation_x(170f32.to_radians());
        let p = f.mr.muscles_from_pose(&r, t[hj]);
        assert!(p.muscles.iter().all(|v| (-1.0..=1.0).contains(v)));
        assert!(p.muscles[mi("Left Forearm Twist In-Out")].abs() > 0.999, "{}", p.muscles[mi("Left Forearm Twist In-Out")]);
    }

    #[test]
    fn the_root_is_the_centre_of_mass_over_the_human_scale() {
        let a = fixture(Opts::default());
        let b = fixture(Opts { s: 1.4, ..Opts::default() });
        assert!(a.mr.human_scale() > 0.5 && a.mr.human_scale() < 1.2, "{}", a.mr.human_scale());
        assert!((b.mr.human_scale() / a.mr.human_scale() - 1.4).abs() < 1e-3);
        // The same normalised root is the same standing pose in both.
        assert!((a.rest().root.position - b.rest().root.position).length() < 1e-3);
        // Lowering the root by 10 percent of the human scale lowers the hips by that much.
        let mut p = a.rest();
        let before = a.pos(&a.fk(&p), "Hips").y;
        p.root.position.y -= 0.1;
        let after = a.pos(&a.fk(&p), "Hips").y;
        assert!((before - after - 0.1 * a.mr.human_scale()).abs() < 1e-3);
    }

    /// A crouching, reaching source performance on `src` with its feet planted.
    fn performance(src: &Fixture) -> MusclePose {
        let mut p = src.rest();
        let reference = {
            let (r, t) = src.mr.reference_locals();
            src.mr.muscles_from_pose(&r, t[src.mr.joint(HumanBone::Hips).unwrap()])
        };
        for side in ["Left", "Right"] {
            p.muscles[mi(&format!("{side} Upper Leg Front-Back"))] = -0.4;
            p.muscles[mi(&format!("{side} Lower Leg Stretch"))] = 0.3;
        }
        p.root.position.y -= 0.12;
        let mut g = reference.goals.unwrap();
        // Hands reach forward at 80 percent of the arm.
        g.relative[0] = Vec3::new(0.3, -0.2, 0.7).normalize() * 0.8;
        g.relative[1] = Vec3::new(-0.3, -0.2, 0.7).normalize() * 0.8;
        p.goals = Some(g);
        // Bake it through the source rig so the goals are what the source's skeleton does.
        let (mut r, mut t) = src.mr.reference_locals();
        src.mr.pose_from_muscles(&p, &mut r, &mut t);
        src.mr.muscles_from_pose(&r, t[src.mr.joint(HumanBone::Hips).unwrap()])
    }

    #[test]
    fn retargeting_between_proportions_keeps_feet_planted_and_hands_proportional() {
        let a = fixture(Opts::default());
        let src = performance(&a);
        let goals = src.goals.unwrap();
        let w = a.fk(&src);
        let ankle = |f: &Fixture, w: &[Mat4], side: &str| f.pos(w, &format!("{side}Foot")).y;
        assert!((ankle(&a, &w, "Left") - 0.08).abs() < 4e-3 && (ankle(&a, &w, "Right") - 0.08).abs() < 4e-3, "the source performance has its feet planted");
        let arm_ratio = |f: &Fixture, w: &[Mat4], side: &str| {
            let (u, m, h) = (f.pos(w, &format!("{side}UpperArm")), f.pos(w, &format!("{side}LowerArm")), f.pos(w, &format!("{side}Hand")));
            (h - u) / ((m - u).length() + (h - m).length())
        };
        let src_hand = arm_ratio(&a, &w, "Left");
        for (label, o) in [
            ("tall", Opts { s: 1.4, ..Opts::default() }),
            ("short", Opts { s: 0.7, ..Opts::default() }),
            ("long arms", Opts { arm: 1.35, ..Opts::default() }),
            ("long legs", Opts { leg: 1.2, ..Opts::default() }),
        ] {
            let t = fixture(o);
            let w = t.fk(&src);
            let k = t.mr.human_scale() / a.mr.human_scale();
            for side in ["Left", "Right"] {
                let foot = t.pos(&w, &format!("{side}Foot"));
                let want_y = 0.08 * o.s;
                assert!((foot.y - want_y).abs() < 0.006 * o.s.max(1.0), "{label} {side}: foot at {} wants {want_y}", foot.y);
                let src_foot = a.pos(&a.fk(&src), &format!("{side}Foot"));
                let scaled = src_foot * k;
                assert!((foot.x - scaled.x).abs() < 0.03 && (foot.z - scaled.z).abs() < 0.03, "{label} {side}: foot {foot:?} vs scaled {scaled:?}");
            }
            let hand = arm_ratio(&t, &w, "Left");
            assert!((hand - src_hand).length() < 0.02, "{label}: hand {hand:?} vs {src_hand:?}");
            // Hand goals are fractions of the limb, so a longer arm reaches further in metres.
            if o.arm > 1.0 {
                let reach = |f: &Fixture, w: &[Mat4]| (f.pos(w, "LeftHand") - f.pos(w, "LeftUpperArm")).length();
                assert!(reach(&t, &w) > reach(&a, &a.fk(&src)) * 1.2);
            }
            // The goals do the aligning: muscles alone leave the long-legged rig's feet off the ground.
            if label == "long legs" {
                let mut bare = src;
                bare.goals = None;
                let wb = t.fk(&bare);
                let err = |w: &[Mat4]| (t.pos(w, "LeftFoot").y - 0.08 * o.s).abs();
                assert!(err(&wb) > err(&w), "no goals {} vs goals {}", err(&wb), err(&w));
            }
        }
        let _ = goals;
    }

    #[test]
    fn an_unreachable_grounded_foot_lowers_the_root() {
        let a = fixture(Opts::default());
        // A standing source retargeted to a rig whose legs are shorter than the scaled hip height.
        let short_legs = fixture(Opts { leg: 0.7, ..Opts::default() });
        let src = performance(&a);
        let w = short_legs.fk(&src);
        let y = short_legs.pos(&w, "LeftFoot").y;
        assert!((y - 0.08).abs() < 0.01, "foot on the ground: {y}");
    }

    #[test]
    fn missing_optional_bones_fold_into_their_parent() {
        let full = fixture(Opts::default());
        let bare = fixture(Opts { chest: false, upper_chest: false, neck: false, shoulders: false, toes: false, ..Opts::default() });
        for b in [HumanBone::Chest, HumanBone::UpperChest, HumanBone::Neck, HumanBone::LeftShoulder, HumanBone::LeftToes] {
            assert!(!bare.mr.has(b) && full.mr.has(b));
        }
        let mut p = full.rest();
        p.muscles[mi("Spine Front-Back")] = 0.3;
        p.muscles[mi("Chest Front-Back")] = 0.4;
        p.muscles[mi("UpperChest Front-Back")] = 0.2;
        p.muscles[mi("Neck Nod Down-Up")] = 0.5;
        p.muscles[mi("Head Nod Down-Up")] = 0.1;
        p.muscles[mi("Left Shoulder Down-Up")] = 1.0;
        p.muscles[mi("Left Shoulder Front-Back")] = 0.5;
        p.muscles[mi("Left Arm Down-Up")] = 0.2;
        p.muscles[mi("Left Arm Front-Back")] = 0.1;
        p.muscles[mi("Left Toes Up-Down")] = 1.0;
        let mut m = p.muscles;
        bare.mr.fold_missing(&mut m);
        assert!((m[mi("Spine Front-Back")] - 0.8).abs() < 1e-5, "0.3 + 0.4 + half of 0.2");
        assert!(m[mi("Chest Front-Back")] == 0.0 && m[mi("UpperChest Front-Back")] == 0.0 && m[mi("Neck Nod Down-Up")] == 0.0);
        assert!((m[mi("Head Nod Down-Up")] - 0.6).abs() < 1e-5);
        assert!((m[mi("Left Arm Down-Up")] - 0.35).abs() < 1e-5, "0.2 + 0.15 of the shoulder");
        assert!((m[mi("Left Arm Front-Back")] - (0.1 + 0.28125 * 0.5)).abs() < 1e-5);
        // A rig that has all of them is untouched.
        let mut same = p.muscles;
        full.mr.fold_missing(&mut same);
        assert_eq!(same, p.muscles);
        // End to end: the bare rig's pose analyses to the folded values; no panic without toes.
        let (mut r, mut t) = bare.mr.reference_locals();
        bare.mr.pose_from_muscles(&p, &mut r, &mut t);
        let back = bare.mr.muscles_from_pose(&r, t[bare.mr.joint(HumanBone::Hips).unwrap()]);
        assert!((back.muscles[mi("Spine Front-Back")] - 0.8).abs() < 3e-3, "{}", back.muscles[mi("Spine Front-Back")]);
        assert!((back.muscles[mi("Head Nod Down-Up")] - 0.6).abs() < 3e-3);
        assert!(back.muscles[mi("Chest Front-Back")] == 0.0);
        // The spine joint bent 0.8 of 40 degrees backwards.
        let sj = bare.mr.joint(HumanBone::Spine).unwrap();
        assert!((r[sj].angle_between(Quat::IDENTITY).to_degrees() - 32.0).abs() < 0.3, "{}", r[sj].angle_between(Quat::IDENTITY).to_degrees());
    }

    #[test]
    fn mirroring_a_muscle_pose_mirrors_the_skeleton() {
        let f = fixture(Opts::default());
        let mut rng = Rng(0x1234_5678_9abc_def1);
        let mut p = f.rest();
        for i in 0..MUSCLE_COUNT {
            p.muscles[i] = rng.signed() * 0.9;
        }
        p.root.position += Vec3::new(0.3, 0.0, 0.2);
        p.root.rotation = Quat::from_rotation_y(0.4);
        let mut m = p;
        m.mirror();
        let (a, b) = (f.fk(&p), f.fk(&m));
        for j in &f.skel.joints {
            let bone = HumanBone::from_name(&j.name).unwrap();
            let partner = other_side(bone);
            let (from, to) = (f.pos(&a, partner.name()), f.pos(&b, &j.name));
            let flipped = Vec3::new(-from.x, from.y, from.z);
            assert!((to - flipped).length() < 2e-3, "{}: {to:?} vs {flipped:?}", j.name);
        }
        let mut twice = m;
        twice.mirror();
        for i in 0..MUSCLE_COUNT {
            assert!((twice.muscles[i] - p.muscles[i]).abs() < 1e-6);
        }
        assert!(twice.root.rotation.angle_between(p.root.rotation) < 1e-5);
    }

    #[test]
    fn lerp_blends_muscles_linearly() {
        let f = fixture(Opts::default());
        let (mut a, mut b) = (f.rest(), f.rest());
        a.muscles[3] = -0.5;
        b.muscles[3] = 0.7;
        a.root.position = Vec3::ZERO;
        b.root.position = Vec3::new(1.0, 0.0, 0.0);
        let m = a.lerp(&b, 0.25);
        assert!((m.muscles[3] - (-0.5 + 0.25 * 1.2)).abs() < 1e-6);
        assert!((m.root.position.x - 0.25).abs() < 1e-6);
    }

    #[test]
    fn per_avatar_limits_override_the_defaults_and_reset() {
        let (skel, mut rig) = humanoid(Opts::default());
        let i = mi("Left Arm Down-Up");
        assert!(!rig.muscles.is_modified(i));
        rig.muscles.set_limit(i, -30.0, 50.0);
        assert!(rig.muscles.is_modified(i));
        assert_eq!(rig.muscles.limit(i), (-30.0, 50.0));
        let custom = MuscleRig::new(&skel, &rig).unwrap();
        assert!((custom.limit_degrees(i).1 - 50.0).abs() < 1e-4);
        // The reference arm sits 40 degrees over the neutral: 40 of 50 now.
        assert!((custom.rest_pose().muscles[i] - 0.8).abs() < 1e-3, "{}", custom.rest_pose().muscles[i]);
        // Limits clamp to min <= 0 <= max.
        rig.muscles.set_limit(i, 20.0, -5.0);
        assert_eq!(rig.muscles.limit(i), (0.0, 0.0));
        // Setting the defaults again clears the override; Reset does too.
        rig.muscles.set_limit(i, -60.0, 100.0);
        assert!(!rig.muscles.is_modified(i));
        rig.muscles.set_limit(i, -10.0, 10.0);
        rig.muscles.reset(i);
        assert!(rig.muscles.limits.is_empty());
    }

    #[test]
    fn rig_import_settings_serialise_and_old_files_load() {
        let (_, mut rig) = humanoid(Opts::default());
        rig.muscles.set_limit(mi("Right Hand In-Out"), -20.0, 30.0);
        let text = ron::ser::to_string(&rig).unwrap();
        let back: RigImport = ron::from_str(&text).unwrap();
        assert_eq!(back, rig);
        assert_eq!(back.muscles.limit(mi("Right Hand In-Out")), (-20.0, 30.0));
        let old: RigImport = ron::from_str("(animation_type: Humanoid, human: [(Hips, \"Hips\")])").unwrap();
        assert!(old.muscles.limits.is_empty());
    }

    #[test]
    fn baking_resamples_and_keeps_the_fallback() {
        let f = fixture(Opts::default());
        let arm = f.skel.joints.iter().position(|j| j.name == "LeftUpperArm").unwrap();
        let up = Quat::from_rotation_z(50f32.to_radians());
        let clip = AnimationClip {
            name: "Wave".into(),
            duration: 1.0,
            channels: vec![AnimChannel { joint: arm, path: ChannelPath::Rotation, times: vec![0.0, 1.0], vec_values: vec![], quat_values: vec![Quat::IDENTITY, up], extra: None }],
            morph_tracks: Vec::new(), playback: None,
        };
        let baked = bake_muscle_clip(&f.skel, &f.mr, &clip, 30.0);
        assert_eq!(baked.kind, ClipKind::Muscle);
        assert_eq!(baked.fallback, "Wave");
        assert!((baked.length - 1.0).abs() < 1e-6 && baked.frame_rate == 30.0);
        let spine = baked.prop_track_index("", TrackProperty::Muscle, "Spine Front-Back", 0).unwrap();
        assert_eq!(baked.tracks[spine].keys.len(), 1, "an unchanging muscle is one key");
        let arm_track = baked.prop_track_index("", TrackProperty::Muscle, "Left Arm Down-Up", 0).unwrap();
        assert_eq!(baked.tracks[arm_track].keys.len(), 31);
        // The baked frame equals the source frame's muscles.
        let rest = f.rest();
        let p = sample_muscle_clip(&baked, 0.5, &rest);
        let locals = clip.sample(&f.skel, 0.5);
        let rot: Vec<Quat> = locals.iter().map(|m| m.to_scale_rotation_translation().1).collect();
        let want = f.mr.muscles_from_pose(&rot, locals[0].w_axis.truncate());
        for i in 0..MUSCLE_COUNT {
            assert!((p.muscles[i] - want.muscles[i]).abs() < 2e-3, "{}", muscles()[i].name);
        }
        assert!(p.goals.is_some(), "the baked clip carries goals");
        // The end of the clip is the raised arm: about 50 of 100 degrees more Down-Up.
        let end = sample_muscle_clip(&AnimClip { looping: false, ..baked.clone() }, 5.0, &rest);
        let rise = end.muscles[mi("Left Arm Down-Up")] - rest.muscles[mi("Left Arm Down-Up")];
        assert!(rise > 0.45 && rise < 0.56, "{rise}");
        // A muscle clip survives RON.
        let text = ron::ser::to_string_pretty(&baked, ron::ser::PrettyConfig::default()).unwrap();
        let back: AnimClip = ron::from_str(&text).unwrap();
        assert_eq!(back, baked);
    }

    #[test]
    fn a_clip_that_drives_some_muscles_leaves_the_rest_to_the_base() {
        let f = fixture(Opts::default());
        let mut clip = AnimClip { kind: ClipKind::Muscle, ..Default::default() };
        let mut p = f.rest();
        p.muscles[mi("Jaw Close")] = 0.4;
        p.driven = 1 << mi("Jaw Close");
        set_muscle_keys(&mut clip, 0.0, &p);
        let bind = MuscleBinding::new(&clip);
        let mut base = f.rest();
        base.muscles[5] = 0.77;
        let s = bind.sample(&clip, 0.0, &base);
        assert_eq!(s.driven, 1 << mi("Jaw Close"));
        assert!((s.muscles[mi("Jaw Close")] - 0.4).abs() < 1e-6 && s.muscles[5] == 0.77);
        assert!(s.goals.is_none());
    }
}

