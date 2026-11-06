//! Humanoid bone set, persisted rig data and per-bone rig validation.
//!
//! [`HumanBone`] is the 55-bone humanoid set (Unity/VRM-compatible names) with a
//! parent table ([`PARENT`]) and the 15 required bones ([`REQUIRED`]).
//! [`RigImport`] is the rig description stored in a model's `.meta`: animation
//! type, avatar setup, root node, bone map and reference pose. [`validate_rig`]
//! checks a bone map against a skeleton and returns one message per problem.

use crate::Skeleton;
use glam::Vec3;
use serde::{Deserialize, Serialize};

/// Number of humanoid bones.
pub const HUMAN_BONE_COUNT: usize = 55;
/// Number of bones a rig must map to count as humanoid.
pub const REQUIRED_BONE_COUNT: usize = 15;

/// One of the 55 humanoid bones, in the stable order used for [`PARENT`] and
/// [`REQUIRED`] (body, then 15 left and 15 right finger bones, then UpperChest).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HumanBone {
    Hips,
    LeftUpperLeg,
    RightUpperLeg,
    LeftLowerLeg,
    RightLowerLeg,
    LeftFoot,
    RightFoot,
    Spine,
    Chest,
    Neck,
    Head,
    LeftShoulder,
    RightShoulder,
    LeftUpperArm,
    RightUpperArm,
    LeftLowerArm,
    RightLowerArm,
    LeftHand,
    RightHand,
    LeftToes,
    RightToes,
    LeftEye,
    RightEye,
    Jaw,
    LeftThumbProximal,
    LeftThumbIntermediate,
    LeftThumbDistal,
    LeftIndexProximal,
    LeftIndexIntermediate,
    LeftIndexDistal,
    LeftMiddleProximal,
    LeftMiddleIntermediate,
    LeftMiddleDistal,
    LeftRingProximal,
    LeftRingIntermediate,
    LeftRingDistal,
    LeftLittleProximal,
    LeftLittleIntermediate,
    LeftLittleDistal,
    RightThumbProximal,
    RightThumbIntermediate,
    RightThumbDistal,
    RightIndexProximal,
    RightIndexIntermediate,
    RightIndexDistal,
    RightMiddleProximal,
    RightMiddleIntermediate,
    RightMiddleDistal,
    RightRingProximal,
    RightRingIntermediate,
    RightRingDistal,
    RightLittleProximal,
    RightLittleIntermediate,
    RightLittleDistal,
    UpperChest,
}

use HumanBone as B;

/// Every bone, indexable by `bone as usize`.
pub const ALL: [HumanBone; HUMAN_BONE_COUNT] = [
    B::Hips, B::LeftUpperLeg, B::RightUpperLeg, B::LeftLowerLeg, B::RightLowerLeg, B::LeftFoot, B::RightFoot,
    B::Spine, B::Chest, B::Neck, B::Head, B::LeftShoulder, B::RightShoulder, B::LeftUpperArm, B::RightUpperArm,
    B::LeftLowerArm, B::RightLowerArm, B::LeftHand, B::RightHand, B::LeftToes, B::RightToes, B::LeftEye,
    B::RightEye, B::Jaw,
    B::LeftThumbProximal, B::LeftThumbIntermediate, B::LeftThumbDistal,
    B::LeftIndexProximal, B::LeftIndexIntermediate, B::LeftIndexDistal,
    B::LeftMiddleProximal, B::LeftMiddleIntermediate, B::LeftMiddleDistal,
    B::LeftRingProximal, B::LeftRingIntermediate, B::LeftRingDistal,
    B::LeftLittleProximal, B::LeftLittleIntermediate, B::LeftLittleDistal,
    B::RightThumbProximal, B::RightThumbIntermediate, B::RightThumbDistal,
    B::RightIndexProximal, B::RightIndexIntermediate, B::RightIndexDistal,
    B::RightMiddleProximal, B::RightMiddleIntermediate, B::RightMiddleDistal,
    B::RightRingProximal, B::RightRingIntermediate, B::RightRingDistal,
    B::RightLittleProximal, B::RightLittleIntermediate, B::RightLittleDistal,
    B::UpperChest,
];

/// The human parent of each bone (`PARENT[bone as usize]`); `None` for Hips.
/// Optional bones can be unmapped, so walk up with [`HumanBone::mapped_ancestor`].
/// Neck and both shoulders hang off UpperChest; the spine chain is
/// Spine, Chest, UpperChest.
pub const PARENT: [Option<HumanBone>; HUMAN_BONE_COUNT] = [
    None,                    // Hips
    Some(B::Hips),           // LeftUpperLeg
    Some(B::Hips),           // RightUpperLeg
    Some(B::LeftUpperLeg),   // LeftLowerLeg
    Some(B::RightUpperLeg),  // RightLowerLeg
    Some(B::LeftLowerLeg),   // LeftFoot
    Some(B::RightLowerLeg),  // RightFoot
    Some(B::Hips),           // Spine
    Some(B::Spine),          // Chest
    Some(B::UpperChest),     // Neck
    Some(B::Neck),           // Head
    Some(B::UpperChest),     // LeftShoulder
    Some(B::UpperChest),     // RightShoulder
    Some(B::LeftShoulder),   // LeftUpperArm
    Some(B::RightShoulder),  // RightUpperArm
    Some(B::LeftUpperArm),   // LeftLowerArm
    Some(B::RightUpperArm),  // RightLowerArm
    Some(B::LeftLowerArm),   // LeftHand
    Some(B::RightLowerArm),  // RightHand
    Some(B::LeftFoot),       // LeftToes
    Some(B::RightFoot),      // RightToes
    Some(B::Head),           // LeftEye
    Some(B::Head),           // RightEye
    Some(B::Head),           // Jaw
    Some(B::LeftHand), Some(B::LeftThumbProximal), Some(B::LeftThumbIntermediate),
    Some(B::LeftHand), Some(B::LeftIndexProximal), Some(B::LeftIndexIntermediate),
    Some(B::LeftHand), Some(B::LeftMiddleProximal), Some(B::LeftMiddleIntermediate),
    Some(B::LeftHand), Some(B::LeftRingProximal), Some(B::LeftRingIntermediate),
    Some(B::LeftHand), Some(B::LeftLittleProximal), Some(B::LeftLittleIntermediate),
    Some(B::RightHand), Some(B::RightThumbProximal), Some(B::RightThumbIntermediate),
    Some(B::RightHand), Some(B::RightIndexProximal), Some(B::RightIndexIntermediate),
    Some(B::RightHand), Some(B::RightMiddleProximal), Some(B::RightMiddleIntermediate),
    Some(B::RightHand), Some(B::RightRingProximal), Some(B::RightRingIntermediate),
    Some(B::RightHand), Some(B::RightLittleProximal), Some(B::RightLittleIntermediate),
    Some(B::Chest),          // UpperChest
];

/// Bones a rig must map to be humanoid (15): Hips, Spine, Head and, per side,
/// upper/lower leg, foot, upper/lower arm and hand.
pub const REQUIRED: [bool; HUMAN_BONE_COUNT] = {
    let mut r = [false; HUMAN_BONE_COUNT];
    let req = [
        B::Hips, B::Spine, B::Head,
        B::LeftUpperLeg, B::RightUpperLeg, B::LeftLowerLeg, B::RightLowerLeg, B::LeftFoot, B::RightFoot,
        B::LeftUpperArm, B::RightUpperArm, B::LeftLowerArm, B::RightLowerArm, B::LeftHand, B::RightHand,
    ];
    let mut i = 0;
    while i < req.len() {
        r[req[i] as usize] = true;
        i += 1;
    }
    r
};

impl HumanBone {
    /// Unity-style name (`LeftUpperArm`).
    pub fn name(self) -> &'static str {
        const NAMES: [&str; HUMAN_BONE_COUNT] = [
            "Hips", "LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg", "LeftFoot", "RightFoot",
            "Spine", "Chest", "Neck", "Head", "LeftShoulder", "RightShoulder", "LeftUpperArm", "RightUpperArm",
            "LeftLowerArm", "RightLowerArm", "LeftHand", "RightHand", "LeftToes", "RightToes", "LeftEye",
            "RightEye", "Jaw",
            "LeftThumbProximal", "LeftThumbIntermediate", "LeftThumbDistal",
            "LeftIndexProximal", "LeftIndexIntermediate", "LeftIndexDistal",
            "LeftMiddleProximal", "LeftMiddleIntermediate", "LeftMiddleDistal",
            "LeftRingProximal", "LeftRingIntermediate", "LeftRingDistal",
            "LeftLittleProximal", "LeftLittleIntermediate", "LeftLittleDistal",
            "RightThumbProximal", "RightThumbIntermediate", "RightThumbDistal",
            "RightIndexProximal", "RightIndexIntermediate", "RightIndexDistal",
            "RightMiddleProximal", "RightMiddleIntermediate", "RightMiddleDistal",
            "RightRingProximal", "RightRingIntermediate", "RightRingDistal",
            "RightLittleProximal", "RightLittleIntermediate", "RightLittleDistal",
            "UpperChest",
        ];
        NAMES[self as usize]
    }

    /// VRM-style name (`leftUpperArm`): the Unity name with a lower-case first letter.
    pub fn vrm_name(self) -> String {
        let n = self.name();
        let mut c = n.chars();
        c.next().map(|f| f.to_lowercase().chain(c).collect()).unwrap_or_default()
    }

    /// Parse a Unity or VRM bone name (case-insensitive; ignores spaces and `_`).
    pub fn from_name(s: &str) -> Option<Self> {
        let key: String = s.chars().filter(|c| !matches!(c, ' ' | '_' | '-')).collect::<String>().to_lowercase();
        ALL.iter().copied().find(|b| b.name().to_lowercase() == key)
    }

    /// The direct human parent (`PARENT`), `None` for Hips.
    pub fn parent(self) -> Option<HumanBone> {
        PARENT[self as usize]
    }

    /// One of the 15 bones every humanoid must map.
    pub fn is_required(self) -> bool {
        REQUIRED[self as usize]
    }

    /// The nearest ancestor for which `is_mapped` holds, skipping unmapped optional bones.
    pub fn mapped_ancestor(self, is_mapped: impl Fn(HumanBone) -> bool) -> Option<HumanBone> {
        let mut p = self.parent();
        while let Some(b) = p {
            if is_mapped(b) {
                return Some(b);
            }
            p = b.parent();
        }
        None
    }
}

/// How the model's animation is treated.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnimationType {
    /// No rig configured: runtime auto-maps by bone name.
    #[default]
    None,
    Generic,
    Humanoid,
}

/// Where the avatar definition comes from.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AvatarSetup {
    #[default]
    CreateFromThisModel,
    /// Copy the rig of another model (asset path or GUID).
    CopyFrom(String),
}

/// One bone of the stored reference pose (local TRS, by joint name).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PoseBone {
    pub name: String,
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

impl Default for PoseBone {
    fn default() -> Self {
        Self { name: String::new(), translation: [0.0; 3], rotation: [0.0, 0.0, 0.0, 1.0], scale: [1.0; 3] }
    }
}

/// Rig description persisted in a model's `.meta` and attached to the imported
/// [`Skeleton`], so runtime code reads it instead of re-mapping bone names.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RigImport {
    pub animation_type: AnimationType,
    pub avatar_setup: AvatarSetup,
    /// Root node (joint name) for Generic rigs; empty = skeleton root.
    pub root_node: String,
    /// Human bone to joint name.
    pub human: Vec<(HumanBone, String)>,
    /// Reference pose (T-pose) as local transforms; empty = the skeleton's rest pose.
    pub reference_pose: Vec<PoseBone>,
    /// Computed from names rather than authored by the user.
    pub auto_generated: bool,
    /// The bone map was read from the model file (a VRM humanoid table) and
    /// not guessed from names. Such a rig is still `auto_generated` (refreshed
    /// on reimport, replaced by any edit), but it outranks the auto-mapper.
    pub from_file: bool,
    /// Muscle limits that differ from the defaults ([`crate::muscle`]).
    pub muscles: crate::muscle::MuscleSettings,
    /// Always store the exact canonical T-pose (Biped Pose) as the reference
    /// pose, instead of the closer of bind and file pose made valid. Honoured
    /// whenever the reference pose is recomputed (reimport, Enforce T-Pose).
    pub force_tpose: bool,
    /// Humanoid Oversampling: the muscle bake of a clip samples at this
    /// multiple (1, 2, 4 or 8; 0 reads as 1) of the 30 Hz base rate, so fast
    /// motion and Hermite-kept curves are followed more closely at the cost
    /// of a larger muscle clip. Not stored with the rig: it is the Animation
    /// section's `humanoid_oversampling`, copied here when the model loads.
    #[serde(skip)]
    pub humanoid_oversampling: u8,
}

impl RigImport {
    /// The oversampling factor, 1/2/4/8.
    pub fn oversampling(&self) -> u32 {
        match self.humanoid_oversampling {
            0 | 1 => 1,
            2 | 3 => 2,
            4..=7 => 4,
            _ => 8,
        }
    }

    /// Joint name mapped to `bone`.
    pub fn joint_name(&self, bone: HumanBone) -> Option<&str> {
        self.human.iter().find(|(b, _)| *b == bone).map(|(_, n)| n.as_str())
    }

    /// Whether all 15 required bones are mapped.
    pub fn has_required_bones(&self) -> bool {
        ALL.iter().filter(|b| b.is_required()).all(|b| self.joint_name(*b).is_some())
    }

    /// Humanoid only when the animation type says so and every required bone
    /// is mapped and valid against `skel`.
    pub fn is_humanoid(&self, skel: &Skeleton) -> bool {
        self.animation_type == AnimationType::Humanoid && self.has_required_bones() && validate_rig(skel, &self.human).is_empty()
    }

    /// Whether this import carries a usable bone map for the runtime
    /// (otherwise callers fall back to name matching).
    pub fn is_configured(&self) -> bool {
        self.animation_type == AnimationType::Humanoid && !self.human.is_empty()
    }
}

/// A problem found by [`validate_rig`].
#[derive(Clone, Debug, PartialEq)]
pub struct RigIssue {
    /// The bone the message is about; `None` for whole-rig problems.
    pub bone: Option<HumanBone>,
    pub message: String,
}

fn issue(bone: Option<HumanBone>, message: String) -> RigIssue {
    RigIssue { bone, message }
}

fn world_pos(skel: &Skeleton, j: usize) -> Vec3 {
    skel.joints[j].inverse_bind.inverse().w_axis.truncate()
}

fn is_descendant(skel: &Skeleton, mut j: usize, ancestor: usize) -> bool {
    while let Some(p) = skel.joints[j].parent {
        if p == ancestor {
            return true;
        }
        j = p;
    }
    false
}

/// Check a human bone map against `skel`. Empty result = valid.
///
/// Per-bone messages: required bone missing, duplicate bone or joint, joint not
/// found or ambiguous, Hips without a parent, joint not under its human
/// parent, UpperChest without Chest, zero bone length.
pub fn validate_rig(skel: &Skeleton, human: &[(HumanBone, String)]) -> Vec<RigIssue> {
    let mut out = Vec::new();
    if human.is_empty() {
        out.push(issue(None, "No human bone found. Ensure both rig type match".into()));
        return out;
    }
    if skel.joints.len() <= REQUIRED_BONE_COUNT {
        out.push(issue(None, format!("Not enough bones to create human avatar (requires {REQUIRED_BONE_COUNT})")));
    }

    // Duplicate human bones.
    for (i, (b, _)) in human.iter().enumerate() {
        if human[..i].iter().any(|(o, _)| o == b) && !out.iter().any(|m| m.bone == Some(*b)) {
            out.push(issue(Some(*b), format!("Found duplicate human bone '{}'", b.name())));
        }
    }

    // Resolve joints; collect per-bone joint index.
    let mut joint_of: [Option<usize>; HUMAN_BONE_COUNT] = [None; HUMAN_BONE_COUNT];
    for (b, name) in human {
        if joint_of[*b as usize].is_some() {
            continue;
        }
        let hits: Vec<usize> = skel.joints.iter().enumerate().filter(|(_, j)| j.name == *name).map(|(i, _)| i).collect();
        match hits.as_slice() {
            [] => out.push(issue(Some(*b), format!("Transform '{name}' for human bone '{}' not found", b.name()))),
            [one] => joint_of[*b as usize] = Some(*one),
            [_, second, ..] => {
                out.push(issue(
                    Some(*b),
                    format!(
                        "Ambiguous Transform '{name}' and '{}' found in hierarchy for human bone '{}'. Transform name mapped to a human bone must be unique.",
                        skel.joints[*second].name,
                        b.name()
                    ),
                ));
            }
        }
    }

    // Same joint on several bones.
    for (i, a) in ALL.iter().enumerate() {
        let Some(ja) = joint_of[*a as usize] else { continue };
        for b in &ALL[i + 1..] {
            if joint_of[*b as usize] == Some(ja) {
                out.push(issue(
                    Some(*b),
                    format!("Found duplicate transform '{}' for human bone '{}' and '{}'", skel.joints[ja].name, a.name(), b.name()),
                ));
            }
        }
    }

    // Required bones.
    for b in ALL.iter().filter(|b| b.is_required()) {
        if !human.iter().any(|(o, _)| o == b) {
            out.push(issue(Some(*b), format!("Required human bone '{}' not found", b.name())));
        }
    }

    let mapped = |b: HumanBone| joint_of[b as usize].is_some();

    // Hips must have a parent.
    if let Some(h) = joint_of[B::Hips as usize]
        && skel.joints[h].parent.is_none()
    {
        out.push(issue(Some(B::Hips), format!("Hips bone '{}' must have a parent", skel.joints[h].name)));
    }

    // Chest before UpperChest.
    if mapped(B::UpperChest) && !mapped(B::Chest) {
        out.push(issue(Some(B::UpperChest), "Chest must be assigned before assigning UpperChest.".into()));
    }

    // Hierarchy: each joint sits under its nearest mapped human ancestor.
    for b in ALL {
        let Some(j) = joint_of[b as usize] else { continue };
        let Some(anc) = b.mapped_ancestor(mapped) else { continue };
        let aj = joint_of[anc as usize].unwrap();
        if !is_descendant(skel, j, aj) {
            out.push(issue(
                Some(b),
                format!("Transform '{}' is not a child of '{}'.", skel.joints[j].name, skel.joints[aj].name),
            ));
        }
    }

    // Zero bone length (every bone except Jaw): distance to its first mapped human child.
    for b in ALL {
        if b == B::Jaw {
            continue;
        }
        let Some(j) = joint_of[b as usize] else { continue };
        let child = ALL.iter().copied().find(|c| c.mapped_ancestor(mapped) == Some(b) && mapped(*c));
        if let Some(c) = child {
            let cj = joint_of[c as usize].unwrap();
            if cj != j && world_pos(skel, j).distance(world_pos(skel, cj)) < 1e-6 {
                out.push(issue(Some(b), format!("Transform '{}' for human bone '{}' has bone length of zero.", skel.joints[j].name, b.name())));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Joint;
    use glam::{Mat4, Quat};

    #[test]
    fn counts_and_order() {
        assert_eq!(ALL.len(), 55);
        for (i, b) in ALL.iter().enumerate() {
            assert_eq!(*b as usize, i, "{b:?}");
        }
        assert_eq!(ALL[54], B::UpperChest);
        assert_eq!(B::Jaw as usize, 23);
        assert_eq!(B::LeftThumbProximal as usize, 24);
        assert_eq!(B::RightLittleDistal as usize, 53);
    }

    #[test]
    fn parent_table() {
        assert_eq!(B::Hips.parent(), None);
        assert_eq!(B::Chest.parent(), Some(B::Spine));
        assert_eq!(B::UpperChest.parent(), Some(B::Chest));
        assert_eq!(B::Neck.parent(), Some(B::UpperChest));
        assert_eq!(B::LeftShoulder.parent(), Some(B::UpperChest));
        assert_eq!(B::RightUpperArm.parent(), Some(B::RightShoulder));
        assert_eq!(B::LeftToes.parent(), Some(B::LeftFoot));
        assert_eq!(B::Jaw.parent(), Some(B::Head));
        assert_eq!(B::LeftRingIntermediate.parent(), Some(B::LeftRingProximal));
        assert_eq!(B::RightIndexProximal.parent(), Some(B::RightHand));
        // Only Hips is a root, every parent chain ends at Hips.
        for b in ALL {
            let mut cur = b;
            let mut steps = 0;
            while let Some(p) = cur.parent() {
                cur = p;
                steps += 1;
                assert!(steps < 12);
            }
            assert_eq!(cur, B::Hips);
        }
    }

    #[test]
    fn required_set() {
        let req: Vec<_> = ALL.iter().filter(|b| b.is_required()).collect();
        assert_eq!(req.len(), REQUIRED_BONE_COUNT);
        for b in [B::Hips, B::Spine, B::Head, B::LeftHand, B::RightFoot, B::LeftLowerArm] {
            assert!(b.is_required(), "{b:?}");
        }
        for b in [B::Chest, B::UpperChest, B::Neck, B::LeftShoulder, B::LeftToes, B::LeftEye, B::Jaw, B::LeftThumbProximal] {
            assert!(!b.is_required(), "{b:?}");
        }
    }

    #[test]
    fn names() {
        assert_eq!(B::LeftUpperArm.vrm_name(), "leftUpperArm");
        assert_eq!(HumanBone::from_name("leftUpperArm"), Some(B::LeftUpperArm));
        assert_eq!(HumanBone::from_name("Left Upper_Arm"), Some(B::LeftUpperArm));
        assert_eq!(HumanBone::from_name("nope"), None);
        assert_eq!(B::Neck.mapped_ancestor(|b| b == B::Spine || b == B::Hips), Some(B::Spine));
    }

    /// Hips -> Spine -> Head chain plus the other 12 required bones in a simple tree.
    fn skel(names: &[(&str, Option<usize>)]) -> Skeleton {
        let mut joints: Vec<Joint> = Vec::new();
        for (i, (n, p)) in names.iter().enumerate() {
            let t = Vec3::new(0.0, i as f32, 0.0);
            joints.push(Joint {
                name: (*n).into(),
                parent: *p,
                inverse_bind: Mat4::from_translation(-t),
                rest_translation: t,
                rest_rotation: Quat::IDENTITY,
                rest_scale: Vec3::ONE,
            });
        }
        Skeleton { joints, rig: None, ..Default::default() }
    }

    fn full() -> (Skeleton, Vec<(HumanBone, String)>) {
        let mut spec: Vec<(&str, Option<usize>)> = vec![("root", None), ("hips", Some(0)), ("spine", Some(1)), ("head", Some(2))];
        let mut map = vec![(B::Hips, "hips"), (B::Spine, "spine"), (B::Head, "head")];
        type ChainSpec = (&'static str, [(B, &'static str); 3], usize);
        let chains: [ChainSpec; 4] = [
            ("l_leg", [(B::LeftUpperLeg, "lul"), (B::LeftLowerLeg, "lll"), (B::LeftFoot, "lf")], 1),
            ("r_leg", [(B::RightUpperLeg, "rul"), (B::RightLowerLeg, "rll"), (B::RightFoot, "rf")], 1),
            ("l_arm", [(B::LeftUpperArm, "lua"), (B::LeftLowerArm, "lla"), (B::LeftHand, "lh")], 2),
            ("r_arm", [(B::RightUpperArm, "rua"), (B::RightLowerArm, "rla"), (B::RightHand, "rh")], 2),
        ];
        for (_, bones, root) in chains {
            let mut parent = root;
            for (b, n) in bones {
                spec.push((n, Some(parent)));
                parent = spec.len() - 1;
                map.push((b, n));
            }
        }
        (skel(&spec), map.into_iter().map(|(b, n)| (b, n.to_string())).collect())
    }

    #[test]
    fn valid_rig_has_no_issues() {
        let (s, m) = full();
        let issues = validate_rig(&s, &m);
        assert!(issues.is_empty(), "{issues:?}");
        let rig = RigImport { animation_type: AnimationType::Humanoid, human: m, ..Default::default() };
        assert!(rig.has_required_bones());
        assert!(rig.is_humanoid(&s));
    }

    #[test]
    fn missing_required_bone() {
        let (s, mut m) = full();
        m.retain(|(b, _)| *b != B::LeftHand);
        let msgs: Vec<_> = validate_rig(&s, &m).into_iter().map(|i| i.message).collect();
        assert!(msgs.contains(&"Required human bone 'LeftHand' not found".to_string()), "{msgs:?}");
        let rig = RigImport { animation_type: AnimationType::Humanoid, human: m, ..Default::default() };
        assert!(!rig.has_required_bones());
        assert!(!rig.is_humanoid(&s));
    }

    #[test]
    fn messages() {
        let (s, mut m) = full();
        // Not found.
        m.push((B::Neck, "ghost".into()));
        // Duplicate transform: Jaw on the head joint.
        m.push((B::Jaw, "head".into()));
        // UpperChest without Chest.
        m.push((B::UpperChest, "spine".into()));
        let msgs: Vec<_> = validate_rig(&s, &m).into_iter().map(|i| i.message).collect();
        assert!(msgs.contains(&"Transform 'ghost' for human bone 'Neck' not found".to_string()), "{msgs:?}");
        assert!(msgs.iter().any(|x| x.starts_with("Found duplicate transform 'head' for human bone")), "{msgs:?}");
        assert!(msgs.contains(&"Chest must be assigned before assigning UpperChest.".to_string()), "{msgs:?}");
    }

    #[test]
    fn hierarchy_hips_and_duplicates() {
        let (s, mut m) = full();
        // Hand under the wrong parent: swap hand joints of the two arms.
        for e in m.iter_mut() {
            if e.0 == B::LeftHand {
                e.1 = "rh".into();
            }
        }
        let msgs: Vec<_> = validate_rig(&s, &m).into_iter().map(|i| i.message).collect();
        assert!(msgs.contains(&"Transform 'rh' is not a child of 'lla'.".to_string()), "{msgs:?}");
        // Duplicate human bone.
        let (s, mut m) = full();
        m.push((B::Head, "head".into()));
        let msgs: Vec<_> = validate_rig(&s, &m).into_iter().map(|i| i.message).collect();
        assert!(msgs.contains(&"Found duplicate human bone 'Head'".to_string()), "{msgs:?}");
        // Hips without a parent.
        let mut s2 = skel(&[("root", None), ("hips", Some(0))]);
        s2.joints[1].parent = None;
        let msgs: Vec<_> = validate_rig(&s2, &[(B::Hips, "hips".into())]).into_iter().map(|i| i.message).collect();
        assert!(msgs.contains(&"Hips bone 'hips' must have a parent".to_string()), "{msgs:?}");
        // Empty map.
        let msgs: Vec<_> = validate_rig(&s, &[]).into_iter().map(|i| i.message).collect();
        assert_eq!(msgs, vec!["No human bone found. Ensure both rig type match".to_string()]);
    }

    #[test]
    fn zero_length() {
        let (mut s, m) = full();
        // Head joint on top of the spine joint.
        s.joints[3].inverse_bind = s.joints[2].inverse_bind;
        let msgs: Vec<_> = validate_rig(&s, &m).into_iter().map(|i| i.message).collect();
        assert!(msgs.contains(&"Transform 'spine' for human bone 'Spine' has bone length of zero.".to_string()), "{msgs:?}");
    }

    #[test]
    fn serde_defaults() {
        let r: RigImport = ron::from_str("(root_node: \"x\")").unwrap();
        assert_eq!(r.animation_type, AnimationType::None);
        assert!(!r.is_configured());
        let back: RigImport = ron::from_str(&ron::to_string(&RigImport {
            animation_type: AnimationType::Humanoid,
            human: vec![(B::Hips, "h".into())],
            ..Default::default()
        }).unwrap()).unwrap();
        assert_eq!(back.joint_name(B::Hips), Some("h"));
    }

    #[test]
    fn force_tpose_defaults_to_false_and_round_trips() {
        let old: RigImport = ron::from_str("(animation_type: Humanoid)").unwrap();
        assert!(!old.force_tpose);
        let on = RigImport { force_tpose: true, ..Default::default() };
        let back: RigImport = ron::from_str(&ron::to_string(&on).unwrap()).unwrap();
        assert!(back.force_tpose);
    }
}
