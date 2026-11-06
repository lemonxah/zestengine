//! Skin rebinding for `Merge Armature`: an outfit mesh skinned to its own
//! skeleton is moved onto the avatar's skeleton so it deforms with the avatar.
//!
//! Skinned meshes bind to the joints of a model's [`Skeleton`], not to scene
//! objects, so merging the object hierarchy alone does nothing for skinning.
//! [`plan_rebind`] works on the two skeletons:
//!
//! - outfit joints are matched to avatar joints by humanoid bone (when both
//!   skeletons carry a rig) and then by name, after the component's prefix and
//!   suffix are removed and case, `_`, `.`, ` ` are ignored;
//! - an outfit joint with no partner is appended to the avatar skeleton as a
//!   child of its nearest matched ancestor, keeping its bind pose
//!   ([`ExtraJoints::Append`]), or folded into that ancestor so the avatar's
//!   skeleton does not grow ([`ExtraJoints::Fold`]); both are reported;
//! - a matched joint whose bind pose differs from the avatar's (a wider arm, an
//!   A-pose against a T-pose) gets a bind-pose correction: the vertex keeps its
//!   position relative to the joint, so the cloth lands on the avatar's body
//!   instead of where the outfit's own body was;
//! - an `align` matrix (outfit model space to avatar model space, from the
//!   hips offset and the body height ratio) moves the outfit mesh, so an
//!   outfit authored at another scale or origin still fits.
//!
//! Matched joints keep the avatar's inverse bind matrix: at rest the skinned
//! vertex is its own position (`world * inverse_bind` is the identity), so
//! remapping the indices and aligning the positions preserves the rest look.
//! The result is pure data; applying it to meshes is [`SkinRebind::apply`].

use glam::{Mat3, Mat4, Vec3};

use crate::human_rig::HumanBone;
use crate::{Joint, Skeleton};

/// How to match and place an outfit.
#[derive(Clone, Debug)]
pub struct RebindOptions {
    /// Removed from the start of an outfit joint name before matching.
    pub prefix: String,
    /// Removed from the end of an outfit joint name before matching.
    pub suffix: String,
    /// Rename unmatched joints whose name collides with an avatar joint.
    pub mangle_names: bool,
    /// Align the outfit to the avatar (hips offset and height ratio). Off keeps
    /// the outfit's own model space.
    pub auto_align: bool,
    /// What happens to outfit joints with no avatar partner.
    pub extras: ExtraJoints,
    /// Move vertices of matched joints by the difference between the outfit's and
    /// the avatar's bind pose (see the module docs). Off keeps the outfit's
    /// positions as authored.
    pub correct_bind_pose: bool,
}

impl Default for RebindOptions {
    fn default() -> Self {
        Self {
            prefix: String::new(),
            suffix: String::new(),
            mangle_names: false,
            auto_align: false,
            extras: ExtraJoints::Append,
            correct_bind_pose: true,
        }
    }
}

/// Outfit joints without an avatar partner.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExtraJoints {
    /// Add them to the avatar skeleton under their nearest matched ancestor, so
    /// skirts, capes and ears keep their own bones.
    #[default]
    Append,
    /// Give their weights to the nearest matched ancestor; the skeleton keeps its size.
    Fold,
}

/// The result of planning a rebind.
#[derive(Clone, Debug)]
pub struct SkinRebind {
    /// The avatar skeleton with the outfit's extra joints appended (existing
    /// joints keep their indices, so clips and meshes bound to the avatar stay
    /// valid).
    pub skeleton: Skeleton,
    /// New joint index per outfit joint index.
    pub joint_map: Vec<u32>,
    /// Outfit model space to avatar model space.
    pub align: Mat4,
    /// Outfit joints that matched an avatar joint: `(outfit, avatar)` names.
    pub matched: Vec<(String, String)>,
    /// Names of the joints appended to the avatar skeleton.
    pub added: Vec<String>,
    /// Outfit joints with no partner that were folded into an ancestor under
    /// [`ExtraJoints::Fold`] (or had to be, with a placeholder parent): `(joint, ancestor)`.
    pub folded: Vec<(String, String)>,
    /// Every outfit joint that found no avatar partner by humanoid bone or name,
    /// whether it was then appended or folded.
    pub missing: Vec<String>,
    pub warnings: Vec<String>,
    /// Per outfit joint: the matrix that carries a vertex (already moved by `align`) from
    /// the outfit's bind pose of that joint to the avatar's (identity for appended joints
    /// and for matching bind poses).
    pub corrections: Vec<Mat4>,
    /// Inverse-transpose linear parts of `corrections`, for normals.
    normal_mats: Vec<Mat3>,
    /// Any correction differs from the identity.
    has_corrections: bool,
}

impl SkinRebind {
    /// Outfit joints whose bind pose differs from the avatar's (their vertices are corrected).
    pub fn corrected_joints(&self) -> usize {
        self.corrections.iter().filter(|c| !approx_identity(c)).count()
    }

    /// Rewrite one vertex's joints and position data. `position`, `normal` and
    /// `tangent` are moved by `align` and then by the weight-blended bind-pose
    /// correction of the joints that influence them; zero-weight slots are
    /// pointed at joint 0 and the weights themselves are left alone.
    pub fn apply_vertex(&self, position: &mut [f32; 3], normal: &mut [f32; 3], tangent: &mut [f32; 4], joints: &mut [u32; 4], weights: &[f32; 4]) {
        let aligned = self.align != Mat4::IDENTITY;
        if aligned || self.has_corrections {
            let n_align = self.normal_matrix();
            let (mut p, mut n, mut t) = (Vec3::from(*position), Vec3::from(*normal), Vec3::new(tangent[0], tangent[1], tangent[2]));
            if aligned {
                p = self.align.transform_point3(p);
                n = (n_align * n).normalize_or_zero();
                t = (n_align * t).normalize_or_zero();
            }
            if self.has_corrections {
                let total: f32 = joints.iter().zip(weights).filter(|(_, w)| **w > 0.0).map(|(_, w)| *w).sum();
                if total > 0.0 {
                    let (mut bp, mut bn, mut bt) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
                    for (j, w) in joints.iter().zip(weights) {
                        if *w <= 0.0 {
                            continue;
                        }
                        let k = *j as usize;
                        let (c, nm) = match (self.corrections.get(k), self.normal_mats.get(k)) {
                            (Some(c), Some(nm)) => (*c, *nm),
                            _ => (Mat4::IDENTITY, Mat3::IDENTITY),
                        };
                        let w = *w / total;
                        bp += c.transform_point3(p) * w;
                        bn += nm * n * w;
                        bt += Mat3::from_mat4(c) * t * w;
                    }
                    p = bp;
                    n = bn.normalize_or_zero();
                    t = bt.normalize_or_zero();
                }
            }
            *position = p.to_array();
            *normal = n.to_array();
            *tangent = [t.x, t.y, t.z, tangent[3]];
        }
        for (j, w) in joints.iter_mut().zip(weights) {
            *j = if *w > 0.0 { self.joint_map.get(*j as usize).copied().unwrap_or(0) } else { 0 };
        }
    }

    /// Rewrite every vertex of a mesh.
    pub fn apply(&self, vertices: &mut [zestengine_render::Vertex]) {
        for v in vertices {
            let w = v.weights;
            self.apply_vertex(&mut v.position, &mut v.normal, &mut v.tangent, &mut v.joints, &w);
        }
    }

    fn normal_matrix(&self) -> glam::Mat3 {
        glam::Mat3::from_mat4(self.align).inverse().transpose()
    }
}

/// Lowercase letters and digits only, so `Upper_Arm.L` and `upperarml` agree.
fn key(name: &str) -> String {
    name.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_lowercase()).collect()
}

fn strip_affixes(name: &str, prefix: &str, suffix: &str) -> String {
    let n = if prefix.is_empty() { name } else { name.strip_prefix(prefix).unwrap_or(name) };
    (if suffix.is_empty() { n } else { n.strip_suffix(suffix).unwrap_or(n) }).to_owned()
}

fn bind_world(j: &Joint) -> Mat4 {
    j.inverse_bind.inverse()
}

fn bone_joint(sk: &Skeleton, bone: HumanBone) -> Option<usize> {
    let name = sk.rig.as_ref()?.joint_name(bone)?;
    sk.joints.iter().position(|j| j.name == name)
}

/// Model-space height (hips to head, else the full bind-pose extent) used for
/// the scale ratio, and the hips position.
fn body_metrics(sk: &Skeleton) -> Option<(Vec3, f32)> {
    let hips = bone_joint(sk, HumanBone::Hips)?;
    let hp = bind_world(&sk.joints[hips]).w_axis.truncate();
    let top = bone_joint(sk, HumanBone::Head)
        .map(|h| bind_world(&sk.joints[h]).w_axis.truncate())
        .or_else(|| sk.joints.iter().map(|j| bind_world(j).w_axis.truncate()).max_by(|a, b| a.y.total_cmp(&b.y)))?;
    let h = (top - hp).length();
    (h > 1e-4).then_some((hp, h))
}

fn approx_identity(m: &Mat4) -> bool {
    let id = Mat4::IDENTITY;
    (0..4).all(|c| (m.col(c) - id.col(c)).abs().max_element() < 1e-5)
}

/// Plan the rebind of `outfit` onto `base`.
pub fn plan_rebind(base: &Skeleton, outfit: &Skeleton, opts: &RebindOptions) -> SkinRebind {
    let mut warnings = Vec::new();
    let mut skeleton = base.clone();
    let mut joint_map: Vec<Option<u32>> = vec![None; outfit.joints.len()];
    let mut matched = Vec::new();

    // Humanoid bones first: same bone on both rigs.
    if base.rig.is_some() && outfit.rig.is_some() {
        for bone in crate::human_rig::ALL.iter().copied() {
            if let (Some(o), Some(b)) = (bone_joint(outfit, bone), bone_joint(base, bone))
                && joint_map[o].is_none()
            {
                joint_map[o] = Some(b as u32);
                matched.push((outfit.joints[o].name.clone(), base.joints[b].name.clone()));
            }
        }
    }
    // Then by name. First base joint wins on a duplicate key.
    let mut by_key: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (i, j) in base.joints.iter().enumerate() {
        by_key.entry(key(&j.name)).or_insert(i);
    }
    for (o, j) in outfit.joints.iter().enumerate() {
        if joint_map[o].is_some() {
            continue;
        }
        let k = key(&strip_affixes(&j.name, &opts.prefix, &opts.suffix));
        if let Some(&b) = by_key.get(&k).filter(|_| !k.is_empty()) {
            joint_map[o] = Some(b as u32);
            matched.push((j.name.clone(), base.joints[b].name.clone()));
        }
    }

    // Placement: hips offset and height ratio.
    let mut align = Mat4::IDENTITY;
    if opts.auto_align {
        match (body_metrics(base), body_metrics(outfit)) {
            (Some((bh, bl)), Some((oh, ol))) => {
                let s = bl / ol;
                // Scale about the outfit hips, then onto the avatar hips.
                align = Mat4::from_translation(bh) * Mat4::from_scale(Vec3::splat(s)) * Mat4::from_translation(-oh);
                if (s - 1.0).abs() > 0.02 || (bh - oh).length() > 0.01 {
                    warnings.push(format!("the outfit is aligned to the avatar (scale {s:.3}, offset {:.3} m)", (bh - oh).length()));
                }
            }
            _ => warnings.push("the outfit cannot be aligned: both skeletons need Hips and Head mapped".into()),
        }
    }

    // Unmatched joints become children of their nearest matched ancestor, in
    // an order that places parents first.
    let n = outfit.joints.len();
    let missing: Vec<String> = (0..n).filter(|&o| joint_map[o].is_none()).map(|o| outfit.joints[o].name.clone()).collect();
    // Bind-pose correction of the matched joints, from the outfit's bind pose (after the
    // alignment) to the avatar's.
    let mut corrections: Vec<Mat4> = vec![Mat4::IDENTITY; n];
    if opts.correct_bind_pose {
        for (o, c) in corrections.iter_mut().enumerate() {
            if let Some(b) = joint_map[o] {
                *c = bind_world(&base.joints[b as usize]) * (outfit.joints[o].inverse_bind * align.inverse());
            }
        }
    }
    let mut added = Vec::new();
    let mut folded: Vec<(String, String)> = Vec::new();
    let mut guard = 0;
    let mut pending: Vec<usize> = (0..n).filter(|&o| joint_map[o].is_none()).collect();
    while !pending.is_empty() && guard <= n + 1 {
        guard += 1;
        let mut next = Vec::new();
        for o in pending {
            let parent_old = outfit.joints[o].parent;
            let parent_new = match parent_old {
                None => None,
                Some(p) => match joint_map.get(p).copied().flatten() {
                    Some(m) => Some(m as usize),
                    None => {
                        next.push(o);
                        continue;
                    }
                },
            };
            if opts.extras == ExtraJoints::Fold
                && let Some(p) = parent_new
            {
                // Share the ancestor's joint and its correction.
                let anc = parent_old.unwrap_or(0);
                joint_map[o] = Some(p as u32);
                corrections[o] = corrections[anc];
                folded.push((outfit.joints[o].name.clone(), skeleton.joints[p].name.clone()));
                continue;
            }
            let j = &outfit.joints[o];
            let mut name = j.name.clone();
            if skeleton.joints.iter().any(|x| x.name == name) {
                if opts.mangle_names {
                    name = format!("{name}.outfit{}", added.len());
                } else {
                    warnings.push(format!("outfit joint '{name}' has no partner and its name is taken"));
                }
            }
            // Inverse bind in the avatar's space: vertices move by `align`.
            let inverse_bind = j.inverse_bind * align.inverse();
            let world = inverse_bind.inverse();
            let local = match parent_new {
                Some(p) => skeleton.joints[p].inverse_bind * world,
                None => world,
            };
            let (s, r, t) = local.to_scale_rotation_translation();
            let idx = skeleton.joints.len();
            skeleton.joints.push(Joint { name: name.clone(), parent: parent_new, inverse_bind, rest_translation: t, rest_rotation: r, rest_scale: s });
            joint_map[o] = Some(idx as u32);
            added.push(name);
        }
        pending = next;
    }
    // A cycle or bad parent index: park what is left on the first joint. (An unmatched root
    // has no ancestor to fold into, so under Fold it is appended like under Append.)
    let joint_map: Vec<u32> = joint_map
        .into_iter()
        .enumerate()
        .map(|(o, m)| {
            m.unwrap_or_else(|| {
                warnings.push(format!("outfit joint '{}' could not be placed; its weights go to the root", outfit.joints[o].name));
                0
            })
        })
        .collect();
    if matched.is_empty() {
        warnings.push("no outfit joint matched an avatar joint; the outfit will not follow the avatar".into());
    }
    if !folded.is_empty() {
        warnings.push(format!("{} outfit joint(s) without a partner were folded into their ancestors", folded.len()));
    }
    let normal_mats: Vec<Mat3> = corrections.iter().map(|c| Mat3::from_mat4(*c).inverse().transpose()).collect();
    let has_corrections = corrections.iter().any(|c| !approx_identity(c));
    SkinRebind { skeleton, joint_map, align, matched, added, folded, missing, warnings, corrections, normal_mats, has_corrections }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    fn joint(name: &str, parent: Option<usize>, world_t: Vec3, parent_t: Vec3) -> Joint {
        Joint {
            name: name.into(),
            parent,
            inverse_bind: Mat4::from_translation(world_t).inverse(),
            rest_translation: world_t - parent_t,
            rest_rotation: Quat::IDENTITY,
            rest_scale: Vec3::ONE,
        }
    }

    /// Hips - Spine - Head, one metre tall in the hips-to-head sense.
    fn biped(prefix: &str, scale: f32, offset: Vec3) -> Skeleton {
        let p = |y: f32| Vec3::new(0.0, y * scale, 0.0) + offset;
        Skeleton {
            joints: vec![
                joint(&format!("{prefix}Hips"), None, p(1.0), Vec3::ZERO),
                joint(&format!("{prefix}Spine"), Some(0), p(1.4), p(1.0)),
                joint(&format!("{prefix}Head"), Some(1), p(2.0), p(1.4)),
            ],
            rig: None,
            ..Default::default()
        }
    }

    #[test]
    fn matches_by_name_ignoring_prefix_and_case() {
        let base = biped("", 1.0, Vec3::ZERO);
        let outfit = biped("Outfit_", 1.0, Vec3::ZERO);
        let plan = plan_rebind(&base, &outfit, &RebindOptions { prefix: "Outfit_".into(), ..Default::default() });
        assert_eq!(plan.joint_map, vec![0, 1, 2]);
        assert!(plan.added.is_empty());
        assert_eq!(plan.skeleton.joints.len(), 3);
    }

    #[test]
    fn extra_bones_become_children_of_the_nearest_matched_ancestor() {
        let base = biped("", 1.0, Vec3::ZERO);
        let mut outfit = biped("", 1.0, Vec3::ZERO);
        // Cape hangs off the spine, its tip hangs off the cape.
        outfit.joints.push(joint("Cape", Some(1), Vec3::new(0.0, 1.3, -0.2), Vec3::new(0.0, 1.4, 0.0)));
        outfit.joints.push(joint("CapeTip", Some(3), Vec3::new(0.0, 0.9, -0.3), Vec3::new(0.0, 1.3, -0.2)));
        let plan = plan_rebind(&base, &outfit, &RebindOptions::default());
        assert_eq!(plan.added, vec!["Cape".to_owned(), "CapeTip".to_owned()]);
        assert_eq!(plan.joint_map, vec![0, 1, 2, 3, 4]);
        let cape = &plan.skeleton.joints[3];
        assert_eq!(cape.parent, Some(1));
        assert!((cape.rest_translation - Vec3::new(0.0, -0.1, -0.2)).length() < 1e-5);
        assert_eq!(plan.skeleton.joints[4].parent, Some(3));
        // Existing avatar joints are untouched.
        assert_eq!(plan.skeleton.joints[..3].len(), base.joints.len());
    }

    #[test]
    fn a_rebound_vertex_stays_put_at_rest_and_follows_the_avatar_bone() {
        let base = biped("", 1.0, Vec3::ZERO);
        let outfit = biped("", 1.0, Vec3::ZERO);
        let plan = plan_rebind(&base, &outfit, &RebindOptions::default());
        let mut v = zestengine_render::Vertex { position: [0.1, 1.4, 0.0], joints: [1, 0, 0, 0], weights: [1.0, 0.0, 0.0, 0.0], ..Default::default() };
        plan.apply(std::slice::from_mut(&mut v));
        assert_eq!(v.joints, [1, 0, 0, 0]);
        // Rest pose palette keeps the vertex where it was.
        let sk = &plan.skeleton;
        let pal = sk.palette(&sk.rest_locals());
        let p = pal[1].transform_point3(Vec3::from(v.position));
        assert!((p - Vec3::new(0.1, 1.4, 0.0)).length() < 1e-5);
        // Bend the avatar's spine 90 degrees about Z: the vertex follows.
        let mut locals = sk.rest_locals();
        locals[1] *= Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let moved = sk.palette(&locals)[1].transform_point3(Vec3::from(v.position));
        assert!((moved - p).length() > 0.05);
    }

    #[test]
    fn an_outfit_at_another_scale_and_origin_is_aligned_to_the_avatar() {
        use crate::human_rig::RigImport;
        let rig = |p: &str| RigImport {
            human: vec![(HumanBone::Hips, format!("{p}Hips")), (HumanBone::Head, format!("{p}Head"))],
            ..Default::default()
        };
        let mut base = biped("", 1.0, Vec3::ZERO);
        base.rig = Some(rig(""));
        // Twice as big, shifted sideways.
        let mut outfit = biped("O_", 2.0, Vec3::new(5.0, 0.0, 0.0));
        outfit.rig = Some(rig("O_"));
        let plan = plan_rebind(&base, &outfit, &RebindOptions { auto_align: true, ..Default::default() });
        // Hips and Head match by bone; the unmapped spine is added under the hips.
        assert_eq!(plan.joint_map, vec![0, 3, 2]);
        assert_eq!(plan.added, vec!["O_Spine".to_owned()]);
        let mut v = zestengine_render::Vertex { position: [5.0, 4.0, 0.0], joints: [2, 0, 0, 0], weights: [1.0, 0.0, 0.0, 0.0], ..Default::default() };
        plan.apply(std::slice::from_mut(&mut v));
        // The outfit's head (5, 4, 0) lands on the avatar's head (0, 2, 0).
        assert!((Vec3::from(v.position) - Vec3::new(0.0, 2.0, 0.0)).length() < 1e-4, "{:?}", v.position);
    }

    #[test]
    fn unmatched_roots_and_collisions_are_reported() {
        let base = biped("", 1.0, Vec3::ZERO);
        let outfit = Skeleton { joints: vec![joint("Zzz", None, Vec3::ONE, Vec3::ZERO)], rig: None, ..Default::default() };
        let plan = plan_rebind(&base, &outfit, &RebindOptions::default());
        assert!(plan.warnings.iter().any(|w| w.contains("no outfit joint matched")));
        assert_eq!(plan.added, vec!["Zzz".to_owned()]);
        assert_eq!(plan.skeleton.joints[3].parent, None);
    }

    /// Skin `p` with a skeleton posed by `locals` (the engine's palette convention).
    fn skin(sk: &Skeleton, locals: &[Mat4], v: &zestengine_render::Vertex) -> Vec3 {
        let pal = sk.palette(locals);
        let p = Vec3::from(v.position);
        let mut out = Vec3::ZERO;
        for (j, w) in v.joints.iter().zip(v.weights) {
            if w > 0.0 {
                out += pal[*j as usize].transform_point3(p) * w;
            }
        }
        out
    }

    /// Hips - Arm, the arm's joint at `arm_x`.
    fn arm_rig(arm_x: f32) -> Skeleton {
        Skeleton {
            joints: vec![
                joint("Hips", None, Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO),
                joint("Arm", Some(0), Vec3::new(arm_x, 1.4, 0.0), Vec3::new(0.0, 1.0, 0.0)),
            ],
            rig: None,
            ..Default::default()
        }
    }

    #[test]
    fn a_wider_outfit_arm_is_pulled_onto_the_avatars_joint_and_keeps_its_offset() {
        let base = arm_rig(0.3);
        let outfit = arm_rig(0.4);
        // A sleeve vertex 5 cm out from the outfit's arm joint, 60 / 40 between arm and hips.
        let sleeve = zestengine_render::Vertex { position: [0.45, 1.4, 0.0], joints: [1, 0, 0, 0], weights: [0.6, 0.4, 0.0, 0.0], ..Default::default() };
        let plan = plan_rebind(&base, &outfit, &RebindOptions::default());
        assert!(plan.missing.is_empty() && plan.added.is_empty());
        let mut v = sleeve;
        plan.apply(std::slice::from_mut(&mut v));
        assert_eq!(v.weights, sleeve.weights, "weights are preserved exactly");
        assert_eq!(v.joints, [1, 0, 0, 0]);
        // The hips are bound identically, so only the arm share (60 %) moved, by -10 cm.
        let rest = skin(&plan.skeleton, &plan.skeleton.rest_locals(), &v);
        assert!((rest - Vec3::new(0.45 - 0.1 * 0.6, 1.4, 0.0)).length() < 1e-5, "{rest}");
        // Pose the avatar's arm 90 degrees about Z: the sleeve turns around the AVATAR's joint.
        let mut locals = plan.skeleton.rest_locals();
        locals[1] *= Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let posed = skin(&plan.skeleton, &locals, &v);
        let arm = Vec3::new(0.3, 1.4, 0.0);
        let full_arm = Vec3::new(0.35, 1.4, 0.0) - arm;
        let want = arm + glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2) * full_arm;
        let mut pure = v;
        pure.weights = [1.0, 0.0, 0.0, 0.0];
        pure.position = [0.35, 1.4, 0.0];
        assert!((skin(&plan.skeleton, &locals, &pure) - want).length() < 1e-5, "a fully arm-weighted vertex pivots on the avatar joint");
        assert!(posed.is_finite());
        // Without the correction the vertex stays where the outfit had it.
        let mut raw = sleeve;
        let off = plan_rebind(&base, &outfit, &RebindOptions { correct_bind_pose: false, ..Default::default() });
        off.apply(std::slice::from_mut(&mut raw));
        assert_eq!(raw.position, sleeve.position);
    }

    #[test]
    fn normals_follow_the_blended_correction_rotation() {
        // The outfit's arm bind is rotated 90 degrees about Z relative to the avatar's.
        let base = arm_rig(0.3);
        let mut outfit = arm_rig(0.3);
        outfit.joints[1].inverse_bind = (Mat4::from_translation(Vec3::new(0.3, 1.4, 0.0)) * Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2)).inverse();
        let plan = plan_rebind(&base, &outfit, &RebindOptions::default());
        let mut v = zestengine_render::Vertex { position: [0.3, 1.4, 0.0], normal: [1.0, 0.0, 0.0], tangent: [0.0, 1.0, 0.0, 1.0], joints: [1, 0, 0, 0], weights: [1.0, 0.0, 0.0, 0.0], ..Default::default() };
        plan.apply(std::slice::from_mut(&mut v));
        // correction = Bind_a * invBind_o = Rz(-90): the +X normal turns to -Y, the +Y tangent to +X.
        assert!((Vec3::from(v.normal) - Vec3::new(0.0, -1.0, 0.0)).length() < 1e-5, "{:?}", v.normal);
        assert!((Vec3::new(v.tangent[0], v.tangent[1], v.tangent[2]) - Vec3::X).length() < 1e-5, "{:?}", v.tangent);
        assert_eq!(v.tangent[3], 1.0, "handedness is kept");
    }

    #[test]
    fn missing_bones_fold_into_the_nearest_ancestor_and_are_reported() {
        let base = biped("", 1.0, Vec3::ZERO);
        let mut outfit = biped("", 1.0, Vec3::ZERO);
        outfit.joints.push(joint("Cape", Some(1), Vec3::new(0.0, 1.3, -0.2), Vec3::new(0.0, 1.4, 0.0)));
        outfit.joints.push(joint("CapeTip", Some(3), Vec3::new(0.0, 0.9, -0.3), Vec3::new(0.0, 1.3, -0.2)));
        let plan = plan_rebind(&base, &outfit, &RebindOptions { extras: ExtraJoints::Fold, ..Default::default() });
        assert_eq!(plan.skeleton.joints.len(), 3, "the avatar skeleton does not grow");
        assert_eq!(plan.joint_map, vec![0, 1, 2, 1, 1], "Cape and CapeTip both land on the Spine");
        assert_eq!(plan.missing, vec!["Cape".to_owned(), "CapeTip".to_owned()]);
        assert_eq!(plan.folded, vec![("Cape".to_owned(), "Spine".to_owned()), ("CapeTip".to_owned(), "Spine".to_owned())]);
        assert!(plan.added.is_empty());
        assert!(plan.warnings.iter().any(|w| w.contains("folded")), "{:?}", plan.warnings);
        // Appending reports the same joints as missing, without folding them.
        let kept = plan_rebind(&base, &outfit, &RebindOptions::default());
        assert_eq!(kept.missing, plan.missing);
        assert!(kept.folded.is_empty() && kept.added.len() == 2);
        // A folded cape vertex follows the spine like the spine's own vertices do (same weights, same joint).
        let mut v = zestengine_render::Vertex { position: [0.0, 1.0, -0.25], joints: [4, 0, 0, 0], weights: [1.0, 0.0, 0.0, 0.0], ..Default::default() };
        plan.apply(std::slice::from_mut(&mut v));
        assert_eq!(v.joints[0], 1);
    }

    #[test]
    fn a_folded_joint_takes_the_ancestors_bind_correction() {
        let base = arm_rig(0.3);
        let mut outfit = arm_rig(0.4);
        // A cuff hangs off the outfit's arm, 20 cm further out.
        outfit.joints.push(joint("Cuff", Some(1), Vec3::new(0.6, 1.4, 0.0), Vec3::new(0.4, 1.4, 0.0)));
        let plan = plan_rebind(&base, &outfit, &RebindOptions { extras: ExtraJoints::Fold, ..Default::default() });
        let mut arm = zestengine_render::Vertex { position: [0.45, 1.4, 0.0], joints: [1, 0, 0, 0], weights: [1.0, 0.0, 0.0, 0.0], ..Default::default() };
        let mut cuff = zestengine_render::Vertex { position: [0.65, 1.4, 0.0], joints: [2, 0, 0, 0], weights: [1.0, 0.0, 0.0, 0.0], ..Default::default() };
        plan.apply(std::slice::from_mut(&mut arm));
        plan.apply(std::slice::from_mut(&mut cuff));
        assert!((arm.position[0] - 0.35).abs() < 1e-5);
        assert!((cuff.position[0] - 0.55).abs() < 1e-5, "the cuff moves with its arm: {}", cuff.position[0]);
    }

    #[test]
    fn an_unmatched_root_is_appended_even_when_folding() {
        let base = biped("", 1.0, Vec3::ZERO);
        let outfit = Skeleton { joints: vec![joint("Prop", None, Vec3::ONE, Vec3::ZERO)], rig: None, ..Default::default() };
        let plan = plan_rebind(&base, &outfit, &RebindOptions { extras: ExtraJoints::Fold, ..Default::default() });
        assert_eq!(plan.added, vec!["Prop".to_owned()]);
        assert!(plan.folded.is_empty());
    }

    /// The default avatars of the repo: one rebound onto the other. Skipped when the files are missing.
    #[test]
    fn real_avatar_rebinds_onto_another_real_avatar() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/avatars/default");
        let (Ok(male), Ok(female)) = (crate::load_model_with_meta(dir.join("male.glb")), crate::load_model_with_meta(dir.join("female.glb"))) else {
            eprintln!("default avatars not found: skipped");
            return;
        };
        let (Some(base), Some(outfit)) = (female.skeletons.first(), male.skeletons.first()) else {
            eprintln!("avatars have no skeleton: skipped");
            return;
        };
        for extras in [ExtraJoints::Append, ExtraJoints::Fold] {
            let plan = plan_rebind(base, outfit, &RebindOptions { auto_align: true, extras, ..Default::default() });
            eprintln!(
                "male onto female ({extras:?}): {} joints, {} matched, {} missing, {} added, {} folded, warnings {:?}",
                outfit.joints.len(), plan.matched.len(), plan.missing.len(), plan.added.len(), plan.folded.len(), plan.warnings
            );
            assert!(plan.matched.len() * 2 > outfit.joints.len(), "two humanoid default avatars share most bones");
            assert_eq!(plan.joint_map.len(), outfit.joints.len());
            assert!(plan.joint_map.iter().all(|&j| (j as usize) < plan.skeleton.joints.len()));
            if extras == ExtraJoints::Fold {
                assert_eq!(plan.skeleton.joints.len(), base.joints.len());
            }
            for mesh in male.meshes.iter().filter(|m| m.vertices.iter().any(|v| v.weights.iter().sum::<f32>() > 0.0)) {
                let mut vs = mesh.vertices.clone();
                plan.apply(&mut vs);
                for (a, b) in mesh.vertices.iter().zip(&vs) {
                    assert_eq!(a.weights, b.weights, "weights are untouched");
                    assert!(b.position.iter().all(|c| c.is_finite()) && b.normal.iter().all(|c| c.is_finite()));
                    for (j, w) in b.joints.iter().zip(b.weights) {
                        assert!(w <= 0.0 || (*j as usize) < plan.skeleton.joints.len());
                    }
                }
            }
        }
        // Rename the last ten joints so they find no partner: they fold into their ancestors (or
        // are appended) and every vertex still lands on a valid joint with its weights intact.
        let mut renamed = outfit.clone();
        let first_odd = renamed.joints.len() - 10;
        for (k, j) in renamed.joints.iter_mut().enumerate().skip(first_odd) {
            j.name = format!("Zq_extra_{k}");
        }
        renamed.rig = None;
        let fold = plan_rebind(base, &renamed, &RebindOptions { extras: ExtraJoints::Fold, ..Default::default() });
        let append = plan_rebind(base, &renamed, &RebindOptions::default());
        assert_eq!(fold.missing.len(), 10);
        assert_eq!(fold.skeleton.joints.len(), base.joints.len() + fold.added.len(), "folded joints do not enlarge the skeleton");
        assert_eq!(append.skeleton.joints.len(), base.joints.len() + append.added.len());
        eprintln!("renamed tail: folded {}, appended {} of 10 missing", fold.folded.len(), append.added.len());
        assert!(fold.folded.len() + fold.added.len() == 10 && append.added.len() == 10);
        for plan in [&fold, &append] {
            for mesh in male.meshes.iter().filter(|m| m.vertices.iter().any(|v| v.weights.iter().sum::<f32>() > 0.0)) {
                let mut vs = mesh.vertices.clone();
                plan.apply(&mut vs);
                assert!(vs.iter().zip(&mesh.vertices).all(|(b, a)| a.weights == b.weights
                    && b.joints.iter().zip(b.weights).all(|(j, w)| w <= 0.0 || (*j as usize) < plan.skeleton.joints.len())));
            }
        }
        // The same avatar onto itself is the identity: nothing moves and nothing is missing.
        let same = plan_rebind(base, base, &RebindOptions::default());
        assert!(same.missing.is_empty() && same.added.is_empty() && same.warnings.is_empty());
        assert_eq!(same.joint_map, (0..base.joints.len() as u32).collect::<Vec<_>>());
        for mesh in female.meshes.iter().take(2) {
            let mut vs = mesh.vertices.clone();
            same.apply(&mut vs);
            for (a, b) in mesh.vertices.iter().zip(&vs) {
                let d = (Vec3::from(a.position) - Vec3::from(b.position)).length();
                assert!(d < 1e-4, "self rebind moved a vertex by {d}");
            }
        }
    }
}
