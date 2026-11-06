//! Muscle clips in the animation graph.
//!
//! A muscle clip ([`ClipKind::Muscle`]) samples into the [`Pose`] as a
//! [`MuscleLayer`] (a [`MusclePose`] plus the rig's [`MuscleRig`]) instead of
//! joint transforms. Blending (cross-fades, blend trees), layer composition and
//! avatar masks work on the muscle values while every pose involved carries
//! one; the layer turns into joint rotations only at the end
//! ([`Pose::resolve_muscles`]) or when it meets a pose that has none. Joints
//! whose muscles a clip does not drive keep their transform, and a clip that
//! drives no root leaves the Hips alone.
//!
//! The track-to-muscle binding of a clip is resolved once per clip and rig
//! ([`MuscleBinds`]); sampling does no name lookups.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use glam::{Quat, Vec3};
use zestengine_assets::muscle::{BodyRoot, LimbGoals, MUSCLE_COUNT, MuscleBinding, MuscleRig, MusclePose};
use zestengine_assets::{AnimClip, AvatarMask, LayerBlend};

use crate::anim_graph::{CH_R, CH_T, Pose, Rig, compose_layer_trs};

/// A humanoid pose in muscle space riding on a [`Pose`].
#[derive(Clone, Debug)]
pub struct MuscleLayer {
    pub rig: Arc<MuscleRig>,
    pub pose: MusclePose,
}

/// Per-rig cache of clip bindings, keyed by clip address and a fingerprint of
/// its muscle tracks. Derived data: cloning a rig starts empty.
#[derive(Default)]
pub struct MuscleBinds(Mutex<HashMap<(usize, u64), Arc<MuscleBinding>>>);

impl Clone for MuscleBinds {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl std::fmt::Debug for MuscleBinds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MuscleBinds")
    }
}

fn fingerprint(clip: &AnimClip) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |x: u64| h = (h ^ x).wrapping_mul(0x0000_0100_0000_01b3);
    for (i, t) in clip.tracks.iter().enumerate() {
        if !t.property.is_muscle() {
            continue;
        }
        mix(i as u64);
        mix(t.property as u64);
        mix(t.keys.is_empty() as u64);
        for b in t.name.bytes() {
            mix(b as u64);
        }
        mix(0x100);
    }
    h
}

impl Rig {
    /// The clip's muscle tracks bound to muscle indices (cached).
    pub(crate) fn muscle_binding(&self, clip: &AnimClip) -> Arc<MuscleBinding> {
        let key = (clip as *const AnimClip as usize, fingerprint(clip));
        let mut g = self.muscle_binds.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(b) = g.get(&key) {
            return b.clone();
        }
        if g.len() > 256 {
            g.clear();
        }
        let b = Arc::new(MuscleBinding::new(clip));
        g.insert(key, b.clone());
        b
    }
}

/// Sample a muscle clip at clip time `t` into `pose` (nothing on a rig without a muscle model).
pub fn sample_clip(rig: &Rig, clip: &AnimClip, t: f32, pose: &mut Pose) {
    let Some(m) = rig.muscle.as_ref() else {
        return;
    };
    let bind = rig.muscle_binding(clip);
    if bind.is_empty() {
        return;
    }
    let sampled = bind.sample(clip, t, m.rest_pose());
    pose.muscle = Some(MuscleLayer { rig: m.clone(), pose: sampled });
}

thread_local! {
    /// Joint rotations and translations the muscle model works in.
    static SCRATCH: RefCell<(Vec<Quat>, Vec<Vec3>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

impl MuscleLayer {
    /// Write the joint transforms of the driven muscles into `pose`.
    fn write(&self, pose: &mut Pose) {
        let n = pose.trs.len();
        SCRATCH.with(|s| {
            let (rot, trans) = &mut *s.borrow_mut();
            rot.clear();
            trans.clear();
            rot.extend(pose.trs.iter().map(|t| t.r));
            trans.extend(pose.trs.iter().map(|t| t.t));
            self.rig.pose_from_muscles(&self.pose, rot, trans);
            for (_, j) in self.rig.driven_joints(self.pose.driven) {
                if j < n {
                    pose.trs[j].r = rot[j];
                    pose.touched[j] |= CH_R;
                }
            }
            if self.pose.root_driven
                && let Some(h) = self.rig.joint(zestengine_assets::human_rig::HumanBone::Hips)
                && h < n
            {
                pose.trs[h].r = rot[h];
                pose.trs[h].t = trans[h];
                pose.touched[h] |= CH_R | CH_T;
            }
        });
    }
}

impl Pose {
    /// Turn a muscle layer into joint transforms (no-op without one).
    pub fn resolve_muscles(&mut self) {
        if let Some(layer) = self.muscle.take() {
            layer.write(self);
        }
    }

    /// A copy with the muscle layer resolved.
    pub fn resolved(&self) -> Pose {
        let mut p = self.clone();
        p.resolve_muscles();
        p
    }

    /// Nothing was driven: a fresh [`Pose::rest`].
    fn is_blank(&self) -> bool {
        self.muscle.is_none() && self.props.is_empty() && self.touched.iter().all(|t| *t == 0)
    }
}

/// The muscle layer both poses can contribute to: a blank pose counts as the
/// rig's rest muscles. `None` when some pose has joint data a muscle blend
/// cannot represent.
fn aligned(poses: &[&Pose]) -> Option<Vec<MuscleLayer>> {
    let rig = poses.iter().find_map(|p| p.muscle.as_ref().map(|m| m.rig.clone()))?;
    poses
        .iter()
        .map(|p| match &p.muscle {
            Some(m) => Some(m.clone()),
            None if p.is_blank() => Some(MuscleLayer { rig: rig.clone(), pose: *rig.rest_pose() }),
            None => None,
        })
        .collect()
}

/// Weighted blend of muscle layers (weights sum to 1): muscles and root position
/// linearly, root rotation by normalised sum, goals when every layer has them.
fn mix(layers: &[(MuscleLayer, f32)]) -> MuscleLayer {
    let first = layers[0].0.clone();
    let mut out = first.clone();
    let reference = first.pose.root.rotation;
    let mut q = glam::Vec4::ZERO;
    let (mut pos, mut driven, mut root_driven) = (Vec3::ZERO, 0u128, false);
    let mut m = [0.0f32; MUSCLE_COUNT];
    let all_goals = layers.iter().all(|(l, _)| l.pose.goals.is_some());
    let mut goals = LimbGoals::default();
    for (l, w) in layers {
        for (i, mi) in m.iter_mut().enumerate() {
            *mi += l.pose.muscles[i] * w;
        }
        pos += l.pose.root.position * w;
        let r = l.pose.root.rotation;
        q += glam::Vec4::from(if r.dot(reference) < 0.0 { -r } else { r }) * *w;
        driven |= l.pose.driven;
        root_driven |= l.pose.root_driven;
        if let Some(g) = &l.pose.goals
            && all_goals
        {
            for k in 0..goals.position.len() {
                goals.position[k] += g.position[k] * w;
                goals.relative[k] += g.relative[k] * w;
            }
        }
    }
    out.pose.muscles = m;
    out.pose.root = BodyRoot { position: pos, rotation: Quat::from_vec4(q).normalize() };
    out.pose.goals = all_goals.then_some(goals);
    out.pose.driven = driven;
    out.pose.root_driven = root_driven;
    out
}

/// [`blend_pose`](crate::anim_graph::blend_pose) when a muscle layer is involved. The joint data
/// blends as usual; the layers blend in muscle space. A pose with joint data
/// next to a muscle layer is resolved first.
pub(crate) fn blend_pair(a: &Pose, b: &Pose, w: f32, trs: impl Fn(&Pose, &Pose, f32) -> Pose) -> Pose {
    match aligned(&[a, b]) {
        Some(l) => {
            let mut out = trs(a, b, w);
            out.muscle = Some(mix(&[(l[0].clone(), 1.0 - w), (l[1].clone(), w)]));
            out
        }
        None => trs(&a.resolved(), &b.resolved(), w),
    }
}

/// [`blend_many`](crate::anim_graph) with muscle layers.
pub(crate) fn blend_all(poses: &[(Pose, f32)], trs: impl Fn(&[(Pose, f32)]) -> Option<Pose>) -> Option<Pose> {
    let refs: Vec<&Pose> = poses.iter().map(|(p, _)| p).collect();
    match aligned(&refs) {
        Some(layers) => {
            let total: f32 = poses.iter().map(|(_, w)| *w).sum::<f32>().max(1e-6);
            let weighted: Vec<(MuscleLayer, f32)> = layers.into_iter().zip(poses).map(|(l, (_, w))| (l, w / total)).collect();
            let mut out = trs(poses)?;
            out.muscle = Some(mix(&weighted));
            Some(out)
        }
        None => {
            let resolved: Vec<(Pose, f32)> = poses.iter().map(|(p, w)| (p.resolved(), *w)).collect();
            trs(&resolved)
        }
    }
}

/// [`compose_layer`](crate::anim_graph::compose_layer) when a muscle layer is involved: both poses
/// muscle: blend per muscle under the mask. Otherwise the muscle side is resolved to joints first.
pub(crate) fn compose(rig: &Rig, base: Pose, layer: &Pose, weight: f32, blend: LayerBlend, mask: &AvatarMask) -> Pose {
    let w = weight.clamp(0.0, 1.0);
    if w <= 0.0 {
        return base;
    }
    let (Some(b), Some(l)) = (base.muscle.clone(), layer.muscle.as_ref()) else {
        return compose_layer_trs(rig, base.resolved(), &layer.resolved(), weight, blend, mask);
    };
    let nodes = rig.mask(mask);
    let allowed = |joint: Option<usize>| match (&nodes, joint) {
        (Some(m), Some(j)) => m.get(j).copied().unwrap_or(true),
        _ => true,
    };
    let rest = *b.rig.rest_pose();
    let mut out = b.clone();
    for i in 0..MUSCLE_COUNT {
        if l.pose.driven >> i & 1 == 0 || !allowed(b.rig.muscle_joint(i)) {
            continue;
        }
        let (x, y) = (b.pose.muscles[i], l.pose.muscles[i]);
        out.pose.muscles[i] = match blend {
            LayerBlend::Override => x + (y - x) * w,
            LayerBlend::Additive => x + (y - rest.muscles[i]) * w,
        };
        out.pose.driven |= 1 << i;
    }
    if l.pose.root_driven && allowed(b.rig.joint(zestengine_assets::human_rig::HumanBone::Hips)) {
        out.pose.root = match blend {
            LayerBlend::Override => BodyRoot {
                position: b.pose.root.position.lerp(l.pose.root.position, w),
                rotation: b.pose.root.rotation.slerp(l.pose.root.rotation, w),
            },
            LayerBlend::Additive => BodyRoot {
                position: b.pose.root.position + (l.pose.root.position - rest.root.position) * w,
                rotation: (b.pose.root.rotation * Quat::IDENTITY.slerp(rest.root.rotation.inverse() * l.pose.root.rotation, w)).normalize(),
            },
        };
        out.pose.root_driven = true;
        if matches!(blend, LayerBlend::Override)
            && let (Some(gb), Some(gl)) = (b.pose.goals, l.pose.goals)
        {
            let mut g = gb;
            for k in 0..g.position.len() {
                g.position[k] = gb.position[k].lerp(gl.position[k], w);
                g.relative[k] = gb.relative[k].lerp(gl.relative[k], w);
            }
            out.pose.goals = Some(g);
        }
    }
    // The joint data of both poses composes as usual, minus the muscle layers.
    let mut joints_base = base;
    joints_base.muscle = None;
    let mut joints_layer = layer.clone();
    joints_layer.muscle = None;
    let mut result = compose_layer_trs(rig, joints_base, &joints_layer, weight, blend, mask);
    result.muscle = Some(out);
    result
}

#[cfg(test)]
pub(crate) mod test_support {
    use glam::{Mat4, Quat, Vec3};
    use zestengine_assets::human_rig::{AnimationType, HumanBone, RigImport};
    use zestengine_assets::{Joint, Skeleton};

    /// Hips, spine, head, arms with shoulders and legs: a T-pose humanoid facing +Z.
    pub fn skeleton() -> Skeleton {
        let mut joints: Vec<Joint> = Vec::new();
        let mut human: Vec<(HumanBone, String)> = Vec::new();
        let mut add = |b: HumanBone, parent: Option<&str>, t: Vec3| {
            let p = parent.map(|n| joints.iter().position(|j| j.name == n).unwrap());
            joints.push(Joint { name: b.name().into(), parent: p, inverse_bind: Mat4::IDENTITY, rest_translation: t, rest_rotation: Quat::IDENTITY, rest_scale: Vec3::ONE });
            human.push((b, b.name().into()));
        };
        add(HumanBone::Hips, None, Vec3::new(0.0, 0.98, 0.0));
        add(HumanBone::Spine, Some("Hips"), Vec3::new(0.0, 0.12, 0.0));
        add(HumanBone::Chest, Some("Spine"), Vec3::new(0.0, 0.3, 0.0));
        add(HumanBone::Head, Some("Chest"), Vec3::new(0.0, 0.3, 0.0));
        for (side, x) in [("Left", 1.0f32), ("Right", -1.0)] {
            let bone = |n: &str| HumanBone::from_name(&format!("{side}{n}")).unwrap();
            let nm = |n: &str| format!("{side}{n}");
            add(bone("Shoulder"), Some("Chest"), Vec3::new(0.04 * x, 0.1, 0.0));
            add(bone("UpperArm"), Some(&nm("Shoulder")), Vec3::new(0.08 * x, 0.0, 0.0));
            add(bone("LowerArm"), Some(&nm("UpperArm")), Vec3::new(0.28 * x, 0.0, 0.0));
            add(bone("Hand"), Some(&nm("LowerArm")), Vec3::new(0.26 * x, 0.0, 0.0));
            add(bone("UpperLeg"), Some("Hips"), Vec3::new(0.1 * x, 0.0, 0.0));
            add(bone("LowerLeg"), Some(&nm("UpperLeg")), Vec3::new(0.0, -0.45, 0.0));
            add(bone("Foot"), Some(&nm("LowerLeg")), Vec3::new(0.0, -0.45, 0.0));
        }
        let rig = RigImport { animation_type: AnimationType::Humanoid, human, ..Default::default() };
        Skeleton { joints, rig: Some(rig), ..Default::default() }
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use super::test_support::skeleton;
    use crate::anim_graph::{ClipRef, ClipSet, blend_pose, compose_layer, sample_motion};
    use zestengine_assets::human_rig::HumanBone;
    use zestengine_assets::muscle::{muscle_index, set_muscle_keys};
    use zestengine_assets::{AnimChannel, AnimationClip, ChannelPath, ClipKind, MaskPart, Motion};

    fn mi(name: &str) -> usize {
        muscle_index(name).unwrap()
    }

    /// A one-frame muscle clip driving the given muscles (and no root).
    fn clip(values: &[(&str, f32)]) -> AnimClip {
        let rig = Rig::from_skeleton(&skeleton());
        let mut p = *rig.muscle.as_ref().unwrap().rest_pose();
        p.driven = 0;
        for (n, v) in values {
            p.muscles[mi(n)] = *v;
            p.driven |= 1 << mi(n);
        }
        let mut c = AnimClip { kind: ClipKind::Muscle, ..Default::default() };
        set_muscle_keys(&mut c, 0.0, &p);
        c.tracks.retain(|t| !matches!(t.property, zestengine_assets::TrackProperty::RootPosition | zestengine_assets::TrackProperty::RootRotation));
        c
    }

    fn sample(rig: &Rig, c: &AnimClip) -> Pose {
        let mut p = Pose::rest(rig);
        ClipRef::Asset(c).sample_with(rig, 0.0, true, false, &mut p);
        p
    }

    /// The muscles a pose's joints analyse to.
    fn muscles_of(rig: &Rig, pose: &Pose) -> [f32; 95] {
        let m = rig.muscle.as_ref().unwrap();
        let rot: Vec<Quat> = pose.trs.iter().map(|t| t.r).collect();
        m.muscles_from_pose(&rot, pose.trs[m.joint(HumanBone::Hips).unwrap()].t).muscles
    }

    #[test]
    fn a_muscle_clip_samples_to_muscles_and_resolves_on_demand() {
        let rig = Rig::from_skeleton(&skeleton());
        assert!(rig.muscle.is_some(), "a configured humanoid gets a muscle model");
        let c = clip(&[("Left Forearm Stretch", 0.25), ("Left Arm Down-Up", 0.6)]);
        let lazy = sample(&rig, &c);
        let layer = lazy.muscle.as_ref().expect("lazy muscle layer");
        assert_eq!(layer.pose.driven, 1 << mi("Left Forearm Stretch") | 1 << mi("Left Arm Down-Up"));
        assert!(!layer.pose.root_driven);
        // Joints are untouched until resolved.
        assert!(lazy.touched.iter().all(|t| *t == 0));
        let mut p = lazy.clone();
        p.resolve_muscles();
        assert!(p.muscle.is_none());
        let m = muscles_of(&rig, &p);
        assert!((m[mi("Left Forearm Stretch")] - 0.25).abs() < 3e-3 && (m[mi("Left Arm Down-Up")] - 0.6).abs() < 3e-3);
        // Only the driven bones were written; the Hips and the right arm kept their rest transform.
        let idx = |n: &str| rig.index(n).unwrap();
        assert_ne!(p.touched[idx("LeftLowerArm")], 0);
        assert_eq!(p.touched[idx("Hips")], 0);
        assert_eq!(p.touched[idx("RightLowerArm")], 0);
        assert_eq!(p.trs[idx("Hips")], rig.rest[idx("Hips")]);
        // The convenience sample() resolves.
        let mut q = Pose::rest(&rig);
        ClipRef::Asset(&c).sample(&rig, 0.0, true, &mut q);
        assert!(q.muscle.is_none() && q.trs[idx("LeftLowerArm")] == p.trs[idx("LeftLowerArm")]);
    }

    #[test]
    fn blending_two_muscle_clips_blends_the_muscles() {
        let rig = Rig::from_skeleton(&skeleton());
        let a = clip(&[("Left Arm Down-Up", 1.0), ("Left Forearm Stretch", 0.2), ("Left Upper Leg Front-Back", -0.9)]);
        let b = clip(&[("Left Arm Down-Up", -0.8), ("Left Forearm Stretch", -0.6), ("Right Hand Down-Up", 0.7)]);
        let (pa, pb) = (sample(&rig, &a), sample(&rig, &b));
        let w = 0.3;
        let mut mid = blend_pose(&pa, &pb, w);
        assert!(mid.muscle.is_some(), "blended in muscle space");
        mid.resolve_muscles();
        let got = muscles_of(&rig, &mid);
        let rest = *rig.muscle.as_ref().unwrap().rest_pose();
        let (ma, mb) = (&pa.muscle.as_ref().unwrap().pose.muscles, &pb.muscle.as_ref().unwrap().pose.muscles);
        for name in ["Left Arm Down-Up", "Left Forearm Stretch", "Left Upper Leg Front-Back", "Right Hand Down-Up"] {
            let i = mi(name);
            let want = ma[i] * (1.0 - w) + mb[i] * w;
            assert!((got[i] - want).abs() < 3e-3, "{name}: {} vs {want}", got[i]);
        }
        // An undriven muscle blends rest with rest.
        assert!((got[mi("Right Arm Down-Up")] - rest.muscles[mi("Right Arm Down-Up")]).abs() < 3e-3);
        // Blending joint rotations of the two resolved poses gives something else for a swing this large.
        let tr = blend_pose(&pa.resolved(), &pb.resolved(), w);
        let by_joints = muscles_of(&rig, &tr);
        assert!((by_joints[mi("Left Arm Down-Up")] - got[mi("Left Arm Down-Up")]).abs() > 1e-3, "the test pair is not trivially linear");
        // A blank pose blends as the rest muscles; a pose with joint data resolves first.
        let blank = Pose::rest(&rig);
        let half = blend_pose(&pa, &blank, 0.5);
        assert!(half.muscle.is_some());
        let mut with_t = Pose::rest(&rig);
        with_t.trs[rig.index("Hips").unwrap()].t.x = 1.0;
        with_t.touched[rig.index("Hips").unwrap()] = crate::anim_graph::CH_T;
        assert!(blend_pose(&pa, &with_t, 0.5).muscle.is_none());
    }

    #[test]
    fn layers_compose_in_muscle_space_under_an_avatar_mask() {
        let rig = Rig::from_skeleton(&skeleton());
        let base = sample(&rig, &clip(&[("Left Arm Down-Up", 0.5), ("Right Arm Down-Up", 0.5), ("Left Index 1 Stretched", 0.1)]));
        let top = sample(&rig, &clip(&[("Left Arm Down-Up", -0.5), ("Right Arm Down-Up", -0.5)]));
        let mask = AvatarMask { parts: vec![MaskPart::LeftArm], elements: Vec::new() };
        let out = compose_layer(&rig, base.clone(), &top, 1.0, LayerBlend::Override, &mask);
        let l = out.muscle.as_ref().expect("still muscle space").pose;
        assert!((l.muscles[mi("Left Arm Down-Up")] + 0.5).abs() < 1e-6, "the masked-in arm takes the layer");
        assert!((l.muscles[mi("Right Arm Down-Up")] - 0.5).abs() < 1e-6, "the masked-out arm keeps the base");
        // Half weight over everything blends per muscle.
        let half = compose_layer(&rig, base.clone(), &top, 0.5, LayerBlend::Override, &AvatarMask::default());
        assert!(half.muscle.as_ref().unwrap().pose.muscles[mi("Right Arm Down-Up")].abs() < 1e-6);
        // Additive adds the layer's offset from the rest muscles.
        let rest = rig.muscle.as_ref().unwrap().rest_pose().muscles[mi("Left Arm Down-Up")];
        let add = compose_layer(&rig, base.clone(), &top, 1.0, LayerBlend::Additive, &AvatarMask::default());
        assert!((add.muscle.as_ref().unwrap().pose.muscles[mi("Left Arm Down-Up")] - (0.5 + (-0.5 - rest))).abs() < 1e-5);
        // The layer drove no fingers: the base keeps its finger value.
        assert!((out.muscle.as_ref().unwrap().pose.muscles[mi("Left Index 1 Stretched")] - 0.1).abs() < 1e-6);
        // A joint-data layer over a muscle base resolves the base.
        let plain = Pose::rest(&rig);
        assert!(compose_layer(&rig, base, &plain, 1.0, LayerBlend::Override, &AvatarMask::default()).muscle.is_none());
    }

    #[test]
    fn muscle_clips_play_through_motions_and_fall_back_on_other_rigs() {
        let rig = Rig::from_skeleton(&skeleton());
        let mut c = clip(&[("Left Forearm Stretch", 0.0)]);
        c.fallback = "Wave".into();
        let mut assets = HashMap::new();
        assets.insert("wave.anim".to_string(), Some(Arc::new(c.clone())));
        // An imported clip that moves joint 3 (the head) up by a metre.
        let imported = vec![AnimationClip {
            name: "Wave".into(),
            duration: 1.0,
            channels: vec![AnimChannel { joint: 3, path: ChannelPath::Translation, times: vec![0.0], vec_values: vec![Vec3::new(0.0, 1.0, 0.0)], quat_values: vec![], extra: None }],
            morph_tracks: Vec::new(), playback: None,
        }];
        let indices = [0usize];
        let set = ClipSet { imported: &imported, indices: &indices, assets: &assets };
        assert!(matches!(set.find("wave.anim"), Some(ClipRef::Baked(..))), "a muscle clip with a fallback finds it");
        let motion = Motion::Clip("wave.anim".into());
        let p = sample_motion(&motion, &HashMap::new(), &set, &rig, 0.0, true, false);
        assert!(p.muscle.is_none(), "sample_motion resolves");
        let m = muscles_of(&rig, &p);
        assert!(m[mi("Left Forearm Stretch")].abs() < 3e-3, "humanoid plays the muscles");
        assert_eq!(p.trs[3].t, rig.rest[3].t, "and not the fallback");
        // A rig without a humanoid map plays the original clip.
        let mut generic = skeleton();
        generic.rig = None;
        let flat = Rig::from_skeleton(&generic);
        assert!(flat.muscle.is_none());
        let p = sample_motion(&motion, &HashMap::new(), &set, &flat, 0.0, true, false);
        assert_eq!(p.trs[3].t, Vec3::new(0.0, 1.0, 0.0));
        // Without the fallback clip the muscle clip is a no-op there.
        let mut bare = c;
        bare.fallback.clear();
        let mut assets = HashMap::new();
        assets.insert("wave.anim".to_string(), Some(Arc::new(bare)));
        let set = ClipSet { imported: &imported, indices: &indices, assets: &assets };
        assert!(sample_motion(&motion, &HashMap::new(), &set, &flat, 0.0, true, false).touched.iter().all(|t| *t == 0));
    }

    #[test]
    fn mirroring_a_muscle_pose_in_a_blend_tree_swaps_the_arms() {
        let rig = Rig::from_skeleton(&skeleton());
        let mut p = sample(&rig, &clip(&[("Left Arm Down-Up", 0.9)]));
        rig.mirror_pose(&mut p);
        let m = &p.muscle.as_ref().unwrap().pose;
        assert!((m.muscles[mi("Right Arm Down-Up")] - 0.9).abs() < 1e-6);
        assert_eq!(m.driven, 1 << mi("Right Arm Down-Up"));
    }

    #[test]
    fn bindings_are_cached_per_clip_and_follow_edits() {
        let rig = Rig::from_skeleton(&skeleton());
        let mut c = clip(&[("Jaw Close", 0.5)]);
        let a = rig.muscle_binding(&c);
        assert!(Arc::ptr_eq(&a, &rig.muscle_binding(&c)), "second lookup hits the cache");
        let mut p = *rig.muscle.as_ref().unwrap().rest_pose();
        p.driven = 1 << mi("Head Nod Down-Up");
        set_muscle_keys(&mut c, 0.0, &p);
        assert!(!Arc::ptr_eq(&a, &rig.muscle_binding(&c)), "adding a track rebinds");
    }
}
