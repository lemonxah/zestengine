//! Animation graph runtime: evaluates Animator Controllers (`.animctrl` state
//! machines) and authored `.anim` clips for an [`Animator`].
//!
//! A [`Rig`] lists what an Animator animates: skeleton joints for skinned
//! models, or the object and its descendants (by relative path) otherwise.
//! Clips sample into a [`Pose`] (TRS + one scalar lane per node, with a bit
//! mask of the channels a clip actually drives), poses cross-fade and layers
//! compose (override with weight, additive over the rest pose, per-node mask).
//! Untouched channels keep their rest value, so only animated properties are
//! written back to objects.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use glam::{Mat4, Quat, Vec3};

use zestengine_assets::muscle::MuscleRig;
use zestengine_assets::{
    AnimClip, AnimParamKind, AnimState, AnimationClip, AnimatorController, AvatarMask, ChannelPath, InterruptionSource, LayerBlend, Motion, Skeleton,
    StateMachine, TrackProperty,
};
use zestengine_core::{
    Animator, AnimatorCullingMode, AnimatorUpdateMode, FrozenPose, LayerPlayback, PropSample, StateBehaviourCtx, StateCallback, StateCallbackKind,
};

use crate::anim_props::{PropTable, compose_props, hold_props, mix_props};

// Humanoid retargeting exposed through the animation system: a precomputed
// role-to-role mapping plus the one-shot pose/clip helpers.
pub use crate::humanoid::{HumanoidRetarget, retarget_clip, retarget_locals};

pub const CH_T: u8 = 1;
pub const CH_R: u8 = 2;
pub const CH_S: u8 = 4;
pub const CH_F: u8 = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trs {
    pub t: Vec3,
    pub r: Quat,
    pub s: Vec3,
}

impl Default for Trs {
    fn default() -> Self {
        Self { t: Vec3::ZERO, r: Quat::IDENTITY, s: Vec3::ONE }
    }
}

impl Trs {
    pub fn to_mat4(self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.s, self.r, self.t)
    }
}

/// A bone name reduced to what rigs agree on: no namespace (`mixamorig:`,
/// `Armature|`), no `mixamorig` prefix, lower case, letters and digits only.
pub fn loose_bone_name(name: &str) -> String {
    let leaf = name.rsplit([':', '|']).next().unwrap_or(name).to_ascii_lowercase();
    let leaf = leaf.strip_prefix("mixamorig").unwrap_or(&leaf);
    leaf.chars().filter(|c| c.is_ascii_alphanumeric()).collect()
}

/// The nodes an Animator drives.
#[derive(Clone, Debug, Default)]
pub struct Rig {
    pub names: Vec<String>,
    pub parents: Vec<Option<usize>>,
    pub rest: Vec<Trs>,
    /// Rest value of the scalar lane (light intensity).
    pub rest_scalar: Vec<f32>,
    /// Imported skeletal clips address nodes by joint index.
    pub skeletal: bool,
    /// Node of each human bone (`HumanBone as usize`), for avatar-mask parts; empty = not humanoid.
    pub human: Vec<Option<usize>>,
    /// Left/right pairs and reference rotations for humanoid mirroring; `None` = not humanoid.
    pub mirror: Option<Arc<crate::anim_mirror::RigMirror>>,
    /// Muscle model of a humanoid skeleton (plays muscle clips); `None` = not humanoid.
    pub muscle: Option<Arc<MuscleRig>>,
    pub(crate) muscle_binds: crate::anim_muscle::MuscleBinds,
    lookup: HashMap<String, usize>,
    /// Bones by [`loose_bone_name`], for clips made on another rig.
    loose: HashMap<String, usize>,
    /// Object-property channels resolved for this rig (see `anim_props`).
    pub(crate) props: PropTable,
    /// Container joints above the bones (see `Skeleton::containers`).
    pub(crate) containers: Vec<usize>,
}

impl Rig {
    pub fn new(names: Vec<String>, parents: Vec<Option<usize>>, rest: Vec<Trs>, rest_scalar: Vec<f32>, skeletal: bool) -> Self {
        let lookup = names.iter().enumerate().map(|(i, n)| (n.clone(), i)).collect();
        let mut loose = HashMap::new();
        if skeletal {
            for (i, n) in names.iter().enumerate() {
                loose.entry(loose_bone_name(n)).or_insert(i);
            }
        }
        Self { names, parents, rest, rest_scalar, skeletal, human: Vec::new(), mirror: None, muscle: None, muscle_binds: Default::default(), lookup, loose, props: PropTable::default(), containers: Vec::new() }
    }

    /// The bone a clip track names: the exact name, else (skeletal rigs only)
    /// the bone with the same [`loose_bone_name`], so a clip made on
    /// `mixamorig:LeftUpLeg` finds `LeftUpLeg` or `mixamorig_LeftUpLeg`.
    pub fn index_loose(&self, name: &str) -> Option<usize> {
        self.index(name).or_else(|| if self.skeletal { self.loose.get(&loose_bone_name(name)).copied() } else { None })
    }

    /// The node that carries the motion (root motion, yaw): the first parentless
    /// one, looking through the skeleton's container joints.
    pub fn motion_root(&self) -> Option<usize> {
        self.parents.iter().enumerate().position(|(i, p)| !self.containers.contains(&i) && p.is_none_or(|p| self.containers.contains(&p)))
    }

    pub fn from_skeleton(skel: &Skeleton) -> Self {
        let n = skel.joints.len();
        let mut rig = Self::new(
            skel.joints.iter().map(|j| j.name.clone()).collect(),
            skel.joints.iter().map(|j| j.parent).collect(),
            skel.joints.iter().map(|j| Trs { t: j.rest_translation, r: j.rest_rotation, s: j.rest_scale }).collect(),
            vec![0.0; n],
            true,
        );
        rig.containers = skel.containers.clone();
        if let Some(r) = skel.rig.as_ref().filter(|r| r.is_configured()) {
            rig.human = vec![None; zestengine_assets::human_rig::HUMAN_BONE_COUNT];
            for (b, name) in &r.human {
                rig.human[*b as usize] = rig.index(name);
            }
            rig.mirror = crate::anim_mirror::RigMirror::build(&rig, reference_world_rotations(skel)).map(Arc::new);
            rig.muscle = MuscleRig::new(skel, r).map(Arc::new);
        }
        rig
    }

    /// Mirror `pose` when it is a humanoid rig (warns once and does nothing otherwise).
    pub fn mirror_pose(&self, pose: &mut Pose) {
        if let Some(m) = pose.muscle.as_mut() {
            m.pose.mirror();
        }
        match &self.mirror {
            Some(m) => m.mirror_pose(self, pose),
            None => crate::anim_mirror::warn_not_humanoid(),
        }
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    pub fn index(&self, name: &str) -> Option<usize> {
        self.lookup.get(name).copied()
    }

    /// Nodes covered by a layer mask (weight above 0.5). None = every node.
    pub fn mask(&self, mask: &AvatarMask) -> Option<Vec<bool>> {
        mask.active(&self.parents, &|p| self.index(p), &|b| self.human.get(b as usize).copied().flatten())
    }
}

/// World rotation of every joint in the skeleton's reference pose.
fn reference_world_rotations(skel: &Skeleton) -> Vec<Quat> {
    let locals = zestengine_assets::ref_pose::reference_locals(skel);
    let mut world: Vec<Option<Quat>> = vec![None; locals.len()];
    fn at(i: usize, skel: &Skeleton, locals: &[Mat4], world: &mut [Option<Quat>]) -> Quat {
        if let Some(q) = world[i] {
            return q;
        }
        let r = locals[i].to_scale_rotation_translation().1;
        let q = match skel.joints[i].parent {
            Some(p) if p != i => at(p, skel, locals, world) * r,
            _ => r,
        };
        world[i] = Some(q.normalize());
        q.normalize()
    }
    (0..locals.len()).map(|i| at(i, skel, &locals, &mut world)).collect()
}

/// Per-node values plus which channels were driven.
#[derive(Clone, Debug)]
pub struct Pose {
    pub trs: Vec<Trs>,
    pub scalar: Vec<f32>,
    pub touched: Vec<u8>,
    /// Object properties driven (active, component enabled, blendshape,
    /// material), sparse and sorted by channel. Empty for most clips.
    pub props: Vec<PropSample>,
    /// A humanoid pose in muscle space from muscle clips, not yet turned into
    /// joint transforms (see `anim_muscle`). The output of the graph never has one.
    pub muscle: Option<crate::anim_muscle::MuscleLayer>,
}

impl Pose {
    pub fn rest(rig: &Rig) -> Self {
        Self { trs: rig.rest.clone(), scalar: rig.rest_scalar.clone(), touched: vec![0; rig.len()], props: Vec::new(), muscle: None }
    }

    pub fn to_locals(&self) -> Vec<Mat4> {
        self.trs.iter().map(|t| t.to_mat4()).collect()
    }
}

/// Per-node TRS blend (`w` = 0 -> `a`, 1 -> `b`). Poses with muscle layers
    /// blend those in muscle space.
pub fn blend_pose(a: &Pose, b: &Pose, w: f32) -> Pose {
    if a.muscle.is_some() || b.muscle.is_some() {
        return crate::anim_muscle::blend_pair(a, b, w, blend_pose_trs);
    }
    blend_pose_trs(a, b, w)
}

fn blend_pose_trs(a: &Pose, b: &Pose, w: f32) -> Pose {
    let n = a.trs.len().min(b.trs.len());
    let mut out = a.clone();
    for i in 0..n {
        let (x, y) = (a.trs[i], b.trs[i]);
        out.trs[i] = Trs { t: x.t.lerp(y.t, w), r: x.r.slerp(y.r, w), s: x.s.lerp(y.s, w) };
        out.scalar[i] = a.scalar[i] + (b.scalar[i] - a.scalar[i]) * w;
        out.touched[i] = a.touched[i] | b.touched[i];
    }
    out.props = mix_props(&[(&a.props, 1.0 - w), (&b.props, w)]);
    out
}

/// Weighted average of several poses (weights need not be normalized).
fn blend_many(poses: &[(Pose, f32)]) -> Option<Pose> {
    if poses.iter().any(|(p, _)| p.muscle.is_some()) {
        return crate::anim_muscle::blend_all(poses, blend_many_trs);
    }
    blend_many_trs(poses)
}

fn blend_many_trs(poses: &[(Pose, f32)]) -> Option<Pose> {
    let (first, _) = poses.first()?;
    let total: f32 = poses.iter().map(|(_, w)| *w).sum::<f32>().max(1e-6);
    let mut out = first.clone();
    for i in 0..first.trs.len() {
        let (mut t, mut s, mut q, mut f) = (Vec3::ZERO, Vec3::ZERO, glam::Vec4::ZERO, 0.0);
        let reference = first.trs[i].r;
        let mut touched = 0;
        for (p, w) in poses {
            let w = w / total;
            let x = p.trs[i];
            t += x.t * w;
            s += x.s * w;
            let r = if x.r.dot(reference) < 0.0 { -x.r } else { x.r };
            q += glam::Vec4::from(r) * w;
            f += p.scalar[i] * w;
            touched |= p.touched[i];
        }
        out.trs[i] = Trs { t, r: Quat::from_vec4(q).normalize(), s };
        out.scalar[i] = f;
        out.touched[i] = touched;
    }
    let parts: Vec<(&[PropSample], f32)> = poses.iter().map(|(p, w)| (p.props.as_slice(), w / total)).collect();
    out.props = mix_props(&parts);
    Some(out)
}

// ------------------------------------------------------------------ clips

/// An imported skeletal clip or an authored `.anim` asset.
#[derive(Clone, Copy)]
pub enum ClipRef<'a> {
    Imported(&'a AnimationClip),
    Asset(&'a AnimClip),
    /// A muscle clip baked from an imported clip, with that clip as the
    /// fallback for rigs that are not humanoid.
    Baked(&'a AnimClip, &'a AnimationClip),
}

/// Which components of a clip's root travel stay in the pose (Bake Into Pose);
/// the rest drives root motion. A clip without playback settings bakes nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Bake {
    xz: bool,
    y: bool,
    rot: bool,
}

impl ClipRef<'_> {
    /// The listed-clip settings of an imported clip (also of the clip a muscle clip was baked from).
    fn playback(&self) -> Option<&zestengine_assets::ClipPlayback> {
        match self {
            ClipRef::Imported(c) | ClipRef::Baked(_, c) => c.playback.as_deref(),
            ClipRef::Asset(_) => None,
        }
    }

    /// Loop Time: the clip wraps (true unless its settings say it clamps).
    fn loops(&self) -> bool {
        self.playback().is_none_or(|p| p.loop_time)
    }

    /// The clip's own cycle offset (normalised).
    fn cycle_offset(&self) -> f32 {
        self.playback().map_or(0.0, |p| p.cycle_offset)
    }

    fn mirrored(&self) -> bool {
        self.playback().is_some_and(|p| p.mirror)
    }

    fn bake(&self) -> Bake {
        // Without settings the turn stays in the pose, as it always did.
        self.playback().map_or(Bake { xz: false, y: false, rot: true }, |p| Bake { xz: p.bake_xz, y: p.bake_y, rot: p.bake_rotation })
    }

    /// Whether the clip's root travel can be measured (a clip played on a rig
    /// by joint or by name, not an object-property clip).
    fn has_root(&self) -> bool {
        match self {
            ClipRef::Imported(_) => true,
            ClipRef::Asset(c) => !c.source_rest.is_empty(),
            ClipRef::Baked(..) => false,
        }
    }

    pub fn duration(&self) -> f32 {
        match self {
            ClipRef::Imported(c) => c.duration,
            ClipRef::Asset(c) | ClipRef::Baked(c, _) => c.length,
        }
    }

    /// Write this clip's channels at time `t` into `pose` (muscle clips resolved to joint transforms).
    pub fn sample(&self, rig: &Rig, t: f32, looping: bool, pose: &mut Pose) {
        self.sample_with(rig, t, looping, false, pose);
        pose.resolve_muscles();
    }

    /// [`ClipRef::sample`], optionally with Loop Pose: the difference between
    /// the clip's first and last frame is spread over the cycle (the value
    /// gains `(start - end) * phase`, rotations the start/end delta scaled by
    /// the phase), so a loop closes without a pop. Only while looping.
    pub fn sample_with(&self, rig: &Rig, t: f32, looping: bool, loop_pose: bool, pose: &mut Pose) {
        let looping = looping && self.loops();
        let d = self.duration();
        let tt = if d <= 1e-6 {
            0.0
        } else if looping {
            t.rem_euclid(d)
        } else {
            t.clamp(0.0, (d - 1e-4).max(0.0))
        };
        let phase = if loop_pose && looping && d > 1e-6 { Some(tt / d) } else { None };
        match self {
            ClipRef::Baked(muscle, imported) => {
                if rig.muscle.is_some() {
                    crate::anim_muscle::sample_clip(rig, muscle, tt, pose);
                } else {
                    ClipRef::Imported(imported).sample_with(rig, t, looping, loop_pose, pose);
                }
            }
            ClipRef::Asset(clip) if clip.kind == zestengine_assets::ClipKind::Muscle => crate::anim_muscle::sample_clip(rig, clip, tt, pose),
            // A clip from another model plays through the muscle model on a humanoid rig.
            ClipRef::Asset(clip) if rig.muscle.is_some() && clip.humanoid.is_some() => {
                if let Some(twin) = clip.humanoid.as_deref() {
                    crate::anim_muscle::sample_clip(rig, twin, tt, pose);
                }
            }
            ClipRef::Imported(clip) => {
                if !rig.skeletal {
                    return;
                }
                for ch in &clip.channels {
                    if ch.joint >= rig.len() || ch.times.is_empty() {
                        continue;
                    }
                    let n = &mut pose.trs[ch.joint];
                    let last = ch.len() - 1;
                    match ch.path {
                        ChannelPath::Translation => {
                            if let Some(v) = ch.sample_vec(tt) {
                                n.t = v;
                                if let Some(ph) = phase
                                    && let (Some(s), Some(e)) = (ch.vec_at(0), ch.vec_at(last))
                                {
                                    n.t += (s - e) * ph;
                                }
                                pose.touched[ch.joint] |= CH_T;
                            }
                        }
                        ChannelPath::Scale => {
                            if let Some(v) = ch.sample_vec(tt) {
                                n.s = v;
                                if let Some(ph) = phase
                                    && let (Some(s), Some(e)) = (ch.vec_at(0), ch.vec_at(last))
                                {
                                    n.s += (s - e) * ph;
                                }
                                pose.touched[ch.joint] |= CH_S;
                            }
                        }
                        ChannelPath::Rotation => {
                            if let Some(q) = ch.sample_quat(tt) {
                                n.r = q;
                                if let Some(ph) = phase
                                    && let (Some(s), Some(e)) = (ch.quat_at(0), ch.quat_at(last))
                                {
                                    let delta = s * e.inverse();
                                    n.r = (Quat::IDENTITY.slerp(delta, ph) * n.r).normalize();
                                }
                                pose.touched[ch.joint] |= CH_R;
                            }
                        }
                    }
                }
                // Blendshape curves imported with the clip: object-property samples
                // (anchored on the skinned object), mixed like any other property.
                if !clip.morph_tracks.is_empty()
                    && let Some(bindings) = rig.bind_tracks(&clip.morph_tracks, clip.morph_tracks.as_ptr() as usize, true)
                {
                    for b in bindings.iter() {
                        let v = clip.morph_tracks[b.track as usize].sample(tt);
                        pose.set_prop(PropSample { chan: b.chan, node: b.node, value: v, cover: 1.0 });
                    }
                }
                if let Some(pb) = clip.playback.as_deref() {
                    adjust_root(rig, pb, pose);
                }
            }
            ClipRef::Asset(clip) => {
                // Object-property tracks: bound to rig channels once per clip.
                if let Some(bindings) = rig.prop_bindings(clip) {
                    for b in bindings.iter() {
                        let v = clip.tracks[b.track as usize].sample(tt);
                        pose.set_prop(PropSample { chan: b.chan, node: b.node, value: v, cover: 1.0 });
                    }
                }
                for track in &clip.tracks {
                    if track.property.is_object_prop() {
                        continue;
                    }
                    let Some(node) = rig.index_loose(&track.target) else {
                        continue;
                    };
                    if track.keys.is_empty() {
                        continue;
                    }
                    let mut v = track.sample(tt);
                    // Loop Pose: gain (start - end) * phase.
                    if let Some(ph) = phase {
                        let (s, e) = (track.sample(0.0), track.sample(d));
                        match track.property {
                            TrackProperty::Rotation => {
                                let delta = Quat::from_array(s).normalize() * Quat::from_array(e).normalize().inverse();
                                v = (Quat::IDENTITY.slerp(delta, ph) * Quat::from_array(v).normalize()).to_array();
                            }
                            _ => (0..4).for_each(|k| v[k] += (s[k] - e[k]) * ph),
                        }
                    }
                    match track.property {
                        TrackProperty::Translation => {
                            let mut t = Vec3::new(v[0], v[1], v[2]);
                            // A clip from another model: its offset from that rig's rest,
                            // scaled by the bone length ratio (units and proportions).
                            if let Some(src) = clip.source_rest_of(&track.target).map(Vec3::from_array) {
                                let own = rig.rest[node].t;
                                let ratio = if src.length() > 1e-6 && own.length() > 1e-6 { own.length() / src.length() } else { 1.0 };
                                t = own + (t - src) * ratio;
                            }
                            pose.trs[node].t = t;
                            pose.touched[node] |= CH_T;
                        }
                        TrackProperty::Rotation => {
                            pose.trs[node].r = Quat::from_array(v).normalize();
                            pose.touched[node] |= CH_R;
                        }
                        TrackProperty::Scale => {
                            pose.trs[node].s = Vec3::new(v[0], v[1], v[2]);
                            pose.touched[node] |= CH_S;
                        }
                        TrackProperty::LightIntensity => {
                            pose.scalar[node] = v[0];
                            pose.touched[node] |= CH_F;
                        }
                        // Handled above through the rig's property channels.
                        _ => {}
                    }
                }
            }
        }
    }
}

/// The "based upon" references and offsets of the baked root components
/// (Unity "Bake Into Pose"): a baked rotation is made relative to the body
/// orientation (unless Original) plus the rotation offset, a baked height to
/// the feet (unless Original) plus the height offset, baked ground-plane travel
/// to the centre of the clip (unless Original). Components that are not baked
/// are root motion and are left as authored.
fn adjust_root(rig: &Rig, pb: &zestengine_assets::ClipPlayback, pose: &mut Pose) {
    if !rig.skeletal {
        return;
    }
    let Some(root) = rig.motion_root() else { return };
    if pb.bake_rotation && pose.touched[root] & CH_R != 0 {
        let delta = (if pb.keep_orientation { 0.0 } else { -pb.body_yaw }) + pb.rotation_offset.to_radians();
        if delta != 0.0 {
            pose.trs[root].r = (Quat::from_rotation_y(delta) * pose.trs[root].r).normalize();
        }
    }
    if pose.touched[root] & CH_T != 0 {
        let n = &mut pose.trs[root];
        if pb.bake_y {
            n.t.y += pb.height_offset - if pb.keep_y { 0.0 } else { pb.feet_y };
        }
        if pb.bake_xz && !pb.keep_xz {
            n.t.x -= pb.center_xz[0];
            n.t.z -= pb.center_xz[1];
        }
    }
}

#[allow(dead_code)] // keyframe index helper used by the pose-graph tests
fn key_indices(times: &[f32], t: f32) -> (usize, usize, f32) {
    if times.len() == 1 || t <= times[0] {
        return (0, 0, 0.0);
    }
    let last = times.len() - 1;
    if t >= times[last] {
        return (last, last, 0.0);
    }
    let i = times.partition_point(|&k| k <= t).saturating_sub(1).min(last - 1);
    let (a, b) = (times[i], times[i + 1]);
    let f = if (b - a).abs() > 1e-6 { (t - a) / (b - a) } else { 0.0 };
    (i, i + 1, f.clamp(0.0, 1.0))
}

/// Clips reachable by name for one Animator: the model's imported clips plus
/// loaded `.anim` assets (looked up by project-relative path).
pub struct ClipSet<'a> {
    pub imported: &'a [AnimationClip],
    /// Indices into `imported` that belong to this Animator's skeleton.
    pub indices: &'a [usize],
    pub assets: &'a HashMap<String, Option<Arc<AnimClip>>>,
}

impl<'a> ClipSet<'a> {
    pub fn find(&self, name: &str) -> Option<ClipRef<'a>> {
        if is_clip_asset(name) {
            let clip = self.assets.get(name)?.as_deref()?;
            if clip.kind == zestengine_assets::ClipKind::Muscle && !clip.fallback.is_empty() {
                let imported = self.imported;
                if let Some(c) = self.indices.iter().copied().find(|&c| imported[c].name == clip.fallback) {
                    return Some(ClipRef::Baked(clip, &imported[c]));
                }
            }
            return Some(ClipRef::Asset(clip));
        }
        let imported = self.imported;
        self.indices.iter().copied().find(|&c| imported[c].name == name).map(|c| ClipRef::Imported(&imported[c]))
    }
}

pub fn is_clip_asset(name: &str) -> bool {
    // `@<id>` references come from clip asset slots.
    zestengine_core::asset_ref::is_ref(name) || name.to_ascii_lowercase().ends_with(".anim") || crate::clip_source::is_model_ref(name)
}

// ----------------------------------------------------------------- motions

/// A clip a motion plays, with its share of the blend and its playback settings.
#[derive(Clone, Copy)]
struct Leaf<'a> {
    /// Slot among the motion's clip leaves (stable across frames; weight-0
    /// leaves keep theirs), for per-clip memory between frames.
    index: usize,
    name: &'a str,
    clip: ClipRef<'a>,
    weight: f32,
    time_scale: f32,
    /// Carried through nested trees; the sampled pose and root motion are
    /// mirrored across the sagittal plane (humanoid rigs only).
    mirror: bool,
    cycle_offset: f32,
}

impl Leaf<'_> {
    /// Clip-local seconds at the motion's normalized time `n`: every leaf sits
    /// at the same phase, shifted by its cycle offset.
    fn time_at(&self, n: f32) -> f32 {
        (n + self.cycle_offset) * self.clip.duration()
    }
}

/// The clips a motion plays and how many clip leaves it has in all (resolved
/// or not, weighted or not). A plain clip is one leaf at weight 1; blend trees
/// expand nested trees and carry per-child time scale, mirror and cycle offset.
fn motion_leaves<'a>(motion: &'a Motion, params: &HashMap<String, f32>, clips: &ClipSet<'a>) -> (Vec<Leaf<'a>>, usize) {
    match motion {
        Motion::Clip(name) => {
            let leaf = clips.find(name).map(|clip| Leaf { index: 0, name, clip, weight: 1.0, time_scale: 1.0, mirror: clip.mirrored(), cycle_offset: clip.cycle_offset() });
            (leaf.into_iter().collect(), 1)
        }
        Motion::BlendTree(tree) => {
            let read = |n: &str| params.get(n).copied().unwrap_or(0.0);
            let all = tree.leaves(&read);
            let total = all.len();
            let leaves = all
                .into_iter()
                .filter_map(|l| {
                    clips.find(l.clip).map(|clip| Leaf {
                        index: l.index,
                        name: l.clip,
                        clip,
                        weight: l.weight,
                        time_scale: l.time_scale,
                        mirror: l.mirror != clip.mirrored(),
                        cycle_offset: l.cycle_offset + clip.cycle_offset(),
                    })
                })
                .collect();
            (leaves, total)
        }
    }
}

/// Length of one pass of a motion. A blend tree's is the weight-averaged
/// length of its clips, each divided by its time scale; the average divides
/// by the weight sum only when that is below 1 (a Direct tree may exceed it).
pub fn motion_duration(motion: &Motion, params: &HashMap<String, f32>, clips: &ClipSet) -> f32 {
    match motion {
        Motion::Clip(name) => clips.find(name).map(|c| c.duration()).unwrap_or(0.0),
        Motion::BlendTree(_) => {
            let (leaves, _) = motion_leaves(motion, params, clips);
            let (mut sum, mut len) = (0.0, 0.0);
            for l in leaves.iter().filter(|l| l.weight > 0.0) {
                sum += l.weight;
                len += l.weight * (l.clip.duration() / l.time_scale.abs().max(1e-3));
            }
            if sum <= 1e-6 {
                1.0
            } else if sum < 1.0 {
                len / sum
            } else {
                len
            }
        }
    }
}

/// Pose of a motion at normalized time `n` (0..1 per pass). Blend-tree clips
/// all sit at the same phase, so clips of different lengths never drift
/// apart. Weight the children leave unclaimed (a Direct tree summing below 1)
/// falls back to the rest pose.
pub fn sample_motion(
    motion: &Motion,
    params: &HashMap<String, f32>,
    clips: &ClipSet,
    rig: &Rig,
    n: f32,
    looping: bool,
    loop_pose: bool,
) -> Pose {
    let mut pose = sample_motion_lazy(motion, params, clips, rig, n, looping, loop_pose);
    pose.resolve_muscles();
    pose
}

/// [`sample_motion`] leaving a muscle clip's pose in muscle space, so it blends there.
fn sample_motion_lazy(
    motion: &Motion,
    params: &HashMap<String, f32>,
    clips: &ClipSet,
    rig: &Rig,
    n: f32,
    looping: bool,
    loop_pose: bool,
) -> Pose {
    let mut pose = Pose::rest(rig);
    match motion {
        Motion::Clip(name) => {
            if let Some(c) = clips.find(name) {
                c.sample_with(rig, (n + c.cycle_offset()) * c.duration(), looping, loop_pose, &mut pose);
                if c.mirrored() && rig.mirror.is_some() {
                    rig.mirror_pose(&mut pose);
                }
            }
            pose
        }
        Motion::BlendTree(_) => {
            let (leaves, _) = motion_leaves(motion, params, clips);
            let mut poses: Vec<(Pose, f32)> = leaves
                .iter()
                .filter(|l| l.weight > 1e-4)
                .map(|l| {
                    let mut p = Pose::rest(rig);
                    l.clip.sample_with(rig, l.time_at(n), looping, loop_pose, &mut p);
                    if l.mirror {
                        rig.mirror_pose(&mut p);
                    }
                    (p, l.weight)
                })
                .collect();
            let claimed: f32 = poses.iter().map(|(_, w)| *w).sum();
            if poses.is_empty() {
                return pose;
            }
            if claimed < 0.999 {
                poses.push((pose, 1.0 - claimed));
            }
            blend_many(&poses).unwrap_or_else(|| Pose::rest(rig))
        }
    }
}

/// How far an event window reaches back on a state's first evaluation, so an
/// event at the very start of a clip fires.
const START_NUDGE: f32 = 1e-6;

/// Event names passed between clip-local times `a` (previous) and `b`
/// (current). Forward playback fires `(a, b]`, reverse `[b, a)`; every loop
/// crossed in the frame fires. `first` reaches the window back by
/// [`START_NUDGE`] so events at the clip start fire.
#[allow(clippy::too_many_arguments)]
fn crossed_events(animator: &Animator, name: &str, clip: ClipRef, a: f32, b: f32, looping: bool, first: bool, out: &mut Vec<String>) {
    let cd = clip.duration();
    if cd <= 1e-6 {
        return;
    }
    let reverse = b < a;
    let looping = looping && clip.loops();
    let (a, b) = if looping { (a, b) } else { (a.clamp(0.0, cd), b.clamp(0.0, cd)) };
    let nudge = if first { START_NUDGE } else { 0.0 };
    let (lo, hi) = if reverse { (b, a + nudge) } else { (a - nudge, b) };
    if hi <= lo {
        return;
    }
    let mut marks: Vec<(f32, &str)> =
        animator.events.iter().filter(|e| e.clip == name).map(|e| (e.time, e.name.as_str())).collect();
    let mut owned: Vec<(f32, String)> = Vec::new();
    if let ClipRef::Asset(c) | ClipRef::Baked(c, _) = clip {
        for (i, e) in c.events.iter().enumerate() {
            match c.event_args.iter().find(|a| a.event == i) {
                Some(a) => owned.push((e.time, zestengine_core::AnimEventArgs { float: a.float, int: a.int, string: a.string.clone(), object: a.object.clone() }.encode(&e.name))),
                None => marks.push((e.time, e.name.as_str())),
            }
        }
    }
    // Events of the model's clip list (function name + parameters).
    if let Some(pb) = clip.playback() {
        for e in &pb.events {
            let args = zestengine_core::AnimEventArgs { float: e.float, int: e.int, string: e.string.clone(), object: e.object.clone() };
            if args.is_empty() { marks.push((e.time, e.name.as_str())) } else { owned.push((e.time, args.encode(&e.name))) }
        }
    }
    marks.extend(owned.iter().map(|(t, n)| (*t, n.as_str())));
    let mut hits: Vec<(f32, &str)> = Vec::new();
    for (time, ev) in marks {
        // A clip that does not loop plays its one pass.
        let (first_loop, last_loop) = if looping { ((lo / cd).floor() as i64, (hi / cd).floor() as i64) } else { (0, 0) };
        for k in first_loop..=last_loop {
            let t = k as f32 * cd + time;
            let inside = if reverse { t >= lo && t < hi } else { t > lo && t <= hi };
            if inside {
                hits.push((t, ev));
            }
        }
    }
    hits.sort_by(|x, y| if reverse { y.0.total_cmp(&x.0) } else { x.0.total_cmp(&y.0) });
    out.extend(hits.into_iter().map(|(_, ev)| ev.to_owned()));
}

/// Events of every clip a motion plays at an effective weight above 0
/// (`weight` is the state's share: transition weight times layer weight),
/// between normalized times `a` and `b`. A clip whose own blend weight just
/// dropped to 0 gets one last window; `memory` holds the clips' weights from
/// the previous call (and is refreshed).
#[allow(clippy::too_many_arguments)]
fn motion_events(
    animator: &Animator,
    motion: &Motion,
    params: &HashMap<String, f32>,
    clips: &ClipSet,
    a: f32,
    b: f32,
    looping: bool,
    first: bool,
    weight: f32,
    memory: &mut Vec<f32>,
    exclude: &[String],
    out: &mut Vec<String>,
) {
    if weight <= 0.0 {
        return;
    }
    let (leaves, total) = motion_leaves(motion, params, clips);
    for l in &leaves {
        let was = memory.get(l.index).copied().unwrap_or(0.0);
        if (l.weight * weight > 0.0 || was > 0.0) && !exclude.iter().any(|e| e == l.name) {
            crossed_events(animator, l.name, l.clip, l.time_at(a), l.time_at(b), looping, first, out);
        }
    }
    memory.clear();
    memory.resize(total, 0.0);
    for l in &leaves {
        memory[l.index] = l.weight;
    }
}

/// The change of a quantity sampled by `at` between clip-local times `a` and
/// `b`: across the loop seam a looping clip adds the travel to its end and from
/// its start (so the seam itself contributes nothing), a clamped one stops at
/// the ends.
fn span_delta<T: Copy + std::ops::Add<Output = T> + std::ops::Sub<Output = T>>(at: impl Fn(f32) -> T, cd: f32, a: f32, b: f32, looping: bool) -> T {
    let wrap = |t: f32| t.rem_euclid(cd);
    let (loop_a, loop_b) = ((a / cd).floor(), (b / cd).floor());
    if looping && loop_b > loop_a {
        (at(cd - 1e-4) - at(wrap(a))) + (at(wrap(b)) - at(0.0))
    } else if looping && loop_b < loop_a {
        (at(0.0) - at(wrap(a))) + (at(wrap(b)) - at(cd - 1e-4))
    } else if looping {
        at(wrap(b)) - at(wrap(a))
    } else {
        at(b.clamp(0.0, cd - 1e-4)) - at(a.clamp(0.0, cd - 1e-4))
    }
}

/// The pose of a clip that has a root, at clip-local time `t` (wrapping).
fn root_pose(rig: &Rig, clip: ClipRef, t: f32) -> Pose {
    let mut p = Pose::rest(rig);
    clip.sample(rig, t, true, &mut p);
    p
}

/// Root-joint motion of an imported clip between local times `a` and `b`
/// (all three axes, rig space).
fn root_motion(rig: &Rig, clip: ClipRef, a: f32, b: f32, looping: bool) -> Option<Vec3> {
    if !clip.has_root() {
        return None;
    }
    let root = rig.motion_root()?;
    let cd = clip.duration();
    if cd <= 1e-6 {
        return None;
    }
    Some(span_delta(|t| root_pose(rig, clip, t).trs[root].t, cd, a, b, looping && clip.loops()))
}

/// Yaw (radians, about the rig's up axis) the root turns through between local
/// times `a` and `b`, each step the shorter way round.
fn root_yaw(rig: &Rig, clip: ClipRef, a: f32, b: f32, looping: bool) -> Option<f32> {
    if !clip.has_root() {
        return None;
    }
    let root = rig.motion_root()?;
    let cd = clip.duration();
    if cd <= 1e-6 {
        return None;
    }
    // Sampling at the ends of the span: the sum of shorter-way steps through
    // the mid points keeps a turn of more than half a revolution whole.
    let yaw_at = |t: f32| zestengine_assets::anim_import::twist_angle_y(root_pose(rig, clip, t).trs[root].r);
    let (a, b) = (a, b);
    let steps = (((b - a).abs() / cd) * 16.0).ceil().clamp(1.0, 64.0) as usize;
    let looping = looping && clip.loops();
    let mut total = 0.0;
    for k in 0..steps {
        let (t0, t1) = (a + (b - a) * k as f32 / steps as f32, a + (b - a) * (k + 1) as f32 / steps as f32);
        let mut d = span_delta(yaw_at, cd, t0, t1, looping);
        d = (d + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
        total += d;
    }
    Some(total)
}

/// Root motion of a whole motion between normalized times `a` and `b`: each
/// clip's delta weighted by its blend weight and normalised by the total, so a
/// clip without root motion (an `.anim` asset) still pulls the blend toward
/// standing still. The components a clip bakes into its pose are not motion.
pub(crate) fn motion_root_delta(rig: &Rig, motion: &Motion, params: &HashMap<String, f32>, clips: &ClipSet, a: f32, b: f32, looping: bool) -> Vec3 {
    let (leaves, _) = motion_leaves(motion, params, clips);
    let total: f32 = leaves.iter().filter(|l| l.weight > 0.0).map(|l| l.weight).sum();
    if total <= 1e-6 {
        return Vec3::ZERO;
    }
    leaves
        .iter()
        .filter(|l| l.weight > 0.0)
        .filter_map(|l| {
            root_motion(rig, l.clip, l.time_at(a), l.time_at(b), looping).map(|d| {
                let mut d = if l.mirror && rig.mirror.is_some() { crate::anim_mirror::mirror_vec(d) } else { d };
                let bake = l.clip.bake();
                if bake.xz {
                    d.x = 0.0;
                    d.z = 0.0;
                }
                if bake.y {
                    d.y = 0.0;
                }
                d * (l.weight / total)
            })
        })
        .sum()
}

/// [`motion_root_delta`] for the turn: yaw in radians, weight averaged, baked
/// rotation excluded.
pub(crate) fn motion_root_yaw(rig: &Rig, motion: &Motion, params: &HashMap<String, f32>, clips: &ClipSet, a: f32, b: f32, looping: bool) -> f32 {
    let (leaves, _) = motion_leaves(motion, params, clips);
    let total: f32 = leaves.iter().filter(|l| l.weight > 0.0).map(|l| l.weight).sum();
    if total <= 1e-6 {
        return 0.0;
    }
    leaves
        .iter()
        .filter(|l| l.weight > 0.0 && !l.clip.bake().rot)
        .filter_map(|l| root_yaw(rig, l.clip, l.time_at(a), l.time_at(b), looping).map(|y| (if l.mirror { -y } else { y }) * (l.weight / total)))
        .sum()
}

/// Where a motion's clips start the root, and how much of each component the
/// pose keeps (baked) instead of leaving it to root motion.
#[derive(Clone, Copy, Debug)]
struct RootStart {
    t: Vec3,
    yaw: f32,
    /// Weight (0..1) of the blend that keeps its authored ground-plane / height / yaw.
    keep_xz: f32,
    keep_y: f32,
    keep_rot: f32,
}

impl RootStart {
    fn lerp(self, o: RootStart, f: f32) -> RootStart {
        RootStart {
            t: self.t.lerp(o.t, f),
            yaw: self.yaw + (o.yaw - self.yaw) * f,
            keep_xz: self.keep_xz + (o.keep_xz - self.keep_xz) * f,
            keep_y: self.keep_y + (o.keep_y - self.keep_y) * f,
            keep_rot: self.keep_rot + (o.keep_rot - self.keep_rot) * f,
        }
    }
}

/// The root at the start of a motion's clips (weighted).
fn motion_root_start(rig: &Rig, motion: &Motion, params: &HashMap<String, f32>, clips: &ClipSet) -> Option<RootStart> {
    let root = rig.motion_root()?;
    let (mut sum, mut total) = (RootStart { t: Vec3::ZERO, yaw: 0.0, keep_xz: 0.0, keep_y: 0.0, keep_rot: 0.0 }, 0.0);
    let (leaves, _) = motion_leaves(motion, params, clips);
    for l in leaves.iter().filter(|l| l.weight > 0.0) {
        if l.clip.has_root() {
            let mut p = Pose::rest(rig);
            l.clip.sample(rig, l.time_at(0.0), true, &mut p);
            if l.mirror {
                rig.mirror_pose(&mut p);
            }
            let bake = l.clip.bake();
            let flag = |b: bool| if b { 1.0 } else { 0.0 };
            sum.t += p.trs[root].t * l.weight;
            sum.yaw += zestengine_assets::anim_import::twist_angle_y(p.trs[root].r) * l.weight;
            sum.keep_xz += flag(bake.xz) * l.weight;
            sum.keep_y += flag(bake.y) * l.weight;
            sum.keep_rot += flag(bake.rot) * l.weight;
            total += l.weight;
        }
    }
    (total > 1e-6).then(|| RootStart { t: sum.t / total, yaw: sum.yaw / total, keep_xz: sum.keep_xz / total, keep_y: sum.keep_y / total, keep_rot: sum.keep_rot / total })
}

// -------------------------------------------------------------- controller

/// Result of advancing an Animator one frame.
pub struct AnimOutput {
    pub pose: Pose,
    /// Root motion (rig space, all axes), blended across clips, cross-fades
    /// and override layers whose mask covers the root.
    pub root_delta: Vec3,
    /// Root motion turn (radians about the rig's up axis) of clips that do not
    /// bake their rotation into the pose; applied to the object like `root_delta`.
    pub root_yaw: f32,
}

fn state_duration(machine: &StateMachine, index: usize, params: &HashMap<String, f32>, clips: &ClipSet) -> f32 {
    machine.states.get(index).map(|s| motion_duration(&s.motion, params, clips)).unwrap_or(0.0)
}

/// Normalizing length: a state without a clip counts as one second so exit
/// times still work.
fn norm_len(d: f32) -> f32 {
    if d > 1e-4 { d } else { 1.0 }
}

/// How far (normalized) past its exit time a state moved this frame, if it
/// crossed it. An exit time up to 1 recurs every loop (`exit + k`), so it
/// fires once per loop on the frame it is crossed; above 1 it fires once, on
/// the frame it is crossed. Reverse playback mirrors the test.
fn exit_crossing(prev: f32, now: f32, exit: f32) -> Option<f32> {
    if now > prev {
        let at = if exit <= 1.0 {
            let k = (now - exit).floor();
            if k < 0.0 {
                return None;
            }
            exit + k
        } else {
            exit
        };
        (at > prev && at <= now).then_some(now - at)
    } else if now < prev {
        let at = if exit <= 1.0 { exit + (now - exit).ceil().max(0.0) } else { exit };
        (at >= now && at < prev).then_some(at - now)
    } else {
        None
    }
}

/// The normalized time a state's time parameter dictates, when it has one
/// (a missing parameter reads 0, like a condition's).
fn driven_time(state: &AnimState, params: &HashMap<String, f32>) -> Option<f32> {
    (!state.time_parameter.is_empty()).then(|| params.get(&state.time_parameter).copied().unwrap_or(0.0))
}

/// Raise a state callback, when the state names behaviours to receive it.
fn note(animator: &mut Animator, machine: &StateMachine, layer: usize, state: usize, kind: StateCallbackKind, n: f32) {
    let Some(st) = machine.states.get(state) else {
        return;
    };
    if !st.behaviours.is_empty() {
        animator.state.callbacks.push(StateCallback { layer, state: st.name.clone(), kind, normalized_time: n });
    }
}

/// Length of a state for the layer's timing: its own motion, averaged with
/// the same state of every sync layer that contributes to timing, each at its
/// layer weight (the layer's own share is 1). The combining formula is a
/// weight-normalised average, the natural reading of the timing weights.
fn timed_len(ctrl: &AnimatorController, layer: usize, state: usize, params: &HashMap<String, f32>, clips: &ClipSet) -> f32 {
    let own = state_duration(&ctrl.layers[layer].machine, state, params, clips);
    let (mut sum, mut weight) = (own, 1.0);
    for l in ctrl.layers.iter().filter(|l| l.sync_timing && l.sync_with == Some(layer)) {
        let w = l.weight.clamp(0.0, 1.0);
        if w > 0.0 {
            sum += w * state_duration(&l.machine, state, params, clips);
            weight += w;
        }
    }
    sum / weight
}

/// Snapshot of a pose to fade out from after an interruption.
fn freeze(pose: &Pose) -> FrozenPose {
    if pose.muscle.is_some() {
        return freeze(&pose.resolved());
    }
    FrozenPose {
        t: pose.trs.iter().map(|t| t.t).collect(),
        r: pose.trs.iter().map(|t| t.r).collect(),
        s: pose.trs.iter().map(|t| t.s).collect(),
        scalar: pose.scalar.clone(),
        touched: pose.touched.clone(),
        props: pose.props.clone(),
    }
}

fn thaw(rig: &Rig, f: &FrozenPose) -> Pose {
    let mut pose = Pose::rest(rig);
    let n = pose.trs.len().min(f.t.len());
    for i in 0..n {
        pose.trs[i] = Trs { t: f.t[i], r: f.r[i], s: f.s[i] };
        pose.scalar[i] = f.scalar[i];
        pose.touched[i] = f.touched[i];
    }
    pose.props = f.props.clone();
    pose
}

/// Write Defaults off: channels the state does not animate hold the value
/// they had last frame (the previous pose joins the mix at full weight)
/// instead of blending toward the defaults.
fn hold_untouched(pose: &mut Pose, held: &FrozenPose) {
    let n = pose.trs.len().min(held.t.len());
    for i in 0..n {
        let missing = held.touched[i] & !pose.touched[i];
        if missing & CH_T != 0 {
            pose.trs[i].t = held.t[i];
        }
        if missing & CH_R != 0 {
            pose.trs[i].r = held.r[i];
        }
        if missing & CH_S != 0 {
            pose.trs[i].s = held.s[i];
        }
        if missing & CH_F != 0 {
            pose.scalar[i] = held.scalar[i];
        }
        pose.touched[i] |= missing;
    }
    hold_props(&mut pose.props, &held.props);
}

/// One state's pose at normalized time `n`, honouring its Write Defaults and
/// Loop Pose settings. `held` is the layer's pose from the last frame.
fn state_pose(st: &AnimState, params: &HashMap<String, f32>, clips: &ClipSet, rig: &Rig, n: f32, held: Option<&FrozenPose>) -> Pose {
    let mut pose = sample_motion_lazy(&st.motion, params, clips, rig, n, st.looping, st.loop_pose);
    if !st.write_defaults
        && let Some(h) = held
    {
        pose.resolve_muscles();
        hold_untouched(&mut pose, h);
    }
    pose
}

/// The layer's pose now: the current state, cross-faded from the previous
/// one (or from the frozen pose of an interrupted fade).
fn layer_pose(machine: &StateMachine, lp: &LayerPlayback, params: &HashMap<String, f32>, clips: &ClipSet, rig: &Rig) -> Pose {
    let st = &machine.states[lp.state];
    let held = lp.held.as_ref();
    let mut pose = state_pose(st, params, clips, rig, lp.normalized_time, held);
    let from = match (&lp.frozen, lp.previous) {
        (Some(f), _) => Some(thaw(rig, f)),
        (None, Some((p, t))) => Some(state_pose(&machine.states[p], params, clips, rig, t, held)),
        (None, None) => None,
    };
    if let Some(from) = from {
        pose = blend_pose(&from, &pose, lp.transition_progress());
    }
    pose
}

/// One state change on a layer.
#[derive(Clone, Copy)]
struct Go {
    to: usize,
    /// Cross-fade length in seconds (0 = cut).
    fade: f32,
    transition: Option<usize>,
    /// Normalized start time in the target.
    offset: f32,
    /// Real seconds the exit time was passed by, carried into the fade and
    /// the target's clock.
    overshoot: f32,
    /// The state whose transition list fired, when interrupting a fade (None
    /// = Any State, which leaves the fade's source as the owner).
    owner: Option<usize>,
    /// Weight of the layer, for the events of the overshoot window.
    layer_weight: f32,
}

/// Start `go` on `layer`. During a fade this is an interruption: the blended
/// pose is frozen and the new fade starts from that snapshot instead of
/// dropping the old source.
fn begin_transition(
    animator: &mut Animator,
    ctrl: &AnimatorController,
    rig: &Rig,
    layer: usize,
    go: Go,
    params: &HashMap<String, f32>,
    clips: &ClipSet,
) {
    let machine = &ctrl.layers[layer].machine;
    let target = &machine.states[go.to];
    // A fade the exit time already ran past is a cut.
    let fade = if go.overshoot >= go.fade { 0.0 } else { go.fade.max(0.0) };
    let (old_src, old_dst, from_name) = {
        let lp = &animator.state.layers[layer];
        (lp.previous, (lp.state, lp.normalized_time), lp.state_name.clone())
    };
    // `source` stays the state the new fade is listed under; the rest stop
    // being evaluated now. A cut, or a source that fades out, exits when it ends.
    let mut leaving: Vec<usize> = Vec::new();
    let mut frozen = None;
    // Whether the fade's source is the state being left now (not the one it
    // was already fading from), to carry that state's clip weights over.
    let mut from_current = true;
    let source = match old_src {
        Some(src) => {
            if fade > 0.0 {
                frozen = Some(freeze(&layer_pose(machine, &animator.state.layers[layer], params, clips, rig)));
            }
            let keep = match go.owner {
                Some(o) if o == old_dst.0 => old_dst,
                _ => {
                    from_current = false;
                    src
                }
            };
            for s in [src.0, old_dst.0] {
                if (s != keep.0 || fade <= 0.0) && !leaving.contains(&s) {
                    leaving.push(s);
                }
            }
            keep
        }
        None => {
            if fade <= 0.0 {
                leaving.push(old_dst.0);
            }
            old_dst
        }
    };
    for s in leaving {
        if let Some(st) = machine.states.get(s)
            && !st.on_exit.is_empty()
        {
            animator.state.fired.push(st.on_exit.clone());
        }
        note(animator, machine, layer, s, StateCallbackKind::Exit, old_dst.1);
    }
    if !target.on_enter.is_empty() {
        animator.state.fired.push(target.on_enter.clone());
    }
    note(animator, machine, layer, go.to, StateCallbackKind::Enter, 0.0);
    // Drained by the play session; bounded when nothing drains it (editor preview).
    if animator.state.state_changes.len() >= 32 {
        animator.state.state_changes.remove(0);
    }
    animator.state.state_changes.push((layer, from_name, target.name.clone()));
    let len = norm_len(timed_len(ctrl, layer, go.to, params, clips));
    let rate = {
        let p = if target.speed_parameter.is_empty() { 1.0 } else { params.get(&target.speed_parameter).copied().unwrap_or(1.0) };
        target.speed * p * animator.speed
    };
    let start = go.offset.max(0.0);
    let lp = &mut animator.state.layers[layer];
    let current_weights = std::mem::take(&mut lp.leaf_weights);
    let source_weights = if fade <= 0.0 {
        Vec::new()
    } else if from_current {
        current_weights
    } else {
        std::mem::take(&mut lp.src_leaf_weights)
    };
    lp.src_leaf_weights = source_weights;
    lp.previous = if fade > 0.0 { Some(source) } else { None };
    lp.frozen = if fade > 0.0 { frozen } else { None };
    lp.fade_elapsed = if fade > 0.0 { go.overshoot.max(0.0) } else { 0.0 };
    lp.fade_duration = fade;
    lp.state = go.to;
    lp.state_name = target.name.clone();
    lp.normalized_time = start + go.overshoot.max(0.0) * rate / len;
    lp.time = lp.normalized_time * len;
    lp.transition = if fade > 0.0 { go.transition } else { None };
    lp.just_entered = true;
    // The exit time was passed mid-frame: the target already played that
    // much, so its first window (start events included) fires now.
    if go.overshoot > 1e-6 {
        let (end, w) = (lp.normalized_time, if fade > 0.0 { lp.fade_elapsed / fade } else { 1.0 });
        lp.just_entered = false;
        let mut fired = Vec::new();
        let mut memory = std::mem::take(&mut lp.leaf_weights);
        motion_events(animator, &target.motion, params, clips, start, end, target.looping, true, w * go.layer_weight, &mut memory, &[], &mut fired);
        animator.state.layers[layer].leaf_weights = memory;
        animator.state.fired.extend(fired);
    }
}

/// (Re)start the controller: parameter defaults (kept when already set from
/// code) and every layer in its default state.
fn init_controller(animator: &mut Animator, ctrl: &AnimatorController, path: &str) {
    for p in &ctrl.parameters {
        animator.state.params.entry(p.name.clone()).or_insert(p.default);
    }
    animator.state.layers = ctrl
        .layers
        .iter()
        .map(|l| {
            let state = l.machine.default_index().unwrap_or(0);
            LayerPlayback {
                state,
                state_name: l.machine.states.get(state).map(|s| s.name.clone()).unwrap_or_default(),
                just_entered: true,
                ..Default::default()
            }
        })
        .collect();
    for (li, l) in ctrl.layers.iter().enumerate() {
        if let Some(i) = l.machine.default_index()
            && let Some(s) = l.machine.states.get(i)
        {
            if !s.on_enter.is_empty() {
                animator.state.fired.push(s.on_enter.clone());
            }
            note(animator, &l.machine, li, i, StateCallbackKind::Enter, 0.0);
        }
    }
    animator.state.controller_ready = Some(path.to_owned());
    animator.state.started = true;
    animator.state.playing = true;
}

/// Which list of transitions a candidate comes from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ListKind {
    Any,
    Source,
    Destination,
}

/// What a plain layer's state clock did this frame, for the sync layers that
/// follow it (their events fire against the same clock).
#[derive(Clone, Copy)]
struct SyncFrame {
    /// The state that ran this frame (before any transition it took).
    cur: usize,
    prev: f32,
    now: f32,
    first: bool,
    fade_w: Option<f32>,
    src: Option<StateClock>,
    frozen: bool,
}

/// Fire the animation events of a sync layer's own clips. Callbacks stay the
/// source layer's (the states and their behaviours are shared); events of
/// clips the source layer also plays are the source's too, so only clips it
/// does not play fire here.
fn sync_events(animator: &mut Animator, ctrl: &AnimatorController, li: usize, si: usize, f: &SyncFrame, clips: &ClipSet) {
    let (layer, source) = (&ctrl.layers[li], &ctrl.layers[si]);
    let (machine, smachine) = (&layer.machine, &source.machine);
    if f.cur >= machine.states.len() || f.cur >= smachine.states.len() {
        return;
    }
    let params = animator.state.params.clone();
    let weight = layer.weight.clamp(0.0, 1.0);
    let names_of = |m: &Motion| motion_leaves(m, &params, clips).0.iter().map(|l| l.name.to_owned()).collect::<Vec<_>>();
    let st = &machine.states[f.cur];
    let exclude = names_of(&smachine.states[f.cur].motion);
    let mut fired = Vec::new();
    let mut memory = std::mem::take(&mut animator.state.layers[li].leaf_weights);
    motion_events(animator, &st.motion, &params, clips, f.prev, f.now, st.looping, f.first, weight * f.fade_w.unwrap_or(1.0), &mut memory, &exclude, &mut fired);
    animator.state.layers[li].leaf_weights = memory;
    if let (Some(w), Some(sc), false) = (f.fade_w, f.src, f.frozen)
        && sc.state < machine.states.len()
        && sc.state < smachine.states.len()
    {
        let ps = &machine.states[sc.state];
        let exclude = names_of(&smachine.states[sc.state].motion);
        let share = if w < 1.0 { 1.0 - w } else { 1.0 };
        let mut memory = std::mem::take(&mut animator.state.layers[li].src_leaf_weights);
        motion_events(animator, &ps.motion, &params, clips, sc.prev, sc.now, ps.looping, false, weight * share, &mut memory, &exclude, &mut fired);
        animator.state.layers[li].src_leaf_weights = memory;
    }
    animator.state.fired.extend(fired);
}

/// A state's normalized time across this frame, to test exit times against.
#[derive(Clone, Copy)]
struct StateClock {
    state: usize,
    prev: f32,
    now: f32,
    /// State length in seconds, to turn an overshoot back into time.
    len: f32,
}

/// How much of a frame to evaluate (see [`FrameTiming`] and culling).
#[derive(Clone, Copy, Default)]
struct Eval {
    /// Posing, IK and the pose output are skipped (the state machine, events
    /// and callbacks still run).
    skip_pose: bool,
    /// Re-pose without moving time or taking transitions (a `Fixed` Animator
    /// between two fixed steps).
    hold: bool,
}

/// Advance an Animator driven by a controller by `dt` seconds.
pub fn advance_controller(
    animator: &mut Animator,
    ctrl: &AnimatorController,
    path: &str,
    rig: &Rig,
    clips: &ClipSet,
    dt: f32,
) -> AnimOutput {
    advance_controller_ex(animator, ctrl, path, rig, clips, dt, Eval::default())
}

fn advance_controller_ex(
    animator: &mut Animator,
    ctrl: &AnimatorController,
    path: &str,
    rig: &Rig,
    clips: &ClipSet,
    dt: f32,
    eval: Eval,
) -> AnimOutput {
    // Sub-state machines run flattened (`Outer/Inner` state names); assets
    // from `AnimAssets` arrive flattened already.
    let flat;
    let ctrl = if ctrl.layers.iter().any(|l| l.machine.needs_flatten()) {
        flat = ctrl.flattened();
        &flat
    } else {
        ctrl
    };
    animator.state.fired.clear();
    animator.state.callbacks.clear();
    if animator.state.controller_ready.as_deref() != Some(path) || animator.state.layers.len() != ctrl.layers.len() {
        init_controller(animator, ctrl, path);
    }

    // Forced states from code.
    for req in std::mem::take(&mut animator.state.requests) {
        let target = match req.layer {
            Some(l) => ctrl.layers.get(l).and_then(|layer| layer.machine.state_index(&req.state).map(|s| (l, s))),
            None => ctrl.layers.iter().enumerate().find_map(|(l, layer)| layer.machine.state_index(&req.state).map(|s| (l, s))),
        };
        if let Some((l, s)) = target {
            let params = animator.state.params.clone();
            let go = Go { to: s, fade: req.fade, transition: None, offset: 0.0, overshoot: 0.0, owner: None, layer_weight: 1.0 };
            begin_transition(animator, ctrl, rig, l, go, &params, clips);
            animator.state.playing = true;
        }
    }

    let playing = animator.state.playing && !eval.hold;
    let aspeed = animator.speed;
    let root = rig.motion_root();
    let mut out: Option<Pose> = None;
    let mut root_delta = Vec3::ZERO;
    let mut root_yaw = 0.0f32;
    // Triggers a taken transition used. Cleared only once every layer has
    // run, so a trigger can drive transitions in several layers in one frame.
    let mut consumed: Vec<String> = Vec::new();
    // Each plain layer's clock this frame, for the sync layers that follow it.
    let mut frames: Vec<Option<SyncFrame>> = vec![None; ctrl.layers.len()];

    for (li, layer) in ctrl.layers.iter().enumerate() {
        let machine = &layer.machine;
        if machine.states.is_empty() {
            continue;
        }
        // A sync layer plays its source layer's state machine and clock with
        // its own motions (the machine was resolved by `flattened`). Its
        // callbacks are the source's (shared states), its events come from the
        // clips it plays itself. A source in an interrupted fade is mirrored by
        // freezing this layer's own pose at the same moment, so the fade
        // leaves from this layer's blend and not from the source state's.
        if let Some(si) = layer.sync_with.filter(|&si| si < li && ctrl.layers[si].sync_with.is_none()) {
            if let Some(f) = frames[si] {
                sync_events(animator, ctrl, li, si, &f, clips);
            }
            let (before, after) = animator.state.layers.split_at_mut(li);
            let (src, lp) = (&before[si], &mut after[0]);
            if src.state >= machine.states.len() {
                continue;
            }
            // Snapshot on the frame an interruption starts (or restarts).
            let snap = (src.frozen.is_some()
                && (lp.frozen.is_none() || src.fade_elapsed < lp.fade_elapsed)
                && lp.state < machine.states.len()
                && lp.previous.is_none_or(|(p, _)| p < machine.states.len()))
            .then(|| freeze(&layer_pose(machine, lp, &animator.state.params, clips, rig)));
            let kept = lp.frozen.take();
            lp.state = src.state;
            if lp.state_name != src.state_name {
                lp.state_name = src.state_name.clone();
            }
            lp.time = src.time;
            lp.normalized_time = src.normalized_time;
            lp.previous = src.previous.filter(|&(p, _)| p < machine.states.len());
            lp.fade_elapsed = src.fade_elapsed;
            lp.fade_duration = src.fade_duration;
            lp.transition = src.transition;
            lp.frozen = if src.frozen.is_some() { snap.or(kept) } else { None };
            lp.just_entered = false;
            if !eval.skip_pose {
                let pose = layer_pose(machine, &animator.state.layers[li], &animator.state.params, clips, rig);
                out = Some(match out {
                    None => pose,
                    Some(base) => compose_layer(rig, base, &pose, layer.weight, layer.blend, &layer.mask),
                });
            }
            continue;
        }
        // The asset may have been edited live: keep indices valid.
        {
            let lp = &mut animator.state.layers[li];
            if lp.state >= machine.states.len() || machine.states[lp.state].name != lp.state_name {
                match machine.state_index(&lp.state_name).or(machine.default_index()) {
                    Some(i) => {
                        lp.state = i;
                        lp.state_name = machine.states[i].name.clone();
                    }
                    None => continue,
                }
            }
            if lp.previous.is_some_and(|(p, _)| p >= machine.states.len()) {
                lp.previous = None;
                lp.frozen = None;
            }
            if lp.transition.is_some_and(|t| t >= machine.transitions.len()) {
                lp.transition = None;
            }
        }

        let params = animator.state.params.clone();
        let speed_of = |i: usize| {
            let s = &machine.states[i];
            let p = if s.speed_parameter.is_empty() { 1.0 } else { params.get(&s.speed_parameter).copied().unwrap_or(1.0) };
            s.speed * p * aspeed
        };
        // The base layer always plays at full weight.
        let layer_weight = if li == 0 { 1.0 } else { layer.weight.clamp(0.0, 1.0) };
        let cur = animator.state.layers[li].state;
        let st = &machine.states[cur];
        let len = norm_len(timed_len(ctrl, li, cur, &params, clips));
        // The state's clock is its normalized time, integrated with the
        // length of the moment: weights that shift a blend tree's length
        // change the rate smoothly and never jump the phase.
        let step = if playing { dt * speed_of(cur) / len } else { 0.0 };
        let first = animator.state.layers[li].just_entered;
        let prev_norm = animator.state.layers[li].normalized_time;
        // The fading-out state: its clock across this frame.
        let src_state = animator.state.layers[li].previous;
        let src_len = src_state.map(|(p, _)| norm_len(timed_len(ctrl, li, p, &params, clips)));
        let mut src_clock: Option<StateClock> = None;
        // Cross-fade weight of the target while fading; None when not fading.
        let mut fade_w: Option<f32> = None;
        {
            let lp = &mut animator.state.layers[li];
            lp.normalized_time += step;
            if let Some(t) = driven_time(st, &params) {
                lp.normalized_time = t;
            }
            if !st.looping {
                lp.normalized_time = lp.normalized_time.clamp(0.0, 1.0);
            }
            lp.time = lp.normalized_time * len;
            if let (Some((p, t)), Some(sl)) = (lp.previous.as_mut(), src_len) {
                let before = *t;
                if playing {
                    *t += dt * speed_of(*p) / sl;
                }
                if let Some(v) = driven_time(&machine.states[*p], &params) {
                    *t = v;
                }
                src_clock = Some(StateClock { state: *p, prev: before, now: *t, len: sl });
            }
            // The fade clock follows the Animator's speed, not the source
            // state's own.
            if playing {
                lp.fade_elapsed += dt * aspeed.abs();
            }
            if lp.previous.is_some() || lp.frozen.is_some() {
                fade_w = Some(if lp.fade_duration > 0.0 { (lp.fade_elapsed / lp.fade_duration).clamp(0.0, 1.0) } else { 1.0 });
            }
        }

        // Events: the target and the fading-out state, each at its share of
        // the blend. A source gets one last window on the frame its weight
        // reaches 0. A frozen source plays nothing.
        let new_time = animator.state.layers[li].normalized_time;
        let frozen = animator.state.layers[li].frozen.is_some();
        {
            let dest_weight = layer_weight * fade_w.unwrap_or(1.0);
            let mut fired = Vec::new();
            let mut memory = std::mem::take(&mut animator.state.layers[li].leaf_weights);
            motion_events(animator, &st.motion, &params, clips, prev_norm, new_time, st.looping, first, dest_weight, &mut memory, &[], &mut fired);
            animator.state.layers[li].leaf_weights = memory;
            if let (Some(w), Some(sc), false) = (fade_w, src_clock, frozen) {
                let ps = &machine.states[sc.state];
                let share = if w < 1.0 { 1.0 - w } else { 1.0 };
                let mut memory = std::mem::take(&mut animator.state.layers[li].src_leaf_weights);
                motion_events(animator, &ps.motion, &params, clips, sc.prev, sc.now, ps.looping, false, layer_weight * share, &mut memory, &[], &mut fired);
                animator.state.layers[li].src_leaf_weights = memory;
            }
            animator.state.fired.extend(fired);
            frames[li] = Some(SyncFrame { cur, prev: prev_norm, now: new_time, first, fade_w, src: src_clock, frozen });
        }

        // Update callbacks: every frame of a state but the one it was entered on.
        if playing && !first {
            note(animator, machine, li, cur, StateCallbackKind::Update, new_time);
        }
        if let (true, Some(sc), false) = (playing, src_clock, frozen) {
            note(animator, machine, li, sc.state, StateCallbackKind::Update, sc.now);
        }

        // Root motion: base layer, plus override layers whose mask covers the root.
        if animator.root_motion {
            let covers = li == 0
                || (layer.blend == LayerBlend::Override
                    && match (rig.mask(&layer.mask), root) {
                        (None, _) => true,
                        (Some(m), Some(r)) => m.get(r).copied().unwrap_or(false),
                        _ => false,
                    });
            if covers {
                let w = fade_w.unwrap_or(1.0);
                let mut d = motion_root_delta(rig, &st.motion, &params, clips, prev_norm, new_time, st.looping) * w;
                if let (Some(sc), false) = (src_clock, frozen) {
                    let ps = &machine.states[sc.state];
                    d += motion_root_delta(rig, &ps.motion, &params, clips, sc.prev, sc.now, ps.looping) * (1.0 - w);
                }
                root_delta = if li == 0 { d } else { root_delta.lerp(d, layer_weight) };
                let mut y = motion_root_yaw(rig, &st.motion, &params, clips, prev_norm, new_time, st.looping) * w;
                if let (Some(sc), false) = (src_clock, frozen) {
                    let ps = &machine.states[sc.state];
                    y += motion_root_yaw(rig, &ps.motion, &params, clips, sc.prev, sc.now, ps.looping) * (1.0 - w);
                }
                root_yaw = if li == 0 { y } else { root_yaw + (y - root_yaw) * layer_weight };
            }
        }

        // The fade ended: the source stops being evaluated and exits.
        {
            let lp = &mut animator.state.layers[li];
            let fading = lp.previous.is_some() || lp.frozen.is_some();
            if fading && (lp.fade_duration <= 0.0 || lp.fade_elapsed >= lp.fade_duration) {
                if let Some((p, t)) = lp.previous {
                    if !machine.states[p].on_exit.is_empty() {
                        animator.state.fired.push(machine.states[p].on_exit.clone());
                    }
                    note(animator, machine, li, p, StateCallbackKind::Exit, t);
                }
                let lp = &mut animator.state.layers[li];
                lp.previous = None;
                lp.frozen = None;
                lp.transition = None;
                lp.src_leaf_weights.clear();
            } else if !fading {
                lp.transition = None;
            }
            animator.state.layers[li].just_entered = false;
        }

        // Transitions. Not fading: Any State, then the current state. Fading:
        // the active transition's interruption source picks the lists.
        if playing {
            let lp = &animator.state.layers[li];
            let (fading, active) = (lp.previous.is_some() || lp.frozen.is_some(), lp.transition);
            let cur_clock = StateClock { state: cur, prev: if first { prev_norm - START_NUDGE } else { prev_norm }, now: lp.normalized_time, len };
            let mut lists: Vec<(ListKind, StateClock)> = Vec::new();
            if !fading {
                lists.push((ListKind::Any, cur_clock));
                lists.push((ListKind::Source, cur_clock));
            } else {
                let at = active.map(|ti| &machine.transitions[ti]);
                let kind = at.map(|t| t.interruption_source()).unwrap_or(InterruptionSource::Destination);
                // The source of a fade whose previous state was dropped by
                // a live edit has no clock: it never interrupts.
                let any_clock = src_clock.unwrap_or(cur_clock);
                let (source, dest) = (src_clock.map(|c| (ListKind::Source, c)), (ListKind::Destination, cur_clock));
                let ordered: Vec<Option<(ListKind, StateClock)>> = match kind {
                    InterruptionSource::None => vec![],
                    InterruptionSource::Source => vec![source],
                    InterruptionSource::Destination => vec![Some(dest)],
                    InterruptionSource::SourceThenDestination => vec![source, Some(dest)],
                    InterruptionSource::DestinationThenSource => vec![Some(dest), source],
                };
                if kind != InterruptionSource::None {
                    lists.push((ListKind::Any, any_clock));
                }
                lists.extend(ordered.iter().flatten().copied());
            }
            let ordered_only = fading && active.is_some_and(|ti| machine.transitions[ti].ordered_interruption);
            let owner_kind = active.map(|ti| if machine.transitions[ti].from.is_none() { ListKind::Any } else { ListKind::Source });
            let mut fire: Option<(usize, usize, f32, ListKind, StateClock)> = None;
            'lists: for (kind, clock) in &lists {
                let list_state = match kind {
                    ListKind::Any => None,
                    _ => Some(machine.states[clock.state].name.as_str()),
                };
                for (ti, t) in machine.transitions.iter().enumerate() {
                    if t.from.as_deref() != list_state {
                        continue;
                    }
                    if Some(ti) == active {
                        if ordered_only && owner_kind == Some(*kind) {
                            break 'lists;
                        }
                        continue;
                    }
                    let Some(to) = machine.state_index(&t.to) else {
                        continue;
                    };
                    if t.from.is_none() && to == cur && !t.can_transition_to_self && !fading {
                        continue;
                    }
                    if !t.has_exit_time && t.conditions.is_empty() {
                        continue;
                    }
                    let mut overshoot = 0.0;
                    if t.has_exit_time {
                        match exit_crossing(clock.prev, clock.now, t.exit_time) {
                            Some(o) => overshoot = o,
                            None => continue,
                        }
                    }
                    if !t.conditions.iter().all(|c| c.op.test(params.get(&c.parameter).copied().unwrap_or(0.0), c.threshold)) {
                        continue;
                    }
                    fire = Some((ti, to, overshoot, *kind, *clock));
                    break 'lists;
                }
            }
            if let Some((ti, to, overshoot, kind, clock)) = fire {
                let t = &machine.transitions[ti];
                for c in &t.conditions {
                    if ctrl.parameter(&c.parameter).is_some_and(|p| p.kind == AnimParamKind::Trigger) {
                        consumed.push(c.parameter.clone());
                    }
                }
                let rate = speed_of(clock.state).abs();
                let secs = if rate > 1e-6 { overshoot * clock.len / rate } else { 0.0 };
                let owner = match kind {
                    ListKind::Any => None,
                    ListKind::Source if fading => Some(clock.state),
                    ListKind::Destination => Some(clock.state),
                    ListKind::Source => None,
                };
                let go = Go { to, fade: t.duration, transition: Some(ti), offset: t.offset, overshoot: secs, owner, layer_weight };
                begin_transition(animator, ctrl, rig, li, go, &params, clips);
            }
        }

        // Sample: current state, cross-faded from the previous one.
        if eval.skip_pose {
            continue;
        }
        let lp = &animator.state.layers[li];
        let mut pose = layer_pose(machine, lp, &params, clips, rig);
        // Write Defaults off keeps the pose to fill next frame's gaps from.
        let holds = machine.states.iter().any(|s| !s.write_defaults);
        if li == 0 && animator.root_motion && rig.skeletal {
            pin_root(rig, &mut pose, machine, lp, &params, clips);
        }
        animator.state.layers[li].held = holds.then(|| freeze(&pose));

        out = Some(match out {
            None => pose,
            Some(base) => compose_layer(rig, base, &pose, layer.weight, layer.blend, &layer.mask),
        });
    }

    for name in consumed {
        animator.state.params.insert(name, 0.0);
    }
    let layer0 = animator.state.layers.first().map(|l| (l.state_name.clone(), l.time));
    if let Some((name, time)) = layer0 {
        animator.state.current = Some(name);
        animator.state.time = time;
    }

    // The root-motion and IK passes: each layer's current state is asked in
    // turn, then everything raised this frame is delivered to the behaviours.
    let ik_pass = ctrl.layers.iter().any(|l| l.ik_pass);
    for (li, layer) in ctrl.layers.iter().enumerate() {
        let Some(lp) = animator.state.layers.get(li) else {
            continue;
        };
        if layer.sync_with.is_some() || lp.state >= layer.machine.states.len() {
            continue;
        }
        let (state, n) = (lp.state, lp.normalized_time);
        if animator.root_motion {
            note(animator, &layer.machine, li, state, StateCallbackKind::Move, n);
        }
        if layer.ik_pass && !eval.skip_pose {
            note(animator, &layer.machine, li, state, StateCallbackKind::Ik, n);
        }
    }
    dispatch_callbacks(animator, ctrl, &mut root_delta);
    let mut pose = out.unwrap_or_else(|| Pose::rest(rig));
    pose.resolve_muscles();
    if ik_pass && !eval.skip_pose {
        apply_ik(rig, &mut pose, &animator.state.ik_goals);
    }
    AnimOutput { pose, root_delta, root_yaw }
}

/// Deliver this frame's state callbacks to the behaviours the states name.
fn dispatch_callbacks(animator: &mut Animator, ctrl: &AnimatorController, root_delta: &mut Vec3) {
    if animator.state.callbacks.is_empty() {
        return;
    }
    let calls = animator.state.callbacks.clone();
    let mut instances = std::mem::take(&mut animator.state.behaviours);
    let mut goals = std::mem::take(&mut animator.state.ik_goals);
    for call in &calls {
        let Some(layer) = ctrl.layers.get(call.layer) else {
            continue;
        };
        let Some(state) = layer.machine.state_index(&call.state).map(|i| &layer.machine.states[i]) else {
            continue;
        };
        let layer_weight = if call.layer == 0 { 1.0 } else { layer.weight.clamp(0.0, 1.0) };
        for name in &state.behaviours {
            let key = (call.layer, call.state.clone(), name.clone());
            if !instances.0.contains_key(&key)
                && let Some(b) = zestengine_core::create_state_behaviour(name)
            {
                instances.0.insert(key.clone(), b);
            }
            let Some(behaviour) = instances.0.get_mut(&key) else {
                continue;
            };
            let mut ctx = StateBehaviourCtx {
                layer: call.layer,
                state: &call.state,
                normalized_time: call.normalized_time,
                layer_weight,
                params: &mut animator.state.params,
                root_delta,
                ik_goals: &mut goals,
                settings: state.behaviour_settings.get(name).map_or("", String::as_str),
            };
            animator.state.params_touched = true;
            match call.kind {
                StateCallbackKind::Enter => behaviour.on_state_enter(&mut ctx),
                StateCallbackKind::Update => behaviour.on_state_update(&mut ctx),
                StateCallbackKind::Exit => behaviour.on_state_exit(&mut ctx),
                StateCallbackKind::Move => behaviour.on_state_move(&mut ctx),
                StateCallbackKind::Ik => behaviour.on_state_ik(&mut ctx),
            }
        }
    }
    animator.state.behaviours = instances;
    animator.state.ik_goals = goals;
}

/// Rig-space world matrix of every node of `pose`.
fn world_matrices(rig: &Rig, pose: &Pose) -> Vec<Mat4> {
    let mut world: Vec<Option<Mat4>> = vec![None; rig.len()];
    fn resolve(i: usize, rig: &Rig, pose: &Pose, world: &mut [Option<Mat4>], depth: usize) -> Mat4 {
        if let Some(m) = world[i] {
            return m;
        }
        let local = pose.trs[i].to_mat4();
        let m = match rig.parents[i] {
            Some(p) if depth < 256 && p != i => resolve(p, rig, pose, world, depth + 1) * local,
            _ => local,
        };
        world[i] = Some(m);
        m
    }
    let mut out: Vec<Mat4> = (0..rig.len()).map(|i| resolve(i, rig, pose, &mut world, 0)).collect();
    // The skeleton's container joints sit above the bones, but the frame stays the one without them
    // (`Skeleton::world_matrices`): take their rest basis back out.
    if !rig.containers.is_empty() {
        let basis = rig.containers.iter().fold(Mat4::IDENTITY, |acc, &c| acc * rig.rest[c].to_mat4());
        let inv = basis.inverse();
        if inv.is_finite() {
            out.iter_mut().for_each(|m| *m = inv * *m);
        }
    }
    out
}

/// Solve the IK goals on a pose: for each, the limb ending at the goal's tip
/// (its parent is the mid joint, that one's parent the root) is bent toward
/// the target with the analytic two-bone solver, blended by the goal weight.
/// Goals naming a joint without two ancestors are skipped.
pub fn apply_ik(rig: &Rig, pose: &mut Pose, goals: &[zestengine_core::IkGoal]) {
    for goal in goals.iter().filter(|g| g.weight > 0.0) {
        let Some(tip) = rig.index(&goal.tip) else {
            continue;
        };
        let Some(mid) = rig.parents.get(tip).copied().flatten() else {
            continue;
        };
        let Some(root) = rig.parents.get(mid).copied().flatten() else {
            continue;
        };
        let world = world_matrices(rig, pose);
        let pos = |i: usize| world[i].w_axis.truncate();
        let pole = goal.pole.unwrap_or_else(|| pos(mid) + (pos(mid) - (pos(root) + pos(tip)) * 0.5).normalize_or(Vec3::Z));
        let sol = zestengine_core::solve_two_bone(pos(root), pos(mid), pos(tip), goal.target, pole);
        let rot = |i: usize| Quat::from_mat4(&world[i]).normalize();
        let parent_rot = |i: usize| rig.parents[i].map(rot).unwrap_or(Quat::IDENTITY);
        let w = goal.weight.clamp(0.0, 1.0);
        let new_root = sol.root_rotation * rot(root);
        let new_mid = sol.mid_rotation * rot(mid);
        let local_root = (parent_rot(root).inverse() * new_root).normalize();
        let local_mid = (new_root.inverse() * new_mid).normalize();
        pose.trs[root].r = pose.trs[root].r.slerp(local_root, w);
        pose.trs[mid].r = pose.trs[mid].r.slerp(local_mid, w);
        pose.touched[root] |= CH_R;
        pose.touched[mid] |= CH_R;
    }
}

/// Keep the skeleton's root where the clips start (root motion moves the
/// object), blending the starts the way the poses blend. A component a clip
/// bakes into its pose stays as authored.
fn pin_root(rig: &Rig, pose: &mut Pose, machine: &StateMachine, lp: &LayerPlayback, params: &HashMap<String, f32>, clips: &ClipSet) {
    let Some(root) = rig.motion_root() else {
        return;
    };
    let dest = motion_root_start(rig, &machine.states[lp.state].motion, params, clips);
    let src = match (&lp.frozen, lp.previous) {
        (None, Some((p, _))) => motion_root_start(rig, &machine.states[p].motion, params, clips),
        _ => None,
    };
    let start = match (src, dest) {
        (Some(s), Some(d)) => Some(s.lerp(d, lp.transition_progress())),
        (s, d) => d.or(s),
    };
    if let Some(start) = start {
        let cur = pose.trs[root];
        let keep = |f: f32, own: f32, pinned: f32| pinned + (own - pinned) * f;
        pose.trs[root].t = Vec3::new(keep(start.keep_xz, cur.t.x, start.t.x), keep(start.keep_y, cur.t.y, start.t.y), keep(start.keep_xz, cur.t.z, start.t.z));
        if start.keep_rot < 1.0 {
            // Hold the yaw where the clips start; what the root tilts by stays.
            let yaw = zestengine_assets::anim_import::twist_angle_y(cur.r);
            let target = keep(start.keep_rot, yaw, start.yaw);
            pose.trs[root].r = (Quat::from_rotation_y(target - yaw) * cur.r).normalize();
        }
    }
}

/// Apply a layer's pose over the layers below it. Two muscle poses compose in
/// muscle space, honouring the mask's body parts.
pub fn compose_layer(rig: &Rig, base: Pose, layer: &Pose, weight: f32, blend: LayerBlend, mask: &AvatarMask) -> Pose {
    if base.muscle.is_some() || layer.muscle.is_some() {
        return crate::anim_muscle::compose(rig, base, layer, weight, blend, mask);
    }
    compose_layer_trs(rig, base, layer, weight, blend, mask)
}

pub(crate) fn compose_layer_trs(rig: &Rig, mut base: Pose, layer: &Pose, weight: f32, blend: LayerBlend, mask: &AvatarMask) -> Pose {
    let w = weight.clamp(0.0, 1.0);
    if w <= 0.0 {
        return base;
    }
    let mask = rig.mask(mask);
    compose_props(&mut base.props, &layer.props, w, |node| mask.as_ref().is_some_and(|m| !m.get(node as usize).copied().unwrap_or(true)));
    for i in 0..base.trs.len().min(layer.trs.len()) {
        if mask.as_ref().is_some_and(|m| !m[i]) {
            continue;
        }
        let ch = layer.touched[i];
        if ch == 0 {
            continue;
        }
        let (b, l, rest) = (base.trs[i], layer.trs[i], rig.rest[i]);
        match blend {
            LayerBlend::Override => {
                if ch & CH_T != 0 {
                    base.trs[i].t = b.t.lerp(l.t, w);
                }
                if ch & CH_R != 0 {
                    base.trs[i].r = b.r.slerp(l.r, w);
                }
                if ch & CH_S != 0 {
                    base.trs[i].s = b.s.lerp(l.s, w);
                }
                if ch & CH_F != 0 {
                    base.scalar[i] += (layer.scalar[i] - base.scalar[i]) * w;
                }
            }
            LayerBlend::Additive => {
                if ch & CH_T != 0 {
                    base.trs[i].t = b.t + (l.t - rest.t) * w;
                }
                if ch & CH_R != 0 {
                    let delta = rest.r.inverse() * l.r;
                    base.trs[i].r = (b.r * Quat::IDENTITY.slerp(delta, w)).normalize();
                }
                if ch & CH_S != 0 {
                    let ratio = l.s / rest.s.max(Vec3::splat(1e-6));
                    base.trs[i].s = b.s * Vec3::ONE.lerp(ratio, w);
                }
                if ch & CH_F != 0 {
                    base.scalar[i] += (layer.scalar[i] - rig.rest_scalar[i]) * w;
                }
            }
        }
        base.touched[i] |= ch;
    }
    base
}

/// Advance an Animator without a controller (direct `play` of clips, `.anim`
/// assets or its own blend trees), with cross-fades and events.
pub fn advance_simple(animator: &mut Animator, rig: &Rig, clips: &ClipSet, dt: f32) -> AnimOutput {
    advance_simple_ex(animator, rig, clips, dt, Eval::default())
}

fn advance_simple_ex(animator: &mut Animator, rig: &Rig, clips: &ClipSet, dt: f32, eval: Eval) -> AnimOutput {
    animator.state.fired.clear();
    if !animator.state.started {
        animator.state.started = true;
        if animator.play_on_start && !animator.clip.is_empty() {
            animator.state.current = Some(animator.clip.clone());
            animator.state.playing = true;
        }
    }
    let motion_of = |name: &str| -> Motion {
        match animator.blend_trees.iter().find(|t| t.name == name) {
            Some(tree) => Motion::BlendTree(tree.clone()),
            None => Motion::Clip(name.to_owned()),
        }
    };
    let Some(current) = animator.state.current.clone() else {
        return AnimOutput { pose: Pose::rest(rig), root_delta: Vec3::ZERO, root_yaw: 0.0 };
    };
    let motion = motion_of(&current);
    let looping = animator.looping;
    let params = animator.state.params.clone();
    // `state.time` is seconds into the motion; the clock underneath is its
    // normalized time, integrated with the length of the moment so a blend
    // tree whose length shifts with its weights never jumps phase.
    let last_len = norm_len(animator.state.duration);
    let n_prev = animator.state.time / last_len;
    let duration = motion_duration(&motion, &params, clips);
    let len = norm_len(duration);
    let mut n = n_prev;
    if animator.state.playing {
        let s = &mut animator.state;
        n += dt * animator.speed / len;
        s.fade_elapsed += dt;
        if let Some((_, t)) = s.previous.as_mut() {
            *t += dt * animator.speed;
        }
    }
    animator.state.duration = duration;
    if !looping && duration > 0.0 {
        n = n.min(1.0);
    }
    animator.state.time = n * len;
    let mut fired = Vec::new();
    let mut memory = std::mem::take(&mut animator.state.leaf_weights);
    motion_events(animator, &motion, &params, clips, n_prev, n, looping, n_prev <= 0.0, 1.0, &mut memory, &[], &mut fired);
    animator.state.leaf_weights = memory;
    animator.state.fired.extend(fired);
    if eval.skip_pose {
        return AnimOutput { pose: Pose::rest(rig), root_delta: Vec3::ZERO, root_yaw: 0.0 };
    }
    let mut pose = sample_motion_lazy(&motion, &params, clips, rig, n, looping, false);
    let s = &mut animator.state;
    let w = if s.fade_duration > 0.0 { (s.fade_elapsed / s.fade_duration).min(1.0) } else { 1.0 };
    if w >= 1.0 {
        s.previous = None;
    }
    if let Some((name, t)) = s.previous.clone() {
        let pm = motion_of(&name);
        let prev = sample_motion_lazy(&pm, &params, clips, rig, t / norm_len(motion_duration(&pm, &params, clips)), looping, false);
        pose = blend_pose(&prev, &pose, w);
    }
    pose.resolve_muscles();
    AnimOutput { pose, root_delta: Vec3::ZERO, root_yaw: 0.0 }
}

/// Whether an Animator needs the graph runtime (controller or `.anim` clips)
/// rather than the classic imported-clip path.
pub fn wants_graph(animator: &Animator) -> bool {
    !animator.controller.is_empty()
        || is_clip_asset(&animator.clip)
        || animator.state.current.as_deref().is_some_and(is_clip_asset)
}

// ------------------------------------------------------------------ assets

/// Loaded controllers and `.anim` clips by project-relative path, plus cached
/// skeleton rigs. Edited assets are pushed in by the editor so changes show
/// live while playing.
#[derive(Default)]
pub struct AnimAssets {
    root: Option<PathBuf>,
    controllers: HashMap<String, Option<Arc<AnimatorController>>>,
    pub clips: HashMap<String, Option<Arc<AnimClip>>>,
    skeleton_rigs: HashMap<usize, Arc<Rig>>,
    /// The real frame delta of the current frame (0 = not set: Animators on
    /// `UnscaledTime` then follow the scaled one). The play session sets it.
    pub unscaled_dt: f32,
    /// Material values the animation set this frame, waiting for
    /// [`crate::anim_props::flush_material_writes`].
    pub material_writes: Vec<crate::anim_props::MaterialWrite>,
}

impl AnimAssets {
    pub fn set_root(&mut self, root: &Path) {
        if self.root.as_deref() != Some(root) {
            self.root = Some(root.to_owned());
            self.controllers.retain(|_, c| c.is_some());
            self.clips.retain(|_, c| c.is_some());
        }
    }

    /// The clocks of this frame for scaled delta `dt`.
    pub fn timing(&self, dt: f32) -> FrameTiming {
        FrameTiming { dt, unscaled_dt: if self.unscaled_dt > 0.0 { self.unscaled_dt } else { dt }, fixed_dt: FIXED_STEP }
    }

    /// Whether two references (paths or `@<id>`s) name the same file.
    pub fn same_asset(&self, a: &str, b: &str) -> bool {
        if a == b {
            return true;
        }
        let Some(root) = self.root.as_deref() else { return false };
        let r = |s: &str| zestengine_core::asset_ref::resolve(root, s);
        r(a).is_some() && r(a) == r(b)
    }

    /// Cache keys that refer to `path` (it, and any `@<id>` naming it).
    fn keys_for<'k>(&self, path: &str, keys: impl Iterator<Item = &'k String>) -> Vec<String> {
        let mut out = vec![path.to_owned()];
        out.extend(keys.filter(|k| zestengine_core::asset_ref::is_ref(k) && self.same_asset(k, path)).cloned());
        out
    }

    /// Replace a controller with an in-memory version (editor live edit).
    pub fn set_controller(&mut self, path: &str, ctrl: AnimatorController) {
        let ctrl = ctrl.flattened();
        self.load_controller_clips(&ctrl);
        let arc = Some(Arc::new(ctrl));
        for k in self.keys_for(path, self.controllers.keys()) {
            self.controllers.insert(k, arc.clone());
        }
    }

    /// Replace a clip with an in-memory version (editor live edit).
    pub fn set_clip(&mut self, path: &str, clip: AnimClip) {
        let arc = Some(Arc::new(clip));
        for k in self.keys_for(path, self.clips.keys()) {
            self.clips.insert(k, arc.clone());
        }
    }

    /// Forget everything loaded from disk (re-read on next use).
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.controllers.clear();
        self.clips.clear();
    }

    /// A controller with its sub-state machines flattened (what the runtime plays).
    pub fn controller(&mut self, path: &str) -> Option<Arc<AnimatorController>> {
        if let Some(c) = self.controllers.get(path) {
            return c.clone();
        }
        let loaded = self.root.as_ref().and_then(|root| match zestengine_assets::load_anim_controller(zestengine_core::asset_ref::resolve_path(root, path)) {
            Ok(mut c) => {
                for p in c.resolve_mask_assets(root) {
                    tracing::warn!("animator controller {path}: {p}");
                }
                Some(c.flattened())
            }
            Err(e) => {
                tracing::warn!("animator controller {path}: {e:#}");
                None
            }
        });
        if let Some(c) = &loaded {
            self.load_controller_clips(c);
        }
        let arc = loaded.map(Arc::new);
        self.controllers.insert(path.to_owned(), arc.clone());
        arc
    }

    pub fn ensure_clip(&mut self, name: &str) {
        if !is_clip_asset(name) || self.clips.contains_key(name) {
            return;
        }
        let loaded = self.root.as_ref().and_then(|root| match crate::clip_source::load_any(root, name) {
            Ok(c) => Some(Arc::new(c)),
            Err(e) => {
                tracing::warn!("animation clip {name}: {e:#}");
                None
            }
        });
        self.clips.insert(name.to_owned(), loaded);
    }

    /// Load every clip of a blend tree, nested trees included.
    fn ensure_tree_clips(&mut self, tree: &zestengine_core::BlendTree) {
        for m in &tree.motions {
            match &m.tree {
                Some(t) => self.ensure_tree_clips(t),
                None => self.ensure_clip(&m.clip),
            }
        }
    }

    fn load_controller_clips(&mut self, ctrl: &AnimatorController) {
        for layer in &ctrl.layers {
            for s in &layer.machine.states {
                match &s.motion {
                    Motion::Clip(c) => self.ensure_clip(c),
                    Motion::BlendTree(t) => self.ensure_tree_clips(t),
                }
            }
        }
    }

    /// Load what an Animator references; returns its controller if it has one.
    pub fn prepare(&mut self, animator: &Animator) -> Option<Arc<AnimatorController>> {
        self.ensure_clip(&animator.clip);
        if let Some(c) = animator.state.current.as_deref() {
            self.ensure_clip(c);
        }
        for t in &animator.blend_trees {
            self.ensure_tree_clips(t);
        }
        if animator.controller.is_empty() { None } else { self.controller(&animator.controller) }
    }

    /// Drop the cached rig of skeleton `index` (its bones were replaced by a reimport).
    pub fn forget_skeleton_rig(&mut self, index: usize) {
        self.skeleton_rigs.remove(&index);
    }

    pub fn skeleton_rig(&mut self, index: usize, skel: &Skeleton) -> Arc<Rig> {
        let rig = self.skeleton_rigs.entry(index).or_insert_with(|| Arc::new(Rig::from_skeleton(skel)));
        if rig.len() != skel.joints.len() {
            *rig = Arc::new(Rig::from_skeleton(skel));
        }
        rig.clone()
    }
}

// ------------------------------------------------------- object hierarchies

/// Rig over an object and its descendants, named by path relative to `root`
/// ("" = root, "Arm/Hand"). Rest values are the objects' current local
/// transforms. Returns the rig and the scene index of each node.
pub fn object_rig(scene: &crate::scene::LoadedScene, root: usize) -> (Rig, Vec<usize>) {
    let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, o) in scene.objects.iter().enumerate() {
        if let Some(p) = o.parent {
            children.entry(p).or_default().push(i);
        }
    }
    let (mut names, mut parents, mut rest, mut scalar, mut nodes) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut stack: Vec<(usize, Option<usize>, String)> = vec![(root, None, String::new())];
    while let Some((i, parent, path)) = stack.pop() {
        if nodes.len() > 4096 {
            break;
        }
        let o = &scene.objects[i];
        let me = nodes.len();
        names.push(path.clone());
        parents.push(parent);
        rest.push(Trs { t: o.translation, r: o.rotation, s: o.scale });
        scalar.push(
            o.components
                .iter()
                .find_map(|c| c.as_any().downcast_ref::<zestengine_core::LightComponent>())
                .map(|l| l.intensity)
                .unwrap_or(0.0),
        );
        nodes.push(i);
        if let Some(kids) = children.get(&i) {
            for &k in kids.iter().rev() {
                let name = &scene.objects[k].name;
                let child_path = if path.is_empty() { name.clone() } else { format!("{path}/{name}") };
                stack.push((k, Some(me), child_path));
            }
        }
    }
    (Rig::new(names, parents, rest, scalar, false), nodes)
}

/// Write the driven channels of `pose` onto the rig's objects.
pub fn apply_pose_to_objects(scene: &mut crate::scene::LoadedScene, nodes: &[usize], pose: &Pose) {
    for (n, &i) in nodes.iter().enumerate() {
        let Some(&ch) = pose.touched.get(n) else {
            break;
        };
        if ch == 0 || i >= scene.objects.len() {
            continue;
        }
        let o = &mut scene.objects[i];
        let v = pose.trs[n];
        if ch & CH_T != 0 {
            o.translation = v.t;
        }
        if ch & CH_R != 0 {
            o.rotation = v.r;
        }
        if ch & CH_S != 0 {
            o.scale = v.s;
        }
        if ch & CH_F != 0
            && let Some(l) = o.components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<zestengine_core::LightComponent>())
        {
            l.intensity = pose.scalar[n];
        }
    }
}

/// The clocks of one frame, for [`advance_timed`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameTiming {
    /// The game's (time-scaled) delta.
    pub dt: f32,
    /// The real delta, ignoring the time scale.
    pub unscaled_dt: f32,
    /// The fixed (physics) step.
    pub fixed_dt: f32,
}

/// The physics step Animators in `Fixed` mode follow by default.
pub const FIXED_STEP: f32 = 1.0 / 60.0;

impl FrameTiming {
    /// A frame where scaled and real time agree.
    pub fn new(dt: f32) -> Self {
        Self { dt, unscaled_dt: dt, fixed_dt: FIXED_STEP }
    }
}

/// Advance an Animator through whichever path applies, with its update mode
/// (which clock) and culling mode (what an invisible object skips) honoured.
/// `advance` is this with scaled and real time equal and nothing culled.
pub fn advance_timed(
    animator: &mut Animator,
    ctrl: Option<&AnimatorController>,
    rig: &Rig,
    clips: &ClipSet,
    timing: FrameTiming,
) -> AnimOutput {
    let culling = animator.culling;
    let culled = animator.state.culled && culling != AnimatorCullingMode::AlwaysAnimate;
    if culled && culling == AnimatorCullingMode::CullCompletely {
        animator.state.fired.clear();
        animator.state.callbacks.clear();
        return AnimOutput { pose: held_pose(animator, rig), root_delta: Vec3::ZERO, root_yaw: 0.0 };
    }
    let (dt, hold) = match animator.update_mode {
        AnimatorUpdateMode::Normal => (timing.dt, false),
        AnimatorUpdateMode::UnscaledTime => (timing.unscaled_dt, false),
        AnimatorUpdateMode::Fixed => {
            // Whole fixed steps only; in between the pose holds.
            let step = timing.fixed_dt.max(1e-4);
            animator.state.fixed_accum += timing.dt.max(0.0);
            let steps = (animator.state.fixed_accum / step).floor();
            animator.state.fixed_accum -= steps * step;
            (steps * step, steps < 1.0)
        }
    };
    // Fixed mode between steps: nothing is evaluated (a parameter change waits
    // for the next step, like the clock), the last pose stays.
    if hold && animator.state.last_output.is_some() {
        animator.state.fired.clear();
        animator.state.callbacks.clear();
        return AnimOutput { pose: held_pose(animator, rig), root_delta: Vec3::ZERO, root_yaw: 0.0 };
    }
    let eval = Eval { skip_pose: culled, hold };
    let mut out = match ctrl {
        Some(c) => {
            let path = animator.controller.clone();
            advance_controller_ex(animator, c, &path, rig, clips, dt, eval)
        }
        None => advance_simple_ex(animator, rig, clips, dt, eval),
    };
    if culling != AnimatorCullingMode::AlwaysAnimate || animator.update_mode == AnimatorUpdateMode::Fixed {
        if culled {
            out.pose = held_pose(animator, rig);
        } else {
            animator.state.last_output = Some(freeze(&out.pose));
        }
    }
    out
}

/// The last pose written, or the rest pose before any.
fn held_pose(animator: &Animator, rig: &Rig) -> Pose {
    match &animator.state.last_output {
        Some(f) => thaw(rig, f),
        None => Pose::rest(rig),
    }
}

/// Advance an Animator through whichever path applies.
pub fn advance(
    animator: &mut Animator,
    ctrl: Option<&AnimatorController>,
    rig: &Rig,
    clips: &ClipSet,
    dt: f32,
) -> AnimOutput {
    advance_timed(animator, ctrl, rig, clips, FrameTiming::new(dt))
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_core::{BlendMotion, BlendTree};
    use zestengine_assets::{
        AnimCondition, AnimLayer, AnimParameter, AnimState, AnimTrack, AnimTransition, CondOp, InterruptionSource, KeyInterp,
        Keyframe,
    };

    /// Two nodes: "" (root) and "Arm" (child).
    fn rig() -> Rig {
        Rig::new(
            vec![String::new(), "Arm".into()],
            vec![None, Some(0)],
            vec![Trs::default(); 2],
            vec![0.0; 2],
            false,
        )
    }

    /// Moves `target` along X from 0 to `to` over one second.
    fn clip(target: &str, to: f32) -> AnimClip {
        AnimClip {
            length: 1.0,
            looping: true,
            tracks: vec![AnimTrack {
                target: target.into(),
                property: TrackProperty::Translation,
                name: String::new(),
                slot: 0,
                keys: vec![
                    Keyframe { time: 0.0, value: [0.0; 4], interp: KeyInterp::Linear, ..Default::default() },
                    Keyframe { time: 1.0, value: [to, 0.0, 0.0, 0.0], interp: KeyInterp::Linear, ..Default::default() },
                ],
            }],
            ..Default::default()
        }
    }

    /// Moves `target` along X from 0 to `to` over `len` seconds.
    fn clip_len(target: &str, to: f32, len: f32) -> AnimClip {
        AnimClip {
            length: len,
            looping: true,
            tracks: vec![AnimTrack {
                target: target.into(),
                property: TrackProperty::Translation,
                name: String::new(),
                slot: 0,
                keys: vec![
                    Keyframe { time: 0.0, value: [0.0; 4], interp: KeyInterp::Linear, ..Default::default() },
                    Keyframe { time: len, value: [to, 0.0, 0.0, 0.0], interp: KeyInterp::Linear, ..Default::default() },
                ],
            }],
            ..Default::default()
        }
    }

    fn assets(list: &[(&str, AnimClip)]) -> HashMap<String, Option<Arc<AnimClip>>> {
        list.iter().map(|(n, c)| (n.to_string(), Some(Arc::new(c.clone())))).collect()
    }

    fn state(name: &str, clip: &str) -> AnimState {
        AnimState { name: name.into(), motion: Motion::Clip(clip.into()), ..Default::default() }
    }

    fn controller(states: Vec<AnimState>, transitions: Vec<AnimTransition>, params: Vec<AnimParameter>) -> AnimatorController {
        AnimatorController {
            parameters: params,
            layers: vec![AnimLayer {
                machine: StateMachine { states, transitions, ..Default::default() },
                ..Default::default()
            }],
        }
    }

    fn animator() -> Animator {
        Animator { controller: "test.animctrl".into(), ..Default::default() }
    }

    fn step(a: &mut Animator, c: &AnimatorController, a_set: &HashMap<String, Option<Arc<AnimClip>>>, dt: f32) -> Pose {
        let set = ClipSet { imported: &[], indices: &[], assets: a_set };
        advance_controller(a, c, "test.animctrl", &rig(), &set, dt).pose
    }

    fn x(p: &Pose, node: usize) -> f32 {
        p.trs[node].t.x
    }

    #[test]
    fn transitions_on_condition() {
        let clips = assets(&[("idle.anim", clip("", 1.0)), ("run.anim", clip("", 10.0))]);
        let c = controller(
            vec![state("Idle", "idle.anim"), state("Run", "run.anim")],
            vec![AnimTransition {
                from: Some("Idle".into()),
                to: "Run".into(),
                conditions: vec![AnimCondition { parameter: "Speed".into(), op: CondOp::Greater, threshold: 0.5 }],
                duration: 0.0,
                ..Default::default()
            }],
            vec![AnimParameter { name: "Speed".into(), ..Default::default() }],
        );
        let mut a = animator();
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.current_state(0), Some("Idle"));
        a.set_float("Speed", 1.0);
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.current_state(0), Some("Run"));
        assert_eq!(a.state.state_changes.last().map(|s| s.2.as_str()), Some("Run"));
        let p = step(&mut a, &c, &clips, 0.5);
        assert!((x(&p, 0) - 5.0).abs() < 1e-3, "{}", x(&p, 0));
    }

    #[test]
    fn trigger_is_consumed() {
        let clips = assets(&[("a.anim", clip("", 1.0))]);
        let c = controller(
            vec![state("A", "a.anim"), state("B", "a.anim")],
            vec![
                AnimTransition {
                    from: Some("A".into()),
                    to: "B".into(),
                    conditions: vec![AnimCondition { parameter: "Go".into(), op: CondOp::If, threshold: 0.0 }],
                    duration: 0.0,
                    ..Default::default()
                },
                AnimTransition {
                    from: Some("B".into()),
                    to: "A".into(),
                    conditions: vec![AnimCondition { parameter: "Go".into(), op: CondOp::If, threshold: 0.0 }],
                    duration: 0.0,
                    ..Default::default()
                },
            ],
            vec![AnimParameter { name: "Go".into(), kind: AnimParamKind::Trigger, default: 0.0 }],
        );
        let mut a = animator();
        step(&mut a, &c, &clips, 0.1);
        a.set_trigger("Go");
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.current_state(0), Some("B"));
        assert!(!a.get_bool("Go"), "trigger consumed");
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.current_state(0), Some("B"), "no second transition without a new trigger");
    }

    #[test]
    fn exit_time_transition() {
        let clips = assets(&[("a.anim", clip("", 1.0))]);
        let c = controller(
            vec![AnimState { looping: false, ..state("Attack", "a.anim") }, state("Idle", "a.anim")],
            vec![AnimTransition {
                from: Some("Attack".into()),
                to: "Idle".into(),
                has_exit_time: true,
                exit_time: 0.75,
                duration: 0.0,
                ..Default::default()
            }],
            vec![],
        );
        let mut a = animator();
        step(&mut a, &c, &clips, 0.5);
        assert_eq!(a.current_state(0), Some("Attack"));
        step(&mut a, &c, &clips, 0.3);
        assert_eq!(a.current_state(0), Some("Idle"));
    }

    #[test]
    fn looping_exit_time_fires_at_the_crossing() {
        let clips = assets(&[("a.anim", clip("", 1.0))]);
        let c = controller(
            vec![state("Loop", "a.anim"), state("Next", "a.anim")],
            vec![AnimTransition {
                from: Some("Loop".into()),
                to: "Next".into(),
                has_exit_time: true,
                exit_time: 0.5,
                conditions: vec![AnimCondition { parameter: "Ok".into(), op: CondOp::If, threshold: 0.0 }],
                duration: 0.0,
                ..Default::default()
            }],
            vec![AnimParameter { name: "Ok".into(), kind: AnimParamKind::Bool, default: 0.0 }],
        );
        let mut a = animator();
        step(&mut a, &c, &clips, 0.6);
        a.set_bool("Ok", true);
        // Past the 0.5 crossing of this loop: waits for the next one.
        step(&mut a, &c, &clips, 0.2);
        assert_eq!(a.current_state(0), Some("Loop"));
        step(&mut a, &c, &clips, 0.8);
        assert_eq!(a.current_state(0), Some("Next"));
    }

    #[test]
    fn crossfade_weight_progresses() {
        let clips = assets(&[("zero.anim", clip("", 0.0)), ("ten.anim", AnimClip {
            tracks: vec![AnimTrack {
                target: String::new(),
                property: TrackProperty::Translation,
                name: String::new(),
                slot: 0,
                keys: vec![Keyframe { time: 0.0, value: [10.0, 0.0, 0.0, 0.0], interp: KeyInterp::Step, ..Default::default() }],
            }],
            ..Default::default()
        })]);
        let c = controller(
            vec![state("A", "zero.anim"), state("B", "ten.anim")],
            vec![AnimTransition {
                from: Some("A".into()),
                to: "B".into(),
                conditions: vec![AnimCondition { parameter: "Go".into(), op: CondOp::If, threshold: 0.0 }],
                duration: 1.0,
                ..Default::default()
            }],
            vec![AnimParameter { name: "Go".into(), kind: AnimParamKind::Bool, default: 0.0 }],
        );
        let mut a = animator();
        step(&mut a, &c, &clips, 0.0);
        a.set_bool("Go", true);
        let p = step(&mut a, &c, &clips, 0.0);
        assert!(a.is_in_transition(0));
        assert!(x(&p, 0).abs() < 1e-3);
        let p = step(&mut a, &c, &clips, 0.25);
        assert!((x(&p, 0) - 2.5).abs() < 1e-3, "{}", x(&p, 0));
        let p = step(&mut a, &c, &clips, 0.5);
        assert!((x(&p, 0) - 7.5).abs() < 1e-3, "{}", x(&p, 0));
        let p = step(&mut a, &c, &clips, 0.5);
        assert!(!a.is_in_transition(0));
        assert!((x(&p, 0) - 10.0).abs() < 1e-3);
    }

    fn child(clip: &str, at: f32) -> BlendMotion {
        BlendMotion { clip: clip.into(), position: [at, 0.0], ..Default::default() }
    }

    fn one_d_tree(children: Vec<BlendMotion>) -> BlendTree {
        BlendTree { name: "T".into(), parameter: "P".into(), motions: children, ..Default::default() }
    }

    fn tree_controller(tree: BlendTree) -> AnimatorController {
        controller(
            vec![AnimState { name: "T".into(), motion: Motion::BlendTree(tree), ..Default::default() }],
            vec![],
            vec![AnimParameter { name: "P".into(), ..Default::default() }],
        )
    }

    #[test]
    fn blend_tree_children_stay_in_phase_while_weights_move() {
        // A 2 s clip (x = 20 * phase) and a 1 s clip (x = 10 * phase). The old
        // raw state clock made the walk jump as the weights moved between them.
        let clips = assets(&[("idle.anim", clip_len("", 20.0, 2.0)), ("walk.anim", clip_len("", 10.0, 1.0))]);
        let c = tree_controller(one_d_tree(vec![child("idle.anim", 0.0), child("walk.anim", 1.0)]));
        let mut a = animator();
        step(&mut a, &c, &clips, 0.0);
        // Pure idle: the tree is 2 s long, so 0.5 s is a quarter turn.
        let p = step(&mut a, &c, &clips, 0.5);
        assert!((a.state_normalized_time(0) - 0.25).abs() < 1e-4);
        assert!((x(&p, 0) - 5.0).abs() < 1e-3, "{}", x(&p, 0));
        // Pure walk: 1 s long, 0.5 s is another half turn.
        a.set_float("P", 1.0);
        let p = step(&mut a, &c, &clips, 0.5);
        assert!((a.state_normalized_time(0) - 0.75).abs() < 1e-4);
        assert!((x(&p, 0) - 7.5).abs() < 1e-3, "{}", x(&p, 0));
        // Back to idle without a time step: the same phase, no pop.
        a.set_float("P", 0.0);
        let p = step(&mut a, &c, &clips, 0.0);
        assert!((x(&p, 0) - 15.0).abs() < 1e-3, "{}", x(&p, 0));
        // Halfway the length is 1.5 s: 0.3 s adds 0.2 of a turn.
        a.set_float("P", 0.5);
        step(&mut a, &c, &clips, 0.3);
        assert!((a.state_normalized_time(0) - 0.95).abs() < 1e-4, "{}", a.state_normalized_time(0));
    }

    #[test]
    fn blend_tree_length_is_the_weighted_child_length() {
        let clips = assets(&[("long.anim", clip_len("", 1.0, 3.0)), ("short.anim", clip_len("", 1.0, 1.0))]);
        let set = ClipSet { imported: &[], indices: &[], assets: &clips };
        let mut tree = one_d_tree(vec![child("long.anim", 0.0), child("short.anim", 1.0)]);
        let motion = |t: &BlendTree| Motion::BlendTree(t.clone());
        let at = |v: f32| HashMap::from([("P".to_string(), v)]);
        assert!((motion_duration(&motion(&tree), &at(0.25), &set) - 2.5).abs() < 1e-4);
        // A child's time scale divides its length (2x = half as long).
        tree.motions[0].time_scale = 2.0;
        assert!((motion_duration(&motion(&tree), &at(0.0), &set) - 1.5).abs() < 1e-4);
        // Direct trees summing below 1 divide by the sum, above 1 they do not.
        let mut direct = one_d_tree(vec![child("long.anim", 0.0), child("short.anim", 0.0)]);
        direct.kind = zestengine_core::BlendTreeKind::Direct;
        direct.motions[0].direct_parameter = "a".into();
        direct.motions[1].direct_parameter = "b".into();
        let ab = |a: f32, b: f32| HashMap::from([("a".to_string(), a), ("b".to_string(), b)]);
        assert!((motion_duration(&motion(&direct), &ab(0.25, 0.25), &set) - 2.0).abs() < 1e-4);
        assert!((motion_duration(&motion(&direct), &ab(1.0, 1.0), &set) - 4.0).abs() < 1e-4);
        assert_eq!(motion_duration(&motion(&direct), &ab(0.0, 0.0), &set), 1.0);
    }

    #[test]
    fn blend_tree_cycle_offset_and_nesting() {
        let clips = assets(&[("a.anim", clip_len("", 10.0, 1.0)), ("b.anim", clip_len("", 100.0, 1.0))]);
        let set = ClipSet { imported: &[], indices: &[], assets: &clips };
        let mut inner = one_d_tree(vec![child("a.anim", 0.0), child("b.anim", 1.0)]);
        inner.parameter = "Q".into();
        inner.motions[0].cycle_offset = 0.5;
        let mut outer = one_d_tree(vec![child("", 0.0), child("a.anim", 1.0)]);
        outer.motions[0].tree = Some(Box::new(inner));
        let motion = Motion::BlendTree(outer);
        let params = HashMap::from([("P".to_string(), 0.0), ("Q".to_string(), 0.0)]);
        // P=0 selects the nested tree, Q=0 its first clip, shifted half a cycle.
        let p = sample_motion(&motion, &params, &set, &rig(), 0.0, true, false);
        assert!((x(&p, 0) - 5.0).abs() < 1e-3, "{}", x(&p, 0));
        let params = HashMap::from([("P".to_string(), 0.0), ("Q".to_string(), 1.0)]);
        let p = sample_motion(&motion, &params, &set, &rig(), 0.25, true, false);
        assert!((x(&p, 0) - 25.0).abs() < 1e-3, "{}", x(&p, 0));
        let (leaves, total) = motion_leaves(&motion, &params, &set);
        assert_eq!(total, 3);
        assert_eq!(leaves.iter().map(|l| l.index).collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn child_dropping_to_zero_weight_gets_one_last_event_window() {
        let clips = assets(&[
            ("idle.anim", with_events(clip("", 1.0), &[])),
            ("walk.anim", with_events(clip("", 1.0), &[(0.5, "Mid"), (0.8, "Late")])),
        ]);
        let c = tree_controller(one_d_tree(vec![child("idle.anim", 0.0), child("walk.anim", 1.0)]));
        let mut a = animator();
        step(&mut a, &c, &clips, 0.0);
        a.set_float("P", 1.0);
        step(&mut a, &c, &clips, 0.3);
        // The weight leaves the walk this frame: its window (0.3, 0.7] still runs.
        a.set_float("P", 0.0);
        step(&mut a, &c, &clips, 0.4);
        assert_eq!(a.state.fired, ["Mid".to_string()]);
        // And only that one: the next frame's window (0.7, 0.9] is not played.
        step(&mut a, &c, &clips, 0.2);
        assert!(a.state.fired.is_empty(), "{:?}", a.state.fired);
    }

    #[test]
    fn any_state_has_priority() {
        let clips = assets(&[("a.anim", clip("", 1.0))]);
        let cond = |p: &str| vec![AnimCondition { parameter: p.into(), op: CondOp::If, threshold: 0.0 }];
        let c = controller(
            vec![state("Idle", "a.anim"), state("Walk", "a.anim"), state("Die", "a.anim")],
            vec![
                AnimTransition { from: Some("Idle".into()), to: "Walk".into(), conditions: cond("Move"), duration: 0.0, ..Default::default() },
                AnimTransition { from: None, to: "Die".into(), conditions: cond("Dead"), duration: 0.0, ..Default::default() },
            ],
            vec![
                AnimParameter { name: "Move".into(), kind: AnimParamKind::Bool, default: 0.0 },
                AnimParameter { name: "Dead".into(), kind: AnimParamKind::Bool, default: 0.0 },
            ],
        );
        let mut a = animator();
        step(&mut a, &c, &clips, 0.1);
        a.set_bool("Move", true);
        a.set_bool("Dead", true);
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.current_state(0), Some("Die"));
        // Any State does not re-enter the state it targets.
        let changes = a.state.state_changes.len();
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.state.state_changes.len(), changes);
    }

    #[test]
    fn layers_override_and_additive_with_mask() {
        let r = rig();
        let base = {
            let mut p = Pose::rest(&r);
            p.trs[0].t.x = 1.0;
            p.trs[1].t.x = 1.0;
            p.touched = vec![CH_T; 2];
            p
        };
        let layer = {
            let mut p = Pose::rest(&r);
            p.trs[0].t.x = 3.0;
            p.trs[1].t.x = 3.0;
            p.touched = vec![CH_T; 2];
            p
        };
        let o = compose_layer(&r, base.clone(), &layer, 0.5, LayerBlend::Override, &Default::default());
        assert!((x(&o, 0) - 2.0).abs() < 1e-5 && (x(&o, 1) - 2.0).abs() < 1e-5);
        let a = compose_layer(&r, base.clone(), &layer, 1.0, LayerBlend::Additive, &Default::default());
        assert!((x(&a, 0) - 4.0).abs() < 1e-5);
        let m = compose_layer(&r, base, &layer, 1.0, LayerBlend::Override, &zestengine_assets::AvatarMask::from_legacy(&["Arm".to_string()]));
        assert!((x(&m, 0) - 1.0).abs() < 1e-5, "root masked out");
        assert!((x(&m, 1) - 3.0).abs() < 1e-5);
    }

    #[test]
    fn controller_layers_compose() {
        let clips = assets(&[("root.anim", clip("", 2.0)), ("arm.anim", clip("Arm", 4.0))]);
        let mut c = controller(vec![state("Base", "root.anim")], vec![], vec![]);
        c.layers.push(AnimLayer {
            name: "Arm".into(),
            weight: 0.5,
            blend: LayerBlend::Override,
            mask: Default::default(),
            machine: StateMachine { states: vec![state("Wave", "arm.anim")], ..Default::default() },
            ..Default::default()
        });
        let mut a = animator();
        let p = step(&mut a, &c, &clips, 0.5);
        assert!((x(&p, 0) - 1.0).abs() < 1e-3);
        assert!((x(&p, 1) - 1.0).abs() < 1e-3, "{}", x(&p, 1));
        assert_eq!(p.touched[1] & CH_T, CH_T);
    }

    #[test]
    fn play_state_forces_and_fires_state_events() {
        let clips = assets(&[("a.anim", clip("", 1.0))]);
        let c = controller(
            vec![state("A", "a.anim"), AnimState { on_enter: "Entered".into(), ..state("B", "a.anim") }],
            vec![],
            vec![],
        );
        let mut a = animator();
        step(&mut a, &c, &clips, 0.1);
        a.play_state("B", 0.0);
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.current_state(0), Some("B"));
        assert_eq!(a.state.fired, vec!["Entered".to_string()]);
    }

    #[test]
    fn simple_path_plays_anim_assets() {
        let clips = assets(&[("wave.anim", clip("Arm", 2.0))]);
        let set = ClipSet { imported: &[], indices: &[], assets: &clips };
        let mut a = Animator { clip: "wave.anim".into(), ..Default::default() };
        let p = advance_simple(&mut a, &rig(), &set, 0.25).pose;
        assert!((x(&p, 1) - 0.5).abs() < 1e-4);
        assert_eq!(p.touched[0], 0, "root untouched");
    }

    #[test]
    fn sub_state_machine_plays_nested_states() {
        let clips = assets(&[("idle.anim", clip("", 1.0)), ("walk.anim", clip("", 4.0)), ("run.anim", clip("", 10.0))]);
        let cond = |p: &str, op, threshold| AnimCondition { parameter: p.into(), op, threshold };
        let go = |from: Option<&str>, to: &str, c: AnimCondition| AnimTransition {
            from: from.map(Into::into),
            to: to.into(),
            conditions: vec![c],
            duration: 0.0,
            ..Default::default()
        };
        let locomotion = StateMachine {
            states: vec![state("Walk", "walk.anim"), state("Run", "run.anim")],
            default_state: "Walk".into(),
            transitions: vec![go(Some("Walk"), "Run", cond("Speed", CondOp::Greater, 2.0))],
            ..Default::default()
        };
        let c = controller(
            vec![state("Idle", "idle.anim"), AnimState { name: "Locomotion".into(), sub_machine: Some(Box::new(locomotion)), ..Default::default() }],
            vec![
                go(Some("Idle"), "Locomotion", cond("Speed", CondOp::Greater, 0.5)),
                go(Some("Locomotion"), "Idle", cond("Speed", CondOp::Less, 0.5)),
            ],
            vec![AnimParameter { name: "Speed".into(), ..Default::default() }],
        );
        let mut a = animator();
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.current_state(0), Some("Idle"));
        // Entering the sub-machine enters its default state.
        a.set_float("Speed", 1.0);
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.current_state(0), Some("Locomotion/Walk"));
        let p = step(&mut a, &c, &clips, 0.25);
        assert!((x(&p, 0) - 1.0).abs() < 1e-3, "walk clip plays: {}", x(&p, 0));
        // An inner transition.
        a.set_float("Speed", 3.0);
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.current_state(0), Some("Locomotion/Run"));
        // Leaving from an inner state through the sub-machine's own transition.
        a.set_float("Speed", 0.0);
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.current_state(0), Some("Idle"));
        assert_eq!(a.state.state_changes.last().map(|s| s.1.as_str()), Some("Locomotion/Run"));
        // Scripts force nested states by path.
        a.set_float("Speed", 1.0);
        a.play_state("Locomotion/Run", 0.0);
        step(&mut a, &c, &clips, 0.0);
        assert_eq!(a.current_state(0), Some("Locomotion/Run"));
    }

    // ---------------------------------------------------- ANIM-1 parity tests

    /// Holds `target` at X = `x` for the whole clip.
    fn still(x: f32) -> AnimClip {
        AnimClip {
            tracks: vec![AnimTrack {
                target: String::new(),
                property: TrackProperty::Translation,
                name: String::new(),
                slot: 0,
                keys: vec![Keyframe { time: 0.0, value: [x, 0.0, 0.0, 0.0], interp: KeyInterp::Step, ..Default::default() }],
            }],
            ..Default::default()
        }
    }

    fn with_events(mut c: AnimClip, events: &[(f32, &str)]) -> AnimClip {
        c.events = events.iter().map(|(time, name)| zestengine_assets::ClipEvent { time: *time, name: (*name).into() }).collect();
        c
    }

    fn when(param: &str) -> Vec<AnimCondition> {
        vec![AnimCondition { parameter: param.into(), op: CondOp::If, threshold: 0.0 }]
    }

    fn flag(name: &str) -> AnimParameter {
        AnimParameter { name: name.into(), kind: AnimParamKind::Bool, default: 0.0 }
    }

    fn trigger(name: &str) -> AnimParameter {
        AnimParameter { name: name.into(), kind: AnimParamKind::Trigger, default: 0.0 }
    }

    #[test]
    fn exit_crossing_fires_once_per_loop_or_once_ever() {
        // e <= 1: once per loop, on the crossing frame, with the overshoot.
        let o = exit_crossing(0.4, 0.6, 0.5).unwrap();
        assert!((o - 0.1).abs() < 1e-5);
        assert!(exit_crossing(0.6, 0.8, 0.5).is_none(), "already past it");
        assert!(exit_crossing(0.4, 0.45, 0.5).is_none(), "not there yet");
        let o = exit_crossing(1.4, 1.6, 0.5).unwrap();
        assert!((o - 0.1).abs() < 1e-5, "second loop");
        let o = exit_crossing(0.9, 1.7, 0.5).unwrap();
        assert!((o - 0.2).abs() < 1e-5, "boundary crossed in the frame: the new loop's crossing");
        // A state held at its end never crosses again.
        assert!(exit_crossing(1.0, 1.0, 1.0).is_none());
        assert!(exit_crossing(0.9, 1.0, 1.0).is_some());
        // e > 1: once, after the loops.
        assert!(exit_crossing(1.4, 1.6, 1.5).is_some());
        assert!(exit_crossing(0.4, 0.6, 1.5).is_none());
        assert!(exit_crossing(1.6, 1.8, 1.5).is_none(), "once ever");
        // Reverse mirrors it.
        let o = exit_crossing(0.6, 0.4, 0.5).unwrap();
        assert!((o - 0.1).abs() < 1e-5);
        assert!(exit_crossing(0.4, 0.2, 0.5).is_none());
    }

    #[test]
    fn exit_time_does_not_fire_after_the_crossing_frame() {
        // Attack (not looping) -> Idle needs exit time 0.5 AND Go. Go comes
        // late: the crossing is gone, so it must not fire (it used to, on
        // every frame past the exit time).
        let clips = assets(&[("a.anim", clip("", 1.0))]);
        let c = controller(
            vec![AnimState { looping: false, ..state("Attack", "a.anim") }, state("Idle", "a.anim")],
            vec![AnimTransition {
                from: Some("Attack".into()),
                to: "Idle".into(),
                has_exit_time: true,
                exit_time: 0.5,
                conditions: when("Go"),
                duration: 0.0,
                ..Default::default()
            }],
            vec![flag("Go")],
        );
        let mut a = animator();
        step(&mut a, &c, &clips, 0.1);
        step(&mut a, &c, &clips, 0.6);
        step(&mut a, &c, &clips, 0.6);
        a.set_bool("Go", true);
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.current_state(0), Some("Attack"));
    }

    #[test]
    fn exit_time_overshoot_carries_into_the_next_state() {
        let clips = assets(&[("zero.anim", clip("", 0.0)), ("ten.anim", clip("", 10.0))]);
        let c = controller(
            vec![state("A", "zero.anim"), state("B", "ten.anim")],
            vec![AnimTransition {
                from: Some("A".into()),
                to: "B".into(),
                has_exit_time: true,
                exit_time: 0.5,
                duration: 0.0,
                ..Default::default()
            }],
            vec![],
        );
        let mut a = animator();
        // Crosses 0.5 at 0.1 s into the 0.6 s step: B has played 0.1 s already.
        let p = step(&mut a, &c, &clips, 0.6);
        assert_eq!(a.current_state(0), Some("B"));
        assert!((x(&p, 0) - 1.0).abs() < 1e-3, "{}", x(&p, 0));
        // And into a fade: it starts 0.1 s in.
        let mut c2 = c.clone();
        c2.layers[0].machine.transitions[0].duration = 1.0;
        let mut a = animator();
        step(&mut a, &c2, &clips, 0.6);
        assert!((a.state.layers[0].fade_elapsed - 0.1).abs() < 1e-4, "{}", a.state.layers[0].fade_elapsed);
    }

    #[test]
    fn interruption_freezes_the_blended_pose() {
        let clips = assets(&[("a.anim", still(0.0)), ("b.anim", still(10.0)), ("c.anim", still(20.0))]);
        let c = controller(
            vec![state("A", "a.anim"), state("B", "b.anim"), state("C", "c.anim")],
            vec![
                AnimTransition {
                    from: Some("A".into()),
                    to: "B".into(),
                    conditions: when("Go"),
                    duration: 1.0,
                    interruption: InterruptionSource::Destination,
                    ..Default::default()
                },
                AnimTransition { from: Some("B".into()), to: "C".into(), conditions: when("Stop"), duration: 1.0, ..Default::default() },
            ],
            vec![flag("Go"), flag("Stop")],
        );
        let mut a = animator();
        step(&mut a, &c, &clips, 0.0);
        a.set_bool("Go", true);
        step(&mut a, &c, &clips, 0.0);
        let p = step(&mut a, &c, &clips, 0.5);
        assert!((x(&p, 0) - 5.0).abs() < 1e-3);
        // B -> C cuts in halfway: the pose stays at 5 instead of popping to B's 10.
        a.set_bool("Go", false);
        a.set_bool("Stop", true);
        let p = step(&mut a, &c, &clips, 0.0);
        assert_eq!(a.current_state(0), Some("C"));
        assert!((x(&p, 0) - 5.0).abs() < 1e-3, "popped to {}", x(&p, 0));
        assert!(a.is_in_transition(0));
        // Halfway through the second fade: halfway from the snapshot to C.
        let p = step(&mut a, &c, &clips, 0.5);
        assert!((x(&p, 0) - 12.5).abs() < 1e-3, "{}", x(&p, 0));
        let p = step(&mut a, &c, &clips, 0.6);
        assert!((x(&p, 0) - 20.0).abs() < 1e-3);
        assert!(!a.is_in_transition(0));
    }

    /// A -> B (fading for 10 s) with `interruption`, plus the given transitions.
    fn fading_setup(first: AnimTransition, more: Vec<AnimTransition>) -> (AnimatorController, HashMap<String, Option<Arc<AnimClip>>>) {
        let clips = assets(&[("a.anim", clip("", 1.0)), ("b.anim", clip("", 1.0))]);
        let mut ts = vec![AnimTransition { from: Some("A".into()), to: "B".into(), duration: 10.0, ..first }];
        ts.extend(more);
        let c = controller(
            vec![state("A", "a.anim"), state("B", "b.anim"), state("C", "a.anim"), state("D", "a.anim")],
            ts,
            vec![flag("Go"), flag("ToC"), flag("ToD"), flag("Dead")],
        );
        (c, clips)
    }

    #[test]
    fn interruption_none_blocks_even_any_state() {
        let (c, clips) = fading_setup(
            AnimTransition { conditions: when("Go"), interruption: InterruptionSource::None, ..Default::default() },
            vec![AnimTransition { from: None, to: "D".into(), conditions: when("Dead"), duration: 0.0, ..Default::default() }],
        );
        let mut a = animator();
        step(&mut a, &c, &clips, 0.1);
        a.set_bool("Go", true);
        step(&mut a, &c, &clips, 0.1);
        assert!(a.is_in_transition(0));
        a.set_bool("Dead", true);
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.current_state(0), Some("B"));
        // The legacy flag still means Destination.
        let (c, clips) = fading_setup(
            AnimTransition { conditions: when("Go"), interruptible: true, ..Default::default() },
            vec![AnimTransition { from: None, to: "D".into(), conditions: when("Dead"), duration: 0.0, ..Default::default() }],
        );
        let mut a = animator();
        step(&mut a, &c, &clips, 0.1);
        a.set_bool("Go", true);
        step(&mut a, &c, &clips, 0.1);
        a.set_bool("Dead", true);
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.current_state(0), Some("D"));
    }

    #[test]
    fn interruption_source_checks_the_source_list_with_its_own_time() {
        // A -> B fades with Source interruption. A -> D has an exit time at
        // 0.9 of A; B -> C is ready the whole time but is B's list.
        let (c, clips) = fading_setup(
            AnimTransition { conditions: when("Go"), interruption: InterruptionSource::Source, ..Default::default() },
            vec![
                AnimTransition { from: Some("B".into()), to: "C".into(), conditions: when("ToC"), duration: 0.0, ..Default::default() },
                AnimTransition { from: Some("A".into()), to: "D".into(), has_exit_time: true, exit_time: 0.9, duration: 0.0, ..Default::default() },
            ],
        );
        let mut a = animator();
        step(&mut a, &c, &clips, 0.5);
        a.set_bool("Go", true);
        a.set_bool("ToC", true);
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.current_state(0), Some("B"), "the destination's list is not checked");
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.current_state(0), Some("B"));
        // A's clock reaches 0.9 (it was 0.6 when the fade began).
        step(&mut a, &c, &clips, 0.3);
        assert_eq!(a.current_state(0), Some("D"), "source exit time interrupts");
    }

    #[test]
    fn interruption_destination_uses_the_destinations_clock() {
        // B -> C needs B's exit time 0.5. B starts at 0, so it fires at 0.5 s
        // into the fade, whatever A's clock says.
        let (c, clips) = fading_setup(
            AnimTransition { conditions: when("Go"), interruption: InterruptionSource::Destination, ..Default::default() },
            vec![AnimTransition { from: Some("B".into()), to: "C".into(), has_exit_time: true, exit_time: 0.5, duration: 0.0, ..Default::default() }],
        );
        let mut a = animator();
        step(&mut a, &c, &clips, 0.1);
        a.set_bool("Go", true);
        step(&mut a, &c, &clips, 0.1);
        step(&mut a, &c, &clips, 0.3);
        assert_eq!(a.current_state(0), Some("B"));
        step(&mut a, &c, &clips, 0.2);
        assert_eq!(a.current_state(0), Some("C"));
    }

    #[test]
    fn ordered_interruption_skips_what_follows_the_active_transition() {
        let both = |ordered: bool| {
            fading_setup(
                AnimTransition { conditions: when("Go"), interruption: InterruptionSource::Source, ordered_interruption: ordered, ..Default::default() },
                vec![AnimTransition { from: Some("A".into()), to: "C".into(), conditions: when("ToC"), duration: 0.0, ..Default::default() }],
            )
        };
        for (ordered, expect) in [(false, "C"), (true, "B")] {
            let (c, clips) = both(ordered);
            let mut a = animator();
            step(&mut a, &c, &clips, 0.1);
            a.set_bool("Go", true);
            step(&mut a, &c, &clips, 0.1);
            a.set_bool("ToC", true);
            step(&mut a, &c, &clips, 0.1);
            assert_eq!(a.current_state(0), Some(expect), "ordered = {ordered}");
        }
    }

    #[test]
    fn triggers_are_cleared_after_every_layer_has_run() {
        let clips = assets(&[("a.anim", clip("", 1.0))]);
        let go = |from: &str, to: &str| AnimTransition {
            from: Some(from.into()),
            to: to.into(),
            conditions: when("Go"),
            duration: 0.0,
            ..Default::default()
        };
        let mut c = controller(vec![state("A", "a.anim"), state("B", "a.anim")], vec![go("A", "B")], vec![trigger("Go")]);
        c.layers.push(AnimLayer {
            name: "Upper".into(),
            machine: StateMachine {
                states: vec![state("X", "a.anim"), state("Y", "a.anim")],
                transitions: vec![go("X", "Y")],
                ..Default::default()
            },
            ..Default::default()
        });
        let mut a = animator();
        step(&mut a, &c, &clips, 0.1);
        a.set_trigger("Go");
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.current_state(0), Some("B"));
        assert_eq!(a.current_state(1), Some("Y"), "the higher layer saw the trigger too");
        assert!(!a.get_bool("Go"));
    }

    #[test]
    fn unconsumed_triggers_stay_set() {
        let clips = assets(&[("a.anim", clip("", 1.0))]);
        let c = controller(
            vec![state("A", "a.anim"), state("B", "a.anim")],
            vec![AnimTransition { from: Some("A".into()), to: "B".into(), conditions: when("Go"), has_exit_time: true, exit_time: 5.0, duration: 0.0, ..Default::default() }],
            vec![trigger("Go")],
        );
        let mut a = animator();
        step(&mut a, &c, &clips, 0.1);
        a.set_trigger("Go");
        step(&mut a, &c, &clips, 0.1);
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.current_state(0), Some("A"));
        assert!(a.get_bool("Go"), "nothing took it");
    }

    #[test]
    fn crossed_events_windows() {
        let a = Animator::default();
        let clip = with_events(clip("", 1.0), &[(0.0, "start"), (0.5, "mid")]);
        let cr = ClipRef::Asset(&clip);
        let run = |from: f32, to: f32, looping: bool, first: bool| {
            let mut out = Vec::new();
            crossed_events(&a, "", cr, from, to, looping, first, &mut out);
            out
        };
        assert_eq!(run(0.0, 0.1, true, true), ["start"], "events at the start fire on the first window");
        assert!(run(0.0, 0.1, true, false).is_empty(), "(prev, cur] excludes the start");
        assert_eq!(run(0.0, 2.6, true, true), ["start", "mid", "start", "mid", "start", "mid"].map(String::from), "every loop crossed fires");
        assert_eq!(run(0.4, 0.5, true, false), ["mid"]);
        assert!(run(0.5, 0.6, true, false).is_empty());
        // Reverse uses [cur, prev).
        assert_eq!(run(0.6, 0.5, true, false), ["mid"]);
        assert!(run(0.5, 0.4, true, false).is_empty());
        assert_eq!(run(0.2, 1.8, false, false), ["mid"], "forward, once");
        assert_eq!(run(1.0, 0.0, false, false), ["mid", "start"].map(String::from), "reverse fires back to the start");
        assert_eq!(run(0.0, 0.0, true, true), ["start"], "a paused first frame still fires the start");
    }

    #[test]
    fn events_fire_from_the_fading_out_state() {
        // A's event at 0.5 passes while A fades out; the fade ends on the
        // frame it passes (A's last window still fires).
        let clips = assets(&[("a.anim", with_events(clip("", 1.0), &[(0.5, "FromA")])), ("b.anim", with_events(clip("", 1.0), &[(0.3, "FromB")]))]);
        let c = controller(
            vec![state("A", "a.anim"), state("B", "b.anim")],
            vec![AnimTransition { from: Some("A".into()), to: "B".into(), conditions: when("Go"), duration: 0.4, ..Default::default() }],
            vec![flag("Go")],
        );
        let mut a = animator();
        step(&mut a, &c, &clips, 0.2);
        a.set_bool("Go", true);
        step(&mut a, &c, &clips, 0.0);
        assert!(a.is_in_transition(0));
        step(&mut a, &c, &clips, 0.4);
        assert!(!a.is_in_transition(0), "fade over");
        let mut fired = a.state.fired.clone();
        fired.sort();
        assert_eq!(fired, ["FromA", "FromB"].map(String::from));
        // Mid-fade, both fire as they pass.
        let mut c2 = c.clone();
        c2.layers[0].machine.transitions[0].duration = 2.0;
        let mut a = animator();
        step(&mut a, &c2, &clips, 0.2);
        a.set_bool("Go", true);
        step(&mut a, &c2, &clips, 0.0);
        step(&mut a, &c2, &clips, 0.4);
        let mut fired = a.state.fired.clone();
        fired.sort();
        assert_eq!(fired, ["FromA", "FromB"].map(String::from));
    }

    #[test]
    fn events_fire_from_every_weighted_blend_tree_clip() {
        let clips = assets(&[
            ("idle.anim", with_events(clip("", 1.0), &[(0.5, "Idle")])),
            ("walk.anim", with_events(clip("", 1.0), &[(0.5, "Walk")])),
            ("far.anim", with_events(clip("", 1.0), &[(0.5, "Far")])),
        ]);
        let tree = BlendTree {
            name: "T".into(),
            parameter: "P".into(),
            parameter_y: String::new(),
            kind: Default::default(),
            normalize_direct: false,
            motions: vec![
                BlendMotion { clip: "idle.anim".into(), position: [0.0, 0.0], ..Default::default() },
                BlendMotion { clip: "walk.anim".into(), position: [1.0, 0.0], ..Default::default() },
                BlendMotion { clip: "far.anim".into(), position: [5.0, 0.0], ..Default::default() },
            ],
        };
        let c = controller(
            vec![AnimState { name: "T".into(), motion: Motion::BlendTree(tree), ..Default::default() }],
            vec![],
            vec![AnimParameter { name: "P".into(), ..Default::default() }],
        );
        let mut a = animator();
        step(&mut a, &c, &clips, 0.0);
        a.set_float("P", 0.5);
        step(&mut a, &c, &clips, 0.6);
        let mut fired = a.state.fired.clone();
        fired.sort();
        assert_eq!(fired, ["Idle", "Walk"].map(String::from), "weighted clips fire, the weight-0 one does not");
    }

    #[test]
    fn fade_clock_follows_animator_speed() {
        let clips = assets(&[("a.anim", still(0.0)), ("b.anim", still(10.0))]);
        let c = controller(
            vec![state("A", "a.anim"), state("B", "b.anim")],
            vec![AnimTransition { from: Some("A".into()), to: "B".into(), conditions: when("Go"), duration: 1.0, ..Default::default() }],
            vec![flag("Go")],
        );
        let mut a = animator();
        a.speed = 2.0;
        step(&mut a, &c, &clips, 0.0);
        a.set_bool("Go", true);
        step(&mut a, &c, &clips, 0.0);
        let p = step(&mut a, &c, &clips, 0.25);
        assert!((x(&p, 0) - 5.0).abs() < 1e-3, "twice the speed, twice the fade: {}", x(&p, 0));
        // Negative speed still moves the fade forward.
        a.speed = -2.0;
        let p = step(&mut a, &c, &clips, 0.125);
        assert!((x(&p, 0) - 7.5).abs() < 1e-3, "{}", x(&p, 0));
    }

    #[test]
    fn on_exit_fires_when_the_transition_completes() {
        let clips = assets(&[("a.anim", clip("", 1.0))]);
        let c = controller(
            vec![AnimState { on_exit: "LeftA".into(), ..state("A", "a.anim") }, AnimState { on_enter: "EnteredB".into(), ..state("B", "a.anim") }],
            vec![AnimTransition { from: Some("A".into()), to: "B".into(), conditions: when("Go"), duration: 0.5, ..Default::default() }],
            vec![flag("Go")],
        );
        let mut a = animator();
        step(&mut a, &c, &clips, 0.1);
        a.set_bool("Go", true);
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.state.fired, vec!["EnteredB".to_string()], "A is still fading out");
        step(&mut a, &c, &clips, 0.3);
        assert!(a.state.fired.is_empty());
        step(&mut a, &c, &clips, 0.2);
        assert_eq!(a.state.fired, vec!["LeftA".to_string()]);
        // A cut exits at once.
        let mut cut = c.clone();
        cut.layers[0].machine.transitions[0].duration = 0.0;
        let mut a = animator();
        step(&mut a, &cut, &clips, 0.1);
        a.set_bool("Go", true);
        step(&mut a, &cut, &clips, 0.1);
        assert_eq!(a.state.fired, vec!["LeftA".to_string(), "EnteredB".to_string()]);
    }

    /// One joint at the root, for imported-clip root motion.
    fn skel_rig() -> Rig {
        Rig::new(vec!["root".into()], vec![None], vec![Trs::default()], vec![0.0], true)
    }

    /// Imported one-second clip moving the root at `vel` per second.
    fn moving(name: &str, vel: Vec3) -> AnimationClip {
        use zestengine_assets::AnimChannel;
        AnimationClip {
            name: name.into(),
            duration: 1.0,
            channels: vec![AnimChannel {
                joint: 0,
                path: ChannelPath::Translation,
                times: vec![0.0, 1.0],
                vec_values: vec![Vec3::ZERO, vel],
                quat_values: vec![], extra: None
            }],
            morph_tracks: Vec::new(), playback: None,
        }
    }

    fn step_imported(a: &mut Animator, c: &AnimatorController, imported: &[AnimationClip], dt: f32) -> AnimOutput {
        let indices: Vec<usize> = (0..imported.len()).collect();
        let no_assets = HashMap::new();
        let set = ClipSet { imported, indices: &indices, assets: &no_assets };
        advance_controller(a, c, "test.animctrl", &skel_rig(), &set, dt)
    }

    fn assert_vec(v: Vec3, e: Vec3) {
        assert!((v - e).length() < 1e-3, "{v:?} != {e:?}");
    }

    #[test]
    fn root_motion_keeps_every_axis() {
        let imported = [moving("up", Vec3::new(0.0, 2.0, 1.0))];
        let c = controller(vec![state("S", "up")], vec![], vec![]);
        let mut a = Animator { root_motion: true, ..animator() };
        step_imported(&mut a, &c, &imported, 0.0);
        let out = step_imported(&mut a, &c, &imported, 0.25);
        assert_vec(out.root_delta, Vec3::new(0.0, 0.5, 0.25));
        assert_vec(out.pose.trs[0].t, Vec3::ZERO);
    }

    /// `clip` with playback settings (a listed clip of the model's clip list).
    fn listed(mut clip: AnimationClip, f: impl FnOnce(&mut zestengine_assets::ClipPlayback)) -> AnimationClip {
        let mut pb = zestengine_assets::ClipPlayback { loop_time: true, keep_orientation: true, keep_y: true, keep_xz: true, ..Default::default() };
        f(&mut pb);
        clip.playback = Some(Box::new(pb));
        clip
    }

    #[test]
    fn a_baked_height_stays_in_the_pose_and_is_not_root_motion() {
        let imported = [listed(moving("up", Vec3::new(2.0, 2.0, 1.0)), |p| p.bake_y = true)];
        let c = controller(vec![state("S", "up")], vec![], vec![]);
        let mut a = Animator { root_motion: true, ..animator() };
        step_imported(&mut a, &c, &imported, 0.0);
        let out = step_imported(&mut a, &c, &imported, 0.25);
        assert_vec(out.root_delta, Vec3::new(0.5, 0.0, 0.25));
        assert_vec(out.pose.trs[0].t, Vec3::new(0.0, 0.5, 0.0));
    }

    #[test]
    fn baked_ground_travel_stays_in_the_pose_and_height_is_motion() {
        let imported = [listed(moving("up", Vec3::new(2.0, 2.0, 1.0)), |p| p.bake_xz = true)];
        let c = controller(vec![state("S", "up")], vec![], vec![]);
        let mut a = Animator { root_motion: true, ..animator() };
        step_imported(&mut a, &c, &imported, 0.0);
        let out = step_imported(&mut a, &c, &imported, 0.25);
        assert_vec(out.root_delta, Vec3::new(0.0, 0.5, 0.0));
        assert_vec(out.pose.trs[0].t, Vec3::new(0.5, 0.0, 0.25));
    }

    /// One-second clip turning the root about +Y by `turn` radians.
    fn turning(name: &str, turn: f32) -> AnimationClip {
        use zestengine_assets::AnimChannel;
        AnimationClip {
            name: name.into(),
            duration: 1.0,
            channels: vec![AnimChannel {
                joint: 0,
                path: ChannelPath::Rotation,
                times: vec![0.0, 1.0],
                vec_values: vec![],
                quat_values: vec![Quat::IDENTITY, Quat::from_rotation_y(turn)], extra: None
            }],
            morph_tracks: Vec::new(),
            playback: None,
        }
    }

    #[test]
    fn an_unbaked_turn_drives_the_object_and_a_baked_one_stays_in_the_pose() {
        let c = controller(vec![state("S", "turn")], vec![], vec![]);
        let yaw = |q: Quat| zestengine_assets::anim_import::twist_angle_y(q);
        // Settings that do not bake the rotation: the root turns the object, the pose holds its start yaw.
        let imported = [listed(turning("turn", 1.2), |_| {})];
        let mut a = Animator { root_motion: true, ..animator() };
        step_imported(&mut a, &c, &imported, 0.0);
        let out = step_imported(&mut a, &c, &imported, 0.5);
        assert!((out.root_yaw - 0.6).abs() < 1e-2, "yaw {}", out.root_yaw);
        assert!(yaw(out.pose.trs[0].r).abs() < 1e-2);
        // Baked: the pose turns, nothing else does.
        let imported = [listed(turning("turn", 1.2), |p| p.bake_rotation = true)];
        let mut a = Animator { root_motion: true, ..animator() };
        step_imported(&mut a, &c, &imported, 0.0);
        let out = step_imported(&mut a, &c, &imported, 0.5);
        assert_eq!(out.root_yaw, 0.0);
        assert!((yaw(out.pose.trs[0].r) - 0.6).abs() < 1e-2);
        // A clip with no settings keeps its turn in the pose, as before.
        let imported = [turning("turn", 1.2)];
        let mut a = Animator { root_motion: true, ..animator() };
        step_imported(&mut a, &c, &imported, 0.0);
        let out = step_imported(&mut a, &c, &imported, 0.5);
        assert_eq!(out.root_yaw, 0.0);
        assert!((yaw(out.pose.trs[0].r) - 0.6).abs() < 1e-2);
    }

    #[test]
    fn based_upon_offsets_the_baked_components() {
        let rig = skel_rig();
        let mut pb = zestengine_assets::ClipPlayback { bake_y: true, bake_xz: true, bake_rotation: true, height_offset: 0.1, feet_y: 0.3, center_xz: [0.5, 0.0], body_yaw: 0.4, rotation_offset: 90.0, ..Default::default() };
        let mut pose = Pose::rest(&rig);
        pose.trs[0] = Trs { t: Vec3::new(1.0, 1.0, 0.0), r: Quat::from_rotation_y(0.4), s: Vec3::ONE };
        pose.touched[0] = CH_T | CH_R;
        // Based upon the body orientation / feet / centre of mass.
        adjust_root(&rig, &pb, &mut pose);
        assert_vec(pose.trs[0].t, Vec3::new(0.5, 1.0 + 0.1 - 0.3, 0.0));
        assert!((zestengine_assets::anim_import::twist_angle_y(pose.trs[0].r) - 90f32.to_radians()).abs() < 1e-3, "faces forward, then the offset");
        // Original keeps the authored values (offsets still apply).
        pb.keep_orientation = true;
        pb.keep_y = true;
        pb.keep_xz = true;
        let mut pose = Pose::rest(&rig);
        pose.trs[0] = Trs { t: Vec3::new(1.0, 1.0, 0.0), r: Quat::from_rotation_y(0.4), s: Vec3::ONE };
        pose.touched[0] = CH_T | CH_R;
        adjust_root(&rig, &pb, &mut pose);
        assert_vec(pose.trs[0].t, Vec3::new(1.0, 1.1, 0.0));
        assert!((zestengine_assets::anim_import::twist_angle_y(pose.trs[0].r) - (0.4 + 90f32.to_radians())).abs() < 1e-3);
    }

    #[test]
    fn loop_time_wraps_or_holds_and_the_cycle_offset_starts_the_pass_later() {
        let rig = skel_rig();
        // Root x at normalized time `n` of a one-clip motion.
        let x_at = |clip: &AnimationClip, n: f32| {
            let imported = vec![clip.clone()];
            let indices = vec![0usize];
            let no_assets = HashMap::new();
            let set = ClipSet { imported: &imported, indices: &indices, assets: &no_assets };
            sample_motion(&Motion::Clip("run".into()), &HashMap::new(), &set, &rig, n, true, false).trs[0].t.x
        };
        let base = moving("run", Vec3::X);
        // Wrapping with a half-cycle offset: normalized 0.75 is clip time 1.25 = 0.25.
        let wrap = listed(base.clone(), |p| p.cycle_offset = 0.5);
        assert!((x_at(&wrap, 0.75) - 0.25).abs() < 1e-3, "{}", x_at(&wrap, 0.75));
        assert!((x_at(&wrap, 0.0) - 0.5).abs() < 1e-3, "the pass starts half way in");
        // Without Loop Time the clip holds its last pose past the end.
        let hold = listed(base, |p| {
            p.cycle_offset = 0.5;
            p.loop_time = false;
        });
        assert!((x_at(&hold, 0.75) - 1.0).abs() < 1e-2, "{}", x_at(&hold, 0.75));
    }

    #[test]
    fn clip_events_fire_once_per_loop_pass_with_their_parameters() {
        use zestengine_assets::ClipEventDef;
        let clip = listed(moving("run", Vec3::X), |p| {
            p.events = vec![
                ClipEventDef { time: 0.5, name: "Step".into(), float: 0.75, int: 3, string: "left".into(), object: "fx/dust".into() },
                ClipEventDef { time: 0.25, name: "Plain".into(), ..Default::default() },
            ];
        });
        let imported = [clip];
        let c = controller(vec![state("S", "run")], vec![], vec![]);
        let mut a = animator();
        step_imported(&mut a, &c, &imported, 0.0);
        let mut fired: Vec<String> = Vec::new();
        // 2.4 s in 0.1 s steps: Step (0.5) fires in passes 0 and 1, Plain (0.25) also in pass 2.
        for _ in 0..24 {
            step_imported(&mut a, &c, &imported, 0.1);
            fired.append(&mut a.state.fired);
        }
        let decoded: Vec<(String, Option<zestengine_core::AnimEventArgs>)> =
            fired.iter().map(|e| { let (n, a) = zestengine_core::AnimEventArgs::decode(e); (n.to_owned(), a) }).collect();
        let steps: Vec<_> = decoded.iter().filter(|(n, _)| n == "Step").collect();
        assert_eq!(steps.len(), 2, "once per pass: {fired:?}");
        let args = steps[0].1.as_ref().expect("the parameters travel with the event");
        assert_eq!((args.float, args.int, args.string.as_str(), args.object.as_str()), (0.75, 3, "left", "fx/dust"));
        assert_eq!(decoded.iter().filter(|(n, a)| n == "Plain" && a.is_none()).count(), 3, "a parameterless event stays a plain name");
    }

    #[test]
    fn root_motion_is_weight_normalised_across_clips() {
        let imported = [moving("fast", Vec3::new(4.0, 0.0, 0.0)), moving("still", Vec3::ZERO)];
        let tree = BlendTree {
            name: "T".into(),
            parameter: "P".into(),
            parameter_y: String::new(),
            kind: Default::default(),
            normalize_direct: false,
            motions: vec![
                BlendMotion { clip: "still".into(), position: [0.0, 0.0], ..Default::default() },
                BlendMotion { clip: "fast".into(), position: [1.0, 0.0], ..Default::default() },
            ],
        };
        let c = controller(
            vec![AnimState { name: "T".into(), motion: Motion::BlendTree(tree), ..Default::default() }],
            vec![],
            vec![AnimParameter { name: "P".into(), ..Default::default() }],
        );
        let mut a = Animator { root_motion: true, ..animator() };
        step_imported(&mut a, &c, &imported, 0.0);
        a.set_float("P", 0.5);
        let out = step_imported(&mut a, &c, &imported, 0.25);
        assert_vec(out.root_delta, Vec3::new(0.5, 0.0, 0.0));
    }

    #[test]
    fn root_motion_blends_across_a_fade() {
        let imported = [moving("east", Vec3::new(2.0, 0.0, 0.0)), moving("north", Vec3::new(0.0, 0.0, 2.0))];
        let c = controller(
            vec![state("E", "east"), state("N", "north")],
            vec![AnimTransition { from: Some("E".into()), to: "N".into(), conditions: when("Go"), duration: 1.0, ..Default::default() }],
            vec![flag("Go")],
        );
        let mut a = Animator { root_motion: true, ..animator() };
        step_imported(&mut a, &c, &imported, 0.0);
        a.set_bool("Go", true);
        step_imported(&mut a, &c, &imported, 0.0);
        // Halfway through the fade over a 0.25 s step the progress is 0.25: the
        // deltas mix 0.75 east / 0.25 north.
        let out = step_imported(&mut a, &c, &imported, 0.25);
        assert_vec(out.root_delta, Vec3::new(0.5 * 0.75, 0.0, 0.5 * 0.25));
    }

    #[test]
    fn additive_layers_add_no_root_motion_override_layers_blend_it() {
        let imported = [moving("walk", Vec3::new(2.0, 0.0, 0.0)), moving("slide", Vec3::new(0.0, 0.0, 4.0))];
        let layered = |blend, weight| {
            let mut c = controller(vec![state("Base", "walk")], vec![], vec![]);
            c.layers.push(AnimLayer {
                name: "Top".into(),
                weight,
                blend,
                mask: Default::default(),
                machine: StateMachine { states: vec![state("Slide", "slide")], ..Default::default() },
                ..Default::default()
            });
            c
        };
        let mut a = Animator { root_motion: true, ..animator() };
        let c = layered(LayerBlend::Additive, 1.0);
        step_imported(&mut a, &c, &imported, 0.0);
        let out = step_imported(&mut a, &c, &imported, 0.25);
        assert_vec(out.root_delta, Vec3::new(0.5, 0.0, 0.0));
        let mut a = Animator { root_motion: true, ..animator() };
        let c = layered(LayerBlend::Override, 0.5);
        step_imported(&mut a, &c, &imported, 0.0);
        let out = step_imported(&mut a, &c, &imported, 0.25);
        assert_vec(out.root_delta, Vec3::new(0.25, 0.0, 0.5));
    }

    #[test]
    fn float_equals_keeps_its_epsilon() {
        // Unity makes Float Equals / Not Equal always false; we keep an epsilon
        // compare on purpose (decision D2).
        assert!(CondOp::Equals.test(1.0, 1.0 + 1e-6));
        assert!(!CondOp::Equals.test(1.0, 1.1));
        assert!(CondOp::NotEquals.test(1.0, 1.1));
    }

    // ---------------------------------------------------------------- ANIM-4

    fn two_layers(base: StateMachine, top: AnimLayer) -> AnimatorController {
        AnimatorController {
            parameters: vec![AnimParameter { name: "Go".into(), kind: AnimParamKind::Bool, default: 0.0 }],
            layers: vec![AnimLayer { machine: base, ..Default::default() }, top],
        }
    }

    #[test]
    fn two_d_locomotion_tree_stays_in_phase() {
        let clips = assets(&[
            ("idle.anim", clip_len("", 20.0, 2.0)),
            ("fwd.anim", clip_len("", 10.0, 1.0)),
            ("back.anim", clip_len("", 15.0, 1.5)),
        ]);
        let child2 = |c: &str, x: f32, y: f32| BlendMotion { clip: c.into(), position: [x, y], ..Default::default() };
        let tree = BlendTree {
            name: "Loco".into(),
            kind: zestengine_core::BlendTreeKind::FreeformCartesian2D,
            parameter: "X".into(),
            parameter_y: "Y".into(),
            motions: vec![child2("idle.anim", 0.0, 0.0), child2("fwd.anim", 0.0, 1.0), child2("back.anim", 0.0, -1.0)],
            ..Default::default()
        };
        let c = controller(
            vec![AnimState { name: "T".into(), motion: Motion::BlendTree(tree), ..Default::default() }],
            vec![],
            vec![],
        );
        let mut a = animator();
        step(&mut a, &c, &clips, 0.0);
        // Idle/forward at y = 0.5 weigh 0.5 each: x = 0.5 * 20n + 0.5 * 10n = 15n whatever the history.
        for (y, dt) in [(0.0, 0.3), (0.5, 0.2), (1.0, 0.4), (0.5, 0.1), (-0.5, 0.25), (0.5, 0.3)] {
            a.set_float("Y", y);
            step(&mut a, &c, &clips, dt);
        }
        a.set_float("Y", 0.5);
        let p = step(&mut a, &c, &clips, 0.0);
        let n = a.state_normalized_time(0);
        assert!((x(&p, 0) - 15.0 * n.fract()).abs() < 1e-2, "x {} vs phase {}", x(&p, 0), n);
    }

    #[test]
    fn sync_layer_plays_its_own_motion_on_the_source_clock() {
        let clips = assets(&[("a.anim", clip_len("", 10.0, 1.0)), ("b.anim", clip_len("", 100.0, 2.0))]);
        let mut top = AnimLayer { name: "Sync".into(), sync_with: Some(0), ..Default::default() };
        top.motion_overrides = vec![zestengine_assets::MotionOverride { state: "A".into(), motion: Motion::Clip("b.anim".into()) }];
        let base = StateMachine { states: vec![state("A", "a.anim")], ..Default::default() };
        let c = two_layers(base, top).flattened();
        let mut a = animator();
        step(&mut a, &c, &clips, 0.0);
        // The sync layer replaces the base layer's pose (override, weight 1)
        // with its clip at the shared normalized time: 100 * 0.5.
        let p = step(&mut a, &c, &clips, 0.5);
        assert!((x(&p, 0) - 50.0).abs() < 1e-2, "{}", x(&p, 0));
        assert_eq!(a.current_state(1), Some("A"));
    }

    #[test]
    fn sync_timing_averages_lengths_by_layer_weight() {
        let clips = assets(&[("a.anim", clip_len("", 10.0, 1.0)), ("b.anim", clip_len("", 100.0, 3.0))]);
        let build = |timing: bool| {
            let mut top = AnimLayer { name: "Sync".into(), sync_with: Some(0), sync_timing: timing, weight: 0.5, ..Default::default() };
            top.motion_overrides = vec![zestengine_assets::MotionOverride { state: "A".into(), motion: Motion::Clip("b.anim".into()) }];
            let base = StateMachine { states: vec![state("A", "a.anim")], ..Default::default() };
            two_layers(base, top).flattened()
        };
        // Without timing the base clip's 1 s rules; with it (1 + 0.5 * 3) / 1.5 = 1.666 s.
        let mut a = animator();
        step(&mut a, &build(false), &clips, 0.0);
        step(&mut a, &build(false), &clips, 0.5);
        assert!((a.state_normalized_time(0) - 0.5).abs() < 1e-4);
        let mut a = animator();
        step(&mut a, &build(true), &clips, 0.0);
        step(&mut a, &build(true), &clips, 0.5);
        assert!((a.state_normalized_time(0) - 0.5 / (5.0 / 3.0)).abs() < 1e-4, "{}", a.state_normalized_time(0));
    }

    #[test]
    fn sync_layers_resolve_in_flattened_and_ignore_bad_sources() {
        let top = AnimLayer { sync_with: Some(5), machine: StateMachine { states: vec![state("Own", "a.anim")], ..Default::default() }, ..Default::default() };
        let c = two_layers(StateMachine { states: vec![state("A", "a.anim")], ..Default::default() }, top).flattened();
        // A source that does not exist leaves the layer's own machine alone.
        assert_eq!(c.layers[1].machine.states[0].name, "Own");
    }

    fn write_default_controller(wd: bool) -> AnimatorController {
        let mut s2 = state("S2", "root.anim");
        s2.write_defaults = wd;
        controller(
            vec![state("S1", "arm.anim"), s2],
            vec![AnimTransition { from: Some("S1".into()), to: "S2".into(), conditions: when("Go"), duration: 0.0, ..Default::default() }],
            vec![flag("Go")],
        )
    }

    #[test]
    fn write_defaults_off_holds_what_the_state_does_not_animate() {
        let clips = assets(&[("arm.anim", clip("Arm", 10.0)), ("root.anim", clip("", 3.0))]);
        for (wd, held) in [(true, false), (false, true)] {
            let c = write_default_controller(wd);
            let mut a = animator();
            step(&mut a, &c, &clips, 0.0);
            let p = step(&mut a, &c, &clips, 0.5);
            assert!((x(&p, 1) - 5.0).abs() < 1e-3);
            a.set_bool("Go", true);
            let p = step(&mut a, &c, &clips, 0.1);
            assert_eq!(a.current_state(0), Some("S2"));
            assert_eq!(p.touched[1] & CH_T != 0, held, "wd {wd}");
            if held {
                assert!((x(&p, 1) - 5.0).abs() < 1e-3, "{}", x(&p, 1));
            }
        }
    }

    #[test]
    fn loop_pose_closes_the_cycle() {
        let clips = assets(&[("walk.anim", clip("", 10.0))]);
        let run = |loop_pose: bool, dt: f32| {
            let mut s = state("W", "walk.anim");
            s.loop_pose = loop_pose;
            let c = controller(vec![s], vec![], vec![]);
            let mut a = animator();
            step(&mut a, &c, &clips, 0.0);
            x(&step(&mut a, &c, &clips, dt), 0)
        };
        // The clip ends 10 away from where it starts; Loop Pose cancels the drift.
        assert!((run(false, 0.5) - 5.0).abs() < 1e-3);
        assert!(run(true, 0.5).abs() < 1e-3);
        assert!(run(true, 0.999).abs() < 0.02);
    }

    use std::sync::Mutex;
    static CALLBACK_LOG: Mutex<Vec<String>> = Mutex::new(Vec::new());

    struct Recorder;

    impl zestengine_core::StateBehaviour for Recorder {
        fn on_state_enter(&mut self, ctx: &mut StateBehaviourCtx) {
            CALLBACK_LOG.lock().unwrap().push(format!("enter {}", ctx.state));
            ctx.params.insert("Entered".into(), 1.0);
        }
        fn on_state_update(&mut self, ctx: &mut StateBehaviourCtx) {
            CALLBACK_LOG.lock().unwrap().push(format!("update {}", ctx.state));
        }
        fn on_state_exit(&mut self, ctx: &mut StateBehaviourCtx) {
            CALLBACK_LOG.lock().unwrap().push(format!("exit {}", ctx.state));
        }
        fn on_state_move(&mut self, ctx: &mut StateBehaviourCtx) {
            CALLBACK_LOG.lock().unwrap().push(format!("move {}", ctx.state));
            *ctx.root_delta *= 2.0;
        }
        fn on_state_ik(&mut self, ctx: &mut StateBehaviourCtx) {
            CALLBACK_LOG.lock().unwrap().push(format!("ik {}", ctx.state));
            ctx.ik_goals.push(zestengine_core::IkGoal { tip: "Arm".into(), target: Vec3::X, pole: None, weight: 1.0 });
        }
    }

    #[test]
    fn state_behaviours_get_enter_update_exit_move_and_ik() {
        zestengine_core::register_state_behaviour("test_recorder", || Box::new(Recorder));
        let clips = assets(&[("a.anim", clip("", 1.0))]);
        let mut a_state = state("A", "a.anim");
        a_state.behaviours = vec!["test_recorder".into(), "not_registered".into()];
        a_state.on_exit = "LeftA".into();
        let mut b_state = state("B", "a.anim");
        b_state.behaviours = vec!["test_recorder".into()];
        let mut c = controller(
            vec![a_state, b_state],
            vec![AnimTransition { from: Some("A".into()), to: "B".into(), conditions: when("Go"), duration: 0.0, ..Default::default() }],
            vec![flag("Go")],
        );
        c.layers[0].ik_pass = true;
        let mut a = animator();
        a.root_motion = true;
        CALLBACK_LOG.lock().unwrap().clear();
        step(&mut a, &c, &clips, 0.0);
        let take = || std::mem::take(&mut *CALLBACK_LOG.lock().unwrap());
        // First frame: Enter, no Update; then Move and IK passes.
        assert_eq!(take(), ["enter A", "move A", "ik A"]);
        assert_eq!(a.get_float("Entered"), 1.0, "behaviours write parameters");
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(take(), ["update A", "move A", "ik A"]);
        assert_eq!(a.state.ik_goals.len(), 2, "the goal each IK pass sets");
        a.set_bool("Go", true);
        step(&mut a, &c, &clips, 0.1);
        // A cut completes the transition: Exit A, Enter B, and the string events still fire.
        assert_eq!(take(), ["update A", "exit A", "enter B", "move B", "ik B"]);
        assert!(a.state.fired.contains(&"LeftA".to_string()));
        assert!(a.state.callbacks.iter().any(|c| c.state == "B" && c.kind == StateCallbackKind::Enter));
    }

    #[test]
    fn time_parameter_picks_the_pose_and_ignores_speed() {
        let clips = assets(&[("run.anim", clip("", 10.0))]);
        let mut st = state("Slider", "run.anim");
        st.time_parameter = "T".into();
        st.speed = 5.0;
        st.looping = false;
        let c = controller(vec![st], vec![], vec![AnimParameter { name: "T".into(), ..Default::default() }]);
        let mut a = animator();
        a.set_float("T", 0.25);
        let p = step(&mut a, &c, &clips, 0.1);
        assert!((x(&p, 0) - 2.5).abs() < 1e-3, "{}", x(&p, 0));
        // Time passing changes nothing; the parameter does.
        let p = step(&mut a, &c, &clips, 0.3);
        assert!((x(&p, 0) - 2.5).abs() < 1e-3, "{}", x(&p, 0));
        a.set_float("T", 0.8);
        let p = step(&mut a, &c, &clips, 0.0);
        assert!((x(&p, 0) - 8.0).abs() < 1e-3, "{}", x(&p, 0));
        assert!((a.state.layers[0].normalized_time - 0.8).abs() < 1e-6);
        // A non-looping state holds its ends.
        a.set_float("T", 3.0);
        step(&mut a, &c, &clips, 0.0);
        assert_eq!(a.state.layers[0].normalized_time, 1.0);
        a.set_float("T", -2.0);
        step(&mut a, &c, &clips, 0.0);
        assert_eq!(a.state.layers[0].normalized_time, 0.0);
    }

    #[test]
    fn time_parameter_drives_exit_time_transitions_and_fading_source() {
        let clips = assets(&[("a.anim", clip("", 10.0))]);
        let mut a_state = state("A", "a.anim");
        a_state.time_parameter = "T".into();
        let c = controller(
            vec![a_state, state("B", "a.anim")],
            vec![AnimTransition { from: Some("A".into()), to: "B".into(), has_exit_time: true, exit_time: 0.5, duration: 0.0, ..Default::default() }],
            vec![AnimParameter { name: "T".into(), ..Default::default() }],
        );
        let mut an = animator();
        an.set_float("T", 0.2);
        step(&mut an, &c, &clips, 0.1);
        assert_eq!(an.current_state(0), Some("A"));
        an.set_float("T", 0.6);
        step(&mut an, &c, &clips, 0.1);
        assert_eq!(an.current_state(0), Some("B"), "the slider crossing the exit time takes the transition");
    }

    #[test]
    fn parameter_driver_state_behaviour_runs_from_the_controller() {
        use zestengine_core::{AvatarParamKind, DriverEntry, DriverOp, PARAMETER_DRIVER, ParamDriverSettings};
        let clips = assets(&[("a.anim", clip("", 1.0))]);
        let mut b_state = state("B", "a.anim");
        b_state.behaviours = vec![PARAMETER_DRIVER.into()];
        b_state.behaviour_settings.insert(
            PARAMETER_DRIVER.into(),
            ParamDriverSettings {
                entries: vec![
                    DriverEntry { name: "Count".into(), kind: AvatarParamKind::Int, op: DriverOp::Add { value: 1.0 } },
                    DriverEntry { name: "Mirror".into(), kind: AvatarParamKind::Float, op: DriverOp::Copy { source: "Level".into(), remap: None } },
                ],
                local_only: true,
            }
            .to_ron(),
        );
        let c = controller(
            vec![state("A", "a.anim"), b_state],
            vec![
                AnimTransition { from: Some("A".into()), to: "B".into(), conditions: when("Go"), duration: 0.0, ..Default::default() },
                AnimTransition { from: Some("B".into()), to: "A".into(), conditions: when("Back"), duration: 0.0, ..Default::default() },
            ],
            vec![flag("Go"), flag("Back")],
        );
        let mut a = animator();
        a.set_float("Level", 0.5);
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.get_float("Count"), 0.0);
        a.state.params_touched = false;
        a.set_bool("Go", true);
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.get_float("Count"), 1.0, "runs once on enter");
        assert_eq!(a.get_float("Mirror"), 0.5);
        assert!(a.state.params_touched);
        step(&mut a, &c, &clips, 0.1);
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.get_float("Count"), 1.0, "not on every frame");
        // Back and in again: a second enter, one more.
        a.set_bool("Go", false);
        a.set_bool("Back", true);
        step(&mut a, &c, &clips, 0.1);
        a.set_bool("Back", false);
        a.set_bool("Go", true);
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.get_float("Count"), 2.0);
        // A remote copy (IsLocal = 0) skips a local-only driver.
        a.set_bool("Go", false);
        a.set_bool("Back", true);
        step(&mut a, &c, &clips, 0.1);
        a.set_bool("Back", false);
        a.set_float("IsLocal", 0.0);
        a.set_bool("Go", true);
        step(&mut a, &c, &clips, 0.1);
        assert_eq!(a.get_float("Count"), 2.0);
    }

    #[test]
    fn ik_pass_bends_a_limb_onto_the_goal() {
        let r = Rig::new(
            vec![String::new(), "Mid".into(), "Tip".into()],
            vec![None, Some(0), Some(1)],
            vec![
                Trs::default(),
                Trs { t: Vec3::Y, ..Default::default() },
                Trs { t: Vec3::Y, ..Default::default() },
            ],
            vec![0.0; 3],
            false,
        );
        let tip_at = |p: &Pose| world_matrices(&r, p)[2].w_axis.truncate();
        let rest = Pose::rest(&r);
        assert!((tip_at(&rest) - Vec3::new(0.0, 2.0, 0.0)).length() < 1e-5);
        let goal = |w: f32| zestengine_core::IkGoal { tip: "Tip".into(), target: Vec3::new(1.0, 1.0, 0.0), pole: Some(Vec3::new(0.0, 1.0, 1.0)), weight: w };
        let mut p = Pose::rest(&r);
        apply_ik(&r, &mut p, &[goal(1.0)]);
        assert!((tip_at(&p) - Vec3::new(1.0, 1.0, 0.0)).length() < 1e-3, "{:?}", tip_at(&p));
        assert!(p.touched[0] & CH_R != 0 && p.touched[1] & CH_R != 0);
        // Weight 0 changes nothing; a joint without two ancestors is skipped.
        let mut p = Pose::rest(&r);
        apply_ik(&r, &mut p, &[goal(0.0)]);
        assert_eq!(p.touched, vec![0; 3]);
        apply_ik(&r, &mut p, &[zestengine_core::IkGoal { tip: "Mid".into(), target: Vec3::X, pole: None, weight: 1.0 }]);
        assert_eq!(p.touched, vec![0; 3]);
    }

    #[test]
    fn update_modes_pick_the_clock() {
        let clips = assets(&[("a.anim", clip("", 1.0))]);
        let c = controller(vec![state("A", "a.anim")], vec![], vec![]);
        let set = ClipSet { imported: &[], indices: &[], assets: &clips };
        let timing = FrameTiming { dt: 0.1, unscaled_dt: 0.5, fixed_dt: 0.25 };
        let run = |mode: AnimatorUpdateMode, frames: usize| {
            let mut a = Animator { update_mode: mode, ..animator() };
            advance_timed(&mut a, Some(&c), &rig(), &set, FrameTiming::new(0.0));
            for _ in 0..frames {
                advance_timed(&mut a, Some(&c), &rig(), &set, timing);
            }
            a.state_normalized_time(0)
        };
        assert!((run(AnimatorUpdateMode::Normal, 1) - 0.1).abs() < 1e-5);
        assert!((run(AnimatorUpdateMode::UnscaledTime, 1) - 0.5).abs() < 1e-5);
        // Fixed: 0.1 s per frame against a 0.25 s step holds for two frames, then steps once.
        assert!(run(AnimatorUpdateMode::Fixed, 2).abs() < 1e-6);
        assert!((run(AnimatorUpdateMode::Fixed, 3) - 0.25).abs() < 1e-5);
    }

    #[test]
    fn culling_modes_stop_what_they_say() {
        let clips = assets(&[("a.anim", clip("", 10.0))]);
        let c = controller(vec![state("A", "a.anim")], vec![], vec![]);
        let set = ClipSet { imported: &[], indices: &[], assets: &clips };
        let t = FrameTiming::new(0.25);
        for mode in [AnimatorCullingMode::AlwaysAnimate, AnimatorCullingMode::CullUpdateTransforms, AnimatorCullingMode::CullCompletely] {
            let mut a = Animator { culling: mode, ..animator() };
            advance_timed(&mut a, Some(&c), &rig(), &set, FrameTiming::new(0.0));
            let seen = advance_timed(&mut a, Some(&c), &rig(), &set, t).pose;
            assert!((x(&seen, 0) - 2.5).abs() < 1e-3);
            a.state.culled = true;
            let held = advance_timed(&mut a, Some(&c), &rig(), &set, t).pose;
            match mode {
                // Not honoured by this mode: keeps animating.
                AnimatorCullingMode::AlwaysAnimate => assert!((x(&held, 0) - 5.0).abs() < 1e-3),
                // The machine runs on, the pose stays.
                AnimatorCullingMode::CullUpdateTransforms => {
                    assert!((x(&held, 0) - 2.5).abs() < 1e-3);
                    assert!((a.state_normalized_time(0) - 0.5).abs() < 1e-5);
                }
                // Nothing moves.
                AnimatorCullingMode::CullCompletely => {
                    assert!((x(&held, 0) - 2.5).abs() < 1e-3);
                    assert!((a.state_normalized_time(0) - 0.25).abs() < 1e-5);
                }
            }
            // Visible again: it picks up where the clock is.
            a.state.culled = false;
            let back = advance_timed(&mut a, Some(&c), &rig(), &set, t).pose;
            assert!(x(&back, 0) > 2.5);
        }
    }

    // ------------------------------------------------- ANIM-4 known gaps

    fn still_ctrl_two_layers(sync_overrides: Vec<(&str, &str)>) -> AnimatorController {
        let base = StateMachine {
            states: vec![state("A", "a.anim"), state("B", "b.anim"), state("C", "c.anim")],
            transitions: vec![
                AnimTransition {
                    from: Some("A".into()),
                    to: "B".into(),
                    conditions: when("Go"),
                    duration: 1.0,
                    interruption: InterruptionSource::Destination,
                    ..Default::default()
                },
                AnimTransition { from: Some("B".into()), to: "C".into(), conditions: when("Stop"), duration: 1.0, ..Default::default() },
            ],
            ..Default::default()
        };
        let top = AnimLayer {
            name: "Sync".into(),
            sync_with: Some(0),
            motion_overrides: sync_overrides
                .into_iter()
                .map(|(s, m)| zestengine_assets::MotionOverride { state: s.into(), motion: Motion::Clip(m.into()) })
                .collect(),
            ..Default::default()
        };
        AnimatorController {
            parameters: vec![flag("Go"), flag("Stop")],
            layers: vec![AnimLayer { machine: base, ..Default::default() }, top],
        }
        .flattened()
    }

    #[test]
    fn sync_layer_keeps_its_own_frozen_blend_through_an_interruption() {
        let clips = assets(&[
            ("a.anim", still(0.0)),
            ("b.anim", still(10.0)),
            ("c.anim", still(20.0)),
            ("a2.anim", still(100.0)),
            ("b2.anim", still(110.0)),
            ("c2.anim", still(120.0)),
        ]);
        let c = still_ctrl_two_layers(vec![("A", "a2.anim"), ("B", "b2.anim"), ("C", "c2.anim")]);
        let mut a = animator();
        step(&mut a, &c, &clips, 0.0);
        a.set_bool("Go", true);
        step(&mut a, &c, &clips, 0.0);
        let p = step(&mut a, &c, &clips, 0.5);
        assert!((x(&p, 0) - 105.0).abs() < 1e-3, "{}", x(&p, 0));
        // B -> C interrupts the A -> B fade: the sync layer must hold 105 (its own
        // blend), not jump to B's 110 as the old-state fade did.
        a.set_bool("Go", false);
        a.set_bool("Stop", true);
        let p = step(&mut a, &c, &clips, 0.0);
        assert!((x(&p, 0) - 105.0).abs() < 1e-3, "popped to {}", x(&p, 0));
        let p = step(&mut a, &c, &clips, 0.5);
        assert!((x(&p, 0) - 112.5).abs() < 1e-3, "{}", x(&p, 0));
        let p = step(&mut a, &c, &clips, 0.6);
        assert!((x(&p, 0) - 120.0).abs() < 1e-3);
    }

    #[test]
    fn sync_layer_fires_events_of_its_own_clips_once() {
        let clips = assets(&[
            ("a.anim", with_events(clip_len("", 1.0, 1.0), &[(0.25, "Base")])),
            ("b.anim", with_events(clip_len("", 1.0, 1.0), &[(0.25, "Own")])),
        ]);
        let base = StateMachine { states: vec![state("A", "a.anim")], ..Default::default() };
        let run = |overrides: Vec<zestengine_assets::MotionOverride>| {
            let top = AnimLayer { name: "Sync".into(), sync_with: Some(0), motion_overrides: overrides, ..Default::default() };
            let c = two_layers(base.clone(), top).flattened();
            let mut a = animator();
            step(&mut a, &c, &clips, 0.0);
            a.state.fired.clear();
            step(&mut a, &c, &clips, 0.5);
            a.state.fired.clone()
        };
        let own = run(vec![zestengine_assets::MotionOverride { state: "A".into(), motion: Motion::Clip("b.anim".into()) }]);
        assert_eq!(own.iter().filter(|e| *e == "Base").count(), 1);
        assert_eq!(own.iter().filter(|e| *e == "Own").count(), 1, "{own:?}");
        // Sharing the source's clip: its event is the source's, fired once.
        assert_eq!(run(vec![]), vec!["Base".to_string()]);
    }

    #[test]
    fn fixed_mode_waits_for_the_step_to_react_to_parameters() {
        let clips = assets(&[("lo.anim", still(0.0)), ("hi.anim", still(10.0))]);
        let child = |c: &str, p: f32| BlendMotion { clip: c.into(), position: [p, 0.0], ..Default::default() };
        let tree = BlendTree { name: "T".into(), parameter: "P".into(), motions: vec![child("lo.anim", 0.0), child("hi.anim", 1.0)], ..Default::default() };
        let c = controller(vec![AnimState { name: "T".into(), motion: Motion::BlendTree(tree), ..Default::default() }], vec![], vec![]);
        let set = ClipSet { imported: &[], indices: &[], assets: &clips };
        let mut a = Animator { update_mode: AnimatorUpdateMode::Fixed, ..animator() };
        let timing = FrameTiming { dt: 0.1, unscaled_dt: 0.1, fixed_dt: 0.25 };
        let go = |a: &mut Animator| advance_timed(a, Some(&c), &rig(), &set, timing).pose;
        assert!(x(&go(&mut a), 0).abs() < 1e-3);
        a.set_float("P", 1.0);
        // 0.2 s of a 0.25 s step: the pose holds.
        assert!(x(&go(&mut a), 0).abs() < 1e-3);
        // The step lands: the new blend shows.
        assert!((x(&go(&mut a), 0) - 10.0).abs() < 1e-3);
    }

    #[test]
    fn container_joints_do_not_change_the_rig_frame() {
        // Armature (rotated -90 about X) -> Hips -> Spine, against the same bones without the Armature.
        let arm = Trs { t: Vec3::new(0.5, 0.0, 0.25), r: Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2), s: Vec3::ONE };
        let hips = Trs { t: Vec3::new(0.0, 0.0, 1.0), ..Default::default() };
        let spine = Trs { t: Vec3::new(0.0, 0.0, 0.5), ..Default::default() };
        let mut with = Rig::new(vec!["Hips".into(), "Spine".into(), "Armature".into()], vec![Some(2), Some(0), None], vec![hips, spine, arm], vec![0.0; 3], true);
        with.containers = vec![2];
        let without = Rig::new(vec!["Hips".into(), "Spine".into()], vec![None, Some(0)], vec![hips, spine], vec![0.0; 2], true);
        assert_eq!(with.motion_root(), Some(0));
        let mut p = Pose::rest(&with);
        p.trs[1].r = Quat::from_rotation_z(0.8);
        let mut q = Pose::rest(&without);
        q.trs[1].r = Quat::from_rotation_z(0.8);
        let (a, b) = (world_matrices(&with, &p), world_matrices(&without, &q));
        for i in 0..2 {
            assert!(a[i].abs_diff_eq(b[i], 1e-5), "joint {i}: {:?} vs {:?}", a[i], b[i]);
        }
    }
}
