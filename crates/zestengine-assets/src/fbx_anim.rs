//! FBX animation import: takes (animation stacks) to clips, clip splitting by
//! frame ranges, resampling or authored key times, keyframe reduction, loop
//! pose, and blendshape weight curves (`DeformPercent`) as sparse tracks.
//!
//! ufbx evaluates the FBX transform stack (Lcl TRS, Pre/PostRotation, rotation
//! order, pivots, inherit modes) so a sampled pose is already what the file's
//! application showed; this module decides *when* to sample, thins the result
//! and keeps the Euler -> quaternion output continuous.

use std::collections::HashSet;

use glam::{Quat, Vec3};

use crate::anim_clip::{AnimTrack, KeyInterp, Keyframe, TrackProperty};
use crate::anim_import::{self, AnimImport, ClipDef, ClipImportInfo, TakeInfo};
use crate::fbx_loader::{sample_times, source_frame_rate};
use crate::skeleton::{AnimChannel, AnimationClip, ChannelPath, Skeleton};
use crate::ModelImport;

/// A blendshape channel of the file with the names its curves are filed under.
pub(crate) struct MorphSource<'a> {
    /// The mesh node (track target).
    pub node: String,
    /// The shape name (the same rule the shapes themselves are named by).
    pub name: String,
    pub channel: &'a ufbx::BlendChannel,
}

/// The shape name for a channel: the leaf after a `Mesh.` / `Mesh|` prefix.
pub(crate) fn shape_name(raw: &str) -> String {
    raw.rsplit(['.', '|']).next().unwrap_or(raw).to_owned()
}

/// Every blendshape channel of every mesh in the file.
pub(crate) fn morph_sources(fbx: &ufbx::Scene) -> Vec<MorphSource<'_>> {
    let mut out = Vec::new();
    for node in &fbx.nodes {
        let Some(mesh) = &node.mesh else { continue };
        for deformer in &mesh.blend_deformers {
            for channel in &deformer.channels {
                if channel.target_shape.is_none() {
                    continue;
                }
                out.push(MorphSource { node: node.element.name.to_string(), name: shape_name(channel.element.name.as_ref()), channel });
            }
        }
    }
    out
}

/// The frames of a blendshape channel: `(shape, None)` for the full target and
/// `(shape, Some(w))` for an in-between that is fully reached at weight `w`
/// (0..1 of the full target). A plain channel is its one target shape.
pub(crate) fn channel_frames(channel: &ufbx::BlendChannel) -> Vec<(&ufbx::BlendShape, Option<f32>)> {
    let mut frames: Vec<(&ufbx::BlendShape, f32)> =
        channel.keyframes.iter().filter(|k| k.target_weight > 0.0).map(|k| (&*k.shape, k.target_weight as f32)).collect();
    if frames.len() < 2 {
        return channel.target_shape.as_ref().map(|s| (&**s, None)).into_iter().collect();
    }
    frames.sort_by(|a, b| a.1.total_cmp(&b.1));
    let full = frames[frames.len() - 1].1;
    let last = frames.len() - 1;
    frames
        .into_iter()
        .enumerate()
        .map(|(i, (shape, w))| (shape, (i != last).then_some(w / full)))
        .filter(|(_, f)| f.is_none_or(|w| w > 0.0 && w < 1.0))
        .collect()
}

/// The takes of the file (animation stacks) with their ranges.
pub(crate) fn list_takes(fbx: &ufbx::Scene) -> Vec<TakeInfo> {
    let fps = source_frame_rate(fbx.settings.frames_per_second);
    fbx.anim_stacks
        .iter()
        .map(|stack| TakeInfo {
            name: stack.element.name.to_string(),
            start: stack.time_begin as f32,
            stop: stack.time_end as f32,
            frame_rate: fps as f32,
            first_frame: (stack.time_begin * fps).round() as f32,
            last_frame: (stack.time_end * fps).round() as f32,
        })
        .collect()
}

/// One clip to cut: a take, a range in seconds of the take's own time, and the
/// settings it came with.
struct Cut<'a> {
    stack: &'a ufbx::AnimStack,
    name: String,
    t0: f64,
    t1: f64,
    def: Option<&'a ClipDef>,
}

/// The clips the import settings ask for, over the file's stacks.
fn plan<'a>(fbx: &'a ufbx::Scene, anim: &'a AnimImport, fps: f64) -> Vec<Cut<'a>> {
    let whole = |stack: &'a ufbx::AnimStack| Cut { stack, name: stack.element.name.to_string(), t0: stack.time_begin, t1: stack.time_end, def: None };
    if anim.clips.is_empty() {
        return fbx.anim_stacks.iter().map(|s| whole(s)).collect();
    }
    let mut cuts = Vec::new();
    let mut used: HashSet<usize> = HashSet::new();
    for def in &anim.clips {
        let stack = if def.take.is_empty() {
            fbx.anim_stacks.first()
        } else {
            fbx.anim_stacks.iter().find(|s| s.element.name.as_ref() == def.take)
        };
        let Some(stack) = stack else { continue };
        used.insert(stack.element.element_id as usize);
        let name = if def.name.is_empty() { stack.element.name.to_string() } else { def.name.clone() };
        cuts.push(Cut { stack, name, t0: def.first_frame as f64 / fps, t1: def.last_frame as f64 / fps, def: Some(def) });
    }
    if anim.import_unreferenced_takes {
        for stack in &fbx.anim_stacks {
            if !used.contains(&(stack.element.element_id as usize)) {
                cuts.push(whole(stack));
            }
        }
    }
    cuts
}

/// Authored key times of an element's animated properties in one stack, inside `[t0, t1]`.
fn key_times(stack: &ufbx::AnimStack, element: &ufbx::Element, prop: Option<&str>, t0: f64, t1: f64) -> Vec<f64> {
    let mut times = Vec::new();
    for layer in &stack.layers {
        for p in ufbx::find_anim_props(layer, element) {
            if prop.is_some_and(|name| p.prop_name.as_ref() != name) {
                continue;
            }
            for curve in p.anim_value.curves.iter().flatten() {
                times.extend(curve.keyframes.iter().map(|k| k.time).filter(|t| *t > t0 && *t < t1));
            }
        }
    }
    times
}

/// Sorted, de-duplicated (within a microsecond) times with both ends of the cut.
fn with_ends(mut times: Vec<f64>, t0: f64, t1: f64) -> Vec<f64> {
    times.push(t0);
    times.push(t1);
    times.sort_by(|a, b| a.total_cmp(b));
    times.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    times
}

fn v3(v: ufbx::Vec3) -> Vec3 {
    Vec3::new(v.x as f32, v.y as f32, v.z as f32)
}

fn q4(q: ufbx::Quat) -> Quat {
    Quat::from_xyzw(q.x as f32, q.y as f32, q.z as f32, q.w as f32).normalize()
}

/// How a joint's file-space local transform becomes the skeleton's: the unit
/// conversion (centimetres to metres) and, for a root joint of an
/// animation-only file, the transform of the nodes above it. Skinned files do not
/// need it (their instance transform carries the conversion); the default is the
/// identity.
#[derive(Clone, Copy, Debug)]
pub(crate) struct JointFix {
    /// Multiplies translations.
    pub unit: f32,
    /// Rotation and translation of the discarded ancestors, for a root joint.
    pub root: Option<(Quat, Vec3)>,
}

impl Default for JointFix {
    fn default() -> Self {
        Self { unit: 1.0, root: None }
    }
}

impl JointFix {
    /// The skeleton-space local transform of a file-space one.
    pub fn apply(&self, t: Vec3, r: Quat) -> (Vec3, Quat) {
        let t = t * self.unit;
        match self.root {
            Some((rq, rt)) => (rq * t + rt, (rq * r).normalize()),
            None => (t, r),
        }
    }
}

/// Samples bone transforms at absolute times of one stack: ufbx's per-node
/// local transform, or (Bake Animations) the full scene evaluation with
/// constraints solved, cached per time.
struct BoneSampler<'a> {
    anim: &'a ufbx::Anim,
    bone_nodes: &'a [Option<&'a ufbx::Node>],
    fixes: &'a [JointFix],
    baker: Option<&'a crate::fbx_extras::Baker<'a>>,
    /// Per joint: the node index of its parent joint's node.
    parent_nodes: &'a [Option<usize>],
    /// Per node index: the node index of its ufbx parent.
    node_parent: &'a [Option<usize>],
    frames: std::cell::RefCell<std::collections::HashMap<u64, Option<std::rc::Rc<Vec<glam::Mat4>>>>>,
}

impl<'a> BoneSampler<'a> {
    fn frame(&self, time: f64) -> Option<std::rc::Rc<Vec<glam::Mat4>>> {
        let baker = self.baker?;
        let mut cache = self.frames.borrow_mut();
        if cache.len() > 4096 {
            cache.clear();
        }
        cache.entry(time.to_bits()).or_insert_with(|| baker.frame(self.anim, time).map(std::rc::Rc::new)).clone()
    }

    /// Local translation, rotation and scale of bone `j` at `time`, in the skeleton's space.
    fn bone(&self, j: usize, time: f64) -> Option<(Vec3, Quat, Vec3)> {
        let node = self.bone_nodes.get(j).copied().flatten()?;
        let fix = self.fixes.get(j).copied().unwrap_or_default();
        let (t, r, s) = match self.frame(time) {
            Some(frame) => {
                // World matrices: the local matrix is taken against the parent joint
                // (or the file's parent node for a root joint), so inherit modes and
                // constraints on the way are part of the result.
                let at = node.element.typed_id as usize;
                let world = *frame.get(at)?;
                let parent = self.parent_nodes.get(j).copied().flatten().or_else(|| self.node_parent.get(at).copied().flatten());
                let local = match parent.and_then(|p| frame.get(p)) {
                    Some(pw) => pw.inverse() * world,
                    None => world,
                };
                let (s, r, t) = local.to_scale_rotation_translation();
                (t, r.normalize(), s)
            }
            None => {
                let tr = node.evaluate_transform(self.anim, time);
                (v3(tr.translation), q4(tr.rotation), v3(tr.scale))
            }
        };
        let (t, r) = fix.apply(t, r);
        Some((t, r, s))
    }
}

/// Error bounds for keeping a bone's curves as Hermite segments.
struct HermiteTol {
    rotation: f32,
    position: f32,
    scale: f32,
}

/// Seconds either side of a key used to measure its tangents.
const TANGENT_H: f64 = 1e-3;

fn log_rot(a: Quat, b: Quat) -> Vec3 {
    crate::skeleton::quat_log((a.inverse() * b).normalize())
}

/// One bone's curves over `[0, duration]` seconds of a cut, keeping the
/// authored key times (`authored`, relative) and subdividing segments until
/// the Hermite form follows `eval` within `tol`.
fn hermite_bone(eval: &dyn Fn(f64) -> Option<(Vec3, Quat, Vec3)>, duration: f64, authored: &[f64], tol: &HermiteTol, joint: usize) -> Option<[AnimChannel; 3]> {
    let mut times: Vec<f64> = vec![0.0, duration];
    times.extend(authored.iter().copied().filter(|t| *t > 0.0 && *t < duration));
    times.sort_by(|a, b| a.total_cmp(b));
    times.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    let key = |t: f64| eval(t);
    // Absolute tolerances from the value ranges over the cut.
    let probes: Vec<(Vec3, Quat, Vec3)> = (0..=16).filter_map(|k| key(duration * k as f64 / 16.0)).collect();
    if probes.is_empty() {
        return None;
    }
    let range = |f: &dyn Fn(&(Vec3, Quat, Vec3)) -> Vec3| {
        let (mut lo, mut hi) = (f(&probes[0]), f(&probes[0]));
        for p in &probes {
            lo = lo.min(f(p));
            hi = hi.max(f(p));
        }
        (hi - lo).max_element()
    };
    let pos_tol = (range(&|p| p.0) * tol.position / 100.0).max(1e-6);
    let scale_tol = (range(&|p| p.2) * tol.scale / 100.0).max(1e-6);
    let rot_tol = tol.rotation.max(1e-4);

    for round in 0..5 {
        let n = times.len();
        let mut vals = Vec::with_capacity(n);
        let mut tan = Vec::with_capacity(n);
        for &t in &times {
            let v = key(t)?;
            // One-sided differences, so a jump in the curve does not smear into its neighbour.
            // Past the cut's ends the take simply continues (or holds its last value).
            let after = key(t + TANGENT_H)?;
            let before = key(t - TANGENT_H)?;
            let (ho, hi) = (TANGENT_H as f32, TANGENT_H as f32);
            tan.push((
                ((v.0 - before.0) / hi, (after.0 - v.0) / ho),
                (log_rot(before.1, v.1) / hi, log_rot(v.1, after.1) / ho),
                ((v.2 - before.2) / hi, (after.2 - v.2) / ho),
            ));
            vals.push(v);
        }
        let mut rots: Vec<Quat> = vals.iter().map(|v| v.1).collect();
        anim_import::make_continuous(&mut rots);
        // At the cut's ends the missing side repeats the present one.
        let ft: Vec<f32> = times.iter().map(|t| *t as f32).collect();
        let build_vec = |path: ChannelPath, pick: fn(&(Vec3, Quat, Vec3)) -> Vec3, tg: &dyn Fn(usize) -> (Vec3, Vec3)| {
            let mut ch = AnimChannel::vec(joint, path, ft.clone(), vals.iter().map(pick).collect());
            ch.extra = Some(Box::new(crate::skeleton::ChannelExtra {
                tangents: Some(crate::skeleton::Tangents { tan_in: (0..n).map(|i| tg(i).0).collect(), tan_out: (0..n).map(|i| tg(i).1).collect() }),
                packed: None,
            }));
            ch
        };
        let tch = build_vec(ChannelPath::Translation, |v| v.0, &|i| tan[i].0);
        let sch = build_vec(ChannelPath::Scale, |v| v.2, &|i| tan[i].2);
        let mut rch = AnimChannel::rot(joint, ft.clone(), rots.clone());
        rch.extra = Some(Box::new(crate::skeleton::ChannelExtra {
            tangents: Some(crate::skeleton::Tangents { tan_in: tan.iter().map(|t| t.1.0).collect(), tan_out: tan.iter().map(|t| t.1.1).collect() }),
            packed: None,
        }));
        // Segments whose Hermite form strays from the truth get a key in the middle.
        let mut split = Vec::new();
        for i in 0..n - 1 {
            let dt = times[i + 1] - times[i];
            if dt < 2e-3 {
                continue;
            }
            let mut bad = false;
            for s in [0.25, 0.5, 0.75] {
                let at = times[i] + dt * s;
                let Some(truth) = key(at) else { continue };
                let tp = tch.sample_vec(at as f32).unwrap_or(truth.0);
                let sp = sch.sample_vec(at as f32).unwrap_or(truth.2);
                let rp = rch.sample_quat(at as f32).unwrap_or(truth.1);
                let rot_err = 2.0 * rp.dot(truth.1).abs().clamp(0.0, 1.0).acos();
                if (tp - truth.0).abs().max_element() > pos_tol || (sp - truth.2).abs().max_element() > scale_tol || rot_err > rot_tol {
                    bad = true;
                    break;
                }
            }
            if bad {
                split.push(times[i] + dt * 0.5);
            }
        }
        if split.is_empty() || n > 4096 || round == 4 {
            // A bone that never moves is one key.
            let flat = vals.iter().all(|v| (v.0 - vals[0].0).abs().max_element() < pos_tol.min(1e-5) && (v.2 - vals[0].2).abs().max_element() < scale_tol.min(1e-5) && v.1.dot(vals[0].1).abs() > 0.999_999);
            if flat {
                let one = |mut c: AnimChannel| {
                    c.times.truncate(1);
                    c.vec_values.truncate(1);
                    c.quat_values.truncate(1);
                    c.extra = None;
                    c
                };
                return Some([one(tch), one(rch), one(sch)]);
            }
            return Some([tch, rch, sch]);
        }
        times.extend(split);
        times.sort_by(|a, b| a.total_cmp(b));
    }
    unreachable!("the loop returns on its last round");
}

/// All clips of the file for the bones in `bone_nodes` (joint order of `skeleton`),
/// with the blendshape curves of `morphs`. `fixes` converts each joint's sampled
/// transform (empty = none, see [`JointFix`]).
pub(crate) fn build_clips(
    fbx: &ufbx::Scene,
    skeleton: &Skeleton,
    bone_nodes: &[Option<&ufbx::Node>],
    settings: &ModelImport,
    morphs: &[MorphSource<'_>],
    fixes: &[JointFix],
) -> (Vec<AnimationClip>, Vec<ClipImportInfo>) {
    let opts = &settings.animation;
    if !opts.import_animation {
        return (Vec::new(), Vec::new());
    }
    let fps = source_frame_rate(fbx.settings.frames_per_second);
    let root = skeleton.motion_root();
    let baker = opts.bake_animations.then(|| crate::fbx_extras::Baker::new(fbx, true));
    let parent_nodes: Vec<Option<usize>> = skeleton
        .joints
        .iter()
        .map(|j| j.parent.and_then(|p| bone_nodes.get(p).copied().flatten()).map(|n| n.element.typed_id as usize))
        .collect();
    let node_parent: Vec<Option<usize>> = fbx.nodes.iter().map(|n| n.parent.as_ref().map(|p| p.element.typed_id as usize)).collect();
    let mut clips: Vec<AnimationClip> = Vec::new();
    let mut infos = Vec::new();
    for cut in plan(fbx, opts, fps) {
        let duration = (cut.t1 - cut.t0).max(0.0);
        if duration <= 1e-4 {
            continue;
        }
        let anim = cut.stack.anim.as_ref();
        let sampler = BoneSampler { anim, bone_nodes, fixes, baker: baker.as_ref(), parent_nodes: &parent_nodes, node_parent: &node_parent, frames: Default::default() };
        // Authored key times (relative to the cut) of one bone; the baked evaluation
        // also depends on every other node (constraint sources, parents), so it takes all of them.
        let authored = |node: &ufbx::Node| -> Vec<f64> {
            let mut keys = Vec::new();
            if baker.is_some() {
                for n in &fbx.nodes {
                    keys.extend(key_times(cut.stack, &n.element, None, cut.t0, cut.t1));
                }
            } else {
                keys.extend(key_times(cut.stack, &node.element, None, cut.t0, cut.t1));
            }
            keys.iter().map(|t| t - cut.t0).collect()
        };
        let keep = opts.keep_tangents;
        // Sample times, in seconds from the clip start.
        let local: Vec<f64> = if opts.resample_curves && !keep {
            sample_times(duration, fps)
        } else {
            let mut keys = Vec::new();
            for node in bone_nodes.iter().flatten() {
                keys.extend(authored(node));
                if baker.is_some() {
                    break; // the union is the same for every bone
                }
            }
            let mut v: Vec<f64> = keys.into_iter().map(|t| t + cut.t0).collect();
            v = with_ends(v, cut.t0, cut.t1);
            v.into_iter().map(|t| t - cut.t0).collect()
        };
        let times: Vec<f32> = local.iter().map(|t| *t as f32).collect();
        let tol = HermiteTol { rotation: opts.rotation_error.to_radians(), position: opts.position_error, scale: opts.scale_error };
        let mut channels = Vec::new();
        let mut shared_authored: Option<Vec<f64>> = None;
        for (j, node) in bone_nodes.iter().enumerate() {
            let Some(node) = node else { continue };
            if keep {
                let auth = if baker.is_some() { shared_authored.get_or_insert_with(|| authored(node)).clone() } else { authored(node) };
                let eval = |t: f64| sampler.bone(j, cut.t0 + t);
                if let Some(chs) = hermite_bone(&eval, duration, &auth, &tol, j) {
                    channels.extend(chs);
                    continue;
                }
            }
            let mut trans = Vec::with_capacity(local.len());
            let mut rots = Vec::with_capacity(local.len());
            let mut scales = Vec::with_capacity(local.len());
            for t in &local {
                let (t, r, s) = sampler.bone(j, cut.t0 + t).unwrap_or((Vec3::ZERO, Quat::IDENTITY, Vec3::ONE));
                trans.push(t);
                rots.push(r);
                scales.push(s);
            }
            // Euler -> quaternion can land on either sign; keep neighbours in one hemisphere.
            anim_import::make_continuous(&mut rots);
            channels.push(AnimChannel::vec(j, ChannelPath::Translation, times.clone(), trans));
            channels.push(AnimChannel::rot(j, times.clone(), rots));
            channels.push(AnimChannel::vec(j, ChannelPath::Scale, times.clone(), scales));
        }
        let mut clip = AnimationClip { name: cut.name.clone(), duration: duration as f32, channels, morph_tracks: Vec::new(), playback: None };
        let (loop_deg, loop_pos) = anim_import::loop_error(&clip);
        if cut.def.is_some_and(|d| d.loop_pose) {
            // The pose correction moves the values; tangents measured before it would be stale.
            for ch in &mut clip.channels {
                ch.extra = None;
            }
            anim_import::apply_loop_pose(&mut clip, root);
        }
        if opts.remove_constant_scale_curves {
            anim_import::remove_constant_scale(&mut clip, |j| skeleton.joints.get(j).map(|jt| jt.rest_scale).unwrap_or(Vec3::ONE));
        }
        if opts.compression.reduces() && opts.resample_curves && !keep {
            crate::anim_pack::compress_clip(&mut clip, opts.compression, opts.rotation_error, opts.position_error, opts.scale_error);
        } else if opts.compression.quantizes() {
            // Authored keys stay; only the storage form changes.
            crate::anim_pack::quantize_clip(&mut clip, opts.rotation_error, opts.position_error, opts.scale_error);
        }
        let mut info = ClipImportInfo {
            take: cut.stack.element.name.to_string(),
            first_frame: (cut.t0 * fps).round() as f32,
            last_frame: (cut.t1 * fps).round() as f32,
            frame_rate: fps as f32,
            loop_error_deg: loop_deg,
            loop_error_pos: loop_pos,
            ..Default::default()
        };
        if let Some(d) = cut.def {
            info.loop_time = d.loop_time;
            info.loop_pose = d.loop_pose;
            info.lock_root_rotation = d.lock_root_rotation;
            info.lock_root_height_y = d.lock_root_height_y;
            info.lock_root_position_xz = d.lock_root_position_xz;
            info.cycle_offset = d.cycle_offset;
            info.events = d.events.clone();
            info.playback = Some(anim_import::ClipPlayback::from_def(d, skeleton, &clip));
        }
        clip.playback = info.playback.clone().map(Box::new);
        if settings.blend_shapes.import_curves && settings.import_blend_shapes {
            for m in morphs {
                if let Some(track) = morph_track(cut.stack, m, cut.t0, cut.t1) {
                    info.morph_tracks.push(track);
                }
            }
        }
        if opts.animated_custom_properties {
            info.morph_tracks.extend(custom_property_tracks(fbx, cut.stack, cut.t0, cut.t1));
        }
        clip.morph_tracks = info.morph_tracks.clone();
        // Clip names must be unique: the animator finds clips by name.
        let base = clip.name.clone();
        let mut n = 1;
        while clips.iter().any(|c| c.name == clip.name) {
            clip.name = format!("{base} {n}");
            n += 1;
        }
        clips.push(clip);
        infos.push(info);
    }
    (clips, infos)
}

/// Property tracks for the animated user properties of every node inside
/// `[t0, t1]` of a stack: `TrackProperty::CustomProperty` tracks named like
/// the property (a vector's lanes are `Name.x`, `Name.y`, `Name.z`). Ints and
/// bools step; floats keep linear or cubic keys as authored.
pub(crate) fn custom_property_tracks(fbx: &ufbx::Scene, stack: &ufbx::AnimStack, t0: f64, t1: f64) -> Vec<AnimTrack> {
    let anim = stack.anim.as_ref();
    let mut out = Vec::new();
    for node in &fbx.nodes {
        let mut done: HashSet<String> = HashSet::new();
        for layer in &stack.layers {
            for p in ufbx::find_anim_props(layer, &node.element) {
                let name = p.prop_name.to_string();
                let Some(prop) = node.element.props.find_prop(&name) else { continue };
                if !prop.flags.has_any(ufbx::PropFlags::USER_DEFINED) || !done.insert(name.clone()) {
                    continue;
                }
                let lanes: &[(usize, Option<&str>)] = match prop.type_ {
                    ufbx::PropType::Boolean | ufbx::PropType::Integer | ufbx::PropType::Number | ufbx::PropType::Distance => &[(0, None)],
                    ufbx::PropType::Vector | ufbx::PropType::Color | ufbx::PropType::Translation | ufbx::PropType::Rotation | ufbx::PropType::Scaling => {
                        &[(0, Some("x")), (1, Some("y")), (2, Some("z"))]
                    }
                    _ => continue,
                };
                let step = matches!(prop.type_, ufbx::PropType::Boolean | ufbx::PropType::Integer);
                for &(lane, suffix) in lanes {
                    let Some(curve) = p.anim_value.curves[lane].as_ref() else { continue };
                    let mut times: Vec<f64> = curve.keyframes.iter().map(|k| k.time).filter(|t| *t > t0 && *t < t1).collect();
                    times = with_ends(times, t0, t1);
                    let eval = |t: f64| -> f32 {
                        let v = ufbx::evaluate_prop(anim, &node.element, &name, t).value_vec4;
                        [v.x, v.y, v.z][lane] as f32
                    };
                    let cubic = curve.keyframes.iter().any(|k| k.interpolation == ufbx::Interpolation::Cubic);
                    let slope = |t: f64| ((eval(t + 1e-4) - eval(t - 1e-4)) / 2e-4).clamp(-1e6, 1e6);
                    let mut keys: Vec<Keyframe> = times
                        .iter()
                        .map(|&t| {
                            let v = eval(t);
                            if step {
                                Keyframe { time: (t - t0) as f32, value: [v, 0.0, 0.0, 0.0], interp: KeyInterp::Step, ..Default::default() }
                            } else if cubic {
                                let s = slope(t);
                                Keyframe { time: (t - t0) as f32, value: [v, 0.0, 0.0, 0.0], interp: KeyInterp::Custom, tangent_in: Some([s, 0.0, 0.0, 0.0]), tangent_out: Some([s, 0.0, 0.0, 0.0]) }
                            } else {
                                Keyframe { time: (t - t0) as f32, value: [v, 0.0, 0.0, 0.0], interp: KeyInterp::Linear, ..Default::default() }
                            }
                        })
                        .collect();
                    if keys.iter().all(|k| (k.value[0] - keys[0].value[0]).abs() < 1e-6) {
                        keys.truncate(1);
                        keys[0].time = 0.0;
                        keys[0].tangent_in = None;
                        keys[0].tangent_out = None;
                    }
                    let full = match suffix {
                        Some(s) => format!("{name}.{s}"),
                        None => name.clone(),
                    };
                    let mut track = AnimTrack::prop(&node_name(node), TrackProperty::CustomProperty, &full, 0);
                    track.keys = keys;
                    out.push(track);
                }
            }
        }
    }
    out
}

fn node_name(n: &ufbx::Node) -> String {
    if n.element.name.is_empty() { format!("node {}", n.element.element_id) } else { n.element.name.to_string() }
}

/// Whether any layer of any stack animates a property of `node`.
pub(crate) fn is_animated(fbx: &ufbx::Scene, node: &ufbx::Node) -> bool {
    fbx.anim_stacks.iter().any(|stack| stack.layers.iter().any(|layer| !ufbx::find_anim_props(layer, &node.element).is_empty()))
}

/// The skeleton and clips of a file with animation but no mesh (Mixamo "Without
/// Skin"): there is no skin to take the joints from, so they come from the node
/// hierarchy. A joint is a bone node or an animated node that is not a mesh,
/// light or camera, plus the nodes between two joints (so the hierarchy stays
/// connected); nodes above the topmost joint are not joints. Those ancestors
/// (the unit conversion on the scene root, an `Armature` group) are folded into
/// the root joint instead, so the skeleton is in metres and in the target axes
/// whatever the file's unit was, and its clips carry the same conversion. `None`
/// when the file has no such nodes or no animation.
pub(crate) fn build_anim_only(fbx: &ufbx::Scene, settings: &ModelImport) -> Option<(Skeleton, Vec<AnimationClip>, Vec<ClipImportInfo>)> {
    if fbx.anim_stacks.is_empty() {
        return None;
    }
    let real = |n: &ufbx::Node| !n.is_root && n.mesh.is_none() && n.light.is_none() && n.camera.is_none();
    let core: HashSet<u32> = fbx.nodes.iter().filter(|n| real(n) && (n.bone.is_some() || is_animated(fbx, n))).map(|n| n.element.element_id).collect();
    if core.is_empty() {
        return None;
    }
    // Plain nodes that sit between two joints.
    let mut joint_ids = core.clone();
    for n in fbx.nodes.iter().filter(|n| core.contains(&n.element.element_id)) {
        let mut chain = Vec::new();
        let mut cur = n.parent.as_deref();
        while let Some(p) = cur {
            if p.is_root {
                chain.clear();
                break;
            }
            if core.contains(&p.element.element_id) || joint_ids.contains(&p.element.element_id) {
                break;
            }
            chain.push(p.element.element_id);
            cur = p.parent.as_deref();
        }
        if cur.is_some() {
            joint_ids.extend(chain);
        }
    }
    let mut index: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    let mut joints: Vec<crate::skeleton::Joint> = Vec::new();
    let mut bone_nodes: Vec<Option<&ufbx::Node>> = Vec::new();
    let mut fixes: Vec<JointFix> = Vec::new();
    let mut worlds: Vec<glam::Mat4> = Vec::new();
    // Nodes come parent first, so a joint's parent joint is already placed.
    for n in fbx.nodes.iter().filter(|n| joint_ids.contains(&n.element.element_id)) {
        let parent = n.parent.as_ref().and_then(|p| index.get(&p.element.element_id).copied());
        let fix = match parent {
            Some(p) => JointFix { unit: fixes[p].unit, root: None },
            None => {
                // A root joint: the nodes above it become part of it.
                let above = n.parent.as_ref().map(|p| crate::fbx_loader::matrix_to_mat4(&p.node_to_world)).unwrap_or(glam::Mat4::IDENTITY);
                let scale = above.x_axis.truncate().length();
                let unit = if scale.is_finite() && scale > 1e-9 { scale } else { 1.0 };
                let rot = glam::Mat3::from_cols(above.x_axis.truncate() / unit, above.y_axis.truncate() / unit, above.z_axis.truncate() / unit);
                JointFix { unit, root: Some((Quat::from_mat3(&rot).normalize(), above.w_axis.truncate())) }
            }
        };
        let (t, r) = fix.apply(v3(n.local_transform.translation), q4(n.local_transform.rotation));
        let s = v3(n.local_transform.scale);
        let local = glam::Mat4::from_scale_rotation_translation(s, r, t);
        let world = match parent {
            Some(p) => worlds[p] * local,
            None => local,
        };
        index.insert(n.element.element_id, joints.len());
        joints.push(crate::skeleton::Joint {
            name: if n.element.name.is_empty() { format!("joint {}", n.element.element_id) } else { n.element.name.to_string() },
            parent,
            inverse_bind: world.inverse(),
            rest_translation: t,
            rest_rotation: r,
            rest_scale: s,
        });
        worlds.push(world);
        bone_nodes.push(Some(n));
        fixes.push(fix);
    }
    let skeleton = Skeleton { joints, rig: None, ..Default::default() };
    let morphs = morph_sources(fbx);
    let (clips, infos) = build_clips(fbx, &skeleton, &bone_nodes, settings, &morphs, &fixes);
    Some((skeleton, clips, infos))
}

/// World position (metres, y-up, as the importer converts) of every bone node
/// of the file's first take `t` seconds in, by ufbx's own scene evaluation (the
/// whole transform stack and parent chain, not this importer's joint transforms):
/// the reference the playback tests compare an imported clip against.
pub fn reference_bone_positions(path: &std::path::Path, t: f64) -> anyhow::Result<Vec<(String, [f32; 3])>> {
    use anyhow::Context as _;
    let opts = ufbx::LoadOpts {
        target_axes: ufbx::CoordinateAxes::right_handed_y_up(),
        target_unit_meters: 1.0,
        space_conversion: ufbx::SpaceConversion::TransformRoot,
        ..Default::default()
    };
    let fbx = ufbx::load_file(path.to_str().context("non-UTF8 path")?, opts).map_err(|e| anyhow::anyhow!("{}", e.description.as_ref()))?;
    let stack = fbx.anim_stacks.first().context("no animation")?;
    let at = ufbx::evaluate_scene(&fbx, &stack.anim, stack.time_begin + t, ufbx::EvaluateOpts::default()).map_err(|e| anyhow::anyhow!("{}", e.description.as_ref()))?;
    Ok(at
        .nodes
        .iter()
        .filter(|n| n.bone.is_some())
        .map(|n| {
            let m = &n.node_to_world;
            (n.element.name.to_string(), [m.m03 as f32, m.m13 as f32, m.m23 as f32])
        })
        .collect())
}

/// The weight curve of one blendshape channel inside `[t0, t1]` of a stack, or
/// `None` when the stack does not animate it. Weights are `DeformPercent / 100`
/// (1 = the shape's full target). Authored keys keep their interpolation and
/// Hermite tangents; the cut's ends are added with finite-difference tangents.
fn morph_track(stack: &ufbx::AnimStack, source: &MorphSource<'_>, t0: f64, t1: f64) -> Option<AnimTrack> {
    let element = &source.channel.element;
    let authored = key_times(stack, element, Some("DeformPercent"), t0, t1);
    // Animated at all (a key anywhere in the stack, even outside the cut)?
    let animated = !authored.is_empty() || !key_times(stack, element, Some("DeformPercent"), f64::MIN, f64::MAX).is_empty();
    if !animated {
        return None;
    }
    // The curve that carries the keys (one layer is the common case).
    let curve = stack
        .layers
        .iter()
        .flat_map(|layer| ufbx::find_anim_props(layer, element).iter())
        .filter(|p| p.prop_name.as_ref() == "DeformPercent")
        .find_map(|p| p.anim_value.curves[0].as_ref());
    let anim = stack.anim.as_ref();
    let eval = |t: f64| source.channel.evaluate_blend_weight(anim, t) as f32;
    let slope = |t: f64| {
        let h = 1e-4;
        ((eval(t + h) - eval(t - h)) / (2.0 * h as f32)).clamp(-1e6, 1e6)
    };
    let keyframe_at = |t: f64| curve.and_then(|c| c.keyframes.iter().find(|k| (k.time - t).abs() < 1e-6));
    let times = with_ends(authored, t0, t1);
    let mut keys: Vec<Keyframe> = Vec::with_capacity(times.len());
    for (i, &t) in times.iter().enumerate() {
        let value = eval(t);
        let key = keyframe_at(t);
        let (interp, tin, tout) = match key {
            Some(k) => match k.interpolation {
                ufbx::Interpolation::ConstantPrev => (KeyInterp::Step, None, None),
                ufbx::Interpolation::ConstantNext => (KeyInterp::Step, None, None),
                ufbx::Interpolation::Linear => (KeyInterp::Linear, None, None),
                ufbx::Interpolation::Cubic => {
                    let s = |tg: ufbx::Tangent| if tg.dx.abs() > 1e-9 { tg.dy / tg.dx / 100.0 } else { 0.0 };
                    (KeyInterp::Custom, Some(s(k.left)), Some(s(k.right)))
                }
            },
            // A cut end inside a segment: smooth if the segment around it is.
            None => {
                let prev = curve.and_then(|c| c.keyframes.iter().rev().find(|k| k.time < t));
                if prev.is_some_and(|k| k.interpolation == ufbx::Interpolation::Cubic) {
                    let s = slope(t);
                    (KeyInterp::Custom, Some(s), Some(s))
                } else if prev.is_some_and(|k| k.interpolation == ufbx::Interpolation::Linear) || prev.is_none() {
                    (KeyInterp::Linear, None, None)
                } else {
                    (KeyInterp::Step, None, None)
                }
            }
        };
        // ConstantNext holds the *next* key's value over the segment.
        let value = match key.map(|k| k.interpolation) {
            Some(ufbx::Interpolation::ConstantNext) if i + 1 < times.len() => eval(times[i + 1]),
            _ => value,
        };
        keys.push(Keyframe {
            time: (t - t0) as f32,
            value: [value, 0.0, 0.0, 0.0],
            interp,
            tangent_in: tin.map(|s| [s, 0.0, 0.0, 0.0]),
            tangent_out: tout.map(|s| [s, 0.0, 0.0, 0.0]),
        });
    }
    // One key for a curve that never moves inside the cut.
    if keys.iter().all(|k| (k.value[0] - keys[0].value[0]).abs() < 1e-6) {
        keys.truncate(1);
        keys[0].time = 0.0;
        keys[0].interp = KeyInterp::Linear;
        keys[0].tangent_in = None;
        keys[0].tangent_out = None;
    }
    let mut track = AnimTrack::prop(&source.node, TrackProperty::BlendShape, &source.name, 0);
    track.keys = keys;
    Some(track)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;
    use std::fmt::Write as _;

    const TICKS: f64 = 46_186_158_000.0;
    fn ticks(sec: f64) -> i64 {
        (sec * TICKS).round() as i64
    }

    /// One animation curve: (seconds, value) with the FBX interpolation flags.
    struct Curve {
        id: i64,
        keys: Vec<(f64, f32)>,
        flags: u32,
    }

    /// A tiny ASCII FBX writer: a grid mesh skinned to one bone (`Root`), blendshape
    /// channels and animation stacks.
    struct Fbx {
        grid: usize,
        pre_rotation: [f32; 3],
        rotation_order: u8,
        /// (channel name, FullWeights of its shapes, offsets per shape)
        channels: Vec<(String, Vec<f64>, usize)>,
        /// (take name, [start, stop] seconds, root rotation curves X/Y/Z, root translation X curve, blend curve for channel 0)
        takes: Vec<Take>,
        fps_mode: u8,
        /// No mesh, geometry or skin: an animation-only file (the bones are `Root` and `Child`).
        skinless: bool,
        /// Initial `DeformPercent` of every channel.
        deform_percent: f32,
    }
    struct Take {
        name: String,
        span: (f64, f64),
        rot: [Option<Vec<(f64, f32)>>; 3],
        tx: Option<Vec<(f64, f32)>>,
        blend: Option<(Vec<(f64, f32)>, u32)>,
    }

    impl Fbx {
        fn new() -> Self {
            Self { grid: 2, pre_rotation: [0.0; 3], rotation_order: 0, channels: Vec::new(), takes: Vec::new(), fps_mode: 6, skinless: false, deform_percent: 0.0 }
        }

        fn render(&self) -> String {
            let mut s = String::new();
            let side = self.grid + 1;
            let mut verts = String::new();
            for y in 0..side {
                for x in 0..side {
                    let _ = write!(verts, "{},{},{},", x as f32, y as f32, 0.0);
                }
            }
            let mut polys = String::new();
            for y in 0..self.grid {
                for x in 0..self.grid {
                    let a = y * side + x;
                    let _ = write!(polys, "{},{},{},{},", a, a + 1, a + side + 1, -((a + side) as i64) - 1);
                }
            }
            let nv = side * side;
            let _ = write!(
                s,
                "; FBX 7.4.0 project file\nFBXHeaderExtension:  {{\n\tFBXHeaderVersion: 1003\n\tFBXVersion: 7400\n}}\n\
GlobalSettings:  {{\n\tVersion: 1000\n\tProperties70:  {{\n\
\t\tP: \"UpAxis\", \"int\", \"Integer\", \"\",1\n\t\tP: \"UpAxisSign\", \"int\", \"Integer\", \"\",1\n\
\t\tP: \"FrontAxis\", \"int\", \"Integer\", \"\",2\n\t\tP: \"FrontAxisSign\", \"int\", \"Integer\", \"\",1\n\
\t\tP: \"CoordAxis\", \"int\", \"Integer\", \"\",0\n\t\tP: \"CoordAxisSign\", \"int\", \"Integer\", \"\",1\n\
\t\tP: \"UnitScaleFactor\", \"double\", \"Number\", \"\",1\n\t\tP: \"TimeMode\", \"enum\", \"\", \"\",{}\n\t}}\n}}\n",
                self.fps_mode
            );
            s += "Objects:  {\n";
            if !self.skinless {
                let _ = write!(
                    s,
                    "\tGeometry: 100, \"Geometry::Body\", \"Mesh\" {{\n\t\tVertices: *{} {{\n\t\t\ta: {}\n\t\t}}\n\t\tPolygonVertexIndex: *{} {{\n\t\t\ta: {}\n\t\t}}\n\t\tGeometryVersion: 124\n\t}}\n",
                    nv * 3,
                    verts.trim_end_matches(','),
                    self.grid * self.grid * 4,
                    polys.trim_end_matches(',')
                );
                let _ = write!(s, "\tModel: 200, \"Model::Body\", \"Mesh\" {{\n\t\tVersion: 232\n\t\tProperties70:  {{\n\t\t}}\n\t}}\n");
            }
            let [px, py, pz] = self.pre_rotation;
            let _ = write!(
                s,
                "\tModel: 150, \"Model::Root\", \"LimbNode\" {{\n\t\tVersion: 232\n\t\tProperties70:  {{\n\t\t\tP: \"RotationOrder\", \"enum\", \"\", \"\",{}\n\t\t\tP: \"PreRotation\", \"Vector3D\", \"Vector\", \"\",{},{},{}\n\t\t\tP: \"RotationActive\", \"bool\", \"\", \"\",1\n\t\t\tP: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\",0,{},0\n\t\t}}\n\t}}\n",
                self.rotation_order, px, py, pz, if self.skinless { 100 } else { 0 }
            );
            if self.skinless {
                s += "\tModel: 151, \"Model::Child\", \"LimbNode\" {\n\t\tVersion: 232\n\t\tProperties70:  {\n\t\t\tP: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\",0,10,0\n\t\t}\n\t}\n";
                // The skeleton attributes that make LimbNodes bones.
                s += "\tNodeAttribute: 160, \"NodeAttribute::\", \"LimbNode\" {\n\t\tTypeFlags: \"Skeleton\"\n\t}\n\tNodeAttribute: 161, \"NodeAttribute::\", \"LimbNode\" {\n\t\tTypeFlags: \"Skeleton\"\n\t}\n";
            }
            if !self.skinless {
                let idx: Vec<String> = (0..nv).map(|i| i.to_string()).collect();
                let ones: Vec<String> = (0..nv).map(|_| "1".to_string()).collect();
                let ident = "1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1";
                let _ = write!(s, "\tDeformer: 300, \"Deformer::Skin\", \"Skin\" {{\n\t\tVersion: 101\n\t}}\n");
                let _ = write!(
                    s,
                    "\tDeformer: 301, \"SubDeformer::Cluster\", \"Cluster\" {{\n\t\tVersion: 100\n\t\tIndexes: *{nv} {{\n\t\t\ta: {}\n\t\t}}\n\t\tWeights: *{nv} {{\n\t\t\ta: {}\n\t\t}}\n\t\tTransform: *16 {{\n\t\t\ta: {ident}\n\t\t}}\n\t\tTransformLink: *16 {{\n\t\t\ta: {ident}\n\t\t}}\n\t}}\n",
                    idx.join(","),
                    ones.join(",")
                );
            }
            // Blendshapes.
            let mut conns = String::new();
            let mut id = 1000i64;
            if !self.channels.is_empty() {
                let _ = write!(s, "\tDeformer: 400, \"Deformer::Shapes\", \"BlendShape\" {{\n\t\tVersion: 100\n\t}}\n");
                conns += "\tC: \"OO\",400,100\n";
            }
            let mut channel_ids = Vec::new();
            for (name, full, offsets) in &self.channels {
                let cid = id;
                id += 1;
                channel_ids.push(cid);
                let fw: Vec<String> = full.iter().map(|f| f.to_string()).collect();
                let _ = write!(
                    s,
                    "\tDeformer: {cid}, \"SubDeformer::{name}\", \"BlendShapeChannel\" {{\n\t\tVersion: 100\n\t\tDeformPercent: {}\n\t\tFullWeights: *{} {{\n\t\t\ta: {}\n\t\t}}\n\t}}\n",
                    self.deform_percent,
                    full.len(),
                    fw.join(",")
                );
                let _ = writeln!(conns, "\tC: \"OO\",{cid},400");
                for (k, _) in full.iter().enumerate() {
                    let gid = id;
                    id += 1;
                    let n = (*offsets).min(nv);
                    let ind: Vec<String> = (0..n).map(|i| i.to_string()).collect();
                    // A zero-offset shape has all-zero deltas (an "empty" shape).
                    let scale = if *offsets == 0 { 0.0 } else { 1.0 };
                    let dv: Vec<String> = (0..n).map(|_| format!("0,0,{}", scale * (k as f32 + 1.0))).collect();
                    let _ = write!(
                        s,
                        "\tGeometry: {gid}, \"Geometry::{name}\", \"Shape\" {{\n\t\tVersion: 100\n\t\tIndexes: *{n} {{\n\t\t\ta: {}\n\t\t}}\n\t\tVertices: *{} {{\n\t\t\ta: {}\n\t\t}}\n\t}}\n",
                        ind.join(","),
                        n * 3,
                        dv.join(",")
                    );
                    let _ = writeln!(conns, "\tC: \"OO\",{gid},{cid}");
                }
            }
            // Animation.
            for (ti, take) in self.takes.iter().enumerate() {
                let stack = 500 + ti as i64 * 100;
                let layer = stack + 1;
                let (a, b) = (ticks(take.span.0), ticks(take.span.1));
                let _ = write!(
                    s,
                    "\tAnimationStack: {stack}, \"AnimStack::{}\", \"\" {{\n\t\tProperties70:  {{\n\t\t\tP: \"LocalStart\", \"KTime\", \"Time\", \"\",{a}\n\t\t\tP: \"LocalStop\", \"KTime\", \"Time\", \"\",{b}\n\t\t\tP: \"ReferenceStart\", \"KTime\", \"Time\", \"\",{a}\n\t\t\tP: \"ReferenceStop\", \"KTime\", \"Time\", \"\",{b}\n\t\t}}\n\t}}\n\tAnimationLayer: {layer}, \"AnimLayer::BaseLayer\", \"\" {{\n\t}}\n",
                    take.name
                );
                let _ = writeln!(conns, "\tC: \"OO\",{layer},{stack}");
                let mut next = stack + 2;
                let mut emit = |s: &mut String, conns: &mut String, node_name: &str, target: i64, prop: &str, curves: Vec<(&str, Curve)>| {
                    let node = next;
                    next += 1;
                    let _ = write!(s, "\tAnimationCurveNode: {node}, \"AnimCurveNode::{node_name}\", \"\" {{\n\t\tProperties70:  {{\n");
                    for (d, _) in &curves {
                        let _ = writeln!(s, "\t\t\tP: \"{d}\", \"Number\", \"\", \"A\",0");
                    }
                    s.push_str("\t\t}\n\t}\n");
                    let _ = write!(conns, "\tC: \"OO\",{node},{layer}\n\tC: \"OP\",{node},{target},\"{prop}\"\n");
                    for (d, c) in curves {
                        let cid = next;
                        next += 1;
                        let kt: Vec<String> = c.keys.iter().map(|k| ticks(k.0).to_string()).collect();
                        let kv: Vec<String> = c.keys.iter().map(|k| k.1.to_string()).collect();
                        let _ = write!(
                            s,
                            "\tAnimationCurve: {cid}, \"AnimCurve::\", \"\" {{\n\t\tDefault: 0\n\t\tKeyVer: 4009\n\t\tKeyTime: *{n} {{\n\t\t\ta: {}\n\t\t}}\n\t\tKeyValueFloat: *{n} {{\n\t\t\ta: {}\n\t\t}}\n\t\tKeyAttrFlags: *1 {{\n\t\t\ta: {}\n\t\t}}\n\t\tKeyAttrDataFloat: *4 {{\n\t\t\ta: 0,0,0,0\n\t\t}}\n\t\tKeyAttrRefCount: *1 {{\n\t\t\ta: {n}\n\t\t}}\n\t}}\n",
                            kt.join(","),
                            kv.join(","),
                            c.flags,
                            n = c.keys.len()
                        );
                        let _ = writeln!(conns, "\tC: \"OP\",{cid},{node},\"{d}\"");
                        let _ = c.id;
                    }
                };
                let dims = ["d|X", "d|Y", "d|Z"];
                if take.rot.iter().any(Option::is_some) {
                    let curves: Vec<(&str, Curve)> = take
                        .rot
                        .iter()
                        .enumerate()
                        .filter_map(|(i, k)| k.as_ref().map(|k| (dims[i], Curve { id: 0, keys: k.clone(), flags: 8456 })))
                        .collect();
                    emit(&mut s, &mut conns, "R", 150, "Lcl Rotation", curves);
                }
                if let Some(tx) = &take.tx {
                    emit(&mut s, &mut conns, "T", 150, "Lcl Translation", vec![("d|X", Curve { id: 0, keys: tx.clone(), flags: 4 })]);
                }
                if let (Some((keys, flags)), Some(&ch)) = (&take.blend, channel_ids.first()) {
                    emit(&mut s, &mut conns, "DeformPercent", ch, "DeformPercent", vec![("d|DeformPercent", Curve { id: 0, keys: keys.clone(), flags: *flags })]);
                }
            }
            s += if self.skinless {
                "}\nConnections:  {\n\tC: \"OO\",150,0\n\tC: \"OO\",151,150\n\tC: \"OO\",160,150\n\tC: \"OO\",161,151\n"
            } else {
                "}\nConnections:  {\n\tC: \"OO\",200,0\n\tC: \"OO\",150,0\n\tC: \"OO\",100,200\n\tC: \"OO\",300,100\n\tC: \"OO\",301,300\n\tC: \"OO\",150,301\n"
            };
            s += &conns;
            s += "}\n";
            s
        }

        fn write(&self, tag: &str) -> std::path::PathBuf {
            let dir = std::env::temp_dir().join(format!("zest_fbx_anim_{tag}_{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("model.fbx");
            std::fs::write(&path, self.render()).unwrap();
            path
        }
    }

    fn import(f: &Fbx, tag: &str, settings: &ModelImport) -> crate::Scene {
        let path = f.write(tag);
        let scene = crate::load_fbx_with(&path, settings).unwrap_or_else(|e| panic!("{e:#}"));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        scene
    }

    fn off() -> ModelImport {
        let mut m = ModelImport::default();
        m.animation.compression = anim_import::AnimCompression::Off;
        m
    }

    fn take(name: &str, span: (f64, f64)) -> Take {
        Take { name: name.into(), span, rot: [None, None, None], tx: None, blend: None }
    }

    #[test]
    fn every_take_becomes_a_named_clip_with_its_range() {
        let mut f = Fbx::new();
        let mut a = take("Walk", (0.0, 1.0));
        a.tx = Some(vec![(0.0, 0.0), (1.0, 3.0)]);
        let mut b = take("Idle", (1.0, 3.0));
        b.tx = Some(vec![(1.0, 0.0), (3.0, 4.0)]);
        f.takes = vec![a, b];
        let scene = import(&f, "takes", &off());
        let names: Vec<&str> = scene.animations.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["Walk", "Idle"]);
        assert!((scene.animations[0].duration - 1.0).abs() < 1e-4 && (scene.animations[1].duration - 2.0).abs() < 1e-4);
        assert_eq!(scene.identity.takes.len(), 2);
        assert_eq!((scene.identity.takes[1].first_frame, scene.identity.takes[1].last_frame), (30.0, 90.0));
        assert_eq!(scene.identity.clip_info.len(), 2);
        assert_eq!(scene.identity.clip_info[1].take, "Idle");
        // The Idle take's own time starts at 1 s: the clip must start at its first key.
        let tr = scene.animations[1].channels.iter().find(|c| c.path == ChannelPath::Translation).unwrap();
        assert!(tr.vec_values[0].x.abs() < 1e-4 && (tr.vec_values.last().unwrap().x - 4.0).abs() < 1e-3);
    }

    #[test]
    fn the_clip_list_splits_a_take_by_frame_range() {
        let mut f = Fbx::new();
        let mut a = take("Take", (0.0, 1.0));
        a.tx = Some(vec![(0.0, 0.0), (1.0, 30.0)]);
        f.takes = vec![a];
        let mut m = off();
        m.animation.clips = vec![
            anim_import::ClipDef { name: "First".into(), take: "Take".into(), first_frame: 0.0, last_frame: 15.0, loop_time: true, ..Default::default() },
            anim_import::ClipDef { name: "Second".into(), take: "Take".into(), first_frame: 15.0, last_frame: 30.0, ..Default::default() },
            anim_import::ClipDef { name: "Lost".into(), take: "Missing".into(), first_frame: 0.0, last_frame: 5.0, ..Default::default() },
        ];
        let scene = import(&f, "split", &m);
        assert_eq!(scene.animations.len(), 2, "a clip of a missing take is ignored");
        assert!((scene.animations[1].duration - 0.5).abs() < 1e-4);
        let x = |c: usize, i: usize| scene.animations[c].channels.iter().find(|ch| ch.path == ChannelPath::Translation).unwrap().vec_values[i].x;
        assert!(x(0, 0).abs() < 1e-3 && (x(1, 0) - 15.0).abs() < 1e-3, "{} {}", x(0, 0), x(1, 0));
        assert!(scene.identity.clip_info[0].loop_time && !scene.identity.clip_info[1].loop_time);
        assert_eq!((scene.identity.clip_info[1].first_frame, scene.identity.clip_info[1].last_frame), (15.0, 30.0));
        // Unreferenced takes only come along when asked for.
        f.takes.push(take("Other", (0.0, 1.0)));
        m.animation.import_unreferenced_takes = true;
        let all = import(&f, "split2", &m);
        assert_eq!(all.animations.len(), 3);
    }

    #[test]
    fn keyframe_reduction_thins_dense_samples_and_respects_off() {
        let mut f = Fbx::new();
        let mut a = take("Take", (0.0, 2.0));
        a.tx = Some(vec![(0.0, 0.0), (2.0, 10.0)]);
        f.takes = vec![a];
        let dense = import(&f, "dense", &off());
        let thin = import(&f, "thin", &ModelImport::default());
        let n = |s: &crate::Scene| s.animations[0].channels.iter().find(|c| c.path == ChannelPath::Translation).unwrap().times.len();
        assert!(n(&dense) >= 60, "{}", n(&dense));
        assert_eq!(n(&thin), 2, "a straight line reduces to its two ends");
    }

    #[test]
    fn rotation_follows_prerotation_and_order() {
        // FBX: local rotation = Rpre * R(euler, order); order ZYX (enum 5) applies Z, then Y, then X
        // ... in FBX terms R = Rx * Ry * Rz for "ZYX".
        let mut f = Fbx::new();
        f.pre_rotation = [10.0, 20.0, 30.0];
        f.rotation_order = 5;
        let mut a = take("Take", (0.0, 1.0));
        a.rot = [Some(vec![(0.0, 45.0), (1.0, 45.0)]), Some(vec![(0.0, 30.0), (1.0, 30.0)]), Some(vec![(0.0, 60.0), (1.0, 60.0)])];
        f.takes = vec![a];
        let scene = import(&f, "rot", &off());
        let ch = scene.animations[0].channels.iter().find(|c| c.joint == 0 && c.path == ChannelPath::Rotation).unwrap();
        let rx = |d: f32, axis: Vec3| Quat::from_axis_angle(axis, d.to_radians());
        let pre = rx(30.0, Vec3::Z) * rx(20.0, Vec3::Y) * rx(10.0, Vec3::X); // PreRotation is XYZ
        let euler = rx(45.0, Vec3::X) * rx(30.0, Vec3::Y) * rx(60.0, Vec3::Z); // order ZYX
        let want = pre * euler;
        // The importer converts axes with the root; compare the joint relative to the rest rotation instead.
        let got = ch.quat_values[0];
        let rest = scene.skeletons[0].joints[0].rest_rotation;
        let _ = rest;
        // A root-level axis conversion is identity for Y-up right-handed, so the match is direct.
        assert!(got.dot(want).abs() > 0.9999, "got {got:?} want {want:?}");
    }

    #[test]
    fn spinning_rotation_stays_continuous() {
        let mut f = Fbx::new();
        let mut a = take("Take", (0.0, 1.0));
        a.rot = [None, None, Some(vec![(0.0, 0.0), (1.0, 350.0)])];
        f.takes = vec![a];
        let scene = import(&f, "spin", &off());
        let ch = scene.animations[0].channels.iter().find(|c| c.path == ChannelPath::Rotation).unwrap();
        assert!(ch.quat_values.windows(2).all(|w| w[0].dot(w[1]) > 0.0), "no hemisphere flips between samples");
        // 350 degrees of Z is a rotation of -10 degrees: the last key is that, reached by the long way round.
        let mid = ch.quat_values[ch.quat_values.len() / 2];
        assert!((mid.to_axis_angle().1.to_degrees() - 175.0).abs() < 5.0 || (360.0 - mid.to_axis_angle().1.to_degrees() - 175.0).abs() < 5.0);
    }

    #[test]
    fn blendshape_percent_becomes_a_weight_track() {
        let mut f = Fbx::new();
        f.channels = vec![("Smile".into(), vec![100.0], 4)];
        let mut a = take("Take", (0.0, 1.0));
        a.blend = Some((vec![(0.0, 0.0), (0.5, 100.0), (1.0, 20.0)], 4)); // linear
        f.takes = vec![a];
        let scene = import(&f, "blend", &off());
        let info = &scene.identity.clip_info[0];
        assert_eq!(info.morph_tracks.len(), 1);
        let t = &info.morph_tracks[0];
        assert_eq!((t.target.as_str(), t.name.as_str(), t.property), ("Body", "Smile", TrackProperty::BlendShape));
        let v: Vec<f32> = t.keys.iter().map(|k| k.value[0]).collect();
        assert_eq!(v.len(), 3, "{v:?}");
        assert!((v[1] - 1.0).abs() < 1e-4 && (v[2] - 0.2).abs() < 1e-4, "percent 100 is weight 1: {v:?}");
        assert!((t.sample(0.25)[0] - 0.5).abs() < 1e-3);
        assert_eq!(scene.morphs.iter().flatten().filter(|m| m.name == "Smile").count(), 1);
    }

    #[test]
    fn cubic_blend_curves_keep_their_shape() {
        let mut f = Fbx::new();
        f.channels = vec![("Smile".into(), vec![100.0], 4)];
        let mut a = take("Take", (0.0, 2.0));
        a.blend = Some((vec![(0.0, 0.0), (0.5, 80.0), (1.2, 10.0), (2.0, 60.0)], 8456)); // cubic, auto tangents
        f.takes = vec![a];
        let path = f.write("cubic");
        let fbx = ufbx::load_file(path.to_str().unwrap(), ufbx::LoadOpts::default()).unwrap();
        let sources = morph_sources(&fbx);
        let stack = &fbx.anim_stacks[0];
        let track = morph_track(stack, &sources[0], stack.time_begin, stack.time_end).unwrap();
        for k in 0..=40 {
            let t = k as f64 * 0.05;
            let want = sources[0].channel.evaluate_blend_weight(stack.anim.as_ref(), t) as f32;
            let got = track.sample(t as f32)[0];
            assert!((got - want).abs() < 0.02, "t={t}: engine {got} vs source {want}");
        }
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn in_between_shapes_are_imported_as_frames() {
        let mut f = Fbx::new();
        f.channels = vec![("Mouth".into(), vec![50.0, 100.0], 4)];
        f.takes = vec![take("T", (0.0, 1.0))];
        let scene = import(&f, "inbetween", &off());
        let names: Vec<&str> = scene.morphs.iter().flatten().map(|m| m.name.as_str()).collect();
        assert!(names.contains(&"Mouth") && names.contains(&"Mouth@50"), "{names:?}");
        let targets = &scene.morphs[0];
        let w = anim_import::inbetween_weights(targets, &|n| if n == "Mouth" { 0.75 } else { 0.0 });
        let by = |n: &str| w[targets.iter().position(|t| t.name == n).unwrap()];
        assert!((by("Mouth") - 0.5).abs() < 1e-5 && (by("Mouth@50") - 0.5).abs() < 1e-5);
    }

    #[test]
    fn empty_shapes_are_dropped_and_blendshapes_can_be_switched_off() {
        let mut f = Fbx::new();
        f.channels = vec![("Real".into(), vec![100.0], 4), ("Empty".into(), vec![100.0], 0)];
        f.takes = vec![take("T", (0.0, 1.0))];
        let scene = import(&f, "empty", &off());
        let names: Vec<&str> = scene.morphs.iter().flatten().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["Real"]);
        let mut none = off();
        none.import_blend_shapes = false;
        assert!(import(&f, "noshapes", &none).morphs.iter().all(Vec::is_empty));
    }

    #[test]
    fn many_shapes_stay_sparse_and_import_fast() {
        let mut f = Fbx::new();
        f.grid = 40; // 1681 vertices
        f.channels = (0..800).map(|i| (format!("Shape{i}"), vec![100.0], 12)).collect();
        f.takes = vec![take("T", (0.0, 1.0))];
        let started = std::time::Instant::now();
        let scene = import(&f, "many", &off());
        let took = started.elapsed();
        let total: usize = scene.morphs.iter().flatten().map(|m| m.deltas.len()).sum();
        assert_eq!(scene.morphs.iter().flatten().count(), 800);
        // 12 control points per shape, each on at most a handful of corners.
        assert!(total < 800 * 12 * 8, "sparse storage: {total} deltas");
        assert!(took.as_secs_f32() < 20.0, "800 shapes took {took:?}");
    }

    #[test]
    fn a_frame_rate_other_than_30_keeps_authored_timing() {
        let mut f = Fbx::new();
        f.fps_mode = 11; // custom/24 in the writer's table: 24 fps
        f.takes = vec![{
            let mut a = take("T", (0.0, 1.0));
            a.tx = Some(vec![(0.0, 0.0), (1.0, 1.0)]);
            a
        }];
        let scene = import(&f, "fps", &off());
        let info = &scene.identity.clip_info[0];
        assert!(info.frame_rate > 0.0 && (info.last_frame / info.frame_rate - 1.0).abs() < 0.05);
    }

    #[test]
    fn real_blender_take_imports_and_reduces() {
        // Blender export: 24 fps, the take starts at frame 1 (not 0).
        let path = std::path::Path::new("/home/lemonxah/storage/blender/pond_unity.fbx");
        if !path.exists() {
            return;
        }
        let thin = crate::load_fbx_with(path, &ModelImport::default()).unwrap();
        let dense = crate::load_fbx_with(path, &off()).unwrap();
        assert_eq!(dense.animations.len(), 1);
        assert_eq!(dense.animations[0].name, "Scene");
        assert_eq!(dense.identity.takes[0].first_frame, 1.0);
        let keys = |s: &crate::Scene| s.animations[0].channels.iter().map(|c| c.times.len()).sum::<usize>();
        assert!(keys(&thin) < keys(&dense), "{} vs {}", keys(&thin), keys(&dense));
        assert!(dense.animations[0].channels.iter().filter(|c| c.path == ChannelPath::Rotation).all(|c| c.quat_values.windows(2).all(|w| w[0].dot(w[1]) >= 0.0)));
    }

    #[test]
    fn real_blender_face_has_hundreds_of_sparse_shapes() {
        let path = std::path::Path::new("/home/lemonxah/storage/blender/lemonbase.fbx");
        if !path.exists() {
            return;
        }
        let scene = crate::load_fbx_with(path, &ModelImport::default()).unwrap();
        let shapes: usize = scene.morphs.iter().map(Vec::len).sum();
        assert!(shapes > 300, "{shapes}");
    }

    #[test]
    fn a_file_without_a_mesh_imports_as_a_clip_library_in_metres() {
        let mut f = Fbx::new();
        f.skinless = true;
        f.pre_rotation = [0.0, 0.0, 90.0];
        let mut a = take("mixamo.com", (0.0, 1.0));
        a.rot = [None, None, Some(vec![(0.0, 0.0), (1.0, 90.0)])];
        f.takes = vec![a];
        let scene = import(&f, "skinless", &off());
        assert!(scene.instances.is_empty() && scene.meshes.is_empty());
        let skel = &scene.skeletons[0];
        assert_eq!(skel.joints.iter().map(|j| j.name.as_str()).collect::<Vec<_>>(), ["Root", "Child"]);
        // FBX centimetres become metres: 100 cm -> 1 m, 10 cm -> 0.1 m.
        assert!((skel.joints[0].rest_translation - Vec3::Y).length() < 1e-4, "{:?}", skel.joints[0].rest_translation);
        assert!((skel.joints[1].rest_translation - Vec3::Y * 0.1).length() < 1e-4);
        assert_eq!(skel.joints[1].parent, Some(0));
        // The inverse bind is the rest world matrix inverse.
        let world_child = skel.world_matrices(&skel.rest_locals())[1];
        assert!((skel.joints[1].inverse_bind * world_child).abs_diff_eq(glam::Mat4::IDENTITY, 1e-4));
        // PreRotation (90 degrees about Z) composes with the animated Lcl Z: the end is 180 degrees.
        let clip = &scene.animations[0];
        assert_eq!(clip.name, "mixamo.com");
        let rot = clip.channels.iter().find(|c| c.joint == 0 && c.path == ChannelPath::Rotation).unwrap();
        let (first, last) = (rot.quat_values[0], *rot.quat_values.last().unwrap());
        assert!(first.abs_diff_eq(Quat::from_rotation_z(90f32.to_radians()), 1e-3) || first.abs_diff_eq(-Quat::from_rotation_z(90f32.to_radians()), 1e-3), "{first:?}");
        assert!(last.abs_diff_eq(Quat::from_rotation_z(180f32.to_radians()), 1e-3) || last.abs_diff_eq(-Quat::from_rotation_z(180f32.to_radians()), 1e-3), "{last:?}");
        let tr = clip.channels.iter().find(|c| c.joint == 0 && c.path == ChannelPath::Translation).unwrap();
        assert!((tr.vec_values[0] - Vec3::Y).length() < 1e-4, "clip translation is in metres too");
    }

    #[test]
    fn a_file_with_neither_mesh_nor_animation_is_still_refused() {
        let mut f = Fbx::new();
        f.skinless = true;
        let path = f.write("empty_skinless");
        let err = crate::load_fbx_with(&path, &off()).err().expect("nothing to import");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
        assert!(format!("{err:#}").contains("no renderable meshes"), "{err:#}");
    }

    #[test]
    fn deform_percent_becomes_an_initial_weight_unless_switched_off() {
        let mut f = Fbx::new();
        f.channels = vec![("Smile".into(), vec![100.0], 4), ("Frown".into(), vec![100.0], 4)];
        f.deform_percent = 40.0;
        let scene = import(&f, "deform", &off());
        let w: Vec<(&str, f32)> = scene.identity.morph_weights.iter().map(|(_, n, w)| (n.as_str(), *w)).collect();
        assert_eq!(w.len(), 2, "{w:?}");
        assert!(w.iter().all(|(_, v)| (v - 0.4).abs() < 1e-5), "percent 40 is weight 0.4: {w:?}");
        let mut m = off();
        m.blend_shapes.import_deform_percent = false;
        assert!(import(&f, "deform_off", &m).identity.morph_weights.is_empty());
    }

    #[test]
    fn listed_clips_carry_playback_settings_and_default_ones_do_not() {
        let mut f = Fbx::new();
        let mut a = take("Take", (0.0, 1.0));
        a.tx = Some(vec![(0.0, 0.0), (1.0, 30.0)]);
        f.takes = vec![a];
        let plain = import(&f, "pb_plain", &off());
        assert!(plain.animations[0].playback.is_none() && plain.identity.clip_info[0].playback.is_none());
        let mut m = off();
        m.animation.clips = vec![anim_import::ClipDef {
            name: "Run".into(),
            take: "Take".into(),
            first_frame: 0.0,
            last_frame: 30.0,
            loop_time: true,
            cycle_offset: 0.25,
            lock_root_height_y: true,
            mirror: true,
            events: vec![anim_import::ClipEventDef { time: 0.5, name: "Step".into(), float: 1.5, string: "left".into(), ..Default::default() }],
            ..Default::default()
        }];
        let scene = import(&f, "pb_listed", &m);
        let pb = scene.animations[0].playback.as_deref().expect("a listed clip has playback settings");
        assert!(pb.loop_time && pb.bake_y && pb.mirror && !pb.bake_xz);
        assert_eq!(pb.cycle_offset, 0.25);
        assert_eq!((pb.events[0].name.as_str(), pb.events[0].float, pb.events[0].string.as_str()), ("Step", 1.5, "left"));
        assert_eq!(scene.identity.clip_info[0].playback.as_ref(), Some(pb));
    }
}
