//! Animation and blendshape import settings of a model (the Animation tab and
//! the Import BlendShapes options of the model inspector), what the importer
//! records about each clip it made, and the format independent passes that run
//! on imported curves: keyframe reduction, loop-pose correction and
//! in-between blendshape resolution.
//!
//! Everything here is `#[serde(default)]`, so `.meta` files written before a
//! field existed load with the defaults.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::anim_clip::AnimTrack;
use crate::skeleton::{AnimationClip, ChannelPath};
#[cfg(test)]
use crate::skeleton::AnimChannel;
use crate::MorphTarget;

/// How the importer thins the sampled keys of a clip.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnimCompression {
    /// Keep every sample.
    Off,
    /// Drop keys that a straight line between their neighbours reproduces within
    /// the rotation / position / scale error.
    #[default]
    KeyframeReduction,
    /// Keyframe reduction, then every curve stored as 16-bit quantised values
    /// over its own range (see [`crate::anim_pack`]); a curve whose quantised
    /// form would break the error tolerance stays in full precision.
    KeyframeReductionAndCompression,
    /// Per curve, the smallest storage that keeps the error tolerances:
    /// constant, reduced keys, dense quantised, or reduced and quantised.
    Optimal,
}

impl AnimCompression {
    /// Whether keys are dropped at all.
    pub fn reduces(self) -> bool {
        self != Self::Off
    }

    /// Whether curves may be stored quantised.
    pub fn quantizes(self) -> bool {
        matches!(self, Self::KeyframeReductionAndCompression | Self::Optimal)
    }
}

/// Where a blendshape's normals come from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlendNormals {
    /// The file's normal offsets.
    Import,
    /// Recomputed from the displaced mesh (area weighted), relative to the
    /// recomputed base normals.
    #[default]
    Calculate,
    /// No normal offsets: the normal stays the base mesh's.
    None,
}

/// Import BlendShapes options.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct BlendShapeImport {
    /// Normals of the shapes (Unity "Blend Shape Normals"; default Calculate).
    pub normals: BlendNormals,
    /// Offsets (position and normal) at or below this length are dropped, and a
    /// shape left without any is dropped altogether (keeps 800-shape face meshes
    /// small). 0 keeps everything except exact zeros.
    pub delta_threshold: f32,
    /// Import the channels' animation curves (`DeformPercent`) as blendshape
    /// tracks of the clips.
    pub import_curves: bool,
    /// Initial weights from the file's `DeformPercent` (Unity "Import Deform
    /// Percent", default on); off leaves every shape at 0.
    pub import_deform_percent: bool,
    /// Legacy Blend Shape Normals (Unity, default off): the old importer's
    /// smoothing-group normals. Here it makes the normals come from the displaced
    /// geometry whatever [`Self::normals`] says (Import and None are ignored).
    pub legacy_normals: bool,
}

impl Default for BlendShapeImport {
    fn default() -> Self {
        Self { normals: BlendNormals::Calculate, delta_threshold: 1e-6, import_curves: true, import_deform_percent: true, legacy_normals: false }
    }
}

/// An event on a clip (at `time` seconds from the clip start).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ClipEventDef {
    pub time: f32,
    /// The function name the event calls.
    pub name: String,
    /// The parameters it carries (Unity `AnimationEvent`: float, int, string,
    /// object); unused ones stay at their default.
    pub float: f32,
    pub int: i32,
    pub string: String,
    /// An asset or object reference, by project path or name.
    pub object: String,
}

/// One clip of the model's clip list (Unity `ModelImporterClipAnimation`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ClipDef {
    pub name: String,
    /// The take (FBX animation stack) it is cut from; empty = the first take.
    pub take: String,
    /// Frame range in the take's own frame numbers (inclusive of both ends).
    pub first_frame: f32,
    pub last_frame: f32,
    /// Loop Time.
    pub loop_time: bool,
    /// Loop Pose: blend the end pose into the start pose so the loop has no pop.
    pub loop_pose: bool,
    /// Normalised start offset of the loop (the player starts every pass this far in).
    pub cycle_offset: f32,
    /// Root Transform Rotation / Position (Y) / Position (XZ): bake into pose.
    pub lock_root_rotation: bool,
    pub lock_root_height_y: bool,
    pub lock_root_position_xz: bool,
    /// Based Upon (at Start / Original) toggles of the bake options.
    pub keep_original_orientation: bool,
    pub keep_original_position_y: bool,
    pub keep_original_position_xz: bool,
    /// Rotation offset (degrees) and height offset of the root.
    pub root_rotation_offset: f32,
    pub height_offset: f32,
    /// Mirror the clip (humanoid).
    pub mirror: bool,
    pub events: Vec<ClipEventDef>,
}

impl Default for ClipDef {
    fn default() -> Self {
        Self {
            name: String::new(),
            take: String::new(),
            first_frame: 0.0,
            last_frame: 0.0,
            loop_time: false,
            loop_pose: false,
            cycle_offset: 0.0,
            lock_root_rotation: false,
            lock_root_height_y: false,
            lock_root_position_xz: false,
            keep_original_orientation: false,
            keep_original_position_y: true,
            keep_original_position_xz: false,
            root_rotation_offset: 0.0,
            height_offset: 0.0,
            mirror: false,
            events: Vec::new(),
        }
    }
}

/// The Animation tab of the model importer.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AnimImport {
    /// Import Animation: off imports no clips at all.
    pub import_animation: bool,
    /// Resample Curves: sample every source frame (on) or only at the authored
    /// key times (off; clip ends are always sampled).
    pub resample_curves: bool,
    pub compression: AnimCompression,
    /// Max rotation deviation in degrees.
    pub rotation_error: f32,
    /// Max position / scale deviation in percent (of the curve's range).
    pub position_error: f32,
    pub scale_error: f32,
    /// Drop scale channels that never leave the rest scale.
    pub remove_constant_scale_curves: bool,
    /// Also import a default clip for each take no listed clip uses (only
    /// matters once the clip list is non-empty).
    pub import_unreferenced_takes: bool,
    /// The clip list. Empty = one clip per take.
    pub clips: Vec<ClipDef>,
    /// Bake Animations: evaluate the file's whole node transform stack (pivots,
    /// offsets, inherited scale) and its constraints per frame and store plain
    /// local TRS curves, so what the DCC showed is what plays. FBX only.
    pub bake_animations: bool,
    /// Keep Tangents: keep the authored key times and their curve shape
    /// (Hermite tangents) instead of sampling every frame. Overrides Resample
    /// Curves; the curves match the source within the error tolerances.
    pub keep_tangents: bool,
    /// Animated Custom Properties: import FBX user properties that carry
    /// animation curves as property tracks onto a Custom Properties component.
    pub animated_custom_properties: bool,
    /// Humanoid Oversampling (1, 2, 4 or 8): the muscle bake of a humanoid
    /// clip samples at this multiple of the 30 Hz base rate.
    pub humanoid_oversampling: u8,
}

impl Default for AnimImport {
    fn default() -> Self {
        Self {
            import_animation: true,
            resample_curves: true,
            compression: AnimCompression::KeyframeReduction,
            rotation_error: 0.5,
            position_error: 0.5,
            scale_error: 0.5,
            remove_constant_scale_curves: false,
            import_unreferenced_takes: false,
            clips: Vec::new(),
            bake_animations: false,
            keep_tangents: false,
            animated_custom_properties: false,
            humanoid_oversampling: 1,
        }
    }
}

impl AnimImport {
    /// Clamp the tolerances and repair the clip list.
    pub fn validate(&mut self) {
        self.humanoid_oversampling = match self.humanoid_oversampling {
            0 | 1 => 1,
            2 | 3 => 2,
            4..=7 => 4,
            _ => 8,
        };
        for e in [&mut self.rotation_error, &mut self.position_error, &mut self.scale_error] {
            *e = if e.is_finite() { e.clamp(0.0, 100.0) } else { 0.5 };
        }
        for c in &mut self.clips {
            for f in [&mut c.first_frame, &mut c.last_frame] {
                if !f.is_finite() {
                    *f = 0.0;
                }
            }
            if c.last_frame < c.first_frame {
                std::mem::swap(&mut c.first_frame, &mut c.last_frame);
            }
            c.cycle_offset = if c.cycle_offset.is_finite() { c.cycle_offset.rem_euclid(1.0) } else { 0.0 };
            c.events.retain(|e| e.time.is_finite());
        }
    }
}

impl AnimImport {
    /// Turn the implicit one-clip-per-take list into an editable one (Unity
    /// `TransferDefaultClipsToCustomClips`). A no-op when clips already exist.
    pub fn customize_from_takes(&mut self, takes: &[TakeInfo]) {
        if !self.clips.is_empty() {
            return;
        }
        self.clips = takes
            .iter()
            .map(|t| ClipDef { name: t.name.clone(), take: t.name.clone(), first_frame: t.first_frame, last_frame: t.last_frame, ..Default::default() })
            .collect();
    }

    /// Add a clip on the same take as the last one (or the first take), over
    /// the whole take, with a name no other clip has.
    pub fn add_clip(&mut self, takes: &[TakeInfo]) {
        let take = self.clips.last().map(|c| c.take.clone()).or_else(|| takes.first().map(|t| t.name.clone())).unwrap_or_default();
        let (first, last) = takes.iter().find(|t| t.name == take).map(|t| (t.first_frame, t.last_frame)).unwrap_or((0.0, 30.0));
        let mut name = String::from("New Clip");
        let mut n = 1;
        while self.clips.iter().any(|c| c.name == name) {
            n += 1;
            name = format!("New Clip {n}");
        }
        self.clips.push(ClipDef { name, take, first_frame: first, last_frame: last, ..Default::default() });
    }
}

/// A take (FBX animation stack) the file carries.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TakeInfo {
    pub name: String,
    /// Seconds.
    pub start: f32,
    pub stop: f32,
    /// The file's frame rate.
    pub frame_rate: f32,
    /// `start` / `stop` in frames (what a clip's frame range is written in).
    pub first_frame: f32,
    pub last_frame: f32,
}

/// What the importer knows about an imported clip beyond its skeletal channels
/// (parallel to `Scene::animations`; a missing entry = plain clip).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ClipImportInfo {
    /// The take it was cut from and its frame range there.
    pub take: String,
    pub first_frame: f32,
    pub last_frame: f32,
    pub frame_rate: f32,
    pub loop_time: bool,
    pub loop_pose: bool,
    /// How far the end pose is from the start pose (degrees of the worst joint
    /// rotation, then the worst translation as a fraction of its range): small
    /// values mean the clip loops cleanly.
    pub loop_error_deg: f32,
    pub loop_error_pos: f32,
    /// Bake-into-pose flags, root offsets and cycle offset as authored.
    pub lock_root_rotation: bool,
    pub lock_root_height_y: bool,
    pub lock_root_position_xz: bool,
    pub cycle_offset: f32,
    pub events: Vec<ClipEventDef>,
    /// What the player reads of the clip's settings; `None` for a clip cut by
    /// default (one per take), which plays as the Animator says.
    pub playback: Option<ClipPlayback>,
    /// Blendshape weight curves: `target` = the mesh node, `name` = the shape,
    /// key values are weights (1 = the shape's full target). Sparse: shapes the
    /// clip does not animate have no track, constant tracks are one key.
    pub morph_tracks: Vec<AnimTrack>,
}

/// What the player needs of a listed clip's import settings, attached to the
/// [`AnimationClip`] (it travels with the clip through reimport and the cache).
/// Only clips of the model's clip list have one: a clip cut by default plays
/// exactly as before.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ClipPlayback {
    /// Loop Time: the clip wraps past its end; off clamps on the last pose.
    pub loop_time: bool,
    /// Normalised start offset of every pass.
    pub cycle_offset: f32,
    /// Bake Into Pose: the root's yaw / height / ground-plane travel stay in
    /// the pose; what is not baked drives root motion instead.
    pub bake_rotation: bool,
    pub bake_y: bool,
    pub bake_xz: bool,
    /// Based Upon: `true` = the authored value ("Original"); `false` = the
    /// body orientation (rotation), the feet (height), the centre of mass (XZ).
    pub keep_orientation: bool,
    pub keep_y: bool,
    pub keep_xz: bool,
    /// Rotation offset (degrees) and height offset added to the root.
    pub rotation_offset: f32,
    pub height_offset: f32,
    /// Mirror the clip (a humanoid rig mirrors; others ignore it).
    pub mirror: bool,
    pub events: Vec<ClipEventDef>,
    /// Measured on the clip at import, in skeleton space: the mean yaw of the
    /// root (radians), its mean ground-plane position, and the height of the
    /// lowest foot at the first frame. They are the "based upon" references.
    pub body_yaw: f32,
    pub center_xz: [f32; 2],
    pub feet_y: f32,
}

impl ClipPlayback {
    /// The playback settings of a listed clip, with the references measured on `clip`.
    pub fn from_def(def: &ClipDef, skeleton: &crate::skeleton::Skeleton, clip: &AnimationClip) -> Self {
        let mut pb = Self {
            loop_time: def.loop_time,
            cycle_offset: def.cycle_offset,
            bake_rotation: def.lock_root_rotation,
            bake_y: def.lock_root_height_y,
            bake_xz: def.lock_root_position_xz,
            keep_orientation: def.keep_original_orientation,
            keep_y: def.keep_original_position_y,
            keep_xz: def.keep_original_position_xz,
            rotation_offset: def.root_rotation_offset,
            height_offset: def.height_offset,
            mirror: def.mirror,
            events: def.events.clone(),
            ..Default::default()
        };
        pb.measure(skeleton, clip);
        pb
    }

    /// Fill the "based upon" references from the clip's root and feet.
    fn measure(&mut self, skeleton: &crate::skeleton::Skeleton, clip: &AnimationClip) {
        let Some(root) = skeleton.motion_root() else { return };
        let samples = 32usize;
        let at = |k: usize| clip.duration * k as f32 / samples as f32;
        let (mut sin, mut cos, mut sum_x, mut sum_z) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for k in 0..=samples {
            let locals = clip.sample(skeleton, at(k).min((clip.duration - 1e-4).max(0.0)));
            let (_, r, t) = locals[root].to_scale_rotation_translation();
            let yaw = twist_angle_y(r);
            sin += yaw.sin();
            cos += yaw.cos();
            sum_x += t.x;
            sum_z += t.z;
        }
        let n = (samples + 1) as f32;
        self.body_yaw = if sin.abs() + cos.abs() > 1e-6 { sin.atan2(cos) } else { 0.0 };
        self.center_xz = [sum_x / n, sum_z / n];
        // Lowest foot at the first frame, in skeleton space.
        let locals = clip.sample(skeleton, 0.0);
        let world = skeleton.world_matrices(&locals);
        let feet = skeleton.joints.iter().enumerate().filter(|(_, j)| is_foot(&j.name)).map(|(i, _)| world[i].w_axis.y);
        self.feet_y = feet.fold(None, |m: Option<f32>, y| Some(m.map_or(y, |m| m.min(y)))).unwrap_or(0.0);
    }
}

/// Whether a bone name is a foot (not a toe): used for the Feet height reference.
fn is_foot(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.contains("foot") && !n.contains("toe") && !n.contains("end")
}

/// The rotation of `q` about +Y (radians, the twist of its swing-twist split).
pub fn twist_angle_y(q: Quat) -> f32 {
    2.0 * q.y.atan2(q.w)
}

// ---------------------------------------------------------------------------
// Keyframe reduction
// ---------------------------------------------------------------------------

/// Indices of `times` to keep so that dropping the rest changes no sample of
/// `err(i, j, k)` (the error of dropping `j` when the line runs `i -> k`) past
/// `tol`. Iterative (a greedy sweep from the first key), so a long dense curve
/// reduces in linear time. The first and last key are always kept.
fn reduce_indices(n: usize, tol: f32, err: impl Fn(usize, usize, usize) -> f32) -> Vec<usize> {
    if n <= 2 || tol <= 0.0 {
        return (0..n).collect();
    }
    let mut keep = vec![0usize];
    let mut anchor = 0usize;
    let mut j = 1usize;
    while j < n - 1 {
        // Extend the segment while every dropped key stays within tolerance.
        let next = j + 1;
        let ok = (anchor + 1..=j).all(|m| err(anchor, m, next) <= tol);
        if !ok {
            keep.push(j);
            anchor = j;
        }
        j += 1;
    }
    keep.push(n - 1);
    keep
}

fn lerp_at(times: &[f32], i: usize, k: usize, m: usize) -> f32 {
    let span = (times[k] - times[i]).max(1e-9);
    ((times[m] - times[i]) / span).clamp(0.0, 1.0)
}

/// Reduce a Vec3 channel; `percent` of the channel's value range is the
/// tolerance (a flat channel keeps only its ends).
pub fn reduce_vec3(times: &[f32], values: &[Vec3], percent: f32) -> (Vec<f32>, Vec<Vec3>) {
    let n = times.len().min(values.len());
    if n <= 2 {
        return (times.to_vec(), values.to_vec());
    }
    let (mut lo, mut hi) = (values[0], values[0]);
    for v in &values[..n] {
        lo = lo.min(*v);
        hi = hi.max(*v);
    }
    let range = (hi - lo).max_element();
    // A relative tolerance of an (almost) flat curve would be 0; keep a tiny absolute floor.
    let tol = (range * percent / 100.0).max(1e-7);
    let keep = reduce_indices(n, tol, |i, m, k| {
        let f = lerp_at(times, i, k, m);
        (values[i].lerp(values[k], f) - values[m]).abs().max_element()
    });
    (keep.iter().map(|&i| times[i]).collect(), keep.iter().map(|&i| values[i]).collect())
}

/// Reduce a rotation channel; `degrees` is the largest allowed angle error.
pub fn reduce_quat(times: &[f32], values: &[Quat], degrees: f32) -> (Vec<f32>, Vec<Quat>) {
    let n = times.len().min(values.len());
    if n <= 2 {
        return (times.to_vec(), values.to_vec());
    }
    let tol = degrees.to_radians().max(1e-6);
    let keep = reduce_indices(n, tol, |i, m, k| {
        let f = lerp_at(times, i, k, m);
        let a = values[i].slerp(values[k], f);
        crate::anim_pack::angle_between(a, values[m])
    });
    (keep.iter().map(|&i| times[i]).collect(), keep.iter().map(|&i| values[i]).collect())
}

/// Reduce a clip's channels in place with the importer's tolerances.
pub fn reduce_clip(clip: &mut AnimationClip, rotation_deg: f32, position_pct: f32, scale_pct: f32) {
    for ch in &mut clip.channels {
        match ch.path {
            ChannelPath::Rotation => {
                let (t, v) = reduce_quat(&ch.times, &ch.quat_values, rotation_deg);
                ch.times = t;
                ch.quat_values = v;
            }
            ChannelPath::Translation | ChannelPath::Scale => {
                let pct = if ch.path == ChannelPath::Translation { position_pct } else { scale_pct };
                let (t, v) = reduce_vec3(&ch.times, &ch.vec_values, pct);
                ch.times = t;
                ch.vec_values = v;
            }
        }
    }
}

/// Remove scale channels that stay at `rest` (per joint) within 1e-5.
pub fn remove_constant_scale(clip: &mut AnimationClip, rest: impl Fn(usize) -> Vec3) {
    clip.channels.retain(|ch| {
        ch.path != ChannelPath::Scale || ch.vec_values.iter().any(|v| (*v - rest(ch.joint)).abs().max_element() > 1e-5)
    });
}

/// Euler-style continuity for sampled rotations: flips the sign of each key so
/// neighbouring quaternions are in the same hemisphere (no 360 degree spins
/// when interpolating). Returns how many keys were flipped.
pub fn make_continuous(values: &mut [Quat]) -> usize {
    let mut flips = 0;
    for i in 1..values.len() {
        if values[i - 1].dot(values[i]) < 0.0 {
            values[i] = -values[i];
            flips += 1;
        }
    }
    flips
}

// ---------------------------------------------------------------------------
// Loop pose
// ---------------------------------------------------------------------------

/// Worst start-to-end difference of a clip's channels: (degrees, fraction of
/// translation range).
pub fn loop_error(clip: &AnimationClip) -> (f32, f32) {
    let (mut deg, mut pos) = (0.0f32, 0.0f32);
    for ch in &clip.channels {
        match ch.path {
            ChannelPath::Rotation => {
                if let (Some(a), Some(b)) = (ch.quat_values.first(), ch.quat_values.last()) {
                    deg = deg.max(2.0 * a.dot(*b).abs().clamp(0.0, 1.0).acos().to_degrees());
                }
            }
            ChannelPath::Translation => {
                if let (Some(a), Some(b)) = (ch.vec_values.first(), ch.vec_values.last()) {
                    let (mut lo, mut hi) = (*a, *a);
                    for v in &ch.vec_values {
                        lo = lo.min(*v);
                        hi = hi.max(*v);
                    }
                    let range = (hi - lo).max_element().max(1e-6);
                    pos = pos.max((*a - *b).abs().max_element() / range);
                }
            }
            ChannelPath::Scale => {}
        }
    }
    (deg, pos)
}

/// Blend the end pose into the start pose across the clip (Loop Pose): every
/// rotation is corrected by a growing share of the end-to-start rotation,
/// scales and translations by a growing share of the end-to-start offset. The
/// `skip_translation` joint (the root, whose travel is the root motion) keeps
/// its translation.
pub fn apply_loop_pose(clip: &mut AnimationClip, skip_translation: Option<usize>) {
    for ch in &mut clip.channels {
        let n = ch.times.len();
        if n < 2 {
            continue;
        }
        let (t0, t1) = (ch.times[0], ch.times[n - 1]);
        let span = (t1 - t0).max(1e-9);
        match ch.path {
            ChannelPath::Rotation => {
                let delta = (ch.quat_values[0] * ch.quat_values[n - 1].inverse()).normalize();
                for (t, q) in ch.times.iter().zip(&mut ch.quat_values) {
                    let f = ((*t - t0) / span).clamp(0.0, 1.0);
                    *q = (Quat::IDENTITY.slerp(delta, f) * *q).normalize();
                }
            }
            ChannelPath::Translation | ChannelPath::Scale => {
                if ch.path == ChannelPath::Translation && skip_translation == Some(ch.joint) {
                    continue;
                }
                let delta = ch.vec_values[0] - ch.vec_values[n - 1];
                for (t, v) in ch.times.iter().zip(&mut ch.vec_values) {
                    *v += delta * ((*t - t0) / span).clamp(0.0, 1.0);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// In-between blendshapes
// ---------------------------------------------------------------------------

/// Separator of an in-between target's name: `Mouth@50` is the frame of shape
/// `Mouth` that is fully reached at weight 0.5 (the shape itself, `Mouth`, is
/// the frame at weight 1).
pub const INBETWEEN_SEP: char = '@';

/// `(base name, frame weight)` of an in-between target name.
pub fn split_inbetween(name: &str) -> Option<(&str, f32)> {
    let (base, tail) = name.rsplit_once(INBETWEEN_SEP)?;
    let percent: f32 = tail.parse().ok()?;
    (!base.is_empty() && percent > 0.0 && percent < 100.0).then_some((base, percent / 100.0))
}

/// The name of the in-between frame at `weight` (0..1) of `base`.
pub fn inbetween_name(base: &str, weight: f32) -> String {
    let p = format!("{:.3}", weight * 100.0);
    format!("{base}{INBETWEEN_SEP}{}", p.trim_end_matches('0').trim_end_matches('.'))
}

/// Per-target weights for `targets` given the weight of each shape by name
/// (`weight_of(base name)`). A shape with in-between frames follows the
/// piecewise-linear path through them: at a weight between two frames the two
/// frames share it (hat functions), so `Mouth@50` alone is the half-open mouth
/// and `Mouth` the fully open one. Plain shapes pass their weight through.
pub fn inbetween_weights(targets: &[MorphTarget], weight_of: &dyn Fn(&str) -> f32) -> Vec<f32> {
    if !targets.iter().any(|t| t.name.contains(INBETWEEN_SEP)) {
        return targets.iter().map(|t| weight_of(&t.name)).collect();
    }
    // frames per base shape: (frame weight, target index), sorted; 1.0 = the base target.
    // First index per name: a face model has hundreds of shapes, so lookups are hashed (not scanned).
    let mut first: std::collections::HashMap<&str, usize> = std::collections::HashMap::with_capacity(targets.len());
    for (i, t) in targets.iter().enumerate() {
        first.entry(t.name.as_str()).or_insert(i);
    }
    let mut groups: std::collections::HashMap<&str, Vec<(f32, usize)>> = std::collections::HashMap::new();
    for (i, t) in targets.iter().enumerate() {
        if let Some((base, w)) = split_inbetween(&t.name)
            && first.contains_key(base)
        {
            groups.entry(base).or_default().push((w, i));
        }
    }
    let mut out: Vec<f32> = targets.iter().map(|t| weight_of(&t.name)).collect();
    for (base, frames) in groups.iter_mut() {
        let Some(&full) = first.get(*base) else { continue };
        let w = weight_of(base);
        frames.sort_by(|a, b| a.0.total_cmp(&b.0));
        frames.push((1.0, full));
        for (_, i) in frames.iter() {
            out[*i] = 0.0;
        }
        if w <= 0.0 {
            // Below the first frame: the first frame ramps from 0, extrapolating linearly.
            let (f0, i0) = frames[0];
            out[i0] = w / f0;
        } else if w >= 1.0 {
            out[full] = w;
        } else {
            let hi = frames.partition_point(|(f, _)| *f <= w).min(frames.len() - 1);
            let lo = hi.checked_sub(1);
            let (fh, ih) = frames[hi];
            let flo = lo.map(|l| frames[l].0).unwrap_or(0.0);
            let t = ((w - flo) / (fh - flo).max(1e-9)).clamp(0.0, 1.0);
            out[ih] = t;
            if let Some(l) = lo {
                out[frames[l].1] = 1.0 - t;
            }
        }
    }
    out
}

/// A sparse delta list's largest offset (position or normal length).
fn max_delta(deltas: &[(u32, [f32; 3], [f32; 3])]) -> f32 {
    deltas.iter().map(|(_, p, n)| Vec3::from(*p).length().max(Vec3::from(*n).length())).fold(0.0, f32::max)
}

/// Apply the Import BlendShapes options that act on the finished targets:
/// drop offsets at or below the threshold, drop shapes left empty, and strip
/// normal offsets for `BlendNormals::None`. Returns how many shapes were dropped.
pub fn prune_morphs(morphs: &mut Vec<MorphTarget>, opts: &BlendShapeImport) -> usize {
    let before = morphs.len();
    let thr = opts.delta_threshold.max(0.0);
    for m in morphs.iter_mut() {
        if opts.normals == BlendNormals::None {
            for d in &mut m.deltas {
                d.2 = [0.0; 3];
            }
        }
        m.deltas.retain(|(_, p, n)| Vec3::from(*p).length() > thr || Vec3::from(*n).length() > thr);
    }
    morphs.retain(|m| !m.deltas.is_empty() && max_delta(&m.deltas) > 0.0);
    before - morphs.len()
}

/// `BlendNormals::Calculate`: replace each shape's normal offsets with the
/// change in the (area weighted) vertex normal when the shape is applied at full
/// weight, relative to the same calculation on the base mesh. Seams and hard
/// edges stay as they are because the mesh is welded exactly as imported.
pub fn calculate_morph_normals(vertices: &[zestengine_render::Vertex], indices: &[u32], morphs: &mut [MorphTarget]) {
    let accumulate = |pos: &dyn Fn(usize) -> Vec3| -> Vec<Vec3> {
        let mut n = vec![Vec3::ZERO; vertices.len()];
        for tri in indices.as_chunks::<3>().0 {
            let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            let f = (pos(b) - pos(a)).cross(pos(c) - pos(a));
            n[a] += f;
            n[b] += f;
            n[c] += f;
        }
        n
    };
    let base_pos = |i: usize| Vec3::from(vertices[i].position);
    let base = accumulate(&base_pos);
    for m in morphs.iter_mut() {
        if m.deltas.is_empty() {
            continue;
        }
        let mut offset: std::collections::HashMap<u32, Vec3> = std::collections::HashMap::with_capacity(m.deltas.len());
        for (v, p, _) in &m.deltas {
            offset.insert(*v, Vec3::from(*p));
        }
        // Only triangles touching a displaced vertex change; recompute just those.
        let pos = |i: usize| base_pos(i) + offset.get(&(i as u32)).copied().unwrap_or(Vec3::ZERO);
        let mut shifted: std::collections::HashMap<usize, Vec3> = std::collections::HashMap::new();
        for tri in indices.as_chunks::<3>().0 {
            let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if ![a, b, c].iter().any(|v| offset.contains_key(&(*v as u32))) {
                continue;
            }
            let old = (base_pos(b) - base_pos(a)).cross(base_pos(c) - base_pos(a));
            let new = (pos(b) - pos(a)).cross(pos(c) - pos(a));
            for v in [a, b, c] {
                *shifted.entry(v).or_insert(Vec3::ZERO) += new - old;
            }
        }
        let mut deltas: std::collections::HashMap<u32, ([f32; 3], [f32; 3])> = m.deltas.iter().map(|(v, p, _)| (*v, (*p, [0.0; 3]))).collect();
        for (v, change) in shifted {
            let b = base[v];
            let n0 = b.try_normalize().unwrap_or(Vec3::Y);
            let n1 = (b + change).try_normalize().unwrap_or(n0);
            let dn = (n1 - n0).to_array();
            deltas.entry(v as u32).and_modify(|e| e.1 = dn).or_insert(([0.0; 3], dn));
        }
        let mut out: Vec<(u32, [f32; 3], [f32; 3])> = deltas.into_iter().map(|(v, (p, n))| (v, p, n)).collect();
        out.sort_by_key(|d| d.0);
        m.deltas = out;
    }
}

/// Run the Import BlendShapes passes on a finished mesh's targets.
pub fn finish_morphs(mesh: &zestengine_render::MeshData, morphs: &mut Vec<MorphTarget>, opts: &BlendShapeImport) {
    if morphs.is_empty() {
        return;
    }
    prune_morphs(morphs, opts);
    if opts.normals == BlendNormals::Calculate || opts.legacy_normals {
        calculate_morph_normals(&mesh.vertices, &mesh.indices, morphs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(name: &str, d: f32) -> MorphTarget {
        MorphTarget { name: name.into(), deltas: vec![(0, [d, 0.0, 0.0], [0.0; 3])] }
    }

    #[test]
    fn reduction_keeps_a_straight_line_to_its_ends() {
        let times: Vec<f32> = (0..30).map(|i| i as f32 / 30.0).collect();
        let values: Vec<Vec3> = times.iter().map(|t| Vec3::new(*t * 2.0, 1.0, 0.0)).collect();
        let (t, v) = reduce_vec3(&times, &values, 0.5);
        assert_eq!(t.len(), 2, "a line is two keys");
        assert_eq!(v[1], values[29]);
    }

    #[test]
    fn reduction_keeps_the_corner_of_a_bent_curve() {
        let times: Vec<f32> = (0..21).map(|i| i as f32).collect();
        let values: Vec<Vec3> = (0..21).map(|i| Vec3::new(if i <= 10 { i as f32 } else { 10.0 }, 0.0, 0.0)).collect();
        let (t, _) = reduce_vec3(&times, &values, 0.5);
        assert_eq!(t, vec![0.0, 10.0, 20.0]);
    }

    #[test]
    fn rotation_reduction_respects_the_degree_tolerance() {
        let times: Vec<f32> = (0..90).map(|i| i as f32 / 30.0).collect();
        let q: Vec<Quat> = (0..90).map(|i| Quat::from_rotation_y((i as f32 * 2.0).to_radians())).collect();
        // A constant-rate spin is a perfect slerp: two keys.
        assert_eq!(reduce_quat(&times, &q, 0.5).0.len(), 2);
        // A bump of 10 degrees survives a 0.5 degree tolerance but not a 20 degree one.
        let mut bumped = q.clone();
        bumped[45] = Quat::from_rotation_y((90.0f32 + 10.0).to_radians());
        assert!(reduce_quat(&times, &bumped, 0.5).0.len() > 2);
        assert_eq!(reduce_quat(&times, &bumped, 20.0).0.len(), 2);
    }

    #[test]
    fn continuity_flips_signs_not_rotations() {
        let a = Quat::from_rotation_z(3.0);
        let mut q = vec![a, -Quat::from_rotation_z(3.1), Quat::from_rotation_z(3.2)];
        assert_eq!(make_continuous(&mut q), 1, "only the flipped key moves");
        assert!(q[0].dot(q[1]) > 0.0 && q[1].dot(q[2]) > 0.0);
    }

    #[test]
    fn loop_pose_closes_the_loop_and_spares_the_root_travel() {
        let mut clip = AnimationClip {
            name: "c".into(),
            duration: 1.0,
            channels: vec![
                AnimChannel { joint: 0, path: ChannelPath::Translation, times: vec![0.0, 0.5, 1.0], vec_values: vec![Vec3::ZERO, Vec3::X, Vec3::new(2.0, 0.0, 0.0)], quat_values: vec![], extra: None },
                AnimChannel { joint: 1, path: ChannelPath::Translation, times: vec![0.0, 0.5, 1.0], vec_values: vec![Vec3::ZERO, Vec3::X, Vec3::new(2.0, 0.0, 0.0)], quat_values: vec![], extra: None },
                AnimChannel { joint: 1, path: ChannelPath::Rotation, times: vec![0.0, 1.0], vec_values: vec![], quat_values: vec![Quat::IDENTITY, Quat::from_rotation_y(0.5)], extra: None },
            ],
            morph_tracks: Vec::new(), playback: None,
        };
        apply_loop_pose(&mut clip, Some(0));
        assert_eq!(clip.channels[0].vec_values[2], Vec3::new(2.0, 0.0, 0.0), "root travel stays");
        assert!(clip.channels[1].vec_values[2].length() < 1e-5, "the end matches the start");
        assert!(clip.channels[2].quat_values[1].angle_between(Quat::IDENTITY) < 1e-4);
        let (deg, _) = loop_error(&clip);
        assert!(deg < 0.01);
    }

    #[test]
    fn inbetween_names_round_trip() {
        assert_eq!(inbetween_name("Mouth", 0.5), "Mouth@50");
        assert_eq!(inbetween_name("Mouth", 0.125), "Mouth@12.5");
        assert_eq!(split_inbetween("Mouth@50"), Some(("Mouth", 0.5)));
        assert_eq!(split_inbetween("Mouth@100"), None);
        assert_eq!(split_inbetween("Mouth@x"), None);
        assert_eq!(split_inbetween("Mouth"), None);
    }

    #[test]
    fn inbetweens_follow_the_piecewise_linear_path() {
        let targets = vec![target("Mouth", 1.0), target("Mouth@50", 0.8), target("Brow", 1.0)];
        let w = |v: f32| move |n: &str| if n == "Mouth" { v } else { 0.25 };
        // At the in-between's own weight only that frame is active.
        let r = inbetween_weights(&targets, &w(0.5));
        assert_eq!((r[0], r[1], r[2]), (0.0, 1.0, 0.25));
        // Halfway from the in-between to the full shape they share the weight.
        let r = inbetween_weights(&targets, &w(0.75));
        assert!((r[0] - 0.5).abs() < 1e-6 && (r[1] - 0.5).abs() < 1e-6);
        // Below the in-between the frame ramps up from nothing, at full the full shape wins.
        let r = inbetween_weights(&targets, &w(0.25));
        assert!((r[1] - 0.5).abs() < 1e-6 && r[0] == 0.0);
        let r = inbetween_weights(&targets, &w(1.0));
        assert_eq!((r[0], r[1]), (1.0, 0.0));
        // Without in-between names the weights pass through.
        let plain = vec![target("A", 1.0), target("B", 1.0)];
        assert_eq!(inbetween_weights(&plain, &|n| if n == "A" { 0.3 } else { 0.9 }), vec![0.3, 0.9]);
    }

    #[test]
    fn pruning_drops_empty_shapes_and_strips_normals() {
        let mut m = vec![
            target("Real", 1.0),
            MorphTarget { name: "Empty".into(), deltas: vec![(0, [0.0; 3], [0.0; 3]), (1, [1e-9, 0.0, 0.0], [0.0; 3])] },
            MorphTarget { name: "N".into(), deltas: vec![(0, [1.0, 0.0, 0.0], [0.5, 0.0, 0.0])] },
        ];
        let opts = BlendShapeImport { normals: BlendNormals::None, ..Default::default() };
        assert_eq!(prune_morphs(&mut m, &opts), 1);
        assert_eq!(m.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(), ["Real", "N"]);
        assert_eq!(m[1].deltas[0].2, [0.0; 3]);
    }

    #[test]
    fn calculated_normals_follow_the_displaced_surface() {
        use zestengine_render::Vertex;
        let v = |p: [f32; 3]| Vertex { position: p, normal: [0.0, 0.0, 1.0], ..Default::default() };
        let verts = vec![v([0.0, 0.0, 0.0]), v([1.0, 0.0, 0.0]), v([0.0, 1.0, 0.0])];
        let mut morphs = vec![MorphTarget { name: "tilt".into(), deltas: vec![(1, [0.0, 0.0, 1.0], [0.0; 3])] }];
        calculate_morph_normals(&verts, &[0, 1, 2], &mut morphs);
        let dn = morphs[0].deltas.iter().find(|d| d.0 == 0).unwrap().2;
        // Lifting the +X corner tilts the normal toward -X.
        assert!(dn[0] < -0.1, "{dn:?}");
    }

    #[test]
    fn the_clip_list_starts_from_the_takes() {
        let takes = vec![
            TakeInfo { name: "Walk".into(), first_frame: 1.0, last_frame: 31.0, ..Default::default() },
            TakeInfo { name: "Idle".into(), first_frame: 0.0, last_frame: 60.0, ..Default::default() },
        ];
        let mut a = AnimImport::default();
        a.customize_from_takes(&takes);
        assert_eq!(a.clips.iter().map(|c| (c.name.as_str(), c.first_frame, c.last_frame)).collect::<Vec<_>>(), [("Walk", 1.0, 31.0), ("Idle", 0.0, 60.0)]);
        a.customize_from_takes(&[]);
        assert_eq!(a.clips.len(), 2, "already custom: unchanged");
        a.add_clip(&takes);
        a.add_clip(&takes);
        assert_eq!((a.clips[2].name.as_str(), a.clips[2].take.as_str()), ("New Clip", "Idle"));
        assert_eq!(a.clips[3].name, "New Clip 2");
    }

    #[test]
    fn old_metas_get_defaults() {
        let a: AnimImport = ron::from_str("(resample_curves: false)").unwrap();
        assert!(a.import_animation && !a.resample_curves && a.compression == AnimCompression::KeyframeReduction);
        assert_eq!((a.rotation_error, a.position_error, a.scale_error), (0.5, 0.5, 0.5));
        let b: BlendShapeImport = ron::from_str("()").unwrap();
        assert_eq!(b, BlendShapeImport::default());
        let c: ClipDef = ron::from_str("(name: \"x\")").unwrap();
        assert!(c.keep_original_position_y && c.name == "x");
        // A whole old model meta (no animation / blend_shapes fields) keeps loading.
        let m: crate::ModelImport = ron::from_str("(scale: 2.0, import_blend_shapes: false)").unwrap();
        assert_eq!((m.scale, m.animation.clone(), m.blend_shapes.clone()), (2.0, AnimImport::default(), BlendShapeImport::default()));
        // Settings round-trip through RON.
        let mut set = AnimImport::default();
        set.clips.push(ClipDef { name: "Run".into(), take: "Take 001".into(), first_frame: 3.0, last_frame: 40.0, loop_time: true, events: vec![ClipEventDef { time: 0.2, name: "Step".into(), ..Default::default() }], ..Default::default() });
        let text = ron::ser::to_string(&set).unwrap();
        assert_eq!(ron::from_str::<AnimImport>(&text).unwrap(), set);
    }

    #[test]
    fn legacy_blend_shape_normals_recompute_whatever_the_mode_says() {
        let mesh = zestengine_render::MeshData {
            vertices: (0..3).map(|i| zestengine_render::Vertex { position: [i as f32, (i % 2) as f32, 0.0], normal: [0.0, 0.0, 1.0], ..Default::default() }).collect(),
            indices: vec![0, 1, 2],
            has_lightmap_uv: false,
        };
        let moved = |opts: &BlendShapeImport| {
            let mut m = vec![MorphTarget { name: "s".into(), deltas: vec![(0, [0.0, 0.0, 1.0], [0.0; 3]), (1, [0.0, 0.0, 1.0], [0.0; 3]), (2, [0.0, 0.0, 0.0], [0.0; 3])] }];
            finish_morphs(&mesh, &mut m, opts);
            m[0].deltas.iter().any(|d| d.2 != [0.0; 3])
        };
        assert!(!moved(&BlendShapeImport { normals: BlendNormals::None, ..Default::default() }));
        assert!(moved(&BlendShapeImport { normals: BlendNormals::None, legacy_normals: true, ..Default::default() }));
        assert!(!BlendShapeImport::default().legacy_normals);
    }
}
