//! Skeletal rigging + animation: imported armature (joint hierarchy + bind
//! pose) and animation clips. The mesh `Vertex` carries up to 4 skin-local
//! joint indices + weights; a `Skeleton`'s joints line up with those indices.
//!
//! Design follows the glTF model (which both Unity and Unreal interop with):
//! flat joint array, parent indices, per-joint inverse-bind matrix, and clips
//! as per-joint TRS channels with keyframe times. Sampling produces a joint
//! matrix palette (`joint_world * inverse_bind`) ready for linear-blend skinning.

use glam::{Mat4, Quat, Vec3};

/// One skeleton joint (bone). `parent` indexes back into `Skeleton::joints`
/// (`None` = root). `inverse_bind` maps a world-space vertex into this joint's
/// bind-pose local space; the rest TRS is the default pose used when no clip
/// drives the joint.
#[derive(Clone, Debug)]
pub struct Joint {
    pub name: String,
    pub parent: Option<usize>,
    pub inverse_bind: Mat4,
    pub rest_translation: Vec3,
    pub rest_rotation: Quat,
    pub rest_scale: Vec3,
}

/// An armature: joints in a fixed order matching the vertex joint indices.
/// Stored topologically (parents before children) so a single forward pass
/// computes world matrices.
#[derive(Clone, Debug, Default)]
pub struct Skeleton {
    pub joints: Vec<Joint>,
    /// Persisted rig description (human bone map etc.); `None` = not imported,
    /// runtime falls back to name matching.
    pub rig: Option<crate::human_rig::RigImport>,
    /// Joints that are not bones but the model's own non-bone ancestors of the
    /// skeleton (the `Armature` empty a Blender FBX parents the bones to), as
    /// Unity keeps them above the Hips. They sit at the END of `joints`, the
    /// topmost first, so vertex joint indices and clip tracks stay valid. Empty
    /// = none (every rig that has no such ancestor, and old data).
    pub containers: Vec<usize>,
}

impl Skeleton {
    /// Whether joint `i` is one of the [`Skeleton::containers`].
    pub fn is_container(&self, i: usize) -> bool {
        self.containers.contains(&i)
    }

    /// Add the model's non-bone ancestors of the skeleton as container joints
    /// (see [`Skeleton::containers`]). `chain` is topmost first: name and rest
    /// transform of each node from just below the model root down to the node
    /// above the bones; `roots` are the bone joints that have no parent and
    /// become children of the last container. The containers are appended, so
    /// no existing joint index moves. Returns the chain's combined transform
    /// `L`. The inverse bind matrices and the skinned instances' transforms are
    /// left as they were: [`Skeleton::world_matrices`] takes `L` back out of the
    /// pose (see there), so a scene saved before the containers existed, and
    /// everything that reads bone world transforms, sees the same frame as
    /// before. `None` (nothing
    /// changed) for an empty chain, containers already present or a chain that
    /// cannot be inverted.
    pub fn add_containers(&mut self, roots: &[usize], chain: &[(String, Vec3, Quat, Vec3)]) -> Option<Mat4> {
        if chain.is_empty() || roots.is_empty() || !self.containers.is_empty() {
            return None;
        }
        let locals: Vec<Mat4> = chain.iter().map(|(_, t, r, s)| Mat4::from_scale_rotation_translation(*s, *r, *t)).collect();
        let combined = locals.iter().fold(Mat4::IDENTITY, |acc, m| acc * *m);
        if !combined.is_finite() || combined.determinant().abs() < 1e-12 {
            return None;
        }
        let base = self.joints.len();
        let mut world = Mat4::IDENTITY;
        for (k, ((name, t, r, s), local)) in chain.iter().zip(&locals).enumerate() {
            world *= *local;
            self.joints.push(Joint {
                name: name.clone(),
                parent: (k > 0).then(|| base + k - 1),
                inverse_bind: world.inverse(),
                rest_translation: *t,
                rest_rotation: *r,
                rest_scale: *s,
            });
            self.containers.push(base + k);
        }
        for &j in roots {
            if let Some(joint) = self.joints.get_mut(j) {
                joint.parent = Some(base + chain.len() - 1);
            }
        }
        Some(combined)
    }

    /// The rest transform of the container chain (identity without containers):
    /// the frame the bones' local motion is expressed in. `world_matrices`
    /// takes it back out of every joint, so it is only needed to relate a joint
    /// to the file's own (container-including) frame.
    pub fn container_basis(&self) -> Mat4 {
        self.containers.iter().fold(Mat4::IDENTITY, |acc, &c| {
            let j = &self.joints[c];
            acc * Mat4::from_scale_rotation_translation(j.rest_scale, j.rest_rotation, j.rest_translation)
        })
    }

    /// The joint that carries the character's motion (root motion, loop pose,
    /// the skeleton's name): the first parentless bone, looking through the
    /// container joints above it. Without containers this is the first joint
    /// with no parent, as before containers existed.
    pub fn motion_root(&self) -> Option<usize> {
        self.joints
            .iter()
            .enumerate()
            .position(|(i, j)| !self.is_container(i) && j.parent.is_none_or(|p| self.is_container(p)))
    }

    /// Default (rest-pose) joint local transforms.
    pub fn rest_locals(&self) -> Vec<Mat4> {
        self.joints
            .iter()
            .map(|j| Mat4::from_scale_rotation_translation(j.rest_scale, j.rest_rotation, j.rest_translation))
            .collect()
    }

    /// Compose per-joint local transforms into the skinning matrix palette
    /// (`joint_world * inverse_bind`).
    pub fn palette(&self, locals: &[Mat4]) -> Vec<Mat4> {
        self.world_matrices(locals)
            .iter()
            .zip(&self.joints)
            .map(|(w, j)| *w * j.inverse_bind)
            .collect()
    }

    /// Per-joint model-space matrices for `locals`. Handles arbitrary joint
    /// order (glTF skins aren't topologically sorted) via a fixpoint that
    /// resolves a joint once its parent's world matrix is known.
    pub fn world_matrices(&self, locals: &[Mat4]) -> Vec<Mat4> {
        let n = self.joints.len();
        let mut world = vec![Mat4::IDENTITY; n];
        let mut done = vec![false; n];
        let mut remaining = n;
        while remaining > 0 {
            let mut progressed = false;
            for i in 0..n {
                if done[i] {
                    continue;
                }
                let ready = match self.joints[i].parent {
                    None => true,
                    Some(p) => done[p],
                };
                if ready {
                    let local = locals.get(i).copied().unwrap_or(Mat4::IDENTITY);
                    world[i] = match self.joints[i].parent {
                        Some(p) => world[p] * local,
                        None => local,
                    };
                    done[i] = true;
                    remaining -= 1;
                    progressed = true;
                }
            }
            if !progressed {
                break; // malformed (cycle / bad parent index); leave the rest identity
            }
        }
        // The containers sit above the bones in the hierarchy, but the model's
        // frame stays the one without them: take their rest basis out again, so
        // bone worlds, the skinning palette and every attachment read the same
        // numbers as before the containers existed (and the skinned instance's
        // transform needs no compensation).
        if !self.containers.is_empty() {
            let inv = self.container_basis().inverse();
            if inv.is_finite() {
                for w in &mut world {
                    *w = inv * *w;
                }
            }
        }
        world
    }
}

/// Which TRS component a channel animates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChannelPath {
    #[default]
    Translation,
    Rotation,
    Scale,
}

/// One animated property of one joint: keyframe times + values (linear interp;
/// rotations slerp). Only the field matching `path` is populated, unless the
/// channel carries [`ChannelExtra`]: then its values may live in packed form
/// (read keys through [`AnimChannel::vec_at`] / [`AnimChannel::quat_at`] and
/// sample through [`AnimChannel::sample_vec`] / [`AnimChannel::sample_quat`]).
#[derive(Clone, Debug, Default)]
pub struct AnimChannel {
    pub joint: usize,
    pub path: ChannelPath,
    pub times: Vec<f32>,
    pub vec_values: Vec<Vec3>,
    pub quat_values: Vec<Quat>,
    /// Hermite tangents and / or packed values; `None` = plain linear keys.
    pub extra: Option<Box<ChannelExtra>>,
}

/// The optional storage forms of a channel.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChannelExtra {
    /// Authored tangents (Keep Tangents): between two keys the value follows a
    /// cubic Hermite curve instead of a straight line.
    pub tangents: Option<Tangents>,
    /// 16-bit quantised values, replacing `vec_values` / `quat_values`.
    pub packed: Option<PackedValues>,
}

/// Per-key tangents of a Hermite channel. For translation and scale they are
/// derivatives per second; for rotation, angular velocities (radians per
/// second) in the key's own local frame. `tan_in` arrives at the key from the
/// left, `tan_out` leaves it to the right.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Tangents {
    pub tan_in: Vec<Vec3>,
    pub tan_out: Vec<Vec3>,
}

/// Values quantised to 16 bits per lane over a per-channel range
/// (`min + extent * code / 65535`). Three lanes for vectors, four for
/// quaternions (decoded values are renormalised).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PackedValues {
    pub lanes: u8,
    pub min: [f32; 4],
    pub extent: [f32; 4],
    /// `keys * lanes` codes, key-major.
    pub codes: Vec<u16>,
}

impl PackedValues {
    pub fn keys(&self) -> usize {
        if self.lanes == 0 { 0 } else { self.codes.len() / self.lanes as usize }
    }

    /// Quantise `rows` (each `lanes` wide).
    pub fn pack(lanes: usize, rows: &[[f32; 4]]) -> Self {
        let mut min = [f32::INFINITY; 4];
        let mut max = [f32::NEG_INFINITY; 4];
        for r in rows {
            for k in 0..lanes {
                min[k] = min[k].min(r[k]);
                max[k] = max[k].max(r[k]);
            }
        }
        let mut extent = [0.0f32; 4];
        for k in 0..lanes {
            if !min[k].is_finite() {
                min[k] = 0.0;
                max[k] = 0.0;
            }
            extent[k] = max[k] - min[k];
        }
        let mut codes = Vec::with_capacity(rows.len() * lanes);
        for r in rows {
            for k in 0..lanes {
                let c = if extent[k] > 0.0 { ((r[k] - min[k]) / extent[k] * 65535.0).round().clamp(0.0, 65535.0) as u16 } else { 0 };
                codes.push(c);
            }
        }
        for m in &mut min[lanes..] {
            *m = 0.0;
        }
        Self { lanes: lanes as u8, min, extent, codes }
    }

    #[inline]
    pub fn lane(&self, key: usize, k: usize) -> f32 {
        self.min[k] + self.extent[k] * (self.codes[key * self.lanes as usize + k] as f32 * (1.0 / 65535.0))
    }

    #[inline]
    pub fn vec3(&self, key: usize) -> Vec3 {
        Vec3::new(self.lane(key, 0), self.lane(key, 1), self.lane(key, 2))
    }

    #[inline]
    pub fn quat(&self, key: usize) -> Quat {
        Quat::from_xyzw(self.lane(key, 0), self.lane(key, 1), self.lane(key, 2), self.lane(key, 3)).normalize()
    }

    /// Bytes this storage takes when written (codes plus the range header).
    pub fn bytes(&self) -> usize {
        self.codes.len() * 2 + 2 * 4 * self.lanes as usize
    }
}

fn hermite(s: f32) -> (f32, f32, f32, f32) {
    let (s2, s3) = (s * s, s * s * s);
    (2.0 * s3 - 3.0 * s2 + 1.0, s3 - 2.0 * s2 + s, -2.0 * s3 + 3.0 * s2, s3 - s2)
}

/// Rotation vector (axis * angle) of `q`, shortest way.
pub fn quat_log(q: Quat) -> Vec3 {
    let q = if q.w < 0.0 { -q } else { q };
    let v = Vec3::new(q.x, q.y, q.z);
    let len = v.length();
    if len < 1e-7 {
        return v * 2.0;
    }
    v / len * (2.0 * len.atan2(q.w))
}

/// The rotation of a rotation vector.
pub fn quat_exp(v: Vec3) -> Quat {
    let a = v.length();
    if a < 1e-7 {
        return Quat::from_xyzw(v.x * 0.5, v.y * 0.5, v.z * 0.5, 1.0).normalize();
    }
    Quat::from_axis_angle(v / a, a)
}

impl AnimChannel {
    /// A plain linear translation / scale channel.
    pub fn vec(joint: usize, path: ChannelPath, times: Vec<f32>, values: Vec<Vec3>) -> Self {
        Self { joint, path, times, vec_values: values, quat_values: Vec::new(), extra: None }
    }

    /// A plain linear rotation channel.
    pub fn rot(joint: usize, times: Vec<f32>, values: Vec<Quat>) -> Self {
        Self { joint, path: ChannelPath::Rotation, times, vec_values: Vec::new(), quat_values: values, extra: None }
    }

    fn packed(&self) -> Option<&PackedValues> {
        self.extra.as_ref().and_then(|e| e.packed.as_ref())
    }

    pub fn tangents(&self) -> Option<&Tangents> {
        self.extra.as_ref().and_then(|e| e.tangents.as_ref())
    }

    pub fn is_packed(&self) -> bool {
        self.packed().is_some()
    }

    /// Number of keys.
    pub fn len(&self) -> usize {
        self.times.len()
    }

    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// Key `i` of a translation / scale channel.
    #[inline]
    pub fn vec_at(&self, i: usize) -> Option<Vec3> {
        match self.packed() {
            Some(p) => (i < p.keys()).then(|| p.vec3(i)),
            None => self.vec_values.get(i).copied(),
        }
    }

    /// Key `i` of a rotation channel.
    #[inline]
    pub fn quat_at(&self, i: usize) -> Option<Quat> {
        match self.packed() {
            Some(p) => (i < p.keys()).then(|| p.quat(i)),
            None => self.quat_values.get(i).copied(),
        }
    }

    /// The value of a translation / scale channel at `t` (seconds).
    pub fn sample_vec(&self, t: f32) -> Option<Vec3> {
        if self.times.is_empty() {
            return None;
        }
        let (i0, i1, f) = key_indices(&self.times, t);
        let (a, b) = (self.vec_at(i0)?, self.vec_at(i1)?);
        if let Some(tg) = self.tangents()
            && i0 != i1
            && f > 0.0
            && let (Some(m0), Some(m1)) = (tg.tan_out.get(i0), tg.tan_in.get(i1))
        {
            let dt = self.times[i1] - self.times[i0];
            let (h00, h10, h01, h11) = hermite(f);
            return Some(a * h00 + *m0 * (h10 * dt) + b * h01 + *m1 * (h11 * dt));
        }
        Some(a.lerp(b, f))
    }

    /// The value of a rotation channel at `t` (seconds).
    pub fn sample_quat(&self, t: f32) -> Option<Quat> {
        if self.times.is_empty() {
            return None;
        }
        let (i0, i1, f) = key_indices(&self.times, t);
        let (a, b) = (self.quat_at(i0)?, self.quat_at(i1)?);
        if let Some(tg) = self.tangents()
            && i0 != i1
            && f > 0.0
            && let (Some(w0), Some(w1)) = (tg.tan_out.get(i0), tg.tan_in.get(i1))
        {
            let dt = self.times[i1] - self.times[i0];
            let b = if a.dot(b) < 0.0 { -b } else { b };
            let rel = a.inverse() * b;
            let d = quat_log(rel);
            // The end velocity is in b's frame; bring it into a's.
            let m0 = *w0 * dt;
            let m1 = (rel * *w1) * dt;
            let (h00, h10, h01, h11) = hermite(f);
            let _ = h00;
            let h = m0 * h10 + d * h01 + m1 * h11;
            return Some((a * quat_exp(h)).normalize());
        }
        Some(a.slerp(b, f))
    }

    /// Replace packed values by plain `f32` keys (tangents stay).
    pub fn unpack(&mut self) {
        let Some(p) = self.extra.as_mut().and_then(|e| e.packed.take()) else { return };
        match self.path {
            ChannelPath::Rotation => self.quat_values = (0..p.keys()).map(|i| p.quat(i)).collect(),
            _ => self.vec_values = (0..p.keys()).map(|i| p.vec3(i)).collect(),
        }
        if self.extra.as_ref().is_some_and(|e| e.tangents.is_none()) {
            self.extra = None;
        }
    }

    /// Multiply the lengths of a translation channel by `s` (a unit change).
    pub fn scale_translation(&mut self, s: f32) {
        for v in &mut self.vec_values {
            *v *= s;
        }
        if let Some(e) = self.extra.as_mut() {
            if let Some(p) = e.packed.as_mut() {
                for k in 0..3 {
                    p.min[k] *= s;
                    p.extent[k] *= s;
                }
                if s < 0.0 {
                    // A negative factor flips the range; requantise through floats.
                    let mut c = self.clone();
                    c.unpack();
                    *self = c;
                    return;
                }
            }
            if let Some(t) = e.tangents.as_mut() {
                for v in t.tan_in.iter_mut().chain(t.tan_out.iter_mut()) {
                    *v *= s;
                }
            }
        }
    }

    /// Bytes the channel's values take when stored (keys + times, the form
    /// the import cache writes).
    pub fn value_bytes(&self) -> usize {
        let lanes = if self.path == ChannelPath::Rotation { 4 } else { 3 };
        let tangents = self.tangents().map_or(0, |t| (t.tan_in.len() + t.tan_out.len()) * 12);
        match self.packed() {
            Some(p) => p.bytes() + tangents,
            None => self.times.len() * lanes * 4 + tangents,
        }
    }
}

/// A skeletal animation clip: per-joint TRS channels over a duration.
#[derive(Clone, Debug)]
pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<AnimChannel>,
    /// Blendshape weight curves imported with the clip (`BlendShape` tracks:
    /// `target` = the mesh node, `name` = the shape, key value = weight).
    /// Empty for clips without any.
    pub morph_tracks: Vec<crate::AnimTrack>,
    /// Playback settings from the model's clip list (Loop Time, cycle offset,
    /// root bake, mirror, events); `None` = play as the Animator says.
    pub playback: Option<Box<crate::ClipPlayback>>,
}

impl AnimationClip {
    /// Sample the clip at time `t` (seconds, looped), returning per-joint local
    /// transforms seeded from the skeleton's rest pose. Feed into
    /// [`Skeleton::palette`] for the skinning matrices.
    pub fn sample(&self, skeleton: &Skeleton, t: f32) -> Vec<Mat4> {
        let n = skeleton.joints.len();
        let mut trans: Vec<Vec3> = skeleton.joints.iter().map(|j| j.rest_translation).collect();
        let mut rot: Vec<Quat> = skeleton.joints.iter().map(|j| j.rest_rotation).collect();
        let mut scale: Vec<Vec3> = skeleton.joints.iter().map(|j| j.rest_scale).collect();
        let tt = if self.duration > 1e-6 { t.rem_euclid(self.duration) } else { 0.0 };
        for ch in &self.channels {
            if ch.joint >= n || ch.times.is_empty() {
                continue;
            }
            match ch.path {
                ChannelPath::Translation => {
                    if let Some(v) = ch.sample_vec(tt) {
                        trans[ch.joint] = v;
                    }
                }
                ChannelPath::Scale => {
                    if let Some(v) = ch.sample_vec(tt) {
                        scale[ch.joint] = v;
                    }
                }
                ChannelPath::Rotation => {
                    if let Some(q) = ch.sample_quat(tt) {
                        rot[ch.joint] = q;
                    }
                }
            }
        }
        (0..n)
            .map(|i| Mat4::from_scale_rotation_translation(scale[i], rot[i], trans[i]))
            .collect()
    }

    /// Replace every packed channel by plain keys (for code that reads
    /// `vec_values` / `quat_values` directly).
    pub fn unpack(&mut self) {
        for ch in &mut self.channels {
            ch.unpack();
        }
    }

    /// Bytes the clip's channels take when stored.
    pub fn stored_bytes(&self) -> usize {
        self.channels.iter().map(|c| c.times.len() * 4 + c.value_bytes()).sum()
    }
}

/// Linear-blend skin one vertex position by a joint matrix palette. `joints`
/// are skin-local indices into `palette`; `weights` are the matching weights.
/// Falls back to the input position when the vertex is unweighted (static).
pub fn skin_position(pos: Vec3, joints: [u32; 4], weights: [f32; 4], palette: &[Mat4]) -> Vec3 {
    let sum: f32 = weights.iter().sum();
    if sum <= 1e-5 {
        return pos;
    }
    let mut out = Vec3::ZERO;
    for k in 0..4 {
        let w = weights[k];
        if w <= 0.0 {
            continue;
        }
        if let Some(m) = palette.get(joints[k] as usize) {
            out += w * m.transform_point3(pos);
        }
    }
    out / sum
}

/// Linear-blend skin a direction (normal/tangent) — same as `skin_position` but
/// using the matrices' rotational part (no translation), then renormalized.
pub fn skin_direction(dir: Vec3, joints: [u32; 4], weights: [f32; 4], palette: &[Mat4]) -> Vec3 {
    let sum: f32 = weights.iter().sum();
    if sum <= 1e-5 {
        return dir;
    }
    let mut out = Vec3::ZERO;
    for k in 0..4 {
        let w = weights[k];
        if w <= 0.0 {
            continue;
        }
        if let Some(m) = palette.get(joints[k] as usize) {
            out += w * m.transform_vector3(dir);
        }
    }
    out.normalize_or(dir)
}

/// Find the surrounding keyframe indices + interpolation factor for time `t`.
pub fn key_indices(times: &[f32], t: f32) -> (usize, usize, f32) {
    if times.len() == 1 || t <= times[0] {
        return (0, 0, 0.0);
    }
    if t >= *times.last().unwrap() {
        let last = times.len() - 1;
        return (last, last, 0.0);
    }
    let i = times.partition_point(|&x| x < t).saturating_sub(1).min(times.len() - 2);
    let (a, b) = (times[i], times[i + 1]);
    let f = if (b - a).abs() > 1e-6 { (t - a) / (b - a) } else { 0.0 };
    (i, i + 1, f.clamp(0.0, 1.0))
}
