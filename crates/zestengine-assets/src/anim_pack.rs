//! Compressed storage of imported animation curves.
//!
//! Keyframe reduction (in [`crate::anim_import`]) removes keys; this module
//! changes how the surviving values are stored. A channel can be:
//!
//! * **constant**: every sample within a tiny epsilon of the first, one key;
//! * **reduced**: fewer keys, `f32` values (what Keyframes mode makes);
//! * **dense quantised**: every sample, 16 bits per lane over the channel's own
//!   range ([`PackedValues`]), decoded on the fly by the sampler;
//! * **reduced and quantised**: both.
//!
//! `Optimal` builds all four per channel and keeps the smallest one that still
//! reproduces the channel within the rotation / position / scale tolerances;
//! `Keyframes + Compress` always reduces and then quantises when the
//! tolerance allows. Rotations quantise their four components (renormalised
//! on decode): the error is checked as an angle against the original samples.

use glam::{Quat, Vec3};

use crate::anim_import::{AnimCompression, reduce_quat, reduce_vec3};
use crate::skeleton::{AnimChannel, AnimationClip, ChannelExtra, ChannelPath, PackedValues};

/// Constant-curve thresholds: a position / scale channel whose samples all lie
/// within this of the first is one key, a rotation within this many degrees.
const CONSTANT_EPS: f32 = 1e-5;
const CONSTANT_DEG: f32 = 0.005;

/// The smallest storage that keeps every channel of `clip` within the
/// tolerances (see the module docs). `mode` must reduce.
pub fn compress_clip(clip: &mut AnimationClip, mode: AnimCompression, rotation_deg: f32, position_pct: f32, scale_pct: f32) {
    if !mode.reduces() {
        return;
    }
    for ch in &mut clip.channels {
        if ch.times.len() < 2 || ch.tangents().is_some() || ch.is_packed() {
            continue;
        }
        let pct = if ch.path == ChannelPath::Scale { scale_pct } else { position_pct };
        match mode {
            AnimCompression::Off => {}
            AnimCompression::KeyframeReduction => reduce_channel(ch, rotation_deg, pct),
            AnimCompression::KeyframeReductionAndCompression => {
                reduce_channel(ch, rotation_deg, pct);
                try_pack(ch, rotation_deg, pct);
            }
            AnimCompression::Optimal => optimal_channel(ch, rotation_deg, pct),
        }
    }
}

/// Quantise every channel of `clip` that stays within the tolerances (no keys are dropped).
pub fn quantize_clip(clip: &mut AnimationClip, rotation_deg: f32, position_pct: f32, scale_pct: f32) {
    for ch in &mut clip.channels {
        if ch.times.len() < 2 || ch.tangents().is_some() || ch.is_packed() {
            continue;
        }
        let pct = if ch.path == ChannelPath::Scale { scale_pct } else { position_pct };
        try_pack(ch, rotation_deg, pct);
    }
}

fn reduce_channel(ch: &mut AnimChannel, rotation_deg: f32, pct: f32) {
    if ch.path == ChannelPath::Rotation {
        let (t, v) = reduce_quat(&ch.times, &ch.quat_values, rotation_deg);
        (ch.times, ch.quat_values) = (t, v);
    } else {
        let (t, v) = reduce_vec3(&ch.times, &ch.vec_values, pct);
        (ch.times, ch.vec_values) = (t, v);
    }
}

fn vec_tolerance(values: &[Vec3], pct: f32) -> f32 {
    let (mut lo, mut hi) = (values[0], values[0]);
    for v in values {
        lo = lo.min(*v);
        hi = hi.max(*v);
    }
    ((hi - lo).max_element() * pct / 100.0).max(1e-7)
}

/// Largest deviation of `b` (at `bt`) from `a` (at `at`), where `b` is sampled
/// linearly: the full-pipeline error of a candidate against the dense original.
fn vec_error(at: &[f32], a: &[Vec3], bt: &[f32], b: &[Vec3]) -> f32 {
    let probe = AnimChannel::vec(0, ChannelPath::Translation, bt.to_vec(), b.to_vec());
    at.iter().zip(a).map(|(t, v)| (probe.sample_vec(*t).unwrap_or(*v) - *v).abs().max_element()).fold(0.0, f32::max)
}

fn quat_error(at: &[f32], a: &[Quat], bt: &[f32], b: &[Quat]) -> f32 {
    let probe = AnimChannel::rot(0, bt.to_vec(), b.to_vec());
    at.iter()
        .zip(a)
        .map(|(t, q)| angle_between(probe.sample_quat(*t).unwrap_or(*q), *q))
        .fold(0.0, f32::max)
}

/// The angle between two rotations in radians, accurate for tiny angles
/// (`acos` of a dot product near 1 is not, in `f32`).
pub fn angle_between(a: Quat, b: Quat) -> f32 {
    let d = a.inverse() * b;
    2.0 * Vec3::new(d.x, d.y, d.z).length().atan2(d.w.abs())
}

fn vec_rows(v: &[Vec3]) -> Vec<[f32; 4]> {
    v.iter().map(|v| [v.x, v.y, v.z, 0.0]).collect()
}

fn pack_vec(v: &[Vec3]) -> PackedValues {
    PackedValues::pack(3, &vec_rows(v))
}

fn pack_quat(q: &[Quat]) -> PackedValues {
    PackedValues::pack(4, &q.iter().map(|q| q.to_array()).collect::<Vec<_>>())
}

fn decoded_vec(p: &PackedValues) -> Vec<Vec3> {
    (0..p.keys()).map(|i| p.vec3(i)).collect()
}

fn decoded_quat(p: &PackedValues) -> Vec<Quat> {
    (0..p.keys()).map(|i| p.quat(i)).collect()
}

/// Replace the channel's `f32` keys by quantised ones when the decoded values
/// stay within the tolerance of the current keys.
fn try_pack(ch: &mut AnimChannel, rotation_deg: f32, pct: f32) -> bool {
    if ch.times.len() < 2 {
        return false;
    }
    // Packing must pay: a short channel's range header can outweigh the saved bytes.
    let lanes = if ch.path == ChannelPath::Rotation { 4 } else { 3 };
    if packed_bytes(ch.times.len(), lanes) >= plain_bytes(ch.times.len(), lanes) {
        return false;
    }
    if ch.path == ChannelPath::Rotation {
        let p = pack_quat(&ch.quat_values);
        let tol = rotation_deg.to_radians().max(1e-6);
        if quat_error(&ch.times, &ch.quat_values, &ch.times, &decoded_quat(&p)) > tol * 0.5 {
            return false;
        }
        ch.quat_values = Vec::new();
        ch.extra = Some(Box::new(ChannelExtra { tangents: None, packed: Some(p) }));
    } else {
        let p = pack_vec(&ch.vec_values);
        let tol = vec_tolerance(&ch.vec_values, pct);
        if vec_error(&ch.times, &ch.vec_values, &ch.times, &decoded_vec(&p)) > tol * 0.5 {
            return false;
        }
        ch.vec_values = Vec::new();
        ch.extra = Some(Box::new(ChannelExtra { tangents: None, packed: Some(p) }));
    }
    true
}

/// Bytes of a candidate: times plus values.
fn plain_bytes(keys: usize, lanes: usize) -> usize {
    keys * (4 + 4 * lanes)
}

fn packed_bytes(keys: usize, lanes: usize) -> usize {
    keys * (4 + 2 * lanes) + 8 * lanes
}

/// Pick the smallest valid storage for one channel (see the module docs).
fn optimal_channel(ch: &mut AnimChannel, rotation_deg: f32, pct: f32) {
    let n = ch.times.len();
    let rot = ch.path == ChannelPath::Rotation;
    let lanes = if rot { 4 } else { 3 };
    let (times, vals_v, vals_q) = (ch.times.clone(), ch.vec_values.clone(), ch.quat_values.clone());

    // Constant.
    let constant = if rot {
        vals_q.iter().all(|q| angle_between(*q, vals_q[0]).to_degrees() <= CONSTANT_DEG)
    } else {
        vals_v.iter().all(|v| (*v - vals_v[0]).abs().max_element() <= CONSTANT_EPS)
    };
    if constant {
        ch.times = vec![times[0]];
        if rot {
            ch.quat_values = vec![vals_q[0]];
        } else {
            ch.vec_values = vec![vals_v[0]];
        }
        return;
    }

    // Candidates: (bytes, builder result).
    struct Cand {
        bytes: usize,
        times: Vec<f32>,
        v: Vec<Vec3>,
        q: Vec<Quat>,
        packed: Option<PackedValues>,
    }
    let mut cands: Vec<Cand> = Vec::new();
    // Reduced at the full tolerance (what Keyframes makes), and with a margin so
    // that a quantised form of it still fits.
    let reduce = |scale: f32| {
        if rot {
            let (t, q) = reduce_quat(&times, &vals_q, rotation_deg * scale);
            (t, Vec::new(), q)
        } else {
            let (t, v) = reduce_vec3(&times, &vals_v, pct * scale);
            (t, v, Vec::new())
        }
    };
    let (rt, rv, rq) = reduce(1.0);
    let (mt, mv, mq) = reduce(0.75);
    let tol_v = if rot { 0.0 } else { vec_tolerance(&vals_v, pct) };
    let tol_q = rotation_deg.to_radians().max(1e-6);
    let within = |bt: &[f32], bv: &[Vec3], bq: &[Quat]| {
        if rot { quat_error(&times, &vals_q, bt, bq) <= tol_q } else { vec_error(&times, &vals_v, bt, bv) <= tol_v }
    };
    // Dense, full precision (always valid).
    cands.push(Cand { bytes: plain_bytes(n, lanes), times: times.clone(), v: vals_v.clone(), q: vals_q.clone(), packed: None });
    // Reduced, full precision.
    if rt.len() < n && within(&rt, &rv, &rq) {
        cands.push(Cand { bytes: plain_bytes(rt.len(), lanes), times: rt.clone(), v: rv.clone(), q: rq.clone(), packed: None });
    }
    // Dense quantised, and reduced quantised.
    for (bt, bv, bq) in [(&times, &vals_v, &vals_q), (&rt, &rv, &rq), (&mt, &mv, &mq)] {
        if bt.len() < 2 {
            continue;
        }
        let p = if rot { pack_quat(bq) } else { pack_vec(bv) };
        let (dv, dq) = if rot { (Vec::new(), decoded_quat(&p)) } else { (decoded_vec(&p), Vec::new()) };
        if within(bt, &dv, &dq) {
            cands.push(Cand { bytes: packed_bytes(bt.len(), lanes), times: bt.clone(), v: Vec::new(), q: Vec::new(), packed: Some(p) });
        }
    }
    let best = cands.into_iter().min_by_key(|c| c.bytes).expect("the dense candidate is always there");
    ch.times = best.times;
    match best.packed {
        Some(p) => {
            ch.vec_values = Vec::new();
            ch.quat_values = Vec::new();
            ch.extra = Some(Box::new(ChannelExtra { tangents: None, packed: Some(p) }));
        }
        None => {
            ch.vec_values = best.v;
            ch.quat_values = best.q;
        }
    }
}

/// Size accounting of a clip set, in stored bytes (see [`AnimationClip::stored_bytes`]).
pub fn total_bytes(clips: &[AnimationClip]) -> usize {
    clips.iter().map(AnimationClip::stored_bytes).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wave_channel(n: usize, path: ChannelPath) -> AnimChannel {
        let times: Vec<f32> = (0..n).map(|i| i as f32 / 30.0).collect();
        if path == ChannelPath::Rotation {
            let q = times.iter().map(|t| Quat::from_euler(glam::EulerRot::XYZ, (t * 3.0).sin() * 0.7, (t * 2.0).cos() * 0.4, t * 0.5)).collect();
            AnimChannel::rot(0, times, q)
        } else {
            let v = times.iter().map(|t| Vec3::new((t * 3.0).sin() * 40.0, t * 10.0, (t * 7.0).cos())).collect();
            AnimChannel::vec(0, path, times, v)
        }
    }

    #[test]
    fn quantisation_error_is_bounded_by_the_range() {
        let ch = wave_channel(120, ChannelPath::Translation);
        let p = pack_vec(&ch.vec_values);
        let worst = (0..p.keys()).map(|i| (p.vec3(i) - ch.vec_values[i]).abs().max_element()).fold(0.0, f32::max);
        let range = 80.0; // x spans +-40
        assert!(worst <= range / 65535.0, "{worst}");
        // A flat lane costs nothing and decodes exactly.
        let flat = pack_vec(&[Vec3::new(1.0, 2.0, 3.0); 4]);
        assert_eq!(flat.vec3(2), Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn packed_rotation_stays_unit_and_close() {
        let ch = wave_channel(120, ChannelPath::Rotation);
        let p = pack_quat(&ch.quat_values);
        for (i, q) in ch.quat_values.iter().enumerate() {
            let d = p.quat(i);
            assert!((d.length() - 1.0).abs() < 1e-5);
            assert!(angle_between(d, *q).to_degrees() < 0.01, "key {i}");
        }
    }

    #[test]
    fn optimal_picks_constant_reduced_or_quantised_by_size() {
        let mut clip = AnimationClip { name: "c".into(), duration: 4.0, channels: Vec::new(), morph_tracks: Vec::new(), playback: None };
        // A constant channel.
        let times: Vec<f32> = (0..120).map(|i| i as f32 / 30.0).collect();
        clip.channels.push(AnimChannel::vec(0, ChannelPath::Translation, times.clone(), vec![Vec3::new(1.0, 2.0, 3.0); 120]));
        // A straight line: reduces to its ends.
        clip.channels.push(AnimChannel::vec(1, ChannelPath::Translation, times.clone(), times.iter().map(|t| Vec3::new(*t, 0.0, 0.0)).collect()));
        // A busy curve: neither constant nor reducible; quantising halves it.
        let busy: Vec<Vec3> = (0..120).map(|i| Vec3::new((i as f32 * 1.7).sin(), (i as f32 * 2.3).cos(), (i as f32 * 0.9).sin())).collect();
        clip.channels.push(AnimChannel::vec(2, ChannelPath::Translation, times.clone(), busy.clone()));
        let before = clip.stored_bytes();
        compress_clip(&mut clip, AnimCompression::Optimal, 0.5, 0.5, 0.5);
        assert_eq!(clip.channels[0].len(), 1, "constant");
        assert_eq!(clip.channels[1].len(), 2, "reduced line");
        assert_eq!(clip.channels[2].len(), 120);
        assert!(clip.channels[2].is_packed(), "dense quantised wins on a busy curve");
        assert!(clip.stored_bytes() < before / 2, "{} vs {before}", clip.stored_bytes());
        // The packed channel samples within the tolerance of the original.
        let tol = 2.0 * 0.5 / 100.0; // range 2 x 0.5 %
        for (i, t) in times.iter().enumerate() {
            let v = clip.channels[2].sample_vec(*t).unwrap();
            assert!((v - busy[i]).abs().max_element() <= tol, "{i}");
        }
    }

    #[test]
    fn modes_differ_as_documented() {
        let make = || AnimationClip { name: "c".into(), duration: 4.0, channels: vec![wave_channel(120, ChannelPath::Rotation), wave_channel(120, ChannelPath::Translation)], morph_tracks: Vec::new(), playback: None };
        let dense = make();
        let mut kr = make();
        compress_clip(&mut kr, AnimCompression::KeyframeReduction, 0.5, 0.5, 0.5);
        let mut krc = make();
        compress_clip(&mut krc, AnimCompression::KeyframeReductionAndCompression, 0.5, 0.5, 0.5);
        let mut opt = make();
        compress_clip(&mut opt, AnimCompression::Optimal, 0.5, 0.5, 0.5);
        assert!(kr.channels.iter().all(|c| !c.is_packed()));
        assert!(krc.channels.iter().all(|c| c.is_packed() || c.len() < 3));
        assert!(kr.stored_bytes() < dense.stored_bytes());
        assert!(krc.stored_bytes() < kr.stored_bytes(), "quantising the reduced keys is smaller still");
        assert!(opt.stored_bytes() <= krc.stored_bytes() + 8, "optimal never loses to the fixed combination: {} vs {}", opt.stored_bytes(), krc.stored_bytes());
        // Off leaves everything.
        let mut off = make();
        compress_clip(&mut off, AnimCompression::Off, 0.5, 0.5, 0.5);
        assert_eq!(off.stored_bytes(), dense.stored_bytes());
        // Error against the dense original, every mode.
        for c in [&kr, &krc, &opt] {
            for (a, b) in dense.channels.iter().zip(&c.channels) {
                for (i, t) in a.times.iter().enumerate() {
                    if a.path == ChannelPath::Rotation {
                        let q = b.sample_quat(*t).unwrap();
                        assert!(angle_between(q, a.quat_values[i]).to_degrees() <= 0.5 + 0.01, "rot key {i}");
                    }
                }
            }
        }
    }

    #[test]
    fn unpack_gives_back_plain_keys_and_scaling_a_packed_channel_works() {
        let mut ch = wave_channel(60, ChannelPath::Translation);
        let orig = ch.vec_values.clone();
        assert!(try_pack(&mut ch, 0.5, 0.5));
        assert!(ch.is_packed() && ch.vec_values.is_empty());
        ch.scale_translation(0.01);
        let s = ch.vec_at(10).unwrap();
        assert!((s - orig[10] * 0.01).abs().max_element() < 1e-3, "{s:?}");
        ch.unpack();
        assert!(!ch.is_packed() && ch.vec_values.len() == 60);
        let mut neg = wave_channel(60, ChannelPath::Translation);
        try_pack(&mut neg, 0.5, 0.5);
        neg.scale_translation(-1.0);
        assert!(!neg.is_packed(), "a mirroring scale falls back to plain keys");
    }

    #[test]
    fn hermite_tangents_follow_a_cubic_better_than_linear_keys() {
        use crate::skeleton::{ChannelExtra, Tangents};
        // x(t) = t^3 over [0, 1] with keys at 0 and 1: tangents 0 and 3.
        let mut ch = AnimChannel::vec(0, ChannelPath::Translation, vec![0.0, 1.0], vec![Vec3::ZERO, Vec3::X]);
        let lin = ch.sample_vec(0.5).unwrap().x;
        ch.extra = Some(Box::new(ChannelExtra { tangents: Some(Tangents { tan_in: vec![Vec3::ZERO, Vec3::X * 3.0], tan_out: vec![Vec3::ZERO, Vec3::X * 3.0] }), packed: None }));
        let herm = ch.sample_vec(0.5).unwrap().x;
        assert!((herm - 0.125).abs() < 1e-5, "{herm}");
        assert!((lin - 0.125).abs() > 0.3);
    }

    #[test]
    #[ignore = "timing"]
    fn sampling_speed_dense_vs_packed() {
        let mut dense = AnimationClip { name: "d".into(), duration: 4.0, channels: Vec::new(), morph_tracks: Vec::new(), playback: None };
        for j in 0..60 {
            let mut r = wave_channel(120, ChannelPath::Rotation);
            r.joint = j;
            let mut t = wave_channel(120, ChannelPath::Translation);
            t.joint = j;
            dense.channels.push(r);
            dense.channels.push(t);
        }
        let mut packed = dense.clone();
        for c in &mut packed.channels {
            try_pack(c, 0.5, 0.5);
        }
        let bench = |clip: &AnimationClip| {
            let t0 = std::time::Instant::now();
            let mut acc = 0.0f32;
            for k in 0..2000 {
                let t = (k as f32 * 0.0137) % 4.0;
                for ch in &clip.channels {
                    if ch.path == ChannelPath::Rotation {
                        acc += ch.sample_quat(t).unwrap().w;
                    } else {
                        acc += ch.sample_vec(t).unwrap().x;
                    }
                }
            }
            (t0.elapsed().as_secs_f64() * 1e9 / (2000.0 * clip.channels.len() as f64), acc)
        };
        let (d, _) = bench(&dense);
        let (p, _) = bench(&packed);
        println!("sample cost per channel: dense {d:.1} ns, packed {p:.1} ns, bytes {} -> {}", dense.stored_bytes(), packed.stored_bytes());
    }
}
