//! Clip motion analysis for the blend tree editor (AUI-2): the average
//! velocity, speed and turning rate of a clip's root, used by Compute
//! Thresholds / Positions, and the time scales of Adjust Time Scale >
//! Homogeneous Speed.
//!
//! The root is the node that carries the clip's root motion: for an imported
//! clip the skeleton root (or the Hips when the root is not animated), for an
//! authored clip the track target `""`, else a node named like a root, else
//! Hips / Pelvis, else the shortest target path. Velocity is the root's
//! displacement over the whole clip divided by its length (in the root's
//! parent space, +Z forward, +X left); the turning rate is the net yaw about
//! +Y over the clip divided by its length, unwrapped step by step so a spin
//! of more than half a turn is counted whole.

use std::collections::HashMap;

use glam::{Quat, Vec3};
use zestengine_assets::{AnimClip, AnimationClip, ChannelPath, TrackProperty};
use zestengine_core::{BlendMotion, BlendTree};

/// Samples per clip: enough that a half-turn per step never aliases.
const STEPS: usize = 64;

/// What a clip's root does on average.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClipMotion {
    /// Clip length in seconds.
    pub duration: f32,
    /// Net displacement per second.
    pub velocity: Vec3,
    /// Net yaw per second, radians (counter-clockwise seen from above is positive).
    pub angular_speed: f32,
}

impl ClipMotion {
    pub fn speed(&self) -> f32 {
        self.velocity.length()
    }

    /// From root samples spaced evenly over `0..=duration` (`STEPS + 1` or any count >= 2).
    pub fn from_samples(duration: f32, pos: &[Vec3], rot: &[Quat]) -> Self {
        let d = duration.max(1e-4);
        let (Some(p0), Some(p1)) = (pos.first(), pos.last()) else {
            return Self { duration, ..Default::default() };
        };
        let mut yaw = 0.0;
        for w in rot.windows(2) {
            let delta = (w[1] * w[0].inverse()).normalize();
            // Twist about +Y of the step, shortest way round.
            let mut a = 2.0 * delta.y.atan2(delta.w);
            if a > std::f32::consts::PI {
                a -= std::f32::consts::TAU;
            } else if a < -std::f32::consts::PI {
                a += std::f32::consts::TAU;
            }
            yaw += a;
        }
        Self { duration, velocity: (*p1 - *p0) / d, angular_speed: yaw / d }
    }
}

#[allow(dead_code)] // keyframe lerp used by clip playback tests
fn lerp_keys<T: Copy>(times: &[f32], values: &[T], t: f32, mix: impl Fn(T, T, f32) -> T) -> Option<T> {
    let n = times.len().min(values.len());
    if n == 0 {
        return None;
    }
    if n == 1 || t <= times[0] {
        return Some(values[0]);
    }
    if t >= times[n - 1] {
        return Some(values[n - 1]);
    }
    let i = times[..n].partition_point(|&k| k <= t).saturating_sub(1).min(n - 2);
    let h = (times[i + 1] - times[i]).max(1e-6);
    Some(mix(values[i], values[i + 1], ((t - times[i]) / h).clamp(0.0, 1.0)))
}

/// Analyse an imported clip through the channels of joint `root`.
/// `None` when the joint has neither a translation nor a rotation channel.
pub fn analyse_imported(clip: &AnimationClip, root: usize) -> Option<ClipMotion> {
    let find = |path: ChannelPath| clip.channels.iter().find(|c| c.joint == root && c.path == path && !c.times.is_empty());
    let (tc, rc) = (find(ChannelPath::Translation), find(ChannelPath::Rotation));
    if tc.is_none() && rc.is_none() {
        return None;
    }
    let at = |k: usize| clip.duration * k as f32 / STEPS as f32;
    let pos: Vec<Vec3> = (0..=STEPS)
        .map(|k| tc.and_then(|c| c.sample_vec(at(k))).unwrap_or(Vec3::ZERO))
        .collect();
    let rot: Vec<Quat> = (0..=STEPS)
        .map(|k| rc.and_then(|c| c.sample_quat(at(k))).unwrap_or(Quat::IDENTITY))
        .collect();
    Some(ClipMotion::from_samples(clip.duration, &pos, &rot))
}

/// The track target that carries an authored clip's root motion.
pub fn authored_root(clip: &AnimClip) -> Option<&str> {
    let targets: Vec<&str> = clip
        .tracks
        .iter()
        .filter(|t| matches!(t.property, TrackProperty::Translation | TrackProperty::Rotation))
        .map(|t| t.target.as_str())
        .collect();
    let lower = |s: &&str| crate::anim_graph::loose_bone_name(s.rsplit('/').next().unwrap_or(s));
    targets
        .iter()
        .copied()
        .find(|t| t.is_empty())
        .or_else(|| targets.iter().copied().find(|t| lower(t).contains("root")))
        .or_else(|| targets.iter().copied().find(|t| matches!(lower(t).as_str(), "hips" | "pelvis")))
        .or_else(|| targets.iter().copied().min_by_key(|t| t.matches('/').count()))
}

/// Analyse an authored clip. `None` without translation / rotation tracks.
pub fn analyse_authored(clip: &AnimClip) -> Option<ClipMotion> {
    let root = authored_root(clip)?;
    let tr = clip.track_index(root, TrackProperty::Translation).filter(|&i| !clip.tracks[i].keys.is_empty());
    let ro = clip.track_index(root, TrackProperty::Rotation).filter(|&i| !clip.tracks[i].keys.is_empty());
    if tr.is_none() && ro.is_none() {
        return None;
    }
    let at = |k: usize| clip.length * k as f32 / STEPS as f32;
    let pos: Vec<Vec3> = (0..=STEPS)
        .map(|k| tr.map(|i| clip.tracks[i].sample(at(k))).map(|v| Vec3::new(v[0], v[1], v[2])).unwrap_or(Vec3::ZERO))
        .collect();
    let rot: Vec<Quat> = (0..=STEPS)
        .map(|k| ro.map(|i| Quat::from_array(clip.tracks[i].sample(at(k))).normalize()).unwrap_or(Quat::IDENTITY))
        .collect();
    Some(ClipMotion::from_samples(clip.length, &pos, &rot))
}

/// What Compute Thresholds / Positions reads from each clip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeFrom {
    Speed,
    VelocityX,
    VelocityY,
    VelocityZ,
    AngularSpeedRad,
    AngularSpeedDeg,
    /// 2D: X = velocity X, Y = velocity Z.
    VelocityXZ,
    /// 2D: X = angular speed (radians per second), Y = speed.
    SpeedAndAngular,
}

impl ComputeFrom {
    pub const ONE_D: [ComputeFrom; 6] = [Self::Speed, Self::VelocityX, Self::VelocityY, Self::VelocityZ, Self::AngularSpeedRad, Self::AngularSpeedDeg];
    pub const TWO_D: [ComputeFrom; 2] = [Self::VelocityXZ, Self::SpeedAndAngular];

    pub fn label(self) -> &'static str {
        match self {
            Self::Speed => "Speed",
            Self::VelocityX => "Velocity X",
            Self::VelocityY => "Velocity Y",
            Self::VelocityZ => "Velocity Z",
            Self::AngularSpeedRad => "Angular Speed (rad/s)",
            Self::AngularSpeedDeg => "Angular Speed (deg/s)",
            Self::VelocityXZ => "Velocity XZ",
            Self::SpeedAndAngular => "Speed and Angular Speed",
        }
    }

    /// The `[x, y]` the clip gives (1D trees read `x`).
    pub fn value(self, m: &ClipMotion) -> [f32; 2] {
        match self {
            Self::Speed => [m.speed(), 0.0],
            Self::VelocityX => [m.velocity.x, 0.0],
            Self::VelocityY => [m.velocity.y, 0.0],
            Self::VelocityZ => [m.velocity.z, 0.0],
            Self::AngularSpeedRad => [m.angular_speed, 0.0],
            Self::AngularSpeedDeg => [m.angular_speed.to_degrees(), 0.0],
            Self::VelocityXZ => [m.velocity.x, m.velocity.z],
            Self::SpeedAndAngular => [m.angular_speed, m.speed()],
        }
    }
}

/// Set each child's threshold / position from its clip's motion. Children
/// that are nested trees or whose clip has no analysis keep their values. A
/// 1D tree is then re-sorted by threshold. Returns whether anything changed.
pub fn apply_compute(tree: &mut BlendTree, motions: &HashMap<String, ClipMotion>, from: ComputeFrom) -> bool {
    let mut changed = false;
    let two_d = tree.effective_kind().is_2d();
    for m in &mut tree.motions {
        if m.tree.is_some() {
            continue;
        }
        let Some(cm) = motions.get(&m.clip) else { continue };
        let v = from.value(cm);
        let new = if two_d { v } else { [v[0], m.position[1]] };
        if new != m.position {
            m.position = new;
            changed = true;
        }
    }
    if changed && !two_d {
        tree.motions.sort_by(|a: &BlendMotion, b: &BlendMotion| a.position[0].total_cmp(&b.position[0]));
    }
    changed
}

/// Time scales that give every child the same duration (the mean of the known
/// lengths): `length / mean`. `None` for children without a known length.
pub fn homogeneous_time_scales(lengths: &[Option<f32>]) -> Vec<Option<f32>> {
    let known: Vec<f32> = lengths.iter().flatten().copied().filter(|l| *l > 1e-4).collect();
    if known.is_empty() {
        return vec![None; lengths.len()];
    }
    let mean = known.iter().sum::<f32>() / known.len() as f32;
    lengths.iter().map(|l| l.filter(|l| *l > 1e-4).map(|l| l / mean)).collect()
}

/// Adjust Time Scale > Homogeneous Speed on a tree's direct clip children.
pub fn apply_homogeneous(tree: &mut BlendTree, lengths: &HashMap<String, f32>) -> bool {
    let idx: Vec<usize> = (0..tree.motions.len()).filter(|&k| tree.motions[k].tree.is_none()).collect();
    let lens: Vec<Option<f32>> = idx.iter().map(|&k| lengths.get(&tree.motions[k].clip).copied()).collect();
    let mut changed = false;
    for (&k, s) in idx.iter().zip(homogeneous_time_scales(&lens)) {
        if let Some(s) = s
            && (tree.motions[k].time_scale - s).abs() > 1e-6
        {
            tree.motions[k].time_scale = s;
            changed = true;
        }
    }
    changed
}

/// Every clip name a tree plays, nested trees included.
pub fn clip_names(tree: &BlendTree, out: &mut Vec<String>) {
    for m in &tree.motions {
        match &m.tree {
            Some(t) => clip_names(t, out),
            None if !m.clip.is_empty() && !out.contains(&m.clip) => out.push(m.clip.clone()),
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_assets::{AnimChannel, AnimTrack, KeyInterp, Keyframe};
    use zestengine_core::BlendTreeKind;

    fn imported(duration: f32, end: Vec3, turn: f32) -> AnimationClip {
        AnimationClip {
            name: "c".into(),
            duration,
            channels: vec![
                AnimChannel { joint: 0, path: ChannelPath::Translation, times: vec![0.0, duration], vec_values: vec![Vec3::ZERO, end], quat_values: vec![], extra: None },
                AnimChannel {
                    joint: 0,
                    path: ChannelPath::Rotation,
                    times: vec![0.0, duration / 2.0, duration],
                    vec_values: vec![],
                    quat_values: vec![Quat::IDENTITY, Quat::from_rotation_y(turn / 2.0), Quat::from_rotation_y(turn)], extra: None
                },
            ],
            morph_tracks: Vec::new(), playback: None,
        }
    }

    #[test]
    fn imported_velocity_and_turn_rate() {
        let m = analyse_imported(&imported(2.0, Vec3::new(1.0, 0.0, 4.0), 1.0), 0).unwrap();
        assert!((m.velocity - Vec3::new(0.5, 0.0, 2.0)).length() < 1e-4);
        assert!((m.speed() - 0.5_f32.hypot(2.0)).abs() < 1e-4);
        assert!((m.angular_speed - 0.5).abs() < 1e-3, "{}", m.angular_speed);
        // A joint with no channels has nothing to analyse.
        assert!(analyse_imported(&imported(1.0, Vec3::X, 0.0), 3).is_none());
    }

    #[test]
    fn full_spins_are_counted_whole() {
        // 1.5 turns as quarter-turn keys (a slerp between keys never exceeds half a turn).
        let quarter = std::f32::consts::FRAC_PI_2;
        let spin = |dir: f32| AnimationClip {
            name: "spin".into(),
            duration: 1.0,
            channels: vec![AnimChannel {
                joint: 0,
                path: ChannelPath::Rotation,
                times: (0..=6).map(|k| k as f32 / 6.0).collect(),
                vec_values: vec![],
                quat_values: (0..=6).map(|k| Quat::from_rotation_y(dir * quarter * k as f32)).collect(), extra: None
            }],
            morph_tracks: Vec::new(), playback: None,
        };
        let m = analyse_imported(&spin(1.0), 0).unwrap();
        assert!((m.angular_speed - quarter * 6.0).abs() < 0.05, "{}", m.angular_speed);
        let back = analyse_imported(&spin(-1.0), 0).unwrap();
        assert!((back.angular_speed + quarter * 6.0).abs() < 0.05);
        assert_eq!(m.velocity, Vec3::ZERO);
    }

    fn track(target: &str, prop: TrackProperty, vals: &[(f32, [f32; 4])]) -> AnimTrack {
        AnimTrack {
            target: target.into(),
            property: prop,
            name: String::new(),
            slot: 0,
            keys: vals.iter().map(|(t, v)| Keyframe { time: *t, value: *v, interp: KeyInterp::Linear, ..Default::default() }).collect(),
        }
    }

    #[test]
    fn authored_clip_picks_the_root_track() {
        let clip = AnimClip {
            length: 2.0,
            tracks: vec![
                track("Hips/Spine", TrackProperty::Translation, &[(0.0, [0.0; 4]), (2.0, [9.0, 0.0, 0.0, 0.0])]),
                track("Root", TrackProperty::Translation, &[(0.0, [0.0; 4]), (2.0, [0.0, 0.0, 6.0, 0.0])]),
            ],
            ..Default::default()
        };
        assert_eq!(authored_root(&clip), Some("Root"));
        let m = analyse_authored(&clip).unwrap();
        assert!((m.velocity - Vec3::new(0.0, 0.0, 3.0)).length() < 1e-4);
        let empty_target = AnimClip { tracks: vec![track("", TrackProperty::Translation, &[(0.0, [0.0; 4]), (1.0, [2.0, 0.0, 0.0, 0.0])])], ..Default::default() };
        assert_eq!(authored_root(&empty_target), Some(""));
        assert!(analyse_authored(&AnimClip::default()).is_none());
        let scale_only = AnimClip { tracks: vec![track("", TrackProperty::Scale, &[(0.0, [1.0; 4])])], ..Default::default() };
        assert!(analyse_authored(&scale_only).is_none());
    }

    fn tree(kind: BlendTreeKind, clips: &[&str]) -> BlendTree {
        BlendTree {
            kind,
            parameter: "P".into(),
            parameter_y: if kind == BlendTreeKind::Simple1D { String::new() } else { "Q".into() },
            motions: clips.iter().map(|c| BlendMotion { clip: (*c).into(), ..Default::default() }).collect(),
            ..Default::default()
        }
    }

    fn motions() -> HashMap<String, ClipMotion> {
        let m = |v: Vec3, a: f32| ClipMotion { duration: 1.0, velocity: v, angular_speed: a };
        HashMap::from([
            ("walk".to_string(), m(Vec3::new(0.0, 0.0, 1.5), 0.0)),
            ("run".to_string(), m(Vec3::new(0.0, 0.0, 4.0), 0.0)),
            ("turn".to_string(), m(Vec3::ZERO, 1.0)),
        ])
    }

    #[test]
    fn compute_1d_sets_thresholds_and_sorts() {
        let mut t = tree(BlendTreeKind::Simple1D, &["run", "mystery", "walk"]);
        assert!(apply_compute(&mut t, &motions(), ComputeFrom::Speed));
        let order: Vec<(&str, f32)> = t.motions.iter().map(|m| (m.clip.as_str(), m.position[0])).collect();
        // "mystery" has no analysis: keeps 0 and sorts first.
        assert_eq!(order, vec![("mystery", 0.0), ("walk", 1.5), ("run", 4.0)]);
        assert!(!apply_compute(&mut t, &motions(), ComputeFrom::Speed), "second run changes nothing");
        apply_compute(&mut t, &motions(), ComputeFrom::AngularSpeedDeg);
        assert!(t.motions.iter().all(|m| m.position[0].abs() < 1e-3 || m.clip == "turn"));
        let mut t = tree(BlendTreeKind::Simple1D, &["turn"]);
        apply_compute(&mut t, &motions(), ComputeFrom::AngularSpeedDeg);
        assert!((t.motions[0].position[0] - 57.2958).abs() < 1e-2);
    }

    #[test]
    fn compute_2d_sets_both_axes() {
        let mut t = tree(BlendTreeKind::FreeformCartesian2D, &["walk", "turn"]);
        assert!(apply_compute(&mut t, &motions(), ComputeFrom::VelocityXZ));
        assert_eq!(t.motions[0].position, [0.0, 1.5]);
        assert_eq!(t.motions[1].position, [0.0, 0.0]);
        apply_compute(&mut t, &motions(), ComputeFrom::SpeedAndAngular);
        assert_eq!(t.motions[0].position, [0.0, 1.5]);
        assert_eq!(t.motions[1].position, [1.0, 0.0]);
    }

    #[test]
    fn homogeneous_speed_equalises_durations() {
        // Lengths 1, 2, 3 -> mean 2: scales 0.5, 1, 1.5; every child then lasts 2 s.
        let s = homogeneous_time_scales(&[Some(1.0), Some(2.0), None, Some(3.0)]);
        assert_eq!(s, vec![Some(0.5), Some(1.0), None, Some(1.5)]);
        let mut t = tree(BlendTreeKind::Simple1D, &["a", "b", "c"]);
        let lens = HashMap::from([("a".to_string(), 1.0), ("b".to_string(), 2.0), ("c".to_string(), 3.0)]);
        assert!(apply_homogeneous(&mut t, &lens));
        for (m, l) in t.motions.iter().zip([1.0, 2.0, 3.0]) {
            assert!((l / m.time_scale - 2.0).abs() < 1e-5);
        }
        assert!(!apply_homogeneous(&mut t, &lens));
        assert_eq!(homogeneous_time_scales(&[None, Some(0.0)]), vec![None, None]);
    }

    #[test]
    fn clip_names_walk_nested_trees() {
        let mut t = tree(BlendTreeKind::Simple1D, &["a", "b"]);
        t.motions[1].tree = Some(Box::new(tree(BlendTreeKind::Simple1D, &["c", "a"])));
        let mut names = Vec::new();
        clip_names(&t, &mut names);
        assert_eq!(names, vec!["a".to_string(), "c".to_string()]);
    }
}
