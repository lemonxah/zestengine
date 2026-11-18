//! Authored cinematics / sequencer asset (`.seq`): scene-anchored tracks that
//! drive object transforms and cameras on a timeline. Unlike an `.anim` clip
//! (which targets an Animator's hierarchy by relative path), a sequencer
//! references scene objects by their stable [`ObjectId`], so the same sequence
//! survives scene reordering and loads.
//!
//! Each track is either a transform track (translation / rotation / scale over
//! an object) or a camera track (the same transforms plus an optional vertical
//! field-of-view curve). Keys interpolate linearly or cubically (automatic
//! Catmull-Rom tangents); rotation always uses shortest-path slerp.

use std::path::Path;

use anyhow::{Context as _, Result};
use glam::Quat;
use serde::{Deserialize, Serialize};
use zestengine_core::ObjectId;

/// File extension for a sequencer document.
pub const SEQUENCER_EXTENSION: &str = "seq";

/// The editable channel of a track.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum SeqChannel {
    #[default]
    Translation,
    Rotation,
    Scale,
    /// Camera only: vertical field of view in degrees (stored in lane `x`).
    Fov,
}

impl SeqChannel {
    pub const TRANSFORM: [SeqChannel; 3] = [Self::Translation, Self::Rotation, Self::Scale];

    pub fn label(self) -> &'static str {
        match self {
            Self::Translation => "Position",
            Self::Rotation => "Rotation",
            Self::Scale => "Scale",
            Self::Fov => "Field of View",
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SeqInterp {
    /// Hold this key's value until the next key.
    Step,
    Linear,
    /// Smooth Hermite curve with automatic (Catmull-Rom) tangents.
    #[default]
    Cubic,
}

impl SeqInterp {
    pub const ALL: [SeqInterp; 3] = [Self::Step, Self::Linear, Self::Cubic];

    pub fn label(self) -> &'static str {
        match self {
            Self::Step => "Constant",
            Self::Linear => "Linear",
            Self::Cubic => "Auto (cubic)",
        }
    }
}

/// Position/scale are `xyz`, rotation is a quaternion `xyzw`, and a scalar
/// (FOV) lives in lane `x`.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Default)]
#[serde(default)]
pub struct SeqKey {
    pub time: f32,
    pub value: [f32; 4],
    pub interp: SeqInterp,
}

/// A curve over one channel of a track, keyed by time.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default)]
pub struct SeqCurve {
    pub channel: SeqChannel,
    /// Sorted by time.
    pub keys: Vec<SeqKey>,
}

impl SeqCurve {
    pub fn new(channel: SeqChannel) -> Self {
        Self { channel, keys: Vec::new() }
    }

    pub fn sort(&mut self) {
        self.keys.sort_by(|a, b| a.time.total_cmp(&b.time));
    }

    pub fn set_key(&mut self, time: f32, value: [f32; 4]) -> usize {
        if let Some(i) = self.keys.iter().position(|k| (k.time - time).abs() < 5e-4) {
            self.keys[i].value = value;
            return i;
        }
        let interp = self.keys.last().map(|k| k.interp).unwrap_or_default();
        self.keys.push(SeqKey { time, value, interp });
        self.sort();
        self.keys.iter().position(|k| (k.time - time).abs() < 5e-4).unwrap_or(0)
    }

    pub fn sample(&self, t: f32) -> [f32; 4] {
        let keys = &self.keys;
        let Some(first) = keys.first() else {
            return [0.0; 4];
        };
        if keys.len() == 1 || t <= first.time {
            return first.value;
        }
        let last = keys[keys.len() - 1];
        if t >= last.time {
            return last.value;
        }
        let i = keys.partition_point(|k| k.time <= t).saturating_sub(1).min(keys.len() - 2);
        let (k0, k1) = (keys[i], keys[i + 1]);
        let h = (k1.time - k0.time).max(1e-6);
        let s = ((t - k0.time) / h).clamp(0.0, 1.0);
        let rotation = self.channel == SeqChannel::Rotation;
        match k0.interp {
            SeqInterp::Step => k0.value,
            SeqInterp::Linear => {
                if rotation {
                    Quat::from_array(k0.value).slerp(Quat::from_array(k1.value), s).normalize().to_array()
                } else {
                    lerp4(k0.value, k1.value, s)
                }
            }
            SeqInterp::Cubic => {
                let align = |reference: [f32; 4], v: [f32; 4]| {
                    if rotation && dot4(reference, v) < 0.0 { v.map(|x| -x) } else { v }
                };
                let p0 = k0.value;
                let p1 = align(p0, k1.value);
                let tangent = |j: usize, pj: [f32; 4]| -> [f32; 4] {
                    if j == 0 || j + 1 >= keys.len() {
                        return [0.0; 4];
                    }
                    let prev = align(pj, keys[j - 1].value);
                    let next = align(pj, keys[j + 1].value);
                    let dt = (keys[j + 1].time - keys[j - 1].time).max(1e-6);
                    std::array::from_fn(|c| (next[c] - prev[c]) / dt)
                };
                let m0 = tangent(i, p0);
                let m1 = tangent(i + 1, p1);
                let (s2, s3) = (s * s, s * s * s);
                let v: [f32; 4] = std::array::from_fn(|c| {
                    (2.0 * s3 - 3.0 * s2 + 1.0) * p0[c]
                        + (s3 - 2.0 * s2 + s) * h * m0[c]
                        + (-2.0 * s3 + 3.0 * s2) * p1[c]
                        + (s3 - s2) * h * m1[c]
                });
                if rotation { Quat::from_array(v).normalize().to_array() } else { v }
            }
        }
    }

    pub fn last_key_time(&self) -> f32 {
        self.keys.last().map(|k| k.time).unwrap_or(0.0)
    }
}

/// One row of the timeline: a transform track or a camera track.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[serde(rename_all = "snake_case")]
pub enum SeqTrackKind {
    #[default]
    Transform,
    Camera,
}

impl SeqTrackKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Transform => "Transform",
            Self::Camera => "Camera",
        }
    }

    pub fn is_camera(self) -> bool {
        matches!(self, Self::Camera)
    }
}

/// Flat on-disk shape for a track, so RON can deserialize the `kind` tag
/// without an internally-tagged enum.
#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct SeqTrack {
    pub kind: SeqTrackKind,
    pub target: ObjectId,
    pub translation: SeqCurve,
    pub rotation: SeqCurve,
    pub scale: SeqCurve,
    /// Camera only: `None` = keep the authored field of view.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fov: Option<SeqCurve>,
}

impl Default for SeqTrack {
    fn default() -> Self {
        Self {
            kind: SeqTrackKind::Transform,
            target: ObjectId::NIL,
            translation: SeqCurve::new(SeqChannel::Translation),
            rotation: SeqCurve::new(SeqChannel::Rotation),
            scale: SeqCurve::new(SeqChannel::Scale),
            fov: None,
        }
    }
}

impl<'de> Deserialize<'de> for SeqTrack {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Default, Deserialize)]
        #[serde(default)]
        struct Wire {
            kind: SeqTrackKind,
            target: ObjectId,
            translation: SeqCurve,
            rotation: SeqCurve,
            scale: SeqCurve,
            fov: Option<SeqCurve>,
        }

        let wire = Wire::deserialize(deserializer)?;
        Ok(Self {
            kind: wire.kind,
            target: wire.target,
            translation: wire.translation,
            rotation: wire.rotation,
            scale: wire.scale,
            fov: if wire.kind.is_camera() { wire.fov } else { None },
        })
    }
}

impl SeqTrack {
    pub fn new(kind: SeqTrackKind, target: ObjectId) -> Self {
        Self {
            kind,
            target,
            translation: SeqCurve::new(SeqChannel::Translation),
            rotation: SeqCurve::new(SeqChannel::Rotation),
            scale: SeqCurve::new(SeqChannel::Scale),
            fov: kind.is_camera().then(|| SeqCurve::new(SeqChannel::Fov)),
        }
    }

    pub fn target(&self) -> ObjectId {
        self.target
    }

    pub fn kind_label(&self) -> &'static str {
        self.kind.label()
    }

    pub fn is_camera(&self) -> bool {
        self.kind.is_camera()
    }

    /// The channels this track exposes, in display order.
    pub fn channels(&self) -> Vec<SeqChannel> {
        let mut out = SeqChannel::TRANSFORM.to_vec();
        if self.is_camera() && self.fov.is_some() {
            out.push(SeqChannel::Fov);
        }
        out
    }

    pub fn curve(&self, channel: SeqChannel) -> Option<&SeqCurve> {
        match channel {
            SeqChannel::Translation => Some(&self.translation),
            SeqChannel::Rotation => Some(&self.rotation),
            SeqChannel::Scale => Some(&self.scale),
            SeqChannel::Fov => self.fov.as_ref(),
        }
    }

    pub fn curve_mut(&mut self, channel: SeqChannel) -> Option<&mut SeqCurve> {
        match channel {
            SeqChannel::Translation => Some(&mut self.translation),
            SeqChannel::Rotation => Some(&mut self.rotation),
            SeqChannel::Scale => Some(&mut self.scale),
            SeqChannel::Fov => self.fov.as_mut(),
        }
    }

    /// Add a FOV curve to a camera track (no-op on a transform track).
    pub fn ensure_fov(&mut self) {
        if self.is_camera() {
            self.fov.get_or_insert_with(|| SeqCurve::new(SeqChannel::Fov));
        }
    }

    /// Remove the FOV curve from a camera track.
    pub fn remove_fov(&mut self) {
        self.fov = None;
    }

    pub fn sort(&mut self) {
        for channel in self.channels() {
            if let Some(curve) = self.curve_mut(channel) {
                curve.sort();
            }
        }
    }
}

/// A whole sequencer document: a timeline of scene-anchored tracks.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default)]
pub struct SeqEvent {
    pub time: f32,
    pub name: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(default)]
pub struct SeqCut {
    pub time: f32,
    pub camera: ObjectId,
    #[serde(skip_serializing_if = "blend_is_off")]
    pub blend: f32,
}

fn blend_is_off(seconds: &f32) -> bool {
    *seconds <= 0.0
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeqCameraBlend {
    pub from: ObjectId,
    pub to: ObjectId,
    pub weight: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct SeqAudio {
    pub time: f32,
    pub source: ObjectId,
    pub clip: String,
    pub volume: f32,
}

impl Default for SeqAudio {
    fn default() -> Self {
        Self { time: 0.0, source: ObjectId::NIL, clip: String::new(), volume: 1.0 }
    }
}

fn cues_between<T>(cues: &[T], time_of: impl Fn(&T) -> f32, from: f32, to: f32) -> Vec<&T> {
    let mut first: Vec<&T> = if to >= from {
        cues.iter().filter(|c| time_of(c) > from && time_of(c) <= to).collect()
    } else {
        cues.iter().filter(|c| time_of(c) > from).collect()
    };
    first.sort_by(|a, b| time_of(a).total_cmp(&time_of(b)));
    if to < from {
        let mut wrapped: Vec<&T> = cues.iter().filter(|c| time_of(c) <= to).collect();
        wrapped.sort_by(|a, b| time_of(a).total_cmp(&time_of(b)));
        first.extend(wrapped);
    }
    first
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeqPose {
    pub target: ObjectId,
    pub translation: Option<[f32; 3]>,
    pub rotation: Option<[f32; 4]>,
    pub scale: Option<[f32; 3]>,
    pub fov: Option<f32>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct Sequencer {
    /// Seconds.
    pub length: f32,
    pub looping: bool,
    /// Editor snapping only.
    pub frame_rate: f32,
    pub tracks: Vec<SeqTrack>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<SeqEvent>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub cuts: Vec<SeqCut>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub audio: Vec<SeqAudio>,
}

impl Default for Sequencer {
    fn default() -> Self {
        Self { length: 5.0, looping: false, frame_rate: 30.0, tracks: Vec::new(), events: Vec::new(), cuts: Vec::new(), audio: Vec::new() }
    }
}

impl Sequencer {
    /// Wrap (looping) or clamp `t` into the sequence.
    pub fn local_time(&self, t: f32, looping: bool) -> f32 {
        let len = self.length.max(1e-4);
        if looping { t.rem_euclid(len) } else { t.clamp(0.0, len) }
    }

    pub fn poses(&self, t: f32) -> Vec<SeqPose> {
        self.tracks
            .iter()
            .map(|track| {
                let sample = |channel: SeqChannel| track.curve(channel).filter(|c| !c.keys.is_empty()).map(|c| c.sample(t));
                SeqPose {
                    target: track.target(),
                    translation: sample(SeqChannel::Translation).map(|v| [v[0], v[1], v[2]]),
                    rotation: sample(SeqChannel::Rotation).map(|v| Quat::from_array(v).normalize().to_array()),
                    scale: sample(SeqChannel::Scale).map(|v| [v[0], v[1], v[2]]),
                    fov: if track.is_camera() { sample(SeqChannel::Fov).map(|v| v[0]) } else { None },
                }
            })
            .collect()
    }

    pub fn camera_target(&self) -> Option<ObjectId> {
        self.tracks.iter().find(|track| track.is_camera()).map(SeqTrack::target)
    }

    pub fn events_between(&self, from: f32, to: f32) -> Vec<&str> {
        cues_between(&self.events, |e| e.time, from, to).into_iter().map(|e| e.name.as_str()).collect()
    }

    pub fn audio_between(&self, from: f32, to: f32) -> Vec<&SeqAudio> {
        cues_between(&self.audio, |a| a.time, from, to)
    }

    pub fn camera_at(&self, time: f32) -> Option<ObjectId> {
        let cut = self.cuts.iter().filter(|c| c.time <= time && !c.camera.is_nil()).max_by(|a, b| a.time.total_cmp(&b.time));
        match cut {
            Some(cut) => Some(cut.camera),
            None => self.camera_target().or_else(|| self.cuts.iter().filter(|c| !c.camera.is_nil()).min_by(|a, b| a.time.total_cmp(&b.time)).map(|c| c.camera)),
        }
    }

    pub fn camera_blend_at(&self, time: f32) -> Option<SeqCameraBlend> {
        let cut = self.cuts.iter().filter(|c| c.time <= time && !c.camera.is_nil()).max_by(|a, b| a.time.total_cmp(&b.time))?;
        if !(cut.blend > 0.0) || time >= cut.time + cut.blend {
            return None;
        }
        let earlier = self.cuts.iter().filter(|c| c.time < cut.time && !c.camera.is_nil()).max_by(|a, b| a.time.total_cmp(&b.time));
        let from = earlier.map(|c| c.camera).or_else(|| self.camera_target())?;
        if from == cut.camera {
            return None;
        }
        let x = ((time - cut.time) / cut.blend).clamp(0.0, 1.0);
        Some(SeqCameraBlend { from, to: cut.camera, weight: x * x * (3.0 - 2.0 * x) })
    }

    pub fn last_key_time(&self) -> f32 {
        self.tracks
            .iter()
            .flat_map(|track| track.channels().into_iter().map(move |channel| (track, channel)))
            .filter_map(|(track, channel)| track.curve(channel))
            .map(SeqCurve::last_key_time)
            .fold(0.0, f32::max)
    }
}

fn lerp4(a: [f32; 4], b: [f32; 4], s: f32) -> [f32; 4] {
    std::array::from_fn(|c| a[c] + (b[c] - a[c]) * s)
}

fn dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
    a.iter().zip(&b).map(|(x, y)| x * y).sum()
}

pub fn load_sequencer(path: impl AsRef<Path>) -> Result<Sequencer> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut sequencer: Sequencer =
        ron::from_str(&text).with_context(|| format!("parsing sequencer {}", path.display()))?;
    for track in &mut sequencer.tracks {
        track.sort();
    }
    Ok(sequencer)
}

pub fn save_sequencer(path: impl AsRef<Path>, sequencer: &Sequencer) -> Result<()> {
    let path = path.as_ref();
    let text = ron::ser::to_string_pretty(sequencer, ron::ser::PrettyConfig::default())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(hex: &str) -> ObjectId {
        hex.parse().unwrap()
    }

    #[test]
    fn events_are_reported_in_time_order_and_across_a_loop() {
        let seq = Sequencer {
            length: 4.0,
            events: vec![SeqEvent { time: 3.0, name: "late".into() }, SeqEvent { time: 1.0, name: "early".into() }, SeqEvent { time: 0.0, name: "start".into() }],
            ..Default::default()
        };
        assert_eq!(seq.events_between(0.0, 4.0), ["early", "late"]);
        assert_eq!(seq.events_between(-1.0, 1.0), ["start", "early"]);
        assert_eq!(seq.events_between(1.0, 3.0), ["late"], "an event at the start of the span already fired");
        assert_eq!(seq.events_between(3.5, 1.0), ["start", "early"], "a span that wraps takes the tail, then the head");
        assert_eq!(seq.events_between(2.0, 0.5), ["late", "start"]);
        assert!(seq.events_between(3.0, 3.0).is_empty());
        let text = ron::to_string(&seq).unwrap();
        assert_eq!(ron::from_str::<Sequencer>(&text).unwrap(), seq);
        let old: Sequencer = ron::from_str("(length: 2.0, looping: true)").unwrap();
        assert!(old.events.is_empty());
        assert!(!ron::to_string(&old).unwrap().contains("events"));
    }

    #[test]
    fn cuts_pick_the_camera_for_a_time_and_audio_cues_fire_like_events() {
        let (a, b) = (id("0000000000000000000000000000000a"), id("0000000000000000000000000000000b"));
        let mut seq = Sequencer { length: 6.0, tracks: vec![SeqTrack::new(SeqTrackKind::Transform, a), SeqTrack::new(SeqTrackKind::Camera, a)], ..Default::default() };
        assert_eq!(seq.camera_at(3.0), Some(a), "without cuts the first camera track is the camera");
        seq.cuts = vec![SeqCut { time: 4.0, camera: a, blend: 0.0 }, SeqCut { time: 2.0, camera: b, blend: 1.0 }, SeqCut { time: 5.0, camera: ObjectId::NIL, blend: 0.0 }];
        assert_eq!(seq.camera_at(0.5), Some(a), "before the first cut: the first camera track");
        assert_eq!(seq.camera_at(2.0), Some(b));
        assert_eq!(seq.camera_at(3.9), Some(b));
        assert_eq!(seq.camera_at(5.5), Some(a), "a cut without a camera is ignored");
        assert_eq!(seq.camera_blend_at(1.9), None, "no blend before the cut");
        let start = seq.camera_blend_at(2.0).unwrap();
        assert_eq!((start.from, start.to, start.weight), (a, b, 0.0), "a blended cut starts on the camera it leaves");
        let half = seq.camera_blend_at(2.5).unwrap();
        assert!((half.weight - 0.5).abs() < 1e-5);
        assert!(seq.camera_blend_at(2.9).unwrap().weight > 0.9, "the blend eases in and out");
        assert_eq!(seq.camera_blend_at(3.0), None, "the blend is over after its length");
        assert_eq!(seq.camera_blend_at(4.5), None, "a hard cut has no blend");
        let text = ron::to_string(&seq).unwrap();
        assert_eq!(text.matches("blend").count(), 1, "hard cuts do not write a blend: {text}");
        seq.tracks.clear();
        assert_eq!(seq.camera_at(0.5), Some(b), "no camera track: the earliest cut's camera from the start");
        assert_eq!(Sequencer::default().camera_at(1.0), None);

        seq.audio = vec![SeqAudio { time: 3.0, source: a, ..Default::default() }, SeqAudio { time: 1.0, clip: "audio/slam.wav".into(), volume: 0.5, ..Default::default() }];
        assert_eq!(seq.audio_between(0.0, 2.0).iter().map(|c| c.clip.as_str()).collect::<Vec<_>>(), ["audio/slam.wav"]);
        assert_eq!(seq.audio_between(2.0, 6.0).len(), 1);
        assert_eq!(seq.audio_between(5.0, 1.5).len(), 1, "across the loop point");
        let text = ron::to_string(&seq).unwrap();
        assert_eq!(ron::from_str::<Sequencer>(&text).unwrap(), seq);
        let old: Sequencer = ron::from_str("(length: 2.0)").unwrap();
        assert!(old.cuts.is_empty() && old.audio.is_empty() && !ron::to_string(&old).unwrap().contains("cuts"));
    }

    #[test]
    fn poses_sample_every_track_and_only_cameras_carry_a_field_of_view() {
        let mut slide = SeqTrack::new(SeqTrackKind::Transform, id("00000000000000000000000000000001"));
        slide.translation.set_key(0.0, [0.0, 0.0, 0.0, 0.0]);
        slide.translation.set_key(2.0, [4.0, 2.0, 0.0, 0.0]);
        let mut shot = SeqTrack::new(SeqTrackKind::Camera, id("00000000000000000000000000000002"));
        shot.ensure_fov();
        shot.fov.as_mut().unwrap().set_key(0.0, [60.0, 0.0, 0.0, 0.0]);
        shot.fov.as_mut().unwrap().set_key(2.0, [40.0, 0.0, 0.0, 0.0]);
        let seq = Sequencer { tracks: vec![slide, shot], ..Default::default() };
        let poses = seq.poses(1.0);
        assert_eq!(poses.len(), 2);
        assert_eq!(poses[0].translation, Some([2.0, 1.0, 0.0]));
        assert_eq!((poses[0].rotation, poses[0].scale, poses[0].fov), (None, None, None), "channels without keys leave the object alone");
        assert_eq!(poses[1].translation, None);
        assert!((poses[1].fov.unwrap() - 50.0).abs() < 1e-3);
        assert_eq!(seq.camera_target(), Some(id("00000000000000000000000000000002")));
        assert_eq!(Sequencer::default().camera_target(), None);
    }

    #[test]
    fn curve_interpolates_channels() {
        let mut curve = SeqCurve::new(SeqChannel::Translation);
        curve.set_key(0.0, [0.0, 0.0, 0.0, 0.0]);
        curve.set_key(1.0, [2.0, 4.0, 6.0, 0.0]);
        assert_eq!(curve.sample(0.5), [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(curve.sample(-1.0), [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(curve.sample(2.0), [2.0, 4.0, 6.0, 0.0]);
    }

    #[test]
    fn rotation_shortest_path() {
        let mut curve = SeqCurve::new(SeqChannel::Rotation);
        curve.set_key(0.0, Quat::IDENTITY.to_array());
        curve.set_key(1.0, Quat::from_rotation_y(std::f32::consts::FRAC_PI_2).to_array());
        let mid = Quat::from_array(curve.sample(0.5)).normalize();
        let expect = Quat::from_rotation_y(std::f32::consts::FRAC_PI_4);
        assert!(mid.angle_between(expect) < 1e-3);
    }

    #[test]
    fn step_holds_until_next_key() {
        let mut curve = SeqCurve::new(SeqChannel::Fov);
        curve.keys.push(SeqKey { time: 0.0, value: [60.0, 0.0, 0.0, 0.0], interp: SeqInterp::Step });
        curve.keys.push(SeqKey { time: 1.0, value: [30.0, 0.0, 0.0, 0.0], interp: SeqInterp::Step });
        assert_eq!(curve.sample(0.9)[0], 60.0);
        assert_eq!(curve.sample(1.0)[0], 30.0);
    }

    #[test]
    fn ron_round_trips() {
        let target = id("01234567-89ab-cdef-0123-456789abcdef");
        let mut sequencer = Sequencer::default();
        sequencer.tracks.push(SeqTrack::new(SeqTrackKind::Transform, target));
        sequencer.tracks.push(SeqTrack::new(SeqTrackKind::Camera, target));
        let text = ron::ser::to_string_pretty(&sequencer, ron::ser::PrettyConfig::default()).unwrap();
        let back: Sequencer = ron::from_str(&text).unwrap();
        assert_eq!(back, sequencer);
    }
}
