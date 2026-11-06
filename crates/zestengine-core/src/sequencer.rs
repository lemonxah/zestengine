//! Cinematic sequencer: a shared timeline of tracks (camera, object, float,
//! event) with keyframe interpolation and playback.

use serde::{Deserialize, Serialize};

/// What a sequencer track drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SeqTrackKind {
    /// The camera track: positions the active camera.
    Camera,
    /// The object track: transforms one object.
    Object,
    /// A float track: drives one scalar (e.g. a light intensity).
    #[default]
    Float,
    /// An event track: fires named events at key times.
    Event,
}

/// One keyframe: `value` at `time` seconds.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SeqKey {
    /// Keyframe time (seconds).
    pub time: f32,
    /// The value at `time`.
    pub value: [f32; 4],
}

/// A sequencer track: keyframes sorted by time.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SeqTrack {
    /// Track name.
    pub name: String,
    /// What the track drives.
    pub kind: SeqTrackKind,
    /// Keyframes, kept sorted by time.
    pub keys: Vec<SeqKey>,
}

impl SeqTrack {
    /// Sort the keyframes by time.
    pub fn sort(&mut self) {
        self.keys.sort_by(|a, b| a.time.total_cmp(&b.time));
    }

    /// Sample the track at `time`: linear interpolation between keys, clamped
    /// to the first/last key. `None` for an empty track.
    pub fn sample(&self, time: f32) -> Option<[f32; 4]> {
        let keys = &self.keys;
        let first = keys.first()?;
        if keys.len() == 1 || time <= first.time {
            return Some(first.value);
        }
        let last = keys.last().unwrap();
        if time >= last.time {
            return Some(last.value);
        }
        let i = keys.partition_point(|k| k.time <= time).saturating_sub(1).min(keys.len() - 2);
        let (a, b) = (keys[i], keys[i + 1]);
        let t = ((time - a.time) / (b.time - a.time).max(1e-6)).clamp(0.0, 1.0);
        Some(std::array::from_fn(|c| a.value[c] + (b.value[c] - a.value[c]) * t))
    }
}

/// A cinematic sequencer: tracks over a shared timeline.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Sequencer {
    /// Timeline length (seconds).
    pub duration: f32,
    /// Loop the playback.
    pub looping: bool,
    /// The tracks.
    pub tracks: Vec<SeqTrack>,
    /// Playhead position (seconds).
    pub time: f32,
    /// Playback is running.
    pub playing: bool,
}

impl Default for Sequencer {
    fn default() -> Self {
        Self { duration: 10.0, looping: false, tracks: Vec::new(), time: 0.0, playing: false }
    }
}

impl Sequencer {
    /// Advance the playhead by `dt`: wrapped when looping, clamped (and then
    /// stopped) otherwise.
    pub fn advance(&mut self, dt: f32) {
        if !self.playing {
            return;
        }
        let len = self.duration.max(1e-4);
        self.time += dt;
        if self.looping {
            self.time = self.time.rem_euclid(len);
        } else {
            self.time = self.time.clamp(0.0, len);
            if self.time >= len {
                self.playing = false;
            }
        }
    }

    /// Events crossed between `from` and `to` (inclusive of `to`), as
    /// `(track name, time)`.
    pub fn events_between(&self, from: f32, to: f32) -> Vec<(String, f32)> {
        let mut out = Vec::new();
        for track in &self.tracks {
            if track.kind != SeqTrackKind::Event {
                continue;
            }
            for key in &track.keys {
                if key.time > from && key.time <= to {
                    out.push((track.name.clone(), key.time));
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn track(kind: SeqTrackKind, keys: &[(f32, f32)]) -> SeqTrack {
        SeqTrack {
            name: "t".into(),
            kind,
            keys: keys.iter().map(|&(t, v)| SeqKey { time: t, value: [v, 0.0, 0.0, 0.0] }).collect(),
        }
    }

    #[test]
    fn track_samples_linear_and_clamps() {
        let t = track(SeqTrackKind::Float, &[(0.0, 0.0), (2.0, 4.0)]);
        assert!((t.sample(1.0).unwrap()[0] - 2.0).abs() < 1e-5);
        assert_eq!(t.sample(-1.0).unwrap()[0], 0.0);
        assert_eq!(t.sample(5.0).unwrap()[0], 4.0);
        assert!(SeqTrack { name: String::new(), kind: SeqTrackKind::Float, keys: vec![] }.sample(0.0).is_none());
    }

    #[test]
    fn sequencer_clamps_and_stops_at_the_end() {
        let mut s = Sequencer { duration: 2.0, playing: true, ..Default::default() };
        s.advance(1.0);
        assert_eq!(s.time, 1.0);
        s.advance(3.0);
        assert_eq!(s.time, 2.0);
        assert!(!s.playing, "non-looping stops at the end");
    }

    #[test]
    fn sequencer_loops_forever() {
        let mut s = Sequencer { duration: 2.0, looping: true, playing: true, ..Default::default() };
        s.advance(2.5);
        assert!((s.time - 0.5).abs() < 1e-5);
        assert!(s.playing);
    }

    #[test]
    fn events_fire_when_crossed() {
        let s = Sequencer {
            duration: 10.0,
            tracks: vec![SeqTrack {
                name: "explosion".into(),
                kind: SeqTrackKind::Event,
                keys: vec![SeqKey { time: 3.0, value: [0.0; 4] }, SeqKey { time: 5.0, value: [0.0; 4] }],
            }],
            ..Default::default()
        };
        assert_eq!(s.events_between(0.0, 4.0), vec![("explosion".to_owned(), 3.0)]);
        assert_eq!(s.events_between(3.0, 6.0), vec![("explosion".to_owned(), 5.0)]);
        assert!(s.events_between(5.0, 6.0).is_empty(), "an event at exactly `from` does not refire");
    }
}
