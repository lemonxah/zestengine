//! Voice chat settings and lip sync: the scene's `Voice Chat` component
//! (proximity / global, falloff, transmit mode) and `Lip Sync`, which drives
//! viseme blendshapes from a speaker's voice.

use serde::{Deserialize, Serialize};

use crate::face::normalize;
use crate::{FaceFrame, TypedComponent};

/// The 15 visemes (Oculus / VRChat order).
pub const VISEMES: [&str; 15] = ["sil", "PP", "FF", "TH", "DD", "kk", "CH", "SS", "nn", "RR", "aa", "E", "ih", "oh", "ou"];

/// Viseme weights (0..1) in [`VISEMES`] order.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VisemeFrame(pub [f32; 15]);

impl VisemeFrame {
    /// The weight of viseme `i` (0..1).
    pub fn get(&self, i: usize) -> f32 {
        self.0.get(i).copied().unwrap_or(0.0)
    }

    /// How open the mouth is overall (0 when silent).
    pub fn openness(&self) -> f32 {
        self.0[1..].iter().sum::<f32>().min(1.0)
    }

    /// Move toward `target` by a share `k` (0..1) of the difference.
    pub fn approach(&mut self, target: &VisemeFrame, k: f32) {
        for (a, b) in self.0.iter_mut().zip(target.0) {
            *a += (b - *a) * k.clamp(0.0, 1.0);
        }
    }

    /// Mouth shapes of a face-tracking frame for avatars that only carry
    /// Unified Expressions shapes (no viseme blendshapes).
    pub fn to_face(&self) -> FaceFrame {
        let mut f = FaceFrame::default();
        let v = |name: &str| self.get(VISEMES.iter().position(|n| *n == name).unwrap_or(0));
        let jaw = 0.7 * v("aa") + 0.5 * v("oh") + 0.4 * v("E") + 0.25 * v("ih") + 0.2 * v("ou") + 0.15 * (v("DD") + v("kk") + v("nn") + v("RR"));
        let funnel = 0.6 * v("oh") + 0.3 * v("ou") + 0.3 * v("CH");
        let pucker = 0.8 * v("ou") + 0.3 * v("oh");
        let stretch = 0.5 * v("E") + 0.5 * v("ih") + 0.4 * v("SS");
        let press = 0.9 * v("PP");
        let suck_lower = 0.7 * v("FF");
        let tongue = 0.6 * v("TH");
        f.set_param("JawOpen", jaw.min(1.0));
        f.set_param("LipFunnel", funnel.min(1.0));
        f.set_param("LipPucker", pucker.min(1.0));
        f.set_param("MouthStretch", stretch.min(1.0));
        f.set_param("MouthPress", press.min(1.0));
        f.set_param("MouthClosed", press.min(1.0));
        f.set_param("LipSuckLower", suck_lower.min(1.0));
        f.set_param("TongueOut", tongue.min(1.0));
        f
    }
}

/// The viseme a blendshape name drives, if any: `vrc.v_aa`, `v_aa`,
/// `viseme_aa`, `Fcl_MTH_A`, VRM `A`/`I`/`U`/`E`/`O`, or a bare viseme name.
pub fn viseme_index(shape: &str) -> Option<usize> {
    let leaf = shape.rsplit(['/']).next().unwrap_or(shape);
    let full = normalize(leaf);
    // A marker anywhere (after a mesh-name prefix such as `Body.`), else a
    // leading `v` / `mth`, else the bare name.
    let tail = ["vrcv", "visemes", "viseme", "fclmth"]
        .iter()
        .find_map(|m| full.find(m).map(|at| full[at + m.len()..].to_owned()))
        .or_else(|| ["mth", "v"].iter().find_map(|p| full.strip_prefix(p).filter(|r| lookup(r).is_some()).map(str::to_owned)))
        .unwrap_or(full);
    lookup(&tail)
}

fn lookup(n: &str) -> Option<usize> {
    let n = match n {
        "a" => "aa",
        "i" => "ih",
        "u" => "ou",
        "o" => "oh",
        other => other,
    };
    VISEMES.iter().position(|v| normalize(v) == n)
}

/// How voice falls off with distance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum VoiceRange {
    /// Volume falls off with distance between speaker and listener.
    #[default]
    Proximity,
    /// Everyone hears everyone at full volume.
    Global,
}

/// How the volume falls off between the minimum and maximum distance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum VoiceRolloff {
    /// Inverse distance past the minimum distance, like real sound.
    #[default]
    Logarithmic,
    /// Straight-line falloff.
    Linear,
}

/// When the microphone is live.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum VoiceTransmit {
    /// Send while the mic is above the activation threshold.
    #[default]
    VoiceActivity,
    /// Send while the push-to-talk action is held.
    PushToTalk,
    /// Always send.
    AlwaysOn,
}

/// Scene-wide voice chat: put one in a networked scene (next to the Network
/// Manager). Without one, voice is off.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VoiceChat {
    /// Whether voice chat is on.
    pub enabled: bool,
    /// How voice falls off with distance.
    pub range: VoiceRange,
    /// How the volume falls off between the minimum and maximum distance.
    pub rolloff: VoiceRolloff,
    /// Full volume within this distance (m).
    pub min_distance: f32,
    /// Silent past this distance (m).
    pub max_distance: f32,
    /// When the microphone is live.
    pub transmit: VoiceTransmit,
    /// Input action held for push-to-talk.
    pub push_to_talk_action: String,
    /// Voice activity threshold (dBFS).
    pub activation_db: f32,
    /// Mic gain (linear).
    pub mic_gain: f32,
    /// Opus bitrate (bits per second) when Opus is available.
    pub bitrate: u32,
}

impl Default for VoiceChat {
    fn default() -> Self {
        Self {
            enabled: true,
            range: VoiceRange::Proximity,
            rolloff: VoiceRolloff::Logarithmic,
            min_distance: 1.0,
            max_distance: 25.0,
            transmit: VoiceTransmit::VoiceActivity,
            push_to_talk_action: "Voice".into(),
            activation_db: -45.0,
            mic_gain: 1.0,
            bitrate: 24_000,
        }
    }
}

impl TypedComponent for VoiceChat {
    const NAME: &'static str = "Voice Chat";
}

impl VoiceChat {
    /// Volume for a speaker `distance` metres away.
    pub fn gain(&self, distance: f32) -> f32 {
        if self.range == VoiceRange::Global {
            return 1.0;
        }
        let (min, max) = (self.min_distance.max(0.01), self.max_distance.max(self.min_distance + 0.01));
        if distance <= min {
            return 1.0;
        }
        if distance >= max {
            return 0.0;
        }
        match self.rolloff {
            VoiceRolloff::Linear => 1.0 - (distance - min) / (max - min),
            // Inverse distance, faded to zero over the last fifth of the range.
            VoiceRolloff::Logarithmic => {
                let fade = ((max - distance) / (0.2 * (max - min))).clamp(0.0, 1.0);
                (min / distance) * fade
            }
        }
    }
}

/// When lip sync drives the viseme shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LipSyncMode {
    /// Only while face tracking isn't tracking this avatar.
    #[default]
    WhenUntracked,
    /// Always, layered on top of face tracking.
    Always,
    /// Off: face tracking drives the mouth alone.
    Off,
}

impl LipSyncMode {
    /// Every mode, in declaration order.
    pub const ALL: [LipSyncMode; 3] = [LipSyncMode::WhenUntracked, LipSyncMode::Always, LipSyncMode::Off];
    /// Inspector label of this mode.
    pub fn label(self) -> &'static str {
        match self {
            LipSyncMode::WhenUntracked => "When face tracking is off",
            LipSyncMode::Always => "Always",
            LipSyncMode::Off => "Off",
        }
    }
}

/// Engine-filled state of a [`LipSync`], not saved.
#[derive(Clone, Debug, Default)]
pub struct LipSyncRuntime {
    /// The current viseme frame (runtime).
    pub frame: VisemeFrame,
    /// Voice is coming in for this avatar.
    pub active: bool,
    /// (blendshape, viseme) pairs found on the model.
    pub matched: Vec<(String, String)>,
}

/// Drives viseme blendshapes from the speaker's voice: this machine's mic on
/// the local avatar, the player's voice on a remote one (analysed as it plays,
/// so it stays in sync). Avatars without viseme shapes can open the jaw and
/// shape the lips through their Unified Expressions shapes instead.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LipSync {
    /// When lip sync drives the mouth.
    pub mode: LipSyncMode,
    /// 0 = raw, toward 1 = smoother.
    pub smoothing: f32,
    /// Scales viseme weights.
    pub gain: f32,
    /// Also drive face shapes (jaw, lips) when the model has no viseme shapes.
    pub drive_face_shapes: bool,
    /// Engine-filled state (not saved).
    #[serde(skip)]
    pub runtime: LipSyncRuntime,
}

impl Default for LipSync {
    fn default() -> Self {
        Self { mode: LipSyncMode::WhenUntracked, smoothing: 0.35, gain: 1.0, drive_face_shapes: true, runtime: LipSyncRuntime::default() }
    }
}

impl TypedComponent for LipSync {
    const NAME: &'static str = "Lip Sync";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viseme_names_match_common_conventions() {
        let aa = VISEMES.iter().position(|v| *v == "aa");
        for name in ["vrc.v_aa", "v_aa", "viseme_aa", "Fcl_MTH_A", "A", "aa", "Body.vrc.v_aa"] {
            assert_eq!(viseme_index(name), aa, "{name}");
        }
        assert_eq!(viseme_index("vrc.v_sil"), Some(0));
        assert_eq!(viseme_index("U"), VISEMES.iter().position(|v| *v == "ou"));
        assert_eq!(viseme_index("JawOpen"), None);
        assert_eq!(viseme_index("Hair"), None);
    }

    #[test]
    fn proximity_falls_off_and_global_does_not() {
        let mut v = VoiceChat::default();
        assert_eq!(v.gain(0.5), 1.0);
        assert!(v.gain(5.0) < 0.3 && v.gain(5.0) > 0.1);
        assert_eq!(v.gain(30.0), 0.0);
        v.rolloff = VoiceRolloff::Linear;
        assert!((v.gain(13.0) - 0.5).abs() < 1e-5);
        v.range = VoiceRange::Global;
        assert_eq!(v.gain(100.0), 1.0);
    }

    #[test]
    fn open_vowel_opens_the_jaw() {
        let mut f = VisemeFrame::default();
        f.0[10] = 1.0;
        let face = f.to_face();
        assert!(crate::ShapeSource::resolve("JawOpen").unwrap().read(&face) > 0.6);
    }
}
