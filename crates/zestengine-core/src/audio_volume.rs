//! Authored audio environments. File I/O belongs to assets; region selection
//! and playback belong to the engine. These types have no audio SDK dependency.

use serde::{Deserialize, Serialize};
use crate::TypedComponent;

/// Reusable `.audiofx` environment, applied by an Audio Volume.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioEnvironmentProfile {
    /// Level of Steam Audio's simulated reflections / listener reverb.
    pub reverb_gain: f32,
    /// Seconds to transition when entering, leaving, or editing a region.
    pub fade_time: f32,
    /// Sounds inside this region heard by a listener outside it.
    pub interior_gain: f32,
    pub interior_lowpass_hz: f32,
    /// Sounds outside this region heard by a listener inside it.
    pub exterior_gain: f32,
    pub exterior_lowpass_hz: f32,
}

impl Default for AudioEnvironmentProfile {
    fn default() -> Self {
        Self { reverb_gain: 1.0, fade_time: 0.5, interior_gain: 1.0,
            interior_lowpass_hz: 22_000.0, exterior_gain: 1.0, exterior_lowpass_hz: 22_000.0 }
    }
}

impl AudioEnvironmentProfile {
    pub fn sanitized(self) -> Self {
        fn finite(v: f32, default: f32, lo: f32, hi: f32) -> f32 {
            if v.is_finite() { v.clamp(lo, hi) } else { default }
        }
        Self {
            reverb_gain: finite(self.reverb_gain, 1.0, 0.0, 2.0),
            fade_time: finite(self.fade_time, 0.5, 0.0, 30.0),
            interior_gain: finite(self.interior_gain, 1.0, 0.0, 1.0),
            exterior_gain: finite(self.exterior_gain, 1.0, 0.0, 1.0),
            interior_lowpass_hz: finite(self.interior_lowpass_hz, 22_000.0, 20.0, 22_000.0),
            exterior_lowpass_hz: finite(self.exterior_lowpass_hz, 22_000.0, 20.0, 22_000.0),
        }
    }
}

/// Highest-priority containing region wins; equal priority follows scene order.
/// Global volumes provide a scene-wide baseline. Local bounds follow the full
/// world transform, including parent rotation and scale.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioVolume {
    pub profile: String,
    pub global: bool,
    pub priority: f32,
    pub half_extents: [f32; 3],
}

impl Default for AudioVolume {
    fn default() -> Self {
        Self { profile: String::new(), global: false, priority: 0.0, half_extents: [5.0, 3.0, 5.0] }
    }
}

impl TypedComponent for AudioVolume {
    const NAME: &'static str = "Audio Volume";
}
