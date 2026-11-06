//! Spatial audio data: project-wide Steam Audio settings and the acoustic
//! scene components (surface materials, probe volumes for baking). The
//! engine's `steam_audio.rs` turns these into Steam Audio objects.

use serde::{Deserialize, Serialize};

use crate::TypedComponent;

/// How reflections are simulated.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum ReflectionMode {
    /// No reflections.
    Off,
    /// Full impulse-response convolution (best quality, most CPU).
    Convolution,
    /// Parametric reverb from per-band reverb times (cheapest).
    #[default]
    Parametric,
    /// Convolution for early reflections, parametric for the tail.
    Hybrid,
}

impl ReflectionMode {
    /// Every mode, in declaration order.
    pub const ALL: [ReflectionMode; 4] = [Self::Off, Self::Convolution, Self::Parametric, Self::Hybrid];
    /// Inspector label of this mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Convolution => "Convolution",
            Self::Parametric => "Parametric",
            Self::Hybrid => "Hybrid",
        }
    }
}

/// The speaker layout the audio engine outputs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum OutputLayout {
    /// Binaural (HRTF) stereo for headphones.
    #[default]
    Headphones,
    /// Stereo speakers.
    Stereo,
    /// Four speakers.
    Quad,
    /// 5.1 surround.
    Surround51,
    /// 7.1 surround.
    Surround71,
}

impl OutputLayout {
    /// Every layout, in declaration order.
    pub const ALL: [OutputLayout; 5] = [Self::Headphones, Self::Stereo, Self::Quad, Self::Surround51, Self::Surround71];
    /// Inspector label of this layout.
    pub fn label(self) -> &'static str {
        match self {
            Self::Headphones => "Headphones (binaural)",
            Self::Stereo => "Stereo speakers",
            Self::Quad => "Quad",
            Self::Surround51 => "5.1",
            Self::Surround71 => "7.1",
        }
    }

    /// Speaker count of this layout.
    pub fn channels(self) -> u16 {
        match self {
            Self::Headphones | Self::Stereo => 2,
            Self::Quad => 4,
            Self::Surround51 => 6,
            Self::Surround71 => 8,
        }
    }
}

/// Which ray tracer bakes the acoustic probes.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum RayTracerKind {
    /// Steam Audio's built-in CPU ray tracer.
    #[default]
    Default,
    /// Intel Embree (faster CPU tracing).
    Embree,
    /// AMD Radeon Rays (GPU, OpenCL).
    RadeonRays,
}

impl RayTracerKind {
    /// Every kind, in declaration order.
    pub const ALL: [RayTracerKind; 3] = [Self::Default, Self::Embree, Self::RadeonRays];
    /// Inspector label of this kind.
    pub fn label(self) -> &'static str {
        match self {
            Self::Default => "Default (CPU)",
            Self::Embree => "Embree",
            Self::RadeonRays => "Radeon Rays (GPU)",
        }
    }
}

/// Steam Audio device/simulation settings, persisted as an `.audiosettings` asset.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpatialAudioSettings {
    /// How reflections are simulated.
    pub reflections: ReflectionMode,
    /// Listener-centric reverb for sources without their own reflections.
    pub listener_reverb: bool,
    /// Real-time reflection rays per simulation.
    pub rays: u32,
    /// Reflection bounces per ray.
    pub bounces: u32,
    /// Impulse-response length in seconds.
    pub duration: f32,
    /// Ambisonic order of reflections and paths (1..=3).
    pub order: u32,
    /// Diffuse (scattering) samples per ray.
    pub diffuse_samples: u32,
    /// Reflection / pathing simulations per second.
    pub update_rate: f32,
    /// Most sources simulated for reflections at once.
    pub max_sources: u32,
    /// Which ray tracer to use.
    pub ray_tracer: RayTracerKind,
    /// Project-relative `.sofa` HRTF; empty = Steam Audio's default.
    pub hrtf_sofa: String,
    /// Speaker layout.
    pub output: OutputLayout,
    /// Baking quality.
    pub bake_rays: u32,
    /// Bake rays per probe (quality).
    pub bake_bounces: u32,
    /// Impulse-response length for baked probes (s).
    pub bake_duration: f32,
}

impl Default for SpatialAudioSettings {
    fn default() -> Self {
        Self {
            reflections: ReflectionMode::Parametric,
            listener_reverb: true,
            rays: 2048,
            bounces: 8,
            duration: 1.0,
            order: 1,
            diffuse_samples: 32,
            update_rate: 10.0,
            max_sources: 16,
            ray_tracer: RayTracerKind::Default,
            hrtf_sofa: String::new(),
            output: OutputLayout::Headphones,
            bake_rays: 16384,
            bake_bounces: 16,
            bake_duration: 2.0,
        }
    }
}

impl SpatialAudioSettings {
    /// Settings clamped to what Steam Audio accepts.
    pub fn sanitized(&self) -> Self {
        let mut s = self.clone();
        s.rays = s.rays.clamp(64, 65536);
        s.bounces = s.bounces.clamp(1, 64);
        s.duration = s.duration.clamp(0.1, 10.0);
        s.order = s.order.clamp(0, 3);
        s.diffuse_samples = s.diffuse_samples.clamp(1, 1024);
        s.update_rate = s.update_rate.clamp(1.0, 60.0);
        s.max_sources = s.max_sources.clamp(1, 256);
        s.bake_rays = s.bake_rays.clamp(256, 262144);
        s.bake_bounces = s.bake_bounces.clamp(1, 256);
        s.bake_duration = s.bake_duration.clamp(0.1, 10.0);
        s
    }
}

/// Steam Audio's built-in surface materials.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum AcousticPreset {
    /// A neutral default surface.
    #[default]
    Generic,
    /// Brick walls.
    Brick,
    /// Concrete.
    Concrete,
    /// Ceramic tiles.
    Ceramic,
    /// Gravel.
    Gravel,
    /// Carpet.
    Carpet,
    /// Glass.
    Glass,
    /// Plaster.
    Plaster,
    /// Wood.
    Wood,
    /// Metal.
    Metal,
    /// Rock.
    Rock,
    /// Values set per surface through the material's own fields.
    Custom,
}

impl AcousticPreset {
    /// Every preset, in declaration order.
    pub const ALL: [AcousticPreset; 12] = [
        Self::Generic,
        Self::Brick,
        Self::Concrete,
        Self::Ceramic,
        Self::Gravel,
        Self::Carpet,
        Self::Glass,
        Self::Plaster,
        Self::Wood,
        Self::Metal,
        Self::Rock,
        Self::Custom,
    ];

    /// Inspector label of this preset.
    pub fn label(self) -> &'static str {
        match self {
            Self::Generic => "Generic",
            Self::Brick => "Brick",
            Self::Concrete => "Concrete",
            Self::Ceramic => "Ceramic",
            Self::Gravel => "Gravel",
            Self::Carpet => "Carpet",
            Self::Glass => "Glass",
            Self::Plaster => "Plaster",
            Self::Wood => "Wood",
            Self::Metal => "Metal",
            Self::Rock => "Rock",
            Self::Custom => "Custom",
        }
    }

    /// Steam Audio's preset values: (absorption, scattering, transmission),
    /// low / mid / high bands. None for Custom.
    pub fn values(self) -> Option<([f32; 3], f32, [f32; 3])> {
        Some(match self {
            Self::Generic => ([0.10, 0.20, 0.30], 0.05, [0.100, 0.050, 0.030]),
            Self::Brick => ([0.03, 0.04, 0.07], 0.05, [0.015, 0.015, 0.015]),
            Self::Concrete => ([0.05, 0.07, 0.08], 0.05, [0.015, 0.002, 0.001]),
            Self::Ceramic => ([0.01, 0.02, 0.02], 0.05, [0.060, 0.044, 0.011]),
            Self::Gravel => ([0.60, 0.70, 0.80], 0.05, [0.031, 0.012, 0.008]),
            Self::Carpet => ([0.24, 0.69, 0.73], 0.05, [0.020, 0.005, 0.003]),
            Self::Glass => ([0.06, 0.03, 0.02], 0.05, [0.060, 0.044, 0.011]),
            Self::Plaster => ([0.12, 0.06, 0.04], 0.05, [0.056, 0.056, 0.004]),
            Self::Wood => ([0.11, 0.07, 0.06], 0.05, [0.070, 0.014, 0.005]),
            Self::Metal => ([0.20, 0.07, 0.06], 0.05, [0.200, 0.025, 0.010]),
            Self::Rock => ([0.13, 0.20, 0.24], 0.05, [0.015, 0.002, 0.001]),
            Self::Custom => return None,
        })
    }
}

/// How an object's mesh sounds: absorption, scattering and transmission for
/// reflections and occlusion. Static geometry without one uses Generic.
/// `dynamic` makes a moving (non-static) object occlude and reflect too.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AcousticMaterial {
    /// The acoustic properties (or Custom for the fields below).
    pub preset: AcousticPreset,
    /// Custom: fraction absorbed per band (low / mid / high).
    pub absorption: [f32; 3],
    /// Custom: fraction scattered (0 specular .. 1 diffuse).
    pub scattering: f32,
    /// Custom: fraction transmitted through per band.
    pub transmission: [f32; 3],
    /// Include this object as a moving occluder (instanced mesh) even when it
    /// isn't marked Static.
    pub dynamic: bool,
}

impl Default for AcousticMaterial {
    fn default() -> Self {
        let (absorption, scattering, transmission) = AcousticPreset::Generic.values().unwrap();
        Self { preset: AcousticPreset::Generic, absorption, scattering, transmission, dynamic: false }
    }
}

impl AcousticMaterial {
    /// The effective (absorption, scattering, transmission).
    pub fn values(&self) -> ([f32; 3], f32, [f32; 3]) {
        self.preset
            .values()
            .unwrap_or((self.absorption, self.scattering.clamp(0.0, 1.0), self.transmission))
    }
}

impl TypedComponent for AcousticMaterial {
    const NAME: &'static str = "Acoustic Material";
}

/// A box filled with sound probes (uniform floor placement) for baked
/// reverb and pathing. Bake with Tools > Bake Audio.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioProbeVolume {
    /// Full box size in local meters (spans -size/2 .. +size/2).
    pub size: [f32; 3],
    /// Distance between probes on the floor grid.
    pub spacing: f32,
    /// Probe height above the floor.
    pub height: f32,
    /// Bake listener-centric reverb.
    pub reverb: bool,
    /// Bake pathing (sound around corners).
    pub pathing: bool,
}

impl Default for AudioProbeVolume {
    fn default() -> Self {
        Self { size: [10.0, 4.0, 10.0], spacing: 2.0, height: 1.5, reverb: true, pathing: true }
    }
}

impl TypedComponent for AudioProbeVolume {
    const NAME: &'static str = "Audio Probe Volume";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_project_settings_load_with_defaults() {
        let s: SpatialAudioSettings = ron::from_str("()").unwrap();
        assert_eq!(s, SpatialAudioSettings::default());
        let clamped = SpatialAudioSettings { order: 9, rays: 1, ..Default::default() }.sanitized();
        assert_eq!((clamped.order, clamped.rays), (3, 64));
    }

    #[test]
    fn custom_material_uses_its_own_values() {
        let m = AcousticMaterial { preset: AcousticPreset::Custom, absorption: [0.5; 3], ..Default::default() };
        assert_eq!(m.values().0, [0.5; 3]);
        let wood = AcousticMaterial { preset: AcousticPreset::Wood, ..Default::default() };
        assert_eq!(wood.values().0, [0.11, 0.07, 0.06]);
    }
}
