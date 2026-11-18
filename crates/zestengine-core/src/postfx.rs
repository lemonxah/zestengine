use serde::{Deserialize, Serialize};

/// Hardware ray-hit shading policy. Default inherits the environment in a profile.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum RayLightingMode {
    #[default]
    Default = 0,
    SurfaceCache = 1,
    HitLightingForReflections = 2,
    HitLighting = 3,
}
impl RayLightingMode {
    pub const ALL: [Self; 4] = [Self::Default, Self::SurfaceCache, Self::HitLightingForReflections, Self::HitLighting];
    pub fn label(self) -> &'static str {
        match self {
            Self::Default => "Environment Default",
            Self::SurfaceCache => "Surface Cache",
            Self::HitLightingForReflections => "Hit Lighting for Reflections",
            Self::HitLighting => "Hit Lighting for GI and Reflections",
        }
    }
}

/// Per-property GI/reflection overrides in a camera's post-process volumes.
/// None inherits the environment. These do not mutate another camera's settings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FluxPostFx {
    /// Resolve visible diffuse hits before tracing world geometry. None inherits the environment.
    pub gi_screen_traces: Option<bool>,
    pub ray_lighting_mode: Option<RayLightingMode>,
    /// Shared surface-cache update budget multiplier (0.5..16), separate from final-gather history.
    /// Angular sample density for cached diffuse bounces (0.5..8).
    /// At 6+, halve cached lighting receiver spacing; angular density caps at 4.
    pub scene_lighting_quality: Option<f32>,
    pub scene_lighting_update_speed: Option<f32>,
    /// Shared capture texel density and maximum card resolution (0.5..1).
    /// The highest requested scale among active views protects their detail.
    pub surface_cache_resolution: Option<f32>,
    /// Ray quality multiplier (0.25..8); at 6+, halve screen-probe spacing.
    pub final_gather_quality: Option<f32>,
    pub final_gather_update_speed: Option<f32>,
    pub max_trace_distance: Option<f32>,
    /// Quality 0.25 enables adaptive half-resolution world reflections.
    /// Coherent rough receivers share rays; mirrors and boundaries stay full resolution.
    pub reflection_quality: Option<f32>,
    pub max_roughness_to_trace: Option<f32>,
    pub max_reflection_bounces: Option<f32>,
    pub reflection_screen_traces: Option<bool>,
}

impl FluxPostFx {
    /// Blend independent overrides. Quality and speed multiply the environment;
    /// their neutral baseline is 1. Absolute settings inherit an explicit base.
    /// Switching from automatic trace distance to an explicit distance is
    /// discrete, since an automatic distance has no numeric value to interpolate.
    pub fn blend(self, other: Self, weight: f32) -> Self {
        fn field(a: Option<f32>, b: Option<f32>, t: f32, neutral: Option<f32>) -> Option<f32> {
            let Some(b) = b.filter(|v| v.is_finite()) else { return a; };
            if t <= 0.0 { return a; }
            match a.or(neutral) {
                Some(a) => Some(a + (b - a) * t),
                None => if t >= 0.5 { Some(b) } else { None },
            }
        }
        let t = if weight.is_finite() { weight.clamp(0.0, 1.0) } else { 0.0 };
        Self {
            gi_screen_traces: if t >= 0.5 { other.gi_screen_traces.or(self.gi_screen_traces) } else { self.gi_screen_traces },
            reflection_screen_traces: if t >= 0.5 { other.reflection_screen_traces.or(self.reflection_screen_traces) } else { self.reflection_screen_traces },
            ray_lighting_mode: if t >= 0.5 { other.ray_lighting_mode.or(self.ray_lighting_mode) } else { self.ray_lighting_mode },
            scene_lighting_quality: field(self.scene_lighting_quality, other.scene_lighting_quality, t, Some(1.0)),
            scene_lighting_update_speed: field(self.scene_lighting_update_speed, other.scene_lighting_update_speed, t, Some(1.0)),
            surface_cache_resolution: field(self.surface_cache_resolution, other.surface_cache_resolution, t, Some(1.0)),
            final_gather_quality: field(self.final_gather_quality, other.final_gather_quality, t, Some(1.0)),
            final_gather_update_speed: field(self.final_gather_update_speed, other.final_gather_update_speed, t, Some(1.0)),
            max_trace_distance: field(self.max_trace_distance, other.max_trace_distance, t, None),
            reflection_quality: field(self.reflection_quality, other.reflection_quality, t, Some(1.0)),
            max_roughness_to_trace: field(self.max_roughness_to_trace, other.max_roughness_to_trace, t, None),
            max_reflection_bounces: field(self.max_reflection_bounces, other.max_reflection_bounces, t, None),
        }
    }
}

/// Histogram metering of the physical HDR image, before display exposure.
/// EV100 uses the standard 1.2 luminance calibration and 18% middle grey.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AutoExposure {
    pub enabled: bool,
    pub min_ev100: f32,
    pub max_ev100: f32,
    /// Stops per second when entering brighter / darker surroundings.
    pub speed_up: f32,
    pub speed_down: f32,
    /// Retained histogram percentile interval, in percent.
    pub low_percent: f32,
    pub high_percent: f32,
    pub histogram_min_ev100: f32,
    pub histogram_max_ev100: f32,
}
impl Default for AutoExposure {
    fn default() -> Self {
        Self {enabled:false,min_ev100:-10.0,max_ev100:20.0,speed_up:3.0,speed_down:1.0,
            low_percent:70.0,high_percent:90.0,histogram_min_ev100:-10.0,histogram_max_ev100:20.0}
    }
}
impl AutoExposure {
    pub fn blend(self, other: Self, t: f32) -> Self {
        let t=t.clamp(0.0,1.0);let mix=|a:f32,b:f32|a+(b-a)*t;
        Self {enabled:if t>=0.5 {other.enabled} else {self.enabled},
            min_ev100:mix(self.min_ev100,other.min_ev100),max_ev100:mix(self.max_ev100,other.max_ev100),
            speed_up:mix(self.speed_up,other.speed_up),speed_down:mix(self.speed_down,other.speed_down),
            low_percent:mix(self.low_percent,other.low_percent),high_percent:mix(self.high_percent,other.high_percent),
            histogram_min_ev100:mix(self.histogram_min_ev100,other.histogram_min_ev100),
            histogram_max_ev100:mix(self.histogram_max_ev100,other.histogram_max_ev100)}
    }
}

/// Resolved per-frame post-processing parameters (the blended profile flattened
/// for the GPU). Per-pixel effects (exposure, tonemap, color grading, vignette)
/// are applied in the surface shaders; chromatic aberration + bloom need a
/// a fullscreen pass (follow-up).

pub const POSTFX_EXTENSION: &str = "postfx";

/// Tonemap operator applied at the end of the post stack (HDR → display).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TonemapMode {
    None,
    Reinhard,
    #[default]
    Aces,
}

impl TonemapMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Reinhard => "Reinhard",
            Self::Aces => "ACES",
        }
    }
    pub const ALL: [TonemapMode; 3] = [Self::None, Self::Reinhard, Self::Aces];
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Tonemap {
    /// In a Volume profile: whether this profile overrides the tonemap mode and
    /// exposure underneath it. Ignored for the scene's global (Environment) post.
    #[serde(default)]
    pub enabled: bool,
    pub mode: TonemapMode,
    /// Exposure in stops (EV): final color is multiplied by 2^exposure.
    pub exposure: f32,
    pub auto_exposure: AutoExposure,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Bloom {
    pub enabled: bool,
    /// Luminance above which pixels bloom.
    pub threshold: f32,
    /// Strength of the added glow.
    pub intensity: f32,
    /// Blur spread (0..1, fraction of screen).
    pub radius: f32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ColorGrading {
    pub enabled: bool,
    /// Post-exposure multiplier (separate from the tonemap EV).
    pub exposure: f32,
    /// 1.0 = neutral; >1 increases contrast around mid-grey.
    pub contrast: f32,
    /// 1.0 = neutral; 0 = greyscale.
    pub saturation: f32,
    /// -1 (cool) .. +1 (warm) white-balance shift.
    pub temperature: f32,
    /// -1 (green) .. +1 (magenta) white-balance shift.
    pub tint: f32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Vignette {
    pub enabled: bool,
    /// Darkening strength at the edges.
    pub intensity: f32,
    /// Edge softness.
    pub smoothness: f32,
    pub color: [f32; 3],
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ChromaticAberration {
    pub enabled: bool,
    /// Channel-split strength toward the edges.
    pub intensity: f32,
}

/// Screen-space ambient occlusion (indirect light only). Off unless a scene or
/// Volume enables it; the algorithm (Low / High) is a Project Setting.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AmbientOcclusion {
    pub enabled: bool,
    /// Strength (0 = none, 1 = physically based).
    pub intensity: f32,
    /// World-space search radius in metres.
    pub radius: f32,
    /// Fraction of the radius over which occluders fade out.
    pub falloff: f32,
    /// Multi-bounce approximation (lightens occlusion on bright surfaces).
    pub multi_bounce: bool,
}

impl Default for AmbientOcclusion {
    fn default() -> Self {
        Self { enabled: false, intensity: 1.0, radius: 0.6, falloff: 0.6, multi_bounce: true }
    }
}

/// A complete post-processing profile: one of every effect's settings.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
#[derive(Default)]
pub struct PostFxProfile {
    pub flux: FluxPostFx,
    pub tonemap: Tonemap,
    pub bloom: Bloom,
    pub color_grading: ColorGrading,
    pub vignette: Vignette,
    pub chromatic_aberration: ChromaticAberration,
    #[serde(default)]
    pub ambient_occlusion: AmbientOcclusion,
}

impl Default for Tonemap {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: TonemapMode::Aces,
            exposure: 0.0,
            auto_exposure: Default::default(),
        }
    }
}
impl Default for Bloom {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: 1.0,
            intensity: 0.5,
            radius: 0.5,
        }
    }
}
impl Default for ColorGrading {
    fn default() -> Self {
        Self {
            enabled: false,
            exposure: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            temperature: 0.0,
            tint: 0.0,
        }
    }
}
impl Default for Vignette {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 0.4,
            smoothness: 0.4,
            color: [0.0, 0.0, 0.0],
        }
    }
}
impl Default for ChromaticAberration {
    fn default() -> Self {
        Self {
            enabled: false,
            intensity: 0.3,
        }
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]
}

impl PostFxProfile {
    /// Blend `t` of the way from `self` toward `other` (Volume weight blend).
    /// Numeric parameters lerp; the tonemap mode switches to `other` once it's
    /// the dominant contributor (`t >= 0.5`). An effect `other` leaves disabled
    /// (or a tonemap it doesn't override)
    /// doesn't override (Unity's per-effect override): a volume that only sets
    /// up colour grading keeps the bloom underneath it. An effect enabled in
    /// `other` but not in `self` fades in from its neutral setting.
    pub fn lerp(&self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        let dom = t >= 0.5;
        let bloom = if !other.bloom.enabled {
            self.bloom
        } else {
            let from = if self.bloom.enabled { self.bloom } else { Bloom { intensity: 0.0, ..other.bloom } };
            Bloom {
                enabled: true,
                threshold: lerp(from.threshold, other.bloom.threshold, t),
                intensity: lerp(from.intensity, other.bloom.intensity, t),
                radius: lerp(from.radius, other.bloom.radius, t),
            }
        };
        let color_grading = if !other.color_grading.enabled {
            self.color_grading
        } else {
            let from = if self.color_grading.enabled {
                self.color_grading
            } else {
                ColorGrading { enabled: true, exposure: 0.0, contrast: 1.0, saturation: 1.0, temperature: 0.0, tint: 0.0 }
            };
            let o = &other.color_grading;
            ColorGrading {
                enabled: true,
                exposure: lerp(from.exposure, o.exposure, t),
                contrast: lerp(from.contrast, o.contrast, t),
                saturation: lerp(from.saturation, o.saturation, t),
                temperature: lerp(from.temperature, o.temperature, t),
                tint: lerp(from.tint, o.tint, t),
            }
        };
        let vignette = if !other.vignette.enabled {
            self.vignette
        } else {
            let from = if self.vignette.enabled { self.vignette } else { Vignette { intensity: 0.0, ..other.vignette } };
            Vignette {
                enabled: true,
                intensity: lerp(from.intensity, other.vignette.intensity, t),
                smoothness: lerp(from.smoothness, other.vignette.smoothness, t),
                color: lerp3(from.color, other.vignette.color, t),
            }
        };
        let chromatic_aberration = if !other.chromatic_aberration.enabled {
            self.chromatic_aberration
        } else {
            let from = if self.chromatic_aberration.enabled { self.chromatic_aberration.intensity } else { 0.0 };
            ChromaticAberration { enabled: true, intensity: lerp(from, other.chromatic_aberration.intensity, t) }
        };
        let ambient_occlusion = if !other.ambient_occlusion.enabled {
            self.ambient_occlusion
        } else {
            let o = &other.ambient_occlusion;
            let from = if self.ambient_occlusion.enabled { self.ambient_occlusion } else { AmbientOcclusion { intensity: 0.0, ..*o } };
            AmbientOcclusion {
                enabled: true,
                intensity: lerp(from.intensity, o.intensity, t),
                radius: lerp(from.radius, o.radius, t),
                falloff: lerp(from.falloff, o.falloff, t),
                multi_bounce: if dom { o.multi_bounce } else { from.multi_bounce },
            }
        };
        let tonemap = if !other.tonemap.enabled {
            self.tonemap
        } else {
            Tonemap {
                enabled: true,
                mode: if dom { other.tonemap.mode } else { self.tonemap.mode },
                exposure: lerp(self.tonemap.exposure, other.tonemap.exposure, t),
                auto_exposure: self.tonemap.auto_exposure.blend(other.tonemap.auto_exposure,t),
            }
        };
        Self {
            flux: self.flux.blend(other.flux, t),
            tonemap,
            bloom,
            color_grading,
            vignette,
            chromatic_aberration,
            ambient_occlusion,
        }
    }
}

/// Blend Volume profiles (sorted by ascending priority, each with a 0..1
/// weight) over `base`, the scene's global post settings, which apply in full.
pub fn blend_over(base: PostFxProfile, stack: &[(PostFxProfile, f32)]) -> PostFxProfile {
    let mut result = base;
    for (profile, weight) in stack {
        result = result.lerp(profile, weight.clamp(0.0, 1.0));
    }
    result
}

/// Blend a stack of (profile, weight) by ascending priority into one effective
/// profile. Caller supplies them already sorted by priority (low → high); each
/// is blended over the running result by its weight (0..1). The base is the
/// default profile so an empty stack yields neutral settings.
pub fn blend_profiles(stack: &[(PostFxProfile, f32)]) -> PostFxProfile {
    let mut result = PostFxProfile::default();
    for (profile, weight) in stack {
        result = result.lerp(profile, weight.clamp(0.0, 1.0));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flux_auto_exposure_profile_roundtrip_and_volume_blend() {
        let legacy:PostFxProfile=ron::from_str("(tonemap:(exposure:1.0))").unwrap();
        assert!(!legacy.tonemap.auto_exposure.enabled);
        let mut volume=PostFxProfile::default();
        volume.tonemap.auto_exposure.enabled=true;
        volume.tonemap.auto_exposure.min_ev100 = -4.0;
        volume.tonemap.auto_exposure.speed_up=7.0;
        assert_eq!(legacy.lerp(&volume,1.0).tonemap.auto_exposure,legacy.tonemap.auto_exposure,
            "non-overriding volume changed exposure");
        volume.tonemap.enabled=true;
        let half=legacy.lerp(&volume,0.5);
        assert!(half.tonemap.auto_exposure.enabled);
        assert_eq!(half.tonemap.auto_exposure.min_ev100,-7.0);
        assert_eq!(half.tonemap.auto_exposure.speed_up,5.0);
        let roundtrip:PostFxProfile=ron::from_str(&ron::to_string(&half).unwrap()).unwrap();
        assert_eq!(roundtrip.tonemap.auto_exposure,half.tonemap.auto_exposure);
        volume.tonemap.auto_exposure.enabled=false;
        assert!(!half.lerp(&volume,1.0).tonemap.auto_exposure.enabled,"volume could not restore manual exposure");
    }

    #[test]
    fn blend_weights_toward_profile() {
        let mut a = PostFxProfile::default();
        a.color_grading.contrast = 1.0;
        let mut b = PostFxProfile::default();
        b.color_grading.enabled = true;
        b.color_grading.contrast = 2.0;

        // Full weight → fully b.
        let full = blend_profiles(&[(b, 1.0)]);
        assert!(full.color_grading.enabled);
        assert!((full.color_grading.contrast - 2.0).abs() < 1e-5);

        // Half weight from a disabled base fades in from neutral contrast.
        let half = a.lerp(&b, 0.5);
        assert!((half.color_grading.contrast - 1.5).abs() < 1e-5);
    }

    #[test]
    fn disabled_effect_does_not_override() {
        let mut env = PostFxProfile::default();
        env.bloom.enabled = true;
        env.bloom.intensity = 2.0;
        let mut vol = PostFxProfile::default();
        vol.bloom.enabled = false;
        vol.color_grading.enabled = true;
        vol.color_grading.saturation = 0.5;
        env.tonemap.exposure = 0.5;
        vol.tonemap.exposure = -2.0;
        let out = blend_over(env, &[(vol, 1.0)]);
        assert!((out.tonemap.exposure - 0.5).abs() < 1e-5);
        assert!(out.bloom.enabled);
        assert!((out.bloom.intensity - 2.0).abs() < 1e-5);
        assert!(out.color_grading.enabled);
        assert!((out.color_grading.saturation - 0.5).abs() < 1e-5);
    }
}
