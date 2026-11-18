use std::path::Path;

use anyhow::{Context as _, Result};

pub use zestengine_core::postfx::{AmbientOcclusion, AutoExposure, Bloom, ChromaticAberration, ColorGrading, FluxPostFx, POSTFX_EXTENSION, PostFxProfile, RayLightingMode, Tonemap, TonemapMode, Vignette, blend_over, blend_profiles};

pub fn load_postfx(path: impl AsRef<Path>) -> Result<PostFxProfile> {
    let path = path.as_ref();
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    ron::from_str(&text).with_context(|| format!("parsing postfx {}", path.display()))
}

pub fn save_postfx(path: impl AsRef<Path>, profile: &PostFxProfile) -> Result<()> {
    let path = path.as_ref();
    let text = ron::ser::to_string_pretty(profile, ron::ser::PrettyConfig::default())?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flux_ray_lighting_modes_inherit_switch_and_roundtrip() {
        use zestengine_core::RayLightingMode as Mode;
        let legacy: PostFxProfile = ron::from_str("()").unwrap();
        assert_eq!(legacy.flux.ray_lighting_mode, None);
        let mut base=legacy;base.flux.ray_lighting_mode=Some(Mode::HitLighting);
        let mut volume=legacy;volume.flux.ray_lighting_mode=Some(Mode::HitLightingForReflections);
        assert_eq!(base.lerp(&volume,0.49).flux.ray_lighting_mode,Some(Mode::HitLighting));
        assert_eq!(base.lerp(&volume,0.5).flux.ray_lighting_mode,Some(Mode::HitLightingForReflections));
        assert_eq!(base.lerp(&legacy,1.0).flux.ray_lighting_mode,Some(Mode::HitLighting));
        volume.flux.ray_lighting_mode=Some(Mode::Default);
        assert_eq!(base.lerp(&volume,1.0).flux.ray_lighting_mode,Some(Mode::Default));
        for mode in Mode::ALL {
            volume.flux.ray_lighting_mode=Some(mode);
            let read:PostFxProfile=ron::from_str(&ron::to_string(&volume).unwrap()).unwrap();
            assert_eq!(read.flux.ray_lighting_mode,Some(mode));
        }
        let scene:crate::RealtimeGi=ron::from_str("()").unwrap();
        assert_eq!(scene.ray_lighting_mode,Mode::SurfaceCache);
    }

    #[test]
    fn flux_overrides_blend_independently_and_preserve_legacy_profiles() {
        let legacy:crate::scene_file::RealtimeGi=ron::from_str("(enabled:true)").unwrap();
        assert!(legacy.gi_screen_traces,"legacy scenes must retain screen tracing");
        let disabled=crate::scene_file::RealtimeGi {gi_screen_traces:false,..legacy};
        let restored:crate::scene_file::RealtimeGi=ron::from_str(&ron::to_string(&disabled).unwrap()).unwrap();
        assert!(!restored.gi_screen_traces);
        let old: PostFxProfile = ron::from_str("(bloom: (enabled: true, intensity: 2.0))").unwrap();
        assert_eq!(old.flux, FluxPostFx::default());
        let mut base = old;
        base.flux.max_roughness_to_trace = Some(0.4);
        base.flux.max_reflection_bounces = Some(1.0);
        base.flux.gi_screen_traces = Some(true);
        base.flux.reflection_screen_traces = Some(true);
        let mut volume = PostFxProfile::default();
        volume.flux.scene_lighting_quality = Some(3.0);
        volume.flux.scene_lighting_update_speed = Some(3.0);
        volume.flux.surface_cache_resolution = Some(0.5);
        volume.flux.final_gather_quality = Some(2.0);
        volume.flux.max_reflection_bounces = Some(8.0);
        volume.flux.gi_screen_traces = Some(false);
        volume.flux.reflection_screen_traces = Some(false);
        let zero = base.lerp(&volume, 0.0);
        assert_eq!(zero.flux, base.flux);
        let half = base.lerp(&volume, 0.5);
        assert_eq!(half.flux.scene_lighting_quality, Some(2.0));
        assert_eq!(half.flux.scene_lighting_update_speed, Some(2.0));
        assert_eq!(half.flux.surface_cache_resolution, Some(0.75));
        assert_eq!(half.flux.final_gather_update_speed, None);
        assert_eq!(half.flux.final_gather_quality, Some(1.5));
        assert_eq!(half.flux.max_reflection_bounces, Some(4.5));
        assert_eq!(base.lerp(&volume, 0.49).flux.reflection_screen_traces, Some(true));
        assert_eq!(half.flux.reflection_screen_traces, Some(false));
        assert_eq!(base.lerp(&volume,0.49).flux.gi_screen_traces,Some(true));
        assert_eq!(half.flux.gi_screen_traces,Some(false));
        assert_eq!(half.flux.max_roughness_to_trace, Some(0.4));
        assert_eq!(half.bloom.intensity, 2.0);
        let roundtrip: PostFxProfile = ron::from_str(&ron::to_string(&half).unwrap()).unwrap();
        assert_eq!(roundtrip.flux, half.flux);
        assert_eq!(half.lerp(&PostFxProfile::default(), 1.0).flux, half.flux);
    }

    #[test]
    fn a_profile_round_trips_through_its_file() {
        let dir = std::env::temp_dir().join(format!("zest-postfx-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let path = dir.join("profiles").join("night.postfx");
        let mut profile = PostFxProfile::default();
        profile.bloom.enabled = true;
        profile.bloom.intensity = 0.42;
        save_postfx(&path, &profile).expect("the folder is created");
        let back = load_postfx(&path).unwrap();
        assert!(back.bloom.enabled && back.bloom.intensity == 0.42);
        assert!(load_postfx(dir.join("missing.postfx")).is_err());
        std::fs::write(&path, "(bloom: (intensity: 0.9))").unwrap();
        assert_eq!(load_postfx(&path).unwrap().bloom.intensity, 0.9, "fields left out take their defaults");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
