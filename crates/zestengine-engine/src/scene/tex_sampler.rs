//! Maps a texture's imported sampler options (`.meta`) to the renderer's
//! per-texture sampler key, and applies it to a freshly uploaded handle.

use std::path::Path;

use zestengine_assets::{FilterMode, SamplerOptions, WrapMode};
use zestengine_render::{Renderer, SamplerFilter, SamplerKey, SamplerWrap, TextureHandle};

/// Mip bias in levels -> the sampler key's 1/8 steps (NaN and anything past +-2 levels, the smallest limit Vulkan guarantees, clamp).
fn bias_eighths(bias: f32) -> i8 {
    if bias.is_finite() { (bias * 8.0).round().clamp(-16.0, 16.0) as i8 } else { 0 }
}

pub(crate) fn sampler_key(o: &SamplerOptions) -> SamplerKey {
    let wrap = |w: WrapMode| match w {
        WrapMode::Repeat => SamplerWrap::Repeat,
        WrapMode::Clamp => SamplerWrap::Clamp,
        WrapMode::Mirror => SamplerWrap::Mirror,
    };
    SamplerKey {
        wrap_u: wrap(o.wrap_u),
        wrap_v: wrap(o.wrap_v),
        filter: match o.filter {
            FilterMode::Point => SamplerFilter::Point,
            FilterMode::Bilinear => SamplerFilter::Bilinear,
            FilterMode::Trilinear => SamplerFilter::Trilinear,
        },
        aniso: o.aniso.max(1),
        mip_bias_eighths: bias_eighths(o.mip_bias),
    }
}

/// Read the texture's `.meta` sampler options and bind them to `handle`.
/// Unconfigured textures resolve to the defaults, which cost nothing.
pub(crate) fn apply(renderer: &mut Renderer, abs: &Path, srgb: bool, handle: TextureHandle) {
    let import = zestengine_assets::resolve_import(abs, srgb);
    renderer.set_texture_sampler(handle, sampler_key(&import.sampler));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_map_to_the_default_key() {
        assert!(sampler_key(&SamplerOptions::default()).is_default());
    }

    #[test]
    fn mip_bias_maps_to_eighths_and_changes_the_key() {
        let k = sampler_key(&SamplerOptions { mip_bias: 1.0, ..Default::default() });
        assert_eq!(k.mip_bias_eighths, 8);
        assert!(!k.is_default(), "a bias is a non-default sampler");
        assert_eq!(sampler_key(&SamplerOptions { mip_bias: -0.5, ..Default::default() }).mip_bias_eighths, -4);
        assert_eq!(sampler_key(&SamplerOptions { mip_bias: 1000.0, ..Default::default() }).mip_bias_eighths, 16, "clamped");
        assert_eq!(sampler_key(&SamplerOptions { mip_bias: f32::NAN, ..Default::default() }).mip_bias_eighths, 0);
        assert_ne!(sampler_key(&SamplerOptions { mip_bias: 0.25, ..Default::default() }), sampler_key(&SamplerOptions { mip_bias: 0.5, ..Default::default() }));
    }

    #[test]
    fn clamp_point_maps_through() {
        let k = sampler_key(&SamplerOptions { wrap_u: WrapMode::Clamp, wrap_v: WrapMode::Mirror, filter: FilterMode::Point, aniso: 0, mip_bias: 0.0 });
        assert_eq!(k.wrap_u, SamplerWrap::Clamp);
        assert_eq!(k.wrap_v, SamplerWrap::Mirror);
        assert_eq!(k.filter, SamplerFilter::Point);
        assert_eq!(k.aniso, 1);
        assert!(!k.is_default());
    }
}
