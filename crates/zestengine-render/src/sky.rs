//! Sky pass support (`shaders/skybox.vert` / `skybox.frag`):
//!   * the equirect sky texture, uploaded with a full mip chain (a large HDRI
//!     sampled without mips aliases and sparkles as soon as the camera moves);
//!   * the sky's sampler (repeat in U across the -X seam, clamp in V so the
//!     poles never blend with the opposite pole, trilinear, anisotropic);
//!   * a CPU copy of the procedural sky model (no sun disc) for the default
//!     reflection environment, so reflections match the visible sky.

use anyhow::Result;
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;

use crate::alloc::{Buffer, one_time_submit};
use crate::sampler_cache::{Filter, SamplerKey, Wrap};
use crate::texture::GpuTexture;
use crate::types::TextureData;

/// The sky's sampler key (made once by the renderer's `SamplerCache`).
pub(crate) const SAMPLER_KEY: SamplerKey = SamplerKey { wrap_u: Wrap::Repeat, wrap_v: Wrap::Clamp, filter: Filter::Trilinear, aniso: 16, mip_bias_eighths: 0 };

/// Mip count of a full chain down to 1x1.
pub(crate) fn mip_levels(w: u32, h: u32) -> u32 {
    32 - w.max(h).max(1).leading_zeros()
}

/// Upload an equirect sky (RGBA8 / sRGB / RGBA16F as `GpuTexture::upload`)
/// and build its mip chain by linear blits (all three formats support linear
/// blits on every Vulkan device). Left in SHADER_READ_ONLY_OPTIMAL.
pub(crate) fn upload_equirect(device: &ash::Device, allocator: &mut Allocator, pool: vk::CommandPool, queue: vk::Queue, data: &TextureData) -> Result<GpuTexture> {
    let format = if data.hdr {
        vk::Format::R16G16B16A16_SFLOAT
    } else if data.srgb {
        vk::Format::R8G8B8A8_SRGB
    } else {
        vk::Format::R8G8B8A8_UNORM
    };
    let (w, h) = (data.width.max(1), data.height.max(1));
    // `ZESTENGINE_SKY_NO_MIPS=1`: mip 0 only (the pre-rewrite upload), for A/B comparisons.
    let levels = if std::env::var_os("ZESTENGINE_SKY_NO_MIPS").is_some_and(|v| v != "0") { 1 } else { mip_levels(w, h) };
    let tex = GpuTexture::cleared_mips(device, allocator, pool, queue, w, h, format, vk::ImageUsageFlags::TRANSFER_SRC, levels)?;
    let mut staging = Buffer::new(device, allocator, data.pixels.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "sky staging")?;
    staging.write(0, &data.pixels);
    let image = tex.image;
    let range = |base: u32, count: u32| vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).base_mip_level(base).level_count(count).layer_count(1);
    let layers = |l: u32| vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).mip_level(l).layer_count(1);
    let edge = |l: u32| vk::Offset3D { x: (w >> l).max(1) as i32, y: (h >> l).max(1) as i32, z: 1 };
    let result = one_time_submit(device, pool, queue, |cb| unsafe {
        let barrier = |r: vk::ImageSubresourceRange, old, new, ss, sa, ds, da| {
            let b = [vk::ImageMemoryBarrier2::default()
                .image(image)
                .old_layout(old)
                .new_layout(new)
                .src_stage_mask(ss)
                .src_access_mask(sa)
                .dst_stage_mask(ds)
                .dst_access_mask(da)
                .subresource_range(r)];
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&b));
        };
        let t = vk::PipelineStageFlags2::TRANSFER;
        let (tw, tr) = (vk::AccessFlags2::TRANSFER_WRITE, vk::AccessFlags2::TRANSFER_READ);
        let ro = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let (dst, src) = (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        barrier(range(0, levels), ro, dst, vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::empty(), t, tw);
        let copy = vk::BufferImageCopy::default().image_subresource(layers(0)).image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
        device.cmd_copy_buffer_to_image(cb, staging.handle, image, dst, &[copy]);
        for l in 1..levels {
            barrier(range(l - 1, 1), dst, src, t, tw, t, tr);
            let blit = vk::ImageBlit::default()
                .src_subresource(layers(l - 1))
                .src_offsets([vk::Offset3D::default(), edge(l - 1)])
                .dst_subresource(layers(l))
                .dst_offsets([vk::Offset3D::default(), edge(l)]);
            device.cmd_blit_image(cb, image, src, image, dst, &[blit], vk::Filter::LINEAR);
        }
        let reader = vk::PipelineStageFlags2::FRAGMENT_SHADER;
        let sampled = vk::AccessFlags2::SHADER_SAMPLED_READ;
        if levels > 1 {
            barrier(range(0, levels - 1), src, ro, t, tr, reader, sampled);
        }
        barrier(range(levels - 1, 1), dst, ro, t, tw, reader, sampled);
    });
    staging.destroy(device, allocator);
    if let Err(e) = result {
        let mut tex = tex;
        tex.destroy(device, allocator);
        return Err(e);
    }
    Ok(tex)
}

// ---- CPU copy of the procedural sky in skybox.frag (keep in sync) ----

const BETA_R: [f32; 3] = [5.802e-3, 13.558e-3, 33.1e-3];
const BETA_M: f32 = 3.996e-3;
const BETA_M_EXT: f32 = 4.44e-3;
const H_R: f32 = 8.0;
const H_M: f32 = 1.2;
const R_EARTH: f32 = 6360.0;
const MIE_G: f32 = 0.76;
const SUN_E: f32 = 20.0;
const NIGHT: [f32; 3] = [0.0025, 0.0035, 0.007];
const GROUND_ALBEDO: [f32; 3] = [0.09, 0.085, 0.08];
const PI: f32 = std::f32::consts::PI;

/// The default sun of the procedural sky when no scene light is known (the
/// default `LightData` direction, reversed).
pub(crate) fn default_sun() -> [f32; 3] {
    let (x, y, z) = (0.4f32, 1.0f32, 0.3f32);
    let l = (x * x + y * y + z * z).sqrt();
    [x / l, y / l, z / l]
}

fn air_mass(mu: f32, h: f32) -> f32 {
    let r = R_EARTH / h;
    (r * r * mu * mu + 2.0 * r + 1.0).sqrt() - r * mu
}

fn phase_rayleigh(nu: f32) -> f32 {
    3.0 / (16.0 * PI) * (1.0 + nu * nu)
}

fn phase_mie(nu: f32) -> f32 {
    let g2 = MIE_G * MIE_G;
    3.0 / (8.0 * PI) * ((1.0 - g2) * (1.0 + nu * nu)) / ((2.0 + g2) * (1.0 + g2 - 2.0 * MIE_G * nu).max(1e-4).powf(1.5))
}

fn transmittance(mu: f32) -> [f32; 3] {
    let (mr, mm) = (air_mass(mu, H_R), air_mass(mu, H_M));
    std::array::from_fn(|c| (-(BETA_R[c] * H_R * mr + BETA_M_EXT * H_M * mm)).exp())
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn sky_scatter(v: [f32; 3], s: [f32; 3]) -> [f32; 3] {
    let mu = v[1].max(0.0);
    let nu = v[0] * s[0] + v[1] * s[1] + v[2] * s[2];
    let (mr, mm) = (air_mass(mu, H_R), air_mass(mu, H_M));
    let sun_t = transmittance(s[1]);
    let day = smoothstep(-0.18, 0.08, s[1]);
    std::array::from_fn(|c| {
        let tau_r = BETA_R[c] * H_R * mr;
        let tau_m = BETA_M * H_M * mm;
        let ext = tau_r + BETA_M_EXT * H_M * mm;
        let frac = (1.0 - (-ext).exp()) / ext.max(1e-6);
        let single = (tau_r * phase_rayleigh(nu) + tau_m * phase_mie(nu)) * frac * sun_t[c];
        let multi = (1.0 - (-ext).exp()) * (BETA_R[c] / BETA_R[2]) * 0.006 * day;
        SUN_E * (single + multi) + NIGHT[c]
    })
}

/// Procedural sky radiance for unit direction `v` and unit sun direction `s`
/// (white sun, no disc): `procedural_sky` in skybox.frag.
pub(crate) fn procedural_radiance(v: [f32; 3], s: [f32; 3]) -> [f32; 3] {
    if v[1] >= 0.0 {
        return sky_scatter(v, s);
    }
    let hl = (v[0] * v[0] + v[2] * v[2]).sqrt();
    let hv = if hl > 1e-5 { [v[0] / hl, 0.0, v[2] / hl] } else { [1.0, 0.0, 0.0] };
    let horizon = sky_scatter(hv, s);
    let zenith = sky_scatter([0.0, 1.0, 0.0], s);
    let sun_t = transmittance(s[1]);
    let sun_up = s[1].max(0.0);
    let haze = (-(-v[1]) * 9.0).exp();
    std::array::from_fn(|c| {
        let lit = GROUND_ALBEDO[c] / PI * (SUN_E * sun_t[c] * sun_up * 0.35 + zenith[c] * PI);
        lit + (horizon[c] - lit) * haze
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn procedural_sky_is_blue_overhead_bright_at_the_horizon_and_dark_below() {
        let s = default_sun();
        let zenith = procedural_radiance([0.0, 1.0, 0.0], s);
        let horizon = procedural_radiance([-s[0], 0.0, -s[2]], s);
        let ground = procedural_radiance([0.0, -1.0, 0.0], s);
        assert!(zenith[2] > zenith[1] && zenith[1] > zenith[0], "zenith {zenith:?}");
        assert!((0.15..0.6).contains(&zenith[2]), "zenith level {zenith:?}");
        assert!(horizon[0] > zenith[0] && horizon[1] > zenith[1], "horizon {horizon:?} vs zenith {zenith:?}");
        assert!(ground.iter().all(|&g| g < 0.2 && g > 0.0), "ground {ground:?}");
        // Continuous across the horizon.
        let above = procedural_radiance([1.0, 1e-4, 0.0], s);
        let below = procedural_radiance([1.0, -1e-4, 0.0], s);
        assert!((0..3).all(|c| (above[c] - below[c]).abs() < 0.01 * above[c].max(0.01)), "{above:?} vs {below:?}");
        // Night: sun well below the horizon -> dark but not black, finite.
        let night = procedural_radiance([0.0, 1.0, 0.0], [0.0, -0.6, 0.8]);
        assert!(night.iter().all(|&c| c.is_finite() && c > 0.0 && c < 0.02), "night {night:?}");
    }

    #[test]
    fn mip_chain_reaches_one_texel() {
        assert_eq!(mip_levels(4096, 2048), 13);
        assert_eq!(mip_levels(1, 1), 1);
        assert_eq!(mip_levels(300, 7), 9);
    }
}
