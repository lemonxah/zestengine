//! Per-texture sampler state (TEX-2): wrap U/V, filter, anisotropy, mip bias.
//!
//! Material texture slots are `COMBINED_IMAGE_SAMPLER` bindings, so each slot's
//! descriptor can carry its own sampler without touching the descriptor-set
//! layout. [`SamplerCache`] makes one `VkSampler` per distinct [`SamplerKey`] on
//! first use and keeps them for the renderer's lifetime (a handful at most).
//! The default key maps to the renderer's shared sampler, so a project that
//! never changes sampler settings creates nothing and takes no map lookup.

use std::collections::HashMap;

use ash::vk;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Wrap {
    #[default]
    Repeat,
    Clamp,
    Mirror,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Filter {
    Point,
    Bilinear,
    #[default]
    Trilinear,
}

/// What a texture asks of its sampler. `Default` matches the renderer's shared
/// material sampler (repeat, trilinear, 8x anisotropy, no bias).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SamplerKey {
    pub wrap_u: Wrap,
    pub wrap_v: Wrap,
    pub filter: Filter,
    /// Requested anisotropy (1 = off); clamped to the device limit.
    pub aniso: u8,
    /// Mip LOD bias in 1/8 steps (positive = blurrier).
    pub mip_bias_eighths: i8,
}

impl Default for SamplerKey {
    fn default() -> Self {
        Self { wrap_u: Wrap::Repeat, wrap_v: Wrap::Repeat, filter: Filter::Trilinear, aniso: 8, mip_bias_eighths: 0 }
    }
}

impl SamplerKey {
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Point sampling and one-texel clamps make anisotropy meaningless; fold it
    /// out so equal samplers share one `VkSampler`.
    fn normalized(mut self) -> Self {
        if self.filter == Filter::Point {
            self.aniso = 1;
        }
        self.aniso = self.aniso.max(1);
        self
    }
}

/// `maxSamplerLodBias` is at least 2.0 on every device; stay inside it.
const MAX_LOD_BIAS: f32 = 2.0;

fn address_mode(w: Wrap) -> vk::SamplerAddressMode {
    match w {
        Wrap::Repeat => vk::SamplerAddressMode::REPEAT,
        Wrap::Clamp => vk::SamplerAddressMode::CLAMP_TO_EDGE,
        Wrap::Mirror => vk::SamplerAddressMode::MIRRORED_REPEAT,
    }
}

/// The `VkSamplerCreateInfo` for a key; `max_aniso` is the device-clamped
/// ceiling (1.0 when the feature is off).
pub fn create_info(key: SamplerKey, max_aniso: f32) -> vk::SamplerCreateInfo<'static> {
    let key = key.normalized();
    let (min_mag, mip) = match key.filter {
        Filter::Point => (vk::Filter::NEAREST, vk::SamplerMipmapMode::NEAREST),
        Filter::Bilinear => (vk::Filter::LINEAR, vk::SamplerMipmapMode::NEAREST),
        Filter::Trilinear => (vk::Filter::LINEAR, vk::SamplerMipmapMode::LINEAR),
    };
    let aniso = (key.aniso as f32).clamp(1.0, max_aniso.max(1.0));
    vk::SamplerCreateInfo::default()
        .mag_filter(min_mag)
        .min_filter(min_mag)
        .mipmap_mode(mip)
        .address_mode_u(address_mode(key.wrap_u))
        .address_mode_v(address_mode(key.wrap_v))
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .mip_lod_bias((key.mip_bias_eighths as f32 / 8.0).clamp(-MAX_LOD_BIAS, MAX_LOD_BIAS))
        .anisotropy_enable(aniso > 1.0)
        .max_anisotropy(aniso)
        .max_lod(vk::LOD_CLAMP_NONE)
}

/// Samplers by key, plus which key each texture handle wants.
pub struct SamplerCache {
    /// The shared material sampler (created by the renderer, not owned here).
    default: vk::Sampler,
    max_aniso: f32,
    made: HashMap<SamplerKey, vk::Sampler>,
    /// Texture index -> key, only for non-default textures.
    by_texture: HashMap<usize, SamplerKey>,
}

impl SamplerCache {
    pub fn new(default: vk::Sampler, max_aniso: f32) -> Self {
        Self { default, max_aniso, made: HashMap::new(), by_texture: HashMap::new() }
    }

    /// Record what texture `index` wants. Returns true if that changed.
    pub fn set_texture(&mut self, index: usize, key: SamplerKey) -> bool {
        let key = key.normalized();
        if key == SamplerKey::default().normalized() {
            self.by_texture.remove(&index).is_some()
        } else {
            self.by_texture.insert(index, key) != Some(key)
        }
    }

    /// The sampler for texture `index`: the shared one unless it asked for
    /// something else (then the cached sampler, created on first use).
    pub fn for_texture(&mut self, device: &ash::Device, index: usize) -> vk::Sampler {
        match self.by_texture.get(&index) {
            None => self.default,
            Some(&key) => self.get(device, key),
        }
    }

    pub fn get(&mut self, device: &ash::Device, key: SamplerKey) -> vk::Sampler {
        let key = key.normalized();
        if key == SamplerKey::default().normalized() {
            return self.default;
        }
        if let Some(&s) = self.made.get(&key) {
            return s;
        }
        match unsafe { device.create_sampler(&create_info(key, self.max_aniso), None) } {
            Ok(s) => {
                self.made.insert(key, s);
                s
            }
            Err(e) => {
                tracing::warn!("creating sampler {key:?}: {e}; using the shared sampler");
                self.default
            }
        }
    }

    /// Number of non-default samplers created so far.
    #[cfg(test)]
    pub fn created(&self) -> usize {
        self.made.len()
    }

    /// Destroy the created samplers. The caller guarantees the GPU is done.
    pub fn destroy(&mut self, device: &ash::Device) {
        for (_, s) in self.made.drain() {
            unsafe { device.destroy_sampler(s, None) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_key_is_the_shared_sampler() {
        let mut c = SamplerCache::new(vk::Sampler::null(), 16.0);
        assert!(!c.set_texture(3, SamplerKey::default()), "default stores nothing");
        assert!(c.by_texture.is_empty());
        assert_eq!(c.created(), 0);
    }

    #[test]
    fn non_default_keys_are_tracked_and_reverted() {
        let mut c = SamplerCache::new(vk::Sampler::null(), 16.0);
        let clamp = SamplerKey { wrap_u: Wrap::Clamp, wrap_v: Wrap::Clamp, ..SamplerKey::default() };
        assert!(c.set_texture(1, clamp));
        assert!(!c.set_texture(1, clamp), "same key again is no change");
        assert!(c.set_texture(1, SamplerKey::default()), "back to default clears the entry");
        assert!(c.by_texture.is_empty());
    }

    #[test]
    fn mip_bias_is_part_of_the_key_and_stays_inside_the_device_limit() {
        let a = SamplerKey { mip_bias_eighths: 4, ..SamplerKey::default() };
        let b = SamplerKey { mip_bias_eighths: 8, ..SamplerKey::default() };
        assert!(!a.is_default());
        assert_ne!(a.normalized(), b.normalized(), "different bias = different VkSampler");
        let mut c = SamplerCache::new(vk::Sampler::null(), 16.0);
        assert!(c.set_texture(2, a));
        assert!(c.set_texture(2, b), "changing only the bias is a change (callers rebind)");
        assert!(!c.set_texture(2, b));
        assert_eq!(create_info(SamplerKey { mip_bias_eighths: 100, ..SamplerKey::default() }, 8.0).mip_lod_bias, 2.0);
        assert_eq!(create_info(SamplerKey { mip_bias_eighths: -100, ..SamplerKey::default() }, 8.0).mip_lod_bias, -2.0);
        // Point sampling keeps the bias distinct from the plain point key.
        let p0 = SamplerKey { filter: Filter::Point, ..SamplerKey::default() }.normalized();
        let p1 = SamplerKey { filter: Filter::Point, mip_bias_eighths: 8, ..SamplerKey::default() }.normalized();
        assert_ne!(p0, p1);
    }

    #[test]
    fn point_filter_folds_anisotropy_away() {
        let a = SamplerKey { filter: Filter::Point, aniso: 16, ..SamplerKey::default() }.normalized();
        let b = SamplerKey { filter: Filter::Point, aniso: 2, ..SamplerKey::default() }.normalized();
        assert_eq!(a, b);
        let info = create_info(a, 16.0);
        assert_eq!(info.anisotropy_enable, vk::FALSE);
        assert_eq!(info.mag_filter, vk::Filter::NEAREST);
        assert_eq!(info.mipmap_mode, vk::SamplerMipmapMode::NEAREST);
    }

    #[test]
    fn create_info_maps_wrap_filter_aniso_and_bias() {
        let key = SamplerKey {
            wrap_u: Wrap::Mirror,
            wrap_v: Wrap::Clamp,
            filter: Filter::Bilinear,
            aniso: 16,
            mip_bias_eighths: -4,
        };
        let info = create_info(key, 4.0);
        assert_eq!(info.address_mode_u, vk::SamplerAddressMode::MIRRORED_REPEAT);
        assert_eq!(info.address_mode_v, vk::SamplerAddressMode::CLAMP_TO_EDGE);
        assert_eq!(info.mipmap_mode, vk::SamplerMipmapMode::NEAREST);
        assert_eq!(info.max_anisotropy, 4.0, "clamped to the device limit");
        assert_eq!(info.mip_lod_bias, -0.5);
        assert_eq!(create_info(SamplerKey { aniso: 8, ..SamplerKey::default() }, 1.0).anisotropy_enable, vk::FALSE, "no feature = off");
    }
}
