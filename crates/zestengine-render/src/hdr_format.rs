//! HDR texture formats for baked lighting (lightmap arrays, reflection-probe
//! cubes) and the device-driven choice of which one sits in VRAM.
//!
//! Bakes are stored on disk as BC6H (UF16) by default, independent of the
//! machine that baked them, so one file ships to every platform. At load the
//! loader thread asks [`HdrFormatSupport::upload_format`] what the device can
//! sample and transcodes if needed: BC6H stays BC6H when `textureCompressionBC`
//! is enabled, otherwise it is decoded to the compact shared-exponent
//! E5B9G9R9 (4 B/texel) or, failing that, RGBA16F (8 B/texel). Uncompressed
//! bakes (the per-scene opt-out) load as RGBA16F.
//!
//! ASTC 4x4 HDR (vendored ARM astcenc) is what a Meta Quest build ships: its
//! cook re-encodes BC6H bakes as ASTC HDR, which the headset samples directly.
//! Stored ASTC HDR data on a device without support decodes to E5B9G9R9 /
//! RGBA16F like BC6H does.

use std::sync::atomic::{AtomicBool, Ordering};

use ash::vk;

use crate::types::TextureData;

/// Pixel format of baked HDR texture data, on disk and in VRAM. The numeric
/// codes are part of the `.lightmap` v2 / `.reflprobe` v4 file formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HdrTextureFormat {
    /// RGBA f32 (16 B/texel): the raw bake output.
    Rgba32f,
    /// RGBA f16 (8 B/texel).
    Rgba16f,
    /// Shared-exponent RGB 9:9:9:5 (4 B/texel), no alpha.
    Rgb9e5,
    /// BC6H unsigned float, 4x4 blocks of 16 B (1 B/texel), no alpha.
    Bc6h,
    /// ASTC 4x4 HDR (signed float), 4x4 blocks of 16 B (1 B/texel).
    AstcHdr4x4,
}

impl HdrTextureFormat {
    pub fn code(self) -> u32 {
        match self {
            Self::Rgba32f => 0,
            Self::Rgba16f => 1,
            Self::Rgb9e5 => 2,
            Self::Bc6h => 3,
            Self::AstcHdr4x4 => 4,
        }
    }

    pub fn from_code(code: u32) -> Option<Self> {
        Some(match code {
            0 => Self::Rgba32f,
            1 => Self::Rgba16f,
            2 => Self::Rgb9e5,
            3 => Self::Bc6h,
            4 => Self::AstcHdr4x4,
            _ => return None,
        })
    }

    pub fn is_block_compressed(self) -> bool {
        matches!(self, Self::Bc6h | Self::AstcHdr4x4)
    }

    /// Exact byte size of one `w`×`h` level. Block formats round each
    /// dimension up to the 4x4 block grid (Vulkan's layout for partial blocks).
    pub fn level_bytes(self, w: u32, h: u32) -> usize {
        let (w, h) = (w.max(1) as usize, h.max(1) as usize);
        match self {
            Self::Rgba32f => w * h * 16,
            Self::Rgba16f => w * h * 8,
            Self::Rgb9e5 => w * h * 4,
            Self::Bc6h | Self::AstcHdr4x4 => w.div_ceil(4) * h.div_ceil(4) * 16,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Rgba32f => "RGBA32F",
            Self::Rgba16f => "RGBA16F",
            Self::Rgb9e5 => "E5B9G9R9",
            Self::Bc6h => "BC6H",
            Self::AstcHdr4x4 => "ASTC 4x4 HDR",
        }
    }

    pub(crate) fn vk_format(self) -> vk::Format {
        match self {
            Self::Rgba32f => vk::Format::R32G32B32A32_SFLOAT,
            Self::Rgba16f => vk::Format::R16G16B16A16_SFLOAT,
            Self::Rgb9e5 => vk::Format::E5B9G9R9_UFLOAT_PACK32,
            Self::Bc6h => vk::Format::BC6H_UFLOAT_BLOCK,
            Self::AstcHdr4x4 => vk::Format::ASTC_4X4_SFLOAT_BLOCK,
        }
    }
}

/// Which HDR formats the active device can sample with linear filtering.
/// ASTC HDR sampling is read from the device flag
/// ([`crate::astc_hdr_supported`]) by [`Self::upload_format`], or passed to
/// [`Self::upload_format_with`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HdrFormatSupport {
    /// `textureCompressionBC` enabled (BC6H sampleable).
    pub bc6h: bool,
    /// E5B9G9R9_UFLOAT_PACK32 supports SAMPLED_IMAGE + FILTER_LINEAR + TRANSFER_DST.
    pub rgb9e5: bool,
}

static RGB9E5_SUPPORTED: AtomicBool = AtomicBool::new(true);

/// Record E5B9G9R9 sampling support (called from device creation).
pub fn set_rgb9e5_supported(v: bool) {
    RGB9E5_SUPPORTED.store(v, Ordering::Relaxed);
}

impl HdrFormatSupport {
    /// The active device's support (both default true until device creation).
    pub fn current() -> Self {
        Self { bc6h: crate::bc_supported(), rgb9e5: RGB9E5_SUPPORTED.load(Ordering::Relaxed) }
    }

    /// Smallest uncompressed format the device samples.
    pub fn compact_uncompressed(self) -> HdrTextureFormat {
        if self.rgb9e5 { HdrTextureFormat::Rgb9e5 } else { HdrTextureFormat::Rgba16f }
    }

    /// VRAM format for data stored as `stored`. Compressed data the device
    /// can't sample decodes to the compact fallback; uncompressed bakes keep
    /// half-float precision.
    pub fn upload_format(self, stored: HdrTextureFormat) -> HdrTextureFormat {
        self.upload_format_with(stored, crate::astc_hdr_supported())
    }

    /// [`Self::upload_format`] with ASTC HDR sampling support given. ASTC HDR
    /// the device can't sample maps to the compact fallback, which the codec
    /// then refuses (no decoder), making it a clean load error.
    pub fn upload_format_with(self, stored: HdrTextureFormat, astc_hdr: bool) -> HdrTextureFormat {
        match stored {
            HdrTextureFormat::Bc6h if self.bc6h => HdrTextureFormat::Bc6h,
            HdrTextureFormat::AstcHdr4x4 if astc_hdr => HdrTextureFormat::AstcHdr4x4,
            HdrTextureFormat::Bc6h | HdrTextureFormat::AstcHdr4x4 => self.compact_uncompressed(),
            HdrTextureFormat::Rgb9e5 => self.compact_uncompressed(),
            HdrTextureFormat::Rgba16f | HdrTextureFormat::Rgba32f => HdrTextureFormat::Rgba16f,
        }
    }

    /// Whether this device samples `format` at all (uploading anything else
    /// is refused rather than handed to the driver).
    pub fn can_sample(self, format: HdrTextureFormat, astc_hdr: bool) -> bool {
        match format {
            HdrTextureFormat::Bc6h => self.bc6h,
            HdrTextureFormat::AstcHdr4x4 => astc_hdr,
            HdrTextureFormat::Rgb9e5 => self.rgb9e5,
            HdrTextureFormat::Rgba16f | HdrTextureFormat::Rgba32f => true,
        }
    }
}

/// Format new bakes are written in: BC6H unless the scene opts out. Not
/// device-dependent, so a bake made anywhere loads everywhere.
pub fn bake_storage_format(compress: bool) -> HdrTextureFormat {
    if compress { HdrTextureFormat::Bc6h } else { HdrTextureFormat::Rgba32f }
}

/// A lightmap array ready for upload: `layers` square layers of `size`, each
/// exactly `format.level_bytes(size, size)` bytes, no mips.
#[derive(Clone, Debug)]
pub struct HdrTextureArray {
    pub format: HdrTextureFormat,
    pub size: u32,
    pub layers: Vec<Vec<u8>>,
}

impl HdrTextureArray {
    pub fn byte_size(&self) -> u64 {
        self.layers.iter().map(|l| l.len() as u64).sum()
    }
}

/// A reflection cube with its full prefiltered mip chain: `faces[f][mip]`
/// (+X,-X,+Y,-Y,+Z,-Z), mip `m` is `max(size >> m, 1)` square, and there are
/// `cube_mip_count(size)` mips (down to 1x1), matching the live capture.
#[derive(Clone, Debug)]
pub struct HdrCube {
    pub format: HdrTextureFormat,
    pub size: u32,
    pub faces: [Vec<Vec<u8>>; 6],
}

impl HdrCube {
    pub fn mips(&self) -> u32 {
        self.faces[0].len() as u32
    }

    pub fn byte_size(&self) -> u64 {
        self.faces.iter().flatten().map(|l| l.len() as u64).sum()
    }

    /// Every face has the full chain and every level its exact byte size.
    pub fn is_well_formed(&self) -> bool {
        let mips = crate::cube_mip_count(self.size);
        self.faces.iter().all(|f| {
            f.len() == mips as usize
                && f.iter().enumerate().all(|(m, l)| {
                    let s = (self.size >> m).max(1);
                    l.len() == self.format.level_bytes(s, s)
                })
        })
    }
}

/// Largest face size a baked cube keeps (larger captures are resampled).
pub const MAX_BAKED_CUBE_SIZE: u32 = 1024;

/// The CPU prefiltered chain (RGBA16F) for 6 captured faces: mip 0 is the faces
/// resampled to at most [`MAX_BAKED_CUBE_SIZE`], rougher mips box-downsample.
/// Identical to what the uncompressed load path uploads.
pub fn reflection_cube_chain(faces: &[TextureData; 6]) -> HdrCube {
    crate::reflection_chain_from_faces(faces)
}

// ---- texel conversions ----------------------------------------------------

/// f32 → f16 clamped to the unsigned-float range BC6H UF16 / E5B9G9R9 hold.
pub fn f32_to_uf16(v: f32) -> half::f16 {
    half::f16::from_f32(if v.is_nan() { 0.0 } else { v.clamp(0.0, 65504.0) })
}

/// RGBA f32 texels → RGBA f16 bytes (little endian), clamped non-negative.
pub fn rgba32f_to_rgba16f(src: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len() * 2);
    for &v in src {
        out.extend_from_slice(&f32_to_uf16(v).to_le_bytes());
    }
    out
}

/// RGBA f16 bytes → RGBA f32 texels.
pub fn rgba16f_to_rgba32f(src: &[u8]) -> Vec<f32> {
    src.as_chunks::<2>().0.iter().map(|b| half::f16::from_le_bytes(*b).to_f32()).collect()
}

const RGB9E5_MANTISSA: i32 = 9;
const RGB9E5_BIAS: i32 = 15;
const RGB9E5_MAX: f32 = 511.0 / 512.0 * 65536.0;

/// Pack linear RGB into E5B9G9R9 (Vulkan spec "Shared Exponent" conversion).
pub fn pack_rgb9e5(rgb: [f32; 3]) -> u32 {
    let c = rgb.map(|v| if v.is_nan() { 0.0 } else { v.clamp(0.0, RGB9E5_MAX) });
    let max = c[0].max(c[1]).max(c[2]);
    let floor_log2 = if max > 0.0 { max.log2().floor() as i32 } else { -RGB9E5_BIAS - 1 };
    let mut exp = floor_log2.max(-RGB9E5_BIAS - 1) + 1 + RGB9E5_BIAS;
    let scale = |e: i32| 2f32.powi(e - RGB9E5_BIAS - RGB9E5_MANTISSA);
    if (max / scale(exp) + 0.5).floor() as i32 == 1 << RGB9E5_MANTISSA {
        exp += 1;
    }
    let s = scale(exp);
    let m = c.map(|v| ((v / s + 0.5).floor() as u32).min(511));
    m[0] | (m[1] << 9) | (m[2] << 18) | ((exp as u32) << 27)
}

pub fn unpack_rgb9e5(p: u32) -> [f32; 3] {
    let exp = (p >> 27) as i32;
    let s = 2f32.powi(exp - RGB9E5_BIAS - RGB9E5_MANTISSA);
    [(p & 511) as f32 * s, ((p >> 9) & 511) as f32 * s, ((p >> 18) & 511) as f32 * s]
}

/// Decode one level of an uncompressed format to RGBA f16 bytes.
fn to_rgba16f(src: &[u8], from: HdrTextureFormat) -> Option<Vec<u8>> {
    Some(match from {
        HdrTextureFormat::Rgba16f => src.to_vec(),
        HdrTextureFormat::Rgba32f => {
            let f: Vec<f32> = src.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
            rgba32f_to_rgba16f(&f)
        }
        HdrTextureFormat::Rgb9e5 => {
            let mut out = Vec::with_capacity(src.len() * 2);
            for b in src.as_chunks::<4>().0 {
                let [r, g, b] = unpack_rgb9e5(u32::from_le_bytes(*b));
                for v in [r, g, b, 1.0] {
                    out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
                }
            }
            out
        }
        HdrTextureFormat::Bc6h | HdrTextureFormat::AstcHdr4x4 => return None,
    })
}

/// Convert one level between uncompressed formats (None for a block format
/// on either side; the asset codec handles those).
pub fn convert_uncompressed(src: &[u8], from: HdrTextureFormat, to: HdrTextureFormat) -> Option<Vec<u8>> {
    if from == to {
        return Some(src.to_vec());
    }
    let half = to_rgba16f(src, from)?;
    Some(match to {
        HdrTextureFormat::Rgba16f => half,
        HdrTextureFormat::Rgba32f => rgba16f_to_rgba32f(&half).iter().flat_map(|v| v.to_le_bytes()).collect(),
        HdrTextureFormat::Rgb9e5 => {
            let f = rgba16f_to_rgba32f(&half);
            f.as_chunks::<4>().0.iter().flat_map(|p| pack_rgb9e5([p[0], p[1], p[2]]).to_le_bytes()).collect()
        }
        HdrTextureFormat::Bc6h | HdrTextureFormat::AstcHdr4x4 => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upload_format_falls_back_without_bc() {
        let all = HdrFormatSupport { bc6h: true, rgb9e5: true };
        let no_bc = HdrFormatSupport { bc6h: false, rgb9e5: true };
        let bare = HdrFormatSupport { bc6h: false, rgb9e5: false };
        assert_eq!(all.upload_format(HdrTextureFormat::Bc6h), HdrTextureFormat::Bc6h);
        assert_eq!(no_bc.upload_format(HdrTextureFormat::Bc6h), HdrTextureFormat::Rgb9e5);
        assert_eq!(bare.upload_format(HdrTextureFormat::Bc6h), HdrTextureFormat::Rgba16f);
        assert_eq!(bare.upload_format(HdrTextureFormat::Rgb9e5), HdrTextureFormat::Rgba16f);
        for s in [all, no_bc, bare] {
            assert_eq!(s.upload_format(HdrTextureFormat::Rgba32f), HdrTextureFormat::Rgba16f);
            assert_eq!(s.upload_format(HdrTextureFormat::Rgba16f), HdrTextureFormat::Rgba16f);
        }
        assert_eq!(bake_storage_format(true), HdrTextureFormat::Bc6h);
        assert_eq!(bake_storage_format(false), HdrTextureFormat::Rgba32f);
    }

    #[test]
    fn format_codes_round_trip() {
        let all = [
            HdrTextureFormat::Rgba32f,
            HdrTextureFormat::Rgba16f,
            HdrTextureFormat::Rgb9e5,
            HdrTextureFormat::Bc6h,
            HdrTextureFormat::AstcHdr4x4,
        ];
        for f in all {
            assert_eq!(HdrTextureFormat::from_code(f.code()), Some(f));
        }
        // Codes are file-format values: existing ones never move.
        assert_eq!(all.map(HdrTextureFormat::code), [0, 1, 2, 3, 4]);
        assert_eq!(HdrTextureFormat::from_code(99), None);
    }

    #[test]
    fn astc_hdr_is_kept_only_when_sampleable() {
        let all = HdrFormatSupport { bc6h: true, rgb9e5: true };
        let mobile = HdrFormatSupport { bc6h: false, rgb9e5: true };
        let bare = HdrFormatSupport { bc6h: false, rgb9e5: false };
        let astc = HdrTextureFormat::AstcHdr4x4;
        assert_eq!(mobile.upload_format_with(astc, true), astc);
        assert_eq!(mobile.upload_format_with(astc, false), HdrTextureFormat::Rgb9e5);
        assert_eq!(bare.upload_format_with(astc, false), HdrTextureFormat::Rgba16f);
        assert_eq!(all.upload_format_with(astc, false), HdrTextureFormat::Rgb9e5);
        // ASTC HDR support never changes the choice for BC6H bakes.
        assert_eq!(mobile.upload_format_with(HdrTextureFormat::Bc6h, true), HdrTextureFormat::Rgb9e5);
        assert_eq!(all.upload_format_with(HdrTextureFormat::Bc6h, true), HdrTextureFormat::Bc6h);
        assert!(mobile.can_sample(astc, true) && !mobile.can_sample(astc, false));
        assert!(!mobile.can_sample(HdrTextureFormat::Bc6h, true) && all.can_sample(HdrTextureFormat::Bc6h, false));
        assert!(!bare.can_sample(HdrTextureFormat::Rgb9e5, true));
        assert!(astc.is_block_compressed());
        assert_eq!(astc.vk_format(), vk::Format::ASTC_4X4_SFLOAT_BLOCK);
        assert_eq!(astc.label(), "ASTC 4x4 HDR");
        assert!(convert_uncompressed(&[0; 16], astc, HdrTextureFormat::Rgba16f).is_none());
        assert!(convert_uncompressed(&[0; 8], HdrTextureFormat::Rgba16f, astc).is_none());
    }

    #[test]
    fn block_sizes_round_up_partial_blocks() {
        assert_eq!(HdrTextureFormat::Bc6h.level_bytes(4, 4), 16);
        assert_eq!(HdrTextureFormat::Bc6h.level_bytes(5, 3), 2 * 16);
        assert_eq!(HdrTextureFormat::Bc6h.level_bytes(1, 1), 16);
        assert_eq!(HdrTextureFormat::Bc6h.level_bytes(1022, 1022), 256 * 256 * 16);
        assert_eq!(HdrTextureFormat::Rgb9e5.level_bytes(3, 3), 36);
        for (w, h) in [(4, 4), (5, 3), (1, 1), (30, 30)] {
            assert_eq!(HdrTextureFormat::AstcHdr4x4.level_bytes(w, h), HdrTextureFormat::Bc6h.level_bytes(w, h));
        }
    }

    #[test]
    fn rgb9e5_round_trips_within_precision() {
        for rgb in [[0.0, 0.0, 0.0], [1.0, 0.5, 0.25], [1000.0, 3.0, 0.01], [0.001, 0.002, 0.0005], [60000.0, 1.0, 0.0]] {
            let back = unpack_rgb9e5(pack_rgb9e5(rgb));
            let max = rgb.iter().cloned().fold(0.0f32, f32::max);
            for c in 0..3 {
                // 9-bit mantissa shared with the largest channel.
                assert!((back[c] - rgb[c]).abs() <= max / 256.0 + 1e-7, "{rgb:?} -> {back:?}");
            }
        }
        assert_eq!(unpack_rgb9e5(pack_rgb9e5([-5.0, f32::NAN, 0.0])), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn uncompressed_conversions_agree() {
        let src = [0.5f32, 2.0, 100.0, 1.0, -1.0, 0.0, 70000.0, 1.0];
        let half = rgba32f_to_rgba16f(&src);
        let back = rgba16f_to_rgba32f(&half);
        assert_eq!(back[0], 0.5);
        assert_eq!(back[4], 0.0, "negatives clamp for unsigned formats");
        assert_eq!(back[6], 65504.0, "overflow clamps to f16 max");
        let e5 = convert_uncompressed(&half, HdrTextureFormat::Rgba16f, HdrTextureFormat::Rgb9e5).unwrap();
        assert_eq!(e5.len(), 8);
        let again = convert_uncompressed(&e5, HdrTextureFormat::Rgb9e5, HdrTextureFormat::Rgba16f).unwrap();
        let f = rgba16f_to_rgba32f(&again);
        assert!((f[2] - 100.0).abs() < 0.5 && (f[0] - 0.5).abs() < 0.5);
        assert!(convert_uncompressed(&half, HdrTextureFormat::Rgba16f, HdrTextureFormat::Bc6h).is_none());
    }
}

/// Upload smoke test on the real GPU (run with `--ignored`): BC6H / E5B9G9R9
/// lightmap arrays and cubes upload, and the blocks read back bit-exact.
#[cfg(test)]
mod gpu_smoke {
    use super::*;
    use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

    fn read_layer(
        device: &ash::Device,
        alloc: &mut Allocator,
        pool: vk::CommandPool,
        queue: vk::Queue,
        image: vk::Image,
        (layer, mip, size, bytes): (u32, u32, u32, usize),
    ) -> Vec<u8> {
        let mut buf = crate::alloc::Buffer::new(
            device,
            alloc,
            bytes as u64,
            vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::GpuToCpu,
            "smoke readback",
        )
        .unwrap();
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(mip)
            .level_count(1)
            .base_array_layer(layer)
            .layer_count(1);
        crate::alloc::one_time_submit(device, pool, queue, |cb| unsafe {
            let b = vk::ImageMemoryBarrier::default()
                .image(image)
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .subresource_range(range);
            device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[b]);
            let copy = vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(mip)
                        .base_array_layer(layer)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D { width: size, height: size, depth: 1 });
            device.cmd_copy_image_to_buffer(cb, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buf.handle, &[copy]);
        })
        .unwrap();
        let out = buf.read()[..bytes].to_vec();
        buf.destroy(device, alloc);
        out
    }

    #[test]
    #[ignore]
    fn bc6h_and_rgb9e5_uploads_read_back_exactly() {
        let entry = unsafe { ash::Entry::load() }.unwrap();
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = unsafe { entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None) }.unwrap();
        let pd = unsafe { instance.enumerate_physical_devices() }.unwrap()[0];
        let qf = unsafe { instance.get_physical_device_queue_family_properties(pd) }
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .unwrap() as u32;
        let bc = unsafe { instance.get_physical_device_features(pd) }.texture_compression_bc == vk::TRUE;
        let prio = [1.0];
        let qi = [vk::DeviceQueueCreateInfo::default().queue_family_index(qf).queue_priorities(&prio)];
        let features = vk::PhysicalDeviceFeatures::default().texture_compression_bc(bc);
        let mut vk13 = vk::PhysicalDeviceVulkan13Features::default().synchronization2(true);
        let device = unsafe {
            instance.create_device(pd, &vk::DeviceCreateInfo::default().queue_create_infos(&qi).enabled_features(&features).push_next(&mut vk13), None)
        }
        .unwrap();
        let queue = unsafe { device.get_device_queue(qf, 0) };
        let pool = unsafe {
            device.create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(qf).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER), None)
        }
        .unwrap();
        let mut alloc = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: pd,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
        .unwrap();

        let mut formats = vec![HdrTextureFormat::Rgb9e5, HdrTextureFormat::Rgba16f];
        if bc {
            formats.push(HdrTextureFormat::Bc6h);
        }
        for format in formats {
            // Lightmap array: 2 layers of a non-multiple-of-4 size.
            let size = 30;
            let len = format.level_bytes(size, size);
            let layers: Vec<Vec<u8>> = (0..2u8).map(|l| (0..len).map(|i| (i as u8).wrapping_mul(7) ^ l).collect()).collect();
            let array = HdrTextureArray { format, size, layers: layers.clone() };
            let mut tex = crate::texture::GpuTexture::upload_lightmap_array(&device, &mut alloc, pool, queue, &array).unwrap();
            for (l, want) in layers.iter().enumerate() {
                let got = read_layer(&device, &mut alloc, pool, queue, tex.image, (l as u32, 0, size, len));
                assert_eq!(&got, want, "{} lightmap layer {l}", format.label());
            }
            tex.destroy(&device, &mut alloc);

            // Cube with a full mip chain down to 1x1.
            let size = 20;
            let mips = crate::cube_mip_count(size);
            let faces: [Vec<Vec<u8>>; 6] = std::array::from_fn(|f| {
                (0..mips)
                    .map(|m| {
                        let s = (size >> m).max(1);
                        (0..format.level_bytes(s, s)).map(|i| (i as u8).wrapping_add(f as u8 * 31 + m as u8)).collect()
                    })
                    .collect()
            });
            let cube = HdrCube { format, size, faces };
            let mut env = crate::ReflectionEnv::upload_chain(&device, &mut alloc, pool, queue, &cube).unwrap();
            assert_eq!(env.format, format.vk_format());
            for (f, chain) in cube.faces.iter().enumerate() {
                for (m, want) in chain.iter().enumerate() {
                    let s = (size >> m).max(1);
                    let got = read_layer(&device, &mut alloc, pool, queue, env.image, (f as u32, m as u32, s, want.len()));
                    assert_eq!(&got, want, "{} cube face {f} mip {m}", format.label());
                }
            }
            env.destroy(&device, &mut alloc);
        }
        drop(alloc);
        unsafe {
            device.destroy_command_pool(pool, None);
            device.destroy_device(None);
            instance.destroy_instance(None);
        }
    }
}
