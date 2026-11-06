//! Texture import cache: decode a source image once, generate a mip chain, and
//! block-compress it, persisting the result to disk so later loads skip the
//! expensive decode + recompress entirely.
//!
//! Which encoding depends on what the device samples: BC (BC7 for LDR
//! color/data, BC6H for HDR) on desktop GPUs, ASTC LDR on mobile / standalone
//! VR GPUs without BC (HDR sources as ASTC 4x4 HDR where the GPU samples it,
//! raw RGBA16F otherwise), raw RGBA
//! otherwise. Per-texture import settings (`TextureImport`, from the `.meta`)
//! pick the format (BC1 / BC4 / BC5 / BC7 / BC6H, ASTC footprints) and drive
//! the pixel steps in `texture_pipeline`. The cache lives in a sibling
//! `.zestengine_texcache/` next to the source, keyed by (filename, hash of the
//! import settings that change pixels): `<name>.<hash>.cbc` for the BC / raw
//! tier and `<name>.<hash>.castc` for the ASTC tier, so both coexist. It is
//! invalidated when the source's mtime or size changes, or when the format
//! version bumps, and a cached format the device can't sample is re-encoded,
//! never uploaded. All work here is CPU-only and runs on the loader thread;
//! the GPU upload (`GpuTexture::upload_compressed`) just stages the cached
//! bytes.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result};
use zestengine_render::{CompressedTexture, TexFormat};

use crate::material_file::load_texture_file;
use crate::texture_import::{
    AlphaSource, Compression, MipFilter, Platform, SourceInfo, TargetFormat, TextureImport, TextureKind, resolve_import,
};
use crate::texture_pipeline::{self as pipeline, MipParams, srgb_to_linear};

/// Bumped to 03 when mips moved to linear-space filtering, a full chain to 1x1
/// and padded (not raw) BC for sizes that aren't a multiple of 4; to 04 with
/// per-texture import settings (BC1 / BC4 / BC5 formats, settings-hash cache
/// keys), so stale caches re-encode.
const MAGIC: &[u8; 8] = b"CITRBC04";
/// Extension of the ASTC-tier cache artefact (the BC / raw tier uses `cbc`).
pub const ASTC_CACHE_EXTENSION: &str = "castc";

/// ASTC block footprints for the mobile texture path, as square block edges
/// (4, 5, 6 or 8; larger is smaller and blurrier). `color` applies to sRGB
/// slots (albedo, emission, matcaps), `data` to linear slots (normal, ORM,
/// masks), which need more precision. Default 6x6 / 4x4 (3.56 / 8 bits per
/// texel). `ZESTENGINE_ASTC_BLOCKS=<color>,<data>` (`6x6,4x4` or `6,4`)
/// overrides both the default and [`set_astc_blocks`]. A change affects new
/// encodes only: a device load accepts an ASTC cache of any footprint, while
/// the Android cook ([`astc_cache_blob`]) re-encodes to the current setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AstcBlocks {
    pub color: u32,
    pub data: u32,
}

impl Default for AstcBlocks {
    fn default() -> Self {
        Self { color: 6, data: 4 }
    }
}

impl AstcBlocks {
    /// Parse `<color>,<data>`, each `N` or `NxN` with N in 4, 5, 6, 8.
    pub fn parse(s: &str) -> Option<Self> {
        let edge = |t: &str| -> Option<u32> {
            let t = t.trim().to_ascii_lowercase();
            let n: u32 = match t.split_once('x') {
                Some((a, b)) if a == b => a.parse().ok()?,
                Some(_) => return None,
                None => t.parse().ok()?,
            };
            TexFormat::ASTC_BLOCKS.contains(&n).then_some(n)
        };
        let (color, data) = s.split_once(',')?;
        Some(Self { color: edge(color)?, data: edge(data)? })
    }

    /// Block edge for a slot loaded with `srgb`.
    pub fn for_slot(self, srgb: bool) -> u32 {
        if srgb { self.color } else { self.data }
    }

    fn is_valid(self) -> bool {
        TexFormat::ASTC_BLOCKS.contains(&self.color) && TexFormat::ASTC_BLOCKS.contains(&self.data)
    }
}

/// `color << 8 | data`, 0 = default.
static ASTC_BLOCKS: AtomicU32 = AtomicU32::new(0);

/// Set the ASTC footprints used by later encodes (see [`AstcBlocks`]).
pub fn set_astc_blocks(blocks: AstcBlocks) -> Result<()> {
    anyhow::ensure!(blocks.is_valid(), "ASTC blocks must be 4, 5, 6 or 8 (got {blocks:?})");
    ASTC_BLOCKS.store(blocks.color << 8 | blocks.data, Ordering::Relaxed);
    Ok(())
}

/// The ASTC footprints in effect: `ZESTENGINE_ASTC_BLOCKS`, else the last
/// [`set_astc_blocks`], else the default.
pub fn astc_blocks() -> AstcBlocks {
    static ENV: OnceLock<Option<AstcBlocks>> = OnceLock::new();
    let env = ENV.get_or_init(|| {
        let v = std::env::var("ZESTENGINE_ASTC_BLOCKS").ok()?;
        let parsed = AstcBlocks::parse(&v);
        if parsed.is_none() {
            tracing::warn!("ZESTENGINE_ASTC_BLOCKS={v:?} ignored (expected e.g. 6x6,4x4)");
        }
        parsed
    });
    if let Some(b) = env {
        return *b;
    }
    match ASTC_BLOCKS.load(Ordering::Relaxed) {
        0 => AstcBlocks::default(),
        v => AstcBlocks { color: v >> 8, data: v & 0xff },
    }
}

/// True when this device has no BC but samples ASTC LDR (the `.castc` tier).
fn astc_tier() -> bool {
    !zestengine_render::bc_supported() && zestengine_render::astc_supported()
}

/// Whether the active device can sample `format`.
fn device_samples(format: TexFormat) -> bool {
    if format.is_bc() {
        zestengine_render::bc_supported()
    } else if format.is_astc() {
        zestengine_render::astc_supported()
    } else {
        true
    }
}

/// (mtime secs, byte length) of a source file, zeros when unreadable.
fn source_stamp(path: &Path) -> (u64, u64) {
    let meta = std::fs::metadata(path).ok();
    let mtime = meta
        .as_ref()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    (mtime, meta.as_ref().map(|m| m.len()).unwrap_or(0))
}

/// The tier this device encodes for: ASTC on BC-less ASTC GPUs (Quest /
/// Android), BC (or raw) otherwise.
fn device_platform() -> Platform {
    if astc_tier() { Platform::Quest } else { Platform::Desktop }
}

/// Project ASTC footprint for `import`'s colour space.
fn astc_default_block(import: &TextureImport) -> u32 {
    astc_blocks().for_slot(import.effective_srgb())
}

/// Load a texture as a GPU-ready compressed (or raw-fallback) mip chain, using
/// the on-disk cache when fresh and encoding + caching otherwise. The import
/// settings come from the texture's `.meta` when it holds a configured
/// texture importer; otherwise `srgb` (the colour space of the referencing
/// material slot) selects the colour space for LDR sources (ignored for
/// HDR/EXR, which is always linear) and the legacy encoding applies. Never
/// fails the load over a cache I/O error: it just re-encodes.
pub fn load_texture_bc(path: impl AsRef<Path>, srgb: bool) -> Result<CompressedTexture> {
    let path = path.as_ref();
    load_texture_import(path, &resolve_import(path, srgb))
}

/// [`load_texture_bc`] with explicit import settings.
pub fn load_texture_import(path: &Path, import: &TextureImport) -> Result<CompressedTexture> {
    let (src_mtime, src_len) = source_stamp(path);
    let platform = device_platform();
    let key = import.cache_key(platform, astc_default_block(import));
    let cache = cache_file(path, &key, platform == Platform::Quest);
    if let Some(cache) = &cache
        && let Some(tex) = read_cache(cache, src_mtime, src_len)
        && device_samples(tex.format)
    {
        return Ok(tex);
    }

    let data = load_texture_file(path, import.effective_srgb())?;
    let tex = encode_data(&data, import, platform);
    if let Some(cache) = &cache
        && let Err(e) = write_cache(cache, &tex, src_mtime, src_len)
    {
        tracing::debug!("texture cache write {}: {e:#}", cache.display());
    }
    Ok(tex)
}

/// Compress texture `index` embedded in the model at `model` (BC on desktop,
/// ASTC on mobile, like [`load_texture_bc`]), cached beside the model and
/// re-encoded when the model file changes. Without a model file on disk (a
/// model built in memory) it's encoded every time.
pub fn compress_embedded_texture(model: Option<&Path>, index: usize, data: &zestengine_render::TextureData) -> CompressedTexture {
    let platform = device_platform();
    let import = TextureImport::legacy(data.srgb);
    let encode = || encode_data(data, &import, platform);
    let Some(model) = model.filter(|m| m.is_file()) else { return encode() };
    let (src_mtime, src_len) = source_stamp(model);
    let key_path = model.with_file_name(format!("{}#{index}", model.file_name().map(|n| n.to_string_lossy()).unwrap_or_default()));
    let key = import.cache_key(platform, astc_default_block(&import));
    let cache = cache_file(&key_path, &key, platform == Platform::Quest);
    if let Some(cache) = &cache
        && let Some(tex) = read_cache(cache, src_mtime, src_len)
        && device_samples(tex.format)
    {
        return tex;
    }
    let tex = encode();
    if let Some(cache) = &cache
        && let Err(e) = write_cache(cache, &tex, src_mtime, src_len)
    {
        tracing::debug!("embedded texture cache write {}: {e:#}", cache.display());
    }
    tex
}

/// Where the import cache of `path` lives for the BC / raw tier (`astc` false)
/// or the ASTC tier, under the settings its `.meta` (or the slot colour space
/// `srgb`) resolve to.
pub fn import_cache_path(path: &Path, srgb: bool, astc: bool) -> Option<PathBuf> {
    let import = resolve_import(path, srgb);
    let platform = if astc { Platform::Quest } else { Platform::Desktop };
    cache_file(path, &import.cache_key(platform, astc_default_block(&import)), astc)
}

/// `<dir>/.zestengine_texcache/<filename>.<key>.cbc` (BC / raw tier) or
/// `.castc` (ASTC tier). `key` is [`TextureImport::cache_key`].
fn cache_file(src: &Path, key: &str, astc: bool) -> Option<PathBuf> {
    let dir = src.parent()?.join(".zestengine_texcache");
    let name = src.file_name()?.to_string_lossy();
    let file = dir.join(format!("{name}.{key}.cbc"));
    Some(if astc { file.with_extension(ASTC_CACHE_EXTENSION) } else { file })
}

/// `<dir>/.zestengine_texcache/<filename>.<srgb>.mean`.
fn mean_path(src: &Path, srgb: bool) -> Option<PathBuf> {
    let dir = src.parent()?.join(".zestengine_texcache");
    let name = src.file_name()?.to_string_lossy();
    Some(dir.join(format!("{name}.{}.mean", srgb as u8)))
}

/// Encode decoded pixels per `import` for `platform`. LDR goes through
/// [`process_ldr`] and the format [`TextureImport::select_format`] picks (the
/// device can't sample BC: raw). HDR (BC6H) still needs multiple-of-4 sizes
/// and otherwise stays one raw mip (still cached so the decode isn't repeated).
fn encode_data(data: &zestengine_render::TextureData, import: &TextureImport, platform: Platform) -> CompressedTexture {
    if data.hdr {
        return encode_hdr(data, import, platform);
    }
    if platform == Platform::Desktop && !zestengine_render::bc_supported() {
        // No BC, no ASTC: one raw level, as the original fallback did.
        return raw(
            data,
            if import.effective_srgb() { TexFormat::RgbaSrgb } else { TexFormat::RgbaUnorm },
        );
    }
    encode_processed(process_ldr(data, import, platform))
}

fn encode_hdr(data: &zestengine_render::TextureData, import: &TextureImport, platform: Platform) -> CompressedTexture {
    let info = SourceInfo { width: data.width, height: data.height, has_alpha: false, hdr: true };
    let target = import.select_format(platform, &info, 4);
    match platform {
        Platform::Desktop => {
            let mult4 = data.width.is_multiple_of(4) && data.height.is_multiple_of(4) && data.width >= 4 && data.height >= 4;
            if target == TargetFormat::Bc6h && mult4 && zestengine_render::bc_supported() {
                return encode_bc6h(data);
            }
            raw(data, TexFormat::RgbaF16)
        }
        Platform::Quest => {
            if target != TargetFormat::RgbaF16
                && zestengine_render::astc_hdr_supported()
                && let Some(level) = encode_astc_hdr_level(&data.pixels, data.width, data.height)
            {
                return CompressedTexture { width: data.width, height: data.height, format: TexFormat::Astc4x4Hdr, mips: vec![level] };
            }
            raw(data, TexFormat::RgbaF16)
        }
    }
}

/// An LDR texture after every pixel step, before block compression.
struct Processed {
    /// Mip levels (largest first). For BC targets the base is padded to 4.
    levels: Vec<(Vec<u8>, u32, u32)>,
    srgb: bool,
    target: TargetFormat,
    /// The processed image still has non-opaque texels (ASTC picks its profile on it).
    alpha: bool,
    /// Compression level for the encoder presets.
    quality: Compression,
    /// What the source contained, before any step.
    source: SourceInfo,
}

/// Run the import pixel pipeline: alpha source, alpha dilation, NPOT / max
/// size resize (linear space), height to normal, flip green, channel select,
/// then the mip chain. Textures nobody configured skip straight to a box mip
/// chain, which is exactly the original behaviour.
fn process_ldr(data: &zestengine_render::TextureData, import: &TextureImport, platform: Platform) -> Processed {
    let srgb = import.effective_srgb();
    let source = SourceInfo { width: data.width, height: data.height, has_alpha: pipeline::has_alpha(&data.pixels), hdr: false };
    let mut px = data.pixels.clone();
    let (mut w, mut h) = (data.width, data.height);
    let configured = import.configured;
    if configured {
        match import.alpha_source {
            AlphaSource::None => pipeline::drop_alpha(&mut px),
            AlphaSource::FromGray => pipeline::gray_to_alpha(&mut px),
            AlphaSource::FromInput => {}
        }
        if import.alpha_is_transparency && import.uses_alpha(&source) {
            pipeline::alpha_dilate(&mut px, w, h);
        }
        let (tw, th) = import.target_size(w, h, platform);
        if (tw, th) != (w, h) {
            px = pipeline::resize(&px, w, h, tw, th, srgb, import.platform_settings(platform).resize);
            (w, h) = (tw, th);
        }
        if import.kind == TextureKind::NormalMap {
            if import.normal.from_height {
                pipeline::height_to_normal(&mut px, w, h, import.normal.bumpiness, import.normal.filter);
            }
            if import.normal.flip_green {
                pipeline::flip_green(&mut px);
            }
        }
        if import.kind == TextureKind::SingleChannel {
            pipeline::select_channel(&mut px, import.single_channel);
        }
    }
    let target = import.select_format(platform, &source, astc_default_block(import));
    if matches!(target, TargetFormat::Bc1 | TargetFormat::Bc4 | TargetFormat::Bc5 | TargetFormat::Bc7) {
        // Whole 4x4 blocks first so the texture (and every mip) stays BC.
        (px, w, h) = pad_rgba8(&px, w, h, 4);
    }
    let alpha = import.uses_alpha(&source) && pipeline::has_alpha(&px);
    let params = MipParams {
        srgb,
        filter: if configured { import.mips.filter } else { MipFilter::Box },
        normal: configured && import.kind == TextureKind::NormalMap,
        coverage_cutoff: (configured && import.mips.preserve_coverage && alpha).then_some(import.mips.cutoff),
        enabled: !configured || import.mips.enabled,
    };
    Processed {
        levels: pipeline::mip_chain(&px, w, h, &params),
        srgb,
        target,
        alpha,
        quality: if configured { import.platform_settings(platform).compression } else { Compression::Normal },
        source,
    }
}

/// The GPU format a processed texture is stored as.
fn tex_format(target: TargetFormat, srgb: bool) -> TexFormat {
    match target {
        TargetFormat::Rgba8 => if srgb { TexFormat::RgbaSrgb } else { TexFormat::RgbaUnorm },
        TargetFormat::RgbaF16 => TexFormat::RgbaF16,
        TargetFormat::Bc1 => if srgb { TexFormat::Bc1Srgb } else { TexFormat::Bc1Unorm },
        TargetFormat::Bc4 => TexFormat::Bc4Unorm,
        TargetFormat::Bc5 => TexFormat::Bc5Unorm,
        TargetFormat::Bc7 => if srgb { TexFormat::Bc7Srgb } else { TexFormat::Bc7Unorm },
        TargetFormat::Bc6h => TexFormat::Bc6h,
        TargetFormat::Astc(b) => TexFormat::astc(b, srgb).unwrap_or(if srgb { TexFormat::Astc6x6Srgb } else { TexFormat::Astc4x4Unorm }),
    }
}

fn encode_processed(p: Processed) -> CompressedTexture {
    let format = tex_format(p.target, p.srgb);
    let (width, height) = (p.levels[0].1, p.levels[0].2);
    let mips: Vec<Vec<u8>> = match p.target {
        TargetFormat::Rgba8 | TargetFormat::RgbaF16 | TargetFormat::Bc6h => p.levels.into_iter().map(|(px, ..)| px).collect(),
        TargetFormat::Bc7 => {
            // Opaque mode is much faster and most maps have no real alpha.
            // Ultra-fast is plenty for a one-time encode; High quality
            // spends more on the encoder.
            let high = p.quality == Compression::High;
            let settings = match (p.alpha, high) {
                (true, false) => intel_tex_2::bc7::alpha_ultra_fast_settings(),
                (false, false) => intel_tex_2::bc7::opaque_ultra_fast_settings(),
                (true, true) => intel_tex_2::bc7::alpha_basic_settings(),
                (false, true) => intel_tex_2::bc7::opaque_basic_settings(),
            };
            p.levels.iter().map(|(px, w, h)| encode_bc_level(px, *w, *h, |s| intel_tex_2::bc7::compress_blocks(&settings, s))).collect()
        }
        TargetFormat::Bc1 => p.levels.iter().map(|(px, w, h)| encode_bc_level(px, *w, *h, intel_tex_2::bc1::compress_blocks)).collect(),
        TargetFormat::Bc4 => p
            .levels
            .iter()
            .map(|(px, w, h)| {
                let (px, w, h) = pad_rgba8(px, *w, *h, 4);
                let plane: Vec<u8> = px.as_chunks::<4>().0.iter().map(|t| t[0]).collect();
                intel_tex_2::bc4::compress_blocks(&intel_tex_2::RSurface { data: &plane, width: w, height: h, stride: w })
            })
            .collect(),
        TargetFormat::Bc5 => p
            .levels
            .iter()
            .map(|(px, w, h)| {
                let (px, w, h) = pad_rgba8(px, *w, *h, 4);
                let plane: Vec<u8> = px.as_chunks::<4>().0.iter().flat_map(|t| [t[0], t[1]]).collect();
                intel_tex_2::bc5::compress_blocks(&intel_tex_2::RgSurface { data: &plane, width: w, height: h, stride: w * 2 })
            })
            .collect(),
        TargetFormat::Astc(block) => {
            let _ = block;
            let block = format.block_dims().0;
            p.levels.iter().map(|(px, w, h)| encode_astc_level(px, *w, *h, block, p.alpha)).collect()
        }
    };
    CompressedTexture { format, width, height, mips }
}

/// Pad a level to whole 4x4 blocks and run a BC encoder over it.
fn encode_bc_level(px: &[u8], w: u32, h: u32, encode: impl Fn(&intel_tex_2::RgbaSurface) -> Vec<u8>) -> Vec<u8> {
    let (px, w, h) = pad_rgba8(px, w, h, 4);
    encode(&intel_tex_2::RgbaSurface { data: &px, width: w, height: h, stride: w * 4 })
}

/// A single uncompressed mip (the fallback for odd dimensions).
fn raw(data: &zestengine_render::TextureData, format: TexFormat) -> CompressedTexture {
    CompressedTexture {
        format,
        width: data.width,
        height: data.height,
        mips: vec![data.pixels.clone()],
    }
}

fn encode_bc6h(data: &zestengine_render::TextureData) -> CompressedTexture {
    let settings = intel_tex_2::bc6h::very_fast_settings();
    let mips = mip_chain_rgba_f16(&data.pixels, data.width, data.height)
        .into_iter()
        .map(|(px, w, h)| {
            let surface = intel_tex_2::RgbaSurface {
                data: &px,
                width: w,
                height: h,
                stride: w * 8,
            };
            intel_tex_2::bc6h::compress_blocks(&settings, &surface)
        })
        .collect();
    CompressedTexture {
        format: TexFormat::Bc6h,
        width: data.width,
        height: data.height,
        mips,
    }
}

#[cfg(test)]
/// Legacy ASTC encode of a decoded texture (project footprint `blocks`), a
/// full mip chain down to 1x1. HDR sources become a single ASTC 4x4 HDR mip
/// where the device samples it, raw RGBA16F otherwise.
fn encode_astc(data: &zestengine_render::TextureData, blocks: AstcBlocks) -> CompressedTexture {
    let import = TextureImport::legacy(data.srgb);
    if data.hdr {
        return encode_hdr(data, &import, Platform::Quest);
    }
    let mut p = process_ldr(data, &import, Platform::Quest);
    p.target = TargetFormat::Astc(blocks.for_slot(data.srgb));
    encode_processed(p)
}

unsafe extern "C" {
    /// third_party/zestengine_astc.cpp over the vendored ARM astcenc.
    fn zestengine_astc_ldr_encode(rgba: *const u8, w: u32, h: u32, bx: u32, by: u32, alpha: i32, out: *mut u8, out_len: usize) -> i32;
    fn zestengine_astc_hdr_encode(rgba_f16: *const u16, w: u32, h: u32, bx: u32, by: u32, out: *mut u8, out_len: usize) -> i32;
    fn zestengine_astc_hdr_decode(blocks: *const u8, len: usize, w: u32, h: u32, bx: u32, by: u32, rgba_f16: *mut u16) -> i32;
}

/// RGBA16F texels (8 bytes each) of one ASTC 4x4 HDR level. `None` if the
/// data isn't a whole level or the decoder fails.
pub fn decode_astc_hdr_level(blocks: &[u8], w: u32, h: u32) -> Option<Vec<u8>> {
    if blocks.len() != TexFormat::Astc4x4Hdr.level_bytes(w, h) {
        return None;
    }
    let mut halves = vec![0u16; (w * h * 4) as usize];
    // SAFETY: `blocks` holds exactly the level's blocks and `halves` w*h RGBA halves.
    let status = unsafe { zestengine_astc_hdr_decode(blocks.as_ptr(), blocks.len(), w, h, 4, 4, halves.as_mut_ptr()) };
    (status == 0).then(|| halves.iter().flat_map(|v| v.to_le_bytes()).collect())
}

/// ASTC 4x4 HDR blocks for one level of RGBA16F texels (`rgba_f16` bytes,
/// 8 per texel). `None` if the encoder fails.
pub fn encode_astc_hdr_level(rgba_f16: &[u8], w: u32, h: u32) -> Option<Vec<u8>> {
    let texels: Vec<u16> = rgba_f16.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect();
    if texels.len() != (w * h * 4) as usize {
        return None;
    }
    let mut out = vec![0u8; TexFormat::Astc4x4Hdr.level_bytes(w, h)];
    // SAFETY: `texels` holds w*h RGBA halves and `out` exactly the level's blocks.
    let status = unsafe { zestengine_astc_hdr_encode(texels.as_ptr(), w, h, 4, 4, out.as_mut_ptr(), out.len()) };
    (status == 0).then_some(out)
}

/// ASTC LDR-encode one RGBA8 level of any size with square `block` edges (4,
/// 5, 6 or 8). The level is padded to whole blocks by edge replication;
/// output is `ceil(w/b) * ceil(h/b)` 16-byte blocks, row-major, as Vulkan
/// expects. `alpha` selects the RGBA profile, otherwise alpha decodes as 1.
pub fn encode_astc_level(rgba: &[u8], w: u32, h: u32, block: u32, alpha: bool) -> Vec<u8> {
    assert!(TexFormat::ASTC_BLOCKS.contains(&block), "unsupported ASTC block {block}");
    let (padded, pw, ph) = pad_rgba8(rgba, w, h, block);
    let mut out = vec![0u8; ((pw / block) * (ph / block) * 16) as usize];
    // SAFETY: `padded` holds pw*ph RGBA8 texels (only read) and `out` exactly
    // one 16-byte block per footprint.
    let status =
        unsafe { zestengine_astc_ldr_encode(padded.as_ptr(), pw, ph, block, block, alpha as i32, out.as_mut_ptr(), out.len()) };
    assert_eq!(status, 0, "astcenc rejected a {block}x{block} ASTC LDR encode");
    out
}

/// Copy an RGBA8 level into a buffer padded up to whole `block`-sized blocks
/// by replicating the last row / column. Returns (pixels, width, height).
fn pad_rgba8(src: &[u8], w: u32, h: u32, block: u32) -> (Vec<u8>, u32, u32) {
    let (w, h) = (w.max(1), h.max(1));
    let (pw, ph) = (w.div_ceil(block) * block, h.div_ceil(block) * block);
    if (pw, ph) == (w, h) && src.len() == (w * h * 4) as usize {
        return (src.to_vec(), w, h);
    }
    let mut out = vec![0u8; (pw * ph * 4) as usize];
    for y in 0..ph {
        let sy = y.min(h - 1);
        for x in 0..pw {
            let si = ((sy * w + x.min(w - 1)) * 4) as usize;
            let di = ((y * pw + x) * 4) as usize;
            if let Some(px) = src.get(si..si + 4) {
                out[di..di + 4].copy_from_slice(px);
            }
        }
    }
    (out, pw, ph)
}

/// Box-filtered mip chain (largest first) of RGBA8 pixels down to 1x1; with
/// `srgb` the colour channels are filtered in linear light.
#[cfg(test)]
fn full_mip_chain_rgba8(pixels: &[u8], width: u32, height: u32, srgb: bool) -> Vec<(Vec<u8>, u32, u32)> {
    pipeline::mip_chain(pixels, width, height, &MipParams { srgb, ..MipParams::default() })
}

fn mip_chain_rgba_f16(pixels: &[u8], width: u32, height: u32) -> Vec<(Vec<u8>, u32, u32)> {
    let mut out = vec![(pixels.to_vec(), width, height)];
    let (mut w, mut h) = (width, height);
    while w >= 8 && h >= 8 && w % 2 == 0 && h % 2 == 0 {
        let prev = &out.last().unwrap().0;
        let down = downsample_rgba_f16(prev, w, h);
        w /= 2;
        h /= 2;
        out.push((down, w, h));
    }
    out
}

fn downsample_rgba_f16(src: &[u8], w: u32, h: u32) -> Vec<u8> {
    let (nw, nh) = (w / 2, h / 2);
    let mut out = vec![0u8; (nw * nh * 8) as usize];
    let read = |i: usize| half::f16::from_le_bytes([src[i], src[i + 1]]).to_f32();
    for y in 0..nh {
        for x in 0..nw {
            for c in 0..4 {
                let mut sum = 0f32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let sx = x * 2 + dx;
                        let sy = y * 2 + dy;
                        sum += read((((sy * w + sx) * 4 + c) * 2) as usize);
                    }
                }
                let bytes = half::f16::from_f32(sum / 4.0).to_le_bytes();
                let o = (((y * nw + x) * 4 + c) * 2) as usize;
                out[o] = bytes[0];
                out[o + 1] = bytes[1];
            }
        }
    }
    out
}

fn format_byte(f: TexFormat) -> u8 {
    match f {
        TexFormat::Bc7Srgb => 0,
        TexFormat::Bc7Unorm => 1,
        TexFormat::Bc6h => 2,
        TexFormat::RgbaSrgb => 3,
        TexFormat::RgbaUnorm => 4,
        TexFormat::RgbaF16 => 5,
        TexFormat::Astc4x4Srgb => 6,
        TexFormat::Astc4x4Unorm => 7,
        TexFormat::Astc5x5Srgb => 8,
        TexFormat::Astc5x5Unorm => 9,
        TexFormat::Astc6x6Srgb => 10,
        TexFormat::Astc6x6Unorm => 11,
        TexFormat::Astc8x8Srgb => 12,
        TexFormat::Astc8x8Unorm => 13,
        TexFormat::Astc4x4Hdr => 14,
        TexFormat::Bc1Srgb => 15,
        TexFormat::Bc1Unorm => 16,
        TexFormat::Bc4Unorm => 17,
        TexFormat::Bc5Unorm => 18,
    }
}

fn byte_format(b: u8) -> Option<TexFormat> {
    Some(match b {
        0 => TexFormat::Bc7Srgb,
        1 => TexFormat::Bc7Unorm,
        2 => TexFormat::Bc6h,
        3 => TexFormat::RgbaSrgb,
        4 => TexFormat::RgbaUnorm,
        5 => TexFormat::RgbaF16,
        6 => TexFormat::Astc4x4Srgb,
        7 => TexFormat::Astc4x4Unorm,
        8 => TexFormat::Astc5x5Srgb,
        9 => TexFormat::Astc5x5Unorm,
        10 => TexFormat::Astc6x6Srgb,
        11 => TexFormat::Astc6x6Unorm,
        12 => TexFormat::Astc8x8Srgb,
        13 => TexFormat::Astc8x8Unorm,
        14 => TexFormat::Astc4x4Hdr,
        15 => TexFormat::Bc1Srgb,
        16 => TexFormat::Bc1Unorm,
        17 => TexFormat::Bc4Unorm,
        18 => TexFormat::Bc5Unorm,
        _ => return None,
    })
}

fn write_cache(
    path: &Path,
    tex: &CompressedTexture,
    src_mtime: u64,
    src_len: u64,
) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let buf = cache_bytes(tex, src_mtime, src_len);
    // Write to a temp then rename so a crash mid-write can't leave a torn cache.
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, &buf).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// The cache file layout: magic, format byte, source mtime + size, width,
/// height, mip count, then per mip (byte length, bytes).
fn cache_bytes(tex: &CompressedTexture, src_mtime: u64, src_len: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(37 + tex.mips.iter().map(|m| m.len() + 4).sum::<usize>());
    buf.extend_from_slice(MAGIC);
    buf.push(format_byte(tex.format));
    buf.extend_from_slice(&src_mtime.to_le_bytes());
    buf.extend_from_slice(&src_len.to_le_bytes());
    buf.extend_from_slice(&tex.width.to_le_bytes());
    buf.extend_from_slice(&tex.height.to_le_bytes());
    buf.extend_from_slice(&(tex.mips.len() as u32).to_le_bytes());
    for mip in &tex.mips {
        buf.extend_from_slice(&(mip.len() as u32).to_le_bytes());
        buf.extend_from_slice(mip);
    }
    buf
}

/// Read + validate a cache file; None on any mismatch (stale, corrupt, version).
fn read_cache(path: &Path, src_mtime: u64, src_len: u64) -> Option<CompressedTexture> {
    parse_cache(&std::fs::read(path).ok()?, src_mtime, src_len)
}

/// [`read_cache`] over bytes already in memory.
fn parse_cache(bytes: &[u8], src_mtime: u64, src_len: u64) -> Option<CompressedTexture> {
    let mut p = 0usize;
    let take = |p: &mut usize, n: usize| -> Option<&[u8]> {
        let s = bytes.get(*p..*p + n)?;
        *p += n;
        Some(s)
    };
    if take(&mut p, 8)? != MAGIC {
        return None;
    }
    let format = byte_format(take(&mut p, 1)?[0])?;
    let u64_at = |p: &mut usize| -> Option<u64> {
        Some(u64::from_le_bytes(take(p, 8)?.try_into().ok()?))
    };
    let u32_at = |p: &mut usize| -> Option<u32> {
        Some(u32::from_le_bytes(take(p, 4)?.try_into().ok()?))
    };
    if u64_at(&mut p)? != src_mtime || u64_at(&mut p)? != src_len {
        return None;
    }
    let width = u32_at(&mut p)?;
    let height = u32_at(&mut p)?;
    let mip_count = u32_at(&mut p)? as usize;
    let mut mips = Vec::with_capacity(mip_count);
    for _ in 0..mip_count {
        let len = u32_at(&mut p)? as usize;
        mips.push(take(&mut p, len)?.to_vec());
    }
    Some(CompressedTexture {
        format,
        width,
        height,
        mips,
    })
}

/// Encode decoded RGBA8 pixels the way a texture with `import` is cooked for
/// `platform` (pixel steps, mips, BC / ASTC). Used for atlas pages (2D-4).
pub fn encode_rgba8(width: u32, height: u32, pixels: Vec<u8>, import: &TextureImport, platform: Platform) -> CompressedTexture {
    let data = zestengine_render::TextureData { width, height, pixels, srgb: import.effective_srgb(), hdr: false };
    encode_data(&data, import, platform)
}

/// Cache key of `import` for `platform` including the project ASTC footprint.
pub fn import_key(import: &TextureImport, platform: Platform) -> String {
    import.cache_key(platform, astc_default_block(import))
}

/// A compressed texture as bytes (the cache file layout, no source stamp).
pub fn compressed_to_bytes(tex: &CompressedTexture) -> Vec<u8> {
    cache_bytes(tex, 0, 0)
}

/// Inverse of [`compressed_to_bytes`]; `None` for a corrupt or old blob.
pub fn compressed_from_bytes(bytes: &[u8]) -> Option<CompressedTexture> {
    parse_cache(bytes, 0, 0)
}

const MEAN_MAGIC: &[u8; 8] = b"CITRMN01";

/// Mean linear RGB of a texture (sRGB sources decoded to linear; HDR and
/// unreadable files give `[1,1,1]`). Cached as `<filename>.<srgb>.mean` beside
/// the BC cache, so only the first load ever decodes the source for it.
pub fn texture_mean(path: impl AsRef<Path>, srgb: bool) -> [f32; 3] {
    let path = path.as_ref();
    let srgb = resolve_import(path, srgb).effective_srgb();
    let (src_mtime, src_len) = source_stamp(path);
    let cache = mean_path(path, srgb);
    if let Some(cache) = &cache
        && let Some(mean) = read_mean(cache, src_mtime, src_len)
    {
        return mean;
    }
    let mean = match load_texture_file(path, srgb) {
        Ok(data) => texture_mean_linear(&data),
        Err(_) => return [1.0; 3],
    };
    if let Some(cache) = &cache {
        let mut buf = Vec::with_capacity(36);
        buf.extend_from_slice(MEAN_MAGIC);
        buf.extend_from_slice(&src_mtime.to_le_bytes());
        buf.extend_from_slice(&src_len.to_le_bytes());
        for c in mean {
            buf.extend_from_slice(&c.to_le_bytes());
        }
        let written = cache
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| std::fs::write(cache, &buf));
        if let Err(e) = written {
            tracing::debug!("texture mean cache write {}: {e:#}", cache.display());
        }
    }
    mean
}

fn read_mean(path: &Path, src_mtime: u64, src_len: u64) -> Option<[f32; 3]> {
    let bytes = std::fs::read(path).ok()?;
    if bytes.len() != 36 || &bytes[..8] != MEAN_MAGIC {
        return None;
    }
    let u64_at = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
    if u64_at(8) != src_mtime || u64_at(16) != src_len {
        return None;
    }
    let f = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    Some([f(24), f(28), f(32)])
}

/// Mean RGB in linear space. HDR maps return `[1,1,1]`: their values exceed 1
/// and aren't a simple coverage factor.
pub fn texture_mean_linear(data: &zestengine_render::TextureData) -> [f32; 3] {
    if data.hdr || data.pixels.len() < 4 {
        return [1.0; 3];
    }
    let mut acc = [0.0f64; 3];
    let mut n = 0u64;
    for px in data.pixels.as_chunks::<4>().0 {
        for c in 0..3 {
            let s = px[c] as f32 / 255.0;
            acc[c] += if data.srgb { srgb_to_linear(s) } else { s } as f64;
        }
        n += 1;
    }
    if n == 0 {
        return [1.0; 3];
    }
    [
        (acc[0] / n as f64) as f32,
        (acc[1] / n as f64) as f32,
        (acc[2] / n as f64) as f32,
    ]
}

/// GPU bytes of a prepared texture, all mips.
pub fn texture_vram_bytes(tex: &CompressedTexture) -> u64 {
    tex.mips.iter().map(|m| m.len() as u64).sum()
}

/// What importing a texture with some settings would produce, without
/// block-compressing it: the processed mip chain (for the inspector preview)
/// and the format / VRAM it would take on `platform`. Cheap enough to redo
/// for an unapplied draft.
pub struct TexturePreview {
    pub source: SourceInfo,
    pub target: TargetFormat,
    pub format: TexFormat,
    /// Stored base size (after resize, and padding for BC).
    pub width: u32,
    pub height: u32,
    /// Processed RGBA8 levels, largest first. Empty for HDR sources.
    pub levels: Vec<(Vec<u8>, u32, u32)>,
    pub mip_count: usize,
    /// GPU bytes of the texture including mips.
    pub vram_bytes: u64,
    pub srgb: bool,
    /// The processed image keeps non-opaque texels.
    pub has_alpha: bool,
}

/// Run the import pipeline for `path` under `import` (a draft is fine) and
/// report what it would become on `platform`.
pub fn preview_texture(path: &Path, import: &TextureImport, platform: Platform) -> Result<TexturePreview> {
    let data = load_texture_file(path, import.effective_srgb())?;
    if data.hdr {
        let source = SourceInfo { width: data.width, height: data.height, has_alpha: false, hdr: true };
        let target = import.select_format(platform, &source, 4);
        let format = match target {
            TargetFormat::Bc6h => TexFormat::Bc6h,
            _ => TexFormat::RgbaF16,
        };
        // The HDR encoders keep a (BC6H: halving while both sides stay >= 8 and even) chain, or one raw level.
        let (mut w, mut h, mut vram, mut mips) = (data.width, data.height, format.level_bytes(data.width, data.height) as u64, 1);
        if format == TexFormat::Bc6h {
            while w >= 8 && h >= 8 && w % 2 == 0 && h % 2 == 0 {
                (w, h) = (w / 2, h / 2);
                vram += format.level_bytes(w, h) as u64;
                mips += 1;
            }
        }
        return Ok(TexturePreview {
            source,
            target,
            format,
            width: data.width,
            height: data.height,
            levels: Vec::new(),
            mip_count: mips,
            vram_bytes: vram,
            srgb: false,
            has_alpha: false,
        });
    }
    let p = process_ldr(&data, import, platform);
    let format = tex_format(p.target, p.srgb);
    let vram = p.levels.iter().map(|(_, w, h)| format.level_bytes(*w, *h) as u64).sum();
    Ok(TexturePreview {
        source: p.source,
        target: p.target,
        format,
        width: p.levels[0].1,
        height: p.levels[0].2,
        mip_count: p.levels.len(),
        vram_bytes: vram,
        srgb: p.srgb,
        has_alpha: p.alpha,
        levels: p.levels,
    })
}

/// The import-cache blob for a texture (encoding it first when stale), with
/// the source mtime/size fields zeroed so it can travel inside a content
/// package and be re-bound to the extracted file by [`seed_texture_cache`].
pub fn texture_cache_blob(path: impl AsRef<Path>, srgb: bool) -> Result<Vec<u8>> {
    let tex = load_texture_bc(path, srgb)?;
    Ok(cache_bytes(&tex, 0, 0))
}

/// Device-independent ASTC-tier cache blob for a texture (the Android /
/// Quest cook): ASTC LDR at `blocks`' footprint for the slot, raw RGBA16F
/// for HDR sources. Reuses the source's own `.castc` cache when it matches
/// (re-encoding and refreshing it otherwise), and zeroes the source stamp
/// like [`texture_cache_blob`] so [`seed_astc_cache`] can re-bind it.
pub fn astc_cache_blob(path: impl AsRef<Path>, srgb: bool, blocks: AstcBlocks) -> Result<Vec<u8>> {
    let path = path.as_ref();
    let import = resolve_import(path, srgb);
    let (src_mtime, src_len) = source_stamp(path);
    let key = import.cache_key(Platform::Quest, blocks.for_slot(import.effective_srgb()));
    let cache = cache_file(path, &key, true);
    let cached = cache.as_ref().and_then(|c| read_cache(c, src_mtime, src_len)).filter(|t| !t.format.is_bc());
    let tex = match cached {
        Some(t) => t,
        None => {
            let data = load_texture_file(path, import.effective_srgb())?;
            let tex = if data.hdr {
                encode_hdr(&data, &import, Platform::Quest)
            } else {
                // The cook always encodes the Quest plan, whatever this machine's GPU is.
                let mut p = process_ldr(&data, &import, Platform::Quest);
                p.target = import.select_format(Platform::Quest, &p.source, blocks.for_slot(import.effective_srgb()));
                encode_processed(p)
            };
            if let Some(cache) = &cache
                && let Err(e) = write_cache(cache, &tex, src_mtime, src_len)
            {
                tracing::debug!("texture cache write {}: {e:#}", cache.display());
            }
            tex
        }
    };
    Ok(cache_bytes(&tex, 0, 0))
}

/// Install a packaged cache blob for `path` (already on disk) so the next
/// `load_texture_bc` skips the decode + encode. The blob is fully validated
/// (format, dimensions, per-mip byte counts) since it may come from untrusted
/// content. BC blobs are skipped on devices without BC support; ASTC blobs
/// go to the ASTC-tier artefact. Returns the texture's GPU byte size when
/// installed, None when skipped.
pub fn seed_texture_cache(path: impl AsRef<Path>, srgb: bool, blob: &[u8], max_dim: u32) -> Result<Option<u64>> {
    let tex = checked_blob(blob, max_dim)?;
    if tex.format.is_bc() && !zestengine_render::bc_supported() {
        return Ok(None);
    }
    let astc = tex.format.is_astc();
    install_seed(path.as_ref(), srgb, &tex, astc).map(Some)
}

/// Install an ASTC-tier blob from [`astc_cache_blob`] (ASTC LDR or raw) as
/// `path`'s `.castc` cache. Independent of the device (it runs before the
/// renderer exists on Android); a device that can't sample it re-encodes at
/// load. Returns the texture's GPU byte size.
pub fn seed_astc_cache(path: impl AsRef<Path>, srgb: bool, blob: &[u8], max_dim: u32) -> Result<u64> {
    let tex = checked_blob(blob, max_dim)?;
    anyhow::ensure!(!tex.format.is_bc(), "{:?} is not an ASTC-tier format", tex.format);
    install_seed(path.as_ref(), srgb, &tex, true)
}

fn checked_blob(blob: &[u8], max_dim: u32) -> Result<CompressedTexture> {
    if blob.len() < 9 + 16 || &blob[..8] != MAGIC {
        anyhow::bail!("not a texture cache blob");
    }
    let tex = parse_blob(blob).context("malformed texture cache blob")?;
    validate_compressed(&tex, max_dim)?;
    Ok(tex)
}

/// Write `tex` as `path`'s cache, stamped with the file's current mtime / size.
/// The cache name uses the settings the loader will look up (the `.meta`
/// next to `path`, else the legacy plan for `srgb`).
fn install_seed(path: &Path, srgb: bool, tex: &CompressedTexture, astc: bool) -> Result<u64> {
    let meta = std::fs::metadata(path).with_context(|| format!("seeding cache for {}", path.display()))?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let import = resolve_import(path, srgb);
    let platform = if astc { Platform::Quest } else { Platform::Desktop };
    let key = import.cache_key(platform, astc_default_block(&import));
    let cache = cache_file(path, &key, astc).context("texture has no parent directory")?;
    write_cache(&cache, tex, mtime, meta.len())?;
    Ok(tex.mips.iter().map(|m| m.len() as u64).sum())
}

fn parse_blob(bytes: &[u8]) -> Option<CompressedTexture> {
    let mut p = 8usize;
    let take = |p: &mut usize, n: usize| -> Option<&[u8]> {
        let s = bytes.get(*p..p.checked_add(n)?)?;
        *p += n;
        Some(s)
    };
    let format = byte_format(take(&mut p, 1)?[0])?;
    take(&mut p, 16)?;
    let u32_at = |p: &mut usize| -> Option<u32> { Some(u32::from_le_bytes(take(p, 4)?.try_into().ok()?)) };
    let width = u32_at(&mut p)?;
    let height = u32_at(&mut p)?;
    let mip_count = u32_at(&mut p)? as usize;
    if mip_count == 0 || mip_count > 16 {
        return None;
    }
    let mut mips = Vec::with_capacity(mip_count);
    for _ in 0..mip_count {
        let len = u32_at(&mut p)? as usize;
        mips.push(take(&mut p, len)?.to_vec());
    }
    (p == bytes.len()).then_some(CompressedTexture {
        format,
        width,
        height,
        mips,
    })
}

/// Check a compressed texture's mip byte counts match its format and size, so
/// a crafted cache can't make the GPU upload read past a buffer.
pub fn validate_compressed(tex: &CompressedTexture, max_dim: u32) -> Result<()> {
    if tex.width == 0 || tex.height == 0 || tex.width > max_dim || tex.height > max_dim {
        anyhow::bail!("texture size {}x{} outside 1..={max_dim}", tex.width, tex.height);
    }
    let full_chain = 32 - tex.width.max(tex.height).leading_zeros() as usize;
    if tex.mips.is_empty() || tex.mips.len() > full_chain {
        anyhow::bail!("{} mips for a {}x{} texture", tex.mips.len(), tex.width, tex.height);
    }
    for (i, mip) in tex.mips.iter().enumerate() {
        let expected = tex.format.level_bytes((tex.width >> i).max(1), (tex.height >> i).max(1));
        if mip.len() != expected {
            anyhow::bail!("mip {i} holds {} bytes, expected {expected}", mip.len());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_FORMATS: [TexFormat; 19] = [
        TexFormat::Bc7Srgb,
        TexFormat::Bc7Unorm,
        TexFormat::Bc6h,
        TexFormat::RgbaSrgb,
        TexFormat::RgbaUnorm,
        TexFormat::RgbaF16,
        TexFormat::Astc4x4Srgb,
        TexFormat::Astc4x4Unorm,
        TexFormat::Astc5x5Srgb,
        TexFormat::Astc5x5Unorm,
        TexFormat::Astc6x6Srgb,
        TexFormat::Astc6x6Unorm,
        TexFormat::Astc8x8Srgb,
        TexFormat::Astc8x8Unorm,
        TexFormat::Astc4x4Hdr,
        TexFormat::Bc1Srgb,
        TexFormat::Bc1Unorm,
        TexFormat::Bc4Unorm,
        TexFormat::Bc5Unorm,
    ];

    /// Where the loader caches a texture nobody configured.
    fn legacy_cache(src: &Path, srgb: bool, astc: bool) -> PathBuf {
        let import = TextureImport::legacy(srgb);
        let platform = if astc { Platform::Quest } else { Platform::Desktop };
        cache_file(src, &import.cache_key(platform, astc_default_block(&import)), astc).unwrap()
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("zestengine_texcache_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// RGBA8 gradient with a hard checker in blue and an alpha ramp.
    fn gradient(w: u32, h: u32, alpha: bool) -> Vec<u8> {
        let mut px = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                px.extend_from_slice(&[
                    (x * 255 / (w - 1).max(1)) as u8,
                    (y * 255 / (h - 1).max(1)) as u8,
                    if (x / 3 + y / 3) % 2 == 0 { 40 } else { 200 },
                    if alpha { ((x + y) * 255 / (w + h - 2).max(1)) as u8 } else { 255 },
                ]);
            }
        }
        px
    }

    fn ldr(w: u32, h: u32, srgb: bool, alpha: bool) -> zestengine_render::TextureData {
        zestengine_render::TextureData { width: w, height: h, pixels: gradient(w, h, alpha), srgb, hdr: false }
    }

    #[test]
    fn srgb_mips_average_in_linear_space() {
        // 2x2 black/white checker: the 1x1 mip is 50% linear light = sRGB ~188,
        // not the naive byte average 128 (which comes out too dark).
        let px = vec![0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255];
        let chain = full_mip_chain_rgba8(&px, 2, 2, true);
        assert_eq!(chain.len(), 2);
        let (top, w, h) = &chain[1];
        assert_eq!((*w, *h), (1, 1));
        assert!((187..=189).contains(&top[0]), "linear-space average, got {}", top[0]);
        assert_eq!(top[3], 255, "alpha averages directly");
        let lin = full_mip_chain_rgba8(&px, 2, 2, false);
        assert_eq!(lin[1].0[0], 128, "linear data averages bytes");
    }

    #[test]
    fn mip_chain_reaches_one_by_one_for_odd_sizes() {
        let chain = full_mip_chain_rgba8(&gradient(12, 5, false), 12, 5, true);
        let dims: Vec<_> = chain.iter().map(|m| (m.1, m.2)).collect();
        assert_eq!(dims, [(12, 5), (6, 2), (3, 1), (1, 1)]);
    }

    #[test]
    fn non_multiple_of_four_bc7_is_padded_with_a_full_chain() {
        let tex = encode_data(&ldr(30, 18, true, false), &TextureImport::legacy(true), Platform::Desktop);
        assert_eq!(tex.format, TexFormat::Bc7Srgb);
        assert_eq!((tex.width, tex.height), (32, 20));
        assert_eq!(tex.mips.len(), 6, "32x20 down to 1x1");
        validate_compressed(&tex, 64).unwrap();
        assert_eq!(tex.mips[5].len(), 16, "1x1 level is one block");
    }


    #[test]
    fn format_bytes_are_unique_and_round_trip() {
        let mut seen = std::collections::BTreeSet::new();
        for f in ALL_FORMATS {
            assert!(seen.insert(format_byte(f)), "{f:?} reuses a format byte");
            assert_eq!(byte_format(format_byte(f)), Some(f));
        }
        // On-disk values of the original formats never move.
        assert_eq!(ALL_FORMATS[..6].iter().map(|&f| format_byte(f)).collect::<Vec<_>>(), [0, 1, 2, 3, 4, 5]);
        assert_eq!(byte_format(19), None);
        for b in TexFormat::ASTC_BLOCKS {
            for srgb in [false, true] {
                let f = TexFormat::astc(b, srgb).unwrap();
                assert!(f.is_astc() && !f.is_bc() && f.block_dims() == (b, b));
            }
        }
        assert!(TexFormat::astc(7, true).is_none());
        assert!(TexFormat::Bc7Srgb.is_bc() && !TexFormat::Bc7Srgb.is_astc() && !TexFormat::RgbaF16.is_astc());
    }

    #[test]
    fn astc_levels_have_vulkan_block_counts() {
        for block in TexFormat::ASTC_BLOCKS {
            let format = TexFormat::astc(block, false).unwrap();
            for (w, h) in [(1, 1), (3, 5), (13, 7), (64, 32), (30, 30), (block * 3, block * 2)] {
                for alpha in [false, true] {
                    let blocks = encode_astc_level(&gradient(w, h, alpha), w, h, block, alpha);
                    let want = (w.div_ceil(block) * h.div_ceil(block) * 16) as usize;
                    assert_eq!(blocks.len(), want, "{w}x{h} at {block}x{block}");
                    assert_eq!(format.level_bytes(w, h), want);
                    assert!(blocks.iter().any(|&b| b != 0), "encoder wrote nothing");
                }
            }
        }
    }

    /// Decode ASTC LDR blocks with an independent decoder to RGBA8.
    fn decode_astc(blocks: &[u8], w: u32, h: u32, block: u32) -> Vec<u8> {
        let mut argb = vec![0u32; (w * h) as usize];
        texture2ddecoder::decode_astc(blocks, w as usize, h as usize, block as usize, block as usize, &mut argb).unwrap();
        argb.iter().flat_map(|p| [(p >> 16) as u8, (p >> 8) as u8, *p as u8, (p >> 24) as u8]).collect()
    }

    /// Smooth RGBA8 test card: gradients with a soft bump and an alpha ramp.
    fn smooth(w: u32, h: u32, alpha: bool) -> Vec<u8> {
        let mut px = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let (u, v) = (x as f32 / (w - 1).max(1) as f32, y as f32 / (h - 1).max(1) as f32);
                let bump = (-((u - 0.5).powi(2) + (v - 0.5).powi(2)) * 8.0).exp();
                px.extend_from_slice(&[
                    (u * 255.0) as u8,
                    (v * 255.0) as u8,
                    (bump * 220.0 + 20.0) as u8,
                    if alpha { ((1.0 - u) * 255.0) as u8 } else { 255 },
                ]);
            }
        }
        px
    }

    #[test]
    fn astc_decodes_back_to_the_source() {
        // Mean / max absolute 8-bit error bounds per block edge (the fast
        // profiles measure ~1.5x below); a block-order or padding bug is far
        // above.
        for (block, mean_max, worst_max) in [(4, 3.0, 32), (5, 4.0, 40), (6, 5.0, 48), (8, 8.0, 64)] {
            // 61x37 has partial edge blocks at every footprint.
            for (w, h) in [(64, 32), (30, 30), (61, 37)] {
                for alpha in [false, true] {
                    let src = smooth(w, h, alpha);
                    let back = decode_astc(&encode_astc_level(&src, w, h, block, alpha), w, h, block);
                    let diffs: Vec<u32> = src.iter().zip(&back).map(|(a, b)| a.abs_diff(*b) as u32).collect();
                    let mean = diffs.iter().sum::<u32>() as f32 / diffs.len() as f32;
                    let worst = *diffs.iter().max().unwrap();
                    assert!(mean < mean_max && worst < worst_max, "{w}x{h} {block}x{block} alpha {alpha}: mean {mean} max {worst}");
                    if !alpha {
                        assert!(back.as_chunks::<4>().0.iter().all(|p| p[3] == 255), "opaque profile keeps alpha 1");
                    }
                }
            }
        }
    }

    #[test]
    fn solid_colour_decodes_exactly() {
        for block in TexFormat::ASTC_BLOCKS {
            for (w, h) in [(24, 24), (13, 11), (1, 1)] {
                let px = [90u8, 160, 30, 255].repeat((w * h) as usize);
                let back = decode_astc(&encode_astc_level(&px, w, h, block, false), w, h, block);
                for p in back.as_chunks::<4>().0 {
                    assert!(p.iter().zip([90u8, 160, 30, 255]).all(|(a, b)| a.abs_diff(b) <= 1), "{w}x{h} {block}x{block}: {p:?}");
                }
            }
        }
    }

    #[test]
    fn astc_encode_is_thread_safe() {
        let (w, h) = (96, 64);
        let px = gradient(w, h, true);
        let single = encode_astc_level(&px, w, h, 6, true);
        let parallel: Vec<Vec<u8>> = std::thread::scope(|s| {
            let jobs: Vec<_> = (0..6).map(|_| s.spawn(|| encode_astc_level(&px, w, h, 6, true))).collect();
            jobs.into_iter().map(|j| j.join().unwrap()).collect()
        });
        assert!(parallel.iter().all(|p| *p == single));
    }

    #[test]
    fn astc_textures_carry_a_full_mip_chain() {
        let blocks = AstcBlocks::default();
        let color = encode_astc(&ldr(40, 24, true, false), blocks);
        assert_eq!(color.format, TexFormat::Astc6x6Srgb);
        // 40x24 -> 20x12 -> 10x6 -> 5x3 -> 2x1 -> 1x1.
        assert_eq!(color.mips.len(), 6);
        validate_compressed(&color, 64).unwrap();
        assert_eq!(color.mips[5].len(), 16);
        let data = encode_astc(&ldr(7, 9, false, true), blocks);
        assert_eq!(data.format, TexFormat::Astc4x4Unorm);
        assert_eq!(data.mips.len(), 4);
        validate_compressed(&data, 64).unwrap();
        let wide = encode_astc(&ldr(32, 32, true, false), AstcBlocks { color: 8, data: 5 });
        assert_eq!(wide.format, TexFormat::Astc8x8Srgb);
        assert_eq!(wide.mips[0].len(), 16 * 16);
        let hdr = zestengine_render::TextureData { width: 4, height: 4, pixels: vec![0; 128], srgb: false, hdr: true };
        assert_eq!(encode_astc(&hdr, blocks).format, TexFormat::RgbaF16, "no device ASTC HDR support: raw halves");
    }

    #[test]
    fn validation_rejects_bad_mip_tables() {
        let mut tex = encode_astc(&ldr(16, 16, false, false), AstcBlocks::default());
        validate_compressed(&tex, 16).unwrap();
        assert!(validate_compressed(&tex, 8).is_err(), "over max_dim");
        tex.mips.push(vec![0; 16]);
        assert!(validate_compressed(&tex, 16).is_err(), "more mips than the chain has");
        tex.mips.pop();
        tex.mips[1].pop();
        assert!(validate_compressed(&tex, 16).is_err(), "short mip");
    }

    #[test]
    fn astc_blocks_parse() {
        assert_eq!(AstcBlocks::parse("6x6,4x4"), Some(AstcBlocks { color: 6, data: 4 }));
        assert_eq!(AstcBlocks::parse(" 8 , 5X5 "), Some(AstcBlocks { color: 8, data: 5 }));
        for bad in ["6x5,4", "7,4", "6", "", "4,4,4", "12x12,4"] {
            assert_eq!(AstcBlocks::parse(bad), None, "{bad:?}");
        }
        assert!(set_astc_blocks(AstcBlocks { color: 3, data: 4 }).is_err());
        assert_eq!(AstcBlocks::default().for_slot(true), 6);
        assert_eq!(AstcBlocks::default().for_slot(false), 4);
    }

    #[test]
    fn astc_cache_round_trips_and_seeds() {
        let dir = temp_dir("astc");
        let png = dir.join("albedo.png");
        image::RgbaImage::from_fn(20, 12, |x, y| image::Rgba([(x * 12) as u8, (y * 20) as u8, 99, 255])).save(&png).unwrap();
        let blocks = AstcBlocks::default();

        // Cook-side blob: zeroed stamp, refreshes the source's .castc cache.
        let blob = astc_cache_blob(&png, true, blocks).unwrap();
        let castc = legacy_cache(&png, true, true);
        assert!(castc.is_file() && !legacy_cache(&png, true, false).exists());
        assert_eq!(&blob[9..25], &[0u8; 16]);
        let tex = parse_blob(&blob).unwrap();
        assert_eq!((tex.format, tex.width, tex.height, tex.mips.len()), (TexFormat::Astc6x6Srgb, 20, 12, 5));

        // The per-file cache reads back identically under the source stamp.
        let (mtime, len) = source_stamp(&png);
        let cached = read_cache(&castc, mtime, len).unwrap();
        assert_eq!((cached.format, cached.mips.clone()), (tex.format, tex.mips.clone()));
        assert!(read_cache(&castc, mtime, len + 1).is_none(), "stale stamp");
        assert_eq!(astc_cache_blob(&png, true, blocks).unwrap(), blob, "cache hit is stable");

        // A different footprint re-encodes for the cook.
        let small = parse_blob(&astc_cache_blob(&png, true, AstcBlocks { color: 4, data: 4 }).unwrap()).unwrap();
        assert_eq!(small.format, TexFormat::Astc4x4Srgb);

        // Seeding a shipped blob next to a freshly extracted copy re-binds it.
        let shipped = temp_dir("astc_seed");
        let copy = shipped.join("albedo.png");
        std::fs::copy(&png, &copy).unwrap();
        let bytes = seed_astc_cache(&copy, true, &blob, 4096).unwrap();
        assert_eq!(bytes, tex.mips.iter().map(|m| m.len() as u64).sum::<u64>());
        let (mtime, len) = source_stamp(&copy);
        let seeded = read_cache(&legacy_cache(&copy, true, true), mtime, len).unwrap();
        assert_eq!(seeded.mips, tex.mips);
        // seed_texture_cache routes ASTC blobs to the same artefact.
        assert_eq!(seed_texture_cache(&copy, true, &blob, 4096).unwrap(), Some(bytes));
        assert!(!legacy_cache(&copy, true, false).exists());
        // Garbage and BC blobs are refused by the ASTC seeder.
        assert!(seed_astc_cache(&copy, true, &blob[..blob.len() - 1], 4096).is_err());
        assert!(seed_astc_cache(&copy, true, &blob, 8).is_err(), "over max_dim");
        let bc = CompressedTexture { format: TexFormat::Bc7Srgb, width: 4, height: 4, mips: vec![vec![0; 16]] };
        assert!(seed_astc_cache(&copy, true, &cache_bytes(&bc, 0, 0), 4096).is_err());
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&shipped);
    }

    #[test]
    fn unsampleable_cache_is_reencoded_not_returned() {
        // Test devices have BC and no ASTC (the render defaults): an ASTC
        // payload planted in the BC-tier cache must never come back.
        assert!(zestengine_render::bc_supported() && !zestengine_render::astc_supported());
        let dir = temp_dir("foreign");
        let png = dir.join("orm.png");
        image::RgbaImage::from_pixel(8, 8, image::Rgba([10, 20, 30, 255])).save(&png).unwrap();
        let astc = encode_astc(&ldr(8, 8, false, false), AstcBlocks::default());
        let (mtime, len) = source_stamp(&png);
        let cbc = legacy_cache(&png, false, false);
        write_cache(&cbc, &astc, mtime, len).unwrap();
        let tex = load_texture_bc(&png, false).unwrap();
        assert_eq!(tex.format, TexFormat::Bc7Unorm);
        assert_eq!(read_cache(&cbc, mtime, len).unwrap().format, TexFormat::Bc7Unorm, "cache rewritten");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn texture_mean_is_cached_and_invalidated() {
        let dir = std::env::temp_dir().join(format!("zestengine_mean_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let png = dir.join("grey.png");
        image::RgbaImage::from_pixel(4, 4, image::Rgba([0, 128, 255, 255])).save(&png).unwrap();

        let mean = texture_mean(&png, false);
        assert!((mean[0] - 0.0).abs() < 1e-4 && (mean[1] - 128.0 / 255.0).abs() < 1e-4 && (mean[2] - 1.0).abs() < 1e-4);
        let sidecar = mean_path(&png, false).unwrap();
        assert!(sidecar.exists());
        assert_eq!(texture_mean(&png, false), mean);

        // A different source (size changes) must not reuse the stale mean.
        image::RgbaImage::from_pixel(8, 8, image::Rgba([255, 255, 255, 255])).save(&png).unwrap();
        assert_eq!(texture_mean(&png, false), [1.0; 3]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Write `t` as `png`'s `.meta`.
    fn write_import(png: &Path, t: TextureImport) {
        let mut meta = crate::AssetMeta::new_for(png);
        meta.importer = crate::ImporterSettings::Texture(t);
        crate::save_asset_meta(png, &meta).unwrap();
    }

    fn photo(dir: &Path, name: &str, w: u32, h: u32) -> PathBuf {
        let png = dir.join(name);
        image::RgbaImage::from_fn(w, h, |x, y| image::Rgba([(x * 255 / w) as u8, (y * 255 / h) as u8, ((x ^ y) & 0xff) as u8, 255])).save(&png).unwrap();
        png
    }

    #[test]
    fn textures_without_a_configured_meta_load_as_before() {
        let dir = temp_dir("legacy_meta");
        let png = photo(&dir, "plain.png", 32, 32);
        // No meta at all, a Generic meta, and a never-applied Texture meta all
        // mean: colour space from the slot, BC7.
        assert_eq!(load_texture_bc(&png, true).unwrap().format, TexFormat::Bc7Srgb);
        assert_eq!(load_texture_bc(&png, false).unwrap().format, TexFormat::Bc7Unorm);
        let mut meta = crate::AssetMeta::new_for(&png);
        meta.importer = crate::ImporterSettings::Generic;
        crate::save_asset_meta(&png, &meta).unwrap();
        assert_eq!(load_texture_bc(&png, true).unwrap().format, TexFormat::Bc7Srgb);
        let mut t = TextureImport::default();
        t.apply_kind(TextureKind::NormalMap); // edited but never applied
        write_import(&png, t);
        let tex = load_texture_bc(&png, true).unwrap();
        assert_eq!(tex.format, TexFormat::Bc7Srgb, "slot wins until the meta is configured");
        assert_eq!(tex.mips.len(), 6);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn configured_meta_picks_format_size_and_colour_space() {
        let dir = temp_dir("configured");
        let png = photo(&dir, "tex.png", 64, 48);
        let mut base = TextureImport { configured: true, ..TextureImport::default() };

        // Opaque sRGB colour -> BC1 (4 bits per texel), whatever the slot says.
        write_import(&png, base.clone());
        let c = load_texture_bc(&png, false).unwrap();
        assert_eq!((c.format, c.width, c.height), (TexFormat::Bc1Srgb, 64, 48));
        validate_compressed(&c, 64).unwrap();
        assert_eq!(c.mips[0].len(), 64 * 48 / 2);

        // Normal map -> BC5, linear.
        let mut n = base.clone();
        n.apply_kind(TextureKind::NormalMap);
        write_import(&png, n);
        let nt = load_texture_bc(&png, true).unwrap();
        assert_eq!(nt.format, TexFormat::Bc5Unorm);
        validate_compressed(&nt, 64).unwrap();
        assert_eq!(nt.mips[0].len(), 64 * 48);

        // Single channel -> BC4.
        let mut s = base.clone();
        s.apply_kind(TextureKind::SingleChannel);
        s.single_channel = crate::Channel::G;
        write_import(&png, s);
        let st = load_texture_bc(&png, true).unwrap();
        assert_eq!(st.format, TexFormat::Bc4Unorm);
        validate_compressed(&st, 64).unwrap();
        assert_eq!(st.mips[0].len(), 64 * 48 / 2);

        // Max size resizes (linear space) and keeps a full chain.
        base.max_size = 16;
        write_import(&png, base.clone());
        let r = load_texture_bc(&png, true).unwrap();
        assert_eq!((r.width, r.height), (16, 12));
        assert_eq!(r.mips.len(), 5);
        validate_compressed(&r, 64).unwrap();

        // Compression None keeps raw pixels; no mips when disabled.
        base.max_size = 4096;
        base.compression = Compression::None;
        base.mips.enabled = false;
        write_import(&png, base);
        let raw = load_texture_bc(&png, true).unwrap();
        assert_eq!((raw.format, raw.mips.len()), (TexFormat::RgbaSrgb, 1));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cache_key_tracks_import_settings_but_not_the_sampler() {
        let dir = temp_dir("keys");
        let png = photo(&dir, "k.png", 16, 16);
        let base = TextureImport { configured: true, ..TextureImport::default() };
        write_import(&png, base.clone());
        let first = import_cache_path(&png, true, false).unwrap();
        load_texture_bc(&png, true).unwrap();
        assert!(first.is_file());
        // Sampler edits keep the cache file (no re-encode)...
        let mut sampler = base.clone();
        sampler.sampler.aniso = 2;
        sampler.sampler.wrap_u = crate::WrapMode::Clamp;
        write_import(&png, sampler);
        assert_eq!(import_cache_path(&png, true, false).unwrap(), first);
        // ...pixel-affecting edits move to a new one, ASTC has its own tier file.
        let mut size = base.clone();
        size.max_size = 8;
        write_import(&png, size);
        let moved = import_cache_path(&png, true, false).unwrap();
        assert_ne!(moved, first);
        load_texture_bc(&png, true).unwrap();
        assert!(moved.is_file() && first.is_file());
        assert!(import_cache_path(&png, true, true).unwrap().to_string_lossy().ends_with(".castc"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quest_override_and_quality_pick_the_astc_footprint() {
        let dir = temp_dir("quest");
        let png = photo(&dir, "q.png", 32, 32);
        let mut t = TextureImport { configured: true, ..TextureImport::default() };
        let import_for = |t: &TextureImport| {
            let data = load_texture_file(&png, true).unwrap();
            encode_data(&data, t, Platform::Quest)
        };
        assert_eq!(import_for(&t).format, TexFormat::Astc6x6Srgb);
        t.compression = Compression::Low;
        assert_eq!(import_for(&t).format, TexFormat::Astc8x8Srgb);
        t.compression = Compression::High;
        assert_eq!(import_for(&t).format, TexFormat::Astc4x4Srgb);
        t.quest = Some(crate::PlatformOverride { format: Some(crate::FormatOverride::Astc(5)), ..Default::default() });
        let five = import_for(&t);
        assert_eq!(five.format, TexFormat::Astc5x5Srgb);
        validate_compressed(&five, 64).unwrap();
        // Desktop is unaffected by the Quest override.
        assert_eq!(encode_data(&load_texture_file(&png, true).unwrap(), &t, Platform::Desktop).format, TexFormat::Bc7Srgb);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn transparency_dilation_and_normal_ops_run_in_the_pipeline() {
        let dir = temp_dir("ops");
        let png = dir.join("leaf.png");
        // Left half opaque green, right half transparent black.
        image::RgbaImage::from_fn(16, 16, |x, _| if x < 8 { image::Rgba([0, 200, 0, 255]) } else { image::Rgba([0, 0, 0, 0]) }).save(&png).unwrap();
        let data = load_texture_file(&png, true).unwrap();
        let mut t = TextureImport { configured: true, alpha_is_transparency: true, ..TextureImport::default() };
        t.compression = Compression::None;
        t.mips.enabled = false;
        let p = process_ldr(&data, &t, Platform::Desktop);
        assert_eq!(&p.levels[0].0[(8 * 4)..(8 * 4 + 4)], &[0, 200, 0, 0], "transparent texels take the neighbour colour, alpha stays 0");
        t.alpha_is_transparency = false;
        let p = process_ldr(&data, &t, Platform::Desktop);
        assert_eq!(&p.levels[0].0[(8 * 4)..(8 * 4 + 4)], &[0, 0, 0, 0]);
        // Alpha source None makes the image opaque and the format BC1.
        t = TextureImport { configured: true, alpha_source: AlphaSource::None, ..TextureImport::default() };
        let p = process_ldr(&data, &t, Platform::Desktop);
        assert_eq!(p.target, TargetFormat::Bc1);
        assert!(!p.alpha && p.levels[0].0.chunks(4).all(|px| px[3] == 255));
        // Normal map: flip green and renormalised mips.
        let mut n = TextureImport { configured: true, ..TextureImport::default() };
        n.apply_kind(TextureKind::NormalMap);
        n.normal.flip_green = true;
        let flat = zestengine_render::TextureData { width: 4, height: 4, pixels: [128u8, 200, 255, 255].repeat(16), srgb: false, hdr: false };
        let p = process_ldr(&flat, &n, Platform::Desktop);
        assert_eq!(p.levels[0].0[1], 55);
        assert_eq!(p.target, TargetFormat::Bc5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Real-texture VRAM log: legacy import vs the automatic format with a
    /// 2048 cap, over the wood-floor sample textures in
    /// `$ZESTENGINE_SAMPLE_TEXTURES` (default `<repo>/textures`; skips when absent).
    #[test]
    fn logs_vram_before_and_after_on_the_sample_textures() {
        let dir = std::env::var_os("ZESTENGINE_SAMPLE_TEXTURES")
            .map(PathBuf::from)
            .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../textures"));
        let samples: [(&str, TextureKind, bool); 4] = [
            ("wood_floor_diff_4k.jpg", TextureKind::Default, true),
            ("wood_floor_ao_4k.jpg", TextureKind::SingleChannel, false),
            ("wood_floor_arm_4k.jpg", TextureKind::Default, false),
            ("wood_floor_disp_4k.png", TextureKind::SingleChannel, false),
        ];
        let (mut before, mut after) = (0u64, 0u64);
        for (name, kind, srgb) in samples {
            let path = dir.join(name);
            if !path.is_file() {
                eprintln!("skipping {name}: not present");
                continue;
            }
            let mut t = TextureImport { configured: true, srgb, max_size: 2048, ..TextureImport::default() };
            t.apply_kind(kind);
            t.srgb = srgb && kind == TextureKind::Default;
            let old = preview_texture(&path, &TextureImport::legacy(srgb), Platform::Desktop).unwrap();
            let new = preview_texture(&path, &t, Platform::Desktop).unwrap();
            eprintln!(
                "VRAM {name}: {}x{} {:?} {} B -> {}x{} {:?} {} B",
                old.width, old.height, old.target, old.vram_bytes, new.width, new.height, new.target, new.vram_bytes
            );
            assert!(new.vram_bytes < old.vram_bytes);
            before += old.vram_bytes;
            after += new.vram_bytes;
        }
        if before > 0 {
            eprintln!("VRAM total: {before} B -> {after} B ({:.0}%)", (after as f64 / before as f64 - 1.0) * 100.0);
        }
    }

    #[test]
    fn astc_hdr_round_trips_values_above_one() {
        let (w, h) = (8u32, 8u32);
        let texel = |x: u32, y: u32| [1.0 + x as f32 * 0.5, 4.0 + y as f32 * 0.25, 0.25, 1.0];
        let halves: Vec<u16> =
            (0..h).flat_map(|y| (0..w).flat_map(move |x| texel(x, y))).map(|v| half::f16::from_f32(v).to_bits()).collect();
        let bytes: Vec<u8> = halves.iter().flat_map(|v| v.to_le_bytes()).collect();
        let blocks = encode_astc_hdr_level(&bytes, w, h).expect("encodes");
        assert_eq!(blocks.len(), 4 * 16, "four 4x4 blocks");
        let mut decoded = vec![0u16; (w * h * 4) as usize];
        // SAFETY: `decoded` holds w*h RGBA halves for the blocks just encoded.
        let status = unsafe { zestengine_astc_hdr_decode(blocks.as_ptr(), blocks.len(), w, h, 4, 4, decoded.as_mut_ptr()) };
        assert_eq!(status, 0);
        for (k, (got, want)) in decoded.iter().zip(&halves).enumerate() {
            let (got, want) = (half::f16::from_bits(*got).to_f32(), half::f16::from_bits(*want).to_f32());
            assert!((got - want).abs() <= want.abs() * 0.08 + 0.02, "texel channel {k}: {got} vs {want}");
        }
    }

    #[test]
    fn embedded_textures_cache_beside_their_model() {
        let dir = std::env::temp_dir().join(format!("zestengine-embedded-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("crate.glb");
        std::fs::write(&model, b"model bytes").unwrap();
        let data = ldr(8, 8, true, false);
        let first = compress_embedded_texture(Some(&model), 2, &data);
        let cached = legacy_cache(&dir.join("crate.glb#2"), true, false);
        assert!(cached.is_file(), "cache written next to the model");
        let second = compress_embedded_texture(Some(&model), 2, &data);
        assert_eq!((first.format, first.mips.clone()), (second.format, second.mips), "second load reads the cache");
        std::fs::remove_dir_all(&dir).ok();
    }
}