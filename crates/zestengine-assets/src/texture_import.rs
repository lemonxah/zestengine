//! Per-texture import settings (`ImporterSettings::Texture`), the automatic
//! GPU-format choice, the import cache key, inspector warnings and material
//! slot validation. Pure data + decisions: no pixel work (see
//! `texture_pipeline`) and no I/O except [`resolve_import`].
//!
//! `configured` separates two worlds. A texture whose `.meta` has no
//! `Texture` importer, or one that was never applied (`configured == false`),
//! loads exactly as before this module existed: colour space from the
//! material slot that references it, BC7 / the project ASTC footprint,
//! full-size, box mips. Applying the inspector sets `configured`, after which
//! the `.meta` is authoritative and the slot is only used for validation.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// What the texture is for. Picks the colour space, the compressed format
/// and which processing steps apply (see [`TextureImport::apply_kind`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TextureKind {
    /// Colour or generic data; `srgb` decides which.
    #[default]
    Default,
    /// Tangent-space normal map: linear, BC5 (two channels, Z rebuilt in the shader).
    NormalMap,
    /// 2D sprite / UI: no mips, alpha is transparency, clamped.
    Sprite,
    /// Mouse cursor: small, uncompressed, no mips.
    Cursor,
    /// Baked lightmap.
    Lightmap,
    /// One channel (roughness, AO, metallic, masks, height): BC4, half of BC7.
    SingleChannel,
}

impl TextureKind {
    pub const ALL: [Self; 6] = [Self::Default, Self::NormalMap, Self::Sprite, Self::Cursor, Self::Lightmap, Self::SingleChannel];

    pub fn label(self) -> &'static str {
        match self {
            Self::Default => "Default",
            Self::NormalMap => "Normal map",
            Self::Sprite => "Sprite",
            Self::Cursor => "Cursor",
            Self::Lightmap => "Lightmap",
            Self::SingleChannel => "Single channel",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlphaSource {
    /// Ignore the file's alpha (the texture is opaque).
    None,
    #[default]
    FromInput,
    /// Alpha = luminance of the colour.
    FromGray,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Channel {
    #[default]
    R,
    G,
    B,
    A,
}

impl Channel {
    pub fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NormalFilter {
    #[default]
    Sharp,
    Smooth,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MipFilter {
    #[default]
    Box,
    Kaiser,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NpotScale {
    #[default]
    None,
    ToNearest,
    ToLarger,
    ToSmaller,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResizeFilter {
    /// Cubic (Catmull-Rom stands in for Mitchell).
    #[default]
    Mitchell,
    Bilinear,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Compression {
    None,
    Low,
    #[default]
    Normal,
    High,
}

impl Compression {
    pub const ALL: [Self; 4] = [Self::None, Self::Low, Self::Normal, Self::High];
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Low => "Low quality",
            Self::Normal => "Normal quality",
            Self::High => "High quality",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WrapMode {
    #[default]
    Repeat,
    Clamp,
    Mirror,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FilterMode {
    Point,
    Bilinear,
    #[default]
    Trilinear,
}

/// An explicit format that replaces Automatic on a platform.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FormatOverride {
    Rgba8,
    RgbaF16,
    Bc1,
    Bc4,
    Bc5,
    Bc7,
    Bc6h,
    /// ASTC LDR with this square block edge (4, 5, 6 or 8).
    Astc(u8),
}

impl FormatOverride {
    pub fn label(self) -> String {
        match self {
            Self::Rgba8 => "RGBA8".into(),
            Self::RgbaF16 => "RGBA16F".into(),
            Self::Bc1 => "BC1".into(),
            Self::Bc4 => "BC4".into(),
            Self::Bc5 => "BC5".into(),
            Self::Bc7 => "BC7".into(),
            Self::Bc6h => "BC6H".into(),
            Self::Astc(b) => format!("ASTC {b}x{b}"),
        }
    }
}

/// The two import targets: BC desktop GPUs and ASTC standalone VR / mobile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Platform {
    Desktop,
    Quest,
}

impl Platform {
    pub fn label(self) -> &'static str {
        match self {
            Self::Desktop => "Desktop",
            Self::Quest => "Quest",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NormalOptions {
    /// DirectX (Y down) to OpenGL (Y up) convention.
    pub flip_green: bool,
    /// Treat the image as a height map and derive normals from it.
    pub from_height: bool,
    pub bumpiness: f32,
    pub filter: NormalFilter,
}

impl Default for NormalOptions {
    fn default() -> Self {
        Self { flip_green: false, from_height: false, bumpiness: 0.25, filter: NormalFilter::Sharp }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MipOptions {
    pub enabled: bool,
    pub filter: MipFilter,
    /// Rescale alpha per mip so alpha-tested coverage matches the base level.
    pub preserve_coverage: bool,
    /// Alpha threshold (0..1) the coverage is measured against.
    pub cutoff: f32,
}

impl Default for MipOptions {
    fn default() -> Self {
        Self { enabled: true, filter: MipFilter::Box, preserve_coverage: false, cutoff: 0.5 }
    }
}

/// Sampler state. Does not change pixels, so it is not part of the cache key.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SamplerOptions {
    pub wrap_u: WrapMode,
    pub wrap_v: WrapMode,
    pub filter: FilterMode,
    pub aniso: u8,
    /// Mip LOD bias in mip levels (positive = blurrier); the sampler keeps it in 1/8 steps.
    pub mip_bias: f32,
}

impl Default for SamplerOptions {
    fn default() -> Self {
        Self { wrap_u: WrapMode::Repeat, wrap_v: WrapMode::Repeat, filter: FilterMode::Trilinear, aniso: 8, mip_bias: 0.0 }
    }
}

/// One platform tab's override of the base size / quality / format.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PlatformOverride {
    pub max_size: u32,
    pub resize: ResizeFilter,
    pub compression: Compression,
    /// `None` = Automatic.
    pub format: Option<FormatOverride>,
}

impl Default for PlatformOverride {
    fn default() -> Self {
        Self { max_size: 2048, resize: ResizeFilter::Mitchell, compression: Compression::Normal, format: None }
    }
}

/// Everything the texture importer can be told about one image.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TextureImport {
    /// False = legacy behaviour (colour space from the material slot, BC7).
    pub configured: bool,
    pub kind: TextureKind,
    /// Colour data (decoded to linear on sample). Forced off for normal maps
    /// and single-channel kinds; ignored for HDR sources.
    pub srgb: bool,
    pub alpha_source: AlphaSource,
    /// Dilate colour into fully transparent texels so filtering doesn't pull
    /// in dark fringes.
    pub alpha_is_transparency: bool,
    /// Which source channel a `SingleChannel` texture keeps.
    pub single_channel: Channel,
    pub normal: NormalOptions,
    pub mips: MipOptions,
    pub npot: NpotScale,
    pub sampler: SamplerOptions,
    /// Largest edge in texels (0 = no limit).
    pub max_size: u32,
    pub resize: ResizeFilter,
    pub compression: Compression,
    pub desktop: Option<PlatformOverride>,
    pub quest: Option<PlatformOverride>,
    /// Sprite table and settings (2D-2). Only used while `kind` is `Sprite`,
    /// but kept when the kind changes so switching back loses nothing.
    pub sprite: crate::sprite::SpriteSheet,
}

impl Default for TextureImport {
    fn default() -> Self {
        Self {
            configured: false,
            kind: TextureKind::Default,
            srgb: true,
            alpha_source: AlphaSource::FromInput,
            alpha_is_transparency: false,
            single_channel: Channel::R,
            normal: NormalOptions::default(),
            mips: MipOptions::default(),
            npot: NpotScale::None,
            sampler: SamplerOptions::default(),
            max_size: 4096,
            resize: ResizeFilter::Mitchell,
            compression: Compression::Normal,
            desktop: None,
            quest: None,
            sprite: crate::sprite::SpriteSheet::default(),
        }
    }
}

/// What a texture asset turned out to contain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceInfo {
    pub width: u32,
    pub height: u32,
    /// Some texel is not fully opaque.
    pub has_alpha: bool,
    pub hdr: bool,
}

/// The encoding chosen for a texture on one platform.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TargetFormat {
    Rgba8,
    RgbaF16,
    Bc1,
    Bc4,
    Bc5,
    Bc7,
    Bc6h,
    Astc(u32),
}

impl TargetFormat {
    pub fn label(self) -> String {
        match self {
            Self::Rgba8 => "RGBA8".into(),
            Self::RgbaF16 => "RGBA16F".into(),
            Self::Bc1 => "BC1".into(),
            Self::Bc4 => "BC4".into(),
            Self::Bc5 => "BC5".into(),
            Self::Bc7 => "BC7".into(),
            Self::Bc6h => "BC6H".into(),
            Self::Astc(b) => format!("ASTC {b}x{b}"),
        }
    }

    pub fn is_block_compressed(self) -> bool {
        !matches!(self, Self::Rgba8 | Self::RgbaF16)
    }

    fn from_override(o: FormatOverride) -> Self {
        match o {
            FormatOverride::Rgba8 => Self::Rgba8,
            FormatOverride::RgbaF16 => Self::RgbaF16,
            FormatOverride::Bc1 => Self::Bc1,
            FormatOverride::Bc4 => Self::Bc4,
            FormatOverride::Bc5 => Self::Bc5,
            FormatOverride::Bc7 => Self::Bc7,
            FormatOverride::Bc6h => Self::Bc6h,
            FormatOverride::Astc(b) => Self::Astc(b as u32),
        }
    }
}

/// Smallest power of two >= n (n >= 1).
fn pot_up(n: u32) -> u32 {
    n.max(1).next_power_of_two()
}
/// Largest power of two <= n (n >= 1).
fn pot_down(n: u32) -> u32 {
    let n = n.max(1);
    1 << (31 - n.leading_zeros())
}
/// Nearest power of two (ties round up).
fn pot_nearest(n: u32) -> u32 {
    let (lo, hi) = (pot_down(n), pot_up(n));
    if n - lo < hi - n { lo } else { hi }
}

impl TextureImport {
    /// The loader's behaviour for textures nobody configured: colour space
    /// from the referencing slot, nothing else changed.
    pub fn legacy(srgb: bool) -> Self {
        Self { srgb, max_size: 0, ..Self::default() }
    }

    /// Texture-type defaults (research phase1b table 1b):
    /// fields that always follow the kind, and fields reset only when the
    /// kind actually changes. "Create from grayscale" is always cleared.
    pub fn apply_kind(&mut self, kind: TextureKind) {
        let changed = self.kind != kind;
        self.kind = kind;
        self.normal.from_height = false;
        match kind {
            TextureKind::Default => {
                if changed {
                    self.alpha_is_transparency = false;
                    self.srgb = true;
                    self.mips.enabled = true;
                }
            }
            TextureKind::NormalMap => {
                self.srgb = false;
                self.alpha_is_transparency = false;
                if changed {
                    self.mips.enabled = true;
                    self.normal.flip_green = false;
                }
            }
            TextureKind::Cursor => {
                self.srgb = true;
                if changed {
                    self.alpha_is_transparency = true;
                    self.mips.enabled = false;
                    self.npot = NpotScale::None;
                    self.sampler.aniso = 1;
                    self.sampler.wrap_u = WrapMode::Clamp;
                    self.sampler.wrap_v = WrapMode::Clamp;
                }
            }
            TextureKind::Sprite => {
                self.npot = NpotScale::None;
                if changed {
                    self.alpha_is_transparency = true;
                    self.srgb = true;
                    self.mips.enabled = false;
                    self.sampler.wrap_u = WrapMode::Clamp;
                    self.sampler.wrap_v = WrapMode::Clamp;
                    // Pixel art and tight meshes want unfiltered texels by default.
                    self.sampler.filter = FilterMode::Point;
                }
                // The sprite table (`self.sprite`) is never reset by a kind change.
            }
            TextureKind::Lightmap => {
                self.npot = NpotScale::ToNearest;
                self.srgb = true;
                self.alpha_is_transparency = false;
                self.alpha_source = AlphaSource::None;
                self.sampler.wrap_u = WrapMode::Clamp;
                self.sampler.wrap_v = WrapMode::Clamp;
            }
            TextureKind::SingleChannel => {
                self.srgb = false;
            }
        }
    }

    /// Colour space actually used for LDR data.
    pub fn effective_srgb(&self) -> bool {
        match self.kind {
            TextureKind::NormalMap | TextureKind::SingleChannel => false,
            _ => self.srgb,
        }
    }

    /// The size / quality / format settings for `platform`: its override when
    /// present, else the base values (Automatic format).
    pub fn platform_settings(&self, platform: Platform) -> PlatformOverride {
        let over = match platform {
            Platform::Desktop => self.desktop,
            Platform::Quest => self.quest,
        };
        over.unwrap_or(PlatformOverride { max_size: self.max_size, resize: self.resize, compression: self.compression, format: None })
    }

    /// Settings a freshly ticked override checkbox starts from.
    pub fn override_seed(&self, platform: Platform) -> PlatformOverride {
        let mut o = self.platform_settings(platform);
        if platform == Platform::Quest && self.quest.is_none() {
            o.max_size = o.max_size.min(2048);
        }
        o
    }

    /// Pixel size after the NPOT rule and the platform's max size.
    pub fn target_size(&self, w: u32, h: u32, platform: Platform) -> (u32, u32) {
        let (mut w, mut h) = (w.max(1), h.max(1));
        let npot = |n: u32| match self.npot {
            NpotScale::None => n,
            NpotScale::ToNearest => pot_nearest(n),
            NpotScale::ToLarger => pot_up(n),
            NpotScale::ToSmaller => pot_down(n),
        };
        (w, h) = (npot(w), npot(h));
        let max = self.platform_settings(platform).max_size;
        if max > 0 && w.max(h) > max {
            if w.is_power_of_two() && h.is_power_of_two() {
                // Halving keeps POT sizes POT and the aspect exact.
                while w.max(h) > max && (w > 1 || h > 1) {
                    w = (w / 2).max(1);
                    h = (h / 2).max(1);
                }
            } else {
                let s = max as f64 / w.max(h) as f64;
                w = ((w as f64 * s).round() as u32).clamp(1, max);
                h = ((h as f64 * s).round() as u32).clamp(1, max);
            }
        }
        (w, h)
    }

    /// Whether the encoded texture carries alpha after the alpha-source rule.
    pub fn uses_alpha(&self, info: &SourceInfo) -> bool {
        match self.alpha_source {
            AlphaSource::None => false,
            AlphaSource::FromInput => info.has_alpha,
            AlphaSource::FromGray => true,
        }
    }

    /// Automatic (or overridden) encoding for `platform`.
    ///
    /// BC tier: BC1 for opaque sRGB colour, BC4 single channel, BC5 normal
    /// maps, BC7 for alpha and for linear data (ORM and friends need the
    /// precision) and for High quality, BC6H for HDR. ASTC tier: 8x8 Low, the
    /// project footprint (`astc_default`, 6x6 colour / 4x4 data) Normal,
    /// 4x4 High. `None` compression is uncompressed. Textures nobody
    /// configured keep the legacy BC7 / project ASTC choice.
    pub fn select_format(&self, platform: Platform, info: &SourceInfo, astc_default: u32) -> TargetFormat {
        let ps = self.platform_settings(platform);
        if let Some(f) = ps.format {
            return TargetFormat::from_override(f);
        }
        if info.hdr {
            return match (platform, ps.compression) {
                (_, Compression::None) => TargetFormat::RgbaF16,
                (Platform::Desktop, _) => TargetFormat::Bc6h,
                // Quest HDR: the cache layer picks ASTC HDR or raw f16.
                (Platform::Quest, _) => TargetFormat::RgbaF16,
            };
        }
        let compression = if self.configured { ps.compression } else { Compression::Normal };
        if compression == Compression::None || (self.configured && self.kind == TextureKind::Cursor) {
            return TargetFormat::Rgba8;
        }
        match platform {
            Platform::Desktop => {
                if !self.configured {
                    return TargetFormat::Bc7;
                }
                match self.kind {
                    TextureKind::NormalMap => TargetFormat::Bc5,
                    TextureKind::SingleChannel => TargetFormat::Bc4,
                    _ => {
                        let alpha = self.uses_alpha(info);
                        if alpha || compression == Compression::High || !self.effective_srgb() {
                            TargetFormat::Bc7
                        } else {
                            TargetFormat::Bc1
                        }
                    }
                }
            }
            Platform::Quest => {
                if !self.configured {
                    return TargetFormat::Astc(astc_default);
                }
                TargetFormat::Astc(match compression {
                    Compression::Low => 8,
                    Compression::High => 4,
                    _ => astc_default,
                })
            }
        }
    }

    /// Stable 16-hex-digit key of everything that changes the encoded pixels
    /// for `platform`. Sampler state is excluded. `astc_default` is the
    /// project's ASTC footprint for this slot (Quest only).
    pub fn cache_key(&self, platform: Platform, astc_default: u32) -> String {
        let ps = self.platform_settings(platform);
        let mut s = format!("v{ENCODER_VERSION}|{}|{:?}|{astc_default}", platform.label(), platform);
        if !self.configured {
            // Legacy: only the colour space (and tier) matters.
            s.push_str(&format!("|legacy|{}", self.srgb));
        } else {
            s.push_str(&format!(
                "|{:?}|{}|{:?}|{}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}",
                self.kind,
                self.effective_srgb(),
                self.alpha_source,
                self.alpha_is_transparency,
                self.single_channel,
                self.normal.filter,
                self.mips.filter,
                self.npot,
                ps,
                (self.normal.flip_green, self.normal.from_height, self.normal.bumpiness.to_bits(), self.mips.enabled, self.mips.preserve_coverage, self.mips.cutoff.to_bits()),
            ));
        }
        format!("{:016x}", fnv1a64(s.as_bytes()))
    }

    /// Structural problems worth surfacing in the inspector.
    pub fn warnings(&self, info: &SourceInfo, platform: Platform, file_name: &str) -> Vec<TextureWarning> {
        let mut out = Vec::new();
        let ps = self.platform_settings(platform);
        let npot = !(info.width.is_power_of_two() && info.height.is_power_of_two());
        if npot && self.npot == NpotScale::None && self.mips.enabled && self.sampler.wrap_u == WrapMode::Repeat {
            out.push(TextureWarning::warn(
                "Non-power-of-two size with mipmaps and Repeat wrap. Some devices don't support this; use NPOT scaling or Clamp.",
            ));
        }
        if let Some(f) = ps.format
            && let Some(why) = format_override_problem(f, platform, self.kind, info)
        {
            out.push(TextureWarning::error(why));
        }
        if self.kind != TextureKind::NormalMap && name_suggests_normal_map(file_name) {
            out.push(TextureWarning::warn("The name looks like a normal map but the type is not Normal map."));
        }
        if self.kind == TextureKind::NormalMap && info.hdr {
            out.push(TextureWarning::warn("HDR sources can't be normal maps; use an LDR image."));
        }
        if self.alpha_is_transparency && (self.alpha_source == AlphaSource::None || (!info.has_alpha && self.alpha_source == AlphaSource::FromInput)) {
            out.push(TextureWarning::info("Alpha Is Transparency has no effect: there is no alpha."));
        }
        if self.kind == TextureKind::SingleChannel && info.hdr {
            out.push(TextureWarning::warn("Single channel isn't applied to HDR sources."));
        }
        if ps.max_size > 0 && !ps.max_size.is_power_of_two() {
            out.push(TextureWarning::info("Max size isn't a power of two."));
        }
        if info.width.max(info.height) > ps.max_size && ps.max_size > 0 {
            out.push(TextureWarning::info(format!(
                "Source is {}x{}; it will be resized to fit {}.",
                info.width, info.height, ps.max_size
            )));
        }
        out
    }

    /// Short problem string if the settings are internally inconsistent.
    pub fn name_hint_kind(file_name: &str) -> Option<TextureKind> {
        let lower = file_name.to_ascii_lowercase();
        let stem = lower.rsplit_once('.').map_or(lower.as_str(), |(s, _)| s);
        if name_suggests_normal_map(file_name) {
            return Some(TextureKind::NormalMap);
        }
        let tail = |names: &[&str]| names.iter().any(|n| stem.ends_with(n));
        if tail(&["_rough", "_roughness", "_ao", "_metal", "_metallic", "_metalness", "_height", "_disp", "_displacement", "_opacity", "_mask"]) {
            return Some(TextureKind::SingleChannel);
        }
        None
    }
}

/// Bump when the encode pipeline changes pixels or layout so caches rebuild.
const ENCODER_VERSION: u32 = 4;

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// `foo_n.png`, `foo_normal.png`, `foo_nrm.png`, `foo_normalgl.png` ...
pub fn name_suggests_normal_map(file_name: &str) -> bool {
    let lower = file_name.to_ascii_lowercase();
    let stem = lower.rsplit_once('.').map_or(lower.as_str(), |(s, _)| s);
    ["_n", "_nrm", "_nor", "_normal", "_normalgl", "_normaldx", "_normalmap", "_norm", ".normal", "-normal", "-n"]
        .iter()
        .any(|suffix| stem.ends_with(suffix))
}

fn format_override_problem(f: FormatOverride, platform: Platform, kind: TextureKind, info: &SourceInfo) -> Option<String> {
    let bc = matches!(f, FormatOverride::Bc1 | FormatOverride::Bc4 | FormatOverride::Bc5 | FormatOverride::Bc7 | FormatOverride::Bc6h);
    if bc && platform == Platform::Quest {
        return Some(format!("{} isn't sampled by Quest GPUs; use ASTC.", f.label()));
    }
    if matches!(f, FormatOverride::Astc(_)) && platform == Platform::Desktop {
        return Some("ASTC isn't sampled by most desktop GPUs; use BC.".into());
    }
    if let FormatOverride::Astc(b) = f
        && ![4, 5, 6, 8].contains(&b)
    {
        return Some(format!("ASTC {b}x{b} isn't supported (4, 5, 6 or 8)."));
    }
    if f == FormatOverride::Bc6h && !info.hdr {
        return Some("BC6H is for HDR sources only.".into());
    }
    if info.hdr && !matches!(f, FormatOverride::Bc6h | FormatOverride::RgbaF16 | FormatOverride::Astc(_)) {
        return Some(format!("{} can't hold HDR values; use BC6H or RGBA16F.", f.label()));
    }
    if f == FormatOverride::Bc5 && kind != TextureKind::NormalMap {
        return Some("BC5 only stores two channels; use it for normal maps.".into());
    }
    if f == FormatOverride::Bc4 && kind != TextureKind::SingleChannel {
        return Some("BC4 only stores one channel; use it for single-channel textures.".into());
    }
    if kind == TextureKind::NormalMap && matches!(f, FormatOverride::Bc1) {
        return Some("BC1 is too coarse for normal maps.".into());
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarnLevel {
    Info,
    Warning,
    Error,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextureWarning {
    pub level: WarnLevel,
    pub text: String,
}

impl TextureWarning {
    fn info(t: impl Into<String>) -> Self {
        Self { level: WarnLevel::Info, text: t.into() }
    }
    fn warn(t: impl Into<String>) -> Self {
        Self { level: WarnLevel::Warning, text: t.into() }
    }
    fn error(t: impl Into<String>) -> Self {
        Self { level: WarnLevel::Error, text: t.into() }
    }
}

/// The settings that apply to `path`: its `.meta` when that holds a
/// configured texture importer, otherwise the legacy settings for
/// `slot_srgb` (the colour space of the material slot loading it).
pub fn resolve_import(path: &Path, slot_srgb: bool) -> TextureImport {
    match crate::asset_meta::load_asset_meta(path) {
        Ok(Some(meta)) => match meta.importer {
            crate::ImporterSettings::Texture(t) if t.configured => t,
            _ => TextureImport::legacy(slot_srgb),
        },
        _ => TextureImport::legacy(slot_srgb),
    }
}

// ---------------------------------------------------------------------------
// Material slot validation (TEX-3)
// ---------------------------------------------------------------------------

/// What a material texture slot needs from its texture's import settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SlotExpect {
    /// sRGB colour (albedo, emission, matcap).
    Color,
    /// Tangent-space normal map.
    Normal,
    /// Linear packed data (ORM).
    Data,
    /// Linear single-channel data (AO, roughness, metallic, masks, height).
    Single,
}

impl SlotExpect {
    /// Slots that can share one texture import without conflict.
    fn class(self) -> u8 {
        match self {
            Self::Color => 0,
            Self::Normal => 1,
            Self::Data | Self::Single => 2,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Color => "sRGB colour",
            Self::Normal => "Normal map",
            Self::Data => "linear data",
            Self::Single => "linear single-channel data",
        }
    }
}

/// Expectation and display name of material texture slot `index` (0..15 in
/// binding order, see `MaterialTexturePaths::slot_mut`). Custom shader slots
/// (>= 16) have none.
pub fn slot_expectation(index: usize) -> Option<(SlotExpect, &'static str)> {
    Some(match index {
        0 => (SlotExpect::Color, "Albedo"),
        1 => (SlotExpect::Normal, "Normal"),
        2 => (SlotExpect::Data, "ORM"),
        3 => (SlotExpect::Color, "Emission"),
        4 => (SlotExpect::Single, "Opacity"),
        5 => (SlotExpect::Single, "Emission mask"),
        6 | 8 | 10 => (SlotExpect::Color, "Matcap"),
        7 | 9 | 11 => (SlotExpect::Single, "Matcap mask"),
        12 => (SlotExpect::Single, "Ambient occlusion"),
        13 => (SlotExpect::Single, "Roughness"),
        14 => (SlotExpect::Single, "Metallic"),
        15 => (SlotExpect::Single, "Displacement"),
        _ => return None,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotIssueKind {
    NotNormalMap,
    ShouldBeSrgb,
    ShouldBeLinear,
    /// Linear data on a slot that only reads one channel: BC4 halves VRAM.
    CouldBeSingleChannel,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotIssue {
    pub kind: SlotIssueKind,
    pub text: String,
    /// Informational (a saving, not a rendering bug).
    pub info: bool,
}

/// Mismatches between a slot's expectation and a configured texture's
/// settings. Textures that are not configured load per-slot, so they can't
/// mismatch: pass `None`.
pub fn slot_issues(expect: SlotExpect, import: Option<&TextureImport>) -> Vec<SlotIssue> {
    let Some(t) = import.filter(|t| t.configured) else { return Vec::new() };
    let mut out = Vec::new();
    let mut push = |kind, info, text: &str| out.push(SlotIssue { kind, text: text.to_owned(), info });
    match expect {
        SlotExpect::Normal => {
            if t.kind != TextureKind::NormalMap {
                push(SlotIssueKind::NotNormalMap, false, "This texture is not marked as a normal map.");
            }
        }
        SlotExpect::Color => {
            if t.kind == TextureKind::NormalMap || t.kind == TextureKind::SingleChannel || !t.srgb {
                push(SlotIssueKind::ShouldBeSrgb, false, "This slot expects sRGB colour but the texture is imported as linear data.");
            }
        }
        SlotExpect::Data | SlotExpect::Single => {
            if t.kind == TextureKind::NormalMap || (t.srgb && t.kind != TextureKind::SingleChannel) {
                push(SlotIssueKind::ShouldBeLinear, false, "This slot expects linear data but the texture is imported as sRGB colour.");
            } else if expect == SlotExpect::Single && t.kind == TextureKind::Default {
                push(SlotIssueKind::CouldBeSingleChannel, true, "Only one channel is read here. Single channel uses BC4 (half the VRAM).");
            }
        }
    }
    out
}

/// Settings change that satisfies `expect` (what Auto-fix applies). Returns
/// whether anything changed.
pub fn apply_expectation(t: &mut TextureImport, expect: SlotExpect) -> bool {
    let before = t.clone();
    match expect {
        SlotExpect::Normal => t.apply_kind(TextureKind::NormalMap),
        SlotExpect::Color => {
            if t.kind != TextureKind::Default {
                t.apply_kind(TextureKind::Default);
            }
            t.srgb = true;
        }
        SlotExpect::Data => {
            if t.kind != TextureKind::Default {
                t.apply_kind(TextureKind::Default);
            }
            t.srgb = false;
        }
        SlotExpect::Single => {
            if t.kind == TextureKind::NormalMap {
                t.apply_kind(TextureKind::Default);
            }
            // Keep a single-channel kind; plain Default data just goes linear.
            t.srgb = false;
            if t.kind == TextureKind::Default {
                t.apply_kind(TextureKind::SingleChannel);
            }
        }
    }
    *t != before
}

/// One use of a texture by a material slot, for conflict detection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotUse {
    /// Material (or object) the use belongs to.
    pub owner: String,
    pub slot: usize,
}

/// Other uses of the same texture whose expectation can't be met by the same
/// import settings as `this` slot.
pub fn conflicting_uses(this: SlotExpect, others: &[SlotUse]) -> Vec<&SlotUse> {
    others
        .iter()
        .filter(|u| slot_expectation(u.slot).is_some_and(|(e, _)| e.class() != this.class()))
        .collect()
}

/// A free sibling name for a duplicate made for `expect`:
/// `wood.png` -> `wood_linear.png` / `wood_normal.png` / `wood_srgb.png`
/// (then `_2`, `_3` ...). `exists` reports taken paths.
pub fn duplicate_path(path: &Path, expect: SlotExpect, exists: impl Fn(&Path) -> bool) -> PathBuf {
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "texture".into());
    let ext = path.extension().map(|e| e.to_string_lossy().into_owned());
    let tag = match expect {
        SlotExpect::Color => "srgb",
        SlotExpect::Normal => "normal",
        SlotExpect::Data | SlotExpect::Single => "linear",
    };
    let make = |n: u32| {
        let name = if n <= 1 { format!("{stem}_{tag}") } else { format!("{stem}_{tag}_{n}") };
        let file = match &ext {
            Some(e) => format!("{name}.{e}"),
            None => name,
        };
        path.with_file_name(file)
    };
    (1..).map(make).find(|p| !exists(p)).expect("unbounded range")
}

/// A new `.meta` importer for a duplicate made for `expect`, configured.
pub fn import_for_expectation(expect: SlotExpect) -> TextureImport {
    let mut t = TextureImport { configured: true, ..TextureImport::default() };
    apply_expectation(&mut t, expect);
    t
}

/// Slot-aware first import: a texture with no `.meta` yet that is first
/// assigned to a material slot gets the importer that slot expects (a normal
/// slot makes a NormalMap, a mask slot a SingleChannel, ...) instead of
/// Default. Existing metas (configured or not) are never touched, so settled
/// projects import as before. Returns whether a `.meta` was written.
pub fn ensure_first_import_for_slot(abs: &Path, slot: usize) -> anyhow::Result<bool> {
    let Some((expect, _)) = slot_expectation(slot) else { return Ok(false) };
    if !abs.is_file() || crate::asset_meta::meta_path(abs).exists() {
        return Ok(false);
    }
    let mut meta = crate::asset_meta::AssetMeta::new_for(abs);
    // `new_for` only gives a texture importer to image files; leave others alone.
    if meta.importer.as_texture().is_none() {
        return Ok(false);
    }
    meta.importer = crate::ImporterSettings::Texture(import_for_expectation(expect));
    crate::asset_meta::save_asset_meta(abs, &meta)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_assignment_to_a_normal_slot_configures_a_normal_map() {
        let dir = std::env::temp_dir().join(format!("zest_first_import_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tex = dir.join("brick_n.png");
        std::fs::write(&tex, b"\x89PNG").unwrap();
        assert!(ensure_first_import_for_slot(&tex, 1).unwrap());
        let t = crate::asset_meta::load_asset_meta(&tex).unwrap().unwrap().importer.as_texture().cloned().unwrap();
        assert!(t.configured);
        assert_eq!(t.kind, TextureKind::NormalMap);
        assert!(!t.srgb);
        // The second slot assignment must not rewrite an existing meta.
        assert!(!ensure_first_import_for_slot(&tex, 0).unwrap());
        let again = crate::asset_meta::load_asset_meta(&tex).unwrap().unwrap().importer.as_texture().cloned().unwrap();
        assert_eq!(again.kind, TextureKind::NormalMap);
        // A mask slot makes a single-channel texture; slots without an expectation do nothing.
        let mask = dir.join("mask.png");
        std::fs::write(&mask, b"\x89PNG").unwrap();
        assert!(!ensure_first_import_for_slot(&mask, 99).unwrap());
        assert!(ensure_first_import_for_slot(&mask, 4).unwrap());
        let m = crate::asset_meta::load_asset_meta(&mask).unwrap().unwrap().importer.as_texture().cloned().unwrap();
        assert_eq!(m.kind, TextureKind::SingleChannel);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn info(w: u32, h: u32, alpha: bool) -> SourceInfo {
        SourceInfo { width: w, height: h, has_alpha: alpha, hdr: false }
    }

    #[test]
    fn old_metas_without_texture_fields_still_parse() {
        // A meta written before this importer existed.
        let generic: crate::AssetMeta = ron::from_str("(id: \"00000000-0000-0000-0000-000000000001\", importer: Generic)").unwrap();
        assert!(matches!(generic.importer, crate::ImporterSettings::Generic));
        // A Texture importer that only knows a couple of fields fills the rest.
        let m: crate::AssetMeta = ron::from_str("(importer: Texture((srgb: false, kind: NormalMap)))").unwrap();
        let t = match m.importer {
            crate::ImporterSettings::Texture(t) => t,
            _ => panic!("texture importer"),
        };
        assert_eq!(t.kind, TextureKind::NormalMap);
        assert!(!t.srgb);
        assert!(!t.configured, "a hand-written partial meta stays legacy until applied");
        assert_eq!(t.max_size, 4096);
        assert_eq!(t.sampler.aniso, 8);
        // And an empty one.
        let e: TextureImport = ron::from_str("()").unwrap();
        assert_eq!(e, TextureImport::default());
    }

    #[test]
    fn texture_import_round_trips() {
        let mut t = TextureImport { configured: true, ..TextureImport::default() };
        t.apply_kind(TextureKind::NormalMap);
        t.quest = Some(PlatformOverride { max_size: 512, format: Some(FormatOverride::Astc(8)), ..PlatformOverride::default() });
        let s = ron::ser::to_string_pretty(&t, Default::default()).unwrap();
        assert_eq!(ron::from_str::<TextureImport>(&s).unwrap(), t);
    }

    #[test]
    fn legacy_textures_keep_bc7_and_slot_colour_space() {
        let l = TextureImport::legacy(false);
        assert!(!l.effective_srgb());
        assert_eq!(l.select_format(Platform::Desktop, &info(512, 512, false), 4), TargetFormat::Bc7);
        assert_eq!(l.select_format(Platform::Quest, &info(512, 512, false), 4), TargetFormat::Astc(4));
        assert_eq!(l.target_size(8192, 8192, Platform::Desktop), (8192, 8192), "legacy never resizes");
    }

    #[test]
    fn apply_kind_ports_the_type_change_resets() {
        let mut t = TextureImport { configured: true, ..TextureImport::default() };
        t.normal.from_height = true;
        t.apply_kind(TextureKind::NormalMap);
        assert!(!t.srgb && !t.alpha_is_transparency && t.mips.enabled && !t.normal.from_height);
        t.normal.flip_green = true;
        t.apply_kind(TextureKind::NormalMap); // unchanged kind: flip_green survives
        assert!(t.normal.flip_green);
        t.apply_kind(TextureKind::Sprite);
        assert!(t.srgb && t.alpha_is_transparency && !t.mips.enabled);
        assert_eq!((t.sampler.wrap_u, t.sampler.wrap_v), (WrapMode::Clamp, WrapMode::Clamp));
        t.apply_kind(TextureKind::Cursor);
        assert_eq!(t.sampler.aniso, 1);
        t.apply_kind(TextureKind::Default);
        assert!(t.srgb && t.mips.enabled && !t.alpha_is_transparency);
        t.apply_kind(TextureKind::Lightmap);
        assert_eq!(t.alpha_source, AlphaSource::None);
        assert_eq!(t.npot, NpotScale::ToNearest);
        t.apply_kind(TextureKind::SingleChannel);
        assert!(!t.srgb && !t.effective_srgb());
    }

    #[test]
    fn automatic_format_follows_kind_alpha_and_quality() {
        let cfg = TextureImport { configured: true, ..TextureImport::default() };
        let d = Platform::Desktop;
        assert_eq!(cfg.select_format(d, &info(256, 256, false), 6), TargetFormat::Bc1);
        assert_eq!(cfg.select_format(d, &info(256, 256, true), 6), TargetFormat::Bc7);
        let mut hq = cfg.clone();
        hq.compression = Compression::High;
        assert_eq!(hq.select_format(d, &info(256, 256, false), 6), TargetFormat::Bc7);
        let mut none_alpha = cfg.clone();
        none_alpha.alpha_source = AlphaSource::None;
        assert_eq!(none_alpha.select_format(d, &info(256, 256, true), 6), TargetFormat::Bc1, "alpha ignored -> opaque row");
        let mut lin = cfg.clone();
        lin.srgb = false;
        assert_eq!(lin.select_format(d, &info(256, 256, false), 6), TargetFormat::Bc7, "linear data keeps BC7 precision");
        let mut n = cfg.clone();
        n.apply_kind(TextureKind::NormalMap);
        assert_eq!(n.select_format(d, &info(256, 256, false), 6), TargetFormat::Bc5);
        let mut s = cfg.clone();
        s.apply_kind(TextureKind::SingleChannel);
        assert_eq!(s.select_format(d, &info(256, 256, false), 6), TargetFormat::Bc4);
        let mut raw = cfg.clone();
        raw.compression = Compression::None;
        assert_eq!(raw.select_format(d, &info(256, 256, true), 6), TargetFormat::Rgba8);
        let hdr = SourceInfo { hdr: true, ..info(256, 256, false) };
        assert_eq!(cfg.select_format(d, &hdr, 6), TargetFormat::Bc6h);
    }

    #[test]
    fn astc_block_follows_quality() {
        let mut t = TextureImport { configured: true, ..TextureImport::default() };
        let q = Platform::Quest;
        assert_eq!(t.select_format(q, &info(64, 64, false), 6), TargetFormat::Astc(6));
        t.compression = Compression::Low;
        assert_eq!(t.select_format(q, &info(64, 64, false), 6), TargetFormat::Astc(8));
        t.compression = Compression::High;
        assert_eq!(t.select_format(q, &info(64, 64, false), 6), TargetFormat::Astc(4));
        t.quest = Some(PlatformOverride { format: Some(FormatOverride::Astc(5)), ..PlatformOverride::default() });
        assert_eq!(t.select_format(q, &info(64, 64, false), 6), TargetFormat::Astc(5));
    }

    #[test]
    fn target_size_applies_npot_then_max_size() {
        let mut t = TextureImport { configured: true, max_size: 1024, ..TextureImport::default() };
        assert_eq!(t.target_size(4096, 2048, Platform::Desktop), (1024, 512));
        assert_eq!(t.target_size(600, 300, Platform::Desktop), (600, 300));
        assert_eq!(t.target_size(3000, 1500, Platform::Desktop), (1024, 512));
        t.npot = NpotScale::ToNearest;
        assert_eq!(t.target_size(600, 300, Platform::Desktop), (512, 256));
        t.npot = NpotScale::ToLarger;
        assert_eq!(t.target_size(600, 300, Platform::Desktop), (1024, 512));
        t.npot = NpotScale::ToSmaller;
        assert_eq!(t.target_size(600, 300, Platform::Desktop), (512, 256));
        t.npot = NpotScale::None;
        t.quest = Some(PlatformOverride { max_size: 256, ..PlatformOverride::default() });
        assert_eq!(t.target_size(1024, 1024, Platform::Quest), (256, 256));
        assert_eq!(t.target_size(1024, 1024, Platform::Desktop), (1024, 1024));
    }

    #[test]
    fn cache_key_ignores_sampler_and_tracks_pixel_settings() {
        let a = TextureImport { configured: true, ..TextureImport::default() };
        let mut b = a.clone();
        b.sampler.aniso = 1;
        b.sampler.wrap_u = WrapMode::Clamp;
        b.sampler.filter = FilterMode::Point;
        b.sampler.mip_bias = 1.5;
        assert_eq!(a.cache_key(Platform::Desktop, 6), b.cache_key(Platform::Desktop, 6));
        let mut c = a.clone();
        c.max_size = 512;
        assert_ne!(a.cache_key(Platform::Desktop, 6), c.cache_key(Platform::Desktop, 6));
        let mut d = a.clone();
        d.normal.flip_green = true;
        assert_ne!(a.cache_key(Platform::Desktop, 6), d.cache_key(Platform::Desktop, 6));
        assert_ne!(a.cache_key(Platform::Desktop, 6), a.cache_key(Platform::Quest, 6));
        // The Quest override doesn't move the desktop key.
        let mut e = a.clone();
        e.quest = Some(PlatformOverride { max_size: 256, ..PlatformOverride::default() });
        assert_eq!(a.cache_key(Platform::Desktop, 6), e.cache_key(Platform::Desktop, 6));
        assert_ne!(a.cache_key(Platform::Quest, 6), e.cache_key(Platform::Quest, 6));
        assert_ne!(TextureImport::legacy(true).cache_key(Platform::Desktop, 6), TextureImport::legacy(false).cache_key(Platform::Desktop, 6));
        assert_eq!(a.cache_key(Platform::Desktop, 6).len(), 16);
    }

    #[test]
    fn warnings_cover_npot_overrides_and_names() {
        let t = TextureImport { configured: true, ..TextureImport::default() };
        let w = t.warnings(&info(300, 200, false), Platform::Desktop, "wood_n.png");
        assert!(w.iter().any(|w| w.text.contains("Non-power-of-two")));
        assert!(w.iter().any(|w| w.text.contains("normal map")));
        let mut bad = t.clone();
        bad.desktop = Some(PlatformOverride { format: Some(FormatOverride::Astc(6)), ..PlatformOverride::default() });
        assert!(bad.warnings(&info(256, 256, false), Platform::Desktop, "a.png").iter().any(|w| w.level == WarnLevel::Error));
        let mut q = t.clone();
        q.quest = Some(PlatformOverride { format: Some(FormatOverride::Bc7), ..PlatformOverride::default() });
        assert!(q.warnings(&info(256, 256, false), Platform::Quest, "a.png").iter().any(|w| w.level == WarnLevel::Error));
    }

    #[test]
    fn name_hints() {
        assert!(name_suggests_normal_map("Brick_Normal.PNG"));
        assert!(name_suggests_normal_map("brick_n.png"));
        assert!(!name_suggests_normal_map("brick_albedo.png"));
        assert_eq!(TextureImport::name_hint_kind("floor_roughness.jpg"), Some(TextureKind::SingleChannel));
        assert_eq!(TextureImport::name_hint_kind("floor_diff.jpg"), None);
    }

    #[test]
    fn slot_validation_and_autofix() {
        let color = TextureImport { configured: true, ..TextureImport::default() };
        // A colour texture in the normal slot.
        let issues = slot_issues(SlotExpect::Normal, Some(&color));
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].kind, SlotIssueKind::NotNormalMap);
        let mut fixed = color.clone();
        assert!(apply_expectation(&mut fixed, SlotExpect::Normal));
        assert!(slot_issues(SlotExpect::Normal, Some(&fixed)).is_empty());
        assert!(!fixed.srgb);
        // The fixed normal map in a colour slot is flagged and repaired.
        assert_eq!(slot_issues(SlotExpect::Color, Some(&fixed))[0].kind, SlotIssueKind::ShouldBeSrgb);
        apply_expectation(&mut fixed, SlotExpect::Color);
        assert!(slot_issues(SlotExpect::Color, Some(&fixed)).is_empty());
        // sRGB colour in a roughness slot.
        assert_eq!(slot_issues(SlotExpect::Single, Some(&color))[0].kind, SlotIssueKind::ShouldBeLinear);
        let mut lin = color.clone();
        apply_expectation(&mut lin, SlotExpect::Single);
        assert_eq!(lin.kind, TextureKind::SingleChannel);
        assert!(slot_issues(SlotExpect::Single, Some(&lin)).is_empty());
        // Linear Default data in a roughness slot is only a BC4 hint.
        let mut data = color.clone();
        data.srgb = false;
        let hint = slot_issues(SlotExpect::Single, Some(&data));
        assert_eq!(hint.len(), 1);
        assert!(hint[0].info);
        // Unconfigured textures never mismatch: they load per slot.
        assert!(slot_issues(SlotExpect::Normal, Some(&TextureImport::default())).is_empty());
        assert!(slot_issues(SlotExpect::Normal, None).is_empty());
    }

    #[test]
    fn conflicting_slots_are_found_by_class() {
        let uses = vec![
            SlotUse { owner: "a.material".into(), slot: 0 },  // albedo
            SlotUse { owner: "b.material".into(), slot: 13 }, // roughness
            SlotUse { owner: "c.material".into(), slot: 2 },  // orm
        ];
        // Used here as roughness: albedo conflicts, ORM (linear data) doesn't.
        let c = conflicting_uses(SlotExpect::Single, &uses);
        assert_eq!(c.iter().map(|u| u.owner.as_str()).collect::<Vec<_>>(), ["a.material"]);
        assert!(conflicting_uses(SlotExpect::Color, &[SlotUse { owner: "x".into(), slot: 3 }]).is_empty());
    }

    #[test]
    fn duplicate_names_are_unique() {
        let p = Path::new("/p/wood.png");
        assert_eq!(duplicate_path(p, SlotExpect::Single, |_| false), PathBuf::from("/p/wood_linear.png"));
        assert_eq!(
            duplicate_path(p, SlotExpect::Single, |q| q == Path::new("/p/wood_linear.png")),
            PathBuf::from("/p/wood_linear_2.png")
        );
        assert_eq!(duplicate_path(Path::new("a/b"), SlotExpect::Normal, |_| false), PathBuf::from("a/b_normal"));
    }
}
