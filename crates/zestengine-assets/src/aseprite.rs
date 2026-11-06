//! Aseprite importer (2D-7): a pure-Rust parser for `.ase` / `.aseprite`
//! (binary, little-endian) and the conversion to a sprite sheet, a sprite
//! library and animation clips.
//!
//! **Parser choice**: our own (about 600 lines), not the `asefile` crate. The
//! format is a flat chunk stream; owning it keeps the dependency tree small,
//! lets the same code cover tilemap cels, slices, user data and old palettes,
//! and the compositor below needs the raw per-cel data anyway.
//!
//! # Parsed
//!
//! Header (size, colour depth RGBA / grayscale / indexed, layer-opacity flag,
//! transparent index, pixel ratio, grid), frames with durations, chunks: layers
//! (0x2004; normal, group, tilemap; visibility, blend mode, opacity, optional
//! UUID), cels (0x2005; raw, linked, zlib-compressed image, zlib-compressed
//! tilemap), tags (0x2018), palettes (0x2019 and the old 0x0004 / 0x0011),
//! slices (0x2022 with 9-patch centre and pivot), tilesets (0x2023, inline
//! tiles), user data (0x2020 text, attached to the previous layer / cel / tag /
//! slice). Unknown chunks are skipped by size. Truncated or inconsistent files
//! return an error (never panic); every length is bounds-checked.
//!
//! # Flattening
//!
//! [`AseFile::flatten_frame`] composites the visible layers (a hidden group
//! hides its children) in layer order, cels ordered by `layer + z_index` as
//! Aseprite does, honouring layer opacity (only when the file header says it is
//! valid), cel opacity and all 19 blend modes (separable ones per the W3C
//! compositing formulas, the four HSL ones non-separably, plus Addition,
//! Subtract and Divide). Linked cels resolve to their source frame, tilemap
//! cels are rendered from their tileset. Indexed cels use the palette; index
//! `transparent_index` is transparent except on the background layer.
//!
//! # Import
//!
//! [`import`] produces, with canvas-space framing so frames never jitter:
//! * one sheet image with one cell per frame (merged mode) or per
//!   layer-and-frame (individual mode, empty cels skipped), padded and laid
//!   out on a grid, with a `SpriteDef` per cell (pivot = the canvas centre, or
//!   the cel offset in individual mode),
//! * extra sprites for slices (frame 0 pixels, slice pivot, 9-patch border),
//! * a [`SpriteLibraryAsset`] with a category per layer (or the file) and a
//!   label per frame,
//! * one [`AnimClip`] per tag (or one for the whole file) whose
//!   `SpriteSwap` track keys the library hash at cumulative frame times, with
//!   `Step` interpolation; ping-pong and reverse directions are honoured
//!   (Unity ignores them), finite repeat counts are unrolled, a cel user-data
//!   text `event:Name` becomes a clip event. Individual mode adds one track per
//!   layer object plus an `Active` track for frames without a cel.

use anyhow::{Result, bail, ensure};
use flate2::read::ZlibDecoder;
use std::io::Read;

use crate::anim_clip::{AnimClip, AnimTrack, ClipEvent, KeyInterp, Keyframe, TrackProperty};
use crate::sprite::{PixelRect, PivotPreset, SpriteDef};
use crate::sprite_library::{SpriteLibraryAsset, entry_hash};

pub const ASE_MAGIC: u16 = 0xA5E0;
pub const FRAME_MAGIC: u16 = 0xF1FA;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorDepth {
    Rgba,
    Grayscale,
    Indexed,
}

impl ColorDepth {
    pub fn bytes(self) -> usize {
        match self {
            Self::Rgba => 4,
            Self::Grayscale => 2,
            Self::Indexed => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BlendMode {
    #[default]
    Normal,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    ColorDodge,
    ColorBurn,
    HardLight,
    SoftLight,
    Difference,
    Exclusion,
    Hue,
    Saturation,
    Color,
    Luminosity,
    Addition,
    Subtract,
    Divide,
}

impl BlendMode {
    pub fn from_u16(v: u16) -> Self {
        use BlendMode::*;
        [Normal, Multiply, Screen, Overlay, Darken, Lighten, ColorDodge, ColorBurn, HardLight, SoftLight, Difference, Exclusion, Hue, Saturation, Color, Luminosity, Addition, Subtract, Divide]
            .get(v as usize)
            .copied()
            .unwrap_or(Normal)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    Normal,
    Group,
    Tilemap,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AseLayer {
    pub name: String,
    pub kind: LayerKind,
    pub visible: bool,
    pub background: bool,
    /// Nesting depth (0 = top level); the parent is the nearest earlier layer
    /// with a smaller depth.
    pub child_level: u16,
    pub blend: BlendMode,
    pub opacity: u8,
    pub tileset: Option<u32>,
    pub uuid: Option<[u8; 16]>,
    pub user_text: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CelData {
    /// Pixels in the file's colour depth.
    Image { w: u16, h: u16, pixels: Vec<u8> },
    /// Same pixels as the cel of this frame on the same layer.
    Linked(u16),
    Tilemap(TilemapCel),
}

#[derive(Clone, Debug, PartialEq)]
pub struct TilemapCel {
    pub w: u16,
    pub h: u16,
    pub id_mask: u32,
    pub xflip_mask: u32,
    pub yflip_mask: u32,
    pub diag_mask: u32,
    pub tiles: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AseCel {
    pub layer: u16,
    pub x: i16,
    pub y: i16,
    pub opacity: u8,
    pub z_index: i16,
    pub data: CelData,
    pub user_text: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AseFrame {
    pub duration_ms: u16,
    pub cels: Vec<AseCel>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopDirection {
    Forward,
    Reverse,
    PingPong,
    PingPongReverse,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AseTag {
    pub name: String,
    pub from: u16,
    pub to: u16,
    pub direction: LoopDirection,
    /// 0 = loops forever.
    pub repeat: u16,
    pub user_text: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AseSlice {
    pub name: String,
    pub keys: Vec<SliceKey>,
    pub user_text: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SliceKey {
    pub frame: u32,
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
    /// 9-patch centre `(x, y, w, h)` relative to the slice.
    pub center: Option<(i32, i32, u32, u32)>,
    pub pivot: Option<(i32, i32)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AseTileset {
    pub id: u32,
    pub name: String,
    pub tile_w: u16,
    pub tile_h: u16,
    /// RGBA tiles, `tile_w * tile_h * 4` bytes each; tile 0 first.
    pub tiles: Vec<Vec<u8>>,
    pub empty_first: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AseFile {
    pub width: u16,
    pub height: u16,
    pub depth: ColorDepth,
    pub layer_opacity_valid: bool,
    pub transparent_index: u8,
    pub pixel_ratio: (u8, u8),
    pub frames: Vec<AseFrame>,
    pub layers: Vec<AseLayer>,
    pub tags: Vec<AseTag>,
    pub slices: Vec<AseSlice>,
    pub tilesets: Vec<AseTileset>,
    /// RGBA palette.
    pub palette: Vec<[u8; 4]>,
}

// ---------------------------------------------------------------------------
// Byte reader
// ---------------------------------------------------------------------------

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.p.checked_add(n).is_some_and(|e| e <= self.b.len()), "aseprite: truncated data");
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn skip(&mut self, n: usize) -> Result<()> {
        self.take(n).map(|_| ())
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn i16(&mut self) -> Result<i16> {
        Ok(self.u16()? as i16)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u16()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn rest(&mut self) -> &'a [u8] {
        let s = &self.b[self.p..];
        self.p = self.b.len();
        s
    }
}

/// Largest decompressed cel / tileset accepted (a guard against zip bombs).
const MAX_INFLATE: usize = 512 << 20;

fn inflate(data: &[u8], expect: usize) -> Result<Vec<u8>> {
    ensure!(expect <= MAX_INFLATE, "aseprite: image of {expect} bytes is too large");
    let mut out = Vec::with_capacity(expect);
    ZlibDecoder::new(data).take(expect as u64 + 1).read_to_end(&mut out).map_err(|e| anyhow::anyhow!("aseprite: bad zlib stream: {e}"))?;
    ensure!(out.len() >= expect, "aseprite: compressed data holds {} of {expect} bytes", out.len());
    out.truncate(expect);
    Ok(out)
}

enum UserTarget {
    Layer(usize),
    Cel(usize, usize),
    Tag(usize),
    Slice(usize),
}

impl AseFile {
    pub fn parse(bytes: &[u8]) -> Result<AseFile> {
        let mut r = Rd { b: bytes, p: 0 };
        let _file_size = r.u32()?;
        ensure!(r.u16()? == ASE_MAGIC, "aseprite: bad magic number");
        let n_frames = r.u16()? as usize;
        let width = r.u16()?;
        let height = r.u16()?;
        let depth = match r.u16()? {
            32 => ColorDepth::Rgba,
            16 => ColorDepth::Grayscale,
            8 => ColorDepth::Indexed,
            d => bail!("aseprite: unsupported colour depth {d}"),
        };
        let flags = r.u32()?;
        r.skip(2 + 4 + 4)?; // speed, reserved
        let transparent_index = r.u8()?;
        r.skip(3)?;
        let _n_colors = r.u16()?;
        let (pw, ph) = (r.u8()?, r.u8()?);
        r.skip(2 + 2 + 2 + 2 + 84)?; // grid + future
        ensure!(width > 0 && height > 0, "aseprite: empty canvas");
        ensure!(n_frames > 0, "aseprite: no frames");
        let mut f = AseFile {
            width,
            height,
            depth,
            layer_opacity_valid: flags & 1 != 0,
            transparent_index,
            pixel_ratio: (pw.max(1), ph.max(1)),
            frames: Vec::with_capacity(n_frames.min(4096)),
            layers: Vec::new(),
            tags: Vec::new(),
            slices: Vec::new(),
            tilesets: Vec::new(),
            palette: Vec::new(),
        };
        let layer_uuid = flags & 4 != 0;
        let mut user: Option<UserTarget> = None;
        for fi in 0..n_frames {
            let frame_start = r.p;
            let frame_bytes = r.u32()? as usize;
            ensure!(r.u16()? == FRAME_MAGIC, "aseprite: bad frame magic in frame {fi}");
            let old_chunks = r.u16()? as u32;
            let duration = r.u16()?;
            r.skip(2)?;
            let new_chunks = r.u32()?;
            let chunks = if new_chunks != 0 { new_chunks } else { old_chunks };
            ensure!(frame_bytes >= 16 && frame_start + frame_bytes <= bytes.len(), "aseprite: frame {fi} overruns the file");
            let frame_end = frame_start + frame_bytes;
            let mut frame = AseFrame { duration_ms: duration, cels: Vec::new() };
            for _ in 0..chunks {
                if r.p + 6 > frame_end {
                    break;
                }
                let chunk_start = r.p;
                let size = r.u32()? as usize;
                let ty = r.u16()?;
                ensure!(size >= 6 && chunk_start + size <= frame_end, "aseprite: chunk of {size} bytes overruns frame {fi}");
                let mut c = Rd { b: &bytes[chunk_start + 6..chunk_start + size], p: 0 };
                match ty {
                    0x2004 => {
                        let lflags = c.u16()?;
                        let kind = match c.u16()? {
                            0 => LayerKind::Normal,
                            1 => LayerKind::Group,
                            2 => LayerKind::Tilemap,
                            k => bail!("aseprite: unknown layer type {k}"),
                        };
                        let child_level = c.u16()?;
                        c.skip(4)?; // default size
                        let blend = BlendMode::from_u16(c.u16()?);
                        let opacity = c.u8()?;
                        c.skip(3)?;
                        let name = c.string()?;
                        let tileset = if kind == LayerKind::Tilemap { Some(c.u32()?) } else { None };
                        let uuid = if layer_uuid { c.take(16).ok().map(|u| <[u8; 16]>::try_from(u).unwrap()) } else { None };
                        f.layers.push(AseLayer {
                            name,
                            kind,
                            visible: lflags & 1 != 0,
                            background: lflags & 8 != 0,
                            child_level,
                            blend,
                            opacity,
                            tileset,
                            uuid,
                            user_text: String::new(),
                        });
                        user = Some(UserTarget::Layer(f.layers.len() - 1));
                    }
                    0x2005 => {
                        let layer = c.u16()?;
                        let x = c.i16()?;
                        let y = c.i16()?;
                        let opacity = c.u8()?;
                        let ty = c.u16()?;
                        let z_index = c.i16()?;
                        c.skip(5)?;
                        let data = match ty {
                            0 | 2 => {
                                let (w, h) = (c.u16()?, c.u16()?);
                                let n = w as usize * h as usize * depth.bytes();
                                let pixels = if ty == 0 { c.take(n)?.to_vec() } else { inflate(c.rest(), n)? };
                                CelData::Image { w, h, pixels }
                            }
                            1 => CelData::Linked(c.u16()?),
                            3 => {
                                let (w, h) = (c.u16()?, c.u16()?);
                                let bits = c.u16()?;
                                ensure!(matches!(bits, 8 | 16 | 32), "aseprite: tile of {bits} bits");
                                let id_mask = c.u32()?;
                                let xflip_mask = c.u32()?;
                                let yflip_mask = c.u32()?;
                                let diag_mask = c.u32()?;
                                c.skip(10)?;
                                let bpt = bits as usize / 8;
                                let raw = inflate(c.rest(), w as usize * h as usize * bpt)?;
                                let tiles = raw
                                    .chunks_exact(bpt)
                                    .map(|b| match bpt {
                                        1 => b[0] as u32,
                                        2 => u16::from_le_bytes([b[0], b[1]]) as u32,
                                        _ => u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                                    })
                                    .collect();
                                CelData::Tilemap(TilemapCel { w, h, id_mask, xflip_mask, yflip_mask, diag_mask, tiles })
                            }
                            t => bail!("aseprite: unknown cel type {t}"),
                        };
                        frame.cels.push(AseCel { layer, x, y, opacity, z_index, data, user_text: String::new() });
                        user = Some(UserTarget::Cel(fi, frame.cels.len() - 1));
                    }
                    0x2018 => {
                        let n = c.u16()?;
                        c.skip(8)?;
                        for _ in 0..n {
                            let from = c.u16()?;
                            let to = c.u16()?;
                            let dir = match c.u8()? {
                                1 => LoopDirection::Reverse,
                                2 => LoopDirection::PingPong,
                                3 => LoopDirection::PingPongReverse,
                                _ => LoopDirection::Forward,
                            };
                            let repeat = c.u16()?;
                            c.skip(6 + 3 + 1)?;
                            let name = c.string()?;
                            f.tags.push(AseTag { name, from, to, direction: dir, repeat, user_text: String::new() });
                        }
                        let first_tag = f.tags.len() - n as usize;
                        user = if n > 0 { Some(UserTarget::Tag(first_tag)) } else { None };
                    }
                    0x2019 => {
                        let size = c.u32()? as usize;
                        let first = c.u32()? as usize;
                        let last = c.u32()? as usize;
                        c.skip(8)?;
                        ensure!(size <= 65536 && first <= last && last < 65536, "aseprite: bad palette range");
                        if f.palette.len() < size {
                            f.palette.resize(size, [0, 0, 0, 255]);
                        }
                        for i in first..=last {
                            let pf = c.u16()?;
                            let col = c.take(4)?;
                            if i < f.palette.len() {
                                f.palette[i] = [col[0], col[1], col[2], col[3]];
                            }
                            if pf & 1 != 0 {
                                c.string()?;
                            }
                        }
                    }
                    0x0004 | 0x0011 => {
                        // Old palettes: only used when no 0x2019 chunk is present.
                        let scale6 = ty == 0x0011;
                        let packets = c.u16()?;
                        let mut i = 0usize;
                        for _ in 0..packets {
                            i += c.u8()? as usize;
                            let cnt = match c.u8()? {
                                0 => 256,
                                n => n as usize,
                            };
                            for _ in 0..cnt {
                                let rgb = c.take(3)?;
                                let m = |v: u8| if scale6 { (v << 2) | (v >> 4) } else { v };
                                if f.palette.len() <= i {
                                    f.palette.resize(i + 1, [0, 0, 0, 255]);
                                }
                                f.palette[i] = [m(rgb[0]), m(rgb[1]), m(rgb[2]), 255];
                                i += 1;
                            }
                        }
                    }
                    0x2022 => {
                        let n = c.u32()?;
                        let sflags = c.u32()?;
                        c.skip(4)?;
                        let name = c.string()?;
                        let mut keys = Vec::new();
                        for _ in 0..n {
                            let frame = c.u32()?;
                            let (x, y) = (c.i32()?, c.i32()?);
                            let (w, h) = (c.u32()?, c.u32()?);
                            let center = if sflags & 1 != 0 { Some((c.i32()?, c.i32()?, c.u32()?, c.u32()?)) } else { None };
                            let pivot = if sflags & 2 != 0 { Some((c.i32()?, c.i32()?)) } else { None };
                            keys.push(SliceKey { frame, x, y, w, h, center, pivot });
                        }
                        f.slices.push(AseSlice { name, keys, user_text: String::new() });
                        user = Some(UserTarget::Slice(f.slices.len() - 1));
                    }
                    0x2023 => {
                        let id = c.u32()?;
                        let tflags = c.u32()?;
                        let count = c.u32()? as usize;
                        let (tw, th) = (c.u16()?, c.u16()?);
                        c.skip(2 + 14)?;
                        let name = c.string()?;
                        let mut tiles = Vec::new();
                        if tflags & 2 != 0 {
                            let clen = c.u32()? as usize;
                            let tile_bytes = tw as usize * th as usize * depth.bytes();
                            ensure!(count <= 65536 && tile_bytes > 0, "aseprite: bad tileset size");
                            let raw = inflate(c.take(clen)?, tile_bytes * count)?;
                            for t in raw.chunks_exact(tile_bytes) {
                                tiles.push(to_rgba(t, depth, &f.palette, f.transparent_index, false));
                            }
                        }
                        f.tilesets.push(AseTileset { id, name, tile_w: tw, tile_h: th, tiles, empty_first: tflags & 4 != 0 });
                    }
                    0x2020 => {
                        let uflags = c.u32()?;
                        if uflags & 1 != 0 {
                            let text = c.string()?;
                            match user {
                                Some(UserTarget::Layer(i)) => f.layers[i].user_text = text,
                                Some(UserTarget::Cel(fr, i)) => {
                                    if fr == fi {
                                        frame.cels[i].user_text = text;
                                    }
                                }
                                Some(UserTarget::Slice(i)) => f.slices[i].user_text = text,
                                Some(UserTarget::Tag(i)) => {
                                    f.tags[i].user_text = text;
                                    // The next user-data chunk belongs to the next tag.
                                    user = (i + 1 < f.tags.len()).then_some(UserTarget::Tag(i + 1));
                                }
                                None => {}
                            }
                        } else if let Some(UserTarget::Tag(i)) = user {
                            user = (i + 1 < f.tags.len()).then_some(UserTarget::Tag(i + 1));
                        }
                    }
                    _ => {} // colour profile, external files, paths, cel extra: not needed
                }
                r.p = chunk_start + size;
            }
            r.p = frame_end;
            f.frames.push(frame);
        }
        Ok(f)
    }

    /// Whether layer `i` and all its parent groups are visible.
    pub fn layer_visible(&self, i: usize) -> bool {
        let Some(l) = self.layers.get(i) else { return false };
        if !l.visible {
            return false;
        }
        let mut level = l.child_level;
        for j in (0..i).rev() {
            if level == 0 {
                break;
            }
            let p = &self.layers[j];
            if p.child_level < level {
                if p.kind == LayerKind::Group && !p.visible {
                    return false;
                }
                level = p.child_level;
            }
        }
        true
    }

    /// Opacity of a layer: the layer's own times its parent groups', as 0..1.
    /// Layer opacity only counts when the header says it is valid.
    fn layer_alpha(&self, i: usize) -> f32 {
        if !self.layer_opacity_valid {
            return 1.0;
        }
        let mut a = self.layers[i].opacity as f32 / 255.0;
        let mut level = self.layers[i].child_level;
        for j in (0..i).rev() {
            if level == 0 {
                break;
            }
            let p = &self.layers[j];
            if p.child_level < level {
                if p.kind == LayerKind::Group {
                    a *= p.opacity as f32 / 255.0;
                }
                level = p.child_level;
            }
        }
        a
    }

    /// The cel of `layer` in `frame`, following linked cels.
    pub fn cel(&self, frame: usize, layer: usize) -> Option<&AseCel> {
        let mut fr = frame;
        for _ in 0..8 {
            let cel = self.frames.get(fr)?.cels.iter().find(|c| c.layer as usize == layer)?;
            match cel.data {
                CelData::Linked(to) => fr = to as usize,
                _ => return Some(cel),
            }
        }
        None
    }

    /// A cel's pixels as RGBA with its canvas offset: `(x, y, w, h, rgba)`.
    pub fn cel_rgba(&self, frame: usize, layer: usize) -> Option<(i32, i32, u32, u32, Vec<u8>)> {
        let cel = self.cel(frame, layer)?;
        let l = self.layers.get(layer)?;
        match &cel.data {
            CelData::Image { w, h, pixels } => Some((cel.x as i32, cel.y as i32, *w as u32, *h as u32, to_rgba(pixels, self.depth, &self.palette, self.transparent_index, l.background))),
            CelData::Tilemap(t) => {
                let ts = self.tilesets.iter().find(|s| Some(s.id) == l.tileset)?;
                let (tw, th) = (ts.tile_w as usize, ts.tile_h as usize);
                let (w, h) = (t.w as usize * tw, t.h as usize * th);
                let mut out = vec![0u8; w * h * 4];
                for ty in 0..t.h as usize {
                    for tx in 0..t.w as usize {
                        let raw = t.tiles[ty * t.w as usize + tx];
                        let id = (raw & t.id_mask) as usize;
                        let Some(tile) = ts.tiles.get(id) else { continue };
                        if id == 0 && ts.empty_first {
                            continue;
                        }
                        let (fx, fy, fd) = (raw & t.xflip_mask != 0, raw & t.yflip_mask != 0, raw & t.diag_mask != 0);
                        for py in 0..th {
                            for px in 0..tw {
                                let (mut sx, mut sy) = (px, py);
                                if fd && tw == th {
                                    std::mem::swap(&mut sx, &mut sy);
                                }
                                if fx {
                                    sx = tw - 1 - sx;
                                }
                                if fy {
                                    sy = th - 1 - sy;
                                }
                                let s = (sy * tw + sx) * 4;
                                let d = ((ty * th + py) * w + tx * tw + px) * 4;
                                out[d..d + 4].copy_from_slice(&tile[s..s + 4]);
                            }
                        }
                    }
                }
                Some((cel.x as i32, cel.y as i32, w as u32, h as u32, out))
            }
            CelData::Linked(_) => None,
        }
    }

    /// Composite every visible layer of `frame` into a canvas-sized RGBA image
    /// (straight alpha).
    pub fn flatten_frame(&self, frame: usize) -> Vec<u8> {
        self.flatten_frame_where(frame, |_| true)
    }

    /// [`Self::flatten_frame`] over the layers `keep` accepts.
    pub fn flatten_frame_where(&self, frame: usize, keep: impl Fn(usize) -> bool) -> Vec<u8> {
        self.flatten_impl(frame, keep, false)
    }

    /// Flatten with an option to draw hidden layers too (their parents' and
    /// own opacity still apply).
    pub fn flatten_impl(&self, frame: usize, keep: impl Fn(usize) -> bool, include_hidden: bool) -> Vec<u8> {
        let (cw, ch) = (self.width as usize, self.height as usize);
        let mut canvas = vec![0u8; cw * ch * 4];
        let Some(fr) = self.frames.get(frame) else { return canvas };
        let mut order: Vec<usize> = (0..fr.cels.len()).collect();
        // Aseprite: sort by layer index + z-index, then by z-index.
        order.sort_by_key(|&i| (fr.cels[i].layer as i32 + fr.cels[i].z_index as i32, fr.cels[i].z_index));
        for ci in order {
            let cel = &fr.cels[ci];
            let li = cel.layer as usize;
            if li >= self.layers.len() || self.layers[li].kind == LayerKind::Group || !(include_hidden || self.layer_visible(li)) || !keep(li) {
                continue;
            }
            let Some((x0, y0, w, h, rgba)) = self.cel_rgba(frame, li) else { continue };
            let opacity = cel.opacity as f32 / 255.0 * self.layer_alpha(li);
            let mode = self.layers[li].blend;
            for y in 0..h as i32 {
                let cy = y + y0;
                if cy < 0 || cy >= ch as i32 {
                    continue;
                }
                for x in 0..w as i32 {
                    let cx = x + x0;
                    if cx < 0 || cx >= cw as i32 {
                        continue;
                    }
                    let s = (y as usize * w as usize + x as usize) * 4;
                    let d = (cy as usize * cw + cx as usize) * 4;
                    let src = [rgba[s], rgba[s + 1], rgba[s + 2], rgba[s + 3]];
                    if src[3] == 0 {
                        continue;
                    }
                    let dst = [canvas[d], canvas[d + 1], canvas[d + 2], canvas[d + 3]];
                    canvas[d..d + 4].copy_from_slice(&blend_pixel(dst, src, mode, opacity));
                }
            }
        }
        canvas
    }

    /// Duration of `frame` in seconds (a zero duration counts as 100 ms, the
    /// format's default).
    pub fn duration_s(&self, frame: usize) -> f32 {
        let ms = self.frames.get(frame).map(|f| f.duration_ms).unwrap_or(100);
        (if ms == 0 { 100 } else { ms }) as f32 / 1000.0
    }
}

/// Convert raw cel pixels to straight RGBA.
pub fn to_rgba(px: &[u8], depth: ColorDepth, palette: &[[u8; 4]], transparent: u8, background: bool) -> Vec<u8> {
    match depth {
        ColorDepth::Rgba => px.to_vec(),
        ColorDepth::Grayscale => px.as_chunks::<2>().0.iter().flat_map(|p| [p[0], p[0], p[0], p[1]]).collect(),
        ColorDepth::Indexed => px
            .iter()
            .flat_map(|&i| {
                if i == transparent && !background {
                    [0, 0, 0, 0]
                } else {
                    palette.get(i as usize).copied().unwrap_or([0, 0, 0, 255])
                }
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// Blending
// ---------------------------------------------------------------------------

fn lum(c: [f32; 3]) -> f32 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

fn clip_color(c: [f32; 3]) -> [f32; 3] {
    let l = lum(c);
    let n = c[0].min(c[1]).min(c[2]);
    let x = c[0].max(c[1]).max(c[2]);
    let mut c = c;
    if n < 0.0 {
        for v in &mut c {
            *v = l + (*v - l) * l / (l - n);
        }
    }
    if x > 1.0 {
        for v in &mut c {
            *v = l + (*v - l) * (1.0 - l) / (x - l);
        }
    }
    c
}

fn set_lum(c: [f32; 3], l: f32) -> [f32; 3] {
    let d = l - lum(c);
    clip_color([c[0] + d, c[1] + d, c[2] + d])
}

fn sat(c: [f32; 3]) -> f32 {
    c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2])
}

fn set_sat(c: [f32; 3], s: f32) -> [f32; 3] {
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| c[a].total_cmp(&c[b]));
    let (mn, md, mx) = (idx[0], idx[1], idx[2]);
    let mut out = [0.0; 3];
    if c[mx] > c[mn] {
        out[md] = (c[md] - c[mn]) * s / (c[mx] - c[mn]);
        out[mx] = s;
    }
    out
}

fn blend_channel(mode: BlendMode, b: f32, s: f32) -> f32 {
    match mode {
        BlendMode::Multiply => b * s,
        BlendMode::Screen => b + s - b * s,
        BlendMode::Overlay => blend_channel(BlendMode::HardLight, s, b),
        BlendMode::Darken => b.min(s),
        BlendMode::Lighten => b.max(s),
        BlendMode::ColorDodge => {
            if b == 0.0 {
                0.0
            } else if s >= 1.0 {
                1.0
            } else {
                (b / (1.0 - s)).min(1.0)
            }
        }
        BlendMode::ColorBurn => {
            if b >= 1.0 {
                1.0
            } else if s <= 0.0 {
                0.0
            } else {
                1.0 - ((1.0 - b) / s).min(1.0)
            }
        }
        BlendMode::HardLight => {
            if s <= 0.5 {
                b * 2.0 * s
            } else {
                let t = 2.0 * s - 1.0;
                b + t - b * t
            }
        }
        BlendMode::SoftLight => {
            if s <= 0.5 {
                b - (1.0 - 2.0 * s) * b * (1.0 - b)
            } else {
                let d = if b <= 0.25 { ((16.0 * b - 12.0) * b + 4.0) * b } else { b.sqrt() };
                b + (2.0 * s - 1.0) * (d - b)
            }
        }
        BlendMode::Difference => (b - s).abs(),
        BlendMode::Exclusion => b + s - 2.0 * b * s,
        BlendMode::Addition => (b + s).min(1.0),
        BlendMode::Subtract => (b - s).max(0.0),
        BlendMode::Divide => {
            if b == 0.0 {
                0.0
            } else if b >= s {
                1.0
            } else {
                b / s
            }
        }
        _ => s,
    }
}

fn blend_color(mode: BlendMode, b: [f32; 3], s: [f32; 3]) -> [f32; 3] {
    match mode {
        BlendMode::Hue => set_lum(set_sat(s, sat(b)), lum(b)),
        BlendMode::Saturation => set_lum(set_sat(b, sat(s)), lum(b)),
        BlendMode::Color => set_lum(s, lum(b)),
        BlendMode::Luminosity => set_lum(b, lum(s)),
        m => [blend_channel(m, b[0], s[0]), blend_channel(m, b[1], s[1]), blend_channel(m, b[2], s[2])],
    }
}

/// Composite one `src` pixel over `dst` (both straight RGBA8) with a blend
/// mode and an extra opacity 0..1.
pub fn blend_pixel(dst: [u8; 4], src: [u8; 4], mode: BlendMode, opacity: f32) -> [u8; 4] {
    let f = |v: u8| v as f32 / 255.0;
    let (ab, asrc) = (f(dst[3]), f(src[3]) * opacity.clamp(0.0, 1.0));
    if asrc <= 0.0 {
        return dst;
    }
    let cb = [f(dst[0]), f(dst[1]), f(dst[2])];
    let cs = [f(src[0]), f(src[1]), f(src[2])];
    let ar = asrc + ab * (1.0 - asrc);
    let blended = if ab > 0.0 { blend_color(mode, cb, cs) } else { cs };
    let mut out = [0u8; 4];
    for k in 0..3 {
        // W3C source-over with a blend function: backdrop weight, source weight, mixed weight.
        let c = ((1.0 - asrc) * ab * cb[k] + asrc * ((1.0 - ab) * cs[k] + ab * blended[k])) / ar;
        out[k] = (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
    out[3] = (ar.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    out
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ImportOptions {
    /// Name stem for sprites and the library category.
    pub name: String,
    /// One sprite per frame with all layers flattened (true), or one per layer cel.
    pub merge_layers: bool,
    pub include_hidden: bool,
    /// Pixels of padding around each sheet cell.
    pub padding: u32,
    pub ppu: f32,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self { name: "sprite".into(), merge_layers: true, include_hidden: false, padding: 1, ppu: 100.0 }
    }
}

/// Everything an import produces; the caller writes the PNG and sidecar files.
#[derive(Clone, Debug)]
pub struct Imported {
    pub sheet_w: u32,
    pub sheet_h: u32,
    /// RGBA8, straight alpha.
    pub sheet: Vec<u8>,
    /// Frame (or layer-cel) sprites first, slice sprites after.
    pub sprites: Vec<SpriteDef>,
    pub library: SpriteLibraryAsset,
    /// `(name, clip)`: one per tag, or `(stem, clip)` for the whole file.
    pub clips: Vec<(String, AnimClip)>,
    /// Palette as RGBA (empty for RGBA files without one).
    pub palette: Vec<[u8; 4]>,
}

struct Cell {
    name: String,
    w: u32,
    h: u32,
    rgba: Vec<u8>,
    /// Pivot in cell pixels, from the top-left.
    pivot_px: [f32; 2],
    border: [u32; 4],
}

/// Frame index of each step of a tag's playback (direction and repeats applied).
pub fn tag_frames(tag: &AseTag) -> Vec<u16> {
    let (a, b) = (tag.from.min(tag.to), tag.from.max(tag.to));
    let fwd: Vec<u16> = (a..=b).collect();
    let rev: Vec<u16> = (a..=b).rev().collect();
    let one_pass = |dir: LoopDirection| -> Vec<u16> {
        match dir {
            LoopDirection::Forward => fwd.clone(),
            LoopDirection::Reverse => rev.clone(),
            LoopDirection::PingPong => {
                let mut v = fwd.clone();
                if b > a + 1 {
                    v.extend((a + 1..b).rev());
                }
                v
            }
            LoopDirection::PingPongReverse => {
                let mut v = rev.clone();
                if b > a + 1 {
                    v.extend(a + 1..b);
                }
                v
            }
        }
    };
    let pass = one_pass(tag.direction);
    if tag.repeat > 1 {
        // A finite repeat plays n passes then stops (ping-pong's extra turn-around frames included).
        (0..tag.repeat.min(64)).flat_map(|_| pass.clone()).collect()
    } else {
        pass
    }
}

/// Build the sheet, sprites, library and clips of a parsed file.
pub fn import(file: &AseFile, opt: &ImportOptions) -> Result<Imported> {
    let (cw, ch) = (file.width as u32, file.height as u32);
    let mut cells: Vec<Cell> = Vec::new();
    // cell index of (layer or usize::MAX for merged, frame); None = no cel.
    let mut cell_of: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
    let visible = |li: usize| opt.include_hidden && file.layers[li].kind != LayerKind::Group || file.layer_visible(li);
    if opt.merge_layers {
        for fi in 0..file.frames.len() {
            let rgba = file.flatten_impl(fi, |_| true, opt.include_hidden);
            cell_of.insert((usize::MAX, fi), cells.len());
            cells.push(Cell { name: format!("{}_{}", opt.name, fi), w: cw, h: ch, rgba, pivot_px: [cw as f32 / 2.0, ch as f32 / 2.0], border: [0; 4] });
        }
    } else {
        for (li, l) in file.layers.iter().enumerate() {
            if l.kind == LayerKind::Group || !visible(li) {
                continue;
            }
            for fi in 0..file.frames.len() {
                let Some((x0, y0, w, h, rgba)) = file.cel_rgba(fi, li) else { continue };
                if w == 0 || h == 0 {
                    continue;
                }
                // Bake cel and layer opacity into the pixels (the format has no per-sprite opacity).
                let a = file.cel(fi, li).map_or(1.0, |c| c.opacity as f32 / 255.0) * file.layer_alpha(li);
                let mut rgba = rgba;
                if a < 0.999 {
                    for p in rgba.as_chunks_mut::<4>().0 {
                        p[3] = (p[3] as f32 * a + 0.5) as u8;
                    }
                }
                // Canvas-space pivot: the canvas centre, expressed in this cel's pixels.
                let pivot_px = [cw as f32 / 2.0 - x0 as f32, ch as f32 / 2.0 - y0 as f32];
                cell_of.insert((li, fi), cells.len());
                cells.push(Cell { name: format!("{}_{}_{}", opt.name, l.name, fi), w, h, rgba, pivot_px, border: [0; 4] });
            }
        }
    }
    let n_frame_cells = cells.len();
    for s in &file.slices {
        let Some(k) = s.keys.first() else { continue };
        let (w, h) = (k.w.max(1), k.h.max(1));
        let flat = file.flatten_frame(k.frame.min(file.frames.len() as u32 - 1) as usize);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let (sx, sy) = (k.x + x, k.y + y);
                if sx >= 0 && sy >= 0 && (sx as u32) < cw && (sy as u32) < ch {
                    let si = ((sy as u32 * cw + sx as u32) * 4) as usize;
                    let di = ((y as u32 * w + x as u32) * 4) as usize;
                    rgba[di..di + 4].copy_from_slice(&flat[si..si + 4]);
                }
            }
        }
        let pivot = k.pivot.map(|(px, py)| [px as f32, py as f32]).unwrap_or([w as f32 / 2.0, h as f32 / 2.0]);
        // 9-patch centre -> left, bottom, right, top borders.
        let border = k.center.map_or([0; 4], |(cx, cy, cw2, ch2)| {
            let l = cx.max(0) as u32;
            let t = cy.max(0) as u32;
            let r = (w as i32 - cx - cw2 as i32).max(0) as u32;
            let b = (h as i32 - cy - ch2 as i32).max(0) as u32;
            [l, b, r, t]
        });
        cells.push(Cell { name: format!("{}_{}", opt.name, s.name), w, h, rgba, pivot_px: pivot, border });
    }
    ensure!(!cells.is_empty(), "aseprite: nothing to import (all layers hidden or empty)");
    // Grid layout.
    let pad = opt.padding;
    let cell_w = cells.iter().map(|c| c.w).max().unwrap() + 2 * pad;
    let cell_h = cells.iter().map(|c| c.h).max().unwrap() + 2 * pad;
    let cols = ((cells.len() as f32).sqrt().ceil() as u32).max(1);
    let rows = (cells.len() as u32).div_ceil(cols);
    let (sheet_w, sheet_h) = (cols * cell_w, rows * cell_h);
    ensure!((sheet_w as u64 * sheet_h as u64) <= (1 << 28), "aseprite: sheet of {sheet_w}x{sheet_h} is too large");
    let mut sheet = vec![0u8; (sheet_w * sheet_h * 4) as usize];
    let mut sprites = Vec::with_capacity(cells.len());
    for (i, c) in cells.iter().enumerate() {
        let (gx, gy) = (i as u32 % cols * cell_w + pad, i as u32 / cols * cell_h + pad);
        for y in 0..c.h {
            let s = (y * c.w * 4) as usize;
            let d = (((gy + y) * sheet_w + gx) * 4) as usize;
            sheet[d..d + (c.w * 4) as usize].copy_from_slice(&c.rgba[s..s + (c.w * 4) as usize]);
        }
        let mut def = SpriteDef::named(&c.name, PixelRect::new(gx, gy, c.w, c.h));
        def.pivot_preset = PivotPreset::Custom;
        // SpriteDef pivots are normalised from the bottom-left.
        def.pivot = [c.pivot_px[0] / c.w as f32, 1.0 - c.pivot_px[1] / c.h as f32];
        def.border = c.border;
        def.ppu = Some(opt.ppu);
        sprites.push(def);
    }
    // Library: category per layer (individual) or the file (merged).
    let mut library = SpriteLibraryAsset::default();
    let mut label_of: std::collections::HashMap<(usize, usize), (String, String)> = std::collections::HashMap::new();
    for (&(li, fi), &ci) in &cell_of {
        let cat = if li == usize::MAX { opt.name.clone() } else { file.layers[li].name.clone() };
        label_of.insert((li, fi), (cat, format!("Frame {fi}")));
        let _ = ci;
    }
    // The library holds the sprite reference text once the caller knows the
    // sheet path; here the sprite *name* stands in and `fill_library` swaps it.
    let mut keys: Vec<_> = cell_of.iter().collect();
    keys.sort();
    for (&(li, fi), &ci) in keys {
        let (cat, label) = &label_of[&(li, fi)];
        library.set(cat, label, &sprites[ci].name);
    }
    // Clips.
    let tag_list: Vec<(String, Vec<u16>, bool)> = if file.tags.is_empty() {
        vec![(opt.name.clone(), (0..file.frames.len() as u16).collect(), true)]
    } else {
        file.tags.iter().map(|t| (t.name.clone(), tag_frames(t), t.repeat == 0)).collect()
    };
    let mut clips = Vec::new();
    for (name, frames, looping) in tag_list {
        let frames: Vec<u16> = frames.into_iter().filter(|&f| (f as usize) < file.frames.len()).collect();
        if frames.is_empty() {
            continue;
        }
        let mut clip = AnimClip { looping, frame_rate: 60.0, ..Default::default() };
        let mut tracks: Vec<(usize, AnimTrack, Option<AnimTrack>)> = Vec::new();
        let layers: Vec<usize> = if opt.merge_layers {
            vec![usize::MAX]
        } else {
            (0..file.layers.len()).filter(|&li| file.layers[li].kind != LayerKind::Group && visible(li)).collect()
        };
        for li in layers {
            let target = if li == usize::MAX { String::new() } else { file.layers[li].name.clone() };
            let swap = AnimTrack::new(&target, TrackProperty::SpriteSwap);
            let active = (li != usize::MAX).then(|| AnimTrack::new(&target, TrackProperty::Active));
            tracks.push((li, swap, active));
        }
        let mut t = 0.0f32;
        for &fi in &frames {
            let fi = fi as usize;
            for (li, swap, active) in tracks.iter_mut() {
                match cell_of.get(&(*li, fi)) {
                    Some(_) => {
                        let (cat, label) = &label_of[&(*li, fi)];
                        swap.keys.push(Keyframe { time: t, value: [entry_hash(cat, label) as f32, 0.0, 0.0, 0.0], interp: KeyInterp::Step, ..Default::default() });
                        if let Some(a) = active {
                            a.keys.push(Keyframe { time: t, value: [1.0, 0.0, 0.0, 0.0], interp: KeyInterp::Step, ..Default::default() });
                        }
                    }
                    None => {
                        if let Some(a) = active {
                            a.keys.push(Keyframe { time: t, value: [0.0, 0.0, 0.0, 0.0], interp: KeyInterp::Step, ..Default::default() });
                        }
                    }
                }
            }
            for cel in &file.frames[fi].cels {
                if let Some(ev) = cel.user_text.strip_prefix("event:") {
                    let ev = ev.trim();
                    if !ev.is_empty() && !clip.events.iter().any(|e: &ClipEvent| e.name == ev && (e.time - t).abs() < 1e-6) {
                        clip.events.push(ClipEvent { time: t, name: ev.to_owned() });
                    }
                }
            }
            t += file.duration_s(fi);
        }
        clip.length = t;
        for (_, swap, active) in tracks {
            if !swap.keys.is_empty() {
                clip.tracks.push(swap);
            }
            if let Some(a) = active
                && a.keys.iter().any(|k| k.value[0] < 0.5)
            {
                clip.tracks.push(a);
            }
        }
        clips.push((name, clip));
    }
    let _ = n_frame_cells;
    Ok(Imported { sheet_w, sheet_h, sheet, sprites, library, clips, palette: file.palette.clone() })
}

impl Imported {
    /// Replace the sprite *names* the library holds by `"<sheet path>#<id>"`
    /// reference texts (call after the sprite ids are assigned).
    pub fn bind_library(&mut self, sheet_path: &str) {
        let by_name: std::collections::HashMap<&str, u64> = self.sprites.iter().map(|s| (s.name.as_str(), s.id.0)).collect();
        for c in &mut self.library.categories {
            for e in &mut c.entries {
                if let Some(id) = by_name.get(e.sprite.as_str()) {
                    e.sprite = crate::sprite::sprite_ref_string(sheet_path, crate::sub_asset::SubAssetId(*id));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: synthesized files
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod synth {
    //! A tiny writer for `.aseprite` files used only by tests.
    use super::*;
    use flate2::{Compression, write::ZlibEncoder};
    use std::io::Write;

    pub fn w16(v: &mut Vec<u8>, x: u16) {
        v.extend(x.to_le_bytes());
    }
    pub fn w32(v: &mut Vec<u8>, x: u32) {
        v.extend(x.to_le_bytes());
    }
    pub fn wstr(v: &mut Vec<u8>, s: &str) {
        w16(v, s.len() as u16);
        v.extend(s.as_bytes());
    }
    pub fn zlib(data: &[u8]) -> Vec<u8> {
        let mut e = ZlibEncoder::new(Vec::new(), Compression::fast());
        e.write_all(data).unwrap();
        e.finish().unwrap()
    }
    pub fn chunk(ty: u16, body: &[u8]) -> Vec<u8> {
        let mut c = Vec::new();
        w32(&mut c, body.len() as u32 + 6);
        w16(&mut c, ty);
        c.extend(body);
        c
    }
    pub fn layer(name: &str, kind: u16, level: u16, flags: u16, blend: u16, opacity: u8, tileset: Option<u32>) -> Vec<u8> {
        let mut b = Vec::new();
        w16(&mut b, flags);
        w16(&mut b, kind);
        w16(&mut b, level);
        w16(&mut b, 0);
        w16(&mut b, 0);
        w16(&mut b, blend);
        b.push(opacity);
        b.extend([0, 0, 0]);
        wstr(&mut b, name);
        if let Some(t) = tileset {
            w32(&mut b, t);
        }
        chunk(0x2004, &b)
    }
    pub enum Cel<'a> {
        Raw(u16, u16, &'a [u8]),
        Linked(u16),
        Zlib(u16, u16, &'a [u8]),
        Tilemap { w: u16, h: u16, tiles: Vec<u8> },
    }
    pub fn cel(layer: u16, x: i16, y: i16, opacity: u8, z: i16, c: Cel) -> Vec<u8> {
        let mut b = Vec::new();
        w16(&mut b, layer);
        w16(&mut b, x as u16);
        w16(&mut b, y as u16);
        b.push(opacity);
        let ty = match &c {
            Cel::Raw(..) => 0,
            Cel::Linked(_) => 1,
            Cel::Zlib(..) => 2,
            Cel::Tilemap { .. } => 3,
        };
        w16(&mut b, ty);
        w16(&mut b, z as u16);
        b.extend([0; 5]);
        match c {
            Cel::Raw(w, h, px) => {
                w16(&mut b, w);
                w16(&mut b, h);
                b.extend(px);
            }
            Cel::Linked(f) => w16(&mut b, f),
            Cel::Zlib(w, h, px) => {
                w16(&mut b, w);
                w16(&mut b, h);
                b.extend(zlib(px));
            }
            Cel::Tilemap { w, h, tiles } => {
                w16(&mut b, w);
                w16(&mut b, h);
                w16(&mut b, 32);
                w32(&mut b, 0x1fff_ffff);
                w32(&mut b, 0x2000_0000);
                w32(&mut b, 0x4000_0000);
                w32(&mut b, 0x8000_0000);
                b.extend([0; 10]);
                b.extend(zlib(&tiles));
            }
        }
        chunk(0x2005, &b)
    }
    pub fn tags(list: &[(&str, u16, u16, u8, u16)]) -> Vec<u8> {
        let mut b = Vec::new();
        w16(&mut b, list.len() as u16);
        b.extend([0; 8]);
        for (name, from, to, dir, rep) in list {
            w16(&mut b, *from);
            w16(&mut b, *to);
            b.push(*dir);
            w16(&mut b, *rep);
            b.extend([0; 6]);
            b.extend([0, 0, 0, 0]);
            wstr(&mut b, name);
        }
        chunk(0x2018, &b)
    }
    pub fn user_text(t: &str) -> Vec<u8> {
        let mut b = Vec::new();
        w32(&mut b, 1);
        wstr(&mut b, t);
        chunk(0x2020, &b)
    }
    pub fn palette(cols: &[[u8; 4]]) -> Vec<u8> {
        let mut b = Vec::new();
        w32(&mut b, cols.len() as u32);
        w32(&mut b, 0);
        w32(&mut b, cols.len() as u32 - 1);
        b.extend([0; 8]);
        for c in cols {
            w16(&mut b, 0);
            b.extend(c);
        }
        chunk(0x2019, &b)
    }
    pub fn slice(name: &str, key: (u32, i32, i32, u32, u32), center: Option<(i32, i32, u32, u32)>, pivot: Option<(i32, i32)>) -> Vec<u8> {
        let mut b = Vec::new();
        w32(&mut b, 1);
        w32(&mut b, center.is_some() as u32 | (pivot.is_some() as u32) << 1);
        w32(&mut b, 0);
        wstr(&mut b, name);
        w32(&mut b, key.0);
        w32(&mut b, key.1 as u32);
        w32(&mut b, key.2 as u32);
        w32(&mut b, key.3);
        w32(&mut b, key.4);
        if let Some((x, y, w, h)) = center {
            w32(&mut b, x as u32);
            w32(&mut b, y as u32);
            w32(&mut b, w);
            w32(&mut b, h);
        }
        if let Some((x, y)) = pivot {
            w32(&mut b, x as u32);
            w32(&mut b, y as u32);
        }
        chunk(0x2022, &b)
    }
    pub fn tileset(id: u32, tw: u16, th: u16, tiles_rgba: &[u8], count: u32) -> Vec<u8> {
        let mut b = Vec::new();
        w32(&mut b, id);
        w32(&mut b, 2);
        w32(&mut b, count);
        w16(&mut b, tw);
        w16(&mut b, th);
        w16(&mut b, 1);
        b.extend([0; 14]);
        wstr(&mut b, "ts");
        let z = zlib(tiles_rgba);
        w32(&mut b, z.len() as u32);
        b.extend(z);
        chunk(0x2023, &b)
    }
    /// frames = (duration, chunks).
    pub fn file(w: u16, h: u16, depth: u16, header_flags: u32, frames: &[(u16, Vec<Vec<u8>>)]) -> Vec<u8> {
        let mut body = Vec::new();
        for (dur, chunks) in frames {
            let mut f: Vec<u8> = Vec::new();
            for c in chunks {
                f.extend(c);
            }
            let mut fh = Vec::new();
            w32(&mut fh, 16 + f.len() as u32);
            w16(&mut fh, FRAME_MAGIC);
            w16(&mut fh, chunks.len() as u16);
            w16(&mut fh, *dur);
            fh.extend([0, 0]);
            w32(&mut fh, 0);
            body.extend(fh);
            body.extend(f);
        }
        let mut h128 = Vec::new();
        w32(&mut h128, 128 + body.len() as u32);
        w16(&mut h128, ASE_MAGIC);
        w16(&mut h128, frames.len() as u16);
        w16(&mut h128, w);
        w16(&mut h128, h);
        w16(&mut h128, depth);
        w32(&mut h128, header_flags);
        w16(&mut h128, 100);
        w32(&mut h128, 0);
        w32(&mut h128, 0);
        h128.push(0);
        h128.extend([0, 0, 0]);
        w16(&mut h128, 0);
        h128.extend([1, 1]);
        h128.extend([0; 8]);
        h128.extend(vec![0; 84]);
        assert_eq!(h128.len(), 128);
        h128.extend(body);
        h128
    }
}

#[cfg(test)]
mod tests {
    use super::synth::*;
    use super::*;

    fn solid(w: usize, h: usize, c: [u8; 4]) -> Vec<u8> {
        (0..w * h).flat_map(|_| c).collect()
    }

    #[test]
    fn every_cel_type_parses_and_flattens() {
        let red = solid(2, 2, [255, 0, 0, 255]);
        let green = solid(2, 2, [0, 255, 0, 255]);
        let blue = solid(2, 2, [0, 0, 255, 255]);
        // Tileset: tile 0 empty, tile 1 solid white, 2x2 tiles.
        let mut tiles = vec![0u8; 2 * 2 * 4];
        tiles.extend(solid(2, 2, [255, 255, 255, 255]));
        let tmap: Vec<u8> = [0u32, 1, 1, 0].iter().flat_map(|t| t.to_le_bytes()).collect();
        let f0 = vec![
            layer("raw", 0, 0, 1, 0, 255, None),
            layer("zip", 0, 0, 1, 0, 255, None),
            layer("tiles", 2, 0, 1, 0, 255, Some(0)),
            tileset(0, 2, 2, &tiles, 2),
            cel(0, 0, 0, 255, 0, Cel::Raw(2, 2, &red)),
            cel(1, 2, 0, 255, 0, Cel::Zlib(2, 2, &green)),
            cel(2, 0, 2, 255, 0, Cel::Tilemap { w: 2, h: 2, tiles: tmap }),
        ];
        let f1 = vec![cel(0, 2, 2, 255, 0, Cel::Zlib(2, 2, &blue)), cel(1, 0, 0, 255, 0, Cel::Linked(0))];
        let bytes = file(8, 8, 32, 1, &[(100, f0), (200, f1)]);
        let a = AseFile::parse(&bytes).unwrap();
        assert_eq!((a.width, a.height, a.frames.len(), a.layers.len()), (8, 8, 2, 3));
        assert_eq!(a.layers[2].kind, LayerKind::Tilemap);
        assert_eq!(a.tilesets[0].tiles.len(), 2);
        let px = |img: &[u8], x: usize, y: usize| -> [u8; 4] { img[(y * 8 + x) * 4..][..4].try_into().unwrap() };
        let f0 = a.flatten_frame(0);
        assert_eq!(px(&f0, 0, 0), [255, 0, 0, 255], "raw");
        assert_eq!(px(&f0, 3, 1), [0, 255, 0, 255], "compressed");
        // Tilemap at (0,2): tiles [0,1;1,0], 2x2 pixels each -> solid at tile (1,0) and (0,1).
        assert_eq!(px(&f0, 2, 2), [255, 255, 255, 255]);
        assert_eq!(px(&f0, 0, 4), [255, 255, 255, 255]);
        assert_eq!(px(&f0, 0, 2)[3], 0, "tile 0 is empty");
        let f1 = a.flatten_frame(1);
        assert_eq!(px(&f1, 3, 3), [0, 0, 255, 255]);
        assert_eq!(px(&f1, 3, 1), [0, 255, 0, 255], "a linked cel shows its source cel");
        assert_eq!(a.duration_s(1), 0.2);
    }

    #[test]
    fn indexed_and_grayscale_depths() {
        let pal = palette(&[[0, 0, 0, 0], [10, 20, 30, 255], [200, 100, 50, 255]]);
        let idx = [0u8, 1, 2, 1];
        let bytes = file(2, 2, 8, 1, &[(50, vec![pal, layer("l", 0, 0, 1, 0, 255, None), cel(0, 0, 0, 255, 0, Cel::Raw(2, 2, &idx))])]);
        let a = AseFile::parse(&bytes).unwrap();
        let f = a.flatten_frame(0);
        assert_eq!(&f[0..4], &[0, 0, 0, 0], "transparent index");
        assert_eq!(&f[4..8], &[10, 20, 30, 255]);
        assert_eq!(&f[8..12], &[200, 100, 50, 255]);
        // The background layer keeps index 0 opaque.
        let bytes = file(2, 2, 8, 1, &[(50, vec![palette(&[[1, 2, 3, 255], [9, 9, 9, 255]]), layer("bg", 0, 0, 1 | 8, 0, 255, None), cel(0, 0, 0, 255, 0, Cel::Raw(2, 2, &[0, 1, 0, 1]))])]);
        assert_eq!(&AseFile::parse(&bytes).unwrap().flatten_frame(0)[0..4], &[1, 2, 3, 255]);
        // Grayscale: value, alpha.
        let bytes = file(2, 1, 16, 1, &[(50, vec![layer("g", 0, 0, 1, 0, 255, None), cel(0, 0, 0, 255, 0, Cel::Raw(2, 1, &[100, 255, 7, 128]))])]);
        let f = AseFile::parse(&bytes).unwrap().flatten_frame(0);
        assert_eq!(&f[0..4], &[100, 100, 100, 255]);
        assert_eq!(&f[4..8], &[7, 7, 7, 128]);
    }

    #[test]
    fn blend_modes_flatten() {
        let base = solid(1, 1, [200, 100, 50, 255]);
        let top = solid(1, 1, [100, 200, 150, 255]);
        let mk = |mode: u16| {
            let bytes = file(1, 1, 32, 1, &[(100, vec![layer("a", 0, 0, 1, 0, 255, None), layer("b", 0, 0, 1, mode, 255, None), cel(0, 0, 0, 255, 0, Cel::Raw(1, 1, &base)), cel(1, 0, 0, 255, 0, Cel::Raw(1, 1, &top))])]);
            let f = AseFile::parse(&bytes).unwrap().flatten_frame(0);
            [f[0], f[1], f[2], f[3]]
        };
        assert_eq!(mk(0), [100, 200, 150, 255], "normal covers");
        let m = mk(1);
        assert_eq!(m[0] as i32, (200.0f32 * 100.0 / 255.0).round() as i32, "multiply");
        assert_eq!(mk(2)[0], (255.0f32 - (255.0 - 200.0) * (255.0 - 100.0) / 255.0).round() as u8, "screen");
        assert_eq!(mk(4), [100, 100, 50, 255], "darken");
        assert_eq!(mk(5), [200, 200, 150, 255], "lighten");
        assert_eq!(mk(10), [100, 100, 100, 255], "difference");
        assert_eq!(mk(16), [255, 255, 200, 255], "addition clamps");
        assert_eq!(mk(17), [100, 0, 0, 255], "subtract clamps");
        // All 19 modes produce a valid opaque pixel; non-separable ones keep luminosity / hue as defined.
        for mode in 0..19 {
            assert_eq!(mk(mode)[3], 255, "mode {mode}");
        }
        let lum_of = |c: [u8; 4]| 0.3 * c[0] as f32 + 0.59 * c[1] as f32 + 0.11 * c[2] as f32;
        let tl = lum_of([100, 200, 150, 255]);
        assert!((lum_of(mk(15)) - tl).abs() < 6.0, "luminosity mode takes the source luminance: {} vs {tl}", lum_of(mk(15)));
        // Over a transparent backdrop every mode is just the source.
        assert_eq!(blend_pixel([0, 0, 0, 0], [10, 20, 30, 255], BlendMode::Multiply, 1.0), [10, 20, 30, 255]);
    }

    #[test]
    fn opacity_visibility_and_groups() {
        let red = solid(1, 1, [255, 0, 0, 255]);
        let layers = |l: Vec<Vec<u8>>, hdr: u32| {
            let mut ch = l;
            ch.push(cel(1, 0, 0, 255, 0, Cel::Raw(1, 1, &red)));
            AseFile::parse(&file(1, 1, 32, hdr, &[(100, ch)])).unwrap()
        };
        // Layer opacity applies only with the header flag.
        let a = layers(vec![layer("g", 1, 0, 1, 0, 255, None), layer("c", 0, 1, 1, 0, 128, None)], 1);
        assert_eq!(a.flatten_frame(0)[3], 128);
        let a = layers(vec![layer("g", 1, 0, 1, 0, 255, None), layer("c", 0, 1, 1, 0, 128, None)], 0);
        assert_eq!(a.flatten_frame(0)[3], 255, "opacity ignored without the flag");
        // A hidden group hides its children.
        let a = layers(vec![layer("g", 1, 0, 0, 0, 255, None), layer("c", 0, 1, 1, 0, 255, None)], 1);
        assert!(!a.layer_visible(1));
        assert_eq!(a.flatten_frame(0)[3], 0);
        // Group opacity multiplies.
        let a = layers(vec![layer("g", 1, 0, 1, 0, 128, None), layer("c", 0, 1, 1, 0, 128, None)], 1);
        let al = a.flatten_frame(0)[3];
        assert!((al as i32 - 64).abs() <= 1, "{al}");
        // Cel opacity.
        let a = AseFile::parse(&file(1, 1, 32, 1, &[(100, vec![layer("c", 0, 0, 1, 0, 255, None), cel(0, 0, 0, 100, 0, Cel::Raw(1, 1, &red))])])).unwrap();
        assert_eq!(a.flatten_frame(0)[3], 100);
    }

    #[test]
    fn z_index_reorders_cels() {
        let red = solid(1, 1, [255, 0, 0, 255]);
        let blue = solid(1, 1, [0, 0, 255, 255]);
        let mk = |z0: i16| {
            let b = file(1, 1, 32, 1, &[(100, vec![layer("lo", 0, 0, 1, 0, 255, None), layer("hi", 0, 0, 1, 0, 255, None), cel(0, 0, 0, 255, z0, Cel::Raw(1, 1, &red)), cel(1, 0, 0, 255, 0, Cel::Raw(1, 1, &blue))])]);
            AseFile::parse(&b).unwrap().flatten_frame(0)
        };
        assert_eq!(&mk(0)[0..4], &[0, 0, 255, 255]);
        assert_eq!(&mk(2)[0..4], &[255, 0, 0, 255], "z-index 2 lifts the lower layer's cel above");
    }

    #[test]
    fn tags_durations_slices_and_user_data() {
        let red = solid(4, 4, [255, 0, 0, 255]);
        let frame = |ev: Option<&str>| {
            let mut v = vec![cel(0, 0, 0, 255, 0, Cel::Raw(4, 4, &red))];
            if let Some(e) = ev {
                v.push(user_text(e));
            }
            v
        };
        let mut f0 = vec![layer("l", 0, 0, 1, 0, 255, None)];
        f0.extend(frame(None));
        f0.push(tags(&[("walk", 0, 2, 0, 0), ("hit", 1, 3, 2, 0), ("once", 0, 1, 1, 2)]));
        f0.push(slice("icon", (0, 1, 1, 2, 2), Some((1, 1, 0, 0)), Some((1, 1))));
        let bytes = file(4, 4, 32, 1, &[(100, f0), (50, frame(Some("event:Step"))), (0, frame(None)), (250, frame(None))]);
        let a = AseFile::parse(&bytes).unwrap();
        assert_eq!(a.tags.len(), 3);
        assert_eq!(a.tags[1].direction, LoopDirection::PingPong);
        assert_eq!(a.tags[2].repeat, 2);
        assert_eq!(a.frames[1].cels[0].user_text, "event:Step");
        assert_eq!(a.duration_s(2), 0.1, "zero duration reads as the default 100 ms");
        assert_eq!(a.slices[0].keys[0].pivot, Some((1, 1)));
        assert_eq!(tag_frames(&a.tags[0]), vec![0, 1, 2]);
        assert_eq!(tag_frames(&a.tags[1]), vec![1, 2, 3, 2], "ping-pong turns around without repeating the ends");
        assert_eq!(tag_frames(&a.tags[2]), vec![1, 0, 1, 0], "reverse, played twice");
        assert_eq!(tag_frames(&AseTag { name: String::new(), from: 0, to: 2, direction: LoopDirection::PingPongReverse, repeat: 0, user_text: String::new() }), vec![2, 1, 0, 1]);

        let imp = import(&a, &ImportOptions { name: "hero".into(), ..Default::default() }).unwrap();
        // 4 frames + 1 slice sprite.
        assert_eq!(imp.sprites.len(), 5);
        assert_eq!(imp.sprites[4].name, "hero_icon");
        assert_eq!(imp.sprites[4].rect.w, 2);
        assert_eq!(imp.clips.len(), 3);
        let walk = &imp.clips[0].1;
        assert!(walk.looping && (walk.length - (0.1 + 0.05 + 0.1)).abs() < 1e-5);
        let swap = &walk.tracks[0];
        assert_eq!(swap.property, TrackProperty::SpriteSwap);
        assert_eq!(swap.keys.len(), 3);
        assert!((swap.keys[1].time - 0.1).abs() < 1e-6 && (swap.keys[2].time - 0.15).abs() < 1e-6);
        assert_eq!(swap.keys[0].value[0] as u32, entry_hash("hero", "Frame 0"));
        assert_eq!(swap.keys[0].interp, KeyInterp::Step);
        assert_eq!(walk.events.len(), 1);
        assert_eq!((walk.events[0].name.as_str(), walk.events[0].time), ("Step", 0.1));
        let once = &imp.clips[2].1;
        assert!(!once.looping);
        assert_eq!(once.tracks[0].keys.len(), 4);
        // The library resolves every key.
        let lib = crate::sprite_library::ResolvedLibrary::from_chain(&[&imp.library]);
        for k in &swap.keys {
            assert!(lib.sprite(k.value[0] as u32).is_some());
        }
        // Slice 9-patch centre (1,1,0,0) in a 2x2 -> borders l1 b1 r1 t1.
        assert_eq!(imp.sprites[4].border, [1, 1, 1, 1]);
        assert!(imp.sprites[4].pivot[0] > 0.0);
    }

    #[test]
    fn import_lays_out_a_padded_sheet_with_canvas_pivots() {
        let red = solid(2, 2, [255, 0, 0, 255]);
        let green = solid(2, 2, [0, 255, 0, 255]);
        let b = file(
            6,
            4,
            32,
            1,
            &[
                (100, vec![layer("l", 0, 0, 1, 0, 255, None), cel(0, 1, 1, 255, 0, Cel::Raw(2, 2, &red))]),
                (100, vec![cel(0, 3, 1, 255, 0, Cel::Raw(2, 2, &green))]),
            ],
        );
        let a = AseFile::parse(&b).unwrap();
        let imp = import(&a, &ImportOptions { name: "x".into(), padding: 2, ..Default::default() }).unwrap();
        assert_eq!(imp.sprites.len(), 2);
        let s0 = imp.sprites[0].rect;
        assert_eq!((s0.w, s0.h), (6, 4), "canvas-sized cells keep frames aligned");
        // No cell overlaps another and padding is transparent.
        let s1 = imp.sprites[1].rect;
        assert!(s0.x + s0.w + 2 <= s1.x || s1.x + s1.w + 2 <= s0.x || s0.y + s0.h + 2 <= s1.y || s1.y + s1.h + 2 <= s0.y);
        let px = |x: u32, y: u32| -> [u8; 4] { imp.sheet[((y * imp.sheet_w + x) * 4) as usize..][..4].try_into().unwrap() };
        assert_eq!(px(s0.x + 1, s0.y + 1), [255, 0, 0, 255]);
        assert_eq!(px(s1.x + 3, s1.y + 1), [0, 255, 0, 255]);
        assert_eq!(px(s0.x - 1, s0.y)[3], 0);
        assert_eq!(imp.sprites[0].pivot, [0.5, 0.5]);

        // Individual layers: pivot in cel pixels keeps the canvas centre, and
        // a frame without a cel switches the layer off.
        let b = file(
            6,
            4,
            32,
            1,
            &[
                (100, vec![layer("hand", 0, 0, 1, 0, 255, None), cel(0, 1, 1, 255, 0, Cel::Raw(2, 2, &red))]),
                (100, vec![]),
            ],
        );
        let a = AseFile::parse(&b).unwrap();
        let imp = import(&a, &ImportOptions { name: "x".into(), merge_layers: false, ..Default::default() }).unwrap();
        assert_eq!(imp.sprites.len(), 1);
        assert_eq!(imp.sprites[0].name, "x_hand_0");
        // Canvas centre (3,2) in the cel at (1,1): pixel (2,1) of a 2x2 cel -> normalised (1.0, 1 - 0.5).
        assert_eq!(imp.sprites[0].pivot, [1.0, 0.5]);
        let clip = &imp.clips[0].1;
        let active = clip.tracks.iter().find(|t| t.property == TrackProperty::Active).expect("an Active track for the empty frame");
        assert_eq!(active.target, "hand");
        assert_eq!(active.keys.iter().map(|k| k.value[0]).collect::<Vec<_>>(), vec![1.0, 0.0]);
    }

    #[test]
    fn bind_library_writes_sprite_refs() {
        let red = solid(1, 1, [1, 2, 3, 255]);
        let a = AseFile::parse(&file(1, 1, 32, 1, &[(100, vec![layer("l", 0, 0, 1, 0, 255, None), cel(0, 0, 0, 255, 0, Cel::Raw(1, 1, &red))])])).unwrap();
        let mut imp = import(&a, &ImportOptions { name: "s".into(), ..Default::default() }).unwrap();
        imp.sprites[0].id = crate::sub_asset::SubAssetId(0xabc);
        imp.bind_library("art/s.png");
        assert_eq!(imp.library.categories[0].entries[0].sprite, "art/s.png#0000000000000abc");
    }

    #[test]
    fn old_palette_chunks() {
        let mut b = Vec::new();
        w16(&mut b, 1);
        b.push(0);
        b.push(2);
        b.extend([255, 0, 0, 0, 255, 0]);
        let a = AseFile::parse(&file(1, 1, 8, 1, &[(100, vec![chunk(0x0004, &b), layer("l", 0, 0, 1, 0, 255, None), cel(0, 0, 0, 255, 0, Cel::Raw(1, 1, &[1]))])])).unwrap();
        assert_eq!(a.palette[1], [0, 255, 0, 255]);
        assert_eq!(&a.flatten_frame(0)[0..4], &[0, 255, 0, 255]);
    }

    #[test]
    fn corrupt_files_error_without_panicking() {
        let red = solid(2, 2, [255, 0, 0, 255]);
        let good = file(4, 4, 32, 1, &[(100, vec![layer("l", 0, 0, 1, 0, 255, None), cel(0, 0, 0, 255, 0, Cel::Zlib(2, 2, &red))])]);
        assert!(AseFile::parse(&good).is_ok());
        assert!(AseFile::parse(&[]).is_err());
        assert!(AseFile::parse(&good[..60]).is_err());
        let mut bad = good.clone();
        bad[4] = 0;
        assert!(AseFile::parse(&bad).is_err(), "bad magic");
        // Every truncation either parses (impossible here) or errors.
        for n in 0..good.len() {
            let _ = AseFile::parse(&good[..n]);
        }
        // Corrupt zlib data.
        let mut bad = good.clone();
        let n = bad.len();
        bad[n - zlib(&red).len()] = 0xFF; // the zlib header byte
        assert!(AseFile::parse(&bad).is_err());
        // A chunk size that overruns its frame.
        let mut bad = good.clone();
        bad[128 + 16..128 + 20].copy_from_slice(&0x00ff_ffffu32.to_le_bytes());
        assert!(AseFile::parse(&bad).is_err());
        // An enormous claimed cel is refused, not allocated.
        let mut huge = Vec::new();
        w16(&mut huge, 0);
        w16(&mut huge, 0);
        w16(&mut huge, 0);
        huge.push(255);
        w16(&mut huge, 2);
        w16(&mut huge, 0);
        huge.extend([0; 5]);
        w16(&mut huge, 65535);
        w16(&mut huge, 65535);
        huge.extend(zlib(&[0; 4]));
        let bytes = file(1, 1, 32, 1, &[(100, vec![layer("l", 0, 0, 1, 0, 255, None), chunk(0x2005, &huge)])]);
        assert!(AseFile::parse(&bytes).is_err());
    }

    #[test]
    fn all_hidden_is_an_error_not_a_panic() {
        let a = AseFile::parse(&file(1, 1, 32, 1, &[(100, vec![layer("l", 0, 0, 0, 0, 255, None), cel(0, 0, 0, 255, 0, Cel::Raw(1, 1, &[0, 0, 0, 255]))])])).unwrap();
        // Merged mode still yields (transparent) frame cells.
        assert_eq!(import(&a, &ImportOptions::default()).unwrap().sprites.len(), 1);
        assert!(import(&a, &ImportOptions { merge_layers: false, ..Default::default() }).is_err());
        let shown = import(&a, &ImportOptions { merge_layers: false, include_hidden: true, ..Default::default() }).unwrap();
        assert_eq!(shown.sprites.len(), 1);
    }
}
