//! PSD / PSB importer (2D-7): a pure-Rust parser for the layer stack of a
//! Photoshop file and its conversion to sprites that keep their position, layer
//! groups, and an optional starter rig.
//!
//! **Parser choice**: our own, not the `psd` crate. The layer path of the
//! format is small (header, layer records, channel data in raw / RLE / ZIP /
//! ZIP-with-prediction), and owning it lets the importer keep what the crate
//! drops (group structure, layer ids, unicode names, blend keys) with bounds
//! checks on every length.
//!
//! # Supported
//!
//! PSD (version 1) and PSB (version 2, 64-bit lengths), 8-bit and 16-bit
//! channels (16-bit is reduced to 8), RGB and grayscale colour modes, all four
//! channel compressions, layer rect, visibility, opacity, blend key, layer ids
//! (`lyid`), unicode names (`luni`), section dividers (`lsct`: groups open /
//! closed and their bounding divider). Not supported (a clear error, no
//! panic): CMYK, Lab, indexed, bitmap, 32-bit float. Layer masks and
//! adjustment / effect layers are ignored; blend modes are carried as the key
//! string but sprites are not pre-blended.
//!
//! # Layers to sprites
//!
//! One sprite per visible, non-empty pixel layer (hidden ones with
//! `include_hidden`), trimmed to its alpha, packed into one sheet with
//! `atlas_pack`. A [`LayerNode`] tree keeps what a prefab needs: the group
//! hierarchy, each sprite's position (world units, relative to the canvas
//! centre, y up, so the layers sit exactly where they were drawn), and a
//! sorting order of `layer_count - index` (higher PSD layers draw on top).
//! Layer opacity is baked into the sprite alpha. Stable ids come from the
//! layer id (default), falling back to the name; [`Imported::match_previous`]
//! lets a re-import keep the previous sprite ids.
//!
//! # Rig seed
//!
//! With `rig_seed` the import also returns a [`Rig2D`]: a `Root` bone at the
//! canvas centre, one bone per top-level group (positioned at the group's
//! bounds, pointing up) and a full-sprite quad part bound to its group's bone
//! (or the root). It is a starting point for the Skinning Editor, not a
//! finished rig.

use std::collections::HashMap;
use std::io::Read;

use anyhow::{Result, bail, ensure};
use flate2::read::ZlibDecoder;

use crate::atlas_pack::{self, PackSettings};
use crate::rig2d::{Bone2D, Rig2D, SkinMesh2D, SkinPart, Skeleton2D};
use crate::sprite::{PixelRect, PivotPreset, SpriteDef};
use crate::sub_asset::SubAssetId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerType {
    Pixel,
    /// Group header (the folder layer; its children precede it in the file).
    Group,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PsdLayer {
    pub name: String,
    pub id: Option<u32>,
    /// Canvas position of the layer's top-left, and its size.
    pub left: i32,
    pub top: i32,
    pub width: u32,
    pub height: u32,
    pub visible: bool,
    pub opacity: u8,
    /// Four-character blend key (`norm`, `mul `, ...).
    pub blend: String,
    pub kind: LayerType,
    /// Index of the parent group in [`PsdFile::layers`].
    pub parent: Option<usize>,
    /// RGBA8, straight alpha, `width * height * 4` (empty for groups).
    pub rgba: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PsdFile {
    pub width: u32,
    pub height: u32,
    pub big: bool,
    /// Bottom-to-top, as stored. Section-divider layers are consumed, never listed.
    pub layers: Vec<PsdLayer>,
}

struct Rd<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Rd<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(self.p.checked_add(n).is_some_and(|e| e <= self.b.len()), "psd: truncated data");
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
        Ok(u16::from_be_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn i16(&mut self) -> Result<i16> {
        Ok(self.u16()? as i16)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_be_bytes(self.take(8)?.try_into().unwrap()))
    }
    /// A length that is 8 bytes in PSB and 4 in PSD.
    fn len(&mut self, big: bool) -> Result<u64> {
        if big { self.u64() } else { Ok(self.u32()? as u64) }
    }
}

/// Largest layer / canvas edge accepted, and largest decoded plane.
const MAX_EDGE: i64 = 300_000;
const MAX_PIXELS: u64 = 1 << 28;

fn packbits(src: &[u8], out_len: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(out_len);
    let mut i = 0;
    while out.len() < out_len && i < src.len() {
        let n = src[i] as i8;
        i += 1;
        if n >= 0 {
            let c = n as usize + 1;
            ensure!(i + c <= src.len(), "psd: RLE literal run overruns its row");
            out.extend_from_slice(&src[i..i + c]);
            i += c;
        } else if n != -128 {
            ensure!(i < src.len(), "psd: RLE repeat run is truncated");
            out.extend(std::iter::repeat_n(src[i], (1 - n as isize) as usize));
            i += 1;
        }
    }
    ensure!(out.len() >= out_len, "psd: RLE row decodes to {} of {out_len} bytes", out.len());
    out.truncate(out_len);
    Ok(out)
}

/// Decode one channel plane to 8-bit.
fn decode_channel(data: &[u8], w: usize, h: usize, depth: u16, big: bool) -> Result<Vec<u8>> {
    let bps = depth as usize / 8;
    let row = w * bps;
    let mut r = Rd { b: data, p: 0 };
    let comp = r.u16()?;
    let raw: Vec<u8> = match comp {
        0 => r.take(row * h)?.to_vec(),
        1 => {
            let counts: Vec<usize> = (0..h).map(|_| if big { r.u32().map(|v| v as usize) } else { r.u16().map(|v| v as usize) }).collect::<Result<_>>()?;
            let mut out = Vec::with_capacity(row * h);
            for c in counts {
                out.extend(packbits(r.take(c)?, row)?);
            }
            out
        }
        2 | 3 => {
            let mut out = Vec::with_capacity(row * h);
            ZlibDecoder::new(&data[2..]).take((row * h) as u64 + 1).read_to_end(&mut out).map_err(|e| anyhow::anyhow!("psd: bad zlib stream: {e}"))?;
            ensure!(out.len() >= row * h, "psd: ZIP data holds {} of {} bytes", out.len(), row * h);
            out.truncate(row * h);
            if comp == 3 {
                for y in 0..h {
                    let line = &mut out[y * row..(y + 1) * row];
                    if bps == 1 {
                        for x in 1..w {
                            line[x] = line[x].wrapping_add(line[x - 1]);
                        }
                    } else {
                        for x in 1..w {
                            let prev = u16::from_be_bytes([line[(x - 1) * 2], line[(x - 1) * 2 + 1]]);
                            let cur = u16::from_be_bytes([line[x * 2], line[x * 2 + 1]]).wrapping_add(prev);
                            line[x * 2..x * 2 + 2].copy_from_slice(&cur.to_be_bytes());
                        }
                    }
                }
            }
            out
        }
        c => bail!("psd: unknown compression {c}"),
    };
    Ok(if bps == 1 { raw } else { raw.chunks_exact(bps).map(|c| c[0]).collect() })
}

fn unicode_name(d: &[u8]) -> Option<String> {
    let n = u32::from_be_bytes(d.get(0..4)?.try_into().ok()?) as usize;
    let units: Vec<u16> = d.get(4..4 + n * 2)?.as_chunks::<2>().0.iter().map(|c| u16::from_be_bytes([c[0], c[1]])).collect();
    Some(String::from_utf16_lossy(&units).trim_end_matches('\0').to_owned())
}

struct Record {
    top: i32,
    left: i32,
    bottom: i32,
    right: i32,
    channels: Vec<(i16, u64)>,
    blend: String,
    opacity: u8,
    visible: bool,
    name: String,
    id: Option<u32>,
    divider: u32,
}

impl PsdFile {
    pub fn parse(bytes: &[u8]) -> Result<PsdFile> {
        let mut r = Rd { b: bytes, p: 0 };
        ensure!(r.take(4)? == b"8BPS", "psd: not a Photoshop file");
        let version = r.u16()?;
        ensure!(version == 1 || version == 2, "psd: unknown version {version}");
        let big = version == 2;
        r.skip(6)?;
        let _channels = r.u16()?;
        let height = r.u32()?;
        let width = r.u32()?;
        let depth = r.u16()?;
        let mode = r.u16()?;
        ensure!(width > 0 && height > 0 && (width as i64) <= MAX_EDGE && (height as i64) <= MAX_EDGE, "psd: canvas of {width}x{height} is out of range");
        ensure!(depth == 8 || depth == 16, "psd: {depth}-bit channels are not supported (8 and 16 are)");
        ensure!(mode == 1 || mode == 3, "psd: colour mode {mode} is not supported (RGB and grayscale are)");
        let skip = r.u32()? as usize;
        r.skip(skip)?; // colour mode data
        let skip = r.u32()? as usize;
        r.skip(skip)?; // image resources
        let layer_mask_len = r.len(big)? as usize;
        let section_end = r.p.checked_add(layer_mask_len).filter(|&e| e <= bytes.len());
        let section_end = section_end.ok_or_else(|| anyhow::anyhow!("psd: layer section overruns the file"))?;
        let mut file = PsdFile { width, height, big, layers: Vec::new() };
        if layer_mask_len == 0 {
            return Ok(file);
        }
        let info_len = r.len(big)? as usize;
        let info_end = r.p.checked_add(info_len).filter(|&e| e <= section_end).ok_or_else(|| anyhow::anyhow!("psd: layer info overruns the section"))?;
        if info_len == 0 {
            return Ok(file);
        }
        let count = r.i16()?;
        let n = count.unsigned_abs() as usize;
        let mut recs: Vec<Record> = Vec::with_capacity(n);
        for _ in 0..n {
            let (top, left, bottom, right) = (r.i32()?, r.i32()?, r.i32()?, r.i32()?);
            let nch = r.u16()? as usize;
            ensure!(nch <= 64, "psd: {nch} channels in a layer");
            let mut channels = Vec::with_capacity(nch);
            for _ in 0..nch {
                let id = r.i16()?;
                channels.push((id, r.len(big)?));
            }
            ensure!(r.take(4)? == b"8BIM", "psd: bad blend signature");
            let blend = String::from_utf8_lossy(r.take(4)?).into_owned();
            let opacity = r.u8()?;
            let _clip = r.u8()?;
            let flags = r.u8()?;
            r.skip(1)?;
            let extra_len = r.u32()? as usize;
            let extra_end = r.p.checked_add(extra_len).filter(|&e| e <= info_end).ok_or_else(|| anyhow::anyhow!("psd: layer extra data overruns"))?;
            let mask_len = r.u32()? as usize;
            r.skip(mask_len)?;
            let blend_ranges = r.u32()? as usize;
            r.skip(blend_ranges)?;
            let plen = r.u8()? as usize;
            let pascal = String::from_utf8_lossy(r.take(plen)?).into_owned();
            r.skip((4 - (1 + plen) % 4) % 4)?;
            let mut rec = Record { top, left, bottom, right, channels, blend, opacity, visible: flags & 2 == 0, name: pascal, id: None, divider: 0 };
            // Additional layer information blocks.
            while r.p + 12 <= extra_end {
                let sig = r.take(4)?;
                if sig != b"8BIM" && sig != b"8B64" {
                    r.p = extra_end;
                    break;
                }
                let key = r.take(4)?;
                let wide = big && matches!(key, b"LMsk" | b"Lr16" | b"Lr32" | b"Layr" | b"Mt16" | b"Mt32" | b"Mtrn" | b"Alph" | b"FMsk" | b"lnk2" | b"FEid" | b"FXid" | b"PxSD");
                let len = r.len(wide)? as usize;
                let data = r.take(len)?;
                match key {
                    b"luni" => {
                        if let Some(n) = unicode_name(data) {
                            rec.name = n;
                        }
                    }
                    b"lyid" if data.len() >= 4 => rec.id = Some(u32::from_be_bytes(data[0..4].try_into().unwrap())),
                    b"lsct" | b"lsdk" if data.len() >= 4 => rec.divider = u32::from_be_bytes(data[0..4].try_into().unwrap()),
                    _ => {}
                }
            }
            r.p = extra_end;
            recs.push(rec);
        }
        // Channel image data, in record order.
        let mut open: Vec<usize> = Vec::new(); // file.layers length when each open group's divider was seen
        for rec in &recs {
            let (w, h) = ((rec.right as i64 - rec.left as i64).max(0), (rec.bottom as i64 - rec.top as i64).max(0));
            ensure!(w <= MAX_EDGE && h <= MAX_EDGE && (w as u64 * h as u64) <= MAX_PIXELS, "psd: layer '{}' of {w}x{h} is too large", rec.name);
            let (w, h) = (w as usize, h as usize);
            let mut planes: HashMap<i16, Vec<u8>> = HashMap::new();
            for &(id, len) in &rec.channels {
                let len = len as usize;
                let data = r.take(len)?;
                if len < 2 {
                    continue;
                }
                if w == 0 || h == 0 || id < -1 {
                    continue; // empty layer or a mask channel: not decoded
                }
                planes.insert(id, decode_channel(data, w, h, depth, big)?);
            }
            let mut rgba = Vec::new();
            if w > 0 && h > 0 {
                rgba = vec![255u8; w * h * 4];
                let colour = |id: i16| planes.get(&id);
                for (k, ids) in [[0i16, 1, 2], [0, 0, 0]].iter().enumerate() {
                    if (mode == 3) == (k == 0) {
                        for (c, id) in ids.iter().enumerate() {
                            match colour(*id) {
                                Some(p) => {
                                    for i in 0..w * h {
                                        rgba[i * 4 + c] = p[i];
                                    }
                                }
                                None => {
                                    for i in 0..w * h {
                                        rgba[i * 4 + c] = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some(a) = colour(-1) {
                    for i in 0..w * h {
                        rgba[i * 4 + 3] = a[i];
                    }
                }
            }
            // Section dividers: type 3 is the bounding divider (the group's lowest
            // entry in the file), 1 / 2 the folder layer that closes it.
            match rec.divider {
                3 => open.push(file.layers.len()),
                1 | 2 => {
                    let start = open.pop().unwrap_or(file.layers.len());
                    let idx = file.layers.len();
                    for l in &mut file.layers[start..idx] {
                        if l.parent.is_none() {
                            l.parent = Some(idx);
                        }
                    }
                    file.layers.push(layer_from(rec, w, h, LayerType::Group, Vec::new()));
                }
                _ => file.layers.push(layer_from(rec, w, h, LayerType::Pixel, rgba)),
            }
        }
        Ok(file)
    }

    /// Whether layer `i` and every group above it is visible.
    pub fn layer_visible(&self, mut i: usize) -> bool {
        for _ in 0..=self.layers.len() {
            if !self.layers[i].visible {
                return false;
            }
            match self.layers[i].parent {
                Some(p) => i = p,
                None => return true,
            }
        }
        false
    }
}

fn layer_from(rec: &Record, w: usize, h: usize, kind: LayerType, rgba: Vec<u8>) -> PsdLayer {
    PsdLayer {
        name: rec.name.clone(),
        id: rec.id,
        left: rec.left,
        top: rec.top,
        width: w as u32,
        height: h as u32,
        visible: rec.visible,
        opacity: rec.opacity,
        blend: rec.blend.clone(),
        kind,
        parent: None,
        rgba,
    }
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ImportOptions {
    pub name: String,
    pub include_hidden: bool,
    pub padding: u32,
    pub ppu: f32,
    /// Also build a starter [`Rig2D`].
    pub rig_seed: bool,
    pub max_sheet: u32,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self { name: "character".into(), include_hidden: false, padding: 2, ppu: 100.0, rig_seed: false, max_sheet: 4096 }
    }
}

/// One node of the prefab hierarchy the importer describes.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerNode {
    pub name: String,
    pub parent: Option<usize>,
    /// Index into [`Imported::sprites`]; `None` for a group.
    pub sprite: Option<usize>,
    /// World position relative to the canvas centre, y up (groups sit at the
    /// origin so children's local positions equal their world ones).
    pub position: [f32; 2],
    /// `layer_count - index_from_top`: higher PSD layers draw on top.
    pub sorting_order: i32,
    pub layer_id: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct Imported {
    pub sheet_w: u32,
    pub sheet_h: u32,
    pub sheet: Vec<u8>,
    pub sprites: Vec<SpriteDef>,
    /// Nodes in file order (children before their group).
    pub nodes: Vec<LayerNode>,
    pub rig: Option<Rig2D>,
}

fn layer_sprite_id(l: &PsdLayer) -> SubAssetId {
    match l.id {
        Some(id) => crate::sub_asset::hashed_id(crate::sub_asset::SubAssetKind::Sprite, &format!("psd-layer:{id}")),
        None => crate::sub_asset::hashed_id(crate::sub_asset::SubAssetKind::Sprite, &format!("psd-name:{}", l.name)),
    }
}

/// Convert the layer stack to a sheet, sprites, a node tree and optionally a rig seed.
pub fn import(file: &PsdFile, opt: &ImportOptions) -> Result<Imported> {
    struct Part {
        layer: usize,
        /// Trimmed rect on the canvas.
        x: i32,
        y: i32,
        w: u32,
        h: u32,
        rgba: Vec<u8>,
    }
    let mut parts: Vec<Part> = Vec::new();
    for (i, l) in file.layers.iter().enumerate() {
        if l.kind != LayerType::Pixel || l.width == 0 || l.height == 0 || !(opt.include_hidden || file.layer_visible(i)) {
            continue;
        }
        let (w, h) = (l.width as usize, l.height as usize);
        let (mut x0, mut y0, mut x1, mut y1) = (w, h, 0usize, 0usize);
        for y in 0..h {
            for x in 0..w {
                if l.rgba[(y * w + x) * 4 + 3] != 0 {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x + 1);
                    y1 = y1.max(y + 1);
                }
            }
        }
        if x1 <= x0 || y1 <= y0 {
            continue; // fully transparent
        }
        let (tw, th) = (x1 - x0, y1 - y0);
        let opacity = l.opacity as f32 / 255.0;
        let mut rgba = Vec::with_capacity(tw * th * 4);
        for y in y0..y1 {
            for x in x0..x1 {
                let p = &l.rgba[(y * w + x) * 4..][..4];
                rgba.extend([p[0], p[1], p[2], (p[3] as f32 * opacity + 0.5) as u8]);
            }
        }
        parts.push(Part { layer: i, x: l.left + x0 as i32, y: l.top + y0 as i32, w: tw as u32, h: th as u32, rgba });
    }
    ensure!(!parts.is_empty(), "psd: no visible layer with pixels to import");
    let items: Vec<(u32, u32)> = parts.iter().map(|p| (p.w, p.h)).collect();
    let packed = atlas_pack::pack(&items, &PackSettings { padding: opt.padding, allow_rotation: false, max_size: opt.max_sheet, pow2: true, uniform_pages: false })
        .map_err(|e| anyhow::anyhow!("psd: packing the layers failed: {e:?}"))?;
    ensure!(packed.pages.len() == 1, "psd: the layers do not fit one {0}x{0} sheet", opt.max_sheet);
    let (sheet_w, sheet_h) = packed.pages[0];
    let mut sheet = vec![0u8; (sheet_w * sheet_h * 4) as usize];
    let mut sprites = Vec::with_capacity(parts.len());
    let mut names: HashMap<String, u32> = HashMap::new();
    for (pi, part) in parts.iter().enumerate() {
        let pl = packed.placements[pi];
        for y in 0..part.h {
            let s = (y * part.w * 4) as usize;
            let d = (((pl.y + y) * sheet_w + pl.x) * 4) as usize;
            sheet[d..d + (part.w * 4) as usize].copy_from_slice(&part.rgba[s..s + (part.w * 4) as usize]);
        }
        let l = &file.layers[part.layer];
        let n = names.entry(l.name.clone()).or_insert(0);
        let name = if *n == 0 { l.name.clone() } else { format!("{}_{}", l.name, n) };
        *n += 1;
        let mut def = SpriteDef::named(&name, PixelRect::new(pl.x, pl.y, part.w, part.h));
        def.id = layer_sprite_id(l);
        def.pivot_preset = PivotPreset::Center;
        def.ppu = Some(opt.ppu);
        sprites.push(def);
    }
    // Node tree.
    let (cw, ch) = (file.width as f32, file.height as f32);
    let sprite_of_layer: HashMap<usize, usize> = parts.iter().enumerate().map(|(si, p)| (p.layer, si)).collect();
    let mut node_of_layer: HashMap<usize, usize> = HashMap::new();
    let mut nodes: Vec<LayerNode> = Vec::new();
    // Keep a node for every kept sprite and every group that has one below it.
    let keep_group = |g: usize| file.layers.iter().enumerate().any(|(i, l)| sprite_of_layer.contains_key(&i) && {
        let mut p = l.parent;
        let mut found = false;
        for _ in 0..=file.layers.len() {
            match p {
                Some(x) if x == g => {
                    found = true;
                    break;
                }
                Some(x) => p = file.layers[x].parent,
                None => break,
            }
        }
        found
    });
    for (i, l) in file.layers.iter().enumerate() {
        let is_group = l.kind == LayerType::Group;
        if !(sprite_of_layer.contains_key(&i) || (is_group && keep_group(i))) {
            continue;
        }
        let position = match sprite_of_layer.get(&i) {
            Some(&si) => {
                let p = &parts[si];
                [((p.x as f32 + p.w as f32 / 2.0) - cw / 2.0) / opt.ppu, (ch / 2.0 - (p.y as f32 + p.h as f32 / 2.0)) / opt.ppu]
            }
            None => [0.0, 0.0],
        };
        node_of_layer.insert(i, nodes.len());
        nodes.push(LayerNode { name: l.name.clone(), parent: None, sprite: sprite_of_layer.get(&i).copied(), position, sorting_order: i as i32 + 1, layer_id: l.id });
    }
    for (i, l) in file.layers.iter().enumerate() {
        if let Some(&n) = node_of_layer.get(&i) {
            // The nearest ancestor that kept a node.
            let mut p = l.parent;
            while let Some(x) = p {
                if let Some(&pn) = node_of_layer.get(&x) {
                    nodes[n].parent = Some(pn);
                    break;
                }
                p = file.layers[x].parent;
            }
        }
    }
    let rig = opt.rig_seed.then(|| rig_seed(file, &parts.iter().map(|p| (p.layer, p.x, p.y, p.w, p.h)).collect::<Vec<_>>(), &sprites, opt));
    Ok(Imported { sheet_w, sheet_h, sheet, sprites, nodes, rig })
}

/// Root bone at the canvas centre, a bone per top-level group, quad parts.
fn rig_seed(file: &PsdFile, parts: &[(usize, i32, i32, u32, u32)], sprites: &[SpriteDef], opt: &ImportOptions) -> Rig2D {
    let (cw, ch) = (file.width as f32, file.height as f32);
    let to_world = |x: f32, y: f32| [(x - cw / 2.0) / opt.ppu, (ch / 2.0 - y) / opt.ppu];
    let top_group = |mut l: usize| -> Option<usize> {
        let mut found = None;
        for _ in 0..=file.layers.len() {
            match file.layers[l].parent {
                Some(p) => {
                    found = Some(p);
                    l = p;
                }
                None => break,
            }
        }
        found
    };
    let mut skel = Skeleton2D::default();
    skel.bones.push(Bone2D { name: "Root".into(), length: 0.25 * (ch / opt.ppu).max(0.2), rotation: std::f32::consts::FRAC_PI_2, ..Default::default() });
    let mut group_bone: HashMap<usize, usize> = HashMap::new();
    // Bounds of each top-level group's sprites.
    let mut bounds: HashMap<usize, (f32, f32, f32, f32)> = HashMap::new();
    for &(layer, x, y, w, h) in parts {
        if let Some(g) = top_group(layer) {
            let b = bounds.entry(g).or_insert((f32::MAX, f32::MAX, f32::MIN, f32::MIN));
            *b = (b.0.min(x as f32), b.1.min(y as f32), b.2.max((x + w as i32) as f32), b.3.max((y + h as i32) as f32));
        }
    }
    let mut groups: Vec<usize> = bounds.keys().copied().collect();
    groups.sort();
    for g in groups {
        let (x0, y0, x1, y1) = bounds[&g];
        let head = to_world((x0 + x1) / 2.0, y1); // bottom-centre of the group
        let tip = to_world((x0 + x1) / 2.0, y0); // top-centre
        let idx = skel.add_bone(Some(0), head, tip, &file.layers[g].name);
        group_bone.insert(g, idx);
    }
    let mut rig = Rig2D { skeleton: skel, parts: Vec::new() };
    for (si, &(layer, _x, _y, w, h)) in parts.iter().enumerate() {
        let bone = top_group(layer).and_then(|g| group_bone.get(&g).copied()).unwrap_or(0) as u16;
        let def = &sprites[si];
        let (wu, hu) = (w as f32 / opt.ppu, h as f32 / opt.ppu);
        let (hx, hy) = (wu / 2.0, hu / 2.0);
        let mesh = SkinMesh2D::new(vec![[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], vec![0, 1, 2, 0, 2, 3], bone);
        rig.parts.push(SkinPart { sprite_id: def.id.0, name: def.name.clone(), mesh, bone_indices: Vec::new() });
    }
    rig
}

impl Imported {
    /// Keep the ids and the user's edits of a previous import: sprites match by
    /// id, then by name; only the rect (where it landed in the sheet) is new.
    pub fn match_previous(&mut self, prev: &[SpriteDef]) -> usize {
        let mut matched = 0;
        for s in &mut self.sprites {
            if let Some(p) = prev.iter().find(|p| p.id == s.id).or_else(|| prev.iter().find(|p| p.name == s.name)) {
                let rect = s.rect;
                *s = p.clone();
                s.rect = rect;
                matched += 1;
            }
        }
        if let Some(rig) = &mut self.rig {
            for part in &mut rig.parts {
                if let Some(s) = self.sprites.iter().find(|s| s.name == part.name) {
                    part.sprite_id = s.id.0;
                }
            }
        }
        matched
    }
}

// ---------------------------------------------------------------------------
// Tests: synthesized files
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod synth {
    use flate2::{Compression, write::ZlibEncoder};
    use std::io::Write;

    pub fn be16(v: &mut Vec<u8>, x: u16) {
        v.extend(x.to_be_bytes());
    }
    pub fn be32(v: &mut Vec<u8>, x: u32) {
        v.extend(x.to_be_bytes());
    }
    pub fn be64(v: &mut Vec<u8>, x: u64) {
        v.extend(x.to_be_bytes());
    }

    #[derive(Clone, Copy, PartialEq)]
    pub enum Comp {
        Raw,
        Rle,
        Zip,
        ZipPredict,
    }

    pub struct L {
        pub name: &'static str,
        pub id: Option<u32>,
        pub rect: (i32, i32, i32, i32), // top, left, bottom, right
        pub rgba: Vec<u8>,
        pub visible: bool,
        pub opacity: u8,
        pub divider: u32,
        pub comp: Comp,
        pub unicode: Option<&'static str>,
    }

    pub fn layer(name: &'static str, id: u32, rect: (i32, i32, i32, i32), color: [u8; 4], comp: Comp) -> L {
        let n = ((rect.2 - rect.0) * (rect.3 - rect.1)).max(0) as usize;
        L { name, id: Some(id), rect, rgba: (0..n).flat_map(|_| color).collect(), visible: true, opacity: 255, divider: 0, comp, unicode: None }
    }
    pub fn group(name: &'static str, id: u32, open: bool) -> L {
        L { name, id: Some(id), rect: (0, 0, 0, 0), rgba: vec![], visible: true, opacity: 255, divider: if open { 1 } else { 2 }, comp: Comp::Raw, unicode: None }
    }
    pub fn divider() -> L {
        L { name: "</Layer group>", id: None, rect: (0, 0, 0, 0), rgba: vec![], visible: true, opacity: 255, divider: 3, comp: Comp::Raw, unicode: None }
    }

    fn rle_row(row: &[u8]) -> Vec<u8> {
        // Literal runs of up to 128 bytes: valid PackBits, plus a repeat for runs.
        let mut out = Vec::new();
        let mut i = 0;
        while i < row.len() {
            let mut run = 1;
            while i + run < row.len() && row[i + run] == row[i] && run < 128 {
                run += 1;
            }
            if run >= 3 {
                out.push((1 - run as i32) as i8 as u8);
                out.push(row[i]);
                i += run;
            } else {
                let mut lit = run;
                while i + lit < row.len() && lit < 128 && !(i + lit + 2 < row.len() && row[i + lit] == row[i + lit + 1] && row[i + lit] == row[i + lit + 2]) {
                    lit += 1;
                }
                out.push((lit - 1) as u8);
                out.extend(&row[i..i + lit]);
                i += lit;
            }
        }
        out
    }

    fn channel(plane: &[u8], w: usize, h: usize, comp: Comp, big: bool) -> Vec<u8> {
        let mut out = Vec::new();
        match comp {
            Comp::Raw => {
                be16(&mut out, 0);
                out.extend(plane);
            }
            Comp::Rle => {
                be16(&mut out, 1);
                let rows: Vec<Vec<u8>> = (0..h).map(|y| rle_row(&plane[y * w..(y + 1) * w])).collect();
                for r in &rows {
                    if big {
                        be32(&mut out, r.len() as u32);
                    } else {
                        be16(&mut out, r.len() as u16);
                    }
                }
                for r in rows {
                    out.extend(r);
                }
            }
            Comp::Zip | Comp::ZipPredict => {
                be16(&mut out, if comp == Comp::Zip { 2 } else { 3 });
                let mut data = plane.to_vec();
                if comp == Comp::ZipPredict {
                    for y in 0..h {
                        for x in (1..w).rev() {
                            data[y * w + x] = data[y * w + x].wrapping_sub(data[y * w + x - 1]);
                        }
                    }
                }
                let mut e = ZlibEncoder::new(Vec::new(), Compression::fast());
                e.write_all(&data).unwrap();
                out.extend(e.finish().unwrap());
            }
        }
        out
    }

    /// `big` writes a PSB.
    pub fn file(w: u32, h: u32, layers: &[L], big: bool) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(b"8BPS");
        be16(&mut out, if big { 2 } else { 1 });
        out.extend([0; 6]);
        be16(&mut out, 4);
        be32(&mut out, h);
        be32(&mut out, w);
        be16(&mut out, 8);
        be16(&mut out, 3);
        be32(&mut out, 0); // colour mode data
        be32(&mut out, 0); // image resources
        let mut records = Vec::new();
        let mut data = Vec::new();
        be16(&mut records, layers.len() as u16);
        for l in layers {
            let (top, left, bottom, right) = l.rect;
            let (lw, lh) = ((right - left).max(0) as usize, (bottom - top).max(0) as usize);
            let mut chans: Vec<(i16, Vec<u8>)> = Vec::new();
            if lw > 0 && lh > 0 {
                for (id, k) in [(-1i16, 3usize), (0, 0), (1, 1), (2, 2)] {
                    let plane: Vec<u8> = l.rgba.as_chunks::<4>().0.iter().map(|p| p[k]).collect();
                    chans.push((id, channel(&plane, lw, lh, l.comp, big)));
                }
            }
            for v in [top, left, bottom, right] {
                be32(&mut records, v as u32);
            }
            be16(&mut records, chans.len() as u16);
            for (id, d) in &chans {
                be16(&mut records, *id as u16);
                if big {
                    be64(&mut records, d.len() as u64);
                } else {
                    be32(&mut records, d.len() as u32);
                }
            }
            records.extend(b"8BIM");
            records.extend(b"norm");
            records.push(l.opacity);
            records.push(0);
            records.push(if l.visible { 0 } else { 2 });
            records.push(0);
            let mut extra = Vec::new();
            be32(&mut extra, 0); // mask
            be32(&mut extra, 0); // blend ranges
            let nm = l.name.as_bytes();
            extra.push(nm.len() as u8);
            extra.extend(nm);
            let pad = (4 - (1 + nm.len()) % 4) % 4;
            extra.extend(std::iter::repeat_n(0u8, pad));
            let mut block = |key: &[u8; 4], body: &[u8]| {
                extra.extend(b"8BIM");
                extra.extend(key);
                be32(&mut extra, body.len() as u32);
                extra.extend(body);
            };
            if let Some(id) = l.id {
                block(b"lyid", &id.to_be_bytes());
            }
            if l.divider != 0 {
                block(b"lsct", &l.divider.to_be_bytes());
            }
            if let Some(u) = l.unicode {
                let units: Vec<u16> = u.encode_utf16().collect();
                let mut body = Vec::new();
                be32(&mut body, units.len() as u32);
                for c in units {
                    be16(&mut body, c);
                }
                body.extend([0, 0]);
                if body.len() % 2 == 1 {
                    body.push(0);
                }
                block(b"luni", &body);
            }
            be32(&mut records, extra.len() as u32);
            records.extend(extra);
            for (_, d) in chans {
                data.extend(d);
            }
        }
        let mut info = records;
        info.extend(data);
        if info.len() % 2 == 1 {
            info.push(0);
        }
        let mut section = Vec::new();
        if big {
            be64(&mut section, info.len() as u64);
        } else {
            be32(&mut section, info.len() as u32);
        }
        section.extend(info);
        be32(&mut section, 0); // global mask
        if big {
            be64(&mut out, section.len() as u64);
        } else {
            be32(&mut out, section.len() as u32);
        }
        out.extend(section);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::synth::*;
    use super::*;

    fn px(l: &PsdLayer, x: usize, y: usize) -> [u8; 4] {
        l.rgba[(y * l.width as usize + x) * 4..][..4].try_into().unwrap()
    }

    #[test]
    fn every_compression_decodes_to_the_same_pixels() {
        for comp in [Comp::Raw, Comp::Rle, Comp::Zip, Comp::ZipPredict] {
            for big in [false, true] {
                let mut l = layer("a", 7, (2, 3, 6, 8), [10, 20, 30, 255], comp);
                // A gradient so prediction and RLE are exercised on non-constant rows.
                for (i, p) in l.rgba.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    p[0] = (i * 7) as u8;
                    p[3] = if i % 5 == 0 { 0 } else { 200 };
                }
                let reference = l.rgba.clone();
                let f = PsdFile::parse(&file(16, 12, &[l], big)).unwrap();
                assert_eq!(f.big, big);
                let got = &f.layers[0];
                assert_eq!((got.left, got.top, got.width, got.height), (3, 2, 5, 4));
                assert_eq!(got.rgba, reference, "{} {}", match comp { Comp::Raw => "raw", Comp::Rle => "rle", Comp::Zip => "zip", Comp::ZipPredict => "zip+predict" }, big);
                assert_eq!(got.id, Some(7));
            }
        }
    }

    #[test]
    fn groups_visibility_and_names() {
        let mut hidden = layer("hidden", 3, (0, 0, 2, 2), [0, 0, 255, 255], Comp::Raw);
        hidden.visible = false;
        let mut uni = layer("x", 4, (0, 0, 2, 2), [1, 1, 1, 255], Comp::Raw);
        uni.unicode = Some("Hand \u{00e9}");
        // File order is bottom to top: bg, [divider, a, b, Arm(folder)], hidden, uni.
        let layers = vec![
            layer("bg", 1, (0, 0, 8, 8), [9, 9, 9, 255], Comp::Raw),
            divider(),
            layer("a", 2, (1, 1, 3, 3), [255, 0, 0, 255], Comp::Rle),
            layer("b", 5, (4, 4, 6, 6), [0, 255, 0, 255], Comp::Zip),
            group("Arm", 6, true),
            hidden,
            uni,
        ];
        let f = PsdFile::parse(&file(8, 8, &layers, false)).unwrap();
        let names: Vec<_> = f.layers.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, vec!["bg", "a", "b", "Arm", "hidden", "Hand \u{00e9}"], "the divider is consumed, luni overrides the name");
        let arm = 3;
        assert_eq!(f.layers[arm].kind, LayerType::Group);
        assert_eq!(f.layers[1].parent, Some(arm));
        assert_eq!(f.layers[2].parent, Some(arm));
        assert_eq!(f.layers[0].parent, None);
        assert!(!f.layer_visible(4));
        assert!(f.layer_visible(1));
        assert_eq!(px(&f.layers[1], 0, 0), [255, 0, 0, 255]);
    }

    #[test]
    fn nested_groups_and_hidden_group_hides_children() {
        let mut outer = group("Outer", 9, false);
        outer.visible = false;
        let layers = vec![
            divider(),
            divider(),
            layer("deep", 1, (0, 0, 2, 2), [5, 5, 5, 255], Comp::Raw),
            group("Inner", 2, true),
            layer("mid", 3, (2, 2, 4, 4), [6, 6, 6, 255], Comp::Raw),
            outer,
        ];
        let f = PsdFile::parse(&file(8, 8, &layers, false)).unwrap();
        assert_eq!(f.layers.len(), 4);
        assert_eq!(f.layers[0].parent, Some(1), "deep is in Inner");
        assert_eq!(f.layers[1].parent, Some(3), "Inner is in Outer");
        assert_eq!(f.layers[2].parent, Some(3));
        assert!(!f.layer_visible(0), "an ancestor group is hidden");
        assert!(import(&f, &ImportOptions::default()).is_err(), "nothing visible");
        let imp = import(&f, &ImportOptions { include_hidden: true, ..Default::default() }).unwrap();
        assert_eq!(imp.sprites.len(), 2);
    }

    #[test]
    fn import_keeps_positions_order_and_hierarchy() {
        let layers = vec![
            layer("bg", 1, (0, 0, 8, 8), [9, 9, 9, 255], Comp::Raw),
            divider(),
            layer("hand", 2, (4, 6, 6, 8), [255, 0, 0, 255], Comp::Raw),
            group("Arm", 3, true),
            layer("empty", 4, (0, 0, 2, 2), [0, 0, 0, 0], Comp::Raw),
        ];
        let f = PsdFile::parse(&file(8, 8, &layers, false)).unwrap();
        let imp = import(&f, &ImportOptions { name: "c".into(), ppu: 10.0, rig_seed: true, ..Default::default() }).unwrap();
        assert_eq!(imp.sprites.len(), 2, "the transparent layer is skipped");
        let hand = imp.nodes.iter().find(|n| n.name == "hand").unwrap();
        // Layer rect x 6..8, y 4..6: centre (7, 5) on an 8x8 canvas -> (3, -1) / ppu 10.
        assert!((hand.position[0] - 0.3).abs() < 1e-5 && (hand.position[1] + 0.1).abs() < 1e-5, "{:?}", hand.position);
        let arm = imp.nodes.iter().position(|n| n.name == "Arm").unwrap();
        assert_eq!(hand.parent, Some(arm));
        assert_eq!(imp.nodes[arm].sprite, None);
        let bg = imp.nodes.iter().find(|n| n.name == "bg").unwrap();
        assert_eq!(bg.parent, None);
        assert!(bg.sorting_order < hand.sorting_order, "higher PSD layers sort on top");
        assert_eq!(bg.sorting_order, 1);
        // Pixels landed in the sheet where the sprite says.
        let s = imp.sprites.iter().find(|s| s.name == "hand").unwrap();
        let at = ((s.rect.y * imp.sheet_w + s.rect.x) * 4) as usize;
        assert_eq!(&imp.sheet[at..at + 4], &[255, 0, 0, 255]);
        assert_eq!((s.rect.w, s.rect.h), (2, 2));
        // The rig seed: root + Arm bone, parts bound by group, validates.
        let rig = imp.rig.as_ref().unwrap();
        assert_eq!(rig.skeleton.bones.len(), 2);
        assert_eq!(rig.skeleton.bones[1].name, "Arm");
        assert_eq!(rig.skeleton.bones[1].parent, 0);
        assert_eq!(rig.parts.len(), 2);
        let hand_part = rig.parts.iter().find(|p| p.name == "hand").unwrap();
        assert_eq!(hand_part.mesh.weights[0].bones[0], 1);
        assert_eq!(rig.parts.iter().find(|p| p.name == "bg").unwrap().mesh.weights[0].bones[0], 0);
        assert!(rig.validate().is_empty(), "{:?}", rig.validate());
        // Rest-pose skinning leaves the part where it is.
        let m = rig.skeleton.skin_matrices(&rig.skeleton.world_rest());
        let mut out = Vec::new();
        hand_part.mesh.skin(&m, &mut out);
        assert!(out.iter().zip(&hand_part.mesh.positions).all(|(a, b)| (a[0] - b[0]).abs() < 1e-5 && (a[1] - b[1]).abs() < 1e-5));
    }

    #[test]
    fn opacity_is_baked_and_ids_are_stable() {
        let mut l = layer("a", 5, (0, 0, 2, 2), [1, 2, 3, 255], Comp::Raw);
        l.opacity = 128;
        let f = PsdFile::parse(&file(4, 4, std::slice::from_ref(&l), false)).unwrap();
        let a = import(&f, &ImportOptions::default()).unwrap();
        let at = ((a.sprites[0].rect.y * a.sheet_w + a.sprites[0].rect.x) * 4) as usize;
        assert_eq!(a.sheet[at + 3], 128);
        // Same layer id after a rename or move keeps the sprite id; match_previous keeps edits.
        let mut l2 = layer("renamed", 5, (1, 1, 3, 3), [1, 2, 3, 255], Comp::Raw);
        l2.opacity = 128;
        let f2 = PsdFile::parse(&file(4, 4, &[l2], false)).unwrap();
        let mut b = import(&f2, &ImportOptions::default()).unwrap();
        assert_eq!(a.sprites[0].id, b.sprites[0].id);
        let mut prev = a.sprites.clone();
        prev[0].pivot = [0.1, 0.9];
        prev[0].pivot_preset = PivotPreset::Custom;
        assert_eq!(b.match_previous(&prev), 1);
        assert_eq!(b.sprites[0].pivot, [0.1, 0.9]);
        assert_eq!(b.sprites[0].rect.w, 2);
        // Duplicate names get suffixes.
        let two = [layer("x", 1, (0, 0, 1, 1), [1, 1, 1, 255], Comp::Raw), layer("x", 2, (1, 1, 2, 2), [1, 1, 1, 255], Comp::Raw)];
        let c = import(&PsdFile::parse(&file(4, 4, &two, false)).unwrap(), &ImportOptions::default()).unwrap();
        assert_eq!(c.sprites[0].name, "x");
        assert_eq!(c.sprites[1].name, "x_1");
        assert_ne!(c.sprites[0].id, c.sprites[1].id);
    }

    #[test]
    fn corrupt_and_unsupported_files_error() {
        let good = file(8, 8, &[layer("a", 1, (0, 0, 4, 4), [1, 1, 1, 255], Comp::Rle)], false);
        assert!(PsdFile::parse(&good).is_ok());
        assert!(PsdFile::parse(&[]).is_err());
        assert!(PsdFile::parse(b"8BPS").is_err());
        let mut bad = good.clone();
        bad[0] = b'X';
        assert!(PsdFile::parse(&bad).is_err());
        let mut cmyk = good.clone();
        cmyk[24..26].copy_from_slice(&4u16.to_be_bytes());
        let e = PsdFile::parse(&cmyk).unwrap_err().to_string();
        assert!(e.contains("colour mode"), "{e}");
        let mut deep = good.clone();
        deep[22..24].copy_from_slice(&32u16.to_be_bytes());
        assert!(PsdFile::parse(&deep).is_err());
        for n in 0..good.len() {
            let _ = PsdFile::parse(&good[..n]);
        }
        // Flip bytes across the file: errors or garbage pixels, never a panic.
        for i in (26..good.len()).step_by(3) {
            let mut b = good.clone();
            b[i] ^= 0xA5;
            let _ = PsdFile::parse(&b);
        }
        // A huge claimed layer is refused.
        let mut l = layer("a", 1, (0, 0, 4, 4), [1, 1, 1, 255], Comp::Raw);
        l.rect = (0, 0, 400_000, 400_000);
        l.rgba.clear();
        let huge = file(8, 8, &[l], false);
        assert!(PsdFile::parse(&huge).is_err());
    }

    #[test]
    fn an_image_with_no_layers_parses_empty() {
        let f = PsdFile::parse(&file(4, 4, &[], false)).unwrap();
        assert!(f.layers.is_empty());
        assert!(import(&f, &ImportOptions::default()).is_err());
    }
}
