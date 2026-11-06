//! Sprite atlases (2D-4): the `.spriteatlas` asset, packing at cook time into
//! array / separate page textures, the cooked sprite table, and resolving a
//! [`SpriteRef`] through an atlas.
//!
//! # The asset
//!
//! A [`SpriteAtlasFile`] (RON, `#[serde(default)]` everywhere so old files and
//! newer fields both load) lists what to pack (`packables`: a sprite texture's
//! project-relative path, a folder (recursive), or `path#<sprite id hex>` for a
//! single sprite), the packing rules (padding, rotation, Rectangle or Tight mode,
//! page size, array layers or separate pages, power-of-two) and the page texture
//! settings (a [`TextureImport`], so BC7 / ASTC, sRGB, mips and the rest go
//! through the same cook as any texture). `desktop` / `quest` override the page
//! size, padding, rotation and which **variant** ships; a variant is the same
//! layout with every page scaled (0.5 = a quarter of the memory), UVs unchanged.
//! `include_in_build` = false keeps the atlas out of game paks (the editor still
//! packs it for Play). A sprite in two atlases belongs to the first (by path).
//!
//! # Authoring is unchanged
//!
//! Sprites keep their textures and ids; the original textures stay for the
//! editor. Cooking packs the sprites and records, per sprite, a [`CookedSprite`]
//! (page, placed rect, rotation, trimmed offset, mesh, physics). A
//! [`SpriteRef`] resolves through [`AtlasDb::resolve`] to the atlas page and
//! [`atlas_geometry`] gives the same [`SpriteGeometry`] an unatlased sprite
//! has, with the UVs pointing into the page and the layer in
//! [`AtlasGeometry::page`].
//!
//! # Packing
//!
//! 1. Every sprite is cropped from its texture. Fully transparent borders are
//!    trimmed (`trim`, not for 9-sliced sprites) and the offset recorded.
//! 2. [`crate::atlas_pack`] places the trimmed rects. **Tight** mode also traces
//!    the alpha polygon (stored in [`CookedSprite::polygon`], inside the trim
//!    rect) but does not interlock concave shapes: it packs the trimmed bounds.
//! 3. The page is composited: the content (rotated 90 degrees clockwise when
//!    `rotated`), then the ring of padding filled by repeating the content's
//!    edge texels, so bilinear filtering and mips never read a neighbour. When
//!    mips are on the padding grows to `2^mip_levels` and the chain is cut to
//!    `mip_levels + 1` levels; pixel-art atlases (`pixel_art`) have no mips.
//! 4. Pages are scaled for the variant, then encoded by `tex_cache`.
//!
//! The result is cached next to the atlas in `.zestengine_atlascache/`, keyed by
//! a hash of the atlas settings, the platform and the **content** of every
//! member texture (and their sprite tables); the game pak cook (`game_pak`)
//! ships those files.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result, bail};
use serde::{Deserialize, Serialize};
use zestengine_core::ObjectId;
use zestengine_render::CompressedTexture;

use crate::atlas_pack::{self, PackSettings};
use crate::sprite::{PixelRect, SpriteDef, SpriteGeometry, SpriteMeshType, SpriteRef, SpriteSheet, pixel_to_world, sprite_physics_pixels, sprite_pixel_mesh};
use crate::sprite_mesh::{self, AlphaMap, P2};
use crate::sub_asset::SubAssetId;
use crate::texture_import::{Platform, TextureImport, TextureKind};

/// File extension of an atlas asset.
pub const ATLAS_EXTENSION: &str = "spriteatlas";
/// Folder (next to the atlas) holding cooked atlases.
pub const ATLAS_CACHE_DIR: &str = ".zestengine_atlascache";
/// Extension of a cooked atlas.
pub const COOKED_ATLAS_EXTENSION: &str = "catlas";
/// Bumped when the packing or the cooked layout changes (invalidates caches).
const ATLAS_VERSION: u32 = 1;
const COOKED_MAGIC: &[u8; 8] = b"ZATLAS01";
/// Most extra mip levels an atlas keeps (padding is `2^levels`).
pub const MAX_MIP_LEVELS: u32 = 6;

// ---------------------------------------------------------------------------
// The asset
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackingMode {
    /// Pack the alpha-trimmed rect of each sprite.
    #[default]
    Rectangle,
    /// As Rectangle, and record each sprite's alpha polygon.
    Tight,
}

impl PackingMode {
    pub const ALL: [Self; 2] = [Self::Rectangle, Self::Tight];
    pub fn label(self) -> &'static str {
        match self {
            Self::Rectangle => "Rectangle",
            Self::Tight => "Tight",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PageLayout {
    /// All pages share one size and become the layers of one array texture.
    #[default]
    Array,
    /// Each page is its own texture, sized to its content.
    Separate,
}

impl PageLayout {
    pub const ALL: [Self; 2] = [Self::Array, Self::Separate];
    pub fn label(self) -> &'static str {
        match self {
            Self::Array => "Texture array",
            Self::Separate => "Separate pages",
        }
    }
}

/// A scaled copy of the atlas (same layout and UVs, smaller pages).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AtlasVariant {
    pub name: String,
    /// Page scale in (0, 1].
    pub scale: f32,
}

impl Default for AtlasVariant {
    fn default() -> Self {
        Self { name: "Half".into(), scale: 0.5 }
    }
}

/// A platform tab's overrides (`None` = the atlas default).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AtlasPlatform {
    pub max_size: Option<u32>,
    pub padding: Option<u32>,
    pub allow_rotation: Option<bool>,
    /// Name of the variant this platform ships (`None` = the full-size atlas).
    pub variant: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpriteAtlasFile {
    /// Sprite textures, folders or `path#id` sprites (project-relative).
    pub packables: Vec<String>,
    pub padding: u32,
    pub allow_rotation: bool,
    pub packing: PackingMode,
    /// Trim fully transparent borders off every sprite.
    pub trim: bool,
    /// Largest page edge.
    pub max_size: u32,
    pub layout: PageLayout,
    pub power_of_two: bool,
    /// Pixel art: no mips, no filtering margin beyond the extruded edge.
    pub pixel_art: bool,
    /// Extra mip levels kept when the page texture has mips (padding `2^levels`).
    pub mip_levels: u32,
    /// How the pages are encoded (kind Sprite, mips, compression, sRGB ...).
    pub texture: TextureImport,
    pub desktop: Option<AtlasPlatform>,
    pub quest: Option<AtlasPlatform>,
    pub variants: Vec<AtlasVariant>,
    pub include_in_build: bool,
}

impl Default for SpriteAtlasFile {
    fn default() -> Self {
        let texture = TextureImport { configured: true, kind: TextureKind::Sprite, alpha_is_transparency: true, max_size: 0, ..TextureImport::default() };
        Self {
            packables: Vec::new(),
            padding: 4,
            allow_rotation: false,
            packing: PackingMode::Rectangle,
            trim: true,
            max_size: 2048,
            layout: PageLayout::Array,
            power_of_two: true,
            pixel_art: false,
            mip_levels: 3,
            texture: TextureImport { mips: crate::texture_import::MipOptions { enabled: false, ..Default::default() }, ..texture },
            desktop: None,
            quest: None,
            variants: Vec::new(),
            include_in_build: true,
        }
    }
}

/// The rules in force for one platform.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectiveAtlas {
    pub max_size: u32,
    /// Padding after raising it for the mip chain.
    pub padding: u32,
    pub allow_rotation: bool,
    /// Variant scale (1 = full size).
    pub scale: f32,
    pub variant: Option<String>,
    /// Mips on in the pages.
    pub mips: bool,
}

impl SpriteAtlasFile {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        ron::de::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let text = ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())?;
        std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
    }

    fn platform(&self, p: Platform) -> Option<&AtlasPlatform> {
        match p {
            Platform::Desktop => self.desktop.as_ref(),
            Platform::Quest => self.quest.as_ref(),
        }
    }

    /// Whether the pages carry a mip chain.
    pub fn uses_mips(&self) -> bool {
        self.texture.mips.enabled && !self.pixel_art
    }

    pub fn effective(&self, platform: Platform) -> EffectiveAtlas {
        let over = self.platform(platform);
        let mips = self.uses_mips();
        let mut padding = over.and_then(|o| o.padding).unwrap_or(self.padding);
        if mips {
            padding = padding.max(1 << self.mip_levels.min(MAX_MIP_LEVELS));
        } else if !self.pixel_art {
            padding = padding.max(1); // bilinear still reads one texel past the edge
        }
        let variant = over.and_then(|o| o.variant.clone()).filter(|v| self.variants.iter().any(|x| x.name == *v));
        let scale = variant.as_ref().and_then(|v| self.variants.iter().find(|x| x.name == *v)).map(|v| v.scale.clamp(0.05, 1.0)).unwrap_or(1.0);
        EffectiveAtlas {
            max_size: over.and_then(|o| o.max_size).unwrap_or(self.max_size).clamp(16, 16384),
            padding: padding.min(64),
            allow_rotation: over.and_then(|o| o.allow_rotation).unwrap_or(self.allow_rotation),
            scale,
            variant,
            mips,
        }
    }

    /// The page encoding settings for `platform`.
    pub fn page_import(&self, platform: Platform) -> TextureImport {
        let mut t = self.texture.clone();
        t.configured = true;
        t.kind = TextureKind::Sprite;
        t.mips.enabled = self.effective(platform).mips;
        t
    }

    /// Does this atlas list `rel` (a texture path) itself or through a folder?
    pub fn lists_texture(&self, rel: &str) -> bool {
        self.packables.iter().any(|p| {
            let (path, _) = split_packable(p);
            let path = path.trim_end_matches('/');
            path == rel || rel.strip_prefix(path).is_some_and(|rest| rest.starts_with('/'))
        })
    }
}

/// `path#id` -> (path, Some(id)).
fn split_packable(p: &str) -> (&str, Option<SubAssetId>) {
    match SpriteRef::from_ref_string(p) {
        Some(r) => (p.rsplit_once('#').map(|(a, _)| a).unwrap_or(p), Some(r.sprite)),
        None => (p, None),
    }
}

// ---------------------------------------------------------------------------
// Members
// ---------------------------------------------------------------------------

/// One sprite texture taking part in an atlas.
#[derive(Clone, Debug)]
pub struct Member {
    /// Project-relative path, forward slashes.
    pub rel: String,
    pub abs: PathBuf,
    pub guid: ObjectId,
    pub width: u32,
    pub height: u32,
    pub sheet: SpriteSheet,
    /// The sprites to pack (a `path#id` packable narrows this).
    pub sprites: Vec<SpriteDef>,
}

fn rel_of(root: &Path, p: &Path) -> String {
    p.strip_prefix(root).unwrap_or(p).to_string_lossy().replace('\\', "/")
}

fn walk_textures(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(read) = std::fs::read_dir(dir) else { return };
    for e in read.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let path = e.path();
        let Ok(ft) = e.file_type() else { continue };
        if ft.is_dir() {
            if !name.starts_with('.') && name != "target" {
                walk_textures(&path, out);
            }
        } else if let Some(stem) = name.strip_suffix(".meta") {
            let tex = path.with_file_name(stem);
            if tex.is_file() {
                out.push(tex);
            }
        }
    }
}

/// The members of `file` under `root`, sorted by path. Textures in `exclude`
/// (claimed by an earlier atlas) and non-Sprite textures are skipped.
pub fn collect_members(root: &Path, file: &SpriteAtlasFile, exclude: &HashSet<String>) -> Vec<Member> {
    // path -> None (all sprites) or the listed ids.
    let mut wanted: BTreeMap<PathBuf, Option<HashSet<SubAssetId>>> = BTreeMap::new();
    for p in &file.packables {
        let (path, id) = split_packable(p);
        let abs = root.join(path.trim_end_matches('/'));
        let mut found = Vec::new();
        if abs.is_dir() {
            walk_textures(&abs, &mut found);
        } else if abs.is_file() {
            found.push(abs);
        }
        for tex in found {
            let slot = wanted.entry(tex).or_insert_with(|| id.map(|_| HashSet::new()));
            match (id, slot.as_mut()) {
                (Some(id), Some(set)) => {
                    set.insert(id);
                }
                (None, _) => *slot = None, // the whole texture wins
                (Some(_), None) => {}
            }
        }
    }
    let mut out = Vec::new();
    for (abs, ids) in wanted {
        let rel = rel_of(root, &abs);
        if exclude.contains(&rel) {
            continue;
        }
        let Ok(Some(table)) = crate::sprite::load_sprite_table(&abs) else { continue };
        let sprites: Vec<SpriteDef> = match &ids {
            Some(set) => table.sprites.iter().filter(|s| set.contains(&s.id)).cloned().collect(),
            None => table.sprites.clone(),
        };
        if sprites.is_empty() {
            continue;
        }
        out.push(Member { rel, abs, guid: table.guid, width: table.width, height: table.height, sheet: table.sheet, sprites });
    }
    out
}

// ---------------------------------------------------------------------------
// Cooked table
// ---------------------------------------------------------------------------

/// Where one sprite lives in a packed atlas. Coordinates: `rect` in master page
/// pixels (a variant scales the pages, not this); `trim`, `polygon`, `mesh` and
/// `physics` in the sprite's own rect-local source pixels (y down).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CookedSprite {
    /// Source texture (project-relative) and sprite id.
    pub texture: String,
    pub id: SubAssetId,
    pub name: String,
    /// Page index (the array layer in [`PageLayout::Array`]).
    pub page: u32,
    /// Placed content on the page (swapped w/h when rotated).
    pub rect: PixelRect,
    /// Content stored turned 90 degrees clockwise.
    pub rotated: bool,
    /// The kept part of the sprite rect (the offset of the trimmed border).
    pub trim: PixelRect,
    /// Full sprite rect size.
    pub size: [u32; 2],
    /// Tight packing mode: alpha polygon(s) inside `trim`.
    pub polygon: Vec<Vec<P2>>,
    /// The render mesh of a Tight sprite (empty = a quad over `trim`).
    pub mesh: Vec<P2>,
    pub mesh_indices: Vec<u32>,
    /// Physics polygons.
    pub physics: Vec<Vec<P2>>,
}

impl CookedSprite {
    /// Atlas UV (page-normalised, v = 0 at the top) of a point in the sprite's
    /// own pixel space (origin top-left of the sprite rect).
    pub fn atlas_uv(&self, p: P2, page: [u32; 2]) -> [f32; 2] {
        let q = [p[0] - self.trim.x as f32, p[1] - self.trim.y as f32];
        let (dx, dy) = if self.rotated { (self.trim.h as f32 - q[1], q[0]) } else { (q[0], q[1]) };
        [(self.rect.x as f32 + dx) / page[0].max(1) as f32, (self.rect.y as f32 + dy) / page[1].max(1) as f32]
    }

    /// The placed content as min u, min v, max u, max v.
    pub fn content_uv_rect(&self, page: [u32; 2]) -> [f32; 4] {
        let (w, h) = (page[0].max(1) as f32, page[1].max(1) as f32);
        [self.rect.x as f32 / w, self.rect.y as f32 / h, (self.rect.x + self.rect.w) as f32 / w, (self.rect.y + self.rect.h) as f32 / h]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CookedAtlasTable {
    pub version: u32,
    pub name: String,
    /// Content hash this was packed from.
    pub key: String,
    pub platform: String,
    pub layout: PageLayout,
    /// Variant scale applied to the stored pages.
    pub scale: f32,
    /// Master page sizes (the stored pages are `scale` times these).
    pub pages: Vec<[u32; 2]>,
    pub sprites: Vec<CookedSprite>,
}

impl Default for CookedAtlasTable {
    fn default() -> Self {
        Self { version: ATLAS_VERSION, name: String::new(), key: String::new(), platform: String::new(), layout: PageLayout::Array, scale: 1.0, pages: Vec::new(), sprites: Vec::new() }
    }
}

impl CookedAtlasTable {
    pub fn find(&self, texture: &str, id: SubAssetId) -> Option<usize> {
        self.sprites.iter().position(|s| s.id == id && s.texture == texture)
    }
}

/// A cooked atlas: the table and the encoded pages (layers).
pub struct CookedAtlas {
    pub table: CookedAtlasTable,
    pub pages: Vec<CompressedTexture>,
}

impl CookedAtlas {
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let table = ron::ser::to_string(&self.table)?.into_bytes();
        let mut b = Vec::new();
        b.extend_from_slice(COOKED_MAGIC);
        b.extend_from_slice(&(table.len() as u32).to_le_bytes());
        b.extend_from_slice(&table);
        b.extend_from_slice(&(self.pages.len() as u32).to_le_bytes());
        for p in &self.pages {
            let bytes = crate::tex_cache::compressed_to_bytes(p);
            b.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            b.extend_from_slice(&bytes);
        }
        Ok(b)
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<CookedAtlas> {
        let mut p = 0usize;
        let mut take = |n: usize| -> Option<&[u8]> {
            let s = bytes.get(p..p.checked_add(n)?)?;
            p += n;
            Some(s)
        };
        if take(8)? != COOKED_MAGIC {
            return None;
        }
        let tl = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
        let table: CookedAtlasTable = ron::de::from_bytes(take(tl)?).ok()?;
        if table.version != ATLAS_VERSION {
            return None;
        }
        let n = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
        let mut pages = Vec::with_capacity(n.min(1024));
        for _ in 0..n {
            let l = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
            pages.push(crate::tex_cache::compressed_from_bytes(take(l)?)?);
        }
        Some(CookedAtlas { table, pages })
    }

    /// Page size in master pixels (what UVs are normalised by).
    pub fn page_dims(&self, page: u32) -> [u32; 2] {
        self.table.pages.get(page as usize).copied().unwrap_or([1, 1])
    }
}

// ---------------------------------------------------------------------------
// Packing
// ---------------------------------------------------------------------------

/// A decoded RGBA8 image.
#[derive(Clone, Debug)]
pub struct Rgba {
    pub w: u32,
    pub h: u32,
    pub px: Vec<u8>,
}

/// Packed pages before encoding: raw RGBA8 plus the table.
pub struct PackedAtlas {
    pub table: CookedAtlasTable,
    /// One RGBA8 buffer per page (master size).
    pub pages: Vec<Rgba>,
}

/// Bounding box of the pixels with alpha > 0 inside `rect`, rect-local. A rect
/// with nothing visible gives 1 x 1 at the origin.
pub fn trim_rect(img: &Rgba, rect: PixelRect) -> PixelRect {
    let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
    for y in 0..rect.h {
        for x in 0..rect.w {
            let i = ((rect.y + y) as usize * img.w as usize + (rect.x + x) as usize) * 4 + 3;
            if img.px[i] > 0 {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
            }
        }
    }
    if x0 == u32::MAX { PixelRect::new(0, 0, 1, 1) } else { PixelRect::new(x0, y0, x1 - x0, y1 - y0) }
}

/// Copy `r` (absolute pixels) out of the image.
fn crop(img: &Rgba, r: PixelRect) -> Vec<u8> {
    let mut out = Vec::with_capacity(r.w as usize * r.h as usize * 4);
    for y in 0..r.h {
        let a = ((r.y + y) as usize * img.w as usize + r.x as usize) * 4;
        out.extend_from_slice(&img.px[a..a + r.w as usize * 4]);
    }
    out
}

/// Turn `w x h` RGBA8 pixels 90 degrees clockwise (result is `h x w`):
/// source (x, y) lands at (h - 1 - y, x).
pub fn rotate_cw(px: &[u8], w: u32, h: u32) -> Vec<u8> {
    let mut out = vec![0u8; px.len()];
    for y in 0..h {
        for x in 0..w {
            let s = (y as usize * w as usize + x as usize) * 4;
            let d = (x as usize * h as usize + (h - 1 - y) as usize) * 4;
            out[d..d + 4].copy_from_slice(&px[s..s + 4]);
        }
    }
    out
}

/// Write `content` (`cw x ch`) at (x, y) and fill the `pad` ring around it by
/// repeating the nearest content texel (clipped to the page).
fn blit_with_ring(page: &mut Rgba, content: &[u8], cw: u32, ch: u32, x: u32, y: u32, pad: u32) {
    let (pw, ph) = (page.w as i64, page.h as i64);
    for ry in -(pad as i64)..(ch as i64 + pad as i64) {
        for rx in -(pad as i64)..(cw as i64 + pad as i64) {
            let (px, py) = (x as i64 + rx, y as i64 + ry);
            if px < 0 || py < 0 || px >= pw || py >= ph {
                continue;
            }
            let (sx, sy) = (rx.clamp(0, cw as i64 - 1) as usize, ry.clamp(0, ch as i64 - 1) as usize);
            let s = (sy * cw as usize + sx) * 4;
            let d = (py as usize * page.w as usize + px as usize) * 4;
            page.px[d..d + 4].copy_from_slice(&content[s..s + 4]);
        }
    }
}

struct PackedSprite {
    member: usize,
    sprite: usize,
    trim: PixelRect,
}

/// Pack `members` into pages. `load` decodes a member's texture (a file in
/// production, a synthetic image in tests). `name` goes into the table.
pub fn pack_members(name: &str, members: &[Member], file: &SpriteAtlasFile, platform: Platform, load: &dyn Fn(&Member) -> Option<Rgba>) -> Result<PackedAtlas, String> {
    let eff = file.effective(platform);
    let images: Vec<Option<Rgba>> = members.iter().map(load).collect();
    let alphas: Vec<Option<AlphaMap>> = images.iter().map(|i| i.as_ref().map(|i| AlphaMap::from_rgba8(i.w, i.h, &i.px))).collect();
    let mut order: Vec<PackedSprite> = Vec::new();
    let mut sizes: Vec<(u32, u32)> = Vec::new();
    for (mi, m) in members.iter().enumerate() {
        let Some(img) = &images[mi] else { return Err(format!("could not read {}", m.rel)) };
        for (si, d) in m.sprites.iter().enumerate() {
            let rect = d.rect.clamped(img.w, img.h);
            if !rect.fits(img.w, img.h) {
                return Err(format!("sprite '{}' of {} is outside the image", d.name, m.rel));
            }
            let trim = if file.trim && d.border == [0; 4] { trim_rect(img, rect) } else { PixelRect::new(0, 0, rect.w, rect.h) };
            sizes.push((trim.w, trim.h));
            order.push(PackedSprite { member: mi, sprite: si, trim });
        }
    }
    let settings = PackSettings {
        padding: eff.padding,
        allow_rotation: eff.allow_rotation,
        max_size: eff.max_size,
        pow2: file.power_of_two,
        uniform_pages: file.layout == PageLayout::Array,
    };
    let res = atlas_pack::pack(&sizes, &settings).map_err(|e| match e {
        atlas_pack::PackError::TooLarge { item, .. } => {
            let it = &order[item];
            let m = &members[it.member];
            format!("sprite '{}' of {} ({}x{} with padding {}) is bigger than the {}x{} page limit", m.sprites[it.sprite].name, m.rel, sizes[item].0, sizes[item].1, eff.padding, eff.max_size, eff.max_size)
        }
        e => e.to_string(),
    })?;
    let mut pages: Vec<Rgba> = res.pages.iter().map(|&(w, h)| Rgba { w, h, px: vec![0u8; w as usize * h as usize * 4] }).collect();
    let mut table = CookedAtlasTable {
        version: ATLAS_VERSION,
        name: name.to_owned(),
        platform: platform.label().to_owned(),
        layout: file.layout,
        scale: eff.scale,
        pages: res.pages.iter().map(|&(w, h)| [w, h]).collect(),
        ..CookedAtlasTable::default()
    };
    for (i, it) in order.iter().enumerate() {
        let m = &members[it.member];
        let d = &m.sprites[it.sprite];
        let img = images[it.member].as_ref().expect("checked above");
        let rect = d.rect.clamped(img.w, img.h);
        let abs = PixelRect::new(rect.x + it.trim.x, rect.y + it.trim.y, it.trim.w, it.trim.h);
        let mut content = crop(img, abs);
        let pl = res.placements[i];
        let (pw, ph) = atlas_pack::placed_size((it.trim.w, it.trim.h), &pl);
        if pl.rotated {
            content = rotate_cw(&content, it.trim.w, it.trim.h);
        }
        blit_with_ring(&mut pages[pl.page], &content, pw, ph, pl.x, pl.y, eff.padding);
        let alpha = alphas[it.member].as_ref();
        let (mesh, tight) = sprite_pixel_mesh(d, &m.sheet, alpha);
        let polygon = if file.packing == PackingMode::Tight {
            alpha
                .map(|a| sprite_mesh::trace_outline(a, rect, m.sheet.alpha_tolerance, 0, sprite_mesh::detail_to_tolerance(m.sheet.outline_detail), true))
                .unwrap_or_default()
                .into_iter()
                .map(|l| l.into_iter().map(|p| clamp_to(p, it.trim)).collect())
                .collect()
        } else {
            Vec::new()
        };
        table.sprites.push(CookedSprite {
            texture: m.rel.clone(),
            id: d.id,
            name: d.name.clone(),
            page: pl.page as u32,
            rect: PixelRect::new(pl.x, pl.y, pw, ph),
            rotated: pl.rotated,
            trim: it.trim,
            size: [rect.w, rect.h],
            polygon,
            mesh: if tight { mesh.verts } else { Vec::new() },
            mesh_indices: if tight { mesh.indices } else { Vec::new() },
            physics: sprite_physics_pixels(d, &m.sheet, alpha),
        });
    }
    Ok(PackedAtlas { table, pages })
}

fn clamp_to(p: P2, r: PixelRect) -> P2 {
    [p[0].clamp(r.x as f32, (r.x + r.w) as f32), p[1].clamp(r.y as f32, (r.y + r.h) as f32)]
}

/// Scale a page for a variant (size rounded, at least 1).
pub fn scale_page(page: &Rgba, scale: f32, srgb: bool) -> Rgba {
    if (scale - 1.0).abs() < 1e-4 {
        return page.clone();
    }
    let (nw, nh) = (((page.w as f32 * scale).round() as u32).max(1), ((page.h as f32 * scale).round() as u32).max(1));
    Rgba { w: nw, h: nh, px: crate::texture_pipeline::resize(&page.px, page.w, page.h, nw, nh, srgb, crate::texture_import::ResizeFilter::Bilinear) }
}

/// Scale (variant) and encode the pages through the texture cook.
pub fn encode_atlas(packed: PackedAtlas, file: &SpriteAtlasFile, platform: Platform, key: &str) -> CookedAtlas {
    let eff = file.effective(platform);
    let import = file.page_import(platform);
    let srgb = import.effective_srgb();
    let keep = if eff.mips { 1 + file.mip_levels.min(MAX_MIP_LEVELS) as usize } else { 1 };
    let mut pages = Vec::with_capacity(packed.pages.len());
    for page in &packed.pages {
        let scaled = scale_page(page, eff.scale, srgb);
        let mut tex = crate::tex_cache::encode_rgba8(scaled.w, scaled.h, scaled.px, &import, platform);
        tex.mips.truncate(keep);
        pages.push(tex);
    }
    let mut table = packed.table;
    table.key = key.to_owned();
    table.scale = eff.scale;
    CookedAtlas { table, pages }
}

// ---------------------------------------------------------------------------
// Geometry through an atlas
// ---------------------------------------------------------------------------

/// A sprite's geometry with UVs into its atlas page.
#[derive(Clone, Debug, PartialEq)]
pub struct AtlasGeometry {
    /// Positions, indices, physics, pivot ... as for an unatlased sprite; `uvs`
    /// are page UVs; `uv_rect` is the untrimmed sprite rect in page UVs (only
    /// meaningful when not `rotated`).
    pub geometry: SpriteGeometry,
    /// Page (array layer) the UVs point into.
    pub page: u32,
    pub rotated: bool,
    /// Page scale of the cooked atlas (variants), 1 = full size.
    pub scale: f32,
}

/// The geometry of `def` drawn from its atlas entry `cs`. Needs no pixels: the
/// mesh and physics were cooked into `cs`.
pub fn atlas_geometry(def: &SpriteDef, sheet: &SpriteSheet, cs: &CookedSprite, page: [u32; 2], scale: f32) -> AtlasGeometry {
    let ppu = def.effective_ppu(sheet);
    let pivot = def.pivot_norm();
    let rect = def.rect;
    let t = cs.trim;
    let (mut verts, indices): (Vec<P2>, Vec<u32>) = if !cs.mesh.is_empty() && def.effective_mesh_type(sheet) == SpriteMeshType::Tight && def.custom_outline.is_empty() {
        (cs.mesh.clone(), cs.mesh_indices.clone())
    } else if !def.custom_outline.is_empty() {
        match sprite_mesh::mesh_from_loops(&def.custom_outline, true) {
            Some(m) => (m.verts, m.indices),
            None => (Vec::new(), Vec::new()),
        }
    } else {
        (Vec::new(), Vec::new())
    };
    let tight = !verts.is_empty();
    let indices = if tight {
        indices
    } else {
        let (x0, y0, x1, y1) = (t.x as f32, t.y as f32, (t.x + t.w) as f32, (t.y + t.h) as f32);
        verts = vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
        vec![0, 1, 2, 0, 2, 3]
    };
    for v in &mut verts {
        *v = clamp_to(*v, t);
    }
    let positions: Vec<[f32; 2]> = verts.iter().map(|v| pixel_to_world(*v, rect, pivot, ppu)).collect();
    let uvs: Vec<[f32; 2]> = verts.iter().map(|v| cs.atlas_uv(*v, page)).collect();
    let indices: Vec<u16> = indices.as_chunks::<3>().0.iter().flat_map(|t| [t[0] as u16, t[2] as u16, t[1] as u16]).collect();
    let physics = cs.physics.iter().map(|poly| poly.iter().map(|p| pixel_to_world(*p, rect, pivot, ppu)).collect()).collect();
    let [l, b, r, tp] = def.border_clamped();
    let a = cs.atlas_uv([0.0, 0.0], page);
    let z = cs.atlas_uv([rect.w as f32, rect.h as f32], page);
    AtlasGeometry {
        geometry: SpriteGeometry {
            id: def.id,
            name: def.name.clone(),
            rect,
            ppu,
            size_world: def.size_world(sheet),
            pivot_world: def.pivot_world(sheet),
            border_world: [l as f32 / ppu, b as f32 / ppu, r as f32 / ppu, tp as f32 / ppu],
            uv_rect: [a[0].min(z[0]), a[1].min(z[1]), a[0].max(z[0]), a[1].max(z[1])],
            positions,
            uvs,
            indices,
            tight,
            physics,
        },
        page: cs.page,
        rotated: cs.rotated,
        scale,
    }
}

// ---------------------------------------------------------------------------
// Cache and the project database
// ---------------------------------------------------------------------------

/// Hash of everything a packed atlas depends on: version, platform, the atlas
/// settings, the page encoding and the **content** of every member plus its
/// sprite table. Unrelated changes (a comment, another atlas) leave it alone.
pub fn atlas_key(file: &SpriteAtlasFile, platform: Platform, members: &[Member], read: &dyn Fn(&Member) -> Option<Vec<u8>>) -> String {
    let mut s = format!("v{ATLAS_VERSION}|{}|{}|", platform.label(), crate::tex_cache::import_key(&file.page_import(platform), platform));
    s.push_str(&ron::ser::to_string(file).unwrap_or_default());
    for m in members {
        let content = read(m).map(|b| crate::package::sha256_hex(&b)).unwrap_or_default();
        let sprites = ron::ser::to_string(&(&m.sheet, &m.sprites)).unwrap_or_default();
        s.push_str(&format!("|{}|{content}|{sprites}", m.rel));
    }
    crate::package::sha256_hex(s.as_bytes())[..32].to_owned()
}

/// `<dir of atlas>/.zestengine_atlascache/<file>.<platform>.<key>.catlas`.
pub fn cache_path(atlas_abs: &Path, platform: Platform, key: &str) -> Option<PathBuf> {
    let dir = atlas_abs.parent()?.join(ATLAS_CACHE_DIR);
    let name = atlas_abs.file_name()?.to_string_lossy();
    Some(dir.join(format!("{name}.{}.{key}.{COOKED_ATLAS_EXTENSION}", platform.label())))
}

fn prune_old(path: &Path) {
    // Keep only the newest key for this atlas and platform.
    let (Some(dir), Some(name)) = (path.parent(), path.file_name().map(|n| n.to_string_lossy().into_owned())) else { return };
    let Some(prefix) = name.rsplitn(3, '.').nth(2).map(|p| format!("{p}.")) else { return };
    let Ok(read) = std::fs::read_dir(dir) else { return };
    for e in read.flatten() {
        let n = e.file_name().to_string_lossy().into_owned();
        if n != name && n.starts_with(&prefix) && n.ends_with(COOKED_ATLAS_EXTENSION) {
            let _ = std::fs::remove_file(e.path());
        }
    }
}

fn read_file_image(m: &Member) -> Option<Rgba> {
    let img = image::open(&m.abs).ok()?.to_rgba8();
    Some(Rgba { w: img.width(), h: img.height(), px: img.into_raw() })
}

/// One atlas of the project with its (claimed) members.
pub struct AtlasEntry {
    /// Project-relative path of the `.spriteatlas`.
    pub rel: String,
    pub abs: PathBuf,
    pub name: String,
    pub file: SpriteAtlasFile,
    pub members: Vec<Member>,
}

/// What a cook / auto-pack did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CookReport {
    pub packed: usize,
    pub from_cache: usize,
    pub warnings: Vec<String>,
}

/// Every atlas of a project, who owns which texture, and a memo of cooked
/// atlases per platform. Build with [`AtlasDb::scan`]; rescan when atlas files
/// or sprite tables change.
pub struct AtlasDb {
    pub root: PathBuf,
    pub atlases: Vec<AtlasEntry>,
    /// Texture path -> index into `atlases`.
    pub owner: HashMap<String, usize>,
    memo: Mutex<HashMap<(usize, u8), Arc<CookedAtlas>>>,
}

fn find_atlas_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(read) = std::fs::read_dir(dir) else { return };
    for e in read.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let path = e.path();
        let Ok(ft) = e.file_type() else { continue };
        if ft.is_dir() {
            if !name.starts_with('.') && name != "target" {
                find_atlas_files(&path, out);
            }
        } else if path.extension().is_some_and(|x| x.eq_ignore_ascii_case(ATLAS_EXTENSION)) {
            out.push(path);
        }
    }
}

impl AtlasDb {
    pub fn empty(root: &Path) -> Self {
        Self { root: root.to_path_buf(), atlases: Vec::new(), owner: HashMap::new(), memo: Mutex::new(HashMap::new()) }
    }

    /// Find and load every `.spriteatlas` under `root` (sorted by path; the
    /// first atlas to list a texture owns it).
    pub fn scan(root: &Path) -> Self {
        let mut files = Vec::new();
        find_atlas_files(root, &mut files);
        files.sort();
        let mut db = Self::empty(root);
        let mut claimed: HashSet<String> = HashSet::new();
        for abs in files {
            let Ok(file) = SpriteAtlasFile::load(&abs) else { continue };
            let members = collect_members(root, &file, &claimed);
            let idx = db.atlases.len();
            for m in &members {
                claimed.insert(m.rel.clone());
                db.owner.insert(m.rel.clone(), idx);
            }
            let name = abs.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            db.atlases.push(AtlasEntry { rel: rel_of(root, &abs), abs, name, file, members });
        }
        db
    }

    /// Name of the atlas packing `texture_rel`, for the "in atlas" badge.
    pub fn atlas_name_of(&self, texture_rel: &str) -> Option<&str> {
        self.owner.get(texture_rel).map(|&i| self.atlases[i].name.as_str())
    }

    /// Pack (or load from the cache) atlas `idx` for `platform`. Returns the
    /// cooked atlas and whether it came from the cache.
    pub fn cook(&self, idx: usize, platform: Platform) -> Result<(Arc<CookedAtlas>, bool)> {
        let memo_key = (idx, platform as u8);
        if let Some(a) = self.memo.lock().ok().and_then(|m| m.get(&memo_key).cloned()) {
            return Ok((a, true));
        }
        let Some(e) = self.atlases.get(idx) else { bail!("no atlas {idx}") };
        let read = |m: &Member| std::fs::read(&m.abs).ok();
        let key = atlas_key(&e.file, platform, &e.members, &read);
        let path = cache_path(&e.abs, platform, &key);
        if let Some(p) = &path
            && let Ok(bytes) = std::fs::read(p)
            && let Some(a) = CookedAtlas::from_bytes(&bytes)
            && a.table.key == key
        {
            let a = Arc::new(a);
            self.memoize(memo_key, a.clone());
            return Ok((a, true));
        }
        let packed = pack_members(&e.name, &e.members, &e.file, platform, &read_file_image).map_err(anyhow::Error::msg)?;
        let cooked = encode_atlas(packed, &e.file, platform, &key);
        if let Some(p) = &path {
            match store(p, &cooked) {
                Ok(()) => prune_old(p),
                Err(err) => tracing::debug!("atlas cache write {}: {err:#}", p.display()),
            }
        }
        let a = Arc::new(cooked);
        self.memoize(memo_key, a.clone());
        Ok((a, false))
    }

    fn memoize(&self, k: (usize, u8), a: Arc<CookedAtlas>) {
        if let Ok(mut m) = self.memo.lock() {
            m.insert(k, a);
        }
    }

    /// Pack every atlas (`only_in_build`: just those with `include_in_build`).
    /// This is the game-pak cook and, with `false`, the editor's pack on Play.
    pub fn cook_all(&self, platform: Platform, only_in_build: bool) -> CookReport {
        let mut r = CookReport::default();
        for (i, e) in self.atlases.iter().enumerate() {
            if only_in_build && !e.file.include_in_build {
                continue;
            }
            if e.members.is_empty() {
                continue;
            }
            match self.cook(i, platform) {
                Ok((_, true)) => r.from_cache += 1,
                Ok((_, false)) => r.packed += 1,
                Err(err) => r.warnings.push(format!("{}: {err:#}", e.rel)),
            }
        }
        r
    }

    /// Resolve a sprite reference through its atlas: the cooked atlas and the
    /// index of the sprite in its table. `None` when the sprite's texture is in
    /// no atlas (draw it from its own texture).
    pub fn resolve(&self, platform: Platform, r: &SpriteRef) -> Option<AtlasHit> {
        let idx = self.owner_of(r)?;
        let e = &self.atlases[idx];
        let m = e.members.iter().find(|m| self.matches(m, r))?;
        let si = r.find_in(&m.sprites)?;
        let def = m.sprites[si].clone();
        let (atlas, _) = self.cook(idx, platform).ok()?;
        let index = atlas.table.find(&m.rel, def.id)?;
        Some(AtlasHit { atlas, index, def, sheet: m.sheet.clone() })
    }

    fn matches(&self, m: &Member, r: &SpriteRef) -> bool {
        match (r.texture, r.path.is_empty()) {
            (Some(g), _) if g == m.guid => true,
            (_, false) => m.rel == r.path.replace('\\', "/"),
            _ => false,
        }
    }

    fn owner_of(&self, r: &SpriteRef) -> Option<usize> {
        if !r.path.is_empty()
            && let Some(&i) = self.owner.get(&r.path.replace('\\', "/"))
            && self.atlases[i].members.iter().any(|m| self.matches(m, r))
        {
            return Some(i);
        }
        let g = r.texture?;
        self.atlases.iter().position(|e| e.members.iter().any(|m| m.guid == g))
    }
}

/// A sprite found in a cooked atlas.
pub struct AtlasHit {
    pub atlas: Arc<CookedAtlas>,
    pub index: usize,
    pub def: SpriteDef,
    pub sheet: SpriteSheet,
}

impl AtlasHit {
    pub fn geometry(&self) -> AtlasGeometry {
        let cs = &self.atlas.table.sprites[self.index];
        atlas_geometry(&self.def, &self.sheet, cs, self.atlas.page_dims(cs.page), self.atlas.table.scale)
    }
}

/// A sprite drawn as one quad from its atlas page (tilemap tiles): the kept
/// (trimmed) content only, so transparent borders cost no fill.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AtlasQuad {
    /// Page = array layer.
    pub page: u32,
    /// The content is stored turned 90 degrees clockwise.
    pub rotated: bool,
    /// Placed content in page UVs: min u, min v, max u, max v (v = 0 at the top).
    pub uv_rect: [f32; 4],
    /// Local rect of the content relative to the pivot, world units: min x, min y, size x, size y.
    pub rect: [f32; 4],
}

/// [`AtlasHit::quad`] without the lookup, for tests.
pub fn atlas_quad(def: &SpriteDef, sheet: &SpriteSheet, cs: &CookedSprite, page: [u32; 2]) -> AtlasQuad {
    let (ppu, pivot, rect, t) = (def.effective_ppu(sheet), def.pivot_norm(), def.rect, cs.trim);
    let lo = pixel_to_world([t.x as f32, (t.y + t.h) as f32], rect, pivot, ppu);
    let hi = pixel_to_world([(t.x + t.w) as f32, t.y as f32], rect, pivot, ppu);
    AtlasQuad { page: cs.page, rotated: cs.rotated, uv_rect: cs.content_uv_rect(page), rect: [lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1]] }
}

impl AtlasHit {
    /// The sprite as a single trimmed quad.
    pub fn quad(&self) -> AtlasQuad {
        let cs = &self.atlas.table.sprites[self.index];
        atlas_quad(&self.def, &self.sheet, cs, self.atlas.page_dims(cs.page))
    }
}

fn store(path: &Path, a: &CookedAtlas) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let bytes = a.to_bytes()?;
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Game-pak cook: pack every in-build atlas under `assets_dir` for the desktop
/// so the archive carries `.zestengine_atlascache/` beside each atlas.
pub fn cook_atlases(assets_dir: &Path, platform: Platform) -> CookReport {
    AtlasDb::scan(assets_dir).cook_all(platform, true)
}

// ---------------------------------------------------------------------------
// Shipping: builds, Android / Quest and content packages
// ---------------------------------------------------------------------------

/// Asset-root folder an Android / Quest build ships cooked atlases in, as
/// `<folder>/<atlas path>.catlas` (not the hidden `.zestengine_atlascache/`,
/// whose dot-prefixed path the APK packager drops, like `geo_cache` and
/// `package::COOKED_TEXCACHE_DIR`). [`seed_cooked_atlases`] moves them into
/// place on device.
pub const COOKED_ATLAS_DIR: &str = "zestengine_atlascache";

/// Shipped path of an atlas's cooked file (see [`COOKED_ATLAS_DIR`]).
pub fn cooked_atlas_rel(atlas_rel: &str) -> String {
    format!("{COOKED_ATLAS_DIR}/{atlas_rel}.{COOKED_ATLAS_EXTENSION}")
}

/// Inverse of [`cooked_atlas_rel`]: the atlas path.
pub fn parse_cooked_atlas_rel(rel: &str) -> Option<&str> {
    let rest = rel.strip_prefix(COOKED_ATLAS_DIR)?.strip_prefix('/')?;
    let atlas = rest.strip_suffix(COOKED_ATLAS_EXTENSION)?.strip_suffix('.')?;
    (crate::package::normalize_rel(atlas).as_deref() == Some(atlas) && atlas.ends_with(ATLAS_EXTENSION)).then_some(atlas)
}

fn platform_from_label(label: &str) -> Option<Platform> {
    [Platform::Desktop, Platform::Quest].into_iter().find(|p| p.label() == label)
}

impl AtlasDb {
    /// What a build must ship for the atlases that go in it: each `.spriteatlas`
    /// with `include_in_build` and members, its member textures and their `.meta`
    /// (the sprite tables a [`SpriteRef`] resolves through). Scenes never name an
    /// atlas, so the build cook adds these to what it reached. The member
    /// textures stay in the build even if no scene uses them (small cost).
    pub fn build_files(&self) -> Vec<String> {
        let mut out = Vec::new();
        for e in self.atlases.iter().filter(|e| e.file.include_in_build && !e.members.is_empty()) {
            out.push(e.rel.clone());
            for m in &e.members {
                out.push(m.rel.clone());
                let meta = crate::meta_path(&m.rel).to_string_lossy().replace('\\', "/");
                if self.root.join(&meta).is_file() {
                    out.push(meta);
                }
            }
        }
        out.sort();
        out.dedup();
        out
    }

    /// Cook (or read from the cache) every in-build atlas whose atlas file and
    /// member textures all satisfy `available`, for `platform`, as
    /// `(atlas path, cooked bytes)`; the rest are reported in the warnings.
    pub fn cooked_blobs(&self, platform: Platform, available: &dyn Fn(&str) -> bool) -> (Vec<(String, Vec<u8>)>, Vec<String>) {
        let (mut out, mut warnings) = (Vec::new(), Vec::new());
        for (i, e) in self.atlases.iter().enumerate() {
            if !e.file.include_in_build || e.members.is_empty() {
                continue;
            }
            if !available(&e.rel) || e.members.iter().any(|m| !available(&m.rel)) {
                warnings.push(format!("{} not cooked: its atlas file or some member textures are not part of this build", e.rel));
                continue;
            }
            match self.cook(i, platform).and_then(|(a, _)| a.to_bytes()) {
                Ok(bytes) => out.push((e.rel.clone(), bytes)),
                Err(err) => warnings.push(format!("{}: {err:#}", e.rel)),
            }
        }
        (out, warnings)
    }
}

/// Put a cooked atlas (`bytes`, from [`CookedAtlas::to_bytes`]) where the next
/// [`AtlasDb::cook`] of `atlas_abs` finds it, keyed by the table's own key and
/// platform. A cache whose key no longer matches the files is simply ignored by
/// the cook (and repacked), so installing is always safe. Returns the cache path.
pub fn install_cooked_atlas(atlas_abs: &Path, bytes: &[u8]) -> Result<PathBuf> {
    let Some(a) = CookedAtlas::from_bytes(bytes) else { bail!("not a cooked atlas (or from another engine version)") };
    let Some(platform) = platform_from_label(&a.table.platform) else { bail!("unknown platform {:?}", a.table.platform) };
    if a.pages.is_empty() || a.table.pages.len() != a.pages.len() {
        bail!("page table and pages disagree");
    }
    if a.table.pages.iter().any(|p| p[0] == 0 || p[1] == 0 || p[0].max(p[1]) > 16384) {
        bail!("page size out of range");
    }
    if a.table.key.is_empty() || !a.table.key.bytes().all(|b| b.is_ascii_hexdigit()) {
        bail!("bad atlas key");
    }
    if a.table.sprites.iter().any(|s| s.page as usize >= a.pages.len()) {
        bail!("a sprite points at a missing page");
    }
    let Some(path) = cache_path(atlas_abs, platform, &a.table.key) else { bail!("no cache folder for {}", atlas_abs.display()) };
    store(&path, &a)?;
    prune_old(&path);
    Ok(path)
}

/// Install every cooked atlas under `assets_root/`[`COOKED_ATLAS_DIR`] beside
/// its `.spriteatlas`, then delete the shipped copies (the Android counterpart
/// of `geo_cache::seed_cooked_geocaches`). Returns (installed, warnings).
pub fn seed_cooked_atlases(assets_root: &Path) -> (usize, Vec<String>) {
    fn walk(base: &Path, dir: &Path, out: &mut Vec<String>) {
        let Ok(read) = std::fs::read_dir(dir) else { return };
        for e in read.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(base, &p, out);
            } else if let Ok(rel) = p.strip_prefix(base) {
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    let dir = assets_root.join(COOKED_ATLAS_DIR);
    let mut found = Vec::new();
    walk(&dir, &dir, &mut found);
    let (mut seeded, mut warnings) = (0, Vec::new());
    for shipped in found {
        let rel = format!("{COOKED_ATLAS_DIR}/{shipped}");
        let Some(atlas) = parse_cooked_atlas_rel(&rel) else {
            warnings.push(format!("ignoring unknown cooked atlas {rel}"));
            continue;
        };
        match std::fs::read(dir.join(&shipped)).map_err(anyhow::Error::from).and_then(|b| install_cooked_atlas(&assets_root.join(atlas), &b)) {
            Ok(_) => seeded += 1,
            Err(e) => warnings.push(format!("cooked atlas {atlas}: {e:#}")),
        }
    }
    if dir.exists()
        && let Err(e) = std::fs::remove_dir_all(&dir)
    {
        warnings.push(format!("removing {}: {e}", dir.display()));
    }
    (seeded, warnings)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::sprite::SpriteMode;

    /// A deterministic coloured image: every pixel tells where it came from.
    fn img(w: u32, h: u32, seed: u8, opaque: impl Fn(u32, u32) -> bool) -> Rgba {
        let mut px = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                if opaque(x, y) {
                    let i = ((y * w + x) * 4) as usize;
                    px[i..i + 4].copy_from_slice(&[seed.wrapping_add((x as u8).wrapping_mul(3)), seed.wrapping_add((y as u8).wrapping_mul(5)), seed, 255]);
                }
            }
        }
        Rgba { w, h, px }
    }

    fn member(rel: &str, w: u32, h: u32, rects: &[(&str, PixelRect)]) -> Member {
        let mut sheet = SpriteSheet { mode: SpriteMode::Multiple, ..SpriteSheet::default() };
        for (n, r) in rects {
            sheet.sprites.push(SpriteDef::named(n, *r));
        }
        sheet.ensure_ids(rel);
        let sprites = sheet.resolve(w, h, rel);
        Member { rel: rel.into(), abs: PathBuf::from(rel), guid: ObjectId::NIL, width: w, height: h, sheet, sprites }
    }

    fn atlas_file() -> SpriteAtlasFile {
        SpriteAtlasFile { padding: 2, max_size: 256, ..SpriteAtlasFile::default() }
    }

    /// Two textures with trimmed sprites; images keyed by member path.
    fn fixture() -> (Vec<Member>, HashMap<String, Rgba>) {
        let a = member("a.png", 64, 32, &[("a0", PixelRect::new(0, 0, 32, 32)), ("a1", PixelRect::new(32, 0, 32, 32))]);
        let b = member("b.png", 40, 40, &[("b0", PixelRect::new(0, 0, 40, 40))]);
        let mut imgs = HashMap::new();
        // a0: opaque 10x20 block at (5,4); a1: full; b0: an L shape.
        imgs.insert("a.png".to_string(), {
            img(64, 32, 10, |x, y| x >= 32 || ((5..15).contains(&x) && (4..24).contains(&y)))
        });
        imgs.insert("b.png".to_string(), img(40, 40, 90, |x, y| x < 10 || y >= 30));
        (vec![a, b], imgs)
    }

    fn loader(imgs: &HashMap<String, Rgba>) -> impl Fn(&Member) -> Option<Rgba> + '_ {
        move |m| imgs.get(&m.rel).cloned()
    }

    /// The source pixel of sprite-local (x, y), or transparent outside the image.
    fn src_px(m: &Member, imgs: &HashMap<String, Rgba>, def: &SpriteDef, x: u32, y: u32) -> [u8; 4] {
        let i = &imgs[&m.rel];
        let o = (((def.rect.y + y) * i.w + def.rect.x + x) * 4) as usize;
        i.px[o..o + 4].try_into().unwrap()
    }

    /// Sample the packed page through the sprite's UV mapping (nearest).
    fn sample(p: &PackedAtlas, cs: &CookedSprite, x: u32, y: u32) -> [u8; 4] {
        let page = &p.pages[cs.page as usize];
        let uv = cs.atlas_uv([x as f32 + 0.5, y as f32 + 0.5], p.table.pages[cs.page as usize]);
        let (px, py) = ((uv[0] * page.w as f32).floor() as usize, (uv[1] * page.h as f32).floor() as usize);
        let o = (py * page.w as usize + px) * 4;
        page.px[o..o + 4].try_into().unwrap()
    }

    #[test]
    fn packed_sprites_sample_back_to_the_source_pixels() {
        let (members, imgs) = fixture();
        for rotation in [false, true] {
            let file = SpriteAtlasFile { allow_rotation: rotation, ..atlas_file() };
            let p = pack_members("t", &members, &file, Platform::Desktop, &loader(&imgs)).unwrap();
            assert_eq!(p.table.sprites.len(), 3);
            for m in &members {
                for d in &m.sprites {
                    let cs = p.table.sprites[p.table.find(&m.rel, d.id).unwrap()].clone();
                    for y in 0..cs.size[1] {
                        for x in 0..cs.size[0] {
                            let inside = x >= cs.trim.x && x < cs.trim.x + cs.trim.w && y >= cs.trim.y && y < cs.trim.y + cs.trim.h;
                            if inside {
                                assert_eq!(sample(&p, &cs, x, y), src_px(m, &imgs, d, x, y), "{} ({x},{y}) rotated={}", d.name, cs.rotated);
                            } else {
                                assert_eq!(src_px(m, &imgs, d, x, y)[3], 0, "trimmed area must be transparent");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn rotation_is_used_when_needed_and_samples_back_exactly() {
        // 60x30 and 30x60 only share a 64 page if one of them is turned.
        let a = member("a.png", 60, 30, &[("wide", PixelRect::new(0, 0, 60, 30))]);
        let b = member("b.png", 30, 60, &[("tall", PixelRect::new(0, 0, 30, 60))]);
        let imgs = HashMap::from([("a.png".to_string(), img(60, 30, 1, |_, _| true)), ("b.png".to_string(), img(30, 60, 7, |_, _| true))]);
        let file = SpriteAtlasFile { allow_rotation: true, pixel_art: true, padding: 0, max_size: 64, power_of_two: false, ..SpriteAtlasFile::default() };
        let members = vec![a, b];
        let p = pack_members("t", &members, &file, Platform::Desktop, &loader(&imgs)).unwrap();
        assert_eq!(p.pages.len(), 1);
        assert!(p.table.sprites.iter().any(|s| s.rotated), "one sprite had to turn");
        for m in &members {
            let d = &m.sprites[0];
            let cs = &p.table.sprites[p.table.find(&m.rel, d.id).unwrap()];
            for y in 0..cs.size[1] {
                for x in 0..cs.size[0] {
                    assert_eq!(sample(&p, cs, x, y), src_px(m, &imgs, d, x, y), "{} ({x},{y})", d.name);
                }
            }
        }
        // rotate_cw on its own: (x, y) -> (h - 1 - y, x).
        let src: Vec<u8> = (0..6u8).flat_map(|i| [i, 0, 0, 255]).collect(); // 3 wide, 2 tall
        let r = rotate_cw(&src, 3, 2);
        // result is 2 wide, 3 tall; source (0,0) -> (1,0); source (2,1) -> (0,2)
        assert_eq!(r[4], 0);
        assert_eq!(r[4 * 4], 5);
        assert_eq!(rotate_cw(&rotate_cw(&rotate_cw(&rotate_cw(&src, 3, 2), 2, 3), 3, 2), 2, 3), src);
    }

    #[test]
    fn padding_ring_repeats_edge_texels_and_stays_clear_of_neighbours() {
        let (members, imgs) = fixture();
        let file = SpriteAtlasFile { padding: 3, ..atlas_file() };
        let p = pack_members("t", &members, &file, Platform::Desktop, &loader(&imgs)).unwrap();
        let pad = file.effective(Platform::Desktop).padding as i64;
        for cs in &p.table.sprites {
            let page = &p.pages[cs.page as usize];
            let at = |x: i64, y: i64| -> [u8; 4] {
                let o = ((y as usize) * page.w as usize + x as usize) * 4;
                page.px[o..o + 4].try_into().unwrap()
            };
            let r = cs.rect;
            // The texel just left of the content equals the content's left edge texel.
            if r.x as i64 > 0 {
                assert_eq!(at(r.x as i64 - 1, r.y as i64), at(r.x as i64, r.y as i64));
            }
            // And the corner of the ring repeats the corner texel.
            let (cx, cy) = (r.x as i64 + r.w as i64 - 1, r.y as i64 + r.h as i64 - 1);
            if cx + pad < page.w as i64 && cy + pad < page.h as i64 {
                assert_eq!(at(cx + pad, cy + pad), at(cx, cy));
            }
        }
        // Padding is at least what was asked.
        for (i, a) in p.table.sprites.iter().enumerate() {
            for b in &p.table.sprites[i + 1..] {
                if a.page != b.page {
                    continue;
                }
                let gx = (b.rect.x as i64 - (a.rect.x + a.rect.w) as i64).max(a.rect.x as i64 - (b.rect.x + b.rect.w) as i64);
                let gy = (b.rect.y as i64 - (a.rect.y + a.rect.h) as i64).max(a.rect.y as i64 - (b.rect.y + b.rect.h) as i64);
                assert!(gx.max(gy) >= 2 * pad, "{gx},{gy}");
            }
        }
    }

    #[test]
    fn trim_offsets_are_recorded_and_borders_disable_trimming() {
        let (mut members, imgs) = fixture();
        let p = pack_members("t", &members, &atlas_file(), Platform::Desktop, &loader(&imgs)).unwrap();
        let a0 = &p.table.sprites[p.table.find("a.png", members[0].sprites[0].id).unwrap()];
        assert_eq!(a0.trim, PixelRect::new(5, 4, 10, 20));
        assert_eq!(a0.size, [32, 32]);
        let a1 = &p.table.sprites[p.table.find("a.png", members[0].sprites[1].id).unwrap()];
        assert_eq!(a1.trim, PixelRect::new(0, 0, 32, 32));
        // A 9-sliced sprite keeps its full rect.
        members[0].sprites[0].border = [2, 2, 2, 2];
        let p = pack_members("t", &members, &atlas_file(), Platform::Desktop, &loader(&imgs)).unwrap();
        let a0 = &p.table.sprites[p.table.find("a.png", members[0].sprites[0].id).unwrap()];
        assert_eq!(a0.trim, PixelRect::new(0, 0, 32, 32));
        // Trim off keeps everything.
        let p = pack_members("t", &members, &SpriteAtlasFile { trim: false, ..atlas_file() }, Platform::Desktop, &loader(&imgs)).unwrap();
        assert!(p.table.sprites.iter().all(|s| s.trim.x == 0 && s.trim.y == 0 && [s.trim.w, s.trim.h] == s.size));
    }

    #[test]
    fn tight_mode_records_polygons_inside_the_trim_rect() {
        let (members, imgs) = fixture();
        let file = SpriteAtlasFile { packing: PackingMode::Tight, ..atlas_file() };
        let p = pack_members("t", &members, &file, Platform::Desktop, &loader(&imgs)).unwrap();
        let b0 = &p.table.sprites[p.table.find("b.png", members[1].sprites[0].id).unwrap()];
        assert!(!b0.polygon.is_empty());
        let t = b0.trim;
        for l in &b0.polygon {
            for q in l {
                assert!(q[0] >= t.x as f32 && q[0] <= (t.x + t.w) as f32 && q[1] >= t.y as f32 && q[1] <= (t.y + t.h) as f32, "{q:?} outside {t:?}");
            }
        }
        // Rectangle mode records none.
        let p = pack_members("t", &members, &atlas_file(), Platform::Desktop, &loader(&imgs)).unwrap();
        assert!(p.table.sprites.iter().all(|s| s.polygon.is_empty()));
    }

    #[test]
    fn deterministic_and_all_sprites_placed_or_a_clear_error() {
        let (members, imgs) = fixture();
        let a = pack_members("t", &members, &atlas_file(), Platform::Desktop, &loader(&imgs)).unwrap();
        let b = pack_members("t", &members, &atlas_file(), Platform::Desktop, &loader(&imgs)).unwrap();
        assert_eq!(a.table, b.table);
        assert_eq!(a.pages.iter().map(|p| &p.px).collect::<Vec<_>>(), b.pages.iter().map(|p| &p.px).collect::<Vec<_>>());
        assert_eq!(a.table.sprites.len(), members.iter().map(|m| m.sprites.len()).sum::<usize>());
        // Too small a page is a readable error naming the sprite.
        let err = pack_members("t", &members, &SpriteAtlasFile { max_size: 32, ..atlas_file() }, Platform::Desktop, &loader(&imgs)).err().unwrap();
        assert!(err.contains("bigger than") && err.contains(".png"), "{err}");
        // An unreadable texture is an error too.
        let none = HashMap::new();
        assert!(pack_members("t", &members, &atlas_file(), Platform::Desktop, &loader(&none)).err().unwrap().contains("could not read"));
    }

    #[test]
    fn array_layout_gives_equal_pages_separate_does_not_need_to() {
        let mut members = Vec::new();
        let mut imgs = HashMap::new();
        for i in 0..8 {
            let rel = format!("s{i}.png");
            members.push(member(&rel, 60, 60, &[("s", PixelRect::new(0, 0, 60, 60))]));
            imgs.insert(rel, img(60, 60, i as u8, |_, _| true));
        }
        let file = SpriteAtlasFile { max_size: 128, padding: 1, ..SpriteAtlasFile::default() };
        let p = pack_members("t", &members, &file, Platform::Desktop, &loader(&imgs)).unwrap();
        assert!(p.table.pages.len() >= 2);
        assert!(p.table.pages.iter().all(|d| *d == p.table.pages[0]));
        let sep = SpriteAtlasFile { layout: PageLayout::Separate, ..file };
        let p = pack_members("t", &members, &sep, Platform::Desktop, &loader(&imgs)).unwrap();
        assert!(p.table.pages.len() >= 2);
        assert_eq!(p.pages.len(), p.table.pages.len());
    }

    #[test]
    fn mips_raise_padding_and_pixel_art_turns_them_off() {
        let mut f = SpriteAtlasFile::default();
        f.texture.mips.enabled = true;
        f.mip_levels = 3;
        f.padding = 2;
        let e = f.effective(Platform::Desktop);
        assert!(e.mips && e.padding == 8, "{e:?}");
        f.mip_levels = 0;
        assert_eq!(f.effective(Platform::Desktop).padding, 2);
        f.pixel_art = true;
        f.mip_levels = 4;
        let e = f.effective(Platform::Desktop);
        assert!(!e.mips && e.padding == 2);
        assert!(!f.page_import(Platform::Desktop).mips.enabled);
        f.padding = 0;
        assert_eq!(f.effective(Platform::Desktop).padding, 0, "point-filtered pixel art needs no margin");
    }

    #[test]
    fn platform_overrides_and_variant_scaling() {
        let mut f = SpriteAtlasFile { max_size: 2048, padding: 4, ..SpriteAtlasFile::default() };
        f.variants.push(AtlasVariant { name: "Half".into(), scale: 0.5 });
        f.quest = Some(AtlasPlatform { max_size: Some(1024), padding: Some(2), allow_rotation: Some(true), variant: Some("Half".into()) });
        let d = f.effective(Platform::Desktop);
        assert_eq!((d.max_size, d.padding, d.allow_rotation, d.scale), (2048, 4, false, 1.0));
        let q = f.effective(Platform::Quest);
        assert_eq!((q.max_size, q.padding, q.allow_rotation, q.scale), (1024, 2, true, 0.5));
        assert_eq!(q.variant.as_deref(), Some("Half"));
        // An unknown variant name falls back to full size.
        f.quest.as_mut().unwrap().variant = Some("Nope".into());
        assert_eq!(f.effective(Platform::Quest).scale, 1.0);
        // Scaling a page halves its size and keeps UVs valid.
        let page = img(64, 32, 3, |_, _| true);
        let s = scale_page(&page, 0.5, true);
        assert_eq!((s.w, s.h), (32, 16));
        assert_eq!(scale_page(&page, 1.0, true).px, page.px);
        // The layout (hence every UV) does not depend on the variant.
        let (members, imgs) = fixture();
        let full = SpriteAtlasFile { padding: 2, max_size: 256, ..SpriteAtlasFile::default() };
        let mut half = full.clone();
        half.variants.push(AtlasVariant { name: "H".into(), scale: 0.5 });
        half.desktop = Some(AtlasPlatform { variant: Some("H".into()), ..Default::default() });
        let a = pack_members("t", &members, &full, Platform::Desktop, &loader(&imgs)).unwrap();
        let b = pack_members("t", &members, &half, Platform::Desktop, &loader(&imgs)).unwrap();
        assert_eq!(a.table.sprites, b.table.sprites);
        assert_eq!(a.table.pages, b.table.pages);
        assert_eq!(b.table.scale, 0.5);
    }

    #[test]
    fn cache_key_tracks_content_settings_platform_and_nothing_else() {
        let (members, _) = fixture();
        let read_a = |m: &Member| Some(m.rel.as_bytes().to_vec());
        let base = atlas_file();
        let k = atlas_key(&base, Platform::Desktop, &members, &read_a);
        assert_eq!(k, atlas_key(&base, Platform::Desktop, &members, &read_a));
        assert_eq!(k.len(), 32);
        // Platform, settings, content and the sprite table each move it.
        assert_ne!(k, atlas_key(&base, Platform::Quest, &members, &read_a));
        assert_ne!(k, atlas_key(&SpriteAtlasFile { padding: 5, ..base.clone() }, Platform::Desktop, &members, &read_a));
        assert_ne!(k, atlas_key(&base, Platform::Desktop, &members, &|m: &Member| Some(format!("{}x", m.rel).into_bytes())));
        let mut moved = members.clone();
        moved[0].sprites[0].rect.x = 1;
        moved[0].sheet.sprites[0].rect.x = 1;
        assert_ne!(k, atlas_key(&base, Platform::Desktop, &moved, &read_a));
        // A different platform's override does not move the other platform's key.
        let mut f = base.clone();
        f.quest = Some(AtlasPlatform { padding: Some(9), ..Default::default() });
        assert_ne!(atlas_key(&f, Platform::Quest, &members, &read_a), atlas_key(&base, Platform::Quest, &members, &read_a));
        let path = cache_path(Path::new("/p/Sprites/ui.spriteatlas"), Platform::Desktop, &k).unwrap();
        assert!(path.ends_with(format!(".zestengine_atlascache/ui.spriteatlas.Desktop.{k}.catlas")));
    }

    #[test]
    fn atlas_file_serde_round_trip_and_old_files() {
        let mut f = atlas_file();
        f.packables = vec!["Sprites".into(), "Hero/idle.png".into()];
        f.packing = PackingMode::Tight;
        f.variants.push(AtlasVariant::default());
        f.quest = Some(AtlasPlatform { variant: Some("Half".into()), ..Default::default() });
        let text = ron::ser::to_string_pretty(&f, ron::ser::PrettyConfig::default()).unwrap();
        assert_eq!(ron::de::from_str::<SpriteAtlasFile>(&text).unwrap(), f);
        // A file with only some fields loads with the defaults for the rest.
        let old: SpriteAtlasFile = ron::de::from_str("(packables: [\"A\"], padding: 7)").unwrap();
        assert_eq!(old.padding, 7);
        assert_eq!(old.max_size, 2048);
        assert!(old.include_in_build && old.texture.configured);
        assert!(old.lists_texture("A/x.png") || old.lists_texture("A"));
        assert!(!old.lists_texture("AB/x.png"));
        let empty: SpriteAtlasFile = ron::de::from_str("()").unwrap();
        assert_eq!(empty, SpriteAtlasFile::default());
    }

    #[test]
    fn packable_ref_strings_split_into_path_and_id() {
        let id = SubAssetId(0xdead_beef);
        let s = crate::sprite::sprite_ref_string("a/b.png", id);
        assert_eq!(split_packable(&s), ("a/b.png", Some(id)));
        assert_eq!(split_packable("a/b.png"), ("a/b.png", None));
        assert_eq!(split_packable("folder"), ("folder", None));
    }

    #[test]
    fn atlas_geometry_matches_the_unatlased_sprite_in_world_space_and_samples_the_page() {
        let (members, imgs) = fixture();
        let file = SpriteAtlasFile { allow_rotation: true, ..atlas_file() };
        let p = pack_members("t", &members, &file, Platform::Desktop, &loader(&imgs)).unwrap();
        for m in &members {
            for d in &m.sprites {
                let cs = &p.table.sprites[p.table.find(&m.rel, d.id).unwrap()];
                let page = p.table.pages[cs.page as usize];
                let g = atlas_geometry(d, &m.sheet, cs, page, 1.0);
                assert_eq!(g.page, cs.page);
                // Positions: inside the sprite's own world rect.
                let size = d.size_world(&m.sheet);
                let pv = d.pivot_world(&m.sheet);
                for pos in &g.geometry.positions {
                    assert!(pos[0] >= -pv[0] - 1e-4 && pos[0] <= size[0] - pv[0] + 1e-4);
                    assert!(pos[1] >= -pv[1] - 1e-4 && pos[1] <= size[1] - pv[1] + 1e-4);
                }
                // Each vertex's UV, sampled, equals the source pixel at that vertex
                // (check the vertices strictly inside the trim rect by nudging inward).
                let src = &imgs[&m.rel];
                for (pos, uv) in g.geometry.positions.iter().zip(&g.geometry.uvs) {
                    let lx = pos[0] + pv[0];
                    let ly = size[1] - (pos[1] + pv[1]);
                    let (px, py) = ((lx * d.effective_ppu(&m.sheet)).round(), (ly * d.effective_ppu(&m.sheet)).round());
                    let on_grid = (lx * d.effective_ppu(&m.sheet) - px).abs() < 1e-3 && (ly * d.effective_ppu(&m.sheet) - py).abs() < 1e-3;
                    let (ix, iy) = ((px as i64).clamp(cs.trim.x as i64, (cs.trim.x + cs.trim.w) as i64 - 1) as u32, (py as i64).clamp(cs.trim.y as i64, (cs.trim.y + cs.trim.h) as i64 - 1) as u32);
                    // UV of that texel's centre.
                    let cuv = cs.atlas_uv([ix as f32 + 0.5, iy as f32 + 0.5], page);
                    let pg = &p.pages[cs.page as usize];
                    let o = (((cuv[1] * pg.h as f32) as usize) * pg.w as usize + (cuv[0] * pg.w as f32) as usize) * 4;
                    let o2 = (((d.rect.y + iy) * src.w + d.rect.x + ix) * 4) as usize;
                    assert_eq!(pg.px[o..o + 4], src.px[o2..o2 + 4]);
                    if on_grid {
                        let want = cs.atlas_uv([(px).clamp(cs.trim.x as f32, (cs.trim.x + cs.trim.w) as f32), (py).clamp(cs.trim.y as f32, (cs.trim.y + cs.trim.h) as f32)], page);
                        assert!((want[0] - uv[0]).abs() < 1e-4 && (want[1] - uv[1]).abs() < 1e-4, "{want:?} vs {uv:?}");
                    }
                    assert!(uv[0] >= 0.0 && uv[0] <= 1.0 && uv[1] >= 0.0 && uv[1] <= 1.0, "uv {uv:?}");
                }
                assert_eq!(g.geometry.indices.len() % 3, 0);
            }
        }
    }

    #[test]
    fn cooked_atlas_bytes_round_trip() {
        let (members, imgs) = fixture();
        let packed = pack_members("t", &members, &atlas_file(), Platform::Desktop, &loader(&imgs)).unwrap();
        let table = packed.table.clone();
        let tex = CompressedTexture { format: zestengine_render::TexFormat::RgbaSrgb, width: 2, height: 2, mips: vec![vec![1; 16], vec![2; 4]] };
        let a = CookedAtlas { table: CookedAtlasTable { key: "k".into(), ..table.clone() }, pages: vec![tex] };
        let b = CookedAtlas::from_bytes(&a.to_bytes().unwrap()).unwrap();
        assert_eq!(b.table, a.table);
        assert_eq!(b.pages.len(), 1);
        assert_eq!((b.pages[0].width, b.pages[0].height, b.pages[0].mips.len()), (2, 2, 2));
        assert_eq!(b.pages[0].mips[1], vec![2; 4]);
        // Corrupt / truncated / foreign blobs are rejected, not trusted.
        let bytes = a.to_bytes().unwrap();
        assert!(CookedAtlas::from_bytes(&bytes[..bytes.len() - 3]).is_none());
        assert!(CookedAtlas::from_bytes(b"nonsense").is_none());
        assert!(CookedAtlas::from_bytes(&[]).is_none());
        assert_eq!(b.page_dims(0), table.pages[0]);
        assert_eq!(b.page_dims(99), [1, 1]);
    }

    #[test]
    fn project_scan_claims_textures_once_and_resolves_refs_through_the_atlas() {
        use crate::{AssetMeta, ImporterSettings};
        let root = std::env::temp_dir().join(format!("zestengine-atlas-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("Sprites")).unwrap();
        let tex = root.join("Sprites/hero.png");
        let im = img(64, 32, 20, |x, y| x < 60 && y < 30);
        image::RgbaImage::from_raw(64, 32, im.px.clone()).unwrap().save(&tex).unwrap();
        let mut meta = AssetMeta::new_for(&tex);
        let mut t = TextureImport { configured: true, kind: TextureKind::Sprite, ..TextureImport::default() };
        t.sprite.mode = SpriteMode::Multiple;
        t.sprite.sprites = vec![SpriteDef::named("left", PixelRect::new(0, 0, 32, 32)), SpriteDef::named("right", PixelRect::new(32, 0, 32, 32))];
        t.sprite.ensure_ids("hero");
        meta.importer = ImporterSettings::Texture(t.clone());
        crate::save_asset_meta(&tex, &meta).unwrap();
        // Two atlases both list the folder: the first (by path) owns the texture.
        let file = SpriteAtlasFile { packables: vec!["Sprites".into()], padding: 2, max_size: 128, ..SpriteAtlasFile::default() };
        file.save(&root.join("a.spriteatlas")).unwrap();
        file.save(&root.join("b.spriteatlas")).unwrap();
        let db = AtlasDb::scan(&root);
        assert_eq!(db.atlases.len(), 2);
        assert_eq!(db.atlas_name_of("Sprites/hero.png"), Some("a"));
        assert!(db.atlases[1].members.is_empty());
        let id = t.sprite.sprites[1].id;
        let r = SpriteRef { texture: Some(meta.id), path: "Sprites/hero.png".into(), sprite: id, name: "right".into() };
        let hit = db.resolve(Platform::Desktop, &r).expect("resolved through the atlas");
        assert_eq!(hit.atlas.table.sprites[hit.index].name, "right");
        assert_eq!(hit.def.id, id);
        let g = hit.geometry();
        assert!(g.geometry.uvs.iter().all(|uv| uv[0] <= 1.0 && uv[1] <= 1.0));
        // A second scan finds the cooked file in the cache; a changed pixel repacks.
        let db2 = AtlasDb::scan(&root);
        let (_, cached) = db2.cook(0, Platform::Desktop).unwrap();
        assert!(cached, "second cook reads .zestengine_atlascache");
        assert!(root.join(ATLAS_CACHE_DIR).exists() || root.join("Sprites").exists());
        let mut im2 = im.clone();
        im2.px[0] = 77;
        image::RgbaImage::from_raw(64, 32, im2.px).unwrap().save(&tex).unwrap();
        let db3 = AtlasDb::scan(&root);
        let (_, cached) = db3.cook(0, Platform::Desktop).unwrap();
        assert!(!cached, "changed content repacks");
        // A reference to a texture in no atlas does not resolve here.
        let none = SpriteRef { texture: None, path: "Other/x.png".into(), sprite: id, name: String::new() };
        assert!(db.resolve(Platform::Desktop, &none).is_none());
        let rep = db3.cook_all(Platform::Desktop, true);
        assert!(rep.warnings.is_empty(), "{:?}", rep.warnings);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn excluded_atlases_are_skipped_by_the_build_cook_only() {
        let root = std::env::temp_dir().join(format!("zestengine-atlas-ex-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        SpriteAtlasFile { include_in_build: false, ..SpriteAtlasFile::default() }.save(&root.join("x.spriteatlas")).unwrap();
        let db = AtlasDb::scan(&root);
        assert_eq!(db.atlases.len(), 1);
        let r = db.cook_all(Platform::Desktop, true);
        assert_eq!((r.packed, r.from_cache), (0, 0));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A project with one atlas of one multi-sprite texture.
    pub(crate) fn shipping_project(tag: &str) -> PathBuf {
        use crate::{AssetMeta, ImporterSettings};
        let root = std::env::temp_dir().join(format!("zestengine-atlas-ship-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("Sprites")).unwrap();
        let tex = root.join("Sprites/hero.png");
        let im = img(64, 32, 20, |x, y| x < 60 && y < 30);
        image::RgbaImage::from_raw(64, 32, im.px).unwrap().save(&tex).unwrap();
        let mut meta = AssetMeta::new_for(&tex);
        let mut t = TextureImport { configured: true, kind: TextureKind::Sprite, ..TextureImport::default() };
        t.sprite.mode = SpriteMode::Multiple;
        t.sprite.sprites = vec![SpriteDef::named("left", PixelRect::new(0, 0, 32, 32)), SpriteDef::named("right", PixelRect::new(32, 0, 32, 32))];
        t.sprite.ensure_ids("hero");
        meta.importer = ImporterSettings::Texture(t);
        crate::save_asset_meta(&tex, &meta).unwrap();
        SpriteAtlasFile { packables: vec!["Sprites".into()], padding: 2, max_size: 128, ..SpriteAtlasFile::default() }.save(&root.join("ui.spriteatlas")).unwrap();
        root
    }

    #[test]
    fn cooked_atlas_paths_round_trip_and_reject_traversal() {
        assert_eq!(cooked_atlas_rel("a/ui.spriteatlas"), "zestengine_atlascache/a/ui.spriteatlas.catlas");
        assert_eq!(parse_cooked_atlas_rel("zestengine_atlascache/a/ui.spriteatlas.catlas"), Some("a/ui.spriteatlas"));
        for bad in ["zestengine_atlascache/../ui.spriteatlas.catlas", "zestengine_atlascache/ui.png.catlas", ".zestengine_atlascache/ui.spriteatlas.catlas", "zestengine_atlascache/ui.spriteatlas.bin"] {
            assert_eq!(parse_cooked_atlas_rel(bad), None, "{bad}");
        }
        assert!(!COOKED_ATLAS_DIR.starts_with(['.', '_']), "the APK packager drops hidden folders");
    }

    #[test]
    fn a_build_ships_the_atlas_its_members_and_metas() {
        let root = shipping_project("files");
        let db = AtlasDb::scan(&root);
        assert_eq!(db.build_files(), vec!["Sprites/hero.png", "Sprites/hero.png.meta", "ui.spriteatlas"]);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_cooked_atlas_is_seeded_on_device_and_hits_the_cache() {
        let root = shipping_project("seed");
        let db = AtlasDb::scan(&root);
        let (blobs, warnings) = db.cooked_blobs(Platform::Quest, &|_| true);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(blobs.len(), 1);
        // The "APK": the same files minus every dot-folder, plus the shipped blob.
        let dev = std::env::temp_dir().join(format!("zestengine-atlas-dev-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dev);
        std::fs::create_dir_all(dev.join("Sprites")).unwrap();
        for f in db.build_files() {
            std::fs::copy(root.join(&f), dev.join(&f)).unwrap();
        }
        let shipped = dev.join(cooked_atlas_rel(&blobs[0].0));
        std::fs::create_dir_all(shipped.parent().unwrap()).unwrap();
        std::fs::write(&shipped, &blobs[0].1).unwrap();
        std::fs::write(dev.join(COOKED_ATLAS_DIR).join("stray.bin"), b"?").unwrap();
        assert!(!dev.join(ATLAS_CACHE_DIR).exists());
        let (seeded, warnings) = seed_cooked_atlases(&dev);
        assert_eq!(seeded, 1);
        assert_eq!(warnings.len(), 1, "the stray file is reported: {warnings:?}");
        assert!(!dev.join(COOKED_ATLAS_DIR).exists(), "shipped copies are removed");
        // The device cook reads the installed cache instead of packing.
        let dev_db = AtlasDb::scan(&dev);
        let (_, cached) = dev_db.cook(0, Platform::Quest).unwrap();
        assert!(cached, "the seeded atlas was found");
        let (_, cached) = dev_db.cook(0, Platform::Desktop).unwrap();
        assert!(!cached, "another platform still packs");
        // A garbage blob is rejected, not installed.
        assert!(install_cooked_atlas(&dev.join("ui.spriteatlas"), b"junk").is_err());
        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&dev);
    }

    #[test]
    fn atlases_with_members_outside_the_build_are_not_cooked() {
        let root = shipping_project("partial");
        let db = AtlasDb::scan(&root);
        let (blobs, warnings) = db.cooked_blobs(Platform::Desktop, &|rel| rel != "Sprites/hero.png");
        assert!(blobs.is_empty());
        assert_eq!(warnings.len(), 1);
        let _ = std::fs::remove_dir_all(&root);
    }
}
