//! Sprite assets (2D-2): the sprite table stored in a texture's `.meta`, stable
//! sprite ids, slicing maths, and the cooked geometry a renderer draws.
//!
//! A texture whose import kind is `Sprite` carries a [`SpriteSheet`] inside
//! `TextureImport` (`#[serde(default)]`, so older `.meta` files load unchanged
//! with an empty Single-mode sheet). In **Single** mode the whole image is one
//! sprite (its pivot / border / physics live in [`SpriteSheet::single`]); in
//! **Multiple** mode [`SpriteSheet::sprites`] is the table. Both are kept when
//! the mode or the texture kind changes, so nothing is lost by flipping back.
//!
//! Coordinates: sprite rects and custom outlines are in **source pixels, origin
//! top-left, y down**; outline points are rect-local. Pivots are normalised to
//! the rect with the origin at the **bottom-left, y up** (so Top = (0.5, 1)).
//! World-space sprite geometry is pivot-relative, y up, in units of
//! `pixels / pixels_per_unit`; UVs have v = 0 at the image's top row.
//!
//! # Identity (matching order)
//!
//! A reference to a sprite is a [`SpriteRef`]: texture GUID (the `.meta` id,
//! the path as a fallback), then the sprite's [`SubAssetId`], then its name.
//! The id merges Unity's sprite GUID and internal file id into one 64-bit value
//! (the `IdTable` machinery of `sub_asset.rs`): the first time a name is seen
//! it gets `hash(Sprite, name)` and the name is recorded in
//! [`SpriteSheet::id_table`]; ids are never reused, even after a sprite is
//! deleted (the retired name keeps its id, so adding the name back restores it).
//! When a sprite list is written back (re-slice, import) each incoming sprite
//! is matched against the existing ones **by id, then by name (live sprites,
//! then retired names in the table), else it gets a fresh id**. Renaming keeps
//! the id; re-slicing with [`SliceMethod::Smart`] keeps name and id of the
//! best-overlapping old rect.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use serde::{Deserialize, Serialize};
use zestengine_core::ObjectId;

use crate::sprite_mesh::{self, AlphaMap, P2, TIGHT_MIN_SIZE, TightParams};
use crate::sub_asset::{IdTable, SiblingNamer, SubAssetId, SubAssetKind};

/// Default pixels per world unit.
pub const DEFAULT_PPU: f32 = 100.0;
/// Smallest pixels-per-unit accepted (guards the division).
pub const MIN_PPU: f32 = 0.001;

// ---------------------------------------------------------------------------
// Small types
// ---------------------------------------------------------------------------

/// An integer pixel rectangle, origin top-left.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(default)]
pub struct PixelRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl PixelRect {
    pub fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }

    pub fn area(&self) -> u64 {
        self.w as u64 * self.h as u64
    }

    /// Clamp into an `iw` x `ih` image, at least 1 x 1 when the image allows.
    pub fn clamped(&self, iw: u32, ih: u32) -> PixelRect {
        let x = self.x.min(iw.saturating_sub(1));
        let y = self.y.min(ih.saturating_sub(1));
        PixelRect { x, y, w: self.w.max(1).min(iw.saturating_sub(x)), h: self.h.max(1).min(ih.saturating_sub(y)) }
    }

    pub fn fits(&self, iw: u32, ih: u32) -> bool {
        self.w > 0 && self.h > 0 && self.x as u64 + self.w as u64 <= iw as u64 && self.y as u64 + self.h as u64 <= ih as u64
    }

    /// Overlapping area in pixels.
    pub fn overlap(&self, o: &PixelRect) -> u64 {
        let x0 = self.x.max(o.x) as i64;
        let y0 = self.y.max(o.y) as i64;
        let x1 = (self.x as i64 + self.w as i64).min(o.x as i64 + o.w as i64);
        let y1 = (self.y as i64 + self.h as i64).min(o.y as i64 + o.h as i64);
        ((x1 - x0).max(0) * (y1 - y0).max(0)) as u64
    }
}

/// Mesh the renderer draws for a sprite.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpriteMeshType {
    /// Always the rect quad.
    FullRect,
    /// The alpha outline (sprites >= 32 x 32 only).
    #[default]
    Tight,
}

/// How the texture is divided into sprites.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpriteMode {
    /// The whole image is one sprite.
    #[default]
    Single,
    /// A table of rects.
    Multiple,
}

impl SpriteMode {
    pub const ALL: [Self; 2] = [Self::Single, Self::Multiple];
    pub fn label(self) -> &'static str {
        match self {
            Self::Single => "Single",
            Self::Multiple => "Multiple",
        }
    }
}

/// Where the pivot sits in the rect (Unity's alignment popup).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PivotPreset {
    #[default]
    Center,
    TopLeft,
    Top,
    TopRight,
    Left,
    Right,
    BottomLeft,
    Bottom,
    BottomRight,
    Custom,
}

impl PivotPreset {
    pub const ALL: [Self; 10] =
        [Self::Center, Self::TopLeft, Self::Top, Self::TopRight, Self::Left, Self::Right, Self::BottomLeft, Self::Bottom, Self::BottomRight, Self::Custom];

    pub fn label(self) -> &'static str {
        match self {
            Self::Center => "Center",
            Self::TopLeft => "Top Left",
            Self::Top => "Top",
            Self::TopRight => "Top Right",
            Self::Left => "Left",
            Self::Right => "Right",
            Self::BottomLeft => "Bottom Left",
            Self::Bottom => "Bottom",
            Self::BottomRight => "Bottom Right",
            Self::Custom => "Custom",
        }
    }

    /// Normalised pivot (origin bottom-left, y up); `None` for `Custom`.
    pub fn normalized(self) -> Option<[f32; 2]> {
        Some(match self {
            Self::Center => [0.5, 0.5],
            Self::TopLeft => [0.0, 1.0],
            Self::Top => [0.5, 1.0],
            Self::TopRight => [1.0, 1.0],
            Self::Left => [0.0, 0.5],
            Self::Right => [1.0, 0.5],
            Self::BottomLeft => [0.0, 0.0],
            Self::Bottom => [0.5, 0.0],
            Self::BottomRight => [1.0, 0.0],
            Self::Custom => return None,
        })
    }

    /// The preset a normalised pivot corresponds to, else `Custom`.
    pub fn detect(p: [f32; 2]) -> Self {
        Self::ALL.iter().copied().find(|k| k.normalized().is_some_and(|n| (n[0] - p[0]).abs() < 1e-4 && (n[1] - p[1]).abs() < 1e-4)).unwrap_or(Self::Custom)
    }
}

/// Atlas-friendly padding rules, read by the 2D-4 packer (nothing packs yet).
/// Mirrors the research (phase5b §3): padding around every packed sprite,
/// optional rotation by flips / 180 degrees only, edge texels extruded into the
/// padding of rectangle sprites and RGB alpha-dilated for tight ones, so
/// bilinear filtering and mips never pull in a neighbour's colour.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PackHints {
    /// Transparent-or-extruded border around each sprite in a page, in pixels.
    pub padding: u32,
    /// Allow the packer to flip / rotate by 180 degrees.
    pub allow_rotation: bool,
    /// Dilate RGB into fully transparent texels before mips.
    pub alpha_dilate: bool,
}

impl Default for PackHints {
    fn default() -> Self {
        Self { padding: 4, allow_rotation: false, alpha_dilate: true }
    }
}

/// What the packer writes into the padding ring of one sprite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeFill {
    /// Repeat the sprite's edge texels outward (rectangle sprites).
    ExtrudeEdge,
    /// Leave transparent; colour comes from alpha dilation (tight sprites).
    AlphaDilate,
}

impl PackHints {
    /// Size a sprite of `rect` takes in a page, padding included.
    pub fn footprint(&self, rect: PixelRect) -> (u32, u32) {
        (rect.w + 2 * self.padding, rect.h + 2 * self.padding)
    }

    pub fn edge_fill(&self, mesh: SpriteMeshType) -> EdgeFill {
        match mesh {
            SpriteMeshType::FullRect => EdgeFill::ExtrudeEdge,
            SpriteMeshType::Tight => EdgeFill::AlphaDilate,
        }
    }
}

// ---------------------------------------------------------------------------
// SpriteDef / SpriteSheet
// ---------------------------------------------------------------------------

/// One sprite of a texture.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpriteDef {
    /// Stable id (see the module docs). `NONE` = not assigned yet.
    pub id: SubAssetId,
    pub name: String,
    pub rect: PixelRect,
    pub pivot_preset: PivotPreset,
    /// Normalised pivot, used when `pivot_preset` is `Custom`.
    pub pivot: [f32; 2],
    /// 9-slice border in pixels: left, bottom, right, top.
    pub border: [u32; 4],
    /// Overrides of the sheet defaults (`None` = use the sheet's).
    pub ppu: Option<f32>,
    pub mesh_type: Option<SpriteMeshType>,
    pub extrude: Option<u32>,
    /// Hand-drawn render outline (rect-local pixels). Empty = trace the alpha.
    pub custom_outline: Vec<Vec<P2>>,
    /// Hand-drawn physics shape (rect-local pixels). Empty = auto.
    pub physics_shape: Vec<Vec<P2>>,
}

impl Default for SpriteDef {
    fn default() -> Self {
        Self {
            id: SubAssetId::NONE,
            name: String::new(),
            rect: PixelRect::default(),
            pivot_preset: PivotPreset::Center,
            pivot: [0.5, 0.5],
            border: [0; 4],
            ppu: None,
            mesh_type: None,
            extrude: None,
            custom_outline: Vec::new(),
            physics_shape: Vec::new(),
        }
    }
}

impl SpriteDef {
    pub fn named(name: &str, rect: PixelRect) -> Self {
        Self { name: name.to_owned(), rect, ..Self::default() }
    }

    /// Normalised pivot, clamped to a sane range.
    pub fn pivot_norm(&self) -> [f32; 2] {
        let p = self.pivot_preset.normalized().unwrap_or(self.pivot);
        [if p[0].is_finite() { p[0].clamp(-100_000.0, 100_000.0) } else { 0.5 }, if p[1].is_finite() { p[1].clamp(-100_000.0, 100_000.0) } else { 0.5 }]
    }

    /// Set a custom pivot from normalised coordinates (snaps to a preset if it matches).
    pub fn set_pivot(&mut self, p: [f32; 2]) {
        self.pivot = p;
        self.pivot_preset = PivotPreset::detect(p);
    }

    /// Pivot in pixels from the rect's bottom-left corner.
    pub fn pivot_pixels(&self) -> [f32; 2] {
        let p = self.pivot_norm();
        [p[0] * self.rect.w as f32, p[1] * self.rect.h as f32]
    }

    /// Set the pivot from pixels measured from the rect's bottom-left.
    pub fn set_pivot_pixels(&mut self, px: [f32; 2]) {
        self.set_pivot([px[0] / self.rect.w.max(1) as f32, px[1] / self.rect.h.max(1) as f32]);
    }

    pub fn effective_ppu(&self, sheet: &SpriteSheet) -> f32 {
        sanitize_ppu(self.ppu.unwrap_or(sheet.ppu))
    }

    pub fn effective_mesh_type(&self, sheet: &SpriteSheet) -> SpriteMeshType {
        self.mesh_type.unwrap_or(sheet.mesh_type)
    }

    pub fn effective_extrude(&self, sheet: &SpriteSheet) -> u32 {
        self.extrude.unwrap_or(sheet.extrude).min(sprite_mesh::MAX_EXTRUDE)
    }

    /// Size in world units (`pixels / ppu`).
    pub fn size_world(&self, sheet: &SpriteSheet) -> [f32; 2] {
        let ppu = self.effective_ppu(sheet);
        [self.rect.w as f32 / ppu, self.rect.h as f32 / ppu]
    }

    /// Pivot offset in world units from the sprite's bottom-left corner.
    pub fn pivot_world(&self, sheet: &SpriteSheet) -> [f32; 2] {
        let ppu = self.effective_ppu(sheet);
        let p = self.pivot_pixels();
        [p[0] / ppu, p[1] / ppu]
    }

    /// Border clamped so left + right <= width and bottom + top <= height.
    pub fn border_clamped(&self) -> [u32; 4] {
        let [l, b, r, t] = self.border;
        let (w, h) = (self.rect.w, self.rect.h);
        let l = l.min(w);
        let r = r.min(w - l);
        let b = b.min(h);
        let t = t.min(h - b);
        [l, b, r, t]
    }
}

fn sanitize_ppu(p: f32) -> f32 {
    if p.is_finite() { p.max(MIN_PPU) } else { DEFAULT_PPU }
}

/// The sprite settings of one texture.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpriteSheet {
    pub mode: SpriteMode,
    /// Pixels per world unit (per-sprite `ppu` overrides it).
    pub ppu: f32,
    pub mesh_type: SpriteMeshType,
    /// Tight-mesh extrusion in pixels, 0..=32.
    pub extrude: u32,
    /// Pixels with alpha above this are solid for the mesh.
    pub alpha_tolerance: u8,
    /// Tight-mesh detail, 0 (coarse) .. 1 (fine).
    pub outline_detail: f32,
    /// Generate a physics outline for sprites without a custom one.
    pub generate_physics_shape: bool,
    pub physics_alpha_tolerance: u8,
    /// 0 (coarse) .. 1 (fine); the editor default is 0.85.
    pub physics_detail: f32,
    pub pack: PackHints,
    /// The one sprite of a Single-mode texture (rect is always the full image).
    pub single: SpriteDef,
    /// The sprite table of a Multiple-mode texture.
    pub sprites: Vec<SpriteDef>,
    /// Name -> id of every sprite ever seen, retired names included.
    pub id_table: IdTable,
}

impl Default for SpriteSheet {
    fn default() -> Self {
        Self {
            mode: SpriteMode::Single,
            ppu: DEFAULT_PPU,
            mesh_type: SpriteMeshType::Tight,
            extrude: 0,
            alpha_tolerance: 1,
            outline_detail: 0.5,
            generate_physics_shape: true,
            physics_alpha_tolerance: 128,
            physics_detail: 0.85,
            pack: PackHints::default(),
            single: SpriteDef::default(),
            sprites: Vec::new(),
            id_table: IdTable::default(),
        }
    }
}

/// How a new set of rects meets the existing sprite table.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SliceMethod {
    /// Drop the table first. Ids of re-created names still come back through
    /// the id table.
    DeleteExisting,
    /// Each new rect inherits the best-overlapping old sprite (name, id, pivot,
    /// border...) and takes the new rect; old sprites nobody claims are removed.
    #[default]
    Smart,
    /// Keep every old sprite; add only rects that overlap none of them.
    Safe,
}

/// A rect of a new slice must overlap the old rect by at least this share of
/// its own area to inherit its identity (research: `kBestFitTolerance = 0.5`).
pub const BEST_FIT: f64 = 0.5;

impl SpriteSheet {
    pub fn tight_params(&self, def: &SpriteDef) -> TightParams {
        TightParams { alpha_threshold: self.alpha_tolerance, detail: self.outline_detail, extrude: def.effective_extrude(self), holes: true }
    }

    /// Make names unique and assign / repair ids: sprites with no id (or a
    /// duplicate) get the recorded id of their name or a fresh hash, and the id
    /// table learns every live (id, name) pair. Returns true when anything
    /// changed. Call before saving so ids persist.
    pub fn ensure_ids(&mut self, stem: &str) -> bool {
        let before = (self.single.clone(), self.sprites.clone(), self.id_table.clone());
        let mut namer = SiblingNamer::default();
        if self.single.name.is_empty() {
            self.single.name = stem.to_owned();
        }
        let mut taken: HashSet<SubAssetId> = HashSet::new();
        let single_name = namer.unique("", &self.single.name);
        self.single.name = single_name;
        let mut all: Vec<&mut SpriteDef> = std::iter::once(&mut self.single).chain(self.sprites.iter_mut()).collect();
        for (i, d) in all.iter_mut().enumerate() {
            if i > 0 {
                d.name = namer.unique("", &d.name);
            }
            if !d.id.is_none() && !taken.insert(d.id) {
                d.id = SubAssetId::NONE; // duplicate (copy / paste): re-issue
            }
        }
        // Sprites that already have an id keep it; make the table agree.
        for d in all.iter().filter(|d| !d.id.is_none()) {
            table_set(&mut self.id_table, d.id, &d.name);
        }
        for d in all.iter_mut().filter(|d| d.id.is_none()) {
            let (id, _) = self.id_table.register(SubAssetKind::Sprite, &d.name, &taken);
            // The recorded id of this name might belong to a live sprite already.
            let id = if taken.contains(&id) {
                let salted = crate::sub_asset::hashed_id(SubAssetKind::Sprite, &format!("{}#{}", d.name, taken.len()));
                table_set(&mut self.id_table, salted, &d.name);
                salted
            } else {
                id
            };
            taken.insert(id);
            d.id = id;
        }
        before != (self.single.clone(), self.sprites.clone(), self.id_table.clone())
    }

    /// The sprites of a texture of `w` x `h`, ready to use: ids assigned (not
    /// stored), rects clamped, Single mode expanded to the full image.
    pub fn resolve(&self, w: u32, h: u32, stem: &str) -> Vec<SpriteDef> {
        let mut s = self.clone();
        s.ensure_ids(stem);
        match s.mode {
            SpriteMode::Single => {
                let mut d = s.single;
                d.rect = PixelRect::new(0, 0, w.max(1), h.max(1));
                vec![d]
            }
            SpriteMode::Multiple => s
                .sprites
                .into_iter()
                .filter(|d| d.rect.w > 0 && d.rect.h > 0 && d.rect.x < w && d.rect.y < h)
                .map(|mut d| {
                    d.rect = d.rect.clamped(w, h);
                    d
                })
                .collect(),
        }
    }

    /// Rename sprite `i` of the Multiple table, keeping its id. Refuses names
    /// already used by another live sprite.
    pub fn rename_sprite(&mut self, i: usize, new: &str) -> bool {
        let new = new.trim();
        if new.is_empty() || i >= self.sprites.len() {
            return false;
        }
        if self.sprites.iter().enumerate().any(|(k, s)| k != i && s.name == new) || self.single.name == new {
            return false;
        }
        let id = self.sprites[i].id;
        self.sprites[i].name = new.to_owned();
        if !id.is_none() {
            // A retired entry that used this name must not shadow the live one.
            self.id_table.entries.retain(|e| !(e.kind == SubAssetKind::Sprite && e.name == new && e.id != id));
            table_set(&mut self.id_table, id, new);
        }
        true
    }

    /// Append a sprite with a free `stem_N` name and a fresh id.
    pub fn add_sprite(&mut self, stem: &str, rect: PixelRect) -> usize {
        let name = self.free_name(stem);
        self.sprites.push(SpriteDef::named(&name, rect));
        self.ensure_ids(stem);
        self.sprites.len() - 1
    }

    /// Remove sprite `i`. Its name stays in the id table (ids are never reused).
    pub fn remove_sprite(&mut self, i: usize) {
        if i < self.sprites.len() {
            self.sprites.remove(i);
        }
    }

    fn free_name(&self, stem: &str) -> String {
        let used: HashSet<&str> = self.sprites.iter().map(|s| s.name.as_str()).chain(std::iter::once(self.single.name.as_str())).collect();
        (0u32..).map(|n| format!("{stem}_{n}")).find(|n| !used.contains(n.as_str())).unwrap_or_default()
    }

    /// Merge freshly cut `rects` into the Multiple table (see [`SliceMethod`]).
    /// Returns how many existing sprites kept their identity.
    pub fn apply_slice(&mut self, stem: &str, rects: &[PixelRect], method: SliceMethod) -> usize {
        self.apply_slice_tracked(stem, rects, method).0
    }

    /// [`Self::apply_slice`] that also says, per sprite of the new table,
    /// whether it inherited an existing sprite (`true`) or is new (`false`),
    /// so the caller can give new sprites their default pivot / outline.
    pub fn apply_slice_tracked(&mut self, stem: &str, rects: &[PixelRect], method: SliceMethod) -> (usize, Vec<bool>) {
        self.ensure_ids(stem);
        let old = std::mem::take(&mut self.sprites);
        let old_len = old.len();
        let mut kept = 0;
        let mut inherited: Vec<bool> = Vec::new();
        match method {
            SliceMethod::DeleteExisting => {
                for r in rects {
                    let name = free_name_among(stem, &self.sprites);
                    self.sprites.push(SpriteDef::named(&name, *r));
                    inherited.push(false);
                }
            }
            SliceMethod::Smart => {
                let mut claimed = vec![false; old.len()];
                let mut out: Vec<SpriteDef> = Vec::new();
                for r in rects {
                    let best = old
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| !claimed[*i])
                        .map(|(i, o)| (i, o.rect.overlap(r)))
                        .filter(|(_, ov)| r.area() > 0 && (*ov as f64 / r.area() as f64 - 1.0).abs() <= BEST_FIT)
                        .max_by_key(|(_, ov)| *ov);
                    if let Some((i, _)) = best {
                        claimed[i] = true;
                        let mut d = old[i].clone();
                        d.rect = *r;
                        out.push(d);
                        inherited.push(true);
                        kept += 1;
                    } else {
                        let name = free_name_among_with(stem, &out, &old);
                        out.push(SpriteDef::named(&name, *r));
                        inherited.push(false);
                    }
                }
                self.sprites = out;
            }
            SliceMethod::Safe => {
                self.sprites = old;
                kept = self.sprites.len();
                inherited = vec![true; old_len];
                for r in rects {
                    if self.sprites.iter().all(|s| s.rect.overlap(r) == 0) {
                        let name = free_name_among(stem, &self.sprites);
                        self.sprites.push(SpriteDef::named(&name, *r));
                        inherited.push(false);
                    }
                }
            }
        }
        self.ensure_ids(stem);
        (kept, inherited)
    }

    /// Problems worth showing next to the table.
    pub fn warnings(&self, w: u32, h: u32) -> Vec<String> {
        let mut out = Vec::new();
        if self.mode != SpriteMode::Multiple {
            return out;
        }
        for (i, s) in self.sprites.iter().enumerate() {
            if !s.rect.fits(w, h) {
                out.push(format!("'{}' is outside the {w}x{h} image", s.name));
            }
            if s.effective_mesh_type(self) == SpriteMeshType::Tight && (s.rect.w < TIGHT_MIN_SIZE || s.rect.h < TIGHT_MIN_SIZE) && s.custom_outline.is_empty() {
                out.push(format!("'{}' is under {TIGHT_MIN_SIZE}x{TIGHT_MIN_SIZE}: it gets a quad, not a tight mesh", s.name));
            }
            for o in &self.sprites[i + 1..] {
                if s.rect.overlap(&o.rect) > 0 {
                    out.push(format!("'{}' and '{}' overlap", s.name, o.name));
                }
            }
        }
        out
    }
}

fn table_set(t: &mut IdTable, id: SubAssetId, name: &str) {
    if let Some(e) = t.entries.iter_mut().find(|e| e.kind == SubAssetKind::Sprite && e.id == id) {
        e.name = name.to_owned();
    } else {
        t.entries.retain(|e| !(e.kind == SubAssetKind::Sprite && e.name == name)); // stale name
        t.entries.push(crate::sub_asset::IdEntry { kind: SubAssetKind::Sprite, name: name.to_owned(), id });
    }
}

fn free_name_among(stem: &str, list: &[SpriteDef]) -> String {
    free_name_among_with(stem, list, &[])
}

fn free_name_among_with(stem: &str, a: &[SpriteDef], b: &[SpriteDef]) -> String {
    let used: HashSet<&str> = a.iter().chain(b).map(|s| s.name.as_str()).collect();
    (0u32..).map(|n| format!("{stem}_{n}")).find(|n| !used.contains(n.as_str())).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Slicing maths
// ---------------------------------------------------------------------------

/// Cell width/height from a count along one axis:
/// `floor((extent - offset - padding * (n - 1)) / n)`.
pub fn cell_size_from_count(extent: u32, count: u32, offset: u32, padding: u32) -> u32 {
    if count == 0 {
        return 0;
    }
    let used = offset as i64 + padding as i64 * (count as i64 - 1);
    ((extent as i64 - used).max(0) / count as i64) as u32
}

/// Full cells of `cell` pixels over a `w` x `h` image, rows top to bottom, each
/// left to right; `offset` is the top-left margin, `padding` the gap between
/// cells. Cells that don't fit completely are skipped.
pub fn grid_rects(w: u32, h: u32, cell: [u32; 2], offset: [u32; 2], padding: [u32; 2]) -> Vec<PixelRect> {
    let mut out = Vec::new();
    if cell[0] == 0 || cell[1] == 0 {
        return out;
    }
    let mut y = offset[1];
    while y + cell[1] <= h {
        let mut x = offset[0];
        while x + cell[0] <= w {
            out.push(PixelRect::new(x, y, cell[0], cell[1]));
            x += cell[0] + padding[0];
        }
        y += cell[1] + padding[1];
    }
    out
}

/// Grid from a column / row count.
pub fn grid_rects_by_count(w: u32, h: u32, cols: u32, rows: u32, offset: [u32; 2], padding: [u32; 2]) -> Vec<PixelRect> {
    let cell = [cell_size_from_count(w, cols, offset[0], padding[0]), cell_size_from_count(h, rows, offset[1], padding[1])];
    grid_rects(w, h, cell, offset, padding)
}

/// Drop rects with no pixel above `threshold`.
pub fn drop_empty(rects: Vec<PixelRect>, a: &AlphaMap, threshold: u8) -> Vec<PixelRect> {
    rects
        .into_iter()
        .filter(|r| {
            let c = r.clamped(a.w, a.h);
            (0..c.h).any(|y| (0..c.w).any(|x| a.at(c.x + x, c.y + y) > threshold))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// References
// ---------------------------------------------------------------------------

/// A serialisable pointer to one sprite: texture GUID (path as a fallback),
/// then sprite id, then name.
///
/// Text form (what a Files-panel drag of a sprite carries and what a drop
/// target or the 2D render pass reads): `<project-relative path>#<sprite id as
/// 16 lowercase hex digits>`, e.g. `Sprites/hero.png#00a1b2c3d4e5f607`. Build it
/// with [`sprite_ref_string`], read it with [`SpriteRef::from_ref_string`]. The
/// id is the sprite's stable `SubAssetId`, so it survives renames and
/// reordering of the sheet; the GUID and name of the full reference are filled
/// in when the target resolves the texture.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpriteRef {
    pub texture: Option<ObjectId>,
    /// Project-relative path, forward slashes.
    pub path: String,
    pub sprite: SubAssetId,
    /// Last known name, the fallback when the id no longer exists.
    pub name: String,
}

/// `<path>#<id>` text form of a sprite reference (see [`SpriteRef`]); the path
/// is made project-relative with forward slashes.
pub fn sprite_ref_string(path: &str, id: SubAssetId) -> String {
    format!("{}#{:016x}", path.replace('\\', "/"), id.0)
}

impl SpriteRef {
    /// Parse the text form of [`sprite_ref_string`]. `None` unless it ends in
    /// `#` and 16 hex digits naming a non-zero id.
    pub fn from_ref_string(text: &str) -> Option<SpriteRef> {
        let (path, id) = text.rsplit_once('#')?;
        if path.is_empty() || id.len() != 16 || !id.bytes().all(|b| b.is_ascii_hexdigit()) {
            return None;
        }
        let id = SubAssetId(u64::from_str_radix(id, 16).ok()?);
        (!id.is_none()).then(|| SpriteRef { texture: None, path: path.replace('\\', "/"), sprite: id, name: String::new() })
    }

    /// The text form of this reference (empty when it names no sprite).
    pub fn to_ref_string(&self) -> String {
        if self.is_none() || self.sprite.is_none() { String::new() } else { sprite_ref_string(&self.path, self.sprite) }
    }

    pub fn is_none(&self) -> bool {
        self.texture.is_none() && self.path.is_empty()
    }

    /// Index of the referenced sprite in `sprites`: by id, then by name.
    pub fn find_in(&self, sprites: &[SpriteDef]) -> Option<usize> {
        if !self.sprite.is_none()
            && let Some(i) = sprites.iter().position(|s| s.id == self.sprite)
        {
            return Some(i);
        }
        if self.name.is_empty() {
            return None;
        }
        sprites.iter().position(|s| s.name == self.name)
    }

    /// A reference to `sprites[i]` of the texture at `path` (project-relative).
    pub fn to(texture: Option<ObjectId>, path: &str, def: &SpriteDef) -> Self {
        Self { texture, path: path.replace('\\', "/"), sprite: def.id, name: def.name.clone() }
    }
}

// ---------------------------------------------------------------------------
// Cooked geometry
// ---------------------------------------------------------------------------

/// What a renderer needs for one sprite. Positions are in world units,
/// pivot-relative, y up; indices are counter-clockwise.
#[derive(Clone, Debug, PartialEq)]
pub struct SpriteGeometry {
    pub id: SubAssetId,
    pub name: String,
    pub rect: PixelRect,
    pub ppu: f32,
    /// Rect size in world units.
    pub size_world: [f32; 2],
    /// Pivot from the rect's bottom-left, in world units.
    pub pivot_world: [f32; 2],
    /// 9-slice border in world units (left, bottom, right, top).
    pub border_world: [f32; 4],
    /// Rect in UV space: min u, min v, max u, max v (v = 0 at the image's top).
    pub uv_rect: [f32; 4],
    pub positions: Vec<[f32; 2]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u16>,
    /// True when `positions` follow the alpha outline instead of the rect quad.
    pub tight: bool,
    /// Physics polygons (world units, pivot-relative; outer loops and holes).
    pub physics: Vec<Vec<[f32; 2]>>,
}

impl SpriteGeometry {
    /// Bounds of the mesh in world units: (min, max).
    pub fn bounds(&self) -> ([f32; 2], [f32; 2]) {
        let mut lo = [f32::MAX; 2];
        let mut hi = [f32::MIN; 2];
        for p in &self.positions {
            for k in 0..2 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        (lo, hi)
    }
}

/// Rect-local pixel point -> pivot-relative world position.
pub fn pixel_to_world(p: P2, rect: PixelRect, pivot: [f32; 2], ppu: f32) -> [f32; 2] {
    [(p[0] - pivot[0] * rect.w as f32) / ppu, ((rect.h as f32 - p[1]) - pivot[1] * rect.h as f32) / ppu]
}

/// Rect-local pixel point -> texture UV.
pub fn pixel_to_uv(p: P2, rect: PixelRect, tex_w: u32, tex_h: u32) -> [f32; 2] {
    [(rect.x as f32 + p[0]) / tex_w.max(1) as f32, (rect.y as f32 + p[1]) / tex_h.max(1) as f32]
}

/// The pixel mesh a sprite draws: custom outline, else the alpha trace for a
/// Tight sprite of at least 32 x 32 (or with a custom outline), else the quad.
/// Returns the mesh and whether it is tight.
pub fn sprite_pixel_mesh(def: &SpriteDef, sheet: &SpriteSheet, alpha: Option<&AlphaMap>) -> (sprite_mesh::PixelMesh, bool) {
    let rect = def.rect;
    let quad = || (sprite_mesh::quad_mesh(rect.w, rect.h), false);
    if def.effective_mesh_type(sheet) == SpriteMeshType::FullRect {
        return quad();
    }
    if !def.custom_outline.is_empty() {
        return match sprite_mesh::mesh_from_loops(&def.custom_outline, true) {
            Some(m) => (m, true),
            None => quad(),
        };
    }
    if rect.w < TIGHT_MIN_SIZE || rect.h < TIGHT_MIN_SIZE {
        return quad();
    }
    match alpha.and_then(|a| sprite_mesh::tight_mesh(a, rect, &sheet.tight_params(def))) {
        Some(m) => (m, true),
        None => quad(),
    }
}

/// Physics polygons in rect-local pixels: the custom shape, else (when
/// generation is on and pixels are available) the traced outline.
pub fn sprite_physics_pixels(def: &SpriteDef, sheet: &SpriteSheet, alpha: Option<&AlphaMap>) -> Vec<Vec<P2>> {
    if !def.physics_shape.is_empty() {
        return def.physics_shape.clone();
    }
    if !sheet.generate_physics_shape {
        return Vec::new();
    }
    match alpha {
        Some(a) => sprite_mesh::physics_outline(a, def.rect, sheet.physics_alpha_tolerance, sprite_mesh::physics_tolerance(sheet.physics_detail)),
        None => Vec::new(),
    }
}

/// Build the cooked geometry of one sprite. `alpha` is the texture's alpha
/// plane (`None` = quad, no auto physics).
pub fn build_geometry(def: &SpriteDef, sheet: &SpriteSheet, tex_w: u32, tex_h: u32, alpha: Option<&AlphaMap>) -> SpriteGeometry {
    let ppu = def.effective_ppu(sheet);
    let pivot = def.pivot_norm();
    let rect = def.rect;
    let (mesh, tight) = sprite_pixel_mesh(def, sheet, alpha);
    let positions: Vec<[f32; 2]> = mesh.verts.iter().map(|v| pixel_to_world(*v, rect, pivot, ppu)).collect();
    let uvs: Vec<[f32; 2]> = mesh.verts.iter().map(|v| pixel_to_uv(*v, rect, tex_w, tex_h)).collect();
    // Pixel-space triangles wind positive in y-down maths; the y flip to world
    // space reverses that, so swap two indices to end up counter-clockwise.
    let indices: Vec<u16> = mesh.indices.as_chunks::<3>().0.iter().flat_map(|t| [t[0] as u16, t[2] as u16, t[1] as u16]).collect();
    let physics = sprite_physics_pixels(def, sheet, alpha)
        .into_iter()
        .map(|poly| poly.into_iter().map(|p| pixel_to_world(p, rect, pivot, ppu)).collect())
        .collect();
    let [l, b, r, t] = def.border_clamped();
    let tl = pixel_to_uv([0.0, 0.0], rect, tex_w, tex_h);
    let br = pixel_to_uv([rect.w as f32, rect.h as f32], rect, tex_w, tex_h);
    SpriteGeometry {
        id: def.id,
        name: def.name.clone(),
        rect,
        ppu,
        size_world: def.size_world(sheet),
        pivot_world: def.pivot_world(sheet),
        border_world: [l as f32 / ppu, b as f32 / ppu, r as f32 / ppu, t as f32 / ppu],
        uv_rect: [tl[0], tl[1], br[0], br[1]],
        positions,
        uvs,
        indices,
        tight,
        physics,
    }
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// A texture's resolved sprite table.
#[derive(Clone, Debug)]
pub struct SpriteTable {
    pub texture_path: PathBuf,
    pub guid: ObjectId,
    pub width: u32,
    pub height: u32,
    pub sheet: SpriteSheet,
    pub sprites: Vec<SpriteDef>,
}

impl SpriteTable {
    pub fn find(&self, id: SubAssetId) -> Option<&SpriteDef> {
        self.sprites.iter().find(|s| s.id == id)
    }

    pub fn find_ref(&self, r: &SpriteRef) -> Option<&SpriteDef> {
        r.find_in(&self.sprites).map(|i| &self.sprites[i])
    }

    /// Geometry of sprite `i`, decoding the image only when a tight mesh or
    /// auto physics needs the alpha plane.
    pub fn geometry(&self, i: usize) -> Option<SpriteGeometry> {
        let def = self.sprites.get(i)?;
        let alpha = if self.needs_alpha(def) { load_alpha(&self.texture_path).ok() } else { None };
        Some(build_geometry(def, &self.sheet, self.width, self.height, alpha.as_ref()))
    }

    /// Geometry of every sprite, decoding the image once.
    pub fn all_geometry(&self) -> Vec<SpriteGeometry> {
        let alpha = if self.sprites.iter().any(|d| self.needs_alpha(d)) { load_alpha(&self.texture_path).ok() } else { None };
        self.sprites.iter().map(|d| build_geometry(d, &self.sheet, self.width, self.height, alpha.as_ref())).collect()
    }

    fn needs_alpha(&self, d: &SpriteDef) -> bool {
        (d.effective_mesh_type(&self.sheet) == SpriteMeshType::Tight && d.custom_outline.is_empty() && d.rect.w >= TIGHT_MIN_SIZE && d.rect.h >= TIGHT_MIN_SIZE)
            || (self.sheet.generate_physics_shape && d.physics_shape.is_empty())
    }
}

/// Decode a texture file's alpha plane.
pub fn load_alpha(path: &Path) -> Result<AlphaMap> {
    let img = image::open(path).with_context(|| format!("decoding {}", path.display()))?.to_rgba8();
    Ok(AlphaMap::from_rgba8(img.width(), img.height(), img.as_raw()))
}

fn stem_of(path: &Path) -> String {
    path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "sprite".into())
}

/// Load the sprite table of the texture at `path`: `None` when its `.meta` is
/// missing or its kind is not Sprite. Reads only the image header for the size.
pub fn load_sprite_table(path: &Path) -> Result<Option<SpriteTable>> {
    let Some(meta) = crate::load_asset_meta(path)? else { return Ok(None) };
    let Some(t) = meta.importer.as_texture() else { return Ok(None) };
    if t.kind != crate::TextureKind::Sprite {
        return Ok(None);
    }
    let (width, height) = image::image_dimensions(path).with_context(|| format!("reading size of {}", path.display()))?;
    let sprites = t.sprite.resolve(width, height, &stem_of(path));
    Ok(Some(SpriteTable { texture_path: path.to_path_buf(), guid: meta.id, width, height, sheet: t.sprite.clone(), sprites }))
}

/// Number of sprites a texture carries, for the Files panel badge. `None` when
/// it is not a sprite texture. Reads the `.meta` only (no image decoding), so
/// a Single-mode texture is 1.
pub fn sprite_count(path: &Path) -> Option<u32> {
    let meta = crate::load_asset_meta(path).ok()??;
    let t = meta.importer.as_texture()?;
    if t.kind != crate::TextureKind::Sprite {
        return None;
    }
    Some(match t.sprite.mode {
        SpriteMode::Single => 1,
        SpriteMode::Multiple => t.sprite.sprites.len() as u32,
    })
}

/// The sprites of a Multiple-mode sheet as (id, name, text reference for
/// `rel_path`), for the Files panel's expanded sheet. Empty for other textures
/// and for sprites without an id yet. Reads the `.meta` only.
pub fn sprite_children(path: &Path, rel_path: &str) -> Vec<(u64, String, String)> {
    let Some(meta) = crate::load_asset_meta(path).ok().flatten() else { return Vec::new() };
    let Some(t) = meta.importer.as_texture() else { return Vec::new() };
    if t.kind != crate::TextureKind::Sprite || t.sprite.mode != SpriteMode::Multiple {
        return Vec::new();
    }
    t.sprite.sprites.iter().filter(|d| !d.id.is_none()).map(|d| (d.id.0, d.name.clone(), sprite_ref_string(rel_path, d.id))).collect()
}

/// Find the texture a reference names under `root` (path first, then a GUID
/// search through the project's textures) and its table.
pub fn resolve_sprite_ref(root: &Path, r: &SpriteRef) -> Option<(SpriteTable, usize)> {
    let try_path = |p: &Path| -> Option<(SpriteTable, usize)> {
        let table = load_sprite_table(p).ok()??;
        if let Some(g) = r.texture
            && g != table.guid
        {
            return None; // another texture lives at that path now
        }
        let i = r.find_in(&table.sprites)?;
        Some((table, i))
    };
    if !r.path.is_empty()
        && let Some(hit) = try_path(&root.join(&r.path))
    {
        return Some(hit);
    }
    let guid = r.texture?;
    let found = texture_guid_index(root).remove(&guid)?;
    try_path(&found)
}

/// Meta GUID -> absolute path of every Sprite-kind texture under `root`.
fn texture_guid_index(root: &Path) -> HashMap<ObjectId, PathBuf> {
    fn walk(dir: &Path, out: &mut HashMap<ObjectId, PathBuf>) {
        let Ok(read) = std::fs::read_dir(dir) else { return };
        for entry in read.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let path = entry.path();
            let Ok(ft) = entry.file_type() else { continue };
            if ft.is_dir() {
                if !name.starts_with('.') && name != "target" {
                    walk(&path, out);
                }
            } else if let Some(tex) = name.strip_suffix(".meta").map(|m| path.with_file_name(m))
                && let Ok(Some(meta)) = crate::load_asset_meta(&tex)
                && meta.importer.as_texture().is_some_and(|t| t.kind == crate::TextureKind::Sprite)
            {
                out.insert(meta.id, tex);
            }
        }
    }
    let mut out = HashMap::new();
    walk(root, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AssetMeta, ImporterSettings, TextureImport, TextureKind};

    fn sheet_multi(names: &[&str]) -> SpriteSheet {
        let mut s = SpriteSheet { mode: SpriteMode::Multiple, ..SpriteSheet::default() };
        for (i, n) in names.iter().enumerate() {
            s.sprites.push(SpriteDef::named(n, PixelRect::new(i as u32 * 32, 0, 32, 32)));
        }
        s.ensure_ids("tex");
        s
    }

    #[test]
    fn table_serde_round_trip_and_defaults() {
        let mut s = sheet_multi(&["a", "b"]);
        s.sprites[0].border = [1, 2, 3, 4];
        s.sprites[0].set_pivot([0.25, 0.75]);
        s.sprites[1].physics_shape = vec![vec![[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]]];
        s.sprites[1].ppu = Some(16.0);
        let text = ron::ser::to_string_pretty(&s, Default::default()).unwrap();
        assert_eq!(ron::from_str::<SpriteSheet>(&text).unwrap(), s);
        // Every field is optional.
        let d: SpriteSheet = ron::from_str("()").unwrap();
        assert_eq!(d, SpriteSheet::default());
        assert_eq!(d.ppu, 100.0);
        assert_eq!(d.mesh_type, SpriteMeshType::Tight);
        let d: SpriteDef = ron::from_str("(name: \"x\", rect: (w: 8, h: 8))").unwrap();
        assert_eq!((d.pivot_norm(), d.rect), ([0.5, 0.5], PixelRect::new(0, 0, 8, 8)));
    }

    #[test]
    fn old_texture_meta_loads_with_an_empty_single_sheet() {
        // A meta written before sprites existed has no `sprite` field.
        let mut meta = AssetMeta::new_for(Path::new("a.png"));
        meta.importer = ImporterSettings::Texture(TextureImport { configured: true, kind: TextureKind::Sprite, ..TextureImport::default() });
        let text = ron::ser::to_string_pretty(&meta, Default::default()).unwrap();
        let stripped: String = {
            let a = text.find("sprite:").expect("sprite field is serialised");
            // Remove the whole `sprite: (...)` entry by brace matching.
            let open = text[a..].find('(').unwrap() + a;
            let mut depth = 0;
            let mut end = open;
            for (i, c) in text[open..].char_indices() {
                match c {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            end = open + i + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let tail = text[end..].trim_start_matches(',');
            format!("{}{}", &text[..a], tail)
        };
        assert!(!stripped.contains("sprite:"));
        let back: AssetMeta = ron::from_str(&stripped).unwrap();
        let t = back.importer.as_texture().unwrap();
        assert_eq!(t.sprite, SpriteSheet::default());
        assert_eq!(t.sprite.mode, SpriteMode::Single);
        // And the Single sprite resolves to the whole image.
        let r = t.sprite.resolve(64, 48, "a");
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].rect, PixelRect::new(0, 0, 64, 48));
        assert_eq!(r[0].name, "a");
        assert!(!r[0].id.is_none());
    }

    #[test]
    fn ids_are_stable_and_distinct() {
        let s = sheet_multi(&["a", "b", "c"]);
        let ids: HashSet<_> = s.sprites.iter().map(|d| d.id).collect();
        assert_eq!(ids.len(), 3);
        assert!(s.sprites.iter().all(|d| !d.id.is_none()));
        // Same names, fresh sheet: the same hash ids (deterministic).
        let t = sheet_multi(&["a", "b", "c"]);
        assert_eq!(s.sprites.iter().map(|d| d.id).collect::<Vec<_>>(), t.sprites.iter().map(|d| d.id).collect::<Vec<_>>());
        // Single and Multiple never collide.
        assert!(!ids.contains(&s.single.id));
        // Duplicate names and copied ids are repaired.
        let mut s = sheet_multi(&["a"]);
        let dup = s.sprites[0].clone();
        s.sprites.push(dup);
        assert!(s.ensure_ids("tex"));
        assert_ne!(s.sprites[0].name, s.sprites[1].name);
        assert_ne!(s.sprites[0].id, s.sprites[1].id);
        assert!(!s.ensure_ids("tex"), "second pass changes nothing");
    }

    #[test]
    fn rename_keeps_the_id_and_a_retired_name_restores_it() {
        let mut s = sheet_multi(&["hero", "foe"]);
        let hero = s.sprites[0].id;
        assert!(s.rename_sprite(0, "knight"));
        assert_eq!(s.sprites[0].id, hero);
        assert!(!s.rename_sprite(0, "foe"), "name in use");
        assert!(!s.rename_sprite(0, "  "));
        // A reference written before the rename still resolves (id first).
        let r = SpriteRef { sprite: hero, name: "hero".into(), ..SpriteRef::default() };
        assert_eq!(r.find_in(&s.sprites), Some(0));
        // Delete then re-add under the same name: the id comes back.
        let foe = s.sprites[1].id;
        s.remove_sprite(1);
        let i = s.add_sprite("tex", PixelRect::new(0, 0, 8, 8));
        assert_ne!(s.sprites[i].id, foe, "a new name is a new sprite");
        s.sprites[i].name = "foe".into();
        s.sprites[i].id = SubAssetId::NONE;
        s.ensure_ids("tex");
        assert_eq!(s.sprites[i].id, foe, "retired name keeps its id");
    }

    #[test]
    fn reference_matching_order_is_id_then_name() {
        let s = sheet_multi(&["a", "b"]);
        let by_id = SpriteRef { sprite: s.sprites[1].id, name: "wrong".into(), ..SpriteRef::default() };
        assert_eq!(by_id.find_in(&s.sprites), Some(1));
        let by_name = SpriteRef { sprite: SubAssetId(12345), name: "a".into(), ..SpriteRef::default() };
        assert_eq!(by_name.find_in(&s.sprites), Some(0));
        let none = SpriteRef { sprite: SubAssetId(12345), name: "zzz".into(), ..SpriteRef::default() };
        assert_eq!(none.find_in(&s.sprites), None);
        assert!(SpriteRef::default().is_none());
        let r = SpriteRef::to(None, "art\\hero.png", &s.sprites[0]);
        assert_eq!(r.path, "art/hero.png");
        let text = ron::to_string(&r).unwrap();
        assert_eq!(ron::from_str::<SpriteRef>(&text).unwrap(), r);
    }

    #[test]
    fn smart_reslice_keeps_identity_of_the_best_overlap() {
        let mut s = SpriteSheet { mode: SpriteMode::Multiple, ..SpriteSheet::default() };
        s.apply_slice("t", &grid_rects(64, 32, [32, 32], [0, 0], [0, 0]), SliceMethod::Smart);
        assert_eq!(s.sprites.len(), 2);
        let ids: Vec<_> = s.sprites.iter().map(|d| d.id).collect();
        s.sprites[0].set_pivot([0.0, 0.0]);
        s.rename_sprite(0, "idle");
        // Re-slice into 4 cells of 16x32 plus a shifted first: cells 0 and 1 overlap old 0 half each.
        let kept = s.apply_slice("t", &grid_rects(64, 32, [30, 30], [1, 1], [0, 0]), SliceMethod::Smart);
        assert_eq!(kept, 2);
        assert_eq!(s.sprites[0].id, ids[0]);
        assert_eq!(s.sprites[0].name, "idle");
        assert_eq!(s.sprites[0].pivot_preset, PivotPreset::BottomLeft, "inherited settings");
        assert_eq!(s.sprites[0].rect, PixelRect::new(1, 1, 30, 30));
        assert_eq!(s.sprites[1].id, ids[1]);
        // Safe mode adds only the non-overlapping rect.
        let mut s2 = s.clone();
        let rects = [PixelRect::new(1, 1, 30, 30), PixelRect::new(0, 100, 10, 10)];
        s2.apply_slice("t", &rects, SliceMethod::Safe);
        assert_eq!(s2.sprites.len(), 3);
        assert_eq!(s2.sprites[2].rect, PixelRect::new(0, 100, 10, 10));
        // Delete-existing starts over with generated names.
        let mut s3 = s.clone();
        s3.apply_slice("t", &[PixelRect::new(0, 0, 8, 8)], SliceMethod::DeleteExisting);
        assert_eq!(s3.sprites.len(), 1);
        assert_eq!(s3.sprites[0].name, "t_0");
    }

    #[test]
    fn unmatched_rects_get_fresh_ids_and_old_ones_are_retired() {
        let mut s = SpriteSheet { mode: SpriteMode::Multiple, ..SpriteSheet::default() };
        s.apply_slice("t", &[PixelRect::new(0, 0, 32, 32)], SliceMethod::Smart);
        let old = s.sprites[0].id;
        s.apply_slice("t", &[PixelRect::new(100, 100, 8, 8)], SliceMethod::Smart);
        assert_eq!(s.sprites.len(), 1);
        assert_ne!(s.sprites[0].id, old);
        assert_ne!(s.sprites[0].name, "");
        assert!(s.id_table.entries.iter().any(|e| e.id == old), "old id stays in the table");
    }

    #[test]
    fn grid_maths() {
        let g = grid_rects(64, 32, [16, 16], [0, 0], [0, 0]);
        assert_eq!(g.len(), 8);
        assert_eq!(g[0], PixelRect::new(0, 0, 16, 16));
        assert_eq!(g[4], PixelRect::new(0, 16, 16, 16), "rows run top to bottom");
        assert_eq!(g[7], PixelRect::new(48, 16, 16, 16));
        // Offset and padding; partial cells are skipped.
        let g = grid_rects(70, 40, [16, 16], [2, 2], [4, 4]);
        assert_eq!(g[1].x, 22);
        assert_eq!(g.len(), 3 * 2);
        assert!(grid_rects(10, 10, [0, 4], [0, 0], [0, 0]).is_empty());
        // Count -> size: floor((W - offset - padding*(n-1)) / n).
        assert_eq!(cell_size_from_count(100, 4, 0, 0), 25);
        assert_eq!(cell_size_from_count(100, 4, 4, 2), (100 - 4 - 6) / 4);
        assert_eq!(cell_size_from_count(10, 0, 0, 0), 0);
        assert_eq!(cell_size_from_count(10, 20, 0, 0), 0);
        let g = grid_rects_by_count(64, 64, 4, 2, [0, 0], [0, 0]);
        assert_eq!((g.len(), g[0].w, g[0].h), (8, 16, 32));
        // Empty cells are dropped on request.
        let alpha = AlphaMap { w: 64, h: 32, alpha: (0..64 * 32).map(|i| if i % 64 < 16 { 255 } else { 0 }).collect() };
        let kept = drop_empty(grid_rects(64, 32, [16, 32], [0, 0], [0, 0]), &alpha, 0);
        assert_eq!(kept, vec![PixelRect::new(0, 0, 16, 32)]);
    }

    #[test]
    fn pivot_presets_and_pixel_conversion() {
        let mut d = SpriteDef::named("a", PixelRect::new(0, 0, 40, 20));
        assert_eq!(d.pivot_norm(), [0.5, 0.5]);
        d.pivot_preset = PivotPreset::TopRight;
        assert_eq!(d.pivot_pixels(), [40.0, 20.0]);
        d.pivot_preset = PivotPreset::Bottom;
        assert_eq!(d.pivot_norm(), [0.5, 0.0]);
        d.set_pivot_pixels([10.0, 5.0]);
        assert_eq!(d.pivot_preset, PivotPreset::Custom);
        assert_eq!(d.pivot_norm(), [0.25, 0.25]);
        d.set_pivot([1.0, 0.5]);
        assert_eq!(d.pivot_preset, PivotPreset::Right);
        for p in PivotPreset::ALL {
            if let Some(n) = p.normalized() {
                assert_eq!(PivotPreset::detect(n), p);
            }
        }
        d.pivot_preset = PivotPreset::Custom;
        d.pivot = [f32::NAN, 0.0];
        assert!(d.pivot_norm()[0].is_finite());
    }

    #[test]
    fn world_size_and_pivot_follow_ppu() {
        let sheet = SpriteSheet { ppu: 32.0, ..SpriteSheet::default() };
        let mut d = SpriteDef::named("a", PixelRect::new(0, 0, 64, 32));
        assert_eq!(d.size_world(&sheet), [2.0, 1.0]);
        d.pivot_preset = PivotPreset::BottomLeft;
        assert_eq!(d.pivot_world(&sheet), [0.0, 0.0]);
        d.pivot_preset = PivotPreset::TopRight;
        assert_eq!(d.pivot_world(&sheet), [2.0, 1.0]);
        d.ppu = Some(64.0);
        assert_eq!(d.size_world(&sheet), [1.0, 0.5], "per-sprite ppu wins");
        d.ppu = Some(0.0);
        assert!(d.size_world(&sheet)[0].is_finite(), "ppu is clamped");
        d.ppu = Some(f32::NAN);
        assert_eq!(d.effective_ppu(&sheet), DEFAULT_PPU);
        // Border clamp.
        d.border = [50, 5, 50, 40];
        assert_eq!(d.border_clamped(), [50, 5, 14, 27]);
    }

    fn solid_alpha(w: u32, h: u32) -> AlphaMap {
        AlphaMap { w, h, alpha: vec![255; (w * h) as usize] }
    }

    #[test]
    fn geometry_is_pivot_relative_in_world_units() {
        let sheet = SpriteSheet { ppu: 10.0, mesh_type: SpriteMeshType::FullRect, ..SpriteSheet::default() };
        let mut d = SpriteDef::named("a", PixelRect::new(16, 8, 40, 20));
        d.pivot_preset = PivotPreset::BottomLeft;
        d.border = [4, 2, 4, 2];
        let g = build_geometry(&d, &sheet, 64, 32, None);
        assert!(!g.tight);
        assert_eq!(g.positions.len(), 4);
        let (lo, hi) = g.bounds();
        assert_eq!((lo, hi), ([0.0, 0.0], [4.0, 2.0]));
        assert_eq!(g.size_world, [4.0, 2.0]);
        assert_eq!(g.border_world, [0.4, 0.2, 0.4, 0.2]);
        assert_eq!(g.uv_rect, [0.25, 0.25, 0.875, 0.875]);
        // The pixel-space top-left corner maps to UV (rect.x, rect.y) and world (0, h).
        let tl = g.positions.iter().position(|p| *p == [0.0, 2.0]).unwrap();
        assert_eq!(g.uvs[tl], [0.25, 0.25]);
        // Triangles wind counter-clockwise in world space.
        for t in g.indices.as_chunks::<3>().0 {
            let (a, b, c) = (g.positions[t[0] as usize], g.positions[t[1] as usize], g.positions[t[2] as usize]);
            assert!((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) > 0.0);
        }
        // Centre pivot centres the mesh on the origin.
        d.pivot_preset = PivotPreset::Center;
        let (lo, hi) = build_geometry(&d, &sheet, 64, 32, None).bounds();
        assert_eq!((lo, hi), ([-2.0, -1.0], [2.0, 1.0]));
    }

    #[test]
    fn tight_only_at_32px_and_with_alpha() {
        let sheet = SpriteSheet::default();
        let a = AlphaMap { w: 64, h: 64, alpha: (0..64 * 64).map(|i| if (i % 64) < 32 && (i / 64) < 32 { 255 } else { 0 }).collect() };
        // 32x32 rect with solid pixels: tight (a 4 vertex outline here, still flagged tight).
        let d = SpriteDef::named("big", PixelRect::new(0, 0, 32, 32));
        assert!(build_geometry(&d, &sheet, 64, 64, Some(&a)).tight);
        // 31 px wide: quad, even though Tight is requested.
        let d = SpriteDef::named("small", PixelRect::new(0, 0, 31, 32));
        assert!(!build_geometry(&d, &sheet, 64, 64, Some(&a)).tight);
        // Fully transparent rect: quad fallback.
        let d = SpriteDef::named("empty", PixelRect::new(32, 32, 32, 32));
        let g = build_geometry(&d, &sheet, 64, 64, Some(&a));
        assert!(!g.tight);
        assert_eq!(g.positions.len(), 4);
        // No pixels available: quad.
        assert!(!build_geometry(&SpriteDef::named("x", PixelRect::new(0, 0, 64, 64)), &sheet, 64, 64, None).tight);
        // FullRect override wins over a Tight sheet.
        let mut d = SpriteDef::named("fr", PixelRect::new(0, 0, 64, 64));
        d.mesh_type = Some(SpriteMeshType::FullRect);
        assert!(!build_geometry(&d, &sheet, 64, 64, Some(&a)).tight);
        // A custom outline makes the 31 px sprite tight.
        let mut d = SpriteDef::named("c", PixelRect::new(0, 0, 20, 20));
        d.custom_outline = vec![vec![[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]]];
        let g = build_geometry(&d, &sheet, 64, 64, Some(&a));
        assert!(g.tight);
        assert_eq!(g.positions.len(), 3);
    }

    #[test]
    fn physics_shape_custom_auto_and_off() {
        let a = solid_alpha(64, 64);
        let mut sheet = SpriteSheet::default();
        let mut d = SpriteDef::named("a", PixelRect::new(0, 0, 64, 64));
        d.pivot_preset = PivotPreset::BottomLeft;
        sheet.ppu = 64.0;
        let g = build_geometry(&d, &sheet, 64, 64, Some(&a));
        assert_eq!(g.physics.len(), 1, "auto outline");
        let (lo, hi) = g.physics[0].iter().fold(([f32::MAX; 2], [f32::MIN; 2]), |(lo, hi), p| ([lo[0].min(p[0]), lo[1].min(p[1])], [hi[0].max(p[0]), hi[1].max(p[1])]));
        assert_eq!((lo, hi), ([0.0, 0.0], [1.0, 1.0]));
        d.physics_shape = vec![vec![[0.0, 0.0], [32.0, 0.0], [0.0, 32.0]]];
        assert_eq!(build_geometry(&d, &sheet, 64, 64, Some(&a)).physics[0].len(), 3, "custom wins");
        d.physics_shape.clear();
        sheet.generate_physics_shape = false;
        assert!(build_geometry(&d, &sheet, 64, 64, Some(&a)).physics.is_empty());
    }

    #[test]
    fn pack_hints_rules() {
        let h = PackHints::default();
        assert_eq!(h.footprint(PixelRect::new(0, 0, 10, 20)), (18, 28));
        assert_eq!(h.edge_fill(SpriteMeshType::FullRect), EdgeFill::ExtrudeEdge);
        assert_eq!(h.edge_fill(SpriteMeshType::Tight), EdgeFill::AlphaDilate);
        assert!(h.alpha_dilate && !h.allow_rotation);
    }

    #[test]
    fn warnings_report_overlap_bounds_and_small_tight() {
        let mut s = sheet_multi(&["a", "b"]);
        s.sprites[1].rect = PixelRect::new(16, 0, 32, 32);
        let w = s.warnings(64, 64);
        assert!(w.iter().any(|m| m.contains("overlap")));
        s.sprites[0].rect = PixelRect::new(60, 0, 32, 32);
        assert!(s.warnings(64, 64).iter().any(|m| m.contains("outside")));
        s.sprites[0].rect = PixelRect::new(0, 0, 8, 8);
        assert!(s.warnings(64, 64).iter().any(|m| m.contains("quad")));
        s.mode = SpriteMode::Single;
        assert!(s.warnings(64, 64).is_empty());
    }

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("zest-sprite-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn sprite_ref_text_form_round_trips() {
        let id = SubAssetId(0x00a1_b2c3_d4e5_f607);
        let s = sprite_ref_string("Sprites\\hero.png", id);
        assert_eq!(s, "Sprites/hero.png#00a1b2c3d4e5f607");
        let r = SpriteRef::from_ref_string(&s).unwrap();
        assert_eq!((r.path.as_str(), r.sprite), ("Sprites/hero.png", id));
        assert_eq!(r.to_ref_string(), s);
        // A path may contain '#': the last one splits.
        assert_eq!(SpriteRef::from_ref_string("a#b/c.png#00a1b2c3d4e5f607").unwrap().path, "a#b/c.png");
        for bad in ["", "hero.png", "#00a1b2c3d4e5f607", "hero.png#123", "hero.png#zzzzzzzzzzzzzzzz", "hero.png#0000000000000000"] {
            assert!(SpriteRef::from_ref_string(bad).is_none(), "{bad}");
        }
        assert_eq!(SpriteRef::default().to_ref_string(), "");
    }

    #[test]
    fn load_table_from_disk_and_resolve_a_reference() {
        let dir = tmp("load");
        let png = dir.join("sheet.png");
        let mut img = image::RgbaImage::new(64, 32);
        for y in 0..32 {
            for x in 0..64 {
                if (x % 32) > 4 && (x % 32) < 28 && y > 4 && y < 28 {
                    img.put_pixel(x, y, image::Rgba([255, 0, 0, 255]));
                }
            }
        }
        img.save(&png).unwrap();
        let mut meta = AssetMeta::new_for(&png);
        let mut t = TextureImport { configured: true, ..TextureImport::default() };
        t.apply_kind(TextureKind::Sprite);
        t.sprite.mode = SpriteMode::Multiple;
        t.sprite.apply_slice("sheet", &grid_rects(64, 32, [32, 32], [0, 0], [0, 0]), SliceMethod::Smart);
        let second = t.sprite.sprites[1].clone();
        meta.importer = ImporterSettings::Texture(t);
        crate::save_asset_meta(&png, &meta).unwrap();

        assert_eq!(sprite_count(&png), Some(2));
        let kids = sprite_children(&png, "sheet.png");
        assert_eq!(kids.len(), 2);
        assert_eq!(kids[1].0, second.id.0);
        assert_eq!(kids[1].1, second.name);
        assert_eq!(SpriteRef::from_ref_string(&kids[1].2).unwrap().sprite, second.id);
        let table = load_sprite_table(&png).unwrap().unwrap();
        assert_eq!((table.width, table.height, table.sprites.len()), (64, 32, 2));
        assert_eq!(table.guid, meta.id);
        let g = table.geometry(1).unwrap();
        assert!(g.tight, "32x32 sprite with an opaque block gets a tight mesh");
        assert_eq!(g.positions.len(), 4);
        assert_eq!(table.all_geometry().len(), 2);

        // Reference by guid + id, then after a rename, then with a stale path.
        let r = SpriteRef::to(Some(meta.id), "sheet.png", &second);
        let (tb, i) = resolve_sprite_ref(&dir, &r).unwrap();
        assert_eq!(tb.sprites[i].id, second.id);
        let moved = SpriteRef { path: "gone/sheet.png".into(), ..r.clone() };
        let (_, i) = resolve_sprite_ref(&dir, &moved).unwrap();
        assert_eq!(i, 1, "found through the GUID search");
        let wrong = SpriteRef { texture: Some(ObjectId::new()), ..r };
        assert!(resolve_sprite_ref(&dir, &wrong).is_none());

        // Not a sprite texture any more: no table, no badge.
        let mut meta2 = crate::load_asset_meta(&png).unwrap().unwrap();
        meta2.importer.as_texture_mut().unwrap().apply_kind(TextureKind::Default);
        crate::save_asset_meta(&png, &meta2).unwrap();
        assert!(load_sprite_table(&png).unwrap().is_none());
        assert_eq!(sprite_count(&png), None);
        assert!(sprite_children(&png, "sheet.png").is_empty());
        // ...but the table is still stored for when it comes back.
        assert_eq!(meta2.importer.as_texture().unwrap().sprite.sprites.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
