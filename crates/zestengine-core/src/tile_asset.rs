//! Tile assets (2D-5): `.tile` ([`TileAsset`], optionally animated), `.ruletile`
//! ([`RuleTile`]) and `.palette` ([`TilePalette`]), plus the rule precompiler.
//! RON files, every field defaulted so old files keep loading.
//!
//! # Rule tiles
//!
//! A rule lists, per neighbour slot of the ring (8 around a rectangle /
//! isometric cell, 6 around a hexagon, see [`crate::neighbor_offset`]), whether
//! that neighbour must be this tile ([`NeighborReq::This`]), must not
//! ([`NeighborReq::NotThis`]) or does not matter. The first rule (in list
//! order) that matches wins; a rule can also match rotated or mirrored
//! ([`RuleTransform`]), which draws its sprite with that transform.
//!
//! [`RuleTile::compile`] turns the rule list into a lookup table once: the
//! "same tile" state of the ring is a bitmask (2^8 or 2^6 values), and for each
//! mask the table holds the winning rule and transform. At runtime a cell
//! costs one mask build (8 neighbour reads) and one table read, independent of
//! the number of rules. Editing a cell only changes the masks of its 8
//! neighbours.
//!
//! "Same tile" compares the neighbour's tile asset path with the cell's. The
//! random output picks a sprite by value noise of the cell position (stable
//! across refreshes, not Perlin); animated outputs use up to four speed buckets
//! between the min and max speed so equal animations still batch.

use std::path::Path;

use glam::IVec2;
use serde::{Deserialize, Serialize};

use crate::tilemap::{CellLayout, TileXform, ROT_STEPS, neighbor_offset};

/// File extension of `.tile` assets.
pub const TILE_EXTENSION: &str = "tile";
/// File extension of `.ruletile` assets.
pub const RULETILE_EXTENSION: &str = "ruletile";
/// File extension of `.palette` assets.
pub const PALETTE_EXTENSION: &str = "palette";

/// Which collider a tile contributes to a `TilemapCollider2D`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum TileColliderType {
    /// No collider.
    None,
    /// The sprite's physics outline.
    #[default]
    Sprite,
    /// The whole cell.
    Grid,
}

impl TileColliderType {
    /// Every type, in declaration order.
    pub const ALL: [TileColliderType; 3] = [Self::None, Self::Sprite, Self::Grid];
    /// Inspector label of this type.
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Sprite => "Sprite",
            Self::Grid => "Grid",
        }
    }
}

fn white() -> [f32; 4] {
    [1.0; 4]
}

/// A frame animation for a tile.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TileAnimation {
    /// Sprite references (`path#id`), one per frame.
    pub sprites: Vec<String>,
    /// Playback speed range (frames per second = speed x the map's
    /// animation frame rate); each cell gets one of four speeds in the range.
    pub min_speed: f32,
    /// Fastest playback speed of the range.
    pub max_speed: f32,
    /// Seconds into the loop every cell starts at.
    pub start_time: f32,
    /// Play once and hold the last frame.
    pub loop_once: bool,
}

impl Default for TileAnimation {
    fn default() -> Self {
        Self { sprites: Vec::new(), min_speed: 1.0, max_speed: 1.0, start_time: 0.0, loop_once: false }
    }
}

/// A `.tile`: one sprite (or an animation) with colour, collider, flags and
/// transform.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TileAsset {
    /// `path#id` sprite reference; empty draws nothing.
    pub sprite: String,
    /// Tint, linear RGBA.
    pub color: [f32; 4],
    /// Which collider the tile contributes.
    pub collider: TileColliderType,
    /// `FLAG_LOCK_*` bits: lock the colour / transform against per-cell overrides.
    pub flags: u8,
    /// Tile transform (rotation / mirror).
    pub transform: TileXform,
    /// `Some` makes it an animated tile; the frames replace `sprite`.
    pub animation: Option<TileAnimation>,
    /// A prefab (project-relative `.prefab`) instantiated once in every cell that uses this tile (a
    /// GameObject tile: see `tile_object.rs`). Empty = none; the tile can still draw a sprite too.
    pub game_object: String,
}

impl Default for TileAsset {
    fn default() -> Self {
        Self { sprite: String::new(), color: white(), collider: TileColliderType::Sprite, flags: 0, transform: TileXform::IDENTITY, animation: None, game_object: String::new() }
    }
}

impl TileAsset {
    /// A tile drawing this sprite.
    pub fn with_sprite(sprite: &str) -> Self {
        Self { sprite: sprite.to_string(), ..Default::default() }
    }
    /// The sprites a cell of this tile can show.
    pub fn sprites(&self) -> Vec<&str> {
        match &self.animation {
            Some(a) if !a.sprites.is_empty() => a.sprites.iter().map(String::as_str).collect(),
            _ if self.sprite.is_empty() => Vec::new(),
            _ => vec![self.sprite.as_str()],
        }
    }
}

/// What a neighbour must be for a rule to match.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum NeighborReq {
    /// Does not matter.
    #[default]
    Any,
    /// Must be this tile.
    This,
    /// Must not be this tile.
    NotThis,
}

impl NeighborReq {
    /// Inspector label of this requirement.
    pub fn label(self) -> &'static str {
        match self {
            Self::Any => "Any",
            Self::This => "This",
            Self::NotThis => "Not This",
        }
    }
    /// Click-to-cycle order of the rule editor.
    pub fn cycled(self) -> Self {
        match self {
            Self::Any => Self::This,
            Self::This => Self::NotThis,
            Self::NotThis => Self::Any,
        }
    }
}

/// Which orientations of a rule are also tried.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum RuleTransform {
    /// Only as written.
    #[default]
    Fixed,
    /// Every rotation (4 on rectangle / isometric grids, 6 on hexagon grids).
    Rotated,
    /// As written and mirrored in X.
    MirrorX,
    /// As written and mirrored in Y.
    MirrorY,
    /// As written and mirrored both ways.
    MirrorXY,
    /// Every rotation, each also mirrored.
    RotatedMirror,
}

impl RuleTransform {
    /// Every transform, in declaration order.
    pub const ALL: [RuleTransform; 6] = [Self::Fixed, Self::Rotated, Self::MirrorX, Self::MirrorY, Self::MirrorXY, Self::RotatedMirror];
    /// Inspector label of this transform.
    pub fn label(self) -> &'static str {
        match self {
            Self::Fixed => "Fixed",
            Self::Rotated => "Rotated",
            Self::MirrorX => "Mirror X",
            Self::MirrorY => "Mirror Y",
            Self::MirrorXY => "Mirror XY",
            Self::RotatedMirror => "Rotated + Mirror",
        }
    }
}

/// How a rule picks its sprite.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum RuleOutput {
    /// One sprite (the first).
    #[default]
    Single,
    /// One of the sprites by position noise.
    Random,
    /// All sprites as an animation.
    Animation,
}

impl RuleOutput {
    /// Every output, in declaration order.
    pub const ALL: [RuleOutput; 3] = [Self::Single, Self::Random, Self::Animation];
    /// Inspector label of this output.
    pub fn label(self) -> &'static str {
        match self {
            Self::Single => "Single",
            Self::Random => "Random",
            Self::Animation => "Animation",
        }
    }
}

fn eight_any() -> Vec<NeighborReq> {
    vec![NeighborReq::Any; 8]
}

/// One rule of a [`RuleTile`].
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TilingRule {
    /// One entry per ring slot (8, or 6 on hexagon grids); missing slots count as Any.
    pub neighbors: Vec<NeighborReq>,
    /// Which orientations also match.
    pub transform: RuleTransform,
    /// How the sprite is picked.
    pub output: RuleOutput,
    /// Sprite references (`path#id`).
    pub sprites: Vec<String>,
    /// Which collider the output contributes.
    pub collider: TileColliderType,
    /// Animation speed range (Animation output).
    pub min_speed: f32,
    /// Fastest animation speed (Animation output).
    pub max_speed: f32,
    /// Noise frequency of the Random output (cells per noise lattice step = 1/scale).
    pub random_scale: f32,
    /// Random output: also rotate by a random multiple of the layout's step.
    pub random_rotation: bool,
}

impl Default for TilingRule {
    fn default() -> Self {
        Self {
            neighbors: eight_any(),
            transform: RuleTransform::Fixed,
            output: RuleOutput::Single,
            sprites: Vec::new(),
            collider: TileColliderType::Sprite,
            min_speed: 1.0,
            max_speed: 1.0,
            random_scale: 0.5,
            random_rotation: false,
        }
    }
}

impl TilingRule {
    /// The requirement for ring slot `i` (missing slots count as Any).
    pub fn req(&self, slot: usize) -> NeighborReq {
        self.neighbors.get(slot).copied().unwrap_or_default()
    }
}

/// A `.ruletile`: sprites chosen by the surrounding tiles.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct RuleTile {
    /// Fixes the ring (8 slots, or 6 for a hexagon grid) and the rotation step.
    pub layout: CellLayout,
    /// Sprite shown when no rule matches.
    pub default_sprite: String,
    /// Collider of the default output.
    pub default_collider: TileColliderType,
    /// Tint, linear RGBA.
    pub color: [f32; 4],
    /// The rules, first match wins.
    pub rules: Vec<TilingRule>,
}

impl Default for RuleTile {
    fn default() -> Self {
        Self { layout: CellLayout::Rectangle, default_sprite: String::new(), default_collider: TileColliderType::Sprite, color: white(), rules: Vec::new() }
    }
}

// ---------------------------------------------------------------------------
// Ring geometry and the precompiler
// ---------------------------------------------------------------------------

/// How ring slots move under rotation and mirroring.
#[derive(Clone, Copy, Debug)]
pub struct RingSpec {
    /// Slots in the ring.
    pub n: usize,
    /// Distinct rotations (4 for rectangles, 6 for hexagons).
    pub rotations: usize,
    /// Ring slots one rotation step moves a neighbour.
    pub slots_per_rotation: usize,
    /// [`TileXform::rot`] units of one rotation step.
    pub rot15: u8,
    /// `slot -> (mirror_x - slot) mod n` is the x mirror.
    pub mirror_x: usize,
}

/// The ring geometry of a layout (slot count, rotations, mirror axis).
pub fn ring_spec(layout: CellLayout) -> RingSpec {
    match layout {
        CellLayout::HexPointy => RingSpec { n: 6, rotations: 6, slots_per_rotation: 1, rot15: ROT_STEPS / 6, mirror_x: 3 },
        CellLayout::HexFlat => RingSpec { n: 6, rotations: 6, slots_per_rotation: 1, rot15: ROT_STEPS / 6, mirror_x: 2 },
        _ => RingSpec { n: 8, rotations: 4, slots_per_rotation: 2, rot15: ROT_STEPS / 4, mirror_x: 4 },
    }
}

impl RingSpec {
    /// Where ring slot `i` lands after mirroring (x -> -x) and then rotating
    /// counter clockwise by `k` steps.
    pub fn permute(&self, k: usize, mirror: bool, i: usize) -> usize {
        let j = if mirror { (self.mirror_x + self.n - i % self.n) % self.n } else { i % self.n };
        (j + k * self.slots_per_rotation) % self.n
    }

    /// The `(rotation steps, mirror)` orientations a rule transform tries, in
    /// priority order (the rule as written first).
    pub fn variants(&self, t: RuleTransform) -> Vec<(usize, bool)> {
        let half = self.rotations / 2;
        match t {
            RuleTransform::Fixed => vec![(0, false)],
            RuleTransform::Rotated => (0..self.rotations).map(|k| (k, false)).collect(),
            RuleTransform::MirrorX => vec![(0, false), (0, true)],
            // Mirror Y = mirror X then a half turn.
            RuleTransform::MirrorY => vec![(0, false), (half, true)],
            RuleTransform::MirrorXY => vec![(0, false), (0, true), (half, true), (half, false)],
            RuleTransform::RotatedMirror => (0..self.rotations).flat_map(|k| [(k, false), (k, true)]).collect(),
        }
    }

    /// The tile transform that draws a sprite designed for the rule as written
    /// when the variant `(k, mirror)` matched.
    pub fn xform(&self, k: usize, mirror: bool) -> TileXform {
        TileXform::new((k as u8) * self.rot15, mirror)
    }
}

/// The winning rule for one neighbour configuration.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RuleHit {
    /// Index into [`RuleTile::rules`]; [`RuleHit::NONE`] = no rule matched.
    pub rule: u16,
    /// The tile transform that draws the matched orientation.
    pub xform: TileXform,
}

impl RuleHit {
    /// The "no rule matched" sentinel.
    pub const NONE: u16 = u16::MAX;
    /// True when a rule matched.
    pub fn matched(&self) -> bool {
        self.rule != Self::NONE
    }
}

/// A rule tile compiled to a lookup table indexed by the neighbour mask.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledRules {
    /// The layout the table was compiled for.
    pub layout: CellLayout,
    /// Ring slots (mask bits).
    pub n: usize,
    table: Vec<RuleHit>,
}

impl CompiledRules {
    /// The winner for a mask (bit `i` set = ring slot `i` holds the same tile).
    #[inline]
    pub fn lookup(&self, mask: u32) -> RuleHit {
        self.table[(mask as usize) & (self.table.len() - 1)]
    }
    /// Number of entries in the lookup table.
    pub fn table_len(&self) -> usize {
        self.table.len()
    }
}

impl RuleTile {
    /// Compile the rules into the bitmask table (2^8 or 2^6 entries).
    pub fn compile(&self) -> CompiledRules {
        let spec = ring_spec(self.layout);
        // Per rule and variant: the slots that must be set / clear.
        let mut variants: Vec<(u16, u32, u32, TileXform)> = Vec::new();
        for (ri, r) in self.rules.iter().enumerate() {
            for (k, mirror) in spec.variants(r.transform) {
                let (mut set, mut clear) = (0u32, 0u32);
                for i in 0..spec.n {
                    let to = spec.permute(k, mirror, i);
                    match r.req(i) {
                        NeighborReq::This => set |= 1 << to,
                        NeighborReq::NotThis => clear |= 1 << to,
                        NeighborReq::Any => {}
                    }
                }
                variants.push((ri as u16, set, clear, spec.xform(k, mirror)));
            }
        }
        let none = RuleHit { rule: RuleHit::NONE, xform: TileXform::IDENTITY };
        let table = (0..1u32 << spec.n)
            .map(|mask| {
                variants
                    .iter()
                    .find(|(_, set, clear, _)| mask & set == *set && mask & clear == 0)
                    .map_or(none, |&(rule, _, _, xform)| RuleHit { rule, xform })
            })
            .collect();
        CompiledRules { layout: self.layout, n: spec.n, table }
    }

    /// Every sprite any output of this tile can show.
    pub fn all_sprites(&self) -> Vec<&str> {
        let mut v: Vec<&str> = Vec::new();
        if !self.default_sprite.is_empty() {
            v.push(&self.default_sprite);
        }
        for r in &self.rules {
            for s in &r.sprites {
                if !s.is_empty() && !v.contains(&s.as_str()) {
                    v.push(s);
                }
            }
        }
        v
    }
}

/// The neighbour mask of `cell`: bit `i` set when ring slot `i` satisfies
/// `same` (it holds the same tile). Reads 8 (or 6) neighbours.
pub fn neighbor_mask(layout: CellLayout, cell: IVec2, same: impl Fn(IVec2) -> bool) -> u32 {
    let n = layout.neighbor_count();
    let mut mask = 0u32;
    for i in 0..n {
        if same(cell + neighbor_offset(layout, cell, i)) {
            mask |= 1 << i;
        }
    }
    mask
}

/// The cells whose appearance can change when `cell` changes: itself and its
/// ring (8 or 6 neighbours).
pub fn refresh_cells(layout: CellLayout, cell: IVec2) -> Vec<IVec2> {
    let mut v = vec![cell];
    v.extend((0..layout.neighbor_count()).map(|i| cell + neighbor_offset(layout, cell, i)));
    v
}

/// Smooth value noise in 0..1 at a lattice position (bilinear over hashed corners).
pub fn value_noise(x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let h = |ix: i32, iy: i32| -> f32 {
        let mut v = (ix as u32).wrapping_mul(0x9E37_79B1) ^ (iy as u32).wrapping_mul(0x85EB_CA6B) ^ 0x27D4_EB2F;
        v ^= v >> 15;
        v = v.wrapping_mul(0x2C1B_3C6D);
        v ^= v >> 12;
        (v >> 8) as f32 / (1u32 << 24) as f32
    };
    let (ix, iy) = (x0 as i32, y0 as i32);
    let s = |t: f32| t * t * (3.0 - 2.0 * t);
    let (sx, sy) = (s(fx), s(fy));
    let top = h(ix, iy) + (h(ix + 1, iy) - h(ix, iy)) * sx;
    let bottom = h(ix, iy + 1) + (h(ix + 1, iy + 1) - h(ix, iy + 1)) * sx;
    top + (bottom - top) * sy
}

/// A hash of a cell in 0..1 (per-cell variation that is not smooth).
pub fn cell_hash(cell: IVec2, salt: u32) -> f32 {
    let mut v = (cell.x as u32).wrapping_mul(0x9E37_79B1) ^ (cell.y as u32).wrapping_mul(0x85EB_CA6B) ^ salt.wrapping_mul(0xC2B2_AE35);
    v ^= v >> 16;
    v = v.wrapping_mul(0x7FEB_352D);
    v ^= v >> 15;
    v = v.wrapping_mul(0x846C_A68B);
    v ^= v >> 16;
    (v >> 8) as f32 / (1u32 << 24) as f32
}

// ---------------------------------------------------------------------------
// Selection: what one cell shows
// ---------------------------------------------------------------------------

/// Animation parameters of a selected output.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct AnimSel {
    /// Speed multiplier (frames per second = this x the map's frame rate).
    pub speed: f32,
    /// Seconds into the loop the cell starts at.
    pub start_time: f32,
    /// Play once and hold the last frame.
    pub loop_once: bool,
}

/// What a cell of a tile shows: the sprite(s) (one unless animated), the
/// tile-side transform and colour, the collider type.
#[derive(Clone, PartialEq, Debug)]
pub struct TileSel<'a> {
    /// Sprite references shown (one unless animated).
    pub sprites: &'a [String],
    /// Tile-side transform.
    pub xform: TileXform,
    /// Tint, linear RGBA.
    pub color: [f32; 4],
    /// Collider type.
    pub collider: TileColliderType,
    /// `FLAG_LOCK_*` bits.
    pub flags: u8,
    /// Animation parameters, when animated.
    pub anim: Option<AnimSel>,
}

fn speed_bucket(cell: IVec2, min: f32, max: f32) -> f32 {
    if (max - min).abs() < 1e-6 {
        return min;
    }
    let b = (cell_hash(cell, 77) * 4.0).floor().min(3.0);
    min + (max - min) * (b / 3.0)
}

impl TileAsset {
    /// What one cell of this tile shows.
    pub fn select(&self, cell: IVec2) -> TileSel<'_> {
        match &self.animation {
            Some(a) if !a.sprites.is_empty() => TileSel {
                sprites: &a.sprites,
                xform: self.transform,
                color: self.color,
                collider: self.collider,
                flags: self.flags,
                anim: (a.sprites.len() > 1).then(|| AnimSel { speed: speed_bucket(cell, a.min_speed, a.max_speed), start_time: a.start_time, loop_once: a.loop_once }),
            },
            _ => TileSel {
                sprites: std::slice::from_ref(&self.sprite),
                xform: self.transform,
                color: self.color,
                collider: self.collider,
                flags: self.flags,
                anim: None,
            },
        }
    }
}

impl RuleTile {
    /// The output for `cell` given the table hit of its neighbour mask.
    pub fn select(&self, hit: RuleHit, cell: IVec2) -> TileSel<'_> {
        let rule = self.rules.get(hit.rule as usize).filter(|_| hit.matched());
        let Some(r) = rule else {
            return TileSel {
                sprites: std::slice::from_ref(&self.default_sprite),
                xform: TileXform::IDENTITY,
                color: self.color,
                collider: self.default_collider,
                flags: crate::tilemap::FLAG_LOCK_TRANSFORM,
                anim: None,
            };
        };
        let sprites: &[String] = if r.sprites.is_empty() { std::slice::from_ref(&self.default_sprite) } else { &r.sprites };
        let mut xform = hit.xform;
        let (sprites, anim) = match r.output {
            RuleOutput::Single => (&sprites[..1], None),
            RuleOutput::Random => {
                let n = value_noise((cell.x as f32 + 100_000.0) * r.random_scale, (cell.y as f32 + 100_000.0) * r.random_scale);
                let i = ((n * sprites.len() as f32) as usize).min(sprites.len() - 1);
                if r.random_rotation {
                    let spec = ring_spec(self.layout);
                    let k = (cell_hash(cell, 5) * spec.rotations as f32) as usize % spec.rotations;
                    xform = xform.rotated_ccw(k as u8 * spec.rot15);
                }
                (&sprites[i..=i], None)
            }
            RuleOutput::Animation => (sprites, (sprites.len() > 1).then(|| AnimSel { speed: speed_bucket(cell, r.min_speed, r.max_speed), start_time: 0.0, loop_once: false })),
        };
        TileSel { sprites, xform, color: self.color, collider: r.collider, flags: crate::tilemap::FLAG_LOCK_TRANSFORM, anim }
    }
}

// ---------------------------------------------------------------------------
// Palette
// ---------------------------------------------------------------------------

/// A `.palette`: tiles laid out on a grid for painting.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TilePalette {
    /// Cell layout of the palette grid.
    pub layout: CellLayout,
    /// Palette cell size.
    pub cell_size: [f32; 3],
    /// `((x, y), tile asset path)`.
    pub cells: Vec<((i32, i32), String)>,
}

impl Default for TilePalette {
    fn default() -> Self {
        Self { layout: CellLayout::Rectangle, cell_size: [1.0; 3], cells: Vec::new() }
    }
}

impl TilePalette {
    /// The tile asset path at (x, y).
    pub fn tile_at(&self, x: i32, y: i32) -> Option<&str> {
        self.cells.iter().find(|(p, _)| *p == (x, y)).map(|(_, t)| t.as_str())
    }
    /// Set a cell (an empty path removes it).
    pub fn set(&mut self, x: i32, y: i32, tile: &str) {
        self.cells.retain(|(p, _)| *p != (x, y));
        if !tile.is_empty() {
            self.cells.push(((x, y), tile.to_string()));
        }
    }
    /// Lay `tiles` out row by row, `columns` per row, top row first (y falls).
    pub fn from_tiles(tiles: &[String], columns: usize) -> Self {
        let columns = columns.max(1);
        let rows = tiles.len().div_ceil(columns) as i32;
        let cells = tiles.iter().enumerate().map(|(i, t)| (((i % columns) as i32, rows - 1 - (i / columns) as i32), t.clone())).collect();
        Self { cells, ..Default::default() }
    }
    /// The cells' inclusive bounds (None for an empty palette).
    pub fn bounds(&self) -> Option<(IVec2, IVec2)> {
        self.cells.iter().fold(None, |b, ((x, y), _)| {
            let c = IVec2::new(*x, *y);
            Some(b.map_or((c, c), |(lo, hi): (IVec2, IVec2)| (lo.min(c), hi.max(c))))
        })
    }
}

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------

/// A loaded tile file.
#[derive(Clone, PartialEq, Debug)]
pub enum TileFile {
    /// A `.tile`.
    Tile(TileAsset),
    /// A `.ruletile`.
    Rule(RuleTile),
}

impl TileFile {
    /// Load by extension (`.tile` or `.ruletile`).
    pub fn load(path: &Path) -> Result<TileFile, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        match ext.as_str() {
            RULETILE_EXTENSION => ron::from_str(&text).map(TileFile::Rule).map_err(|e| format!("{}: {e}", path.display())),
            _ => ron::from_str(&text).map(TileFile::Tile).map_err(|e| format!("{}: {e}", path.display())),
        }
    }

    /// Serialize and write, by extension (`.tile` or `.ruletile`).
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let cfg = ron::ser::PrettyConfig::default();
        let text = match self {
            TileFile::Tile(t) => ron::ser::to_string_pretty(t, cfg),
            TileFile::Rule(r) => ron::ser::to_string_pretty(r, cfg),
        }
        .map_err(|e| e.to_string())?;
        std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// Read and parse a `.palette`.
pub fn load_palette(path: &Path) -> Result<TilePalette, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    ron::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// Serialize and write a `.palette`.
pub fn save_palette(path: &Path, p: &TilePalette) -> Result<(), String> {
    let text = ron::ser::to_string_pretty(p, ron::ser::PrettyConfig::default()).map_err(|e| e.to_string())?;
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tilemap::Tilemap;

    fn req(spec: &[(usize, NeighborReq)]) -> Vec<NeighborReq> {
        let mut v = vec![NeighborReq::Any; 8];
        for &(i, r) in spec {
            v[i] = r;
        }
        v
    }

    /// Ring slots: 0 E, 1 NE, 2 N, 3 NW, 4 W, 5 SW, 6 S, 7 SE.
    fn wall_tile() -> RuleTile {
        use NeighborReq::*;
        let rule = |n: Vec<NeighborReq>, t, s: &str| TilingRule { neighbors: n, transform: t, sprites: vec![s.to_string()], ..Default::default() };
        RuleTile {
            default_sprite: "def#0000000000000001".into(),
            rules: vec![
                // Lone block: nothing on the four sides.
                rule(req(&[(0, NotThis), (2, NotThis), (4, NotThis), (6, NotThis)]), RuleTransform::Fixed, "lone#0000000000000001"),
                // A top edge: ground to the south, none north; also rotated.
                rule(req(&[(2, NotThis), (6, This)]), RuleTransform::Rotated, "edge#0000000000000001"),
                // A cap on the east side only, mirrored.
                rule(req(&[(0, This), (4, NotThis)]), RuleTransform::MirrorX, "cap#0000000000000001"),
            ],
            ..Default::default()
        }
    }

    #[test]
    fn compile_matches_in_list_order_with_transforms() {
        let t = wall_tile();
        let c = t.compile();
        assert_eq!(c.table_len(), 256);
        // Nothing around: the first rule.
        assert_eq!(c.lookup(0), RuleHit { rule: 0, xform: TileXform::IDENTITY });
        // South neighbour only: the edge rule as written.
        assert_eq!(c.lookup(1 << 6).rule, 1);
        assert_eq!(c.lookup(1 << 6).xform, TileXform::IDENTITY);
        // North neighbour only: the same edge rule rotated a half turn.
        let h = c.lookup(1 << 2);
        assert_eq!((h.rule, h.xform), (1, TileXform::new(12, false)));
        // East only: the edge rule rotated (S -> E needs a quarter turn) matches before `cap`.
        assert_eq!(c.lookup(1 << 0).rule, 1);
        // West and east: no rule, falls to the default.
        assert!(!c.lookup((1 << 0) | (1 << 4) | (1 << 2) | (1 << 6)).matched());
    }

    /// An independent matcher: no bitmask, no ring permutation. The rule's
    /// offsets are transformed geometrically and compared against a real tilemap.
    fn brute(rules: &[TilingRule], map: &Tilemap, cell: IVec2) -> RuleHit {
        let me = map.tile_at(cell).unwrap().to_string();
        let same = |c: IVec2| map.tile_at(c) == Some(me.as_str());
        let rot = |o: IVec2, k: usize| (0..k).fold(o, |o, _| IVec2::new(-o.y, o.x));
        for (ri, r) in rules.iter().enumerate() {
            let variants: Vec<(usize, bool)> = match r.transform {
                RuleTransform::Fixed => vec![(0, false)],
                RuleTransform::Rotated => (0..4).map(|k| (k, false)).collect(),
                RuleTransform::MirrorX => vec![(0, false), (0, true)],
                RuleTransform::MirrorY => vec![(0, false), (2, true)],
                RuleTransform::MirrorXY => vec![(0, false), (0, true), (2, true), (2, false)],
                RuleTransform::RotatedMirror => (0..4).flat_map(|k| [(k, false), (k, true)]).collect(),
            };
            for (k, mirror) in variants {
                let ok = (0..8).all(|i| {
                    let o = neighbor_offset(CellLayout::Rectangle, cell, i);
                    let o = if mirror { IVec2::new(-o.x, o.y) } else { o };
                    let to = rot(o, k);
                    match r.req(i) {
                        NeighborReq::Any => true,
                        NeighborReq::This => same(cell + to),
                        NeighborReq::NotThis => !same(cell + to),
                    }
                });
                if ok {
                    return RuleHit { rule: ri as u16, xform: TileXform::new(k as u8 * 6, mirror) };
                }
            }
        }
        RuleHit { rule: RuleHit::NONE, xform: TileXform::IDENTITY }
    }

    #[test]
    fn precompiled_table_equals_brute_force_on_random_neighbourhoods() {
        use NeighborReq::*;
        let mut seed = 0x1234_5678u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for round in 0..40 {
            // Random rules over all transforms.
            let rules: Vec<TilingRule> = (0..1 + rnd() % 6)
                .map(|_| TilingRule {
                    neighbors: (0..8).map(|_| [Any, Any, This, NotThis][(rnd() % 4) as usize]).collect(),
                    transform: RuleTransform::ALL[(rnd() % 6) as usize],
                    ..Default::default()
                })
                .collect();
            let tile = RuleTile { rules: rules.clone(), ..Default::default() };
            let compiled = tile.compile();
            // A random map of two tile kinds in a 9 x 9 area.
            let mut map = Tilemap::default();
            for x in -4..5 {
                for y in -4..5 {
                    match rnd() % 3 {
                        0 => {}
                        1 => {
                            map.set_tile(IVec2::new(x, y), Some("me.ruletile"));
                        }
                        _ => {
                            map.set_tile(IVec2::new(x, y), Some("other.tile"));
                        }
                    }
                }
            }
            for x in -3..4 {
                for y in -3..4 {
                    let cell = IVec2::new(x, y);
                    if map.tile_at(cell) != Some("me.ruletile") {
                        continue;
                    }
                    let mask = neighbor_mask(CellLayout::Rectangle, cell, |c| map.tile_at(c) == Some("me.ruletile"));
                    let want = brute(&rules, &map, cell);
                    let got = compiled.lookup(mask);
                    assert_eq!(got, want, "round {round} cell {cell} mask {mask:08b}");
                }
            }
        }
    }

    #[test]
    fn hex_rotation_cycles_through_six_and_mirror_involutes() {
        for layout in [CellLayout::HexPointy, CellLayout::HexFlat] {
            let spec = ring_spec(layout);
            for i in 0..6 {
                // Six rotations return to the start; two mirrors cancel.
                assert_eq!(spec.permute(6, false, i), i);
                assert_eq!(spec.permute(0, true, spec.permute(0, true, i)), i);
            }
            // A rule needing one neighbour matches rotated against all six.
            let mut n = vec![NeighborReq::Any; 6];
            n[0] = NeighborReq::This;
            let t = RuleTile { layout, rules: vec![TilingRule { neighbors: n, transform: RuleTransform::Rotated, ..Default::default() }], ..Default::default() };
            let c = t.compile();
            assert_eq!(c.table_len(), 64);
            for slot in 0..6 {
                let h = c.lookup(1 << slot);
                assert!(h.matched(), "{layout:?} slot {slot}");
                assert_eq!(h.xform, TileXform::new(slot as u8 * 4, false));
            }
        }
    }

    #[test]
    fn rect_mirror_y_is_x_mirror_plus_half_turn_in_the_ring() {
        let spec = ring_spec(CellLayout::Rectangle);
        for i in 0..8 {
            let o = neighbor_offset(CellLayout::Rectangle, IVec2::ZERO, i);
            let mirrored_y = IVec2::new(o.x, -o.y);
            let j = spec.permute(2, true, i);
            assert_eq!(neighbor_offset(CellLayout::Rectangle, IVec2::ZERO, j), mirrored_y, "slot {i}");
        }
    }

    #[test]
    fn refreshing_a_cell_touches_only_its_ring() {
        let r = refresh_cells(CellLayout::Rectangle, IVec2::new(-1, 0));
        assert_eq!(r.len(), 9);
        assert!(r.iter().all(|c| (c.x + 1).abs() <= 1 && c.y.abs() <= 1));
        let h = refresh_cells(CellLayout::HexPointy, IVec2::new(3, 1));
        assert_eq!(h.len(), 7);
        let mut seen = std::collections::HashSet::new();
        assert!(h.iter().all(|c| seen.insert(*c)), "no duplicates");
    }

    #[test]
    fn selection_applies_outputs() {
        let mut t = wall_tile();
        t.rules[1].output = RuleOutput::Random;
        t.rules[1].sprites = (0..4).map(|i| format!("r{i}#0000000000000001")).collect();
        let c = t.compile();
        let h = c.lookup(1 << 6);
        // Random output: stable per cell, varies across cells.
        let a = t.select(h, IVec2::new(3, 4)).sprites.to_vec();
        assert_eq!(a, t.select(h, IVec2::new(3, 4)).sprites.to_vec());
        let picks: std::collections::HashSet<_> = (0..200).map(|x| t.select(h, IVec2::new(x * 3, 7)).sprites[0].clone()).collect();
        assert!(picks.len() > 1, "{picks:?}");
        // Default output when nothing matched.
        let none = t.select(c.lookup(0b0101_0101), IVec2::ZERO);
        assert_eq!(none.sprites, &[t.default_sprite.clone()]);
        // Animation output carries the speed.
        t.rules[0].output = RuleOutput::Animation;
        t.rules[0].sprites = vec!["a#0000000000000001".into(), "b#0000000000000001".into()];
        t.rules[0].min_speed = 2.0;
        t.rules[0].max_speed = 6.0;
        let s = t.select(c.lookup(0), IVec2::new(1, 1));
        assert_eq!(s.sprites.len(), 2);
        let sp = s.anim.unwrap().speed;
        assert!((2.0..=6.0).contains(&sp));
    }

    #[test]
    fn value_noise_is_smooth_and_bounded() {
        for i in 0..200 {
            let v = value_noise(i as f32 * 0.37, i as f32 * 0.11);
            assert!((0.0..=1.0).contains(&v));
        }
        let (a, b) = (value_noise(5.0, 5.0), value_noise(5.001, 5.0));
        assert!((a - b).abs() < 0.01);
    }

    #[test]
    fn assets_round_trip_and_old_files_load() {
        let t = TileAsset {
            sprite: "t.png#0000000000000002".into(),
            collider: TileColliderType::Grid,
            animation: Some(TileAnimation { sprites: vec!["a#0000000000000001".into()], min_speed: 2.0, ..Default::default() }),
            ..Default::default()
        };
        let text = ron::to_string(&t).unwrap();
        assert_eq!(ron::from_str::<TileAsset>(&text).unwrap(), t);
        let d: TileAsset = ron::from_str("(sprite: \"x#0000000000000001\")").unwrap();
        assert_eq!((d.collider, d.color), (TileColliderType::Sprite, [1.0; 4]));
        assert!(d.game_object.is_empty(), "a tile file from before GameObject tiles has no prefab");
        let g = TileAsset { game_object: "prefabs/crate.prefab".into(), ..Default::default() };
        assert_eq!(ron::from_str::<TileAsset>(&ron::to_string(&g).unwrap()).unwrap().game_object, "prefabs/crate.prefab");
        let r = wall_tile();
        assert_eq!(ron::from_str::<RuleTile>(&ron::to_string(&r).unwrap()).unwrap(), r);
        let e: RuleTile = ron::from_str("()").unwrap();
        assert!(e.rules.is_empty());
        let p = TilePalette::from_tiles(&["a.tile".into(), "b.tile".into(), "c.tile".into()], 2);
        assert_eq!(p.tile_at(0, 1), Some("a.tile"));
        assert_eq!(p.tile_at(0, 0), Some("c.tile"));
        assert_eq!(ron::from_str::<TilePalette>(&ron::to_string(&p).unwrap()).unwrap(), p);
        assert_eq!(p.bounds(), Some((IVec2::new(0, 0), IVec2::new(1, 1))));
        // Files on disk, by extension.
        let dir = std::env::temp_dir().join(format!("zest_tile_assets_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        TileFile::Tile(t.clone()).save(&dir.join("a.tile")).unwrap();
        TileFile::Rule(wall_tile()).save(&dir.join("w.ruletile")).unwrap();
        assert_eq!(TileFile::load(&dir.join("a.tile")).unwrap(), TileFile::Tile(t));
        assert_eq!(TileFile::load(&dir.join("w.ruletile")).unwrap(), TileFile::Rule(wall_tile()));
        save_palette(&dir.join("p.palette"), &p).unwrap();
        assert_eq!(load_palette(&dir.join("p.palette")).unwrap(), p);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
