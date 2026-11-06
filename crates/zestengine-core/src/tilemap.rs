//! Tilemap data model (2D-5): [`Grid`] (cell layouts and the cell <-> local
//! conversions), [`Tilemap`] (sparse 16 x 16 chunk storage with de-duplicated
//! tables), and the sibling components [`TilemapRenderer`] and
//! [`TilemapCollider2D`]. Plain serde data, egui-free; the tile assets live in
//! `tile_asset.rs`, the painting algorithms in `tile_paint.rs`, rendering in
//! the engine's `scene/tilemaps2d.rs`, colliders in `physics2d.rs`.
//!
//! # Storage
//!
//! A cell is addressed by an `IVec2` (x, y; layers are separate tilemaps). The
//! map is a hash map from chunk coordinates `(x.div_euclid(16), y.div_euclid(16))`
//! to a [`TileChunk`]: a dense array of 256 `u16` tile indices (0 = empty,
//! `n` = entry `n - 1` of the map's tile table) plus three optional side
//! arrays that exist only once a cell in the chunk needs them: a colour index,
//! a transform index and a flags byte. The colour and transform tables are
//! shared by the whole map and de-duplicated (entry 0 = white / identity, so a
//! default cell stores 0 and costs nothing). Cell index in a chunk =
//! `(x & 15) | (y & 15) << 4`, correct for negative coordinates.
//!
//! Every chunk has a `version` that any edit bumps, and an edit on a chunk's
//! border also bumps the adjacent chunks (rule tiles look at their neighbours),
//! so a consumer (render cache, collider builder) re-does exactly the chunks
//! whose version changed.
//!
//! # File format
//!
//! `Tilemap` serialises through [`TilemapData`]: the three tables (without
//! unreferenced entries) and per chunk the tile array run-length encoded as
//! `(run, value)` pairs, the optional arrays as sparse `(cell, value)` lists.
//! Every field has a serde default, so a scene saved before tilemaps existed
//! (no component) and a partial component both load.

use std::collections::HashMap;

use glam::{IVec2, IVec3, Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::TypedComponent;

/// Cells per chunk side.
pub const CHUNK_SIZE: i32 = 16;
const CHUNK_CELLS: usize = (CHUNK_SIZE * CHUNK_SIZE) as usize;

// ---------------------------------------------------------------------------
// Grid
// ---------------------------------------------------------------------------

/// How cells are laid out.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum CellLayout {
    /// Axis-aligned square cells.
    #[default]
    Rectangle,
    /// Hexagons with a vertex on top; odd rows shift half a cell right.
    HexPointy,
    /// Hexagons with a flat top; odd columns shift half a cell up.
    HexFlat,
    /// Diamond cells; `x = (cx - cy) W / 2`, `y = (cx + cy) H / 2`.
    Isometric,
    /// As Isometric, and the cell's Z layer lifts it by `z * cell_size.z` on screen.
    IsometricZAsY,
}

impl CellLayout {
    /// Every layout, in declaration order.
    pub const ALL: [CellLayout; 5] = [Self::Rectangle, Self::HexPointy, Self::HexFlat, Self::Isometric, Self::IsometricZAsY];
    /// Inspector label of this layout.
    pub fn label(self) -> &'static str {
        match self {
            Self::Rectangle => "Rectangle",
            Self::HexPointy => "Hexagon (pointy)",
            Self::HexFlat => "Hexagon (flat)",
            Self::Isometric => "Isometric",
            Self::IsometricZAsY => "Isometric (Z as Y)",
        }
    }
    /// True for the two hexagon layouts.
    pub fn is_hex(self) -> bool {
        matches!(self, Self::HexPointy | Self::HexFlat)
    }
    /// True for the two isometric layouts.
    pub fn is_iso(self) -> bool {
        matches!(self, Self::Isometric | Self::IsometricZAsY)
    }
    /// Neighbours a rule looks at: 6 on a hexagon grid, 8 otherwise.
    pub fn neighbor_count(self) -> usize {
        if self.is_hex() { 6 } else { 8 }
    }
}

/// How cell axes map onto local axes.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum CellSwizzle {
    /// Cell x / y / z map to local x / y / z.
    #[default]
    XYZ,
    /// Cell x / y / z map to local x / z / y.
    XZY,
    /// Cell x / y / z map to local y / x / z.
    YXZ,
    /// Cell x / y / z map to local y / z / x.
    YZX,
    /// Cell x / y / z map to local z / x / y.
    ZXY,
    /// Cell x / y / z map to local z / y / x.
    ZYX,
}

impl CellSwizzle {
    /// Every swizzle, in declaration order.
    pub const ALL: [CellSwizzle; 6] = [Self::XYZ, Self::XZY, Self::YXZ, Self::YZX, Self::ZXY, Self::ZYX];
    /// Inspector label of this swizzle.
    pub fn label(self) -> &'static str {
        match self {
            Self::XYZ => "XYZ",
            Self::XZY => "XZY",
            Self::YXZ => "YXZ",
            Self::YZX => "YZX",
            Self::ZXY => "ZXY",
            Self::ZYX => "ZYX",
        }
    }
    /// Cell-space vector -> local axes.
    pub fn apply(self, v: Vec3) -> Vec3 {
        match self {
            Self::XYZ => v,
            Self::XZY => Vec3::new(v.x, v.z, v.y),
            Self::YXZ => Vec3::new(v.y, v.x, v.z),
            Self::YZX => Vec3::new(v.y, v.z, v.x),
            Self::ZXY => Vec3::new(v.z, v.x, v.y),
            Self::ZYX => Vec3::new(v.z, v.y, v.x),
        }
    }
    /// Local axes -> cell-space vector.
    pub fn inverse(self, v: Vec3) -> Vec3 {
        match self {
            Self::XYZ => v,
            Self::XZY => Vec3::new(v.x, v.z, v.y),
            Self::YXZ => Vec3::new(v.y, v.x, v.z),
            Self::YZX => Vec3::new(v.z, v.x, v.y),
            Self::ZXY => Vec3::new(v.y, v.z, v.x),
            Self::ZYX => Vec3::new(v.z, v.y, v.x),
        }
    }
}

fn one3() -> [f32; 3] {
    [1.0; 3]
}

/// Cell layout of the tilemaps below it. The conversions are pure functions of
/// the fields, so editor, renderer and physics agree.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Grid {
    /// How cells are laid out.
    pub layout: CellLayout,
    /// Cell size (local units; z is the layer thickness).
    pub cell_size: [f32; 3],
    /// Extra gap between cells (local units).
    pub cell_gap: [f32; 3],
    /// How cell axes map onto local axes.
    pub swizzle: CellSwizzle,
}

impl Default for Grid {
    fn default() -> Self {
        Self { layout: CellLayout::Rectangle, cell_size: one3(), cell_gap: [0.0; 3], swizzle: CellSwizzle::XYZ }
    }
}

impl TypedComponent for Grid {
    const NAME: &'static str = "Grid";
}

/// Row stride of a pointy hexagon grid relative to the cell height.
pub const HEX_VERTICAL: f32 = 0.75;

impl Grid {
    fn size(&self) -> Vec3 {
        Vec3::from(self.cell_size)
    }
    fn stride(&self) -> Vec3 {
        Vec3::from(self.cell_size) + Vec3::from(self.cell_gap)
    }

    /// Local position (before the swizzle) of a possibly fractional cell
    /// coordinate: the cell's min corner for Rectangle, its bottom vertex for
    /// Isometric, its centre for hexagons. `HexPointy` / `HexFlat` use the
    /// fractional part linearly, the parity shift only from the integer part.
    fn unswizzled(&self, c: Vec3) -> Vec3 {
        let st = self.stride();
        match self.layout {
            CellLayout::Rectangle => c * st,
            CellLayout::Isometric => Vec3::new((c.x - c.y) * st.x * 0.5, (c.x + c.y) * st.y * 0.5, c.z * st.z),
            CellLayout::IsometricZAsY => Vec3::new((c.x - c.y) * st.x * 0.5, (c.x + c.y) * st.y * 0.5 + c.z * self.size().z, c.z * st.z),
            CellLayout::HexPointy => {
                let row = c.y.floor() as i32;
                let shift = if row.rem_euclid(2) == 1 { st.x * 0.5 } else { 0.0 };
                Vec3::new(c.x * st.x + shift, c.y * (self.size().y * HEX_VERTICAL + self.cell_gap[1]), c.z * st.z)
            }
            CellLayout::HexFlat => {
                let col = c.x.floor() as i32;
                let shift = if col.rem_euclid(2) == 1 { st.y * 0.5 } else { 0.0 };
                Vec3::new(c.x * (self.size().x * HEX_VERTICAL + self.cell_gap[0]), c.y * st.y + shift, c.z * st.z)
            }
        }
    }

    /// Local position of the cell's origin (the swizzle applied). For
    /// hexagons that is the cell centre, otherwise the cell's min corner.
    pub fn cell_to_local(&self, cell: IVec3) -> Vec3 {
        self.swizzle.apply(self.unswizzled(cell.as_vec3()))
    }

    /// Local position of the cell's centre.
    pub fn cell_center(&self, cell: IVec3) -> Vec3 {
        if self.layout.is_hex() {
            return self.cell_to_local(cell);
        }
        self.swizzle.apply(self.unswizzled(cell.as_vec3() + Vec3::new(0.5, 0.5, 0.0)))
    }

    /// The cell containing a local position (z gives the layer for the
    /// layouts that have one; the plane position decides x and y).
    pub fn local_to_cell(&self, local: Vec3) -> IVec3 {
        let p = self.swizzle.inverse(local);
        let st = self.stride();
        let z = if st.z.abs() > 1e-9 { (p.z / st.z).floor() as i32 } else { 0 };
        match self.layout {
            CellLayout::Rectangle => IVec3::new((p.x / st.x).floor() as i32, (p.y / st.y).floor() as i32, z),
            CellLayout::Isometric | CellLayout::IsometricZAsY => {
                let y = p.y;
                let a = p.x / (st.x * 0.5);
                let b = y / (st.y * 0.5);
                IVec3::new(((a + b) * 0.5).floor() as i32, ((b - a) * 0.5).floor() as i32, z)
            }
            CellLayout::HexPointy | CellLayout::HexFlat => {
                // The nearest cell centre among the candidates around the
                // grid-aligned estimate is the containing hexagon.
                let (row_stride, col_stride) = (self.size().y * HEX_VERTICAL + self.cell_gap[1], self.size().x * HEX_VERTICAL + self.cell_gap[0]);
                let guess = if self.layout == CellLayout::HexPointy {
                    IVec2::new((p.x / st.x).round() as i32, (p.y / row_stride).round() as i32)
                } else {
                    IVec2::new((p.x / col_stride).round() as i32, (p.y / st.y).round() as i32)
                };
                let mut best = (f32::MAX, guess);
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let c = guess + IVec2::new(dx, dy);
                        let o = self.unswizzled(IVec3::new(c.x, c.y, 0).as_vec3());
                        let d = (o.x - p.x).powi(2) + (o.y - p.y).powi(2);
                        if d < best.0 {
                            best = (d, c);
                        }
                    }
                }
                IVec3::new(best.1.x, best.1.y, z)
            }
        }
    }

    /// Local position where a tile with `anchor` (0.5, 0.5 = centred) puts its
    /// sprite pivot: the cell centre moved by `(anchor - 0.5)` cell sizes.
    pub fn tile_pivot(&self, cell: IVec3, anchor: [f32; 2]) -> Vec3 {
        let sz = self.size();
        let off = self.swizzle.apply(Vec3::new((anchor[0] - 0.5) * sz.x, (anchor[1] - 0.5) * sz.y, 0.0));
        self.cell_center(cell) + off
    }

    /// The (up to six) corner points of a cell in local space, counter
    /// clockwise: 4 for rectangle / isometric, 6 for hexagons.
    pub fn cell_corners(&self, cell: IVec3) -> Vec<Vec3> {
        let c = self.cell_center(cell);
        let s = self.size();
        let pts: Vec<Vec3> = match self.layout {
            CellLayout::Rectangle => [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)].iter().map(|&(x, y)| Vec3::new(x * s.x, y * s.y, 0.0)).collect(),
            CellLayout::Isometric | CellLayout::IsometricZAsY => [(0.0, -0.5), (0.5, 0.0), (0.0, 0.5), (-0.5, 0.0)].iter().map(|&(x, y)| Vec3::new(x * s.x, y * s.y, 0.0)).collect(),
            CellLayout::HexPointy => [(0.0, -0.5), (0.5, -0.25), (0.5, 0.25), (0.0, 0.5), (-0.5, 0.25), (-0.5, -0.25)].iter().map(|&(x, y)| Vec3::new(x * s.x, y * s.y, 0.0)).collect(),
            CellLayout::HexFlat => [(-0.25, -0.5), (0.25, -0.5), (0.5, 0.0), (0.25, 0.5), (-0.25, 0.5), (-0.5, 0.0)].iter().map(|&(x, y)| Vec3::new(x * s.x, y * s.y, 0.0)).collect(),
        };
        pts.into_iter().map(|p| c + self.swizzle.apply(p)).collect()
    }
}

// ---------------------------------------------------------------------------
// Neighbours
// ---------------------------------------------------------------------------

/// Rectangle / isometric ring (counter clockwise from east).
const RING_RECT: [(i32, i32); 8] = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];

/// Offset of ring slot `i` for the cell `at` (hexagon parity matters).
/// Hexagon rings: pointy `E NE NW W SW SE`, flat `NE N NW SW S SE`.
pub fn neighbor_offset(layout: CellLayout, at: IVec2, i: usize) -> IVec2 {
    match layout {
        CellLayout::Rectangle | CellLayout::Isometric | CellLayout::IsometricZAsY => {
            let (x, y) = RING_RECT[i % 8];
            IVec2::new(x, y)
        }
        CellLayout::HexPointy => {
            let odd = at.y.rem_euclid(2) == 1;
            let (x, y) = match (i % 6, odd) {
                (0, _) => (1, 0),
                (1, false) => (0, 1),
                (1, true) => (1, 1),
                (2, false) => (-1, 1),
                (2, true) => (0, 1),
                (3, _) => (-1, 0),
                (4, false) => (-1, -1),
                (4, true) => (0, -1),
                (_, false) => (0, -1),
                (_, true) => (1, -1),
            };
            IVec2::new(x, y)
        }
        CellLayout::HexFlat => {
            let odd = at.x.rem_euclid(2) == 1;
            let (x, y) = match (i % 6, odd) {
                (0, false) => (1, 0),
                (0, true) => (1, 1),
                (1, _) => (0, 1),
                (2, false) => (-1, 0),
                (2, true) => (-1, 1),
                (3, false) => (-1, -1),
                (3, true) => (-1, 0),
                (4, _) => (0, -1),
                (_, false) => (1, -1),
                (_, true) => (1, 0),
            };
            IVec2::new(x, y)
        }
    }
}

// ---------------------------------------------------------------------------
// Per-cell transform
// ---------------------------------------------------------------------------

/// Steps of [`TileXform::rot`] in a full turn (15 degrees each: covers the 90
/// degree steps of rectangle / isometric rules and the 60 degree hexagon ones).
pub const ROT_STEPS: u8 = 24;

/// A tile's orientation: mirror along x first, then rotate counter clockwise
/// by `rot * 15` degrees.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TileXform {
    /// Counter-clockwise rotation in 15-degree steps (0..23).
    pub rot: u8,
    /// Mirror along x before rotating.
    pub mirror: bool,
}

impl TileXform {
    /// No rotation, no mirror.
    pub const IDENTITY: TileXform = TileXform { rot: 0, mirror: false };

    /// True when the transform is the identity.
    pub fn is_identity(self) -> bool {
        self == Self::IDENTITY
    }
    /// Build a transform, wrapping `rot` into 0..23.
    pub fn new(rot: u8, mirror: bool) -> Self {
        Self { rot: rot % ROT_STEPS, mirror }
    }
    /// Rotation in radians (positive = counter-clockwise).
    pub fn angle(self) -> f32 {
        self.rot as f32 * (std::f32::consts::TAU / ROT_STEPS as f32)
    }
    /// The 2x2 matrix `R * F` as `[a, b, c, d]` (columns `(a, b)` and `(c, d)`).
    pub fn matrix(self) -> [f32; 4] {
        let (s, c) = self.angle().sin_cos();
        let f = if self.mirror { -1.0 } else { 1.0 };
        [c * f, s * f, -s, c]
    }
    /// Apply to a local point.
    pub fn apply(self, p: Vec2) -> Vec2 {
        let m = self.matrix();
        Vec2::new(m[0] * p.x + m[2] * p.y, m[1] * p.x + m[3] * p.y)
    }
    /// `self` applied after `first`.
    pub fn after(self, first: TileXform) -> TileXform {
        // R2 F2 R1 F1 = R2 R1^(+-1) F2 F1, a mirror flips the sign of R1.
        let r1 = if self.mirror { (ROT_STEPS - first.rot % ROT_STEPS) % ROT_STEPS } else { first.rot % ROT_STEPS };
        TileXform::new(self.rot + r1, self.mirror ^ first.mirror)
    }
    /// Rotated counter-clockwise by `steps * 15` degrees.
    pub fn rotated_ccw(self, steps: u8) -> TileXform {
        TileXform::new(steps, false).after(self)
    }
    /// Rotated clockwise by `steps * 15` degrees.
    pub fn rotated_cw(self, steps: u8) -> TileXform {
        self.rotated_ccw(ROT_STEPS - steps % ROT_STEPS)
    }
    /// Mirrored left-right in the cell (x -> -x after the existing transform).
    pub fn flipped_x(self) -> TileXform {
        // x -> -x applied last is exactly the mirror F = diag(-1, 1).
        TileXform::new(0, true).after(self)
    }
    /// Mirrored top-bottom in the cell.
    pub fn flipped_y(self) -> TileXform {
        // y -> -y = diag(1, -1) = R(180) * F.
        TileXform::new(ROT_STEPS / 2, true).after(self)
    }
    #[allow(dead_code)] // packed tile transforms for an upcoming path
    fn bits(self) -> u16 {
        self.rot as u16 | (self.mirror as u16) << 8
    }
    #[allow(dead_code)] // packed tile transforms for an upcoming path
    fn from_bits(b: u16) -> Self {
        Self { rot: (b & 0xFF) as u8 % ROT_STEPS, mirror: b & 0x100 != 0 }
    }
}

/// Per-cell flag: the tile's own colour wins over the cell's colour override.
pub const FLAG_LOCK_COLOR: u8 = 1;
/// Per-cell flag: the tile's own transform wins over the cell's transform override.
pub const FLAG_LOCK_TRANSFORM: u8 = 2;

// ---------------------------------------------------------------------------
// Tilemap
// ---------------------------------------------------------------------------

/// A tile asset in a tilemap's table.
#[derive(Clone, PartialEq, Eq, Hash, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TileRef {
    /// Project-relative path of the `.tile` / `.ruletile`.
    pub tile: String,
}

/// Everything a cell holds, by value (undo diffs, copy and paste).
#[derive(Clone, PartialEq, Debug)]
pub struct CellState {
    /// Empty = no tile.
    pub tile: String,
    /// Per-cell colour override (255 white = none).
    pub color: [u8; 4],
    /// Per-cell transform override.
    pub xform: TileXform,
    /// Per-cell flags (lock colour / lock transform).
    pub flags: u8,
}

impl Default for CellState {
    fn default() -> Self {
        Self { tile: String::new(), color: [255; 4], xform: TileXform::IDENTITY, flags: 0 }
    }
}

impl CellState {
    /// A cell holding the tile at this project-relative path.
    pub fn tile(path: &str) -> Self {
        Self { tile: path.to_string(), ..Default::default() }
    }
    /// True when the cell holds no tile.
    pub fn is_empty(&self) -> bool {
        self.tile.is_empty()
    }
}

/// One 16 x 16 block of cells.
#[derive(Clone, Debug)]
pub struct TileChunk {
    /// 0 = empty, `n` = tile table entry `n - 1`.
    pub tiles: [u16; CHUNK_CELLS],
    /// Colour-table index per cell, present once a cell overrides.
    pub color: Option<Box<[u16; CHUNK_CELLS]>>,
    /// Transform-table index per cell, present once a cell overrides.
    pub xform: Option<Box<[u16; CHUNK_CELLS]>>,
    /// Flag byte per cell, present once a cell sets one.
    pub flags: Option<Box<[u8; CHUNK_CELLS]>>,
    /// Non-empty cells.
    pub count: u16,
    /// Bumped by every edit of this chunk or of a cell next to it.
    pub version: u32,
}

impl TileChunk {
    fn new(version: u32) -> Self {
        Self { tiles: [0; CHUNK_CELLS], color: None, xform: None, flags: None, count: 0, version }
    }
    /// True when the chunk holds no tiles.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Chunk coordinates of a cell.
pub fn chunk_of(cell: IVec2) -> IVec2 {
    IVec2::new(cell.x.div_euclid(CHUNK_SIZE), cell.y.div_euclid(CHUNK_SIZE))
}

/// Index of a cell inside its chunk.
pub fn cell_index(cell: IVec2) -> usize {
    (cell.x.rem_euclid(CHUNK_SIZE) | cell.y.rem_euclid(CHUNK_SIZE) << 4) as usize
}

/// The cell at `index` of chunk `chunk`.
pub fn cell_at(chunk: IVec2, index: usize) -> IVec2 {
    IVec2::new(chunk.x * CHUNK_SIZE + (index & 15) as i32, chunk.y * CHUNK_SIZE + (index >> 4) as i32)
}

/// Which plane the map's cell (x, y) lies in.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum TilemapOrientation {
    /// Cell (x, y) at local (x, y): the 2D sprite plane.
    #[default]
    XY,
    /// Cell (x, y) at local (x, 0, y): a ground plane. Stored and edited like
    /// any map; the 2D sprite pass draws XY maps only.
    XZ,
}

impl TilemapOrientation {
    /// Every orientation, in declaration order.
    pub const ALL: [TilemapOrientation; 2] = [Self::XY, Self::XZ];
    /// Inspector label of this orientation.
    pub fn label(self) -> &'static str {
        match self {
            Self::XY => "XY",
            Self::XZ => "XZ",
        }
    }
}

fn half2() -> [f32; 2] {
    [0.5, 0.5]
}
fn white4() -> [f32; 4] {
    [1.0; 4]
}
fn one() -> f32 {
    1.0
}

/// A sparse map of tiles. See the module docs for the storage design.
#[derive(Debug, Deserialize)]
#[serde(from = "TilemapData")]
pub struct Tilemap {
    /// Which plane the map's cells lie in.
    pub orientation: TilemapOrientation,
    /// Where a tile's sprite pivot sits in its cell (0.5, 0.5 = centre).
    pub anchor: [f32; 2],
    /// Tint of the whole map, linear RGBA.
    pub color: [f32; 4],
    /// Tile animation frames per second at speed 1.
    pub animation_frame_rate: f32,
    /// Seconds added to every animated tile's clock (a start offset for the map).
    pub animation_time_offset: f32,
    tiles: Vec<TileRef>,
    tile_ix: HashMap<String, u16>,
    colors: Vec<[u8; 4]>,
    color_ix: HashMap<[u8; 4], u16>,
    xforms: Vec<TileXform>,
    xform_ix: HashMap<TileXform, u16>,
    chunks: HashMap<(i32, i32), Box<TileChunk>>,
    /// Inclusive cell bounds of everything ever written (min, max).
    bounds: Option<(IVec2, IVec2)>,
    /// Bumped by every edit (cheap "anything changed" check).
    serial: u64,
    next_version: u32,
    /// The file form for `serial`, so serialising an unchanged map (the editor
    /// diffs the selected object's components every frame) is a lookup.
    save_cache: std::sync::Mutex<Option<(u64, std::sync::Arc<TilemapData>)>>,
}

impl Clone for Tilemap {
    fn clone(&self) -> Self {
        Self {
            orientation: self.orientation,
            anchor: self.anchor,
            color: self.color,
            animation_frame_rate: self.animation_frame_rate,
            animation_time_offset: self.animation_time_offset,
            tiles: self.tiles.clone(),
            tile_ix: self.tile_ix.clone(),
            colors: self.colors.clone(),
            color_ix: self.color_ix.clone(),
            xforms: self.xforms.clone(),
            xform_ix: self.xform_ix.clone(),
            chunks: self.chunks.clone(),
            bounds: self.bounds,
            serial: self.serial,
            next_version: self.next_version,
            save_cache: std::sync::Mutex::new(None),
        }
    }
}

impl Serialize for Tilemap {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.file_data().serialize(s)
    }
}

impl PartialEq for Tilemap {
    fn eq(&self, other: &Self) -> bool {
        *self.file_data() == *other.file_data()
    }
}

impl Default for Tilemap {
    fn default() -> Self {
        let mut t = Self {
            orientation: TilemapOrientation::XY,
            anchor: half2(),
            color: white4(),
            animation_frame_rate: one(),
            animation_time_offset: 0.0,
            tiles: Vec::new(),
            tile_ix: HashMap::new(),
            colors: vec![[255; 4]],
            color_ix: HashMap::new(),
            xforms: vec![TileXform::IDENTITY],
            xform_ix: HashMap::new(),
            chunks: HashMap::new(),
            bounds: None,
            serial: 0,
            next_version: 1,
            save_cache: std::sync::Mutex::new(None),
        };
        t.color_ix.insert([255; 4], 0);
        t.xform_ix.insert(TileXform::IDENTITY, 0);
        t
    }
}

impl TypedComponent for Tilemap {
    const NAME: &'static str = "Tilemap";
}

impl Tilemap {
    /// The edit serial (bumped by every edit).
    pub fn serial(&self) -> u64 {
        self.serial
    }
    /// Non-empty cell count.
    pub fn cell_count(&self) -> usize {
        self.chunks.values().map(|c| c.count as usize).sum()
    }
    /// Number of chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
    /// True when the map holds no tiles.
    pub fn is_empty(&self) -> bool {
        self.cell_count() == 0
    }
    /// The tile table (entry `n - 1` = chunk index `n`).
    pub fn tile_table(&self) -> &[TileRef] {
        &self.tiles
    }
    /// Colour table length (entry 0 = white).
    pub fn color_table_len(&self) -> usize {
        self.colors.len()
    }
    /// Transform table length (entry 0 = identity).
    pub fn xform_table_len(&self) -> usize {
        self.xforms.len()
    }
    /// The tile asset of table entry `index` (1-based, as stored in chunks).
    pub fn tile_path(&self, index: u16) -> Option<&str> {
        self.tiles.get((index as usize).checked_sub(1)?).map(|t| t.tile.as_str())
    }
    /// The chunk at `c`, if present.
    pub fn chunk(&self, c: IVec2) -> Option<&TileChunk> {
        self.chunks.get(&(c.x, c.y)).map(|b| &**b)
    }
    /// Chunk coordinates in a deterministic order (row major, y then x).
    pub fn chunk_keys_sorted(&self) -> Vec<IVec2> {
        let mut k: Vec<IVec2> = self.chunks.keys().map(|&(x, y)| IVec2::new(x, y)).collect();
        k.sort_by_key(|c| (c.y, c.x));
        k
    }
    /// Chunk coordinates (any order).
    pub fn chunk_keys(&self) -> impl Iterator<Item = IVec2> + '_ {
        self.chunks.keys().map(|&(x, y)| IVec2::new(x, y))
    }
    /// Inclusive bounds of the cells written so far (`None` for a map that
    /// never held a tile). Grows on write; [`Self::compress_bounds`] shrinks it.
    pub fn bounds(&self) -> Option<(IVec2, IVec2)> {
        self.bounds
    }

    /// Recompute the bounds from the cells that are set.
    pub fn compress_bounds(&mut self) {
        let mut b: Option<(IVec2, IVec2)> = None;
        for (&(cx, cy), ch) in &self.chunks {
            for i in 0..CHUNK_CELLS {
                if ch.tiles[i] != 0 {
                    let c = cell_at(IVec2::new(cx, cy), i);
                    b = Some(match b {
                        Some((lo, hi)) => (lo.min(c), hi.max(c)),
                        None => (c, c),
                    });
                }
            }
        }
        self.bounds = b;
    }

    /// The tile table index (1-based) at a cell; 0 = empty.
    pub fn tile_index(&self, cell: IVec2) -> u16 {
        match self.chunks.get(&(chunk_of(cell).x, chunk_of(cell).y)) {
            Some(ch) => ch.tiles[cell_index(cell)],
            None => 0,
        }
    }

    /// The tile asset path at a cell.
    pub fn tile_at(&self, cell: IVec2) -> Option<&str> {
        self.tile_path(self.tile_index(cell))
    }

    /// True when the cell holds a tile.
    pub fn has_tile(&self, cell: IVec2) -> bool {
        self.tile_index(cell) != 0
    }

    /// The cell's effective colour (override or white).
    pub fn color_at(&self, cell: IVec2) -> [u8; 4] {
        let i = self.chunk(chunk_of(cell)).and_then(|c| c.color.as_ref()).map_or(0, |a| a[cell_index(cell)]);
        self.colors.get(i as usize).copied().unwrap_or([255; 4])
    }

    /// The cell's effective transform (override or identity).
    pub fn xform_at(&self, cell: IVec2) -> TileXform {
        let i = self.chunk(chunk_of(cell)).and_then(|c| c.xform.as_ref()).map_or(0, |a| a[cell_index(cell)]);
        self.xforms.get(i as usize).copied().unwrap_or_default()
    }

    /// The cell's flag byte (0 when unset).
    pub fn flags_at(&self, cell: IVec2) -> u8 {
        self.chunk(chunk_of(cell)).and_then(|c| c.flags.as_ref()).map_or(0, |a| a[cell_index(cell)])
    }

    /// Everything at a cell, by value.
    pub fn cell_state(&self, cell: IVec2) -> CellState {
        match self.tile_at(cell) {
            Some(t) => CellState { tile: t.to_string(), color: self.color_at(cell), xform: self.xform_at(cell), flags: self.flags_at(cell) },
            None => CellState::default(),
        }
    }

    fn intern_tile(&mut self, path: &str) -> u16 {
        if let Some(&i) = self.tile_ix.get(path) {
            return i;
        }
        self.tiles.push(TileRef { tile: path.to_string() });
        let i = self.tiles.len() as u16;
        self.tile_ix.insert(path.to_string(), i);
        i
    }

    fn intern_color(&mut self, c: [u8; 4]) -> u16 {
        if let Some(&i) = self.color_ix.get(&c) {
            return i;
        }
        self.colors.push(c);
        let i = (self.colors.len() - 1) as u16;
        self.color_ix.insert(c, i);
        i
    }

    fn intern_xform(&mut self, x: TileXform) -> u16 {
        if let Some(&i) = self.xform_ix.get(&x) {
            return i;
        }
        self.xforms.push(x);
        let i = (self.xforms.len() - 1) as u16;
        self.xform_ix.insert(x, i);
        i
    }

    fn fresh_version(&mut self) -> u32 {
        self.next_version = self.next_version.wrapping_add(1).max(1);
        self.next_version
    }

    /// Bump the chunk of `cell` and, for a cell on the chunk border, its
    /// neighbouring chunks (rule tiles read across the border).
    fn touch(&mut self, cell: IVec2) {
        self.serial = self.serial.wrapping_add(1);
        let c = chunk_of(cell);
        let (lx, ly) = (cell.x.rem_euclid(CHUNK_SIZE), cell.y.rem_euclid(CHUNK_SIZE));
        let dx = [if lx == 0 { -1 } else { 0 }, 0, if lx == CHUNK_SIZE - 1 { 1 } else { 0 }];
        let dy = [if ly == 0 { -1 } else { 0 }, 0, if ly == CHUNK_SIZE - 1 { 1 } else { 0 }];
        for &y in &dy {
            for &x in &dx {
                let k = (c.x + x, c.y + y);
                if (x, y) == (0, 0) || self.chunks.contains_key(&k) {
                    let v = self.fresh_version();
                    if let Some(ch) = self.chunks.get_mut(&k) {
                        ch.version = v;
                    }
                }
            }
        }
    }

    fn grow_bounds(&mut self, cell: IVec2) {
        self.bounds = Some(match self.bounds {
            Some((lo, hi)) => (lo.min(cell), hi.max(cell)),
            None => (cell, cell),
        });
    }

    /// Set (`Some(path)`) or clear (`None`) the tile at a cell. A new tile
    /// keeps nothing from the previous cell: colour, transform and flags reset.
    /// Returns whether anything changed.
    pub fn set_tile(&mut self, cell: IVec2, tile: Option<&str>) -> bool {
        let idx = match tile {
            Some(p) if !p.is_empty() => self.intern_tile(p),
            _ => 0,
        };
        let key = (chunk_of(cell).x, chunk_of(cell).y);
        let i = cell_index(cell);
        let old = self.chunks.get(&key).map_or(0, |c| c.tiles[i]);
        if old == idx {
            return false;
        }
        if idx == 0 {
            let ch = self.chunks.get_mut(&key).expect("old tile implies a chunk");
            ch.tiles[i] = 0;
            ch.count -= 1;
            if let Some(a) = ch.color.as_mut() {
                a[i] = 0;
            }
            if let Some(a) = ch.xform.as_mut() {
                a[i] = 0;
            }
            if let Some(a) = ch.flags.as_mut() {
                a[i] = 0;
            }
        } else {
            let v = self.fresh_version();
            let ch = self.chunks.entry(key).or_insert_with(|| Box::new(TileChunk::new(v)));
            if old == 0 {
                ch.count += 1;
            }
            ch.tiles[i] = idx;
            if old != 0 {
                // A replaced tile starts clean.
                if let Some(a) = ch.color.as_mut() {
                    a[i] = 0;
                }
                if let Some(a) = ch.xform.as_mut() {
                    a[i] = 0;
                }
                if let Some(a) = ch.flags.as_mut() {
                    a[i] = 0;
                }
            }
            self.grow_bounds(cell);
        }
        self.touch(cell);
        if self.chunks.get(&key).is_some_and(|c| c.count == 0) {
            // Keep the chunk until the next touch pass so its border neighbours
            // saw the version bump, then drop the empty block.
            self.chunks.remove(&key);
        }
        true
    }

    /// Per-cell colour override (255 white = none). Needs a tile at the cell.
    pub fn set_color(&mut self, cell: IVec2, color: [u8; 4]) -> bool {
        if !self.has_tile(cell) || self.color_at(cell) == color {
            return false;
        }
        let ci = self.intern_color(color);
        let key = (chunk_of(cell).x, chunk_of(cell).y);
        let ch = self.chunks.get_mut(&key).expect("tile implies chunk");
        ch.color.get_or_insert_with(|| Box::new([0; CHUNK_CELLS]))[cell_index(cell)] = ci;
        self.touch(cell);
        true
    }

    /// Set the cell's transform override; returns whether it changed.
    pub fn set_xform(&mut self, cell: IVec2, x: TileXform) -> bool {
        if !self.has_tile(cell) || self.xform_at(cell) == x {
            return false;
        }
        let xi = self.intern_xform(x);
        let key = (chunk_of(cell).x, chunk_of(cell).y);
        let ch = self.chunks.get_mut(&key).expect("tile implies chunk");
        ch.xform.get_or_insert_with(|| Box::new([0; CHUNK_CELLS]))[cell_index(cell)] = xi;
        self.touch(cell);
        true
    }

    /// Set the cell's flag byte; returns whether it changed.
    pub fn set_flags(&mut self, cell: IVec2, flags: u8) -> bool {
        if !self.has_tile(cell) || self.flags_at(cell) == flags {
            return false;
        }
        let key = (chunk_of(cell).x, chunk_of(cell).y);
        let ch = self.chunks.get_mut(&key).expect("tile implies chunk");
        ch.flags.get_or_insert_with(|| Box::new([0; CHUNK_CELLS]))[cell_index(cell)] = flags;
        self.touch(cell);
        true
    }

    /// Replace a whole cell. Returns whether anything changed.
    pub fn set_cell(&mut self, cell: IVec2, s: &CellState) -> bool {
        if self.cell_state(cell) == *s {
            return false;
        }
        if s.is_empty() {
            return self.set_tile(cell, None);
        }
        self.set_tile(cell, Some(&s.tile));
        self.set_color(cell, s.color);
        self.set_xform(cell, s.xform);
        self.set_flags(cell, s.flags);
        true
    }

    /// Remove every tile.
    pub fn clear(&mut self) {
        let keys: Vec<(i32, i32)> = self.chunks.keys().copied().collect();
        self.chunks.clear();
        self.bounds = None;
        self.serial = self.serial.wrapping_add(1);
        let _ = keys;
    }

    /// Every set cell in chunk-major, row-major order.
    pub fn for_each_cell(&self, mut f: impl FnMut(IVec2, u16)) {
        for k in self.chunk_keys_sorted() {
            let ch = &self.chunks[&(k.x, k.y)];
            for i in 0..CHUNK_CELLS {
                if ch.tiles[i] != 0 {
                    f(cell_at(k, i), ch.tiles[i]);
                }
            }
        }
    }

    /// Drop table entries no cell uses (indices change; saving does this anyway).
    pub fn compact(&mut self) {
        let data = self.build_data();
        let (serial, version) = (self.serial, self.next_version);
        *self = Tilemap::from(data);
        self.serial = serial.wrapping_add(1);
        self.next_version = version.wrapping_add(1);
    }

    /// The file form (compacted), cached per edit serial.
    pub fn file_data(&self) -> std::sync::Arc<TilemapData> {
        let mut g = self.save_cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((s, d)) = g.as_ref()
            && *s == self.serial
        {
            return d.clone();
        }
        let d = std::sync::Arc::new(self.build_data());
        *g = Some((self.serial, d.clone()));
        d
    }
}

// ---------------------------------------------------------------------------
// Serialisation
// ---------------------------------------------------------------------------

/// One chunk in the file.
#[derive(Clone, PartialEq, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ChunkData {
    /// Chunk coordinates.
    pub pos: (i32, i32),
    /// Run-length encoded tile array: `(run, value)` pairs covering 256 cells.
    pub tiles: Vec<(u16, u16)>,
    /// Sparse overrides: `(cell index, table index)`.
    pub colors: Vec<(u8, u16)>,
    /// Sparse transform overrides: (cell index, table index).
    pub xforms: Vec<(u8, u16)>,
    /// Sparse flag overrides: (cell index, flags).
    pub flags: Vec<(u8, u8)>,
}

/// The file form of a [`Tilemap`].
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TilemapData {
    /// Which plane the map's cells lie in.
    pub orientation: TilemapOrientation,
    /// Where a tile's sprite pivot sits in its cell (0.5, 0.5 = centre).
    pub anchor: [f32; 2],
    /// Tint of the whole map, linear RGBA.
    pub color: [f32; 4],
    /// Tile animation frames per second at speed 1.
    pub animation_frame_rate: f32,
    /// Seconds added to every animated tile's clock.
    pub animation_time_offset: f32,
    /// Tile asset paths; index `n` = chunk table entry `n + 1`.
    pub tiles: Vec<String>,
    /// Colour table without the implicit white entry 0.
    pub colors: Vec<[u8; 4]>,
    /// Transform table without the implicit identity entry 0.
    pub xforms: Vec<(u8, bool)>,
    /// The map's chunks.
    pub chunks: Vec<ChunkData>,
}

impl Default for TilemapData {
    fn default() -> Self {
        Self {
            orientation: TilemapOrientation::XY,
            anchor: half2(),
            color: white4(),
            animation_frame_rate: one(),
            animation_time_offset: 0.0,
            tiles: Vec::new(),
            colors: Vec::new(),
            xforms: Vec::new(),
            chunks: Vec::new(),
        }
    }
}

fn rle_encode(tiles: &[u16; CHUNK_CELLS]) -> Vec<(u16, u16)> {
    let mut out: Vec<(u16, u16)> = Vec::new();
    for &t in tiles {
        match out.last_mut() {
            Some((run, v)) if *v == t && *run < u16::MAX => *run += 1,
            _ => out.push((1, t)),
        }
    }
    out
}

impl From<Tilemap> for TilemapData {
    fn from(t: Tilemap) -> Self {
        t.build_data()
    }
}

impl Tilemap {
    fn build_data(&self) -> TilemapData {
        let t = self;
        // Remap to the entries actually used.
        let mut tile_map: HashMap<u16, u16> = HashMap::new();
        let mut tiles: Vec<String> = Vec::new();
        let mut color_map: HashMap<u16, u16> = HashMap::new();
        let mut colors: Vec<[u8; 4]> = Vec::new();
        let mut xform_map: HashMap<u16, u16> = HashMap::new();
        let mut xforms: Vec<(u8, bool)> = Vec::new();
        let mut chunks = Vec::new();
        let mut keys: Vec<(i32, i32)> = t.chunks.keys().copied().collect();
        keys.sort_by_key(|&(x, y)| (y, x));
        for k in keys {
            let ch = &t.chunks[&k];
            if ch.count == 0 {
                continue;
            }
            let mut cd = ChunkData { pos: k, ..Default::default() };
            let mut dense = [0u16; CHUNK_CELLS];
            for i in 0..CHUNK_CELLS {
                let old = ch.tiles[i];
                if old == 0 {
                    continue;
                }
                dense[i] = *tile_map.entry(old).or_insert_with(|| {
                    tiles.push(t.tiles[old as usize - 1].tile.clone());
                    tiles.len() as u16
                });
                if let Some(a) = &ch.color
                    && a[i] != 0
                {
                    let n = *color_map.entry(a[i]).or_insert_with(|| {
                        colors.push(t.colors[a[i] as usize]);
                        colors.len() as u16
                    });
                    cd.colors.push((i as u8, n));
                }
                if let Some(a) = &ch.xform
                    && a[i] != 0
                {
                    let n = *xform_map.entry(a[i]).or_insert_with(|| {
                        let x = t.xforms[a[i] as usize];
                        xforms.push((x.rot, x.mirror));
                        xforms.len() as u16
                    });
                    cd.xforms.push((i as u8, n));
                }
                if let Some(a) = &ch.flags
                    && a[i] != 0
                {
                    cd.flags.push((i as u8, a[i]));
                }
            }
            cd.tiles = rle_encode(&dense);
            chunks.push(cd);
        }
        TilemapData {
            orientation: t.orientation,
            anchor: t.anchor,
            color: t.color,
            animation_frame_rate: t.animation_frame_rate,
            animation_time_offset: t.animation_time_offset,
            tiles,
            colors,
            xforms,
            chunks,
        }
    }
}

impl From<TilemapData> for Tilemap {
    fn from(d: TilemapData) -> Self {
        let mut t = Tilemap { orientation: d.orientation, anchor: d.anchor, color: d.color, animation_frame_rate: d.animation_frame_rate, animation_time_offset: d.animation_time_offset, ..Tilemap::default() };
        for p in &d.tiles {
            t.intern_tile(p);
        }
        for c in &d.colors {
            t.intern_color(*c);
        }
        for &(r, m) in &d.xforms {
            t.intern_xform(TileXform::new(r, m));
        }
        // The file's indices are 1-based into its tables; ours match as long
        // as interning kept the order (duplicates in a hand-edited file shift
        // them, so translate through the original positions).
        let tile_of: Vec<u16> = d.tiles.iter().map(|p| t.tile_ix[p.as_str()]).collect();
        let color_of: Vec<u16> = d.colors.iter().map(|c| t.color_ix[c]).collect();
        let xform_of: Vec<u16> = d.xforms.iter().map(|&(r, m)| t.xform_ix[&TileXform::new(r, m)]).collect();
        for cd in &d.chunks {
            let mut ch = TileChunk::new(t.next_version);
            let mut i = 0usize;
            for &(run, v) in &cd.tiles {
                for _ in 0..run {
                    if i < CHUNK_CELLS {
                        ch.tiles[i] = if v == 0 { 0 } else { tile_of.get(v as usize - 1).copied().unwrap_or(0) };
                        i += 1;
                    }
                }
            }
            ch.count = ch.tiles.iter().filter(|&&v| v != 0).count() as u16;
            if ch.count == 0 {
                continue;
            }
            if !cd.colors.is_empty() {
                let mut a = Box::new([0u16; CHUNK_CELLS]);
                for &(c, n) in &cd.colors {
                    a[c as usize] = color_of.get((n as usize).wrapping_sub(1)).copied().unwrap_or(0);
                }
                ch.color = Some(a);
            }
            if !cd.xforms.is_empty() {
                let mut a = Box::new([0u16; CHUNK_CELLS]);
                for &(c, n) in &cd.xforms {
                    a[c as usize] = xform_of.get((n as usize).wrapping_sub(1)).copied().unwrap_or(0);
                }
                ch.xform = Some(a);
            }
            if !cd.flags.is_empty() {
                let mut a = Box::new([0u8; CHUNK_CELLS]);
                for &(c, f) in &cd.flags {
                    a[c as usize] = f;
                }
                ch.flags = Some(a);
            }
            t.chunks.insert(cd.pos, Box::new(ch));
        }
        t.compress_bounds();
        t
    }
}

// ---------------------------------------------------------------------------
// TilemapRenderer
// ---------------------------------------------------------------------------

/// How a map's tiles are drawn.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum TilemapMode {
    /// Each chunk's tiles are one sorted unit: the whole map has a single sort
    /// key (layer and order), tiles keep their drawing order inside it.
    #[default]
    Chunk,
    /// Every tile sorts on its own against other sprites (depth along the
    /// sorting axis), for isometric props and Y-sorted worlds.
    Individual,
}

impl TilemapMode {
    /// Every mode, in declaration order.
    pub const ALL: [TilemapMode; 2] = [Self::Chunk, Self::Individual];
    /// Inspector label of this mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::Chunk => "Chunk",
            Self::Individual => "Individual",
        }
    }
}

/// Which corner tiles draw first.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum TileSortOrder {
    /// Tiles draw from the bottom-left corner first.
    #[default]
    BottomLeft,
    /// Tiles draw from the bottom-right corner first.
    BottomRight,
    /// Tiles draw from the top-left corner first.
    TopLeft,
    /// Tiles draw from the top-right corner first.
    TopRight,
}

impl TileSortOrder {
    /// Every order, in declaration order.
    pub const ALL: [TileSortOrder; 4] = [Self::BottomLeft, Self::BottomRight, Self::TopLeft, Self::TopRight];
    /// Inspector label of this order.
    pub fn label(self) -> &'static str {
        match self {
            Self::BottomLeft => "Bottom Left",
            Self::BottomRight => "Bottom Right",
            Self::TopLeft => "Top Left",
            Self::TopRight => "Top Right",
        }
    }
    /// `(rows ascending, columns ascending)`.
    pub fn directions(self) -> (bool, bool) {
        match self {
            Self::BottomLeft => (true, true),
            Self::BottomRight => (true, false),
            Self::TopLeft => (false, true),
            Self::TopRight => (false, false),
        }
    }
    /// Ordering key of a cell: smaller draws first.
    pub fn key(self, c: IVec2) -> (i32, i32) {
        let (ry, rx) = self.directions();
        (if ry { c.y } else { -c.y }, if rx { c.x } else { -c.x })
    }
}

/// How the per-chunk culling margin is chosen.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum CullingBounds {
    /// The chunk's real tile geometry (sprites overhanging their cell included).
    #[default]
    Auto,
    /// The real geometry plus `chunk_culling_margin` world units.
    Manual,
}

/// Draws the sibling [`Tilemap`] through the 2D sprite pass.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TilemapRenderer {
    /// Sorting layer the map draws on.
    pub sorting_layer: crate::SortingLayerId,
    /// Sort key inside the layer.
    pub order_in_layer: i16,
    /// How tiles sort.
    pub mode: TilemapMode,
    /// Which corner tiles draw first.
    pub sort_order: TileSortOrder,
    /// Material override (`.mat`); empty = the default sprite material. Not
    /// consumed yet (the sprite pass has one built-in material).
    pub material: String,
    /// How the map interacts with sprite masks.
    pub mask_interaction: crate::MaskInteraction2D,
    /// How the culling margin is chosen.
    pub detect_chunk_culling_bounds: CullingBounds,
    /// Extra culling margin in world units (Manual mode).
    pub chunk_culling_margin: f32,
    /// Built chunks kept for fast panning (the cache is trimmed to the visible
    /// chunks plus this many, oldest first).
    pub max_chunk_count: u32,
    /// Frames an unseen chunk stays cached before it may be dropped.
    pub max_frame_age: u32,
}

impl Default for TilemapRenderer {
    fn default() -> Self {
        Self {
            sorting_layer: crate::DEFAULT_SORTING_LAYER,
            order_in_layer: 0,
            mode: TilemapMode::Chunk,
            sort_order: TileSortOrder::BottomLeft,
            material: String::new(),
            mask_interaction: crate::MaskInteraction2D::None,
            detect_chunk_culling_bounds: CullingBounds::Auto,
            chunk_culling_margin: 0.0,
            max_chunk_count: 32,
            max_frame_age: 64,
        }
    }
}

impl TypedComponent for TilemapRenderer {
    const NAME: &'static str = "Tilemap Renderer";
    const NET_ROLE: crate::NetRole = crate::NetRole::Client;
}

// ---------------------------------------------------------------------------
// TilemapCollider2D
// ---------------------------------------------------------------------------

fn tile_change_limit() -> u32 {
    1000
}

/// Physics for the sibling [`Tilemap`]: one compound collider per 16 x 16
/// chunk on the object's static body, from each tile's collider type (`Sprite`
/// = the sprite's physics outline, `Grid` = the cell). Rebuilds only the
/// chunks that changed. With `used_by_composite` the chunks are merged into
/// the sibling Composite Collider 2D's single compound instead (see the engine
/// docs in `physics2d.rs`).
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TilemapCollider2D {
    /// Detect overlaps without a collision response.
    pub is_trigger: bool,
    /// An [`crate::Effector2D`] on this object acts on the collider.
    pub used_by_effector: bool,
    /// Merged into the sibling Composite Collider 2D's compound.
    pub used_by_composite: bool,
    /// `.physmat` override (empty = the project default).
    pub material: String,
    /// Grow every generated polygon by this much (world units) to close
    /// hairline gaps between neighbouring sprite outlines.
    pub extrusion_factor: f32,
    /// More changed chunks than this in one step rebuild the map in one pass
    /// (it makes no difference to the result, only to the work scheduling).
    pub maximum_tile_change_count: u32,
    /// Generate Grid-collider cells as hollow outlines (a polyline chain per
    /// island) instead of solid rectangles. Cheap for big static terrain; a
    /// fast body can tunnel into the hollow unless it uses continuous
    /// collision.
    pub outlines: bool,
    /// Merge Grid-collider cells into maximal rectangles.
    pub merge_cells: bool,
}

impl Default for TilemapCollider2D {
    fn default() -> Self {
        Self {
            is_trigger: false,
            used_by_effector: false,
            used_by_composite: false,
            material: String::new(),
            extrusion_factor: 0.0,
            maximum_tile_change_count: tile_change_limit(),
            outlines: false,
            merge_cells: true,
        }
    }
}

impl TypedComponent for TilemapCollider2D {
    const NAME: &'static str = "Tilemap Collider 2D";
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(layout: CellLayout) -> Grid {
        Grid { layout, cell_size: [2.0, 1.0, 1.0], cell_gap: [0.25, 0.125, 0.0], ..Default::default() }
    }

    #[test]
    fn rectangle_conversions_with_negative_cells() {
        let g = grid(CellLayout::Rectangle);
        assert_eq!(g.cell_to_local(IVec3::new(2, -3, 0)), Vec3::new(4.5, -3.375, 0.0));
        assert_eq!(g.cell_center(IVec3::new(0, 0, 0)), Vec3::new(1.125, 0.5625, 0.0));
        for (x, y) in [(0, 0), (-1, -1), (5, -7), (-9, 4)] {
            let c = IVec3::new(x, y, 0);
            assert_eq!(g.local_to_cell(g.cell_center(c)), c);
            // Just inside the min corner.
            assert_eq!(g.local_to_cell(g.cell_to_local(c) + Vec3::new(0.01, 0.01, 0.0)), c);
        }
    }

    #[test]
    fn isometric_round_trips_all_quadrants() {
        for layout in [CellLayout::Isometric, CellLayout::IsometricZAsY] {
            let g = grid(layout);
            for x in -4..4 {
                for y in -4..4 {
                    let c = IVec3::new(x, y, 0);
                    assert_eq!(g.local_to_cell(g.cell_center(c)), c, "{layout:?} {c}");
                }
            }
        }
        let g = Grid { layout: CellLayout::Isometric, cell_size: [2.0, 1.0, 1.0], ..Default::default() };
        // x = (cx - cy) * W/2, y = (cx + cy) * H/2.
        assert_eq!(g.cell_to_local(IVec3::new(3, 1, 0)), Vec3::new(2.0, 2.0, 0.0));
        // Z lifts the cell in Z-as-Y mode only.
        let z = Grid { layout: CellLayout::IsometricZAsY, ..g.clone() };
        assert_eq!(z.cell_to_local(IVec3::new(0, 0, 2)).y, 2.0);
        assert_eq!(g.cell_to_local(IVec3::new(0, 0, 2)).y, 0.0);
    }

    #[test]
    fn hexagon_layouts_shift_odd_rows_and_round_trip() {
        let g = Grid { layout: CellLayout::HexPointy, cell_size: [1.0, 1.0, 1.0], ..Default::default() };
        assert_eq!(g.cell_to_local(IVec3::new(0, 0, 0)), Vec3::ZERO);
        assert_eq!(g.cell_to_local(IVec3::new(0, 1, 0)), Vec3::new(0.5, 0.75, 0.0));
        assert_eq!(g.cell_to_local(IVec3::new(0, -1, 0)), Vec3::new(0.5, -0.75, 0.0), "negative odd rows shift too");
        assert_eq!(g.cell_to_local(IVec3::new(0, -2, 0)), Vec3::new(0.0, -1.5, 0.0));
        let f = Grid { layout: CellLayout::HexFlat, cell_size: [1.0, 1.0, 1.0], ..Default::default() };
        assert_eq!(f.cell_to_local(IVec3::new(1, 0, 0)), Vec3::new(0.75, 0.5, 0.0));
        assert_eq!(f.cell_to_local(IVec3::new(-1, 0, 0)), Vec3::new(-0.75, 0.5, 0.0));
        for gr in [&g, &f] {
            for x in -5..5 {
                for y in -5..5 {
                    let c = IVec3::new(x, y, 0);
                    assert_eq!(gr.local_to_cell(gr.cell_center(c)), c, "{:?} {c}", gr.layout);
                    // A point a little off the centre stays in the cell.
                    assert_eq!(gr.local_to_cell(gr.cell_center(c) + Vec3::new(0.1, 0.05, 0.0)), c);
                }
            }
        }
    }

    #[test]
    fn swizzle_inverse_undoes_apply() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        for s in CellSwizzle::ALL {
            assert_eq!(s.inverse(s.apply(v)), v, "{s:?}");
        }
        assert_eq!(CellSwizzle::XZY.apply(v), Vec3::new(1.0, 3.0, 2.0));
        let g = Grid { swizzle: CellSwizzle::XZY, ..Default::default() };
        // A ground-plane grid: cell y becomes local z.
        assert_eq!(g.cell_to_local(IVec3::new(1, 2, 0)), Vec3::new(1.0, 0.0, 2.0));
        assert_eq!(g.local_to_cell(Vec3::new(1.5, 0.0, 2.5)), IVec3::new(1, 2, 0));
    }

    #[test]
    fn hex_neighbors_are_symmetric() {
        for layout in [CellLayout::HexPointy, CellLayout::HexFlat] {
            for at in [IVec2::new(0, 0), IVec2::new(1, 1), IVec2::new(-1, -1), IVec2::new(2, -3)] {
                for i in 0..6 {
                    let n = at + neighbor_offset(layout, at, i);
                    // Going to the neighbour and looking for `at` finds it in one slot.
                    assert!((0..6).any(|j| n + neighbor_offset(layout, n, j) == at), "{layout:?} {at} slot {i}");
                }
            }
        }
    }

    #[test]
    fn xform_matrix_and_composition() {
        let r90 = TileXform::new(6, false);
        let p = r90.apply(Vec2::new(1.0, 0.0));
        assert!((p - Vec2::new(0.0, 1.0)).length() < 1e-5);
        let m = TileXform::new(0, true);
        assert!((m.apply(Vec2::new(1.0, 2.0)) - Vec2::new(-1.0, 2.0)).length() < 1e-5);
        let f = TileXform::IDENTITY.flipped_x();
        assert!((f.apply(Vec2::new(1.0, 2.0)) - Vec2::new(-1.0, 2.0)).length() < 1e-5);
        let fy = TileXform::IDENTITY.flipped_y();
        assert!((fy.apply(Vec2::new(1.0, 2.0)) - Vec2::new(1.0, -2.0)).length() < 1e-5);
        // `after` agrees with applying the matrices in order for every pair.
        for a in 0..ROT_STEPS {
            for b in [0u8, 5, 11] {
                for (ma, mb) in [(false, false), (true, false), (false, true), (true, true)] {
                    let (x, y) = (TileXform::new(a, ma), TileXform::new(b, mb));
                    let p = Vec2::new(0.3, -1.7);
                    let want = y.apply(x.apply(p));
                    let got = y.after(x).apply(p);
                    assert!((want - got).length() < 1e-4, "{x:?} then {y:?}");
                }
            }
        }
        assert_eq!(TileXform::IDENTITY.rotated_ccw(6).rotated_cw(6), TileXform::IDENTITY);
    }

    #[test]
    fn chunk_addressing_handles_negatives() {
        assert_eq!(chunk_of(IVec2::new(0, 0)), IVec2::ZERO);
        assert_eq!(chunk_of(IVec2::new(15, 15)), IVec2::ZERO);
        assert_eq!(chunk_of(IVec2::new(16, -1)), IVec2::new(1, -1));
        assert_eq!(chunk_of(IVec2::new(-1, -16)), IVec2::new(-1, -1));
        assert_eq!(chunk_of(IVec2::new(-17, 0)), IVec2::new(-2, 0));
        assert_eq!(cell_index(IVec2::new(-1, -1)), 255);
        assert_eq!(cell_index(IVec2::new(-16, -16)), 0);
        for x in -40..40 {
            for y in -40..40 {
                let c = IVec2::new(x, y);
                assert_eq!(cell_at(chunk_of(c), cell_index(c)), c);
            }
        }
    }

    #[test]
    fn set_get_and_dedup_tables() {
        let mut t = Tilemap::default();
        assert!(t.set_tile(IVec2::new(-3, 5), Some("a.tile")));
        assert!(!t.set_tile(IVec2::new(-3, 5), Some("a.tile")), "same tile is a no-op");
        t.set_tile(IVec2::new(0, 0), Some("a.tile"));
        t.set_tile(IVec2::new(1, 0), Some("b.tile"));
        assert_eq!(t.tile_table().len(), 2, "tile refs are de-duplicated");
        assert_eq!(t.tile_at(IVec2::new(-3, 5)), Some("a.tile"));
        assert_eq!(t.cell_count(), 3);
        assert_eq!(t.bounds(), Some((IVec2::new(-3, 0), IVec2::new(1, 5))));
        // Overrides are side tables and only exist when non-default.
        assert!(t.chunk(IVec2::ZERO).unwrap().color.is_none());
        for x in 0..6 {
            t.set_tile(IVec2::new(x, 1), Some("a.tile"));
            t.set_color(IVec2::new(x, 1), [255, 0, 0, 255]);
            t.set_xform(IVec2::new(x, 1), TileXform::new(6, false));
        }
        assert_eq!(t.color_table_len(), 2, "one shared red");
        assert_eq!(t.xform_table_len(), 2, "one shared rotation");
        assert_eq!(t.color_at(IVec2::new(2, 1)), [255, 0, 0, 255]);
        assert_eq!(t.xform_at(IVec2::new(2, 1)), TileXform::new(6, false));
        assert_eq!(t.color_at(IVec2::new(1, 0)), [255; 4]);
        // A replaced tile starts clean; clearing resets the overrides.
        t.set_tile(IVec2::new(2, 1), Some("b.tile"));
        assert_eq!(t.color_at(IVec2::new(2, 1)), [255; 4]);
        t.set_tile(IVec2::new(3, 1), None);
        assert!(!t.has_tile(IVec2::new(3, 1)));
        assert_eq!(t.cell_state(IVec2::new(3, 1)), CellState::default());
        // Colour overrides need a tile.
        assert!(!t.set_color(IVec2::new(50, 50), [1, 2, 3, 4]));
    }

    #[test]
    fn edits_bump_only_the_affected_chunks() {
        let mut t = Tilemap::default();
        t.set_tile(IVec2::new(0, 0), Some("a.tile"));
        t.set_tile(IVec2::new(16, 0), Some("a.tile"));
        t.set_tile(IVec2::new(40, 40), Some("a.tile"));
        let v = |t: &Tilemap, c: IVec2| t.chunk(c).unwrap().version;
        let before = (v(&t, IVec2::new(0, 0)), v(&t, IVec2::new(1, 0)), v(&t, IVec2::new(2, 2)));
        // An interior cell touches its own chunk only.
        t.set_tile(IVec2::new(5, 5), Some("a.tile"));
        assert_ne!(v(&t, IVec2::new(0, 0)), before.0);
        assert_eq!(v(&t, IVec2::new(1, 0)), before.1);
        assert_eq!(v(&t, IVec2::new(2, 2)), before.2);
        // A border cell also bumps the chunk across the border.
        let b1 = v(&t, IVec2::new(1, 0));
        t.set_tile(IVec2::new(15, 3), Some("a.tile"));
        assert_ne!(v(&t, IVec2::new(1, 0)), b1);
        assert_eq!(v(&t, IVec2::new(2, 2)), before.2);
    }

    #[test]
    fn serialisation_round_trips_and_old_data_defaults() {
        let mut t = Tilemap {
            color: [0.5, 0.5, 0.5, 1.0],
            animation_time_offset: 2.0,
            ..Default::default()
        };
        for x in -20..20 {
            for y in -3..3 {
                t.set_tile(IVec2::new(x, y), Some(if (x + y) % 3 == 0 { "g.tile" } else { "w.tile" }));
            }
        }
        t.set_color(IVec2::new(-5, 1), [10, 20, 30, 255]);
        t.set_xform(IVec2::new(7, -2), TileXform::new(12, true));
        t.set_flags(IVec2::new(7, -2), FLAG_LOCK_TRANSFORM);
        // A tile no cell uses anymore must not be written.
        t.set_tile(IVec2::new(100, 100), Some("gone.tile"));
        t.set_tile(IVec2::new(100, 100), None);
        let text = ron::to_string(&t).unwrap();
        assert!(!text.contains("gone.tile"));
        let back: Tilemap = ron::from_str(&text).unwrap();
        assert_eq!(back, t);
        assert_eq!(back.cell_count(), t.cell_count());
        assert_eq!(back.color_at(IVec2::new(-5, 1)), [10, 20, 30, 255]);
        assert_eq!(back.xform_at(IVec2::new(7, -2)), TileXform::new(12, true));
        assert_eq!(back.flags_at(IVec2::new(7, -2)), FLAG_LOCK_TRANSFORM);
        assert_eq!(back.bounds(), Some((IVec2::new(-20, -3), IVec2::new(19, 2))));
        // Runs keep the file small: 40 x 6 cells are far fewer than 240 entries.
        assert!(text.len() < 40 * 6 * 8, "{} bytes", text.len());
        // Old / partial data.
        let d: Tilemap = ron::from_str("()").unwrap();
        assert!(d.is_empty() && d.anchor == [0.5, 0.5] && d.animation_frame_rate == 1.0 && d.color == [1.0; 4]);
        let r: TilemapRenderer = ron::from_str("(order_in_layer: 4)").unwrap();
        assert_eq!((r.order_in_layer, r.max_chunk_count), (4, 32));
        let c: TilemapCollider2D = ron::from_str("()").unwrap();
        assert_eq!(c, TilemapCollider2D::default());
        let g: Grid = ron::from_str("(cell_size: (2.0, 2.0, 1.0))").unwrap();
        assert_eq!(g.layout, CellLayout::Rectangle);
    }

    #[test]
    fn compact_drops_unused_entries_and_keeps_cells() {
        let mut t = Tilemap::default();
        t.set_tile(IVec2::new(0, 0), Some("a.tile"));
        t.set_tile(IVec2::new(1, 0), Some("b.tile"));
        t.set_color(IVec2::new(1, 0), [1, 2, 3, 4]);
        t.set_tile(IVec2::new(0, 0), None);
        t.set_tile(IVec2::new(1, 0), None);
        t.set_tile(IVec2::new(2, 0), Some("c.tile"));
        t.compact();
        assert_eq!(t.tile_table().len(), 1);
        assert_eq!(t.color_table_len(), 1);
        assert_eq!(t.tile_at(IVec2::new(2, 0)), Some("c.tile"));
    }

    #[test]
    fn sort_order_keys() {
        let (a, b) = (IVec2::new(0, 0), IVec2::new(5, 3));
        assert!(TileSortOrder::BottomLeft.key(a) < TileSortOrder::BottomLeft.key(b));
        assert!(TileSortOrder::TopRight.key(b) < TileSortOrder::TopRight.key(a));
        assert!(TileSortOrder::BottomRight.key(IVec2::new(2, 0)) < TileSortOrder::BottomRight.key(IVec2::new(1, 0)));
    }
}
