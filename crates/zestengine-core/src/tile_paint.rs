//! Tile painting algorithms (2D-5): brushes, box / flood fill, lines, pick,
//! selection copy / move, and the per-cell undo diffs. Pure functions over a
//! [`Tilemap`]; the Tile Palette window and the scene-view glue call them.
//!
//! # Undo
//!
//! A stroke is a [`StrokeRecord`]: every cell it changed once, as
//! `(cell, old, new)` by value ([`CellEdit`]). Repeated writes to a cell inside
//! one stroke coalesce (the first `old`, the last `new`), and cells that end up
//! where they started are dropped. Undo applies the edits backwards, redo
//! forwards; nothing snapshots the map, so a stroke on a 1M-tile map costs as
//! much as the cells it touched.
//!
//! # Shortcuts
//!
//! [`tool_for_key`]: `S` select, `M` move, `B` paint (brush), `U` box fill,
//! `I` picker (eyedropper), `D` erase, `G` flood fill, `L` line. In the scene view,
//! Shift while painting erases, Ctrl picks, Esc cancels a box or selection.

use std::collections::{HashMap, HashSet};

use glam::IVec2;

use crate::tilemap::{CellState, TileXform, Tilemap};

/// The Tile Palette tools.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum TileTool {
    /// Select cells with a box.
    Select,
    /// Move the selection.
    Move,
    /// Paint the brush.
    #[default]
    Paint,
    /// Fill a rectangle.
    BoxFill,
    /// Eyedropper: copy a cell's state.
    Pick,
    /// Clear cells.
    Erase,
    /// Fill a connected area.
    FloodFill,
    /// Draw a line.
    Line,
}

impl TileTool {
    /// Every tool, in declaration order.
    pub const ALL: [TileTool; 8] = [Self::Select, Self::Move, Self::Paint, Self::BoxFill, Self::Pick, Self::Erase, Self::FloodFill, Self::Line];
    /// Inspector label of this tool.
    pub fn label(self) -> &'static str {
        match self {
            Self::Select => "Select",
            Self::Move => "Move",
            Self::Paint => "Paint",
            Self::BoxFill => "Box Fill",
            Self::Pick => "Pick",
            Self::Erase => "Erase",
            Self::FloodFill => "Flood Fill",
            Self::Line => "Line",
        }
    }
    /// The shortcut key that selects this tool.
    pub fn key(self) -> char {
        match self {
            Self::Select => 'S',
            Self::Move => 'M',
            Self::Paint => 'B',
            Self::BoxFill => 'U',
            Self::Pick => 'I',
            Self::Erase => 'D',
            Self::FloodFill => 'G',
            Self::Line => 'L',
        }
    }
}

/// The tool a shortcut key selects.
pub fn tool_for_key(c: char) -> Option<TileTool> {
    TileTool::ALL.into_iter().find(|t| t.key().eq_ignore_ascii_case(&c))
}

/// One changed cell.
#[derive(Clone, PartialEq, Debug)]
pub struct CellEdit {
    /// The changed cell.
    pub cell: IVec2,
    /// Its state before.
    pub old: CellState,
    /// Its state after.
    pub new: CellState,
}

/// Apply edits forward (`new`) or backward (`old`). Backward walks in reverse
/// so overlapping edits of one list restore exactly.
pub fn apply_edits(map: &mut Tilemap, edits: &[CellEdit], forward: bool) {
    if forward {
        for e in edits {
            map.set_cell(e.cell, &e.new);
        }
    } else {
        for e in edits.iter().rev() {
            map.set_cell(e.cell, &e.old);
        }
    }
}

/// The edits of one mouse stroke, coalesced per cell.
#[derive(Clone, Default, Debug)]
pub struct StrokeRecord {
    edits: Vec<CellEdit>,
    at: HashMap<IVec2, usize>,
}

impl StrokeRecord {
    /// An empty record.
    pub fn new() -> Self {
        Self::default()
    }
    /// Add edits (already applied to the map by the caller).
    pub fn push(&mut self, edits: Vec<CellEdit>) {
        for e in edits {
            match self.at.get(&e.cell) {
                Some(&i) => self.edits[i].new = e.new,
                None => {
                    self.at.insert(e.cell, self.edits.len());
                    self.edits.push(e);
                }
            }
        }
    }
    /// True when every edit is a no-op.
    pub fn is_empty(&self) -> bool {
        self.edits.iter().all(|e| e.old == e.new)
    }
    /// Number of coalesced edits.
    pub fn len(&self) -> usize {
        self.edits.len()
    }
    /// The coalesced edits without no-ops.
    pub fn finish(self) -> Vec<CellEdit> {
        self.edits.into_iter().filter(|e| e.old != e.new).collect()
    }
}

/// A rectangle of cells (inclusive corners).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Selection {
    /// First corner (inclusive).
    pub min: IVec2,
    /// Second corner (inclusive).
    pub max: IVec2,
}

impl Selection {
    /// A selection spanning the two corners.
    pub fn from_corners(a: IVec2, b: IVec2) -> Self {
        Self { min: a.min(b), max: a.max(b) }
    }
    /// Size in cells.
    pub fn size(&self) -> IVec2 {
        self.max - self.min + IVec2::ONE
    }
    /// True when the cell is inside.
    pub fn contains(&self, c: IVec2) -> bool {
        c.cmpge(self.min).all() && c.cmple(self.max).all()
    }
    /// Every cell, row-major.
    pub fn cells(&self) -> impl Iterator<Item = IVec2> {
        let (min, max) = (self.min, self.max);
        (min.y..=max.y).flat_map(move |y| (min.x..=max.x).map(move |x| IVec2::new(x, y)))
    }
}

/// A block of cells to paint, row-major from the bottom row (`y * w + x`);
/// `None` leaves the cell alone.
#[derive(Clone, PartialEq, Debug)]
pub struct Brush {
    /// Size in cells.
    pub size: IVec2,
    /// Cells row-major from the bottom row; None leaves the cell alone.
    pub cells: Vec<Option<CellState>>,
    /// The cell under the mouse.
    pub pivot: IVec2,
}

/// Copied cells (the same shape as a brush).
pub type TileClipboard = Brush;

impl Brush {
    /// A 1x1 brush.
    pub fn single(c: CellState) -> Self {
        Self { size: IVec2::ONE, cells: vec![Some(c)], pivot: IVec2::ZERO }
    }
    /// A 1x1 brush holding the tile at this project-relative path.
    pub fn tile(path: &str) -> Self {
        Self::single(CellState::tile(path))
    }
    /// The cell at (x, y), or None outside the brush.
    pub fn get(&self, x: i32, y: i32) -> Option<&CellState> {
        if x < 0 || y < 0 || x >= self.size.x || y >= self.size.y {
            return None;
        }
        self.cells[(y * self.size.x + x) as usize].as_ref()
    }
    /// True when every cell is None.
    pub fn is_empty(&self) -> bool {
        self.cells.iter().all(Option::is_none)
    }
    /// Mirror left-right (tile transforms flip too).
    pub fn flipped_x(&self) -> Brush {
        let mut b = self.clone();
        for y in 0..self.size.y {
            for x in 0..self.size.x {
                b.cells[(y * self.size.x + x) as usize] = self.get(self.size.x - 1 - x, y).cloned().map(|mut c| {
                    c.xform = c.xform.flipped_x();
                    c
                });
            }
        }
        b.pivot = IVec2::new(self.size.x - 1 - self.pivot.x, self.pivot.y);
        b
    }
    /// Mirrored top-bottom (tile transforms flip too).
    pub fn flipped_y(&self) -> Brush {
        let mut b = self.clone();
        for y in 0..self.size.y {
            for x in 0..self.size.x {
                b.cells[(y * self.size.x + x) as usize] = self.get(x, self.size.y - 1 - y).cloned().map(|mut c| {
                    c.xform = c.xform.flipped_y();
                    c
                });
            }
        }
        b.pivot = IVec2::new(self.pivot.x, self.size.y - 1 - self.pivot.y);
        b
    }
}

/// Rotate a brush a quarter turn clockwise: the cell block transposes and every
/// cell's own transform turns with it (`step` = [`TileXform::rot`] units of a
/// quarter turn on rectangle grids).
pub fn rotate_brush_cw(b: &Brush) -> Brush {
    let (w, h) = (b.size.x, b.size.y);
    let mut out = Brush { size: IVec2::new(h, w), cells: vec![None; b.cells.len()], pivot: IVec2::new(b.pivot.y, w - 1 - b.pivot.x) };
    for y in 0..h {
        for x in 0..w {
            // (x, y) -> (y, w - 1 - x) is a clockwise quarter turn of the block.
            let (nx, ny) = (y, w - 1 - x);
            out.cells[(ny * h + nx) as usize] = b.get(x, y).cloned().map(|mut c| {
                c.xform = c.xform.rotated_cw(crate::tilemap::ROT_STEPS / 4);
                c
            });
        }
    }
    out
}

/// Edits painting `brush` with its pivot on `at`.
pub fn paint_brush(map: &Tilemap, brush: &Brush, at: IVec2) -> Vec<CellEdit> {
    let origin = at - brush.pivot;
    let mut v = Vec::new();
    for y in 0..brush.size.y {
        for x in 0..brush.size.x {
            if let Some(c) = brush.get(x, y) {
                let cell = origin + IVec2::new(x, y);
                let old = map.cell_state(cell);
                if old != *c {
                    v.push(CellEdit { cell, old, new: c.clone() });
                }
            }
        }
    }
    v
}

/// A one-cell brush holding one of `brush`'s tiles, chosen by hashing the cell
/// and `seed`: the same cell always gets the same pick (a repaint is a no-op,
/// undo / redo are exact), and a flip or rotation of the source tile travels
/// with it. An empty brush gives an empty one.
pub fn random_pick(brush: &Brush, cell: IVec2, seed: u32) -> Brush {
    let tiles: Vec<&CellState> = brush.cells.iter().flatten().collect();
    if tiles.is_empty() {
        return Brush { size: IVec2::ONE, cells: vec![None], pivot: IVec2::ZERO };
    }
    let mut h = (cell.x as u32).wrapping_mul(0x9E37_79B1) ^ (cell.y as u32).wrapping_mul(0x85EB_CA6B) ^ seed.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    Brush::single(tiles[h as usize % tiles.len()].clone())
}

/// [`paint_brush`], or with `random` one random tile of the brush on `at`.
pub fn paint_brush_mode(map: &Tilemap, brush: &Brush, at: IVec2, random: bool, seed: u32) -> Vec<CellEdit> {
    if random { paint_brush(map, &random_pick(brush, at, seed), at) } else { paint_brush(map, brush, at) }
}

/// Edits drawing the brush along the line between two cells (stamped on every
/// cell of it, overlaps coalesced), or with `random` a random tile per cell.
pub fn line_brush(map: &Tilemap, a: IVec2, b: IVec2, brush: &Brush, random: bool, seed: u32) -> Vec<CellEdit> {
    let mut rec = StrokeRecord::new();
    for cell in line_cells(a, b, false) {
        rec.push(paint_brush_mode(map, brush, cell, random, seed));
    }
    rec.finish()
}

/// [`box_fill`] with a random tile of the brush per cell.
pub fn box_fill_random(map: &Tilemap, a: IVec2, b: IVec2, brush: &Brush, seed: u32) -> Vec<CellEdit> {
    let mut v = Vec::new();
    for cell in Selection::from_corners(a, b).cells() {
        let pick = random_pick(brush, cell, seed);
        let Some(c) = pick.get(0, 0) else { continue };
        let old = map.cell_state(cell);
        if old != *c {
            v.push(CellEdit { cell, old, new: c.clone() });
        }
    }
    v
}

/// Edits clearing the cells (empty cells are skipped).
pub fn erase_cells(map: &Tilemap, cells: impl IntoIterator<Item = IVec2>) -> Vec<CellEdit> {
    cells
        .into_iter()
        .filter(|&c| map.has_tile(c))
        .map(|cell| CellEdit { cell, old: map.cell_state(cell), new: CellState::default() })
        .collect()
}

/// The cells of the box between two corners (any order), row by row.
pub fn box_cells(a: IVec2, b: IVec2) -> Vec<IVec2> {
    Selection::from_corners(a, b).cells().collect()
}

/// Edits filling a box with the brush repeated across it (a one-cell brush
/// fills it with that tile; a larger one tiles from the box's min corner).
pub fn box_fill(map: &Tilemap, a: IVec2, b: IVec2, brush: &Brush) -> Vec<CellEdit> {
    let sel = Selection::from_corners(a, b);
    let mut v = Vec::new();
    for cell in sel.cells() {
        let l = cell - sel.min;
        let Some(c) = brush.get(l.x.rem_euclid(brush.size.x.max(1)), l.y.rem_euclid(brush.size.y.max(1))) else { continue };
        let old = map.cell_state(cell);
        if old != *c {
            v.push(CellEdit { cell, old, new: c.clone() });
        }
    }
    v
}

/// The cells a flood fill from `start` replaces: the 4-connected region of
/// cells showing the same tile as `start` (empty counts as a tile) inside
/// `limit`, capped at `cap` cells. With `contiguous` false every matching cell
/// inside `limit` is returned instead (swap all instances).
pub fn flood_fill_cells(map: &Tilemap, start: IVec2, limit: Selection, cap: usize, contiguous: bool) -> Vec<IVec2> {
    if !limit.contains(start) {
        return Vec::new();
    }
    let target = map.tile_at(start).map(str::to_string);
    let same = |c: IVec2| map.tile_at(c).map(str::to_string) == target;
    if !contiguous {
        return limit.cells().filter(|&c| same(c)).take(cap).collect();
    }
    let mut seen: HashSet<IVec2> = HashSet::new();
    let mut out = Vec::new();
    let mut stack = vec![start];
    seen.insert(start);
    while let Some(c) = stack.pop() {
        if out.len() >= cap {
            break;
        }
        out.push(c);
        for d in [IVec2::X, IVec2::NEG_X, IVec2::Y, IVec2::NEG_Y] {
            let n = c + d;
            if limit.contains(n) && same(n) && seen.insert(n) {
                stack.push(n);
            }
        }
    }
    out
}

/// Edits flood filling with the brush's first cell.
pub fn flood_fill(map: &Tilemap, start: IVec2, limit: Selection, cap: usize, contiguous: bool, brush: &Brush) -> Vec<CellEdit> {
    let Some(c) = brush.cells.iter().flatten().next() else { return Vec::new() };
    flood_fill_cells(map, start, limit, cap, contiguous)
        .into_iter()
        .filter_map(|cell| {
            let old = map.cell_state(cell);
            (old != *c).then(|| CellEdit { cell, old, new: c.clone() })
        })
        .collect()
}

/// The cells of a line between two cells (Bresenham); `fill_gaps` adds the
/// corner cells that make a diagonal 4-connected.
pub fn line_cells(a: IVec2, b: IVec2, fill_gaps: bool) -> Vec<IVec2> {
    let mut out = Vec::new();
    let (mut x, mut y) = (a.x, a.y);
    let (dx, dy) = ((b.x - a.x).abs(), -(b.y - a.y).abs());
    let (sx, sy) = (if a.x < b.x { 1 } else { -1 }, if a.y < b.y { 1 } else { -1 });
    let mut err = dx + dy;
    loop {
        out.push(IVec2::new(x, y));
        if x == b.x && y == b.y {
            break;
        }
        let e2 = 2 * err;
        let (mut stepped_x, mut stepped_y) = (false, false);
        if e2 >= dy {
            err += dy;
            x += sx;
            stepped_x = true;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
            stepped_y = true;
        }
        if fill_gaps && stepped_x && stepped_y {
            out.push(IVec2::new(x - sx, y));
        }
    }
    out
}

/// Copy the cells of a selection into a brush (empty cells stay `None`).
pub fn pick_brush(map: &Tilemap, sel: Selection) -> Brush {
    let size = sel.size();
    let mut cells = Vec::with_capacity((size.x * size.y) as usize);
    for c in sel.cells() {
        cells.push(map.has_tile(c).then(|| map.cell_state(c)));
    }
    Brush { size, cells, pivot: IVec2::ZERO }
}

/// Edits moving the selection's cells by `delta` (the source cells empty
/// afterwards, the destination overwritten by the moved non-empty cells).
pub fn move_cells(map: &Tilemap, sel: Selection, delta: IVec2) -> Vec<CellEdit> {
    if delta == IVec2::ZERO {
        return Vec::new();
    }
    let moved: Vec<(IVec2, CellState)> = sel.cells().filter(|&c| map.has_tile(c)).map(|c| (c + delta, map.cell_state(c))).collect();
    let mut after: HashMap<IVec2, CellState> = HashMap::new();
    for c in sel.cells().filter(|&c| map.has_tile(c)) {
        after.insert(c, CellState::default());
    }
    for (c, s) in moved {
        after.insert(c, s);
    }
    let mut keys: Vec<IVec2> = after.keys().copied().collect();
    keys.sort_by_key(|c| (c.y, c.x));
    keys.into_iter()
        .filter_map(|cell| {
            let old = map.cell_state(cell);
            let new = after.remove(&cell).unwrap();
            (old != new).then_some(CellEdit { cell, old, new })
        })
        .collect()
}

/// Set a whole cell to a transform-rotated copy (the Rotate / Flip tools on a selection).
pub fn transform_cells(map: &Tilemap, sel: Selection, f: impl Fn(TileXform) -> TileXform) -> Vec<CellEdit> {
    sel.cells()
        .filter(|&c| map.has_tile(c))
        .filter_map(|cell| {
            let old = map.cell_state(cell);
            let mut new = old.clone();
            new.xform = f(old.xform);
            (old != new).then_some(CellEdit { cell, old, new })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map_with(cells: &[(i32, i32, &str)]) -> Tilemap {
        let mut m = Tilemap::default();
        for &(x, y, t) in cells {
            m.set_tile(IVec2::new(x, y), Some(t));
        }
        m
    }

    fn snapshot(m: &Tilemap) -> Vec<(IVec2, CellState)> {
        let mut v = Vec::new();
        m.for_each_cell(|c, _| v.push((c, m.cell_state(c))));
        v.sort_by_key(|(c, _)| (c.y, c.x));
        v
    }

    #[test]
    fn shortcut_keys_pick_tools() {
        for t in TileTool::ALL {
            assert_eq!(tool_for_key(t.key()), Some(t));
            assert_eq!(tool_for_key(t.key().to_ascii_lowercase()), Some(t));
        }
        assert_eq!(tool_for_key('b'), Some(TileTool::Paint));
        assert_eq!(tool_for_key('x'), None);
    }

    #[test]
    fn box_fill_covers_the_box_and_repeats_a_brush() {
        let m = Tilemap::default();
        let e = box_fill(&m, IVec2::new(3, 2), IVec2::new(-1, 0), &Brush::tile("g.tile"));
        assert_eq!(e.len(), 5 * 3);
        assert_eq!(box_cells(IVec2::new(1, 1), IVec2::new(0, 0)).len(), 4);
        // A 2 x 1 brush alternates along the box.
        let b = Brush { size: IVec2::new(2, 1), cells: vec![Some(CellState::tile("a")), Some(CellState::tile("b"))], pivot: IVec2::ZERO };
        let mut m = Tilemap::default();
        let e = box_fill(&m, IVec2::new(0, 0), IVec2::new(3, 0), &b);
        apply_edits(&mut m, &e, true);
        let row: Vec<_> = (0..4).map(|x| m.tile_at(IVec2::new(x, 0)).unwrap().to_string()).collect();
        assert_eq!(row, ["a", "b", "a", "b"]);
    }

    fn three_tile_brush() -> Brush {
        Brush { size: IVec2::new(3, 1), cells: vec![Some(CellState::tile("a")), Some(CellState::tile("b")), Some(CellState::tile("c"))], pivot: IVec2::ZERO }
    }

    #[test]
    fn line_brush_draws_a_contiguous_line_and_undoes_exactly() {
        let mut m = map_with(&[(2, 2, "old")]);
        let before = m.clone();
        let e = line_brush(&m, IVec2::new(0, 0), IVec2::new(5, 3), &Brush::tile("g"), false, 0);
        let expect = line_cells(IVec2::new(0, 0), IVec2::new(5, 3), false);
        assert_eq!(e.len(), expect.len());
        apply_edits(&mut m, &e, true);
        assert!(expect.iter().all(|c| m.tile_at(*c) == Some("g")));
        apply_edits(&mut m, &e, false);
        assert_eq!(m.tile_at(IVec2::new(2, 2)), before.tile_at(IVec2::new(2, 2)));
        assert!(!m.has_tile(IVec2::new(0, 0)));
        // A larger brush stamps overlapping footprints; the coalesced edits still undo exactly.
        let e = line_brush(&m, IVec2::new(0, 0), IVec2::new(4, 0), &three_tile_brush(), false, 0);
        apply_edits(&mut m, &e, true);
        apply_edits(&mut m, &e, false);
        assert_eq!(m.tile_at(IVec2::new(2, 2)), before.tile_at(IVec2::new(2, 2)));
        assert!(!m.has_tile(IVec2::new(1, 0)));
    }

    #[test]
    fn random_brush_is_stable_per_cell_uses_every_tile_and_keeps_transforms() {
        let b = three_tile_brush();
        let seen: HashSet<String> = (0..200).map(|x| random_pick(&b, IVec2::new(x, 7), 1).get(0, 0).unwrap().tile.clone()).collect();
        assert_eq!(seen.len(), 3, "all tiles appear: {seen:?}");
        assert_eq!(random_pick(&b, IVec2::new(4, 4), 9), random_pick(&b, IVec2::new(4, 4), 9), "same cell, same seed, same pick");
        assert!((0..50).any(|x| random_pick(&b, IVec2::new(x, 0), 1) != random_pick(&b, IVec2::new(x, 0), 2)), "the seed changes the picks");
        let mut flipped = three_tile_brush();
        flipped.cells[0].as_mut().unwrap().xform.mirror = true;
        assert!((0..100).map(|x| random_pick(&flipped, IVec2::new(x, 0), 3)).any(|p| p.get(0, 0).unwrap().xform.mirror));
        assert!(random_pick(&Brush { size: IVec2::ONE, cells: vec![None], pivot: IVec2::ZERO }, IVec2::ZERO, 0).is_empty());
        // Painting twice at one cell with the same seed changes nothing the second time.
        let mut m = Tilemap::default();
        let e = paint_brush_mode(&m, &b, IVec2::new(3, 3), true, 5);
        assert_eq!(e.len(), 1);
        apply_edits(&mut m, &e, true);
        assert!(paint_brush_mode(&m, &b, IVec2::new(3, 3), true, 5).is_empty());
        // Box fill: every cell is one of the tiles and more than one tile shows up.
        let e = box_fill_random(&Tilemap::default(), IVec2::ZERO, IVec2::new(9, 9), &b, 2);
        assert_eq!(e.len(), 100);
        assert!(e.iter().map(|x| x.new.tile.clone()).collect::<HashSet<_>>().len() == 3);
    }

    #[test]
    fn flood_fill_stays_inside_the_region_and_the_limit() {
        // A walled room: fill the inside only.
        let mut cells = Vec::new();
        for i in 0..6 {
            for (x, y) in [(i, 0), (i, 5), (0, i), (5, i)] {
                cells.push((x, y, "wall"));
            }
        }
        let m = map_with(&cells);
        let lim = Selection { min: IVec2::new(-3, -3), max: IVec2::new(9, 9) };
        let inside = flood_fill_cells(&m, IVec2::new(2, 2), lim, 1000, true);
        assert_eq!(inside.len(), 16);
        assert!(inside.iter().all(|c| (1..=4).contains(&c.x) && (1..=4).contains(&c.y)));
        // Outside is bounded by the limit: 13x13 minus the wall ring minus the room.
        let outside = flood_fill_cells(&m, IVec2::new(-1, -1), lim, 10_000, true);
        assert_eq!(outside.len(), 13 * 13 - 36 + 16 - 16);
        // The cap stops a runaway fill.
        assert_eq!(flood_fill_cells(&m, IVec2::new(-1, -1), lim, 20, true).len(), 20);
        // Filling a wall cell swaps the connected wall; non-contiguous swaps all instances.
        let mut m2 = map_with(&[(0, 0, "wall"), (1, 0, "wall"), (5, 5, "wall")]);
        assert_eq!(flood_fill_cells(&m2, IVec2::ZERO, lim, 100, true).len(), 2);
        assert_eq!(flood_fill_cells(&m2, IVec2::ZERO, lim, 100, false).len(), 3);
        let e = flood_fill(&m2, IVec2::ZERO, lim, 100, false, &Brush::tile("ice"));
        apply_edits(&mut m2, &e, true);
        assert_eq!(m2.tile_at(IVec2::new(5, 5)), Some("ice"));
        // Outside the limit does nothing.
        assert!(flood_fill_cells(&m, IVec2::new(50, 50), lim, 100, true).is_empty());
    }

    #[test]
    fn lines_are_connected() {
        let l = line_cells(IVec2::new(0, 0), IVec2::new(5, 2), false);
        assert_eq!((l[0], *l.last().unwrap()), (IVec2::ZERO, IVec2::new(5, 2)));
        assert!(l.windows(2).all(|w| (w[1] - w[0]).abs().max_element() == 1));
        let g = line_cells(IVec2::new(0, 0), IVec2::new(4, 4), true);
        // 4-connected: consecutive cells differ along exactly one axis.
        assert!(g.windows(2).all(|w| (w[1] - w[0]).abs().element_sum() <= 1), "{g:?}");
        assert_eq!(line_cells(IVec2::new(3, 3), IVec2::new(3, 3), false), vec![IVec2::new(3, 3)]);
    }

    #[test]
    fn rotating_a_brush_keeps_cells_and_turns_transforms() {
        let b = Brush {
            size: IVec2::new(3, 2),
            cells: (0..6).map(|i| Some(CellState::tile(&format!("t{i}")))).collect(),
            pivot: IVec2::new(0, 0),
        };
        let r = rotate_brush_cw(&b);
        assert_eq!(r.size, IVec2::new(2, 3));
        // The bottom-left cell goes to the top-left of the rotated block.
        assert_eq!(r.get(0, 2).unwrap().tile, "t0");
        assert_eq!(r.get(0, 2).unwrap().xform, TileXform::new(18, false));
        // Four quarter turns come back.
        let back = rotate_brush_cw(&rotate_brush_cw(&rotate_brush_cw(&r)));
        assert_eq!(back, b);
        let f = b.flipped_x().flipped_x();
        assert_eq!(f, b);
    }

    #[test]
    fn stroke_coalesces_and_undo_redo_are_exact_over_random_strokes() {
        let mut seed = 99u32;
        let mut rnd = move |n: u32| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed % n) as i32
        };
        let tiles = ["a", "b", "c"];
        let mut m = Tilemap::default();
        type PaintHistory = (Vec<CellEdit>, Vec<(IVec2, CellState)>);
        let mut history: Vec<PaintHistory> = Vec::new();
        for _ in 0..30 {
            let before = snapshot(&m);
            let mut rec = StrokeRecord::new();
            for _ in 0..rnd(12) + 1 {
                let c = IVec2::new(rnd(9) - 4, rnd(9) - 4);
                let edits = match rnd(4) {
                    0 => erase_cells(&m, [c]),
                    1 => box_fill(&m, c, c + IVec2::new(rnd(3), rnd(3)), &Brush::tile(tiles[rnd(3) as usize])),
                    2 => flood_fill(&m, c, Selection { min: IVec2::splat(-6), max: IVec2::splat(6) }, 500, rnd(2) == 0, &Brush::tile(tiles[rnd(3) as usize])),
                    _ => paint_brush(&m, &Brush::tile(tiles[rnd(3) as usize]), c),
                };
                apply_edits(&mut m, &edits, true);
                rec.push(edits);
            }
            let edits = rec.finish();
            history.push((edits, before));
        }
        let end = snapshot(&m);
        // Undo everything: each step restores the exact earlier state.
        for (edits, before) in history.iter().rev() {
            apply_edits(&mut m, edits, false);
            assert_eq!(&snapshot(&m), before);
        }
        assert_eq!(m.cell_count(), 0);
        // Redo everything: the original result again.
        for (edits, _) in &history {
            apply_edits(&mut m, edits, true);
        }
        assert_eq!(snapshot(&m), end);
    }

    #[test]
    fn stroke_drops_cells_that_end_where_they_started() {
        let m = map_with(&[(0, 0, "a")]);
        let mut rec = StrokeRecord::new();
        rec.push(vec![CellEdit { cell: IVec2::ZERO, old: m.cell_state(IVec2::ZERO), new: CellState::tile("b") }]);
        rec.push(vec![CellEdit { cell: IVec2::ZERO, old: CellState::tile("b"), new: CellState::tile("a") }]);
        rec.push(vec![CellEdit { cell: IVec2::X, old: CellState::default(), new: CellState::tile("c") }]);
        assert_eq!(rec.len(), 2);
        let e = rec.finish();
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].cell, IVec2::X);
    }

    #[test]
    fn select_copy_paste_and_move() {
        let m = map_with(&[(0, 0, "a"), (1, 0, "b"), (1, 1, "c")]);
        let sel = Selection::from_corners(IVec2::new(1, 1), IVec2::new(0, 0));
        let clip = pick_brush(&m, sel);
        assert_eq!(clip.size, IVec2::new(2, 2));
        assert!(clip.get(0, 1).is_none(), "an empty cell stays empty");
        // Paste elsewhere leaves the gap alone.
        let mut m2 = map_with(&[(10, 11, "keep")]);
        let e = paint_brush(&m2, &clip, IVec2::new(10, 10));
        apply_edits(&mut m2, &e, true);
        assert_eq!(m2.tile_at(IVec2::new(10, 10)), Some("a"));
        assert_eq!(m2.tile_at(IVec2::new(10, 11)), Some("keep"));
        assert_eq!(m2.tile_at(IVec2::new(11, 11)), Some("c"));
        // Move the selection one to the right.
        let mut m3 = m.clone();
        let e = move_cells(&m3, sel, IVec2::new(1, 0));
        apply_edits(&mut m3, &e, true);
        assert!(!m3.has_tile(IVec2::new(0, 0)));
        assert_eq!(m3.tile_at(IVec2::new(1, 0)), Some("a"));
        assert_eq!(m3.tile_at(IVec2::new(2, 0)), Some("b"));
        assert_eq!(m3.tile_at(IVec2::new(2, 1)), Some("c"));
        apply_edits(&mut m3, &e, false);
        assert_eq!(snapshot(&m3), snapshot(&m));
    }

    #[test]
    fn overrides_survive_pick_and_paste() {
        let mut m = map_with(&[(0, 0, "a")]);
        m.set_color(IVec2::ZERO, [1, 2, 3, 255]);
        m.set_xform(IVec2::ZERO, TileXform::new(6, true));
        let clip = pick_brush(&m, Selection::from_corners(IVec2::ZERO, IVec2::ZERO));
        let mut m2 = Tilemap::default();
        let e = paint_brush(&m2, &clip, IVec2::new(4, 4));
        apply_edits(&mut m2, &e, true);
        assert_eq!(m2.color_at(IVec2::new(4, 4)), [1, 2, 3, 255]);
        assert_eq!(m2.xform_at(IVec2::new(4, 4)), TileXform::new(6, true));
        let e = transform_cells(&m, Selection::from_corners(IVec2::ZERO, IVec2::ZERO), |x| x.flipped_y());
        assert_eq!(e.len(), 1);
    }
}
