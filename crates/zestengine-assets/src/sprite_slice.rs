//! Slicing algorithms for the Sprite Editor (2D-4), pure and tested: automatic
//! (alpha island detection), the isometric diamond grid, and applying a cut to a
//! [`SpriteSheet`] with default pivots for the new sprites. The plain grids
//! (cell size / cell count) live in `sprite.rs`.
//!
//! Automatic slicing is described in the research only by its shape (flood fill
//! over a visited set, nested rects dropped, result sorted; connectivity and
//! threshold unknown), so the choices here are ours: a pixel is solid when its
//! alpha is **above** the threshold, islands are **8-connected** (touching
//! corners join), rects fully inside another island's rect are dropped (a
//! sprite inside the hole of a ring is not a sprite of its own), islands with a
//! side below the minimum are dropped, and the result reads left to right in
//! rows from the top.

use std::collections::VecDeque;

use crate::sprite::{PivotPreset, PixelRect, SliceMethod, SpriteSheet, drop_empty, grid_rects, grid_rects_by_count};
use crate::sprite_mesh::{AlphaMap, P2};

/// Default minimum island side for automatic slicing (the research: 4).
pub const AUTO_MIN_SIZE: u32 = 4;

/// Bounding rects of the connected solid islands of `a`.
pub fn auto_slice(a: &AlphaMap, threshold: u8, min_size: u32) -> Vec<PixelRect> {
    let (w, h) = (a.w as usize, a.h as usize);
    let mut seen = vec![false; w * h];
    let mut rects: Vec<PixelRect> = Vec::new();
    let mut queue: VecDeque<(u32, u32)> = VecDeque::new();
    for sy in 0..h {
        for sx in 0..w {
            if seen[sy * w + sx] || a.alpha[sy * w + sx] <= threshold {
                continue;
            }
            seen[sy * w + sx] = true;
            queue.push_back((sx as u32, sy as u32));
            let (mut x0, mut y0, mut x1, mut y1) = (sx as u32, sy as u32, sx as u32, sy as u32);
            while let Some((x, y)) = queue.pop_front() {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                        if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                            continue;
                        }
                        let i = ny as usize * w + nx as usize;
                        if !seen[i] && a.alpha[i] > threshold {
                            seen[i] = true;
                            queue.push_back((nx as u32, ny as u32));
                        }
                    }
                }
            }
            rects.push(PixelRect::new(x0, y0, x1 - x0 + 1, y1 - y0 + 1));
        }
    }
    rects.retain(|r| r.w >= min_size && r.h >= min_size);
    // Drop rects contained in another one (equal rects keep the first).
    let keep: Vec<bool> = (0..rects.len())
        .map(|i| !(0..rects.len()).any(|j| j != i && contains(&rects[j], &rects[i]) && (rects[j] != rects[i] || j < i)))
        .collect();
    let rects: Vec<PixelRect> = rects.into_iter().zip(keep).filter_map(|(r, k)| k.then_some(r)).collect();
    reading_order(rects)
}

fn contains(a: &PixelRect, b: &PixelRect) -> bool {
    b.x >= a.x && b.y >= a.y && b.x + b.w <= a.x + a.w && b.y + b.h <= a.y + a.h
}

/// Left to right in rows from the top; a rect joins the current row while its
/// centre is above the bottom of the row's first rect.
pub fn reading_order(mut rects: Vec<PixelRect>) -> Vec<PixelRect> {
    rects.sort_by_key(|r| (r.y, r.x));
    let mut out: Vec<PixelRect> = Vec::with_capacity(rects.len());
    let mut row: Vec<PixelRect> = Vec::new();
    let mut row_bottom = 0u32;
    for r in rects {
        let cy = r.y + r.h / 2;
        if row.is_empty() || cy < row_bottom {
            if row.is_empty() {
                row_bottom = r.y + r.h;
            }
            row.push(r);
        } else {
            row.sort_by_key(|r| (r.x, r.y));
            out.append(&mut row);
            row_bottom = r.y + r.h;
            row.push(r);
        }
    }
    row.sort_by_key(|r| (r.x, r.y));
    out.append(&mut row);
    out
}

/// Settings of the isometric grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IsoSettings {
    /// Full diamond bounding box.
    pub cell: [u32; 2],
    pub offset: [u32; 2],
    /// Which rows start shifted by half a cell.
    pub alternate: bool,
}

impl Default for IsoSettings {
    fn default() -> Self {
        Self { cell: [64, 32], offset: [0, 0], alternate: false }
    }
}

/// One isometric cell: its rect and the diamond outline (rect-local, y down).
#[derive(Clone, Debug, PartialEq)]
pub struct IsoCell {
    pub rect: PixelRect,
    pub outline: Vec<P2>,
}

/// A cell is kept when more than this share of its diamond is solid.
pub const ISO_MIN_COVERAGE: f32 = 0.01;

/// The diamond inside a `w x h` rect (top, right, bottom, left).
pub fn diamond(w: u32, h: u32) -> Vec<P2> {
    let (w, h) = (w as f32, h as f32);
    vec![[w / 2.0, 0.0], [w, h / 2.0], [w / 2.0, h], [0.0, h / 2.0]]
}

/// Cut an `iw x ih` image into isometric diamonds. Rows run down in steps of
/// `cell.y / 2`; every other row is shifted right by `cell.x / 2` (`alternate`
/// picks which). Only cells fully inside the image count; with `alpha`, cells
/// whose diamond is not more than 1% solid are dropped.
pub fn isometric_cells(iw: u32, ih: u32, s: &IsoSettings, alpha: Option<(&AlphaMap, u8)>) -> Vec<IsoCell> {
    let mut out = Vec::new();
    let [cw, ch] = s.cell;
    if cw < 2 || ch < 2 {
        return out;
    }
    let step = (ch / 2).max(1);
    let mut row = 0u32;
    let mut y = s.offset[1];
    while y + ch <= ih {
        let shifted = (row % 2 == 1) != s.alternate;
        let mut x = s.offset[0] + if shifted { cw / 2 } else { 0 };
        while x + cw <= iw {
            let rect = PixelRect::new(x, y, cw, ch);
            let keep = alpha.is_none_or(|(a, t)| diamond_coverage(a, rect, t) > ISO_MIN_COVERAGE);
            if keep {
                out.push(IsoCell { rect, outline: diamond(cw, ch) });
            }
            x += cw;
        }
        y += step;
        row += 1;
    }
    out
}

/// Share of the diamond inside `rect` whose pixels are solid.
pub fn diamond_coverage(a: &AlphaMap, rect: PixelRect, threshold: u8) -> f32 {
    let (hw, hh) = (rect.w as f32 / 2.0, rect.h as f32 / 2.0);
    let (mut inside, mut solid) = (0u32, 0u32);
    for y in 0..rect.h {
        for x in 0..rect.w {
            let (dx, dy) = ((x as f32 + 0.5 - hw).abs() / hw, (y as f32 + 0.5 - hh).abs() / hh);
            if dx + dy <= 1.0 {
                inside += 1;
                if a.at(rect.x + x, rect.y + y) > threshold {
                    solid += 1;
                }
            }
        }
    }
    if inside == 0 { 0.0 } else { solid as f32 / inside as f32 }
}

/// Pivot given to new sprites by a slice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NewSpritePivot {
    /// Normalised, origin bottom-left.
    pub pivot: [f32; 2],
}

impl Default for NewSpritePivot {
    fn default() -> Self {
        Self { pivot: [0.5, 0.5] }
    }
}

/// What a slice did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SliceReport {
    /// Existing sprites that kept their identity.
    pub kept: usize,
    /// Indices (into the new table) of sprites created by the slice.
    pub created: Vec<usize>,
}

/// Cut `rects` into the Multiple table with `method` and give the new sprites
/// `pivot`. Ids, names and settings of inherited sprites are untouched.
pub fn slice_into(sheet: &mut SpriteSheet, stem: &str, rects: &[PixelRect], method: SliceMethod, pivot: NewSpritePivot) -> SliceReport {
    sheet.mode = crate::sprite::SpriteMode::Multiple;
    let (kept, inherited) = sheet.apply_slice_tracked(stem, rects, method);
    let mut created = Vec::new();
    for (i, inh) in inherited.iter().enumerate() {
        if !inh && let Some(d) = sheet.sprites.get_mut(i) {
            d.set_pivot(pivot.pivot);
            created.push(i);
        }
    }
    SliceReport { kept, created }
}

/// [`slice_into`] for isometric cells: new sprites also get the diamond as
/// their custom outline (so the mesh follows the tile, not the alpha).
pub fn slice_isometric(sheet: &mut SpriteSheet, stem: &str, cells: &[IsoCell], method: SliceMethod, pivot: NewSpritePivot) -> SliceReport {
    let rects: Vec<PixelRect> = cells.iter().map(|c| c.rect).collect();
    let report = slice_into(sheet, stem, &rects, method, pivot);
    for &i in &report.created {
        if let Some(c) = cells.iter().find(|c| c.rect == sheet.sprites[i].rect) {
            sheet.sprites[i].custom_outline = vec![c.outline.clone()];
        }
    }
    report
}

/// The slicing type chosen in the Sprite Editor's Slice panel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SliceType {
    #[default]
    Automatic,
    GridByCellSize,
    GridByCellCount,
    Isometric,
}

impl SliceType {
    pub const ALL: [Self; 4] = [Self::Automatic, Self::GridByCellSize, Self::GridByCellCount, Self::Isometric];

    pub fn label(self) -> &'static str {
        match self {
            Self::Automatic => "Automatic",
            Self::GridByCellSize => "Grid By Cell Size",
            Self::GridByCellCount => "Grid By Cell Count",
            Self::Isometric => "Isometric Grid",
        }
    }
}

/// Everything the Slice panel edits.
#[derive(Clone, Debug, PartialEq)]
pub struct SliceSettings {
    pub ty: SliceType,
    /// Automatic: smallest island side.
    pub min_size: u32,
    pub cell: [u32; 2],
    pub count: [u32; 2],
    pub offset: [u32; 2],
    pub padding: [u32; 2],
    pub keep_empty: bool,
    /// Isometric: which rows start shifted.
    pub alternate: bool,
    /// Pivot of the new sprites.
    pub pivot: PivotPreset,
    pub custom_pivot: [f32; 2],
    /// What to do with the existing sprites.
    pub method: SliceMethod,
}

impl Default for SliceSettings {
    fn default() -> Self {
        Self {
            ty: SliceType::Automatic,
            min_size: AUTO_MIN_SIZE,
            cell: [64, 64],
            count: [4, 4],
            offset: [0, 0],
            padding: [0, 0],
            keep_empty: false,
            alternate: false,
            pivot: PivotPreset::Center,
            custom_pivot: [0.5, 0.5],
            method: SliceMethod::Smart,
        }
    }
}

impl SliceSettings {
    pub fn new_pivot(&self) -> NewSpritePivot {
        NewSpritePivot { pivot: self.pivot.normalized().unwrap_or(self.custom_pivot) }
    }
}

/// The cut a settings block produces (nothing is applied yet: the editor draws
/// it as the red preview).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SlicePlan {
    pub rects: Vec<PixelRect>,
    /// Isometric only: the cells with their diamond outlines.
    pub iso: Vec<IsoCell>,
}

/// Compute the rects for `s` on a `w x h` image. `alpha` (with `threshold`)
/// drives Automatic and the empty-cell filter; without it grids keep every cell.
pub fn plan_slice(s: &SliceSettings, w: u32, h: u32, alpha: Option<&AlphaMap>, threshold: u8) -> SlicePlan {
    match s.ty {
        SliceType::Automatic => SlicePlan { rects: alpha.map(|a| auto_slice(a, threshold, s.min_size)).unwrap_or_default(), iso: Vec::new() },
        SliceType::GridByCellSize | SliceType::GridByCellCount => {
            let rects = if s.ty == SliceType::GridByCellSize {
                grid_rects(w, h, s.cell, s.offset, s.padding)
            } else {
                grid_rects_by_count(w, h, s.count[0], s.count[1], s.offset, s.padding)
            };
            let rects = match alpha {
                Some(a) if !s.keep_empty => drop_empty(rects, a, threshold),
                _ => rects,
            };
            SlicePlan { rects, iso: Vec::new() }
        }
        SliceType::Isometric => {
            let iso = isometric_cells(w, h, &IsoSettings { cell: s.cell, offset: s.offset, alternate: s.alternate }, if s.keep_empty { None } else { alpha.map(|a| (a, threshold)) });
            SlicePlan { rects: iso.iter().map(|c| c.rect).collect(), iso }
        }
    }
}

/// Apply a plan to the sheet with the settings' method and pivot.
pub fn apply_plan(sheet: &mut SpriteSheet, stem: &str, plan: &SlicePlan, s: &SliceSettings) -> SliceReport {
    if s.ty == SliceType::Isometric {
        slice_isometric(sheet, stem, &plan.iso, s.method, s.new_pivot())
    } else {
        slice_into(sheet, stem, &plan.rects, s.method, s.new_pivot())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sprite::{SpriteDef, SpriteMode};

    fn map(w: u32, h: u32, f: impl Fn(u32, u32) -> bool) -> AlphaMap {
        AlphaMap { w, h, alpha: (0..w * h).map(|i| if f(i % w, i / w) { 255 } else { 0 }).collect() }
    }

    fn block(x: u32, y: u32, w: u32, h: u32) -> impl Fn(u32, u32) -> bool {
        move |px, py| px >= x && px < x + w && py >= y && py < y + h
    }

    #[test]
    fn separate_islands_each_get_a_rect_in_reading_order() {
        let a = map(40, 30, |x, y| block(2, 2, 8, 8)(x, y) || block(20, 3, 6, 9)(x, y) || block(5, 18, 10, 6)(x, y));
        let r = auto_slice(&a, 0, 4);
        assert_eq!(r, vec![PixelRect::new(2, 2, 8, 8), PixelRect::new(20, 3, 6, 9), PixelRect::new(5, 18, 10, 6)]);
    }

    #[test]
    fn diagonal_touching_pixels_join_but_a_gap_separates() {
        // Two 5x5 blocks touching only at one corner: one island under 8-connectivity.
        let a = map(20, 10, |x, y| block(0, 0, 5, 5)(x, y) || block(5, 5, 5, 5)(x, y));
        assert_eq!(auto_slice(&a, 0, 4), vec![PixelRect::new(0, 0, 10, 10)]);
        // One pixel of gap: two islands.
        let a = map(20, 10, |x, y| block(0, 0, 5, 5)(x, y) || block(6, 6, 5, 4)(x, y));
        assert_eq!(auto_slice(&a, 0, 4).len(), 2);
    }

    #[test]
    fn holes_do_not_split_and_nested_islands_are_dropped() {
        // A ring (hole inside) with a small solid block floating in the hole.
        let ring = |x: u32, y: u32| block(0, 0, 20, 20)(x, y) && !block(4, 4, 12, 12)(x, y);
        let a = map(20, 20, |x, y| ring(x, y) || block(8, 8, 5, 5)(x, y));
        assert_eq!(auto_slice(&a, 0, 4), vec![PixelRect::new(0, 0, 20, 20)]);
        // The same ring alone is still one sprite.
        let a = map(20, 20, ring);
        assert_eq!(auto_slice(&a, 0, 4), vec![PixelRect::new(0, 0, 20, 20)]);
    }

    #[test]
    fn minimum_size_and_threshold() {
        let a = map(30, 10, |x, y| block(0, 0, 3, 8)(x, y) || block(10, 0, 8, 8)(x, y));
        assert_eq!(auto_slice(&a, 0, 4), vec![PixelRect::new(10, 0, 8, 8)], "the 3 wide island is below the minimum");
        assert_eq!(auto_slice(&a, 0, 1).len(), 2);
        // Alpha at or below the threshold is empty.
        let mut b = map(10, 10, block(0, 0, 10, 10));
        b.alpha.iter_mut().for_each(|v| *v = 100);
        assert!(auto_slice(&b, 100, 1).is_empty());
        assert_eq!(auto_slice(&b, 99, 1).len(), 1);
        assert!(auto_slice(&map(10, 10, |_, _| false), 0, 1).is_empty());
    }

    #[test]
    fn rows_group_by_height_not_by_exact_top() {
        // Three sprites on one row with slightly different tops, one below.
        let a = map(60, 40, |x, y| block(2, 3, 8, 8)(x, y) || block(20, 1, 8, 10)(x, y) || block(40, 5, 8, 6)(x, y) || block(2, 25, 8, 8)(x, y));
        let r = auto_slice(&a, 0, 4);
        assert_eq!(r.iter().map(|r| r.x).collect::<Vec<_>>(), vec![2, 20, 40, 2]);
    }

    #[test]
    fn isometric_geometry_rows_shift_and_alternate() {
        let s = IsoSettings { cell: [8, 4], offset: [0, 0], alternate: false };
        let c = isometric_cells(16, 8, &s, None);
        // Rows at y = 0, 2, 4 (y + 4 <= 8); odd rows are shifted by 4.
        let pos: Vec<(u32, u32)> = c.iter().map(|c| (c.rect.x, c.rect.y)).collect();
        assert_eq!(pos, vec![(0, 0), (8, 0), (4, 2), (0, 4), (8, 4)]);
        // Alternate flips which rows start shifted.
        let alt = isometric_cells(16, 8, &IsoSettings { alternate: true, ..s }, None);
        let pos: Vec<(u32, u32)> = alt.iter().map(|c| (c.rect.x, c.rect.y)).collect();
        assert_eq!(pos, vec![(4, 0), (0, 2), (8, 2), (4, 4)]);
        // Offsets move the whole grid and only whole cells count.
        let off = isometric_cells(16, 8, &IsoSettings { offset: [1, 1], ..s }, None);
        assert!(off.iter().all(|c| c.rect.x >= 1 && c.rect.y >= 1 && c.rect.x + c.rect.w <= 16 && c.rect.y + c.rect.h <= 8));
        // The outline is the diamond of the cell.
        assert_eq!(c[0].outline, vec![[4.0, 0.0], [8.0, 2.0], [4.0, 4.0], [0.0, 2.0]]);
        assert!(isometric_cells(16, 8, &IsoSettings { cell: [1, 1], ..s }, None).is_empty());
    }

    #[test]
    fn isometric_cells_without_enough_pixels_are_dropped() {
        // Opaque only in the top-left cell's diamond.
        let a = map(16, 8, block(0, 0, 8, 4));
        let s = IsoSettings { cell: [8, 4], offset: [0, 0], alternate: false };
        let c = isometric_cells(16, 8, &s, Some((&a, 0)));
        assert_eq!(c.len(), 2, "the cell at (0,0) and the half-shifted one overlapping it: {:?}", c.iter().map(|c| c.rect).collect::<Vec<_>>());
        assert!(c.iter().any(|c| c.rect == PixelRect::new(0, 0, 8, 4)));
        // Corners of the rect outside the diamond don't count: solid only in a corner.
        let corner = map(8, 4, block(0, 0, 1, 1));
        assert_eq!(diamond_coverage(&corner, PixelRect::new(0, 0, 8, 4), 0), 0.0);
        let full = map(8, 4, |_, _| true);
        assert!((diamond_coverage(&full, PixelRect::new(0, 0, 8, 4), 0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn slice_into_gives_new_sprites_the_pivot_and_keeps_inherited_ones() {
        let mut sheet = SpriteSheet { mode: SpriteMode::Multiple, ..Default::default() };
        sheet.sprites.push(SpriteDef::named("keep", PixelRect::new(0, 0, 16, 16)));
        sheet.sprites[0].set_pivot([0.1, 0.2]);
        sheet.sprites[0].border = [1, 1, 1, 1];
        sheet.ensure_ids("t");
        let id = sheet.sprites[0].id;
        let rects = [PixelRect::new(1, 1, 16, 16), PixelRect::new(32, 0, 16, 16)];
        let rep = slice_into(&mut sheet, "t", &rects, SliceMethod::Smart, NewSpritePivot { pivot: [0.5, 0.0] });
        assert_eq!(rep.kept, 1);
        assert_eq!(rep.created, vec![1]);
        assert_eq!(sheet.sprites[0].id, id);
        assert_eq!(sheet.sprites[0].pivot, [0.1, 0.2], "inherited pivot untouched");
        assert_eq!(sheet.sprites[0].border, [1, 1, 1, 1]);
        assert_eq!(sheet.sprites[1].pivot_norm(), [0.5, 0.0]);
        assert_eq!(sheet.sprites[1].pivot_preset, crate::sprite::PivotPreset::Bottom);
    }

    #[test]
    fn isometric_slice_sets_diamond_outlines_on_new_sprites() {
        let s = IsoSettings { cell: [32, 16], offset: [0, 0], alternate: false };
        let cells = isometric_cells(64, 32, &s, None);
        let mut sheet = SpriteSheet::default();
        let rep = slice_isometric(&mut sheet, "iso", &cells, SliceMethod::DeleteExisting, NewSpritePivot::default());
        assert_eq!(rep.created.len(), cells.len());
        assert_eq!(sheet.sprites.len(), cells.len());
        for d in &sheet.sprites {
            assert_eq!(d.custom_outline, vec![diamond(32, 16)]);
        }
        // Re-slicing Smart keeps ids and does not overwrite a hand-edited outline.
        let ids: Vec<_> = sheet.sprites.iter().map(|d| d.id).collect();
        sheet.sprites[0].custom_outline = vec![vec![[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]]];
        let rep = slice_isometric(&mut sheet, "iso", &cells, SliceMethod::Smart, NewSpritePivot::default());
        assert_eq!(rep.kept, cells.len());
        assert_eq!(sheet.sprites.iter().map(|d| d.id).collect::<Vec<_>>(), ids);
        assert_eq!(sheet.sprites[0].custom_outline[0].len(), 3);
    }

    #[test]
    fn grid_by_count_matches_the_cell_size_formula() {
        // 100 wide, 4 columns, 2 padding, offset 4: floor((100 - 4 - 2*3) / 4) = 22.
        assert_eq!(crate::sprite::cell_size_from_count(100, 4, 4, 2), 22);
        let r = crate::sprite::grid_rects_by_count(100, 50, 4, 2, [4, 0], [2, 0]);
        assert_eq!(r.len(), 8);
        assert_eq!(r[0], PixelRect::new(4, 0, 22, 25));
        assert_eq!(r[1].x, 4 + 22 + 2);
        assert!(r.iter().all(|c| c.fits(100, 50)));
    }

    #[test]
    fn plan_slice_covers_every_type_and_the_empty_cell_filter() {
        let a = map(64, 32, block(0, 0, 32, 32));
        let mut s = SliceSettings { ty: SliceType::GridByCellSize, cell: [32, 32], ..SliceSettings::default() };
        let p = plan_slice(&s, 64, 32, Some(&a), 0);
        assert_eq!(p.rects, vec![PixelRect::new(0, 0, 32, 32)], "the empty right cell is dropped");
        s.keep_empty = true;
        assert_eq!(plan_slice(&s, 64, 32, Some(&a), 0).rects.len(), 2);
        s.ty = SliceType::GridByCellCount;
        s.count = [4, 2];
        assert_eq!(plan_slice(&s, 64, 32, None, 0).rects.len(), 8);
        s.ty = SliceType::Automatic;
        assert_eq!(plan_slice(&s, 64, 32, Some(&a), 0).rects, vec![PixelRect::new(0, 0, 32, 32)]);
        assert!(plan_slice(&s, 64, 32, None, 0).rects.is_empty(), "automatic needs pixels");
        s.ty = SliceType::Isometric;
        s.cell = [32, 16];
        let p = plan_slice(&s, 64, 32, None, 0);
        assert_eq!(p.rects.len(), p.iso.len());
        assert!(!p.iso.is_empty());
        // Applying: pivot preset reaches the new sprites.
        s.pivot = PivotPreset::Bottom;
        let mut sheet = SpriteSheet::default();
        let rep = apply_plan(&mut sheet, "iso", &p, &s);
        assert_eq!(rep.created.len(), p.rects.len());
        assert!(sheet.sprites.iter().all(|d| d.pivot_norm() == [0.5, 0.0] && d.custom_outline.len() == 1));
        s.ty = SliceType::GridByCellSize;
        s.cell = [16, 16];
        s.pivot = PivotPreset::Custom;
        s.custom_pivot = [0.25, 0.75];
        s.method = SliceMethod::DeleteExisting;
        let p = plan_slice(&s, 64, 32, None, 0);
        apply_plan(&mut sheet, "g", &p, &s);
        assert_eq!(sheet.sprites.len(), 8);
        assert_eq!(sheet.sprites[0].pivot, [0.25, 0.75]);
        assert_eq!(SliceType::ALL.map(|t| t.label()).len(), 4);
    }
}
