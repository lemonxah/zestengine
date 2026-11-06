//! Editing logic of the Sprite Editor window (2D-4), kept free of egui so it can
//! be unit tested: the view transform, rect handles (move / resize / create with
//! clamping and snapping), selection, 9-slice border and pivot dragging, custom
//! outline / physics-shape point editing, and the undo stack.
//!
//! The window (`zestengine-engine/src/sprite_editor.rs`) only turns pointer
//! events into calls here and draws the result. Undo is whole-sheet
//! snapshotting like Unity's (the sheet is small, the logic trivial to trust):
//! call [`SpriteEdit::checkpoint`] **before** an edit (once at the start of a
//! drag), then [`SpriteEdit::undo`] / [`SpriteEdit::redo`].
//!
//! Coordinates: image pixels, origin top-left, y down; outline points are
//! rect-local; the pivot is normalised with the origin bottom-left (see
//! `sprite.rs`).

use crate::sprite::{PivotPreset, PixelRect, SpriteDef, SpriteSheet};
use crate::sprite_mesh::{self, AlphaMap, P2};

// ---------------------------------------------------------------------------
// View
// ---------------------------------------------------------------------------

pub const MIN_ZOOM: f32 = 0.05;
/// Research: zoom up to 50x.
pub const MAX_ZOOM: f32 = 50.0;

/// Image <-> canvas mapping: `canvas = pan + image * zoom` (canvas coordinates
/// are relative to the canvas' top-left corner).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    pub zoom: f32,
    pub pan: [f32; 2],
}

impl Default for View {
    fn default() -> Self {
        Self { zoom: 1.0, pan: [0.0, 0.0] }
    }
}

impl View {
    /// Fit the image into `avail` (canvas pixels) at 90% and centre it.
    pub fn fit(avail: [f32; 2], img: [u32; 2]) -> View {
        let (iw, ih) = (img[0].max(1) as f32, img[1].max(1) as f32);
        let zoom = ((avail[0] / iw).min(avail[1] / ih) * 0.9).clamp(MIN_ZOOM, MAX_ZOOM);
        View { zoom, pan: [(avail[0] - iw * zoom) / 2.0, (avail[1] - ih * zoom) / 2.0] }
    }

    pub fn to_canvas(&self, p: P2) -> P2 {
        [self.pan[0] + p[0] * self.zoom, self.pan[1] + p[1] * self.zoom]
    }

    pub fn to_image(&self, c: P2) -> P2 {
        [(c[0] - self.pan[0]) / self.zoom, (c[1] - self.pan[1]) / self.zoom]
    }

    /// Multiply the zoom by `factor`, keeping the image point under `cursor`
    /// (canvas coordinates) where it is.
    pub fn zoom_at(&mut self, cursor: P2, factor: f32) {
        let before = self.to_image(cursor);
        self.zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        self.pan = [cursor[0] - before[0] * self.zoom, cursor[1] - before[1] * self.zoom];
    }

    pub fn pan_by(&mut self, d: P2) {
        self.pan = [self.pan[0] + d[0], self.pan[1] + d[1]];
    }
}

// ---------------------------------------------------------------------------
// Rect handles
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Handle {
    TopLeft,
    Top,
    TopRight,
    Left,
    Right,
    BottomLeft,
    Bottom,
    BottomRight,
}

impl Handle {
    pub const ALL: [Handle; 8] = [Self::TopLeft, Self::Top, Self::TopRight, Self::Left, Self::Right, Self::BottomLeft, Self::Bottom, Self::BottomRight];

    /// Position of the handle on a rect, as fractions (0..1, y down).
    pub fn frac(self) -> [f32; 2] {
        match self {
            Self::TopLeft => [0.0, 0.0],
            Self::Top => [0.5, 0.0],
            Self::TopRight => [1.0, 0.0],
            Self::Left => [0.0, 0.5],
            Self::Right => [1.0, 0.5],
            Self::BottomLeft => [0.0, 1.0],
            Self::Bottom => [0.5, 1.0],
            Self::BottomRight => [1.0, 1.0],
        }
    }

    fn left(self) -> bool {
        matches!(self, Self::TopLeft | Self::Left | Self::BottomLeft)
    }
    fn right(self) -> bool {
        matches!(self, Self::TopRight | Self::Right | Self::BottomRight)
    }
    fn top(self) -> bool {
        matches!(self, Self::TopLeft | Self::Top | Self::TopRight)
    }
    fn bottom(self) -> bool {
        matches!(self, Self::BottomLeft | Self::Bottom | Self::BottomRight)
    }
}

/// Round `v` to the nearest multiple of `grid` (0 or 1 = whole pixels).
pub fn snap(v: f32, grid: u32) -> i32 {
    let g = grid.max(1) as f32;
    ((v / g).round() * g) as i32
}

/// Move `rect` by (dx, dy), keeping it inside the `iw x ih` image.
pub fn move_rect(rect: PixelRect, dx: i32, dy: i32, iw: u32, ih: u32) -> PixelRect {
    let x = (rect.x as i64 + dx as i64).clamp(0, iw.saturating_sub(rect.w) as i64) as u32;
    let y = (rect.y as i64 + dy as i64).clamp(0, ih.saturating_sub(rect.h) as i64) as u32;
    PixelRect { x, y, ..rect }
}

/// The common delta for moving `rects` as one block by (dx, dy): shrunk so no
/// member leaves the image.
pub fn clamp_group_move(rects: &[PixelRect], dx: i32, dy: i32, iw: u32, ih: u32) -> (i32, i32) {
    let (mut lo_x, mut hi_x, mut lo_y, mut hi_y) = (i64::MIN, i64::MAX, i64::MIN, i64::MAX);
    for r in rects {
        lo_x = lo_x.max(-(r.x as i64));
        hi_x = hi_x.min(iw as i64 - (r.x + r.w) as i64);
        lo_y = lo_y.max(-(r.y as i64));
        hi_y = hi_y.min(ih as i64 - (r.y + r.h) as i64);
    }
    if rects.is_empty() {
        return (0, 0);
    }
    ((dx as i64).clamp(lo_x, hi_x.max(lo_x)) as i32, (dy as i64).clamp(lo_y, hi_y.max(lo_y)) as i32)
}

/// Drag `handle` of `rect` to image point `to`: the edges the handle owns move
/// to `to` (clamped to the image and leaving at least `min` pixels), the
/// opposite edges stay put. A drag past the opposite edge stops at `min`
/// instead of flipping.
pub fn resize_rect(rect: PixelRect, handle: Handle, to: [i32; 2], iw: u32, ih: u32, min: u32) -> PixelRect {
    let min = min.max(1) as i64;
    let (mut l, mut t, mut r, mut b) = (rect.x as i64, rect.y as i64, (rect.x + rect.w) as i64, (rect.y + rect.h) as i64);
    if handle.left() {
        l = (to[0] as i64).clamp(0, (r - min).max(0));
    }
    if handle.right() {
        r = (to[0] as i64).clamp((l + min).min(iw as i64), iw as i64);
    }
    if handle.top() {
        t = (to[1] as i64).clamp(0, (b - min).max(0));
    }
    if handle.bottom() {
        b = (to[1] as i64).clamp((t + min).min(ih as i64), ih as i64);
    }
    PixelRect::new(l as u32, t as u32, (r - l).max(1) as u32, (b - t).max(1) as u32)
}

/// The rect spanned by a drag from `a` to `b` (any direction), clamped to the
/// image. `None` when it is empty.
pub fn rect_from_points(a: [i32; 2], b: [i32; 2], iw: u32, ih: u32) -> Option<PixelRect> {
    let (x0, x1) = (a[0].min(b[0]).clamp(0, iw as i32), a[0].max(b[0]).clamp(0, iw as i32));
    let (y0, y1) = (a[1].min(b[1]).clamp(0, ih as i32), a[1].max(b[1]).clamp(0, ih as i32));
    (x1 > x0 && y1 > y0).then(|| PixelRect::new(x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32))
}

/// The handle of `rect` under a canvas point (`radius` canvas pixels), tried
/// corners first so a small rect still offers its corners.
pub fn hit_handle(rect: PixelRect, view: &View, canvas_pt: P2, radius: f32) -> Option<Handle> {
    let order = [
        Handle::TopLeft,
        Handle::TopRight,
        Handle::BottomLeft,
        Handle::BottomRight,
        Handle::Top,
        Handle::Bottom,
        Handle::Left,
        Handle::Right,
    ];
    order.into_iter().find(|h| {
        let f = h.frac();
        let c = view.to_canvas([rect.x as f32 + f[0] * rect.w as f32, rect.y as f32 + f[1] * rect.h as f32]);
        (c[0] - canvas_pt[0]).powi(2) + (c[1] - canvas_pt[1]).powi(2) <= radius * radius
    })
}

/// The sprite under an image point: the smallest rect containing it (so a small
/// sprite on top of a big one is still pickable).
pub fn pick_sprite(sprites: &[SpriteDef], p: P2) -> Option<usize> {
    sprites
        .iter()
        .enumerate()
        .filter(|(_, s)| p[0] >= s.rect.x as f32 && p[0] < (s.rect.x + s.rect.w) as f32 && p[1] >= s.rect.y as f32 && p[1] < (s.rect.y + s.rect.h) as f32)
        .min_by_key(|(i, s)| (s.rect.area(), std::cmp::Reverse(*i)))
        .map(|(i, _)| i)
}

/// Sprites whose rect touches the box (marquee selection).
pub fn sprites_in_box(sprites: &[SpriteDef], a: P2, b: P2) -> Vec<usize> {
    let (x0, x1, y0, y1) = (a[0].min(b[0]), a[0].max(b[0]), a[1].min(b[1]), a[1].max(b[1]));
    sprites
        .iter()
        .enumerate()
        .filter(|(_, s)| (s.rect.x + s.rect.w) as f32 > x0 && (s.rect.x as f32) < x1 && (s.rect.y + s.rect.h) as f32 > y0 && (s.rect.y as f32) < y1)
        .map(|(i, _)| i)
        .collect()
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

/// Selected sprite indices (sorted, unique); the first is the "primary" shown in
/// the side panel.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Selection {
    pub items: Vec<usize>,
}

impl Selection {
    pub fn single(i: usize) -> Self {
        Self { items: vec![i] }
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }

    pub fn contains(&self, i: usize) -> bool {
        self.items.contains(&i)
    }

    pub fn primary(&self) -> Option<usize> {
        self.items.first().copied()
    }

    /// Plain click selects only `i`; with `additive` (Shift / Ctrl) it toggles.
    pub fn click(&mut self, i: usize, additive: bool) {
        if !additive {
            self.items = vec![i];
        } else if let Some(p) = self.items.iter().position(|&k| k == i) {
            self.items.remove(p);
        } else {
            self.items.push(i);
            self.items.sort_unstable();
        }
    }

    pub fn set(&mut self, mut items: Vec<usize>, additive: bool) {
        if additive {
            items.extend(self.items.iter().copied());
        }
        items.sort_unstable();
        items.dedup();
        self.items = items;
    }

    pub fn select_all(&mut self, n: usize) {
        self.items = (0..n).collect();
    }

    /// Drop indices >= `n` (after the table shrank).
    pub fn retain_below(&mut self, n: usize) {
        self.items.retain(|&i| i < n);
    }
}

// ---------------------------------------------------------------------------
// Undo
// ---------------------------------------------------------------------------

/// Snapshot undo / redo of any cloneable state.
#[derive(Clone, Debug)]
pub struct UndoStack<T: Clone> {
    undo: Vec<T>,
    redo: Vec<T>,
    cap: usize,
}

impl<T: Clone> UndoStack<T> {
    pub fn new(cap: usize) -> Self {
        Self { undo: Vec::new(), redo: Vec::new(), cap: cap.max(1) }
    }

    /// Remember `before` (the state about to change). Clears redo.
    pub fn push(&mut self, before: T) {
        self.undo.push(before);
        if self.undo.len() > self.cap {
            self.undo.remove(0);
        }
        self.redo.clear();
    }

    /// Step back: returns the state to restore, remembering `current` for redo.
    pub fn undo(&mut self, current: &T) -> Option<T> {
        let prev = self.undo.pop()?;
        self.redo.push(current.clone());
        Some(prev)
    }

    pub fn redo(&mut self, current: &T) -> Option<T> {
        let next = self.redo.pop()?;
        self.undo.push(current.clone());
        Some(next)
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }
}

/// The editor session over one sheet: selection, undo and the baseline the
/// Revert button returns to (the sheet as last applied).
#[derive(Clone, Debug)]
pub struct SpriteEdit {
    pub sel: Selection,
    undo: UndoStack<SpriteSheet>,
    baseline: SpriteSheet,
}

impl SpriteEdit {
    pub fn new(sheet: &SpriteSheet) -> Self {
        Self { sel: Selection::default(), undo: UndoStack::new(100), baseline: sheet.clone() }
    }

    /// Call before changing `sheet` (once per drag / click / command).
    pub fn checkpoint(&mut self, sheet: &SpriteSheet) {
        self.undo.push(sheet.clone());
    }

    pub fn undo(&mut self, sheet: &mut SpriteSheet) -> bool {
        match self.undo.undo(sheet) {
            Some(prev) => {
                *sheet = prev;
                self.sel.retain_below(sheet.sprites.len());
                true
            }
            None => false,
        }
    }

    pub fn redo(&mut self, sheet: &mut SpriteSheet) -> bool {
        match self.undo.redo(sheet) {
            Some(next) => {
                *sheet = next;
                self.sel.retain_below(sheet.sprites.len());
                true
            }
            None => false,
        }
    }

    pub fn can_undo(&self) -> bool {
        self.undo.can_undo()
    }

    pub fn can_redo(&self) -> bool {
        self.undo.can_redo()
    }

    /// The sheet differs from the last applied one.
    pub fn is_dirty(&self, sheet: &SpriteSheet) -> bool {
        *sheet != self.baseline
    }

    /// Go back to the last applied sheet (undoable).
    pub fn revert(&mut self, sheet: &mut SpriteSheet) {
        if *sheet != self.baseline {
            self.checkpoint(sheet);
            *sheet = self.baseline.clone();
            self.sel.retain_below(sheet.sprites.len());
        }
    }

    /// The sheet was applied: it becomes the baseline and history is dropped
    /// (as in Unity, where Apply clears the undo history).
    pub fn mark_applied(&mut self, sheet: &SpriteSheet) {
        self.baseline = sheet.clone();
        self.undo.clear();
    }

    /// Adopt a sheet changed from outside (the inspector reverted, say): the new
    /// baseline, no history.
    pub fn rebase(&mut self, sheet: &SpriteSheet) {
        self.mark_applied(sheet);
        self.sel.retain_below(sheet.sprites.len());
    }
}

// ---------------------------------------------------------------------------
// Sheet operations
// ---------------------------------------------------------------------------

/// Move the selected sprites as a block.
pub fn move_selected(sheet: &mut SpriteSheet, sel: &Selection, dx: i32, dy: i32, iw: u32, ih: u32) {
    let rects: Vec<PixelRect> = sel.items.iter().filter_map(|&i| sheet.sprites.get(i).map(|s| s.rect)).collect();
    let (dx, dy) = clamp_group_move(&rects, dx, dy, iw, ih);
    for &i in &sel.items {
        if let Some(s) = sheet.sprites.get_mut(i) {
            s.rect = move_rect(s.rect, dx, dy, iw, ih);
        }
    }
}

/// Resize one sprite by a handle drag (see [`resize_rect`]). Custom outlines are
/// rect-local, so points that would fall outside the new rect are clamped; the
/// border is re-clamped.
pub fn resize_sprite(sheet: &mut SpriteSheet, i: usize, handle: Handle, to: [i32; 2], iw: u32, ih: u32) {
    let Some(s) = sheet.sprites.get_mut(i) else { return };
    let old = s.rect;
    s.rect = resize_rect(old, handle, to, iw, ih, 1);
    let (dx, dy) = (old.x as f32 - s.rect.x as f32, old.y as f32 - s.rect.y as f32);
    let (w, h) = (s.rect.w as f32, s.rect.h as f32);
    for loops in [&mut s.custom_outline, &mut s.physics_shape] {
        for l in loops.iter_mut() {
            for p in l.iter_mut() {
                // Keep the points where they were on the image, then clamp into the rect.
                *p = [(p[0] + dx).clamp(0.0, w), (p[1] + dy).clamp(0.0, h)];
            }
        }
    }
    s.border = s.border_clamped();
}

/// Delete the selected sprites (ids stay reserved in the id table). Clears the
/// selection.
pub fn delete_selected(sheet: &mut SpriteSheet, sel: &mut Selection) -> usize {
    let mut idx = sel.items.clone();
    idx.sort_unstable_by(|a, b| b.cmp(a));
    let mut n = 0;
    for i in idx {
        if i < sheet.sprites.len() {
            sheet.remove_sprite(i);
            n += 1;
        }
    }
    sel.clear();
    n
}

/// Create a sprite for a drag-created rect and select it.
pub fn create_sprite(sheet: &mut SpriteSheet, stem: &str, rect: PixelRect, sel: &mut Selection) -> usize {
    sheet.mode = crate::sprite::SpriteMode::Multiple;
    let i = sheet.add_sprite(stem, rect);
    *sel = Selection::single(i);
    i
}

/// Duplicate the selected sprites, offset by (8, 8) inside the image, with fresh
/// names and ids.
pub fn duplicate_selected(sheet: &mut SpriteSheet, stem: &str, sel: &mut Selection, iw: u32, ih: u32) {
    let sources: Vec<SpriteDef> = sel.items.iter().filter_map(|&i| sheet.sprites.get(i).cloned()).collect();
    let mut made = Vec::new();
    for s in sources {
        let i = sheet.add_sprite(stem, move_rect(s.rect, 8, 8, iw, ih));
        let d = &mut sheet.sprites[i];
        let (name, id, rect) = (d.name.clone(), d.id, d.rect);
        *d = s;
        d.name = name;
        d.id = id;
        d.rect = rect;
        made.push(i);
    }
    sel.set(made, false);
}

/// Set the pivot of every selected sprite from a preset.
pub fn set_pivot_preset(sheet: &mut SpriteSheet, sel: &Selection, preset: PivotPreset) {
    let Some(n) = preset.normalized() else { return };
    for &i in &sel.items {
        if let Some(s) = sheet.sprites.get_mut(i) {
            s.pivot = n;
            s.pivot_preset = preset;
        }
    }
}

/// Set a custom normalised pivot on every selected sprite.
pub fn set_pivot_all(sheet: &mut SpriteSheet, sel: &Selection, pivot: [f32; 2]) {
    for &i in &sel.items {
        if let Some(s) = sheet.sprites.get_mut(i) {
            s.set_pivot(pivot);
        }
    }
}

/// Normalised pivot for a pivot-handle drag to rect-local pixel `local`
/// (y down). `pixel_snap` rounds to whole pixels; `snap_points` snaps to the
/// nearest of the nine preset points when within `tol` pixels.
pub fn pivot_from_drag(rect: PixelRect, local: P2, pixel_snap: bool, snap_points: Option<f32>) -> [f32; 2] {
    let (w, h) = (rect.w.max(1) as f32, rect.h.max(1) as f32);
    let (mut lx, mut ly) = (local[0], local[1]);
    if pixel_snap {
        lx = lx.round();
        ly = ly.round();
    }
    let mut p = [lx / w, 1.0 - ly / h];
    if let Some(tol) = snap_points {
        let best = PivotPreset::ALL
            .iter()
            .filter_map(|k| k.normalized())
            .map(|n| (n, ((n[0] - p[0]) * w).powi(2) + ((n[1] - p[1]) * h).powi(2)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        if let Some((n, d2)) = best
            && d2 <= tol * tol
        {
            p = n;
        }
    }
    p
}

/// A side of the 9-slice border.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BorderEdge {
    Left,
    Bottom,
    Right,
    Top,
}

impl BorderEdge {
    pub const ALL: [BorderEdge; 4] = [Self::Left, Self::Bottom, Self::Right, Self::Top];

    fn index(self) -> usize {
        self as usize
    }
}

/// The border after dragging `edge` to rect-local pixel `local` (y down): whole
/// pixels, never negative, and the opposite side limits it (L + R <= w,
/// B + T <= h).
pub fn border_from_drag(d: &SpriteDef, edge: BorderEdge, local: P2) -> [u32; 4] {
    let (w, h) = (d.rect.w as f32, d.rect.h as f32);
    let mut b = d.border_clamped();
    let v = match edge {
        BorderEdge::Left => local[0],
        BorderEdge::Right => w - local[0],
        BorderEdge::Top => local[1],
        BorderEdge::Bottom => h - local[1],
    };
    let limit = match edge {
        BorderEdge::Left => d.rect.w - b[2],
        BorderEdge::Right => d.rect.w - b[0],
        BorderEdge::Top => d.rect.h - b[1],
        BorderEdge::Bottom => d.rect.h - b[3],
    };
    b[edge.index()] = (v.round().max(0.0) as u32).min(limit);
    b
}

/// Rect-local canvas lines of the border for drawing: for each edge, the
/// segment (a, b) in rect-local pixels.
pub fn border_lines(d: &SpriteDef) -> [(BorderEdge, [P2; 2]); 4] {
    let [l, b, r, t] = d.border_clamped();
    let (w, h) = (d.rect.w as f32, d.rect.h as f32);
    [
        (BorderEdge::Left, [[l as f32, 0.0], [l as f32, h]]),
        (BorderEdge::Bottom, [[0.0, h - b as f32], [w, h - b as f32]]),
        (BorderEdge::Right, [[w - r as f32, 0.0], [w - r as f32, h]]),
        (BorderEdge::Top, [[0.0, t as f32], [w, t as f32]]),
    ]
}

// ---------------------------------------------------------------------------
// Outline / physics shape editing
// ---------------------------------------------------------------------------

/// Which hand-drawn shape of a sprite is edited.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeKind {
    /// The render outline (`custom_outline`).
    Outline,
    /// The physics shape (`physics_shape`).
    Physics,
}

pub fn shape(d: &SpriteDef, k: ShapeKind) -> &Vec<Vec<P2>> {
    match k {
        ShapeKind::Outline => &d.custom_outline,
        ShapeKind::Physics => &d.physics_shape,
    }
}

pub fn shape_mut(d: &mut SpriteDef, k: ShapeKind) -> &mut Vec<Vec<P2>> {
    match k {
        ShapeKind::Outline => &mut d.custom_outline,
        ShapeKind::Physics => &mut d.physics_shape,
    }
}

/// (loop index, point index).
pub type PointId = (usize, usize);

/// The point nearest to `p` within `max_d`.
pub fn nearest_point(loops: &[Vec<P2>], p: P2, max_d: f32) -> Option<PointId> {
    let mut best: Option<(f32, PointId)> = None;
    for (li, l) in loops.iter().enumerate() {
        for (pi, q) in l.iter().enumerate() {
            let d = ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)).sqrt();
            if d <= max_d && best.is_none_or(|b| d < b.0) {
                best = Some((d, (li, pi)));
            }
        }
    }
    best.map(|b| b.1)
}

fn snap_pt(p: P2, size: [u32; 2], pixel_snap: bool) -> P2 {
    let q = if pixel_snap { [p[0].round(), p[1].round()] } else { p };
    [q[0].clamp(0.0, size[0] as f32), q[1].clamp(0.0, size[1] as f32)]
}

/// Insert a point on the edge nearest to `p` (within `max_d`), at the projected
/// position, and return its id.
pub fn insert_point_on_edge(loops: &mut [Vec<P2>], p: P2, max_d: f32, size: [u32; 2], pixel_snap: bool) -> Option<PointId> {
    let mut best: Option<(f32, usize, usize, P2)> = None;
    for (li, l) in loops.iter().enumerate() {
        for i in 0..l.len() {
            let (a, b) = (l[i], l[(i + 1) % l.len()]);
            let ab = [b[0] - a[0], b[1] - a[1]];
            let len2 = ab[0] * ab[0] + ab[1] * ab[1];
            if len2 <= 0.0 {
                continue;
            }
            let t = (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / len2).clamp(0.0, 1.0);
            let q = [a[0] + ab[0] * t, a[1] + ab[1] * t];
            let d = ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)).sqrt();
            if d <= max_d && best.is_none_or(|b| d < b.0) {
                best = Some((d, li, i, q));
            }
        }
    }
    let (_, li, i, q) = best?;
    loops[li].insert(i + 1, snap_pt(q, size, pixel_snap));
    Some((li, i + 1))
}

/// Move a point, clamped into the rect (`size`) and snapped to whole pixels.
pub fn move_point(loops: &mut [Vec<P2>], id: PointId, to: P2, size: [u32; 2], pixel_snap: bool) {
    if let Some(q) = loops.get_mut(id.0).and_then(|l| l.get_mut(id.1)) {
        *q = snap_pt(to, size, pixel_snap);
    }
}

/// Move a set of points by a delta (Ctrl+drag on an edge moves both ends).
pub fn move_points(loops: &mut [Vec<P2>], ids: &[PointId], d: P2, size: [u32; 2], pixel_snap: bool) {
    for &(li, pi) in ids {
        if let Some(q) = loops.get_mut(li).and_then(|l| l.get_mut(pi)) {
            *q = snap_pt([q[0] + d[0], q[1] + d[1]], size, pixel_snap);
        }
    }
}

/// Delete points; loops left with fewer than 3 points disappear. Returns how
/// many points were removed.
pub fn delete_points(loops: &mut Vec<Vec<P2>>, ids: &[PointId]) -> usize {
    let mut sorted: Vec<PointId> = ids.to_vec();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    sorted.dedup();
    let mut n = 0;
    for (li, pi) in sorted {
        if let Some(l) = loops.get_mut(li)
            && pi < l.len()
        {
            l.remove(pi);
            n += 1;
        }
    }
    loops.retain(|l| l.len() >= 3);
    n
}

/// A new rectangular loop for a drag inside the sprite (shape editing: "dragging
/// inside the sprite with nothing selected creates a new quad path").
pub fn new_quad(loops: &mut Vec<Vec<P2>>, a: P2, b: P2, size: [u32; 2], pixel_snap: bool) -> bool {
    let (a, b) = (snap_pt(a, size, pixel_snap), snap_pt(b, size, pixel_snap));
    let (x0, x1, y0, y1) = (a[0].min(b[0]), a[0].max(b[0]), a[1].min(b[1]), a[1].max(b[1]));
    if x1 - x0 < 1.0 || y1 - y0 < 1.0 {
        return false;
    }
    loops.push(vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
    true
}

/// Points inside the box (rect selection of points).
pub fn points_in_box(loops: &[Vec<P2>], a: P2, b: P2) -> Vec<PointId> {
    let (x0, x1, y0, y1) = (a[0].min(b[0]), a[0].max(b[0]), a[1].min(b[1]), a[1].max(b[1]));
    let mut out = Vec::new();
    for (li, l) in loops.iter().enumerate() {
        for (pi, q) in l.iter().enumerate() {
            if q[0] >= x0 && q[0] <= x1 && q[1] >= y0 && q[1] <= y1 {
                out.push((li, pi));
            }
        }
    }
    out
}

/// Trace the alpha outline of the sprite at `rect` (rect-local points):
/// `detail` 0..1 like the inspector's slider.
pub fn generate_outline(alpha: &AlphaMap, rect: PixelRect, threshold: u8, detail: f32) -> Vec<Vec<P2>> {
    sprite_mesh::trace_outline(alpha, rect, threshold, 0, sprite_mesh::detail_to_tolerance(detail), true)
}

/// The physics-shape generator (finer tolerance range, editor default detail
/// 0.85 and alpha tolerance 128).
pub fn generate_physics(alpha: &AlphaMap, rect: PixelRect, threshold: u8, detail: f32) -> Vec<Vec<P2>> {
    sprite_mesh::physics_outline(alpha, rect, threshold, sprite_mesh::physics_tolerance(detail))
}

/// Convex pieces of a physics shape (what a convex-only physics engine gets).
pub fn convex_pieces_of(loops: &[Vec<P2>]) -> Vec<Vec<P2>> {
    sprite_mesh::mesh_from_loops(loops, true).map(|m| sprite_mesh::convex_pieces(&m)).unwrap_or_default()
}

/// Problems with a hand-drawn shape (shown next to the editor).
pub fn shape_problems(loops: &[Vec<P2>]) -> Vec<String> {
    let mut out = Vec::new();
    if loops.iter().any(|l| l.len() < 3) {
        out.push("a path has fewer than 3 points".to_owned());
    }
    if !sprite_mesh::loops_are_simple(loops) {
        out.push("paths cross each other".to_owned());
    }
    out
}

/// Copy a sprite's outline into its physics shape or back (Unity lets you paste
/// between the two modules).
pub fn copy_shape(d: &mut SpriteDef, from: ShapeKind, to: ShapeKind) {
    let src = shape(d, from).clone();
    *shape_mut(d, to) = src;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sprite::SliceMethod;
    use crate::sprite::SpriteMode;

    fn sheet(rects: &[PixelRect]) -> SpriteSheet {
        let mut s = SpriteSheet { mode: SpriteMode::Multiple, ..SpriteSheet::default() };
        for (i, r) in rects.iter().enumerate() {
            s.sprites.push(SpriteDef::named(&format!("s{i}"), *r));
        }
        s.ensure_ids("t");
        s
    }

    #[test]
    fn view_zoom_keeps_the_point_under_the_cursor_and_clamps() {
        let mut v = View { zoom: 2.0, pan: [10.0, 20.0] };
        let cursor = [100.0, 80.0];
        let before = v.to_image(cursor);
        v.zoom_at(cursor, 1.5);
        let after = v.to_image(cursor);
        assert!((before[0] - after[0]).abs() < 1e-4 && (before[1] - after[1]).abs() < 1e-4);
        v.zoom_at(cursor, 1e6);
        assert_eq!(v.zoom, MAX_ZOOM);
        v.zoom_at(cursor, 1e-9);
        assert_eq!(v.zoom, MIN_ZOOM);
        let f = View::fit([400.0, 200.0], [100, 100]);
        assert!((f.zoom - 1.8).abs() < 1e-4, "fit is 90% of the limiting axis: {}", f.zoom);
        let c = f.to_canvas([50.0, 50.0]);
        assert!((c[0] - 200.0).abs() < 1e-3 && (c[1] - 100.0).abs() < 1e-3, "centred");
        let back = f.to_image(f.to_canvas([7.0, 9.0]));
        assert!((back[0] - 7.0).abs() < 1e-4 && (back[1] - 9.0).abs() < 1e-4);
    }

    #[test]
    fn move_clamps_to_the_image_and_groups_move_rigidly() {
        let r = PixelRect::new(10, 10, 20, 20);
        assert_eq!(move_rect(r, -50, -50, 100, 100), PixelRect::new(0, 0, 20, 20));
        assert_eq!(move_rect(r, 500, 500, 100, 100), PixelRect::new(80, 80, 20, 20));
        assert_eq!(move_rect(r, 5, -3, 100, 100), PixelRect::new(15, 7, 20, 20));
        // A group stops when its first member hits the wall; relative offsets survive.
        let g = [PixelRect::new(0, 0, 10, 10), PixelRect::new(50, 5, 10, 10)];
        assert_eq!(clamp_group_move(&g, -20, 0, 100, 100), (0, 0));
        assert_eq!(clamp_group_move(&g, 100, 100, 100, 100), (40, 85));
        assert_eq!(clamp_group_move(&[], 5, 5, 10, 10), (0, 0));
        let mut s = sheet(&g);
        let sel = Selection { items: vec![0, 1] };
        move_selected(&mut s, &sel, 100, 0, 100, 100);
        assert_eq!(s.sprites[0].rect.x, 40);
        assert_eq!(s.sprites[1].rect.x, 90);
    }

    #[test]
    fn resize_moves_only_the_handle_edges_and_never_flips() {
        let r = PixelRect::new(20, 20, 40, 30);
        assert_eq!(resize_rect(r, Handle::Right, [80, 0], 100, 100, 1), PixelRect::new(20, 20, 60, 30));
        assert_eq!(resize_rect(r, Handle::Left, [10, 999], 100, 100, 1), PixelRect::new(10, 20, 50, 30));
        assert_eq!(resize_rect(r, Handle::TopLeft, [25, 30], 100, 100, 1), PixelRect::new(25, 30, 35, 20));
        assert_eq!(resize_rect(r, Handle::BottomRight, [200, 200], 100, 100, 1), PixelRect::new(20, 20, 80, 80), "clamped to the image");
        // Dragging the right edge past the left stops at the minimum width.
        assert_eq!(resize_rect(r, Handle::Right, [0, 0], 100, 100, 4), PixelRect::new(20, 20, 4, 30));
        assert_eq!(resize_rect(r, Handle::Top, [0, 500], 100, 100, 1), PixelRect::new(20, 49, 40, 1));
        assert_eq!(resize_rect(r, Handle::Left, [-30, 0], 100, 100, 1).x, 0);
        // Every handle keeps the rect inside the image.
        for h in Handle::ALL {
            for to in [[-10, -10], [0, 0], [50, 50], [200, 200]] {
                let n = resize_rect(r, h, to, 100, 100, 1);
                assert!(n.fits(100, 100), "{h:?} {to:?} {n:?}");
            }
        }
    }

    #[test]
    fn resizing_a_sprite_keeps_outline_points_on_the_image_and_clamps_borders() {
        let mut s = sheet(&[PixelRect::new(10, 10, 40, 40)]);
        s.sprites[0].custom_outline = vec![vec![[0.0, 0.0], [40.0, 0.0], [20.0, 40.0]]];
        s.sprites[0].border = [20, 0, 20, 0];
        resize_sprite(&mut s, 0, Handle::Left, [20, 0], 100, 100);
        assert_eq!(s.sprites[0].rect, PixelRect::new(20, 10, 30, 40));
        // Points keep their place on the image where they still fit, else sit on the new edge.
        assert_eq!(s.sprites[0].custom_outline[0], vec![[0.0, 0.0], [30.0, 0.0], [10.0, 40.0]]);
        assert_eq!(s.sprites[0].border, [20, 0, 10, 0]);
    }

    #[test]
    fn rect_from_points_normalises_and_clamps() {
        assert_eq!(rect_from_points([30, 40], [10, 10], 100, 100), Some(PixelRect::new(10, 10, 20, 30)));
        assert_eq!(rect_from_points([-5, -5], [200, 200], 100, 100), Some(PixelRect::new(0, 0, 100, 100)));
        assert_eq!(rect_from_points([10, 10], [10, 50], 100, 100), None);
        assert_eq!(rect_from_points([150, 150], [200, 200], 100, 100), None);
    }

    #[test]
    fn snapping_rounds_to_the_grid() {
        assert_eq!(snap(13.4, 1), 13);
        assert_eq!(snap(13.6, 1), 14);
        assert_eq!(snap(13.0, 8), 16);
        assert_eq!(snap(11.0, 8), 8);
        assert_eq!(snap(5.0, 0), 5);
        assert_eq!(snap(-3.0, 4), -4);
    }

    #[test]
    fn handle_hit_testing_prefers_corners_and_respects_zoom() {
        let r = PixelRect::new(10, 10, 20, 20);
        let v = View { zoom: 2.0, pan: [0.0, 0.0] };
        assert_eq!(hit_handle(r, &v, [20.0, 20.0], 6.0), Some(Handle::TopLeft));
        assert_eq!(hit_handle(r, &v, [60.0, 60.0], 6.0), Some(Handle::BottomRight));
        assert_eq!(hit_handle(r, &v, [40.0, 20.0], 6.0), Some(Handle::Top));
        assert_eq!(hit_handle(r, &v, [40.0, 40.0], 6.0), None, "centre has no handle");
        // A tiny rect: corner wins over the edge midpoint.
        let tiny = PixelRect::new(0, 0, 2, 2);
        assert_eq!(hit_handle(tiny, &View { zoom: 1.0, pan: [0.0, 0.0] }, [0.0, 0.0], 3.0), Some(Handle::TopLeft));
    }

    #[test]
    fn picking_prefers_the_smallest_rect_and_marquee_selects_touching() {
        let s = sheet(&[PixelRect::new(0, 0, 100, 100), PixelRect::new(10, 10, 20, 20), PixelRect::new(200, 0, 10, 10)]);
        assert_eq!(pick_sprite(&s.sprites, [15.0, 15.0]), Some(1));
        assert_eq!(pick_sprite(&s.sprites, [60.0, 60.0]), Some(0));
        assert_eq!(pick_sprite(&s.sprites, [150.0, 150.0]), None);
        assert_eq!(sprites_in_box(&s.sprites, [5.0, 5.0], [15.0, 15.0]), vec![0, 1]);
        assert_eq!(sprites_in_box(&s.sprites, [195.0, 0.0], [260.0, 5.0]), vec![2]);
        assert!(sprites_in_box(&s.sprites, [120.0, 120.0], [130.0, 130.0]).is_empty());
    }

    #[test]
    fn multi_select_click_toggle_and_group_delete() {
        let mut sel = Selection::default();
        sel.click(2, false);
        sel.click(0, true);
        sel.click(2, true);
        assert_eq!(sel.items, vec![0]);
        sel.click(1, true);
        assert_eq!(sel.items, vec![0, 1]);
        assert_eq!(sel.primary(), Some(0));
        sel.click(3, false);
        assert_eq!(sel.items, vec![3]);
        sel.set(vec![2, 1, 2], true);
        assert_eq!(sel.items, vec![1, 2, 3]);
        sel.select_all(2);
        assert_eq!(sel.items, vec![0, 1]);
        sel.retain_below(1);
        assert_eq!(sel.items, vec![0]);

        let mut s = sheet(&[PixelRect::new(0, 0, 8, 8), PixelRect::new(10, 0, 8, 8), PixelRect::new(20, 0, 8, 8), PixelRect::new(30, 0, 8, 8)]);
        let ids: Vec<_> = s.sprites.iter().map(|d| d.id).collect();
        let mut sel = Selection { items: vec![0, 2] };
        assert_eq!(delete_selected(&mut s, &mut sel), 2);
        assert!(sel.items.is_empty());
        assert_eq!(s.sprites.iter().map(|d| d.id).collect::<Vec<_>>(), vec![ids[1], ids[3]]);
        // A deleted sprite's name keeps its id: adding the name back restores it.
        let i = s.add_sprite("t", PixelRect::new(0, 0, 4, 4));
        assert_eq!(s.sprites[i].name, "t_0");
        let mut s2 = s.clone();
        let mut sel = Selection::default();
        let j = create_sprite(&mut s2, "t", PixelRect::new(0, 8, 4, 4), &mut sel);
        assert_eq!(sel.items, vec![j]);
    }

    #[test]
    fn duplicate_copies_settings_with_new_names_and_ids() {
        let mut s = sheet(&[PixelRect::new(0, 0, 16, 16)]);
        s.sprites[0].border = [2, 2, 2, 2];
        s.sprites[0].set_pivot([0.25, 0.75]);
        let mut sel = Selection::single(0);
        duplicate_selected(&mut s, "t", &mut sel, 100, 100);
        assert_eq!(s.sprites.len(), 2);
        assert_eq!(sel.items, vec![1]);
        assert_eq!(s.sprites[1].rect, PixelRect::new(8, 8, 16, 16));
        assert_eq!(s.sprites[1].border, [2, 2, 2, 2]);
        assert_eq!(s.sprites[1].pivot, [0.25, 0.75]);
        assert_ne!(s.sprites[1].id, s.sprites[0].id);
        assert_ne!(s.sprites[1].name, s.sprites[0].name);
    }

    #[test]
    fn undo_redo_stack_snapshots_and_truncates_redo_on_new_edits() {
        let mut s = sheet(&[PixelRect::new(0, 0, 8, 8)]);
        let mut ed = SpriteEdit::new(&s);
        assert!(!ed.is_dirty(&s));
        assert!(!ed.undo(&mut s) && !ed.redo(&mut s));
        for x in [4u32, 8, 12] {
            ed.checkpoint(&s);
            s.sprites[0].rect.x = x;
        }
        assert!(ed.is_dirty(&s));
        assert!(ed.undo(&mut s));
        assert_eq!(s.sprites[0].rect.x, 8);
        assert!(ed.undo(&mut s));
        assert_eq!(s.sprites[0].rect.x, 4);
        assert!(ed.redo(&mut s));
        assert_eq!(s.sprites[0].rect.x, 8);
        // A new edit drops the redo branch.
        ed.checkpoint(&s);
        s.sprites[0].rect.x = 99;
        assert!(!ed.can_redo());
        assert!(ed.undo(&mut s));
        assert_eq!(s.sprites[0].rect.x, 8);
        // Undoing past a creation shrinks the table and the selection with it.
        ed.checkpoint(&s);
        s.add_sprite("t", PixelRect::new(0, 0, 4, 4));
        ed.sel = Selection::single(1);
        assert!(ed.undo(&mut s));
        assert!(ed.sel.items.is_empty());
        // The depth is capped.
        let mut st = UndoStack::new(3);
        for i in 0..10 {
            st.push(i);
        }
        let mut cur = 10;
        let mut n = 0;
        while let Some(p) = st.undo(&cur) {
            cur = p;
            n += 1;
        }
        assert_eq!(n, 3);
        assert_eq!(cur, 7);
    }

    #[test]
    fn revert_goes_back_to_the_applied_sheet_and_apply_rebases() {
        let mut s = sheet(&[PixelRect::new(0, 0, 8, 8)]);
        let mut ed = SpriteEdit::new(&s);
        ed.checkpoint(&s);
        s.sprites[0].rect.w = 5;
        assert!(ed.is_dirty(&s));
        ed.revert(&mut s);
        assert_eq!(s.sprites[0].rect.w, 8);
        assert!(!ed.is_dirty(&s));
        assert!(ed.undo(&mut s), "revert is undoable");
        assert_eq!(s.sprites[0].rect.w, 5);
        ed.mark_applied(&s);
        assert!(!ed.is_dirty(&s));
        assert!(!ed.can_undo(), "apply clears history");
        ed.revert(&mut s);
        assert_eq!(s.sprites[0].rect.w, 5);
    }

    #[test]
    fn pivot_drag_snaps_to_pixels_and_to_preset_points() {
        let r = PixelRect::new(0, 0, 32, 16);
        assert_eq!(pivot_from_drag(r, [16.0, 8.0], false, None), [0.5, 0.5]);
        // y is flipped: the top of the rect is pivot y = 1.
        assert_eq!(pivot_from_drag(r, [0.0, 0.0], false, None), [0.0, 1.0]);
        let p = pivot_from_drag(r, [10.4, 3.6], true, None);
        assert_eq!(p, [10.0 / 32.0, 1.0 - 4.0 / 16.0]);
        // Within 3 px of the centre it snaps to the Center preset.
        assert_eq!(pivot_from_drag(r, [17.0, 9.0], false, Some(3.0)), [0.5, 0.5]);
        assert_eq!(PivotPreset::detect(pivot_from_drag(r, [17.0, 9.0], false, Some(3.0))), PivotPreset::Center);
        // Too far: stays where it is.
        let q = pivot_from_drag(r, [20.0, 8.0], false, Some(2.0));
        assert!((q[0] - 20.0 / 32.0).abs() < 1e-6);
        let mut s = sheet(&[r]);
        set_pivot_preset(&mut s, &Selection::single(0), PivotPreset::Bottom);
        assert_eq!(s.sprites[0].pivot_norm(), [0.5, 0.0]);
        set_pivot_all(&mut s, &Selection::single(0), [0.25, 0.5]);
        assert_eq!(s.sprites[0].pivot_preset, PivotPreset::Custom);
        set_pivot_preset(&mut s, &Selection::single(0), PivotPreset::Custom);
        assert_eq!(s.sprites[0].pivot, [0.25, 0.5], "Custom is not a position");
    }

    #[test]
    fn border_drag_is_integer_clamped_and_limited_by_the_opposite_side() {
        let mut d = SpriteDef::named("a", PixelRect::new(0, 0, 40, 20));
        d.border = [10, 4, 10, 4];
        assert_eq!(border_from_drag(&d, BorderEdge::Left, [14.4, 0.0]), [14, 4, 10, 4]);
        assert_eq!(border_from_drag(&d, BorderEdge::Left, [-5.0, 0.0]), [0, 4, 10, 4]);
        // L + R <= w: left cannot exceed 40 - 10.
        assert_eq!(border_from_drag(&d, BorderEdge::Left, [100.0, 0.0]), [30, 4, 10, 4]);
        // Right edge dragged to local x = 25 -> right border 15.
        assert_eq!(border_from_drag(&d, BorderEdge::Right, [25.0, 0.0]), [10, 4, 15, 4]);
        // Top is measured from the top (y down), bottom from the bottom.
        assert_eq!(border_from_drag(&d, BorderEdge::Top, [0.0, 7.0]), [10, 4, 10, 7]);
        assert_eq!(border_from_drag(&d, BorderEdge::Bottom, [0.0, 12.0]), [10, 8, 10, 4]);
        assert_eq!(border_from_drag(&d, BorderEdge::Bottom, [0.0, -50.0]), [10, 16, 10, 4], "B + T <= h");
        let lines = border_lines(&d);
        assert_eq!(lines[0], (BorderEdge::Left, [[10.0, 0.0], [10.0, 20.0]]));
        assert_eq!(lines[1], (BorderEdge::Bottom, [[0.0, 16.0], [40.0, 16.0]]));
        assert_eq!(lines[2], (BorderEdge::Right, [[30.0, 0.0], [30.0, 20.0]]));
        assert_eq!(lines[3], (BorderEdge::Top, [[0.0, 4.0], [40.0, 4.0]]));
    }

    fn tri() -> Vec<Vec<P2>> {
        vec![vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]]
    }

    #[test]
    fn outline_points_insert_move_delete() {
        let mut l = tri();
        // Insert on the top edge near x = 5.
        let id = insert_point_on_edge(&mut l, [5.2, 1.0], 3.0, [10, 10], true).unwrap();
        assert_eq!(id, (0, 1));
        assert_eq!(l[0][1], [5.0, 0.0]);
        assert_eq!(l[0].len(), 5);
        // Too far from every edge: no insert.
        assert!(insert_point_on_edge(&mut l, [5.0, 5.0], 1.0, [10, 10], true).is_none());
        // Move clamps into the rect and snaps.
        move_point(&mut l, (0, 2), [40.0, -9.0], [10, 10], true);
        assert_eq!(l[0][2], [10.0, 0.0]);
        move_point(&mut l, (0, 2), [7.4, 2.6], [10, 10], true);
        assert_eq!(l[0][2], [7.0, 3.0]);
        move_point(&mut l, (0, 2), [7.4, 2.6], [10, 10], false);
        assert_eq!(l[0][2], [7.4, 2.6]);
        move_point(&mut l, (9, 9), [1.0, 1.0], [10, 10], true); // no panic
        assert_eq!(nearest_point(&l, [7.5, 2.5], 2.0), Some((0, 2)));
        assert_eq!(nearest_point(&l, [50.0, 50.0], 2.0), None);
        // Delete: 5 -> 4 points; deleting two more leaves 2 and removes the loop.
        assert_eq!(delete_points(&mut l, &[(0, 1)]), 1);
        assert_eq!(l[0].len(), 4);
        assert_eq!(delete_points(&mut l, &[(0, 0), (0, 1), (0, 1), (5, 0)]), 2);
        assert!(l.is_empty(), "a loop under 3 points is dropped");
        // Group move.
        let mut l = tri();
        move_points(&mut l, &[(0, 0), (0, 1)], [3.0, 3.0], [10, 10], true);
        assert_eq!(l[0][0], [3.0, 3.0]);
        assert_eq!(l[0][1], [10.0, 3.0], "clamped at the rect edge");
    }

    #[test]
    fn new_quad_and_box_selection_of_points() {
        let mut l: Vec<Vec<P2>> = Vec::new();
        assert!(new_quad(&mut l, [8.0, 8.0], [2.0, 3.0], [10, 10], true));
        assert_eq!(l[0], vec![[2.0, 3.0], [8.0, 3.0], [8.0, 8.0], [2.0, 8.0]]);
        assert!(!new_quad(&mut l, [1.0, 1.0], [1.2, 5.0], [10, 10], true), "degenerate drag");
        assert!(new_quad(&mut l, [-5.0, -5.0], [50.0, 50.0], [10, 10], true));
        assert_eq!(l[1][2], [10.0, 10.0]);
        let pts = points_in_box(&l, [0.0, 0.0], [4.0, 4.0]);
        assert!(pts.contains(&(0, 0)) && pts.contains(&(1, 0)));
        assert!(!pts.contains(&(0, 2)));
    }

    #[test]
    fn outline_generation_follows_alpha_and_detail_and_shapes_are_checked() {
        // An opaque disc.
        let (w, h) = (48u32, 48u32);
        let alpha = AlphaMap { w, h, alpha: (0..w * h).map(|i| { let (x, y) = ((i % w) as f32 - 24.0, (i / w) as f32 - 24.0); if x * x + y * y < 400.0 { 255 } else { 0 } }).collect() };
        let rect = PixelRect::new(0, 0, w, h);
        let coarse = generate_outline(&alpha, rect, 1, 0.0);
        let fine = generate_outline(&alpha, rect, 1, 1.0);
        assert_eq!(coarse.len(), 1);
        assert!(fine[0].len() > coarse[0].len(), "more detail, more points: {} vs {}", fine[0].len(), coarse[0].len());
        assert!(shape_problems(&fine).is_empty());
        assert!(!shape_problems(&[vec![[0.0, 0.0], [1.0, 1.0]]]).is_empty());
        let bow = vec![vec![[0.0, 0.0], [10.0, 10.0], [10.0, 0.0], [0.0, 10.0]]];
        assert!(shape_problems(&bow).iter().any(|p| p.contains("cross")));
        // Physics pieces are convex and cover the shape.
        let phys = generate_physics(&alpha, rect, 128, 0.85);
        let pieces = convex_pieces_of(&phys);
        assert!(!pieces.is_empty());
        assert!(convex_pieces_of(&[vec![[0.0, 0.0]]]).is_empty());
        // Copy between outline and physics.
        let mut d = SpriteDef::named("a", rect);
        *shape_mut(&mut d, ShapeKind::Outline) = fine.clone();
        copy_shape(&mut d, ShapeKind::Outline, ShapeKind::Physics);
        assert_eq!(shape(&d, ShapeKind::Physics), &fine);
    }

    #[test]
    fn reslicing_through_apply_slice_keeps_ids_and_references() {
        use crate::sprite::SpriteRef;
        let mut s = sheet(&[PixelRect::new(0, 0, 16, 16), PixelRect::new(16, 0, 16, 16)]);
        // Name them like a slice would, so a Replace re-slice produces the same names.
        assert!(s.rename_sprite(0, "t_0") && s.rename_sprite(1, "t_1"));
        s.sprites[0].set_pivot([0.5, 0.0]);
        let ids: Vec<_> = s.sprites.iter().map(|d| d.id).collect();
        let refs: Vec<SpriteRef> = s.sprites.iter().map(|d| SpriteRef::to(None, "a.png", d)).collect();
        // Smart: shifted cells keep names, ids and pivots.
        let (kept, _) = s.apply_slice_tracked("t", &[PixelRect::new(1, 1, 16, 16), PixelRect::new(17, 1, 16, 16)], SliceMethod::Smart);
        assert_eq!(kept, 2);
        assert_eq!(s.sprites.iter().map(|d| d.id).collect::<Vec<_>>(), ids);
        assert_eq!(s.sprites[0].pivot_norm(), [0.5, 0.0]);
        // Delete existing: new sprites get the retired names back, hence the same ids.
        s.apply_slice("t", &[PixelRect::new(0, 0, 16, 16), PixelRect::new(16, 0, 16, 16)], SliceMethod::DeleteExisting);
        assert_eq!(s.sprites.iter().map(|d| d.id).collect::<Vec<_>>(), ids, "same names after a re-slice come back with their old ids");
        for (r, d) in refs.iter().zip(&s.sprites) {
            assert_eq!(r.find_in(&s.sprites), Some(s.sprites.iter().position(|x| x.id == d.id).unwrap()));
        }
        // A different slice count: the reference falls back to the name, then is lost.
        s.apply_slice("t", &[PixelRect::new(0, 0, 32, 32)], SliceMethod::DeleteExisting);
        assert_eq!(s.sprites.len(), 1);
        assert_eq!(refs[0].find_in(&s.sprites), Some(0), "t_0 reappears with the first sprite's name and id");
        assert_eq!(refs[1].find_in(&s.sprites), None);
        // Renaming keeps the id.
        let id = s.sprites[0].id;
        assert!(s.rename_sprite(0, "hero"));
        assert_eq!(s.sprites[0].id, id);
        // Old meta files without a sprite section load as an empty Single sheet.
        let old: SpriteSheet = ron::de::from_str("()").unwrap();
        assert_eq!(old, SpriteSheet::default());
    }
}
