//! Pure maths behind the Animator Controller editor: the transition timeline,
//! grouping of parallel transitions into edges, grid snapping, box selection,
//! framing and live-follow. No egui state, so it is unit-tested.

use egui::{Pos2, Rect, Vec2};

// ------------------------------------------------------------ timeline maths

/// Where a transition sits on a shared seconds axis. The source state's clip
/// starts at 0; the cross-fade runs `start..stop`; the destination's clip
/// starts at `dst_start` so that its normalized `offset` lines up with `start`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TimelineGeom {
    pub start: f32,
    pub stop: f32,
    pub dst_start: f32,
    pub src_dur: f32,
    pub dst_dur: f32,
}

/// Lengths below this are treated as unknown/degenerate.
const MIN_DUR: f32 = 1e-3;

impl TimelineGeom {
    /// Layout from the transition's stored values (`exit_time` and `offset`
    /// are normalized; `duration` is seconds).
    pub fn new(exit_time: f32, duration: f32, offset: f32, src_dur: f32, dst_dur: f32) -> Self {
        let src_dur = src_dur.max(MIN_DUR);
        let dst_dur = dst_dur.max(MIN_DUR);
        let start = exit_time.max(0.0) * src_dur;
        Self { start, stop: start + duration.max(0.0), dst_start: start - offset.clamp(0.0, 1.0) * dst_dur, src_dur, dst_dur }
    }

    pub fn duration(&self) -> f32 {
        (self.stop - self.start).max(0.0)
    }

    /// The stored values: `(exit_time, duration, offset)`, rounded to 1e-3.
    pub fn values(&self) -> (f32, f32, f32) {
        let r = |v: f32| (v * 1000.0).round() / 1000.0;
        (r((self.start / self.src_dur).max(0.0)), r(self.duration()), r(((self.start - self.dst_start) / self.dst_dur).clamp(0.0, 1.0)))
    }

    /// Source state's normalized time at `t` seconds.
    pub fn src_percent(&self, t: f32) -> f32 {
        t / self.src_dur * 100.0
    }

    /// Destination state's normalized time at `t` seconds.
    pub fn dst_percent(&self, t: f32) -> f32 {
        (t - self.dst_start) / self.dst_dur * 100.0
    }
}

/// What a timeline drag edits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TimelineDrag {
    /// Left handle: Exit Time. The fade end stays put, so the duration changes.
    ExitTime,
    /// Right handle: Duration.
    Duration,
    /// Destination bar: Transition Offset.
    Offset,
    /// Region head in the ruler: moves the whole fade (exit time), keeping
    /// duration and offset.
    Region,
    /// Source bar / empty space: pans the time view, edits nothing.
    Pan,
}

/// Apply a drag of `dt` seconds (total since the mouse went down) to the
/// geometry captured at mouse-down. Never mutates stored data: the caller
/// commits the result on release.
pub fn apply_timeline_drag(kind: TimelineDrag, g0: &TimelineGeom, dt: f32) -> TimelineGeom {
    let mut g = *g0;
    match kind {
        TimelineDrag::ExitTime => {
            g.start = (g0.start + dt).clamp(0.0, g0.stop);
            g.dst_start = g.start - (g0.start - g0.dst_start);
        }
        TimelineDrag::Duration => {
            g.stop = (g0.stop + dt).max(g0.start);
        }
        TimelineDrag::Offset => {
            // The destination bar slides: dragging right lowers the offset.
            let offset = ((g0.start - (g0.dst_start + dt)) / g0.dst_dur).clamp(0.0, 1.0);
            g.dst_start = g0.start - offset * g0.dst_dur;
        }
        TimelineDrag::Region => {
            let start = (g0.start + dt).max(0.0);
            let shift = start - g0.start;
            g.start = start;
            g.stop = g0.stop + shift;
            g.dst_start = g0.dst_start + shift;
        }
        TimelineDrag::Pan => {}
    }
    g
}

/// Seconds <-> pixels for the timeline body.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TimeAxis {
    /// Time at `x0`.
    pub t0: f32,
    pub x0: f32,
    pub px_per_sec: f32,
}

impl TimeAxis {
    pub fn x(&self, t: f32) -> f32 {
        self.x0 + (t - self.t0) * self.px_per_sec
    }
    pub fn t(&self, x: f32) -> f32 {
        self.t0 + (x - self.x0) / self.px_per_sec
    }
}

/// The span of time worth showing: both bars and the fade, looping bars
/// covering the fade.
pub fn timeline_extent(g: &TimelineGeom) -> (f32, f32) {
    let lo = 0.0_f32.min(g.dst_start);
    let hi = g.src_dur.max(g.dst_start + g.dst_dur).max(g.stop);
    (lo, hi.max(lo + MIN_DUR))
}

/// An axis fitting `lo..hi` seconds into `x0..x1` pixels with `pad` px each side.
pub fn fit_axis(lo: f32, hi: f32, x0: f32, x1: f32, pad: f32) -> TimeAxis {
    let width = (x1 - x0 - 2.0 * pad).max(1.0);
    TimeAxis { t0: lo, x0: x0 + pad, px_per_sec: width / (hi - lo).max(MIN_DUR) }
}

/// A round tick spacing (1, 2, 5 times a power of ten) of at least
/// `min_seconds`.
pub fn tick_step(min_seconds: f32) -> f32 {
    let min = min_seconds.max(1e-4);
    let e = min.log10().floor();
    [1.0, 2.0, 5.0, 10.0].iter().map(|m| m * 10f32.powf(e)).find(|v| *v >= min * 0.9999).unwrap_or(min)
}

/// The spans a bar of length `dur` starting at `start` is drawn as. A looping
/// state repeats until it covers `until`.
pub fn bar_spans(start: f32, dur: f32, looping: bool, until: f32) -> Vec<(f32, f32)> {
    let dur = dur.max(MIN_DUR);
    let mut out = vec![(start, start + dur)];
    if looping {
        while out.last().is_some_and(|s| s.1 < until) && out.len() < 64 {
            let e = out.last().unwrap().1;
            out.push((e, e + dur));
        }
    }
    out
}

/// Pixel layout of the timeline, relative to its rect's top-left (spec: the
/// control is 150 px tall).
pub const TL_HEIGHT: f32 = 150.0;
pub const TL_RULER_H: f32 = 18.0;
pub const TL_SRC_Y: f32 = 85.0;
pub const TL_DST_Y: f32 = 117.0;
pub const TL_BAR_H: f32 = 32.0;
pub const TL_HANDLE_W: f32 = 9.0;
pub const TL_HANDLE_H: f32 = 15.0;

/// What `p` (relative to the timeline rect's top-left, x in the same space as
/// `axis`) grabs. Hit order: handles, destination bar, region head, source bar.
/// `has_exit`: without an exit time the left handle and the region are inert.
pub fn timeline_hit(g: &TimelineGeom, axis: &TimeAxis, p: Pos2, has_exit: bool) -> TimelineDrag {
    let xs = axis.x(g.start);
    let xe = axis.x(g.stop);
    let handle_y = TL_RULER_H..=TL_RULER_H + TL_HANDLE_H;
    let near = |x: f32, cx: f32| (x - cx).abs() <= TL_HANDLE_W * 0.5 + 2.0;
    if handle_y.contains(&p.y) || (p.y > TL_RULER_H && p.y < TL_SRC_Y) {
        if near(p.x, xe) {
            return TimelineDrag::Duration;
        }
        if has_exit && near(p.x, xs) {
            return TimelineDrag::ExitTime;
        }
    }
    if (TL_DST_Y..=TL_DST_Y + TL_BAR_H).contains(&p.y) {
        return TimelineDrag::Offset;
    }
    if p.y < TL_RULER_H && has_exit && p.x >= xs.min(xe) && p.x <= xs.max(xe) {
        return TimelineDrag::Region;
    }
    TimelineDrag::Pan
}

/// Estimated length of a state's motion: the clip's, or for a blend tree the
/// mean of its known clips. `None` when nothing is known.
pub fn mean_known(lengths: impl IntoIterator<Item = Option<f32>>) -> Option<f32> {
    let known: Vec<f32> = lengths.into_iter().flatten().filter(|l| *l > MIN_DUR).collect();
    if known.is_empty() { None } else { Some(known.iter().sum::<f32>() / known.len() as f32) }
}

// ------------------------------------------------------------ edge grouping

/// One drawn edge: every transition between the same pair of nodes.
#[derive(Clone, PartialEq, Debug)]
pub struct EdgeGroup {
    /// Source state index; `None` = Any State.
    pub from: Option<usize>,
    pub to: usize,
    /// Transition indices in authored order.
    pub transitions: Vec<usize>,
}

/// Group transitions (as `(from, to)` pairs, one per transition index; `None`
/// entries are unresolved and skipped) into one edge per node pair.
pub fn group_edges(pairs: &[Option<(Option<usize>, usize)>]) -> Vec<EdgeGroup> {
    let mut out: Vec<EdgeGroup> = Vec::new();
    for (k, p) in pairs.iter().enumerate() {
        let Some((from, to)) = *p else { continue };
        match out.iter_mut().find(|g| g.from == from && g.to == to) {
            Some(g) => g.transitions.push(k),
            None => out.push(EdgeGroup { from, to, transitions: vec![k] }),
        }
    }
    out
}

/// Whether an edge between the same nodes runs the other way.
pub fn has_reverse(groups: &[EdgeGroup], g: &EdgeGroup) -> bool {
    g.from.is_some_and(|f| groups.iter().any(|o| o.from == Some(g.to) && o.to == f))
}

/// Sideways offset (screen px) that separates A->B from B->A. Both edges
/// offset along their own direction's left normal, so they land on opposite
/// sides. Proportional to the zoom, like the nodes, so zooming out keeps the
/// pair tight; the floor keeps them distinguishable and the cap stops a wide
/// gap when zoomed far in.
pub fn edge_offset(a: Pos2, b: Pos2, reverse: bool, zoom: f32) -> Vec2 {
    if !reverse {
        return Vec2::ZERO;
    }
    let d = b - a;
    if d.length_sq() < 1e-6 {
        return Vec2::ZERO;
    }
    let dir = d.normalized();
    Vec2::new(-dir.y, dir.x) * (5.0 * zoom).clamp(2.0, 10.0)
}

/// Fractions along an edge at which arrowheads sit: one, or three when the
/// edge holds several transitions.
pub fn arrow_fractions(count: usize) -> &'static [f32] {
    if count > 1 { &[0.3, 0.5, 0.7] } else { &[0.5] }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArrowTint {
    Normal,
    Muted,
    Solo,
}

/// Arrow colours for an edge from its transitions' `(muted, solo)` flags.
/// All muted: red; all soloed: green. Mixed: the first arrow shows muted
/// (else solo), the last turns green when both kinds are present.
pub fn arrow_tints(flags: &[(bool, bool)]) -> Vec<ArrowTint> {
    let n = arrow_fractions(flags.len()).len();
    let muted = flags.iter().any(|f| f.0);
    let solo = flags.iter().any(|f| f.1);
    if !flags.is_empty() && flags.iter().all(|f| f.0) {
        return vec![ArrowTint::Muted; n];
    }
    if !flags.is_empty() && flags.iter().all(|f| f.1) {
        return vec![ArrowTint::Solo; n];
    }
    let first = if muted {
        ArrowTint::Muted
    } else if solo {
        ArrowTint::Solo
    } else {
        ArrowTint::Normal
    };
    let mut out = vec![ArrowTint::Normal; n];
    out[0] = first;
    if n == 3 && muted && solo {
        out[2] = ArrowTint::Solo;
    }
    out
}

/// Distance from `p` to the segment `a..b`.
pub fn dist_to_segment(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_sq().max(1e-6)).clamp(0.0, 1.0);
    (a + ab * t - p).length()
}

// ------------------------------------------------------ snapping, selection

/// Nodes snap to this many canvas units while dragging.
pub const GRID: f32 = 10.0;

pub fn snap(v: f32, grid: f32) -> f32 {
    if grid <= 0.0 { v } else { (v / grid).round() * grid }
}

pub fn snap_pos(p: [f32; 2], grid: f32) -> [f32; 2] {
    [snap(p[0], grid), snap(p[1], grid)]
}

/// Canvas positions for `n` new states laid out left to right in rows of
/// `cols`, the first centred on `origin`, `step` apart.
pub fn grid_positions(n: usize, origin: [f32; 2], cols: usize, step: [f32; 2]) -> Vec<[f32; 2]> {
    let cols = cols.max(1);
    (0..n).map(|k| [origin[0] + (k % cols) as f32 * step[0], origin[1] + (k / cols) as f32 * step[1]]).collect()
}

/// Ids whose rect touches the selection box.
pub fn nodes_in_rect<T: Copy>(rects: &[(T, Rect)], sel: Rect) -> Vec<T> {
    rects.iter().filter(|(_, r)| r.intersects(sel)).map(|(id, _)| *id).collect()
}

/// New selection after a box: replaces `prev` unless `additive` (Ctrl/Shift),
/// which toggles each hit instead.
pub fn combine_selection<T: Copy + PartialEq>(prev: &[T], hits: &[T], additive: bool) -> Vec<T> {
    if !additive {
        return hits.to_vec();
    }
    let mut out = prev.to_vec();
    for h in hits {
        if let Some(i) = out.iter().position(|o| o == h) {
            out.remove(i);
        } else {
            out.push(*h);
        }
    }
    out
}

/// Click toggling (Ctrl/Shift) or replacing.
pub fn click_selection<T: Copy + PartialEq>(prev: &[T], hit: T, additive: bool) -> Vec<T> {
    combine_selection(prev, &[hit], additive)
}

// ------------------------------------------------------------------ framing

pub const FRAME_BORDER: f32 = 30.0;
pub const ZOOM_MIN: f32 = 0.1;
pub const ZOOM_MAX: f32 = 3.0;

/// Smallest rect holding every rect (canvas units).
pub fn bounds_of(rects: impl IntoIterator<Item = Rect>) -> Option<Rect> {
    rects.into_iter().reduce(|a, b| a.union(b))
}

/// `(pan, zoom)` that fit `bounds` (canvas units) into a view of `view`
/// pixels with a border, zoom capped at `ZOOM_MAX`. Matches the canvas
/// transform `screen = center + (pan + p) * zoom`.
pub fn frame_bounds(bounds: Rect, view: Vec2) -> (Vec2, f32) {
    let w = (view.x - 2.0 * FRAME_BORDER).max(1.0) / bounds.width().max(1.0);
    let h = (view.y - 2.0 * FRAME_BORDER).max(1.0) / bounds.height().max(1.0);
    (-bounds.center().to_vec2(), w.min(h).clamp(ZOOM_MIN, ZOOM_MAX))
}

/// The smallest screen-space scroll that brings `node` fully inside `view`
/// (padded). Zero when already visible. Add it to the nodes' screen
/// positions, i.e. subtract `/zoom` from the pan.
pub fn scroll_to_keep_visible(view: Rect, node: Rect, pad: f32) -> Vec2 {
    let inner = view.shrink(pad);
    let axis = |lo: f32, hi: f32, vlo: f32, vhi: f32| {
        if hi - lo > vhi - vlo {
            (vlo + vhi) * 0.5 - (lo + hi) * 0.5
        } else if lo < vlo {
            vlo - lo
        } else if hi > vhi {
            vhi - hi
        } else {
            0.0
        }
    };
    Vec2::new(axis(node.left(), node.right(), inner.left(), inner.right()), axis(node.top(), node.bottom(), inner.top(), inner.bottom()))
}

// -------------------------------------------------------------- live follow

/// The sub-machine path (names, outermost first) the live view should show:
/// the machine holding the active state, or during a transition the source's
/// until progress passes one half, then the destination's (immediately when
/// the transition leaves Any State). State names are runtime paths
/// (`Outer/Inner/State`).
pub fn follow_machine_path(current: &str, source: Option<&str>, in_transition: bool, progress: f32, from_any: bool) -> Vec<String> {
    let state = match source {
        Some(src) if in_transition && !from_any && progress < 0.5 => src,
        _ => current,
    };
    let mut parts: Vec<&str> = state.split('/').collect();
    parts.pop();
    parts.into_iter().map(str::to_owned).collect()
}

/// Width fraction of the in-node progress strip: wraps for looping states
/// (reversed when time runs backwards), clamps otherwise.
pub fn node_progress(normalized_time: f32, looping: bool) -> f32 {
    if looping {
        if normalized_time < 0.0 { 1.0 - (-normalized_time).rem_euclid(1.0) } else { normalized_time.rem_euclid(1.0) }
    } else {
        normalized_time.clamp(0.0, 1.0)
    }
}

// ------------------------------------------------------ blend tree maths

/// Positions closer than this (in parameter units) count as the same point.
const BT_CLOSE: f32 = 1e-3;
/// Simple Directional children closer than this many degrees are ambiguous.
const BT_MIN_ANGLE_DEG: f32 = 2.0;

/// Where child `k` belongs once its value changed: the index it takes in the
/// list sorted by `values` (equal values keep their order, the moved child
/// goes after its equals).
pub fn sorted_slot(values: &[f32], k: usize) -> usize {
    let v = values[k];
    let before = values.iter().enumerate().filter(|&(i, &x)| i != k && x <= v).count();
    before.min(values.len() - 1)
}

/// Move element `from` to index `to` (the index it ends up at).
pub fn move_item<T>(items: &mut Vec<T>, from: usize, to: usize) {
    if from < items.len() && to < items.len() && from != to {
        let it = items.remove(from);
        items.insert(to, it);
    }
}

/// A threshold dragged within `snap` units of a neighbour's value takes it.
pub fn snap_threshold(v: f32, others: impl IntoIterator<Item = f32>, snap: f32) -> f32 {
    let mut best = (snap, v);
    for o in others {
        let d = (o - v).abs();
        if d <= best.0 {
            best = (d, o);
        }
    }
    best.1
}

/// The 1D tent of child `i` among thresholds sorted ascending: the x of its
/// zero on the left, its peak, its zero on the right (the ends stay flat, so
/// the first and last children start and end at their own threshold).
pub fn triangle_1d(sorted: &[f32], i: usize) -> (f32, f32, f32) {
    let x = sorted[i];
    let left = if i == 0 { x } else { sorted[i - 1] };
    let right = sorted.get(i + 1).copied().unwrap_or(x);
    (left, x, right)
}

/// Parameter range shown by the 1D strip (never zero-width).
pub fn range_1d(thresholds: &[f32]) -> (f32, f32) {
    let lo = thresholds.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = thresholds.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !lo.is_finite() || !hi.is_finite() {
        return (0.0, 1.0);
    }
    if hi - lo < 1e-4 { (lo - 0.5, hi + 0.5) } else { (lo, hi) }
}

/// Square view of a 2D tree: `(centre, half size)` in parameter units.
/// Directional kinds centre on the origin and reach the largest |x| or |y|;
/// Cartesian centres on the bounding box. Both leave 35% padding.
pub fn view_2d(directional: bool, pts: &[[f32; 2]]) -> ([f32; 2], f32) {
    if pts.is_empty() {
        return ([0.0, 0.0], 1.0);
    }
    let (mut lo, mut hi) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]);
    for p in pts {
        for a in 0..2 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    let (centre, half) = if directional {
        ([0.0, 0.0], lo.iter().chain(hi.iter()).fold(0.0_f32, |m, v| m.max(v.abs())))
    } else {
        ([(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5], ((hi[0] - lo[0]).max(hi[1] - lo[1])) * 0.5)
    };
    (centre, (half * 1.35).max(0.5))
}

/// Parameter point to a screen point in a square diagram (y points up).
pub fn to_diagram(p: [f32; 2], centre: [f32; 2], half: f32, rect: Rect) -> Pos2 {
    let k = rect.width().min(rect.height()) * 0.5 / half;
    Pos2::new(rect.center().x + (p[0] - centre[0]) * k, rect.center().y - (p[1] - centre[1]) * k)
}

/// Inverse of [`to_diagram`].
pub fn from_diagram(s: Pos2, centre: [f32; 2], half: f32, rect: Rect) -> [f32; 2] {
    let k = rect.width().min(rect.height()) * 0.5 / half;
    [centre[0] + (s.x - rect.center().x) / k, centre[1] - (s.y - rect.center().y) / k]
}

/// Radius in pixels of a child's live weight circle.
pub fn weight_radius(w: f32) -> f32 {
    20.0 * w.max(0.0).sqrt()
}

/// What is wrong with the positions of a 2D tree.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlendWarning {
    /// Two children sit on (almost) the same point or direction.
    TooClose,
    /// Simple Directional: the directions leave a gap of 180 degrees or more.
    DirectionsTooWide,
    /// Freeform Directional wants a child at the origin.
    NoOriginMotion,
}

impl BlendWarning {
    pub fn text(self) -> &'static str {
        match self {
            Self::TooClose => "Two or more of the positions are too close to each other.",
            Self::DirectionsTooWide => "Directions must be less than 180 degrees apart.",
            Self::NoOriginMotion => "Freeform Directional should have one motion at position (0, 0).",
        }
    }
}

/// Problems with `pos` for a 2D `kind` (1D and Direct have none).
pub fn blend_warnings(kind: zestengine_core::BlendTreeKind, pos: &[[f32; 2]]) -> Vec<BlendWarning> {
    use zestengine_core::BlendTreeKind as K;
    let mut out = Vec::new();
    if !kind.is_2d() {
        return out;
    }
    let len = |p: [f32; 2]| (p[0] * p[0] + p[1] * p[1]).sqrt();
    let mut close = false;
    for i in 0..pos.len() {
        for j in i + 1..pos.len() {
            let d = [pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]];
            close |= len(d) < BT_CLOSE;
        }
    }
    let dirs: Vec<f32> = pos.iter().filter(|&&p| len(p) > BT_CLOSE).map(|p| p[1].atan2(p[0])).collect();
    if kind == K::SimpleDirectional2D && dirs.len() >= 2 {
        let mut a = dirs.clone();
        a.sort_by(f32::total_cmp);
        let tau = std::f32::consts::TAU;
        let mut widest = 0.0_f32;
        let mut narrowest = f32::INFINITY;
        for k in 0..a.len() {
            let next = if k + 1 < a.len() { a[k + 1] } else { a[0] + tau };
            widest = widest.max(next - a[k]);
            narrowest = narrowest.min(next - a[k]);
        }
        close |= narrowest.to_degrees() < BT_MIN_ANGLE_DEG;
        if widest >= std::f32::consts::PI - 1e-4 {
            out.push(BlendWarning::DirectionsTooWide);
        }
    }
    if close {
        out.insert(0, BlendWarning::TooClose);
    }
    if kind == K::FreeformDirectional2D && !pos.iter().any(|&p| len(p) <= BT_CLOSE) {
        out.push(BlendWarning::NoOriginMotion);
    }
    out
}

/// Evenly spaced thresholds from `lo` to `hi` (Compute / Automate Thresholds).
pub fn spread_thresholds(n: usize, lo: f32, hi: f32) -> Vec<f32> {
    match n {
        0 => Vec::new(),
        1 => vec![lo],
        _ => (0..n).map(|i| lo + (hi - lo) * i as f32 / (n - 1) as f32).collect(),
    }
}

/// Positions on a circle of `radius`, the first child at the origin when
/// `centre` is set (Compute Positions for directional trees).
pub fn ring_positions(n: usize, radius: f32, centre: bool) -> Vec<[f32; 2]> {
    let ring = n - usize::from(centre && n > 0);
    let mut out = Vec::with_capacity(n);
    if centre && n > 0 {
        out.push([0.0, 0.0]);
    }
    for i in 0..ring {
        let a = std::f32::consts::TAU * i as f32 / ring as f32 + std::f32::consts::FRAC_PI_2;
        out.push([a.cos() * radius, a.sin() * radius]);
    }
    out
}

/// Grid of per-cell values over a square diagram, row-major from the top:
/// `f` gets the parameter point of each cell centre.
pub fn weight_field(res: usize, centre: [f32; 2], half: f32, f: impl Fn([f32; 2]) -> f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(res * res);
    for row in 0..res {
        for col in 0..res {
            let u = (col as f32 + 0.5) / res as f32 * 2.0 - 1.0;
            let v = 1.0 - (row as f32 + 0.5) / res as f32 * 2.0;
            out.push(f([centre[0] + u * half, centre[1] + v * half]));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_positions_fill_rows() {
        let p = grid_positions(5, [10.0, 20.0], 2, [100.0, 50.0]);
        assert_eq!(p, vec![[10.0, 20.0], [110.0, 20.0], [10.0, 70.0], [110.0, 70.0], [10.0, 120.0]]);
        assert_eq!(grid_positions(2, [0.0; 2], 0, [5.0, 5.0]), vec![[0.0, 0.0], [0.0, 5.0]], "zero columns counts as one");
        assert!(grid_positions(0, [0.0; 2], 3, [1.0, 1.0]).is_empty());
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn geom_round_trips_stored_values() {
        let g = TimelineGeom::new(0.75, 0.25, 0.3, 2.0, 1.0);
        assert!(close(g.start, 1.5) && close(g.stop, 1.75) && close(g.dst_start, 1.2));
        let (e, d, o) = g.values();
        assert!(close(e, 0.75) && close(d, 0.25) && close(o, 0.3));
    }

    #[test]
    fn percent_badges() {
        let g = TimelineGeom::new(0.5, 0.5, 0.25, 2.0, 4.0);
        assert!(close(g.src_percent(g.start), 50.0));
        assert!(close(g.dst_percent(g.start), 25.0));
    }

    #[test]
    fn exit_handle_keeps_fade_end_and_clamps() {
        let g = TimelineGeom::new(0.5, 0.5, 0.0, 2.0, 2.0);
        let d = apply_timeline_drag(TimelineDrag::ExitTime, &g, -0.5);
        assert!(close(d.start, 0.5) && close(d.stop, g.stop) && close(d.duration(), 1.0));
        let past = apply_timeline_drag(TimelineDrag::ExitTime, &g, 9.0);
        assert!(close(past.start, g.stop) && close(past.duration(), 0.0));
        let before = apply_timeline_drag(TimelineDrag::ExitTime, &g, -9.0);
        assert!(close(before.start, 0.0));
        // Offset stays the same normalized value.
        assert!(close(d.values().2, g.values().2));
    }

    #[test]
    fn duration_handle_cannot_cross_start() {
        let g = TimelineGeom::new(0.5, 0.5, 0.0, 2.0, 2.0);
        assert!(close(apply_timeline_drag(TimelineDrag::Duration, &g, -5.0).duration(), 0.0));
        assert!(close(apply_timeline_drag(TimelineDrag::Duration, &g, 0.5).duration(), 1.0));
        assert!(close(apply_timeline_drag(TimelineDrag::Duration, &g, 0.5).start, g.start));
    }

    #[test]
    fn dst_bar_drag_edits_offset_only() {
        let g = TimelineGeom::new(0.5, 0.25, 0.5, 2.0, 2.0);
        // Drag right by 0.5 s: offset 0.5 -> 0.25.
        let d = apply_timeline_drag(TimelineDrag::Offset, &g, 0.5);
        let (e, dur, o) = d.values();
        assert!(close(o, 0.25) && close(e, 0.5) && close(dur, 0.25));
        // Clamped to [0, 1].
        assert!(close(apply_timeline_drag(TimelineDrag::Offset, &g, 9.0).values().2, 0.0));
        assert!(close(apply_timeline_drag(TimelineDrag::Offset, &g, -9.0).values().2, 1.0));
    }

    #[test]
    fn region_drag_keeps_duration_and_offset() {
        let g = TimelineGeom::new(0.5, 0.25, 0.4, 2.0, 1.0);
        let d = apply_timeline_drag(TimelineDrag::Region, &g, 0.3);
        let (e, dur, o) = d.values();
        assert!(close(e, 0.65) && close(dur, 0.25) && close(o, 0.4));
        assert!(close(apply_timeline_drag(TimelineDrag::Region, &g, -9.0).start, 0.0));
    }

    #[test]
    fn pan_changes_nothing() {
        let g = TimelineGeom::new(0.5, 0.25, 0.4, 2.0, 1.0);
        assert_eq!(apply_timeline_drag(TimelineDrag::Pan, &g, 3.0), g);
    }

    #[test]
    fn zero_length_clip_does_not_divide_by_zero() {
        let g = TimelineGeom::new(1.0, 0.25, 0.5, 0.0, 0.0);
        let (e, d, o) = g.values();
        assert!(e.is_finite() && d.is_finite() && o.is_finite());
    }

    #[test]
    fn axis_fits_extent() {
        let g = TimelineGeom::new(0.5, 0.5, 0.5, 2.0, 2.0);
        let (lo, hi) = timeline_extent(&g);
        assert!(close(lo, 0.0) && close(hi, 2.0));
        let ax = fit_axis(lo, hi, 0.0, 220.0, 10.0);
        assert!(close(ax.x(0.0), 10.0) && close(ax.x(2.0), 210.0));
        assert!(close(ax.t(ax.x(1.3)), 1.3));
        // A destination offset past the start extends the axis to the left.
        let neg = TimelineGeom { dst_start: -0.5, ..g };
        assert!(close(timeline_extent(&neg).0, -0.5));
    }

    #[test]
    fn tick_steps_are_round() {
        assert!(close(tick_step(0.03), 0.05) && close(tick_step(0.1), 0.1) && close(tick_step(0.11), 0.2));
        assert!(close(tick_step(3.0), 5.0) && close(tick_step(7.0), 10.0));
    }

    #[test]
    fn looping_bars_repeat_to_cover_the_fade() {
        assert_eq!(bar_spans(0.0, 1.0, false, 3.0).len(), 1);
        let s = bar_spans(0.0, 1.0, true, 2.5);
        assert_eq!(s.len(), 3);
        assert!(close(s[2].1, 3.0));
        assert_eq!(bar_spans(0.0, 0.0, true, 1e9).len(), 64);
    }

    #[test]
    fn hit_test_order() {
        let g = TimelineGeom::new(0.5, 0.5, 0.0, 2.0, 2.0);
        let ax = fit_axis(0.0, 3.0, 0.0, 300.0, 0.0);
        let xs = ax.x(g.start);
        let xe = ax.x(g.stop);
        assert_eq!(timeline_hit(&g, &ax, Pos2::new(xs, 25.0), true), TimelineDrag::ExitTime);
        assert_eq!(timeline_hit(&g, &ax, Pos2::new(xe, 25.0), true), TimelineDrag::Duration);
        assert_eq!(timeline_hit(&g, &ax, Pos2::new(xs, 25.0), false), TimelineDrag::Pan);
        assert_eq!(timeline_hit(&g, &ax, Pos2::new(10.0, TL_DST_Y + 5.0), true), TimelineDrag::Offset);
        assert_eq!(timeline_hit(&g, &ax, Pos2::new((xs + xe) * 0.5, 5.0), true), TimelineDrag::Region);
        assert_eq!(timeline_hit(&g, &ax, Pos2::new((xs + xe) * 0.5, 5.0), false), TimelineDrag::Pan);
        assert_eq!(timeline_hit(&g, &ax, Pos2::new(10.0, TL_SRC_Y + 5.0), true), TimelineDrag::Pan);
    }

    #[test]
    fn mean_of_known_lengths() {
        assert_eq!(mean_known([None, None]), None);
        assert!(close(mean_known([Some(1.0), None, Some(3.0)]).unwrap(), 2.0));
    }

    #[test]
    fn parallel_transitions_form_one_edge() {
        let pairs = [Some((Some(0), 1)), Some((Some(0), 1)), Some((Some(1), 0)), None, Some((None, 1))];
        let g = group_edges(&pairs);
        assert_eq!(g.len(), 3);
        assert_eq!(g[0].transitions, vec![0, 1]);
        assert!(has_reverse(&g, &g[0]));
        assert!(has_reverse(&g, &g[1]));
        // Any State -> 1 is not reversed by 1 -> 0.
        assert!(!has_reverse(&g, &g[2]));
    }

    #[test]
    fn opposite_edges_offset_to_opposite_sides() {
        let (a, b) = (Pos2::new(0.0, 0.0), Pos2::new(100.0, 0.0));
        let ab = edge_offset(a, b, true, 1.0);
        let ba = edge_offset(b, a, true, 1.0);
        assert!(close(ab.y, -ba.y) && ab.y.abs() > 4.9);
        assert_eq!(edge_offset(a, b, false, 1.0), Vec2::ZERO);
        // Zooming out keeps the pair tight (it used to widen, splitting the
        // two lines apart), but never below the 2 px floor; zooming in is capped.
        let far = edge_offset(a, b, true, 0.1).length();
        assert!(far < ab.length() && close(far, 2.0));
        assert!(close(edge_offset(a, b, true, 0.33).length(), 2.0));
        assert!(close(edge_offset(a, b, true, 50.0).length(), 10.0));
    }

    #[test]
    fn arrow_counts_and_tints() {
        assert_eq!(arrow_fractions(1).len(), 1);
        assert_eq!(arrow_fractions(2).len(), 3);
        assert_eq!(arrow_tints(&[(false, false)]), vec![ArrowTint::Normal]);
        assert_eq!(arrow_tints(&[(true, false)]), vec![ArrowTint::Muted]);
        assert_eq!(arrow_tints(&[(true, false), (true, false)]), vec![ArrowTint::Muted; 3]);
        assert_eq!(arrow_tints(&[(false, true), (false, true)]), vec![ArrowTint::Solo; 3]);
        assert_eq!(arrow_tints(&[(true, false), (false, true)]), vec![ArrowTint::Muted, ArrowTint::Normal, ArrowTint::Solo]);
        assert_eq!(arrow_tints(&[(false, true), (false, false)]), vec![ArrowTint::Solo, ArrowTint::Normal, ArrowTint::Normal]);
    }

    #[test]
    fn snapping() {
        assert!(close(snap(14.9, 10.0), 10.0) && close(snap(15.0, 10.0), 20.0) && close(snap(-4.0, 10.0), 0.0));
        assert!(close(snap(3.3, 0.0), 3.3));
        assert_eq!(snap_pos([12.0, -26.0], GRID), [10.0, -30.0]);
    }

    #[test]
    fn box_select() {
        let rects = [(0, Rect::from_min_size(Pos2::new(0.0, 0.0), Vec2::splat(10.0))), (1, Rect::from_min_size(Pos2::new(50.0, 0.0), Vec2::splat(10.0)))];
        let hit = nodes_in_rect(&rects, Rect::from_min_max(Pos2::new(5.0, 5.0), Pos2::new(55.0, 8.0)));
        assert_eq!(hit, vec![0, 1]);
        assert_eq!(nodes_in_rect(&rects, Rect::from_min_max(Pos2::new(20.0, 0.0), Pos2::new(30.0, 8.0))), Vec::<i32>::new());
        // Touching an edge counts as inside; a box fully inside a node too.
        assert_eq!(nodes_in_rect(&rects, Rect::from_min_max(Pos2::new(2.0, 2.0), Pos2::new(3.0, 3.0))), vec![0]);
    }

    #[test]
    fn selection_combine() {
        assert_eq!(combine_selection(&[1, 2], &[3], false), vec![3]);
        assert_eq!(combine_selection(&[1, 2], &[2, 3], true), vec![1, 3]);
        assert_eq!(click_selection(&[1, 2], 2, true), vec![1]);
        assert_eq!(click_selection(&[1, 2], 3, false), vec![3]);
        assert_eq!(click_selection(&[1], 2, true), vec![1, 2]);
    }

    #[test]
    fn framing() {
        let b = Rect::from_min_max(Pos2::new(-100.0, -50.0), Pos2::new(100.0, 50.0));
        let (pan, zoom) = frame_bounds(b, Vec2::new(460.0, 300.0));
        assert!(close(zoom, 2.0) && close(pan.x, 0.0) && close(pan.y, 0.0));
        // Tiny bounds cap at 3x, huge ones floor at 0.1x.
        assert!(close(frame_bounds(Rect::from_center_size(Pos2::new(10.0, 0.0), Vec2::splat(2.0)), Vec2::splat(800.0)).1, ZOOM_MAX));
        assert!(close(frame_bounds(Rect::from_center_size(Pos2::ZERO, Vec2::splat(1e6)), Vec2::splat(800.0)).1, ZOOM_MIN));
        // Off-centre bounds pan to their centre.
        let (pan, _) = frame_bounds(Rect::from_center_size(Pos2::new(40.0, 20.0), Vec2::splat(10.0)), Vec2::splat(400.0));
        assert!(close(pan.x, -40.0) && close(pan.y, -20.0));
        assert_eq!(bounds_of([]), None);
        assert_eq!(bounds_of([b, Rect::from_center_size(Pos2::new(300.0, 0.0), Vec2::splat(10.0))]).unwrap().max.x, 305.0);
    }

    #[test]
    fn minimal_scroll_for_live_follow() {
        let view = Rect::from_min_size(Pos2::ZERO, Vec2::new(400.0, 300.0));
        let inside = Rect::from_center_size(Pos2::new(200.0, 150.0), Vec2::new(50.0, 30.0));
        assert_eq!(scroll_to_keep_visible(view, inside, 30.0), Vec2::ZERO);
        let right = Rect::from_center_size(Pos2::new(390.0, 150.0), Vec2::new(50.0, 30.0));
        // Right edge 415 must reach 370.
        assert_eq!(scroll_to_keep_visible(view, right, 30.0), Vec2::new(-45.0, 0.0));
        let above = Rect::from_center_size(Pos2::new(200.0, 10.0), Vec2::new(50.0, 30.0));
        assert_eq!(scroll_to_keep_visible(view, above, 30.0), Vec2::new(0.0, 35.0));
    }

    #[test]
    fn live_follow_switches_machines_halfway() {
        let p = |c: &str, s: Option<&str>, t, pr, any| follow_machine_path(c, s, t, pr, any);
        assert_eq!(p("Idle", None, false, 0.0, false), Vec::<String>::new());
        assert_eq!(p("Combat/Swing", None, false, 0.0, false), vec!["Combat".to_owned()]);
        // Idle -> Combat/Swing: stays on the source's level below 0.5.
        assert!(p("Combat/Swing", Some("Idle"), true, 0.4, false).is_empty());
        assert_eq!(p("Combat/Swing", Some("Idle"), true, 0.5, false), vec!["Combat".to_owned()]);
        // Any State transitions jump immediately.
        assert_eq!(p("Combat/Swing", Some("Idle"), true, 0.0, true), vec!["Combat".to_owned()]);
        assert_eq!(p("A/B/C", None, false, 0.0, false), vec!["A".to_owned(), "B".to_owned()]);
    }

    #[test]
    fn node_strip_progress() {
        assert!(close(node_progress(2.25, true), 0.25));
        assert!(close(node_progress(-0.25, true), 0.75));
        assert!(close(node_progress(1.7, false), 1.0));
        assert!(close(node_progress(-1.0, false), 0.0));
    }
}

#[cfg(test)]
mod blend_tests {
    use super::*;
    use zestengine_core::BlendTreeKind as K;

    #[test]
    fn dragged_threshold_resorts() {
        assert_eq!(sorted_slot(&[0.0, 1.0, 2.0], 0), 0);
        assert_eq!(sorted_slot(&[1.5, 1.0, 2.0], 0), 1);
        assert_eq!(sorted_slot(&[3.0, 1.0, 2.0], 0), 2);
        assert_eq!(sorted_slot(&[0.0, 1.0, -1.0], 2), 0);
        let mut v = vec!['a', 'b', 'c'];
        move_item(&mut v, 0, 2);
        assert_eq!(v, ['b', 'c', 'a']);
    }

    #[test]
    fn thresholds_snap_to_neighbours() {
        assert_eq!(snap_threshold(1.04, [1.0, 2.0], 0.05), 1.0);
        assert_eq!(snap_threshold(1.5, [1.0, 2.0], 0.05), 1.5);
    }

    #[test]
    fn tents_span_the_neighbours() {
        let t = [0.0, 1.0, 3.0];
        assert_eq!(triangle_1d(&t, 0), (0.0, 0.0, 1.0));
        assert_eq!(triangle_1d(&t, 1), (0.0, 1.0, 3.0));
        assert_eq!(triangle_1d(&t, 2), (1.0, 3.0, 3.0));
        assert_eq!(range_1d(&[2.0]), (1.5, 2.5));
    }

    #[test]
    fn views_follow_the_kind() {
        let (c, h) = view_2d(true, &[[1.0, 0.0], [-2.0, 1.0]]);
        assert_eq!(c, [0.0, 0.0]);
        assert!((h - 2.7).abs() < 1e-5);
        let (c, h) = view_2d(false, &[[2.0, 2.0], [4.0, 3.0]]);
        assert_eq!(c, [3.0, 2.5]);
        assert!((h - 1.35).abs() < 1e-5);
        assert_eq!(view_2d(true, &[]), ([0.0, 0.0], 1.0));
    }

    #[test]
    fn diagram_mapping_round_trips_with_y_up() {
        let r = Rect::from_min_size(Pos2::new(10.0, 20.0), Vec2::splat(200.0));
        let s = to_diagram([1.0, 1.0], [0.0, 0.0], 2.0, r);
        assert_eq!(s, Pos2::new(160.0, 70.0));
        let back = from_diagram(s, [0.0, 0.0], 2.0, r);
        assert!((back[0] - 1.0).abs() < 1e-5 && (back[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn warnings() {
        assert!(blend_warnings(K::Simple1D, &[[0.0; 2], [0.0; 2]]).is_empty());
        assert_eq!(blend_warnings(K::FreeformCartesian2D, &[[1.0, 1.0], [1.0, 1.0]]), [BlendWarning::TooClose]);
        // Only two opposite directions: a 180 degree gap.
        let w = blend_warnings(K::SimpleDirectional2D, &[[1.0, 0.0], [-1.0, 0.0]]);
        assert_eq!(w, [BlendWarning::DirectionsTooWide]);
        // Four around the circle plus origin is fine.
        let ok = blend_warnings(K::SimpleDirectional2D, &[[0.0; 2], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]]);
        assert!(ok.is_empty());
        // Almost the same direction.
        let near = blend_warnings(K::SimpleDirectional2D, &[[1.0, 0.0], [1.0, 0.01], [-1.0, -1.0]]);
        assert!(near.contains(&BlendWarning::TooClose));
        assert_eq!(blend_warnings(K::FreeformDirectional2D, &[[1.0, 0.0], [0.0, 1.0]]), [BlendWarning::NoOriginMotion]);
        assert!(blend_warnings(K::FreeformDirectional2D, &[[0.0, 0.0], [0.0, 1.0]]).is_empty());
    }

    #[test]
    fn compute_positions() {
        assert_eq!(spread_thresholds(3, 0.0, 2.0), [0.0, 1.0, 2.0]);
        assert_eq!(spread_thresholds(1, 4.0, 9.0), [4.0]);
        let r = ring_positions(5, 2.0, true);
        assert_eq!(r.len(), 5);
        assert_eq!(r[0], [0.0, 0.0]);
        assert!(r[1..].iter().all(|p| ((p[0] * p[0] + p[1] * p[1]).sqrt() - 2.0).abs() < 1e-4));
        assert!(ring_positions(0, 1.0, true).is_empty());
    }

    #[test]
    fn field_samples_cell_centres() {
        let f = weight_field(2, [0.0, 0.0], 1.0, |p| p[0] + 10.0 * p[1]);
        assert_eq!(f, [-0.5 + 5.0, 0.5 + 5.0, -0.5 - 5.0, 0.5 - 5.0]);
        assert!((weight_radius(0.25) - 10.0).abs() < 1e-5);
    }
}
