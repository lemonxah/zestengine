//! Pure geometry of sprite shapes: bezier tessellation with an error bound and
//! a point cap, polygon area / orientation, ear-clipping triangulation, polygon
//! offset and Douglas-Peucker simplification. No scene, no GPU.

use glam::Vec2;

use super::ShapePoint;

/// How curved segments are cut into line pieces.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum TessMode {
    /// Subdivide until the polyline stays within `tolerance` of the curve.
    Adaptive {
        /// Largest allowed distance between the curve and its polyline.
        tolerance: f32,
    },
    /// `detail` equal parameter steps per curved segment.
    Fixed {
        /// Pieces per curved segment.
        detail: u32,
    },
}

/// The tessellated spline. Segment `i` runs `pts[i] -> pts[i + 1]` (and, for a
/// closed polyline, the last point back to the first).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Poly {
    /// The polyline's points.
    pub pts: Vec<Vec2>,
    /// Interpolated point height at every vertex.
    pub height: Vec<f32>,
    /// The control point a vertex is, `None` for points inside a curve.
    pub ctrl: Vec<Option<usize>>,
    /// Per segment, the control point whose segment it belongs to (its sprite
    /// variant, height owner).
    pub owner: Vec<usize>,
    /// Closed (a ring) or open (a strip).
    pub closed: bool,
}

impl Poly {
    /// Number of points.
    pub fn len(&self) -> usize {
        self.pts.len()
    }
    /// True when the polyline has no points.
    pub fn is_empty(&self) -> bool {
        self.pts.is_empty()
    }
    /// Number of segments (a closed ring has one per point).
    pub fn segment_count(&self) -> usize {
        if self.pts.len() < 2 { 0 } else if self.closed { self.pts.len() } else { self.pts.len() - 1 }
    }
    /// End vertex of segment `s`.
    pub fn next(&self, s: usize) -> usize {
        (s + 1) % self.pts.len()
    }
    /// Total length of the polyline.
    pub fn length(&self) -> f32 {
        (0..self.segment_count()).map(|s| self.pts[s].distance(self.pts[self.next(s)])).sum()
    }
    /// The same polyline walked backwards (owners and control flags follow).
    pub fn reversed(&self) -> Poly {
        let n = self.pts.len();
        let segs = self.segment_count();
        let owner = (0..segs)
            .map(|j| {
                // Reversed segment j runs old vertex n-1-j -> n-2-j, i.e. old segment n-2-j (mod n).
                let old = if self.closed { (2 * n - 2 - j) % n } else { n - 2 - j };
                self.owner[old]
            })
            .collect();
        Poly {
            pts: self.pts.iter().rev().copied().collect(),
            height: self.height.iter().rev().copied().collect(),
            ctrl: self.ctrl.iter().rev().copied().collect(),
            owner,
            closed: self.closed,
        }
    }
}

fn v(a: [f32; 2]) -> Vec2 {
    Vec2::from(a)
}

/// The four bezier points of the segment from control point `a` to `b`.
pub fn segment_bezier(a: &ShapePoint, b: &ShapePoint) -> [Vec2; 4] {
    let (p0, p3) = (v(a.position), v(b.position));
    [p0, p0 + v(a.out_handle()), p3 + v(b.in_handle()), p3]
}

/// A cubic bezier point at parameter `t` (0..1).
pub fn bezier_at(c: &[Vec2; 4], t: f32) -> Vec2 {
    let u = 1.0 - t;
    c[0] * (u * u * u) + c[1] * (3.0 * u * u * t) + c[2] * (3.0 * u * t * t) + c[3] * (t * t * t)
}

fn is_straight(c: &[Vec2; 4]) -> bool {
    c[0].distance_squared(c[1]) < 1e-12 && c[2].distance_squared(c[3]) < 1e-12
}

fn dist_to_line(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let l2 = ab.length_squared();
    if l2 < 1e-18 {
        return p.distance(a);
    }
    ((p - a).perp_dot(ab)).abs() / l2.sqrt()
}

const MAX_DEPTH: u32 = 18;

/// Flatten one bezier into `(point, t)` pieces after the start point. The
/// control polygon's distance from the chord bounds the curve's by 3/4, so the
/// test `0.75 * max(d1, d2) <= tol` keeps every piece within `tol` of the curve.
pub fn flatten_bezier(c: &[Vec2; 4], tol: f32, out: &mut Vec<(Vec2, f32)>) {
    fn rec(c: [Vec2; 4], t0: f32, t1: f32, tol: f32, depth: u32, out: &mut Vec<(Vec2, f32)>) {
        let d = dist_to_line(c[1], c[0], c[3]).max(dist_to_line(c[2], c[0], c[3]));
        if depth >= MAX_DEPTH || 0.75 * d <= tol {
            out.push((c[3], t1));
            return;
        }
        let (a01, a12, a23) = ((c[0] + c[1]) * 0.5, (c[1] + c[2]) * 0.5, (c[2] + c[3]) * 0.5);
        let (b0, b1) = ((a01 + a12) * 0.5, (a12 + a23) * 0.5);
        let m = (b0 + b1) * 0.5;
        let tm = (t0 + t1) * 0.5;
        rec([c[0], a01, b0, m], t0, tm, tol, depth + 1, out);
        rec([m, b1, a23, c[3]], tm, t1, tol, depth + 1, out);
    }
    if is_straight(c) {
        out.push((c[3], 1.0));
        return;
    }
    rec(*c, 0.0, 1.0, tol.max(1e-6), 0, out);
}

/// Equal parameter steps.
pub fn flatten_fixed(c: &[Vec2; 4], detail: u32, out: &mut Vec<(Vec2, f32)>) {
    if is_straight(c) {
        out.push((c[3], 1.0));
        return;
    }
    let n = detail.max(1);
    for i in 1..=n {
        let t = i as f32 / n as f32;
        out.push((if i == n { c[3] } else { bezier_at(c, t) }, t));
    }
}

/// Tessellate a spline. Straight segments stay one piece. `max_points` bounds
/// the vertex count: the adaptive tolerance is doubled (and the fixed detail
/// reduced) until the result fits, down to one piece per segment, so the cap
/// can only be exceeded by a spline with more control points than the cap.
/// Vertices closer than 1e-5 are merged.
pub fn tessellate(points: &[ShapePoint], closed: bool, mode: TessMode, max_points: u32) -> Poly {
    let n = points.len();
    if n == 0 {
        return Poly { closed, ..Default::default() };
    }
    if n == 1 {
        return Poly { pts: vec![v(points[0].position)], height: vec![points[0].height], ctrl: vec![Some(0)], owner: vec![], closed };
    }
    let segs = if closed { n } else { n - 1 };
    let curves: Vec<[Vec2; 4]> = (0..segs).map(|k| segment_bezier(&points[k], &points[(k + 1) % n])).collect();
    let cap = (max_points as usize).max(segs + 1);
    type Piece = (Vec2, f32);
    let build = |pieces: &dyn Fn(&[Vec2; 4], &mut Vec<Piece>)| -> Vec<Vec<Piece>> {
        curves
            .iter()
            .map(|c| {
                let mut o = Vec::new();
                pieces(c, &mut o);
                o
            })
            .collect()
    };
    let count = |s: &[Vec<(Vec2, f32)>]| s.iter().map(Vec::len).sum::<usize>() + 1 - closed as usize;
    let pieces = match mode {
        TessMode::Adaptive { tolerance } => {
            let mut tol = tolerance.max(1e-6);
            loop {
                let s = build(&|c, o| flatten_bezier(c, tol, o));
                if count(&s) <= cap || tol > 1e7 {
                    break s;
                }
                tol *= 2.0;
            }
        }
        TessMode::Fixed { detail } => {
            let mut d = detail.max(1);
            loop {
                let s = build(&|c, o| flatten_fixed(c, d, o));
                if count(&s) <= cap || d == 1 {
                    break s;
                }
                // Scale down in proportion, at least by one.
                let over = count(&s) as f32 / cap as f32;
                d = ((d as f32 / over).floor() as u32).clamp(1, d - 1);
            }
        }
    };
    let mut poly = Poly { closed, ..Default::default() };
    let mut arrive: Vec<usize> = Vec::new(); // owner of the segment arriving at each vertex
    let push = |poly: &mut Poly, p: Vec2, h: f32, ctrl: Option<usize>, owner: usize, arrive: &mut Vec<usize>| {
        if let Some(&last) = poly.pts.last()
            && last.distance_squared(p) < 1e-10
        {
            if poly.ctrl.last() == Some(&None) {
                *poly.ctrl.last_mut().unwrap() = ctrl;
            }
            return;
        }
        poly.pts.push(p);
        poly.height.push(h);
        poly.ctrl.push(ctrl);
        arrive.push(owner);
    };
    push(&mut poly, curves[0][0], points[0].height, Some(0), 0, &mut arrive);
    let mut closing_owner = 0;
    for (k, ps) in pieces.iter().enumerate() {
        let (ha, hb) = (points[k].height, points[(k + 1) % n].height);
        for (i, &(p, t)) in ps.iter().enumerate() {
            let last = i + 1 == ps.len();
            if closed && k + 1 == segs && last {
                closing_owner = k; // the wrap-around back to point 0: no new vertex
                continue;
            }
            let ctrl = last.then_some((k + 1) % n);
            push(&mut poly, p, ha + (hb - ha) * t, ctrl, k, &mut arrive);
        }
    }
    // Closing vertex coincident with the first one (a zero-length last curve).
    if closed && poly.pts.len() > 1 && poly.pts[0].distance_squared(*poly.pts.last().unwrap()) < 1e-10 {
        poly.pts.pop();
        poly.height.pop();
        poly.ctrl.pop();
        arrive.pop();
    }
    let m = poly.pts.len();
    let segs_out = if m < 2 { 0 } else if closed { m } else { m - 1 };
    poly.owner = (0..segs_out).map(|j| if j + 1 < m { arrive[j + 1] } else { closing_owner }).collect();
    poly
}

// ---------------------------------------------------------------------------
// Polygons
// ---------------------------------------------------------------------------

/// Shoelace area, positive for counter-clockwise rings.
pub fn signed_area(p: &[Vec2]) -> f32 {
    let n = p.len();
    if n < 3 {
        return 0.0;
    }
    let mut a = 0.0f64;
    for i in 0..n {
        let (u, w) = (p[i], p[(i + 1) % n]);
        a += u.x as f64 * w.y as f64 - w.x as f64 * u.y as f64;
    }
    (a * 0.5) as f32
}

fn cross(o: Vec2, a: Vec2, b: Vec2) -> f32 {
    (a - o).perp_dot(b - o)
}

fn in_triangle(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
    let (d1, d2, d3) = (cross(a, b, p), cross(b, c, p), cross(c, a, p));
    d1 >= 0.0 && d2 >= 0.0 && d3 >= 0.0
}

/// Ear-clipping triangulation of a simple polygon (no holes) in either
/// orientation. Returns CCW triangles as indices into `pts`. Collinear vertices
/// are dropped; a self-intersecting ring still terminates (the least-bad ear is
/// clipped) but its fill is wrong where it crosses itself.
pub fn triangulate(pts: &[Vec2]) -> Vec<[u32; 3]> {
    let n = pts.len();
    if n < 3 {
        return Vec::new();
    }
    let mut idx: Vec<u32> = (0..n as u32).collect();
    if signed_area(pts) < 0.0 {
        idx.reverse();
    }
    // Drop vertices that add nothing: duplicates and collinear points.
    let scale = pts.iter().fold(0.0f32, |m, p| m.max(p.x.abs()).max(p.y.abs())).max(1.0);
    let eps = 1e-7 * scale * scale;
    loop {
        let m = idx.len();
        if m < 3 {
            return Vec::new();
        }
        let before = m;
        let mut i = 0;
        while i < idx.len() {
            let m = idx.len();
            let (a, b, c) = (pts[idx[(i + m - 1) % m] as usize], pts[idx[i] as usize], pts[idx[(i + 1) % m] as usize]);
            if cross(a, b, c).abs() <= eps && (b - a).dot(c - b) >= 0.0 {
                idx.remove(i);
            } else {
                i += 1;
            }
            if idx.len() < 3 {
                return Vec::new();
            }
        }
        if idx.len() == before {
            break;
        }
    }
    let mut tris = Vec::with_capacity(idx.len().saturating_sub(2));
    let mut i = 0;
    let mut stalled = 0;
    while idx.len() > 3 {
        let m = idx.len();
        i %= m;
        let (ia, ib, ic) = (idx[(i + m - 1) % m], idx[i], idx[(i + 1) % m]);
        let (a, b, c) = (pts[ia as usize], pts[ib as usize], pts[ic as usize]);
        let convex = cross(a, b, c) > eps;
        let ear = convex && !idx.iter().any(|&k| k != ia && k != ib && k != ic && {
            let p = pts[k as usize];
            // Vertices coincident with a corner (a pinched ring) don't block.
            p != a && p != b && p != c && in_triangle(p, a, b, c)
        });
        if ear {
            tris.push([ia, ib, ic]);
            idx.remove(i);
            stalled = 0;
        } else {
            i += 1;
            stalled += 1;
            if stalled > m {
                // No ear anywhere: the ring crosses itself. Clip the sharpest convex corner.
                let best = (0..m)
                    .filter(|&k| cross(pts[idx[(k + m - 1) % m] as usize], pts[idx[k] as usize], pts[idx[(k + 1) % m] as usize]) > 0.0)
                    .max_by(|&x, &y| {
                        let f = |k: usize| cross(pts[idx[(k + m - 1) % m] as usize], pts[idx[k] as usize], pts[idx[(k + 1) % m] as usize]);
                        f(x).total_cmp(&f(y))
                    })
                    .unwrap_or(0);
                let (ia, ib, ic) = (idx[(best + m - 1) % m], idx[best], idx[(best + 1) % m]);
                if cross(pts[ia as usize], pts[ib as usize], pts[ic as usize]) > 0.0 {
                    tris.push([ia, ib, ic]);
                }
                idx.remove(best);
                stalled = 0;
            }
        }
    }
    if idx.len() == 3 && cross(pts[idx[0] as usize], pts[idx[1] as usize], pts[idx[2] as usize]) > 0.0 {
        tris.push([idx[0], idx[1], idx[2]]);
    }
    tris
}

// ---- polygons with holes -------------------------------------------------------------------

/// Even-odd point in ring.
pub fn point_in_ring(ring: &[Vec2], p: Vec2) -> bool {
    let n = ring.len();
    let mut inside = false;
    let mut j = n.wrapping_sub(1);
    for i in 0..n {
        let (a, b) = (ring[i], ring[j]);
        if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Whether segments `ab` and `cd` cross at a point interior to both (touching at shared
/// endpoints or collinear overlap are not crossings here).
fn proper_cross(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> bool {
    let (d1, d2) = (cross(a, b, c), cross(a, b, d));
    let (d3, d4) = (cross(c, d, a), cross(c, d, b));
    ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
}

/// Whether the diagonal from hole vertex `m` to ring vertex `v` stays inside the filled region:
/// it crosses no edge of the working ring or of the hole itself, touches no other vertex, and its
/// midpoint is inside the ring but not inside the hole.
fn bridge_ok(ring: &[Vec2], hole: &[Vec2], m: Vec2, v: Vec2) -> bool {
    let scale = (m - v).length().max(1e-9);
    let near_seg = |q: Vec2| {
        if q == m || q == v {
            return false;
        }
        let ab = v - m;
        let t = ((q - m).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
        q.distance(m + ab * t) < 1e-6 * scale.max(1.0)
    };
    for set in [ring, hole] {
        let n = set.len();
        for i in 0..n {
            let (c, d) = (set[i], set[(i + 1) % n]);
            if proper_cross(m, v, c, d) || near_seg(c) {
                return false;
            }
        }
    }
    let mid = (m + v) * 0.5;
    point_in_ring(ring, mid) && !point_in_ring(hole, mid)
}

/// Triangulate `outer` minus `holes`: each hole is joined to the outer ring by a zero-width
/// bridge (two coincident edges), holes taken left to right, and the single weakly simple ring
/// is ear clipped. A hole with no valid bridge (outside the outer ring, crossing it) is
/// skipped. Returns the ring's vertices (bridge ends duplicated) and CCW triangles into them,
/// so the triangle area equals the outer area minus the holes' (for holes that do not overlap
/// each other).
pub fn triangulate_with_holes(outer: &[Vec2], holes: &[Vec<Vec2>]) -> (Vec<Vec2>, Vec<[u32; 3]>) {
    let mut ring: Vec<Vec2> = outer.to_vec();
    if signed_area(&ring) < 0.0 {
        ring.reverse();
    }
    // Holes clockwise, leftmost first.
    let mut hs: Vec<Vec<Vec2>> = holes.iter().filter(|h| h.len() >= 3 && signed_area(h).abs() > 1e-12).cloned().collect();
    for h in &mut hs {
        if signed_area(h) > 0.0 {
            h.reverse();
        }
    }
    let leftmost = |h: &Vec<Vec2>| h.iter().enumerate().fold((0usize, f32::INFINITY), |a, (i, p)| if p.x < a.1 { (i, p.x) } else { a });
    hs.sort_by(|a, b| leftmost(a).1.total_cmp(&leftmost(b).1));
    for hole in &hs {
        let (li, _) = leftmost(hole);
        let m = hole[li];
        // Nearest ring vertices first; the first that sees the hole's vertex wins.
        let mut order: Vec<usize> = (0..ring.len()).collect();
        order.sort_by(|&a, &b| ring[a].distance_squared(m).total_cmp(&ring[b].distance_squared(m)));
        let Some(bi) = order.into_iter().find(|&i| bridge_ok(&ring, hole, m, ring[i])) else { continue };
        let mut merged = Vec::with_capacity(ring.len() + hole.len() + 2);
        merged.extend_from_slice(&ring[..=bi]);
        for k in 0..=hole.len() {
            merged.push(hole[(li + k) % hole.len()]);
        }
        merged.extend_from_slice(&ring[bi..]);
        ring = merged;
    }
    let tris = triangulate(&ring);
    (ring, tris)
}

/// Move every vertex of a ring along its outward miter normal by `d`
/// (positive grows the polygon, either orientation). The miter is limited to
/// twice the distance so a sharp spike cannot shoot off.
pub fn offset_ring(pts: &[Vec2], d: f32) -> Vec<Vec2> {
    let n = pts.len();
    if n < 3 || d == 0.0 {
        return pts.to_vec();
    }
    let ccw = signed_area(pts) >= 0.0;
    // Outward normal of an edge: right of travel for CCW, left for CW.
    let out = |a: Vec2, b: Vec2| -> Vec2 {
        let e = (b - a).normalize_or_zero();
        if ccw { Vec2::new(e.y, -e.x) } else { Vec2::new(-e.y, e.x) }
    };
    (0..n)
        .map(|i| {
            let (p, a, b) = (pts[i], pts[(i + n - 1) % n], pts[(i + 1) % n]);
            let (n0, n1) = (out(a, p), out(p, b));
            let denom = 1.0 + n0.dot(n1);
            let mut m = if denom > 1e-3 { (n0 + n1) / denom } else { n0 };
            if m.length() > 2.0 {
                m = m.normalize() * 2.0;
            }
            p + m * d
        })
        .collect()
}

fn seg_dist(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let l2 = ab.length_squared();
    if l2 < 1e-18 {
        return p.distance(a);
    }
    let t = ((p - a).dot(ab) / l2).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

/// Distance from `p` to the polyline `pts` (closed = also the last-to-first edge).
pub fn distance_to_polyline(p: Vec2, pts: &[Vec2], closed: bool) -> f32 {
    let n = pts.len();
    match n {
        0 => f32::INFINITY,
        1 => p.distance(pts[0]),
        _ => (0..if closed { n } else { n - 1 }).map(|i| seg_dist(p, pts[i], pts[(i + 1) % n])).fold(f32::INFINITY, f32::min),
    }
}

/// Douglas-Peucker. An open polyline keeps both ends; a closed ring is split at
/// its two farthest-apart vertices so it never collapses. Every dropped vertex
/// stays within `tol` of the result.
pub fn simplify(pts: &[Vec2], tol: f32, closed: bool) -> Vec<Vec2> {
    let n = pts.len();
    if tol <= 0.0 || n < 3 {
        return pts.to_vec();
    }
    fn dp(pts: &[Vec2], lo: usize, hi: usize, tol: f32, keep: &mut [bool]) {
        if hi <= lo + 1 {
            return;
        }
        let (mut far, mut fd) = (lo, 0.0f32);
        for i in lo + 1..hi {
            let d = seg_dist(pts[i], pts[lo], pts[hi]);
            if d > fd {
                (far, fd) = (i, d);
            }
        }
        if fd > tol {
            keep[far] = true;
            dp(pts, lo, far, tol, keep);
            dp(pts, far, hi, tol, keep);
        }
    }
    let mut keep = vec![false; n];
    if !closed {
        keep[0] = true;
        keep[n - 1] = true;
        dp(pts, 0, n - 1, tol, &mut keep);
    } else {
        // Anchor: vertex 0 and the vertex farthest from it.
        let far = (1..n).max_by(|&a, &b| pts[a].distance_squared(pts[0]).total_cmp(&pts[b].distance_squared(pts[0]))).unwrap();
        keep[0] = true;
        keep[far] = true;
        dp(pts, 0, far, tol, &mut keep);
        let mut tail: Vec<Vec2> = pts[far..].to_vec();
        tail.push(pts[0]);
        let mut tk = vec![false; tail.len()];
        tk[0] = true;
        *tk.last_mut().unwrap() = true;
        dp(&tail, 0, tail.len() - 1, tol, &mut tk);
        for (k, &flag) in tk.iter().enumerate().take(tail.len() - 1) {
            if flag {
                keep[far + k] = true;
            }
        }
    }
    pts.iter().zip(&keep).filter(|(_, k)| **k).map(|(p, _)| *p).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sprite_shape::ShapeTangentMode;

    fn curve_pts() -> Vec<ShapePoint> {
        let mut a = ShapePoint::at(0.0, 0.0);
        a.mode = ShapeTangentMode::Broken;
        a.tangent_out = [0.0, 4.0];
        let mut b = ShapePoint::at(6.0, 0.0);
        b.mode = ShapeTangentMode::Broken;
        b.tangent_in = [0.0, 4.0];
        vec![a, b]
    }

    #[test]
    fn straight_segments_are_not_subdivided() {
        let pts = vec![ShapePoint::at(0.0, 0.0), ShapePoint::at(3.0, 0.0), ShapePoint::at(3.0, 4.0)];
        for mode in [TessMode::Adaptive { tolerance: 0.001 }, TessMode::Fixed { detail: 16 }] {
            let p = tessellate(&pts, false, mode, 1000);
            assert_eq!(p.pts.len(), 3);
            assert_eq!(p.owner, vec![0, 1]);
            assert_eq!(p.ctrl, vec![Some(0), Some(1), Some(2)]);
            assert!((p.length() - 7.0).abs() < 1e-5);
        }
    }

    #[test]
    fn adaptive_tessellation_respects_its_error_bound() {
        let pts = curve_pts();
        let c = segment_bezier(&pts[0], &pts[1]);
        for tol in [0.2, 0.05, 0.01, 0.002] {
            let p = tessellate(&pts, false, TessMode::Adaptive { tolerance: tol }, 100_000);
            // Every point of the true curve is within tol of the polyline.
            let worst = (0..=2000).map(|i| distance_to_polyline(bezier_at(&c, i as f32 / 2000.0), &p.pts, false)).fold(0.0f32, f32::max);
            assert!(worst <= tol * 1.001 + 1e-5, "tol {tol}: deviation {worst}");
            // ... and the tolerance is used, not wasted: tighter tolerance, more points.
            assert!(p.pts.len() >= 3);
        }
        let coarse = tessellate(&pts, false, TessMode::Adaptive { tolerance: 0.2 }, 100_000).pts.len();
        let fine = tessellate(&pts, false, TessMode::Adaptive { tolerance: 0.002 }, 100_000).pts.len();
        assert!(fine > coarse * 3, "{coarse} -> {fine}");
    }

    #[test]
    fn point_cap_is_honoured_for_both_modes() {
        let pts = curve_pts();
        for cap in [3u32, 8, 20, 64] {
            let a = tessellate(&pts, false, TessMode::Adaptive { tolerance: 1e-5 }, cap);
            assert!(a.pts.len() <= cap as usize, "adaptive cap {cap}: {}", a.pts.len());
            let f = tessellate(&pts, false, TessMode::Fixed { detail: 1000 }, cap);
            assert!(f.pts.len() <= cap as usize, "fixed cap {cap}: {}", f.pts.len());
        }
        // A cap below the control-point count still yields one piece per segment.
        let mut many: Vec<ShapePoint> = (0..10).map(|i| ShapePoint::at(i as f32, 0.0)).collect();
        many[3].mode = ShapeTangentMode::Broken;
        assert_eq!(tessellate(&many, false, TessMode::Fixed { detail: 16 }, 2).pts.len(), 10);
        // Closed rings count without the wrap vertex.
        let mut ring = curve_pts();
        ring.push(ShapePoint::at(3.0, -5.0));
        let r = tessellate(&ring, true, TessMode::Fixed { detail: 16 }, 40);
        assert!(r.pts.len() <= 40 && r.segment_count() == r.pts.len());
    }

    #[test]
    fn heights_interpolate_along_curves() {
        let mut pts = curve_pts();
        pts[0].height = 1.0;
        pts[1].height = 3.0;
        let p = tessellate(&pts, false, TessMode::Fixed { detail: 4 }, 100);
        assert_eq!(p.height.len(), 5);
        assert!((p.height[0] - 1.0).abs() < 1e-6 && (p.height[2] - 2.0).abs() < 1e-6 && (p.height[4] - 3.0).abs() < 1e-6);
        assert!(p.height.windows(2).all(|w| w[1] > w[0]));
    }

    #[test]
    fn closed_ring_has_a_closing_segment_and_owners() {
        let pts = vec![ShapePoint::at(0.0, 0.0), ShapePoint::at(4.0, 0.0), ShapePoint::at(4.0, 4.0), ShapePoint::at(0.0, 4.0)];
        let p = tessellate(&pts, true, TessMode::Adaptive { tolerance: 0.01 }, 100);
        assert_eq!(p.pts.len(), 4);
        assert_eq!(p.segment_count(), 4);
        assert_eq!(p.owner, vec![0, 1, 2, 3]);
        assert!((p.length() - 16.0).abs() < 1e-5);
        let r = p.reversed();
        assert_eq!(r.pts[0], p.pts[3]);
        // Reversed segment j runs old vertex 3-j -> 2-j: old segment (2-j) mod 4.
        assert_eq!(r.owner, vec![2, 1, 0, 3]);
        assert!((r.length() - 16.0).abs() < 1e-5);
        let open = tessellate(&pts, false, TessMode::Adaptive { tolerance: 0.01 }, 100).reversed();
        assert_eq!(open.owner, vec![2, 1, 0]);
    }

    #[test]
    fn coincident_points_merge() {
        let pts = vec![ShapePoint::at(0.0, 0.0), ShapePoint::at(0.0, 0.0), ShapePoint::at(2.0, 0.0)];
        let p = tessellate(&pts, false, TessMode::Fixed { detail: 4 }, 100);
        assert_eq!(p.pts.len(), 2);
        assert_eq!(p.ctrl[1], Some(2));
        assert_eq!(p.owner.len(), 1);
        let one = tessellate(&pts[..1], false, TessMode::Fixed { detail: 4 }, 100);
        assert_eq!((one.pts.len(), one.segment_count()), (1, 0));
    }

    fn poly_area(p: &[Vec2], tris: &[[u32; 3]]) -> f32 {
        tris.iter().map(|t| cross(p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]) * 0.5).sum()
    }

    #[test]
    fn triangulation_area_matches_shoelace() {
        let v = |a: &[[f32; 2]]| a.iter().map(|&p| Vec2::from(p)).collect::<Vec<_>>();
        let square = v(&[[0., 0.], [4., 0.], [4., 4.], [0., 4.]]);
        let l_shape = v(&[[0., 0.], [4., 0.], [4., 2.], [2., 2.], [2., 4.], [0., 4.]]);
        let comb = v(&[[0., 0.], [9., 0.], [9., 3.], [7., 3.], [7., 1.], [5., 1.], [5., 3.], [3., 3.], [3., 1.], [1., 1.], [1., 3.], [0., 3.]]);
        let with_collinear = v(&[[0., 0.], [2., 0.], [4., 0.], [4., 4.], [0., 4.], [0., 2.]]);
        for (name, poly) in [("square", square), ("L", l_shape), ("comb", comb), ("collinear", with_collinear)] {
            let want = signed_area(&poly);
            let t = triangulate(&poly);
            assert!((poly_area(&poly, &t) - want).abs() < 1e-4, "{name} ccw: {} vs {want}", poly_area(&poly, &t));
            let mut cw = poly.clone();
            cw.reverse();
            let t = triangulate(&cw);
            assert!((poly_area(&cw, &t) - want.abs()).abs() < 1e-4, "{name} cw");
            assert!(t.iter().all(|tri| cross(cw[tri[0] as usize], cw[tri[1] as usize], cw[tri[2] as usize]) > 0.0), "{name}: all triangles CCW");
        }
        // A tessellated blob (open curve closed by a straight segment) - the realistic input.
        let mut pts = vec![ShapePoint::at(-3.0, 0.0), ShapePoint::at(0.0, 3.0), ShapePoint::at(3.0, 0.0), ShapePoint::at(0.0, -3.0)];
        for p in pts.iter_mut() {
            let r = Vec2::from(p.position);
            p.set_mode(ShapeTangentMode::Continuous, [r.y, -r.x], 1.6); // clockwise round blob
        }
        let blob = tessellate(&pts, true, TessMode::Adaptive { tolerance: 0.01 }, 1000);
        let t = triangulate(&blob.pts);
        assert!(blob.pts.len() > 12);
        assert!((poly_area(&blob.pts, &t) - signed_area(&blob.pts).abs()).abs() < 1e-3, "CCW triangles sum to the ring's absolute area");
    }

    #[test]
    fn triangulation_survives_degenerate_and_crossing_input() {
        assert!(triangulate(&[]).is_empty());
        assert!(triangulate(&[Vec2::ZERO, Vec2::X]).is_empty());
        assert!(triangulate(&[Vec2::ZERO, Vec2::X, Vec2::new(2.0, 0.0)]).is_empty(), "collinear");
        // A bow-tie terminates (the fill is wrong where it crosses, but it returns).
        let bow = [Vec2::new(0., 0.), Vec2::new(2., 2.), Vec2::new(2., 0.), Vec2::new(0., 2.)];
        let _ = triangulate(&bow);
        // Duplicate vertices.
        let dup = [Vec2::new(0., 0.), Vec2::new(0., 0.), Vec2::new(3., 0.), Vec2::new(3., 3.), Vec2::new(0., 3.)];
        let t = triangulate(&dup);
        assert!((poly_area(&dup, &t) - 9.0).abs() < 1e-4);
    }

    #[test]
    fn triangulation_scales_to_a_thousand_points() {
        let n = 1500;
        let ring: Vec<Vec2> = (0..n)
            .map(|i| {
                let a = i as f32 / n as f32 * std::f32::consts::TAU;
                let r = 10.0 + 2.0 * (a * 7.0).sin();
                Vec2::new(a.cos() * r, a.sin() * r)
            })
            .collect();
        let t0 = std::time::Instant::now();
        let t = triangulate(&ring);
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        assert!(t.len() <= n - 2 && t.len() > n - 400, "near-collinear vertices may be dropped: {}", t.len());
        assert!((poly_area(&ring, &t) - signed_area(&ring)).abs() < 1e-2);
        eprintln!("ear clipping {n} points: {ms:.1} ms");
    }

    #[test]
    fn offset_grows_and_shrinks_either_orientation() {
        let sq = [Vec2::new(0., 0.), Vec2::new(2., 0.), Vec2::new(2., 2.), Vec2::new(0., 2.)];
        let grown = offset_ring(&sq, 0.5);
        assert!((signed_area(&grown) - 9.0).abs() < 1e-4, "{}", signed_area(&grown));
        let shrunk = offset_ring(&sq, -0.5);
        assert!((signed_area(&shrunk) - 1.0).abs() < 1e-4);
        let mut cw = sq;
        cw.reverse();
        assert!((signed_area(&offset_ring(&cw, 0.5)).abs() - 9.0).abs() < 1e-4);
        assert_eq!(offset_ring(&sq, 0.0), sq.to_vec());
    }

    #[test]
    fn simplify_keeps_every_dropped_point_within_tolerance() {
        let pts: Vec<Vec2> = (0..200).map(|i| Vec2::new(i as f32 * 0.1, (i as f32 * 0.1).sin())).collect();
        for tol in [0.0, 0.01, 0.05, 0.2] {
            let s = simplify(&pts, tol, false);
            assert_eq!(s.first(), pts.first());
            assert_eq!(s.last(), pts.last());
            if tol == 0.0 {
                assert_eq!(s.len(), pts.len());
            }
            for p in &pts {
                assert!(distance_to_polyline(*p, &s, false) <= tol + 1e-5);
            }
        }
        assert!(simplify(&pts, 0.05, false).len() < 60);
        let ring: Vec<Vec2> = (0..64).map(|i| Vec2::from_angle(i as f32 / 64.0 * std::f32::consts::TAU) * 3.0).collect();
        let s = simplify(&ring, 0.05, true);
        assert!(s.len() >= 8 && s.len() < 64);
        for p in &ring {
            assert!(distance_to_polyline(*p, &s, true) <= 0.05 + 1e-5);
        }
    }
}
