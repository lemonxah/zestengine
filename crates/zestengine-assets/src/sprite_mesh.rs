//! Tight sprite meshes and physics outlines (2D-2). Pure geometry on an alpha
//! plane: no I/O, no GPU, deterministic, so every step is unit tested.
//!
//! Pipeline (research phase5b-2d-sprites §1.3, written from the description,
//! not from any Unity source):
//!
//! 1. alpha mask of the sprite rect (`alpha > threshold`), optionally dilated
//!    by `extrude` pixels (inside the rect, so UVs never leave it);
//! 2. contour tracing on the pixel-edge lattice: every boundary edge between a
//!    solid and an empty pixel is chained into closed loops (outer boundaries
//!    positive, holes negative). Diagonal contacts are split so each loop is a
//!    simple polygon;
//! 3. closed Douglas-Peucker simplification within a pixel tolerance, retried
//!    with a smaller tolerance (finally none) if the result would self-cross;
//! 4. loops are grouped by nesting (outer / hole / island-in-hole) and
//!    triangulated with ear clipping (`earcutr`, ISC licence).
//!
//! All coordinates here are **rect-local pixels, origin at the rect's top-left,
//! y down** ([`P2`]). `sprite.rs` converts them to pivot-relative world units.

use std::collections::HashMap;

use crate::sprite::PixelRect;

/// A point in rect-local pixel space (x right, y down).
pub type P2 = [f32; 2];

/// Sprites smaller than this on either axis always get a quad (research §1.2).
pub const TIGHT_MIN_SIZE: u32 = 32;
/// Largest extrude, in pixels.
pub const MAX_EXTRUDE: u32 = 32;
/// 16-bit index budget of a sprite mesh.
pub const MAX_MESH_VERTS: usize = u16::MAX as usize;

/// The alpha plane of a whole image.
#[derive(Clone, Debug)]
pub struct AlphaMap {
    pub w: u32,
    pub h: u32,
    pub alpha: Vec<u8>,
}

impl AlphaMap {
    /// Alpha plane of tightly packed RGBA8 pixels.
    pub fn from_rgba8(w: u32, h: u32, rgba: &[u8]) -> Self {
        let alpha = rgba.as_chunks::<4>().0.iter().take(w as usize * h as usize).map(|p| p[3]).collect();
        Self { w, h, alpha }
    }

    pub fn at(&self, x: u32, y: u32) -> u8 {
        self.alpha.get(y as usize * self.w as usize + x as usize).copied().unwrap_or(0)
    }
}

/// Solid / empty pixels of one rect.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mask {
    pub w: u32,
    pub h: u32,
    pub bits: Vec<bool>,
}

impl Mask {
    /// Pixels of `rect` (clamped to the image) with `alpha > threshold`.
    pub fn from_alpha(a: &AlphaMap, rect: PixelRect, threshold: u8) -> Self {
        let r = rect.clamped(a.w, a.h);
        let mut bits = Vec::with_capacity((r.w * r.h) as usize);
        for y in 0..r.h {
            for x in 0..r.w {
                bits.push(a.at(r.x + x, r.y + y) > threshold);
            }
        }
        Self { w: r.w, h: r.h, bits }
    }

    pub fn get(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as u32) < self.w && (y as u32) < self.h && self.bits[y as usize * self.w as usize + x as usize]
    }

    pub fn solid_count(&self) -> usize {
        self.bits.iter().filter(|b| **b).count()
    }

    pub fn is_empty(&self) -> bool {
        !self.bits.iter().any(|b| *b)
    }

    /// Grow the solid area by `n` pixels (square structuring element, done as
    /// two separable passes so it is O(pixels * n)). Stays inside the rect.
    pub fn dilate(&mut self, n: u32) {
        if n == 0 || self.is_empty() {
            return;
        }
        let (w, h, n) = (self.w as usize, self.h as usize, n as usize);
        let mut tmp = vec![false; self.bits.len()];
        for y in 0..h {
            // Horizontal: nearest solid pixel within n columns.
            let row = &self.bits[y * w..(y + 1) * w];
            let mut last: Option<usize> = None;
            for x in 0..w {
                if row[x] {
                    last = Some(x);
                }
                if last.is_some_and(|l| x - l <= n) {
                    tmp[y * w + x] = true;
                }
            }
            last = None;
            for x in (0..w).rev() {
                if row[x] {
                    last = Some(x);
                }
                if last.is_some_and(|l| l - x <= n) {
                    tmp[y * w + x] = true;
                }
            }
        }
        let mut out = vec![false; self.bits.len()];
        for x in 0..w {
            let mut last: Option<usize> = None;
            for y in 0..h {
                if tmp[y * w + x] {
                    last = Some(y);
                }
                if last.is_some_and(|l| y - l <= n) {
                    out[y * w + x] = true;
                }
            }
            last = None;
            for y in (0..h).rev() {
                if tmp[y * w + x] {
                    last = Some(y);
                }
                if last.is_some_and(|l| l - y <= n) {
                    out[y * w + x] = true;
                }
            }
        }
        self.bits = out;
    }
}

// ---------------------------------------------------------------------------
// Contours
// ---------------------------------------------------------------------------

/// Signed area (shoelace) of a loop in y-down pixel space: positive for the
/// clockwise-on-screen outer loops [`trace_loops`] produces, negative for holes.
pub fn signed_area(l: &[P2]) -> f32 {
    let mut s = 0.0f64;
    for i in 0..l.len() {
        let (a, b) = (l[i], l[(i + 1) % l.len()]);
        s += a[0] as f64 * b[1] as f64 - b[0] as f64 * a[1] as f64;
    }
    (s * 0.5) as f32
}

/// Every closed boundary of the mask as a loop of corner points on the pixel
/// lattice (collinear points removed). Outer loops have positive
/// [`signed_area`], holes negative. Solid pixels that only touch diagonally
/// give separate loops.
pub fn trace_loops(m: &Mask) -> Vec<Vec<P2>> {
    // Directed edges with the solid pixel on the right of travel.
    let mut edges: Vec<((i32, i32), (i32, i32))> = Vec::new();
    for y in 0..m.h as i32 {
        for x in 0..m.w as i32 {
            if !m.get(x, y) {
                continue;
            }
            if !m.get(x, y - 1) {
                edges.push(((x, y), (x + 1, y)));
            }
            if !m.get(x + 1, y) {
                edges.push(((x + 1, y), (x + 1, y + 1)));
            }
            if !m.get(x, y + 1) {
                edges.push(((x + 1, y + 1), (x, y + 1)));
            }
            if !m.get(x - 1, y) {
                edges.push(((x, y + 1), (x, y)));
            }
        }
    }
    let mut starts: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
    for (i, e) in edges.iter().enumerate() {
        starts.entry(e.0).or_default().push(i);
    }
    let mut used = vec![false; edges.len()];
    let mut loops = Vec::new();
    for first in 0..edges.len() {
        if used[first] {
            continue;
        }
        let mut pts: Vec<(i32, i32)> = vec![edges[first].0];
        let mut cur = first;
        loop {
            used[cur] = true;
            let (a, b) = edges[cur];
            let d = (b.0 - a.0, b.1 - a.1);
            if b == edges[first].0 {
                break;
            }
            pts.push(b);
            // Prefer a right turn (hugging the solid pixel), then straight,
            // then left: that keeps diagonal contacts as separate loops.
            let right = (-d.1, d.0);
            let left = (d.1, -d.0);
            let pick = |want: (i32, i32)| starts.get(&b).and_then(|v| v.iter().copied().find(|&i| !used[i] && (edges[i].1.0 - b.0, edges[i].1.1 - b.1) == want));
            let Some(next) = pick(right).or_else(|| pick(d)).or_else(|| pick(left)) else { break };
            cur = next;
        }
        // Drop collinear middle points.
        let n = pts.len();
        let mut corners: Vec<P2> = Vec::new();
        for i in 0..n {
            let (p, q, r) = (pts[(i + n - 1) % n], pts[i], pts[(i + 1) % n]);
            let cross = (q.0 - p.0) as i64 * (r.1 - q.1) as i64 - (q.1 - p.1) as i64 * (r.0 - q.0) as i64;
            if cross != 0 {
                corners.push([q.0 as f32, q.1 as f32]);
            }
        }
        if corners.len() >= 3 {
            loops.push(corners);
        }
    }
    loops
}

// ---------------------------------------------------------------------------
// Simplification
// ---------------------------------------------------------------------------

fn dist_to_segment(p: P2, a: P2, b: P2) -> f32 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 { (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2).clamp(0.0, 1.0) } else { 0.0 };
    let (cx, cy) = (a[0] + t * dx, a[1] + t * dy);
    ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt()
}

/// Closed Douglas-Peucker: keep the vertex farthest from the first one, then
/// recursively keep any vertex farther than `tol` from its chord. At least 3
/// vertices survive (or the input is returned).
pub fn simplify_loop(l: &[P2], tol: f32) -> Vec<P2> {
    let n = l.len();
    if n <= 3 || tol <= 0.0 {
        return l.to_vec();
    }
    let far = (1..n)
        .max_by(|&a, &b| {
            let d = |i: usize| (l[i][0] - l[0][0]).powi(2) + (l[i][1] - l[0][1]).powi(2);
            d(a).partial_cmp(&d(b)).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(1);
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[far] = true;
    // Two open chains: 0..=far and far..=n (n wraps to 0).
    let mut stack = vec![(0usize, far), (far, n)];
    while let Some((i, j)) = stack.pop() {
        if j <= i + 1 {
            continue;
        }
        let (a, b) = (l[i], l[j % n]);
        let mut worst = (0.0f32, i);
        for (k, &p) in l.iter().enumerate().take(j).skip(i + 1) {
            let d = dist_to_segment(p, a, b);
            if d > worst.0 {
                worst = (d, k);
            }
        }
        if worst.0 > tol {
            keep[worst.1] = true;
            stack.push((i, worst.1));
            stack.push((worst.1, j));
        }
    }
    let out: Vec<P2> = l.iter().zip(&keep).filter(|(_, k)| **k).map(|(p, _)| *p).collect();
    if out.len() >= 3 { out } else { l.to_vec() }
}

fn orient(a: P2, b: P2, c: P2) -> f64 {
    (b[0] as f64 - a[0] as f64) * (c[1] as f64 - a[1] as f64) - (b[1] as f64 - a[1] as f64) * (c[0] as f64 - a[0] as f64)
}

/// Do segments `ab` and `cd` cross at an interior point of both?
fn proper_cross(a: P2, b: P2, c: P2, d: P2) -> bool {
    let (o1, o2, o3, o4) = (orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b));
    ((o1 > 1e-9 && o2 < -1e-9) || (o1 < -1e-9 && o2 > 1e-9)) && ((o3 > 1e-9 && o4 < -1e-9) || (o3 < -1e-9 && o4 > 1e-9))
}

/// True when no two edges (of one loop or of different loops) properly cross.
/// Touching at a shared vertex is allowed (4-connected islands may meet there).
pub fn loops_are_simple(loops: &[Vec<P2>]) -> bool {
    let mut segs: Vec<(P2, P2, [f32; 4])> = Vec::new();
    for l in loops {
        for i in 0..l.len() {
            let (a, b) = (l[i], l[(i + 1) % l.len()]);
            segs.push((a, b, [a[0].min(b[0]), a[1].min(b[1]), a[0].max(b[0]), a[1].max(b[1])]));
        }
    }
    for i in 0..segs.len() {
        for j in i + 1..segs.len() {
            let (bi, bj) = (segs[i].2, segs[j].2);
            if bi[2] < bj[0] || bj[2] < bi[0] || bi[3] < bj[1] || bj[3] < bi[1] {
                continue;
            }
            if proper_cross(segs[i].0, segs[i].1, segs[j].0, segs[j].1) {
                return false;
            }
        }
    }
    true
}

/// Simplify every loop within `tol`, halving the tolerance until the set is
/// free of crossings; at tolerance 0 the raw lattice loops are returned (always
/// valid). Loops that collapse to under 3 points are dropped.
pub fn simplify_loops(loops: &[Vec<P2>], tol: f32) -> Vec<Vec<P2>> {
    let mut t = tol;
    for _ in 0..4 {
        if t <= 0.0 {
            break;
        }
        let s: Vec<Vec<P2>> = loops.iter().map(|l| simplify_loop(l, t)).filter(|l| l.len() >= 3 && signed_area(l).abs() > 1e-4).collect();
        if loops_are_simple(&s) {
            return s;
        }
        t *= 0.5;
    }
    loops.to_vec()
}

// ---------------------------------------------------------------------------
// Grouping and triangulation
// ---------------------------------------------------------------------------

fn point_in_loop(p: P2, l: &[P2]) -> bool {
    let mut inside = false;
    let n = l.len();
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (l[i], l[j]);
        if (a[1] > p[1]) != (b[1] > p[1]) && (p[0] as f64) < (b[0] as f64 - a[0] as f64) * (p[1] as f64 - a[1] as f64) / (b[1] as f64 - a[1] as f64) + a[0] as f64 {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// A point strictly on the boundary of `l` that is not shared with a neighbour
/// loop: the midpoint of its longest edge.
fn probe_point(l: &[P2]) -> P2 {
    let mut best = (-1.0f32, [0.0, 0.0]);
    for i in 0..l.len() {
        let (a, b) = (l[i], l[(i + 1) % l.len()]);
        let len = (b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2);
        if len > best.0 {
            best = (len, [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]);
        }
    }
    best.1
}

/// One filled region: an outer boundary and the holes punched in it.
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    pub outer: Vec<P2>,
    pub holes: Vec<Vec<P2>>,
}

/// Group loops by nesting depth: even depth = outer boundary, odd = hole of the
/// closest enclosing loop. Winding is ignored, so it works for traced loops and
/// for hand-drawn custom outlines alike.
pub fn group_loops(loops: &[Vec<P2>], keep_holes: bool) -> Vec<Region> {
    let probes: Vec<P2> = loops.iter().map(|l| probe_point(l)).collect();
    let areas: Vec<f32> = loops.iter().map(|l| signed_area(l).abs()).collect();
    // containers[i] = loops strictly enclosing loop i.
    let containers: Vec<Vec<usize>> =
        (0..loops.len()).map(|i| (0..loops.len()).filter(|&j| j != i && areas[j] > areas[i] && point_in_loop(probes[i], &loops[j])).collect()).collect();
    let mut regions: Vec<Region> = Vec::new();
    let mut region_of: HashMap<usize, usize> = HashMap::new();
    let mut order: Vec<usize> = (0..loops.len()).collect();
    order.sort_by_key(|&i| containers[i].len());
    for i in order {
        let depth = containers[i].len();
        if depth.is_multiple_of(2) {
            region_of.insert(i, regions.len());
            regions.push(Region { outer: loops[i].clone(), holes: Vec::new() });
        } else if keep_holes {
            // Direct parent = the container with the most containers itself.
            if let Some(&p) = containers[i].iter().max_by_key(|&&j| containers[j].len())
                && let Some(&r) = region_of.get(&p)
            {
                regions[r].holes.push(loops[i].clone());
            }
        }
    }
    regions
}

/// A triangle mesh in rect-local pixel space. Indices are `u32` here; the
/// sprite layer narrows them to `u16`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PixelMesh {
    pub verts: Vec<P2>,
    pub indices: Vec<u32>,
}

impl PixelMesh {
    /// Sum of the triangle areas.
    pub fn area(&self) -> f32 {
        self.indices
            .as_chunks::<3>().0.iter()
            .map(|t| orient(self.verts[t[0] as usize], self.verts[t[1] as usize], self.verts[t[2] as usize]).abs() as f32 * 0.5)
            .sum()
    }
}

/// Ear-clip the regions into one mesh. Every triangle has the same winding:
/// positive `orient` in this y-down space, which `sprite.rs` reverses when it
/// flips y into counter-clockwise world space.
pub fn triangulate(regions: &[Region]) -> Option<PixelMesh> {
    let mut mesh = PixelMesh::default();
    for r in regions {
        let mut flat: Vec<f64> = Vec::new();
        let mut hole_starts: Vec<usize> = Vec::new();
        let mut verts: Vec<P2> = Vec::new();
        for p in &r.outer {
            flat.extend([p[0] as f64, p[1] as f64]);
            verts.push(*p);
        }
        for h in &r.holes {
            hole_starts.push(verts.len());
            for p in h {
                flat.extend([p[0] as f64, p[1] as f64]);
                verts.push(*p);
            }
        }
        let idx = earcutr::earcut(&flat, &hole_starts, 2).ok()?;
        let base = mesh.verts.len() as u32;
        mesh.verts.extend(&verts);
        for t in idx.as_chunks::<3>().0 {
            let (a, b, c) = (t[0], t[1], t[2]);
            if orient(verts[a], verts[b], verts[c]).abs() < 1e-9 {
                continue; // sliver from collinear points
            }
            if orient(verts[a], verts[b], verts[c]) > 0.0 {
                mesh.indices.extend([base + a as u32, base + b as u32, base + c as u32]);
            } else {
                mesh.indices.extend([base + a as u32, base + c as u32, base + b as u32]);
            }
        }
    }
    if mesh.indices.is_empty() { None } else { Some(mesh) }
}

// ---------------------------------------------------------------------------
// Public generators
// ---------------------------------------------------------------------------

/// How to build a tight mesh.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TightParams {
    /// A pixel is solid when its alpha is greater than this.
    pub alpha_threshold: u8,
    /// 0 = coarse (few triangles) .. 1 = fine (follows every pixel step).
    pub detail: f32,
    /// Pixels the mesh is pushed outward (0..=[`MAX_EXTRUDE`]).
    pub extrude: u32,
    /// Keep interior holes (rings, windows).
    pub holes: bool,
}

impl Default for TightParams {
    fn default() -> Self {
        Self { alpha_threshold: 1, detail: 0.5, extrude: 0, holes: true }
    }
}

/// Simplification tolerance in pixels for a detail value: 2.5 px at 0 down to
/// 0.35 px at 1 (research: the physics generator uses lerp(1.5, 0.05, detail)).
pub fn detail_to_tolerance(detail: f32) -> f32 {
    let d = detail.clamp(0.0, 1.0);
    2.5 + (0.35 - 2.5) * d
}

/// The simplified alpha outline of `rect`: outer loops first-class, holes
/// included when `holes`. Empty for a fully transparent rect.
pub fn trace_outline(a: &AlphaMap, rect: PixelRect, alpha_threshold: u8, extrude: u32, tol: f32, holes: bool) -> Vec<Vec<P2>> {
    let mut mask = Mask::from_alpha(a, rect, alpha_threshold);
    mask.dilate(extrude.min(MAX_EXTRUDE));
    let loops = trace_loops(&mask);
    let loops = if holes { loops } else { loops.into_iter().filter(|l| signed_area(l) > 0.0).collect() };
    simplify_loops(&loops, tol)
}

/// The full-rect quad (4 vertices, 2 triangles).
pub fn quad_mesh(w: u32, h: u32) -> PixelMesh {
    let (w, h) = (w as f32, h as f32);
    // y-down: (0,0) top-left. Winding normalised like `triangulate`.
    PixelMesh { verts: vec![[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], indices: vec![0, 1, 2, 0, 2, 3] }
}

/// Tight mesh of the sprite at `rect`, or `None` when it should fall back to
/// the quad (fully transparent, nothing survives simplification, triangulation
/// failed, or more vertices than a 16-bit index allows).
pub fn tight_mesh(a: &AlphaMap, rect: PixelRect, p: &TightParams) -> Option<PixelMesh> {
    let loops = trace_outline(a, rect, p.alpha_threshold, p.extrude, detail_to_tolerance(p.detail), p.holes);
    if loops.is_empty() {
        return None;
    }
    mesh_from_loops(&loops, p.holes)
}

/// Triangulate arbitrary (custom) outline polygons.
pub fn mesh_from_loops(loops: &[Vec<P2>], holes: bool) -> Option<PixelMesh> {
    let regions = group_loops(loops, holes);
    let mesh = triangulate(&regions)?;
    (mesh.verts.len() <= MAX_MESH_VERTS).then_some(mesh)
}

/// Physics outline of the sprite: simplified polygons (outer loops and holes)
/// at `tol` pixels. Defaults in the research: alpha tolerance 128 for the
/// editor generator, 200 for the fallback shape.
pub fn physics_outline(a: &AlphaMap, rect: PixelRect, alpha_threshold: u8, tol: f32) -> Vec<Vec<P2>> {
    trace_outline(a, rect, alpha_threshold, 0, tol, true)
}

/// Physics tolerance in pixels for a detail value (research §5.5:
/// lerp(1.5 px, 0.05 px, detail)).
pub fn physics_tolerance(detail: f32) -> f32 {
    let d = detail.clamp(0.0, 1.0);
    1.5 + (0.05 - 1.5) * d
}

fn is_convex(poly: &[u32], verts: &[P2]) -> bool {
    let n = poly.len();
    (0..n).all(|i| orient(verts[poly[i] as usize], verts[poly[(i + 1) % n] as usize], verts[poly[(i + 2) % n] as usize]) >= -1e-9)
}

/// Merge a triangle mesh into convex polygons (Hertel-Mehlhorn: repeatedly drop
/// a shared diagonal while the union stays convex). Every polygon is wound like
/// the mesh triangles. Not minimal, but never worse than 4x the optimum.
pub fn convex_pieces(mesh: &PixelMesh) -> Vec<Vec<P2>> {
    let mut polys: Vec<Option<Vec<u32>>> = mesh.indices.as_chunks::<3>().0.iter().map(|t| Some(t.to_vec())).collect();
    loop {
        // Directed edge -> polygon that owns it.
        let mut owner: HashMap<(u32, u32), usize> = HashMap::new();
        for (i, p) in polys.iter().enumerate() {
            if let Some(p) = p {
                for k in 0..p.len() {
                    owner.insert((p[k], p[(k + 1) % p.len()]), i);
                }
            }
        }
        let mut merged = false;
        'search: for i in 0..polys.len() {
            let Some(p) = polys[i].clone() else { continue };
            for k in 0..p.len() {
                let (a, b) = (p[k], p[(k + 1) % p.len()]);
                let Some(&j) = owner.get(&(b, a)) else { continue };
                if j == i {
                    continue;
                }
                let Some(q) = polys[j].clone() else { continue };
                // Walk p from b round to a (p.len() points), then q from the
                // point after a round to the point before b.
                let pk = (k + 1) % p.len(); // p[pk] == b
                let qa = q.iter().position(|&v| v == a).unwrap_or(0); // q: b -> a -> x ... -> y -> b
                let mut joined: Vec<u32> = Vec::with_capacity(p.len() + q.len() - 2);
                for s in 0..p.len() {
                    joined.push(p[(pk + s) % p.len()]);
                }
                for s in 1..q.len() - 1 {
                    joined.push(q[(qa + s) % q.len()]);
                }
                if joined.len() >= 3 && is_convex(&joined, &mesh.verts) {
                    polys[i] = Some(joined);
                    polys[j] = None;
                    merged = true;
                    break 'search;
                }
            }
        }
        if !merged {
            break;
        }
    }
    polys.into_iter().flatten().map(|p| p.into_iter().map(|i| mesh.verts[i as usize]).collect()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alpha_from(w: u32, h: u32, f: impl Fn(u32, u32) -> bool) -> AlphaMap {
        let mut alpha = Vec::new();
        for y in 0..h {
            for x in 0..w {
                alpha.push(if f(x, y) { 255 } else { 0 });
            }
        }
        AlphaMap { w, h, alpha }
    }

    fn full(a: &AlphaMap) -> PixelRect {
        PixelRect { x: 0, y: 0, w: a.w, h: a.h }
    }

    fn circle(w: u32, r: f32) -> AlphaMap {
        let c = w as f32 * 0.5;
        alpha_from(w, w, |x, y| ((x as f32 + 0.5 - c).powi(2) + (y as f32 + 0.5 - c).powi(2)).sqrt() <= r)
    }

    fn ring(w: u32, ro: f32, ri: f32) -> AlphaMap {
        let c = w as f32 * 0.5;
        alpha_from(w, w, |x, y| {
            let d = ((x as f32 + 0.5 - c).powi(2) + (y as f32 + 0.5 - c).powi(2)).sqrt();
            d <= ro && d > ri
        })
    }

    fn tri_overlap(a: [P2; 3], b: [P2; 3]) -> bool {
        // Strict separating-axis test: touching edges / vertices do not count.
        let axes = |t: &[P2; 3]| -> Vec<P2> { (0..3).map(|i| { let (p, q) = (t[i], t[(i + 1) % 3]); [-(q[1] - p[1]), q[0] - p[0]] }).collect() };
        for ax in axes(&a).into_iter().chain(axes(&b)) {
            let proj = |t: &[P2; 3]| {
                let v: Vec<f32> = t.iter().map(|p| p[0] * ax[0] + p[1] * ax[1]).collect();
                (v.iter().cloned().fold(f32::MAX, f32::min), v.iter().cloned().fold(f32::MIN, f32::max))
            };
            let (pa, pb) = (proj(&a), proj(&b));
            if pa.1 <= pb.0 + 1e-4 || pb.1 <= pa.0 + 1e-4 {
                return false;
            }
        }
        true
    }

    fn assert_no_overlap(m: &PixelMesh) {
        let tris: Vec<[P2; 3]> = m.indices.as_chunks::<3>().0.iter().map(|t| [m.verts[t[0] as usize], m.verts[t[1] as usize], m.verts[t[2] as usize]]).collect();
        for i in 0..tris.len() {
            for j in i + 1..tris.len() {
                assert!(!tri_overlap(tris[i], tris[j]), "triangles {i} and {j} overlap");
            }
        }
    }

    fn loops_area(loops: &[Vec<P2>]) -> f32 {
        // Outer positive, holes negative (traced orientation).
        loops.iter().map(|l| signed_area(l)).sum()
    }

    #[test]
    fn mask_threshold_and_dilate() {
        let a = alpha_from(8, 8, |x, y| x == 4 && y == 4);
        let mut m = Mask::from_alpha(&a, full(&a), 0);
        assert_eq!(m.solid_count(), 1);
        m.dilate(2);
        assert_eq!(m.solid_count(), 25, "square structuring element, 5x5");
        let m2 = Mask::from_alpha(&a, full(&a), 255);
        assert!(m2.is_empty(), "alpha must be strictly greater than the threshold");
        // Dilation stays inside the rect.
        let a = alpha_from(4, 4, |x, y| x == 0 && y == 0);
        let mut m = Mask::from_alpha(&a, full(&a), 0);
        m.dilate(32);
        assert_eq!(m.solid_count(), 16);
    }

    #[test]
    fn trace_square_and_l_shape() {
        let a = alpha_from(10, 10, |x, y| (2..8).contains(&x) && (2..8).contains(&y));
        let loops = trace_loops(&Mask::from_alpha(&a, full(&a), 0));
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].len(), 4);
        assert_eq!(signed_area(&loops[0]), 36.0);
        let a = alpha_from(10, 10, |x, y| x < 4 || y < 4);
        let loops = trace_loops(&Mask::from_alpha(&a, full(&a), 0));
        assert_eq!(loops[0].len(), 6);
        assert_eq!(signed_area(&loops[0]), (10 * 10 - 6 * 6) as f32);
    }

    #[test]
    fn diagonal_pixels_are_separate_loops() {
        let a = alpha_from(4, 4, |x, y| (x == 1 && y == 1) || (x == 2 && y == 2));
        let loops = trace_loops(&Mask::from_alpha(&a, full(&a), 0));
        assert_eq!(loops.len(), 2, "4-connected: diagonal contact is not one loop");
        assert!(loops.iter().all(|l| l.len() == 4 && signed_area(l) == 1.0));
        let mesh = tight_mesh(&a, full(&a), &TightParams::default()).unwrap();
        assert_eq!(mesh.area(), 2.0);
    }

    #[test]
    fn circle_mesh_is_valid_and_matches_the_outline_area() {
        let a = circle(64, 28.0);
        let m = Mask::from_alpha(&a, full(&a), 0);
        let solid = m.solid_count() as f32;
        let p = TightParams { detail: 0.5, ..TightParams::default() };
        let mesh = tight_mesh(&a, full(&a), &p).unwrap();
        assert!(mesh.verts.len() < 120, "simplified: {} verts", mesh.verts.len());
        assert!(mesh.verts.len() >= 8);
        assert_no_overlap(&mesh);
        // Area equals the simplified outline's area exactly, and tracks the pixels.
        let outline = trace_outline(&a, full(&a), 0, 0, detail_to_tolerance(0.5), true);
        assert!((mesh.area() - loops_area(&outline)).abs() < 1e-2 * solid, "mesh {} vs outline {}", mesh.area(), loops_area(&outline));
        assert!((mesh.area() - solid).abs() / solid < 0.05, "mesh {} vs pixels {}", mesh.area(), solid);
        // All triangles share one winding.
        assert!(mesh.indices.as_chunks::<3>().0.iter().all(|t| orient(mesh.verts[t[0] as usize], mesh.verts[t[1] as usize], mesh.verts[t[2] as usize]) > 0.0));
        // Finer detail follows the pixels more closely.
        let fine = tight_mesh(&a, full(&a), &TightParams { detail: 1.0, ..p }).unwrap();
        assert!(fine.verts.len() > mesh.verts.len());
        assert!((fine.area() - solid).abs() <= (mesh.area() - solid).abs() + 1.0);
    }

    #[test]
    fn ring_keeps_its_hole() {
        let a = ring(64, 28.0, 14.0);
        let solid = Mask::from_alpha(&a, full(&a), 0).solid_count() as f32;
        let with = tight_mesh(&a, full(&a), &TightParams::default()).unwrap();
        assert_no_overlap(&with);
        assert!((with.area() - solid).abs() / solid < 0.06, "{} vs {}", with.area(), solid);
        // No triangle centroid falls inside the hole.
        for t in with.indices.as_chunks::<3>().0 {
            let c = [0, 1, 2].map(|i| with.verts[t[i] as usize]);
            let (cx, cy) = ((c[0][0] + c[1][0] + c[2][0]) / 3.0, (c[0][1] + c[1][1] + c[2][1]) / 3.0);
            let d = ((cx - 32.0).powi(2) + (cy - 32.0).powi(2)).sqrt();
            assert!(d > 12.0, "centroid {d} inside the hole");
        }
        let no_holes = tight_mesh(&a, full(&a), &TightParams { holes: false, ..TightParams::default() }).unwrap();
        assert!(no_holes.area() > with.area() * 1.3, "hole detection off fills the ring");
    }

    #[test]
    fn two_islands_and_an_island_in_a_hole() {
        let a = alpha_from(64, 32, |x, y| (4..20).contains(&x) && (4..20).contains(&y) || (40..60).contains(&x) && (8..28).contains(&y));
        let mesh = tight_mesh(&a, full(&a), &TightParams::default()).unwrap();
        assert_eq!(mesh.area(), 16.0 * 16.0 + 20.0 * 20.0);
        assert_no_overlap(&mesh);
        // Square ring with a solid core: three nested loops.
        let a = alpha_from(48, 48, |x, y| {
            let (dx, dy) = ((x as i32 - 24).abs(), (y as i32 - 24).abs());
            let d = dx.max(dy);
            d < 20 && !(10..15).contains(&d) && d < 20
        });
        let mesh = tight_mesh(&a, full(&a), &TightParams::default()).unwrap();
        let solid = Mask::from_alpha(&a, full(&a), 0).solid_count() as f32;
        assert!((mesh.area() - solid).abs() < 1.0, "{} vs {}", mesh.area(), solid);
        assert_no_overlap(&mesh);
    }

    #[test]
    fn thin_line_survives() {
        let a = alpha_from(64, 64, |x, y| y == 32 && (4..60).contains(&x));
        let mesh = tight_mesh(&a, full(&a), &TightParams::default()).unwrap();
        assert_eq!(mesh.verts.len(), 4);
        assert_eq!(mesh.area(), 56.0);
        // A diagonal 1-px line: staircase must not collapse or self-cross.
        let a = alpha_from(64, 64, |x, y| x == y);
        let mesh = tight_mesh(&a, full(&a), &TightParams::default()).unwrap();
        assert_no_overlap(&mesh);
        assert!(mesh.area() > 0.0);
    }

    #[test]
    fn fully_transparent_falls_back() {
        let a = alpha_from(64, 64, |_, _| false);
        assert!(tight_mesh(&a, full(&a), &TightParams::default()).is_none());
        assert!(physics_outline(&a, full(&a), 128, 1.0).is_empty());
        let q = quad_mesh(64, 64);
        assert_eq!(q.verts.len(), 4);
        assert_eq!(q.area(), 64.0 * 64.0);
    }

    #[test]
    fn extrude_grows_the_mesh_inside_the_rect() {
        let a = circle(64, 20.0);
        let base = tight_mesh(&a, full(&a), &TightParams::default()).unwrap();
        let ext = tight_mesh(&a, full(&a), &TightParams { extrude: 4, ..TightParams::default() }).unwrap();
        assert!(ext.area() > base.area() + 100.0);
        assert!(ext.verts.iter().all(|v| (0.0..=64.0).contains(&v[0]) && (0.0..=64.0).contains(&v[1])));
    }

    #[test]
    fn sub_rect_coordinates_are_rect_local() {
        let a = alpha_from(128, 128, |x, y| (70..100).contains(&x) && (50..90).contains(&y));
        let rect = PixelRect { x: 64, y: 32, w: 64, h: 64 };
        let mesh = tight_mesh(&a, rect, &TightParams::default()).unwrap();
        let (minx, maxx) = mesh.verts.iter().fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(v[0]), hi.max(v[0])));
        assert_eq!((minx, maxx), (6.0, 36.0));
        assert_eq!(mesh.area(), 30.0 * 40.0);
    }

    #[test]
    fn simplification_never_introduces_crossings() {
        // A noisy blob stresses the retry path.
        let a = alpha_from(64, 64, |x, y| {
            let (dx, dy) = (x as f32 - 32.0, y as f32 - 32.0);
            let ang = dy.atan2(dx);
            dx.hypot(dy) < 22.0 + 6.0 * (ang * 7.0).sin() + 2.0 * (ang * 19.0).cos()
        });
        for detail in [0.0, 0.3, 0.7, 1.0] {
            let loops = trace_outline(&a, full(&a), 0, 0, detail_to_tolerance(detail), true);
            assert!(loops_are_simple(&loops), "detail {detail}");
            let mesh = mesh_from_loops(&loops, true).unwrap();
            assert_no_overlap(&mesh);
        }
    }

    #[test]
    fn custom_outline_with_unknown_winding_and_hole() {
        let outer: Vec<P2> = vec![[0.0, 0.0], [40.0, 0.0], [40.0, 40.0], [0.0, 40.0]];
        let hole: Vec<P2> = vec![[10.0, 10.0], [10.0, 20.0], [20.0, 20.0], [20.0, 10.0]]; // opposite winding
        let mesh = mesh_from_loops(&[hole.clone(), outer.clone()], true).unwrap();
        assert!((mesh.area() - (1600.0 - 100.0)).abs() < 1e-3);
        assert_no_overlap(&mesh);
        let solid = mesh_from_loops(&[hole, outer], false).unwrap();
        assert!((solid.area() - 1600.0).abs() < 1e-3);
    }

    #[test]
    fn convex_pieces_cover_the_polygon() {
        let a = ring(64, 28.0, 14.0);
        let mesh = tight_mesh(&a, full(&a), &TightParams::default()).unwrap();
        let tris = mesh.indices.len() / 3;
        let pieces = convex_pieces(&mesh);
        assert!(pieces.len() < tris, "{} pieces from {} triangles", pieces.len(), tris);
        let area: f32 = pieces.iter().map(|p| signed_area(p).abs()).sum();
        assert!((area - mesh.area()).abs() < 0.5, "{area} vs {}", mesh.area());
        for p in &pieces {
            assert!(p.len() >= 3);
            let n = p.len();
            assert!((0..n).all(|i| orient(p[i], p[(i + 1) % n], p[(i + 2) % n]) >= -1e-3), "piece is convex");
        }
    }

    #[test]
    fn physics_outline_is_coarser_than_the_mesh() {
        let a = circle(64, 28.0);
        let fine = physics_outline(&a, full(&a), 128, physics_tolerance(1.0));
        let coarse = physics_outline(&a, full(&a), 128, physics_tolerance(0.0));
        assert_eq!(fine.len(), 1);
        assert!(coarse[0].len() < fine[0].len());
        assert!(coarse[0].len() >= 8);
        let r = ring(64, 28.0, 14.0);
        assert_eq!(physics_outline(&r, full(&r), 128, 1.0).len(), 2, "outer + hole");
    }
}
