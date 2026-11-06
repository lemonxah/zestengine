//! Signed distance fields from glyph outlines: single-channel SDF and MTSDF
//! (three edge-coloured median channels plus a true-distance alpha).
//!
//! Outlines are flattened to line
//! segments in pixel units, contours are edge-coloured at corners, and every
//! texel takes the nearest edge per channel. The RGB median reconstructs sharp
//! corners at any magnification; the alpha channel stays an exact distance for
//! outline, shadow and glow, which need values far from the edge.
//!
//! Encoding: `v = 0.5 + d / (2 * spread)`, `d` in atlas texels, positive inside.

use skrifa::PathSink;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct V2 {
    x: f64,
    y: f64,
}

impl V2 {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
    fn sub(self, o: V2) -> V2 {
        V2::new(self.x - o.x, self.y - o.y)
    }
    fn add(self, o: V2) -> V2 {
        V2::new(self.x + o.x, self.y + o.y)
    }
    fn mul(self, k: f64) -> V2 {
        V2::new(self.x * k, self.y * k)
    }
    fn dot(self, o: V2) -> f64 {
        self.x * o.x + self.y * o.y
    }
    fn cross(self, o: V2) -> f64 {
        self.x * o.y - self.y * o.x
    }
    fn len(self) -> f64 {
        self.dot(self).sqrt()
    }
    fn norm(self) -> V2 {
        let l = self.len();
        if l > 1e-12 { self.mul(1.0 / l) } else { V2::default() }
    }
}

/// Channel bits of an edge colour.
const R: u8 = 1;
const G: u8 = 2;
const B: u8 = 4;
const WHITE: u8 = R | G | B;
const CYAN: u8 = G | B;
const MAGENTA: u8 = R | B;
const YELLOW: u8 = R | G;

/// A flattened line segment of a contour (pixel units, y up).
#[derive(Clone, Copy, Debug)]
struct Edge {
    a: V2,
    b: V2,
    dir: V2,
    len: f64,
    color: u8,
    min: V2,
    max: V2,
}

impl Edge {
    fn new(a: V2, b: V2) -> Self {
        let d = b.sub(a);
        let len = d.len();
        Self {
            a,
            b,
            dir: d.norm(),
            len,
            color: WHITE,
            min: V2::new(a.x.min(b.x), a.y.min(b.y)),
            max: V2::new(a.x.max(b.x), a.y.max(b.y)),
        }
    }
}

/// Source segment bookkeeping for corner detection.
#[derive(Clone, Copy, Debug)]
struct SegInfo {
    /// Range into the contour's flattened edges.
    start: usize,
    end: usize,
    t_start: V2,
    t_end: V2,
}

#[derive(Clone, Debug, Default)]
struct Contour {
    edges: Vec<Edge>,
    segs: Vec<SegInfo>,
}

/// A glyph outline flattened into pixel space, ready for [`generate`].
#[derive(Clone, Debug, Default)]
pub struct Shape {
    contours: Vec<Contour>,
}

/// Collects a font outline (font units) into a [`Shape`] scaled to pixels.
pub struct ShapeBuilder {
    scale: f64,
    shape: Shape,
    cur: Contour,
    start: V2,
    pos: V2,
    open: bool,
}

/// Flattening tolerance in pixels.
const FLATNESS: f64 = 0.04;

impl ShapeBuilder {
    /// `scale` converts font units to pixels (`em_px / units_per_em`).
    pub fn new(scale: f32) -> Self {
        Self {
            scale: scale as f64,
            shape: Shape::default(),
            cur: Contour::default(),
            start: V2::default(),
            pos: V2::default(),
            open: false,
        }
    }

    pub fn finish(mut self) -> Shape {
        self.end_contour();
        self.shape
    }

    fn pt(&self, x: f32, y: f32) -> V2 {
        V2::new(x as f64 * self.scale, y as f64 * self.scale)
    }

    fn end_contour(&mut self) {
        if !self.open {
            return;
        }
        if self.pos != self.start {
            let (s, p) = (self.start, self.pos);
            self.push_line(p, s);
        }
        let c = std::mem::take(&mut self.cur);
        if c.edges.len() >= 2 {
            self.shape.contours.push(c);
        }
        self.open = false;
    }

    fn push_line(&mut self, a: V2, b: V2) {
        if a == b {
            return;
        }
        let i = self.cur.edges.len();
        self.cur.edges.push(Edge::new(a, b));
        let t = b.sub(a);
        self.cur.segs.push(SegInfo { start: i, end: i + 1, t_start: t, t_end: t });
    }

    /// Flattened polyline for one curve; `pts` are sampled at `n` steps.
    fn push_curve(&mut self, n: usize, t0: V2, t1: V2, eval: impl Fn(f64) -> V2) {
        let first = self.cur.edges.len();
        let mut prev = self.pos;
        for i in 1..=n {
            let p = eval(i as f64 / n as f64);
            if p != prev {
                self.cur.edges.push(Edge::new(prev, p));
            }
            prev = p;
        }
        let last = self.cur.edges.len();
        if last > first {
            self.cur.segs.push(SegInfo { start: first, end: last, t_start: t0, t_end: t1 });
        }
        self.pos = prev;
    }
}

impl PathSink for ShapeBuilder {
    fn move_to(&mut self, x: f32, y: f32) {
        self.end_contour();
        self.start = self.pt(x, y);
        self.pos = self.start;
        self.open = true;
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.pt(x, y);
        let a = self.pos;
        self.push_line(a, p);
        self.pos = p;
    }

    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        let (p0, p1, p2) = (self.pos, self.pt(cx, cy), self.pt(x, y));
        let dev = p0.sub(p1.mul(2.0)).add(p2).len() * 0.25;
        let n = ((dev / FLATNESS).sqrt().ceil() as usize).clamp(1, 48);
        let tan = |a: V2, b: V2, fb: V2| if b.sub(a).len() > 1e-9 { b.sub(a) } else { fb };
        let t0 = tan(p0, p1, p2.sub(p0));
        let t1 = tan(p1, p2, p2.sub(p0));
        self.push_curve(n, t0, t1, |t| {
            let u = 1.0 - t;
            p0.mul(u * u).add(p1.mul(2.0 * u * t)).add(p2.mul(t * t))
        });
    }

    fn curve_to(&mut self, c0x: f32, c0y: f32, c1x: f32, c1y: f32, x: f32, y: f32) {
        let (p0, p1, p2, p3) = (self.pos, self.pt(c0x, c0y), self.pt(c1x, c1y), self.pt(x, y));
        let d1 = p0.sub(p1.mul(2.0)).add(p2).len();
        let d2 = p1.sub(p2.mul(2.0)).add(p3).len();
        let dev = d1.max(d2) * 0.75;
        let n = ((dev / FLATNESS).sqrt().ceil() as usize).clamp(1, 64);
        let chord = p3.sub(p0);
        let pick = |cands: [V2; 3]| cands.into_iter().find(|v| v.len() > 1e-9).unwrap_or(chord);
        let t0 = pick([p1.sub(p0), p2.sub(p0), chord]);
        let t1 = pick([p3.sub(p2), p3.sub(p1), chord]);
        self.push_curve(n, t0, t1, |t| {
            let u = 1.0 - t;
            p0.mul(u * u * u)
                .add(p1.mul(3.0 * u * u * t))
                .add(p2.mul(3.0 * u * t * t))
                .add(p3.mul(t * t * t))
        });
    }

    fn close(&mut self) {
        self.end_contour();
    }
}

impl Shape {
    pub fn is_empty(&self) -> bool {
        self.contours.is_empty()
    }

    /// Bounds `(min_x, min_y, max_x, max_y)` in pixels, y up.
    pub fn bounds(&self) -> Option<[f32; 4]> {
        let mut it = self.contours.iter().flat_map(|c| &c.edges);
        let first = it.next()?;
        let (mut lo, mut hi) = (first.min, first.max);
        for e in it {
            lo = V2::new(lo.x.min(e.min.x), lo.y.min(e.min.y));
            hi = V2::new(hi.x.max(e.max.x), hi.y.max(e.max.y));
        }
        Some([lo.x as f32, lo.y as f32, hi.x as f32, hi.y as f32])
    }

    /// Twice the signed area (positive = counter-clockwise, y up).
    fn signed_area(&self) -> f64 {
        self.contours
            .iter()
            .flat_map(|c| &c.edges)
            .map(|e| e.a.cross(e.b))
            .sum()
    }

    /// Assign edge colours so corners stay sharp in the median channels.
    pub fn colour_edges(&mut self) {
        for c in &mut self.contours {
            colour_contour(c);
        }
    }

    fn all_edges(&self) -> Vec<Edge> {
        self.contours.iter().flat_map(|c| c.edges.iter().copied()).collect()
    }
}

fn is_corner(a: V2, b: V2) -> bool {
    let (a, b) = (a.norm(), b.norm());
    // msdfgen's default corner angle (3 rad): sharp turns and reversals.
    a.dot(b) <= 0.0 || a.cross(b).abs() > 3.0_f64.sin()
}

fn colour_contour(c: &mut Contour) {
    let n = c.segs.len();
    let corners: Vec<usize> = (0..n)
        .filter(|&i| is_corner(c.segs[i].t_end, c.segs[(i + 1) % n].t_start))
        .collect();
    match corners.len() {
        0 => c.edges.iter_mut().for_each(|e| e.color = WHITE),
        1 => {
            // Teardrop: thirds of the flattened chain starting after the corner.
            let m = c.edges.len();
            let first = c.segs[(corners[0] + 1) % n].start;
            for k in 0..m {
                let e = (first + k) % m;
                c.edges[e].color = if m < 3 {
                    WHITE
                } else if k < m / 3 {
                    CYAN
                } else if k < 2 * m / 3 {
                    WHITE
                } else {
                    YELLOW
                };
            }
        }
        k => {
            let palette = [CYAN, MAGENTA, YELLOW];
            let mut cols: Vec<u8> = (0..k).map(|s| palette[s % 3]).collect();
            // The last spline wraps onto the first: keep them distinct.
            if k > 1 && cols[k - 1] == cols[0] {
                let avoid = [cols[k - 2], cols[0]];
                cols[k - 1] = *palette.iter().find(|p| !avoid.contains(p)).unwrap_or(&WHITE);
            }
            for s in 0..k {
                let from = (corners[s] + 1) % n;
                let to = corners[(s + 1) % k];
                let mut i = from;
                loop {
                    let seg = c.segs[i];
                    for e in seg.start..seg.end {
                        c.edges[e].color = cols[s];
                    }
                    if i == to {
                        break;
                    }
                    i = (i + 1) % n;
                }
            }
        }
    }
}

/// What a generated bitmap holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdfMode {
    /// One byte per texel: true signed distance.
    Sdf,
    /// RGBA: median-reconstructed edges in RGB, true distance in alpha.
    Mtsdf,
}

/// A generated distance bitmap and where it sits relative to the glyph origin.
#[derive(Clone, Debug)]
pub struct GlyphBitmap {
    pub width: u32,
    pub height: u32,
    /// 1 (SDF) or 4 (MTSDF).
    pub channels: u8,
    pub data: Vec<u8>,
    /// Pixels from the glyph origin to the bitmap's left edge.
    pub left: f32,
    /// Pixels from the baseline up to the bitmap's top edge.
    pub top: f32,
}

fn box_dist(p: V2, lo: V2, hi: V2) -> f64 {
    let dx = (lo.x - p.x).max(0.0).max(p.x - hi.x);
    let dy = (lo.y - p.y).max(0.0).max(p.y - hi.y);
    (dx * dx + dy * dy).sqrt()
}

#[derive(Clone, Copy)]
struct Best {
    /// Absolute distance of the winning edge.
    dist: f64,
    edge: usize,
    ortho: f64,
}

/// Rasterize `shape` (already scaled to pixels, coloured for MTSDF) into a
/// distance bitmap with `spread` texels of range each side of the edge.
pub fn generate(shape: &Shape, spread: f32, mode: SdfMode) -> Option<GlyphBitmap> {
    let bounds = shape.bounds()?;
    let spread = spread.max(1.0) as f64;
    let pad = spread.ceil() + 1.0;
    let (minx, miny, maxx, maxy) = (bounds[0] as f64, bounds[1] as f64, bounds[2] as f64, bounds[3] as f64);
    let x0 = (minx - pad).floor();
    let y1 = (maxy + pad).ceil();
    let w = ((maxx + pad).ceil() - x0).max(1.0) as u32;
    let h = (y1 - (miny - pad).floor()).max(1.0) as u32;
    let edges = shape.all_edges();
    // Inside is to the left of the edge direction for counter-clockwise outers.
    let inside_sign = if shape.signed_area() >= 0.0 { 1.0 } else { -1.0 };
    let channels = if mode == SdfMode::Mtsdf { 4 } else { 1 };
    let mut data = vec![0u8; (w * h) as usize * channels as usize];
    let enc = |d: f64| -> u8 { ((0.5 + d / (2.0 * spread)).clamp(0.0, 1.0) * 255.0).round() as u8 };

    for j in 0..h {
        for i in 0..w {
            let p = V2::new(x0 + i as f64 + 0.5, y1 - (j as f64 + 0.5));
            // Nearest edge overall (true distance) and per colour channel.
            let mut all = Best { dist: f64::INFINITY, edge: usize::MAX, ortho: 1.0 };
            let mut ch = [all; 3];
            let mut winding = 0i32;
            for (ei, e) in edges.iter().enumerate() {
                // Winding number (non-zero fill) by ray crossing.
                if (e.a.y <= p.y) != (e.b.y <= p.y) {
                    let t = (p.y - e.a.y) / (e.b.y - e.a.y);
                    if p.x < e.a.x + t * (e.b.x - e.a.x) {
                        winding += if e.b.y > e.a.y { 1 } else { -1 };
                    }
                }
                let bd = box_dist(p, e.min, e.max);
                let relevant = |c: usize| e.color & (1 << c) != 0 && bd <= ch[c].dist;
                let want_ch = mode == SdfMode::Mtsdf && (relevant(0) || relevant(1) || relevant(2));
                if bd > all.dist && !want_ch {
                    continue;
                }
                let ap = p.sub(e.a);
                let t = ap.dot(e.dir).clamp(0.0, e.len);
                let q = e.a.add(e.dir.mul(t));
                let to = p.sub(q);
                let dist = to.len();
                let ortho = if dist > 1e-12 { e.dir.dot(to.mul(1.0 / dist)).abs() } else { 0.0 };
                let better = |b: &Best| dist < b.dist - 1e-9 || (dist < b.dist + 1e-9 && ortho < b.ortho);
                if better(&all) {
                    all = Best { dist, edge: ei, ortho };
                }
                if mode == SdfMode::Mtsdf {
                    for (c, best_ch) in ch.iter_mut().enumerate() {
                        if e.color & (1 << c) != 0 && better(best_ch) {
                            *best_ch = Best { dist, edge: ei, ortho };
                        }
                    }
                }
            }
            let inside = winding != 0;
            let true_sd = if inside { all.dist } else { -all.dist };
            let o = (j * w + i) as usize * channels as usize;
            if mode == SdfMode::Sdf {
                data[o] = enc(true_sd);
                continue;
            }
            let mut sd = [true_sd; 3];
            for c in 0..3 {
                let b = ch[c];
                if b.edge == usize::MAX {
                    continue;
                }
                let e = &edges[b.edge];
                let ap = p.sub(e.a);
                let along = ap.dot(e.dir);
                let perp = e.dir.cross(ap);
                // Pseudo-distance: past an endpoint, use the distance to the
                // edge's supporting line, which extends corners smoothly.
                let mag = if along < 0.0 || along > e.len { perp.abs().min(b.dist) } else { b.dist };
                let side = if perp >= 0.0 { 1.0 } else { -1.0 } * inside_sign;
                sd[c] = mag * side;
            }
            let med = sd[0].max(sd[1].min(sd[2])).min(sd[0].min(sd[1]).max(sd[2]));
            // Overlapping contours break the orientation assumption: where the
            // median disagrees with the winding fill, fall back to true distance.
            if (med > 0.0) != inside {
                sd = [true_sd; 3];
            }
            data[o] = enc(sd[0]);
            data[o + 1] = enc(sd[1]);
            data[o + 2] = enc(sd[2]);
            data[o + 3] = enc(true_sd);
        }
    }
    Some(GlyphBitmap { width: w, height: h, channels, data, left: x0 as f32, top: y1 as f32 })
}

/// Median of three, for reading MTSDF edges back (tests and CPU previews).
pub fn median(r: u8, g: u8, b: u8) -> u8 {
    r.max(g.min(b)).min(r.min(g).max(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(size: f32) -> Shape {
        let mut b = ShapeBuilder::new(1.0);
        // Counter-clockwise, y up.
        b.move_to(0.0, 0.0);
        b.line_to(size, 0.0);
        b.line_to(size, size);
        b.line_to(0.0, size);
        b.close();
        let mut s = b.finish();
        s.colour_edges();
        s
    }

    fn sample(bm: &GlyphBitmap, px: f32, py: f32, ch: usize) -> u8 {
        // Shape coords (y up) -> bitmap texel.
        let i = (px - bm.left).floor() as i32;
        let j = (bm.top - py).floor() as i32;
        bm.data[((j as u32 * bm.width + i as u32) as usize) * bm.channels as usize + ch]
    }

    fn dist(v: u8, spread: f32) -> f32 {
        (v as f32 / 255.0 - 0.5) * 2.0 * spread
    }

    #[test]
    fn sdf_sign_and_edge_value() {
        let s = square(20.0);
        let bm = generate(&s, 4.0, SdfMode::Sdf).unwrap();
        assert_eq!(bm.channels, 1);
        // Centre is inside and clamps at the spread; far outside is negative.
        assert!(sample(&bm, 10.0, 10.0, 0) > 250);
        assert!(sample(&bm, -3.5, 10.0, 0) < 40);
        // One texel inside the left edge reads about +0.5 texel.
        let d = dist(sample(&bm, 0.5, 10.0, 0), 4.0);
        assert!((d - 0.5).abs() < 0.1, "got {d}");
        let d = dist(sample(&bm, -0.5, 10.0, 0), 4.0);
        assert!((d + 0.5).abs() < 0.1, "got {d}");
    }

    #[test]
    fn mtsdf_keeps_corners_sharp() {
        let s = square(20.0);
        let bm = generate(&s, 4.0, SdfMode::Mtsdf).unwrap();
        assert_eq!(bm.channels, 4);
        // Diagonally outside the corner by (1.5,1.5): true distance is 2.12,
        // the corner-preserving median reads about 1.5.
        let (x, y) = (21.5, 21.5);
        let med = dist(median(sample(&bm, x, y, 0), sample(&bm, x, y, 1), sample(&bm, x, y, 2)), 4.0);
        let tru = dist(sample(&bm, x, y, 3), 4.0);
        assert!((tru + 2.12).abs() < 0.2, "true {tru}");
        assert!((med + 1.5).abs() < 0.3, "median {med}");
        // Inside stays positive in both.
        let m = median(sample(&bm, 10.5, 10.5, 0), sample(&bm, 10.5, 10.5, 1), sample(&bm, 10.5, 10.5, 2));
        assert!(m > 200);
    }

    #[test]
    fn colours_adjacent_corner_edges_differently() {
        let s = square(10.0);
        let c = &s.contours[0];
        let cols: Vec<u8> = c.edges.iter().map(|e| e.color).collect();
        assert_eq!(cols.len(), 4);
        for i in 0..4 {
            assert_ne!(cols[i], cols[(i + 1) % 4], "{cols:?}");
            assert!(cols[i] != WHITE);
        }
    }

    #[test]
    fn real_glyph_coverage_matches_outline_area() {
        // Reconstruct coverage the way the shader does (median edge at 0.5,
        // 1 px ramp) and compare with the outline's analytic area.
        let face = crate::font::test_fonts::plex_sans();
        for ch in ['O', 'g', 'B', '&', 'a'] {
            let gid = face.glyph_for(ch).unwrap();
            let mut shape = face.outline(gid, 64.0).unwrap();
            let area = shape.signed_area().abs() * 0.5;
            shape.colour_edges();
            let bm = generate(&shape, 6.0, SdfMode::Mtsdf).unwrap();
            let mut cov = 0.0f64;
            let mut alpha_cov = 0.0f64;
            for px in bm.data.as_chunks::<4>().0 {
                let m = median(px[0], px[1], px[2]) as f64 / 255.0;
                let ramp = 12.0; // texels per unit distance at 1:1 magnification
                cov += ((m - 0.5) * ramp + 0.5).clamp(0.0, 1.0);
                alpha_cov += ((px[3] as f64 / 255.0 - 0.5) * ramp + 0.5).clamp(0.0, 1.0);
            }
            let err = (cov - area).abs() / area;
            assert!(err < 0.02, "{ch}: median coverage {cov:.0} vs outline area {area:.0}");
            let err = (alpha_cov - area).abs() / area;
            assert!(err < 0.02, "{ch}: true-distance coverage {alpha_cov:.0} vs outline area {area:.0}");
        }
    }

    #[test]
    fn hole_is_outside() {
        let mut b = ShapeBuilder::new(1.0);
        // Outer ccw, inner (hole) cw.
        b.move_to(0.0, 0.0);
        b.line_to(30.0, 0.0);
        b.line_to(30.0, 30.0);
        b.line_to(0.0, 30.0);
        b.close();
        b.move_to(10.0, 10.0);
        b.line_to(10.0, 20.0);
        b.line_to(20.0, 20.0);
        b.line_to(20.0, 10.0);
        b.close();
        let mut s = b.finish();
        s.colour_edges();
        let bm = generate(&s, 4.0, SdfMode::Mtsdf).unwrap();
        assert!(sample(&bm, 15.5, 15.5, 3) < 120, "hole centre must read outside");
        assert!(sample(&bm, 5.5, 5.5, 3) > 140);
    }

    #[test]
    fn curves_flatten_and_empty_shape_is_none() {
        let mut b = ShapeBuilder::new(1.0);
        b.move_to(0.0, 0.0);
        b.quad_to(20.0, 0.0, 20.0, 20.0);
        b.curve_to(20.0, 30.0, 0.0, 30.0, 0.0, 20.0);
        b.close();
        let mut s = b.finish();
        assert!(s.contours[0].edges.len() > 8);
        s.colour_edges();
        assert!(generate(&s, 3.0, SdfMode::Sdf).is_some());
        assert!(generate(&Shape::default(), 3.0, SdfMode::Sdf).is_none());
    }
}
