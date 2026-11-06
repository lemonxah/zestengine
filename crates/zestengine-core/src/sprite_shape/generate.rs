//! Spline + profile -> sprite-shape mesh (2D-8), and the collider outline.
//!
//! # Pipeline (one pass, in this order)
//!
//! 1. **Tessellate** the spline ([`super::spline::tessellate`]).
//! 2. **Fill** (closed shapes): triangulate the outline (offset by the
//!    profile's `fill_edge_offset`), UVs from local position / the fill
//!    texture's size in world units (tiling), or the sprite stretched over the
//!    bounds. Holes are not supported.
//! 3. **Edges**: the outline is walked clockwise (a counter-clockwise ring is
//!    reversed; `flip_edge` reverses again) so the left normal points out of the
//!    shape. Every segment gets the angle range its normal angle selects
//!    ([`super::select_range`]); consecutive segments with the same range and no
//!    corner form a *run*. A run is cut into tiles ([`plan_tiles`], the
//!    stretch / fit maths) and each tile is a strip of quads that follows the
//!    curve, thickness = sprite height x point height, UV `u` linear in arc
//!    length across the tile.
//! 4. **Corners**: a control point that may be a corner and turns more than the
//!    threshold breaks the run; the profile's outer / inner corner sprite is
//!    centred on the corner along the bisector. Without a corner sprite the two
//!    runs join with a (length-limited) miter instead.
//! 5. **Caps** (open shapes): start / end cap sprites placed beyond the ends.
//!
//! Output is one [`MeshGroup`] per (fill | edge, texture), each a plain
//! position / UV / `u16` index list in object-local units (UV `v` is top-down
//! like the sprite pool's meshes). Within a group fill triangles come first, then
//! edge strips, then corners, then caps (later draws over earlier).

use std::collections::HashMap;

use glam::Vec2;

use super::spline::{Poly, TessMode, offset_ring, signed_area, simplify, tessellate, triangulate, triangulate_with_holes};
use super::{EdgeFit, SpriteShapeProfile, SpriteShapeRenderer, select_range};

/// What the generator needs to know about a resolved sprite.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpriteMetrics {
    /// Opaque texture key (the engine's registered texture id); sprites with the
    /// same key share one mesh group.
    pub texture: u32,
    /// `u0, v_top, u1, v_bottom` of the sprite's rect in its texture.
    pub uv_rect: [f32; 4],
    /// Sprite size in world units.
    pub size: [f32; 2],
}

/// A fill texture the profile resolved to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FillTexture {
    /// Opaque texture key (the engine's registered texture id).
    pub texture: u32,
    /// Texture size in pixels.
    pub size_px: [u32; 2],
}

/// Everything the generator needs for one shape.
pub struct GenInput<'a> {
    /// The shape component.
    pub shape: &'a SpriteShapeRenderer,
    /// Its profile.
    pub profile: &'a SpriteShapeProfile,
    /// Resolve a `path#id` sprite reference.
    pub sprite: &'a dyn Fn(&str) -> Option<SpriteMetrics>,
    /// The loaded fill texture, if the profile names one that resolved.
    pub fill_texture: Option<FillTexture>,
}

/// Which part of the shape a mesh group is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GroupKind {
    /// The filled inside of a closed shape.
    Fill,
    /// Edge strips.
    Edge,
}

/// One mesh group: everything that draws from one texture.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshGroup {
    /// What part of the shape this group is.
    pub kind: GroupKind,
    /// The texture key (sprites with the same key share a group).
    pub texture: u32,
    /// Vertex positions in object-local units.
    pub positions: Vec<[f32; 2]>,
    /// Vertex UVs.
    pub uvs: Vec<[f32; 2]>,
    /// Triangle indices into the arrays above.
    pub indices: Vec<u16>,
}

const MAX_VERTS: usize = u16::MAX as usize;

impl MeshGroup {
    fn new(kind: GroupKind, texture: u32) -> Self {
        Self { kind, texture, positions: Vec::new(), uvs: Vec::new(), indices: Vec::new() }
    }
    fn room(&self, n: usize) -> bool {
        self.positions.len() + n <= MAX_VERTS
    }
    fn vertex(&mut self, p: Vec2, uv: [f32; 2]) -> u16 {
        self.positions.push(p.to_array());
        self.uvs.push(uv);
        (self.positions.len() - 1) as u16
    }
}

/// Counts of what a generation produced.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ShapeStats {
    /// Tessellated spline vertices.
    pub points: usize,
    /// Edge runs.
    pub runs: usize,
    /// Edge tiles.
    pub tiles: usize,
    /// Corner sprites.
    pub corners: usize,
    /// Cap sprites.
    pub caps: usize,
    /// Fill triangles.
    pub fill_triangles: usize,
    /// Geometry was dropped to stay inside 65535 vertices per group.
    pub truncated: bool,
}

/// The generated mesh: one group per (part, texture).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ShapeMesh {
    /// Fill groups first, then edge groups in first-use order.
    pub groups: Vec<MeshGroup>,
    /// Local bounds of everything generated.
    pub bounds: Option<([f32; 2], [f32; 2])>,
    /// Counts of what was generated.
    pub stats: ShapeStats,
}

impl ShapeMesh {
    /// Total vertices across the groups.
    pub fn vertex_count(&self) -> usize {
        self.groups.iter().map(|g| g.positions.len()).sum()
    }
    /// Total triangles across the groups.
    pub fn triangle_count(&self) -> usize {
        self.groups.iter().map(|g| g.indices.len() / 3).sum()
    }
}

#[derive(Default)]
struct Builder {
    groups: Vec<MeshGroup>,
    stats: ShapeStats,
}

impl Builder {
    fn group(&mut self, kind: GroupKind, texture: u32) -> &mut MeshGroup {
        let i = match self.groups.iter().position(|g| g.kind == kind && g.texture == texture) {
            Some(i) => i,
            None => {
                self.groups.push(MeshGroup::new(kind, texture));
                self.groups.len() - 1
            }
        };
        &mut self.groups[i]
    }

    /// A quad given bottom-left, bottom-right, top-right, top-left (CCW).
    fn quad(&mut self, kind: GroupKind, texture: u32, corners: [Vec2; 4], uv: [[f32; 2]; 4]) {
        let truncated = {
            let g = self.group(kind, texture);
            if !g.room(4) {
                true
            } else {
                let i = [g.vertex(corners[0], uv[0]), g.vertex(corners[1], uv[1]), g.vertex(corners[2], uv[2]), g.vertex(corners[3], uv[3])];
                g.indices.extend_from_slice(&[i[0], i[1], i[2], i[0], i[2], i[3]]);
                false
            }
        };
        self.stats.truncated |= truncated;
    }
}

/// One tile of a run: `s0..s1` along the run, showing `frac` of the sprite's width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TilePlan {
    /// Edge sprite variant (index into the range's sprite list).
    pub variant: usize,
    /// Start of the tile along the run.
    pub s0: f32,
    /// End of the tile along the run.
    pub s1: f32,
    /// Fraction of the sprite's width shown (Tile fit cuts the last tile).
    pub frac: f32,
}

/// Most tiles one run can get (a degenerate sprite width cannot explode the mesh).
pub const MAX_TILES: usize = 4096;

/// Lay tiles along a run of `length`. `pick(arc)` returns the variant index and
/// natural width of the tile starting at `arc`. Stretch rounds the tile count to
/// the nearest and scales all widths to fill the run exactly, Fit rounds down
/// (at least one) so tiles only widen, Tile keeps natural widths and cuts the
/// last tile.
pub fn plan_tiles(length: f32, fit: EdgeFit, mut pick: impl FnMut(f32) -> (usize, f32)) -> Vec<TilePlan> {
    if length <= 1e-6 {
        return Vec::new();
    }
    let min_w = length / MAX_TILES as f32;
    let eps = length * 1e-5;
    let mut raw: Vec<(usize, f32)> = Vec::new();
    let mut acc = 0.0f32;
    while acc < length - eps && raw.len() < MAX_TILES {
        let (variant, w) = pick(acc);
        let w = w.max(min_w);
        let stop = match fit {
            EdgeFit::Stretch => !raw.is_empty() && acc + 0.5 * w > length,
            EdgeFit::Fit => !raw.is_empty() && acc + w > length + eps,
            EdgeFit::Tile => false,
        };
        if stop {
            break;
        }
        raw.push((variant, w));
        acc += w;
    }
    let mut out = Vec::with_capacity(raw.len());
    let scale = if fit == EdgeFit::Tile { 1.0 } else { length / acc };
    let mut s = 0.0f32;
    for (k, &(variant, w)) in raw.iter().enumerate() {
        let (s0, mut s1) = (s * scale, (s + w) * scale);
        let mut frac = 1.0;
        if fit == EdgeFit::Tile && k + 1 == raw.len() && s1 > length {
            frac = ((length - s0) / w).clamp(0.0, 1.0);
            s1 = length;
        } else if k + 1 == raw.len() {
            s1 = length; // no float drift on the last tile
        }
        out.push(TilePlan { variant, s0, s1, frac });
        s += w;
    }
    out
}

/// Which way a corner turns relative to the shape's normal side.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CornerKind {
    /// Turns away from the normal side (convex corner of a clockwise outline).
    Outer,
    /// Turns toward the normal side (concave).
    Inner,
}

/// The angle (degrees, `[0, 360)`) of a normal: counter-clockwise from +Y.
pub fn normal_angle(n: Vec2) -> f32 {
    (-n.x).atan2(n.y).to_degrees().rem_euclid(360.0)
}

/// Resolve sprite references once per generation.
struct Resolver<'a> {
    f: &'a dyn Fn(&str) -> Option<SpriteMetrics>,
    cache: HashMap<String, Option<SpriteMetrics>>,
}

impl Resolver<'_> {
    fn get(&mut self, name: &str) -> Option<SpriteMetrics> {
        if name.is_empty() {
            return None;
        }
        *self.cache.entry(name.to_owned()).or_insert_with(|| (self.f)(name))
    }
}

fn miter(n0: Vec2, n1: Vec2) -> Vec2 {
    let denom = 1.0 + n0.dot(n1);
    let m = if denom > 1e-3 { (n0 + n1) / denom } else { n0 };
    if m.length() > 2.0 { m.normalize() * 2.0 } else { m }
}

/// Generate the mesh of a shape.
pub fn generate_shape(inp: &GenInput) -> ShapeMesh {
    let shape = inp.shape;
    let poly = tessellate(&shape.points, shape.closed, shape.tess_mode(), shape.max_points);
    let mut b = Builder::default();
    b.stats.points = poly.pts.len();
    let mut res = Resolver { f: inp.sprite, cache: HashMap::new() };
    // Holes only exist in a closed shape.
    let hole_polys: Vec<(&super::ShapeHole, Poly)> = if shape.closed {
        shape.holes.iter().filter(|h| h.points.len() >= 3).map(|h| (h, tessellate(&h.points, true, shape.tess_mode(), shape.max_points))).filter(|(_, p)| p.pts.len() >= 3).collect()
    } else {
        Vec::new()
    };
    if shape.closed && shape.fill && poly.pts.len() >= 3 {
        fill(&poly, &hole_polys.iter().map(|(_, p)| p.clone()).collect::<Vec<_>>(), inp, &mut res, &mut b);
    }
    if poly.segment_count() > 0 {
        edges(&poly, inp, &mut res, &mut b);
    }
    if !hole_polys.is_empty() {
        // The rim of a hole faces into the hole: the same edge pass on the hole's ring, flipped.
        for (hole, hp) in &hole_polys {
            let hole_shape = SpriteShapeRenderer { points: hole.points.clone(), holes: Vec::new(), closed: true, flip_edge: !shape.flip_edge, ..shape.clone() };
            let hole_inp = GenInput { shape: &hole_shape, profile: inp.profile, sprite: inp.sprite, fill_texture: inp.fill_texture };
            edges(hp, &hole_inp, &mut res, &mut b);
        }
    }
    let mut groups = b.groups;
    groups.retain(|g| !g.indices.is_empty());
    let mut lo = Vec2::splat(f32::INFINITY);
    let mut hi = Vec2::splat(f32::NEG_INFINITY);
    for g in &groups {
        for p in &g.positions {
            lo = lo.min(Vec2::from(*p));
            hi = hi.max(Vec2::from(*p));
        }
    }
    ShapeMesh { groups, bounds: (lo.x <= hi.x).then(|| (lo.to_array(), hi.to_array())), stats: b.stats }
}

fn fill(poly: &Poly, holes: &[Poly], inp: &GenInput, res: &mut Resolver, b: &mut Builder) {
    let outer = offset_ring(&poly.pts, inp.profile.fill_edge_offset);
    let (ring, tris) = if holes.is_empty() {
        let t = triangulate(&outer);
        (outer, t)
    } else {
        // A hole's fill boundary moves the other way: tucking the fill under the edge sprites
        // (negative offset) grows the hole.
        let hole_rings: Vec<Vec<Vec2>> = holes.iter().map(|h| offset_ring(&h.pts, -inp.profile.fill_edge_offset)).collect();
        triangulate_with_holes(&outer, &hole_rings)
    };
    if tris.is_empty() {
        return;
    }
    let p = inp.profile;
    let (texture, uv): (u32, Box<dyn Fn(Vec2) -> [f32; 2]>) = if let Some(t) = inp.fill_texture {
        let ppu = p.fill_pixels_per_unit.max(1e-3);
        let (tw, th) = (t.size_px[0].max(1) as f32 / ppu, t.size_px[1].max(1) as f32 / ppu);
        let off = Vec2::from(p.fill_uv_offset);
        (t.texture, Box::new(move |q: Vec2| [(q.x - off.x) / tw, (off.y - q.y) / th]))
    } else if let Some(m) = res.get(&p.fill_sprite) {
        let lo = ring.iter().fold(Vec2::splat(f32::INFINITY), |a, q| a.min(*q));
        let hi = ring.iter().fold(Vec2::splat(f32::NEG_INFINITY), |a, q| a.max(*q));
        let span = (hi - lo).max(Vec2::splat(1e-6));
        (
            m.texture,
            Box::new(move |q: Vec2| {
                let f = (q - lo) / span;
                [m.uv_rect[0] + (m.uv_rect[2] - m.uv_rect[0]) * f.x, m.uv_rect[3] + (m.uv_rect[1] - m.uv_rect[3]) * f.y]
            }),
        )
    } else {
        return;
    };
    let g = b.group(GroupKind::Fill, texture);
    if !g.room(ring.len()) {
        b.stats.truncated = true;
        return;
    }
    let base = g.positions.len();
    for q in &ring {
        g.vertex(*q, uv(*q));
    }
    for t in &tris {
        g.indices.extend_from_slice(&[(base + t[0] as usize) as u16, (base + t[1] as usize) as u16, (base + t[2] as usize) as u16]);
    }
    b.stats.fill_triangles = tris.len();
}

/// Per-run vertex data.
struct RunVert {
    pos: Vec2,
    normal_a: Vec2, // offset direction used by the segment arriving here... see below
    normal_b: Vec2, // ... and the one leaving
    height: f32,
}

fn edges(poly: &Poly, inp: &GenInput, res: &mut Resolver, b: &mut Builder) {
    let shape = inp.shape;
    let profile = inp.profile;
    let reverse = (shape.closed && signed_area(&poly.pts) > 0.0) ^ shape.flip_edge;
    let p = if reverse { poly.reversed() } else { poly.clone() };
    let n = p.pts.len();
    let segs = p.segment_count();
    let closed = p.closed;
    if segs == 0 {
        return;
    }
    let dirs: Vec<Vec2> = (0..segs).map(|s| (p.pts[p.next(s)] - p.pts[s]).normalize_or_zero()).collect();
    let normals: Vec<Vec2> = dirs.iter().map(|d| Vec2::new(-d.y, d.x)).collect();
    let range: Vec<Option<usize>> = normals.iter().map(|nm| select_range(&profile.angle_ranges, normal_angle(*nm))).collect();
    // Joint at vertex v: the segments arriving and leaving.
    let joint = |v: usize| -> Option<(usize, usize)> {
        if closed {
            Some(((v + segs - 1) % segs, v))
        } else if v >= 1 && v + 1 < n {
            Some((v - 1, v))
        } else {
            None
        }
    };
    let threshold = shape.corner_angle_threshold.max(0.0).to_radians();
    let corner_kind: Vec<Option<CornerKind>> = (0..n)
        .map(|v| {
            let (s0, s1) = joint(v)?;
            let c = p.ctrl[v]?;
            if !shape.points.get(c)?.corner {
                return None;
            }
            let turn = dirs[s0].dot(dirs[s1]).clamp(-1.0, 1.0).acos();
            if turn <= threshold {
                return None;
            }
            Some(if dirs[s0].perp_dot(dirs[s1]) > 0.0 { CornerKind::Inner } else { CornerKind::Outer })
        })
        .collect();
    let corner_sprite: Vec<Option<SpriteMetrics>> = corner_kind
        .iter()
        .map(|k| match k {
            Some(CornerKind::Outer) => res.get(&profile.outer_corner),
            Some(CornerKind::Inner) => res.get(&profile.inner_corner),
            None => None,
        })
        .collect();
    // A run ends at a joint when the angle range changes or a corner sprite goes there.
    let breaks = |v: usize| -> bool {
        match joint(v) {
            Some((s0, s1)) => range[s0] != range[s1] || corner_sprite[v].is_some(),
            None => true,
        }
    };
    // Offset direction at the start / end of a segment (a miter unless a sprite corner or an end).
    let start_normal = |s: usize| -> Vec2 {
        match joint(s) {
            Some((s0, s1)) if corner_sprite[s].is_none() => miter(normals[s0], normals[s1]),
            _ => normals[s],
        }
    };
    let end_normal = |s: usize| -> Vec2 {
        let v = p.next(s);
        match joint(v) {
            Some((s0, s1)) if corner_sprite[v].is_none() => miter(normals[s0], normals[s1]),
            _ => normals[s],
        }
    };

    // Runs of segments.
    let mut runs: Vec<(Vec<usize>, bool)> = Vec::new();
    let all_one_loop = closed && range.iter().all(|r| r.is_some()) && (0..n).all(|v| !breaks(v));
    if all_one_loop {
        runs.push(((0..segs).collect(), true));
    } else {
        for s in 0..segs {
            if range[s].is_none() {
                continue;
            }
            let starts = if closed { breaks(s) } else { s == 0 || breaks(s) };
            if !starts {
                continue;
            }
            let mut list = vec![s];
            let mut cur = s;
            loop {
                let nxt = p.next(cur);
                let next_seg = if closed { nxt % segs } else { nxt };
                if (!closed && next_seg >= segs) || breaks(nxt) || range[next_seg].is_none() || list.len() >= segs {
                    break;
                }
                list.push(next_seg);
                cur = next_seg;
            }
            runs.push((list, false));
        }
    }

    for (list, is_loop) in &runs {
        let ri = range[list[0]].expect("runs have a range");
        let sprites: Vec<Option<SpriteMetrics>> = profile.angle_ranges[ri].sprites.iter().map(|s| res.get(s)).collect();
        if sprites.iter().all(Option::is_none) {
            continue;
        }
        let verts: Vec<RunVert> = list
            .iter()
            .map(|&s| RunVert { pos: p.pts[s], normal_a: start_normal(s), normal_b: end_normal(s), height: p.height[s] })
            .collect();
        let end_height = p.height[p.next(*list.last().unwrap())];
        let end_pos = p.pts[p.next(*list.last().unwrap())];
        emit_run(&verts, end_pos, end_height, list, &p, shape, &sprites, is_loop, b);
    }

    // Corners.
    for (v, m) in corner_sprite.iter().enumerate() {
        let Some(m) = m else { continue };
        let (s0, s1) = joint(v).expect("corners are interior");
        let mut up = normals[s0] + normals[s1];
        up = if up.length() < 1e-4 { normals[s0] } else { up.normalize() };
        let h = p.height[v];
        let (w, ht) = (m.size[0] * h, m.size[1] * h);
        let r = Vec2::new(up.y, -up.x);
        let centre = p.pts[v] + up * (ht * shape.border_pivot);
        let [u0, vt, u1, vb] = m.uv_rect;
        b.quad(
            GroupKind::Edge,
            m.texture,
            [centre - r * (w * 0.5) - up * (ht * 0.5), centre + r * (w * 0.5) - up * (ht * 0.5), centre + r * (w * 0.5) + up * (ht * 0.5), centre - r * (w * 0.5) + up * (ht * 0.5)],
            [[u0, vb], [u1, vb], [u1, vt], [u0, vt]],
        );
        b.stats.corners += 1;
    }

    // Caps on open shapes. A reversed walk swaps which sprite sits at the walk's start.
    if !closed && n >= 2 {
        let (first, last) = if reverse { (&profile.end_cap, &profile.start_cap) } else { (&profile.start_cap, &profile.end_cap) };
        for (name, v, dir, normal, at_end) in [(first, 0usize, dirs[0], normals[0], false), (last, n - 1, dirs[segs - 1], normals[segs - 1], true)] {
            let Some(m) = res.get(name) else { continue };
            let h = p.height[v];
            let (w, ht) = (m.size[0] * h, m.size[1] * h);
            let (a, c) = if at_end { (p.pts[v], p.pts[v] + dir * w) } else { (p.pts[v] - dir * w, p.pts[v]) };
            let lo = -normal * (ht * (0.5 - shape.border_pivot));
            let hi = normal * (ht * (0.5 + shape.border_pivot));
            let [u0, vt, u1, vb] = m.uv_rect;
            b.quad(GroupKind::Edge, m.texture, [a + lo, c + lo, c + hi, a + hi], [[u0, vb], [u1, vb], [u1, vt], [u0, vt]]);
            b.stats.caps += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_run(verts: &[RunVert], end_pos: Vec2, end_height: f32, list: &[usize], p: &Poly, shape: &SpriteShapeRenderer, sprites: &[Option<SpriteMetrics>], _is_loop: &bool, b: &mut Builder) {
    // Cumulative arc length at the start of every run segment; the last entry is the run length.
    let k = verts.len();
    let mut cum = Vec::with_capacity(k + 1);
    cum.push(0.0f32);
    for j in 0..k {
        let a = verts[j].pos;
        let c = if j + 1 < k { verts[j + 1].pos } else { end_pos };
        cum.push(cum[j] + a.distance(c));
    }
    let length = cum[k];
    let locate = |s: f32| -> usize { (cum.partition_point(|&c| c <= s).saturating_sub(1)).min(k - 1) };
    let first_ok = sprites.iter().position(Option::is_some).expect("checked by the caller");
    let variant_at = |arc: f32| -> usize {
        let j = locate(arc);
        let owner = p.owner[list[j]];
        let want = shape.points.get(owner).map_or(0, |pt| pt.sprite_index as usize) % sprites.len();
        if sprites[want].is_some() { want } else { first_ok }
    };
    let plan = plan_tiles(length, shape.fit, |arc| {
        let v = variant_at(arc);
        (v, sprites[v].expect("resolved").size[0])
    });
    if plan.is_empty() {
        return;
    }
    b.stats.runs += 1;
    // Sample of the run at arc length `s`: position, offset direction (unscaled), height.
    let at = |s: f32| -> (Vec2, Vec2, f32) {
        let j = locate(s);
        let (a, c) = (verts[j].pos, if j + 1 < k { verts[j + 1].pos } else { end_pos });
        let span = cum[j + 1] - cum[j];
        let t = if span > 1e-9 { ((s - cum[j]) / span).clamp(0.0, 1.0) } else { 0.0 };
        let (ha, hb) = (verts[j].height, if j + 1 < k { verts[j + 1].height } else { end_height });
        (a.lerp(c, t), verts[j].normal_a.lerp(verts[j].normal_b, t), ha + (hb - ha) * t)
    };
    for tile in &plan {
        let m = sprites[tile.variant].expect("planned from resolved variants");
        let [u0, vt, u1, vb] = m.uv_rect;
        let u_end = u0 + (u1 - u0) * tile.frac;
        // Sample arcs: the tile ends and every polyline vertex strictly inside.
        let mut arcs = vec![tile.s0];
        arcs.extend(cum[1..k].iter().copied().filter(|&c| c > tile.s0 + 1e-5 && c < tile.s1 - 1e-5));
        arcs.push(tile.s1);
        let span = (tile.s1 - tile.s0).max(1e-9);
        let truncated = {
            let g = b.group(GroupKind::Edge, m.texture);
            if !g.room(arcs.len() * 2) {
                true
            } else {
                let mut prev: Option<(u16, u16)> = None; // (outer, inner)
                for &s in &arcs {
                    let (pos, nrm, h) = at(s);
                    let thick = m.size[1] * h;
                    let u = u0 + (u_end - u0) * ((s - tile.s0) / span);
                    let outer = g.vertex(pos + nrm * (thick * (0.5 + shape.border_pivot)), [u, vt]);
                    let inner = g.vertex(pos - nrm * (thick * (0.5 - shape.border_pivot)), [u, vb]);
                    if let Some((po, pi)) = prev {
                        g.indices.extend_from_slice(&[pi, inner, outer, pi, outer, po]);
                    }
                    prev = Some((outer, inner));
                }
                false
            }
        };
        b.stats.truncated |= truncated;
        b.stats.tiles += 1;
    }
}

/// A collider outline: a polyline (open shapes -> Edge Collider 2D) or a ring
/// (closed shapes -> Polygon Collider 2D), in object-local units.
#[derive(Clone, Debug, PartialEq)]
pub struct ColliderOutline {
    /// The outline's points.
    pub points: Vec<[f32; 2]>,
    /// Closed (a ring) or open (a polyline).
    pub closed: bool,
    /// Closed shapes: one extra ring per hole (a Polygon Collider 2D path each; the collider
    /// fills paths even-odd, so these cut the outer one).
    pub holes: Vec<Vec<[f32; 2]>>,
}

/// The tessellated spline at the shape's collider detail, optionally
/// Douglas-Peucker simplified and (closed) offset. With `collider_optimize == 0`
/// and no offset the points are exactly the fixed tessellation of the spline.
pub fn collider_outline(shape: &SpriteShapeRenderer) -> ColliderOutline {
    let poly = tessellate(&shape.points, shape.closed, TessMode::Fixed { detail: shape.collider_detail.max(1) }, shape.max_points);
    let mut pts = poly.pts;
    if shape.collider_optimize > 0.0 {
        pts = simplify(&pts, shape.collider_optimize, shape.closed);
    }
    if shape.closed && shape.collider_offset != 0.0 {
        pts = offset_ring(&pts, shape.collider_offset);
    }
    let holes = if shape.closed {
        shape.holes.iter().filter(|h| h.points.len() >= 3).map(|h| {
            let hp = tessellate(&h.points, true, TessMode::Fixed { detail: shape.collider_detail.max(1) }, shape.max_points);
            let mut pts = hp.pts;
            if shape.collider_optimize > 0.0 {
                pts = simplify(&pts, shape.collider_optimize, true);
            }
            if shape.collider_offset != 0.0 {
                pts = offset_ring(&pts, -shape.collider_offset);
            }
            pts.into_iter().map(|p| p.to_array()).collect::<Vec<_>>()
        }).filter(|r| r.len() >= 3).collect()
    } else {
        Vec::new()
    };
    ColliderOutline { points: pts.into_iter().map(|p| p.to_array()).collect(), closed: shape.closed, holes }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sprite_shape::spline::distance_to_polyline;
    use crate::sprite_shape::{AngleRange, ShapePoint, ShapeTangentMode, Tessellation};

    /// Metrics for fake sprites named `edge`, `grass`, `corner_o`, `corner_i`,
    /// `cap_s`, `cap_e`, `fill`, `wide`: each its own texture so groups can be told apart.
    fn sprite(name: &str) -> Option<SpriteMetrics> {
        let (tex, w, h) = match name {
            "edge" => (1, 1.0, 0.5),
            "grass" => (2, 0.5, 0.5),
            "corner_o" => (3, 0.6, 0.6),
            "corner_i" => (4, 0.6, 0.6),
            "cap_s" => (5, 0.4, 0.5),
            "cap_e" => (6, 0.4, 0.5),
            "fillsprite" => (7, 1.0, 1.0),
            "wide" => (1, 2.0, 0.5),
            _ => return None,
        };
        Some(SpriteMetrics { texture: tex, uv_rect: [0.1, 0.2, 0.9, 0.8], size: [w, h] })
    }

    fn profile(sprites: &[&str]) -> SpriteShapeProfile {
        SpriteShapeProfile {
            angle_ranges: vec![AngleRange { start: 0.0, end: 360.0, order: 0, sprites: sprites.iter().map(|s| s.to_string()).collect() }],
            ..Default::default()
        }
    }

    fn square(size: f32, closed: bool) -> SpriteShapeRenderer {
        // Clockwise: top-left -> top-right -> bottom-right -> bottom-left.
        let s = size / 2.0;
        SpriteShapeRenderer { points: vec![ShapePoint::at(-s, s), ShapePoint::at(s, s), ShapePoint::at(s, -s), ShapePoint::at(-s, -s)], closed, ..Default::default() }
    }

    fn build(shape: &SpriteShapeRenderer, profile: &SpriteShapeProfile, fill: Option<FillTexture>) -> ShapeMesh {
        generate_shape(&GenInput { shape, profile, sprite: &sprite, fill_texture: fill })
    }

    fn group(m: &ShapeMesh, kind: GroupKind, tex: u32) -> Option<&MeshGroup> {
        m.groups.iter().find(|g| g.kind == kind && g.texture == tex)
    }

    fn group_area(g: &MeshGroup) -> f32 {
        g.indices
            .chunks(3)
            .map(|t| {
                let (a, b, c) = (Vec2::from(g.positions[t[0] as usize]), Vec2::from(g.positions[t[1] as usize]), Vec2::from(g.positions[t[2] as usize]));
                (b - a).perp_dot(c - a) * 0.5
            })
            .sum()
    }

    fn ring_area(shape: &SpriteShapeRenderer) -> f32 {
        let poly = tessellate(&shape.points, shape.closed, shape.tess_mode(), shape.max_points);
        signed_area(&poly.pts).abs()
    }

    #[test]
    fn fill_area_matches_the_shoelace_formula_for_open_and_closed_input() {
        let tex = Some(FillTexture { texture: 9, size_px: [64, 64] });
        let p = profile(&[]);
        // Closed square: fill = 16, no edge sprites.
        let m = build(&square(4.0, true), &p, tex);
        let g = group(&m, GroupKind::Fill, 9).unwrap();
        assert!((group_area(g) - 16.0).abs() < 1e-4);
        assert!(g.indices.chunks(3).all(|t| {
            let (a, b, c) = (Vec2::from(g.positions[t[0] as usize]), Vec2::from(g.positions[t[1] as usize]), Vec2::from(g.positions[t[2] as usize]));
            (b - a).perp_dot(c - a) > 0.0
        }));
        // Counter-clockwise input gives the same fill.
        let mut ccw = square(4.0, true);
        ccw.points.reverse();
        assert!((group_area(group(&build(&ccw, &p, tex), GroupKind::Fill, 9).unwrap()) - 16.0).abs() < 1e-4);
        // A curved blob: area equals the shoelace area of its tessellated outline.
        let mut blob = SpriteShapeRenderer { points: vec![ShapePoint::at(-3.0, 0.0), ShapePoint::at(0.0, 3.0), ShapePoint::at(3.0, 0.0), ShapePoint::at(0.0, -3.0)], closed: true, ..Default::default() };
        for pt in &mut blob.points {
            let along = [-pt.position[1], pt.position[0]];
            pt.set_mode(ShapeTangentMode::Continuous, [-along[0], -along[1]], 1.6);
        }
        let m = build(&blob, &p, tex);
        let a = group_area(group(&m, GroupKind::Fill, 9).unwrap());
        assert!((a - ring_area(&blob)).abs() < 1e-2, "{a} vs {}", ring_area(&blob));
        assert!(a > 20.0 && a < 29.0, "round-ish blob of radius 3 (28.3): {a}");
        // An open shape has no fill; `fill: false` turns it off for a closed one.
        let open = build(&square(4.0, false), &p, tex);
        assert!(group(&open, GroupKind::Fill, 9).is_none());
        let mut nofill = square(4.0, true);
        nofill.fill = false;
        assert!(build(&nofill, &p, tex).groups.is_empty());
    }

    fn hole_area(shape: &SpriteShapeRenderer) -> f32 {
        shape.holes.iter().map(|h| signed_area(&tessellate(&h.points, true, shape.tess_mode(), shape.max_points).pts).abs()).sum()
    }

    #[test]
    fn fill_with_holes_is_outer_minus_holes_by_the_shoelace_formula() {
        let tex = Some(FillTexture { texture: 9, size_px: [64, 64] });
        let p = profile(&[]);
        let fill_area = |s: &SpriteShapeRenderer| group_area(group(&build(s, &p, tex), GroupKind::Fill, 9).unwrap());
        let tris_ccw = |s: &SpriteShapeRenderer| {
            let m = build(s, &p, tex);
            let g = group(&m, GroupKind::Fill, 9).unwrap();
            g.indices.chunks(3).all(|t| (Vec2::from(g.positions[t[1] as usize]) - Vec2::from(g.positions[t[0] as usize])).perp_dot(Vec2::from(g.positions[t[2] as usize]) - Vec2::from(g.positions[t[0] as usize])) >= -1e-6)
        };
        // One square hole.
        let mut s = square(8.0, true);
        s.holes = vec![crate::sprite_shape::ShapeHole { points: vec![ShapePoint::at(-1.0, 1.0), ShapePoint::at(1.0, 1.0), ShapePoint::at(1.0, -1.0), ShapePoint::at(-1.0, -1.0)] }];
        assert!((fill_area(&s) - (64.0 - 4.0)).abs() < 1e-3, "{}", fill_area(&s));
        assert!(tris_ccw(&s));
        // A counter-clockwise hole subtracts the same.
        s.holes[0].points.reverse();
        assert!((fill_area(&s) - 60.0).abs() < 1e-3);
        // Three holes of different sizes, an off-centre one near the border.
        s.holes = vec![
            crate::sprite_shape::ShapeHole::ngon([-2.5, 0.0], 1.0, 12),
            crate::sprite_shape::ShapeHole::ngon([2.0, 2.0], 1.5, 7),
            crate::sprite_shape::ShapeHole::ngon([2.5, -2.8], 0.8, 5),
        ];
        let want = 64.0 - hole_area(&s);
        assert!((fill_area(&s) - want).abs() < 1e-2, "{} vs {want}", fill_area(&s));
        assert!(tris_ccw(&s));
        // A concave (L-shaped) outer with a hole in each arm; curved outline with a curved hole.
        let l = SpriteShapeRenderer {
            points: [(-4.0, 4.0), (0.0, 4.0), (0.0, 0.0), (4.0, 0.0), (4.0, -4.0), (-4.0, -4.0)].iter().map(|p| ShapePoint::at(p.0, p.1)).collect(),
            closed: true,
            holes: vec![crate::sprite_shape::ShapeHole::ngon([-2.0, 2.0], 1.0, 16), crate::sprite_shape::ShapeHole::ngon([2.0, -2.0], 1.5, 16)],
            ..Default::default()
        };
        let want = signed_area(&tessellate(&l.points, true, l.tess_mode(), l.max_points).pts).abs() - hole_area(&l);
        assert!((fill_area(&l) - want).abs() < 1e-2, "{} vs {want}", fill_area(&l));
        assert!(tris_ccw(&l));
        let mut blob = SpriteShapeRenderer { points: vec![ShapePoint::at(-3.0, 0.0), ShapePoint::at(0.0, 3.0), ShapePoint::at(3.0, 0.0), ShapePoint::at(0.0, -3.0)], closed: true, ..Default::default() };
        for pt in &mut blob.points {
            let along = [-pt.position[1], pt.position[0]];
            pt.set_mode(ShapeTangentMode::Continuous, [-along[0], -along[1]], 1.6);
        }
        let mut hole = crate::sprite_shape::ShapeHole::ngon([0.0, 0.0], 1.0, 4);
        for pt in &mut hole.points {
            let along = [-pt.position[1], pt.position[0]];
            pt.set_mode(ShapeTangentMode::Continuous, [-along[0], -along[1]], 0.55);
        }
        blob.holes = vec![hole];
        let want = ring_area(&blob) - hole_area(&blob);
        assert!((fill_area(&blob) - want).abs() < 1e-2, "{} vs {want}", fill_area(&blob));
        // Holes of an open shape are ignored; a degenerate hole (2 points) too.
        let mut open = square(4.0, false);
        open.holes = vec![crate::sprite_shape::ShapeHole::ngon([0.0, 0.0], 1.0, 4)];
        assert!(group(&build(&open, &p, tex), GroupKind::Fill, 9).is_none());
        let mut deg = square(4.0, true);
        deg.holes = vec![crate::sprite_shape::ShapeHole { points: vec![ShapePoint::at(0.0, 0.0), ShapePoint::at(1.0, 0.0)] }];
        assert!((fill_area(&deg) - 16.0).abs() < 1e-3);
        // Fill edge offset shrinks the fill and grows the hole.
        let mut shrunk = square(8.0, true);
        shrunk.holes = vec![crate::sprite_shape::ShapeHole::ngon([0.0, 0.0], 1.0, 4)];
        let mut p2 = p.clone();
        p2.fill_edge_offset = -0.5;
        let a = group_area(group(&build(&shrunk, &p2, tex), GroupKind::Fill, 9).unwrap());
        assert!(a < 64.0 - 2.0 - 8.0, "outer 7x7=49 minus a grown hole: {a}");
    }

    #[test]
    fn holes_get_their_own_inward_facing_edges_and_collider_ring() {
        let p = profile(&["edge"]);
        let mut s = square(8.0, true);
        s.fill = false;
        s.holes = vec![crate::sprite_shape::ShapeHole::ngon([0.0, 0.0], 1.5, 4)];
        let none = build(&SpriteShapeRenderer { holes: vec![], ..s.clone() }, &p, None);
        let with = build(&s, &p, None);
        assert!(with.triangle_count() > none.triangle_count(), "the hole rim has edge sprites: {} vs {}", with.triangle_count(), none.triangle_count());
        // Every rim vertex of the hole lies near the hole ring (not out at the border).
        let hole_ring = tessellate(&s.holes[0].points, true, s.tess_mode(), s.max_points).pts;
        let g = group(&with, GroupKind::Edge, 1).unwrap();
        let near_hole = g.positions.iter().filter(|q| distance_to_polyline(Vec2::from(**q), &hole_ring, true) < 0.5 + 1e-3).count();
        assert!(near_hole >= 8, "{near_hole}");
        let o = collider_outline(&s);
        assert_eq!((o.points.len(), o.holes.len()), (4, 1));
        assert!(o.holes[0].len() >= 3);
        assert!(collider_outline(&SpriteShapeRenderer { closed: false, ..s.clone() }).holes.is_empty());
    }

    #[test]
    fn holes_change_the_content_hash_and_round_trip() {
        let mut s = square(4.0, true);
        let h0 = s.content_hash();
        s.holes.push(crate::sprite_shape::ShapeHole::ngon([0.0, 0.0], 0.5, 6));
        let h1 = s.content_hash();
        assert_ne!(h0, h1);
        s.holes[0].points[0].position[0] += 0.1;
        assert_ne!(h1, s.content_hash());
        let text = ron::to_string(&s).unwrap();
        let back: SpriteShapeRenderer = ron::from_str(&text).unwrap();
        assert_eq!(back, s);
        // An old file without `holes` still loads.
        let old: SpriteShapeRenderer = ron::from_str("(closed: true)").unwrap();
        assert!(old.holes.is_empty());
    }

    #[test]
    fn fill_uvs_tile_by_world_units_and_offset() {
        let mut p = profile(&[]);
        p.fill_pixels_per_unit = 32.0; // a 64 px texture repeats every 2 units
        let tex = Some(FillTexture { texture: 9, size_px: [64, 64] });
        let m = build(&square(4.0, true), &p, tex);
        let g = &m.groups[0];
        for (pos, uv) in g.positions.iter().zip(&g.uvs) {
            assert!((uv[0] - pos[0] / 2.0).abs() < 1e-5 && (uv[1] + pos[1] / 2.0).abs() < 1e-5, "{pos:?} -> {uv:?}");
        }
        p.fill_uv_offset = [1.0, 0.0];
        let g2 = build(&square(4.0, true), &p, tex).groups.remove(0);
        assert!((g2.uvs[0][0] - (g.uvs[0][0] - 0.5)).abs() < 1e-5, "offset shifts the pattern");
        // The pattern is anchored to local space: a larger shape continues it.
        let big = build(&square(8.0, true), &p, tex).groups.remove(0);
        let uv_at = |gr: &MeshGroup, x: f32, y: f32| gr.positions.iter().position(|q| (q[0] - x).abs() < 1e-5 && (q[1] - y).abs() < 1e-5).map(|i| gr.uvs[i]);
        assert!(uv_at(&g2, 2.0, 2.0).is_some() && uv_at(&big, 4.0, 4.0).is_some());
    }

    #[test]
    fn fill_sprite_stretches_over_the_bounds_and_inset_shrinks_the_fill() {
        let mut p = profile(&[]);
        p.fill_sprite = "fillsprite".into();
        let m = build(&square(4.0, true), &p, None);
        let g = group(&m, GroupKind::Fill, 7).unwrap();
        let i = g.positions.iter().position(|q| *q == [-2.0, 2.0]).unwrap(); // top-left
        let near = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).abs() < 1e-5 && (a[1] - b[1]).abs() < 1e-5;
        assert!(near(g.uvs[i], [0.1, 0.2]));
        let j = g.positions.iter().position(|q| *q == [2.0, -2.0]).unwrap();
        assert!(near(g.uvs[j], [0.9, 0.8]));
        p.fill_edge_offset = -0.5;
        let g = build(&square(4.0, true), &p, None).groups.remove(0);
        assert!((group_area(&g) - 9.0).abs() < 1e-4, "{}", group_area(&g));
        // An unresolved fill sprite draws no fill.
        p.fill_sprite = "nope".into();
        assert!(build(&square(4.0, true), &p, None).groups.is_empty());
    }

    #[test]
    fn tile_plans_stretch_fit_and_cut() {
        let one = |_: f32| (0usize, 1.0f32);
        for (len, want_n) in [(3.4, 3usize), (3.6, 4), (0.3, 1), (1.0, 1), (5.0, 5), (2.5, 3)] {
            let t = plan_tiles(len, EdgeFit::Stretch, one);
            assert_eq!(t.len(), want_n, "stretch {len}");
            assert!((t.last().unwrap().s1 - len).abs() < 1e-5, "fills the run exactly");
            assert!(t.windows(2).all(|w| (w[0].s1 - w[1].s0).abs() < 1e-6), "no gaps");
            if len >= 1.0 {
                let f = (t[0].s1 - t[0].s0) / 1.0;
                assert!(f > 0.66 && f < 1.34 + 1e-3, "stretch factor {f} for length {len}");
            }
        }
        for (len, want_n) in [(3.4, 3usize), (3.99, 3), (4.0, 4), (0.4, 1), (7.2, 7)] {
            let t = plan_tiles(len, EdgeFit::Fit, one);
            assert_eq!(t.len(), want_n, "fit {len}");
            let w = t[0].s1 - t[0].s0;
            assert!(len < 1.0 || (1.0 - 1e-5..2.0).contains(&w), "tiles only widen: {w}");
            assert!((t.last().unwrap().s1 - len).abs() < 1e-5);
        }
        let t = plan_tiles(3.4, EdgeFit::Tile, one);
        assert_eq!(t.len(), 4);
        assert!((t[0].s1 - t[0].s0 - 1.0).abs() < 1e-6 && t[0].frac == 1.0);
        assert!((t[3].frac - 0.4).abs() < 1e-5 && (t[3].s1 - 3.4).abs() < 1e-6);
        let exact = plan_tiles(3.0, EdgeFit::Tile, one);
        assert_eq!(exact.len(), 3);
        assert!(exact.iter().all(|t| t.frac == 1.0));
        // Variants of different widths fill exactly too.
        let mixed = |arc: f32| if (arc as i32) % 2 == 0 { (0usize, 1.0f32) } else { (1, 0.6) };
        let t = plan_tiles(5.0, EdgeFit::Stretch, mixed);
        assert!((t.last().unwrap().s1 - 5.0).abs() < 1e-5 && t.iter().any(|x| x.variant == 1));
        // Degenerate inputs.
        assert!(plan_tiles(0.0, EdgeFit::Stretch, one).is_empty());
        assert_eq!(plan_tiles(10.0, EdgeFit::Tile, |_| (0, 0.0)).len(), MAX_TILES, "a zero-width sprite cannot run away");
    }

    #[test]
    fn edge_strip_follows_the_line_with_continuous_uvs() {
        let shape = SpriteShapeRenderer { points: vec![ShapePoint::at(0.0, 0.0), ShapePoint::at(3.4, 0.0)], fit: EdgeFit::Stretch, ..Default::default() };
        let p = profile(&["edge"]);
        let m = build(&shape, &p, None);
        let g = group(&m, GroupKind::Edge, 1).unwrap();
        assert_eq!(m.stats.tiles, 3);
        assert_eq!(g.positions.len(), 12, "3 tiles x 2 samples x 2 sides");
        // Heights: sprite height 0.5, centred on the spline.
        assert!(g.positions.iter().all(|q| (q[1].abs() - 0.25).abs() < 1e-5));
        // Outer (top) vertices carry v_top, inner v_bottom.
        for (q, uv) in g.positions.iter().zip(&g.uvs) {
            assert_eq!(uv[1], if q[1] > 0.0 { 0.2 } else { 0.8 });
        }
        // Per tile: u runs u0..u1 across the tile length, and tiles are contiguous in space.
        let tile_w = 3.4 / 3.0;
        for t in 0..3 {
            let (a, b) = (t * 4, t * 4 + 2);
            assert!((g.positions[a][0] - tile_w * t as f32).abs() < 1e-4);
            assert!((g.positions[b][0] - tile_w * (t + 1) as f32).abs() < 1e-4);
            assert!((g.uvs[a][0] - 0.1).abs() < 1e-6 && (g.uvs[b][0] - 0.9).abs() < 1e-6);
        }
        // Texel density along the edge is constant to the stretch factor.
        let density = (g.uvs[2][0] - g.uvs[0][0]) / (g.positions[2][0] - g.positions[0][0]);
        assert!((density - 0.8 / tile_w).abs() < 1e-4);
        // Triangles face the viewer.
        assert!(group_area(g) > 0.0);
        assert!((group_area(g) - 3.4 * 0.5).abs() < 1e-4, "strip area = length x sprite height");
    }

    #[test]
    fn uvs_stay_continuous_along_a_curved_edge() {
        let mut a = ShapePoint::at(-4.0, 0.0);
        a.set_mode(ShapeTangentMode::Broken, [0.0, 1.0], 3.0);
        let mut b = ShapePoint::at(4.0, 0.0);
        b.set_mode(ShapeTangentMode::Broken, [0.0, 1.0], 3.0);
        b.tangent_in = [0.0, 3.0];
        let shape = SpriteShapeRenderer { points: vec![a, b], tessellation: Tessellation::Fixed, spline_detail: 24, ..Default::default() };
        let m = build(&shape, &profile(&["edge"]), None);
        let g = group(&m, GroupKind::Edge, 1).unwrap();
        // The outer vertices form the strip's top edge: arc-length parametrised u inside each tile.
        let outer: Vec<(Vec2, f32)> = g.positions.iter().zip(&g.uvs).filter(|(_, uv)| uv[1] == 0.2).map(|(p, uv)| (Vec2::from(*p), uv[0])).collect();
        let centre: Vec<Vec2> = g.positions.chunks(2).map(|c| (Vec2::from(c[0]) + Vec2::from(c[1])) * 0.5).collect();
        let mut worst_ratio = 0.0f32;
        let mut seams = 0;
        for i in 1..centre.len() {
            let du = outer[i].1 - outer[i - 1].1;
            if du < -1e-6 {
                seams += 1; // a new tile restarts u: its start sample repeats the previous end position
                assert!(centre[i].distance(centre[i - 1]) < 1e-4, "tiles meet without a gap");
                continue;
            }
            // u advances in proportion to the centre line's length: texel density within 1% of the tile average.
            let dl = centre[i].distance(centre[i - 1]);
            if dl > 1e-5 {
                worst_ratio = worst_ratio.max((du / dl / (0.8 / (shape_len(&shape) / m.stats.tiles as f32)) - 1.0).abs());
            }
        }
        assert!(m.stats.tiles >= 6 && seams == m.stats.tiles - 1, "tiles {} seams {seams}", m.stats.tiles);
        assert!(worst_ratio < 0.08, "uv stretch beyond tolerance: {worst_ratio}");
    }

    fn shape_len(s: &SpriteShapeRenderer) -> f32 {
        tessellate(&s.points, s.closed, s.tess_mode(), s.max_points).length()
    }

    #[test]
    fn point_height_scales_the_thickness() {
        let mut shape = SpriteShapeRenderer { points: vec![ShapePoint::at(0.0, 0.0), ShapePoint::at(2.0, 0.0)], ..Default::default() };
        shape.points[0].height = 1.0;
        shape.points[1].height = 3.0;
        let m = build(&shape, &profile(&["edge"]), None);
        let g = group(&m, GroupKind::Edge, 1).unwrap();
        assert_eq!(m.stats.tiles, 2);
        let thickness_at = |x: f32| {
            let ys: Vec<f32> = g.positions.iter().filter(|q| (q[0] - x).abs() < 1e-4).map(|q| q[1]).collect();
            ys.iter().cloned().fold(f32::MIN, f32::max) - ys.iter().cloned().fold(f32::MAX, f32::min)
        };
        assert!((thickness_at(0.0) - 0.5).abs() < 1e-5);
        assert!((thickness_at(1.0) - 1.0).abs() < 1e-5, "halfway: height 2");
        assert!((thickness_at(2.0) - 1.5).abs() < 1e-5);
        // Zero height collapses the strip; it never goes negative.
        shape.points[1].height = 0.0;
        let g = &build(&shape, &profile(&["edge"]), None).groups[0].clone();
        assert!(g.positions.iter().all(|q| q[1].abs() <= 0.25 + 1e-6));
    }

    #[test]
    fn border_pivot_shifts_the_strip_along_the_normal() {
        let mut shape = SpriteShapeRenderer { points: vec![ShapePoint::at(0.0, 0.0), ShapePoint::at(1.0, 0.0)], border_pivot: 0.5, ..Default::default() };
        let g = build(&shape, &profile(&["edge"]), None).groups.remove(0);
        let (lo, hi) = g.positions.iter().fold((f32::MAX, f32::MIN), |a, q| (a.0.min(q[1]), a.1.max(q[1])));
        assert!(lo.abs() < 1e-6 && (hi - 0.5).abs() < 1e-6, "sprite sits fully on the normal side: {lo}..{hi}");
        shape.flip_edge = true;
        let g = build(&shape, &profile(&["edge"]), None).groups.remove(0);
        let (lo, hi) = g.positions.iter().fold((f32::MAX, f32::MIN), |a, q| (a.0.min(q[1]), a.1.max(q[1])));
        assert!(hi.abs() < 1e-6 && (lo + 0.5).abs() < 1e-6, "flipped: the other side");
    }

    #[test]
    fn angle_ranges_pick_sprites_per_segment_with_wraparound() {
        // A closed square (clockwise): top segment faces up (0 deg), right faces 270, bottom 180, left 90.
        let mut p = SpriteShapeProfile {
            angle_ranges: vec![
                AngleRange { start: -45.0, end: 45.0, order: 0, sprites: vec!["edge".into()] },   // up, wraps through 0
                AngleRange { start: 45.0, end: 315.0, order: 0, sprites: vec!["grass".into()] },  // everything else
            ],
            ..Default::default()
        };
        let mut shape = square(4.0, true);
        shape.fill = false;
        shape.corner_angle_threshold = 179.0; // no corners: isolate the selection
        let m = build(&shape, &p, None);
        let edge = group(&m, GroupKind::Edge, 1).unwrap();
        let grass = group(&m, GroupKind::Edge, 2).unwrap();
        // The top run is 4 long (4 tiles of 1.0); the other three sides share the other sprite.
        assert!((group_area(edge) - 4.0 * 0.5).abs() < 1e-3, "top only: {}", group_area(edge));
        assert!((group_area(grass) - 12.0 * 0.5).abs() < 1e-3, "left, bottom and right: {}", group_area(grass));
        assert!(edge.positions.iter().all(|q| q[1] > 1.5), "the up-facing sprite is on the top side");
        // Overlap: a higher order range takes the bottom segment from the catch-all.
        p.angle_ranges.push(AngleRange { start: 135.0, end: 225.0, order: 5, sprites: vec!["wide".into()] });
        let m = build(&shape, &p, None);
        assert!((group_area(group(&m, GroupKind::Edge, 2).unwrap()) - 4.0).abs() < 1e-3, "grass keeps only the left and right sides");
        // wide shares texture 1 with edge: the bottom adds 4 x 0.5 area to that group.
        assert!((group_area(group(&m, GroupKind::Edge, 1).unwrap()) - 4.0).abs() < 1e-3);
        // A gap in the ranges leaves those segments without an edge sprite.
        let gap = SpriteShapeProfile {
            angle_ranges: vec![AngleRange { start: -45.0, end: 45.0, order: 0, sprites: vec!["edge".into()] }],
            ..Default::default()
        };
        let m = build(&shape, &gap, None);
        assert_eq!(m.groups.len(), 1);
        assert!(m.stats.runs == 1);
    }

    #[test]
    fn sprite_variants_follow_the_start_point_of_each_segment() {
        let mut shape = SpriteShapeRenderer { points: vec![ShapePoint::at(0.0, 0.0), ShapePoint::at(2.0, 0.0), ShapePoint::at(4.0, 0.0)], fit: EdgeFit::Stretch, ..Default::default() };
        shape.points[0].sprite_index = 0;
        shape.points[1].sprite_index = 1;
        // Variants of different texture, so the groups tell them apart.
        let p = profile(&["edge", "grass"]);
        let m = build(&shape, &p, None);
        let edge = group(&m, GroupKind::Edge, 1).unwrap();
        let grass = group(&m, GroupKind::Edge, 2).unwrap();
        assert!(edge.positions.iter().all(|q| q[0] <= 2.0 + 1e-4), "first segment uses variant 0");
        assert!(grass.positions.iter().all(|q| q[0] >= 2.0 - 1e-4), "second uses variant 1");
        // Index wraps; a missing sprite falls back to the first that resolves.
        shape.points[1].sprite_index = 3;
        let m = build(&shape, &profile(&["edge", "grass"]), None);
        assert!(group(&m, GroupKind::Edge, 2).is_some(), "3 % 2 = 1");
        let m = build(&shape, &profile(&["edge", "missing"]), None);
        assert_eq!(m.groups.len(), 1);
        assert_eq!(m.groups[0].texture, 1);
        assert!(build(&shape, &profile(&["missing"]), None).groups.is_empty());
    }

    fn square_with_corners() -> (SpriteShapeRenderer, SpriteShapeProfile) {
        let mut shape = square(4.0, true);
        shape.fill = false;
        let mut p = profile(&["edge"]);
        p.outer_corner = "corner_o".into();
        p.inner_corner = "corner_i".into();
        shape.corner_angle_threshold = 30.0;
        (shape, p)
    }

    #[test]
    fn sharp_corners_get_corner_sprites_and_break_runs() {
        let (shape, p) = square_with_corners();
        let m = build(&shape, &p, None);
        assert_eq!(m.stats.corners, 4, "four 90 degree corners");
        assert_eq!(m.stats.runs, 4, "each side is its own run");
        let c = group(&m, GroupKind::Edge, 3).unwrap();
        assert_eq!(c.positions.len(), 16, "four outer-corner quads, no inner ones");
        assert!(group(&m, GroupKind::Edge, 4).is_none());
        // Each corner quad is centred on a square corner and sized to the sprite.
        let corners = [[-2.0, 2.0], [2.0, 2.0], [2.0, -2.0], [-2.0, -2.0]];
        for q in c.positions.chunks(4) {
            let centre = (Vec2::from(q[0]) + Vec2::from(q[2])) * 0.5;
            assert!(corners.iter().any(|k| Vec2::from(*k).distance(centre) < 1e-4), "centre {centre}");
            assert!((Vec2::from(q[0]).distance(Vec2::from(q[1])) - 0.6).abs() < 1e-4);
        }
        // The corner quad is rotated to the diagonal: its edges are not axis aligned.
        let q = &c.positions[0..4];
        assert!((q[1][1] - q[0][1]).abs() > 0.3, "rotated to the bisector");
        // Concave corner on a reversed (flipped) ring... an L shape has one inner corner.
        let mut l = SpriteShapeRenderer {
            points: [[0., 4.], [2., 4.], [2., 2.], [4., 2.], [4., 0.], [0., 0.]].iter().map(|p| ShapePoint::at(p[0], p[1])).collect(),
            closed: true,
            fill: false,
            ..Default::default()
        };
        l.corner_angle_threshold = 30.0;
        let m = build(&l, &p, None);
        assert_eq!(m.stats.corners, 6);
        assert_eq!(group(&m, GroupKind::Edge, 4).unwrap().positions.len(), 4, "exactly one inner corner");
        assert_eq!(group(&m, GroupKind::Edge, 3).unwrap().positions.len(), 20);
    }

    #[test]
    fn corner_threshold_flag_and_missing_sprites() {
        let (mut shape, p) = square_with_corners();
        shape.corner_angle_threshold = 100.0; // 90 degree turns no longer qualify
        let m = build(&shape, &p, None);
        assert_eq!(m.stats.corners, 0);
        assert_eq!(m.stats.runs, 1, "one closed run");
        shape.corner_angle_threshold = 30.0;
        shape.points[1].corner = false;
        assert_eq!(build(&shape, &p, None).stats.corners, 3, "a point can opt out");
        // No corner sprite in the profile: miter join, a single closed run, no gap at the corners.
        let mut bare = p.clone();
        bare.outer_corner.clear();
        shape.points[1].corner = true;
        let m = build(&shape, &bare, None);
        assert_eq!(m.stats.corners, 0);
        assert_eq!(m.stats.runs, 1);
        let g = &m.groups[0];
        // The mitered strip reaches the outer corner point exactly, so its extent is 4.5 (centre line 4 + 0.25 each side).
        let max = g.positions.iter().fold(0.0f32, |a, q| a.max(q[0].abs()).max(q[1].abs()));
        assert!((max - 2.25).abs() < 1e-4, "{max}");
        // Smooth curves never produce corners.
        let mut round = SpriteShapeRenderer { points: (0..8).map(|i| ShapePoint::at(3.0 * (i as f32 * 0.785).cos(), 3.0 * (i as f32 * 0.785).sin())).collect(), closed: true, fill: false, ..Default::default() };
        round.corner_angle_threshold = 60.0;
        assert_eq!(build(&round, &p, None).stats.corners, 0, "45 degree turns are under the 60 degree threshold");
    }

    #[test]
    fn caps_sit_beyond_the_ends_of_open_shapes() {
        let mut p = profile(&["edge"]);
        p.start_cap = "cap_s".into();
        p.end_cap = "cap_e".into();
        let shape = SpriteShapeRenderer { points: vec![ShapePoint::at(0.0, 0.0), ShapePoint::at(3.0, 0.0)], ..Default::default() };
        let m = build(&shape, &p, None);
        assert_eq!(m.stats.caps, 2);
        let s = group(&m, GroupKind::Edge, 5).unwrap();
        let e = group(&m, GroupKind::Edge, 6).unwrap();
        let xs = |g: &MeshGroup| (g.positions.iter().map(|q| q[0]).fold(f32::MAX, f32::min), g.positions.iter().map(|q| q[0]).fold(f32::MIN, f32::max));
        let (a, b) = xs(s);
        assert!((a + 0.4).abs() < 1e-5 && b.abs() < 1e-5, "start cap before the first point: {a}..{b}");
        let (a, b) = xs(e);
        assert!((a - 3.0).abs() < 1e-5 && (b - 3.4).abs() < 1e-5);
        // u runs left to right (image right is attached to the path at the start cap).
        let i = s.positions.iter().position(|q| q[0] < -0.39).unwrap();
        assert_eq!(s.uvs[i][0], 0.1);
        // Flipped walk: the start cap stays at the first control point.
        let mut flipped = shape.clone();
        flipped.flip_edge = true;
        let m = build(&flipped, &p, None);
        let s = group(&m, GroupKind::Edge, 5).unwrap();
        let (a, b) = xs(s);
        assert!((a + 0.4).abs() < 1e-5 && b.abs() < 1e-5, "the start cap stays at the first control point, outside the path: {a}..{b}");
        // Closed shapes have no caps.
        assert_eq!(build(&square(4.0, true), &p, None).stats.caps, 0);
    }

    #[test]
    fn closed_loop_without_breaks_is_one_watertight_run() {
        let mut round = SpriteShapeRenderer { points: (0..8).map(|i| {
            let a = -(i as f32) * std::f32::consts::TAU / 8.0;
            ShapePoint::at(3.0 * a.cos(), 3.0 * a.sin())
        }).collect(), closed: true, fill: false, ..Default::default() };
        for p in &mut round.points {
            let r = Vec2::from(p.position);
            p.set_mode(ShapeTangentMode::Continuous, [r.y, -r.x], 1.2);
        }
        let m = build(&round, &profile(&["edge"]), None);
        assert_eq!(m.stats.runs, 1);
        let g = &m.groups[0];
        // Every tile's start repeats the previous tile's end vertex-for-vertex (positions), including the wrap.
        let first = Vec2::from(g.positions[0]);
        let last_outer = Vec2::from(g.positions[g.positions.len() - 2]);
        assert!(first.distance(last_outer) < 1e-4, "the loop closes: {first} vs {last_outer}");
        let total_len = shape_len(&round);
        assert!((group_area(g) - total_len * 0.5).abs() / (total_len * 0.5) < 0.06, "strip area ~ length x thickness: {} vs {}", group_area(g), total_len * 0.5);
        // The strip sits outside the circle (clockwise ring, normal outward) by the border pivot 0: half in, half out.
        let outermost = g.positions.iter().map(|q| Vec2::from(*q).length()).fold(0.0f32, f32::max);
        assert!(outermost > 3.2 && outermost < 3.4, "{outermost}");
    }

    #[test]
    fn counter_clockwise_input_gets_outward_edges_and_flip_goes_inward() {
        let (mut shape, _) = square_with_corners();
        shape.points.reverse(); // counter-clockwise
        let p = profile(&["edge"]);
        let outer = |s: &SpriteShapeRenderer| {
            let g = build(s, &p, None).groups.remove(0);
            g.positions.iter().map(|q| q[0].abs().max(q[1].abs())).fold(0.0f32, f32::max)
        };
        shape.corner_angle_threshold = 179.0;
        assert!((outer(&shape) - 2.25).abs() < 1e-4, "outward either way (miter reaches 2.25)");
        let mut cw = shape.clone();
        cw.points.reverse();
        assert!((outer(&cw) - 2.25).abs() < 1e-4);
        // flip_edge: the strip is centred on the line, so the extent is the same; use border_pivot to see the side.
        shape.border_pivot = 0.5;
        cw.border_pivot = 0.5;
        assert!((outer(&shape) - 2.5).abs() < 0.01 && (outer(&cw) - 2.5).abs() < 0.01);
        shape.flip_edge = true;
        assert!((outer(&shape) - 2.0).abs() < 0.01, "flipped puts the sprites inside: {}", outer(&shape));
    }

    #[test]
    fn point_cap_and_truncation_bound_the_mesh() {
        let mut shape = SpriteShapeRenderer { points: vec![ShapePoint::at(0.0, 0.0), ShapePoint::at(6.0, 4.0)], ..Default::default() };
        shape.points[0].set_mode(ShapeTangentMode::Broken, [0.0, 3.0], 3.0);
        shape.tessellation = Tessellation::Fixed;
        shape.spline_detail = 500;
        shape.max_points = 40;
        let m = build(&shape, &profile(&["edge"]), None);
        assert!(m.stats.points <= 40);
        assert!(m.vertex_count() < 40 * 2 * 8, "a bounded mesh: {}", m.vertex_count());
        // Overflowing a group's u16 indices drops geometry instead of wrapping: a zigzag of
        // thousands of sharp corners makes a run + a corner quad per segment.
        let zig: Vec<ShapePoint> = (0..17_000).map(|i| ShapePoint::at(i as f32 * 0.5, if i % 2 == 0 { 0.0 } else { 1.0 })).collect();
        let mut huge = SpriteShapeRenderer { points: zig, fill: false, ..Default::default() };
        huge.max_points = 20_000;
        let mut p = profile(&["edge"]);
        p.outer_corner = "corner_o".into();
        p.inner_corner = "corner_i".into();
        let m = build(&huge, &p, None);
        assert!(m.stats.truncated, "{:?}", m.stats);
        assert!(m.groups.iter().all(|g| g.positions.len() <= 65535 && g.indices.iter().all(|&i| (i as usize) < g.positions.len())));
    }

    #[test]
    fn collider_outline_matches_the_tessellated_spline() {
        let mut shape = SpriteShapeRenderer { points: vec![ShapePoint::at(-3.0, 0.0), ShapePoint::at(0.0, 2.0), ShapePoint::at(3.0, 0.0)], collider_optimize: 0.0, collider_detail: 12, ..Default::default() };
        shape.points[1].set_mode(ShapeTangentMode::Continuous, [1.0, 0.0], 1.5);
        let o = collider_outline(&shape);
        assert!(!o.closed);
        let poly = tessellate(&shape.points, false, TessMode::Fixed { detail: 12 }, shape.max_points);
        assert_eq!(o.points, poly.pts.iter().map(|p| p.to_array()).collect::<Vec<_>>(), "no optimisation: exactly the tessellation");
        // Optimising keeps the outline within the tolerance of the full one and drops points.
        shape.collider_optimize = 0.05;
        let simple = collider_outline(&shape);
        assert!(simple.points.len() < o.points.len());
        let sp: Vec<Vec2> = simple.points.iter().map(|p| Vec2::from(*p)).collect();
        for p in &poly.pts {
            assert!(distance_to_polyline(*p, &sp, false) <= 0.05 + 1e-5);
        }
        // The detail setting changes the density.
        shape.collider_optimize = 0.0;
        shape.collider_detail = 4;
        assert!(collider_outline(&shape).points.len() < o.points.len());
        // Closed shapes give a ring; an offset grows it.
        let mut sq = square(4.0, true);
        sq.collider_optimize = 0.0;
        let ring = collider_outline(&sq);
        assert!(ring.closed && ring.points.len() == 4);
        sq.collider_offset = 0.5;
        let grown = collider_outline(&sq);
        let area = |o: &ColliderOutline| signed_area(&o.points.iter().map(|p| Vec2::from(*p)).collect::<Vec<_>>()).abs();
        assert!((area(&grown) - 25.0).abs() < 1e-3);
        // Offsets never apply to open outlines.
        let mut open = square(4.0, false);
        open.collider_offset = 3.0;
        assert_eq!(collider_outline(&open).points.len(), 4);
        assert_eq!(collider_outline(&open).points[0], [-2.0, 2.0]);
    }

    #[test]
    fn empty_and_tiny_splines_produce_nothing() {
        let p = profile(&["edge"]);
        let tex = Some(FillTexture { texture: 9, size_px: [8, 8] });
        for pts in [vec![], vec![ShapePoint::at(0.0, 0.0)], vec![ShapePoint::at(1.0, 1.0), ShapePoint::at(1.0, 1.0)]] {
            for closed in [false, true] {
                let s = SpriteShapeRenderer { points: pts.clone(), closed, ..Default::default() };
                let m = build(&s, &p, tex);
                assert!(m.groups.is_empty() && m.bounds.is_none());
            }
        }
        let two = SpriteShapeRenderer { points: vec![ShapePoint::at(0.0, 0.0), ShapePoint::at(2.0, 0.0)], closed: true, ..Default::default() };
        let m = build(&two, &p, tex);
        assert!(group(&m, GroupKind::Fill, 9).is_none(), "two points enclose nothing");
    }
}
