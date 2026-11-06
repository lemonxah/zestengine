//! Per-mesh signed distance field generation for the software GI
//! path. A mesh is voxelized once into a local-space SDF grid: each voxel stores
//! the signed distance to the nearest triangle (negative inside). The software
//! GI marches these instead of a hardware BVH, so it needs no RT cores.
//!
//! Sign uses angle-weighted pseudo-normals (Baerentzen & Aanaes 2005): the
//! closest feature of the nearest triangle (face interior, edge or vertex)
//! selects the face normal, the edge pseudo-normal (sum of adjacent face
//! normals) or the angle-weighted vertex pseudo-normal. Plain face normals
//! mis-sign voxels whose closest point is a shared edge/vertex. Vertices are
//! welded by position first since meshes split them along UV seams.
//!
//! Nearest-triangle queries go through a small BVH, and z-slices are spread
//! over a capped pool of scoped threads.
//!
//! Realtime Flux uploads these fields into its shared mesh-distance atlas;
//! the coarse merged field remains the far-field tracing fallback.
#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Mutex;

use glam::Vec3;

/// A mesh's local-space signed distance field over a padded AABB. `data` is
/// `dims.x * dims.y * dims.z` distances, x fastest then y then z.
#[derive(Clone, Debug)]
pub struct SdfVolume {
    pub dims: [u32; 3],
    /// World/local-space bounds the grid spans (mesh AABB + padding).
    pub min: Vec3,
    pub max: Vec3,
    /// Unpadded geometry bounds for ray culling; the sampling volume above
    /// includes a margin that must not be treated as occupied geometry.
    pub geometry_bounds: [Vec3;2],
    pub data: Vec<f32>,
}

impl SdfVolume {
    /// Trilinearly sample the signed distance at a local-space point. Points
    /// outside the grid clamp to the border (distance grows roughly linearly
    /// outside via the clamp + the padding margin).
    pub fn sample(&self, p: Vec3) -> f32 {
        // The volume boundary is not geometry. Returning only the distance to
        // its box creates a zero-distance shell that falsely occludes light.
        // Extend the sampled boundary distance continuously outside the grid.
        let q = p.clamp(self.min, self.max);
        let outside = p.distance(q);
        let size = self.max - self.min;
        let inv = Vec3::new(
            (self.dims[0].max(1) - 1).max(1) as f32 / size.x.max(1e-6),
            (self.dims[1].max(1) - 1).max(1) as f32 / size.y.max(1e-6),
            (self.dims[2].max(1) - 1).max(1) as f32 / size.z.max(1e-6),
        );
        let g = (q - self.min) * inv;
        let (dx, dy, dz) = (self.dims[0] as i32, self.dims[1] as i32, self.dims[2] as i32);
        let clampi = |v: f32, n: i32| (v.floor() as i32).clamp(0, n - 1);
        let x0 = clampi(g.x, dx);
        let y0 = clampi(g.y, dy);
        let z0 = clampi(g.z, dz);
        let x1 = (x0 + 1).min(dx - 1);
        let y1 = (y0 + 1).min(dy - 1);
        let z1 = (z0 + 1).min(dz - 1);
        let fx = (g.x - x0 as f32).clamp(0.0, 1.0);
        let fy = (g.y - y0 as f32).clamp(0.0, 1.0);
        let fz = (g.z - z0 as f32).clamp(0.0, 1.0);
        let at = |x: i32, y: i32, z: i32| {
            self.data[(z * dy * dx + y * dx + x) as usize]
        };
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let c00 = lerp(at(x0, y0, z0), at(x1, y0, z0), fx);
        let c10 = lerp(at(x0, y1, z0), at(x1, y1, z0), fx);
        let c01 = lerp(at(x0, y0, z1), at(x1, y0, z1), fx);
        let c11 = lerp(at(x0, y1, z1), at(x1, y1, z1), fx);
        let boundary = lerp(lerp(c00, c10, fy), lerp(c01, c11, fy), fz);
        if outside > 0.0 { outside.hypot(boundary.max(0.0)) } else { boundary }
    }
}

/// Which part of a triangle the closest point lies on. Edges: 0 = ab,
/// 1 = bc, 2 = ca. Vertices: 0 = a, 1 = b, 2 = c.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Feature {
    Face,
    Edge(u8),
    Vertex(u8),
}

/// Closest point on triangle (a,b,c) to p and the feature it lies on
/// (Ericson, Real-Time Collision Detection).
fn closest_on_tri(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Vec3, Feature) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, Feature::Vertex(0));
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, Feature::Vertex(1));
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, Feature::Edge(0));
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, Feature::Vertex(2));
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, Feature::Edge(2));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w, Feature::Edge(1));
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (a + ab * v + ac * w, Feature::Face)
}

/// A triangle with its face normal and the pseudo-normals of its three edges
/// (ab, bc, ca) and vertices (a, b, c). Pseudo-normals are unnormalized; only
/// their direction matters for the sign test.
#[derive(Clone, Copy)]
struct Tri {
    v: [Vec3; 3],
    n: Vec3,
    edge_n: [Vec3; 3],
    vert_n: [Vec3; 3],
}

impl Tri {
    fn pseudo_normal(&self, f: Feature) -> Vec3 {
        match f {
            Feature::Face => self.n,
            Feature::Edge(i) => self.edge_n[i as usize],
            Feature::Vertex(i) => self.vert_n[i as usize],
        }
    }
}

/// Weld vertices by position and build triangles carrying edge / vertex
/// pseudo-normals. Degenerate triangles are dropped (their edges coincide with
/// neighbours, which carry the geometry).
fn build_tris(positions: &[Vec3], indices: &[u32], weld_eps: f32) -> Vec<Tri> {
    let inv = 1.0 / weld_eps;
    let mut weld: HashMap<[i64; 3], u32> = HashMap::with_capacity(positions.len());
    let welded: Vec<u32> = positions
        .iter()
        .map(|p| {
            let k = (*p * inv).round();
            let key = [k.x as i64, k.y as i64, k.z as i64];
            let next = weld.len() as u32;
            *weld.entry(key).or_insert(next)
        })
        .collect();

    struct Face {
        v: [Vec3; 3],
        w: [u32; 3],
        n: Vec3,
    }
    let faces: Vec<Face> = indices
        .as_chunks::<3>().0.iter()
        .filter_map(|t| {
            let i = [t[0] as usize, t[1] as usize, t[2] as usize];
            if i.iter().any(|&k| k >= positions.len()) {
                return None;
            }
            let w = [welded[i[0]], welded[i[1]], welded[i[2]]];
            if w[0] == w[1] || w[1] == w[2] || w[0] == w[2] {
                return None;
            }
            let v = [positions[i[0]], positions[i[1]], positions[i[2]]];
            let n = (v[1] - v[0]).cross(v[2] - v[0]).normalize_or_zero();
            if n == Vec3::ZERO {
                return None;
            }
            Some(Face { v, w, n })
        })
        .collect();

    let edge_key = |a: u32, b: u32| if a < b { (a, b) } else { (b, a) };
    let mut vert_n = vec![Vec3::ZERO; weld.len()];
    let mut edge_n: HashMap<(u32, u32), Vec3> = HashMap::with_capacity(faces.len() * 2);
    for f in &faces {
        for k in 0..3 {
            let (p, q, r) = (f.v[k], f.v[(k + 1) % 3], f.v[(k + 2) % 3]);
            let angle = (q - p)
                .normalize_or_zero()
                .dot((r - p).normalize_or_zero())
                .clamp(-1.0, 1.0)
                .acos();
            vert_n[f.w[k] as usize] += f.n * angle;
            *edge_n.entry(edge_key(f.w[k], f.w[(k + 1) % 3])).or_insert(Vec3::ZERO) += f.n;
        }
    }

    faces
        .iter()
        .map(|f| {
            let e = |k: usize| edge_n[&edge_key(f.w[k], f.w[(k + 1) % 3])];
            Tri {
                v: f.v,
                n: f.n,
                edge_n: [e(0), e(1), e(2)],
                vert_n: [
                    vert_n[f.w[0] as usize],
                    vert_n[f.w[1] as usize],
                    vert_n[f.w[2] as usize],
                ],
            }
        })
        .collect()
}

/// BVH node. Leaves have `count > 0` and cover `tris[start..start + count]`;
/// interior nodes have their left child at `self + 1` and right at `start`.
struct Node {
    min: Vec3,
    max: Vec3,
    start: u32,
    count: u32,
}

const LEAF_SIZE: usize = 4;

/// Nearest-triangle BVH (median split on the longest centroid axis).
struct Bvh {
    nodes: Vec<Node>,
    tris: Vec<Tri>,
}

impl Bvh {
    fn build(tris: Vec<Tri>) -> Self {
        let mut order: Vec<u32> = (0..tris.len() as u32).collect();
        let centroids: Vec<Vec3> = tris.iter().map(|t| (t.v[0] + t.v[1] + t.v[2]) / 3.0).collect();
        let mut nodes = Vec::with_capacity((tris.len() / LEAF_SIZE).max(1) * 2);
        if !tris.is_empty() {
            Self::build_node(&mut nodes, &tris, &centroids, &mut order, 0);
        }
        let tris = order.iter().map(|&i| tris[i as usize]).collect();
        Bvh { nodes, tris }
    }

    fn build_node(
        nodes: &mut Vec<Node>,
        tris: &[Tri],
        centroids: &[Vec3],
        order: &mut [u32],
        offset: u32,
    ) -> usize {
        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        let (mut clo, mut chi) = (lo, hi);
        for &i in order.iter() {
            for v in tris[i as usize].v {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            let c = centroids[i as usize];
            clo = clo.min(c);
            chi = chi.max(c);
        }
        let me = nodes.len();
        nodes.push(Node { min: lo, max: hi, start: offset, count: order.len() as u32 });
        if order.len() <= LEAF_SIZE {
            return me;
        }
        let ext = chi - clo;
        let axis = if ext.x >= ext.y && ext.x >= ext.z {
            0
        } else if ext.y >= ext.z {
            1
        } else {
            2
        };
        let mid = order.len() / 2;
        order.select_nth_unstable_by(mid, |&a, &b| {
            centroids[a as usize][axis].total_cmp(&centroids[b as usize][axis])
        });
        let (left, right) = order.split_at_mut(mid);
        Self::build_node(nodes, tris, centroids, left, offset);
        let r = Self::build_node(nodes, tris, centroids, right, offset + mid as u32);
        nodes[me].count = 0;
        nodes[me].start = r as u32;
        me
    }

    /// Nearest triangle to `p`. `hint` (a triangle index, e.g. the previous
    /// voxel's answer) seeds the search bound so most nodes are pruned early.
    /// Returns (squared distance, triangle index, closest point, feature).
    fn nearest(&self, p: Vec3, hint: usize) -> (f32, usize, Vec3, Feature) {
        let t = &self.tris[hint];
        let (cp, f) = closest_on_tri(p, t.v[0], t.v[1], t.v[2]);
        let mut best = ((p - cp).length_squared(), hint, cp, f);
        let box_d2 = |n: &Node| (n.min - p).max(p - n.max).max(Vec3::ZERO).length_squared();
        let mut stack = [0u32; 64];
        let mut sp = 1usize;
        while sp > 0 {
            sp -= 1;
            let node = &self.nodes[stack[sp] as usize];
            if box_d2(node) >= best.0 {
                continue;
            }
            if node.count > 0 {
                let s = node.start as usize;
                for (k, t) in self.tris[s..s + node.count as usize].iter().enumerate() {
                    let (cp, f) = closest_on_tri(p, t.v[0], t.v[1], t.v[2]);
                    let d2 = (p - cp).length_squared();
                    if d2 < best.0 {
                        best = (d2, s + k, cp, f);
                    }
                }
                continue;
            }
            let l = stack[sp] as usize + 1;
            let r = node.start as usize;
            let (dl, dr) = (box_d2(&self.nodes[l]), box_d2(&self.nodes[r]));
            let (near, far, dfar) = if dl <= dr { (l, r, dr) } else { (r, l, dl) };
            if dfar < best.0 {
                stack[sp] = far as u32;
                sp += 1;
            }
            stack[sp] = near as u32;
            sp += 1;
        }
        best
    }
}

/// Worker threads for SDF generation: 80% of the cores, at least one, leaving
/// one core free for the UI/OS (mirrors `zestengine_engine::sw_gi::bake_worker_count`).
fn sdf_worker_count() -> usize {
    let cores = std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(4)
        .max(1);
    ((cores * 8) / 10).clamp(1, cores.saturating_sub(1).max(1))
}

/// Voxelize `positions`/`indices` into a signed distance field. `res` is the
/// voxel count along the longest axis (voxels are kept ~cubic); `pad` is extra
/// world-units of margin around the AABB so the surface isn't at the very edge.
pub fn generate_sdf(positions: &[Vec3], indices: &[u32], res: u32, pad: f32) -> SdfVolume {
    let res = res.max(2);
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for p in positions {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    if !lo.is_finite() {
        return SdfVolume {
            dims: [2, 2, 2],
            min: Vec3::splat(-1.0),
            max: Vec3::splat(1.0),
            geometry_bounds: [Vec3::ZERO;2],
            data: vec![1.0; 8],
        };
    }
    let diag = (hi - lo).length();
    let margin = pad.max(diag * 0.05 + 1e-3);
    let (min, max) = (lo - margin, hi + margin);
    let extent = (max - min).max(Vec3::splat(1e-3));
    let voxel = extent.max_element() / res as f32;
    let dims = [
        ((extent.x / voxel).ceil() as u32).max(2),
        ((extent.y / voxel).ceil() as u32).max(2),
        ((extent.z / voxel).ceil() as u32).max(2),
    ];
    let slice = (dims[0] * dims[1]) as usize;
    let mut data = vec![0.0f32; slice * dims[2] as usize];

    let weld_eps = (diag * 1e-6).max(1e-7);
    let bvh = Bvh::build(build_tris(positions, indices, weld_eps));
    if bvh.tris.is_empty() {
        data.fill(extent.length());
        return SdfVolume { dims, min, max, geometry_bounds:[lo,hi], data };
    }

    let step = Vec3::new(
        extent.x / (dims[0] - 1) as f32,
        extent.y / (dims[1] - 1) as f32,
        extent.z / (dims[2] - 1) as f32,
    );
    let fill_slice = |z: usize, out: &mut [f32]| {
        let mut hint = 0usize;
        for y in 0..dims[1] as usize {
            for x in 0..dims[0] as usize {
                let p = min + Vec3::new(x as f32, y as f32, z as f32) * step;
                let (d2, ti, cp, f) = bvh.nearest(p, hint);
                hint = ti;
                let pn = bvh.tris[ti].pseudo_normal(f);
                let sign = if (p - cp).dot(pn) < 0.0 { -1.0 } else { 1.0 };
                out[y * dims[0] as usize + x] = d2.sqrt() * sign;
            }
        }
    };

    let workers = sdf_worker_count().min(dims[2] as usize);
    if workers <= 1 {
        for (z, out) in data.chunks_mut(slice).enumerate() {
            fill_slice(z, out);
        }
    } else {
        let work = Mutex::new(data.chunks_mut(slice).enumerate());
        std::thread::scope(|s| {
            for _ in 0..workers {
                s.spawn(|| loop {
                    let next = work.lock().unwrap_or_else(|e| e.into_inner()).next();
                    let Some((z, out)) = next else { break };
                    fill_slice(z, out);
                });
            }
        });
    }
    SdfVolume { dims, min, max, geometry_bounds:[lo,hi], data }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A unit cube (centered) voxelized: center is inside (negative ~ -0.5),
    // a point well outside is positive.
    fn unit_cube() -> (Vec<Vec3>, Vec<u32>) {
        let h = 0.5;
        let v = [
            [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
            [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
        ]
        .map(Vec3::from_array)
        .to_vec();
        // 12 triangles, outward winding.
        let idx = vec![
            0, 2, 1, 0, 3, 2, // -z
            4, 5, 6, 4, 6, 7, // +z
            0, 1, 5, 0, 5, 4, // -y
            3, 7, 6, 3, 6, 2, // +y
            0, 4, 7, 0, 7, 3, // -x
            1, 2, 6, 1, 6, 5, // +x
        ];
        (v, idx)
    }

    fn grid_point(sdf: &SdfVolume, x: u32, y: u32, z: u32) -> Vec3 {
        let size = sdf.max - sdf.min;
        let step = Vec3::new(
            size.x / (sdf.dims[0] - 1) as f32,
            size.y / (sdf.dims[1] - 1) as f32,
            size.z / (sdf.dims[2] - 1) as f32,
        );
        sdf.min + Vec3::new(x as f32, y as f32, z as f32) * step
    }

    /// Visit every grid point as (point, stored distance).
    fn for_each_voxel(sdf: &SdfVolume, mut f: impl FnMut(Vec3, f32)) {
        for z in 0..sdf.dims[2] {
            for y in 0..sdf.dims[1] {
                for x in 0..sdf.dims[0] {
                    let i = (z * sdf.dims[1] * sdf.dims[0] + y * sdf.dims[0] + x) as usize;
                    f(grid_point(sdf, x, y, z), sdf.data[i]);
                }
            }
        }
    }

    #[test]
    fn sdf_sign_inside_outside() {
        let (pos, idx) = unit_cube();
        let sdf = generate_sdf(&pos, &idx, 24, 0.25);
        // Center is inside → negative, ~ -0.5 to a face.
        let c = sdf.sample(Vec3::ZERO);
        assert!(c < 0.0, "center should be inside (negative), got {c}");
        assert!((c + 0.5).abs() < 0.15, "center distance ~ -0.5, got {c}");
        // A point a unit outside along +x → positive ~0.5.
        let o = sdf.sample(Vec3::new(1.0, 0.0, 0.0));
        assert!(o > 0.0, "outside should be positive, got {o}");
        // Every grid point inside the closed cube is negative, outside positive.
        for_each_voxel(&sdf, |p, d| {
            if d.abs() < 1e-4 {
                return;
            }
            let inside = p.abs().max_element() < 0.5;
            assert_eq!(d < 0.0, inside, "cube voxel {p} mis-signed: {d}");
        });
    }

    #[test]
    fn padded_volume_boundary_is_not_an_occluder() {
        let (pos, idx) = unit_cube();
        let sdf = generate_sdf(&pos, &idx, 32, 0.25);
        let border = Vec3::new(sdf.max.x, 0.0, 0.0);
        let inside = sdf.sample(border - Vec3::X * 0.001);
        let outside = sdf.sample(border + Vec3::X * 0.001);
        assert!(outside > 0.2, "padding became a false surface: {outside}");
        assert!((inside - outside).abs() < 0.01, "discontinuous field at volume boundary");
        // A ray passing through the padded box above the actual cube is clear.
        let mut t = 0.0;
        for _ in 0..96 {
            if t >= 2.0 { break; }
            let d = sdf.sample(Vec3::new(-1.0 + t, 0.65, 0.0));
            assert!(d > 0.01, "clear ray hit the padded volume at t={t}: {d}");
            t += d;
        }
        assert!(t >= 2.0, "clear ray failed to escape");
    }

    /// Five-point star (sharp convex tips, reflex inner corners = concave
    /// creases) extruded along z. Every face gets its own vertices, like a
    /// mesh split along UV seams, so the pseudo-normal tables must weld.
    fn star_prism(h: f32) -> (Vec<[f32; 2]>, Vec<Vec3>, Vec<u32>) {
        let poly: Vec<[f32; 2]> = (0..10)
            .map(|i| {
                let r = if i % 2 == 0 { 1.0 } else { 0.38 };
                let a = std::f32::consts::FRAC_PI_2 + i as f32 * std::f32::consts::PI / 5.0;
                [r * a.cos(), r * a.sin()]
            })
            .collect();
        let mut pos = Vec::new();
        let mut idx = Vec::new();
        let mut tri = |a: Vec3, b: Vec3, c: Vec3| {
            let base = pos.len() as u32;
            pos.extend([a, b, c]);
            idx.extend([base, base + 1, base + 2]);
        };
        let n = poly.len();
        for i in 0..n {
            let (p, q) = (poly[i], poly[(i + 1) % n]);
            let v = |pt: [f32; 2], z: f32| Vec3::new(pt[0], pt[1], z);
            tri(Vec3::new(0.0, 0.0, h), v(p, h), v(q, h));
            tri(Vec3::ZERO, v(q, 0.0), v(p, 0.0));
            tri(v(p, 0.0), v(q, 0.0), v(q, h));
            tri(v(p, 0.0), v(q, h), v(p, h));
        }
        (poly, pos, idx)
    }

    fn point_in_poly(poly: &[[f32; 2]], x: f32, y: f32) -> bool {
        let mut inside = false;
        let mut j = poly.len() - 1;
        for i in 0..poly.len() {
            let (a, b) = (poly[i], poly[j]);
            if (a[1] > y) != (b[1] > y) && x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0] {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// Reference: brute-force nearest triangle with the old face-normal sign.
    fn brute_face_sign(pos: &[Vec3], idx: &[u32], p: Vec3) -> f32 {
        let mut best = f32::INFINITY;
        let mut sign = 1.0;
        for t in idx.as_chunks::<3>().0 {
            let (a, b, c) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
            let n = (b - a).cross(c - a).normalize_or_zero();
            let (cp, _) = closest_on_tri(p, a, b, c);
            let d2 = (p - cp).length_squared();
            if d2 < best {
                best = d2;
                sign = if (p - cp).dot(n) < 0.0 { -1.0 } else { 1.0 };
            }
        }
        best.sqrt() * sign
    }

    #[test]
    fn sdf_concave_star_signs_and_distances() {
        let h = 0.4;
        let (poly, pos, idx) = star_prism(h);
        let sdf = generate_sdf(&pos, &idx, 40, 0.3);
        let mut checked = 0;
        let mut old_wrong = 0;
        for_each_voxel(&sdf, |p, d| {
            let reference = brute_face_sign(&pos, &idx, p);
            assert!(
                (d.abs() - reference.abs()).abs() < 1e-4,
                "BVH distance {d} != brute force {reference} at {p}"
            );
            if d.abs() < 1e-4 {
                return;
            }
            let inside = point_in_poly(&poly, p.x, p.y) && p.z > 0.0 && p.z < h;
            assert_eq!(d < 0.0, inside, "star voxel {p} mis-signed: {d}");
            if (reference < 0.0) != inside {
                old_wrong += 1;
            }
            checked += 1;
        });
        assert!(checked > 1000);
        // The geometry must actually exercise the shared-edge/vertex case the
        // face-normal sign gets wrong, or this test proves nothing.
        assert!(old_wrong > 0, "face-normal sign had no errors; test geometry too easy");
    }

    #[test]
    fn sdf_open_space_at_concave_crease_positive() {
        // L-shaped closed prism: the reflex edge at (1, 1, z) is a concave
        // crease; the open quadrant x > 1, y > 1 next to it must be positive.
        let (poly, pos, idx) = {
            let poly = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0f32]];
            let mut pos = Vec::new();
            let mut idx = Vec::new();
            let v = |pt: [f32; 2], z: f32| Vec3::new(pt[0], pt[1], z);
            let mut tri = |a: Vec3, b: Vec3, c: Vec3| {
                let base = pos.len() as u32;
                pos.extend([a, b, c]);
                idx.extend([base, base + 1, base + 2]);
            };
            for k in 0..4 {
                let (a, b, c) = (poly[0], poly[k + 1], poly[k + 2]);
                tri(v(a, 1.0), v(b, 1.0), v(c, 1.0));
                tri(v(a, 0.0), v(c, 0.0), v(b, 0.0));
            }
            for i in 0..poly.len() {
                let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
                tri(v(p, 0.0), v(q, 0.0), v(q, 1.0));
                tri(v(p, 0.0), v(q, 1.0), v(p, 1.0));
            }
            (poly, pos, idx)
        };
        let sdf = generate_sdf(&pos, &idx, 32, 0.2);
        let mut near_crease = 0;
        for_each_voxel(&sdf, |p, d| {
            if d.abs() < 1e-4 {
                return;
            }
            let inside = point_in_poly(&poly, p.x, p.y) && p.z > 0.0 && p.z < 1.0;
            assert_eq!(d < 0.0, inside, "L voxel {p} mis-signed: {d}");
            if p.x > 1.0 && p.y > 1.0 && (p - Vec3::new(1.0, 1.0, p.z)).length() < 0.3 {
                assert!(d > 0.0, "open space near crease must be positive at {p}: {d}");
                near_crease += 1;
            }
        });
        assert!(near_crease > 0);
    }
}
