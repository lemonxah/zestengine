//! Lightmap UV generation for imported meshes that ship without a second UV set.
//!
//! Many glTF/FBX models only carry TEXCOORD_0 (their texture UVs), which usually
//! overlap (parts share texture space). Baking a lightmap into overlapping UVs
//! produces garbage. When the user opts in (the bake offers it), we generate a
//! **non-overlapping** lightmap UV into `uv1`.
//!
//! Approach: a robust **per-triangle atlas**. Every triangle is given its own
//! cell in a square grid, so charts can never overlap. Each triangle's vertices
//! are split (duplicated) so a vertex shared by two triangles can hold a
//! different lightmap UV per triangle. Quality is modest (a seam at every edge,
//! some texel waste) but it is *correct and deterministic* — the same mesh always
//! yields the same UVs, so a small marker file is enough to reproduce it on load
//! (no need to persist the whole rebuilt mesh). The bake's seam-stitch + dilate
//! passes hide most of the per-edge seams.

use zestengine_render::{MeshData, Vertex};
use glam::Vec3;

/// Whether a mesh's `uv0` is usable AS-IS for lightmapping — i.e. it's a
/// non-overlapping chart layout inside roughly the [0,1] square. Most models'
/// texture UVs already are (a single atlas), so they bake fine without a second
/// UV set and must NOT be regenerated. Heuristic: a non-overlapping unwrap inside
/// the unit square has a total triangle-UV-area sum ≤ 1 (you can't pack more area
/// than the square holds without overlapping); tiled/overlapping UVs (texture
/// reuse, mirrored islands) sum to well over 1 and/or spill outside [0,1].
pub fn uv0_is_lightmappable(vertices: &[Vertex], indices: &[u32]) -> bool {
    if indices.len() < 3 {
        return false;
    }
    let mut area_sum = 0.0f32;
    let mut lo = [f32::INFINITY; 2];
    let mut hi = [f32::NEG_INFINITY; 2];
    for tri in indices.as_chunks::<3>().0 {
        let a = vertices[tri[0] as usize].uv;
        let b = vertices[tri[1] as usize].uv;
        let c = vertices[tri[2] as usize].uv;
        area_sum += 0.5
            * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])).abs();
        for uv in [a, b, c] {
            lo[0] = lo[0].min(uv[0]);
            lo[1] = lo[1].min(uv[1]);
            hi[0] = hi[0].max(uv[0]);
            hi[1] = hi[1].max(uv[1]);
        }
    }
    // Inside (a small margin around) [0,1] AND total area ≤ 1 ⇒ non-overlapping.
    let in_unit = lo[0] > -0.02 && lo[1] > -0.02 && hi[0] < 1.02 && hi[1] < 1.02;
    in_unit && area_sum <= 1.02
}

/// A chart to pack: a quad (2 triangles, 4 shared verts → seamless interior) or a
/// lone triangle. Projected into 2D at **world scale** (metres).
struct Island {
    /// Unique vertices of this chart (the quad shares its diagonal verts).
    verts: Vec<Vertex>,
    /// Per-vertex 2D coord (metres), relative to this chart's bbox min.
    local: Vec<[f32; 2]>,
    /// Triangles as local indices into `verts` (1 for a tri chart, 2 for a quad).
    tris: Vec<[u32; 3]>,
    w: f32,
    h: f32,
}

/// World-space normal of a triangle (object-space positions; only used for
/// grouping + projection, so the object→world scale is irrelevant).
fn tri_normal(mesh: &MeshData, t: [u32; 3]) -> Vec3 {
    let p0 = Vec3::from(mesh.vertices[t[0] as usize].position);
    let p1 = Vec3::from(mesh.vertices[t[1] as usize].position);
    let p2 = Vec3::from(mesh.vertices[t[2] as usize].position);
    (p1 - p0).cross(p2 - p0).normalize_or(Vec3::Z)
}

/// Project a chart's vertices onto the plane of `normal` at world scale, returning
/// (local 2D coords relative to bbox-min, width, height).
fn project(mesh: &MeshData, orig: &[u32], normal: Vec3) -> (Vec<[f32; 2]>, f32, f32) {
    let p: Vec<Vec3> = orig.iter().map(|&i| Vec3::from(mesh.vertices[i as usize].position)).collect();
    // In-plane U from the first edge (normal component removed); V = N × U.
    let e = if p.len() > 1 { p[1] - p[0] } else { Vec3::X };
    let u_axis = (e - normal * e.dot(normal)).normalize_or(Vec3::X);
    let v_axis = normal.cross(u_axis).normalize_or(Vec3::Y);
    let uv: Vec<[f32; 2]> = p.iter().map(|q| [(*q - p[0]).dot(u_axis), (*q - p[0]).dot(v_axis)]).collect();
    let min = [
        uv.iter().map(|c| c[0]).fold(f32::INFINITY, f32::min),
        uv.iter().map(|c| c[1]).fold(f32::INFINITY, f32::min),
    ];
    let max = [
        uv.iter().map(|c| c[0]).fold(f32::NEG_INFINITY, f32::max),
        uv.iter().map(|c| c[1]).fold(f32::NEG_INFINITY, f32::max),
    ];
    let local = uv.iter().map(|c| [c[0] - min[0], c[1] - min[1]]).collect();
    ((local), (max[0] - min[0]).max(1e-6), (max[1] - min[1]).max(1e-6))
}

/// Generate a **uniform-density** non-overlapping lightmap UV set (`uv1`) for
/// `mesh`, returning a new mesh whose topology is split along chart boundaries.
/// All other vertex attributes are preserved per corner. Deterministic.
///
/// Modelled on Unity's "Generate Lightmap UVs": the mesh is segmented into
/// **charts** along hard edges (a flood-fill that stops where adjacent-triangle
/// normals diverge past a hard-angle threshold), so a smooth region is one chart
/// with no internal seams. Each chart is planar-projected at **true world scale**,
/// the charts are shelf-packed with a gutter (never overlapping), then a single
/// uniform scale fits the atlas into [0,1] — so texels-per-metre is constant
/// across the surface. (We don't run Unity's conformal/LSCM solve, so charts stay
/// near-planar; curved surfaces split into a few charts instead of one.)
pub fn generate_lightmap_uv(mesh: &MeshData) -> MeshData {
    let tri_count = mesh.indices.len() / 3;
    if tri_count == 0 || mesh.vertices.is_empty() {
        return MeshData {
            vertices: mesh.vertices.clone(),
            indices: mesh.indices.clone(),
            has_lightmap_uv: true,
        };
    }

    let tris: Vec<[u32; 3]> = mesh.indices.as_chunks::<3>().0.iter().map(|c| [c[0], c[1], c[2]]).collect();
    let normals: Vec<Vec3> = tris.iter().map(|&t| tri_normal(mesh, t)).collect();

    // 1. Edge → triangles sharing it (manifold edges have 2). Edge key = the two
    //    shared vertex indices, sorted.
    use std::collections::HashMap;
    let edge_key = |a: u32, b: u32| if a < b { (a, b) } else { (b, a) };
    let mut edges: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (ti, t) in tris.iter().enumerate() {
        for &(a, b) in &[(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            edges.entry(edge_key(a, b)).or_default().push(ti);
        }
    }

    // 2. Segment into CHARTS by flood-filling across smooth edges, like Unity's
    //    "Generate Lightmap UVs" (its Hard Angle = 88° default). An edge between two
    //    triangles is a SEAM (chart boundary) when their normals diverge past the
    //    hard-angle threshold; below it, they're one chart with no internal seam.
    //    Each chart is then planar-projected. Unlike Unity we don't run a conformal
    //    (LSCM) solve, so to keep a planar projection injective (no self-overlap) we
    //    grow a chart only while every triangle stays within HARD_ANGLE of the SEED
    //    normal — flat regions (panels, cushions, walls) become one large seamless
    //    chart; curved regions split into a few near-planar charts.
    const HARD_ANGLE_COS: f32 = 0.80; // ≈ 37° from the seed (×2 ≈ 74° total span)
    let mut chart_of = vec![usize::MAX; tri_count];
    let mut islands: Vec<Island> = Vec::new();
    for seed in 0..tri_count {
        if chart_of[seed] != usize::MAX {
            continue;
        }
        let cid = islands.len();
        let seed_n = normals[seed];
        chart_of[seed] = cid;
        let mut members = vec![seed];
        let mut stack = vec![seed];
        while let Some(t) = stack.pop() {
            let tri = tris[t];
            for &(a, b) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                for &nb in &edges[&edge_key(a, b)] {
                    if chart_of[nb] == usize::MAX && normals[nb].dot(seed_n) > HARD_ANGLE_COS {
                        chart_of[nb] = cid;
                        members.push(nb);
                        stack.push(nb);
                    }
                }
            }
        }
        // Unique chart vertices (shared edges keep ONE vertex → seamless interior),
        // with a vert→local-slot map so chart building stays O(verts), not O(verts²).
        let mut slot_of: HashMap<u32, u32> = HashMap::new();
        let mut orig: Vec<u32> = Vec::new();
        let mut local_tris: Vec<[u32; 3]> = Vec::with_capacity(members.len());
        let mut avg_n = Vec3::ZERO;
        for &m in &members {
            avg_n += normals[m];
            let lt = tris[m].map(|v| *slot_of.entry(v).or_insert_with(|| {
                orig.push(v);
                orig.len() as u32 - 1
            }));
            local_tris.push(lt);
        }
        let (local, w, h) = project(mesh, &orig, avg_n.normalize_or(seed_n));
        islands.push(Island {
            verts: orig.iter().map(|&i| mesh.vertices[i as usize]).collect(),
            local,
            tris: local_tris,
            w,
            h,
        });
    }

    // 3. Shelf-pack the charts at world scale (next-fit-decreasing-height), gutter
    //    ≈ 1% of the atlas so dilation/bilinear can't bleed between charts.
    let total_area: f32 = islands.iter().map(|c| c.w * c.h).sum();
    let atlas_w = (total_area.sqrt() * 1.4).max(1e-6);
    let gutter = atlas_w * 0.01;
    let mut order: Vec<usize> = (0..islands.len()).collect();
    order.sort_by(|&a, &b| islands[b].h.partial_cmp(&islands[a].h).unwrap_or(core::cmp::Ordering::Equal));
    let mut placed = vec![[0.0f32; 2]; islands.len()];
    let (mut cx, mut cy, mut row_h, mut max_x, mut max_y) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for &i in &order {
        let (w, h) = (islands[i].w, islands[i].h);
        if cx > 0.0 && cx + w > atlas_w {
            cx = 0.0;
            cy += row_h + gutter;
            row_h = 0.0;
        }
        placed[i] = [cx, cy];
        cx += w + gutter;
        row_h = row_h.max(h);
        max_x = max_x.max(cx);
        max_y = max_y.max(cy + h);
    }

    // 4. One uniform scale fits the atlas into [inset, 1-inset] — constant density.
    let inset = 0.01f32;
    let extent = max_x.max(max_y).max(1e-6);
    let scale = (1.0 - 2.0 * inset) / extent;

    // 5. Emit. Each chart's verts are appended once (quad diagonal shared → seamless).
    let mut vertices: Vec<Vertex> = Vec::with_capacity(tri_count * 3);
    let mut indices: Vec<u32> = Vec::with_capacity(tri_count * 3);
    for (i, c) in islands.iter().enumerate() {
        let base = vertices.len() as u32;
        for (k, mut v) in c.verts.iter().copied().enumerate() {
            let ax = placed[i][0] + c.local[k][0];
            let ay = placed[i][1] + c.local[k][1];
            v.uv1 = [
                (inset + ax * scale).clamp(0.0, 1.0),
                (inset + ay * scale).clamp(0.0, 1.0),
            ];
            vertices.push(v);
        }
        for lt in &c.tris {
            indices.extend_from_slice(&[base + lt[0], base + lt[1], base + lt[2]]);
        }
    }

    MeshData {
        vertices,
        indices,
        has_lightmap_uv: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri_mesh(n: usize) -> MeshData {
        // n separate triangles in a row.
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for i in 0..n {
            let x = i as f32;
            for pos in [[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0], [x, 1.0, 0.0]] {
                indices.push(vertices.len() as u32);
                vertices.push(Vertex {
                    position: pos,
                    normal: [0.0, 0.0, 1.0],
                    uv: [0.0, 0.0],
                    color: [1.0; 4],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                    uv1: [0.0, 0.0],
                    joints: [0; 4],
                    weights: [0.0; 4],
                });
            }
        }
        MeshData { vertices, indices, has_lightmap_uv: false }
    }

    #[test]
    fn splits_topology_and_sets_flag() {
        let out = generate_lightmap_uv(&tri_mesh(5));
        assert_eq!(out.vertices.len(), 15); // 3 unique verts per triangle
        assert_eq!(out.indices.len(), 15);
        assert!(out.has_lightmap_uv);
    }

    #[test]
    fn all_uvs_in_unit_range() {
        let out = generate_lightmap_uv(&tri_mesh(17));
        for v in &out.vertices {
            assert!(v.uv1[0] >= 0.0 && v.uv1[0] <= 1.0, "u out of range: {:?}", v.uv1);
            assert!(v.uv1[1] >= 0.0 && v.uv1[1] <= 1.0, "v out of range: {:?}", v.uv1);
        }
    }

    /// A mesh of `n` separate triangles whose world size varies (triangle i is
    /// scaled by `1 + i`), to test that density stays uniform regardless of size.
    fn varied_tri_mesh(n: usize) -> MeshData {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for i in 0..n {
            let s = (1 + i) as f32;
            let x = (i * 10) as f32;
            for pos in [[x, 0.0, 0.0], [x + s, 0.0, 0.0], [x, s, 0.0]] {
                indices.push(vertices.len() as u32);
                vertices.push(Vertex {
                    position: pos,
                    normal: [0.0, 0.0, 1.0],
                    uv: [0.0, 0.0],
                    color: [1.0; 4],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                    uv1: [0.0, 0.0],
                    joints: [0; 4],
                    weights: [0.0; 4],
                });
            }
        }
        MeshData { vertices, indices, has_lightmap_uv: false }
    }

    fn tri_area(uv: [[f32; 2]; 3]) -> f32 {
        0.5 * ((uv[1][0] - uv[0][0]) * (uv[2][1] - uv[0][1])
            - (uv[2][0] - uv[0][0]) * (uv[1][1] - uv[0][1]))
            .abs()
    }

    fn mk_vert(pos: [f32; 3]) -> Vertex {
        Vertex {
            position: pos,
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
            color: [1.0; 4],
            tangent: [1.0, 0.0, 0.0, 1.0],
            uv1: [0.0, 0.0],
            joints: [0; 4],
            weights: [0.0; 4],
        }
    }

    #[test]
    fn coplanar_pair_forms_one_quad_chart() {
        // Two triangles sharing the diagonal 0-2, both facing +Z → one quad chart
        // of 4 shared verts (not 6 split verts), so the diagonal has no seam.
        let verts = vec![
            mk_vert([0.0, 0.0, 0.0]),
            mk_vert([1.0, 0.0, 0.0]),
            mk_vert([1.0, 1.0, 0.0]),
            mk_vert([0.0, 1.0, 0.0]),
        ];
        let mesh = MeshData { vertices: verts, indices: vec![0, 1, 2, 0, 2, 3], has_lightmap_uv: false };
        let out = generate_lightmap_uv(&mesh);
        assert_eq!(out.vertices.len(), 4, "coplanar quad should be one 4-vert chart");
        assert_eq!(out.indices.len(), 6);
        assert!(out.has_lightmap_uv);
    }

    #[test]
    fn texel_density_is_uniform() {
        // The whole point: uv-area / world-area must be ~constant across triangles
        // of very different sizes (it was wildly uneven in the per-cell version).
        let out = generate_lightmap_uv(&varied_tri_mesh(8));
        let mut ratios = Vec::new();
        for tri in out.indices.as_chunks::<3>().0 {
            let uv = [
                out.vertices[tri[0] as usize].uv1,
                out.vertices[tri[1] as usize].uv1,
                out.vertices[tri[2] as usize].uv1,
            ];
            let wp = [
                Vec3::from(out.vertices[tri[0] as usize].position),
                Vec3::from(out.vertices[tri[1] as usize].position),
                Vec3::from(out.vertices[tri[2] as usize].position),
            ];
            let world_area = 0.5 * (wp[1] - wp[0]).cross(wp[2] - wp[0]).length();
            ratios.push(tri_area(uv) / world_area);
        }
        let mean = ratios.iter().sum::<f32>() / ratios.len() as f32;
        for r in &ratios {
            assert!(
                (r - mean).abs() / mean < 0.05,
                "non-uniform density: ratio {r} vs mean {mean}"
            );
        }
    }

    #[test]
    fn triangles_do_not_overlap() {
        // Pairwise: no two triangles' uv1 bounding boxes overlap (shelf packing +
        // gutter guarantees this), so charts never bleed into each other.
        let out = generate_lightmap_uv(&varied_tri_mesh(12));
        let boxes: Vec<([f32; 2], [f32; 2])> = out
            .indices
            .as_chunks::<3>().0.iter()
            .map(|t| {
                let uv: Vec<[f32; 2]> = t.iter().map(|&i| out.vertices[i as usize].uv1).collect();
                let lo = [
                    uv.iter().map(|u| u[0]).fold(f32::MAX, f32::min),
                    uv.iter().map(|u| u[1]).fold(f32::MAX, f32::min),
                ];
                let hi = [
                    uv.iter().map(|u| u[0]).fold(f32::MIN, f32::max),
                    uv.iter().map(|u| u[1]).fold(f32::MIN, f32::max),
                ];
                (lo, hi)
            })
            .collect();
        for a in 0..boxes.len() {
            for b in (a + 1)..boxes.len() {
                let (la, ha) = boxes[a];
                let (lb, hb) = boxes[b];
                let overlap = la[0] < hb[0] && lb[0] < ha[0] && la[1] < hb[1] && lb[1] < ha[1];
                assert!(!overlap, "triangles {a} and {b} overlap in lightmap UV");
            }
        }
    }

    fn quad(uvs: [[f32; 2]; 4]) -> (Vec<Vertex>, Vec<u32>) {
        let pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]];
        let verts: Vec<Vertex> = (0..4)
            .map(|i| Vertex {
                position: pos[i],
                normal: [0.0, 0.0, 1.0],
                uv: uvs[i],
                color: [1.0; 4],
                tangent: [1.0, 0.0, 0.0, 1.0],
                uv1: [0.0, 0.0],
                joints: [0; 4],
                weights: [0.0; 4],
            })
            .collect();
        (verts, vec![0, 1, 2, 0, 2, 3])
    }

    #[test]
    fn clean_unit_uv0_is_lightmappable() {
        // A quad filling [0,1] (area 1) is non-overlapping → usable (like the couch
        // at area 0.73, bounds [0,1]).
        let (v, i) = quad([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        assert!(uv0_is_lightmappable(&v, &i));
    }

    #[test]
    fn tiled_uv0_is_not_lightmappable() {
        // UVs spilling to v=2 (tiled) → not usable (like the DamagedHelmet, bounds
        // [0,2], area 1.35).
        let (v, i) = quad([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]]);
        assert!(!uv0_is_lightmappable(&v, &i));
    }

    #[test]
    fn overlapping_uv0_is_not_lightmappable() {
        // Two quads stacked on the SAME [0,1] region: in-bounds but total area 2 >
        // 1, so they must overlap → not usable.
        let (mut v, mut i) = quad([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let (v2, i2) = quad([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let base = v.len() as u32;
        v.extend(v2);
        i.extend(i2.iter().map(|x| x + base));
        assert!(!uv0_is_lightmappable(&v, &i));
    }

    #[test]
    fn empty_mesh_is_safe() {
        let out = generate_lightmap_uv(&MeshData { vertices: vec![], indices: vec![], has_lightmap_uv: false });
        assert!(out.has_lightmap_uv);
        assert!(out.vertices.is_empty());
    }
}
