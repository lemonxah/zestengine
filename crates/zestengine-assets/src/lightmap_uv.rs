//! Lightmap UV generation for imported meshes that ship without a second UV set.
//!
//! Many glTF/FBX models only carry TEXCOORD_0 (their texture UVs), which usually
//! overlap (parts share texture space). Baking a lightmap into overlapping UVs
//! produces garbage. When the user opts in (the bake offers it), we generate a
//! **non-overlapping** lightmap UV into `uv1`.
//!
//! Charts come from geometry, independently of UV0 and its seams or overlaps.
//! Planar projections are checked for overlapping triangles and split as needed,
//! then packed into UV1 with gutters and a common scale. Texture UVs and other
//! vertex attributes are preserved. The result is deterministic on reimport.

use zestengine_render::{MeshData, Vertex};
use glam::Vec3;

/// Whether a mesh's `uv0` is usable AS-IS for lightmapping — i.e. it's a
/// non-overlapping chart layout inside roughly the [0,1] square. An area sum
/// below one alone is insufficient: small mirrored/stacked islands can overlap
/// while leaving most of the square empty. Check triangle interiors as well.
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
    let in_unit = lo[0] > -0.02 && lo[1] > -0.02 && hi[0] < 1.02 && hi[1] < 1.02;
    in_unit && area_sum > 0.0 && area_sum <= 1.02
        && !has_overlap(&indices.as_chunks::<3>().0.iter().map(|t| [vertices[t[0] as usize].uv, vertices[t[1] as usize].uv, vertices[t[2] as usize].uv]).collect::<Vec<_>>())
}

/// Strict interior intersection by separating axes. Shared edges/vertices are
/// allowed. Double precision keeps the test stable for small atlas triangles.
fn overlaps(a: [[f32; 2]; 3], b: [[f32; 2]; 3]) -> bool {
    let cross = |a: [f32; 2], b: [f32; 2], p: [f32; 2]| {
        (b[0] as f64 - a[0] as f64) * (p[1] as f64 - a[1] as f64)
            - (b[1] as f64 - a[1] as f64) * (p[0] as f64 - a[0] as f64)
    };
    if cross(a[0], a[1], a[2]) == 0.0 || cross(b[0], b[1], b[2]) == 0.0 {
        return false;
    }
    for tri in [a, b] {
        for i in 0..3 {
            let projection = |p| cross(tri[i], tri[(i + 1) % 3], p);
            let aa = a.map(projection);
            let bb = b.map(projection);
            let min = |p: [f64; 3]| p.into_iter().fold(f64::INFINITY, f64::min);
            let max = |p: [f64; 3]| p.into_iter().fold(f64::NEG_INFINITY, f64::max);
            let tolerance = aa.into_iter().chain(bb).map(f64::abs).fold(0.0, f64::max) * 1e-12;
            if max(aa).min(max(bb)) <= min(aa).max(min(bb)) + tolerance {
                return false;
            }
        }
    }
    true
}

fn has_overlap(tris: &[[[f32; 2]; 3]]) -> bool {
    let bounds: Vec<_> = tris.iter().map(|t| {
        let lo = std::array::from_fn::<_, 2, _>(|k| t.iter().map(|p| p[k]).fold(f32::INFINITY, f32::min));
        let hi = std::array::from_fn::<_, 2, _>(|k| t.iter().map(|p| p[k]).fold(f32::NEG_INFINITY, f32::max));
        (lo, hi)
    }).collect();
    let mut order: Vec<_> = (0..tris.len()).collect();
    order.sort_by(|&a, &b| bounds[a].0[0].total_cmp(&bounds[b].0[0]).then(a.cmp(&b)));
    let mut active: Vec<usize> = Vec::new();
    for i in order {
        active.retain(|&j| bounds[j].1[0] > bounds[i].0[0]);
        for &j in &active {
            if bounds[j].0[1] < bounds[i].1[1] && bounds[i].0[1] < bounds[j].1[1] && overlaps(tris[i], tris[j]) {
                return true;
            }
        }
        active.push(i);
    }
    false
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

/// Locally consistent normals do not guarantee a globally injective planar
/// projection (e.g. a spiral can wind over itself). Split only failing charts,
/// retaining contiguous, seamless charts everywhere the projection is valid.
fn append_nonoverlapping(island: Island, out: &mut Vec<Island>) {
    let projected: Vec<_> = island.tris.iter().map(|t| t.map(|i| island.local[i as usize])).collect();
    if !has_overlap(&projected) {
        out.push(island);
        return;
    }
    for tris in island.tris.chunks(island.tris.len().div_ceil(2)) {
        let mut remap = std::collections::HashMap::new();
        let mut verts = Vec::new();
        let mut local = Vec::new();
        let tris = tris.iter().map(|t| t.map(|i| *remap.entry(i).or_insert_with(|| {
            let next = verts.len() as u32;
            verts.push(island.verts[i as usize]);
            local.push(island.local[i as usize]);
            next
        }))).collect();
        let (local, w, h) = fit(local);
        append_nonoverlapping(Island { verts, local, tris, w, h }, out);
    }
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
/// the charts are packed with a gutter (never overlapping), then a single
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
    let areas: Vec<f32> = tris
        .iter()
        .map(|t| {
            let p = t.map(|i| Vec3::from(mesh.vertices[i as usize].position));
            0.5 * (p[1] - p[0]).cross(p[2] - p[0]).length()
        })
        .collect();

    use std::collections::HashMap;
    let mut position_ids: HashMap<[u32; 3], u32> = HashMap::new();
    let pid: Vec<u32> = mesh
        .vertices
        .iter()
        .map(|v| {
            let next = position_ids.len() as u32;
            *position_ids.entry(v.position.map(|c| (c + 0.0).to_bits())).or_insert(next)
        })
        .collect();
    let edge_key = |a: u32, b: u32| {
        let (a, b) = (pid[a as usize], pid[b as usize]);
        if a < b { (a, b) } else { (b, a) }
    };
    let mut edges: HashMap<(u32, u32), Vec<(usize, u32)>> = HashMap::new();
    for (ti, t) in tris.iter().enumerate() {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let key = edge_key(a, b);
            if key.0 != key.1 {
                edges.entry(key).or_default().push((ti, pid[a as usize]));
            }
        }
    }
    let neighbours = |t: usize| {
        let tri = tris[t];
        let edges = &edges;
        [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
            .into_iter()
            .filter_map(move |(a, b)| {
                let edge = edges.get(&edge_key(a, b))?;
                // Only weld an unambiguous, oppositely directed shared edge.
                // Coincident shells and non-manifold branches need separate charts.
                if edge.len() != 2 || edge[0].1 == edge[1].1 { return None; }
                Some(if edge[0].0 == t { edge[1].0 } else { edge[0].0 })
            })
    };

    const CHART_COS: f32 = 0.5;
    const MERGE_COS: f32 = 0.2;
    const SMALL_CHART: f32 = 1e-3;
    const UNASSIGNED: usize = usize::MAX;
    let mut chart_of = vec![UNASSIGNED; tri_count];
    let mut axes: Vec<Vec3> = Vec::new();
    let mut chart_area: Vec<f32> = Vec::new();
    let grow = |seed: usize, chart_of: &mut Vec<usize>, axes: &mut Vec<Vec3>, chart_area: &mut Vec<f32>| {
        let cid = axes.len();
        let axis = normals[seed];
        chart_of[seed] = cid;
        let mut area = areas[seed];
        let mut stack = vec![seed];
        while let Some(t) = stack.pop() {
            for nb in neighbours(t) {
                if chart_of[nb] == UNASSIGNED && normals[nb].dot(axis) > CHART_COS {
                    chart_of[nb] = cid;
                    area += areas[nb];
                    stack.push(nb);
                }
            }
        }
        axes.push(axis);
        chart_area.push(area);
    };
    for seed in 0..tri_count {
        if chart_of[seed] == UNASSIGNED {
            grow(seed, &mut chart_of, &mut axes, &mut chart_area);
        }
    }

    let total_area: f32 = areas.iter().sum();
    let small: Vec<bool> = chart_area.iter().map(|&a| a < total_area * SMALL_CHART).collect();
    if small.iter().any(|s| !s) {
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        for t in 0..tri_count {
            if small[chart_of[t]] {
                chart_of[t] = UNASSIGNED;
                queue.push_back(t);
            }
        }
        while let Some(t) = queue.pop_front() {
            if chart_of[t] != UNASSIGNED {
                continue;
            }
            let mut best: Option<(f32, usize)> = None;
            for nb in neighbours(t) {
                let c = chart_of[nb];
                if c == UNASSIGNED {
                    continue;
                }
                let d = normals[t].dot(axes[c]);
                if d > MERGE_COS && best.is_none_or(|(bd, bc)| d > bd || (d == bd && c < bc)) {
                    best = Some((d, c));
                }
            }
            if let Some((_, c)) = best {
                chart_of[t] = c;
                queue.extend(neighbours(t).filter(|&nb| chart_of[nb] == UNASSIGNED));
            }
        }
        for seed in 0..tri_count {
            if chart_of[seed] == UNASSIGNED {
                grow(seed, &mut chart_of, &mut axes, &mut chart_area);
            }
        }
    }

    let mut members: Vec<Vec<usize>> = vec![Vec::new(); axes.len()];
    for (t, &c) in chart_of.iter().enumerate() {
        members[c].push(t);
    }
    let mut islands: Vec<Island> = Vec::new();
    for (cid, members) in members.iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        let mut slot_of: HashMap<u32, u32> = HashMap::new();
        let mut orig: Vec<u32> = Vec::new();
        let mut local_tris: Vec<[u32; 3]> = Vec::with_capacity(members.len());
        let mut avg_n = Vec3::ZERO;
        for &m in members {
            avg_n += normals[m] * areas[m].max(1e-12);
            let lt = tris[m].map(|v| {
                *slot_of.entry(v).or_insert_with(|| {
                    orig.push(v);
                    orig.len() as u32 - 1
                })
            });
            local_tris.push(lt);
        }
        let avg_n = avg_n.normalize_or(axes[cid]);
        let axis = if members.iter().all(|&m| normals[m].dot(avg_n) > MERGE_COS) { avg_n } else { axes[cid] };
        let (local, w, h) = fit(project(mesh, &orig, axis).0);
        append_nonoverlapping(Island { verts: orig.iter().map(|&i| mesh.vertices[i as usize]).collect(), local, tris: local_tris, w, h }, &mut islands);
    }

    let placed = pack_islands(&islands);
    let mut vertices: Vec<Vertex> = Vec::with_capacity(islands.iter().map(|c| c.verts.len()).sum());
    let mut indices: Vec<u32> = Vec::with_capacity(tri_count * 3);
    for (c, place) in islands.iter().zip(&placed) {
        let base = vertices.len() as u32;
        for (k, mut v) in c.verts.iter().copied().enumerate() {
            let l = if place.turned { [c.local[k][1], c.local[k][0]] } else { c.local[k] };
            v.uv1 = [(place.origin[0] + l[0] * place.scale).clamp(0.0, 1.0), (place.origin[1] + l[1] * place.scale).clamp(0.0, 1.0)];
            vertices.push(v);
        }
        for lt in &c.tris {
            indices.extend_from_slice(&[base + lt[0], base + lt[1], base + lt[2]]);
        }
    }

    MeshData { vertices, indices, has_lightmap_uv: true }
}

struct Placed {
    origin: [f32; 2],
    scale: f32,
    turned: bool,
}

fn fit(local: Vec<[f32; 2]>) -> (Vec<[f32; 2]>, f32, f32) {
    let bounds = |angle: f32| {
        let (s, c) = angle.sin_cos();
        let mut lo = [f32::INFINITY; 2];
        let mut hi = [f32::NEG_INFINITY; 2];
        for p in &local {
            let q = [p[0] * c - p[1] * s, p[0] * s + p[1] * c];
            for k in 0..2 {
                lo[k] = lo[k].min(q[k]);
                hi[k] = hi[k].max(q[k]);
            }
        }
        (lo, hi)
    };
    const STEPS: u32 = 16;
    let mut best = (0.0f32, f32::INFINITY);
    for step in 0..STEPS {
        let angle = step as f32 * std::f32::consts::FRAC_PI_2 / STEPS as f32;
        let (lo, hi) = bounds(angle);
        let area = (hi[0] - lo[0]) * (hi[1] - lo[1]);
        if area < best.1 * 0.999 {
            best = (angle, area);
        }
    }
    let (lo, hi) = bounds(best.0);
    let (s, c) = best.0.sin_cos();
    let turned = local.iter().map(|p| [p[0] * c - p[1] * s - lo[0], p[0] * s + p[1] * c - lo[1]]).collect();
    (turned, (hi[0] - lo[0]).max(1e-6), (hi[1] - lo[1]).max(1e-6))
}

fn pack_islands(islands: &[Island]) -> Vec<Placed> {
    const GRID: u32 = 1024;
    const MAX_PACKED: usize = 6000;
    const NOMINAL_DENSITY: f32 = 40.0;
    const GUTTER_TEXELS: f32 = 1.5;
    let bbox_area: f32 = islands.iter().map(|c| c.w * c.h).sum();
    let longest = islands.iter().map(|c| c.w.max(c.h)).fold(0.0f32, f32::max);
    if islands.len() <= MAX_PACKED && bbox_area > 0.0 {
        let expected_texels = (NOMINAL_DENSITY * (bbox_area / 0.6).sqrt()).max(64.0);
        let padding = ((GUTTER_TEXELS * GRID as f32 / expected_texels).ceil() as u32).clamp(2, 16);
        let settings = crate::atlas_pack::PackSettings { padding, allow_rotation: true, max_size: GRID, pow2: false, uniform_pages: false };
        let cells = |scale: f32| -> Vec<(u32, u32)> { islands.iter().map(|c| (((c.w * scale).ceil() as u32).max(1), ((c.h * scale).ceil() as u32).max(1))).collect() };
        let upper = (GRID as f32 / bbox_area.sqrt()).min((GRID - 2 * padding) as f32 / longest.max(1e-6));
        let attempt = |scale: f32| crate::atlas_pack::pack(&cells(scale), &settings).ok().filter(|r| r.pages.len() == 1);
        let mut found: Option<(f32, crate::atlas_pack::PackResult)> = None;
        let (mut lo, mut hi) = (0.0f32, 1.0f32);
        for _ in 0..9 {
            let mid = 0.5 * (lo + hi);
            let scale = upper * 0.5f32.powf(4.0 * (1.0 - mid));
            match attempt(scale) {
                Some(result) => {
                    found = Some((scale, result));
                    lo = mid;
                }
                None => hi = mid,
            }
        }
        if found.is_none() {
            let scale = upper / 16.0;
            found = attempt(scale).map(|r| (scale, r));
        }
        if let Some((scale, result)) = found {
            return result
                .placements
                .iter()
                .map(|p| Placed { origin: [p.x as f32 / GRID as f32, p.y as f32 / GRID as f32], scale: scale / GRID as f32, turned: p.rotated })
                .collect();
        }
    }
    let atlas_w = (bbox_area.sqrt() * 1.4).max(1e-6);
    let gutter = atlas_w * 0.01;
    let mut order: Vec<usize> = (0..islands.len()).collect();
    order.sort_by(|&a, &b| islands[b].h.partial_cmp(&islands[a].h).unwrap_or(core::cmp::Ordering::Equal));
    let mut at = vec![[0.0f32; 2]; islands.len()];
    let (mut cx, mut cy, mut row_h, mut max_x, mut max_y) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for &i in &order {
        let (w, h) = (islands[i].w, islands[i].h);
        if cx > 0.0 && cx + w > atlas_w {
            cx = 0.0;
            cy += row_h + gutter;
            row_h = 0.0;
        }
        at[i] = [cx, cy];
        cx += w + gutter;
        row_h = row_h.max(h);
        max_x = max_x.max(cx);
        max_y = max_y.max(cy + h);
    }
    let inset = 0.01f32;
    let scale = (1.0 - 2.0 * inset) / max_x.max(max_y).max(1e-6);
    at.iter().map(|p| Placed { origin: [inset + p[0] * scale, inset + p[1] * scale], scale, turned: false }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent check: clip each triangle against the other and measure the
    // intersection polygon, rather than relying on the generator's SAT test.
    fn assert_no_uv_overlap(mesh: &MeshData) {
        let tris: Vec<_> = mesh.indices.as_chunks::<3>().0.iter().map(|t| t.iter().map(|&i| mesh.vertices[i as usize].uv1.map(f64::from)).collect::<Vec<_>>()).collect();
        let cross = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        for (i, a) in tris.iter().enumerate() {
            for b in &tris[i + 1..] {
                let winding = cross(b[0], b[1], b[2]).signum();
                let mut polygon = a.clone();
                for k in 0..3 {
                    let mut clipped = Vec::new();
                    for j in 0..polygon.len() {
                        let (p, q) = (polygon[j], polygon[(j + 1) % polygon.len()]);
                        let dp = winding * cross(b[k], b[(k + 1) % 3], p);
                        let dq = winding * cross(b[k], b[(k + 1) % 3], q);
                        if dp >= 0.0 { clipped.push(p); }
                        if (dp > 0.0 && dq < 0.0) || (dp < 0.0 && dq > 0.0) {
                            let t = dp / (dp - dq);
                            clipped.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
                        }
                    }
                    polygon = clipped;
                }
                let area: f64 = (1..polygon.len().saturating_sub(1)).map(|k| cross(polygon[0], polygon[k], polygon[k + 1]).abs() * 0.5).sum();
                assert!(area < 1e-12, "UV1 triangles overlap by {area}: {a:?} and {b:?}");
            }
        }
    }

    #[test]
    fn touching_shells_do_not_share_overlapping_charts() {
        for height in [0.0, 0.25] {
            let positions = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                [0., 0., 0.], [1., 0., 0.], [0., 1., height]];
            let mesh = MeshData { vertices: positions.map(mk_vert).to_vec(), indices: (0..6).collect(), has_lightmap_uv: false };
            let out = generate_lightmap_uv(&mesh);
            assert_eq!(out.indices.len(), 6);
            assert_no_uv_overlap(&out);
        }
    }

    #[test]
    fn a_spiral_is_split_where_its_projection_overlaps() {
        let mut mesh = MeshData { vertices: Vec::new(), indices: Vec::new(), has_lightmap_uv: false };
        for i in 0..=96 {
            let angle = i as f32 * std::f32::consts::TAU / 32.0;
            for radius in [1.0, 2.0] {
                mesh.vertices.push(mk_vert([radius * angle.cos(), radius * angle.sin(), angle * 0.01]));
            }
            if i < 96 {
                let a = i * 2;
                mesh.indices.extend_from_slice(&[a, a + 1, a + 2, a + 1, a + 3, a + 2]);
            }
        }
        let out = generate_lightmap_uv(&mesh);
        assert_eq!(out.indices.len(), mesh.indices.len());
        assert_no_uv_overlap(&out);
    }

    #[test]
    fn generated_uv1_ignores_texture_uvs_and_preserves_them() {
        let mut mesh = varied_tri_mesh(4);
        let first = generate_lightmap_uv(&mesh);
        for (i, vertex) in mesh.vertices.iter_mut().enumerate() {
            vertex.uv = [i as f32 * -7.0, 12.0];
            vertex.uv1 = [0.5; 2];
        }
        let second = generate_lightmap_uv(&mesh);
        assert_eq!(first.indices, second.indices);
        assert_eq!(first.vertices.iter().map(|v| v.uv1).collect::<Vec<_>>(), second.vertices.iter().map(|v| v.uv1).collect::<Vec<_>>());
        for v in &second.vertices {
            let source = mesh.vertices.iter().find(|s| s.position == v.position).unwrap();
            assert_eq!(v.uv, source.uv);
            assert_eq!(v.normal, source.normal);
            assert_eq!(v.tangent, source.tangent);
        }
        assert_no_uv_overlap(&second);
    }

    #[test]
    fn small_stacked_texture_islands_are_not_lightmappable() {
        let mut mesh = tri_mesh(2);
        for triangle in mesh.vertices.as_chunks_mut::<3>().0 {
            for (v, uv) in triangle.iter_mut().zip([[0., 0.], [0.25, 0.], [0., 0.25]]) { v.uv = uv; }
        }
        assert!(!uv0_is_lightmappable(&mesh.vertices, &mesh.indices));
        assert_no_uv_overlap(&generate_lightmap_uv(&mesh));
    }

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
    fn a_texture_seam_does_not_split_a_flat_surface() {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for (tri, uv) in [([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], [0.1, 0.1]), ([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], [0.7, 0.7])] {
            for pos in tri {
                indices.push(vertices.len() as u32);
                vertices.push(Vertex { uv, ..mk_vert(pos) });
            }
        }
        let out = generate_lightmap_uv(&MeshData { vertices, indices, has_lightmap_uv: false });
        let at = |pos: [f32; 3]| -> Vec<[f32; 2]> { out.vertices.iter().filter(|v| v.position == pos).map(|v| v.uv1).collect() };
        for corner in [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]] {
            let uvs = at(corner);
            assert_eq!(uvs.len(), 2);
            assert!((uvs[0][0] - uvs[1][0]).abs() < 1e-6 && (uvs[0][1] - uvs[1][1]).abs() < 1e-6, "the shared edge is split in the lightmap: {uvs:?}");
        }
    }

    #[test]
    fn a_curved_strip_stays_in_few_charts_with_a_gutter_between_them() {
        let (rings, segments) = (24usize, 48usize);
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for r in 0..=rings {
            for s in 0..segments {
                let a = s as f32 / segments as f32 * std::f32::consts::TAU;
                vertices.push(mk_vert([a.cos(), r as f32 * 0.1, a.sin()]));
            }
        }
        for r in 0..rings {
            for s in 0..segments {
                let (a, b) = ((r * segments + s) as u32, (r * segments + (s + 1) % segments) as u32);
                let (c, d) = (a + segments as u32, b + segments as u32);
                indices.extend_from_slice(&[a, c, b, b, c, d]);
            }
        }
        let out = generate_lightmap_uv(&MeshData { vertices, indices, has_lightmap_uv: false });
        let mut keys: std::collections::HashMap<[u32; 5], usize> = Default::default();
        let ids: Vec<usize> = out.vertices.iter().map(|v| {
            let next = keys.len();
            *keys.entry([v.position[0].to_bits(), v.position[1].to_bits(), v.position[2].to_bits(), v.uv1[0].to_bits(), v.uv1[1].to_bits()]).or_insert(next)
        }).collect();
        let mut parent: Vec<usize> = (0..keys.len()).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize { while p[x] != x { p[x] = p[p[x]]; x = p[x]; } x }
        for t in out.indices.as_chunks::<3>().0 {
            let a = find(&mut parent, ids[t[0] as usize]);
            for k in 1..3 { let b = find(&mut parent, ids[t[k] as usize]); parent[b] = a; }
        }
        let mut boxes: std::collections::HashMap<usize, ([f32; 2], [f32; 2])> = Default::default();
        for (v, &id) in out.vertices.iter().zip(&ids) {
            let root = find(&mut parent, id);
            let b = boxes.entry(root).or_insert(([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]));
            for k in 0..2 { b.0[k] = b.0[k].min(v.uv1[k]); b.1[k] = b.1[k].max(v.uv1[k]); }
        }
        assert!(boxes.len() <= 8, "a cylinder wall became {} charts", boxes.len());
        let boxes: Vec<_> = boxes.values().collect();
        for (i, a) in boxes.iter().enumerate() {
            for b in &boxes[i + 1..] {
                let gap = (0..2).map(|k| (a.0[k] - b.1[k]).max(b.0[k] - a.1[k])).fold(f32::NEG_INFINITY, f32::max);
                assert!(gap >= 3.0 / 1024.0, "charts are {gap} apart");
            }
        }
        let area: f32 = out.indices.as_chunks::<3>().0.iter().map(|t| tri_area([out.vertices[t[0] as usize].uv1, out.vertices[t[1] as usize].uv1, out.vertices[t[2] as usize].uv1])).sum();
        assert!(area > 0.3, "the charts fill only {area} of the map");
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
