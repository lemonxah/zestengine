//! The cluster DAG builder: snap, weld, cut into clusters, then group,
//! simplify and re-cut level by level, and finally encode and index.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use zestengine_render::{MeshData, Vertex};

use super::partition::{Csr, group_clusters, partition_triangles};
use super::*;
use crate::MorphTarget;
use crate::simplify::{GroupSimplify, simplify_group};

/// Cost of collapsing across a vertex-colour change, so colour edges and gradients survive to coarse levels.
const COLOUR_PENALTY: f32 = 1.0;

/// A cluster while building.
struct Cl {
    tris: Vec<[u32; 3]>,
    bounds: [f32; 4],
    lod_sphere: [f32; 4],
    lod_error: f32,
    parent_sphere: [f32; 4],
    parent_error: f32,
    group: u32,
    parent_group: u32,
    level: u32,
}

/// The mesh on the quantisation grid.
struct Prep {
    /// Source vertices with positions snapped to the grid.
    verts: Vec<Vertex>,
    q: Vec<[u16; 3]>,
    /// Welded point per vertex (equal grid position = equal point).
    point: Vec<u32>,
    points: usize,
    tris: Vec<[u32; 3]>,
    dropped: u32,
    min: [f32; 3],
    step: f32,
    skinned: bool,
    has_extra: bool,
}

/// Whether the DAG must carry the colour stream (`FLAG_EXTRA`): only a real non-white vertex colour does.
/// A second (lightmap) UV set does not: the clustered shader never samples it, and an object with a baked
/// lightmap draws on the ordinary path from the ordinary mesh (`gpu_cull.rs`), so the DAG needs no copy of
/// it. Until `GEO_BUILDER_VERSION` 2 the flag also followed `has_lightmap_uv`, which kept most FBX meshes
/// off the clustered path.
pub fn needs_extra_streams(mesh: &MeshData) -> bool {
    mesh.vertices.iter().any(|v| v.color != [1.0; 4])
}

fn prepare(mesh: &MeshData) -> Result<Prep, String> {
    if mesh.vertices.is_empty() || mesh.indices.len() < 3 {
        return Err("mesh has no triangles".into());
    }
    if mesh.indices.iter().any(|&i| i as usize >= mesh.vertices.len()) {
        return Err("mesh index out of range".into());
    }
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for v in &mesh.vertices {
        for k in 0..3 {
            if !v.position[k].is_finite() {
                return Err("mesh has non-finite positions".into());
            }
            lo[k] = lo[k].min(v.position[k]);
            hi[k] = hi[k].max(v.position[k]);
        }
    }
    let extent = (0..3).map(|k| hi[k] - lo[k]).fold(0.0f32, f32::max);
    let step = if extent > 0.0 { 2f32.powi(((extent / 65535.0).log2().ceil() as i32).clamp(-100, 100)) } else { 1.0 };
    let mut verts = mesh.vertices.clone();
    let mut q = Vec::with_capacity(verts.len());
    let mut weld: crate::simplify::FxMap<[u16; 3], u32> = crate::simplify::FxMap::with_capacity_and_hasher(verts.len(), Default::default());
    let mut point = Vec::with_capacity(verts.len());
    for v in &mut verts {
        let qv: [u16; 3] = std::array::from_fn(|k| (((v.position[k] - lo[k]) as f64 / step as f64).round()).clamp(0.0, 65535.0) as u16);
        v.position = std::array::from_fn(|k| lo[k] + qv[k] as f32 * step);
        let next = weld.len() as u32;
        point.push(*weld.entry(qv).or_insert(next));
        q.push(qv);
    }
    let mut tris = Vec::with_capacity(mesh.indices.len() / 3);
    let mut dropped = 0;
    for t in mesh.indices.as_chunks::<3>().0 {
        let p = t.map(|v| point[v as usize]);
        if p[0] == p[1] || p[1] == p[2] || p[0] == p[2] {
            dropped += 1;
        } else {
            tris.push(*t);
        }
    }
    if tris.is_empty() {
        return Err("every triangle is degenerate".into());
    }
    let skinned = mesh.vertices.iter().any(|v| v.weights.iter().any(|&w| w > 0.0));
    let has_extra = needs_extra_streams(mesh);
    Ok(Prep { points: weld.len(), verts, q, point, tris, dropped, min: lo, step, skinned, has_extra })
}

// ------------------------------------------------------------------ spheres

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn len(a: [f32; 3]) -> f32 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// Grow a sphere over `items` (centre, radius) by Ritter's method and then
/// fix the radius exactly, with a little room for float rounding.
fn enclose(items: &[[f32; 4]]) -> [f32; 4] {
    if items.is_empty() {
        return [0.0; 4];
    }
    let c = |s: &[f32; 4]| [s[0], s[1], s[2]];
    let far = |from: [f32; 3]| items.iter().enumerate().max_by(|a, b| (len(sub(c(a.1), from)) + a.1[3]).total_cmp(&(len(sub(c(b.1), from)) + b.1[3])).then(b.0.cmp(&a.0))).unwrap().1;
    let a = *far(c(&items[0]));
    let b = *far(c(&a));
    let mut ctr = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5];
    let mut r = (len(sub(c(&a), c(&b))) + a[3] + b[3]) * 0.5;
    for _ in 0..2 {
        for s in items {
            let d = len(sub(c(s), ctr));
            if d + s[3] > r {
                let nr = (r + d + s[3]) * 0.5;
                let k = (nr - r) / d.max(1e-30);
                for i in 0..3 {
                    ctr[i] += (s[i] - ctr[i]) * k;
                }
                r = nr;
            }
        }
    }
    let exact = items.iter().map(|s| len(sub(c(s), ctr)) + s[3]).fold(0.0f32, f32::max);
    let mag = ctr.iter().fold(0.0f32, |m, v| m.max(v.abs())) + exact;
    [ctr[0], ctr[1], ctr[2], exact * (1.0 + 1e-5) + mag * 4e-7]
}

fn sphere_of_points(points: &[[f32; 3]]) -> [f32; 4] {
    let items: Vec<[f32; 4]> = points.iter().map(|p| [p[0], p[1], p[2], 0.0]).collect();
    enclose(&items)
}

fn cluster_points(verts: &[Vertex], tris: &[[u32; 3]]) -> Vec<[f32; 3]> {
    let mut ids: Vec<u32> = tris.iter().flatten().copied().collect();
    ids.sort_unstable();
    ids.dedup();
    ids.iter().map(|&v| verts[v as usize].position).collect()
}

// ------------------------------------------------------------------ build

/// Threads a build with `opts` uses (0 = half the cores).
pub fn thread_count(opts: &DagOptions) -> usize {
    // Default: half the cores, so a build on a worker leaves the frame loop room.
    if opts.threads > 0 { opts.threads } else { (std::thread::available_parallelism().map_or(4, |n| n.get()) / 2).clamp(1, 32) }
}

/// Build the DAG of `mesh` (see the module docs). `morphs` are the mesh's
/// blendshapes (sparse deltas by source vertex), carried into the clusters.
/// Deterministic: the same input gives the same bytes on any thread count.
pub fn build_cluster_dag(mesh: &MeshData, morphs: &[MorphTarget], opts: &DagOptions) -> Result<ClusterDag, String> {
    build_cluster_dag_progress(mesh, morphs, opts, &|_| {})
}

/// [`build_cluster_dag`] reporting progress in 0..=1.
pub fn build_cluster_dag_progress(mesh: &MeshData, morphs: &[MorphTarget], opts: &DagOptions, progress: &(dyn Fn(f32) + Sync)) -> Result<ClusterDag, String> {
    build_cluster_dag_ctl(mesh, morphs, opts, progress, &|| false)
}

/// Error text of a build stopped by its `cancelled` hook.
pub const CANCELLED: &str = "cancelled";

/// [`build_cluster_dag_progress`] that also stops (`Err(`[`CANCELLED`]`)`) between
/// levels and groups once `cancelled` says so. The hook is polled at every
/// level and before every group, never inside a single group's simplification
/// (the first partition and the encode are not interruptible either).
pub fn build_cluster_dag_ctl(
    mesh: &MeshData,
    morphs: &[MorphTarget],
    opts: &DagOptions,
    progress: &(dyn Fn(f32) + Sync),
    cancelled: &(dyn Fn() -> bool + Sync),
) -> Result<ClusterDag, String> {
    if morphs.len() > u16::MAX as usize {
        return Err("too many blendshapes".into());
    }
    let prep = prepare(mesh)?;
    progress(0.02);
    let cluster_tris = opts.cluster_triangles.clamp(16, CLUSTER_MAX_TRIS);
    let threads = thread_count(opts);

    // Vertices a blendshape moves cost more to remove (face detail).
    let point_cost: Option<Vec<f32>> = (opts.morph_error_bias > 0.0 && morphs.iter().any(|m| !m.deltas.is_empty())).then(|| {
        let mut cost = vec![1.0f32; prep.points];
        for m in morphs {
            for &(v, dp, _) in &m.deltas {
                if (v as usize) < prep.point.len() && dp.iter().any(|d| d.abs() > 1e-6) {
                    cost[prep.point[v as usize] as usize] = 1.0 + opts.morph_error_bias;
                }
            }
        }
        cost
    });

    // Points on edges shared by more than two triangles (or twice in one
    // direction) are never removed: such seams are not a surface the
    // simplifier can reason about, so they stay as they are in every level.
    let non_manifold: Vec<bool> = {
        let mut dir: Vec<u64> = Vec::with_capacity(prep.tris.len() * 3);
        for t in &prep.tris {
            let p = t.map(|v| prep.point[v as usize]);
            for k in 0..3 {
                dir.push((u64::from(p[k]) << 32) | u64::from(p[(k + 1) % 3]));
            }
        }
        dir.sort_unstable();
        let mut und: Vec<u64> = dir.iter().map(|&k| (k >> 32).min(k & 0xffff_ffff) << 32 | (k >> 32).max(k & 0xffff_ffff)).collect();
        und.sort_unstable();
        let mut bad = vec![false; prep.points];
        let mut mark = |key: u64, limit: usize, list: &[u64]| {
            let lo = list.partition_point(|&e| e < key);
            if list[lo..].iter().take_while(|&&e| e == key).count() > limit {
                bad[(key >> 32) as usize] = true;
                bad[(key & 0xffff_ffff) as usize] = true;
            }
        };
        let mut i = 0;
        while i < dir.len() {
            mark(dir[i], 1, &dir);
            i += 1;
            while i < dir.len() && dir[i] == dir[i - 1] {
                i += 1;
            }
        }
        let mut i = 0;
        while i < und.len() {
            mark(und[i], 2, &und);
            i += 1;
            while i < und.len() && und[i] == und[i - 1] {
                i += 1;
            }
        }
        bad
    };

    // Finest level.
    let mut clusters: Vec<Cl> = Vec::new();
    let leaf_parts = partition_triangles(&prep.tris, &prep.verts, &prep.point, prep.skinned, cluster_tris);
    for part in &leaf_parts {
        let tris: Vec<[u32; 3]> = part.iter().map(|&t| prep.tris[t as usize]).collect();
        let b = sphere_of_points(&cluster_points(&prep.verts, &tris));
        clusters.push(Cl { tris, bounds: b, lod_sphere: b, lod_error: 0.0, parent_sphere: b, parent_error: ROOT_PARENT_ERROR, group: NO_LINK, parent_group: NO_LINK, level: 0 });
    }
    let leaf_count = clusters.len();
    let mut groups: Vec<GroupRecord> = Vec::new();
    let mut group_children: Vec<u32> = Vec::new();
    let mut current: Vec<u32> = (0..leaf_count as u32).collect();
    let mut boundary_ratio = 0.0f32;
    let (mut stall, mut level) = (0u32, 0u32);
    progress(0.1);

    while current.len() > 1 {
        if cancelled() {
            return Err(CANCELLED.into());
        }
        let n = current.len();
        let final_merge = stall >= 2 || level >= 40;
        let level_tris: usize = current.iter().map(|&c| clusters[c as usize].tris.len()).sum();
        // Directed point edges of the level, sorted, with their owner (index into `current`).
        let mut edges: Vec<(u64, u32)> = Vec::new();
        for (i, &c) in current.iter().enumerate() {
            for t in &clusters[c as usize].tris {
                let p = t.map(|v| prep.point[v as usize]);
                for k in 0..3 {
                    edges.push(((u64::from(p[k]) << 32) | u64::from(p[(k + 1) % 3]), i as u32));
                }
            }
        }
        edges.sort_unstable();
        let twins = |a: u32, b: u32| -> &[(u64, u32)] {
            let key = (u64::from(b) << 32) | u64::from(a);
            let lo = edges.partition_point(|e| e.0 < key);
            let hi = lo + edges[lo..].partition_point(|e| e.0 == key);
            &edges[lo..hi]
        };
        let mut links: Vec<(u32, u32, f32)> = Vec::new();
        let mut border_edges = 0usize;
        for (i, &c) in current.iter().enumerate() {
            for t in &clusters[c as usize].tris {
                let p = t.map(|v| prep.point[v as usize]);
                for k in 0..3 {
                    for &(_, d) in twins(p[k], p[(k + 1) % 3]) {
                        if d as usize != i {
                            border_edges += 1;
                            let dc = current[d as usize];
                            let sibling = clusters[c as usize].group != NO_LINK && clusters[c as usize].group == clusters[dc as usize].group;
                            links.push((i as u32, d, if sibling { 0.5 } else { 1.0 }));
                        }
                    }
                }
            }
        }
        if level == 0 {
            boundary_ratio = border_edges as f32 / (prep.tris.len() * 3) as f32;
        }
        let adj = Csr::from_links(n, links);
        let centroids: Vec<[f32; 3]> = current.iter().map(|&c| [clusters[c as usize].bounds[0], clusters[c as usize].bounds[1], clusters[c as usize].bounds[2]]).collect();
        let plan: Vec<Vec<u32>> = if final_merge { vec![(0..n as u32).collect()] } else { group_clusters(&centroids, &adj, opts.group_clusters.clamp(8, 32), 8, 32) };
        let mut group_of = vec![0u32; n];
        for (gi, g) in plan.iter().enumerate() {
            for &m in g {
                group_of[m as usize] = gi as u32;
            }
        }

        let reduce = |gi: usize| -> (Vec<Cl>, f32, [f32; 4]) {
            let members = &plan[gi];
            let mut tris: Vec<[u32; 3]> = Vec::new();
            let mut locked: Vec<u32> = Vec::new();
            let mut outline: Vec<u32> = Vec::new();
            for &m in members {
                for t in &clusters[current[m as usize] as usize].tris {
                    tris.push(*t);
                    let p = t.map(|v| prep.point[v as usize]);
                    for k in 0..3 {
                        let (a, b) = (p[k], p[(k + 1) % 3]);
                        if non_manifold[a as usize] {
                            locked.push(a);
                        }
                        if twins(a, b).iter().any(|&(_, d)| group_of[d as usize] != gi as u32) {
                            locked.push(a);
                            locked.push(b);
                            outline.push(a);
                            outline.push(b);
                        }
                    }
                }
            }
            locked.sort_unstable();
            locked.dedup();
            outline.sort_unstable();
            outline.dedup();
            let parents = if final_merge { 1 } else { tris.len().div_ceil(2 * cluster_tris) };
            let target = parents * cluster_tris;
            let simplify = |relax_seams: bool| {
                simplify_group(&GroupSimplify {
                    vertices: &prep.verts,
                    point: &prep.point,
                    tris: &tris,
                    locked: &locked,
                    target_tris: target,
                    point_cost: point_cost.as_deref(),
                    skin_penalty: if prep.skinned { opts.skin_penalty } else { 0.0 },
                    colour_penalty: if prep.has_extra { COLOUR_PENALTY } else { 0.0 },
                    relax_seams,
                    pinned: &outline,
                })
            };
            let mut res = simplify(false);
            // A group stuck far above its target (hair cards, faceted or seam-riddled parts: every corner of a
            // removed point needs a partner on the kept one) is retried allowing attribute jumps at seams;
            // without that, such meshes keep thousands of triangles at every distance.
            if res.tris.len() > target + target / 4 {
                let relaxed = simplify(true);
                if relaxed.tris.len() < res.tris.len() {
                    res = relaxed;
                }
            }
            let child_err = members.iter().map(|&m| clusters[current[m as usize] as usize].lod_error).fold(0.0f32, f32::max);
            let error = child_err + res.error;
            let spheres: Vec<[f32; 4]> = members.iter().map(|&m| clusters[current[m as usize] as usize].lod_sphere).collect();
            let sphere = enclose(&spheres);
            let parts = partition_triangles(&res.tris, &prep.verts, &prep.point, prep.skinned, cluster_tris);
            let outs = parts
                .iter()
                .map(|part| {
                    let tris: Vec<[u32; 3]> = part.iter().map(|&t| res.tris[t as usize]).collect();
                    let b = sphere_of_points(&cluster_points(&prep.verts, &tris));
                    Cl { tris, bounds: b, lod_sphere: sphere, lod_error: error, parent_sphere: sphere, parent_error: ROOT_PARENT_ERROR, group: NO_LINK, parent_group: NO_LINK, level: level + 1 }
                })
                .collect();
            (outs, error, sphere)
        };
        /// One worker's output: its cluster list, worst error and enclosing sphere.
        type BuildResult<Cl> = (Vec<Cl>, f32, [f32; 4]);
        let results: Mutex<Vec<Option<BuildResult<Cl>>>> = Mutex::new((0..plan.len()).map(|_| None).collect());
        let next = AtomicUsize::new(0);
        std::thread::scope(|s| {
            for _ in 0..threads.min(plan.len()).max(1) {
                s.spawn(|| {
                    loop {
                        let gi = next.fetch_add(1, Ordering::Relaxed);
                        if gi >= plan.len() || cancelled() {
                            break;
                        }
                        let r = reduce(gi);
                        results.lock().unwrap()[gi] = Some(r);
                    }
                });
            }
        });
        if cancelled() {
            return Err(CANCELLED.into());
        }
        let mut produced: Vec<u32> = Vec::new();
        for (gi, res) in results.into_inner().unwrap().into_iter().enumerate() {
            let (outs, error, sphere) = res.expect("every group is reduced");
            let id = groups.len() as u32;
            let first_child = group_children.len() as u32;
            for &m in &plan[gi] {
                let c = current[m as usize];
                group_children.push(c);
                let cl = &mut clusters[c as usize];
                cl.parent_group = id;
                cl.parent_error = error;
                cl.parent_sphere = sphere;
            }
            let first_output = clusters.len() as u32;
            let output_count = outs.len() as u32;
            for mut o in outs {
                o.group = id;
                produced.push(clusters.len() as u32);
                clusters.push(o);
            }
            groups.push(GroupRecord { first_child, child_count: plan[gi].len() as u32, first_output, output_count, level: level + 1, error });
        }
        // A level that removes under 5% of the triangles made no progress (the cluster count can oscillate
        // while the same triangles come back, so it is not a usable signal).
        let produced_tris: usize = produced.iter().map(|&c| clusters[c as usize].tris.len()).sum();
        stall = if produced.len() >= n || produced_tris * 100 >= level_tris * 95 { stall + 1 } else { 0 };
        level += 1;
        progress(0.1 + 0.7 * (1.0 - produced.len() as f32 / leaf_count.max(1) as f32));
        current = produced;
        if final_merge {
            break;
        }
    }
    // Roots judge themselves on their own sphere.
    for &c in &current {
        let cl = &mut clusters[c as usize];
        cl.parent_sphere = cl.lod_sphere;
        cl.parent_error = ROOT_PARENT_ERROR;
    }
    progress(0.85);
    let mut dag = encode(&prep, &clusters, groups, group_children, morphs, opts)?;
    dag.header.levels = level + 1;
    dag.header.boundary_edge_ratio = boundary_ratio;
    dag.header.leaf_clusters = leaf_count as u32;
    dag.nodes = build_bvh(&dag);
    progress(1.0);
    Ok(dag)
}

// ---------------------------------------------------------------- encode

/// One source vertex's morph deltas: (target index, delta position, delta normal).
type MorphDeltas = Vec<(u16, [f32; 3], [f32; 3])>;

fn encode(prep: &Prep, clusters: &[Cl], groups: Vec<GroupRecord>, group_children: Vec<u32>, morphs: &[MorphTarget], opts: &DagOptions) -> Result<ClusterDag, String> {
    let mut dag = ClusterDag {
        header: DagHeader {
            mesh_min: prep.min,
            quant_step: prep.step,
            source_vertices: prep.verts.len() as u32,
            source_triangles: prep.tris.len() as u32,
            dropped_triangles: prep.dropped,
            morph_targets: morphs.len() as u32,
            flags: if prep.has_extra { FLAG_EXTRA } else { 0 } | if prep.skinned { FLAG_SKINNED } else { 0 },
            ..Default::default()
        },
        groups,
        group_children,
        ..Default::default()
    };
    // Blendshape deltas by source vertex.
    let mut delta_at: HashMap<u32, MorphDeltas> = HashMap::new();
    for (ti, m) in morphs.iter().enumerate() {
        for &(v, dp, dn) in &m.deltas {
            if (v as usize) < prep.verts.len() {
                delta_at.entry(v).or_default().push((ti as u16, dp, dn));
            }
        }
    }
    let f16 = |v: [f32; 3]| v.map(|x| half::f16::from_f32(x).to_bits());
    let mut local = vec![u32::MAX; prep.verts.len()];
    let (mut fill_sum, mut tight_sum, mut tight_n) = (0.0f64, 0.0f64, 0usize);
    for cl in clusters {
        let vertex_offset = dag.vertices.len();
        let mut src: Vec<u32> = Vec::new();
        let triangle_offset = dag.triangles.len();
        for t in &cl.tris {
            for &v in t {
                if local[v as usize] == u32::MAX {
                    local[v as usize] = src.len() as u32;
                    src.push(v);
                }
                dag.triangles.push(local[v as usize] as u8);
            }
        }
        while !dag.triangles.len().is_multiple_of(4) {
            dag.triangles.push(0);
        }
        for &v in &src {
            local[v as usize] = u32::MAX;
        }
        if src.len() > CLUSTER_MAX_VERTS {
            return Err(format!("cluster with {} vertices", src.len()));
        }
        // Bones: the distinct joints this cluster's vertices use.
        let mut bones: Vec<u32> = Vec::new();
        if prep.skinned {
            for &v in &src {
                let vert = &prep.verts[v as usize];
                for k in 0..4 {
                    if vert.weights[k] > 0.0 {
                        bones.push(vert.joints[k]);
                    }
                }
            }
            bones.sort_unstable();
            bones.dedup();
            if bones.len() > CLUSTER_MAX_BONES {
                return Err(format!("cluster with {} bones", bones.len()));
            }
            if bones.last().is_some_and(|&b| b > u16::MAX as u32) {
                return Err("joint index beyond 65535".into());
            }
        }
        let bone_offset = dag.bone_ids.len();
        dag.bone_ids.extend(bones.iter().map(|&b| b as u16));
        let mut points: Vec<[f32; 3]> = Vec::with_capacity(src.len());
        for &v in &src {
            let vert = &prep.verts[v as usize];
            let src_v = &prep.verts[v as usize];
            let t = vert.tangent;
            let mut flags = 0u16;
            if t[3] < 0.0 {
                flags |= 1;
            }
            dag.vertices.push(ClusterVertex { pos: prep.q[v as usize], flags, normal: oct_encode(vert.normal), tangent: oct_encode([t[0], t[1], t[2]]), uv: src_v.uv, color: vert.color.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8) });
            dag.vertex_source.push(v);
            points.push(vert.position);
            if prep.skinned {
                let mut s = ClusterVertexSkin::default();
                let sum: f32 = (0..4).filter(|&k| vert.weights[k] > 0.0).map(|k| vert.weights[k]).sum();
                if sum > 0.0 {
                    let mut total = 0i32;
                    let mut best = 0usize;
                    for k in 0..4 {
                        if vert.weights[k] > 0.0 {
                            s.bones[k] = bones.binary_search(&vert.joints[k]).unwrap() as u8;
                            s.weights[k] = (vert.weights[k] / sum * 255.0).round() as u8;
                            total += i32::from(s.weights[k]);
                            if s.weights[k] > s.weights[best] {
                                best = k;
                            }
                        }
                    }
                    s.weights[best] = (i32::from(s.weights[best]) + 255 - total).clamp(0, 255) as u8;
                }
                dag.vertex_skin.push(s);
            }
        }
        // Blendshape deltas of the cluster's vertices, inherited by identity of the source vertex.
        let morph_offset = dag.morph_entries.len();
        if !delta_at.is_empty() {
            let mut entries: Vec<MorphEntry> = Vec::new();
            for (li, v) in src.iter().enumerate() {
                for &(target, dp, dn) in delta_at.get(v).into_iter().flatten() {
                    entries.push(MorphEntry { target, vertex: li as u16, d_pos: f16(dp), d_normal: f16(dn) });
                }
            }
            entries.sort_by_key(|e| (e.target, e.vertex));
            dag.morph_entries.extend(entries);
        }
        // Backface cone from the decoded triangles.
        let mut axis = [0.0f32; 3];
        let mut normals: Vec<[f32; 3]> = Vec::with_capacity(cl.tris.len());
        for t in &cl.tris {
            let p = t.map(|v| prep.verts[v as usize].position);
            let (e1, e2) = (sub(p[1], p[0]), sub(p[2], p[0]));
            let c = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            let l = len(c);
            if l > 0.0 {
                for k in 0..3 {
                    axis[k] += c[k];
                }
                normals.push([c[0] / l, c[1] / l, c[2] / l]);
            }
        }
        let al = len(axis);
        let (mut cone_axis, mut cone_cutoff) = ([0i8; 3], 255u8);
        if al > 0.0 {
            let a = [axis[0] / al, axis[1] / al, axis[2] / al];
            let q: [i8; 3] = a.map(|x| (x * 127.0).round().clamp(-127.0, 127.0) as i8);
            let qa = {
                let v = q.map(|x| x as f32 / 127.0);
                let l = len(v).max(1e-20);
                [v[0] / l, v[1] / l, v[2] / l]
            };
            let min_dot = normals.iter().map(|n| qa[0] * n[0] + qa[1] * n[1] + qa[2] * n[2]).fold(1.0f32, f32::min);
            if min_dot > 0.05 {
                let alpha = min_dot.clamp(-1.0, 1.0).acos() + 0.03;
                cone_axis = q;
                cone_cutoff = if alpha >= std::f32::consts::FRAC_PI_2 { 255 } else { (alpha.sin() * 255.0).ceil().min(255.0) as u8 };
            }
        }
        let lo = points.iter().fold([f32::MAX; 3], |m, p| std::array::from_fn(|k| m[k].min(p[k])));
        let hi = points.iter().fold([f32::MIN; 3], |m, p| std::array::from_fn(|k| m[k].max(p[k])));
        if cl.level == 0 {
            fill_sum += cl.tris.len() as f64 / CLUSTER_MAX_TRIS as f64;
            let r_box = len(sub(hi, lo)) * 0.5;
            if r_box > 0.0 {
                tight_sum += (cl.bounds[3] / r_box) as f64;
                tight_n += 1;
            }
        }
        dag.clusters.push(GpuCluster {
            bounds: cl.bounds,
            lod_sphere: cl.lod_sphere,
            parent_lod_sphere: cl.parent_sphere,
            lod_error: cl.lod_error,
            parent_lod_error: cl.parent_error,
            cone_axis,
            cone_cutoff,
            vertex_offset: vertex_offset as u32,
            triangle_offset: triangle_offset as u32,
            vertex_count: src.len() as u16,
            triangle_count: cl.tris.len() as u16,
            bone_offset: bone_offset as u32,
            bone_count: bones.len() as u16,
            flags: u16::from(prep.skinned),
            morph_offset: morph_offset as u32,
            morph_count: (dag.morph_entries.len() - morph_offset) as u32,
            section: opts.section,
            level: cl.level,
        });
        dag.links.push(ClusterLink { group: cl.group, parent_group: cl.parent_group, level: cl.level });
    }
    let leafs = clusters.iter().filter(|c| c.level == 0).count().max(1);
    dag.header.fill = (fill_sum / leafs as f64) as f32;
    dag.header.bounds_tightness = (tight_sum / tight_n.max(1) as f64) as f32;
    Ok(dag)
}

// -------------------------------------------------------------------- BVH

struct Run {
    first: u32,
    count: u32,
    bounds: [f32; 4],
    lod: [f32; 4],
    max_err: f32,
}

fn make_child(runs: &[Run], idx: &mut [usize], nodes: &mut Vec<BvhNode>) -> BvhChild {
    if idx.len() == 1 {
        let r = &runs[idx[0]];
        return BvhChild { bounds: r.bounds, lod_sphere: r.lod, max_parent_error: r.max_err, child: BVH_LEAF | r.first, count: r.count, _pad: 0 };
    }
    let slot = nodes.len();
    nodes.push(BvhNode { children: [BvhChild::EMPTY; 4] });
    let centre = |i: usize| [runs[i].bounds[0], runs[i].bounds[1], runs[i].bounds[2]];
    // Split along the widest axis of the centres, twice: up to four parts.
    fn halves<'a>(idx: &'a mut [usize], centre: &dyn Fn(usize) -> [f32; 3]) -> (&'a mut [usize], &'a mut [usize]) {
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for &i in idx.iter() {
            let c = centre(i);
            for k in 0..3 {
                lo[k] = lo[k].min(c[k]);
                hi[k] = hi[k].max(c[k]);
            }
        }
        let ax = (0..3).max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])).then(b.cmp(&a))).unwrap();
        idx.sort_by(|&a, &b| centre(a)[ax].total_cmp(&centre(b)[ax]).then(a.cmp(&b)));
        let mid = idx.len() / 2;
        idx.split_at_mut(mid)
    }
    let (a, b) = halves(idx, &centre);
    let mut parts: Vec<&mut [usize]> = Vec::with_capacity(4);
    for half in [a, b] {
        if half.len() > 1 {
            let (x, y) = halves(half, &centre);
            parts.push(x);
            parts.push(y);
        } else {
            parts.push(half);
        }
    }
    let mut kids = [BvhChild::EMPTY; 4];
    for (k, part) in parts.into_iter().enumerate() {
        kids[k] = make_child(runs, part, nodes);
    }
    nodes[slot].children = kids;
    let live: Vec<&BvhChild> = kids.iter().filter(|k| k.child != BVH_EMPTY).collect();
    let bounds = enclose(&live.iter().map(|k| k.bounds).collect::<Vec<_>>());
    let lod_sphere = enclose(&live.iter().map(|k| k.lod_sphere).collect::<Vec<_>>());
    let max_parent_error = live.iter().map(|k| k.max_parent_error).fold(0.0f32, f32::max);
    BvhChild { bounds, lod_sphere, max_parent_error, child: slot as u32, count: 0, _pad: 0 }
}

/// A 4-wide BVH over runs of consecutive clusters (a run never crosses the
/// group that produced it, so a run shares its parent data).
fn build_bvh(dag: &ClusterDag) -> Vec<BvhNode> {
    let mut runs: Vec<Run> = Vec::new();
    let mut i = 0;
    while i < dag.clusters.len() {
        let g = dag.links[i].group;
        let mut j = i + 1;
        while j < dag.clusters.len() && j - i < BVH_RUN_MAX && dag.links[j].group == g {
            j += 1;
        }
        let cs = &dag.clusters[i..j];
        runs.push(Run {
            first: i as u32,
            count: (j - i) as u32,
            bounds: enclose(&cs.iter().map(|c| c.bounds).collect::<Vec<_>>()),
            lod: enclose(&cs.iter().map(|c| if c.is_root() { c.lod_sphere } else { c.parent_lod_sphere }).collect::<Vec<_>>()),
            max_err: cs.iter().map(|c| c.parent_lod_error).fold(0.0f32, f32::max),
        });
        i = j;
    }
    let mut nodes = Vec::new();
    let mut idx: Vec<usize> = (0..runs.len()).collect();
    let top = make_child(&runs, &mut idx, &mut nodes);
    if nodes.is_empty() {
        let mut root = BvhNode { children: [BvhChild::EMPTY; 4] };
        root.children[0] = top;
        nodes.push(root);
    }
    nodes
}
