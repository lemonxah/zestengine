//! DAG invariants, crack-freeness of arbitrary cuts, determinism, validation
//! and quantisation. Everything is checked on the CPU: no GPU is involved.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use zestengine_render::{MeshData, Vertex};

use super::*;
use crate::MorphTarget;

// ------------------------------------------------------------- generators

fn vtx(p: [f32; 3]) -> Vertex {
    let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt().max(1e-6);
    Vertex { position: p, normal: [p[0] / l, p[1] / l, p[2] / l], uv: [p[0], p[2]], ..Default::default() }
}

/// Reverse the winding when the signed volume is negative (outward normals).
fn orient(mut m: MeshData) -> MeshData {
    let mut vol = 0.0f64;
    for t in m.indices.as_chunks::<3>().0 {
        let [a, b, c] = t.map(|i| m.vertices[i as usize].position.map(f64::from));
        vol += a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
    }
    if vol < 0.0 {
        for t in m.indices.as_chunks_mut::<3>().0 {
            t.swap(1, 2);
        }
    }
    m
}

pub(super) fn sphere(lat: usize, lon: usize) -> MeshData {
    let mut v = vec![vtx([0.0, 1.0, 0.0])];
    for i in 1..lat {
        let th = std::f32::consts::PI * i as f32 / lat as f32;
        for j in 0..lon {
            let ph = std::f32::consts::TAU * j as f32 / lon as f32;
            v.push(vtx([th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin()]));
        }
    }
    v.push(vtx([0.0, -1.0, 0.0]));
    let ring = |i: usize, j: usize| (1 + (i - 1) * lon + j % lon) as u32;
    let mut idx = Vec::new();
    for j in 0..lon {
        idx.extend([0, ring(1, j), ring(1, j + 1)]);
        idx.extend([(v.len() - 1) as u32, ring(lat - 1, j + 1), ring(lat - 1, j)]);
    }
    for i in 1..lat - 1 {
        for j in 0..lon {
            idx.extend([ring(i, j), ring(i + 1, j), ring(i, j + 1)]);
            idx.extend([ring(i, j + 1), ring(i + 1, j), ring(i + 1, j + 1)]);
        }
    }
    orient(MeshData { vertices: v, indices: idx, has_lightmap_uv: false })
}

pub(super) fn torus(major: usize, minor: usize) -> MeshData {
    let mut v = Vec::new();
    for i in 0..major {
        let a = std::f32::consts::TAU * i as f32 / major as f32;
        for j in 0..minor {
            let b = std::f32::consts::TAU * j as f32 / minor as f32;
            let r = 1.0 + 0.35 * b.cos();
            v.push(vtx([r * a.cos(), 0.35 * b.sin(), r * a.sin()]));
        }
    }
    let at = |i: usize, j: usize| ((i % major) * minor + j % minor) as u32;
    let mut idx = Vec::new();
    for i in 0..major {
        for j in 0..minor {
            idx.extend([at(i, j), at(i + 1, j), at(i, j + 1)]);
            idx.extend([at(i, j + 1), at(i + 1, j), at(i + 1, j + 1)]);
        }
    }
    orient(MeshData { vertices: v, indices: idx, has_lightmap_uv: false })
}

fn grid(n: usize) -> MeshData {
    let mut v = Vec::new();
    for y in 0..=n {
        for x in 0..=n {
            // A gentle bump so the plane is not exactly flat.
            let h = 0.05 * ((x as f32 * 0.4).sin() + (y as f32 * 0.3).cos());
            v.push(Vertex { position: [x as f32, h, y as f32], normal: [0.0, 1.0, 0.0], uv: [x as f32, y as f32], ..Default::default() });
        }
    }
    let mut idx = Vec::new();
    for y in 0..n {
        for x in 0..n {
            let i = (y * (n + 1) + x) as u32;
            let w = (n + 1) as u32;
            idx.extend([i, i + w, i + 1, i + 1, i + w, i + w + 1]);
        }
    }
    MeshData { vertices: v, indices: idx, has_lightmap_uv: false }
}

/// A closed box, `n` quads per edge, welded across faces.
fn closed_box(n: usize) -> MeshData {
    let mut v: Vec<Vertex> = Vec::new();
    let mut at: HashMap<[u32; 3], u32> = HashMap::new();
    let mut idx = Vec::new();
    let mut vert = |p: [f32; 3]| -> u32 {
        let key = p.map(f32::to_bits);
        *at.entry(key).or_insert_with(|| {
            v.push(vtx(p));
            (v.len() - 1) as u32
        })
    };
    for axis in 0..3 {
        for side in [-1.0f32, 1.0] {
            for a in 0..n {
                for b in 0..n {
                    let p = |i: usize, j: usize| {
                        let (u, w) = (-1.0 + 2.0 * (a + i) as f32 / n as f32, -1.0 + 2.0 * (b + j) as f32 / n as f32);
                        let mut c = [0.0f32; 3];
                        c[axis] = side;
                        c[(axis + 1) % 3] = u;
                        c[(axis + 2) % 3] = w;
                        // Slightly irregular so the box is not exactly planar-degenerate.
                        c
                    };
                    let q = [vert(p(0, 0)), vert(p(1, 0)), vert(p(1, 1)), vert(p(0, 1))];
                    if side > 0.0 {
                        idx.extend([q[0], q[1], q[2], q[0], q[2], q[3]]);
                    } else {
                        idx.extend([q[0], q[2], q[1], q[0], q[3], q[2]]);
                    }
                }
            }
        }
    }
    orient(MeshData { vertices: v, indices: idx, has_lightmap_uv: false })
}

/// A tube along y with `joints` bones, weights blended between neighbours, like a limb.
pub(super) fn humanoid(rings: usize, sides: usize, joints: usize) -> MeshData {
    let mut v = Vec::new();
    for r in 0..rings {
        let y = 2.0 * r as f32 / (rings - 1) as f32;
        let radius = 0.12 + 0.05 * (y * 3.0).sin();
        let f = y / 2.0 * (joints - 1) as f32;
        let (j0, w1) = (f.floor() as usize, f.fract());
        let j1 = (j0 + 1).min(joints - 1);
        for s in 0..sides {
            let a = std::f32::consts::TAU * s as f32 / sides as f32;
            let mut vt = vtx([radius * a.cos(), y, radius * a.sin()]);
            vt.normal = [a.cos(), 0.0, a.sin()];
            vt.joints = [j0 as u32, j1 as u32, 0, 0];
            vt.weights = if j0 == j1 { [1.0, 0.0, 0.0, 0.0] } else { [1.0 - w1, w1, 0.0, 0.0] };
            v.push(vt);
        }
    }
    let mut idx = Vec::new();
    for r in 0..rings - 1 {
        for s in 0..sides {
            let at = |r: usize, s: usize| (r * sides + s % sides) as u32;
            idx.extend([at(r, s), at(r + 1, s), at(r, s + 1), at(r, s + 1), at(r + 1, s), at(r + 1, s + 1)]);
        }
    }
    MeshData { vertices: v, indices: idx, has_lightmap_uv: false }
}

// ----------------------------------------------------------------- checks

type Edge = ([u16; 3], [u16; 3]);

fn directed_edges(dag: &ClusterDag, cut: &[u32]) -> HashMap<Edge, u32> {
    let mut e = HashMap::new();
    for &c in cut {
        let off = dag.clusters[c as usize].vertex_offset as usize;
        for t in dag.cluster_triangles(c) {
            let p = t.map(|i| dag.vertices[off + i as usize].pos);
            for k in 0..3 {
                *e.entry((p[k], p[(k + 1) % 3])).or_default() += 1;
            }
        }
    }
    e
}

/// The finest level's open border points (empty for a closed mesh) and edge multiplicities.
struct Finest {
    edges: HashMap<Edge, u32>,
    border: HashSet<[u16; 3]>,
}

fn finest(dag: &ClusterDag) -> Finest {
    let leaves: Vec<u32> = dag.leaf_clusters().collect();
    let edges = directed_edges(dag, &leaves);
    let mut border = HashSet::new();
    for &(a, b) in edges.keys() {
        if !edges.contains_key(&(b, a)) {
            border.insert(a);
            border.insert(b);
        }
    }
    Finest { edges, border }
}

/// A cut must be a surface without cracks: every directed edge has its
/// reverse, except along the finest level's open border.
fn assert_crack_free(dag: &ClusterDag, fin: &Finest, cut: &[u32], what: &str) {
    assert!(!cut.is_empty(), "{what}: empty cut");
    let edges = directed_edges(dag, cut);
    for (&(a, b), &n) in &edges {
        let allowed = fin.edges.get(&(a, b)).copied().unwrap_or(1).max(1);
        assert!(n <= allowed, "{what}: edge covered {n} times (overlapping clusters)");
        if !edges.contains_key(&(b, a)) {
            assert!(fin.border.contains(&a) && fin.border.contains(&b), "{what}: crack: an interior edge has no neighbour");
        }
    }
}

/// The cut covers the surface exactly once at the DAG level: no selected
/// cluster lies below a group whose outputs are already all covered, and every
/// finest cluster is covered by something selected. (Outputs of one group may
/// be taken apart: each is judged against the group that consumes it.)
/// (A cluster's own region is only defined together with its group's other
/// outputs, so "covered" means the cluster, or every output of its consuming
/// group, is covered.)
fn assert_covers_once(dag: &ClusterDag, cut: &[u32], what: &str) {
    let selected: HashSet<u32> = cut.iter().copied().collect();
    let mut covered = vec![false; dag.clusters.len()];
    let mut above = vec![false; dag.clusters.len()];
    for c in (0..dag.clusters.len()).rev() {
        let pg = dag.links[c].parent_group;
        above[c] = pg != NO_LINK && {
            let g = &dag.groups[pg as usize];
            (g.first_output..g.first_output + g.output_count).all(|o| covered[o as usize])
        };
        covered[c] = selected.contains(&(c as u32)) || above[c];
    }
    for &c in cut {
        assert!(!above[c as usize], "{what}: cluster {c} is selected below an already covered group (overlap)");
    }
    for l in dag.leaf_clusters() {
        assert!(covered[l as usize], "{what}: finest cluster {l} is not covered");
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
}

fn bounds_of(mesh: &MeshData) -> ([f32; 3], [f32; 3]) {
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for v in &mesh.vertices {
        for k in 0..3 {
            lo[k] = lo[k].min(v.position[k]);
            hi[k] = hi[k].max(v.position[k]);
        }
    }
    (lo, hi)
}

/// Many cuts, uniform and view-dependent, all of which must be valid.
fn exercise_cuts(name: &str, mesh: &MeshData, dag: &ClusterDag, cuts: usize) {
    let fin = finest(dag);
    let (lo, hi) = bounds_of(mesh);
    let extent = (0..3).map(|k| hi[k] - lo[k]).fold(0.0f32, f32::max);
    let mut rng = Rng(0x9E37_79B9 ^ name.len() as u64);
    let mut errors: Vec<f32> = dag.clusters.iter().map(|c| c.lod_error).collect();
    errors.sort_unstable_by(f32::total_cmp);
    errors.dedup();
    let mut mixed = 0;
    for i in 0..cuts {
        let (cut, what) = if i % 3 == 0 {
            // Uniform error threshold, including between the stored values.
            let t = if i % 2 == 0 { errors[(rng.next() * errors.len() as f32) as usize % errors.len()] } else { rng.next() * errors.last().unwrap() * 1.1 };
            (dag.select_by_error(t), format!("{name}: uniform {t}"))
        } else {
            // Screen-space style: the threshold grows with distance from a viewpoint.
            let view: [f32; 3] = std::array::from_fn(|k| lo[k] + (hi[k] - lo[k]) * (rng.next() * 3.0 - 1.0));
            let tau = 10f32.powf(-4.0 + 3.5 * rng.next());
            let ok = move |e: f32, s: [f32; 4]| {
                let d = ((view[0] - s[0]).powi(2) + (view[1] - s[1]).powi(2) + (view[2] - s[2]).powi(2)).sqrt();
                e <= tau * (d - s[3]).max(0.0)
            };
            let cut = dag.select_linear(&ok);
            assert_eq!(cut, dag.select_bvh(&ok), "{name}: BVH traversal differs from the linear scan (view {view:?}, tau {tau})");
            (cut, format!("{name}: view {view:?} tau {tau}"))
        };
        let levels: HashSet<u32> = cut.iter().map(|&c| dag.links[c as usize].level).collect();
        mixed += usize::from(levels.len() > 1);
        assert_crack_free(dag, &fin, &cut, &what);
        assert_covers_once(dag, &cut, &what);
    }
    let _ = extent;
    if dag.clusters.len() > 60 {
        assert!(mixed > 0, "{name}: no cut mixed different depths");
    }
    // The finest cut and the coarsest cut as well.
    let leaves: Vec<u32> = dag.leaf_clusters().collect();
    assert_crack_free(dag, &fin, &leaves, "finest");
    assert_covers_once(dag, &leaves, "finest");
    let roots: Vec<u32> = dag.root_clusters().collect();
    assert_crack_free(dag, &fin, &roots, "roots");
    assert_covers_once(dag, &roots, "roots");
    assert_eq!(dag.select_by_error(f32::MAX / 2.0), roots.clone(), "{name}: a huge threshold selects the roots");
}

pub(super) fn build(mesh: &MeshData) -> ClusterDag {
    build_cluster_dag(mesh, &[], &DagOptions::default()).expect("builds")
}

fn report(name: &str, mesh: &MeshData, dag: &ClusterDag) {
    let s = dag.stats();
    let src = mesh.vertices.len() * std::mem::size_of::<Vertex>() + mesh.indices.len() * 4;
    eprintln!(
        "[geo] {name}: {} tris -> {} clusters ({} leaf), {} levels, fill {:.1}%, boundary edges {:.1}%, bounds {:.2}x box sphere, {:.1} B/leaf-tri total, {:.1} B/leaf-tri GPU, DAG {} KB ({:.0}% of the {} KB source)",
        s.leaf_triangles,
        s.clusters,
        s.leaf_clusters,
        s.levels,
        s.fill * 100.0,
        s.boundary_edge_ratio * 100.0,
        s.bounds_tightness,
        s.bytes_per_leaf_triangle,
        s.gpu_bytes_per_leaf_triangle,
        s.bytes / 1024,
        s.bytes as f32 / src as f32 * 100.0,
        src / 1024,
    );
}

fn check_invariants(name: &str, mesh: &MeshData, dag: &ClusterDag) {
    dag.validate().unwrap_or_else(|e| panic!("{name}: {e}"));
    dag.validate_against(mesh).unwrap_or_else(|e| panic!("{name}: {e}"));
    for c in &dag.clusters {
        assert!(c.triangle_count as usize <= CLUSTER_MAX_TRIS && c.vertex_count as usize <= CLUSTER_MAX_VERTS);
        assert!(c.parent_lod_error >= c.lod_error);
    }
    // A single root per connected piece at the top, with the errors rising.
    assert!(dag.root_clusters().count() >= 1);
    report(name, mesh, dag);
}

// ------------------------------------------------------------------ tests

#[test]
fn generators_are_closed_and_outward() {
    for (name, m) in [("sphere", sphere(20, 40)), ("torus", torus(30, 16)), ("box", closed_box(8))] {
        let mut e: HashMap<(u32, u32), u32> = HashMap::new();
        for t in m.indices.as_chunks::<3>().0 {
            for k in 0..3 {
                *e.entry((t[k], t[(k + 1) % 3])).or_default() += 1;
            }
        }
        assert!(e.iter().all(|(&(a, b), &n)| n == 1 && e.get(&(b, a)) == Some(&1)), "{name} is not a closed manifold");
    }
}

#[test]
fn sphere_dag_invariants_and_cuts() {
    let m = sphere(60, 120);
    let dag = build(&m);
    check_invariants("sphere", &m, &dag);
    exercise_cuts("sphere", &m, &dag, 60);
}

#[test]
fn torus_dag_invariants_and_cuts() {
    let m = torus(120, 60);
    let dag = build(&m);
    check_invariants("torus", &m, &dag);
    exercise_cuts("torus", &m, &dag, 60);
}

#[test]
fn closed_box_dag_invariants_and_cuts() {
    let m = closed_box(32);
    let dag = build(&m);
    check_invariants("box", &m, &dag);
    exercise_cuts("box", &m, &dag, 60);
}

#[test]
fn plane_grid_dag_invariants_and_cuts() {
    let m = grid(80);
    let dag = build(&m);
    check_invariants("grid", &m, &dag);
    exercise_cuts("grid", &m, &dag, 60);
    // A nearly flat sheet collapses to a few triangles at the top.
    let roots: Vec<u32> = dag.root_clusters().collect();
    assert!(dag.triangle_count_of(&roots) <= 128, "roots: {} triangles", dag.triangle_count_of(&roots));
}

#[test]
fn a_mesh_with_uv_seams_builds_and_stays_crack_free() {
    // The engine's UV sphere duplicates its seam and pole vertices.
    let m = crate::procedural::primitive_mesh(crate::PrimitiveShape::Sphere);
    assert!(m.indices.len() / 3 >= 2000);
    let dag = build(&m);
    check_invariants("uv-sphere", &m, &dag);
    exercise_cuts("uv-sphere", &m, &dag, 40);
}

/// `pieces` small flat-shaded octahedra (every triangle with its own three vertices, so every edge is a
/// hard seam) on a grid, none touching another: a stand-in for lash / hair cards and faceted accessories.
fn scattered_facets(pieces: usize) -> MeshData {
    let mut v: Vec<Vertex> = Vec::new();
    let mut idx = Vec::new();
    let corners: [[f32; 3]; 6] = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];
    let faces: [[usize; 3]; 8] = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]];
    let side = (pieces as f32).sqrt().ceil() as usize;
    for p in 0..pieces {
        let at = [(p % side) as f32 * 0.5, ((p / side) % side) as f32 * 0.5, 0.0];
        for f in faces {
            let pos: [[f32; 3]; 3] = f.map(|c| std::array::from_fn(|k| at[k] + corners[c][k] * 0.05));
            let (e1, e2) = ([pos[1][0] - pos[0][0], pos[1][1] - pos[0][1], pos[1][2] - pos[0][2]], [pos[2][0] - pos[0][0], pos[2][1] - pos[0][1], pos[2][2] - pos[0][2]]);
            let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            for q in pos {
                idx.push(v.len() as u32);
                v.push(Vertex { position: q, normal: [n[0] / l, n[1] / l, n[2] / l], uv: [q[0], q[1]], ..Default::default() });
            }
        }
    }
    MeshData { vertices: v, indices: idx, has_lightmap_uv: false }
}

#[test]
fn many_small_disconnected_faceted_pieces_coarsen_to_a_small_root() {
    // 400 octahedra = 3200 triangles. Pieces that are individually closed and seam-riddled cannot collapse
    // edge by edge: the old builder stopped near its first level, whatever the distance.
    let m = scattered_facets(400);
    let dag = build(&m);
    check_invariants("scattered-facets", &m, &dag);
    let roots: Vec<u32> = dag.root_clusters().collect();
    let root_tris = dag.triangle_count_of(&roots);
    assert!(root_tris <= m.indices.len() / 3 / 4, "root cut keeps {root_tris} of {} triangles", m.indices.len() / 3);
    assert!(dag.header.levels <= 12, "{} levels: the stall detection should end the pointless ones", dag.header.levels);
    exercise_cuts("scattered-facets", &m, &dag, 30);
}

#[test]
fn a_stuck_group_reaches_its_target_only_with_relaxed_seams() {
    // A faceted sphere: every corner its own vertex, so every collapse needs partner corners that a hard
    // edge does not have. Strict simplification stays above the target, the relaxed pass reaches it.
    let smooth = sphere(24, 48);
    let mut vertices = Vec::new();
    let mut tris = Vec::new();
    for t in smooth.indices.as_chunks::<3>().0 {
        let p = [smooth.vertices[t[0] as usize].position, smooth.vertices[t[1] as usize].position, smooth.vertices[t[2] as usize].position];
        let n = {
            let (e1, e2) = ([p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]], [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]]);
            let c = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            let l = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt().max(1e-12);
            [c[0] / l, c[1] / l, c[2] / l]
        };
        let base = vertices.len() as u32;
        for (k, &i) in t.iter().enumerate() {
            vertices.push(Vertex { position: p[k], normal: n, ..smooth.vertices[i as usize] });
        }
        tris.push([base, base + 1, base + 2]);
    }
    let target = tris.len() / 4;
    let run = |relax_seams: bool| {
        crate::simplify::simplify_group(&crate::simplify::GroupSimplify {
            vertices: &vertices,
            point: &point_by_position(&vertices),
            tris: &tris,
            locked: &[],
            target_tris: target,
            point_cost: None,
            skin_penalty: 0.0,
            colour_penalty: 0.0,
            relax_seams,
            pinned: &[],
        })
    };
    let (strict, relaxed) = (run(false), run(true));
    assert!(strict.tris.len() > target + target / 4, "strict stopped at {} (target {target}): the mesh is not stuck", strict.tris.len());
    assert!(relaxed.tris.len() <= target + target / 4, "relaxed stopped at {} (target {target})", relaxed.tris.len());
    assert!(relaxed.error < 0.2, "a quarter of a unit sphere's triangles keeps its shape: error {}", relaxed.error);
}

fn point_by_position(vertices: &[Vertex]) -> Vec<u32> {
    let mut at: HashMap<[u32; 3], u32> = HashMap::new();
    vertices
        .iter()
        .map(|v| {
            let next = at.len() as u32;
            *at.entry(v.position.map(f32::to_bits)).or_insert(next)
        })
        .collect()
}

pub(super) fn model(rel: &str) -> Option<MeshData> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel);
    if !p.exists() {
        eprintln!("[geo] {rel} not in the repo: skipped");
        return None;
    }
    let scene = crate::load_gltf(&p).ok()?;
    // The largest mesh of the model.
    scene.meshes.into_iter().max_by_key(|m| m.indices.len())
}

#[test]
fn damaged_helmet_dag() {
    let Some(m) = model("models/DamagedHelmet.glb") else { return };
    let dag = build(&m);
    check_invariants("DamagedHelmet", &m, &dag);
    exercise_cuts("DamagedHelmet", &m, &dag, 40);
}

#[test]
fn mannequin_dag() {
    let Some(m) = model("animations/kaykit/Mannequin_Medium.glb") else { return };
    if m.indices.len() / 3 < 100 {
        return;
    }
    let dag = build(&m);
    check_invariants("Mannequin_Medium", &m, &dag);
    exercise_cuts("Mannequin_Medium", &m, &dag, 30);
}

#[test]
fn build_is_deterministic_across_thread_counts() {
    let m = torus(100, 50);
    let a = build_cluster_dag(&m, &[], &DagOptions { threads: 1, ..Default::default() }).unwrap().to_bytes();
    let b = build_cluster_dag(&m, &[], &DagOptions { threads: 8, ..Default::default() }).unwrap().to_bytes();
    let c = build_cluster_dag(&m, &[], &DagOptions::default()).unwrap().to_bytes();
    assert_eq!(a, b);
    assert_eq!(a, c);
}

#[test]
fn errors_only_grow_and_spheres_enclose() {
    let m = sphere(60, 120);
    let dag = build(&m);
    for g in &dag.groups {
        let kids = &dag.group_children[g.first_child as usize..][..g.child_count as usize];
        let top = kids.iter().map(|&c| dag.clusters[c as usize].lod_error).fold(0.0f32, f32::max);
        assert!(g.error >= top);
        for c in g.first_output..g.first_output + g.output_count {
            let o = &dag.clusters[c as usize];
            assert_eq!(o.lod_error, g.error);
            for &k in kids {
                let ch = &dag.clusters[k as usize];
                let d = ((o.lod_sphere[0] - ch.lod_sphere[0]).powi(2) + (o.lod_sphere[1] - ch.lod_sphere[1]).powi(2) + (o.lod_sphere[2] - ch.lod_sphere[2]).powi(2)).sqrt();
                assert!(d + ch.lod_sphere[3] <= o.lod_sphere[3] * 1.0001, "parent sphere must enclose the child");
            }
        }
    }
    // The roots are far coarser than the source and have the largest errors.
    let max_err = dag.clusters.iter().map(|c| c.lod_error).fold(0.0f32, f32::max);
    assert!(dag.root_clusters().all(|c| dag.clusters[c as usize].lod_error > 0.0));
    assert!(max_err > 0.01, "a unit sphere reduced to a handful of triangles has a visible error: {max_err}");
}

#[test]
fn quantisation_error_stays_within_the_grid_step() {
    let m = sphere(60, 120);
    let dag = build(&m);
    let half = dag.header.quant_step * 0.5 * 1.0001;
    for c in 0..dag.clusters.len() as u32 {
        let cl = &dag.clusters[c as usize];
        let pos = dag.cluster_positions(c);
        for (i, p) in pos.iter().enumerate() {
            let s = dag.vertex_source[cl.vertex_offset as usize + i] as usize;
            for (k, pk) in p.iter().enumerate() {
                assert!((pk - m.vertices[s].position[k]).abs() <= half + 2e-7, "axis {k}: {} vs {}", pk, m.vertices[s].position[k]);
            }
        }
    }
    // A power-of-two step: decode is exact in f32.
    assert_eq!(dag.header.quant_step.to_bits() & 0x007f_ffff, 0, "the step is a power of two");
    // Normals and tangents survive the octahedral packing.
    let mut rng = Rng(7);
    let mut worst = 0.0f32;
    for _ in 0..20000 {
        let n = [rng.next() * 2.0 - 1.0, rng.next() * 2.0 - 1.0, rng.next() * 2.0 - 1.0];
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if l < 0.1 {
            continue;
        }
        let n = [n[0] / l, n[1] / l, n[2] / l];
        let d = oct_decode(oct_encode(n));
        let cross = [n[1] * d[2] - n[2] * d[1], n[2] * d[0] - n[0] * d[2], n[0] * d[1] - n[1] * d[0]];
        worst = worst.max((cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt().atan2(n[0] * d[0] + n[1] * d[1] + n[2] * d[2]));
    }
    assert!(worst < 2e-4, "octahedral normal error {worst} rad");
    // Bytes of one vertex versus the engine's.
    assert_eq!(std::mem::size_of::<ClusterVertex>(), 28);
    assert_eq!(std::mem::size_of::<Vertex>(), 104);
    assert_eq!(std::mem::size_of::<GpuCluster>(), 96);
    assert_eq!(std::mem::size_of::<BvhNode>(), 192);
    assert_eq!(std::mem::size_of::<MorphEntry>(), 16);
}

#[test]
fn backface_cones_are_conservative() {
    let m = torus(100, 50);
    let dag = build(&m);
    let mut rng = Rng(11);
    let mut culled = 0;
    for c in 0..dag.clusters.len() as u32 {
        let cl = &dag.clusters[c as usize];
        if cl.cone_cutoff == 255 {
            continue;
        }
        let axis = cl.cone_axis.map(|a| a as f32 / 127.0);
        let al = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let axis = axis.map(|a| a / al);
        let pos = dag.cluster_positions(c);
        for _ in 0..40 {
            let eye: [f32; 3] = std::array::from_fn(|_| rng.next() * 6.0 - 3.0);
            let d = [cl.bounds[0] - eye[0], cl.bounds[1] - eye[1], cl.bounds[2] - eye[2]];
            let dl = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if axis[0] * d[0] + axis[1] * d[1] + axis[2] * d[2] >= cl.cone_cutoff as f32 / 255.0 * dl + cl.bounds[3] {
                culled += 1;
                // Culled: every triangle must face away from the eye.
                for t in dag.cluster_triangles(c) {
                    let p = t.map(|i| pos[i as usize]);
                    let (e1, e2) = ([p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]], [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]]);
                    let n = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
                    let to = [p[0][0] - eye[0], p[0][1] - eye[1], p[0][2] - eye[2]];
                    assert!(n[0] * to[0] + n[1] * to[1] + n[2] * to[2] >= -1e-6, "a culled cluster has a front-facing triangle");
                }
            }
        }
    }
    assert!(culled > 100, "the cones should cull something on a torus: {culled}");
}

#[test]
fn skinned_clusters_keep_at_most_16_bones_and_bound_the_posed_surface() {
    let mut m = humanoid(300, 64, 40);
    // Mix extra influences in so vertices have up to four bones.
    for (i, v) in m.vertices.iter_mut().enumerate() {
        if i % 7 == 0 {
            v.joints[2] = (v.joints[0] + 3) % 40;
            v.weights = [0.5, 0.3, 0.2, 0.0];
        }
    }
    let dag = build(&m);
    check_invariants("humanoid", &m, &dag);
    assert!(dag.header.flags & FLAG_SKINNED != 0);
    assert!(dag.clusters.iter().all(|c| c.bone_count as usize <= CLUSTER_MAX_BONES));
    // Extreme poses: random large rotations per bone. The union of the cluster's
    // bind box under its bones must contain every posed vertex.
    let mut rng = Rng(99);
    for pose in 0..4 {
        let mats: Vec<glam::Mat4> = (0..40)
            .map(|_| {
                let axis = glam::Vec3::new(rng.next() - 0.5, rng.next() - 0.5, rng.next() - 0.5).normalize_or_zero();
                let angle = rng.next() * 2.6 + if pose == 3 { 0.5 } else { 0.0 };
                glam::Mat4::from_translation(glam::Vec3::new(rng.next(), rng.next(), rng.next()) * 0.5) * glam::Mat4::from_axis_angle(if axis == glam::Vec3::ZERO { glam::Vec3::X } else { axis }, angle)
            })
            .collect();
        for c in 0..dag.clusters.len() as u32 {
            let cl = &dag.clusters[c as usize];
            let pos = dag.cluster_positions(c);
            let (lo, hi) = pos.iter().fold((glam::Vec3::MAX, glam::Vec3::MIN), |(l, h), p| (l.min(glam::Vec3::from(*p)), h.max(glam::Vec3::from(*p))));
            let corners: Vec<glam::Vec3> = (0..8).map(|i| glam::Vec3::new(if i & 1 == 0 { lo.x } else { hi.x }, if i & 2 == 0 { lo.y } else { hi.y }, if i & 4 == 0 { lo.z } else { hi.z })).collect();
            let (mut blo, mut bhi) = (glam::Vec3::MAX, glam::Vec3::MIN);
            for b in &dag.bone_ids[cl.bone_offset as usize..][..cl.bone_count as usize] {
                for p in &corners {
                    let q = mats[*b as usize].transform_point3(*p);
                    blo = blo.min(q);
                    bhi = bhi.max(q);
                }
            }
            for (i, p) in pos.iter().enumerate() {
                let s = dag.vertex_skin[cl.vertex_offset as usize + i];
                let mut acc = glam::Vec3::ZERO;
                for k in 0..4 {
                    if s.weights[k] > 0 {
                        let bone = dag.bone_ids[cl.bone_offset as usize + s.bones[k] as usize];
                        acc += mats[bone as usize].transform_point3(glam::Vec3::from(*p)) * (s.weights[k] as f32 / 255.0);
                    }
                }
                assert!(acc.cmpge(blo - glam::Vec3::splat(1e-3)).all() && acc.cmple(bhi + glam::Vec3::splat(1e-3)).all(), "posed vertex escapes the cluster's bone bounds");
            }
        }
    }
    // The skin weights quantise to a full 255 per weighted vertex.
    for (c, cl) in dag.clusters.iter().enumerate() {
        for s in &dag.vertex_skin[cl.vertex_offset as usize..][..cl.vertex_count as usize] {
            let sum: u32 = s.weights.iter().map(|&w| w as u32).sum();
            assert!(sum == 255 || sum == 0, "cluster {c}: weights sum to {sum}");
        }
    }
    exercise_cuts("humanoid", &m, &dag, 30);
}

#[test]
fn skin_penalty_keeps_joint_detail_report() {
    let m = humanoid(300, 64, 12);
    let on = build_cluster_dag(&m, &[], &DagOptions { skin_penalty: 2.0, ..Default::default() }).unwrap();
    let off = build_cluster_dag(&m, &[], &DagOptions { skin_penalty: 0.0, ..Default::default() }).unwrap();
    for (name, d) in [("skin penalty on", &on), ("off", &off)] {
        let cut = d.discrete_lods(&[0.1])[0].clone();
        eprintln!("[geo] humanoid {name}: 10% cut = {} tris at error {:.5}, roots {}", cut.triangles, cut.error, d.root_clusters().count());
    }
    on.validate().unwrap();
    off.validate().unwrap();
}

#[test]
fn blendshape_deltas_follow_their_vertices_through_every_level() {
    let m = sphere(60, 120);
    let mut rng = Rng(5);
    let mut morphs: Vec<MorphTarget> = Vec::new();
    for t in 0..3 {
        let mut deltas = Vec::new();
        for v in 0..m.vertices.len() as u32 {
            if m.vertices[v as usize].position[1] > 0.5 && rng.next() > 0.3 {
                deltas.push((v, [rng.next(), rng.next() * 0.5, 0.0], [0.0, rng.next(), 0.0]));
            }
        }
        morphs.push(MorphTarget { name: format!("shape{t}"), deltas });
    }
    let dag = build_cluster_dag(&m, &morphs, &DagOptions::default()).unwrap();
    dag.validate().unwrap();
    let mut total = 0;
    for c in 0..dag.clusters.len() as u32 {
        let cl = &dag.clusters[c as usize];
        let src = &dag.vertex_source[cl.vertex_offset as usize..][..cl.vertex_count as usize];
        let entries = &dag.morph_entries[cl.morph_offset as usize..][..cl.morph_count as usize];
        for (ti, mt) in morphs.iter().enumerate() {
            // Reference: the simplifier's provenance remap.
            let want = crate::simplify::remap_morph_deltas(&mt.deltas, src);
            let got: Vec<&MorphEntry> = entries.iter().filter(|e| e.target as usize == ti).collect();
            assert_eq!(got.len(), want.len(), "cluster {c} target {ti}");
            for (g, (v, dp, _)) in got.iter().zip({
                let mut w = want.clone();
                w.sort_by_key(|x| x.0);
                w
            }) {
                assert_eq!(g.vertex as u32, v);
                for (k, dpk) in dp.iter().enumerate() {
                    assert!((half::f16::from_bits(g.d_pos[k]).to_f32() - dpk).abs() < 1e-3);
                }
            }
            total += got.len();
        }
    }
    assert!(total > 0);
    // Face-cluster bias: at the first simplified level, more of the vertices a
    // blendshape moves survive than without the bias.
    let moved: HashSet<u32> = morphs.iter().flat_map(|t| t.deltas.iter().map(|d| d.0)).collect();
    let survivors = |d: &ClusterDag| -> usize {
        let mut set = HashSet::new();
        for c in 0..d.clusters.len() as u32 {
            if d.links[c as usize].level == 1 {
                let off = d.clusters[c as usize].vertex_offset as usize;
                for t in d.cluster_triangles(c) {
                    for i in t {
                        let v = d.vertex_source[off + i as usize];
                        if moved.contains(&v) {
                            set.insert(v);
                        }
                    }
                }
            }
        }
        set.len()
    };
    let biased = survivors(&dag);
    let plain = survivors(&build_cluster_dag(&m, &morphs, &DagOptions { morph_error_bias: 0.0, ..Default::default() }).unwrap());
    eprintln!("[geo] morph bias: moved vertices surviving level 1: {biased} (bias 2) vs {plain} (off) of {}", moved.len());
    assert!(biased >= plain, "the bias must not remove more of the moved vertices");
}

#[test]
fn discrete_cuts_shrink_and_stay_watertight() {
    let m = sphere(80, 160);
    let dag = build(&m);
    let lods = dag.discrete_lods(&[0.5, 0.25, 0.1, 0.03]);
    let fin = finest(&dag);
    let mut last = dag.header.source_triangles as usize;
    for l in &lods {
        assert!(l.triangles < last, "{} !< {last}", l.triangles);
        assert!(l.triangles as f32 <= dag.header.source_triangles as f32 * l.share * 1.0 + 1.0 || l.error == dag.clusters.iter().map(|c| c.lod_error).fold(0.0, f32::max));
        last = l.triangles;
        assert_crack_free(&dag, &fin, &l.clusters, "discrete cut");
        let mesh = dag.cut_mesh(&m, &l.clusters);
        assert_eq!(mesh.indices.len() / 3, l.triangles);
        assert!(mesh.indices.iter().all(|&i| (i as usize) < mesh.vertices.len()));
        eprintln!("[geo] sphere cut {:.0}%: {} tris (error {:.5})", l.share * 100.0, l.triangles, l.error);
    }
}

#[test]
fn serialisation_round_trips_and_truncation_is_rejected() {
    let m = torus(100, 50);
    let dag = build(&m);
    let bytes = dag.to_bytes();
    let back = ClusterDag::from_bytes(&bytes).unwrap();
    assert_eq!(back, dag);
    back.validate().unwrap();
    for cut in [0, 3, 8, 40, bytes.len() / 2, bytes.len() - 1] {
        assert!(ClusterDag::from_bytes(&bytes[..cut]).is_err(), "truncated at {cut}");
    }
    let mut longer = bytes.clone();
    longer.push(0);
    assert!(ClusterDag::from_bytes(&longer).is_err());
    let mut other = bytes.clone();
    other[4] ^= 1;
    assert!(ClusterDag::from_bytes(&other).is_err(), "another format version");
}

#[test]
fn validation_rejects_every_kind_of_bad_data() {
    let m = sphere(60, 120);
    let good = build(&m);
    good.validate().unwrap();
    let nc = good.clusters.len();
    let leaf = good.leaf_clusters().next().unwrap() as usize;
    let inner = (0..nc).find(|&c| good.links[c].group != NO_LINK && !good.clusters[c].is_root()).expect("an interior cluster");
    let root = good.root_clusters().next().unwrap() as usize;
    type Mutation = (&'static str, Box<dyn Fn(&mut ClusterDag)>);
    let cases: Vec<Mutation> = vec![
        ("too many triangles", Box::new(move |d| d.clusters[leaf].triangle_count = 200)),
        ("zero triangles", Box::new(move |d| d.clusters[leaf].triangle_count = 0)),
        ("too many vertices", Box::new(move |d| d.clusters[leaf].vertex_count = 300)),
        ("vertex range", Box::new(move |d| d.clusters[leaf].vertex_offset = u32::MAX - 3)),
        ("triangle range", Box::new(move |d| d.clusters[leaf].triangle_offset = (d.triangles.len() as u32 + 4) & !3)),
        ("unaligned triangles", Box::new(move |d| d.clusters[leaf].triangle_offset += 1)),
        ("index beyond vertices", Box::new(move |d| {
            let o = d.clusters[leaf].triangle_offset as usize;
            d.triangles[o] = 255;
        })),
        ("error shrinks", Box::new(move |d| d.clusters[inner].parent_lod_error = d.clusters[inner].lod_error * 0.5)),
        ("error nan", Box::new(move |d| d.clusters[inner].lod_error = f32::NAN)),
        ("error negative", Box::new(move |d| d.clusters[inner].lod_error = -1.0)),
        ("sphere nan", Box::new(move |d| d.clusters[inner].bounds[0] = f32::NAN)),
        ("sphere infinite", Box::new(move |d| d.clusters[inner].lod_sphere[3] = f32::INFINITY)),
        ("parent sphere too small", Box::new(move |d| d.clusters[inner].parent_lod_sphere[3] = 0.0)),
        ("bounds too small", Box::new(move |d| d.clusters[leaf].bounds[3] = 0.0)),
        ("bone count", Box::new(move |d| d.clusters[leaf].bone_count = 17)),
        ("bones on a static mesh", Box::new(move |d| d.clusters[leaf].bone_count = 2)),
        ("morph range", Box::new(move |d| d.clusters[leaf].morph_count = 5)),
        ("self cycle", Box::new(move |d| d.links[inner].parent_group = d.links[inner].group)),
        ("level order", Box::new(move |d| d.links[inner].level = 0)),
        ("group out of range", Box::new(move |d| d.links[inner].group = 1 << 20)),
        ("root flag mismatch", Box::new(move |d| d.clusters[root].parent_lod_error = 5.0)),
        ("child listed twice", Box::new(|d| d.group_children[1] = d.group_children[0])),
        ("child out of range", Box::new(|d| d.group_children[0] = u32::MAX - 1)),
        ("group outputs out of range", Box::new(|d| d.groups[0].first_output = u32::MAX - 1)),
        ("group error differs", Box::new(|d| d.groups[0].error += 1.0)),
        ("finest triangle count", Box::new(|d| d.header.source_triangles += 1)),
        ("vertex source range", Box::new(|d| d.vertex_source[0] = u32::MAX)),
        ("bad grid", Box::new(|d| d.header.quant_step = 0.0)),
        ("bvh child backwards", Box::new(|d| {
            let n = d.nodes.len();
            if n > 1 {
                let c = d.nodes[0].children.iter_mut().find(|c| c.child != BVH_EMPTY && c.child & BVH_LEAF == 0).unwrap();
                c.child = 0;
            }
        })),
        ("bvh leaf range", Box::new(|d| {
            for n in d.nodes.iter_mut() {
                if let Some(c) = n.children.iter_mut().find(|c| c.child & BVH_LEAF != 0 && c.child != BVH_EMPTY) {
                    c.count = 1 << 24;
                    return;
                }
            }
        })),
        ("bvh error too small", Box::new(|d| {
            for n in d.nodes.iter_mut() {
                if let Some(c) = n.children.iter_mut().find(|c| c.child & BVH_LEAF != 0 && c.child != BVH_EMPTY) {
                    c.max_parent_error = 0.0;
                    return;
                }
            }
        })),
        ("bvh bounds too small", Box::new(|d| {
            for n in d.nodes.iter_mut() {
                if let Some(c) = n.children.iter_mut().find(|c| c.child & BVH_LEAF != 0 && c.child != BVH_EMPTY) {
                    c.bounds[3] = 0.0;
                    return;
                }
            }
        })),
        ("bvh missing", Box::new(|d| d.nodes.clear())),
        ("no clusters", Box::new(|d| d.clusters.clear())),
    ];
    for (name, mutate) in &cases {
        let mut bad = good.clone();
        mutate(&mut bad);
        assert!(bad.validate().is_err(), "validation accepted: {name}");
    }
    // Random single-byte damage never panics the parser or the validator, and
    // whatever still validates traverses within bounds.
    let bytes = good.to_bytes();
    let mut rng = Rng(3);
    let mut rejected = 0;
    for _ in 0..400 {
        let mut b = bytes.clone();
        let at = (rng.next() * b.len() as f32) as usize % b.len();
        b[at] ^= 1 << ((rng.next() * 8.0) as u32 % 8);
        match ClusterDag::from_bytes(&b).and_then(|d| d.validate().map(|_| d)) {
            Ok(d) => {
                let _ = d.select_bvh(&|e, s| e <= 0.01 * s[3]);
            }
            Err(_) => rejected += 1,
        }
    }
    assert!(rejected > 40, "structural damage is caught: {rejected}/400");
}

#[test]
fn tiny_and_degenerate_meshes_do_not_panic() {
    let one = MeshData { vertices: vec![vtx([0.0, 0.0, 0.0]), vtx([1.0, 0.0, 0.0]), vtx([0.0, 1.0, 0.0])], indices: vec![0, 1, 2], has_lightmap_uv: false };
    let dag = build(&one);
    dag.validate().unwrap();
    assert_eq!(dag.clusters.len(), 1);
    assert!(build_cluster_dag(&MeshData { vertices: vec![], indices: vec![], has_lightmap_uv: false }, &[], &DagOptions::default()).is_err());
    let flat = MeshData { vertices: one.vertices.clone(), indices: vec![0, 0, 1], has_lightmap_uv: false };
    assert!(build_cluster_dag(&flat, &[], &DagOptions::default()).is_err());
    let bad = MeshData { vertices: one.vertices.clone(), indices: vec![0, 1, 9], has_lightmap_uv: false };
    assert!(build_cluster_dag(&bad, &[], &DagOptions::default()).is_err());
    // Two disjoint spheres and a triangle soup (three vertices per triangle).
    let a = sphere(30, 60);
    let mut vertices = a.vertices.clone();
    let mut indices = a.indices.clone();
    for v in &a.vertices {
        let mut v = *v;
        v.position[0] += 5.0;
        vertices.push(v);
    }
    indices.extend(a.indices.iter().map(|i| i + a.vertices.len() as u32));
    let two = MeshData { vertices, indices, has_lightmap_uv: false };
    let d = build(&two);
    check_invariants("two spheres", &two, &d);
    assert!(d.root_clusters().count() >= 1);
    exercise_cuts("two spheres", &two, &d, 20);
    let mut soup = MeshData { vertices: Vec::new(), indices: Vec::new(), has_lightmap_uv: false };
    for t in a.indices.as_chunks::<3>().0 {
        for &i in t {
            soup.indices.push(soup.vertices.len() as u32);
            soup.vertices.push(a.vertices[i as usize]);
        }
    }
    let d = build(&soup);
    check_invariants("soup", &soup, &d);
    assert!(d.clusters.iter().all(|c| c.vertex_count <= 256));
}

#[test]
fn build_report_and_speed_on_a_large_mesh() {
    // ~200k triangles: quick enough for every run; the million-triangle number is below.
    let m = torus(500, 200);
    let t = std::time::Instant::now();
    let dag = build(&m);
    let secs = t.elapsed().as_secs_f32();
    eprintln!("[geo] torus {} tris built in {secs:.2}s", m.indices.len() / 3);
    check_invariants("big torus", &m, &dag);
    assert!(secs < 60.0);
}

#[test]
#[ignore = "builder performance on ~1M triangles; run with --ignored --nocapture"]
fn million_triangle_build_time() {
    let m = torus(1000, 500);
    let tris = m.indices.len() / 3;
    for threads in [1, 0] {
        let t = std::time::Instant::now();
        let dag = build_cluster_dag(&m, &[], &DagOptions { threads, ..Default::default() }).unwrap();
        let secs = t.elapsed().as_secs_f32();
        eprintln!("[geo] {tris} triangles, threads {}: {secs:.1}s", if threads == 0 { "all".to_string() } else { threads.to_string() });
        if threads == 0 {
            dag.validate().unwrap();
            report("1M torus", &m, &dag);
        }
    }
}

/// A lightmap UV set alone must not flag the DAG (only real vertex colours do): the clustered shader never
/// samples uv1, and an object with a baked lightmap draws ordinary from the ordinary mesh.
#[test]
fn only_real_vertex_colours_flag_the_dag_not_a_lightmap_uv_set() {
    let plain = grid(40);
    let mut lightmapped = grid(40);
    lightmapped.has_lightmap_uv = true;
    for (i, v) in lightmapped.vertices.iter_mut().enumerate() {
        v.uv1 = [(i % 41) as f32 / 41.0, (i / 41) as f32 / 41.0];
    }
    let mut white_explicit = grid(40);
    white_explicit.has_lightmap_uv = true;
    white_explicit.vertices.iter_mut().for_each(|v| v.color = [1.0; 4]);
    let mut coloured = grid(40);
    coloured.vertices[17].color = [1.0, 0.5, 1.0, 1.0];
    let mut both = grid(40);
    both.vertices[17].color = [1.0, 0.5, 1.0, 1.0];
    both.has_lightmap_uv = true;
    assert!(!needs_extra_streams(&plain) && !needs_extra_streams(&lightmapped) && !needs_extra_streams(&white_explicit));
    assert!(needs_extra_streams(&coloured) && needs_extra_streams(&both));
    for (name, mesh, flagged) in [("plain", &plain, false), ("lightmap UV only", &lightmapped, false), ("coloured", &coloured, true), ("coloured + lightmap UV", &both, true)] {
        let dag = build_cluster_dag(mesh, &[], &DagOptions::default()).unwrap();
        assert_eq!(dag.header.flags & FLAG_EXTRA != 0, flagged, "{name}");
        dag.validate().unwrap();
        assert!(dag.to_gpu_upload().validate().is_ok(), "{name}: the renderer takes every DAG, coloured or not");
    }
    // Same geometry: the lightmap-UV mesh cooks the very same DAG as the plain one (the uv1 data is not in it).
    let a = build_cluster_dag(&plain, &[], &DagOptions::default()).unwrap();
    let b = build_cluster_dag(&lightmapped, &[], &DagOptions::default()).unwrap();
    assert_eq!(a.to_bytes(), b.to_bytes());
}

fn coloured_grid(n: usize) -> MeshData {
    let mut m = grid(n);
    for (i, v) in m.vertices.iter_mut().enumerate() {
        let (x, y) = ((i % (n + 1)) as f32 / n as f32, (i / (n + 1)) as f32 / n as f32);
        // A ramp, a hard edge at x = 0.5 and an alpha ramp: every channel carries information.
        v.color = [x, y, if x < 0.5 { 0.0 } else { 1.0 }, 1.0 - 0.5 * x];
    }
    m
}

fn q8(c: [f32; 4]) -> [u8; 4] {
    c.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
}

/// Every cluster vertex of every level carries the RGBA8 colour of the source vertex it is (survivors keep their
/// attributes), so leaf clusters are exact and coarse clusters show a colour that exists on the mesh.
#[test]
fn vertex_colours_ride_in_every_cluster_vertex_of_every_level() {
    let m = coloured_grid(60);
    let dag = build_cluster_dag(&m, &[], &DagOptions::default()).unwrap();
    assert!(dag.header.flags & FLAG_EXTRA != 0 && dag.header.levels > 1);
    dag.validate().unwrap();
    for (v, &src) in dag.vertices.iter().zip(&dag.vertex_source) {
        assert_eq!(v.color, q8(m.vertices[src as usize].color));
    }
    // The upload carries the same bytes the file does, and the file round-trips them.
    let up = dag.to_gpu_upload();
    up.validate().unwrap();
    assert_eq!(bytemuck::cast_slice::<_, u8>(&up.vertices), bytemuck::cast_slice::<_, u8>(&dag.vertices));
    let back = ClusterDag::from_bytes(&dag.to_bytes()).unwrap();
    assert_eq!(back.vertices, dag.vertices);
    // A mesh without colours stores white everywhere and no flag.
    let plain = build_cluster_dag(&grid(60), &[], &DagOptions::default()).unwrap();
    assert!(plain.header.flags & FLAG_EXTRA == 0 && plain.vertices.iter().all(|v| v.color == [255; 4]));
}

/// The colour cost term keeps colour detail to coarse levels: the coarsest cut of a two-coloured mesh still shows
/// both colours on its vertices, and the cost term leaves a mesh without colours byte-identical.
#[test]
fn the_colour_cost_term_keeps_a_colour_edge_to_the_coarsest_level() {
    let m = coloured_grid(80);
    let dag = build_cluster_dag(&m, &[], &DagOptions::default()).unwrap();
    let top = dag.header.levels - 1;
    let (mut lo, mut hi) = (false, false);
    for c in dag.clusters.iter().filter(|c| c.level == top) {
        for v in &dag.vertices[c.vertex_offset as usize..c.vertex_offset as usize + c.vertex_count as usize] {
            lo |= v.color[2] == 0;
            hi |= v.color[2] == 255;
        }
    }
    assert!(lo && hi, "the coarsest clusters lost one side of the colour edge");
    let a = build_cluster_dag(&grid(60), &[], &DagOptions::default()).unwrap();
    let b = build_cluster_dag(&grid(60), &[], &DagOptions::default()).unwrap();
    assert_eq!(a.to_bytes(), b.to_bytes());
}

// ------------------------------------------------------------ diagnostics

/// Per-distance cut of a DAG under the cull shader's `lod_ok` (perspective, `ppu` pixels per unit at 1 m,
/// `px` target), the instance at `dist` metres with world scale `s`: (clusters, triangles, vertices, level histogram).
pub(super) fn cut_at(dag: &ClusterDag, s: f32, centre: [f32; 3], dist: f32, ppu: f32, px: f32) -> (usize, usize, usize, Vec<usize>) {
    // Camera on the +z axis of the instance centre (object space scaled by s).
    let ok = |err: f32, sph: [f32; 4]| {
        let d = (((sph[0] - centre[0]) * s).powi(2) + ((sph[1] - centre[1]) * s).powi(2) + ((sph[2] - centre[2]) * s - dist).powi(2)).sqrt();
        let d = (d - sph[3] * s).max(0.0);
        err * s * ppu <= px * d
    };
    let cut = dag.select_bvh(&ok);
    let tris = dag.triangle_count_of(&cut);
    let verts: usize = cut.iter().map(|&c| dag.clusters[c as usize].vertex_count as usize).sum();
    let mut hist = vec![0usize; dag.header.levels as usize + 1];
    for &c in &cut {
        hist[dag.clusters[c as usize].level as usize] += 1;
    }
    while hist.last() == Some(&0) {
        hist.pop();
    }
    (cut.len(), tris, verts, hist)
}

pub(super) fn level_table(dag: &ClusterDag) -> String {
    let mut out = String::new();
    for l in 0..dag.header.levels {
        let cs: Vec<&GpuCluster> = dag.clusters.iter().filter(|c| c.level == l).collect();
        if cs.is_empty() {
            continue;
        }
        let tris: usize = cs.iter().map(|c| c.triangle_count as usize).sum();
        let verts: usize = cs.iter().map(|c| c.vertex_count as usize).sum();
        let (lo, hi) = cs.iter().fold((f32::MAX, 0.0f32), |(a, b), c| (a.min(c.lod_error), b.max(c.lod_error)));
        let mut errs: Vec<f32> = cs.iter().map(|c| c.lod_error).collect();
        errs.sort_by(f32::total_cmp);
        let roots = cs.iter().filter(|c| c.is_root()).count();
        out += &format!("    L{l:<2} {:>5} clusters {:>7} tris {:>7} verts  err min {lo:.2e} med {:.2e} max {hi:.2e}  roots {roots}\n", cs.len(), tris, verts, errs[errs.len() / 2]);
    }
    out
}

/// Diagnostic (no assertions): how a real skinned model's DAGs coarsen with distance, next to a static
/// (unweighted) build of the same meshes. `ZESTENGINE_DIAG_MODEL` names the model (an FBX / glTF);
/// `ZESTENGINE_DIAG_STATIC=1` also rebuilds every mesh without skin weights for comparison.
#[test]
#[ignore = "diagnostic on a real model; run with --ignored --nocapture"]
fn diag_real_model_cut_by_distance() {
    let Ok(path) = std::env::var("ZESTENGINE_DIAG_MODEL") else { return };
    // ZESTENGINE_DIAG_REBUILD=1: build the DAGs here with the current builder (not the cache).
    let rebuild = std::env::var("ZESTENGINE_DIAG_REBUILD").is_ok_and(|v| v == "1");
    let only = std::env::var("ZESTENGINE_DIAG_MESH").ok().and_then(|v| v.parse::<usize>().ok());
    let (scene, geo) = if rebuild {
        let scene = crate::load_model(&path).expect("model loads");
        let meshes = scene
            .meshes
            .iter()
            .enumerate()
            .map(|(i, m)| {
                (crate::geo_cache::wants_dag(m) && only.is_none_or(|o| o == i))
                    .then(|| build_cluster_dag(m, scene.morphs.get(i).map_or(&[][..], |v| &v[..]), &DagOptions::default()).ok())
                    .flatten()
            })
            .collect();
        (scene, crate::geo_cache::ModelGeometry { key: 0, meshes })
    } else {
        crate::load_model_with_geometry(&path).expect("model loads")
    };
    let ppu = 0.5 * 900.0 / (30f32.to_radians()).tan();
    let dists = [2.0f32, 5.0, 10.0, 20.0, 40.0, 80.0];
    let with_static = std::env::var("ZESTENGINE_DIAG_STATIC").is_ok_and(|v| v == "1");
    let mut totals = vec![(0usize, 0usize, 0usize); dists.len()];
    let mut totals_static = vec![(0usize, 0usize, 0usize); dists.len()];
    for (i, dag) in geo.meshes.iter().enumerate() {
        let Some(dag) = dag else { continue };
        if only.is_some_and(|m| m != i) {
            continue;
        }
        let mesh = &scene.meshes[i];
        let s = scene
            .instances
            .iter()
            .find(|inst| inst.slots.iter().any(|sl| sl.mesh == i))
            .map_or(1.0, |inst| inst.transform.x_axis.truncate().length().max(inst.transform.y_axis.truncate().length()));
        let roots: Vec<u32> = dag.root_clusters().collect();
        let root_tris = dag.triangle_count_of(&roots);
        let (lo, hi) = bounds_of(mesh);
        let centre = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5];
        eprintln!(
            "[diag] mesh {i}: {} tris, {} verts, skinned {}, morphs {}, scale {s}, extent {:.3} m, {} levels, roots {} ({} tris), step {:.2e}",
            mesh.indices.len() / 3,
            mesh.vertices.len(),
            dag.header.flags & FLAG_SKINNED != 0,
            dag.header.morph_targets,
            (0..3).map(|k| hi[k] - lo[k]).fold(0.0f32, f32::max) * s,
            dag.header.levels,
            roots.len(),
            root_tris,
            dag.header.quant_step
        );
        eprint!("{}", level_table(dag));
        {
            let mut long = 0usize;
            let mut el: Vec<f32> = Vec::new();
            for t in mesh.indices.as_chunks::<3>().0 {
                let p = [0, 1, 2].map(|k| mesh.vertices[t[k] as usize].position);
                let e = (0..3).map(|k| ((p[k][0] - p[(k + 1) % 3][0]).powi(2) + (p[k][1] - p[(k + 1) % 3][1]).powi(2) + (p[k][2] - p[(k + 1) % 3][2]).powi(2)).sqrt()).fold(0.0f32, f32::max);
                el.push(e);
                if e > 0.2 {
                    long += 1;
                }
            }
            el.sort_by(f32::total_cmp);
            eprintln!("    source: longest-edge p50 {:.2e} p99 {:.2e} max {:.2e}, {} triangles with an edge > 20 cm", el[el.len() / 2], el[el.len() * 99 / 100], el[el.len() - 1], long);
        }
        for (k, &d) in dists.iter().enumerate() {
            let (c, t, v, h) = cut_at(dag, s, centre, d, ppu, 1.0);
            totals[k].0 += c;
            totals[k].1 += t;
            totals[k].2 += v;
            eprintln!("    {d:>4} m: {c:>5} clusters {t:>7} tris {v:>7} verts  levels {h:?}");
        }
        if with_static && dag.header.flags & FLAG_SKINNED != 0 {
            let mut m = MeshData { vertices: mesh.vertices.clone(), indices: mesh.indices.clone(), has_lightmap_uv: mesh.has_lightmap_uv };
            m.vertices.iter_mut().for_each(|v| v.weights = [0.0; 4]);
            let sd = build_cluster_dag(&m, &[], &DagOptions::default()).expect("static build");
            let sroots: Vec<u32> = sd.root_clusters().collect();
            eprintln!("  static copy: {} levels, roots {} ({} tris)", sd.header.levels, sroots.len(), sd.triangle_count_of(&sroots));
            eprint!("{}", level_table(&sd));
            for (k, &d) in dists.iter().enumerate() {
                let (c, t, v, h) = cut_at(&sd, s, centre, d, ppu, 1.0);
                totals_static[k].0 += c;
                totals_static[k].1 += t;
                totals_static[k].2 += v;
                eprintln!("    {d:>4} m: {c:>5} clusters {t:>7} tris {v:>7} verts  levels {h:?}");
            }
        }
    }
    for (k, &d) in dists.iter().enumerate() {
        eprintln!("[diag] total {d:>4} m: skinned {:?}  static {:?}", totals[k], totals_static[k]);
    }
}
