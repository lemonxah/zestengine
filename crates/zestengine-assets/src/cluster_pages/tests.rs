//! Page splitter, dependency closure, the streaming cut, the paged file
//! format and its rejection of bad data. CPU only.

use std::collections::{HashMap, HashSet};

use zestengine_render::{MeshData, Vertex};

use super::*;
use crate::cluster_dag::{DagOptions, FLAG_SKINNED, build_cluster_dag};

fn torus(nu: usize, nv: usize) -> MeshData {
    let mut vertices = Vec::new();
    for i in 0..nu {
        let u = std::f32::consts::TAU * i as f32 / nu as f32;
        for j in 0..nv {
            let v = std::f32::consts::TAU * j as f32 / nv as f32;
            let n = [v.cos() * u.cos(), v.sin(), v.cos() * u.sin()];
            let p = [(3.0 + v.cos()) * u.cos(), v.sin(), (3.0 + v.cos()) * u.sin()];
            vertices.push(Vertex { position: p, normal: n, uv: [i as f32 / nu as f32, j as f32 / nv as f32], ..Default::default() });
        }
    }
    let mut indices = Vec::new();
    for i in 0..nu {
        for j in 0..nv {
            let a = (i * nv + j) as u32;
            let b = (((i + 1) % nu) * nv + j) as u32;
            let c = (i * nv + (j + 1) % nv) as u32;
            let d = (((i + 1) % nu) * nv + (j + 1) % nv) as u32;
            indices.extend([a, c, b, b, c, d]);
        }
    }
    MeshData { vertices, indices, has_lightmap_uv: false }
}

fn dag(nu: usize, nv: usize) -> ClusterDag {
    build_cluster_dag(&torus(nu, nv), &[], &DagOptions { threads: 2, ..Default::default() }).expect("torus builds")
}

/// Small pages so a test mesh has many of them.
fn small() -> PageOptions {
    PageOptions { page_bytes: 48 * 1024, resident_bytes: 96 * 1024, min_stream_bytes: 1 }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn unit(&mut self) -> f32 {
        (self.next() % 100_000) as f32 / 100_000.0
    }
}

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

/// The finest level's directed edges and open border points (empty for a closed mesh).
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

/// The cut is a closed surface (every directed edge has its reverse, except
/// along the finest level's border), nothing is drawn twice, and every finest
/// cluster is covered by a drawn cluster or by every output of its consuming group.
fn assert_watertight(dag: &ClusterDag, fin: &Finest, cut: &[u32], what: &str) {
    assert!(!cut.is_empty(), "{what}: empty cut");
    let edges = directed_edges(dag, cut);
    for (&(a, b), &n) in &edges {
        let allowed = fin.edges.get(&(a, b)).copied().unwrap_or(1).max(1);
        assert!(n <= allowed, "{what}: edge covered {n} times (overlapping clusters)");
        if !edges.contains_key(&(b, a)) {
            assert!(fin.border.contains(&a) && fin.border.contains(&b), "{what}: crack: an interior edge has no neighbour");
        }
    }
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
        assert!(!above[c as usize], "{what}: cluster {c} is drawn below an already covered group (overlap)");
    }
    for l in dag.leaf_clusters() {
        assert!(covered[l as usize], "{what}: finest cluster {l} is not covered (a hole)");
    }
}

#[allow(clippy::type_complexity)] // two view closures over one camera
fn view(cam: [f32; 3], thr: f32) -> (impl Fn(f32, [f32; 4]) -> bool, impl Fn(f32, [f32; 4]) -> f32) {
    let dist = move |s: [f32; 4]| (((s[0] - cam[0]).powi(2) + (s[1] - cam[1]).powi(2) + (s[2] - cam[2]).powi(2)).sqrt() - s[3]).max(0.0);
    (move |e: f32, s: [f32; 4]| e <= thr * dist(s), move |e: f32, s: [f32; 4]| e / (thr * dist(s)).max(1e-12))
}

#[test]
fn every_cluster_is_in_exactly_one_page_and_the_split_is_deterministic() {
    let d = dag(220, 110);
    let opts = small();
    let plan = PagePlan::split(&d, &opts);
    assert!(plan.pages.len() > 6, "expected a real split, got {} pages", plan.pages.len());
    let mut seen = vec![0u32; d.clusters.len()];
    let (mut verts, mut tris) = (0usize, 0usize);
    for (i, pg) in plan.pages.iter().enumerate() {
        assert!(!pg.clusters.is_empty());
        assert!(pg.clusters.windows(2).all(|w| w[0] < w[1]), "page {i}: clusters not ascending");
        for &c in &pg.clusters {
            seen[c as usize] += 1;
            assert_eq!(plan.cluster_page[c as usize], i as u32);
        }
        verts += pg.vertices as usize;
        tris += pg.tri_bytes as usize;
        // Pages never mix levels, except the root page that holds the top ones.
        if i > 0 {
            assert!(pg.clusters.iter().all(|&c| d.links[c as usize].level == pg.level), "page {i} mixes levels");
        }
    }
    assert!(seen.iter().all(|&n| n == 1), "a cluster is in no page or in two");
    assert_eq!(verts, d.vertices.len(), "pages do not carry every vertex");
    assert_eq!(tris, d.triangles.len(), "pages do not carry every triangle byte");
    assert_eq!(plan, PagePlan::split(&d, &opts), "the split is not deterministic");
    // Page sizes: packed to the target unless one input set alone is bigger.
    let st = plan.stats();
    assert!(st.root_bytes <= opts.resident_bytes.max(plan.pages[0].bytes()), "root page {} over its budget", st.root_bytes);
    assert!(st.largest_page <= opts.page_bytes * 2, "a page is {} bytes (target {})", st.largest_page, opts.page_bytes);
    eprintln!("[pages] {} clusters -> {} pages: root {} B, streamed {} B, mean {} B, largest {} B, max deps {}", d.clusters.len(), st.pages, st.root_bytes, st.streamed_bytes, st.mean_page, st.largest_page, st.max_deps);
}

#[test]
fn dependency_closure_holds() {
    let d = dag(220, 110);
    let plan = PagePlan::split(&d, &small());
    assert!(plan.pages[0].deps.is_empty(), "the root page needs nothing");
    for (i, pg) in plan.pages.iter().enumerate() {
        // R2: the outputs of the group that consumes a cluster are on the page or a listed dependency.
        for &c in &pg.clusters {
            let pgrp = d.links[c as usize].parent_group;
            if pgrp == NO_LINK {
                assert_eq!(i, 0, "root cluster {c} outside the root page");
                continue;
            }
            let g = d.groups[pgrp as usize];
            for o in g.first_output..g.first_output + g.output_count {
                let q = plan.cluster_page[o as usize];
                assert!(q as usize == i || pg.deps.contains(&q), "page {i}: parent {o} on page {q} is not a dependency");
            }
        }
        // Dependencies point up (the closure terminates at the root page).
        for &dep in &pg.deps {
            assert!(dep == 0 || plan.pages[dep as usize].level > pg.level);
        }
        // R: the finer children of a cluster sit on one page.
        for &c in &pg.clusters {
            let cp = plan.child_page[c as usize];
            if d.links[c as usize].group == NO_LINK {
                assert_eq!(cp, NO_LINK);
            } else {
                let g = d.groups[d.links[c as usize].group as usize];
                for &k in &d.group_children[g.first_child as usize..][..g.child_count as usize] {
                    assert_eq!(plan.cluster_page[k as usize], cp);
                }
            }
        }
    }
    // The closure of any page ends at the root and lists parents first.
    let layout = PagedLayout { dag: &d, plan: &plan };
    for p in 0..plan.pages.len() as u32 {
        let cl = layout.closure(p);
        assert_eq!(*cl.last().unwrap(), p);
        assert!(cl.contains(&0));
        let pos = |q: u32| cl.iter().position(|&x| x == q).unwrap();
        for &q in &cl {
            for &dep in &plan.pages[q as usize].deps {
                assert!(pos(dep) < pos(q), "closure of {p}: {dep} must come before {q}");
            }
        }
    }
}

#[test]
fn small_dags_stay_one_page_and_a_single_plan_is_valid_for_any_dag() {
    let d = dag(40, 30);
    assert!(PagePlan::split(&d, &PageOptions::default()).is_single(), "a small mesh must not be split");
    let (sk, _) = skinned_dag(40, 24);
    assert!(PagePlan::split(&sk, &PageOptions::default()).is_single(), "a small skinned mesh must not be split");
    // A single plan is valid for any DAG and puts everything on the root page.
    let big = dag(220, 110);
    let one = PagePlan::single(&big);
    assert_eq!(one.pages.len(), 1);
    assert_eq!(one.pages[0].clusters.len(), big.clusters.len());
    assert!(one.cluster_page.iter().all(|&p| p == 0));
    // ... however large (a one-page plan has no page limit; it is the whole mesh resident).
    let one = PagePlan::single(&sk);
    assert_eq!(one.pages.len(), 1);
    assert!(one.morph_reach.iter().any(|&r| r > 0.0));
}

/// A skinned tube (bones along its length, <= 16 per cluster) with three sparse blendshapes.
fn skinned_dag(rings: usize, sides: usize) -> (ClusterDag, Vec<crate::MorphTarget>) {
    let joints = 40usize;
    let mut vertices = Vec::new();
    for r in 0..rings {
        let y = 2.0 * r as f32 / (rings - 1) as f32;
        let radius = 0.12 + 0.05 * (y * 3.0).sin();
        let f = y / 2.0 * (joints - 1) as f32;
        let (j0, w1) = (f.floor() as usize, f.fract());
        let j1 = (j0 + 1).min(joints - 1);
        for s in 0..sides {
            let a = std::f32::consts::TAU * s as f32 / sides as f32;
            vertices.push(Vertex {
                position: [radius * a.cos(), y, radius * a.sin()],
                normal: [a.cos(), 0.0, a.sin()],
                uv: [s as f32 / sides as f32, y * 0.5],
                joints: [j0 as u32, j1 as u32, 0, 0],
                weights: if j0 == j1 { [1.0, 0.0, 0.0, 0.0] } else { [1.0 - w1, w1, 0.0, 0.0] },
                ..Default::default()
            });
        }
    }
    let mut indices = Vec::new();
    for r in 0..rings - 1 {
        for s in 0..sides {
            let at = |r: usize, s: usize| (r * sides + s % sides) as u32;
            indices.extend([at(r, s), at(r + 1, s), at(r, s + 1), at(r, s + 1), at(r + 1, s), at(r + 1, s + 1)]);
        }
    }
    let mesh = MeshData { vertices, indices, has_lightmap_uv: false };
    let mut rng = Rng(11);
    let mut morphs = Vec::new();
    for t in 0..3 {
        let mut deltas = Vec::new();
        for v in 0..mesh.vertices.len() as u32 {
            if mesh.vertices[v as usize].position[1] > 0.4 && !rng.next().is_multiple_of(3) {
                deltas.push((v, [rng.unit() * 0.02, rng.unit() * 0.01, 0.0], [0.0, rng.unit() * 0.1, 0.0]));
            }
        }
        morphs.push(crate::MorphTarget { name: format!("shape{t}"), deltas });
    }
    let dag = build_cluster_dag(&mesh, &morphs, &DagOptions { threads: 2, ..Default::default() }).expect("skinned tube builds");
    (dag, morphs)
}

#[test]
fn skinned_and_blendshape_dags_split_and_carry_their_skin_data_in_the_pages() {
    let (d, _) = skinned_dag(260, 100);
    assert!(d.header.flags & FLAG_SKINNED != 0 && d.header.morph_targets == 3 && !d.morph_entries.is_empty());
    let plan = PagePlan::split(&d, &small());
    assert!(plan.pages.len() > 4, "a skinned mesh must be split into pages now, got {}", plan.pages.len());
    // Page sizes: vertices + triangle region + skin records + aux, summed over the clusters.
    let (mut skin, mut aux) = (0usize, 0usize);
    for pg in &plan.pages {
        let spans = page_spans(&d, pg);
        assert_eq!(spans.skin_bytes as usize, pg.vertices as usize * SKIN_BYTES);
        skin += spans.skin_bytes as usize;
        let tris: usize = pg.clusters.iter().map(|&c| cluster_payload(&d.clusters[c as usize]).1 as usize).sum();
        let aux_words: usize = pg.clusters.iter().map(|&c| cluster_aux_words(&d.clusters[c as usize]) as usize).sum();
        aux += aux_words * 4;
        assert_eq!(pg.tri_bytes as usize, spans.skin_bytes as usize + tris + aux_words * 4, "page triangle block size");
        // The aux records tile the end of the block.
        let last = spans.spans.last().unwrap();
        let end = last.aux_word as usize + cluster_aux_words(&d.clusters[*pg.clusters.last().unwrap() as usize]) as usize;
        assert_eq!(end * 4, pg.tri_bytes as usize);
    }
    eprintln!("[pages] skinned tube: {} pages, skin {} KB, aux (bones + blendshape entries) {} KB, vertices {} KB", plan.pages.len(), skin / 1024, aux / 1024, d.vertices.len() * 24 / 1024);
    assert_eq!(skin, d.vertex_skin.len() * SKIN_BYTES);
    // Cut and crack freedom do not depend on the payload: same dependency rules as static meshes.
    let fin = finest(&d);
    let layout = PagedLayout { dag: &d, plan: &plan };
    let mut rng = Rng(21);
    for k in 0..6 {
        let res = random_closed_set(&layout, &mut rng, 15 + 12 * k);
        let (ok, ratio) = view([0.4, 1.0, 2.0], 0.002 + 0.004 * k as f32);
        let cut = layout.select(&ok, &ratio, &res);
        assert_watertight(&d, &fin, &cut.clusters, "skinned partial residency");
    }
    assert_eq!(plan.morph_reach.len(), 3);
    assert!(plan.morph_reach.iter().all(|&r| r > 0.0), "{:?}", plan.morph_reach);
}

/// What the shaders do (cluster_common.glsl `page_skin_bytes`, cluster_skin.comp), on the CPU:
/// read a cluster's skin records, joint ids, blendshape entries and triangles from the page
/// block of the GPU cluster record. Must equal the original DAG's arrays.
#[test]
fn the_page_block_addressing_of_the_skin_stage_reproduces_the_original_arrays() {
    let (d, _) = skinned_dag(260, 100);
    let plan = PagePlan::split(&d, &small());
    assert!(plan.pages.len() > 4);
    let gpu = plan.gpu_clusters(&d);
    let mem = MemPageSource { dag: Arc::new(d.clone()), plan: Arc::new(plan.clone()) };
    for (p, info) in plan.pages.iter().enumerate() {
        let payload = mem.read_page(p as u32).unwrap();
        let block = &payload.triangles;
        let word = |i: usize| u32::from_le_bytes(block[i * 4..i * 4 + 4].try_into().unwrap());
        let skin_bytes = info.vertices as usize * 8;
        for &c in &info.clusters {
            let (g, o) = (&gpu[c as usize], &d.clusters[c as usize]);
            assert_eq!(g.morph_offset, p as u32);
            for t in 0..g.vertex_count as usize {
                // skin[blk + 2 * (vertex_offset + t)]
                let w = (g.vertex_offset as usize + t) * 2;
                let (x, y) = (word(w), word(w + 1));
                let want = d.vertex_skin[o.vertex_offset as usize + t];
                assert_eq!(x.to_le_bytes(), want.bones);
                assert_eq!(y.to_le_bytes(), want.weights);
            }
            // triangles at skin_bytes + triangle_offset
            let tb = &block[skin_bytes + g.triangle_offset as usize..][..g.triangle_count as usize * 3];
            assert_eq!(tb, &d.triangles[o.triangle_offset as usize..][..o.triangle_count as usize * 3]);
            // aux at word bone_offset: morph count, joint ids (two per word), morph entries
            let aux = g.bone_offset as usize;
            assert!(aux * 4 >= skin_bytes, "aux inside the skin/triangle area");
            assert_eq!(word(aux), o.morph_count);
            for k in 0..o.bone_count as usize {
                let joint = (word(aux + 1 + (k >> 1)) >> ((k & 1) * 16)) & 0xFFFF;
                assert_eq!(joint, u32::from(d.bone_ids[o.bone_offset as usize + k]), "cluster {c} bone slot {k}");
            }
            let mbase = aux + 1 + (o.bone_count as usize).div_ceil(2);
            for k in 0..o.morph_count as usize {
                let me = [word(mbase + 4 * k), word(mbase + 4 * k + 1), word(mbase + 4 * k + 2), word(mbase + 4 * k + 3)];
                let want = d.morph_entries[o.morph_offset as usize + k];
                assert_eq!(me[0] & 0xFFFF, u32::from(want.target));
                assert_eq!(me[0] >> 16, u32::from(want.vertex));
                assert_eq!(me[1], u32::from(want.d_pos[0]) | u32::from(want.d_pos[1]) << 16);
                assert_eq!(me[2], u32::from(want.d_pos[2]) | u32::from(want.d_normal[0]) << 16);
                assert_eq!(me[3], u32::from(want.d_normal[1]) | u32::from(want.d_normal[2]) << 16);
            }
        }
    }
}

#[test]
fn skinned_paged_files_round_trip_validate_and_reject_bad_pages() {
    let (d, _) = skinned_dag(260, 100);
    let plan = PagePlan::split(&d, &small());
    let bytes = encode_paged(&d, &plan);
    let (back, back_plan) = decode_paged(&bytes).expect("a skinned paged mesh decodes");
    assert_eq!(back, d, "the assembled skinned DAG differs from the original (skin stream, bone ids or blendshape entries)");
    assert_eq!(back_plan, plan, "plan (incl. the stored blendshape reach)");
    back.validate().expect("the assembled DAG validates");
    let file = PagedFile::from_bytes(bytes.clone().into()).unwrap();
    assert_eq!(file.plan.morph_reach, morph_reach_of(&d));
    let mem = MemPageSource { dag: Arc::new(d.clone()), plan: Arc::new(plan.clone()) };
    for p in 0..plan.pages.len() as u32 {
        assert_eq!(file.read_page(p).unwrap(), mem.read_page(p).unwrap(), "page {p}");
    }
    // Every flipped byte is caught by a checksum.
    let mut rng = Rng(5);
    for _ in 0..100 {
        let mut b = bytes.clone();
        let at = (rng.next() as usize) % b.len();
        b[at] ^= 1 << (rng.next() % 8);
        assert!(decode_paged(&b).is_err(), "flipped byte {at}");
    }
    // Bad page contents never reach the GPU: a joint slot past the bone list, a wrong aux count,
    // a blendshape entry on a vertex the cluster does not have.
    let page = (1..plan.pages.len() as u32).find(|&p| plan.pages[p as usize].clusters.iter().any(|&c| d.clusters[c as usize].morph_count > 0 && d.clusters[c as usize].bone_count > 0)).expect("a page with blendshapes and bones");
    let good = mem.read_page(page).unwrap();
    assert!(validate_payload(&d, &plan, page, &good).is_ok());
    let info = &plan.pages[page as usize];
    let spans = page_spans(&d, info);
    let (ci, &c) = info.clusters.iter().enumerate().find(|&(_, &c)| d.clusters[c as usize].morph_count > 0 && d.clusters[c as usize].bone_count > 0).unwrap();
    let cl = &d.clusters[c as usize];
    let mut bad = good.clone();
    let first_weighted = (0..cl.vertex_count as usize).find(|&t| d.vertex_skin[cl.vertex_offset as usize + t].weights[0] != 0).unwrap();
    bad.triangles[(spans.spans[ci].vertex as usize + first_weighted) * 8] = 200; // bone slot 0
    assert!(validate_payload(&d, &plan, page, &bad).is_err(), "a skin index past the bone list was accepted");
    let mut bad = good.clone();
    bad.triangles[spans.spans[ci].aux_word as usize * 4] ^= 1;
    assert!(validate_payload(&d, &plan, page, &bad).is_err(), "a wrong aux morph count was accepted");
    let mut bad = good.clone();
    let m0 = (spans.spans[ci].aux_word as usize + 1 + (cl.bone_count as usize).div_ceil(2)) * 4;
    bad.triangles[m0 + 2] = 255;
    bad.triangles[m0 + 3] = 255;
    assert!(validate_payload(&d, &plan, page, &bad).is_err(), "a blendshape entry beyond the cluster was accepted");
    let mut bad = good.clone();
    bad.triangles[m0] = 99;
    assert!(validate_payload(&d, &plan, page, &bad).is_err(), "a blendshape target beyond the mesh's was accepted");
}

#[cfg(feature = "client")]
#[test]
fn a_skinned_streamed_upload_validates_and_carries_the_blendshape_reach() {
    let (d, _) = skinned_dag(260, 100);
    let plan = PagePlan::split(&d, &small());
    let mem = MemPageSource { dag: Arc::new(d.clone()), plan: Arc::new(plan.clone()) };
    let up = plan.to_gpu_upload(&skeleton_of(&d), &mem.read_page(0).unwrap());
    up.validate().expect("the streamed skinned upload validates");
    assert!(up.skinned && up.skin.is_empty() && up.bones.is_empty() && up.morphs.is_empty());
    assert_eq!(up.morph_targets, 3);
    // The reach equals what the non-streamed upload derives from its morph entries.
    assert_eq!(up.morph_reach(), d.to_gpu_upload().morph_reach());
    // Skin data riding in a streamed upload is refused.
    let mut bad = up.clone();
    bad.bones = vec![0; 4];
    assert!(bad.validate().is_err());
    // A cluster whose aux record points outside its page is refused.
    let mut bad = up.clone();
    bad.clusters[0].bone_offset = u32::MAX / 8;
    assert!(bad.validate().is_err());
    let st = up.stream.as_ref().unwrap();
    eprintln!("[pages] skinned streamed upload: tri share of the streamed pages {:.2}", st.tri_share());
    assert!(st.tri_share() > 0.3, "a skinned mesh's pages are mostly triangle block");
}

fn random_closed_set(layout: &PagedLayout, rng: &mut Rng, density: u64) -> Vec<bool> {
    let n = layout.plan.pages.len();
    let mut res = vec![false; n];
    res[0] = true;
    for p in 1..n as u32 {
        if rng.next() % 100 < density {
            for q in layout.closure(p) {
                res[q as usize] = true;
            }
        }
    }
    res
}

#[test]
fn the_cut_stays_watertight_for_every_resident_set_closed_under_dependencies() {
    let d = dag(220, 110);
    let plan = PagePlan::split(&d, &small());
    let layout = PagedLayout { dag: &d, plan: &plan };
    let fin = finest(&d);
    let mut rng = Rng(7);
    let mut checked = 0;
    for density in [0u64, 5, 20, 50, 90, 100] {
        for _ in 0..12 {
            let res = random_closed_set(&layout, &mut rng, density);
            let cam = [rng.unit() * 20.0 - 10.0, rng.unit() * 10.0 - 5.0, rng.unit() * 20.0 - 10.0];
            let thr = 0.0005 + rng.unit() * 0.02;
            let (ok, ratio) = view(cam, thr);
            let cut = layout.select(&ok, &ratio, &res);
            assert!(!cut.clusters.is_empty());
            assert_watertight(&d, &fin, &cut.clusters, &format!("density {density}"));
            // Every drawn cluster is on a resident page; every request is for a missing page.
            assert!(cut.clusters.iter().all(|&c| res[plan.cluster_page[c as usize] as usize]));
            assert!(cut.requests.iter().all(|&(p, r)| !res[p as usize] && r >= 1.0));
            // A request is loadable: its closure is a valid resident set (parents first).
            for &(p, _) in &cut.requests {
                let mut next = res.clone();
                for q in layout.closure(p) {
                    next[q as usize] = true;
                }
                let after = layout.select(&ok, &ratio, &next);
                assert_watertight(&d, &fin, &after.clusters, "after serving a request");
            }
            checked += 1;
        }
    }
    assert!(checked >= 70);
}

#[test]
fn fully_resident_streaming_cut_equals_the_ordinary_cut() {
    let d = dag(220, 110);
    let plan = PagePlan::split(&d, &small());
    let layout = PagedLayout { dag: &d, plan: &plan };
    let all = vec![true; plan.pages.len()];
    let mut rng = Rng(11);
    for _ in 0..20 {
        let cam = [rng.unit() * 16.0 - 8.0, rng.unit() * 8.0 - 4.0, rng.unit() * 16.0 - 8.0];
        let (ok, ratio) = view(cam, 0.0005 + rng.unit() * 0.01);
        let cut = layout.select(&ok, &ratio, &all);
        assert_eq!(cut.clusters, d.select_linear(&ok));
        assert!(cut.requests.is_empty());
    }
}

#[test]
fn a_missing_page_is_requested_and_its_parent_stands_in_until_it_arrives() {
    let d = dag(220, 110);
    let plan = PagePlan::split(&d, &small());
    let layout = PagedLayout { dag: &d, plan: &plan };
    let fin = finest(&d);
    // Camera very close: the root alone is far too coarse.
    let (ok, ratio) = view([3.0, 0.5, 0.5], 0.0004);
    let mut res = vec![false; plan.pages.len()];
    res[0] = true;
    let mut last = layout.select(&ok, &ratio, &res);
    let coarse: usize = d.triangle_count_of(&last.clusters);
    assert!(!last.requests.is_empty(), "a coarse cut under a close camera must ask for pages");
    // Serve the requests (parents first) until nothing is missing: the triangle count only grows, no hole at any step.
    let mut tris = coarse;
    for _ in 0..40 {
        if last.requests.is_empty() {
            break;
        }
        for &(p, _) in &last.requests {
            for q in layout.closure(p) {
                res[q as usize] = true;
            }
        }
        last = layout.select(&ok, &ratio, &res);
        assert_watertight(&d, &fin, &last.clusters, "converging");
        let now = d.triangle_count_of(&last.clusters);
        assert!(now >= tris, "the cut got coarser while pages arrived");
        tris = now;
    }
    assert!(last.requests.is_empty(), "never converged");
    assert_eq!(last.clusters, d.select_linear(&ok), "converged cut differs from the fully resident one");
    assert!(tris > coarse);
}

#[test]
fn paged_file_round_trips_and_pages_read_back() {
    let d = dag(220, 110);
    let plan = PagePlan::split(&d, &small());
    let bytes = encode_paged(&d, &plan);
    assert!(is_paged(&bytes));
    let (back, back_plan) = decode_paged(&bytes).expect("decodes");
    assert_eq!(back, d, "the assembled DAG differs from the original");
    assert_eq!(back_plan, plan);
    back.validate().expect("the assembled DAG validates");
    let file = PagedFile::from_bytes(bytes.clone().into()).unwrap();
    let mem = MemPageSource { dag: Arc::new(d.clone()), plan: Arc::new(plan.clone()) };
    for p in 0..plan.pages.len() as u32 {
        assert_eq!(file.read_page(p).unwrap(), mem.read_page(p).unwrap(), "page {p}");
    }
    // File backed, at a non-zero offset (as inside a geometry cache), reading single pages.
    let dir = std::env::temp_dir().join(format!("zest_pages_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.bin");
    let mut blob = vec![0xAAu8; 37];
    blob.extend_from_slice(&bytes);
    blob.extend_from_slice(&[0xBB; 11]);
    std::fs::write(&path, &blob).unwrap();
    let f = PagedFile::open_file(&path, 37, bytes.len() as u64).expect("opens from a file");
    assert_eq!(f.plan, plan);
    for p in [0u32, 1, plan.pages.len() as u32 - 1] {
        assert_eq!(f.read_page(p).unwrap(), mem.read_page(p).unwrap());
    }
    assert!(PagedFile::open_file(&path, 37, bytes.len() as u64 - 1).is_err(), "a wrong length must be refused");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn gpu_cluster_records_are_page_relative() {
    let d = dag(220, 110);
    let plan = PagePlan::split(&d, &small());
    let gpu = plan.gpu_clusters(&d);
    let mem = MemPageSource { dag: Arc::new(d.clone()), plan: Arc::new(plan.clone()) };
    for (p, info) in plan.pages.iter().enumerate() {
        let payload = mem.read_page(p as u32).unwrap();
        for &c in &info.clusters {
            let g = &gpu[c as usize];
            assert_eq!(g.morph_offset, p as u32);
            assert_eq!(g.morph_count, plan.child_page[c as usize]);
            assert!((g.vertex_offset + u32::from(g.vertex_count)) <= info.vertices);
            assert!(g.triangle_offset.is_multiple_of(4) && g.triangle_offset + u32::from(g.triangle_count) * 3 <= info.tri_bytes);
            // The relocated cluster decodes to the same vertices as the original.
            let o = &d.clusters[c as usize];
            let a = &payload.vertices[g.vertex_offset as usize..][..g.vertex_count as usize];
            let b = &d.vertices[o.vertex_offset as usize..][..o.vertex_count as usize];
            assert_eq!(a, b);
            let ta = &payload.triangles[g.triangle_offset as usize..][..g.triangle_count as usize * 3];
            let tb = &d.triangles[o.triangle_offset as usize..][..o.triangle_count as usize * 3];
            assert_eq!(ta, tb);
        }
    }
}

#[test]
fn corrupt_paged_data_is_rejected_never_a_panic() {
    let d = dag(120, 60);
    let plan = PagePlan::split(&d, &PageOptions { page_bytes: 40 * 1024, resident_bytes: 64 * 1024, min_stream_bytes: 1 });
    assert!(plan.pages.len() > 4);
    let bytes = encode_paged(&d, &plan);
    assert!(decode_paged(&bytes).is_ok());
    // Truncation anywhere.
    for cut in [0usize, 3, 8, 40, 64, 100, bytes.len() / 3, bytes.len() / 2, bytes.len() - 1] {
        assert!(decode_paged(&bytes[..cut]).is_err(), "truncation at {cut} accepted");
    }
    // A flipped byte anywhere is caught by a checksum (header region, directory or a page blob).
    let mut rng = Rng(3);
    for _ in 0..200 {
        let mut b = bytes.clone();
        let at = (rng.next() as usize) % b.len();
        b[at] ^= 1 << (rng.next() % 8);
        assert!(decode_paged(&b).is_err(), "a flipped byte at {at} was accepted");
    }
    // A forged plan (with valid checksums) that splits the inputs of a group over two pages.
    let mut bad = plan.cluster_page.clone();
    let g = &d.groups[d.groups.len() / 3];
    let kid = d.group_children[g.first_child as usize] as usize;
    let other = (bad[kid] + 1) % plan.pages.len() as u32;
    bad[kid] = other;
    assert!(PagePlan::from_assignment(&d, bad).is_err(), "a split input set was accepted");
    // A root cluster off the root page.
    let mut bad = plan.cluster_page.clone();
    let root = d.root_clusters().next().unwrap() as usize;
    bad[root] = 1;
    assert!(PagePlan::from_assignment(&d, bad).is_err(), "a root cluster outside page 0 was accepted");
    // A payload whose triangle indexes past its vertices, and one with wrong sizes.
    let mem = MemPageSource { dag: Arc::new(d.clone()), plan: Arc::new(plan.clone()) };
    let mut p = mem.read_page(1).unwrap();
    assert!(validate_payload(&d, &plan, 1, &p).is_ok());
    p.triangles[0] = 255;
    assert!(validate_payload(&d, &plan, 1, &p).is_err());
    let mut p = mem.read_page(1).unwrap();
    p.vertices.pop();
    assert!(validate_payload(&d, &plan, 1, &p).is_err());
    let mut p = mem.read_page(1).unwrap();
    p.vertices[0].pos = [u16::MAX; 3];
    assert!(validate_payload(&d, &plan, 1, &p).is_err(), "a vertex far outside its cluster was accepted");
    assert!(validate_payload(&d, &plan, 9999, &p).is_err());
}

#[test]
fn geometry_cache_formats_old_and_new() {
    use crate::geo_cache::{GEO_FILE_FORMAT, ModelGeometry, decode, encode};
    let small = dag(40, 30);
    let big = dag(220, 110);
    // The default options leave the 1 MB test mesh whole; force pages by checking the real split on a mesh over the limit is covered elsewhere,
    // so here the paged slot is written through the same encoder the cache uses on a DAG that exceeds the defaults.
    let geo = ModelGeometry { key: 0xABCD, meshes: vec![Some(small.clone()), None] };
    let bytes = encode(&geo);
    assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), GEO_FILE_FORMAT);
    assert_eq!(decode(&bytes).unwrap(), geo);
    // A format 1 file (what older builds wrote): same plain slots, other version word, own checksum.
    let mut old = bytes[..bytes.len() - 16].to_vec();
    old[4..8].copy_from_slice(&1u32.to_le_bytes());
    let mut h = crate::import_cache::Hasher128::new();
    h.update(&old);
    old.extend_from_slice(&h.finish().to_le_bytes());
    assert_eq!(decode(&old).unwrap(), geo, "an old cache must still load (as a single resident page)");
    // The split of the loaded DAG is the single page the old format implies.
    assert!(PagePlan::split(&small, &PageOptions::default()).is_single());
    // A future format is refused.
    let mut future = bytes[..bytes.len() - 16].to_vec();
    future[4..8].copy_from_slice(&99u32.to_le_bytes());
    let mut h = crate::import_cache::Hasher128::new();
    h.update(&future);
    future.extend_from_slice(&h.finish().to_le_bytes());
    assert!(decode(&future).is_err());
    // A paged slot inside a cache file: build the file by hand around encode_paged and decode it through the cache reader.
    let plan = PagePlan::split(&big, &small_opts());
    let paged = encode_paged(&big, &plan);
    let mut file = Vec::new();
    file.extend_from_slice(b"ZGEO");
    file.extend_from_slice(&GEO_FILE_FORMAT.to_le_bytes());
    file.extend_from_slice(&crate::cluster_dag::GEO_BUILDER_VERSION.to_le_bytes());
    file.extend_from_slice(&7u128.to_le_bytes());
    file.extend_from_slice(&1u32.to_le_bytes());
    file.extend_from_slice(&(paged.len() as u64).to_le_bytes());
    file.extend_from_slice(&paged);
    let mut h = crate::import_cache::Hasher128::new();
    h.update(&file);
    file.extend_from_slice(&h.finish().to_le_bytes());
    let g = decode(&file).expect("a paged slot decodes");
    assert_eq!(g.meshes[0].as_ref().unwrap(), &big);
    // And a flipped byte inside it is refused by the container checksum.
    let mut bad = file.clone();
    let at = file.len() / 2;
    bad[at] ^= 0x10;
    assert!(decode(&bad).is_err());
}

/// A geometry cache file holding `slots` (a paged blob or a plain DAG per slot, `None` = no DAG).
fn cache_file(key: u128, slots: &[Option<Vec<u8>>]) -> Vec<u8> {
    let mut file = Vec::new();
    file.extend_from_slice(b"ZGEO");
    file.extend_from_slice(&crate::geo_cache::GEO_FILE_FORMAT.to_le_bytes());
    file.extend_from_slice(&crate::cluster_dag::GEO_BUILDER_VERSION.to_le_bytes());
    file.extend_from_slice(&key.to_le_bytes());
    file.extend_from_slice(&(slots.len() as u32).to_le_bytes());
    for s in slots {
        match s {
            Some(b) => {
                file.extend_from_slice(&(b.len() as u64).to_le_bytes());
                file.extend_from_slice(b);
            }
            None => file.extend_from_slice(&0u64.to_le_bytes()),
        }
    }
    let mut h = crate::import_cache::Hasher128::new();
    h.update(&file);
    file.extend_from_slice(&h.finish().to_le_bytes());
    file
}

#[test]
fn compressed_pages_read_back_identically_and_old_raw_files_still_load() {
    let d = dag(220, 110);
    let plan = PagePlan::split(&d, &small());
    let raw = encode_paged_with(&d, &plan, PageCodec::Raw);
    let z = encode_paged_with(&d, &plan, PageCodec::Zstd(3));
    let v1 = encode_paged_format(&d, &plan, PageCodec::Raw, 1);
    eprintln!("[pages] {} pages: raw {} bytes, zstd 3 {} bytes ({:.1} % of raw), format 1 {} bytes", plan.pages.len(), raw.len(), z.len(), 100.0 * z.len() as f64 / raw.len() as f64, v1.len());
    assert!(z.len() < raw.len(), "zstd did not shrink the file");
    assert!(v1.len() < raw.len(), "format 1 entries are smaller");
    let (fr, fz, f1) = (PagedFile::from_bytes(raw.into()).unwrap(), PagedFile::from_bytes(z.clone().into()).unwrap(), PagedFile::from_bytes(v1.into()).unwrap());
    let mem = MemPageSource { dag: Arc::new(d.clone()), plan: Arc::new(plan.clone()) };
    for p in 0..plan.pages.len() as u32 {
        let want = mem.read_page(p).unwrap();
        assert_eq!(fr.read_page(p).unwrap(), want, "raw page {p}");
        assert_eq!(fz.read_page(p).unwrap(), want, "zstd page {p}");
        assert_eq!(f1.read_page(p).unwrap(), want, "format 1 page {p}");
        assert_eq!(fz.read_cold(p).unwrap(), fr.read_cold(p).unwrap(), "source ids of page {p}");
    }
    assert_eq!(fz.assemble().unwrap(), d, "a compressed file assembles to the original DAG");
    assert_eq!(f1.assemble().unwrap(), d);
    // The default encoder writes the current format.
    assert_eq!(u32::from_le_bytes(encode_paged(&d, &plan)[4..8].try_into().unwrap()), PAGE_FORMAT);
    // A flipped byte in a compressed file is rejected (stored bytes are checksummed).
    let mut rng = Rng(11);
    for _ in 0..200 {
        let mut b = z.clone();
        let at = (rng.next() as usize) % b.len();
        b[at] ^= 1 << (rng.next() % 8);
        assert!(decode_paged(&b).is_err(), "a flipped byte at {at} of a compressed file was accepted");
    }
}

#[test]
fn a_skeleton_validates_without_its_payload_and_bad_links_are_still_caught() {
    let d = dag(120, 60);
    let sk = skeleton_of(&d);
    assert!(sk.vertices.is_empty() && sk.triangles.is_empty());
    sk.validate_skeleton().expect("the skeleton of a good DAG validates");
    assert!(sk.validate().is_err(), "a skeleton is not a drawable DAG");
    let mut bad = sk.clone();
    let leaf = d.leaf_clusters().next().unwrap() as usize;
    bad.links[leaf].level = 5;
    assert!(bad.validate_skeleton().is_err(), "a broken group link was accepted");
    let mut bad = sk.clone();
    bad.nodes[0].children[0].max_parent_error = f32::NAN;
    assert!(bad.validate_skeleton().is_err(), "a NaN BVH error was accepted");
    let mut bad = sk.clone();
    bad.vertices.push(Default::default());
    assert!(bad.validate_skeleton().is_err(), "a skeleton that carries payload was accepted");
}

#[test]
fn lazy_loading_leaves_paged_meshes_on_disk() {
    use crate::geo_cache::{self, LazyGeometry};
    let paged_dag = dag(220, 110);
    let plain = dag(40, 30);
    let plan = PagePlan::split(&paged_dag, &small_opts());
    assert!(plan.pages.len() > 4);
    let key = 0x1234_5678_9ABC_DEF0u128;
    let file = cache_file(key, &[Some(plain.to_bytes()), None, Some(encode_paged(&paged_dag, &plan))]);
    let dir = std::env::temp_dir().join(format!("zest_lazy_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let source = dir.join("m.glb");
    let path = geo_cache::geo_path(&source).expect("cache path");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, &file).unwrap();

    let lazy = geo_cache::load_lazy(&source, key).expect("a good file loads lazily");
    assert_eq!(lazy.geo.meshes.len(), 3);
    assert_eq!(lazy.geo.meshes[0].as_ref().unwrap(), &plain, "plain slots are decoded as before");
    assert!(lazy.geo.meshes[1].is_none() && lazy.paged[1].is_none());
    let sk = lazy.geo.meshes[2].as_ref().unwrap();
    assert!(sk.vertices.is_empty() && sk.triangles.is_empty() && !sk.clusters.is_empty(), "the paged slot holds the skeleton only");
    let pf = lazy.paged[2].as_ref().expect("paged slot keeps its file");
    assert_eq!(pf.plan, plan);
    let mem = MemPageSource { dag: Arc::new(paged_dag.clone()), plan: Arc::new(plan.clone()) };
    for p in [0u32, 1, plan.pages.len() as u32 - 1] {
        assert_eq!(pf.read_page(p).unwrap(), mem.read_page(p).unwrap());
    }
    // The decoded copy of the same file holds the payload this one does not.
    let eager = geo_cache::decode(&file).unwrap();
    let eager_bytes: usize = eager.dag_bytes();
    let lazy_bytes: usize = lazy.geo.dag_bytes();
    eprintln!("[lazy] cache file {} bytes; decoded DAGs {} bytes, lazily loaded {} bytes ({:.1} %)", file.len(), eager_bytes, lazy_bytes, 100.0 * lazy_bytes as f64 / eager_bytes as f64);
    assert!(lazy_bytes * 3 < eager_bytes, "the lazy load kept most of the payload");
    // Assembling on demand gives the original DAG.
    let mut lazy = lazy;
    assert_eq!(lazy.make_full(2).unwrap(), &paged_dag);
    assert!(lazy.paged[2].is_none());
    // Another key, a flipped byte, a truncated file and a missing file: no lazy geometry.
    assert!(geo_cache::load_lazy(&source, key + 1).is_none());
    let mut bad = file.clone();
    bad[file.len() / 2] ^= 4;
    std::fs::write(&path, &bad).unwrap();
    assert!(geo_cache::load_lazy(&source, key).is_none(), "a corrupt file must fall back to the full path");
    std::fs::write(&path, &file[..file.len() - 40]).unwrap();
    assert!(geo_cache::load_lazy(&source, key).is_none());
    std::fs::remove_dir_all(&dir).ok();
    assert!(geo_cache::load_lazy(&source, key).is_none());
    let _ = LazyGeometry::eager(geo_cache::ModelGeometry::default());
}

fn small_opts() -> PageOptions {
    small()
}

#[cfg(feature = "client")]
#[test]
fn streamed_upload_validates_and_rejects_bad_layouts() {
    let d = dag(220, 110);
    let plan = PagePlan::split(&d, &small());
    let mem = MemPageSource { dag: Arc::new(d.clone()), plan: Arc::new(plan.clone()) };
    let up = plan.to_gpu_upload(&skeleton_of(&d), &mem.read_page(0).unwrap());
    up.validate().expect("a streamed upload validates");
    let st = up.stream.as_ref().unwrap();
    assert_eq!(st.pages.len(), plan.pages.len());
    let mut bad = up.clone();
    bad.clusters[0].morph_offset = 9999;
    assert!(bad.validate().is_err(), "a page id out of range");
    let mut bad = up.clone();
    bad.clusters[0].morph_count = 12345;
    assert!(bad.validate().is_err(), "a child page out of range");
    let mut bad = up.clone();
    bad.stream.as_mut().unwrap().pages[1].deps = vec![1];
    assert!(bad.validate().is_err(), "a page depending on itself");
    let mut bad = up.clone();
    bad.stream.as_mut().unwrap().pages[0].deps = vec![1];
    assert!(bad.validate().is_err(), "a root page with dependencies");
    let mut bad = up.clone();
    bad.triangles.pop();
    assert!(bad.validate().is_err(), "root payload shorter than its layout");
    let mut bad = up.clone();
    let c = bad.clusters.iter().position(|c| c.morph_offset != 0).unwrap();
    bad.clusters[c].vertex_offset = 1 << 20;
    assert!(bad.validate().is_err(), "cluster vertices outside their page");
}
