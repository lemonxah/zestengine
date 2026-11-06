//! GEO-4: the GPU cluster cull against the cooked DAG. The CPU parts (the Rust
//! reference of the shader's math against `ClusterDag::select_bvh`, record
//! sizes) always run. The device parts run `cluster_cull.comp` for real through
//! `zestengine_render::cluster_headless` and skip (pass with a note) when the
//! machine has no Vulkan 1.3 device; set `ZESTENGINE_TEST_GPU=0` to force the
//! skip. With the Khronos validation layer installed it is enabled and any
//! error fails the run.

use std::collections::HashSet;
use std::path::Path;
use glam::{Mat4, Quat, Vec3};
use zestengine_render::cluster::{ClusterMeshUpload, ClusterRec, counter};
use zestengine_render::cluster_headless::{
    HeadlessGpu, HeadlessHiz, HeadlessInstance, HeadlessOptions, HeadlessRun, HeadlessView, mesh_sphere,
};
use zestengine_render::cluster_ref::{self, RefInstance, RefView};

use super::tests::{build, model, sphere, torus};
use super::*;

struct Rng(u64);

impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
}

fn gpu() -> Option<HeadlessGpu> {
    let g = HeadlessGpu::new().expect("headless device setup");
    match &g {
        Some(g) => eprintln!("[gpu] {} (validation layer: {})", g.device_name(), g.validation_enabled()),
        None => eprintln!("[gpu] no Vulkan 1.3 device: device checks skipped"),
    }
    g
}

fn upload(dag: &ClusterDag) -> ClusterMeshUpload {
    let up = dag.to_gpu_upload();
    up.validate().expect("a cooked DAG validates");
    up
}

struct Scenario {
    view: HeadlessView,
    rv: RefView,
    insts: Vec<HeadlessInstance>,
    refs: Vec<RefInstance>,
    eye: Vec3,
}

fn scenario(rng: &mut Rng, sphere: [f32; 4], instances: usize, batches: u32) -> Scenario {
    let centre = Vec3::new(sphere[0], sphere[1], sphere[2]);
    let r = sphere[3].max(1e-3);
    let dir = Vec3::new(rng.range(-1.0, 1.0), rng.range(-0.6, 0.8), rng.range(-1.0, 1.0)).normalize_or_zero();
    let dist = r * if rng.f() < 0.1 { rng.range(0.0, 1.5) } else { rng.range(1.2, 30.0) };
    let eye = centre + dir * dist;
    let target = centre + Vec3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)) * r * 0.5;
    let fov = rng.range(0.5, 1.5);
    let aspect = rng.range(1.0, 1.9);
    let height = [720.0, 1080.0, 2160.0][(rng.f() * 3.0) as usize % 3];
    let target_px = [0.5, 1.0, 1.0, 2.0][(rng.f() * 4.0) as usize % 4];
    let mut rv = cluster_ref::make_view(eye, target, fov, aspect, height, (r * 0.01).max(0.01), r * 200.0, target_px);
    let proj = Mat4::perspective_rh(fov, aspect, (r * 0.01).max(0.01), r * 200.0);
    let view_proj = proj * Mat4::look_at_rh(eye, target, Vec3::Y);
    rv.near_test = true;
    let view = HeadlessView { view_proj, cam: eye, pixels_per_unit: rv.pixels_per_unit, target_px };
    let mut insts = Vec::new();
    let mut refs = Vec::new();
    for i in 0..instances {
        let q = Quat::from_euler(glam::EulerRot::YXZ, rng.range(0.0, std::f32::consts::TAU), rng.range(0.0, std::f32::consts::TAU), rng.range(0.0, std::f32::consts::TAU));
        let mut scale = Vec3::splat(rng.range(0.4, 2.5));
        if rng.f() < 0.2 {
            scale *= Vec3::new(1.0, rng.range(0.5, 2.0), rng.range(0.5, 2.0)); // non-uniform: the cone test is off
        }
        if rng.f() < 0.1 {
            scale.x = -scale.x; // mirrored: the cone test is off
        }
        let shift = Vec3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)) * r * if i == 0 { 0.2 } else { 1.5 };
        let model = Mat4::from_scale_rotation_translation(scale, q, shift);
        insts.push(HeadlessInstance { mesh: 0, model, batch: i as u32 % batches, skin: None, lod_scale: 1.0 });
        refs.push(RefInstance { model: model.to_cols_array_2d(), eye: cluster_ref::object_space_eye(model, eye) });
    }
    Scenario { view, rv, insts, refs, eye }
}

/// The cut `select_bvh` gives under the shader's own `lod_ok`, then the
/// per-cluster tests. No node-level pruning: the plain CPU reference.
fn plain_expected(dag: &ClusterDag, rv: &RefView, inst: &RefInstance) -> Vec<u32> {
    let s = cluster_ref::max_scale(&inst.model);
    let sel = dag.select_bvh(&|err, sph| cluster_ref::lod_ok(rv, &inst.model, s, err, sph));
    sel.into_iter()
        .filter(|&c| {
            let cl: ClusterRec = bytemuck::cast(dag.clusters[c as usize]);
            let ctr = cluster_ref::transform_point(&inst.model, [cl.bounds[0], cl.bounds[1], cl.bounds[2]]);
            !cluster_ref::outside_frustum(rv, ctr, cl.bounds[3] * s) && !cluster_ref::cone_culled(&cl, inst.eye, rv.slack)
        })
        .collect()
}

fn instance_culled(rv: &RefView, inst: &RefInstance, sph: [f32; 4]) -> bool {
    let s = cluster_ref::max_scale(&inst.model);
    let c = cluster_ref::transform_point(&inst.model, [sph[0], sph[1], sph[2]]);
    cluster_ref::outside_frustum(rv, c, sph[3] * s)
}

#[test]
fn record_layouts_match_the_assets_structs() {
    use zestengine_render::cluster::{BvhNodeRec, ClusterVertexRec};
    assert_eq!(size_of::<GpuCluster>(), size_of::<ClusterRec>());
    assert_eq!(size_of::<ClusterVertex>(), size_of::<ClusterVertexRec>());
    assert_eq!(size_of::<BvhNode>(), size_of::<BvhNodeRec>());
    assert_eq!(std::mem::align_of::<GpuCluster>(), std::mem::align_of::<ClusterRec>());
    // And the bytes survive the cast field for field.
    let m = sphere(24, 48);
    let dag = build(&m);
    // The engine checks the installed mesh against this before using a DAG.
    assert_eq!(
        (dag.header.source_triangles + dag.header.dropped_triangles) as usize,
        m.indices.len() / 3
    );
    let up = dag.to_gpu_upload();
    for (a, b) in dag.clusters.iter().zip(&up.clusters) {
        assert_eq!(a.bounds, b.bounds);
        assert_eq!(a.parent_lod_error.to_bits(), b.parent_lod_error.to_bits());
        assert_eq!([a.cone_axis[0] as u8, a.cone_axis[1] as u8, a.cone_axis[2] as u8, a.cone_cutoff], b.cone);
        assert_eq!((a.vertex_offset, a.triangle_offset, a.vertex_count, a.triangle_count), (b.vertex_offset, b.triangle_offset, b.vertex_count, b.triangle_count));
        assert_eq!((a.section, a.level), (b.section, b.level));
    }
    for (a, b) in dag.nodes.iter().zip(&up.nodes) {
        for (x, y) in a.children.iter().zip(&b.children) {
            assert_eq!((x.child, x.count, x.bounds, x.lod_sphere), (y.child, y.count, y.bounds, y.lod_sphere));
            assert_eq!(x.max_parent_error.to_bits(), y.max_parent_error.to_bits());
        }
    }
    for (a, b) in dag.vertices.iter().zip(&up.vertices) {
        assert_eq!((a.pos, a.flags, a.normal, a.tangent, a.uv), (b.pos, b.flags, b.normal, b.tangent, b.uv));
    }
    assert!(up.validate().is_ok());
}

/// The Rust reference of the shader's traversal and tests equals
/// `ClusterDag::select_bvh` + the cull rules, exactly, over many cameras (CPU only).
#[test]
fn reference_equals_select_bvh() {
    let meshes = [("sphere", sphere(48, 96)), ("torus", torus(160, 80))];
    for (name, m) in meshes {
        let dag = build(&m);
        let up = upload(&dag);
        let sph = mesh_sphere(&up);
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let (mut total, mut nonempty) = (0usize, 0usize);
        for _ in 0..200 {
            let sc = scenario(&mut rng, sph, 2, 1);
            for inst in &sc.refs {
                let want = plain_expected(&dag, &sc.rv, inst);
                let (got, _) = cluster_ref::cut(&up.clusters, &up.nodes, &sc.rv, inst, false);
                assert_eq!(got, want, "{name}: reference without node pruning must equal select_bvh + cull rules");
                // Node-level frustum pruning is conservative (the BVH bounds enclose their clusters).
                let (pruned, _) = cluster_ref::cut(&up.clusters, &up.nodes, &sc.rv, inst, true);
                let missing = want.iter().filter(|c| !pruned.contains(c)).count();
                assert!(missing * 1000 <= want.len().max(1), "{name}: node pruning dropped {missing} of {} clusters", want.len());
                total += want.len();
                nonempty += usize::from(!want.is_empty());
            }
        }
        eprintln!("[cpu-ref] {name}: {total} clusters over {nonempty} non-empty cuts equal select_bvh");
        assert!(nonempty > 100 && total > 1000);
    }
}

/// Compare a device run with the reference, instance by instance; clusters
/// whose membership flips under a 0.1% perturbation are not compared (float
/// rounding in the shader may differ from the CPU's in the last ulp).
fn check_run(name: &str, up: &ClusterMeshUpload, sc: &Scenario, run: &HeadlessRun, batches: usize) -> (usize, usize) {
    assert!(run.out_of_range_writes.is_empty(), "{name}: {:?}", run.out_of_range_writes);
    assert_eq!(run.validation_errors, 0, "{name}: validation errors");
    assert_eq!(run.counters[counter::FLAGS], 0, "{name}: overflow / bad index flags {:#x}", run.counters[counter::FLAGS]);
    let sph = mesh_sphere(up);
    let mut lo = sc.rv;
    lo.slack = -1e-3;
    let mut hi = sc.rv;
    hi.slack = 1e-3;
    let mut compared = 0;
    let mut mismatched = 0;
    let mut got: Vec<Vec<u32>> = vec![Vec::new(); sc.insts.len()];
    for b in 0..batches {
        for &(i, c) in &run.early[b] {
            assert_eq!(sc.insts[i as usize].batch as usize, b, "{name}: instance in the wrong batch");
            got[i as usize].push(c);
        }
    }
    for (i, inst) in sc.refs.iter().enumerate() {
        got[i].sort_unstable();
        assert!(got[i].windows(2).all(|w| w[0] != w[1]), "{name}: a cluster was listed twice for instance {i}");
        let skip_inst = instance_culled(&lo, inst, sph) != instance_culled(&hi, inst, sph);
        if skip_inst {
            continue;
        }
        if instance_culled(&sc.rv, inst, sph) {
            assert!(got[i].is_empty(), "{name}: culled instance {i} still listed {} clusters", got[i].len());
            continue;
        }
        let (a, _) = cluster_ref::cut(&up.clusters, &up.nodes, &lo, inst, true);
        let (b, _) = cluster_ref::cut(&up.clusters, &up.nodes, &hi, inst, true);
        let (exact, _) = cluster_ref::cut(&up.clusters, &up.nodes, &sc.rv, inst, true);
        // Every cluster either side of the perturbation, plus what the GPU returned.
        let mut all: Vec<u32> = a.iter().chain(&b).chain(&exact).chain(&got[i]).copied().collect();
        all.sort_unstable();
        all.dedup();
        for c in all {
            let (in_a, in_b) = (a.binary_search(&c).is_ok(), b.binary_search(&c).is_ok());
            if in_a != in_b {
                continue; // borderline
            }
            compared += 1;
            if got[i].binary_search(&c).is_ok() != in_a {
                mismatched += 1;
                eprintln!("[{name}] instance {i} cluster {c}: gpu {} reference {}", !in_a, in_a);
            }
        }
    }
    (compared, mismatched)
}

fn device_cut_test(name: &str, m: &MeshData, scenarios: usize, instances: usize, seed: u64) {
    let Some(gpu) = gpu() else { return };
    let dag = build(m);
    let up = upload(&dag);
    let sph = mesh_sphere(&up);
    let mut rng = Rng(seed);
    let batches = 2usize;
    let (mut compared, mut mismatched, mut visible, mut max_depth_seen) = (0, 0, 0usize, 0u32);
    let t = std::time::Instant::now();
    for _ in 0..scenarios {
        let sc = scenario(&mut rng, sph, instances, batches as u32);
        let run = gpu
            .run(std::slice::from_ref(&up), &sc.insts, batches as u32, &sc.view, HeadlessHiz::Off, &HeadlessOptions::default())
            .expect("device run");
        let (c, mm) = check_run(name, &up, &sc, &run, batches);
        compared += c;
        mismatched += mm;
        visible += run.early.iter().map(Vec::len).sum::<usize>();
        // The queue levels the run needed.
        max_depth_seen = max_depth_seen.max(run.plan.levels);
        // The commands: 384 vertices, instanceCount = the list length, late = 0.
        for b in 0..batches {
            assert_eq!(run.commands[b][0], 384);
            assert_eq!(run.commands[b][1] as usize, run.early[b].len());
            assert_eq!(run.commands[batches + b][1], 0, "no late instances without Hi-Z");
        }
        // Counter bookkeeping: visible equals what was listed.
        assert_eq!(run.counters[counter::VISIBLE] as usize, run.early.iter().map(Vec::len).sum::<usize>());
        let _ = sc.eye;
    }
    eprintln!(
        "[gpu] {name}: {scenarios} cameras x {instances} instances, {compared} cluster decisions compared, {mismatched} mismatches, \
         {visible} clusters drawn, {} BVH levels, {:.2}s",
        max_depth_seen,
        t.elapsed().as_secs_f32()
    );
    assert!(compared > 0 && visible > 0, "the test saw nothing");
    assert_eq!(mismatched, 0, "{name}: the shader disagrees with the reference");
}

#[test]
fn gpu_cut_matches_reference_sphere() {
    device_cut_test("sphere", &sphere(64, 128), 120, 3, 11);
}

#[test]
fn gpu_cut_matches_reference_torus() {
    device_cut_test("torus", &torus(200, 100), 120, 3, 12);
}

#[test]
fn gpu_cut_matches_reference_damaged_helmet() {
    let Some(m) = model("models/DamagedHelmet.glb") else { return };
    device_cut_test("DamagedHelmet", &m, 100, 3, 13);
}

#[test]
#[ignore = "builds a 1M-triangle DAG (seconds); run with --ignored --nocapture"]
fn gpu_cut_matches_reference_million_triangle_torus() {
    device_cut_test("1M torus", &torus(1000, 500), 40, 2, 14);
}

/// The shader's vertex-pulling decode equals the CPU decode: positions bit for
/// bit, normals and tangents within the octahedral decode tolerance, uv exact, colour RGBA8 exact;
/// vertices past the cluster's triangle count and bad requests are dead.
#[test]
fn gpu_decode_equals_cpu_decode() {
    let Some(gpu) = gpu() else { return };
    let mut coloured = torus(120, 60);
    for (i, v) in coloured.vertices.iter_mut().enumerate() {
        // Every channel distinct, including a non-trivial alpha (RGBA8 survives the pool as the CPU quantised it).
        v.color = [(i % 61) as f32 / 60.0, (i / 61 % 121) as f32 / 120.0, ((i * 7) % 255) as f32 / 255.0, 0.25 + 0.75 * ((i % 5) as f32 / 4.0)];
    }
    for (name, m) in [("sphere", sphere(48, 96)), ("torus", torus(120, 60)), ("coloured torus", coloured)] {
        let dag = build(&m);
        let up = upload(&dag);
        let mut requests = Vec::new();
        let mut rng = Rng(77);
        for _ in 0..2000 {
            let c = (rng.f() * dag.clusters.len() as f32) as u32 % dag.clusters.len() as u32;
            requests.push([c, (rng.f() * 384.0) as u32 % 384, 0, 0]);
        }
        // Every vertex of a few clusters, and out-of-range requests.
        for c in [0u32, dag.clusters.len() as u32 / 2, dag.clusters.len() as u32 - 1] {
            requests.extend((0..384).map(|v| [c, v, 0, 0]));
        }
        requests.push([dag.clusters.len() as u32, 0, 0, 0]);
        requests.push([u32::MAX, 5, 0, 0]);
        requests.push([0, 384, 0, 0]);
        requests.push([0, u32::MAX, 0, 0]);
        let out = gpu.decode(&up, &requests).expect("decode run");
        let (mut live, mut dead) = (0, 0);
        for (rq, o) in requests.iter().zip(&out) {
            let (c, v) = (rq[0] as usize, rq[1] as usize);
            let expect_live = c < dag.clusters.len() && v < 384 && v / 3 < dag.clusters[c].triangle_count as usize;
            assert_eq!(o[3] == 1.0, expect_live, "{name}: cluster {c} vertex {v} liveness");
            if !expect_live {
                dead += 1;
                assert_eq!(o[..3], [0.0, 0.0, 0.0], "{name}: a dead vertex must read as zero");
                continue;
            }
            live += 1;
            let cl = &dag.clusters[c];
            let tri = &dag.triangles[cl.triangle_offset as usize + (v / 3) * 3..][..3];
            let vert = &dag.vertices[cl.vertex_offset as usize + tri[v % 3] as usize];
            let pos = dag.header.decode_position(vert.pos);
            assert_eq!(
                [o[0].to_bits(), o[1].to_bits(), o[2].to_bits()],
                pos.map(f32::to_bits),
                "{name}: cluster {c} vertex {v} position is not bit-identical"
            );
            let n = oct_decode(vert.normal);
            let t = oct_decode(vert.tangent);
            for k in 0..3 {
                assert!((o[4 + k] - n[k]).abs() < 2e-6, "{name}: normal {k}: {} vs {}", o[4 + k], n[k]);
                assert!((o[8 + k] - t[k]).abs() < 2e-6, "{name}: tangent {k}: {} vs {}", o[8 + k], t[k]);
            }
            assert_eq!(o[11], if vert.flags & 1 != 0 { -1.0 } else { 1.0 }, "{name}: handedness");
            assert_eq!([o[12].to_bits(), o[13].to_bits()], vert.uv.map(f32::to_bits), "{name}: uv");
            assert_eq!(o[14] as u32, cl.level, "{name}: level tag");
            assert_eq!(o[15].to_bits().to_le_bytes(), vert.color, "{name}: cluster {c} vertex {v} colour");
        }
        eprintln!("[gpu] {name}: decoded {live} live and {dead} dead vertices, positions bit-identical");
        assert!(live > 2000 && dead > 4);
    }
}

/// Two-phase Hi-Z with a single-texel pyramid: a pyramid of 0.0 (everything
/// nearer) hides every cluster that lies wholly on screen in the early pass,
/// which queues them; a pyramid of 1.0 in the late pass recovers all of them.
#[test]
fn gpu_two_phase_hiz_queues_and_recovers() {
    let Some(gpu) = gpu() else { return };
    let m = sphere(48, 96);
    let dag = build(&m);
    let up = upload(&dag);
    let sph = mesh_sphere(&up);
    let r = sph[3];
    let eye = Vec3::new(sph[0] + 0.3 * r, sph[1] + 0.4 * r, sph[2] + 4.0 * r);
    let target = Vec3::new(sph[0], sph[1], sph[2]);
    let proj = Mat4::perspective_rh(1.0, 1.5, 0.05 * r, 100.0 * r);
    let view_proj = proj * Mat4::look_at_rh(eye, target, Vec3::Y);
    let rv = cluster_ref::make_view(eye, target, 1.0, 1.5, 1080.0, 0.05 * r, 100.0 * r, 1.0);
    let hv = HeadlessView { view_proj, cam: eye, pixels_per_unit: rv.pixels_per_unit, target_px: 1.0 };
    let inst = HeadlessInstance { mesh: 0, model: Mat4::IDENTITY, batch: 0, skin: None, lod_scale: 1.0 };
    let refi = RefInstance { model: Mat4::IDENTITY.to_cols_array_2d(), eye: cluster_ref::object_space_eye(Mat4::IDENTITY, eye) };
    let (want, _) = cluster_ref::cut(&up.clusters, &up.nodes, &rv, &refi, true);
    assert!(want.len() > 20);
    let run = gpu
        .run(
            std::slice::from_ref(&up),
            &[inst],
            1,
            &hv,
            HeadlessHiz::Uniform { early: 0.0, late: 1.0, late_vp: view_proj },
            &HeadlessOptions::default(),
        )
        .expect("device run");
    assert!(run.out_of_range_writes.is_empty(), "{:?}", run.out_of_range_writes);
    assert_eq!(run.validation_errors, 0);
    assert_eq!(run.counters[counter::FLAGS], 0);
    assert!(run.early[0].is_empty(), "the early pass drew {} clusters against a fully occluding pyramid", run.early[0].len());
    let mut occ: Vec<u32> = run.occluded.iter().map(|e| e.1).collect();
    occ.sort_unstable();
    assert_eq!(occ, want, "the early pass queued exactly the clusters that survive the other tests");
    let mut late: Vec<u32> = run.late[0].iter().map(|e| e.1).collect();
    late.sort_unstable();
    assert_eq!(late, want, "the late pass recovers every queued cluster against an empty pyramid");
    assert_eq!(run.counters[counter::LATE_RECOVERED] as usize, want.len());
    assert_eq!(run.counters[counter::LATE_OCCLUDED], 0);
    assert_eq!(run.commands[1][1] as usize, want.len(), "late indirect command");
    assert_eq!(run.commands[0][1], 0);
    eprintln!("[gpu] two-phase Hi-Z: {} clusters queued early, {} recovered late", occ.len(), late.len());
}

/// Corrupt cooked data (random byte flips in clusters and BVH nodes, bypassing
/// the install checks) must never make the shader write outside its buffers
/// or hang. Only the sentinel / flag checks apply: the output is garbage by design.
#[test]
fn gpu_survives_corrupt_dags() {
    let Some(gpu) = gpu() else { return };
    let m = sphere(40, 80);
    let dag = build(&m);
    let good = upload(&dag);
    let sph = mesh_sphere(&good);
    let mut rng = Rng(4242);
    for round in 0..24 {
        let mut up = good.clone();
        let flips = 1 + (rng.f() * 24.0) as usize;
        for _ in 0..flips {
            match (rng.f() * 3.0) as u32 {
                0 => {
                    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut up.nodes);
                    let i = (rng.f() * bytes.len() as f32) as usize % bytes.len();
                    bytes[i] ^= 1 << ((rng.f() * 8.0) as u32 % 8);
                }
                1 => {
                    let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut up.clusters);
                    let i = (rng.f() * bytes.len() as f32) as usize % bytes.len();
                    bytes[i] ^= 1 << ((rng.f() * 8.0) as u32 % 8);
                }
                _ => {
                    // A whole field: an out-of-range or cyclic child.
                    let n = (rng.f() * up.nodes.len() as f32) as usize % up.nodes.len();
                    let k = (rng.f() * 4.0) as usize % 4;
                    up.nodes[n].children[k].child = [0, 1, u32::MAX - 1, 0x7FFF_FFFF, 0x8000_0000 | 0x00FF_FFFF, n as u32][(rng.f() * 6.0) as usize % 6];
                    up.nodes[n].children[k].count = [0, 9, 1_000_000, u32::MAX][(rng.f() * 4.0) as usize % 4];
                }
            }
        }
        let sc = scenario(&mut rng, sph, 3, 2);
        let run = gpu
            .run(
                std::slice::from_ref(&up),
                &sc.insts,
                2,
                &sc.view,
                HeadlessHiz::Uniform { early: 0.0, late: 1.0, late_vp: sc.view.view_proj },
                &HeadlessOptions { unchecked: true, ..Default::default() },
            )
            .expect("a corrupt DAG must not fail the device");
        assert!(run.out_of_range_writes.is_empty(), "round {round}: {:?}", run.out_of_range_writes);
        assert_eq!(run.validation_errors, 0, "round {round}");
    }
    eprintln!("[gpu] 24 corrupted DAGs: no out-of-range write, no hang");
}

/// Capacity limits: with queues and lists far too small, the run overflows,
/// reports it in the flags and counters, writes nothing outside its buffers,
/// and the commands never exceed the capacity.
#[test]
fn gpu_overflow_is_flagged_and_bounded() {
    let Some(gpu) = gpu() else { return };
    let m = torus(120, 60);
    let dag = build(&m);
    let up = upload(&dag);
    let sph = mesh_sphere(&up);
    let mut rng = Rng(5);
    let mut flagged = 0;
    for _ in 0..12 {
        let sc = scenario(&mut rng, sph, 6, 2);
        let run = gpu
            .run(
                std::slice::from_ref(&up),
                &sc.insts,
                2,
                &sc.view,
                HeadlessHiz::Off,
                &HeadlessOptions { cap_limit: Some(3), ..Default::default() },
            )
            .expect("device run");
        assert!(run.out_of_range_writes.is_empty(), "{:?}", run.out_of_range_writes);
        assert_eq!(run.validation_errors, 0);
        for b in 0..2 {
            assert!(run.commands[b][1] <= 3, "indirect instanceCount {} past the region capacity", run.commands[b][1]);
            assert_eq!(run.early[b].len(), run.commands[b][1] as usize);
        }
        if run.counters[counter::FLAGS] != 0 {
            flagged += 1;
            assert!(run.counters[counter::OVERFLOW] > 0);
        }
    }
    eprintln!("[gpu] overflow: flagged in {flagged} of 12 runs, never out of range");
    assert!(flagged >= 6);
}

/// A pool that is too small for the second mesh grows (a bigger buffer, the old
/// DAG copied) instead of failing the install: the cut lists equal those of a
/// run with generous pools, and the first mesh's data survived the copy.
#[test]
fn gpu_full_pool_grows_instead_of_failing() {
    let Some(gpu) = gpu() else { return };
    let ball = upload(&build(&sphere(40, 80)));
    let ring = upload(&build(&torus(80, 40)));
    let ups = [ball.clone(), ring.clone()];
    // Room for the sphere's arrays, not for both meshes.
    let tight = zestengine_render::cluster::PoolSizes {
        clusters: ball.clusters.len() as u32 + 8,
        vertices: ball.vertices.len() as u32 + 64,
        tri_words: ball.triangles.len() as u32 / 4 + 64,
        nodes: ball.nodes.len() as u32 + 8,
        ..zestengine_render::cluster::PoolSizes::default()
    };
    let mut rng = Rng(77);
    let sph = mesh_sphere(&ring);
    let mut compared = 0usize;
    for _ in 0..4 {
        let mut sc = scenario(&mut rng, sph, 6, 2);
        for (i, inst) in sc.insts.iter_mut().enumerate() {
            inst.mesh = i % 2;
        }
        let big = gpu.run(&ups, &sc.insts, 2, &sc.view, HeadlessHiz::Off, &HeadlessOptions::default()).expect("generous pools");
        let grown = gpu.run(&ups, &sc.insts, 2, &sc.view, HeadlessHiz::Off, &HeadlessOptions { sizes: Some(tight), ..Default::default() }).expect("tight pools must grow, not fail");
        assert!(grown.out_of_range_writes.is_empty() && grown.validation_errors == 0);
        assert_eq!(grown.counters[counter::FLAGS], 0);
        for b in 0..2 {
            let (mut a, mut g) = (big.early[b].clone(), grown.early[b].clone());
            a.sort_unstable();
            g.sort_unstable();
            assert_eq!(a, g, "batch {b}: the grown pool selects different clusters");
            compared += a.len();
        }
    }
    eprintln!("[gpu] pool growth: {compared} listed clusters equal a run with generous pools");
    assert!(compared > 100);
}

/// End to end on the device with the engine's own pipelines: the clustered
/// prepass (real cull, `vkCmdDrawIndirect`, vertex pulling) renders what the
/// ordinary prepass of the original mesh renders, and a second clustered
/// pipeline variant (the error shader with depth compare EQUAL) passes the
/// depth test on every pixel the first one wrote: the prepass and main pass
/// positions are bit-identical.
fn render_compare_test(name: &str, m: &MeshData) {
    let Some(gpu) = gpu() else { return };
    let dag = build(m);
    let up = upload(&dag);
    let r = gpu.render_compare(m, &up).expect("render comparison");
    eprintln!(
        "[gpu] {name}: covered {} px ordinary / {} px clustered, {} px differ, depth diff p99.9 {:.2e} max {:.2e}, EQUAL failures {}",
        r.covered_standard, r.covered_cluster, r.coverage_mismatch, r.depth_diff_p999, r.depth_diff_max, r.equal_failures
    );
    assert_eq!(r.validation_errors, 0);
    assert!(r.covered_standard > 2000, "{name}: nothing was drawn by the ordinary path");
    assert!(r.covered_cluster > 2000, "{name}: nothing was drawn by the cluster path");
    // The cooked mesh is quantised to 16 bits per axis and the finest level keeps every triangle.
    assert!(r.coverage_mismatch * 200 <= r.covered_standard, "{name}: coverage differs by {} px", r.coverage_mismatch);
    assert!(r.depth_diff_p999 < 1e-3, "{name}: depth differs by {}", r.depth_diff_p999);
    assert_eq!(r.equal_failures, 0, "{name}: prepass and main-pass positions are not bit-identical");
}

#[test]
fn gpu_cluster_draw_matches_the_ordinary_draw_sphere() {
    render_compare_test("sphere", &sphere(64, 128));
}

#[test]
fn gpu_cluster_draw_matches_the_ordinary_draw_torus() {
    render_compare_test("torus", &torus(200, 100));
}

#[test]
fn gpu_cluster_draw_matches_the_ordinary_draw_damaged_helmet() {
    let Some(m) = model("models/DamagedHelmet.glb") else { return };
    render_compare_test("DamagedHelmet", &m);
}

/// A real FBX whose meshes carry a second (lightmap) UV set: until builder version 2 every one of them was
/// refused for it. Their DAGs now carry no `FLAG_EXTRA`, install through the plain `validate` (no test-only
/// flag reset), and the clustered draw covers the same pixels at the same depths as the ordinary draw.
/// `ZESTENGINE_LIGHTMAP_UV_MODEL` names another model; skipped when the file is absent.
#[test]
fn gpu_cluster_draw_matches_the_ordinary_draw_on_real_meshes_with_a_second_uv_set() {
    let path = std::env::var("ZESTENGINE_LIGHTMAP_UV_MODEL").unwrap_or_else(|_| "/home/lemonxah/shared/avi/lemonnew.fbx".into());
    let Ok(scene) = crate::load_model(&path) else {
        eprintln!("[gpu] {path}: not loadable, skipped");
        return;
    };
    let mut picks: Vec<(usize, usize)> = scene
        .meshes
        .iter()
        .enumerate()
        .filter(|(_, m)| m.has_lightmap_uv && m.indices.len() / 3 >= GEO_MIN_TRIANGLES && !m.vertices.iter().any(|v| v.color != [1.0; 4]))
        .map(|(i, m)| (m.indices.len() / 3, i))
        .collect();
    assert!(!picks.is_empty(), "{path}: no DAG-sized mesh with a lightmap UV set and no colours");
    picks.sort_by(|a, b| b.cmp(a));
    let Some(gpu) = gpu() else { return };
    for &(tris, i) in picks.iter().take(4) {
        let m = &scene.meshes[i];
        // Bind pose, no skin: this checks the geometry path, the skinned path has its own tests.
        let mut vertices = m.vertices.clone();
        vertices.iter_mut().for_each(|v| v.weights = [0.0; 4]);
        let mesh = MeshData { vertices, indices: m.indices.clone(), has_lightmap_uv: m.has_lightmap_uv };
        let dag = build(&mesh);
        assert_eq!(dag.header.flags & FLAG_EXTRA, 0, "mesh {i}: a lightmap UV set must not flag the DAG");
        let up = dag.to_gpu_upload();
        up.validate().expect("the renderer takes the DAG as cooked");
        let r = gpu.render_compare(&mesh, &up).expect("render comparison");
        let (a, b) = (r.covered_standard, r.covered_cluster);
        eprintln!(
            "[gpu] {path} mesh {i} ({tris} tris, lightmap UV set): covered {a} px ordinary / {b} px clustered, {} px differ ({:.3}%), depth diff p99.9 {:.2e}, EQUAL failures {}",
            r.coverage_mismatch,
            100.0 * r.coverage_mismatch as f64 / a.max(1) as f64,
            r.depth_diff_p999,
            r.equal_failures
        );
        assert_eq!(r.validation_errors, 0);
        assert!(a > 300 && b > 300, "mesh {i}: nothing drawn ({a} / {b})");
        assert!(r.coverage_mismatch * 200 <= a, "mesh {i}: coverage differs by {} px", r.coverage_mismatch);
        assert!(r.depth_diff_p999 < 1e-3, "mesh {i}: depth differs by {}", r.depth_diff_p999);
        assert_eq!(r.equal_failures, 0, "mesh {i}: prepass and main pass positions differ");
    }
}

/// CPU properties of the XR cut (no device): the one cut judged from the centre
/// eye keeps the pixel target in EACH eye (within the IPD slack, for content at
/// least 1 m away), and the union frustum keeps every cluster either eye sees.
mod stereo_cpu {
    use super::*;
    use zestengine_render::cluster_vr::{StereoView, UnionFrustum, centre_eye, eye_separation, lod_slack, pixels_per_unit};

    fn planes(vp: Mat4) -> [glam::Vec4; 6] {
        let (r0, r1, r2, r3) = (vp.row(0), vp.row(1), vp.row(2), vp.row(3));
        let raw = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2];
        std::array::from_fn(|i| raw[i] / raw[i].truncate().length().max(1e-6))
    }

    /// A head at `pos` looking along `yaw`, eyes `ipd` apart, each canted outward
    /// by `cant` radians, with slightly different (asymmetric) projections.
    fn eyes(pos: Vec3, yaw: f32, ipd: f32, cant: f32) -> StereoView {
        let head = Mat4::from_rotation_translation(Quat::from_rotation_y(yaw), pos);
        let eye = |side: f32| {
            let world = head * Mat4::from_rotation_translation(Quat::from_rotation_y(-side * cant), Vec3::new(side * ipd * 0.5, 0.0, 0.0));
            world.inverse()
        };
        let proj = |l: f32, r: f32, u: f32, d: f32, n: f32, f: f32| {
            let (w, h) = (r - l, u - d);
            Mat4::from_cols_array(&[2.0 / w, 0.0, 0.0, 0.0, 0.0, 2.0 / h, 0.0, 0.0, (r + l) / w, (u + d) / h, -f / (f - n), -1.0, 0.0, 0.0, -(f * n) / (f - n), 0.0])
        };
        StereoView { view: [eye(-1.0), eye(1.0)], proj: [proj(-1.3, 0.9, 1.1, -1.2, 0.05, 500.0), proj(-0.9, 1.3, 1.1, -1.2, 0.05, 500.0)] }
    }

    #[test]
    fn the_centre_eye_cut_keeps_the_error_target_in_both_eyes() {
        let dag = build(&torus(200, 100));
        let height = 1832.0;
        let (mut worst, mut checked) = (0.0f32, 0usize);
        let mut rows = Vec::new();
        for (distance, yaw, lateral) in [(1.5f32, 0.0f32, 0.0f32), (3.0, 0.2, 0.5), (8.0, -0.3, 1.0), (20.0, 0.0, 0.0), (60.0, 0.4, 3.0)] {
            let sv = eyes(Vec3::new(0.0, 0.0, 0.0), yaw, 0.064, 0.05);
            // The torus is ~12 units across: 0.1 makes it a person-sized object whose cut changes with distance.
            let s = 0.1f32;
            let model = Mat4::from_translation(Vec3::new(lateral, 0.0, -distance)) * Mat4::from_scale(Vec3::splat(s));
            let centre = centre_eye(&sv).inverse().w_axis.truncate();
            let ppu_wide = pixels_per_unit(&sv, height);
            let target = 1.0 / (1.0 + lod_slack(eye_separation(&sv), 1.0));
            let ok_from = |pos: Vec3, ppu: f32, px: f32| {
                move |err: f32, sph: [f32; 4]| {
                    let c = model.transform_point3(Vec3::new(sph[0], sph[1], sph[2]));
                    let d = (c.distance(pos) - sph[3] * s).max(0.0);
                    err * s * ppu <= px * d
                }
            };
            let cut = dag.select_linear(&ok_from(centre, ppu_wide, target));
            // The same cut with the slack the true per-distance optimum uses:
            // production tightens with the IPD slack at 1 m (conservative for
            // all content), which can cross a LOD boundary one level early
            // when a cluster's projected error sits right on it.
            let target_far = 1.0 / (1.0 + lod_slack(eye_separation(&sv), distance));
            let tris_far = dag.triangle_count_of(&dag.select_linear(&ok_from(centre, ppu_wide, target_far)));
            // Each eye's own cut at the plain 1 px target, for comparison.
            let own: Vec<usize> = (0..2)
                .map(|e| {
                    let pos = sv.view[e].inverse().w_axis.truncate();
                    dag.triangle_count_of(&dag.select_linear(&ok_from(pos, 0.5 * height * sv.proj[e].y_axis.y.abs(), 1.0)))
                })
                .collect();
            let tris = dag.triangle_count_of(&cut);
            for e in 0..2 {
                let pos = sv.view[e].inverse().w_axis.truncate();
                let ppu = 0.5 * height * sv.proj[e].y_axis.y.abs();
                for &c in &cut {
                    let cl = &dag.clusters[c as usize];
                    let sph = cl.lod_sphere;
                    let cw = model.transform_point3(Vec3::new(sph[0], sph[1], sph[2]));
                    // Content closer than 1 m has a larger slack (documented); skip it.
                    if cw.distance(centre) - sph[3] * s < 1.0 {
                        continue;
                    }
                    let d = (cw.distance(pos) - sph[3] * s).max(1e-6);
                    let px = cl.lod_error * s * ppu / d;
                    worst = worst.max(px);
                    checked += 1;
                    assert!(px <= 1.01, "distance {distance}: eye {e} sees cluster {c} at {px} px (target 1)");
                }
            }
            rows.push((distance, tris, tris_far, own[0], own[1]));
        }
        eprintln!("[cpu] centre-eye cut, worst projected error in either eye {worst:.4} px over {checked} clusters (target 1 px, 64 mm IPD, 4 degrees of cant)");
        for (d, c, cf, l, r) in rows {
            eprintln!("[cpu]   {d:>5} m: conservative centre cut {c} triangles, per-distance centre cut {cf}, left eye's own cut {l}, right {r}");
            // When a cluster's projected error sits on the LOD boundary, the
            // centre eye's slightly different position (and the conservative
            // slack) can push the shared cut one level finer - a level halves
            // the triangles, so at most 2x. The centre cut must never cost
            // more than one level over either eye's own cut.
            let bound = l.max(r) as f32 * 2.0 + 64.0;
            assert!(c as f32 <= bound, "{d} m: the conservative centre cut is more than one LOD level finer than a single eye's");
            assert!(cf as f32 <= bound, "{d} m: the per-distance centre cut is more than one LOD level finer than a single eye's");
            // Off the knife-edge (the near and mid rows) the shared cut should
            // match an eye's own cut closely.
            if d <= 8.0 {
                assert!(c as f32 <= l.max(r) as f32 * 1.35 + 64.0, "{d} m: the shared cut is much bigger than a single eye's");
            }
        }
        assert!(checked > 100);
    }

    #[test]
    fn the_union_frustum_keeps_every_cluster_either_eye_sees() {
        let dag = build(&torus(120, 60));
        let mut seen_by_eye = 0usize;
        for (yaw, lateral, depth, cant) in [(0.0f32, 0.0f32, 4.0f32, 0.0f32), (0.3, 1.5, 3.0, 0.05), (-0.5, -2.0, 6.0, 0.1), (1.0, 0.0, 2.0, 0.0), (0.0, 4.0, 5.0, 0.07)] {
            let sv = eyes(Vec3::ZERO, yaw, 0.064, cant);
            let union = UnionFrustum::from_eyes(&sv);
            let model = Mat4::from_translation(Vec3::new(lateral, 0.2, -depth));
            let eye_planes = [planes(sv.proj[0] * sv.view[0]), planes(sv.proj[1] * sv.view[1])];
            for cl in &dag.clusters {
                let c = model.transform_point3(Vec3::new(cl.bounds[0], cl.bounds[1], cl.bounds[2]));
                let r = cl.bounds[3];
                let inside = |pl: &[glam::Vec4; 6]| pl.iter().all(|p| p.truncate().dot(c) + p.w >= -r);
                if eye_planes.iter().any(inside) {
                    seen_by_eye += 1;
                    assert!(!union.outside(c, r), "a cluster an eye sees is outside the union frustum (yaw {yaw}, lateral {lateral})");
                }
            }
        }
        eprintln!("[cpu] union frustum: {seen_by_eye} cluster-views seen by an eye, none outside the union");
        assert!(seen_by_eye > 500);
    }
}

/// VR without a headset (GEO-5): one cull from the centre eye with the union
/// frustum, drawn once by the multiview clustered vertex shader into a two-layer
/// target, against the ordinary mesh rendered per eye with that eye's matrices.
/// Silhouettes must agree in BOTH layers (the finest level, so nothing but the
/// pipeline is compared), and a realistic 1 px cut must stay close.
fn stereo_compare_test(name: &str, m: &MeshData) {
    let Some(gpu) = gpu() else { return };
    let dag = build(m);
    let up = upload(&dag);
    let Some(exact) = gpu.render_stereo_compare(m, &up, 1e-6).expect("stereo comparison") else {
        eprintln!("[gpu] {name}: this device has no multiview: stereo comparison skipped");
        return;
    };
    eprintln!(
        "[gpu] {name} multiview, finest cut: covered L {} / R {} px ordinary, L {} / R {} px clustered, mismatch L {} / R {}, colour written L {} / R {}, {} px differ between the layers, {} clusters {} triangles from one cut",
        exact.covered_standard[0], exact.covered_standard[1], exact.covered_cluster[0], exact.covered_cluster[1], exact.mismatch[0], exact.mismatch[1],
        exact.colour_written[0], exact.colour_written[1], exact.layers_differ, exact.visible_clusters, exact.triangles
    );
    assert_eq!(exact.validation_errors, 0);
    for e in 0..2 {
        assert!(exact.covered_standard[e] > 1000 && exact.covered_cluster[e] > 1000, "{name}: eye {e} drew nothing");
        assert!(exact.mismatch[e] * 100 <= exact.covered_standard[e], "{name}: eye {e} silhouette differs by {} px of {}", exact.mismatch[e], exact.covered_standard[e]);
        assert!(exact.colour_written[e] > 1000, "{name}: the multiview pass wrote no colour in layer {e}");
    }
    // The two eyes see different pictures from the one list (the layers are not copies of each other).
    assert!(exact.layers_differ > 50, "{name}: the two layers are (nearly) the same picture ({} px differ)", exact.layers_differ);
    let coarse = gpu.render_stereo_compare(m, &up, 1.0).unwrap().unwrap();
    eprintln!(
        "[gpu] {name} multiview, 1 px cut: covered L {} / R {} px, mismatch L {} / R {}, {} triangles (finest {})",
        coarse.covered_cluster[0], coarse.covered_cluster[1], coarse.mismatch[0], coarse.mismatch[1], coarse.triangles, exact.triangles
    );
    for e in 0..2 {
        assert!(coarse.mismatch[e] * 25 <= coarse.covered_standard[e], "{name}: eye {e} 1 px cut differs by {} px of {}", coarse.mismatch[e], coarse.covered_standard[e]);
    }
    assert!(coarse.triangles <= exact.triangles);
    // A deliberately coarse cut (16 px) really simplifies, in both eyes' picture from the one list.
    let rough = gpu.render_stereo_compare(m, &up, 16.0).unwrap().unwrap();
    eprintln!("[gpu] {name} multiview, 16 px cut: {} triangles (finest {}), mismatch L {} / R {} of {} px", rough.triangles, exact.triangles, rough.mismatch[0], rough.mismatch[1], exact.covered_standard[0]);
    assert!(rough.triangles < exact.triangles, "{name}: a 16 px target did not simplify");
    for e in 0..2 {
        assert!(rough.covered_cluster[e] > 1000 && rough.mismatch[e] * 5 <= exact.covered_standard[e], "{name}: eye {e} rough cut differs by {} px", rough.mismatch[e]);
    }
}

#[test]
fn gpu_multiview_cluster_draw_matches_the_per_eye_ordinary_draw_torus() {
    stereo_compare_test("torus", &torus(200, 100));
}

#[test]
fn gpu_multiview_cluster_draw_matches_the_per_eye_ordinary_draw_sphere() {
    stereo_compare_test("sphere", &sphere(64, 128));
}

// ------------------------------------------------------------------------
// Skinned clusters (GEO-5): cluster_skin.comp against the CPU reference.

use zestengine_render::cluster_headless::HeadlessSkin;
use zestengine_render::cluster_skin::{self, NO_SLOT, SkinPose, ctl, reference};

use super::tests::humanoid;

fn rand_palette(rng: &mut Rng, joints: usize, max_angle: f32, shift: f32) -> Vec<Mat4> {
    (0..joints)
        .map(|_| {
            let axis = Vec3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)).normalize_or_zero();
            let axis = if axis == Vec3::ZERO { Vec3::X } else { axis };
            let t = Vec3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)) * shift;
            Mat4::from_translation(t) * Mat4::from_axis_angle(axis, rng.range(-max_angle, max_angle))
        })
        .collect()
}

fn pose_of(sk: &HeadlessSkin, margin: f32) -> SkinPose<'_> {
    SkinPose { palette: &sk.palette, prev_palette: sk.prev_palette.as_deref(), weights: &sk.weights, prev_weights: sk.prev_weights.as_deref(), margin }
}

/// A skinned tube with blendshapes and its DAG.
fn tube_with_morphs(seed: u64) -> (MeshData, Vec<crate::MorphTarget>, ClusterDag, usize) {
    let m = humanoid(300, 64, 40);
    let mut rng = Rng(seed);
    let mut morphs = Vec::new();
    for t in 0..3 {
        let mut deltas = Vec::new();
        for (v, vert) in m.vertices.iter().enumerate() {
            let y = vert.position[1];
            if y > 0.3 * t as f32 && y < 0.3 * t as f32 + 1.2 && rng.f() > 0.2 {
                deltas.push((v as u32, [rng.range(-0.1, 0.1), rng.range(-0.1, 0.1), rng.range(-0.1, 0.1)], [rng.range(-0.3, 0.3), 0.0, rng.range(-0.3, 0.3)]));
            }
        }
        morphs.push(crate::MorphTarget { name: format!("shape{t}"), deltas });
    }
    let dag = build_cluster_dag(&m, &morphs, &DagOptions::default()).unwrap();
    dag.validate().unwrap();
    assert!(dag.header.flags & FLAG_SKINNED != 0 && dag.header.morph_targets == 3);
    let joints = 40;
    (m, morphs, dag, joints)
}

/// Posed vertices of every listed skinned cluster against the reference.
/// Returns (clusters compared, vertices compared, clusters dropped, worst error).
fn check_posed(name: &str, ups: &[&ClusterMeshUpload], meshes: &[usize], poses: &[Option<HeadlessSkin>], run: &HeadlessRun) -> (usize, usize, usize, f32) {
    assert!(run.out_of_range_writes.is_empty(), "{name}: {:?}", run.out_of_range_writes);
    assert_eq!(run.validation_errors, 0, "{name}: validation errors");
    let (mut clusters, mut verts, mut dropped, mut worst) = (0usize, 0usize, 0usize, 0f32);
    for (b, bases) in run.region_bases.iter().enumerate() {
        for (late, list) in [(0usize, &run.early[b]), (1usize, &run.late[b])] {
            for (k, &(i, c)) in list.iter().enumerate() {
                let slot = run.skin_slots[bases[late] as usize + k];
                let Some(sk) = &poses[i as usize] else {
                    assert_eq!(slot, NO_SLOT, "{name}: a static instance was given pool space");
                    continue;
                };
                if slot == NO_SLOT {
                    dropped += 1;
                    continue;
                }
                let up = ups[meshes[i as usize]];
                let sph = mesh_sphere(up);
                let margin = sk.margin.unwrap_or_else(|| cluster_skin::pose_margin([sph[0], sph[1], sph[2]], sph[3], &sk.palette, &sk.weights, &up.morph_reach()));
                let want = reference::pose_cluster(up, c as usize, &pose_of(sk, margin));
                for (t, w) in want.iter().enumerate() {
                    let at = (slot as usize + t) * 16;
                    let g = &run.skin_pool[at..at + 16];
                    for (what, got, exp) in [("pos", &g[0..3], w.pos), ("normal", &g[4..7], w.nrm), ("tangent", &g[8..11], w.tan), ("prev", &g[12..15], w.prev)] {
                        for a in 0..3 {
                            let err = (got[a] - exp[a]).abs();
                            worst = worst.max(err);
                            assert!(err <= 2e-4 + 2e-5 * exp[a].abs(), "{name}: instance {i} cluster {c} vertex {t} {what}[{a}]: gpu {} cpu {}", got[a], exp[a]);
                        }
                    }
                }
                clusters += 1;
                verts += want.len();
            }
        }
    }
    (clusters, verts, dropped, worst)
}

fn look_view(sph: [f32; 4], dist: f32, dir: Vec3) -> (HeadlessView, cluster_ref::RefView, Mat4) {
    let centre = Vec3::new(sph[0], sph[1], sph[2]);
    let r = sph[3];
    let eye = centre + dir.normalize() * r * dist;
    let (near, far) = ((r * 0.02).max(0.01), r * 100.0);
    let rv = cluster_ref::make_view(eye, centre, 1.0, 1.5, 1080.0, near, far, 1.0);
    let view_proj = Mat4::perspective_rh(1.0, 1.5, near, far) * Mat4::look_at_rh(eye, centre, Vec3::Y);
    (HeadlessView { view_proj, cam: eye, pixels_per_unit: rv.pixels_per_unit, target_px: 1.0 }, rv, view_proj)
}

/// Extreme poses plus blendshapes on a skinned tube (and one static mesh in the
/// same run): every visible cluster's posed vertices equal the CPU reference,
/// the static instance takes no pool space.
#[test]
fn gpu_skinned_tube_matches_cpu_reference() {
    let Some(gpu) = gpu() else { return };
    let (_m, _morphs, dag, joints) = tube_with_morphs(7);
    let tube = upload(&dag);
    let ball = upload(&build(&sphere(40, 80)));
    let ups = [&tube, &ball];
    let sph = mesh_sphere(&tube);
    let mut rng = Rng(0xBADC0DE);
    let (mut clusters, mut verts, mut worst) = (0, 0, 0f32);
    for round in 0..6 {
        let angle = if round < 3 { 2.6 } else { 0.4 };
        let mut insts = Vec::new();
        let mut poses = Vec::new();
        let mut meshes = Vec::new();
        for n in 0..3 {
            let model = Mat4::from_translation(Vec3::new(n as f32 * 0.3 - 0.3, 0.0, 0.0));
            let palette = rand_palette(&mut rng, joints, angle, 0.4);
            let prev = rand_palette(&mut rng, joints, angle, 0.4);
            let weights = vec![rng.range(0.0, 1.0), rng.range(-0.5, 0.5), rng.range(0.0, 1.0)];
            // Half the rounds move the blendshapes between frames: the previous position morphs with last frame's weights.
            let prev_weights = (round % 2 == 0).then(|| vec![rng.range(0.0, 1.0), rng.range(-0.5, 0.5), rng.range(0.0, 1.0)]);
            let sk = HeadlessSkin { palette, prev_palette: Some(prev), weights, prev_weights, margin: None };
            insts.push(HeadlessInstance { mesh: 0, model, batch: n % 2, skin: Some(sk.clone()), lod_scale: 1.0 });
            poses.push(Some(sk));
            meshes.push(0);
        }
        insts.push(HeadlessInstance { mesh: 1, model: Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)), batch: 0, skin: None, lod_scale: 1.0 });
        poses.push(None);
        meshes.push(1);
        let (hv, _rv, vp) = look_view(sph, rng.range(0.9, 4.0), Vec3::new(rng.range(-1.0, 1.0), 0.3, 1.0));
        let _ = vp;
        let run = gpu
            .run(&[tube.clone(), ball.clone()], &insts, 2, &hv, HeadlessHiz::Off, &HeadlessOptions::default())
            .expect("device run");
        assert_eq!(run.counters[counter::FLAGS], 0);
        assert_eq!(run.skin_ctl[ctl::FLAGS], 0, "skin flags");
        let (c, v, dropped, w) = check_posed("tube", &ups, &meshes, &poses, &run);
        assert_eq!(dropped, 0);
        assert_eq!(run.skin_ctl[ctl::SKINNED] as usize, c, "every skinned entry was posed once");
        clusters += c;
        verts += v;
        worst = worst.max(w);
    }
    eprintln!("[gpu] skinned tube: {clusters} clusters / {verts} vertices equal the CPU reference (worst error {worst:.2e})");
    assert!(clusters > 100);
}

/// Blendshape motion reaches the previous position: with the same palette both
/// frames, a vertex a shape moves between last frame's weights and this frame's
/// has `prev != pos` (and equals the CPU reference); with no previous weights
/// it has `prev == pos`.
#[test]
fn gpu_blendshape_motion_reaches_the_previous_position() {
    let Some(gpu) = gpu() else { return };
    let (_m, _morphs, dag, joints) = tube_with_morphs(21);
    let tube = upload(&dag);
    let sph = mesh_sphere(&tube);
    let palette = vec![Mat4::IDENTITY; joints];
    let mut moved_total = 0usize;
    for (label, prev_weights) in [("moving", Some(vec![0.0, 0.0, 0.0])), ("still", None)] {
        let sk = HeadlessSkin { palette: palette.clone(), prev_palette: Some(palette.clone()), weights: vec![0.9, 0.7, 0.5], prev_weights, margin: Some(1.0) };
        let (hv, _, _) = look_view(sph, 3.0, Vec3::new(0.2, 0.2, 1.0));
        let inst = HeadlessInstance { mesh: 0, model: Mat4::IDENTITY, batch: 0, skin: Some(sk.clone()), lod_scale: 1.0 };
        let run = gpu.run(std::slice::from_ref(&tube), &[inst], 1, &hv, HeadlessHiz::Off, &HeadlessOptions::default()).expect("device run");
        let (clusters, verts, dropped, worst) = check_posed(label, &[&tube], &[0], &[Some(sk.clone())], &run);
        assert_eq!(dropped, 0);
        // Count vertices whose previous position differs from the current one.
        let mut moved = 0usize;
        let mut total = 0usize;
        for (b, bases) in run.region_bases.iter().enumerate() {
            for (late, list) in [(0usize, &run.early[b]), (1usize, &run.late[b])] {
                for (k, &(_, c)) in list.iter().enumerate() {
                    let slot = run.skin_slots[bases[late] as usize + k];
                    if slot == NO_SLOT {
                        continue;
                    }
                    let n = tube.clusters[c as usize].vertex_count as usize;
                    for t in 0..n {
                        let g = &run.skin_pool[(slot as usize + t) * 16..][..16];
                        total += 1;
                        if (0..3).any(|a| (g[a] - g[12 + a]).abs() > 1e-6) {
                            moved += 1;
                        }
                    }
                }
            }
        }
        eprintln!("[gpu] blendshape motion ({label}): {clusters} clusters / {verts} vertices equal the CPU reference (worst {worst:.2e}); {moved} of {total} pool vertices have prev != pos");
        if label == "moving" {
            assert!(moved > 50, "blendshape motion never reached the previous position ({moved} of {total})");
            moved_total = moved;
        } else {
            assert_eq!(moved, 0, "no previous weights must mean no blendshape motion");
        }
    }
    assert!(moved_total > 0);
}

/// Culling on bind-pose bounds expanded by the pose margin never loses a
/// cluster whose posed surface is on screen.
#[test]
fn gpu_skinned_cull_is_conservative_for_extreme_poses() {
    let Some(gpu) = gpu() else { return };
    let (_m, _morphs, dag, joints) = tube_with_morphs(11);
    let tube = upload(&dag);
    let sph = mesh_sphere(&tube);
    let mut rng = Rng(77);
    let (mut must, mut missed, mut seen_by_margin) = (0usize, 0usize, 0usize);
    for _ in 0..8 {
        let model = Mat4::from_translation(Vec3::new(rng.range(-1.0, 1.0), rng.range(-0.5, 0.5), rng.range(-1.0, 1.0)));
        let sk = HeadlessSkin {
            palette: rand_palette(&mut rng, joints, 2.6, 0.6),
            prev_palette: None,
            weights: vec![rng.range(0.0, 1.0), 0.0, rng.range(0.0, 1.0)],
            prev_weights: None,
            margin: None,
        };
        // A narrow camera so much of the posed mesh lands outside the frustum.
        let centre = Vec3::new(sph[0], sph[1], sph[2]);
        let eye = centre + Vec3::new(rng.range(-1.0, 1.0), 0.2, 1.0).normalize() * sph[3] * rng.range(1.5, 4.0);
        let target = centre + Vec3::new(rng.range(-0.5, 0.5), rng.range(-0.5, 0.5), 0.0) * sph[3];
        let (near, far) = (sph[3] * 0.02, sph[3] * 100.0);
        let rv = cluster_ref::make_view(eye, target, 0.5, 1.5, 1080.0, near, far, 1.0);
        let vp = Mat4::perspective_rh(0.5, 1.5, near, far) * Mat4::look_at_rh(eye, target, Vec3::Y);
        let hv = HeadlessView { view_proj: vp, cam: eye, pixels_per_unit: rv.pixels_per_unit, target_px: 1.0 };
        let run = gpu
            .run(std::slice::from_ref(&tube), &[HeadlessInstance { mesh: 0, model, batch: 0, skin: Some(sk.clone()), lod_scale: 1.0 }], 1, &hv, HeadlessHiz::Off, &HeadlessOptions::default())
            .expect("device run");
        assert_eq!(run.counters[counter::FLAGS], 0);
        let got: HashSet<u32> = run.early[0].iter().map(|e| e.1).collect();
        // The LOD cut is judged on the bind pose; take it without any culling.
        let m_arr = model.to_cols_array_2d();
        let s = cluster_ref::max_scale(&m_arr);
        let cut = dag.select_bvh(&|err, sp| cluster_ref::lod_ok(&rv, &m_arr, s, err, sp));
        let margin = cluster_skin::pose_margin([sph[0], sph[1], sph[2]], sph[3], &sk.palette, &sk.weights, &tube.morph_reach());
        for c in cut {
            let posed = reference::pose_cluster(&tube, c as usize, &pose_of(&sk, margin));
            let on_screen = posed.iter().any(|p| {
                let clip = vp * model * Vec3::from(p.pos).extend(1.0);
                let w = clip.w * 0.999;
                clip.w > near && clip.x.abs() < w && clip.y.abs() < w && clip.z > 0.0 && clip.z < w
            });
            if on_screen {
                must += 1;
                if !got.contains(&c) {
                    missed += 1;
                }
            }
            // Would the bind-pose bounds alone have lost it?
            let cl: ClusterRec = bytemuck::cast(dag.clusters[c as usize]);
            let ctr = cluster_ref::transform_point(&m_arr, [cl.bounds[0], cl.bounds[1], cl.bounds[2]]);
            if on_screen && cluster_ref::outside_frustum(&rv, ctr, cl.bounds[3] * s) {
                seen_by_margin += 1;
            }
        }
    }
    eprintln!("[gpu] skinned cull: {must} on-screen posed clusters, {missed} missed; {seen_by_margin} would be lost without the pose margin");
    assert!(must > 200);
    assert_eq!(missed, 0, "the pose margin must keep every on-screen posed cluster");
}

/// A pool smaller than the visible clusters needs: the clusters that did not fit are
/// dropped (slot NONE, flagged), nothing is written past the reservation, and the
/// ones that fit are still right.
#[test]
fn gpu_skin_pool_overflow_is_handled() {
    let Some(gpu) = gpu() else { return };
    let (_m, _morphs, dag, joints) = tube_with_morphs(3);
    let tube = upload(&dag);
    let sph = mesh_sphere(&tube);
    let mut rng = Rng(5);
    let sk = HeadlessSkin { palette: rand_palette(&mut rng, joints, 1.0, 0.2), prev_palette: None, weights: vec![0.5, 0.5, 0.5], prev_weights: None, margin: None };
    let insts: Vec<_> = (0..3)
        .map(|n| HeadlessInstance { mesh: 0, model: Mat4::from_translation(Vec3::new(n as f32 * 0.4, 0.0, 0.0)), batch: 0, skin: Some(sk.clone()), lod_scale: 1.0 })
        .collect();
    let (hv, _, _) = look_view(sph, 2.5, Vec3::new(0.2, 0.2, 1.0));
    let opts = HeadlessOptions { skin_pool_vertices: Some(1500), ..Default::default() };
    let run = gpu.run(std::slice::from_ref(&tube), &insts, 1, &hv, HeadlessHiz::Off, &opts).expect("device run");
    let poses = vec![Some(sk); 3];
    let (clusters, _verts, dropped, _) = check_posed("overflow", &[&tube], &[0, 0, 0], &poses, &run);
    eprintln!("[gpu] skin pool overflow: {clusters} clusters posed, {dropped} dropped, {} vertices wanted of 1500", run.skin_ctl[ctl::USED]);
    assert!(dropped > 0 && clusters > 0);
    assert_ne!(run.skin_ctl[ctl::FLAGS] & ctl::FLAG_POOL_FULL, 0, "overflow is flagged");
    assert!(run.skin_ctl[ctl::USED] > 1500);
}

/// The built-in avatar (a real skinned model, blendshapes when it has any).
fn avatar_skinned_test(rel: &str) {
    let Some(gpu) = gpu() else { return };
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel);
    if !path.exists() {
        eprintln!("[geo] {rel} not in the repo: skipped");
        return;
    }
    let scene = crate::load_gltf(&path).expect("load avatar");
    let Some((mi, mesh)) = scene.meshes.iter().enumerate().max_by_key(|(_, m)| m.indices.len()) else { return };
    let joints = mesh.vertices.iter().flat_map(|v| v.joints.iter().zip(v.weights)).filter(|(_, w)| *w > 0.0).map(|(j, _)| *j as usize + 1).max().unwrap_or(0);
    if joints == 0 || mesh.indices.len() / 3 < 2000 {
        eprintln!("[geo] {rel}: not a skinned mesh of useful size: skipped");
        return;
    }
    let mut morphs: Vec<crate::MorphTarget> = scene.morphs.get(mi).cloned().unwrap_or_default();
    morphs.truncate(64);
    let synthetic = morphs.is_empty();
    if synthetic {
        // No authored blendshapes: three synthetic ones over the head-high vertices.
        let top = mesh.vertices.iter().map(|v| v.position[1]).fold(f32::MIN, f32::max);
        for t in 0..3 {
            let deltas = mesh.vertices.iter().enumerate().filter(|(v, vert)| vert.position[1] > top * 0.7 && v % (t + 2) == 0).map(|(v, _)| (v as u32, [0.01, 0.0, 0.01 * t as f32], [0.0, 0.1, 0.0])).collect();
            morphs.push(crate::MorphTarget { name: format!("syn{t}"), deltas });
        }
    }
    let dag = build_cluster_dag(mesh, &morphs, &DagOptions::default()).unwrap();
    dag.validate().unwrap();
    assert!(dag.header.flags & FLAG_SKINNED != 0, "{rel} should carry skin data");
    let up = upload(&dag);
    let sph = mesh_sphere(&up);
    let mut rng = Rng(0xA7A7);
    let targets = morphs.len();
    let (mut clusters, mut verts, mut worst) = (0, 0, 0f32);
    for round in 0..5 {
        let weights: Vec<f32> = (0..targets).map(|_| if rng.f() < 0.3 { rng.range(0.0, 1.0) } else { 0.0 }).collect();
        let sk = HeadlessSkin {
            palette: rand_palette(&mut rng, joints, if round < 2 { 1.2 } else { 0.5 }, 0.05),
            prev_palette: Some(rand_palette(&mut rng, joints, 0.5, 0.05)),
            weights,
            prev_weights: None,
            margin: None,
        };
        let (hv, _, _) = look_view(sph, [1.2, 2.0, 3.0, 6.0, 12.0][round], Vec3::new(0.3, 0.2, 1.0));
        let inst = HeadlessInstance { mesh: 0, model: Mat4::IDENTITY, batch: 0, skin: Some(sk.clone()), lod_scale: 1.0 };
        let run = gpu.run(std::slice::from_ref(&up), &[inst], 1, &hv, HeadlessHiz::Off, &HeadlessOptions::default()).expect("device run");
        assert_eq!(run.counters[counter::FLAGS], 0);
        assert_eq!(run.skin_ctl[ctl::FLAGS], 0);
        let (c, v, dropped, w) = check_posed(rel, &[&up], &[0], &[Some(sk)], &run);
        assert_eq!(dropped, 0);
        clusters += c;
        verts += v;
        worst = worst.max(w);
    }
    eprintln!(
        "[gpu] {rel}: {joints} joints, {targets} blendshapes ({}), {clusters} clusters / {verts} vertices equal the CPU reference (worst error {worst:.2e})",
        if synthetic { "synthetic" } else { "authored" }
    );
    assert!(clusters > 50);
}

#[test]
fn gpu_skinned_clusters_male_avatar() {
    avatar_skinned_test("models/avatars/default/male.glb");
}

#[test]
fn gpu_skinned_clusters_female_avatar() {
    avatar_skinned_test("models/avatars/default/female.glb");
}

/// Identity pose without blendshapes poses every vertex to its decoded bind position (exact decode).
#[test]
fn gpu_skinned_identity_pose_is_the_bind_pose() {
    let Some(gpu) = gpu() else { return };
    let (_m, _morphs, dag, joints) = tube_with_morphs(21);
    let tube = upload(&dag);
    let sph = mesh_sphere(&tube);
    let sk = HeadlessSkin { palette: vec![Mat4::IDENTITY; joints], prev_palette: None, weights: vec![0.0; 3], prev_weights: None, margin: Some(0.0) };
    let (hv, _, _) = look_view(sph, 2.0, Vec3::new(0.2, 0.2, 1.0));
    let inst = HeadlessInstance { mesh: 0, model: Mat4::IDENTITY, batch: 0, skin: Some(sk), lod_scale: 1.0 };
    let run = gpu.run(std::slice::from_ref(&tube), &[inst], 1, &hv, HeadlessHiz::Off, &HeadlessOptions::default()).expect("device run");
    assert!(run.out_of_range_writes.is_empty() && run.validation_errors == 0);
    let mut worst = 0f32;
    let mut n = 0;
    for (k, &(_, c)) in run.early[0].iter().enumerate() {
        let slot = run.skin_slots[run.region_bases[0][0] as usize + k];
        assert_ne!(slot, NO_SLOT);
        let bind = dag.cluster_positions(c);
        for (t, p) in bind.iter().enumerate() {
            let g = &run.skin_pool[(slot as usize + t) * 16..][..3];
            for a in 0..3 {
                worst = worst.max((g[a] - p[a]).abs());
            }
            n += 1;
        }
    }
    eprintln!("[gpu] identity pose: {n} vertices, worst deviation from the decoded bind position {worst:.2e}");
    assert!(n > 1000 && worst < 1e-5);
}

/// CPU-posed copy of a skinned mesh (blendshapes first, then linear blend), the
/// original the cluster draw is compared with.
fn pose_mesh(m: &MeshData, morphs: &[crate::MorphTarget], sk: &HeadlessSkin) -> MeshData {
    let mut out = MeshData { vertices: m.vertices.clone(), indices: m.indices.clone(), has_lightmap_uv: m.has_lightmap_uv };
    for (t, mt) in morphs.iter().enumerate() {
        let w = sk.weights.get(t).copied().unwrap_or(0.0);
        if w.abs() <= 1e-4 {
            continue;
        }
        for &(v, dp, dn) in &mt.deltas {
            let vert = &mut out.vertices[v as usize];
            for a in 0..3 {
                vert.position[a] += dp[a] * w;
                vert.normal[a] += dn[a] * w;
            }
        }
    }
    for v in &mut out.vertices {
        let sum: f32 = v.weights.iter().sum();
        if sum <= 1e-5 {
            continue;
        }
        let mut acc = Mat4::ZERO;
        for k in 0..4 {
            if v.weights[k] > 0.0 && (v.joints[k] as usize) < sk.palette.len() {
                acc += sk.palette[v.joints[k] as usize] * v.weights[k];
            }
        }
        acc *= 1.0 / sum;
        v.position = acc.transform_point3(Vec3::from(v.position)).to_array();
        v.normal = (glam::Mat3::from_mat4(acc) * Vec3::from(v.normal)).normalize_or_zero().to_array();
    }
    out
}

/// The skinned cluster draw covers the same pixels at the same depth as the
/// ordinary draw of the CPU-posed mesh, and the depth-EQUAL second draw (the
/// main pass) passes everywhere the prepass wrote: both passes read one pool.
#[test]
fn gpu_skinned_cluster_draw_matches_the_posed_mesh() {
    let Some(gpu) = gpu() else { return };
    let (m, mts, dag, joints) = tube_with_morphs(31);
    let up = upload(&dag);
    let mut prng = Rng(99);
    for round in 0..3 {
        let sk = HeadlessSkin {
            palette: rand_palette(&mut prng, joints, 0.35 + 0.1 * round as f32, 0.02),
            prev_palette: None,
            weights: vec![0.6, 0.3, 0.9],
            prev_weights: None,
            margin: None,
        };
        let posed = pose_mesh(&m, &mts, &sk);
        let r = gpu.render_compare_posed(&posed, &up, Some(&sk)).expect("render comparison");
        eprintln!(
            "[gpu] skinned tube round {round}: covered {} px ordinary / {} px clustered, {} px differ, depth diff p99.9 {:.2e} max {:.2e}, EQUAL failures {}",
            r.covered_standard, r.covered_cluster, r.coverage_mismatch, r.depth_diff_p999, r.depth_diff_max, r.equal_failures
        );
        assert_eq!(r.validation_errors, 0);
        assert!(r.covered_standard > 1000 && r.covered_cluster > 1000);
        assert!(r.coverage_mismatch * 100 <= r.covered_standard, "coverage differs by {} px", r.coverage_mismatch);
        assert!(r.depth_diff_p999 < 2e-3, "depth differs by {}", r.depth_diff_p999);
        assert_eq!(r.equal_failures, 0, "prepass and main-pass skinned positions are not bit-identical");
    }
}

/// The per-instance error scale (triangle budget / fair share) coarsens the
/// cut exactly as the reference with a larger pixel target, instance by
/// instance (two instances of one mesh with different scales in one run).
#[test]
fn gpu_instance_lod_scale_coarsens_the_cut() {
    let Some(gpu) = gpu() else { return };
    let m = torus(200, 100);
    let dag = build(&m);
    let up = upload(&dag);
    let sph = mesh_sphere(&up);
    let (hv, rv, _) = look_view(sph, 14.0, Vec3::new(0.3, 0.4, 1.0));
    let tris = |ids: &[u32]| ids.iter().map(|&c| dag.clusters[c as usize].triangle_count as u64).sum::<u64>();
    let mut prev = u64::MAX;
    let mut first = 0u64;
    let (mut total, mut wrong) = (0usize, 0usize);
    for scale in [1.0f32, 2.0, 4.0, 8.0, 16.0, 1000.0] {
        let insts = [
            HeadlessInstance { mesh: 0, model: Mat4::IDENTITY, batch: 0, skin: None, lod_scale: scale },
            HeadlessInstance { mesh: 0, model: Mat4::from_translation(Vec3::new(0.0, 0.0, -0.1)), batch: 1, skin: None, lod_scale: 1.0 },
        ];
        let run = gpu.run(std::slice::from_ref(&up), &insts, 2, &hv, HeadlessHiz::Off, &HeadlessOptions::default()).expect("device run");
        assert!(run.out_of_range_writes.is_empty() && run.validation_errors == 0);
        let mut want_rv = rv;
        want_rv.target_px = rv.target_px * scale;
        let inst0 = RefInstance { model: Mat4::IDENTITY.to_cols_array_2d(), eye: cluster_ref::object_space_eye(Mat4::IDENTITY, hv.cam) };
        let (want, _) = cluster_ref::cut(&up.clusters, &up.nodes, &want_rv, &inst0, true);
        let mut got: Vec<u32> = run.early[0].iter().map(|e| e.1).collect();
        got.sort_unstable();
        total += want.len();
        wrong += want.iter().filter(|c| got.binary_search(c).is_err()).count() + got.iter().filter(|c| want.binary_search(c).is_err()).count();
        let t = tris(&got);
        eprintln!("[gpu] lod scale {scale}: {} clusters, {t} triangles (unscaled neighbour: {} clusters)", got.len(), run.early[1].len());
        assert!(t <= prev, "triangles must not grow with the scale");
        if scale == 1.0 {
            first = t;
        }
        prev = t;
        // The unscaled instance in the other batch is unaffected.
        assert!(run.early[1].len() >= run.early[0].len());
    }
    assert!(prev * 3 < first, "scale 16 should cut the triangles by well over 3x ({prev} vs {first})");
    assert!(wrong * 200 <= total, "{wrong} of {total} cluster decisions differ from the reference");
}

/// The triangle budget through the real upload path: the global scale (controller
/// output) coarsens every instance, and a fair-share limit coarsens only the
/// instances that exceed their share, leaving small ones untouched.
#[test]
fn gpu_budget_global_scale_and_fair_share() {
    let Some(gpu) = gpu() else { return };
    let m = torus(200, 100);
    let dag = build(&m);
    let up = upload(&dag);
    let sph = mesh_sphere(&up);
    // One close instance and three far ones in the same view.
    let (hv, _rv, _) = look_view(sph, 9.0, Vec3::new(0.3, 0.4, 1.0));
    let dir = (hv.cam - Vec3::new(sph[0], sph[1], sph[2])).normalize();
    let mk = |at: f32, batch: u32| HeadlessInstance {
        mesh: 0,
        model: Mat4::from_translation(dir * at * sph[3]),
        batch,
        skin: None,
        lod_scale: 1.0,
    };
    let insts = [mk(2.0, 0), mk(-60.0, 1), mk(-70.0, 1), mk(-80.0, 1)];
    let tris = |run: &HeadlessRun, b: usize| run.early[b].iter().map(|e| dag.clusters[e.1 as usize].triangle_count as u64).sum::<u64>();
    let base = gpu.run(std::slice::from_ref(&up), &insts, 2, &hv, HeadlessHiz::Off, &HeadlessOptions::default()).unwrap();
    assert!(base.inst_scales.iter().all(|&s| s == 1.0));
    let (near0, far0) = (tris(&base, 0), tris(&base, 1));
    // Global scale 4 (a stressed controller): both batches shrink.
    let g = gpu
        .run(std::slice::from_ref(&up), &insts, 2, &hv, HeadlessHiz::Off, &HeadlessOptions { global_scale: Some(4.0), ..Default::default() })
        .unwrap();
    assert!(g.inst_scales.iter().all(|&s| s == 4.0));
    assert!(tris(&g, 0) < near0, "global scale must coarsen the near instance");
    // A fair share much smaller than the near instance's demand but above the far ones'.
    let limit = (near0 / 3) as u32;
    let f = gpu
        .run(std::slice::from_ref(&up), &insts, 2, &hv, HeadlessHiz::Off, &HeadlessOptions { max_triangles: Some(limit), ..Default::default() })
        .unwrap();
    eprintln!(
        "[gpu] budget: near {near0} tris unlimited, {} at global x4, {} with a fair-share limit of {limit}; far {far0} -> {}; instance scales {:?}",
        tris(&g, 0),
        tris(&f, 0),
        tris(&f, 1),
        f.inst_scales
    );
    assert!(f.inst_scales[0] > 1.0, "the near instance is over its share");
    assert!(tris(&f, 0) < near0);
    assert!(f.inst_scales[1..].iter().all(|&s| s == 1.0), "small instances keep full detail: {:?}", f.inst_scales);
    assert_eq!(tris(&f, 1), far0);
}
