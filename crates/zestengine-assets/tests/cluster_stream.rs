//! GEO-6 geometry streaming on a real device, in the real `Renderer` frame
//! loop (invisible window): a 1M+ triangle mesh whose DAG is split into pages
//! and streamed through a small budget is compared with the same mesh fully
//! resident.
//!
//!  - convergence: after the pages arrive the image equals the fully resident
//!    one (silhouette IoU, mean difference), and at every frame while they arrive
//!    the silhouette has no hole (the coarse root page and the resident parents
//!    stand in),
//!  - budget pressure: a budget far below the working set never exceeds itself,
//!    evicts, stalls, and still never leaves a hole,
//!  - fly-through: a camera orbit with frame-time statistics against the fully
//!    resident run, page traffic, thrash, and no device idle anywhere,
//!  - file backed: the pages are read from a paged geometry file on disk,
//!  - the comparison image goes to `docs/design/geo6_streaming_comparison.png`.
//!
//! One `#[test]` runs every phase (winit allows one event loop per process).
//! Skips cleanly with no display / GPU (`ZESTENGINE_TEST_GPU=0` forces that).
//! `cargo test -p zestengine-assets --test cluster_stream -- --nocapture`
//! (`ZESTENGINE_CS_TRIS=<n>` mesh size in millions of quads*2, default 1.4M).

use std::sync::Arc;

use glam::{Mat4, Vec3};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::{Window, WindowId};
use zestengine_assets::cluster_dag::{ClusterDag, DagOptions, build_cluster_dag};
use zestengine_assets::cluster_pages::{MemPageSource, PageOptions, PagePlan, PageSource, PagedFile, encode_paged, skeleton_of};
use zestengine_assets::cluster_stream::ClusterStreamer;
use zestengine_render::{
    CameraData, ClusterMode, ClusterSettings, DrawCmd, FrameInput, GpuCullSettings, LightData, LightInstance, LightKind, MAX_REFLECTION_PROBES,
    MaterialDesc, MaterialHandle, MaterialParams, MeshData, MeshHandle, PostFx, RenderStats, Renderer, ShadowCasting, Vertex,
};

const BG: [u8; 3] = [26, 31, 51];

fn env_n(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// A smooth torus of `nu * nv * 2` triangles (R = 1, r = 0.35).
fn torus(nu: u32, nv: u32) -> MeshData {
    let (big, small) = (1.0f32, 0.35f32);
    let mut vertices = Vec::with_capacity(((nu + 1) * (nv + 1)) as usize);
    for i in 0..=nu {
        let u = std::f32::consts::TAU * i as f32 / nu as f32;
        for j in 0..=nv {
            let v = std::f32::consts::TAU * j as f32 / nv as f32;
            let n = Vec3::new(v.cos() * u.cos(), v.sin(), v.cos() * u.sin());
            let p = Vec3::new((big + small * v.cos()) * u.cos(), small * v.sin(), (big + small * v.cos()) * u.sin());
            vertices.push(Vertex {
                position: p.to_array(),
                normal: n.to_array(),
                tangent: [-u.sin(), 0.0, u.cos(), 1.0],
                // A vertex colour ramp: streamed pages must carry the colours, or the resident image differs.
                color: [i as f32 / nu as f32, if j < nv / 2 { 0.3 } else { 1.0 }, j as f32 / nv as f32, 1.0],
                uv: [i as f32 / nu as f32, j as f32 / nv as f32],
                ..Default::default()
            });
        }
    }
    let mut indices = Vec::with_capacity((nu * nv * 6) as usize);
    for i in 0..nu {
        for j in 0..nv {
            let a = i * (nv + 1) + j;
            let b = a + nv + 1;
            indices.extend([a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    MeshData { vertices, indices, has_lightmap_uv: false }
}

struct Image {
    w: u32,
    h: u32,
    px: Vec<u8>,
}

impl Image {
    fn is_object(&self, i: usize) -> bool {
        let p = &self.px[i * 4..i * 4 + 3];
        (0..3).any(|c| p[c].abs_diff(BG[c]) > 6)
    }
}

#[derive(Clone, Copy, Debug)]
struct Diff {
    iou: f64,
    /// Reference object pixels that are background in the other image (holes), as a share of the object.
    holes: f64,
    mean_abs: f64,
}

fn compare(reference: &Image, other: &Image) -> Diff {
    let n = (reference.w * reference.h) as usize;
    let (mut inter, mut union, mut holes, mut obj, mut sum) = (0usize, 0usize, 0usize, 0usize, 0u64);
    for i in 0..n {
        let (a, b) = (reference.is_object(i), other.is_object(i));
        inter += (a && b) as usize;
        union += (a || b) as usize;
        holes += (a && !b) as usize;
        obj += a as usize;
        sum += (0..3).map(|c| reference.px[i * 4 + c].abs_diff(other.px[i * 4 + c]) as u64).sum::<u64>();
    }
    Diff { iou: if union == 0 { 1.0 } else { inter as f64 / union as f64 }, holes: holes as f64 / obj.max(1) as f64, mean_abs: sum as f64 / (n * 3) as f64 }
}

struct Rig {
    r: Renderer,
    ext: [u32; 2],
    eye: Vec3,
    target: Vec3,
    draw: Option<(MeshHandle, MaterialHandle)>,
    center: Vec3,
    radius: f32,
    streamer: Option<ClusterStreamer>,
    t: f32,
    /// Device idles caused by reading the viewport back (the test's own).
    shot_idles: u64,
    /// The sun casts cascaded shadows (the cluster shadow path, whose cuts request pages too).
    shadows: bool,
    /// Viewport debug view (8 = Cluster LOD).
    debug_view: u32,
}

impl Rig {
    fn set_stream(&mut self, on: bool, mb: u32) {
        let mut s = self.r.cluster_settings();
        s.mode = ClusterMode::On;
        s.stream = on;
        s.stream_budget_mb = mb;
        self.r.set_cluster_settings(s);
    }

    fn frame(&mut self) -> (RenderStats, f64) {
        self.t += 1.0 / 60.0;
        let draws: Vec<DrawCmd> = self
            .draw
            .iter()
            .map(|&(mesh, material)| DrawCmd {
                gi_source: None,
                mesh,
                material,
                transform: Mat4::IDENTITY,
                highlight: 0.0,
                mesh_center: self.center,
                bound_radius: self.radius,
                lightmap_layer: -1,
                lightmap_size: 0,
                static_geometry: false,
                planar_index: -1,
                occluded: false,
                shadow_casting: ShadowCasting::On,
                instance_color: None,
                instance_param: None,
            })
            .collect();
        let sun = LightInstance {
            kind: LightKind::Directional,
            position: Vec3::ZERO,
            // A swaying sun while shadows are on, so every frame re-renders the shadow cuts (a static one is cached).
            direction: Vec3::new(-0.45 + if self.shadows { 0.25 * (self.t * 4.0).sin() } else { 0.0 }, -1.0, -0.3).normalize(),
            color: [1.0, 0.97, 0.9],
            intensity: 3.0,
            range: 0.0,
            spot_inner_deg: 0.0,
            spot_outer_deg: 0.0,
            cast_shadows: self.shadows,
            soft_shadows: true, radius: 0.0,
            shadow_bias: 0.005,
            baked: false,
        };
        let view = Mat4::look_at_rh(self.eye, self.target, Vec3::Y);
        let aspect = self.ext[0] as f32 / self.ext[1] as f32;
        let camera = CameraData { view, proj: Mat4::perspective_rh(1.0, aspect, 0.05, 100.0), position: self.eye };
        let input = FrameInput {
            clear_color: [0.1, 0.12, 0.2, 1.0],
            camera,
            light: LightData::default(),
            lights: std::slice::from_ref(&sun),
            capture_lights: &[],
            voxel_emitters: &[],
            camera_preview: None,
            model_preview: None,
            draw_skybox: false,
            shadow_pcf_texel: 0.0,
            shadow_bias_scale: 0.0,
            shadow_normal_offset: 0.0,
            shadow_distance: 60.0,
            shadow_cascades: 0,
            shadow_cascade_lambda: 0.0,
            time: self.t,
            water_time: 0.0,
            water: &[],
            draws: &draws,
            decals: &[],
            lightmap_preview: false,
            gi_debug: self.debug_view as _,
            render_viewport: false,
            viewport_extent: Some(self.ext),
            postfx: PostFx::default(),
            preview_postfx: PostFx::default(),
            reflection_probe: None,
            reflection_probes: [None; MAX_REFLECTION_PROBES],
            reflection_probe_b: None,
            reflection_primary: 0,
            fog: None,
            voxel_specular: false,
            planar: &[],
            camera_targets: &[],
            egui: None,
            world_ui: Vec::new(),
        };
        let t0 = std::time::Instant::now();
        self.r.render(&input).expect("render");
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if let Some(s) = self.streamer.as_mut() {
            s.pump(&mut self.r);
        }
        (self.r.stats(), ms)
    }

    fn shot(&mut self) -> Image {
        let before = zestengine_render::hitch::idle_wait_count();
        let read = self.r.debug_read_viewport();
        self.shot_idles += zestengine_render::hitch::idle_wait_count() - before;
        let (w, h, px) = read.expect("readback").expect("viewport target");
        Image { w, h, px }
    }
}

fn quantile(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f64 * q) as usize]
}

fn save_png(path: &std::path::Path, images: &[&Image], labels_h: u32) {
    let (w, h) = (images[0].w, images[0].h);
    let mut img = image::RgbImage::new(w * images.len() as u32, h + labels_h);
    for p in img.pixels_mut() {
        *p = image::Rgb([16, 16, 16]);
    }
    for (k, im) in images.iter().enumerate() {
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                img.put_pixel(k as u32 * w + x, y + labels_h, image::Rgb([im.px[i], im.px[i + 1], im.px[i + 2]]));
            }
        }
    }
    let _ = std::fs::create_dir_all(path.parent().unwrap());
    img.save(path).expect("png written");
}

/// |a - b| amplified, as an image.
fn diff_image(a: &Image, b: &Image) -> Image {
    let mut px = vec![255u8; a.px.len()];
    for i in 0..(a.w * a.h) as usize {
        for c in 0..3 {
            px[i * 4 + c] = (a.px[i * 4 + c].abs_diff(b.px[i * 4 + c]) as u32 * 8).min(255) as u8;
        }
    }
    Image { w: a.w, h: a.h, px }
}

fn run_main(window: Arc<Window>) {
    // A page table of 6 000 entries (read once, before the first streamed mesh): phase 6 fills it.
    unsafe { std::env::set_var("ZESTENGINE_CLUSTER_STREAM_MAX_PAGES", "6000") };
    let mut r = Renderer::new(window).expect("Renderer::new");
    r.set_vsync(false);
    r.set_gpu_zone_timing(true);
    r.set_gpu_cull_settings(GpuCullSettings { enabled: true, hiz_occlusion: false, min_pixel_radius: 0.0, max_distance: 0.0 });
    r.set_cluster_settings(ClusterSettings { mode: ClusterMode::On, stream: false, ..r.cluster_settings() });
    eprintln!("[cs] {}", r.gpu_caps().report().lines().next().unwrap_or(""));

    // ---- the mesh and its DAG
    let tris_target = env_n("ZESTENGINE_CS_TRIS", 1_400_000);
    let nu = (tris_target as f64).sqrt() as u32; // nu = 2 * nv, so 2 * nu * nv = nu * nu triangles
    let nv = nu / 2;
    let mesh = torus(nu, nv);
    let t = std::time::Instant::now();
    let dag: ClusterDag = build_cluster_dag(&mesh, &[], &DagOptions::default()).expect("torus DAG");
    let opts = PageOptions::default();
    let plan = PagePlan::split(&dag, &opts);
    let st = plan.stats();
    eprintln!(
        "[cs] torus {} triangles, {} clusters, DAG built in {:.1} s; {} pages: root {:.2} MB, streamed {:.1} MB (mean page {} KB, largest {} KB, max deps {})",
        mesh.indices.len() / 3,
        dag.clusters.len(),
        t.elapsed().as_secs_f32(),
        st.pages,
        st.root_bytes as f64 / 1048576.0,
        st.streamed_bytes as f64 / 1048576.0,
        st.mean_page / 1024,
        st.largest_page / 1024,
        st.max_deps,
    );
    assert!(mesh.indices.len() / 3 >= 1_000_000, "the test wants a 1M+ triangle mesh");
    assert!(!plan.is_single(), "the mesh must be split into pages");
    let skeleton = skeleton_of(&dag);
    let mem: Arc<MemPageSource> = Arc::new(MemPageSource { dag: Arc::new(dag.clone()), plan: Arc::new(plan.clone()) });

    let mat = r
        .create_material(&MaterialDesc {
            name: "cs".into(),
            params: MaterialParams { base_color: [0.8, 0.45, 0.2, 1.0], roughness: 0.4, metallic: 0.2, ..Default::default() },
            ..Default::default()
        })
        .unwrap();
    let center = Vec3::ZERO;
    let radius = 1.4f32;
    let full = r.upload_mesh(&mesh).unwrap();
    r.install_cluster_dag(full, &dag.to_gpu_upload()).expect("full DAG installs");
    let mut rig = Rig {
        r,
        ext: [960, 540],
        eye: Vec3::new(0.0, 1.6, 3.4),
        target: Vec3::new(0.0, 0.0, 0.0),
        draw: Some((full, mat)),
        center,
        radius,
        streamer: None,
        t: 0.0,
        shot_idles: 0,
        shadows: false,
        debug_view: 0,
    };

    // ---- reference: fully resident
    for _ in 0..12 {
        rig.frame();
    }
    let reference = rig.shot();
    let ref_tris = rig.r.stats().cluster.triangles;
    eprintln!("[cs] reference (fully resident): {} triangles drawn, {} object pixels", ref_tris, (0..(reference.w * reference.h) as usize).filter(|&i| reference.is_object(i)).count());
    assert!(ref_tris > 200_000, "the reference view is not demanding enough ({ref_tris} triangles)");

    // ---- streamed copy
    let idle0 = zestengine_render::hitch::idle_wait_count();
    rig.shot_idles = 0;
    let budget_mb = env_n("ZESTENGINE_CS_BUDGET_MB", 40) as u32;
    rig.set_stream(true, budget_mb);
    let streamed = rig.r.upload_mesh(&mesh).unwrap();
    let root = mem.read_page(0).unwrap();
    let up = plan.to_gpu_upload(&skeleton, &root);
    rig.r.install_cluster_dag(streamed, &up).expect("streamed DAG installs");
    let mut streamer = ClusterStreamer::new(&rig.r);
    streamer.register(streamed, mem.clone());
    rig.streamer = Some(streamer);
    rig.draw = Some((streamed, mat));
    assert!(rig.r.cluster_stream_enabled());

    // ---- phase 1: convergence, with a hole check at every frame
    let mut worst_early = Diff { iou: 1.0, holes: 0.0, mean_abs: 0.0 };
    let mut converged_at = None;
    let mut first = None;
    let mut mid = None;
    let mut stable = 0;
    let mut last_loads = 0;
    for f in 1..=900u32 {
        let (stats, _) = rig.frame();
        let s = stats.cluster;
        if f == 1 || f <= 40 && f % 3 == 0 || f % 20 == 0 {
            let img = rig.shot();
            let d = compare(&reference, &img);
            if f > 2 {
                worst_early.iou = worst_early.iou.min(d.iou);
                worst_early.holes = worst_early.holes.max(d.holes);
            }
            if f == 3 {
                first = Some(img);
            } else if f == 30 {
                mid = Some(img);
            }
            if f <= 60 || f % 100 == 0 {
                eprintln!("[cs] frame {f:3}: IoU {:.5} holes {:.5} mean|d| {:.3}; {} pages resident ({:.1} MB), {} loading, {} standins, {} triangles", d.iou, d.holes, d.mean_abs, s.stream_pages, s.stream_resident_mb, s.stream_loading, s.stream_standins, s.triangles);
            }
        }
        if s.stream_loads == last_loads && s.stream_loading == 0 && s.stream_pending == 0 && s.stream_standins == 0 {
            stable += 1;
        } else {
            stable = 0;
        }
        last_loads = s.stream_loads;
        if stable >= 10 && converged_at.is_none() {
            converged_at = Some(f);
            break;
        }
    }
    let converged_at = converged_at.expect("never converged in 900 frames");
    for _ in 0..4 {
        rig.frame();
    }
    let converged = rig.shot();
    let dc = compare(&reference, &converged);
    let s1 = rig.r.stats().cluster;
    eprintln!(
        "[cs] converged after {converged_at} frames: IoU {:.5}, holes {:.6}, mean|d| {:.3}/255; {} pages ({:.1} of {:.0} MB), {} requests, {} loads, {:.1} MB streamed, {} evictions, {} stall frames; triangles {} vs {} resident",
        dc.iou, dc.holes, dc.mean_abs, s1.stream_pages, s1.stream_resident_mb, s1.stream_budget_mb, s1.stream_requests, s1.stream_loads, s1.stream_mb_streamed, s1.stream_evictions, s1.stream_stall_frames, s1.triangles, ref_tris
    );
    assert!(dc.iou > 0.998 && dc.holes < 0.001, "converged image differs from the fully resident one: {dc:?}");
    assert!(dc.mean_abs < 1.5, "converged image differs in shading: {dc:?}");
    assert!(worst_early.iou > 0.97 && worst_early.holes < 0.01, "holes while pages stream in: {worst_early:?}");
    assert!(s1.stream_resident_mb <= s1.stream_budget_mb + 0.01);
    assert!(s1.triangles as f64 > ref_tris as f64 * 0.9, "the converged cut is still much coarser than the resident one");
    let first = first.expect("frame 3 captured");
    let mid = mid.expect("frame 30 captured");

    // ---- phase 2: budget pressure
    let small_mb = (s1.stream_resident_mb / 4.0).max(6.0) as u32;
    rig.set_stream(true, small_mb);
    let mut worst_p = Diff { iou: 1.0, holes: 0.0, mean_abs: 0.0 };
    for f in 1..=300u32 {
        let (stats, _) = rig.frame();
        let s = stats.cluster;
        assert!(s.stream_resident_mb <= s.stream_budget_mb + 0.01, "frame {f}: {:.2} MB resident over a {:.1} MB budget", s.stream_resident_mb, s.stream_budget_mb);
        if f % 15 == 0 {
            let d = compare(&reference, &rig.shot());
            worst_p.iou = worst_p.iou.min(d.iou);
            worst_p.holes = worst_p.holes.max(d.holes);
        }
    }
    let pressure = rig.shot();
    let s2 = rig.r.stats().cluster;
    let dp = compare(&reference, &pressure);
    eprintln!(
        "[cs] budget {small_mb} MB (working set {:.1} MB): resident {:.1} MB, {} evictions, {} stall frames, {} thrash reloads; IoU {:.5} (worst {:.5}), holes {:.6}; triangles {} ({:.0}% of resident)",
        s1.stream_resident_mb, s2.stream_resident_mb, s2.stream_evictions, s2.stream_stall_frames, s2.stream_thrash, dp.iou, worst_p.iou, worst_p.holes, s2.triangles, s2.triangles as f64 * 100.0 / ref_tris as f64
    );
    assert!(s2.stream_evictions > 0, "the small budget never evicted");
    assert!(s2.stream_stall_frames > 0, "the small budget never stalled");
    assert!(worst_p.iou > 0.97 && worst_p.holes < 0.01, "holes under budget pressure: {worst_p:?}");
    assert!(s2.stream_resident_mb <= s2.stream_budget_mb + 0.01);

    // ---- phase 3: fly-through, streamed vs fully resident
    let orbit = |rig: &mut Rig, f: u32| {
        let a = f as f32 * 0.012;
        let d = 2.6 + 1.0 * (f as f32 * 0.02).sin();
        rig.eye = Vec3::new(a.sin() * d, 0.9 + 0.7 * (a * 1.7).cos(), a.cos() * d);
        rig.target = Vec3::new(0.0, 0.0, 0.0);
    };
    let flight = 700u32;
    rig.set_stream(true, budget_mb);
    for _ in 0..30 {
        rig.frame();
    }
    let run = |rig: &mut Rig, label: &str, streamed: bool| -> (Vec<f64>, Diff, RenderStats) {
        rig.draw = Some((if streamed { streamed_handle(&mut *rig) } else { full }, mat));
        orbit(rig, 0);
        for _ in 0..5 {
            rig.frame();
        }
        let mut times = Vec::new();
        let mut worst = Diff { iou: 1.0, holes: 0.0, mean_abs: 0.0 };
        let mut last = RenderStats::default();
        for f in 0..flight {
            orbit(rig, f);
            let (st, ms) = rig.frame();
            times.push(ms);
            last = st;
            if streamed && f % 50 == 25 {
                // Compare with the fully resident mesh drawn from the same pose.
                let img = rig.shot();
                rig.draw = Some((full, mat));
                rig.frame();
                rig.frame();
                let refimg = rig.shot();
                rig.draw = Some((streamed_handle(rig), mat));
                let d = compare(&refimg, &img);
                worst.iou = worst.iou.min(d.iou);
                worst.holes = worst.holes.max(d.holes);
                worst.mean_abs = worst.mean_abs.max(d.mean_abs);
            }
        }
        let mut sorted = times.clone();
        let (mean, p50, p99, max) = (times.iter().sum::<f64>() / times.len() as f64, quantile(&mut sorted, 0.5), quantile(&mut sorted, 0.99), quantile(&mut sorted, 1.0));
        eprintln!("[cs] fly-through {label}: CPU frame ms mean {mean:.2}, p50 {p50:.2}, p99 {p99:.2}, max {max:.2}; GPU frame {:.2} ms, cull {:.2} ms, draw {:.2} ms", last.gpu_frame_ms, last.cluster.cull_ms, last.cluster.draw_ms);
        (times, worst, last)
    };
    fn streamed_handle(_rig: &mut Rig) -> MeshHandle {
        STREAMED.with(|c| c.get()).expect("streamed handle set")
    }
    thread_local! {
        static STREAMED: std::cell::Cell<Option<MeshHandle>> = const { std::cell::Cell::new(None) };
    }
    STREAMED.with(|c| c.set(Some(streamed)));
    let (_, _, resident_last) = run(&mut rig, "fully resident", false);
    let before = rig.r.cluster_stream_stats();
    let (times_s, worst_f, streamed_last) = run(&mut rig, "streamed", true);
    let after = rig.r.cluster_stream_stats();
    let s3 = streamed_last.cluster;
    eprintln!(
        "[cs] fly-through traffic: {} loads, {} evictions, {:.1} MB streamed ({:.2} MB/s at 60 fps), {} stall frames, {} thrash reloads; worst IoU {:.5}, holes {:.6}; last frame {} resident pages ({:.1} MB)",
        after.loads - before.loads,
        after.evictions - before.evictions,
        (after.bytes_streamed - before.bytes_streamed) as f64 / 1048576.0,
        (after.bytes_streamed - before.bytes_streamed) as f64 / 1048576.0 / (flight as f64 / 60.0),
        after.stall_frames - before.stall_frames,
        after.thrash_reloads - before.thrash_reloads,
        worst_f.iou,
        worst_f.holes,
        s3.stream_pages,
        s3.stream_resident_mb,
    );
    assert!(worst_f.iou > 0.97 && worst_f.holes < 0.01, "holes during the fly-through: {worst_f:?}");
    assert!(after.loads > before.loads, "the fly-through streamed nothing");
    let _ = (resident_last, times_s);

    // ---- phase 4: pages read from a paged file on disk
    let dir = std::env::temp_dir().join(format!("cs_paged_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("torus.zpag");
    let bytes = encode_paged(&dag, &plan);
    std::fs::write(&file, &bytes).unwrap();
    let paged = PagedFile::open_file(&file, 0, bytes.len() as u64).expect("opens");
    assert_eq!(paged.plan, plan);
    let from_file: Arc<dyn PageSource> = Arc::new(paged);
    let streamed2 = rig.r.upload_mesh(&mesh).unwrap();
    rig.r.install_cluster_dag(streamed2, &up).expect("second streamed DAG installs");
    rig.streamer.as_mut().unwrap().register(streamed2, from_file);
    rig.r.cluster_stream_evict_all();
    rig.draw = Some((streamed2, mat));
    rig.eye = Vec3::new(0.0, 1.6, 3.4);
    rig.target = Vec3::ZERO;
    for _ in 0..200 {
        rig.frame();
    }
    let from_disk = rig.shot();
    let dd = compare(&reference, &from_disk);
    let stats = rig.streamer.as_ref().unwrap().stats();
    eprintln!("[cs] file backed: IoU {:.5} holes {:.6}; streamer read {:.1} MB in {:.2} s ({} pages, {} failed)", dd.iou, dd.holes, stats.bytes_read as f64 / 1048576.0, stats.read_secs, stats.delivered, stats.failed);
    assert!(dd.iou > 0.99 && dd.holes < 0.005, "image from file-backed pages: {dd:?}");
    assert_eq!(stats.failed, 0);
    std::fs::remove_dir_all(&dir).ok();

    // ---- phase 5: shadow views request pages, behind the main view (GEO-6 leftover)
    // A shadow cut finer than the camera's (0.25 texel target against 1 px) wants pages the main view
    // does not: they are requested as a low-priority class. The main view must still converge in
    // about the same time and to the same image as a fully resident mesh with the same shadows.
    {
        let mut st = rig.r.cluster_settings();
        st.shadow_px = 0.25;
        rig.r.set_cluster_settings(st);
        rig.shadows = true;
        rig.eye = Vec3::new(0.0, 1.6, 3.4);
        rig.target = Vec3::ZERO;
        rig.draw = Some((full, mat));
        for _ in 0..12 {
            rig.frame();
        }
        let shadow_ref = rig.shot();
        rig.set_stream(true, budget_mb);
        rig.r.cluster_stream_evict_all();
        rig.draw = Some((streamed, mat));
        let before = rig.r.cluster_stream_stats();
        let mut conv = None;
        let (mut stable, mut last_loads) = (0, 0);
        for f in 1..=900u32 {
            let s = rig.frame().0.cluster;
            if s.stream_loads == last_loads && s.stream_loading == 0 && s.stream_standins == 0 {
                stable += 1;
            } else {
                stable = 0;
            }
            last_loads = s.stream_loads;
            if stable >= 10 {
                conv = Some(f);
                break;
            }
        }
        let conv = conv.expect("never converged with shadows on");
        for _ in 0..4 {
            rig.frame();
        }
        let sh = rig.shot();
        let d = compare(&shadow_ref, &sh);
        let s5 = rig.r.stats().cluster;
        let after = rig.r.cluster_stream_stats();
        eprintln!(
            "[cs] shadows on (cascaded sun, 0.25 texel shadow cuts; shadow cuts: {} instances, {} clusters, {} triangles): main view converged after {conv} frames (without shadows {converged_at}); {} requests of which {} only from shadow views; {} loads, {} evictions; silhouette vs fully resident: IoU {:.5}, holes {:.6}; {:.1} of {:.0} MB",
            s5.shadow_instances, s5.shadow_clusters, s5.shadow_triangles, s5.stream_requests, s5.stream_shadow_requests, after.loads - before.loads, after.evictions - before.evictions, d.iou, d.holes, s5.stream_resident_mb, s5.stream_budget_mb
        );
        assert!(s5.stream_shadow_requests > 0, "shadow views requested no pages of their own");
        assert!(conv as f64 <= converged_at as f64 * 1.5 + 15.0, "shadow requests slowed the main view: {conv} frames against {converged_at}");
        assert!(d.iou > 0.995 && d.holes < 0.002, "image with shadows differs from fully resident: {d:?}");
        assert!(s5.stream_resident_mb <= s5.stream_budget_mb + 0.01);
        rig.shadows = false;
    }

    // ---- phase 9: the engine's lazy load path (GEO-6 leftover): a stored paged mesh is not decoded
    // The same calls the engine's cluster worker makes: `GeoSource::load_or_build_lazy` leaves the paged
    // mesh on disk (skeleton only in memory), the root page and the skeleton make the upload, the file
    // serves the pages.
    {
        use zestengine_assets::geo_cache::GeoSource;
        let scene = zestengine_assets::Scene {
            meshes: vec![torus(nu, nv)],
            morphs: Vec::new(),
            springs: Default::default(),
            textures: Vec::new(),
            materials: Vec::new(),
            instances: Vec::new(),
            skeletons: Vec::new(),
            animations: Vec::new(),
            collision: Vec::new(),
            identity: Default::default(),
        };
        let dir = std::env::temp_dir().join(format!("cs_lazy_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("torus.glb");
        let source = GeoSource::of(&scene);
        let t = std::time::Instant::now();
        let built = source.load_or_build(&model); // builds and stores the cache
        let build_s = t.elapsed().as_secs_f32();
        let t = std::time::Instant::now();
        let lazy = source.load_or_build_lazy(&model);
        let lazy_ms = t.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
        let eager = source.load_or_build(&model);
        let eager_ms = t.elapsed().as_secs_f64() * 1000.0;
        let file_bytes = std::fs::metadata(zestengine_assets::geo_cache::geo_path(&model).unwrap()).unwrap().len();
        let (lb, eb) = (lazy.geo.dag_bytes(), eager.dag_bytes());
        eprintln!(
            "[cs] lazy engine load: cache file {:.1} MB (built in {build_s:.1} s); decoded eagerly: {:.1} MB held in {eager_ms:.0} ms; lazily: {:.2} MB held in {lazy_ms:.0} ms ({:.1} %)",
            file_bytes as f64 / 1048576.0, eb as f64 / 1048576.0, lb as f64 / 1048576.0, 100.0 * lb as f64 / eb as f64
        );
        assert_eq!(built.meshes.len(), 1);
        assert!(lazy.paged[0].is_some(), "the stored mesh was not left paged");
        assert!(lb * 5 < eb, "the lazy load kept most of the payload");
        let mut lazy = lazy;
        let file = lazy.paged[0].take().unwrap();
        let root = file.read_page(0).unwrap();
        let up = file.plan.to_gpu_upload(&file.skeleton, &root);
        let lh = rig.r.upload_mesh(&mesh).unwrap();
        rig.r.install_cluster_dag(lh, &up).expect("the lazily prepared DAG installs");
        rig.streamer.as_mut().unwrap().register(lh, Arc::new(file));
        rig.r.cluster_stream_evict_all();
        rig.draw = Some((lh, mat));
        rig.eye = Vec3::new(0.0, 1.6, 3.4);
        rig.target = Vec3::ZERO;
        for _ in 0..240 {
            rig.frame();
        }
        let img = rig.shot();
        let d = compare(&reference, &img);
        let s9 = rig.r.stats().cluster;
        eprintln!("[cs] lazy file-backed mesh: IoU {:.5}, holes {:.6}, mean|d| {:.3}/255; {} pages resident, {} standins", d.iou, d.holes, d.mean_abs, s9.stream_pages, s9.stream_standins);
        assert!(d.iou > 0.998 && d.holes < 0.001 && s9.stream_standins == 0, "lazily loaded mesh differs from fully resident: {d:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    // ---- phase 6: the page table is a runtime bound (here 6 000 entries instead of the default 262 144)
    // Small streamed meshes fill what the big ones left; the last one lives high in the table (its touched
    // flags sit after the 6 000 priority words, found through the push constant) and must stream and
    // draw exactly like the first; the next one is refused with an error that names the knob.
    {
        let small = torus(200, 100);
        let sdag = build_cluster_dag(&small, &[], &DagOptions::default()).expect("small torus DAG");
        let popts = PageOptions { page_bytes: 3 * 1024, resident_bytes: 24 * 1024, min_stream_bytes: 1 };
        let splan = PagePlan::split(&sdag, &popts);
        let sskel = skeleton_of(&sdag);
        let ssrc: Arc<MemPageSource> = Arc::new(MemPageSource { dag: Arc::new(sdag.clone()), plan: Arc::new(splan.clone()) });
        let sup = splan.to_gpu_upload(&sskel, &ssrc.read_page(0).unwrap());
        let cap = zestengine_render::stream_max_pages();
        assert_eq!(cap, 6000, "the page table bound did not come from the environment");
        let ref_full = rig.r.upload_mesh(&small).unwrap();
        rig.r.install_cluster_dag(ref_full, &sdag.to_gpu_upload()).expect("small full DAG");
        let mut handles = Vec::new();
        let used = 2 * plan.pages.len(); // the big streamed meshes of the earlier phases
        let mut error = None;
        for _ in 0..200 {
            let h = rig.r.upload_mesh(&small).unwrap();
            match rig.r.install_cluster_dag(h, &sup) {
                Ok(()) => {
                    rig.streamer.as_mut().unwrap().register(h, ssrc.clone());
                    handles.push(h);
                }
                Err(e) => {
                    error = Some(format!("{e:#}"));
                    break;
                }
            }
        }
        let error = error.expect("the page table never filled");
        eprintln!("[cs] page table bound {cap}: {} entries used by the big meshes, {} small meshes of {} pages installed, then: {error}", used, handles.len(), splan.pages.len());
        assert!(error.contains("ZESTENGINE_CLUSTER_STREAM_MAX_PAGES") || error.contains("pool"), "unhelpful error: {error}");
        assert!(handles.len() >= 10, "too few meshes fit ({})", handles.len());
        rig.eye = Vec3::new(0.0, 1.0, 2.2);
        rig.target = Vec3::ZERO;
        rig.center = Vec3::ZERO;
        rig.radius = 1.4;
        rig.draw = Some((ref_full, mat));
        for _ in 0..12 {
            rig.frame();
        }
        let small_ref = rig.shot();
        let last = *handles.last().unwrap();
        rig.draw = Some((last, mat));
        let (mut stable, mut last_loads) = (0, 0);
        for _ in 0..600 {
            let s = rig.frame().0.cluster;
            if s.stream_loads == last_loads && s.stream_loading == 0 && s.stream_pending == 0 && s.stream_standins == 0 {
                stable += 1;
            } else {
                stable = 0;
            }
            last_loads = s.stream_loads;
            if stable >= 10 {
                break;
            }
        }
        for _ in 0..4 {
            rig.frame();
        }
        let hi = rig.shot();
        let d = compare(&small_ref, &hi);
        let s6 = rig.r.stats().cluster;
        eprintln!("[cs] last small mesh (mesh #{}, high in the table): IoU {:.5}, holes {:.6}, mean|d| {:.3}/255; {} pages resident, {} standins left", handles.len(), d.iou, d.holes, d.mean_abs, s6.stream_pages, s6.stream_standins);
        assert!(d.iou > 0.995 && d.holes < 0.002 && s6.stream_standins == 0, "a mesh high in the page table does not stream like the others: {d:?}");
        let _ = save_png;
    }

    // ---- phase 8: the Cluster LOD debug view tints clusters that stand in for missing pages (GEO-6)
    {
        rig.set_stream(true, budget_mb);
        rig.eye = Vec3::new(0.0, 1.6, 3.4);
        rig.target = Vec3::ZERO;
        rig.draw = Some((streamed, mat));
        rig.debug_view = 8;
        rig.r.cluster_stream_evict_all();
        // Share (per mille) of the object's pixels in the stand-in tint's orange; the per-cluster hues are spread over the wheel.
        let tinted = |img: &Image| -> usize {
            let (mut hit, mut n) = (0usize, 0usize);
            for i in 0..(img.w * img.h) as usize {
                if img.is_object(i) {
                    let p = &img.px[i * 4..i * 4 + 3];
                    n += 1;
                    hit += (p[0] > 200 && p[1] > 100 && p[1] < 200 && p[2] < 130) as usize;
                }
            }
            hit * 1000 / n.max(1)
        };
        let mut shots = Vec::new();
        for f in 1..=200u32 {
            let s = rig.frame().0.cluster;
            if f == 2 || f == 20 || f == 40 {
                shots.push((f, s.stream_standins, rig.shot()));
            }
        }
        let done = rig.shot();
        let counts: Vec<String> = shots.iter().map(|(f, st, img)| format!("frame {f}: {st} standin clusters, orange per mille {}", tinted(img))).collect();
        eprintln!("[cs] Cluster LOD view tint: {}; converged: {} standins, orange per mille {}", counts.join("; "), rig.r.stats().cluster.stream_standins, tinted(&done));
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/design/geo6_cluster_lod_tint.png");
        save_png(&root, &[&shots[0].2, &shots[1].2, &shots[2].2, &done], 0);
        eprintln!("[cs] wrote {} (Cluster LOD view at frames 2, 20, 40 after evicting every page, and converged; red-orange = standing in for a page that has not arrived)", root.display());
        assert!(tinted(&shots[0].2) > tinted(&done) * 5 + 20, "the stand-ins are not tinted in the Cluster LOD view");
        assert!(tinted(&done) < 250, "tint stays after the pages arrived");
        rig.debug_view = 0;
    }

    // ---- no device idle anywhere in the streaming phases (the Hi-Z phase below builds the pyramid once, which waits)
    let idles = zestengine_render::hitch::idle_wait_count() - idle0 - rig.shot_idles;
    eprintln!("[cs] device idles during streaming (the {} viewport read-backs of this test excluded): {idles}", rig.shot_idles);
    assert_eq!(idles, 0, "streaming idled the device");

    // ---- phase 7: Hi-Z on: clusters hidden by the near side of the torus warm their pages (GEO-6 leftover)
    {
        rig.r.set_gpu_cull_settings(GpuCullSettings { enabled: true, hiz_occlusion: true, min_pixel_radius: 0.0, max_distance: 0.0 });
        rig.set_stream(true, budget_mb);
        // A low, side-on view: the near half of the ring hides much of the far half.
        rig.eye = Vec3::new(0.0, 0.12, 3.0);
        rig.target = Vec3::ZERO;
        rig.draw = Some((full, mat));
        for _ in 0..14 {
            rig.frame();
        }
        let reference = rig.shot();
        rig.draw = Some((streamed, mat));
        rig.r.cluster_stream_evict_all();
        let before = rig.r.cluster_stream_stats();
        let (mut stable, mut last_loads, mut conv, mut hidden, mut recovered) = (0, 0, None, 0u32, 0u32);
        for f in 1..=900u32 {
            let s = rig.frame().0.cluster;
            hidden = hidden.max(s.culled_hiz);
            recovered = recovered.max(s.recovered);
            if s.stream_loads == last_loads && s.stream_loading == 0 && s.stream_standins == 0 {
                stable += 1;
            } else {
                stable = 0;
            }
            last_loads = s.stream_loads;
            if stable >= 10 {
                conv = Some(f);
                break;
            }
        }
        let conv = conv.expect("never converged with Hi-Z on");
        for _ in 0..4 {
            rig.frame();
        }
        let img = rig.shot();
        let d = compare(&reference, &img);
        let s7 = rig.r.stats().cluster;
        let after = rig.r.cluster_stream_stats();
        eprintln!(
            "[cs] Hi-Z on: converged after {conv} frames (Hi-Z off: {converged_at}); most clusters hidden by Hi-Z in a frame {hidden}, recovered by the late pass at most {recovered}; {} loads, {:.1} MB resident (the Hi-Z-off run above: 420 loads, 29.8 MB, another view); {} requests of which {} low-priority (shadow / hidden); image vs fully resident: IoU {:.5}, mean|d| {:.3}/255",
            after.loads - before.loads, s7.stream_resident_mb, after.requests - before.requests, after.shadow_requests - before.shadow_requests, d.iou, d.mean_abs
        );
        assert!(d.iou > 0.998 && d.holes < 0.001, "Hi-Z with streaming differs from fully resident: {d:?}");
        assert!(s7.stream_resident_mb <= s7.stream_budget_mb + 0.01);
        rig.r.set_gpu_cull_settings(GpuCullSettings { enabled: true, hiz_occlusion: false, min_pixel_radius: 0.0, max_distance: 0.0 });
    }


    if let Some(mut s) = rig.streamer.take() {
        s.shutdown(&mut rig.r);
    }

    // ---- the comparison image
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/design/geo6_streaming_comparison.png");
    let d = diff_image(&reference, &converged);
    save_png(&root, &[&reference, &first, &mid, &converged, &pressure, &d], 0);
    eprintln!("[cs] wrote {} (left to right: fully resident, streamed at frame 3, frame 30, converged, {small_mb} MB budget, |converged - resident| x8)", root.display());
}

struct App {
    result: Option<Result<(), String>>,
    done: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.done {
            return;
        }
        self.done = true;
        let attrs = Window::default_attributes().with_visible(false).with_inner_size(winit::dpi::LogicalSize::new(640.0, 360.0));
        match event_loop.create_window(attrs) {
            Ok(w) => {
                let w = Arc::new(w);
                let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_main(w)));
                self.result = Some(out.map_err(|e| {
                    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "panic".into())
                }));
            }
            Err(e) => eprintln!("[cs] no window ({e}): skipped"),
        }
        event_loop.exit();
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
}

#[test]
fn geometry_streaming_in_the_real_frame_loop() {
    if std::env::var("ZESTENGINE_TEST_GPU").is_ok_and(|v| v == "0") || std::env::var_os("DISPLAY").is_none() {
        eprintln!("[cs] no display / GPU disabled: skipped");
        return;
    }
    let _gpu = zestengine_render::cluster_headless::gpu_lock();
    let Ok(mut event_loop) = EventLoop::builder().with_x11().with_any_thread(true).build() else {
        eprintln!("[cs] no X11 event loop: skipped");
        return;
    };
    let mut app = App { result: None, done: false };
    event_loop.run_app_on_demand(&mut app).unwrap();
    if let Some(Err(e)) = app.result {
        panic!("{e}");
    }
}
