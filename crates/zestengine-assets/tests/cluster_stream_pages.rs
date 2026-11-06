//! GEO-6 leftover: a page table above 65,536 entries on a real device, and the error path above the
//! configured bound. Thousands of small streamed meshes (cheap placeholder GPU meshes carrying real
//! tiny-paged DAGs) fill the table past 65,536 entries; then a streamed torus installed LAST (its
//! entries sit above index 65,536, where the old 16-bit bound ended) streams to an image identical to the
//! fully resident one. The table is then filled to its configured bound and the next install must fail
//! with the message naming the setting. Frame cost is measured with a nearly empty table and with a full one.
//! `cargo test -p zestengine-assets --test cluster_stream_pages -- --nocapture`

use std::sync::Arc;

use glam::{Mat4, Vec3};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::{Window, WindowId};
use zestengine_assets::cluster_dag::{ClusterDag, DagOptions, build_cluster_dag};
use zestengine_assets::cluster_pages::{MemPageSource, PageOptions, PagePlan, PageSource, skeleton_of};
use zestengine_assets::cluster_stream::ClusterStreamer;
use zestengine_render::{
    CameraData, ClusterMode, ClusterSettings, DrawCmd, FrameInput, GpuCullSettings, LightData, LightInstance, LightKind, MAX_REFLECTION_PROBES,
    MaterialDesc, MaterialHandle, MaterialParams, MeshData, MeshHandle, PostFx, RenderStats, Renderer, ShadowCasting, Vertex,
};

fn torus(nu: u32, nv: u32) -> MeshData {
    let (big, small) = (1.0f32, 0.35f32);
    let mut vertices = Vec::new();
    for i in 0..=nu {
        let u = std::f32::consts::TAU * i as f32 / nu as f32;
        for j in 0..=nv {
            let v = std::f32::consts::TAU * j as f32 / nv as f32;
            let n = Vec3::new(v.cos() * u.cos(), v.sin(), v.cos() * u.sin());
            let p = Vec3::new((big + small * v.cos()) * u.cos(), small * v.sin(), (big + small * v.cos()) * u.sin());
            vertices.push(Vertex { position: p.to_array(), normal: n.to_array(), tangent: [-u.sin(), 0.0, u.cos(), 1.0], color: [1.0; 4], uv: [i as f32 / nu as f32, j as f32 / nv as f32], ..Default::default() });
        }
    }
    let mut indices = Vec::new();
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

fn diff(a: &Image, b: &Image) -> (f64, f64, usize) {
    let n = (a.w * a.h) as usize;
    let bg = [a.px[0], a.px[1], a.px[2]];
    let obj = |im: &Image, i: usize| (0..3).any(|c| im.px[i * 4 + c].abs_diff(bg[c]) > 6);
    let (mut inter, mut union, mut sum, mut changed) = (0usize, 0usize, 0u64, 0usize);
    for i in 0..n {
        let (x, y) = (obj(a, i), obj(b, i));
        inter += (x && y) as usize;
        union += (x || y) as usize;
        let d: u64 = (0..3).map(|c| a.px[i * 4 + c].abs_diff(b.px[i * 4 + c]) as u64).sum();
        sum += d;
        changed += (d != 0) as usize;
    }
    (inter as f64 / union.max(1) as f64, sum as f64 / (n * 3) as f64, changed)
}

struct Rig {
    r: Renderer,
    draw: Vec<(MeshHandle, MaterialHandle, Vec3)>,
    streamer: Option<ClusterStreamer>,
    t: f32,
}

impl Rig {
    fn frame(&mut self) -> (RenderStats, f64) {
        self.t += 1.0 / 60.0;
        let draws: Vec<DrawCmd> = self
            .draw
            .iter()
            .map(|&(mesh, material, at)| DrawCmd {
                gi_source: None,
                mesh,
                material,
                transform: Mat4::from_translation(at),
                highlight: 0.0,
                mesh_center: Vec3::ZERO,
                bound_radius: 1.4,
                lightmap_layer: -1,
                lightmap_size: 0,
                static_geometry: false,
                planar_index: -1,
                occluded: false,
                shadow_casting: ShadowCasting::Off,
                instance_color: None,
                instance_param: None,
            })
            .collect();
        let sun = LightInstance { kind: LightKind::Directional, position: Vec3::ZERO, direction: Vec3::new(-0.45, -1.0, -0.3).normalize(), color: [1.0, 0.97, 0.9], intensity: 3.0, range: 0.0, spot_inner_deg: 0.0, spot_outer_deg: 0.0, cast_shadows: false, soft_shadows: true, radius: 0.0, shadow_bias: 0.005, baked: false };
        let eye = Vec3::new(0.0, 1.4, 3.6);
        let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
        let camera = CameraData { view, proj: Mat4::perspective_rh(1.0, 16.0 / 9.0, 0.05, 100.0), position: eye };
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
            gi_debug: 0,
            render_viewport: false,
            viewport_extent: Some([960, 540]),
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
        let (w, h, px) = self.r.debug_read_viewport().expect("readback").expect("viewport target");
        Image { w, h, px }
    }

    fn converge(&mut self, max: u32) -> Option<u32> {
        let (mut stable, mut last) = (0, 0);
        for f in 1..=max {
            let s = self.frame().0.cluster;
            if s.stream_loads == last && s.stream_loading == 0 && s.stream_pending == 0 && s.stream_standins == 0 {
                stable += 1;
            } else {
                stable = 0;
            }
            last = s.stream_loads;
            if stable >= 10 {
                return Some(f);
            }
        }
        None
    }

    fn timing(&mut self, frames: u32) -> (f64, f64, f64) {
        for _ in 0..20 {
            self.frame();
        }
        let (mut cpu, mut gpu, mut cull) = (Vec::new(), 0.0, 0.0);
        for _ in 0..frames {
            let (s, ms) = self.frame();
            cpu.push(ms);
            gpu += s.gpu_frame_ms as f64;
            cull += s.cluster.cull_ms as f64;
        }
        cpu.sort_by(|a, b| a.total_cmp(b));
        (cpu.iter().sum::<f64>() / cpu.len() as f64, gpu / frames as f64, cull / frames as f64)
    }
}


fn run_main(window: Arc<Window>) {
    // Table bound for this run: above 65,536 and small enough to fill (read once, before the first streamed mesh).
    let bound: usize = 100_000;
    unsafe { std::env::set_var("ZESTENGINE_CLUSTER_STREAM_MAX_PAGES", bound.to_string()) };
    unsafe { std::env::set_var("ZESTENGINE_CLUSTER_POOL_SCALE", "8") };
    let mut r = Renderer::new(window).expect("Renderer::new");
    r.set_vsync(false);
    r.set_gpu_zone_timing(true);
    r.set_gpu_cull_settings(GpuCullSettings { enabled: true, hiz_occlusion: false, min_pixel_radius: 0.0, max_distance: 0.0 });
    r.set_cluster_settings(ClusterSettings { mode: ClusterMode::On, stream: false, ..r.cluster_settings() });
    eprintln!("[csp] {}", r.gpu_caps().report().lines().next().unwrap_or(""));
    assert_eq!(zestengine_render::cluster::stream_max_pages(), bound);

    // The DAG of the big torus, and the tiny-paged DAG used for the fillers.
    let mesh = torus(150, 90);
    let dag: ClusterDag = build_cluster_dag(&mesh, &[], &DagOptions::default()).expect("torus DAG");
    let big_opts = PageOptions { page_bytes: 24 * 1024, resident_bytes: 48 * 1024, min_stream_bytes: 1 };
    let plan = PagePlan::split(&dag, &big_opts);
    let filler_mesh = torus(40, 25);
    let filler: ClusterDag = build_cluster_dag(&filler_mesh, &[], &DagOptions::default()).expect("filler DAG");
    let tiny = PageOptions { page_bytes: 1, resident_bytes: 1, min_stream_bytes: 1 };
    let fplan = PagePlan::split(&filler, &tiny);
    let fmem = Arc::new(MemPageSource { dag: Arc::new(filler.clone()), plan: Arc::new(fplan.clone()) });
    eprintln!("[csp] big torus: {} triangles, {} clusters, {} pages; filler: {} triangles, {} clusters, {} pages (one input set per page)", mesh.indices.len() / 3, dag.clusters.len(), plan.pages.len(), filler_mesh.indices.len() / 3, filler.clusters.len(), fplan.pages.len());
    assert!(fplan.pages.len() >= 4 && plan.pages.len() >= 8);

    let mat = r.create_material(&MaterialDesc { name: "csp".into(), params: MaterialParams { base_color: [0.8, 0.45, 0.2, 1.0], roughness: 0.4, metallic: 0.2, ..Default::default() }, ..Default::default() }).unwrap();
    let full = r.upload_mesh(&mesh).unwrap();
    r.install_cluster_dag(full, &dag.to_gpu_upload()).expect("resident DAG");
    let mut rig = Rig { r, draw: vec![(full, mat, Vec3::ZERO)], streamer: None, t: 0.0 };
    for _ in 0..12 {
        rig.frame();
    }
    let reference = rig.shot();

    // Streamed torus A: the first table entries.
    let mut s = rig.r.cluster_settings();
    s.stream = true;
    s.stream_budget_mb = 40;
    rig.r.set_cluster_settings(s);
    let skeleton = skeleton_of(&dag);
    let mem = Arc::new(MemPageSource { dag: Arc::new(dag.clone()), plan: Arc::new(plan.clone()) });
    let mut streamer = ClusterStreamer::new(&rig.r);
    let a = rig.r.upload_mesh(&mesh).unwrap();
    rig.r.install_cluster_dag(a, &plan.to_gpu_upload(&skeleton, &mem.read_page(0).unwrap())).expect("streamed A installs");
    streamer.register(a, mem.clone());
    rig.streamer = Some(streamer);
    rig.draw = vec![(a, mat, Vec3::ZERO)];
    let conv_a = rig.converge(600).expect("A converges");
    let (cpu0, gpu0, cull0) = rig.timing(300);
    eprintln!("[csp] streamed torus A (table entries 0..{}): converged in {conv_a} frames; frame with a nearly empty table: CPU {cpu0:.3} ms, GPU {gpu0:.3} ms, cull {cull0:.3} ms", plan.pages.len());

    // Fill the table past 65,536 entries with placeholder meshes carrying tiny-paged DAGs.
    let tri = MeshData { vertices: vec![Vertex::default(); 3], indices: vec![0, 1, 2], has_lightmap_uv: false };
    let fup = fplan.to_gpu_upload(&skeleton_of(&filler), &fmem.read_page(0).unwrap());
    let mut entries = plan.pages.len();
    let t = std::time::Instant::now();
    let mut n_fill = 0usize;
    while entries < 70_000 {
        let h = rig.r.upload_mesh(&tri).unwrap();
        rig.r.install_cluster_dag(h, &fup).unwrap_or_else(|e| panic!("filler {n_fill} install failed at {entries} entries: {e:#}"));
        entries += fplan.pages.len();
        n_fill += 1;
    }
    eprintln!("[csp] {n_fill} filler meshes installed in {:.1} s: {entries} page table entries in use (bound {bound}; the old 16-bit table ended at 65536)", t.elapsed().as_secs_f32());
    assert!(entries > 65_536);

    // Streamed torus B: installed last, its entries are above 65,536.
    let b = rig.r.upload_mesh(&mesh).unwrap();
    rig.r.install_cluster_dag(b, &plan.to_gpu_upload(&skeleton, &mem.read_page(0).unwrap())).expect("streamed B installs above 65,536 entries");
    let b_first = entries;
    entries += plan.pages.len();
    rig.streamer.as_mut().unwrap().register(b, mem.clone());
    assert!(b_first >= 65_536, "B must sit above the old bound");
    rig.draw = vec![(b, mat, Vec3::ZERO)];
    let conv_b = rig.converge(900).expect("B converges");
    for _ in 0..4 {
        rig.frame();
    }
    let img_b = rig.shot();
    let (iou, mean, changed) = diff(&reference, &img_b);
    let sb = rig.r.stats().cluster;
    eprintln!("[csp] torus B (table entries {b_first}..{entries}) converged in {conv_b} frames: IoU {iou:.5}, mean|d| {mean:.4}/255, {changed} pixels differ from the fully resident torus; {} pages resident, {} loads", sb.stream_pages, sb.stream_loads);
    assert!(iou > 0.9999 && changed == 0, "streaming from table entries above 65,536 differs from fully resident");

    // Both A and B together, the same image twice (A low, B high in the table).
    rig.draw = vec![(a, mat, Vec3::new(-1.3, 0.0, 0.0)), (b, mat, Vec3::new(1.3, 0.0, 0.0))];
    assert!(rig.converge(900).is_some());
    let s2 = rig.r.stats().cluster;
    eprintln!("[csp] A (low entries) and B (entries above 65,536) drawn together: {} pages resident, {} instances", s2.stream_pages, s2.instances);

    // Frame cost with the table at 70k+ entries (the touched flags and priorities are sized to it).
    rig.draw = vec![(b, mat, Vec3::ZERO)];
    let (cpu1, gpu1, cull1) = rig.timing(300);
    eprintln!("[csp] frame with {entries} table entries: CPU {cpu1:.3} ms, GPU {gpu1:.3} ms, cull {cull1:.3} ms (nearly empty table: {cpu0:.3} / {gpu0:.3} / {cull0:.3}); page table {:.2} MB, stream I/O {:.2} MB, per-frame readback {:.2} MB",
        bound as f64 * 16.0 / 1048576.0, ((zestengine_render::cluster::stream_touch_base(bound) + bound) * 4) as f64 / 1048576.0, ((zestengine_render::cluster::stream_touch_base(bound) + bound) * 4) as f64 / 1048576.0);

    // The error path: fill to the bound, the next install fails and names the setting; nothing is corrupted.
    let mut failure = None;
    for _ in 0..200_000 {
        let h = rig.r.upload_mesh(&tri).unwrap();
        match rig.r.install_cluster_dag(h, &fup) {
            Ok(()) => entries += fplan.pages.len(),
            Err(e) => {
                failure = Some(format!("{e:#}"));
                break;
            }
        }
        if entries > bound + 100 {
            break;
        }
    }
    let msg = failure.expect("installs past the configured bound must fail");
    eprintln!("[csp] filled to {entries} of {bound} entries; the next install failed with: {msg}");
    assert!(msg.contains("ZESTENGINE_CLUSTER_STREAM_MAX_PAGES") && entries <= bound, "{msg}");
    // Still rendering correctly after the refused install.
    rig.draw = vec![(b, mat, Vec3::ZERO)];
    assert!(rig.converge(300).is_some());
    for _ in 0..4 {
        rig.frame();
    }
    let (iou2, _, changed2) = diff(&reference, &rig.shot());
    eprintln!("[csp] after the refused install, torus B still draws: IoU {iou2:.5}, {changed2} pixels differ");
    assert!(iou2 > 0.9999 && changed2 == 0);
    if let Some(mut s) = rig.streamer.take() {
        s.shutdown(&mut rig.r);
    }
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
                self.result = Some(out.map_err(|e| e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "panic".into())));
            }
            Err(e) => eprintln!("[csp] no window ({e}): skipped"),
        }
        event_loop.exit();
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
}

#[test]
fn a_page_table_above_65536_entries_streams_and_the_bound_is_an_error() {
    if std::env::var("ZESTENGINE_TEST_GPU").is_ok_and(|v| v == "0") || std::env::var_os("DISPLAY").is_none() {
        eprintln!("[csp] no display / GPU disabled: skipped");
        return;
    }
    let _gpu = zestengine_render::cluster_headless::gpu_lock();
    let Ok(mut event_loop) = EventLoop::builder().with_x11().with_any_thread(true).build() else {
        eprintln!("[csp] no X11 event loop: skipped");
        return;
    };
    let mut app = App { result: None, done: false };
    event_loop.run_app_on_demand(&mut app).unwrap();
    if let Some(Err(e)) = app.result {
        panic!("{e}");
    }
}
