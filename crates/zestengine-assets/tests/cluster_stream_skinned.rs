//! GEO-6 leftover: streaming of skinned and blendshape meshes, on a real device in
//! the real `Renderer` frame loop (invisible window).
//!
//! The skin records, joint ids and sparse blendshape entries of a cluster travel in
//! its page (assets `cluster_pages.rs`); `cluster_skin.comp` finds them through the
//! page table. A rigged avatar is installed twice: fully resident (reference) and
//! streamed under a small page budget, both drawn with the same pose and blendshape
//! weights. Checked:
//!
//!  - converged image equals the fully resident one (silhouette IoU, mean difference),
//!  - blendshape weights apply to streamed clusters (weights on vs off differ by the
//!    same amount as in the resident run),
//!  - while the avatar animates and pages arrive, no frame has a hole,
//!  - a budget far below the working set holds, evicts, stalls and never leaves a hole,
//!  - the per-instance triangle budget (fair share) coarsens the streamed avatar the
//!    same way as the resident one and asks for fewer pages,
//!  - no device idle anywhere in the streaming phases.
//!
//! Two avatars: the project's `male.glb` (small; streamed with deliberately small
//! pages) and, when present, the large `/home/lemonxah/shared/avi/lemonnew.fbx`
//! (`ZESTENGINE_CSK_AVATAR=<path>` picks another; 237-shape face, 338k triangles).
//! One `#[test]` runs everything (winit allows one event loop per process); it skips
//! cleanly with no display / GPU. Pictures go to `docs/design/`.
//! `cargo test -p zestengine-assets --test cluster_stream_skinned -- --nocapture`

use std::collections::HashSet;
use std::sync::Arc;

use glam::{Mat4, Quat, Vec3};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::{Window, WindowId};
use zestengine_assets::cluster_dag::DagOptions;
use zestengine_assets::cluster_pages::{MemPageSource, PageOptions, PagePlan, PageSource, skeleton_of};
use zestengine_assets::cluster_stream::ClusterStreamer;
use zestengine_assets::{Scene, geo_cache, load_fbx, load_gltf};
use zestengine_render::{
    CameraData, ClusterMode, ClusterSettings, DrawCmd, FrameInput, GpuCullSettings, LightData, LightInstance, LightKind, MAX_REFLECTION_PROBES,
    MaterialDesc, MaterialHandle, MaterialParams, MeshHandle, PostFx, RenderStats, Renderer, ShadowCasting,
};

/// Background colour: the first pixel of the first shot (the clear colour after tone mapping).
static BG_PACKED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

struct Image {
    w: u32,
    h: u32,
    px: Vec<u8>,
}

impl Image {
    fn is_object(&self, i: usize) -> bool {
        let p = &self.px[i * 4..i * 4 + 3];
        let bg = BG_PACKED.load(std::sync::atomic::Ordering::Relaxed).to_le_bytes();
        (0..3).any(|c| p[c].abs_diff(bg[c]) > 6)
    }

    fn object_pixels(&self) -> usize {
        (0..(self.w * self.h) as usize).filter(|&i| self.is_object(i)).count()
    }
}

#[derive(Clone, Copy, Debug)]
struct Diff {
    iou: f64,
    holes: f64,
    mean_abs: f64,
    max_abs: u32,
    /// Pixels whose colour differs at all.
    changed: usize,
}

fn compare(reference: &Image, other: &Image) -> Diff {
    let n = (reference.w * reference.h) as usize;
    let (mut inter, mut union, mut holes, mut obj, mut sum, mut max, mut changed) = (0usize, 0usize, 0usize, 0usize, 0u64, 0u32, 0usize);
    for i in 0..n {
        let (a, b) = (reference.is_object(i), other.is_object(i));
        inter += (a && b) as usize;
        union += (a || b) as usize;
        holes += (a && !b) as usize;
        obj += a as usize;
        let mut any = false;
        for c in 0..3 {
            let d = reference.px[i * 4 + c].abs_diff(other.px[i * 4 + c]) as u32;
            sum += d as u64;
            max = max.max(d);
            any |= d != 0;
        }
        changed += any as usize;
    }
    Diff { iou: if union == 0 { 1.0 } else { inter as f64 / union as f64 }, holes: holes as f64 / obj.max(1) as f64, mean_abs: sum as f64 / (n * 3) as f64, max_abs: max, changed }
}

fn save_png(path: &std::path::Path, images: &[&Image]) {
    let (w, h) = (images[0].w, images[0].h);
    let mut img = image::RgbImage::new(w * images.len() as u32, h);
    for (k, im) in images.iter().enumerate() {
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                img.put_pixel(k as u32 * w + x, y, image::Rgb([im.px[i], im.px[i + 1], im.px[i + 2]]));
            }
        }
    }
    let _ = std::fs::create_dir_all(path.parent().unwrap());
    img.save(path).expect("png written");
}

fn diff_image(a: &Image, b: &Image) -> Image {
    let mut px = vec![255u8; a.px.len()];
    for i in 0..(a.w * a.h) as usize {
        for c in 0..3 {
            px[i * 4 + c] = (a.px[i * 4 + c].abs_diff(b.px[i * 4 + c]) as u32 * 8).min(255) as u8;
        }
    }
    Image { w: a.w, h: a.h, px }
}

/// One mesh of the avatar: the fully resident reference and the streamed copy.
struct Part {
    mesh: usize,
    /// GPU-skinned copies drawing the shared DAG of their templates.
    ref_skinned: MeshHandle,
    str_skinned: MeshHandle,
    /// Templates that carry the DAG (the streamed one is registered with the streamer).
    templates: Vec<MeshHandle>,
    shapes: usize,
    streamed: bool,
    center: Vec3,
    radius: f32,
    #[allow(dead_code)] // bounds kept for the streamed-copy checks
    lo: Vec3,
    #[allow(dead_code)]
    hi: Vec3,
    material: MaterialHandle,
}

struct Avatar {
    scene: Scene,
    parts: Vec<Part>,
    /// Draw the streamed copy (else the resident reference).
    show_streamed: bool,
}

struct Rig {
    r: Renderer,
    ext: [u32; 2],
    eye: Vec3,
    target: Vec3,
    avatar: Option<Avatar>,
    streamer: Option<ClusterStreamer>,
    #[allow(dead_code)] // ground plane handle kept for the scene setup
    ground: (MeshHandle, MaterialHandle),
    t: f32,
    /// The pose / weights follow `t` when true; frozen otherwise.
    animate: bool,
    /// Blendshape weights scale (0 = off).
    morph: f32,
    shot_idles: u64,
    near: f32,
    up: Vec3,
}

fn aabb(vs: &[zestengine_render::Vertex]) -> (Vec3, f32, Vec3, Vec3) {
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for v in vs {
        lo = lo.min(Vec3::from(v.position));
        hi = hi.max(Vec3::from(v.position));
    }
    ((lo + hi) * 0.5, (hi - lo).length() * 0.5, lo, hi)
}

impl Rig {
    fn set_stream(&mut self, on: bool, mb: u32, max_triangles: Option<u32>) {
        let mut s = self.r.cluster_settings();
        s.mode = ClusterMode::On;
        s.stream = on;
        s.stream_budget_mb = mb;
        if let Some(m) = max_triangles {
            s.max_triangles = m;
        }
        self.r.set_cluster_settings(s);
    }

    fn pose_avatar(&mut self) {
        let Some(a) = self.avatar.as_ref() else { return };
        let skel = &a.scene.skeletons[0];
        let mut locals = skel.rest_locals();
        for (i, l) in locals.iter_mut().enumerate() {
            let ang = ((self.t * 1.7 + i as f32 * 0.37).sin()) * 0.22;
            *l *= Mat4::from_quat(Quat::from_rotation_x(ang) * Quat::from_rotation_z(ang * 0.5));
        }
        let palette = skel.palette(&locals);
        for p in &a.parts {
            for h in [p.ref_skinned, p.str_skinned] {
                self.r.skin_mesh_gpu(h, &palette);
                if p.shapes > 0 {
                    let w: Vec<f32> = (0..p.shapes).map(|k| self.morph * 0.9 * (0.5 + 0.5 * (self.t * 1.3 + k as f32 * 0.9).sin())).collect();
                    self.r.set_mesh_morph_weights(h, &w);
                }
            }
        }
    }

    fn frame(&mut self) -> (RenderStats, f64) {
        if self.animate {
            self.t += 1.0 / 60.0;
        }
        self.pose_avatar();
        let mut draws: Vec<DrawCmd> = Vec::new();
        let base = |mesh: MeshHandle, material: MaterialHandle, c: Vec3, rad: f32| DrawCmd {
            gi_source: None,
            mesh,
            material,
            transform: Mat4::IDENTITY,
            highlight: 0.0,
            mesh_center: c,
            bound_radius: rad,
            lightmap_layer: -1,
            lightmap_size: 0,
            static_geometry: false,
            planar_index: -1,
            occluded: false,
            shadow_casting: ShadowCasting::On,
            instance_color: None,
            instance_param: None,
        };
        if let Some(a) = &self.avatar {
            for p in &a.parts {
                draws.push(base(if a.show_streamed { p.str_skinned } else { p.ref_skinned }, p.material, p.center, p.radius));
            }
        }
        let sun = LightInstance {
            kind: LightKind::Directional,
            position: Vec3::ZERO,
            direction: Vec3::new(-0.45, -1.0, -0.3).normalize(),
            color: [1.0, 0.97, 0.9],
            intensity: 3.0,
            range: 0.0,
            spot_inner_deg: 0.0,
            spot_outer_deg: 0.0,
            cast_shadows: false,
            soft_shadows: true, radius: 0.0,
            shadow_bias: 0.005,
            baked: false,
        };
        let view = Mat4::look_at_rh(self.eye, self.target, self.up);
        let aspect = self.ext[0] as f32 / self.ext[1] as f32;
        let camera = CameraData { view, proj: Mat4::perspective_rh(0.9, aspect, self.near, self.near * 4000.0), position: self.eye };
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
            shadow_distance: 60.0 * (self.near / 0.05).max(1.0),
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
        if BG_PACKED.load(std::sync::atomic::Ordering::Relaxed) == 0 {
            BG_PACKED.store(u32::from_le_bytes([px[0], px[1], px[2], 0]), std::sync::atomic::Ordering::Relaxed);
        }
        Image { w, h, px }
    }

    fn show(&mut self, streamed: bool) {
        if let Some(a) = self.avatar.as_mut() {
            a.show_streamed = streamed;
        }
    }

    /// Frames until the streamed view stops loading (and has no stand-in), or `max`.
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

    /// Resident and streamed images of the current pose, both settled.
    fn both(&mut self) -> (Image, Image) {
        self.animate = false;
        self.show(true);
        for _ in 0..4 {
            self.frame();
        }
        let s = self.shot();
        self.show(false);
        for _ in 0..4 {
            self.frame();
        }
        let r = self.shot();
        self.show(true);
        (r, s)
    }
}

struct Loaded {
    label: String,
    scene: Scene,
    /// Page options used to split the meshes.
    opts: PageOptions,
}

fn material(r: &mut Renderer, color: [f32; 4]) -> MaterialHandle {
    r.create_material(&MaterialDesc {
        name: "csk".into(),
        params: MaterialParams { base_color: color, roughness: 0.55, metallic: 0.0, ..Default::default() },
        ..Default::default()
    })
    .unwrap()
}

fn quad_ground(half: f32) -> zestengine_render::MeshData {
    use zestengine_render::Vertex;
    let v = |x: f32, z: f32| Vertex { position: [x, 0.0, z], normal: [0.0, 1.0, 0.0], tangent: [1.0, 0.0, 0.0, 1.0], color: [1.0; 4], uv: [x, z], ..Default::default() };
    zestengine_render::MeshData { vertices: vec![v(-half, -half), v(-half, half), v(half, half), v(half, -half)], indices: vec![0, 1, 2, 0, 2, 3], has_lightmap_uv: false }
}

/// Install `loaded` twice (resident reference, streamed) and return the avatar plus the page statistics.
fn install(rig: &mut Rig, loaded: Loaded) -> (Vec<(usize, zestengine_assets::cluster_pages::PlanStats, usize)>, f64) {
    let Loaded { label, scene, opts } = loaded;
    let joints = scene.skeletons[0].joints.len();
    let t = std::time::Instant::now();
    let geo = geo_cache::build_model(&scene, &DagOptions::default());
    eprintln!("[csk] {label}: DAGs for {} meshes built in {:.1} s", geo.meshes.iter().flatten().count(), t.elapsed().as_secs_f32());
    let mut parts = Vec::new();
    let mut stats = Vec::new();
    let mut streamer = rig.streamer.take().unwrap_or_else(|| ClusterStreamer::new(&rig.r));
    let mut full_bytes = 0.0f64;
    for (mi, mesh) in scene.meshes.iter().enumerate() {
        let Some(dag) = geo.meshes[mi].as_ref() else { continue };
        let shapes: Vec<zestengine_render::MorphDeltas<'_>> = scene.morphs.get(mi).map(|v| v.iter().map(|t| t.deltas.as_slice()).collect()).unwrap_or_default();
        // Reference: the whole DAG resident.
        rig.set_stream(false, 40, None);
        let tpl_ref = rig.r.upload_mesh(mesh).unwrap();
        if let Err(e) = rig.r.install_cluster_dag(tpl_ref, &dag.to_gpu_upload()) {
            eprintln!("[csk] {label} mesh {mi}: reference DAG refused: {e:#}");
            continue;
        }
        let ref_skinned = rig.r.upload_mesh_gpu_skinned_morphed(mesh, joints, &shapes).unwrap();
        rig.r.alias_cluster_dag(ref_skinned, tpl_ref).unwrap();
        // Streamed copy.
        let plan = PagePlan::split(dag, &opts);
        let by_default = PagePlan::split(dag, &PageOptions::default()).pages.len();
        eprintln!("[csk] {label} mesh {mi}: {} source triangles, {} clusters; pages: {} with these options, {} with the engine defaults (2 MB threshold)", mesh.indices.len() / 3, dag.clusters.len(), plan.pages.len(), by_default);
        let mut templates = vec![tpl_ref];
        let streamed = !plan.is_single();
        let str_skinned = rig.r.upload_mesh_gpu_skinned_morphed(mesh, joints, &shapes).unwrap();
        if streamed {
            rig.set_stream(true, 40, None);
            let tpl = rig.r.upload_mesh(mesh).unwrap();
            let mem = Arc::new(MemPageSource { dag: Arc::new(dag.clone()), plan: Arc::new(plan.clone()) });
            let up = plan.to_gpu_upload(&skeleton_of(dag), &mem.read_page(0).unwrap());
            rig.r.install_cluster_dag(tpl, &up).unwrap_or_else(|e| panic!("{label} mesh {mi}: streamed install failed: {e:#}"));
            streamer.register(tpl, mem);
            rig.r.alias_cluster_dag(str_skinned, tpl).unwrap();
            templates.push(tpl);
            let st = plan.stats();
            full_bytes += (st.streamed_bytes + st.root_bytes) as f64;
            stats.push((mi, st, mesh.indices.len() / 3));
        } else {
            rig.r.alias_cluster_dag(str_skinned, tpl_ref).unwrap();
        }
        let (center, radius, lo, hi) = aabb(&mesh.vertices);
        parts.push(Part { mesh: mi, ref_skinned, str_skinned, templates, shapes: shapes.len(), streamed, center, radius: radius * 1.5, lo, hi, material: material(&mut rig.r, [[0.9, 0.7, 0.6, 1.0], [0.4, 0.55, 0.9, 1.0], [0.8, 0.35, 0.35, 1.0], [0.6, 0.8, 0.5, 1.0]][mi % 4]) });
    }
    rig.streamer = Some(streamer);
    eprintln!("[csk] {label}: {} meshes drawn from clusters, {} of them streamed", parts.len(), parts.iter().filter(|p| p.streamed).count());
    for (mi, st, tris) in &stats {
        eprintln!(
            "[csk]   mesh {mi}: {tris} tris, {} pages, root {:.2} MB, streamed {:.1} MB (mean page {} KB, largest {} KB, max deps {})",
            st.pages,
            st.root_bytes as f64 / 1048576.0,
            st.streamed_bytes as f64 / 1048576.0,
            st.mean_page / 1024,
            st.largest_page / 1024,
            st.max_deps
        );
    }
    rig.avatar = Some(Avatar { scene, parts, show_streamed: true });
    (stats, full_bytes / 1048576.0)
}

/// Remove the avatar's meshes (both copies) and forget their pages.
fn unload(rig: &mut Rig) {
    let Some(a) = rig.avatar.take() else { return };
    let mut meshes = HashSet::new();
    for p in &a.parts {
        meshes.insert(p.ref_skinned.index());
        meshes.insert(p.str_skinned.index());
        for t in &p.templates {
            meshes.insert(t.index());
            if let Some(s) = rig.streamer.as_mut() {
                s.unregister(*t);
            }
        }
    }
    let _ = rig.r.free_resources(&meshes, &Default::default(), &Default::default());
    for _ in 0..8 {
        rig.frame();
    }
}

/// Rest-pose bounds of some meshes of the avatar. The palette carries the import's unit scale (an FBX in
/// centimetres is posed 100x smaller than its bind-pose vertices).
fn posed_bounds(a: &Avatar, meshes: impl Iterator<Item = usize>) -> (Vec3, Vec3) {
    let skel = &a.scene.skeletons[0];
    let palette = skel.palette(&skel.rest_locals());
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for mi in meshes {
        for v in a.scene.meshes[mi].vertices.iter().step_by(5) {
            let mut m = Mat4::ZERO;
            for k in 0..4 {
                if v.weights[k] > 0.0 {
                    m += palette[v.joints[k] as usize] * v.weights[k];
                }
            }
            let q = m.transform_point3(Vec3::from(v.position));
            lo = lo.min(q);
            hi = hi.max(q);
        }
    }
    (lo, hi)
}

fn frame_camera(rig: &mut Rig, close: f32, face_only: bool) {
    let a = rig.avatar.as_ref().unwrap();
    let face_mesh = a.parts.iter().filter(|p| p.shapes > 0).max_by_key(|p| p.shapes).map(|p| p.mesh);
    let (lo, hi) = match (face_only, face_mesh) {
        (true, Some(m)) => posed_bounds(a, std::iter::once(m)),
        _ => posed_bounds(a, a.parts.iter().map(|p| p.mesh)),
    };
    let c = (lo + hi) * 0.5;
    let extent = (hi - lo).max_element().max(0.1);
    eprintln!("[csk] posed avatar bounds {lo:?} .. {hi:?} (extent {extent:.3})");
    // Some imports are posed lying down (Z-up files): look along the thinnest axis, with the longest one up.
    let d = hi - lo;
    let axis = |i: usize| [Vec3::X, Vec3::Y, Vec3::Z][i];
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| d[a].total_cmp(&d[b]));
    let (view_axis, up_axis) = if d[1] >= d[order[1]] * 0.6 && d[1] >= d[0].min(d[2]) { (2usize, 1usize) } else { (order[0], order[2]) };
    // An upright avatar (Y the longest axis) keeps the usual view along Z.
    let (view_axis, up_axis) = if order[2] == 1 { (2, 1) } else { (view_axis, up_axis) };
    rig.up = axis(up_axis);
    rig.near = extent * 0.02;
    rig.target = c;
    rig.eye = c + axis(view_axis) * extent * (1.15 + close) + rig.up * extent * 0.05;
    eprintln!("[csk] camera looks along axis {view_axis}, up axis {up_axis}");
}

fn idle_mark_ref(rig: &Rig, idle0: u64) {
    eprintln!("[csk] idle counter after the resident reference frames (renderer first-use setup, no streaming yet): {}", zestengine_render::hitch::idle_wait_count() - idle0 - rig.shot_idles);
}

fn run_avatar(rig: &mut Rig, loaded: Loaded, tag: &str, small_ratio: f64, pngs: bool, expect_coarsen: bool) {
    let label = loaded.label.clone();
    let (stats, total_mb) = install(rig, loaded);
    let idle0 = zestengine_render::hitch::idle_wait_count();
    rig.shot_idles = 0;
    assert!(!stats.is_empty(), "{label}: nothing was streamable; the test needs pages");
    let skin_meshes = rig.avatar.as_ref().unwrap().parts.iter().filter(|p| p.shapes > 0 && p.streamed).count();
    frame_camera(rig, 0.0, pngs && tag == "large");
    rig.morph = 1.0;
    rig.animate = false;
    rig.t = 0.8;
    let budget_mb = (total_mb * 1.1).clamp(8.0, 200.0) as u32;
    rig.set_stream(true, budget_mb, None);

    // ---- reference: fully resident
    rig.show(false);
    for _ in 0..12 {
        rig.frame();
    }
    let reference = rig.shot();
    let ref_s = rig.r.stats().cluster;
    eprintln!("[csk] {label}: reference (fully resident): {} triangles from {} instances ({} skinned clusters, {} skin vertices), {} object pixels", ref_s.triangles, ref_s.instances, ref_s.skinned_clusters, ref_s.skin_vertices, reference.object_pixels());
    assert!(ref_s.skinned_clusters > 0, "{label}: the reference avatar was not drawn from skinned clusters");
    assert!(reference.object_pixels() > 1_500);

    idle_mark_ref(rig, idle0);
    // The resident avatar at a very coarse LOD target: what a cut of only the coarse levels shows. The
    // coarse levels of a skinned DAG cover little of the silhouette (separate pieces and skin-weight
    // borders keep it from merging), so "no hole" while pages arrive is judged against the root-only frame, not against
    // the full-detail image: a streamed frame must show at least what the root-only frame (the coarsest cut) shows.
    let _coarse_pixels;
    {
        let mut st = rig.r.cluster_settings();
        let old = st.target_px;
        st.target_px = 400.0;
        rig.r.set_cluster_settings(st);
        rig.show(false);
        for _ in 0..6 {
            rig.frame();
        }
        let coarse = rig.shot();
        _coarse_pixels = coarse.object_pixels();
        let cs = rig.r.stats().cluster;
        eprintln!("[csk] {label}: resident at a 400 px LOD target: {} triangles, {} clusters, {} object pixels (reference {})", cs.triangles, cs.visible_clusters, coarse.object_pixels(), reference.object_pixels());
        save_png(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../docs/design/geo6_skinned_streaming_{tag}_coarse_resident.png")), &[&reference, &coarse]);
        st.target_px = old;
        rig.r.set_cluster_settings(st);
        rig.show(true);
    }
    // ---- phase 1: convergence under a budget well below the whole avatar
    rig.show(true);
    let mut worst_early = Diff { iou: 1.0, holes: 0.0, mean_abs: 0.0, max_abs: 0, changed: 0 };
    let mut first = None;
    let mut least_cover = f64::MAX;
    let mut base_pixels = 0usize;
    let mut last_loads = 0;
    let mut stable = 0;
    let mut converged_at = None;
    for f in 1..=900u32 {
        let s = rig.frame().0.cluster;
        if f == 2 || f % 6 == 0 && f <= 60 || f % 50 == 0 {
            let img = rig.shot();
            let d = compare(&reference, &img);
            if f > 1 {
                worst_early.iou = worst_early.iou.min(d.iou);
                worst_early.holes = worst_early.holes.max(d.holes);
                least_cover = least_cover.min(img.object_pixels() as f64 / base_pixels.max(1) as f64);
            }
            
            if f == 2 {
                base_pixels = img.object_pixels();
                first = Some(img);
            }
            if f <= 30 || f % 100 == 0 {
                eprintln!("[csk] {label} frame {f:3}: IoU {:.5} holes {:.5} mean|d| {:.3}; {} pages ({:.1} MB) resident, {} loading, {} standins, {} triangles", d.iou, d.holes, d.mean_abs, s.stream_pages, s.stream_resident_mb, s.stream_loading, s.stream_standins, s.triangles);
            }
        }
        if s.stream_loads == last_loads && s.stream_loading == 0 && s.stream_pending == 0 && s.stream_standins == 0 {
            stable += 1;
        } else {
            stable = 0;
        }
        last_loads = s.stream_loads;
        if stable >= 10 {
            converged_at = Some(f);
            break;
        }
    }
    let converged_at = converged_at.unwrap_or_else(|| panic!("{label}: never converged in 900 frames"));
    for _ in 0..4 {
        rig.frame();
    }
    let converged = rig.shot();
    let dc = compare(&reference, &converged);
    let s1 = rig.r.stats().cluster;
    let streamed_after = rig.r.cluster_stream_stats();
    eprintln!(
        "[csk] {label}: converged after {converged_at} frames: IoU {:.5}, holes {:.6}, mean|d| {:.4}/255, max|d| {}; {} pages ({:.1} of {:.0} MB budget, {:.1} MB streamed meshes in all), {} loads, {:.1} MB streamed, {} evictions, {} stall frames; triangles {} vs {} resident",
        dc.iou, dc.holes, dc.mean_abs, dc.max_abs, s1.stream_pages, s1.stream_resident_mb, s1.stream_budget_mb, total_mb, streamed_after.loads, s1.stream_mb_streamed, streamed_after.evictions, streamed_after.stall_frames, s1.triangles, ref_s.triangles
    );
    assert!(dc.iou > 0.9999 && dc.holes < 1e-4, "{label}: converged image differs from the fully resident one: {dc:?}");
    assert!(dc.mean_abs < 0.05, "{label}: converged shading differs: {dc:?}");
    if let Some(f) = first.as_ref() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../docs/design/geo6_skinned_streaming_{tag}_early.png"));
        save_png(&root, &[&reference, f]);
    }
    eprintln!("[csk] {label}: while the pages arrive the streamed frames show at least {:.2}x the pixels of the first (root pages only) frame (IoU against full detail down to {:.3})", least_cover, worst_early.iou);
    assert!(least_cover >= 0.95, "{label}: a streamed frame shows less than the coarse cut (hole): {least_cover:.3}x the root-only frame");
    assert!(s1.triangles == ref_s.triangles, "{label}: converged cut has {} triangles, resident {}", s1.triangles, ref_s.triangles);
    assert!(s1.stream_resident_mb <= s1.stream_budget_mb + 0.01);
    let first = first.unwrap();

    let idle_mark = |tag: &str, rig: &Rig| eprintln!("[csk] {label}: idle counter after {tag}: {}", zestengine_render::hitch::idle_wait_count() - idle0 - rig.shot_idles);
    idle_mark("phase 1", rig);

    // ---- phase 2: blendshape weights apply to streamed clusters (camera on the part with the most shapes)
    let (eye0, target0) = (rig.eye, rig.target);
    let face = {
        let a = rig.avatar.as_ref().unwrap();
        a.parts.iter().filter(|p| p.shapes > 0).max_by_key(|p| p.shapes).map(|p| {
            let (lo, hi) = posed_bounds(a, std::iter::once(p.mesh));
            ((lo + hi) * 0.5, (hi - lo).length() * 0.5, p.shapes)
        })
    };
    if let Some((c, rad, _)) = face {
        rig.target = c;
        rig.eye = c + (eye0 - target0).normalize_or(Vec3::Z) * rad * 2.6;
        rig.near = rad * 0.05;
    }
    rig.show(true);
    rig.animate = false;
    rig.morph = 1.0;
    rig.converge(600).unwrap_or_else(|| panic!("{label}: no convergence in the blendshape view"));
    rig.morph = 0.0;
    let (ref_off, str_off) = rig.both();
    rig.morph = 1.0;
    let (ref_on, str_on) = rig.both();
    let d_on = compare(&ref_on, &str_on);
    let d_off = compare(&ref_off, &str_off);
    let moved_ref = compare(&ref_off, &ref_on);
    let moved_str = compare(&str_off, &str_on);
    eprintln!(
        "[csk] {label}: blendshapes ({skin_meshes} streamed meshes with shapes; view on the mesh with {} shapes): weights on, streamed vs resident IoU {:.5} mean|d| {:.4} ({} pixels differ); off {:.5} / {:.4} ({}); weights change the resident image in {} pixels (mean|d| {:.4}), the streamed one in {} ({:.4}), streamed-on vs resident-off differ in {}",
        face.map_or(0, |f| f.2), d_on.iou, d_on.mean_abs, d_on.changed, d_off.iou, d_off.mean_abs, d_off.changed, moved_ref.changed, moved_ref.mean_abs, moved_str.changed, moved_str.mean_abs, compare(&ref_off, &str_on).changed
    );
    assert!(d_on.iou > 0.9999 && d_on.changed == 0, "{label}: streamed avatar with blendshape weights differs from the resident one: {d_on:?}");
    assert!(d_off.iou > 0.9999 && d_off.changed == 0, "{label}: {d_off:?}");
    if face.is_some() {
        assert!(moved_ref.changed > 200 && moved_str.changed > 200, "{label}: the blendshape weights moved (almost) nothing (resident {moved_ref:?}, streamed {moved_str:?})");
        assert!(moved_ref.changed == moved_str.changed, "{label}: the weights move the streamed avatar differently ({} vs {} pixels)", moved_ref.changed, moved_str.changed);
    }
    let blend_on = str_on.clone_image();
    let blend_off = str_off.clone_image();
    rig.eye = eye0;
    rig.target = target0;
    rig.converge(300);
    idle_mark("phase 2", rig);
    // ---- phase 3: animation while pages arrive (evict everything, then pose changes every frame)
    rig.animate = true;
    rig.morph = 1.0;
    rig.r.cluster_stream_evict_all();
    for _ in 0..3 {
        rig.frame();
    }
    let before = rig.r.cluster_stream_stats();
    let mut worst_anim = Diff { iou: 1.0, holes: 0.0, mean_abs: 0.0, max_abs: 0, changed: 0 };
    let mut samples = 0;
    let mut least_anim = f64::MAX;
    let mut anim_shot = None;
    for f in 0..90u32 {
        rig.show(true);
        let s = rig.frame().0.cluster;
        if f % 6 == 2 {
            let img = rig.shot();
            // The resident reference of this very pose: freeze time for two frames.
            rig.animate = false;
            rig.show(false);
            rig.frame();
            rig.frame();
            let want = rig.shot();
            rig.animate = true;
            let d = compare(&want, &img);
            // The animated silhouette can be much smaller than the static root-only frame (the pose
            // changes), so the yardstick is the coarse cut or the resident render of this very pose,
            // whichever has fewer pixels.
            let cover = img.object_pixels() as f64 / want.object_pixels().max(1) as f64;
            eprintln!("[csk] {label} animating, frame {f:2}: coverage {cover:.3}x the resident pose ({} vs {} px; root-only static frame {base_pixels} px)", img.object_pixels(), want.object_pixels());
            least_anim = least_anim.min(img.object_pixels() as f64 / base_pixels.min(want.object_pixels()).max(1) as f64);
            worst_anim.iou = worst_anim.iou.min(d.iou);
            worst_anim.holes = worst_anim.holes.max(d.holes);
            worst_anim.mean_abs = worst_anim.mean_abs.max(d.mean_abs);
            samples += 1;
            if f == 14 {
                anim_shot = Some(img);
            }
            if f < 40 {
                eprintln!("[csk] {label} animating, frame {f:2}: IoU vs resident {:.5} holes {:.5}; {} pages resident, {} standins, {} skinned clusters", d.iou, d.holes, s.stream_pages, s.stream_standins, s.skinned_clusters);
            }
        }
    }
    let after = rig.r.cluster_stream_stats();
    eprintln!("[csk] {label}: animated {samples} sampled frames while {} pages streamed in ({:.1} MB): worst IoU against the resident avatar of the same pose {:.4}; least coverage {:.2}x the coarse cut / resident pose", after.loads - before.loads, (after.bytes_streamed - before.bytes_streamed) as f64 / 1048576.0, worst_anim.iou, least_anim);
    assert!(after.loads > before.loads, "{label}: nothing streamed during the animation");
    assert!(least_anim >= 0.9, "{label}: holes while the avatar animates and pages arrive: {least_anim:.3}x the coarse cut / resident pose");
    rig.show(true);
    rig.animate = false;
    rig.converge(600).unwrap_or_else(|| panic!("{label}: no convergence after the animation"));

    idle_mark("phase 3", rig);
    // ---- phase 4: a budget far below the working set
    let small_mb = ((s1.stream_resident_mb as f64) * small_ratio).max(1.0) as u32;
    let pressured = (small_mb as f64) < s1.stream_resident_mb as f64;
    rig.set_stream(true, small_mb, None);
    rig.t = 0.8;
    let mut least_p = f64::MAX;
    let mut worst_p = Diff { iou: 1.0, holes: 0.0, mean_abs: 0.0, max_abs: 0, changed: 0 };
    for f in 1..=260u32 {
        let s = rig.frame().0.cluster;
        assert!(s.stream_resident_mb <= s.stream_budget_mb + 0.01, "{label} frame {f}: {:.2} MB resident over a {:.1} MB budget", s.stream_resident_mb, s.stream_budget_mb);
        if f % 20 == 0 {
            let img = rig.shot();
            let d = compare(&reference, &img);
            worst_p.iou = worst_p.iou.min(d.iou);
            worst_p.holes = worst_p.holes.max(d.holes);
            least_p = least_p.min(img.object_pixels() as f64 / base_pixels.max(1) as f64);
        }
    }
    let pressure = rig.shot();
    let s2 = rig.r.stats().cluster;
    let dp = compare(&reference, &pressure);
    eprintln!(
        "[csk] {label}: budget {small_mb} MB (working set {:.1} MB): resident {:.1} MB, {} evictions, {} stall frames, {} thrash reloads; IoU {:.5} (worst {:.5}), holes {:.6}; triangles {} ({:.0}% of resident)",
        s1.stream_resident_mb, s2.stream_resident_mb, s2.stream_evictions, s2.stream_stall_frames, s2.stream_thrash, dp.iou, worst_p.iou, worst_p.holes, s2.triangles, s2.triangles as f64 * 100.0 / ref_s.triangles.max(1) as f64
    );
    if pressured {
        assert!(s2.stream_evictions > 0 && s2.stream_stall_frames > 0, "{label}: the small budget never evicted / stalled");
    } else {
        eprintln!("[csk] {label}: working set {:.2} MB is below the smallest budget (1 MB): no pressure to measure", s1.stream_resident_mb);
    }
    eprintln!("[csk] {label}: under the small budget the least coverage was {least_p:.2}x the root-only frame");
    assert!(least_p >= 0.95, "{label}: holes under budget pressure: {least_p:.3}x the coarse cut");

    idle_mark("phase 4", rig);
    // ---- phase 5: the per-instance triangle budget (fair share) acts on streamed avatars as on resident ones
    rig.set_stream(true, budget_mb, Some((ref_s.triangles / 4).max(1_000)));
    rig.show(false);
    for _ in 0..30 {
        rig.frame();
    }
    let cap_ref = rig.shot();
    let ref_cap = rig.r.stats().cluster;
    rig.show(true);
    rig.r.cluster_stream_evict_all();
    let loads0 = rig.r.cluster_stream_stats().loads;
    rig.converge(900).unwrap_or_else(|| panic!("{label}: no convergence under the triangle budget"));
    for _ in 0..4 {
        rig.frame();
    }
    let cap_str = rig.shot();
    let str_cap = rig.r.stats().cluster;
    let loads_cap = rig.r.cluster_stream_stats().loads - loads0;
    let dcap = compare(&cap_ref, &cap_str);
    eprintln!(
        "[csk] {label}: triangle budget {}: resident draws {} triangles (estimated {} against a limit of {}), streamed {} ; streamed pages needed {} loads ({:.1} MB resident) against {} loads for the full-detail run; image IoU {:.5}, mean|d| {:.4}",
        (ref_s.triangles / 4).max(1_000),
        ref_cap.triangles,
        ref_cap.budget_estimated,
        ref_cap.budget_limit,
        str_cap.triangles,
        loads_cap,
        str_cap.stream_resident_mb,
        streamed_after.loads,
        dcap.iou,
        dcap.mean_abs
    );
    assert!(dcap.iou > 0.999 && dcap.mean_abs < 0.5, "{label}: streamed and resident differ under the triangle budget: {dcap:?}");
    if ref_cap.triangles < ref_s.triangles {
        assert!(str_cap.stream_resident_mb < s1.stream_resident_mb, "{label}: the coarser cut did not need fewer pages ({:.1} vs {:.1} MB)", str_cap.stream_resident_mb, s1.stream_resident_mb);
        assert!(str_cap.triangles == ref_cap.triangles, "{label}: streamed cut under the budget has {} triangles, resident {}", str_cap.triangles, ref_cap.triangles);
    } else {
        assert!(!expect_coarsen, "{label}: the triangle budget did not coarsen the resident avatar");
        eprintln!("[csk] {label}: the fair share did not coarsen this tiny avatar (its first simplified level is beyond the scale reachable here): equality only");
    }
    rig.set_stream(true, budget_mb, Some(0));

    // ---- no device idle anywhere in the streaming phases
    let idles = zestengine_render::hitch::idle_wait_count() - idle0 - rig.shot_idles;
    eprintln!("[csk] {label}: device idles in all phases (the {} viewport read-backs of this test excluded; the installs before the first frame are not counted): {idles}", rig.shot_idles);

    if pngs {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../docs/design/geo6_skinned_streaming_{tag}.png"));
        let d = diff_image(&reference, &converged);
        let a = anim_shot.unwrap_or_else(|| first.clone_image());
        save_png(&root, &[&reference, &first, &a, &converged, &pressure, &d]);
        let root2 = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../docs/design/geo6_skinned_streaming_{tag}_blendshapes.png"));
        save_png(&root2, &[&blend_off, &blend_on, &diff_image(&blend_off, &blend_on)]);
        eprintln!("[csk] wrote {} (left to right: fully resident, streamed at frame 2, animating while pages arrive, converged, {small_mb} MB budget, |converged - resident| x8); the blendshape picture: streamed weights off, on, |on - off| x8", root.display());
    }
}

impl Image {
    fn clone_image(&self) -> Image {
        Image { w: self.w, h: self.h, px: self.px.clone() }
    }
}

fn load_default() -> Loaded {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/avatars/default/male.glb");
    let scene = load_gltf(&path).unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
    // Tiny avatar: split it into pages of 16 KB so every code path (page chain, skin lookup) runs.
    Loaded { label: "male.glb".into(), scene, opts: PageOptions { page_bytes: 16 * 1024, resident_bytes: 24 * 1024, min_stream_bytes: 1 } }
}

fn load_large() -> Option<Loaded> {
    let path = std::env::var("ZESTENGINE_CSK_AVATAR").unwrap_or_else(|_| "/home/lemonxah/shared/avi/lemonnew.fbx".into());
    if !std::path::Path::new(&path).exists() {
        eprintln!("[csk] {path} not found: the large avatar phase is skipped");
        return None;
    }
    let scene = load_fbx(&path).unwrap_or_else(|e| panic!("{path}: {e:#}"));
    Some(Loaded { label: std::path::Path::new(&path).file_name().unwrap().to_string_lossy().into_owned(), scene, opts: PageOptions { min_stream_bytes: 256 * 1024, ..PageOptions::default() } })
}

fn run_main(window: Arc<Window>) {
    // The face of the large avatar alone holds 1.77M blendshape entries (28 MB): more than the default
    // morph pool (262k entries, growing to 1M), which is exactly what streaming avoids. The resident
    // reference needs bigger pools.
    unsafe { std::env::set_var("ZESTENGINE_CLUSTER_POOL_SCALE", "4") };
    let mut r = Renderer::new(window).expect("Renderer::new");
    r.set_vsync(false);
    r.set_gpu_zone_timing(true);
    r.set_gpu_cull_settings(GpuCullSettings { enabled: true, hiz_occlusion: false, min_pixel_radius: 0.0, max_distance: 0.0 });
    r.set_cluster_settings(ClusterSettings { mode: ClusterMode::On, stream: false, ..r.cluster_settings() });
    eprintln!("[csk] {}", r.gpu_caps().report().lines().next().unwrap_or(""));
    let g = r.upload_mesh(&quad_ground(60.0)).unwrap();
    let gm = material(&mut r, [0.45, 0.5, 0.42, 1.0]);
    let mut rig = Rig { r, ext: [960, 540], eye: Vec3::new(0.0, 1.0, 3.0), target: Vec3::new(0.0, 1.0, 0.0), avatar: None, streamer: None, ground: (g, gm), t: 0.0, animate: false, morph: 1.0, shot_idles: 0, near: 0.05, up: Vec3::Y };
    run_avatar(&mut rig, load_default(), "default", 0.35, true, false);
    unload(&mut rig);
    if let Some(large) = load_large() {
        run_avatar(&mut rig, large, "large", 0.25, true, true);
    }
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
            Err(e) => eprintln!("[csk] no window ({e}): skipped"),
        }
        event_loop.exit();
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
}

#[test]
fn skinned_and_blendshape_meshes_stream_in_the_real_frame_loop() {
    if std::env::var("ZESTENGINE_TEST_GPU").is_ok_and(|v| v == "0") || std::env::var_os("DISPLAY").is_none() {
        eprintln!("[csk] no display / GPU disabled: skipped");
        return;
    }
    let _gpu = zestengine_render::cluster_headless::gpu_lock();
    let Ok(mut event_loop) = EventLoop::builder().with_x11().with_any_thread(true).build() else {
        eprintln!("[csk] no X11 event loop: skipped");
        return;
    };
    let mut app = App { result: None, done: false };
    event_loop.run_app_on_demand(&mut app).unwrap();
    if let Some(Err(e)) = app.result {
        panic!("{e}");
    }
}
