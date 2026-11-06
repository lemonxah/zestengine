//! GEO-4/5 in the REAL frame loop: a full `Renderer` (invisible X11 window, real
//! device) renders a scene (ground, DamagedHelmets, GPU-skinned avatars) with
//! `cluster_rendering` Off and On and the final viewport images are compared;
//! then the long-running behaviours of the live loop: 300+ frames with the
//! early / late passes (Hi-Z on), resize, runtime toggling, skinned-pool
//! overflow pause-and-resume, the budget controller against the real
//! `gpu_frame_ms`, and the measured timing table.
//!
//! One `#[test]` runs every phase in order: winit allows one event loop per
//! process. Skips cleanly with no display / GPU (`ZESTENGINE_TEST_GPU=0` forces
//! that). Run with `--nocapture` to see the numbers:
//! `cargo test -p zestengine-assets --test cluster_frame -- --nocapture`
//! (`ZESTENGINE_CF_FRAMES=<n>` changes the long-run length, `ZESTENGINE_CF_AVATARS=<n>` the bench count).

use std::sync::Arc;

use glam::{Mat4, Quat, Vec3};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::{Window, WindowId};
use zestengine_assets::cluster_dag::DagOptions;
use zestengine_assets::{Scene, geo_cache, load_gltf};
use zestengine_render::{
    CameraData, ClusterMode, ClusterSettings, DrawCmd, FrameInput, GpuCullSettings, LightData, LightInstance, LightKind,
    MAX_REFLECTION_PROBES, MaterialDesc, MaterialHandle, MaterialParams, MeshData, MeshHandle, PostFx, RenderStats, Renderer,
    ShadowCasting, Vertex,
};

fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn env_n(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

// ---------------------------------------------------------------------------
// Scene

struct Avatar {
    /// (per-copy mesh handle, base material) per skinned mesh of the model.
    slots: Vec<(MeshHandle, MaterialHandle, f32, Vec3)>,
    model: usize,
    pos: Vec3,
    phase: f32,
}

struct Model {
    scene: Scene,
    /// Skinned mesh indices.
    skinned: Vec<usize>,
}

/// A smooth torus of `nu * nv * 2` triangles (R = 1, r = 0.35), centred on the origin.
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
                color: [1.0; 4],
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

/// Upright cylinder from y = 0 to `height`, `segs` x `rings` quads (smooth normals).
fn cylinder(radius: f32, height: f32, segs: u32, rings: u32) -> MeshData {
    let mut vertices = Vec::new();
    for r in 0..=rings {
        let y = height * r as f32 / rings as f32;
        for s in 0..=segs {
            let a = std::f32::consts::TAU * s as f32 / segs as f32;
            let n = Vec3::new(a.cos(), 0.0, a.sin());
            vertices.push(Vertex {
                position: [n.x * radius, y, n.z * radius],
                normal: n.to_array(),
                tangent: [-a.sin(), 0.0, a.cos(), 1.0],
                color: [1.0; 4],
                uv: [s as f32 / segs as f32, r as f32 / rings as f32],
                ..Default::default()
            });
        }
    }
    let mut indices = Vec::new();
    for r in 0..rings {
        for s in 0..segs {
            let a = r * (segs + 1) + s;
            let b = a + segs + 1;
            indices.extend([a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    MeshData { vertices, indices, has_lightmap_uv: false }
}

struct World {
    /// A dense smooth torus (the heavy static mesh) drawn at `torus_grid`.
    torus: Option<(MeshHandle, MaterialHandle, Vec3, f32)>,
    torus_grid: Vec<Vec3>,
    ground: (MeshHandle, MaterialHandle),
    helmet: Option<(MeshHandle, MaterialHandle, Vec3, f32)>,
    helmet_grid: Vec<Vec3>,
    models: Vec<Model>,
    avatars: Vec<Avatar>,
    /// Template (DAG-carrying) mesh handles per (model, skinned slot).
    templates: Vec<Vec<MeshHandle>>,
    materials: Vec<MaterialHandle>,
    /// An impostor billboard drawn each frame (atlas, model transform).
    impostor: Option<(zestengine_render::ImpostorHandle, Mat4)>,
    /// Extra static draws (mesh, material, transform, centre, radius).
    extra: Vec<(MeshHandle, MaterialHandle, Mat4, Vec3, f32)>,
}

fn quad_ground(half: f32) -> MeshData {
    let v = |x: f32, z: f32| Vertex {
        position: [x, 0.0, z],
        normal: [0.0, 1.0, 0.0],
        tangent: [1.0, 0.0, 0.0, 1.0],
        color: [1.0; 4],
        uv: [x * 0.1, z * 0.1],
        ..Default::default()
    };
    MeshData { vertices: vec![v(-half, -half), v(half, -half), v(half, half), v(-half, half)], indices: vec![0, 2, 1, 0, 3, 2], has_lightmap_uv: false }
}

fn aabb(m: &MeshData) -> (Vec3, f32) {
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for v in &m.vertices {
        lo = lo.min(Vec3::from(v.position));
        hi = hi.max(Vec3::from(v.position));
    }
    ((lo + hi) * 0.5, (hi - lo).length() * 0.5)
}

fn material(r: &mut Renderer, color: [f32; 4], rough: f32, metal: f32) -> MaterialHandle {
    r.create_material(&MaterialDesc {
        name: "cluster_frame".into(),
        params: MaterialParams { base_color: color, roughness: rough, metallic: metal, ..Default::default() },
        ..Default::default()
    })
    .unwrap()
}

fn has_weights(m: &MeshData) -> bool {
    !m.vertices.is_empty() && m.vertices.iter().any(|v| v.weights.iter().sum::<f32>() > 0.5)
}

fn load_model(file: &str) -> Model {
    let path = root().join("models/avatars/default").join(file);
    let scene = load_gltf(&path).unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
    let skinned = (0..scene.meshes.len()).filter(|&i| has_weights(&scene.meshes[i])).collect();
    Model { scene, skinned }
}

impl World {
    fn new(r: &mut Renderer, helmet: bool, models: bool) -> Self {
        let g = r.upload_mesh(&quad_ground(60.0)).unwrap();
        let gm = material(r, [0.45, 0.5, 0.42, 1.0], 0.8, 0.0);
        let mut w = World {
            torus: None,
            torus_grid: Vec::new(),
            extra: Vec::new(),
            impostor: None,
            ground: (g, gm),
            helmet: None,
            helmet_grid: Vec::new(),
            models: Vec::new(),
            avatars: Vec::new(),
            templates: Vec::new(),
            materials: (0..4).map(|i| material(r, [[0.9, 0.7, 0.6, 1.0], [0.3, 0.5, 0.9, 1.0], [0.8, 0.3, 0.3, 1.0], [0.6, 0.8, 0.5, 1.0]][i], 0.5, 0.0)).collect(),
        };
        if helmet {
            let (scene, _) = {
                let s = load_gltf(root().join("models/DamagedHelmet.glb")).expect("DamagedHelmet");
                (s, ())
            };
            let geo = geo_cache::build_model(&scene, &DagOptions::default());
            let mesh = &scene.meshes[0];
            let h = r.upload_mesh(mesh).unwrap();
            let dag = geo.meshes[0].as_ref().expect("the helmet qualifies for a DAG");
            eprintln!(
                "[cf] DamagedHelmet: {} tris, {} clusters, coloured {}",
                mesh.indices.len() / 3,
                dag.clusters.len(),
                dag.header.flags & zestengine_assets::cluster_dag::FLAG_EXTRA != 0
            );
            r.install_cluster_dag(h, &dag.to_gpu_upload()).ok();
            let (c, rad) = aabb(mesh);
            let hm = material(r, [0.7, 0.7, 0.75, 1.0], 0.35, 0.6);
            w.helmet = Some((h, hm, c, rad));
        }
        if models {
            for f in ["male.glb", "female.glb"] {
                w.models.push(load_model(f));
            }
            for m in &w.models {
                let geo = geo_cache::build_model(&m.scene, &DagOptions::default());
                let mut t = Vec::new();
                for &mi in &m.skinned {
                    let mesh = &m.scene.meshes[mi];
                    let h = r.upload_mesh(mesh).unwrap();
                    match geo.meshes[mi].as_ref() {
                        Some(dag) => {
                            let up = dag.to_gpu_upload();
                            if let Err(e) = r.install_cluster_dag(h, &up) {
                                eprintln!("[cf] avatar mesh {mi} DAG refused: {e:#}");
                            }
                        }
                        None => eprintln!("[cf] avatar mesh {mi} ({} tris) has no DAG", mesh.indices.len() / 3),
                    }
                    t.push(h);
                }
                w.templates.push(t);
            }
        }
        w
    }

    /// The 1M-triangle torus with its DAG (the path must be requested).
    fn add_torus(&mut self, r: &mut Renderer, nu: u32, nv: u32) {
        let mesh = torus(nu, nv);
        let t = std::time::Instant::now();
        let dag = zestengine_assets::cluster_dag::build_cluster_dag(&mesh, &[], &DagOptions::default()).expect("torus DAG");
        let mesh = &mesh;
        eprintln!("[cf] torus: {} tris, {} clusters, DAG built in {:.1} s", mesh.indices.len() / 3, dag.clusters.len(), t.elapsed().as_secs_f32());
        let h = r.upload_mesh(mesh).unwrap();
        r.install_cluster_dag(h, &dag.to_gpu_upload()).expect("torus DAG installs");
        let (c, rad) = aabb(mesh);
        let m = material(r, [0.8, 0.45, 0.2, 1.0], 0.4, 0.2);
        self.torus = Some((h, m, c, rad));
    }

    fn add_avatars(&mut self, r: &mut Renderer, n: usize) {
        let cols = (n as f32).sqrt().ceil().max(1.0) as usize;
        let rows = n.div_ceil(cols);
        let (cx, cz) = ((cols - 1) as f32 * 0.5, (rows - 1) as f32 * 0.5);
        for k in 0..n {
            let model = k % self.models.len();
            let m = &self.models[model];
            let joints = m.scene.skeletons[0].joints.len();
            let mut slots = Vec::new();
            for (si, &mi) in m.skinned.iter().enumerate() {
                let mesh = &m.scene.meshes[mi];
                let shapes: Vec<zestengine_render::MorphDeltas<'_>> =
                    m.scene.morphs.get(mi).map(|v| v.iter().map(|t| t.deltas.as_slice()).collect()).unwrap_or_default();
                let h = r.upload_mesh_gpu_skinned_morphed(mesh, joints, &shapes).unwrap();
                if r.has_cluster_dag(self.templates[model][si]) {
                    if let Err(e) = r.alias_cluster_dag(h, self.templates[model][si]) {
                        eprintln!("[cf] alias failed: {e:#}");
                    }
                } else if k == 0 {
                    eprintln!("[cf] template {model}/{si} lost its DAG");
                }
                let (c, rad) = aabb(mesh);
                slots.push((h, self.materials[(k + si) % self.materials.len()], rad * 1.4, c));
            }
            self.avatars.push(Avatar {
                slots,
                model,
                pos: Vec3::new(((k % cols) as f32 - cx) * 1.5, 0.0, ((k / cols) as f32 - cz) * 1.5),
                phase: k as f32 * 0.7,
            });
        }
    }

    fn clear_avatars(&mut self, r: &mut Renderer) {
        let meshes: std::collections::HashSet<usize> = self.avatars.iter().flat_map(|a| a.slots.iter().map(|s| s.0.index())).collect();
        // The renderer frees GPU meshes by handle set.
        let _ = r.free_resources(&meshes, &Default::default(), &Default::default());
        self.avatars.clear();
    }

    /// Pose every avatar for time `t` (a sway of every joint) and queue the palettes.
    fn pose(&self, r: &mut Renderer, t: f32) {
        for a in &self.avatars {
            let skel = &self.models[a.model].scene.skeletons[0];
            let mut locals = skel.rest_locals();
            for (i, l) in locals.iter_mut().enumerate() {
                let ang = ((t * 1.7 + a.phase + i as f32 * 0.37).sin()) * 0.25;
                *l *= Mat4::from_quat(Quat::from_rotation_x(ang) * Quat::from_rotation_z(ang * 0.5));
            }
            let palette = skel.palette(&locals);
            for s in &a.slots {
                r.skin_mesh_gpu(s.0, &palette);
            }
        }
    }

    fn draws(&self) -> Vec<DrawCmd> {
        let base = |mesh: MeshHandle, material: MaterialHandle, transform: Mat4, c: Vec3, rad: f32| DrawCmd {
            gi_source: None,
            mesh,
            material,
            transform,
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
        let mut d = vec![base(self.ground.0, self.ground.1, Mat4::IDENTITY, Vec3::ZERO, 90.0)];
        if let Some((h, m, c, rad)) = self.helmet {
            for p in &self.helmet_grid {
                d.push(base(h, m, Mat4::from_translation(*p) * Mat4::from_rotation_x(std::f32::consts::FRAC_PI_2), c, rad));
            }
        }
        for (h, m, t, c, rad) in &self.extra {
            d.push(base(*h, *m, *t, *c, *rad));
        }
        if let Some((h, m, c, rad)) = self.torus {
            for p in &self.torus_grid {
                d.push(base(h, m, Mat4::from_translation(*p) * Mat4::from_scale(Vec3::splat(2.0)), c, rad * 2.0));
            }
        }
        for a in &self.avatars {
            for s in &a.slots {
                d.push(base(s.0, s.1, Mat4::from_translation(a.pos), s.3, s.2));
            }
        }
        d
    }
}

// ---------------------------------------------------------------------------
// Frame driver

struct Rig {
    r: Renderer,
    w: World,
    ext: [u32; 2],
    t: f32,
    frames: u64,
    shadows: bool,
    eye: Vec3,
    target: Vec3,
    /// Draw only the ground (the reference for object masks).
    ground_only: bool,
    hiz: bool,
    /// Hold the animation clock (image comparisons need identical poses).
    frozen: bool,
    /// The shipped game's path: the scene renders into the HDR target and the
    /// post pass writes the swapchain (no readable viewport image).
    game: bool,
    /// Replaces the default sun (cast_shadows follows `shadows`).
    lights: Vec<LightInstance>,
    /// 0 = everything, 1 = nothing, 2 = the extra draws only, 3 = the impostor only.
    mode: u8,
    /// Vertical field of view (radians) of the frame's camera.
    fov: f32,
}

fn sun(shadows: bool) -> LightInstance {
    LightInstance {
        kind: LightKind::Directional,
        position: Vec3::ZERO,
        direction: Vec3::new(-0.45, -1.0, -0.3).normalize(),
        color: [1.0, 0.97, 0.9],
        intensity: 3.0,
        range: 0.0,
        spot_inner_deg: 0.0,
        spot_outer_deg: 0.0,
        cast_shadows: shadows,
        soft_shadows: true, radius: 0.0,
        shadow_bias: 0.005,
        baked: false,
    }
}

impl Rig {
    fn set_clusters(&mut self, on: bool) {
        let mut s = self.r.cluster_settings();
        s.mode = if on { ClusterMode::On } else { ClusterMode::Off };
        self.r.set_cluster_settings(s);
    }

    fn step(&mut self) -> RenderStats {
        if !self.frozen {
            self.t += 1.0 / 60.0;
        }
        self.w.pose(&mut self.r, self.t);
        let mut draws = self.w.draws();
        if self.ground_only {
            draws.truncate(1);
        }
        match self.mode {
            1 | 3 => draws.clear(),
            2 => {
                draws.remove(0);
            }
            _ => {}
        }
        if let Some((h, m)) = self.w.impostor.filter(|_| self.mode == 0 || self.mode == 3) {
            self.r.draw_impostor(h, m, [0.0; 4]);
        }
        let lights: Vec<LightInstance> = if self.lights.is_empty() {
            vec![sun(self.shadows)]
        } else {
            self.lights.iter().map(|l| LightInstance { cast_shadows: self.shadows, ..*l }).collect()
        };
        let view = Mat4::look_at_rh(self.eye, self.target, Vec3::Y);
        let aspect = self.ext[0] as f32 / self.ext[1] as f32;
        let camera = CameraData { view, proj: Mat4::perspective_rh(self.fov, aspect, 0.1, 300.0), position: self.eye };
        let input = FrameInput {
            clear_color: [0.1, 0.12, 0.2, 1.0],
            camera,
            light: LightData::default(),
            lights: &lights,
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
            render_viewport: self.game,
            viewport_extent: if self.game { None } else { Some(self.ext) },
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
        self.r.render(&input).expect("render");
        self.frames += 1;
        self.r.stats()
    }

    fn run(&mut self, n: usize) -> RenderStats {
        let mut s = RenderStats::default();
        for _ in 0..n {
            s = self.step();
        }
        s
    }

    /// Frames until the cluster path (or the ordinary one) is steady, then the image.
    fn shot(&mut self, settle: usize) -> Image {
        self.run(settle);
        let (w, h, px) = self.r.debug_read_viewport().expect("readback").expect("viewport target");
        Image { w, h, px }
    }

    fn opts(&mut self, hiz: bool) {
        self.hiz = hiz;
        self.r.set_gpu_cull_settings(GpuCullSettings { enabled: true, hiz_occlusion: hiz, min_pixel_radius: 0.0, max_distance: 0.0 });
    }
}

// ---------------------------------------------------------------------------
// Image comparison

struct Image {
    w: u32,
    h: u32,
    px: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
struct Diff {
    mean_abs: f64,
    max: u8,
    over16: f64,
    iou: f64,
    obj_pixels: usize,
}

impl Image {
    /// Pixels where any colour channel differs from `other` by more than `thr`.
    fn mask_vs(&self, other: &Image, thr: u8) -> Vec<bool> {
        self.px.as_chunks::<4>().0.iter().zip(other.px.as_chunks::<4>().0.iter()).map(|(a, b)| (0..3).any(|c| a[c].abs_diff(b[c]) > thr)).collect()
    }
}

/// Compare `a` (reference) with `b`; the object mask is "differs from the
/// ground-only render `ground`".
fn compare(a: &Image, b: &Image, ground: &Image) -> Diff {
    assert_eq!((a.w, a.h), (b.w, b.h));
    let n = (a.w * a.h) as usize;
    let (mut sum, mut max, mut over) = (0u64, 0u8, 0usize);
    for (p, q) in a.px.as_chunks::<4>().0.iter().zip(b.px.as_chunks::<4>().0.iter()) {
        let d = (0..3).map(|c| p[c].abs_diff(q[c])).max().unwrap();
        sum += (0..3).map(|c| p[c].abs_diff(q[c]) as u64).sum::<u64>();
        max = max.max(d);
        over += (d > 16) as usize;
    }
    let (ma, mb) = (a.mask_vs(ground, 6), b.mask_vs(ground, 6));
    let inter = ma.iter().zip(&mb).filter(|(x, y)| **x && **y).count();
    let union = ma.iter().zip(&mb).filter(|(x, y)| **x || **y).count();
    Diff {
        mean_abs: sum as f64 / (n * 3) as f64,
        max,
        over16: over as f64 / n as f64,
        iou: if union == 0 { 1.0 } else { inter as f64 / union as f64 },
        obj_pixels: ma.iter().filter(|x| **x).count(),
    }
}

fn dump(img: &Image, name: &str) {
    if let Some(dir) = std::env::var_os("ZESTENGINE_TEST_DUMP_DIR") {
        let mut out = format!("P6\n{} {}\n255\n", img.w, img.h).into_bytes();
        for p in img.px.as_chunks::<4>().0 {
            out.extend_from_slice(&p[..3]);
        }
        let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{name}.ppm")), out);
    }
}

// ---------------------------------------------------------------------------
// Phases

fn cams() -> [(Vec3, Vec3); 3] {
    [
        (Vec3::new(0.0, 2.2, 9.0), Vec3::new(0.0, 1.0, 0.0)),
        (Vec3::new(7.0, 3.5, 6.0), Vec3::new(0.0, 1.0, 0.0)),
        (Vec3::new(-3.0, 1.4, 4.0), Vec3::new(0.5, 1.0, 0.0)),
    ]
}

/// Off vs On images of the scene from several cameras. Returns the worst IoU.
fn phase_compare(rig: &mut Rig, name: &str) -> (f64, f64) {
    let (mut worst_iou, mut worst_mean) = (1.0f64, 0.0f64);
    rig.shadows = false;
    rig.frozen = true;
    for (i, (eye, target)) in cams().into_iter().enumerate() {
        rig.eye = eye;
        rig.target = target;
        rig.ground_only = true;
        rig.set_clusters(false);
        let ground = rig.shot(4);
        rig.ground_only = false;
        let off_a = rig.shot(4);
        let off_b = rig.shot(2);
        let base = compare(&off_a, &off_b, &ground);
        rig.set_clusters(true);
        let on = rig.shot(6);
        let s = rig.r.stats().cluster;
        let d = compare(&off_a, &on, &ground);
        eprintln!(
            "[cf] {name} cam {i}: Off-vs-Off mean {:.4} max {} | Off-vs-On mean|d| {:.4}/255 max {} px>16 {:.4}% silhouette IoU {:.5} (object px {}) | on: {} instances, {} visible clusters, {} tris, scale {:.2}, paused {}",
            base.mean_abs, base.max, d.mean_abs, d.max, d.over16 * 100.0, d.iou, d.obj_pixels,
            s.instances, s.visible_clusters, s.triangles, s.error_scale, s.paused
        );
        dump(&off_a, &format!("{name}_cam{i}_off"));
        dump(&on, &format!("{name}_cam{i}_on"));
        assert!(base.max == 0, "{name}: two Off frames differ (non-deterministic baseline, max {})", base.max);
        assert!(s.instances > 0 && s.visible_clusters > 0, "{name}: the cluster path drew nothing ({s:?})");
        assert!(!s.paused && s.queue_overflows == 0, "{name}: cluster path paused/overflowed ({s:?})");
        worst_iou = worst_iou.min(d.iou);
        worst_mean = worst_mean.max(d.mean_abs);
    }
    rig.set_clusters(false);
    rig.frozen = false;
    (worst_iou, worst_mean)
}

#[derive(Default, Clone, Copy)]
struct Zones {
    frame: f32,
    cull: f32,
    skin_c: f32,
    draw: f32,
    pre: f32,
    skin: f32,
    shadows: f32,
    scene: f32,
    tris: u32,
    clusters: u32,
}

fn measure(rig: &mut Rig, warm: usize, n: usize) -> Zones {
    rig.run(warm);
    let mut z = Zones::default();
    for _ in 0..n {
        let s = rig.step();
        z.frame += s.gpu_frame_ms;
        z.cull += s.cluster.cull_ms;
        z.skin_c += s.cluster.skin_ms;
        z.draw += s.cluster.draw_ms;
        z.pre += s.cluster.prepass_ms;
        z.skin += s.gpu_skin_ms;
        z.shadows += s.gpu_shadows_ms;
        z.scene += s.gpu_scene_ms;
        z.tris = s.cluster.triangles;
        z.clusters = s.cluster.visible_clusters;
    }
    let f = n as f32;
    Zones { frame: z.frame / f, cull: z.cull / f, skin_c: z.skin_c / f, draw: z.draw / f, pre: z.pre / f, skin: z.skin / f, shadows: z.shadows / f, scene: z.scene / f, ..z }
}

fn row(label: &str, z: Zones) {
    eprintln!(
        "| {label:<34} | {:>8.3} | {:>8.3} | {:>10.3} | {:>8.3} | {:>8.3} | {:>8.3} | {:>8.3} | {:>8.3} | {:>10} |",
        z.frame, z.cull, z.skin_c, z.pre, z.draw, z.skin, z.shadows, z.scene, z.tris
    );
}

fn header(title: &str) {
    eprintln!("\n{title}");
    eprintln!("| {:<34} | {:>8} | {:>8} | {:>10} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>10} |", "config", "frame ms", "CL_CULL", "CL_SKIN", "CL_PRE", "CL_DRAW", "SKIN", "SHADOWS", "SCENE", "cluster tri");
    eprintln!("|{}|", ["-"; 10].iter().zip([36, 10, 10, 12, 10, 10, 10, 10, 10, 12]).map(|(_, w)| "-".repeat(w)).collect::<Vec<_>>().join("|"));
}

/// Cluster-only numbers are meaningless with the path off; the table prints 0.
fn phase_timing(rig: &mut Rig, title: &str, with_shadows: &[bool]) {
    header(title);
    for &sh in with_shadows {
        rig.shadows = sh;
        for hiz in [false, true] {
            rig.opts(hiz);
            for on in [false, true] {
                rig.set_clusters(on);
                let z = measure(rig, 40, 120);
                if on {
                    let c = rig.r.stats().cluster;
                    eprintln!("[cf]   state: paused {} overflows {} skin-pool overflows {} instances {} skinned clusters {} skin vertices {} pool {}%", c.paused, c.queue_overflows, c.skin_pool_overflows, c.instances, c.skinned_clusters, c.skin_vertices, c.pool_percent);
                }
                row(&format!("{} / {} / Hi-Z {}", if on { "clusters On " } else { "clusters Off" }, if sh { "shadows" } else { "no shadow" }, if hiz { "on" } else { "off" }), z);
            }
        }
    }
    rig.opts(false);
}

fn phase_long(rig: &mut Rig, frames: usize) {
    // Orbit the camera with Hi-Z on: the early/late split runs every frame.
    rig.opts(true);
    rig.set_clusters(true);
    rig.shadows = true;
    let (mut max_tris, mut late, mut overflow, mut paused_frames, mut tested) = (0u32, 0u64, 0u32, 0, 0u64);
    let mut hiz_culled = 0u64;
    for f in 0..frames {
        let a = f as f32 * 0.02;
        rig.eye = Vec3::new(a.sin() * 9.0, 2.5 + (a * 0.7).sin(), a.cos() * 9.0);
        rig.target = Vec3::new(0.0, 1.0, 0.0);
        let s = rig.step();
        let c = s.cluster;
        max_tris = max_tris.max(c.triangles);
        late += c.recovered as u64;
        hiz_culled += c.culled_hiz as u64;
        tested += c.clusters_tested as u64;
        overflow = overflow.max(c.queue_overflows);
        paused_frames += c.paused as u32;
    }
    let c = rig.r.stats().cluster;
    eprintln!(
        "[cf] long run: {frames} frames Hi-Z on, orbiting: max {} cluster tris, total clusters tested {}, rejected by Hi-Z {}, recovered by the late pass {}, queue overflows {}, paused frames {}, pool {}%",
        max_tris, tested, hiz_culled, late, overflow, paused_frames, c.pool_percent
    );
    assert_eq!(overflow, 0, "queue overflowed in the long run");
    assert_eq!(paused_frames, 0, "cluster path paused in the long run");
    assert!(max_tris > 0);
    rig.opts(false);
}

fn phase_resize(rig: &mut Rig) {
    rig.set_clusters(true);
    rig.opts(true);
    let (eye, target) = cams()[0];
    rig.eye = eye;
    rig.target = target;
    let sizes = [[640, 360], [1280, 720], [641, 359], [320, 200], [1920, 1080], [800, 800], [1280, 720]];
    for (i, sz) in sizes.into_iter().enumerate() {
        rig.ext = sz;
        let img = rig.shot(8);
        let s = rig.r.stats().cluster;
        assert_eq!((img.w, img.h), (sz[0], sz[1]));
        assert!(s.visible_clusters > 0 && !s.paused && s.queue_overflows == 0, "resize {i} {sz:?}: {s:?}");
        eprintln!("[cf] resize -> {}x{}: {} visible clusters, {} tris, Hi-Z culled {}, recovered {}", sz[0], sz[1], s.visible_clusters, s.triangles, s.culled_hiz, s.recovered);
    }
    // A resize every frame (a dragged panel border).
    for f in 0..40u32 {
        rig.ext = [600 + (f % 7) * 37, 330 + (f % 5) * 29];
        rig.step();
    }
    let s = rig.r.stats().cluster;
    assert!(s.queue_overflows == 0 && !s.paused, "{s:?}");
    rig.ext = [960, 540];
    rig.opts(false);
}

/// The game path (no viewport target): cluster draws inside the HDR scene pass.
fn phase_game(rig: &mut Rig) {
    rig.game = true;
    rig.shadows = true;
    for hiz in [false, true] {
        rig.opts(hiz);
        rig.set_clusters(true);
        let (mut max_inst, mut overflow, mut paused) = (0, 0, 0);
        for f in 0..150 {
            let a = f as f32 * 0.03;
            rig.eye = Vec3::new(a.sin() * 8.0, 2.5, a.cos() * 8.0);
            rig.target = Vec3::new(0.0, 1.0, 0.0);
            let s = rig.step();
            max_inst = max_inst.max(s.cluster.instances);
            overflow = overflow.max(s.cluster.queue_overflows);
            paused += s.cluster.paused as u32;
        }
        let s = rig.r.stats();
        eprintln!(
            "[cf] game path (HDR target + post, Hi-Z {hiz}): 150 frames, {} cluster instances, {} cluster tris, overflows {overflow}, paused frames {paused}, gpu_frame_ms {:.3}, draw calls {}",
            max_inst, s.cluster.triangles, s.gpu_frame_ms, s.draw_calls
        );
        assert!(max_inst > 0 && overflow == 0 && paused == 0);
    }
    rig.opts(false);
    rig.set_clusters(false);
    rig.game = false;
}

fn phase_toggle(rig: &mut Rig) {
    rig.shadows = false;
    rig.frozen = true;
    let (eye, target) = cams()[0];
    rig.eye = eye;
    rig.target = target;
    rig.ground_only = true;
    rig.set_clusters(false);
    let ground = rig.shot(4);
    rig.ground_only = false;
    rig.set_clusters(false);
    let off = rig.shot(4);
    let mut worst: f64 = 1.0;
    for round in 0..6 {
        rig.set_clusters(round % 2 == 0);
        let img = rig.shot(if round % 2 == 0 { 3 } else { 2 });
        let d = compare(&off, &img, &ground);
        let c = rig.r.stats().cluster;
        eprintln!("[cf] toggle round {round} ({}): IoU {:.5} mean|d| {:.4} instances {}", if round % 2 == 0 { "On" } else { "Off" }, d.iou, d.mean_abs, c.instances);
        worst = worst.min(d.iou);
        if round % 2 == 1 {
            assert_eq!(d.max, 0, "after toggling Off the image differs from the original Off image");
        }
    }
    assert!(worst > 0.98);
    rig.set_clusters(false);
    rig.frozen = false;
}

fn phase_budget(rig: &mut Rig) {
    rig.shadows = true;
    rig.opts(false);
    rig.set_clusters(true);
    let (eye, target) = (Vec3::new(0.0, 2.0, 7.0), Vec3::new(0.0, 1.0, 0.0));
    rig.eye = eye;
    rig.target = target;
    let mut s = rig.r.cluster_settings();
    s.target_frame_ms = 0.0;
    rig.r.set_cluster_settings(s);
    let base = measure(rig, 30, 90);
    eprintln!("[cf] budget: baseline (controller off) gpu_frame_ms {:.3}, cluster tris {}", base.frame, base.tris);
    for (label, target_ms) in [("target = 2x baseline (no pressure)", base.frame * 2.0), ("target = 0.5x baseline", base.frame * 0.5), ("target = 0.25x baseline", base.frame * 0.25)] {
        let mut s = rig.r.cluster_settings();
        s.target_frame_ms = target_ms;
        rig.r.set_cluster_settings(s);
        let mut scales = Vec::new();
        let mut frames = Vec::new();
        for _ in 0..360 {
            let st = rig.step();
            scales.push(st.cluster.error_scale);
            frames.push(st.gpu_frame_ms);
        }
        let tail = &scales[240..];
        let mean = tail.iter().sum::<f32>() / tail.len() as f32;
        let (lo, hi) = tail.iter().fold((f32::MAX, f32::MIN), |(l, h), &v| (l.min(v), h.max(v)));
        let reversals = tail.windows(3).filter(|w| (w[1] - w[0]) * (w[2] - w[1]) < 0.0).count();
        let fmean = frames[240..].iter().sum::<f32>() / 120.0;
        eprintln!(
            "[cf] budget {label}: target {:.3} ms -> error scale final {:.3} (last 120 frames: mean {:.3} min {:.3} max {:.3}, direction reversals {}), gpu_frame_ms {:.3} (tail mean), tris {}",
            target_ms, scales[scales.len() - 1], mean, lo, hi, reversals, fmean, rig.r.stats().cluster.triangles
        );
        if label.starts_with("target = 2x") {
            assert!(mean <= 1.05, "controller coarsened with no pressure: {mean}");
        }
    }
    let mut s = rig.r.cluster_settings();
    s.target_frame_ms = 0.0;
    rig.r.set_cluster_settings(s);
    rig.run(10);
    assert!((rig.r.stats().cluster.error_scale - 1.0).abs() < 1e-3, "turning the controller off must restore scale 1");
    rig.set_clusters(false);
}

/// A tiny skinned-vertex pool: 20 avatars overflow it, the path pauses (the
/// ordinary path draws, image identical to Off), and after the load shrinks it
/// resumes by itself once the cool-down passed.
fn run_overflow(window: Arc<Window>) {
    // Read when the cluster system is created (the first DAG install below).
    unsafe { std::env::set_var("ZESTENGINE_CLUSTER_SKIN_VERTICES", "40000") };
    let mut r = Renderer::new(window).expect("Renderer::new");
    r.set_vsync(false);
    r.set_gpu_zone_timing(true);
    r.set_cluster_settings(ClusterSettings { mode: ClusterMode::On, ..r.cluster_settings() });
    let mut w = World::new(&mut r, false, true);
    w.add_avatars(&mut r, 20);
    let mut rig = Rig { r, w, ext: [960, 540], t: 0.0, frames: 0, shadows: false, eye: Vec3::new(0.0, 3.0, 12.0), target: Vec3::new(0.0, 1.0, 0.0), ground_only: false, hiz: false, frozen: true, game: false, lights: Vec::new(), mode: 0, fov: 1.0 };
    rig.set_clusters(false);
    rig.ground_only = true;
    let ground = rig.shot(4);
    rig.ground_only = false;
    let off = rig.shot(4);
    rig.set_clusters(true);
    // The pool is far too small for 20 avatars. It used to pause the whole path;
    // now the farthest avatars fall back to the ordinary skinned draw one by one.
    let (mut paused_ever, mut first_overflow) = (false, None);
    for f in 0..40 {
        let s = rig.step();
        paused_ever |= s.cluster.paused;
        if s.cluster.skin_pool_overflows >= 1 && first_overflow.is_none() {
            first_overflow = Some(f);
        }
    }
    let before = rig.r.stats().cluster.skin_pool_overflows;
    rig.run(120);
    let c = rig.r.stats().cluster;
    eprintln!(
        "[cf] overflow: 20 avatars into a 40 000-vertex skin pool: first overflow at frame {first_overflow:?}, never paused {}, overflows {} -> {} over 120 frames, \
         {} instances on the ordinary draw ({} held back), {} skinned clusters, {} skin vertices",
        !paused_ever, before, c.skin_pool_overflows, c.skin_fallbacks, c.skin_denied, c.skinned_clusters, c.skin_vertices
    );
    assert!(!paused_ever && !c.paused, "a full skin pool must not pause the cluster path: {c:?}");
    assert!(first_overflow.is_some(), "the small pool never overflowed: {c:?}");
    assert!(c.skin_fallbacks > 0 && c.skin_denied > 0, "nothing fell back to the ordinary draw: {c:?}");
    assert!(c.skinned_clusters > 0, "the nearest avatars lost their clusters too: {c:?}");
    assert!(c.skin_pool_overflows <= before + 1, "the pool keeps overflowing after the fallback ({before} -> {})", c.skin_pool_overflows);
    // Mixed clustered / ordinary skinned draws still show the same picture.
    let mixed = rig.shot(2);
    let d = compare(&off, &mixed, &ground);
    eprintln!("[cf] overflow: image with part of the avatars clustered vs Off: max channel diff {}, IoU {:.5}, mean|d| {:.4}", d.max, d.iou, d.mean_abs);
    assert!(d.iou > 0.97, "the mixed draw differs from the ordinary image (IoU {})", d.iou);
    // Shrink the load: the held-back avatars come back on their own.
    rig.w.clear_avatars(&mut rig.r);
    rig.w.add_avatars(&mut rig.r, 5);
    rig.set_clusters(false);
    rig.ground_only = true;
    let ground5 = rig.shot(4);
    rig.ground_only = false;
    let off5 = rig.shot(4);
    rig.set_clusters(true);
    let mut back_at = None;
    for f in 0..400 {
        let s = rig.step();
        if back_at.is_none() && f > 5 && s.cluster.skin_denied == 0 && s.cluster.skin_fallbacks == 0 && s.cluster.skinned_clusters > 0 {
            back_at = Some(f);
        }
    }
    let img = rig.shot(4);
    let c = rig.r.stats().cluster;
    let d = compare(&off5, &img, &ground5);
    eprintln!(
        "[cf] overflow: with 5 avatars every avatar had its clusters back at frame {back_at:?}, skinned clusters {}, pool overflows total {}, IoU vs Off {:.5} mean|d| {:.4}",
        c.skinned_clusters, c.skin_pool_overflows, d.iou, d.mean_abs
    );
    assert!(back_at.is_some(), "the held-back avatars never returned: {c:?}");
    assert!(d.iou > 0.99 && !c.paused);
}

// ---------------------------------------------------------------------------
// Cluster shadows (GEO-5)

fn luma(px: &[u8], w: u32, x: i32, y: i32) -> f32 {
    let (x, y) = (x.clamp(0, w as i32 - 1) as usize, y.max(0) as usize);
    let i = (y * w as usize + x) * 4;
    0.2126 * px[i] as f32 + 0.7152 * px[i + 1] as f32 + 0.0722 * px[i + 2] as f32
}

fn project(cam: &CameraData, p: Vec3, w: u32, h: u32) -> (i32, i32) {
    let c = cam.proj * cam.view * p.extend(1.0);
    let n = c.truncate() / c.w;
    (((n.x * 0.5 + 0.5) * w as f32) as i32, ((0.5 - n.y * 0.5) * h as f32) as i32)
}

/// Thin legs + torso (dense meshes, so they qualify for DAGs) and avatars on a
/// ground plane under point / spot / sun lights at 5, 10 and 20 m: the
/// shadow of the 0.15 m legs and the torso must survive the coarser cluster
/// cut, and lit surfaces must not get acne.
fn run_shadow(window: Arc<Window>) {
    let mut r = Renderer::new(window).expect("Renderer::new");
    r.set_vsync(false);
    r.set_gpu_zone_timing(true);
    r.set_cluster_settings(ClusterSettings { mode: ClusterMode::On, ..r.cluster_settings() });
    let mut w = World::new(&mut r, false, true);
    let dense = |r: &mut Renderer, mesh: MeshData, color: [f32; 4]| {
        let dag = zestengine_assets::cluster_dag::build_cluster_dag(&mesh, &[], &DagOptions::default()).expect("DAG");
        let h = r.upload_mesh(&mesh).unwrap();
        r.install_cluster_dag(h, &dag.to_gpu_upload()).expect("install");
        let (c, rad) = aabb(&mesh);
        (h, material(r, color, 0.6, 0.0), c, rad, mesh.indices.len() / 3)
    };
    let (leg, leg_m, lc, lr, lt) = dense(&mut r, cylinder(0.075, 0.8, 32, 64), [0.8, 0.8, 0.85, 1.0]);
    let (torso, torso_m, tc, tr, tt) = dense(&mut r, cylinder(0.22, 0.6, 48, 48), [0.8, 0.5, 0.4, 1.0]);
    eprintln!("[cf] shadow scene: leg {lt} tris, torso {tt} tris");
    for x in [-0.12f32, 0.12] {
        w.extra.push((leg, leg_m, Mat4::from_translation(Vec3::new(x, 0.0, 0.0)), lc, lr));
    }
    w.extra.push((torso, torso_m, Mat4::from_translation(Vec3::new(0.0, 0.8, 0.0)), tc, tr));
    w.add_avatars(&mut r, 2);
    w.avatars[0].pos = Vec3::new(1.6, 0.0, 0.0);
    w.avatars[1].pos = Vec3::new(-1.6, 0.0, 0.0);
    let mut rig = Rig { r, w, ext: [960, 540], t: 0.0, frames: 0, shadows: false, eye: Vec3::ZERO, target: Vec3::ZERO, ground_only: false, hiz: false, frozen: true, game: false, lights: Vec::new(), mode: 0, fov: 1.0 };
    rig.opts(false);
    eprintln!("[cf] cluster shadows, shadow_px = {} texels", rig.r.cluster_settings().shadow_px);
    eprintln!("| light          | dist | leg shadow lit-ratio (ambient removed) Off / On | torso shadow Off / On | shadow-mask IoU | mean abs diff | object luma diff | shadow tris On |");
    let mut worst_iou = 1.0f64;
    for (kind, dist) in [("point", 5.0f32), ("point", 10.0), ("point", 20.0), ("spot", 5.0), ("spot", 10.0), ("spot", 20.0), ("sun", 6.0), ("sun", 12.0), ("sun", 22.0), ("sun", 40.0)] {
        let (eye, light) = if kind == "sun" {
            (
                Vec3::new(2.0, 1.6, dist),
                sun(true),
            )
        } else {
            let pos = Vec3::new(0.55 * dist, 0.7 * dist, 0.3 * dist);
            let mut l = LightInstance { kind: if kind == "spot" { LightKind::Spot } else { LightKind::Point }, position: pos, direction: (-pos).normalize(), color: [1.0, 0.97, 0.9], intensity: 30.0 * dist, range: dist * 3.0, spot_inner_deg: 40.0, spot_outer_deg: 70.0, ..sun(true) };
            l.cast_shadows = true;
            (Vec3::new(1.0, 2.5, 5.5), l)
        };
        rig.eye = eye;
        rig.target = Vec3::new(0.0, 0.5, 0.0);
        rig.lights = vec![light];
        // At 40 m a 1 rad lens shows the avatar a few pixels wide: narrow it so the 0.15 m leg stays ~15 px.
        rig.fov = if dist >= 40.0 { 0.25 } else { 1.0 };
        let proj = Mat4::perspective_rh(rig.fov, 960.0 / 540.0, 0.1, 300.0);
        let cam = CameraData { view: Mat4::look_at_rh(eye, rig.target, Vec3::Y), proj, position: eye };
        // Unshadowed reference, then shadowed Off and On.
        rig.shadows = false;
        rig.set_clusters(false);
        let lit = rig.shot(4);
        // Ambient-only reference (the light at intensity 0): the floor a fully shadowed pixel reads.
        rig.lights = vec![LightInstance { intensity: 0.0, ..light }];
        let floor = rig.shot(4);
        rig.lights = vec![light];
        rig.shadows = true;
        let off = rig.shot(4);
        rig.set_clusters(true);
        let on = rig.shot(6);
        let c = rig.r.stats().cluster;
        // Shadow mask: darker than 75 % of the unshadowed render.
        let mask = |img: &Image| -> Vec<bool> { (0..(img.w * img.h) as usize).map(|i| luma(&img.px, img.w, (i % img.w as usize) as i32, (i / img.w as usize) as i32) < 0.75 * luma(&lit.px, lit.w, (i % img.w as usize) as i32, (i / img.w as usize) as i32).max(8.0)).collect() };
        let (mo, mn) = (mask(&off), mask(&on));
        let inter = mo.iter().zip(&mn).filter(|(a, b)| **a && **b).count() as f64;
        let union = mo.iter().zip(&mn).filter(|(a, b)| **a || **b).count() as f64;
        let iou = if union == 0.0 { 1.0 } else { inter / union };
        // Analytic ground points in the legs' / torso's shadows.
        let ground_point = |p: Vec3| -> Vec3 {
            if kind == "sun" {
                let d = light.direction;
                p - d * (p.y / d.y)
            } else {
                let l = light.position;
                l + (p - l) * (l.y / (l.y - p.y))
            }
        };
        // Lit ratio with the ambient floor removed: 0 = fully shadowed, 1 = fully lit.
        let ratio = |img: &Image, pts: &[Vec3]| -> f32 {
            let (mut a, mut b) = (0.0, 0.0);
            for p in pts {
                let (x, y) = project(&cam, ground_point(*p), img.w, img.h);
                let f = luma(&floor.px, floor.w, x, y);
                a += luma(&img.px, img.w, x, y) - f;
                b += luma(&lit.px, lit.w, x, y) - f;
            }
            a / b.max(1.0)
        };
        let leg_pts: Vec<Vec3> = (1..8).map(|k| Vec3::new(-0.12, 0.1 * k as f32, 0.0)).collect();
        let torso_pts: Vec<Vec3> = (1..5).map(|k| Vec3::new(0.0, 0.85 + 0.12 * k as f32, 0.0)).collect();
        let (lo, ln, to, tn) = (ratio(&off, &leg_pts), ratio(&on, &leg_pts), ratio(&off, &torso_pts), ratio(&on, &torso_pts));
        // Object surfaces: the luma of pixels that differ between lit-ground-only and the object render.
        rig.ground_only = true;
        rig.shadows = false;
        rig.set_clusters(false);
        let ground = rig.shot(3);
        rig.ground_only = false;
        let obj_mask = lit.mask_vs(&ground, 6);
        let obj_luma = |img: &Image| -> f64 {
            let (mut sum, mut n) = (0.0f64, 0usize);
            for (i, m) in obj_mask.iter().enumerate() {
                if *m {
                    sum += luma(&img.px, img.w, (i % img.w as usize) as i32, (i / img.w as usize) as i32) as f64;
                    n += 1;
                }
            }
            sum / n.max(1) as f64
        };
        let d = compare(&off, &on, &ground);
        let (lo_l, ln_l) = (obj_luma(&off), obj_luma(&on));
        eprintln!(
            "| {kind:<14} | {dist:>4} | {lo:>6.3} / {ln:<6.3}              | {to:>6.3} / {tn:<6.3}      | {iou:>13.4} | {:>13.4} | {:>16.3} | {:>14} |",
            d.mean_abs,
            ln_l - lo_l,
            c.shadow_triangles
        );
        dump(&off, &format!("shadow_{kind}{dist}_off"));
        dump(&on, &format!("shadow_{kind}{dist}_on"));
        assert!(c.shadow_instances > 0, "{kind} {dist}: no cluster shadow casters were culled ({c:?})");
        // A shadow the original geometry casts must survive; no new darkening of lit surfaces.
        for (name, off_r, on_r) in [("leg", lo, ln), ("torso", to, tn)] {
            if off_r < 0.8 {
                assert!(on_r < 0.9 && on_r < off_r + 0.1, "{kind} {dist}: {name} shadow lost or weakened ({off_r:.3} -> {on_r:.3})");
            }
        }
        assert!((ln_l - lo_l).abs() < 6.0, "{kind} {dist}: lit object surfaces changed by {} luma (acne?)", ln_l - lo_l);
        worst_iou = worst_iou.min(iou);
    }
    eprintln!("[cf] cluster shadows: worst shadow-mask IoU {worst_iou:.4}");
    assert!(worst_iou > 0.9);
}

// ---------------------------------------------------------------------------
// Impostor (GEO-5)

/// The male avatar's bind pose as one static mesh, an octahedral impostor
/// captured from it, and both drawn from several angles by the real renderer:
/// silhouette IoU and mean colour difference, impostor against mesh, plus the
/// frame-to-frame change while the camera turns (blending should not pop).
fn run_impostor(window: Arc<Window>) {
    let mut r = Renderer::new(window).expect("Renderer::new");
    r.set_vsync(false);
    let model = load_model("male.glb");
    let mut merged = MeshData { vertices: Vec::new(), indices: Vec::new(), has_lightmap_uv: false };
    for m in &model.scene.meshes {
        let base = merged.vertices.len() as u32;
        merged.vertices.extend(m.vertices.iter().copied());
        merged.indices.extend(m.indices.iter().map(|i| i + base));
    }
    let colour = [0.8f32, 0.6, 0.5, 1.0];
    let t = std::time::Instant::now();
    let imp = zestengine_assets::impostor::Impostor::capture(&merged, colour, &Default::default()).expect("capture");
    eprintln!("[cf] impostor: {} triangles captured in {:.2} s (CPU rasteriser, {}x{} views of {}px, {} KB)", merged.indices.len() / 3, t.elapsed().as_secs_f32(), imp.grid, imp.grid, imp.tile, imp.byte_len() / 1024);
    let handle = r
        .upload_impostor(&zestengine_render::ImpostorUpload { grid: imp.grid, tile: imp.tile, centre: imp.centre, radius: imp.radius, albedo: &imp.albedo, aux: &imp.aux })
        .expect("upload");
    let mut w = World::new(&mut r, false, false);
    let mesh = r.upload_mesh(&merged).unwrap();
    let (c, rad) = aabb(&merged);
    let mat = material(&mut r, colour, 1.0, 0.0);
    w.extra.push((mesh, mat, Mat4::IDENTITY, c, rad));
    let model_xform = Mat4::from_translation(Vec3::ZERO);
    w.impostor = Some((handle, model_xform));
    let mut rig = Rig { r, w, ext: [640, 480], t: 0.0, frames: 0, shadows: false, eye: Vec3::ZERO, target: Vec3::ZERO, ground_only: false, hiz: false, frozen: true, game: false, lights: Vec::new(), mode: 1, fov: 1.0 };
    rig.opts(false);
    rig.target = Vec3::new(imp.centre[0], imp.centre[1], imp.centre[2]);
    let dist = 3.6f32;
    let shot_at = |rig: &mut Rig, yaw: f32, pitch: f32, mode: u8| -> Image {
        let tgt = rig.target;
        rig.eye = tgt + Vec3::new(yaw.sin() * pitch.cos(), pitch.sin(), yaw.cos() * pitch.cos()) * dist;
        rig.mode = mode;
        rig.shot(3)
    };
    eprintln!("| yaw | pitch | silhouette IoU | mean colour diff (0..1) | mesh px | impostor px | nearest view centre |");
    let (mut worst, mut sum, mut n) = (1.0f64, 0.0f64, 0);
    for pitch in [0.0f32, 0.5, -0.4] {
        for k in 0..8 {
            let yaw = k as f32 * std::f32::consts::FRAC_PI_4 + 0.3;
            let bg = shot_at(&mut rig, yaw, pitch, 1);
            let mesh_img = shot_at(&mut rig, yaw, pitch, 2);
            let imp_img = shot_at(&mut rig, yaw, pitch, 3);
            let (mm, mi) = (mesh_img.mask_vs(&bg, 8), imp_img.mask_vs(&bg, 8));
            let inter = mm.iter().zip(&mi).filter(|(a, b)| **a && **b).count();
            let union = mm.iter().zip(&mi).filter(|(a, b)| **a || **b).count();
            let iou = inter as f64 / union.max(1) as f64;
            let mut diff = 0.0f64;
            let mut cnt = 0usize;
            for (i, (a, b)) in mm.iter().zip(&mi).enumerate() {
                if *a && *b {
                    for c in 0..3 {
                        diff += (mesh_img.px[i * 4 + c] as f64 - imp_img.px[i * 4 + c] as f64).abs() / 255.0;
                    }
                    cnt += 3;
                }
            }
            let cdiff = diff / cnt.max(1) as f64;
            let dir = Vec3::new(yaw.sin() * pitch.cos(), pitch.sin(), yaw.cos() * pitch.cos());
            let lk = zestengine_assets::impostor::lookup(imp.grid, dir.to_array());
            let best = lk.iter().fold((0u32, -1.0f32), |b, &(c, w)| if w > b.1 { (c, w) } else { b });
            eprintln!(
                "| {:>5.0} | {:>5.2} | {:>14.4} | {:>22.4} | {:>7} | {:>11} | {} (w {:.2}) |",
                yaw.to_degrees().rem_euclid(360.0), pitch, iou, cdiff, mm.iter().filter(|x| **x).count(), mi.iter().filter(|x| **x).count(), best.0, best.1
            );
            dump(&mesh_img, &format!("imp_mesh_{k}_{pitch}"));
            dump(&imp_img, &format!("imp_imp_{k}_{pitch}"));
            worst = worst.min(iou);
            sum += iou;
            n += 1;
            assert!(mi.iter().filter(|x| **x).count() > 100, "the impostor drew nothing at yaw {yaw}");
        }
    }
    eprintln!("[cf] impostor vs mesh over {n} views: mean silhouette IoU {:.4}, worst {worst:.4}", sum / n as f64);
    assert!(worst > 0.5, "an impostor silhouette matches the mesh badly (IoU {worst})");
    // Turning the camera one degree at a time: the biggest frame-to-frame change.
    let mut prev: Option<Image> = None;
    let mut max_step = 0.0f64;
    for k in 0..90 {
        let img = shot_at(&mut rig, 0.3 + (k as f32).to_radians(), 0.2, 3);
        if let Some(p) = &prev {
            let d: f64 = p.px.iter().zip(&img.px).map(|(a, b)| a.abs_diff(*b) as f64).sum::<f64>() / (img.px.len() as f64 * 255.0);
            max_step = max_step.max(d);
        }
        prev = Some(img);
    }
    eprintln!("[cf] impostor turning 1 degree per frame over 90 degrees: largest frame-to-frame mean change {max_step:.5} (of 1.0)");
    assert!(max_step < 0.02, "the impostor pops between views ({max_step})");
}


// ---------------------------------------------------------------------------
// Textured impostor capture through the renderer (GEO-5 follow-up)

/// The scene's materials as renderer materials (textures uploaded as imported).
fn scene_materials(r: &mut Renderer, scene: &Scene) -> Vec<MaterialHandle> {
    let textures: Vec<_> = scene.textures.iter().map(|t| r.upload_texture(t).expect("texture")).collect();
    scene
        .materials
        .iter()
        .map(|m| {
            r.create_material(&MaterialDesc {
                name: m.name.clone(),
                params: m.params,
                features: m.features,
                albedo: m.albedo.map(|i| textures[i]),
                normal: m.normal.map(|i| textures[i]),
                orm: m.orm.map(|i| textures[i]),
                emission: m.emission.map(|i| textures[i]),
                ..Default::default()
            })
            .expect("material")
        })
        .collect()
}

/// Silhouette IoU and mean colour difference (of the pixels both cover) of the billboard against
/// the real mesh draws, over 24 views; returns (mean IoU, worst IoU, mean colour diff, worst colour diff).
fn compare_views(rig: &mut Rig, imp: &zestengine_assets::impostor::Impostor, dist: f32) -> (f64, f64, f64, f64) {
    let shot_at = |rig: &mut Rig, yaw: f32, pitch: f32, mode: u8| -> Image {
        let tgt = rig.target;
        rig.eye = tgt + Vec3::new(yaw.sin() * pitch.cos(), pitch.sin(), yaw.cos() * pitch.cos()) * dist;
        rig.mode = mode;
        rig.shot(3)
    };
    let (mut worst, mut sum, mut n, mut cd_sum, mut cd_worst) = (1.0f64, 0.0f64, 0, 0.0f64, 0.0f64);
    for pitch in [0.0f32, 0.5, -0.4] {
        for k in 0..8 {
            let yaw = k as f32 * std::f32::consts::FRAC_PI_4 + 0.3;
            let bg = shot_at(rig, yaw, pitch, 1);
            let mesh_img = shot_at(rig, yaw, pitch, 2);
            let imp_img = shot_at(rig, yaw, pitch, 3);
            let (mm, mi) = (mesh_img.mask_vs(&bg, 8), imp_img.mask_vs(&bg, 8));
            let inter = mm.iter().zip(&mi).filter(|(a, b)| **a && **b).count();
            let union = mm.iter().zip(&mi).filter(|(a, b)| **a || **b).count();
            let iou = inter as f64 / union.max(1) as f64;
            let (mut diff, mut cnt) = (0.0f64, 0usize);
            for (i, (a, b)) in mm.iter().zip(&mi).enumerate() {
                if *a && *b {
                    for c in 0..3 {
                        diff += (mesh_img.px[i * 4 + c] as f64 - imp_img.px[i * 4 + c] as f64).abs() / 255.0;
                    }
                    cnt += 3;
                }
            }
            let cdiff = diff / cnt.max(1) as f64;
            if k == 1 && pitch == 0.0 {
                dump(&mesh_img, &format!("timp_mesh_{}_{}", dist as u32, imp.kind as u32));
                dump(&imp_img, &format!("timp_imp_{}_{}", dist as u32, imp.kind as u32));
            }
            worst = worst.min(iou);
            sum += iou;
            cd_sum += cdiff;
            cd_worst = cd_worst.max(cdiff);
            n += 1;
        }
    }
    (sum / n as f64, worst, cd_sum / n as f64, cd_worst)
}

/// The male avatar with its real (textured) materials: the CPU capture's flat colours against the
/// renderer capture, both compared with the real mesh draws (silhouette IoU, colour), the
/// capture's determinism and that the main thread keeps running while it works.
fn run_impostor_textured(window: Arc<Window>) {
    textured_case(window.clone(), "male avatar", load_model("male.glb").scene);
    textured_case(window, "DamagedHelmet", load_gltf(root().join("models/DamagedHelmet.glb")).expect("DamagedHelmet"));
}

fn textured_case(window: Arc<Window>, label: &str, scene: Scene) {
    use zestengine_assets::impostor::{CaptureKind, Impostor, ImpostorSettings, bounds, key_textured};
    let mut r = Renderer::new(window).expect("Renderer::new");
    r.set_vsync(false);
    let model = Model { skinned: Vec::new(), scene };
    let mats = scene_materials(&mut r, &model.scene);
    eprintln!("[cf] textured impostor ({label}): {} textures, {} materials ({} with an albedo texture)", model.scene.textures.len(), model.scene.materials.len(), model.scene.materials.iter().filter(|m| m.albedo.is_some()).count());
    let w = World::new(&mut r, false, false);
    let mut rig = Rig { r, w, ext: [640, 480], t: 0.0, frames: 0, shadows: false, eye: Vec3::ZERO, target: Vec3::ZERO, ground_only: false, hiz: false, frozen: true, game: false, lights: Vec::new(), mode: 1, fov: 1.0 };
    rig.opts(false);
    // One merged mesh (CPU capture + bounds) and the per-part draws (renderer capture + real render).
    let mut merged = MeshData { vertices: Vec::new(), indices: Vec::new(), has_lightmap_uv: false };
    let mut parts = Vec::new();
    for inst in &model.scene.instances {
        for slot in &inst.slots {
            let m = &model.scene.meshes[slot.mesh];
            let base = merged.vertices.len() as u32;
            merged.vertices.extend(m.vertices.iter().copied());
            merged.indices.extend(m.indices.iter().map(|i| i + base));
            let h = rig.r.upload_mesh(m).unwrap();
            let (c, rad) = aabb(m);
            parts.push((h, mats[slot.material], c, rad));
        }
    }
    for &(h, m, c, rad) in &parts {
        rig.w.extra.push((h, m, Mat4::IDENTITY, c, rad));
    }
    let settings = ImpostorSettings::default();
    let (centre, radius) = bounds(&merged).expect("bounds");
    rig.target = Vec3::from(centre);
    // The previous (CPU, flat colour) capture: the baseline.
    let t = std::time::Instant::now();
    let cpu = Impostor::capture(&merged, [0.8, 0.6, 0.5, 1.0], &settings).expect("cpu capture");
    let cpu_s = t.elapsed().as_secs_f32();
    // Frame times while idle, then during the renderer capture.
    let mut idle_ms = Vec::new();
    for _ in 0..30 {
        let t = std::time::Instant::now();
        rig.step();
        idle_ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let request = |rig: &Rig| zestengine_render::ImpostorCaptureRequest {
        grid: settings.grid,
        tile: settings.tile,
        centre,
        radius,
        draws: rig
            .w
            .extra
            .iter()
            .map(|&(mesh, material, transform, c, rad)| DrawCmd {
                gi_source: None,
                mesh,
                material,
                transform,
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
            })
            .collect(),
    };
    let capture = |rig: &mut Rig| -> (zestengine_render::CapturedImpostor, usize, f64, f64) {
        let t0 = std::time::Instant::now();
        let ticket = rig.r.start_impostor_capture(request(rig)).expect("queue capture");
        let (mut frames, mut worst_ms) = (0usize, 0.0f64);
        loop {
            let t = std::time::Instant::now();
            rig.step();
            worst_ms = worst_ms.max(t.elapsed().as_secs_f64() * 1e3);
            frames += 1;
            if let Some(res) = rig.r.take_impostor_capture(ticket) {
                return (res.expect("capture"), frames, t0.elapsed().as_secs_f64() * 1e3, worst_ms);
            }
            assert!(frames < 400, "the capture never finished");
        }
    };
    let (first, frames, total_ms, worst_ms) = capture(&mut rig);
    let (second, frames2, total2, worst2) = capture(&mut rig);
    assert_eq!(first, second, "the renderer capture is not deterministic");
    eprintln!("[cf] textured impostor ({label}): second capture (pipelines warm): {frames2} frames / {total2:.1} ms wall, worst frame {worst2:.1} ms");
    idle_ms.sort_by(|a, b| a.total_cmp(b));
    eprintln!(
        "[cf] textured impostor ({label}): renderer capture of {} draws, {} tris: {} frames / {:.1} ms wall end to end; worst frame during it {:.1} ms (idle median {:.1} ms); CPU capture took {:.2} s",
        parts.len(), merged.indices.len() / 3, frames, total_ms, worst_ms, idle_ms[idle_ms.len() / 2], cpu_s
    );
    // Atlas layout.
    let side = (settings.grid * settings.tile) as usize;
    assert_eq!(first.albedo.len(), side * side * 4);
    let covered = first.albedo.iter().skip(3).step_by(4).filter(|&&a| a == 255).count();
    let cpu_covered = cpu.albedo.iter().skip(3).step_by(4).filter(|&&a| a == 255).count();
    eprintln!("[cf] textured impostor ({label}): covered texels {covered} (CPU capture {cpu_covered})");
    assert!(covered > cpu_covered / 2, "the renderer capture covers far less than the CPU one");
    let imp = Impostor::from_render(key_textured(&merged, &settings, 0), &settings, centre, radius, first.albedo, first.aux).expect("from_render");
    assert_eq!(imp.kind, CaptureKind::Textured);
    let up = |r: &mut Renderer, i: &Impostor| r.upload_impostor(&zestengine_render::ImpostorUpload { grid: i.grid, tile: i.tile, centre: i.centre, radius: i.radius, albedo: &i.albedo, aux: &i.aux }).expect("upload");
    let dist = radius * 3.5;
    eprintln!("[cf] impostor vs the real {label}, 24 views (previous CPU capture first):");
    let h_cpu = up(&mut rig.r, &cpu);
    rig.w.impostor = Some((h_cpu, Mat4::IDENTITY));
    let (c_mean, c_worst, c_cd, c_cdw) = compare_views(&mut rig, &cpu, dist);
    eprintln!("[cf]   CPU rasteriser (flat colour):  IoU mean {c_mean:.3} worst {c_worst:.3} | colour diff mean {c_cd:.4} worst {c_cdw:.4}");
    let h_gpu = up(&mut rig.r, &imp);
    rig.w.impostor = Some((h_gpu, Mat4::IDENTITY));
    let (g_mean, g_worst, g_cd, g_cdw) = compare_views(&mut rig, &imp, dist);
    eprintln!("[cf]   renderer capture (textured):   IoU mean {g_mean:.3} worst {g_worst:.3} | colour diff mean {g_cd:.4} worst {g_cdw:.4}");
    assert!(g_worst > 0.5, "the textured impostor silhouette matches the mesh badly ({g_worst})");
    // At the distance where the engine swaps to the impostor (`lod::IMPOSTOR_COVERAGE` of the screen
    // height), on a taller target so the silhouette is more than a few pixels.
    let switch = zestengine_core::lod::impostor_distance(radius, (rig.fov / 2.0).tan());
    rig.ext = [1280, 1024];
    let (s_mean, s_worst, s_cd, _) = compare_views(&mut rig, &imp, switch);
    eprintln!("[cf]   at the switch distance ({switch:.1} m = {:.1}% screen height, {:.0} px): IoU mean {s_mean:.3} worst {s_worst:.3} | colour diff mean {s_cd:.4}", zestengine_core::lod::IMPOSTOR_COVERAGE * 100.0, zestengine_core::lod::IMPOSTOR_COVERAGE * 1024.0);
    assert!(s_worst > 0.5, "the impostor silhouette differs from the mesh at the switch distance (worst IoU {s_worst})");
    assert!(g_cd < c_cd, "the textured capture should be nearer the real colours ({g_cd} vs CPU {c_cd})");
}

// ---------------------------------------------------------------------------
// Deferred frees (PERF-2)

/// `Renderer::free_resources` across many frames: meshes, textured materials and
/// textures are created, drawn and freed in rounds while a reference object keeps
/// rendering. No device idle on those paths, nothing left parked, device memory
/// returns to the baseline, the reference image is unchanged, and every round's
/// textured quad shows its own colour (a freed slot never bleeds into a new one).
fn run_free_churn(window: Arc<Window>) {
    use std::collections::HashSet;
    let mut r = Renderer::new(window).expect("Renderer::new");
    r.set_vsync(false);
    let w = World::new(&mut r, false, false);
    let mut rig = Rig { r, w, ext: [320, 240], t: 0.0, frames: 0, shadows: false, eye: Vec3::new(0.0, 4.0, 1.5), target: Vec3::ZERO, ground_only: false, hiz: false, frozen: true, game: false, lights: Vec::new(), mode: 2, fov: 1.0 };
    rig.opts(false);
    // The persistent reference: a small torus off to the side.
    let tor = torus(24, 12);
    let (tc, trad) = aabb(&tor);
    let tm = rig.r.upload_mesh(&tor).unwrap();
    let tmat = material(&mut rig.r, [0.2, 0.7, 0.3, 1.0], 0.6, 0.0);
    rig.w.extra.push((tm, tmat, Mat4::from_translation(Vec3::new(-2.5, 0.4, -0.5)) * Mat4::from_scale(Vec3::splat(0.5)), tc, trad * 0.5));
    let ballast = torus(128, 96);
    let quad = quad_ground(0.8);
    let (qc, qrad) = aabb(&quad);
    let solid = |rgb: [u8; 3]| zestengine_render::TextureData { width: 8, height: 8, pixels: (0..64).flat_map(|_| [rgb[0], rgb[1], rgb[2], 255]).collect(), srgb: true, hdr: false };
    let centre_px = |img: &Image| -> [u8; 3] {
        let i = ((img.h / 2) * img.w + img.w / 2) as usize * 4;
        [img.px[i], img.px[i + 1], img.px[i + 2]]
    };
    rig.run(4);
    let base_img = rig.shot(2);
    let base_bytes = rig.r.gpu_allocated_bytes();
    let per_round = 6usize;
    let rounds = 40usize;
    let (mut peak, mut stalls, mut ok_colours) = (base_bytes, 0u64, 0usize);
    let extra_len = rig.w.extra.len();
    for round in 0..rounds {
        let (mut ms, mut ts, mut mats) = (Vec::new(), Vec::new(), Vec::new());
        let red = round % 2 == 0;
        for k in 0..per_round {
            let m = rig.r.upload_mesh(&quad).unwrap();
            let t = rig.r.upload_texture(&solid(if red { [230, 20, 20] } else { [20, 20, 230] })).unwrap();
            let mat = rig.r.create_material(&zestengine_render::MaterialDesc { name: format!("churn{round}_{k}"), albedo: Some(t), ..Default::default() }).unwrap();
            ms.push(m);
            ts.push(t);
            mats.push(mat);
        }
        // Ballast that is never drawn but makes the device memory visible: a 12k-vertex mesh and a 256x256 mipped texture each.
        for _ in 0..per_round {
            ms.push(rig.r.upload_mesh(&ballast).unwrap());
            ts.push(rig.r.upload_texture(&zestengine_render::TextureData { width: 256, height: 256, pixels: vec![90; 256 * 256 * 4], srgb: true, hdr: false }).unwrap());
        }
        // Only the first quad sits at the origin (in view); the others hide under it.
        for k in 0..per_round {
            rig.w.extra.push((ms[k], mats[k], Mat4::from_translation(Vec3::new(0.0, 0.01 * k as f32, 0.0)), qc, qrad));
        }
        // The topmost quad (highest y) is the last one: its colour must be this round's.
        rig.run(2);
        if round % 8 == 0 {
            let img = rig.shot(1);
            let c = centre_px(&img);
            let good = if red { c[0] > 100 && c[2] < c[0] / 2 } else { c[2] > 100 && c[0] < c[2] / 2 };
            assert!(good, "round {round}: centre pixel {c:?} is not this round's {} quad", if red { "red" } else { "blue" });
            ok_colours += 1;
        }
        rig.w.extra.truncate(extra_len);
        let before = zestengine_render::hitch::idle_wait_count();
        let set = |v: &Vec<usize>| v.iter().copied().collect::<HashSet<usize>>();
        rig.r
            .free_resources(&set(&ms.iter().map(|m| m.index()).collect()), &set(&ts.iter().map(|t| t.index()).collect()), &set(&mats.iter().map(|m| m.index()).collect()))
            .unwrap();
        rig.run(2);
        stalls += zestengine_render::hitch::idle_wait_count() - before;
        peak = peak.max(rig.r.gpu_allocated_bytes());
    }
    rig.run(6);
    let end_bytes = rig.r.gpu_allocated_bytes();
    let parked = rig.r.deferred_free_count();
    let end_img = rig.shot(2);
    eprintln!(
        "[cf] free churn: {rounds} rounds x ({per_round} meshes + textures + materials), device idles during frees/frames {stalls}, allocated base {} KB, peak {} KB, end {} KB, parked {parked}, colour checks {ok_colours}",
        base_bytes / 1024, peak / 1024, end_bytes / 1024
    );
    assert_eq!(stalls, 0, "free_resources / the frames after it waited for the device");
    assert_eq!(parked, 0, "deferred frees left parked after the frames retired");
    // The ordinary allocator may keep a block or two around; growth proportional to the rounds would be a leak.
    assert!(end_bytes <= base_bytes + (2 << 20), "device memory grew from {base_bytes} to {end_bytes} bytes");
    assert!(peak > base_bytes + (1 << 20), "the ballast never showed up in the allocator ({peak} vs {base_bytes})");
    assert!(peak < base_bytes + 16 * (1 << 20), "peak {peak} suggests frees are not retiring");
    let d: usize = base_img.px.iter().zip(&end_img.px).map(|(a, b)| a.abs_diff(*b) as usize).max().unwrap_or(0);
    assert!(d <= 2, "the reference object changed after the churn (max channel diff {d})");
    // A synchronous flush leaves nothing behind either.
    rig.r.flush_deferred_frees().unwrap();
    assert_eq!(rig.r.deferred_free_count(), 0);
}

// ---------------------------------------------------------------------------
// Live sampler changes (TEX-2)

/// A texture that is half red, half blue on a quad whose UVs run across zero:
/// Repeat shows blue left of the middle, Clamp shows red. Switching the sampler
/// with `update_texture_sampler` changes the picture within a frame, with no
/// device idle, and switching back restores it.
fn run_live_sampler(window: Arc<Window>) {
    use zestengine_render::{SamplerKey, SamplerWrap};
    let mut r = Renderer::new(window).expect("Renderer::new");
    r.set_vsync(false);
    let w = World::new(&mut r, false, false);
    let mut rig = Rig { r, w, ext: [320, 240], t: 0.0, frames: 0, shadows: false, eye: Vec3::new(0.0, 4.0, 1.5), target: Vec3::ZERO, ground_only: false, hiz: false, frozen: true, game: false, lights: Vec::new(), mode: 2, fov: 1.0 };
    rig.opts(false);
    let quad = quad_ground(0.8);
    let (qc, qrad) = aabb(&quad);
    let mesh = rig.r.upload_mesh(&quad).unwrap();
    let mut pixels = Vec::new();
    for _y in 0..8 {
        for x in 0..8 {
            pixels.extend_from_slice(if x < 4 { &[230u8, 20, 20, 255] } else { &[20u8, 20, 230, 255] });
        }
    }
    let tex = rig.r.upload_texture(&zestengine_render::TextureData { width: 8, height: 8, pixels, srgb: true, hdr: false }).unwrap();
    let mat = rig.r.create_material(&zestengine_render::MaterialDesc { name: "sampler".into(), albedo: Some(tex), ..Default::default() }).unwrap();
    rig.w.extra.push((mesh, mat, Mat4::IDENTITY, qc, qrad));
    let left = |img: &Image| -> [u8; 3] {
        let i = ((img.h / 2) * img.w + img.w / 2 - img.w / 8) as usize * 4;
        [img.px[i], img.px[i + 1], img.px[i + 2]]
    };
    let is_red = |c: [u8; 3]| c[0] > 100 && c[2] < c[0] / 2;
    let is_blue = |c: [u8; 3]| c[2] > 100 && c[0] < c[2] / 2;
    let repeat = rig.shot(3);
    assert!(is_blue(left(&repeat)), "repeat should wrap to the blue half: {:?}", left(&repeat));
    let idles = zestengine_render::hitch::idle_wait_count();
    let clamp = SamplerKey { wrap_u: SamplerWrap::Clamp, wrap_v: SamplerWrap::Clamp, ..SamplerKey::default() };
    let n = rig.r.update_texture_sampler(tex, clamp);
    assert_eq!(n, 1, "one material samples the texture");
    assert_eq!(rig.r.update_texture_sampler(tex, clamp), 0, "the same key again rebinds nothing");
    rig.run(1);
    let stalls_a = zestengine_render::hitch::idle_wait_count() - idles;
    let clamped = rig.shot(2);
    assert!(is_red(left(&clamped)), "clamp should show the red edge: {:?}", left(&clamped));
    let idles = zestengine_render::hitch::idle_wait_count();
    assert_eq!(rig.r.update_texture_sampler(tex, SamplerKey::default()), 1);
    rig.run(1);
    let stalls_b = zestengine_render::hitch::idle_wait_count() - idles;
    let back = rig.shot(2);
    assert!(is_blue(left(&back)), "back to repeat: {:?}", left(&back));
    eprintln!("[cf] live sampler: repeat {:?} -> clamp {:?} -> repeat {:?}, device idles during the rebinds {}", left(&repeat), left(&clamped), left(&back), stalls_a + stalls_b);
    assert_eq!(stalls_a + stalls_b, 0, "the live rebind idled the device");
    // A mip bias is its own sampler: it rebinds too.
    assert_eq!(rig.r.update_texture_sampler(tex, SamplerKey { mip_bias_eighths: 8, ..SamplerKey::default() }), 1);
    rig.run(2);
    rig.r.flush_deferred_frees().unwrap();
}

// ---------------------------------------------------------------------------
// Impostor atlas release (GEO-5)

/// Atlases are freed (deferred, no idle) and a scene reset frees whatever is left;
/// device memory returns to the baseline. More upload/free cycles than the
/// 64-atlas cap prove the atlas descriptor sets are released too.
fn run_impostor_release(window: Arc<Window>) {
    let mut r = Renderer::new(window).expect("Renderer::new");
    r.set_vsync(false);
    let w = World::new(&mut r, false, false);
    let mut rig = Rig { r, w, ext: [160, 120], t: 0.0, frames: 0, shadows: false, eye: Vec3::new(0.0, 2.0, 4.0), target: Vec3::ZERO, ground_only: false, hiz: false, frozen: true, game: false, lights: Vec::new(), mode: 1, fov: 1.0 };
    rig.opts(false);
    let (grid, tile) = (8u32, 128u32);
    let texels = vec![180u8; (grid * tile * grid * tile * 4) as usize];
    let up = |r: &mut Renderer| r.upload_impostor(&zestengine_render::ImpostorUpload { grid, tile, centre: [0.0; 3], radius: 1.0, albedo: &texels, aux: &texels });
    rig.run(4);
    let base = rig.r.gpu_allocated_bytes();
    let stalls0 = zestengine_render::hitch::idle_wait_count();
    let mut peak = base;
    // 150 cycles of 4 atlases: far past the cap if any set or image leaked.
    for _ in 0..150 {
        let hs: Vec<_> = (0..4).map(|_| up(&mut rig.r).expect("upload")).collect();
        for h in &hs {
            rig.r.draw_impostor(*h, Mat4::IDENTITY, [0.0; 4]);
        }
        rig.run(1);
        for h in hs {
            rig.r.free_impostor(h);
        }
        rig.run(1);
        peak = peak.max(rig.r.gpu_allocated_bytes());
    }
    rig.run(6);
    let stalls = zestengine_render::hitch::idle_wait_count() - stalls0;
    let after_free = rig.r.gpu_allocated_bytes();
    assert_eq!(rig.r.impostor_count(), 0);
    assert_eq!(rig.r.deferred_free_count(), 0);
    // A scene reset releases atlases nobody freed.
    let keep: Vec<_> = (0..5).map(|_| up(&mut rig.r).expect("upload")).collect();
    assert_eq!(rig.r.impostor_count(), keep.len());
    let with_atlases = rig.r.gpu_allocated_bytes();
    rig.r.reset_scene().unwrap();
    assert_eq!(rig.r.impostor_count(), 0);
    let after_reset = rig.r.gpu_allocated_bytes();
    eprintln!(
        "[cf] impostor release: 600 atlases uploaded/freed, idles {stalls}, allocated base {} KB, peak {} KB, after frees {} KB; 5 live {} KB, after scene reset {} KB",
        base / 1024, peak / 1024, after_free / 1024, with_atlases / 1024, after_reset / 1024
    );
    assert_eq!(stalls, 0, "free_impostor waited for the device");
    assert!(after_free <= base + (1 << 20), "atlases leaked: {base} -> {after_free}");
    assert!(with_atlases > after_free + 5 * (grid * tile * grid * tile * 4) as u64 / 2, "5 atlases should occupy memory");
    assert!(after_reset <= with_atlases - 5 * (grid * tile * grid * tile * 4) as u64 / 2, "the scene reset kept the atlases ({with_atlases} -> {after_reset})");
}

// ---------------------------------------------------------------------------
// Shared descriptor-set swaps without a device idle (PERF-2)

/// Baked probes, the lightmap array, the reflection slots, the skybox and the
/// shadow atlas are swapped while frames are in flight: no `device_wait_idle`,
/// and the very next frame (either frame slot) already shows the new data.
fn run_swap_churn(window: Arc<Window>) {
    for loaded in [false, true] {
        swap_churn(window.clone(), loaded);
    }
}

fn swap_churn(window: Arc<Window>, loaded: bool) {
    use std::time::Instant;
    use zestengine_render::{HdrTextureArray, HdrTextureFormat, ProbeSh};
    let mut r = Renderer::new(window).expect("Renderer::new");
    r.set_vsync(false);
    let mut w = World::new(&mut r, false, false);
    if loaded {
        // Frames with real GPU work in flight: 64 plain 1M-triangle tori (no DAG), so an idle wait has something to wait for.
        let mesh = torus(1000, 500);
        let (c, rad) = aabb(&mesh);
        let h = r.upload_mesh(&mesh).unwrap();
        let m = material(&mut r, [0.8, 0.45, 0.2, 1.0], 0.4, 0.2);
        w.torus = Some((h, m, c, rad));
        w.torus_grid = (0..64).map(|i| Vec3::new((i % 8) as f32 * 3.0 - 10.0, 1.0, (i / 8) as f32 * 3.0 - 10.0)).collect();
    }
    let mut rig = Rig { r, w, ext: [if loaded { 1280 } else { 320 }, if loaded { 720 } else { 240 }], t: 0.0, frames: 0, shadows: false, eye: Vec3::new(0.0, 4.0, 1.5), target: Vec3::ZERO, ground_only: false, hiz: false, frozen: true, game: false, lights: Vec::new(), mode: 0, fov: 1.0 };
    rig.opts(false);
    rig.run(6);
    if loaded {
        eprintln!("[cf] loaded run: median frame {:.1} ms (CPU wall)", { let t = Instant::now(); rig.run(10); t.elapsed().as_secs_f64() * 100.0 });
    }
    let centre = |img: &Image| -> [i32; 3] {
        let i = ((img.h / 2) * img.w + img.w / 2) as usize * 4;
        [img.px[i] as i32, img.px[i + 1] as i32, img.px[i + 2] as i32]
    };
    let volume = [(Mat4::IDENTITY, [400.0f32; 3], [2u32; 3], 0u32)];
    let probes = |rgb: [f32; 3]| -> Vec<ProbeSh> { (0..8).map(|_| ProbeSh { coeffs: [rgb, [0.0; 3], [0.0; 3], [0.0; 3]], ..Default::default() }).collect() };
    let lightmaps = |salt: u8| HdrTextureArray { format: HdrTextureFormat::Rgba32f, size: 256, layers: (0..8).map(|l| vec![salt.wrapping_add(l as u8); 256 * 256 * 16]).collect() };
    let mut stalls = 0u64;
    let mut times: Vec<(&str, f64)> = Vec::new();
    let mut time = |rig: &mut Rig, name: &'static str, f: &dyn Fn(&mut Renderer)| {
        let idle = zestengine_render::hitch::idle_wait_count();
        let t = Instant::now();
        f(&mut rig.r);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        stalls += zestengine_render::hitch::idle_wait_count() - idle;
        times.push((name, ms));
    };
    let base = centre(&rig.shot(1));
    // Probes: red then blue then red again; every frame after the call must already show it.
    for (k, rgb) in [[6.0, 0.0, 0.0], [0.0, 0.0, 6.0], [6.0, 0.0, 0.0], [0.0, 0.0, 6.0]].into_iter().enumerate() {
        let (p, v) = (probes(rgb), volume.to_vec());
        time(&mut rig, "set_baked_probes", &|r| r.set_baked_probes(&p, &v));
        for f in 0..4 {
            let c = centre(&rig.shot(1));
            let good = if rgb[0] > 0.0 { c[0] > c[2] + 20 } else { c[2] > c[0] + 20 };
            assert!(good, "probes swap {k}, frame {f} after it: centre pixel {c:?} (base {base:?}) does not show the new probes");
        }
    }
    for salt in [3u8, 9, 27] {
        let a = lightmaps(salt);
        time(&mut rig, "set_lightmap_array", &|r| r.set_lightmap_array(&a));
        rig.run(3);
    }
    for _ in 0..3 {
        time(&mut rig, "swap_reflection_slots", &|r| r.swap_reflection_slots());
        rig.run(3);
    }
    let sky = zestengine_render::TextureData { width: 64, height: 32, pixels: (0..64 * 32).flat_map(|i| [(i % 251) as u8, 90, 200, 255]).collect(), srgb: true, hdr: false };
    for _ in 0..3 {
        time(&mut rig, "set_skybox", &|r| r.set_skybox(Some(&sky)).unwrap());
        rig.run(3);
    }
    time(&mut rig, "set_skybox(None)", &|r| r.set_skybox(None).unwrap());
    rig.run(3);
    for dim in [1024u32, 2048, 1024, 4096, 2048] {
        time(&mut rig, "set_shadow_resolution", &|r| r.set_shadow_resolution(dim).unwrap());
        rig.run(3);
    }
    rig.run(8);
    let parked = rig.r.deferred_free_count();
    let mut by: std::collections::BTreeMap<&str, Vec<f64>> = Default::default();
    for (n, ms) in &times {
        by.entry(n).or_default().push(*ms);
    }
    eprintln!("[cf] swaps ({}) without a device idle: {stalls} idles in {} calls; frees still parked after the run: {parked}", if loaded { "GPU loaded" } else { "GPU idle" }, times.len());
    eprintln!("| call | n | mean ms | max ms |");
    eprintln!("|---|---|---|---|");
    for (n, v) in &by {
        eprintln!("| {n} | {} | {:.3} | {:.3} |", v.len(), v.iter().sum::<f64>() / v.len() as f64, v.iter().cloned().fold(0.0, f64::max));
    }
    if std::env::var_os("ZESTENGINE_CF_EXPECT_IDLES").is_none() {
        assert_eq!(stalls, 0, "a shared-binding swap still idled the device");
    }
    assert_eq!(parked, 0, "swapped-out resources were never freed");
}

// ---------------------------------------------------------------------------
// DAG install without a main-thread submit (GEO-4)

/// The heavy meshes to install: a 1M-triangle torus, and the biggest meshes of
/// `ZESTENGINE_CF_MODEL` (an FBX / glTF; a copy is imported so no cache lands
/// beside the source) when set.
fn dag_cases() -> Vec<(String, MeshData, zestengine_render::ClusterMeshUpload)> {
    let mut out = Vec::new();
    let torus = torus(1000, 500);
    let dag = zestengine_assets::cluster_dag::build_cluster_dag(&torus, &[], &DagOptions::default()).expect("torus DAG");
    out.push(("torus 2M tris".to_string(), torus, dag.to_gpu_upload()));
    if let Some(path) = std::env::var_os("ZESTENGINE_CF_MODEL") {
        let src = std::path::PathBuf::from(&path);
        let dir = std::env::temp_dir().join(format!("cf_model_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let copy = dir.join(src.file_name().unwrap());
        std::fs::copy(&src, &copy).unwrap();
        let t = std::time::Instant::now();
        match zestengine_assets::load_model_with_geometry(&copy) {
            Ok((scene, geo)) => {
                eprintln!("[cf] {}: imported + DAG in {:.1} s ({} meshes)", src.display(), t.elapsed().as_secs_f32(), scene.meshes.len());
                let mut ids: Vec<usize> = (0..scene.meshes.len()).filter(|&i| geo.meshes[i].as_ref().is_some_and(|d| { let u = d.to_gpu_upload(); !u.skinned })).collect();
                ids.sort_by_key(|&i| std::cmp::Reverse(scene.meshes[i].indices.len()));
                for &i in ids.iter().take(6) {
                    let up = geo.meshes[i].as_ref().unwrap().to_gpu_upload();
                    out.push((format!("{} mesh {i} {}k tris", src.file_name().unwrap().to_string_lossy(), scene.meshes[i].indices.len() / 3000), MeshData { vertices: scene.meshes[i].vertices.clone(), indices: scene.meshes[i].indices.clone(), has_lightmap_uv: scene.meshes[i].has_lightmap_uv }, up));
                }
            }
            Err(e) => eprintln!("[cf] {}: {e:#}: model skipped", src.display()),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    out
}

/// Install the same DAGs three ways and compare the main-thread time, the worst
/// frame after, and the resulting image: the old synchronous one-time submit
/// (`install_cluster_dag`), staging on the main thread with the copy recorded by
/// the next frame (`install_cluster_dag_deferred`), and staging on a loader
/// thread with an `Uploader` (`stage_cluster_dag` + `install_staged_cluster_dag`).
fn run_dag_install(window: Arc<Window>) {
    use std::time::Instant;
    let cases = dag_cases();
    let total_tris: usize = cases.iter().map(|c| c.1.indices.len() / 3).sum();
    let mut images: Vec<(&str, Image)> = Vec::new();
    let mut rows = Vec::new();
    for mode in ["sync", "deferred", "staged"] {
        let mut r = Renderer::new(window.clone()).expect("Renderer::new");
        r.set_vsync(false);
        r.set_cluster_settings(ClusterSettings { mode: ClusterMode::On, ..r.cluster_settings() });
        let mut w = World::new(&mut r, false, false);
        // Upload the meshes first (not part of the measurement), then draw them in a row.
        let handles: Vec<MeshHandle> = cases.iter().map(|c| r.upload_mesh(&c.1).unwrap()).collect();
        for (k, (_, mesh, _)) in cases.iter().enumerate() {
            let (c, rad) = aabb(mesh);
            let m = material(&mut r, [0.8, 0.45, 0.2, 1.0], 0.4, 0.2);
            let at = Mat4::from_translation(Vec3::new(k as f32 * 4.0 - 2.0, 1.0, 0.0));
            w.extra.push((handles[k], m, at, c, rad));
        }
        let mut rig = Rig { r, w, ext: [640, 360], t: 0.0, frames: 0, shadows: false, eye: Vec3::new(2.0, 4.0, 14.0), target: Vec3::new(2.0, 0.5, 0.0), ground_only: false, hiz: false, frozen: true, game: false, lights: Vec::new(), mode: 0, fov: 1.0 };
        rig.opts(false);
        rig.run(4);
        let idle = zestengine_render::hitch::idle_wait_count();
        let t = Instant::now();
        match mode {
            "sync" => cases.iter().enumerate().for_each(|(k, c)| rig.r.install_cluster_dag(handles[k], &c.2).expect("install")),
            "deferred" => cases.iter().enumerate().for_each(|(k, c)| rig.r.install_cluster_dag_deferred(handles[k], &c.2).expect("install")),
            _ => {
                // Staged on a loader thread (its time is off the main thread), installed here.
                let up = rig.r.uploader().expect("a transfer queue");
                let staged = std::thread::scope(|s| {
                    s.spawn(|| cases.iter().map(|c| up.stage_cluster_dag(&c.2).expect("stage")).collect::<Vec<_>>()).join().unwrap()
                });
                let stage_ms = t.elapsed().as_secs_f64() * 1000.0;
                let t2 = Instant::now();
                for (k, st) in staged.into_iter().enumerate() {
                    rig.r.install_staged_cluster_dag(handles[k], st).expect("install staged");
                }
                eprintln!("[cf] dag {mode}: staging on the loader thread {stage_ms:.1} ms (off the main thread), install_staged on the main thread {:.2} ms", t2.elapsed().as_secs_f64() * 1000.0);
            }
        }
        let main_ms = if mode == "staged" { 0.0 } else { t.elapsed().as_secs_f64() * 1000.0 };
        let queued = rig.r.cluster_uploads_queued();
        if mode != "sync" {
            assert_eq!(queued, cases.len(), "{mode}: the copies should wait for the next frame");
            assert!(!handles.iter().any(|h| rig.r.has_cluster_dag(*h)), "{mode}: drawn from clusters before its copy was recorded");
        }
        // Frames after the install: the worst one, and when the meshes switch to clusters.
        let mut worst = 0f64;
        for _ in 0..6 {
            let t = Instant::now();
            rig.run(1);
            worst = worst.max(t.elapsed().as_secs_f64() * 1000.0);
        }
        assert_eq!(rig.r.cluster_uploads_queued(), 0, "{mode}: copies left unrecorded");
        assert!(handles.iter().all(|h| rig.r.has_cluster_dag(*h)), "{mode}: not drawn from clusters after the frames");
        let idles = zestengine_render::hitch::idle_wait_count() - idle;
        rows.push((mode, main_ms, worst, idles));
        images.push((mode, rig.shot(3)));
    }
    eprintln!("[cf] DAG install of {} meshes ({} k triangles):", cases.len(), total_tris / 1000);
    eprintln!("| way | main-thread ms | worst frame after (ms) | device idles |");
    eprintln!("|---|---|---|---|");
    for (mode, ms, worst, idles) in &rows {
        eprintln!("| {mode} | {} | {worst:.1} | {idles} |", if *mode == "staged" { "see above (install_staged only)".to_string() } else { format!("{ms:.1}") });
    }
    let base = &images[0].1;
    for (mode, img) in &images[1..] {
        let diff: u64 = base.px.iter().zip(&img.px).map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u64).sum();
        let mean = diff as f64 / base.px.len() as f64;
        eprintln!("[cf] image {mode} vs sync: mean |d| {mean:.4}/255");
        assert!(mean < 0.5, "{mode} install renders differently from the synchronous one ({mean})");
    }
}


// ---------------------------------------------------------------------------
// Vertex colours on the cluster path

/// A torus with a vertex colour ramp (red along u, a hard green step halfway round the tube, blue along v, alpha 1).
fn coloured_torus(nu: u32, nv: u32) -> MeshData {
    let mut m = torus(nu, nv);
    for i in 0..=nu {
        for j in 0..=nv {
            let v = &mut m.vertices[(i * (nv + 1) + j) as usize];
            v.color = [i as f32 / nu as f32, if j < nv / 2 { 0.15 } else { 1.0 }, j as f32 / nv as f32, 1.0];
        }
    }
    m
}

/// The same coloured mesh drawn from clusters (builder v3: the colour rides in the cluster vertex) and on the
/// ordinary path must look the same, and the colours must really show (not a white fallback).
fn run_colour(window: Arc<Window>) {
    let mut r = Renderer::new(window).expect("Renderer::new");
    r.set_vsync(false);
    r.set_cluster_settings(ClusterSettings { mode: ClusterMode::On, ..r.cluster_settings() });
    let mut w = World::new(&mut r, false, false);
    let m = material(&mut r, [1.0, 1.0, 1.0, 1.0], 0.6, 0.0);
    let mut handles = Vec::new();
    for (name, mesh) in [("coloured", coloured_torus(400, 200)), ("white", torus(400, 200))] {
        let dag = zestengine_assets::cluster_dag::build_cluster_dag(&mesh, &[], &DagOptions::default()).expect("DAG");
        assert_eq!(dag.header.flags & zestengine_assets::cluster_dag::FLAG_EXTRA != 0, name == "coloured");
        let h = r.upload_mesh(&mesh).unwrap();
        r.install_cluster_dag(h, &dag.to_gpu_upload()).expect("a vertex-coloured DAG installs");
        let (c, rad) = aabb(&mesh);
        eprintln!("[cf] colour torus {name}: {} tris, {} clusters, {} levels", mesh.indices.len() / 3, dag.clusters.len(), dag.header.levels);
        handles.push((h, c, rad));
    }
    let at = Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)) * Mat4::from_scale(Vec3::splat(1.5));
    w.extra.push((handles[0].0, m, at, handles[0].1, handles[0].2 * 1.5));
    let mut rig = Rig { r, w, ext: [960, 540], t: 0.0, frames: 0, shadows: false, eye: cams()[0].0, target: cams()[0].1, ground_only: false, hiz: false, frozen: true, game: false, lights: Vec::new(), mode: 0, fov: 1.0 };
    rig.opts(false);
    let (iou, mean) = phase_compare(&mut rig, "colour");
    eprintln!("[cf] coloured torus Off vs On: worst silhouette IoU {iou:.5}, worst mean |d| {mean:.4}/255");
    assert!(iou > 0.97 && mean < 1.5, "the clustered coloured mesh differs from the ordinary one");
    // Colours really reach the pixels: the same torus without colours (clusters On) is clearly different.
    rig.set_clusters(true);
    let coloured = rig.shot(6);
    rig.w.extra[0].0 = handles[1].0;
    let plain = rig.shot(6);
    let d: u64 = coloured.px.iter().zip(&plain.px).map(|(a, b)| a.abs_diff(*b) as u64).sum();
    let mean_cw = d as f64 / coloured.px.len() as f64;
    eprintln!("[cf] clustered coloured vs clustered white torus: mean |d| {mean_cw:.2}/255 (colours reach the pixels)");
    dump(&coloured, "colour_on_coloured");
    dump(&plain, "colour_on_white");
    assert!(mean_cw > 3.0, "vertex colours do not show on the cluster path (mean {mean_cw})");
}

// ---------------------------------------------------------------------------
// Install census of a real model

/// `ZESTENGINE_CF_MODEL` (an FBX / glTF, imported from a copy): every mesh with a DAG is uploaded and installed
/// the way the engine does; prints which stay on the ordinary path, with the vertex-coloured ones marked.
fn run_census(window: Arc<Window>) {
    let Some(path) = std::env::var_os("ZESTENGINE_CF_MODEL") else {
        eprintln!("[cf] census: ZESTENGINE_CF_MODEL not set: skipped");
        return;
    };
    let src = std::path::PathBuf::from(&path);
    let dir = std::env::temp_dir().join(format!("cf_census_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let copy = dir.join(src.file_name().unwrap());
    std::fs::copy(&src, &copy).unwrap();
    let t = std::time::Instant::now();
    let loaded = zestengine_assets::load_model_with_geometry(&copy);
    let _ = std::fs::remove_dir_all(&dir);
    let (scene, geo) = loaded.expect("model loads");
    eprintln!("[cf] census {}: imported + DAGs in {:.1} s ({} meshes)", src.display(), t.elapsed().as_secs_f32(), scene.meshes.len());
    let mut r = Renderer::new(window).expect("Renderer::new");
    r.set_vsync(false);
    r.set_cluster_settings(ClusterSettings { mode: ClusterMode::On, ..r.cluster_settings() });
    let (mut tris_all, mut tris_on, mut meshes_on, mut dag_meshes, mut coloured_on, mut coloured_all) = (0u64, 0u64, 0, 0, 0, 0);
    for (i, mesh) in scene.meshes.iter().enumerate() {
        let tris = (mesh.indices.len() / 3) as u64;
        tris_all += tris;
        let coloured = mesh.vertices.iter().any(|v| v.color != [1.0; 4]);
        coloured_all += coloured as usize;
        let Some(dag) = geo.meshes[i].as_ref() else { continue };
        dag_meshes += 1;
        let h = r.upload_mesh(mesh).unwrap();
        let res = r.install_cluster_dag(h, &dag.to_gpu_upload());
        match &res {
            Ok(()) => {
                meshes_on += 1;
                tris_on += tris;
                coloured_on += coloured as usize;
            }
            Err(e) => eprintln!("[cf] census mesh {i}: {tris} tris REFUSED: {e:#}"),
        }
        if coloured || res.is_err() {
            eprintln!("[cf] census mesh {i}: {tris} tris, vertex colours {coloured}, skinned {}, {}", dag.to_gpu_upload().skinned, if res.is_ok() { "installed" } else { "ordinary path" });
        }
    }
    eprintln!(
        "[cf] census: {dag_meshes} DAG-sized meshes, {meshes_on} installed ({} coloured of {coloured_all} coloured meshes); installed triangles {tris_on} of {tris_all} ({:.1}%)",
        coloured_on,
        100.0 * tris_on as f64 / tris_all.max(1) as f64
    );
    assert_eq!(meshes_on, dag_meshes, "some DAG-sized meshes were refused by the renderer");
}

// ---------------------------------------------------------------------------
// Harness

struct App {
    result: Option<Result<(), String>>,
    done: bool,
}

/// `ZESTENGINE_CF_ONLY=<name>` runs one phase (see `run_all`); unset runs them all.
fn wanted(name: &str) -> bool {
    // The census needs a model and is only run on request.
    std::env::var("ZESTENGINE_CF_ONLY").map_or(name != "census", |v| v == name)
}

fn run_all(window: Arc<Window>) {
    if wanted("main") {
        run_main(window.clone());
    }
    if wanted("sampler") {
        run_live_sampler(window.clone());
    }
    if wanted("churn") {
        run_free_churn(window.clone());
    }
    if wanted("swaps") {
        run_swap_churn(window.clone());
    }
    if wanted("dag") {
        run_dag_install(window.clone());
    }
    if wanted("colour") {
        run_colour(window.clone());
    }
    if wanted("census") {
        run_census(window.clone());
    }
    if wanted("release") {
        run_impostor_release(window.clone());
    }
    if wanted("impostor") {
        run_impostor(window.clone());
    }
    if wanted("textured") {
        run_impostor_textured(window.clone());
    }
    if wanted("shadow") {
        run_shadow(window.clone());
    }
    if wanted("overflow") {
        run_overflow(window);
    }
}

fn run_main(window: Arc<Window>) {
    let mut r = Renderer::new(window).expect("Renderer::new");
    r.set_vsync(false);
    r.set_gpu_zone_timing(true);
    // DAGs can only be installed while the path is requested; the phases toggle it.
    r.set_cluster_settings(ClusterSettings { mode: ClusterMode::On, ..r.cluster_settings() });
    let frames = env_n("ZESTENGINE_CF_FRAMES", 360);
    let bench_avatars = env_n("ZESTENGINE_CF_AVATARS", 20);
    eprintln!("[cf] {}", r.gpu_caps().report().lines().next().unwrap_or(""));

    // ---- Scene A: ground + helmets + avatars (image comparison, live-loop behaviours)
    let mut w = World::new(&mut r, true, true);
    w.helmet_grid = vec![Vec3::new(-3.2, 1.0, -2.5), Vec3::new(3.2, 1.0, -2.5), Vec3::new(0.0, 1.0, -6.0)];
    w.add_avatars(&mut r, 4);
    let mut rig = Rig { r, w, ext: [960, 540], t: 0.0, frames: 0, shadows: false, eye: cams()[0].0, target: cams()[0].1, ground_only: false, hiz: false, frozen: false, game: false, lights: Vec::new(), mode: 0, fov: 1.0 };
    rig.opts(false);
    let (iou, mean) = phase_compare(&mut rig, "mixed");
    eprintln!("[cf] images Off vs On (no Hi-Z): worst silhouette IoU {iou:.5}, worst mean |d| {mean:.4}/255");
    rig.opts(true);
    let (iou2, mean2) = phase_compare(&mut rig, "mixed+hiz");
    eprintln!("[cf] images Off vs On (Hi-Z on, two-phase): worst silhouette IoU {iou2:.5}, worst mean |d| {mean2:.4}/255");
    rig.opts(false);
    assert!(iou > 0.97 && iou2 > 0.97, "silhouettes differ");
    phase_toggle(&mut rig);
    phase_resize(&mut rig);
    phase_game(&mut rig);
    phase_long(&mut rig, frames);
    phase_budget(&mut rig);

    // ---- Timing (a): bench avatars
    rig.w.clear_avatars(&mut rig.r);
    rig.w.helmet_grid.clear();
    rig.set_clusters(true); // DAG sharing needs the path requested
    rig.w.add_avatars(&mut rig.r, bench_avatars);
    rig.eye = Vec3::new(0.0, 3.0, 12.0);
    rig.target = Vec3::new(0.0, 1.0, 0.0);
    rig.ext = [1920, 1080];
    phase_timing(&mut rig, &format!("(a) {bench_avatars} skinned avatars, 1920x1080"), &[false, true]);

    // ---- Timing (b): heavy static
    rig.w.clear_avatars(&mut rig.r);
    rig.set_clusters(true);
    rig.w.add_torus(&mut rig.r, 1000, 500);
    for (side, spacing, eye) in [(10usize, 3.0f32, Vec3::new(0.0, 12.0, 26.0)), (32, 2.5, Vec3::new(0.0, 30.0, 70.0))] {
        let half = (side - 1) as f32 * spacing * 0.5;
        let pts: Vec<Vec3> = (0..side * side).map(|i| Vec3::new((i % side) as f32 * spacing - half, 1.0, (i / side) as f32 * spacing - half)).collect();
        rig.eye = eye;
        rig.target = Vec3::ZERO;
        rig.w.torus = rig.w.torus.take();
        rig.w.helmet_grid = pts.clone();
        rig.w.torus_grid.clear();
        phase_timing(&mut rig, &format!("(b1) {} DamagedHelmets (15 452 triangles each, {:.1}M total), 1920x1080", side * side, (side * side) as f32 * 15452.0 / 1e6), &[false]);
        rig.w.helmet_grid.clear();
        rig.w.torus_grid = pts;
        phase_timing(&mut rig, &format!("(b2) {} tori (1M triangles each, {}M total), 1920x1080", side * side, side * side), &[false]);
    }
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
                let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_all(w)));
                self.result = Some(out.map_err(|e| {
                    e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "panic".into())
                }));
            }
            Err(e) => eprintln!("[cf] no window ({e}): skipped"),
        }
        event_loop.exit();
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
}

#[test]
fn cluster_path_in_the_real_frame_loop() {
    if std::env::var("ZESTENGINE_TEST_GPU").is_ok_and(|v| v == "0") || std::env::var_os("DISPLAY").is_none() {
        eprintln!("[cf] no display / GPU disabled: skipped");
        return;
    }
    let _gpu = zestengine_render::cluster_headless::gpu_lock();
    let Ok(mut event_loop) = EventLoop::builder().with_x11().with_any_thread(true).build() else {
        eprintln!("[cf] no X11 event loop: skipped");
        return;
    };
    let mut app = App { result: None, done: false };
    event_loop.run_app_on_demand(&mut app).unwrap();
    if let Some(Err(e)) = app.result {
        panic!("{e}");
    }
}
