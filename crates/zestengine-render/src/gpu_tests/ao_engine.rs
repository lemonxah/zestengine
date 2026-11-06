//! Screen-space AO inside the real `Renderer` (the `ao.rs` unit tests drive the
//! passes alone): two walls and a floor, rendered into the editor viewport
//! target with AO off and on.
//!   * ambient-only light: the inner corner gets darker (> 20 %), an open floor
//!     pixel changes < 2 %;
//!   * a directional light only (no ambient, no environment): AO must not touch
//!     direct light, the images are identical;
//!   * the AO-only view mode (`VIEW_AO`) shows the corner dark and open floor white;
//!   * the game path (`render_viewport = true`, no viewport target) runs the AO
//!     zone too, and the whole AO chain costs < 0.3 ms at 2560x1440.
//!
//! Needs a display; skipped without one or with `ZESTENGINE_TEST_GPU=0`.

use std::sync::Arc;

use glam::{Mat4, Vec3, Vec4};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::{Window, WindowId};

use crate::{
    AoQuality, AoSettings, CameraData, DrawCmd, LightData, LightInstance, LightKind, MaterialDesc, MaterialHandle, MeshData, MeshHandle, Renderer,
    ShadowCasting, Vertex, VIEW_AO,
};

const W: u32 = 960;
const H: u32 = 540; // the corner seam (walls ~12 cm from the sample point) must span real pixels

fn quad(half: f32) -> MeshData {
    let mut vertices = Vec::new();
    for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        vertices.push(Vertex {
            position: [x * half, y * half, 0.0],
            normal: [0.0, 0.0, 1.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            color: [1.0; 4],
            ..Default::default()
        });
    }
    MeshData { vertices, indices: vec![0, 1, 2, 0, 2, 3], has_lightmap_uv: false }
}

fn cube() -> MeshData {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for (n, up, right) in [
        (Vec3::X, Vec3::Y, Vec3::NEG_Z),
        (Vec3::NEG_X, Vec3::Y, Vec3::Z),
        (Vec3::Y, Vec3::NEG_Z, Vec3::X),
        (Vec3::NEG_Y, Vec3::Z, Vec3::X),
        (Vec3::Z, Vec3::Y, Vec3::X),
        (Vec3::NEG_Z, Vec3::Y, Vec3::NEG_X),
    ] {
        let base = vertices.len() as u32;
        for (u, v) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let p = n * 0.5 + right * (u * 0.5) + up * (v * 0.5);
            vertices.push(Vertex { position: p.to_array(), normal: n.to_array(), tangent: [right.x, right.y, right.z, 1.0], color: [1.0; 4], ..Default::default() });
        }
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    MeshData { vertices, indices, has_lightmap_uv: false }
}

#[derive(Clone, Copy, PartialEq)]
enum Light {
    /// Ambient fill only.
    Ambient,
    /// One directional light only (no ambient, no environment).
    Direct,
}

struct Rig {
    r: Renderer,
    ground: MeshHandle,
    cube: MeshHandle,
    mat: MaterialHandle,
    frame: u32,
}

fn camera(w: u32, h: u32) -> CameraData {
    let eye = Vec3::new(1.5, 2.0, 3.5);
    let view = Mat4::look_at_rh(eye, Vec3::new(-0.8, 0.0, 0.0), Vec3::Y);
    CameraData { view, proj: Mat4::perspective_rh(1.0, w as f32 / h as f32, 0.05, 100.0), position: eye }
}

/// Where a world point lands in the viewport image (row 0 = top).
fn pixel_of(cam: &CameraData, p: Vec3, w: u32, h: u32) -> (usize, usize) {
    let c = cam.proj * cam.view * Vec4::new(p.x, p.y, p.z, 1.0);
    let n = c.truncate() / c.w;
    (((n.x * 0.5 + 0.5) * w as f32) as usize, ((1.0 - (n.y * 0.5 + 0.5)) * h as f32) as usize)
}

/// 5x5 mean of the summed colour channels at the pixel of `p`.
fn lum_at(px: &[u8], cam: &CameraData, p: Vec3, w: u32, h: u32) -> f32 {
    let (x, y) = pixel_of(cam, p, w, h);
    let mut s = 0.0;
    for dy in -2i32..=2 {
        for dx in -2i32..=2 {
            let xx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
            let yy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
            let i = (yy * w as usize + xx) * 4;
            s += px[i] as f32 + px[i + 1] as f32 + px[i + 2] as f32;
        }
    }
    s / 25.0
}

impl Rig {
    /// One frame. `viewport` = Some(extent) renders the editor viewport target,
    /// None the game path (swapchain).
    fn render(&mut self, light: Light, ao: Option<AoQuality>, view_mode: u32, viewport: Option<[u32; 2]>) {
        let mk = |mesh, transform| DrawCmd {
            gi_source: None,
            mesh,
            material: self.mat,
            transform,
            highlight: 0.0,
            mesh_center: Vec3::ZERO,
            bound_radius: 40.0,
            lightmap_layer: -1,
            lightmap_size: 0,
            static_geometry: false,
            planar_index: -1,
            occluded: false,
            shadow_casting: ShadowCasting::TwoSided,
            instance_color: None,
            instance_param: None,
        };
        let draws = [
            mk(self.ground, Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2) * Mat4::from_scale(Vec3::splat(10.0))),
            mk(self.cube, Mat4::from_translation(Vec3::new(0.0, 1.0, -1.0)) * Mat4::from_scale(Vec3::new(4.0, 2.0, 0.2))),
            mk(self.cube, Mat4::from_translation(Vec3::new(-2.0, 1.0, 1.0)) * Mat4::from_scale(Vec3::new(0.2, 2.0, 4.0))),
        ];
        let sun = [LightInstance {
            kind: LightKind::Directional,
            position: Vec3::ZERO,
            direction: Vec3::new(-0.3, -1.0, -0.4).normalize(),
            color: [1.0, 0.98, 0.92],
            intensity: 3.0,
            range: 0.0,
            spot_inner_deg: 0.0,
            spot_outer_deg: 0.0,
            cast_shadows: false,
            soft_shadows: false, radius: 0.0,
            shadow_bias: 0.005,
            baked: false,
        }];
        let (w, h) = viewport.map_or((W, H), |e| (e[0], e[1]));
        let mut input = crate::cascade_tests::input(camera(w, h));
        input.draws = &draws;
        match light {
            Light::Ambient => {
                input.light = LightData { intensity: 0.0, ambient: [0.5; 3], env_intensity: 0.0, ..Default::default() };
            }
            Light::Direct => {
                input.light = LightData { intensity: 3.0, ambient: [0.0; 3], env_intensity: 0.0, ..Default::default() };
                input.lights = &sun;
            }
        }
        input.gi_debug = view_mode;
        input.viewport_extent = viewport;
        input.render_viewport = viewport.is_none();
        // Multi-bounce off: on a white albedo it deliberately lightens the occlusion.
        input.postfx.ao = AoSettings { enabled: ao.is_some(), intensity: 1.0, radius: 0.8, falloff: 0.6, multi_bounce: false };
        if let Some(q) = ao {
            self.r.set_ao_quality(q);
        }
        input.time = self.frame as f32 / 60.0;
        self.frame += 1;
        self.r.render(&input).unwrap_or_else(|e| panic!("render: {e:#}"));
    }

    /// Several frames (the temporal filter converges), then a readback of the viewport.
    fn shot(&mut self, light: Light, ao: Option<AoQuality>, view_mode: u32) -> Vec<u8> {
        for _ in 0..14 {
            self.render(light, ao, view_mode, Some([W, H]));
            self.r.debug_read_viewport().expect("readback");
        }
        let (w, h, px) = self.r.debug_read_viewport().expect("readback").expect("viewport target");
        assert_eq!((w, h), (W, H));
        px
    }
}

struct App {
    done: bool,
    error: Option<String>,
    report: Vec<String>,
}

impl App {
    fn run(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes().with_visible(false).with_inner_size(winit::dpi::LogicalSize::new(W as f64, H as f64));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                self.error = Some(format!("window: {e}"));
                return;
            }
        };
        let mut r = match Renderer::new(window) {
            Ok(r) => r,
            Err(e) => {
                self.error = Some(format!("Renderer::new: {e:#}"));
                return;
            }
        };
        r.set_vsync(false);
        r.set_gpu_zone_timing(true);
        let ground = r.upload_mesh(&quad(1.0)).unwrap();
        let cube = r.upload_mesh(&cube()).unwrap();
        let mat = r.create_material(&MaterialDesc::default()).unwrap();
        let mut rig = Rig { r, ground, cube, mat, frame: 0 };
        let cam = camera(W, H);
        let corner = Vec3::new(-1.87, 0.0, -0.87);
        let open = Vec3::new(1.2, 0.0, 1.5);
        let wall = Vec3::new(0.5, 1.0, -0.9);
        let mut report = Vec::new();
        let mut check = |ok: bool, msg: String| {
            eprintln!("[ao engine] {}{msg}", if ok { "ok   " } else { "FAIL " });
            if !ok {
                report.push(msg);
            }
        };

        for quality in AoQuality::ALL {
            let off = rig.shot(Light::Ambient, None, 0);
            let on = rig.shot(Light::Ambient, Some(quality), 0);
            let ratio = |p: Vec3| lum_at(&on, &cam, p, W, H) / lum_at(&off, &cam, p, W, H).max(1e-3);
            let (c, o, wl) = (ratio(corner), ratio(open), ratio(wall));
            check(c < 0.8, format!("{quality:?} ambient-only: corner luminance on/off = {c:.3} (want < 0.8)"));
            check((0.98..=1.02).contains(&o), format!("{quality:?} ambient-only: open floor on/off = {o:.3} (want 0.98..1.02)"));
            check(wl > 0.95, format!("{quality:?} ambient-only: open wall on/off = {wl:.3} (want > 0.95)"));

            let d_off = rig.shot(Light::Direct, None, 0);
            let d_on = rig.shot(Light::Direct, Some(quality), 0);
            let max_diff = d_off.iter().zip(&d_on).map(|(a, b)| (*a as i32 - *b as i32).abs()).max().unwrap();
            check(max_diff <= 2, format!("{quality:?} direct light only: AO changed the image by up to {max_diff} levels (want <= 2)"));

            let view = rig.shot(Light::Ambient, Some(quality), VIEW_AO);
            if let Some(dir) = std::env::var_os("ZESTENGINE_TEST_DUMP_DIR") {
                let mut ppm = format!("P6\n{W} {H}\n255\n").into_bytes();
                ppm.extend(view.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]));
                let _ = std::fs::write(std::path::Path::new(&dir).join(format!("ao_view_{quality:?}.ppm")), ppm);
            }
            let (vc, vo) = (lum_at(&view, &cam, corner, W, H) / 765.0, lum_at(&view, &cam, open, W, H) / 765.0);
            // The view goes through the post pass, whose tonemap compresses the
            // display (AO 1.0 shows ~0.91); the corner must still read clearly
            // darker than the open floor. The real strength gate is the
            // ambient-only ratio above.
            check(vo > 0.8 && vc < vo * 0.88, format!("{quality:?} AO-only view: corner {vc:.3}, open floor {vo:.3} (want open > 0.8, corner < 0.88 * open; AO 1.0 shows ~0.91 after tonemap)"));
        }

        // Game path (no viewport target) and the cost at 2560x1440 in the viewport path.
        rig.r.set_ao_quality(AoQuality::High);
        for _ in 0..30 {
            rig.render(Light::Ambient, Some(AoQuality::High), 0, None);
        }
        let game_ms = rig.r.stats().gpu_ao_ms;
        check(game_ms > 0.0, format!("game path: AO GPU zone ran ({game_ms:.3} ms at {W}x{H})"));
        let mut best = f32::MAX;
        for _ in 0..40 {
            rig.render(Light::Ambient, Some(AoQuality::High), 0, Some([2560, 1440]));
            best = best.min(rig.r.stats().gpu_ao_ms);
        }
        eprintln!("[ao engine] viewport path 2560x1440: High AO GPU zone {best:.3} ms (best of 40)");
        check(best > 0.0 && best < 0.3, format!("High AO at 2560x1440 took {best:.3} ms (want 0 < t < 0.3)"));
        self.report = report;
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !self.done {
            self.done = true;
            self.run(event_loop);
            event_loop.exit();
        }
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
}

#[test]
fn ao_in_the_full_renderer_viewport_and_game_paths() {
    if std::env::var("ZESTENGINE_TEST_GPU").is_ok_and(|v| v == "0") || std::env::var_os("DISPLAY").is_none() {
        eprintln!("[gpu test] no display / GPU disabled: skipped");
        return;
    }
    let _gpu = crate::cluster_headless::gpu_lock();
    let Ok(mut event_loop) = EventLoop::builder().with_x11().with_any_thread(true).build() else {
        eprintln!("[gpu test] no X11 event loop: skipped");
        return;
    };
    let mut app = App { done: false, error: None, report: Vec::new() };
    event_loop.run_app_on_demand(&mut app).unwrap();
    if let Some(e) = &app.error {
        panic!("{e}");
    }
    assert!(app.report.is_empty(), "AO in the renderer:\n{}", app.report.join("\n"));
}
