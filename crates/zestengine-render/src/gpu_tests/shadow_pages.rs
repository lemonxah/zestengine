use std::sync::Arc;

use glam::{Mat4, Vec3, Vec4};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::{Window, WindowId};

use crate::{CameraData, DrawCmd, LightData, LightInstance, LightKind, MaterialDesc, MaterialHandle, MeshHandle, Renderer, ShadowCasting};

const W: u32 = 640;
const H: u32 = 360;

struct Rig {
    r: Renderer,
    ground: MeshHandle,
    cube: MeshHandle,
    mat: MaterialHandle,
    frame: u32,
    rod: bool,
}

const ROD_X: f32 = 1.5;
const ROD_HEIGHT: f32 = 0.5;
const ROD_NEAR: f32 = 3.0;
const ROD_FAR: f32 = -40.0;

fn sun_dir() -> Vec3 {
    Vec3::new(-1.0, -1.0, 0.0).normalize()
}

fn rod_shadow(z: f32) -> Vec3 {
    let d = sun_dir();
    Vec3::new(ROD_X + d.x / -d.y * ROD_HEIGHT, 0.0, z)
}

fn level0_boundary(cam: &CameraData) -> Option<f32> {
    let basis = crate::vsm::light_basis(sun_dir(), Vec3::Y);
    let page_cam = crate::vsm::PageCamera::from_matrices(cam.view, cam.proj, cam.position);
    let keys = crate::vsm::needed_pages(basis, &page_cam, 0);
    let size = crate::vsm::page_size(0);
    let offset = size * 0.381_966;
    let (x0, x1) = (keys.iter().map(|k| k.x).min()?, keys.iter().map(|k| k.x).max()? + 1);
    let (y0, y1) = (keys.iter().map(|k| k.y).min()?, keys.iter().map(|k| k.y).max()? + 1);
    let inside = |p: Vec3| {
        let l = basis.transform_point3(p);
        l.x >= x0 as f32 * size + offset && l.x <= x1 as f32 * size + offset && l.y >= y0 as f32 * size + offset && l.y <= y1 as f32 * size + offset
    };
    let mut z = ROD_NEAR - 1.0;
    if !inside(rod_shadow(z)) {
        return None;
    }
    while z > ROD_FAR + 2.0 {
        if !inside(rod_shadow(z)) {
            return Some(z);
        }
        z -= 0.01;
    }
    None
}

fn line_darkness(px: &[u8], cam: &CameraData, z: f32) -> f32 {
    let pixel = |p: Vec3| {
        let c = cam.proj * cam.view * Vec4::new(p.x, p.y, p.z, 1.0);
        let n = c.truncate() / c.w;
        (((n.x * 0.5 + 0.5) * W as f32) as i32, ((1.0 - (n.y * 0.5 + 0.5)) * H as f32) as i32)
    };
    let lum = |x: i32, y: i32| {
        let i = (y.clamp(0, H as i32 - 1) as usize * W as usize + x.clamp(0, W as i32 - 1) as usize) * 4;
        px[i] as f32 + px[i + 1] as f32 + px[i + 2] as f32
    };
    let (x, y) = pixel(rod_shadow(z));
    let (ox, oy) = pixel(rod_shadow(z) + Vec3::new(1.0, 0.0, 0.0));
    let open = lum(ox, oy);
    (-8..=8).map(|dx| (open - lum(x + dx, y)).max(0.0)).sum()
}

fn camera(eye: Vec3) -> CameraData {
    let view = Mat4::look_at_rh(eye, eye + Vec3::new(-0.35, -0.45, -1.0), Vec3::Y);
    CameraData { view, proj: Mat4::perspective_rh(1.0, W as f32 / H as f32, 0.05, 200.0), position: eye }
}

fn lum_at(px: &[u8], cam: &CameraData, p: Vec3) -> f32 {
    let c = cam.proj * cam.view * Vec4::new(p.x, p.y, p.z, 1.0);
    let n = c.truncate() / c.w;
    let (x, y) = (((n.x * 0.5 + 0.5) * W as f32) as i32, ((1.0 - (n.y * 0.5 + 0.5)) * H as f32) as i32);
    let mut sum = 0.0;
    for dy in -2..=2 {
        for dx in -2..=2 {
            let i = ((y + dy).clamp(0, H as i32 - 1) as usize * W as usize + (x + dx).clamp(0, W as i32 - 1) as usize) * 4;
            sum += px[i] as f32 + px[i + 1] as f32 + px[i + 2] as f32;
        }
    }
    sum / 25.0
}

impl Rig {
    fn render(&mut self, paged: bool, eye: Vec3, cube_x: f32) -> u32 {
        let mk = |mesh, transform, radius| DrawCmd {
            gi_source: None,
            mesh,
            material: self.mat,
            transform,
            highlight: 0.0,
            mesh_center: Vec3::ZERO,
            bound_radius: radius,
            lightmap_layer: -1,
            lightmap_size: 0,
            static_geometry: false,
            planar_index: -1,
            occluded: false,
            shadow_casting: ShadowCasting::On,
            instance_color: None,
            instance_param: None,
        };
        let draws = [
            mk(self.ground, Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2) * Mat4::from_scale(Vec3::splat(60.0)), 1.5),
            mk(self.cube, Mat4::from_translation(Vec3::new(cube_x, 0.5, 0.0)), 0.9),
            mk(self.cube, Mat4::from_translation(Vec3::new(ROD_X, ROD_HEIGHT, (ROD_NEAR + ROD_FAR) * 0.5)) * Mat4::from_scale(Vec3::new(0.03, 0.03, ROD_NEAR - ROD_FAR)), 0.9),
        ];
        let draws = &draws[..if self.rod { 3 } else { 2 }];
        let sun = [LightInstance {
            kind: LightKind::Directional,
            position: Vec3::ZERO,
            direction: sun_dir(),
            color: [1.0, 1.0, 1.0],
            intensity: 3.0,
            range: 0.0,
            spot_inner_deg: 0.0,
            spot_outer_deg: 0.0,
            cast_shadows: true,
            soft_shadows: true,
            radius: 0.0,
            shadow_bias: 0.002,
            shadow_clip: None,
            baked: false,
        }];
        let mut input = crate::cascade_tests::input(camera(eye));
        input.draws = draws;
        input.light = LightData { intensity: 3.0, ambient: [0.02; 3], env_intensity: 0.0, ..Default::default() };
        input.lights = &sun;
        input.shadow_distance = 60.0;
        input.shadow_virtual = paged;
        input.viewport_extent = Some([W, H]);
        input.render_viewport = false;
        input.time = self.frame as f32 / 60.0;
        self.frame += 1;
        self.r.render(&input).unwrap_or_else(|e| panic!("render: {e:#}"));
        self.r.stats().shadow_layers_drawn
    }

    fn shot(&mut self, paged: bool, eye: Vec3, cube_x: f32) -> Vec<u8> {
        for _ in 0..10 {
            self.render(paged, eye, cube_x);
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
        let ground = r.upload_mesh(&super::ao_engine::quad(1.0)).unwrap();
        let cube = r.upload_mesh(&super::planar::cube()).unwrap();
        let mat = r.create_material(&MaterialDesc::default()).unwrap();
        let mut rig = Rig { r, ground, cube, mat, frame: 0, rod: false };
        let eye = Vec3::new(2.5, 3.0, 6.0);
        let cam = camera(eye);
        let shadowed = Vec3::new(-1.2, 0.0, 0.0);
        let open = Vec3::new(1.5, 0.0, 1.5);
        let mut report = Vec::new();
        let mut check = |ok: bool, msg: String| {
            eprintln!("[shadow pages] {}{msg}", if ok { "ok   " } else { "FAIL " });
            if !ok {
                report.push(msg);
            }
        };

        let cascaded = rig.shot(false, eye, 0.0);
        let paged = rig.shot(true, eye, 0.0);
        let dump = |name: &str, px: &[u8]| {
            if let Some(dir) = std::env::var_os("ZESTENGINE_TEST_DUMP_DIR") {
                let mut ppm = format!("P6\n{W} {H}\n255\n").into_bytes();
                ppm.extend(px.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]));
                let _ = std::fs::write(std::path::Path::new(&dir).join(format!("shadow_pages_{name}.ppm")), ppm);
            }
        };
        dump("cascaded", &cascaded);
        dump("paged", &paged);
        let (cs, co) = (lum_at(&cascaded, &cam, shadowed), lum_at(&cascaded, &cam, open));
        let (ps, po) = (lum_at(&paged, &cam, shadowed), lum_at(&paged, &cam, open));
        check(cs < co * 0.5, format!("cascades: shadowed ground {cs:.1} vs open {co:.1} (want < half)"));
        check(ps < po * 0.5, format!("pages: shadowed ground {ps:.1} vs open {po:.1} (want < half)"));
        check((po / co.max(1.0) - 1.0).abs() < 0.03, format!("open ground reads the same in both modes: pages {po:.1}, cascades {co:.1}"));
        check((ps - cs).abs() < co * 0.08, format!("the shadow is as dark in both modes: pages {ps:.1}, cascades {cs:.1}"));
        let differing = cascaded.as_chunks::<4>().0.iter().zip(paged.as_chunks::<4>().0.iter()).filter(|(a, b)| (0..3).any(|k| (a[k] as i32 - b[k] as i32).abs() > 24)).count();
        let share = differing as f32 / (W * H) as f32;
        check(share < 0.02, format!("pages and cascades disagree on {:.2}% of pixels (want < 2%)", share * 100.0));

        for _ in 0..6 {
            rig.render(true, eye, 0.0);
        }
        let mut walked = 0;
        for step in 1..=30 {
            walked += rig.render(true, eye + Vec3::new(0.01, 0.0, -0.015) * step as f32, 0.0);
        }
        check(walked == 0, format!("pages: a 0.5 m walk over a still scene redrew {walked} shadow layers (want 0)"));
        for _ in 0..6 {
            rig.render(false, eye, 0.0);
        }
        let mut walked_cascades = 0;
        for step in 1..=30 {
            walked_cascades += rig.render(false, eye + Vec3::new(0.01, 0.0, -0.015) * step as f32, 0.0);
        }
        check(walked_cascades > walked, format!("cascades redrew {walked_cascades} layers on the same walk"));

        for _ in 0..6 {
            rig.render(true, eye, 0.0);
        }
        let mut moved = 0;
        for step in 1..=10 {
            moved += rig.render(true, eye, step as f32 * 0.02);
        }
        check(moved >= 10 && moved <= 10 * 6, format!("pages: a moving caster redrew {moved} layers over 10 frames (want only the pages it is in)"));
        let after = rig.shot(true, eye, 0.2);
        let moved_shadow = lum_at(&after, &cam, Vec3::new(-1.0, 0.0, 0.0));
        check(moved_shadow < po * 0.5, format!("pages: the shadow followed the caster ({moved_shadow:.1} vs open {po:.1})"));
        rig.rod = true;
        let rod_cam = camera(eye);
        match level0_boundary(&rod_cam) {
            None => check(false, "the rod's shadow line does not cross the level 0 boundary".into()),
            Some(edge) => {
                let lined = rig.shot(true, eye, 0.0);
                dump("rod", &lined);
                let profile: Vec<(f32, f32)> = (0..=80).map(|k| edge + 2.0 - k as f32 * 0.05).map(|z| (z, line_darkness(&lined, &rod_cam, z))).collect();
                let steps: Vec<(f32, f32)> = profile.windows(2).map(|w| ((w[0].0 + w[1].0) * 0.5, (w[1].1 - w[0].1).abs())).collect();
                let at_edge = steps.iter().filter(|(z, _)| (z - edge).abs() < 0.6).map(|s| s.1).fold(0.0f32, f32::max);
                let mut away: Vec<f32> = steps.iter().filter(|(z, _)| (z - edge).abs() >= 0.6).map(|s| s.1).collect();
                away.sort_by(f32::total_cmp);
                let typical = away[away.len() * 9 / 10];
                eprintln!("[shadow pages] rod edge z {edge:.2}: darkness {:?}", profile.iter().step_by(4).map(|(z, d)| format!("{z:.1}:{d:.0}")).collect::<Vec<_>>());
                check(at_edge <= typical * 3.0, format!("pages: the rod's shadow loses {at_edge:.1} in one 5 cm step where it leaves level 0 (z {edge:.2}), {typical:.1} elsewhere (want under three times that: no seam between levels)"));
            }
        }
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
fn virtual_shadow_pages_match_cascades_and_are_not_redrawn_while_the_camera_moves() {
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
    assert!(app.report.is_empty(), "virtual shadow pages in the renderer:\n{}", app.report.join("\n"));
}
