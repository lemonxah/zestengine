//! The editor ground grid on a REAL `Renderer`: drawn into the editor viewport
//! target only, with the plane / axis colours / occlusion / fades it promises.
//! Every check compares a grid-on render against the same frame with the grid
//! off (an empty scene, no skybox), so only the grid's own pixels differ:
//!   * from above: grid lines near the origin, a red X axis line (z = 0) and a
//!     blue Z axis line (x = 0), untouched cell centres, unit-spaced lines;
//!   * at a grazing angle: grid pixels near the origin, nothing above the
//!     horizon, the far field faded instead of aliasing;
//!   * a cube in front hides it; turning it off restores the exact frame;
//!   * the Camera tab preview and the game (swapchain) path never show it;
//!   * a 2D (orthographic, looking down -Z) view puts it on the XY plane with a
//!     green Y axis line.
//!
//! Needs a display; skipped without one or with `ZESTENGINE_TEST_GPU=0`.

use std::sync::Arc;

use glam::{Mat4, Vec2, Vec3};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::{Window, WindowId};

use super::planar::{cube, draw, lit};
use crate::{CameraData, DrawCmd, EditorGridStyle, GridPlane, LightInstance, LightKind, Renderer};

const W: u32 = 640;
const H: u32 = 360;

fn persp(eye: Vec3, target: Vec3) -> CameraData {
    let view = Mat4::look_at_rh(eye, target, Vec3::Y);
    CameraData { view, proj: Mat4::perspective_rh(0.9, W as f32 / H as f32, 0.1, 500.0), position: eye }
}

fn style(enabled: bool, unit: f32) -> EditorGridStyle {
    EditorGridStyle { enabled, plane: GridPlane::Auto, unit, ..EditorGridStyle::default() }
}

/// Pixel (x, y) of a world point (top row = NDC y +1, as the viewport draws).
fn to_px(cam: &CameraData, p: Vec3) -> (i32, i32) {
    let c = cam.proj * cam.view * p.extend(1.0);
    let n = c.truncate() / c.w;
    (((n.x + 1.0) * 0.5 * W as f32) as i32, ((1.0 - n.y) * 0.5 * H as f32) as i32)
}

fn diff(a: &[u8], b: &[u8], i: usize) -> i32 {
    (0..3).map(|c| (a[i * 4 + c] as i32 - b[i * 4 + c] as i32).abs()).max().unwrap()
}

fn rgb(px: &[u8], x: i32, y: i32) -> [i32; 3] {
    let i = ((y.clamp(0, H as i32 - 1) as u32 * W + x.clamp(0, W as i32 - 1) as u32) * 4) as usize;
    [px[i] as i32, px[i + 1] as i32, px[i + 2] as i32]
}

/// Best score of `f` over the 5x5 pixels around (x, y).
fn best_near(px: &[u8], (x, y): (i32, i32), f: impl Fn([i32; 3]) -> i32) -> i32 {
    let mut best = i32::MIN;
    for dy in -2..=2 {
        for dx in -2..=2 {
            best = best.max(f(rgb(px, x + dx, y + dy)));
        }
    }
    best
}

/// Largest change between `a` and `b` within `r` pixels of (x, y).
fn changed_near(a: &[u8], b: &[u8], (x, y): (i32, i32), r: i32) -> i32 {
    let mut best = 0;
    for dy in -r..=r {
        for dx in -r..=r {
            let (qx, qy) = (x + dx, y + dy);
            if qx >= 0 && qy >= 0 && qx < W as i32 && qy < H as i32 {
                best = best.max(diff(a, b, (qy as u32 * W + qx as u32) as usize));
            }
        }
    }
    best
}

struct Rig {
    r: Renderer,
    frame: u32,
}

#[derive(Default)]
struct Frame {
    viewport: Vec<u8>,
    preview: Option<Vec<u8>>,
    swapchain: Option<Vec<u8>>,
}

impl Rig {
    /// A few frames with the grid set to `grid`; returns the last one.
    fn render(&mut self, cam: &CameraData, draws: &[DrawCmd], grid: EditorGridStyle, preview: bool, game: bool) -> Frame {
        let sun = [LightInstance {
            kind: LightKind::Directional,
            position: Vec3::ZERO,
            direction: Vec3::new(-0.4, -1.0, -0.5).normalize(),
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
        self.r.set_editor_grid(grid);
        let mut out = Frame::default();
        if game {
            self.r.debug_capture_viewport_frames(true);
            let _ = self.r.debug_take_swapchain_frames();
        }
        for _ in 0..3 {
            let mut input = crate::cascade_tests::input(*cam);
            input.draws = draws;
            input.lights = &sun;
            input.draw_skybox = false;
            input.viewport_extent = (!game).then_some([W, H]);
            input.camera_preview = preview.then_some(*cam);
            input.time = self.frame as f32 / 60.0;
            self.frame += 1;
            self.r.render(&input).unwrap_or_else(|e| panic!("render: {e:#}"));
        }
        if game {
            out.swapchain = self.r.debug_take_swapchain_frames().ok().and_then(|mut v| v.pop()).map(|(_, _, p)| p);
            self.r.debug_capture_viewport_frames(false);
        } else {
            let (w, h, p) = self.r.debug_read_viewport().expect("readback").expect("viewport target");
            assert_eq!((w, h), (W, H));
            out.viewport = p;
        }
        if preview {
            out.preview = self.r.debug_read_camera_preview().expect("preview readback").map(|(_, _, p)| p);
        }
        out
    }
}

/// `ZESTENGINE_GRID_PNG=<dir>` saves the grid-on views there for a look.
fn dump(name: &str, px: &[u8]) {
    if let Some(dir) = std::env::var_os("ZESTENGINE_GRID_PNG") {
        let path = std::path::Path::new(&dir).join(format!("grid-{name}.png"));
        if let Err(e) = std::fs::write(&path, super::stencil::png_rgba(W, H, px)) {
            eprintln!("[grid] could not write {}: {e}", path.display());
        }
    }
}

fn run(r: Renderer, fails: &mut Vec<String>, notes: &mut Vec<String>) {
    let mut rig = Rig { r, frame: 0 };
    // (passed, what) per check; reported at the end.
    let mut res: Vec<(bool, String)> = Vec::new();
    let n = (W * H) as usize;

    // (a) From above (tilted), unit 1 m.
    let top = persp(Vec3::new(0.0, 6.0, 3.0), Vec3::ZERO);
    let off = rig.render(&top, &[], style(false, 1.0), false, false).viewport;
    let on = rig.render(&top, &[], style(true, 1.0), false, false).viewport;
    dump("top", &on);
    let changed = (0..n).filter(|&i| diff(&on, &off, i) > 6).count();
    res.push((changed > n / 100, format!("top view: {changed} grid pixels (> 1% of the frame)")));
    let red = best_near(&on, to_px(&top, Vec3::new(1.5, 0.0, 0.0)), |c| (c[0] - c[2]).min(c[0] - c[1]));
    res.push((red > 30, format!("top view: X axis line (z = 0) is red (r - max(g, b) = {red})")));
    let blue = best_near(&on, to_px(&top, Vec3::new(0.0, 0.0, 1.5)), |c| (c[2] - c[0]).min(c[2] - c[1]));
    res.push((blue > 30, format!("top view: Z axis line (x = 0) is blue (b - r = {blue})")));
    let line = changed_near(&on, &off, to_px(&top, Vec3::new(1.0, 0.0, 0.55)), 1);
    res.push((line > 10, format!("top view: the x = 1 minor line shows ({line})")));
    let centre = changed_near(&on, &off, to_px(&top, Vec3::new(0.5, 0.0, 0.5)), 1);
    res.push((centre <= 3, format!("top view: a cell centre is untouched ({centre})")));

    // Grid unit follows the snap step: 0.5 m adds the x = 0.5 line.
    let half = rig.render(&top, &[], style(true, 0.5), false, false).viewport;
    let l05 = changed_near(&half, &off, to_px(&top, Vec3::new(0.5, 0.0, 0.3)), 1);
    let l05_unit1 = changed_near(&on, &off, to_px(&top, Vec3::new(0.5, 0.0, 0.3)), 1);
    res.push((l05 > 10 && l05_unit1 <= 3, format!("unit 0.5: an x = 0.5 line ({l05}), absent with unit 1 ({l05_unit1})")));

    // (b) Toggle off after on: exactly the grid-less frame.
    let off2 = rig.render(&top, &[], style(false, 1.0), false, false).viewport;
    let leaked = (0..n).filter(|&i| diff(&off2, &off, i) > 1).count();
    res.push((leaked == 0, format!("toggle off: {leaked} px differ from the grid-less frame")));

    // (c) A cube in front hides the grid.
    {
        let mat = lit(&mut rig.r, [0.9, 0.55, 0.15, 1.0], [0.3, 0.2, 0.05], 0.6, 0.0);
        let mesh = rig.r.upload_mesh(&cube()).unwrap();
        let cube_draw = draw(mesh, mat, Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)) * Mat4::from_scale(Vec3::splat(1.6)), -1);
        let c_off = rig.render(&top, &[cube_draw], style(false, 1.0), false, false).viewport;
        let c_on = rig.render(&top, &[cube_draw], style(true, 1.0), false, false).viewport;
        dump("cube", &c_on);
        let mask: Vec<bool> = (0..n).map(|i| diff(&c_off, &off, i) > 1).collect();
        // Eroded by 2 px (the cube's anti-aliased rim mixes with the grid behind it).
        let inner = |i: usize| {
            let (x, y) = ((i as u32 % W) as i32, (i as u32 / W) as i32);
            (-2..=2).all(|dy| (-2..=2).all(|dx| {
                let (qx, qy) = (x + dx, y + dy);
                qx >= 0 && qy >= 0 && qx < W as i32 && qy < H as i32 && mask[(qy as u32 * W + qx as u32) as usize]
            }))
        };
        let (mut inside, mut shown) = (0usize, 0usize);
        for i in 0..n {
            if inner(i) {
                inside += 1;
                shown += (diff(&c_on, &c_off, i) > 3) as usize;
            }
        }
        res.push((inside > 2000 && shown == 0, format!("cube in front: {shown} of its {inside} px show the grid")));
        let outside = (0..n).filter(|&i| !mask[i] && diff(&c_on, &c_off, i) > 6).count();
        res.push((outside > n / 200, format!("cube in front: the grid still shows around it ({outside} px)")));
    }

    // (d) Grazing angle: eye 0.5 m above the ground looking at the horizon.
    {
        let cam = persp(Vec3::new(0.0, 0.5, 6.0), Vec3::new(0.0, 0.35, -20.0));
        let g_off = rig.render(&cam, &[], style(false, 1.0), false, false).viewport;
        let g_on = rig.render(&cam, &[], style(true, 1.0), false, false).viewport;
        dump("grazing", &g_on);
        let inv = (cam.proj * cam.view).inverse();
        let (mut above, mut above_changed, mut far, mut far_sum, mut near_changed) = (0usize, 0i32, 0usize, 0i64, 0usize);
        for y in 0..H {
            for x in 0..W {
                let i = (y * W + x) as usize;
                let ndc = Vec2::new((x as f32 + 0.5) / W as f32 * 2.0 - 1.0, 1.0 - (y as f32 + 0.5) / H as f32 * 2.0);
                let d = diff(&g_on, &g_off, i);
                match crate::grid::plane_hit(inv, ndc, 1) {
                    None => {
                        above += 1;
                        above_changed = above_changed.max(d);
                    }
                    Some(p) => {
                        let dist = (p - cam.position).length();
                        if dist > 80.0 {
                            far += 1;
                            far_sum += d as i64;
                        } else if dist < 8.0 && d > 6 {
                            near_changed += 1;
                        }
                    }
                }
            }
        }
        let far_mean = far_sum as f32 / far.max(1) as f32;
        res.push((above > 1000 && above_changed <= 2, format!("grazing: nothing above the horizon changed (max {above_changed} over {above} px)")));
        res.push((near_changed > 500, format!("grazing: grid pixels near the origin ({near_changed})")));
        res.push((far > 200 && far_mean < 2.0, format!("grazing: the far field (> 80 m, {far} px) is faded, mean change {far_mean:.2}")));
        let x_axis = best_near(&g_on, to_px(&cam, Vec3::new(1.2, 0.0, 0.0)), |c| (c[0] - c[2]).min(c[0] - c[1]));
        res.push((x_axis > 20, format!("grazing: X axis line is red ({x_axis})")));
    }

    // (e) Never in the Camera tab preview or the game (swapchain) output.
    {
        let p_off = rig.render(&top, &[], style(false, 1.0), true, false);
        let p_on = rig.render(&top, &[], style(true, 1.0), true, false);
        match (p_off.preview, p_on.preview) {
            (Some(a), Some(b)) => {
                let d = (0..a.len() / 4).filter(|&i| diff(&a, &b, i) > 1).count();
                res.push((d == 0, format!("Camera tab preview: {d} px differ with the grid on")));
                let vp = (0..n).filter(|&i| diff(&p_on.viewport, &p_off.viewport, i) > 6).count();
                res.push((vp > n / 100, format!("same frame's editor viewport has the grid ({vp} px)")));
            }
            _ => res.push((false, "Camera tab preview target missing".into())),
        }
        let s_off = rig.render(&top, &[], style(false, 1.0), false, true).swapchain;
        let s_on = rig.render(&top, &[], style(true, 1.0), false, true).swapchain;
        match (s_off, s_on) {
            (Some(a), Some(b)) if a.len() == b.len() => {
                let d = (0..a.len() / 4).filter(|&i| diff(&a, &b, i) > 1).count();
                res.push((d == 0, format!("game (swapchain) output: {d} px differ with the grid on")));
            }
            _ => notes.push("skip game path: no swapchain capture on this device".into()),
        }
    }

    // (f) 2D: orthographic, looking down -Z, the grid lies on XY.
    {
        let eye = Vec3::new(0.3, 0.2, 10.0);
        let view = Mat4::look_at_rh(eye, eye - Vec3::Z, Vec3::Y);
        let a = W as f32 / H as f32;
        let cam = CameraData { view, proj: Mat4::orthographic_rh(-4.0 * a, 4.0 * a, -4.0, 4.0, 0.1, 100.0), position: eye };
        let o_off = rig.render(&cam, &[], style(false, 1.0), false, false).viewport;
        let o_on = rig.render(&cam, &[], style(true, 1.0), false, false).viewport;
        dump("2d", &o_on);
        let changed = (0..n).filter(|&i| diff(&o_on, &o_off, i) > 6).count();
        res.push((changed > n / 100, format!("2D view: {changed} grid pixels on the XY plane")));
        let red = best_near(&o_on, to_px(&cam, Vec3::new(1.5, 0.0, 0.0)), |c| (c[0] - c[2]).min(c[0] - c[1]));
        res.push((red > 30, format!("2D view: X axis line (y = 0) is red ({red})")));
        let green = best_near(&o_on, to_px(&cam, Vec3::new(0.0, 1.5, 0.0)), |c| (c[1] - c[0]).min(c[1] - c[2]));
        res.push((green > 20, format!("2D view: Y axis line (x = 0) is green ({green})")));
    }
    for (ok, what) in res {
        notes.push(format!("{} {what}", if ok { "ok  " } else { "FAIL" }));
        if !ok {
            fails.push(what);
        }
    }
}

#[derive(Default)]
struct App {
    done: bool,
    error: Option<String>,
    fails: Vec<String>,
    notes: Vec<String>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.done {
            return;
        }
        self.done = true;
        let attrs = Window::default_attributes().with_visible(false).with_inner_size(winit::dpi::LogicalSize::new(W as f64, H as f64));
        match event_loop.create_window(attrs).map(Arc::new) {
            Ok(w) => match Renderer::new(w) {
                Ok(mut r) => {
                    r.set_vsync(false);
                    run(r, &mut self.fails, &mut self.notes);
                }
                Err(e) => self.error = Some(format!("Renderer::new: {e:#}")),
            },
            Err(e) => self.error = Some(format!("window: {e}")),
        }
        event_loop.exit();
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
}

#[test]
fn editor_grid_draws_in_the_scene_view_only() {
    if std::env::var("ZESTENGINE_TEST_GPU").is_ok_and(|v| v == "0") || std::env::var_os("DISPLAY").is_none() {
        eprintln!("[gpu test] no display / GPU disabled: skipped");
        return;
    }
    let _gpu = crate::cluster_headless::gpu_lock();
    let Ok(mut event_loop) = EventLoop::builder().with_x11().with_any_thread(true).build() else {
        eprintln!("[gpu test] no X11 event loop: skipped");
        return;
    };
    let mut app = App::default();
    event_loop.run_app_on_demand(&mut app).unwrap();
    if let Some(e) = &app.error {
        panic!("{e}");
    }
    for n in &app.notes {
        eprintln!("[grid] {n}");
    }
    assert!(app.fails.is_empty(), "editor grid failures:\n  {}", app.fails.join("\n  "));
}
