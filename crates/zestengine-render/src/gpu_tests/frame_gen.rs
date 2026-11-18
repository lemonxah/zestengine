use std::sync::Arc;

use glam::{Mat4, Vec3};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::{Window, WindowId};

use crate::{CameraData, DrawCmd, LightData, LightInstance, LightKind, MaterialDesc, Renderer, ShadowCasting, UpscaleQuality, Upscaler};

const W: u32 = 960;
const H: u32 = 540;

struct App {
    done: bool,
    error: Option<String>,
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
        let eye = Vec3::new(2.5, 3.0, 6.0);
        let mk = |mesh, transform, radius| DrawCmd {
            gi_source: None,
            mesh,
            material: mat,
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
        let sun = [LightInstance {
            kind: LightKind::Directional,
            position: Vec3::ZERO,
            direction: Vec3::new(-1.0, -1.0, 0.0).normalize(),
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
        let frames = |r: &mut Renderer, count: u32, first: u32| {
            for k in 0..count {
                let frame = first + k;
                let spin = Mat4::from_rotation_y(frame as f32 * 0.0);
                let draws = [
                    mk(ground, Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2) * Mat4::from_scale(Vec3::splat(30.0)), 1.5),
                    mk(cube, Mat4::from_translation(Vec3::new(0.0, 0.5, 0.0)) * spin, 0.9),
                ];
                let eye = Mat4::from_rotation_y(frame as f32 * 0.03).transform_point3(eye);
                let view = Mat4::look_at_rh(eye, Vec3::new(0.0, 0.4, 0.0), Vec3::Y);
                let camera = CameraData { view, proj: Mat4::perspective_rh(1.0, W as f32 / H as f32, 0.05, 200.0), position: eye };
                let mut input = crate::cascade_tests::input(camera);
                input.draws = &draws;
                input.light = LightData { intensity: 3.0, ambient: [0.05; 3], env_intensity: 0.0, ..Default::default() };
                input.lights = &sun;
                input.viewport_extent = None;
                input.render_viewport = true;
                input.time = frame as f32 / 60.0;
                r.render(&input).unwrap_or_else(|e| panic!("render frame {frame}: {e:#}"));
            }
        };
        frames(&mut r, 8, 0);
        r.set_upscale_quality(UpscaleQuality::Quality);
        r.set_frame_gen(true);
        frames(&mut r, 40, 8);
        if r.upscaler() != Upscaler::Fsr3 || !r.frame_gen_active() {
            self.error = Some(format!(
                "frame generation did not start: upscaler {:?}, upscaler error {:?}, frame generation error {:?}",
                r.upscaler(),
                r.upscaler_error(),
                r.frame_gen_error()
            ));
            return;
        }
        let (real_before, generated_before) = r.frame_gen_presents();
        frames(&mut r, 60, 48);
        let started = std::time::Instant::now();
        let (real, generated) = loop {
            let counts = r.frame_gen_presents();
            if counts.0 >= real_before + 60 || started.elapsed() > std::time::Duration::from_secs(5) {
                break counts;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        };
        let (real, generated) = (real - real_before, generated - generated_before);
        eprintln!("[frame generation] 60 rendered frames: {real} real and {generated} generated presents, {:.3} ms GPU", r.stats().gpu_frame_ms);
        if let Some(e) = r.frame_gen_error() {
            self.error = Some(format!("frame generation dispatch failed: {e}"));
            return;
        }
        if real < 60 || generated < 50 {
            self.error = Some(format!("expected about 60 real and 60 generated presents, got {real} and {generated}"));
            return;
        }
        r.debug_capture_viewport_frames(true);
        frames(&mut r, 2, 200);
        let shots = r.debug_take_swapchain_frames().unwrap_or_default();
        r.debug_capture_viewport_frames(false);
        if shots.len() == 4 && shots.iter().all(|s| (s.0, s.1) == (shots[0].0, shots[0].1) && !s.2.is_empty()) {
            let (w, h) = (shots[0].0, shots[0].1);
            let pixels = (w * h) as f64;
            let difference = |a: &[u8], b: &[u8]| {
                a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0.iter()).map(|(p, q)| (0..3).map(|k| (p[k] as f64 - q[k] as f64).abs()).sum::<f64>()).sum::<f64>() / (pixels * 3.0)
            };
            let mean = |px: &[u8]| px.as_chunks::<4>().0.iter().map(|p| p[0] as f64 + p[1] as f64 + p[2] as f64).sum::<f64>() / (pixels * 3.0);
            let (previous, generated_frame, current) = (&shots[1].2, &shots[2].2, &shots[3].2);
            let step = difference(previous, current);
            let to_previous = difference(generated_frame, previous);
            let to_current = difference(generated_frame, current);
            eprintln!(
                "[frame generation] {w}x{h}: real to real {step:.3}, generated to previous {to_previous:.3}, generated to current {to_current:.3}, means {:.1} / {:.1} / {:.1}",
                mean(previous),
                mean(generated_frame),
                mean(current)
            );
            if let Some(dir) = std::env::var_os("ZESTENGINE_TEST_DUMP_DIR") {
                for (name, px) in [("previous", previous), ("generated", generated_frame), ("current", current)] {
                    let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
                    ppm.extend(px.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]));
                    let _ = std::fs::write(std::path::Path::new(&dir).join(format!("framegen_{name}.ppm")), ppm);
                }
            }
            if step < 0.05 || to_previous >= step || to_current >= step || (mean(generated_frame) - mean(current)).abs() > 3.0 {
                self.error = Some(format!(
                    "the generated frame is not between its neighbours: real to real {step:.3}, to previous {to_previous:.3}, to current {to_current:.3}"
                ));
                return;
            }
        } else {
            eprintln!(
                "[frame generation] capture gave {} images {:?}: the picture was not compared",
                shots.len(),
                shots.iter().map(|s| (s.0, s.1, s.2.len())).collect::<Vec<_>>()
            );
        }
        r.set_vsync(true);
        frames(&mut r, 4, 220);
        let (real_before, generated_before) = r.frame_gen_presents();
        frames(&mut r, 20, 230);
        let started = std::time::Instant::now();
        let (real, generated) = loop {
            let counts = r.frame_gen_presents();
            if counts.0 >= real_before + 20 || started.elapsed() > std::time::Duration::from_secs(5) {
                break counts;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        };
        eprintln!("[frame generation] with vsync: {} real and {} generated presents for 20 rendered frames", real - real_before, generated - generated_before);
        if real - real_before < 20 || generated - generated_before < 16 {
            self.error = Some(format!("with vsync expected about 20 real and 20 generated presents, got {} and {}", real - real_before, generated - generated_before));
            return;
        }
        r.set_vsync(false);
        r.set_frame_gen(false);
        frames(&mut r, 10, 300);
        if r.frame_gen_active() || r.upscaler() != Upscaler::Fsr3 {
            self.error = Some("switching frame generation off did not return to plain upscaling".into());
            return;
        }
        r.set_upscale_quality(UpscaleQuality::Native);
        frames(&mut r, 6, 400);
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
fn frame_generation_presents_an_in_between_frame_for_every_rendered_one() {
    if std::env::var("ZESTENGINE_TEST_GPU").is_ok_and(|v| v == "0") || std::env::var_os("DISPLAY").is_none() {
        eprintln!("[gpu test] no display / GPU disabled: skipped");
        return;
    }
    let _gpu = crate::cluster_headless::gpu_lock();
    let Ok(mut event_loop) = EventLoop::builder().with_x11().with_any_thread(true).build() else {
        eprintln!("[gpu test] no X11 event loop: skipped");
        return;
    };
    let mut app = App { done: false, error: None };
    event_loop.run_app_on_demand(&mut app).unwrap();
    if let Some(e) = &app.error {
        panic!("{e}");
    }
}
