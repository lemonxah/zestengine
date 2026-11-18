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
                let spin = Mat4::from_rotation_y(frame as f32 * 0.02);
                let draws = [
                    mk(ground, Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2) * Mat4::from_scale(Vec3::splat(30.0)), 1.5),
                    mk(cube, Mat4::from_translation(Vec3::new(0.0, 0.5, 0.0)) * spin, 0.9),
                ];
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
        frames(&mut r, 30, 8);
        if r.upscaler() != Upscaler::Fsr3 {
            let reason = r.upscaler_error().unwrap_or("no reason recorded").to_owned();
            if reason.is_empty() || r.upscale_quality() != UpscaleQuality::Native {
                self.error = Some(format!("the upscaler is off without a recorded reason or a clean fallback: {reason:?}"));
                return;
            }
            frames(&mut r, 6, 38);
            eprintln!("[upscaler path] FSR3 UNAVAILABLE on this device, the renderer fell back to native and kept rendering: {reason}");
            return;
        }
        eprintln!("[upscaler path] 30 frames at {:?} with FSR3 upscaling, {:.3} ms GPU", r.upscale_quality(), r.stats().gpu_frame_ms);
        let still = |r: &mut Renderer, count: u32| {
            for _ in 0..count {
                frames(r, 1, 1000);
            }
        };
        let grab = |r: &mut Renderer| -> Option<(u32, u32, Vec<u8>)> {
            r.debug_capture_viewport_frames(true);
            frames(r, 1, 1000);
            let shot = r.debug_take_swapchain_frames().ok().and_then(|mut shots| shots.pop());
            r.debug_capture_viewport_frames(false);
            shot
        };
        still(&mut r, 40);
        let upscaled = grab(&mut r);
        r.set_upscale_quality(UpscaleQuality::Native);
        still(&mut r, 12);
        let native = grab(&mut r);
        r.set_upscale_quality(UpscaleQuality::Quality);
        still(&mut r, 4);
        match (native, upscaled) {
            (Some((nw, nh, native)), Some((uw, uh, upscaled))) if (nw, nh) == (uw, uh) && !native.is_empty() => {
                let pixels = (nw * nh) as f64;
                let mean = |px: &[u8]| px.as_chunks::<4>().0.iter().map(|p| p[0] as f64 + p[1] as f64 + p[2] as f64).sum::<f64>() / (pixels * 3.0);
                let difference = native.as_chunks::<4>().0.iter().zip(upscaled.as_chunks::<4>().0.iter()).map(|(a, b)| (0..3).map(|k| (a[k] as f64 - b[k] as f64).abs()).sum::<f64>()).sum::<f64>() / (pixels * 3.0);
                let far_off = native.as_chunks::<4>().0.iter().zip(upscaled.as_chunks::<4>().0.iter()).filter(|(a, b)| (0..3).any(|k| (a[k] as i32 - b[k] as i32).abs() > 40)).count() as f64 / pixels;
                eprintln!("[upscaler path] {nw}x{nh}: native mean {:.1}, upscaled mean {:.1}, mean difference {difference:.2} levels, {:.2}% of pixels off by more than 40", mean(&native), mean(&upscaled), far_off * 100.0);
                if let Some(dir) = std::env::var_os("ZESTENGINE_TEST_DUMP_DIR") {
                    for (name, px) in [("native", &native), ("upscaled", &upscaled)] {
                        let mut ppm = format!("P6\n{nw} {nh}\n255\n").into_bytes();
                        ppm.extend(px.as_chunks::<4>().0.iter().flat_map(|p| [p[0], p[1], p[2]]));
                        let _ = std::fs::write(std::path::Path::new(&dir).join(format!("upscaler_{name}.ppm")), ppm);
                    }
                }
                if mean(&upscaled) < 10.0 || difference > 6.0 || far_off > 0.03 {
                    self.error = Some(format!("the upscaled frame does not match the native one: mean difference {difference:.2}, {:.2}% far off, upscaled mean {:.1}", far_off * 100.0, mean(&upscaled)));
                    return;
                }
            }
            (native, upscaled) => eprintln!(
                "[upscaler path] no frame capture on this swapchain (copy allowed {}, native {:?}, upscaled {:?}): the picture was not compared",
                r.swapchain.can_copy_from,
                native.map(|(w, h, px)| (w, h, px.len())),
                upscaled.map(|(w, h, px)| (w, h, px.len()))
            ),
        }
        r.set_upscale_quality(UpscaleQuality::Performance);
        frames(&mut r, 12, 1100);
        r.set_upscale_quality(UpscaleQuality::Native);
        frames(&mut r, 6, 1200);
        if r.upscaler() != Upscaler::Off {
            self.error = Some("Native did not switch the upscaler off".into());
        }
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
fn the_game_path_renders_through_the_upscaler_and_back_to_native() {
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
