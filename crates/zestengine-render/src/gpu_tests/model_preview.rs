//! The model preview target (`FrameInput::model_preview`) on a REAL `Renderer`:
//! it draws only the listed draws with its own camera into its own texture,
//! the target is created on demand, re-created on a size change, and freed a
//! few frames after the input goes away.
//!
//! Needs a display (X server / Xwayland); skipped without one or with
//! `ZESTENGINE_TEST_GPU=0`.

use std::sync::Arc;

use glam::{Mat4, Quat, Vec3, Vec4};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::{Window, WindowId};

use super::planar::{cube, draw, lit};
use crate::{CameraData, ModelPreviewInput, Renderer};

const SIZE: [u32; 2] = [384, 512];

#[derive(Default)]
struct Probe {
    done: bool,
    error: Option<String>,
    pixels: Vec<u8>,
    size: [u32; 2],
    /// Screen pixel of the "model" cube centre and of the distractor cube centre.
    model_px: (i32, i32),
    other_px: (i32, i32),
    /// (exists after render, exists after the input went away, exists after a resize).
    lifetime: (bool, bool, [u32; 2]),
}

impl Probe {
    fn run(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes().with_visible(false).with_inner_size(winit::dpi::LogicalSize::new(320.0, 240.0));
        let Ok(window) = event_loop.create_window(attrs).map(Arc::new) else {
            self.error = Some("window".into());
            return;
        };
        let mut r = match Renderer::new(window) {
            Ok(r) => r,
            Err(e) => {
                self.error = Some(format!("Renderer::new: {e:#}"));
                return;
            }
        };
        r.set_vsync(false);
        let mesh = r.upload_mesh(&cube()).unwrap();
        let bright = lit(&mut r, [0.9, 0.8, 0.2, 1.0], [0.0; 3], 0.9, 0.0);
        let draws = vec![
            draw(mesh, bright, Mat4::from_scale_rotation_translation(Vec3::splat(1.0), Quat::IDENTITY, Vec3::ZERO), -1),
            draw(mesh, bright, Mat4::from_scale_rotation_translation(Vec3::splat(0.8), Quat::IDENTITY, Vec3::new(0.0, 2.2, 0.0)), -1),
        ];
        let eye = Vec3::new(0.0, 0.0, 6.0);
        let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(1.0, SIZE[0] as f32 / SIZE[1] as f32, 0.1, 50.0);
        let camera = CameraData { view, proj, position: eye };
        let only_first = [0usize];
        let to_px = |p: Vec3| -> (i32, i32) {
            let c = proj * view * Vec4::new(p.x, p.y, p.z, 1.0);
            (((c.x / c.w * 0.5 + 0.5) * SIZE[0] as f32) as i32, ((0.5 - c.y / c.w * 0.5) * SIZE[1] as f32) as i32)
        };
        self.model_px = to_px(Vec3::ZERO);
        self.other_px = to_px(Vec3::new(0.0, 2.2, 0.0));
        let frame = |r: &mut Renderer, preview: Option<ModelPreviewInput>| -> anyhow::Result<()> {
            let mut input = crate::cascade_tests::input(CameraData { view: Mat4::IDENTITY, proj, position: Vec3::ZERO });
            input.render_viewport = false;
            input.light.ambient = [0.8, 0.8, 0.8];
            input.draws = &draws;
            input.model_preview = preview;
            r.render(&input)
        };
        let want = ModelPreviewInput { camera, size: SIZE, draws: &only_first };
        for _ in 0..4 {
            if let Err(e) = frame(&mut r, Some(want)) {
                self.error = Some(format!("render: {e:#}"));
                return;
            }
        }
        match r.debug_read_model_preview() {
            Ok(Some((w, h, px))) => {
                self.size = [w, h];
                self.pixels = px;
            }
            Ok(None) => {
                self.error = Some("no model preview target".into());
                return;
            }
            Err(e) => {
                self.error = Some(format!("readback: {e:#}"));
                return;
            }
        }
        self.lifetime.0 = r.model_preview_texture().is_some();
        // A resize re-creates the target.
        let bigger = ModelPreviewInput { size: [SIZE[0] + 64, SIZE[1] + 64], ..want };
        for _ in 0..4 {
            if let Err(e) = frame(&mut r, Some(bigger)) {
                self.error = Some(format!("resized render: {e:#}"));
                return;
            }
        }
        self.lifetime.2 = r.model_preview_texture().map(|(_, s)| [s[0] as u32, s[1] as u32]).unwrap_or([0; 2]);
        // Closing the window: the target goes away, rendering carries on.
        for _ in 0..6 {
            if let Err(e) = frame(&mut r, None) {
                self.error = Some(format!("closed render: {e:#}"));
                return;
            }
        }
        self.lifetime.1 = r.model_preview_texture().is_some();
    }
}

impl ApplicationHandler for Probe {
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
fn the_model_preview_draws_only_its_draws_and_frees_itself() {
    if std::env::var("ZESTENGINE_TEST_GPU").is_ok_and(|v| v == "0") || std::env::var_os("DISPLAY").is_none() {
        eprintln!("[gpu test] no display / GPU disabled: skipped");
        return;
    }
    let _gpu = crate::cluster_headless::gpu_lock();
    let Ok(mut event_loop) = EventLoop::builder().with_x11().with_any_thread(true).build() else {
        eprintln!("[gpu test] no X11 event loop: skipped");
        return;
    };
    let mut p = Probe::default();
    event_loop.run_app_on_demand(&mut p).unwrap();
    if let Some(e) = &p.error {
        panic!("{e}");
    }
    assert_eq!(p.size, SIZE);
    let at = |x: i32, y: i32| {
        let i = (y.clamp(0, SIZE[1] as i32 - 1) as usize * SIZE[0] as usize + x.clamp(0, SIZE[0] as i32 - 1) as usize) * 4;
        [p.pixels[i] as i32, p.pixels[i + 1] as i32, p.pixels[i + 2] as i32]
    };
    let background = at(4, 4);
    let diff = |a: [i32; 3], b: [i32; 3]| a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum::<i32>();
    let model = at(p.model_px.0, p.model_px.1);
    assert!(diff(model, background) > 120, "the model cube is not drawn: {model:?} vs background {background:?}");
    let other = at(p.other_px.0, p.other_px.1);
    assert!(diff(other, background) < 12, "a draw outside the list was drawn: {other:?} vs {background:?}");
    assert!(p.lifetime.0, "the target exists while requested");
    assert_eq!(p.lifetime.2, [SIZE[0] + 64, SIZE[1] + 64], "re-created at the new size");
    assert!(!p.lifetime.1, "the target is released when the input goes away");
}
