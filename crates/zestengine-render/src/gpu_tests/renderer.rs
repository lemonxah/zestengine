//! A whole `Renderer` on the real device: creation with the Vulkan 1.2/1.3
//! feature chain and present-mode extension (capability report), then a few
//! frames into a never-mapped X11 window to exercise the swapchain path, the
//! always-on FRAME timestamp pair and its one-frame readback delay.
//!
//! Needs a display (an X server, Xwayland counts); skipped without one or with
//! `ZESTENGINE_TEST_GPU=0`. The window is created invisible and never shown.

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::{Window, WindowId};

use crate::{CameraData, Renderer};

#[derive(Default)]
struct Probe {
    report: String,
    frame_ms: Vec<f32>,
    error: Option<String>,
    done: bool,
    xr: String,
}

struct App {
    probe: Probe,
}

impl App {
    fn run(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes().with_visible(false).with_inner_size(winit::dpi::LogicalSize::new(640.0, 360.0));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                self.probe.error = Some(format!("window: {e}"));
                return;
            }
        };
        let mut r = match Renderer::new(window) {
            Ok(r) => r,
            Err(e) => {
                self.probe.error = Some(format!("Renderer::new: {e:#}"));
                return;
            }
        };
        self.probe.report = r.gpu_caps().report();
        r.set_vsync(false);
        let eye = glam::Vec3::new(0.0, 2.0, 5.0);
        let view = glam::Mat4::look_at_rh(eye, glam::Vec3::ZERO, glam::Vec3::Y);
        let camera = CameraData { view, proj: glam::Mat4::perspective_rh(1.0, 16.0 / 9.0, 0.1, 100.0), position: eye };
        let mut input = crate::cascade_tests::input(camera);
        input.render_viewport = false;
        // A GPU-skinned, morphed, cloth-patched mesh: the skin / morph / cloth passes run
        // inside the real frame (descriptor sets per frame slot, barriers, timestamp zones).
        let (verts, idx) = super::skin::skinned_tube_for_renderer();
        let data = crate::MeshData { vertices: verts, indices: idx, has_lightmap_uv: false };
        let shape: Vec<(u32, [f32; 3], [f32; 3])> = (0..data.vertices.len() as u32).step_by(2).map(|v| (v, [0.0, 0.02, 0.0], [0.0; 3])).collect();
        let mesh = match r.upload_mesh_gpu_skinned_morphed(&data, 3, &[shape.as_slice()]) {
            Ok(m) => m,
            Err(e) => {
                self.probe.error = Some(format!("skinned upload: {e:#}"));
                return;
            }
        };
        let mut packed = Vec::new();
        for i in 0..8 {
            let t = i as f32 * 0.4;
            r.skin_mesh_gpu(mesh, &[glam::Mat4::from_rotation_z(t), glam::Mat4::from_rotation_x(-t), glam::Mat4::IDENTITY]);
            r.set_mesh_morph_weights(mesh, &[(i % 3) as f32 * 0.5]);
            let mut moved = data.vertices.clone();
            for v in moved.iter_mut().step_by(5) {
                v.position[1] -= 0.1;
            }
            let n = crate::pack_cloth_patch(&data.vertices, &moved, 1e-5, &mut packed);
            r.set_mesh_cloth_patch(mesh, &packed, if i % 2 == 0 { n } else { 0 });
            if let Err(e) = r.render(&input) {
                self.probe.error = Some(format!("render: {e:#}"));
                return;
            }
            self.probe.frame_ms.push(r.stats().gpu_frame_ms);
        }
        match super::xr_cluster::run(&mut r) {
            Ok(report) => self.probe.xr = report,
            Err(e) => self.probe.error = Some(format!("xr cluster: {e:#}")),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if !self.probe.done {
            self.probe.done = true;
            self.run(event_loop);
            event_loop.exit();
        }
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
}

#[test]
fn renderer_creates_headless_and_reports_gpu_frame_time() {
    if std::env::var("ZESTENGINE_TEST_GPU").is_ok_and(|v| v == "0") || std::env::var_os("DISPLAY").is_none() {
        eprintln!("[gpu test] no display / GPU disabled: skipped");
        return;
    }
    let _gpu = crate::cluster_headless::gpu_lock();
    let Ok(mut event_loop) = EventLoop::builder().with_x11().with_any_thread(true).build() else {
        eprintln!("[gpu test] no X11 event loop: skipped");
        return;
    };
    let mut app = App { probe: Probe::default() };
    event_loop.run_app_on_demand(&mut app).unwrap();
    let p = app.probe;
    if let Some(e) = &p.error {
        panic!("{e}");
    }
    eprintln!("{}", p.report);
    eprintln!("{}", p.xr);
    eprintln!("gpu_frame_ms per frame: {:?}", p.frame_ms);
    // FRAMES_IN_FLIGHT = 2: a slot's timestamps are read back the next time that slot
    // is used, so the first two frames report 0 and later ones a real FRAME zone.
    assert_eq!(p.frame_ms[0], 0.0);
    assert!(p.frame_ms[4..].iter().all(|&m| m > 0.0 && m < 1000.0), "implausible FRAME times {:?}", p.frame_ms);
}
