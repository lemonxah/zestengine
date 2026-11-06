//! Regression guard for the per-pixel cost of the default material on the real
//! `Renderer`: a large plane with the stock PBR material and one sun must cost a
//! fraction of a millisecond in the SCENE GPU zone, not tens of milliseconds.
//!
//! The disease it guards against: `area_light.glsl` kept the 576-entry LTC matrix
//! table as a dynamically indexed `const vec4[]`. The compiler turns that into a
//! per-invocation local array initialised at shader entry whether or not an area
//! light exists, so every pixel of every material paid for a 9 KB copy (an empty
//! scene's ground plane went from 0.2 ms to 20 ms at editor size). The table now
//! lives in a texture (set 1 binding 23).
//!
//! Needs a display (an X server, Xwayland counts); skipped without one or with
//! `ZESTENGINE_TEST_GPU=0`. The window is created invisible and never shown.

use std::sync::Arc;

use glam::{Mat4, Vec3};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::{Window, WindowId};

use super::shadow::vert;
use crate::{CameraData, DrawCmd, LightInstance, LightKind, MaterialDesc, MeshData, Renderer, ShadowCasting};

/// Generous: a healthy frame is well under 1 ms here, the broken shader took ~25 ms at 1280x720.
const SCENE_MS_BOUND: f32 = 3.0;

#[derive(Default)]
struct Probe {
    scene_ms: Vec<f32>,
    extent: (u32, u32),
    error: Option<String>,
    done: bool,
}

struct App {
    probe: Probe,
}

impl App {
    fn run(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes().with_visible(false).with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                self.probe.error = Some(format!("window: {e}"));
                return;
            }
        };
        if let Err(e) = self.frames(window) {
            self.probe.error = Some(format!("{e:#}"));
        }
    }

    fn frames(&mut self, window: Arc<Window>) -> anyhow::Result<()> {
        let size = window.inner_size();
        self.probe.extent = (size.width, size.height);
        let mut r = Renderer::new(window)?;
        r.set_vsync(false);
        r.set_gpu_zone_timing(true);
        // A 40 x 40 m ground plane (the editor's Plane primitive scaled up), normal +Y.
        let h = 20.0;
        let n = Vec3::Y;
        let mesh = r.upload_mesh(&MeshData {
            vertices: vec![vert(Vec3::new(-h, 0.0, -h), n), vert(Vec3::new(h, 0.0, -h), n), vert(Vec3::new(h, 0.0, h), n), vert(Vec3::new(-h, 0.0, h), n)],
            indices: vec![0, 2, 1, 0, 3, 2],
            has_lightmap_uv: false,
        })?;
        // The stock material: standard shader, every feature off, no textures.
        let material = r.create_material(&MaterialDesc { name: "default".into(), ..Default::default() })?;
        let eye = Vec3::new(0.0, 3.0, 6.0);
        let view = Mat4::look_at_rh(eye, Vec3::new(0.0, 0.0, -4.0), Vec3::Y);
        let aspect = size.width.max(1) as f32 / size.height.max(1) as f32;
        let camera = CameraData { view, proj: Mat4::perspective_rh(1.2, aspect, 0.1, 200.0), position: eye };
        let draws = [DrawCmd {
            gi_source: None,
            mesh,
            material,
            transform: Mat4::IDENTITY,
            highlight: 0.0,
            mesh_center: Vec3::ZERO,
            bound_radius: 0.0,
            lightmap_layer: -1,
            lightmap_size: 0,
            static_geometry: false,
            planar_index: -1,
            occluded: false,
            shadow_casting: ShadowCasting::Off,
            instance_color: None,
            instance_param: None,
        }];
        let sun = [LightInstance {
            kind: LightKind::Directional,
            position: Vec3::ZERO,
            direction: Vec3::new(-0.3, -1.0, -0.2).normalize(),
            color: [1.0; 3],
            intensity: 3.0,
            range: 100.0,
            spot_inner_deg: 0.0,
            spot_outer_deg: 0.0,
            cast_shadows: false,
            soft_shadows: false, radius: 0.0,
            shadow_bias: 0.0,
            baked: false,
        }];
        let mut input = crate::cascade_tests::input(camera);
        input.render_viewport = true;
        input.draws = &draws;
        input.lights = &sun;
        for _ in 0..14 {
            r.render(&input)?;
            self.probe.scene_ms.push(r.stats().gpu_scene_ms);
        }
        Ok(())
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
fn default_material_ground_plane_stays_cheap_per_pixel() {
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
    eprintln!("{}x{} default-material ground plane, SCENE zone ms per frame: {:?}", p.extent.0, p.extent.1, p.scene_ms);
    // Timestamps are read back a frame slot later: skip the warm-up frames (idle GPU clocks, pipeline
    // creation) and judge the best of the rest, which other GPU users can only inflate.
    let best = p.scene_ms[6..].iter().copied().fold(f32::INFINITY, f32::min);
    assert!(best > 0.0, "the SCENE zone was never stamped: {:?}", p.scene_ms);
    // Scale the bound with the window so a smaller surface cannot hide the regression.
    let scale = ((p.extent.0 * p.extent.1) as f32 / (1280.0 * 720.0)).max(0.05);
    assert!(
        best < SCENE_MS_BOUND * scale.max(1.0),
        "default material plane costs {best:.2} ms in the SCENE zone at {}x{} (bound {SCENE_MS_BOUND} ms): \
         a per-pixel cost regression in standard.frag (dynamically indexed constant arrays, new unconditional work)",
        p.extent.0, p.extent.1
    );
}
