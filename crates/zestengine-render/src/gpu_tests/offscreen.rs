//! The offscreen model renders on a REAL `Renderer`: the MODEL-ONLY preview (own draws,
//! neutral light, background colour, depth readback, no scene instance needed) and the
//! textured impostor capture (atlas layout, textures reaching the albedo layer, normals,
//! depth, determinism, queueing).
//!
//! Needs a display (X server / Xwayland); skipped without one or with `ZESTENGINE_TEST_GPU=0`.

use std::sync::Arc;

use glam::{Mat4, Quat, Vec3};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::{Window, WindowId};

use super::planar::{cube, draw, lit};
use crate::offscreen::{cell_dir, neutral_light};
use crate::{CameraData, CapturedImpostor, ImpostorCaptureRequest, MaterialDesc, MaterialParams, ModelOnlyPreview, PostFx, Renderer, TextureData};

const SIZE: [u32; 2] = [256, 256];
const GRID: u32 = 8;
const TILE: u32 = 32;

#[derive(Default)]
struct Probe {
    done: bool,
    error: Option<String>,
    // model-only
    preview_pixels: Vec<u8>,
    preview_size: [u32; 2],
    depth_centre: f32,
    depth_corner: f32,
    hidden_behind: bool,
    visible_front: bool,
    no_depth_without_request: bool,
    // capture
    first: Option<CapturedImpostor>,
    second: Option<CapturedImpostor>,
    queued_both: bool,
    bad_request_rejected: bool,
}

fn checker() -> TextureData {
    // Left half red, right half blue.
    let px: Vec<u8> = (0..8 * 8).flat_map(|i| if i % 8 < 4 { [255u8, 0, 0, 255] } else { [0, 0, 255, 255] }).collect();
    TextureData { width: 8, height: 8, pixels: px, srgb: false, hdr: false }
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
        if let Err(e) = self.go(&mut r) {
            self.error = Some(format!("{e:#}"));
        }
    }

    fn go(&mut self, r: &mut Renderer) -> anyhow::Result<()> {
        // The test cube has no uvs: u runs 0..1 across each face along its first axis.
        let mut cube_data = cube();
        for face in cube_data.vertices.chunks_mut(4) {
            for (v, uv) in face.iter_mut().zip([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]) {
                v.uv = uv;
            }
        }
        let mesh = r.upload_mesh(&cube_data)?;
        let bright = lit(r, [0.9, 0.8, 0.2, 1.0], [0.0; 3], 0.9, 0.0);
        let tex = r.upload_texture(&checker())?;
        let textured = r.create_material(&MaterialDesc { name: "checker".into(), params: MaterialParams { base_color: [1.0; 4], ..Default::default() }, albedo: Some(tex), ..Default::default() })?;
        let frame = |r: &mut Renderer| -> anyhow::Result<()> {
            let mut input = crate::cascade_tests::input(CameraData { view: Mat4::IDENTITY, proj: Mat4::IDENTITY, position: Vec3::ZERO });
            input.render_viewport = false;
            r.render(&input)
        };

        // ---- model-only preview: the scene has no draws at all.
        let eye = Vec3::new(0.0, 0.0, 4.0);
        let view = Mat4::look_at_rh(eye, Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(0.9, 1.0, 0.1, 50.0);
        let camera = CameraData { view, proj, position: eye };
        let draws = [draw(mesh, bright, Mat4::from_scale_rotation_translation(Vec3::splat(1.0), Quat::IDENTITY, Vec3::ZERO), -1)];
        let postfx = PostFx { tonemap: 0, bloom_enabled: false, grading_enabled: false, vignette_enabled: false, ca_enabled: false, ..PostFx::default() };
        let want = |depth: bool| ModelOnlyPreview { camera, size: SIZE, draws: &draws, background: [0.2, 0.0, 0.3, 1.0], light: neutral_light(), postfx, want_depth: depth };
        for _ in 0..4 {
            r.set_model_only_preview(Some(&want(false)));
            frame(r)?;
        }
        self.no_depth_without_request = r.model_preview_depth().is_none();
        for _ in 0..6 {
            r.set_model_only_preview(Some(&want(true)));
            frame(r)?;
        }
        let (w, h, px) = r.debug_read_model_preview()?.ok_or_else(|| anyhow::anyhow!("no model preview target"))?;
        self.preview_size = [w, h];
        self.preview_pixels = px;
        if let Some(d) = r.model_preview_depth() {
            self.depth_centre = d.at(d.size[0] as i32 / 2, d.size[1] as i32 / 2).unwrap_or(-1.0);
            self.depth_corner = d.at(2, 2).unwrap_or(-1.0);
            self.hidden_behind = d.occludes(&camera, Vec3::new(0.0, 0.0, -0.4), 1e-5);
            self.visible_front = !d.occludes(&camera, Vec3::new(0.0, 0.0, 1.5), 1e-5);
        } else {
            anyhow::bail!("no depth readback arrived");
        }

        // ---- impostor capture: a textured cube (+Z face carries the red/blue halves) and a flat one.
        let req = |draws: Vec<crate::DrawCmd>| ImpostorCaptureRequest { grid: GRID, tile: TILE, centre: [0.0; 3], radius: 0.9, draws };
        let cap_draws = vec![draw(mesh, textured, Mat4::IDENTITY, -1)];
        self.bad_request_rejected = r.start_impostor_capture(ImpostorCaptureRequest { grid: 0, ..req(cap_draws.clone()) }).is_err()
            && r.start_impostor_capture(ImpostorCaptureRequest { radius: -1.0, ..req(cap_draws.clone()) }).is_err()
            && r.start_impostor_capture(req(vec![])).is_err();
        let a = r.start_impostor_capture(req(cap_draws.clone()))?;
        let b = r.start_impostor_capture(req(cap_draws.clone()))?;
        self.queued_both = r.impostor_captures_pending() == 2;
        let (mut ra, mut rb) = (None, None);
        for _ in 0..60 {
            frame(r)?;
            ra = ra.or_else(|| r.take_impostor_capture(a));
            rb = rb.or_else(|| r.take_impostor_capture(b));
            if ra.is_some() && rb.is_some() {
                break;
            }
        }
        self.first = Some(ra.ok_or_else(|| anyhow::anyhow!("capture A never finished"))?.map_err(anyhow::Error::msg)?);
        self.second = Some(rb.ok_or_else(|| anyhow::anyhow!("capture B never finished"))?.map_err(anyhow::Error::msg)?);
        Ok(())
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

/// sRGB-encode one linear channel to a byte (the preview target is an sRGB format).
fn srgb8(v: f32) -> i32 {
    let e = if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    (e.clamp(0.0, 1.0) * 255.0 + 0.5) as i32
}

#[test]
fn model_only_preview_and_textured_impostor_capture_on_a_real_renderer() {
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

    // ---- model-only preview
    assert_eq!(p.preview_size, SIZE);
    let at = |x: u32, y: u32| {
        let i = ((y * SIZE[0] + x) * 4) as usize;
        [p.preview_pixels[i] as i32, p.preview_pixels[i + 1] as i32, p.preview_pixels[i + 2] as i32]
    };
    let bg = at(3, 3);
    let want_bg = [srgb8(0.2), srgb8(0.0), srgb8(0.3)];
    assert!(bg.iter().zip(&want_bg).all(|(a, b)| (a - b).abs() <= 3), "background colour {bg:?}, wanted {want_bg:?}");
    let model = at(SIZE[0] / 2, SIZE[1] / 2);
    assert!(model[0] > 60 && model[0] > model[2], "the model is not drawn in its own colour: {model:?}");
    assert!(p.no_depth_without_request, "a depth readback appeared without being asked for");
    assert!(p.depth_centre > 0.0 && p.depth_centre < 1.0, "centre depth {}", p.depth_centre);
    assert_eq!(p.depth_corner, 1.0, "empty background must read depth 1");
    assert!(p.hidden_behind && p.visible_front, "occlusion query: behind {} in front {}", p.hidden_behind, p.visible_front);
    eprintln!("model-only preview: bg {bg:?} model {model:?} depth centre {:.4} corner {}", p.depth_centre, p.depth_corner);

    // ---- impostor capture
    assert!(p.bad_request_rejected, "invalid capture requests must be rejected");
    assert!(p.queued_both, "both captures should have been queued");
    let (a, b) = (p.first.as_ref().unwrap(), p.second.as_ref().unwrap());
    let size = (GRID * TILE) as usize;
    assert_eq!(a.albedo.len(), size * size * 4);
    assert_eq!(a.aux.len(), size * size * 4);
    assert_eq!(a, b, "the same capture twice must be bit-identical");
    let texel = |layer: &[u8], cell: (u32, u32), x: u32, y: u32| {
        let at = (((cell.1 * TILE + y) as usize) * size + (cell.0 * TILE + x) as usize) * 4;
        [layer[at], layer[at + 1], layer[at + 2], layer[at + 3]]
    };
    let mid = TILE / 2;
    // Every view sees the cube at its centre and nothing in its corners.
    for j in 0..GRID {
        for i in 0..GRID {
            assert_eq!(texel(&a.albedo, (i, j), mid, mid)[3], 255, "cell ({i},{j}) misses the cube");
            assert_eq!(texel(&a.albedo, (i, j), 0, 0), [0, 0, 0, 0], "cell ({i},{j}) corner is covered");
            assert_eq!(texel(&a.aux, (i, j), mid, mid)[3], 255);
        }
    }
    // A view that looks at the +Z face: the textured halves reach the atlas (the cube's +Z face is
    // left red / right blue as seen from +Z, under the engine's uv convention at least one of the
    // two colours dominates each side and the two sides differ).
    let (mut best, mut best_dot) = ((0, 0), -2.0f32);
    for j in 0..GRID {
        for i in 0..GRID {
            let d = Vec3::from(cell_dir(GRID, i, j)).dot(Vec3::Z);
            if d > best_dot {
                (best, best_dot) = ((i, j), d);
            }
        }
    }
    let (left, right) = (texel(&a.albedo, best, mid - 6, mid), texel(&a.albedo, best, mid + 6, mid));
    assert!(left[0] > 150 && left[2] < 60 && right[2] > 150 && right[0] < 60 || left[2] > 150 && left[0] < 60 && right[0] > 150 && right[2] < 60, "texture halves {left:?} {right:?}");
    // The face's world normal (+Z): decode the aux layer's octahedral normal at the centre.
    let n = texel(&a.aux, best, mid, mid);
    let o = [n[0] as f32 / 255.0 * 2.0 - 1.0, n[1] as f32 / 255.0 * 2.0 - 1.0];
    let n = Vec3::from(crate::offscreen::oct_to_dir(o));
    assert!(n.dot(Vec3::Z) > 0.97, "normal {n:?} is not the +Z face's");
    // Depth: the nearest surface is nearer than the cube's centre plane (aux.b > 0.5).
    assert!(texel(&a.aux, best, mid, mid)[2] > 128, "depth {:?}", texel(&a.aux, best, mid, mid));
    eprintln!("impostor capture: {size}x{size} atlas, texture halves {left:?} / {right:?}, +Z normal dot {:.3}", n.dot(Vec3::Z));
}
