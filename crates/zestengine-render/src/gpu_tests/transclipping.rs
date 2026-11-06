//! Transclipping alpha mode on a REAL `Renderer`: alpha test and blending
//! together, the midpoint between Cutout and Transparent. A cube whose base
//! colour alpha is below the cutoff is discarded (blending on, nothing to
//! blend); one above it shows BLENDED against the background, which is what
//! separates transclipping from plain cutout (a pure colour write).
//!
//! Needs a display; skipped without one or with `ZESTENGINE_TEST_GPU=0`.

use std::sync::Arc;

use glam::{Mat4, Quat, Vec3};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::{Window, WindowId};

use super::planar::{cube, draw, lit};
use crate::{AlphaMode, CameraData, MaterialDesc, MaterialFeatures, MaterialHandle, MaterialParams, Renderer, ShadowCasting};

const W: u32 = 320;
const H: u32 = 240;
/// Base colour of the test cube; far from the wall's, with alpha above the cutoff.
const CUBE: [f32; 4] = [0.9, 0.2, 0.1, 0.8];

fn wall() -> crate::MeshData {
    let mut v = Vec::new();
    let mut i = Vec::new();
    for y in 0..2 {
        for x in 0..2 {
            v.push(crate::Vertex {
                position: [x as f32 - 0.5, y as f32 - 0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [x as f32, y as f32],
                ..Default::default()
            });
        }
    }
    i.extend_from_slice(&[0, 1, 2, 1, 3, 2]);
    crate::MeshData { vertices: v, indices: i, has_lightmap_uv: false }
}

fn camera() -> CameraData {
    let eye = Vec3::new(0.0, 1.5, 6.0);
    let view = Mat4::look_at_rh(eye, Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
    CameraData { view, proj: Mat4::perspective_rh(0.9, W as f32 / H as f32, 0.1, 100.0), position: eye }
}

type Frames = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);

fn run(r: Renderer) -> Frames {
    let mut r = r;
    let cube_mesh = r.upload_mesh(&cube()).unwrap();
    let wall = r.upload_mesh(&wall()).unwrap();
    let wall_mat = lit(&mut r, [0.3, 0.8, 0.3, 1.0], [0.0; 3], 0.8, 0.0);
    let clipped = r
        .create_material(&MaterialDesc {
            name: "clipped".into(),
            params: MaterialParams { base_color: [CUBE[0], CUBE[1], CUBE[2], 0.1], roughness: 0.6, ..Default::default() },
            features: MaterialFeatures { alpha_mode: AlphaMode::Transclipping, ..Default::default() },
            ..Default::default()
        })
        .unwrap();
    let blended = r
        .create_material(&MaterialDesc {
            name: "blended".into(),
            params: MaterialParams { base_color: CUBE, roughness: 0.6, ..Default::default() },
            features: MaterialFeatures { alpha_mode: AlphaMode::Transclipping, ..Default::default() },
            ..Default::default()
        })
        .unwrap();
    let cutout = r
        .create_material(&MaterialDesc {
            name: "cutout".into(),
            params: MaterialParams { base_color: CUBE, roughness: 0.6, ..Default::default() },
            features: MaterialFeatures { alpha_mode: AlphaMode::Cutout, ..Default::default() },
            ..Default::default()
        })
        .unwrap();
    let plain_blend = r
        .create_material(&MaterialDesc {
            name: "plain blend".into(),
            params: MaterialParams { base_color: CUBE, roughness: 0.6, ..Default::default() },
            features: MaterialFeatures { alpha_mode: AlphaMode::Blend, ..Default::default() },
            ..Default::default()
        })
        .unwrap();

    let at = Mat4::from_scale_rotation_translation(Vec3::splat(1.4), Quat::from_rotation_y(0.4), Vec3::new(0.0, 1.0, 0.0));
    let wall_draw = draw(wall, wall_mat, Mat4::from_translation(Vec3::new(0.0, 1.0, -2.0)) * Mat4::from_scale(Vec3::splat(4.0)), 0);
    let render = |r: &mut Renderer, cube_mat: MaterialHandle| {
        let mut draws = vec![wall_draw, draw(cube_mesh, cube_mat, at, -1)];
        for d in &mut draws {
            d.shadow_casting = ShadowCasting::Off;
            d.bound_radius = 3.0;
        }
        let mut px = Vec::new();
        for _ in 0..3 {
            let mut input = crate::cascade_tests::input(camera());
            input.draws = &draws;
            input.draw_skybox = false;
            input.viewport_extent = Some([W, H]);
            r.render(&input).unwrap_or_else(|e| panic!("render: {e:#}"));
            let (w, h, p) = r.debug_read_viewport().expect("readback").expect("viewport target");
            assert_eq!((w, h), (W, H));
            px = p;
        }
        px
    };
    let background = {
        let mut draws = vec![wall_draw];
        for d in &mut draws {
            d.shadow_casting = ShadowCasting::Off;
        }
        let mut px = Vec::new();
        for _ in 0..3 {
            let mut input = crate::cascade_tests::input(camera());
            input.draws = &draws;
            input.draw_skybox = false;
            input.viewport_extent = Some([W, H]);
            r.render(&input).unwrap();
            let (_, _, p) = r.debug_read_viewport().expect("readback").expect("viewport target");
            px = p;
        }
        px
    };
    let clip_img = render(&mut r, clipped);
    let blend_img = render(&mut r, blended);
    let cut_img = render(&mut r, cutout);
    let plain_img = render(&mut r, plain_blend);
    let center = |img: &[u8]| [img[((H / 2) as usize * W as usize + W as usize / 2) * 4], img[((H / 2) as usize * W as usize + W as usize / 2) * 4 + 1], img[((H / 2) as usize * W as usize + W as usize / 2) * 4 + 2]];
    eprintln!("[transclipping] center px: background {:?} clip {:?} blend {:?} cutout {:?} plain-blend {:?}", center(&background), center(&clip_img), center(&blend_img), center(&cut_img), center(&plain_img));
    (background, clip_img, blend_img, cut_img, plain_img)
}

#[derive(Default)]
struct App {
    done: bool,
    error: Option<String>,
    images: Option<Frames>,
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
                Ok(r) => {
                    let mut r = r;
                    r.set_vsync(false);
                    self.images = Some(run(r));
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
fn transclipping_tests_then_blends() {
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
    let (background, clip_img, blend_img, cut_img, plain_img) = app.images.unwrap();

    let diff = |a: &[u8], b: &[u8]| -> (usize, i32) {
        let mut changed = 0;
        let mut max = 0;
        for (i, p) in a.as_chunks::<4>().0.iter().enumerate() {
            let d = (0..3).map(|c| (p[c] as i32 - b[i * 4 + c] as i32).abs()).max().unwrap();
            max = max.max(d);
            if d > 4 {
                changed += 1;
            }
        }
        (changed, max)
    };

    // Below the cutoff the whole cube is discarded: the image matches the wall alone.
    let (clip_diff, clip_max) = diff(&background, &clip_img);
    assert_eq!(clip_diff, 0, "the below-cutoff cube must be fully discarded ({} px differ, max {clip_max})", clip_diff);

    // Above the cutoff the cube shows...
    let (blend_diff, _) = diff(&background, &blend_img);
    assert!(blend_diff > 500, "the above-cutoff cube must be visible ({blend_diff} px differ from the wall)", );

    // ...and it BLENDS: a pure cutout of the same material writes the pure
    // colour, so the two images must differ where the cube is.
    let (cut_vs_blend, _) = diff(&cut_img, &blend_img);
    let (plain_vs_blend, _) = diff(&plain_img, &blend_img);
    assert!(cut_vs_blend > 500, "transclipping must blend with the background like transparent ({cut_vs_blend} px differ from cutout)");
    assert!(plain_vs_blend < 20, "transclipping must blend exactly like a plain blend material ({plain_vs_blend} px differ from Blend)");
}
