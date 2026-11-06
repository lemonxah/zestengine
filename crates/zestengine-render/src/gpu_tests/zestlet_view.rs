//! The Zestlet viewport modes on the real device: a `Renderer` draws the XR test scene (32 one-quad
//! zestlets and an ordinary wall in front of them) in each `gi_debug` mode into an image we own, and the
//! pixels are checked against the colours the pure Rust mirrors (`cluster_coverage.rs`) promise.
//!
//!  * 9 Zestlets: every zestlet its own hash colour, the ordinary wall flat grey,
//!  * 8 Zestlet LOD: all zestlets the level-0 colour (the quad grid has one level),
//!  * 10 Zestlet coverage: zestlets green, the wall in the colour of its recorded refusal reason (every
//!    reason is tried, plus the unrecorded "too small" fallback),
//!  * 11 Zestlet triangles: a colour per triangle id, wall and zestlets alike.
//!
//! Tone mapping is off in these frames, so a written colour comes out as `round(255 * c)`. One test creates
//! its own invisible X11 window (needs a display; skipped without one or with `ZESTENGINE_TEST_GPU=0`).
//! The contact sheet goes to `docs/design/zestlet_views.png`.

use std::sync::Arc;

use anyhow::Result;
use ash::vk;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::{Window, WindowId};

use super::xr_cluster::{FORMAT, H, Scene, Target, W, eyes, frame_input, save_sheet, set_mode};
use crate::{ClusterRefusal, Renderer, VIEW_ZESTLET_COVERAGE, VIEW_ZESTLET_LOD, VIEW_ZESTLET_TRIANGLES, VIEW_ZESTLETS};

/// One frame of the XR scene in viewport mode `mode`, read back (RGBA8).
fn frame(r: &mut Renderer, scene: &Scene, target: &Target, mode: u32) -> Result<Vec<u8>> {
    let (eyes, cull) = eyes();
    set_mode(r, true, false);
    // The scene's wall hides the whole grid at this camera; move it aside, to the right of the grid.
    let mut draws = scene.draws(true);
    draws[1].transform = glam::Mat4::from_translation(glam::Vec3::new(1.9, 0.0, -3.5)) * glam::Mat4::from_scale(glam::Vec3::new(0.6, 0.6, 1.0));
    let mut input = frame_input(&draws, cull.1);
    input.gi_debug = mode;
    input.postfx.tonemap = 0;
    // Three frames, like the XR tests: the deferred DAG copy and the cluster cull need a few to arm.
    for _ in 0..3 {
        r.render_xr_frame(&input)?;
        r.render_xr_eye(target.raw(), FORMAT.as_raw() as u32, W, H, eyes[0].view, eyes[0].proj, &input, None)?;
    }
    target.read(r)
}

fn px(img: &[u8], i: usize) -> [u8; 3] {
    [img[i * 4], img[i * 4 + 1], img[i * 4 + 2]]
}

fn bytes(c: [f32; 3]) -> [u8; 3] {
    c.map(|v| (v * 255.0).round() as u8)
}

fn near(a: [u8; 3], b: [u8; 3], tol: u8) -> bool {
    (0..3).all(|k| a[k].abs_diff(b[k]) <= tol)
}

fn lit_pixels(img: &[u8]) -> Vec<usize> {
    (0..(W * H) as usize).filter(|&i| px(img, i) != [0, 0, 0]).collect()
}

/// Pixels of `img` within `tol` of `colour`.
fn count_near(img: &[u8], colour: [f32; 3], tol: u8) -> usize {
    (0..(W * H) as usize).filter(|&i| near(px(img, i), bytes(colour), tol)).count()
}

pub(super) fn run(r: &mut Renderer) -> Result<String> {
    let scene = Scene::new(r)?;
    let target = Target::new(r, 1)?;
    let mut report = String::new();
    let tol = 3u8;

    let lit = frame(r, &scene, &target, 0)?;
    let zest = frame(r, &scene, &target, VIEW_ZESTLETS)?;
    let lod = frame(r, &scene, &target, VIEW_ZESTLET_LOD)?;
    let cov = frame(r, &scene, &target, VIEW_ZESTLET_COVERAGE)?;
    let tri = frame(r, &scene, &target, VIEW_ZESTLET_TRIANGLES)?;
    let (instances, batches, armed) = r.cluster.as_ref().map(|c| (c.last_instances, c.last_batches, c.drawable_armed())).unwrap_or_default();
    report += &format!("\n[zestlet view] cluster cull: {instances} instances, {batches} batches, armed {armed}");
    eprintln!("{report}");
    let covered = lit_pixels(&zest).len();
    anyhow::ensure!(covered > 2000, "the Zestlets view drew only {covered} px");
    anyhow::ensure!(lit != cov && lit != zest, "the debug modes did not change the picture");

    // ---- 9: the wall is flat grey, every other pixel is some zestlet's hash colour
    let grey = bytes(crate::ORDINARY_GREY);
    let wall: Vec<usize> = (0..(W * H) as usize).filter(|&i| near(px(&zest, i), grey, tol)).collect();
    anyhow::ensure!(wall.len() > 500, "the ordinary wall is not flat grey in the Zestlets view ({} px)", wall.len());
    let expected: Vec<[u8; 3]> = (0..32).map(|k| bytes(crate::zestlet_hash_colour(crate::zestlet_id(0, k)))).collect();
    let mut seen = std::collections::HashSet::new();
    let mut stray = 0usize;
    for i in lit_pixels(&zest) {
        let p = px(&zest, i);
        if near(p, grey, tol) {
            continue;
        }
        match expected.iter().position(|&e| near(p, e, tol)) {
            Some(k) => {
                seen.insert(k);
            }
            None => stray += 1,
        }
    }
    report += &format!("\n[zestlet view] Zestlets: {} px, {} px wall grey, {} of 32 zestlet colours seen, {stray} edge px matched nothing", covered, wall.len(), seen.len());
    if seen.len() < 18 {
        let mut hist = std::collections::HashMap::new();
        for i in lit_pixels(&zest) {
            *hist.entry(px(&zest, i)).or_insert(0usize) += 1;
        }
        let mut top: Vec<_> = hist.into_iter().collect();
        top.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        report += &format!("\n[zestlet view] most common colours in the Zestlets view: {:?}; first expected {:?}", &top[..top.len().min(8)], &expected[..4]);
        eprintln!("{report}");
    }
    anyhow::ensure!(seen.len() >= 18, "only {} distinct zestlet colours showed (the wall hides some)", seen.len());
    anyhow::ensure!(stray * 20 < covered, "{stray} of {covered} px are neither grey nor a zestlet colour");

    // ---- 8: all zestlets one level colour, the wall grey
    let level = crate::zestlet_level_colour(0);
    let on_level = count_near(&lod, level, tol);
    let lod_grey = wall.iter().filter(|&&i| near(px(&lod, i), grey, tol)).count();
    report += &format!("\n[zestlet view] LOD: {on_level} px in the level-0 colour, wall grey on {lod_grey} of {} px", wall.len());
    anyhow::ensure!(on_level > 1500 && on_level + wall.len() > covered * 9 / 10, "LOD view: {on_level} px in the level colour");
    anyhow::ensure!(lod_grey * 10 > wall.len() * 9, "LOD view: the wall is not grey");

    // ---- 10: zestlets green; the wall takes the colour of its refusal reason
    let green = count_near(&cov, crate::COVERAGE_ZESTLET, tol);
    anyhow::ensure!(green > 1500, "coverage view: only {green} green px");
    let wall_mesh = scene.wall_mesh();
    report += &format!("\n[zestlet view] coverage: {green} green px; wall colours by reason:");
    let mut sheet_cov = cov.clone();
    for reason in ClusterRefusal::ALL {
        r.set_mesh_refusal(wall_mesh, Some(reason));
        let img = frame(r, &scene, &target, VIEW_ZESTLET_COVERAGE)?;
        let on_wall = wall.iter().filter(|&&i| near(px(&img, i), bytes(reason.view_colour()), tol)).count();
        report += &format!(" {}={}", reason.label(), on_wall);
        anyhow::ensure!(on_wall * 10 > wall.len() * 9, "coverage view: the wall for '{}' shows {on_wall} of {} px in {:?}", reason.label(), wall.len(), reason.view_colour());
        if reason == ClusterRefusal::CustomShader {
            sheet_cov = img;
        }
    }
    // No record: a 2-triangle mesh is "too small"; and the green zestlets never change colour.
    r.set_mesh_refusal(wall_mesh, None);
    let img = frame(r, &scene, &target, VIEW_ZESTLET_COVERAGE)?;
    let small = wall.iter().filter(|&&i| near(px(&img, i), bytes(ClusterRefusal::TooSmall.view_colour()), tol)).count();
    anyhow::ensure!(small * 10 > wall.len() * 9, "coverage view: an unrecorded small mesh is not 'too small' ({small} px)");
    anyhow::ensure!(count_near(&img, crate::COVERAGE_ZESTLET, tol) == green, "coverage view: the zestlets changed with the wall's reason");

    // ---- 11: colour per triangle
    let mut tri_colours = std::collections::HashSet::new();
    let mut wall_colours = std::collections::HashSet::new();
    for i in lit_pixels(&tri) {
        let p = px(&tri, i);
        if wall.contains(&i) { wall_colours.insert(p) } else { tri_colours.insert(p) };
    }
    report += &format!("\n[zestlet view] triangles: {} colours on the zestlets, {} on the wall (2 triangles)", tri_colours.len(), wall_colours.len());
    anyhow::ensure!(tri_colours.len() >= 24, "triangles view: only {} colours on the zestlets", tri_colours.len());
    let id_colours: Vec<[u8; 3]> = (0..64).map(|id| bytes(crate::zestlet_hash_colour(id))).collect();
    let main_wall: Vec<[u8; 3]> = wall_colours.iter().copied().filter(|c| id_colours.iter().any(|&e| near(*c, e, tol))).collect();
    anyhow::ensure!(main_wall.len() >= 2, "triangles view: the wall's two triangles do not show two id colours ({} of {:?})", main_wall.len(), wall_colours.len());

    save_sheet("zestlet_views.png", &[&lit, &zest, &lod, &sheet_cov, &tri]);
    report += "\n[zestlet view] picture: docs/design/zestlet_views.png (Lit, Zestlets, Zestlet LOD, Zestlet coverage with the wall as 'custom shader', Zestlet triangles)";
    target.destroy(r);
    let _ = vk::Format::UNDEFINED;
    Ok(report)
}

struct App {
    result: Option<Result<String>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.result.is_some() {
            return;
        }
        let attrs = Window::default_attributes().with_visible(false).with_inner_size(winit::dpi::LogicalSize::new(320.0, 200.0));
        self.result = Some((|| {
            let window = Arc::new(event_loop.create_window(attrs)?);
            let mut r = Renderer::new(window)?;
            r.set_vsync(false);
            run(&mut r)
        })());
        event_loop.exit();
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
}

#[test]
fn zestlet_view_modes_paint_the_expected_colours() {
    if std::env::var("ZESTENGINE_TEST_GPU").is_ok_and(|v| v == "0") || std::env::var_os("DISPLAY").is_none() {
        eprintln!("[gpu test] no display / GPU disabled: skipped");
        return;
    }
    let _gpu = crate::cluster_headless::gpu_lock();
    let Ok(mut event_loop) = EventLoop::builder().with_x11().with_any_thread(true).build() else {
        eprintln!("[gpu test] no X11 event loop: skipped");
        return;
    };
    let mut app = App { result: None };
    event_loop.run_app_on_demand(&mut app).unwrap();
    match app.result {
        Some(Ok(report)) => eprintln!("{report}"),
        Some(Err(e)) => panic!("{e:#}"),
        None => eprintln!("[gpu test] the window never appeared: skipped"),
    }
}
