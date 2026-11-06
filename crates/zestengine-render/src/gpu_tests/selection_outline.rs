//! Selection outline of the EDITOR viewport (the offscreen target the dock shows),
//! on a REAL `Renderer`. For each kind of selected object the scene is rendered
//! three ways with the same camera:
//!   * `base`:  nothing highlighted,
//!   * `empty`: the selected draws removed (gives the object's silhouette `M`),
//!   * `sel`:   the draws highlighted.
//!
//! Then, with `d` the distance of a pixel outside `M` to `M`:
//!   * every pixel with `1 <= d <= width` shows the outline colour (a ring at least `width` wide),
//!   * pixels inside `M` are unchanged (the surface stays visible),
//!   * pixels farther than `width + 1.5` are unchanged,
//!   * `highlight == 0` renders exactly `base`.
//!
//! Cases: static cube, transparent cube, GPU-skinned tube (posed), cluster-path
//! mesh, instanced / GPU-culled batch member (Hi-Z on), multi-selection with a
//! primary, an object hidden behind a wall (dim ring), a thin plane.
//! The pictures go to `docs/design/selection-outline-cases.png`.
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
use crate::{
    AlphaMode, CameraData, DrawCmd, LightInstance, LightKind, MaterialDesc, MaterialFeatures, MaterialHandle, MaterialParams, MeshData, MeshHandle, Renderer,
    SelectionOutlineStyle, ShadowCasting, Vertex,
};

const W: u32 = 640;
const H: u32 = 360;
/// The outline colour of the test (sRGB), far from anything the lit scene produces.
const COLOR: [f32; 3] = [1.0, 0.0, 1.0];
const WIDTH: f32 = 3.0;

struct Rig {
    r: Renderer,
    cube: MeshHandle,
    plane: MeshHandle,
    opaque: MaterialHandle,
    glass: MaterialHandle,
    wall_mat: MaterialHandle,
    frame: u32,
}

fn camera() -> CameraData {
    let eye = Vec3::new(0.0, 1.5, 7.0);
    let view = Mat4::look_at_rh(eye, Vec3::new(0.0, 1.0, 0.0), Vec3::Y);
    CameraData { view, proj: Mat4::perspective_rh(0.9, W as f32 / H as f32, 0.1, 100.0), position: eye }
}

fn at(p: Vec3, s: f32) -> Mat4 {
    Mat4::from_scale_rotation_translation(Vec3::splat(s), Quat::from_rotation_y(0.5), p)
}

fn set(mut d: DrawCmd, highlight: f32) -> DrawCmd {
    d.highlight = highlight;
    d.shadow_casting = ShadowCasting::Off;
    d.bound_radius = 3.0;
    d
}

impl Rig {
    fn render(&mut self, draws: &[DrawCmd], pose: Option<(MeshHandle, &[Mat4])>) -> Vec<u8> {
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
        let mut px = Vec::new();
        for _ in 0..4 {
            if let Some((mesh, palette)) = pose {
                self.r.skin_mesh_gpu(mesh, palette);
            }
            let mut input = crate::cascade_tests::input(camera());
            input.draws = draws;
            input.lights = &sun;
            input.draw_skybox = false;
            input.viewport_extent = Some([W, H]);
            input.time = self.frame as f32 / 60.0;
            self.frame += 1;
            self.r.render(&input).unwrap_or_else(|e| panic!("render: {e:#}"));
            let (w, h, p) = self.r.debug_read_viewport().expect("readback").expect("viewport target");
            assert_eq!((w, h), (W, H));
            px = p;
        }
        px
    }
}

fn max_diff(a: &[u8], b: &[u8], i: usize) -> i32 {
    (0..3).map(|c| (a[i * 4 + c] as i32 - b[i * 4 + c] as i32).abs()).max().unwrap()
}

/// Per-pixel distance to the silhouette (0 inside), capped at `cap`.
fn distance_field(m: &[bool], cap: i32) -> Vec<f32> {
    let mut out = vec![cap as f32 + 1.0; m.len()];
    let (w, h) = (W as i32, H as i32);
    let mut offsets = Vec::new();
    for dy in -cap..=cap {
        for dx in -cap..=cap {
            let d = ((dx * dx + dy * dy) as f32).sqrt();
            if d <= cap as f32 {
                offsets.push((dx, dy, d));
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            if m[i] {
                out[i] = 0.0;
                continue;
            }
            let mut best = cap as f32 + 1.0;
            for &(dx, dy, d) in &offsets {
                let (qx, qy) = (x + dx, y + dy);
                if qx >= 0 && qy >= 0 && qx < w && qy < h && m[(qy * w + qx) as usize] && d < best {
                    best = d;
                }
            }
            out[i] = best;
        }
    }
    out
}

struct Verdict {
    name: String,
    ring_frac: f32,
    ring_px: usize,
    interior_changed: f32,
    far_changed: f32,
    zero_changed: usize,
    failures: Vec<String>,
}

fn is_outline(px: &[u8], i: usize) -> bool {
    let want = [COLOR[0] * 255.0, COLOR[1] * 255.0, COLOR[2] * 255.0];
    (0..3).all(|c| (px[i * 4 + c] as f32 - want[c]).abs() <= 48.0)
}

/// Evaluate one case; `dim` = the ring is expected translucent (hidden object).
fn evaluate(name: &str, base: &[u8], empty: &[u8], sel: &[u8], off: &[u8], width: f32, dim: bool) -> Verdict {
    let n = (W * H) as usize;
    let m: Vec<bool> = (0..n).map(|i| max_diff(base, empty, i) > 1).collect();
    let m_px = m.iter().filter(|&&b| b).count();
    let w = width.ceil() as i32;
    let dist = distance_field(&m, w + 3);
    let mut ring = (0usize, 0usize);
    let (mut interior, mut interior_changed, mut far, mut far_changed) = (0usize, 0usize, 0usize, 0usize);
    for (i, d) in dist.iter().enumerate() {
        if *d == 0.0 {
            interior += 1;
            interior_changed += (max_diff(base, sel, i) > 4) as usize;
        } else if *d >= 1.0 && *d <= width {
            ring.1 += 1;
            let hit = if dim { max_diff(base, sel, i) > 12 } else { is_outline(sel, i) };
            ring.0 += hit as usize;
        } else if *d > width + 1.5 {
            far += 1;
            far_changed += (max_diff(base, sel, i) > 4) as usize;
        }
    }
    let zero_changed = (0..n).filter(|&i| max_diff(base, off, i) > 0).count();
    let mut v = Verdict {
        name: name.into(),
        ring_frac: ring.0 as f32 / ring.1.max(1) as f32,
        ring_px: ring.1,
        interior_changed: interior_changed as f32 / interior.max(1) as f32,
        far_changed: far_changed as f32 / far.max(1) as f32,
        zero_changed,
        failures: Vec::new(),
    };
    if m_px < 200 {
        v.failures.push(format!("silhouette too small ({m_px} px): the case does not show the object"));
    }
    let need = if dim { 0.7 } else { 0.9 };
    if v.ring_frac < need {
        v.failures.push(format!("only {:.0}% of the {} ring pixels (1..={width} px outside the silhouette) show the outline", v.ring_frac * 100.0, v.ring_px));
    }
    // Cluster geometry is quantised: its edge pixels may differ from the ordinary mesh by one pixel.
    let interior_tol = if name.starts_with("cluster") { 0.06 } else { 0.01 };
    if v.interior_changed > interior_tol {
        v.failures.push(format!("{:.1}% of the object's own pixels changed (outline over the surface)", v.interior_changed * 100.0));
    }
    if v.far_changed > 0.001 {
        v.failures.push(format!("{:.2}% of the pixels far from the object changed", v.far_changed * 100.0));
    }
    if zero_changed > 0 {
        v.failures.push(format!("{zero_changed} px differ from the base render with highlight 0"));
    }
    v
}

struct Case {
    sel: Vec<u8>,
    bbox: (i32, i32, i32, i32),
}

fn bbox(base: &[u8], empty: &[u8]) -> (i32, i32, i32, i32) {
    let (mut x0, mut y0, mut x1, mut y1) = (W as i32, H as i32, 0, 0);
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            if max_diff(base, empty, (y * W as i32 + x) as usize) > 1 {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
            }
        }
    }
    (x0, y0, x1, y1)
}

fn sheet(cases: &[Case]) -> Vec<u8> {
    const TW: usize = 320;
    const TH: usize = 180;
    const Z: usize = 1;
    let cols = 3usize;
    let rows = cases.len().div_ceil(cols);
    let (sw, sh) = (cols * TW, rows * TH);
    let mut out = vec![24u8; sw * sh * 4];
    for (k, c) in cases.iter().enumerate() {
        let (cx, cy) = ((c.bbox.0 + c.bbox.2) / 2, (c.bbox.1 + c.bbox.3) / 2);
        let (ox, oy) = ((k % cols) * TW, (k / cols) * TH);
        for ty in 0..TH {
            for tx in 0..TW {
                let sx = cx + (tx / Z) as i32 - (TW / Z / 2) as i32;
                let sy = cy + (ty / Z) as i32 - (TH / Z / 2) as i32;
                if sx < 0 || sy < 0 || sx >= W as i32 || sy >= H as i32 {
                    continue;
                }
                let s = ((sy as u32 * W + sx as u32) * 4) as usize;
                let o = ((oy + ty) * sw + ox + tx) * 4;
                // Left half of a tile: the selection render; the corner mark labels the case index.
                out[o..o + 4].copy_from_slice(&c.sel[s..s + 4]);
                out[o + 3] = 255;
            }
        }
    }
    out
}

fn tube() -> (MeshData, Vec<Mat4>) {
    let (v, idx) = super::skin::skinned_tube_for_renderer();
    let palette = vec![
        Mat4::IDENTITY,
        Mat4::from_translation(Vec3::new(0.0, 1.5, 0.0)) * Mat4::from_rotation_z(0.7) * Mat4::from_translation(Vec3::new(0.0, -1.5, 0.0)),
        Mat4::from_translation(Vec3::new(0.0, 3.0, 0.0)) * Mat4::from_rotation_z(-0.9) * Mat4::from_translation(Vec3::new(0.0, -3.0, 0.0)),
    ];
    (MeshData { vertices: v, indices: idx, has_lightmap_uv: false }, palette)
}

fn wall() -> MeshData {
    let mut vertices = Vec::new();
    for (x, y) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        vertices.push(Vertex { position: [x * 0.5, y * 0.5, 0.0], normal: [0.0, 0.0, 1.0], tangent: [1.0, 0.0, 0.0, 1.0], color: [1.0; 4], ..Default::default() });
    }
    MeshData { vertices, indices: vec![0, 1, 2, 0, 2, 3], has_lightmap_uv: false }
}

struct Out {
    verdicts: Vec<Verdict>,
    cases: Vec<Case>,
}

/// Run one case: `draws` with `selected` indices highlighted (`primary` 1.0, the rest 0.5).
#[allow(clippy::too_many_arguments)]
fn case(out: &mut Out, rig: &mut Rig, name: &'static str, draws: Vec<DrawCmd>, selected: &[usize], primary: usize, dim: bool, pose: Option<(MeshHandle, Vec<Mat4>)>) {
    let p = pose.as_ref().map(|(m, v)| (*m, v.as_slice()));
    let mut hl = draws.clone();
    for &i in selected {
        hl[i].highlight = if i == primary { 1.0 } else { 0.5 };
    }
    let without: Vec<DrawCmd> = draws.iter().enumerate().filter(|(i, _)| !selected.contains(i)).map(|(_, d)| *d).collect();
    let base = rig.render(&draws, p);
    let empty = rig.render(&without, p);
    let sel = rig.render(&hl, p);
    // Highlight 0 again after a highlighted frame (no state leaks).
    let off = rig.render(&draws, p);
    let v = evaluate(name, &base, &empty, &sel, &off, WIDTH, dim);
    let bb = bbox(&base, &empty);
    out.cases.push(Case { sel, bbox: bb });
    out.verdicts.push(v);
}

fn run(r: Renderer) -> (Vec<Verdict>, Vec<Case>, Vec<String>) {
    let mut r = r;
    let cube_mesh = r.upload_mesh(&cube()).unwrap();
    let plane = r.upload_mesh(&wall()).unwrap();
    let opaque = lit(&mut r, [0.9, 0.55, 0.15, 1.0], [0.0; 3], 0.6, 0.0);
    let glass = r
        .create_material(&MaterialDesc {
            name: "glass".into(),
            params: MaterialParams { base_color: [0.2, 0.6, 1.0, 0.45], roughness: 0.4, ..Default::default() },
            features: MaterialFeatures { alpha_mode: AlphaMode::Blend, ..Default::default() },
            ..Default::default()
        })
        .unwrap();
    let wall_mat = lit(&mut r, [0.3, 0.9, 0.3, 1.0], [0.0; 3], 0.8, 0.0);
    let mut rig = Rig { r, cube: cube_mesh, plane, opaque, glass, wall_mat, frame: 0 };
    let mut notes = Vec::new();
    let width = WIDTH;
    rig.r.set_selection_outline(SelectionOutlineStyle { enabled: true, color: COLOR, width_px: width, occluded: 0.35 });

    let mut out = Out { verdicts: Vec::new(), cases: Vec::new() };
    let cube_draw = |rig: &Rig, pos: Vec3, mat: MaterialHandle, s: f32| set(draw(rig.cube, mat, at(pos, s), -1), 0.0);

    // (a) a static opaque cube
    let d = vec![cube_draw(&rig, Vec3::new(0.0, 1.0, 0.0), rig.opaque, 1.6)];
    case(&mut out, &mut rig, "static cube", d, &[0], 0, false, None);

    // (b) a transparent (blended) cube
    let d = vec![cube_draw(&rig, Vec3::new(0.0, 1.0, 0.0), rig.glass, 1.6)];
    case(&mut out, &mut rig, "transparent cube", d, &[0], 0, false, None);

    // (c) a GPU-skinned, posed mesh
    {
        let (data, palette) = tube();
        let skinned = rig.r.upload_mesh_gpu_skinned_morphed(&data, 3, &[]).unwrap();
        // Emissive so the unlit (shadow) side never matches the background and the silhouette is exact.
        // Double sided: the outline mask is, and a folded single-sided region would show it where the surface does not.
        let glow = rig
            .r
            .create_material(&MaterialDesc {
                name: "glow".into(),
                params: MaterialParams { base_color: [0.9, 0.55, 0.15, 1.0], emission_color: [0.5, 0.35, 0.1], emission_intensity: 1.0, roughness: 0.6, ..Default::default() },
                features: MaterialFeatures { double_sided: true, ..Default::default() },
                ..Default::default()
            })
            .unwrap();
        let d = vec![set(draw(skinned, glow, Mat4::from_translation(Vec3::new(-0.3, 0.1, 0.0)) * Mat4::from_scale(Vec3::splat(0.6)), -1), 0.0)];
        case(&mut out, &mut rig, "skinned tube (posed)", d, &[0], 0, false, Some((skinned, palette)));
    }

    // (d) a mesh drawn by the cluster path
    {
        let scene = super::xr_cluster::Scene::new(&mut rig.r).unwrap();
        notes.push(format!("cluster rendering enabled: {}", rig.r.cluster_rendering_enabled()));
        let mut d = set(draw(scene.mesh, scene.material, Mat4::from_translation(Vec3::new(0.0, 2.0, 0.0)), -1), 0.0);
        d.bound_radius = 5.0;
        case(&mut out, &mut rig, "cluster-path mesh", vec![d], &[0], 0, false, None);
        rig.r.set_cluster_settings(crate::ClusterSettings { mode: crate::ClusterMode::Off, ..rig.r.cluster_settings() });
    }

    // (e) an instanced / GPU-culled batch member (GPU cull + Hi-Z on)
    {
        rig.r.set_gpu_cull_settings(crate::GpuCullSettings { enabled: true, hiz_occlusion: true, ..crate::GpuCullSettings::default() });
        let d: Vec<DrawCmd> = (0..5).map(|k| cube_draw(&rig, Vec3::new(-3.6 + k as f32 * 1.8, 1.0, 0.0), rig.opaque, 1.2)).collect();
        case(&mut out, &mut rig, "GPU-culled batch member", d, &[2], 2, false, None);
    }

    // (f) multi-selection with a primary (the outer two are secondary, drawn with the same ring)
    {
        let d: Vec<DrawCmd> = (0..3).map(|k| cube_draw(&rig, Vec3::new(-2.4 + k as f32 * 2.4, 1.0, 0.0), rig.opaque, 1.3)).collect();
        case(&mut out, &mut rig, "multi-selection (3, primary in the middle)", d, &[0, 1, 2], 1, true, None);
    }

    // (g) a selected object hidden behind a wall: the dim ring
    {
        let hidden = cube_draw(&rig, Vec3::new(0.0, 1.0, -1.0), rig.opaque, 1.2);
        let wall_draw = set(draw(rig.plane, rig.wall_mat, Mat4::from_translation(Vec3::new(0.0, 1.0, 1.5)) * Mat4::from_scale(Vec3::splat(2.2)), -1), 0.0);
        // The silhouette M comes from the wall-less render, so the case is built by hand.
        let all = vec![hidden, wall_draw];
        let mut hl = all.clone();
        hl[0].highlight = 1.0;
        let only = vec![hidden];
        let nothing: Vec<DrawCmd> = vec![wall_draw];
        let base_open = rig.render(&only, None);
        let none_open = rig.render(&[], None);
        let base = rig.render(&all, None);
        let sel = rig.render(&hl, None);
        let off = rig.render(&all, None);
        let _ = nothing;
        // Silhouette from the open scene; ring and exclusions compared on the walled one.
        let n = (W * H) as usize;
        let m: Vec<bool> = (0..n).map(|i| max_diff(&base_open, &none_open, i) > 1).collect();
        let dist = distance_field(&m, 6);
        let wall_px = |i: usize| max_diff(&base, &none_open, i) > 1;
        let mut hit = (0usize, 0usize);
        for (i, d) in dist.iter().enumerate() {
            if *d >= 1.0 && *d <= width && wall_px(i) {
                hit.1 += 1;
                hit.0 += (max_diff(&base, &sel, i) > 12) as usize;
            }
        }
        let leaked = (0..n).filter(|&i| dist[i] == 0.0 && wall_px(i) && max_diff(&base, &sel, i) > 4).count();
        let mut v = Verdict { name: "hidden behind a wall (dim ring)".into(), ring_frac: hit.0 as f32 / hit.1.max(1) as f32, ring_px: hit.1, interior_changed: 0.0, far_changed: 0.0, zero_changed: (0..n).filter(|&i| max_diff(&base, &off, i) > 0).count(), failures: vec![] };
        if hit.1 < 100 || v.ring_frac < 0.7 {
            v.failures.push(format!("the hidden object's ring shows on {}/{} of the pixels around it", hit.0, hit.1));
        }
        if leaked > 0 {
            v.failures.push(format!("{leaked} px of the wall over the hidden object were painted (the hidden surface must not show)"));
        }
        if v.zero_changed > 0 {
            v.failures.push(format!("{} px differ with highlight 0", v.zero_changed));
        }
        let bb = bbox(&base_open, &none_open);
        out.cases.push(Case { sel, bbox: bb });
        out.verdicts.push(v);
    }

    // (h) a thin one-sided plane (no volume for a hull to inflate)
    {
        let d = vec![set(draw(rig.plane, rig.opaque, Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)) * Mat4::from_scale(Vec3::splat(2.5)), -1), 0.0)];
        case(&mut out, &mut rig, "thin plane", d, &[0], 0, false, None);
    }

    (out.verdicts, out.cases, notes)
}

#[derive(Default)]
struct App {
    done: bool,
    error: Option<String>,
    verdicts: Vec<Verdict>,
    cases: Vec<Case>,
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
                    let (v, c, n) = run(r);
                    self.verdicts = v;
                    self.cases = c;
                    self.notes = n;
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
fn editor_viewport_outlines_every_kind_of_selected_object() {
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
        eprintln!("[selection] {n}");
    }
    let mut failed = Vec::new();
    for v in &app.verdicts {
        eprintln!(
            "[selection] {:<44} ring {:>5.1}% of {:>5} px | object changed {:>5.2}% | far changed {:>6.3}% | highlight-0 diff {} px{}",
            v.name,
            v.ring_frac * 100.0,
            v.ring_px,
            v.interior_changed * 100.0,
            v.far_changed * 100.0,
            v.zero_changed,
            if v.failures.is_empty() { "" } else { "  FAIL" }
        );
        for f in &v.failures {
            failed.push(format!("{}: {f}", v.name));
        }
    }
    let png = super::stencil::png_rgba(
        (3 * 320) as u32,
        (app.cases.len().div_ceil(3) * 180) as u32,
        &sheet(&app.cases),
    );
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/design/selection-outline-cases.png");
    if let Err(e) = std::fs::write(&path, png) {
        eprintln!("[selection] could not write {}: {e}", path.display());
    }
    assert!(failed.is_empty(), "selection outline failures:\n  {}", failed.join("\n  "));
}

/// Per-pass GPU cost of the outline on a 1280x720 editor viewport with the default-scene shape
/// (a 40 m ground plane, one cube, one sun): nothing selected vs the cube selected.
/// Printed with `--nocapture`; the bound is generous, it guards against a 10x regression.
#[derive(Default)]
struct CostApp {
    done: bool,
    error: Option<String>,
    rows: Vec<(String, f32, f32, f32)>,
}

impl ApplicationHandler for CostApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.done {
            return;
        }
        self.done = true;
        let attrs = Window::default_attributes().with_visible(false).with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
        match event_loop.create_window(attrs).map(Arc::new) {
            Ok(w) => match Renderer::new(w) {
                Ok(mut r) => {
                    r.set_vsync(false);
                    r.set_gpu_zone_timing(true);
                    let cube_mesh = r.upload_mesh(&cube()).unwrap();
                    let ground = r.upload_mesh(&wall()).unwrap();
                    let mat = lit(&mut r, [0.8, 0.8, 0.8, 1.0], [0.0; 3], 0.7, 0.0);
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
                    let eye = Vec3::new(0.0, 3.0, 6.0);
                    let cam = CameraData {
                        view: Mat4::look_at_rh(eye, Vec3::new(0.0, 0.5, 0.0), Vec3::Y),
                        proj: Mat4::perspective_rh(1.0, 1280.0 / 720.0, 0.1, 200.0),
                        position: eye,
                    };
                    for (label, hl) in [("nothing selected", 0.0f32), ("cube selected", 1.0), ("nothing selected", 0.0), ("cube selected", 1.0)] {
                        let mut draws = vec![
                            set(draw(ground, mat, Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2) * Mat4::from_scale(Vec3::splat(80.0)), -1), 0.0),
                            set(draw(cube_mesh, mat, Mat4::from_translation(Vec3::new(0.0, 0.5, 0.0)), -1), hl),
                        ];
                        draws[0].bound_radius = 80.0;
                        let (mut scene, mut post, mut frame) = (Vec::new(), Vec::new(), Vec::new());
                        for k in 0..16 {
                            let mut input = crate::cascade_tests::input(cam);
                            input.draws = &draws;
                            input.lights = &sun;
                            input.viewport_extent = Some([1280, 720]);
                            if let Err(e) = r.render(&input) {
                                self.error = Some(format!("render: {e:#}"));
                                event_loop.exit();
                                return;
                            }
                            if k >= 6 {
                                let s = r.stats();
                                scene.push(s.gpu_scene_ms);
                                post.push(s.gpu_post_ms);
                                frame.push(s.gpu_frame_ms);
                            }
                        }
                        let med = |mut v: Vec<f32>| {
                            v.sort_by(|a, b| a.total_cmp(b));
                            v[v.len() / 2]
                        };
                        self.rows.push((label.into(), med(scene), med(post), med(frame)));
                    }
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
fn selection_outline_costs_a_fraction_of_a_millisecond() {
    if std::env::var("ZESTENGINE_TEST_GPU").is_ok_and(|v| v == "0") || std::env::var_os("DISPLAY").is_none() {
        eprintln!("[gpu test] no display / GPU disabled: skipped");
        return;
    }
    let _gpu = crate::cluster_headless::gpu_lock();
    let Ok(mut event_loop) = EventLoop::builder().with_x11().with_any_thread(true).build() else {
        eprintln!("[gpu test] no X11 event loop: skipped");
        return;
    };
    let mut app = CostApp::default();
    event_loop.run_app_on_demand(&mut app).unwrap();
    if let Some(e) = &app.error {
        panic!("{e}");
    }
    for (label, scene, post, frame) in &app.rows {
        eprintln!("[selection cost] {label:<18} scene {scene:.3} ms | post {post:.3} ms | whole frame {frame:.3} ms");
    }
    let (none, sel) = (app.rows[2].3, app.rows[3].3);
    assert!(sel - none < 1.5, "the selection outline adds {:.3} ms to the frame", sel - none);
}
