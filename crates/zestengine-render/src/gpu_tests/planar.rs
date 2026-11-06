//! Planar-reflection mirror on a REAL `Renderer` (editor RTT viewport, perspective
//! camera): a vertical mirror quad in a small room with three emissive props. The
//! mirror must show the props at their mirrored positions from every viewing
//! angle, including a near corner close to the camera and very grazing views.
//! `ZESTENGINE_TEST_DUMP_DIR` receives each readback as a PPM for inspection.
//!
//! Needs a display (X server / Xwayland); skipped without one or with
//! `ZESTENGINE_TEST_GPU=0`.

use std::sync::Arc;

use glam::{Mat4, Vec3, Vec4};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::{Window, WindowId};

use crate::{CameraData, DrawCmd, MaterialDesc, MaterialParams, MeshData, PlanarSurface, Renderer, Vertex};

const W: u32 = 640;
const H: u32 = 360;
/// Mirror quad: x in [-2, 2], y in [0.5, 2.5], on the plane z = 0 facing +Z.
const MIRROR_HALF: (f32, f32) = (2.0, 1.0);
const MIRROR_CENTER: Vec3 = Vec3::new(0.0, 1.5, 0.0);

pub(super) fn cube() -> MeshData {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let faces = [
        (Vec3::X, Vec3::Y, Vec3::Z),
        (Vec3::NEG_X, Vec3::Y, Vec3::NEG_Z),
        (Vec3::Y, Vec3::Z, Vec3::X),
        (Vec3::NEG_Y, Vec3::Z, Vec3::NEG_X),
        (Vec3::Z, Vec3::Y, Vec3::NEG_X),
        (Vec3::NEG_Z, Vec3::Y, Vec3::X),
    ];
    for (n, up, right) in faces {
        let base = vertices.len() as u32;
        for (u, v) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let p = n * 0.5 + right * (u * 0.5) + up * (v * 0.5);
            vertices.push(Vertex {
                position: p.to_array(),
                normal: n.to_array(),
                tangent: [right.x, right.y, right.z, 1.0],
                color: [1.0; 4],
                ..Default::default()
            });
        }
        // Wind counter-clockwise seen from outside whatever the face basis handedness.
        let flip = right.cross(up).dot(n) < 0.0;
        let q = if flip { [0, 2, 1, 0, 3, 2] } else { [0, 1, 2, 0, 2, 3] };
        indices.extend(q.iter().map(|i| base + i));
    }
    MeshData { vertices, indices, has_lightmap_uv: false }
}

/// A unit XY quad facing +Z (the mirror), subdivided so it is not one triangle pair.
fn quad() -> MeshData {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let n = 8u32;
    for j in 0..=n {
        for i in 0..=n {
            let (u, v) = (i as f32 / n as f32, j as f32 / n as f32);
            vertices.push(Vertex {
                position: [u - 0.5, v - 0.5, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [u, 1.0 - v],
                tangent: [1.0, 0.0, 0.0, 1.0],
                color: [1.0; 4],
                ..Default::default()
            });
        }
    }
    for j in 0..n {
        for i in 0..n {
            let a = j * (n + 1) + i;
            indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
        }
    }
    MeshData { vertices, indices, has_lightmap_uv: false }
}

pub(super) fn draw(mesh: crate::MeshHandle, material: crate::MaterialHandle, transform: Mat4, planar_index: i32) -> DrawCmd {
    DrawCmd {
        gi_source: None,
        mesh,
        material,
        transform,
        highlight: 0.0,
        mesh_center: Vec3::ZERO,
        bound_radius: 40.0,
        lightmap_layer: -1,
        lightmap_size: 0,
        static_geometry: false,
        planar_index,
        occluded: false,
        shadow_casting: crate::ShadowCasting::Off,
        instance_color: None,
        instance_param: None,
    }
}

/// Bright props (name, colour, centre): the reflection must show these at their mirror images.
const PROPS: [([f32; 3], Vec3); 3] = [
    ([1.0, 0.0, 0.0], Vec3::new(-1.0, 1.2, 1.5)),
    ([0.0, 1.0, 0.0], Vec3::new(1.2, 1.8, 2.0)),
    ([0.0, 0.2, 1.0], Vec3::new(0.2, 0.9, 1.0)),
];
const PROP_SIZE: f32 = 0.35;

pub(super) fn lit(r: &mut Renderer, color: [f32; 4], emissive: [f32; 3], rough: f32, metal: f32) -> crate::MaterialHandle {
    r.create_material(&MaterialDesc {
        name: "planar test".into(),
        features: crate::MaterialFeatures { emission: emissive != [0.0;3], ..Default::default() },
        params: MaterialParams {
            metallic: metal,
            base_color: color,
            emission_color: emissive,
            emission_intensity: if emissive == [0.0; 3] { 0.0 } else { 1.0 },
            roughness: rough,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap()
}

struct Shot {
    name: &'static str,
    eye: Vec3,
    /// Prop colour channel -> expected mirror-image pixel inside the mirror.
    px: Vec<(usize, Option<(f32, f32)>)>,
    pixels: Vec<u8>,
    mirror_quad: Vec<[f32; 2]>,
}

#[derive(Default)]
struct App {
    done: bool,
    error: Option<String>,
    shots: Vec<Shot>,
    configs: Vec<(&'static str, Vec3)>,
    mutate: RendererMutate,
}

impl App {
    fn run(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes().with_visible(false).with_inner_size(winit::dpi::LogicalSize::new(W as f64, H as f64));
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
        if let Some(m) = &self.mutate {
            m(&mut r);
        }
        let cube_mesh = r.upload_mesh(&cube()).unwrap();
        let quad_mesh = r.upload_mesh(&quad()).unwrap();
        let grey = lit(&mut r, [0.5, 0.5, 0.5, 1.0], [0.0; 3], 0.9, 0.0);
        let wood = lit(&mut r, [0.45, 0.28, 0.12, 1.0], [0.0; 3], 0.9, 0.0);
        let mirror_mat = lit(&mut r, [1.0, 1.0, 1.0, 1.0], [0.0; 3], 0.05, 1.0);
        let prop_mats: Vec<_> = PROPS.iter().map(|(c, _)| lit(&mut r, [c[0].max(0.02), c[1].max(0.02), c[2].max(0.02), 1.0], [0.0; 3], 0.9, 0.0)).collect();

        let mut draws = vec![
            // floor, back wall behind the mirror, side walls
            draw(cube_mesh, wood, Mat4::from_scale_rotation_translation(Vec3::new(14.0, 0.2, 14.0), glam::Quat::IDENTITY, Vec3::new(0.0, -0.1, 7.0)), -1),
            draw(cube_mesh, grey, Mat4::from_scale_rotation_translation(Vec3::new(14.0, 5.0, 0.2), glam::Quat::IDENTITY, Vec3::new(0.0, 2.5, -0.2)), -1),
            draw(cube_mesh, grey, Mat4::from_scale_rotation_translation(Vec3::new(0.2, 5.0, 14.0), glam::Quat::IDENTITY, Vec3::new(-7.0, 2.5, 7.0)), -1),
            draw(cube_mesh, grey, Mat4::from_scale_rotation_translation(Vec3::new(0.2, 5.0, 14.0), glam::Quat::IDENTITY, Vec3::new(7.0, 2.5, 7.0)), -1),
            draw(cube_mesh, grey, Mat4::from_scale_rotation_translation(Vec3::new(14.0, 0.2, 14.0), glam::Quat::IDENTITY, Vec3::new(0.0, 3.1, 7.0)), -1),
            draw(cube_mesh, grey, Mat4::from_scale_rotation_translation(Vec3::new(14.0, 5.0, 0.2), glam::Quat::IDENTITY, Vec3::new(0.0, 2.5, 14.0)), -1),
            // the mirror quad (z = 0, facing +Z)
            draw(quad_mesh, mirror_mat, Mat4::from_scale_rotation_translation(Vec3::new(MIRROR_HALF.0 * 2.0, MIRROR_HALF.1 * 2.0, 1.0), glam::Quat::IDENTITY, MIRROR_CENTER), 0),
        ];
        for ((_, c), m) in PROPS.iter().zip(&prop_mats) {
            draws.push(draw(cube_mesh, *m, Mat4::from_scale_rotation_translation(Vec3::splat(PROP_SIZE), glam::Quat::IDENTITY, *c), -1));
        }
        let planar = [PlanarSurface { point: MIRROR_CENTER.to_array(), normal: [0.0, 0.0, 1.0], max_roughness: 0.35 }];

        for (name, eye) in self.configs.clone() {
            let target = MIRROR_CENTER;
            let view = Mat4::look_at_rh(eye, target, Vec3::Y);
            let proj = Mat4::perspective_rh(1.0, W as f32 / H as f32, 0.1, 100.0);
            let mut input = crate::cascade_tests::input(CameraData { view, proj, position: eye });
            input.viewport_extent = Some([W, H]);
            input.clear_color = [1.0, 0.0, 1.0, 1.0];
            input.light.ambient = [0.6, 0.6, 0.6];
            input.light.intensity = 0.0;
            input.draws = &draws;
            input.planar = &planar;
            let mut px = None;
            for _ in 0..4 {
                if let Err(e) = r.render(&input) {
                    self.error = Some(format!("{name}: render: {e:#}"));
                    return;
                }
                px = match r.debug_read_viewport() {
                    Ok(p) => p,
                    Err(e) => {
                        self.error = Some(format!("{name}: readback {e:#}"));
                        return;
                    }
                };
            }
            let Some((w, h, pixels)) = px else {
                self.error = Some(format!("{name}: no viewport target"));
                return;
            };
            assert_eq!((w, h), (W, H));
            let vp = proj * view;
            let to_px = |p: Vec3| -> Option<(f32, f32)> {
                let c = vp * Vec4::new(p.x, p.y, p.z, 1.0);
                (c.w > 0.0).then(|| ((c.x / c.w * 0.5 + 0.5) * W as f32, (0.5 - c.y / c.w * 0.5) * H as f32))
            };
            let mirror_quad = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
                .iter()
                .filter_map(|(u, v)| to_px(MIRROR_CENTER + Vec3::new(u * MIRROR_HALF.0, v * MIRROR_HALF.1, 0.0)).map(|p| [p.0, p.1]))
                .collect();
            let px = PROPS.iter().enumerate().map(|(i, (_, c))| (i, to_px(Vec3::new(c.x, c.y, -c.z)))).collect();
            self.shots.push(Shot { name, eye, px, pixels, mirror_quad });
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

fn dump(name: &str, px: &[u8]) {
    let Ok(dir) = std::env::var("ZESTENGINE_TEST_DUMP_DIR") else { return };
    let mut out = format!("P6\n{W} {H}\n255\n").into_bytes();
    for p in px.as_chunks::<4>().0 {
        out.extend_from_slice(&p[..3]);
    }
    let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{name}.ppm")), out);
}

fn at(px: &[u8], x: i32, y: i32) -> [u8; 3] {
    let (x, y) = (x.clamp(0, W as i32 - 1) as usize, y.clamp(0, H as i32 - 1) as usize);
    let i = (y * W as usize + x) * 4;
    [px[i], px[i + 1], px[i + 2]]
}

fn point_in_poly(poly: &[[f32; 2]], x: f32, y: f32) -> bool {
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[j]);
        if (a[1] > y) != (b[1] > y) && x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0] {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// A renderer mutation applied between scene render passes.
type RendererMutate = Option<Box<dyn Fn(&mut Renderer)>>;

fn run(configs: Vec<(&'static str, Vec3)>, mutate: RendererMutate) -> Option<Vec<Shot>> {
    if std::env::var("ZESTENGINE_TEST_GPU").is_ok_and(|v| v == "0") || std::env::var_os("DISPLAY").is_none() {
        eprintln!("[gpu test] no display / GPU disabled: skipped");
        return None;
    }
    let _gpu = crate::cluster_headless::gpu_lock();
    let Ok(mut event_loop) = EventLoop::builder().with_x11().with_any_thread(true).build() else {
        eprintln!("[gpu test] no X11 event loop: skipped");
        return None;
    };
    let mut app = App { configs, mutate, ..Default::default() };
    event_loop.run_app_on_demand(&mut app).unwrap();
    if let Some(e) = &app.error {
        panic!("{e}");
    }
    Some(app.shots)
}

/// Fraction of mirror-interior pixels that are the (unrendered) clear colour or saturated white.
fn bad_fraction(s: &Shot) -> (f32, usize) {
    let (mut bad, mut total) = (0usize, 0usize);
    for y in (0..H as i32).step_by(2) {
        for x in (0..W as i32).step_by(2) {
            if point_in_poly(&s.mirror_quad, x as f32 + 0.5, y as f32 + 0.5) {
                total += 1;
                let [r, g, b] = at(&s.pixels, x, y);
                let clear = r > 200 && b > 200 && g < 80;
                let white = r > 235 && g > 235 && b > 235;
                if clear || white {
                    bad += 1;
                }
            }
        }
    }
    (bad as f32 / total.max(1) as f32, total)
}

/// Regression: the deferred SSR resolve used to run on planar mirror pixels and
/// add `ssr - probe_env` on top of the planar image (blown complementary colours,
/// noise, slanted bands at oblique angles). Fixed in standard.frag by withholding
/// the planar share from the G-buffer reflectivity; this fails without it.
#[test]
fn planar_mirror_shows_props_at_mirrored_positions_from_every_angle() {
    let c = MIRROR_CENTER;
    let configs = vec![
        ("head_on", c + Vec3::new(0.0, 0.0, 5.0)),
        ("deg45", c + Vec3::new(3.5, -0.3, 3.5)),
        ("corner_close", c + Vec3::new(-0.9, -0.5, 1.3)),
        ("deg80", c + Vec3::new(4.9, -0.3, 0.9)),
        ("grazing", c + Vec3::new(5.0, -0.3, 0.25)),
        ("grazing_behind", c + Vec3::new(5.0, -0.3, -0.25)),
    ];
    let Some(shots) = run(configs, None) else { return };
    let mut failures = Vec::new();
    for s in &shots {
        dump(&format!("planar_{}", s.name), &s.pixels);
        let (bad, total) = bad_fraction(s);
        eprintln!("[planar] {:<15} eye {:?} mirror px {total} bad fraction {bad:.3}", s.name, s.eye);
        if s.eye.z < 0.0 {
            continue; // behind the mirror: only needs to not crash
        }
        for (i, p) in &s.px {
            let Some((x, y)) = p else { continue };
            if !point_in_poly(&s.mirror_quad, *x, *y) {
                continue;
            }
            let col = at(&s.pixels, *x as i32, *y as i32);
            let ok = match i {
                0 => col[0] > 150 && col[1] < 80 && col[2] < 80,
                1 => col[1] > 150 && col[0] < 80 && col[2] < 80,
                _ => col[2] > 150 && col[0] < 80 && (60..170).contains(&col[1]),
            };
            eprintln!("[planar]   prop {i} expected at ({x:.0},{y:.0}) colour {col:?} {}", if ok { "ok" } else { "WRONG" });
            if !ok {
                failures.push(format!("{} prop {i} at ({x:.0},{y:.0}) = {col:?}", s.name));
            }
        }
        if bad > 0.02 && s.eye.z > 0.0 {
            failures.push(format!("{}: {:.1}% of the mirror shows clear colour / blown white", s.name, bad * 100.0));
        }
    }
    assert!(failures.is_empty(), "planar mirror wrong: {failures:#?}");
}
