//! UNITY-11 end to end with the SHIPPED example (`shaders/zestengine_wind_sway.frag`): the
//! assets crate cuts the `// @vertex` section out of the file, assembles the hook chunk with
//! the shader's own property block, and the renderer builds the hooked vertex modules; a
//! row of columns then sways, rooted at the ground, and their shadows sway with them.
//! Numbers are checked against the shader's own formula projected into the picture.
//!
//! One `#[test]` (winit allows one event loop per process); skips cleanly with no display / GPU.
//! Contact sheet: `docs/design/wind-sway-contact-sheet.png`.

use std::sync::Arc;

use glam::{Mat4, Vec3};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::{Window, WindowId};
use zestengine_assets::{ShaderPropKind, ShaderSource, TextureDefault, compile_shader, parse_shader_source, vertex_hook_glsl};
use zestengine_render::{
    CameraData, CustomTextureDefault, DrawCmd, FrameInput, LightData, LightInstance, LightKind, MAX_REFLECTION_PROBES, MaterialDesc, MaterialFeatures, MaterialHandle,
    MaterialParams, MeshData, MeshHandle, PostFx, Renderer, ShadowCasting, Vertex,
};

const W: u32 = 480;
const H: u32 = 270;
const FOV: f32 = std::f32::consts::FRAC_PI_3;

/// `n x n` grid on `origin + u s + v t`, facing `u x v`.
fn face_grid(origin: Vec3, u: Vec3, v: Vec3, n: u32) -> (Vec<Vertex>, Vec<u32>) {
    let normal = u.cross(v).normalize();
    let tangent = u.normalize();
    let w = if normal.cross(tangent).dot(v) >= 0.0 { 1.0 } else { -1.0 };
    let mut vertices = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            let (s, t) = (i as f32 / n as f32, j as f32 / n as f32);
            vertices.push(Vertex {
                position: (origin + u * s + v * t).to_array(),
                normal: normal.to_array(),
                uv: [s, t],
                tangent: [tangent.x, tangent.y, tangent.z, w],
                color: [1.0; 4],
                ..Default::default()
            });
        }
    }
    let mut indices = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let a = j * (n + 1) + i;
            indices.extend([a, a + 1, a + n + 1, a + 1, a + n + 2, a + n + 1]);
        }
    }
    (vertices, indices)
}

/// A column standing on y = 0, 2 m tall, `hw` half-width, with enough rings to bend smoothly.
fn column(hw: f32, h: f32) -> MeshData {
    let (x, y, z) = (Vec3::X * (2.0 * hw), Vec3::Y * h, Vec3::Z * (2.0 * hw));
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for (origin, u, v) in [
        (Vec3::new(-hw, 0.0, hw), x, y),
        (Vec3::new(hw, 0.0, -hw), -x, y),
        (Vec3::new(hw, 0.0, hw), -z, y),
        (Vec3::new(-hw, 0.0, -hw), z, y),
        (Vec3::new(-hw, h, hw), x, -z),
    ] {
        let (vs, is) = face_grid(origin, u, v, 20);
        let base = vertices.len() as u32;
        vertices.extend(vs);
        indices.extend(is.into_iter().map(|i| i + base));
    }
    MeshData { vertices, indices, has_lightmap_uv: false }
}

fn ground(half: f32) -> MeshData {
    let (vertices, indices) = face_grid(Vec3::new(-half, 0.0, -half), Vec3::Z * (2.0 * half), Vec3::X * (2.0 * half), 2);
    MeshData { vertices, indices, has_lightmap_uv: false }
}

#[derive(Clone, Copy)]
struct Cam {
    eye: Vec3,
    target: Vec3,
    up: Vec3,
}

impl Cam {
    fn project(&self, p: Vec3) -> (f32, f32) {
        let view = Mat4::look_at_rh(self.eye, self.target, self.up);
        let c = Mat4::perspective_rh(FOV, W as f32 / H as f32, 0.1, 100.0) * view * p.extend(1.0);
        ((c.x / c.w * 0.5 + 0.5) * W as f32, (0.5 - c.y / c.w * 0.5) * H as f32)
    }
}

struct Img(Vec<u8>);

impl Img {
    fn lum(&self, x: i32, y: i32) -> i32 {
        let (x, y) = (x.clamp(0, W as i32 - 1) as usize, y.clamp(0, H as i32 - 1) as usize);
        let i = (y * W as usize + x) * 4;
        self.0[i] as i32 + self.0[i + 1] as i32 + self.0[i + 2] as i32
    }

}

struct Rig {
    r: Renderer,
    source: ShaderSource,
    sheet: Vec<(String, Vec<u8>)>,
}

impl Rig {
    /// A sway material with these property values.
    fn material(&mut self, id: zestengine_render::ShaderId, set: &[(&str, &[f32])]) -> MaterialHandle {
        let mut values = self.source.defaults();
        for (name, v) in set {
            let p = self.source.prop(name).unwrap_or_else(|| panic!("no property {name}"));
            values[p.offset..p.offset + v.len()].copy_from_slice(v);
        }
        let h = self.r.create_material(&MaterialDesc { name: "sway".into(), features: MaterialFeatures { double_sided: true, ..Default::default() }, params: MaterialParams::default(), ..Default::default() }).unwrap();
        self.r.set_material_shader(h, Some(id));
        self.r.set_material_custom_data(h, self.source.push_data(&values));
        if self.source.ubo_size > 0 {
            self.r.set_material_custom_block(h, &self.source.uniform_bytes(&values));
        }
        let slots: Vec<_> = self
            .source
            .textures()
            .map(|p| {
                let ShaderPropKind::Texture { default, .. } = p.kind else { unreachable!() };
                let d = match default {
                    TextureDefault::White => CustomTextureDefault::White,
                    TextureDefault::Black => CustomTextureDefault::Black,
                    TextureDefault::Normal => CustomTextureDefault::Normal,
                    TextureDefault::Gray => CustomTextureDefault::Gray,
                };
                (None, d)
            })
            .collect();
        self.r.set_material_custom_textures(h, &slots);
        self.r.set_material_custom_keywords(h, self.source.keywords(&values));
        h
    }

    fn shot(&mut self, label: &str, cam: Cam, draws: &[DrawCmd], lights: &[LightInstance], ambient: f32, time: f32) -> Img {
        let camera = CameraData {
            view: Mat4::look_at_rh(cam.eye, cam.target, cam.up),
            proj: Mat4::perspective_rh(FOV, W as f32 / H as f32, 0.1, 100.0),
            position: cam.eye,
        };
        let input = FrameInput {
            clear_color: [0.55, 0.7, 0.9, 1.0],
            camera,
            light: LightData { ambient: [ambient; 3], intensity: 0.0, env_intensity: 0.0, ..LightData::default() },
            lights,
            capture_lights: &[],
            voxel_emitters: &[],
            camera_preview: None,
            model_preview: None,
            draw_skybox: false,
            shadow_pcf_texel: 0.0,
            shadow_bias_scale: 0.0,
            shadow_normal_offset: 0.0,
            shadow_distance: 60.0,
            shadow_cascades: 0,
            shadow_cascade_lambda: 0.0,
            time,
            water_time: 0.0,
            water: &[],
            draws,
            decals: &[],
            lightmap_preview: false,
            gi_debug: 0,
            render_viewport: false,
            viewport_extent: Some([W, H]),
            postfx: PostFx::default(),
            preview_postfx: PostFx::default(),
            reflection_probe: None,
            reflection_probes: [None; MAX_REFLECTION_PROBES],
            reflection_probe_b: None,
            reflection_primary: 0,
            fog: None,
            voxel_specular: false,
            planar: &[],
            camera_targets: &[],
            egui: None,
            world_ui: Vec::new(),
        };
        let mut px = None;
        for _ in 0..5 {
            self.r.render(&input).unwrap_or_else(|e| panic!("{label}: render {e:#}"));
            px = self.r.debug_read_viewport().unwrap_or_else(|e| panic!("{label}: readback {e:#}"));
        }
        let (w, h, px) = px.unwrap_or_else(|| panic!("{label}: no viewport target"));
        assert_eq!((w, h), (W, H));
        self.sheet.push((label.to_owned(), px.clone()));
        Img(px)
    }
}

fn draw(mesh: MeshHandle, material: MaterialHandle, transform: Mat4, casts: bool) -> DrawCmd {
    DrawCmd {
        gi_source: None,
        mesh,
        material,
        transform,
        highlight: 0.0,
        mesh_center: Vec3::new(0.0, 1.0, 0.0),
        bound_radius: 4.0,
        lightmap_layer: -1,
        lightmap_size: 0,
        static_geometry: false,
        planar_index: -1,
        occluded: false,
        shadow_casting: if casts { ShadowCasting::On } else { ShadowCasting::Off },
        instance_color: None,
        instance_param: None,
    }
}

fn run(rig: &mut Rig) {
    // The shipped file, through the same two calls the engine's shader library makes.
    let hook = vertex_hook_glsl(&rig.source, None).expect("the example has a vertex hook");
    let spirv = compile_shader(&rig.source, "zestengine_wind_sway").expect("fragment compiles");
    let id = rig.r.register_shader_with_vertex(&spirv, &hook).unwrap_or_else(|e| panic!("hook: {e:#}"));
    // A hook that does not compile is reported at registration, naming the section.
    let bad = format!("{hook}\nvoid broken() {{ not_a_function(); }}\n");
    let err = rig.r.register_shader_with_vertex(&spirv, &bad).err().map(|e| format!("{e:#}"));
    assert!(err.as_deref().is_some_and(|e| e.contains("vertex hook") && e.contains("not_a_function")), "bad hook error: {err:?}");

    let col = rig.r.upload_mesh(&column(0.1, 2.0)).unwrap();
    let floor = rig.r.upload_mesh(&ground(9.0)).unwrap();
    let grey = rig.r.create_material(&MaterialDesc { name: "ground".into(), params: MaterialParams { base_color: [0.75, 0.7, 0.6, 1.0], roughness: 1.0, ..Default::default() }, ..Default::default() }).unwrap();
    // sway 0.5 m, 2 rad/s, plant height 2 m, gust 0.5, wind along +x.
    let swaying = rig.material(id, &[("sway", &[0.5]), ("sway_speed", &[2.0]), ("sway_height", &[2.0]), ("sway_gust", &[0.5]), ("sway_dir", &[0.0])]);
    let calm = rig.material(id, &[("sway", &[0.0]), ("sway_height", &[2.0])]);

    // The shader's own formula for a vertex at the column's axis (x = z = 0, so phase 0).
    let top_offset = |t: f32, sway: f32, speed: f32, gust: f32| {
        let wave = (t * speed).sin() + gust * (t * speed * 2.3).sin();
        (wave * sway, -wave.abs() * sway * sway * 0.5)
    };
    let t = 0.5;
    let (dx, dy) = top_offset(t, 0.5, 2.0, 0.5);

    // ---- a row of columns, seen from the front: they sway, rooted at the ground
    let front = Cam { eye: Vec3::new(0.0, 1.2, 8.0), target: Vec3::new(0.0, 1.0, 0.0), up: Vec3::Y };
    let row: Vec<DrawCmd> = (-3..=3).map(|i| draw(col, swaying, Mat4::from_translation(Vec3::new(i as f32 * 1.1, 0.0, 0.0)), false)).collect();
    rig.shot("W00 row of columns at time 0", front, &row, &[], 0.8, 0.0);
    rig.shot("W01 the same row at time 0.5 s", front, &row, &[], 0.8, t);
    // The measurements use one column on its own (its neighbours lean into its window).
    let one = |m: MaterialHandle| vec![draw(col, m, Mat4::IDENTITY, false)];
    let rest = rig.shot("W02 one column at time 0", front, &one(swaying), &[], 0.8, 0.0);
    let moved = rig.shot("W03 the same column at time 0.5 s", front, &one(swaying), &[], 0.8, t);
    let calm_img = rig.shot("W04 sway 0: hook running, amplitude 0", front, &one(calm), &[], 0.8, t);
    // Centre column (x = 0): leaf centroid in a band near the top and one near the roots.
    let centroid = |img: &Img, y: f32| {
        let row = y.round() as i32;
        let (mut sum, mut n) = (0i64, 0i64);
        for x in (W as i32 / 2 - 24)..(W as i32 / 2 + 24) {
            if img.lum(x, row) > 150 {
                sum += x as i64;
                n += 1;
            }
        }
        (n > 0).then(|| sum as f32 / n as f32)
    };
    let probe_y = 1.6;
    let top_y = front.project(Vec3::new(0.0, probe_y, 0.0)).1;
    let root_y = front.project(Vec3::new(0.0, 0.15, 0.0)).1;
    let (a, b, c) = (centroid(&rest, top_y).expect("top at rest"), centroid(&moved, top_y).expect("top swayed"), centroid(&calm_img, top_y).expect("top calm"));
    let (ra, rb) = (centroid(&rest, root_y).expect("root at rest"), centroid(&moved, root_y).expect("root swayed"));
    // The lean of the vertex that shows at world height `probe_y`: the shader leans a vertex by
    // `dx * (y/2)^2` and lowers it by `dip * (y/2)^2` (solved for the object height by iteration).
    let mut y_obj = probe_y;
    for _ in 0..20 {
        y_obj = probe_y + (dy.abs() / 1.0) * (y_obj / 2.0).powi(2);
    }
    let lean = dx * (y_obj / 2.0).powi(2);
    let want = front.project(Vec3::new(lean, probe_y, 0.0)).0 - front.project(Vec3::new(0.0, probe_y, 0.0)).0;
    eprintln!("wind sway: top band centroid {a:.1} -> {b:.1} (shift {:.1}, formula {want:.1} px for {lean:.3} m, dip {dy:.3}); calm {c:.1}; root band {ra:.1} -> {rb:.1}", b - a);
    assert!((b - a - want).abs() <= 3.0, "top shift {} vs formula {want}", b - a);
    assert!((c - a).abs() <= 1.0, "amplitude 0 moved the column ({c} vs {a})");
    assert!((rb - ra).abs() <= 1.5, "the roots moved ({ra} -> {rb})");

    // ---- shadows sway with them: sun slanted toward -x, camera straight down
    let down = Cam { eye: Vec3::new(0.0, 9.0, 0.0), target: Vec3::ZERO, up: -Vec3::Z };
    let sun = [LightInstance {
        kind: LightKind::Directional,
        position: Vec3::ZERO,
        direction: Vec3::new(-0.6, -1.0, 0.0).normalize(),
        color: [1.0; 3],
        intensity: 3.0,
        range: 100.0,
        spot_inner_deg: 0.0,
        spot_outer_deg: 0.0,
        cast_shadows: true,
        soft_shadows: false, radius: 0.0,
        shadow_bias: 0.0,
        baked: false,
    }];
    let single = |m: MaterialHandle| vec![draw(col, m, Mat4::IDENTITY, true), draw(floor, grey, Mat4::IDENTITY, false)];
    let s0 = rig.shot("W05 shadow of one column at time 0", down, &single(swaying), &sun, 0.0, 0.0);
    let s1 = rig.shot("W06 shadow at time 0.5 s", down, &single(swaying), &sun, 0.0, t);
    let s_calm = rig.shot("W07 shadow with sway 0", down, &single(calm), &sun, 0.0, t);
    let row_y = down.project(Vec3::ZERO).1.round() as i32;
    let tip = |img: &Img| {
        let lit = img.lum(W as i32 - 4, row_y);
        (0..W as i32).find(|&x| img.lum(x, row_y) < lit / 3).unwrap_or(-1)
    };
    // The top (y = 2) shadows 1.2 m toward -x, minus the lean; its left edge is the column's -0.1.
    let want_tip = |lean: f32| down.project(Vec3::new(-0.1 + lean - 1.2, 0.0, 0.0)).0;
    let (t0, t1, tc) = (tip(&s0), tip(&s1), tip(&s_calm));
    eprintln!("wind sway shadow tips: time 0 {t0} (formula {:.1}), time 0.5 {t1} ({:.1}), calm {tc} ({:.1})", want_tip(0.0), want_tip(dx), want_tip(0.0));
    assert!((t1 as f32 - want_tip(dx)).abs() <= 5.0, "swayed shadow tip {t1} vs {}", want_tip(dx));
    assert!((tc as f32 - want_tip(0.0)).abs() <= 5.0, "calm shadow tip {tc} vs {}", want_tip(0.0));
    assert!((t0 as f32 - want_tip(0.0)).abs() <= 5.0, "shadow tip at time 0 {t0} vs {}", want_tip(0.0));
    assert!((t1 - t0).abs() >= 8, "the shadow did not sway ({t0} -> {t1})");
}

#[derive(Default)]
struct App {
    done: bool,
    error: Option<String>,
    sheet: Vec<(String, Vec<u8>)>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.done {
            return;
        }
        self.done = true;
        let attrs = Window::default_attributes().with_visible(false).with_inner_size(winit::dpi::LogicalSize::new(W as f64, H as f64));
        let Ok(window) = event_loop.create_window(attrs).map(Arc::new) else {
            self.error = Some("window".into());
            event_loop.exit();
            return;
        };
        match Renderer::new(window) {
            Ok(mut r) => {
                r.set_vsync(false);
                let source = parse_shader_source(include_str!("../../../shaders/zestengine_wind_sway.frag"), "zestengine_wind_sway").unwrap();
                let mut rig = Rig { r, source, sheet: Vec::new() };
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&mut rig)));
                self.sheet = std::mem::take(&mut rig.sheet);
                if let Err(e) = result {
                    self.error = Some(e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "panic".into()));
                }
            }
            Err(e) => self.error = Some(format!("Renderer::new: {e:#}")),
        }
        event_loop.exit();
    }

    fn window_event(&mut self, _: &ActiveEventLoop, _: WindowId, _: WindowEvent) {}
}

fn write_sheet(sheet: &[(String, Vec<u8>)]) {
    if std::env::var("ZESTENGINE_FX_SHEET").is_ok_and(|v| v == "0") || sheet.is_empty() {
        return;
    }
    let cols = 3usize;
    let rows = sheet.len().div_ceil(cols);
    let mut out = image::RgbaImage::from_pixel((cols as u32) * W, (rows as u32) * H, image::Rgba([24, 24, 24, 255]));
    for (i, (_, px)) in sheet.iter().enumerate() {
        let (ox, oy) = ((i % cols) as u32 * W, (i / cols) as u32 * H);
        for y in 0..H {
            for x in 0..W {
                let s = ((y * W + x) * 4) as usize;
                out.put_pixel(ox + x, oy + y, image::Rgba([px[s], px[s + 1], px[s + 2], 255]));
            }
        }
    }
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/design/wind-sway-contact-sheet.png");
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    out.save(&path).expect("write contact sheet");
    eprintln!("contact sheet ({cols} per row): {}", path.display());
    for (i, (label, _)) in sheet.iter().enumerate() {
        eprintln!("  [{} , {}] {label}", i / cols, i % cols);
    }
}

#[test]
fn wind_sway_vertex_hook_moves_the_surface_and_its_shadow() {
    if std::env::var("ZESTENGINE_TEST_GPU").is_ok_and(|v| v == "0") || std::env::var_os("DISPLAY").is_none() {
        eprintln!("[gpu test] no display / GPU disabled: skipped");
        return;
    }
    let _gpu = zestengine_render::cluster_headless::gpu_lock();
    let Ok(mut event_loop) = EventLoop::builder().with_x11().with_any_thread(true).build() else {
        eprintln!("[gpu test] no X11 event loop: skipped");
        return;
    };
    let mut app = App::default();
    event_loop.run_app_on_demand(&mut app).unwrap();
    write_sheet(&app.sheet);
    if let Some(e) = &app.error {
        panic!("{e}");
    }
}
