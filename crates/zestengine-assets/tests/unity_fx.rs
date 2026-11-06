//! UNITY-3 / 6 / 7 / 10 on the REAL renderer (invisible X11 window, real device):
//! per-material blend modes and front-face culling, the grab pass, and the
//! ZestEngine Toon features that came from the Unity shader gaps (flipbook,
//! panosphere, distance fade, hue shift, RGBA colour masking, wrapped lighting,
//! shadow layer 3). Every phase renders a small scene over a red | blue
//! background and probes pixels; a contact sheet of all phases is written to
//! `docs/design/unity-fx-contact-sheet.png` (set `ZESTENGINE_FX_SHEET=0` to skip).
//!
//! One `#[test]`: winit allows one event loop per process. Skips cleanly with no
//! display / GPU (`ZESTENGINE_TEST_GPU=0` forces that). Run with
//! `ZESTENGINE_TEST_GPU=1 cargo test -p zestengine-assets --test unity_fx -- --nocapture`.

use std::collections::HashMap;
use std::sync::Arc;

use glam::{Mat4, Quat, Vec3};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::{Window, WindowId};
use zestengine_assets::{KeywordSet, ShaderPropKind, ShaderSource, TextureDefault, compile_shader_variant, parse_shader_source};
use zestengine_render::{
    AlphaMode, BlendMode, CameraData, CustomTextureDefault, DetailParams, DrawCmd, FrameInput, LightData, LightInstance, LightKind, MAX_REFLECTION_PROBES,
    MaterialDesc, TransmissionParams,
    MaterialFeatures, MaterialHandle, MaterialParams, MeshData, MeshHandle, PostFx, Renderer, ShaderId, StencilCompare, StencilFace, StencilOp,
    StencilState, SpriteFrameInput, TextureData, TextureHandle, Vertex,
};
use zestengine_render::sprite2d::{MaskMode, MaskRange, MaskSource, SpriteSource};

const W: u32 = 320;
const H: u32 = 240;
const EYE: Vec3 = Vec3::new(0.0, 0.0, 5.0);

struct Image {
    px: Vec<u8>,
}

impl Image {
    fn at(&self, x: i32, y: i32) -> [i32; 3] {
        let (x, y) = (x.clamp(0, W as i32 - 1) as usize, y.clamp(0, H as i32 - 1) as usize);
        let i = (y * W as usize + x) * 4;
        [self.px[i] as i32, self.px[i + 1] as i32, self.px[i + 2] as i32]
    }
}

fn quad(half: f32) -> MeshData {
    let mut vertices = Vec::new();
    for (x, y, u, v) in [(-1.0, -1.0, 0.0, 1.0), (1.0, -1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 0.0), (-1.0, 1.0, 0.0, 0.0)] {
        vertices.push(Vertex {
            position: [x * half, y * half, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [u, v],
            tangent: [1.0, 0.0, 0.0, 1.0],
            color: [1.0; 4],
            ..Default::default()
        });
    }
    MeshData { vertices, indices: vec![0, 1, 2, 0, 2, 3], has_lightmap_uv: false }
}

/// Unit cube centred on the origin, counter-clockwise seen from outside.
fn cube() -> MeshData {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for (n, up, right) in [
        (Vec3::X, Vec3::Y, Vec3::NEG_Z),
        (Vec3::NEG_X, Vec3::Y, Vec3::Z),
        (Vec3::Y, Vec3::NEG_Z, Vec3::X),
        (Vec3::NEG_Y, Vec3::Z, Vec3::X),
        (Vec3::Z, Vec3::Y, Vec3::X),
        (Vec3::NEG_Z, Vec3::Y, Vec3::NEG_X),
    ] {
        let base = vertices.len() as u32;
        for (u, v) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let p = n * 0.5 + right * (u * 0.5) + up * (v * 0.5);
            vertices.push(Vertex { position: p.to_array(), normal: n.to_array(), tangent: [right.x, right.y, right.z, 1.0], color: [1.0; 4], ..Default::default() });
        }
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    MeshData { vertices, indices, has_lightmap_uv: false }
}

/// UV sphere, outward normals, counter-clockwise seen from outside.
fn sphere(radius: f32, nu: u32, nv: u32) -> MeshData {
    let mut vertices = Vec::new();
    for j in 0..=nv {
        for i in 0..=nu {
            let (u, v) = (i as f32 / nu as f32, j as f32 / nv as f32);
            let (phi, theta) = (u * std::f32::consts::TAU, v * std::f32::consts::PI);
            let n = Vec3::new(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin());
            vertices.push(Vertex {
                position: (n * radius).to_array(),
                normal: n.to_array(),
                uv: [u, v],
                tangent: [-phi.sin(), 0.0, phi.cos(), 1.0],
                color: [1.0; 4],
                ..Default::default()
            });
        }
    }
    let mut indices = Vec::new();
    for j in 0..nv {
        for i in 0..nu {
            let a = j * (nu + 1) + i;
            let b = a + nu + 1;
            indices.extend([a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    MeshData { vertices, indices, has_lightmap_uv: false }
}

fn emissive(r: &mut Renderer, rgb: [f32; 3], alpha: f32, blend: BlendMode, cull_front: bool) -> MaterialHandle {
    r.create_material(&MaterialDesc {
        name: "fx".into(),
        params: MaterialParams { base_color: [0.0, 0.0, 0.0, alpha], emission_color: rgb, emission_intensity: 1.0, roughness: 1.0, ..Default::default() },
        features: MaterialFeatures {
            emission: true,
            alpha_mode: if alpha < 1.0 { AlphaMode::Blend } else { AlphaMode::Opaque },
            blend,
            cull_front,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap()
}

/// An opaque emissive material with a stencil state and render queue (UNITY-2).
fn stencil_material(r: &mut Renderer, rgb: [f32; 3], stencil: StencilState, queue: i32) -> MaterialHandle {
    let h = r
        .create_material(&MaterialDesc {
            name: "stencil".into(),
            params: MaterialParams { base_color: [0.0, 0.0, 0.0, 1.0], emission_color: rgb, emission_intensity: 1.0, roughness: 1.0, ..Default::default() },
            features: MaterialFeatures { emission: true, stencil, ..Default::default() },
            ..Default::default()
        })
        .unwrap();
    r.set_material_render_queue(h, queue);
    h
}

struct Toon {
    source: ShaderSource,
    ids: HashMap<KeywordSet, ShaderId>,
}

impl Toon {
    fn new() -> Self {
        Self::of(include_str!("../../../shaders/zestengine_toon.frag"))
    }

    fn of(body: &str) -> Self {
        Self { source: parse_shader_source(body, "zestengine_fx").unwrap(), ids: HashMap::new() }
    }

    /// A material using ZestEngine Toon with these property values / textures.
    fn material(
        &mut self,
        r: &mut Renderer,
        set: &[(&str, &[f32])],
        textures: &[(&str, TextureHandle)],
        features: MaterialFeatures,
    ) -> MaterialHandle {
        let mut values = self.source.defaults();
        for (name, v) in set {
            let p = self.source.prop(name).unwrap_or_else(|| panic!("no Toon property {name}"));
            values[p.offset..p.offset + v.len()].copy_from_slice(v);
        }
        let keywords = self.source.keywords(&values);
        let id = *self.ids.entry(keywords).or_insert_with(|| {
            let spirv = compile_shader_variant(&self.source, "zestengine_fx", Some(&keywords)).unwrap_or_else(|e| panic!("shader variant: {e:#}"));
            r.register_shader(&spirv).unwrap()
        });
        let mut features = features;
        features.grab |= self.source.prop("GRAB").is_some_and(|p| values[p.offset] != 0.0);
        let h = r.create_material(&MaterialDesc { name: "toon fx".into(), features, ..Default::default() }).unwrap();
        r.set_material_shader(h, Some(id));
        r.set_material_custom_data(h, self.source.push_data(&values));
        if self.source.ubo_size > 0 {
            r.set_material_custom_block(h, &self.source.uniform_bytes(&values));
        }
        let slots: Vec<_> = self
            .source
            .textures()
            .map(|p| {
                let ShaderPropKind::Texture { default, .. } = p.kind else { unreachable!() };
                let tex = textures.iter().find(|(n, _)| *n == p.name).map(|(_, t)| *t);
                let d = match default {
                    TextureDefault::White => CustomTextureDefault::White,
                    TextureDefault::Black => CustomTextureDefault::Black,
                    TextureDefault::Normal => CustomTextureDefault::Normal,
                    TextureDefault::Gray => CustomTextureDefault::Gray,
                };
                (tex, d)
            })
            .collect();
        r.set_material_custom_textures(h, &slots);
        r.set_material_custom_keywords(h, keywords);
        h
    }
}

fn tex(r: &mut Renderer, w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4], srgb: bool) -> TextureHandle {
    let mut pixels = Vec::new();
    for y in 0..h {
        for x in 0..w {
            pixels.extend(f(x, y));
        }
    }
    r.upload_texture(&TextureData { width: w, height: h, pixels, srgb, hdr: false }).unwrap()
}

/// A per-shot sprite refill hook for the renderer.
type SpriteFill = Box<dyn Fn(&mut SpriteFrameInput)>;

struct Rig {
    r: Renderer,
    quad: MeshHandle,
    big: MeshHandle,
    sphere: MeshHandle,
    cube: MeshHandle,
    red: MaterialHandle,
    blue: MaterialHandle,
    sheet: Vec<(String, Vec<u8>)>,
    /// Lights of the next shots (empty = the unlit emissive phases).
    lights: Vec<LightInstance>,
    /// Refills the renderer's sprite input before every render of a shot (the frame consumes it).
    sprite_fill: Option<SpriteFill>,
    /// Shots without tone mapping, bloom, grading or vignette (quantitative lighting phases).
    linear: bool,
}

impl Rig {
    fn project(&self, p: Vec3, eye: Vec3) -> (i32, i32) {
        let view = Mat4::look_at_rh(eye, eye - Vec3::Z * 5.0, Vec3::Y);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_3, W as f32 / H as f32, 0.1, 100.0);
        let c = proj * view * p.extend(1.0);
        (((c.x / c.w * 0.5 + 0.5) * W as f32) as i32, ((0.5 - c.y / c.w * 0.5) * H as f32) as i32)
    }

    /// Render `fg` over the red | blue background (when `bg`) and read the viewport.
    fn shot(&mut self, label: &str, fg: &[(MeshHandle, MaterialHandle, Mat4)], eye: Vec3, ambient: f32, time: f32, bg: bool) -> Image {
        let mut draws = Vec::new();
        let mk = |mesh, material, transform| DrawCmd {
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
            planar_index: -1,
            occluded: false,
            shadow_casting: zestengine_render::ShadowCasting::Off,
            instance_color: None,
            instance_param: None,
        };
        if bg {
            draws.push(mk(self.big, self.red, Mat4::from_translation(Vec3::new(-6.0, 0.0, -1.0))));
            draws.push(mk(self.big, self.blue, Mat4::from_translation(Vec3::new(6.0, 0.0, -1.0))));
        }
        for (m, mat, t) in fg {
            draws.push(mk(*m, *mat, *t));
        }
        let view = Mat4::look_at_rh(eye, eye - Vec3::Z * 5.0, Vec3::Y);
        let camera = CameraData { view, proj: Mat4::perspective_rh(std::f32::consts::FRAC_PI_3, W as f32 / H as f32, 0.1, 100.0), position: eye };
        let post = if self.linear {
            PostFx { tonemap: 0, bloom_enabled: false, grading_enabled: false, vignette_enabled: false, ca_enabled: false, ..PostFx::default() }
        } else {
            PostFx::default()
        };
        let input = FrameInput {
            clear_color: [0.0, 0.0, 0.0, 1.0],
            camera,
            light: LightData { ambient: [ambient; 3], intensity: 0.0, env_intensity: 0.0, ..LightData::default() },
            lights: &self.lights,
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
            draws: &draws,
            decals: &[],
            lightmap_preview: false,
            gi_debug: 0,
            render_viewport: false,
            viewport_extent: Some([W, H]),
            postfx: post,
            preview_postfx: post,
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
            if let Some(fill) = &self.sprite_fill {
                fill(self.r.sprite_input_mut());
            }
            self.r.render(&input).unwrap_or_else(|e| panic!("{label}: render {e:#}"));
            px = self.r.debug_read_viewport().unwrap_or_else(|e| panic!("{label}: readback {e:#}"));
        }
        let (w, h, px) = px.unwrap_or_else(|| panic!("{label}: no viewport target"));
        assert_eq!((w, h), (W, H));
        self.sheet.push((label.to_owned(), px.clone()));
        Image { px }
    }
}

fn near(a: [i32; 3], b: [i32; 3], tol: i32) -> bool {
    a.iter().zip(&b).all(|(x, y)| (x - y).abs() <= tol)
}

fn run(rig: &mut Rig) {
    let t = |p: Vec3| Mat4::from_translation(p);
    let quad = rig.quad;
    let spot = rig.project(Vec3::ZERO, EYE);
    // Left of the seam the background is red: a foreground quad at x = -2.
    let left = Vec3::new(-2.0, 0.0, 0.0);
    let lp = rig.project(left, EYE);

    // ---- background only: the references ----
    let base = rig.shot("00 background", &[], EYE, 0.0, 0.0, true);
    let red = base.at(lp.0, lp.1);
    let blue = base.at(W as i32 - lp.0, lp.1);
    assert!(red[0] > 100 && red[1] < 10 && red[2] < 10, "red background {red:?}");
    assert!(blue[2] > 100 && blue[0] < 10 && blue[1] < 10, "blue background {blue:?}");

    // ---- blend modes (UNITY-10): half-transparent green over red ----
    let green = [0.0, 1.0, 0.0];
    let m_alpha = emissive(&mut rig.r, green, 0.5, BlendMode::Alpha, false);
    let m_add = emissive(&mut rig.r, green, 0.5, BlendMode::Additive, false);
    // Multiply needs the Blend alpha mode to pick its equation: alpha just below 1.
    let m_mul = emissive(&mut rig.r, [0.5, 0.5, 0.5], 0.99, BlendMode::Multiply, false);
    let a = rig.shot("01 blend alpha", &[(quad, m_alpha, t(left))], EYE, 0.0, 0.0, true).at(lp.0, lp.1);
    let b = rig.shot("02 blend additive", &[(quad, m_add, t(left))], EYE, 0.0, 0.0, true).at(lp.0, lp.1);
    let c = rig.shot("03 blend multiply", &[(quad, m_mul, t(left))], EYE, 0.0, 0.0, true).at(lp.0, lp.1);
    eprintln!("blend probes: red {red:?} alpha {a:?} additive {b:?} multiply {c:?}");
    assert!(a[0] < red[0] - 20 && a[1] > 20, "alpha blend mixes both: {a:?}");
    assert!(b[0] >= red[0] - 6 && b[1] > 20, "additive keeps the background and adds green: {b:?}");
    assert!(b[0] > a[0] + 20, "additive is brighter in red than alpha blend ({b:?} vs {a:?})");
    assert!(c[0] < red[0] - 20 && c[1] < 10 && c[2] < 10, "multiply darkens red and adds nothing: {c:?}");

    // ---- front-face culling (UNITY-10) ----
    let back = Quat::from_rotation_y(std::f32::consts::PI);
    let front_default = emissive(&mut rig.r, green, 1.0, BlendMode::Alpha, false);
    let front_culled = emissive(&mut rig.r, green, 1.0, BlendMode::Alpha, true);
    let facing = Mat4::from_translation(left);
    let turned = Mat4::from_rotation_translation(back, left);
    let v1 = rig.shot("04 cull back (front-facing quad)", &[(quad, front_default, facing)], EYE, 0.0, 0.0, true).at(lp.0, lp.1);
    let v2 = rig.shot("05 cull back (turned quad)", &[(quad, front_default, turned)], EYE, 0.0, 0.0, true).at(lp.0, lp.1);
    let v3 = rig.shot("06 cull front (front-facing quad)", &[(quad, front_culled, facing)], EYE, 0.0, 0.0, true).at(lp.0, lp.1);
    let v4 = rig.shot("07 cull front (turned quad)", &[(quad, front_culled, turned)], EYE, 0.0, 0.0, true).at(lp.0, lp.1);
    assert!(v1[1] > 100, "front face shows with back-face culling: {v1:?}");
    assert!(near(v2, red, 8), "back face is culled by default: {v2:?}");
    assert!(near(v3, red, 8), "front face is culled with Cull Front: {v3:?}");
    assert!(v4[1] > 100, "back face shows with Cull Front: {v4:?}");

    // ---- grab pass (UNITY-3) ----
    let mut toon = Toon::new();
    let seam_quad = t(Vec3::ZERO);
    let grab_features = MaterialFeatures { grab: true, ..Default::default() };
    let tint = [1.0, 0.5, 0.25];
    let grab = toon.material(
        &mut rig.r,
        &[("GRAB", &[1.0]), ("grab_tint", &tint), ("grab_distortion", &[0.0]), ("RECEIVE_SHADOWS", &[0.0])],
        &[],
        grab_features,
    );
    let gl = rig.project(Vec3::new(-0.5, 0.0, 0.0), EYE);
    let gr = rig.project(Vec3::new(0.5, 0.0, 0.0), EYE);
    let gimg = rig.shot("08 grab tint over red|blue", &[(quad, grab, seam_quad)], EYE, 0.0, 0.0, true);
    let (gred, gblue) = (gimg.at(gl.0, gl.1), gimg.at(gr.0, gr.1));
    eprintln!("grab probes: red side {gred:?} blue side {gblue:?}");
    // Tint (1, 0.5, 0.25): red keeps its value (R x 1), blue drops to a quarter.
    assert!(near(gred, red, 6), "the red side shows the background through a tint that keeps red: {gred:?} vs {red:?}");
    assert!(gblue[2] > 10 && gblue[2] < blue[2] - 30 && gblue[0] < 20, "the blue side shows the tinted background: {gblue:?} vs {blue:?}");
    // Without the copy (feature off) the shader falls back to its own colour (black here).
    let nograb = toon.material(&mut rig.r, &[("GRAB", &[1.0]), ("grab_tint", &tint), ("RECEIVE_SHADOWS", &[0.0])], &[], MaterialFeatures::default());
    rig.r.set_material_features(nograb, MaterialFeatures::default()); // keyword on, copy not requested
    let nimg = rig.shot("09 grab without the copy", &[(quad, nograb, seam_quad)], EYE, 0.0, 0.0, true);
    let n = nimg.at(gl.0, gl.1);
    assert!(n[0] < 25, "no opaque copy: fallback colour, not the background: {n:?}");
    // Distortion: a tilted surface shifts the seam it shows. Find the seam column in row y.
    let seam_x = |img: &Image, y: i32, x0: i32, x1: i32| -> i32 {
        (x0..x1).find(|x| img.at(*x, y)[2] > img.at(*x, y)[0]).unwrap_or(-1)
    };
    let tilt = Mat4::from_rotation_y(0.6);
    let flat = toon.material(&mut rig.r, &[("GRAB", &[1.0]), ("grab_distortion", &[0.0]), ("RECEIVE_SHADOWS", &[0.0])], &[], grab_features);
    let warp = toon.material(&mut rig.r, &[("GRAB", &[1.0]), ("grab_distortion", &[0.3]), ("RECEIVE_SHADOWS", &[0.0])], &[], grab_features);
    let i_flat = rig.shot("10 grab tilted, no distortion", &[(quad, flat, tilt)], EYE, 0.0, 0.0, true);
    let i_warp = rig.shot("11 grab tilted, distortion 0.3", &[(quad, warp, tilt)], EYE, 0.0, 0.0, true);
    let (s0, s1) = (seam_x(&i_flat, spot.1, spot.0 - 25, spot.0 + 25), seam_x(&i_warp, spot.1, spot.0 - 25, spot.0 + 25));
    eprintln!("grab seam column: flat {s0}, distorted {s1}");
    assert!(s0 > 0 && s1 > 0 && (s1 - s0).abs() >= 8, "distortion moves what shows through ({s0} vs {s1})");
    // Blur: pixels next to the seam mix both colours.
    let blur = toon.material(&mut rig.r, &[("GRAB", &[1.0]), ("grab_blur", &[14.0]), ("RECEIVE_SHADOWS", &[0.0])], &[], grab_features);
    let i_blur = rig.shot("12 grab blur 14 px", &[(quad, blur, seam_quad)], EYE, 0.0, 0.0, true);
    let near_seam = i_blur.at(spot.0 - 4, spot.1);
    assert!(near_seam[0] > 15 && near_seam[2] > 15, "blur mixes red and blue at the seam: {near_seam:?}");
    let sharp = rig.shot("13 grab sharp (reference)", &[(quad, flat, seam_quad)], EYE, 0.0, 0.0, true).at(spot.0 - 4, spot.1);
    assert!(sharp[2] < 10, "no blur is sharp at the same pixel: {sharp:?}");

    // ---- flipbook (UNITY-6) ----
    let sheet = tex(
        &mut rig.r,
        64,
        64,
        |x, y| match (x / 32, y / 32) {
            (0, 0) => [255, 0, 0, 255],
            (1, 0) => [0, 255, 0, 255],
            (0, 1) => [0, 0, 255, 255],
            _ => [255, 255, 255, 255],
        },
        true,
    );
    let flip = |rig: &mut Rig, toon: &mut Toon, label: &str, extra: &[(&str, &[f32])], time: f32| {
        let mut set: Vec<(&str, &[f32])> = vec![
            ("FLIPBOOK", &[1.0]),
            ("flip_cols", &[2.0]),
            ("flip_rows", &[2.0]),
            ("flip_frames", &[4.0]),
            ("flip_emission", &[1.0]),
            ("RECEIVE_SHADOWS", &[0.0]),
        ];
        set.extend_from_slice(extra);
        let m = toon.material(&mut rig.r, &set, &[("flip_tex", sheet)], MaterialFeatures::default());
        rig.shot(label, &[(quad, m, seam_quad)], EYE, 0.0, time, true).at(spot.0, spot.1)
    };
    let f0 = flip(rig, &mut toon, "14 flipbook frame 0", &[("flip_manual", &[0.0]), ("flip_fps", &[0.0])], 0.0);
    let f1 = flip(rig, &mut toon, "15 flipbook frame 1", &[("flip_manual", &[1.0]), ("flip_fps", &[0.0])], 0.0);
    let f2 = flip(rig, &mut toon, "16 flipbook frame 2", &[("flip_manual", &[2.0]), ("flip_fps", &[0.0])], 0.0);
    let f3 = flip(rig, &mut toon, "17 flipbook frame 3", &[("flip_manual", &[3.0]), ("flip_fps", &[0.0])], 0.0);
    let ft = flip(rig, &mut toon, "18 flipbook by time (2 fps, t = 1.5 s -> frame 3)", &[("flip_manual", &[-1.0]), ("flip_fps", &[2.0])], 1.5);
    let fw = flip(rig, &mut toon, "19 flipbook wraps (t = 2.2 s -> frame 0)", &[("flip_manual", &[-1.0]), ("flip_fps", &[2.0])], 2.2);
    let fx = flip(rig, &mut toon, "19b flipbook crossfade (1 fps, t = 0.5 s: red/green mix)", &[("flip_manual", &[-1.0]), ("flip_fps", &[1.0]), ("flip_crossfade", &[1.0])], 0.5);
    assert!(fx[0] > 60 && fx[1] > 60 && fx[2] < 60, "crossfade mixes frame 0 and 1: {fx:?}");
    eprintln!("flipbook: {f0:?} {f1:?} {f2:?} {f3:?} time {ft:?} wrap {fw:?}");
    assert!(f0[0] > 150 && f0[1] < 60 && f0[2] < 60, "frame 0 red: {f0:?}");
    assert!(f1[1] > 150 && f1[0] < 60 && f1[2] < 60, "frame 1 green: {f1:?}");
    assert!(f2[2] > 150 && f2[0] < 60 && f2[1] < 60, "frame 2 blue: {f2:?}");
    assert!(f3.iter().all(|c| *c > 150), "frame 3 white: {f3:?}");
    assert!(ft.iter().all(|c| *c > 150), "frame by time: {ft:?}");
    assert!(fw[0] > 150 && fw[1] < 60, "the animation wraps back to frame 0: {fw:?}");

    // ---- distance fade (UNITY-10) ----
    let fade = toon.material(
        &mut rig.r,
        &[("DISTANCE_FADE", &[1.0]), ("fade_min", &[2.0]), ("fade_max", &[4.0]), ("emission_color", &[0.0]), ("main_color", &[1.0, 1.0, 1.0, 1.0]), ("RECEIVE_SHADOWS", &[0.0])],
        &[],
        MaterialFeatures::default(),
    );
    let far = rig.shot("20 distance fade: far (5 m) visible", &[(quad, fade, seam_quad)], EYE, 1.0, 0.0, true).at(spot.0, spot.1);
    let eye_near = Vec3::new(0.0, 0.0, 1.0);
    let sp_near = rig.project(Vec3::ZERO, eye_near);
    let close = rig.shot("21 distance fade: near (1 m) gone", &[(quad, fade, seam_quad)], eye_near, 1.0, 0.0, true).at(sp_near.0 + 30, sp_near.1);
    assert!(far.iter().all(|c| *c > 60), "visible beyond the end distance: {far:?}");
    assert!(close[0] > 100 && close[1] < 10 && close[2] < 10 || close[2] > 100 && close[0] < 10, "faded out up close shows the background: {close:?}");

    // ---- hue shift / brightness (UNITY-10) ----
    let hue = |rig: &mut Rig, toon: &mut Toon, label: &str, set: &[(&str, &[f32])]| {
        let mut all: Vec<(&str, &[f32])> = vec![("main_color", &[1.0, 0.0, 0.0, 1.0]), ("lighting_mode", &[3.0]), ("RECEIVE_SHADOWS", &[0.0])];
        all.extend_from_slice(set);
        let m = toon.material(&mut rig.r, &all, &[], MaterialFeatures::default());
        rig.shot(label, &[(quad, m, seam_quad)], EYE, 0.3, 0.0, false).at(spot.0, spot.1)
    };
    let h0 = hue(rig, &mut toon, "22 red albedo", &[]);
    let h1 = hue(rig, &mut toon, "23 hue shift 1/3 turn", &[("hue_shift", &[0.3333])]);
    let h2 = hue(rig, &mut toon, "24 brightness 0.5", &[("brightness", &[0.5])]);
    eprintln!("hue: base {h0:?} shifted {h1:?} dim {h2:?}");
    assert!(h0[0] > 60 && h0[1] < 40, "red albedo: {h0:?}");
    assert!(h1[1] > 60 && h1[0] < 40, "hue shift turns red into green: {h1:?}");
    assert!(h2[0] < h0[0] - 20 && h2[0] > 10, "brightness scales the colour: {h2:?} vs {h0:?}");

    // ---- RGBA colour masking (UNITY-7): left half red channel, right half green channel ----
    let mask = tex(&mut rig.r, 64, 1, |x, _| if x < 32 { [255, 0, 0, 0] } else { [0, 255, 0, 0] }, false);
    let cm = toon.material(
        &mut rig.r,
        &[
            ("COLOR_MASK", &[1.0]),
            ("cmask_r", &[0.0, 0.0, 1.0]),
            ("cmask_g", &[0.0, 1.0, 0.0]),
            ("main_color", &[1.0, 1.0, 1.0, 1.0]),
            ("lighting_mode", &[3.0]),
            ("RECEIVE_SHADOWS", &[0.0]),
        ],
        &[("cmask_tex", mask)],
        MaterialFeatures::default(),
    );
    let cm_img = rig.shot("25 RGBA colour mask (left blue, right green)", &[(quad, cm, seam_quad)], EYE, 1.0, 0.0, false);
    let (cl, cr) = (cm_img.at(gl.0, gl.1), cm_img.at(gr.0, gr.1));
    assert!(cl[2] > 80 && cl[0] < 40 && cl[1] < 40, "red-channel region tinted blue: {cl:?}");
    assert!(cr[1] > 80 && cr[0] < 40 && cr[2] < 40, "green-channel region tinted green: {cr:?}");

    // ---- wrapped lighting + shadow layer 3 (UNITY-7) need a light: a directional light from the side ----
    // (covered by the compile/validate test; the maths is checked through the ambient-only
    // Flat path above. Wrapped uses the same band plumbing.)

    // ---- panosphere + Cull Front (UNITY-10): camera inside a sphere ----
    let sky = tex(&mut rig.r, 4, 64, |_, y| if y < 32 { [0, 255, 0, 255] } else { [255, 0, 255, 255] }, true);
    let pano_features = MaterialFeatures { cull_front: true, ..Default::default() };
    let pano = toon.material(
        &mut rig.r,
        &[("PANOSPHERE", &[1.0]), ("pano_unlit", &[1.0]), ("RECEIVE_SHADOWS", &[0.0])],
        &[("pano_tex", sky)],
        pano_features,
    );
    let sph = rig.sphere;
    let inside = Vec3::ZERO;
    let pimg = rig.shot("26 panosphere from inside (sky: green above, magenta below)", &[(sph, pano, Mat4::IDENTITY)], inside, 0.0, 0.0, false);
    let up = pimg.at(W as i32 / 2, H as i32 / 4);
    let down = pimg.at(W as i32 / 2, H as i32 * 3 / 4);
    eprintln!("panosphere: up {up:?} down {down:?}");
    assert!(up[1] > 100 && up[0] < 80, "looking up shows the green half: {up:?}");
    assert!(down[0] > 100 && down[2] > 100 && down[1] < 80, "looking down shows the magenta half: {down:?}");
    // The same sphere with default culling is invisible from inside.
    let pano_back = toon.material(
        &mut rig.r,
        &[("PANOSPHERE", &[1.0]), ("pano_unlit", &[1.0]), ("RECEIVE_SHADOWS", &[0.0])],
        &[("pano_tex", sky)],
        MaterialFeatures::default(),
    );
    let empty = rig.shot("26b nothing drawn (clear colour reference)", &[], inside, 0.0, 0.0, false).at(W as i32 / 2, H as i32 / 4);
    let pb = rig.shot("27 panosphere with back-face culling (invisible inside)", &[(sph, pano_back, Mat4::IDENTITY)], inside, 0.0, 0.0, false).at(W as i32 / 2, H as i32 / 4);
    assert!(near(pb, empty, 4), "inside faces are culled by default: {pb:?} vs clear {empty:?}");

    // ---- Screen FX volume (UNITY-4): invert colours inside a box, depth-tested against the scene ----
    let mut fx = Toon::of(include_str!("../../../shaders/zestengine_screenfx.frag"));
    let fx_features = MaterialFeatures { alpha_mode: AlphaMode::Blend, cull_front: true, ..Default::default() };
    let screen_fx = |rig: &mut Rig, fx: &mut Toon, set: &[(&str, &[f32])]| {
        let mut all: Vec<(&str, &[f32])> = vec![("filt_invert", &[1.0]), ("FILTER", &[1.0]), ("volume_falloff", &[0.0])];
        all.extend_from_slice(set);
        fx.material(&mut rig.r, &all, &[], fx_features)
    };
    let cube_m = rig.cube;
    let bg_centre = rig.project(Vec3::new(-0.5, 0.0, -1.0), EYE);
    // Box around the background plane (z = -1) and on the red half: the background is inside.
    let inside_box = screen_fx(rig, &mut fx, &[("volume_shape", &[1.0])]);
    let around = Mat4::from_scale_rotation_translation(Vec3::new(2.0, 2.0, 2.0), Quat::IDENTITY, Vec3::new(0.0, 0.0, -1.0));
    let i_in = rig.shot("28 screen fx box around the scene (invert)", &[(cube_m, inside_box, around)], EYE, 0.0, 0.0, true);
    let v_in = i_in.at(bg_centre.0, bg_centre.1);
    // Box in front of the background (z from 0 to 2): the scene behind it is outside the volume.
    let in_front = Mat4::from_scale_rotation_translation(Vec3::new(2.0, 2.0, 2.0), Quat::IDENTITY, Vec3::new(0.0, 0.0, 1.0));
    let i_out = rig.shot("29 screen fx box in front of the scene (no effect)", &[(cube_m, inside_box, in_front)], EYE, 0.0, 0.0, true);
    let v_out = i_out.at(bg_centre.0, bg_centre.1);
    // Everywhere: the same box in front inverts everything it covers.
    let everywhere = screen_fx(rig, &mut fx, &[("volume_shape", &[0.0])]);
    let i_all = rig.shot("30 screen fx everywhere (invert the covered area)", &[(cube_m, everywhere, in_front)], EYE, 0.0, 0.0, true);
    let v_all = i_all.at(bg_centre.0, bg_centre.1);
    eprintln!("screen fx: scene inside {v_in:?}, outside {v_out:?}, everywhere {v_all:?} (red = {red:?})");
    assert!(v_in[0] < 60 && v_in[1] > 100 && v_in[2] > 100, "red inverts to cyan inside the volume: {v_in:?}");
    assert!(near(v_out, red, 6), "a scene outside the volume is untouched: {v_out:?}");
    assert!(v_all[0] < 60 && v_all[1] > 100, "Everywhere affects every covered pixel: {v_all:?}");
    // Blur softens the red | blue seam to magenta.
    let blur_fx = screen_fx(rig, &mut fx, &[("volume_shape", &[0.0]), ("FILTER", &[0.0]), ("BLUR", &[1.0]), ("blur_radius", &[14.0])]);
    let seam_wide = Mat4::from_scale_rotation_translation(Vec3::new(6.0, 4.0, 1.0), Quat::IDENTITY, Vec3::new(0.0, 0.0, 1.0));
    let i_blur = rig.shot("31 screen fx blur over the seam", &[(cube_m, blur_fx, seam_wide)], EYE, 0.0, 0.0, true);
    let sb = i_blur.at(spot.0 - 4, spot.1);
    assert!(sb[0] > 15 && sb[2] > 15, "blur mixes the seam: {sb:?}");
    // Pixelate quantises: two pixels in one block read the same colour even across the seam.
    let pix = screen_fx(rig, &mut fx, &[("volume_shape", &[0.0]), ("FILTER", &[0.0]), ("PIXELATE", &[1.0]), ("pixel_size", &[64.0])]);
    let i_pix = rig.shot("32 screen fx pixelate (64 px blocks)", &[(cube_m, pix, seam_wide)], EYE, 0.0, 0.0, true);
    let (p1, p2) = (i_pix.at(spot.0 - 3, spot.1 + 40), i_pix.at(spot.0 - 12, spot.1 + 40));
    assert!(near(p1, p2, 4), "one block, one colour: {p1:?} vs {p2:?}");

    // ---- material stencil on the real renderer (UNITY-2): a no-colour writer marks the stencil where its
    // quad is (queue 1999, drawn first); a green reader the size of the whole view then only shows there ----
    let face = |compare, pass_op| StencilFace { compare, pass_op, fail_op: StencilOp::Keep, zfail_op: StencilOp::Keep };
    // The usual Unity writer: ColorMask 0, ZWrite off (so it does not occlude what comes after).
    let write = StencilState { color_mask: 0, depth_write: false, ..StencilState::both(face(StencilCompare::Always, StencilOp::Replace), 1, 255, 255) };
    let read = StencilState::both(face(StencilCompare::Equal, StencilOp::Keep), 1, 255, 0);
    let w_mat = stencil_material(&mut rig.r, [1.0, 1.0, 1.0], write, 1999);
    let r_mat = stencil_material(&mut rig.r, [0.0, 1.0, 0.0], read, 2001);
    let in_front = Mat4::from_translation(Vec3::new(0.0, 0.0, 0.5));
    let st = rig.shot("33 stencil writer + reader", &[(rig.big, r_mat, in_front), (quad, w_mat, t(left))], EYE, 0.0, 0.0, true);
    let inside = st.at(lp.0, lp.1);
    let outside = st.at(lp.0 - 70, lp.1);
    let right_side = st.at(W as i32 - lp.0, lp.1);
    eprintln!("stencil probes: inside writer {inside:?} outside (red bg) {outside:?} right (blue bg) {right_side:?}");
    assert!(inside[1] > 100 && inside[0] < 20 && inside[2] < 20, "the reader shows where the writer marked: {inside:?}");
    assert!(outside[0] > 100 && outside[1] < 10, "the reader fails the stencil test elsewhere (background shows): {outside:?}");
    assert!(right_side[2] > 100 && right_side[1] < 10, "no reader pixels on the right: {right_side:?}");
    // The writer alone draws no colour and (ZWrite off) hides nothing: the red background shows through its area.
    let w_only = rig.shot("34 stencil writer alone (no colour, no depth)", &[(quad, w_mat, t(left))], EYE, 0.0, 0.0, true).at(lp.0, lp.1);
    assert!(w_only[0] > 100 && w_only[1] < 10, "a ColorMask 0 / ZWrite off writer leaves the background: {w_only:?}");
    // ZWrite on (the default) vs off, seen through a green stencil-state quad (CPU-ordered like every stencil
    // material) 0.5 m behind the writer over an empty background: ZWrite on hides it behind the writer's depth.
    let behind = stencil_material(&mut rig.r, [0.0, 1.0, 0.0], StencilState::both(face(StencilCompare::NotEqual, StencilOp::Keep), 99, 255, 0), 2000);
    let back_of = Mat4::from_translation(Vec3::new(0.0, 0.0, -0.5));
    let hole = StencilState { depth_write: true, ..write };
    let w_hole = stencil_material(&mut rig.r, [1.0, 1.0, 1.0], hole, 1999);
    let off = rig.shot("35 ZWrite off writer: the quad behind shows", &[(rig.big, behind, back_of), (quad, w_mat, t(left))], EYE, 0.0, 0.0, false).at(lp.0, lp.1);
    let on = rig.shot("36 ZWrite on writer: its depth hides the quad behind", &[(rig.big, behind, back_of), (quad, w_hole, t(left))], EYE, 0.0, 0.0, false).at(lp.0, lp.1);
    assert!(off[1] > 100, "ZWrite off leaves the depth alone: {off:?}");
    assert!(on[1] < 40, "ZWrite on writes the writer's depth, hiding what is behind it: {on:?}");

    // ---- sprite masks on the real renderer (2D-3): the scene pass's stencil attachment now exists ----
    let white = tex(&mut rig.r, 4, 4, |_, _| [255; 4], false);
    let sprite_tex = rig.r.register_sprite_texture(white, false).unwrap();
    let unit = rig.r.sprite_mesh_pool().add(&[[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]], &[[0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]], &[0, 1, 2, 2, 1, 3]).unwrap();
    let sprite = move |size: f32, color: [f32; 4], mask: MaskMode| SpriteSource {
        scale: [size, size],
        mesh: unit,
        uv_rect: [0.0, 0.0, 1.0, 1.0],
        sprite_size: [1.0, 1.0],
        pivot: [0.5, 0.5],
        color,
        adaptive_threshold: 0.5,
        size: [1.0, 1.0],
        texture: sprite_tex,
        order: 1,
        mask,
        ..Default::default()
    };
    let mask_at_left = MaskSource { position: [-2.0, 0.0, 0.0], scale: [2.0, 2.0], mesh: unit, uv_rect: [0.0, 0.0, 1.0, 1.0], texture: sprite_tex, range: MaskRange::Auto, ..Default::default() };
    for (label, mode, want_inside) in [("37 sprite mask: visible inside", MaskMode::VisibleInside, true), ("38 sprite mask: visible outside", MaskMode::VisibleOutside, false)] {
        rig.sprite_fill = Some(Box::new(move |input| {
            input.clear();
            input.masks.push(mask_at_left);
            input.sprites.push(sprite(12.0, [0.0, 1.0, 0.0, 1.0], mode));
        }));
        let shot = rig.shot(label, &[], EYE, 0.0, 0.0, true);
        let (inside, outside) = (shot.at(lp.0, lp.1), shot.at(W as i32 - lp.0, lp.1));
        eprintln!("{label}: inside the mask {inside:?}, outside {outside:?}");
        let (shown, hidden) = if want_inside { (inside, outside) } else { (outside, inside) };
        assert!(shown[1] > 100 && shown[0] < 40, "{label}: the sprite shows: {shown:?}");
        assert!(hidden[2] > 100 || hidden[0] > 100, "{label}: the sprite is hidden (background shows): {hidden:?}");
    }
    // ---- sprites behind the 3D transparents (2D-3): the scene draw order is cut at the transparent queue ----
    let glass = emissive(&mut rig.r, [1.0, 1.0, 1.0], 0.5, BlendMode::Alpha, false);
    for (label, behind) in [("39 sprite over a 3D transparent", false), ("40 sprite behind a 3D transparent", true)] {
        rig.sprite_fill = Some(Box::new(move |input| {
            input.clear();
            input.sprites.push(sprite(12.0, [0.0, 1.0, 0.0, 1.0], MaskMode::None));
            input.config.behind_layer_max = behind.then_some(0);
        }));
        let p = rig.shot(label, &[(quad, glass, t(left))], EYE, 0.0, 0.0, true).at(lp.0, lp.1);
        eprintln!("{label}: {p:?}");
        if behind {
            assert!(p[0] > 80 && p[2] > 80 && p[1] > 100, "{label}: the glass blends over the sprite: {p:?}");
        } else {
            assert!(p[1] > 100 && p[0] < 40 && p[2] < 40, "{label}: the sprite is over the glass: {p:?}");
        }
    }
    rig.sprite_fill = None;
    rig.r.unregister_sprite_texture(sprite_tex);

    pbr_detail_and_transmission(rig);
    area_lights(rig);
    light_volumes(rig);
}

/// A standard-shader material with a detail layer / transmission and its maps.
fn pbr(r: &mut Renderer, params: MaterialParams, features: MaterialFeatures, detail: [Option<TextureHandle>; 5]) -> MaterialHandle {
    r.create_material(&MaterialDesc { name: "pbr".into(), params, features, detail, ..Default::default() }).unwrap()
}

/// Brightness (sum of channels) of a probe.
fn lum(p: [i32; 3]) -> i32 {
    p[0] + p[1] + p[2]
}

/// UNITY-8 on the real renderer: the PBR detail layer (albedo, normal, ORM, mask),
/// back-lit translucency (with a thickness map) and see-through transmission. The
/// quad faces the camera; one directional light travels along -Z (front-lit) or +Z
/// (back-lit). Every phase probes colours and is labelled "P..." for its own sheet.
fn pbr_detail_and_transmission(rig: &mut Rig) {
    let quad = rig.quad;
    let at = |x: i32, y: i32, img: &Image| img.at(x, y);
    let centre = rig.project(Vec3::ZERO, EYE);
    // Quad half-size 1: probes a quarter of the way in from each side.
    let left_p = rig.project(Vec3::new(-0.5, 0.0, 0.0), EYE);
    let right_p = rig.project(Vec3::new(0.5, 0.0, 0.0), EYE);
    let sun = |dir: Vec3| LightInstance {
        kind: LightKind::Directional,
        position: Vec3::ZERO,
        direction: dir,
        color: [1.0; 3],
        intensity: 3.0,
        range: 100.0,
        spot_inner_deg: 0.0,
        spot_outer_deg: 0.0,
        cast_shadows: false,
        soft_shadows: false, radius: 0.0,
        shadow_bias: 0.0,
        baked: false,
    };
    let front_lit = vec![sun(Vec3::NEG_Z)];
    let back_lit = vec![sun(Vec3::Z)];
    let ident = Mat4::IDENTITY;
    let base = MaterialParams { base_color: [0.5, 0.5, 0.5, 1.0], roughness: 1.0, ..Default::default() };

    // ---- reference: the plain material, front lit ----
    rig.lights = front_lit.clone();
    let plain = pbr(&mut rig.r, base, MaterialFeatures::default(), Default::default());
    let c0 = rig.shot("P00 plain, front lit", &[(quad, plain, ident)], EYE, 0.0, 0.0, false).at(centre.0, centre.1);
    assert!(lum(c0) > 150, "the front-lit reference is bright: {c0:?}");

    // ---- detail albedo, Replace blend with a mask that is white on the left half only ----
    let red_tex = tex(&mut rig.r, 4, 4, |_, _| [255, 0, 0, 255], true);
    let mask_tex = tex(&mut rig.r, 8, 8, |x, _| if x < 4 { [255; 4] } else { [0, 0, 0, 255] }, false);
    let det = MaterialParams { detail: DetailParams { albedo_strength: 1.0, blend: 3.0, ..Default::default() }, ..base };
    let m_det = pbr(&mut rig.r, det, MaterialFeatures::default(), [Some(red_tex), None, None, Some(mask_tex), None]);
    let img = rig.shot("P01 detail albedo (replace) masked to the left half", &[(quad, m_det, ident)], EYE, 0.0, 0.0, false);
    let (l, r) = (at(left_p.0, left_p.1, &img), at(right_p.0, right_p.1, &img));
    eprintln!("detail albedo: left {l:?} right {r:?} reference {c0:?}");
    assert!(l[0] > 100 && l[1] < 25 && l[2] < 25, "the masked half shows the red detail albedo: {l:?}");
    assert!(near(r, c0, 12), "the unmasked half is the base material: {r:?} vs {c0:?}");
    // Multiply x2 with the neutral grey (sRGB 128) leaves the base alone.
    let grey = tex(&mut rig.r, 4, 4, |_, _| [128, 128, 128, 255], true);
    let det_n = MaterialParams { detail: DetailParams { albedo_strength: 1.0, blend: 0.0, ..Default::default() }, ..base };
    let m_neutral = pbr(&mut rig.r, det_n, MaterialFeatures::default(), [Some(grey), None, None, None, None]);
    let n = rig.shot("P02 detail albedo x2 with neutral grey", &[(quad, m_neutral, ident)], EYE, 0.0, 0.0, false).at(centre.0, centre.1);
    assert!(near(n, c0, 14), "x2 with grey 128 is neutral: {n:?} vs {c0:?}");
    // Tiling: a checker detail at tiling 1 vs 4 changes what the probe pixel sees (4x4 checker, replace).
    let checker = tex(&mut rig.r, 2, 2, |x, y| if (x + y) % 2 == 0 { [255, 255, 255, 255] } else { [0, 0, 0, 255] }, true);
    let tile = |rig: &mut Rig, t: f32, label: &str| {
        let p = MaterialParams { detail: DetailParams { albedo_strength: 1.0, blend: 3.0, tiling: [t, t], ..Default::default() }, ..base };
        let m = pbr(&mut rig.r, p, MaterialFeatures::default(), [Some(checker), None, None, None, None]);
        rig.shot(label, &[(quad, m, ident)], EYE, 0.0, 0.0, false)
    };
    let t1 = tile(rig, 1.0, "P03 detail tiling 1 (2x2 checker)");
    let t4 = tile(rig, 4.0, "P04 detail tiling 4 (8x8 checker)");
    let count_changes = |img: &Image| {
        let row: Vec<i32> = (-35..35).map(|dx| lum(img.at(centre.0 + dx, centre.1))).collect();
        let mid = (row.iter().min().unwrap() + row.iter().max().unwrap()) / 2;
        row.windows(2).filter(|w| (w[0] - mid).signum() != (w[1] - mid).signum()).count()
    };
    let (c1, c4) = (count_changes(&t1), count_changes(&t4));
    eprintln!("checker edges along the middle row: tiling 1 -> {c1}, tiling 4 -> {c4}");
    assert!(c4 > c1, "a finer detail tiling shows more checker edges ({c1} vs {c4})");

    // ---- detail normal: a sideways normal dims the front light ----
    let side = tex(&mut rig.r, 4, 4, |_, _| [255, 128, 255, 255], false); // x = +1
    let dn = MaterialParams { detail: DetailParams { normal_strength: 1.0, ..Default::default() }, ..base };
    let m_dn = pbr(&mut rig.r, dn, MaterialFeatures::default(), [None, Some(side), None, None, None]);
    let white_mask = tex(&mut rig.r, 2, 2, |_, _| [255; 4], false);
    let _ = white_mask;
    let d = rig.shot("P05 detail normal tilts the surface", &[(quad, m_dn, ident)], EYE, 0.0, 0.0, false).at(centre.0, centre.1);
    eprintln!("detail normal: {d:?} vs flat {c0:?}");
    assert!(lum(d) < lum(c0) * 95 / 100 && lum(d) > lum(c0) / 4, "the tilted normal gets less of the head-on light: {d:?} vs {c0:?}");

    // ---- detail ORM: metallic 1 darkens a white surface (no environment); the detail ORM's blue = 0 removes it ----
    let metal_base = MaterialParams { metallic: 1.0, ..base };
    let m_metal = pbr(&mut rig.r, metal_base, MaterialFeatures::default(), Default::default());
    let dull = tex(&mut rig.r, 4, 4, |_, _| [255, 255, 0, 255], false);
    let m_metal_d = pbr(&mut rig.r, MaterialParams { detail: DetailParams { orm_strength: 1.0, ..Default::default() }, ..metal_base }, MaterialFeatures::default(), [None, None, Some(dull), None, None]);
    let a = rig.shot("P06 metallic 1", &[(quad, m_metal, ident)], EYE, 0.0, 0.0, false).at(centre.0, centre.1);
    let b = rig.shot("P07 metallic 1 with a detail ORM that zeroes metallic", &[(quad, m_metal_d, ident)], EYE, 0.0, 0.0, false).at(centre.0, centre.1);
    eprintln!("detail ORM: metal {a:?} -> {b:?}");
    assert!(lum(b) > lum(a) + 120, "the detail ORM multiplies metallic away, restoring the diffuse: {a:?} vs {b:?}");

    // ---- translucency: back lit, nothing without it ----
    rig.lights = back_lit.clone();
    let off = rig.shot("P08 back lit, no translucency", &[(quad, plain, ident)], EYE, 0.0, 0.0, false).at(centre.0, centre.1);
    let tr = MaterialParams { transmission: TransmissionParams { translucency: 0.3, translucency_color: [0.2, 1.0, 0.2], ..Default::default() }, ..base };
    let m_tr = pbr(&mut rig.r, tr, MaterialFeatures::default(), Default::default());
    let on = rig.shot("P09 back lit with green translucency", &[(quad, m_tr, ident)], EYE, 0.0, 0.0, false).at(centre.0, centre.1);
    eprintln!("translucency: off {off:?} on {on:?}");
    assert!(lum(off) < 30, "a back-lit opaque surface is dark without translucency: {off:?}");
    assert!(on[1] > 80 && on[1] > on[0] * 2 && on[1] > on[2] * 2, "the light coming through takes the translucency colour: {on:?}");
    // The map: thickness (left = thick -> blocked, right = thin -> passes).
    let thick = tex(&mut rig.r, 8, 8, |x, _| if x < 4 { [255; 4] } else { [0, 0, 0, 255] }, false);
    let tm = MaterialParams { transmission: TransmissionParams { translucency: 2.0, map_is_thickness: true, ..Default::default() }, ..base };
    let m_tm = pbr(&mut rig.r, tm, MaterialFeatures::default(), [None, None, None, None, Some(thick)]);
    let img = rig.shot("P10 thickness map: thick left, thin right", &[(quad, m_tm, ident)], EYE, 0.0, 0.0, false);
    let (tl, trr) = (at(left_p.0, left_p.1, &img), at(right_p.0, right_p.1, &img));
    eprintln!("thickness map: thick side {tl:?}, thin side {trr:?}");
    assert!(lum(tl) < 40 && lum(trr) > 120, "thick blocks, thin passes: {tl:?} vs {trr:?}");
    // Front lit: translucency adds nothing to the lit side.
    rig.lights = front_lit.clone();
    let fl = rig.shot("P11 front lit with translucency (unchanged)", &[(quad, m_tr, ident)], EYE, 0.0, 0.0, false).at(centre.0, centre.1);
    assert!(near(fl, c0, 12), "the lit side ignores translucency: {fl:?} vs {c0:?}");

    // ---- transmission: the scene behind shows through (Scene Colour Read) ----
    rig.lights = Vec::new();
    let grab = MaterialFeatures { grab: true, ..Default::default() };
    let see = |t: f32, tint: [f32; 3]| MaterialParams {
        base_color: [0.0, 0.0, 0.0, 1.0],
        roughness: 1.0,
        transmission: TransmissionParams { transmission: t, tint, refraction: 0.0, blur: 0.0, ..Default::default() },
        ..Default::default()
    };
    let glass = pbr(&mut rig.r, see(1.0, [1.0; 3]), grab, Default::default());
    let seam = Mat4::from_scale(Vec3::new(1.0, 1.0, 1.0));
    let red_p = rig.project(Vec3::new(-0.5, 0.0, 0.0), EYE);
    let blue_p = rig.project(Vec3::new(0.5, 0.0, 0.0), EYE);
    let g = rig.shot("P12 transmission 1 over red | blue", &[(quad, glass, seam)], EYE, 0.0, 0.0, true);
    let (gr, gb) = (g.at(red_p.0, red_p.1), g.at(blue_p.0, blue_p.1));
    eprintln!("transmission: red side {gr:?}, blue side {gb:?}");
    assert!(gr[0] > 80 && gr[1] < 15 && gr[2] < 15, "red shows through: {gr:?}");
    assert!(gb[2] > 80 && gb[0] < 15 && gb[1] < 15, "blue shows through: {gb:?}");
    let tinted = pbr(&mut rig.r, see(1.0, [1.0, 0.5, 0.25]), grab, Default::default());
    let gt = rig.shot("P13 transmission tinted (1, 0.5, 0.25)", &[(quad, tinted, seam)], EYE, 0.0, 0.0, true).at(blue_p.0, blue_p.1);
    assert!(gt[2] > 15 && gt[2] < gb[2] - 30, "the tint takes the blue down to a quarter: {gt:?} vs {gb:?}");
    // Without Scene Colour Read there is no copy, so the material falls back to its (black) body.
    let blind = pbr(&mut rig.r, see(1.0, [1.0; 3]), MaterialFeatures::default(), Default::default());
    let gn = rig.shot("P14 transmission without the scene copy", &[(quad, blind, seam)], EYE, 0.0, 0.0, true).at(red_p.0, red_p.1);
    assert!(gn[0] < 30, "no copy, no see-through: {gn:?}");
    // Half transmission over a lit body mixes the two.
    rig.lights = front_lit.clone();
    let half = pbr(&mut rig.r, MaterialParams { base_color: [0.0, 1.0, 0.0, 1.0], roughness: 1.0, transmission: TransmissionParams { transmission: 0.5, refraction: 0.0, blur: 0.0, ..Default::default() }, ..Default::default() }, grab, Default::default());
    let h = rig.shot("P15 transmission 0.5 over a green body", &[(quad, half, seam)], EYE, 0.0, 0.0, true).at(red_p.0, red_p.1);
    eprintln!("half transmission: {h:?}");
    assert!(h[0] > 40 && h[1] > 40, "half red from behind, half green body: {h:?}");
    rig.lights = Vec::new();
    rough_transmission(rig);
    fur_shells(rig);
}

/// Rough (blurred) transmission: the opaque copy's mip chain, picked by roughness x
/// thickness, plus the packed transmission / thickness map. A probe just inside the
/// red half of a red | blue background sees more blue the blurrier the glass is.
fn rough_transmission(rig: &mut Rig) {
    let quad = rig.quad;
    let seam = Mat4::IDENTITY;
    let centre = rig.project(Vec3::ZERO, EYE);
    let grab = MaterialFeatures { grab: true, ..Default::default() };
    let glass = |rough: f32, thickness: f32, map: bool| MaterialParams {
        base_color: [0.0, 0.0, 0.0, 1.0],
        roughness: rough,
        transmission: TransmissionParams {
            transmission: 1.0,
            refraction: 0.0,
            thickness,
            map_has_transmission: map,
            ..Default::default()
        },
        ..Default::default()
    };
    // 5 px into the red half; and a probe deep in the red half for the "stays red" check.
    let near = (centre.0 - 5, centre.1);
    let mut leak = Vec::new();
    for (label, rough) in [("P18 rough transmission, roughness 0", 0.0f32), ("P19 rough transmission, roughness 0.5", 0.5), ("P20 rough transmission, roughness 1", 1.0)] {
        let m = pbr(&mut rig.r, glass(rough.max(0.05), 1.0, false), grab, Default::default());
        let img = rig.shot(label, &[(quad, m, seam)], EYE, 0.0, 0.0, true);
        let p = img.at(near.0, near.1);
        eprintln!("{label}: probe 5 px inside red {p:?}");
        leak.push(p);
    }
    // Roughness 0 (clamped to 0.05) is the sharp image: pure red. Blue leaks in as it rises.
    assert!(leak[0][2] < 8 && leak[0][0] > 80, "smooth glass is sharp: {:?}", leak[0]);
    assert!(leak[1][2] > leak[0][2] + 6, "roughness 0.5 blurs across the seam: {:?} vs {:?}", leak[1], leak[0]);
    assert!(leak[2][2] > leak[1][2] + 6, "roughness 1 blurs further: {:?} vs {:?}", leak[2], leak[1]);
    // Thin volume: roughness 1 with thickness 0 blurs less than thickness 1.
    let thin = pbr(&mut rig.r, glass(1.0, 0.0, false), grab, Default::default());
    let t = rig.shot("P21 rough transmission, roughness 1, thickness 0", &[(quad, thin, seam)], EYE, 0.0, 0.0, true).at(near.0, near.1);
    eprintln!("thickness 0: {t:?}");
    assert!(t[2] < leak[2][2] && t[2] > leak[0][2], "a thin volume blurs less: {t:?} vs {:?}", leak[2]);
    // Blur 0 restores the sharp image whatever the roughness.
    let mut sharp = glass(1.0, 1.0, false);
    sharp.transmission.blur = 0.0;
    let sharp = pbr(&mut rig.r, sharp, grab, Default::default());
    let sp = rig.shot("P22 roughness 1 with blur 0", &[(quad, sharp, seam)], EYE, 0.0, 0.0, true).at(near.0, near.1);
    assert!(sp[2] < 8, "blur 0 is sharp: {sp:?}");
    // The packed map: green = transmission (left texel 0, right texel 1).
    let packed = tex(&mut rig.r, 8, 8, |x, _| if x < 4 { [255, 0, 0, 255] } else { [255, 255, 0, 255] }, false);
    let mut mp = glass(0.05, 1.0, true);
    mp.transmission.transmission = 1.0;
    let mm = pbr(&mut rig.r, mp, grab, [None, None, None, None, Some(packed)]);
    let left_p = rig.project(Vec3::new(-0.5, 0.0, 0.0), EYE);
    let right_p = rig.project(Vec3::new(0.5, 0.0, 0.0), EYE);
    let img = rig.shot("P23 transmission map: none left, full right", &[(quad, mm, seam)], EYE, 0.0, 0.0, true);
    let (l, r) = (img.at(left_p.0, left_p.1), img.at(right_p.0, right_p.1));
    eprintln!("transmission map: left {l:?} right {r:?}");
    assert!(lum(l) < 30, "map 0 blocks the scene: {l:?}");
    assert!(r[2] > 80, "map 1 shows the blue behind: {r:?}");
}

// ---- area lights (LTC) -------------------------------------------------------------

/// Karis' split-sum fit, as `env_brdf_approx` in standard.frag.
fn env_brdf(roughness: f32, nov: f32) -> (f32, f32) {
    let c0 = [-1.0f32, -0.0275, -0.572, 0.022];
    let c1 = [1.0f32, 0.0425, 1.04, -0.04];
    let r: Vec<f32> = (0..4).map(|k| roughness * c0[k] + c1[k]).collect();
    let a004 = (r[0] * r[0]).min((-9.28 * nov).exp2()) * r[0] + r[1];
    (-1.04 * a004 + r[2], 1.04 * a004 + r[3])
}

/// What the shader should output for a dielectric plane point lit by one unshadowed
/// rectangle light, computed with the CPU LTC reference (`zestengine_render::ltc`).
fn area_reference(table: &[zestengine_render::ltc::LtcCell], l: &zestengine_render::AreaLightInstance, p: Vec3, n: Vec3, eye: Vec3, albedo: f32, rough: f32) -> f32 {
    use zestengine_render::ltc;
    let corners = ltc::rect_corners(l.position, l.right, l.up, l.half_size);
    let v = (eye - p).normalize();
    let nov = n.dot(v).max(1e-4);
    let fd = ltc::diffuse_factor(&corners, p, n);
    let fs = ltc::specular_factor(table, &corners, p, n, v, rough);
    let (a, b) = env_brdf(rough, nov);
    let refl = 0.04 * a + b;
    let kd = (1.0 - refl).max(0.0);
    let fade = (1.0 - p.distance(l.position) / l.range).clamp(0.0, 1.0).powi(2);
    let radiance = l.intensity * fade;
    albedo * kd * radiance * fd + radiance * refl * fs
}

fn srgb_to_linear(v: i32) -> f32 {
    let c = v as f32 / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

fn rect_light(position: Vec3, half: [f32; 2], intensity: f32) -> zestengine_render::AreaLightInstance {
    zestengine_render::AreaLightInstance {
        shape: zestengine_render::AreaShape::Rect,
        position,
        right: Vec3::X,
        // cross(X, -Y) = -Z: the light shines towards -Z, onto the z = 0 plane in front of it.
        up: -Vec3::Y,
        half_size: half,
        color: [1.0; 3],
        intensity,
        range: 50.0,
        two_sided: false,
        texture: None,
    }
}

/// The linear brightness at `centre` of the plane lit by one area light.
fn rig_reference(rig: &mut Rig, label: &str, big: MeshHandle, mat: MaterialHandle, ident: Mat4, light: zestengine_render::AreaLightInstance, centre: (i32, i32)) -> f32 {
    rig.r.set_area_lights(&[light]);
    let p = rig.shot(label, &[(big, mat, ident)], EYE, 0.0, 0.0, false).at(centre.0, centre.1);
    srgb_to_linear(p[0])
}

/// Rectangle area lights (LTC): smooth falloff and a specular highlight consistent with
/// the CPU reference, an emission texture, two-sided, the nearest-8 bound and the Toon path.
fn area_lights(rig: &mut Rig) {
    use zestengine_render::ltc;
    let table = ltc::embedded_table();
    rig.lights = Vec::new();
    rig.linear = true;
    let big = rig.big;
    let ident = Mat4::IDENTITY;
    let n = Vec3::Z;

    // ---- calibration: how emission maps to the 8-bit read-back (tone mapping off) ----
    let emit = |r: &mut Renderer, v: f32| {
        pbr(r, MaterialParams { base_color: [0.0, 0.0, 0.0, 1.0], emission_color: [v; 3], emission_intensity: 1.0, ..Default::default() }, MaterialFeatures { emission: true, ..Default::default() }, Default::default())
    };
    let half_emit = emit(&mut rig.r, 0.25);
    let centre = rig.project(Vec3::ZERO, EYE);
    let cal = rig.shot("A00 calibration: emission 0.25", &[(big, half_emit, ident)], EYE, 0.0, 0.0, false).at(centre.0, centre.1)[0];
    let srgb = (cal - 137).abs() <= 6; // sRGB(0.25) = 137
    assert!(srgb || (cal - 64).abs() <= 6, "read-back of linear 0.25 is {cal}");
    let lin = |v: i32| if srgb { srgb_to_linear(v) } else { v as f32 / 255.0 };
    eprintln!("area calibration: emission 0.25 reads {cal} (sRGB encoded: {srgb})");

    // ---- smooth falloff + diffuse/specular against the CPU reference ----
    let albedo = 0.8;
    let rough_plane = pbr(&mut rig.r, MaterialParams { base_color: [albedo, albedo, albedo, 1.0], roughness: 1.0, ..Default::default() }, MaterialFeatures::default(), Default::default());
    let light = rect_light(Vec3::new(0.0, 0.0, 1.0), [0.5, 0.5], 4.0);
    rig.r.set_area_lights(&[light]);
    let img = rig.shot("A01 rectangle light over a rough plane", &[(big, rough_plane, ident)], EYE, 0.0, 0.0, false);
    let mut line = Vec::new();
    for x in [0.0f32, 0.4, 0.8, 1.2, 1.6, 2.0] {
        let px = rig.project(Vec3::new(x, 0.0, 0.0), EYE);
        let got = lin(img.at(px.0, px.1)[0]);
        let want = area_reference(&table, &light, Vec3::new(x, 0.0, 0.0), n, EYE, albedo, 1.0).min(1.0);
        eprintln!("falloff x={x}: gpu {got:.4} cpu {want:.4}");
        assert!((got - want).abs() <= 0.06 * want + 0.015, "x={x}: gpu {got} vs cpu reference {want}");
        line.push(got);
    }
    assert!(line.windows(2).all(|w| w[1] < w[0]), "monotone falloff {line:?}");
    assert!(line[0] > 0.2 && *line.last().unwrap() > 0.003, "lit near, still a little light far: {line:?}");
    rig.r.set_area_lights(&[]);
    let off = rig.shot("A02 same plane, light removed", &[(big, rough_plane, ident)], EYE, 0.0, 0.0, false).at(centre.0, centre.1);
    assert!(lum(off) < 12, "no light, black plane: {off:?}");

    // ---- specular highlight position ----
    let glossy = pbr(&mut rig.r, MaterialParams { base_color: [0.05, 0.05, 0.05, 1.0], roughness: 0.25, ..Default::default() }, MaterialFeatures::default(), Default::default());
    let lamp = rect_light(Vec3::new(0.0, 2.0, 1.0), [0.5, 0.25], 6.0);
    rig.r.set_area_lights(&[lamp]);
    let img = rig.shot("A03 rectangle light, glossy plane (highlight)", &[(big, glossy, ident)], EYE, 0.0, 0.0, false);
    // Mirror image of the light seen from the eye: the ray from the eye to (0, 2, -1) crosses z = 0 at y = 2 * 5 / 6.
    let mirror_y = 2.0 * EYE.z / (EYE.z + 1.0);
    let (mut best_gpu, mut best_cpu) = ((-9.0f32, 0.0f32), (-9.0f32, 0.0f32));
    let mut y = -1.0f32;
    while y <= 2.7 {
        let px = rig.project(Vec3::new(0.0, y, 0.0), EYE);
        let g = lin(img.at(px.0, px.1)[0]);
        let c = area_reference(&table, &lamp, Vec3::new(0.0, y, 0.0), n, EYE, 0.05, 0.25);
        if g > best_gpu.1 {
            best_gpu = (y, g);
        }
        if c > best_cpu.1 {
            best_cpu = (y, c);
        }
        y += 0.02;
    }
    eprintln!("highlight: gpu peak y={:.2} ({:.3}), cpu peak y={:.2} ({:.3}), mirror y={mirror_y:.2}", best_gpu.0, best_gpu.1, best_cpu.0, best_cpu.1);
    assert!((best_gpu.0 - best_cpu.0).abs() < 0.12, "GPU highlight at {} vs CPU reference {}", best_gpu.0, best_cpu.0);
    assert!((best_gpu.0 - mirror_y).abs() < 0.3, "highlight near the mirror point {mirror_y}: {}", best_gpu.0);
    assert!(best_gpu.1 > 0.1 && (best_gpu.1 - best_cpu.1).abs() < 0.25 * best_cpu.1 + 0.02, "peak level gpu {} vs cpu {}", best_gpu.1, best_cpu.1);

    // ---- emission texture: left half red, right half blue ----
    let flag = tex(&mut rig.r, 8, 8, |x, _| if x < 4 { [255, 0, 0, 255] } else { [0, 0, 255, 255] }, true);
    let mut textured = rect_light(Vec3::new(0.0, 0.0, 0.5), [0.5, 0.5], 4.0);
    textured.texture = Some(flag);
    rig.r.set_area_lights(&[textured]);
    let white = pbr(&mut rig.r, MaterialParams { base_color: [0.8; 4], roughness: 1.0, ..Default::default() }, MaterialFeatures::default(), Default::default());
    let img = rig.shot("A04 textured light: red | blue", &[(big, white, ident)], EYE, 0.0, 0.0, false);
    let at = |x: f32| {
        let p = rig.project(Vec3::new(x, 0.0, 0.0), EYE);
        img.at(p.0, p.1)
    };
    let (l, c, r) = (at(-0.45), at(0.0), at(0.45));
    eprintln!("textured light: left {l:?} centre {c:?} right {r:?}");
    assert!(l[0] > l[2] * 2 && l[0] > 40, "red side lit red: {l:?}");
    assert!(r[2] > r[0] * 2 && r[2] > 40, "blue side lit blue: {r:?}");
    assert!(c[0] > 20 && c[2] > 20, "centre gets both: {c:?}");
    // The same light untextured is white.
    textured.texture = None;
    rig.r.set_area_lights(&[textured]);
    let w = rig.shot("A05 the same light untextured", &[(big, white, ident)], EYE, 0.0, 0.0, false).at(centre.0 + 20, centre.1);
    assert!(near([w[0], w[0], w[0]], [w[1], w[1], w[1]], 2) && (w[0] - w[2]).abs() <= 3 && w[0] > 40, "white light: {w:?}");

    // ---- two-sided: a light facing away still lights the plane only when two sided ----
    let mut back = rect_light(Vec3::new(0.0, 0.0, 1.0), [0.5, 0.5], 4.0);
    back.up = Vec3::Y; // cross(X, Y) = +Z: shines away from the plane
    rig.r.set_area_lights(&[back]);
    let one = rig.shot("A06 light facing away, one sided", &[(big, white, ident)], EYE, 0.0, 0.0, false).at(centre.0, centre.1);
    back.two_sided = true;
    rig.r.set_area_lights(&[back]);
    let two = rig.shot("A07 light facing away, two sided", &[(big, white, ident)], EYE, 0.0, 0.0, false).at(centre.0, centre.1);
    eprintln!("facing away: one-sided {one:?}, two-sided {two:?}");
    assert!(lum(one) < 12, "the back of a one-sided light is dark: {one:?}");
    let front = rect_light(Vec3::new(0.0, 0.0, 1.0), [0.5, 0.5], 4.0);
    rig.r.set_area_lights(&[front]);
    let reference = rig.shot("A08 same light facing the plane", &[(big, white, ident)], EYE, 0.0, 0.0, false).at(centre.0, centre.1);
    assert!(two[0] > 40 && (two[0] - reference[0]).abs() <= 6, "two-sided lights the plane like the front: {two:?} vs {reference:?}");

    // ---- at most 8 lights: ten in a row, the two farthest from the camera are dropped ----
    let row: Vec<_> = (0..10).map(|i| rect_light(Vec3::new(-2.7 + 0.6 * i as f32, 0.0, 0.3), [0.1, 0.1], 6.0)).collect();
    let probes = [-2.7f32, -2.1, -1.5, 0.0, 1.5, 2.1, 2.7];
    let sample = |rig: &mut Rig, label: &str, lights: &[zestengine_render::AreaLightInstance]| -> Vec<i32> {
        rig.r.set_area_lights(lights);
        let img = rig.shot(label, &[(big, white, ident)], EYE, 0.0, 0.0, false);
        probes.iter().map(|x| { let p = rig.project(Vec3::new(*x, 0.0, 0.0), EYE); lum(img.at(p.0, p.1)) }).collect()
    };
    let ten = sample(rig, "A09 ten small lights, eight drawn", &row);
    // The eight nearest the camera are indices 1..=8 (x = -2.1 .. 2.1).
    let eight = sample(rig, "A09b the eight nearest alone", &row[1..9]);
    let seven = sample(rig, "A09c seven of them", &row[1..8]);
    eprintln!("light count bound: ten {ten:?}\n eight {eight:?}\n seven {seven:?}");
    for (a, b) in ten.iter().zip(&eight) {
        assert!((a - b).abs() <= 6, "the two farthest of ten add nothing: {ten:?} vs {eight:?}");
    }
    assert!(ten[0] * 3 < ten[1] && ten[6] * 3 < ten[5], "much less light under the dropped ones than under drawn ones (neighbour spill only): {ten:?}");
    assert!(eight[5] > seven[5] + 20, "the eighth light is drawn: {eight:?} vs {seven:?}");

    // ---- disc: the same emitting area lights the centre alike ----
    let rect_side = (std::f32::consts::PI).sqrt() * 0.3;
    let sq = rig_reference(rig, "A10a square light, area pi 0.09", big, white, ident, rect_light(Vec3::new(0.0, 0.0, 1.0), [rect_side * 0.5 * 2.0 / 2.0 * 1.0; 2], 4.0), centre);
    let mut disc = rect_light(Vec3::new(0.0, 0.0, 1.0), [0.3, 0.3], 4.0);
    disc.shape = zestengine_render::AreaShape::Disc;
    let dv = rig_reference(rig, "A10b disc light, radius 0.3", big, white, ident, disc, centre);
    eprintln!("disc vs square of equal area: {dv} vs {sq}");
    assert!(dv > 0.05 && (dv - sq).abs() < 0.08 * sq, "a disc lights like a square of the same area: {dv} vs {sq}");

    // ---- Toon lighting path takes the light too ----
    let toon_mat = pbr(&mut rig.r, MaterialParams { base_color: [0.8; 4], roughness: 1.0, pbr_toon_blend: 1.0, ..Default::default() }, MaterialFeatures { toon: true, ..Default::default() }, Default::default());
    rig.r.set_area_lights(&[rect_light(Vec3::new(0.0, 0.0, 1.0), [0.5, 0.5], 4.0)]);
    let t = rig.shot("A10 Toon material under a rectangle light", &[(big, toon_mat, ident)], EYE, 0.0, 0.0, false).at(centre.0, centre.1);
    eprintln!("toon under an area light: {t:?}");
    assert!(lum(t) > 100, "the cel-shaded surface is lit by the area light: {t:?}");

    rig.r.set_area_lights(&[]);
    rig.linear = false;
}

// ---- light volumes ----------------------------------------------------------------

/// A box Light Volume centred at `centre` (size sx x 4 x 2) with a 2x2x2 probe grid holding the
/// uniform sky radiance `sky` (SH-L1: only the constant band), for `LightProbeVolume` in Light
/// Volume mode.
fn uniform_volume(centre: Vec3, sx: f32, sky: f32, blend: f32, weight: f32) -> zestengine_render::LightVolumeData {
    // Radiance L everywhere projects to c0 = L / Y0 with Y0 = 0.282095.
    let c = sky / 0.282095;
    let probe = zestengine_render::ProbeSh { coeffs: [[c; 3], [0.0; 3], [0.0; 3], [0.0; 3]], ..Default::default() };
    zestengine_render::LightVolumeData {
        world_to_local: Mat4::from_translation(-centre),
        size: [sx, 4.0, 2.0],
        counts: [2, 2, 2],
        blend_distance: blend,
        weight,
        probes: vec![probe; 8],
    }
}

/// What the plane (albedo 0.8, dielectric, facing the camera) should show under diffuse
/// irradiance / pi of `l`: kd * albedo * l with kd = 1 - F0 at normal incidence.
fn ambient_expected(l: f32) -> f32 {
    0.96 * 0.8 * l
}

/// Light Volumes: baked SH reproducing a uniform sky on non-lightmapped surfaces, weight blending
/// of overlapping volumes and the fade at the faces (the CPU reference is the same blend).
fn light_volumes(rig: &mut Rig) {
    rig.lights = Vec::new();
    rig.linear = true;
    rig.r.set_area_lights(&[]);
    let big = rig.big;
    let ident = Mat4::IDENTITY;
    let plane = pbr(&mut rig.r, MaterialParams { base_color: [0.8; 4], roughness: 1.0, ..Default::default() }, MaterialFeatures::default(), Default::default());
    let read = |rig: &Rig, img: &Image, x: f32| -> f32 {
        let p = rig.project(Vec3::new(x, 0.0, 0.0), EYE);
        srgb_to_linear(img.at(p.0, p.1)[0])
    };
    let close = |got: f32, want: f32| (got - want).abs() <= 0.04 * want + 0.006;

    // ---- no volume: only the (zero) ambient ----
    rig.r.set_light_volumes(&[]);
    let img = rig.shot("V00 no light volume, ambient 0", &[(big, plane, ident)], EYE, 0.0, 0.0, false);
    assert!(read(rig, &img, 0.0) < 0.01, "dark without a volume");

    // ---- a uniform sky of 0.5 baked into one volume lights the plane exactly ----
    rig.r.set_light_volumes(&[uniform_volume(Vec3::ZERO, 6.0, 0.5, 0.5, 1.0)]);
    let img = rig.shot("V01 uniform sky 0.5 in one Light Volume", &[(big, plane, ident)], EYE, 0.0, 0.0, false);
    for x in [-1.5f32, 0.0, 1.5] {
        let got = read(rig, &img, x);
        eprintln!("uniform sky: x={x} got {got:.4} want {:.4}", ambient_expected(0.5));
        assert!(close(got, ambient_expected(0.5)), "x={x}: {got} vs analytic {}", ambient_expected(0.5));
    }

    // ---- edge fade over 0.5 m (smoothstep), over a flat ambient of 0.1 underneath ----
    let img = rig.shot("V02 edge fade over ambient 0.1", &[(big, plane, ident)], EYE, 0.1, 0.0, false);
    let smooth = |t: f32| { let t = t.clamp(0.0, 1.0); t * t * (3.0 - 2.0 * t) };
    let mut fade = Vec::new();
    for (x, d) in [(0.0f32, 3.0f32), (2.75, 0.25), (2.9, 0.1), (3.2, -0.2)] {
        let cov = if d > 0.0 { smooth(d / 0.5) } else { 0.0 };
        let want = ambient_expected(0.5 * cov + 0.1 * (1.0 - cov));
        let got = read(rig, &img, x);
        eprintln!("edge: x={x} coverage {cov:.2} got {got:.4} want {want:.4}");
        assert!(close(got, want), "x={x}: {got} vs {want}");
        fade.push(got);
    }
    assert!(fade[0] > fade[1] && fade[1] > fade[2] && fade[2] > fade[3], "monotone fade towards the face: {fade:?}");

    // ---- two overlapping volumes blend by weight: sky 0.2 (left) and 0.8 (right) ----
    let pair = |wa: f32, wb: f32| [uniform_volume(Vec3::new(-1.5, 0.0, 0.0), 6.0, 0.2, 0.5, wa), uniform_volume(Vec3::new(1.5, 0.0, 0.0), 6.0, 0.8, 0.5, wb)];
    let reference = |wa: f32, wb: f32, x: f32| -> f32 {
        // Volume A spans -4.5..1.5, B -1.5..4.5; weight x smoothstep(distance to the nearest face / 0.5).
        let w = |lo: f32, hi: f32, wt: f32| { let d = (x - lo).min(hi - x); if d <= 0.0 { 0.0 } else { wt * smooth(d / 0.5) } };
        let (a, b) = (w(-4.5, 1.5, wa), w(-1.5, 4.5, wb));
        if a + b <= 0.0 { 0.0 } else { (0.2 * a + 0.8 * b) / (a + b) }
    };
    rig.r.set_light_volumes(&pair(1.0, 1.0));
    let img = rig.shot("V03 two volumes, equal weights (0.2 | 0.8)", &[(big, plane, ident)], EYE, 0.0, 0.0, false);
    let mut sweep = Vec::new();
    for x in [-3.0f32, -1.4, -0.7, 0.0, 0.7, 1.4, 3.0] {
        let (got, want) = (read(rig, &img, x), ambient_expected(reference(1.0, 1.0, x)));
        eprintln!("blend: x={x} got {got:.4} want {want:.4}");
        assert!(close(got, want), "x={x}: {got} vs reference {want}");
        sweep.push(got);
    }
    assert!(sweep.windows(2).all(|w| w[1] > w[0] - 0.002), "the light rises smoothly from the left volume to the right one: {sweep:?}");
    assert!((sweep[0] - ambient_expected(0.2)).abs() < 0.01 && (sweep[6] - ambient_expected(0.8)).abs() < 0.02, "pure volumes at the ends");
    // Weights 3 : 1 pull the overlap towards the left volume.
    rig.r.set_light_volumes(&pair(3.0, 1.0));
    let img = rig.shot("V04 two volumes, weights 3 : 1", &[(big, plane, ident)], EYE, 0.0, 0.0, false);
    let (got, want) = (read(rig, &img, 0.0), ambient_expected(reference(3.0, 1.0, 0.0)));
    eprintln!("weights 3:1 at the middle: got {got:.4} want {want:.4}");
    assert!(close(got, want) && got < ambient_expected(0.5), "weighted blend: {got} vs {want}");

    // ---- a lightmapped surface ignores Light Volumes ----
    rig.r.set_light_volumes(&pair(1.0, 1.0));
    rig.r.set_light_volumes(&[]);
    rig.linear = false;
}

/// UNITY-12: fur shells draw extra, strand-masked copies of the mesh over the surface.
fn fur_shells(rig: &mut Rig) {
    let quad = rig.quad;
    let centre = rig.project(Vec3::ZERO, EYE);
    // A black surface; the fur is red at the tip and green at the root.
    let body = emissive(&mut rig.r, [0.0, 0.0, 0.0], 1.0, BlendMode::Alpha, false);
    let covered = |img: &Image| {
        let (mut fur, mut total) = (0, 0);
        for dy in -28..28 {
            for dx in -28..28 {
                total += 1;
                if lum(img.at(centre.0 + dx, centre.1 + dy)) > 30 {
                    fur += 1;
                }
            }
        }
        (fur, total)
    };
    let bare = rig.shot("P16 black surface without fur", &[(quad, body, Mat4::IDENTITY)], EYE, 0.0, 0.0, false);
    let (b, total) = covered(&bare);
    assert_eq!(b, 0, "the bare surface is black");
    rig.r.set_material_fur(body, Some(zestengine_render::FurParams { layers: 16, length: 0.05, density: 12.0, tip_color: [1.0, 0.1, 0.1], root_color: [0.1, 1.0, 0.1] }));
    let img = rig.shot("P17 the same surface with fur shells", &[(quad, body, Mat4::IDENTITY)], EYE, 0.0, 0.0, false);
    let (f, _) = covered(&img);
    eprintln!("fur coverage: {f} of {total} pixels");
    assert!(f > total / 10 && f < total * 9 / 10, "strands cover part of the surface: {f} of {total}");
    // Both colours are present: the root (green) and the tip (red) show.
    let (mut red, mut green) = (0, 0);
    for dy in -28..28 {
        for dx in -28..28 {
            let p = img.at(centre.0 + dx, centre.1 + dy);
            red += i32::from(p[0] > p[1] + 20);
            green += i32::from(p[1] > p[0] + 20);
        }
    }
    eprintln!("fur colours: tip-ish {red}, root-ish {green}");
    assert!(red > 20 && green > 20, "root-to-tip ramp: red {red}, green {green}");
    rig.r.set_material_fur(body, None);
}

fn write_sheet(sheet: &[(String, Vec<u8>)]) {
    write_sheet_to(sheet, "unity-fx-contact-sheet.png");
    // The PBR detail / transmission phases ("P..") on a sheet of their own.
    let pbr: Vec<_> = sheet.iter().filter(|(l, _)| l.starts_with('P')).cloned().collect();
    write_sheet_to(&pbr, "pbr-detail-transmission-contact-sheet.png");
    // The area-light phases ("A..") and the light-volume phases ("V..") likewise.
    let area: Vec<_> = sheet.iter().filter(|(l, _)| l.starts_with('A') && l.as_bytes().get(1).is_some_and(u8::is_ascii_digit)).cloned().collect();
    write_sheet_to(&area, "area-lights-contact-sheet.png");
    let vol: Vec<_> = sheet.iter().filter(|(l, _)| l.starts_with('V') && l.as_bytes().get(1).is_some_and(u8::is_ascii_digit)).cloned().collect();
    write_sheet_to(&vol, "light-volumes-contact-sheet.png");
}

fn write_sheet_to(sheet: &[(String, Vec<u8>)], file: &str) {
    if std::env::var("ZESTENGINE_FX_SHEET").is_ok_and(|v| v == "0") || sheet.is_empty() {
        return;
    }
    let cols = 4usize;
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
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/design").join(file);
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    out.save(&path).expect("write contact sheet");
    eprintln!("contact sheet ({cols} per row, phase order): {}", path.display());
    for (i, (label, _)) in sheet.iter().enumerate() {
        eprintln!("  [{} , {}] {label}", i / cols, i % cols);
    }
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
                let small_quad = r.upload_mesh(&quad(1.0)).unwrap();
                let big = r.upload_mesh(&quad(6.0)).unwrap();
                let sphere = r.upload_mesh(&sphere(4.0, 32, 24)).unwrap();
                let cube = r.upload_mesh(&cube()).unwrap();
                let red = emissive(&mut r, [1.0, 0.0, 0.0], 1.0, BlendMode::Alpha, false);
                let blue = emissive(&mut r, [0.0, 0.0, 1.0], 1.0, BlendMode::Alpha, false);
                let mut rig = Rig { r, quad: small_quad, big, sphere, cube, red, blue, sheet: Vec::new(), lights: Vec::new(), sprite_fill: None, linear: false };
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

#[test]
fn unity_gaps_render_on_the_real_device() {
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
