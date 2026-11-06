//! Skybox stability while the camera rotates (the editor's "sky flickers when I
//! rotate the scene view"). The real `Renderer` renders the default scene (ground,
//! cube, one directional light, sky on) into the editor viewport target. Sky-only
//! pixels (the upper part of a level camera's image) of every rotated frame are
//! compared with a settled reference of the SAME pose:
//!   * `settled`: each pose rendered several times with a full readback (idle);
//!   * `stall`:   rotating, read back every frame (serialises the pipeline);
//!   * `flight`:  rotating with frames in flight, read back only every 4th frame.
//!
//! For the procedural sky, an equirect texture and a cube-map sky.
//!
//! Needs a display; skipped without one or with `ZESTENGINE_TEST_GPU=0`.

use std::sync::Arc;

use glam::{Mat4, Vec3};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::{Window, WindowId};

use crate::{
    CameraData, DrawCmd, LightInstance, LightKind, MaterialDesc, MaterialHandle, MeshData, MeshHandle, Renderer, ShadowCasting, TextureData, Vertex,
};

const W: u32 = 1280;
const H: u32 = 720;
/// Rows [0, SKY_ROWS) are sky for a level camera at eye height 2 (the ground
/// horizon is at the middle row; the cube sits below the top third).
const SKY_ROWS: u32 = H / 3;
const POSES: usize = 36;
const STEP_DEG: f32 = 1.5;
/// Max per-channel difference (of 255) a sky pixel may have against the reference.
const TOL: i32 = 3;

fn quad(half: f32) -> MeshData {
    let mut vertices = Vec::new();
    for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
        vertices.push(Vertex {
            position: [x * half, y * half, 0.0],
            normal: [0.0, 0.0, 1.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            color: [1.0; 4],
            ..Default::default()
        });
    }
    MeshData { vertices, indices: vec![0, 1, 2, 0, 2, 3], has_lightmap_uv: false }
}

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

struct Rig {
    r: Renderer,
    ground: MeshHandle,
    cube: MeshHandle,
    mat: MaterialHandle,
    frame: u32,
    /// Request a live reflection-probe capture whenever none is running.
    capture: bool,
    /// Explicit pitch (degrees) instead of the yaw-linked wobble.
    pitch: Option<f32>,
    /// Render the sky only (no ground / cube): the mask of pure sky pixels.
    sky_only: bool,
    /// Orthographic camera (half height 6 m) instead of the 57-degree perspective.
    ortho: bool,
    /// Post-process override for the next renders (None = the default ACES).
    postfx: Option<crate::PostFx>,
    /// Draw no sky at all (the cost baseline).
    no_sky: bool,
}

/// Vertical field of view (f32 bits) of the perspective camera; 1.0 rad by default.
static FOV: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0x3f80_0000);

/// A saturated, direction-dependent HDR-ish texture so a stale or wrong sky binding
/// shows (a horizontal rainbow with a bright band).
fn sky_tex(w: u32, h: u32, seed: u8) -> TextureData {
    let mut pixels = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let u = x as f32 / w as f32;
            let v = y as f32 / h as f32;
            pixels.extend([
                (u * 255.0) as u8,
                (v * 255.0) as u8,
                ((u * 6.0 + v * 3.0 + seed as f32).sin().abs() * 255.0) as u8,
                255,
            ]);
        }
    }
    TextureData { width: w, height: h, pixels, srgb: false, hdr: false }
}

/// Camera distance multiplier (f32 bits) for the zoom sweep; 1.0 = the usual eye.
static EYE_SCALE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0x3f80_0000);

impl Rig {
    fn camera_ex(yaw_deg: f32, pitch_deg: Option<f32>, ortho: bool) -> CameraData {
        let scale = f32::from_bits(EYE_SCALE.load(std::sync::atomic::Ordering::Relaxed));
        let eye = Vec3::new(0.0, 2.0, 5.0) * scale;
        let yaw = yaw_deg.to_radians();
        // Pitch wobbles with the yaw (the procedural sky only varies with pitch).
        let pitch = pitch_deg.map_or((yaw_deg * 0.2).sin().abs().to_radians() * 12.0, f32::to_radians);
        let fwd = Vec3::new(yaw.sin() * pitch.cos(), pitch.sin(), -yaw.cos() * pitch.cos());
        // Straight up / down: Y is parallel to the view axis, take the yaw's horizontal as up.
        let up = if pitch.cos().abs() < 1e-3 { Vec3::new(-yaw.sin(), 0.0, yaw.cos()) * pitch.sin().signum() } else { Vec3::Y };
        let view = Mat4::look_at_rh(eye, eye + fwd, up);
        let aspect = W as f32 / H as f32;
        let proj = if ortho { Mat4::orthographic_rh(-6.0 * aspect, 6.0 * aspect, -6.0, 6.0, 0.05, 500.0) } else { Mat4::perspective_rh(f32::from_bits(FOV.load(std::sync::atomic::Ordering::Relaxed)), aspect, 0.05, 500.0) };
        CameraData { view, proj, position: eye }
    }

    fn render(&mut self, yaw_deg: f32) {
        let mk = |mesh, transform| DrawCmd {
            gi_source: None,
            mesh,
            material: self.mat,
            transform,
            highlight: 0.0,
            mesh_center: Vec3::ZERO,
            bound_radius: 40.0,
            lightmap_layer: -1,
            lightmap_size: 0,
            static_geometry: false,
            planar_index: -1,
            occluded: false,
            shadow_casting: ShadowCasting::TwoSided,
            instance_color: None,
            instance_param: None,
        };
        // `ZESTENGINE_SKY_GROUND=n`: ground half-size (a big ground reaches past the far plane).
        let ground_scale: f32 = std::env::var("ZESTENGINE_SKY_GROUND").ok().and_then(|v| v.parse().ok()).unwrap_or(10.0);
        let mut draws = vec![
            mk(self.ground, Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2) * Mat4::from_scale(Vec3::splat(ground_scale))),
            mk(self.cube, Mat4::from_translation(Vec3::new(1.5, 0.5, 0.0))),
        ];
        // GPU load (`ZESTENGINE_SKY_OVERDRAW=n`): n more copies of the ground, so the
        // GPU is the slow side and consecutive frames really overlap on the queue.
        let extra: usize = std::env::var("ZESTENGINE_SKY_OVERDRAW").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        for k in 0..extra {
            draws.push(mk(self.ground, Mat4::from_translation(Vec3::new(0.0, -0.0001 * (k + 1) as f32, 0.0)) * Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2) * Mat4::from_scale(Vec3::splat(10.0))));
        }
        let sun = [LightInstance {
            kind: LightKind::Directional,
            position: Vec3::ZERO,
            direction: Vec3::new(-0.4, -1.0, -0.3).normalize(),
            color: [1.0, 0.98, 0.92],
            intensity: 3.0,
            range: 0.0,
            spot_inner_deg: 0.0,
            spot_outer_deg: 0.0,
            cast_shadows: true,
            soft_shadows: true, radius: 0.0,
            shadow_bias: 0.005,
            baked: false,
        }];
        if self.capture && !self.r.reflection_capture_in_flight() {
            self.r.request_reflection_capture(Vec3::new(0.0, 1.0, 0.0), 128);
        }
        let mut input = crate::cascade_tests::input(Self::camera_ex(yaw_deg, self.pitch, self.ortho));
        if let Some(fx) = self.postfx {
            input.postfx = fx;
        }
        input.draws = if self.sky_only { &[] } else { &draws };
        input.lights = &sun;
        input.draw_skybox = !self.no_sky;
        input.viewport_extent = Some([W, H]);
        input.time = self.frame as f32 / 60.0;
        self.frame += 1;
        self.r.render(&input).unwrap_or_else(|e| panic!("render: {e:#}"));
    }

    fn read(&mut self) -> Vec<u8> {
        let (w, h, px) = self.r.debug_read_viewport().expect("readback").expect("viewport target");
        assert_eq!((w, h), (W, H));
        px
    }

    fn settled(&mut self, yaw: f32) -> Vec<u8> {
        for _ in 0..4 {
            self.render(yaw);
            self.read();
        }
        self.render(yaw);
        self.read()
    }
}

/// (max channel diff, pixels over TOL) in the sky rows.
fn sky_diff(a: &[u8], b: &[u8]) -> (i32, usize) {
    region_diff(a, b, SKY_ROWS)
}

/// The same over the whole image (ground, cube, sky).
fn full_diff(a: &[u8], b: &[u8]) -> (i32, usize) {
    region_diff(a, b, H)
}

fn region_diff(a: &[u8], b: &[u8], rows: u32) -> (i32, usize) {
    let (mut max, mut over) = (0, 0);
    for i in 0..(rows * W) as usize {
        let d = (0..3).map(|c| (a[i * 4 + c] as i32 - b[i * 4 + c] as i32).abs()).max().unwrap();
        max = max.max(d);
        over += (d > TOL) as usize;
    }
    (max, over)
}

fn yaw(i: usize) -> f32 {
    i as f32 * STEP_DEG
}

/// Run all three schedules for the current sky; returns the failures.
fn run_mode(rig: &mut Rig, label: &str, report: &mut Vec<String>, capture: bool) {
    rig.capture = false;
    // Warm up (pipelines, shadow atlas, env cube).
    for _ in 0..8 {
        rig.render(0.0);
        rig.read();
    }
    let refs: Vec<Vec<u8>> = (0..POSES).map(|i| rig.settled(yaw(i))).collect();
    // The settled render of a pose must itself be repeatable (still camera).
    let mut worst = (0, 0usize, 0usize);
    for (i, r) in refs.iter().enumerate() {
        let again = rig.settled(yaw(i));
        let (m, n) = sky_diff(r, &again);
        if m > worst.0 {
            worst = (m, n, i);
        }
    }
    eprintln!("[{label}] settled repeat: max sky diff {} ({} px over tol) at pose {}", worst.0, worst.1, worst.2);
    if worst.0 > TOL {
        report.push(format!("{label}: still camera sky differs between settled renders (max {} at pose {})", worst.0, worst.2));
    }
    let (m, _) = sky_diff(&refs[0], &refs[POSES / 2]);
    assert!(m > 8 || label == "procedural", "{label}: poses do not differ, the rotation has no effect (diff {m})");
    // Sky must actually be sky (non-black, varying): sanity of the harness.
    let top = &refs[0][..(W * 4) as usize];
    assert!(top.as_chunks::<4>().0.iter().any(|p| p[0] > 8 || p[1] > 8 || p[2] > 8), "{label}: sky rows are black");

    rig.capture = capture;
    // Frames in flight, none stalled: every frame's final viewport colour is copied
    // out at the end of its own command buffer and compared with its pose's reference.
    rig.r.debug_capture_viewport_frames(true);
    for lap in 0..3 {
        let mut poses = Vec::new();
        for k in 0..POSES {
            let i = if lap % 2 == 0 { k } else { POSES - 1 - k };
            poses.push(i);
            rig.render(yaw(i));
        }
        let frames = rig.r.debug_take_viewport_frames().expect("captures");
        assert_eq!(frames.len(), POSES);
        let mut worst = (0, 0usize, 0usize);
        let mut bad_frames = 0;
        let mut full_worst = (0, 0usize);
        for ((_, _, px), &i) in frames.iter().zip(&poses) {
            let f = full_diff(&refs[i], px);
            if f.0 > full_worst.0 {
                full_worst = f;
            }
            let (m, n) = sky_diff(&refs[i], px);
            bad_frames += (m > TOL) as usize;
            if m > worst.0 {
                worst = (m, n, i);
            }
        }
        eprintln!("[{label}] pipelined lap {lap}: whole image max diff {} ({} px)", full_worst.0, full_worst.1);
        eprintln!("[{label}] pipelined lap {lap}: max sky diff {} ({} px over tol) at pose {}, {bad_frames}/{POSES} frames over tol", worst.0, worst.1, worst.2);
        if worst.0 > TOL {
            report.push(format!("{label}/pipelined: {bad_frames} of {POSES} frames differ from the settled pose (max {} at pose {})", worst.0, worst.2));
        }
    }
    rig.r.debug_capture_viewport_frames(false);

    for (name, read_every) in [("stall", 1usize), ("flight", 4usize), ("flight-free", 12usize)] {
        let mut worst = (0, 0usize, 0usize);
        let mut checked = 0;
        for lap in 0..2 {
            for k in 0..POSES {
                // Back and forth so successive poses are adjacent either way.
                let i = if lap == 0 { k } else { POSES - 1 - k };
                rig.render(yaw(i));
                if (lap * POSES + k) % read_every == read_every - 1 {
                    let px = rig.read();
                    checked += 1;
                    let (m, n) = sky_diff(&refs[i], &px);
                    if m > worst.0 {
                        worst = (m, n, i);
                    }
                }
            }
        }
        eprintln!("[{label}] {name}: max sky diff {} ({} px over tol) at pose {} ({checked} frames read)", worst.0, worst.1, worst.2);
        if worst.0 > TOL {
            report.push(format!("{label}/{name}: rotated sky differs from the settled pose by {} (pose {}, {} px)", worst.0, worst.2, worst.1));
        }
    }
}

/// Fast camera motion: pitch (and yaw) change by 3..20 degrees EVERY frame, up and down,
/// crossing the horizon and reaching +-88 degrees. Frames are rendered back to back with
/// no stall; each is compared with a settled render of the same pose, over the pixels
/// that are pure sky in a sky-only render of that pose, and over the whole image.
fn fast_pitch_sweep(rig: &mut Rig, label: &str, report: &mut Vec<String>) {
    for _ in 0..8 {
        rig.render(0.0);
        rig.read();
    }
    for step in [3.0f32, 7.0, 13.0, 20.0] {
        let mut poses = Vec::new();
        let (mut pitch, mut dir) = (0.0f32, 1.0f32);
        for k in 0..60 {
            poses.push((k as f32 * 1.7, pitch));
            pitch += dir * step;
            if pitch > 88.0 {
                pitch = 88.0;
                dir = -1.0;
            } else if pitch < -88.0 {
                pitch = -88.0;
                dir = 1.0;
            }
        }
        // References and sky masks (settled, stalled).
        let mut refs = Vec::new();
        let mut masks = Vec::new();
        for &(yaw, pitch) in &poses {
            rig.pitch = Some(pitch);
            rig.sky_only = false;
            refs.push(rig.settled(yaw));
            rig.sky_only = true;
            masks.push(rig.settled(yaw));
        }
        rig.sky_only = false;
        rig.r.debug_capture_viewport_frames(true);
        for lap in 0..2 {
            for &(yaw, pitch) in &poses {
                rig.pitch = Some(pitch);
                rig.render(yaw + lap as f32 * 0.0);
            }
        }
        let frames = rig.r.debug_take_viewport_frames().expect("captures");
        assert_eq!(frames.len(), poses.len() * 2);
        let (mut sky_worst, mut sky_bad, mut full_worst, mut full_bad) = (0, 0, 0, 0);
        let mut first_bad = None;
        for (n, (_, _, px)) in frames.iter().enumerate() {
            let i = n % poses.len();
            let mut sky_max = 0;
            let mut full_max = 0;
            for p in 0..(W * H) as usize {
                let d = (0..3).map(|c| (px[p * 4 + c] as i32 - refs[i][p * 4 + c] as i32).abs()).max().unwrap();
                full_max = full_max.max(d);
                let is_sky = (0..3).all(|c| (masks[i][p * 4 + c] as i32 - refs[i][p * 4 + c] as i32).abs() <= 1);
                if is_sky {
                    sky_max = sky_max.max(d);
                }
            }
            sky_worst = sky_worst.max(sky_max);
            full_worst = full_worst.max(full_max);
            if sky_max > TOL {
                sky_bad += 1;
                first_bad.get_or_insert((n, poses[i]));
            }
            full_bad += (full_max > TOL) as usize;
        }
        eprintln!(
            "[{label}] step {step} deg/frame: sky pixels worst diff {sky_worst} ({sky_bad}/{} frames over tol), whole image worst {full_worst} ({full_bad} frames)",
            frames.len()
        );
        rig.r.debug_capture_viewport_frames(false);
        if sky_worst > TOL {
            report.push(format!("{label}: step {step}: sky differs from the settled pose by {sky_worst} in {sky_bad} frames (first {first_bad:?})"));
        }
    }
    rig.pitch = None;
}

/// Zoom out: the eye moves away from the scene (x1 .. x120, past the 500 far plane and back) while
/// looking at it, frames back to back; each is compared with a settled render of the same distance.
fn zoom_sweep(rig: &mut Rig, label: &str, report: &mut Vec<String>) {
    for _ in 0..8 {
        rig.render(0.0);
        rig.read();
    }
    let mut scales = Vec::new();
    for k in 0..80 {
        scales.push(1.0 + 119.0 * (k as f32 / 79.0).powi(2));
    }
    for k in (0..80).rev() {
        scales.push(1.0 + 119.0 * (k as f32 / 79.0).powi(2));
    }
    let set = |s: f32| EYE_SCALE.store(s.to_bits(), std::sync::atomic::Ordering::Relaxed);
    rig.pitch = Some(-8.0);
    // Look back at the scene (yaw 180 = +Z... the eye sits on +Z looking toward -Z at yaw 0).
    let mut refs = Vec::new();
    let mut masks = Vec::new();
    for &sc in &scales {
        set(sc);
        rig.sky_only = false;
        refs.push(rig.settled(0.0));
        rig.sky_only = true;
        masks.push(rig.settled(0.0));
    }
    rig.sky_only = false;
    rig.r.debug_capture_viewport_frames(true);
    for &sc in &scales {
        set(sc);
        rig.render(0.0);
    }
    let frames = rig.r.debug_take_viewport_frames().expect("captures");
    let (mut sky_worst, mut sky_bad, mut full_worst, mut full_bad) = (0, 0, 0, 0);
    let mut first_bad = None;
    for (n, (_, _, px)) in frames.iter().enumerate() {
        let (mut sky_max, mut full_max) = (0, 0);
        for p in 0..(W * H) as usize {
            let d = (0..3).map(|c| (px[p * 4 + c] as i32 - refs[n][p * 4 + c] as i32).abs()).max().unwrap();
            full_max = full_max.max(d);
            let is_sky = (0..3).all(|c| (masks[n][p * 4 + c] as i32 - refs[n][p * 4 + c] as i32).abs() <= 1);
            if is_sky {
                sky_max = sky_max.max(d);
            }
        }
        sky_worst = sky_worst.max(sky_max);
        full_worst = full_worst.max(full_max);
        if sky_max > TOL {
            sky_bad += 1;
            first_bad.get_or_insert((n, scales[n]));
        }
        full_bad += (full_max > TOL) as usize;
    }
    eprintln!("[{label}] zoom sweep: sky worst diff {sky_worst} ({sky_bad}/{} frames over tol, first {first_bad:?}), whole image worst {full_worst} ({full_bad} frames)", frames.len());
    rig.r.debug_capture_viewport_frames(false);
    // The settled references themselves across distance: sky rows must not depend on distance.
    let mut dist_worst = 0;
    for i in 1..refs.len() / 2 {
        let (m, _) = sky_diff(&refs[0], &refs[i]);
        dist_worst = dist_worst.max(m);
    }
    eprintln!("[{label}] zoom sweep: top-rows difference between near and far settled frames (any distance) worst {dist_worst}");
    if sky_worst > TOL {
        report.push(format!("{label}: zoom sweep: sky differs from the settled frame by {sky_worst} in {sky_bad} frames (first {first_bad:?})"));
    }
    set(1.0);
    rig.pitch = None;
}

/// An equirect texture of `f(direction)` with the sky shader's mapping
/// (u = atan(z, x) / 2pi + 0.5, v = acos(y) / pi). `hdr`: RGBA16F, else linear UNORM.
fn equirect(w: u32, h: u32, hdr: bool, f: impl Fn(Vec3, u32, u32) -> [f32; 3]) -> TextureData {
    let mut pixels = Vec::with_capacity((w * h * if hdr { 8 } else { 4 }) as usize);
    for y in 0..h {
        for x in 0..w {
            let u = (x as f32 + 0.5) / w as f32;
            let v = (y as f32 + 0.5) / h as f32;
            let (phi, theta) = ((u - 0.5) * std::f32::consts::TAU, v * std::f32::consts::PI);
            let d = Vec3::new(phi.cos() * theta.sin(), theta.cos(), phi.sin() * theta.sin());
            let c = f(d, x, y);
            if hdr {
                for v in [c[0], c[1], c[2], 1.0] {
                    pixels.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
                }
            } else {
                pixels.extend([(c[0].clamp(0.0, 1.0) * 255.0).round() as u8, (c[1].clamp(0.0, 1.0) * 255.0).round() as u8, (c[2].clamp(0.0, 1.0) * 255.0).round() as u8, 255]);
            }
        }
    }
    TextureData { width: w, height: h, pixels, srgb: false, hdr }
}

/// The world direction the sky should show at pixel centre (px, py) for `cam`
/// (image row 0 is the top). Orthographic: the sky pass's fixed 60-degree pseudo-perspective.
fn pixel_dir(cam: &CameraData, ortho: bool, px: u32, py: u32) -> Vec3 {
    let ndc = glam::Vec2::new((px as f32 + 0.5) / W as f32 * 2.0 - 1.0, 1.0 - (py as f32 + 0.5) / H as f32 * 2.0);
    let rot_inv = glam::Mat3::from_mat4(cam.view).inverse();
    let v = if ortho {
        let f = 1.0 / 30f32.to_radians().tan();
        Vec3::new(ndc.x * (W as f32 / H as f32) / f, ndc.y / f, -1.0)
    } else {
        let p = cam.proj.inverse() * glam::Vec4::new(ndc.x, ndc.y, 0.5, 1.0);
        p.truncate() / p.w
    };
    (rot_inv * v).normalize()
}

fn max_neighbour_step(px: &[u8]) -> (i32, (u32, u32)) {
    let mut worst = (0, (0, 0));
    for y in 0..H - 1 {
        for x in 0..W - 1 {
            let i = ((y * W + x) * 4) as usize;
            for j in [i + 4, i + (W * 4) as usize] {
                let d = (0..3).map(|c| (px[i + c] as i32 - px[j + c] as i32).abs()).max().unwrap();
                if d > worst.0 {
                    worst = (d, (x, y));
                }
            }
        }
    }
    worst
}

fn dump_ppm(name: &str, px: &[u8]) {
    let Some(dir) = std::env::var_os("ZESTENGINE_SKY_DUMP") else { return };
    let dir = std::path::PathBuf::from(dir);
    let _ = std::fs::create_dir_all(&dir);
    let mut out = format!("P6\n{W} {H}\n255\n").into_bytes();
    for p in px.as_chunks::<4>().0 {
        out.extend_from_slice(&p[..3]);
    }
    let _ = std::fs::write(dir.join(format!("{name}.ppm")), out);
}

/// Direction accuracy: an octant-coloured equirect (r = x > 0, g = y > 0, b = z > 0) must
/// show, at every pixel, the octant of that pixel's view ray computed on the CPU. Poses
/// cross the -X meridian (the equirect seam), look straight up / down (poles), at the sun
/// azimuth, and use an orthographic camera.
fn direction_accuracy(rig: &mut Rig, report: &mut Vec<String>) {
    rig.r.set_skybox(Some(&equirect(1024, 512, false, |d, _, _| [(d.x > 0.0) as u8 as f32, (d.y > 0.0) as u8 as f32, (d.z > 0.0) as u8 as f32]))).unwrap();
    rig.sky_only = true;
    let mut poses: Vec<(f32, f32, bool)> = Vec::new();
    for k in 0..13 {
        poses.push((255.0 + 2.5 * k as f32, 0.0, false));
        poses.push((255.0 + 2.5 * k as f32, 35.0, false));
    }
    poses.extend([(0.0, 89.5, false), (0.0, 90.0, false), (45.0, -90.0, false), (0.0, -89.5, false), (126.87, 59.0, false)]);
    poses.extend([(270.0, 0.0, true), (30.0, 20.0, true), (270.0, 90.0, true), (100.0, -60.0, true)]);
    let mut worst = (0usize, (0.0, 0.0, false));
    for &(yaw, pitch, ortho) in &poses {
        rig.pitch = Some(pitch);
        rig.ortho = ortho;
        let px = rig.settled(yaw);
        let cam = Rig::camera_ex(yaw, Some(pitch), ortho);
        let mut bad = 0;
        for y in 0..H {
            for x in 0..W {
                let d = pixel_dir(&cam, ortho, x, y);
                let i = ((y * W + x) * 4) as usize;
                for (c, dc) in [d.x, d.y, d.z].into_iter().enumerate() {
                    if dc.abs() > 0.02 && ((px[i + c] > 100) != (dc > 0.0)) {
                        bad += 1;
                        break;
                    }
                }
            }
        }
        if bad > 0 {
            eprintln!("[accuracy] pose yaw {yaw} pitch {pitch} ortho {ortho}: {bad} px in the wrong octant");
        }
        if bad > worst.0 {
            worst = (bad, (yaw, pitch, ortho));
        }
    }
    eprintln!("[accuracy] {} poses: worst {} px show the wrong octant (pose {:?})", poses.len(), worst.0, worst.1);
    if worst.0 > 0 {
        report.push(format!("direction accuracy: {} px show the wrong octant at pose {:?}", worst.0, worst.1));
    }
    rig.ortho = false;
    rig.pitch = None;
    rig.sky_only = false;
}

/// Seam / pole smoothness: an equirect of a smooth function of direction (rgb = 0.5 + 0.5 dir)
/// must render without any step between neighbouring pixels: no seam line at the -X meridian
/// while yawing across it, no spot or streak at the poles (also orthographic).
fn seam_and_poles(rig: &mut Rig, report: &mut Vec<String>) {
    rig.r.set_skybox(Some(&equirect(4096, 2048, true, |d, _, _| [0.5 + 0.5 * d.x, 0.5 + 0.5 * d.y, 0.5 + 0.5 * d.z]))).unwrap();
    rig.sky_only = true;
    rig.postfx = Some(crate::PostFx { tonemap: 0, ..Default::default() });
    let mut poses: Vec<(f32, f32, bool)> = (0..40).map(|k| (262.0 + 0.41 * k as f32, 10.0, false)).collect();
    poses.extend([(270.0, 60.0, false), (0.0, 85.0, false), (0.0, 89.5, false), (10.0, 90.0, false), (0.0, -89.5, false), (200.0, -90.0, false), (270.0, 90.0, true), (270.0, 0.0, true)]);
    let mut worst = (0, (0.0, 0.0, false), (0, 0));
    for &(yaw, pitch, ortho) in &poses {
        rig.pitch = Some(pitch);
        rig.ortho = ortho;
        let px = rig.settled(yaw);
        let (m, at) = max_neighbour_step(&px);
        if m > 6 {
            eprintln!("[seam/poles] pose yaw {yaw} pitch {pitch} ortho {ortho}: step {m} at {at:?}");
        }
        if m > worst.0 {
            worst = (m, (yaw, pitch, ortho), at);
        }
        if (yaw, pitch) == (270.0, 60.0) || (yaw, pitch) == (0.0, 89.5) {
            dump_ppm(&format!("smooth_yaw{yaw}_pitch{pitch}"), &px);
        }
    }
    eprintln!("[seam/poles] {} poses: worst neighbouring-pixel step {} at pose {:?} pixel {:?}", poses.len(), worst.0, worst.1, worst.2);
    if worst.0 > 6 {
        report.push(format!("seam/poles: a {}-level step between neighbouring pixels at pose {:?} pixel {:?}", worst.0, worst.1, worst.2));
    }
    rig.postfx = None;
    rig.ortho = false;
    rig.pitch = None;
    rig.sky_only = false;
}

/// A detailed HDR equirect (near-Nyquist stripes + bright single-texel "stars", 4096 x 2048,
/// so it is minified at the edges and towards the poles).
fn detailed_hdr(stars: bool) -> TextureData {
    equirect(4096, 2048, true, |d, x, y| {
        let base = 0.25 + 0.2 * d.y.max(0.0);
        let stripes = 0.12 * ((x as f32 * 2.7).sin() * (y as f32 * 3.1).sin());
        let h = (x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663)) % 509;
        let star = if stars && h == 0 { 30.0 } else { 0.0 };
        [base + stripes + star, base * 1.1 + stripes + star, base * 1.4 + stripes + star]
    })
}

/// Temporal stability under sub-pixel motion: yaw 0.05 deg per frame (~0.6 px) over a detailed
/// HDR sky; a well-filtered sky changes little between frames, an aliasing one sparkles.
fn aliasing(rig: &mut Rig, report: &mut Vec<String>) {
    // Stripes only: filtered, they average to a flat tone; unfiltered they moire and crawl.
    rig.r.set_skybox(Some(&detailed_hdr(false))).unwrap();
    // 57-degree FOV: the texture is magnified (the motion of real detail, a baseline).
    // 140-degree FOV: minified ~4x, where an unfiltered sky sparkles.
    for fov in [1.0f32, 2.45] {
        FOV.store(fov.to_bits(), std::sync::atomic::Ordering::Relaxed);
        aliasing_at(rig, report, fov);
    }
    FOV.store(1f32.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

fn aliasing_at(rig: &mut Rig, report: &mut Vec<String>, fov: f32) {
    let pitch = 25.0;
    rig.sky_only = true;
    rig.pitch = Some(pitch);
    for _ in 0..6 {
        rig.render(40.0);
        rig.read();
    }
    rig.r.debug_capture_viewport_frames(true);
    for k in 0..48 {
        rig.render(40.0 + 0.05 * k as f32);
    }
    let frames = rig.r.debug_take_viewport_frames().expect("captures");
    rig.r.debug_capture_viewport_frames(false);
    let mut sum = 0f64;
    let mut hist = vec![0usize; 256];
    let mut n = 0usize;
    // The central quarter of the image (where the wide view is minified the most).
    for w in frames.windows(2) {
        for (k, (a, b)) in w[0].2.as_chunks::<4>().0.iter().zip(w[1].2.as_chunks::<4>().0).enumerate() {
            let (x, y) = (k as u32 % W, k as u32 / W);
            if !((W / 4..3 * W / 4).contains(&x) && (H / 4..3 * H / 4).contains(&y)) {
                continue;
            }
            let d = (0..3).map(|c| (a[c] as i32 - b[c] as i32).abs()).max().unwrap() as usize;
            sum += d as f64;
            hist[d] += 1;
            n += 1;
        }
    }
    let pct = |p: f64| {
        let mut acc = 0;
        for (v, c) in hist.iter().enumerate() {
            acc += c;
            if acc as f64 >= p * n as f64 {
                return v;
            }
        }
        255
    };
    let mean = sum / n as f64;
    // Flicker: the mean of each 16x16 block barely changes under sub-pixel motion when the
    // sky is filtered; detail that pops in and out (aliasing) makes the block means jump.
    let block_means = |px: &[u8]| -> Vec<f64> {
        let (bw, bh) = (W / 16, H / 16);
        let mut out = vec![0f64; (bw * bh) as usize];
        for y in 0..bh * 16 {
            for x in 0..bw * 16 {
                let i = ((y * W + x) * 4) as usize;
                out[((y / 16) * bw + x / 16) as usize] += (px[i] as f64 + px[i + 1] as f64 + px[i + 2] as f64) / (3.0 * 256.0);
            }
        }
        out
    };
    let means: Vec<Vec<f64>> = frames.iter().map(|f| block_means(&f.2)).collect();
    let (mut bsum, mut bmax, mut bn) = (0f64, 0f64, 0f64);
    for w in means.windows(2) {
        for (a, b) in w[0].iter().zip(&w[1]) {
            bsum += (a - b).abs();
            bmax = f64::max(bmax, (a - b).abs());
            bn += 1.0;
        }
    }
    let block_mean = bsum / bn;
    eprintln!("[aliasing] fov {fov} rad: 16x16 block-mean change per frame: mean {block_mean:.4}, max {bmax:.3} levels");
    eprintln!("[aliasing] fov {fov} rad: 0.05 deg/frame over a detailed HDR sky, central quarter: mean frame-to-frame change {mean:.3}, p99 {}, p99.9 {}, max {}", pct(0.99), pct(0.999), pct(1.0));
    dump_ppm(&format!("detailed_hdr_fov{fov}"), &frames[0].2);
    // Only the minified view is judged (magnified, the change is the texture's own detail moving).
    if fov > 2.0 && (mean > 1.0 || block_mean > 0.25 || bmax > 3.0) {
        report.push(format!("aliasing: sub-pixel motion changes the centre by {mean:.2} per pixel, 16x16 block means by {block_mean:.3} on average, up to {bmax:.2} levels per frame"));
    }
    rig.pitch = None;
    rig.sky_only = false;
}

/// The procedural sky's sun: looking straight at the sun (the default light,
/// direction (-0.4, -1, -0.3)), the brightest pixel is the screen centre and the disc is
/// clearly brighter than the sky around it.
fn procedural_sun(rig: &mut Rig, report: &mut Vec<String>) {
    rig.r.set_skybox(None).unwrap();
    rig.sky_only = true;
    let sun = Vec3::new(0.4, 1.0, 0.3).normalize();
    let (yaw, pitch) = (sun.x.atan2(-sun.z).to_degrees(), sun.y.asin().to_degrees());
    rig.pitch = Some(pitch);
    let px = rig.settled(yaw);
    dump_ppm("procedural_sun", &px);
    let lum = |x: u32, y: u32| {
        let i = ((y * W + x) * 4) as usize;
        px[i] as u32 + px[i + 1] as u32 + px[i + 2] as u32
    };
    let mut best = (0, (0, 0));
    for y in 0..H {
        for x in 0..W {
            if lum(x, y) > best.0 {
                best = (lum(x, y), (x, y));
            }
        }
    }
    // Centroid of the brightest pixels (the disc saturates after the tonemap).
    let (mut cx, mut cy, mut cn) = (0f64, 0f64, 0f64);
    for y in 0..H {
        for x in 0..W {
            if lum(x, y) + 3 >= best.0 {
                cx += x as f64;
                cy += y as f64;
                cn += 1.0;
            }
        }
    }
    best.1 = ((cx / cn).round() as u32, (cy / cn).round() as u32);
    let ring = lum(W / 2 + 60, H / 2);
    let off = ((best.1.0 as i32 - W as i32 / 2).abs()).max((best.1.1 as i32 - H as i32 / 2).abs());
    eprintln!("[sun] yaw {yaw:.2} pitch {pitch:.2}: brightest {} at {:?} ({off} px from centre), 60 px away {ring}", best.0, best.1);
    if off > 4 || best.0 < ring + 60 {
        report.push(format!("procedural sun: brightest pixel {} at {:?} ({off} px off centre), ring {ring}", best.0, best.1));
    }
    rig.pitch = None;
    rig.sky_only = false;
}

/// XR: the multiview stereo pass (MULTIVIEW vertex shader: per-eye projection derived from
/// the eye's view-projection) must draw the same sky as one ordinary pass per eye (frame.proj),
/// for rotated eyes with asymmetric frusta, procedural and equirect.
fn xr_sky(rig: &mut Rig, report: &mut Vec<String>) {
    use super::xr_cluster::{FORMAT, H as XH, Target, W as XW, frame_input};
    if !rig.r.ctx.multiview {
        eprintln!("[xr-sky] no multiview on this device: skipped");
        return;
    }
    let proj = |l: f32, rr: f32| {
        let (u, d, n, f) = (0.9f32, -0.9f32, 0.05f32, 100.0f32);
        let (w, h) = (rr - l, u - d);
        Mat4::from_cols_array(&[2.0 / w, 0.0, 0.0, 0.0, 0.0, 2.0 / h, 0.0, 0.0, (rr + l) / w, (u + d) / h, -f / (f - n), -1.0, 0.0, 0.0, -(f * n) / (f - n), 0.0])
    };
    let head = Vec3::new(1200.0, 1.7, -800.0);
    let fwd = Vec3::new(0.5, 0.35, -0.8).normalize();
    let centre = Mat4::look_at_rh(head, head + fwd, Vec3::Y);
    let eyes = [
        crate::XrEyeCamera { view: Mat4::from_translation(Vec3::new(0.032, 0.0, 0.0)) * centre, proj: proj(-1.0, 0.8), postfx: crate::PostFx::default() },
        crate::XrEyeCamera { view: Mat4::from_translation(Vec3::new(-0.032, 0.0, 0.0)) * centre, proj: proj(-0.8, 1.0), postfx: crate::PostFx::default() },
    ];
    let cull = (centre, proj(-1.1, 1.1));
    let stereo = Target::new(&rig.r, 2).unwrap();
    let singles = [Target::new(&rig.r, 1).unwrap(), Target::new(&rig.r, 1).unwrap()];
    let smooth = equirect(1024, 512, true, |d, _, _| [0.5 + 0.5 * d.x, 0.5 + 0.5 * d.y, 0.5 + 0.5 * d.z]);
    for (label, tex) in [("procedural", None), ("equirect", Some(&smooth))] {
        rig.r.set_skybox(tex).unwrap();
        let mut input = frame_input(&[], cull.1);
        input.camera = CameraData { view: centre, proj: cull.1, position: head };
        input.draw_skybox = true;
        let (mut mv, mut pe) = (Vec::new(), [Vec::new(), Vec::new()]);
        for _ in 0..3 {
            rig.r.render_xr_frame(&input).unwrap();
            rig.r.render_xr_stereo(stereo.raw(), FORMAT.as_raw() as u32, XW, XH, &eyes, cull, &input).unwrap();
            mv = stereo.read(&rig.r).unwrap();
            rig.r.render_xr_frame(&input).unwrap();
            for e in 0..2 {
                rig.r.render_xr_eye(singles[e].raw(), FORMAT.as_raw() as u32, XW, XH, eyes[e].view, eyes[e].proj, &input, None).unwrap();
            }
            pe = [singles[0].read(&rig.r).unwrap(), singles[1].read(&rig.r).unwrap()];
        }
        let n = (XW * XH * 4) as usize;
        for e in 0..2 {
            let (a, b) = (&mv[e * n..(e + 1) * n], &pe[e][..n]);
            let max = a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0).map(|(x, y)| (0..3).map(|c| x[c].abs_diff(y[c])).max().unwrap()).max().unwrap();
            let lit = a.as_chunks::<4>().0.iter().filter(|p| p[0] as u32 + p[1] as u32 + p[2] as u32 > 24).count();
            eprintln!("[xr-sky] {label} eye {e}: multiview vs per-eye max diff {max}, {lit}/{} sky px lit", XW * XH);
            if max > 2 || lit < (XW * XH) as usize / 2 {
                report.push(format!("xr sky ({label}, eye {e}): multiview differs from the per-eye pass by {max} ({lit} px lit)"));
            }
        }
        let eyes_differ = mv[..n].as_chunks::<4>().0.iter().zip(mv[n..2 * n].as_chunks::<4>().0).filter(|(x, y)| x != y).count();
        eprintln!("[xr-sky] {label}: the two eyes differ in {eyes_differ} px (asymmetric frusta)");
    }
}

/// Reference shots for the old-vs-new comparison (`ZESTENGINE_SKY_DUMP=<dir>`).
fn dumps(rig: &mut Rig) {
    if std::env::var_os("ZESTENGINE_SKY_DUMP").is_none() {
        return;
    }
    rig.r.set_skybox(None).unwrap();
    for (name, yaw, pitch, ortho, sky_only) in [
        ("proc_level", 0.0, 4.0, false, false),
        ("proc_sunward", 126.87, 20.0, false, false),
        ("proc_up", 300.0, 70.0, false, true),
        ("proc_ortho", 30.0, 10.0, true, false),
    ] {
        rig.pitch = Some(pitch);
        rig.ortho = ortho;
        rig.sky_only = sky_only;
        let px = rig.settled(yaw);
        dump_ppm(name, &px);
    }
    rig.r.set_skybox(Some(&detailed_hdr(true))).unwrap();
    for (name, yaw, pitch) in [("hdr_seam", 270.0, 10.0), ("hdr_pole", 0.0, 88.0)] {
        rig.pitch = Some(pitch);
        rig.ortho = false;
        rig.sky_only = true;
        let px = rig.settled(yaw);
        dump_ppm(name, &px);
    }
    rig.pitch = None;
    rig.ortho = false;
    rig.sky_only = false;
}

/// GPU cost of the sky pass: the SCENE zone of a sky-only frame with and without the sky.
fn cost(rig: &mut Rig) {
    rig.r.set_gpu_zone_timing(true);
    let measure = |rig: &mut Rig, no_sky: bool| {
        rig.no_sky = no_sky;
        rig.sky_only = true;
        rig.pitch = Some(20.0);
        let mut v = Vec::new();
        for k in 0..160 {
            rig.render(k as f32 * 0.3);
            if k >= 40 {
                v.push(rig.r.stats().gpu_scene_ms);
            }
        }
        v.sort_by(f32::total_cmp);
        rig.no_sky = false;
        v[v.len() / 2]
    };
    let base = measure(rig, true);
    rig.r.set_skybox(None).unwrap();
    let proc_ms = measure(rig, false);
    rig.r.set_skybox(Some(&detailed_hdr(true))).unwrap();
    let eq_ms = measure(rig, false);
    eprintln!(
        "[cost] SCENE zone median at {W}x{H}: no sky {base:.4} ms, procedural {proc_ms:.4} ms (+{:.4}), equirect 4096x2048 HDR {eq_ms:.4} ms (+{:.4})",
        proc_ms - base,
        eq_ms - base
    );
    rig.r.set_gpu_zone_timing(false);
    rig.pitch = None;
    rig.sky_only = false;
}

struct App {
    done: bool,
    error: Option<String>,
    report: Vec<String>,
}

impl App {
    fn run(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes().with_visible(false).with_inner_size(winit::dpi::LogicalSize::new(W as f64, H as f64));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                self.error = Some(format!("window: {e}"));
                return;
            }
        };
        let mut r = match Renderer::new(window) {
            Ok(r) => r,
            Err(e) => {
                self.error = Some(format!("Renderer::new: {e:#}"));
                return;
            }
        };
        r.set_vsync(false);
        let ground = r.upload_mesh(&quad(1.0)).unwrap();
        let cube = r.upload_mesh(&cube()).unwrap();
        let mat = r.create_material(&MaterialDesc::default()).unwrap();
        let mut rig = Rig { r, ground, cube, mat, frame: 0, capture: false, pitch: None, sky_only: false, ortho: false, postfx: None, no_sky: false };
        let mut report = Vec::new();
        let only = std::env::var("ZESTENGINE_SKY_MODE").ok();
        // `ZESTENGINE_SKY_MODE=a,b,...` runs only those modes.
        let want = |m: &str| only.as_deref().is_none_or(|o| o.split(',').any(|x| x == m));
        if want("procedural") {
            run_mode(&mut rig, "procedural", &mut report, false);
        }
        if want("equirect") {
            rig.r.set_skybox(Some(&sky_tex(256, 128, 0))).unwrap();
            run_mode(&mut rig, "equirect", &mut report, false);
            run_mode(&mut rig, "equirect+live-probe", &mut report, true);
        }
        if want("cube") {
            // Orientation: six solid faces, the sky straight ahead must show the face looked at.
            let solid = |rgb: [u8; 3]| TextureData { width: 16, height: 16, pixels: (0..256).flat_map(|_| [rgb[0], rgb[1], rgb[2], 255]).collect(), srgb: false, hdr: false };
            let cols: [[u8; 3]; 6] = [[255, 0, 0], [0, 255, 255], [0, 255, 0], [255, 0, 255], [0, 0, 255], [255, 255, 0]];
            let f: Vec<TextureData> = cols.iter().map(|c| solid(*c)).collect();
            rig.r.set_skybox_cube([&f[0], &f[1], &f[2], &f[3], &f[4], &f[5]]).unwrap();
            // (yaw, pitch, expected face): -Z yellow, +X red, +Z blue, -X cyan, +Y green.
            for (yaw, pitch, face) in [(0.0, 30.0, 5), (90.0, 30.0, 0), (180.0, 30.0, 4), (270.0, 30.0, 1), (0.0, 80.0, 2)] {
                rig.pitch = Some(pitch);
                let px = rig.settled(yaw);
                let o = ((H / 2) * W + W / 2) as usize * 4;
                let got = [px[o], px[o + 1], px[o + 2]];
                let want = cols[face];
                let ok = (0..3).all(|c| (got[c] > 128) == (want[c] > 128));
                eprintln!("[cube-orientation] yaw {yaw} pitch {pitch}: got {got:?}, expected face colour {want:?}");
                if !ok {
                    report.push(format!("cube sky orientation: yaw {yaw} pitch {pitch} shows {got:?}, expected {want:?}"));
                }
            }
            rig.pitch = None;
            let faces: Vec<TextureData> = (0..6).map(|i| sky_tex(64, 64, i * 40)).collect();
            rig.r.set_skybox_cube([&faces[0], &faces[1], &faces[2], &faces[3], &faces[4], &faces[5]]).unwrap();
            run_mode(&mut rig, "cube", &mut report, false);
            run_mode(&mut rig, "cube+live-probe", &mut report, true);
        }
        if want("fast") {
            rig.r.set_skybox(None).unwrap();
            fast_pitch_sweep(&mut rig, "fast-pitch", &mut report);
        }
        if want("zoom") {
            rig.r.set_skybox(None).unwrap();
            zoom_sweep(&mut rig, "zoom-procedural", &mut report);
            rig.r.set_skybox(Some(&sky_tex(256, 128, 0))).unwrap();
            zoom_sweep(&mut rig, "zoom-equirect", &mut report);
            let faces: Vec<TextureData> = (0..6).map(|i| sky_tex(64, 64, i * 40)).collect();
            rig.r.set_skybox_cube([&faces[0], &faces[1], &faces[2], &faces[3], &faces[4], &faces[5]]).unwrap();
            zoom_sweep(&mut rig, "zoom-cube", &mut report);
        }
        if want("swap") {
            // Swap the sky while rotating with frames in flight, then settle.
            rig.r.set_skybox(None).unwrap();
            for round in 0..4u8 {
                let t = sky_tex(128, 64, round * 60);
                for k in 0..10 {
                    rig.render(yaw(k));
                    if k == 3 {
                        rig.r.set_skybox(Some(&t)).unwrap();
                    }
                }
            }
            rig.r.set_skybox(Some(&sky_tex(256, 128, 0))).unwrap();
            run_mode(&mut rig, "after-swaps", &mut report, false);
        }
        if want("accuracy") {
            direction_accuracy(&mut rig, &mut report);
        }
        if want("seam") {
            seam_and_poles(&mut rig, &mut report);
        }
        if want("alias") {
            aliasing(&mut rig, &mut report);
        }
        if want("sun") {
            procedural_sun(&mut rig, &mut report);
        }
        if want("ortho") {
            // The stability schedules with an orthographic camera (procedural and equirect).
            rig.ortho = true;
            rig.r.set_skybox(None).unwrap();
            run_mode(&mut rig, "ortho-procedural", &mut report, false);
            rig.r.set_skybox(Some(&sky_tex(256, 128, 0))).unwrap();
            run_mode(&mut rig, "ortho-equirect", &mut report, false);
            rig.ortho = false;
        }
        if want("xr") {
            xr_sky(&mut rig, &mut report);
        }
        if want("dump") {
            dumps(&mut rig);
        }
        if want("cost") {
            cost(&mut rig);
        }
        self.report = report;
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

#[test]
fn sky_is_stable_while_the_camera_rotates() {
    if std::env::var("ZESTENGINE_TEST_GPU").is_ok_and(|v| v == "0") || std::env::var_os("DISPLAY").is_none() {
        eprintln!("[gpu test] no display / GPU disabled: skipped");
        return;
    }
    let _gpu = crate::cluster_headless::gpu_lock();
    let Ok(mut event_loop) = EventLoop::builder().with_x11().with_any_thread(true).build() else {
        eprintln!("[gpu test] no X11 event loop: skipped");
        return;
    };
    let mut app = App { done: false, error: None, report: Vec::new() };
    event_loop.run_app_on_demand(&mut app).unwrap();
    if let Some(e) = &app.error {
        panic!("{e}");
    }
    assert!(app.report.is_empty(), "sky flicker:\n{}", app.report.join("\n"));
}
