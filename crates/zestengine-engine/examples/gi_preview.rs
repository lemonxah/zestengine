//! Headless GI preview: a minimal scene (one ground plane + one emissive sphere,
//! no lights) marched with the REAL `sw_gi` probe code, then the resulting probe
//! SH grid is sampled per-pixel on the CPU (mirroring the shader's smoothstepped
//! trilinear + SH-L1 eval) and written to PNGs. Lets us see the GI, including
//! the blotchy emitter fill, and compare denoising strategies without the editor.
//!
//! Run: cargo run --example gi_preview --release

use std::sync::Arc;

use zestengine_assets::{PrimitiveShape, primitive_mesh};
use zestengine_engine::sw_gi::{self, SdfInstance};
use zestengine_render::ProbeSh;
use zestengine_render::sdf::generate_sdf;
use glam::{Mat4, Vec3};

// Probe grid bounds + resolution (x fastest, then y, then z; matches sw_gi).
const BMIN: Vec3 = Vec3::new(-6.0, -0.5, -6.0);
const BMAX: Vec3 = Vec3::new(6.0, 4.0, 6.0);
const COUNTS: [usize; 3] = [40, 16, 40];

const W: usize = 640;
const H: usize = 360;

const SPHERE_C: Vec3 = Vec3::new(0.0, 0.8, 0.0);
const SPHERE_R: f32 = 0.5;
const EMISSION: Vec3 = Vec3::new(9.0, 0.0, 9.0); // bright magenta (HDR)

fn mesh_pos_idx(shape: PrimitiveShape) -> (Vec<Vec3>, Vec<u32>) {
    let m = primitive_mesh(shape);
    let pos = m.vertices.iter().map(|v| Vec3::from(v.position)).collect();
    (pos, m.indices)
}

fn scene_instances() -> Vec<SdfInstance> {
    let (pp, pi) = mesh_pos_idx(PrimitiveShape::Plane);
    let (sp, si) = mesh_pos_idx(PrimitiveShape::Sphere);
    let plane_sdf = Arc::new(generate_sdf(&pp, &pi, 32, 0.1));
    let sphere_sdf = Arc::new(generate_sdf(&sp, &si, 32, 0.1));

    // Plane: 2-unit mesh scaled x5 -> 10x10 ground at y=0.
    let plane_world = Mat4::from_scale(Vec3::new(5.0, 5.0, 5.0));
    // Sphere: 0.5-radius mesh, unit scale, lifted to SPHERE_C.
    let sphere_world = Mat4::from_translation(SPHERE_C);
    vec![
        SdfInstance {
            gi_source: None,
            inv: plane_world.inverse(),
            scale: 5.0,
            sdf: plane_sdf,
            albedo: [0.8, 0.8, 0.8],
            emission: [0.0, 0.0, 0.0],
            metallic: 0.0,
            roughness: 0.7,
            static_geometry: true,
        },
        SdfInstance {
            gi_source: None,
            inv: sphere_world.inverse(),
            scale: 1.0,
            sdf: sphere_sdf,
            albedo: [0.1, 0.1, 0.1],
            emission: EMISSION.to_array(),
            metallic: 0.0,
            roughness: 0.7,
            static_geometry: true,
        },
    ]
}

fn probe_positions() -> Vec<Vec3> {
    let [nx, ny, nz] = COUNTS;
    let mut out = Vec::with_capacity(nx * ny * nz);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let f = Vec3::new(
                    x as f32 / (nx - 1) as f32,
                    y as f32 / (ny - 1) as f32,
                    z as f32 / (nz - 1) as f32,
                );
                out.push(BMIN + f * (BMAX - BMIN));
            }
        }
    }
    out
}

/// Average `traces` independent-seed marches. This is the temporal accumulation
/// the runtime does over time, collapsed into one offline result.
fn march_accumulated(insts: &[SdfInstance], probes: &[Vec3], samples: u32, bounces: u32, traces: u32) -> Vec<ProbeSh> {
    let scene_size = (BMAX - BMIN).length();
    let mut acc = vec![ProbeSh::default(); probes.len()];
    for t in 0..traces.max(1) {
        let seed = 0x9E37_79B9u32.wrapping_mul(t + 1);
        let r = sw_gi::march_probes(insts, &[], probes, [0.01, 0.01, 0.02], samples, scene_size, bounces, seed);
        for (a, s) in acc.iter_mut().zip(&r) {
            for b in 0..4 {
                for c in 0..3 {
                    a.coeffs[b][c] += s.coeffs[b][c];
                }
                a.dist[b] += s.dist[b];
            }
        }
    }
    let inv = 1.0 / traces.max(1) as f32;
    for a in &mut acc {
        for b in 0..4 {
            for c in 0..3 {
                a.coeffs[b][c] *= inv;
            }
            a.dist[b] *= inv;
        }
    }
    acc
}

fn sh_eval(sh: &ProbeSh, n: Vec3) -> Vec3 {
    let c = |k: usize| Vec3::from(sh.coeffs[k]);
    c(0) * 0.282095 + (c(1) * n.y + c(2) * n.z + c(3) * n.x) * 0.325735
}

/// CPU mirror of the shader's smoothstepped trilinear 8-corner SH blend.
fn sample_gi(probes: &[ProbeSh], p: Vec3, n: Vec3) -> Vec3 {
    let [nx, ny, nz] = COUNTS;
    let t = (p - BMIN) / (BMAX - BMIN);
    if t.cmplt(Vec3::ZERO).any() || t.cmpgt(Vec3::ONE).any() {
        return Vec3::ZERO;
    }
    let cnt = Vec3::new((nx - 1) as f32, (ny - 1) as f32, (nz - 1) as f32);
    let g = t.clamp(Vec3::ZERO, Vec3::ONE) * cnt;
    let g0 = g.floor();
    let mut f = g - g0;
    f = f * f * (Vec3::splat(3.0) - 2.0 * f); // smoothstep (match shader)
    let g0 = g0.as_ivec3();
    let (mut sum, mut wsum) = (Vec3::ZERO, 0.0f32);
    for i in 0..8 {
        let off = glam::IVec3::new(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        let gc = (g0 + off).clamp(glam::IVec3::ZERO, glam::IVec3::new(nx as i32 - 1, ny as i32 - 1, nz as i32 - 1));
        let tw = Vec3::new(
            if off.x == 1 { f.x } else { 1.0 - f.x },
            if off.y == 1 { f.y } else { 1.0 - f.y },
            if off.z == 1 { f.z } else { 1.0 - f.z },
        );
        let w = tw.x * tw.y * tw.z;
        let idx = gc.x as usize + gc.y as usize * nx + gc.z as usize * nx * ny;
        sum += w * sh_eval(&probes[idx], n);
        wsum += w;
    }
    if wsum > 1e-5 { (sum / wsum).max(Vec3::ZERO) } else { Vec3::ZERO }
}

fn ray_sphere(o: Vec3, d: Vec3, c: Vec3, r: f32) -> Option<f32> {
    let oc = o - c;
    let b = oc.dot(d);
    let cc = oc.dot(oc) - r * r;
    let disc = b * b - cc;
    if disc < 0.0 {
        return None;
    }
    let t = -b - disc.sqrt();
    (t > 1e-3).then_some(t)
}

fn tonemap(c: Vec3) -> [u8; 3] {
    let m = Vec3::ONE - (-c).exp(); // 1 - e^-x
    let g = m.powf(1.0 / 2.2);
    [
        (g.x.clamp(0.0, 1.0) * 255.0) as u8,
        (g.y.clamp(0.0, 1.0) * 255.0) as u8,
        (g.z.clamp(0.0, 1.0) * 255.0) as u8,
    ]
}

fn render(probes: &[ProbeSh], path: &str) {
    let eye = Vec3::new(0.0, 3.2, 7.0);
    let target = Vec3::new(0.0, 0.4, 0.0);
    let fwd = (target - eye).normalize();
    let right = fwd.cross(Vec3::Y).normalize();
    let up = right.cross(fwd);
    let aspect = W as f32 / H as f32;
    let tan = (50.0f32.to_radians() * 0.5).tan();

    let mut buf = vec![0u8; W * H * 3];
    for y in 0..H {
        for x in 0..W {
            let ndc_x = (2.0 * (x as f32 + 0.5) / W as f32 - 1.0) * aspect * tan;
            let ndc_y = (1.0 - 2.0 * (y as f32 + 0.5) / H as f32) * tan;
            let dir = (fwd + right * ndc_x + up * ndc_y).normalize();

            let col;
            let t_sphere = ray_sphere(eye, dir, SPHERE_C, SPHERE_R);
            let t_plane = if dir.y.abs() > 1e-5 { Some(-eye.y / dir.y) } else { None }
                .filter(|&t| t > 1e-3);
            let plane_hit = t_plane.map(|t| eye + dir * t).filter(|p| p.x.abs() < 5.0 && p.z.abs() < 5.0);

            if let Some(ts) = t_sphere {
                if plane_hit.is_none() || ts < t_plane.unwrap_or(f32::MAX) {
                    col = EMISSION; // emitter shows its emission directly
                } else {
                    let p = eye + dir * t_plane.unwrap();
                    col = Vec3::splat(0.8) * sample_gi(probes, p, Vec3::Y);
                }
            } else if let Some(p) = plane_hit {
                col = Vec3::splat(0.8) * sample_gi(probes, p, Vec3::Y);
            } else {
                col = Vec3::new(0.02, 0.02, 0.03);
            }

            let px = tonemap(col);
            let o = (y * W + x) * 3;
            buf[o..o + 3].copy_from_slice(&px);
        }
    }
    image::save_buffer(path, &buf, W as u32, H as u32, image::ColorType::Rgb8).unwrap();
    println!("wrote {path}");
}

fn main() {
    let insts = scene_instances();
    let probes = probe_positions();
    println!("marching {} probes...", probes.len());

    // A: single trace at editor spp (what ONE frame looks like, no accumulation).
    let raw = march_accumulated(&insts, &probes, 256, 2, 1);
    render(&raw, "/tmp/gi_raw.png");

    // B: 32 accumulated traces (the runtime's temporal convergence) + grid blur.
    let mut conv = march_accumulated(&insts, &probes, 64, 2, 32);
    sw_gi::blur_probe_grid(&mut conv, COUNTS, 6);
    render(&conv, "/tmp/gi_converged.png");
}
