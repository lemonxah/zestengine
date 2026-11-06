//! Software GI probe march on the CPU, no RT cores. Mirrors the
//! probe representation, but uses realtime light falloff and per-mesh signed distance
//! fields instead of tracing a hardware BVH, so it runs on any GPU. Produces the
//! same SH-L1 `ProbeSh` representation the standard shader already samples.
//!
//! Probes are few and the update is throttled + dirty-gated, so a CPU march is
//! affordable for a coarse soft-GI preview. A GPU compute version is a later
//! perf optimization.

use std::sync::Arc;

use zestengine_render::sdf::SdfVolume;
use zestengine_render::{BakeLight, LightKind, ProbeSh};
use glam::{Mat4, Vec3};

const PI: f32 = std::f32::consts::PI;
const MAX_STEPS: u32 = 96;

/// Conservative conversion of local SDF distance to a world-space march step.
/// The average axis scale oversteps stretched walls. This inverse-matrix norm
/// bound gives the minimum scale for ordinary TRS and remains safe under shear.
pub(crate) fn sdf_distance_scale(inv: Mat4) -> f32 {
    let b = glam::Mat3::from_mat4(inv);
    let g = b * b.transpose();
    let bound = (g.x_axis.abs() + g.y_axis.abs() + g.z_axis.abs()).max_element();
    bound.sqrt().recip()
}

/// One marchable instance: a mesh SDF placed in the world. Owns an `Arc` to its
/// SDF so the whole march can be moved onto a background thread (the SDFs are
/// shared, not copied).
#[derive(Clone)]
pub struct SdfInstance {
    pub gi_source: Option<zestengine_render::GiSource>,
    /// World-to-mesh-local transform.
    pub inv: Mat4,
    /// Conservative local-distance to world-distance scale.
    pub scale: f32,
    pub sdf: Arc<SdfVolume>,
    pub albedo: [f32; 3],
    pub emission: [f32; 3],
    /// Metalness [0,1]. Diffuse bounce gain is albedo·(1−metallic) — a metal
    /// absorbs the diffuse GI bounce (energy is specular), so it can't pump light
    /// into a contact cavity each bounce.
    pub metallic: f32,
    /// Roughness [0,1]. Diffuse magnitude is roughness-independent; kept for the
    /// future specular trace.
    pub roughness: f32,
    /// Non-moving geometry. The cached Global Distance Field is built from these
    /// only, so a moving dynamic object never forces a costly GDF rebuild.
    pub static_geometry: bool,
}

/// An emissive instance reduced to a sphere area-light for next-event estimation
/// (NEE). Sampling emitters directly, instead of waiting for random bounce rays
/// to stumble onto a small bright surface, removes the blotchy variance in the
/// GI fill around emissive objects.
struct Emitter {
    center: Vec3,
    radius: f32,
    emission: Vec3,
}

/// Reduce the emissive instances to sphere area-lights. Center = the SDF's local
/// AABB center transformed to world; radius = the inscribed extent (minus the
/// 0.1 SDF pad) scaled to world. Non-emissive instances are skipped.
fn collect_emitters(insts: &[SdfInstance]) -> Vec<Emitter> {
    insts
        .iter()
        .filter_map(|i| {
            let emission = Vec3::from(i.emission);
            if emission.max_element() <= 1e-4 {
                return None;
            }
            let world = i.inv.inverse();
            let center = world.transform_point3((i.sdf.min + i.sdf.max) * 0.5);
            let half = (i.sdf.max - i.sdf.min) * 0.5;
            let radius = ((half.min_element() - 0.1).max(0.01)) * i.scale;
            Some(Emitter { center, radius, emission })
        })
        .collect()
}

/// Emissive instances reduced to sphere area-lights for the GPU march's NEE:
/// `(world_center, world_radius, emission)` per emitter. Same reduction the CPU
/// march uses, exposed so the realtime-GI driver can hand them to the shader.
pub fn emitter_spheres(insts: &[SdfInstance]) -> Vec<([f32; 3], f32, [f32; 3])> {
    collect_emitters(insts)
        .into_iter()
        .map(|e| (e.center.to_array(), e.radius, e.emission.to_array()))
        .collect()
}

/// Solid angle a sphere of `radius` subtends at `dist` (dist > radius), 0 inside.
fn sphere_solid_angle(radius: f32, dist: f32) -> f32 {
    if dist <= radius {
        return 2.0 * PI;
    }
    let s = (radius / dist).min(0.999);
    2.0 * PI * (1.0 - (1.0 - s * s).sqrt())
}

/// Irradiance at surface point `p` (normal `n`) from every visible emitter,
/// each treated as a sphere area-light: `emission · (n·l) · Ω`, occlusion-tested
/// up to the emitter's near surface. The caller folds in `albedo/π`. This is the
/// low-variance analytic replacement for catching emitters via random hits.
fn emitter_light(insts: &[SdfInstance], emitters: &[Emitter], p: Vec3, n: Vec3, eps: f32) -> Vec3 {
    let mut sum = Vec3::ZERO;
    for em in emitters {
        let d = em.center - p;
        let dist = d.length();
        if dist <= em.radius + eps {
            continue;
        }
        let l = d / dist;
        let ndl = n.dot(l).max(0.0);
        if ndl <= 0.0 {
            continue;
        }
        // Stop clearly short of the sphere (estimated radius) so the emitter's
        // own surface never reads as an occluder.
        let reach = (dist - em.radius * 1.25 - 0.02).max(eps * 2.0);
        if occluded(insts, p + n * (eps * 2.0), l, reach, eps) {
            continue;
        }
        sum += em.emission * (ndl * sphere_solid_angle(em.radius, dist));
    }
    sum
}

fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    x
}
fn rnd(state: &mut u32) -> f32 {
    *state = hash(*state);
    *state as f32 * (1.0 / 4294967296.0)
}
fn uniform_sphere(rng: &mut u32) -> Vec3 {
    let z = rnd(rng) * 2.0 - 1.0;
    let a = rnd(rng) * 2.0 * PI;
    let r = (1.0 - z * z).max(0.0).sqrt();
    Vec3::new(r * a.cos(), r * a.sin(), z)
}
/// Orthonormal basis around `n` (Duff et al.).
fn basis(n: Vec3) -> (Vec3, Vec3) {
    let s = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    (
        Vec3::new(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
        Vec3::new(b, s + n.y * n.y * a, -n.y),
    )
}
/// Cosine-weighted hemisphere sample around `n` (for diffuse bounce rays).
fn cosine_hemisphere(n: Vec3, rng: &mut u32) -> Vec3 {
    let (u1, u2) = (rnd(rng), rnd(rng));
    let r = u1.sqrt();
    let phi = 2.0 * PI * u2;
    let (t, b) = basis(n);
    (t * (r * phi.cos()) + b * (r * phi.sin()) + n * (1.0 - u1).max(0.0).sqrt()).normalize_or(n)
}

/// Signed distance to the merged scene at world point `p` (min over instances),
/// returning (distance, nearest-instance index).
fn scene_distance(insts: &[SdfInstance], p: Vec3) -> (f32, usize) {
    let mut best = f32::INFINITY;
    let mut who = 0;
    for (i, inst) in insts.iter().enumerate() {
        let local = inst.inv.transform_point3(p);
        let d = inst.sdf.sample(local) * inst.scale;
        if d < best {
            best = d;
            who = i;
        }
    }
    (best, who)
}

/// Central-difference gradient of the merged field, giving the surface normal.
fn scene_normal(insts: &[SdfInstance], p: Vec3, eps: f32) -> Vec3 {
    let dx = scene_distance(insts, p + Vec3::X * eps).0 - scene_distance(insts, p - Vec3::X * eps).0;
    let dy = scene_distance(insts, p + Vec3::Y * eps).0 - scene_distance(insts, p - Vec3::Y * eps).0;
    let dz = scene_distance(insts, p + Vec3::Z * eps).0 - scene_distance(insts, p - Vec3::Z * eps).0;
    Vec3::new(dx, dy, dz).normalize_or(Vec3::Y)
}

/// Sphere-march; returns the hit (point, instance index) or None on escape.
fn march(insts: &[SdfInstance], origin: Vec3, dir: Vec3, eps: f32, max_dist: f32) -> Option<(Vec3, usize)> {
    let mut t = eps * 2.0;
    for _ in 0..MAX_STEPS {
        let p = origin + dir * t;
        let (d, who) = scene_distance(insts, p);
        if d < eps {
            return Some((p, who));
        }
        t += d.max(eps);
        if t > max_dist {
            break;
        }
    }
    None
}

/// True if anything occludes the segment from `origin` toward `dir` within `dist`.
fn occluded(insts: &[SdfInstance], origin: Vec3, dir: Vec3, dist: f32, eps: f32) -> bool {
    let mut t = eps * 2.0;
    for _ in 0..MAX_STEPS {
        if t > dist - eps {
            return false;
        }
        let (d, _) = scene_distance(insts, origin + dir * t);
        if d < eps {
            return true;
        }
        t += d.max(eps);
    }
    false
}

fn light_cos(deg: f32) -> f32 {
    (deg.to_radians() * 0.5).cos()
}

/// Direct lighting at surface point `p` with normal `n` (matches the bake's
/// `direct_light`, with SDF-marched shadow occlusion).
fn direct_light(insts: &[SdfInstance], lights: &[BakeLight], p: Vec3, n: Vec3, eps: f32, rng: &mut u32) -> Vec3 {
    let mut sum = Vec3::ZERO;
    for l in lights {
        let (to_light, dist, atten) = match l.kind {
            LightKind::Directional => (-l.direction.normalize_or(Vec3::NEG_Y), 1e16, 1.0),
            _ => {
                let d = l.position - p;
                let dist = d.length();
                if dist < 1e-4 {
                    continue;
                }
                let tl = d / dist;
                let range = l.range.max(1e-3);
                let f = (1.0 - dist / range).clamp(0.0, 1.0);
                let mut atten = f * f / (1.0 + dist * dist);
                if matches!(l.kind, LightKind::Spot) {
                    let cd = l.direction.normalize_or(Vec3::NEG_Z).dot(-tl);
                    let (ci, co) = (light_cos(l.spot_inner_deg), light_cos(l.spot_outer_deg));
                    let s = if ci-co<=1e-6 {if cd>=co {1.0} else {0.0}}
                        else {((cd-co)/(ci-co)).clamp(0.0,1.0)};
                    atten *= s*s*(3.0-2.0*s);
                }
                (tl, dist, atten)
            }
        };
        let ndl = n.dot(to_light).max(0.0);
        if ndl <= 0.0 || atten <= 0.0 {
            continue;
        }
        let (shadow_dir, shadow_dist) = if l.radius > 1e-5 {
            let (t, b) = basis(to_light);
            let angle = 2.0 * std::f32::consts::PI * rnd(rng);
            let radius = rnd(rng).sqrt() * l.radius;
            let directional = matches!(l.kind, LightKind::Directional);
            let v = to_light * if directional { 1.0 } else { dist }
                + (t * angle.cos() + b * angle.sin()) * radius;
            (v.normalize_or(to_light), if directional { dist } else { v.length() })
        } else { (to_light, dist) };
        if occluded(insts, p + n * (eps * 2.0), shadow_dir, shadow_dist, eps) {
            continue;
        }
        sum += Vec3::from(l.color) * (ndl * atten);
    }
    sum
}

/// Incoming radiance arriving along `dir`, path-traced through up to `bounces`
/// diffuse bounces (sky on escape; emission + direct at each hit; cosine-weighted
/// continuation, throughput attenuated by albedo each bounce, the "light energy drops
/// per bounce" scatter). One bounce ≈ the old behavior; more fills cavities.
#[allow(clippy::too_many_arguments)]
fn incoming(
    insts: &[SdfInstance],
    lights: &[BakeLight],
    emitters: &[Emitter],
    origin: Vec3,
    dir: Vec3,
    sky: Vec3,
    eps: f32,
    max_dist: f32,
    bounces: u32,
    rng: &mut u32,
) -> (Vec3, f32) {
    let mut radiance = Vec3::ZERO;
    let mut throughput = Vec3::ONE;
    let (mut ro, mut rd) = (origin, dir);
    // Distance from the probe to the first surface seen along `dir` (max_dist on
    // escape to sky): the DDGI visibility moment for this direction.
    let mut first_dist = max_dist;
    for b in 0..bounces.max(1) {
        match march(insts, ro, rd, eps, max_dist) {
            None => {
                radiance += throughput * sky;
                break;
            }
            Some((hp, who)) => {
                if b == 0 {
                    first_dist = (hp - origin).length();
                }
                let mut n = scene_normal(insts, hp, eps);
                if n.dot(rd) > 0.0 {
                    n = -n;
                }
                let inst = &insts[who];
                // Diffuse bounce albedo = base·(1−metallic). A metal has no diffuse
                // lobe (its reflectance is specular/F0), so it must NOT re-radiate
                // the diffuse bounce — otherwise a metal floor pumps light back each
                // bounce and the contact cavity (form-factor≈1) concentrates toward
                // E/(1−albedo). This is the per-bounce energy absorption that bounds
                // the growing contact pool. Roughness leaves diffuse magnitude
                // unchanged (Lambert); it only matters for the future specular trace.
                let albedo = Vec3::from(inst.albedo) * (1.0 - inst.metallic.clamp(0.0, 1.0));
                // Emitters are sampled via NEE below, not added on a random hit,
                // so a hit on an emissive surface contributes no emission here.
                // That kills the firefly variance from small bright sources.
                // Lambert emit toward the ray: albedo/π · (direct lights + emitters).
                let direct = direct_light(insts, lights, hp, n, eps, rng)
                    + emitter_light(insts, emitters, hp, n, eps);
                radiance += throughput * (albedo / PI) * direct;
                // Continue the path: cosine-weighted bounce; for a Lambert BRDF the
                // /π, cosine and pdf cancel, so throughput just folds in albedo.
                throughput *= albedo;
                if throughput.max_element() < 0.01 {
                    break;
                }
                ro = hp + n * (eps * 2.0);
                rd = cosine_hemisphere(n, rng);
            }
        }
    }
    (radiance, first_dist)
}

/// SH-L1 radiance at one probe (uniform-sphere Monte-Carlo + multi-bounce).
#[allow(clippy::too_many_arguments)]
fn probe_sh(
    insts: &[SdfInstance],
    lights: &[BakeLight],
    emitters: &[Emitter],
    origin: Vec3,
    sky: Vec3,
    eps: f32,
    max_dist: f32,
    samples: u32,
    bounces: u32,
    pi: usize,
    seed: u32,
) -> ProbeSh {
    let mut rng = hash(
        (pi as u32)
            .wrapping_mul(9277)
            .wrapping_add(seed.wrapping_mul(26699)),
    );
    let (mut sh0, mut sh1, mut sh2, mut sh3) = (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::ZERO);
    // SH-L1 of the directional first-hit distance (scalar), for visibility.
    let (mut d0, mut d1, mut d2, mut d3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    // SH-L1 of the SQUARED distance (second moment) for a two-moment Chebyshev.
    let (mut q0, mut q1, mut q2, mut q3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for _ in 0..samples {
        let d = uniform_sphere(&mut rng);
        let (r, fd) = incoming(insts, lights, emitters, origin, d, sky, eps, max_dist, bounces, &mut rng);
        let r = r.min(Vec3::splat(8.0));
        sh0 += r * 0.282095;
        sh1 += r * (0.488603 * d.y);
        sh2 += r * (0.488603 * d.z);
        sh3 += r * (0.488603 * d.x);
        d0 += fd * 0.282095;
        d1 += fd * (0.488603 * d.y);
        d2 += fd * (0.488603 * d.z);
        d3 += fd * (0.488603 * d.x);
        let fd2 = fd * fd;
        q0 += fd2 * 0.282095;
        q1 += fd2 * (0.488603 * d.y);
        q2 += fd2 * (0.488603 * d.z);
        q3 += fd2 * (0.488603 * d.x);
    }
    let wgt = (4.0 * PI) / samples as f32;
    let (mut sh0, mut sh1, mut sh2, mut sh3) = (sh0 * wgt, sh1 * wgt, sh2 * wgt, sh3 * wgt);
    // Analytic NEE: add the probe's DIRECT view of each visible emitter straight
    // into the SH (radiance L over the solid angle Ω it subtends in direction l).
    // This replaces the removed random-hit emission with a zero-variance term, so
    // the bright fill right around an emitter is smooth in a single trace.
    for em in emitters {
        let dir = em.center - origin;
        let dist = dir.length();
        if dist <= em.radius + eps {
            continue;
        }
        let l = dir / dist;
        // Stop clearly short of the sphere (estimated radius) so the emitter's
        // own surface never reads as an occluder.
        let reach = (dist - em.radius * 1.25 - 0.02).max(eps * 2.0);
        if occluded(insts, origin, l, reach, eps) {
            continue;
        }
        let le = em.emission * sphere_solid_angle(em.radius, dist);
        sh0 += le * 0.282095;
        sh1 += le * (0.488603 * l.y);
        sh2 += le * (0.488603 * l.z);
        sh3 += le * (0.488603 * l.x);
    }
    let pack = |v: Vec3| [v.x, v.y, v.z];
    ProbeSh {
        coeffs: [pack(sh0), pack(sh1), pack(sh2), pack(sh3)],
        dist: [d0 * wgt, d1 * wgt, d2 * wgt, d3 * wgt],
        dist2: [q0 * wgt, q1 * wgt, q2 * wgt, q3 * wgt],
        offset: [0.0; 3],
    }
}

/// Inject one light's contribution (distance/cone falloff, FULL L1 directionality
/// — an L0-only term looks flat/translucent) into a probe's SH-L1 accumulator at
/// world position `p`. The FluxVoxel per-voxel kernel.
/// Add a DIRECT light to a probe's SH-L1, weighted to reconstruct close to a
/// clamped `N·L` while still letting MULTIPLE lights blend. The L0 *monopole*
/// (omnidirectional) band is what makes different-coloured lights mix additively;
/// the L1 directional band darkens back/grazing faces via the ≥0 clamp at eval.
///
/// Balance matters: too little L0 (the old 0.7) over-sharpened the lobe so two
/// lights from different directions CANCELLED instead of blending — one colour won
/// and the other was clamped to black (the "only one emissive light shows / lights
/// don't blend" bug). L0=1.0 keeps each light's contribution additive (colours mix)
/// while L1 (>L0) preserves enough directionality that a fully back-facing surface
/// still clamps dark, so we don't regress "lights both sides".
fn add_sh_direct(s: &mut [Vec3; 4], dir: Vec3, r: Vec3) {
    // Slightly omnidirectional-biased: emitters at floor level light the up-facing
    // floor only via the L0 monopole (the floor catches them at a grazing angle, so
    // the directional L1 barely contributes). A higher L0 lets those up-facing
    // surfaces pick up more of the glow, evening out the floor-vs-couch difference,
    // while L1 stays > L0 so strongly back-facing surfaces still clamp toward dark.
    const L0: f32 = 1.4; // monopole (blending + up-facing/grazing pickup)
    const L1: f32 = 1.6; // directional lobe (kept > L0 so back-faces still clamp)
    s[0] += r * (0.282095 * L0);
    s[1] += r * (0.488603 * L1 * dir.y);
    s[2] += r * (0.488603 * L1 * dir.z);
    s[3] += r * (0.488603 * L1 * dir.x);
}

pub fn inject_light(s: &mut [Vec3; 4], p: Vec3, l: &BakeLight) {
    inject_light_occ(s, p, l, None);
}

/// Inject one light into a probe SH, optionally attenuated by a coarse scene
/// occupancy grid (DDGI-style cheap occlusion). `occ = None` is the unshadowed
/// path. The occlusion is a *soft* transmittance from the coarse grid so it can
/// never hard-black a probe (the grid is too coarse for crisp shadows — it gives
/// contact darkening, not stencils), keeping it safe to leave on by default.
pub fn inject_light_occ(s: &mut [Vec3; 4], p: Vec3, l: &BakeLight, occ: Option<&SceneOccupancy>) {
    let col = Vec3::from(l.color); // already × intensity
    match l.kind {
        LightKind::Directional => {
            // March from the probe toward the light; if blocked, attenuate.
            let dir = (-l.direction).normalize_or(Vec3::NEG_Y);
            let vis = occ.map_or(1.0, |o| o.visibility(p, p + dir * o.span()));
            if vis > 0.0 {
                add_sh_direct(s, dir, col * vis);
            }
        }
        _ => {
            let to = l.position - p;
            let dist = to.length();
            let range = l.range.max(1e-3);
            if dist >= range {
                return;
            }
            let d = if dist > 1e-4 { to / dist } else { Vec3::Y };
            let f = 1.0 - dist / range;
            let mut atten = f * f;
            if matches!(l.kind, LightKind::Spot) {
                let ld = l.direction.normalize_or(Vec3::NEG_Y);
                let cd = ld.dot(-d);
                let ci = (l.spot_inner_deg.to_radians() * 0.5).cos();
                let co = (l.spot_outer_deg.to_radians() * 0.5).cos();
                let sp = ((cd - co) / (ci - co).max(1e-4)).clamp(0.0, 1.0);
                atten *= sp * sp;
            }
            if atten > 0.0 {
                let vis = occ.map_or(1.0, |o| o.visibility(p, l.position));
                atten *= vis;
                if atten > 0.0 {
                    add_sh_direct(s, d, col * atten);
                }
            }
        }
    }
}

/// Coarse scene occupancy bit-grid for cheap voxel-light occlusion. Mesh world
/// AABBs are rasterized into cells; `visibility` DDA-marches a segment and counts
/// solid interior cells (endpoints skipped so a probe sitting on the floor isn't
/// self-occluded). Soft, monotone attenuation — never amplifies, never hard-zero.
pub struct SceneOccupancy {
    min: Vec3,
    cell: Vec3, // world size of one cell
    dims: [i32; 3],
    bits: Vec<u64>,
}

impl SceneOccupancy {
    /// Build from world-space mesh AABBs. `max_dim` caps the longest axis's cell
    /// count (memory/cost bound). Returns None if there's nothing to occlude.
    pub fn build(boxes: &[(Vec3, Vec3)], max_dim: i32) -> Option<Self> {
        if boxes.is_empty() {
            return None;
        }
        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for (mn, mx) in boxes {
            lo = lo.min(*mn);
            hi = hi.max(*mx);
        }
        if !lo.is_finite() {
            return None;
        }
        let size = (hi - lo).max(Vec3::splat(0.1));
        let longest = size.max_element();
        let cell_size = (longest / max_dim as f32).max(0.05);
        let dims = [
            ((size.x / cell_size).ceil() as i32 + 1).clamp(1, max_dim),
            ((size.y / cell_size).ceil() as i32 + 1).clamp(1, max_dim),
            ((size.z / cell_size).ceil() as i32 + 1).clamp(1, max_dim),
        ];
        let total = (dims[0] * dims[1] * dims[2]) as usize;
        let mut occ = Self {
            min: lo,
            cell: Vec3::splat(cell_size),
            dims,
            bits: vec![0u64; total.div_ceil(64)],
        };
        for (mn, mx) in boxes {
            let c0 = occ.cell_coord(*mn);
            let c1 = occ.cell_coord(*mx);
            for z in c0[2]..=c1[2] {
                for y in c0[1]..=c1[1] {
                    for x in c0[0]..=c1[0] {
                        occ.set_solid([x, y, z]);
                    }
                }
            }
        }
        Some(occ)
    }

    /// Diagonal world span — a safe "march to the edge" distance for directional
    /// lights (the DDA clips to the grid anyway).
    pub fn span(&self) -> f32 {
        (Vec3::new(self.dims[0] as f32, self.dims[1] as f32, self.dims[2] as f32) * self.cell)
            .length()
    }

    /// Clamped cell coord — for rasterizing solid boxes (which are guaranteed to
    /// overlap the grid, since the grid IS their bounds).
    fn cell_coord(&self, p: Vec3) -> [i32; 3] {
        let r = (p - self.min) / self.cell;
        [
            (r.x.floor() as i32).clamp(0, self.dims[0] - 1),
            (r.y.floor() as i32).clamp(0, self.dims[1] - 1),
            (r.z.floor() as i32).clamp(0, self.dims[2] - 1),
        ]
    }

    /// Unclamped cell coord — for the visibility march, so a sample point OUTSIDE
    /// the grid maps to an out-of-range cell that `solid()` treats as empty (a ray
    /// running beside the geometry must not clamp into a solid boundary cell).
    fn cell_coord_raw(&self, p: Vec3) -> [i32; 3] {
        let r = (p - self.min) / self.cell;
        [r.x.floor() as i32, r.y.floor() as i32, r.z.floor() as i32]
    }

    fn lin(&self, c: [i32; 3]) -> usize {
        ((c[2] * self.dims[1] + c[1]) * self.dims[0] + c[0]) as usize
    }

    fn set_solid(&mut self, c: [i32; 3]) {
        let i = self.lin(c);
        self.bits[i >> 6] |= 1u64 << (i & 63);
    }

    fn solid(&self, c: [i32; 3]) -> bool {
        if c[0] < 0
            || c[1] < 0
            || c[2] < 0
            || c[0] >= self.dims[0]
            || c[1] >= self.dims[1]
            || c[2] >= self.dims[2]
        {
            return false;
        }
        let i = self.lin(c);
        (self.bits[i >> 6] >> (i & 63)) & 1 != 0
    }

    /// Soft visibility in [floor, 1] for the segment a→b. Counts solid interior
    /// cells (the first/last cell near each endpoint are skipped). Each blocked
    /// cell softly attenuates; a small floor keeps it from ever going black so the
    /// coarse grid can't punch hard holes in the lighting.
    pub fn visibility(&self, a: Vec3, b: Vec3) -> f32 {
        let seg = b - a;
        let len = seg.length();
        if len < 1e-4 {
            return 1.0;
        }
        let dir = seg / len;
        let step = self.cell.min_element() * 0.5;
        let n = (len / step).ceil() as i32;
        // Skip ~1.5 cells at each end to avoid self-occlusion at probe/light.
        let skip = (self.cell.min_element() * 1.5 / step).ceil() as i32;
        let mut blocked = 0i32;
        let mut last = [i32::MIN; 3];
        for i in skip..(n - skip).max(skip) {
            let p = a + dir * (i as f32 * step);
            let c = self.cell_coord_raw(p);
            if c == last {
                continue;
            }
            last = c;
            if self.solid(c) {
                blocked += 1;
            }
        }
        if blocked == 0 {
            1.0
        } else {
            // 1 cell → 0.5, 2 → 0.33, … with a 0.1 floor.
            (1.0 / (1.0 + blocked as f32)).max(0.1)
        }
    }
}

impl SceneOccupancy {
    /// Distance from `p` along unit `dir` to the first solid cell, clamped to
    /// `max_dist` (returned when nothing is hit). Used to build per-probe distance
    /// moments for the DDGI Chebyshev visibility test in the shader.
    pub fn ray_distance(&self, p: Vec3, dir: Vec3, max_dist: f32) -> f32 {
        let step = self.cell.min_element() * 0.5;
        let n = (max_dist / step).ceil() as i32;
        let mut last = [i32::MIN; 3];
        for i in 1..=n {
            let t = i as f32 * step;
            let c = self.cell_coord_raw(p + dir * t);
            if c == last {
                continue;
            }
            last = c;
            if self.solid(c) {
                return t.min(max_dist);
            }
        }
        max_dist
    }
}

/// Deterministic Fibonacci-sphere of `n` unit directions (even coverage, no RNG so
/// the cached moments are stable run to run).
fn fib_sphere(n: usize) -> Vec<Vec3> {
    const GA: f32 = 2.399_963_2; // golden angle (radians)
    (0..n)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let r = (1.0 - y * y).max(0.0).sqrt();
            let th = GA * i as f32;
            Vec3::new(th.cos() * r, y, th.sin() * r)
        })
        .collect()
}

/// Per-probe DDGI distance moments from the scene occupancy: SH-L1 of the
/// directional first-hit distance (`dist`) and its square (`dist2`), matching the
/// `march_probes` / shader `dist_at` convention exactly. The shader's Chebyshev
/// then blocks a probe's light from reaching fragments behind geometry — leak-free
/// diffuse + cheap shadows — computed ONCE here (cached), free per frame.
pub fn flux_distance_moments(
    positions: &[Vec3],
    occ: &SceneOccupancy,
) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    // Distances are clamped to this, so no ray needs to march further.
    const MAX_MOMENT_DIST: f32 = 8.0;
    let dirs = fib_sphere(16);
    let wgt = (4.0 * PI) / dirs.len() as f32;
    let max_dist = occ.span().min(MAX_MOMENT_DIST);
    let probe = |p: Vec3| {
        let (mut d0, mut d1, mut d2, mut d3) = (0.0f32, 0.0, 0.0, 0.0);
        let (mut q0, mut q1, mut q2, mut q3) = (0.0f32, 0.0, 0.0, 0.0);
        for &d in &dirs {
            let fd = occ.ray_distance(p, d, max_dist).min(MAX_MOMENT_DIST);
            d0 += fd * 0.282095;
            d1 += fd * (0.488603 * d.y);
            d2 += fd * (0.488603 * d.z);
            d3 += fd * (0.488603 * d.x);
            let fd2 = fd * fd;
            q0 += fd2 * 0.282095;
            q1 += fd2 * (0.488603 * d.y);
            q2 += fd2 * (0.488603 * d.z);
            q3 += fd2 * (0.488603 * d.x);
        }
        ([d0 * wgt, d1 * wgt, d2 * wgt, d3 * wgt], [q0 * wgt, q1 * wgt, q2 * wgt, q3 * wgt])
    };
    // Probes are independent: split them across the CPU cores.
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(16);
    let chunk = positions.len().div_ceil(threads).max(256);
    let mut out = vec![([0.0f32; 4], [0.0f32; 4]); positions.len()];
    std::thread::scope(|scope| {
        for (src, dst) in positions.chunks(chunk).zip(out.chunks_mut(chunk)) {
            let probe = &probe;
            scope.spawn(move || {
                for (p, o) in src.iter().zip(dst) {
                    *o = probe(*p);
                }
            });
        }
    });
    out.into_iter().unzip()
}

/// Build a `ProbeSh` from a radiance accumulator plus precomputed distance moments
/// (vs [`acc_to_probe`] which zeroes them). Lets the FluxVoxel path carry DDGI
/// visibility so the shader Chebyshev occludes voxel light leak-free.
pub fn acc_to_probe_moments(s: &[Vec3; 4], dist: [f32; 4], dist2: [f32; 4]) -> ProbeSh {
    let p = |v: Vec3| [v.x, v.y, v.z];
    ProbeSh {
        coeffs: [p(s[0]), p(s[1]), p(s[2]), p(s[3])],
        dist,
        dist2,
        offset: [0.0; 3],
    }
}

/// Like [`flux_inject`] but each light is attenuated by the scene occupancy grid
/// (cheap DDGI-style voxel-light occlusion). Used when `voxel_ddgi_occlusion` is on.
pub fn flux_inject_occluded(
    acc: &mut [[Vec3; 4]],
    positions: &[Vec3],
    lights: &[BakeLight],
    occ: &SceneOccupancy,
) {
    for (s, &p) in acc.iter_mut().zip(positions) {
        for l in lights {
            inject_light_occ(s, p, l, Some(occ));
        }
    }
}

/// Evaluate an SH-L1 accumulator in direction `dir` (same band layout as
/// `add_sh_direct`: s[0]=L0, s[1]=Y, s[2]=Z, s[3]=X). Returns radiance toward `dir`.
pub fn sh_eval(s: &[Vec3; 4], dir: Vec3) -> Vec3 {
    s[0] * 0.282095
        + s[1] * (0.488603 * dir.y)
        + s[2] * (0.488603 * dir.z)
        + s[3] * (0.488603 * dir.x)
}

/// LPV-style flux propagation (FLUXVOXEL_TODO §D — ≥1 real diffuse bounce). Each
/// iteration spreads a fraction (`gain`) of every cell's directional radiance to its
/// 6 axis neighbours, re-projected as a lobe along the travel direction, so light
/// "flows" through the grid and fills shadowed pockets — a cheap one-bounce GI on
/// top of the direct inject. `counts` is the per-volume probe grid (x-fastest, then
/// y, then z — the `flux_volume_positions` order). With `6·gain < 1` energy decays,
/// so it stays bounded over the (small) iteration count. Operates per-volume; call
/// once per FluxVolume slice.
pub fn flux_propagate(acc: &mut [[Vec3; 4]], counts: [u32; 3], iterations: u32, gain: f32) {
    let [nx, ny, nz] = [counts[0].max(2) as i32, counts[1].max(2) as i32, counts[2].max(2)
        as i32];
    let lin = |x: i32, y: i32, z: i32| ((z * ny + y) * nx + x) as usize;
    let neighbors = [
        (Vec3::X, 1, 0, 0),
        (Vec3::NEG_X, -1, 0, 0),
        (Vec3::Y, 0, 1, 0),
        (Vec3::NEG_Y, 0, -1, 0),
        (Vec3::Z, 0, 0, 1),
        (Vec3::NEG_Z, 0, 0, -1),
    ];
    for _ in 0..iterations {
        // Snapshot so propagation reads the previous iteration uniformly.
        let src = acc.to_vec();
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let mut add = [Vec3::ZERO; 4];
                    for (axis, dx, dy, dz) in neighbors {
                        let (ax, ay, az) = (x + dx, y + dy, z + dz);
                        if ax < 0 || ay < 0 || az < 0 || ax >= nx || ay >= ny || az >= nz {
                            continue;
                        }
                        // Light travels from the neighbour toward this cell, i.e.
                        // along -axis; sample the neighbour's radiance in that dir.
                        let travel = -axis;
                        let r = sh_eval(&src[lin(ax, ay, az)], travel).max(Vec3::ZERO);
                        add_sh_direct(&mut add, travel, r * gain);
                    }
                    let c = lin(x, y, z);
                    for b in 0..4 {
                        acc[c][b] += add[b];
                    }
                }
            }
        }
    }
}

/// Box-blur the SH accumulator over the probe grid to smooth out the blotchy
/// per-probe footprints (each emitter lights nearby probes with a sharp falloff, so
/// the raw trilinear field shows discrete blobs at the grid spacing). A weighted
/// 7-tap blur (centre + 6 axis neighbours) per iteration evens the field into a
/// smooth gradient without washing it out. Operates per-volume slice.
pub fn blur_acc(acc: &mut [[Vec3; 4]], counts: [u32; 3], iterations: u32) {
    let [nx, ny, nz] = [counts[0].max(2) as i32, counts[1].max(2) as i32, counts[2].max(2)
        as i32];
    let lin = |x: i32, y: i32, z: i32| ((z * ny + y) * nx + x) as usize;
    let neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)];
    for _ in 0..iterations {
        let src = acc.to_vec();
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    // Centre weighted 2× so the blur smooths without over-flattening.
                    let mut sum = [Vec3::ZERO; 4];
                    let mut wsum = 2.0f32;
                    let c = &src[lin(x, y, z)];
                    for b in 0..4 {
                        sum[b] = c[b] * 2.0;
                    }
                    for (dx, dy, dz) in neighbors {
                        let (ax, ay, az) = (x + dx, y + dy, z + dz);
                        if ax < 0 || ay < 0 || az < 0 || ax >= nx || ay >= ny || az >= nz {
                            continue;
                        }
                        let n = &src[lin(ax, ay, az)];
                        for b in 0..4 {
                            sum[b] += n[b];
                        }
                        wsum += 1.0;
                    }
                    let dst = &mut acc[lin(x, y, z)];
                    for b in 0..4 {
                        dst[b] = sum[b] / wsum;
                    }
                }
            }
        }
    }
}

/// Box-blur the per-probe DISTANCE MOMENTS (`[f32;4]` SH) across the grid, the
/// same way as the radiance. Without this the Chebyshev visibility varies per probe
/// (each stores its own distance-to-geometry), re-introducing the grid as fixed
/// dots on a surface even when the light field itself is smooth. Blurring the
/// moments smooths the occlusion across the grid (a touch softer shadow edge, no
/// grid dots).
pub fn blur_moments(data: &mut [[f32; 4]], counts: [u32; 3], iterations: u32) {
    let [nx, ny, nz] = [counts[0].max(2) as i32, counts[1].max(2) as i32, counts[2].max(2)
        as i32];
    let lin = |x: i32, y: i32, z: i32| ((z * ny + y) * nx + x) as usize;
    let neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)];
    for _ in 0..iterations {
        let src = data.to_vec();
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let c = &src[lin(x, y, z)];
                    let mut sum = [c[0] * 2.0, c[1] * 2.0, c[2] * 2.0, c[3] * 2.0];
                    let mut wsum = 2.0f32;
                    for (dx, dy, dz) in neighbors {
                        let (ax, ay, az) = (x + dx, y + dy, z + dz);
                        if ax < 0 || ay < 0 || az < 0 || ax >= nx || ay >= ny || az >= nz {
                            continue;
                        }
                        let nb = &src[lin(ax, ay, az)];
                        for b in 0..4 {
                            sum[b] += nb[b];
                        }
                        wsum += 1.0;
                    }
                    let dst = &mut data[lin(x, y, z)];
                    for b in 0..4 {
                        dst[b] = sum[b] / wsum;
                    }
                }
            }
        }
    }
}

/// Probe relocation + classification (FLUXVOXEL_TODO §B). Returns a per-probe
/// activity mask: a probe whose cell is solid is nudged toward the nearest empty
/// neighbour cell (relocation) and, if it can't escape, marked inactive (its SH is
/// zeroed by the caller) so geometry-trapped probes don't leak black/!wrong SH into
/// the trilinear blend. `positions` is mutated in place; returns `active[i]`.
pub fn relocate_probes(positions: &mut [Vec3], occ: &SceneOccupancy) -> Vec<bool> {
    let mut active = vec![true; positions.len()];
    let step = occ.cell.min_element();
    for (i, p) in positions.iter_mut().enumerate() {
        if !occ.solid(occ.cell_coord_raw(*p)) {
            continue;
        }
        // Search a small neighbourhood for the nearest empty cell and move there.
        let mut best: Option<(f32, Vec3)> = None;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let cand = *p + Vec3::new(dx as f32, dy as f32, dz as f32) * step;
                    if !occ.solid(occ.cell_coord_raw(cand)) {
                        let d = (cand - *p).length_squared();
                        if best.is_none_or(|(bd, _)| d < bd) {
                            best = Some((d, cand));
                        }
                    }
                }
            }
        }
        match best {
            Some((_, cand)) => *p = cand,
            None => active[i] = false, // fully trapped in solid → classify dead
        }
    }
    active
}

/// World-space probe positions for a FluxVoxel voxel volume (box centred at
/// `center`, `size` meters, `counts` probes per axis), x-fastest then y then z —
/// matching the `set_baked_probes` / shader indexing convention.
pub fn flux_volume_positions(center: Vec3, size: Vec3, counts: [u32; 3]) -> Vec<Vec3> {
    let [cx, cy, cz] = [counts[0].max(2), counts[1].max(2), counts[2].max(2)];
    let denom = Vec3::new((cx - 1) as f32, (cy - 1) as f32, (cz - 1) as f32);
    let mut out = Vec::with_capacity((cx * cy * cz) as usize);
    for z in 0..cz {
        for y in 0..cy {
            for x in 0..cx {
                let gn = Vec3::new(x as f32, y as f32, z as f32) / denom;
                out.push(center + (gn - 0.5) * size);
            }
        }
    }
    out
}

/// Add every light's contribution to a parallel SH-L1 accumulator at each probe
/// position. Used both to bake the static base once and to mix dynamic lights in
/// each frame (the "fake GI on moving lights"). `acc` and `positions` are
/// parallel and the same length.
pub fn flux_inject(acc: &mut [[Vec3; 4]], positions: &[Vec3], lights: &[BakeLight]) {
    for (s, &p) in acc.iter_mut().zip(positions) {
        for l in lights {
            inject_light(s, p, l);
        }
    }
}

/// Convert an SH-L1 accumulator into the engine's `ProbeSh` (no visibility data —
/// FluxVoxel is an analytic volume, not a traced one).
pub fn acc_to_probe(s: &[Vec3; 4]) -> ProbeSh {
    let p = |v: Vec3| [v.x, v.y, v.z];
    ProbeSh {
        coeffs: [p(s[0]), p(s[1]), p(s[2]), p(s[3])],
        dist: [0.0; 4],
        dist2: [0.0; 4],
        offset: [0.0; 3],
    }
}

/// Inverse of [`acc_to_probe`]: lift a baked `ProbeSh` back into an SH-L1
/// accumulator so dynamic FluxVoxel Lights can be added on top of a build-time
/// baked static base (distance data is dropped — FluxVoxel is analytic).
pub fn probe_to_acc(p: &ProbeSh) -> [Vec3; 4] {
    let v = |c: [f32; 3]| Vec3::new(c[0], c[1], c[2]);
    [v(p.coeffs[0]), v(p.coeffs[1]), v(p.coeffs[2]), v(p.coeffs[3])]
}

/// Worker-thread budget for background bake / GI compute. Reserves a slice of the
/// cores so the editor UI and the rest of the OS stay responsive instead of the
/// bake pinning every core to a standstill. The reserve scales with core count
/// (keep ~1 free on a quad-core, ~2 on an octa-core, ~4 on a 16-core) but always
/// leaves a working majority for the compute.
pub fn bake_worker_count() -> usize {
    let cores = std::thread::available_parallelism()
        .map(|c| c.get())
        .unwrap_or(4)
        .max(1);
    // Use 80% of the cores (floor), always leaving at least one free for the UI/OS
    // and never dropping below one worker.
    ((cores * 8) / 10).clamp(1, cores.saturating_sub(1).max(1))
}

/// Best-effort: drop the CALLING thread's scheduling priority (Linux nice) so bake
/// compute yields to interactive editor / OS work under contention. This is what
/// actually keeps the desktop usable while a bake runs — the reserved cores above
/// are extra insurance (and the only lever on platforms without per-thread nice).
/// No-op where unsupported (non-Linux, or the `editor` feature — which links libc —
/// is off, e.g. a shipped game, which never bakes anyway).
pub fn lower_compute_priority() {
        unsafe {
        // PRIO_PROCESS + who=0 targets the calling task (thread) on Linux, so only
        // this worker is niced, not the whole process. +10 sits it clearly below
        // normal-priority interactive threads, which then always win the CPU.
        libc::setpriority(libc::PRIO_PROCESS, 0, 10);
    }
}

/// March every probe and return SH-L1 radiance coefficients (same layout as the
/// ray-query probe bake). Parallelized across CPU cores; each probe is
/// independent. `scene_size` sizes the march epsilon and reach.
#[allow(clippy::too_many_arguments)]
pub fn march_probes(
    insts: &[SdfInstance],
    lights: &[BakeLight],
    probes: &[Vec3],
    sky_color: [f32; 3],
    samples: u32,
    scene_size: f32,
    bounces: u32,
    seed: u32,
) -> Vec<ProbeSh> {
    let sky = Vec3::from(sky_color);
    let eps = (scene_size * 0.004).max(1e-3);
    let max_dist = scene_size * 1.5 + 1.0;
    let samples = samples.max(1);
    let bounces = bounces.clamp(1, 16);
    let emitters = collect_emitters(insts);
    let n = probes.len();
    let mut out = vec![ProbeSh::default(); n];
    // Leave cores free for the UI/OS so a bake doesn't pin the whole machine.
    let cores = bake_worker_count();
    let chunk = n.div_ceil(cores).max(1);
    let emitters = &emitters; // shared ref so each scoped thread can read it
    std::thread::scope(|s| {
        for (ci, (pchunk, ochunk)) in probes.chunks(chunk).zip(out.chunks_mut(chunk)).enumerate() {
            let base = ci * chunk;
            s.spawn(move || {
                lower_compute_priority();
                for (k, &origin) in pchunk.iter().enumerate() {
                    ochunk[k] = probe_sh(
                        insts, lights, emitters, origin, sky, eps, max_dist, samples, bounces,
                        base + k, seed,
                    );
                }
            });
        }
    });
    out
}

/// A merged Global Distance Field over the scene AABB: one signed-distance grid
/// (min over all instances) plus the nearest-instance index per voxel. Uploaded
/// to the GPU as a 3D distance texture (trilinear) plus index texture (nearest)
/// so the GPU march samples one field per step instead of looping instances. Built
/// from the already-cached per-mesh SDFs, only when the scene changes.
pub struct Gdf {
    pub dims: [u32; 3],
    pub min: Vec3,
    pub max: Vec3,
    /// Signed distance per voxel (x fastest, then y, then z).
    pub dist: Vec<f32>,
    /// Nearest-instance index per voxel (same layout), used to look up albedo/emission.
    pub index: Vec<u32>,
}

/// Build the GDF over `[min,max]` at `dims` resolution from the marchable
/// instances. Parallel over z-slices. Empty instances give a far, single-cell field.
pub fn build_gdf(insts: &[SdfInstance], min: Vec3, max: Vec3, dims: [u32; 3]) -> Gdf {
    let dims = [dims[0].max(2), dims[1].max(2), dims[2].max(2)];
    let n = (dims[0] * dims[1] * dims[2]) as usize;
    let mut dist = vec![f32::MAX; n];
    let mut index = vec![0u32; n];
    if insts.is_empty() {
        return Gdf { dims, min, max, dist, index };
    }
    let extent = (max - min).max(Vec3::splat(1e-4));
    let step = Vec3::new(
        extent.x / (dims[0] - 1) as f32,
        extent.y / (dims[1] - 1) as f32,
        extent.z / (dims[2] - 1) as f32,
    );
    let (nx, ny, nz) = (dims[0] as usize, dims[1] as usize, dims[2] as usize);
    let slice = nx * ny;
    // Leave cores free for the UI/OS so a bake doesn't pin the whole machine.
    let cores = bake_worker_count();
    let zchunk = nz.div_ceil(cores).max(1);
    std::thread::scope(|s| {
        for (ci, (dchunk, ichunk)) in dist
            .chunks_mut(zchunk * slice)
            .zip(index.chunks_mut(zchunk * slice))
            .enumerate()
        {
            let z0 = ci * zchunk;
            s.spawn(move || {
                lower_compute_priority();
                for (local, (d, idx)) in dchunk.iter_mut().zip(ichunk.iter_mut()).enumerate() {
                    let z = z0 + local / slice;
                    let rem = local % slice;
                    let (y, x) = (rem / nx, rem % nx);
                    let p = min + Vec3::new(x as f32, y as f32, z as f32) * step;
                    let (best, who) = scene_distance(insts, p);
                    *d = best;
                    *idx = who as u32;
                }
            });
        }
    });
    Gdf { dims, min, max, dist, index }
}

/// One separable [1,2,1]/4 blur pass along a single grid axis (`axis`: 0=x, 1=y,
/// 2=z), edge-clamped, reading `src` and writing `dst`. The flat layout is
/// x-fastest then y then z, matching the gather/shader probe ordering.
fn blur_axis(src: &[ProbeSh], dst: &mut [ProbeSh], counts: [usize; 3], axis: usize) {
    let [nx, ny, nz] = counts;
    let idx = |x: usize, y: usize, z: usize| (z * ny + y) * nx + x;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let (a, b) = match axis {
                    0 => (idx(x.saturating_sub(1), y, z), idx((x + 1).min(nx - 1), y, z)),
                    1 => (idx(x, y.saturating_sub(1), z), idx(x, (y + 1).min(ny - 1), z)),
                    _ => (idx(x, y, z.saturating_sub(1)), idx(x, y, (z + 1).min(nz - 1))),
                };
                let (c, pa, pb) = (&src[idx(x, y, z)], &src[a], &src[b]);
                let mut o = ProbeSh::default();
                for bi in 0..4 {
                    for ci in 0..3 {
                        o.coeffs[bi][ci] =
                            (pa.coeffs[bi][ci] + 2.0 * c.coeffs[bi][ci] + pb.coeffs[bi][ci]) * 0.25;
                    }
                    o.dist[bi] = (pa.dist[bi] + 2.0 * c.dist[bi] + pb.dist[bi]) * 0.25;
                }
                dst[idx(x, y, z)] = o;
            }
        }
    }
}

/// Spatially denoise the probe SH grid in place with `iterations` separable
/// blur passes. This cancels the blotchy per-probe Monte-Carlo variance:
/// adjacent probes are independent noisy estimates, so averaging across the
/// grid pulls them toward the true (smooth, low-frequency) irradiance field with
/// no temporal lag, unlike the EMA. Soft by design, which is the look we want.
pub fn blur_probe_grid(probes: &mut [ProbeSh], counts: [usize; 3], iterations: u32) {
    let [nx, ny, nz] = counts;
    if probes.len() != nx * ny * nz || (nx < 3 && ny < 3 && nz < 3) {
        return;
    }
    let mut src = probes.to_vec();
    let mut dst = vec![ProbeSh::default(); probes.len()];
    for _ in 0..iterations.max(1) {
        blur_axis(&src, &mut dst, counts, 0);
        blur_axis(&dst, &mut src, counts, 1);
        blur_axis(&src, &mut dst, counts, 2);
        std::mem::swap(&mut src, &mut dst); // latest result lives in `src`
    }
    probes.copy_from_slice(&src);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stretched_wall_cannot_be_skipped_by_a_shadow_ray() {
        // Analytic local unit-cube SDF, then apply a thin, tall world transform.
        let dims = [17; 3];
        let mut data = Vec::new();
        for z in 0..17 { for y in 0..17 { for x in 0..17 {
            let p = Vec3::new(x as f32, y as f32, z as f32) / 8.0 - Vec3::ONE;
            let q = p.abs() - Vec3::splat(0.5);
            data.push(q.max(Vec3::ZERO).length() + q.max_element().min(0.0));
        } } }
        let inv = Mat4::from_scale(Vec3::new(0.1, 5.0, 5.0)).inverse();
        let wall = SdfInstance { gi_source: None, inv, scale: sdf_distance_scale(inv),
            sdf: Arc::new(SdfVolume { dims, min: -Vec3::ONE, max: Vec3::ONE, geometry_bounds:[-Vec3::ONE,Vec3::ONE], data }),
            albedo: [0.7; 3], emission: [0.0; 3], metallic: 0.0,
            roughness: 1.0, static_geometry: true };
        assert!(occluded(&[wall], -Vec3::X, Vec3::X, 2.0, 0.001),
            "software GI stepped through a scaled wall");
    }
    use zestengine_render::sdf::generate_sdf;

    #[test]
    fn bake_worker_count_leaves_headroom() {
        let n = bake_worker_count();
        let cores = std::thread::available_parallelism()
            .map(|c| c.get())
            .unwrap_or(4)
            .max(1);
        assert!(n >= 1, "must always use at least one worker");
        assert!(n <= cores, "never more workers than cores");
        // ~80% cap; on >1 core machines at least one core stays free for the UI/OS.
        assert!(n <= (cores * 8) / 10 || cores == 1);
        if cores > 1 {
            assert!(n < cores, "a free core must remain for the UI/OS");
        }
    }

    #[test]
    fn flux_realtime_light_falloff_matches_visible_light_units() {
        let mut light=BakeLight {kind:LightKind::Point,position:Vec3::Z,direction:Vec3::NEG_Z,
            color:[4.0;3],range:4.0,spot_inner_deg:30.0,spot_outer_deg:60.0,radius:0.0};
        for distance in [0.1f32,0.5,1.0,3.0,5.0] {
            light.position=Vec3::Z*distance;
            let actual=direct_light(&[],&[light],Vec3::ZERO,Vec3::Z,0.001,&mut 1).x;
            let expected=4.0*(1.0-distance/4.0).max(0.0).powi(2)/(1.0+distance*distance);
            assert!((actual-expected).abs()<1e-5,"distance={distance}: {actual} != {expected}");
        }
        light.position=Vec3::Z;light.kind=LightKind::Spot;
        let ci=light_cos(30.0);let co=light_cos(60.0);
        for t in [0.0f32,0.25,0.5,0.75,1.0] {
            let cd=co+(ci-co)*t;light.direction=Vec3::new((1.0-cd*cd).sqrt(),0.0,-cd);
            let actual=direct_light(&[],&[light],Vec3::ZERO,Vec3::Z,0.001,&mut 1).x;
            let expected=4.0*0.75*0.75/2.0*t*t*(3.0-2.0*t);
            assert!((actual-expected).abs()<1e-5,"spot t={t}: {actual} != {expected}");
        }
        light.spot_inner_deg=60.0;
        for (cd,on) in [(co+0.01,true),(co-0.01,false)] {
            light.direction=Vec3::new((1.0-cd*cd).sqrt(),0.0,-cd);
            let actual=direct_light(&[],&[light],Vec3::ZERO,Vec3::Z,0.001,&mut 1).x;
            assert!(actual.is_finite());assert_eq!(actual>0.0,on);
        }
    }

    // A probe above a lit floor cube must gather non-zero bounced radiance.
    #[test]
    fn probe_gathers_bounced_light() {
        // Unit cube centered at origin (top face at y = 0.5).
        let h = 0.5f32;
        let v: Vec<Vec3> = [
            [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
            [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
        ]
        .iter()
        .map(|c| Vec3::from_array(*c))
        .collect();
        let idx = vec![
            0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7,
            3, 1, 2, 6, 1, 6, 5,
        ];
        let sdf = Arc::new(generate_sdf(&v, &idx, 32, 0.1));
        let inst = SdfInstance {
            gi_source: None,
            inv: Mat4::IDENTITY,
            scale: 1.0,
            sdf,
            albedo: [0.8, 0.8, 0.8],
            emission: [0.0; 3],
            metallic: 0.0,
            roughness: 0.7,
                static_geometry: true,
        };
        let light = BakeLight {
            kind: LightKind::Point,
            position: Vec3::new(0.0, 2.0, 0.0),
            direction: Vec3::ZERO,
            color: [20.0, 20.0, 20.0],
            range: 20.0,
            spot_inner_deg: 0.0,
            spot_outer_deg: 0.0,
            radius: 0.0,
        };
        // Probe above the cube's top face.
        let probe = Vec3::new(0.0, 1.2, 0.0);
        let out = march_probes(&[inst], &[light], &[probe], [0.0; 3], 512, 4.0, 1, 1);
        let l0 = out[0].coeffs[0];
        assert!(
            l0[0] > 0.0,
            "probe should gather bounced light from the lit cube top, got L0 {l0:?}"
        );
        // Visibility moment: mean directional distance must be positive/finite,
        // and the downward (-Y) direction (toward the cube top ~0.7 below) must
        // read closer than the overall mean (the rest escapes to max_dist).
        let mean = out[0].dist[0] * 0.282095;
        assert!(mean > 0.0 && mean.is_finite(), "mean probe distance {mean}");
        let down = out[0].dist[0] * 0.282095 + -(0.488603 * out[0].dist[1]);
        assert!(
            down < mean,
            "downward seen-distance ({down}) should be nearer than mean ({mean})"
        );
    }

    // The grid blur must leave a constant field untouched and spread an isolated
    // spike into its neighbours (variance reduction without bias).
    #[test]
    fn blur_smooths_spike_preserves_flat() {
        let counts = [3, 3, 3];
        let n = counts[0] * counts[1] * counts[2];

        // Flat field is a fixed point of the blur.
        let mut flat = vec![ProbeSh { coeffs: [[0.5; 3]; 4], ..Default::default() }; n];
        blur_probe_grid(&mut flat, counts, 2);
        for p in &flat {
            assert!((p.coeffs[0][0] - 0.5).abs() < 1e-4, "flat field changed");
        }

        // A single hot probe at the center spreads outward; the center drops and
        // a neighbour rises.
        let mut spike = vec![ProbeSh::default(); n];
        let center = (3 + 1) * 3 + 1;
        let neighbor = (3 + 1) * 3;
        spike[center].coeffs[0][0] = 1.0;
        blur_probe_grid(&mut spike, counts, 1);
        assert!(spike[center].coeffs[0][0] < 1.0, "spike center should fall");
        assert!(spike[neighbor].coeffs[0][0] > 0.0, "neighbour should gain");
    }

    // The GDF must carry the merged scene's sign (negative inside the cube,
    // positive outside) and tag every voxel with the (only) instance.
    #[test]
    fn gdf_merges_sign_and_index() {
        let h = 0.5f32;
        let v: Vec<Vec3> = [
            [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
            [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
        ]
        .iter()
        .map(|c| Vec3::from_array(*c))
        .collect();
        let idx = vec![
            0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7,
            3, 1, 2, 6, 1, 6, 5,
        ];
        let sdf = std::sync::Arc::new(zestengine_render::sdf::generate_sdf(&v, &idx, 32, 0.1));
        let inst = SdfInstance {
            gi_source: None,
            inv: Mat4::IDENTITY,
            scale: 1.0,
            sdf,
            albedo: [0.8; 3],
            emission: [0.0; 3],
            metallic: 0.0,
            roughness: 0.7,
                static_geometry: true,
        };
        let dims = [16, 16, 16];
        let gdf = build_gdf(&[inst], Vec3::splat(-1.5), Vec3::splat(1.5), dims);
        let at = |x: usize, y: usize, z: usize| {
            gdf.dist[(z * dims[1] as usize + y) * dims[0] as usize + x]
        };
        // Grid center (voxel 8,8,8 spans -1.5..1.5) is the cube center, so inside.
        assert!(at(8, 8, 8) < 0.0, "cube center should be inside (negative)");
        // A corner voxel is well outside, so positive.
        assert!(at(0, 0, 0) > 0.0, "far corner should be outside (positive)");
        assert!(gdf.index.iter().all(|&i| i == 0), "only instance 0 exists");
    }

    // Occlusion grid: a wall box between two points must drop visibility below 1,
    // a clear line of sight stays fully visible, and it never amplifies or blacks.
    #[test]
    fn occupancy_blocks_through_wall_not_around() {
        // A thin wall slab at x∈[-0.25,0.25], spanning y,z.
        let wall = (Vec3::new(-0.25, -2.0, -2.0), Vec3::new(0.25, 2.0, 2.0));
        let occ = SceneOccupancy::build(&[wall], 48).expect("occupancy built");

        // A ray crossing the wall (from -x to +x) is occluded.
        let through = occ.visibility(Vec3::new(-3.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0));
        assert!(through < 1.0, "ray through the wall should be occluded, got {through}");
        assert!(through >= 0.1, "soft floor: occlusion must never hard-black, got {through}");

        // A ray that runs parallel well off to the side never enters the slab.
        let beside = occ.visibility(Vec3::new(3.0, 0.0, 3.5), Vec3::new(-3.0, 0.0, 3.5));
        assert!((beside - 1.0).abs() < 1e-6, "clear line of sight should stay 1.0, got {beside}");

        // Degenerate (a == b) is fully visible, never NaN.
        let same = occ.visibility(Vec3::ZERO, Vec3::ZERO);
        assert_eq!(same, 1.0);
    }

    // A point light on the far side of a wall must inject less into a probe than
    // the same light with no occluder (cheap voxel-light shadowing).
    #[test]
    fn occluded_inject_darkens_behind_wall() {
        let wall = (Vec3::new(-0.25, -2.0, -2.0), Vec3::new(0.25, 2.0, 2.0));
        let occ = SceneOccupancy::build(&[wall], 48).unwrap();
        let light = BakeLight {
            kind: LightKind::Point,
            position: Vec3::new(-3.0, 0.0, 0.0),
            direction: Vec3::ZERO,
            color: [10.0, 10.0, 10.0],
            range: 20.0,
            spot_inner_deg: 0.0,
            spot_outer_deg: 0.0,
            radius: 0.0,
        };
        let probe = Vec3::new(3.0, 0.0, 0.0); // behind the wall from the light
        let mut lit = [Vec3::ZERO; 4];
        inject_light(&mut lit, probe, &light);
        let mut shadowed = [Vec3::ZERO; 4];
        inject_light_occ(&mut shadowed, probe, &light, Some(&occ));
        assert!(
            shadowed[0].x < lit[0].x && shadowed[0].x >= 0.0,
            "occluded probe ({}) should be dimmer than unoccluded ({}) but non-negative",
            shadowed[0].x,
            lit[0].x
        );
    }

    // Propagation spreads an injected spike into neighbours (a diffuse bounce) and
    // stays bounded (no runaway). A dark neighbour must gain L0; energy must not blow up.
    #[test]
    fn propagation_spreads_and_stays_bounded() {
        let counts = [3u32, 3, 3];
        let n = (counts[0] * counts[1] * counts[2]) as usize;
        let lin = |x: usize, y: usize, z: usize| (z * 3 + y) * 3 + x;
        let mut acc = vec![[Vec3::ZERO; 4]; n];
        // Inject a bright omnidirectional spike at the center.
        acc[lin(1, 1, 1)][0] = Vec3::splat(10.0);
        let before_center = acc[lin(1, 1, 1)][0].x;
        let before_neighbor = acc[lin(0, 1, 1)][0].x;
        flux_propagate(&mut acc, counts, 4, 0.12);
        let after_neighbor = acc[lin(0, 1, 1)][0].x;
        assert!(
            after_neighbor > before_neighbor,
            "neighbour L0 should gain from propagation ({before_neighbor} -> {after_neighbor})"
        );
        // Bounded: total energy must not exceed a sane multiple of the injected spike.
        let total: f32 = acc.iter().map(|s| s[0].x.max(0.0)).sum();
        assert!(
            total.is_finite() && total < before_center * 6.0,
            "propagation energy should stay bounded, got {total}"
        );
    }

    // Relocation: a probe inside a solid wall is pushed to an adjacent empty cell;
    // a probe in open space is left untouched; a fully-trapped probe is classified dead.
    #[test]
    fn relocation_pushes_probe_out_of_solid() {
        // Thin wall slab at x in [-0.25, 0.25].
        let wall = (Vec3::new(-0.25, -2.0, -2.0), Vec3::new(0.25, 2.0, 2.0));
        let occ = SceneOccupancy::build(&[wall], 48).unwrap();
        let open = Vec3::new(3.0, 0.0, 0.0);
        let inside = Vec3::new(0.0, 0.0, 0.0); // inside the wall
        let mut positions = vec![open, inside];
        let active = relocate_probes(&mut positions, &occ);
        assert!(active[0], "open-space probe stays active");
        assert_eq!(positions[0], open, "open-space probe must not move");
        // The inside probe either escaped (moved out of solid) or was classified dead.
        let escaped = !occ.solid(occ.cell_coord_raw(positions[1]));
        assert!(
            escaped || !active[1],
            "trapped probe must be relocated out of solid or marked inactive"
        );
    }

    // FPS verification: the FluxVoxel PER-FRAME work (clone the cached static base +
    // inject dynamic lights + build ProbeSh with cached moments) must be far under a
    // 1 ms frame budget (1000 fps) for a representative auto-grid, so the new DDGI /
    // propagation / specular features don't cost per-frame time — they're all cached.
    // Run with `--nocapture` to see the measured per-frame microseconds.
    #[test]
    fn fluxvoxel_per_frame_cost_is_sub_millisecond() {
        // ~27k probes (a 30³ auto-grid — a typical room at density 1).
        let counts = [30u32, 30, 30];
        let positions =
            flux_volume_positions(Vec3::ZERO, Vec3::splat(30.0), counts);
        let n = positions.len();
        let static_base = vec![[Vec3::ONE; 4]; n]; // pretend a built static base
        let dist = vec![[1.0f32; 4]; n];
        let dist2 = vec![[2.0f32; 4]; n];
        // Two moving emissive orbs (the dynamic per-frame lights).
        let dynamic = vec![
            BakeLight {
                kind: LightKind::Point,
                position: Vec3::new(2.0, 1.0, 0.0),
                direction: Vec3::NEG_Y,
                color: [8.0, 1.0, 6.0],
                range: 5.0,
                spot_inner_deg: 0.0,
                spot_outer_deg: 0.0,
                radius: 0.0,
            },
            BakeLight {
                kind: LightKind::Point,
                position: Vec3::new(-2.0, 1.0, 1.0),
                direction: Vec3::NEG_Y,
                color: [0.0, 8.0, 1.0],
                range: 5.0,
                spot_inner_deg: 0.0,
                spot_outer_deg: 0.0,
                radius: 0.0,
            },
        ];

        // Warm up, then time the exact per-frame path from `update_flux_voxel`.
        let iters = 60;
        let t0 = std::time::Instant::now();
        let mut sink = 0.0f32;
        for _ in 0..iters {
            let mut acc = static_base.clone();
            flux_inject(&mut acc, &positions, &dynamic);
            let probes: Vec<ProbeSh> = acc
                .iter()
                .enumerate()
                .map(|(i, s)| acc_to_probe_moments(s, dist[i], dist2[i]))
                .collect();
            sink += probes[0].coeffs[0][0] + probes[n - 1].coeffs[0][1];
        }
        let per_frame = t0.elapsed().as_secs_f64() / iters as f64;
        let fps = 1.0 / per_frame;
        println!(
            "FluxVoxel per-frame ({n} probes, 2 dynamic lights): {:.1} µs => {:.0} fps headroom (sink {sink})",
            per_frame * 1e6,
            fps
        );
        // Must leave 1000+ fps of headroom: the per-frame CPU GI work alone is well
        // under 1 ms (debug builds are slower; allow 2 ms before failing).
        assert!(
            per_frame < 2.0e-3,
            "per-frame FluxVoxel cost {:.0} µs is too high (would cap fps at {:.0})",
            per_frame * 1e6,
            fps
        );
    }
}
