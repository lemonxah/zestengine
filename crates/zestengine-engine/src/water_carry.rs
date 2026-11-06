//! Carried water: what Water Containers do in Play. Each fixed step a
//! container's water settles under its effective gravity (gravity minus the
//! container's own acceleration, so swinging sloshes it and stopping a swing
//! throws it out), pours what no longer fits over the rim as a stream of
//! drops, fills through an opening dipped under a water surface, and weighs
//! its rigid body down. Drops carry their volume: one landing in another
//! container's opening adds to it, one hitting a water body pushes its
//! surface, anything else splashes and is lost.

use std::collections::HashMap;

use zestengine_core::{LITRES_PER_M3, ObjectId, Water, WaterContainer, WaterPush};
use glam::{Mat4, Quat, Vec2, Vec3};

use crate::physics::{PhysicsWorld, WaterStep};
use crate::scene::LoadedScene;

/// Drops in flight across all containers; past it new water merges into
/// existing drops.
const MAX_DROPS: usize = 500;
/// Litres one drop carries at most when it leaves the rim.
const DROP_LITRES: f32 = 0.04;
const DROPS_PER_STEP: usize = 3;
/// Drops ignore their own container this long after leaving it (s).
const OWN_GRACE: f32 = 0.2;
const DROP_LIFE: f32 = 8.0;
/// Discharge coefficient for flow over the rim and through the opening.
const DISCHARGE: f32 = 0.6;
/// Largest acceleration the water feels (m/s^2): contact spikes beyond
/// this don't empty a container in one step.
const MAX_ACCEL: f32 = 80.0;
/// Litres a falling waterfall streak carries into a container it lands in.
pub const SHEET_LITRES: f32 = 0.2;

/// A drop to draw (world space, interpolated to the frame).
#[derive(Clone, Copy, Debug)]
pub struct PourDrop {
    pub pos: Vec3,
    pub vel: Vec3,
    pub size: f32,
    pub color: [f32; 3],
}

#[derive(Clone, Copy, Debug)]
struct Drop {
    pos: Vec3,
    prev: Vec3,
    vel: Vec3,
    litres: f32,
    color: [f32; 3],
    age: f32,
    source: ObjectId,
}

/// How a container moved over the last frames (world, box centre).
#[derive(Clone, Copy, Debug)]
struct Motion {
    pos: Vec3,
    rot: Quat,
    vel: Vec3,
    ang: Vec3,
    accel: Vec3,
    /// Down as the water feels it, rocking toward the effective gravity
    /// like the surface's first slosh mode, and how fast it's turning.
    down: Vec3,
    down_rate: Vec3,
    tracked: bool,
}

impl Motion {
    fn at(pos: Vec3, rot: Quat, down: Vec3) -> Self {
        Self { pos, rot, vel: Vec3::ZERO, ang: Vec3::ZERO, accel: Vec3::ZERO, down, down_rate: Vec3::ZERO, tracked: false }
    }

    /// Rock `down` toward `target` (the effective gravity over g: longer
    /// pulls harder, near zero in free fall leaves it be) for `dt`: natural
    /// frequency `omega`, damping ratio `zeta`.
    fn slosh(&mut self, target: Vec3, omega: f32, zeta: f32, dt: f32) {
        let acc = (target - self.down) * (omega * omega) - self.down_rate * (2.0 * zeta * omega);
        self.down_rate += acc * dt;
        let fallback = self.down;
        self.down = (self.down + self.down_rate * dt).normalize_or(fallback);
        self.down_rate -= self.down * self.down.dot(self.down_rate);
    }

    fn point_velocity(&self, p: Vec3) -> Vec3 {
        self.vel + self.ang.cross(p - self.pos)
    }
}

/// One container's state for a step.
struct Vessel {
    index: usize,
    id: ObjectId,
    frame: Mat4,
    inv: Mat4,
    half: Vec3,
    color: [f32; 3],
}

#[derive(Default)]
pub struct CarrySim {
    drops: Vec<Drop>,
    motion: HashMap<ObjectId, Motion>,
    rng: u64,
}

fn smooth(dt: f32, tau: f32) -> f32 {
    1.0 - (-dt / tau.max(1e-3)).exp()
}

fn vessels(scene: &LoadedScene, world: &[Mat4]) -> Vec<Vessel> {
    let mut out = Vec::new();
    for (i, o) in scene.objects.iter().enumerate() {
        if !scene.is_active(i) {
            continue;
        }
        if let Some(c) = o.components.iter().find_map(|c| c.as_any().downcast_ref::<WaterContainer>()) {
            let (frame, half) = c.frame(world[i]);
            out.push(Vessel { index: i, id: o.id, frame, inv: frame.inverse(), half, color: c.color });
        }
    }
    out
}

/// Water bodies (index, world matrix).
fn water_bodies(scene: &LoadedScene, world: &[Mat4]) -> Vec<(usize, Mat4)> {
    (0..scene.objects.len())
        .filter(|&i| scene.is_active(i) && scene.objects[i].components.iter().any(|c| c.as_any().is::<Water>()))
        .map(|i| (i, world[i]))
        .collect()
}

fn water_of(scene: &LoadedScene, i: usize) -> Option<&Water> {
    scene.objects[i].components.iter().find_map(|c| c.as_any().downcast_ref::<Water>())
}

/// The surface over `p`: water index, its matrix and the height.
fn surface_at(scene: &LoadedScene, waters: &[(usize, Mat4)], p: Vec3, t: f32) -> Option<(usize, Mat4, f32)> {
    waters.iter().find_map(|(i, m)| water_of(scene, *i)?.height_at(*m, p, t).map(|(h, _)| (*i, *m, h)))
}

/// Push water body `wi` at world `p`.
fn push_water(scene: &LoadedScene, out: &mut WaterStep, wi: usize, m: Mat4, p: Vec3, radius: f32, strength: f32) {
    if water_of(scene, wi).is_some_and(|w| w.interactive) && strength.abs() > 1e-3 {
        let l = m.inverse().transform_point3(p);
        out.fx_pushes.push((wi, WaterPush { at: Vec2::new(l.x, l.z), radius, strength }));
    }
}

impl CarrySim {
    pub fn reset(&mut self) {
        self.drops.clear();
        self.motion.clear();
    }

    fn rand(&mut self) -> f32 {
        if self.rng == 0 {
            self.rng = 0x2545F4914F6CDD1D;
        }
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng.wrapping_mul(0x9E3779B97F4A7C15) >> 40) as f32 / (1u64 << 24) as f32
    }

    fn sym(&mut self) -> f32 {
        self.rand() * 2.0 - 1.0
    }

    /// Follow every container's motion over a frame of `dt` (its pose as
    /// drawn: held, kinematic or simulated alike).
    pub fn track(&mut self, scene: &LoadedScene, dt: f32) {
        let any = scene.objects.iter().any(|o| o.components.iter().any(|c| c.as_any().is::<WaterContainer>()));
        if dt <= 0.0 || (!any && self.motion.is_empty()) {
            return;
        }
        let world = scene.world_transforms();
        let down = Vec3::NEG_Y;
        let vs = vessels(scene, &world);
        self.motion.retain(|id, _| vs.iter().any(|v| v.id == *id));
        for v in vs {
            let (_, rot, pos) = v.frame.to_scale_rotation_translation();
            let m = self.motion.entry(v.id).or_insert_with(|| Motion::at(pos, rot, down));
            let vel = (pos - m.pos) / dt;
            if vel.length() > 60.0 {
                // Teleported: start over from here.
                *m = Motion { down: m.down, down_rate: m.down_rate, ..Motion::at(pos, rot, down) };
                continue;
            }
            let dq = rot * m.rot.inverse();
            let dq = if dq.w < 0.0 { -dq } else { dq };
            let (axis, angle) = dq.to_axis_angle();
            let ang = if angle.is_finite() { axis * (angle / dt) } else { Vec3::ZERO };
            let prev_vel = m.vel;
            m.vel = m.vel.lerp(vel, smooth(dt, 0.02));
            m.ang = m.ang.lerp(ang, smooth(dt, 0.02));
            if m.tracked {
                let accel = ((m.vel - prev_vel) / dt).clamp_length_max(MAX_ACCEL);
                m.accel = m.accel.lerp(accel, smooth(dt, 0.04));
            }
            m.tracked = true;
            m.pos = pos;
            m.rot = rot;
        }
    }

    /// Litres in flight (tests and the inspector).
    pub fn litres_in_flight(&self) -> f32 {
        self.drops.iter().map(|d| d.litres).sum()
    }

    /// The drops to draw, `alpha` (0..1) of the way from the last step to
    /// the next.
    pub fn snapshot(&self, alpha: f32) -> Vec<PourDrop> {
        self.drops
            .iter()
            .map(|d| {
                let r = (d.litres.max(1e-4) / LITRES_PER_M3 * 3.0 / (4.0 * std::f32::consts::PI)).cbrt();
                PourDrop { pos: d.prev.lerp(d.pos, alpha), vel: d.vel, size: (r * 2.6).clamp(0.012, 0.12), color: d.color }
            })
            .collect()
    }

    /// One fixed step: move the drops, settle, pour and fill every
    /// container, and hand their water's weight to `physics`. Pushes on
    /// water bodies go into `out`.
    pub fn step(&mut self, scene: &mut LoadedScene, world: &[Mat4], mut physics: Option<&mut PhysicsWorld>, t: f32, dt: f32, out: &mut WaterStep) {
        let vs = vessels(scene, world);
        if vs.is_empty() && self.drops.is_empty() {
            return;
        }
        let gravity = physics.as_deref().map_or(Vec3::new(0.0, -zestengine_core::GRAVITY, 0.0), |p| p.gravity());
        let g: f32 = gravity.length().max(1e-3);
        let waters = water_bodies(scene, world);
        let mut caught: HashMap<usize, f32> = HashMap::new();
        self.move_drops(scene, &vs, &waters, physics.as_deref(), gravity, t, dt, out, &mut caught);

        for v in &vs {
            let i = v.index;
            let Some(c) = scene.objects[i].components.iter().find_map(|c| c.as_any().downcast_ref::<WaterContainer>()) else { continue };
            let (scale, _, _) = world[i].to_scale_rotation_translation();
            let half = v.half;
            let full = 8.0 * half.x * half.y * half.z;
            let mut rt = c.runtime.clone();
            if !rt.started {
                rt.started = true;
                rt.litres = c.initial_fill.clamp(0.0, 1.0) * full * LITRES_PER_M3;
            }
            rt.litres += std::mem::take(&mut rt.add) + caught.get(&i).copied().unwrap_or(0.0);
            let (pour_rate, inflow, slosh) = (c.pour_rate.max(0.0), c.inflow.max(0.0), c.slosh);
            let own_mass = scene.objects[i].components.iter().find_map(|c| c.as_any().downcast_ref::<zestengine_core::RigidBody>()).map_or(0.0, |r| r.mass);

            // The water rocks toward the effective gravity (what's left of
            // gravity after the container's own acceleration) at the slosh
            // frequency of water this wide and deep.
            let (_, rot, centre) = v.frame.to_scale_rotation_translation();
            // More than fits (poured in, added) pours back out over the rim.
            let mut vol = (rt.litres / LITRES_PER_M3).max(0.0);
            let m = self.motion.entry(v.id).or_insert_with(|| Motion::at(centre, rot, gravity / g));
            let target = (gravity - m.accel) / g;
            let width = 2.0 * half.x.max(half.z);
            let depth = (vol.min(full) / (4.0 * half.x * half.z)).max(0.02);
            let k = std::f32::consts::PI / width;
            let omega = (g * k * (k * depth).tanh()).sqrt().clamp(1.0, 40.0);
            let zeta = 0.05 + (1.0 - slosh.clamp(0.0, 1.0)) * 0.85;
            m.slosh(target, omega, zeta, dt);
            let motion = *m;
            let surge = motion.down_rate.length() * width * 0.5;
            let up = v.inv.transform_vector3(-motion.down).normalize_or(Vec3::Y);

            // Opening under a water surface: flow in (or out) toward the
            // outside level, by how much of the opening is under.
            let mouth = v.frame.transform_point3(Vec3::new(0.0, half.y, 0.0));
            let mut submerged = 0.0;
            let mut filling = 0.0;
            if let Some((wi, wm, h_out)) = surface_at(scene, &waters, mouth, t) {
                let mut under = 0;
                for a in [-2.0f32 / 3.0, 0.0, 2.0 / 3.0] {
                    for b in [-2.0f32 / 3.0, 0.0, 2.0 / 3.0] {
                        if v.frame.transform_point3(Vec3::new(a * half.x, half.y, b * half.z)).y < h_out {
                            under += 1;
                        }
                    }
                }
                submerged = under as f32 / 9.0;
                if submerged > 0.0 {
                    let level_eq = up.dot(v.inv.transform_point3(Vec3::new(mouth.x, h_out, mouth.z)));
                    let target = zestengine_core::volume_below(half, up, level_eq).0;
                    let level = zestengine_core::level_for_volume(half, up, vol);
                    let head = (level_eq - level).abs();
                    let area = 4.0 * half.x * half.z * submerged;
                    let q = DISCHARGE * area * (2.0 * g * head).sqrt() * inflow;
                    vol = vol.min(full);
                    let dv = (target - vol).clamp(-q * dt, q * dt);
                    vol += dv;
                    filling = dv / dt * LITRES_PER_M3;
                    // The water drawn in dips the surface over the mouth.
                    push_water(scene, out, wi, wm, mouth, half.x.max(half.z), (dv / dt / (4.0 * half.x * half.z)).clamp(-1.0, 1.0));
                }
            }

            // Over the rim: flow like water over a weir, faster the deeper
            // the rim dips under the surface, plus what a slosh carries
            // over it. Gameplay pours come on top.
            let mut spilled = 0.0;
            let mut emit: Vec<(Vec3, f32)> = Vec::new();
            let held = zestengine_core::held_volume(half, up);
            if submerged <= 0.0 && vol > held {
                let level = zestengine_core::level_for_volume(half, up, vol);
                let rim = zestengine_core::rim_level(half, up);
                let (lip, wet) = zestengine_core::wet_lip(half, up, level, 24);
                let head = (level - rim).max(0.0);
                let q = pour_rate * wet.max(0.02) * (DISCHARGE * (2.0 / 3.0) * (2.0 * g).sqrt() * head.powf(1.5) + head * surge);
                let excess = vol - held;
                let mut s = (q * dt).min(excess);
                if excess - s < 1e-5 {
                    s = excess;
                }
                spilled += s;
                emit = lip;
                if emit.is_empty() {
                    emit.push((zestengine_core::spill_point(half, up), 0.01));
                }
            }
            if rt.pour > 0.0 {
                let want = (rt.pour / LITRES_PER_M3).min(vol - spilled).max(0.0);
                let s = want.min((want * 5.0).max(0.002) * dt);
                spilled += s;
                rt.pour = if vol - spilled <= 1e-6 { 0.0 } else { (rt.pour - s * LITRES_PER_M3).max(0.0) };
                if emit.is_empty() {
                    emit.push((zestengine_core::spill_point(half, up), 0.03));
                }
            }
            vol = (vol - spilled).clamp(0.0, full);
            if spilled > 0.0 {
                self.emit(v, &motion, g, surge, spilled * LITRES_PER_M3, &emit);
            }

            let level = zestengine_core::level_for_volume(half, up, vol);
            let (_, centroid) = zestengine_core::volume_below(half, up, level);
            rt.litres = vol * LITRES_PER_M3;
            rt.capacity = full * LITRES_PER_M3;
            rt.scale = scale.to_array();
            rt.up = up;
            rt.level = level;
            rt.spilling = spilled / dt * LITRES_PER_M3;
            rt.filling = filling;
            if let Some(p) = physics.as_deref_mut() {
                let fill = (vol / full).clamp(0.0, 1.0);
                let com = Vec3::from(c.center) * scale + centroid;
                p.set_carried_water(v.id, own_mass, vol * LITRES_PER_M3, com, half * Vec3::new(1.0, fill, 1.0));
            }
            if let Some(c) = scene.objects[i].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<WaterContainer>()) {
                c.runtime = rt;
            }
        }
    }

    /// New drops for `litres` leaving over the wet lip points (box frame,
    /// with their depth under the surface).
    fn emit(&mut self, v: &Vessel, motion: &Motion, g: f32, surge: f32, litres: f32, lip: &[(Vec3, f32)]) {
        let n = ((litres / DROP_LITRES).ceil() as usize).clamp(1, DROPS_PER_STEP);
        let each = litres / n as f32;
        let fling = (-motion.accel * 0.06).clamp_length_max(4.0);
        for _ in 0..n {
            let k = ((self.rand() * lip.len() as f32) as usize).min(lip.len() - 1);
            let (p, head) = lip[k];
            let outward = Vec3::new(p.x, 0.0, p.z).normalize_or(Vec3::X);
            let world_p = v.frame.transform_point3(p + outward * 0.01);
            let out_w = v.frame.transform_vector3(outward);
            let jitter = Vec3::new(self.sym(), self.sym(), self.sym()) * 0.12;
            let vel = motion.point_velocity(world_p) + out_w * ((g * head.max(0.0)).sqrt() * 0.8 + 0.15 + surge.min(4.0)) + fling + jitter;
            let drop = Drop { pos: world_p, prev: world_p, vel, litres: each, color: v.color, age: 0.0, source: v.id };
            if self.drops.len() < MAX_DROPS {
                self.drops.push(drop);
            } else if let Some(d) = self.drops.iter_mut().rev().find(|d| d.source == v.id) {
                d.litres += each;
            } else {
                let k = ((self.rand() * self.drops.len() as f32) as usize).min(self.drops.len() - 1);
                self.drops[k].litres += each;
            }
        }
    }

    /// Fly the drops: into a container's opening (adds to `caught`, by
    /// object index), onto a water surface (a push), or against anything
    /// solid (a splash), else onward.
    #[allow(clippy::too_many_arguments)]
    fn move_drops(
        &mut self,
        scene: &mut LoadedScene,
        vs: &[Vessel],
        waters: &[(usize, Mat4)],
        physics: Option<&PhysicsWorld>,
        gravity: Vec3,
        t: f32,
        dt: f32,
        out: &mut WaterStep,
        caught: &mut HashMap<usize, f32>,
    ) {
        if self.drops.is_empty() {
            return;
        }
        for d in &mut self.drops {
            d.prev = d.pos;
            d.vel = d.vel * (1.0 - 0.05 * dt) + gravity * dt;
            d.pos += d.vel * dt;
            d.age += dt;
        }
        let mut alive = vec![true; self.drops.len()];
        for (k, d) in self.drops.iter().enumerate() {
            for v in vs {
                if v.id == d.source && d.age < OWN_GRACE {
                    continue;
                }
                let (a, b) = (v.inv.transform_point3(d.prev), v.inv.transform_point3(d.pos));
                let h = v.half;
                // Down through the opening, or already inside and falling.
                let through = a.y >= h.y && b.y < h.y && {
                    let s = (a.y - h.y) / (a.y - b.y).max(1e-6);
                    let p = a.lerp(b, s);
                    p.x.abs() <= h.x && p.z.abs() <= h.z
                };
                if through || (zestengine_core::inside_box(h, b) && b.y < a.y) {
                    *caught.entry(v.index).or_insert(0.0) += d.litres;
                    alive[k] = false;
                    break;
                }
            }
            if !alive[k] {
                continue;
            }
            if let Some((wi, wm, h)) = surface_at(scene, waters, d.pos, t)
                && d.pos.y <= h
            {
                let at = Vec3::new(d.pos.x, h, d.pos.z);
                let strength = (-d.vel.y).max(0.5) * 0.35 * (d.litres / DROP_LITRES).sqrt().min(3.0);
                push_water(scene, out, wi, wm, at, 0.15, strength);
                if d.vel.y < -2.0 {
                    scene.water_drips.push((at, d.vel, d.litres));
                }
                alive[k] = false;
            }
        }
        if let Some(p) = physics {
            let idx: Vec<usize> = (0..self.drops.len()).filter(|&k| alive[k]).collect();
            let segs: Vec<(Vec3, Vec3)> = idx.iter().map(|&k| (self.drops[k].prev, self.drops[k].pos)).collect();
            for (&k, hit) in idx.iter().zip(p.cast_segments(&segs)) {
                let d = &self.drops[k];
                if let Some((at, id)) = hit
                    && !(id == d.source && d.age < OWN_GRACE)
                {
                    scene.water_drips.push((at, d.vel, d.litres));
                    alive[k] = false;
                }
            }
        }
        let mut k = 0;
        self.drops.retain(|d| {
            let keep = alive[k] && d.age < DROP_LIFE && d.pos.y > -1000.0;
            k += 1;
            keep
        });
    }
}

/// Most containers whose water is drawn.
const MAX_SURFACES: usize = 32;
/// Corners a surface polygon has at most (a plane cuts a box in six).
const SURFACE_RING: usize = 6;
const MAX_DRIPS: usize = 300;

#[derive(Clone, Copy, Debug)]
struct Drip {
    pos: Vec3,
    vel: Vec3,
    age: f32,
    life: f32,
    size: f32,
}

struct CarryGpu {
    drops: zestengine_render::MeshHandle,
    drop_material: zestengine_render::MaterialHandle,
    surfaces: zestengine_render::MeshHandle,
    surface_material: zestengine_render::MaterialHandle,
}

/// Draws carried water: the surface inside each container and the streams
/// and droplets poured out. Also keeps containers' inspector readouts and
/// fit requests current, and lets falling waterfall water land in them.
#[derive(Default)]
pub struct CarryFx {
    gpu: Option<CarryGpu>,
    failed: bool,
    drips: Vec<Drip>,
    rng: u64,
}

impl CarryFx {
    fn rand(&mut self) -> f32 {
        if self.rng == 0 {
            self.rng = 0x9E3779B97F4A7C15;
        }
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Once per frame. `fx` is the water effects' particle pool: waterfall
    /// streaks that fall into a container's opening during Play are caught.
    pub fn update(&mut self, renderer: &mut zestengine_render::Renderer, scene: &mut LoadedScene, cam: Mat4, playing: bool, dt: f32, fx: &mut [crate::water_fx::FxParticle]) {
        let pours = std::mem::take(&mut scene.water_pours);
        let drips = std::mem::take(&mut scene.water_drips);
        let has = scene.objects.iter().any(|o| o.components.iter().any(|c| c.as_any().is::<WaterContainer>()));
        if !has && pours.is_empty() && drips.is_empty() && self.drips.is_empty() {
            return;
        }
        let world = scene.world_transforms();
        let surfaces = self.containers(scene, &world, playing, fx);
        for (at, vel, litres) in drips {
            self.splatter(at, vel, litres);
        }
        self.step_drips(dt.min(0.05));
        if self.gpu.is_none() && !self.failed {
            self.gpu = create_gpu(renderer).map_err(|e| tracing::error!("carried water buffers: {e:#}")).ok();
            self.failed = self.gpu.is_none();
        }
        let Some(gpu) = &self.gpu else { return };
        let (verts, lo, hi) = drop_vertices(&pours, &self.drips, cam);
        renderer.update_mesh_vertices(gpu.drops, &verts);
        if hi.x >= lo.x {
            scene.extra_draws.push(extra_draw(gpu.drops, gpu.drop_material, lo, hi));
        }
        let (verts, lo, hi) = surface_vertices(&surfaces);
        renderer.update_mesh_vertices(gpu.surfaces, &verts);
        if hi.x >= lo.x {
            scene.extra_draws.push(extra_draw(gpu.surfaces, gpu.surface_material, lo, hi));
        }
    }

    /// Readouts, fit requests and waterfall catches; returns the surfaces
    /// to draw (world polygon, normal, colour with opacity).
    fn containers(&mut self, scene: &mut LoadedScene, world: &[Mat4], playing: bool, fx: &mut [crate::water_fx::FxParticle]) -> Vec<(Vec<Vec3>, Vec3, [f32; 4])> {
        let mut out = Vec::new();
        for (i, world) in world.iter().enumerate() {
            if !scene.is_active(i) {
                continue;
            }
            let bounds = scene.render_mesh_bounds(i);
            let Some(c) = scene.objects[i].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<WaterContainer>()) else { continue };
            if c.runtime.fit_requested {
                c.runtime.fit_requested = false;
                if let Some((lo, hi)) = bounds {
                    c.fit_to_bounds(lo, hi);
                }
            }
            let (scale, _, _) = world.to_scale_rotation_translation();
            c.runtime.scale = scale.to_array();
            if !c.runtime.started {
                c.runtime.capacity = c.capacity_litres(scale);
            }
            if playing && c.runtime.started {
                let (frame, half) = c.frame(*world);
                let inv = frame.inverse();
                for p in fx.iter_mut().filter(|p| p.kind == crate::water_fx::FxKind::Sheet && p.age < p.life) {
                    let l = inv.transform_point3(p.pos);
                    if l.x.abs() <= half.x && l.z.abs() <= half.z && l.y <= half.y && l.y > -half.y {
                        c.runtime.add += SHEET_LITRES;
                        p.age = p.life;
                    }
                }
            }
            if out.len() < MAX_SURFACES
                && let Some((frame, half, up, level)) = c.surface(*world)
            {
                let poly = zestengine_core::surface_polygon(half, up, level);
                if poly.len() >= 3 {
                    let pts = poly.iter().map(|p| frame.transform_point3(*p)).collect();
                    let n = frame.transform_vector3(up).normalize_or(Vec3::Y);
                    out.push((pts, n, [c.color[0], c.color[1], c.color[2], c.opacity.clamp(0.0, 1.0)]));
                }
            }
        }
        out
    }

    /// A few droplets where poured water hit something.
    fn splatter(&mut self, at: Vec3, vel: Vec3, litres: f32) {
        let n = 1 + ((litres / DROP_LITRES).sqrt() * 1.5) as usize;
        let speed = vel.length().min(8.0);
        for _ in 0..n.min(4) {
            if self.drips.len() >= MAX_DRIPS {
                return;
            }
            let a = self.rand() * std::f32::consts::TAU;
            let out = Vec3::new(a.cos(), 0.0, a.sin()) * (0.3 + self.rand() * 0.7) * (0.3 + speed * 0.15);
            let v = out + Vec3::Y * (0.4 + self.rand() * 0.8) * (0.3 + speed * 0.12) + Vec3::new(vel.x, 0.0, vel.z) * 0.2;
            let life = 0.35 + self.rand() * 0.3;
            let size = 0.012 + self.rand() * 0.015;
            self.drips.push(Drip { pos: at + Vec3::Y * 0.01, vel: v, age: 0.0, life, size });
        }
    }

    fn step_drips(&mut self, dt: f32) {
        let g = Vec3::new(0.0, -zestengine_core::GRAVITY, 0.0);
        for d in &mut self.drips {
            d.age += dt;
            d.vel += g * dt;
            d.pos += d.vel * dt;
        }
        self.drips.retain(|d| d.age < d.life);
    }
}

fn extra_draw(mesh: zestengine_render::MeshHandle, material: zestengine_render::MaterialHandle, lo: Vec3, hi: Vec3) -> zestengine_render::DrawCmd {
    zestengine_render::DrawCmd {
        gi_source: None,
        mesh,
        material,
        transform: Mat4::IDENTITY,
        highlight: 0.0,
        mesh_center: (lo + hi) * 0.5,
        bound_radius: ((hi - lo) * 0.5).length(),
        lightmap_layer: -1,
        lightmap_size: 0,
        static_geometry: false,
        planar_index: -1,
        occluded: false,
        shadow_casting: zestengine_render::ShadowCasting::On,
        instance_color: None,
        instance_param: None,
    }
}

fn create_gpu(renderer: &mut zestengine_render::Renderer) -> anyhow::Result<CarryGpu> {
    let tex = renderer.upload_texture(&soft_dot())?;
    let drops = renderer.upload_mesh_skinned(&crate::particles::quads(MAX_DROPS + MAX_DRIPS))?;
    let drop_material = crate::render_util::unlit_material(renderer, Some(tex), "carried_water_drops")?;
    let surfaces = renderer.upload_mesh_skinned(&surface_mesh())?;
    let normal = renderer.upload_texture(&ripple_normals())?;
    let features = zestengine_render::MaterialFeatures { alpha_mode: zestengine_render::AlphaMode::Blend, double_sided: true, ..Default::default() };
    let params = zestengine_render::MaterialParams {
        base_color: [1.0, 1.0, 1.0, 1.0],
        roughness: 0.06,
        metallic: 0.0,
        normal_strength: 0.5,
        base_scroll: [0.035, 0.02],
        ..Default::default()
    };
    let desc = zestengine_render::MaterialDesc { name: "carried_water".into(), params, features, normal: Some(normal), ..Default::default() };
    let surface_material = renderer.create_material(&desc)?;
    renderer.set_material_features(surface_material, features);
    Ok(CarryGpu { drops, drop_material, surfaces, surface_material })
}

/// Stretched camera-facing quads for the pour drops and splash droplets,
/// back to front and padded to the pool, with their bounds.
fn drop_vertices(pours: &[PourDrop], drips: &[Drip], cam: Mat4) -> (Vec<zestengine_render::Vertex>, Vec3, Vec3) {
    let cam_pos = cam.w_axis.truncate();
    let right = cam.x_axis.truncate().normalize_or(Vec3::X);
    let dot = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let mut items: Vec<(Vec3, Vec3, f32, [f32; 4])> = pours
        .iter()
        .map(|d| (d.pos, d.vel, d.size, [d.color[0] * 0.5 + 0.4, d.color[1] * 0.5 + 0.45, d.color[2] * 0.5 + 0.5, 0.8]))
        .chain(drips.iter().map(|d| (d.pos, d.vel, d.size, [0.8, 0.86, 0.9, 0.7 * (1.0 - d.age / d.life.max(1e-3))])))
        .take(MAX_DROPS + MAX_DRIPS)
        .collect();
    items.sort_by(|a, b| b.0.distance_squared(cam_pos).total_cmp(&a.0.distance_squared(cam_pos)));
    let mut verts = Vec::with_capacity((MAX_DROPS + MAX_DRIPS) * 4);
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for (pos, vel, size, color) in items {
        let along = vel.normalize_or(Vec3::Y);
        let side = along.cross(pos - cam_pos).normalize_or(right);
        let a = along * (vel.length() * 0.035).clamp(1.0, 5.0) * size;
        let b = side * size * 0.5;
        for (c, uv) in [-b + a, b + a, b - a, -b - a].into_iter().zip(dot) {
            verts.push(crate::particles::vertex(pos + c, uv, color));
        }
        let ext = a.abs() + b.abs();
        lo = lo.min(pos - ext);
        hi = hi.max(pos + ext);
    }
    verts.resize((MAX_DROPS + MAX_DRIPS) * 4, crate::particles::vertex(Vec3::ZERO, [0.0, 0.0], [0.0; 4]));
    (verts, lo, hi)
}

/// One fan per container slot: a centre and `SURFACE_RING` corners.
fn surface_mesh() -> zestengine_render::MeshData {
    let per = SURFACE_RING as u32 + 1;
    let mut indices = Vec::with_capacity(MAX_SURFACES * SURFACE_RING * 3);
    for s in 0..MAX_SURFACES as u32 {
        let base = s * per;
        for k in 0..SURFACE_RING as u32 {
            indices.extend_from_slice(&[base, base + 1 + k, base + 1 + (k + 1) % SURFACE_RING as u32]);
        }
    }
    let blank = crate::particles::vertex(Vec3::ZERO, [0.0, 0.0], [0.0; 4]);
    zestengine_render::MeshData { vertices: vec![blank; MAX_SURFACES * per as usize], indices, has_lightmap_uv: false }
}

/// The surfaces' fan vertices (world space), padded, with their bounds.
fn surface_vertices(surfaces: &[(Vec<Vec3>, Vec3, [f32; 4])]) -> (Vec<zestengine_render::Vertex>, Vec3, Vec3) {
    let per = SURFACE_RING + 1;
    let mut verts = Vec::with_capacity(MAX_SURFACES * per);
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for (poly, n, color) in surfaces.iter().take(MAX_SURFACES) {
        let centre = poly.iter().copied().sum::<Vec3>() / poly.len() as f32;
        let t = (poly[0] - centre).normalize_or(Vec3::X);
        let vert = |p: Vec3| zestengine_render::Vertex {
            position: p.to_array(),
            normal: n.to_array(),
            uv: [p.x * 3.0, p.z * 3.0],
            color: *color,
            tangent: [t.x, t.y, t.z, 1.0],
            uv1: [0.0, 0.0],
            joints: [0; 4],
            weights: [0.0; 4],
        };
        verts.push(vert(centre));
        for k in 0..SURFACE_RING {
            let p = poly[k.min(poly.len() - 1)];
            verts.push(vert(p));
            lo = lo.min(p);
            hi = hi.max(p);
        }
    }
    verts.resize(MAX_SURFACES * per, crate::particles::vertex(Vec3::ZERO, [0.0, 0.0], [0.0; 4]));
    (verts, lo, hi)
}

/// Soft white dot with alpha.
fn soft_dot() -> zestengine_render::TextureData {
    const N: u32 = 32;
    let mut pixels = Vec::with_capacity((N * N * 4) as usize);
    for y in 0..N {
        for x in 0..N {
            let d = Vec2::new((x as f32 + 0.5) / N as f32 * 2.0 - 1.0, (y as f32 + 0.5) / N as f32 * 2.0 - 1.0).length();
            let a = (1.0 - d).clamp(0.0, 1.0).powf(1.2);
            pixels.extend_from_slice(&[255, 255, 255, (a * 255.0) as u8]);
        }
    }
    zestengine_render::TextureData { width: N, height: N, pixels, srgb: true, hdr: false }
}

/// Tiling small ripples for the surface's normal map.
fn ripple_normals() -> zestengine_render::TextureData {
    const N: u32 = 64;
    let waves: [(f32, f32, f32, f32); 4] = [(1.0, 2.0, 0.0, 1.0), (-3.0, 1.0, 1.3, 0.6), (2.0, -3.0, 2.1, 0.45), (5.0, 4.0, 0.7, 0.25)];
    let mut pixels = Vec::with_capacity((N * N * 4) as usize);
    for y in 0..N {
        for x in 0..N {
            let (u, v) = (x as f32 / N as f32, y as f32 / N as f32);
            let (mut dx, mut dy) = (0.0, 0.0);
            for (kx, ky, phase, amp) in waves {
                let a = std::f32::consts::TAU * (kx * u + ky * v) + phase;
                dx += amp * kx * a.cos();
                dy += amp * ky * a.cos();
            }
            let n = Vec3::new(-dx * 0.04, -dy * 0.04, 1.0).normalize();
            pixels.extend_from_slice(&[((n.x * 0.5 + 0.5) * 255.0) as u8, ((n.y * 0.5 + 0.5) * 255.0) as u8, ((n.z * 0.5 + 0.5) * 255.0) as u8, 255]);
        }
    }
    zestengine_render::TextureData { width: N, height: N, pixels, srgb: false, hdr: false }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::FIXED_DT;
    use crate::scene::SceneObject;
    use zestengine_assets::ObjectSource;
    use zestengine_core::{BoxCollider, Component, RigidBody};

    fn obj(scene: &mut LoadedScene, pos: Vec3, rot: Quat, components: Vec<Box<dyn Component>>) -> ObjectId {
        let id = ObjectId::new();
        scene.objects.push(SceneObject {
            net_role: Default::default(),
            component_roles: Default::default(),
            id,
            name: String::new(),
            render: None,
            extra_render: Vec::new(),
            source: ObjectSource::Empty,
            enabled: true,
            static_geometry: false,
            layer: 0,
            parent: None,
            translation: pos,
            rotation: rot,
            scale: Vec3::ONE,
            components,
        });
        id
    }

    /// An open-top 1 m box: floor and four walls around a 0.92 x 0.95 x 0.92 hollow.
    fn bucket(mass: f32, fill: f32) -> Vec<Box<dyn Component>> {
        let wall = |center: [f32; 3], size: [f32; 3]| -> Box<dyn Component> { Box::new(BoxCollider { center, size, ..Default::default() }) };
        vec![
            wall([0.48, 0.0, 0.0], [0.04, 1.0, 1.0]),
            wall([-0.48, 0.0, 0.0], [0.04, 1.0, 1.0]),
            wall([0.0, 0.0, 0.48], [0.92, 1.0, 0.04]),
            wall([0.0, 0.0, -0.48], [0.92, 1.0, 0.04]),
            wall([0.0, -0.475, 0.0], [0.92, 0.05, 0.92]),
            Box::new(RigidBody { mass, ..Default::default() }),
            Box::new(WaterContainer { initial_fill: fill, ..Default::default() }),
        ]
    }

    fn container(scene: &LoadedScene, id: ObjectId) -> &WaterContainer {
        let i = scene.index_of(id).unwrap();
        scene.objects[i].components.iter().find_map(|c| c.as_any().downcast_ref::<WaterContainer>()).unwrap()
    }

    fn run(pw: &mut PhysicsWorld, sim: &mut CarrySim, scene: &mut LoadedScene, steps: usize) {
        for k in 0..steps {
            pw.sync_structure(scene);
            let ids = scene.id_index();
            let world = scene.world_transforms();
            pw.push_poses(scene, &ids, &world);
            let t = k as f32 * FIXED_DT;
            let mut water = pw.apply_water(scene, &world, t, &[]);
            sim.step(scene, &world, Some(&mut *pw), t, FIXED_DT, &mut water);
            pw.step();
            pw.reset_forces();
            pw.write_back(scene, &ids, 1.0);
            sim.track(scene, FIXED_DT);
        }
    }

    fn calm_lake(scene: &mut LoadedScene) {
        obj(scene, Vec3::ZERO, Quat::IDENTITY, vec![Box::new(Water { waves: Vec::new(), interactive: false, ..Default::default() })]);
    }

    #[test]
    fn a_full_container_floats_lower_than_an_empty_one() {
        let mut scene = LoadedScene::empty();
        calm_lake(&mut scene);
        // Set down near where each settles, so the plunge doesn't flood one.
        let empty = obj(&mut scene, Vec3::new(-3.0, 0.5, 0.0), Quat::IDENTITY, bucket(150.0, 0.0));
        let half = obj(&mut scene, Vec3::new(3.0, 0.1, 0.0), Quat::IDENTITY, bucket(150.0, 0.5));
        let mut pw = PhysicsWorld::build(&scene);
        let mut sim = CarrySim::default();
        run(&mut pw, &mut sim, &mut scene, 900);
        let y = |id: ObjectId| scene.world_transform(scene.index_of(id).unwrap()).w_axis.y;
        assert!(y(empty) > 0.25, "150 kg on a 1 m3 hull floats high: {}", y(empty));
        assert!(y(half) < y(empty) - 0.3, "about 400 kg more sits about 0.4 m lower: {} vs {}", y(half), y(empty));
        assert!(y(half) > -0.5, "and still floats: {}", y(half));
        let c = container(&scene, half);
        assert!((c.runtime.litres - 0.5 * c.runtime.capacity).abs() < 5.0, "kept its water: {}", c.runtime.litres);
    }

    #[test]
    fn a_container_whose_rim_dips_under_floods_and_sinks() {
        let mut scene = LoadedScene::empty();
        calm_lake(&mut scene);
        // Walls heavier than the water they displace: afloat only while empty.
        let id = obj(&mut scene, Vec3::new(0.0, 0.6, 0.0), Quat::IDENTITY, bucket(400.0, 0.0));
        let mut pw = PhysicsWorld::build(&scene);
        let mut sim = CarrySim::default();
        run(&mut pw, &mut sim, &mut scene, 300);
        let y0 = scene.world_transform(scene.index_of(id).unwrap()).w_axis.y;
        assert!(y0 > -0.2, "floats empty: {y0}");
        assert!(container(&scene, id).runtime.litres < 1.0);
        // Push it down past its rim: it fills and goes under.
        for _ in 0..30 {
            pw.add_force(id, Vec3::new(0.0, -20000.0, 0.0));
            run(&mut pw, &mut sim, &mut scene, 1);
        }
        run(&mut pw, &mut sim, &mut scene, 600);
        let c = container(&scene, id);
        let y1 = scene.world_transform(scene.index_of(id).unwrap()).w_axis.y;
        assert!(c.runtime.litres > 0.8 * c.runtime.capacity, "flooded: {}", c.runtime.litres);
        assert!(y1 < -1.5, "and sank: {y1}");
    }

    #[test]
    fn tipping_a_full_container_empties_it_as_a_stream() {
        let mut scene = LoadedScene::empty();
        // A fixed (no Rigid Body) tub, turned over 120 degrees.
        let mut comps = bucket(0.0, 1.0);
        comps.remove(5);
        let id = obj(&mut scene, Vec3::new(0.0, 5.0, 0.0), Quat::from_rotation_x(120f32.to_radians()), comps);
        let mut pw = PhysicsWorld::build(&scene);
        let mut sim = CarrySim::default();
        let mut peak = 0.0f32;
        let mut first: Option<f32> = None;
        for k in 0..240 {
            run(&mut pw, &mut sim, &mut scene, 1);
            let l = container(&scene, id).runtime.litres;
            peak = peak.max(sim.litres_in_flight());
            if k == 0 {
                first = Some(l);
            }
        }
        let c = container(&scene, id);
        assert!(first.unwrap() > 0.3 * c.runtime.capacity, "pours fast but not all at once: {first:?}");
        assert!(c.runtime.litres < 1.0, "empties: {}", c.runtime.litres);
        assert!(peak > 50.0, "the water left as drops: {peak}");
        assert!(sim.drops.len() <= MAX_DROPS);
    }

    #[test]
    fn a_stream_into_a_second_container_fills_it() {
        let mut scene = LoadedScene::empty();
        let mut src = bucket(0.0, 0.0);
        src.remove(5);
        let mut dst = bucket(0.0, 0.0);
        dst.remove(5);
        // The top one pours off its -Z edge, which lands about 0.9 m out.
        let from = obj(&mut scene, Vec3::new(0.0, 3.0, 0.0), Quat::IDENTITY, src);
        let into = obj(&mut scene, Vec3::new(0.0, 0.0, -0.9), Quat::IDENTITY, dst);
        let mut pw = PhysicsWorld::build(&scene);
        let mut sim = CarrySim::default();
        run(&mut pw, &mut sim, &mut scene, 1);
        // Gameplay fills the top one and pours it out.
        let i = scene.index_of(from).unwrap();
        if let Some(c) = scene.objects[i].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<WaterContainer>()) {
            c.runtime.add = 100.0;
            c.runtime.pour = 60.0;
        }
        run(&mut pw, &mut sim, &mut scene, 600);
        let (a, b) = (container(&scene, from).runtime.litres, container(&scene, into).runtime.litres);
        assert!((a - 40.0).abs() < 1.0, "poured what it was asked to: {a}");
        assert!(b > 50.0 && b <= 60.5, "the stream landed in the one below: {b}");
    }

    #[test]
    fn more_than_fits_runs_over_the_rim() {
        let mut scene = LoadedScene::empty();
        let mut comps = bucket(0.0, 0.0);
        comps.remove(5);
        let id = obj(&mut scene, Vec3::new(0.0, 2.0, 0.0), Quat::IDENTITY, comps);
        let i = scene.index_of(id).unwrap();
        scene.objects[i].scale = Vec3::new(0.28, 0.3, 0.28);
        let mut pw = PhysicsWorld::build(&scene);
        let mut sim = CarrySim::default();
        run(&mut pw, &mut sim, &mut scene, 1);
        if let Some(c) = scene.objects[i].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<WaterContainer>()) {
            c.runtime.add = 30.0;
        }
        run(&mut pw, &mut sim, &mut scene, 300);
        let c = container(&scene, id);
        assert!(c.runtime.capacity > 18.0 && c.runtime.capacity < 20.0, "{}", c.runtime.capacity);
        assert!((c.runtime.litres - c.runtime.capacity).abs() < 0.5, "brimful, the rest ran off: {}", c.runtime.litres);
    }

    #[test]
    fn dipping_the_opening_under_water_fills_it_to_the_outside_level() {
        let mut scene = LoadedScene::empty();
        calm_lake(&mut scene);
        let mut comps = bucket(0.0, 0.0);
        comps.remove(5);
        // Held fixed with its rim 0.2 m under the surface.
        let id = obj(&mut scene, Vec3::new(0.0, -0.7, 0.0), Quat::IDENTITY, comps);
        let mut pw = PhysicsWorld::build(&scene);
        let mut sim = CarrySim::default();
        run(&mut pw, &mut sim, &mut scene, 120);
        let c = container(&scene, id);
        assert!(c.runtime.litres > 0.97 * c.runtime.capacity, "scooped full: {} of {}", c.runtime.litres, c.runtime.capacity);
        assert!(c.runtime.filling.abs() < 1.0);
    }

    #[test]
    fn surfaces_fan_out_from_their_centre() {
        let poly = vec![Vec3::new(-1.0, 0.0, -1.0), Vec3::new(1.0, 0.0, -1.0), Vec3::new(1.0, 0.0, 1.0), Vec3::new(-1.0, 0.0, 1.0)];
        let (v, lo, hi) = surface_vertices(&[(poly, Vec3::Y, [0.2, 0.4, 0.5, 0.7])]);
        assert_eq!(v.len(), MAX_SURFACES * (SURFACE_RING + 1));
        assert_eq!(v[0].position, [0.0, 0.0, 0.0]);
        assert!(lo.x == -1.0 && hi.z == 1.0);
        assert_eq!(surface_mesh().indices.len(), MAX_SURFACES * SURFACE_RING * 3);
        let (d, lo, hi) = drop_vertices(&[PourDrop { pos: Vec3::ONE, vel: Vec3::NEG_Y * 3.0, size: 0.02, color: [0.2; 3] }], &[], Mat4::from_translation(Vec3::Z * 5.0));
        assert_eq!(d.len(), (MAX_DROPS + MAX_DRIPS) * 4);
        assert!(hi.y > lo.y && d[0].color[3] > 0.5);
    }

    #[test]
    fn swinging_sloshes_and_stopping_hard_throws_it_out() {
        let mut scene = LoadedScene::empty();
        let mut comps = bucket(0.0, 0.5);
        comps.remove(5);
        let id = obj(&mut scene, Vec3::new(0.0, 2.0, 0.0), Quat::IDENTITY, comps);
        let mut sim = CarrySim::default();
        let mut out = WaterStep::default();
        let i = scene.index_of(id).unwrap();
        // The Bucket's size: about 19 L, half full.
        scene.objects[i].scale = Vec3::new(0.28, 0.3, 0.28);
        let mut step = |scene: &mut LoadedScene, sim: &mut CarrySim| {
            sim.track(scene, FIXED_DT);
            let world = scene.world_transforms();
            sim.step(scene, &world, None, 0.0, FIXED_DT, &mut out);
        };
        for _ in 0..10 {
            step(&mut scene, &mut sim);
        }
        let before = container(&scene, id).runtime.litres;
        // Speeding up gently along +X to 3 m/s and carried on, then stopped dead.
        let mut speed = 0.0f32;
        for _ in 0..75 {
            speed = (speed + 4.0 * FIXED_DT).min(3.0);
            scene.objects[i].translation.x += speed * FIXED_DT;
            step(&mut scene, &mut sim);
        }
        assert!((container(&scene, id).runtime.litres - before).abs() < 0.05, "steady motion keeps it in");
        for _ in 0..30 {
            step(&mut scene, &mut sim);
        }
        let after = container(&scene, id).runtime.litres;
        assert!(after < before * 0.6, "the stop threw most of it out: {before} -> {after}");
        assert!(sim.drops.iter().any(|d| d.vel.x > 0.5), "forward, the way it was going");
    }
}
