//! Particles: CPU simulation of every active `ParticleEmitter`, drawn as
//! camera-facing quads (color + size over lifetime) through the built-in
//! unlit, alpha-blended material. Each emitter owns a host-visible mesh sized
//! for `max_particles` that is rewritten every frame; unused quads collapse to
//! zero size. Runs in the editor too, so effects preview while authoring.

use std::collections::HashMap;
use std::path::Path;

use zestengine_core::{ObjectId, ParticleEmitter};
use glam::{Mat4, Quat, Vec3};

use crate::scene::LoadedScene;

#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub pos: Vec3,
    pub vel: Vec3,
    pub age: f32,
    pub life: f32,
}

impl Particle {
    /// Normalized age 0..1.
    pub fn t(&self) -> f32 {
        (self.age / self.life.max(1e-4)).clamp(0.0, 1.0)
    }
}

/// Simulation state of one emitter.
#[derive(Default)]
pub struct EmitterSim {
    pub particles: Vec<Particle>,
    accum: f32,
    elapsed: f32,
    playing: bool,
    started: bool,
    rng: u64,
}

impl EmitterSim {
    fn rand(&mut self) -> f32 {
        // xorshift64*; deterministic per emitter.
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / (1u64 << 24) as f32
    }

    fn range(&mut self, r: [f32; 2]) -> f32 {
        r[0] + (r[1] - r[0]) * self.rand()
    }

    fn spawn(&mut self, cfg: &ParticleEmitter, world: Mat4) {
        if self.particles.len() >= cfg.max_particles as usize {
            return;
        }
        let (_, rot, origin) = world.to_scale_rotation_translation();
        let base = (rot * Vec3::from(cfg.direction)).normalize_or(Vec3::Y);
        // Random direction inside the cone around `base`.
        let cos_max = cfg.cone_angle.clamp(0.0, 180.0).to_radians().cos();
        let z = 1.0 - self.rand() * (1.0 - cos_max);
        let phi = self.rand() * std::f32::consts::TAU;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let local = Vec3::new(r * phi.cos(), z, r * phi.sin());
        let dir = Quat::from_rotation_arc(Vec3::Y, base) * local;
        let offset = loop {
            let p = Vec3::new(self.rand(), self.rand(), self.rand()) * 2.0 - Vec3::ONE;
            if p.length_squared() <= 1.0 {
                break p * cfg.shape_radius;
            }
        };
        let pos = if cfg.world_space { origin + offset } else { offset };
        let speed = self.range(cfg.speed);
        let life = self.range(cfg.lifetime).max(0.01);
        let vel = if cfg.world_space { dir * speed } else { rot.inverse() * dir * speed };
        self.particles.push(Particle { pos, vel, age: 0.0, life });
    }

    /// Advance by `dt` with the emitter at `world`.
    pub fn update(&mut self, cfg: &mut ParticleEmitter, world: Mat4, dt: f32, seed: u64) {
        if self.rng == 0 {
            self.rng = seed | 1;
        }
        let c = &mut cfg.control;
        if !self.started {
            self.started = true;
            self.playing = cfg.play_on_start;
            if self.playing {
                c.emit += cfg.burst;
            }
        }
        if c.clear {
            self.particles.clear();
        }
        if c.play {
            self.playing = true;
            self.elapsed = 0.0;
            c.emit += cfg.burst;
        }
        if c.stop {
            self.playing = false;
        }
        let burst = std::mem::take(&mut c.emit);
        c.play = false;
        c.stop = false;
        c.clear = false;

        if self.playing && !cfg.looping && self.elapsed >= cfg.duration {
            self.playing = false;
        }
        self.elapsed += dt;
        for _ in 0..burst {
            self.spawn(cfg, world);
        }
        if self.playing {
            self.accum += cfg.rate.max(0.0) * dt;
            while self.accum >= 1.0 {
                self.accum -= 1.0;
                self.spawn(cfg, world);
            }
        }
        let gravity = Vec3::new(0.0, -9.81, 0.0) * cfg.gravity;
        let drag = (1.0 - cfg.drag.max(0.0) * dt).max(0.0);
        for p in &mut self.particles {
            p.age += dt;
            p.vel = p.vel * drag + gravity * dt;
            p.pos += p.vel * dt;
        }
        self.particles.retain(|p| p.age < p.life);
        cfg.control.alive = self.particles.len() as u32;
        cfg.control.playing = self.playing;
    }
}

struct EmitterGpu {
    mesh: zestengine_render::MeshHandle,
    capacity: usize,
    material: zestengine_render::MaterialHandle,
    texture: String,
}

/// Simulation + GPU buffers for the scene's emitters.
#[derive(Default)]
pub struct ParticleRenderer {
    sims: HashMap<ObjectId, EmitterSim>,
    gpu: HashMap<ObjectId, EmitterGpu>,
    dot: Option<zestengine_render::TextureHandle>,
    textures: HashMap<String, Option<zestengine_render::TextureHandle>>,
}

impl ParticleRenderer {
    /// Drop all simulation state (Play start / stop, scene switch).
    pub fn reset(&mut self) {
        self.sims.clear();
    }

    /// Simulate every active emitter by `dt` and queue its draw on
    /// `scene.extra_draws`. `cam` is the viewing camera's world transform.
    pub fn update(&mut self, renderer: &mut zestengine_render::Renderer, scene: &mut LoadedScene, root: &Path, cam: Mat4, dt: f32) {
        let active = scene.visible_flags();
        let world = scene.world_transforms();
        let cam_pos = cam.w_axis.truncate();
        let right = cam.x_axis.truncate().normalize_or(Vec3::X);
        let up = cam.y_axis.truncate().normalize_or(Vec3::Y);
        let mut alive = Vec::new();
        for i in 0..scene.objects.len() {
            if !active[i] {
                continue;
            }
            let id = scene.objects[i].id;
            let Some(cfg) = scene.objects[i]
                .components
                .iter_mut()
                .find_map(|c| c.as_any_mut().downcast_mut::<ParticleEmitter>())
            else {
                continue;
            };
            alive.push(id);
            let capacity = cfg.max_particles.clamp(1, 20_000) as usize;
            let rebuild = self
                .gpu
                .get(&id)
                .is_none_or(|g| g.capacity != capacity || g.texture != cfg.texture);
            if rebuild {
                let tex = if cfg.texture.is_empty() {
                    self.dot_texture(renderer)
                } else {
                    let path = cfg.texture.clone();
                    *self.textures.entry(path.clone()).or_insert_with(|| {
                        zestengine_assets::load_texture_file(zestengine_core::asset_ref::resolve_path(root, &path), true)
                            .and_then(|t| renderer.upload_texture(&t))
                            .map_err(|e| tracing::warn!("particle texture {path}: {e:#}"))
                            .ok()
                    })
                };
                let mesh = renderer.upload_mesh_skinned(&quads(capacity));
                let material = crate::render_util::unlit_material(renderer, tex, "particles");
                match (mesh, material) {
                    (Ok(mesh), Ok(material)) => {
                        self.gpu.insert(
                            id,
                            EmitterGpu { mesh, capacity, material, texture: cfg.texture.clone() },
                        );
                    }
                    (Err(e), _) | (_, Err(e)) => {
                        tracing::error!("particle buffers: {e:#}");
                        continue;
                    }
                }
            }
            let sim = self.sims.entry(id).or_default();
            sim.update(cfg, world[i], dt.min(0.1), id.raw() as u64);
            let gpu = &self.gpu[&id];
            let to_world = if cfg.world_space { Mat4::IDENTITY } else { world[i] };
            // Back-to-front for alpha blending.
            let mut order: Vec<(f32, Vec3, f32)> = sim
                .particles
                .iter()
                .take(capacity)
                .map(|p| {
                    let wp = to_world.transform_point3(p.pos);
                    (wp.distance_squared(cam_pos), wp, p.t())
                })
                .collect();
            order.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut verts = Vec::with_capacity(capacity * 4);
            let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for &(_, p, t) in &order {
                let size = cfg.size[0] + (cfg.size[1] - cfg.size[0]) * t;
                let color = lerp4(cfg.color_start, cfg.color_end, t);
                let (r, u) = (right * size * 0.5, up * size * 0.5);
                for (corner, uv) in [(-r + u, [0.0, 0.0]), (r + u, [1.0, 0.0]), (r - u, [1.0, 1.0]), (-r - u, [0.0, 1.0])] {
                    verts.push(vertex(p + corner, uv, color));
                }
                lo = lo.min(p - Vec3::splat(size));
                hi = hi.max(p + Vec3::splat(size));
            }
            verts.resize(capacity * 4, vertex(Vec3::ZERO, [0.0, 0.0], [0.0; 4]));
            renderer.update_mesh_vertices(gpu.mesh, &verts);
            if order.is_empty() {
                continue;
            }
            scene.extra_draws.push(zestengine_render::DrawCmd {
                gi_source: None,
                mesh: gpu.mesh,
                material: gpu.material,
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
            });
        }
        self.sims.retain(|id, _| alive.contains(id));
        self.gpu.retain(|id, _| alive.contains(id));
    }

    fn dot_texture(&mut self, renderer: &mut zestengine_render::Renderer) -> Option<zestengine_render::TextureHandle> {
        if self.dot.is_none() {
            const N: u32 = 64;
            let mut pixels = Vec::with_capacity((N * N * 4) as usize);
            for y in 0..N {
                for x in 0..N {
                    let d = ((x as f32 + 0.5) / N as f32 * 2.0 - 1.0).hypot((y as f32 + 0.5) / N as f32 * 2.0 - 1.0);
                    let a = (1.0 - d).clamp(0.0, 1.0).powf(1.5);
                    pixels.extend_from_slice(&[255, 255, 255, (a * 255.0) as u8]);
                }
            }
            let data = zestengine_render::TextureData { width: N, height: N, pixels, srgb: true, hdr: false };
            self.dot = renderer.upload_texture(&data).ok();
        }
        self.dot
    }
}

fn lerp4(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [0, 1, 2, 3].map(|k| a[k] + (b[k] - a[k]) * t)
}

pub(crate) fn vertex(p: Vec3, uv: [f32; 2], color: [f32; 4]) -> zestengine_render::Vertex {
    zestengine_render::Vertex {
        position: p.to_array(),
        normal: [0.0, 0.0, 1.0],
        uv,
        color,
        tangent: [1.0, 0.0, 0.0, 1.0],
        uv1: uv,
        joints: [0; 4],
        weights: [0.0; 4],
    }
}

/// `n` quads worth of (degenerate) vertices + indices.
pub(crate) fn quads(n: usize) -> zestengine_render::MeshData {
    let mut indices = Vec::with_capacity(n * 6);
    for q in 0..n as u32 {
        let b = q * 4;
        indices.extend_from_slice(&[b, b + 2, b + 1, b, b + 3, b + 2]);
    }
    zestengine_render::MeshData {
        vertices: vec![vertex(Vec3::ZERO, [0.0, 0.0], [0.0; 4]); n * 4],
        indices,
        has_lightmap_uv: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_burst_and_lifetime() {
        let mut cfg = ParticleEmitter {
            rate: 10.0,
            burst: 5,
            lifetime: [1.0, 1.0],
            ..Default::default()
        };
        let mut sim = EmitterSim::default();
        sim.update(&mut cfg, Mat4::IDENTITY, 0.5, 7);
        assert_eq!(sim.particles.len(), 5 + 5, "burst + half a second at 10/s");
        for _ in 0..20 {
            sim.update(&mut cfg, Mat4::IDENTITY, 0.1, 7);
        }
        // Steady state: ~rate * lifetime alive.
        assert!((9..=11).contains(&sim.particles.len()), "{}", sim.particles.len());
        assert_eq!(cfg.control.alive as usize, sim.particles.len());
    }

    #[test]
    fn stop_and_gameplay_burst() {
        let mut cfg = ParticleEmitter {
            rate: 100.0,
            lifetime: [0.2, 0.2],
            ..Default::default()
        };
        let mut sim = EmitterSim::default();
        sim.update(&mut cfg, Mat4::IDENTITY, 0.1, 3);
        cfg.control.stop = true;
        for _ in 0..5 {
            sim.update(&mut cfg, Mat4::IDENTITY, 0.1, 3);
        }
        assert!(sim.particles.is_empty() && !cfg.control.playing);
        cfg.control.emit = 30;
        sim.update(&mut cfg, Mat4::IDENTITY, 0.01, 3);
        assert_eq!(sim.particles.len(), 30);
    }

    #[test]
    fn cone_and_gravity() {
        let mut cfg = ParticleEmitter {
            rate: 0.0,
            burst: 200,
            cone_angle: 10.0,
            speed: [1.0, 1.0],
            gravity: 1.0,
            lifetime: [5.0, 5.0],
            ..Default::default()
        };
        let mut sim = EmitterSim::default();
        sim.update(&mut cfg, Mat4::IDENTITY, 0.0, 11);
        let cos10 = 10f32.to_radians().cos();
        assert!(sim.particles.iter().all(|p| p.vel.normalize().y >= cos10 - 1e-4));
        for _ in 0..30 {
            sim.update(&mut cfg, Mat4::IDENTITY, 0.1, 11);
        }
        assert!(sim.particles.iter().all(|p| p.vel.y < 0.0), "gravity pulled them down");
    }
}
