//! Water effects: one CPU particle pool for spray, mist and steam, drawn as a
//! single alpha-blended mesh. Ripples are the water's own height-field
//! simulation: landing water and splashes push it. Waterfalls pour a sheet of
//! falling streaks off their lip that bursts into spray and mist where it
//! lands; the churned water there foams on the surface itself (`foam_spots`,
//! drawn by the water shader). Splashes and wakes from physics feed the same
//! pool, warm pools steam. Runs in the editor too, so waterfalls preview while
//! authoring.

use std::collections::HashMap;

use zestengine_core::{Vortex, Water, WaterSplash, Waterfall, Whirlpool};
use glam::{Mat4, Vec2, Vec3};

use crate::particles::{quads, vertex};
use crate::scene::LoadedScene;

const MAX_PARTICLES: usize = 6000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FxKind {
    /// Falling waterfall streak; bursts where it lands.
    Sheet,
    /// Flying droplet, stretched along its motion.
    Spray,
    /// Soft rising cloud.
    Mist,
    /// Wisp rising off warm water, drifting with the wind.
    Steam,
}

#[derive(Clone, Copy, Debug)]
pub struct FxParticle {
    pub kind: FxKind,
    pub pos: Vec3,
    pub vel: Vec3,
    pub age: f32,
    pub life: f32,
    pub size: f32,
    pub color: [f32; 3],
    /// Height the particle dies at (sheet, spray).
    pub floor: f32,
    /// What a landing sheet streak bursts into: spray, mist.
    pub burst: [f32; 2],
}

impl FxParticle {
    fn t(&self) -> f32 {
        (self.age / self.life.max(1e-4)).clamp(0.0, 1.0)
    }
}

/// Wake fed from physics: water moving past something at the surface.
#[derive(Clone, Copy, Debug)]
pub struct FxWake {
    pub at: Vec3,
    pub flow: Vec3,
    pub radius: f32,
}

/// Churned water that foams on the surface: each waterfall's plunge (as
/// wide as its sheet) and each whirlpool, as (world centre, radius, amount).
pub fn foam_spots(scene: &LoadedScene, world: &[Mat4], t: f32) -> Vec<(Vec3, f32, f32)> {
    let ws = waters(scene, world);
    let refs: Vec<(Mat4, &Water)> = ws.iter().map(|(m, w)| (*m, w)).collect();
    let active = scene.active_flags();
    let mut out = Vec::new();
    for (i, o) in scene.objects.iter().enumerate() {
        if !active[i] {
            continue;
        }
        for c in &o.components {
            if let Some(f) = c.as_any().downcast_ref::<Waterfall>()
                && f.foam > 0.0
            {
                let at = f.plunge(world[i], |p| catch_height(scene, &refs, p, t));
                let volume = f.volume.clamp(0.0, 1.0);
                out.push((at, f.width(world[i]) * 0.5 + 0.6 + 1.2 * volume, f.foam * (0.4 + 0.6 * volume)));
            } else if let Some(w) = c.as_any().downcast_ref::<Whirlpool>()
                && w.foam > 0.0
            {
                out.push((world[i].w_axis.truncate(), w.radius * 0.7, w.foam * 0.5));
            }
        }
    }
    out
}

/// Height the water (or, failing that, the ground) catches falling water
/// at, below `p`.
pub fn catch_height(scene: &LoadedScene, waters: &[(Mat4, &Water)], p: Vec3, t: f32) -> Option<f32> {
    let water = waters.iter().filter_map(|(m, w)| w.height_at(*m, p, t).map(|(h, _)| h)).fold(None, |a: Option<f32>, h| Some(a.map_or(h, |a| a.max(h))));
    let ground = scene.ray_hit(p, Vec3::NEG_Y).map(|d| p.y - d);
    match (water, ground) {
        (Some(w), Some(g)) => Some(w.max(g)),
        (w, g) => w.or(g),
    }
}

fn waters(scene: &LoadedScene, world: &[Mat4]) -> Vec<(Mat4, Water)> {
    scene
        .objects
        .iter()
        .enumerate()
        .filter(|(i, _)| scene.is_active(*i))
        .filter_map(|(i, o)| o.components.iter().find_map(|c| c.as_any().downcast_ref::<Water>()).map(|w| (world[i], w.clone())))
        .collect()
}

/// Every swirl in the scene: Whirlpools and waterfall plunge pools.
pub fn scene_vortices(scene: &LoadedScene, world: &[Mat4], t: f32) -> Vec<Vortex> {
    let mut out = Vec::new();
    let mut ws = None;
    for (i, o) in scene.objects.iter().enumerate() {
        if !scene.is_active(i) {
            continue;
        }
        for c in &o.components {
            if let Some(w) = c.as_any().downcast_ref::<Whirlpool>() {
                out.push(w.vortex(world[i]));
            } else if let Some(f) = c.as_any().downcast_ref::<Waterfall>()
                && f.swirl.abs() > 1e-3
            {
                let ws = ws.get_or_insert_with(|| waters(scene, world));
                let refs: Vec<(Mat4, &Water)> = ws.iter().map(|(m, w)| (*m, w)).collect();
                let plunge = f.plunge(world[i], |p| catch_height(scene, &refs, p, t));
                out.push(f.vortex(world[i], plunge));
            }
        }
    }
    out
}

/// The particle pool and its simulation (renderer-free, so it's testable).
#[derive(Default)]
pub struct WaterFxSim {
    pub particles: Vec<FxParticle>,
    /// Where falling water hit the surface this step (world point, radius,
    /// push speed), for the water's height-field simulation.
    pub impacts: Vec<(Vec3, f32, f32)>,
    rng: u64,
    /// Fractional spawns carried between frames, per emitter key.
    carry: HashMap<u64, f32>,
}

impl WaterFxSim {
    fn rand(&mut self) -> f32 {
        if self.rng == 0 {
            self.rng = 0x9E3779B97F4A7C15;
        }
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32 / (1u64 << 24) as f32
    }

    fn sym(&mut self) -> f32 {
        self.rand() * 2.0 - 1.0
    }

    fn push(&mut self, p: FxParticle) {
        if self.particles.len() < MAX_PARTICLES {
            self.particles.push(p);
        }
    }

    /// Whole spawns due this frame for `rate` per second.
    fn due(&mut self, key: u64, rate: f32, dt: f32) -> usize {
        let c = self.carry.entry(key).or_insert(0.0);
        *c += rate.max(0.0) * dt;
        let n = c.floor();
        *c -= n;
        n as usize
    }

    /// Burst of spray, foam and some mist where something hit (the dent in
    /// the surface comes from the height-field simulation).
    /// Droplets thrown up by a fast entry. The splash itself (the cavity,
    /// crown and rings) is the water simulation's displacement; this only
    /// adds the few drops that leave the surface on a really hard hit.
    pub fn splash(&mut self, s: &WaterSplash, color: [f32; 3], amount: f32) {
        let speed = s.velocity.length();
        if speed < 4.0 {
            return;
        }
        let energy = ((speed - 4.0) * s.size).sqrt();
        let drops = ((energy * 6.0 * amount) as usize).min(24);
        for _ in 0..drops {
            let a = self.rand() * std::f32::consts::TAU;
            let out = Vec3::new(a.cos(), 0.0, a.sin());
            let up = 1.0 + self.rand() * 2.0;
            let vel = out * (0.5 + self.rand()) * energy.min(3.0) + Vec3::Y * up * energy.min(2.5);
            let pos = s.at + out * s.size * (0.6 + self.rand() * 0.4);
            let size = 0.015 + self.rand() * 0.025;
            let life = 1.2 + self.rand();
            self.push(FxParticle { kind: FxKind::Spray, pos, vel, age: 0.0, life, size, color, floor: s.at.y, burst: [1.0; 2] });
        }
    }

    fn mist(&mut self, at: Vec3, size: f32, color: [f32; 3]) {
        let vel = Vec3::new(self.sym() * 0.4, 0.3 + self.rand() * 0.5, self.sym() * 0.4);
        let life = 2.5 + self.rand() * 2.5;
        let size = size * (0.6 + self.rand() * 0.8);
        self.push(FxParticle { kind: FxKind::Mist, pos: at, vel, age: 0.0, life, size, color, floor: at.y, burst: [1.0; 2] });
    }

    /// Pour one waterfall for `dt`: new sheet streaks along the lip, and the
    /// streaks that reached `plunge_y` burst into spray and mist.
    pub fn pour(&mut self, key: u64, fall: &Waterfall, world: Mat4, plunge_y: f32, dt: f32) {
        let width = fall.width(world).max(0.1);
        let color = fall.color.map(|c| c * fall.brightness);
        let n = self.due(key, fall.volume.clamp(0.0, 1.0) * width * 140.0, dt);
        for _ in 0..n {
            let u = self.rand();
            let (start, v) = fall.emit(world, u);
            let lip = (Vec3::from(fall.lip[1]) - Vec3::from(fall.lip[0])).normalize_or(Vec3::X);
            let side = world.transform_vector3(lip).normalize_or(Vec3::X);
            let vel = v * (0.85 + self.rand() * 0.3) + side * (self.sym() * fall.spread) + Vec3::Y * (self.sym() * 0.1);
            let size = 0.06 + self.rand() * 0.08;
            // Age the first frame a little so the lip isn't a hard line.
            let age = self.rand() * dt;
            self.push(FxParticle { kind: FxKind::Sheet, pos: start + vel * age, vel, age, life: 30.0, size, color, floor: plunge_y, burst: [fall.splash, fall.mist] });
        }
    }

    /// Advance everything by `dt`; `wind` (m/s) carries the steam.
    pub fn step(&mut self, dt: f32, wind: Vec3) {
        let g = Vec3::new(0.0, -zestengine_core::GRAVITY, 0.0);
        let mut bursts: Vec<(Vec3, Vec3, [f32; 3], [f32; 2])> = Vec::new();
        for p in &mut self.particles {
            p.age += dt;
            match p.kind {
                FxKind::Sheet | FxKind::Spray => {
                    p.vel = p.vel * (1.0 - 0.15 * dt) + g * dt;
                    p.pos += p.vel * dt;
                    if p.kind == FxKind::Sheet && p.pos.y <= p.floor {
                        bursts.push((Vec3::new(p.pos.x, p.floor, p.pos.z), p.vel, p.color, p.burst));
                        p.age = p.life;
                    } else if p.kind == FxKind::Spray && p.vel.y < 0.0 && p.pos.y < p.floor {
                        p.age = p.life;
                    }
                }
                FxKind::Mist => {
                    p.vel *= 1.0 - 0.6 * dt;
                    p.vel.y += 0.15 * dt;
                    p.pos += p.vel * dt;
                    p.size *= 1.0 + 0.35 * dt;
                }
                FxKind::Steam => {
                    let drift = Vec3::new(wind.x, p.vel.y, wind.z);
                    p.vel = p.vel.lerp(drift, (dt * 0.8).min(1.0));
                    p.vel.y += 0.08 * dt;
                    p.pos += p.vel * dt;
                    p.size *= 1.0 + 0.5 * dt;
                }
            }
        }
        self.particles.retain(|p| p.age < p.life);
        // Each landed streak pushes the water down (the plunge churns the
        // surface simulation) and throws spray, sometimes mist.
        for (at, vel, color, [splash, mist]) in bursts {
            let impact = (-vel.y).max(0.0);
            self.impacts.push((at, 0.35, impact * 0.35));
            if self.rand() < 0.5 * splash {
                let a = self.rand() * std::f32::consts::TAU;
                let up = impact * (0.15 + self.rand() * 0.25);
                let v = Vec3::new(a.cos(), 0.0, a.sin()) * (0.5 + self.rand() * 1.5) + Vec3::Y * up + Vec3::new(vel.x, 0.0, vel.z) * 0.3;
                let size = 0.03 + self.rand() * 0.05;
                self.push(FxParticle { kind: FxKind::Spray, pos: at, vel: v, age: 0.0, life: 2.0, size, color, floor: at.y, burst: [1.0; 2] });
            }
            if self.rand() < 0.05 * mist {
                let s = 0.6 + impact * 0.08;
                self.mist(at + Vec3::Y * 0.3, s, color);
            }
        }
    }

    /// Steam off a warm pool for `dt`: wisps born just above random spots of
    /// its surface, as many as its area and `steam` call for.
    pub fn steam(&mut self, key: u64, water: &Water, world: Mat4, t: f32, dt: f32) {
        if water.steam <= 0.0 || self.particles.len() > MAX_PARTICLES * 2 / 3 {
            return;
        }
        let Some((lo, hi)) = water.extent() else { return };
        let size = (hi - lo).max(Vec2::splat(0.5));
        let n = self.due(key, (size.x * size.y * 1.5 * water.steam).min(240.0), dt);
        for _ in 0..n {
            let xz = lo + size * Vec2::new(self.rand(), self.rand());
            if !water.covers(xz) {
                continue;
            }
            let (h, _) = water.surface(xz, t);
            let pos = world.transform_point3(Vec3::new(xz.x, h + 0.05, xz.y));
            let vel = Vec3::new(self.sym() * 0.05, 0.12 + self.rand() * 0.15, self.sym() * 0.05);
            let life = 3.0 + self.rand() * 3.0;
            let size = 0.25 + self.rand() * 0.35;
            self.push(FxParticle { kind: FxKind::Steam, pos, vel, age: 0.0, life, size, color: [0.92, 0.94, 0.96], floor: pos.y, burst: [0.0; 2] });
        }
    }

    /// Spray off something moving fast through the water (the wake itself is
    /// the simulation's displacement).
    pub fn wake(&mut self, key: u64, w: &FxWake, color: [f32; 3], dt: f32) {
        let u = w.flow.length();
        let dir = w.flow / u.max(1e-4);
        let side = Vec3::new(-dir.z, 0.0, dir.x);
        if u > 3.0 {
            for _ in 0..self.due(key ^ 0xABCD, (u - 3.0) * w.radius * 12.0, dt) {
                let vel = -dir * u * 0.2 + side * self.sym() * u * 0.3 + Vec3::Y * (1.0 + self.rand() * u * 0.3);
                let pos = w.at - dir * w.radius;
                let size = 0.03 + self.rand() * 0.03;
                self.push(FxParticle { kind: FxKind::Spray, pos, vel, age: 0.0, life: 1.5, size, color, floor: w.at.y, burst: [1.0; 2] });
            }
        }
    }
}

struct FxGpu {
    mesh: zestengine_render::MeshHandle,
    material: zestengine_render::MaterialHandle,
}

/// Simulates and draws the scene's water effects.
#[derive(Default)]
pub struct WaterFx {
    pub sim: WaterFxSim,
    gpu: Option<FxGpu>,
    edit_time: f32,
    /// Each waterfall's solid falling sheet.
    sheets: HashMap<zestengine_core::ObjectId, SheetGpu>,
    /// Streaked colour/opacity and normal maps the sheets flow with.
    sheet_maps: Option<(zestengine_render::TextureHandle, zestengine_render::TextureHandle)>,
}

struct SheetGpu {
    mesh: zestengine_render::MeshHandle,
    material: zestengine_render::MaterialHandle,
    /// Look the material was made with (colour, brightness, flow speed).
    look: [f32; 5],
}

/// Rows down and columns across a waterfall sheet.
const SHEET_ROWS: usize = 40;
const SHEET_COLS: usize = 14;
/// Falling water draws after every water surface: neither writes depth, so a
/// surface drawn later paints over the sheet falling in front of it.
const SHEET_QUEUE: i32 = zestengine_render::RENDER_QUEUE_TRANSPARENT + 50;

impl WaterFx {
    pub fn reset(&mut self) {
        self.sim = WaterFxSim::default();
    }

    /// Pour waterfalls, take the splashes and wakes queued on the scene,
    /// step, and queue the draw. `cam` is the viewing camera's world
    /// transform; `play_time` the water clock in play (None = editor clock).
    pub fn update(&mut self, renderer: &mut zestengine_render::Renderer, scene: &mut LoadedScene, cam: Mat4, play_time: Option<f32>, dt: f32) {
        self.edit_time += dt.max(0.0);
        let t = play_time.unwrap_or(self.edit_time);
        let dt = dt.min(0.05);
        let world = scene.world_transforms();
        let ws = waters(scene, &world);
        let refs: Vec<(Mat4, &Water)> = ws.iter().map(|(m, w)| (*m, w)).collect();
        let splashes = std::mem::take(&mut scene.water_splashes);
        let wakes = std::mem::take(&mut scene.water_wakes);
        let has_falls = scene.objects.iter().any(|o| o.components.iter().any(|c| c.as_any().is::<Waterfall>()));
        // Last step's landing water pushes the surface simulation: queued
        // for Play's physics step, or stepped here in the editor so
        // waterfalls churn their pools while authoring.
        let pushes = self.surface_pushes(scene, &world);
        if play_time.is_some() {
            scene.water_fx_pushes.extend(pushes);
        } else if !pushes.is_empty() || ws.iter().any(|(_, w)| w.ripples.is_some()) {
            let step = crate::physics::WaterStep { fx_pushes: pushes, ..Default::default() };
            crate::physics::step_ripples(scene, &world, cam.w_axis.truncate(), &step, None, t, dt);
        }
        let has_steam = ws.iter().any(|(_, w)| w.steam > 0.0);
        if self.sim.particles.is_empty() && splashes.is_empty() && wakes.is_empty() && !has_falls && !has_steam {
            self.sheets.clear();
            return;
        }
        let tint = refs.first().map_or([0.85, 0.9, 0.95], |(_, w)| {
            let s = w.look.shallow_color;
            [0.55 + s[0] * 0.3, 0.6 + s[1] * 0.3, 0.62 + s[2] * 0.3]
        });
        // Thick fluids swallow impacts without throwing droplets.
        let splashiness = |p: Vec3| refs.iter().find(|(m, w)| w.height_at(*m, p + Vec3::Y * 2.0, t).is_some()).map_or(1.0, |(_, w)| w.splashiness());
        for s in &splashes {
            self.sim.splash(s, tint, splashiness(s.at));
        }
        for (k, w) in wakes.iter().enumerate() {
            self.sim.wake(0x5EED_0000 + k as u64, w, tint, dt);
        }
        let active = scene.active_flags();
        let mut falls = Vec::new();
        for (i, o) in scene.objects.iter().enumerate() {
            if !active[i] {
                continue;
            }
            for c in &o.components {
                if let Some(f) = c.as_any().downcast_ref::<Waterfall>() {
                    let plunge = f.plunge(world[i], |p| catch_height(scene, &refs, p, t));
                    let sp = splashiness(plunge);
                    let poured = Waterfall { splash: f.splash * sp, mist: f.mist * sp, ..f.clone() };
                    self.sim.pour(o.id.raw() as u64, &poured, world[i], plunge.y, dt);
                    falls.push((o.id, f.clone(), world[i], plunge.y));
                }
            }
        }
        let mut wind = Vec3::ZERO;
        for (i, o) in scene.objects.iter().enumerate() {
            if !active[i] {
                continue;
            }
            if let Some(w) = o.components.iter().find_map(|c| c.as_any().downcast_ref::<Water>()) {
                self.sim.steam(o.id.raw() as u64, w, world[i], t, dt);
                if w.steam > 0.0 {
                    let d = w.runtime.wind_dir * w.runtime.wind_speed * 0.3;
                    wind = Vec3::new(d.x, 0.0, d.y);
                }
            }
        }
        self.sim.step(dt, wind);
        self.draw(renderer, scene, cam);
        self.draw_sheets(renderer, scene, &falls);
    }

    /// The falling water itself: one double-sided sheet per waterfall along
    /// its fall arc, flowing with scrolling streaks (the sheet particles only
    /// drive the landing splash and the pool's displacement).
    fn draw_sheets(&mut self, renderer: &mut zestengine_render::Renderer, scene: &mut LoadedScene, falls: &[(zestengine_core::ObjectId, Waterfall, Mat4, f32)]) {
        self.sheets.retain(|id, _| falls.iter().any(|(f, ..)| f == id));
        if falls.is_empty() {
            return;
        }
        if self.sheet_maps.is_none() {
            let (albedo, normal) = sheet_textures();
            match (renderer.upload_texture(&albedo), renderer.upload_texture(&normal)) {
                (Ok(a), Ok(n)) => self.sheet_maps = Some((a, n)),
                (Err(e), _) | (_, Err(e)) => {
                    tracing::error!("waterfall textures: {e:#}");
                    return;
                }
            }
        }
        let Some((albedo, normal)) = self.sheet_maps else { return };
        for (id, fall, world, plunge_y) in falls {
            let (verts, lo, hi) = sheet_vertices(fall, *world, *plunge_y);
            let speed = fall.flow_speed.max(0.5) + 3.0;
            let look = [fall.color[0], fall.color[1], fall.color[2], fall.brightness, speed];
            let stale = self.sheets.get(id).is_some_and(|g| g.look != look);
            if stale {
                self.sheets.remove(id);
            }
            if !self.sheets.contains_key(id) {
                let mesh = renderer.upload_mesh_skinned(&sheet_mesh(verts.clone()));
                let material = sheet_material(renderer, fall, albedo, normal, speed);
                match (mesh, material) {
                    (Ok(mesh), Ok(material)) => {
                        self.sheets.insert(*id, SheetGpu { mesh, material, look });
                    }
                    (Err(e), _) | (_, Err(e)) => {
                        tracing::error!("waterfall sheet: {e:#}");
                        continue;
                    }
                }
            }
            let g = &self.sheets[id];
            renderer.update_mesh_vertices(g.mesh, &verts);
            scene.extra_draws.push(zestengine_render::DrawCmd {
                gi_source: None,
                mesh: g.mesh,
                material: g.material,
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
    }

    /// This step's impacts as pushes on the water bodies they landed in.
    fn surface_pushes(&mut self, scene: &LoadedScene, world: &[Mat4]) -> Vec<(usize, zestengine_core::WaterPush)> {
        let impacts = std::mem::take(&mut self.sim.impacts);
        if impacts.is_empty() {
            return Vec::new();
        }
        let bodies: Vec<(usize, Mat4, &Water)> = scene
            .objects
            .iter()
            .enumerate()
            .filter(|(i, _)| scene.is_active(*i))
            .filter_map(|(i, o)| o.components.iter().find_map(|c| c.as_any().downcast_ref::<Water>()).filter(|w| w.interactive).map(|w| (i, world[i], w)))
            .collect();
        let mut out = Vec::new();
        for (at, radius, strength) in impacts {
            for (i, m, w) in &bodies {
                let l = m.inverse().transform_point3(at);
                if w.covers(glam::Vec2::new(l.x, l.z)) {
                    out.push((*i, zestengine_core::WaterPush { at: glam::Vec2::new(l.x, l.z), radius, strength }));
                    break;
                }
            }
        }
        out
    }

    fn draw(&mut self, renderer: &mut zestengine_render::Renderer, scene: &mut LoadedScene, cam: Mat4) {
        if self.gpu.is_none() {
            let tex = renderer.upload_texture(&atlas()).ok();
            let mesh = renderer.upload_mesh_skinned(&quads(MAX_PARTICLES));
            let material = crate::render_util::unlit_material(renderer, tex, "water_fx");
            match (mesh, material) {
                (Ok(mesh), Ok(material)) => {
                    // After the water and the sheets (see `SHEET_QUEUE`).
                    renderer.set_material_render_queue(material, SHEET_QUEUE + 5);
                    self.gpu = Some(FxGpu { mesh, material });
                }
                (Err(e), _) | (_, Err(e)) => {
                    tracing::error!("water fx buffers: {e:#}");
                    return;
                }
            }
        }
        let Some(gpu) = &self.gpu else { return };
        let verts = build_vertices(&self.sim.particles, cam);
        renderer.update_mesh_vertices(gpu.mesh, &verts.0);
        if self.sim.particles.is_empty() {
            return;
        }
        let (lo, hi) = verts.1;
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
}

/// Quads for every particle, back to front, padded to the pool size, and
/// their bounds.
fn build_vertices(particles: &[FxParticle], cam: Mat4) -> (Vec<zestengine_render::Vertex>, (Vec3, Vec3)) {
    let cam_pos = cam.w_axis.truncate();
    let right = cam.x_axis.truncate().normalize_or(Vec3::X);
    let up = cam.y_axis.truncate().normalize_or(Vec3::Y);
    let mut order: Vec<(f32, usize)> = particles.iter().enumerate().map(|(i, p)| (p.pos.distance_squared(cam_pos), i)).collect();
    order.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut verts = Vec::with_capacity(MAX_PARTICLES * 4);
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    // Atlas: the soft dot in the left half.
    let dot = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]];
    for &(_, i) in order.iter().take(MAX_PARTICLES) {
        let p = &particles[i];
        // Sheet streaks only drive the landing; the sheet mesh is the water.
        if p.age < 0.0 || p.kind == FxKind::Sheet {
            continue;
        }
        let t = p.t();
        let fade_in = (p.age / 0.15).min(1.0);
        let (a, b, uv, alpha) = match p.kind {
            FxKind::Sheet | FxKind::Spray => {
                // Stretched along the motion, facing the camera.
                let v = p.vel;
                let along = v.normalize_or(Vec3::Y);
                let side = along.cross(p.pos - cam_pos).normalize_or(right);
                let stretch = (v.length() * 0.035).clamp(1.0, 6.0) * p.size;
                let alpha = if p.kind == FxKind::Sheet { 0.35 } else { 0.7 * (1.0 - t) };
                (along * stretch, side * p.size * 0.5, dot, alpha)
            }
            FxKind::Mist => (up * p.size * 0.5, right * p.size * 0.5, dot, 0.07 * (1.0 - t) * fade_in),
            FxKind::Steam => {
                let fade_in = (p.age / 0.8).min(1.0);
                (up * p.size * 0.5, right * p.size * 0.5, dot, 0.07 * (1.0 - t) * (1.0 - t) * fade_in)
            }
        };
        let color = [p.color[0], p.color[1], p.color[2], alpha];
        let corners = [-b + a, b + a, b - a, -b - a];
        for (c, uv) in corners.into_iter().zip(uv) {
            verts.push(vertex(p.pos + c, uv, color));
        }
        let ext = a.abs() + b.abs();
        lo = lo.min(p.pos - ext);
        hi = hi.max(p.pos + ext);
    }
    verts.resize(MAX_PARTICLES * 4, vertex(Vec3::ZERO, [0.0, 0.0], [0.0; 4]));
    (verts, (lo, hi))
}

/// The sheet's vertices (world space) along the fall arc, and their bounds.
/// It widens with the spread as it falls and thins out over the last stretch
/// and at its sides.
fn sheet_vertices(fall: &Waterfall, world: Mat4, plunge_y: f32) -> (Vec<zestengine_render::Vertex>, Vec3, Vec3) {
    let lip = (Vec3::from(fall.lip[1]) - Vec3::from(fall.lip[0])).normalize_or(Vec3::X);
    let side = world.transform_vector3(lip).normalize_or(Vec3::X);
    let width = fall.width(world).max(0.05);
    let (start_mid, _) = fall.emit(world, 0.5);
    let t_end = zestengine_core::fall_time((start_mid.y - plunge_y).max(0.05));
    let at = |u: f32, t: f32| {
        let (start, v) = fall.emit(world, u);
        start + v * t - Vec3::Y * (0.5 * zestengine_core::GRAVITY * t * t) + side * ((u - 0.5) * 2.0 * fall.spread * t)
    };
    let mut verts = Vec::with_capacity((SHEET_ROWS + 1) * (SHEET_COLS + 1));
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    let mut dist = [0.0f32; SHEET_COLS + 1];
    let mut prev: Vec<Vec3> = Vec::new();
    for r in 0..=SHEET_ROWS {
        let s = r as f32 / SHEET_ROWS as f32;
        let t = t_end * s;
        let row: Vec<Vec3> = (0..=SHEET_COLS).map(|c| at(c as f32 / SHEET_COLS as f32, t)).collect();
        for c in 0..=SHEET_COLS {
            if let Some(p) = prev.get(c) {
                dist[c] += p.distance(row[c]);
            }
            let u = c as f32 / SHEET_COLS as f32;
            let p = row[c];
            let down = (at(u, t + 0.01) - p).normalize_or(-Vec3::Y);
            let n = down.cross(side).normalize_or(world.transform_vector3(Vec3::Z).normalize_or(Vec3::Z));
            // Solid near the lip, breaking up toward the plunge and the sides.
            let edge = (u.min(1.0 - u) / 0.12).clamp(0.0, 1.0);
            let tail = 1.0 - ((s - 0.7) / 0.3).clamp(0.0, 1.0);
            let alpha = (0.25 + 0.75 * edge) * (0.2 + 0.8 * tail);
            let uv = [u * width / 1.5, dist[c] / 1.5];
            verts.push(zestengine_render::Vertex {
                position: p.to_array(),
                normal: n.to_array(),
                uv,
                color: [1.0, 1.0, 1.0, alpha],
                tangent: [side.x, side.y, side.z, 1.0],
                uv1: uv,
                joints: [0; 4],
                weights: [0.0; 4],
            });
            lo = lo.min(p);
            hi = hi.max(p);
        }
        prev = row;
    }
    (verts, lo, hi)
}

/// Index buffer for the sheet grid.
fn sheet_mesh(vertices: Vec<zestengine_render::Vertex>) -> zestengine_render::MeshData {
    let cols = SHEET_COLS as u32 + 1;
    let mut indices = Vec::with_capacity(SHEET_ROWS * SHEET_COLS * 6);
    for r in 0..SHEET_ROWS as u32 {
        for c in 0..SHEET_COLS as u32 {
            let a = r * cols + c;
            let (b, d, e) = (a + 1, a + cols, a + cols + 1);
            indices.extend_from_slice(&[a, d, b, b, d, e]);
        }
    }
    zestengine_render::MeshData { vertices, indices, has_lightmap_uv: false }
}

/// Lit, alpha-blended, double-sided water with its streaks scrolling down
/// the sheet at about the water's speed.
fn sheet_material(
    renderer: &mut zestengine_render::Renderer,
    fall: &Waterfall,
    albedo: zestengine_render::TextureHandle,
    normal: zestengine_render::TextureHandle,
    speed: f32,
) -> anyhow::Result<zestengine_render::MaterialHandle> {
    let features = zestengine_render::MaterialFeatures { alpha_mode: zestengine_render::AlphaMode::Blend, double_sided: true, ..Default::default() };
    let c = fall.color.map(|v| v * fall.brightness);
    let params = zestengine_render::MaterialParams {
        base_color: [c[0], c[1], c[2], 0.85],
        roughness: 0.12,
        metallic: 0.0,
        normal_strength: 1.0,
        base_scroll: [0.0, -speed / 1.5],
        ..Default::default()
    };
    let desc = zestengine_render::MaterialDesc { name: "waterfall".into(), params, features, albedo: Some(albedo), normal: Some(normal), ..Default::default() };
    let material = renderer.create_material(&desc)?;
    renderer.set_material_features(material, features);
    renderer.set_material_render_queue(material, SHEET_QUEUE);
    Ok(material)
}

/// Streaks along the flow (v): opacity/brightness in the albedo, matching
/// ripples in the normal map. Tiles in both directions.
fn sheet_textures() -> (zestengine_render::TextureData, zestengine_render::TextureData) {
    const W: u32 = 128;
    const H: u32 = 256;
    // Periodic value noise along x, stretched along y.
    let hash = |i: i32, j: i32| -> f32 {
        let mut h = (i.rem_euclid(64) as u32).wrapping_mul(374761393) ^ (j.rem_euclid(16) as u32).wrapping_mul(668265263);
        h = (h ^ (h >> 13)).wrapping_mul(1274126177);
        (h ^ (h >> 16)) as f32 / u32::MAX as f32
    };
    let noise = |x: f32, y: f32| -> f32 {
        let (xi, yi) = (x.floor() as i32, y.floor() as i32);
        let (fx, fy) = (x - x.floor(), y - y.floor());
        let s = |v: f32| v * v * (3.0 - 2.0 * v);
        let a = hash(xi, yi) + (hash(xi + 1, yi) - hash(xi, yi)) * s(fx);
        let b = hash(xi, yi + 1) + (hash(xi + 1, yi + 1) - hash(xi, yi + 1)) * s(fx);
        a + (b - a) * s(fy)
    };
    let field = |x: u32, y: u32| -> f32 {
        let (u, v) = (x as f32 / W as f32, y as f32 / H as f32);
        0.6 * noise(u * 64.0, v * 4.0) + 0.3 * noise(u * 32.0 + 7.0, v * 8.0) + 0.1 * noise(u * 64.0 + 3.0, v * 16.0)
    };
    let mut albedo = Vec::with_capacity((W * H * 4) as usize);
    let mut normal = Vec::with_capacity((W * H * 4) as usize);
    for y in 0..H {
        for x in 0..W {
            let f = field(x, y);
            let streak = ((f - 0.35) * 2.2).clamp(0.0, 1.0);
            let shade = (205.0 + 50.0 * streak) as u8;
            albedo.extend_from_slice(&[shade, shade, 255, (150.0 + 105.0 * streak) as u8]);
            let dx = field((x + 1) % W, y) - field((x + W - 1) % W, y);
            let dy = field(x, (y + 1) % H) - field(x, (y + H - 1) % H);
            let n = Vec3::new(-dx * 6.0, -dy * 1.5, 1.0).normalize();
            normal.extend_from_slice(&[((n.x * 0.5 + 0.5) * 255.0) as u8, ((n.y * 0.5 + 0.5) * 255.0) as u8, ((n.z * 0.5 + 0.5) * 255.0) as u8, 255]);
        }
    }
    (
        zestengine_render::TextureData { width: W, height: H, pixels: albedo, srgb: true, hdr: false },
        zestengine_render::TextureData { width: W, height: H, pixels: normal, srgb: false, hdr: false },
    )
}

/// Soft dot (left) and soft ring (right), white with alpha.
fn atlas() -> zestengine_render::TextureData {
    const H: u32 = 64;
    let mut pixels = Vec::with_capacity((H * 2 * H * 4) as usize);
    for y in 0..H {
        for x in 0..H * 2 {
            let lx = (x % H) as f32 + 0.5;
            let d = Vec2::new(lx / H as f32 * 2.0 - 1.0, (y as f32 + 0.5) / H as f32 * 2.0 - 1.0).length();
            let a = if x < H {
                (1.0 - d).clamp(0.0, 1.0).powf(1.5)
            } else {
                (1.0 - ((d - 0.8).abs() / 0.15)).clamp(0.0, 1.0).powi(2)
            };
            pixels.extend_from_slice(&[255, 255, 255, (a * 255.0) as u8]);
        }
    }
    zestengine_render::TextureData { width: H * 2, height: H, pixels, srgb: true, hdr: false }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_waterfall_pours_lands_and_bursts() {
        let mut sim = WaterFxSim::default();
        let fall = Waterfall { volume: 1.0, ..Default::default() };
        let world = Mat4::from_translation(Vec3::new(0.0, 4.0, 0.0));
        for k in 0..240 {
            sim.pour(1, &fall, world, 0.0, 1.0 / 60.0);
            sim.step(1.0 / 60.0, Vec3::ZERO);
            if k == 10 {
                assert!(sim.particles.iter().all(|p| p.kind == FxKind::Sheet), "nothing has landed yet");
            }
        }
        let count = |k: FxKind| sim.particles.iter().filter(|p| p.kind == k).count();
        assert!(count(FxKind::Sheet) > 100, "a steady sheet: {}", count(FxKind::Sheet));
        assert!(count(FxKind::Spray) > 0 && count(FxKind::Mist) > 0);
        assert!(sim.impacts.len() > 100 && sim.impacts.iter().all(|(at, _, push)| at.y == 0.0 && *push > 1.0), "the landing water pushes the surface");
        assert!(sim.particles.iter().filter(|p| p.kind == FxKind::Sheet).all(|p| p.pos.y > -0.5));
    }

    #[test]
    fn only_hard_hits_throw_spray() {
        let mut sim = WaterFxSim::default();
        let at = Vec3::new(1.0, 0.0, 2.0);
        for speed in [0.0, 0.5, 2.5, 3.9] {
            sim.splash(&WaterSplash { at, velocity: Vec3::new(0.0, -speed, 0.0), size: 0.4 }, [1.0; 3], 1.0);
            sim.wake(7, &FxWake { at, flow: Vec3::new(speed.min(2.9), 0.0, 0.0), radius: 0.4 }, [1.0; 3], 1.0 / 90.0);
        }
        assert!(sim.particles.is_empty(), "slow or still bodies throw no droplets: {}", sim.particles.len());
        sim.splash(&WaterSplash { at, velocity: Vec3::new(0.0, -8.0, 0.0), size: 0.4 }, [1.0; 3], 1.0);
        assert!(!sim.particles.is_empty() && sim.particles.iter().all(|p| p.kind == FxKind::Spray && p.size < 0.05), "a hard hit throws a few small drops");
    }

    #[test]
    fn the_sheet_runs_from_the_lip_to_the_plunge_and_thins_out() {
        let fall = Waterfall::default();
        let world = Mat4::from_translation(Vec3::new(0.0, 4.0, 0.0));
        let (v, lo, hi) = sheet_vertices(&fall, world, 0.0);
        assert_eq!(v.len(), (SHEET_ROWS + 1) * (SHEET_COLS + 1));
        let top = &v[SHEET_COLS / 2];
        let bottom = &v[SHEET_ROWS * (SHEET_COLS + 1) + SHEET_COLS / 2];
        assert!((top.position[1] - 4.0).abs() < 1e-4 && bottom.position[1].abs() < 1e-3, "{:?} {:?}", top.position, bottom.position);
        assert!(bottom.position[2] > top.position[2], "thrown out along +Z");
        assert!(top.color[3] > 0.9 && bottom.color[3] < 0.3, "solid at the lip, breaking up at the plunge");
        assert!(lo.y < 0.01 && hi.y > 3.99);
        assert_eq!(sheet_mesh(v).indices.len(), SHEET_ROWS * SHEET_COLS * 6);
    }

    #[test]
    fn a_warm_pool_steams_and_the_wind_carries_it() {
        let mut sim = WaterFxSim::default();
        let mut pool = Water { size: [4.0, 4.0], steam: 1.0, ..Default::default() };
        pool.make_pool();
        let wind = Vec3::new(1.0, 0.0, 0.0);
        for _ in 0..120 {
            sim.steam(1, &pool, Mat4::IDENTITY, 0.0, 1.0 / 60.0);
            sim.step(1.0 / 60.0, wind);
        }
        assert!(sim.particles.len() > 10 && sim.particles.iter().all(|p| p.kind == FxKind::Steam && p.pos.y > 0.0));
        assert!(sim.particles.iter().map(|p| p.vel.x).sum::<f32>() > 0.0, "drifts downwind");
        let cold = Water { steam: 0.0, ..pool };
        let mut none = WaterFxSim::default();
        none.steam(1, &cold, Mat4::IDENTITY, 0.0, 1.0);
        assert!(none.particles.is_empty());
    }

    #[test]
    fn a_splash_throws_spray_that_falls_back() {
        let mut sim = WaterFxSim::default();
        sim.splash(&WaterSplash { at: Vec3::ZERO, velocity: Vec3::new(0.0, -6.0, 0.0), size: 0.5 }, [1.0; 3], 1.0);
        assert!(sim.particles.iter().any(|p| p.kind == FxKind::Spray && p.vel.y > 0.0));
        for _ in 0..600 {
            sim.step(1.0 / 60.0, Vec3::ZERO);
        }
        assert!(sim.particles.iter().all(|p| p.kind != FxKind::Spray), "spray is gone after landing");
    }
}
