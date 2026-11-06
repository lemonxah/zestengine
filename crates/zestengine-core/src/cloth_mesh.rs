//! Mesh cloth: simulate a skinned garment from its own triangles. Render
//! vertices are welded (UV seams) and optionally clustered into a coarser
//! particle set; triangle edges become stretch constraints, edges across
//! neighbouring triangles bending constraints, and free particles are
//! tethered to their nearest pinned one. Pinned particles follow the
//! animation; every render vertex is moved by its particles' offset from the
//! animated pose.

use std::collections::HashMap;

use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

use crate::cloth::{ClothShapeWorld, ClothTeleport, WindSample, penetration};
use crate::cloth_pins::{ClothPaint, WeldTopology};
use crate::{ObjectRef, TypedComponent};

/// Where a [`MeshCloth`] takes its pin weights from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PinSource {
    /// A vertex colour channel: 1 = pinned to the body, 0 = free.
    #[default]
    VertexColor,
    /// Vertices skinned mostly to the listed bones are pinned.
    Bones,
    /// The top `pin_height` share of the garment is pinned.
    Top,
    /// Weights painted in the editor (`MeshCloth::paint`).
    Painted,
}

impl PinSource {
    /// Every source, in declaration order.
    pub const ALL: [PinSource; 4] = [PinSource::VertexColor, PinSource::Bones, PinSource::Top, PinSource::Painted];
    /// Inspector label of this source.
    pub fn label(self) -> &'static str {
        match self {
            PinSource::VertexColor => "Vertex colour",
            PinSource::Bones => "Bones",
            PinSource::Top => "Top of the mesh",
            PinSource::Painted => "Painted",
        }
    }
}

/// Engine-filled state of a [`MeshCloth`], not saved.
#[derive(Default)]
pub struct MeshClothRuntime {
    /// The running simulation (not saved).
    pub sim: Option<MeshClothSim>,
    /// Render slot names on the object (for the slot picker).
    pub slots: Vec<String>,
    /// Bone names (for the pin bone picker).
    pub bones: Vec<String>,
    /// Cloth Collider names on the object.
    pub collider_names: Vec<String>,
    /// Set by the inspector; the engine rebuilds the simulation.
    pub reset_requested: bool,
    /// Bind-pose positions and triangles of the simulated submesh (object
    /// space), captured when the simulation is built.
    pub bind: Vec<Vec3>,
    /// Triangles of the simulated submesh (bind pose).
    pub indices: Vec<u32>,
    /// `topology_hash` of `bind` + `indices`.
    pub topology: u64,
    /// Pins the current (non-painted) source gives, to seed painting from.
    pub source_pins: Vec<f32>,
    /// The pin painter is open on this cloth (set by the editor each frame).
    pub paint_mode: bool,
    /// Hold the animated pose instead of simulating (painting without Simulate).
    pub hold: bool,
    /// The shown pose this frame (object space), kept while painting.
    pub posed: Vec<Vec3>,
    /// Painted values changed: push them into the running simulation;
    /// `retether` also rebuilds the tethers (end of a stroke).
    pub pins_dirty: bool,
    /// Also rebuild the tethers when the pins change.
    pub retether: bool,
    /// Welded topology for the brush, built on demand.
    pub weld: Option<WeldTopology>,
    /// Shown in the inspector (e.g. the paint was remapped to a new mesh).
    pub paint_note: String,
    /// Beyond the cull distance from the viewer this frame.
    pub culled: bool,
    /// Held or culled last step: restart from the animated pose.
    pub idle: bool,
}

impl MeshClothRuntime {
    /// The brush topology for the captured bind mesh.
    pub fn weld(&mut self) -> Option<&WeldTopology> {
        if self.bind.is_empty() {
            return None;
        }
        if self.weld.as_ref().is_none_or(|w| w.vertex_count() != self.bind.len()) {
            self.weld = Some(WeldTopology::build(&self.bind, &self.indices));
        }
        self.weld.as_ref()
    }
}

/// Simulates one skinned mesh of this object as cloth.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct MeshCloth {
    /// Render slot (submesh) of the object to simulate.
    pub slot: usize,
    pub pin: PinSource,
    /// Vertex colour channel for `PinSource::VertexColor` (0 R, 1 G, 2 B, 3 A).
    pub pin_channel: u32,
    /// Bones whose vertices are pinned (PinSource::Bones).
    pub pin_bones: Vec<String>,
    /// Share of the mesh height pinned for `PinSource::Top`.
    pub pin_height: f32,
    /// Merge vertices closer than this into one particle (m, 0 = weld seams only).
    pub particle_spacing: f32,
    /// Stretch resistance (0 to 1).
    pub stiffness: f32,
    /// Bending resistance (0 to 1).
    pub bending: f32,
    /// Pull toward the animated shape per 1/90 s (0 to 1).
    pub shape_stiffness: f32,
    /// How far (share of rest distance) a free point may stray from its
    /// nearest pinned point before the tether stops it.
    pub tether_slack: f32,
    /// Gravity (m/s^2).
    pub gravity: [f32; 3],
    /// Velocity lost per 1/60 s (0..1).
    pub damping: f32,
    /// Particle collision radius (m).
    pub radius: f32,
    /// Tangential velocity lost on contact (0..1).
    pub friction: f32,
    /// Particles collide with Cloth Colliders.
    pub collide: bool,
    /// Collider names on this object (empty = all).
    pub colliders: Vec<String>,
    /// Other objects whose Cloth Colliders also apply.
    pub collider_objects: Vec<ObjectRef>,
    /// Particles collide with each other.
    pub self_collision: bool,
    /// Minimum distance kept between non-neighbouring particles (m).
    pub thickness: f32,
    /// How much the cloth resists the object's own movement: 1 = fully
    /// physical, 0 = moves rigidly with the object.
    pub world_inertia: f32,
    /// Object speed past which the extra movement is carried rigidly (m/s,
    /// 0 = no limit).
    pub max_speed: f32,
    /// A jump farther than this in one frame is a teleport (m, 0 = never).
    pub teleport_distance: f32,
    /// A turn sharper than this in one frame is a teleport (degrees, 0 = never).
    pub teleport_angle: f32,
    /// What a teleport does to the cloth.
    pub teleport: ClothTeleport,
    /// How strongly Wind Zones push this cloth (0 = ignores wind).
    pub wind: f32,
    /// Simulation steps per second.
    pub frequency: f32,
    /// Constraint passes per step.
    pub iterations: u32,
    /// Painted pin weights and max distances (`PinSource::Painted`).
    pub paint: ClothPaint,
    /// How far (object units, about metres) a vertex whose painted max
    /// distance is 1 may stray from its animated position.
    pub max_distance: f32,
    /// Stop simulating (hold the animated pose) beyond this distance from the
    /// camera (m, 0 = never).
    pub cull_distance: f32,
    /// Engine-filled state (not saved).
    #[serde(skip)]
    pub runtime: MeshClothRuntime,
}

impl Default for MeshCloth {
    fn default() -> Self {
        Self {
            slot: 0,
            pin: PinSource::VertexColor,
            pin_channel: 0,
            pin_bones: Vec::new(),
            pin_height: 0.15,
            particle_spacing: 0.0,
            stiffness: 0.9,
            bending: 0.3,
            shape_stiffness: 0.02,
            tether_slack: 0.1,
            gravity: [0.0, -9.81, 0.0],
            damping: 0.05,
            radius: 0.01,
            friction: 0.3,
            collide: true,
            colliders: Vec::new(),
            collider_objects: Vec::new(),
            self_collision: false,
            thickness: 0.01,
            world_inertia: 1.0,
            max_speed: 4.0,
            teleport_distance: 1.0,
            teleport_angle: 90.0,
            teleport: ClothTeleport::Reset,
            wind: 1.0,
            frequency: 90.0,
            iterations: 4,
            paint: ClothPaint::default(),
            max_distance: 0.1,
            cull_distance: 0.0,
            runtime: MeshClothRuntime::default(),
        }
    }
}

impl TypedComponent for MeshCloth {
    const NAME: &'static str = "Mesh Cloth";
}

/// One simulated particle (a cluster of welded render vertices).
#[derive(Clone, Debug)]
pub struct MeshParticle {
    /// World position.
    pub pos: Vec3,
    /// Velocity (m/s).
    pub vel: Vec3,
    /// 1 = follows the animation, 0 = free.
    pub pin: f32,
    /// Farthest it may stray from its animated position (m, world).
    pub max_distance: f32,
    prev_anim: Vec3,
    contact: Option<Vec3>,
}

/// The running simulation of one [`MeshCloth`].
#[derive(Clone, Debug)]
pub struct MeshClothSim {
    /// The particles.
    pub particles: Vec<MeshParticle>,
    /// Stretch constraints (a, b) with rest lengths taken from the animated
    /// pose each step.
    pub edges: Vec<(u32, u32)>,
    bends: Vec<(u32, u32)>,
    /// (free particle, its nearest pinned particle).
    tethers: Vec<(u32, u32)>,
    /// Render vertex -> particle.
    owner: Vec<u32>,
    /// Particle -> its render vertices.
    members: Vec<Vec<u32>>,
    indices: Vec<u32>,
    vertex_count: usize,
    last_world: Mat4,
    anim: Vec<Vec3>,
    /// Particle rest positions (object space), for re-tethering.
    rest: Vec<Vec3>,
}

impl MeshClothSim {
    /// Render vertices in the simulated submesh.
    pub fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    /// Build from the mesh at rest (object space), its triangles and each
    /// render vertex's pin weight. `world` is the object's world matrix.
    pub fn build(positions: &[Vec3], indices: &[u32], pins: &[f32], spacing: f32, world: Mat4) -> Option<Self> {
        if positions.is_empty() || indices.len() < 3 {
            return None;
        }
        // Weld (UV seams), then cluster to the particle spacing.
        let cell = if spacing > 0.0 { spacing } else { 1e-4 };
        let key = |p: Vec3| ((p.x / cell).round() as i64, (p.y / cell).round() as i64, (p.z / cell).round() as i64);
        let mut by_cell: HashMap<(i64, i64, i64), u32> = HashMap::new();
        let mut owner = Vec::with_capacity(positions.len());
        let mut members: Vec<Vec<u32>> = Vec::new();
        for (v, p) in positions.iter().enumerate() {
            let next = members.len() as u32;
            let id = *by_cell.entry(key(*p)).or_insert(next);
            if id == next {
                members.push(Vec::new());
            }
            members[id as usize].push(v as u32);
            owner.push(id);
        }
        let mut edge_set: std::collections::HashSet<(u32, u32)> = Default::default();
        let mut tri_of_edge: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
        for tri in indices.as_chunks::<3>().0 {
            let t = [owner[tri[0] as usize], owner[tri[1] as usize], owner[tri[2] as usize]];
            if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
                continue;
            }
            for k in 0..3 {
                let (a, b) = (t[k].min(t[(k + 1) % 3]), t[k].max(t[(k + 1) % 3]));
                edge_set.insert((a, b));
                tri_of_edge.entry((a, b)).or_default().push(t[(k + 2) % 3]);
            }
        }
        let edges: Vec<(u32, u32)> = edge_set.into_iter().collect();
        let mut bends = Vec::new();
        for opposite in tri_of_edge.values() {
            if let [a, b, ..] = opposite.as_slice()
                && a != b
            {
                bends.push((*a.min(b), *a.max(b)));
            }
        }
        bends.sort_unstable();
        bends.dedup();
        let pin_of = |m: &Vec<u32>| m.iter().map(|&v| pins.get(v as usize).copied().unwrap_or(0.0)).fold(0.0f32, f32::max);
        let rest = |m: &Vec<u32>| m.iter().map(|&v| positions[v as usize]).sum::<Vec3>() / m.len().max(1) as f32;
        let rest_pos: Vec<Vec3> = members.iter().map(rest).collect();
        let particle_pins: Vec<f32> = members.iter().map(pin_of).collect();
        let tethers = tethers(&rest_pos, &particle_pins);
        let particles = members
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let w = world.transform_point3(rest_pos[i]);
                MeshParticle { pos: w, vel: Vec3::ZERO, pin: pin_of(m), max_distance: f32::INFINITY, prev_anim: w, contact: None }
            })
            .collect();
        Some(Self {
            particles,
            edges,
            bends,
            tethers,
            owner,
            members,
            indices: indices.to_vec(),
            vertex_count: positions.len(),
            last_world: world,
            anim: Vec::new(),
            rest: rest_pos,
        })
    }

    /// Take new per-render-vertex pin weights (a particle takes its
    /// strongest vertex) without rebuilding; `retether` also re-links free
    /// particles to their nearest pinned one.
    pub fn set_pins(&mut self, pins: &[f32], retether: bool) {
        if pins.len() != self.vertex_count {
            return;
        }
        for (p, m) in self.particles.iter_mut().zip(&self.members) {
            p.pin = m.iter().map(|&v| pins[v as usize]).fold(0.0f32, f32::max).clamp(0.0, 1.0);
        }
        if retether {
            let pins: Vec<f32> = self.particles.iter().map(|p| p.pin).collect();
            self.tethers = tethers(&self.rest, &pins);
        }
    }

    /// Per-render-vertex distance limits (object units; a particle takes its
    /// tightest vertex), or None to lift them.
    pub fn set_limits(&mut self, limits: Option<&[f32]>) {
        let scale = crate::cloth::max_scale(self.last_world);
        for (p, m) in self.particles.iter_mut().zip(&self.members) {
            p.max_distance = match limits {
                Some(l) if l.len() == self.vertex_count => m.iter().map(|&v| l[v as usize]).fold(f32::INFINITY, f32::min).max(0.0) * scale,
                _ => f32::INFINITY,
            };
        }
    }

    /// Animated particle positions (world) from the skinned render vertices.
    fn animated(&self, skinned: &[Vec3], world: Mat4) -> Vec<Vec3> {
        self.members
            .iter()
            .map(|m| world.transform_point3(m.iter().map(|&v| skinned[v as usize]).sum::<Vec3>() / m.len().max(1) as f32))
            .collect()
    }

    /// Snap every particle back to the animated pose (e.g. after a teleport).
    pub fn reset(&mut self, skinned: &[Vec3], world: Mat4) {
        let anim = self.animated(skinned, world);
        for (p, a) in self.particles.iter_mut().zip(&anim) {
            p.pos = *a;
            p.prev_anim = *a;
            p.vel = Vec3::ZERO;
        }
        self.last_world = world;
    }

    /// Advance by `dt`. `skinned` is this frame's skinned render vertex
    /// positions (object space).
    #[allow(clippy::too_many_arguments)]
    pub fn step(&mut self, cloth: &MeshCloth, skinned: &[Vec3], world: Mat4, colliders: &[ClothShapeWorld], winds: &[WindSample], time: f32, dt: f32) {
        if dt <= 0.0 || skinned.len() != self.vertex_count {
            return;
        }
        let dt = dt.min(0.1);
        let anim = self.animated(skinned, world);
        let rot = |m: &Mat4| m.to_scale_rotation_translation().1;
        let moved = world.w_axis.truncate() - self.last_world.w_axis.truncate();
        let turned = rot(&self.last_world).angle_between(rot(&world)).to_degrees();
        let last_inv = self.last_world.inverse();
        if (cloth.teleport_distance > 0.0 && moved.length() > cloth.teleport_distance)
            || (cloth.teleport_angle > 0.0 && turned > cloth.teleport_angle)
        {
            match cloth.teleport {
                ClothTeleport::Reset => {
                    self.reset(skinned, world);
                    return;
                }
                ClothTeleport::Keep => {
                    let delta = world * last_inv;
                    for p in &mut self.particles {
                        p.pos = delta.transform_point3(p.pos);
                        p.prev_anim = delta.transform_point3(p.prev_anim);
                    }
                }
            }
        } else {
            let mut follow = (1.0 - cloth.world_inertia).clamp(0.0, 1.0);
            let speed = moved.length() / dt;
            if cloth.max_speed > 0.0 && speed > cloth.max_speed {
                follow = follow.max(1.0 - cloth.max_speed / speed);
            }
            if follow > 0.0 {
                for p in self.particles.iter_mut() {
                    let carried = world.transform_point3(last_inv.transform_point3(p.pos));
                    p.pos = p.pos.lerp(carried, follow);
                }
            }
        }
        self.last_world = world;

        let steps = ((dt * cloth.frequency.max(10.0)).ceil() as usize).clamp(1, 8);
        let h = dt / steps as f32;
        let gravity = Vec3::from(cloth.gravity);
        let keep = (1.0 - cloth.damping.clamp(0.0, 1.0)).powf(h * 60.0);
        let shape_k = 1.0 - (1.0 - cloth.shape_stiffness.clamp(0.0, 1.0)).powf(h * 90.0);
        let n = self.particles.len();
        let mut target = vec![Vec3::ZERO; n];
        let mut old = vec![Vec3::ZERO; n];
        let shapes: &[ClothShapeWorld] = if cloth.collide { colliders } else { &[] };
        let scale = crate::cloth::max_scale(world);
        let radius = cloth.radius.max(0.0) * scale;
        for k in 0..steps {
            let f = (k + 1) as f32 / steps as f32;
            for (i, p) in self.particles.iter_mut().enumerate() {
                target[i] = p.prev_anim.lerp(anim[i], f);
                old[i] = p.pos;
                p.contact = None;
                if p.pin >= 1.0 {
                    p.pos = target[i];
                    continue;
                }
                p.vel = (p.vel + gravity * h) * keep;
                if !winds.is_empty() && cloth.wind > 0.0 {
                    let w: Vec3 = winds.iter().map(|z| z.at(p.pos, time + k as f32 * h)).sum();
                    p.vel += (w - p.vel) * (1.0 - (-1.5 * cloth.wind * h).exp());
                }
                p.pos += p.vel * h;
                // Partly pinned points lean toward the animation by their pin
                // (per 1/90 s, so the pull doesn't depend on the step count).
                let pin_k = 1.0 - (1.0 - p.pin.clamp(0.0, 1.0)).powf(h * 90.0);
                p.pos = p.pos.lerp(target[i], pin_k.max(shape_k));
            }
            for _ in 0..cloth.iterations.max(1) {
                solve_pairs(&mut self.particles, &self.edges, &target, cloth.stiffness, true);
                solve_pairs(&mut self.particles, &self.bends, &target, cloth.bending, false);
                for &(i, a) in &self.tethers {
                    let (i, a) = (i as usize, a as usize);
                    let rest = (target[i] - target[a]).length() * (1.0 + cloth.tether_slack.max(0.0));
                    let d = self.particles[i].pos - self.particles[a].pos;
                    if d.length() > rest {
                        self.particles[i].pos = self.particles[a].pos + d.normalize_or_zero() * rest;
                    }
                }
                for (p, t) in self.particles.iter_mut().zip(&target) {
                    if p.max_distance.is_finite() {
                        let d = p.pos - *t;
                        if d.length_squared() > p.max_distance * p.max_distance {
                            p.pos = *t + d.normalize_or_zero() * p.max_distance;
                        }
                    }
                }
                if cloth.self_collision {
                    self.self_collide(cloth.thickness * scale);
                }
                if !shapes.is_empty() {
                    for p in self.particles.iter_mut().filter(|p| p.pin < 1.0) {
                        for s in shapes {
                            if let Some((push, nrm)) = penetration(p.pos, radius, s) {
                                p.pos += push;
                                p.contact = Some(nrm);
                            }
                        }
                    }
                }
            }
            let friction = cloth.friction.clamp(0.0, 1.0);
            for (i, p) in self.particles.iter_mut().enumerate() {
                if p.pin >= 1.0 {
                    p.vel = Vec3::ZERO;
                    continue;
                }
                let mut v = (p.pos - old[i]) / h;
                if let Some(nrm) = p.contact {
                    let vn = v.dot(nrm);
                    v = nrm * vn.max(0.0) + (v - nrm * vn) * (1.0 - friction);
                }
                p.vel = v.clamp_length_max(30.0);
            }
        }
        for (p, a) in self.particles.iter_mut().zip(&anim) {
            p.prev_anim = *a;
        }
        self.anim = anim;
    }

    /// Push apart particles closer than `thickness` that aren't stretch
    /// neighbours (spatial hash).
    fn self_collide(&mut self, thickness: f32) {
        if thickness <= 0.0 {
            return;
        }
        let cell = thickness;
        let key = |p: Vec3| ((p.x / cell).floor() as i64, (p.y / cell).floor() as i64, (p.z / cell).floor() as i64);
        let mut grid: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();
        for (i, p) in self.particles.iter().enumerate() {
            grid.entry(key(p.pos)).or_default().push(i as u32);
        }
        let neighbours: std::collections::HashSet<(u32, u32)> = self.edges.iter().copied().collect();
        for i in 0..self.particles.len() {
            let (cx, cy, cz) = key(self.particles[i].pos);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let Some(list) = grid.get(&(cx + dx, cy + dy, cz + dz)) else { continue };
                        for &j in list {
                            let j = j as usize;
                            if j <= i || neighbours.contains(&(i.min(j) as u32, i.max(j) as u32)) {
                                continue;
                            }
                            let d = self.particles[j].pos - self.particles[i].pos;
                            let len = d.length();
                            if len >= thickness || len < 1e-7 {
                                continue;
                            }
                            let (wi, wj) = (1.0 - self.particles[i].pin.min(1.0), 1.0 - self.particles[j].pin.min(1.0));
                            let w = wi + wj;
                            if w <= 1e-6 {
                                continue;
                            }
                            let corr = d / len * (thickness - len) / w;
                            self.particles[i].pos -= corr * wi;
                            self.particles[j].pos += corr * wj;
                        }
                    }
                }
            }
        }
    }

    /// Move the skinned render vertices (object space, in place) by their
    /// particles' offset from the animated pose and rebuild their normals.
    pub fn apply(&self, positions: &mut [Vec3], normals: &mut [Vec3], world: Mat4) {
        if positions.len() != self.vertex_count || self.anim.len() != self.particles.len() {
            return;
        }
        let inv = world.inverse();
        let offsets: Vec<Vec3> = self
            .particles
            .iter()
            .zip(&self.anim)
            .map(|(p, a)| inv.transform_vector3(p.pos - *a))
            .collect();
        for (v, pos) in positions.iter_mut().enumerate() {
            *pos += offsets[self.owner[v] as usize];
        }
        let mut acc = vec![Vec3::ZERO; positions.len()];
        for tri in self.indices.as_chunks::<3>().0 {
            let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if a >= acc.len() || b >= acc.len() || c >= acc.len() {
                continue;
            }
            let fnorm = (positions[b] - positions[a]).cross(positions[c] - positions[a]);
            acc[a] += fnorm;
            acc[b] += fnorm;
            acc[c] += fnorm;
        }
        // Welded copies share their particle's normal so seams stay smooth.
        let mut by_particle = vec![Vec3::ZERO; self.particles.len()];
        for (v, n) in acc.iter().enumerate() {
            by_particle[self.owner[v] as usize] += *n;
        }
        for (v, n) in normals.iter_mut().enumerate() {
            let s = by_particle[self.owner[v] as usize];
            if s.length_squared() > 1e-20 {
                *n = s.normalize();
            }
        }
    }
}

/// (free particle, its nearest pinned particle) for every particle with a
/// pin of at most 0.5, by rest position.
fn tethers(rest: &[Vec3], pins: &[f32]) -> Vec<(u32, u32)> {
    let pinned: Vec<u32> = (0..rest.len() as u32).filter(|&i| pins[i as usize] > 0.5).collect();
    if pinned.is_empty() {
        return Vec::new();
    }
    (0..rest.len() as u32)
        .filter(|&i| pins[i as usize] <= 0.5)
        .map(|i| {
            let a = rest[i as usize];
            let near = *pinned
                .iter()
                .min_by(|x, y| (rest[**x as usize] - a).length_squared().total_cmp(&(rest[**y as usize] - a).length_squared()))
                .unwrap();
            (i, near)
        })
        .collect()
}

/// Distance constraints between particle pairs, rest lengths from the
/// animated pose. `stretch_only` lets bends compress freely.
fn solve_pairs(particles: &mut [MeshParticle], pairs: &[(u32, u32)], target: &[Vec3], stiffness: f32, stretch_only: bool) {
    let k = stiffness.clamp(0.0, 1.0);
    if k <= 0.0 {
        return;
    }
    for &(a, b) in pairs {
        let (a, b) = (a as usize, b as usize);
        let rest = (target[b] - target[a]).length();
        let d = particles[b].pos - particles[a].pos;
        let len = d.length();
        if len < 1e-7 || (!stretch_only && len < rest && k < 1.0 && len > rest * 0.5) {
            continue;
        }
        let (wa, wb) = (1.0 - particles[a].pin.min(1.0), 1.0 - particles[b].pin.min(1.0));
        let w = wa + wb;
        if w <= 1e-6 {
            continue;
        }
        let corr = d / len * (len - rest) * k / w;
        particles[a].pos += corr * wa;
        particles[b].pos -= corr * wb;
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// A square sheet in the XY plane, `n` x `n` quads, top row at y = 0.
    fn sheet(n: usize, size: f32) -> (Vec<Vec3>, Vec<u32>) {
        let mut pos = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                pos.push(Vec3::new(x as f32 / n as f32 * size, -(y as f32) / n as f32 * size, 0.0));
            }
        }
        let mut idx = Vec::new();
        let w = n as u32 + 1;
        for y in 0..n as u32 {
            for x in 0..n as u32 {
                let i = y * w + x;
                idx.extend([i, i + w, i + 1, i + 1, i + w, i + w + 1]);
            }
        }
        (pos, idx)
    }

    #[test]
    fn a_pinned_sheet_hangs_and_keeps_its_size() {
        let (pos, idx) = sheet(8, 1.0);
        let pins: Vec<f32> = pos.iter().map(|p| if p.y > -1e-3 { 1.0 } else { 0.0 }).collect();
        let cloth = MeshCloth { shape_stiffness: 0.0, ..Default::default() };
        let mut sim = MeshClothSim::build(&pos, &idx, &pins, 0.0, Mat4::IDENTITY).unwrap();
        // Tilt the rest pose so gravity has something to do: the sheet starts horizontal.
        let flat: Vec<Vec3> = pos.iter().map(|p| Vec3::new(p.x, 0.0, -p.y)).collect();
        let mut sim_flat = MeshClothSim::build(&flat, &idx, &pins, 0.0, Mat4::IDENTITY).unwrap();
        for _ in 0..240 {
            sim_flat.step(&cloth, &flat, Mat4::IDENTITY, &[], &[], 0.0, 1.0 / 60.0);
            sim.step(&cloth, &pos, Mat4::IDENTITY, &[], &[], 0.0, 1.0 / 60.0);
        }
        let low = sim_flat.particles.iter().map(|p| p.pos.y).fold(f32::MAX, f32::min);
        assert!(low < -0.7, "the free edge fell: {low}");
        let top_ok = sim_flat.particles.iter().filter(|p| p.pin >= 1.0).all(|p| p.pos.y.abs() < 1e-4);
        assert!(top_ok, "pinned row stays put");
        let mut positions = flat.clone();
        let mut normals = vec![Vec3::Y; flat.len()];
        sim_flat.apply(&mut positions, &mut normals, Mat4::IDENTITY);
        let span = positions.iter().map(|p| p.length()).fold(0.0, f32::max);
        assert!(span < 1.6, "stretch limited: {span}");
        assert!(normals.iter().all(|n| (n.length() - 1.0).abs() < 1e-3));
    }

    #[test]
    fn seams_weld_and_spacing_reduces() {
        let (mut pos, mut idx) = sheet(4, 1.0);
        // Duplicate one vertex as a UV-seam copy used by a new triangle.
        pos.push(pos[0]);
        idx.extend([pos.len() as u32 - 1, 5, 1]);
        let pins = vec![0.0; pos.len()];
        let sim = MeshClothSim::build(&pos, &idx, &pins, 0.0, Mat4::IDENTITY).unwrap();
        assert_eq!(sim.particles.len(), 25, "the seam copy welds");
        let coarse = MeshClothSim::build(&pos, &idx, &pins, 0.5, Mat4::IDENTITY).unwrap();
        assert!(coarse.particles.len() < 12, "{}", coarse.particles.len());
    }

    #[test]
    fn max_distance_keeps_free_points_near_the_animation() {
        let (pos, idx) = sheet(6, 1.0);
        let flat: Vec<Vec3> = pos.iter().map(|p| Vec3::new(p.x, 0.0, -p.y)).collect();
        let pins: Vec<f32> = flat.iter().map(|p| if p.z.abs() < 1e-3 { 1.0 } else { 0.0 }).collect();
        let cloth = MeshCloth { shape_stiffness: 0.0, ..Default::default() };
        let mut sim = MeshClothSim::build(&flat, &idx, &pins, 0.0, Mat4::IDENTITY).unwrap();
        sim.set_limits(Some(&vec![0.05; flat.len()]));
        for _ in 0..240 {
            sim.step(&cloth, &flat, Mat4::IDENTITY, &[], &[], 0.0, 1.0 / 60.0);
        }
        let anim = sim.animated(&flat, Mat4::IDENTITY);
        for (p, a) in sim.particles.iter().zip(&anim) {
            assert!((p.pos - *a).length() < 0.05 + 1e-4, "strayed {}", (p.pos - *a).length());
        }
        sim.set_limits(None);
        for _ in 0..240 {
            sim.step(&cloth, &flat, Mat4::IDENTITY, &[], &[], 0.0, 1.0 / 60.0);
        }
        let low = sim.particles.iter().map(|p| p.pos.y).fold(f32::MAX, f32::min);
        assert!(low < -0.5, "lifting the limit lets it fall: {low}");
    }

    #[test]
    fn new_pins_apply_without_a_rebuild() {
        let (pos, idx) = sheet(6, 1.0);
        let flat: Vec<Vec3> = pos.iter().map(|p| Vec3::new(p.x, 0.0, -p.y)).collect();
        let cloth = MeshCloth { shape_stiffness: 0.0, ..Default::default() };
        let mut sim = MeshClothSim::build(&flat, &idx, &vec![0.0; flat.len()], 0.0, Mat4::IDENTITY).unwrap();
        let top: Vec<f32> = flat.iter().map(|p| if p.z.abs() < 1e-3 { 1.0 } else { 0.0 }).collect();
        sim.set_pins(&top, true);
        assert!(!sim.tethers.is_empty());
        for _ in 0..120 {
            sim.step(&cloth, &flat, Mat4::IDENTITY, &[], &[], 0.0, 1.0 / 60.0);
        }
        for (p, rest) in sim.particles.iter().zip(&sim.rest) {
            if p.pin >= 1.0 {
                assert!((p.pos - *rest).length() < 1e-4, "pinned stays");
            }
        }
        assert!(sim.particles.iter().any(|p| p.pin == 0.0 && p.pos.y < -0.3), "free ones fall");
    }

    #[test]
    fn old_scenes_load_and_paint_round_trips() {
        let old: MeshCloth = ron::from_str("(slot: 1, pin: Top, pin_height: 0.2, stiffness: 0.5)").unwrap();
        assert_eq!((old.slot, old.pin, old.pin_height, old.stiffness), (1, PinSource::Top, 0.2, 0.5));
        assert!(old.paint.is_empty());
        assert_eq!((old.max_distance, old.cull_distance), (0.1, 0.0));
        let (pos, idx) = sheet(4, 1.0);
        let pins: Vec<f32> = pos.iter().map(|p| if p.y > -0.3 { 1.0 } else { 0.25 }).collect();
        let cloth = MeshCloth { pin: PinSource::Painted, paint: crate::ClothPaint::seed(&pos, &idx, &pins), ..Default::default() };
        let text = ron::to_string(&cloth).unwrap();
        let back: MeshCloth = ron::from_str(&text).unwrap();
        assert_eq!(back.pin, PinSource::Painted);
        assert_eq!(back.paint, cloth.paint);
        let w = back.paint.pin_weights();
        assert!(w.iter().zip(&pins).all(|(a, b)| (a - b).abs() < 1.0 / 255.0));
    }

    #[test]
    fn colliders_and_self_collision_push_apart() {
        let (pos, idx) = sheet(6, 1.0);
        let flat: Vec<Vec3> = pos.iter().map(|p| Vec3::new(p.x, 0.0, -p.y)).collect();
        let pins: Vec<f32> = flat.iter().map(|p| if p.z.abs() < 1e-3 { 1.0 } else { 0.0 }).collect();
        let cloth = MeshCloth { self_collision: true, thickness: 0.05, ..Default::default() };
        let mut sim = MeshClothSim::build(&flat, &idx, &pins, 0.0, Mat4::IDENTITY).unwrap();
        let ball = ClothShapeWorld::Sphere { c: Vec3::new(0.5, -0.6, 0.4), r: 0.3 };
        for _ in 0..300 {
            sim.step(&cloth, &flat, Mat4::IDENTITY, &[ball], &[], 0.0, 1.0 / 60.0);
        }
        for p in &sim.particles {
            assert!((p.pos - Vec3::new(0.5, -0.6, 0.4)).length() > 0.3 + cloth.radius - 0.02);
        }
    }
}
