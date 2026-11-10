//! Mesh cloth: simulate a skinned garment from its own triangles. Render
//! vertices are welded (UV seams) and optionally clustered into a coarser
//! particle set; triangle edges become stretch constraints, edges across
//! neighbouring triangles bending constraints, and free particles are
//! tethered to their nearest pinned one. Pinned particles follow the
//! animation; every render vertex is moved by its particles' offset from the
//! animated pose.

use std::collections::HashMap;

use glam::{Mat4, Vec2, Vec3};
#[path = "cloth_fabric.rs"]
mod fabric;
#[path = "cloth_lod_transfer.rs"]
mod lod_transfer;
#[path="cloth_mesh_contact.rs"]
mod mesh_contact;
#[path="cloth_mesh_self_contact.rs"]
mod self_contact;
#[path="cloth_mesh_cluster.rs"]
mod clustering;
#[path="cloth_sewing.rs"]
mod sewing;
use sewing::ParticleStitch;
pub use fabric::{ClothFabric, ClothFabricPreset};
use serde::{Deserialize, Serialize};

use crate::cloth::{ClothShapeWorld, ClothTeleport, WindSample, penetration};
use crate::cloth_contact::ClothColliderMotion;
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
    pub cache:crate::cloth_cache::ClothCacheRuntime,
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
    /// Any painted channel changed: push it into the running simulation;
    /// `retether` also rebuilds the tethers (end of a stroke).
    pub pins_dirty: bool,
    /// Also rebuild the tethers when the pins change.
    pub retether: bool,
    /// Welded topology for the brush, built on demand.
    pub weld: Option<WeldTopology>,
    /// Shown in the inspector (e.g. the paint was remapped to a new mesh).
    pub paint_note: String,
    /// Engine source/build error shown by the cloth inspector.
    pub build_error: String,
    /// Beyond the cull distance from the viewer this frame.
    pub culled: bool,
    /// Identity of mesh, skin bindings and build settings used by this simulation.
    pub source_key: u64,
    /// Stable collider identities keep animation sweeps separate from replacement.
    pub collider_history: HashMap<String,ClothShapeWorld>,
    /// Held or culled last step: restart from the animated pose.
    pub idle: bool,
    /// Selected simulation tier (zero = near).
    pub lod_level: usize,
    /// Spacing actually used by the current simulation.
    pub lod_spacing: f32,
    /// Last requested spacing, including a too-coarse request clamped by the engine.
    pub lod_requested_spacing: f32,
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

/// Simulates one static or skinned render mesh of this object as cloth.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct MeshCloth {
    pub cache:crate::ClothCacheSettings,
    /// Render slot (submesh) of the object to simulate.
    pub slot: usize,
    /// Reusable garment asset used to seed this instance; settings remain local.
    pub garment: String,
    /// Render-vertex sewing constraints, preserved across simulation LOD.
    pub stitches: Vec<crate::ClothStitch>,
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
    /// Physical fabric model. Missing in saved scenes = original artistic solver.
    #[serde(default)]
    pub fabric: Option<ClothFabric>,
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
    /// Sweep particles and cloth features over each substep.
    pub continuous_collision: bool,
    /// Resolve existing non-adjacent edge/face intersections.
    pub untangle: bool,
    /// Garments in the same nonzero group collide after their local simulation.
    /// Zero disables garment-to-garment contacts without changing body collision.
    pub collision_group: u32,
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
    /// Independent painted pins, movement limits and material/drive/backstop maps.
    /// Only the pin channel requires `PinSource::Painted`.
    pub paint: ClothPaint,
    /// How far (object units, about metres) a vertex whose painted max
    /// distance is 1 may stray from its animated position.
    pub max_distance: f32,
    /// Stop simulating (hold the animated pose) beyond this distance from the
    /// camera (m, 0 = never).
    pub cull_distance: f32,
    /// Distance-based simulation resolution and transition controls.
    pub lod: crate::ClothLod,
    /// Engine-filled state (not saved).
    #[serde(skip)]
    pub runtime: MeshClothRuntime,
}

/// Cloning a cloth copies its authored settings, never its live solver state.
impl Clone for MeshCloth {fn clone(&self)->Self{Self{
    cache:self.cache.clone(),
    slot:self.slot.clone(),
    garment:self.garment.clone(),
    stitches:self.stitches.clone(),
    pin:self.pin.clone(),
    pin_channel:self.pin_channel.clone(),
    pin_bones:self.pin_bones.clone(),
    pin_height:self.pin_height.clone(),
    particle_spacing:self.particle_spacing.clone(),
    stiffness:self.stiffness.clone(),
    fabric:self.fabric.clone(),
    bending:self.bending.clone(),
    shape_stiffness:self.shape_stiffness.clone(),
    tether_slack:self.tether_slack.clone(),
    gravity:self.gravity.clone(),
    damping:self.damping.clone(),
    radius:self.radius.clone(),
    friction:self.friction.clone(),
    collide:self.collide.clone(),
    continuous_collision:self.continuous_collision.clone(),
    untangle:self.untangle.clone(),
    collision_group:self.collision_group.clone(),
    colliders:self.colliders.clone(),
    collider_objects:self.collider_objects.clone(),
    self_collision:self.self_collision.clone(),
    thickness:self.thickness.clone(),
    world_inertia:self.world_inertia.clone(),
    max_speed:self.max_speed.clone(),
    teleport_distance:self.teleport_distance.clone(),
    teleport_angle:self.teleport_angle.clone(),
    teleport:self.teleport.clone(),
    wind:self.wind.clone(),
    frequency:self.frequency.clone(),
    iterations:self.iterations.clone(),
    paint:self.paint.clone(),
    max_distance:self.max_distance.clone(),
    cull_distance:self.cull_distance.clone(),
    lod:self.lod.clone(),
    runtime:Default::default(),
}}}

impl Default for MeshCloth {
    fn default() -> Self {
        Self {
            slot: 0,
            cache:Default::default(),
            garment:String::new(),stitches:Vec::new(),
            pin: PinSource::VertexColor,
            pin_channel: 0,
            pin_bones: Vec::new(),
            pin_height: 0.15,
            particle_spacing: 0.0,
            stiffness: 0.9,
            fabric: Some(ClothFabric::default()),
            bending: 0.3,
            shape_stiffness: 0.02,
            tether_slack: 0.1,
            gravity: [0.0, -9.81, 0.0],
            damping: 0.05,
            radius: 0.01,
            friction: 0.3,
            collide: true,
            continuous_collision: true,
            untangle: true,
            collision_group: 1,
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
            lod: crate::ClothLod::default(),
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
    fabric: fabric::FabricState,
    frame_previous:Vec<Vec3>,
    visible_frame_previous:Vec<Vec3>,
    animated_vertices:Vec<Vec3>,
    normal_owner:Vec<usize>,
    surface_contacts:crate::cloth_contact::ClothContactTopology,
    surface_mapping:crate::cloth_contact::ClothContactMapping,
    transition_offsets:Vec<Vec3>,
    transition_remaining:f32,
    transition_duration:f32,
    /// Warp, weft, shear, bend, drive, backstop radius and gap multipliers.
    material_masks:Vec<[f32;7]>,
    sewing:Vec<ParticleStitch>,
    seam_neighbours:std::collections::HashSet<(u32,u32)>,
}

/// World-space snapshot used by the scene's shared garment contact pass.
pub struct ClothCollisionGeometry {
    pub positions:Vec<Vec3>,
    pub previous:Vec<Vec3>,
    pub inverse_mass:Vec<f32>,
    pub triangles:Vec<[usize;3]>,
}

impl MeshClothSim {
    /// Render vertices in the simulated submesh.
    pub fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    /// Build from the mesh at rest (object space), its triangles and each
    /// render vertex's pin weight. `world` is the object's world matrix.
    pub fn build(positions: &[Vec3], indices: &[u32], pins: &[f32], spacing: f32, world: Mat4) -> Option<Self> {
        if positions.is_empty() || indices.len() < 3 || indices.len()%3!=0
            || positions.iter().any(|p|!p.is_finite()) || indices.iter().any(|&i|i as usize>=positions.len())
            || !world.is_finite() || world.determinant().abs()<1e-10 {
            return None;
        }
        let (owner,members)=clustering::build(positions,indices,pins,spacing);
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
        let mut edges: Vec<(u32, u32)> = edge_set.into_iter().collect();
        edges.sort_unstable();
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
        let fabric=fabric::FabricState::build(&rest_pos,&owner,indices,&[],&edges,&particle_pins);
        // Normal sharing uses only the original seam weld, never the coarse LOD.
        let mut normal_cells=HashMap::new();
        let normal_owner:Vec<usize>=positions.iter().map(|p|{let next=normal_cells.len();*normal_cells.entry(((p.x/1e-4).round() as i64,(p.y/1e-4).round() as i64,(p.z/1e-4).round() as i64)).or_insert(next)}).collect();
        let surface_contacts=self_contact::topology(indices,&normal_owner);
        let surface_mapping=crate::cloth_contact::ClothContactMapping::new(&owner.iter().map(|&i|i as usize).collect::<Vec<_>>());
        Some(Self {
            surface_contacts,surface_mapping,
            sewing:Vec::new(),seam_neighbours:Default::default(),
            material_masks:vec![[1.0;7];members.len()],
            animated_vertices:positions.to_vec(),normal_owner,transition_offsets:Vec::new(),transition_remaining:0.0,transition_duration:0.0,
            frame_previous:rest_pos.iter().map(|p|world.transform_point3(*p)).collect(),
            visible_frame_previous:Vec::new(),
            fabric,
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
            self.fabric.rebuild_tethers(&self.rest,&self.edges,&pins);
        }
    }

    /// Supply original render-vertex UVs; keeps seam orientation per triangle.
    pub fn set_texcoords(&mut self, uv:&[Vec2]) {
        let pins:Vec<_>=self.particles.iter().map(|p|p.pin).collect();
        self.fabric=fabric::FabricState::build(&self.rest,&self.owner,&self.indices,uv,&self.edges,&pins);
        self.fabric.set_material_masks(&self.material_masks);
        self.fabric.exclude_seams(&self.seam_neighbours);
    }

    /// Fit authored material maps to this simulation resolution. Neutral maps
    /// are all ones; changes apply without resetting positions or velocities.
    pub fn set_painted_maps(&mut self,paint:&ClothPaint){
        self.material_masks.fill([1.0;7]);
        if paint.matches(self.vertex_count,crate::topology_hash(self.vertex_count,&self.indices)){
            for (channel_index,channel) in crate::PaintChannel::MATERIAL.into_iter().enumerate(){
                let Some(bytes)=paint.channel(channel)else{continue};
                for (mask,members) in self.material_masks.iter_mut().zip(&self.members){
                    mask[channel_index]=members.iter().map(|&v|crate::cloth_pins::dequantize(bytes.as_slice()[v as usize])).sum::<f32>()/members.len() as f32;
                }
            }
        }
        self.fabric.set_material_masks(&self.material_masks);
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
        if skinned.len()!=self.vertex_count || skinned.iter().any(|p|!p.is_finite()) || !world.is_finite() || world.determinant().abs()<1e-10{return;}
        let anim = self.animated(skinned, world);
        self.animated_vertices=skinned.to_vec();
        self.transition_offsets.clear();self.transition_remaining=0.0;
        self.anim=anim.clone();
        self.frame_previous=anim.clone();
        self.visible_frame_previous.clear();
        let scale=crate::cloth::max_scale(world)/crate::cloth::max_scale(self.last_world).max(1e-6);
        for (p, a) in self.particles.iter_mut().zip(&anim) {
            if p.max_distance.is_finite(){p.max_distance*=scale;}
            p.pos = *a;
            p.prev_anim = *a;
            p.vel = Vec3::ZERO;
        }
        self.last_world = world;
        self.visible_frame_previous=if self.has_collision_detail(){self.visible_collision_positions()}else{Vec::new()};
    }

    /// Advance by `dt`. `skinned` is this frame's skinned render vertex
    /// positions (object space).
    #[allow(clippy::too_many_arguments)]
    pub fn step(&mut self, cloth: &MeshCloth, skinned: &[Vec3], world: Mat4, colliders: &[ClothShapeWorld], winds: &[WindSample], time: f32, dt: f32) {
        let motions:Vec<_>=colliders.iter().cloned().map(ClothColliderMotion::stationary).collect();
        self.step_moving(cloth,skinned,world,&motions,winds,time,dt);
    }

    /// Moving collider endpoints are supplied by the scene's stable collider
    /// identities; newly added/replaced colliders start with zero sweep.
    #[allow(clippy::too_many_arguments)]
    pub fn step_moving(&mut self, cloth:&MeshCloth, skinned:&[Vec3], world:Mat4, colliders:&[ClothColliderMotion], winds:&[WindSample], time:f32, dt:f32) {
        if !dt.is_finite() || dt <= 0.0 || skinned.len() != self.vertex_count
            || !world.is_finite() || world.determinant().abs()<1e-10 || skinned.iter().any(|p|!p.is_finite()) {
            return;
        }
        self.visible_frame_previous=if self.has_collision_detail()||(cloth.collide&&!colliders.is_empty()){self.visible_collision_positions()}else{Vec::new()};
        let detail_previous=(!self.visible_frame_previous.is_empty()).then(||self.visible_frame_previous.clone());
        let dt = dt.min(0.1);
        self.transition_remaining=(self.transition_remaining-dt).max(0.0);
        if self.transition_remaining==0.0{self.transition_offsets.clear();}
        let anim = self.animated(skinned, world);
        self.prepare_seams(skinned,world,&anim);
        self.animated_vertices=skinned.to_vec();
        let rot = |m: &Mat4| m.to_scale_rotation_translation().1;
        let moved = world.w_axis.truncate() - self.last_world.w_axis.truncate();
        let turned = rot(&self.last_world).angle_between(rot(&world)).to_degrees();
        let last_inv = self.last_world.inverse();
        let teleported=(cloth.teleport_distance > 0.0 && moved.length() > cloth.teleport_distance)
            || (cloth.teleport_angle > 0.0 && turned > cloth.teleport_angle);
        if teleported {
            match cloth.teleport {
                ClothTeleport::Reset => {
                    self.reset(skinned, world);
                    return;
                }
                ClothTeleport::Keep => {
                    let delta = world * last_inv;
                    self.carry_seams(delta);
                    for p in &mut self.particles {
                        p.pos = delta.transform_point3(p.pos);
                        p.vel = delta.transform_vector3(p.vel);
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
        let previous_scale=crate::cloth::max_scale(self.last_world).max(1e-6);
        let current_scale=crate::cloth::max_scale(world);
        for p in &mut self.particles {if p.max_distance.is_finite(){p.max_distance*=current_scale/previous_scale;}}
        self.last_world = world;
        if let Some(cfg)=&cloth.fabric{self.fabric.prepare(&self.rest,world,&self.particles,cfg);}
        else{self.fabric.weights=self.particles.iter().map(|p|1.0-p.pin.clamp(0.0,1.0)).collect();}
        // Recover an imported/intersecting pose before it becomes the sweep
        // origin. Material forces must not build on an invalid contact state.
        if cloth.self_collision&&cloth.untangle{
            self.fabric.recover_intersections(&mut self.particles,cloth.thickness*current_scale);
        }
        self.frame_previous=self.particles.iter().map(|p|p.pos).collect();

        let (_,frequency,iterations)=cloth.lod.settings(cloth.runtime.lod_level,cloth.particle_spacing,cloth.frequency,cloth.iterations);
        let steps = ((dt * frequency.max(10.0)).ceil() as usize).clamp(1, 8);
        let h = dt / steps as f32;
        let gravity = Vec3::from(cloth.gravity);
        let keep = (1.0 - cloth.damping.clamp(0.0, 1.0)).powf(h * 60.0);
        let shape_k = 1.0 - (1.0 - cloth.shape_stiffness.clamp(0.0, 1.0)).powf(h * 90.0);
        let n = self.particles.len();
        let mut target = vec![Vec3::ZERO; n];
        let mut old = vec![Vec3::ZERO; n];
        let shapes: &[ClothColliderMotion] = if cloth.collide { colliders } else { &[] };
        let mut surface_velocity=vec![Vec3::ZERO;n];
        let scale = crate::cloth::max_scale(world);
        let radius = cloth.radius.max(0.0) * scale;
        for k in 0..steps {
            for seam in &mut self.sewing{seam.lambda=0.0;}
            let f = (k + 1) as f32 / steps as f32;
            let substep_shapes:Vec<_>=shapes.iter().map(|s|if teleported{ClothColliderMotion::stationary(s.end.clone())}else{s.interval(k as f32/steps as f32,f)}).collect();
            let queries:Vec<_>=substep_shapes.iter().map(ClothColliderMotion::prepare).collect();
            surface_velocity.fill(Vec3::ZERO);
            for (i,p) in self.particles.iter().enumerate(){target[i]=p.prev_anim.lerp(anim[i],f);}
            let forces=cloth.fabric.as_ref().map(|cfg|{
                self.fabric.begin_step(&target);
                self.fabric.forces(&self.particles,cfg,winds,cloth.wind,time+k as f32*h)
            });
            for (i, p) in self.particles.iter_mut().enumerate() {
                target[i] = p.prev_anim.lerp(anim[i], f);
                old[i] = p.pos;
                p.contact = None;
                if p.pin >= 1.0 {
                    p.pos = target[i];
                    continue;
                }
                p.vel = (p.vel + gravity * h) * keep;
                if let Some(cfg)=&cloth.fabric {
                    p.vel+=(forces.as_ref().unwrap()[i]*self.fabric.weights[i]*h).clamp_length_max(30.0);
                    let omega=std::f32::consts::TAU*(cfg.animation_hz*self.material_masks[i][4]).clamp(0.0,60.0);
                    if omega>0.0 {
                        let damping=2.0*omega*cfg.animation_damping.clamp(0.0,4.0);
                        let target_velocity=(anim[i]-p.prev_anim)/dt;
                        p.vel=(p.vel+(target[i]-p.pos)*(h*omega*omega)+target_velocity*(h*damping))/(1.0+h*damping+h*h*omega*omega);
                    }
                } else if !winds.is_empty() && cloth.wind > 0.0 {
                    let w: Vec3 = winds.iter().map(|z| z.at(p.pos, time + k as f32 * h)).sum();
                    p.vel += (w - p.vel) * (1.0 - (-1.5 * cloth.wind * h).exp());
                }
                p.pos += p.vel * h;
                // Partly pinned points lean toward the animation by their pin
                // (per 1/90 s, so the pull doesn't depend on the step count).
                let pin_k = 1.0 - (1.0 - p.pin.clamp(0.0, 1.0)).powf(h * 90.0);
                p.pos = p.pos.lerp(target[i], if cloth.fabric.is_some(){pin_k}else{pin_k.max(shape_k)});
            }
            let iterations=iterations.clamp(1,64);
            let contact_passes=cloth.fabric.as_ref().map_or(iterations,|f|f.self_collision_passes.clamp(1,iterations));
            for iteration in 0..iterations {
                if let Some(cfg)=&cloth.fabric {
                    self.fabric.solve(&mut self.particles,cfg,h);
                    for &(i,a,length) in &self.fabric.tethers {
                        let max=length*scale*(1.0+cloth.tether_slack.max(0.0));
                        let d=self.particles[i].pos-self.particles[a].pos;
                        if d.length()>max{self.particles[i].pos=self.particles[a].pos+d.normalize_or_zero()*max;}
                    }
                } else {
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
                }
                self.solve_seams(cloth.fabric.is_some(),h,f);
                for (i,(p, t)) in self.particles.iter_mut().zip(&target).enumerate() {
                    if p.pin>=1.0{continue;}
                    if let Some(cfg)=&cloth.fabric {if cfg.backstop_radius>0.0 {
                        let n=self.fabric.normals[i];let radius=cfg.backstop_radius*self.material_masks[i][5]*scale;
                        let center=*t-n*(radius+cfg.backstop_distance*self.material_masks[i][6]*scale);
                        let d=p.pos-center;let len=d.length();
                        if len<radius {p.pos=center+if len>1e-8{d*(radius/len)}else{n*radius};}
                    }}
                    if p.max_distance.is_finite() {
                        let d = p.pos - *t;
                        if d.length_squared() > p.max_distance * p.max_distance {
                            p.pos = *t + d.normalize_or_zero() * p.max_distance;
                        }
                    }
                }
                if cloth.self_collision && (iteration+1)*contact_passes/iterations>iteration*contact_passes/iterations {
                    self.self_collide(cloth.thickness * scale,cloth.fabric.is_some());
                    if cloth.fabric.is_some()||iteration+1==iterations{self.fabric.self_collide_swept(&mut self.particles,&old,cloth.thickness*scale,cloth.continuous_collision,cloth.untangle);}
                }
                if !shapes.is_empty() {
                    for (i,p) in self.particles.iter_mut().enumerate().filter(|(_,p)|p.pin<1.0) {
                        for query in &queries {
                            if cloth.continuous_collision {
                                if let Some((position,nrm,carried))=query.resolve(old[i],p.pos,radius){
                                    p.pos=position;p.contact=Some(nrm);surface_velocity[i]=carried/h;
                                }
                            } else if let Some((push,nrm))=penetration(p.pos,radius,query.end()){p.pos+=push;p.contact=Some(nrm);}
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
                    let relative=v-surface_velocity[i];let vn=relative.dot(nrm);
                    v=surface_velocity[i]+nrm*vn.max(0.0)+(relative-nrm*vn)*(1.0-friction);
                }
                p.vel = v.clamp_length_max(30.0);
            }
        }
        for (p, a) in self.particles.iter_mut().zip(&anim) {
            p.prev_anim = *a;
        }
        self.anim = anim;
        if detail_previous.is_some(){
            let stationary:Vec<_>=if teleported||!cloth.continuous_collision{shapes.iter().map(|s|ClothColliderMotion::stationary(s.end.clone())).collect()}else{Vec::new()};
            let shapes=if teleported||!cloth.continuous_collision{&stationary}else{shapes};
            let queries:Vec<_>=shapes.iter().map(ClothColliderMotion::prepare).collect();
            self.collide_visible_constraints(cloth,&queries,if teleported{None}else{detail_previous.as_deref()},true,dt);
        }
    }

    pub fn collision_geometry(&self,physical:bool)->ClothCollisionGeometry {
        ClothCollisionGeometry {
            positions:self.particles.iter().map(|p|p.pos).collect(),
            previous:self.frame_previous.clone(),
            inverse_mass:if physical&&self.fabric.weights.len()==self.particles.len(){self.fabric.weights.clone()}else{self.particles.iter().map(|p|1.0-p.pin.clamp(0.0,1.0)).collect()},
            triangles:self.fabric.contact_triangles(),
        }
    }
    /// Commit shared contacts while keeping hard pins, animation limits and
    /// body colliders authoritative. The scene iterates shared/body constraints
    /// together rather than letting the last garment push another into its body.
    pub fn apply_contact_positions(&mut self,cloth:&MeshCloth,positions:&[Vec3],colliders:&[ClothShapeWorld],dt:f32){
        if positions.len()!=self.particles.len(){return;}
        let detail_previous=(cloth.self_collision||(cloth.collide&&!colliders.is_empty())).then(||self.visible_collision_positions());
        let scale=crate::cloth::max_scale(self.last_world);
        let motions:Vec<_>=colliders.iter().cloned().map(ClothColliderMotion::stationary).collect();
        let queries:Vec<_>=motions.iter().map(ClothColliderMotion::prepare).collect();
        for (i,(p,&candidate)) in self.particles.iter_mut().zip(positions).enumerate(){
            if p.pin>=1.0||!candidate.is_finite(){continue;}let old=p.pos;let mut position=candidate;
            if let Some(&target)=self.anim.get(i){
                if p.max_distance.is_finite(){position=target+(position-target).clamp_length_max(p.max_distance);}
                if let Some(cfg)=&cloth.fabric {if cfg.backstop_radius>0.0&&i<self.fabric.normals.len(){
                    let radius=cfg.backstop_radius*self.material_masks[i][5]*scale;let n=self.fabric.normals[i];let center=target-n*(radius+cfg.backstop_distance*self.material_masks[i][6]*scale);let delta=position-center;
                    if delta.length()<radius{position=center+delta.normalize_or(n)*radius;}
                }}
            }
            if cloth.collide {for query in &queries {
                if let Some((out,normal,_))=query.resolve(old,position,cloth.radius*scale){position=out;p.contact=Some(normal);}
            }}
            p.pos=position;
            if dt>1e-6{p.vel=(p.vel+(position-old)/dt).clamp_length_max(30.0);}
        }
        self.collide_visible_constraints(cloth,&queries,detail_previous.as_deref(),true,dt);
    }

    /// Push apart particles closer than `thickness` that aren't stretch
    /// neighbours (spatial hash).
    fn self_collide(&mut self, thickness: f32, physical:bool) {
        if thickness <= 0.0 || !thickness.is_finite() {
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
                            if j <= i || neighbours.contains(&(i.min(j) as u32, i.max(j) as u32)) || self.seam_neighbours.contains(&(i.min(j) as u32,i.max(j) as u32)) {
                                continue;
                            }
                            let d = self.particles[j].pos - self.particles[i].pos;
                            let len = d.length();
                            if len >= thickness {
                                continue;
                            }
                            let (wi, wj) = if physical {(self.fabric.weights[i],self.fabric.weights[j])}else{(1.0 - self.particles[i].pin.min(1.0), 1.0 - self.particles[j].pin.min(1.0))};
                            let w = wi + wj;
                            if w <= 1e-6 {
                                continue;
                            }
                            let direction=if len>1e-7 {d/len}else{(self.rest[j]-self.rest[i]).normalize_or(Vec3::X)};
                            let corr = direction * (thickness - len) / w;
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
        if positions.len() != self.vertex_count || normals.len()!=self.vertex_count || self.anim.len() != self.particles.len()
            || !world.is_finite() || world.determinant().abs()<1e-10 {
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
            *pos += offsets[self.owner[v] as usize]+self.transition_offset(v);
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
        let mut by_particle = vec![Vec3::ZERO; self.vertex_count];
        for (v, n) in acc.iter().enumerate() {
            by_particle[self.normal_owner[v]] += *n;
        }
        for (v, n) in normals.iter_mut().enumerate() {
            let s = by_particle[self.normal_owner[v]];
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
        let cloth = MeshCloth { fabric: None, shape_stiffness: 0.0, ..Default::default() };
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
        let cloth = MeshCloth { fabric: None, shape_stiffness: 0.0, ..Default::default() };
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
        let cloth = MeshCloth { fabric: None, shape_stiffness: 0.0, ..Default::default() };
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
        assert!(old.fabric.is_none());
        assert!(MeshCloth::default().fabric.is_some());
        assert_eq!((old.max_distance, old.cull_distance), (0.1, 0.0));
        let (pos, idx) = sheet(4, 1.0);
        let pins: Vec<f32> = pos.iter().map(|p| if p.y > -0.3 { 1.0 } else { 0.25 }).collect();
        let cloth = MeshCloth { fabric: None, pin: PinSource::Painted, paint: crate::ClothPaint::seed(&pos, &idx, &pins), ..Default::default() };
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
        let cloth = MeshCloth { fabric: None, self_collision: true, thickness: 0.05, ..Default::default() };
        let mut sim = MeshClothSim::build(&flat, &idx, &pins, 0.0, Mat4::IDENTITY).unwrap();
        let ball = ClothShapeWorld::Sphere { c: Vec3::new(0.5, -0.6, 0.4), r: 0.3 };
        for _ in 0..300 {
            sim.step(&cloth, &flat, Mat4::IDENTITY, std::slice::from_ref(&ball), &[], 0.0, 1.0 / 60.0);
        }
        for p in &sim.particles {
            assert!((p.pos - Vec3::new(0.5, -0.6, 0.4)).length() > 0.3 + cloth.radius - 0.02);
        }
    }
}
