//! Bone cloth: position-based secondary motion for joint chains (hair, tails,
//! ears, bone-rigged skirts) and the cloth colliders it pushes against
//! (spheres, tapered capsules, planes) attached to skeleton joints.
//!
//! The engine drives both from the skinning pass: it hands the simulation the
//! animated joint matrices, steps it, and writes the simulated rotations back
//! into the joint locals before the skinning palette is built. Everything is
//! simulated in world space so moving the avatar produces inertia.

use glam::{Mat3, Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::{ObjectRef, TypedComponent};

/// The shape of a [`ClothCollider`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ClothShape {
    /// A (possibly tapered) capsule.
    #[default]
    Capsule,
    /// A sphere.
    Sphere,
    /// An infinite plane.
    Plane,
}

impl ClothShape {
    /// Every shape, in declaration order.
    pub const ALL: [ClothShape; 3] = [ClothShape::Capsule, ClothShape::Sphere, ClothShape::Plane];
    /// Inspector label of this shape.
    pub fn label(self) -> &'static str {
        match self {
            ClothShape::Capsule => "Capsule",
            ClothShape::Sphere => "Sphere",
            ClothShape::Plane => "Plane",
        }
    }
}

/// One collider, positioned in the frame of the joint it follows. The frame is
/// the joint's object-space translation and rotation with scale removed, so
/// offsets and radii are in the object's units whatever the rig's bone scale.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ClothCollider {
    /// Collider name.
    pub name: String,
    /// Whether the collider pushes cloth.
    pub enabled: bool,
    /// Joint the collider follows (empty = the object itself).
    pub bone: String,
    /// The collider's shape.
    pub shape: ClothShape,
    /// Capsule start, sphere centre or a point on the plane.
    pub center: [f32; 3],
    /// Capsule end.
    /// Capsule end.
    pub end: [f32; 3],
    /// Sphere radius / capsule radius at `center` (object units).
    pub radius: f32,
    /// Capsule radius at `end` (differs from `radius` for a tapered capsule).
    pub end_radius: f32,
    /// Plane normal (the side cloth stays on).
    pub normal: [f32; 3],
    /// Made by auto-fit; fitting again replaces these and keeps the others.
    pub auto: bool,
}

impl Default for ClothCollider {
    fn default() -> Self {
        Self {
            name: "Collider".into(),
            enabled: true,
            bone: String::new(),
            shape: ClothShape::Capsule,
            center: [0.0; 3],
            end: [0.0, 0.3, 0.0],
            radius: 0.06,
            end_radius: 0.05,
            normal: [0.0, 1.0, 0.0],
            auto: false,
        }
    }
}

/// A collider resolved into one space (object or world).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClothShapeWorld {
    /// A sphere.
    Sphere {
        /// Sphere centre.
        c: Vec3,
        /// Sphere radius.
        r: f32,
    },
    /// A (possibly tapered) capsule.
    Capsule {
        /// Capsule start.
        a: Vec3,
        /// Capsule end.
        b: Vec3,
        /// Radius at `a`.
        ra: f32,
        /// Radius at `b`.
        rb: f32,
    },
    /// An infinite plane.
    Plane {
        /// A point on the plane.
        p: Vec3,
        /// Unit normal (the side cloth stays on).
        n: Vec3,
    },
}

impl ClothCollider {
    /// The collider in the space `frame` is expressed in, scaled by `scale`
    /// for the radii. `frame` is the bone's rigid frame in that space.
    pub fn shape_in(&self, frame: Mat4, scale: f32) -> ClothShapeWorld {
        let c = frame.transform_point3(Vec3::from(self.center));
        match self.shape {
            ClothShape::Sphere => ClothShapeWorld::Sphere { c, r: self.radius.max(0.0) * scale },
            ClothShape::Capsule => ClothShapeWorld::Capsule {
                a: c,
                b: frame.transform_point3(Vec3::from(self.end)),
                ra: self.radius.max(0.0) * scale,
                rb: self.end_radius.max(0.0) * scale,
            },
            ClothShape::Plane => ClothShapeWorld::Plane {
                p: c,
                n: frame.transform_vector3(Vec3::from(self.normal)).normalize_or(Vec3::Y),
            },
        }
    }

    /// The collider in world space: `frame` is the bone's object-space rigid
    /// frame, `world` the object's world matrix.
    pub fn world_shape(&self, frame: Mat4, world: Mat4) -> ClothShapeWorld {
        self.shape_in(world * frame, max_scale(world))
    }
}

/// Largest axis scale of a matrix (radii scale by it).
pub fn max_scale(m: Mat4) -> f32 {
    let s = Vec3::new(m.x_axis.truncate().length(), m.y_axis.truncate().length(), m.z_axis.truncate().length());
    s.max_element().max(1e-6)
}

/// Translation + rotation of `m` with its scale removed.
pub fn rigid_frame(m: Mat4) -> Mat4 {
    let (_, r, t) = m.to_scale_rotation_translation();
    Mat4::from_rotation_translation(r.normalize(), t)
}

/// Which bones auto-fit wraps in capsules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ClothFitPreset {
    /// Hips, legs and arms.
    #[default]
    Body,
    /// Legs only.
    Legs,
    /// Arms only.
    Arms,
    /// The bones listed in `ClothFit::bones`.
    Selected,
}

impl ClothFitPreset {
    /// Every preset, in declaration order.
    pub const ALL: [ClothFitPreset; 4] =
        [ClothFitPreset::Body, ClothFitPreset::Legs, ClothFitPreset::Arms, ClothFitPreset::Selected];
    /// Inspector label of this preset.
    pub fn label(self) -> &'static str {
        match self {
            ClothFitPreset::Body => "Body (hips, legs, arms)",
            ClothFitPreset::Legs => "Legs",
            ClothFitPreset::Arms => "Arms",
            ClothFitPreset::Selected => "Selected bones",
        }
    }
}

/// Auto-fit settings: which bones get a capsule and how snugly it wraps the
/// vertices skinned to them.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ClothFit {
    /// Which bones to fit.
    pub preset: ClothFitPreset,
    /// The bones to fit when `preset` is Selected.
    pub bones: Vec<String>,
    /// Fraction of a bone's vertices the capsule encloses (0.5 to 1). Lower
    /// ignores stray vertices; 1 wraps everything.
    pub coverage: f32,
    /// Added to every fitted radius (negative shrinks the capsule into the skin).
    pub inset: f32,
}

impl Default for ClothFit {
    fn default() -> Self {
        Self { preset: ClothFitPreset::Body, bones: Vec::new(), coverage: 0.85, inset: 0.0 }
    }
}

/// Engine-filled and inspector-set state of a [`ClothColliders`], not saved.
#[derive(Default)]
pub struct ClothColliderRuntime {
    /// Joint names of the object's skeleton.
    pub bones: Vec<String>,
    /// Object-space rigid frame of each of `bones` in this frame's pose.
    pub frames: Vec<Mat4>,
    /// Set by the inspector; the engine runs auto-fit and clears it.
    pub fit_requested: bool,
    /// The last auto-fit result, shown in the inspector.
    pub fit_report: String,
    /// Collider being edited (gets viewport handles).
    pub selected: Option<usize>,
}

impl ClothColliderRuntime {
    /// Object-space frame of `bone` (identity for the object itself).
    pub fn frame(&self, bone: &str) -> Option<Mat4> {
        if bone.is_empty() {
            return Some(Mat4::IDENTITY);
        }
        let i = self.bones.iter().position(|b| b == bone)?;
        self.frames.get(i).copied()
    }
}

/// Colliders that bone cloth on this object (or on objects listing this one)
/// pushes against. On a rigged model each collider follows a joint.
#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ClothColliders {
    /// The colliders.
    pub colliders: Vec<ClothCollider>,
    /// Auto-fit settings.
    pub fit: ClothFit,
    /// Engine-filled state, not saved.
    #[serde(skip)]
    pub runtime: ClothColliderRuntime,
}

impl TypedComponent for ClothColliders {
    const NAME: &'static str = "Cloth Colliders";
}

impl ClothColliders {
    /// World-space shapes of the enabled colliders whose name passes `filter`
    /// (empty filter = all).
    pub fn world_shapes(&self, world: Mat4, filter: &[String]) -> Vec<ClothShapeWorld> {
        self.colliders
            .iter()
            .filter(|c| c.enabled && (filter.is_empty() || filter.contains(&c.name)))
            .filter_map(|c| Some(c.world_shape(self.runtime.frame(&c.bone)?, world)))
            .collect()
    }
}

/// How neighbouring chains of a [`BoneCloth`] link together.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ClothConnection {
    /// Independent chains (hair strands, tails).
    #[default]
    Chains,
    /// Neighbouring chains linked, last back to first (skirts, capes that wrap).
    Loop,
    /// Neighbouring chains linked, ends open (capes, flaps).
    Open,
}

impl ClothConnection {
    /// Every connection, in declaration order.
    pub const ALL: [ClothConnection; 3] = [ClothConnection::Chains, ClothConnection::Loop, ClothConnection::Open];
    /// Inspector label of this connection.
    pub fn label(self) -> &'static str {
        match self {
            ClothConnection::Chains => "Chains",
            ClothConnection::Loop => "Linked loop",
            ClothConnection::Open => "Linked open",
        }
    }
}

/// What a teleport (a long jump / sharp turn in one frame) does to the cloth.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ClothTeleport {
    /// Snap the cloth back to the animated pose.
    #[default]
    Reset,
    /// Carry the cloth along with its current shape.
    Keep,
}

impl ClothTeleport {
    /// Every behaviour, in declaration order.
    pub const ALL: [ClothTeleport; 2] = [ClothTeleport::Reset, ClothTeleport::Keep];
    /// Inspector label of this behaviour.
    pub fn label(self) -> &'static str {
        match self {
            ClothTeleport::Reset => "Reset",
            ClothTeleport::Keep => "Keep shape",
        }
    }
}

/// Engine-filled state of a [`BoneCloth`], not saved.
#[derive(Default)]
pub struct BoneClothRuntime {
    /// Joint names of the object's skeleton.
    pub bones: Vec<String>,
    /// Names of the Cloth Colliders on this object.
    pub colliders: Vec<String>,
    /// The running simulation, not saved.
    pub sim: Option<BoneClothSim>,
    /// Set by the inspector; the engine rebuilds the simulation.
    pub reset_requested: bool,
    /// Beyond the cull distance from the viewer this frame.
    pub culled: bool,
}

/// Simulates every joint below each root as a chain of particles and turns
/// the result back into joint rotations. Values given as `[root, tip]` are
/// blended along each chain.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct BoneCloth {
    /// Joints the chains start from.
    pub roots: Vec<String>,
    /// How neighbouring chains link.
    pub connection: ClothConnection,
    /// World-space acceleration (m/s^2).
    pub gravity: [f32; 3],
    /// Velocity lost per 1/60 s (0 to 1).
    pub damping: f32,
    /// Pull back toward the animated pose per 1/90 s, root to tip (0 to 1).
    pub stiffness: [f32; 2],
    /// Max swing away from the animated pose in degrees (0 = no limit).
    pub angle_limit: f32,
    /// How firmly linked chains keep their spacing (0 to 1).
    pub link_stiffness: f32,
    /// Particle collision radius, root to tip (object units).
    pub radius: [f32; 2],
    /// Tangential velocity lost on contact (0 to 1).
    pub friction: f32,
    /// Particles collide with Cloth Colliders.
    pub collide: bool,
    /// Also collide the segments between particles, so colliders can't slip
    /// between them.
    pub edge_collision: bool,
    /// Collider names on this object to use (empty = all).
    pub colliders: Vec<String>,
    /// Other objects whose Cloth Colliders this cloth also collides with.
    pub collider_objects: Vec<ObjectRef>,
    /// How much the cloth resists the object's own movement: 1 = fully
    /// physical, 0 = moves rigidly with the object.
    pub world_inertia: f32,
    /// Object speed past which the extra movement is carried rigidly (m/s,
    /// 0 = no limit), so locomotion and snap turns don't whip the cloth.
    pub max_speed: f32,
    /// A jump farther than this in one frame is a teleport (m, 0 = never).
    pub teleport_distance: f32,
    /// A turn sharper than this in one frame is a teleport (degrees, 0 = never).
    pub teleport_angle: f32,
    /// What a teleport does to the cloth.
    pub teleport: ClothTeleport,
    /// Simulation steps per second.
    pub frequency: f32,
    /// Constraint passes per step.
    pub iterations: u32,
    /// Add a virtual particle past each chain's last joint so it swings too.
    pub end_tail: bool,
    /// Mix of simulated (1) and animated (0) rotations.
    pub blend: f32,
    /// How strongly Wind Zones push this cloth (0 = ignores wind).
    pub wind: f32,
    /// Stop simulating (keep the animated pose) beyond this distance from the
    /// camera (m, 0 = never).
    pub cull_distance: f32,
    /// Engine-filled state, not saved.
    #[serde(skip)]
    pub runtime: BoneClothRuntime,
}

impl Default for BoneCloth {
    fn default() -> Self {
        Self {
            roots: Vec::new(),
            connection: ClothConnection::Chains,
            gravity: [0.0, -9.81, 0.0],
            damping: 0.08,
            stiffness: [0.2, 0.04],
            angle_limit: 0.0,
            link_stiffness: 0.5,
            radius: [0.02, 0.02],
            friction: 0.3,
            collide: true,
            edge_collision: true,
            colliders: Vec::new(),
            collider_objects: Vec::new(),
            world_inertia: 1.0,
            max_speed: 4.0,
            teleport_distance: 1.0,
            teleport_angle: 90.0,
            teleport: ClothTeleport::Reset,
            frequency: 90.0,
            iterations: 3,
            end_tail: true,
            blend: 1.0,
            wind: 1.0,
            cull_distance: 0.0,
            runtime: BoneClothRuntime::default(),
        }
    }
}

impl TypedComponent for BoneCloth {
    const NAME: &'static str = "Bone Cloth";
}

/// Where a [`ClothWind`] blows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum WindShape {
    /// Everywhere.
    #[default]
    Global,
    /// Inside `radius`, fading out toward the edge.
    Sphere,
    /// Inside the `size` box.
    Box,
}

impl WindShape {
    /// Every shape, in declaration order.
    pub const ALL: [WindShape; 3] = [WindShape::Global, WindShape::Sphere, WindShape::Box];
    /// Inspector label of this shape.
    pub fn label(self) -> &'static str {
        match self {
            WindShape::Global => "Global",
            WindShape::Sphere => "Sphere",
            WindShape::Box => "Box",
        }
    }
}

/// Wind for cloth, blowing along the object's forward (-Z).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ClothWind {
    /// Where the wind blows.
    pub shape: WindShape,
    /// Wind speed (m/s).
    pub strength: f32,
    /// Gust and direction wobble (0 = steady).
    pub turbulence: f32,
    /// Gusts per second.
    /// Gusts per second.
    pub frequency: f32,
    /// Sphere reach (m).
    pub radius: f32,
    /// Box extents (m).
    pub size: [f32; 3],
}

impl Default for ClothWind {
    fn default() -> Self {
        Self { shape: WindShape::Global, strength: 3.0, turbulence: 0.5, frequency: 0.4, radius: 5.0, size: [10.0, 10.0, 10.0] }
    }
}

impl TypedComponent for ClothWind {
    const NAME: &'static str = "Wind Zone";
}

/// A Wind Zone resolved into world space for the simulation.
#[derive(Clone, Copy, Debug)]
pub struct WindSample {
    /// The zone's resolved parameters.
    pub wind: ClothWindParams,
    /// The zone's world matrix and its inverse.
    pub world: Mat4,
    /// Inverse of `world` (local-space tests).
    pub inv: Mat4,
}

/// The resolved parameters of a [`ClothWind`], in world space.
#[derive(Clone, Copy, Debug)]
pub struct ClothWindParams {
    /// Where the wind blows.
    pub shape: WindShape,
    /// Wind speed (m/s).
    pub strength: f32,
    /// Gust and direction wobble (0 = steady).
    pub turbulence: f32,
    /// Gusts per second.
    pub frequency: f32,
    /// Sphere reach (m).
    pub radius: f32,
    /// Box extents (m).
    pub size: Vec3,
}

impl WindSample {
    /// Resolve a Wind Zone into world space.
    pub fn new(zone: &ClothWind, world: Mat4) -> Self {
        Self {
            wind: ClothWindParams {
                shape: zone.shape,
                strength: zone.strength,
                turbulence: zone.turbulence.clamp(0.0, 1.0),
                frequency: zone.frequency.max(0.0),
                radius: zone.radius.max(1e-3),
                size: Vec3::from(zone.size),
            },
            world,
            inv: world.inverse(),
        }
    }

    /// Wind velocity at world `p` at time `t`.
    pub fn at(&self, p: Vec3, t: f32) -> Vec3 {
        let w = &self.wind;
        let local = self.inv.transform_point3(p);
        let reach = match w.shape {
            WindShape::Global => 1.0,
            WindShape::Sphere => (1.0 - local.length() / w.radius).clamp(0.0, 1.0).sqrt(),
            WindShape::Box => {
                let inside = (local.abs() - w.size * 0.5).max_element() <= 0.0;
                if inside { 1.0 } else { 0.0 }
            }
        };
        if reach <= 0.0 {
            return Vec3::ZERO;
        }
        let dir = self.world.transform_vector3(Vec3::NEG_Z).normalize_or(Vec3::NEG_Z);
        let tau = std::f32::consts::TAU;
        let f = w.frequency;
        let gust = (t * f * tau + p.x * 0.7).sin() * 0.5 + (t * f * 2.13 * tau + p.z * 1.3).sin() * 0.3 + (t * f * 3.71 * tau + p.y * 0.9).sin() * 0.2;
        let sway = (t * f * 1.37 * tau + p.y * 1.1).sin();
        let side = dir.cross(Vec3::Y).normalize_or(Vec3::X);
        (dir * (1.0 + w.turbulence * gust) + side * (w.turbulence * 0.3 * sway)) * (w.strength * reach)
    }
}

/// One simulated point: a joint, or the virtual tail past a chain's last joint.
#[derive(Clone, Debug)]
pub struct ClothParticle {
    /// Joint index (None = virtual tail).
    pub joint: Option<usize>,
    /// Particle index of the parent (None = a root).
    pub parent: Option<usize>,
    /// Root-fixed (never moves from the animated pose).
    pub fixed: bool,
    /// World position.
    pub pos: Vec3,
    /// Current velocity (m/s).
    pub vel: Vec3,
    /// 0 at the root, 1 at the chain's tip.
    pub depth: f32,
    /// World collision radius this step.
    pub radius: f32,
    /// Tail offset in the parent joint's rigid frame (tails only).
    tail: Vec3,
    /// Animated world position at the end of the previous frame.
    prev_anim: Vec3,
    contact: Option<Vec3>,
}

/// The running simulation of one [`BoneCloth`].
#[derive(Clone, Debug)]
pub struct BoneClothSim {
    /// Parents before children.
    pub particles: Vec<ClothParticle>,
    /// Links between neighbouring chains (particle indices).
    pub links: Vec<(usize, usize)>,
    first_child: Vec<Option<usize>>,
    signature: Signature,
    last_world: Mat4,
}

#[derive(Clone, Debug, PartialEq)]
struct Signature {
    roots: Vec<String>,
    connection: ClothConnection,
    end_tail: bool,
    joints: usize,
}

fn signature(cloth: &BoneCloth, joints: usize) -> Signature {
    Signature { roots: cloth.roots.clone(), connection: cloth.connection, end_tail: cloth.end_tail, joints }
}

fn translation(m: &Mat4) -> Vec3 {
    m.w_axis.truncate()
}

fn rotation(m: &Mat4) -> Quat {
    let (_, r, _) = m.to_scale_rotation_translation();
    r.normalize()
}

impl BoneClothSim {
    /// Whether the simulation was built for this cloth setup.
    pub fn matches(&self, cloth: &BoneCloth, joints: usize) -> bool {
        self.signature == signature(cloth, joints)
    }

    /// Build the particle topology below `cloth.roots`. `parents` and `names`
    /// describe the skeleton, `models` is the animated object-space pose and
    /// `world` the object's world matrix. `None` when no root resolves.
    pub fn build(cloth: &BoneCloth, parents: &[Option<usize>], names: &[String], models: &[Mat4], world: Mat4) -> Option<Self> {
        let n = parents.len().min(names.len()).min(models.len());
        let mut children = vec![Vec::new(); n];
        for (j, p) in parents.iter().enumerate().take(n) {
            if let Some(p) = *p
                && p < n
            {
                children[p].push(j);
            }
        }
        let mut particles: Vec<ClothParticle> = Vec::new();
        let mut of_joint: Vec<Option<usize>> = vec![None; n];
        let mut chains: Vec<Vec<usize>> = Vec::new();
        for root_name in &cloth.roots {
            let Some(root) = names.iter().take(n).position(|b| b == root_name) else {
                continue;
            };
            if of_joint[root].is_some() {
                continue;
            }
            let first = particles.len();
            let mut queue = std::collections::VecDeque::from([(root, None::<usize>, 0u32)]);
            let mut max_depth = 1u32;
            let mut depths = Vec::new();
            while let Some((j, parent, d)) = queue.pop_front() {
                if of_joint[j].is_some() {
                    continue;
                }
                let idx = particles.len();
                of_joint[j] = Some(idx);
                particles.push(ClothParticle {
                    joint: Some(j),
                    parent,
                    fixed: parent.is_none(),
                    pos: Vec3::ZERO,
                    vel: Vec3::ZERO,
                    depth: 0.0,
                    radius: 0.0,
                    tail: Vec3::ZERO,
                    prev_anim: Vec3::ZERO,
                    contact: None,
                });
                depths.push(d);
                max_depth = max_depth.max(d);
                for &c in &children[j] {
                    queue.push_back((c, Some(idx), d + 1));
                }
                if children[j].is_empty()
                    && cloth.end_tail
                    && let Some(pj) = parents[j]
                    && pj < n
                {
                    let seg = translation(&models[j]) - translation(&models[pj]);
                    if seg.length_squared() > 1e-10 {
                        particles.push(ClothParticle {
                            joint: None,
                            parent: Some(idx),
                            fixed: false,
                            pos: Vec3::ZERO,
                            vel: Vec3::ZERO,
                            depth: 0.0,
                            radius: 0.0,
                            tail: rotation(&models[j]).inverse() * seg,
                            prev_anim: Vec3::ZERO,
                            contact: None,
                        });
                        depths.push(d + 1);
                        max_depth = max_depth.max(d + 1);
                    }
                }
            }
            for (k, d) in depths.iter().enumerate() {
                particles[first + k].depth = *d as f32 / max_depth as f32;
            }
            chains.extend(root_chains(&particles, first, root, &children, models, cloth.connection));
        }
        if particles.len() < 2 {
            return None;
        }
        let mut first_child = vec![None; particles.len()];
        for i in (0..particles.len()).rev() {
            if let Some(p) = particles[i].parent {
                first_child[p] = Some(i);
            }
        }
        let mut links = Vec::new();
        if cloth.connection != ClothConnection::Chains && chains.len() >= 2 {
            let count = chains.len();
            let pairs = if cloth.connection == ClothConnection::Loop && count >= 3 { count } else { count - 1 };
            for c in 0..pairs {
                let (a, b) = (&chains[c], &chains[(c + 1) % count]);
                for k in 0..a.len().min(b.len()) {
                    let (pa, pb) = (a[k], b[k]);
                    if !(particles[pa].fixed && particles[pb].fixed) && pa != pb {
                        links.push((pa, pb));
                    }
                }
            }
        }
        let mut sim = Self { particles, links, first_child, signature: signature(cloth, n), last_world: world };
        let anim = sim.anim_positions(models, world);
        for (p, a) in sim.particles.iter_mut().zip(&anim) {
            p.pos = *a;
            p.prev_anim = *a;
        }
        Some(sim)
    }

    /// Animated world position of every particle.
    fn anim_positions(&self, models: &[Mat4], world: Mat4) -> Vec<Vec3> {
        self.particles
            .iter()
            .map(|p| match p.joint {
                Some(j) => world.transform_point3(translation(&models[j])),
                None => {
                    let pj = self.particles[p.parent.unwrap_or(0)].joint.unwrap_or(0);
                    let m = &models[pj];
                    world.transform_point3(translation(m) + rotation(m) * p.tail)
                }
            })
            .collect()
    }

    /// Snap every particle to the animated pose and stop it.
    pub fn reset(&mut self, models: &[Mat4], world: Mat4) {
        let anim = self.anim_positions(models, world);
        for (p, a) in self.particles.iter_mut().zip(&anim) {
            p.pos = *a;
            p.prev_anim = *a;
            p.vel = Vec3::ZERO;
            p.contact = None;
        }
        self.last_world = world;
    }

    /// Advance by `dt` seconds. `models` is this frame's animated object-space
    /// pose, `world` the object's world matrix and `colliders` world-space.
    pub fn step(&mut self, cloth: &BoneCloth, models: &[Mat4], world: Mat4, colliders: &[ClothShapeWorld], dt: f32) {
        self.step_in_wind(cloth, models, world, colliders, &[], 0.0, dt);
    }

    /// [`Self::step`] with wind from `winds` at time `time`.
    #[allow(clippy::too_many_arguments)]
    pub fn step_in_wind(
        &mut self,
        cloth: &BoneCloth,
        models: &[Mat4],
        world: Mat4,
        colliders: &[ClothShapeWorld],
        winds: &[WindSample],
        time: f32,
        dt: f32,
    ) {
        if dt <= 0.0 || self.particles.iter().any(|p| p.joint.is_some_and(|j| j >= models.len())) {
            return;
        }
        let dt = dt.min(0.1);
        let anim = self.anim_positions(models, world);
        let scale = max_scale(world);

        let moved = translation(&world) - translation(&self.last_world);
        let turned = rotation(&self.last_world).angle_between(rotation(&world)).to_degrees();
        let teleported = (cloth.teleport_distance > 0.0 && moved.length() > cloth.teleport_distance)
            || (cloth.teleport_angle > 0.0 && turned > cloth.teleport_angle);
        let last_inv = self.last_world.inverse();
        if teleported {
            match cloth.teleport {
                ClothTeleport::Reset => {
                    self.reset(models, world);
                    return;
                }
                ClothTeleport::Keep => {
                    let delta = world * last_inv;
                    let rot = rotation(&delta);
                    for p in &mut self.particles {
                        p.pos = delta.transform_point3(p.pos);
                        p.vel = rot * p.vel;
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
                for p in self.particles.iter_mut().filter(|p| !p.fixed) {
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
        let stiff = |depth: f32| {
            let s = (cloth.stiffness[0] + (cloth.stiffness[1] - cloth.stiffness[0]) * depth).clamp(0.0, 1.0);
            1.0 - (1.0 - s).powf(h * 90.0)
        };
        let link_k = cloth.link_stiffness.clamp(0.0, 1.0);
        let limit = cloth.angle_limit.to_radians();
        for p in &mut self.particles {
            p.radius = (cloth.radius[0] + (cloth.radius[1] - cloth.radius[0]) * p.depth).max(0.0) * scale;
        }
        let shapes: &[ClothShapeWorld] = if cloth.collide { colliders } else { &[] };
        let mut target = vec![Vec3::ZERO; self.particles.len()];
        let mut old = vec![Vec3::ZERO; self.particles.len()];

        for k in 0..steps {
            let f = (k + 1) as f32 / steps as f32;
            for (i, p) in self.particles.iter_mut().enumerate() {
                target[i] = p.prev_anim.lerp(anim[i], f);
                old[i] = p.pos;
                p.contact = None;
                if p.fixed {
                    p.pos = target[i];
                    p.vel = Vec3::ZERO;
                } else {
                    p.vel = (p.vel + gravity * h) * keep;
                    if !winds.is_empty() && cloth.wind > 0.0 {
                        let w: Vec3 = winds.iter().map(|z| z.at(p.pos, time + k as f32 * h)).sum();
                        p.vel += (w - p.vel) * (1.0 - (-1.5 * cloth.wind * h).exp());
                    }
                    p.pos += p.vel * h;
                }
            }
            for it in 0..cloth.iterations.max(1) {
                for i in 0..self.particles.len() {
                    let Some(pi) = self.particles[i].parent else { continue };
                    if self.particles[i].fixed {
                        continue;
                    }
                    let rest_vec = target[i] - target[pi];
                    let rest = rest_vec.length();
                    let dir = rest_vec.normalize_or(Vec3::NEG_Y);
                    let parent = self.particles[pi].pos;
                    let p = &mut self.particles[i];
                    if it == 0 {
                        let s = stiff(p.depth);
                        if s > 0.0 {
                            p.pos += (parent + dir * rest - p.pos) * s;
                        }
                    }
                    if limit > 0.0 {
                        let v = p.pos - parent;
                        let len = v.length();
                        if len > 1e-6 && dir.angle_between(v / len) > limit {
                            let axis = dir.cross(v).normalize_or(dir.any_orthonormal_vector());
                            p.pos = parent + Quat::from_axis_angle(axis, limit) * dir * len;
                        }
                    }
                }
                for &(a, b) in &self.links {
                    let rest = (target[b] - target[a]).length();
                    let d = self.particles[b].pos - self.particles[a].pos;
                    let len = d.length();
                    if len < 1e-6 {
                        continue;
                    }
                    let diff = len - rest;
                    let k = if diff > 0.0 { link_k } else { link_k * 0.5 };
                    let (wa, wb) = weights(self.particles[a].fixed, self.particles[b].fixed);
                    let corr = d / len * diff * k;
                    self.particles[a].pos += corr * wa;
                    self.particles[b].pos -= corr * wb;
                }
                for i in 0..self.particles.len() {
                    let Some(pi) = self.particles[i].parent else { continue };
                    if self.particles[i].fixed {
                        continue;
                    }
                    let rest = (target[i] - target[pi]).length();
                    let parent = self.particles[pi].pos;
                    let p = &mut self.particles[i];
                    let fallback = (target[i] - target[pi]).normalize_or(Vec3::NEG_Y);
                    p.pos = parent + (p.pos - parent).normalize_or(fallback) * rest;
                }
                if !shapes.is_empty() {
                    self.collide_particles(shapes);
                    if cloth.edge_collision {
                        self.collide_edges(shapes);
                    }
                }
            }
            let friction = cloth.friction.clamp(0.0, 1.0);
            for (i, p) in self.particles.iter_mut().enumerate() {
                if p.fixed {
                    continue;
                }
                let mut v = (p.pos - old[i]) / h;
                if let Some(n) = p.contact {
                    let vn = v.dot(n);
                    let tangent = v - n * vn;
                    v = n * vn.max(0.0) + tangent * (1.0 - friction);
                }
                p.vel = v.clamp_length_max(30.0);
            }
        }
        for (p, a) in self.particles.iter_mut().zip(anim) {
            p.prev_anim = a;
        }
    }

    fn collide_particles(&mut self, shapes: &[ClothShapeWorld]) {
        for p in self.particles.iter_mut().filter(|p| !p.fixed) {
            for s in shapes {
                if let Some((push, n)) = penetration(p.pos, p.radius, s) {
                    p.pos += push;
                    p.contact = Some(n);
                }
            }
        }
    }

    fn collide_edges(&mut self, shapes: &[ClothShapeWorld]) {
        let parent_edges = self.particles.iter().enumerate().filter_map(|(i, p)| p.parent.map(|pi| (pi, i)));
        let edges: Vec<(usize, usize)> = parent_edges.chain(self.links.iter().copied()).collect();
        for (a, b) in edges {
            let (fa, fb) = (self.particles[a].fixed, self.particles[b].fixed);
            if fa && fb {
                continue;
            }
            for s in shapes {
                let (pa, pb) = (self.particles[a].pos, self.particles[b].pos);
                let (ra, rb) = (self.particles[a].radius, self.particles[b].radius);
                let Some((t, push, n)) = edge_penetration(pa, pb, ra, rb, s) else { continue };
                if t <= 0.02 || t >= 0.98 {
                    continue;
                }
                let wa = if fa { 0.0 } else { 1.0 - t };
                let wb = if fb { 0.0 } else { t };
                let denom = wa * (1.0 - t) + wb * t;
                if denom < 1e-4 {
                    continue;
                }
                self.particles[a].pos += push * (wa / denom);
                self.particles[b].pos += push * (wb / denom);
                if !fa {
                    self.particles[a].contact = Some(n);
                }
                if !fb {
                    self.particles[b].contact = Some(n);
                }
            }
        }
    }

    /// Turn the particles into joint rotations: each simulated joint is
    /// rotated so its child lands on the child particle. `locals` (this
    /// frame's animated locals) are rewritten in place. `blend` mixes the
    /// simulated rotation with the animated one.
    pub fn write_pose(&self, parents: &[Option<usize>], models: &[Mat4], locals: &mut [Mat4], world: Mat4, blend: f32) {
        let n = locals.len().min(models.len()).min(parents.len());
        if self.particles.iter().any(|p| p.joint.is_some_and(|j| j >= n)) {
            return;
        }
        let inv_world = world.inverse();
        let blend = blend.clamp(0.0, 1.0);
        let mut posed: Vec<Option<Mat4>> = vec![None; n];
        for (i, p) in self.particles.iter().enumerate() {
            let Some(j) = p.joint else { continue };
            let parent = parents[j].filter(|&pj| pj < n).map(|pj| posed[pj].unwrap_or(models[pj]));
            let mut m = match parent {
                Some(pm) => pm * locals[j],
                None => locals[j],
            };
            if let Some(c) = self.first_child[i] {
                let origin = translation(&m);
                let child = &self.particles[c];
                let current = match child.joint {
                    Some(cj) => translation(&(m * locals[cj])),
                    None => origin + rotation(&m) * child.tail,
                };
                let a = current - origin;
                let b = inv_world.transform_point3(child.pos) - origin;
                if a.length_squared() > 1e-12 && b.length_squared() > 1e-12 {
                    let q = Quat::IDENTITY.slerp(Quat::from_rotation_arc(a.normalize(), b.normalize()), blend);
                    m = Mat4::from_translation(origin) * Mat4::from_quat(q) * Mat4::from_translation(-origin) * m;
                }
            }
            posed[j] = Some(m);
            locals[j] = match parent {
                Some(pm) => pm.inverse() * m,
                None => m,
            };
        }
    }
}

/// The chains a root contributes, as particle indices from root to tip. A root
/// with several children (a skirt ring) yields one chain per child, sorted
/// around the root so neighbours link; otherwise the root starts one chain.
fn root_chains(
    particles: &[ClothParticle],
    first: usize,
    root: usize,
    children: &[Vec<usize>],
    models: &[Mat4],
    connection: ClothConnection,
) -> Vec<Vec<usize>> {
    let follow = |start: usize| {
        let mut chain = vec![start];
        let mut cur = start;
        while let Some(next) = (cur + 1..particles.len()).find(|&k| particles[k].parent == Some(cur)) {
            chain.push(next);
            cur = next;
        }
        chain
    };
    let direct: Vec<usize> = (first + 1..particles.len()).filter(|&k| particles[k].parent == Some(first)).collect();
    if connection == ClothConnection::Chains || direct.len() < 2 || children[root].len() < 2 {
        return vec![follow(first)];
    }
    let origin = translation(&models[root]);
    let tip = |k: usize| {
        let chain = follow(k);
        let last = chain.iter().rev().find_map(|&i| particles[i].joint).unwrap_or(root);
        translation(&models[last])
    };
    let tips: Vec<Vec3> = direct.iter().map(|&k| tip(k)).collect();
    let axis = (tips.iter().copied().sum::<Vec3>() / tips.len() as f32 - origin).normalize_or(Vec3::NEG_Y);
    let (u, v) = axis.any_orthonormal_pair();
    let mut sorted: Vec<(f32, usize)> = direct
        .iter()
        .zip(&tips)
        .map(|(&k, t)| {
            let d = *t - origin;
            (d.dot(v).atan2(d.dot(u)), k)
        })
        .collect();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
    sorted
        .into_iter()
        .map(|(_, k)| {
            let mut chain = vec![first];
            chain.extend(follow(k));
            chain
        })
        .collect()
}

fn weights(fixed_a: bool, fixed_b: bool) -> (f32, f32) {
    match (fixed_a, fixed_b) {
        (true, true) => (0.0, 0.0),
        (true, false) => (0.0, 1.0),
        (false, true) => (1.0, 0.0),
        (false, false) => (0.5, 0.5),
    }
}

fn closest_on_segment(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let len2 = ab.length_squared();
    if len2 < 1e-12 { 0.0 } else { ((p - a).dot(ab) / len2).clamp(0.0, 1.0) }
}

/// Push that moves a sphere at `p` of radius `r` out of `shape`, and the
/// contact normal.
pub fn penetration(p: Vec3, r: f32, shape: &ClothShapeWorld) -> Option<(Vec3, Vec3)> {
    let out = |c: Vec3, reach: f32| {
        let d = p - c;
        let len = d.length();
        if len >= reach {
            return None;
        }
        let n = if len > 1e-6 { d / len } else { Vec3::Y };
        Some((n * (reach - len), n))
    };
    match *shape {
        ClothShapeWorld::Sphere { c, r: sr } => out(c, sr + r),
        ClothShapeWorld::Capsule { a, b, ra, rb } => {
            let t = closest_on_segment(p, a, b);
            out(a.lerp(b, t), ra + (rb - ra) * t + r)
        }
        ClothShapeWorld::Plane { p: o, n } => {
            let d = (p - o).dot(n) - r;
            (d < 0.0).then_some((-n * d, n))
        }
    }
}

/// Closest points between segments `p1q1` and `p2q2` as parameters (s, t).
fn segment_segment(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (f32, f32) {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);
    if a < 1e-12 && e < 1e-12 {
        return (0.0, 0.0);
    }
    if a < 1e-12 {
        return (0.0, (f / e).clamp(0.0, 1.0));
    }
    let c = d1.dot(r);
    if e < 1e-12 {
        return ((-c / a).clamp(0.0, 1.0), 0.0);
    }
    let b = d1.dot(d2);
    let denom = a * e - b * b;
    let mut s = if denom > 1e-12 { ((b * f - c * e) / denom).clamp(0.0, 1.0) } else { 0.0 };
    let mut t = (b * s + f) / e;
    if t < 0.0 {
        t = 0.0;
        s = (-c / a).clamp(0.0, 1.0);
    } else if t > 1.0 {
        t = 1.0;
        s = ((b - c) / a).clamp(0.0, 1.0);
    }
    (s, t)
}

/// Penetration of the segment `ab` (radii `ra`, `rb`) into `shape`: the
/// parameter along the segment, the push for that point and the normal.
fn edge_penetration(a: Vec3, b: Vec3, ra: f32, rb: f32, shape: &ClothShapeWorld) -> Option<(f32, Vec3, Vec3)> {
    let (t, center, reach) = match *shape {
        ClothShapeWorld::Sphere { c, r } => {
            let t = closest_on_segment(c, a, b);
            (t, c, r)
        }
        ClothShapeWorld::Capsule { a: ca, b: cb, ra: cra, rb: crb } => {
            let (t, u) = segment_segment(a, b, ca, cb);
            (t, ca.lerp(cb, u), cra + (crb - cra) * u)
        }
        ClothShapeWorld::Plane { .. } => return None,
    };
    let p = a.lerp(b, t);
    let reach = reach + ra + (rb - ra) * t;
    let d = p - center;
    let len = d.length();
    if len >= reach || len < 1e-6 {
        return None;
    }
    let n = d / len;
    Some((t, n * (reach - len), n))
}

/// Principal axis of a point set (unit), by power iteration on its covariance.
pub fn principal_axis(points: &[Vec3]) -> Option<(Vec3, Vec3)> {
    if points.len() < 2 {
        return None;
    }
    let mean = points.iter().copied().sum::<Vec3>() / points.len() as f32;
    let mut cov = Mat3::ZERO;
    for p in points {
        let d = *p - mean;
        cov += Mat3::from_cols(d * d.x, d * d.y, d * d.z);
    }
    let mut v = Vec3::new(0.577, 0.577, 0.577);
    for _ in 0..32 {
        let next = cov * v;
        if next.length_squared() < 1e-20 {
            return None;
        }
        v = next.normalize();
    }
    Some((mean, v))
}

/// Fit a tapered capsule to `points` along the segment `start`-`end`: radii
/// are the `coverage` quantile of the distances to the axis near each end.
/// Returns `(start, end, start_radius, end_radius)` or `None` with too few points.
pub fn fit_capsule(points: &[Vec3], start: Vec3, end: Vec3, coverage: f32) -> Option<(Vec3, Vec3, f32, f32)> {
    let axis = end - start;
    let len = axis.length();
    if len < 1e-5 || points.len() < 6 {
        return None;
    }
    let dir = axis / len;
    let (mut near, mut far, mut all) = (Vec::new(), Vec::new(), Vec::new());
    for p in points {
        let t = (*p - start).dot(dir);
        if t < -0.1 * len || t > 1.1 * len {
            continue;
        }
        let d = (*p - start - dir * t).length();
        all.push(d);
        if t < 0.4 * len {
            near.push(d);
        } else if t > 0.6 * len {
            far.push(d);
        }
    }
    if all.len() < 6 {
        return None;
    }
    let q = |v: &mut Vec<f32>| -> Option<f32> {
        if v.len() < 3 {
            return None;
        }
        v.sort_by(f32::total_cmp);
        let k = ((v.len() - 1) as f32 * coverage.clamp(0.0, 1.0)).round() as usize;
        Some(v[k])
    };
    let overall = q(&mut all)?;
    let rn = q(&mut near).unwrap_or(overall);
    let rf = q(&mut far).unwrap_or(overall);
    // The two quantiles sit at about 20% and 80% along the axis; extend the
    // line through them to the ends, bounded so a stray region can't blow up.
    let reach = (rf - rn) / 3.0;
    let (lo, hi) = (rn.min(rf) * 0.5, overall.max(rn).max(rf) * 1.25);
    Some((start, end, (rn - reach).clamp(lo, hi), (rf + reach).clamp(lo, hi)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A straight chain of `n` joints along +X, one unit apart.
    fn chain(n: usize) -> (Vec<Option<usize>>, Vec<String>, Vec<Mat4>, Vec<Mat4>) {
        let parents: Vec<Option<usize>> = (0..n).map(|i| i.checked_sub(1)).collect();
        let names = (0..n).map(|i| format!("b{i}")).collect();
        let locals: Vec<Mat4> =
            (0..n).map(|i| Mat4::from_translation(if i == 0 { Vec3::ZERO } else { Vec3::X })).collect();
        let mut models = locals.clone();
        for i in 1..n {
            models[i] = models[i - 1] * locals[i];
        }
        (parents, names, locals, models)
    }

    fn cloth(stiffness: f32) -> BoneCloth {
        BoneCloth { roots: vec!["b0".into()], stiffness: [stiffness, stiffness], damping: 0.2, ..Default::default() }
    }

    #[test]
    fn chain_hangs_under_gravity() {
        let (parents, names, locals, models) = chain(4);
        let c = cloth(0.0);
        let mut sim = BoneClothSim::build(&c, &parents, &names, &models, Mat4::IDENTITY).unwrap();
        assert_eq!(sim.particles.len(), 5, "4 joints + a tail");
        for _ in 0..600 {
            sim.step(&c, &models, Mat4::IDENTITY, &[], 1.0 / 60.0);
        }
        let tip = sim.particles.last().unwrap().pos;
        assert!(tip.y < -3.5, "tip should hang below the root, got {tip}");
        assert!(sim.particles[0].pos.length() < 1e-4, "root stays pinned");
        let mut out = locals.clone();
        sim.write_pose(&parents, &models, &mut out, Mat4::IDENTITY, 1.0);
        let posed = {
            let mut m = out.clone();
            for i in 1..m.len() {
                m[i] = m[i - 1] * out[i];
            }
            m
        };
        let last = posed[3].w_axis.truncate();
        assert!(last.y < -2.5 && last.x.abs() < 0.5, "joints follow the particles, got {last}");
    }

    #[test]
    fn full_stiffness_keeps_the_pose() {
        let (parents, names, _, models) = chain(3);
        let c = cloth(1.0);
        let mut sim = BoneClothSim::build(&c, &parents, &names, &models, Mat4::IDENTITY).unwrap();
        for _ in 0..120 {
            sim.step(&c, &models, Mat4::IDENTITY, &[], 1.0 / 60.0);
        }
        let tip = sim.particles[2].pos;
        assert!((tip - Vec3::new(2.0, 0.0, 0.0)).length() < 0.05, "got {tip}");
    }

    #[test]
    fn sphere_collider_pushes_particles_out() {
        let (parents, names, _, models) = chain(4);
        let c = BoneCloth { radius: [0.05, 0.05], ..cloth(0.0) };
        let mut sim = BoneClothSim::build(&c, &parents, &names, &models, Mat4::IDENTITY).unwrap();
        let ball = ClothShapeWorld::Sphere { c: Vec3::new(0.8, -1.6, 0.0), r: 0.6 };
        for _ in 0..600 {
            sim.step(&c, &models, Mat4::IDENTITY, &[ball], 1.0 / 60.0);
        }
        for p in sim.particles.iter().filter(|p| !p.fixed) {
            let d = (p.pos - Vec3::new(0.8, -1.6, 0.0)).length();
            assert!(d >= 0.6 + 0.05 - 0.02, "particle inside the sphere: {d}");
        }
    }

    #[test]
    fn edge_collision_catches_a_thin_capsule_between_particles() {
        let shape = ClothShapeWorld::Capsule { a: Vec3::new(0.5, 0.0, -1.0), b: Vec3::new(0.5, 0.0, 1.0), ra: 0.05, rb: 0.05 };
        let hit = edge_penetration(Vec3::new(0.0, 0.01, 0.0), Vec3::new(1.0, 0.01, 0.0), 0.01, 0.01, &shape);
        let (t, push, n) = hit.expect("segment crosses the capsule");
        assert!((t - 0.5).abs() < 0.01);
        assert!(n.y > 0.9 && push.y > 0.0);
        assert!(penetration(Vec3::new(0.0, 0.01, 0.0), 0.01, &shape).is_none(), "the particles alone miss it");
    }

    #[test]
    fn wind_pushes_a_hanging_chain_downwind() {
        let (parents, names, _, models) = chain(4);
        let c = cloth(0.0);
        let mut sim = BoneClothSim::build(&c, &parents, &names, &models, Mat4::IDENTITY).unwrap();
        let zone = WindSample::new(&ClothWind { strength: 6.0, turbulence: 0.0, ..Default::default() }, Mat4::IDENTITY);
        for k in 0..600 {
            sim.step_in_wind(&c, &models, Mat4::IDENTITY, &[], &[zone], k as f32 / 60.0, 1.0 / 60.0);
        }
        let tip = sim.particles.last().unwrap().pos;
        assert!(tip.z < -1.0, "blown toward -Z: {tip}");
        let sphere = WindSample::new(&ClothWind { shape: WindShape::Sphere, radius: 2.0, ..Default::default() }, Mat4::IDENTITY);
        assert_eq!(sphere.at(Vec3::new(5.0, 0.0, 0.0), 0.0), Vec3::ZERO, "outside the sphere is calm");
    }

    #[test]
    fn teleport_resets_to_the_animated_pose() {
        let (parents, names, _, models) = chain(3);
        let c = cloth(0.0);
        let mut sim = BoneClothSim::build(&c, &parents, &names, &models, Mat4::IDENTITY).unwrap();
        for _ in 0..60 {
            sim.step(&c, &models, Mat4::IDENTITY, &[], 1.0 / 60.0);
        }
        let far = Mat4::from_translation(Vec3::new(50.0, 0.0, 0.0));
        sim.step(&c, &models, far, &[], 1.0 / 60.0);
        let tip = sim.particles[2].pos;
        assert!((tip - Vec3::new(52.0, 0.0, 0.0)).length() < 1e-3, "got {tip}");
    }

    #[test]
    fn zero_inertia_moves_rigidly_with_the_object() {
        let (parents, names, _, models) = chain(3);
        let c = BoneCloth { world_inertia: 0.0, gravity: [0.0; 3], ..cloth(0.0) };
        let mut sim = BoneClothSim::build(&c, &parents, &names, &models, Mat4::IDENTITY).unwrap();
        let moved = Mat4::from_translation(Vec3::new(0.0, 0.0, 0.5));
        sim.step(&c, &models, moved, &[], 1.0 / 60.0);
        let tip = sim.particles[2].pos;
        assert!((tip - Vec3::new(2.0, 0.0, 0.5)).length() < 0.02, "got {tip}");
    }

    #[test]
    fn skirt_ring_links_neighbouring_chains() {
        let mut parents = vec![None];
        let mut names = vec!["root".to_string()];
        let mut locals = vec![Mat4::IDENTITY];
        for k in 0..6 {
            let a = k as f32 / 6.0 * std::f32::consts::TAU;
            parents.push(Some(0));
            names.push(format!("s{k}"));
            locals.push(Mat4::from_translation(Vec3::new(a.cos() * 0.2, -0.1, a.sin() * 0.2)));
            let idx = parents.len() - 1;
            parents.push(Some(idx));
            names.push(format!("s{k}_end"));
            locals.push(Mat4::from_translation(Vec3::new(0.0, -0.3, 0.0)));
        }
        let mut models = locals.clone();
        for i in 1..models.len() {
            models[i] = models[parents[i].unwrap()] * locals[i];
        }
        let c = BoneCloth { roots: vec!["root".into()], connection: ClothConnection::Loop, ..Default::default() };
        let sim = BoneClothSim::build(&c, &parents, &names, &models, Mat4::IDENTITY).unwrap();
        assert_eq!(sim.links.len(), 6 * 3, "each of 6 neighbour pairs links 3 levels (joint, end, tail)");
    }

    #[test]
    fn capsule_fit_tracks_a_tapered_cylinder() {
        let mut pts = Vec::new();
        for i in 0..=20 {
            let t = i as f32 / 20.0;
            let r = 0.1 + (0.05 - 0.1) * t;
            for k in 0..16 {
                let a = k as f32 / 16.0 * std::f32::consts::TAU;
                pts.push(Vec3::new(a.cos() * r, -t * 0.5, a.sin() * r));
            }
        }
        let (_, _, ra, rb) = fit_capsule(&pts, Vec3::ZERO, Vec3::new(0.0, -0.5, 0.0), 0.9).unwrap();
        assert!(ra > rb, "tapers: {ra} {rb}");
        assert!((ra - 0.1).abs() < 0.015 && (rb - 0.05).abs() < 0.015, "{ra} {rb}");
    }

    #[test]
    fn world_shape_follows_the_bone_frame() {
        let col = ClothCollider { shape: ClothShape::Sphere, center: [0.0, 0.1, 0.0], radius: 0.05, ..Default::default() };
        let frame = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0));
        let world = Mat4::from_scale_rotation_translation(Vec3::splat(2.0), Quat::IDENTITY, Vec3::new(0.0, 0.0, 3.0));
        let ClothShapeWorld::Sphere { c, r } = col.world_shape(frame, world) else { panic!() };
        assert!((c - Vec3::new(2.0, 0.2, 3.0)).length() < 1e-5, "{c}");
        assert!((r - 0.1).abs() < 1e-6);
    }
}
