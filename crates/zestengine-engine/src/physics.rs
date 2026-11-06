//! Runtime physics (rapier3d), driven by [`crate::play::PlaySession`].
//!
//! Bodies are keyed by `ObjectId`, so spawning / destroying / disabling objects
//! mid-play just marks the world dirty and [`PhysicsWorld::sync_structure`]
//! adds or removes the affected bodies. The simulation runs at a fixed
//! [`FIXED_DT`]; dynamic bodies are interpolated between the last two steps when
//! written back, so motion stays smooth at any frame rate.
//!
//! Per object: every collider component becomes a collider on one body. The
//! body kind comes from `RigidBody` (no `RigidBody` = fixed). Contact and water
//! response come from `.physmat` physics materials: the collider's, else the
//! body's, else the project default. A
//! `Character` makes a kinematic capsule moved by `move_character`.
//! Kinematic and fixed bodies follow their scene transform; dynamic bodies own
//! theirs, unless gameplay writes a new transform (treated as a teleport).
//!
//! rapier3d 0.33 here uses a glam math backend of a different glam version, so
//! vectors and rotations cross the boundary component-wise.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};

use glam::{Mat4, Quat, Vec3};
use rapier3d::control::{CharacterAutostep, CharacterLength, KinematicCharacterController};
use rapier3d::math::{Pose, Rotation, Vector};
use rapier3d::prelude::*;

use zestengine_core::{
    BodyKind, BoxCollider, Character, CombineMode, JointKind, MeshCollider, ObjectId, PhysicsJoint,
    PhysicsMaterial, PhysicsSettings, RayFilter, RayHit, RigidBody as RbComponent, SphereCollider, CapsuleCollider, Terrain,
    TerrainCollider,
    ColliderAxis,
};

use crate::scene::LoadedScene;

/// Fixed simulation step (seconds).
pub const FIXED_DT: f32 = 1.0 / 60.0;

/// Density (kg/m^3) of a body whose mass comes from its colliders when it
/// has no physics material: floats about half under water.
pub const DEFAULT_DENSITY: f32 = 500.0;

/// Lift on a body never exceeds this many times its weight, so something far
/// lighter than the water it displaces bobs up instead of being launched.
const MAX_LIFT_RATIO: f32 = 6.0;

/// Entry speed (m/s, downward) and time clear of the water before it that
/// make a splash; slower contact (waves lapping, bobbing) doesn't.
const SPLASH_SPEED: f32 = 2.5;
const SPLASH_DRY_SECS: f32 = 0.2;
/// Relative speed (m/s) at the surface before a body sheds a foam wake.
const WAKE_SPEED: f32 = 0.6;

/// Joint velocity-motor gain (rapier's acceleration-based "factor"): high so
/// motors reach their target speed in a few steps; `motor_max_force` caps it.
const MOTOR_GAIN: f32 = 50.0;

/// A collision / trigger contact change between two objects.
#[derive(Clone, Copy, Debug)]
pub struct ContactEvent {
    pub a: ObjectId,
    pub b: ObjectId,
    pub started: bool,
    pub trigger: bool,
}

struct Body {
    handle: RigidBodyHandle,
    kind: BodyKind,
    character: Option<KinematicCharacterController>,
    capsule: Option<ColliderHandle>,
    /// Character capsule as built: component (radius, height, center) and the
    /// object's world scale, so runtime edits (crouch) reshape it.
    capsule_dims: Option<(f32, f32, [f32; 3], Vec3)>,
    /// World pose after the previous / latest step (dynamic interpolation).
    prev: (Vec3, Quat),
    curr: (Vec3, Quat),
    /// Local translation/rotation last written into (or read from) the scene;
    /// a mismatch means gameplay moved the object.
    written: (Vec3, Quat),
    grounded: bool,
    /// The body material's buoyancy and water-drag multipliers.
    water: (f32, f32),
    /// Share of it under water last step.
    wet: f32,
    /// Seconds it has been clear of the water (splashes need a real entry).
    dry_for: f32,
    /// Water a Water Container on it carries: kg and centre (body space).
    carried: (f32, Vec3),
    /// Steps left that a fixed body (a collider without a Rigid Body) still
    /// counts as moving since gameplay or animation last moved it: it then
    /// pushes the water like a body instead of standing in it as a wall.
    moving: u32,
}

/// Steps a moved fixed collider keeps pushing the water before it settles
/// back into a wall.
const MOVED_STEPS: u32 = 20;

/// Advance every interactive Water's ripple window by `dt`, centred near
/// `focus` (world, e.g. the listener) and fed by this step's pushes. Solid
/// colliders through the surface (from `physics`) reflect the ripples, and
/// water flowing past them sheds alternating eddies downstream. Water the
/// GPU simulation owns is only fed: this step's displacers and pushes, and
/// its obstacle cells now and then.
pub fn step_ripples(
    scene: &mut LoadedScene,
    world: &[Mat4],
    focus: Vec3,
    step: &WaterStep,
    physics: Option<&PhysicsWorld>,
    t: f32,
    dt: f32,
) {
    // Each water's layer: what its colliders count against.
    let masks: Vec<u32> = scene.objects.iter().map(|o| scene.layers.collision_mask(o.layer)).collect();
    for (i, o) in scene.objects.iter_mut().enumerate() {
        let layers = masks[i];
        let Some(water) = o.components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<zestengine_core::Water>()) else { continue };
        if !water.interactive {
            water.ripples = None;
            continue;
        }
        let to_local = world[i].inverse();
        if water.ripples.as_ref().is_some_and(|r| r.gpu) {
            feed_gpu_sim(water, i, &to_local, step, physics, t);
            continue;
        }
        let (n, size) = (water.ripple_resolution as usize, water.ripple_area);
        let sim = water.ripples.get_or_insert_with(|| zestengine_core::RippleSim::new(n, size));
        if sim.n != n.clamp(8, 512) || (sim.size - size.max(1.0)).abs() > 1e-3 {
            *sim = zestengine_core::RippleSim::new(n, size);
        }
        let local = to_local.transform_point3(focus);
        sim.follow(glam::Vec2::new(local.x, local.z));
        if let Some(physics) = physics {
            sim.set_blocked(physics.ripple_blockers(world[i], sim, layers));
        }
        let mut mine: Vec<zestengine_core::WaterPush> =
            step.pushes.iter().chain(&step.fx_pushes).filter(|(w, _)| *w == i).map(|(_, p)| *p).collect();
        for wake in step.wakes.iter().filter(|w| w.water == i) {
            mine.extend(wake_pushes(wake, &to_local, t));
        }
        let depth = water.depth.max(0.1);
        let damping = water.ripple_damping;
        if let Some(sim) = water.ripples.as_mut() {
            sim.step(dt, depth, damping, &mine);
        }
    }
}

/// Pushes a GPU-simulated water queues between frames.
const MAX_QUEUED_PUSHES: usize = 256;

/// Hand a GPU-simulated water this step's bodies and pushes (water `i`).
fn feed_gpu_sim(
    water: &mut zestengine_core::Water,
    i: usize,
    to_local: &Mat4,
    step: &WaterStep,
    physics: Option<&PhysicsWorld>,
    t: f32,
) {
    let feed = &mut water.sim_feed;
    if physics.is_some() {
        feed.displacers = step.displacers.iter().filter(|(w, _)| *w == i).map(|(_, d)| *d).collect();
    }
    feed.pushes.extend(step.fx_pushes.iter().filter(|(w, _)| *w == i).map(|(_, p)| *p));
    // Fixed things in a flow shed eddies; bodies' own wakes come from the
    // simulation itself.
    for wake in step.wakes.iter().filter(|w| w.water == i) {
        feed.pushes.push(wake_pushes(wake, to_local, t)[0]);
    }
    let excess = feed.pushes.len().saturating_sub(MAX_QUEUED_PUSHES);
    feed.pushes.drain(..excess);
    // Its wall cells come from `WaterWalls` (every static object and still
    // collider, by layer).
}

/// A moving collider's footprint in the GPU-simulated water it crosses
/// (`lo`..`hi` its local bounds, `pose` its offset on `rb`).
#[allow(clippy::too_many_arguments)]
fn collider_displacer(
    waters: &[(usize, Mat4, &zestengine_core::Water)],
    rb: &rapier3d::dynamics::RigidBody,
    pose: &Pose,
    lo: Vec3,
    hi: Vec3,
    volume: f32,
    floats: bool,
    t: f32,
) -> Option<(usize, zestengine_core::WaterDisplacer)> {
    let to_world = |p: Vec3| gv(rb.position().transform_point(pose.transform_point(rv(p))));
    let c = (lo + hi) * 0.5;
    let cw = to_world(c);
    let (wi, m, water) = waters.iter().find(|(_, m, w)| {
        w.interactive && w.ripples.as_ref().is_some_and(|r| r.gpu) && {
            let l = m.inverse().transform_point3(cw);
            w.covers(glam::Vec2::new(l.x, l.z))
        }
    })?;
    let inv = m.inverse();
    let corners: [Vec3; 8] = std::array::from_fn(|k| {
        let s = Vec3::new(if k & 1 == 0 { lo.x } else { hi.x }, if k & 2 == 0 { lo.y } else { hi.y }, if k & 4 == 0 { lo.z } else { hi.z });
        inv.transform_point3(to_world(s))
    });
    let axis = inv.transform_vector3(to_world(c + Vec3::X) - cw);
    let lc = inv.transform_point3(cw);
    let xz = glam::Vec2::new(lc.x, lc.z);
    // Floating bodies ride the waves, so their footprint is measured against
    // the moving surface; bodies held in place or resting on the bottom
    // against the rest level, so passing waves don't make them pump rings
    // into the water.
    let surface = if rb.is_dynamic() && floats { water.base_surface(xz, t)? } else { water.column(xz)?.surface };
    let col = water.column(xz)?;
    let vel = inv.transform_vector3(gv(rb.velocity_at_point(rv(cw))));
    let rel = glam::Vec2::new(vel.x - col.flow.x, vel.z - col.flow.z);
    let size = hi - lo;
    let boxy = volume / (size.x * size.y * size.z).max(1e-6) > 0.8;
    zestengine_core::WaterDisplacer::from_corners(&corners, axis, surface, water.sim_wave_depth.min(col.depth), rel, !boxy).map(|d| (*wi, d))
}

/// The simulated water's velocity around a body (world): sampled a little
/// outside its reach, where it isn't dragging the water itself.
fn ambient_sim_flow(water: &zestengine_core::Water, water_world: Mat4, centre: Vec3, reach: f32) -> Vec3 {
    if water.ripples.as_ref().is_none_or(|r| !r.gpu) {
        return Vec3::ZERO;
    }
    let r = reach.max(0.1) * 1.3;
    [Vec3::X, Vec3::NEG_X, Vec3::Z, Vec3::NEG_Z].iter().map(|d| water.sim_flow_at(water_world, centre + *d * r)).sum::<Vec3>() * 0.25
}

/// Eddies shed behind something the water flows past: alternating sides at
/// the Strouhal frequency (0.2 U / D), plus a bow push upstream.
fn wake_pushes(wake: &Wake, to_local: &Mat4, t: f32) -> [zestengine_core::WaterPush; 2] {
    let u = wake.flow.length().max(1e-3);
    let dir = wake.flow / u;
    let side = Vec3::new(-dir.z, 0.0, dir.x);
    let d = wake.radius * 2.0;
    let phase = (std::f32::consts::TAU * 0.2 * u / d.max(0.05) * t).sin();
    let eddy = wake.at + dir * wake.radius * 1.4 + side * (wake.radius * 0.6 * phase.signum());
    let bow = wake.at - dir * wake.radius;
    let local = |p: Vec3| {
        let l = to_local.transform_point3(p);
        glam::Vec2::new(l.x, l.z)
    };
    let strength = (u * 0.35).min(2.0);
    [
        zestengine_core::WaterPush { at: local(eddy), radius: wake.radius * 0.6, strength: strength * phase.abs() },
        zestengine_core::WaterPush { at: local(bow), radius: wake.radius * 0.5, strength: -strength * 0.3 },
    ]
}

/// What one water step produced: each Buoyancy body's submerged share and
/// the pushes bodies gave each Water (by object index).
#[derive(Default)]
pub struct WaterStep {
    pub submerged: Vec<(usize, f32)>,
    /// Bodies' pushes (the CPU ripples) and landing water's (`water_fx`).
    pub pushes: Vec<(usize, zestengine_core::WaterPush)>,
    pub fx_pushes: Vec<(usize, zestengine_core::WaterPush)>,
    /// Bodies' footprints in the water (the GPU simulation).
    pub displacers: Vec<(usize, zestengine_core::WaterDisplacer)>,
    /// Fast entries this step.
    pub splashes: Vec<zestengine_core::WaterSplash>,
    /// Water moving past something at the surface this step.
    pub wakes: Vec<Wake>,
}

/// Water flowing past a body at the surface (world space).
#[derive(Clone, Copy, Debug)]
pub struct Wake {
    pub water: usize,
    pub water_world: Mat4,
    pub at: Vec3,
    /// Water velocity relative to the body (horizontal).
    pub flow: Vec3,
    pub radius: f32,
}

pub struct PhysicsWorld {
    pipeline: PhysicsPipeline,
    gravity: Vector,
    params: IntegrationParameters,
    islands: IslandManager,
    broad_phase: DefaultBroadPhase,
    narrow_phase: NarrowPhase,
    rigid_bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd: CCDSolver,
    bodies: HashMap<ObjectId, Body>,
    /// Kept past collider removal so the final "stopped" event still resolves.
    collider_owner: HashMap<ColliderHandle, ObjectId>,
    collector: ChannelEventCollector,
    collision_rx: Receiver<CollisionEvent>,
    _force_rx: Receiver<ContactForceEvent>,
    pending: Vec<ContactEvent>,
    dirty: bool,
    /// Joint per object carrying a `Joint` component.
    joints: HashMap<ObjectId, (ImpulseJointHandle, JointKind)>,
    /// Static body at the origin that world-anchored joints attach to.
    world_anchor: RigidBodyHandle,
    /// Project root `.physmat` paths resolve against (None = materials ignored).
    root: Option<PathBuf>,
    default_material: String,
    materials: HashMap<String, Option<PhysicsMaterial>>,
    /// Colliders whose static friction differs from their sliding friction:
    /// (static coefficient, its combine rule), applied by `FrictionHooks`.
    static_friction: HashMap<ColliderHandle, (f32, CoefficientCombineRule)>,
}

fn rv(v: Vec3) -> Vector {
    Vector::new(v.x, v.y, v.z)
}

fn gv(v: Vector) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

fn rq(q: Quat) -> Rotation {
    Rotation::from_xyzw(q.x, q.y, q.z, q.w)
}

fn gq(r: &Rotation) -> Quat {
    Quat::from_xyzw(r.x, r.y, r.z, r.w)
}

fn pose(t: Vec3, r: Quat) -> Pose {
    Pose::from_parts(rv(t), rq(r))
}

impl PhysicsWorld {
    /// Build a world from the scene's active collider / rigid-body objects.
    pub fn build(scene: &LoadedScene) -> Self {
        Self::build_with(scene, None, &PhysicsSettings::default())
    }

    /// [`Self::build`] with project physics settings, resolving physics
    /// materials under `root`.
    pub fn build_with(scene: &LoadedScene, root: Option<&Path>, settings: &PhysicsSettings) -> Self {
        let (collision_tx, collision_rx) = channel();
        let (force_tx, force_rx) = channel();
        let mut rigid_bodies = RigidBodySet::new();
        let world_anchor = rigid_bodies.insert(RigidBodyBuilder::fixed().build());
        let mut world = Self {
            pipeline: PhysicsPipeline::new(),
            gravity: rv(Vec3::from(settings.gravity)),
            params: IntegrationParameters {
                dt: FIXED_DT,
                contact_softness: SpringCoefficients::new(
                    settings.contact_stiffness.clamp(1.0, 1000.0),
                    settings.contact_damping.clamp(0.0, 100.0),
                ),
                num_solver_iterations: settings.solver_iterations.clamp(1, 32) as usize,
                ..Default::default()
            },
            islands: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            rigid_bodies,
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd: CCDSolver::new(),
            bodies: HashMap::new(),
            collider_owner: HashMap::new(),
            collector: ChannelEventCollector::new(collision_tx, force_tx),
            collision_rx,
            _force_rx: force_rx,
            pending: Vec::new(),
            dirty: true,
            joints: HashMap::new(),
            world_anchor,
            root: root.map(Path::to_owned),
            default_material: settings.default_material.clone(),
            materials: HashMap::new(),
            static_friction: HashMap::new(),
        };
        world.sync_structure(scene);
        world
    }

    /// Objects were spawned, destroyed or toggled; rebuild membership next sync.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Add bodies for newly active physics objects and drop bodies whose object
    /// is gone or disabled. Cheap no-op unless marked dirty.
    pub fn sync_structure(&mut self, scene: &LoadedScene) {
        if !self.dirty {
            return;
        }
        self.dirty = false;
        let world = scene.world_transforms();
        let mut wanted = HashMap::new();
        for (i, object) in scene.objects.iter().enumerate() {
            if is_physics_object(scene, i) && scene.is_active(i) {
                wanted.insert(object.id, i);
            }
        }
        let stale: Vec<ObjectId> = self
            .bodies
            .keys()
            .filter(|id| !wanted.contains_key(id))
            .copied()
            .collect();
        for id in stale {
            if let Some(body) = self.bodies.remove(&id) {
                self.rigid_bodies.remove(
                    body.handle,
                    &mut self.islands,
                    &mut self.colliders,
                    &mut self.impulse_joints,
                    &mut self.multibody_joints,
                    true,
                );
            }
        }
        let colliders = &self.colliders;
        self.static_friction.retain(|h, _| colliders.contains(*h));
        for (&id, &i) in &wanted {
            if !self.bodies.contains_key(&id) {
                self.add_body(scene, i, world[i]);
            }
        }
        // Joints: drop those whose bodies went away (rapier removed them with
        // the body), then create missing ones once both ends exist.
        let impulse_joints = &self.impulse_joints;
        self.joints
            .retain(|id, (h, _)| wanted.contains_key(id) && impulse_joints.get(*h).is_some());
        for (&id, &i) in &wanted {
            if self.joints.contains_key(&id) {
                continue;
            }
            if let Some(j) = scene.objects[i]
                .components
                .iter()
                .find_map(|c| c.as_any().downcast_ref::<PhysicsJoint>())
            {
                self.add_joint(id, j, world[i]);
            }
        }
    }

    fn add_joint(&mut self, id: ObjectId, j: &PhysicsJoint, world: Mat4) {
        let Some(b1) = self.bodies.get(&id).map(|b| b.handle) else {
            return;
        };
        let b2 = match j.connected.id() {
            Some(other) => match self.bodies.get(&other) {
                Some(b) => b.handle,
                None => return,
            },
            None => self.world_anchor,
        };
        let (Some(rb1), Some(rb2)) = (self.rigid_bodies.get(b1), self.rigid_bodies.get(b2)) else {
            return;
        };
        let (t1, r1) = (gv(rb1.translation()), gq(rb1.rotation()));
        let (t2, r2) = (gv(rb2.translation()), gq(rb2.rotation()));
        let scale = world.to_scale_rotation_translation().0;
        let a1 = Vec3::from(j.anchor) * scale;
        let anchor_world = t1 + r1 * a1;
        let a2 = r2.inverse() * (anchor_world - t2);
        let axis1 = Vec3::from(j.axis).normalize_or(Vec3::Y);
        let axis2 = r2.inverse() * (r1 * axis1);
        // Springs / ropes to another body attach at its origin, so the start
        // distance is the rest length; world-anchored ones pull to the start point.
        let far2 = if j.connected.is_set() { Vec3::ZERO } else { a2 };
        let rest = if j.rest_length > 0.0 {
            j.rest_length
        } else {
            (t2 + r2 * far2).distance(anchor_world)
        };
        let lim = |deg: bool| {
            let f = |v: f32| if deg { v.to_radians() } else { v };
            [f(j.limits[0].min(j.limits[1])), f(j.limits[0].max(j.limits[1]))]
        };
        let mut data: GenericJoint = match j.kind {
            JointKind::Fixed => FixedJointBuilder::new()
                .local_frame1(Pose::from_parts(rv(a1), Rotation::IDENTITY))
                .local_frame2(Pose::from_parts(rv(a2), rq(r2.inverse() * r1)))
                .build()
                .into(),
            JointKind::Hinge => {
                let mut b = RevoluteJointBuilder::new(rv(axis1)).local_anchor1(rv(a1)).local_anchor2(rv(a2));
                if j.limits_enabled {
                    b = b.limits(lim(true));
                }
                if j.motor_enabled {
                    b = b.motor_velocity(j.motor_speed.to_radians(), MOTOR_GAIN).motor_max_force(j.motor_max_force);
                }
                let mut data = b.build().data;
                data.set_local_axis2(rv(axis2));
                data
            }
            JointKind::Ball => SphericalJointBuilder::new()
                .local_anchor1(rv(a1))
                .local_anchor2(rv(a2))
                .build()
                .into(),
            JointKind::Slider => {
                let mut b = PrismaticJointBuilder::new(rv(axis1))
                    .local_anchor1(rv(a1))
                    .local_anchor2(rv(a2))
                    .local_axis2(rv(axis2));
                if j.limits_enabled {
                    b = b.limits(lim(false));
                }
                if j.motor_enabled {
                    b = b.motor_velocity(j.motor_speed, MOTOR_GAIN).motor_max_force(j.motor_max_force);
                }
                b.build().into()
            }
            JointKind::Spring => SpringJointBuilder::new(rest, j.stiffness, j.damping)
                .local_anchor1(rv(a1))
                .local_anchor2(rv(far2))
                .build()
                .into(),
            JointKind::Rope => RopeJointBuilder::new(rest)
                .local_anchor1(rv(a1))
                .local_anchor2(rv(far2))
                .build()
                .into(),
        };
        data.set_contacts_enabled(j.collide_connected);
        let h = self.impulse_joints.insert(b1, b2, data, true);
        self.joints.insert(id, (h, j.kind));
    }

    pub fn set_joint_motor(&mut self, id: ObjectId, speed: f32, max_force: f32) {
        let Some(&(h, kind)) = self.joints.get(&id) else {
            return;
        };
        let (axis, speed) = match kind {
            JointKind::Hinge => (JointAxis::AngX, speed.to_radians()),
            JointKind::Slider => (JointAxis::LinX, speed),
            _ => return,
        };
        if let Some(joint) = self.impulse_joints.get_mut(h, true) {
            joint.data.set_motor_velocity(axis, speed, MOTOR_GAIN);
            joint.data.set_motor_max_force(axis, max_force);
        }
    }

    /// The material at `path` (project-relative), loaded once.
    fn material(&mut self, path: &str) -> Option<PhysicsMaterial> {
        let root = self.root.as_ref()?;
        if path.is_empty() {
            return None;
        }
        *self.materials.entry(path.to_owned()).or_insert_with(|| {
            zestengine_core::load_physmat(zestengine_core::asset_ref::resolve_path(root, path))
                .map_err(|e| tracing::warn!("physics material: {e}"))
                .ok()
        })
    }

    fn add_body(&mut self, scene: &LoadedScene, i: usize, world: Mat4) {
        let object = &scene.objects[i];
        let id = object.id;
        let (scale, rot, trans) = world.to_scale_rotation_translation();
        let character = object
            .components
            .iter()
            .find_map(|c| c.as_any().downcast_ref::<Character>());
        let rb = object
            .components
            .iter()
            .find_map(|c| c.as_any().downcast_ref::<RbComponent>());
        let kind = if character.is_some() {
            BodyKind::Kinematic
        } else {
            rb.map(|r| r.kind).unwrap_or(BodyKind::Fixed)
        };
        let builder = match kind {
            BodyKind::Dynamic => RigidBodyBuilder::dynamic(),
            BodyKind::Kinematic => RigidBodyBuilder::kinematic_position_based(),
            BodyKind::Fixed => RigidBodyBuilder::fixed(),
        }
        .pose(pose(trans, rot))
        .user_data(id.raw());
        let builder = match rb {
            Some(r) if kind == BodyKind::Dynamic => {
                let b = builder.gravity_scale(r.gravity_scale).ccd_enabled(true);
                if r.mass > 0.0 { b.additional_mass(r.mass) } else { b }
            }
            _ => builder,
        };
        let handle = self.rigid_bodies.insert(builder.build());

        let layer = object.layer;
        let groups = InteractionGroups::new(
            Group::from_bits_truncate(1u32 << (layer as u32 & 31)),
            Group::from_bits_truncate(scene.layers.collision_mask(layer)),
            InteractionTestMode::And,
        );
        let default_material = self.default_material.clone();
        let body_material = rb.and_then(|r| self.material(&r.material)).or_else(|| self.material(&default_material));
        // Without a material the body's own restitution / friction apply.
        let legacy = rb.map(|r| PhysicsMaterial { bounciness: r.restitution, friction: r.friction, ..Default::default() });
        let auto_mass = rb.is_none_or(|r| r.mass <= 0.0);
        let finish = |b: ColliderBuilder, trigger: bool, mat: Option<PhysicsMaterial>| {
            let m = mat.or(legacy).unwrap_or_default();
            let b = b
                .restitution(m.bounciness.clamp(0.0, 1.0))
                .restitution_combine_rule(combine_rule(m.bounce_combine))
                .friction(m.friction.max(0.0))
                .friction_combine_rule(combine_rule(m.friction_combine))
                .contact_skin(m.contact_offset.max(0.0))
                .collision_groups(groups)
                .active_events(ActiveEvents::COLLISION_EVENTS)
                .user_data(id.raw());
            let b = if auto_mass { b.density(mat.map_or(DEFAULT_DENSITY, |m| m.density).max(1e-3)) } else { b };
            let b = if (m.static_mu() - m.friction.max(0.0)).abs() > 1e-4 { b.active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS) } else { b };
            if trigger {
                b.sensor(true).active_collision_types(ActiveCollisionTypes::all())
            } else {
                b
            }
        };

        let mut controller = None;
        let mut capsule = None;
        let mut capsule_dims = None;
        if let Some(cc) = character.filter(|c| c.use_default_capsule) {
            capsule_dims = Some((cc.radius, cc.height, cc.center, scale));
            let s = scale.max_element().max(1e-3);
            let radius = (cc.radius * s).max(0.01);
            let (shape, center) = capsule_shape(cc, scale);
            let collider = finish(ColliderBuilder::new(shape).translation(rv(center)), false, body_material).build();
            let h = self
                .colliders
                .insert_with_parent(collider, handle, &mut self.rigid_bodies);
            self.collider_owner.insert(h, id);
            capsule = Some(h);
            controller = Some(KinematicCharacterController {
                autostep: (cc.step_height > 0.0).then_some(CharacterAutostep {
                    max_height: CharacterLength::Absolute(cc.step_height * s),
                    min_width: CharacterLength::Absolute(radius * 0.5),
                    include_dynamic_bodies: false,
                }),
                max_slope_climb_angle: cc.max_slope.to_radians(),
                min_slope_slide_angle: cc.max_slope.to_radians(),
                snap_to_ground: (cc.snap_to_ground > 0.0).then_some(CharacterLength::Absolute(cc.snap_to_ground)),
                offset: CharacterLength::Absolute(0.02),
                ..Default::default()
            });
        } else {
            for (shape, trigger, material) in collider_shapes(scene, i, scale, kind) {
                let mat = self.material(material).or(body_material);
                let h = self.colliders.insert_with_parent(
                    finish(shape, trigger, mat).build(),
                    handle,
                    &mut self.rigid_bodies,
                );
                self.collider_owner.insert(h, id);
                let m = mat.or(legacy).unwrap_or_default();
                if (m.static_mu() - m.friction.max(0.0)).abs() > 1e-4 {
                    self.static_friction.insert(h, (m.static_mu(), combine_rule(m.friction_combine)));
                }
            }
        }
        self.bodies.insert(
            id,
            Body {
                handle,
                kind,
                character: controller,
                capsule,
                capsule_dims,
                prev: (trans, rot),
                curr: (trans, rot),
                written: (object.translation, object.rotation),
                grounded: false,
                water: body_material.map_or((1.0, 1.0), |m| (m.buoyancy.max(0.0), m.water_drag.max(0.0))),
                wet: 0.0,
                dry_for: 0.0,
                carried: (0.0, Vec3::ZERO),
                moving: 0,
            },
        );
    }

    /// Push gameplay-driven transforms into the simulation: kinematic bodies get
    /// their next target pose; any other body whose scene transform changed
    /// since the last write-back is teleported.
    pub fn push_poses(&mut self, scene: &LoadedScene, id_index: &HashMap<ObjectId, usize>, world: &[Mat4]) {
        for (id, body) in &mut self.bodies {
            let Some(&i) = id_index.get(id) else {
                continue;
            };
            let object = &scene.objects[i];
            let (_, r, t) = world[i].to_scale_rotation_translation();
            let Some(rb) = self.rigid_bodies.get_mut(body.handle) else {
                continue;
            };
            let moved = (object.translation, object.rotation) != body.written;
            if body.kind == BodyKind::Kinematic && body.character.is_none() {
                rb.set_next_kinematic_position(pose(t, r));
            } else if moved {
                if body.kind == BodyKind::Fixed {
                    body.moving = MOVED_STEPS;
                }
                rb.set_position(pose(t, r), true);
                if body.kind == BodyKind::Kinematic {
                    rb.set_next_kinematic_position(pose(t, r));
                }
                body.prev = (t, r);
                body.curr = (t, r);
            }
            body.written = (object.translation, object.rotation);
        }
    }

    /// Advance one fixed step and queue the resulting contact events.
    pub fn step(&mut self) {
        self.params.dt = FIXED_DT;
        self.pipeline.step(
            self.gravity,
            &self.params,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd,
            &FrictionHooks { statics: &self.static_friction },
            &self.collector,
        );
        for body in self.bodies.values_mut() {
            body.moving = body.moving.saturating_sub(1);
            if body.kind != BodyKind::Dynamic {
                continue;
            }
            if let Some(rb) = self.rigid_bodies.get(body.handle) {
                body.prev = body.curr;
                body.curr = (gv(rb.translation()), gq(rb.rotation()));
            }
        }
        while let Ok(event) = self.collision_rx.try_recv() {
            let (h1, h2) = (event.collider1(), event.collider2());
            let (Some(&a), Some(&b)) = (self.collider_owner.get(&h1), self.collider_owner.get(&h2)) else {
                continue;
            };
            let trigger = match event {
                CollisionEvent::Started(_, _, f) | CollisionEvent::Stopped(_, _, f) => {
                    f.contains(CollisionEventFlags::SENSOR)
                }
            };
            self.pending.push(ContactEvent {
                a,
                b,
                started: event.started(),
                trigger,
            });
            if event.stopped() {
                for h in [h1, h2] {
                    if self.colliders.get(h).is_none() {
                        self.collider_owner.remove(&h);
                    }
                }
            }
        }
    }

    /// Write dynamic bodies back to the scene, interpolated `alpha` (0..1)
    /// between the last two steps.
    pub fn write_back(&mut self, scene: &mut LoadedScene, id_index: &HashMap<ObjectId, usize>, alpha: f32) {
        for (id, body) in &mut self.bodies {
            if body.kind != BodyKind::Dynamic {
                continue;
            }
            let Some(&i) = id_index.get(id) else {
                continue;
            };
            let t = body.prev.0.lerp(body.curr.0, alpha);
            let r = body.prev.1.slerp(body.curr.1, alpha);
            let local = world_to_local(scene, i, t, r);
            scene.objects[i].translation = local.0;
            scene.objects[i].rotation = local.1;
            body.written = local;
        }
    }

    pub fn take_events(&mut self) -> Vec<ContactEvent> {
        std::mem::take(&mut self.pending)
    }

    fn dynamic_body(&mut self, id: ObjectId) -> Option<&mut rapier3d::dynamics::RigidBody> {
        let body = self.bodies.get(&id)?;
        if body.kind != BodyKind::Dynamic {
            return None;
        }
        self.rigid_bodies.get_mut(body.handle)
    }

    /// Water forces at time `t` (the clock the water shader animates with):
    /// every dynamic body floats by Archimedes from its real density (or by a
    /// Buoyancy component's override), feels drag, current and `vortices`;
    /// every body crossing the surface pushes ripples, splashes when it hits
    /// fast and leaves a wake when water moves past it.
    pub fn apply_water(&mut self, scene: &LoadedScene, world: &[Mat4], t: f32, vortices: &[zestengine_core::Vortex]) -> WaterStep {
        let waters: Vec<(usize, Mat4, &zestengine_core::Water)> = scene
            .objects
            .iter()
            .enumerate()
            .filter(|(i, _)| scene.is_active(*i))
            .filter_map(|(i, o)| o.components.iter().find_map(|c| c.as_any().downcast_ref::<zestengine_core::Water>()).map(|w| (i, world[i], w)))
            .collect();
        let mut out = WaterStep::default();
        if waters.is_empty() {
            return out;
        }
        let g = -self.gravity.y.min(-1e-3);
        let all_waters = waters;
        for (i, o) in scene.objects.iter().enumerate() {
            let Some(body) = self.bodies.get(&o.id) else { continue };
            // Only waters whose layer collides with this object's.
            let waters: Vec<(usize, Mat4, &zestengine_core::Water)> =
                all_waters.iter().copied().filter(|(wi, _, _)| layers_meet(scene, *wi, i)).collect();
            if waters.is_empty() {
                continue;
            }
            let moved_fixed = body.moving > 0;
            let b = o.components.iter().find_map(|c| c.as_any().downcast_ref::<zestengine_core::Buoyancy>());
            let (handle, kind, dry_for) = (body.handle, body.kind, body.dry_for);
            let (lift_scale, drag_scale) = body.water;
            let Some(rb) = self.rigid_bodies.get(handle) else { continue };
            // A container displaces its whole hull: the walls plus the hollow
            // they enclose (the water inside is part of its mass instead).
            let hull = o.components.iter().find_map(|c| c.as_any().downcast_ref::<zestengine_core::WaterContainer>()).map(|c| {
                let (s, _, _) = world[i].to_scale_rotation_translation();
                let (cc, ch) = (Vec3::from(c.center) * s, c.half_m(s));
                let at = rb.position().transform_point(rv(cc));
                let solid = rb.colliders().iter().filter_map(|h| self.colliders.get(*h)).any(|col| !col.is_sensor() && col.shape().contains_point(col.position(), at));
                (cc, ch, if solid { 0.0 } else { 8.0 * ch.x * ch.y * ch.z })
            });
            let hollow = hull.map_or(0.0, |h| h.2);
            // Whether it floats: lighter than the water it can displace (or
            // lifted past its weight by a Buoyancy override).
            let floats = match b {
                Some(b) => b.buoyancy * lift_scale >= 1.0,
                None => {
                    let full: f32 = hollow + rb.colliders().iter().filter_map(|h| self.colliders.get(*h)).filter(|c| !c.is_sensor()).map(|c| c.volume()).sum::<f32>();
                    let density = waters.iter().map(|(_, _, w)| w.density).fold(0.0f32, f32::max);
                    density * full * lift_scale >= rb.mass()
                }
            };
            // Sample points in world space: authored, else the corners (pulled
            // in a little) and centre of each collider's bounds.
            let mut depth = b.map_or(0.0, |b| b.point_depth);
            let mut volume = 0.0;
            let mut reach = 0.0f32;
            let mut points: Vec<Vec3> = b.map(|b| b.points.iter().map(|p| world[i].transform_point3(Vec3::from(*p))).collect()).unwrap_or_default();
            let custom_points = b.is_some_and(|b| !b.points.is_empty());
            let (mut hull_lo, mut hull_hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
            for ch in rb.colliders() {
                // Movement collision and water interaction are independent. Bone-attached
                // limb bodies still contribute when the large movement capsule opts out.
                if body.capsule == Some(*ch) && o.components.iter().any(|c|
                    c.as_any().downcast_ref::<Character>().is_some_and(|c| !c.capsule_affects_water)) {
                    continue;
                }
                let Some(col) = self.colliders.get(*ch) else { continue };
                if col.is_sensor() {
                    continue;
                }
                volume += col.volume();
                let aabb = col.shape().compute_local_aabb();
                let (lo, hi) = (gv(aabb.mins), gv(aabb.maxs));
                let (c, h) = ((lo + hi) * 0.5, (hi - lo) * 0.5 * 0.8);
                reach = reach.max(((hi - lo) * 0.5).length());
                if depth <= 0.0 {
                    depth = (hi.y - lo.y) * 0.5;
                }
                let local = col.position_wrt_parent().copied().unwrap_or(Pose::IDENTITY);
                if (kind != BodyKind::Fixed || moved_fixed)
                    && let Some(d) = collider_displacer(&waters, rb, &local, lo, hi, col.volume(), floats, t)
                {
                    out.displacers.push(d);
                }
                if hull.is_some() {
                    for k in 0..8 {
                        let s = Vec3::new(if k & 1 == 0 { lo.x } else { hi.x }, if k & 2 == 0 { lo.y } else { hi.y }, if k & 4 == 0 { lo.z } else { hi.z });
                        let q = gv(local.transform_point(rv(s)));
                        hull_lo = hull_lo.min(q);
                        hull_hi = hull_hi.max(q);
                    }
                }
                if custom_points || hull.is_some() {
                    continue;
                }
                let to_world = |p: Vec3| gv(rb.position().transform_point(local.transform_point(rv(p))));
                points.push(to_world(c));
                for k in 0..8 {
                    let s = Vec3::new(if k & 1 == 0 { -1.0 } else { 1.0 }, if k & 2 == 0 { -1.0 } else { 1.0 }, if k & 4 == 0 { -1.0 } else { 1.0 });
                    points.push(to_world(c + s * h));
                }
            }
            // A container samples its hull's bounds as one box.
            if let Some((cc, ch, _)) = hull {
                volume += hollow;
                hull_lo = hull_lo.min(cc - ch);
                hull_hi = hull_hi.max(cc + ch);
                if b.is_none_or(|b| b.point_depth <= 0.0) {
                    depth = (hull_hi.y - hull_lo.y) * 0.5;
                }
                if !custom_points {
                    let (c, h) = ((hull_lo + hull_hi) * 0.5, (hull_hi - hull_lo) * 0.5 * 0.8);
                    let to_world = |p: Vec3| gv(rb.position().transform_point(rv(p)));
                    points.push(to_world(c));
                    for k in 0..8 {
                        let s = Vec3::new(if k & 1 == 0 { -1.0 } else { 1.0 }, if k & 2 == 0 { -1.0 } else { 1.0 }, if k & 4 == 0 { -1.0 } else { 1.0 });
                        points.push(to_world(c + s * h));
                    }
                }
            }
            if points.is_empty() {
                continue;
            }
            let n = points.len() as f32;
            let depth = depth.max(0.05);
            let mass = rb.mass().max(1e-4);
            let dynamic = kind == BodyKind::Dynamic;
            let follow_current = b.is_none_or(|b| b.follow_current);
            let body_drag = b.map_or(1.0, |b| b.drag_scale) * drag_scale;
            let centre = gv(rb.translation());
            let body_vel = gv(rb.linvel());
            let mut forces = Vec::new();
            let mut submerged = 0.0;
            let mut drag_k = 0.0f32;
            let mut ang_k = 0.0f32;
            // The fluid's own drag, by the volume's equivalent sphere.
            let diameter = (6.0 * volume / std::f32::consts::PI).cbrt().max(0.01);
            let area = std::f32::consts::PI * diameter * diameter * 0.25;
            let mut spin_visc = 0.0f32;
            let mut surface: Option<(usize, Mat4, f32, Vec3)> = None;
            for p in &points {
                let Some((water_index, water_world, water, (h, nrm))) =
                    waters.iter().find_map(|(wi, m, w)| w.height_at(*m, *p, t).map(|s| (*wi, *m, *w, s)))
                else {
                    continue;
                };
                // Each point stands for a slab `depth` tall centred on it.
                let sub = ((h - p.y) / depth + 0.5).clamp(0.0, 1.0);
                if sub <= 0.0 {
                    continue;
                }
                submerged += sub;
                // The simulated layer's flow is taken around the body, not
                // under it, where the body drags the water itself.
                let mut flow = if follow_current { water.base_flow_at(water_world, *p) } else { Vec3::ZERO };
                flow += zestengine_core::vortex_flow(vortices, *p);
                // Wakes come from the steady flow only: the simulated layer's
                // own waves sloshing past a still body would otherwise shed
                // eddies that stir it further.
                if surface.is_none() {
                    surface = Some((water_index, water_world, h, flow));
                }
                flow += ambient_sim_flow(water, water_world, centre, reach);
                let point_vel = gv(rb.velocity_at_point(rv(*p)));
                let rel = point_vel - flow;
                if water.interactive {
                    // Bodies moving through the surface push it: falling in, and a
                    // wake from horizontal motion.
                    let push = (-point_vel.y).max(0.0) * 0.5 + Vec3::new(rel.x, 0.0, rel.z).length() * 0.25;
                    if push > 0.02 && (h - p.y).abs() < depth {
                        let local = water_world.inverse().transform_point3(*p);
                        out.pushes.push((
                            water_index,
                            zestengine_core::WaterPush { at: glam::Vec2::new(local.x, local.z), radius: depth.max(0.2), strength: push * sub },
                        ));
                    }
                }
                if !dynamic {
                    continue;
                }
                let lift = match b {
                    Some(b) => b.buoyancy.max(0.0) * mass * g / n,
                    // Archimedes: the weight of the water this slab displaces.
                    None => (water.density.max(0.0) * volume).min(mass * MAX_LIFT_RATIO) * g / n,
                } * lift_scale;
                // Very light floats get extra heave damping so they settle
                // instead of bouncing off the surface.
                let ratio = (water.density.max(0.0) * volume / mass).clamp(1.0, MAX_LIFT_RATIO);
                let heave = Vec3::new(0.0, -rel.y * ratio.sqrt() * 2.0 * mass * sub / n, 0.0);
                let drag = -rel * (water.drag * body_drag * mass * sub / n + fluid_drag(water, diameter, area, rel.length(), mass) * body_drag * sub / n) + heave;
                // Lift along the local surface normal (blended toward up so it
                // can't push sideways much on steep crests).
                let up = (Vec3::Y * 0.7 + nrm * 0.3).normalize_or(Vec3::Y);
                forces.push((up * lift * sub + drag, *p));
                drag_k += water.angular_drag * sub / n;
                // Stokes rotational drag, pi mu D^3, for thick fluids.
                spin_visc += std::f32::consts::PI * water.effective_viscosity() * diameter.powi(3) * sub / n;
                ang_k += sub / n;
            }
            let frac = submerged / n;
            out.submerged.push((i, frac));
            if let Some((water_index, water_world, h, flow)) = surface {
                let at = Vec3::new(centre.x, h, centre.z);
                let size = reach.max(0.1);
                // A splash is a real entry: clear of the water for a moment,
                // then hitting it fast (not a wave lapping at something slow).
                if dry_for >= SPLASH_DRY_SECS && frac > 0.0 && body_vel.y < -SPLASH_SPEED {
                    out.splashes.push(zestengine_core::WaterSplash { at, velocity: body_vel, size });
                    // The splash is the water's: punch a cavity the size of
                    // the body, deeper the harder it hits; the simulation
                    // throws up the crown and rings.
                    let local = water_world.inverse().transform_point3(at);
                    let push = zestengine_core::WaterPush { at: glam::Vec2::new(local.x, local.z), radius: size, strength: -body_vel.y * 1.5 };
                    out.fx_pushes.push((water_index, push));
                }
                // Water and body moving past each other at the surface.
                let rel = Vec3::new(flow.x - body_vel.x, 0.0, flow.z - body_vel.z);
                if frac < 0.999 && rel.length() > WAKE_SPEED {
                    out.wakes.push(Wake { water: water_index, water_world, at, flow: rel, radius: size });
                }
            }
            if let Some(body) = self.bodies.get_mut(&o.id) {
                body.wet = frac;
                body.dry_for = if frac > 0.0 { 0.0 } else { body.dry_for + FIXED_DT };
            }
            if forces.is_empty() {
                continue;
            }
            let Some(rb) = self.rigid_bodies.get(handle) else { continue };
            let inertia = rb.mass_properties().local_mprops.principal_inertia();
            let spin = gv(rb.angvel());
            let Some(rb) = self.rigid_bodies.get_mut(handle) else { continue };
            for (f, p) in forces {
                rb.add_force_at_point(rv(f), rv(p), true);
            }
            if ang_k > 0.0 {
                let avg_inertia = (inertia.x + inertia.y + inertia.z) / 3.0;
                let visc = spin_visc.min(0.9 * avg_inertia / FIXED_DT);
                rb.add_torque(rv(-spin * (drag_k * avg_inertia + visc)), true);
            }
        }
        out
    }

    /// Mark the ripple cells of a water body that a solid collider passes
    /// through at the still surface (local y = 0), so ripples reflect off it.
    pub fn ripple_blockers(&self, water_world: Mat4, sim: &zestengine_core::RippleSim, layers: u32) -> Vec<bool> {
        self.ripple_blockers_in(water_world, sim.center, sim.size, sim.n, false, layers)
    }

    /// The same for a window of `n` x `n` cells, `size` wide around `center`
    /// (local XZ); `fixed_only` leaves out moving bodies (the GPU
    /// simulation takes those as displacers).
    pub fn ripple_blockers_in(&self, water_world: Mat4, center: glam::Vec2, size: f32, n: usize, fixed_only: bool, layers: u32) -> Vec<bool> {
        let cell = size / n.max(1) as f32;
        let half = size * 0.5;
        let cell_center = |x: usize, z: usize| center - glam::Vec2::splat(half) + glam::Vec2::new((x as f32 + 0.5) * cell, (z as f32 + 0.5) * cell);
        let (lo, hi) = (center - glam::Vec2::splat(half), center + glam::Vec2::splat(half));
        let corners = [(lo.x, lo.y), (hi.x, lo.y), (lo.x, hi.y), (hi.x, hi.y)].map(|(x, z)| water_world.transform_point3(Vec3::new(x, 0.0, z)));
        let (wlo, whi) = corners.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(a, b), p| (a.min(*p), b.max(*p)));
        let surface_y = water_world.w_axis.y;
        let mut mask = vec![false; n * n];
        let to_local = water_world.inverse();
        for (h, col) in self.colliders.iter() {
            if col.is_sensor() {
                continue;
            }
            if fixed_only && col.parent().and_then(|h| self.rigid_bodies.get(h)).is_some_and(|rb| !rb.is_fixed()) {
                continue;
            }
            if col.collision_groups().memberships.bits() & layers == 0 {
                continue;
            }
            // A fixed collider being moved pushes the water instead.
            if fixed_only
                && let Some(id) = self.collider_owner.get(&h)
                && self.bodies.get(id).is_some_and(|b| b.moving > 0)
            {
                continue;
            }
            let aabb = col.compute_aabb();
            let (clo, chi) = (gv(aabb.mins), gv(aabb.maxs));
            if chi.y < surface_y || clo.y > surface_y || chi.x < wlo.x || clo.x > whi.x || chi.z < wlo.z || clo.z > whi.z {
                continue;
            }
            // Cells under the collider's footprint, tested at the surface.
            let a = to_local.transform_point3(clo);
            let b = to_local.transform_point3(chi);
            let (fl, fh) = (a.min(b), a.max(b));
            let to_cell = |v: f32, c: f32| ((v - c + half) / cell).floor();
            let (x0, x1) = (to_cell(fl.x, center.x).max(0.0) as usize, (to_cell(fh.x, center.x) + 1.0).clamp(0.0, n as f32) as usize);
            let (z0, z1) = (to_cell(fl.z, center.y).max(0.0) as usize, (to_cell(fh.z, center.y) + 1.0).clamp(0.0, n as f32) as usize);
            for z in z0..z1 {
                for x in x0..x1 {
                    let c = cell_center(x, z);
                    let p = water_world.transform_point3(Vec3::new(c.x, 0.0, c.y));
                    if col.shape().contains_point(col.position(), rv(p)) {
                        mask[z * n + x] = true;
                    }
                }
            }
        }
        mask
    }

    /// Project gravity (m/s^2).
    pub fn gravity(&self) -> Vec3 {
        gv(self.gravity)
    }

    /// Water carried by a Water Container on a dynamic body: `kg` with its
    /// centre at `com` (body space, metres), `half` the water's rough extent.
    /// It adds to the body's own mass (`own_mass` > 0 = the Rigid Body's set
    /// mass, else what its colliders weigh), so a full bucket floats lower.
    pub fn set_carried_water(&mut self, id: ObjectId, own_mass: f32, kg: f32, com: Vec3, half: Vec3) {
        let Some(body) = self.bodies.get_mut(&id) else { return };
        if body.kind != BodyKind::Dynamic {
            return;
        }
        let (last_kg, last_com) = body.carried;
        if (last_kg - kg).abs() < 0.01 && last_com.distance(com) < 1e-3 {
            return;
        }
        body.carried = (kg, com);
        let Some(rb) = self.rigid_bodies.get_mut(body.handle) else { return };
        let mut extra = MassProperties::default();
        if own_mass > 0.0 {
            // What `additional_mass(own_mass)` added: the colliders' mass
            // distribution scaled to that mass.
            let mut base: MassProperties = rb
                .colliders()
                .iter()
                .filter_map(|h| self.colliders.get(*h))
                .filter(|c| c.is_enabled())
                .map(|c| c.mass_properties().transform_by(c.position_wrt_parent().unwrap_or(&Pose::IDENTITY)))
                .sum();
            if base.mass() > 0.0 {
                base.set_mass(own_mass, true);
                extra = base;
            } else {
                extra = MassProperties::new(Vector::ZERO, own_mass, Vector::splat(own_mass * 0.01));
            }
        }
        if kg > 0.0 {
            // Water barely turns with its container: a fraction of a solid
            // block's inertia.
            let d = half * 2.0;
            let inertia = Vec3::new(d.y * d.y + d.z * d.z, d.x * d.x + d.z * d.z, d.x * d.x + d.y * d.y) * (kg / 12.0 * 0.3);
            extra += MassProperties::new(rv(com), kg, rv(inertia));
        }
        rb.set_additional_mass_properties(extra, true);
    }

    /// Cast each segment against the solid colliders: the first hit's point
    /// and object, per segment.
    pub fn cast_segments(&self, segments: &[(Vec3, Vec3)]) -> Vec<Option<(Vec3, ObjectId)>> {
        if segments.is_empty() {
            return Vec::new();
        }
        let qp = self.broad_phase.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.rigid_bodies,
            &self.colliders,
            QueryFilter::default().exclude_sensors(),
        );
        segments
            .iter()
            .map(|(a, b)| {
                let d = *b - *a;
                let len = d.length();
                if len < 1e-6 {
                    return None;
                }
                let ray = Ray::new(rv(*a), rv(d / len));
                let (h, toi) = qp.cast_ray(&ray, len, true)?;
                let c = self.colliders.get(h)?;
                Some((*a + d / len * toi, ObjectId::from_raw(c.user_data)))
            })
            .collect()
    }

    /// A body's simulated mass (kg) and the volume of its solid colliders (m^3).
    pub fn mass_and_volume(&self, id: ObjectId) -> Option<(f32, f32)> {
        let rb = self.rigid_bodies.get(self.bodies.get(&id)?.handle)?;
        let volume = rb.colliders().iter().filter_map(|h| self.colliders.get(*h)).filter(|c| !c.is_sensor()).map(|c| c.volume()).sum();
        Some((rb.mass(), volume))
    }

    pub fn add_force(&mut self, id: ObjectId, force: Vec3) {
        if let Some(rb) = self.dynamic_body(id) {
            rb.add_force(rv(force), true);
        }
    }

    pub fn add_impulse(&mut self, id: ObjectId, impulse: Vec3) {
        if let Some(rb) = self.dynamic_body(id) {
            rb.apply_impulse(rv(impulse), true);
        }
    }

    pub fn add_torque(&mut self, id: ObjectId, torque: Vec3) {
        if let Some(rb) = self.dynamic_body(id) {
            rb.add_torque(rv(torque), true);
        }
    }

    pub fn set_velocity(&mut self, id: ObjectId, linear: Vec3) {
        if let Some(rb) = self.dynamic_body(id) {
            rb.set_linvel(rv(linear), true);
        }
    }

    /// Raycast-suspension wheeled vehicles: each wheel casts down from its
    /// rest position and pushes the body up, then the driven / steered wheels
    /// apply throttle, braking and lateral grip at the contact point. Control
    /// inputs are written by the `Vehicle` component's `fixed_update` (the
    /// same Move / Jump / Sprint actions as the character controllers).
    pub fn drive_vehicles(&mut self, scene: &LoadedScene, world: &[Mat4]) {
        let down_world = self.gravity().normalize_or(Vec3::NEG_Y);
        let mut forces: Vec<(RigidBodyHandle, Vec3, Vec3)> = Vec::new();
        for (i, o) in scene.objects.iter().enumerate() {
            let Some(vehicle) = o.components.iter().find_map(|c| c.as_any().downcast_ref::<zestengine_core::Vehicle>()) else {
                continue;
            };
            let Some(body) = self.bodies.get(&o.id) else {
                continue;
            };
            if body.kind != BodyKind::Dynamic || vehicle.wheels.is_empty() {
                continue;
            }
            let handle = body.handle;
            let Some(rb) = self.rigid_bodies.get(handle) else {
                continue;
            };
            let (scale, _, _) = world[i].to_scale_rotation_translation();
            let pose = *rb.position();
            let rot = gq(&pose.rotation);
            let trans = gv(pose.translation);
            let up = rot * Vec3::Y;
            let down = -up;
            let mass = rb.mass().max(1e-3);
            let body_vel = gv(rb.linvel());
            let body_forward = rot * Vec3::NEG_Z;
            let forward_speed = body_vel.dot(body_forward);
            let driven = vehicle.wheels.iter().filter(|w| w.drive).count().max(1) as f32;
            let filter = QueryFilter::default().exclude_rigid_body(handle).exclude_sensors();
            let qp = self.broad_phase.as_query_pipeline(
                self.narrow_phase.query_dispatcher(),
                &self.rigid_bodies,
                &self.colliders,
                filter,
            );
            for wheel in &vehicle.wheels {
                let anchor = trans + rot * (Vec3::from(wheel.anchor) * scale);
                let rest = wheel.rest_length.max(0.0);
                let origin = anchor + down * rest;
                let max_dist = wheel.radius + wheel.travel.max(0.0) + rest.max(0.05);
                let ray = Ray::new(rv(origin), rv(down));
                let Some((_, toi)) = qp.cast_ray(&ray, max_dist, true) else {
                    continue;
                };
                if toi > wheel.radius {
                    continue;
                }
                let contact = origin + down * toi;
                let compression = (wheel.radius - toi).clamp(0.0, wheel.travel.max(0.0));
                let v_point = gv(rb.velocity_at_point(rv(contact)));
                let approach = down.dot(v_point);
                let suspension = zestengine_core::vehicle::suspension_force(wheel, compression, approach).max(0.0);
                if suspension > 0.0 {
                    forces.push((handle, up * suspension, contact));
                }

                let steer = if wheel.steer { vehicle.steer * vehicle.steering_degrees.to_radians() } else { 0.0 };
                let forward = (rot * (Quat::from_rotation_y(steer) * Vec3::NEG_Z)).normalize_or(body_forward);
                let long = v_point.dot(forward);
                let lateral = v_point - forward * long - up * v_point.dot(up);
                let mut grip = wheel.grip.max(0.0);
                if vehicle.handbrake && !wheel.steer {
                    grip *= 0.3;
                }
                if lateral.length_squared() > 1e-8 && grip > 0.0 {
                    let desired = (-lateral / FIXED_DT) * mass;
                    let max_friction = grip * suspension;
                    forces.push((handle, desired.clamp_length_max(max_friction.max(0.0)), contact));
                }

                if wheel.drive {
                    let engine_per_wheel = vehicle.engine_force.max(0.0) / driven;
                    let taper = 1.0 - (forward_speed.abs() / vehicle.top_speed.max(1e-3)).clamp(0.0, 1.0);
                    let brake = if vehicle.handbrake && !wheel.steer {
                        1.0
                    } else {
                        vehicle.brake.clamp(0.0, 1.0)
                    };
                    let long_force = zestengine_core::vehicle::drive_force(
                        engine_per_wheel,
                        vehicle.throttle.clamp(0.0, 1.0) * taper,
                        brake,
                        long,
                    );
                    let max_traction = grip * suspension;
                    forces.push((handle, forward * long_force.clamp(-max_traction, max_traction), contact));
                }
            }

            if vehicle.downforce > 0.0 {
                let k = (forward_speed.abs() / vehicle.top_speed.max(1e-3)).clamp(0.0, 1.0).powi(2);
                let com = gv(rb.center_of_mass());
                forces.push((handle, down_world * (vehicle.downforce * k), com));
            }
        }
        for (handle, force, point) in forces {
            if let Some(rb) = self.rigid_bodies.get_mut(handle) {
                rb.add_force_at_point(rv(force), rv(point), true);
            }
        }
    }

    /// Networking: a dynamic body another peer simulates turns kinematic and
    /// follows its scene transform; when authority returns it turns dynamic
    /// again, starting at `vel`.
    pub fn set_net_driven(&mut self, scene: &LoadedScene, id: ObjectId, driven: bool, vel: Vec3) {
        let authored_dynamic = scene
            .index_of(id)
            .and_then(|i| scene.objects[i].components.iter().find_map(|c| c.as_any().downcast_ref::<RbComponent>()))
            .is_some_and(|r| r.kind == BodyKind::Dynamic);
        let Some(body) = self.bodies.get_mut(&id) else {
            return;
        };
        if !authored_dynamic || body.character.is_some() {
            return;
        }
        let Some(rb) = self.rigid_bodies.get_mut(body.handle) else {
            return;
        };
        match (driven, body.kind) {
            (true, BodyKind::Dynamic) => {
                rb.set_body_type(RigidBodyType::KinematicPositionBased, true);
                body.kind = BodyKind::Kinematic;
            }
            (false, BodyKind::Kinematic) => {
                rb.set_body_type(RigidBodyType::Dynamic, true);
                rb.set_linvel(rv(vel), true);
                let pose = (gv(rb.translation()), gq(rb.rotation()));
                body.prev = pose;
                body.curr = pose;
                body.kind = BodyKind::Dynamic;
            }
            _ => {}
        }
    }

    /// Clear per-step forces once all fixed steps of a frame ran (forces added
    /// in `update` apply to the next frame's steps only).
    pub fn reset_forces(&mut self) {
        for body in self.bodies.values() {
            if body.kind == BodyKind::Dynamic
                && let Some(rb) = self.rigid_bodies.get_mut(body.handle)
            {
                rb.reset_forces(false);
                rb.reset_torques(false);
            }
        }
    }

    /// Put a character controller's body where its object is now (a
    /// prediction rewind) without sweeping it there, with its ground contact.
    pub fn place_character(&mut self, scene: &LoadedScene, index: usize, grounded: bool) {
        let object = &scene.objects[index];
        let Some(body) = self.bodies.get_mut(&object.id) else {
            return;
        };
        let (_, r, t) = scene.world_transform(index).to_scale_rotation_translation();
        if let Some(rb) = self.rigid_bodies.get_mut(body.handle) {
            rb.set_position(pose(t, r), true);
            rb.set_next_kinematic_position(pose(t, r));
        }
        body.prev = (t, r);
        body.curr = (t, r);
        body.written = (object.translation, object.rotation);
        body.grounded = grounded;
    }

    /// Move a character controller by `delta` (world space), resolving
    /// collisions now, and write the result into the scene.
    ///
    /// The sweep starts from the body's current pose (the collider's own
    /// world pose only catches up on a physics step), so several moves
    /// between steps chain. The controller keeps a skin off what it touches
    /// but never recovers once inside it, so a capsule that starts there
    /// (spawned with its feet exactly on the floor, grown back from a
    /// crouch, rewound) is lifted clear first, and one a sweep left inside
    /// something is pushed out at once.
    pub fn move_character(&mut self, scene: &mut LoadedScene, index: usize, delta: Vec3) {
        let id = scene.objects[index].id;
        let scene_pose = scene.world_transform(index).to_scale_rotation_translation();
        let Some(body) = self.bodies.get_mut(&id) else {
            return;
        };
        let (Some(controller), Some(capsule)) = (body.character, body.capsule) else {
            return;
        };
        if (scene.objects[index].translation, scene.objects[index].rotation) != body.written {
            if let Some(rb) = self.rigid_bodies.get_mut(body.handle) {
                rb.set_position(pose(scene_pose.2, scene_pose.1), true);
                rb.set_next_kinematic_position(pose(scene_pose.2, scene_pose.1));
            }
        }
        let mut center_shift = Vec3::ZERO;
        if let (Some((r0, h0, c0, scale)), Some(cc)) = (
            body.capsule_dims,
            scene.objects[index].components.iter().find_map(|c| c.as_any().downcast_ref::<Character>()),
        ) && (cc.radius, cc.height, cc.center) != (r0, h0, c0)
        {
            body.capsule_dims = Some((cc.radius, cc.height, cc.center, scale));
            let (shape, center) = capsule_shape(cc, scale);
            // Sweep horizontal tracking movement from the previous body centre. Resizing
            // to the new HMD pose first would teleport the capsule through a wall.
            center_shift = scene_pose.1 * ((Vec3::from(cc.center) - Vec3::from(c0)) * scale).with_y(0.0);
            if let Some(col) = self.colliders.get_mut(capsule) {
                col.set_shape(shape);
                col.set_translation_wrt_parent(rv(center));
            }
        }
        let Some(rb) = self.rigid_bodies.get(body.handle) else {
            return;
        };
        let Some(col) = self.colliders.get(capsule) else {
            return;
        };
        let body_pose = *rb.position();
        let rel = col.position_wrt_parent().copied().unwrap_or(Pose::IDENTITY);
        let filter = QueryFilter::default()
            .groups(col.collision_groups())
            .exclude_rigid_body(body.handle)
            .exclude_sensors();
        let qp = self.broad_phase.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.rigid_bodies,
            &self.colliders,
            filter,
        );
        let skin = match controller.offset {
            CharacterLength::Absolute(d) => d,
            CharacterLength::Relative(f) => f * col.shape().compute_local_aabb().extents().y,
        };
        let shape_pos = Pose::from_translation(rv(-center_shift)) * body_pose * rel;
        // Sweep from a clear start: lifted off the ground it stands on (or is
        // inside of after a spawn, crouch or rewind), aiming back down at the
        // same target. A sweep that starts right at the skin gets unreliable
        // contact normals (a sideways one off the floor stops the character,
        // a missed one lets it sink).
        let backoff = clearance(&qp, col.shape(), &shape_pos, skin * 1.5, skin * 2.0, true);
        let start = Pose::from_translation(rv(backoff)) * shape_pos;
        let movement = controller.move_shape(FIXED_DT, &qp, col.shape(), &start, rv(delta + center_shift - backoff), |_| {});
        // Still inside something (a sweep that missed): out now, not a frame later.
        let moved = Pose::from_translation(movement.translation) * start;
        let after = clearance(&qp, col.shape(), &moved, skin * 0.5, skin, false);
        let new_t = gv(body_pose.translation) + backoff + gv(movement.translation) + after - center_shift;
        let rot = gq(&body_pose.rotation);
        body.grounded = movement.grounded || after.y > after.length() * 0.7;
        if let Some(rb) = self.rigid_bodies.get_mut(body.handle) {
            rb.set_position(pose(new_t, rot), true);
            rb.set_next_kinematic_position(pose(new_t, rot));
        }
        let local = world_to_local(scene, index, new_t, rot);
        scene.objects[index].translation = local.0;
        body.written = (local.0, scene.objects[index].rotation);
    }

    fn query_filter<'a>(filter: &RayFilter, predicate: &'a impl Fn(ColliderHandle, &Collider) -> bool) -> QueryFilter<'a> {
        let mut f = QueryFilter::default().predicate(predicate);
        if !filter.include_triggers {
            f = f.exclude_sensors();
        }
        f
    }
}

/// How far to move a character's shape at `pos` so nothing nearer than
/// `within` stays nearer than `to`: out of the deepest contact, a few times
/// over so corners settle. `ground_only` lifts it straight up off what is
/// under it and ignores walls.
fn clearance(qp: &QueryPipeline, shape: &dyn Shape, pos: &Pose, within: f32, to: f32, ground_only: bool) -> Vec3 {
    let mut push = Vec3::ZERO;
    for _ in 0..4 {
        let at = Pose::from_translation(rv(push)) * *pos;
        let aabb = shape.compute_aabb(&at).loosened(within);
        let mut deepest: Option<(f32, Vec3)> = None;
        for (_, other) in qp.intersect_aabb_conservative(aabb) {
            let Ok(Some(c)) = rapier3d::parry::query::contact(&at, shape, other.position(), other.shape(), within) else {
                continue;
            };
            let out = -gv(c.normal1);
            if ground_only && out.y < 0.7 {
                continue;
            }
            if c.dist < within && deepest.is_none_or(|(d, _)| c.dist < d) {
                deepest = Some((c.dist, if ground_only { Vec3::Y / out.y.max(0.7) } else { out }));
            }
        }
        let Some((dist, out)) = deepest else {
            break;
        };
        push += out * (to - dist);
    }
    push
}

/// A character controller's capsule (scaled by the object's world scale)
/// and its offset from the body origin.
fn capsule_shape(cc: &Character, scale: Vec3) -> (SharedShape, Vec3) {
    let s = scale.max_element().max(1e-3);
    let radius = (cc.radius * s).max(0.01);
    let half = ((cc.height * s) * 0.5 - radius).max(0.0);
    (SharedShape::capsule_y(half, radius), Vec3::from(cc.center) * scale)
}

impl zestengine_core::PhysicsQuery for PhysicsWorld {
    fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32, filter: &RayFilter) -> Option<RayHit> {
        let dir = dir.normalize_or_zero();
        if dir == Vec3::ZERO {
            return None;
        }
        let (mask, exclude) = (filter.layer_mask, filter.exclude.map(|e| e.raw()));
        let predicate = move |_: ColliderHandle, c: &Collider| {
            c.collision_groups().memberships.bits() & mask != 0 && Some(c.user_data) != exclude
        };
        let qp = self.broad_phase.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.rigid_bodies,
            &self.colliders,
            Self::query_filter(filter, &predicate),
        );
        let ray = Ray::new(rv(origin), rv(dir));
        let (handle, hit) = qp.cast_ray_and_get_normal(&ray, max_dist, true)?;
        let collider = self.colliders.get(handle)?;
        Some(RayHit {
            object: ObjectId::from_raw(collider.user_data),
            point: origin + dir * hit.time_of_impact,
            normal: gv(hit.normal),
            distance: hit.time_of_impact,
        })
    }

    fn overlap_sphere(&self, center: Vec3, radius: f32, filter: &RayFilter) -> Vec<ObjectId> {
        let (mask, exclude) = (filter.layer_mask, filter.exclude.map(|e| e.raw()));
        let predicate = move |_: ColliderHandle, c: &Collider| {
            c.collision_groups().memberships.bits() & mask != 0 && Some(c.user_data) != exclude
        };
        let qp = self.broad_phase.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.rigid_bodies,
            &self.colliders,
            Self::query_filter(filter, &predicate),
        );
        let ball = Ball::new(radius.max(1e-4));
        let mut out: Vec<ObjectId> = Vec::new();
        for (_, c) in qp.intersect_shape(Pose::from_translation(rv(center)), &ball) {
            let id = ObjectId::from_raw(c.user_data);
            if !out.contains(&id) {
                out.push(id);
            }
        }
        out
    }

    fn velocity(&self, id: ObjectId) -> Option<Vec3> {
        let body = self.bodies.get(&id)?;
        self.rigid_bodies.get(body.handle).map(|rb| gv(rb.linvel()))
    }

    fn grounded(&self, id: ObjectId) -> bool {
        self.bodies.get(&id).is_some_and(|b| b.grounded)
    }
}

/// Convert a world pose into object `i`'s local translation/rotation.
fn world_to_local(scene: &LoadedScene, i: usize, t: Vec3, r: Quat) -> (Vec3, Quat) {
    match scene.objects[i].parent {
        Some(p) if p < scene.objects.len() => {
            let parent = scene.world_transform(p);
            let (_, pr, _) = parent.to_scale_rotation_translation();
            (parent.inverse().transform_point3(t), pr.inverse() * r)
        }
        _ => (t, r),
    }
}

fn is_physics_object(scene: &LoadedScene, i: usize) -> bool {
    scene.objects[i].components.iter().any(|c| {
        let a = c.as_any();
        a.is::<BoxCollider>()
            || a.is::<SphereCollider>()
            || a.is::<CapsuleCollider>()
            || a.is::<MeshCollider>()
            || a.is::<Character>()
    })
}

/// Rapier collider shapes (with object-space offsets baked in) for every
/// collider component on object `i`, scaled by its world scale. Returns
/// `(builder, is_trigger)`.
/// Relative sliding speed (m/s) at a contact below which it holds by static
/// friction.
const STATIC_SPEED: f32 = 0.05;

/// Static friction: contacts barely sliding take the colliders' static
/// coefficient (combined like the sliding one), so it takes more to start
/// something moving than to keep it going.
struct FrictionHooks<'a> {
    statics: &'a HashMap<ColliderHandle, (f32, CoefficientCombineRule)>,
}

impl PhysicsHooks for FrictionHooks<'_> {
    fn modify_solver_contacts(&self, context: &mut ContactModificationContext) {
        let side = |h: ColliderHandle| {
            self.statics.get(&h).copied().or_else(|| context.colliders.get(h).map(|c| (c.friction(), c.friction_combine_rule())))
        };
        let (Some((a, ra)), Some((b, rb))) = (side(context.collider1), side(context.collider2)) else { return };
        let mu = combine_coefficients(a, ra, b, rb);
        let body = |h: Option<RigidBodyHandle>| h.and_then(|h| context.bodies.get(h));
        let (b1, b2) = (body(context.rigid_body1), body(context.rigid_body2));
        let n = gv(*context.normal);
        for c in context.solver_contacts.iter_mut() {
            let v = |b: Option<&RigidBody>| b.map_or(Vec3::ZERO, |b| gv(b.velocity_at_point(c.point)));
            let rel = v(b1) - v(b2);
            if (rel - n * rel.dot(n)).length() < STATIC_SPEED {
                c.friction = mu;
            }
        }
    }
}

/// Two coefficients combined the way rapier does: the higher-priority rule
/// of the two (Average < Min < Multiply < Max).
fn combine_coefficients(a: f32, ra: CoefficientCombineRule, b: f32, rb: CoefficientCombineRule) -> f32 {
    let rank = |r: CoefficientCombineRule| match r {
        CoefficientCombineRule::Average => 0,
        CoefficientCombineRule::Min => 1,
        CoefficientCombineRule::Multiply => 2,
        _ => 3,
    };
    match if rank(ra) >= rank(rb) { ra } else { rb } {
        CoefficientCombineRule::Average => (a + b) * 0.5,
        CoefficientCombineRule::Min => a.min(b),
        CoefficientCombineRule::Multiply => a * b,
        _ => a.max(b),
    }
}

fn combine_rule(mode: CombineMode) -> CoefficientCombineRule {
    match mode {
        CombineMode::Average => CoefficientCombineRule::Average,
        CombineMode::Minimum => CoefficientCombineRule::Min,
        CombineMode::Multiply => CoefficientCombineRule::Multiply,
        CombineMode::Maximum => CoefficientCombineRule::Max,
    }
}

/// Whether objects `a` and `b`'s layers collide (the Layers matrix): water
/// only moves, lifts and is pushed by what its layer meets.
pub fn layers_meet(scene: &LoadedScene, a: usize, b: usize) -> bool {
    let (la, lb) = (scene.objects[a].layer, scene.objects[b].layer);
    scene.layers.collision_mask(la) & (1u32 << (lb as u32 & 31)) != 0
}

/// Drag coefficient (N per m/s) a body of equivalent-sphere `diameter` and
/// frontal `area` feels moving at `speed` through `water`'s fluid: the
/// sphere drag curve Cd = 24/Re + 6/(1 + sqrt Re) + 0.4 (White), so thick
/// fluids give Stokes drag (3 pi mu D v) and thin ones the quadratic form
/// drag. Capped so one step never more than stops the body.
fn fluid_drag(water: &zestengine_core::Water, diameter: f32, area: f32, speed: f32, mass: f32) -> f32 {
    let nu = water.kinematic_viscosity().max(1e-7);
    let re = speed * diameter / nu;
    let k = 0.5 * water.density.max(0.0) * area * (24.0 * nu / diameter + (6.0 / (1.0 + re.sqrt()) + 0.4) * speed);
    k.min(0.9 * mass / FIXED_DT)
}

/// A solid of an object, posed in the world, for the water to test against.
pub(crate) enum Solid {
    /// A closed shape: points inside it are solid.
    Shape(Collider),
    /// Triangles (world space): solid where they cross a level.
    Mesh(Vec<Vec3>, Vec<[u32; 3]>),
}

/// Object `i`'s solids placed at `world`: its collider components (triggers
/// left out), or for a static object with none, its render mesh. Triangle
/// meshes stay triangles (they need not be closed); everything else is a
/// shape to test points against.
pub(crate) fn world_solids(scene: &LoadedScene, i: usize, world: Mat4) -> Vec<Solid> {
    let (scale, r, t) = world.to_scale_rotation_translation();
    let at = pose(t, r);
    let mut builders: Vec<ColliderBuilder> =
        collider_shapes(scene, i, scale, BodyKind::Fixed).into_iter().filter(|(_, trigger, _)| !trigger).map(|(b, _, _)| b).collect();
    if builders.is_empty()
        && scene.objects[i].static_geometry
        && let Some(shape) = mesh_collider_shape(scene, i, MeshShapeMode::Exact)
        && let MeshColliderShape::Triangles(p, tris) = &*shape
        && let Ok(b) = ColliderBuilder::trimesh(p.iter().map(|v| rv(*v * scale)).collect(), tris.clone())
    {
        builders.push(b);
    }
    builders
        .into_iter()
        .map(|b| {
            let mut c = b.build();
            let placed = at * *c.position();
            c.set_position(placed);
            match c.shape().as_trimesh() {
                Some(m) => Solid::Mesh(m.vertices().iter().map(|v| gv(placed.transform_point(*v))).collect(), m.indices().to_vec()),
                None => Solid::Shape(c),
            }
        })
        .collect()
}

/// Whether world point `p` is inside `c`.
pub(crate) fn solid_contains(c: &Collider, p: Vec3) -> bool {
    c.shape().contains_point(c.position(), rv(p))
}

/// How far down from world point `from` the top of `c` is, within `max`.
pub(crate) fn solid_below(c: &Collider, from: Vec3, max: f32) -> Option<f32> {
    solid_along(c, from, -Vec3::Y, max)
}

/// How far along unit `dir` from world point `from` `c` is, within `max`.
pub(crate) fn solid_along(c: &Collider, from: Vec3, dir: Vec3, max: f32) -> Option<f32> {
    let ray = Ray::new(rv(from), rv(dir));
    c.shape().cast_ray(c.position(), &ray, max, true)
}

/// Each collider component's shape, trigger flag and material path.
fn collider_shapes(scene: &LoadedScene, i: usize, scale: Vec3, kind: BodyKind) -> Vec<(ColliderBuilder, bool, &str)> {
    let obj = &scene.objects[i];
    let mut out = Vec::new();
    for c in &obj.components {
        let any = c.as_any();
        if let Some(b) = any.downcast_ref::<BoxCollider>() {
            let half = (Vec3::from(b.size) * 0.5 * scale).max(Vec3::splat(1e-3));
            let off = Vec3::from(b.center) * scale;
            out.push((ColliderBuilder::cuboid(half.x, half.y, half.z).translation(rv(off)), b.is_trigger, b.material.as_str()));
        } else if let Some(s) = any.downcast_ref::<SphereCollider>() {
            let r = (s.radius * scale.max_element()).max(1e-3);
            let off = Vec3::from(s.center) * scale;
            out.push((ColliderBuilder::ball(r).translation(rv(off)), s.is_trigger, s.material.as_str()));
        } else if let Some(c) = any.downcast_ref::<CapsuleCollider>() {
            // Radius scales with the larger of the two cross axes, length with its own.
            let (along, across) = match c.direction {
                ColliderAxis::X => (scale.x, scale.y.max(scale.z)),
                ColliderAxis::Y => (scale.y, scale.x.max(scale.z)),
                ColliderAxis::Z => (scale.z, scale.x.max(scale.y)),
            };
            let r = (c.radius * across).max(1e-3);
            let half = ((c.height * along) * 0.5 - r).max(0.0);
            let off = Vec3::from(c.center) * scale;
            let shape = match c.direction {
                ColliderAxis::X => ColliderBuilder::capsule_x(half, r),
                ColliderAxis::Y => ColliderBuilder::capsule_y(half, r),
                ColliderAxis::Z => ColliderBuilder::capsule_z(half, r),
            };
            out.push((shape.translation(rv(off)), c.is_trigger, c.material.as_str()));
        } else if let Some(m) = any.downcast_ref::<MeshCollider>()
            && let Some(shape) = mesh_shape(scene, i, scale, mesh_shape_mode(scene, i, m, kind != BodyKind::Fixed))
        {
            out.push((shape, m.is_trigger, m.material.as_str()));
        } else if let Some(tc) = any.downcast_ref::<TerrainCollider>() {
            // The collider is a sibling component; it collides with the
            // Terrain's heightfield on the same object.
            if let Some(t) = obj.components.iter().find_map(|c| c.as_any().downcast_ref::<Terrain>()) {
                let mesh = zestengine_assets::heightfield_mesh(t.size, t.resolution, &t.heights);
                let points: Vec<Vec3> = mesh.vertices.iter().map(|v| Vec3::from(v.position) * scale).collect();
                let tris: Vec<[u32; 3]> = mesh.indices.chunks(3).map(|t| [t[0], t[1], t[2]]).collect();
                if let Ok(shape) = ColliderBuilder::trimesh(points.iter().map(|v| rv(*v)).collect(), tris) {
                    out.push((shape, tc.is_trigger, tc.material.as_str()));
                }
            }
        }
    }
    out
}

/// What a Mesh Collider becomes, in the object's local space (unscaled).
#[derive(Debug)]
pub enum MeshColliderShape {
    /// The exact triangles (static geometry).
    Triangles(Vec<Vec3>, Vec<[u32; 3]>),
    /// Convex pieces: one hull ("Convex"), or a decomposition that follows
    /// a concave mesh for moving bodies (rapier can't simulate concave
    /// dynamic meshes).
    Hulls(Vec<(Vec<Vec3>, Vec<[u32; 3]>)>),
    /// A mesh that is a ball: an exact sphere (centre, radius).
    Ball(Vec3, f32),
    /// Collision pieces modelled with the mesh (`UCX_` etc.).
    Authored(Vec<crate::authored_collision::ColliderPiece>),
}

/// Whether object `i`'s meshes form a ball (see [`ball_of`]).
pub fn mesh_is_ball(scene: &LoadedScene, i: usize) -> bool {
    let verts: Vec<Vec3> = scene.objects[i].render_slots().filter_map(|r| scene.mesh_geometry(r.mesh)).flat_map(|(p, _)| p.iter().copied()).collect();
    ball_of(&verts).is_some()
}

/// Centre and radius when every vertex sits about the same distance from
/// the bounds' centre (a sphere mesh, however finely tessellated).
pub fn ball_of(points: &[Vec3]) -> Option<(Vec3, f32)> {
    if points.len() < 12 {
        return None;
    }
    let (lo, hi) = points.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(a, b), p| (a.min(*p), b.max(*p)));
    let size = hi - lo;
    if size.min_element() < size.max_element() * 0.95 {
        return None;
    }
    let c = (lo + hi) * 0.5;
    let (near, far) = points.iter().fold((f32::MAX, 0.0f32), |(n, f), p| {
        let d = p.distance(c);
        (n.min(d), f.max(d))
    });
    (far > 1e-4 && near > far * 0.94).then_some((c, far))
}

/// How a Mesh Collider turns its mesh into a shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MeshShapeMode {
    /// The exact triangles (static only).
    Exact,
    /// One simplified convex hull.
    Hull(zestengine_core::ColliderDetail),
    /// Simplified convex pieces that follow a concave shape.
    Pieces(zestengine_core::ColliderDetail),
    /// The object's authored collision pieces (static or moving).
    Authored,
}

impl MeshShapeMode {
    pub fn of(collider: &MeshCollider, moving: bool) -> Self {
        let d = collider.detail;
        match (collider.convex, moving, collider.follow_shape) {
            (true, _, _) => Self::Hull(d),
            (false, false, _) => Self::Exact,
            (false, true, true) => Self::Pieces(d),
            (false, true, false) => Self::Hull(d),
        }
    }
}

/// How object `i`'s Mesh Collider `m` builds its shape: from its authored
/// collision pieces when it has some and uses them, else by [`MeshShapeMode::of`].
pub fn mesh_shape_mode(scene: &LoadedScene, i: usize, m: &MeshCollider, moving: bool) -> MeshShapeMode {
    if m.use_authored && crate::authored_collision::authored_set(scene, i).is_some() {
        MeshShapeMode::Authored
    } else {
        MeshShapeMode::of(m, moving)
    }
}

/// Corner budget of a whole-mesh hull, and the pieces' count, corners and
/// voxel resolution, per detail level.
fn budgets(detail: zestengine_core::ColliderDetail) -> (usize, u32, usize, u32) {
    match detail {
        zestengine_core::ColliderDetail::Low => (64, 8, 24, 48),
        zestengine_core::ColliderDetail::Medium => (128, 16, 48, 64),
        zestengine_core::ColliderDetail::High => (256, 32, 96, 96),
    }
}

/// Key for [`LoadedScene::collider_shapes`]: the object's meshes and how
/// they're turned into a collider (for `Authored`, its piece set instead).
pub type ColliderShapeKey = (Vec<usize>, MeshShapeMode);

fn shape_key(scene: &LoadedScene, i: usize, mode: MeshShapeMode) -> ColliderShapeKey {
    let ids = match mode {
        MeshShapeMode::Authored => crate::authored_collision::authored_set(scene, i).into_iter().collect(),
        _ => scene.objects[i].render_slots().map(|r| r.mesh).collect(),
    };
    (ids, mode)
}

/// Object `i`'s authored pieces fitted to their primitives.
fn authored_shape(scene: &LoadedScene, i: usize) -> Option<MeshColliderShape> {
    let pieces = crate::authored_collision::authored_pieces(scene, i)?;
    let fitted: Vec<_> = pieces.iter().filter_map(crate::authored_collision::fit_piece).collect();
    (!fitted.is_empty()).then_some(MeshColliderShape::Authored(fitted))
}

/// The collision shape object `i`'s Mesh Collider builds from its render
/// meshes in `mode`. Cached on the scene (decomposing a big mesh is slow).
/// None when there's no CPU geometry.
pub fn mesh_collider_shape(scene: &LoadedScene, i: usize, mode: MeshShapeMode) -> Option<std::sync::Arc<MeshColliderShape>> {
    let key = shape_key(scene, i, mode);
    if let Some(s) = scene.collider_shapes.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return Some(s.clone());
    }
    if mode == MeshShapeMode::Authored {
        let shape = std::sync::Arc::new(authored_shape(scene, i)?);
        scene.collider_shapes.lock().unwrap_or_else(|e| e.into_inner()).insert(key, shape.clone());
        return Some(shape);
    }
    let meshes = key.0.clone();
    let mut verts: Vec<Vec3> = Vec::new();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    for m in meshes {
        if let Some((pos, idx)) = scene.mesh_geometry(m) {
            let base = verts.len() as u32;
            verts.extend_from_slice(pos);
            tris.extend(idx.as_chunks::<3>().0.iter().map(|t| [base + t[0], base + t[1], base + t[2]]));
        }
    }
    if verts.is_empty() || tris.is_empty() {
        return None;
    }
    if mode != MeshShapeMode::Exact
        && let Some((c, r)) = ball_of(&verts)
    {
        let shape = std::sync::Arc::new(MeshColliderShape::Ball(c, r));
        scene.collider_shapes.lock().unwrap_or_else(|e| e.into_inner()).insert(key, shape.clone());
        return Some(shape);
    }
    let shape = match mode {
        MeshShapeMode::Hull(d) => MeshColliderShape::Hulls(vec![hull_of(&verts, budgets(d).0)?]),
        MeshShapeMode::Pieces(d) => {
            let hull = hull_of(&verts, budgets(d).0)?;
            // Already (nearly) convex, like a ball: one hull is the shape.
            let parts = if mesh_volume(&verts, &tris) > 0.85 * mesh_volume(&hull.0, &hull.1) { Vec::new() } else { convex_pieces(&verts, &tris, d) };
            MeshColliderShape::Hulls(if parts.is_empty() { vec![hull] } else { parts })
        }
        MeshShapeMode::Exact => MeshColliderShape::Triangles(verts, tris),
        MeshShapeMode::Authored => return None,
    };
    let shape = std::sync::Arc::new(shape);
    scene.collider_shapes.lock().unwrap_or_else(|e| e.into_inner()).insert(key, shape.clone());
    Some(shape)
}


/// Convex hull of `points` kept to at most `max_points` corners (spread
/// evenly over the full hull); None for degenerate input (flat or too few
/// points), which the hull builder can't take.
pub(crate) fn hull_of(points: &[Vec3], max_points: usize) -> Option<(Vec<Vec3>, Vec<[u32; 3]>)> {
    let build = |pts: &[Vec3]| -> Option<(Vec<Vec3>, Vec<[u32; 3]>)> {
        let pts: Vec<Vector> = pts.iter().map(|p| rv(*p)).collect();
        let (p, t) = std::panic::catch_unwind(|| rapier3d::parry::transformation::convex_hull(&pts)).ok()?;
        (p.len() >= 4 && !t.is_empty()).then(|| (p.into_iter().map(gv).collect(), t))
    };
    let (p, t) = build(points)?;
    if p.len() <= max_points {
        return Some((p, t));
    }
    // Farthest-point sampling keeps the extremes, so the hull loses detail
    // evenly instead of whole corners.
    let mut keep = vec![0usize];
    let mut dist: Vec<f32> = p.iter().map(|q| q.distance_squared(p[0])).collect();
    while keep.len() < max_points {
        let (far, _) = dist.iter().enumerate().fold((0, -1.0f32), |b, (i, d)| if *d > b.1 { (i, *d) } else { b });
        keep.push(far);
        for (i, q) in p.iter().enumerate() {
            dist[i] = dist[i].min(q.distance_squared(p[far]));
        }
    }
    let picked: Vec<Vec3> = keep.into_iter().map(|i| p[i]).collect();
    build(&picked).or(Some((p, t)))
}

/// Enclosed volume of a closed triangle mesh (0-ish for open ones).
pub(crate) fn mesh_volume(p: &[Vec3], t: &[[u32; 3]]) -> f32 {
    t.iter().map(|t| p[t[0] as usize].dot(p[t[1] as usize].cross(p[t[2] as usize]))).sum::<f32>().abs() / 6.0
}

/// Convex pieces that follow a (possibly concave) mesh. Empty when the
/// decomposition fails, so the caller can fall back to one hull.
fn convex_pieces(points: &[Vec3], tris: &[[u32; 3]], detail: zestengine_core::ColliderDetail) -> Vec<(Vec<Vec3>, Vec<[u32; 3]>)> {
    let (_, max_pieces, piece_points, resolution) = budgets(detail);
    use rapier3d::parry::transformation::vhacd;
    let pts: Vec<Vector> = points.iter().map(|p| rv(*p)).collect();
    let params = vhacd::VHACDParameters { resolution, concavity: 0.025, max_convex_hulls: max_pieces, ..Default::default() };
    // Exact hulls need the voxel-to-triangle map (`true`).
    let parts = std::panic::catch_unwind(|| vhacd::VHACD::decompose(&params, &pts, tris, true).compute_exact_convex_hulls(&pts, tris))
        .map_err(|_| tracing::warn!("mesh collider: convex decomposition failed, using one hull"))
        .unwrap_or_default();
    parts
        .into_iter()
        .filter(|(p, t)| p.len() >= 4 && !t.is_empty())
        .filter_map(|(p, _)| hull_of(&p.into_iter().map(gv).collect::<Vec<_>>(), piece_points))
        .collect()
}

/// Object `i`'s Mesh Collider as wire edges (local space), cached.
pub fn mesh_collider_wire_for(scene: &LoadedScene, i: usize, mode: MeshShapeMode) -> Option<std::sync::Arc<Vec<(Vec3, Vec3)>>> {
    let key = shape_key(scene, i, mode);
    if let Some(w) = scene.collider_wires.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return Some(w.clone());
    }
    let wire = std::sync::Arc::new(mesh_collider_wire(&*mesh_collider_shape(scene, i, mode)?));
    scene.collider_wires.lock().unwrap_or_else(|e| e.into_inner()).insert(key, wire.clone());
    Some(wire)
}

/// The edges that outline a collider shape (local space): where its surface
/// folds by more than a few degrees, and open borders, not the diagonals
/// across flat faces. Capped for display.
pub fn mesh_collider_wire(shape: &MeshColliderShape) -> Vec<(Vec3, Vec3)> {
    const MAX_EDGES: usize = 12_000;
    // cos of the smallest fold that counts as an edge (~15 degrees).
    const FOLD: f32 = 0.966;
    let mut out = Vec::new();
    let add = |out: &mut Vec<(Vec3, Vec3)>, pts: &[Vec3], tris: &[[u32; 3]]| {
        let normal = |t: &[u32; 3]| (pts[t[1] as usize] - pts[t[0] as usize]).cross(pts[t[2] as usize] - pts[t[0] as usize]).normalize_or_zero();
        let mut faces: HashMap<(u32, u32), Vec<Vec3>> = HashMap::new();
        for t in tris {
            let n = normal(t);
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                faces.entry((a.min(b), a.max(b))).or_default().push(n);
            }
        }
        for ((a, b), ns) in faces {
            let fold = ns.len() != 2 || ns[0].dot(ns[1]) < FOLD;
            if fold && out.len() < MAX_EDGES {
                out.push((pts[a as usize], pts[b as usize]));
            }
        }
    };
    match shape {
        MeshColliderShape::Triangles(p, t) => add(&mut out, p, t),
        MeshColliderShape::Hulls(parts) => {
            for (p, t) in parts {
                add(&mut out, p, t);
            }
        }
        MeshColliderShape::Authored(pieces) => {
            for piece in pieces {
                match piece {
                    crate::authored_collision::ColliderPiece::Hull(p, t) => add(&mut out, p, t),
                    _ => crate::authored_collision::primitive_wire(piece, &mut out),
                }
            }
        }
        // Three great circles.
        MeshColliderShape::Ball(c, r) => crate::authored_collision::primitive_wire(&crate::authored_collision::ColliderPiece::Ball(*c, *r), &mut out),
    }
    out
}

/// One authored piece as a rapier shape placed in its object, scaled by the
/// object's world `scale`. Boxes and capsules keep their exact primitive under
/// uniform scale; a non-uniformly scaled box becomes the hull of its corners.
fn authored_piece_shape(piece: &crate::authored_collision::ColliderPiece, scale: Vec3) -> Option<(Pose, SharedShape)> {
    use crate::authored_collision::ColliderPiece;
    let uniform = scale.max_element() - scale.min_element() <= 1e-4 * scale.max_element().abs();
    let s = scale.max_element().max(1e-6);
    match piece {
        ColliderPiece::Hull(p, _) => {
            let pts: Vec<Vector> = p.iter().map(|v| rv(*v * scale)).collect();
            SharedShape::convex_hull(&pts).map(|shape| (Pose::IDENTITY, shape))
        }
        ColliderPiece::Cuboid { center, rotation, half } if uniform => {
            let h = (*half * s).max(Vec3::splat(1e-3));
            Some((pose(*center * s, *rotation), SharedShape::cuboid(h.x, h.y, h.z)))
        }
        ColliderPiece::Cuboid { center, rotation, half } => {
            let pts: Vec<Vector> = (0..8)
                .map(|k| {
                    let c = Vec3::new(if k & 1 == 0 { -1.0 } else { 1.0 }, if k & 2 == 0 { -1.0 } else { 1.0 }, if k & 4 == 0 { -1.0 } else { 1.0 });
                    rv((*center + *rotation * (c * *half)) * scale)
                })
                .collect();
            SharedShape::convex_hull(&pts).map(|shape| (Pose::IDENTITY, shape))
        }
        ColliderPiece::Ball(c, r) => Some((pose(*c * scale, Quat::IDENTITY), SharedShape::ball((*r * s).max(1e-3)))),
        ColliderPiece::Capsule(a, b, r) => Some((Pose::IDENTITY, SharedShape::capsule(rv(*a * scale), rv(*b * scale), (*r * s).max(1e-3)))),
    }
}

/// Collider from the object's render meshes (see [`mesh_collider_shape`]),
/// scaled. Falls back to the mesh AABB when no CPU geometry exists.
fn mesh_shape(scene: &LoadedScene, i: usize, scale: Vec3, mode: MeshShapeMode) -> Option<ColliderBuilder> {
    if let Some(shape) = mesh_collider_shape(scene, i, mode) {
        let built = match &*shape {
            MeshColliderShape::Ball(c, r) => Some(ColliderBuilder::ball((*r * scale.max_element()).max(1e-3)).translation(rv(*c * scale))),
            MeshColliderShape::Triangles(p, t) => ColliderBuilder::trimesh(p.iter().map(|v| rv(*v * scale)).collect(), t.clone()).ok(),
            MeshColliderShape::Hulls(parts) => {
                let pieces: Vec<(Pose, SharedShape)> = parts
                    .iter()
                    .filter_map(|(p, _)| {
                        let pts: Vec<Vector> = p.iter().map(|v| rv(*v * scale)).collect();
                        SharedShape::convex_hull(&pts).map(|s| (Pose::IDENTITY, s))
                    })
                    .collect();
                match pieces.len() {
                    0 => None,
                    1 => Some(ColliderBuilder::new(pieces.into_iter().next().unwrap().1)),
                    _ => Some(ColliderBuilder::compound(pieces)),
                }
            }
            MeshColliderShape::Authored(parts) => {
                let mut pieces: Vec<(Pose, SharedShape)> = parts.iter().filter_map(|p| authored_piece_shape(p, scale)).collect();
                match pieces.len() {
                    0 => None,
                    1 => {
                        let (at, shape) = pieces.remove(0);
                        Some(ColliderBuilder::new(shape).position(at))
                    }
                    _ => Some(ColliderBuilder::compound(pieces)),
                }
            }
        };
        if built.is_some() {
            return built;
        }
    }
    let obj = &scene.objects[i];
    let render = obj.render?;
    let (min, max) = scene.mesh_aabb(render.mesh);
    let half = ((max - min) * 0.5 * scale).max(Vec3::splat(1e-3));
    let center = (min + max) * 0.5 * scale;
    Some(ColliderBuilder::cuboid(half.x, half.y, half.z).translation(rv(center)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SceneObject;
    use zestengine_assets::ObjectSource;
    use zestengine_core::{Component, PhysicsQuery};

    fn obj(scene: &mut LoadedScene, pos: Vec3, components: Vec<Box<dyn Component>>) -> ObjectId {
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
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            components,
        });
        id
    }

    fn floor(scene: &mut LoadedScene) -> ObjectId {
        obj(
            scene,
            Vec3::ZERO,
            vec![Box::new(BoxCollider {
                size: [20.0, 1.0, 20.0],
                ..Default::default()
            })],
        )
    }

    fn dynamic_ball(scene: &mut LoadedScene, pos: Vec3) -> ObjectId {
        obj(
            scene,
            pos,
            vec![Box::new(SphereCollider::default()), Box::new(RbComponent::default())],
        )
    }

    fn run(pw: &mut PhysicsWorld, scene: &mut LoadedScene, steps: usize) -> Vec<ContactEvent> {
        let mut events = Vec::new();
        for _ in 0..steps {
            pw.sync_structure(scene);
            let ids = scene.id_index();
            let world = scene.world_transforms();
            pw.push_poses(scene, &ids, &world);
            pw.step();
            pw.write_back(scene, &ids, 1.0);
            events.extend(pw.take_events());
        }
        events
    }

    fn run_vehicle(pw: &mut PhysicsWorld, scene: &mut LoadedScene, steps: usize) {
        for _ in 0..steps {
            pw.sync_structure(scene);
            let ids = scene.id_index();
            let world = scene.world_transforms();
            pw.push_poses(scene, &ids, &world);
            pw.drive_vehicles(scene, &world);
            pw.step();
            pw.reset_forces();
            pw.write_back(scene, &ids, 1.0);
        }
    }

    #[test]
    fn vehicle_drives_forward_on_suspension() {
        let mut scene = LoadedScene::empty();
        floor(&mut scene);
        let vehicle = zestengine_core::Vehicle { throttle: 0.0, steer: 0.0, downforce: 0.0, ..Default::default() };
        let car = obj(
            &mut scene,
            Vec3::new(0.0, 1.65, 0.0),
            vec![
                Box::new(BoxCollider { size: [1.8, 0.5, 3.6], ..Default::default() }),
                Box::new(RbComponent { mass: 800.0, ..Default::default() }),
                Box::new(vehicle),
            ],
        );
        let ci = scene.index_of(car).unwrap();
        let mut pw = PhysicsWorld::build(&scene);
        // Settle onto the suspension before driving.
        run_vehicle(&mut pw, &mut scene, 90);
        let settle = scene.world_transform(ci).w_axis;
        assert!(settle.y > 1.2 && settle.y < 2.5, "settles on its wheels: {settle:?}");
        let vehicle = scene.objects[ci]
            .components
            .iter_mut()
            .find_map(|c| c.as_any_mut().downcast_mut::<zestengine_core::Vehicle>())
            .unwrap();
        vehicle.throttle = 1.0;
        let start = scene.world_transform(ci).w_axis;
        run_vehicle(&mut pw, &mut scene, 120);
        let end = scene.world_transform(ci).w_axis;
        assert!(end.z - start.z < -1.0, "the car rolls forward (-Z): {start:?} -> {end:?}");
        assert!(end.y > 1.2 && end.y < 3.0, "stays on its wheels: {end:?}");
    }

    fn run_water(pw: &mut PhysicsWorld, scene: &mut LoadedScene, steps: usize) {
        for k in 0..steps {
            pw.sync_structure(scene);
            let ids = scene.id_index();
            let world = scene.world_transforms();
            pw.push_poses(scene, &ids, &world);
            pw.apply_water(scene, &world, k as f32 * FIXED_DT, &[]);
            pw.step();
            pw.reset_forces();
            pw.write_back(scene, &ids, 1.0);
        }
    }

    fn crate_box(scene: &mut LoadedScene, pos: Vec3, buoyancy: f32) -> ObjectId {
        obj(
            scene,
            pos,
            vec![
                Box::new(BoxCollider { size: [1.0, 1.0, 1.0], ..Default::default() }),
                Box::new(RbComponent::default()),
                Box::new(zestengine_core::Buoyancy { buoyancy, ..Default::default() }),
            ],
        )
    }

    fn material_dir(name: &str, mats: &[(&str, PhysicsMaterial)]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("zestengine-physmat-{name}-{}", std::process::id()));
        for (file, m) in mats {
            zestengine_core::save_physmat(dir.join(file), m).unwrap();
        }
        dir
    }

    /// How far (m) a box set on a 26 degree slope slides in two seconds,
    /// both slope and box of material `m`.
    fn slide_on_slope(name: &str, m: PhysicsMaterial) -> f32 {
        let dir = material_dir(name, &[("m.physmat", m)]);
        let mut scene = LoadedScene::empty();
        let tilt = glam::Quat::from_rotation_z(26f32.to_radians());
        let slope = obj(&mut scene, Vec3::ZERO, vec![Box::new(BoxCollider { size: [20.0, 1.0, 4.0], material: "m.physmat".into(), ..Default::default() })]);
        let si = scene.index_of(slope).unwrap();
        scene.objects[si].rotation = tilt;
        let at = tilt * Vec3::new(0.0, 0.5 + 0.25, 0.0);
        let block = obj(
            &mut scene,
            at,
            vec![
                Box::new(BoxCollider { size: [0.5, 0.5, 0.5], ..Default::default() }),
                Box::new(RbComponent { mass: 2.0, material: "m.physmat".into(), ..Default::default() }),
            ],
        );
        let bi = scene.index_of(block).unwrap();
        scene.objects[bi].rotation = tilt;
        let mut pw = PhysicsWorld::build_with(&scene, Some(&dir), &PhysicsSettings::default());
        run(&mut pw, &mut scene, 120);
        (scene.world_transform(bi).w_axis.truncate() - at).length()
    }

    #[test]
    fn a_switched_off_component_is_ignored_and_survives_a_save() {
        let mut scene = LoadedScene::empty();
        floor(&mut scene);
        let ball = obj(&mut scene, Vec3::new(0.0, 3.0, 0.0), vec![Box::new(SphereCollider::default()), Box::new(RbComponent::default())]);
        let i = scene.index_of(ball).unwrap();
        // Rigid Body off: the ball is a fixed collider and stays put.
        assert!(zestengine_core::set_component_enabled(&mut scene.objects[i].components[1], false));
        let mut pw = PhysicsWorld::build(&scene);
        run(&mut pw, &mut scene, 60);
        assert!((scene.world_transform(i).w_axis.y - 3.0).abs() < 1e-4, "a switched-off Rigid Body doesn't fall");
        // Snapshots keep it off, settings and all.
        let saved = scene.objects[i].save_components();
        assert!(!saved[1].enabled && saved[0].enabled);
        let registry = zestengine_core::ComponentRegistry::with_builtins();
        scene.objects[i].load_components(&saved, &registry);
        assert!(!zestengine_core::component_enabled(scene.objects[i].components[1].as_ref()));
        // Back on, it falls.
        assert!(zestengine_core::set_component_enabled(&mut scene.objects[i].components[1], true));
        let mut pw = PhysicsWorld::build(&scene);
        run(&mut pw, &mut scene, 60);
        assert!(scene.world_transform(i).w_axis.y < 2.5, "switched back on, it falls");
    }

    #[test]
    fn static_friction_holds_what_sliding_friction_would_let_go() {
        // tan 26 = 0.49: more than 0.4 sliding friction, less than 0.6 static.
        let sliding = PhysicsMaterial { friction: 0.4, static_friction: None, ..Default::default() };
        let holding = PhysicsMaterial { static_friction: Some(0.6), ..sliding };
        let slid = slide_on_slope("slides", sliding);
        let held = slide_on_slope("holds", holding);
        assert!(slid > 0.5, "0.4 alone lets it slide: {slid}");
        assert!(held < 0.05, "0.6 static holds it: {held}");
    }

    fn peak_after_bounce(material: &str, dir: &Path) -> f32 {
        let mut scene = LoadedScene::empty();
        floor(&mut scene);
        let ball = obj(
            &mut scene,
            Vec3::new(0.0, 3.0, 0.0),
            vec![Box::new(SphereCollider::default()), Box::new(RbComponent { material: material.into(), ..Default::default() })],
        );
        let mut pw = PhysicsWorld::build_with(&scene, Some(dir), &PhysicsSettings::default());
        let y = |scene: &LoadedScene| scene.world_transform(scene.index_of(ball).unwrap()).w_axis.y;
        let mut landed = false;
        let mut peak = f32::MIN;
        for _ in 0..240 {
            let before = y(&scene);
            run(&mut pw, &mut scene, 1);
            let now = y(&scene);
            if !landed && now >= before && now < 1.2 {
                landed = true;
            }
            if landed {
                peak = peak.max(now);
            }
        }
        peak
    }

    #[test]
    fn a_bouncy_material_bounces_and_the_body_slot_applies_it() {
        let rubber = PhysicsMaterial { bounciness: 0.9, bounce_combine: CombineMode::Maximum, ..Default::default() };
        let dir = material_dir("bounce", &[("rubber.physmat", rubber), ("clay.physmat", PhysicsMaterial::default())]);
        let bouncy = peak_after_bounce("rubber.physmat", &dir);
        let dead = peak_after_bounce("clay.physmat", &dir);
        assert!(bouncy > 2.0, "rubber comes most of the way back up: {bouncy}");
        assert!(dead < 1.1, "a dead material stays on the floor: {dead}");
    }

    #[test]
    fn density_sets_mass_and_buoyancy_scales_lift() {
        let light = PhysicsMaterial { density: 100.0, buoyancy: 0.25, ..Default::default() };
        let dir = material_dir("density", &[("light.physmat", light)]);
        let mut scene = LoadedScene::empty();
        obj(&mut scene, Vec3::ZERO, vec![Box::new(zestengine_core::Water { waves: Vec::new(), ..Default::default() })]);
        let id = obj(
            &mut scene,
            Vec3::new(0.0, 2.0, 0.0),
            vec![
                Box::new(BoxCollider { size: [1.0, 1.0, 1.0], ..Default::default() }),
                Box::new(RbComponent { mass: 0.0, material: "light.physmat".into(), ..Default::default() }),
                Box::new(zestengine_core::Buoyancy { buoyancy: 2.0, ..Default::default() }),
            ],
        );
        let mut pw = PhysicsWorld::build_with(&scene, Some(&dir), &PhysicsSettings::default());
        let mass = pw.rigid_bodies[pw.bodies[&id].handle].mass();
        assert!((mass - 100.0).abs() < 1.0, "1 m3 at 100 kg/m3: {mass}");
        run_water(&mut pw, &mut scene, 900);
        let y = scene.world_transform(scene.index_of(id).unwrap()).w_axis.y;
        assert!(y < -1.0, "2x lift scaled by 0.25 is less than its weight, so it sinks: {y}");
    }

    #[test]
    fn any_rigid_body_floats_by_its_density_and_splashes_in() {
        let mut scene = LoadedScene::empty();
        obj(&mut scene, Vec3::ZERO, vec![Box::new(zestengine_core::Water { waves: Vec::new(), ..Default::default() })]);
        let body = |mass: f32| -> Vec<Box<dyn Component>> {
            vec![Box::new(BoxCollider { size: [1.0, 1.0, 1.0], ..Default::default() }), Box::new(RbComponent { mass, ..Default::default() })]
        };
        // 1 m3 boxes: half the density of water, and heavier than water.
        let wood = obj(&mut scene, Vec3::new(0.0, 3.0, 0.0), body(500.0));
        let iron = obj(&mut scene, Vec3::new(4.0, 3.0, 0.0), body(3000.0));
        let mut pw = PhysicsWorld::build(&scene);
        let mut splashes = 0;
        for k in 0..900 {
            pw.sync_structure(&scene);
            let ids = scene.id_index();
            let world = scene.world_transforms();
            pw.push_poses(&scene, &ids, &world);
            splashes += pw.apply_water(&scene, &world, k as f32 * FIXED_DT, &[]).splashes.len();
            pw.step();
            pw.reset_forces();
            pw.write_back(&mut scene, &ids, 1.0);
        }
        let y = |id: ObjectId| scene.world_transform(scene.index_of(id).unwrap()).w_axis.y;
        assert!(y(wood).abs() < 0.15, "half the density of water floats half under: {}", y(wood));
        assert!(y(iron) < -1.0, "denser than water sinks: {}", y(iron));
        assert!(splashes >= 2, "both hit the water hard enough to splash: {splashes}");
    }

    /// A closed box mesh between `lo` and `hi`.
    fn box_mesh(lo: Vec3, hi: Vec3, base: u32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let v: Vec<Vec3> = (0..8).map(|c| Vec3::new(if c & 1 == 0 { lo.x } else { hi.x }, if c & 2 == 0 { lo.y } else { hi.y }, if c & 4 == 0 { lo.z } else { hi.z })).collect();
        let quads = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]];
        let t = quads.iter().flat_map(|q| [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]).map(|t: [u32; 3]| t.map(|i| i + base)).collect();
        (v, t)
    }

    #[test]
    fn a_concave_mesh_splits_into_pieces_and_bad_input_falls_back() {
        // An L: a seat and a back, like a chair.
        let (mut v, mut t) = box_mesh(Vec3::ZERO, Vec3::new(2.0, 0.5, 1.0), 0);
        let (v2, t2) = box_mesh(Vec3::new(0.0, 0.5, 0.0), Vec3::new(0.4, 2.0, 1.0), 8);
        v.extend(v2);
        t.extend(t2);
        let parts = convex_pieces(&v, &t, zestengine_core::ColliderDetail::Medium);
        let (hull_points, max_pieces, piece_points, _) = budgets(zestengine_core::ColliderDetail::Medium);
        assert!(parts.len() >= 2, "the seat and back are separate pieces: {}", parts.len());
        assert!(hull_of(&[Vec3::ZERO, Vec3::X, Vec3::Z], hull_points).is_none(), "a flat triangle has no hull");
        assert!(hull_of(&v, hull_points).is_some());
        assert!(parts.len() <= max_pieces as usize && parts.iter().all(|(p, _)| p.len() <= piece_points));
        // A dense ball's hull is cut down to the corner budget.
        let ball: Vec<Vec3> = (0..2000).map(|k| {
            let (a, b) = (k as f32 * 2.399, (k as f32 / 2000.0) * 2.0 - 1.0);
            let r = (1.0 - b * b).sqrt();
            Vec3::new(r * a.cos(), b, r * a.sin())
        }).collect();
        let (hp, _) = hull_of(&ball, hull_points).unwrap();
        assert!(hp.len() <= hull_points, "{}", hp.len());
    }

    #[test]
    fn ball_meshes_are_spheres_and_boxes_are_not() {
        let ball: Vec<Vec3> = (0..400).map(|k| {
            let (a, b) = (k as f32 * 2.399, (k as f32 / 400.0) * 2.0 - 1.0);
            let r = (1.0 - b * b).sqrt();
            Vec3::new(1.0 + r * a.cos() * 0.5, b * 0.5, r * a.sin() * 0.5)
        }).collect();
        let (c, r) = ball_of(&ball).expect("a ball");
        assert!(c.distance(Vec3::X) < 0.02 && (r - 0.5).abs() < 0.01);
        let (v, _) = box_mesh(Vec3::ZERO, Vec3::ONE, 0);
        assert!(ball_of(&v).is_none());
    }

    /// A loaded "couch" model (one render mesh) with three authored pieces
    /// on it and one unmatched piece on the model root; returns the couch's
    /// and the root's object index.
    fn authored_couch(scene: &mut LoadedScene, components: Vec<Box<dyn Component>>) -> (usize, usize) {
        use zestengine_assets::{CollisionKind, CollisionPiece, Instance};
        let (seat, _) = box_mesh(Vec3::new(-1.0, 0.0, -0.5), Vec3::new(1.0, 0.5, 0.5), 0);
        let (back, _) = box_mesh(Vec3::new(-1.0, 0.5, 0.3), Vec3::new(1.0, 1.2, 0.5), 0);
        let ball: Vec<Vec3> = (0..64).map(|k| Vec3::new(1.2, 0.8, 0.0) + Vec3::new((k as f32).cos(), (k as f32 * 0.7).sin(), (k as f32 * 1.3).sin()).normalize() * 0.2).collect();
        let piece = |name: &str, kind, target, points: Vec<Vec3>| CollisionPiece { name: name.into(), kind, target, points };
        let model = zestengine_assets::Scene {
            meshes: Vec::new(),
            morphs: Vec::new(),
            springs: Default::default(),
            textures: Vec::new(),
            materials: Vec::new(),
            instances: vec![Instance::single("Couch", 0, 0, Mat4::IDENTITY)],
            skeletons: Vec::new(),
            animations: Vec::new(),
            collision: vec![
                piece("UCX_Couch_00", CollisionKind::Convex, Some(0), seat),
                piece("UBX_Couch_01", CollisionKind::Box, Some(0), back),
                piece("USP_Couch_02", CollisionKind::Sphere, Some(0), ball),
                piece("UBX_Floor", CollisionKind::Box, None, box_mesh(Vec3::splat(-2.0), Vec3::new(2.0, -1.9, 2.0), 0).0),
            ],
            identity: Default::default(),
        };
        scene.authored_collision.register(&model, 0, Some("models/couch.glb"));
        obj(scene, Vec3::ZERO, Vec::new());
        let root = scene.objects.len() - 1;
        obj(scene, Vec3::new(0.0, 3.0, 0.0), components);
        let couch = scene.objects.len() - 1;
        let o = &mut scene.objects[couch];
        o.parent = Some(root);
        o.render = Some(crate::scene::RenderInfo { mesh: 0, material: 0 });
        o.source = ObjectSource::Model { path: "models/couch.glb".into(), mesh: 0, extra_meshes: Vec::new(), model: None, mesh_id: None, extra_mesh_ids: Vec::new() };
        (couch, root)
    }

    #[test]
    fn authored_pieces_become_the_mesh_collider_compound() {
        let mut scene = LoadedScene::empty();
        let (couch, root) = authored_couch(&mut scene, vec![Box::new(MeshCollider::default()), Box::new(RbComponent::default())]);
        let mc = MeshCollider { convex: true, ..Default::default() };
        assert_eq!(mesh_shape_mode(&scene, couch, &mc, true), MeshShapeMode::Authored);
        assert_eq!(mesh_shape_mode(&scene, couch, &MeshCollider { use_authored: false, ..mc }, true), MeshShapeMode::Hull(mc.detail));
        let shape = mesh_collider_shape(&scene, couch, MeshShapeMode::Authored).expect("authored shape");
        let MeshColliderShape::Authored(pieces) = &*shape else { panic!("not authored: {shape:?}") };
        use crate::authored_collision::ColliderPiece;
        assert!(matches!(pieces[..], [ColliderPiece::Hull(..), ColliderPiece::Cuboid { .. }, ColliderPiece::Ball(..)]), "{pieces:?}");
        assert!(!mesh_collider_wire_for(&scene, couch, MeshShapeMode::Authored).unwrap().is_empty());

        let shapes = collider_shapes(&scene, couch, Vec3::splat(2.0), BodyKind::Dynamic);
        assert_eq!(shapes.len(), 1);
        let built = shapes.into_iter().next().unwrap().0.build();
        assert_eq!(built.shape().as_compound().map(|c| c.shapes().len()), Some(3));

        // The model root owns the piece that matched no mesh.
        assert_eq!(crate::authored_collision::authored_pieces(&scene, root).map(|p| p.len()), Some(1));
        assert!(matches!(&*mesh_collider_shape(&scene, root, MeshShapeMode::Authored).unwrap(), MeshColliderShape::Authored(p) if p.len() == 1));

        // The body's volume (what buoyancy uses) is the pieces' total.
        let mut pw = PhysicsWorld::build(&scene);
        run(&mut pw, &mut scene, 1);
        let (_, volume) = pw.mass_and_volume(scene.objects[couch].id).unwrap();
        let expected = 1.0 + 2.0 * 0.7 * 0.2 + 4.0 / 3.0 * std::f32::consts::PI * 0.2f32.powi(3);
        assert!((volume - expected).abs() < 0.02 * expected, "{volume} vs {expected}");
    }

    #[test]
    fn collider_wires_share_edges() {
        // A flat quad: its four sides, not the diagonal across it.
        let quad = MeshColliderShape::Triangles(vec![Vec3::ZERO, Vec3::X, Vec3::new(1.0, 1.0, 0.0), Vec3::Y], vec![[0, 1, 2], [0, 2, 3]]);
        assert_eq!(mesh_collider_wire(&quad).len(), 4);
        // A box: its 12 edges, not the 6 face diagonals.
        let (v, t) = box_mesh(Vec3::ZERO, Vec3::ONE, 0);
        assert_eq!(mesh_collider_wire(&MeshColliderShape::Triangles(v, t)).len(), 12);
    }

    #[test]
    fn a_very_light_body_bobs_up_without_being_launched() {
        let mut scene = LoadedScene::empty();
        obj(&mut scene, Vec3::ZERO, vec![Box::new(zestengine_core::Water { waves: Vec::new(), ..Default::default() })]);
        let id = obj(
            &mut scene,
            Vec3::new(0.0, -0.2, 0.0),
            vec![Box::new(BoxCollider { size: [2.0, 1.0, 1.0], ..Default::default() }), Box::new(RbComponent { mass: 1.0, ..Default::default() })],
        );
        let mut pw = PhysicsWorld::build(&scene);
        let mut highest = f32::MIN;
        for k in 0..600 {
            pw.sync_structure(&scene);
            let ids = scene.id_index();
            let world = scene.world_transforms();
            pw.push_poses(&scene, &ids, &world);
            pw.apply_water(&scene, &world, k as f32 * FIXED_DT, &[]);
            pw.step();
            pw.reset_forces();
            pw.write_back(&mut scene, &ids, 1.0);
            highest = highest.max(scene.world_transform(scene.index_of(id).unwrap()).w_axis.y);
        }
        let y = scene.world_transform(scene.index_of(id).unwrap()).w_axis.y;
        assert!(highest < 1.5, "never flung far out: {highest}");
        assert!(y > 0.0 && y < 0.8, "settles riding high on the surface: {y}");
    }

    #[test]
    fn a_post_through_the_surface_blocks_ripples() {
        let mut scene = LoadedScene::empty();
        let water = obj(&mut scene, Vec3::ZERO, vec![Box::new(zestengine_core::Water { waves: Vec::new(), ..Default::default() })]);
        obj(&mut scene, Vec3::new(1.0, 0.0, 0.0), vec![Box::new(CapsuleCollider { radius: 0.3, height: 4.0, ..Default::default() })]);
        let pw = PhysicsWorld::build(&scene);
        let sim = zestengine_core::RippleSim::new(64, 16.0);
        let w = scene.world_transform(scene.index_of(water).unwrap());
        let mask = pw.ripple_blockers(w, &sim, u32::MAX);
        let cell_of = |x: f32, z: f32| {
            let c = ((x + 8.0) / 0.25) as usize;
            let r = ((z + 8.0) / 0.25) as usize;
            mask[r * 64 + c]
        };
        assert!(cell_of(1.0, 0.0), "the post's cell is solid");
        assert!(!cell_of(0.0, 0.0) && !cell_of(2.0, 2.0), "open water elsewhere");
        assert!(mask.iter().filter(|b| **b).count() < 12, "only the post's footprint");
    }

    #[test]
    fn a_crate_floats_half_under_and_a_heavy_one_sinks() {
        let mut scene = LoadedScene::empty();
        obj(&mut scene, Vec3::ZERO, vec![Box::new(zestengine_core::Water { waves: Vec::new(), ..Default::default() })]);
        let floater = crate_box(&mut scene, Vec3::new(0.0, 2.0, 0.0), 2.0);
        let sinker = crate_box(&mut scene, Vec3::new(4.0, 2.0, 0.0), 0.5);
        let mut pw = PhysicsWorld::build(&scene);
        run_water(&mut pw, &mut scene, 900);
        let y = |id: ObjectId| scene.world_transform(scene.index_of(id).unwrap()).w_axis.y;
        assert!(y(floater).abs() < 0.15, "rests with the waterline at its middle: {}", y(floater));
        assert!(y(sinker) < -1.0, "a body lighter than its weight in lift sinks: {}", y(sinker));
    }

    /// Height of a 0.5 m ball of `mass` kg dropped at the surface of a fluid
    /// of `density` and `viscosity` after `secs`.
    fn ball_in_fluid(density: f32, viscosity: f32, mass: f32, secs: f32) -> f32 {
        let mut scene = LoadedScene::empty();
        obj(&mut scene, Vec3::ZERO, vec![Box::new(zestengine_core::Water { waves: Vec::new(), density, viscosity, depth: 50.0, ..Default::default() })]);
        let ball = obj(&mut scene, Vec3::new(0.0, -0.3, 0.0), vec![Box::new(SphereCollider { radius: 0.25, ..Default::default() }), Box::new(RbComponent { mass, ..Default::default() })]);
        let mut pw = PhysicsWorld::build(&scene);
        run_water(&mut pw, &mut scene, (secs / FIXED_DT) as usize);
        scene.world_transform(scene.index_of(ball).unwrap()).w_axis.y
    }

    #[test]
    fn thick_fluids_slow_sinking_and_density_sets_what_floats() {
        // 0.065 m3 ball of 100 kg (1530 kg/m3) sinks in all three; the
        // thicker the fluid, the slower.
        let in_water = ball_in_fluid(1000.0, 0.001, 100.0, 1.0);
        let in_honey = ball_in_fluid(1420.0, 10.0, 100.0, 1.0);
        let in_mud = ball_in_fluid(1700.0, 50.0, 100.0, 1.0);
        assert!(in_water < -0.8, "sinks through water: {in_water}");
        assert!(in_honey > in_water + 0.4 && in_mud > in_honey, "thicker holds it back more: water {in_water}, honey {in_honey}, mud {in_mud}");
        // 80 kg (1220 kg/m3) sinks in water but floats in honey (1420).
        assert!(ball_in_fluid(1000.0, 0.001, 80.0, 6.0) < -1.0);
        assert!(ball_in_fluid(1420.0, 10.0, 80.0, 6.0) > -0.6, "denser fluid holds it up");
    }

    fn gpu_water(scene: &mut LoadedScene) -> ObjectId {
        let mut water = zestengine_core::Water { waves: Vec::new(), ..Default::default() };
        water.ripples = Some(zestengine_core::RippleSim::gpu_owned(64, 24.0));
        obj(scene, Vec3::ZERO, vec![Box::new(water)])
    }

    fn water_mut(scene: &mut LoadedScene, id: ObjectId) -> &mut zestengine_core::Water {
        let i = scene.index_of(id).unwrap();
        scene.objects[i].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<zestengine_core::Water>()).unwrap()
    }

    #[test]
    fn gpu_water_is_fed_bodies_and_fixed_obstacles() {
        let mut scene = LoadedScene::empty();
        let water = gpu_water(&mut scene);
        crate_box(&mut scene, Vec3::new(-2.0, 0.0, 0.0), 2.0);
        obj(&mut scene, Vec3::new(3.0, 0.0, 0.0), vec![Box::new(CapsuleCollider { radius: 0.3, height: 4.0, ..Default::default() })]);
        let mut pw = PhysicsWorld::build(&scene);
        let world = scene.world_transforms();
        let step = pw.apply_water(&scene, &world, 0.0, &[]);
        let wi = scene.index_of(water).unwrap();
        assert_eq!(step.displacers.len(), 1, "the crate, not the fixed post");
        let (i, d) = step.displacers[0];
        assert_eq!(i, wi);
        assert!((d.center - glam::Vec2::new(-2.0, 0.0)).length() < 1e-3);
        assert!((d.thickness - 0.5).abs() < 0.02 && (d.half.x - 0.5).abs() < 0.02 && !d.round, "{d:?}");
        let w = water_mut(&mut scene, water);
        let window = zestengine_core::choose_window(w, glam::Vec2::ZERO, None);
        w.sim_feed.window = Some(window);
        step_ripples(&mut scene, &world, Vec3::ZERO, &step, Some(&pw), 0.0, FIXED_DT);
        crate::water_walls::WaterWalls::default().update(&mut scene, &world);
        let w = water_mut(&mut scene, water);
        assert_eq!(w.sim_feed.displacers.len(), 1);
        let (bw, mask) = w.sim_feed.blocked.clone().expect("obstacle cells");
        assert_eq!(bw, window);
        let cell_at = |x: f32| {
            let c = ((glam::Vec2::new(x, 0.0) - window.lo()) / window.cell).floor();
            mask[c.y as usize * window.n as usize + c.x as usize]
        };
        assert!(cell_at(3.0) && !cell_at(-2.0) && !cell_at(0.0), "the post blocks, the floating crate does not");
        assert!(w.ripples.as_ref().unwrap().gpu, "still owned by the GPU simulation");
    }

    #[test]
    fn a_water_fits_the_basin_around_it() {
        let mut scene = LoadedScene::empty();
        let water = obj(&mut scene, Vec3::ZERO, vec![Box::new(zestengine_core::Water { body: zestengine_core::WaterBody::Pool, waves: Vec::new(), ..Default::default() })]);
        // A 4 x 4 m pool, 1 m deep: four walls through the surface, a floor.
        for (at, size) in [
            (Vec3::new(2.25, 0.0, 0.0), [0.5, 3.0, 5.0]),
            (Vec3::new(-2.25, 0.0, 0.0), [0.5, 3.0, 5.0]),
            (Vec3::new(0.0, 0.0, 2.25), [5.0, 3.0, 0.5]),
            (Vec3::new(0.0, 0.0, -2.25), [5.0, 3.0, 0.5]),
            (Vec3::new(0.0, -1.25, 0.0), [5.0, 0.5, 5.0]),
        ] {
            obj(&mut scene, at, vec![Box::new(BoxCollider { size, ..Default::default() })]);
        }
        let world = scene.world_transforms();
        let wi = scene.index_of(water).unwrap();
        let (outline, depth) = crate::water_walls::fit_basin(&scene, wi, &world).expect("walls around it");
        assert!((depth.unwrap() - 1.0).abs() < 0.02, "the floor 1 m down: {depth:?}");
        assert!(outline.len() >= 4 && outline.len() <= 16, "a square needs few points: {}", outline.len());
        for p in &outline {
            let m = p[0].abs().max(p[1].abs());
            assert!((m - 2.0).abs() < 0.03, "on the walls' inner faces: {p:?}");
        }
    }

    #[test]
    fn the_floor_under_the_water_sets_its_depth_and_dry_ground_is_shore() {
        let mut scene = LoadedScene::empty();
        let water = gpu_water(&mut scene);
        // A shelf whose top is 0.4 m under the surface, a beach 0.2 m above
        // it and a bank 1 m above it.
        obj(&mut scene, Vec3::new(-3.0, -1.4, 0.0), vec![Box::new(BoxCollider { size: [2.0, 2.0, 2.0], ..Default::default() })]);
        obj(&mut scene, Vec3::new(3.0, -0.8, 0.0), vec![Box::new(BoxCollider { size: [2.0, 2.0, 2.0], ..Default::default() })]);
        obj(&mut scene, Vec3::new(6.0, 0.0, 0.0), vec![Box::new(BoxCollider { size: [2.0, 2.0, 2.0], ..Default::default() })]);
        let w = water_mut(&mut scene, water);
        w.sim_feed.window = Some(zestengine_core::choose_window(w, glam::Vec2::ZERO, None));
        let world = scene.world_transforms();
        crate::water_walls::WaterWalls::default().update(&mut scene, &world);
        let w = water_mut(&mut scene, water);
        let window = w.sim_feed.window.unwrap();
        let (_, bed) = w.sim_feed.bed.clone().expect("floor depths");
        let at = |x: f32| {
            let c = ((glam::Vec2::new(x, 0.0) - window.lo()) / window.cell).floor();
            bed[c.y as usize * window.n as usize + c.x as usize]
        };
        assert!((at(-3.0) - 0.4).abs() < 0.02, "the shelf: {}", at(-3.0));
        assert!(at(3.0) < 0.0, "the beach stands out of the water: {}", at(3.0));
        assert!(at(0.0).is_nan(), "open water keeps its own depth");
        let field = zestengine_core::SimField::build_with_bed(w, window, None, Some(&bed));
        let cell = |x: f32| {
            let c = ((glam::Vec2::new(x, 0.0) - window.lo()) / window.cell).floor();
            field.cells[c.y as usize * window.n as usize + c.x as usize]
        };
        assert!((cell(-3.0)[0] - 0.4).abs() < 0.02 && cell(-3.0)[3] == zestengine_core::CELL_WATER);
        assert!(cell(3.0)[0] < 0.0 && cell(3.0)[3] == zestengine_core::CELL_WATER, "low ground is a beach the water can run up");
        assert_eq!(cell(6.0)[3], zestengine_core::CELL_SHORE, "a high bank is shore");
    }

    #[test]
    fn walls_are_still_colliders_on_layers_the_water_meets() {
        let mut scene = LoadedScene::empty();
        let water = gpu_water(&mut scene);
        let post = obj(&mut scene, Vec3::new(3.0, 0.0, 0.0), vec![Box::new(CapsuleCollider { radius: 0.3, height: 4.0, ..Default::default() })]);
        let ghost = obj(&mut scene, Vec3::new(-3.0, 0.0, 0.0), vec![Box::new(BoxCollider { size: [1.0, 4.0, 1.0], ..Default::default() })]);
        // The ghost's layer doesn't collide with the water's.
        let gi = scene.index_of(ghost).unwrap();
        scene.objects[gi].layer = 5;
        scene.layers.collision[0] &= !(1 << 5);
        scene.layers.collision[5] &= !1;
        let w = water_mut(&mut scene, water);
        w.sim_feed.window = Some(zestengine_core::choose_window(w, glam::Vec2::ZERO, None));
        let mut walls = crate::water_walls::WaterWalls::default();
        let cell_at = |scene: &mut LoadedScene, x: f32| {
            let w = water_mut(scene, water);
            let window = w.sim_feed.window.unwrap();
            let (_, mask) = w.sim_feed.blocked.clone().expect("wall cells");
            let c = ((glam::Vec2::new(x, 0.0) - window.lo()) / window.cell).floor();
            mask[c.y as usize * window.n as usize + c.x as usize]
        };
        let world = scene.world_transforms();
        walls.update(&mut scene, &world);
        assert!(cell_at(&mut scene, 3.0), "the post is a wall");
        assert!(!cell_at(&mut scene, -3.0), "the ghost's layer passes through the water");
        // Moved: it pushes the water while moving, a wall again once still.
        let pi = scene.index_of(post).unwrap();
        scene.objects[pi].translation.x = 2.0;
        let world = scene.world_transforms();
        walls.update(&mut scene, &world);
        assert!(!cell_at(&mut scene, 2.0), "moving, not a wall yet");
        for _ in 0..12 {
            walls.update(&mut scene, &world);
        }
        assert!(cell_at(&mut scene, 2.0) && !cell_at(&mut scene, 3.0), "settled where it stopped");
    }

    #[test]
    fn simulated_flow_carries_a_floating_body() {
        let mut scene = LoadedScene::empty();
        let water = gpu_water(&mut scene);
        let floater = crate_box(&mut scene, Vec3::ZERO, 2.0);
        let sim = water_mut(&mut scene, water).ripples.as_mut().unwrap();
        sim.set_field(64, 24.0, glam::Vec2::ZERO, &vec![[0.0, 1.0, 0.0, 0.0]; 64 * 64]);
        let mut pw = PhysicsWorld::build(&scene);
        run_water(&mut pw, &mut scene, 300);
        let x = scene.world_transform(scene.index_of(floater).unwrap()).w_axis.x;
        assert!(x > 1.5, "the moving water dragged it along: {x}");
    }

    #[test]
    fn current_carries_a_floating_body() {
        let mut scene = LoadedScene::empty();
        obj(&mut scene, Vec3::ZERO, vec![Box::new(zestengine_core::Water { waves: Vec::new(), current: [1.0, 0.0, 0.0], ..Default::default() })]);
        let floater = crate_box(&mut scene, Vec3::new(0.0, 0.0, 0.0), 2.0);
        let mut pw = PhysicsWorld::build(&scene);
        run_water(&mut pw, &mut scene, 300);
        let x = scene.world_transform(scene.index_of(floater).unwrap()).w_axis.x;
        assert!(x > 1.5, "drifted downstream: {x}");
    }

    #[test]
    fn a_capsule_rests_on_the_floor_at_its_radius() {
        let mut scene = LoadedScene::empty();
        floor(&mut scene);
        let pill = obj(
            &mut scene,
            Vec3::new(0.0, 3.0, 0.0),
            vec![
                Box::new(CapsuleCollider { direction: ColliderAxis::X, radius: 0.25, height: 2.0, ..Default::default() }),
                Box::new(RbComponent::default()),
            ],
        );
        let mut pw = PhysicsWorld::build(&scene);
        run(&mut pw, &mut scene, 240);
        let y = scene.world_transform(scene.index_of(pill).unwrap()).w_axis.y;
        assert!((y - (0.5 + 0.25)).abs() < 0.05, "lying on its side on the floor top: {y}");
        let hit = pw.raycast(Vec3::new(0.9, 5.0, 0.0), Vec3::NEG_Y, 10.0, &RayFilter::default()).unwrap();
        // The cap starts 0.75 m out; 0.9 m out its top is sqrt(0.25^2 - 0.15^2) = 0.2 above the axis.
        assert!((hit.point.y - 0.95).abs() < 0.02, "the rounded end is 0.9 m out along X: {}", hit.point.y);
    }

    #[test]
    fn queries_on_empty_world_are_safe() {
        let scene = LoadedScene::empty();
        let pw = PhysicsWorld::build(&scene);
        let f = RayFilter::default();
        assert!(pw.raycast(Vec3::ZERO, Vec3::NEG_Y, 100.0, &f).is_none());
        assert!(pw.overlap_sphere(Vec3::ZERO, 5.0, &f).is_empty());
        assert!(pw.raycast(Vec3::ZERO, Vec3::ZERO, 100.0, &f).is_none());
    }

    #[test]
    fn dynamic_body_lands_on_floor_and_reports_contact() {
        let mut scene = LoadedScene::empty();
        let floor_id = floor(&mut scene);
        let ball = dynamic_ball(&mut scene, Vec3::new(0.0, 3.0, 0.0));
        let mut pw = PhysicsWorld::build(&scene);
        let events = run(&mut pw, &mut scene, 180);
        let y = scene.objects[1].translation.y;
        assert!((y - 1.0).abs() < 0.05, "ball rests on the floor top (y=1), got {y}");
        assert!(events.iter().any(|e| e.started
            && !e.trigger
            && ((e.a == ball && e.b == floor_id) || (e.a == floor_id && e.b == ball))));
    }

    #[test]
    fn raycast_respects_exclude_and_layers() {
        let mut scene = LoadedScene::empty();
        let floor_id = floor(&mut scene);
        let mut pw = PhysicsWorld::build(&scene);
        run(&mut pw, &mut scene, 1);
        let from = Vec3::new(0.0, 5.0, 0.0);
        let hit = pw
            .raycast(from, Vec3::NEG_Y, 100.0, &RayFilter::default())
            .expect("hits floor");
        assert_eq!(hit.object, floor_id);
        assert!((hit.point.y - 0.5).abs() < 1e-3);
        assert!(hit.normal.y > 0.99);
        let excluded = RayFilter {
            exclude: Some(floor_id),
            ..Default::default()
        };
        assert!(pw.raycast(from, Vec3::NEG_Y, 100.0, &excluded).is_none());
        let other_layer = RayFilter {
            layer_mask: 1 << 3,
            ..Default::default()
        };
        assert!(pw.raycast(from, Vec3::NEG_Y, 100.0, &other_layer).is_none());
        assert_eq!(pw.overlap_sphere(Vec3::new(0.0, 0.8, 0.0), 0.5, &RayFilter::default()), vec![floor_id]);
    }

    #[test]
    fn trigger_reports_enter_without_blocking() {
        let mut scene = LoadedScene::empty();
        floor(&mut scene);
        let zone = obj(
            &mut scene,
            Vec3::new(0.0, 2.0, 0.0),
            vec![Box::new(BoxCollider {
                size: [2.0, 1.0, 2.0],
                is_trigger: true,
                ..Default::default()
            })],
        );
        let ball = dynamic_ball(&mut scene, Vec3::new(0.0, 4.0, 0.0));
        let mut pw = PhysicsWorld::build(&scene);
        let events = run(&mut pw, &mut scene, 180);
        assert!(events.iter().any(|e| e.trigger && e.started && (e.a == zone || e.b == zone)));
        assert!(events.iter().any(|e| e.trigger && !e.started && (e.a == ball || e.b == ball)));
        assert!(scene.objects[2].translation.y < 1.1, "fell through the trigger onto the floor");
    }

    #[test]
    fn character_capsule_can_opt_out_of_water_without_losing_wall_collision() {
        let mut scene = LoadedScene::empty();
        gpu_water(&mut scene);
        let player = obj(&mut scene, Vec3::new(0.0, -0.8, 0.0), vec![Box::new(Character { capsule_affects_water: false, ..Default::default() })]);
        let i = scene.index_of(player).unwrap();
        let mut pw = PhysicsWorld::build(&scene);
        let step = pw.apply_water(&scene, &scene.world_transforms(), 0.0, &[]);
        assert!(step.displacers.is_empty() && step.pushes.is_empty() && step.wakes.is_empty() && step.splashes.is_empty());
        assert!(pw.bodies[&player].capsule.is_some(), "world collision remains enabled");
        scene.objects[i].components[0].as_any_mut().downcast_mut::<Character>().unwrap().capsule_affects_water = true;
        assert_eq!(pw.apply_water(&scene, &scene.world_transforms(), 0.0, &[]).displacers.len(), 1);
        scene.objects[i].components[0].as_any_mut().downcast_mut::<Character>().unwrap().capsule_affects_water = false;
        // Separate bone-driven kinematic leg bodies use the same path in desktop and VR.
        for x in [-0.2, 0.2] {
            obj(&mut scene, Vec3::new(x, -0.1, 0.0), vec![
                Box::new(CapsuleCollider { radius: 0.08, height: 0.5, ..Default::default() }),
                Box::new(RbComponent { kind: BodyKind::Kinematic, ..Default::default() }),
            ]);
        }
        pw.mark_dirty(); pw.sync_structure(&scene);
        let step = pw.apply_water(&scene, &scene.world_transforms(), 0.0, &[]);
        assert_eq!(step.displacers.len(), 2, "two legs, no combined body footprint");
        assert!(step.displacers[0].1.center.distance(step.displacers[1].1.center) > 0.3);
    }

    #[test]
    fn character_default_capsule_can_be_disabled_for_custom_colliders() {
        let mut scene = LoadedScene::empty();
        let player = obj(&mut scene, Vec3::ZERO, vec![
            Box::new(Character { use_default_capsule: false, ..Default::default() }),
            Box::new(SphereCollider { radius: 0.1, ..Default::default() }),
        ]);
        let pw = PhysicsWorld::build(&scene);
        let body = &pw.bodies[&player];
        assert!(body.capsule.is_none() && body.character.is_none());
        assert_eq!(pw.rigid_bodies[body.handle].colliders().len(), 1, "authored custom collider survives");
    }

    #[test]
    fn character_tracking_offset_sweeps_instead_of_teleporting_through_walls() {
        let mut scene = LoadedScene::empty();
        floor(&mut scene);
        obj(&mut scene, Vec3::new(1.0, 2.0, 0.0), vec![Box::new(BoxCollider { size: [0.1, 4.0, 4.0], ..Default::default() })]);
        let player = obj(&mut scene, Vec3::new(0.0, 0.5, 0.0), vec![Box::new(Character::default())]);
        let i = scene.index_of(player).unwrap();
        let mut pw = PhysicsWorld::build(&scene);
        run(&mut pw, &mut scene, 1);
        scene.objects[i].components[0].as_any_mut().downcast_mut::<Character>().unwrap().center[0] = 2.0;
        pw.move_character(&mut scene, i, Vec3::new(0.0, -0.01, 0.0));
        let head_x = scene.objects[i].translation.x + 2.0;
        assert!(head_x < 0.65 && head_x > 0.5, "body stops at wall even for a large tracking step: {head_x}");
    }

    #[test]
    fn character_controller_is_blocked_by_walls_and_grounds() {
        let mut scene = LoadedScene::empty();
        floor(&mut scene);
        obj(
            &mut scene,
            Vec3::new(3.0, 2.0, 0.0),
            vec![Box::new(BoxCollider {
                size: [1.0, 4.0, 4.0],
                ..Default::default()
            })],
        );
        let player = obj(&mut scene, Vec3::new(0.0, 0.6, 0.0), vec![Box::new(Character::default())]);
        let mut pw = PhysicsWorld::build(&scene);
        run(&mut pw, &mut scene, 1);
        for _ in 0..30 {
            pw.move_character(&mut scene, 2, Vec3::new(0.2, -0.1, 0.0));
            run(&mut pw, &mut scene, 1);
        }
        let p = scene.objects[2].translation;
        assert!(p.x < 2.5 - 0.3, "stopped at the wall (face at x=2.5), got {}", p.x);
        assert!((p.y - 0.5).abs() < 0.1, "standing on the floor (top y=0.5), got {}", p.y);
        assert!(pw.grounded(player));
    }

    /// A character with its feet on the floor (`y`), stepped like a play
    /// session: one physics step, then `moves` moves of `delta`.
    fn walk_character(scene: &mut LoadedScene, pw: &mut PhysicsWorld, i: usize, frames: usize, moves: usize, delta: Vec3) -> (f32, f32) {
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for _ in 0..frames {
            run(pw, scene, 1);
            for _ in 0..moves {
                pw.move_character(scene, i, delta);
                let y = scene.objects[i].translation.y;
                (lo, hi) = (lo.min(y), hi.max(y));
            }
        }
        (lo, hi)
    }

    #[test]
    fn moves_between_physics_steps_chain() {
        let mut scene = LoadedScene::empty();
        floor(&mut scene);
        // A wall whose near face is at z = -1.5.
        obj(&mut scene, Vec3::new(0.0, 1.5, -2.0), vec![Box::new(BoxCollider { size: [6.0, 2.0, 1.0], ..Default::default() })]);
        let player = obj(&mut scene, Vec3::new(0.0, 0.5, 0.0), vec![Box::new(Character::default())]);
        let i = scene.index_of(player).unwrap();
        let mut pw = PhysicsWorld::build(&scene);
        run(&mut pw, &mut scene, 1);
        // Several moves before the next step (a fast frame rate, or a
        // prediction replay): each starts where the last one ended.
        for _ in 0..4 {
            pw.move_character(&mut scene, i, Vec3::new(0.3, 0.0, 0.0));
        }
        let p = scene.objects[i].translation;
        assert!((p.x - 1.2).abs() < 0.01, "four 0.3 m moves add up: {p}");
        // ... so they collide where the previous one left off too.
        for _ in 0..20 {
            pw.move_character(&mut scene, i, Vec3::new(0.0, 0.0, -0.2));
        }
        let p = scene.objects[i].translation;
        assert!(p.z > -1.5 + 0.35 - 0.01 && p.z < -1.5 + 0.35 + 0.05, "stopped at the wall, not through it: {p}");
    }

    #[test]
    fn a_character_spawned_on_the_floor_does_not_sink() {
        let mut scene = LoadedScene::empty();
        floor(&mut scene);
        // Feet exactly on the floor top (y = 0.5), the default capsule.
        let player = obj(&mut scene, Vec3::new(0.0, 0.5, 0.0), vec![Box::new(Character::default())]);
        let i = scene.index_of(player).unwrap();
        let mut pw = PhysicsWorld::build(&scene);
        // Standing (gravity only), two moves a step now and then.
        let (lo, hi) = walk_character(&mut scene, &mut pw, i, 120, 1, Vec3::new(0.0, -0.005, 0.0));
        assert!(lo > 0.5 - 1e-3 && hi < 0.5 + 0.03, "stays on the floor while standing: {lo}..{hi}");
        assert!(pw.grounded(player));
        // Walking, held down hard (a controller's grounded fall speed).
        let (lo, hi) = walk_character(&mut scene, &mut pw, i, 120, 2, Vec3::new(0.0, -0.0222, -0.0136));
        let p = scene.objects[i].translation;
        assert!(lo > 0.5 - 1e-3 && hi < 0.5 + 0.03, "stays on the floor while walking: {lo}..{hi}");
        assert!(p.z < -3.0, "the floor never stops it: {p}");
        // Spawned a little inside the floor: pushed out on its first move.
        scene.objects[i].translation = Vec3::new(0.0, 0.4, 0.0);
        pw.place_character(&scene, i, true);
        pw.move_character(&mut scene, i, Vec3::new(0.0, -0.005, 0.0));
        assert!(scene.objects[i].translation.y > 0.5 - 1e-3, "out of the floor: {}", scene.objects[i].translation);
    }

    #[test]
    fn destroyed_and_disabled_objects_leave_the_simulation() {
        let mut scene = LoadedScene::empty();
        floor(&mut scene);
        let mut pw = PhysicsWorld::build(&scene);
        run(&mut pw, &mut scene, 1);
        let down = |pw: &PhysicsWorld| pw.raycast(Vec3::new(0.0, 5.0, 0.0), Vec3::NEG_Y, 100.0, &RayFilter::default());
        assert!(down(&pw).is_some());
        scene.objects[0].enabled = false;
        pw.mark_dirty();
        run(&mut pw, &mut scene, 1);
        assert!(down(&pw).is_none());
        scene.objects[0].enabled = true;
        pw.mark_dirty();
        run(&mut pw, &mut scene, 1);
        assert!(down(&pw).is_some());
        scene.remove_object(0);
        pw.mark_dirty();
        run(&mut pw, &mut scene, 1);
        assert!(down(&pw).is_none());
    }

    fn jointed(scene: &mut LoadedScene, pos: Vec3, joint: PhysicsJoint) -> ObjectId {
        obj(
            scene,
            pos,
            vec![
                Box::new(BoxCollider { size: [0.5, 0.5, 0.5], ..Default::default() }),
                Box::new(RbComponent::default()),
                Box::new(joint),
            ],
        )
    }

    #[test]
    fn fixed_joint_to_world_holds_body() {
        let mut scene = LoadedScene::empty();
        jointed(&mut scene, Vec3::new(0.0, 5.0, 0.0), PhysicsJoint { kind: JointKind::Fixed, ..Default::default() });
        let mut pw = PhysicsWorld::build(&scene);
        run(&mut pw, &mut scene, 120);
        assert!((scene.objects[0].translation.y - 5.0).abs() < 0.05, "{}", scene.objects[0].translation);
    }

    #[test]
    fn hinge_and_rope_constrain_distance() {
        let mut scene = LoadedScene::empty();
        // Pendulum: hinge anchor 1 m above the body, axis Z.
        jointed(
            &mut scene,
            Vec3::new(0.0, 5.0, 0.0),
            PhysicsJoint {
                kind: JointKind::Hinge,
                anchor: [1.0, 1.0, 0.0],
                axis: [0.0, 0.0, 1.0],
                ..Default::default()
            },
        );
        // Rope from the world start point, 2 m long, body starts 1 m below it.
        jointed(
            &mut scene,
            Vec3::new(10.0, 5.0, 0.0),
            PhysicsJoint {
                kind: JointKind::Rope,
                anchor: [0.0, 1.0, 0.0],
                rest_length: 2.0,
                ..Default::default()
            },
        );
        let mut pw = PhysicsWorld::build(&scene);
        let pivot = Vec3::new(1.0, 6.0, 0.0);
        let rope_top = Vec3::new(10.0, 6.0, 0.0);
        let start = scene.objects[0].translation.distance(pivot);
        for _ in 0..180 {
            run(&mut pw, &mut scene, 1);
            let d = scene.objects[0].translation.distance(pivot);
            assert!((d - start).abs() < 0.05, "hinge distance drifted to {d}");
            assert!(scene.objects[1].translation.distance(rope_top) < 3.0 + 0.05, "rope stretched");
        }
        assert!(scene.objects[0].translation.y < 5.0, "pendulum swung down");
    }

    #[test]
    fn hinge_motor_spins_body() {
        let mut scene = LoadedScene::empty();
        let id = obj(
            &mut scene,
            Vec3::new(0.0, 5.0, 0.0),
            vec![
                Box::new(BoxCollider::default()),
                Box::new(RbComponent { gravity_scale: 0.0, ..Default::default() }),
                Box::new(PhysicsJoint { kind: JointKind::Hinge, ..Default::default() }),
            ],
        );
        let mut pw = PhysicsWorld::build(&scene);
        pw.set_joint_motor(id, 180.0, 1000.0);
        run(&mut pw, &mut scene, 30);
        let yaw = scene.objects[0].rotation.to_euler(glam::EulerRot::YXZ).0.to_degrees().abs();
        assert!(yaw > 30.0, "wheel turned {yaw} deg");
    }

    #[test]
    fn gameplay_teleport_moves_dynamic_body() {
        let mut scene = LoadedScene::empty();
        floor(&mut scene);
        dynamic_ball(&mut scene, Vec3::new(0.0, 3.0, 0.0));
        let mut pw = PhysicsWorld::build(&scene);
        run(&mut pw, &mut scene, 60);
        scene.objects[1].translation = Vec3::new(5.0, 3.0, 0.0);
        run(&mut pw, &mut scene, 1);
        assert!((scene.objects[1].translation.x - 5.0).abs() < 1e-3);
    }
}
