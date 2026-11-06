//! Runtime 2D physics (rapier2d), driven by [`crate::play::PlaySession`].
//!
//! Mirrors [`crate::physics`]: bodies are keyed by `ObjectId`, spawning /
//! destroying / disabling objects just marks the world dirty and
//! [`PhysicsWorld2D::sync_structure`] adds or removes the affected bodies. The
//! simulation runs at the shared [`FIXED_DT`]; dynamic bodies are interpolated
//! between the last two steps when written back.
//!
//! The 2D world lives in the XY plane and rotates about Z only: a body's pose
//! is the object's world X, Y and Z-rotation, and write-back leaves the
//! object's Z, X-tilt and Y-tilt alone.
//!
//! Choices (rapier defaults wherever Unity / Box2D v3 numbers were still an
//! open research item):
//!
//! - **Layers**: one matrix for 2D and 3D (the project's `LayerSettings`), by
//!   the object's layer, mapped to rapier interaction groups.
//! - **Materials**: `.physmat`. Friction, bounciness and the combine modes
//!   apply; density, buoyancy and water drag do not (mass comes from the
//!   collider `density` fields, area-based, default 1). No material: friction
//!   0.4, bounciness 0 (Unity's default 2D material).
//! - **Settings**: the project Physics settings (gravity x / y, contact
//!   stiffness and damping, solver iterations). rapier's default solver, not
//!   Box2D v3's soft-step sub-stepping (4 sub-steps, 30 Hz / ratio 10).
//! - **Body groups**: a collider on an object without a `Rigidbody 2D` attaches
//!   to its nearest ancestor's body, else it is a static body of its own.
//! - **Composite**: colliders flagged `used_by_composite` under a Composite
//!   Collider 2D become one compound collider (see [`PhysicsWorld2D::set_compound`],
//!   the same API tilemap chunks use).
//! - **Events**: contact begin / end and trigger begin / end after each step,
//!   plus a "stay" list per frame; delivered to the colliders' objects and to
//!   the objects owning their bodies.
//! - **Determinism**: bodies and joints are created in scene order, effector
//!   variation is a counter hash, nothing iterates a hash map to decide
//!   physics, so the same scene and inputs give identical steps on one machine.
//!
//! Not supported (rapier2d has no equivalent): Unity's Friction / Relative /
//! Target joints, joint break force, `ConstantForce2D`, per-collider layer
//! overrides, composite boolean union (parts stay separate shapes inside one
//! compound; Outlines geometry is the unmerged boundary).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};

use glam::{EulerRot, Mat4, Quat, Vec2, Vec3};
use rapier2d::math::{Pose, Rotation, Vector};
use rapier2d::parry::query::ShapeCastOptions;
use rapier2d::prelude::*;

use zestengine_core::{
    BodyType2D, BoxCollider2D, CapsuleCollider2D, CapsuleDirection2D, CircleCollider2D, Collider2D, CompositeCollider2D, CompositeGeometry2D,
    ContactPoint2D, CombineMode, EdgeCollider2D, Effector2D, EffectorKind2D, Interpolation2D, Joint2D, Joint2DKind, ObjectId, PhysicsMaterial,
    PhysicsSettings, PointForceMode2D, PolygonCollider2D, RayFilter, RayHit2D, Rigidbody2D, SleepMode2D,
};

mod tile;
pub use tile::{TileColliderState, TileColliderStats};
use crate::physics::{FIXED_DT, PhysicsWorld as PhysicsWorld3D};
use crate::scene::LoadedScene;

/// Joint velocity-motor gain, as in the 3D world.
const MOTOR_GAIN: f32 = 50.0;

/// Sleep thresholds: m/s, rad/s and seconds still (Box2D's 0.05 m/s, Unity's 0.5 s).
const SLEEP_LINEAR: f32 = 0.05;
const SLEEP_ANGULAR: f32 = 0.05;
const SLEEP_TIME: f32 = 0.5;

/// Friction of a collider with no material (Unity's default 2D material).
const DEFAULT_FRICTION: f32 = 0.4;

/// A collision / trigger contact change between two objects.
#[derive(Clone, Copy, Debug)]
pub struct ContactEvent2D {
    /// The objects whose colliders touched.
    pub a: ObjectId,
    pub b: ObjectId,
    /// The objects owning those colliders' bodies (equal to `a` / `b` unless a
    /// child collider belongs to a parent's body).
    pub a_body: ObjectId,
    pub b_body: ObjectId,
    pub started: bool,
    pub trigger: bool,
}

/// One body: where it is, and what it is made of.
struct Body2D {
    handle: RigidBodyHandle,
    kind: BodyType2D,
    interp: Interpolation2D,
    /// Objects whose colliders are on this body (in scene order); a change
    /// rebuilds the body.
    attached: Vec<ObjectId>,
    /// World pose (position, angle) after the previous / latest step.
    prev: (Vec2, f32),
    curr: (Vec2, f32),
    /// Local translation / rotation last written into (or read from) the scene;
    /// a mismatch means gameplay moved the object.
    written: (Vec3, Quat),
    debug: bool,
    z: f32,
    /// Externally supplied compound colliders (tilemap chunks) by key.
    compounds: HashMap<u64, ColliderHandle>,
    /// Material path every collider on the body inherits.
    material: String,
    layer: u8,
}

/// An effector acting on what overlaps one of its trigger colliders.
struct VolumeEffector {
    collider: ColliderHandle,
    params: Effector2D,
}

struct PlatformInfo {
    up_local: Vec2,
    one_way: bool,
    cos_arc: f32,
    side_friction: bool,
    side_bounce: bool,
}

struct SurfaceInfo {
    speed: f32,
    variation: f32,
}

pub struct PhysicsWorld2D {
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
    bodies: HashMap<ObjectId, Body2D>,
    /// Collider -> (its object, its body's object). Kept past collider removal
    /// so the final "stopped" event still resolves.
    collider_owner: HashMap<ColliderHandle, (ObjectId, ObjectId)>,
    collector: ChannelEventCollector,
    collision_rx: Receiver<CollisionEvent>,
    _force_rx: Receiver<ContactForceEvent>,
    pending: Vec<ContactEvent2D>,
    /// Touching pairs (collider, collider, trigger), for the per-frame stay list.
    active: HashSet<(ColliderHandle, ColliderHandle, bool)>,
    dirty: bool,
    joints: HashMap<ObjectId, ImpulseJointHandle>,
    /// Static body at the origin that world-anchored joints attach to.
    world_anchor: RigidBodyHandle,
    root: Option<PathBuf>,
    default_material: String,
    materials: HashMap<String, Option<PhysicsMaterial>>,
    /// Effectors in creation order (deterministic iteration).
    volume_effectors: Vec<VolumeEffector>,
    platforms: HashMap<ColliderHandle, PlatformInfo>,
    surfaces: HashMap<ColliderHandle, SurfaceInfo>,
    steps: u64,
    /// Forces were added since the last step (so a quiet world skips the reset).
    forces_dirty: bool,
    /// Scratch for effector passes (no per-step allocation once warm).
    scratch: Vec<RigidBodyHandle>,
    /// Tilemap colliders: `(tilemap object, owning body object)` per sync.
    tile_objs: Vec<(ObjectId, ObjectId)>,
    tile_state: HashMap<ObjectId, TileColliderState>,
    tile_db: Option<crate::tile_db::TileDb>,
    sprite_pieces: HashMap<String, Vec<Vec<Vec2>>>,
    tile_stats: TileColliderStats,
}

fn rv(v: Vec2) -> Vector {
    Vector::new(v.x, v.y)
}

fn gv(v: Vector) -> Vec2 {
    Vec2::new(v.x, v.y)
}

fn pose2(p: Vec2, angle: f32) -> Pose {
    Pose::from_parts(rv(p), Rotation::from_angle(angle))
}

/// A world matrix as (position, angle about Z, scale magnitude in x / y).
fn decompose(m: Mat4) -> (Vec2, f32, Vec2) {
    let (s, r, t) = m.to_scale_rotation_translation();
    let (z, _, _) = r.to_euler(EulerRot::ZYX);
    (t.truncate(), z, s.truncate().abs().max(Vec2::splat(1e-3)))
}

fn lerp_angle(a: f32, b: f32, t: f32) -> f32 {
    let mut d = (b - a) % std::f32::consts::TAU;
    if d > std::f32::consts::PI {
        d -= std::f32::consts::TAU;
    } else if d < -std::f32::consts::PI {
        d += std::f32::consts::TAU;
    }
    a + d * t
}

/// A hash to a value in -1..1, so variation is deterministic.
fn jitter(step: u64, salt: u64) -> f32 {
    let mut x = step.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ salt.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 30;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 27;
    (x >> 40) as f32 / (1u64 << 23) as f32 - 1.0
}

fn combine_rule(mode: CombineMode) -> CoefficientCombineRule {
    match mode {
        CombineMode::Average => CoefficientCombineRule::Average,
        CombineMode::Minimum => CoefficientCombineRule::Min,
        CombineMode::Multiply => CoefficientCombineRule::Multiply,
        CombineMode::Maximum => CoefficientCombineRule::Max,
    }
}

/// Whether a component belongs to the 2D world.
fn is_2d_component(c: &dyn zestengine_core::Component) -> bool {
    matches!(
        c.type_name(),
        "Rigidbody 2D"
            | "Box Collider 2D"
            | "Circle Collider 2D"
            | "Capsule Collider 2D"
            | "Polygon Collider 2D"
            | "Edge Collider 2D"
            | "Composite Collider 2D"
            | "Tilemap Collider 2D"
    )
}

/// Whether the scene has anything for the 2D world.
pub fn scene_has_2d(scene: &LoadedScene) -> bool {
    scene.objects.iter().any(|o| o.components.iter().any(|c| is_2d_component(c.as_ref())))
}

// ------------------------------------------------------------------ shapes

/// What a collider component builds: shapes posed in the collider object's
/// local space (offset and scale applied), plus its boundary rings for
/// Outlines composites.
#[derive(Default)]
pub struct Built {
    pub shapes: Vec<(Pose, SharedShape)>,
    /// Boundary rings (and whether closed), collider-object space.
    pub rings: Vec<(Vec<Vec2>, bool)>,
}

fn circle_ring(c: Vec2, r: f32) -> Vec<Vec2> {
    (0..24).map(|k| c + Vec2::from_angle(k as f32 / 24.0 * std::f32::consts::TAU) * r).collect()
}

fn is_convex(p: &[Vec2]) -> bool {
    let n = p.len();
    if n < 3 {
        return false;
    }
    let mut sign = 0.0f32;
    for i in 0..n {
        let c = (p[(i + 1) % n] - p[i]).perp_dot(p[(i + 2) % n] - p[(i + 1) % n]);
        if c.abs() < 1e-9 {
            continue;
        }
        if sign == 0.0 {
            sign = c.signum();
        } else if c.signum() != sign {
            return false;
        }
    }
    true
}

/// Convex pieces of a set of rings (outer boundaries and holes, any winding,
/// nesting decides which): a single convex ring is used as is, anything else
/// goes through the sprite physics code (ear clipping, then merging triangles
/// into convex polygons).
pub fn polygon_pieces(paths: &[Vec<[f32; 2]>]) -> Vec<Vec<Vec2>> {
    let paths: Vec<&Vec<[f32; 2]>> = paths.iter().filter(|p| p.len() >= 3).collect();
    if paths.len() == 1 {
        let ring: Vec<Vec2> = paths[0].iter().map(|p| Vec2::from(*p)).collect();
        if is_convex(&ring) {
            return vec![ring];
        }
    }
    let loops: Vec<Vec<[f32; 2]>> = paths.iter().map(|p| (*p).clone()).collect();
    match zestengine_assets::sprite_mesh::mesh_from_loops(&loops, true) {
        Some(mesh) => zestengine_assets::sprite_mesh::convex_pieces(&mesh)
            .into_iter()
            .map(|piece| piece.into_iter().map(Vec2::from).collect())
            .collect(),
        None => Vec::new(),
    }
}

/// A Polygon Collider 2D filled from a sprite's physics shape (world units,
/// pivot-relative, as `SpriteGeometry::physics`); a sprite without one gets
/// its bounding rectangle.
pub fn polygon_from_sprite(geometry: &zestengine_assets::sprite::SpriteGeometry) -> PolygonCollider2D {
    let paths = if geometry.physics.is_empty() {
        let (lo, hi) = geometry.bounds();
        vec![vec![[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]]
    } else {
        geometry.physics.clone()
    };
    PolygonCollider2D { paths, ..Default::default() }
}

fn scaled_ring(p: &[[f32; 2]], off: Vec2, s: Vec2) -> Vec<Vec2> {
    p.iter().map(|v| Vec2::from(*v) * s + off).collect()
}

/// The shapes of any 2D collider component (None for other components).
pub fn build_shapes(any: &dyn std::any::Any, scale: Vec2) -> Option<Built> {
    let mut out = Built::default();
    if let Some(b) = any.downcast_ref::<BoxCollider2D>() {
        let off = Vec2::from(b.offset) * scale;
        let half = (Vec2::from(b.size) * 0.5 * scale).abs().max(Vec2::splat(1e-3));
        out.shapes.push((Pose::from_translation(rv(off)), SharedShape::cuboid(half.x, half.y)));
        out.rings.push((vec![off + Vec2::new(-half.x, -half.y), off + Vec2::new(half.x, -half.y), off + half, off + Vec2::new(-half.x, half.y)], true));
    } else if let Some(c) = any.downcast_ref::<CircleCollider2D>() {
        let off = Vec2::from(c.offset) * scale;
        let r = (c.radius * scale.max_element()).max(1e-3);
        out.shapes.push((Pose::from_translation(rv(off)), SharedShape::ball(r)));
        out.rings.push((circle_ring(off, r), true));
    } else if let Some(c) = any.downcast_ref::<CapsuleCollider2D>() {
        let off = Vec2::from(c.offset) * scale;
        let size = (Vec2::from(c.size) * scale).abs().max(Vec2::splat(2e-3));
        let (shape, r, half, vertical) = match c.direction {
            CapsuleDirection2D::Vertical => {
                let r = size.x * 0.5;
                let half = (size.y * 0.5 - r).max(0.0);
                (SharedShape::capsule_y(half, r), r, half, true)
            }
            CapsuleDirection2D::Horizontal => {
                let r = size.y * 0.5;
                let half = (size.x * 0.5 - r).max(0.0);
                (SharedShape::capsule_x(half, r), r, half, false)
            }
        };
        out.shapes.push((Pose::from_translation(rv(off)), shape));
        // Outline: two half circles joined by the straight sides.
        let axis = if vertical { Vec2::Y } else { Vec2::X };
        let mut ring = Vec::new();
        for k in 0..=12 {
            let a = k as f32 / 12.0 * std::f32::consts::PI;
            ring.push(off + axis * half + Vec2::from_angle(a + if vertical { 0.0 } else { -std::f32::consts::FRAC_PI_2 }) * r);
        }
        for k in 0..=12 {
            let a = k as f32 / 12.0 * std::f32::consts::PI + std::f32::consts::PI;
            ring.push(off - axis * half + Vec2::from_angle(a + if vertical { 0.0 } else { -std::f32::consts::FRAC_PI_2 }) * r);
        }
        out.rings.push((ring, true));
    } else if let Some(c) = any.downcast_ref::<PolygonCollider2D>() {
        let off = Vec2::from(c.offset) * scale;
        let paths: Vec<Vec<[f32; 2]>> = c.paths.iter().map(|p| p.iter().map(|v| (Vec2::from(*v) * scale).to_array()).collect()).collect();
        let mut pieces: Vec<(Pose, SharedShape)> = Vec::new();
        for ring in polygon_pieces(&paths) {
            let pts: Vec<Vector> = ring.iter().map(|p| rv(*p)).collect();
            if let Some(shape) = SharedShape::convex_hull(&pts) {
                pieces.push((Pose::IDENTITY, shape));
            }
        }
        if pieces.is_empty() {
            return None;
        }
        for (p, _) in &mut pieces {
            *p = Pose::from_translation(rv(off));
        }
        out.shapes = pieces;
        for p in &c.paths {
            out.rings.push((scaled_ring(p, off, scale), true));
        }
    } else {
        let c = any.downcast_ref::<EdgeCollider2D>()?;
        if c.points.len() < 2 {
            return None;
        }
        let off = Vec2::from(c.offset) * scale;
        let pts = scaled_ring(&c.points, off, scale);
        let n = pts.len() as u32;
        let mut idx: Vec<[u32; 2]> = (0..n - 1).map(|i| [i, i + 1]).collect();
        if c.closed && n > 2 {
            idx.push([n - 1, 0]);
        }
        out.shapes.push((Pose::IDENTITY, SharedShape::polyline(pts.iter().map(|p| rv(*p)).collect(), Some(idx))));
        out.rings.push((pts, c.closed));
    }
    Some(out)
}

/// One compound shape out of several posed ones (None if empty); one shape
/// stays itself. A compound is a single broadphase entry, so a tilemap chunk
/// of thousands of tiles costs one collider.
pub fn compound_shape(mut parts: Vec<(Pose, SharedShape)>) -> Option<SharedShape> {
    match parts.len() {
        0 => None,
        1 if parts[0].0 == Pose::IDENTITY => Some(parts.remove(0).1),
        _ => Some(SharedShape::compound(parts)),
    }
}

fn collider_props(c: &dyn std::any::Any) -> Option<zestengine_core::ColliderProps<'_>> {
    macro_rules! try_props {
        ($($t:ty),*) => {$(
            if let Some(x) = c.downcast_ref::<$t>() { return Some(Collider2D::props(x)); }
        )*};
    }
    try_props!(BoxCollider2D, CircleCollider2D, CapsuleCollider2D, PolygonCollider2D, EdgeCollider2D);
    None
}

/// How a compound collider is made (tilemap chunks and composites).
#[derive(Clone, Debug, Default)]
pub struct CompoundProps {
    pub is_trigger: bool,
    pub material: String,
}

impl PhysicsWorld2D {
    /// Build a world from the scene's active 2D bodies and colliders.
    pub fn build(scene: &LoadedScene) -> Self {
        Self::build_with(scene, None, &PhysicsSettings::default())
    }

    /// [`Self::build`] with project physics settings, resolving materials
    /// under `root`.
    pub fn build_with(scene: &LoadedScene, root: Option<&Path>, settings: &PhysicsSettings) -> Self {
        let (collision_tx, collision_rx) = channel();
        let (force_tx, force_rx) = channel();
        let mut rigid_bodies = RigidBodySet::new();
        let world_anchor = rigid_bodies.insert(RigidBodyBuilder::fixed().build());
        let mut world = Self {
            pipeline: PhysicsPipeline::new(),
            gravity: Vector::new(settings.gravity[0], settings.gravity[1]),
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
            active: HashSet::new(),
            dirty: true,
            joints: HashMap::new(),
            world_anchor,
            root: root.map(Path::to_owned),
            default_material: settings.default_material.clone(),
            materials: HashMap::new(),
            volume_effectors: Vec::new(),
            platforms: HashMap::new(),
            surfaces: HashMap::new(),
            steps: 0,
            forces_dirty: false,
            scratch: Vec::new(),
            tile_objs: Vec::new(),
            tile_state: HashMap::new(),
            tile_db: None,
            sprite_pieces: HashMap::new(),
            tile_stats: TileColliderStats::default(),
        };
        world.sync_structure(scene);
        world
    }

    /// Objects were spawned, destroyed or toggled; rebuild membership next sync.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Colliders in the world (the origin anchor has none).
    pub fn collider_count(&self) -> usize {
        self.colliders.len()
    }

    /// Register a material under `path` without a file (tests, tools).
    pub fn set_material(&mut self, path: &str, material: PhysicsMaterial) {
        self.materials.insert(path.to_owned(), Some(material));
    }

    fn material(&mut self, path: &str) -> Option<PhysicsMaterial> {
        if path.is_empty() {
            return None;
        }
        if let Some(m) = self.materials.get(path) {
            return *m;
        }
        let root = self.root.as_ref()?;
        let m = zestengine_core::load_physmat(zestengine_core::asset_ref::resolve_path(root, path))
            .map_err(|e| tracing::warn!("physics material: {e}"))
            .ok();
        self.materials.insert(path.to_owned(), m);
        m
    }

    /// Add bodies for newly active 2D objects and drop bodies whose object is
    /// gone, disabled or changed shape. Cheap no-op unless marked dirty.
    pub fn sync_structure(&mut self, scene: &LoadedScene) {
        if !self.dirty {
            return;
        }
        self.dirty = false;
        let world = scene.world_transforms();
        let plans = plan_bodies(scene);
        self.tile_objs.clear();
        for p in &plans {
            for &ci in &p.colliders {
                if has_component::<zestengine_core::TilemapCollider2D>(scene, ci) && has_component::<zestengine_core::Tilemap>(scene, ci) {
                    self.tile_objs.push((scene.objects[ci].id, scene.objects[p.body].id));
                }
            }
        }
        self.tile_state.retain(|id, _| self.tile_objs.iter().any(|(t, _)| t == id));
        let wanted: HashMap<ObjectId, &Plan> = plans.iter().map(|p| (scene.objects[p.body].id, p)).collect();
        let stale: Vec<ObjectId> = self
            .bodies
            .iter()
            .filter(|(id, b)| wanted.get(id).is_none_or(|p| p.collider_ids(scene) != b.attached))
            .map(|(id, _)| *id)
            .collect();
        for id in stale {
            self.remove_body(id);
        }
        for plan in &plans {
            let id = scene.objects[plan.body].id;
            if !self.bodies.contains_key(&id) {
                self.add_body(scene, plan, &world);
            }
        }
        // Joints: drop those whose bodies went away (rapier removed them with
        // the body), then create missing ones once both ends exist.
        let impulse_joints = &self.impulse_joints;
        self.joints.retain(|id, h| wanted.contains_key(id) && impulse_joints.get(*h).is_some());
        for (i, object) in scene.objects.iter().enumerate() {
            if self.joints.contains_key(&object.id) || !wanted.contains_key(&object.id) {
                continue;
            }
            if let Some(j) = object.components.iter().find_map(|c| c.as_any().downcast_ref::<Joint2D>()) {
                self.add_joint(scene, object.id, j, world[i]);
            }
        }
    }

    fn remove_body(&mut self, id: ObjectId) {
        self.tile_state.retain(|_, st| st.owner != Some(id));
        let Some(body) = self.bodies.remove(&id) else { return };
        let gone: HashSet<ColliderHandle> = self
            .rigid_bodies
            .get(body.handle)
            .map(|rb| rb.colliders().iter().copied().collect())
            .unwrap_or_default();
        self.volume_effectors.retain(|e| !gone.contains(&e.collider));
        for h in &gone {
            self.platforms.remove(h);
            self.surfaces.remove(h);
        }
        self.active.retain(|(a, b, _)| !gone.contains(a) && !gone.contains(b));
        self.rigid_bodies.remove(
            body.handle,
            &mut self.islands,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            true,
        );
    }

    fn add_body(&mut self, scene: &LoadedScene, plan: &Plan, world: &[Mat4]) {
        let object = &scene.objects[plan.body];
        let id = object.id;
        let (pos, angle, _) = decompose(world[plan.body]);
        let rb = object.components.iter().find_map(|c| c.as_any().downcast_ref::<Rigidbody2D>());
        let kind = rb.map_or(BodyType2D::Static, |r| r.body_type);
        let builder = match kind {
            BodyType2D::Dynamic => RigidBodyBuilder::dynamic(),
            BodyType2D::Kinematic => RigidBodyBuilder::kinematic_position_based(),
            BodyType2D::Static => RigidBodyBuilder::fixed(),
        }
        .pose(pose2(pos, angle))
        .user_data(id.raw());
        let builder = match rb {
            Some(r) if kind == BodyType2D::Dynamic => {
                let mut locked = LockedAxes::empty();
                locked.set(LockedAxes::TRANSLATION_LOCKED_X, r.freeze_x);
                locked.set(LockedAxes::TRANSLATION_LOCKED_Y, r.freeze_y);
                locked.set(LockedAxes::ROTATION_LOCKED, r.freeze_rotation);
                builder
                    .gravity_scale(r.gravity_scale)
                    .linear_damping(r.linear_damping.max(0.0))
                    .angular_damping(r.angular_damping.max(0.0))
                    .locked_axes(locked)
                    .ccd_enabled(r.continuous)
                    .can_sleep(r.sleep_mode != SleepMode2D::NeverSleep)
                    .sleeping(r.sleep_mode == SleepMode2D::StartAsleep)
            }
            _ => builder,
        };
        let mut rigid_body = builder.build();
        // Sleep like Unity / Box2D, not rapier's coarser 0.4 m/s and 2 s (a
        // 0.3 m/s conveyor load or a small pendulum would freeze).
        let act = rigid_body.activation_mut();
        act.normalized_linear_threshold = SLEEP_LINEAR;
        act.angular_threshold = SLEEP_ANGULAR;
        act.time_until_sleep = SLEEP_TIME;
        let handle = self.rigid_bodies.insert(rigid_body);
        let body_material = rb.map(|r| r.material.clone()).unwrap_or_default();
        let body_pose = pose2(pos, angle);

        // Collect every collider first so an explicit mass can scale densities.
        struct Spec {
            shape: SharedShape,
            density: f32,
            trigger: bool,
            effector: bool,
            material: String,
            object: usize,
            layer: u8,
        }
        let mut specs: Vec<Spec> = Vec::new();
        // Composite parts by composite object index.
        let mut composites: Vec<(usize, Vec<Built>, Vec<Pose>)> = Vec::new();
        for &ci in &plan.colliders {
            let co = &scene.objects[ci];
            let (cp, ca, cs) = decompose(world[ci]);
            // Collider object pose relative to the body.
            let rel = body_pose.inverse() * pose2(cp, ca);
            let composite_owner = composite_of(scene, ci, plan);
            for c in &co.components {
                let any = c.as_any();
                let Some(props) = collider_props(any) else { continue };
                let Some(built) = build_shapes(any, cs) else { continue };
                if props.used_by_composite && let Some(owner) = composite_owner {
                    match composites.iter_mut().find(|(o, _, _)| *o == owner) {
                        Some((_, parts, poses)) => {
                            parts.push(built);
                            poses.push(rel);
                        }
                        None => composites.push((owner, vec![built], vec![rel])),
                    }
                    continue;
                }
                let placed: Vec<(Pose, SharedShape)> = built.shapes.into_iter().map(|(p, s)| (rel * p, s)).collect();
                if let Some(shape) = compound_shape(placed) {
                    specs.push(Spec {
                        shape,
                        density: props.density.max(0.0),
                        trigger: props.is_trigger,
                        effector: props.used_by_effector,
                        material: props.material.to_owned(),
                        object: ci,
                        layer: co.layer,
                    });
                }
            }
        }
        for (owner, parts, poses) in composites {
            let co = &scene.objects[owner];
            let comp = co.components.iter().find_map(|c| c.as_any().downcast_ref::<CompositeCollider2D>());
            let Some(comp) = comp else { continue };
            let (_, _, cs) = decompose(world[owner]);
            let (cp, ca, _) = decompose(world[owner]);
            let rel_owner = body_pose.inverse() * pose2(cp, ca);
            let off = Pose::from_translation(rv(Vec2::from(comp.offset) * cs));
            let shape = match comp.geometry {
                CompositeGeometry2D::Polygons => {
                    let mut all: Vec<(Pose, SharedShape)> = Vec::new();
                    for (b, rel) in parts.into_iter().zip(&poses) {
                        all.extend(b.shapes.into_iter().map(|(p, s)| (off * *rel * p, s)));
                    }
                    compound_shape(all)
                }
                CompositeGeometry2D::Outlines => {
                    let mut verts: Vec<Vector> = Vec::new();
                    let mut idx: Vec<[u32; 2]> = Vec::new();
                    for (b, rel) in parts.iter().zip(&poses) {
                        for (ring, closed) in &b.rings {
                            let base = verts.len() as u32;
                            verts.extend(ring.iter().map(|p| (off * *rel).transform_point(rv(*p))));
                            let n = ring.len() as u32;
                            idx.extend((0..n.saturating_sub(1)).map(|i| [base + i, base + i + 1]));
                            if *closed && n > 2 {
                                idx.push([base + n - 1, base]);
                            }
                        }
                    }
                    (!idx.is_empty()).then(|| SharedShape::polyline(verts, Some(idx)))
                }
            };
            let _ = rel_owner;
            if let Some(shape) = shape {
                specs.push(Spec {
                    shape,
                    density: 1.0,
                    trigger: comp.is_trigger,
                    effector: comp.used_by_effector,
                    material: comp.material.clone(),
                    object: owner,
                    layer: co.layer,
                });
            }
        }

        // An explicit mass replaces the colliders' density-based one.
        let mut density_scale = 1.0;
        if let Some(r) = rb.filter(|r| r.mass > 0.0 && kind == BodyType2D::Dynamic) {
            let total: f32 = specs.iter().filter(|s| !s.trigger).map(|s| s.shape.mass_properties(s.density).mass()).sum();
            if total > 1e-9 {
                density_scale = r.mass / total;
            } else if let Some(b) = self.rigid_bodies.get_mut(handle) {
                b.set_additional_mass(r.mass, false);
            }
        }

        for spec in specs {
            let own = scene.objects[spec.object].id;
            let mat_path = if !spec.material.is_empty() { spec.material.clone() } else { body_material.clone() };
            let default_path = self.default_material.clone();
            let mat = self.material(&mat_path).or_else(|| self.material(&default_path));
            let m = mat.unwrap_or(PhysicsMaterial { friction: DEFAULT_FRICTION, ..Default::default() });
            let groups = InteractionGroups::new(
                Group::from_bits_truncate(1u32 << (spec.layer as u32 & 31)),
                Group::from_bits_truncate(scene.layers.collision_mask(spec.layer)),
                InteractionTestMode::And,
            );
            let mut cb = ColliderBuilder::new(spec.shape)
                .restitution(m.bounciness.clamp(0.0, 1.0))
                .restitution_combine_rule(combine_rule(m.bounce_combine))
                .friction(m.friction.max(0.0))
                .friction_combine_rule(combine_rule(m.friction_combine))
                .collision_groups(groups)
                .active_events(ActiveEvents::COLLISION_EVENTS)
                .user_data(own.raw())
                .density(if spec.trigger { 0.0 } else { spec.density * density_scale });
            if spec.trigger {
                cb = cb.sensor(true).active_collision_types(ActiveCollisionTypes::all());
            }
            // Effector on the collider's object.
            let eff = spec
                .effector
                .then(|| scene.objects[spec.object].components.iter().find_map(|c| c.as_any().downcast_ref::<Effector2D>()))
                .flatten();
            if let Some(e) = eff
                && !e.kind.uses_trigger()
            {
                cb = cb.active_hooks(ActiveHooks::MODIFY_SOLVER_CONTACTS);
            }
            let h = self.colliders.insert_with_parent(cb.build(), handle, &mut self.rigid_bodies);
            self.collider_owner.insert(h, (own, id));
            if let Some(e) = eff {
                self.register_effector(h, e, spec.trigger);
            }
        }

        let (rb_debug, interp) = rb.map_or((false, Interpolation2D::None), |r| (r.debug_draw, r.interpolation));
        self.bodies.insert(
            id,
            Body2D {
                handle,
                kind,
                interp,
                attached: plan.collider_ids(scene),
                prev: (pos, angle),
                curr: (pos, angle),
                written: (object.translation, object.rotation),
                debug: rb_debug,
                z: world[plan.body].w_axis.z,
                compounds: HashMap::new(),
                material: body_material,
                layer: object.layer,
            },
        );
    }

    fn register_effector(&mut self, h: ColliderHandle, e: &Effector2D, trigger: bool) {
        let mask = if e.use_layer_mask { e.layer_mask } else { u32::MAX };
        let mut params = e.clone();
        params.layer_mask = mask;
        match e.kind {
            EffectorKind2D::Area | EffectorKind2D::Buoyancy | EffectorKind2D::Point if trigger => {
                self.volume_effectors.push(VolumeEffector { collider: h, params });
            }
            EffectorKind2D::Platform => {
                self.platforms.insert(
                    h,
                    PlatformInfo {
                        up_local: Vec2::Y,
                        one_way: e.use_one_way,
                        cos_arc: e.surface_arc.clamp(0.0, 180.0).to_radians().cos(),
                        side_friction: e.use_side_friction,
                        side_bounce: e.use_side_bounce,
                    },
                );
            }
            EffectorKind2D::Surface => {
                self.surfaces.insert(h, SurfaceInfo { speed: e.speed, variation: e.speed_variation });
            }
            _ => {}
        }
    }

    fn add_joint(&mut self, scene: &LoadedScene, id: ObjectId, j: &Joint2D, world: Mat4) {
        let Some(b1) = self.bodies.get(&id).map(|b| b.handle) else { return };
        let b2 = match j.connected.id() {
            Some(other) => match self.bodies.get(&other) {
                Some(b) => b.handle,
                None => return,
            },
            None => self.world_anchor,
        };
        let (Some(rb1), Some(rb2)) = (self.rigid_bodies.get(b1), self.rigid_bodies.get(b2)) else { return };
        let (p1, a1r) = (gv(rb1.translation()), rb1.rotation().angle());
        let (p2, a2r) = (gv(rb2.translation()), rb2.rotation().angle());
        let (_, _, scale) = decompose(world);
        let a1 = Vec2::from(j.anchor) * scale;
        let anchor_world = p1 + Vec2::from_angle(a1r).rotate(a1);
        let a2 = Vec2::from_angle(-a2r).rotate(anchor_world - p2);
        let axis_deg = j.axis_angle.to_radians();
        let axis1 = Vec2::from_angle(axis_deg);
        let axis2 = Vec2::from_angle(axis_deg + a1r - a2r);
        // Springs / ropes to another body attach at its origin, so the start
        // distance is the rest length; world-anchored ones pull to the start point.
        let far2 = if j.connected.is_set() { Vec2::ZERO } else { a2 };
        let rest = if j.distance > 0.0 { j.distance } else { (p2 + Vec2::from_angle(a2r).rotate(far2)).distance(anchor_world) };
        let lim = |deg: bool| {
            let f = |v: f32| if deg { v.to_radians() } else { v };
            [f(j.limits[0].min(j.limits[1])), f(j.limits[0].max(j.limits[1]))]
        };
        let mut data: GenericJoint = match j.kind {
            Joint2DKind::Fixed => FixedJointBuilder::new()
                .local_frame1(Pose::from_translation(rv(a1)))
                .local_frame2(Pose::from_parts(rv(a2), Rotation::from_angle(a1r - a2r)))
                .build()
                .into(),
            Joint2DKind::Hinge => {
                let mut b = RevoluteJointBuilder::new().local_anchor1(rv(a1)).local_anchor2(rv(a2));
                if j.limits_enabled {
                    b = b.limits(lim(true));
                }
                if j.motor_enabled {
                    b = b.motor_velocity(j.motor_speed.to_radians(), MOTOR_GAIN).motor_max_force(j.motor_max_force);
                }
                b.build().into()
            }
            Joint2DKind::Slider => {
                let mut b = PrismaticJointBuilder::new(rv(axis1)).local_anchor1(rv(a1)).local_anchor2(rv(a2)).local_axis2(rv(axis2));
                if j.limits_enabled {
                    b = b.limits(lim(false));
                }
                if j.motor_enabled {
                    b = b.motor_velocity(j.motor_speed, MOTOR_GAIN).motor_max_force(j.motor_max_force);
                }
                b.build().into()
            }
            Joint2DKind::Spring => SpringJointBuilder::new(rest, j.stiffness, j.damping)
                .local_anchor1(rv(a1))
                .local_anchor2(rv(far2))
                .build()
                .into(),
            Joint2DKind::Distance => {
                // The two anchors may be [min, max] apart: both coupled linear
                // axes limited together is a distance constraint.
                let range = if j.max_distance_only { [0.0, rest] } else { [rest, rest] };
                GenericJointBuilder::new(JointAxesMask::empty())
                    .local_anchor1(rv(a1))
                    .local_anchor2(rv(far2))
                    .coupled_axes(JointAxesMask::LIN_X | JointAxesMask::LIN_Y)
                    .limits(JointAxis::LinX, range)
                    .build()
            }
            Joint2DKind::Wheel => {
                // Frames rotated so their Y axis is the suspension axis: X is
                // locked, Y slides on a spring, rotation spins freely.
                let tilt = axis_deg - std::f32::consts::FRAC_PI_2;
                let mut b = GenericJointBuilder::new(JointAxesMask::LIN_X)
                    .local_frame1(Pose::from_parts(rv(a1), Rotation::from_angle(tilt)))
                    .local_frame2(Pose::from_parts(rv(a2), Rotation::from_angle(tilt + a1r - a2r)))
                    .motor_position(JointAxis::LinY, 0.0, j.stiffness, j.damping);
                if j.limits_enabled {
                    b = b.limits(JointAxis::LinY, lim(false));
                }
                if j.motor_enabled {
                    b = b
                        .motor_velocity(JointAxis::AngX, j.motor_speed.to_radians(), MOTOR_GAIN)
                        .motor_max_force(JointAxis::AngX, j.motor_max_force);
                }
                b.build()
            }
        };
        data.set_contacts_enabled(j.collide_connected);
        let h = self.impulse_joints.insert(b1, b2, data, true);
        self.joints.insert(id, h);
        let _ = scene;
    }

    /// Drive the motor of a Hinge / Slider / Wheel joint on object `id`.
    pub fn set_joint_motor(&mut self, id: ObjectId, kind: Joint2DKind, speed: f32, max_force: f32) {
        let Some(&h) = self.joints.get(&id) else { return };
        let (axis, speed) = match kind {
            Joint2DKind::Hinge => (JointAxis::AngX, speed.to_radians()),
            Joint2DKind::Wheel => (JointAxis::AngX, speed.to_radians()),
            Joint2DKind::Slider => (JointAxis::LinX, speed),
            _ => return,
        };
        if let Some(joint) = self.impulse_joints.get_mut(h, true) {
            joint.data.set_motor_velocity(axis, speed, MOTOR_GAIN);
            joint.data.set_motor_max_force(axis, max_force);
        }
    }

    /// Replace (or with `None`, remove) the compound collider `key` of the
    /// static body of `owner` (a tilemap chunk, say): a local broadphase
    /// update, not a rebuild of the body. `shape` is in the owner's body space.
    /// Returns false if `owner` has no body.
    pub fn set_compound(&mut self, owner: ObjectId, key: u64, shape: Option<SharedShape>, props: &CompoundProps, scene: &LoadedScene) -> bool {
        let Some(body) = self.bodies.get_mut(&owner) else { return false };
        let handle = body.handle;
        let old = body.compounds.remove(&key);
        let layer = body.layer;
        let body_material = body.material.clone();
        if let Some(old) = old {
            self.colliders.remove(old, &mut self.islands, &mut self.rigid_bodies, true);
            self.volume_effectors.retain(|e| e.collider != old);
            self.platforms.remove(&old);
            self.surfaces.remove(&old);
        }
        let Some(shape) = shape else { return true };
        let mat_path = if props.material.is_empty() { body_material } else { props.material.clone() };
        let default_path = self.default_material.clone();
        let mat = self.material(&mat_path).or_else(|| self.material(&default_path));
        let m = mat.unwrap_or(PhysicsMaterial { friction: DEFAULT_FRICTION, ..Default::default() });
        let groups = InteractionGroups::new(
            Group::from_bits_truncate(1u32 << (layer as u32 & 31)),
            Group::from_bits_truncate(scene.layers.collision_mask(layer)),
            InteractionTestMode::And,
        );
        let mut cb = ColliderBuilder::new(shape)
            .restitution(m.bounciness.clamp(0.0, 1.0))
            .restitution_combine_rule(combine_rule(m.bounce_combine))
            .friction(m.friction.max(0.0))
            .friction_combine_rule(combine_rule(m.friction_combine))
            .collision_groups(groups)
            .active_events(ActiveEvents::COLLISION_EVENTS)
            .user_data(owner.raw())
            .density(0.0);
        if props.is_trigger {
            cb = cb.sensor(true).active_collision_types(ActiveCollisionTypes::all());
        }
        let h = self.colliders.insert_with_parent(cb.build(), handle, &mut self.rigid_bodies);
        self.collider_owner.insert(h, (owner, owner));
        if let Some(b) = self.bodies.get_mut(&owner) {
            b.compounds.insert(key, h);
        }
        true
    }

    /// Push gameplay-driven transforms into the simulation: kinematic bodies
    /// get their next target pose; any other body whose scene transform changed
    /// since the last write-back is teleported.
    pub fn push_poses(&mut self, scene: &LoadedScene, id_index: &HashMap<ObjectId, usize>, world: &[Mat4]) {
        for (id, body) in &mut self.bodies {
            let Some(&i) = id_index.get(id) else { continue };
            let object = &scene.objects[i];
            let (p, a, _) = decompose(world[i]);
            let Some(rb) = self.rigid_bodies.get_mut(body.handle) else { continue };
            let moved = (object.translation, object.rotation) != body.written;
            if body.kind == BodyType2D::Kinematic {
                rb.set_next_kinematic_position(pose2(p, a));
            } else if moved {
                rb.set_position(pose2(p, a), true);
                body.prev = (p, a);
                body.curr = (p, a);
            }
            body.written = (object.translation, object.rotation);
        }
    }

    /// Advance one fixed step and queue the resulting contact events.
    pub fn step(&mut self) {
        self.params.dt = FIXED_DT;
        self.apply_effectors();
        let hooks = Hooks2D { platforms: &self.platforms, surfaces: &self.surfaces, steps: self.steps };
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
            &hooks,
            &self.collector,
        );
        self.steps += 1;
        // Forces last one step, like Unity's.
        if std::mem::take(&mut self.forces_dirty) {
            for (_, rb) in self.rigid_bodies.iter_mut() {
                if rb.is_dynamic() {
                    rb.reset_forces(false);
                }
            }
        }
        for body in self.bodies.values_mut() {
            if body.kind != BodyType2D::Dynamic {
                continue;
            }
            if let Some(rb) = self.rigid_bodies.get(body.handle) {
                body.prev = body.curr;
                body.curr = (gv(rb.translation()), rb.rotation().angle());
            }
        }
        while let Ok(event) = self.collision_rx.try_recv() {
            let (h1, h2) = (event.collider1(), event.collider2());
            let (Some(&(a, a_body)), Some(&(b, b_body))) = (self.collider_owner.get(&h1), self.collider_owner.get(&h2)) else { continue };
            let trigger = match event {
                CollisionEvent::Started(_, _, f) | CollisionEvent::Stopped(_, _, f) => f.contains(CollisionEventFlags::SENSOR),
            };
            if event.started() {
                self.active.insert((h1, h2, trigger));
            } else {
                self.active.remove(&(h1, h2, trigger));
            }
            self.pending.push(ContactEvent2D { a, b, a_body, b_body, started: event.started(), trigger });
            if event.stopped() {
                for h in [h1, h2] {
                    if self.colliders.get(h).is_none() {
                        self.collider_owner.remove(&h);
                    }
                }
            }
        }
    }

    pub fn take_events(&mut self) -> Vec<ContactEvent2D> {
        std::mem::take(&mut self.pending)
    }

    /// Fill `out` with a "stay" event for every pair still touching, except
    /// pairs where both sides are asleep or static (a settled pile costs
    /// nothing). Order follows the hash set; stay callbacks must not feed back
    /// into physics.
    pub fn stay_events(&self, out: &mut Vec<ContactEvent2D>) {
        out.clear();
        let at_rest = |h: ColliderHandle| {
            self.colliders
                .get(h)
                .and_then(|c| c.parent())
                .and_then(|p| self.rigid_bodies.get(p))
                .is_none_or(|rb| rb.is_sleeping() || rb.is_fixed())
        };
        for &(h1, h2, trigger) in &self.active {
            if at_rest(h1) && at_rest(h2) {
                continue;
            }
            let (Some(&(a, a_body)), Some(&(b, b_body))) = (self.collider_owner.get(&h1), self.collider_owner.get(&h2)) else { continue };
            out.push(ContactEvent2D { a, b, a_body, b_body, started: true, trigger });
        }
    }

    /// Write dynamic bodies back to the scene, interpolated `alpha` (0..1)
    /// between the last two steps (or not, or extrapolated, per body).
    pub fn write_back(&mut self, scene: &mut LoadedScene, id_index: &HashMap<ObjectId, usize>, alpha: f32) {
        for (id, body) in &mut self.bodies {
            if body.kind != BodyType2D::Dynamic {
                continue;
            }
            let Some(&i) = id_index.get(id) else { continue };
            let (p, a) = match body.interp {
                Interpolation2D::None => body.curr,
                Interpolation2D::Interpolate => (body.prev.0.lerp(body.curr.0, alpha), lerp_angle(body.prev.1, body.curr.1, alpha)),
                Interpolation2D::Extrapolate => (
                    body.curr.0 + (body.curr.0 - body.prev.0) * alpha,
                    body.curr.1 + (body.curr.1 - body.prev.1) * alpha,
                ),
            };
            let local = world_to_local(scene, i, p, a);
            scene.objects[i].translation = local.0;
            scene.objects[i].rotation = local.1;
            body.written = local;
        }
    }

    /// Mass (kg) and collider area (m^2) of a body, for the inspector.
    pub fn mass_and_area(&self, id: ObjectId) -> Option<(f32, f32)> {
        let body = self.bodies.get(&id)?;
        let rb = self.rigid_bodies.get(body.handle)?;
        let area: f32 = rb
            .colliders()
            .iter()
            .filter_map(|h| self.colliders.get(*h))
            .filter(|c| !c.is_sensor())
            .map(|c| c.shape().mass_properties(1.0).mass())
            .sum();
        Some((rb.mass(), area))
    }

    // ---------------------------------------------------------- body access

    fn dynamic_body(&mut self, id: ObjectId) -> Option<&mut rapier2d::dynamics::RigidBody> {
        let body = self.bodies.get(&id)?;
        if body.kind != BodyType2D::Dynamic {
            return None;
        }
        self.rigid_bodies.get_mut(body.handle)
    }

    pub fn add_force(&mut self, id: ObjectId, force: Vec2) {
        self.forces_dirty = true;
        if let Some(rb) = self.dynamic_body(id) {
            rb.add_force(rv(force), true);
        }
    }

    pub fn add_impulse(&mut self, id: ObjectId, impulse: Vec2) {
        if let Some(rb) = self.dynamic_body(id) {
            rb.apply_impulse(rv(impulse), true);
        }
    }

    pub fn add_torque(&mut self, id: ObjectId, torque: f32) {
        if let Some(rb) = self.dynamic_body(id) {
            rb.apply_torque_impulse(torque * FIXED_DT, true);
        }
    }

    pub fn set_velocity(&mut self, id: ObjectId, linear: Vec2) {
        if let Some(rb) = self.dynamic_body(id) {
            rb.set_linvel(rv(linear), true);
        }
    }

    pub fn set_angular_velocity(&mut self, id: ObjectId, w: f32) {
        if let Some(rb) = self.dynamic_body(id) {
            rb.set_angvel(w, true);
        }
    }

    pub fn velocity(&self, id: ObjectId) -> Option<Vec2> {
        let body = self.bodies.get(&id)?;
        self.rigid_bodies.get(body.handle).map(|rb| gv(rb.linvel()))
    }

    pub fn angular_velocity(&self, id: ObjectId) -> Option<f32> {
        let body = self.bodies.get(&id)?;
        self.rigid_bodies.get(body.handle).map(|rb| rb.angvel())
    }

    /// World position and angle of a body right after the latest step.
    pub fn pose(&self, id: ObjectId) -> Option<(Vec2, f32)> {
        self.bodies.get(&id).map(|b| b.curr)
    }

    pub fn is_sleeping(&self, id: ObjectId) -> bool {
        self.bodies.get(&id).and_then(|b| self.rigid_bodies.get(b.handle)).is_some_and(|rb| rb.is_sleeping())
    }

    /// Bit-exact digest of every body's pose and velocity, to compare runs.
    pub fn state_hash(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut feed = |v: f32| {
            h ^= v.to_bits() as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        };
        for (_, rb) in self.rigid_bodies.iter() {
            let (t, r, v) = (rb.translation(), rb.rotation().angle(), rb.linvel());
            for x in [t.x, t.y, r, v.x, v.y, rb.angvel()] {
                feed(x);
            }
        }
        h
    }

    // -------------------------------------------------------------- effectors

    fn apply_effectors(&mut self) {
        if self.volume_effectors.is_empty() {
            return;
        }
        self.forces_dirty = true;
        let g = gv(self.gravity);
        let up = if g.length_squared() > 1e-9 { -g.normalize() } else { Vec2::Y };
        let dt = FIXED_DT;
        for k in 0..self.volume_effectors.len() {
            let (h, e) = (self.volume_effectors[k].collider, &self.volume_effectors[k].params);
            let Some(sensor) = self.colliders.get(h) else { continue };
            let sensor_pose = *sensor.position();
            let sensor_center = gv(sensor_pose.translation);
            let angle = sensor_pose.rotation.angle();
            self.scratch.clear();
            for (c1, c2, intersecting) in self.narrow_phase.intersection_pairs_with(h) {
                if !intersecting {
                    continue;
                }
                let other = if c1 == h { c2 } else { c1 };
                let Some(oc) = self.colliders.get(other) else { continue };
                let Some(parent) = oc.parent() else { continue };
                if oc.collision_groups().memberships.bits() & e.layer_mask == 0 {
                    continue;
                }
                let Some(rb) = self.rigid_bodies.get_mut(parent) else { continue };
                if !rb.is_dynamic() {
                    continue;
                }
                let salt = parent.into_raw_parts().0 as u64;
                let sub_scale = match e.kind {
                    EffectorKind2D::Buoyancy => {
                        let aabb = oc.compute_aabb();
                        let height = (aabb.maxs.y - aabb.mins.y).max(1e-6);
                        let surface = sensor_pose.transform_point(Vector::new(0.0, e.surface_level)).y;
                        let sub = ((surface - aabb.mins.y) / height).clamp(0.0, 1.0);
                        if sub <= 0.0 {
                            continue;
                        }
                        let area = oc.shape().mass_properties(1.0).mass();
                        let lift = up * (e.fluid_density * g.length() * area * sub);
                        // Applied at the middle of the submerged part.
                        let cx = (aabb.mins.x + aabb.maxs.x) * 0.5;
                        let cy = (aabb.mins.y + surface.min(aabb.maxs.y)) * 0.5;
                        rb.add_force_at_point(rv(lift), Vector::new(cx, cy), true);
                        // Flow: drag the body's velocity towards the water's.
                        let flow_dir = Vec2::from_angle(e.flow_angle.to_radians());
                        let flow_mag = e.flow_magnitude + e.flow_variation * jitter(self.steps, salt);
                        let flow = flow_dir * flow_mag;
                        let v = gv(rb.linvel());
                        let kd = (e.linear_drag * sub * dt).min(1.0);
                        rb.set_linvel(rv(v + (flow - v) * kd), false);
                        rb.set_angvel(rb.angvel() * (1.0 - (e.angular_drag * sub * dt).min(1.0)), false);
                        continue;
                    }
                    _ => 1.0,
                };
                let body_center = gv(rb.translation());
                match e.kind {
                    EffectorKind2D::Area => {
                        let a = e.force_angle.to_radians() + if e.use_global_angle { 0.0 } else { angle };
                        let mag = e.force_magnitude + e.force_variation * jitter(self.steps, salt);
                        rb.add_force(rv(Vec2::from_angle(a) * mag * sub_scale), true);
                        let v = gv(rb.linvel());
                        rb.set_linvel(rv(v * (1.0 - (e.linear_drag * dt).min(1.0))), false);
                        rb.set_angvel(rb.angvel() * (1.0 - (e.angular_drag * dt).min(1.0)), false);
                    }
                    EffectorKind2D::Point => {
                        let to = sensor_center - body_center;
                        let d = to.length();
                        if d < 1e-6 {
                            continue;
                        }
                        let scaled = (d * e.distance_scale).max(1.0);
                        let falloff = match e.force_mode {
                            PointForceMode2D::Constant => 1.0,
                            PointForceMode2D::InverseLinear => 1.0 / scaled,
                            PointForceMode2D::InverseSquared => 1.0 / (scaled * scaled),
                        };
                        let mag = (e.force_magnitude + e.force_variation * jitter(self.steps, salt)) * falloff;
                        rb.add_force(rv(to / d * mag), true);
                        let v = gv(rb.linvel());
                        rb.set_linvel(rv(v * (1.0 - (e.linear_drag * dt).min(1.0))), false);
                    }
                    _ => {}
                }
            }
        }
    }

    // ---------------------------------------------------------------- queries

    fn query_filter<'a>(filter: &RayFilter, predicate: &'a impl Fn(ColliderHandle, &Collider) -> bool) -> QueryFilter<'a> {
        let mut f = QueryFilter::default().predicate(predicate);
        if !filter.include_triggers {
            f = f.exclude_sensors();
        }
        f
    }

    fn with_pipeline<R>(&self, filter: &RayFilter, f: impl FnOnce(&QueryPipeline) -> R) -> R {
        let (mask, exclude) = (filter.layer_mask, filter.exclude.map(|e| e.raw()));
        let predicate = move |_: ColliderHandle, c: &Collider| c.collision_groups().memberships.bits() & mask != 0 && Some(c.user_data) != exclude;
        let qp = self.broad_phase.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.rigid_bodies,
            &self.colliders,
            Self::query_filter(filter, &predicate),
        );
        f(&qp)
    }

    fn object_of(&self, h: ColliderHandle) -> Option<ObjectId> {
        self.colliders.get(h).map(|c| ObjectId::from_raw(c.user_data))
    }

    pub fn raycast(&self, origin: Vec2, dir: Vec2, max_dist: f32, filter: &RayFilter) -> Option<RayHit2D> {
        let dir = dir.normalize_or_zero();
        if dir == Vec2::ZERO {
            return None;
        }
        self.with_pipeline(filter, |qp| {
            let ray = Ray::new(rv(origin), rv(dir));
            let (h, hit) = qp.cast_ray_and_get_normal(&ray, max_dist, true)?;
            Some(RayHit2D { object: self.object_of(h)?, point: origin + dir * hit.time_of_impact, normal: gv(hit.normal), distance: hit.time_of_impact })
        })
    }

    /// Every collider along the ray, nearest first.
    pub fn raycast_all(&self, origin: Vec2, dir: Vec2, max_dist: f32, filter: &RayFilter) -> Vec<RayHit2D> {
        let dir = dir.normalize_or_zero();
        if dir == Vec2::ZERO {
            return Vec::new();
        }
        let mut out: Vec<RayHit2D> = self.with_pipeline(filter, |qp| {
            let ray = Ray::new(rv(origin), rv(dir));
            qp.intersect_ray(ray, max_dist, true)
                .map(|(_, c, hit)| RayHit2D {
                    object: ObjectId::from_raw(c.user_data),
                    point: origin + dir * hit.time_of_impact,
                    normal: gv(hit.normal),
                    distance: hit.time_of_impact,
                })
                .collect()
        });
        out.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        out
    }

    pub fn overlap_point(&self, point: Vec2, filter: &RayFilter) -> Vec<ObjectId> {
        self.with_pipeline(filter, |qp| {
            let mut out: Vec<ObjectId> = Vec::new();
            for (_, c) in qp.intersect_point(rv(point)) {
                let id = ObjectId::from_raw(c.user_data);
                if !out.contains(&id) {
                    out.push(id);
                }
            }
            out
        })
    }

    fn overlap_shape(&self, pose: Pose, shape: &dyn Shape, filter: &RayFilter) -> Vec<ObjectId> {
        self.with_pipeline(filter, |qp| {
            let mut out: Vec<ObjectId> = Vec::new();
            for (_, c) in qp.intersect_shape(pose, shape) {
                let id = ObjectId::from_raw(c.user_data);
                if !out.contains(&id) {
                    out.push(id);
                }
            }
            out
        })
    }

    pub fn overlap_circle(&self, center: Vec2, radius: f32, filter: &RayFilter) -> Vec<ObjectId> {
        self.overlap_shape(Pose::from_translation(rv(center)), &Ball::new(radius.max(1e-4)), filter)
    }

    pub fn overlap_box(&self, center: Vec2, size: Vec2, angle: f32, filter: &RayFilter) -> Vec<ObjectId> {
        let half = (size * 0.5).max(Vec2::splat(1e-4));
        self.overlap_shape(pose2(center, angle), &Cuboid::new(Vector::new(half.x, half.y)), filter)
    }

    fn cast_shape(&self, pose: Pose, shape: &dyn Shape, dir: Vec2, max_dist: f32, filter: &RayFilter) -> Option<RayHit2D> {
        let dir = dir.normalize_or_zero();
        if dir == Vec2::ZERO {
            return None;
        }
        self.with_pipeline(filter, |qp| {
            let (h, hit) = qp.cast_shape(&pose, rv(dir), shape, ShapeCastOptions::with_max_time_of_impact(max_dist))?;
            Some(RayHit2D { object: self.object_of(h)?, point: gv(hit.witness2), normal: gv(hit.normal2), distance: hit.time_of_impact })
        })
    }

    pub fn circle_cast(&self, center: Vec2, radius: f32, dir: Vec2, max_dist: f32, filter: &RayFilter) -> Option<RayHit2D> {
        self.cast_shape(Pose::from_translation(rv(center)), &Ball::new(radius.max(1e-4)), dir, max_dist, filter)
    }

    pub fn box_cast(&self, center: Vec2, size: Vec2, angle: f32, dir: Vec2, max_dist: f32, filter: &RayFilter) -> Option<RayHit2D> {
        let half = (size * 0.5).max(Vec2::splat(1e-4));
        self.cast_shape(pose2(center, angle), &Cuboid::new(Vector::new(half.x, half.y)), dir, max_dist, filter)
    }

    /// Contact points of body `id` with everything touching it.
    pub fn contacts(&self, id: ObjectId) -> Vec<ContactPoint2D> {
        let mut out = Vec::new();
        let Some(body) = self.bodies.get(&id) else { return out };
        let Some(rb) = self.rigid_bodies.get(body.handle) else { return out };
        for &mine in rb.colliders() {
            for pair in self.narrow_phase.contact_pairs_with(mine) {
                let first = pair.collider1 == mine;
                let other = if first { pair.collider2 } else { pair.collider1 };
                let Some(&(other_obj, _)) = self.collider_owner.get(&other) else { continue };
                for m in &pair.manifolds {
                    let n = gv(m.data.normal);
                    for c in &m.data.solver_contacts {
                        out.push(ContactPoint2D { other: other_obj, point: gv(c.point), normal: if first { -n } else { n }, separation: c.dist });
                    }
                }
            }
        }
        out
    }

    // ------------------------------------------------------------- debug draw

    /// World-space outline segments (x, y, z) of the colliders of every body
    /// that asks for debug draw, or of every body when `all`.
    pub fn debug_lines(&self, all: bool, out: &mut Vec<(Vec3, Vec3)>) {
        for body in self.bodies.values() {
            if !(all || body.debug) {
                continue;
            }
            let Some(rb) = self.rigid_bodies.get(body.handle) else { continue };
            for &h in rb.colliders() {
                if let Some(c) = self.colliders.get(h) {
                    let start = out.len();
                    shape_lines(c.shape(), *c.position(), out);
                    for seg in &mut out[start..] {
                        seg.0.z = body.z;
                        seg.1.z = body.z;
                    }
                }
            }
        }
    }
}

/// Segments outlining `shape` placed at `pose`.
pub fn shape_lines(shape: &dyn Shape, pose: Pose, out: &mut Vec<(Vec3, Vec3)>) {
    let p3 = |v: Vector| Vec3::new(v.x, v.y, 0.0);
    let mut ring = |pts: &[Vector], closed: bool| {
        let n = pts.len();
        for k in 0..if closed { n } else { n.saturating_sub(1) } {
            out.push((p3(pose.transform_point(pts[k])), p3(pose.transform_point(pts[(k + 1) % n]))));
        }
    };
    if let Some(b) = shape.as_ball() {
        let pts: Vec<Vector> = (0..24).map(|k| rv(Vec2::from_angle(k as f32 / 24.0 * std::f32::consts::TAU) * b.radius)).collect();
        ring(&pts, true);
    } else if let Some(c) = shape.as_cuboid() {
        let h = c.half_extents;
        ring(&[Vector::new(-h.x, -h.y), Vector::new(h.x, -h.y), Vector::new(h.x, h.y), Vector::new(-h.x, h.y)], true);
    } else if let Some(c) = shape.as_capsule() {
        let (a, b, r) = (gv(c.segment.a), gv(c.segment.b), c.radius);
        let axis = (b - a).normalize_or(Vec2::Y);
        let base = axis.to_angle();
        let mut pts = Vec::new();
        for k in 0..=12 {
            pts.push(rv(b + Vec2::from_angle(base - std::f32::consts::FRAC_PI_2 + k as f32 / 12.0 * std::f32::consts::PI) * r));
        }
        for k in 0..=12 {
            pts.push(rv(a + Vec2::from_angle(base + std::f32::consts::FRAC_PI_2 + k as f32 / 12.0 * std::f32::consts::PI) * r));
        }
        ring(&pts, true);
    } else if let Some(p) = shape.as_convex_polygon() {
        ring(p.points(), true);
    } else if let Some(pl) = shape.as_polyline() {
        for seg in pl.segments() {
            out.push((p3(pose.transform_point(seg.a)), p3(pose.transform_point(seg.b))));
        }
    } else if let Some(c) = shape.as_compound() {
        for (p, s) in c.shapes() {
            shape_lines(s.as_ref(), pose * *p, out);
        }
    }
}

/// Convert a world pose into object `i`'s local translation / rotation,
/// keeping its Z and its X / Y tilt.
fn world_to_local(scene: &LoadedScene, i: usize, p: Vec2, angle: f32) -> (Vec3, Quat) {
    let o = &scene.objects[i];
    match o.parent {
        Some(par) if par < scene.objects.len() => {
            let parent = scene.world_transform(par);
            let (_, pr, _) = parent.to_scale_rotation_translation();
            let world_z = parent.transform_point3(o.translation).z;
            let world_rot = pr * o.rotation;
            let (_, ey, ex) = world_rot.to_euler(EulerRot::ZYX);
            let new_world = Quat::from_euler(EulerRot::ZYX, angle, ey, ex);
            (parent.inverse().transform_point3(Vec3::new(p.x, p.y, world_z)), pr.inverse() * new_world)
        }
        _ => {
            let (_, ey, ex) = o.rotation.to_euler(EulerRot::ZYX);
            (Vec3::new(p.x, p.y, o.translation.z), Quat::from_euler(EulerRot::ZYX, angle, ey, ex))
        }
    }
}

// --------------------------------------------------------------- structure

/// A body to create: the object owning it and the objects whose colliders
/// belong to it, in scene order.
struct Plan {
    body: usize,
    colliders: Vec<usize>,
}

impl Plan {
    fn collider_ids(&self, scene: &LoadedScene) -> Vec<ObjectId> {
        self.colliders.iter().map(|&c| scene.objects[c].id).collect()
    }
}

fn has_component<T: 'static>(scene: &LoadedScene, i: usize) -> bool {
    scene.objects[i].components.iter().any(|c| c.as_any().is::<T>())
}

fn has_collider(scene: &LoadedScene, i: usize) -> bool {
    scene.objects[i].components.iter().any(|c| {
        let a = c.as_any();
        a.is::<BoxCollider2D>()
            || a.is::<CircleCollider2D>()
            || a.is::<CapsuleCollider2D>()
            || a.is::<PolygonCollider2D>()
            || a.is::<EdgeCollider2D>()
            || a.is::<zestengine_core::TilemapCollider2D>()
    })
}

/// The nearest composite on `i` or above it that belongs to the same body.
fn composite_of(scene: &LoadedScene, i: usize, plan: &Plan) -> Option<usize> {
    let mut at = Some(i);
    while let Some(k) = at {
        if has_component::<CompositeCollider2D>(scene, k) && (k == plan.body || plan.colliders.contains(&k)) {
            return Some(k);
        }
        if k == plan.body {
            return None;
        }
        at = scene.objects[k].parent.filter(|&p| p < scene.objects.len());
    }
    None
}

/// Which 2D bodies the scene wants: every active object with a Rigidbody 2D,
/// plus a static body for each collider with no body above it.
fn plan_bodies(scene: &LoadedScene) -> Vec<Plan> {
    let n = scene.objects.len();
    let mut plans: Vec<Plan> = Vec::new();
    let mut slot: HashMap<usize, usize> = HashMap::new();
    let rb_of = |i: usize| scene.objects[i].components.iter().find_map(|c| c.as_any().downcast_ref::<Rigidbody2D>());
    for i in 0..n {
        if !scene.is_active(i) {
            continue;
        }
        let collider = has_collider(scene, i) || has_component::<CompositeCollider2D>(scene, i);
        let rb = rb_of(i);
        if !collider && rb.is_none() {
            continue;
        }
        // The nearest body at or above.
        let mut owner = None;
        let mut at = Some(i);
        while let Some(k) = at {
            if rb_of(k).is_some() || has_component::<CompositeCollider2D>(scene, k) {
                owner = Some(k);
                break;
            }
            at = scene.objects[k].parent.filter(|&p| p < n);
        }
        let body = owner.unwrap_or(i);
        if rb_of(body).is_some_and(|r| !r.simulated) || !scene.is_active(body) {
            continue;
        }
        let s = *slot.entry(body).or_insert_with(|| {
            plans.push(Plan { body, colliders: Vec::new() });
            plans.len() - 1
        });
        if collider {
            plans[s].colliders.push(i);
        }
    }
    plans
}

// ------------------------------------------------------------------- hooks

struct Hooks2D<'a> {
    platforms: &'a HashMap<ColliderHandle, PlatformInfo>,
    surfaces: &'a HashMap<ColliderHandle, SurfaceInfo>,
    steps: u64,
}

const CONFIG_UNKNOWN: u32 = 0;
const CONFIG_ALLOWED: u32 = 1;
const CONFIG_FORBIDDEN: u32 = 2;

impl PhysicsHooks for Hooks2D<'_> {
    fn modify_solver_contacts(&self, ctx: &mut ContactModificationContext) {
        for (is_first, h) in [(true, ctx.collider1), (false, ctx.collider2)] {
            // The contact normal points from collider 1 to collider 2; these
            // want it pointing out of the effector's collider.
            let n_out = if is_first { gv(*ctx.normal) } else { -gv(*ctx.normal) };
            if let Some(p) = self.platforms.get(&h) {
                let Some(c) = ctx.colliders.get(h) else { continue };
                let up = gv(c.position().rotation * rv(p.up_local));
                let in_arc = n_out.dot(up) >= p.cos_arc;
                if p.one_way {
                    // Same state machine as rapier's one-way helper: forbid the
                    // contact until the bodies separate once it starts from the
                    // wrong side.
                    match *ctx.user_data {
                        CONFIG_UNKNOWN => {
                            if in_arc {
                                *ctx.user_data = CONFIG_ALLOWED;
                            } else {
                                ctx.solver_contacts.clear();
                                if n_out.length_squared() > 0.1 {
                                    *ctx.user_data = CONFIG_FORBIDDEN;
                                }
                            }
                        }
                        CONFIG_FORBIDDEN => {
                            if in_arc && ctx.solver_contacts.iter().all(|c| c.dist > 0.0) {
                                *ctx.user_data = CONFIG_ALLOWED;
                            } else {
                                ctx.solver_contacts.clear();
                            }
                        }
                        _ => {
                            if ctx.solver_contacts.is_empty() {
                                *ctx.user_data = CONFIG_UNKNOWN;
                            }
                        }
                    }
                } else if !in_arc {
                    for c in ctx.solver_contacts.iter_mut() {
                        if !p.side_friction {
                            c.friction = 0.0;
                        }
                        if !p.side_bounce {
                            c.restitution = 0.0;
                        }
                    }
                }
            }
            if let Some(s) = self.surfaces.get(&h) {
                let speed = s.speed + s.variation * jitter(self.steps, h.into_raw_parts().0 as u64);
                // Along the surface, to the right of its outward normal; the
                // solver wants the velocity of collider 2 relative to 1.
                let along = Vec2::new(n_out.y, -n_out.x) * speed;
                let t = if is_first { along } else { -along };
                for c in ctx.solver_contacts.iter_mut() {
                    c.tangent_velocity = rv(t);
                }
            }
        }
    }
}

// ------------------------------------------------------------ session glue

/// The 2D world as the play loop holds it: created the first time a scene has
/// 2D objects, so a scene without any costs one bool check per structure change
/// and nothing per step.
pub struct Physics2D {
    world: Option<Box<PhysicsWorld2D>>,
    dirty: bool,
    root: Option<PathBuf>,
    settings: PhysicsSettings,
}

impl Physics2D {
    pub fn new(root: Option<&Path>, settings: &PhysicsSettings) -> Self {
        Self { world: None, dirty: true, root: root.map(Path::to_owned), settings: settings.clone() }
    }

    /// New project physics settings: the world is rebuilt on the next sync.
    pub fn set_settings(&mut self, settings: &PhysicsSettings) {
        self.settings = settings.clone();
        self.world = None;
        self.dirty = true;
    }

    /// A different scene: start over.
    pub fn reset(&mut self) {
        self.world = None;
        self.dirty = true;
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
        if let Some(w) = self.world.as_mut() {
            w.mark_dirty();
        }
    }

    /// The world, if the scene has 2D objects.
    pub fn world(&self) -> Option<&PhysicsWorld2D> {
        self.world.as_deref()
    }

    pub fn world_mut(&mut self) -> Option<&mut PhysicsWorld2D> {
        self.world.as_deref_mut()
    }

    /// Whether anything changed shape this sync (the caller refreshes derived
    /// values such as the inspector's mass).
    pub fn sync_structure(&mut self, scene: &LoadedScene) -> bool {
        if !self.dirty {
            return false;
        }
        self.dirty = false;
        match self.world.as_mut() {
            Some(w) => {
                w.mark_dirty();
                w.sync_structure(scene);
            }
            None if scene_has_2d(scene) => {
                self.world = Some(Box::new(PhysicsWorld2D::build_with(scene, self.root.as_deref(), &self.settings)));
            }
            None => return false,
        }
        true
    }

    /// Regenerate tilemap collider chunks that changed (no-op without tilemaps).
    pub fn sync_tilemaps(&mut self, scene: &LoadedScene) {
        if let Some(w) = self.world.as_mut() {
            w.sync_tilemaps(scene);
        }
    }

    pub fn push_poses(&mut self, scene: &LoadedScene, ids: &HashMap<ObjectId, usize>, world: &[Mat4]) {
        if let Some(w) = self.world.as_mut() {
            w.push_poses(scene, ids, world);
        }
    }

    pub fn step(&mut self) {
        if let Some(w) = self.world.as_mut() {
            w.step();
        }
    }

    pub fn write_back(&mut self, scene: &mut LoadedScene, ids: &HashMap<ObjectId, usize>, alpha: f32) {
        if let Some(w) = self.world.as_mut() {
            w.write_back(scene, ids, alpha);
        }
    }

    pub fn take_events(&mut self) -> Vec<ContactEvent2D> {
        match self.world.as_mut() {
            Some(w) => w.take_events(),
            None => Vec::new(),
        }
    }

    /// Refresh each Rigidbody 2D's measured mass / area for the inspector.
    pub fn write_measured(&self, scene: &mut LoadedScene) {
        let Some(w) = self.world.as_ref() else { return };
        for o in scene.objects.iter_mut() {
            let id = o.id;
            if let Some(rb) = o.components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<Rigidbody2D>()) {
                rb.measured = w.mass_and_area(id);
            }
        }
    }

    pub fn add_force(&mut self, id: ObjectId, force: Vec3) {
        if let Some(w) = self.world.as_mut() {
            w.add_force(id, force.truncate());
        }
    }

    pub fn add_impulse(&mut self, id: ObjectId, impulse: Vec3) {
        if let Some(w) = self.world.as_mut() {
            w.add_impulse(id, impulse.truncate());
        }
    }

    pub fn add_torque(&mut self, id: ObjectId, torque: Vec3) {
        if let Some(w) = self.world.as_mut() {
            w.add_torque(id, torque.z);
        }
    }

    pub fn set_velocity(&mut self, id: ObjectId, linear: Vec3) {
        if let Some(w) = self.world.as_mut() {
            w.set_velocity(id, linear.truncate());
        }
    }

    pub fn set_joint_motor(&mut self, scene: &LoadedScene, id: ObjectId, speed: f32, max_force: f32) {
        let Some(w) = self.world.as_mut() else { return };
        let kind = scene.index_of(id).and_then(|i| scene.objects[i].components.iter().find_map(|c| c.as_any().downcast_ref::<Joint2D>())).map(|j| j.kind);
        if let Some(kind) = kind {
            w.set_joint_motor(id, kind, speed, max_force);
        }
    }

    pub fn debug_lines(&self, all: bool, out: &mut Vec<(Vec3, Vec3)>) {
        if let Some(w) = self.world.as_ref() {
            w.debug_lines(all, out);
        }
    }
}

/// The 3D and 2D worlds as one [`zestengine_core::PhysicsQuery`], so
/// `ComponentCtx` reaches both through one reference.
pub struct QueryHub<'a> {
    pub world3: &'a PhysicsWorld3D,
    pub world2: Option<&'a PhysicsWorld2D>,
}

impl zestengine_core::PhysicsQuery for QueryHub<'_> {
    fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32, filter: &RayFilter) -> Option<zestengine_core::RayHit> {
        self.world3.raycast(origin, dir, max_dist, filter)
    }
    fn overlap_sphere(&self, center: Vec3, radius: f32, filter: &RayFilter) -> Vec<ObjectId> {
        self.world3.overlap_sphere(center, radius, filter)
    }
    fn velocity(&self, id: ObjectId) -> Option<Vec3> {
        self.world3.velocity(id).or_else(|| self.world2?.velocity(id).map(|v| v.extend(0.0)))
    }
    fn grounded(&self, id: ObjectId) -> bool {
        self.world3.grounded(id)
    }
    fn raycast2d(&self, origin: Vec2, dir: Vec2, max_dist: f32, filter: &RayFilter) -> Option<RayHit2D> {
        self.world2?.raycast(origin, dir, max_dist, filter)
    }
    fn raycast_all2d(&self, origin: Vec2, dir: Vec2, max_dist: f32, filter: &RayFilter) -> Vec<RayHit2D> {
        self.world2.map(|w| w.raycast_all(origin, dir, max_dist, filter)).unwrap_or_default()
    }
    fn overlap_point2d(&self, point: Vec2, filter: &RayFilter) -> Vec<ObjectId> {
        self.world2.map(|w| w.overlap_point(point, filter)).unwrap_or_default()
    }
    fn overlap_circle2d(&self, center: Vec2, radius: f32, filter: &RayFilter) -> Vec<ObjectId> {
        self.world2.map(|w| w.overlap_circle(center, radius, filter)).unwrap_or_default()
    }
    fn overlap_box2d(&self, center: Vec2, size: Vec2, angle: f32, filter: &RayFilter) -> Vec<ObjectId> {
        self.world2.map(|w| w.overlap_box(center, size, angle, filter)).unwrap_or_default()
    }
    fn circle_cast2d(&self, center: Vec2, radius: f32, dir: Vec2, max_dist: f32, filter: &RayFilter) -> Option<RayHit2D> {
        self.world2?.circle_cast(center, radius, dir, max_dist, filter)
    }
    fn box_cast2d(&self, center: Vec2, size: Vec2, angle: f32, dir: Vec2, max_dist: f32, filter: &RayFilter) -> Option<RayHit2D> {
        self.world2?.box_cast(center, size, angle, dir, max_dist, filter)
    }
    fn contacts2d(&self, id: ObjectId) -> Vec<ContactPoint2D> {
        self.world2.map(|w| w.contacts(id)).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::SceneObject;
    use zestengine_assets::ObjectSource;
    use zestengine_core::{Component, ComponentRegistry};

    fn obj(scene: &mut LoadedScene, pos: Vec3, comps: Vec<Box<dyn Component>>) -> ObjectId {
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
            components: comps,
        });
        id
    }

    fn at(scene: &LoadedScene, id: ObjectId) -> Vec3 {
        scene.objects[scene.index_of(id).unwrap()].translation
    }

    fn body() -> Box<dyn Component> {
        Box::new(Rigidbody2D::default())
    }

    fn rect(w: f32, h: f32) -> Box<dyn Component> {
        Box::new(BoxCollider2D { size: [w, h], ..Default::default() })
    }

    fn ground(scene: &mut LoadedScene, y: f32) -> ObjectId {
        obj(scene, Vec3::new(0.0, y - 0.5, 0.0), vec![rect(100.0, 1.0)])
    }

    fn make(scene: &LoadedScene, mats: &[(&str, PhysicsMaterial)], gravity: [f32; 3]) -> PhysicsWorld2D {
        let settings = PhysicsSettings { gravity, ..Default::default() };
        let mut w = PhysicsWorld2D::build_with(&LoadedScene::empty(), None, &settings);
        for (p, m) in mats {
            w.set_material(p, *m);
        }
        w.mark_dirty();
        w.sync_structure(scene);
        w
    }

    fn run(w: &mut PhysicsWorld2D, scene: &mut LoadedScene, steps: usize) {
        for _ in 0..steps {
            w.sync_structure(scene);
            let ids = scene.id_index();
            let world = scene.world_transforms();
            w.push_poses(scene, &ids, &world);
            w.step();
            w.write_back(scene, &ids, 1.0);
        }
    }

    const G: [f32; 3] = [0.0, -9.81, 0.0];

    #[test]
    fn same_scene_same_steps_gives_identical_state() {
        let build = || {
            let mut scene = LoadedScene::empty();
            ground(&mut scene, 0.0);
            for k in 0..12 {
                let c: Box<dyn Component> = if k % 3 == 0 { Box::new(CircleCollider2D::default()) } else { rect(0.8, 0.6) };
                obj(&mut scene, Vec3::new((k % 4) as f32 * 0.7 - 1.0, 1.0 + k as f32 * 0.9, 0.0), vec![body(), c]);
            }
            let w = make(&scene, &[], G);
            (scene, w)
        };
        let (mut sa, mut wa) = build();
        let (mut sb, mut wb) = build();
        run(&mut wa, &mut sa, 240);
        run(&mut wb, &mut sb, 240);
        assert_eq!(wa.state_hash(), wb.state_hash());
        assert_eq!(wa.body_count(), 13);
    }

    #[test]
    fn box_falls_onto_ground_and_sleeps() {
        let mut scene = LoadedScene::empty();
        ground(&mut scene, 0.0);
        let b = obj(&mut scene, Vec3::new(0.0, 3.0, 0.0), vec![body(), rect(1.0, 1.0)]);
        let mut w = make(&scene, &[], G);
        run(&mut w, &mut scene, 360);
        let y = at(&scene, b).y;
        assert!((y - 0.5).abs() < 0.03, "rests on the ground top (y={y})");
        assert!(w.is_sleeping(b), "settled bodies sleep");
        // Z is untouched and the rotation stays about Z.
        assert_eq!(at(&scene, b).z, 0.0);
    }

    #[test]
    fn restitution_bounces_back() {
        let bouncy = PhysicsMaterial { bounciness: 0.8, bounce_combine: CombineMode::Maximum, ..Default::default() };
        let mut scene = LoadedScene::empty();
        obj(&mut scene, Vec3::new(0.0, -0.5, 0.0), vec![Box::new(BoxCollider2D { size: [50.0, 1.0], material: "bouncy".into(), ..Default::default() })]);
        let ball = obj(
            &mut scene,
            Vec3::new(0.0, 5.0, 0.0),
            vec![body(), Box::new(CircleCollider2D { material: "bouncy".into(), ..Default::default() })],
        );
        let mut w = make(&scene, &[("bouncy", bouncy)], G);
        let (mut impact, mut rebound) = (0.0f32, 0.0f32);
        let mut hit = false;
        for _ in 0..200 {
            let before = w.velocity(ball).unwrap().y;
            run(&mut w, &mut scene, 1);
            let after = w.velocity(ball).unwrap().y;
            if !hit && before < -1.0 && after > 0.0 {
                hit = true;
                impact = before.abs();
                rebound = after;
            }
        }
        assert!(hit, "the ball bounced");
        let ratio = rebound / impact;
        assert!((0.7..0.9).contains(&ratio), "rebound/impact = {ratio}");
    }

    fn slide_distance(friction: f32) -> f32 {
        let m = PhysicsMaterial { friction, friction_combine: CombineMode::Minimum, ..Default::default() };
        let mut scene = LoadedScene::empty();
        let ramp = obj(&mut scene, Vec3::ZERO, vec![Box::new(BoxCollider2D { size: [20.0, 1.0], material: "m".into(), ..Default::default() })]);
        let i = scene.index_of(ramp).unwrap();
        scene.objects[i].rotation = Quat::from_rotation_z(-30f32.to_radians());
        // Sit the box on the ramp's top face, about its middle.
        let up = Vec3::new(30f32.to_radians().sin(), 30f32.to_radians().cos(), 0.0);
        let b = obj(
            &mut scene,
            up * 1.0 + Vec3::new(0.0, 0.0, 0.0),
            vec![body(), Box::new(BoxCollider2D { size: [1.0, 1.0], material: "m".into(), ..Default::default() })],
        );
        let j = scene.index_of(b).unwrap();
        scene.objects[j].rotation = Quat::from_rotation_z(-30f32.to_radians());
        let start = at(&scene, b);
        let mut w = make(&scene, &[("m", m)], G);
        run(&mut w, &mut scene, 120);
        at(&scene, b).distance(start)
    }

    #[test]
    fn friction_decides_whether_a_box_slides_down_a_ramp() {
        let slippery = slide_distance(0.1);
        let grippy = slide_distance(1.0);
        assert!(slippery > 3.0, "low friction slides ({slippery})");
        assert!(grippy < 0.3, "high friction holds ({grippy})");
    }

    #[test]
    fn triggers_report_enter_and_exit() {
        let mut scene = LoadedScene::empty();
        let zone = obj(&mut scene, Vec3::new(0.0, 0.0, 0.0), vec![Box::new(BoxCollider2D { size: [4.0, 2.0], is_trigger: true, ..Default::default() })]);
        let ball = obj(&mut scene, Vec3::new(0.0, 4.0, 0.0), vec![body(), Box::new(CircleCollider2D { radius: 0.25, ..Default::default() })]);
        let mut w = make(&scene, &[], G);
        let mut enters = 0;
        let mut exits = 0;
        let mut stays = Vec::new();
        for _ in 0..180 {
            run(&mut w, &mut scene, 1);
            for e in w.take_events() {
                assert!(e.trigger);
                assert!((e.a == zone && e.b == ball) || (e.a == ball && e.b == zone));
                if e.started { enters += 1 } else { exits += 1 }
            }
            w.stay_events(&mut stays);
        }
        assert_eq!((enters, exits), (1, 1));
        // Falling through is not blocked.
        assert!(at(&scene, ball).y < -3.0);
    }

    #[test]
    fn layer_matrix_filters_collisions() {
        let mut scene = LoadedScene::empty();
        let g = ground(&mut scene, 0.0);
        let gi = scene.index_of(g).unwrap();
        scene.objects[gi].layer = 2;
        let blocked = obj(&mut scene, Vec3::new(-2.0, 2.0, 0.0), vec![body(), rect(1.0, 1.0)]);
        let passes = obj(&mut scene, Vec3::new(2.0, 2.0, 0.0), vec![body(), rect(1.0, 1.0)]);
        let pi = scene.index_of(passes).unwrap();
        scene.objects[pi].layer = 1;
        scene.layers.set_collide(1, 2, false);
        let mut w = make(&scene, &[], G);
        run(&mut w, &mut scene, 240);
        assert!((at(&scene, blocked).y - 0.5).abs() < 0.05, "layer 0 meets layer 2");
        assert!(at(&scene, passes).y < -3.0, "layers 1 and 2 are set to ignore each other");
    }

    #[test]
    fn hinge_pendulum_has_the_physical_period() {
        let theta = 0.1f32;
        let l = 1.0f32;
        let r = 0.05f32;
        let mut scene = LoadedScene::empty();
        let bob = obj(
            &mut scene,
            Vec3::new(l * theta.sin(), -l * theta.cos(), 0.0),
            vec![
                body(),
                Box::new(CircleCollider2D { radius: r, ..Default::default() }),
                Box::new(Joint2D { kind: Joint2DKind::Hinge, anchor: [-l * theta.sin(), l * theta.cos()], ..Default::default() }),
            ],
        );
        let mut w = make(&scene, &[], G);
        // x(t) crosses zero twice per period.
        let mut crossings = Vec::new();
        let mut prev = at(&scene, bob).x;
        for k in 0..400 {
            run(&mut w, &mut scene, 1);
            let x = at(&scene, bob).x;
            if prev > 0.0 && x <= 0.0 || prev < 0.0 && x >= 0.0 {
                crossings.push(k as f32 * FIXED_DT);
            }
            prev = x;
        }
        assert!(crossings.len() >= 3, "{crossings:?}");
        let period = crossings[2] - crossings[0];
        let expected = std::f32::consts::TAU * ((l * l + 0.5 * r * r) / (9.81 * l)).sqrt();
        assert!((period - expected).abs() / expected < 0.03, "period {period} vs {expected}");
        // The hinge holds the length.
        let p = at(&scene, bob);
        assert!((p.truncate().length() - l).abs() < 0.02, "length {}", p.truncate().length());
    }

    #[test]
    fn fixed_distance_and_slider_joints_hold() {
        let mut scene = LoadedScene::empty();
        let anchor = obj(&mut scene, Vec3::new(0.0, 10.0, 0.0), vec![Box::new(Rigidbody2D { body_type: BodyType2D::Static, ..Default::default() }), rect(0.5, 0.5)]);
        // Fixed: hangs without rotating or dropping.
        let fixed = obj(
            &mut scene,
            Vec3::new(2.0, 10.0, 0.0),
            vec![body(), rect(0.5, 0.5), Box::new(Joint2D { kind: Joint2DKind::Fixed, connected: anchor.into(), ..Default::default() })],
        );
        // Distance: a rope-length pendulum.
        let rope = obj(
            &mut scene,
            Vec3::new(-3.0, 10.0, 0.0),
            vec![
                body(),
                Box::new(CircleCollider2D { radius: 0.2, ..Default::default() }),
                Box::new(Joint2D { kind: Joint2DKind::Distance, connected: anchor.into(), ..Default::default() }),
            ],
        );
        // Slider: only slides along x, limited to 1 m.
        let slider = obj(
            &mut scene,
            Vec3::new(0.0, 6.0, 0.0),
            vec![
                body(),
                rect(0.5, 0.5),
                Box::new(Joint2D { kind: Joint2DKind::Slider, axis_angle: 0.0, limits_enabled: true, limits: [-1.0, 1.0], ..Default::default() }),
            ],
        );
        let mut w = make(&scene, &[], G);
        run(&mut w, &mut scene, 300);
        let f = at(&scene, fixed);
        assert!(f.distance(Vec3::new(2.0, 10.0, 0.0)) < 0.05, "fixed joint holds: {f}");
        let r = at(&scene, rope);
        assert!((r.truncate().distance(Vec2::new(0.0, 10.0)) - 3.0).abs() < 0.05, "distance joint keeps its length: {r}");
        let s = at(&scene, slider);
        assert!((s.y - 6.0).abs() < 0.05, "slider does not fall: {s}");
    }

    #[test]
    fn area_and_point_effectors_push_bodies() {
        let mut scene = LoadedScene::empty();
        obj(
            &mut scene,
            Vec3::ZERO,
            vec![
                Box::new(BoxCollider2D { size: [6.0, 6.0], is_trigger: true, used_by_effector: true, ..Default::default() }),
                Box::new(Effector2D { kind: EffectorKind2D::Area, force_angle: 0.0, force_magnitude: 20.0, ..Default::default() }),
            ],
        );
        let a = obj(&mut scene, Vec3::new(-2.0, 0.0, 0.0), vec![body(), rect(0.5, 0.5)]);
        obj(
            &mut scene,
            Vec3::new(0.0, 20.0, 0.0),
            vec![
                Box::new(CircleCollider2D { radius: 3.0, is_trigger: true, used_by_effector: true, ..Default::default() }),
                Box::new(Effector2D { kind: EffectorKind2D::Point, force_magnitude: 30.0, ..Default::default() }),
            ],
        );
        let p = obj(&mut scene, Vec3::new(2.0, 20.0, 0.0), vec![body(), rect(0.5, 0.5)]);
        let mut w = make(&scene, &[], [0.0, 0.0, 0.0]);
        run(&mut w, &mut scene, 30);
        assert!(w.velocity(a).unwrap().x > 1.0, "area force pushes along +x");
        assert!(w.velocity(p).unwrap().x < -1.0, "positive point force attracts towards the effector");
    }

    #[test]
    fn buoyancy_floats_light_bodies_and_sinks_heavy_ones() {
        let mut scene = LoadedScene::empty();
        // Water: surface at y = 0, 10 m deep.
        obj(
            &mut scene,
            Vec3::new(0.0, -5.0, 0.0),
            vec![
                Box::new(BoxCollider2D { size: [20.0, 10.0], is_trigger: true, used_by_effector: true, ..Default::default() }),
                Box::new(Effector2D { kind: EffectorKind2D::Buoyancy, surface_level: 5.0, fluid_density: 1.0, linear_drag: 2.0, angular_drag: 1.0, ..Default::default() }),
            ],
        );
        let light = obj(&mut scene, Vec3::new(-3.0, -1.0, 0.0), vec![body(), Box::new(BoxCollider2D { density: 0.3, ..Default::default() })]);
        let heavy = obj(&mut scene, Vec3::new(3.0, -1.0, 0.0), vec![body(), Box::new(BoxCollider2D { density: 3.0, ..Default::default() })]);
        let mut w = make(&scene, &[], G);
        run(&mut w, &mut scene, 600);
        let ly = at(&scene, light).y;
        assert!(ly > -0.6 && ly < 0.6, "light body bobs at the surface (y={ly})");
        assert!(at(&scene, heavy).y < -7.0, "heavy body sinks (y={})", at(&scene, heavy).y);
    }

    #[test]
    fn one_way_platform_lets_bodies_up_but_not_down() {
        let platform = |scene: &mut LoadedScene| {
            obj(
                scene,
                Vec3::new(0.0, 0.0, 0.0),
                vec![
                    Box::new(BoxCollider2D { size: [6.0, 0.4], used_by_effector: true, ..Default::default() }),
                    Box::new(Effector2D { kind: EffectorKind2D::Platform, surface_arc: 90.0, ..Default::default() }),
                ],
            )
        };
        // From above: lands.
        let mut scene = LoadedScene::empty();
        platform(&mut scene);
        let down = obj(&mut scene, Vec3::new(0.0, 3.0, 0.0), vec![body(), rect(0.5, 0.5)]);
        let mut w = make(&scene, &[], G);
        run(&mut w, &mut scene, 240);
        assert!((at(&scene, down).y - 0.45).abs() < 0.05, "lands on top: {}", at(&scene, down).y);
        // From below, moving up: passes through and then lands on top.
        let mut scene = LoadedScene::empty();
        platform(&mut scene);
        let up = obj(&mut scene, Vec3::new(0.0, -2.0, 0.0), vec![body(), rect(0.5, 0.5)]);
        let mut w = make(&scene, &[], G);
        w.set_velocity(up, Vec2::new(0.0, 12.0));
        run(&mut w, &mut scene, 300);
        assert!((at(&scene, up).y - 0.45).abs() < 0.05, "jumped up through, landed on top: {}", at(&scene, up).y);
    }

    #[test]
    fn surface_effector_moves_what_rests_on_it() {
        // Either creation order (so either side of the contact pair is the conveyor).
        for conveyor_first in [true, false] {
            let mut scene = LoadedScene::empty();
            let belt = |scene: &mut LoadedScene| {
                obj(
                    scene,
                    Vec3::new(0.0, -0.25, 0.0),
                    vec![
                        Box::new(BoxCollider2D { size: [30.0, 0.5], used_by_effector: true, ..Default::default() }),
                        Box::new(Effector2D { kind: EffectorKind2D::Surface, speed: 2.0, ..Default::default() }),
                    ],
                )
            };
            let b;
            if conveyor_first {
                belt(&mut scene);
                b = obj(&mut scene, Vec3::new(0.0, 0.5, 0.0), vec![body(), rect(1.0, 1.0)]);
            } else {
                b = obj(&mut scene, Vec3::new(0.0, 0.5, 0.0), vec![body(), rect(1.0, 1.0)]);
                belt(&mut scene);
            }
            let mut w = make(&scene, &[], G);
            run(&mut w, &mut scene, 180);
            let x = at(&scene, b).x;
            assert!(x > 2.0, "a conveyor with positive speed carries the box right (x={x}, conveyor first: {conveyor_first})");
            assert!(x < 12.0, "and no faster than its speed ({x})");
        }
    }

    #[test]
    fn queries_find_colliders() {
        let mut scene = LoadedScene::empty();
        let wall = obj(&mut scene, Vec3::new(5.0, 0.0, 0.0), vec![rect(2.0, 4.0)]);
        let trig = obj(&mut scene, Vec3::new(-5.0, 0.0, 0.0), vec![Box::new(BoxCollider2D { size: [2.0, 2.0], is_trigger: true, ..Default::default() })]);
        let mut w = make(&scene, &[], G);
        w.step(); // the broadphase is built by a step
        let f = RayFilter::default();
        let hit = w.raycast(Vec2::ZERO, Vec2::X, 20.0, &f).expect("hits the wall");
        assert_eq!(hit.object, wall);
        assert!((hit.distance - 4.0).abs() < 1e-3 && hit.normal.x < -0.99);
        assert!(w.raycast(Vec2::ZERO, Vec2::X, 3.0, &f).is_none(), "too short");
        assert!(w.raycast(Vec2::ZERO, Vec2::NEG_X, 20.0, &f).is_none(), "triggers are ignored by default");
        let with_triggers = RayFilter { include_triggers: true, ..Default::default() };
        assert_eq!(w.raycast(Vec2::ZERO, Vec2::NEG_X, 20.0, &with_triggers).unwrap().object, trig);
        let skip = RayFilter { exclude: Some(wall), ..Default::default() };
        assert!(w.raycast(Vec2::ZERO, Vec2::X, 20.0, &skip).is_none());
        let none = RayFilter { layer_mask: 1 << 5, ..Default::default() };
        assert!(w.raycast(Vec2::ZERO, Vec2::X, 20.0, &none).is_none(), "layer mask");
        assert_eq!(w.overlap_circle(Vec2::new(3.5, 0.0), 0.6, &f), vec![wall]);
        assert!(w.overlap_circle(Vec2::new(0.0, 0.0), 0.6, &f).is_empty());
        assert_eq!(w.overlap_box(Vec2::new(5.0, 1.5), Vec2::new(1.0, 1.0), 0.3, &f), vec![wall]);
        assert_eq!(w.overlap_point(Vec2::new(5.5, 1.0), &f), vec![wall]);
        assert!(w.overlap_point(Vec2::new(0.0, 0.0), &f).is_empty());
        assert_eq!(w.raycast_all(Vec2::new(-10.0, 0.0), Vec2::X, 30.0, &with_triggers).len(), 2);
        let cast = w.circle_cast(Vec2::ZERO, 0.5, Vec2::X, 20.0, &f).expect("circle hits the wall");
        assert!((cast.distance - 3.5).abs() < 0.02, "swept circle stops at {}", cast.distance);
        let bc = w.box_cast(Vec2::ZERO, Vec2::new(1.0, 1.0), 0.0, Vec2::X, 20.0, &f).expect("box hits the wall");
        assert!((bc.distance - 3.5).abs() < 0.02);
    }

    #[test]
    fn contact_points_are_reported() {
        let mut scene = LoadedScene::empty();
        let g = ground(&mut scene, 0.0);
        let b = obj(&mut scene, Vec3::new(0.0, 0.52, 0.0), vec![body(), rect(1.0, 1.0)]);
        let mut w = make(&scene, &[], G);
        run(&mut w, &mut scene, 30);
        let pts = w.contacts(b);
        assert!(!pts.is_empty());
        assert!(pts.iter().all(|c| c.other == g && c.normal.y > 0.99), "{pts:?}");
        assert!(pts.iter().all(|c| (c.point.y - 0.0).abs() < 0.06));
    }

    #[test]
    fn kinematic_platform_carries_a_body() {
        let mut scene = LoadedScene::empty();
        let plat = obj(
            &mut scene,
            Vec3::new(0.0, -0.25, 0.0),
            vec![Box::new(Rigidbody2D { body_type: BodyType2D::Kinematic, ..Default::default() }), rect(10.0, 0.5)],
        );
        let rider = obj(
            &mut scene,
            Vec3::new(0.0, 0.5, 0.0),
            vec![body(), Box::new(BoxCollider2D { size: [1.0, 1.0], material: "grip".into(), ..Default::default() })],
        );
        let grip = PhysicsMaterial { friction: 1.0, friction_combine: CombineMode::Maximum, ..Default::default() };
        let mut w = make(&scene, &[("grip", grip)], G);
        let pi = scene.index_of(plat).unwrap();
        for _ in 0..120 {
            scene.objects[pi].translation.x += 1.0 * FIXED_DT;
            run(&mut w, &mut scene, 1);
        }
        let dx = at(&scene, rider).x - 0.0;
        let px = at(&scene, plat).x;
        assert!(px > 1.9, "the platform moved ({px})");
        assert!((dx - px).abs() < 0.3, "rider moved with it: rider {dx}, platform {px}");
    }

    #[test]
    fn polygon_from_sprite_physics_outline_and_concave_split() {
        use zestengine_assets::sprite::{PixelRect, SpriteDef, SpriteSheet, build_geometry};
        // An L shape drawn as the sprite's physics shape: 4 x 4 px with a 2 x 2 notch.
        let mut def = SpriteDef::named("l", PixelRect::new(0, 0, 4, 4));
        def.physics_shape = vec![vec![[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [2.0, 2.0], [2.0, 4.0], [0.0, 4.0]]];
        let sheet = SpriteSheet { ppu: 2.0, ..Default::default() };
        let geometry = build_geometry(&def, &sheet, 4, 4, None);
        let poly = polygon_from_sprite(&geometry);
        // L area: 12 px^2 at 2 px/unit = 3 units^2.
        let built = build_shapes(&poly, Vec2::ONE).expect("builds");
        assert!(built.shapes.len() >= 2, "a concave outline is split into several convex pieces");
        let area: f32 = built.shapes.iter().map(|(_, s)| s.mass_properties(1.0).mass()).sum();
        assert!((area - 3.0).abs() < 1e-3, "pieces cover the outline exactly: {area}");
        assert!(built.shapes.iter().all(|(_, s)| s.as_convex_polygon().is_some()));
        // No physics shape: its bounding box.
        let plain = build_geometry(&SpriteDef::named("q", PixelRect::new(0, 0, 8, 4)), &SpriteSheet { generate_physics_shape: false, ppu: 4.0, ..Default::default() }, 8, 4, None);
        let q = build_shapes(&polygon_from_sprite(&plain), Vec2::ONE).unwrap();
        assert!((q.shapes[0].1.mass_properties(1.0).mass() - 2.0).abs() < 1e-3);
        // A ring (hole) keeps its hole.
        let ring = PolygonCollider2D {
            paths: vec![
                vec![[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]],
                vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
            ],
            ..Default::default()
        };
        let a: f32 = build_shapes(&ring, Vec2::ONE).unwrap().shapes.iter().map(|(_, s)| s.mass_properties(1.0).mass()).sum();
        assert!((a - 12.0).abs() < 1e-3, "16 - 4 = {a}");
    }

    #[test]
    fn components_save_and_load_and_old_scenes_are_unaffected() {
        let registry = ComponentRegistry::with_builtins();
        let samples: Vec<Box<dyn Component>> = vec![
            Box::new(Rigidbody2D { mass: 4.0, freeze_rotation: true, ..Default::default() }),
            Box::new(BoxCollider2D { size: [2.0, 3.0], is_trigger: true, ..Default::default() }),
            Box::new(CircleCollider2D { radius: 0.7, ..Default::default() }),
            Box::new(CapsuleCollider2D { direction: CapsuleDirection2D::Horizontal, ..Default::default() }),
            Box::new(PolygonCollider2D::default()),
            Box::new(EdgeCollider2D { closed: true, ..Default::default() }),
            Box::new(CompositeCollider2D { geometry: CompositeGeometry2D::Outlines, ..Default::default() }),
            Box::new(Joint2D { kind: Joint2DKind::Wheel, ..Default::default() }),
            Box::new(Effector2D { kind: EffectorKind2D::Surface, speed: -3.0, ..Default::default() }),
        ];
        for c in samples {
            let text = c.save();
            let back = registry.load(c.type_name(), &text).unwrap_or_else(|| panic!("{} reloads", c.type_name()));
            assert_eq!(back.type_name(), c.type_name());
            assert_eq!(back.save(), text, "{} round trips", c.type_name());
        }
        // A scene with no 2D components costs nothing: no world is built.
        let mut scene = LoadedScene::empty();
        obj(&mut scene, Vec3::ZERO, vec![Box::new(zestengine_core::BoxCollider::default())]);
        let mut p = Physics2D::new(None, &PhysicsSettings::default());
        assert!(!p.sync_structure(&scene));
        assert!(p.world().is_none());
        p.step();
        p.mark_dirty();
        assert!(!p.sync_structure(&scene) && p.world().is_none());
    }

    #[test]
    fn bodies_can_be_removed_and_spawned_while_running() {
        let mut scene = LoadedScene::empty();
        ground(&mut scene, 0.0);
        let first = obj(&mut scene, Vec3::new(0.0, 2.0, 0.0), vec![body(), rect(1.0, 1.0)]);
        let mut p = Physics2D::new(None, &PhysicsSettings::default());
        assert!(p.sync_structure(&scene));
        assert_eq!(p.world().unwrap().body_count(), 2);
        // Spawn.
        let second = obj(&mut scene, Vec3::new(3.0, 2.0, 0.0), vec![body(), rect(1.0, 1.0)]);
        p.mark_dirty();
        p.sync_structure(&scene);
        assert_eq!(p.world().unwrap().body_count(), 3);
        // Run, then destroy the first (its index shifts, ids do not).
        for _ in 0..30 {
            let ids = scene.id_index();
            let world = scene.world_transforms();
            p.push_poses(&scene, &ids, &world);
            p.step();
            p.write_back(&mut scene, &ids, 1.0);
        }
        scene.remove_object(scene.index_of(first).unwrap());
        p.mark_dirty();
        p.sync_structure(&scene);
        assert_eq!(p.world().unwrap().body_count(), 2);
        assert!(p.world().unwrap().velocity(first).is_none());
        for _ in 0..240 {
            let ids = scene.id_index();
            let world = scene.world_transforms();
            p.push_poses(&scene, &ids, &world);
            p.step();
            p.write_back(&mut scene, &ids, 1.0);
        }
        assert!((at(&scene, second).y - 0.5).abs() < 0.05, "the survivor keeps simulating");
        // Disable.
        let si = scene.index_of(second).unwrap();
        scene.objects[si].enabled = false;
        p.mark_dirty();
        p.sync_structure(&scene);
        assert_eq!(p.world().unwrap().body_count(), 1);
    }

    #[test]
    fn composite_merges_parts_and_compound_chunks_replace_incrementally() {
        let mut scene = LoadedScene::empty();
        let parent = obj(&mut scene, Vec3::ZERO, vec![Box::new(CompositeCollider2D::default())]);
        for x in [-1.0, 0.0, 1.0] {
            let c = obj(
                &mut scene,
                Vec3::new(x, 0.0, 0.0),
                vec![Box::new(BoxCollider2D { size: [1.0, 1.0], used_by_composite: true, ..Default::default() })],
            );
            let ci = scene.index_of(c).unwrap();
            scene.objects[ci].parent = Some(scene.index_of(parent).unwrap());
        }
        let mut w = make(&scene, &[], G);
        assert_eq!(w.body_count(), 1, "children belong to the parent's body");
        assert_eq!(w.collider_count(), 1, "three parts, one compound collider");
        w.step();
        let f = RayFilter::default();
        assert_eq!(w.raycast(Vec2::new(-5.0, 0.0), Vec2::X, 10.0, &f).unwrap().object, parent);
        assert!((w.raycast(Vec2::new(-5.0, 0.0), Vec2::X, 10.0, &f).unwrap().distance - 3.5).abs() < 1e-3);

        // Tilemap-style chunks on the same static body: replace one, leave the other.
        let chunk = |x: f32| compound_shape(vec![(Pose::from_translation(Vector::new(x, 5.0)), SharedShape::cuboid(0.5, 0.5)), (Pose::from_translation(Vector::new(x + 1.0, 5.0)), SharedShape::cuboid(0.5, 0.5))]);
        let props = CompoundProps::default();
        assert!(w.set_compound(parent, 1, chunk(10.0), &props, &scene));
        assert!(w.set_compound(parent, 2, chunk(20.0), &props, &scene));
        assert_eq!(w.collider_count(), 3);
        assert!(w.set_compound(parent, 1, chunk(30.0), &props, &scene), "rebuild chunk 1 only");
        assert_eq!(w.collider_count(), 3);
        assert!(w.set_compound(parent, 2, None, &props, &scene), "empty chunk removed");
        assert_eq!(w.collider_count(), 2);
        w.step(); // the broadphase picks the edits up
        assert!(w.raycast(Vec2::new(9.0, 5.0), Vec2::X, 3.0, &f).is_none(), "old chunk position is clear");
        assert!(w.raycast(Vec2::new(29.0, 5.0), Vec2::X, 3.0, &f).is_some(), "new chunk position is solid");
        assert!(!w.set_compound(ObjectId::new(), 1, None, &props, &scene));
    }

    #[test]
    fn outlines_composite_is_a_single_hollow_polyline() {
        let mut scene = LoadedScene::empty();
        obj(
            &mut scene,
            Vec3::ZERO,
            vec![
                Box::new(CompositeCollider2D { geometry: CompositeGeometry2D::Outlines, ..Default::default() }),
                Box::new(BoxCollider2D { size: [4.0, 4.0], used_by_composite: true, ..Default::default() }),
            ],
        );
        let w = make(&scene, &[], G);
        assert_eq!(w.collider_count(), 1);
        let mut lines = Vec::new();
        w.debug_lines(true, &mut lines);
        assert_eq!(lines.len(), 4, "the box boundary");
    }

    #[test]
    fn rigidbody_flags_apply() {
        let mut scene = LoadedScene::empty();
        let frozen = obj(
            &mut scene,
            Vec3::new(0.0, 5.0, 0.0),
            vec![Box::new(Rigidbody2D { freeze_x: true, freeze_rotation: true, gravity_scale: 0.0, ..Default::default() }), rect(1.0, 1.0)],
        );
        let heavy = obj(&mut scene, Vec3::new(5.0, 5.0, 0.0), vec![Box::new(Rigidbody2D { mass: 10.0, ..Default::default() }), rect(1.0, 1.0)]);
        let mut w = make(&scene, &[], G);
        w.add_impulse(frozen, Vec2::new(5.0, 0.0));
        w.add_impulse(heavy, Vec2::new(10.0, 0.0));
        let (mass, _) = w.mass_and_area(heavy).unwrap();
        assert!((mass - 10.0).abs() < 1e-3, "explicit mass: {mass}");
        run(&mut w, &mut scene, 30);
        assert_eq!(at(&scene, frozen).x, 0.0, "x is frozen");
        assert!((at(&scene, frozen).y - 5.0).abs() < 1e-3, "no gravity");
        assert!((w.velocity(heavy).unwrap().x - 1.0).abs() < 0.05, "impulse / mass");
    }

    #[test]
    fn one_thousand_boxes_step_quickly() {
        let mut scene = LoadedScene::empty();
        ground(&mut scene, 0.0);
        for k in 0..1000 {
            let (i, j) = (k % 40, k / 40);
            obj(&mut scene, Vec3::new(i as f32 * 1.1 - 22.0, 0.5 + j as f32 * 1.05, 0.0), vec![body(), rect(1.0, 1.0)]);
        }
        let mut w = make(&scene, &[], G);
        for _ in 0..20 {
            w.step();
        }
        let t = std::time::Instant::now();
        let n = 60;
        for _ in 0..n {
            w.step();
        }
        let ms = t.elapsed().as_secs_f32() * 1000.0 / n as f32;
        eprintln!("1000 boxes: {ms:.2} ms per step");
        assert!(ms < 12.0, "{ms} ms per step");
    }

    // ---- through the play loop: callbacks and zero cost ----

    use crate::play::{PlaySession, TickInput, World};
    use crate::shaders::ShaderLibrary;
    use serde::{Deserialize, Serialize};
    use zestengine_core::{ComponentCtx, InputState, NetView, TypedComponent};

    #[derive(Serialize, Deserialize, Default)]
    struct Hits {
        enter_2d: u32,
        stay_2d: u32,
        exit_2d: u32,
        trigger_enter: u32,
        plain_enter: u32,
        saw_raycast: bool,
    }
    impl TypedComponent for Hits {
        const NAME: &'static str = "Test Hits 2D";
        fn on_collision_enter(&mut self, _ctx: &mut ComponentCtx, _other: ObjectId) {
            self.plain_enter += 1;
        }
        fn on_collision_enter_2d(&mut self, _ctx: &mut ComponentCtx, _other: ObjectId) {
            self.enter_2d += 1;
        }
        fn on_collision_stay_2d(&mut self, _ctx: &mut ComponentCtx, _other: ObjectId) {
            self.stay_2d += 1;
        }
        fn on_collision_exit_2d(&mut self, _ctx: &mut ComponentCtx, _other: ObjectId) {
            self.exit_2d += 1;
        }
        fn on_trigger_enter(&mut self, _ctx: &mut ComponentCtx, _other: ObjectId) {
            self.trigger_enter += 1;
        }
        fn update(&mut self, ctx: &mut ComponentCtx) {
            if ctx.raycast2d(Vec2::new(0.0, 10.0), Vec2::NEG_Y, 20.0).is_some() {
                self.saw_raycast = true;
            }
        }
    }

    fn play(scene: &mut LoadedScene, session: &mut PlaySession, frames: usize) {
        let mut registry = ComponentRegistry::with_builtins();
        registry.register::<Hits>();
        let mut shaders = ShaderLibrary::default();
        let input = InputState::default();
        let net = NetView::default();
        let screen = zestengine_core::ScreenInfo::default();
        for _ in 0..frames {
            session.tick(
                &mut World { scene, renderer: None, registry: &registry, shaders: &mut shaders },
                &TickInput { dt: 1.0 / 60.0, input: &input, net: &net, listener_fallback: Mat4::IDENTITY, screen: &screen },
            );
        }
    }

    #[test]
    fn play_loop_delivers_2d_callbacks_and_queries() {
        let mut scene = LoadedScene::empty();
        ground(&mut scene, 0.0);
        let ball = obj(&mut scene, Vec3::new(0.0, 2.0, 0.0), vec![body(), Box::new(CircleCollider2D::default()), Box::new(Hits::default())]);
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        play(&mut scene, &mut session, 90);
        let i = scene.index_of(ball).unwrap();
        let h = scene.objects[i].components.iter().find_map(|c| c.as_any().downcast_ref::<Hits>()).unwrap();
        assert!(h.enter_2d >= 1, "2D enter: {}", h.enter_2d);
        assert_eq!(h.plain_enter, 0, "the 2D override replaces the forwarding default");
        assert!(h.saw_raycast, "ctx.raycast2d reaches the 2D world");
        let y = scene.objects[i].translation.y;
        assert!((y - 0.5).abs() < 0.1, "the ball lies on the ground: y = {y}");
    }

    #[test]
    fn plain_callbacks_still_fire_for_2d_contacts() {
        // A component written for 3D (only `on_trigger_enter`) reacts to 2D triggers through the default forwarding.
        let mut scene = LoadedScene::empty();
        obj(&mut scene, Vec3::new(0.0, 0.0, 0.0), vec![Box::new(BoxCollider2D { size: [6.0, 2.0], is_trigger: true, ..Default::default() }), Box::new(Hits::default())]);
        obj(&mut scene, Vec3::new(0.0, 3.0, 0.0), vec![body(), Box::new(CircleCollider2D::default())]);
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        play(&mut scene, &mut session, 90);
        let h = scene.objects[0].components.iter().find_map(|c| c.as_any().downcast_ref::<Hits>()).unwrap();
        assert_eq!(h.trigger_enter, 1);
    }

    #[test]
    fn a_scene_without_2d_never_builds_a_2d_world() {
        let mut scene = LoadedScene::empty();
        obj(&mut scene, Vec3::ZERO, vec![Box::new(zestengine_core::RigidBody::default()), Box::new(zestengine_core::BoxCollider::default())]);
        let mut session = PlaySession::new(&scene, Path::new("."), None);
        play(&mut scene, &mut session, 10);
        assert!(session.physics2d.world().is_none());
    }

    // ---------------------------------------------------------- tilemaps

    mod tilemaps {
        use super::*;
        use glam::IVec2;
        use zestengine_core::{TileAsset, TileColliderType, TileFile, Tilemap, TilemapCollider2D};

        fn grid_tile() -> TileFile {
            TileFile::Tile(TileAsset { sprite: "s.png#0000000000000001".into(), collider: TileColliderType::Grid, ..Default::default() })
        }

        fn map_obj(scene: &mut LoadedScene, tc: TilemapCollider2D, cells: &[(i32, i32)]) -> ObjectId {
            let mut m = Tilemap::default();
            for &(x, y) in cells {
                m.set_tile(IVec2::new(x, y), Some("g.tile"));
            }
            obj(scene, Vec3::ZERO, vec![Box::new(m), Box::new(tc)])
        }

        fn with_map<R>(scene: &mut LoadedScene, id: ObjectId, f: impl FnOnce(&mut Tilemap) -> R) -> R {
            let i = scene.index_of(id).unwrap();
            let m = scene.objects[i].components.iter_mut().find_map(|c| c.as_any_mut().downcast_mut::<Tilemap>()).unwrap();
            f(m)
        }

        fn world(scene: &LoadedScene) -> PhysicsWorld2D {
            let mut w = make(scene, &[], G);
            w.tile_db_mut().insert("g.tile", grid_tile());
            w.sync_tilemaps(scene);
            w
        }

        #[test]
        fn ball_rolls_over_a_tile_floor_and_stops_at_a_wall() {
            let mut scene = LoadedScene::empty();
            let mut cells: Vec<(i32, i32)> = (0..40).map(|x| (x, 0)).collect();
            cells.extend((1..5).map(|y| (30, y)));
            map_obj(&mut scene, TilemapCollider2D::default(), &cells);
            let ball = obj(&mut scene, Vec3::new(2.0, 1.6, 0.0), vec![body(), Box::new(CircleCollider2D { radius: 0.5, ..Default::default() })]);
            let mut w = world(&scene);
            assert_eq!(w.collider_count(), 4, "floor+wall in chunks 0..2: three chunk compounds, plus the ball");
            w.set_velocity(ball, Vec2::new(14.0, 0.0));
            for _ in 0..360 {
                w.sync_tilemaps(&scene);
                run(&mut w, &mut scene, 1);
            }
            let p = at(&scene, ball);
            assert!(p.y > 1.4 && p.y < 1.6, "rests on the floor (top y = 1, radius 0.5): {p}");
            assert!(p.x > 27.0 && p.x < 29.6, "stopped by the wall at x = 30: {p}");
        }

        #[test]
        fn editing_a_cell_rebuilds_only_its_chunk() {
            let mut scene = LoadedScene::empty();
            let cells: Vec<(i32, i32)> = (0..48).flat_map(|x| (0..2).map(move |y| (x, y))).collect();
            let id = map_obj(&mut scene, TilemapCollider2D::default(), &cells);
            let mut w = world(&scene);
            assert_eq!(w.tile_stats().chunks_rebuilt, 3);
            let colliders = w.collider_count();
            // Nothing changed: nothing rebuilt.
            w.sync_tilemaps(&scene);
            assert_eq!(w.tile_stats(), Default::default());
            // An interior edit in the middle chunk.
            with_map(&mut scene, id, |m| m.set_tile(IVec2::new(20, 1), None));
            w.sync_tilemaps(&scene);
            assert_eq!(w.tile_stats().chunks_rebuilt, 1);
            assert_eq!(w.collider_count(), colliders);
            w.step();
            let f = RayFilter::default();
            let d = w.raycast(Vec2::new(20.5, 5.0), -Vec2::Y, 10.0, &f).map(|h| h.distance).unwrap_or(0.0);
            assert!((d - 4.0).abs() < 1e-3, "the hole is open down to the floor row (top at y = 1): {d}");
            // A border edit also refreshes the neighbour chunk.
            with_map(&mut scene, id, |m| m.set_tile(IVec2::new(16, 1), None));
            w.sync_tilemaps(&scene);
            assert_eq!(w.tile_stats().chunks_rebuilt, 2);
            // Emptying a chunk removes its compound.
            with_map(&mut scene, id, |m| {
                for x in 32..48 {
                    for y in 0..2 {
                        m.set_tile(IVec2::new(x, y), None);
                    }
                }
            });
            w.sync_tilemaps(&scene);
            assert_eq!(w.tile_stats().chunks_removed, 1);
            assert_eq!(w.collider_count(), colliders - 1);
        }

        #[test]
        fn merged_cells_become_few_cuboids_and_outlines_become_one_polyline() {
            let mut scene = LoadedScene::empty();
            let cells: Vec<(i32, i32)> = (0..16).flat_map(|x| (0..16).map(move |y| (x, y))).collect();
            let a = map_obj(&mut scene, TilemapCollider2D::default(), &cells);
            let _ = a;
            let w = world(&scene);
            let mut lines = Vec::new();
            w.debug_lines(true, &mut lines);
            assert_eq!(lines.len(), 4, "a full chunk is one cuboid");
            let mut scene = LoadedScene::empty();
            map_obj(&mut scene, TilemapCollider2D { outlines: true, ..Default::default() }, &cells);
            let w = world(&scene);
            let mut lines = Vec::new();
            w.debug_lines(true, &mut lines);
            assert_eq!(lines.len(), 4, "its outline is four segments");
            assert_eq!(w.collider_count(), 1);
        }

        #[test]
        fn sprite_outline_tiles_produce_the_sprite_polygon() {
            let mut scene = LoadedScene::empty();
            let mut m = Tilemap::default();
            m.set_tile(IVec2::new(2, 3), Some("tri.tile"));
            obj(&mut scene, Vec3::ZERO, vec![Box::new(m), Box::new(TilemapCollider2D::default())]);
            let mut w = make(&scene, &[], G);
            w.tile_db_mut().insert("tri.tile", TileFile::Tile(TileAsset { sprite: "tri.png#0000000000000001".into(), collider: TileColliderType::Sprite, ..Default::default() }));
            // A right triangle, pivot-relative: its legs on the bottom-left of the pivot.
            w.set_sprite_physics("tri.png#0000000000000001", &[vec![[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]]]);
            w.sync_tilemaps(&scene);
            w.step();
            let f = RayFilter::default();
            // The tile pivot is the cell centre (2.5, 3.5).
            let inside = |x: f32, y: f32| !w.overlap_point(Vec2::new(x, y), &f).is_empty();
            assert!(inside(2.2, 3.3), "inside the triangle");
            assert!(!inside(2.9, 3.9), "the empty corner of the cell is open");
            assert!(!inside(5.0, 3.5));
            let top = w.raycast(Vec2::new(2.2, 6.0), -Vec2::Y, 6.0, &f).unwrap();
            assert!((top.point.y - 3.8).abs() < 1e-3, "the hypotenuse at x = 2.2: {}", top.point.y);
        }

        #[test]
        fn removing_the_component_clears_the_compounds() {
            let mut scene = LoadedScene::empty();
            let id = map_obj(&mut scene, TilemapCollider2D::default(), &[(0, 0), (1, 0), (40, 0)]);
            let mut w = world(&scene);
            assert_eq!(w.collider_count(), 2);
            let i = scene.index_of(id).unwrap();
            scene.objects[i].components.retain(|c| c.type_name() != "Tilemap Collider 2D");
            w.mark_dirty();
            w.sync_structure(&scene);
            w.sync_tilemaps(&scene);
            assert_eq!(w.collider_count(), 0);
        }

        #[test]
        fn composite_flag_merges_all_chunks_into_one_compound() {
            let mut scene = LoadedScene::empty();
            let cells: Vec<(i32, i32)> = (0..48).map(|x| (x, 0)).collect();
            let mut m = Tilemap::default();
            for &(x, y) in &cells {
                m.set_tile(IVec2::new(x, y), Some("g.tile"));
            }
            let id = obj(
                &mut scene,
                Vec3::ZERO,
                vec![Box::new(m), Box::new(TilemapCollider2D { used_by_composite: true, ..Default::default() }), Box::new(CompositeCollider2D::default())],
            );
            let mut w = world(&scene);
            assert_eq!(w.collider_count(), 1, "three chunks, one composite compound");
            with_map(&mut scene, id, |m| m.set_tile(IVec2::new(40, 0), None));
            w.sync_tilemaps(&scene);
            assert_eq!(w.tile_stats().chunks_rebuilt, 1);
            assert_eq!(w.collider_count(), 1);
        }

        #[test]
        fn big_map_build_and_edit_timings() {
            let mut scene = LoadedScene::empty();
            let mut m = Tilemap::default();
            for x in 0..256 {
                for y in 0..256 {
                    m.set_tile(IVec2::new(x, y), Some("g.tile"));
                }
            }
            let id = obj(&mut scene, Vec3::ZERO, vec![Box::new(m), Box::new(TilemapCollider2D::default())]);
            let mut w = make(&scene, &[], G);
            w.tile_db_mut().insert("g.tile", grid_tile());
            let t = std::time::Instant::now();
            w.sync_tilemaps(&scene);
            let full = t.elapsed();
            assert_eq!(w.tile_stats().chunks_rebuilt, 256);
            assert_eq!(w.collider_count(), 256);
            with_map(&mut scene, id, |m| m.set_tile(IVec2::new(100, 100), None));
            let t = std::time::Instant::now();
            w.sync_tilemaps(&scene);
            let edit = t.elapsed();
            assert_eq!(w.tile_stats().chunks_rebuilt, 1);
            eprintln!("TILE PERF 256x256 solid: full build {full:?}, single-cell edit {edit:?}");
            assert!(edit < std::time::Duration::from_millis(50));
        }
    }
}
