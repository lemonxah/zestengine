//! 2D physics components (2D-6): the authoring data for `physics2d.rs` in the
//! engine (rapier2d). Everything here is plain serde data, egui-free.
//!
//! The 2D world lives in the XY plane of the scene, rotating about Z only.
//! It shares the 3D world's conventions on purpose:
//!
//! - **Layers**: an object's `layer` and the project's layer collision matrix
//!   (`LayerSettings`) decide what collides, same as 3D. There is no separate
//!   2D matrix, so the Layers editor and camera masks stay the one place to look.
//! - **Materials**: `.physmat` assets (friction, bounciness, combine rules,
//!   density for auto-mass). Buoyancy / water fields are ignored in 2D.
//! - **Settings**: the project Physics settings (gravity x / y, contact
//!   stiffness and damping, solver iterations, default material).
//! - **Callbacks**: the existing `on_collision_*` / `on_trigger_*` methods
//!   are called for 2D contacts too; the `_2d` variants let a component tell
//!   them apart (their default forwards to the plain callback).
//!
//! A collider on an object without a [`Rigidbody2D`] attaches to the nearest
//! ancestor's body (a compound body), else it is static. [`CompositeCollider2D`]
//! merges the colliders flagged `used_by_composite` below it into one collider.

use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::{ActionCtx, ActionDesc, ArgValue, ComponentCommand, ObjectId, ObjectRef, ParamDesc, TypedComponent, actions};

/// How a [`Rigidbody2D`] moves.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum BodyType2D {
    /// Simulated: forces, gravity, collisions.
    #[default]
    Dynamic,
    /// Follows its transform (animation, scripts) and pushes dynamic bodies.
    Kinematic,
    /// Never moves (walls, ground).
    Static,
}

impl BodyType2D {
    /// Every body type, in declaration order.
    pub const ALL: [BodyType2D; 3] = [Self::Dynamic, Self::Kinematic, Self::Static];
    /// Inspector label of this body type.
    pub fn label(self) -> &'static str {
        match self {
            Self::Dynamic => "Dynamic",
            Self::Kinematic => "Kinematic",
            Self::Static => "Static",
        }
    }
}

/// When a [`Rigidbody2D`] sleeps (stops simulating while at rest).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum SleepMode2D {
    /// Awake at spawn, sleeps when it settles.
    #[default]
    StartAwake,
    /// Asleep at spawn until something disturbs it.
    StartAsleep,
    /// Never sleeps.
    NeverSleep,
}

impl SleepMode2D {
    /// Every mode, in declaration order.
    pub const ALL: [SleepMode2D; 3] = [Self::StartAwake, Self::StartAsleep, Self::NeverSleep];
    /// Inspector label of this mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::StartAwake => "Start Awake",
            Self::StartAsleep => "Start Asleep",
            Self::NeverSleep => "Never Sleep",
        }
    }
}

/// What the rendered pose does between two fixed steps.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum Interpolation2D {
    /// The pose of the latest step (steppy at low physics rates).
    None,
    /// Blend the last two steps (smooth, a step behind).
    #[default]
    Interpolate,
    /// Project the latest step forward by its velocity.
    Extrapolate,
}

impl Interpolation2D {
    /// Every mode, in declaration order.
    pub const ALL: [Interpolation2D; 3] = [Self::None, Self::Interpolate, Self::Extrapolate];
    /// Inspector label of this mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Interpolate => "Interpolate",
            Self::Extrapolate => "Extrapolate",
        }
    }
}

/// A 2D rigid body. Needs at least one 2D collider (on this object or below)
/// to have a shape; without one the body has no mass and no contacts.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct Rigidbody2D {
    /// How the body moves: simulated, driven by its transform, or fixed.
    pub body_type: BodyType2D,
    /// Take part in the simulation. Off removes the body and its colliders from
    /// the world without disabling the object.
    pub simulated: bool,
    /// Mass in kg (Dynamic only; 0 = from the colliders' area and density).
    pub mass: f32,
    /// Gravity multiplier on this body (0 = none, negative = inverted).
    pub gravity_scale: f32,
    /// Linear drag (1/s).
    pub linear_damping: f32,
    /// Angular drag (1/s).
    pub angular_damping: f32,
    /// Lock translation along X.
    pub freeze_x: bool,
    /// Lock translation along Y.
    pub freeze_y: bool,
    /// Lock rotation about Z.
    pub freeze_rotation: bool,
    /// Continuous collision detection: fast bodies do not tunnel through thin
    /// colliders.
    pub continuous: bool,
    /// When the body sleeps.
    pub sleep_mode: SleepMode2D,
    /// What the rendered pose does between two fixed steps.
    pub interpolation: Interpolation2D,
    /// Physics material (`.physmat`, project-relative); colliders inherit it.
    pub material: String,
    /// Draw this body's colliders in the editor's Play view.
    pub debug_draw: bool,
    /// Engine-filled in Play: the simulated mass (kg) and collider area (m^2).
    #[serde(skip)]
    pub measured: Option<(f32, f32)>,
}

impl Default for Rigidbody2D {
    fn default() -> Self {
        Self {
            body_type: BodyType2D::Dynamic,
            simulated: true,
            mass: 0.0,
            gravity_scale: 1.0,
            linear_damping: 0.0,
            angular_damping: 0.05,
            freeze_x: false,
            freeze_y: false,
            freeze_rotation: false,
            continuous: false,
            sleep_mode: SleepMode2D::StartAwake,
            interpolation: Interpolation2D::Interpolate,
            material: String::new(),
            debug_draw: false,
            measured: None,
        }
    }
}

static RIGIDBODY2D_ACTIONS: &[ActionDesc] = &[
    ActionDesc::new("AddForce2D", "Add Force 2D", &[ParamDesc::float("x", 0.0), ParamDesc::float("y", 0.0)], true),
    ActionDesc::new("AddImpulse2D", "Add Impulse 2D", &[ParamDesc::float("x", 0.0), ParamDesc::float("y", 5.0)], true),
    ActionDesc::new("AddTorque2D", "Add Torque 2D", &[ParamDesc::float("torque", 1.0)], true),
    ActionDesc::new("SetVelocity2D", "Set Velocity 2D", &[ParamDesc::float("x", 0.0), ParamDesc::float("y", 0.0)], true),
];

impl TypedComponent for Rigidbody2D {
    const NAME: &'static str = "Rigidbody 2D";

    fn actions(&self) -> &'static [ActionDesc] {
        RIGIDBODY2D_ACTIONS
    }
    fn invoke_action(&mut self, ctx: &mut ActionCtx, name: &str, args: &[ArgValue]) -> Result<(), String> {
        let id = ctx.self_id;
        let v = |a: &[ArgValue]| glam::Vec3::new(a[0].as_f32(), a[1].as_f32(), 0.0);
        match name {
            "AddForce2D" => ctx.commands.push(ComponentCommand::AddForce { id, force: v(args) }),
            "AddImpulse2D" => ctx.commands.push(ComponentCommand::AddImpulse { id, impulse: v(args) }),
            "AddTorque2D" => ctx.commands.push(ComponentCommand::AddTorque { id, torque: glam::Vec3::new(0.0, 0.0, args[0].as_f32()) }),
            "SetVelocity2D" => ctx.commands.push(ComponentCommand::SetVelocity { id, linear: v(args) }),
            _ => return Err(actions::unknown_action(Self::NAME, name)),
        }
        Ok(())
    }
}

/// What every 2D collider component shares, borrowed for the engine.
pub struct ColliderProps<'a> {
    /// Offset from the object origin, object-local units (scaled by the object).
    pub offset: Vec2,
    /// Detect overlaps without a collision response.
    pub is_trigger: bool,
    /// An [`Effector2D`] on this object acts on this collider.
    pub used_by_effector: bool,
    /// Merged into the nearest [`CompositeCollider2D`] above (or on) this object.
    pub used_by_composite: bool,
    /// Mass per area for auto-mass bodies (the material's density wins when set).
    pub density: f32,
    /// `.physmat` override ("" = the body's, else the project default).
    pub material: &'a str,
}

/// Implemented by every 2D collider component so the engine reads their
/// shared settings uniformly.
pub trait Collider2D {
    /// The shared collider settings, borrowed for the engine.
    fn props(&self) -> ColliderProps<'_>;
}

macro_rules! collider2d_common {
    ($t:ty) => {
        impl Collider2D for $t {
            fn props(&self) -> ColliderProps<'_> {
                ColliderProps {
                    offset: Vec2::from(self.offset),
                    is_trigger: self.is_trigger,
                    used_by_effector: self.used_by_effector,
                    used_by_composite: self.used_by_composite,
                    density: self.density,
                    material: &self.material,
                }
            }
        }
    };
}

/// Axis-aligned (object space) rectangle.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct BoxCollider2D {
    /// Offset from the object origin, object-local units (scaled by the object).
    pub offset: [f32; 2],
    /// Half-extents (width, height) of the box.
    pub size: [f32; 2],
    /// Detect overlaps without a collision response.
    pub is_trigger: bool,
    /// An [`Effector2D`] on this object acts on this collider.
    pub used_by_effector: bool,
    /// Merged into the nearest [`CompositeCollider2D`] above (or on) this object.
    pub used_by_composite: bool,
    /// Mass per area for auto-mass bodies (kg/m^2).
    pub density: f32,
    /// `.physmat` override (empty = the body's, else the project default).
    pub material: String,
}

impl Default for BoxCollider2D {
    fn default() -> Self {
        Self { offset: [0.0; 2], size: [1.0; 2], is_trigger: false, used_by_effector: false, used_by_composite: false, density: 1.0, material: String::new() }
    }
}
collider2d_common!(BoxCollider2D);
impl TypedComponent for BoxCollider2D {
    const NAME: &'static str = "Box Collider 2D";
}

/// A circle.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct CircleCollider2D {
    /// Offset from the object origin, object-local units (scaled by the object).
    pub offset: [f32; 2],
    /// Circle radius.
    pub radius: f32,
    /// Detect overlaps without a collision response.
    pub is_trigger: bool,
    /// An [`Effector2D`] on this object acts on this collider.
    pub used_by_effector: bool,
    /// Merged into the nearest [`CompositeCollider2D`] above (or on) this object.
    pub used_by_composite: bool,
    /// Mass per area for auto-mass bodies (kg/m^2).
    pub density: f32,
    /// `.physmat` override (empty = the body's, else the project default).
    pub material: String,
}

impl Default for CircleCollider2D {
    fn default() -> Self {
        Self { offset: [0.0; 2], radius: 0.5, is_trigger: false, used_by_effector: false, used_by_composite: false, density: 1.0, material: String::new() }
    }
}
collider2d_common!(CircleCollider2D);
impl TypedComponent for CircleCollider2D {
    const NAME: &'static str = "Circle Collider 2D";
}

/// Which way the capsule's caps face.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum CapsuleDirection2D {
    /// Caps along Y (a standing capsule).
    #[default]
    Vertical,
    /// Caps along X (a lying capsule).
    Horizontal,
}

impl CapsuleDirection2D {
    /// Every direction, in declaration order.
    pub const ALL: [CapsuleDirection2D; 2] = [Self::Vertical, Self::Horizontal];
    /// Inspector label of this direction.
    pub fn label(self) -> &'static str {
        match self {
            Self::Vertical => "Vertical",
            Self::Horizontal => "Horizontal",
        }
    }
}

/// A capsule `size` = (width, height) of its bounding box; the caps are
/// half-circles of the shorter side.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct CapsuleCollider2D {
    /// Offset from the object origin, object-local units (scaled by the object).
    pub offset: [f32; 2],
    /// Bounding box (width, height) the capsule fits in.
    pub size: [f32; 2],
    /// Which way the caps face.
    pub direction: CapsuleDirection2D,
    /// Detect overlaps without a collision response.
    pub is_trigger: bool,
    /// An [`Effector2D`] on this object acts on this collider.
    pub used_by_effector: bool,
    /// Merged into the nearest [`CompositeCollider2D`] above (or on) this object.
    pub used_by_composite: bool,
    /// Mass per area for auto-mass bodies (kg/m^2).
    pub density: f32,
    /// `.physmat` override (empty = the body's, else the project default).
    pub material: String,
}

impl Default for CapsuleCollider2D {
    fn default() -> Self {
        Self {
            offset: [0.0; 2],
            size: [0.5, 1.0],
            direction: CapsuleDirection2D::Vertical,
            is_trigger: false,
            used_by_effector: false,
            used_by_composite: false,
            density: 1.0,
            material: String::new(),
        }
    }
}
collider2d_common!(CapsuleCollider2D);
impl TypedComponent for CapsuleCollider2D {
    const NAME: &'static str = "Capsule Collider 2D";
}

/// A filled polygon. `paths` are rings in object-local units (y up); a concave
/// ring is split into convex pieces on build (ear clipping + convex merge, the
/// sprite physics-outline code), and rings nested in another become holes.
/// Fill it from a sprite's physics shape with `physics2d::polygon_from_sprite`.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct PolygonCollider2D {
    /// Offset from the object origin, object-local units (scaled by the object).
    pub offset: [f32; 2],
    /// Rings in object-local units (y up); nested rings become holes.
    pub paths: Vec<Vec<[f32; 2]>>,
    /// Detect overlaps without a collision response.
    pub is_trigger: bool,
    /// An [`Effector2D`] on this object acts on this collider.
    pub used_by_effector: bool,
    /// Merged into the nearest [`CompositeCollider2D`] above (or on) this object.
    pub used_by_composite: bool,
    /// Mass per area for auto-mass bodies (kg/m^2).
    pub density: f32,
    /// `.physmat` override (empty = the body's, else the project default).
    pub material: String,
}

impl Default for PolygonCollider2D {
    fn default() -> Self {
        Self {
            offset: [0.0; 2],
            paths: vec![vec![[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]],
            is_trigger: false,
            used_by_effector: false,
            used_by_composite: false,
            density: 1.0,
            material: String::new(),
        }
    }
}
collider2d_common!(PolygonCollider2D);
impl TypedComponent for PolygonCollider2D {
    const NAME: &'static str = "Polygon Collider 2D";
}

/// A chain of segments (terrain lines, thin walls). Has no area: on a dynamic
/// body it adds no mass, and it only collides with shapes that have some.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct EdgeCollider2D {
    /// Offset from the object origin, object-local units (scaled by the object).
    pub offset: [f32; 2],
    /// The chain's vertices in object-local units.
    pub points: Vec<[f32; 2]>,
    /// Join the last point back to the first.
    /// Join the last point back to the first.
    pub closed: bool,
    /// Detect overlaps without a collision response.
    pub is_trigger: bool,
    /// An [`Effector2D`] on this object acts on this collider.
    pub used_by_effector: bool,
    /// Merged into the nearest [`CompositeCollider2D`] above (or on) this object.
    pub used_by_composite: bool,
    /// Mass per area for auto-mass bodies (kg/m^2).
    pub density: f32,
    /// `.physmat` override (empty = the body's, else the project default).
    pub material: String,
}

impl Default for EdgeCollider2D {
    fn default() -> Self {
        Self {
            offset: [0.0; 2],
            points: vec![[-1.0, 0.0], [1.0, 0.0]],
            closed: false,
            is_trigger: false,
            used_by_effector: false,
            used_by_composite: false,
            density: 1.0,
            material: String::new(),
        }
    }
}
collider2d_common!(EdgeCollider2D);
impl TypedComponent for EdgeCollider2D {
    const NAME: &'static str = "Edge Collider 2D";
}

/// What a composite builds from its parts.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum CompositeGeometry2D {
    /// One compound collider of every part's solid shape (one broadphase
    /// entry, no internal-edge removal).
    #[default]
    Polygons,
    /// One compound collider of every part's boundary as segments: hollow, for
    /// static terrain (cheap narrow phase; things can fall inside).
    Outlines,
}

impl CompositeGeometry2D {
    /// Every geometry, in declaration order.
    pub const ALL: [CompositeGeometry2D; 2] = [Self::Polygons, Self::Outlines];
    /// Inspector label of this geometry.
    pub fn label(self) -> &'static str {
        match self {
            Self::Polygons => "Polygons",
            Self::Outlines => "Outlines",
        }
    }
}

/// Merges the colliders flagged `used_by_composite` (on this object and the
/// objects below it, up to the next body) into one collider on the body. The
/// parts stop being separate colliders (and contact / trigger events then come
/// from the composite's object). Tilemaps feed their per-chunk shapes through
/// the same compound API (`PhysicsWorld2D::set_compound`).
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct CompositeCollider2D {
    /// What the composite builds from its parts.
    pub geometry: CompositeGeometry2D,
    /// Offset from the object origin, object-local units (scaled by the object).
    pub offset: [f32; 2],
    /// Detect overlaps without a collision response.
    pub is_trigger: bool,
    /// An [`Effector2D`] on this object acts on the composite.
    pub used_by_effector: bool,
    /// `.physmat` override (empty = the body's, else the project default).
    pub material: String,
}

impl Default for CompositeCollider2D {
    fn default() -> Self {
        Self { geometry: CompositeGeometry2D::Polygons, offset: [0.0; 2], is_trigger: false, used_by_effector: false, material: String::new() }
    }
}
impl TypedComponent for CompositeCollider2D {
    const NAME: &'static str = "Composite Collider 2D";
}

/// What kind of joint constrains the two bodies.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum Joint2DKind {
    /// Welds the two bodies together.
    Fixed,
    /// Keeps a distance (rigid), or a maximum one (a rope).
    Distance,
    /// A pin: rotation about the anchor (doors, wheels, pendulums).
    #[default]
    Hinge,
    /// A soft distance (spring and damper).
    Spring,
    /// Translation along an axis (pistons, sliding doors).
    Slider,
    /// A wheel on a suspension: slides along the axis on a spring and spins freely.
    Wheel,
}

impl Joint2DKind {
    /// Every kind, in declaration order.
    pub const ALL: [Joint2DKind; 6] = [Self::Fixed, Self::Distance, Self::Hinge, Self::Spring, Self::Slider, Self::Wheel];
    /// Inspector label of this kind.
    pub fn label(self) -> &'static str {
        match self {
            Self::Fixed => "Fixed",
            Self::Distance => "Distance",
            Self::Hinge => "Hinge",
            Self::Spring => "Spring",
            Self::Slider => "Slider",
            Self::Wheel => "Wheel",
        }
    }
}

/// A joint between this object's body and `connected` (unset = the world,
/// anchored where the anchor starts). Not supported by rapier2d: Unity's
/// Friction, Relative and Target joints (use a Hinge / Fixed + motor, or a
/// script), break force / break action.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct Joint2D {
    /// What kind of joint constrains the bodies.
    pub kind: Joint2DKind,
    /// The other body (unset = the world).
    pub connected: ObjectRef,
    /// Anchor in this object's local space.
    pub anchor: [f32; 2],
    /// Slider / wheel axis, degrees from +X in this object's local space.
    pub axis_angle: f32,
    /// Hinge: degrees; slider: meters along the axis.
    /// Hinge: degrees; slider: meters along the axis.
    pub limits_enabled: bool,
    /// Lower / upper limit.
    pub limits: [f32; 2],
    /// Hinge / slider / wheel motor (deg/s or m/s).
    pub motor_enabled: bool,
    /// Target motor speed (deg/s or m/s).
    pub motor_speed: f32,
    /// Maximum force the motor may apply.
    pub motor_max_force: f32,
    /// Distance / spring length (0 = the distance at start).
    pub distance: f32,
    /// Distance joint: only stop the bodies drifting further apart (a rope).
    pub max_distance_only: bool,
    /// Spring / wheel suspension stiffness (N/m) and damping (N s/m).
    pub stiffness: f32,
    /// Spring / wheel suspension damping (N s/m).
    pub damping: f32,
    /// Let the two connected bodies collide with each other.
    pub collide_connected: bool,
}

impl Default for Joint2D {
    fn default() -> Self {
        Self {
            kind: Joint2DKind::Hinge,
            connected: ObjectRef::NONE,
            anchor: [0.0; 2],
            axis_angle: 90.0,
            limits_enabled: false,
            limits: [-45.0, 45.0],
            motor_enabled: false,
            motor_speed: 90.0,
            motor_max_force: 100.0,
            distance: 0.0,
            max_distance_only: false,
            stiffness: 100.0,
            damping: 5.0,
            collide_connected: false,
        }
    }
}
impl TypedComponent for Joint2D {
    const NAME: &'static str = "Joint 2D";
}

/// What kind of force an [`Effector2D`] applies.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum EffectorKind2D {
    /// A constant force (wind, a current) on what is inside the trigger.
    #[default]
    Area,
    /// Water: lift by the submerged share below a surface level, with drag and flow.
    Buoyancy,
    /// Pulls towards (or pushes from) the effector's centre.
    Point,
    /// A one-way platform: contacts only hold from the top arc.
    Platform,
    /// A conveyor: the touching surface moves along its edge.
    Surface,
}

impl EffectorKind2D {
    /// Every kind, in declaration order.
    pub const ALL: [EffectorKind2D; 5] = [Self::Area, Self::Buoyancy, Self::Point, Self::Platform, Self::Surface];
    /// Inspector label of this kind.
    pub fn label(self) -> &'static str {
        match self {
            Self::Area => "Area",
            Self::Buoyancy => "Buoyancy",
            Self::Point => "Point",
            Self::Platform => "Platform",
            Self::Surface => "Surface",
        }
    }
    /// Whether it acts on what overlaps a trigger collider (the others work on
    /// solid contacts).
    pub fn uses_trigger(self) -> bool {
        matches!(self, Self::Area | Self::Buoyancy | Self::Point)
    }
}

/// How a point force falls off with distance.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum PointForceMode2D {
    /// The same force at any distance.
    Constant,
    /// Force falls off as 1 / distance.
    #[default]
    InverseLinear,
    /// Force falls off as 1 / distance^2.
    InverseSquared,
}

impl PointForceMode2D {
    /// Every mode, in declaration order.
    pub const ALL: [PointForceMode2D; 3] = [Self::Constant, Self::InverseLinear, Self::InverseSquared];
    /// Inspector label of this mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::Constant => "Constant",
            Self::InverseLinear => "Inverse Linear",
            Self::InverseSquared => "Inverse Squared",
        }
    }
}

/// An effector on this object, acting through the object's colliders flagged
/// `used_by_effector` (a trigger for Area / Buoyancy / Point, a solid collider
/// for Platform / Surface). Only the fields of the chosen `kind` are used.
/// Variation is deterministic (a counter hash), so replays and rollback agree.
#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Effector2D {
    /// What kind of force the effector applies.
    pub kind: EffectorKind2D,
    /// Only affect bodies on these layers (bit per layer); off = every layer.
    /// Only affect bodies on these layers (bit per layer).
    pub use_layer_mask: bool,
    /// Layers the effector acts on (only when `use_layer_mask`).
    pub layer_mask: u32,
    // Area / Point / Buoyancy.
    /// Force direction, degrees from +X (Area).
    pub force_angle: f32,
    /// The angle is in world space, not rotated with the object (Area).
    pub use_global_angle: bool,
    /// Force in newtons on each body inside (Area, Point).
    pub force_magnitude: f32,
    /// Random +- on the magnitude, per body per step (Area, Point, Surface speed).
    pub force_variation: f32,
    /// Linear and angular drag on bodies inside (Area, Buoyancy).
    pub linear_drag: f32,
    /// Angular drag on bodies inside (1/s; Area, Buoyancy).
    pub angular_drag: f32,
    // Buoyancy.
    /// Local-space y of the water surface (above the object's origin).
    pub surface_level: f32,
    /// Fluid density (kg/m^3 equivalent; 1 = the default collider density).
    pub fluid_density: f32,
    /// Flow direction, degrees from +X (Buoyancy).
    pub flow_angle: f32,
    /// Flow strength (Buoyancy).
    pub flow_magnitude: f32,
    /// Random +- on the flow, per body per step (Buoyancy).
    pub flow_variation: f32,
    // Point.
    /// Falloff scale of the point force (Point).
    pub distance_scale: f32,
    /// How the point force falls off with distance.
    pub force_mode: PointForceMode2D,
    // Platform.
    /// Only hold contacts from above (a one-way platform).
    pub use_one_way: bool,
    /// Half-width (degrees) of the arc around "up" contacts hold in.
    pub surface_arc: f32,
    /// Sideways contacts (outside the arc) keep friction / bounce.
    pub use_side_friction: bool,
    /// Sideways contacts (outside the arc) keep bounce.
    pub use_side_bounce: bool,
    // Surface.
    /// Speed (m/s) along the surface, positive = to the right of its up.
    pub speed: f32,
    /// Random +- on the surface speed (Surface).
    pub speed_variation: f32,
}

impl Default for Effector2D {
    fn default() -> Self {
        Self {
            kind: EffectorKind2D::Area,
            use_layer_mask: false,
            layer_mask: u32::MAX,
            force_angle: 0.0,
            use_global_angle: false,
            force_magnitude: 10.0,
            force_variation: 0.0,
            linear_drag: 0.0,
            angular_drag: 0.0,
            surface_level: 0.0,
            fluid_density: 2.0,
            flow_angle: 0.0,
            flow_magnitude: 0.0,
            flow_variation: 0.0,
            distance_scale: 1.0,
            force_mode: PointForceMode2D::InverseLinear,
            use_one_way: true,
            surface_arc: 180.0,
            use_side_friction: false,
            use_side_bounce: false,
            speed: 2.0,
            speed_variation: 0.0,
        }
    }
}
impl TypedComponent for Effector2D {
    const NAME: &'static str = "Effector 2D";
}

/// A 2D ray hit.
#[derive(Clone, Copy, Debug)]
pub struct RayHit2D {
    /// The object owning the collider that was hit.
    pub object: ObjectId,
    /// Hit position in world space.
    pub point: Vec2,
    /// Hit surface normal in world space.
    pub normal: Vec2,
    /// Distance from the origin (for shape casts, how far the shape moved).
    pub distance: f32,
}

/// One contact point of a touching pair.
#[derive(Clone, Copy, Debug)]
pub struct ContactPoint2D {
    /// The other object.
    pub other: ObjectId,
    /// Contact position in world space.
    pub point: Vec2,
    /// Pointing from the other collider towards the queried one.
    pub normal: Vec2,
    /// Negative = penetrating.
    pub separation: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn components_round_trip_and_old_data_fills_defaults() {
        let b = Rigidbody2D { mass: 3.0, freeze_rotation: true, material: "m.physmat".into(), ..Default::default() };
        let text = ron::ser::to_string(&b).unwrap();
        let back: Rigidbody2D = ron::from_str(&text).unwrap();
        assert_eq!(back.mass, 3.0);
        assert!(back.freeze_rotation && back.simulated);
        assert_eq!(back.material, "m.physmat");
        // A file from before a field existed still loads.
        let partial: Rigidbody2D = ron::from_str("(mass: 2.0)").unwrap();
        assert_eq!(partial.gravity_scale, 1.0);
        let poly: PolygonCollider2D = ron::from_str("(is_trigger: true)").unwrap();
        assert_eq!(poly.paths.len(), 1);
        let e: Effector2D = ron::from_str("(kind: Platform, surface_arc: 90.0)").unwrap();
        assert_eq!(e.kind, EffectorKind2D::Platform);
        assert!(e.use_one_way);
        let j: Joint2D = ron::from_str("(kind: Wheel)").unwrap();
        assert_eq!(j.kind, Joint2DKind::Wheel);
    }
}
