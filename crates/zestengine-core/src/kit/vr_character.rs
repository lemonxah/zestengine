//! VR player rig: head + hands driven by the headset, teleport / smooth
//! locomotion with snap turning, grabbing, and an avatar body that follows
//! the head. Without a headset the rig falls back to a desktop simulation
//! (mouse look, WASD, the right hand held in front of the camera) so VR games
//! are playable in the editor.

use glam::{EulerRot, Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

use super::character_movement::{CharacterBraking, CharacterMovement, Stride, SwimInput};
use super::swim::{breathe, float_accel, stroke, swims};
use crate::{
    ComponentCommand, ComponentCtx, InputState, MotionState, MoveHost, ObjectId, ObjectRef, Predictable, RayFilter, TypedComponent,
    driven_remotely,
};

/// Share of a hand's speed through the water a stroke turns into the
/// swimmer's speed (the other way).
const ARM_STROKE: f32 = 0.6;

/// How a VR player moves through the world.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum VrLocomotion {
    /// Physical room-scale movement only.
    Disabled,
    /// Point and teleport.
    Teleport,
    /// Smooth stick movement.
    Smooth,
    /// Both methods available.
    Both,
}

impl VrLocomotion {
    /// Every mode, in declaration order.
    pub const ALL: [VrLocomotion; 4] = [Self::Disabled, Self::Teleport, Self::Smooth, Self::Both];
    /// Inspector label of this mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::Disabled => "Room-scale only",
            Self::Teleport => "Teleport",
            Self::Smooth => "Smooth",
            Self::Both => "Teleport + smooth",
        }
    }
    fn teleport(self) -> bool {
        matches!(self, Self::Teleport | Self::Both)
    }
    fn smooth(self) -> bool {
        matches!(self, Self::Smooth | Self::Both)
    }
}

/// Artificial turning; physical headset rotation is always tracked.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum VrTurning { Disabled, #[default] Snap, Smooth }
impl VrTurning {
    pub const ALL: [Self; 3] = [Self::Disabled, Self::Snap, Self::Smooth];
    pub fn label(self) -> &'static str {
        match self { Self::Disabled => "Disabled", Self::Snap => "Snap", Self::Smooth => "Smooth" }
    }
}

/// Which controller a hand slot refers to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VrHand {
    /// The left controller.
    Left,
    /// The right controller.
    Right,
}

/// Whether a teleport a client announces lands somewhere a body can stand: clear head room above
/// the target and ground within reach below it. `hit(origin, dir, len)` says whether a ray meets
/// anything solid. A server runs this against the owner's claimed target (NET-4); the owner's
/// own arc only ever lands on a surface, so an honest teleport passes.
pub fn teleport_landing_ok(target: Vec3, hit: impl Fn(Vec3, Vec3, f32) -> bool) -> bool {
    target.is_finite() && !hit(target + Vec3::Y * 0.05, Vec3::Y, 1.2) && hit(target + Vec3::Y * 0.4, Vec3::NEG_Y, 1.2)
}

/// Where a ballistic teleport arc from `origin` along `dir` lands. `cast`
/// is a ray query `(origin, unit dir, max distance) -> (point, normal)`. Only
/// walkable (upward-facing) surfaces are accepted.
pub fn teleport_arc(
    origin: Vec3,
    dir: Vec3,
    speed: f32,
    max_steps: usize,
    mut cast: impl FnMut(Vec3, Vec3, f32) -> Option<(Vec3, Vec3)>,
) -> Option<Vec3> {
    const STEP: f32 = 0.05;
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let mut p = origin;
    let mut v = dir.normalize_or(Vec3::NEG_Z) * speed;
    for _ in 0..max_steps {
        let next = p + v * STEP;
        v += gravity * STEP;
        let seg = next - p;
        let len = seg.length();
        if len > 1e-5
            && let Some((point, normal)) = cast(p, seg / len, len)
        {
            return (normal.y > 0.7).then_some(point);
        }
        p = next;
    }
    None
}

/// The active scheme is a VR controller scheme that binds `action`. When it
/// isn't (a desktop scheme left active, an older project) the VR Character
/// reads the raw controller input that action defaults to.
fn vr_bound(ctx: &ComponentCtx, action: &str) -> bool {
    ctx.control_scheme().is_some_and(|s| s.device() == Some(crate::InputDevice::Vr) && s.action(action).is_some())
}

fn vr_button(ctx: &ComponentCtx, action: &str, raw: bool) -> bool {
    if vr_bound(ctx, action) { ctx.input.down(action) } else { raw }
}

/// One frame of a VR Character's locomotion input as the rig read it (stick,
/// turn, headset pose), so networked prediction can replay it without the
/// headset.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[doc(hidden)]
pub struct VrStride {
    /// The play space's rotation before the frame's turn.
    pub rig: Quat,
    pub stick: glam::Vec2,
    /// Degrees the play space turns about the head.
    pub turn: f32,
    /// Head pose in stage space.
    pub head: (Vec3, Quat),
    /// Hands in stage space and their world velocity, while there is water
    /// at the head (arm strokes).
    pub hands: Option<[(Vec3, Vec3); 2]>,
    /// Feet target of a teleport this frame.
    pub teleport: Option<Vec3>,
}

const STRIDE_HANDS: u32 = 1;
const STRIDE_TELEPORT: u32 = 2;

impl VrStride {
    /// Flat form carried by an input command.
    pub fn pack(&self) -> Vec<f32> {
        let flags = if self.hands.is_some() { STRIDE_HANDS } else { 0 } | if self.teleport.is_some() { STRIDE_TELEPORT } else { 0 };
        let ((p, q), r) = (self.head, self.rig);
        let mut v = vec![flags as f32, self.stick.x, self.stick.y, self.turn, p.x, p.y, p.z, q.x, q.y, q.z, q.w, r.x, r.y, r.z, r.w];
        if let Some(hands) = self.hands {
            for (pos, vel) in hands {
                v.extend_from_slice(&[pos.x, pos.y, pos.z, vel.x, vel.y, vel.z]);
            }
        }
        if let Some(t) = self.teleport {
            v.extend_from_slice(&[t.x, t.y, t.z]);
        }
        v
    }

    pub fn unpack(v: &[f32]) -> Option<Self> {
        fn vec3(s: &[f32]) -> Vec3 {
            Vec3::new(s[0], s[1], s[2])
        }
        if v.iter().any(|x| !x.is_finite()) { return None; }
        let (&flags, rest) = v.split_first()?;
        let flags = flags as u32;
        let (fixed, mut tail) = rest.split_at_checked(14)?;
        let mut take = |n: usize| {
            let (a, b) = tail.split_at_checked(n)?;
            tail = b;
            Some(a)
        };
        let hands = if flags & STRIDE_HANDS != 0 {
            let h = take(12)?;
            Some([(vec3(&h[0..]), vec3(&h[3..])), (vec3(&h[6..]), vec3(&h[9..]))])
        } else {
            None
        };
        let teleport = if flags & STRIDE_TELEPORT != 0 { Some(vec3(take(3)?)) } else { None };
        let rot = Quat::from_xyzw(fixed[6], fixed[7], fixed[8], fixed[9]);
        let rig = Quat::from_xyzw(fixed[10], fixed[11], fixed[12], fixed[13]);
        (rot.length_squared() > 0.5 && rig.length_squared() > 0.5).then(|| Self {
            rig: rig.normalize(),
            stick: glam::Vec2::new(fixed[0], fixed[1]),
            turn: fixed[2],
            head: (vec3(&fixed[3..]), rot.normalize()),
            hands,
            teleport,
        })
    }
}

#[derive(Clone, Copy)]
#[doc(hidden)]
pub struct Held {
    id: ObjectId,
    offset: Mat4,
}

/// The local VR player: put it on a root object with child head (camera) and
/// hand objects. Moves the play space (this object) with teleport / smooth
/// locomotion and snap turns, writes the tracked head / hand poses into the
/// children, and grabs [`Grabbable`] objects with the grips (trigger is passed
/// to the held object, e.g. a gun). Desktop fallback without a headset.
/// Controller input goes through the bound actions (Move, Turn, Teleport,
/// Grab / Use Left / Right), so players can remap it.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct VrCharacter {
    /// Head (camera) object.
    pub head: ObjectRef,
    /// Left hand object.
    pub left_hand: ObjectRef,
    /// Right hand object.
    pub right_hand: ObjectRef,
    /// How the player moves.
    pub locomotion: VrLocomotion,
    /// Smooth locomotion speed (m/s).
    pub move_speed: f32,
    /// Locomotion acceleration in m/s² (tracked head/hands are never smoothed).
    pub acceleration: f32,
    pub air_control: f32,
    pub gravity: f32,
    pub braking: CharacterBraking,
    /// Fit the movement capsule below the tracked head, including physical crouching.
    pub track_capsule: bool,
    #[serde(skip)]
    pub walker: CharacterMovement,
    /// Degrees per snap turn; 0 = smooth turning.
    pub snap_turn: f32,
    pub turning: VrTurning,
    /// Smooth turn speed (degrees per second; used when `snap_turn` is 0).
    pub smooth_turn_speed: f32,
    /// Shown where a teleport would land (a root object).
    pub teleport_marker: ObjectRef,
    /// Launch speed of the teleport arc (higher reaches further).
    pub teleport_speed: f32,
    /// How close a hand must be to grab an object (m).
    pub grab_radius: f32,
    /// Start at a `Spawn Point` with this tag (picked by network peer id).
    pub spawn_tag: String,
    /// Mouse + keyboard simulation when no headset is running.
    pub desktop_fallback: bool,
    /// Camera height above the floor (m).
    pub eye_height: f32,
    /// Desktop look sensitivity.
    pub look_sensitivity: f32,
    /// Input action for movement (Axis2).
    pub move_action: String,
    /// Input action for desktop look.
    pub look_action: String,
    /// Desktop grab action.
    pub grab_action: String,
    /// Desktop fire action.
    pub fire_action: String,
    /// Teleport action.
    pub teleport_action: String,
    /// VR actions (the "VR Controllers" scheme). An action the active scheme
    /// lacks falls back to the raw controller input it defaults to.
    pub turn_action: String,
    /// Left-hand grab action.
    pub grab_left_action: String,
    /// Right-hand grab action.
    pub grab_right_action: String,
    /// Left-hand use (trigger) action.
    pub use_left_action: String,
    /// Right-hand use (trigger) action.
    pub use_right_action: String,
    /// How the player's avatar is fitted to their measured height.
    pub avatar_scaling: crate::AvatarScaling,
    /// Also map the player's arm span onto the avatar's arms (hand reach).
    pub match_arm_span: bool,
    /// Opens the body menu (Calibrate / Height / Done) in a headset; empty
    /// leaves the button to the game.
    pub body_menu_action: String,
    /// Body menu button state last frame (runtime).
    #[serde(skip)]
    pub body_menu_prev: bool,
    /// A headset is running (runtime).
    #[serde(skip)]
    pub vr_active: bool,
    /// Desktop look pitch (degrees, runtime).
    #[serde(skip)]
    pub pitch: f32,
    /// What each hand holds (runtime).
    #[serde(skip)]
    pub held: [Option<Held>; 2],
    /// Grip state last frame per hand (runtime).
    #[serde(skip)]
    pub grip_prev: [bool; 2],
    /// Hand positions last frame (runtime).
    #[serde(skip)]
    pub hand_prev: [Option<Vec3>; 2],
    /// Hand velocities (runtime).
    #[serde(skip)]
    pub hand_vel: [Vec3; 2],
    /// A snap turn is in progress (runtime).
    #[serde(skip)]
    pub turn_latched: bool,
    /// Aiming (runtime).
    #[serde(skip)]
    pub aiming: bool,
    /// Where the teleport arc lands (runtime).
    #[serde(skip)]
    pub teleport_target: Option<Vec3>,
    /// Started (runtime).
    #[serde(skip)]
    pub started: bool,
    /// Swimming (water past the chest): velocity, and seconds of breath
    /// left while the head has been under.
    #[serde(skip)]
    pub swimming: bool,
    /// Swimming velocity (runtime).
    #[serde(skip)]
    pub swim_vel: Vec3,
    /// Breath seconds left while the head has been under (runtime).
    #[serde(skip)]
    pub breath: Option<f32>,
    /// The locomotion input of the last frame (for networked prediction).
    #[serde(skip)]
    pub last_stride: Option<VrStride>,
}

impl Default for VrCharacter {
    fn default() -> Self {
        Self {
            head: ObjectRef::NONE,
            left_hand: ObjectRef::NONE,
            right_hand: ObjectRef::NONE,
            locomotion: VrLocomotion::Both,
            move_speed: 2.5,
            acceleration: 20.48,
            air_control: 0.3,
            gravity: 20.0,
            braking: CharacterBraking::default(),
            track_capsule: true,
            walker: CharacterMovement::default(),
            snap_turn: 30.0,
            turning: VrTurning::Snap,
            smooth_turn_speed: 120.0,
            teleport_marker: ObjectRef::NONE,
            teleport_speed: 9.0,
            grab_radius: 0.18,
            spawn_tag: String::new(),
            desktop_fallback: true,
            eye_height: 1.65,
            look_sensitivity: 0.12,
            move_action: "Move".into(),
            look_action: "Look".into(),
            grab_action: "Grab".into(),
            fire_action: "Fire".into(),
            teleport_action: "Teleport".into(),
            turn_action: "Turn".into(),
            grab_left_action: "Grab Left".into(),
            grab_right_action: "Grab Right".into(),
            use_left_action: "Use Left".into(),
            use_right_action: "Use Right".into(),
            avatar_scaling: crate::AvatarScaling::ScaleAvatar,
            match_arm_span: false,
            body_menu_action: "Menu".into(),
            body_menu_prev: false,
            vr_active: false,
            pitch: 0.0,
            held: [None; 2],
            grip_prev: [false; 2],
            hand_prev: [None; 2],
            hand_vel: [Vec3::ZERO; 2],
            turn_latched: false,
            aiming: false,
            teleport_target: None,
            started: false,
            swimming: false,
            swim_vel: Vec3::ZERO,
            breath: None,
            last_stride: None,
        }
    }
}

impl VrCharacter {
    /// Object held in a hand, if any.
    pub fn holding(&self, hand: VrHand) -> Option<ObjectId> {
        self.held[hand as usize].map(|h| h.id)
    }

    fn rig_world(ctx: &ComponentCtx) -> Mat4 {
        MoveHost::world(ctx)
    }

    /// Turn the play space by `delta` degrees about a world pivot (the head).
    fn turn(host: &mut dyn MoveHost, pivot: Vec3, delta: f32) {
        let r = Quat::from_rotation_y(delta.to_radians());
        let origin = host.world().w_axis.truncate();
        host.set_world_position(pivot + r * (origin - pivot));
        host.set_rotation((r * host.rotation()).normalize());
    }

    /// Head / hand poses in the rig's local (stage) space.
    fn stage_poses(&self, ctx: &ComponentCtx) -> [(Vec3, Quat); 3] {
        let vr = &ctx.screen.vr;
        if vr.active {
            let head = vr.head.unwrap_or((Vec3::new(0.0, self.eye_height, 0.0), Quat::IDENTITY));
            let hand = |h: Option<(Vec3, Quat)>, side: f32| {
                h.unwrap_or((head.0 + head.1 * Vec3::new(0.2 * side, -0.35, -0.3), head.1))
            };
            [head, hand(vr.left_hand, -1.0), hand(vr.right_hand, 1.0)]
        } else {
            let head_rot = Quat::from_rotation_x(self.pitch.to_radians());
            let head = Vec3::new(0.0, self.eye_height, 0.0);
            [
                (head, head_rot),
                (head + head_rot * Vec3::new(-0.22, -0.28, -0.38), head_rot),
                (head + head_rot * Vec3::new(0.2, -0.2, -0.42), head_rot),
            ]
        }
    }

    /// Swim when the water is past the chest (see [`super::Swimmer`], tuned
    /// by one on this object): the stick strokes where the head looks,
    /// pulling a hand back through the water pushes the other way, the rig
    /// floats with the head just out, and breath runs out with it under.
    /// True while swimming (the walk and teleport are skipped).
    fn swim(&mut self, host: &mut dyn MoveHost, stride: &VrStride, dt: f32) -> bool {
        let s = host.swimmer();
        let rig = host.world();
        let (head_local, head_rot) = stride.head;
        let head = rig.transform_point3(head_local);
        let feet = rig.w_axis.truncate();
        let height = (head.y - feet.y).max(1.0) / 0.92;
        let surface = host.water_surface(head);
        self.swimming = surface.is_some_and(|h| swims(&s, self.swimming, h - feet.y, height));
        let Some(h) = surface.filter(|_| self.swimming) else {
            self.swim_vel = Vec3::ZERO;
            self.breath = self.breath.map(|b| breathe(&s, b, false, dt)).filter(|b| *b < s.breath);
            return false;
        };
        self.breath = Some(breathe(&s, self.breath.unwrap_or(s.breath), head.y < h, dt));
        let look = Quat::from_mat4(&rig) * head_rot;
        let mut target = stroke(look, stride.stick, 0.0, s.swim_speed);
        // Arm strokes: hands under the surface push against the water.
        for (p, vel) in stride.hands.into_iter().flatten() {
            if rig.transform_point3(p).y < h {
                target -= (vel - self.swim_vel) * ARM_STROKE;
            }
        }
        let target = target.clamp_length_max(s.sprint_speed);
        let rate = if target.length_squared() > 1e-4 { s.stroke_accel } else { s.water_drag };
        self.swim_vel = self.swim_vel.lerp(target, 1.0 - (-rate * dt).exp());
        if stride.stick.length_squared() < 0.02 {
            self.swim_vel.y += float_accel(&s, head.y, h, self.swim_vel.y) * dt;
        }
        if host.has_character() {
            host.move_character(self.swim_vel * dt);
            return true;
        }
        let mut origin = feet + self.swim_vel * dt;
        // Not through the floor.
        let filter = RayFilter { exclude: Some(host.self_id()), ..Default::default() };
        if let Some(floor) = host.raycast(origin + Vec3::Y * 1.0, Vec3::NEG_Y, 1.2, &filter)
            && origin.y < floor.point.y
        {
            origin.y = floor.point.y;
            self.swim_vel.y = self.swim_vel.y.max(0.0);
        }
        host.set_world_position(origin);
        true
    }

    /// The movement part of a frame: turn, then swim or walk with the stick.
    /// Live frames and networked prediction's replays both run this. True
    /// while swimming.
    fn stride(&mut self, host: &mut dyn MoveHost, stride: &VrStride, dt: f32) -> bool {
        // The owner's facing, like a first-person view: a replay adopts it.
        host.set_rotation(stride.rig);
        let rig = host.world();
        let head_world = rig.transform_point3(stride.head.0);
        if stride.turn != 0.0 {
            Self::turn(host, head_world, stride.turn);
        }
        // The body follows tracking; the camera and hands remain exact tracked poses.
        if self.track_capsule && let Some((radius, _, _)) = host.capsule() {
            let head = stride.head.0;
            let height = (head.y + 0.12).clamp(radius * 2.0, 3.5_f32.max(radius * 2.0));
            host.set_capsule(height, [head.x, height * 0.5, head.z]);
        }
        if self.swim(host, stride, dt) {
            return true;
        }
        let f = (Quat::from_mat4(&host.world()) * stride.head.1) * Vec3::NEG_Z;
        let head_yaw = Quat::from_rotation_y((-f.x).atan2(-f.z));
        let mv = if self.locomotion.smooth() && stride.stick.length_squared() > 0.02 {
            stride.stick.clamp_length_max(1.0)
        } else { glam::Vec2::ZERO };
        let dir = head_yaw * Vec3::new(mv.x, 0.0, -mv.y);
        self.walker.walk(host, Stride {
            wish: dir * self.move_speed,
            accel: self.acceleration * mv.length(), air_control: self.air_control,
            braking: self.braking, jump: None, gravity: self.gravity,
            swim: SwimInput::default(),
        }, dt);
        false
    }

    /// Snap the play space so the head stands over `target`.
    fn teleport_to(host: &mut dyn MoveHost, head_local: Vec3, target: Vec3) {
        let rig = host.world();
        let head_offset = (rig.transform_point3(head_local) - rig.w_axis.truncate()).with_y(0.0);
        host.set_world_position(target - head_offset);
    }

    fn turn_input(&mut self, rx: f32, dt: f32) -> f32 {
        let mut turn = 0.0;
        if self.turning == VrTurning::Disabled {
            self.turn_latched = false;
        } else if self.turning == VrTurning::Snap && self.snap_turn > 0.0 {
            if rx.abs() > 0.7 && !self.turn_latched {
                self.turn_latched = true;
                turn = -rx.signum() * self.snap_turn;
            } else if rx.abs() < 0.3 {
                self.turn_latched = false;
            }
        } else if rx.abs() > 0.15 {
            turn = -rx * self.smooth_turn_speed * dt;
        }
        turn
    }

    fn locomotion(&mut self, ctx: &mut ComponentCtx, dt: f32) {
        let vr = ctx.screen.vr;
        let (mv, turn, teleport_down) = if vr.active {
            let rx = if vr_bound(ctx, &self.turn_action) { ctx.input.axis(&self.turn_action) } else { vr.right_stick.x };
            let turn = self.turn_input(rx, dt);
            let mv = if vr_bound(ctx, &self.move_action) { ctx.input.axis2(&self.move_action) } else { vr.left_stick };
            (mv, turn, vr_button(ctx, &self.teleport_action, vr.right_stick.y > 0.6))
        } else {
            let look = ctx.input.axis2(&self.look_action);
            self.pitch = (self.pitch - look.y * self.look_sensitivity).clamp(-85.0, 85.0);
            (ctx.input.axis2(&self.move_action), -look.x * self.look_sensitivity, ctx.input.down(&self.teleport_action))
        };
        let poses = self.stage_poses(ctx);
        let rig = Self::rig_world(ctx);
        let wet = ctx.water_surface(rig.transform_point3(poses[0].0)).is_some();
        let hands = wet.then(|| [(poses[1].0, self.hand_vel[0]), (poses[2].0, self.hand_vel[1])]);
        let mut stride = VrStride { rig: *ctx.rotation, stick: mv, turn, head: poses[0], hands, teleport: None };
        if self.stride(ctx, &stride, dt) {
            self.last_stride = Some(stride);
            return;
        }

        if self.locomotion.teleport() {
            if teleport_down {
                self.aiming = true;
                let rig = Self::rig_world(ctx);
                let (from, rot) = if vr.active { poses[2] } else { poses[0] };
                let origin = rig.transform_point3(from);
                let dir = Quat::from_mat4(&rig) * rot * Vec3::NEG_Z;
                let filter = RayFilter { exclude: Some(ctx.self_id()), ..Default::default() };
                self.teleport_target = teleport_arc(origin, dir, self.teleport_speed, 60, |o, d, len| {
                    ctx.raycast_filtered(o, d, len, &filter).map(|h| (h.point, h.normal))
                });
                if self.teleport_marker.is_set() {
                    match self.teleport_target {
                        Some(p) => {
                            ctx.set_local_transform(self.teleport_marker, Some(p + Vec3::Y * 0.02), None, None);
                            ctx.set_active(self.teleport_marker, true);
                        }
                        None => ctx.set_active(self.teleport_marker, false),
                    }
                }
            } else if self.aiming {
                self.aiming = false;
                if let Some(target) = self.teleport_target.take() {
                    Self::teleport_to(ctx, poses[0].0, target);
                    self.walker = CharacterMovement { floor_y: ctx.translation.y, grounded: true, ..Default::default() };
                    // Drop the pre-teleport movement queued earlier in this pass.
                    let me = ctx.self_id();
                    ctx.commands.retain(|c| !matches!(c, ComponentCommand::MoveCharacter { id, .. } if *id == me));
                    if MoveHost::has_character(ctx) { ctx.move_character(Vec3::ZERO); }
                    stride.teleport = Some(target);
                }
                if self.teleport_marker.is_set() {
                    ctx.set_active(self.teleport_marker, false);
                }
            }
        }
        self.last_stride = Some(stride);
    }

    /// Pick up the nearest free [`Grabbable`] within reach of the hand, or
    /// (desktop fallback, `look` = head ray) the one looked at within 2.5 m.
    fn grab(&mut self, ctx: &mut ComponentCtx, hand: usize, hand_world: Mat4, look: Option<(Vec3, Vec3)>) -> Option<Held> {
        const LOOK_REACH: f32 = 2.5;
        let pos = hand_world.w_axis.truncate();
        let taken: Vec<ObjectId> = self.held.iter().flatten().map(|h| h.id).collect();
        let mut best: Option<(f32, ObjectId)> = None;
        for id in super::objects_with::<Grabbable>(ctx) {
            if taken.contains(&id) || !ctx.is_active(id) {
                continue;
            }
            let (Some(g), Some(p)) = (ctx.get::<Grabbable>(id), ctx.position_of(id.into())) else {
                continue;
            };
            let score = match look {
                Some((o, d)) => {
                    let t = (p - o).dot(d);
                    let miss = (o + d * t.clamp(0.0, LOOK_REACH)).distance(p);
                    (t > 0.0 && t <= LOOK_REACH && miss <= g.radius + 0.15).then_some(t)
                }
                None => {
                    let d = p.distance(pos);
                    (d <= self.grab_radius + g.radius).then_some(d)
                }
            };
            if let Some(score) = score
                && best.is_none_or(|(b, _)| score < b)
            {
                best = Some((score, id));
            }
        }
        let (_, id) = best?;
        let g = ctx.get::<Grabbable>(id)?;
        let offset = if g.snap_to_hand {
            let [x, y, z] = g.hold_rotation.map(f32::to_radians);
            Mat4::from_rotation_translation(Quat::from_euler(EulerRot::XYZ, x, y, z), Vec3::from(g.hold_position))
        } else {
            let rel = hand_world.inverse() * ctx.object_matrix(ctx.index_of(id)?)?;
            if look.is_some() {
                // Desktop: bring it into the simulated hand instead of holding it at a distance.
                let (_, rot, _) = rel.to_scale_rotation_translation();
                Mat4::from_rotation_translation(rot, Vec3::new(0.0, 0.0, -0.12))
            } else {
                rel
            }
        };
        ctx.modify::<Grabbable>(id, move |g| {
            g.held = true;
            g.held_by = Some(if hand == 0 { VrHand::Left } else { VrHand::Right });
        });
        if ctx.net.connected {
            ctx.request_ownership_of(id.into());
        }
        Some(Held { id, offset })
    }

    fn release(&mut self, ctx: &mut ComponentCtx, hand: usize) {
        let Some(held) = self.held[hand].take() else {
            return;
        };
        let throw = ctx.get::<Grabbable>(held.id).is_some_and(|g| g.throwable);
        if throw {
            ctx.set_velocity(held.id, self.hand_vel[hand]);
        }
        ctx.modify::<Grabbable>(held.id, |g| {
            g.held = false;
            g.held_by = None;
            g.trigger_down = false;
        });
        if ctx.net.connected {
            ctx.commands.push(ComponentCommand::ReleaseOwnership(held.id));
        }
    }
}

impl Predictable for VrCharacter {
    fn motion(&self) -> MotionState {
        let mut m = MotionState::default();
        self.walker.save(&mut m);
        if self.swimming { m.vel = self.swim_vel; m.grounded = false; }
        m.pitch = self.pitch;
        m.swimming = self.swimming;
        m.breath = self.breath;
        m
    }

    fn set_motion(&mut self, m: &MotionState, _host: &mut dyn MoveHost) {
        self.walker.load(m);
        self.swim_vel = if m.swimming { m.vel } else { Vec3::ZERO };
        self.pitch = m.pitch;
        self.swimming = m.swimming;
        self.breath = m.breath;
    }

    fn frame_extra(&self) -> Vec<f32> {
        self.last_stride.map(|s| s.pack()).unwrap_or_default()
    }

    fn replay(&mut self, host: &mut dyn MoveHost, _input: &InputState, extra: &[f32], dt: f32) {
        let dt = dt.min(0.1);
        let Some(stride) = VrStride::unpack(extra).filter(|_| dt > 0.0) else {
            return;
        };
        if self.stride(host, &stride, dt) {
            return;
        }
        // The owner's teleport, if the arc could have reached it and it lands on open ground.
        if let Some(target) = stride.teleport
            && self.locomotion.teleport()
            && (target - host.world_position()).with_y(0.0).length() <= self.teleport_speed * 3.0 + 1.0
            && {
                let filter = RayFilter { exclude: Some(host.self_id()), ..Default::default() };
                teleport_landing_ok(target, |o, d, len| host.raycast(o, d, len, &filter).is_some())
            }
        {
            Self::teleport_to(host, stride.head.0, target);
            self.walker = CharacterMovement { floor_y: host.translation().y, grounded: true, ..Default::default() };
            if host.has_character() { host.move_character(Vec3::ZERO); }
        }
    }
}

impl TypedComponent for VrCharacter {
    const NAME: &'static str = "VR Character";
    const ALIASES: &'static [&'static str] = &["VR Player"];

    fn predictable(&self) -> Option<&dyn Predictable> {
        Some(self)
    }
    fn predictable_mut(&mut self) -> Option<&mut dyn Predictable> {
        Some(self)
    }

    fn start(&mut self, ctx: &mut ComponentCtx) {
        if !self.spawn_tag.is_empty() {
            let spots: Vec<usize> =
                ctx.spawn_points.iter().filter(|(_, t)| *t == self.spawn_tag).map(|(i, _)| *i).collect();
            if !spots.is_empty() {
                let pick = spots[(ctx.net.local_peer as usize) % spots.len()];
                if let Some(t) = ctx.object_transform(pick) {
                    ctx.set_world_position(t.translation);
                    let yaw = t.rotation.to_euler(EulerRot::YXZ).0;
                    *ctx.rotation = Quat::from_rotation_y(yaw);
                }
            }
        }
        if self.head.is_set() {
            ctx.set_active_camera(self.head);
        }
        if !ctx.screen.vr.active && self.desktop_fallback {
            ctx.set_cursor_locked(true);
        }
        if self.teleport_marker.is_set() {
            ctx.set_active(self.teleport_marker, false);
        }
        self.walker = CharacterMovement { floor_y: ctx.translation.y, grounded: true, ..Default::default() };
        self.started = true;
    }

    fn update(&mut self, ctx: &mut ComponentCtx) {
        if !self.started {
            TypedComponent::start(self, ctx);
        }
        self.last_stride = None;
        if driven_remotely(ctx) {
            return;
        }
        let vr = ctx.screen.vr;
        self.vr_active = vr.active;
        let menu = vr.active && !self.body_menu_action.is_empty() && vr_button(ctx, &self.body_menu_action, vr.menu);
        if menu && !self.body_menu_prev {
            ctx.commands.push(ComponentCommand::Body(crate::BodyCommand::Menu(None)));
        }
        self.body_menu_prev = menu;
        if !vr.active && !self.desktop_fallback {
            return;
        }
        let dt = ctx.dt.min(0.1);
        if dt > 0.0 && !ctx.ui_text_editing() && !ctx.screen.body.calibrating {
            self.locomotion(ctx, dt);
        }

        let poses = self.stage_poses(ctx);
        for (target, (p, r)) in [self.head, self.left_hand, self.right_hand].into_iter().zip(poses) {
            ctx.set_local_transform(target, Some(p), Some(r), None);
        }
        if ctx.screen.body.calibrating { return; }


        let rig = Self::rig_world(ctx);
        let grips = if vr.active {
            [
                vr_button(ctx, &self.grab_left_action, vr.left_grip > 0.6),
                vr_button(ctx, &self.grab_right_action, vr.right_grip > 0.6),
            ]
        } else {
            [false, ctx.input.down(&self.grab_action)]
        };
        let triggers = if vr.active {
            [
                vr_button(ctx, &self.use_left_action, vr.left_trigger),
                vr_button(ctx, &self.use_right_action, vr.right_trigger),
            ]
        } else {
            [false, ctx.input.down(&self.fire_action)]
        };
        for hand in 0..2 {
            let (p, r) = poses[hand + 1];
            let hand_world = rig * Mat4::from_rotation_translation(r, p);
            let pos = hand_world.w_axis.truncate();
            if let Some(prev) = self.hand_prev[hand]
                && dt > 0.0
            {
                self.hand_vel[hand] = self.hand_vel[hand].lerp((pos - prev) / dt, 0.5);
            }
            self.hand_prev[hand] = Some(pos);

            let grip = grips[hand];
            if grip && !self.grip_prev[hand] && self.held[hand].is_none() {
                let look = (!vr.active).then(|| {
                    let (p, r) = poses[0];
                    (rig.transform_point3(p), (Quat::from_mat4(&rig) * r * Vec3::NEG_Z).normalize())
                });
                self.held[hand] = self.grab(ctx, hand, hand_world, look);
            } else if !grip && self.grip_prev[hand] {
                self.release(ctx, hand);
            }
            self.grip_prev[hand] = grip;

            if let Some(held) = self.held[hand] {
                if ctx.index_of(held.id).is_none() {
                    self.held[hand] = None;
                    continue;
                }
                let (_, rot, t) = (hand_world * held.offset).to_scale_rotation_translation();
                ctx.set_local_transform(held.id.into(), Some(t), Some(rot), None);
                ctx.set_velocity(held.id, Vec3::ZERO);
                let trig = triggers[hand];
                ctx.modify::<Grabbable>(held.id, move |g| g.trigger_down = trig);
            }
        }
    }
}

/// Something a [`VrCharacter`] can pick up. Must be a root object. Weapons read
/// `held` / `trigger_down` (see `Hitscan Weapon`'s held trigger mode).
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct Grabbable {
    /// Snap into the hand at `hold_position` / `hold_rotation` (guns, tools);
    /// otherwise it keeps the offset it was grabbed at.
    pub snap_to_hand: bool,
    /// Object position in hand space when snapped.
    pub hold_position: [f32; 3],
    /// Object rotation in hand space when snapped (Euler XYZ, degrees).
    pub hold_rotation: [f32; 3],
    /// Extra reach around the object's origin.
    pub radius: f32,
    /// Keep the hand's velocity when released (needs a dynamic Rigid Body).
    pub throwable: bool,
    /// Return to the start pose after this many seconds untouched away from it (0 = never).
    pub return_after: f32,
    /// Held by someone (runtime).
    #[serde(skip)]
    pub held: bool,
    /// The hand holding it (runtime).
    #[serde(skip)]
    pub held_by: Option<VrHand>,
    /// The holder's trigger is pressed (runtime).
    #[serde(skip)]
    pub trigger_down: bool,
    /// The grab point's rest pose (runtime).
    #[serde(skip)]
    pub home: Option<(Vec3, Quat)>,
    /// Seconds untouched away from its home (runtime).
    #[serde(skip)]
    pub idle: f32,
}

impl Default for Grabbable {
    fn default() -> Self {
        Self {
            snap_to_hand: false,
            hold_position: [0.0; 3],
            hold_rotation: [0.0; 3],
            radius: 0.15,
            throwable: true,
            return_after: 0.0,
            held: false,
            held_by: None,
            trigger_down: false,
            home: None,
            idle: 0.0,
        }
    }
}

impl TypedComponent for Grabbable {
    const NAME: &'static str = "Grabbable";

    fn start(&mut self, ctx: &mut ComponentCtx) {
        self.home = Some((*ctx.translation, *ctx.rotation));
    }

    fn update(&mut self, ctx: &mut ComponentCtx) {
        let Some((pos, rot)) = self.home else {
            self.home = Some((*ctx.translation, *ctx.rotation));
            return;
        };
        if self.held || self.return_after <= 0.0 || ctx.translation.distance(pos) < 0.05 {
            self.idle = 0.0;
            return;
        }
        self.idle += ctx.dt;
        if self.idle >= self.return_after {
            self.idle = 0.0;
            *ctx.translation = pos;
            *ctx.rotation = rot;
            ctx.set_velocity(ctx.self_id(), Vec3::ZERO);
        }
    }
}

/// Avatar body that follows a head: sits `neck_drop` below `head` and turns
/// toward the head's facing. Put the torso / limbs under it. A humanoid
/// skinned model can be driven by the engine's tracker IK instead.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct VrAvatar {
    /// The head object the body follows.
    pub head: ObjectRef,
    /// How far the neck drops below the head (m).
    pub neck_drop: f32,
    /// Body turn rate toward the head (1/s).
    pub turn_rate: f32,
    /// Optional yaw override (degrees; the engine sets it).
    #[serde(skip)]
    pub yaw: Option<f32>,
}

impl Default for VrAvatar {
    fn default() -> Self {
        Self { head: ObjectRef::NONE, neck_drop: 0.22, turn_rate: 6.0, yaw: None }
    }
}

impl TypedComponent for VrAvatar {
    const NAME: &'static str = "VR Avatar";

    fn late_update(&mut self, ctx: &mut ComponentCtx) {
        let Some(head) = ctx.transform_of(self.head) else {
            return;
        };
        let f = head.forward();
        let goal = (-f.x).atan2(-f.z);
        let k = 1.0 - (-self.turn_rate.max(0.0) * ctx.unscaled_dt.min(0.1)).exp();
        let yaw = match self.yaw {
            Some(y) => {
                let diff = (goal - y + std::f32::consts::PI * 3.0).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
                y + diff * k
            }
            None => goal,
        };
        self.yaw = Some(yaw);
        ctx.set_world_position(head.translation - Vec3::Y * self.neck_drop);
        let world = Quat::from_rotation_y(yaw);
        *ctx.rotation = match ctx.parent_world {
            Some(p) => (p.to_scale_rotation_translation().1.inverse() * world).normalize(),
            None => world,
        };
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn turning_and_locomotion_can_be_disabled_independently() {
        let mut vr = super::VrCharacter { turning: super::VrTurning::Disabled, locomotion: super::VrLocomotion::Smooth, ..Default::default() };
        assert_eq!(vr.turn_input(1.0, 0.1), 0.0);
        assert!(vr.locomotion.smooth() && !vr.locomotion.teleport());
        vr.turning = super::VrTurning::Snap;
        assert_eq!(vr.turn_input(1.0, 0.1), -30.0);
        assert_eq!(vr.turn_input(1.0, 0.1), 0.0, "held stick snaps once");
        vr.turn_input(0.0, 0.1);
        assert_eq!(vr.turn_input(1.0, 0.1), -30.0);
        vr.turning = super::VrTurning::Smooth;
        assert!((vr.turn_input(1.0, 0.1) + 12.0).abs() < 1e-5);
        vr.locomotion = super::VrLocomotion::Disabled;
        assert!(!vr.locomotion.smooth() && !vr.locomotion.teleport());
    }

    use super::*;

    #[test]
    fn teleport_landing_needs_head_room_and_ground() {
        let floor_y = 0.0;
        let ceiling_y = 3.0;
        // A world with a floor and (optionally) a ceiling: ray hits when it crosses either plane.
        let world = |ceiling: bool| {
            move |o: Vec3, d: Vec3, len: f32| {
                let end = o + d * len;
                let crosses = |y: f32| (o.y - y) * (end.y - y) <= 0.0 && (o.y - end.y).abs() > 1e-6;
                crosses(floor_y) || (ceiling && crosses(ceiling_y))
            }
        };
        assert!(teleport_landing_ok(Vec3::new(1.0, 0.0, 2.0), world(false)), "open floor");
        assert!(!teleport_landing_ok(Vec3::new(1.0, 5.0, 2.0), world(false)), "floating in the air");
        assert!(!teleport_landing_ok(Vec3::new(1.0, 2.5, 2.0), world(true)), "a ceiling right above the target");
        assert!(!teleport_landing_ok(Vec3::new(f32::NAN, 0.0, 0.0), world(false)));
    }

    fn ground(o: Vec3, d: Vec3, len: f32) -> Option<(Vec3, Vec3)> {
        if d.y >= 0.0 || o.y < 0.0 {
            return None;
        }
        let t = -o.y / d.y;
        (t <= len).then(|| (o + d * t, Vec3::Y))
    }

    #[test]
    fn teleport_arc_lands_on_the_ground_ahead() {
        let hit = teleport_arc(Vec3::new(0.0, 1.2, 0.0), Vec3::new(0.0, 0.3, -1.0), 8.0, 80, ground).expect("lands");
        assert!(hit.y.abs() < 1e-3);
        assert!(hit.z < -2.0, "travels forward: {hit:?}");
        assert!(hit.x.abs() < 1e-3);
    }

    #[test]
    fn locomotion_input_packs_for_input_commands() {
        let full = VrStride {
            rig: Quat::from_rotation_y(0.4),
            stick: glam::Vec2::new(0.2, -1.0),
            turn: -30.0,
            head: (Vec3::new(0.1, 1.6, -0.05), Quat::from_rotation_x(-0.3)),
            hands: Some([(Vec3::new(-0.2, 1.1, -0.3), Vec3::X), (Vec3::new(0.2, 1.0, -0.3), Vec3::NEG_Y)]),
            teleport: Some(Vec3::new(4.0, 0.0, -6.0)),
        };
        for s in [full, VrStride { hands: None, teleport: None, ..full }] {
            let back = VrStride::unpack(&s.pack()).expect("round trip");
            assert!(back.rig.angle_between(s.rig) < 1e-5 && back.head.1.angle_between(s.head.1) < 1e-5);
            assert_eq!((back.stick, back.turn, back.head.0, back.hands, back.teleport), (s.stick, s.turn, s.head.0, s.hands, s.teleport));
        }
        assert!(VrStride::unpack(&full.pack()[..20]).is_none(), "cut short");
        assert!(VrStride::unpack(&[]).is_none());
    }

    #[test]
    fn teleport_rejects_walls_and_misses() {
        let wall = |o: Vec3, d: Vec3, len: f32| {
            let t = (o.z + 2.0) / -d.z;
            (d.z < 0.0 && t >= 0.0 && t <= len).then(|| (o + d * t, Vec3::Z))
        };
        assert!(teleport_arc(Vec3::new(0.0, 1.2, 0.0), Vec3::NEG_Z, 8.0, 80, wall).is_none());
        assert!(teleport_arc(Vec3::new(0.0, 1.2, 0.0), Vec3::Y, 8.0, 5, ground).is_none());
    }
}
