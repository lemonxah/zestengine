//! Character views: first person, third person (orbit /
//! over-the-shoulder boom with collision) and a smoothed follow camera.

use glam::{EulerRot, Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

use super::character_movement::{CharacterBraking, CharacterMovement, Stride, SwimInput};
use crate::motion::view_of;
use crate::{ComponentCtx, InputState, MotionState, MoveHost, ObjectRef, Predictable, RayFilter, TypedComponent, driven_remotely};

/// Rotation whose forward (-Z) points along `dir`, keeping +Y up.
pub fn look_rotation(dir: Vec3) -> Quat {
    let f = dir.normalize_or(Vec3::NEG_Z);
    let up = if f.y.abs() > 0.999 { Vec3::Z } else { Vec3::Y };
    Quat::from_mat4(&Mat4::look_to_rh(Vec3::ZERO, f, up).inverse()).normalize()
}

/// Exponential smoothing factor for a rate (1/s) over `dt`.
fn smooth(rate: f32, dt: f32) -> f32 {
    1.0 - (-rate.max(0.0) * dt).exp()
}

/// Move `current` toward `target` (degrees) by at most `max_step`, the short way round.
fn approach_angle(current: f32, target: f32, max_step: f32) -> f32 {
    let diff = (target - current + 540.0).rem_euclid(360.0) - 180.0;
    current + diff.clamp(-max_step, max_step)
}

/// First-person shooter movement: mouse look (body yaws, child camera
/// pitches), WASD, jump, sprint and crouch. Walks through the physics world
/// with a [`crate::CharacterController`] on the same object.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct FirstPersonCharacter {
    /// Child camera; the controller writes its local height and pitch.
    pub camera: ObjectRef,
    /// Walk speed (m/s).
    pub walk_speed: f32,
    /// Sprint speed (m/s).
    pub sprint_speed: f32,
    /// Crouch speed (m/s).
    pub crouch_speed: f32,
    /// Horizontal acceleration (m/s^2).
    pub acceleration: f32,
    /// Ground and falling friction/braking, shared by every character view.
    pub braking: CharacterBraking,
    /// Fraction of `acceleration` available in the air.
    pub air_control: f32,
    /// Jump height (m).
    pub jump_height: f32,
    /// Gravity while airborne (m/s^2).
    pub gravity: f32,
    /// Look sensitivity (degrees per input unit).
    pub look_sensitivity: f32,
    /// Camera height above the feet (m).
    pub eye_height: f32,
    /// Camera height while crouched (m).
    pub crouch_eye_height: f32,
    /// Character capsule height while crouched (feet stay put); 0 keeps the
    /// capsule unchanged. Standing back up waits for headroom.
    pub crouch_height: f32,
    /// Crouch toggles on press instead of while held.
    pub crouch_toggle: bool,
    /// Lowest camera pitch (degrees, negative = up).
    pub min_pitch: f32,
    /// Highest camera pitch (degrees, positive = down).
    pub max_pitch: f32,
    /// Lock the cursor while this controller drives.
    pub lock_cursor: bool,
    /// Input action for movement (Axis2).
    pub move_action: String,
    /// Input action for look (Axis2).
    pub look_action: String,
    /// Input action for jump.
    pub jump_action: String,
    /// Input action for sprint.
    pub sprint_action: String,
    /// Input action for crouch.
    pub crouch_action: String,
    /// Body yaw (degrees, runtime).
    #[serde(skip)]
    pub yaw: f32,
    /// Camera pitch (degrees, runtime).
    #[serde(skip)]
    pub pitch: f32,
    /// Crouched (runtime).
    #[serde(skip)]
    pub crouched: bool,
    /// Sprinting (runtime).
    #[serde(skip)]
    pub sprinting: bool,
    /// Smoothed camera height (runtime).
    #[serde(skip)]
    pub eye: f32,
    /// Walk / swim state (runtime).
    #[serde(skip)]
    pub walker: CharacterMovement,
    /// Started (runtime).
    #[serde(skip)]
    pub started: bool,
    /// Standing capsule (height, center), captured on the first crouch.
    #[serde(skip)]
    pub stand_capsule: Option<(f32, [f32; 3])>,
    /// The capsule is currently resized for crouch (runtime).
    #[serde(skip)]
    pub capsule_crouched: bool,
    /// The local player drives it and it has taken the camera / cursor.
    #[serde(skip)]
    pub controlling: bool,
}

impl Default for FirstPersonCharacter {
    fn default() -> Self {
        Self {
            camera: ObjectRef::NONE,
            walk_speed: 4.5,
            sprint_speed: 7.5,
            crouch_speed: 2.2,
            acceleration: 20.48,
            braking: CharacterBraking::default(),
            air_control: 0.3,
            jump_height: 1.1,
            gravity: 20.0,
            look_sensitivity: 0.12,
            eye_height: 1.65,
            crouch_eye_height: 1.0,
            crouch_height: 1.2,
            crouch_toggle: false,
            min_pitch: -85.0,
            max_pitch: 85.0,
            lock_cursor: true,
            move_action: "Move".into(),
            look_action: "Look".into(),
            jump_action: "Jump".into(),
            sprint_action: "Sprint".into(),
            crouch_action: "Crouch".into(),
            yaw: 0.0,
            pitch: 0.0,
            crouched: false,
            sprinting: false,
            eye: 1.65,
            walker: CharacterMovement::default(),
            started: false,
            stand_capsule: None,
            capsule_crouched: false,
            controlling: false,
        }
    }
}

impl FirstPersonCharacter {
    /// Whether something above a crouched capsule stops it from standing up.
    fn blocked_overhead(&self, host: &dyn MoveHost) -> bool {
        let (Some((stand_h, _)), Some((radius, height, _))) = (self.stand_capsule, host.capsule()) else {
            return false;
        };
        let feet = host.world_position();
        let rise = stand_h - self.crouch_height.min(stand_h) + 0.05;
        let filter = RayFilter { exclude: Some(host.self_id()), ..Default::default() };
        let r = radius * 0.7;
        [Vec3::ZERO, Vec3::X * r, Vec3::NEG_X * r, Vec3::Z * r, Vec3::NEG_Z * r].iter().any(|o| {
            let from = feet + *o + Vec3::Y * (height - 0.05).max(0.05);
            host.raycast(from, Vec3::Y, rise, &filter).is_some()
        })
    }

    /// Resize the character capsule to its crouched or standing size (feet
    /// stay put).
    fn resize_capsule(&mut self, host: &mut dyn MoveHost, crouched: bool) {
        let Some((_, height, center)) = host.capsule() else {
            return;
        };
        let stand = *self.stand_capsule.get_or_insert((height, center));
        let (height, center) = if crouched {
            let h = self.crouch_height.min(stand.0);
            (h, [stand.1[0], stand.1[1] - (stand.0 - h) * 0.5, stand.1[2]])
        } else {
            stand
        };
        host.set_capsule(height, center);
        self.capsule_crouched = crouched;
    }

    /// Resize the character capsule to match the crouch state.
    fn apply_crouch_capsule(&mut self, host: &mut dyn MoveHost) {
        if self.crouch_height <= 0.0 || self.crouched == self.capsule_crouched {
            return;
        }
        self.resize_capsule(host, self.crouched);
    }

    /// Current horizontal speed limit from the sprint / crouch state.
    pub fn target_speed(&self) -> f32 {
        if self.crouched {
            self.crouch_speed
        } else if self.sprinting {
            self.sprint_speed
        } else {
            self.walk_speed
        }
    }

    /// One frame of movement: look (or the recorded `view`), crouch, then
    /// walk, swim or climb, facing the view's yaw. `update` runs exactly this
    /// through its context, and networked prediction replays it.
    pub fn step(&mut self, host: &mut dyn MoveHost, input: &InputState, dt: f32, typing: bool, view: Option<(f32, f32)>) {
        let dt = dt.min(0.1);
        if dt <= 0.0 {
            return;
        }
        match view {
            Some((yaw, pitch)) => {
                self.yaw = yaw;
                // A networked view is the owner's claim: held to this controller's limits.
                self.pitch = pitch.clamp(self.min_pitch, self.max_pitch);
            }
            None if !typing => {
                let look = input.axis2(&self.look_action);
                self.yaw -= look.x * self.look_sensitivity;
                self.pitch = (self.pitch - look.y * self.look_sensitivity).clamp(self.min_pitch, self.max_pitch);
            }
            None => {}
        }
        let mv = if typing { glam::Vec2::ZERO } else { input.axis2(&self.move_action) };
        if self.crouch_toggle {
            if input.pressed(&self.crouch_action) {
                self.crouched = !self.crouched;
            }
        } else {
            self.crouched = input.down(&self.crouch_action);
        }
        if !self.crouched && self.capsule_crouched && self.blocked_overhead(host) {
            self.crouched = true;
        }
        self.apply_crouch_capsule(host);
        self.sprinting = !self.crouched && mv.y > 0.1 && input.down(&self.sprint_action);

        let yaw_rot = Quat::from_rotation_y(self.yaw.to_radians());
        let dir = (yaw_rot * Vec3::X * mv.x + yaw_rot * Vec3::NEG_Z * mv.y).clamp_length_max(1.0);
        let accel = self.acceleration * dir.length();
        let jump = (!typing && !self.crouched && input.pressed(&self.jump_action))
            .then(|| (2.0 * self.gravity * self.jump_height.max(0.0)).sqrt());
        let swim = SwimInput {
            look: yaw_rot * Quat::from_rotation_x(self.pitch.to_radians()),
            input: mv,
            rise: if typing { 0.0 } else { input.down(&self.jump_action) as u8 as f32 - input.down(&self.crouch_action) as u8 as f32 },
            sprint: !typing && input.down(&self.sprint_action),
        };
        let stride = Stride { wish: dir * self.target_speed(), accel, air_control: self.air_control, braking: self.braking, jump, gravity: self.gravity, swim };
        self.walker.step(host, stride, dt);
        host.set_rotation(yaw_rot);
    }

    /// The first frame the local player drives it: take the camera and cursor.
    fn take_control(&mut self, ctx: &mut ComponentCtx) {
        if self.lock_cursor {
            ctx.set_cursor_locked(true);
        }
        if self.camera.is_set() {
            ctx.set_active_camera(self.camera);
        }
        self.controlling = true;
    }
}

impl Predictable for FirstPersonCharacter {
    fn motion(&self) -> MotionState {
        let mut m = MotionState {
            yaw: self.yaw,
            pitch: self.pitch,
            crouched: self.crouched,
            crouch_capsule: self.capsule_crouched,
            ..Default::default()
        };
        self.walker.save(&mut m);
        m
    }

    fn set_motion(&mut self, m: &MotionState, host: &mut dyn MoveHost) {
        self.walker.load(m);
        self.yaw = m.yaw;
        self.pitch = m.pitch;
        self.crouched = m.crouched;
        if self.crouch_height > 0.0 && m.crouch_capsule != self.capsule_crouched {
            self.resize_capsule(host, m.crouch_capsule);
        }
    }

    fn frame_extra(&self) -> Vec<f32> {
        vec![self.yaw, self.pitch]
    }

    fn replay(&mut self, host: &mut dyn MoveHost, input: &InputState, extra: &[f32], dt: f32) {
        self.step(host, input, dt, false, view_of(extra));
    }
}

impl TypedComponent for FirstPersonCharacter {
    const NAME: &'static str = "First Person Character";
    const ALIASES: &'static [&'static str] = &["First Person Controller"];

    fn predictable(&self) -> Option<&dyn Predictable> {
        Some(self)
    }
    fn predictable_mut(&mut self) -> Option<&mut dyn Predictable> {
        Some(self)
    }

    fn start(&mut self, ctx: &mut ComponentCtx) {
        self.yaw = ctx.rotation.to_euler(EulerRot::YXZ).0.to_degrees();
        self.walker = CharacterMovement { floor_y: ctx.translation.y, grounded: true, ..Default::default() };
        self.eye = self.eye_height;
        self.controlling = false;
        if !driven_remotely(ctx) {
            self.take_control(ctx);
        }
        self.started = true;
    }

    fn update(&mut self, ctx: &mut ComponentCtx) {
        if !self.started {
            TypedComponent::start(self, ctx);
        }
        // Another peer drives this copy (its owner, or the host for
        // predicted movement).
        if driven_remotely(ctx) {
            self.controlling = false;
            return;
        }
        if !self.controlling {
            self.take_control(ctx);
        }
        let dt = ctx.dt.min(0.1);
        if dt <= 0.0 {
            return;
        }
        let (input, typing) = (ctx.input, ctx.ui_text_editing());
        self.step(ctx, input, dt, typing, None);

        let eye_goal = if self.crouched { self.crouch_eye_height } else { self.eye_height };
        self.eye += (eye_goal - self.eye) * smooth(12.0, dt);
        ctx.set_local_transform(
            self.camera,
            Some(Vec3::new(0.0, self.eye, 0.0)),
            Some(Quat::from_rotation_x(self.pitch.to_radians())),
            None,
        );
    }
}

/// Camera position on a boom: `distance` behind and `shoulder` to the right of
/// `pivot` along `rot`, pulled in to `hit_distance - radius` when the boom
/// would pass through geometry.
pub fn boom_position(pivot: Vec3, rot: Quat, shoulder: f32, distance: f32, hit_distance: Option<f32>, radius: f32) -> Vec3 {
    let offset = rot * Vec3::new(shoulder, 0.0, distance);
    let len = offset.length();
    if len < 1e-4 {
        return pivot;
    }
    let dir = offset / len;
    let reach = match hit_distance {
        Some(d) => (d - radius).clamp(0.1, len),
        None => len,
    };
    pivot + dir * reach
}

/// Third-person character: camera orbits on a collision-aware boom over the
/// shoulder, the body turns toward its movement, and holding Aim pulls the
/// camera in and makes the body face where the camera looks (strafing).
/// `camera` must be a root object (the boom places it in world space).
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct ThirdPersonCharacter {
    /// Root object for the camera; the boom places it in world space.
    pub camera: ObjectRef,
    /// Walk speed (m/s).
    pub walk_speed: f32,
    /// Sprint speed (m/s).
    pub sprint_speed: f32,
    /// Movement speed while aiming (m/s).
    pub aim_speed: f32,
    /// Horizontal acceleration (m/s^2).
    pub acceleration: f32,
    pub air_control: f32,
    /// Ground and falling friction/braking, shared by every character view.
    pub braking: CharacterBraking,
    /// Body turn rate, degrees per second.
    pub turn_speed: f32,
    /// Jump height (m).
    pub jump_height: f32,
    /// Gravity while airborne (m/s^2).
    pub gravity: f32,
    /// Look sensitivity (degrees per input unit).
    pub look_sensitivity: f32,
    /// Lowest camera pitch (degrees).
    pub min_pitch: f32,
    /// Highest camera pitch (degrees).
    pub max_pitch: f32,
    /// Boom pivot above the character's feet.
    pub pivot_height: f32,
    /// Boom length (m).
    pub distance: f32,
    /// Boom length while aiming (m).
    pub aim_distance: f32,
    /// Camera offset to the right of the pivot (m).
    pub shoulder_offset: f32,
    /// Shoulder offset while aiming (m).
    pub aim_shoulder_offset: f32,
    /// Keep the camera this far from walls.
    pub camera_radius: f32,
    /// How fast the boom moves between hip and aim framing.
    pub zoom_rate: f32,
    /// Lock the cursor while this controller drives.
    pub lock_cursor: bool,
    /// Input action for movement (Axis2).
    pub move_action: String,
    /// Input action for look (Axis2).
    pub look_action: String,
    /// Input action for jump.
    pub jump_action: String,
    /// Input action for sprint.
    pub sprint_action: String,
    /// Input action for aiming.
    pub aim_action: String,
    /// Hip-fire: turning the body toward the camera while this is held keeps
    /// the weapon pointing at the crosshair when shooting without aiming.
    pub fire_action: String,
    /// Aiming (runtime).
    #[serde(skip)]
    pub aiming: bool,
    /// Camera yaw (degrees, runtime).
    #[serde(skip)]
    pub cam_yaw: f32,
    /// Camera pitch (degrees, runtime).
    #[serde(skip)]
    pub cam_pitch: f32,
    /// Body yaw (degrees, runtime).
    #[serde(skip)]
    pub body_yaw: f32,
    /// Smoothed boom length (runtime).
    #[serde(skip)]
    pub cur_distance: f32,
    /// Smoothed shoulder offset (runtime).
    #[serde(skip)]
    pub cur_shoulder: f32,
    /// Walk / swim state (runtime).
    #[serde(skip)]
    pub walker: CharacterMovement,
    /// Started (runtime).
    #[serde(skip)]
    pub started: bool,
    /// The local player drives it and it has taken the camera / cursor.
    #[serde(skip)]
    pub controlling: bool,
}

impl Default for ThirdPersonCharacter {
    fn default() -> Self {
        Self {
            camera: ObjectRef::NONE,
            walk_speed: 4.0,
            sprint_speed: 7.0,
            aim_speed: 2.5,
            acceleration: 20.48,
            braking: CharacterBraking::default(),
            air_control: 0.3,
            turn_speed: 720.0,
            jump_height: 1.1,
            gravity: 20.0,
            look_sensitivity: 0.12,
            min_pitch: -60.0,
            max_pitch: 70.0,
            pivot_height: 1.55,
            distance: 3.6,
            aim_distance: 1.5,
            shoulder_offset: 0.55,
            aim_shoulder_offset: 0.7,
            camera_radius: 0.25,
            zoom_rate: 10.0,
            lock_cursor: true,
            move_action: "Move".into(),
            look_action: "Look".into(),
            jump_action: "Jump".into(),
            sprint_action: "Sprint".into(),
            aim_action: "Aim".into(),
            fire_action: "Fire".into(),
            aiming: false,
            cam_yaw: 0.0,
            cam_pitch: -10.0,
            body_yaw: 0.0,
            cur_distance: 3.6,
            cur_shoulder: 0.55,
            walker: CharacterMovement::default(),
            started: false,
            controlling: false,
        }
    }
}

impl ThirdPersonCharacter {
    /// One frame of movement: orbit the view by the look input (or set the
    /// recorded `view`), turn the body toward the movement (toward the view
    /// while aiming), then walk, swim or climb. `update` runs exactly this
    /// through its context, and networked prediction replays it.
    pub fn step(&mut self, host: &mut dyn MoveHost, input: &InputState, dt: f32, typing: bool, view: Option<(f32, f32)>) {
        let dt = dt.min(0.1);
        if dt <= 0.0 {
            return;
        }
        match view {
            Some((yaw, pitch)) => {
                self.cam_yaw = yaw;
                self.cam_pitch = pitch.clamp(self.min_pitch, self.max_pitch);
            }
            None if !typing => {
                let look = input.axis2(&self.look_action);
                self.cam_yaw -= look.x * self.look_sensitivity;
                self.cam_pitch = (self.cam_pitch - look.y * self.look_sensitivity).clamp(self.min_pitch, self.max_pitch);
            }
            None => {}
        }
        self.aiming = !typing && input.down(&self.aim_action);
        let firing = !typing && input.down(&self.fire_action);
        let mv = if typing { glam::Vec2::ZERO } else { input.axis2(&self.move_action) };
        let cam_rot = Quat::from_rotation_y(self.cam_yaw.to_radians());
        let dir = (cam_rot * Vec3::X * mv.x + cam_rot * Vec3::NEG_Z * mv.y).clamp_length_max(1.0);
        let sprint = !self.aiming && mv.length() > 0.1 && input.down(&self.sprint_action);
        let speed = if self.aiming {
            self.aim_speed
        } else if sprint {
            self.sprint_speed
        } else {
            self.walk_speed
        };

        let step = self.turn_speed * dt;
        if self.aiming || firing {
            // Face away from the camera (back toward it) so the weapon points
            // at the crosshair whether aiming or firing from the hip.
            self.body_yaw = approach_angle(self.body_yaw, self.cam_yaw, step * 2.0);
        } else if dir.length_squared() > 1e-3 {
            let goal = (-dir.x).atan2(-dir.z).to_degrees();
            self.body_yaw = approach_angle(self.body_yaw, goal, step);
        }
        let jump = (!typing && input.pressed(&self.jump_action))
            .then(|| (2.0 * self.gravity * self.jump_height.max(0.0)).sqrt());
        let accel = self.acceleration * dir.length();
        let swim = SwimInput {
            look: cam_rot * Quat::from_rotation_x(self.cam_pitch.to_radians()),
            input: mv,
            rise: if typing { 0.0 } else { input.down(&self.jump_action) as u8 as f32 },
            sprint,
        };
        self.walker.step(host, Stride { wish: dir * speed, accel, air_control: self.air_control, braking: self.braking, jump, gravity: self.gravity, swim }, dt);
        host.set_rotation(Quat::from_rotation_y(self.body_yaw.to_radians()));
    }

    /// The first frame the local player drives it: take the camera and cursor.
    fn take_control(&mut self, ctx: &mut ComponentCtx) {
        if self.lock_cursor {
            ctx.set_cursor_locked(true);
        }
        if self.camera.is_set() {
            ctx.set_active_camera(self.camera);
        }
        self.controlling = true;
    }
}

impl Predictable for ThirdPersonCharacter {
    fn motion(&self) -> MotionState {
        let mut m = MotionState { yaw: self.cam_yaw, pitch: self.cam_pitch, body_yaw: self.body_yaw, ..Default::default() };
        self.walker.save(&mut m);
        m
    }

    fn set_motion(&mut self, m: &MotionState, _host: &mut dyn MoveHost) {
        self.walker.load(m);
        self.cam_yaw = m.yaw;
        self.cam_pitch = m.pitch;
        self.body_yaw = m.body_yaw;
    }

    fn frame_extra(&self) -> Vec<f32> {
        vec![self.cam_yaw, self.cam_pitch]
    }

    fn replay(&mut self, host: &mut dyn MoveHost, input: &InputState, extra: &[f32], dt: f32) {
        self.step(host, input, dt, false, view_of(extra));
    }
}

impl TypedComponent for ThirdPersonCharacter {
    const NAME: &'static str = "Third Person Character";
    const ALIASES: &'static [&'static str] = &["Third Person Controller"];

    fn predictable(&self) -> Option<&dyn Predictable> {
        Some(self)
    }
    fn predictable_mut(&mut self) -> Option<&mut dyn Predictable> {
        Some(self)
    }

    fn start(&mut self, ctx: &mut ComponentCtx) {
        self.body_yaw = ctx.rotation.to_euler(EulerRot::YXZ).0.to_degrees();
        self.cam_yaw = self.body_yaw;
        self.cur_distance = self.distance;
        self.cur_shoulder = self.shoulder_offset;
        self.walker = CharacterMovement { floor_y: ctx.translation.y, grounded: true, ..Default::default() };
        self.controlling = false;
        if !driven_remotely(ctx) {
            self.take_control(ctx);
        }
        self.started = true;
    }

    fn update(&mut self, ctx: &mut ComponentCtx) {
        if !self.started {
            TypedComponent::start(self, ctx);
        }
        if driven_remotely(ctx) {
            self.controlling = false;
            return;
        }
        if !self.controlling {
            self.take_control(ctx);
        }
        let dt = ctx.dt.min(0.1);
        if dt <= 0.0 {
            return;
        }
        let (input, typing) = (ctx.input, ctx.ui_text_editing());
        self.step(ctx, input, dt, typing, None);
    }

    fn late_update(&mut self, ctx: &mut ComponentCtx) {
        if !self.started || !self.camera.is_set() || !self.controlling {
            return;
        }
        let dt = ctx.unscaled_dt.min(0.1);
        let (dist_goal, shoulder_goal) = if self.aiming {
            (self.aim_distance, self.aim_shoulder_offset)
        } else {
            (self.distance, self.shoulder_offset)
        };
        let k = smooth(self.zoom_rate, dt);
        self.cur_distance += (dist_goal - self.cur_distance) * k;
        self.cur_shoulder += (shoulder_goal - self.cur_shoulder) * k;

        let pivot = ctx.self_transform().translation + Vec3::Y * self.pivot_height;
        let rot = Quat::from_euler(EulerRot::YXZ, self.cam_yaw.to_radians(), self.cam_pitch.to_radians(), 0.0);
        let offset = rot * Vec3::new(self.cur_shoulder, 0.0, self.cur_distance);
        let hit = offset
            .try_normalize()
            .and_then(|d| ctx.raycast(pivot, d, offset.length() + self.camera_radius))
            .map(|h| h.distance);
        let pos = boom_position(pivot, rot, self.cur_shoulder, self.cur_distance, hit, self.camera_radius);
        ctx.set_local_transform(self.camera, Some(pos), Some(rot), None);
    }
}

/// Smoothed follow camera for top-down / isometric views: keeps a world
/// offset from `target`, looks at it, and zooms the offset with an axis action.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct FollowCamera {
    /// The object to follow.
    pub target: ObjectRef,
    /// World-space offset from the target (before zoom).
    pub offset: [f32; 3],
    /// Look at this point relative to the target.
    pub look_offset: [f32; 3],
    /// Follow rate (1/s); 0 snaps.
    pub smoothing: f32,
    /// Input action for zoom (Axis1).
    pub zoom_action: String,
    /// Zoom multiplier per full axis deflection per second.
    pub zoom_step: f32,
    /// Smallest zoom multiplier.
    pub min_zoom: f32,
    /// Largest zoom multiplier.
    pub max_zoom: f32,
    /// Current zoom multiplier (runtime).
    #[serde(skip)]
    pub zoom: f32,
    /// Camera position the engine last wrote (runtime).
    #[serde(skip)]
    pub pos: Option<Vec3>,
}

impl Default for FollowCamera {
    fn default() -> Self {
        Self {
            target: ObjectRef::NONE,
            offset: [0.0, 12.0, 9.0],
            look_offset: [0.0, 0.5, 0.0],
            smoothing: 6.0,
            zoom_action: "Zoom".into(),
            zoom_step: 0.1,
            min_zoom: 0.4,
            max_zoom: 2.0,
            zoom: 1.0,
            pos: None,
        }
    }
}

impl TypedComponent for FollowCamera {
    const NAME: &'static str = "Follow Camera";

    fn start(&mut self, ctx: &mut ComponentCtx) {
        self.zoom = 1.0;
        self.pos = None;
        ctx.set_active_camera(ctx.self_id().into());
    }

    fn late_update(&mut self, ctx: &mut ComponentCtx) {
        let Some(target) = ctx.position_of(self.target) else {
            return;
        };
        if !self.zoom_action.is_empty() && !ctx.pointer_over_ui() {
            let wheel = ctx.input.axis(&self.zoom_action);
            self.zoom = (self.zoom * (1.0 - wheel * self.zoom_step)).clamp(self.min_zoom, self.max_zoom);
        }
        let desired = target + Vec3::from(self.offset) * self.zoom;
        let pos = match self.pos {
            Some(p) if self.smoothing > 0.0 => p.lerp(desired, smooth(self.smoothing, ctx.unscaled_dt.min(0.1))),
            _ => desired,
        };
        self.pos = Some(pos);
        ctx.set_world_position(pos);
        *ctx.rotation = look_rotation(target + Vec3::from(self.look_offset) - pos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn look_rotation_points_forward() {
        for dir in [Vec3::X, Vec3::NEG_Z, Vec3::new(1.0, -1.0, 0.5), Vec3::NEG_Y] {
            let q = look_rotation(dir);
            assert!((q * Vec3::NEG_Z - dir.normalize()).length() < 1e-4, "{dir:?}");
        }
    }

    #[test]
    fn boom_pulls_in_on_hit() {
        let pivot = Vec3::new(0.0, 1.5, 0.0);
        let free = boom_position(pivot, Quat::IDENTITY, 0.0, 4.0, None, 0.2);
        assert!((free - Vec3::new(0.0, 1.5, 4.0)).length() < 1e-5);
        let blocked = boom_position(pivot, Quat::IDENTITY, 0.0, 4.0, Some(2.0), 0.2);
        assert!((blocked - Vec3::new(0.0, 1.5, 1.8)).length() < 1e-5);
        let shoulder = boom_position(pivot, Quat::IDENTITY, 0.5, 2.0, None, 0.2);
        assert!(shoulder.x > 0.0 && shoulder.z > 0.0);
    }

    /// A root object with no Character Controller: moves freely on a flat floor.
    struct FreeHost {
        id: crate::ObjectId,
        pos: Vec3,
        rot: Quat,
    }

    impl MoveHost for FreeHost {
        fn self_id(&self) -> crate::ObjectId {
            self.id
        }
        fn translation(&self) -> Vec3 {
            self.pos
        }
        fn set_translation(&mut self, t: Vec3) {
            self.pos = t;
        }
        fn rotation(&self) -> Quat {
            self.rot
        }
        fn set_rotation(&mut self, r: Quat) {
            self.rot = r;
        }
        fn parent_world(&self) -> Option<Mat4> {
            None
        }
        fn has_character(&self) -> bool {
            false
        }
        fn move_character(&mut self, _: Vec3) {}
        fn grounded(&self) -> bool {
            true
        }
        fn capsule(&self) -> Option<(f32, f32, [f32; 3])> {
            None
        }
        fn set_capsule(&mut self, _: f32, _: [f32; 3]) {}
        fn swimmer(&self) -> super::super::swim::Swimmer {
            Default::default()
        }
        fn water_surface(&self, _: Vec3) -> Option<f32> {
            None
        }
        fn raycast(&self, _: Vec3, _: Vec3, _: f32, _: &RayFilter) -> Option<crate::RayHit> {
            None
        }
        fn set_animator_bool(&mut self, _: &str, _: bool) {}
    }

    /// Ground speed after two seconds of holding forward (and Sprint when `sprint`).
    fn settled_speed(sprint: bool, mut step: impl FnMut(&mut FreeHost, &InputState)) -> f32 {
        let forward = crate::ActionValue { down: true, value: 1.0, axis2: glam::Vec2::Y, ..Default::default() };
        let held = crate::ActionValue { down: true, value: 1.0, ..Default::default() };
        let mut actions = vec![("Move".to_string(), forward)];
        if sprint {
            actions.push(("Sprint".to_string(), held));
        }
        let input = InputState::from_actions(actions, glam::Vec2::ZERO);
        let mut host = FreeHost { id: crate::ObjectId::new(), pos: Vec3::ZERO, rot: Quat::IDENTITY };
        for _ in 0..120 {
            step(&mut host, &input);
        }
        let before = host.pos;
        step(&mut host, &input);
        Vec3::new(host.pos.x - before.x, 0.0, host.pos.z - before.z).length() * 60.0
    }

    #[test]
    fn holding_sprint_runs_in_first_and_third_person() {
        let mut fps = FirstPersonCharacter::default();
        let (walk, sprint) = (fps.walk_speed, fps.sprint_speed);
        assert!((settled_speed(false, |h, i| fps.step(h, i, 1.0 / 60.0, false, None)) - walk).abs() < 0.1);
        let mut fps = FirstPersonCharacter::default();
        assert!((settled_speed(true, |h, i| fps.step(h, i, 1.0 / 60.0, false, None)) - sprint).abs() < 0.1);

        let mut tps = ThirdPersonCharacter::default();
        let (walk, sprint) = (tps.walk_speed, tps.sprint_speed);
        assert!((settled_speed(false, |h, i| tps.step(h, i, 1.0 / 60.0, false, None)) - walk).abs() < 0.1);
        let mut tps = ThirdPersonCharacter::default();
        assert!((settled_speed(true, |h, i| tps.step(h, i, 1.0 / 60.0, false, None)) - sprint).abs() < 0.1);
    }

    #[test]
    fn a_networked_view_is_held_to_the_pitch_limits() {
        let input = InputState::default();
        let mut host = FreeHost { id: crate::ObjectId::new(), pos: Vec3::ZERO, rot: Quat::IDENTITY };
        let mut fps = FirstPersonCharacter::default();
        fps.step(&mut host, &input, 1.0 / 60.0, false, Some((10.0, 5000.0)));
        assert_eq!((fps.yaw, fps.pitch), (10.0, fps.max_pitch));
        fps.step(&mut host, &input, 1.0 / 60.0, false, Some((10.0, -5000.0)));
        assert_eq!(fps.pitch, fps.min_pitch);
        let mut tps = ThirdPersonCharacter::default();
        tps.step(&mut host, &input, 1.0 / 60.0, false, Some((0.0, 1e9)));
        assert_eq!(tps.cam_pitch, tps.max_pitch);
        let mut pawn = crate::Pawn::default();
        let mut zero = Vec3::ZERO;
        pawn.step(&input, 1.0 / 60.0, &mut zero, None, Some((0.0, 1e9)));
        assert_eq!(pawn.pitch, pawn.max_pitch);
    }

    #[test]
    fn angles_approach_the_short_way() {
        assert_eq!(approach_angle(170.0, -170.0, 5.0), 175.0);
        assert_eq!(approach_angle(0.0, 90.0, 30.0), 30.0);
        assert_eq!(approach_angle(10.0, 12.0, 30.0), 12.0);
    }
}
