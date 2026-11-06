//! Shared walking/falling/swimming state for desktop and VR characters.
use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use crate::{MotionState, MoveHost, RayFilter};
use super::swim::{Climb, breathe, float_accel, stroke, swims};

fn smooth(rate: f32, dt: f32) -> f32 { 1.0 - (-rate.max(0.0) * dt).exp() }

/// How a character changes direction and loses speed. Units are metres and seconds.
/// Walking and falling have separate friction and braking, as in Character Movement.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct CharacterBraking {
    pub ground_friction: f32,
    pub walking_deceleration: f32,
    pub falling_friction: f32,
    pub falling_deceleration: f32,
}
impl Default for CharacterBraking {
    fn default() -> Self {
        Self { ground_friction: 8.0, walking_deceleration: 20.48, falling_friction: 0.0, falling_deceleration: 0.0 }
    }
}
impl CharacterBraking {
    /// Integrate horizontal velocity; input magnitude controls both acceleration and top speed.
    pub fn velocity(self, velocity: Vec3, wish: Vec3, acceleration: f32, air_control: f32, grounded: bool, dt: f32) -> Vec3 {
        if !dt.is_finite() || dt <= 0.0 { return velocity; }
        let friction = if grounded { self.ground_friction } else { self.falling_friction }.max(0.0);
        let braking = if grounded { self.walking_deceleration } else { self.falling_deceleration }.max(0.0);
        let speed_limit = wish.length();
        let direction = wish.normalize_or_zero();
        let accel = acceleration.max(0.0) * if grounded { 1.0 } else { air_control.clamp(0.0, 1.0) };
        let mut v = velocity;
        // Bound integration steps for consistent braking/steering at 30–144 Hz.
        let steps = (dt.min(0.1) / (1.0 / 120.0)).ceil().max(1.0) as u32;
        let h = dt.min(0.1) / steps as f32;
        for _ in 0..steps {
            let speed = v.length();
            let over = speed > speed_limit + 1e-5;
            if speed_limit < 1e-5 || over {
                let next = (speed - (braking + friction * speed) * h).max(0.0);
                let next = if over && direction.dot(v) > 0.0 { next.max(speed_limit) } else { next };
                v = v.normalize_or_zero() * next;
            } else if accel > 0.0 {
                v = v.lerp(direction * speed, (friction * h).min(1.0));
            }
            if speed_limit > 1e-5 {
                let cap = speed_limit.max(v.length());
                v = (v + direction * accel * h).clamp_length_max(cap);
            }
        }
        v
    }
}

/// Locomotion mode, independent of the camera or input device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CharacterMovementMode { Walking, Falling, Swimming, Climbing }

/// What the controller asks of a swim this frame.
#[derive(Clone, Copy, Debug, Default)]
#[doc(hidden)]
pub struct SwimInput {
    /// Where the character looks (strokes go that way, diving when it looks
    /// down).
    pub look: Quat,
    /// Move input: x right, y forward (-1..1).
    pub input: glam::Vec2,
    /// -1 sink (crouch) .. 1 rise (jump).
    pub rise: f32,
    pub sprint: bool,
}

/// One frame of a controller's movement for [`CharacterMovement::step`].
pub(super) struct Stride {
    /// Horizontal velocity it wants (m/s).
    pub wish: Vec3,
    pub accel: f32,
    pub air_control: f32,
    pub braking: CharacterBraking,
    /// Take-off speed when jumping this frame.
    pub jump: Option<f32>,
    pub gravity: f32,
    pub swim: SwimInput,
}

/// Eye height as a share of the character's height.
const EYE_SHARE: f32 = 0.92;

/// Shared walking state: horizontal acceleration, gravity, jumping, and either
/// the physics character controller or a flat floor at the start height. In
/// water deeper than its chest it swims instead ([`super::Swimmer`]).
#[derive(Default)]
#[doc(hidden)]
pub struct CharacterMovement {
    pub(super) vel: Vec3,
    pub(super) grounded: bool,
    pub(super) floor_y: f32,
    pub(super) swimming: bool,
    /// Seconds of breath left (None = full).
    pub(super) breath: Option<f32>,
    pub(super) climb: Option<Climb>,
}

impl CharacterMovement {
    pub fn mode(&self) -> CharacterMovementMode {
        if self.climb.is_some() { CharacterMovementMode::Climbing }
        else if self.swimming { CharacterMovementMode::Swimming }
        else if self.grounded { CharacterMovementMode::Walking }
        else { CharacterMovementMode::Falling }
    }

    pub fn is_grounded(&self) -> bool {
        self.grounded
    }
    /// World-space velocity the walker is moving at (what it asked the
    /// character to move by, per second).
    pub fn velocity(&self) -> Vec3 {
        self.vel
    }
    /// Swimming (or climbing out of the water) this frame.
    pub fn swimming(&self) -> bool {
        self.swimming || self.climb.is_some()
    }

    /// Seconds of breath left, while it has been under at all.
    pub fn breath(&self) -> Option<f32> {
        self.breath
    }

    pub(super) fn save(&self, m: &mut MotionState) {
        m.vel = self.vel;
        m.grounded = self.grounded;
        m.floor_y = self.floor_y;
        m.swimming = self.swimming;
        m.breath = self.breath;
        m.climb = self.climb;
    }

    pub(super) fn load(&mut self, m: &MotionState) {
        *self = Self { vel: m.vel, grounded: m.grounded, floor_y: m.floor_y, swimming: m.swimming, breath: m.breath, climb: m.climb };
    }

    /// Swim instead of walk when the water is deep enough: returns true when
    /// it did (or is climbing out), so the walk is skipped.
    fn swim(&mut self, host: &mut dyn MoveHost, swim: SwimInput, dt: f32) -> bool {
        let s = host.swimmer();
        let (height, radius) = host.capsule().map_or((1.8, 0.35), |(r, h, _)| (h, r));
        if let Some(c) = &mut self.climb {
            c.t += dt / s.climb_time.max(0.05);
            host.set_translation(c.at(c.t));
            if c.t >= 1.0 {
                self.climb = None;
                self.vel = Vec3::ZERO;
                self.grounded = true;
            }
            return true;
        }
        let feet = host.translation();
        let surface = host.water_surface(feet + Vec3::Y * height);
        let was = self.swimming;
        self.swimming = surface.is_some_and(|h| swims(&s, self.swimming, h - feet.y, height));
        if self.swimming != was {
            host.set_animator_bool("Swimming", self.swimming);
        }
        let eye = feet.y + height * EYE_SHARE;
        let breath = self.breath.unwrap_or(s.breath);
        let Some(h) = surface.filter(|_| self.swimming) else {
            self.breath = (breath < s.breath).then(|| breathe(&s, breath, false, dt)).filter(|b| *b < s.breath);
            return false;
        };
        self.breath = Some(breathe(&s, breath, eye < h, dt));
        let speed = if swim.sprint { s.sprint_speed } else { s.swim_speed };
        let target = stroke(swim.look, swim.input, swim.rise, speed);
        let rate = if target.length_squared() > 1e-4 { s.stroke_accel } else { s.water_drag };
        self.vel = self.vel.lerp(target, smooth(rate, dt));
        // Float to the surface unless stroking up or down.
        if swim.rise.abs() < 0.1 && target.y > -0.3 * speed {
            self.vel.y += float_accel(&s, eye, h, self.vel.y) * dt;
        }
        // Swimming into an edge near the surface: climb out over it.
        if eye > h - 0.3 && swim.input.y > 0.3 {
            let fwd = swim.look * Vec3::NEG_Z;
            let dir = Vec3::new(fwd.x, 0.0, fwd.z).normalize_or_zero();
            if let Some(to) = ledge(host, feet, h, dir, radius, s.climb_height) {
                self.climb = Some(Climb { from: feet, to, t: 0.0 });
                self.vel = Vec3::ZERO;
                return true;
            }
        }
        self.grounded = false;
        if host.has_character() {
            host.move_character(self.vel * dt);
        } else {
            host.set_translation(feet + self.vel * dt);
        }
        true
    }

    pub(super) fn step(&mut self, host: &mut dyn MoveHost, m: Stride, dt: f32) {
        if self.swim(host, m.swim, dt) {
            return;
        }
        self.walk(host, m, dt);
    }

    /// Ground/falling movement, also used by VR after its tracked-hand swimming.
    pub(super) fn walk(&mut self, host: &mut dyn MoveHost, m: Stride, dt: f32) {
        let Stride { wish, accel, air_control, braking, jump: jump_speed, gravity, .. } = m;
        let physical = host.has_character();
        if physical {
            // An upward takeoff must not be cancelled by last frame's floor contact.
            self.grounded = host.grounded() && self.vel.y <= 0.0;
        }
        let horizontal = braking.velocity(self.vel.with_y(0.0), wish, accel, air_control, self.grounded, dt);
        self.vel.x = horizontal.x;
        self.vel.z = horizontal.z;
        if self.grounded && self.vel.y < 0.0 {
            self.vel.y = 0.0;
        }
        if let Some(v) = jump_speed
            && self.grounded
        {
            self.vel.y = v;
            self.grounded = false;
        }
        let old_y = self.vel.y;
        self.vel.y -= gravity.max(0.0) * dt;
        // Integrate gravity using average velocity so jump height is stable across frame rates.
        let delta = Vec3::new(self.vel.x, (old_y + self.vel.y) * 0.5, self.vel.z) * dt;
        if physical {
            host.move_character(delta);
        } else {
            let mut pos = host.translation() + delta;
            if pos.y <= self.floor_y {
                pos.y = self.floor_y;
                self.vel.y = 0.0;
                self.grounded = true;
            }
            host.set_translation(pos);
        }
    }
}

/// An edge to climb out onto: a wall `dir` of a swimmer at `feet` (surface
/// `surface`) whose top is no more than `climb` above the water. Where the
/// feet end up, a body width onto it.
fn ledge(host: &dyn MoveHost, feet: Vec3, surface: f32, dir: Vec3, radius: f32, climb: f32) -> Option<Vec3> {
    if dir.length_squared() < 0.5 {
        return None;
    }
    let filter = RayFilter { exclude: Some(host.self_id()), ..Default::default() };
    let from = Vec3::new(feet.x, surface + 0.05, feet.z);
    let wall = host.raycast(from, dir, radius + 0.5, &filter)?;
    if wall.normal.y.abs() > 0.5 {
        return None;
    }
    let above = wall.point + dir * (radius + 0.05) + Vec3::Y * (climb + 0.1);
    let top = host.raycast(above, Vec3::NEG_Y, climb + 0.4, &filter)?;
    (top.normal.y > 0.7 && top.point.y - surface <= climb && top.point.y > surface - 0.1).then_some(top.point + Vec3::Y * 0.02)
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn walking_accelerates_in_metres_per_second_squared() {
        let tune = CharacterBraking { ground_friction: 0.0, ..Default::default() };
        let v = tune.velocity(Vec3::ZERO, Vec3::X * 6.0, 20.0, 0.3, true, 0.1);
        assert!((v.x - 2.0).abs() < 1e-5);
        let v = tune.velocity(Vec3::X * 5.9, Vec3::X * 6.0, 20.0, 0.3, true, 0.1);
        assert!((v.x - 6.0).abs() < 1e-5);
    }
    #[test]
    fn releasing_input_brakes_on_ground_but_preserves_air_momentum() {
        let tune = CharacterBraking::default();
        let airborne = tune.velocity(Vec3::X * 5.0, Vec3::ZERO, 20.0, 0.3, false, 0.1);
        assert_eq!(airborne, Vec3::X * 5.0);
        let mut walking = airborne;
        for _ in 0..10 { walking = tune.velocity(walking, Vec3::ZERO, 20.0, 0.3, true, 0.1); }
        assert_eq!(walking, Vec3::ZERO);
    }
    #[test]
    fn air_control_scales_acceleration_and_never_turns_with_zero_control() {
        let tune = CharacterBraking::default();
        assert_eq!(tune.velocity(Vec3::X * 5.0, Vec3::Z * 5.0, 20.0, 0.0, false, 0.1), Vec3::X * 5.0);
        let v = tune.velocity(Vec3::ZERO, Vec3::X * 5.0, 20.0, 0.25, false, 0.1);
        assert!((v.x - 0.5).abs() < 1e-5);
    }
    #[test]
    fn braking_is_consistent_across_frame_rates() {
        let end = |hz: u32| {
            let mut v = Vec3::X * 6.0;
            for _ in 0..hz / 10 { v = CharacterBraking::default().velocity(v, Vec3::ZERO, 20.0, 0.3, true, 1.0 / hz as f32); }
            v.x
        };
        assert!((end(30) - end(120)).abs() < 0.02);
    }
}
