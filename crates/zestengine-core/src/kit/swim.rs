//! Swimming for the walking controllers: once the water is deeper than the
//! character's chest it swims instead of walks. It floats with its head just
//! out of the water, strokes along where it looks (diving when looking down,
//! rising with jump, sinking with crouch), runs out of breath under water,
//! and climbs out over an edge it swims into. A [`Swimmer`] on the same
//! object tunes it; without one the defaults apply.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::TypedComponent;

/// How a character swims.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Swimmer {
    /// Water this deep, as a share of the character's height, makes it swim
    /// (it wades below that).
    pub swim_depth: f32,
    /// Swimming speed (m/s).
    pub swim_speed: f32,
    /// Sprint swimming speed (m/s).
    pub sprint_speed: f32,
    /// How quickly strokes reach their speed (1/s).
    pub stroke_accel: f32,
    /// How quickly it slows when it stops stroking (1/s).
    pub water_drag: f32,
    /// How high the eyes float above the surface at rest (m).
    pub float_height: f32,
    /// Seconds it can stay with its head under.
    pub breath: f32,
    /// Highest edge above the surface it climbs out over (m).
    pub climb_height: f32,
    /// Seconds a climb out takes.
    pub climb_time: f32,
}

impl Default for Swimmer {
    fn default() -> Self {
        Self {
            swim_depth: 0.7,
            swim_speed: 2.0,
            sprint_speed: 3.5,
            stroke_accel: 3.0,
            water_drag: 1.5,
            float_height: 0.12,
            breath: 30.0,
            climb_height: 0.9,
            climb_time: 0.6,
        }
    }
}

impl TypedComponent for Swimmer {
    const NAME: &'static str = "Swimmer";
}

/// Whether to swim with the surface `depth` metres above the feet of a
/// character `height` tall: in past `swim_depth`, out again only a little
/// shallower so it doesn't flicker at the threshold.
pub fn swims(s: &Swimmer, swimming: bool, depth: f32, height: f32) -> bool {
    let enter = s.swim_depth.clamp(0.2, 1.5) * height;
    if swimming { depth > enter - 0.15 * height } else { depth > enter }
}

/// The stroke velocity: `input` (x right, y forward, -1..1) along `look`
/// (diving when it looks down), plus `rise` (-1 sink .. 1 rise), at
/// `speed`.
pub fn stroke(look: Quat, input: glam::Vec2, rise: f32, speed: f32) -> Vec3 {
    let wish = look * Vec3::X * input.x + look * Vec3::NEG_Z * input.y + Vec3::Y * rise;
    wish.clamp_length_max(1.0) * speed
}

/// Vertical acceleration that floats the eyes `float_height` above the
/// surface: a damped spring on the eye height, by the character's buoyancy
/// (none once diving takes it under).
pub fn float_accel(s: &Swimmer, eye_y: f32, surface: f32, vel_y: f32) -> f32 {
    let error = surface + s.float_height - eye_y;
    // Critically damped at about 1.5 Hz: settles without bobbing for long.
    let w = std::f32::consts::TAU * 1.5;
    w * w * error.clamp(-1.0, 1.0) - 2.0 * w * vel_y
}

/// A climb out in progress: from `from` to `to` (feet, world) over the
/// swimmer's `climb_time`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Climb {
    /// Climb start (feet, world).
    pub from: Vec3,
    /// Climb end (feet, world).
    pub to: Vec3,
    /// Progress 0..1 through the climb.
    pub t: f32,
}

impl Climb {
    /// Feet position at progress `t` (0..1): up over the edge first, then in.
    pub fn at(&self, t: f32) -> Vec3 {
        let t = t.clamp(0.0, 1.0);
        let up = (t / 0.6).min(1.0);
        let over = ((t - 0.4) / 0.6).clamp(0.0, 1.0);
        let ease = |x: f32| x * x * (3.0 - 2.0 * x);
        Vec3::new(
            self.from.x + (self.to.x - self.from.x) * ease(over),
            self.from.y + (self.to.y - self.from.y) * ease(up),
            self.from.z + (self.to.z - self.from.z) * ease(over),
        )
    }
}

/// Breath left after `dt` with the head under (`under`) or out: it runs out
/// at one second per second and comes back three times as fast.
pub fn breathe(s: &Swimmer, breath: f32, under: bool, dt: f32) -> f32 {
    if under { (breath - dt).max(0.0) } else { (breath + 3.0 * dt).min(s.breath.max(0.0)) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swims_past_chest_depth_with_some_hysteresis() {
        let s = Swimmer::default();
        assert!(!swims(&s, false, 1.0, 1.8), "waist deep: wades");
        assert!(swims(&s, false, 1.4, 1.8), "chest deep: swims");
        assert!(swims(&s, true, 1.2, 1.8), "keeps swimming just under the threshold");
        assert!(!swims(&s, true, 0.8, 1.8), "stands up in the shallows");
    }

    #[test]
    fn strokes_dive_where_it_looks_and_rise_on_jump() {
        let down = Quat::from_rotation_x(-0.6);
        let v = stroke(down, glam::Vec2::new(0.0, 1.0), 0.0, 2.0);
        assert!(v.y < -0.5 && v.z < -1.0 && (v.length() - 2.0).abs() < 1e-4, "{v}");
        let up = stroke(Quat::IDENTITY, glam::Vec2::ZERO, 1.0, 2.0);
        assert!((up - Vec3::Y * 2.0).length() < 1e-4);
    }

    #[test]
    fn floats_with_the_eyes_just_out_of_the_water() {
        let s = Swimmer::default();
        let (mut eye, mut v) = (-1.0f32, 0.0f32);
        for _ in 0..600 {
            v += float_accel(&s, eye, 0.0, v) / 120.0;
            eye += v / 120.0;
        }
        assert!((eye - s.float_height).abs() < 0.01 && v.abs() < 0.01, "{eye} {v}");
    }

    #[test]
    fn breath_runs_out_under_water_and_comes_back() {
        let s = Swimmer { breath: 10.0, ..Default::default() };
        let mut b = s.breath;
        for _ in 0..12 {
            b = breathe(&s, b, true, 1.0);
        }
        assert_eq!(b, 0.0);
        b = breathe(&s, b, false, 1.0);
        assert!((b - 3.0).abs() < 1e-5);
        for _ in 0..10 {
            b = breathe(&s, b, false, 1.0);
        }
        assert_eq!(b, s.breath);
    }

    #[test]
    fn a_climb_goes_up_then_over() {
        let c = Climb { from: Vec3::ZERO, to: Vec3::new(0.0, 1.0, -1.0), t: 0.0 };
        let mid = c.at(0.4);
        assert!(mid.y > 0.6 && mid.z.abs() < 1e-4, "up first: {mid}");
        assert!((c.at(1.0) - c.to).length() < 1e-5);
    }
}
