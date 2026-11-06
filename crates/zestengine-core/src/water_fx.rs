//! Falling and swirling water: waterfalls (a sheet poured off a lip that
//! lands in a plunge pool), whirlpools, and the splash events bodies make
//! when they hit the water. The engine turns these into spray, mist, foam
//! and ripple particles, and feeds vortex flow into floating bodies.

use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

use crate::TypedComponent;

/// Standard gravity (m/s^2).
pub const GRAVITY: f32 = 9.81;

/// Something hit the water (world space). The engine queues these from
/// physics and turns them into spray, foam and rings.
#[derive(Clone, Copy, Debug)]
pub struct WaterSplash {
    /// Where the splash is (world space).
    pub at: Vec3,
    /// Velocity of the thing hitting the water.
    pub velocity: Vec3,
    /// Rough radius of what hit (m).
    pub size: f32,
}

/// A swirl in the water (world space).
#[derive(Clone, Copy, Debug)]
pub struct Vortex {
    /// Centre of the swirl (world space).
    pub center: Vec3,
    /// Swirl radius (m).
    pub radius: f32,
    /// Peak tangential speed (m/s), reached halfway out; negative turns
    /// clockwise seen from above.
    pub swirl: f32,
    /// Inward speed at the centre (m/s), fading to 0 at the rim.
    pub pull: f32,
}

impl Vortex {
    /// Horizontal flow the vortex adds at `p`.
    pub fn flow(&self, p: Vec3) -> Vec3 {
        let d = Vec3::new(p.x - self.center.x, 0.0, p.z - self.center.z);
        let r = d.length();
        if r >= self.radius || r < 1e-4 {
            return Vec3::ZERO;
        }
        let x = r / self.radius;
        let tangent = Vec3::new(-d.z, 0.0, d.x) / r;
        let inward = -d / r;
        tangent * (self.swirl * 4.0 * x * (1.0 - x)) + inward * (self.pull * (1.0 - x))
    }
}

/// Horizontal flow the vortices add at `p`.
pub fn vortex_flow(vortices: &[Vortex], p: Vec3) -> Vec3 {
    vortices.iter().map(|v| v.flow(p)).sum()
}

/// A waterfall poured off the line between `lip[0]` and `lip[1]` (local
/// space), leaving along the object's local +Z.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Waterfall {
    /// The lip line the water leaves from (local space).
    pub lip: [[f32; 3]; 2],
    /// Speed the water leaves the lip (m/s).
    pub flow_speed: f32,
    /// How dense the falling sheet is (0..1).
    pub volume: f32,
    /// Sideways spread of the sheet as it falls (m/s).
    pub spread: f32,
    /// Fall height when nothing below catches it (m).
    pub max_drop: f32,
    /// Spray and droplets at the plunge (multiplier).
    pub splash: f32,
    /// Mist rising from the plunge (multiplier).
    pub mist: f32,
    /// Foam floating away from the plunge (multiplier).
    pub foam: f32,
    /// Plunge pool swirl: peak speed (m/s, negative = clockwise) and radius
    /// as a multiple of the lip width.
    pub swirl: f32,
    /// Plunge pool radius as a multiple of the lip width.
    pub swirl_radius: f32,
    /// Linear RGB tint of the water and spray.
    pub color: [f32; 3],
    /// Brightness of the water and spray (multiplier).
    pub brightness: f32,
}

impl Default for Waterfall {
    fn default() -> Self {
        Self {
            lip: [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            flow_speed: 1.5,
            volume: 0.6,
            spread: 0.15,
            max_drop: 6.0,
            splash: 1.0,
            mist: 1.0,
            foam: 1.0,
            swirl: 0.8,
            swirl_radius: 1.5,
            color: [0.82, 0.9, 0.95],
            brightness: 0.85,
        }
    }
}

impl TypedComponent for Waterfall {
    const NAME: &'static str = "Waterfall";
}

impl Waterfall {
    /// World start point, exit velocity and lip width for lip fraction `u`.
    pub fn emit(&self, world: Mat4, u: f32) -> (Vec3, Vec3) {
        let a = world.transform_point3(Vec3::from(self.lip[0]));
        let b = world.transform_point3(Vec3::from(self.lip[1]));
        let out = world.transform_vector3(Vec3::Z);
        let out = Vec3::new(out.x, 0.0, out.z).normalize_or(Vec3::Z);
        (a.lerp(b, u), out * self.flow_speed)
    }

    /// World width of the lip line (m).
    pub fn width(&self, world: Mat4) -> f32 {
        world.transform_point3(Vec3::from(self.lip[0])).distance(world.transform_point3(Vec3::from(self.lip[1])))
    }

    /// Where the middle of the sheet lands, given the height of whatever
    /// catches it at a point (None = falls the full `max_drop`).
    pub fn plunge(&self, world: Mat4, catch: impl Fn(Vec3) -> Option<f32>) -> Vec3 {
        let (start, v) = self.emit(world, 0.5);
        let mut floor = start.y - self.max_drop.max(0.1);
        let mut at = start;
        // Twice: the landing spot moves with the height that catches it.
        for _ in 0..3 {
            let t = fall_time(start.y - floor);
            at = Vec3::new(start.x + v.x * t, floor, start.z + v.z * t);
            if let Some(h) = catch(Vec3::new(at.x, start.y, at.z)) {
                floor = h.clamp(start.y - self.max_drop.max(0.1), start.y);
            }
        }
        at
    }

    /// The falling sheet's centre line from the lip to `plunge_y`, for gizmos.
    pub fn arc(&self, world: Mat4, u: f32, plunge_y: f32, steps: usize) -> Vec<Vec3> {
        let (start, v) = self.emit(world, u);
        let t_end = fall_time((start.y - plunge_y).max(0.0));
        (0..=steps)
            .map(|k| {
                let t = t_end * k as f32 / steps.max(1) as f32;
                start + v * t - Vec3::Y * (0.5 * GRAVITY * t * t)
            })
            .collect()
    }

    /// The plunge pool's swirl.
    pub fn vortex(&self, world: Mat4, plunge: Vec3) -> Vortex {
        Vortex { center: plunge, radius: (self.width(world) * self.swirl_radius).max(0.5), swirl: self.swirl, pull: self.swirl.abs() * 0.15 }
    }
}

/// Seconds to fall `h` metres from rest.
pub fn fall_time(h: f32) -> f32 {
    (2.0 * h.max(0.0) / GRAVITY).sqrt()
}

/// A standalone swirl in the water around the object.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Whirlpool {
    /// Swirl radius (m).
    pub radius: f32,
    /// Peak swirl speed (m/s).
    pub strength: f32,
    /// Inward pull at the centre (m/s).
    pub pull: f32,
    /// Clockwise (seen from above) instead of counter-clockwise.
    pub clockwise: bool,
    /// Foam drawn into the swirl (multiplier).
    pub foam: f32,
}

impl Default for Whirlpool {
    fn default() -> Self {
        Self { radius: 4.0, strength: 2.0, pull: 0.5, clockwise: false, foam: 1.0 }
    }
}

impl TypedComponent for Whirlpool {
    const NAME: &'static str = "Whirlpool";
}

impl Whirlpool {
    /// The swirl this whirlpool produces.
    pub fn vortex(&self, world: Mat4) -> Vortex {
        let s = if self.clockwise { -1.0 } else { 1.0 };
        Vortex { center: world.w_axis.truncate(), radius: self.radius.max(0.1), swirl: self.strength * s, pull: self.pull }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vortex_swirls_and_pulls_inside_its_radius_only() {
        let v = Vortex { center: Vec3::ZERO, radius: 4.0, swirl: 2.0, pull: 0.5 };
        let f = v.flow(Vec3::new(2.0, 0.0, 0.0));
        assert!((f.z - 2.0).abs() < 1e-4, "peak swirl halfway out, counter-clockwise: {f}");
        assert!(f.x < 0.0, "pulled inward: {f}");
        assert_eq!(v.flow(Vec3::new(5.0, 0.0, 0.0)), Vec3::ZERO);
    }

    #[test]
    fn plunge_lands_where_the_arc_meets_the_catch() {
        let fall = Waterfall { flow_speed: 2.0, max_drop: 20.0, ..Default::default() };
        let world = Mat4::from_translation(Vec3::new(0.0, 5.0, 0.0));
        let p = fall.plunge(world, |_| Some(0.0));
        let t = fall_time(5.0);
        assert!((p.y - 0.0).abs() < 1e-4 && (p.z - 2.0 * t).abs() < 1e-3, "{p}");
        let free = fall.plunge(world, |_| None);
        assert!((free.y - (5.0 - 20.0)).abs() < 1e-4);
        let arc = fall.arc(world, 0.5, 0.0, 8);
        assert!(arc.last().unwrap().distance(p) < 1e-3);
    }
}
