//! Destructible objects: health and deterministic debris scatter when an
//! object breaks.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

/// A destructible object's health.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Destructible {
    /// Current health.
    pub health: f32,
    /// Health when intact.
    pub max_health: f32,
}

impl Default for Destructible {
    fn default() -> Self {
        Self { health: 100.0, max_health: 100.0 }
    }
}

impl Destructible {
    /// Apply damage; returns true when the object breaks (health reaches zero).
    pub fn damage(&mut self, amount: f32) -> bool {
        self.health = (self.health - amount.max(0.0)).max(0.0);
        self.health <= 0.0
    }
}

/// One debris piece after an object breaks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Debris {
    /// Launch velocity (m/s).
    pub velocity: Vec3,
    /// Spin orientation.
    pub rotation: Quat,
}

fn hash01(seed: u32, i: usize) -> f32 {
    let mut n = seed.wrapping_add((i as u32).wrapping_mul(0x9E37_79B9));
    n = (n ^ (n >> 16)).wrapping_mul(0x7FEB_352D);
    n ^= n >> 15;
    (n & 0xF_FFFF) as f32 / 0xF_FFFF as f32
}

fn unit_dir(seed: u32, i: usize) -> Vec3 {
    let u = hash01(seed, i) * 2.0 - 1.0;
    let phi = hash01(seed, i.wrapping_add(0x4000_0000)) * std::f32::consts::TAU;
    let s = (1.0 - u * u).max(0.0).sqrt();
    Vec3::new(s * phi.cos(), u, s * phi.sin())
}

/// Deterministically scatter `count` debris pieces: an isotropic direction with
/// a speed in `[0.5, 1.0] × strength`, and a random spin, seeded for replay.
pub fn scatter_debris(strength: f32, count: usize, seed: u32) -> Vec<Debris> {
    (0..count)
        .map(|i| {
            let speed = strength * (0.5 + 0.5 * hash01(seed, i.wrapping_add(0x2000_0000)));
            let spin = hash01(seed, i.wrapping_add(0x6000_0000)) * std::f32::consts::TAU;
            Debris { velocity: unit_dir(seed, i) * speed, rotation: Quat::from_rotation_x(spin) }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn destructible_breaks_when_health_runs_out() {
        let mut d = Destructible::default();
        assert!(!d.damage(50.0));
        assert_eq!(d.health, 50.0);
        assert!(d.damage(60.0));
        assert_eq!(d.health, 0.0);
    }

    #[test]
    fn debris_is_deterministic_and_bounded() {
        let a = scatter_debris(10.0, 8, 42);
        let b = scatter_debris(10.0, 8, 42);
        assert_eq!(a.len(), 8);
        assert_eq!(a, b, "same seed must scatter identically");
        for d in &a {
            assert!(d.velocity.length() <= 10.0 + 1e-4);
            assert!(d.velocity.length() >= 5.0 - 1e-4);
        }
    }
}
