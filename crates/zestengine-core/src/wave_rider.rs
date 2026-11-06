//! Objects that ride the water without a physics body: a ball bobbing on the
//! swell, a buoy, a fish near the surface. The ride is a pure function of the
//! object's anchor (its synced transform), the water's settings and the water
//! clock, so every networked client computes the same motion locally and only
//! the calm anchor travels over the network. The waves move the rendered
//! object; drift from steep waves moves the anchor (on the owner only).

use glam::{Mat4, Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::{TypedComponent, Water};

/// How a WaveRider moves with the water.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RideMode {
    /// Sits on the surface at its draft whenever its anchor is within reach
    /// of it (a ball, a buoy, a boat).
    #[default]
    Float,
    /// Keeps its own depth; the water's orbital motion carries it, fading
    /// with depth (a fish, weed, a suspended object).
    Suspended,
}

impl RideMode {
    /// Every mode, in declaration order.
    pub const ALL: [RideMode; 2] = [Self::Float, Self::Suspended];
    /// Inspector label of this mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::Float => "Float",
            Self::Suspended => "Suspended",
        }
    }
}

/// Rides the water without a physics body (see the module docs).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct WaveRider {
    /// How the rider moves with the water.
    pub mode: RideMode,
    /// How deep it sits below the surface (m) when floating.
    pub draft: f32,
    /// Floating: snaps to the surface when its anchor is within this
    /// distance of it (m).
    pub reach: f32,
    /// Footprint across (m): waves much shorter than it barely move it.
    pub size: f32,
    /// 0 = light as a cork (rides every crest and swings with the orbit),
    /// 1 = heavy (bobs on the average swell, hardly swings).
    pub weight: f32,
    /// How much it tilts with the surface (0..1).
    pub tilt: f32,
    /// Steep waves (amplitude x wavenumber above this) push it along their
    /// travel direction, like surf; 0 disables.
    pub drift_steepness: f32,
    /// Scales that push.
    pub drift: f32,
    /// Engine-filled each frame.
    #[serde(skip)]
    pub state: RideState,
}

impl Default for WaveRider {
    fn default() -> Self {
        Self {
            mode: RideMode::Float,
            draft: 0.1,
            reach: 1.0,
            size: 0.5,
            weight: 0.3,
            tilt: 0.7,
            drift_steepness: 0.12,
            drift: 1.0,
            state: RideState::default(),
        }
    }
}

impl TypedComponent for WaveRider {
    const NAME: &'static str = "Wave Rider";
}

/// Where the water puts a rider this frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RideState {
    /// On (floating at) the surface, or within the water's motion.
    pub on_surface: bool,
    /// In a water body at all.
    pub in_water: bool,
    /// Rendered position minus anchor (world).
    pub offset: Vec3,
    /// Extra rotation from the surface tilt (world).
    pub tilt: Quat,
    /// Depth of the anchor below the moving surface (m; negative = above).
    pub depth: f32,
    /// Surface height above the rider (world).
    pub surface: f32,
    /// Anchor drift the waves push (m/s, world), applied by the owner.
    pub drift: Vec3,
}

impl WaveRider {
    /// The ride for an object anchored at `anchor` (world) over `water`.
    pub fn ride(&self, water: &Water, water_world: Mat4, anchor: Vec3, t: f32) -> Option<RideState> {
        let to_local = water_world.inverse();
        let a = to_local.transform_point3(anchor);
        let xz = Vec2::new(a.x, a.z);
        let col = water.column(xz)?;
        if a.y < col.surface - col.depth - 0.5 {
            return None;
        }
        let (rest_surface, _) = water.surface(xz, t);
        let floating = self.mode == RideMode::Float && (rest_surface - a.y).abs() <= self.reach.max(0.0) + self.draft.max(0.0);
        // Depth the orbital motion is felt at: the draft when floating.
        let depth = if floating { self.draft.max(0.0) } else { (col.surface - a.y).max(0.0) };
        let weight = self.weight.clamp(0.0, 1.0);
        let mut d = Vec3::ZERO;
        let mut slope = Vec2::ZERO;
        let mut drift = Vec2::ZERO;
        for w in water.resolved_waves() {
            // Waves short against the footprint average out under it.
            let footprint = 1.0 / (1.0 + (w.k * self.size.max(0.0) * 0.5).powi(2));
            let decay = (-w.k * depth).exp();
            let gain = footprint * decay;
            let phase = w.k * w.dir.dot(xz) - w.omega * t;
            let (s, c) = phase.sin_cos();
            let swing = gain * (1.0 - weight * 0.8);
            d.x += w.qa * w.dir.x * c * swing;
            d.z += w.qa * w.dir.y * c * swing;
            d.y += w.amplitude * s * gain;
            slope += w.dir * (w.k * w.amplitude * c * gain);
            let steep = w.k * w.amplitude;
            if self.drift_steepness > 0.0 && steep > self.drift_steepness {
                // Stokes drift, faded in past the threshold, lighter goes further.
                let u = w.omega * w.k * w.amplitude * w.amplitude * (-2.0 * w.k * depth).exp();
                let over = ((steep - self.drift_steepness) / self.drift_steepness).min(1.0);
                drift += w.dir * u * over * (1.0 - weight * 0.9) * footprint;
            }
        }
        if let Some(r) = &water.ripples {
            d.y += r.height(xz) * (-depth * 2.0).exp();
        }
        let surface_local = rest_surface;
        let target = if floating {
            Vec3::new(a.x + d.x, col.surface + d.y - self.draft, a.z + d.z)
        } else {
            a + d
        };
        let world_pos = water_world.transform_point3(target);
        let n_local = Vec3::new(-slope.x, 1.0, -slope.y).normalize_or(Vec3::Y);
        let n = water_world.transform_vector3(n_local).normalize_or(Vec3::Y);
        let tilt = Quat::from_rotation_arc(Vec3::Y, Vec3::Y.lerp(n, self.tilt.clamp(0.0, 1.0)).normalize_or(Vec3::Y));
        let in_motion = floating || depth < 1.0 / water.resolved_waves().iter().map(|w| w.k).fold(f32::MAX, f32::min).max(1e-3);
        Some(RideState {
            on_surface: floating || (in_motion && depth < 0.5),
            in_water: true,
            offset: world_pos - anchor,
            tilt: if floating || in_motion { tilt } else { Quat::IDENTITY },
            depth: surface_local - a.y,
            surface: water_world.transform_point3(Vec3::new(a.x, surface_local, a.z)).y,
            drift: water_world.transform_vector3(Vec3::new(drift.x, 0.0, drift.y)) * self.drift.max(0.0) + water_world.transform_vector3(col.flow) * if floating { 1.0 } else { 0.0 },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GerstnerWave;

    fn swell() -> Water {
        Water { waves: vec![GerstnerWave { direction: 0.0, wavelength: 10.0, amplitude: 0.4, steepness: 0.5, speed: 1.0 }], ..Default::default() }
    }

    #[test]
    fn a_float_rides_the_crest_and_everyone_agrees() {
        let w = swell();
        let r = WaveRider { draft: 0.0, size: 0.1, weight: 0.0, ..Default::default() };
        let anchor = Vec3::new(1.0, 0.3, 2.0);
        let mut seen = (f32::MAX, f32::MIN);
        for k in 0..200 {
            let t = k as f32 * 0.05;
            let s = r.ride(&w, Mat4::IDENTITY, anchor, t).unwrap();
            assert!(s.on_surface);
            let y = anchor.y + s.offset.y;
            seen = (seen.0.min(y), seen.1.max(y));
            // Same inputs, same answer: what each client computes.
            assert_eq!(s, r.ride(&w, Mat4::IDENTITY, anchor, t).unwrap());
        }
        assert!(seen.0 < -0.3 && seen.1 > 0.3, "bobs through the full swell: {seen:?}");
    }

    #[test]
    fn heavy_and_wide_things_move_less_and_deep_fish_barely_feel_it() {
        let w = swell();
        let span = |r: &WaveRider, anchor: Vec3| {
            let ys: Vec<f32> = (0..200).map(|k| r.ride(&w, Mat4::IDENTITY, anchor, k as f32 * 0.05).unwrap().offset.y).collect();
            ys.iter().cloned().fold(f32::MIN, f32::max) - ys.iter().cloned().fold(f32::MAX, f32::min)
        };
        let cork = span(&WaveRider { size: 0.1, ..Default::default() }, Vec3::ZERO);
        let raft = span(&WaveRider { size: 12.0, ..Default::default() }, Vec3::ZERO);
        assert!(raft < cork * 0.5, "a raft longer than the waves averages them: {raft} vs {cork}");
        let fish = WaveRider { mode: RideMode::Suspended, ..Default::default() };
        let shallow = span(&fish, Vec3::new(0.0, -0.3, 0.0));
        let deep = span(&fish, Vec3::new(0.0, -3.0, 0.0));
        assert!(deep < shallow * 0.2, "orbits fade with depth: {deep} vs {shallow}");
    }

    #[test]
    fn steep_waves_push_light_things_along() {
        let steep = Water { waves: vec![GerstnerWave { direction: 0.0, wavelength: 6.0, amplitude: 0.5, steepness: 0.8, speed: 1.0 }], ..Default::default() };
        let light = WaveRider { weight: 0.0, size: 0.1, ..Default::default() }.ride(&steep, Mat4::IDENTITY, Vec3::ZERO, 0.0).unwrap();
        let heavy = WaveRider { weight: 1.0, size: 0.1, ..Default::default() }.ride(&steep, Mat4::IDENTITY, Vec3::ZERO, 0.0).unwrap();
        assert!(light.drift.x > 0.05, "pushed along the wave's travel: {}", light.drift);
        assert!(heavy.drift.x < light.drift.x * 0.2);
        let calm = WaveRider { drift_steepness: 0.9, ..Default::default() }.ride(&steep, Mat4::IDENTITY, Vec3::ZERO, 0.0).unwrap();
        assert_eq!(calm.drift.x, 0.0, "below the threshold nothing drifts");
    }
}
