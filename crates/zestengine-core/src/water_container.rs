//! Water Container: something that holds water (a bucket, a trough, a boat's
//! hull). Its inner volume is a local box whose top face is the opening. In
//! Play the engine keeps the water level, pours what no longer fits over the
//! rim as a stream when it's tipped or swung, fills it from water it's dipped
//! into or that falls in, and adds the water's weight to its rigid body.
//!
//! Geometry works in the container's "box frame": origin at the box centre,
//! axes along the object's local axes, units in metres (the object's scale
//! applied). The water is everything in the box below the plane
//! `up . p = level`, where `up` is the free surface's normal in that frame.

use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

use crate::TypedComponent;

/// Litres in one cubic metre.
pub const LITRES_PER_M3: f32 = 1000.0;

/// Columns per side the volume integral samples.
const SLICE_N: usize = 16;

/// Holds water. See the module docs.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct WaterContainer {
    /// Inner volume (object space, before scale): centre and half extents.
    /// The box's +Y face is the opening.
    pub center: [f32; 3],
    /// Half extents of the inner volume (object space).
    pub half_extents: [f32; 3],
    /// Water it starts with, as a share of its capacity.
    pub initial_fill: f32,
    /// Scales how fast water pours over the rim.
    pub pour_rate: f32,
    /// Scales how fast water flows in through a submerged opening.
    pub inflow: f32,
    /// How freely the water sloshes when the container moves: 0 settles at
    /// once, 1 rocks on for a long time.
    pub slosh: f32,
    /// Linear RGB of the water inside and of what pours out.
    pub color: [f32; 3],
    /// Opacity of the water inside.
    pub opacity: f32,
    /// Engine-filled state (not saved).
    #[serde(skip)]
    pub runtime: WaterContainerRuntime,
}

/// Engine-filled state of a [`WaterContainer`], not saved.
#[derive(Clone, Debug, Default)]
pub struct WaterContainerRuntime {
    /// Set once Play has taken the initial fill.
    pub started: bool,
    /// Water currently inside (litres, runtime).
    pub litres: f32,
    /// Capacity at the object's current scale (litres).
    pub capacity: f32,
    /// The object's world scale, so the inspector can show litres.
    pub scale: [f32; 3],
    /// The free surface (box frame): normal and plane offset.
    pub up: Vec3,
    /// The free surface's plane offset in the box frame (runtime).
    pub level: f32,
    /// Litres per second leaving over the rim / coming in.
    pub spilling: f32,
    /// Litres per second leaving over the rim (runtime).
    pub filling: f32,
    /// Inspector asked to fit the box to the mesh.
    pub fit_requested: bool,
    /// Gameplay requests, taken on the next step: litres to add, litres to
    /// pour out over the rim.
    pub add: f32,
    /// Gameplay pour request: litres to pour out (runtime).
    pub pour: f32,
}

impl Default for WaterContainer {
    fn default() -> Self {
        // Fits the Bucket primitive (and roughly a unit cube).
        Self {
            center: [0.0, 0.025, 0.0],
            half_extents: [0.46, 0.475, 0.46],
            initial_fill: 0.0,
            pour_rate: 1.0,
            inflow: 1.0,
            slosh: 0.7,
            color: [0.16, 0.42, 0.5],
            opacity: 0.75,
            runtime: WaterContainerRuntime::default(),
        }
    }
}

impl TypedComponent for WaterContainer {
    const NAME: &'static str = "Water Container";
}

impl WaterContainer {
    /// Half extents in metres at world scale `scale`.
    pub fn half_m(&self, scale: Vec3) -> Vec3 {
        (Vec3::from(self.half_extents) * scale.abs()).max(Vec3::splat(1e-3))
    }

    /// Box frame to world (rotation and translation only) and the half
    /// extents in metres, for the object's world matrix.
    pub fn frame(&self, world: Mat4) -> (Mat4, Vec3) {
        let (s, r, t) = world.to_scale_rotation_translation();
        let c = Vec3::from(self.center) * s;
        (Mat4::from_rotation_translation(r, t + r * c), self.half_m(s))
    }

    /// Capacity in litres at this object scale.
    pub fn capacity_litres(&self, scale: Vec3) -> f32 {
        let h = self.half_m(scale);
        8.0 * h.x * h.y * h.z * LITRES_PER_M3
    }

    /// Box fitted inside mesh bounds `lo`..`hi` (object space): pulled in a
    /// little from the sides and the bottom for the walls, open at the top.
    pub fn fit_to_bounds(&mut self, lo: Vec3, hi: Vec3) {
        let size = (hi - lo).max(Vec3::splat(1e-3));
        let lo = lo + size * Vec3::new(0.04, 0.05, 0.04);
        let hi = hi - size * Vec3::new(0.04, 0.0, 0.04);
        self.center = ((lo + hi) * 0.5).to_array();
        self.half_extents = ((hi - lo) * 0.5).to_array();
    }

    /// The water surface to draw: box frame, half extents and the plane
    /// (`up`, `level`). Outside Play it shows the initial fill, level in the
    /// world. None when empty.
    pub fn surface(&self, world: Mat4) -> Option<(Mat4, Vec3, Vec3, f32)> {
        let (frame, half) = self.frame(world);
        let (up, level, litres) = if self.runtime.started {
            (self.runtime.up, self.runtime.level, self.runtime.litres)
        } else {
            let up = frame.inverse().transform_vector3(Vec3::Y).normalize_or(Vec3::Y);
            let v = self.initial_fill.clamp(0.0, 1.0) * 8.0 * half.x * half.y * half.z;
            let held = held_volume(half, up);
            (up, level_for_volume(half, up, v.min(held)), v.min(held) * LITRES_PER_M3)
        };
        (litres > 1e-3).then_some((frame, half, up, level))
    }
}

/// Water volume (m^3) in the box below the plane `up . p = level`, and its
/// centroid (box frame). `up` must be unit length.
pub fn volume_below(half: Vec3, up: Vec3, level: f32) -> (f32, Vec3) {
    // Integrate columns along the axis most aligned with `up`: each column
    // is cut by the plane once.
    let a = up.abs();
    let k = if a.x >= a.y && a.x >= a.z { 0 } else if a.y >= a.z { 1 } else { 2 };
    let (i, j) = ((k + 1) % 3, (k + 2) % 3);
    let (hi, hj, hk) = (half[i], half[j], half[k]);
    let (ui, uj, uk) = (up[i], up[j], up[k]);
    let (di, dj) = (2.0 * hi / SLICE_N as f32, 2.0 * hj / SLICE_N as f32);
    let cell = di * dj;
    let mut volume = 0.0;
    let mut moment = Vec3::ZERO;
    for si in 0..SLICE_N {
        let x = -hi + (si as f32 + 0.5) * di;
        for sj in 0..SLICE_N {
            let y = -hj + (sj as f32 + 0.5) * dj;
            let cut = (level - ui * x - uj * y) / uk;
            let (len, mid) = if uk > 0.0 {
                let len = (cut + hk).clamp(0.0, 2.0 * hk);
                (len, -hk + len * 0.5)
            } else {
                let len = (hk - cut).clamp(0.0, 2.0 * hk);
                (len, hk - len * 0.5)
            };
            if len <= 0.0 {
                continue;
            }
            let v = len * cell;
            volume += v;
            let mut p = Vec3::ZERO;
            p[i] = x;
            p[j] = y;
            p[k] = mid;
            moment += p * v;
        }
    }
    let centroid = if volume > 0.0 { moment / volume } else { Vec3::ZERO };
    (volume, centroid)
}

/// Lowest and highest plane offsets along `up` that touch the box.
fn level_range(half: Vec3, up: Vec3) -> (f32, f32) {
    let r = up.abs().dot(half);
    (-r, r)
}

/// Plane offset at which the box holds `volume` (m^3) below the plane.
pub fn level_for_volume(half: Vec3, up: Vec3, volume: f32) -> f32 {
    let (mut lo, mut hi) = level_range(half, up);
    let full = 8.0 * half.x * half.y * half.z;
    if volume <= 0.0 {
        return lo;
    }
    if volume >= full {
        return hi;
    }
    for _ in 0..22 {
        let mid = (lo + hi) * 0.5;
        if volume_below(half, up, mid).0 < volume {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) * 0.5
}

/// The opening's four corners (the box's +Y face).
pub fn rim_corners(half: Vec3) -> [Vec3; 4] {
    [
        Vec3::new(-half.x, half.y, -half.z),
        Vec3::new(half.x, half.y, -half.z),
        Vec3::new(half.x, half.y, half.z),
        Vec3::new(-half.x, half.y, half.z),
    ]
}

/// Plane offset of the rim's lowest point: water above it pours out.
pub fn rim_level(half: Vec3, up: Vec3) -> f32 {
    rim_corners(half).iter().map(|c| up.dot(*c)).fold(f32::MAX, f32::min)
}

/// Most water (m^3) the box holds at this tilt: what fits below the rim.
pub fn held_volume(half: Vec3, up: Vec3) -> f32 {
    volume_below(half, up, rim_level(half, up)).0
}

/// Where water leaves first: the rim's lowest point (the middle of the
/// lowest edge when it's level along an edge).
pub fn spill_point(half: Vec3, up: Vec3) -> Vec3 {
    let c = rim_corners(half);
    let h: Vec<f32> = c.iter().map(|p| up.dot(*p)).collect();
    let low = (0..4).min_by(|a, b| h[*a].total_cmp(&h[*b])).unwrap_or(0);
    let (prev, next) = ((low + 3) % 4, (low + 1) % 4);
    let tol = 1e-3 * half.length();
    let other = if h[next] - h[low] < tol { Some(next) } else if h[prev] - h[low] < tol { Some(prev) } else { None };
    other.map_or(c[low], |o| (c[low] + c[o]) * 0.5)
}

/// Points along the rim under the water surface, spaced evenly round it,
/// with how far under each is, and the length of rim they cover (m).
pub fn wet_lip(half: Vec3, up: Vec3, level: f32, samples: usize) -> (Vec<(Vec3, f32)>, f32) {
    let c = rim_corners(half);
    let edges = [(c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0])];
    let perimeter: f32 = edges.iter().map(|(a, b)| a.distance(*b)).sum();
    let mut out = Vec::new();
    let samples = samples.max(4);
    for s in 0..samples {
        let mut d = (s as f32 + 0.5) / samples as f32 * perimeter;
        for (a, b) in edges {
            let len = a.distance(b);
            if d <= len {
                let p = a.lerp(b, d / len.max(1e-6));
                let head = level - up.dot(p);
                if head > 0.0 {
                    out.push((p, head));
                }
                break;
            }
            d -= len;
        }
    }
    let wet = perimeter * out.len() as f32 / samples as f32;
    (out, wet)
}

/// The polygon where the plane cuts the box (box frame, in order round
/// it); empty when it misses.
pub fn surface_polygon(half: Vec3, up: Vec3, level: f32) -> Vec<Vec3> {
    let corner = |k: usize| Vec3::new(if k & 1 == 0 { -half.x } else { half.x }, if k & 2 == 0 { -half.y } else { half.y }, if k & 4 == 0 { -half.z } else { half.z });
    let mut pts: Vec<Vec3> = Vec::with_capacity(6);
    for a in 0..8usize {
        for bit in [1usize, 2, 4] {
            if a & bit != 0 {
                continue;
            }
            let (pa, pb) = (corner(a), corner(a | bit));
            let (da, db) = (up.dot(pa) - level, up.dot(pb) - level);
            if (da < 0.0) == (db < 0.0) {
                continue;
            }
            let t = da / (da - db);
            let p = pa.lerp(pb, t);
            if pts.iter().all(|q| q.distance_squared(p) > 1e-10) {
                pts.push(p);
            }
        }
    }
    if pts.len() < 3 {
        return Vec::new();
    }
    let centre = pts.iter().copied().sum::<Vec3>() / pts.len() as f32;
    let ax = if up.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let u = ax.cross(up).normalize_or(Vec3::X);
    let v = up.cross(u);
    pts.sort_by(|a, b| {
        let (da, db) = (*a - centre, *b - centre);
        da.dot(v).atan2(da.dot(u)).total_cmp(&db.dot(v).atan2(db.dot(u)))
    });
    pts
}

/// Whether box-frame point `p` is inside the box.
pub fn inside_box(half: Vec3, p: Vec3) -> bool {
    p.x.abs() <= half.x && p.y.abs() <= half.y && p.z.abs() <= half.z
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    const HALF: Vec3 = Vec3::new(0.5, 0.5, 0.5);

    fn up_for(rot: Quat) -> Vec3 {
        // World up seen from a box rotated by `rot`.
        (rot.inverse() * Vec3::Y).normalize()
    }

    #[test]
    fn upright_box_holds_its_full_capacity() {
        let up = Vec3::Y;
        let held = held_volume(HALF, up);
        assert!((held - 1.0).abs() < 1e-4, "{held}");
        let lvl = level_for_volume(HALF, up, 0.25);
        assert!((lvl + 0.25).abs() < 1e-3, "a quarter full sits a quarter up: {lvl}");
        let (v, c) = volume_below(HALF, up, lvl);
        assert!((v - 0.25).abs() < 1e-3 && (c.y + 0.375).abs() < 1e-3 && c.x.abs() < 1e-4, "{v} {c}");
    }

    #[test]
    fn tilted_45_degrees_holds_about_half() {
        for rot in [Quat::from_rotation_x(45f32.to_radians()), Quat::from_rotation_z(-45f32.to_radians())] {
            let held = held_volume(HALF, up_for(rot));
            assert!((held - 0.5).abs() < 0.02, "{held}");
        }
        // Tipped about both axes it spills from a corner and holds less.
        let corner = Quat::from_rotation_x(45f32.to_radians()) * Quat::from_rotation_z(45f32.to_radians());
        let held = held_volume(HALF, up_for(corner));
        assert!(held < 0.5 && held > 0.05, "{held}");
        let sp = spill_point(HALF, up_for(Quat::from_rotation_x(45f32.to_radians())));
        assert!((sp.y - 0.5).abs() < 1e-5 && sp.x.abs() < 1e-4, "an edge pours from its middle: {sp}");
    }

    #[test]
    fn upside_down_holds_nothing() {
        let held = held_volume(HALF, up_for(Quat::from_rotation_x(std::f32::consts::PI)));
        assert!(held < 1e-4, "{held}");
        let lying = held_volume(HALF, up_for(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)));
        assert!(lying < 1e-4, "on its side the rim is a vertical edge: {lying}");
    }

    #[test]
    fn the_level_stays_level_in_the_world_while_the_box_tilts() {
        let c = WaterContainer { half_extents: [0.5; 3], center: [0.0; 3], ..Default::default() };
        for deg in [0.0f32, 15.0, 30.0, 40.0] {
            let rot = Quat::from_rotation_x(deg.to_radians()) * Quat::from_rotation_y(0.3);
            let world = Mat4::from_rotation_translation(rot, Vec3::new(1.0, 2.0, 3.0));
            let (frame, half) = c.frame(world);
            let up = frame.inverse().transform_vector3(Vec3::Y).normalize();
            let v = 0.3f32.min(held_volume(half, up));
            let level = level_for_volume(half, up, v);
            assert!((volume_below(half, up, level).0 - v).abs() < 2e-3);
            let poly = surface_polygon(half, up, level);
            assert!(poly.len() >= 3);
            let ys: Vec<f32> = poly.iter().map(|p| frame.transform_point3(*p).y).collect();
            let (lo, hi) = ys.iter().fold((f32::MAX, f32::MIN), |(a, b), y| (a.min(*y), b.max(*y)));
            assert!(hi - lo < 1e-4, "flat in the world at {deg} deg: {ys:?}");
            // The water sits low: its centroid is below the surface.
            let (_, centroid) = volume_below(half, up, level);
            assert!(frame.transform_point3(centroid).y < lo);
        }
    }

    #[test]
    fn the_lip_is_wet_where_the_rim_dips_under_the_surface() {
        let up = up_for(Quat::from_rotation_x(30f32.to_radians()));
        let (pts, len) = wet_lip(HALF, up, rim_level(HALF, up) + 0.05, 64);
        assert!(!pts.is_empty() && len > 0.0 && len < 4.0);
        assert!(pts.iter().all(|(p, h)| *h > 0.0 && (p.y - 0.5).abs() < 1e-5));
        let (none, _) = wet_lip(HALF, Vec3::Y, 0.4, 64);
        assert!(none.is_empty(), "upright and below the rim: nothing pours");
    }

    #[test]
    fn fitting_pulls_the_box_inside_the_mesh() {
        let mut c = WaterContainer::default();
        c.fit_to_bounds(Vec3::splat(-0.5), Vec3::splat(0.5));
        let d = WaterContainer::default();
        for k in 0..3 {
            assert!((c.center[k] - d.center[k]).abs() < 1e-5 && (c.half_extents[k] - d.half_extents[k]).abs() < 1e-5);
        }
        assert!((c.capacity_litres(Vec3::ONE) - 0.92 * 0.95 * 0.92 * 1000.0).abs() < 0.5);
    }
}
