//! Interactive water: a height-field wave simulation over a square window of
//! a Water body that follows the action. Floating bodies and characters push
//! the surface as they move, and the disturbance travels at the shallow-water
//! wave speed for the local depth, so they leave ripples and wakes. Cells
//! marked blocked (posts, walls, rocks through the surface) reflect waves.
//! The result adds to the body's analytic waves for physics and rendering.

use glam::Vec2;

/// Something disturbing the water this step (water-local XZ).
#[derive(Clone, Copy, Debug)]
pub struct WaterPush {
    /// Where the water is disturbed (water-local XZ).
    pub at: Vec2,
    /// Radius of the disturbance (m).
    pub radius: f32,
    /// Downward push speed (m/s): vertical motion plus a share of horizontal
    /// speed for wakes.
    pub strength: f32,
}

/// The height-field simulation over one Water's interactive window.
#[derive(Clone, Debug)]
pub struct RippleSim {
    /// Cells per side.
    pub n: usize,
    /// Window width (m).
    pub size: f32,
    /// Window centre (local XZ), snapped to whole cells.
    pub center: Vec2,
    height: Vec<f32>,
    velocity: Vec<f32>,
    /// Cells something solid passes through (row-major, empty = none).
    blocked: Vec<bool>,
    /// Filled from the GPU simulation's readback (not stepped here).
    pub gpu: bool,
    /// Horizontal water velocity (local XZ, m/s) per cell; GPU-fed only.
    flow: Vec<Vec2>,
    /// Largest |height| and flow speed in the field (GPU-fed only).
    pub max_height: f32,
    /// Largest flow speed in the field (GPU-fed only).
    pub max_speed: f32,
}

impl RippleSim {
    /// An empty simulation window of `n` cells per side over `size` metres.
    pub fn new(n: usize, size: f32) -> Self {
        let n = n.clamp(8, 512);
        Self {
            n,
            size: size.max(1.0),
            center: Vec2::ZERO,
            height: vec![0.0; n * n],
            velocity: vec![0.0; n * n],
            blocked: Vec::new(),
            gpu: false,
            flow: Vec::new(),
            max_height: 0.0,
            max_speed: 0.0,
        }
    }

    /// An empty field owned by the GPU simulation, filled by [`Self::set_field`].
    pub fn gpu_owned(n: usize, size: f32) -> Self {
        Self { gpu: true, ..Self::new(n, size) }
    }

    /// Replace the field with a readback: `n` x `n` cells over `size` metres
    /// around `center` (local XZ), each (height offset, flow x, flow z).
    pub fn set_field(&mut self, n: usize, size: f32, center: Vec2, cells: &[[f32; 4]]) {
        if cells.len() != n * n || n < 2 {
            return;
        }
        self.n = n;
        self.size = size.max(1e-3);
        self.center = center;
        self.height.clear();
        self.flow.clear();
        self.velocity.clear();
        let (mut mh, mut ms) = (0.0f32, 0.0f32);
        for c in cells {
            self.height.push(c[0]);
            let f = Vec2::new(c[1], c[2]);
            self.flow.push(f);
            mh = mh.max(c[0].abs());
            ms = ms.max(f.length());
        }
        self.max_height = mh;
        self.max_speed = ms;
    }

    /// Simulated horizontal water velocity at local `xz` (zero outside the
    /// window or without GPU data).
    pub fn flow(&self, xz: Vec2) -> Vec2 {
        if self.flow.len() != self.n * self.n {
            return Vec2::ZERO;
        }
        self.sample(xz, |i| self.flow[i]).unwrap_or(Vec2::ZERO)
    }

    fn sample<T: Copy + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<f32, Output = T>>(
        &self,
        xz: Vec2,
        at: impl Fn(usize) -> T,
    ) -> Option<T> {
        let cell = self.cell();
        let g = (xz - self.center + Vec2::splat(self.size * 0.5)) / cell - Vec2::splat(0.5);
        let (x0, z0) = (g.x.floor(), g.y.floor());
        if x0 < 0.0 || z0 < 0.0 || x0 + 1.0 >= self.n as f32 || z0 + 1.0 >= self.n as f32 {
            return None;
        }
        let (fx, fz) = (g.x - x0, g.y - z0);
        let (x0, z0) = (x0 as usize, z0 as usize);
        let n = self.n;
        let v = |x: usize, z: usize| at(z * n + x);
        let top = v(x0, z0) + (v(x0 + 1, z0) - v(x0, z0)) * fx;
        let bot = v(x0, z0 + 1) + (v(x0 + 1, z0 + 1) - v(x0, z0 + 1)) * fx;
        Some(top + (bot - top) * fz)
    }

    /// The cell size (window width / cells per side, m).
    pub fn cell(&self) -> f32 {
        self.size / self.n as f32
    }

    /// Move the window toward `focus`, shifting the field by whole cells so
    /// the water already disturbed stays put in the world.
    pub fn follow(&mut self, focus: Vec2) {
        let cell = self.cell();
        let shift = ((focus - self.center) / cell).round();
        if shift == Vec2::ZERO {
            return;
        }
        let (sx, sz) = (shift.x as i64, shift.y as i64);
        let n = self.n as i64;
        let mut h = vec![0.0; self.height.len()];
        let mut v = vec![0.0; self.velocity.len()];
        for z in 0..n {
            for x in 0..n {
                let (ox, oz) = (x + sx, z + sz);
                if (0..n).contains(&ox) && (0..n).contains(&oz) {
                    h[(z * n + x) as usize] = self.height[(oz * n + ox) as usize];
                    v[(z * n + x) as usize] = self.velocity[(oz * n + ox) as usize];
                }
            }
        }
        self.height = h;
        self.velocity = v;
        self.blocked.clear();
        self.flow.clear();
        self.center += shift * cell;
    }

    /// Local XZ of cell (`x`, `z`)'s centre.
    pub fn cell_center(&self, x: usize, z: usize) -> Vec2 {
        let cell = self.cell();
        self.center - Vec2::splat(self.size * 0.5) + Vec2::new((x as f32 + 0.5) * cell, (z as f32 + 0.5) * cell)
    }

    /// Mark which cells are solid (row-major `n` x `n`; empty clears).
    pub fn set_blocked(&mut self, blocked: Vec<bool>) {
        self.blocked = if blocked.len() == self.n * self.n && blocked.iter().any(|b| *b) { blocked } else { Vec::new() };
    }

    /// Advance by `dt`. `depth` is the still-water depth (sets the wave
    /// speed, sqrt(g h)); `damping` the share of motion lost per second.
    pub fn step(&mut self, dt: f32, depth: f32, damping: f32, pushes: &[WaterPush]) {
        if dt <= 0.0 {
            return;
        }
        let n = self.n;
        let cell = self.cell();
        let half = self.size * 0.5;
        for p in pushes {
            let local = p.at - self.center + Vec2::splat(half);
            let r = p.radius.max(cell);
            let (lo, hi) = (((local - Vec2::splat(r)) / cell).floor(), ((local + Vec2::splat(r)) / cell).ceil());
            for z in lo.y.max(0.0) as usize..(hi.y.max(0.0) as usize).min(n) {
                for x in lo.x.max(0.0) as usize..(hi.x.max(0.0) as usize).min(n) {
                    let c = Vec2::new((x as f32 + 0.5) * cell, (z as f32 + 0.5) * cell);
                    let fall = (1.0 - c.distance(local) / r).max(0.0);
                    self.velocity[z * n + x] -= p.strength * fall * fall * dt * 8.0;
                }
            }
        }
        let speed = (9.81 * depth.clamp(0.05, 4.0)).sqrt();
        // Stay inside the explicit scheme's stability limit.
        let steps = ((speed * dt / (cell * 0.5)).ceil() as usize).clamp(1, 16);
        let h_dt = dt / steps as f32;
        let k = speed * speed / (cell * cell);
        let keep = (1.0 - damping.clamp(0.0, 1.0)).powf(h_dt);
        let border = (n / 10).max(2);
        for _ in 0..steps {
            for z in 0..n {
                for x in 0..n {
                    let i = z * n + x;
                    let here = self.height[i];
                    let blocked = &self.blocked;
                    // A solid neighbour mirrors this cell, so waves bounce off it.
                    let at = |xx: usize, zz: usize| {
                        let j = zz * n + xx;
                        if blocked.get(j).copied().unwrap_or(false) { here } else { self.height[j] }
                    };
                    let l = if x > 0 { at(x - 1, z) } else { 0.0 };
                    let r = if x + 1 < n { at(x + 1, z) } else { 0.0 };
                    let d = if z > 0 { at(x, z - 1) } else { 0.0 };
                    let u = if z + 1 < n { at(x, z + 1) } else { 0.0 };
                    let lap = l + r + d + u - 4.0 * self.height[i];
                    self.velocity[i] = (self.velocity[i] + k * lap * h_dt) * keep;
                }
            }
            for z in 0..n {
                for x in 0..n {
                    let i = z * n + x;
                    // A soft border absorbs waves leaving the window.
                    let edge = x.min(z).min(n - 1 - x).min(n - 1 - z);
                    let sponge = if edge < border { 0.85 + 0.15 * edge as f32 / border as f32 } else { 1.0 };
                    if self.blocked.get(i).copied().unwrap_or(false) {
                        self.velocity[i] = 0.0;
                        self.height[i] = 0.0;
                        continue;
                    }
                    self.velocity[i] *= sponge;
                    self.height[i] = (self.height[i] + self.velocity[i] * h_dt) * sponge;
                }
            }
        }
    }

    /// Simulated height offset at local `xz` (0 outside the window).
    pub fn height(&self, xz: Vec2) -> f32 {
        if self.height.len() != self.n * self.n {
            return 0.0;
        }
        self.sample(xz, |i| self.height[i]).unwrap_or(0.0)
    }

    /// The raw field (row-major, `n` x `n`), for uploading to the GPU.
    pub fn heights(&self) -> &[f32] {
        &self.height
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_push_makes_a_ring_that_spreads_and_fades() {
        let mut sim = RippleSim::new(96, 24.0);
        let push = WaterPush { at: Vec2::ZERO, radius: 0.5, strength: 2.0 };
        for _ in 0..6 {
            sim.step(1.0 / 60.0, 1.0, 0.5, &[push]);
        }
        assert!(sim.height(Vec2::ZERO) < -0.01, "pushed down where it hit: {}", sim.height(Vec2::ZERO));
        for _ in 0..40 {
            sim.step(1.0 / 60.0, 1.0, 0.5, &[]);
        }
        let ring: f32 = (0..32).map(|k| {
            let a = k as f32 / 32.0 * std::f32::consts::TAU;
            sim.height(Vec2::new(a.cos(), a.sin()) * 2.0).abs()
        }).sum::<f32>() / 32.0;
        assert!(ring > 1e-4, "the ripple travelled outward: {ring}");
        for _ in 0..1200 {
            sim.step(1.0 / 60.0, 1.0, 0.5, &[]);
        }
        let left: f32 = sim.heights().iter().map(|h| h.abs()).fold(0.0, f32::max);
        assert!(left < 0.002, "damping and the border settle it: {left}");
    }

    #[test]
    fn a_wall_reflects_waves() {
        let run = |wall: bool| {
            let mut sim = RippleSim::new(64, 16.0);
            if wall {
                let mut b = vec![false; 64 * 64];
                for z in 0..64 {
                    b[z * 64 + 40] = true;
                }
                sim.set_blocked(b);
            }
            sim.step(1.0 / 60.0, 1.0, 0.0, &[WaterPush { at: Vec2::new(0.0, 0.0), radius: 0.5, strength: 3.0 }]);
            for _ in 0..90 {
                sim.step(1.0 / 60.0, 1.0, 0.0, &[]);
            }
            (sim.height(Vec2::new(3.5, 0.0)).abs(), sim.height(Vec2::new(4.6, 0.0)).abs())
        };
        let (open_near, _) = run(false);
        let (walled_near, behind) = run(true);
        assert!(behind < 1e-6, "nothing passes the wall: {behind}");
        assert!((walled_near - open_near).abs() > 1e-4, "the reflection changes the water in front of it: {walled_near} vs {open_near}");
    }

    #[test]
    fn following_keeps_disturbances_in_place() {
        let mut sim = RippleSim::new(64, 16.0);
        sim.step(1.0 / 60.0, 1.0, 0.0, &[WaterPush { at: Vec2::new(2.0, 1.0), radius: 0.5, strength: 3.0 }]);
        let before = sim.height(Vec2::new(2.0, 1.0));
        sim.follow(Vec2::new(3.0, -2.0));
        let after = sim.height(Vec2::new(2.0, 1.0));
        assert!(before < 0.0 && (before - after).abs() < 1e-5, "{before} {after}");
        assert_eq!(sim.height(Vec2::new(100.0, 0.0)), 0.0);
    }
}
