//! Dispersive layer of the interactive water: the eWave scheme (Tessendorf
//! 2014, "eWave: Using an Exponential Solver on the iWave Problem"). On the
//! pipe simulation's window it keeps a surface height `h` and the velocity
//! potential `phi` at the surface, and each frame moves every Fourier mode
//! forward exactly:
//!
//!   h'   = cos(w dt) h   + (k tanh(kd) / w) sin(w dt) phi
//!   phi' = cos(w dt) phi - ((g + s k^2) / w) sin(w dt) h
//!
//! with w^2 = (g k + s k^3) tanh(k d) (`s` surface tension over density), so
//! short waves run slower than long ones: wave trains behind an impact,
//! Kelvin wakes behind a moving body, fine chop. Each step is a rotation of
//! the mode's (h, phi), stable at any time step. In deep water the pair is
//! the paper's (k/w, g/w); at finite depth the k tanh(kd) and g + s k^2
//! factors keep the step energy-preserving.
//!
//! The GPU runs it in `water_sim.comp` (sources, mode update, mask) and
//! `water_fft.comp` (the 2D transform); [`EwaveLayer`] is the same passes on
//! the CPU, which `PipeSim` drives as its reference.
//!
//! The state lives in the pipe sim's toroidal store (world cell g at g mod
//! n), which is exactly the periodic domain the FFT assumes, so a moving
//! window keeps its waves in place. What would wrap round crosses the
//! window's border first: a following window's sponge absorbs it, and a
//! window over a whole lake has a shore margin, solid cells being masked to
//! rest after every step. Cells the window newly covers start at rest.
//!
//! The pipe sim carries the water's mass, flow and sloshing; this layer only
//! its waves: it holds no mean level (the k = 0 mode is dropped) and hands
//! its longest modes to the pipe sim where the shallow-water speed is right
//! (kd under [`HANDOVER_KD`], in water no deeper than the wave depth).

use glam::Vec2;

use crate::water::Water;
use crate::water_pipe::SimWindow;

const GRAVITY: f32 = 9.81;
/// Surface tension over density of water (m^3/s^2).
pub const SURFACE_TENSION: f32 = 7.3e-5;
/// Height the layer takes per metre of body entering, at strength 1.
const SOURCE_GAIN: f32 = 0.5;
/// Most height one cell takes from its sources per frame (m).
pub const MAX_SOURCE: f32 = 0.25;
/// Modes longer than kd = this are the pipe sim's (when its waves run at
/// the real depth).
pub const HANDOVER_KD: f32 = 0.5;
/// Rate (1/s) the longest modes fade at in the handover.
pub const HANDOVER_RATE: f32 = 3.0;
/// Damping (1/s) at a following window's outer edge, easing in across its
/// sponge border.
pub const SPONGE_RATE: f32 = 30.0;
/// Draft (m) at which a body blocks the waves fully (`EW_BLOCK_DRAFT`).
pub const BLOCK_DRAFT: f32 = 0.08;
/// Fastest orbital velocity reported (m/s).
pub const MAX_ORBITAL: f32 = 4.0;

/// Whether a window of `n` cells per side can run the layer (the FFT needs
/// a power of two; 512 is the shader's line limit).
pub fn ewave_supported(n: u32) -> bool {
    n.is_power_of_two() && (8..=512).contains(&n)
}

/// The share of the layer's strength water `depth` metres deep runs at.
/// Over shallow water tanh(kd) ~ kd makes the window's waves nearly
/// non-dispersive, which the pipe sim already shows right, so the layer
/// mostly adds the short chop there and is turned down; from 1.5 m it runs
/// in full.
pub fn dispersive_share(depth: f32) -> f32 {
    let t = ((depth - 0.1) / 1.4).clamp(0.0, 1.0);
    0.25 + 0.75 * t * t * (3.0 - 2.0 * t)
}

/// Angular frequency (rad/s) of a wave of wave number `k` (rad/m) over
/// `depth` metres of water.
pub fn ewave_omega(k: f32, depth: f32, gravity: f32) -> f32 {
    ((gravity * k + SURFACE_TENSION * k * k * k) * (k * depth).min(10.0).tanh()).max(0.0).sqrt()
}

/// The layer's constants for a frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EwaveParams {
    pub gravity: f32,
    /// Rest depth the waves feel (m).
    pub depth: f32,
    /// Height per metre of body occupancy change (0 = no sources).
    pub gain: f32,
    /// Damping of every mode (1/s).
    pub damping: f32,
    /// Extra damping at the grid's shortest waves (1/s).
    pub grid_damping: f32,
    /// Wave number (rad/m) below which modes fade to the pipe sim (0 = none).
    pub split: f32,
    /// Uniform current carrying the waves (local XZ, m/s).
    pub current: Vec2,
    /// Share of a body's occupancy change that starts waves (drops and
    /// pushes keep full strength).
    pub body_gain: f32,
    /// Seconds a body's occupancy change is spread over (0 = at once).
    pub smoothing: f32,
    /// Kinematic viscosity (m^2/s): each mode also decays at 2 nu k^2.
    pub viscosity: f32,
    /// Share of the layer that shows on the surface (0..1).
    pub mix: f32,
}

impl EwaveParams {
    /// Share of the occupancy change still owed that a step of `dt` lets
    /// out: a low-pass on the source, so bodies moved once a frame don't
    /// kick the surface at frame rate (grid-sized ripples all round).
    pub fn follow(&self, dt: f32) -> f32 {
        if self.smoothing <= 1e-4 { 1.0 } else { 1.0 - (-dt.max(0.0) / self.smoothing).exp() }
    }

    /// For `water` on `window` with rest depth `depth`; None when the layer
    /// is off or the window can't run it.
    pub fn for_water(water: &Water, window: SimWindow, depth: f32) -> Option<Self> {
        let strength = water.sim_dispersion.max(0.0);
        let mix = water.sim_ripple_mix.clamp(0.0, 1.0);
        if strength <= 0.0 || mix <= 0.0 || !ewave_supported(window.n) {
            return None;
        }
        // Past the window's size every mode is deep already.
        let depth = depth.clamp(0.05, window.size().max(1.0));
        let wave_depth = water.sim_wave_depth.max(0.05);
        let current = if water.body == crate::water::WaterBody::River { Vec2::ZERO } else { Vec2::new(water.current[0], water.current[2]) };
        Some(Self {
            gravity: GRAVITY,
            depth,
            gain: strength * SOURCE_GAIN * dispersive_share(depth),
            damping: -(1.0 - water.ripple_damping.clamp(0.0, 0.99)).ln(),
            grid_damping: water.sim_fine_damping.max(0.0),
            split: if depth <= wave_depth { HANDOVER_KD / depth } else { 0.0 },
            current,
            body_gain: water.sim_wake_ripples.max(0.0),
            smoothing: water.sim_ripple_smoothing.max(0.0),
            viscosity: water.kinematic_viscosity(),
            mix,
        })
    }

    /// Undamped, unsourced: for checking the scheme.
    pub fn free(depth: f32) -> Self {
        Self { gravity: GRAVITY, depth, gain: 1.0, damping: 0.0, grid_damping: 0.0, split: 0.0, current: Vec2::ZERO, body_gain: 1.0, smoothing: 0.0, viscosity: 0.0, mix: 1.0 }
    }
}

type C = [f32; 2];

fn cmul(a: C, b: C) -> C {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

/// One line of complex values (a power of two long): radix-2 Stockham,
/// natural order in and out, as `water_fft.comp` does it. `tmp` is scratch
/// of the same length.
fn fft_line(line: &mut Vec<C>, tmp: &mut Vec<C>, inverse: bool) {
    let n = line.len();
    let half = n / 2;
    let sgn = if inverse { 1.0 } else { -1.0 };
    let mut ns = 1;
    while ns < n {
        for j in 0..half {
            let v0 = line[j];
            let r = j & (ns - 1);
            let (s, c) = (sgn * std::f32::consts::PI * r as f32 / ns as f32).sin_cos();
            let v1 = cmul(line[j + half], [c, s]);
            let d = (j - r) * 2 + r;
            tmp[d] = [v0[0] + v1[0], v0[1] + v1[1]];
            tmp[d + ns] = [v0[0] - v1[0], v0[1] - v1[1]];
        }
        std::mem::swap(line, tmp);
        ns *= 2;
    }
}

/// One pass of the 2D transform: each row of `src` transformed and written
/// transposed into `dst`, times `scale`.
fn fft_pass(src: &[C], dst: &mut [C], n: usize, inverse: bool, scale: f32) {
    let mut line = vec![[0.0; 2]; n];
    let mut tmp = vec![[0.0; 2]; n];
    for r in 0..n {
        line.copy_from_slice(&src[r * n..(r + 1) * n]);
        fft_line(&mut line, &mut tmp, inverse);
        for (i, v) in line.iter().enumerate() {
            dst[i * n + r] = [v[0] * scale, v[1] * scale];
        }
    }
}

/// 2D transform of `n` x `n` row-major values in place (the inverse scaled
/// by 1/n^2): a row pass and a column pass, as on the GPU.
fn fft2(data: &mut [C], n: usize, inverse: bool) {
    let mut tmp = vec![[0.0; 2]; n * n];
    fft_pass(data, &mut tmp, n, inverse, 1.0);
    let scale = if inverse { 1.0 / (n * n) as f32 } else { 1.0 };
    fft_pass(&tmp, data, n, inverse, scale);
}

/// Signed frequency index of transform index `i`.
fn freq(i: usize, n: usize) -> i32 {
    if i < n / 2 { i as i32 } else { i as i32 - n as i32 }
}

/// The real fields' transforms (H, Phi) from the packed one at `i` and its
/// mirror `m` (Z = H + i Phi, both fields real).
fn unpack(z: C, zm: C) -> (C, C) {
    let zc = [zm[0], -zm[1]];
    let hk = [0.5 * (z[0] + zc[0]), 0.5 * (z[1] + zc[1])];
    let d = [z[0] - zc[0], z[1] - zc[1]];
    (hk, [0.5 * d[1], -0.5 * d[0]])
}

/// The dispersive layer's state on the CPU, in the pipe sim's store layout.
#[derive(Clone, Debug)]
pub struct EwaveLayer {
    pub n: usize,
    /// (h, phi) per store cell.
    pub state: Vec<C>,
    /// Body occupancy at the last frame per store cell; negative until seen.
    prev: Vec<f32>,
    spec: Vec<C>,
}

impl EwaveLayer {
    pub fn new(n: usize) -> Self {
        Self { n, state: vec![[0.0; 2]; n * n], prev: vec![-1.0; n * n], spec: vec![[0.0; 2]; n * n] }
    }

    /// Store cell `s` back at rest (RESET).
    pub fn reset_cell(&mut self, s: usize) {
        self.state[s] = [0.0; 2];
        self.prev[s] = -1.0;
    }

    /// SOURCE: store cell `s` this frame holds bodies `occupancy` metres
    /// deep and one-frame dents `impulse` deep. The change in occupancy
    /// (the `follow` share of it still owed, `EwaveParams::follow`) and the
    /// dents push the surface down (when `step`, else only remembered).
    pub fn inject(&mut self, s: usize, occupancy: f32, impulse: f32, p: &EwaveParams, follow: f32, step: bool) {
        let prev = self.prev[s];
        if !step {
            self.prev[s] = occupancy;
            return;
        }
        let d = if prev < -0.5 { 0.0 } else { (occupancy - prev) * follow.clamp(0.0, 1.0) };
        self.prev[s] = if prev < -0.5 { occupancy } else { prev + d };
        self.state[s][0] -= p.gain * (d * p.body_gain + impulse).clamp(-MAX_SOURCE, MAX_SOURCE);
    }

    /// Advance every mode by `dt` on `cell`-metre cells (forward FFT, mode
    /// update, inverse FFT).
    pub fn propagate(&mut self, dt: f32, cell: f32, p: &EwaveParams) {
        let n = self.n;
        let mut spec = std::mem::take(&mut self.spec);
        spec.copy_from_slice(&self.state);
        fft2(&mut spec, n, false);
        let mut out = vec![[0.0; 2]; n * n];
        for ky in 0..n {
            for kx in 0..n {
                let i = ky * n + kx;
                let m = ((n - ky) % n) * n + (n - kx) % n;
                out[i] = mode_step(spec[i], spec[m], kx, ky, n, cell, dt, p, i == m);
            }
        }
        fft2(&mut out, n, true);
        self.state = out;
        self.spec = spec;
    }

    /// The body occupancy cell `s`'s source has let out so far (m).
    pub fn smoothed(&self, s: usize) -> f32 {
        self.prev[s].max(0.0)
    }

    /// MASK: store cell `s` to rest when solid, else damped by the sponge
    /// (`sponge` 0 inside, 1 at the window's edge) and held down where a
    /// body blocks it (`block` 0 open, 1 blocked: waves reflect off it).
    pub fn mask_cell(&mut self, s: usize, solid: bool, sponge: f32, block: f32, dt: f32) {
        let v = self.state[s];
        if solid || !v[0].is_finite() || !v[1].is_finite() {
            self.state[s] = [0.0; 2];
            return;
        }
        let k = (-SPONGE_RATE * sponge * sponge * dt).exp() * (1.0 - block.clamp(0.0, 1.0));
        self.state[s] = [v[0] * k, v[1] * k];
    }

    /// Wave energy (per unit density and cell area, up to a constant): the
    /// sum over modes of (g + s k^2)|H|^2 + k tanh(kd)|Phi|^2.
    pub fn energy(&self, cell: f32, p: &EwaveParams) -> f64 {
        let n = self.n;
        let mut spec = self.state.clone();
        fft2(&mut spec, n, false);
        let mut e = 0.0f64;
        for ky in 0..n {
            for kx in 0..n {
                let m = ((n - ky) % n) * n + (n - kx) % n;
                let (hk, pk) = unpack(spec[ky * n + kx], spec[m]);
                let k = wave_vector(kx, ky, n, cell).length();
                let restore = p.gravity + SURFACE_TENSION * k * k;
                let kin = k * (k * p.depth).min(10.0).tanh();
                e += (restore * (hk[0] * hk[0] + hk[1] * hk[1]) + kin * (pk[0] * pk[0] + pk[1] * pk[1])) as f64;
            }
        }
        e / (n * n) as f64
    }
}

fn wave_vector(kx: usize, ky: usize, n: usize, cell: f32) -> Vec2 {
    Vec2::new(freq(kx, n) as f32, freq(ky, n) as f32) * (std::f32::consts::TAU / (n as f32 * cell))
}

/// The packed transform at one mode after `dt` (`z` there, `zm` at its
/// mirror; `own` when the mode is its own mirror). Matches `ew_update` in
/// water_sim.comp.
#[allow(clippy::too_many_arguments)]
fn mode_step(z: C, zm: C, kx: usize, ky: usize, n: usize, cell: f32, dt: f32, p: &EwaveParams, own: bool) -> C {
    let kv = wave_vector(kx, ky, n, cell);
    let k = kv.length();
    if k <= 0.0 {
        return [0.0; 2];
    }
    let (hk, pk) = unpack(z, zm);
    let th = (k * p.depth).min(10.0).tanh();
    let restore = p.gravity + SURFACE_TENSION * k * k;
    let omega = (restore * k * th).sqrt();
    let (sn, c) = (omega * dt).sin_cos();
    let q = k * cell / std::f32::consts::PI;
    // Viscous decay of a surface wave: 2 nu k^2 (Lamb), so thick fluids keep
    // only their long, slow waves.
    let mut gamma = p.damping + p.grid_damping * q * q * q * q + 2.0 * p.viscosity * k * k;
    if p.split > 0.0 {
        let r = k / p.split;
        gamma += HANDOVER_RATE * (-r * r).exp();
    }
    let damp = (-gamma * dt).exp();
    let a = k * th / omega * sn;
    let b = restore / omega * sn;
    let mut h2 = [(c * hk[0] + a * pk[0]) * damp, (c * hk[1] + a * pk[1]) * damp];
    let mut p2 = [(c * pk[0] - b * hk[0]) * damp, (c * pk[1] - b * hk[1]) * damp];
    if !own && p.current != Vec2::ZERO {
        let (s, c) = (-kv.dot(p.current) * dt).sin_cos();
        h2 = cmul(h2, [c, s]);
        p2 = cmul(p2, [c, s]);
    }
    [h2[0] - p2[1], h2[1] + p2[0]]
}

/// How far into a window's sponge border window cell `w` is (0 inside, 1 at
/// the edge), `width` the border in cells (0 = none).
pub fn sponge_depth(w: glam::IVec2, n: i32, width: u32) -> f32 {
    if width == 0 {
        return 0.0;
    }
    let e = w.x.min(w.y).min(n - 1 - w.x).min(n - 1 - w.y) as f32;
    let s = width as f32;
    if e < s { 1.0 - e / s } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viscosity_damps_short_waves_much_more_than_long_ones() {
        let left = |nu: f32, k: usize| {
            let n = 64;
            let cell = 0.1;
            let p = EwaveParams { viscosity: nu, ..EwaveParams::free(5.0) };
            let mut layer = EwaveLayer::new(n);
            for z in 0..n {
                for x in 0..n {
                    layer.state[z * n + x][0] = (std::f32::consts::TAU * (k * x) as f32 / n as f32).cos() * 0.01;
                }
            }
            for _ in 0..60 {
                layer.propagate(1.0 / 60.0, cell, &p);
            }
            layer.state.iter().map(|s| s[0] * s[0] + s[1] * s[1]).sum::<f32>()
        };
        // Water barely touches either; honey (7e-3 m2/s) wipes the short one.
        assert!(left(1e-6, 16) > left(0.0, 16) * 0.95);
        let (long, short) = (left(7e-3, 2) / left(0.0, 2), left(7e-3, 16) / left(0.0, 16));
        assert!(short < 0.05 && long > 0.5, "honey keeps the long wave ({long}), not the short ({short})");
    }
    use crate::water_pipe::WaterDisplacer;

    fn dft2(data: &[C], n: usize, inverse: bool) -> Vec<C> {
        let sgn = if inverse { 1.0 } else { -1.0 };
        let mut out = vec![[0.0; 2]; n * n];
        for ky in 0..n {
            for kx in 0..n {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for y in 0..n {
                    for x in 0..n {
                        let a = sgn * std::f64::consts::TAU * ((kx * x + ky * y) as f64) / n as f64;
                        let v = data[y * n + x];
                        re += v[0] as f64 * a.cos() - v[1] as f64 * a.sin();
                        im += v[0] as f64 * a.sin() + v[1] as f64 * a.cos();
                    }
                }
                let s = if inverse { 1.0 / (n * n) as f64 } else { 1.0 };
                out[ky * n + kx] = [(re * s) as f32, (im * s) as f32];
            }
        }
        out
    }

    fn noise(n: usize) -> Vec<C> {
        (0..n * n).map(|i| [((i * 37 % 17) as f32 - 8.0) * 0.1, ((i * 11 % 13) as f32 - 6.0) * 0.07]).collect()
    }

    #[test]
    fn the_fft_matches_a_naive_dft() {
        for n in [8usize, 16] {
            let data = noise(n);
            let want = dft2(&data, n, false);
            let mut got = data.clone();
            fft2(&mut got, n, false);
            for (a, b) in got.iter().zip(&want) {
                assert!((a[0] - b[0]).abs() < 1e-3 && (a[1] - b[1]).abs() < 1e-3, "{a:?} vs {b:?}");
            }
            let back = dft2(&want, n, true);
            fft2(&mut got, n, true);
            for ((a, b), c) in got.iter().zip(&back).zip(&data) {
                assert!((a[0] - b[0]).abs() < 1e-4 && (a[1] - b[1]).abs() < 1e-4);
                assert!((a[0] - c[0]).abs() < 1e-4 && (a[1] - c[1]).abs() < 1e-4, "round trip");
            }
        }
    }

    #[test]
    fn the_dispersion_has_its_limits() {
        let g = GRAVITY;
        let deep = ewave_omega(2.0, 50.0, g);
        assert!((deep - (g * 2.0).sqrt()).abs() < 1e-3, "deep: w^2 = gk");
        let k = 0.05;
        let shallow = ewave_omega(k, 0.5, g) / k;
        assert!((shallow - (g * 0.5).sqrt()).abs() < 0.01, "shallow: c = sqrt(gd)");
        // Capillary-gravity waves are slowest near 1.7 cm, at ~0.23 m/s.
        let (kmin, cmin) = (100..2000)
            .map(|i| i as f32)
            .map(|k| (k, ewave_omega(k, 10.0, g) / k))
            .fold((0.0, f32::MAX), |a, b| if b.1 < a.1 { b } else { a });
        assert!((cmin - 0.23).abs() < 0.01 && (std::f32::consts::TAU / kmin - 0.017).abs() < 0.002, "{kmin} {cmin}");
        assert!(dispersive_share(0.1) < 0.3 && dispersive_share(3.0) == 1.0);
    }

    fn bump(layer: &mut EwaveLayer, c: Vec2, r: f32, a: f32, cell: f32) {
        let n = layer.n;
        let mut mean = 0.0;
        for y in 0..n {
            for x in 0..n {
                let d = (Vec2::new(x as f32, y as f32) - c) * cell;
                let v = a * (-d.length_squared() / (r * r)).exp();
                layer.state[y * n + x][0] += v;
                mean += v;
            }
        }
        mean /= (n * n) as f32;
        for s in &mut layer.state {
            s[0] -= mean;
        }
    }

    #[test]
    fn energy_is_kept_without_damping() {
        let (n, cell) = (64, 0.1);
        let p = EwaveParams::free(2.0);
        let mut layer = EwaveLayer::new(n);
        bump(&mut layer, Vec2::new(20.0, 30.0), 0.3, 0.05, cell);
        bump(&mut layer, Vec2::new(45.0, 12.0), 0.15, -0.03, cell);
        let before = layer.energy(cell, &p);
        let mut moved = 0.0f32;
        for _ in 0..200 {
            layer.propagate(1.0 / 90.0, cell, &p);
            moved = moved.max(layer.state.iter().map(|s| s[1].abs()).fold(0.0, f32::max));
        }
        let after = layer.energy(cell, &p);
        assert!(moved > 1e-3, "the waves move: {moved}");
        assert!(((after - before) / before).abs() < 2e-3, "energy {before} -> {after}");
        // One long step lands where many short ones do.
        let mut a = EwaveLayer::new(n);
        bump(&mut a, Vec2::new(20.0, 30.0), 0.3, 0.05, cell);
        let mut b = a.clone();
        a.propagate(1.5, cell, &p);
        for _ in 0..30 {
            b.propagate(0.05, cell, &p);
        }
        let err = a.state.iter().zip(&b.state).map(|(x, y)| (x[0] - y[0]).abs()).fold(0.0, f32::max);
        assert!(err < 1e-4, "exact in time: {err}");
    }

    /// `h` keeping only wave numbers in `lo..hi`.
    fn band(layer: &EwaveLayer, cell: f32, lo: f32, hi: f32) -> Vec<f32> {
        let n = layer.n;
        let mut spec: Vec<C> = layer.state.iter().map(|s| [s[0], 0.0]).collect();
        fft2(&mut spec, n, false);
        for ky in 0..n {
            for kx in 0..n {
                let k = wave_vector(kx, ky, n, cell).length();
                if k < lo || k >= hi {
                    spec[ky * n + kx] = [0.0; 2];
                }
            }
        }
        fft2(&mut spec, n, true);
        spec.iter().map(|v| v[0]).collect()
    }

    fn mean_radius(h: &[f32], n: usize, c: Vec2, cell: f32) -> f32 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for y in 0..n {
            for x in 0..n {
                let e = (h[y * n + x] as f64).powi(2);
                num += e * ((Vec2::new(x as f32, y as f32) - c).length() * cell) as f64;
                den += e;
            }
        }
        (num / den.max(1e-30)) as f32
    }

    #[test]
    fn an_impulse_spreads_as_a_dispersive_ring() {
        let (n, cell) = (128, 0.1);
        let p = EwaveParams::free(10.0);
        let mut layer = EwaveLayer::new(n);
        let c = Vec2::splat(64.0);
        bump(&mut layer, c, 0.12, 0.02, cell);
        for _ in 0..30 {
            layer.propagate(0.1, cell, &p);
        }
        let long = mean_radius(&band(&layer, cell, 1.5, 4.0), n, c, cell);
        let short = mean_radius(&band(&layer, cell, 8.0, 16.0), n, c, cell);
        // Deep-water group speed is half of sqrt(g / k): ~1 m/s and ~0.45 m/s.
        assert!(long > 2.0 && long < 4.5, "long waves got out to {long} m");
        assert!(short > 0.6 && short < 2.0, "short waves to {short} m");
        assert!(long > short * 1.6, "short waves trail the long ones: {short} vs {long}");
    }

    #[test]
    fn a_moving_body_leaves_a_kelvin_wake() {
        let (n, cell) = (128, 0.15);
        let p = EwaveParams { damping: 0.1, grid_damping: 1.5, ..EwaveParams::free(20.0) };
        let mut layer = EwaveLayer::new(n);
        let lo = -(n as f32) * cell * 0.5;
        let (dt, speed) = (1.0 / 30.0, 1.2);
        let mut x = -6.0;
        let steps = 240;
        let sponge = (n / 16) as u32;
        for k in 0..steps {
            let ramp = (k as f32 / 45.0).min(1.0);
            x += speed * ramp * dt;
            let d = WaterDisplacer {
                center: Vec2::new(x, 0.0),
                half: Vec2::new(0.4, 0.25),
                axis: Vec2::X,
                thickness: 0.15 * ramp,
                round: true,
                velocity: Vec2::new(speed, 0.0),
                rate: 0.0,
                momentum: 1.0,
                impulse: false,
            };
            for z in 0..n {
                for xi in 0..n {
                    let pos = Vec2::new(lo + (xi as f32 + 0.5) * cell, lo + (z as f32 + 0.5) * cell);
                    layer.inject(z * n + xi, d.thickness * d.profile(pos, cell), 0.0, &p, 1.0, true);
                }
            }
            layer.propagate(dt, cell, &p);
            for z in 0..n as i32 {
                for xi in 0..n as i32 {
                    let t = sponge_depth(glam::IVec2::new(xi, z), n as i32, sponge);
                    layer.mask_cell(z as usize * n + xi as usize, false, t, 0.0, dt);
                }
            }
        }
        // Energy behind the body by angle off its track.
        let kelvin = (1.0f32 / 3.0).asin().to_degrees();
        let (mut inside, mut ni, mut outside, mut no) = (0.0f64, 0, 0.0f64, 0);
        let mut by_angle = [0.0f64; 9];
        for z in 0..n {
            for xi in 0..n {
                let pos = Vec2::new(lo + (xi as f32 + 0.5) * cell, lo + (z as f32 + 0.5) * cell);
                let behind = x - pos.x;
                if !(1.5..=5.5).contains(&behind) {
                    continue;
                }
                let angle = (pos.y.abs() / behind).atan().to_degrees();
                let e = (layer.state[z * n + xi][0] as f64).powi(2);
                if angle < kelvin + 2.0 {
                    inside += e;
                    ni += 1;
                } else if angle > kelvin + 10.0 && angle < 70.0 {
                    outside += e;
                    no += 1;
                }
                if angle < 45.0 {
                    by_angle[(angle / 5.0) as usize] += e;
                }
            }
        }
        let (inside, outside) = (inside / ni as f64, outside / no.max(1) as f64);
        assert!(inside > 1e-9, "a wake: {inside}");
        assert!(outside < inside * 0.2, "the wake stays inside ~{kelvin:.1} deg: {inside:e} inside vs {outside:e} outside ({by_angle:?})");
    }
}
