//! Triangle budgets for the cluster path (GEO-5): coarsen the LOD error scale
//! instead of dropping frames.
//!
//! Two independent parts, both pure (no GPU, no allocation, unit-tested):
//!
//! - [`BudgetController`]: a PID-like loop on the always-on GPU frame time
//!   (`RenderStats::gpu_frame_ms`) against a target frame time (the display
//!   refresh / frame limiter cap). Its output is a global multiplier on the
//!   cluster LOD error target (>= 1: coarser). It works in log2 of the scale,
//!   has a dead band (hysteresis: it leaves a frame time that is close enough
//!   alone), coarsens faster than it refines, limits the change per frame
//!   (slew) and clamps to `[min_scale, max_scale]`.
//! - [`fair_shares`] / [`instance_scale`]: a per-instance fair share of a
//!   triangle budget (water-filling by weight, so an avatar cannot take more
//!   than its share, other players' untrusted avatars get a smaller weight
//!   than the local one) and the per-instance error scale that brings an
//!   instance's estimated triangles down to its share.
//!
//! The triangle estimate is a screen-space heuristic (surface area at the
//! error target), not a measurement; the GPU frame-time loop corrects what it
//! gets wrong. Documented limits: the GPU frame time includes everything, so a
//! frame slow for other reasons drives the scale to `max_scale` and holds it
//! there (bounded, and it refines again as soon as time frees up).

/// Tunables of the controller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BudgetConfig {
    /// Finest scale (1 = the cluster error target as set).
    pub min_scale: f32,
    /// Coarsest scale.
    pub max_scale: f32,
    /// Relative frame-time error inside which nothing changes (hysteresis).
    pub deadband: f32,
    pub kp: f32,
    pub ki: f32,
    pub kd: f32,
    /// Largest change of `log2(scale)` per frame when coarsening.
    pub max_step: f32,
    /// Refining is this many times slower than coarsening.
    pub refine_slowdown: f32,
    /// Exponential smoothing of the frame time (weight of the newest sample).
    pub smoothing: f32,
    /// Below this many drawn triangles coarsening cannot help: hold the scale.
    pub min_triangles: u32,
    /// Frames over which the "coarsening is not helping" detector compares the
    /// frame-time response with the scale it applied (0 disables it).
    pub futility_window: u32,
    /// Smallest rise of `log2(scale)` over a window that is judged at all.
    pub futility_min_rise: f32,
    /// Frame-time reduction (fraction of the window's starting time) a rise of
    /// one octave of scale must buy, or the window counts as futile.
    pub futility_gain_per_octave: f32,
    /// Frames the scale is held at `min_scale` after a futile window; doubles
    /// on every consecutive futile window (up to `futility_max_hold`).
    pub futility_hold: u32,
    pub futility_max_hold: u32,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            min_scale: 1.0,
            max_scale: 16.0,
            deadband: 0.06,
            kp: 0.30,
            ki: 0.03,
            kd: 0.10,
            max_step: 0.05,
            refine_slowdown: 4.0,
            smoothing: 0.2,
            min_triangles: 20_000,
            futility_window: 120,
            futility_min_rise: 0.5,
            futility_gain_per_octave: 0.02,
            futility_hold: 600,
            futility_max_hold: 9600,
        }
    }
}

/// The global error-scale controller.
#[derive(Clone, Debug)]
pub struct BudgetController {
    cfg: BudgetConfig,
    log_scale: f32,
    integral: f32,
    prev_err: f32,
    smoothed: f32,
    primed: bool,
    // "Coarsening is not helping" detector: the window it is judging, the hold
    // after a futile one and how many in a row there were.
    win_frames: u32,
    win_log: f32,
    win_ms: f32,
    hold_left: u32,
    futile_streak: u32,
    futile_total: u32,
}

impl BudgetController {
    pub fn new(cfg: BudgetConfig) -> Self {
        Self {
            cfg,
            log_scale: cfg.min_scale.max(1e-3).log2(),
            integral: 0.0,
            prev_err: 0.0,
            smoothed: 0.0,
            primed: false,
            win_frames: 0,
            win_log: 0.0,
            win_ms: 0.0,
            hold_left: 0,
            futile_streak: 0,
            futile_total: 0,
        }
    }

    /// True while the scale is held at the finest level after coarsening was
    /// found not to reduce the frame time.
    pub fn holding(&self) -> bool {
        self.hold_left > 0
    }

    /// How many windows have been judged futile since the last reset.
    pub fn futile_events(&self) -> u32 {
        self.futile_total
    }

    pub fn config(&self) -> &BudgetConfig {
        &self.cfg
    }

    /// The current error scale.
    pub fn scale(&self) -> f32 {
        self.log_scale.exp2()
    }

    /// Smoothed frame time the loop is acting on (ms; 0 before the first sample).
    pub fn smoothed_ms(&self) -> f32 {
        self.smoothed
    }

    /// Forget the history and return to the finest scale (scene change, setting toggled).
    pub fn reset(&mut self) {
        *self = Self::new(self.cfg);
    }

    /// One frame: `gpu_ms` the last completed frame's GPU time, `target_ms`
    /// the budget (1000 / target Hz), `triangles` what the cluster path drew.
    /// Returns the new scale. A missing or invalid measurement (0 ms: no
    /// timestamps, NaN) holds the scale.
    pub fn update(&mut self, gpu_ms: f32, target_ms: f32, triangles: u32) -> f32 {
        let c = self.cfg;
        if !(gpu_ms.is_finite() && gpu_ms > 0.0 && target_ms.is_finite() && target_ms > 0.0) {
            return self.scale();
        }
        self.smoothed = if self.primed { self.smoothed + c.smoothing * (gpu_ms - self.smoothed) } else { gpu_ms };
        self.primed = true;
        let err = (self.smoothed - target_ms) / target_ms;
        if self.hold_left > 0 {
            // The last coarsening bought nothing: stay fine until the hold runs out, then probe again.
            self.hold_left -= 1;
            self.integral = 0.0;
            self.prev_err = err;
            self.win_frames = 0;
            return self.scale();
        }
        let (lo, hi) = (c.min_scale.max(1e-3).log2(), c.max_scale.max(c.min_scale).log2());
        let mut delta = 0.0;
        if err.abs() <= c.deadband {
            // Close enough: bleed the integral so an old excursion cannot kick later.
            self.integral *= 0.9;
        } else {
            // Only the part outside the dead band drives the loop.
            let e = err - err.signum() * c.deadband;
            let saturated = (self.log_scale >= hi && e > 0.0) || (self.log_scale <= lo && e < 0.0);
            if !saturated {
                self.integral = (self.integral + e).clamp(-8.0, 8.0);
            }
            delta = c.kp * e + c.ki * self.integral + c.kd * (err - self.prev_err);
            if delta > 0.0 {
                // Coarsening that cannot help (nothing to simplify): hold.
                if triangles < c.min_triangles {
                    delta = 0.0;
                }
                delta = delta.min(c.max_step);
            } else {
                delta = delta.max(-c.max_step / c.refine_slowdown.max(1.0));
            }
        }
        self.prev_err = err;
        self.log_scale = (self.log_scale + delta).clamp(lo, hi);
        self.judge_response(err > c.deadband, lo);
        self.scale()
    }

    /// Frame time versus scale over a window of over-budget frames: when the
    /// scale rose a good deal and the frame time did not fall in proportion,
    /// the cost is not in the triangles (an unreachable target, a CPU or fill
    /// bound frame); go back to the finest scale and hold there.
    fn judge_response(&mut self, over_budget: bool, lo: f32) {
        let c = self.cfg;
        if c.futility_window == 0 || !over_budget {
            // Within budget (or detector off): nothing to judge; a recovered frame forgets the streak.
            self.win_frames = 0;
            if !over_budget {
                self.futile_streak = 0;
            }
            return;
        }
        if self.win_frames == 0 {
            self.win_log = self.log_scale;
            self.win_ms = self.smoothed;
        }
        self.win_frames += 1;
        if self.win_frames < c.futility_window {
            return;
        }
        self.win_frames = 0;
        let rise = self.log_scale - self.win_log;
        if rise < c.futility_min_rise {
            return;
        }
        let gain = (self.win_ms - self.smoothed) / self.win_ms.max(1e-3);
        if gain >= c.futility_gain_per_octave * rise {
            self.futile_streak = 0;
            return;
        }
        let shift = self.futile_streak.min(16);
        self.hold_left = c.futility_hold.saturating_mul(1u32 << shift).min(c.futility_max_hold.max(c.futility_hold));
        self.futile_streak += 1;
        self.futile_total += 1;
        if self.futile_total == 1 {
            tracing::warn!(
                "Zestlet budget: coarsening the LOD error {:.1}x cut the frame time only {:.0}% ({:.1} ms); the cost is not in the triangles, holding the finest scale for {} frames",
                self.log_scale.exp2(),
                gain * 100.0,
                self.smoothed,
                self.hold_left
            );
        }
        self.log_scale = lo;
        self.integral = 0.0;
    }
}

/// Weight of an instance in the fair share.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShareClass {
    /// The local player's own avatar.
    Local,
    /// Everything else: other players' avatars that passed the safety policy, props.
    #[default]
    Trusted,
    /// Other players' avatars the policy flags (over a rank, unknown source).
    Untrusted,
}

impl ShareClass {
    pub fn weight(self) -> f32 {
        match self {
            Self::Local => 4.0,
            Self::Trusted => 1.0,
            Self::Untrusted => 0.5,
        }
    }
}

/// Split `budget` triangles over instances by `weight`, water-filling: nobody
/// gets more than its `demand`, and what an instance leaves unused is shared
/// out among the others by weight. `out[i]` is instance `i`'s allowance;
/// the sum never exceeds `budget`. Slices must have the same length.
pub fn fair_shares(budget: f32, demand: &[f32], weight: &[f32], out: &mut [f32]) {
    let n = demand.len().min(weight.len()).min(out.len());
    // NaN marks "still sharing"; settled instances hold their demand.
    out[..n].fill(f32::NAN);
    let mut remaining = budget.max(0.0);
    loop {
        let wsum: f32 = (0..n).filter(|&i| out[i].is_nan()).map(|i| weight[i].max(0.0)).sum();
        if wsum <= 0.0 {
            break;
        }
        let mut settled = false;
        for i in 0..n {
            if !out[i].is_nan() {
                continue;
            }
            let share = remaining * weight[i].max(0.0) / wsum;
            if demand[i].max(0.0) <= share {
                out[i] = demand[i].max(0.0);
                remaining -= out[i];
                settled = true;
            }
        }
        if !settled {
            // Nobody fits its weighted share: satisfy the smallest demand the
            // remaining budget can still afford outright, so an instance
            // whose demand is under the budget is never coarsened just
            // because a much larger demand dominates the split.
            let mut best: Option<usize> = None;
            for i in 0..n {
                if out[i].is_nan() && demand[i].max(0.0) <= remaining {
                    best = Some(match best {
                        Some(b) if demand[i] < demand[b] => i,
                        Some(b) => b,
                        None => i,
                    });
                }
            }
            if let Some(b) = best {
                out[b] = demand[b].max(0.0);
                remaining -= out[b];
                continue;
            }
            // Nothing fits at all: split what is left by weight.
            for i in 0..n {
                if out[i].is_nan() {
                    out[i] = remaining * weight[i].max(0.0) / wsum;
                }
            }
            return;
        }
    }
    // Zero-weight leftovers get nothing.
    for o in out[..n].iter_mut().filter(|o| o.is_nan()) {
        *o = 0.0;
    }
}

/// Estimated triangles an instance draws: its surface at `target_px` error
/// covers about `area_px / (k * target_px^2)` triangles, never more than the
/// finest level has. `radius_px` is the instance bounding sphere's projected
/// radius. A heuristic (triangle edge ~ 1.4 * target px), corrected by the
/// frame-time loop.
pub fn estimate_triangles(leaf_triangles: u32, radius_px: f32, target_px: f32) -> f32 {
    if !(radius_px.is_finite() && target_px.is_finite()) || target_px <= 0.0 {
        return leaf_triangles as f32;
    }
    // A closed surface shows about its sphere's cross section twice (front and back faces
    // are both submitted before the backface cone culls them).
    let area = std::f32::consts::PI * radius_px * radius_px * 2.0;
    (area / (2.0 * target_px * target_px)).min(leaf_triangles as f32).max(0.0)
}

/// Error scale that brings `estimate_at_one` triangles (estimated at scale 1)
/// down to `allowance`. Triangles fall with the square of the error, so the
/// scale is `sqrt(estimate / allowance)`; never below 1 or above `max_scale`.
pub fn instance_scale(estimate_at_one: f32, allowance: f32, max_scale: f32) -> f32 {
    if estimate_at_one <= allowance || estimate_at_one <= 0.0 {
        return 1.0;
    }
    if allowance <= 0.0 {
        return max_scale.max(1.0);
    }
    (estimate_at_one / allowance).sqrt().clamp(1.0, max_scale.max(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plant: frame time = fixed + per-triangle cost, triangles ~ 1/scale^2.
    fn frame_ms(scale: f32, fixed: f32, tris_at_one: f32, ms_per_mtri: f32) -> (f32, u32) {
        let tris = tris_at_one / (scale * scale);
        (fixed + ms_per_mtri * tris / 1e6, tris as u32)
    }

    #[test]
    fn converges_into_the_dead_band_without_oscillating() {
        let mut c = BudgetController::new(BudgetConfig::default());
        let target = 11.1;
        let (mut ms, mut tris) = frame_ms(1.0, 3.0, 30e6, 0.5); // 18 ms at scale 1
        let mut signs = 0;
        let mut last_delta = 0.0f32;
        let mut prev = c.scale();
        let max_step = c.config().max_step;
        for _ in 0..600 {
            let s = c.update(ms, target, tris);
            let d = s.log2() - prev.log2();
            assert!(d.abs() <= max_step + 1e-6, "slew limit: {d}");
            if d * last_delta < 0.0 && d.abs() > 1e-4 && last_delta.abs() > 1e-4 {
                signs += 1;
            }
            if d.abs() > 1e-4 {
                last_delta = d;
            }
            prev = s;
            (ms, tris) = frame_ms(s, 3.0, 30e6, 0.5);
        }
        let (final_ms, _) = frame_ms(c.scale(), 3.0, 30e6, 0.5);
        assert!((final_ms - target).abs() / target < 0.10, "settled at {final_ms} ms for a {target} ms target (scale {})", c.scale());
        assert!(c.scale() > 1.2 && c.scale() < 16.0);
        assert!(signs <= 6, "the scale changed direction {signs} times");
    }

    #[test]
    fn leaves_a_frame_inside_the_dead_band_alone() {
        let mut c = BudgetController::new(BudgetConfig::default());
        for _ in 0..200 {
            c.update(11.1 * 1.04, 11.1, 5_000_000);
        }
        assert_eq!(c.scale(), 1.0);
        for _ in 0..200 {
            c.update(11.1 * 0.95, 11.1, 5_000_000);
        }
        assert_eq!(c.scale(), 1.0);
    }

    #[test]
    fn refines_again_when_the_load_drops_and_clamps() {
        // The constant 40 ms input is "hopeless but responsive" here: detector off.
        let mut c = BudgetController::new(BudgetConfig { futility_window: 0, ..Default::default() });
        for _ in 0..400 {
            c.update(40.0, 11.1, 9_000_000); // hopeless: bounded
        }
        assert_eq!(c.scale(), c.config().max_scale);
        for _ in 0..4000 {
            c.update(3.0, 11.1, 9_000_000);
        }
        assert_eq!(c.scale(), c.config().min_scale, "refines back to the finest scale");
        // Never outside the limits, whatever the input.
        let mut c = BudgetController::new(BudgetConfig { min_scale: 1.5, max_scale: 4.0, ..Default::default() });
        for i in 0..1000 {
            let ms = if i % 7 < 3 { 100.0 } else { 0.5 };
            let s = c.update(ms, 11.1, 1_000_000);
            assert!((1.5..=4.0).contains(&s));
        }
    }

    #[test]
    fn refining_is_slower_than_coarsening() {
        let cfg = BudgetConfig::default();
        let mut up = BudgetController::new(cfg);
        let a = up.update(30.0, 11.1, 5_000_000).log2();
        let mut down = BudgetController::new(cfg);
        for _ in 0..40 {
            down.update(30.0, 11.1, 5_000_000);
        }
        let before = down.scale().log2();
        // Prime the smoothing at a low time first so the comparison is of the step size.
        let mut d = BudgetController::new(cfg);
        d.log_scale = before;
        d.primed = true;
        d.smoothed = 3.0;
        d.prev_err = (3.0 - 11.1) / 11.1;
        let b = before - d.update(3.0, 11.1, 5_000_000).log2();
        assert!(a <= cfg.max_step + 1e-6);
        assert!(b <= cfg.max_step / cfg.refine_slowdown + 1e-6);
        assert!(b < a);
    }

    #[test]
    fn invalid_measurements_and_tiny_scenes_hold_the_scale() {
        let mut c = BudgetController::new(BudgetConfig::default());
        for _ in 0..50 {
            c.update(30.0, 11.1, 5_000_000);
        }
        let s = c.scale();
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(c.update(bad, 11.1, 5_000_000), s);
        }
        assert_eq!(c.update(30.0, 0.0, 5_000_000), s);
        assert_eq!(c.update(30.0, f32::NAN, 5_000_000), s);
        // Few triangles: coarsening cannot help, so it does not coarsen further.
        let mut c = BudgetController::new(BudgetConfig::default());
        for _ in 0..100 {
            c.update(30.0, 11.1, 100);
        }
        assert_eq!(c.scale(), 1.0);
        c.reset();
        assert_eq!(c.scale(), 1.0);
        assert_eq!(c.smoothed_ms(), 0.0);
    }

    /// Run `frames` frames of the plant `fixed + ms_per_mtri * tris / 1e6` and
    /// return the scale after each frame.
    fn run(c: &mut BudgetController, frames: usize, target: f32, fixed: f32, tris_at_one: f32, ms_per_mtri: f32) -> Vec<f32> {
        let mut s = c.scale();
        (0..frames)
            .map(|_| {
                let (ms, tris) = frame_ms(s, fixed, tris_at_one, ms_per_mtri);
                s = c.update(ms, target, tris);
                s
            })
            .collect()
    }

    #[test]
    fn unreachable_target_backs_off_to_the_finest_scale_and_holds() {
        let mut c = BudgetController::new(BudgetConfig::default());
        // 25 ms of fixed cost against an 11 ms target: no scale can reach it, the triangles are 1% of the frame.
        let hist = run(&mut c, 3000, 11.1, 25.0, 30e6, 0.05);
        let peak = hist.iter().cloned().fold(0.0f32, f32::max);
        assert!(peak > 1.3, "it probes upwards first (peak {peak})");
        assert!(c.futile_events() >= 1);
        // It spends most of the run held at the finest scale, never parked at the ceiling
        let held = hist.iter().filter(|&&s| s == 1.0).count();
        assert!(held > 1500, "only {held} of 3000 frames at the finest scale");
        let at_ceiling = hist.iter().filter(|&&s| s >= 15.99).count();
        // Each probe may touch the ceiling for the tail of one judging window, no longer.
        assert!(at_ceiling <= 2 * c.config().futility_window as usize, "{at_ceiling} frames at the 16x ceiling");
    }

    #[test]
    fn futile_holds_back_off_exponentially_and_reset_clears() {
        let cfg = BudgetConfig::default();
        let mut c = BudgetController::new(cfg);
        let mut holds = Vec::new();
        let mut was = false;
        let mut s = 1.0;
        for _ in 0..8000 {
            let (ms, tris) = frame_ms(s, 25.0, 30e6, 0.05);
            s = c.update(ms, 11.1, tris);
            if c.holding() && !was {
                holds.push(c.hold_left);
            }
            was = c.holding();
        }
        assert!(holds.len() >= 3, "{holds:?}");
        assert_eq!(holds[0], cfg.futility_hold);
        assert!(holds[1] > holds[0] && holds[2] >= holds[1], "{holds:?}");
        assert!(holds.iter().all(|&h| h <= cfg.futility_max_hold));
        c.reset();
        assert!(!c.holding() && c.futile_events() == 0);
    }

    #[test]
    fn a_responsive_plant_is_never_called_futile() {
        // The triangles are the frame: the existing convergence case, plus a weakly responsive one.
        for (fixed, mtri_ms) in [(3.0, 0.5), (14.0, 0.5)] {
            let mut c = BudgetController::new(BudgetConfig::default());
            let hist = run(&mut c, 1500, 11.1, fixed, 30e6, mtri_ms);
            assert_eq!(c.futile_events(), 0, "fixed {fixed}");
            assert!(hist.last().unwrap() > &1.2);
        }
        // A hopeless but responsive load that saturates at the ceiling is bounded, not futile.
        let mut c = BudgetController::new(BudgetConfig::default());
        run(&mut c, 2000, 11.1, 3.0, 600e6, 0.5);
        assert_eq!(c.futile_events(), 0);
    }

    #[test]
    fn detector_probes_again_after_the_hold_and_recovers_when_the_load_becomes_triangle_bound() {
        let mut c = BudgetController::new(BudgetConfig::default());
        run(&mut c, 1500, 11.1, 25.0, 30e6, 0.05);
        assert!(c.futile_events() >= 1);
        // The fixed cost goes away (e.g. a loading spike ended) but the scene is triangle heavy: it coarsens again.
        let hist = run(&mut c, 6000, 11.1, 3.0, 30e6, 0.5);
        assert!(hist.iter().cloned().fold(0.0f32, f32::max) > 1.2);
    }

    #[test]
    fn fair_shares_water_fill() {
        let mut out = [0.0f32; 4];
        // Light instances keep their demand; the heavy one gets the rest.
        fair_shares(1000.0, &[100.0, 100.0, 5000.0, 50.0], &[1.0, 1.0, 1.0, 1.0], &mut out);
        assert_eq!(out[0], 100.0);
        assert_eq!(out[1], 100.0);
        assert_eq!(out[3], 50.0);
        assert!((out[2] - 750.0).abs() < 1e-3);
        // Budget bigger than the demand: everyone gets what they asked for.
        fair_shares(1e9, &[10.0, 20.0, 30.0, 40.0], &[1.0, 2.0, 3.0, 4.0], &mut out);
        assert_eq!(out, [10.0, 20.0, 30.0, 40.0]);
        // Saturated: shares follow the weights, the sum is the budget, nobody exceeds demand.
        fair_shares(900.0, &[1e6, 1e6, 1e6, 1e6], &[4.0, 1.0, 0.5, 0.5], &mut out);
        assert!((out[0] - 600.0).abs() < 1e-2 && (out[1] - 150.0).abs() < 1e-2 && (out[2] - 75.0).abs() < 1e-2);
        assert!((out.iter().sum::<f32>() - 900.0).abs() < 1e-2);
        // A zero-weight instance gets nothing, an empty budget gives nothing.
        fair_shares(100.0, &[50.0, 50.0], &[0.0, 1.0], &mut out[..2]);
        assert_eq!(out[..2], [0.0, 50.0]);
        fair_shares(0.0, &[50.0, 50.0], &[1.0, 1.0], &mut out[..2]);
        assert_eq!(out[..2], [0.0, 0.0]);
    }

    #[test]
    fn fair_shares_random_invariants() {
        let mut seed = 7u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        for _ in 0..500 {
            let n = 1 + (rnd() * 30.0) as usize;
            let demand: Vec<f32> = (0..n).map(|_| rnd() * 1e5).collect();
            let weight: Vec<f32> = (0..n).map(|_| [4.0, 1.0, 0.5][(rnd() * 3.0) as usize % 3]).collect();
            let budget = rnd() * 2e5;
            let mut out = vec![0.0; n];
            fair_shares(budget, &demand, &weight, &mut out);
            assert!(out.iter().sum::<f32>() <= budget * 1.0001 + 1e-3);
            for i in 0..n {
                assert!(out[i] >= 0.0 && out[i] <= demand[i] * 1.0001 + 1e-3);
            }
            // Max-min fairness: an instance below its demand gets at least its weight's share of
            // what any other capped instance gets.
            for i in 0..n {
                for j in 0..n {
                    if out[i] < demand[i] * 0.999 && out[j] < demand[j] * 0.999 {
                        assert!((out[i] / weight[i] - out[j] / weight[j]).abs() <= 1e-2 * (out[i] / weight[i]).max(1.0));
                    }
                }
            }
        }
    }

    #[test]
    fn local_avatar_outweighs_untrusted_ones() {
        assert!(ShareClass::Local.weight() > ShareClass::Trusted.weight());
        assert!(ShareClass::Trusted.weight() > ShareClass::Untrusted.weight());
        let mut out = [0.0; 3];
        let w = [ShareClass::Local.weight(), ShareClass::Untrusted.weight(), ShareClass::Untrusted.weight()];
        fair_shares(1_000_000.0, &[900_000.0; 3], &w, &mut out);
        assert!(out[0] > 8.0 * out[1] * 0.99 && out[1] == out[2]);
    }

    #[test]
    fn estimate_and_scale_are_monotone() {
        let mut prev = f32::MAX;
        for r in [400.0f32, 200.0, 100.0, 50.0, 10.0, 1.0] {
            let t = estimate_triangles(1_000_000, r, 1.0);
            assert!(t <= prev && t <= 1_000_000.0);
            prev = t;
        }
        assert_eq!(estimate_triangles(5000, 2000.0, 1.0), 5000.0);
        assert!(estimate_triangles(1_000_000, 100.0, 2.0) < estimate_triangles(1_000_000, 100.0, 1.0));
        assert_eq!(estimate_triangles(77, f32::NAN, 1.0), 77.0);
        assert_eq!(instance_scale(100.0, 200.0, 16.0), 1.0);
        assert_eq!(instance_scale(400.0, 100.0, 16.0), 2.0);
        assert_eq!(instance_scale(1e9, 1.0, 16.0), 16.0);
        assert_eq!(instance_scale(100.0, 0.0, 16.0), 16.0);
        // Using the scale really brings the estimate under the allowance.
        let est1 = estimate_triangles(2_000_000, 300.0, 1.0);
        let s = instance_scale(est1, est1 / 6.0, 16.0);
        let est_s = estimate_triangles(2_000_000, 300.0, s);
        assert!(est_s <= est1 / 6.0 * 1.01);
    }
}
