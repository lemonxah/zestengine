//! 2D inverse kinematics (2D-7): pure solvers over a planar bone chain, no
//! scene types. The runtime (`engine/src/scene/rig2d.rs`) reads the chain's
//! joint angles from the bone objects, solves, and writes the rotations back.
//!
//! Behaviour (Unity's 2D IK package ships the same three solver families; this
//! is an independent implementation of the textbook algorithms):
//!
//! * **CCD** (cyclic coordinate descent): from the tip bone to the root, rotate
//!   each joint so the tip swings toward the target, clamping to the joint's
//!   limit, repeated until the tip is within `tolerance` or `iterations` ends.
//!   Limits are exact because every step clamps.
//! * **FABRIK**: backward/forward reaching on joint *positions* (bone lengths
//!   stay exact), then the positions are converted back to relative angles. FABRIK
//!   itself knows nothing about limits, so when a chain has any, the angles are
//!   clamped and a few CCD passes polish the result (limits stay exact; the
//!   position is then only as good as the limits allow).
//! * **Limb** (two bones): closed form with the law of cosines, a `flip` bend
//!   direction, and a straight limb toward an unreachable target. With limits
//!   it also tries the other bend and keeps the better one.
//!
//! A chain is a base point and heading plus bones with a length and an angle
//! *relative to the parent bone* (the first bone is relative to `base_angle`).
//! Bones point along their local +X, like the rig bones in `rig2d.rs`.

use std::f32::consts::PI;

/// One bone of a chain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IkBone {
    pub length: f32,
    /// Rotation relative to the parent bone, radians.
    pub angle: f32,
    /// Allowed relative angle range `(min, max)`, radians; `None` = free.
    pub limit: Option<(f32, f32)>,
}

impl IkBone {
    pub fn new(length: f32, angle: f32) -> Self {
        Self { length, angle, limit: None }
    }
}

/// Outcome of a solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IkResult {
    pub iterations: u32,
    /// Distance from the tip to the target after solving.
    pub error: f32,
    /// `error <= tolerance`.
    pub reached: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IkChain {
    pub base: [f32; 2],
    pub base_angle: f32,
    pub bones: Vec<IkBone>,
}

/// Wrap an angle to `(-PI, PI]`.
pub fn wrap_angle(a: f32) -> f32 {
    let mut a = a % (2.0 * PI);
    if a > PI {
        a -= 2.0 * PI;
    } else if a <= -PI {
        a += 2.0 * PI;
    }
    a
}

fn clamp_limit(a: f32, limit: Option<(f32, f32)>) -> f32 {
    let a = wrap_angle(a);
    match limit {
        Some((lo, hi)) => a.clamp(lo.min(hi), lo.max(hi)),
        None => a,
    }
}

fn sub(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn len(a: [f32; 2]) -> f32 {
    (a[0] * a[0] + a[1] * a[1]).sqrt()
}

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    len(sub(a, b))
}

impl IkChain {
    pub fn new(base: [f32; 2], base_angle: f32, bones: Vec<IkBone>) -> Self {
        Self { base, base_angle, bones }
    }

    /// Sum of the bone lengths: the farthest the tip can be from the base.
    pub fn reach(&self) -> f32 {
        self.bones.iter().map(|b| b.length).sum()
    }

    /// Joint positions, base first (`bones.len() + 1` points).
    pub fn joints(&self) -> Vec<[f32; 2]> {
        let mut out = Vec::with_capacity(self.bones.len() + 1);
        let mut p = self.base;
        let mut a = self.base_angle;
        out.push(p);
        for b in &self.bones {
            a += b.angle;
            p = [p[0] + a.cos() * b.length, p[1] + a.sin() * b.length];
            out.push(p);
        }
        out
    }

    pub fn tip(&self) -> [f32; 2] {
        *self.joints().last().unwrap_or(&self.base)
    }

    /// Absolute angle of bone `i` (base angle plus the relative angles up to it).
    pub fn absolute_angle(&self, i: usize) -> f32 {
        self.base_angle + self.bones[..=i.min(self.bones.len().saturating_sub(1))].iter().map(|b| b.angle).sum::<f32>()
    }

    /// Clamp every relative angle to its limit.
    pub fn enforce_limits(&mut self) {
        for b in &mut self.bones {
            b.angle = clamp_limit(b.angle, b.limit);
        }
    }

    /// Set the relative angles from joint positions (as FABRIK produces them).
    pub fn set_from_joints(&mut self, joints: &[[f32; 2]]) {
        let mut parent = self.base_angle;
        for (i, b) in self.bones.iter_mut().enumerate() {
            let (Some(&p0), Some(&p1)) = (joints.get(i), joints.get(i + 1)) else { break };
            let d = sub(p1, p0);
            if len(d) < 1e-9 {
                continue; // keep the old angle for a collapsed bone
            }
            let abs = d[1].atan2(d[0]);
            b.angle = clamp_limit(abs - parent, b.limit);
            parent += b.angle;
        }
    }

    fn finish(&self, target: [f32; 2], tolerance: f32, iterations: u32) -> IkResult {
        let error = dist(self.tip(), target);
        IkResult { iterations, error, reached: error <= tolerance }
    }

    /// Cyclic coordinate descent.
    pub fn solve_ccd(&mut self, target: [f32; 2], iterations: u32, tolerance: f32) -> IkResult {
        if self.bones.is_empty() {
            return self.finish(target, tolerance, 0);
        }
        let mut done = 0;
        for it in 0..iterations.max(1) {
            done = it + 1;
            if dist(self.tip(), target) <= tolerance {
                break;
            }
            for i in (0..self.bones.len()).rev() {
                let joints = self.joints();
                let pivot = joints[i];
                let to_tip = sub(*joints.last().unwrap(), pivot);
                let to_target = sub(target, pivot);
                if len(to_tip) < 1e-9 || len(to_target) < 1e-9 {
                    continue;
                }
                let delta = wrap_angle(to_target[1].atan2(to_target[0]) - to_tip[1].atan2(to_tip[0]));
                let b = &mut self.bones[i];
                b.angle = clamp_limit(b.angle + delta, b.limit);
            }
        }
        self.finish(target, tolerance, done)
    }

    /// Forward and backward reaching.
    pub fn solve_fabrik(&mut self, target: [f32; 2], iterations: u32, tolerance: f32) -> IkResult {
        if self.bones.is_empty() {
            return self.finish(target, tolerance, 0);
        }
        let lengths: Vec<f32> = self.bones.iter().map(|b| b.length).collect();
        let mut p = self.joints();
        let base = self.base;
        let mut done = 0;
        if dist(base, target) >= self.reach() {
            // Out of reach: straight toward the target.
            let d = sub(target, base);
            let l = len(d).max(1e-9);
            let u = [d[0] / l, d[1] / l];
            for i in 0..lengths.len() {
                p[i + 1] = [p[i][0] + u[0] * lengths[i], p[i][1] + u[1] * lengths[i]];
            }
        } else {
            let n = p.len();
            for it in 0..iterations.max(1) {
                done = it + 1;
                if dist(p[n - 1], target) <= tolerance {
                    break;
                }
                p[n - 1] = target;
                for i in (0..n - 1).rev() {
                    let d = sub(p[i], p[i + 1]);
                    let l = len(d).max(1e-9);
                    p[i] = [p[i + 1][0] + d[0] / l * lengths[i], p[i + 1][1] + d[1] / l * lengths[i]];
                }
                p[0] = base;
                for i in 0..n - 1 {
                    let d = sub(p[i + 1], p[i]);
                    let l = len(d).max(1e-9);
                    p[i + 1] = [p[i][0] + d[0] / l * lengths[i], p[i][1] + d[1] / l * lengths[i]];
                }
            }
        }
        self.set_from_joints(&p);
        if self.bones.iter().any(|b| b.limit.is_some()) {
            // Limits clamped the angles: polish with CCD, which respects them.
            let left = iterations.saturating_sub(done).max(8);
            self.solve_ccd(target, left, tolerance);
        }
        self.finish(target, tolerance, done)
    }

    /// Closed-form two-bone solve. `flip` picks the other bend direction.
    /// A chain that is not exactly two bones falls back to CCD.
    pub fn solve_limb(&mut self, target: [f32; 2], flip: bool, tolerance: f32) -> IkResult {
        if self.bones.len() != 2 {
            return self.solve_ccd(target, 32, tolerance);
        }
        let start = self.bones.clone();
        let mut best: Option<(f32, Vec<IkBone>)> = None;
        let tries: &[bool] = if self.bones.iter().any(|b| b.limit.is_some()) { &[false, true] } else { &[false] };
        for &alt in tries {
            self.bones = start.clone();
            self.limb_pose(target, flip ^ alt);
            self.enforce_limits();
            let e = dist(self.tip(), target);
            if best.as_ref().is_none_or(|(be, _)| e < *be - 1e-6) {
                best = Some((e, self.bones.clone()));
            }
        }
        if let Some((_, b)) = best {
            self.bones = b;
        }
        self.finish(target, tolerance, 1)
    }

    fn limb_pose(&mut self, target: [f32; 2], flip: bool) {
        let (l1, l2) = (self.bones[0].length, self.bones[1].length);
        let d = sub(target, self.base);
        let dist_t = len(d);
        let reach = l1 + l2;
        let min_reach = (l1 - l2).abs();
        let dc = dist_t.clamp(min_reach + 1e-6, (reach - 1e-6).max(min_reach + 1e-6));
        let heading = if dist_t < 1e-9 { self.base_angle } else { d[1].atan2(d[0]) };
        // Interior angle at the elbow from the law of cosines.
        let cos_e = ((l1 * l1 + l2 * l2 - dc * dc) / (2.0 * l1 * l2).max(1e-9)).clamp(-1.0, 1.0);
        let elbow = PI - cos_e.acos(); // bend away from straight
        let a2 = if flip { -elbow } else { elbow };
        let a1_abs = heading - (l2 * a2.sin()).atan2(l1 + l2 * a2.cos());
        self.bones[0].angle = wrap_angle(a1_abs - self.base_angle);
        self.bones[1].angle = a2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arm(n: usize) -> IkChain {
        IkChain::new([0.0, 0.0], 0.0, (0..n).map(|_| IkBone::new(1.0, 0.1)).collect())
    }

    fn lengths_ok(c: &IkChain) -> bool {
        let j = c.joints();
        c.bones.iter().enumerate().all(|(i, b)| (dist(j[i], j[i + 1]) - b.length).abs() < 1e-4)
    }

    #[test]
    fn forward_kinematics() {
        let c = IkChain::new([1.0, 2.0], PI / 2.0, vec![IkBone::new(2.0, 0.0), IkBone::new(1.0, -PI / 2.0)]);
        let j = c.joints();
        assert!(dist(j[1], [1.0, 4.0]) < 1e-5);
        assert!(dist(j[2], [2.0, 4.0]) < 1e-5);
        assert!((c.absolute_angle(1)).abs() < 1e-5);
    }

    #[test]
    fn ccd_converges_on_reachable_targets() {
        for &t in &[[1.5, 1.0], [-1.0, 1.5], [0.5, -2.0], [2.5, 0.2], [0.0, 2.9]] {
            let mut c = arm(3);
            let r = c.solve_ccd(t, 400, 1e-3);
            assert!(r.reached, "target {t:?} error {}", r.error);
            assert!(lengths_ok(&c));
        }
    }

    #[test]
    fn fabrik_converges_on_reachable_targets() {
        for &t in &[[1.5, 1.0], [-1.0, 1.5], [0.5, -2.0], [2.5, 0.2], [0.0, 2.9]] {
            let mut c = arm(3);
            let r = c.solve_fabrik(t, 64, 1e-3);
            assert!(r.reached, "target {t:?} error {}", r.error);
            assert!(lengths_ok(&c));
        }
    }

    #[test]
    fn unreachable_target_stretches_straight() {
        for solver in 0..2 {
            let mut c = arm(3);
            let t = [10.0, 0.0];
            let r = if solver == 0 { c.solve_fabrik(t, 32, 1e-3) } else { c.solve_ccd(t, 64, 1e-3) };
            assert!(!r.reached);
            let tip = c.tip();
            assert!((tip[0] - 3.0).abs() < 1e-2 && tip[1].abs() < 1e-2, "solver {solver} tip {tip:?}");
        }
    }

    #[test]
    fn limits_are_never_violated() {
        let lim = Some((-0.5, 0.5));
        for solver in 0..3 {
            let mut c = arm(3);
            for b in &mut c.bones {
                b.limit = lim;
            }
            for &t in &[[0.0, 2.5], [-2.0, 1.0], [1.0, -1.0]] {
                match solver {
                    0 => c.solve_ccd(t, 64, 1e-3),
                    1 => c.solve_fabrik(t, 64, 1e-3),
                    _ => {
                        c.bones.truncate(2);
                        c.solve_limb(t, false, 1e-3)
                    }
                };
                for b in &c.bones {
                    assert!(b.angle >= -0.5 - 1e-5 && b.angle <= 0.5 + 1e-5, "solver {solver} angle {}", b.angle);
                }
                assert!(lengths_ok(&c));
            }
        }
    }

    #[test]
    fn limited_chain_reaches_what_its_limits_allow() {
        // A hinge limited to [0, 0.2] rad cannot bend down: the tip stays above the target.
        let mut c = IkChain::new([0.0, 0.0], 0.0, vec![IkBone { length: 1.0, angle: 0.1, limit: Some((0.0, 0.2)) }, IkBone { length: 1.0, angle: 0.1, limit: Some((0.0, 0.2)) }]);
        let r = c.solve_ccd([1.0, -1.0], 32, 1e-3);
        assert!(!r.reached);
        assert!(c.bones.iter().all(|b| b.angle >= 0.0));
    }

    #[test]
    fn limb_is_exact_and_flip_mirrors_the_bend() {
        let mut a = IkChain::new([0.0, 0.0], 0.0, vec![IkBone::new(1.0, 0.0), IkBone::new(1.0, 0.0)]);
        let mut b = a.clone();
        let t = [1.2, 0.7];
        assert!(a.solve_limb(t, false, 1e-4).reached);
        assert!(b.solve_limb(t, true, 1e-4).reached);
        assert!(dist(a.tip(), t) < 1e-4 && dist(b.tip(), t) < 1e-4);
        assert!(a.bones[1].angle * b.bones[1].angle < 0.0, "elbows bend opposite ways");
        assert!((a.bones[1].angle + b.bones[1].angle).abs() < 1e-5);
    }

    #[test]
    fn limb_out_of_reach_is_straight_toward_the_target() {
        let mut c = IkChain::new([0.0, 0.0], 0.0, vec![IkBone::new(1.0, 0.3), IkBone::new(1.5, 0.3)]);
        c.solve_limb([0.0, 10.0], false, 1e-3);
        let tip = c.tip();
        assert!(tip[0].abs() < 1e-3 && (tip[1] - 2.5).abs() < 1e-3, "{tip:?}");
    }

    #[test]
    fn limb_target_at_base_does_not_nan() {
        let mut c = IkChain::new([0.0, 0.0], 0.0, vec![IkBone::new(1.0, 0.0), IkBone::new(1.0, 0.0)]);
        c.solve_limb([0.0, 0.0], false, 1e-3);
        assert!(c.bones.iter().all(|b| b.angle.is_finite()));
    }

    #[test]
    fn solving_from_the_solved_pose_is_stable() {
        let mut c = arm(4);
        c.solve_fabrik([1.0, 2.0], 64, 1e-4);
        let before = c.clone();
        c.solve_fabrik([1.0, 2.0], 64, 1e-4);
        for (a, b) in before.bones.iter().zip(&c.bones) {
            assert!((a.angle - b.angle).abs() < 1e-3);
        }
    }

    #[test]
    fn wrap_angle_range() {
        for k in -20..20 {
            let a = wrap_angle(k as f32 * 0.9);
            assert!(a > -PI - 1e-6 && a <= PI + 1e-6);
        }
    }
}
