//! Spline edit operations of the Scene-view tool, as a pure model: every
//! operation applies to a [`SpriteShapeRenderer`] and returns its exact
//! inverse, so an undo entry is just the inverse stored on a stack. (The editor
//! also records the frame's whole-component diff, so these are the testable
//! core of what the tool does, not a second undo system.)

use glam::Vec2;

use super::spline::{bezier_at, segment_bezier};
use super::{ShapePoint, ShapeTangentMode, SpriteShapeRenderer};

/// One spline edit with an exact inverse.
#[derive(Clone, PartialEq, Debug)]
pub enum SplineOp {
    /// Insert `point` so it ends up at `index` (`index == len` appends).
    Insert {
        /// Where the point ends up.
        index: usize,
        /// The point.
        point: ShapePoint,
    },
    /// Remove the point at `index`.
    Remove {
        /// Index of the point to remove.
        index: usize,
    },
    /// Replace the point at `index` (move, tangents, mode, height, corner, variant).
    Set {
        /// Index of the point to replace.
        index: usize,
        /// Its new state.
        point: ShapePoint,
    },
    /// Open or close the spline.
    SetClosed(bool),
    /// Several operations as one undo step (a segment split changes three points).
    Batch(Vec<SplineOp>),
}

/// Apply `op`; returns the operation that undoes it, or `None` (and changes
/// nothing) when an index is out of range.
pub fn apply_op(shape: &mut SpriteShapeRenderer, op: &SplineOp) -> Option<SplineOp> {
    match op {
        SplineOp::Insert { index, point } => {
            if *index > shape.points.len() {
                return None;
            }
            shape.points.insert(*index, *point);
            Some(SplineOp::Remove { index: *index })
        }
        SplineOp::Remove { index } => {
            if *index >= shape.points.len() {
                return None;
            }
            let point = shape.points.remove(*index);
            Some(SplineOp::Insert { index: *index, point })
        }
        SplineOp::Set { index, point } => {
            let slot = shape.points.get_mut(*index)?;
            let old = std::mem::replace(slot, *point);
            Some(SplineOp::Set { index: *index, point: old })
        }
        SplineOp::SetClosed(c) => {
            let old = std::mem::replace(&mut shape.closed, *c);
            Some(SplineOp::SetClosed(old))
        }
        SplineOp::Batch(ops) => {
            let mut inverse = Vec::with_capacity(ops.len());
            for o in ops {
                match apply_op(shape, o) {
                    Some(i) => inverse.push(i),
                    None => {
                        // Roll back what already applied.
                        for i in inverse.iter().rev() {
                            apply_op(shape, i);
                        }
                        return None;
                    }
                }
            }
            inverse.reverse();
            Some(SplineOp::Batch(inverse))
        }
    }
}

/// An undo / redo history of operations on one shape.
#[derive(Default, Debug)]
pub struct SplineHistory {
    undo: Vec<SplineOp>,
    redo: Vec<SplineOp>,
}

impl SplineHistory {
    /// Apply and remember `op` (clears the redo stack).
    pub fn apply(&mut self, shape: &mut SpriteShapeRenderer, op: SplineOp) -> bool {
        match apply_op(shape, &op) {
            Some(inv) => {
                self.undo.push(inv);
                self.redo.clear();
                true
            }
            None => false,
        }
    }
    /// Undo the last applied operation; false when the stack is empty.
    pub fn undo(&mut self, shape: &mut SpriteShapeRenderer) -> bool {
        let Some(op) = self.undo.pop() else { return false };
        match apply_op(shape, &op) {
            Some(inv) => {
                self.redo.push(inv);
                true
            }
            None => false,
        }
    }
    /// Redo the last undone operation; false when the stack is empty.
    pub fn redo(&mut self, shape: &mut SpriteShapeRenderer) -> bool {
        let Some(op) = self.redo.pop() else { return false };
        match apply_op(shape, &op) {
            Some(inv) => {
                self.undo.push(inv);
                true
            }
            None => false,
        }
    }
    /// True when an undo is available.
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }
    /// True when a redo is available.
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
}

/// Number of control segments.
pub fn segment_count(shape: &SpriteShapeRenderer) -> usize {
    let n = shape.points.len();
    if n < 2 { 0 } else if shape.closed { n } else { n - 1 }
}

/// Move a point (tangents travel with it, being relative).
pub fn move_point(shape: &SpriteShapeRenderer, index: usize, to: [f32; 2]) -> Option<SplineOp> {
    let mut p = *shape.points.get(index)?;
    p.position = to;
    Some(SplineOp::Set { index, point: p })
}

/// Set the outgoing (`out`) or incoming tangent from an absolute handle position.
pub fn move_tangent(shape: &SpriteShapeRenderer, index: usize, out: bool, handle: [f32; 2]) -> Option<SplineOp> {
    let mut p = *shape.points.get(index)?;
    let t = [handle[0] - p.position[0], handle[1] - p.position[1]];
    // Dragging a handle of a Linear point turns it into a smooth one.
    if p.mode == ShapeTangentMode::Linear {
        p.mode = ShapeTangentMode::Continuous;
        p.tangent_in = [0.0; 2];
        p.tangent_out = [0.0; 2];
    }
    if out { p.set_tangent_out(t) } else { p.set_tangent_in(t) }
    Some(SplineOp::Set { index, point: p })
}

/// Change a point's tangent mode (a fresh curve gets tame default handles).
pub fn set_mode(shape: &SpriteShapeRenderer, index: usize, mode: ShapeTangentMode) -> Option<SplineOp> {
    let n = shape.points.len();
    let mut p = *shape.points.get(index)?;
    let prev = shape.points.get(if index == 0 { if shape.closed { n - 1 } else { 0 } } else { index - 1 })?;
    let next = shape.points.get(if index + 1 < n { index + 1 } else if shape.closed { 0 } else { index })?;
    let along = [next.position[0] - prev.position[0], next.position[1] - prev.position[1]];
    let len = Vec2::from(along).length();
    // A third of the way to the nearer neighbour keeps a fresh curve tame.
    let reach = Vec2::from(p.position).distance(Vec2::from(prev.position)).min(Vec2::from(p.position).distance(Vec2::from(next.position)));
    p.set_mode(mode, along, if len > 1e-6 { reach / 3.0 } else { 0.0 });
    Some(SplineOp::Set { index, point: p })
}

/// Cycle Linear -> Continuous -> Broken -> Linear.
pub fn cycle_mode(shape: &SpriteShapeRenderer, index: usize) -> Option<SplineOp> {
    let next = match shape.points.get(index)?.mode {
        ShapeTangentMode::Linear => ShapeTangentMode::Continuous,
        ShapeTangentMode::Continuous => ShapeTangentMode::Broken,
        ShapeTangentMode::Broken => ShapeTangentMode::Linear,
    };
    set_mode(shape, index, next)
}

/// Flip a point's corner flag.
pub fn toggle_corner(shape: &SpriteShapeRenderer, index: usize) -> Option<SplineOp> {
    let mut p = *shape.points.get(index)?;
    p.corner = !p.corner;
    Some(SplineOp::Set { index, point: p })
}

/// Set a point's edge-thickness scale (clamped to 0+).
pub fn set_height(shape: &SpriteShapeRenderer, index: usize, height: f32) -> Option<SplineOp> {
    let mut p = *shape.points.get(index)?;
    p.height = height.max(0.0);
    Some(SplineOp::Set { index, point: p })
}

/// Split control segment `seg` at parameter `t` so the curve keeps its shape
/// (de Casteljau): the new point sits at `index = seg + 1`, and the neighbours'
/// tangents shrink to the halves. A straight segment gets a Linear point.
pub fn split_segment(shape: &SpriteShapeRenderer, seg: usize, t: f32) -> Option<SplineOp> {
    let n = shape.points.len();
    if seg >= segment_count(shape) {
        return None;
    }
    let t = t.clamp(0.001, 0.999);
    let (ia, ib) = (seg, (seg + 1) % n);
    let (a, b) = (shape.points[ia], shape.points[ib]);
    let c = segment_bezier(&a, &b);
    let straight = c[1] == c[0] && c[2] == c[3];
    let lerp = |x: Vec2, y: Vec2| x + (y - x) * t;
    let (q0, q1, q2) = (lerp(c[0], c[1]), lerp(c[1], c[2]), lerp(c[2], c[3]));
    let (r0, r1) = (lerp(q0, q1), lerp(q1, q2));
    let m = lerp(r0, r1);
    let mut point = ShapePoint {
        position: m.to_array(),
        height: a.height + (b.height - a.height) * t,
        sprite_index: a.sprite_index,
        corner: true,
        ..Default::default()
    };
    let mut ops = vec![];
    if !straight {
        point.mode = ShapeTangentMode::Broken;
        point.tangent_in = (r0 - m).to_array();
        point.tangent_out = (r1 - m).to_array();
        // Shrunk neighbours (a Linear end keeps no handle, which stays correct: its handle was zero).
        let mut na = a;
        if a.mode != ShapeTangentMode::Linear {
            na.tangent_out = (q0 - c[0]).to_array();
            if a.mode == ShapeTangentMode::Continuous {
                // The other side of a continuous point keeps its length; mirror would shorten it.
                na.mode = ShapeTangentMode::Broken;
            }
        }
        let mut nb = b;
        if b.mode != ShapeTangentMode::Linear {
            nb.tangent_in = (q2 - c[3]).to_array();
            if b.mode == ShapeTangentMode::Continuous {
                nb.mode = ShapeTangentMode::Broken;
            }
        }
        ops.push(SplineOp::Set { index: ia, point: na });
        ops.push(SplineOp::Set { index: ib, point: nb });
    }
    // Insert after the sets (indices of the neighbours don't move before it).
    // A wrap-around segment (closed, last -> first) appends at the end.
    ops.push(SplineOp::Insert { index: if ib == 0 { n } else { ib }, point });
    Some(SplineOp::Batch(ops))
}

/// Where `p` is closest to the spline: `(segment, t, distance)`.
pub fn nearest_on_spline(shape: &SpriteShapeRenderer, p: Vec2) -> Option<(usize, f32, f32)> {
    let n = shape.points.len();
    let mut best: Option<(usize, f32, f32)> = None;
    for seg in 0..segment_count(shape) {
        let c = segment_bezier(&shape.points[seg], &shape.points[(seg + 1) % n]);
        const STEPS: usize = 32;
        let mut prev = c[0];
        for i in 1..=STEPS {
            let (t0, t1) = ((i - 1) as f32 / STEPS as f32, i as f32 / STEPS as f32);
            let cur = bezier_at(&c, t1);
            let ab = cur - prev;
            let l2 = ab.length_squared();
            let u = if l2 > 1e-12 { ((p - prev).dot(ab) / l2).clamp(0.0, 1.0) } else { 0.0 };
            let d = p.distance(prev + ab * u);
            if best.is_none_or(|b| d < b.2) {
                best = Some((seg, t0 + (t1 - t0) * u, d));
            }
            prev = cur;
        }
    }
    best
}

/// Snap a coordinate pair to a grid (`step <= 0` = off).
pub fn snap(p: [f32; 2], step: f32) -> [f32; 2] {
    if step <= 0.0 { p } else { [(p[0] / step).round() * step, (p[1] / step).round() * step] }
}

/// Where a click at `p` adds a point to an open spline: after the last point.
/// Closed shapes insert on the nearest segment instead (see [`split_segment`]).
pub fn append_point(shape: &SpriteShapeRenderer, p: [f32; 2]) -> SplineOp {
    let mut pt = ShapePoint::at(p[0], p[1]);
    if let Some(last) = shape.points.last() {
        pt.height = last.height;
    }
    SplineOp::Insert { index: shape.points.len(), point: pt }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sprite_shape::spline::{TessMode, tessellate};

    fn shape() -> SpriteShapeRenderer {
        let mut s = SpriteShapeRenderer { points: vec![ShapePoint::at(0.0, 0.0), ShapePoint::at(4.0, 0.0), ShapePoint::at(4.0, 4.0)], ..Default::default() };
        s.points[1].set_mode(ShapeTangentMode::Continuous, [0.0, 4.0], 1.0);
        s
    }

    #[test]
    fn every_op_undoes_exactly() {
        let base = shape();
        let ops = vec![
            append_point(&base, [9.0, 9.0]),
            SplineOp::Insert { index: 1, point: ShapePoint::at(2.0, 1.0) },
            SplineOp::Remove { index: 1 },
            move_point(&base, 2, [5.0, 5.0]).unwrap(),
            move_tangent(&base, 1, true, [4.0, 2.0]).unwrap(),
            move_tangent(&base, 0, false, [-1.0, 0.0]).unwrap(),
            cycle_mode(&base, 0).unwrap(),
            cycle_mode(&base, 1).unwrap(),
            toggle_corner(&base, 2).unwrap(),
            set_height(&base, 1, 2.5).unwrap(),
            SplineOp::SetClosed(true),
            split_segment(&base, 0, 0.3).unwrap(),
            split_segment(&base, 1, 0.5).unwrap(),
        ];
        for op in ops {
            let mut s = base.clone();
            let inv = apply_op(&mut s, &op).unwrap_or_else(|| panic!("{op:?} applies"));
            assert_ne!(s, base, "{op:?} changed something");
            let redo = apply_op(&mut s, &inv).unwrap();
            assert_eq!(s, base, "undo of {op:?}");
            apply_op(&mut s, &redo);
            let mut again = base.clone();
            apply_op(&mut again, &op);
            assert_eq!(s, again, "redo of {op:?}");
        }
    }

    #[test]
    fn bad_indices_change_nothing_and_batches_roll_back() {
        let mut s = shape();
        let before = s.clone();
        assert!(apply_op(&mut s, &SplineOp::Remove { index: 9 }).is_none());
        assert!(apply_op(&mut s, &SplineOp::Insert { index: 9, point: ShapePoint::default() }).is_none());
        assert!(apply_op(&mut s, &SplineOp::Set { index: 3, point: ShapePoint::default() }).is_none());
        let batch = SplineOp::Batch(vec![SplineOp::Remove { index: 0 }, SplineOp::Set { index: 7, point: ShapePoint::default() }]);
        assert!(apply_op(&mut s, &batch).is_none());
        assert_eq!(s, before, "the failed batch rolled its first step back");
    }

    #[test]
    fn history_undoes_and_redoes_a_sequence() {
        let base = shape();
        let mut s = base.clone();
        let mut h = SplineHistory::default();
        assert!(!h.undo(&mut s) && !h.can_undo());
        let ops = vec![append_point(&s, [8.0, 0.0]), SplineOp::SetClosed(true), set_height(&s, 0, 3.0).unwrap()];
        let mut states = vec![s.clone()];
        for op in ops {
            assert!(h.apply(&mut s, op));
            states.push(s.clone());
        }
        for want in states.iter().rev().skip(1) {
            assert!(h.undo(&mut s));
            assert_eq!(&s, want);
        }
        assert_eq!(s, base);
        assert!(h.redo(&mut s) && h.redo(&mut s));
        assert_eq!(s, states[2]);
        // A new edit drops the redo branch.
        assert!(h.apply(&mut s, SplineOp::SetClosed(false)));
        assert!(!h.can_redo());
    }

    #[test]
    fn splitting_a_segment_keeps_the_curve() {
        let mut base = SpriteShapeRenderer { points: vec![ShapePoint::at(0.0, 0.0), ShapePoint::at(6.0, 0.0), ShapePoint::at(6.0, 5.0)], ..Default::default() };
        base.points[0].set_mode(ShapeTangentMode::Broken, [0.0, 1.0], 3.0);
        base.points[1].set_mode(ShapeTangentMode::Continuous, [1.0, 0.0], 2.0);
        for closed in [false, true] {
            for seg in 0..(if closed { 3 } else { 2 }) {
                for t in [0.2, 0.5, 0.8] {
                    let mut s = base.clone();
                    s.closed = closed;
                    let before = tessellate(&s.points, closed, TessMode::Fixed { detail: 64 }, 10_000);
                    let op = split_segment(&s, seg, t).unwrap();
                    apply_op(&mut s, &op).unwrap();
                    assert_eq!(s.points.len(), 4);
                    let after = tessellate(&s.points, closed, TessMode::Fixed { detail: 64 }, 10_000);
                    // The new polyline lies on the old curve (compare against a fine polyline of it).
                    let fine = tessellate(&base.points.to_vec(), closed, TessMode::Fixed { detail: 512 }, 100_000);
                    let worst = after.pts.iter().map(|p| crate::sprite_shape::spline::distance_to_polyline(*p, &fine.pts, closed)).fold(0.0f32, f32::max);
                    assert!(worst < 2e-3, "closed {closed} seg {seg} t {t}: {worst}");
                    assert!((after.length() - before.length()).abs() / before.length() < 5e-3);
                }
            }
        }
    }

    #[test]
    fn nearest_point_and_snapping() {
        let s = shape();
        let (seg, t, d) = nearest_on_spline(&s, Vec2::new(2.0, 1.0)).unwrap();
        assert_eq!(seg, 0);
        assert!(d < 1.5, "{d}");
        assert!(t > 0.2 && t < 0.8);
        let (seg, _, _) = nearest_on_spline(&s, Vec2::new(5.0, 3.0)).unwrap();
        assert_eq!(seg, 1);
        assert!(nearest_on_spline(&SpriteShapeRenderer { points: vec![], ..Default::default() }, Vec2::ZERO).is_none());
        assert_eq!(snap([1.26, -0.74], 0.5), [1.5, -0.5]);
        assert_eq!(snap([1.26, -0.74], 0.0), [1.26, -0.74]);
    }

    #[test]
    fn dragging_a_linear_handle_makes_it_smooth() {
        let s = shape();
        let SplineOp::Set { point, .. } = move_tangent(&s, 0, true, [0.0, 2.0]).unwrap() else { panic!() };
        assert_eq!(point.mode, ShapeTangentMode::Continuous);
        assert_eq!((point.tangent_out, point.tangent_in), ([0.0, 2.0], [0.0, -2.0]));
    }
}
