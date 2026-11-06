//! Authored collision: `UCX_` / `UBX_` / `USP_` / `UCP_` pieces modelled with
//! a model (see [`zestengine_assets::CollisionPiece`]). Tracks which objects own
//! pieces and fits each piece's primitive; a Mesh Collider on such an object
//! uses them instead of building a shape from its render mesh.

use std::collections::HashMap;
use std::sync::Arc;

use zestengine_assets::{CollisionKind, CollisionPiece, ObjectSource};
use glam::{Mat3, Quat, Vec3};

use crate::scene::LoadedScene;

/// Every loaded model's pieces, by the render mesh they belong to (global
/// mesh index) and, for pieces that matched no mesh, by model path (owned by
/// the model's root object).
#[derive(Default)]
pub struct AuthoredCollision {
    sets: Vec<Arc<Vec<CollisionPiece>>>,
    by_mesh: HashMap<usize, usize>,
    by_model: HashMap<String, usize>,
}

impl AuthoredCollision {
    /// Record `model`'s pieces; its meshes start at global index `mesh_base`.
    /// When several nodes share a render mesh, the first one's pieces win.
    pub fn register(&mut self, model: &zestengine_assets::Scene, mesh_base: usize, path: Option<&str>) {
        self.register_mapped(model, path, &|m| Some(mesh_base + m));
    }

    /// Replace the pieces of a model that is being reimported in place: forget
    /// the ones registered for its `count` meshes from `mesh_base` (and its
    /// unmatched set), then register the new file's, with `map` giving each new
    /// local mesh index its global scene mesh (`None` = no slot, skipped).
    pub fn replace(&mut self, model: &zestengine_assets::Scene, path: &str, mesh_base: usize, count: usize, map: &dyn Fn(usize) -> Option<usize>) {
        self.by_mesh.retain(|m, _| !(mesh_base..mesh_base + count).contains(m));
        self.by_model.remove(path);
        self.register_mapped(model, Some(path), map);
    }

    fn register_mapped(&mut self, model: &zestengine_assets::Scene, path: Option<&str>, map: &dyn Fn(usize) -> Option<usize>) {
        let mut groups: Vec<(Option<usize>, Vec<CollisionPiece>)> = Vec::new();
        for p in &model.collision {
            match groups.iter_mut().find(|g| g.0 == p.target) {
                Some(g) => g.1.push(p.clone()),
                None => groups.push((p.target, vec![p.clone()])),
            }
        }
        for (target, pieces) in groups {
            match target {
                Some(t) => {
                    let Some(slot) = model.instances.get(t).and_then(|inst| inst.slots.first()) else { continue };
                    let Some(mesh) = map(slot.mesh) else { continue };
                    if !self.by_mesh.contains_key(&mesh) {
                        self.sets.push(Arc::new(pieces));
                        self.by_mesh.insert(mesh, self.sets.len() - 1);
                    }
                }
                None => {
                    if let Some(path) = path {
                        self.sets.push(Arc::new(pieces));
                        self.by_model.insert(path.to_owned(), self.sets.len() - 1);
                    }
                }
            }
        }
    }

    /// The pieces of set `id` (see [`authored_set`]).
    pub fn set(&self, id: usize) -> Option<&Arc<Vec<CollisionPiece>>> {
        self.sets.get(id)
    }
}

/// The authored piece set object `i` owns: its render mesh's, or for a model
/// root (an empty whose children are the model's parts) the model's pieces
/// that matched no mesh.
pub fn authored_set(scene: &LoadedScene, i: usize) -> Option<usize> {
    let reg = &scene.authored_collision;
    let obj = scene.objects.get(i)?;
    if let Some(s) = obj.render_slots().find_map(|r| reg.by_mesh.get(&r.mesh)) {
        return Some(*s);
    }
    if obj.render.is_some() || reg.by_model.is_empty() || !matches!(obj.source, ObjectSource::Empty) {
        return None;
    }
    scene.objects.iter().filter(|o| o.parent == Some(i)).find_map(|o| match &o.source {
        ObjectSource::Model { path, .. } => reg.by_model.get(path).copied(),
        _ => None,
    })
}

/// Object `i`'s authored pieces, if it has any.
pub fn authored_pieces(scene: &LoadedScene, i: usize) -> Option<Arc<Vec<CollisionPiece>>> {
    authored_set(scene, i).and_then(|s| scene.authored_collision.set(s)).cloned()
}

/// The prefixes in use, e.g. "UCX_ x2, UBX_".
pub fn pieces_summary(pieces: &[CollisionPiece]) -> String {
    CollisionKind::ALL
        .into_iter()
        .filter_map(|k| match pieces.iter().filter(|p| p.kind == k).count() {
            0 => None,
            1 => Some(k.prefix().to_owned()),
            n => Some(format!("{} x{n}", k.prefix())),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// The primitive an authored piece becomes, in its object's local space
/// (unscaled).
#[derive(Clone, Debug)]
pub enum ColliderPiece {
    Hull(Vec<Vec3>, Vec<[u32; 3]>),
    Cuboid { center: Vec3, rotation: Quat, half: Vec3 },
    Ball(Vec3, f32),
    /// Segment ends and radius.
    Capsule(Vec3, Vec3, f32),
}

/// Corner cap for an authored hull (they are modelled low-poly already).
const MAX_HULL_POINTS: usize = 1024;

/// Fit `piece`'s primitive: convex hull (`UCX_`), oriented box (`UBX_`),
/// bounding sphere (`USP_`) or capsule (`UCP_`); box and capsule fall back to
/// the hull. None for a degenerate piece (a flat hull).
pub fn fit_piece(piece: &CollisionPiece) -> Option<ColliderPiece> {
    let hull = || {
        crate::physics::hull_of(&piece.points, MAX_HULL_POINTS)
            .filter(|(p, t)| crate::physics::mesh_volume(p, t) > 1e-9)
            .map(|(p, t)| ColliderPiece::Hull(p, t))
    };
    match piece.kind {
        CollisionKind::Convex => hull(),
        CollisionKind::Box => fit_box(&piece.points).or_else(hull),
        CollisionKind::Sphere => fit_sphere(&piece.points),
        CollisionKind::Capsule => fit_capsule(&piece.points).or_else(hull),
    }
}

/// Min and max of `points` along the columns of `axes`.
fn extents(points: &[Vec3], axes: Mat3) -> (Vec3, Vec3) {
    let to_axes = axes.transpose();
    points.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), p| {
        let q = to_axes * *p;
        (lo.min(q), hi.max(q))
    })
}

/// Smallest box over the hull's perpendicular face-normal pairs (exact for a
/// box however it is rotated); the axis-aligned bounds when the piece has
/// no perpendicular faces.
fn fit_box(points: &[Vec3]) -> Option<ColliderPiece> {
    if points.is_empty() {
        return None;
    }
    let hull = crate::physics::hull_of(points, MAX_HULL_POINTS);
    let pts = hull.as_ref().map_or(points, |h| h.0.as_slice());
    let mut normals: Vec<Vec3> = Vec::new();
    for t in hull.as_ref().map_or(&[][..], |h| h.1.as_slice()) {
        let [a, b, c] = t.map(|k| pts[k as usize]);
        let n = (b - a).cross(c - a).normalize_or_zero();
        if n != Vec3::ZERO && !normals.iter().any(|m| m.dot(n).abs() > 0.999) {
            normals.push(n);
            if normals.len() >= 64 {
                break;
            }
        }
    }
    let mut best: Option<(f32, Mat3)> = None;
    for (k, a) in normals.iter().enumerate() {
        for b in &normals[k + 1..] {
            if a.dot(*b).abs() > 1e-3 {
                continue;
            }
            let axes = Mat3::from_cols(*a, *b, a.cross(*b).normalize());
            let (lo, hi) = extents(pts, axes);
            let size = hi - lo;
            let volume = size.x * size.y * size.z;
            if best.is_none_or(|(v, _)| volume < v - 1e-6) {
                best = Some((volume, axes));
            }
        }
    }
    let axes = best.map_or(Mat3::IDENTITY, |b| b.1);
    let (lo, hi) = extents(pts, axes);
    Some(ColliderPiece::Cuboid { center: axes * ((lo + hi) * 0.5), rotation: Quat::from_mat3(&axes).normalize(), half: (hi - lo) * 0.5 })
}

/// Bounds centre and the farthest vertex as radius.
fn fit_sphere(points: &[Vec3]) -> Option<ColliderPiece> {
    if points.is_empty() {
        return None;
    }
    let (lo, hi) = extents(points, Mat3::IDENTITY);
    let c = (lo + hi) * 0.5;
    let r = points.iter().map(|p| p.distance(c)).fold(0.0f32, f32::max);
    (r > 1e-5).then_some(ColliderPiece::Ball(c, r))
}

/// Capsule along the points' principal axis: radius from the farthest vertex
/// off the axis, ends from the extremes along it. A piece about as long as
/// it is wide becomes a ball.
fn fit_capsule(points: &[Vec3]) -> Option<ColliderPiece> {
    if points.len() < 4 {
        return None;
    }
    let mean = points.iter().copied().sum::<Vec3>() / points.len() as f32;
    let mut cov = Mat3::ZERO;
    for p in points {
        let d = *p - mean;
        cov += Mat3::from_cols(d * d.x, d * d.y, d * d.z);
    }
    let (lo, hi) = extents(points, Mat3::IDENTITY);
    let size = hi - lo;
    let start = if size.x >= size.y && size.x >= size.z {
        Vec3::X
    } else if size.y >= size.z {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let mut axis = (start + Vec3::new(0.01, 0.02, 0.03)).normalize();
    for _ in 0..32 {
        let next = (cov * axis).normalize_or_zero();
        if next == Vec3::ZERO {
            axis = start;
            break;
        }
        axis = next;
    }
    let (mut t0, mut t1, mut r) = (f32::MAX, f32::MIN, 0.0f32);
    for p in points {
        let d = *p - mean;
        let t = d.dot(axis);
        t0 = t0.min(t);
        t1 = t1.max(t);
        r = r.max((d - axis * t).length());
    }
    if r <= 1e-5 {
        return None;
    }
    let mid = mean + axis * ((t0 + t1) * 0.5);
    let half = (t1 - t0) * 0.5 - r;
    if half <= 1e-4 {
        return Some(ColliderPiece::Ball(mid, r.max((t1 - t0) * 0.5)));
    }
    Some(ColliderPiece::Capsule(mid - axis * half, mid + axis * half, r))
}

/// A circle of `segments` edges around `c` in the plane of `a` and `b`
/// (unit, perpendicular); `arc` is the fraction of a turn drawn.
fn circle(out: &mut Vec<(Vec3, Vec3)>, c: Vec3, a: Vec3, b: Vec3, r: f32, arc: f32, segments: usize) {
    for k in 0..segments {
        let t0 = k as f32 / segments as f32 * arc * std::f32::consts::TAU;
        let t1 = (k + 1) as f32 / segments as f32 * arc * std::f32::consts::TAU;
        out.push((c + (a * t0.cos() + b * t0.sin()) * r, c + (a * t1.cos() + b * t1.sin()) * r));
    }
}

/// Outline edges of a box, ball or capsule piece (hulls are drawn by the
/// caller's fold-edge outline).
pub fn primitive_wire(piece: &ColliderPiece, out: &mut Vec<(Vec3, Vec3)>) {
    match piece {
        ColliderPiece::Hull(..) => {}
        ColliderPiece::Cuboid { center, rotation, half } => {
            let corner = |k: usize| {
                let s = Vec3::new(if k & 1 == 0 { -1.0 } else { 1.0 }, if k & 2 == 0 { -1.0 } else { 1.0 }, if k & 4 == 0 { -1.0 } else { 1.0 });
                *center + *rotation * (s * *half)
            };
            for k in 0..8 {
                for bit in [1, 2, 4] {
                    if k & bit == 0 {
                        out.push((corner(k), corner(k | bit)));
                    }
                }
            }
        }
        ColliderPiece::Ball(c, r) => {
            for (a, b) in [(Vec3::X, Vec3::Y), (Vec3::Y, Vec3::Z), (Vec3::Z, Vec3::X)] {
                circle(out, *c, a, b, *r, 1.0, 48);
            }
        }
        ColliderPiece::Capsule(a, b, r) => {
            let axis = (*b - *a).normalize_or_zero();
            let (u, v) = axis.any_orthonormal_pair();
            circle(out, *a, u, v, *r, 1.0, 32);
            circle(out, *b, u, v, *r, 1.0, 32);
            for side in [u, v, -u, -v] {
                out.push((*a + side * *r, *b + side * *r));
            }
            for side in [u, v] {
                circle(out, *b, side, axis, *r, 0.5, 16);
                circle(out, *a, -side, -axis, *r, 0.5, 16);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn piece(kind: CollisionKind, points: Vec<Vec3>) -> CollisionPiece {
        CollisionPiece { name: "piece".into(), kind, target: None, points }
    }

    fn box_points(center: Vec3, rot: Quat, half: Vec3) -> Vec<Vec3> {
        (0..8)
            .map(|k| {
                let s = Vec3::new(if k & 1 == 0 { -1.0 } else { 1.0 }, if k & 2 == 0 { -1.0 } else { 1.0 }, if k & 4 == 0 { -1.0 } else { 1.0 });
                center + rot * (s * half)
            })
            .collect()
    }

    #[test]
    fn a_rotated_box_piece_fits_its_own_box() {
        let rot = Quat::from_rotation_y(0.6) * Quat::from_rotation_x(0.3);
        let pts = box_points(Vec3::new(1.0, 2.0, 3.0), rot, Vec3::new(0.5, 1.0, 2.0));
        let Some(ColliderPiece::Cuboid { center, rotation, half }) = fit_piece(&piece(CollisionKind::Box, pts.clone())) else { panic!("not a box") };
        assert!(center.abs_diff_eq(Vec3::new(1.0, 2.0, 3.0), 1e-3), "{center}");
        let vol = half.x * half.y * half.z * 8.0;
        assert!((vol - 8.0).abs() < 1e-2, "{vol}");
        // Every corner lies on the fitted box.
        for p in pts {
            let local = (rotation.inverse() * (p - center)).abs();
            assert!(local.abs_diff_eq(half, 1e-3), "{local} {half}");
        }
    }

    #[test]
    fn sphere_and_capsule_pieces_fit_their_primitives() {
        let mut ball = Vec::new();
        let mut capsule = Vec::new();
        for i in 0..16 {
            for j in 0..8 {
                let (t, p) = (i as f32 / 16.0 * std::f32::consts::TAU, j as f32 / 7.0 * std::f32::consts::PI);
                let d = Vec3::new(p.sin() * t.cos(), p.cos(), p.sin() * t.sin());
                ball.push(Vec3::new(0.0, 5.0, 0.0) + d * 2.0);
                // Radius 0.5, segment from z = -1 to z = 1.
                let d = Vec3::new(d.x, d.z, d.y);
                capsule.push(d * 0.5 + Vec3::Z * if d.z >= 0.0 { 1.0 } else { -1.0 });
            }
        }
        let Some(ColliderPiece::Ball(c, r)) = fit_piece(&piece(CollisionKind::Sphere, ball)) else { panic!("not a ball") };
        assert!(c.abs_diff_eq(Vec3::new(0.0, 5.0, 0.0), 1e-3) && (r - 2.0).abs() < 1e-3, "{c} {r}");
        let Some(ColliderPiece::Capsule(a, b, r)) = fit_piece(&piece(CollisionKind::Capsule, capsule)) else { panic!("not a capsule") };
        assert!((r - 0.5).abs() < 0.02, "{r}");
        assert!((a.distance(b) - 2.0).abs() < 0.05, "{a} {b}");
        assert!(a.x.abs() < 1e-3 && a.y.abs() < 1e-3 && b.x.abs() < 1e-3 && b.y.abs() < 1e-3);
    }

    #[test]
    fn convex_pieces_become_hulls_and_flat_ones_are_dropped() {
        let pts = box_points(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE);
        assert!(matches!(fit_piece(&piece(CollisionKind::Convex, pts)), Some(ColliderPiece::Hull(p, _)) if p.len() == 8));
        let flat = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::ONE.with_z(0.0)];
        assert!(fit_piece(&piece(CollisionKind::Convex, flat)).is_none());
        assert_eq!(
            pieces_summary(&[piece(CollisionKind::Convex, vec![]), piece(CollisionKind::Convex, vec![]), piece(CollisionKind::Box, vec![])]),
            "UCX_ x2, UBX_"
        );
    }
}
