//! Pure helpers for editing an object's transform relative to its parent.
//!
//! Position, rotation and scale of a `SceneObject` are always LOCAL (relative
//! to the parent; world = parent_world * local). The Inspector shows and edits
//! those local values. Anything that works in world space (the transform
//! gizmo, re-parenting) converts back through [`local_from_world`].
//!
//! Shear: a parent with a rotation and a non-uniform scale makes a rotated
//! child's world matrix sheared, which a translation + rotation + scale triple
//! cannot represent. Every conversion here therefore decomposes to the NEAREST
//! TRS: translation is exact, scale is the length of each matrix column (sign
//! from the determinant), and the rotation is the Gram-Schmidt orthonormalised
//! basis. The result is always finite (degenerate input falls back to `None`
//! so the caller keeps the previous local values).

use glam::{Mat3, Mat4, Quat, Vec3};

/// `(scale, rotation, translation)`, in the order glam's decomposition uses.
pub type Trs = (Vec3, Quat, Vec3);

/// -0.0 and float noise around zero become exactly 0.0 (so the UI never shows
/// "-0.0").
pub fn clean_zero(v: f32) -> f32 {
    if v.abs() < 1e-4 { 0.0 } else { v }
}

/// Decompose to the nearest scale / rotation / translation. `None` when the
/// matrix is not finite.
pub fn decompose_trs(m: Mat4) -> Option<Trs> {
    if !m.is_finite() {
        return None;
    }
    let (c0, c1, c2) = (m.x_axis.truncate(), m.y_axis.truncate(), m.z_axis.truncate());
    let mut sx = c0.length();
    let sy = c1.length();
    let sz = c2.length();
    if c0.cross(c1).dot(c2) < 0.0 {
        sx = -sx; // mirrored: carry the flip on X, like glam
    }
    let x = if sx.abs() > 1e-12 { c0 / sx } else { Vec3::X };
    let y_raw = if sy > 1e-12 { c1 / sy } else { Vec3::Y };
    let mut y = y_raw - x * x.dot(y_raw);
    y = if y.length_squared() > 1e-12 { y.normalize() } else { x.any_orthonormal_vector() };
    let z = x.cross(y);
    let rotation = Quat::from_mat3(&Mat3::from_cols(x, y, z)).normalize();
    let out = (Vec3::new(sx, sy, sz), rotation, m.w_axis.truncate());
    (out.0.is_finite() && out.1.is_finite() && out.2.is_finite()).then_some(out)
}

/// The local TRS that puts an object at `world` under a parent whose world
/// matrix is `parent_world`. `None` when the parent is singular / non-finite.
pub fn local_from_world(parent_world: Mat4, world: Mat4) -> Option<Trs> {
    if !parent_world.is_finite() || parent_world.determinant().abs() < 1e-18 {
        return None;
    }
    decompose_trs(parent_world.inverse() * world)
}

/// Write-back of the transform gizmo. The gizmo edits the object's WORLD
/// `(scale, rotation, translation)` as decomposed from `old_world`; `new` is
/// what it produced. The edit is applied as a delta to the true world matrix
/// (new translation; rotation delta applied about the origin; per-object-axis
/// scale ratio), so parts of a sheared world matrix the gizmo did not touch
/// survive, then converted to the parent's space.
///
/// Meaning of the modes: Translate moves the object by the world delta.
/// Rotate turns it by the world rotation delta (Local or Global axes only
/// change which world axis the gizmo turns about). Scale multiplies the
/// object's own axes by the gizmo's per-axis ratio: in Global mode a
/// world-axis scale on a rotated object is applied along the object's own
/// axes (a world-axis stretch of a rotated child would be a shear).
pub fn gizmo_writeback(parent_world: Mat4, old_world: Mat4, new: (&Vec3, &Quat, &Vec3)) -> Option<Trs> {
    let (s0, r0, _) = decompose_trs(old_world)?;
    let (s1, r1, t1) = (*new.0, *new.1, *new.2);
    let dr = r1 * r0.inverse();
    let ds = Vec3::select(s0.cmpne(Vec3::ZERO), s1 / s0, Vec3::ONE);
    let lin = Mat3::from_quat(dr) * Mat3::from_mat4(old_world) * Mat3::from_diagonal(ds);
    let world = Mat4::from_cols(lin.x_axis.extend(0.0), lin.y_axis.extend(0.0), lin.z_axis.extend(0.0), t1.extend(1.0));
    local_from_world(parent_world, world)
}

/// Euler XYZ degrees of a rotation, with float noise / -0.0 cleaned.
pub fn euler_deg(q: Quat) -> [f32; 3] {
    let (x, y, z) = q.to_euler(glam::EulerRot::XYZ);
    // Rounded to 1e-3 degrees: f32 noise (-39.999996) never reaches the UI.
    let tidy = |r: f32| clean_zero((r.to_degrees() * 1e3).round() / 1e3);
    [tidy(x), tidy(y), tidy(z)]
}

/// Rotation from Euler XYZ degrees (the inverse of [`euler_deg`]).
pub fn euler_to_quat(e: [f32; 3]) -> Quat {
    Quat::from_euler(glam::EulerRot::XYZ, e[0].to_radians(), e[1].to_radians(), e[2].to_radians())
}

/// Last Euler triple the Inspector wrote for an object, with the quaternion it
/// produced. The quaternion alone has many equivalent Euler triples
/// (e.g. X=200 reads back as X=-160, Y=180-Y, Z+180), so while the rotation is
/// still exactly what the Inspector wrote, the cached triple is shown instead
/// of a re-derived one. Anything else changing the rotation (gizmo, undo,
/// reparenting) invalidates it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EulerCache {
    pub quat: Quat,
    pub euler: [f32; 3],
}

impl EulerCache {
    pub fn new(euler: [f32; 3]) -> Self {
        Self { quat: euler_to_quat(euler), euler }
    }
}

/// Euler degrees to display for `q`: the cached triple while it still
/// describes `q`, else a fresh derivation.
pub fn display_euler(cache: Option<EulerCache>, q: Quat) -> [f32; 3] {
    match cache {
        Some(c) if same_rotation(c.quat, q) => c.euler,
        _ => euler_deg(q),
    }
}

fn same_rotation(a: Quat, b: Quat) -> bool {
    let close = |p: Quat, q: Quat| (p.x - q.x).abs() < 1e-6 && (p.y - q.y).abs() < 1e-6 && (p.z - q.z).abs() < 1e-6 && (p.w - q.w).abs() < 1e-6;
    close(a, b) || close(a, -b)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    fn mat_close(a: Mat4, b: Mat4, eps: f32) -> bool {
        a.to_cols_array().iter().zip(b.to_cols_array()).all(|(p, q)| (p - q).abs() < eps)
    }

    fn trs(t: Trs) -> Mat4 {
        Mat4::from_scale_rotation_translation(t.0, t.1, t.2)
    }

    /// Parent rotated about Y and scaled non-uniformly.
    fn parent() -> Mat4 {
        Mat4::from_scale_rotation_translation(Vec3::new(2.0, 1.0, 0.5), Quat::from_rotation_y(30f32.to_radians()), Vec3::new(5.0, 1.0, -2.0))
    }

    fn child_local() -> Trs {
        (Vec3::new(1.5, 2.0, 0.75), Quat::IDENTITY, Vec3::new(1.0, 2.0, 3.0))
    }

    #[test]
    fn world_is_parent_times_local_for_all_three_channels() {
        let (p, l) = (parent(), trs(child_local()));
        let world = p * l;
        // Position: local translation sits in the parent's scaled, rotated space.
        let expect = p.transform_point3(Vec3::new(1.0, 2.0, 3.0));
        assert!((world.w_axis.truncate() - expect).length() < EPS);
        // Rotation / scale: the child's world axes are the parent's axes scaled by both.
        let (ws, wr, _) = decompose_trs(world).unwrap();
        assert!((ws - Vec3::new(3.0, 2.0, 0.375)).abs().max_element() < EPS, "{ws:?}");
        assert!(wr.angle_between(Quat::from_rotation_y(30f32.to_radians())) < EPS);
    }

    #[test]
    fn reparent_preserves_world_when_representable() {
        // Child with no relative shear: identity local rotation under the parent.
        let world = trs((Vec3::new(3.0, 2.0, 0.375), Quat::from_rotation_y(30f32.to_radians()), Vec3::new(9.0, 3.0, 1.0)));
        let local = local_from_world(parent(), world).unwrap();
        assert!(mat_close(parent() * trs(local), world, EPS));
        // Local scale/rotation were recomputed (not copied from the world).
        assert!((local.0 - Vec3::new(1.5, 2.0, 0.75)).abs().max_element() < EPS);
        assert!(local.1.angle_between(Quat::IDENTITY) < EPS);
        // To the root: local == world.
        let root = local_from_world(Mat4::IDENTITY, world).unwrap();
        assert!(mat_close(trs(root), world, EPS));
    }

    #[test]
    fn reparent_under_rotated_nonuniform_parent_is_finite_and_keeps_the_origin() {
        // A rotated child of a differently rotated non-uniform parent shears.
        for deg in [0.0f32, 15.0, 45.0, 90.0, 135.0, 180.0, 270.0, 359.0] {
            let world = Mat4::from_scale_rotation_translation(Vec3::new(1.0, 2.0, 3.0), Quat::from_euler(glam::EulerRot::XYZ, 0.4, deg.to_radians(), 0.2), Vec3::new(1.0, 2.0, 3.0));
            let local = local_from_world(parent(), world).unwrap();
            assert!(local.0.is_finite() && local.1.is_finite() && local.2.is_finite());
            assert!((local.1.length() - 1.0).abs() < 1e-5);
            let back = parent() * trs(local);
            // Origin exact; the linear part is the nearest TRS (shear dropped).
            assert!((back.w_axis.truncate() - world.w_axis.truncate()).length() < EPS, "origin at {deg}");
            assert!(back.is_finite());
        }
    }

    #[test]
    fn degenerate_matrices_never_produce_nan() {
        assert!(local_from_world(Mat4::from_scale(Vec3::ZERO), Mat4::IDENTITY).is_none(), "singular parent keeps old local");
        assert!(local_from_world(Mat4::IDENTITY, Mat4::from_scale(Vec3::new(0.0, 1.0, 1.0))).is_some_and(|t| t.0.is_finite() && t.1.is_finite()));
        assert!(local_from_world(Mat4::IDENTITY, Mat4::ZERO).is_some_and(|t| t.1.is_finite() && t.0 == Vec3::ZERO));
        assert!(decompose_trs(Mat4::from_cols_array(&[f32::NAN; 16])).is_none());
        // Mirrored (negative) scale keeps its sign on X.
        let m = Mat4::from_scale(Vec3::new(-2.0, 1.0, 1.0));
        let t = decompose_trs(m).unwrap();
        assert!(mat_close(trs(t), m, EPS));
    }

    fn gizmo_input(world: Mat4) -> (Vec3, Quat, Vec3) {
        let (s, r, t) = decompose_trs(world).unwrap();
        (t, r, s)
    }

    #[test]
    fn gizmo_translate_moves_only_the_position_under_a_rotated_scaled_parent() {
        let old_local = child_local();
        let old_world = parent() * trs(old_local);
        let (mut t, r, s) = gizmo_input(old_world);
        let d = Vec3::new(0.3, -1.0, 2.0);
        t += d;
        let new = gizmo_writeback(parent(), old_world, (&s, &r, &t)).unwrap();
        // Local rotation and scale are untouched.
        assert!((new.0 - old_local.0).abs().max_element() < EPS);
        assert!(new.1.angle_between(old_local.1) < EPS);
        // World origin moved by exactly the world delta.
        let world = parent() * trs(new);
        assert!((world.w_axis.truncate() - (old_world.w_axis.truncate() + d)).length() < EPS);
    }

    #[test]
    fn gizmo_with_no_change_leaves_local_values_alone_even_when_sheared() {
        // A rotated child under the non-uniform parent has a sheared world matrix.
        let old_local: Trs = (Vec3::new(1.0, 2.0, 1.0), Quat::from_rotation_z(40f32.to_radians()), Vec3::new(1.0, 0.0, 0.0));
        let old_world = parent() * trs(old_local);
        let (t, r, s) = gizmo_input(old_world);
        let new = gizmo_writeback(parent(), old_world, (&s, &r, &t)).unwrap();
        assert!(mat_close(parent().inverse() * old_world, parent().inverse() * parent() * trs(new), 1e-3));
        assert!((new.2 - old_local.2).length() < EPS);
        assert!((new.0 - old_local.0).abs().max_element() < 1e-3);
        assert!(new.1.angle_between(old_local.1) < 1e-3);
    }

    #[test]
    fn gizmo_rotate_local_and_global_axes_are_exact_for_representable_turns() {
        let old_world = parent() * trs(child_local());
        let (t, r0, s) = gizmo_input(old_world);
        let r_parent = Quat::from_rotation_y(30f32.to_radians());
        // A half turn about the parent's own X axis (commutes with diagonal scale):
        // Global mode = the world axis, Local mode = the object's axis (same here).
        for axis_world in [r_parent * Vec3::X, r0 * Vec3::X] {
            let r1 = Quat::from_axis_angle(axis_world, std::f32::consts::PI) * r0;
            let new = gizmo_writeback(parent(), old_world, (&s, &r1, &t)).unwrap();
            assert!(new.0.is_finite() && new.1.is_finite() && new.2.is_finite());
            let world = parent() * trs(new);
            assert!((world.w_axis.truncate() - old_world.w_axis.truncate()).length() < EPS, "origin stays");
            // Scale magnitudes unchanged.
            assert!((new.0.abs() - child_local().0).abs().max_element() < EPS, "{:?}", new.0);
        }
    }

    #[test]
    fn gizmo_rotate_arbitrary_angles_stays_finite_and_continuous() {
        let old_local = child_local();
        let mut world = parent() * trs(old_local);
        let mut last: Option<Quat> = None;
        for _ in 0..360 {
            let (t, r0, s) = gizmo_input(world);
            let r1 = Quat::from_rotation_x(1f32.to_radians()) * r0; // Global X, 1 degree per frame
            let new = gizmo_writeback(parent(), world, (&s, &r1, &t)).unwrap();
            assert!(new.0.is_finite() && new.1.is_finite() && new.2.is_finite());
            if let Some(l) = last {
                assert!(l.angle_between(new.1) < 0.2, "no flips between frames");
            }
            last = Some(new.1);
            world = parent() * trs(new);
        }
    }

    #[test]
    fn gizmo_scale_multiplies_local_scale_per_object_axis() {
        let old_local = child_local();
        let old_world = parent() * trs(old_local);
        let (t, r, s0) = gizmo_input(old_world);
        let s1 = s0 * Vec3::new(1.0, 2.0, 1.0);
        let new = gizmo_writeback(parent(), old_world, (&s1, &r, &t)).unwrap();
        assert!((new.0 - old_local.0 * Vec3::new(1.0, 2.0, 1.0)).abs().max_element() < EPS, "{:?}", new.0);
        assert!(new.1.angle_between(old_local.1) < EPS);
        assert!((new.2 - old_local.2).length() < EPS);
        // Uniform scale likewise.
        let s2 = s0 * 0.5;
        let new = gizmo_writeback(parent(), old_world, (&s2, &r, &t)).unwrap();
        assert!((new.0 - old_local.0 * 0.5).abs().max_element() < EPS);
    }

    #[test]
    fn display_euler_never_shows_negative_zero() {
        for q in [Quat::IDENTITY, Quat::from_rotation_x(1e-7), Quat::from_euler(glam::EulerRot::XYZ, -0.0, 0.0, -1e-6)] {
            for v in euler_deg(q) {
                assert!(v == 0.0 && v.is_sign_positive(), "{v}");
            }
        }
        assert!(clean_zero(-0.0).is_sign_positive());
    }

    #[test]
    fn dragging_x_through_all_branches_keeps_y_and_z_and_typed_values_round_trip() {
        let (y, z) = (25.0f32, -40.0f32);
        let mut cache: Option<EulerCache> = None;
        let mut q = euler_to_quat([0.0, y, z]);
        let mut x = 0.0f32;
        let path = (0..=108).map(|i| i as f32 * 2.5).chain((0..=108).rev().map(|i| i as f32 * 2.5)); // 0 -> 270 -> 0
        for step in path {
            // The inspector shows the (cached) triple, the user changes X only.
            let mut shown = display_euler(cache, q);
            shown[0] = step;
            x = step;
            q = euler_to_quat(shown);
            cache = Some(EulerCache::new(shown));
            let next = display_euler(cache, q);
            assert_eq!(next, [x, y, z], "axes must not jump at X={x}");
        }
        assert_eq!(x, 0.0);
        // Typed values round-trip exactly, including beyond +-180 and gimbal pose.
        for e in [[123.4, -56.7, 89.1], [270.0, 0.0, 0.0], [10.0, 90.0, 20.0], [-720.0, 180.0, 359.9]] {
            let cache = Some(EulerCache::new(e));
            assert_eq!(display_euler(cache, euler_to_quat(e)), e);
        }
    }

    #[test]
    fn external_rotation_change_rederives_the_triple() {
        let cache = Some(EulerCache::new([200.0, 10.0, 20.0]));
        let other = Quat::from_rotation_z(0.5);
        let e = display_euler(cache, other);
        assert!(euler_to_quat(e).angle_between(other) < 1e-5);
        assert_ne!(e, [200.0, 10.0, 20.0]);
        // Without a cache an equivalent branch is shown (documented behaviour).
        let derived = display_euler(None, euler_to_quat([200.0, 10.0, 20.0]));
        assert!(euler_to_quat(derived).angle_between(euler_to_quat([200.0, 10.0, 20.0])) < 1e-3);
        assert!(derived[0] < 0.0, "X=200 reads back as -160 without the cache: {derived:?}");
    }

    #[test]
    fn reset_is_identity_in_local_space() {
        // The Reset button writes translation 0, rotation 0, scale 1 to the
        // LOCAL fields: the object then sits exactly on its parent's origin.
        let local: Trs = (Vec3::ONE, euler_to_quat([0.0; 3]), Vec3::ZERO);
        assert!(mat_close(parent() * trs(local), parent(), EPS));
    }
}
