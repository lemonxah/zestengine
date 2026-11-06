//! Stereo cluster culling math (GEO-5): one LOD cut and one visible list for
//! both eyes.
//!
//! Both eyes draw the same clusters (multiview, `standard.vert -DCLUSTER
//! -DMULTIVIEW`), so the cull runs once with
//! - the LOD error judged from the centre eye: the eyes are half an IPD from
//!   it, so a cluster's distance differs by at most that, a relative change of
//!   `ipd / 2 / distance` in projected error (about 3 % at 1 m, under 1 % at
//!   3 m for 64 mm; [`lod_slack`]), and the projection scale of the wider eye;
//! - the frustum a conservative union of both eyes' frusta ([`UnionFrustum`]):
//!   a cluster visible to either eye is inside it; it may keep a few that
//!   neither eye sees (a sliver at the sides near the viewer).
//!
//! Pure math, unit-tested here. XR cannot run on the development machine, so
//! the multiview cluster draws, the union frustum on a headset and the
//! per-eye pyramid for Hi-Z are NOT exercised on hardware: the frame loop
//! still draws cluster-eligible meshes through the ordinary path in VR.

use glam::{Mat4, Vec3, Vec4};

/// The two eyes of a stereo frame: view matrices (world -> eye) and
/// projections (Vulkan clip space, depth 0..1).
#[derive(Clone, Copy, Debug)]
pub struct StereoView {
    pub view: [Mat4; 2],
    pub proj: [Mat4; 2],
}

/// The cyclopean (centre) eye: position halfway between the eyes, orientation
/// the half-way rotation (eyes of a canted display are rotated slightly apart).
pub fn centre_eye(v: &StereoView) -> Mat4 {
    let pose = |m: &Mat4| {
        let world = m.inverse();
        let (_, rot, pos) = world.to_scale_rotation_translation();
        (rot, pos)
    };
    let (r0, p0) = pose(&v.view[0]);
    let (r1, p1) = pose(&v.view[1]);
    let rot = r0.slerp(r1, 0.5).normalize();
    Mat4::from_rotation_translation(rot, (p0 + p1) * 0.5).inverse()
}

/// Distance between the eyes (metres, world units).
pub fn eye_separation(v: &StereoView) -> f32 {
    let p = |m: &Mat4| m.inverse().w_axis.truncate();
    p(&v.view[0]).distance(p(&v.view[1]))
}

/// Relative error the centre-eye LOD judgement can have against either eye for
/// content at least `distance` away: `ipd / 2 / distance`. Cull with the target
/// pixel error divided by `1 + lod_slack` to guarantee the target in both eyes.
pub fn lod_slack(ipd: f32, distance: f32) -> f32 {
    if distance <= 0.0 { f32::MAX } else { ipd * 0.5 / distance }
}

/// Pixels per world unit at distance 1 of the wider eye (`0.5 * height *
/// proj[1][1]`), the value the cull's LOD test uses for both eyes.
pub fn pixels_per_unit(v: &StereoView, height_px: f32) -> f32 {
    0.5 * height_px * v.proj[0].y_axis.y.abs().max(v.proj[1].y_axis.y.abs())
}

/// A frustum containing the frusta of both eyes: planes in world space,
/// normalised, `dot(n, p) + d >= 0` inside (the layout of `frustum_planes`:
/// left, right, bottom, top, near, far).
#[derive(Clone, Copy, Debug)]
pub struct UnionFrustum {
    pub planes: [Vec4; 6],
    /// World-space apex of the four side planes (behind the centre eye).
    pub apex: Vec3,
}

impl UnionFrustum {
    /// Build from the two eyes. The four side planes pass through an apex set
    /// back from the centre eye (so the union is tight at depth rather than
    /// wide at the eyes) and bound every corner of both frusta; near is the
    /// nearer of the two near planes and far the farther far plane, measured
    /// along the centre eye's forward axis.
    pub fn from_eyes(v: &StereoView) -> Self {
        let centre_view = centre_eye(v);
        let centre_world = centre_view.inverse();
        let ipd = eye_separation(v);
        // Corners of both frusta in the centre eye's space (-z forward).
        let mut corners = [Vec3::ZERO; 16];
        let mut n = 0;
        let mut near = f32::MAX;
        let mut far = 0.0f32;
        let mut min_half_fov_tan = f32::MAX;
        for e in 0..2 {
            let inv = (v.proj[e] * v.view[e]).inverse();
            for z in [0.0f32, 1.0] {
                for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
                    let w = inv * Vec4::new(x, y, z, 1.0);
                    corners[n] = centre_view.transform_point3(w.truncate() / w.w);
                    n += 1;
                }
            }
            // Depth range and the narrowest half angle of this eye.
            let p = v.proj[e];
            let (nr, fr) = depth_range(&p);
            near = near.min(nr);
            far = far.max(fr);
            min_half_fov_tan = min_half_fov_tan.min((1.0 / p.x_axis.x.abs().max(1e-6)).min(1.0 / p.y_axis.y.abs().max(1e-6)));
        }
        // Apex back along +z so the eyes' lateral offset fits inside the narrowest half angle.
        let mut back = ipd * 0.5 / min_half_fov_tan.max(1e-3);
        // Every corner must be in front of the apex.
        let nearest_z = corners.iter().map(|c| -c.z).fold(f32::MAX, f32::min);
        if nearest_z + back <= 1e-4 {
            back = -nearest_z + 1e-3;
        }
        let apex_view = Vec3::new(0.0, 0.0, back);
        // Extreme tangents of all corners seen from the apex.
        let (mut l, mut r, mut b, mut t) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for c in &corners {
            let d = -(c.z - apex_view.z); // depth along forward from the apex
            let d = d.max(1e-4);
            l = l.min(c.x / d);
            r = r.max(c.x / d);
            b = b.min(c.y / d);
            t = t.max(c.y / d);
        }
        // Planes through the apex in centre-eye space (inside = positive side).
        let side = |n: Vec3| {
            let n = n.normalize();
            Vec4::new(n.x, n.y, n.z, -n.dot(apex_view))
        };
        let view_planes = [
            side(Vec3::new(1.0, 0.0, l)),   // left: x >= l * depth
            side(Vec3::new(-1.0, 0.0, -r)), // right
            side(Vec3::new(0.0, 1.0, b)),   // bottom
            side(Vec3::new(0.0, -1.0, -t)), // top
            Vec4::new(0.0, 0.0, -1.0, -near), // near: -z - near >= 0
            Vec4::new(0.0, 0.0, 1.0, far),  // far: far + z >= 0
        ];
        // To world space: p_view = centre_view * p_world, so plane_world = plane_view * centre_view.
        let mut planes = [Vec4::ZERO; 6];
        for (i, p) in view_planes.iter().enumerate() {
            let w = centre_view.transpose() * *p;
            let len = w.truncate().length().max(1e-12);
            planes[i] = w / len;
        }
        Self { planes, apex: centre_world.transform_point3(apex_view) }
    }

    /// Whether the sphere is entirely outside (the cull's `outside_frustum`).
    pub fn outside(&self, c: Vec3, r: f32) -> bool {
        self.planes.iter().any(|p| p.truncate().dot(c) + p.w < -r)
    }
}

/// Near and far distances of a Vulkan 0..1 reverse-or-forward perspective.
fn depth_range(p: &Mat4) -> (f32, f32) {
    // For a standard perspective: z_ndc = (c * z_eye + d) / -z_eye, c = p[2][2], d = p[3][2].
    let (c, d) = (p.z_axis.z, p.w_axis.z);
    let n = d / c;
    let f = if (c + 1.0).abs() > 1e-9 { d / (c + 1.0) } else { 1e30 };
    // Forward-Z: c = f / (n - f); reverse-Z swaps them. Report (min, max).
    (n.abs().min(f.abs()), n.abs().max(f.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    fn eyes(ipd: f32, yaw: f32) -> StereoView {
        let head = Mat4::from_rotation_translation(Quat::from_rotation_y(yaw), Vec3::new(0.3, 1.6, -0.7));
        let eye = |sign: f32, cant: f32| {
            (head * Mat4::from_rotation_translation(Quat::from_rotation_y(cant), Vec3::new(sign * ipd * 0.5, 0.0, 0.0))).inverse()
        };
        // Asymmetric per-eye projections like a headset's (wider outward).
        let proj = |l: f32, r: f32| {
            let (n, f) = (0.05f32, 200.0f32);
            let (tl, tr, tb, tt) = (-l, r, -1.0f32, 1.1f32);
            Mat4::from_cols(
                Vec4::new(2.0 / (tr - tl), 0.0, 0.0, 0.0),
                Vec4::new(0.0, 2.0 / (tt - tb), 0.0, 0.0),
                Vec4::new((tr + tl) / (tr - tl), (tt + tb) / (tt - tb), f / (n - f), -1.0),
                Vec4::new(0.0, 0.0, n * f / (n - f), 0.0),
            )
        };
        StereoView { view: [eye(-1.0, 0.05), eye(1.0, -0.05)], proj: [proj(1.3, 0.9), proj(0.9, 1.3)] }
    }

    fn inside_eye(v: &StereoView, e: usize, p: Vec3) -> bool {
        let clip = v.proj[e] * v.view[e] * p.extend(1.0);
        clip.w > 0.0 && clip.x.abs() <= clip.w && clip.y.abs() <= clip.w && clip.z >= 0.0 && clip.z <= clip.w
    }

    #[test]
    fn centre_eye_is_between_the_eyes() {
        let v = eyes(0.064, 0.4);
        let c = centre_eye(&v).inverse();
        let p = |m: &Mat4| m.inverse().w_axis.truncate();
        let mid = (p(&v.view[0]) + p(&v.view[1])) * 0.5;
        assert!(c.w_axis.truncate().distance(mid) < 1e-5);
        assert!((eye_separation(&v) - 0.064).abs() < 1e-5);
        // Its forward axis lies between the eyes' (the cants cancel).
        let fwd = |m: &Mat4| -m.inverse().z_axis.truncate().normalize();
        let f = -c.z_axis.truncate().normalize();
        let want = (fwd(&v.view[0]) + fwd(&v.view[1])).normalize();
        assert!(f.dot(want) > 0.9999);
    }

    #[test]
    fn union_contains_every_point_either_eye_sees() {
        let mut seed = 99u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        for (ipd, yaw) in [(0.064, 0.0), (0.07, 1.3), (0.058, -2.0), (0.0, 0.5)] {
            let v = eyes(ipd, yaw);
            let u = UnionFrustum::from_eyes(&v);
            let (mut seen, mut missed, mut union_only) = (0, 0, 0);
            for _ in 0..40_000 {
                let p = Vec3::new(rnd() * 80.0 - 40.0, rnd() * 40.0 - 20.0, rnd() * 80.0 - 40.0) + Vec3::new(0.3, 1.6, -0.7);
                let any = inside_eye(&v, 0, p) || inside_eye(&v, 1, p);
                let in_union = !u.outside(p, 0.0);
                if any {
                    seen += 1;
                    if !in_union {
                        missed += 1;
                    }
                } else if in_union {
                    union_only += 1;
                }
            }
            assert!(seen > 500, "the sampling should hit the frusta ({seen})");
            assert_eq!(missed, 0, "ipd {ipd}: {missed} of {seen} points visible to an eye fell outside the union");
            // Not absurdly loose: the union adds at most a fraction of what the eyes see.
            assert!(union_only < seen * 2, "ipd {ipd}: union holds {union_only} extra points vs {seen}");
        }
    }

    #[test]
    fn union_contains_spheres_touching_either_frustum() {
        let v = eyes(0.064, 0.9);
        let u = UnionFrustum::from_eyes(&v);
        let mut seed = 5u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        for _ in 0..20_000 {
            let c = Vec3::new(rnd() * 30.0 - 15.0, rnd() * 10.0, rnd() * 30.0 - 15.0);
            let r = rnd() * 2.0;
            // A sphere whose centre is inside an eye's frustum is certainly not culled.
            if inside_eye(&v, 0, c) || inside_eye(&v, 1, c) {
                assert!(!u.outside(c, r));
            }
        }
    }

    #[test]
    fn lod_slack_bounds_the_centre_eye_error() {
        // Distance from the centre eye vs either eye differs by at most ipd / 2.
        let v = eyes(0.064, 0.0);
        let p = |m: &Mat4| m.inverse().w_axis.truncate();
        let c = centre_eye(&v).inverse().w_axis.truncate();
        let ipd = eye_separation(&v);
        for d in [0.3f32, 1.0, 3.0, 10.0] {
            let at = c + Vec3::new(0.2, 0.1, -1.0).normalize() * d;
            let dc = at.distance(c);
            for e in 0..2 {
                let de = at.distance(p(&v.view[e]));
                assert!((de - dc).abs() <= ipd * 0.5 + 1e-4);
                // Error projects with 1 / distance: relative difference is bounded by the slack.
                assert!((dc / de - 1.0).abs() <= lod_slack(ipd, de.min(dc)) * 1.0001 + 1e-6);
            }
        }
        assert!(lod_slack(0.064, 1.0) < 0.033 && lod_slack(0.064, 3.0) < 0.011);
        assert!(lod_slack(0.064, 0.0) > 1e30);
        assert!(pixels_per_unit(&v, 2000.0) >= 0.5 * 2000.0 * v.proj[0].y_axis.y);
    }

    #[test]
    fn depth_range_reads_the_projection() {
        let v = eyes(0.064, 0.0);
        let (n, f) = depth_range(&v.proj[0]);
        assert!((n - 0.05).abs() < 1e-4 && (f - 200.0).abs() < 0.1, "{n} {f}");
    }
}
