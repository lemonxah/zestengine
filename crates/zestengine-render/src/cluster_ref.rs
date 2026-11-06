//! Rust reference of `shaders/cluster_cull.comp` (GEO-4): the same constants,
//! the same operation order, on the same repr(C) records. The shader is the
//! product; this exists so the math can be tested without a device (against the
//! cooked DAG's own `ClusterDag::select_bvh` in zestengine-assets) and so a run
//! on a real device can be compared with it.
//!
//! Keep it in step with the shader: every function names the GLSL function it
//! mirrors.

use crate::cluster::{BvhNodeRec, ClusterRec};

pub const BVH_EMPTY: u32 = u32::MAX;
pub const BVH_LEAF: u32 = 0x8000_0000;

/// The view the cull tests against (`View` in hiz_test.glsl, the fields the
/// cluster pass reads).
#[derive(Clone, Copy, Debug)]
pub struct RefView {
    /// left, right, bottom, top, near, far (normalised, `dot(n, p) + d >= 0` inside).
    pub planes: [[f32; 4]; 6],
    pub cam: [f32; 3],
    /// Instance distance limit (0 = off).
    pub max_distance: f32,
    /// Pixels per world unit at distance 1 (`0.5 * height * proj[1][1]`).
    pub pixels_per_unit: f32,
    /// Target error in pixels.
    pub target_px: f32,
    /// Test the near plane.
    pub near_test: bool,
    /// Relative loosening used by tests to find results that flip under a tiny
    /// numeric difference (0 = the shader's exact rule).
    pub slack: f32,
}

/// One drawn instance: model matrix (column major) and the object-space camera
/// for the cone test (`eye[3] = 1` when valid).
#[derive(Clone, Copy, Debug)]
pub struct RefInstance {
    pub model: [[f32; 4]; 4],
    pub eye: [f32; 4],
}

fn col(m: &[[f32; 4]; 4], c: usize) -> [f32; 3] {
    [m[c][0], m[c][1], m[c][2]]
}

fn len3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// `max_scale` in the shader.
pub fn max_scale(m: &[[f32; 4]; 4]) -> f32 {
    len3(col(m, 0)).max(len3(col(m, 1))).max(len3(col(m, 2)))
}

/// `model * vec4(p, 1)`.
pub fn transform_point(m: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|r| m[0][r] * p[0] + m[1][r] * p[1] + m[2][r] * p[2] + m[3][r])
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    len3([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
}

/// `lod_ok`: the object-space error of `sphere` projects to at most the target.
/// Monotone in (error, sphere), as `ClusterDag::select_*` requires.
pub fn lod_ok(v: &RefView, m: &[[f32; 4]; 4], s: f32, err: f32, sphere: [f32; 4]) -> bool {
    let c = transform_point(m, [sphere[0], sphere[1], sphere[2]]);
    let d = (distance(c, v.cam) - sphere[3] * s).max(0.0);
    err * s * v.pixels_per_unit <= v.target_px * (1.0 + v.slack) * d
}

/// `outside_frustum`.
pub fn outside_frustum(v: &RefView, c: [f32; 3], r: f32) -> bool {
    for (k, p) in v.planes.iter().enumerate() {
        if k == 4 && !v.near_test {
            continue;
        }
        if p[0] * c[0] + p[1] * c[1] + p[2] * c[2] + p[3] < -(r * (1.0 + v.slack)) - v.slack {
            return true;
        }
    }
    false
}

/// `cone_culled`.
pub fn cone_culled(cl: &ClusterRec, eye: [f32; 4], slack: f32) -> bool {
    let cutoff = cl.cone[3];
    if eye[3] < 0.5 || cutoff == u8::MAX {
        return false;
    }
    let axis = [cl.cone[0] as i8 as f32 / 127.0, cl.cone[1] as i8 as f32 / 127.0, cl.cone[2] as i8 as f32 / 127.0];
    let d = [cl.bounds[0] - eye[0], cl.bounds[1] - eye[1], cl.bounds[2] - eye[2]];
    let dot = axis[0] * d[0] + axis[1] * d[1] + axis[2] * d[2];
    dot >= (cutoff as f32 / 255.0 * len3(d) + cl.bounds[3]) * (1.0 + slack)
}

/// Why a tested cluster did not end up drawn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RefCounts {
    pub nodes: u32,
    pub tested: u32,
    pub lod_rejected: u32,
    pub frustum: u32,
    pub cone: u32,
    pub visible: u32,
    pub triangles: u32,
}

/// The cut selection + cluster culls of one instance over a whole DAG:
/// the clusters (ascending) the shader appends for it with Hi-Z off.
/// `node_cull = false` skips the node-level frustum prune (what remains is the
/// plain `select_bvh` followed by the per-cluster tests).
pub fn cut(
    clusters: &[ClusterRec],
    nodes: &[BvhNodeRec],
    v: &RefView,
    inst: &RefInstance,
    node_cull: bool,
) -> (Vec<u32>, RefCounts) {
    let s = max_scale(&inst.model);
    let mut out = Vec::new();
    let mut counts = RefCounts::default();
    let mut stack = vec![0u32];
    while let Some(n) = stack.pop() {
        counts.nodes += 1;
        for ch in &nodes[n as usize].children {
            if ch.child == BVH_EMPTY || lod_ok(v, &inst.model, s, ch.max_parent_error, ch.lod_sphere) {
                continue;
            }
            if node_cull {
                let bc = transform_point(&inst.model, [ch.bounds[0], ch.bounds[1], ch.bounds[2]]);
                if outside_frustum(v, bc, ch.bounds[3] * s) {
                    continue;
                }
            }
            if ch.child & BVH_LEAF == 0 {
                stack.push(ch.child);
                continue;
            }
            let first = ch.child & !BVH_LEAF;
            for k in first..first + ch.count {
                let cl = &clusters[k as usize];
                counts.tested += 1;
                if !(lod_ok(v, &inst.model, s, cl.lod_error, cl.lod_sphere)
                    && !lod_ok(v, &inst.model, s, cl.parent_lod_error, cl.parent_lod_sphere))
                {
                    counts.lod_rejected += 1;
                    continue;
                }
                let c = transform_point(&inst.model, [cl.bounds[0], cl.bounds[1], cl.bounds[2]]);
                if outside_frustum(v, c, cl.bounds[3] * s) {
                    counts.frustum += 1;
                    continue;
                }
                if cone_culled(cl, inst.eye, v.slack) {
                    counts.cone += 1;
                    continue;
                }
                counts.visible += 1;
                counts.triangles += cl.triangle_count as u32;
                out.push(k);
            }
        }
    }
    out.sort_unstable();
    (out, counts)
}

/// A pinhole view from a camera at `eye` looking at `target`: planes from the
/// view-projection (Vulkan clip space, 0..1 depth), `fov_y` radians.
#[allow(clippy::too_many_arguments)]
pub fn make_view(
    eye: glam::Vec3,
    target: glam::Vec3,
    fov_y: f32,
    aspect: f32,
    height_px: f32,
    near: f32,
    far: f32,
    target_px: f32,
) -> RefView {
    let view = glam::Mat4::look_at_rh(eye, target, glam::Vec3::Y);
    let proj = glam::Mat4::perspective_rh(fov_y, aspect, near, far);
    RefView {
        planes: crate::frustum_planes(proj * view).map(|p| p.to_array()),
        cam: eye.to_array(),
        max_distance: 0.0,
        pixels_per_unit: 0.5 * height_px * proj.y_axis.y.abs(),
        target_px,
        near_test: true,
        slack: 0.0,
    }
}

/// The object-space eye for the cone test, as `Renderer` computes it for an
/// instance: valid only for a similarity transform (uniform scale, no mirror),
/// where the cone test is exact.
pub fn object_space_eye(model: glam::Mat4, cam: glam::Vec3) -> [f32; 4] {
    let x = model.x_axis.truncate().length();
    let y = model.y_axis.truncate().length();
    let z = model.z_axis.truncate().length();
    let (lo, hi) = (x.min(y).min(z), x.max(y).max(z));
    let det = model.determinant();
    if lo.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) || hi / lo > 1.01 || det <= 0.0 || !det.is_finite() {
        return [0.0; 4];
    }
    let e = model.inverse().transform_point3(cam);
    [e.x, e.y, e.z, 1.0]
}

/// Octahedral encoding of +Z (tests).
pub fn oct_encode_z() -> [i16; 2] {
    [0, 0]
}
