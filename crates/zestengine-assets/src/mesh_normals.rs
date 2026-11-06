//! Recomputing vertex normals with a crease angle (the model importer's
//! "Recalculate Normals" + "Smoothing Angle" settings).
//!
//! Works on any triangle list, indexed or not: corners that share a position are
//! grouped, and each corner averages the (corner-angle weighted) normals of the
//! faces in its group whose own normal is within the smoothing angle of the
//! corner's face. Hard edges therefore stay hard while gentle ones smooth out.

use std::collections::HashMap;

use glam::Vec3;
use zestengine_render::MeshData;

use crate::NormalWeighting;

/// Recompute `mesh`'s normals; edges sharper than `smoothing_angle_deg` stay creased.
#[allow(dead_code)] // normals are recomputed on the import path today
#[cfg_attr(not(feature = "client"), allow(dead_code))]
pub fn recalculate_normals(mesh: &mut MeshData, smoothing_angle_deg: f32) {
    recalculate_normals_weighted(mesh, smoothing_angle_deg, NormalWeighting::Angle);
}

/// [`recalculate_normals`] with a choice of how the faces around a vertex count.
pub fn recalculate_normals_weighted(mesh: &mut MeshData, smoothing_angle_deg: f32, weighting: NormalWeighting) {
    let tri_count = mesh.indices.len() / 3;
    if tri_count == 0 {
        return;
    }
    let pos = |i: u32| Vec3::from(mesh.vertices[i as usize].position);
    // Per triangle: unit face normal (zero for degenerate triangles).
    let face_normal: Vec<Vec3> = (0..tri_count)
        .map(|t| {
            let (a, b, c) = (pos(mesh.indices[t * 3]), pos(mesh.indices[t * 3 + 1]), pos(mesh.indices[t * 3 + 2]));
            (b - a).cross(c - a).normalize_or_zero()
        })
        .collect();
    let face_area: Vec<f32> = (0..tri_count)
        .map(|t| {
            let (a, b, c) = (pos(mesh.indices[t * 3]), pos(mesh.indices[t * 3 + 1]), pos(mesh.indices[t * 3 + 2]));
            (b - a).cross(c - a).length() * 0.5
        })
        .collect();
    // Corner angle at corner k of triangle t.
    let corner_angle = |t: usize, k: usize| -> f32 {
        let p = |j: usize| pos(mesh.indices[t * 3 + (k + j) % 3]);
        let (o, a, b) = (p(0), p(1), p(2));
        (a - o).normalize_or_zero().dot((b - o).normalize_or_zero()).clamp(-1.0, 1.0).acos()
    };

    // Group corners by (quantised) position.
    let quant = |v: Vec3| [(v.x * 1e5).round() as i64, (v.y * 1e5).round() as i64, (v.z * 1e5).round() as i64];
    let mut groups: HashMap<[i64; 3], Vec<(usize, usize)>> = HashMap::new();
    for (t, n) in face_normal.iter().enumerate() {
        if *n == Vec3::ZERO {
            continue;
        }
        for k in 0..3 {
            groups.entry(quant(pos(mesh.indices[t * 3 + k]))).or_default().push((t, k));
        }
    }

    let cos_limit = smoothing_angle_deg.clamp(0.0, 180.0).to_radians().cos() - 1e-5;
    // One new normal per (triangle, corner); written to the vertex afterwards so
    // indexed meshes that share a vertex between creased faces still get one value.
    let mut out: HashMap<(usize, usize), Vec3> = HashMap::new();
    for corners in groups.values() {
        for &(t, k) in corners {
            let mut sum = Vec3::ZERO;
            for &(u, j) in corners {
                if face_normal[t].dot(face_normal[u]) >= cos_limit {
                    let w = match weighting {
                        NormalWeighting::Angle => corner_angle(u, j),
                        NormalWeighting::Area => face_area[u],
                        NormalWeighting::AreaAndAngle => face_area[u] * corner_angle(u, j),
                        NormalWeighting::Unweighted => 1.0,
                    };
                    sum += face_normal[u] * w;
                }
            }
            out.insert((t, k), sum.normalize_or(face_normal[t]));
        }
    }
    for ((t, k), n) in out {
        mesh.vertices[mesh.indices[t * 3 + k] as usize].normal = n.to_array();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_render::Vertex;

    fn vert(p: [f32; 3]) -> Vertex {
        Vertex { position: p, normal: [0.0, 1.0, 0.0], uv: [0.0; 2], color: [1.0; 4], tangent: [1.0, 0.0, 0.0, 1.0], uv1: [0.0; 2], joints: [0; 4], weights: [0.0; 4] }
    }

    /// An un-indexed unit cube (12 triangles, 36 vertices) centred on the origin.
    fn cube() -> MeshData {
        let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
            ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
            ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
            ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
            ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
        ];
        let mut vertices = Vec::new();
        for (n, u, v) in faces {
            let (n, u, v) = (Vec3::from(n), Vec3::from(u), Vec3::from(v));
            let c = |a: f32, b: f32| (n + u * a + v * b) * 0.5;
            for p in [c(-1.0, -1.0), c(1.0, -1.0), c(1.0, 1.0), c(-1.0, -1.0), c(1.0, 1.0), c(-1.0, 1.0)] {
                vertices.push(vert(p.to_array()));
            }
        }
        let indices = (0..vertices.len() as u32).collect();
        MeshData { vertices, indices, has_lightmap_uv: false }
    }

    #[test]
    fn right_angle_edges_stay_hard_below_the_crease_angle() {
        let mut m = cube();
        recalculate_normals(&mut m, 60.0);
        for tri in m.vertices.chunks(3) {
            let face = (Vec3::from(tri[1].position) - Vec3::from(tri[0].position)).cross(Vec3::from(tri[2].position) - Vec3::from(tri[0].position)).normalize();
            for v in tri {
                assert!((Vec3::from(v.normal) - face).length() < 1e-4, "flat normal, got {:?} want {face:?}", v.normal);
            }
        }
    }

    #[test]
    fn right_angle_edges_smooth_above_the_crease_angle() {
        let mut m = cube();
        recalculate_normals(&mut m, 120.0);
        for v in &m.vertices {
            let p = Vec3::from(v.position).normalize();
            assert!((Vec3::from(v.normal) - p).length() < 1e-3, "smooth normal points away from the centre");
        }
    }

    #[test]
    fn indexed_coplanar_quad_gets_one_normal() {
        let mut m = MeshData {
            vertices: vec![vert([0.0, 0.0, 0.0]), vert([1.0, 0.0, 0.0]), vert([1.0, 0.0, -1.0]), vert([0.0, 0.0, -1.0])],
            indices: vec![0, 1, 2, 0, 2, 3],
            has_lightmap_uv: false,
        };
        for v in &mut m.vertices {
            v.normal = [1.0, 0.0, 0.0];
        }
        recalculate_normals(&mut m, 0.0);
        for v in &m.vertices {
            assert!((Vec3::from(v.normal) - Vec3::Y).length() < 1e-5);
        }
    }

    /// A vertex shared by a big flat floor triangle and a tiny wall triangle.
    fn floor_and_wall() -> MeshData {
        let mut m = MeshData {
            vertices: vec![
                vert([0.0, 0.0, 0.0]),
                vert([10.0, 0.0, 0.0]),
                vert([0.0, 0.0, -10.0]),
                vert([0.1, 0.0, 0.0]),
                vert([0.0, 0.1, 0.0]),
            ],
            // floor faces +Y; the wall (in the XY plane) faces +Z
            indices: vec![0, 1, 2, 0, 3, 4],
            has_lightmap_uv: false,
        };
        for v in &mut m.vertices {
            v.normal = [0.0; 3];
        }
        m
    }

    #[test]
    fn weighting_changes_how_much_a_small_face_pulls_a_shared_vertex() {
        let normal_at_origin = |w: NormalWeighting| {
            let mut m = floor_and_wall();
            recalculate_normals_weighted(&mut m, 180.0, w);
            Vec3::from(m.vertices[0].normal)
        };
        // Floor normal is +Y, wall normal is +Z (or -Z: whichever the winding gives).
        let by_area = normal_at_origin(NormalWeighting::Area);
        let flat = normal_at_origin(NormalWeighting::Unweighted);
        let angle = normal_at_origin(NormalWeighting::Angle);
        assert!(by_area.y > 0.99, "the huge floor face dominates by area: {by_area:?}");
        assert!(flat.y < by_area.y - 0.1, "unweighted lets the tiny face pull as hard as the big one: {flat:?}");
        assert!((flat.y - flat.z.abs()).abs() < 1e-3, "two faces, equal pull: {flat:?}");
        assert!(angle.length() > 0.99);
        // Area and angle together is not the same as either alone.
        let both = normal_at_origin(NormalWeighting::AreaAndAngle);
        assert!(both.y >= by_area.y - 1e-3 || both.y > angle.y);
        // Calling the old entry point is the angle weighting.
        let mut m = floor_and_wall();
        recalculate_normals(&mut m, 180.0);
        assert!((Vec3::from(m.vertices[0].normal) - angle).length() < 1e-6);
    }

    #[test]
    fn degenerate_and_empty_meshes_are_left_alone() {
        let mut empty = MeshData { vertices: Vec::new(), indices: Vec::new(), has_lightmap_uv: false };
        recalculate_normals(&mut empty, 60.0);
        let mut m = MeshData { vertices: vec![vert([0.0; 3]); 3], indices: vec![0, 1, 2], has_lightmap_uv: false };
        recalculate_normals(&mut m, 60.0);
        assert_eq!(m.vertices[0].normal, [0.0, 1.0, 0.0]);
    }
}
