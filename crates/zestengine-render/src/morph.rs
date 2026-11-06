//! GPU blendshapes: sparse per-vertex deltas packed for `morph.comp`, plus a
//! CPU reference of exactly what that shader computes (the tests, and the
//! engine's cross-check against its CPU morph path, use it).
//!
//! The pass runs before skinning and rewrites only the vertices some shape
//! touches, in place in the skinner's bind-pose buffer. Each table entry keeps
//! the vertex's original position and normal, so re-running with new weights is
//! idempotent and weights back at zero restore the bind pose.

use crate::types::Vertex;

/// Floats per touched-vertex entry: vertex, first delta, delta count (as u32
/// bits), then the bind-pose position and normal.
pub(crate) const TOUCHED_STRIDE: usize = 9;
/// Floats per delta: shape index (u32 bits), position offset, normal offset.
pub(crate) const DELTA_STRIDE: usize = 7;
/// Weights at or below this magnitude count as zero (matches the CPU path).
pub(crate) const WEIGHT_EPS: f32 = 1e-4;

/// One blendshape's sparse deltas: `(vertex index, position offset, normal offset)`.
pub type MorphDeltas<'a> = &'a [(u32, [f32; 3], [f32; 3])];

/// Packed sparse blendshape data for one mesh.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MorphTables {
    pub(crate) touched: Vec<f32>,
    pub(crate) deltas: Vec<f32>,
    pub(crate) shape_count: u32,
    pub(crate) touched_count: u32,
}

/// Deltas accumulated per vertex: (shape index, delta position, delta normal).
type VertexMorphs = std::collections::BTreeMap<u32, Vec<(u32, [f32; 3], [f32; 3])>>;

impl MorphTables {
    /// Pack `shapes` (one delta list per blendshape, in weight order) for
    /// `vertices`. Deltas past the end of the mesh are dropped.
    pub fn build(vertices: &[Vertex], shapes: &[MorphDeltas<'_>]) -> Self {
        let mut per_vertex: VertexMorphs = VertexMorphs::new();
        for (shape, deltas) in shapes.iter().enumerate() {
            for &(v, dp, dn) in deltas.iter() {
                if (v as usize) < vertices.len() {
                    per_vertex.entry(v).or_default().push((shape as u32, dp, dn));
                }
            }
        }
        let mut touched = Vec::with_capacity(per_vertex.len() * TOUCHED_STRIDE);
        let mut deltas = Vec::new();
        for (v, list) in &per_vertex {
            let base = &vertices[*v as usize];
            touched.push(f32::from_bits(*v));
            touched.push(f32::from_bits((deltas.len() / DELTA_STRIDE) as u32));
            touched.push(f32::from_bits(list.len() as u32));
            touched.extend(base.position);
            touched.extend(base.normal);
            for (shape, dp, dn) in list {
                deltas.push(f32::from_bits(*shape));
                deltas.extend(dp);
                deltas.extend(dn);
            }
        }
        Self {
            touched_count: per_vertex.len() as u32,
            touched,
            deltas,
            shape_count: shapes.len() as u32,
        }
    }

    /// No vertex is touched by any shape (nothing for the GPU pass to do).
    pub fn is_empty(&self) -> bool {
        self.touched_count == 0
    }
}

fn normalize_or_y(n: [f32; 3]) -> [f32; 3] {
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if l > 1e-20 && l.is_finite() { [n[0] / l, n[1] / l, n[2] / l] } else { [0.0, 1.0, 0.0] }
}

/// What `morph.comp` leaves in the vertex buffer: `vertices` with every
/// touched vertex's position/normal offset by the weighted deltas (normals
/// re-normalised), untouched vertices unchanged. Weights past the shape count
/// read as zero.
pub fn morph_reference(vertices: &[Vertex], tables: &MorphTables, weights: &[f32]) -> Vec<Vertex> {
    let mut out = vertices.to_vec();
    for e in 0..tables.touched_count as usize {
        let t = &tables.touched[e * TOUCHED_STRIDE..(e + 1) * TOUCHED_STRIDE];
        let (v, first, count) = (t[0].to_bits() as usize, t[1].to_bits() as usize, t[2].to_bits() as usize);
        let mut p = [t[3], t[4], t[5]];
        let mut n = [t[6], t[7], t[8]];
        let mut any = false;
        for d in first..first + count {
            let d = &tables.deltas[d * DELTA_STRIDE..(d + 1) * DELTA_STRIDE];
            let w = weights.get(d[0].to_bits() as usize).copied().unwrap_or(0.0);
            if w.abs() <= WEIGHT_EPS {
                continue;
            }
            any = true;
            for k in 0..3 {
                p[k] += d[1 + k] * w;
                n[k] += d[4 + k] * w;
            }
        }
        out[v].position = p;
        out[v].normal = if any { normalize_or_y(n) } else { [t[6], t[7], t[8]] };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn same(a: &[Vertex], b: &[Vertex]) -> bool {
        bytemuck::cast_slice::<Vertex, u8>(a) == bytemuck::cast_slice::<Vertex, u8>(b)
    }

    fn mesh() -> Vec<Vertex> {
        (0..5)
            .map(|i| Vertex { position: [i as f32, 0.0, 0.0], normal: [0.0, 0.0, 1.0], ..Default::default() })
            .collect()
    }

    #[test]
    fn zero_weights_leave_the_bind_pose_untouched() {
        let base = mesh();
        let jaw: Vec<(u32, [f32; 3], [f32; 3])> = vec![(1, [0.0, -1.0, 0.0], [0.0, 0.5, 0.0]), (3, [0.0, -2.0, 0.0], [0.0; 3])];
        let t = MorphTables::build(&base, &[&jaw]);
        assert_eq!(t.touched_count, 2);
        let out = morph_reference(&base, &t, &[0.0]);
        assert!(same(&out, &base));
        let out = morph_reference(&base, &t, &[5e-5]);
        assert!(same(&out, &base), "below the dead zone is zero");
    }

    #[test]
    fn weighted_deltas_add_up_and_normals_renormalise() {
        let base = mesh();
        let a: Vec<(u32, [f32; 3], [f32; 3])> = vec![(2, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0])];
        let b: Vec<(u32, [f32; 3], [f32; 3])> = vec![(2, [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]), (4, [0.0, 0.0, 3.0], [0.0; 3])];
        let t = MorphTables::build(&base, &[&a, &b]);
        let out = morph_reference(&base, &t, &[0.5, 2.0]);
        assert_eq!(out[2].position, [2.0 + 2.0, 0.5, 0.0]);
        let n = out[2].normal;
        let l = (0.25f32 + 1.0).sqrt();
        assert!((n[0] - 0.5 / l).abs() < 1e-6 && (n[2] - 1.0 / l).abs() < 1e-6, "{n:?}");
        assert_eq!(out[4].position, [4.0, 0.0, 6.0]);
        assert_eq!(out[4].normal, [0.0, 0.0, 1.0], "only a zero-delta normal: unchanged unit normal");
        assert!(same(&out[..2], &base[..2]) && same(&out[3..4], &base[3..4]));
    }

    #[test]
    fn applying_new_weights_is_idempotent_and_reversible() {
        let base = mesh();
        let a: Vec<(u32, [f32; 3], [f32; 3])> = vec![(1, [0.0, 1.0, 0.0], [0.0; 3])];
        let t = MorphTables::build(&base, &[&a]);
        // The GPU pass reads the saved bind pose, not the buffer, so the output of
        // one weight set never leaks into the next.
        let once = morph_reference(&base, &t, &[1.0]);
        let again = morph_reference(&once, &t, &[1.0]);
        assert!(same(&once, &again));
        assert!(same(&morph_reference(&once, &t, &[0.0]), &base));
    }

    #[test]
    fn out_of_range_vertices_and_missing_weights_are_ignored() {
        let base = mesh();
        let a: Vec<(u32, [f32; 3], [f32; 3])> = vec![(99, [1.0; 3], [0.0; 3]), (0, [1.0, 0.0, 0.0], [0.0; 3])];
        let t = MorphTables::build(&base, &[&a]);
        assert_eq!(t.touched_count, 1);
        assert!(same(&morph_reference(&base, &t, &[]), &base));
        assert!(MorphTables::build(&base, &[]).is_empty());
    }
}
