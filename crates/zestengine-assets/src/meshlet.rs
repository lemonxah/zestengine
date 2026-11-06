//! Meshlet (cluster) generation: partition a triangle mesh into small clusters
//! of shared vertices, the input a mesh-shader or cluster-culling pipeline
//! needs. Greedy clustering by index order with per-cluster vertex/triangle
//! budgets and local triangle indexing.
//!
//! Superseded for the GPU cluster path by [`crate::cluster_dag`] (spatial
//! partitioner, simplification DAG, quantised cluster data); this stays for its
//! current users until they migrate.

use std::collections::HashMap;

use zestengine_render::MeshData;

/// One meshlet: a small cluster of triangles with shared vertices. Local
/// `indices` (three per triangle) address `vertices`.
#[derive(Clone, Debug, PartialEq)]
pub struct Meshlet {
    /// Unique vertex indices into the source mesh (≤ `max_vertices`).
    pub vertices: Vec<u32>,
    /// Triangle indices into `vertices` (≤ `max_triangles * 3`).
    pub indices: Vec<u32>,
    /// World-space axis-aligned bounds of the cluster (for culling).
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
}

struct MeshletBuilder {
    vertices: Vec<u32>,
    indices: Vec<u32>,
    /// Source vertex index -> local index in `vertices`.
    map: HashMap<u32, u32>,
    bmin: [f32; 3],
    bmax: [f32; 3],
}

impl MeshletBuilder {
    fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            map: HashMap::new(),
            bmin: [f32::MAX; 3],
            bmax: [f32::MIN; 3],
        }
    }

    fn add_triangle(&mut self, tri: [u32; 3], mesh: &MeshData) {
        for v in tri {
            let local = *self.map.entry(v).or_insert_with(|| {
                let i = self.vertices.len() as u32;
                self.vertices.push(v);
                let p = mesh.vertices[v as usize].position;
                self.bmin[0] = self.bmin[0].min(p[0]);
                self.bmin[1] = self.bmin[1].min(p[1]);
                self.bmin[2] = self.bmin[2].min(p[2]);
                self.bmax[0] = self.bmax[0].max(p[0]);
                self.bmax[1] = self.bmax[1].max(p[1]);
                self.bmax[2] = self.bmax[2].max(p[2]);
                i
            });
            self.indices.push(local);
        }
    }

    fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// How many new vertices `tri` would add to this cluster.
    fn would_add(&self, tri: [u32; 3]) -> usize {
        tri.iter().filter(|v| !self.map.contains_key(v)).count()
    }

    fn finish(self) -> Meshlet {
        Meshlet {
            vertices: self.vertices,
            indices: self.indices,
            bounds_min: self.bmin,
            bounds_max: self.bmax,
        }
    }
}

/// Greedily partition a triangle mesh into meshlets of at most `max_vertices`
/// vertices and `max_triangles` triangles. Triangles are visited in index
/// order; every triangle lands in exactly one meshlet.
pub fn build_meshlets(mesh: &MeshData, max_vertices: usize, max_triangles: usize) -> Vec<Meshlet> {
    let tri_count = mesh.indices.len() / 3;
    let mut out = Vec::new();
    let mut cur: Option<MeshletBuilder> = None;
    for t in 0..tri_count {
        let tri = [mesh.indices[t * 3], mesh.indices[t * 3 + 1], mesh.indices[t * 3 + 2]];
        let full = match &cur {
            Some(b) => b.vertex_count() + b.would_add(tri) > max_vertices || b.triangle_count() + 1 > max_triangles,
            None => false,
        };
        if full {
            out.push(cur.take().unwrap().finish());
        }
        cur.get_or_insert_with(MeshletBuilder::new).add_triangle(tri, mesh);
    }
    if let Some(b) = cur {
        out.push(b.finish());
    }
    out
}

/// Decimate `mesh` to `keep` of its triangles with [`crate::simplify::simplify`]
/// and partition the result into meshlet clusters. Returns the simplified mesh
/// (the LOD to upload) alongside its clusters (the input a mesh-shader or
/// cluster-culling pipeline consumes).
pub fn build_lod_meshlets(
    mesh: &MeshData,
    keep: f32,
    max_vertices: usize,
    max_triangles: usize,
) -> (MeshData, Vec<Meshlet>) {
    let lod = crate::simplify::simplify(mesh, keep);
    let meshlets = build_meshlets(&lod, max_vertices, max_triangles);
    (lod, meshlets)
}

/// [`build_lod_meshlets`] that also returns each LOD vertex's source vertex
/// (see [`crate::simplify::simplify_with_provenance`]), so blendshapes can be
/// remapped onto the generated LOD.
pub fn build_lod_meshlets_mapped(
    mesh: &MeshData,
    keep: f32,
    max_vertices: usize,
    max_triangles: usize,
) -> (MeshData, Vec<Meshlet>, Vec<u32>) {
    let (lod, provenance) = crate::simplify::simplify_with_provenance(mesh, keep);
    let meshlets = build_meshlets(&lod, max_vertices, max_triangles);
    (lod, meshlets, provenance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zestengine_render::Vertex;

    fn quad() -> MeshData {
        let mut vertices = Vec::new();
        for p in [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]] {
            let v = Vertex { position: p, ..Default::default() };
            vertices.push(v);
        }
        MeshData { vertices, indices: vec![0, 1, 2, 0, 2, 3], has_lightmap_uv: false }
    }

    #[test]
    fn meshlets_preserve_every_triangle() {
        let mesh = quad();
        let meshlets = build_meshlets(&mesh, 64, 126);
        let total: usize = meshlets.iter().map(|m| m.indices.len() / 3).sum();
        assert_eq!(total, mesh.indices.len() / 3);
    }

    #[test]
    fn meshlets_respect_budgets() {
        let mesh = quad();
        // Tight budgets force the 2-triangle quad into 2 meshlets (3 unique
        // vertices, one triangle each).
        let meshlets = build_meshlets(&mesh, 3, 1);
        assert_eq!(meshlets.len(), 2);
        for m in &meshlets {
            assert!(m.vertices.len() <= 3);
            assert!(m.indices.len() / 3 <= 1);
        }
    }

    #[test]
    fn meshlet_indices_are_valid_and_bounds_enclose() {
        let mesh = quad();
        let meshlets = build_meshlets(&mesh, 64, 126);
        assert_eq!(meshlets.len(), 1);
        let m = &meshlets[0];
        assert!(m.indices.iter().all(|&i| (i as usize) < m.vertices.len()));
        for &i in &m.vertices {
            let p = mesh.vertices[i as usize].position;
            assert!(p[0] >= m.bounds_min[0] && p[0] <= m.bounds_max[0]);
            assert!(p[1] >= m.bounds_min[1] && p[1] <= m.bounds_max[1]);
        }
    }

    #[test]
    fn lod_meshlets_cover_the_simplified_mesh() {
        let sphere = crate::procedural::primitive_mesh(crate::PrimitiveShape::Sphere);
        let (lod, meshlets) = build_lod_meshlets(&sphere, 0.5, 64, 126);
        assert!(lod.indices.len() / 3 < sphere.indices.len() / 3, "decimation reduced the mesh");
        let total: usize = meshlets.iter().map(|m| m.indices.len() / 3).sum();
        assert_eq!(total, lod.indices.len() / 3, "every simplified triangle lands in a meshlet");
    }

    #[test]
    fn lod_meshlets_respect_budgets_and_bounds() {
        let sphere = crate::procedural::primitive_mesh(crate::PrimitiveShape::Sphere);
        let (lod, meshlets) = build_lod_meshlets(&sphere, 0.25, 64, 126);
        assert!(!meshlets.is_empty());
        for m in &meshlets {
            assert!(m.vertices.len() <= 64);
            assert!(m.indices.len() / 3 <= 126);
            assert!(m.indices.iter().all(|&i| (i as usize) < m.vertices.len()));
            for &v in &m.vertices {
                let p = lod.vertices[v as usize].position;
                assert!(p[0] >= m.bounds_min[0] && p[0] <= m.bounds_max[0]);
                assert!(p[1] >= m.bounds_min[1] && p[1] <= m.bounds_max[1]);
                assert!(p[2] >= m.bounds_min[2] && p[2] <= m.bounds_max[2]);
            }
        }
    }
}
