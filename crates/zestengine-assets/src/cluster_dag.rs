//! Cluster DAG geometry (GEO-3): a mesh cooked into clusters of at most 128
//! triangles arranged as a directed acyclic graph of levels of detail, the
//! data a GPU-driven cluster renderer culls and selects from.
//!
//! The finest level is the (grid-snapped) source mesh cut into clusters.
//! Groups of 8 to 32 neighbouring clusters are merged, simplified to about
//! half their triangles with the group's outer edges pinned, and split into
//! new clusters one level up, until a single cluster per connected piece is
//! left. A cluster carries the object-space `lod_error` of its group and the
//! `lod_sphere` it is judged on; both only grow going up the DAG (parent error
//! is the largest child error plus the group's own simplification error; the
//! parent sphere encloses the children's). A viewer draws a cluster when its
//! own error is small enough on screen and its parent's is not. Every
//! cluster a group produces shares one error and one sphere, and a group's
//! outline never changes, so any such cut is a closed surface without cracks
//! (the outputs of one group may be consumed by different groups one level up,
//! which is what moves the group boundaries from level to level).
//!
//! This supersedes the greedy index-order [`crate::meshlet`] partitioner for
//! the cluster path; `meshlet` stays for its current users.
//!
//! # Layout the GPU stage reads
//!
//! Everything below is `#[repr(C)]` plain data, little endian. A cluster's
//! vertices and triangle bytes are contiguous, so one cluster is one small
//! upload range.
//!
//! | struct / array        | bytes | holds |
//! |-----------------------|-------|-------|
//! | [`GpuCluster`]        | 96    | per-cluster culling, LOD, ranges |
//! | [`ClusterVertex`]     | 28    | quantised position, packed normal and tangent, UV, RGBA8 vertex colour |
//! | [`ClusterVertexSkin`] | 8     | cluster-local bone indices and weights (only if [`FLAG_SKINNED`]) |
//! | triangle bytes        | 3     | per triangle, cluster-local vertex indices; each cluster's run padded to 4 |
//! | [`MorphEntry`]        | 16    | sparse blendshape delta for a cluster vertex |
//! | [`BvhNode`]           | 192   | four [`BvhChild`] of 48 bytes: the hierarchy over runs of clusters |
//!
//! # Quantisation
//!
//! The mesh is snapped to one grid before building: `step` is the smallest
//! power of two with `extent / step <= 65535`, anchored at the mesh minimum,
//! so a position is three `u16` and decodes to `min + q * step` exactly (the
//! multiply is exact; one rounding in the add, the same on CPU and GPU). One
//! grid for the whole mesh is what keeps neighbouring clusters, including
//! clusters of different levels, bit-identical along shared edges. The error
//! of snapping is at most `step * sqrt(3) / 2`. Normals and tangents are
//! octahedral `i16` pairs (angular error below 0.012 degrees); the tangent's
//! handedness is bit 0 of [`ClusterVertex::flags`].

mod build;
mod partition;
#[cfg(all(test, feature = "client"))]
mod gpu_tests;
#[cfg(test)]
mod tests;

use bytemuck::{Pod, Zeroable};
use zestengine_render::{MeshData, Vertex};

pub use build::{CANCELLED, needs_extra_streams, build_cluster_dag, build_cluster_dag_ctl, build_cluster_dag_progress, thread_count};

/// Bump when the builder's output changes for the same input (part of the cache key).
/// 5: groups stuck above their target are retried allowing attribute jumps at seams, with whole small
/// pieces (cards, buttons) pruned when collapses run dry, and a level that removes under 5% of the
/// triangles counts as a stall (the avatar's root cuts fell from 16 273 to ~4 700 triangles).
/// 4: the group simplifier's quadrics start empty (they started as the identity, which added
/// `|v|^2 + 1` per merged vertex to every collapse cost), and a piece that simplifies away entirely
/// is judged by its own size instead of its distance to the next surviving surface.
/// 3: the vertex colour moved into `ClusterVertex` (28 bytes, white when the mesh has none; the separate
/// extra stream is gone), and coloured meshes simplify with a colour cost term, so cached DAGs are rebuilt.
/// 2: a lightmap UV set alone no longer sets `FLAG_EXTRA` (only real vertex colours do), so DAGs cooked
/// before it, which refused those meshes, are rejected and rebuilt once.
pub const GEO_BUILDER_VERSION: u32 = 5;
/// Meshes with fewer triangles skip the DAG: one cluster or two gain nothing.
pub const GEO_MIN_TRIANGLES: usize = 2000;
pub const CLUSTER_MAX_TRIS: usize = 128;
pub const CLUSTER_MAX_VERTS: usize = 256;
pub const CLUSTER_MAX_BONES: usize = 16;
/// `parent_lod_error` of a root cluster: never accurate enough, so it is always drawable.
pub const ROOT_PARENT_ERROR: f32 = f32::MAX;
pub const NO_LINK: u32 = u32::MAX;
/// [`BvhChild::child`] of an unused slot.
pub const BVH_EMPTY: u32 = u32::MAX;
/// [`BvhChild::child`] bit marking a leaf (the low bits are the first cluster).
pub const BVH_LEAF: u32 = 0x8000_0000;
/// Clusters per BVH leaf run, at most.
pub const BVH_RUN_MAX: usize = 8;
pub const BVH_MAX_DEPTH: usize = 32;
/// Hard cap when reading untrusted data.
pub const MAX_CLUSTERS: usize = 1 << 24;

/// [`DagHeader::flags`]: the mesh has real (non-white) vertex colours; the colours themselves are in every `ClusterVertex`.
pub const FLAG_EXTRA: u32 = 1;
/// [`DagHeader::flags`]: skin stream and per-cluster bone lists present.
pub const FLAG_SKINNED: u32 = 2;

/// One cluster. 96 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuCluster {
    /// Culling sphere over the cluster's own (decoded) vertices: xyz, radius.
    pub bounds: [f32; 4],
    /// Sphere the LOD error of this cluster is judged on. Shared by all
    /// clusters of one group; encloses the children's.
    pub lod_sphere: [f32; 4],
    /// The same for the group that consumes this cluster (own values for a root).
    pub parent_lod_sphere: [f32; 4],
    /// Object-space error of this cluster; 0 for the finest level.
    pub lod_error: f32,
    /// Error of the consuming group, [`ROOT_PARENT_ERROR`] for a root.
    pub parent_lod_error: f32,
    /// Backface cone axis, snorm8 (`x / 127`).
    pub cone_axis: [i8; 3],
    /// Backface cone cutoff, unorm8 (`x / 255`): the cluster faces away when
    /// `dot(axis, centre - eye) >= cutoff * |centre - eye| + radius`. 255 = never cull.
    pub cone_cutoff: u8,
    /// First vertex in `vertices` (and the parallel skin / source arrays).
    pub vertex_offset: u32,
    /// Byte offset of the triangle bytes in `triangles`; a multiple of 4.
    pub triangle_offset: u32,
    /// 1..=256.
    pub vertex_count: u16,
    /// 1..=128.
    pub triangle_count: u16,
    /// First entry in `bone_ids` (skinned meshes).
    pub bone_offset: u32,
    /// 0..=16.
    pub bone_count: u16,
    pub flags: u16,
    /// First entry in `morph_entries`.
    pub morph_offset: u32,
    pub morph_count: u32,
    /// Material / sub-mesh slot the cluster draws with.
    pub section: u32,
    /// DAG level (0 = finest); for debug views and statistics.
    pub level: u32,
}

/// Quantised vertex. 28 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ClusterVertex {
    /// `min + q * step` on the mesh grid.
    pub pos: [u16; 3],
    /// Bit 0: tangent handedness is negative.
    pub flags: u16,
    /// Octahedral snorm16.
    pub normal: [i16; 2],
    pub tangent: [i16; 2],
    pub uv: [f32; 2],
    /// Vertex colour, RGBA8 (`round(clamp(c) * 255)`; 255 x4 when the mesh has no colours).
    pub color: [u8; 4],
}

/// Skin data, parallel to `vertices` when [`FLAG_SKINNED`]. 8 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ClusterVertexSkin {
    /// Indices into the cluster's bone list (`bone_ids[bone_offset + i]`).
    pub bones: [u8; 4],
    /// Unorm8, summing to 255 (all zero for an unweighted vertex).
    pub weights: [u8; 4],
}

/// Sparse blendshape delta of one cluster vertex. 16 bytes; deltas are f16.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct MorphEntry {
    pub target: u16,
    /// Cluster-local vertex.
    pub vertex: u16,
    pub d_pos: [u16; 3],
    pub d_normal: [u16; 3],
}

/// One slot of a [`BvhNode`]. 48 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct BvhChild {
    /// Sphere around everything below, for culling.
    pub bounds: [f32; 4],
    /// Sphere enclosing the parent LOD spheres of everything below.
    pub lod_sphere: [f32; 4],
    /// Largest `parent_lod_error` below. When this is already accurate enough
    /// on `lod_sphere`, no cluster below can be selected: skip the subtree.
    pub max_parent_error: f32,
    /// [`BVH_EMPTY`], a node index, or [`BVH_LEAF`] | first cluster of a run.
    pub child: u32,
    /// Clusters in the run (leaf only).
    pub count: u32,
    pub _pad: u32,
}

impl BvhChild {
    pub const EMPTY: Self = Self { bounds: [0.0; 4], lod_sphere: [0.0; 4], max_parent_error: 0.0, child: BVH_EMPTY, count: 0, _pad: 0 };
}

/// Four-wide BVH node. 192 bytes. Node 0 is the root; children have larger indices.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct BvhNode {
    pub children: [BvhChild; 4],
}

/// CPU-side DAG links of a cluster (not for the GPU). 12 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ClusterLink {
    /// Group that produced this cluster, [`NO_LINK`] for the finest level.
    pub group: u32,
    /// Group that consumes it, [`NO_LINK`] for a root.
    pub parent_group: u32,
    pub level: u32,
}

/// One simplification step. CPU side. 24 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GroupRecord {
    /// Range in `group_children`.
    pub first_child: u32,
    pub child_count: u32,
    /// Range of the produced clusters (contiguous).
    pub first_output: u32,
    pub output_count: u32,
    pub level: u32,
    /// Error of the outputs.
    pub error: f32,
}

/// Quantisation grid, content counts and build measurements.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DagHeader {
    pub mesh_min: [f32; 3],
    pub quant_step: f32,
    pub source_vertices: u32,
    /// Triangles of the finest level (source triangles minus degenerate ones).
    pub source_triangles: u32,
    pub dropped_triangles: u32,
    pub morph_targets: u32,
    pub flags: u32,
    pub levels: u32,
    pub leaf_clusters: u32,
    /// Mean finest-cluster triangle count over 128.
    pub fill: f32,
    /// Share of finest-level triangle edges that border another cluster.
    pub boundary_edge_ratio: f32,
    /// Mean cluster sphere radius over the radius of its box's circumsphere.
    pub bounds_tightness: f32,
}

/// What to build.
#[derive(Clone, Debug, PartialEq)]
pub struct DagOptions {
    /// Triangles per finest cluster the partitioner aims for (<= 128).
    pub cluster_triangles: usize,
    /// Clusters per group the grouper aims for (8..=32).
    pub group_clusters: usize,
    /// Extra cost multiplier (1 + bias) on removing a vertex a blendshape
    /// moves, so faces keep detail; 0 = off. Used only when morphs are given.
    pub morph_error_bias: f32,
    /// Penalty on collapsing across skin-weight changes (skinned meshes).
    pub skin_penalty: f32,
    /// Material / sub-mesh slot written into every cluster.
    pub section: u32,
    /// Worker threads (0 = half the cores, leaving the frame loop room). Not
    /// part of the cache key; the result does not depend on it.
    pub threads: usize,
}

impl Default for DagOptions {
    fn default() -> Self {
        Self { cluster_triangles: 124, group_clusters: 16, morph_error_bias: 2.0, skin_penalty: 0.5, section: 0, threads: 0 }
    }
}

impl DagOptions {
    /// The settings that change the output, for hashing into a cache key.
    pub fn fingerprint(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&(self.cluster_triangles as u32).to_le_bytes());
        b.extend_from_slice(&(self.group_clusters as u32).to_le_bytes());
        b.extend_from_slice(&self.morph_error_bias.to_bits().to_le_bytes());
        b.extend_from_slice(&self.skin_penalty.to_bits().to_le_bytes());
        b.extend_from_slice(&self.section.to_le_bytes());
        b
    }
}

/// A built DAG.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClusterDag {
    pub header: DagHeader,
    pub clusters: Vec<GpuCluster>,
    pub links: Vec<ClusterLink>,
    pub groups: Vec<GroupRecord>,
    pub group_children: Vec<u32>,
    pub nodes: Vec<BvhNode>,
    pub vertices: Vec<ClusterVertex>,
    pub vertex_skin: Vec<ClusterVertexSkin>,
    pub triangles: Vec<u8>,
    pub bone_ids: Vec<u16>,
    pub morph_entries: Vec<MorphEntry>,
    /// Source vertex of every cluster vertex (CPU only: discrete cuts, blendshape remap).
    pub vertex_source: Vec<u32>,
}

/// Sizes and measurements of a DAG, for reports and budgets.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DagStats {
    pub clusters: usize,
    pub leaf_clusters: usize,
    pub levels: u32,
    pub groups: usize,
    pub leaf_triangles: usize,
    pub total_triangles: usize,
    pub total_vertices: usize,
    /// Everything serialised.
    pub bytes: usize,
    /// What the GPU stage uploads (clusters, vertices, skin, triangles, bones, morphs, BVH).
    pub gpu_bytes: usize,
    pub bytes_per_leaf_triangle: f32,
    pub gpu_bytes_per_leaf_triangle: f32,
    pub fill: f32,
    pub boundary_edge_ratio: f32,
    pub bounds_tightness: f32,
}

/// A discrete LOD cut of the DAG.
#[derive(Clone, Debug, PartialEq)]
pub struct LodCut {
    /// Triangle share that was asked for.
    pub share: f32,
    /// Uniform error threshold the cut was taken at.
    pub error: f32,
    pub clusters: Vec<u32>,
    pub triangles: usize,
}

// ---------------------------------------------------------------- encodings

/// Octahedral encoding of a unit vector into snorm16 pairs.
pub fn oct_encode(n: [f32; 3]) -> [i16; 2] {
    let l1 = n[0].abs() + n[1].abs() + n[2].abs();
    if l1.partial_cmp(&1e-20) != Some(std::cmp::Ordering::Greater) || !l1.is_finite() {
        return oct_encode([0.0, 1.0, 0.0]);
    }
    let (mut x, mut y) = (n[0] / l1, n[1] / l1);
    if n[2] < 0.0 {
        let (ox, oy) = (x, y);
        x = (1.0 - oy.abs()) * if ox >= 0.0 { 1.0 } else { -1.0 };
        y = (1.0 - ox.abs()) * if oy >= 0.0 { 1.0 } else { -1.0 };
    }
    [(x * 32767.0).round().clamp(-32767.0, 32767.0) as i16, (y * 32767.0).round().clamp(-32767.0, 32767.0) as i16]
}

/// Inverse of [`oct_encode`] (unit length).
pub fn oct_decode(e: [i16; 2]) -> [f32; 3] {
    let (mut x, mut y) = (e[0] as f32 / 32767.0, e[1] as f32 / 32767.0);
    let z = 1.0 - x.abs() - y.abs();
    if z < 0.0 {
        let (ox, oy) = (x, y);
        x = (1.0 - oy.abs()) * if ox >= 0.0 { 1.0 } else { -1.0 };
        y = (1.0 - ox.abs()) * if oy >= 0.0 { 1.0 } else { -1.0 };
    }
    let l = (x * x + y * y + z * z).sqrt().max(1e-20);
    [x / l, y / l, z / l]
}

impl DagHeader {
    /// Position of a quantised point.
    pub fn decode_position(&self, q: [u16; 3]) -> [f32; 3] {
        std::array::from_fn(|k| self.mesh_min[k] + q[k] as f32 * self.quant_step)
    }
}

fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

impl GpuCluster {
    pub fn is_root(&self) -> bool {
        self.parent_lod_error == ROOT_PARENT_ERROR
    }
}

impl ClusterDag {
    /// The DAG as the renderer's cluster store takes it (GEO-4). The GPU
    /// records are the same bytes as [`GpuCluster`], [`ClusterVertex`] and
    /// [`BvhNode`] (checked by a test), so this is a copy of the four arrays.
    /// Skinned and blendshape DAGs carry their skin, bone-list and morph
    /// streams (drawn through the renderer's skin stage); DAGs with colour /
    /// lightmap streams are flagged and refused.
    #[cfg(feature = "client")]
    pub fn to_gpu_upload(&self) -> zestengine_render::ClusterMeshUpload {
        let skinned = self.header.flags & FLAG_SKINNED != 0 || self.header.morph_targets > 0;
        zestengine_render::ClusterMeshUpload {
            clusters: bytemuck::cast_slice(&self.clusters).to_vec(),
            vertices: bytemuck::cast_slice(&self.vertices).to_vec(),
            triangles: self.triangles.clone(),
            nodes: bytemuck::cast_slice(&self.nodes).to_vec(),
            mesh_min: self.header.mesh_min,
            step: self.header.quant_step,
            skinned,
            skin: if self.header.flags & FLAG_SKINNED != 0 { bytemuck::cast_slice(&self.vertex_skin).to_vec() } else { Vec::new() },
            bones: if skinned { self.bone_ids.iter().map(|&b| u32::from(b)).collect() } else { Vec::new() },
            morphs: if skinned { bytemuck::cast_slice(&self.morph_entries).to_vec() } else { Vec::new() },
            morph_targets: self.header.morph_targets,
            stream: None,
            morph_reach: Vec::new(),
        }
    }

    pub fn triangle_count_of(&self, clusters: &[u32]) -> usize {
        clusters.iter().map(|&c| self.clusters[c as usize].triangle_count as usize).sum()
    }

    /// Decoded positions of a cluster's vertices.
    pub fn cluster_positions(&self, c: u32) -> Vec<[f32; 3]> {
        let cl = &self.clusters[c as usize];
        self.vertices[cl.vertex_offset as usize..][..cl.vertex_count as usize].iter().map(|v| self.header.decode_position(v.pos)).collect()
    }

    /// A cluster's triangles as cluster-local vertex indices.
    pub fn cluster_triangles(&self, c: u32) -> impl Iterator<Item = [u8; 3]> + '_ {
        let cl = &self.clusters[c as usize];
        self.triangles[cl.triangle_offset as usize..][..cl.triangle_count as usize * 3].as_chunks::<3>().0.iter().map(|t| [t[0], t[1], t[2]])
    }

    /// A cluster's triangles as source vertex indices.
    pub fn cluster_source_triangles(&self, c: u32) -> impl Iterator<Item = [u32; 3]> + '_ {
        let off = self.clusters[c as usize].vertex_offset as usize;
        self.cluster_triangles(c).map(move |t| t.map(|i| self.vertex_source[off + i as usize]))
    }

    pub fn leaf_clusters(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.clusters.len() as u32).filter(|&c| self.links[c as usize].group == NO_LINK)
    }

    pub fn root_clusters(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.clusters.len() as u32).filter(|&c| self.clusters[c as usize].is_root())
    }

    /// The cut under a LOD predicate `ok(error, sphere)` ("accurate enough"),
    /// which must be monotone: true for a (larger error, larger sphere) pair
    /// implies true for every smaller one. A cluster is in the cut when it is
    /// accurate enough and its parent is not. Reference implementation: the
    /// GPU traversal must select exactly this set.
    pub fn select_linear(&self, ok: &dyn Fn(f32, [f32; 4]) -> bool) -> Vec<u32> {
        (0..self.clusters.len() as u32).filter(|&c| self.selected(c, ok)).collect()
    }

    fn selected(&self, c: u32, ok: &dyn Fn(f32, [f32; 4]) -> bool) -> bool {
        let cl = &self.clusters[c as usize];
        ok(cl.lod_error, cl.lod_sphere) && !ok(cl.parent_lod_error, cl.parent_lod_sphere)
    }

    /// [`select_linear`](Self::select_linear) through the BVH, pruning subtrees
    /// whose parents are all accurate enough. Same result, sorted.
    pub fn select_bvh(&self, ok: &dyn Fn(f32, [f32; 4]) -> bool) -> Vec<u32> {
        let mut out = Vec::new();
        let mut stack = vec![0u32];
        while let Some(n) = stack.pop() {
            for ch in &self.nodes[n as usize].children {
                if ch.child == BVH_EMPTY || ok(ch.max_parent_error, ch.lod_sphere) {
                    continue;
                }
                if ch.child & BVH_LEAF != 0 {
                    let first = ch.child & !BVH_LEAF;
                    out.extend((first..first + ch.count).filter(|&c| self.selected(c, ok)));
                } else {
                    stack.push(ch.child);
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// Cut at one object-space error threshold for the whole mesh.
    pub fn select_by_error(&self, threshold: f32) -> Vec<u32> {
        self.select_linear(&|e, _| e <= threshold)
    }

    /// The finest cut that fits `budget` triangles, by uniform error
    /// (the coarsest cut when even the roots exceed it).
    pub fn cut_for_budget(&self, budget: usize) -> (f32, Vec<u32>) {
        let mut errors: Vec<f32> = self.clusters.iter().map(|c| c.lod_error).collect();
        errors.push(0.0);
        errors.sort_unstable_by(f32::total_cmp);
        errors.dedup();
        let tris = |t: f32| self.triangle_count_of(&self.select_by_error(t));
        // Triangle count only falls as the threshold rises: binary search the first that fits.
        let last = errors.len() - 1;
        let hi = if tris(errors[0]) <= budget {
            0
        } else if tris(errors[last]) > budget {
            last
        } else {
            let (mut lo, mut hi) = (0, last);
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if tris(errors[mid]) <= budget {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            hi
        };
        let t = errors[hi];
        (t, self.select_by_error(t))
    }

    /// Discrete cuts for triangle `shares` of the finest level (the fallback
    /// LODs of tiers without the cluster path, and the existing auto-LOD chain).
    pub fn discrete_lods(&self, shares: &[f32]) -> Vec<LodCut> {
        shares
            .iter()
            .map(|&share| {
                let budget = (self.header.source_triangles as f32 * share).round() as usize;
                let (error, clusters) = self.cut_for_budget(budget.max(1));
                let triangles = self.triangle_count_of(&clusters);
                LodCut { share, error, clusters, triangles }
            })
            .collect()
    }

    /// A cut as an ordinary mesh over `source`'s vertices (positions and
    /// attributes exactly the source's).
    pub fn cut_mesh(&self, source: &MeshData, cut: &[u32]) -> MeshData {
        self.cut_mesh_with_provenance(source, cut).0
    }

    /// [`cut_mesh`](Self::cut_mesh) plus, per output vertex, the source vertex
    /// it is, so blendshape deltas follow with
    /// [`remap_morph_deltas`](crate::simplify::remap_morph_deltas).
    pub fn cut_mesh_with_provenance(&self, source: &MeshData, cut: &[u32]) -> (MeshData, Vec<u32>) {
        let mut remap = std::collections::HashMap::<u32, u32>::new();
        let mut vertices: Vec<Vertex> = Vec::new();
        let mut provenance: Vec<u32> = Vec::new();
        let mut indices = Vec::with_capacity(self.triangle_count_of(cut) * 3);
        for &c in cut {
            for t in self.cluster_source_triangles(c) {
                for v in t {
                    let next = vertices.len() as u32;
                    let id = *remap.entry(v).or_insert(next);
                    if id == next {
                        vertices.push(source.vertices[v as usize]);
                        provenance.push(v);
                    }
                    indices.push(id);
                }
            }
        }
        (MeshData { vertices, indices, has_lightmap_uv: source.has_lightmap_uv }, provenance)
    }

    /// Length of [`to_bytes`](Self::to_bytes) without building it.
    pub fn byte_len(&self) -> usize {
        // magic + format + grid (4 floats) + 7 counts + 3 measurements, then 11 length-prefixed arrays.
        4 + 4 + 16 + 28 + 12 + 11 * 8 + self.section_bytes().iter().sum::<usize>()
    }

    pub fn stats(&self) -> DagStats {
        let sizes = self.section_bytes();
        let gpu = sizes.iter().take(7).sum::<usize>();
        let leaf = self.header.source_triangles.max(1) as f32;
        let total_bytes = self.byte_len();
        DagStats {
            clusters: self.clusters.len(),
            leaf_clusters: self.header.leaf_clusters as usize,
            levels: self.header.levels,
            groups: self.groups.len(),
            leaf_triangles: self.header.source_triangles as usize,
            total_triangles: self.clusters.iter().map(|c| c.triangle_count as usize).sum(),
            total_vertices: self.vertices.len(),
            bytes: total_bytes,
            gpu_bytes: gpu,
            bytes_per_leaf_triangle: total_bytes as f32 / leaf,
            gpu_bytes_per_leaf_triangle: gpu as f32 / leaf,
            fill: self.header.fill,
            boundary_edge_ratio: self.header.boundary_edge_ratio,
            bounds_tightness: self.header.bounds_tightness,
        }
    }

    /// Byte sizes of the arrays: clusters, vertices, skin, triangles,
    /// bones, morphs, BVH (the GPU set), then links, groups, group children, sources.
    fn section_bytes(&self) -> [usize; 11] {
        use std::mem::size_of_val;
        [
            size_of_val(&self.clusters[..]),
            size_of_val(&self.vertices[..]),
            size_of_val(&self.vertex_skin[..]),
            self.triangles.len(),
            size_of_val(&self.bone_ids[..]),
            size_of_val(&self.morph_entries[..]),
            size_of_val(&self.nodes[..]),
            size_of_val(&self.links[..]),
            size_of_val(&self.groups[..]),
            size_of_val(&self.group_children[..]),
            size_of_val(&self.vertex_source[..]),
        ]
    }
}

// ---------------------------------------------------------------- bytes

const DAG_MAGIC: &[u8; 4] = b"ZDG1";
/// Layout version of [`ClusterDag::to_bytes`].
pub const DAG_FORMAT: u32 = 2;

struct Writer(Vec<u8>);

impl Writer {
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn pod<T: Pod>(&mut self, v: &[T]) {
        self.0.extend_from_slice(&(v.len() as u64).to_le_bytes());
        self.0.extend_from_slice(bytemuck::cast_slice(v));
    }
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        if self.0.len() < n {
            return Err("truncated cluster data".into());
        }
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Ok(a)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    /// A length-prefixed array; the count is checked against what is left before allocating.
    fn pod<T: Pod>(&mut self) -> Result<Vec<T>, String> {
        let n = u64::from_le_bytes(self.take(8)?.try_into().unwrap());
        let size = std::mem::size_of::<T>() as u64;
        if n.checked_mul(size).is_none_or(|b| b > self.0.len() as u64) {
            return Err("array longer than the data".into());
        }
        Ok(bytemuck::pod_collect_to_vec(self.take((n * size) as usize)?))
    }
}

impl ClusterDag {
    pub fn to_bytes(&self) -> Vec<u8> {
        let h = &self.header;
        let mut w = Writer(Vec::with_capacity(self.section_bytes().iter().sum::<usize>() + 128));
        w.0.extend_from_slice(DAG_MAGIC);
        w.u32(DAG_FORMAT);
        for v in h.mesh_min {
            w.f32(v);
        }
        w.f32(h.quant_step);
        for v in [h.source_vertices, h.source_triangles, h.dropped_triangles, h.morph_targets, h.flags, h.levels, h.leaf_clusters] {
            w.u32(v);
        }
        for v in [h.fill, h.boundary_edge_ratio, h.bounds_tightness] {
            w.f32(v);
        }
        w.pod(&self.clusters);
        w.pod(&self.links);
        w.pod(&self.groups);
        w.pod(&self.group_children);
        w.pod(&self.nodes);
        w.pod(&self.vertices);
        w.pod(&self.vertex_skin);
        w.pod(&self.triangles);
        w.pod(&self.bone_ids);
        w.pod(&self.morph_entries);
        w.pod(&self.vertex_source);
        w.0
    }

    /// Parse (bounds-checked, no structural validation: see [`validate`](Self::validate)).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut r = Reader(bytes);
        if r.take(4)? != DAG_MAGIC {
            return Err("not cluster DAG data".into());
        }
        if r.u32()? != DAG_FORMAT {
            return Err("cluster DAG format version differs".into());
        }
        let mut h = DagHeader::default();
        for k in 0..3 {
            h.mesh_min[k] = r.f32()?;
        }
        h.quant_step = r.f32()?;
        h.source_vertices = r.u32()?;
        h.source_triangles = r.u32()?;
        h.dropped_triangles = r.u32()?;
        h.morph_targets = r.u32()?;
        h.flags = r.u32()?;
        h.levels = r.u32()?;
        h.leaf_clusters = r.u32()?;
        h.fill = r.f32()?;
        h.boundary_edge_ratio = r.f32()?;
        h.bounds_tightness = r.f32()?;
        let dag = Self {
            header: h,
            clusters: r.pod()?,
            links: r.pod()?,
            groups: r.pod()?,
            group_children: r.pod()?,
            nodes: r.pod()?,
            vertices: r.pod()?,
            vertex_skin: r.pod()?,
            triangles: r.pod()?,
            bone_ids: r.pod()?,
            morph_entries: r.pod()?,
            vertex_source: r.pod()?,
        };
        if !r.0.is_empty() {
            return Err("trailing bytes after cluster data".into());
        }
        Ok(dag)
    }
}

// ------------------------------------------------------------ validation

fn encloses(outer: [f32; 4], inner: [f32; 4], slack: f32) -> bool {
    dist3([outer[0], outer[1], outer[2]], [inner[0], inner[1], inner[2]]) + inner[3] <= outer[3] * (1.0 + 1e-4) + slack
}

fn sphere_ok(s: [f32; 4]) -> bool {
    s.iter().all(|v| v.is_finite()) && s[3] >= 0.0
}

impl ClusterDag {
    /// Check everything a renderer relies on. A bad DAG can hang the GPU
    /// (out-of-range ranges and indices, cycles, runaway traversal), so data
    /// from outside (a cache file, a content package) must pass this before
    /// it is uploaded. `Err` names the first problem.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_with(true)
    }

    /// [`validate`](Self::validate) for a paged mesh's skeleton (clusters, links,
    /// groups, BVH, no vertex or triangle payload): everything that does not read
    /// the payload. The pages are checked one by one as they are read
    /// (`cluster_pages::validate_payload`).
    pub fn validate_skeleton(&self) -> Result<(), String> {
        self.validate_with(false)
    }

    #[allow(clippy::neg_cmp_op_on_partial_ord)] // check! expands `if !($c)` over f32 fields
    fn validate_with(&self, payload: bool) -> Result<(), String> {
        macro_rules! check {
            ($c:expr, $($m:tt)*) => {
                if !($c) {
                    return Err(format!($($m)*));
                }
            };
        }
        let h = &self.header;
        let nc = self.clusters.len();
        check!(nc > 0 && nc <= MAX_CLUSTERS, "cluster count {nc} out of range");
        check!(h.mesh_min.iter().all(|v| v.is_finite()) && h.quant_step.is_finite() && h.quant_step > 0.0, "bad quantisation grid");
        check!(self.links.len() == nc, "link count differs from cluster count");
        let nv = self.vertices.len();
        if payload {
            check!(self.vertex_source.len() == nv, "vertex source count differs");
            check!(self.vertex_skin.len() == if h.flags & FLAG_SKINNED != 0 { nv } else { 0 }, "skin stream length wrong");
            check!(self.triangles.len().is_multiple_of(4), "triangle bytes not padded to 4");
            check!(self.vertex_source.iter().all(|&s| s < h.source_vertices), "vertex source out of range");
        } else {
            check!(nv == 0 && self.triangles.is_empty() && self.vertex_source.is_empty(), "a skeleton carries payload");
        }
        check!(!self.groups.is_empty() || nc == h.leaf_clusters as usize, "missing groups");
        let slack = h.quant_step * 4.0;
        let mut leaf_tris = 0u64;
        let mut child_refs = vec![0u32; nc];
        let mut produced = vec![false; nc];
        for (gi, g) in self.groups.iter().enumerate() {
            check!(g.child_count > 0 && g.output_count > 0, "group {gi} is empty");
            let (cf, cn) = (g.first_child as usize, g.child_count as usize);
            check!(cf.checked_add(cn).is_some_and(|e| e <= self.group_children.len()), "group {gi} children out of range");
            let (of, on) = (g.first_output as usize, g.output_count as usize);
            check!(of.checked_add(on).is_some_and(|e| e <= nc), "group {gi} outputs out of range");
            check!(g.error.is_finite() && g.error >= 0.0, "group {gi} error not finite");
            for (o, p) in produced.iter_mut().enumerate().skip(of).take(on) {
                check!(self.links[o].group == gi as u32, "cluster {o} is not linked to the group that lists it");
                check!(!*p, "cluster {o} produced twice");
                *p = true;
                check!(self.links[o].level == g.level, "cluster {o} level differs from its group");
                check!(self.clusters[o].lod_error == g.error, "cluster {o} error differs from its group's");
            }
            for &c in &self.group_children[cf..cf + cn] {
                check!((c as usize) < nc, "group {gi} child out of range");
                check!(self.links[c as usize].parent_group == gi as u32, "child {c} is not linked back to group {gi}");
                check!(self.links[c as usize].level < g.level, "group {gi} is not above its children (cycle?)");
                child_refs[c as usize] += 1;
            }
        }
        for (i, (c, l)) in self.clusters.iter().zip(&self.links).enumerate() {
            check!((1..=CLUSTER_MAX_TRIS).contains(&(c.triangle_count as usize)), "cluster {i}: triangle count {}", c.triangle_count);
            check!((1..=CLUSTER_MAX_VERTS).contains(&(c.vertex_count as usize)), "cluster {i}: vertex count {}", c.vertex_count);
            check!(c.bone_count as usize <= CLUSTER_MAX_BONES, "cluster {i}: {} bones", c.bone_count);
            check!(c.triangle_offset % 4 == 0, "cluster {i}: unaligned triangle offset");
            if payload {
                check!((c.vertex_offset as u64) + c.vertex_count as u64 <= nv as u64, "cluster {i}: vertices out of range");
                let tb = c.triangle_count as u64 * 3;
                check!(c.triangle_offset as u64 + tb <= self.triangles.len() as u64, "cluster {i}: triangles out of range");
                let tris = &self.triangles[c.triangle_offset as usize..][..tb as usize];
                check!(tris.iter().all(|&t| (t as u16) < c.vertex_count), "cluster {i}: triangle index beyond its vertices");
            }
            // A skeleton (paged mesh) carries no bone lists or blendshape entries: they live in the pages.
            if payload {
                check!((c.bone_offset as u64) + c.bone_count as u64 <= self.bone_ids.len() as u64, "cluster {i}: bones out of range");
                check!((c.morph_offset as u64) + c.morph_count as u64 <= self.morph_entries.len() as u64, "cluster {i}: morphs out of range");
                for m in &self.morph_entries[c.morph_offset as usize..][..c.morph_count as usize] {
                    check!((m.target as u32) < h.morph_targets && m.vertex < c.vertex_count, "cluster {i}: morph entry out of range");
                }
            } else {
                check!(h.morph_targets > 0 || c.morph_count == 0, "cluster {i}: blendshape entries without targets");
                check!(c.bone_offset.checked_add(u32::from(c.bone_count)).is_some() && c.morph_offset.checked_add(c.morph_count).is_some(), "cluster {i}: skin ranges overflow");
            }
            let range = c.vertex_offset as usize..c.vertex_offset as usize + c.vertex_count as usize;
            if !payload {
                check!(h.flags & FLAG_SKINNED != 0 || c.bone_count == 0, "cluster {i}: bones on a static mesh");
            } else if h.flags & FLAG_SKINNED != 0 {
                for s in &self.vertex_skin[range.clone()] {
                    for k in 0..4 {
                        check!(s.weights[k] == 0 || (s.bones[k] as u16) < c.bone_count, "cluster {i}: skin index beyond its bone list");
                    }
                }
            } else {
                check!(c.bone_count == 0, "cluster {i}: bones on a static mesh");
            }
            check!(sphere_ok(c.bounds) && sphere_ok(c.lod_sphere) && sphere_ok(c.parent_lod_sphere), "cluster {i}: sphere not finite");
            check!(c.lod_error.is_finite() && c.lod_error >= 0.0, "cluster {i}: error {} not finite", c.lod_error);
            check!(c.parent_lod_error >= c.lod_error && !c.parent_lod_error.is_nan(), "cluster {i}: error shrinks going up");
            check!(c.parent_lod_error == ROOT_PARENT_ERROR || c.parent_lod_error.is_finite(), "cluster {i}: parent error not finite");
            check!(c.is_root() == (l.parent_group == NO_LINK), "cluster {i}: root flag and parent group disagree");
            check!(l.parent_group == NO_LINK || (l.parent_group as usize) < self.groups.len(), "cluster {i}: parent group out of range");
            check!(child_refs[i] == u32::from(l.parent_group != NO_LINK), "cluster {i} is not consumed by exactly one group");
            if l.parent_group != NO_LINK {
                let pg = &self.groups[l.parent_group as usize];
                check!(c.parent_lod_error == pg.error, "cluster {i}: parent error differs from its group's");
                check!(encloses(c.parent_lod_sphere, c.lod_sphere, slack), "cluster {i}: parent sphere does not enclose it");
            }
            if l.group == NO_LINK {
                check!(l.level == 0 && c.lod_error == 0.0, "cluster {i}: finest cluster with level or error");
                leaf_tris += c.triangle_count as u64;
            } else {
                check!((l.group as usize) < self.groups.len() && produced[i], "cluster {i}: producer group invalid");
                check!(l.level > 0, "cluster {i}: produced cluster at level 0");
            }
            if payload {
                for v in &self.vertices[range] {
                    let p = h.decode_position(v.pos);
                    check!(dist3(p, [c.bounds[0], c.bounds[1], c.bounds[2]]) <= c.bounds[3] * (1.0 + 1e-4) + slack, "cluster {i}: vertex outside its bounds");
                }
            }
        }
        check!(self.groups.iter().map(|g| g.output_count as usize).sum::<usize>() == self.links.iter().filter(|l| l.group != NO_LINK).count(), "clusters not accounted for by groups");
        check!(self.group_children.len() == child_refs.iter().filter(|&&r| r > 0).count(), "group children list has strays");
        check!(leaf_tris == h.source_triangles as u64, "finest level has {leaf_tris} triangles, header says {}", h.source_triangles);
        self.validate_bvh(slack)
    }

    #[allow(clippy::neg_cmp_op_on_partial_ord)] // check! expands `if !($c)` over f32 fields
    fn validate_bvh(&self, slack: f32) -> Result<(), String> {
        macro_rules! check {
            ($c:expr, $($m:tt)*) => {
                if !($c) {
                    return Err(format!($($m)*));
                }
            };
        }
        let nc = self.clusters.len();
        let nn = self.nodes.len();
        check!(nn > 0 && nn <= nc * 2 + 4, "BVH node count {nn} out of range");
        let mut covered = vec![false; nc];
        let mut seen = vec![false; nn];
        seen[0] = true;
        let mut stack = vec![(0usize, 1usize)];
        let mut visited = 0usize;
        while let Some((n, depth)) = stack.pop() {
            visited += 1;
            check!(depth <= BVH_MAX_DEPTH, "BVH deeper than {BVH_MAX_DEPTH}");
            for ch in &self.nodes[n].children {
                if ch.child == BVH_EMPTY {
                    check!(ch.count == 0, "BVH: empty slot with a count");
                    continue;
                }
                check!(sphere_ok(ch.bounds) && sphere_ok(ch.lod_sphere), "BVH: sphere not finite");
                check!(!ch.max_parent_error.is_nan() && ch.max_parent_error >= 0.0, "BVH: bad error");
                if ch.child & BVH_LEAF != 0 {
                    let first = (ch.child & !BVH_LEAF) as usize;
                    let count = ch.count as usize;
                    check!(count >= 1 && first.checked_add(count).is_some_and(|e| e <= nc), "BVH: cluster run out of range");
                    for (c, cov) in covered.iter_mut().enumerate().skip(first).take(count) {
                        check!(!*cov, "BVH: cluster {c} listed twice");
                        *cov = true;
                        let cl = &self.clusters[c];
                        check!(encloses(ch.bounds, cl.bounds, slack), "BVH: leaf bounds do not enclose cluster {c}");
                        check!(cl.parent_lod_error <= ch.max_parent_error, "BVH: max parent error below cluster {c}'s");
                        let want = if cl.is_root() { cl.lod_sphere } else { cl.parent_lod_sphere };
                        check!(encloses(ch.lod_sphere, want, slack), "BVH: LOD sphere does not enclose cluster {c}'s parent");
                    }
                } else {
                    let child = ch.child as usize;
                    check!(child > n && child < nn, "BVH: child {child} of node {n} out of order");
                    check!(!seen[child], "BVH: node {child} reached twice");
                    seen[child] = true;
                    let sub = &self.nodes[child];
                    for g in sub.children.iter().filter(|g| g.child != BVH_EMPTY) {
                        check!(encloses(ch.bounds, g.bounds, slack), "BVH: node {child} bounds do not enclose its children");
                        check!(g.max_parent_error <= ch.max_parent_error, "BVH: node {child} error exceeds its slot's");
                        check!(encloses(ch.lod_sphere, g.lod_sphere, slack), "BVH: node {child} LOD sphere not enclosed");
                    }
                    stack.push((child, depth + 1));
                }
            }
        }
        check!(visited == nn, "BVH has unreachable nodes");
        check!(covered.iter().all(|&c| c), "BVH does not reach every cluster");
        Ok(())
    }

    /// [`validate`](Self::validate) plus the checks that need the source mesh:
    /// vertex counts match, and every finest triangle is covered exactly once.
    pub fn validate_against(&self, mesh: &MeshData) -> Result<(), String> {
        self.validate()?;
        if self.header.source_vertices as usize != mesh.vertices.len() {
            return Err("DAG was built for a mesh with another vertex count".into());
        }
        let mut want: std::collections::HashMap<[u32; 3], u32> = std::collections::HashMap::new();
        for t in mesh.indices.as_chunks::<3>().0 {
            *want.entry(canonical(*t)).or_default() += 1;
        }
        for c in self.leaf_clusters() {
            for t in self.cluster_source_triangles(c) {
                match want.get_mut(&canonical(t)) {
                    Some(n) if *n > 0 => *n -= 1,
                    _ => return Err("finest level has a triangle the source does not (or twice)".into()),
                }
            }
        }
        let missing: u32 = want.values().sum();
        let dropped = self.header.dropped_triangles;
        if missing != dropped {
            return Err(format!("{missing} source triangles are not covered ({dropped} were dropped as degenerate)"));
        }
        Ok(())
    }
}

/// A triangle's vertices rotated to start at the smallest (orientation kept).
pub(crate) fn canonical(t: [u32; 3]) -> [u32; 3] {
    let k = (0..3).min_by_key(|&i| t[i]).unwrap();
    [t[k], t[(k + 1) % 3], t[(k + 2) % 3]]
}
