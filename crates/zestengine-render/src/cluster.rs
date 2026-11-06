//! GPU cluster culling and cluster draws (GEO-4).
//!
//! Meshes with a cooked cluster DAG (`zestengine-assets::cluster_dag`) can be
//! drawn from it: a pooled GPU store holds every installed DAG, a compute pass
//! (`cluster_cull.comp`) walks each instance's BVH, cuts the DAG at a pixel
//! error target, culls the selected clusters (frustum, backface cone, two-phase
//! Hi-Z) and writes per-material-batch visible lists plus indirect draw
//! arguments, and the clustered variant of `standard.vert` draws them by
//! vertex pulling. Forward shading, MSAA and every material stay as they are.
//!
//! Off by default (`ClusterMode`, `ZESTENGINE_CLUSTERS=1/0`). The path is
//! skipped for: orthographic cameras, VR (multiview; `cluster_vr.rs` has the
//! centre-eye / union-frustum math, the XR frame loop is not hooked up),
//! lightmapped draws, skinned meshes without a queued pose (skinned and
//! blendshape meshes otherwise draw through `cluster_skin.rs`), meshes without an installed DAG, and whenever the
//! store, the pipelines or a frame's queues fail - those draw the original
//! mesh through the ordinary (GPU-culled or CPU) path, with one logged warning.
//! Shadow passes, ray tracing and the discrete LOD chain keep using the
//! original mesh buffers.
//!
//! Design decisions (phase6b §7.2, §7.3):
//! - Traversal: level by level, one indirect dispatch per BVH level with an
//!   atomic work queue (ping-pong between two bounded queues), not persistent
//!   threads. Leaf runs (at most 8 clusters) are tested in place.
//! - Raster: one `vkCmdDrawIndirect` per material batch, 384 vertices (128
//!   triangles * 3) per cluster instance, `instanceCount` = visible clusters,
//!   vertices of triangle indices >= `triangle_count` collapse. Chosen over index
//!   compaction because it needs no second pass and no index-buffer growth; the
//!   wasted vertex invocations (clusters average ~100 of 128 triangles) are the
//!   price, and compaction stays the better choice for tiled mobile GPUs.
//! - Draws: batches bind different material descriptor sets, so there is no
//!   draw to merge into a multi-draw; `multiDrawIndirect` / `drawIndirectCount`
//!   are therefore not needed, and the path works without them and without
//!   buffer device address (everything is storage-buffer descriptors with 32-bit
//!   indices). The batch's slice of the visible list is found through a push
//!   constant, so `drawIndirectFirstInstance` is not needed either.

use std::collections::HashMap;

use anyhow::{Result, bail};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;
use serde::{Deserialize, Serialize};

mod stream;
pub use stream::{PageLoad, PageStager, StagedPage};

use crate::alloc::{self, Buffer};
use crate::frame::FRAMES_IN_FLIGHT;
use crate::gpu_cull::{GpuView, GrowBuffer};
use crate::cluster_stream::{StreamPolicy, StreamStats};
use crate::{
    BoundDraw, DrawCmd, DrawMode, FrameInput, MeshHandle, PushData, RenderStats, Renderer, gpu_zone,
    scene_pipeline_key,
};

pub(crate) const CULL_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cluster_cull.comp.spv"));
pub(crate) const SKIN_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cluster_skin.comp.spv"));
pub(crate) const DECODE_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cluster_decode_test.comp.spv"));

/// Vertices one cluster instance draws (128 triangles * 3); `CLUSTER_DRAW_VERTICES` in the shaders.
pub const CLUSTER_DRAW_VERTICES: u32 = 384;
pub const CLUSTER_MAX_TRIS: usize = 128;
pub const CLUSTER_MAX_VERTS: usize = 256;
/// Hard cap when reading untrusted data (matches the cooked format).
pub const MAX_CLUSTERS: usize = 1 << 24;
/// A BVH deeper than this is rejected at install (the cooked builder caps at 32).
pub const BVH_MAX_DEPTH: u32 = 32;
pub const BVH_EMPTY: u32 = u32::MAX;
pub const BVH_LEAF: u32 = 0x8000_0000;
pub const BVH_RUN_MAX: u32 = 8;
/// Bones per cluster, at most (the cooked builder's cap).
pub const CLUSTER_MAX_BONES: usize = 16;
/// Frame sets of the cluster path: one per frame in flight, then one per VR eye-render slot.
pub(crate) const CLUSTER_SETS: usize = FRAMES_IN_FLIGHT + crate::XR_SLOTS;
/// Descriptor bindings of the cluster set (shared by the cull, the decode test
/// and the vertex stage of the clustered pipelines).
/// 17 and 18: the skinned vertex pool and each visible entry's base in it (GEO-5).
const STORAGE_BINDINGS: [u32; 20] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20];
const HIZ_BINDING: u32 = 16;
/// The right eye's pyramid of a VR stereo cull (the left eye's / the camera's is `HIZ_BINDING`).
const HIZ_R_BINDING: u32 = 21;
/// Pool bindings the vertex stage reads: clusters, vertices, triangles, meshes, instances, visible list.
const VERTEX_BINDINGS: [u32; 10] = [0, 1, 2, 4, 5, 7, 11, 17, 18, 19];
/// Entries of one node queue / of the early (and late) visible list, at most.
const MAX_QUEUE_ENTRIES: usize = 1 << 22;
const MAX_VISIBLE_ENTRIES: usize = 1 << 22;
const MIN_ENTRIES: usize = 1024;
/// Counter slots read back per frame (`C_*` in the shader).
pub const COUNTER_WORDS: usize = 32;
const COUNTER_BYTES: u64 = (COUNTER_WORDS * 4) as u64;
/// Frames the cluster path stays off after a queue overflow.
const OVERFLOW_COOLDOWN: u64 = 180;
/// Frames a skinned instance stays on the ordinary skinned draw after the
/// skinned-vertex pool pushed it out (`SkinAdmission`).
const SKIN_DENY_FRAMES: u64 = 120;
/// Mesh records the cull shader can address.
const MAX_MESH_RECORDS: usize = 1 << 20;

/// Counter indices (mirror of the `C_*` constants in cluster_cull.comp; a test
/// compares them).
pub mod counter {
    pub const FLAGS: usize = 0;
    pub const QUEUE: usize = 1;
    pub const OCCLUDED: usize = 3;
    pub const INSTANCES: usize = 4;
    pub const INSTANCES_CULLED: usize = 5;
    pub const NODES: usize = 6;
    pub const TESTED: usize = 7;
    pub const LOD: usize = 8;
    pub const FRUSTUM: usize = 9;
    pub const CONE: usize = 10;
    pub const HIZ: usize = 11;
    pub const VISIBLE: usize = 12;
    pub const TRIANGLES: usize = 13;
    pub const OVERFLOW: usize = 14;
    pub const LATE_TESTED: usize = 15;
    pub const LATE_RECOVERED: usize = 16;
    pub const LATE_OCCLUDED: usize = 17;
    /// Clusters / triangles in the shadow views' cuts, and shadow instances seeded.
    pub const SHADOW_VISIBLE: usize = 18;
    pub const SHADOW_TRIANGLES: usize = 19;
    pub const SHADOW_INSTANCES: usize = 20;
    /// Streaming (GEO-6): tested clusters on a page that is not resident, and
    /// clusters drawn in place of children whose page is missing.
    pub const STREAM_SKIPPED: usize = 21;
    pub const STREAM_STANDIN: usize = 22;
    /// `counters[FLAGS]` bits.
    pub const FLAG_QUEUE: u32 = 1;
    pub const FLAG_VISIBLE: u32 = 2;
    pub const FLAG_OCCLUDED: u32 = 4;
    pub const FLAG_BAD: u32 = 8;
}

mod mode {
    pub const SEED: u32 = 0;
    pub const NODES: u32 = 1;
    pub const LATE: u32 = 2;
    pub const FINALIZE: u32 = 3;
    pub const ADVANCE: u32 = 4;
    pub const ADVANCE_OCC: u32 = 5;
    pub const STREAM: u32 = 6;
}

/// Streaming I/O buffer layout (words), mirrored in `cluster_cull.comp`:
/// `[0]` requests appended, `[1]` requests that did not fit, `STREAM_REQ_CAP`
/// (page, priority bits) pairs from `STREAM_REQ_BASE`, a priority word per page
/// from `STREAM_PRIO_BASE`, a touched flag per page from `STREAM_TOUCH_BASE`.
pub const STREAM_REQ_CAP: usize = 1024;
pub const STREAM_REQ_BASE: usize = 4;
pub const STREAM_PRIO_BASE: usize = STREAM_REQ_BASE + 2 * STREAM_REQ_CAP;
/// Pages the page table holds across all streamed meshes when nothing says
/// otherwise (96 KB pages: about 24 GB of streamed geometry).
pub const STREAM_DEFAULT_MAX_PAGES: usize = 262_144;
/// The most pages `ZESTENGINE_CLUSTER_STREAM_MAX_PAGES` may ask for (the table
/// costs 16 bytes per page, the I/O buffer 8, each frame's readback 4).
pub const STREAM_HARD_MAX_PAGES: usize = 16_777_216;

/// Pages the page table holds across all streamed meshes: the page-table
/// capacity is a runtime bound (the shader gets it in `CullPush::stream_pages`),
/// `ZESTENGINE_CLUSTER_STREAM_MAX_PAGES` (default [`STREAM_DEFAULT_MAX_PAGES`], read once).
pub fn stream_max_pages() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ZESTENGINE_CLUSTER_STREAM_MAX_PAGES")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map_or(STREAM_DEFAULT_MAX_PAGES, |n| n.clamp(1024, STREAM_HARD_MAX_PAGES))
    })
}

/// First word of the touched flags in the stream I/O buffer.
pub fn stream_touch_base(max_pages: usize) -> usize {
    STREAM_PRIO_BASE + max_pages
}

/// Size of the stream I/O buffer for a page table of `max_pages`.
fn stream_io_bytes(max_pages: usize) -> u64 {
    ((stream_touch_base(max_pages) + max_pages) * 4) as u64
}
/// Bytes of the request header and list copied back each frame.
const STREAM_REQ_BYTES: u64 = (STREAM_PRIO_BASE * 4) as u64;
/// A page table entry of a page that is not resident (`x` = `PAGE_NONE`).
const PAGE_NONE: u32 = u32::MAX;

// ---------------------------------------------------------------------------
// Settings

/// Project / engine setting: draw meshes that have a cluster DAG through the
/// GPU cluster path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClusterMode {
    #[default]
    Off,
    On,
}

impl ClusterMode {
    pub const ALL: [ClusterMode; 2] = [ClusterMode::Off, ClusterMode::On];

    pub fn label(self) -> &'static str {
        match self {
            ClusterMode::Off => "Off",
            ClusterMode::On => "On",
        }
    }
}

/// Renderer option: cluster rendering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClusterSettings {
    pub mode: ClusterMode,
    /// Error target in screen pixels (default 1).
    pub target_px: f32,
    /// `ZESTENGINE_CLUSTERS=1/0` (wins over `mode`).
    pub env_override: Option<bool>,
    /// Frame time the budget controller aims for, in ms (the display refresh /
    /// frame limiter cap); 0 = no controller. `ZESTENGINE_CLUSTER_BUDGET_MS`.
    pub target_frame_ms: f32,
    /// `target_frame_ms` came from the environment (the engine does not override it).
    pub target_from_env: bool,
    /// Triangles the visible instances may use together before the fair share
    /// coarsens the largest ones (estimated; 0 = no limit).
    /// `ZESTENGINE_CLUSTER_MAX_TRIANGLES`.
    pub max_triangles: u32,
    /// Error target of the shadow views' cuts, in shadow-map texels (the
    /// camera's is `target_px` pixels): shadows tolerate a coarser cut because
    /// the receiver bias is already a few texels. 0 = shadows keep drawing the
    /// original meshes. `ZESTENGINE_CLUSTER_SHADOW_PX`.
    pub shadow_px: f32,
    /// Geometry streaming (GEO-6): big static meshes keep their vertices in
    /// pages that are loaded on demand under a memory budget; the renderer asks
    /// for pages, the engine's streamer serves them. Off by default; only
    /// meshes installed while it is on are streamed. `ZESTENGINE_CLUSTER_STREAM=1/0`.
    pub stream: bool,
    /// Budget of resident streamed pages, in MB (the always-resident coarse
    /// root pages are outside it). `ZESTENGINE_CLUSTER_STREAM_MB`.
    pub stream_budget_mb: u32,
}

/// Default of [`ClusterSettings::stream_budget_mb`].
pub const DEFAULT_STREAM_MB: u32 = 256;

/// Default of [`ClusterSettings::shadow_px`].
pub const DEFAULT_SHADOW_PX: f32 = 2.0;

/// Default of [`ClusterSettings::max_triangles`].
pub const DEFAULT_MAX_TRIANGLES: u32 = 6_000_000;

impl Default for ClusterSettings {
    fn default() -> Self {
        Self { mode: ClusterMode::Off, target_px: 1.0, env_override: None, target_frame_ms: 0.0, target_from_env: false, max_triangles: DEFAULT_MAX_TRIANGLES, shadow_px: DEFAULT_SHADOW_PX, stream: false, stream_budget_mb: DEFAULT_STREAM_MB }
    }
}

/// The on / off override from the environment: `ZESTENGINE_ZESTLETS` (the Zestlets name) wins over the
/// older `ZESTENGINE_CLUSTERS`; `0` or an empty value is off, anything else on, unset is no override.
pub(crate) fn zestlets_env_override(zestlets: Option<&str>, clusters: Option<&str>) -> Option<bool> {
    zestlets.or(clusters).map(|v| v != "0" && !v.is_empty())
}

impl ClusterSettings {
    /// Defaults plus `ZESTENGINE_ZESTLETS` or its older name `ZESTENGINE_CLUSTERS` (`1` on, `0` off) and
    /// `ZESTENGINE_CLUSTER_PIXELS` (error target).
    pub(crate) fn from_env() -> Self {
        let mut s = Self {
            env_override: zestlets_env_override(std::env::var("ZESTENGINE_ZESTLETS").ok().as_deref(), std::env::var("ZESTENGINE_CLUSTERS").ok().as_deref()),
            ..Default::default()
        };
        if let Some(px) = std::env::var("ZESTENGINE_CLUSTER_PIXELS").ok().and_then(|v| v.parse::<f32>().ok())
            && px.is_finite()
            && px > 0.0
        {
            s.target_px = px;
        }
        if let Some(ms) = std::env::var("ZESTENGINE_CLUSTER_BUDGET_MS").ok().and_then(|v| v.parse::<f32>().ok())
            && ms.is_finite()
            && ms >= 0.0
        {
            s.target_frame_ms = ms;
            s.target_from_env = true;
        }
        if let Some(px) = std::env::var("ZESTENGINE_CLUSTER_SHADOW_PX").ok().and_then(|v| v.parse::<f32>().ok())
            && px.is_finite()
            && px >= 0.0
        {
            s.shadow_px = px;
        }
        if let Ok(v) = std::env::var("ZESTENGINE_CLUSTER_STREAM") {
            s.stream = v != "0" && !v.is_empty();
        }
        if let Some(mb) = std::env::var("ZESTENGINE_CLUSTER_STREAM_MB").ok().and_then(|v| v.parse::<u32>().ok()) {
            s.stream_budget_mb = mb.max(1);
        }
        if let Some(n) = std::env::var("ZESTENGINE_CLUSTER_MAX_TRIANGLES").ok().and_then(|v| v.parse::<u32>().ok()) {
            s.max_triangles = n;
        }
        s
    }

    /// Whether the path is requested (the environment wins over the setting).
    pub fn enabled(&self) -> bool {
        self.env_override.unwrap_or(self.mode == ClusterMode::On)
    }
}

// ---------------------------------------------------------------------------
// GPU records (std430; sizes checked by the tests and static asserts)

/// One cluster. Same bytes as zestengine-assets `GpuCluster`. 96 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ClusterRec {
    pub bounds: [f32; 4],
    pub lod_sphere: [f32; 4],
    pub parent_lod_sphere: [f32; 4],
    pub lod_error: f32,
    pub parent_lod_error: f32,
    /// Axis snorm8 x3, cutoff unorm8 (255 = never cull).
    pub cone: [u8; 4],
    pub vertex_offset: u32,
    /// Bytes, relative to the mesh's triangle bytes; a multiple of 4.
    pub triangle_offset: u32,
    pub vertex_count: u16,
    pub triangle_count: u16,
    pub bone_offset: u32,
    pub bone_count: u16,
    pub flags: u16,
    pub morph_offset: u32,
    pub morph_count: u32,
    pub section: u32,
    pub level: u32,
}

/// Quantised vertex. Same bytes as zestengine-assets `ClusterVertex`. 28 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ClusterVertexRec {
    pub pos: [u16; 3],
    /// Bit 0: tangent handedness negative.
    pub flags: u16,
    pub normal: [i16; 2],
    pub tangent: [i16; 2],
    pub uv: [f32; 2],
    /// Vertex colour, RGBA8 unorm (white when the mesh has none).
    pub color: [u8; 4],
}

/// One BVH slot. Same bytes as zestengine-assets `BvhChild`. 48 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct BvhChildRec {
    pub bounds: [f32; 4],
    pub lod_sphere: [f32; 4],
    pub max_parent_error: f32,
    pub child: u32,
    pub count: u32,
    pub pad: u32,
}

/// Four-wide BVH node. Same bytes as zestengine-assets `BvhNode`. 192 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct BvhNodeRec {
    pub children: [BvhChildRec; 4],
}

/// One DAG in the pool (`MeshRec` in cluster_common.glsl). 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct MeshRec {
    /// node_base, node_count, cluster_base, cluster_count.
    pub a: [u32; 4],
    /// vertex_base, vertex_count, tri_base (bytes), tri_len (bytes).
    pub b: [u32; 4],
    /// mesh_min xyz, quantisation step.
    pub grid: [f32; 4],
    /// Skin stream base, bone list base, morph entry base (pool entries); w 1 =
    /// the mesh has skin / blendshape data (GEO-5).
    pub c: [u32; 4],
}

/// Skin weights of one cluster vertex (`ClusterVertexSkin`). 8 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ClusterSkinRec {
    /// Indices into the cluster's bone list.
    pub bones: [u8; 4],
    /// Unorm8, summing to 255 (all zero = unweighted).
    pub weights: [u8; 4],
}

/// One sparse blendshape delta of a cluster vertex (`MorphEntry`). 16 bytes; f16 deltas.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct MorphEntryRec {
    pub target: u16,
    /// Cluster-local vertex.
    pub vertex: u16,
    pub d_pos: [u16; 3],
    pub d_normal: [u16; 3],
}

/// Per skinned instance (`SkinInst` in cluster_skin.comp). 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct SkinInstRec {
    /// Palette offset, previous palette offset (matrices), joint count, weights offset.
    pub a: [u32; 4],
    /// Shape count.
    pub b: [u32; 4],
}

/// Push constants of cluster_skin.comp. 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SkinPush {
    pub mode: u32,
    pub late: u32,
    pub batch_count: u32,
    pub list_cap: u32,
    pub pool_cap: u32,
    pub pad: [u32; 3],
}

/// A drawn instance (`Inst` in cluster_common.glsl). 160 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct InstRec {
    pub model: [[f32; 4]; 4],
    pub color: [f32; 4],
    pub param: [f32; 4],
    /// Object-space bounding sphere.
    pub sphere: [f32; 4],
    /// x mesh record, y material batch, z skin instance + 1 (0 = static), w pose
    /// margin (f32 bits; skinned instances).
    pub mesh_batch: [u32; 4],
    /// Object-space camera xyz, w 1 = the cone test is valid.
    pub eye: [f32; 4],
    /// x LOD error scale (the triangle budget's coarsening; 0 or 1 = none).
    pub extra: [f32; 4],
}

/// Per-batch region of the visible list: early base, capacity, late base, -.
pub type BatchRec = [u32; 4];

/// Push constants of cluster_cull.comp. 96 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CullPush {
    pub late_vp: [[f32; 4]; 4],
    pub mode: u32,
    pub count: u32,
    pub parity: u32,
    pub late: u32,
    pub node_cap: u32,
    pub occ_cap: u32,
    pub batch_count: u32,
    /// Streaming: capacity of the page table (offset of the touched flags in the I/O buffer).
    pub stream_pages: u32,
}

const _: () = assert!(size_of::<ClusterRec>() == 96);
const _: () = assert!(size_of::<ClusterVertexRec>() == 28);
const _: () = assert!(size_of::<BvhChildRec>() == 48);
const _: () = assert!(size_of::<BvhNodeRec>() == 192);
const _: () = assert!(size_of::<MeshRec>() == 64);
const _: () = assert!(size_of::<ClusterSkinRec>() == 8);
const _: () = assert!(size_of::<MorphEntryRec>() == 16);
const _: () = assert!(size_of::<SkinInstRec>() == 32);
const _: () = assert!(size_of::<SkinPush>() == 32);
const _: () = assert!(size_of::<InstRec>() == 160);
const _: () = assert!(size_of::<CullPush>() == 96);
const _: () = assert!(size_of::<GpuView>() == 208);

// ---------------------------------------------------------------------------
// A DAG to install

/// Everything the GPU needs of one mesh's DAG (static meshes only). Built by
/// the engine from `zestengine-assets::ClusterDag` (the records are the same
/// bytes, so the conversion is a `bytemuck` cast).
#[derive(Clone, Debug, Default)]
pub struct ClusterMeshUpload {
    pub clusters: Vec<ClusterRec>,
    pub vertices: Vec<ClusterVertexRec>,
    /// Cluster-local triangle indices, 3 bytes per triangle, runs padded to 4.
    pub triangles: Vec<u8>,
    pub nodes: Vec<BvhNodeRec>,
    pub mesh_min: [f32; 3],
    /// Power-of-two quantisation step (position = mesh_min + q * step).
    pub step: f32,
    /// The DAG carries skin data or blendshapes (drawn through cluster_skin.comp).
    pub skinned: bool,
    /// Skin weights parallel to `vertices` (empty = unweighted, blendshapes only).
    pub skin: Vec<ClusterSkinRec>,
    /// Joint index of each cluster bone slot (`GpuCluster::bone_offset` indexes it).
    pub bones: Vec<u32>,
    /// Sparse blendshape deltas (`GpuCluster::morph_offset` indexes it).
    pub morphs: Vec<MorphEntryRec>,
    /// Blendshape targets (entries' `target` is below this).
    pub morph_targets: u32,
    /// Geometry streaming (GEO-6): `vertices` / `triangles` then hold only the
    /// root page (page 0), the cluster records are page-relative (`morph_offset`
    /// = page, `morph_count` = page of the finer children, [`PAGE_NONE`] for
    /// none) and the other pages are installed on demand.
    pub stream: Option<StreamLayout>,
    /// Largest displacement of each blendshape target, when `morphs` cannot give it:
    /// a streamed skinned DAG keeps its blendshape entries in the pages, not in the
    /// upload, so the loader supplies the reach it computed from them (empty = derive
    /// it from `morphs`). The pose margin of the cull depends on it.
    pub morph_reach: Vec<f32>,
}

/// Pages of a streamed DAG (page 0 is the always-resident root page).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StreamLayout {
    pub pages: Vec<PageLayout>,
}

/// One page: payload sizes and the pages that must be resident with it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PageLayout {
    pub vertices: u32,
    /// Triangle bytes, a multiple of 4.
    pub tri_bytes: u32,
    pub deps: Vec<u32>,
}

impl PageLayout {
    pub fn bytes(&self) -> u32 {
        self.vertices * size_of::<ClusterVertexRec>() as u32 + self.tri_bytes
    }
}

impl StreamLayout {
    /// Share of the streamed (non-root) pages' bytes that is triangle block.
    pub fn tri_share(&self) -> f64 {
        let (tri, all) = self.pages.iter().skip(1).fold((0u64, 0u64), |(t, a), p| (t + u64::from(p.tri_bytes), a + u64::from(p.bytes())));
        if all == 0 { 0.0 } else { tri as f64 / all as f64 }
    }
}

/// The page id meaning "none" in a cluster's `morph_count`.
pub const NO_PAGE: u32 = u32::MAX;

impl ClusterMeshUpload {
    /// Structural checks the shader relies on: counts, offsets inside their
    /// arrays, triangle bytes inside the cluster's vertex count, BVH children
    /// forward and in range. Everything the GPU indexes with is covered, so an
    /// installed DAG cannot make the shader read outside its own ranges.
    pub fn validate(&self) -> Result<(), String> {
        macro_rules! check {
            ($c:expr, $($m:tt)*) => {
                if !$c {
                    return Err(format!($($m)*));
                }
            };
        }
        check!(!self.clusters.is_empty() && self.clusters.len() <= MAX_CLUSTERS, "cluster count {}", self.clusters.len());
        check!(!self.nodes.is_empty() && self.nodes.len() <= MAX_CLUSTERS, "node count {}", self.nodes.len());
        check!(!self.vertices.is_empty() && self.vertices.len() <= MAX_CLUSTERS * CLUSTER_MAX_VERTS, "vertex count");
        if let Some(st) = &self.stream {
            return self.validate_streamed(st);
        }
        check!(self.triangles.len().is_multiple_of(4) && self.triangles.len() < u32::MAX as usize / 2, "triangle bytes {}", self.triangles.len());
        check!(self.step.is_finite() && self.step > 0.0, "quantisation step {}", self.step);
        check!(self.mesh_min.iter().all(|v| v.is_finite()), "mesh_min");
        if self.skinned {
            check!(self.skin.is_empty() || self.skin.len() == self.vertices.len(), "skin stream length {}", self.skin.len());
            check!(self.bones.len() <= MAX_CLUSTERS * CLUSTER_MAX_BONES, "bone list length");
            check!(self.morphs.len() <= MAX_CLUSTERS * CLUSTER_MAX_VERTS, "morph entry count");
        } else {
            check!(self.skin.is_empty() && self.bones.is_empty() && self.morphs.is_empty(), "skin data on a static DAG");
        }
        for (i, c) in self.clusters.iter().enumerate() {
            let (tc, vc) = (c.triangle_count as usize, c.vertex_count as usize);
            check!((1..=CLUSTER_MAX_TRIS).contains(&tc), "cluster {i}: {tc} triangles");
            check!((1..=CLUSTER_MAX_VERTS).contains(&vc), "cluster {i}: {vc} vertices");
            check!(c.vertex_offset as usize + vc <= self.vertices.len(), "cluster {i}: vertices out of range");
            check!(c.triangle_offset % 4 == 0 && c.triangle_offset as usize + tc * 3 <= self.triangles.len(), "cluster {i}: triangles out of range");
            let t = &self.triangles[c.triangle_offset as usize..][..tc * 3];
            check!(t.iter().all(|&k| (k as usize) < vc), "cluster {i}: triangle index past the vertex count");
            check!(c.bounds.iter().chain(&c.lod_sphere).chain(&c.parent_lod_sphere).all(|v| v.is_finite()), "cluster {i}: non-finite sphere");
            check!(c.lod_error.is_finite() && c.lod_error >= 0.0 && !c.parent_lod_error.is_nan(), "cluster {i}: error");
            if self.skinned {
                let (bo, bc) = (c.bone_offset as usize, c.bone_count as usize);
                check!(bc <= CLUSTER_MAX_BONES && bo + bc <= self.bones.len(), "cluster {i}: bone list out of range");
                let (mo, mc) = (c.morph_offset as usize, c.morph_count as usize);
                check!(mo + mc <= self.morphs.len(), "cluster {i}: morph range out of range");
                check!(
                    self.morphs[mo..mo + mc].iter().all(|e| (e.vertex as usize) < vc && (e.target as u32) < self.morph_targets),
                    "cluster {i}: morph entry out of range"
                );
                if !self.skin.is_empty() {
                    let sk = &self.skin[c.vertex_offset as usize..][..vc];
                    check!(
                        sk.iter().all(|s| (0..4).all(|k| s.weights[k] == 0 || (s.bones[k] as usize) < bc)),
                        "cluster {i}: skin bone index past the cluster's bone list"
                    );
                }
            }
        }
        for (n, node) in self.nodes.iter().enumerate() {
            for ch in &node.children {
                if ch.child == BVH_EMPTY {
                    continue;
                }
                check!(
                    ch.bounds.iter().chain(&ch.lod_sphere).all(|v| v.is_finite()) && !ch.max_parent_error.is_nan(),
                    "node {n}: non-finite child"
                );
                if ch.child & BVH_LEAF != 0 {
                    let first = (ch.child & !BVH_LEAF) as usize;
                    check!(
                        ch.count >= 1 && ch.count <= BVH_RUN_MAX && first + ch.count as usize <= self.clusters.len(),
                        "node {n}: leaf run out of range"
                    );
                } else {
                    check!((ch.child as usize) > n && (ch.child as usize) < self.nodes.len(), "node {n}: child {} not forward", ch.child);
                }
            }
        }
        Ok(())
    }

    /// The streamed variant of [`validate`](Self::validate): offsets are page
    /// relative, only the root page's payload is present (its triangle indices
    /// are checked now, the other pages' when they are read).
    fn validate_streamed(&self, st: &StreamLayout) -> Result<(), String> {
        macro_rules! check {
            ($c:expr, $($m:tt)*) => {
                if !$c {
                    return Err(format!($($m)*));
                }
            };
        }
        let np = st.pages.len();
        check!((1..=STREAM_HARD_MAX_PAGES).contains(&np), "page count {np}");
        // Skin streams, bone lists and blendshape entries of a streamed DAG live in the pages (the
        // triangle block of each page, see assets cluster_pages.rs): none may ride in the upload.
        check!(self.skin.is_empty() && self.bones.is_empty() && self.morphs.is_empty(), "skin data in the upload of a streamed DAG");
        check!(self.skinned || (self.morph_targets == 0 && self.morph_reach.is_empty()), "blendshapes on a static streamed DAG");
        check!(self.morph_reach.is_empty() || self.morph_reach.len() == self.morph_targets as usize, "blendshape reach count");
        check!(self.morph_reach.iter().all(|r| r.is_finite() && *r >= 0.0), "blendshape reach not finite");
        check!(self.step.is_finite() && self.step > 0.0 && self.mesh_min.iter().all(|v| v.is_finite()), "quantisation grid");
        let root = &st.pages[0];
        check!(root.deps.is_empty(), "the root page has dependencies");
        check!(self.vertices.len() == root.vertices as usize && self.triangles.len() == root.tri_bytes as usize, "root page payload differs from its layout");
        check!(self.triangles.len().is_multiple_of(4), "triangle bytes not padded to 4");
        for (i, p) in st.pages.iter().enumerate() {
            check!(p.tri_bytes % 4 == 0 && p.vertices as usize <= MAX_CLUSTERS * CLUSTER_MAX_VERTS, "page {i}: sizes");
            check!(p.bytes() as usize <= (64 << 20), "page {i}: {} bytes", p.bytes());
            check!(p.deps.iter().all(|&d| (d as usize) < np && d as usize != i), "page {i}: dependency out of range");
            check!(i == 0 || !p.deps.is_empty() || np == 1, "page {i}: no dependency (every page needs the root chain)");
        }
        for (i, c) in self.clusters.iter().enumerate() {
            let (tc, vc) = (c.triangle_count as usize, c.vertex_count as usize);
            check!((1..=CLUSTER_MAX_TRIS).contains(&tc) && (1..=CLUSTER_MAX_VERTS).contains(&vc), "cluster {i}: counts");
            let page = c.morph_offset as usize;
            check!(page < np, "cluster {i}: page {page} out of range");
            check!(c.morph_count == NO_PAGE || (c.morph_count as usize) < np, "cluster {i}: child page out of range");
            let p = &st.pages[page];
            // A skinned page holds the skin records (8 bytes a vertex) before the triangles.
            let skin_bytes = if self.skinned { p.vertices as usize * size_of::<ClusterSkinRec>() } else { 0 };
            check!(c.vertex_offset as usize + vc <= p.vertices as usize, "cluster {i}: vertices outside its page");
            check!(c.triangle_offset % 4 == 0 && skin_bytes + c.triangle_offset as usize + tc * 3 <= p.tri_bytes as usize, "cluster {i}: triangles outside its page");
            if page == 0 {
                let t = &self.triangles[skin_bytes + c.triangle_offset as usize..][..tc * 3];
                check!(t.iter().all(|&k| (k as usize) < vc), "cluster {i}: triangle index past the vertex count");
            }
            if self.skinned {
                check!((c.bone_count as usize) <= CLUSTER_MAX_BONES, "cluster {i}: {} bones", c.bone_count);
                // The aux record (morph count, joint ids, morph entries) starts inside the page.
                let aux = c.bone_offset as usize * 4;
                check!(aux >= skin_bytes && aux + 4 * (1 + (c.bone_count as usize).div_ceil(2)) <= p.tri_bytes as usize, "cluster {i}: aux record outside its page");
            } else {
                check!(c.bone_count == 0, "cluster {i}: bones on a static mesh");
            }
            check!(c.bounds.iter().chain(&c.lod_sphere).chain(&c.parent_lod_sphere).all(|v| v.is_finite()), "cluster {i}: non-finite sphere");
            check!(c.lod_error.is_finite() && c.lod_error >= 0.0 && !c.parent_lod_error.is_nan(), "cluster {i}: error");
        }
        for (n, node) in self.nodes.iter().enumerate() {
            for ch in &node.children {
                if ch.child == BVH_EMPTY {
                    continue;
                }
                check!(ch.bounds.iter().chain(&ch.lod_sphere).all(|v| v.is_finite()) && !ch.max_parent_error.is_nan(), "node {n}: non-finite child");
                if ch.child & BVH_LEAF != 0 {
                    let first = (ch.child & !BVH_LEAF) as usize;
                    check!(ch.count >= 1 && ch.count <= BVH_RUN_MAX && first + ch.count as usize <= self.clusters.len(), "node {n}: leaf run out of range");
                } else {
                    check!((ch.child as usize) > n && (ch.child as usize) < self.nodes.len(), "node {n}: child {} not forward", ch.child);
                }
            }
        }
        Ok(())
    }

    /// Deepest BVH level (the root is 0); links that are not forward are ignored.
    pub fn max_depth(&self) -> u32 {
        let mut depth = vec![0u32; self.nodes.len()];
        let mut deepest = 0;
        for n in 0..self.nodes.len() {
            deepest = deepest.max(depth[n]);
            for ch in &self.nodes[n].children {
                // Forward, in-range inner children only (validated uploads have no others).
                if ch.child != BVH_EMPTY && ch.child & BVH_LEAF == 0 && (ch.child as usize) > n && (ch.child as usize) < depth.len() {
                    let c = ch.child as usize;
                    depth[c] = depth[c].max(depth[n] + 1);
                }
            }
        }
        deepest
    }

    /// Largest displacement of each blendshape target (object space), for the
    /// pose margin: a weight `w` moves a vertex at most `w * reach[target]`.
    pub fn morph_reach(&self) -> Vec<f32> {
        if !self.morph_reach.is_empty() {
            return self.morph_reach.clone();
        }
        let mut reach = vec![0.0f32; self.morph_targets as usize];
        for e in &self.morphs {
            if let Some(r) = reach.get_mut(e.target as usize) {
                let d = e.d_pos.map(|b| half::f16::from_bits(b).to_f32());
                *r = r.max((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
            }
        }
        reach
    }

    /// Vertices of the coarsest cut (every root cluster): the least a skinned
    /// instance of this DAG can pose in a frame (skin pool admission).
    pub fn root_vertices(&self) -> u32 {
        self.clusters.iter().filter(|c| c.parent_lod_error == f32::MAX).map(|c| c.vertex_count as u32).sum()
    }

    /// Triangles of the finest level.
    pub fn leaf_triangles(&self) -> u64 {
        self.clusters.iter().filter(|c| c.level == 0).map(|c| c.triangle_count as u64).sum()
    }

    fn tri_words(&self) -> usize {
        self.triangles.len() / 4
    }
}

// ---------------------------------------------------------------------------
// Sub-allocation

/// First-fit range allocator with coalescing, over `cap` units. The free list
/// is kept sorted by start. Used for the pool's four arrays.
#[derive(Clone, Debug)]
pub struct RangeAlloc {
    cap: u32,
    free: Vec<(u32, u32)>,
}

impl RangeAlloc {
    pub fn new(cap: u32) -> Self {
        Self { cap, free: if cap > 0 { vec![(0, cap)] } else { Vec::new() } }
    }

    pub fn capacity(&self) -> u32 {
        self.cap
    }

    pub fn used(&self) -> u32 {
        self.cap - self.free.iter().map(|r| r.1).sum::<u32>()
    }

    pub fn largest_free(&self) -> u32 {
        self.free.iter().map(|r| r.1).max().unwrap_or(0)
    }

    /// Extend the managed range to `new_cap` units; the new tail joins the free list.
    pub fn grow(&mut self, new_cap: u32) {
        if new_cap > self.cap {
            let (old, n) = (self.cap, new_cap - self.cap);
            self.cap = new_cap;
            self.free(old, n);
        }
    }

    /// Reserve `n` units; None when no free run is long enough.
    pub fn alloc(&mut self, n: u32) -> Option<u32> {
        if n == 0 {
            return Some(0);
        }
        let i = self.free.iter().position(|r| r.1 >= n)?;
        let (start, len) = self.free[i];
        if len == n {
            self.free.remove(i);
        } else {
            self.free[i] = (start + n, len - n);
        }
        Some(start)
    }

    /// Return a range; merges with its neighbours. Ranges must come from `alloc`.
    pub fn free(&mut self, start: u32, n: u32) {
        if n == 0 {
            return;
        }
        debug_assert!(start + n <= self.cap);
        let i = self.free.partition_point(|r| r.0 < start);
        self.free.insert(i, (start, n));
        // Merge with the next run, then the previous one.
        if i + 1 < self.free.len() && self.free[i].0 + self.free[i].1 == self.free[i + 1].0 {
            self.free[i].1 += self.free[i + 1].1;
            self.free.remove(i + 1);
        }
        if i > 0 && self.free[i - 1].0 + self.free[i - 1].1 == self.free[i].0 {
            self.free[i - 1].1 += self.free[i].1;
            self.free.remove(i);
        }
    }
}

/// Pool capacities in elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolSizes {
    pub clusters: u32,
    pub vertices: u32,
    /// 4-byte words of triangle bytes.
    pub tri_words: u32,
    pub nodes: u32,
    /// Skin weight records (skinned DAGs; 8 bytes each).
    pub skin: u32,
    /// Cluster bone list entries (4 bytes each).
    pub bones: u32,
    /// Blendshape entries (16 bytes each).
    pub morphs: u32,
    /// The device's storage buffer range (bytes) a pool may grow to; set by
    /// [`PoolSizes::clamped`] (`u32::MAX` = unlimited).
    pub range: u32,
}

impl Default for PoolSizes {
    /// About 140 MB: room for roughly 20 million leaf triangles; plus 13 MB for
    /// skin weights (a million skinned vertices), bone lists and blendshape deltas.
    fn default() -> Self {
        Self { clusters: 1 << 17, vertices: 1 << 22, tri_words: 1 << 22, nodes: 1 << 16, skin: 1 << 20, bones: 1 << 18, morphs: 1 << 18, range: u32::MAX }
    }
}

impl PoolSizes {
    /// Shrink to fit a device's storage buffer range limit (bytes).
    pub fn clamped(mut self, max_range: u32) -> Self {
        let fit = |n: u32, elem: u32| n.min((max_range / elem).max(1));
        self.clusters = fit(self.clusters, size_of::<ClusterRec>() as u32);
        self.vertices = fit(self.vertices, size_of::<ClusterVertexRec>() as u32);
        self.tri_words = fit(self.tri_words, 4);
        self.nodes = fit(self.nodes, size_of::<BvhNodeRec>() as u32);
        self.skin = fit(self.skin, size_of::<ClusterSkinRec>() as u32);
        self.bones = fit(self.bones, 4);
        self.morphs = fit(self.morphs, size_of::<MorphEntryRec>() as u32);
        self.range = max_range;
        self
    }
}

/// Per-pool growth limits in elements: `GROW_LIMIT` x the start size, within the device range.
fn pool_limits(sizes: PoolSizes) -> [u32; 7] {
    let start = [sizes.clusters, sizes.vertices, sizes.tri_words, sizes.nodes, sizes.skin, sizes.bones, sizes.morphs];
    let elem = [size_of::<ClusterRec>(), size_of::<ClusterVertexRec>(), 4, size_of::<BvhNodeRec>(), size_of::<ClusterSkinRec>(), 4, size_of::<MorphEntryRec>()];
    std::array::from_fn(|k| {
        start[k].saturating_mul(GROW_LIMIT).min((sizes.range / elem[k] as u32).max(start[k]))
    })
}

struct Pool {
    buffer: Buffer,
    ranges: RangeAlloc,
}

/// A DAG resident in the pool.
#[derive(Clone, Debug)]
pub(crate) struct Resident {
    rec: MeshRec,
    /// (start, len) in elements of the pools: clusters, vertices, triangle
    /// words, nodes, skin records, bone entries, morph entries.
    ranges: [(u32, u32); 7],
    depth: u32,
    /// Meshes drawing from this record (instances of one avatar share one DAG).
    refs: u32,
    /// Triangles of the finest level (the budget's demand estimate).
    leaf_triangles: u32,
    /// Vertices of the coarsest cut (skin pool admission).
    root_vertices: u32,
    /// Largest displacement of each blendshape target (pose margin).
    reach: Box<[f32]>,
    /// The pool ranges hold the DAG. False between a deferred install
    /// (`install_staged`) and the frame that records its copies.
    ready: bool,
    /// Pages of a streamed DAG (GEO-6).
    stream: Option<stream::Streamed>,
}

/// A DAG staged in a host-visible buffer (on a loader thread, by
/// `Uploader::stage_cluster_dag`), waiting for `Renderer::install_staged_cluster_dag`.
/// `Send`: the staging buffer and plain data.
pub struct StagedClusterDag {
    staging: Buffer,
    /// Byte length of each of the seven arrays inside the staging buffer, back to back.
    part_bytes: [usize; 7],
    want: [u32; 7],
    mesh_min: [f32; 3],
    step: f32,
    skinned: bool,
    triangles_len: u32,
    depth: u32,
    leaf_triangles: u32,
    root_vertices: u32,
    reach: Box<[f32]>,
    stream: Option<StreamLayout>,
}

impl StagedClusterDag {
    /// Validate `up` and copy its arrays into one host-visible staging buffer.
    pub(crate) fn new(device: &ash::Device, allocator: &mut Allocator, up: &ClusterMeshUpload) -> Result<Self> {
        up.validate().map_err(anyhow::Error::msg)?;
        if up.max_depth() >= BVH_MAX_DEPTH {
            bail!("BVH deeper than {BVH_MAX_DEPTH} levels");
        }
        let want = Self::want(up);
        let zero_skin = Self::zero_skin(up);
        let part_bytes: [usize; 7] = std::array::from_fn(|k| Self::part(up, &zero_skin, k).len());
        let total: usize = part_bytes.iter().sum();
        let mut staging = Buffer::new(device, allocator, total.max(4) as u64, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "cluster-staging")?;
        let mut at = 0;
        for k in 0..7 {
            let part = Self::part(up, &zero_skin, k);
            if !part.is_empty() {
                staging.write(at, part);
                at += part.len();
            }
        }
        Ok(Self {
            staging,
            part_bytes,
            want,
            mesh_min: up.mesh_min,
            step: up.step,
            skinned: up.skinned,
            triangles_len: up.triangles.len() as u32,
            depth: up.max_depth(),
            leaf_triangles: up.leaf_triangles().min(u32::MAX as u64) as u32,
            root_vertices: up.root_vertices(),
            reach: up.morph_reach().into(),
            stream: up.stream.clone(),
        })
    }

    /// Element counts per pool. A skinned DAG gets a skin record per vertex
    /// (zero = unweighted when it only has blendshapes), so the shader can
    /// index it unconditionally.
    fn want(up: &ClusterMeshUpload) -> [u32; 7] {
        // A streamed DAG's skin records are in its pages; the skin pool holds nothing of it.
        let skin_len = if up.skinned && up.stream.is_none() { up.vertices.len() as u32 } else { 0 };
        [up.clusters.len() as u32, up.vertices.len() as u32, up.tri_words() as u32, up.nodes.len() as u32, skin_len, up.bones.len() as u32, up.morphs.len() as u32]
    }

    fn zero_skin(up: &ClusterMeshUpload) -> Vec<ClusterSkinRec> {
        if up.skinned && up.skin.is_empty() && up.stream.is_none() { vec![ClusterSkinRec::default(); up.vertices.len()] } else { Vec::new() }
    }

    /// The bytes of array `k` (pool order).
    fn part<'a>(up: &'a ClusterMeshUpload, zero_skin: &'a [ClusterSkinRec], k: usize) -> &'a [u8] {
        match k {
            0 => bytemuck::cast_slice(&up.clusters),
            1 => bytemuck::cast_slice(&up.vertices),
            2 => &up.triangles,
            3 => bytemuck::cast_slice(&up.nodes),
            4 if !up.skinned || up.stream.is_some() => &[],
            4 if up.skin.is_empty() => bytemuck::cast_slice(zero_skin),
            4 => bytemuck::cast_slice(&up.skin),
            5 => bytemuck::cast_slice(&up.bones),
            _ => bytemuck::cast_slice(&up.morphs),
        }
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        self.staging.destroy(device, allocator);
    }
}

/// Copies of an installed DAG not yet recorded into a frame's command buffer.
struct QueuedUpload {
    id: u32,
    staging: Buffer,
    /// (pool, region) per non-empty array.
    copies: Vec<(usize, vk::BufferCopy)>,
}

/// Bytes per element of each pool (clusters, vertices, triangle words, nodes, skin, bones, morphs).
const POOL_ELEM: [usize; 7] = [size_of::<ClusterRec>(), size_of::<ClusterVertexRec>(), 4, size_of::<BvhNodeRec>(), size_of::<ClusterSkinRec>(), 4, size_of::<MorphEntryRec>()];

struct Denied {
    mesh: usize,
    dist: f32,
    /// Serial before which it stays denied.
    until: u64,
    /// Serial the mesh was last asked about (entries nobody asks for are dropped).
    asked: u64,
}

/// Smallest per-frame skinned-vertex pool.
const SKIN_MIN_CAP: u32 = 64 * 1024;
/// Largest admission error scale (`SkinAdmission::scale`).
const SKIN_MAX_SCALE: f32 = 256.0;
/// Fraction of the pool the coarsest cuts of the admitted instances may take before the farthest are denied.
const SKIN_ROOT_FIT: f32 = 0.9;

/// Which skinned instances may draw from clusters, and how coarse they must be, while the skinned-vertex
/// pool is under pressure.
///
/// The per-frame pool cap follows the selected cut (`frame_cap`), growing when a retired frame used
/// more than 60% of it (or overflowed). Only at the largest pool does pressure coarsen: `scale` (>= 1)
/// multiplies the LOD error of every skinned instance by `scale^f`, `f` rising from 0 for the nearest to
/// 1 for the farther half (`ClusterSystem::upload_frame`), and relaxes after a calm spell. An instance
/// goes to the ordinary skinned draw (`allowed` is false) only when even the coarsest cuts of the admitted
/// instances cannot fit the pool: the farthest are denied until they do, and one warning says so.
/// Pure bookkeeping (CPU-tested); `ClusterSystem::cluster_begin` feeds it the retired frame's pool use.
pub(crate) struct SkinAdmission {
    /// (mesh, camera distance) admitted this frame / the last completed one.
    cur: Vec<(usize, f32)>,
    prev: Vec<(usize, f32)>,
    /// Vertices of each skinned mesh's coarsest cut (noted when an instance of it is added).
    roots: std::collections::HashMap<usize, u32>,
    denied: Vec<Denied>,
    /// Instances sent to the ordinary path this frame / last frame (stats).
    fallbacks: u32,
    pub(crate) last_fallbacks: u32,
    /// Pressure signals before this serial describe frames recorded before the
    /// last change and are ignored.
    settle_until: u64,
    calm: u32,
    /// Admission error scale, 1 = none.
    scale: f32,
    /// Current per-frame pool cap (vertices); 0 until the first `frame_cap`.
    cap: u32,
    warned_roots: bool,
}

impl Default for SkinAdmission {
    fn default() -> Self {
        Self {
            cur: Vec::new(),
            prev: Vec::new(),
            roots: Default::default(),
            denied: Vec::new(),
            fallbacks: 0,
            last_fallbacks: 0,
            settle_until: 0,
            calm: 0,
            scale: 1.0,
            cap: 0,
            warned_roots: false,
        }
    }
}

impl SkinAdmission {
    pub(crate) fn begin_frame(&mut self) {
        std::mem::swap(&mut self.cur, &mut self.prev);
        self.cur.clear();
        self.last_fallbacks = std::mem::take(&mut self.fallbacks);
    }

    /// Whether skinned `mesh` at `dist` from the camera may use clusters this frame.
    pub(crate) fn allowed(&mut self, mesh: usize, dist: f32, serial: u64) -> bool {
        if let Some(d) = self.denied.iter_mut().find(|d| d.mesh == mesh) {
            d.dist = dist;
            d.asked = serial;
            self.fallbacks += 1;
            return false;
        }
        self.cur.push((mesh, dist));
        true
    }

    /// Record the vertices of `mesh`'s coarsest cut (what it costs at best).
    pub(crate) fn note_root_vertices(&mut self, mesh: usize, vertices: u32) {
        self.roots.insert(mesh, vertices);
    }

    pub(crate) fn is_denied(&self, mesh: usize) -> bool {
        self.denied.iter().any(|d| d.mesh == mesh)
    }

    pub(crate) fn denied_count(&self) -> usize {
        self.denied.len()
    }

    /// The error scale applied to the farthest instances (1 = none).
    pub(crate) fn scale(&self) -> f32 {
        self.scale
    }

    /// This frame's pool cap: the selected cut's estimated skinned vertices with 25% headroom, never
    /// below the cap in force or the smallest pool, never above `pool_vertices`.
    pub(crate) fn frame_cap(&mut self, estimated_vertices: u64, pool_vertices: u32) -> u32 {
        let want = (estimated_vertices as f64 * 1.25).min(u32::MAX as f64) as u32;
        self.cap = self.cap.max(want).max(SKIN_MIN_CAP).checked_next_power_of_two().unwrap_or(u32::MAX).min(pool_vertices.max(1));
        self.cap
    }

    fn root_sum(&self, meshes: impl Iterator<Item = usize>) -> u64 {
        meshes.map(|m| self.roots.get(&m).copied().unwrap_or(0) as u64).sum()
    }

    /// React to the pool use of a retired frame (`used` of its `cap` vertices; `overflow`: the shader
    /// ran out; `pool_vertices`: the largest pool). Below the largest pool: grow the cap. At it: raise or
    /// relax the error scale, and deny the farthest only while even the coarsest cuts do not fit.
    pub(crate) fn pressure(&mut self, used: u32, cap: u32, pool_vertices: u32, overflow: bool, serial: u64) {
        // A held-back mesh nothing draws any more (freed, culled away) is forgotten.
        self.denied.retain(|d| serial.saturating_sub(d.asked) < 90 || d.until > serial);
        if cap == 0 || serial < self.settle_until {
            return;
        }
        let settle = serial + FRAMES_IN_FLIGHT as u64 + 2;
        if cap < pool_vertices {
            // The shader's `used` counts the whole demand even on overflow: jump straight to it.
            if overflow || used as f32 > 0.6 * cap as f32 {
                let want = ((used as f64 * 1.25) as u32).max(if overflow { 0 } else { cap.saturating_mul(2) });
                self.cap = self.cap.max(want).checked_next_power_of_two().unwrap_or(u32::MAX).min(pool_vertices);
                self.settle_until = settle;
            }
            return;
        }
        if overflow || used as f32 > 0.85 * cap as f32 {
            let k = (used as f32 / (0.7 * cap as f32)).sqrt().clamp(1.25, 4.0);
            self.scale = (self.scale * k).min(SKIN_MAX_SCALE);
            self.calm = 0;
            self.settle_until = settle;
        } else if used as f32 > 0.5 * cap as f32 {
            self.calm = 0;
        } else {
            self.calm += 1;
            if self.calm >= 30 {
                self.calm = 0;
                if self.scale > 1.0 {
                    self.scale = (self.scale / 1.2).max(1.0);
                    self.settle_until = settle;
                }
            }
        }
        // Denial: only when the coarsest cuts of what is admitted do not fit the pool even at the largest scale.
        let fit = SKIN_ROOT_FIT as f64 * pool_vertices as f64;
        if self.scale >= SKIN_MAX_SCALE && !self.prev.is_empty() {
            let mut by_near = self.prev.clone();
            by_near.sort_by(|a, b| a.1.total_cmp(&b.1));
            let total = self.root_sum(by_near.iter().map(|e| e.0));
            if total as f64 > fit {
                let (mut sum, mut cut) = (0u64, by_near.len());
                for (i, e) in by_near.iter().enumerate() {
                    sum += self.roots.get(&e.0).copied().unwrap_or(0) as u64;
                    if sum as f64 > fit {
                        cut = i;
                        break;
                    }
                }
                if !self.warned_roots {
                    self.warned_roots = true;
                    tracing::warn!(
                        "cluster path: the coarsest cuts of {} skinned instances need {} vertices, over {:.0}% of the {} vertex skin pool; \
                         the farthest {} fall back to the ordinary skinned draw (raise ZESTENGINE_CLUSTER_SKIN_VERTICES)",
                        by_near.len(),
                        total,
                        SKIN_ROOT_FIT * 100.0,
                        pool_vertices,
                        by_near.len() - cut
                    );
                }
                for &(mesh, dist) in &by_near[cut..] {
                    if !self.denied.iter().any(|d| d.mesh == mesh) {
                        self.denied.push(Denied { mesh, dist, until: serial + SKIN_DENY_FRAMES, asked: serial });
                    }
                }
                self.settle_until = settle;
                self.calm = 0;
                return;
            }
        }
        // The nearest denied instance returns once its coarsest cut fits beside what is admitted.
        if !self.denied.is_empty() && self.calm >= 1 {
            let nearest = self.denied.iter().enumerate().filter(|(_, d)| d.until <= serial).min_by(|a, b| a.1.dist.total_cmp(&b.1.dist)).map(|(i, _)| i);
            if let Some(i) = nearest {
                let admitted = self.root_sum(self.prev.iter().map(|e| e.0));
                let back = self.roots.get(&self.denied[i].mesh).copied().unwrap_or(0) as u64;
                if (admitted + back) as f64 <= fit {
                    self.denied.swap_remove(i);
                    self.calm = 0;
                    self.settle_until = settle;
                }
            }
        }
    }
}

/// The pooled DAG store: big storage buffers with sub-allocation, a free list
/// per pool and deferred frees (a freed DAG's ranges stay valid until the
/// frames that may still read them have retired).
pub(crate) struct ClusterStore {
    clusters: Pool,
    vertices: Pool,
    triangles: Pool,
    nodes: Pool,
    skin: Pool,
    bones: Pool,
    morphs: Pool,
    /// Mesh records by id; `None` = free id.
    records: Vec<Option<Resident>>,
    free_ids: Vec<u32>,
    by_mesh: HashMap<usize, u32>,
    /// (frame serial of the removal, record id): freed once FRAMES_IN_FLIGHT frames later.
    pending: Vec<(u64, u32)>,
    /// Most each pool may grow to, in elements (the device range, at most `GROW_LIMIT` x the start size).
    limits: [u32; 7],
    /// Buffers replaced by a bigger one: (serial of the swap, buffer), freed once
    /// no in-flight frame's descriptors can name them.
    retired: Vec<(u64, Buffer)>,
    /// Installed DAGs whose copies the next frame records (`install_staged`).
    uploads: Vec<QueuedUpload>,
    /// Staging buffers whose copies were recorded at a serial; destroyed once
    /// that frame has retired.
    spent: Vec<(u64, Buffer)>,
    /// Latest serial seen by `reclaim` (stamps retirements).
    serial: u64,
    /// How many times a pool grew (stats, tests).
    pub(crate) grown: u32,
    /// Geometry streaming: page table, blocks, policy (GEO-6).
    stream: stream::StreamState,
    /// A small buffer bound to the streaming bindings while streaming was never used.
    dummy: Buffer,
}

/// A full pool grows (reallocates bigger and copies) up to this multiple of its start size (and the device range).
/// Memory is only taken when a mesh needs it: with 4 the blendshape pool (1M entries) refused the lemonjuice face
/// (1.7M) and the streamed triangle pool refused two skinned meshes, leaving them on the ordinary path.
const GROW_LIMIT: u32 = 32;

impl ClusterStore {
    pub(crate) fn new(device: &ash::Device, allocator: &mut Allocator, sizes: PoolSizes) -> Result<Self> {
        let mut pool = |elems: u32, elem: usize, name: &str| -> Result<Pool> {
            Ok(Pool {
                buffer: Buffer::new(
                    device,
                    allocator,
                    elems as u64 * elem as u64,
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                    MemoryLocation::GpuOnly,
                    name,
                )?,
                ranges: RangeAlloc::new(elems),
            })
        };
        Ok(Self {
            clusters: pool(sizes.clusters, size_of::<ClusterRec>(), "cluster-pool-clusters")?,
            vertices: pool(sizes.vertices, size_of::<ClusterVertexRec>(), "cluster-pool-vertices")?,
            triangles: pool(sizes.tri_words, 4, "cluster-pool-triangles")?,
            nodes: pool(sizes.nodes, size_of::<BvhNodeRec>(), "cluster-pool-nodes")?,
            skin: pool(sizes.skin, size_of::<ClusterSkinRec>(), "cluster-pool-skin")?,
            bones: pool(sizes.bones, 4, "cluster-pool-bones")?,
            morphs: pool(sizes.morphs, size_of::<MorphEntryRec>(), "cluster-pool-morphs")?,
            records: Vec::new(),
            free_ids: Vec::new(),
            by_mesh: HashMap::new(),
            pending: Vec::new(),
            limits: pool_limits(sizes),
            retired: Vec::new(),
            uploads: Vec::new(),
            spent: Vec::new(),
            serial: 0,
            grown: 0,
            stream: Default::default(),
            dummy: Buffer::new(device, allocator, 64, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, "cluster-stream-dummy")?,
        })
    }

    pub(crate) fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        self.destroy_stream(device, allocator);
        self.dummy.destroy(device, allocator);
        for p in [
            &mut self.clusters,
            &mut self.vertices,
            &mut self.triangles,
            &mut self.nodes,
            &mut self.skin,
            &mut self.bones,
            &mut self.morphs,
        ] {
            p.buffer.destroy(device, allocator);
        }
        for (_, mut b) in self.retired.drain(..).chain(self.spent.drain(..)) {
            b.destroy(device, allocator);
        }
        for mut u in self.uploads.drain(..) {
            u.staging.destroy(device, allocator);
        }
    }

    /// The record `mesh` draws from, once its pool ranges hold the DAG.
    pub(crate) fn record_of(&self, mesh: usize) -> Option<u32> {
        let id = self.by_mesh.get(&mesh).copied()?;
        self.resident(id).is_some_and(|r| r.ready).then_some(id)
    }

    pub(crate) fn resident(&self, id: u32) -> Option<&Resident> {
        self.records.get(id as usize).and_then(Option::as_ref)
    }

    pub(crate) fn mesh_count(&self) -> usize {
        self.by_mesh.len()
    }

    /// A mesh's (clusters, nodes, BVH depth), for queue sizing.
    fn dims(&self, id: u32) -> (u32, u32, u32) {
        self.resident(id).map_or((0, 0, 0), |r| (r.rec.a[3], r.rec.a[1], r.depth))
    }

    /// The mesh table the shader reads (one record per id; free ids are zero:
    /// `node_count == 0` makes the seed step skip them).
    fn table(&self, out: &mut Vec<MeshRec>) {
        out.clear();
        out.extend(self.records.iter().map(|r| r.as_ref().map_or_else(MeshRec::zeroed, |r| r.rec)));
        if out.is_empty() {
            out.push(MeshRec::zeroed());
        }
    }

    /// Sub-allocate and upload a DAG for renderer mesh `mesh`. The copy runs as
    /// a one-time submit (the same synchronous path `upload_mesh` uses).
    pub(crate) fn install(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        mesh: usize,
        up: &ClusterMeshUpload,
    ) -> Result<u32> {
        up.validate().map_err(anyhow::Error::msg)?;
        if up.max_depth() >= BVH_MAX_DEPTH {
            bail!("BVH deeper than {BVH_MAX_DEPTH} levels");
        }
        self.install_unchecked(device, allocator, command_pool, queue, mesh, up)
    }

    /// `install` without the structural checks (robustness tests feed the
    /// shader corrupt data on purpose; everything else uses `install`).
    pub(crate) fn install_unchecked(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        mesh: usize,
        up: &ClusterMeshUpload,
    ) -> Result<u32> {
        if self.by_mesh.contains_key(&mesh) {
            bail!("mesh {mesh} already has a cluster DAG");
        }
        if self.records.len() - self.free_ids.len() >= MAX_MESH_RECORDS {
            bail!("too many cluster meshes");
        }
        let want = StagedClusterDag::want(up);
        if let Some(st) = &up.stream {
            self.ensure_tri_share(device, allocator, command_pool, queue, st)?;
        }
        let table = match &up.stream {
            Some(st) => Some(self.alloc_table(device, allocator, st.pages.len())?),
            None => None,
        };
        let starts = match self.reserve(device, allocator, command_pool, queue, want) {
            Ok(s) => s,
            Err(e) => {
                if let (Some(base), Some(st)) = (table, &up.stream) {
                    self.free_table(base, st.pages.len());
                }
                return Err(e);
            }
        };
        // One staging buffer holding the arrays back to back.
        let zero_skin = StagedClusterDag::zero_skin(up);
        let parts: [&[u8]; 7] = std::array::from_fn(|k| StagedClusterDag::part(up, &zero_skin, k));
        let total: usize = parts.iter().map(|p| p.len()).sum();
        let result = (|| -> Result<()> {
            let mut staging = Buffer::new(
                device,
                allocator,
                total as u64,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                "cluster-staging",
            )?;
            let mut at = 0;
            let mut copies = Vec::new();
            for (k, part) in parts.iter().enumerate() {
                if part.is_empty() {
                    continue;
                }
                staging.write(at, part);
                let elem = POOL_ELEM[k];
                copies.push((
                    k,
                    vk::BufferCopy::default()
                        .src_offset(at as u64)
                        .dst_offset(starts[k] as u64 * elem as u64)
                        .size(part.len() as u64),
                ));
                at += part.len();
            }
            let handles = [
                self.clusters.buffer.handle,
                self.vertices.buffer.handle,
                self.triangles.buffer.handle,
                self.nodes.buffer.handle,
                self.skin.buffer.handle,
                self.bones.buffer.handle,
                self.morphs.buffer.handle,
            ];
            let submitted = alloc::one_time_submit(device, command_pool, queue, |cb| unsafe {
                for (k, region) in &copies {
                    device.cmd_copy_buffer(cb, staging.handle, handles[*k], std::slice::from_ref(region));
                }
            });
            staging.destroy(device, allocator);
            submitted
        })();
        if let Err(e) = result {
            for k in 0..7 {
                self.pool_mut(k).ranges.free(starts[k], want[k]);
            }
            if let (Some(base), Some(st)) = (table, &up.stream) {
                self.free_table(base, st.pages.len());
            }
            return Err(e);
        }
        let id = self.register(mesh, starts, want, up.mesh_min, up.step, up.skinned, up.triangles.len() as u32, up.max_depth(), (up.leaf_triangles().min(u32::MAX as u64) as u32, up.root_vertices()), up.morph_reach().into(), true, table.is_some())?;
        if let (Some(base), Some(st)) = (table, up.stream.clone()) {
            self.register_stream(id, mesh, base, st, (starts[1], starts[2]));
        }
        Ok(id)
    }

    /// Sub-allocate the pool ranges of a DAG with `want` elements per pool (a
    /// full pool grows); the starts, or an error with everything released.
    fn reserve(&mut self, device: &ash::Device, allocator: &mut Allocator, command_pool: vk::CommandPool, queue: vk::Queue, want: [u32; 7]) -> Result<[u32; 7]> {
        if self.records.len() - self.free_ids.len() >= MAX_MESH_RECORDS {
            bail!("too many cluster meshes");
        }
        let mut got: Vec<(usize, u32)> = Vec::new();
        let mut starts = [0u32; 7];
        let mut failed = None;
        for k in 0..7 {
            let mut at = self.pool_mut(k).ranges.alloc(want[k]);
            // A full pool grows (a bigger buffer, the old contents copied) instead of leaving the mesh on the ordinary path.
            if at.is_none() && self.grow(k, want[k], device, allocator, command_pool, queue) {
                at = self.pool_mut(k).ranges.alloc(want[k]);
            }
            match at {
                Some(s) => {
                    starts[k] = s;
                    got.push((k, s));
                }
                None => {
                    failed = Some(k);
                    break;
                }
            }
        }
        if let Some(k) = failed {
            for (j, s) in got {
                self.pool_mut(j).ranges.free(s, want[j]);
            }
            bail!(
                "cluster pool full ({} needs {} entries, {} free in the largest run)",
                ["clusters", "vertices", "triangles", "nodes", "skin weights", "bone lists", "blendshape deltas"][k],
                want[k],
                self.pool_mut(k).ranges.largest_free()
            );
        }
        Ok(starts)
    }

    #[allow(clippy::too_many_arguments)]
    fn register(&mut self, mesh: usize, starts: [u32; 7], want: [u32; 7], mesh_min: [f32; 3], step: f32, skinned: bool, triangles_len: u32, depth: u32, (leaf_triangles, root_vertices): (u32, u32), reach: Box<[f32]>, ready: bool, streamed: bool) -> Result<u32> {
        let mut rec = MeshRec {
            a: [starts[3], want[3], starts[0], want[0]],
            b: [starts[1], want[1], starts[2] * 4, triangles_len],
            grid: [mesh_min[0], mesh_min[1], mesh_min[2], step],
            c: [starts[4], starts[5], starts[6], skinned as u32],
        };
        if streamed {
            // c = (table base, page count, -, flags): filled in by `register_stream`. Bit 0 is
            // "skinned" (the skin data is in the pages), bit 1 "streamed".
            rec.c = [0, 0, 0, 2 | skinned as u32];
        }
        let resident = Resident {
            rec,
            ranges: std::array::from_fn(|k| (starts[k], want[k])),
            depth,
            refs: 1,
            leaf_triangles,
            root_vertices,
            reach,
            ready,
            stream: None,
        };
        let id = match self.free_ids.pop() {
            Some(id) => {
                self.records[id as usize] = Some(resident);
                id
            }
            None => {
                self.records.push(Some(resident));
                self.records.len() as u32 - 1
            }
        };
        self.by_mesh.insert(mesh, id);
        Ok(id)
    }

    /// Install a DAG staged off-thread (`StagedClusterDag`): the pool ranges are
    /// reserved now, the copies are recorded by the next frame
    /// (`record_uploads`); no queue submit and no wait. The mesh draws from
    /// clusters from that frame on.
    pub(crate) fn install_staged(&mut self, device: &ash::Device, allocator: &mut Allocator, command_pool: vk::CommandPool, queue: vk::Queue, mesh: usize, staged: StagedClusterDag) -> Result<u32> {
        if self.by_mesh.contains_key(&mesh) {
            bail!("mesh {mesh} already has a cluster DAG");
        }
        if let Some(st) = &staged.stream
            && let Err(e) = self.ensure_tri_share(device, allocator, command_pool, queue, st)
        {
            let mut staged = staged;
            staged.destroy(device, allocator);
            return Err(e);
        }
        let table = match &staged.stream {
            Some(st) => match self.alloc_table(device, allocator, st.pages.len()) {
                Ok(b) => Some(b),
                Err(e) => {
                    let mut staged = staged;
                    staged.destroy(device, allocator);
                    return Err(e);
                }
            },
            None => None,
        };
        let starts = match self.reserve(device, allocator, command_pool, queue, staged.want) {
            Ok(s) => s,
            Err(e) => {
                let mut staged = staged;
                if let (Some(base), Some(st)) = (table, &staged.stream) {
                    self.free_table(base, st.pages.len());
                }
                staged.destroy(device, allocator);
                return Err(e);
            }
        };
        let mut copies = Vec::new();
        let mut at = 0u64;
        for k in 0..7 {
            let bytes = staged.part_bytes[k] as u64;
            if bytes > 0 {
                copies.push((k, vk::BufferCopy::default().src_offset(at).dst_offset(starts[k] as u64 * POOL_ELEM[k] as u64).size(bytes)));
                at += bytes;
            }
        }
        let id = self.register(mesh, starts, staged.want, staged.mesh_min, staged.step, staged.skinned, staged.triangles_len, staged.depth, (staged.leaf_triangles, staged.root_vertices), staged.reach, false, table.is_some())?;
        if let (Some(base), Some(st)) = (table, staged.stream) {
            self.register_stream(id, mesh, base, st, (starts[1], starts[2]));
        }
        self.uploads.push(QueuedUpload { id, staging: staged.staging, copies });
        Ok(id)
    }

    /// Grow pool `k` so that `need` more units can fit: double it (at least by
    /// `need`), within the pool's limit. The new buffer starts as a copy of the
    /// old one (a one-time submit, as installs are); the old buffer is retired
    /// until the frames that may still name it have retired. False when the
    /// pool is already at its limit or the allocation fails.
    fn grow(&mut self, k: usize, need: u32, device: &ash::Device, allocator: &mut Allocator, command_pool: vk::CommandPool, queue: vk::Queue) -> bool {
        const ELEM: [usize; 7] = [size_of::<ClusterRec>(), size_of::<ClusterVertexRec>(), 4, size_of::<BvhNodeRec>(), size_of::<ClusterSkinRec>(), 4, size_of::<MorphEntryRec>()];
        let cap = self.pool_mut(k).ranges.capacity();
        let limit = self.limits[k];
        let free_tail = self.pool_mut(k).ranges.largest_free();
        let target = cap.saturating_mul(2).max(cap.saturating_add(need.saturating_sub(free_tail))).min(limit);
        if target <= cap || need == 0 {
            return false;
        }
        let bytes = target as u64 * ELEM[k] as u64;
        let mut new = match Buffer::new(
            device,
            allocator,
            bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuOnly,
            "cluster-pool-grown",
        ) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!("cluster pool {k} cannot grow to {target} entries: {e:#}");
                return false;
            }
        };
        // Frames that recorded deferred DAG copies into the old buffer may still be
        // running: the copy below must see their writes (rare, growth only).
        if !self.spent.is_empty() {
            unsafe { let _ = device.queue_wait_idle(queue); }
        }
        let old = self.pool_mut(k).buffer.handle;
        let copy = vk::BufferCopy::default().size(cap as u64 * ELEM[k] as u64);
        let dst = new.handle;
        let submitted = alloc::one_time_submit(device, command_pool, queue, |cb| unsafe {
            device.cmd_copy_buffer(cb, old, dst, std::slice::from_ref(&copy));
        });
        if let Err(e) = submitted {
            tracing::warn!("cluster pool {k} grow copy failed: {e:#}");
            new.destroy(device, allocator);
            return false;
        }
        let p = self.pool_mut(k);
        let old_buf = std::mem::replace(&mut p.buffer, new);
        p.ranges.grow(target);
        self.retired.push((self.serial, old_buf));
        self.grown += 1;
        tracing::info!("cluster pool {k} grew from {cap} to {target} entries");
        true
    }

    fn pool_mut(&mut self, k: usize) -> &mut Pool {
        match k {
            0 => &mut self.clusters,
            1 => &mut self.vertices,
            2 => &mut self.triangles,
            3 => &mut self.nodes,
            4 => &mut self.skin,
            5 => &mut self.bones,
            _ => &mut self.morphs,
        }
    }

    /// Stop drawing `mesh` from clusters; its ranges are released once the
    /// frames in flight at `serial` have retired (no device idle).
    pub(crate) fn remove(&mut self, mesh: usize, serial: u64) {
        if let Some(id) = self.by_mesh.remove(&mesh)
            && let Some(Some(r)) = self.records.get_mut(id as usize)
        {
            r.refs = r.refs.saturating_sub(1);
            if r.refs == 0 {
                self.pending.push((serial, id));
                self.drop_stream(id, serial);
                // Its copies were never recorded: drop them (the ranges are
                // reused later) and let the staging buffer go with the frames.
                if let Some(at) = self.uploads.iter().position(|u| u.id == id) {
                    let u = self.uploads.swap_remove(at);
                    self.spent.push((serial, u.staging));
                }
            }
        }
    }

    /// Draw `mesh` from the DAG already installed for `source` (skinned copies
    /// of one model share its DAG; each keeps its own pose). False when `source`
    /// has none or `mesh` already has one.
    pub(crate) fn alias(&mut self, mesh: usize, source: usize) -> bool {
        let Some(&id) = self.by_mesh.get(&source) else { return false };
        if self.by_mesh.contains_key(&mesh) {
            return false;
        }
        if let Some(Some(r)) = self.records.get_mut(id as usize) {
            r.refs += 1;
            self.by_mesh.insert(mesh, id);
            true
        } else {
            false
        }
    }

    /// Vertices of record `id`'s coarsest cut (0 for an unknown id).
    pub(crate) fn root_vertices(&self, id: u32) -> u32 {
        self.resident(id).map_or(0, |r| r.root_vertices)
    }

    /// Whether record `id` carries skin / blendshape data.
    pub(crate) fn is_skinned(&self, id: u32) -> bool {
        self.resident(id).is_some_and(|r| r.rec.c[3] & 1 != 0)
    }

    /// Pose margin of record `id` under a pose (see `cluster_skin::pose_margin`).
    pub(crate) fn pose_margin(&self, id: u32, centre: [f32; 3], radius: f32, palette: &[Mat4], weights: &[f32]) -> f32 {
        let reach: &[f32] = self.resident(id).map_or(&[], |r| &r.reach);
        crate::cluster_skin::pose_margin(centre, radius, palette, weights, reach)
    }

    /// Destroy buffers replaced by growth once no frame in flight can name them.
    pub(crate) fn reclaim_buffers(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        let serial = self.serial;
        let mut i = 0;
        while i < self.retired.len() {
            if (self.retired[i].0 + FRAMES_IN_FLIGHT as u64) < serial {
                let (_, mut b) = self.retired.swap_remove(i);
                b.destroy(device, allocator);
            } else {
                i += 1;
            }
        }
        let mut i = 0;
        while i < self.spent.len() {
            if (self.spent[i].0 + FRAMES_IN_FLIGHT as u64) < serial {
                let (_, mut b) = self.spent.swap_remove(i);
                b.destroy(device, allocator);
            } else {
                i += 1;
            }
        }
    }

    /// Retired (replaced by growth, not yet destroyed) buffers.
    pub(crate) fn retired_buffers(&self) -> usize {
        self.retired.len() + self.spent.len()
    }

    /// DAGs installed but not yet recorded into a frame.
    pub(crate) fn queued_uploads(&self) -> usize {
        self.uploads.len()
    }

    /// Record the queued DAG copies into `cb` (the frame's command buffer, before
    /// any cluster work) and make them visible to the cull / skin / draw
    /// shaders. The records become drawable (`record_of`) from this frame on.
    pub(crate) fn record_uploads(&mut self, device: &ash::Device, cb: vk::CommandBuffer, serial: u64) {
        let streaming = self.record_stream_pending();
        if self.uploads.is_empty() && !streaming {
            return;
        }
        if streaming {
            self.record_stream(device, cb, serial);
        }
        let handles = [
            self.clusters.buffer.handle,
            self.vertices.buffer.handle,
            self.triangles.buffer.handle,
            self.nodes.buffer.handle,
            self.skin.buffer.handle,
            self.bones.buffer.handle,
            self.morphs.buffer.handle,
        ];
        for u in std::mem::take(&mut self.uploads) {
            for (k, region) in &u.copies {
                unsafe { device.cmd_copy_buffer(cb, u.staging.handle, handles[*k], std::slice::from_ref(region)) };
            }
            if let Some(Some(r)) = self.records.get_mut(u.id as usize) {
                r.ready = true;
            }
            self.spent.push((serial, u.staging));
        }
        let b = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)];
        unsafe { device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&b)) };
    }

    /// Release the ranges of removals that no frame in flight can still read.
    pub(crate) fn reclaim(&mut self, serial: u64) {
        self.serial = serial;
        self.reclaim_stream(serial);
        let mut i = 0;
        while i < self.pending.len() {
            let (at, id) = self.pending[i];
            if at + FRAMES_IN_FLIGHT as u64 <= serial {
                self.pending.swap_remove(i);
                if let Some(mut r) = self.records[id as usize].take() {
                    for (k, (s, n)) in r.ranges.iter().enumerate() {
                        self.pool_mut(k).ranges.free(*s, *n);
                    }
                    if let Some(st) = r.stream.take() {
                        self.release_stream(st);
                    }
                    self.free_ids.push(id);
                }
            } else {
                i += 1;
            }
        }
    }

    /// (used, capacity) per pool: clusters, vertices, triangle words, nodes.
    pub(crate) fn usage(&self) -> [(u32, u32); 4] {
        [&self.clusters, &self.vertices, &self.triangles, &self.nodes].map(|p| (p.ranges.used(), p.ranges.capacity()))
    }
}

// ---------------------------------------------------------------------------
// Descriptor set layout and pipelines

/// The cluster descriptor set layout (set 0 of the cull, set 2 of the clustered
/// graphics pipelines). Created by `PipelineCache`.
pub(crate) fn create_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    let mut bindings: Vec<vk::DescriptorSetLayoutBinding> = STORAGE_BINDINGS
        .iter()
        .map(|&b| {
            let mut stages = vk::ShaderStageFlags::COMPUTE;
            if VERTEX_BINDINGS.contains(&b) {
                stages |= vk::ShaderStageFlags::VERTEX;
            }
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(stages)
        })
        .collect();
    for b in [HIZ_BINDING, HIZ_R_BINDING] {
        bindings.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        );
    }
    Ok(unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)? })
}

/// The compute pipelines of the cluster pass.
pub struct ClusterPipes {
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) cull: vk::Pipeline,
    pub(crate) decode: vk::Pipeline,
}

impl ClusterPipes {
    pub(crate) fn new(device: &ash::Device, set_layout: vk::DescriptorSetLayout) -> Result<Self> {
        let layouts = [set_layout];
        let ranges = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(128)];
        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&ranges),
                None,
            )?
        };
        let cull = crate::gpu_cull::compute_pipeline(device, layout, CULL_SPV)?;
        let decode = crate::gpu_cull::compute_pipeline(device, layout, DECODE_SPV)?;
        Ok(Self { layout, cull, decode })
    }

    pub(crate) fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.cull, None);
            device.destroy_pipeline(self.decode, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

// ---------------------------------------------------------------------------
// Per-frame buffers and the recording

/// Statistics of one cull read back from a retired frame slot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClusterCounts {
    pub instances: u32,
    pub instances_culled: u32,
    pub nodes: u32,
    pub tested: u32,
    pub lod_rejected: u32,
    pub frustum: u32,
    pub cone: u32,
    pub hiz: u32,
    pub visible: u32,
    pub triangles: u32,
    pub late_recovered: u32,
    pub flags: u32,
    /// Shadow views (cluster shadows): instances seeded, clusters and triangles of their cuts.
    pub shadow_instances: u32,
    pub shadow_visible: u32,
    pub shadow_triangles: u32,
    /// Streaming: tested clusters on a page that is not resident, and clusters
    /// drawn in place of children whose page is missing.
    pub stream_skipped: u32,
    pub stream_standin: u32,
}

impl ClusterCounts {
    pub fn from_words(w: &[u32]) -> Self {
        let g = |i: usize| w.get(i).copied().unwrap_or(0);
        Self {
            instances: g(counter::INSTANCES),
            instances_culled: g(counter::INSTANCES_CULLED),
            nodes: g(counter::NODES),
            tested: g(counter::TESTED),
            lod_rejected: g(counter::LOD),
            frustum: g(counter::FRUSTUM),
            cone: g(counter::CONE),
            hiz: g(counter::HIZ),
            visible: g(counter::VISIBLE),
            triangles: g(counter::TRIANGLES),
            late_recovered: g(counter::LATE_RECOVERED),
            flags: g(counter::FLAGS),
            shadow_instances: g(counter::SHADOW_INSTANCES),
            shadow_visible: g(counter::SHADOW_VISIBLE),
            shadow_triangles: g(counter::SHADOW_TRIANGLES),
            stream_skipped: g(counter::STREAM_SKIPPED),
            stream_standin: g(counter::STREAM_STANDIN),
        }
    }

    /// Whether any queue / list was full or an index was bad.
    pub fn overflowed(&self) -> bool {
        self.flags != 0
    }
}

/// Sizes and bases of one frame's lists, derived from the instances and batches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CullPlan {
    pub instances: u32,
    pub batches: u32,
    /// Entries per node queue.
    pub node_cap: u32,
    /// Entries of the occluded list.
    pub occ_cap: u32,
    /// Entries of the early (and the late) half of the visible list.
    pub visible_half: u32,
    /// Node dispatches (deepest BVH + 1).
    pub levels: u32,
}

/// Sizes the queues from what the frame draws. `per_instance` = (clusters,
/// nodes, depth) of each instance's mesh; `batch_bounds` = Σ clusters of each
/// batch's instances. Returns the plan and the (base, capacity) of each
/// batch's early region (late regions follow at `visible_half`).
pub fn plan_cull(per_instance: &[(u32, u32, u32)], batch_bounds: &[u64]) -> (CullPlan, Vec<(u32, u32)>) {
    let mut regions = Vec::with_capacity(batch_bounds.len());
    let plan = plan_cull_into(per_instance, batch_bounds, &mut regions);
    (plan, regions)
}

/// [`plan_cull`] into a caller-owned (reused) region list.
pub fn plan_cull_into(per_instance: &[(u32, u32, u32)], batch_bounds: &[u64], regions: &mut Vec<(u32, u32)>) -> CullPlan {
    regions.clear();
    let nodes: u64 = per_instance.iter().map(|d| d.1 as u64).sum();
    let levels = per_instance.iter().map(|d| d.2).max().map_or(1, |d| d + 1);
    let node_cap = (nodes as usize).clamp(MIN_ENTRIES, MAX_QUEUE_ENTRIES) as u32;
    let mut base = 0u64;
    for &bound in batch_bounds {
        // A region is as large as the batch can possibly produce (every
        // cluster of every instance once), within the list's budget.
        let room = MAX_VISIBLE_ENTRIES as u64 - base.min(MAX_VISIBLE_ENTRIES as u64);
        let cap = bound.min(room);
        regions.push((base as u32, cap as u32));
        base += cap;
    }
    let half = (base as usize).max(MIN_ENTRIES) as u32;
    CullPlan {
        instances: per_instance.len() as u32,
        batches: batch_bounds.len() as u32,
        node_cap,
        occ_cap: half,
        visible_half: half,
        levels,
    }
}

struct ClusterFrame {
    instances: GrowBuffer,
    batches: GrowBuffer,
    views: GrowBuffer,
    mesh_table: GrowBuffer,
    queue_a: GrowBuffer,
    queue_b: GrowBuffer,
    visible: GrowBuffer,
    occluded: GrowBuffer,
    cmds: GrowBuffer,
    batch_counts: GrowBuffer,
    /// Posed vertices of the visible skinned clusters, and each visible entry's base in it.
    skin_pool: GrowBuffer,
    skin_slots: GrowBuffer,
    counters: Buffer,
    dispatch: Buffer,
    readback: Buffer,
    /// Page requests and touched flags copied back (streaming only; created on first use).
    stream_readback: Option<Buffer>,
    set: vk::DescriptorSet,
    /// A cull was recorded into this slot: read the counters when it retires.
    recorded: bool,
    /// The streaming I/O was recorded into this slot too.
    stream_recorded: bool,
}

impl ClusterFrame {
    fn new(device: &ash::Device, allocator: &mut Allocator, set: vk::DescriptorSet) -> Result<Self> {
        let host = |name| GrowBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, name);
        let gpu = |name| GrowBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuOnly, name);
        Ok(Self {
            instances: host("cluster-instances"),
            batches: host("cluster-batches"),
            views: GrowBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::CpuToGpu, "cluster-views"),
            mesh_table: host("cluster-mesh-table"),
            queue_a: gpu("cluster-queue-a"),
            queue_b: gpu("cluster-queue-b"),
            visible: gpu("cluster-visible"),
            occluded: gpu("cluster-occluded"),
            cmds: GrowBuffer::new(
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
                MemoryLocation::GpuOnly,
                "cluster-indirect",
            ),
            batch_counts: GrowBuffer::new(
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
                "cluster-batch-counts",
            ),
            skin_pool: GrowBuffer::new(
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::GpuOnly,
                "cluster-skin-pool",
            ),
            skin_slots: GrowBuffer::new(
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::GpuOnly,
                "cluster-skin-slots",
            ),
            counters: Buffer::new(
                device,
                allocator,
                COUNTER_BYTES,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::GpuOnly,
                "cluster-counters",
            )?,
            dispatch: Buffer::new(
                device,
                allocator,
                32,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
                "cluster-dispatch",
            )?,
            readback: Buffer::new(
                device,
                allocator,
                COUNTER_BYTES,
                vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
                "cluster-readback",
            )?,
            stream_readback: None,
            set,
            recorded: false,
            stream_recorded: false,
        })
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for b in [
            &mut self.instances,
            &mut self.batches,
            &mut self.views,
            &mut self.mesh_table,
            &mut self.queue_a,
            &mut self.queue_b,
            &mut self.visible,
            &mut self.occluded,
            &mut self.cmds,
            &mut self.batch_counts,
            &mut self.skin_pool,
            &mut self.skin_slots,
        ] {
            b.destroy(device, allocator);
        }
        self.counters.destroy(device, allocator);
        self.dispatch.destroy(device, allocator);
        self.readback.destroy(device, allocator);
        if let Some(b) = self.stream_readback.as_mut() {
            b.destroy(device, allocator);
        }
    }

    /// Size the buffers for `plan`, write the CPU lists and point the set at
    /// everything.
    #[allow(clippy::too_many_arguments)]
    fn upload(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        pools: [vk::Buffer; 4],
        stream: (vk::Buffer, vk::Buffer),
        skin_vertices: u32,
        plan: &CullPlan,
        instances: &[InstRec],
        batches: &[BatchRec],
        views: &[GpuView],
        table: &[MeshRec],
        hiz: (vk::Sampler, vk::ImageView),
        hiz_right: vk::ImageView,
    ) -> Result<()> {
        self.instances.ensure(device, allocator, std::mem::size_of_val(instances))?;
        self.batches.ensure(device, allocator, std::mem::size_of_val(batches))?;
        self.views.ensure(device, allocator, std::mem::size_of_val(views))?;
        self.mesh_table.ensure(device, allocator, std::mem::size_of_val(table))?;
        self.queue_a.ensure(device, allocator, plan.node_cap as usize * 8)?;
        self.queue_b.ensure(device, allocator, plan.node_cap as usize * 8)?;
        self.visible.ensure(device, allocator, plan.visible_half as usize * 16)?;
        self.occluded.ensure(device, allocator, plan.occ_cap as usize * 8)?;
        self.cmds.ensure(device, allocator, plan.batches as usize * 32)?;
        self.batch_counts.ensure(device, allocator, plan.batches as usize * 8)?;
        self.skin_pool.ensure(device, allocator, skin_vertices.max(1) as usize * crate::cluster_skin::SKIN_VERT_BYTES)?;
        self.skin_slots.ensure(device, allocator, plan.visible_half as usize * 8)?;
        self.instances.write(bytemuck::cast_slice(instances));
        self.batches.write(bytemuck::cast_slice(batches));
        self.views.write(bytemuck::cast_slice(views));
        self.mesh_table.write(bytemuck::cast_slice(table));
        // Binding order = STORAGE_BINDINGS.
        let buffers = [
            pools[0],
            pools[1],
            pools[2],
            pools[3],
            self.mesh_table.handle(),
            self.instances.handle(),
            self.batches.handle(),
            self.views.handle(),
            self.counters.handle,
            self.queue_a.handle(),
            self.queue_b.handle(),
            self.visible.handle(),
            self.cmds.handle(),
            self.occluded.handle(),
            self.batch_counts.handle(),
            self.dispatch.handle,
            self.skin_pool.handle(),
            self.skin_slots.handle(),
            stream.0,
            stream.1,
        ];
        write_set(device, self.set, &buffers, hiz, hiz_right);
        Ok(())
    }
}

/// Point every binding of a cluster set at its buffer.
pub(crate) fn write_set(device: &ash::Device, set: vk::DescriptorSet, buffers: &[vk::Buffer; 20], hiz: (vk::Sampler, vk::ImageView), hiz_right: vk::ImageView) {
    let infos: Vec<[vk::DescriptorBufferInfo; 1]> =
        buffers.iter().map(|&b| [vk::DescriptorBufferInfo::default().buffer(b).range(vk::WHOLE_SIZE)]).collect();
    let image = [vk::DescriptorImageInfo::default().sampler(hiz.0).image_view(hiz.1).image_layout(vk::ImageLayout::GENERAL)];
    let image_r = [vk::DescriptorImageInfo::default().sampler(hiz.0).image_view(hiz_right).image_layout(vk::ImageLayout::GENERAL)];
    let mut writes: Vec<vk::WriteDescriptorSet> = infos
        .iter()
        .zip(STORAGE_BINDINGS)
        .map(|(info, binding)| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(info)
        })
        .collect();
    writes.push(
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(HIZ_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image),
    );
    writes.push(
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(HIZ_R_BINDING)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_r),
    );
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

fn barrier(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    src: (vk::PipelineStageFlags2, vk::AccessFlags2),
    dst: (vk::PipelineStageFlags2, vk::AccessFlags2),
) {
    let b = [vk::MemoryBarrier2::default()
        .src_stage_mask(src.0)
        .src_access_mask(src.1)
        .dst_stage_mask(dst.0)
        .dst_access_mask(dst.1)];
    unsafe { device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&b)) };
}

const CS: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::COMPUTE_SHADER;
const WRITE: vk::AccessFlags2 = vk::AccessFlags2::SHADER_STORAGE_WRITE;
const RW: vk::AccessFlags2 = vk::AccessFlags2::from_raw(
    vk::AccessFlags2::SHADER_STORAGE_READ.as_raw() | vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw(),
);

/// Compute -> compute, and the indirect-dispatch argument read.
fn step_barrier(device: &ash::Device, cb: vk::CommandBuffer) {
    barrier(
        device,
        cb,
        (CS, WRITE),
        (CS | vk::PipelineStageFlags2::DRAW_INDIRECT, RW | vk::AccessFlags2::INDIRECT_COMMAND_READ),
    );
}

fn dispatch_mode(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    pipes: &ClusterPipes,
    push: &CullPush,
    groups: u32,
) {
    unsafe {
        device.cmd_push_constants(cb, pipes.layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(push));
        device.cmd_dispatch(cb, groups, 1, 1);
    }
}

fn dispatch_mode_indirect(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    pipes: &ClusterPipes,
    push: &CullPush,
    args: vk::Buffer,
    offset: u64,
) {
    unsafe {
        device.cmd_push_constants(cb, pipes.layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(push));
        device.cmd_dispatch_indirect(cb, args, offset);
    }
}

/// Buffers a cull records against (a frame's, or the headless test's).
pub(crate) struct CullTargets {
    pub set: vk::DescriptorSet,
    pub counters: vk::Buffer,
    pub dispatch: vk::Buffer,
    pub batch_counts: vk::Buffer,
    pub batch_count_bytes: u64,
    pub readback: Option<vk::Buffer>,
    /// The skin stage to run after each cull half (skinned instances present).
    pub skin: Option<(crate::cluster_skin::SkinTargets, vk::Pipeline, vk::PipelineLayout)>,
    /// GPU timestamp pool and the (early, late) skin zones' first query, when
    /// zone timing is on: the skin stage is bracketed with a begin/end pair.
    pub skin_stamps: Option<(vk::QueryPool, [u32; 2])>,
    /// Page requests and touched flags (streamed meshes present).
    pub stream: Option<StreamTargets>,
}

/// Buffers of the streaming I/O of one cull.
#[derive(Clone, Copy)]
pub(crate) struct StreamTargets {
    pub io: vk::Buffer,
    pub readback: vk::Buffer,
    /// Touched flags to clear and copy back (the page table's high-water mark).
    pub touch_words: u32,
    /// Capacity of the page table the I/O buffer was laid out for.
    pub max_pages: u32,
}

/// Bytes of the per-frame stream readback: the request list, then the touched flags.
fn stream_readback_bytes(max_pages: usize) -> u64 {
    STREAM_REQ_BYTES + (max_pages * 4) as u64
}

/// Record the early pass: clear, seed, walk the BVH level by level, finalize
/// the indirect commands. `hiz_ready`: the pyramid is sampleable (its layout
/// transition and the previous frame's build are ordered before this).
pub(crate) fn record_early(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    pipes: &ClusterPipes,
    t: &CullTargets,
    plan: &CullPlan,
) {
    let mut push = CullPush::zeroed();
    push.node_cap = plan.node_cap;
    push.occ_cap = plan.occ_cap;
    push.batch_count = plan.batches;
    push.late_vp = Mat4::IDENTITY.to_cols_array_2d();
    push.stream_pages = t.stream.as_ref().map_or(0, |st| st.max_pages);
    unsafe {
        device.cmd_fill_buffer(cb, t.counters, 0, COUNTER_BYTES, 0);
        device.cmd_fill_buffer(cb, t.batch_counts, 0, t.batch_count_bytes.max(4), 0);
        device.cmd_fill_buffer(cb, t.dispatch, 0, 32, 0);
        if let Some(st) = &t.stream {
            // The previous frame's copy-out and shader writes of the I/O buffer come first.
            barrier(
                device,
                cb,
                (vk::PipelineStageFlags2::TRANSFER | CS, vk::AccessFlags2::TRANSFER_READ | RW),
                (vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE),
            );
            device.cmd_fill_buffer(cb, st.io, 0, (STREAM_REQ_BASE * 4) as u64, 0);
            device.cmd_fill_buffer(cb, st.io, (stream_touch_base(st.max_pages as usize) * 4) as u64, (st.touch_words.max(1) * 4) as u64, 0);
        }
        // Also orders the previous frame's pyramid build before the sampled read.
        barrier(
            device,
            cb,
            (vk::PipelineStageFlags2::TRANSFER | CS, vk::AccessFlags2::TRANSFER_WRITE | WRITE),
            (CS, RW | vk::AccessFlags2::SHADER_SAMPLED_READ),
        );
        device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipes.cull);
        device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, pipes.layout, 0, &[t.set], &[]);
    }
    push.mode = mode::SEED;
    push.count = plan.instances;
    dispatch_mode(device, cb, pipes, &push, plan.instances.div_ceil(64));
    step_barrier(device, cb);
    for level in 0..plan.levels {
        push.parity = level & 1;
        push.mode = mode::ADVANCE;
        dispatch_mode(device, cb, pipes, &push, 1);
        step_barrier(device, cb);
        push.mode = mode::NODES;
        dispatch_mode_indirect(device, cb, pipes, &push, t.dispatch, 0);
        step_barrier(device, cb);
    }
    push.mode = mode::FINALIZE;
    dispatch_mode(device, cb, pipes, &push, plan.batches.div_ceil(64));
    settle_requests(device, cb, pipes, &mut push, t);
    skin_half(device, cb, t, false);
    finish(device, cb, t);
}

/// Settle the priorities of the page requests the pass appended (streamed meshes present).
fn settle_requests(device: &ash::Device, cb: vk::CommandBuffer, pipes: &ClusterPipes, push: &mut CullPush, t: &CullTargets) {
    if t.stream.is_none() {
        return;
    }
    step_barrier(device, cb);
    push.mode = mode::STREAM;
    dispatch_mode(device, cb, pipes, push, (STREAM_REQ_CAP / 64) as u32);
}

/// Record the late pass (outside rendering, after this frame's pyramid was
/// built): re-test the clusters the early pass found occluded.
pub(crate) fn record_late(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    pipes: &ClusterPipes,
    t: &CullTargets,
    plan: &CullPlan,
    late_vp: Mat4,
) {
    let mut push = CullPush::zeroed();
    push.node_cap = plan.node_cap;
    push.occ_cap = plan.occ_cap;
    push.batch_count = plan.batches;
    push.late = 1;
    push.stream_pages = t.stream.as_ref().map_or(0, |st| st.max_pages);
    push.late_vp = late_vp.to_cols_array_2d();
    barrier(
        device,
        cb,
        (vk::PipelineStageFlags2::TRANSFER | CS, vk::AccessFlags2::TRANSFER_WRITE | WRITE),
        (CS, RW | vk::AccessFlags2::SHADER_SAMPLED_READ),
    );
    unsafe {
        device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipes.cull);
        device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, pipes.layout, 0, &[t.set], &[]);
    }
    push.mode = mode::ADVANCE_OCC;
    dispatch_mode(device, cb, pipes, &push, 1);
    step_barrier(device, cb);
    push.mode = mode::LATE;
    dispatch_mode_indirect(device, cb, pipes, &push, t.dispatch, 16);
    step_barrier(device, cb);
    push.mode = mode::FINALIZE;
    dispatch_mode(device, cb, pipes, &push, plan.batches.div_ceil(64));
    settle_requests(device, cb, pipes, &mut push, t);
    skin_half(device, cb, t, true);
    finish(device, cb, t);
}

/// Pose the visible skinned clusters of the cull half that just finished.
fn skin_half(device: &ash::Device, cb: vk::CommandBuffer, t: &CullTargets, late: bool) {
    if let Some((targets, pipeline, layout)) = &t.skin {
        let stamp = |q: u32| {
            if let Some((pool, base)) = t.skin_stamps {
                unsafe { device.cmd_write_timestamp2(cb, vk::PipelineStageFlags2::ALL_COMMANDS, pool, base[late as usize] + q) };
            }
        };
        stamp(0);
        crate::cluster_skin::record_skin(device, cb, *pipeline, *layout, targets, late);
        stamp(1);
    }
}

/// Make the commands and visible list readable by the draw, and copy the
/// counters out for the CPU.
fn finish(device: &ash::Device, cb: vk::CommandBuffer, t: &CullTargets) {
    barrier(
        device,
        cb,
        (CS, WRITE),
        (
            vk::PipelineStageFlags2::DRAW_INDIRECT | vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::INDIRECT_COMMAND_READ | vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::TRANSFER_READ,
        ),
    );
    if let Some(readback) = t.readback {
        unsafe {
            device.cmd_copy_buffer(cb, t.counters, readback, &[vk::BufferCopy::default().size(COUNTER_BYTES)]);
            if let Some(st) = &t.stream {
                device.cmd_copy_buffer(
                    cb,
                    st.io,
                    st.readback,
                    &[
                        vk::BufferCopy::default().src_offset(0).dst_offset(0).size(STREAM_REQ_BYTES),
                        vk::BufferCopy::default().src_offset((stream_touch_base(st.max_pages as usize) * 4) as u64).dst_offset(STREAM_REQ_BYTES).size((st.touch_words.max(1) * 4) as u64),
                    ],
                );
            }
        }
        barrier(
            device,
            cb,
            (vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE),
            (vk::PipelineStageFlags2::HOST, vk::AccessFlags2::HOST_READ),
        );
    }
}

// ---------------------------------------------------------------------------
// The system

/// One material batch of cluster instances.
pub(crate) struct ClusterBatch {
    pub material: usize,
    pub push: PushData,
    pub mask: [u32; 4],
    pub bounds: Option<(Vec3, Vec3)>,
    /// Σ clusters of the batch's instances (the region capacity bound).
    pub bound: u64,
    pub early_base: u32,
    /// A shadow layer's batch (cluster shadows): drawn by that layer's pass, not the camera's.
    pub shadow_layer: Option<usize>,
}

/// What the budget did for the last uploaded frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct BudgetStats {
    /// Estimated triangles of all instances at the global scale.
    pub estimated: f32,
    /// Instances the fair share coarsened.
    pub clamped: u32,
    /// Largest per-instance scale.
    pub max_scale: f32,
}

/// The cull's view of a shadow map: the light's frustum and the texel math of
/// its cut. `dim` = shadow map size, `px` = error target in texels. Texels per
/// world unit come from the projection's y scale (`|row 1|` of the combined
/// matrix: the same for an orthographic cascade and a perspective spot / cube
/// face); a perspective light divides it by the distance from the light, an
/// orthographic one has no distance (`lod.x` = 1 selects the constant-distance
/// rule in `cluster_cull.comp`). No near-plane test (casters between the light
/// and the frustum still cast), no Hi-Z, no distance limit.
pub(crate) fn shadow_cull_view(view_proj: Mat4, dim: u32, px: f32) -> GpuView {
    let mut v = GpuView::new(view_proj, Vec3::ZERO, false);
    let (r0, r1, r3) = (view_proj.row(0), view_proj.row(1), view_proj.row(3));
    v.lod[1] = 0.5 * dim as f32 * r1.truncate().length();
    v.lod[3] = px.max(1e-3);
    if r3.truncate().length() < 1e-6 {
        v.lod[0] = 1.0;
    } else {
        // The light's position: clip x = y = w = 0.
        let m = glam::Mat3::from_cols(r0.truncate(), r1.truncate(), r3.truncate()).transpose();
        if m.determinant().abs() > 1e-12 {
            let apex = m.inverse() * -glam::Vec3::new(r0.w, r1.w, r3.w);
            v.cam = apex.extend(0.0).to_array();
        }
    }
    v
}

/// VR Hi-Z for the next stereo cull: the right eye's pyramid and its view record.
pub(crate) struct StereoHiz {
    pub right: vk::ImageView,
    pub partner: GpuView,
}

/// The cull's view of one VR eye drawn on its own (the per-eye, non-multiview
/// pass): an ordinary perspective camera. `height_px` = the eye image height.
pub(crate) fn eye_cull_view(view_proj: Mat4, cam_pos: Vec3, height_px: f32, target_px: f32) -> GpuView {
    let mut v = GpuView::new(view_proj, cam_pos, true);
    // Pixels per world unit at distance 1: the projection's y scale, |row 1| of proj * view (the view is rigid).
    v.lod[1] = 0.5 * height_px * view_proj.row(1).truncate().length();
    v.lod[3] = target_px.max(0.01);
    v
}

/// The cull's view of a stereo (multiview) frame: LOD judged from the centre
/// eye with the wider eye's projection scale and the pixel target tightened by
/// the IPD slack at 1 m (so the target also holds in each eye), the frustum a
/// conservative union of both eyes'. See `cluster_vr`.
pub(crate) fn stereo_cull_view(sv: &crate::cluster_vr::StereoView, height_px: f32, target_px: f32) -> GpuView {
    let centre = crate::cluster_vr::centre_eye(sv).inverse().w_axis.truncate();
    let union = crate::cluster_vr::UnionFrustum::from_eyes(sv);
    let mut v = GpuView::new(Mat4::IDENTITY, centre, true);
    v.planes = union.planes.map(|p| p.to_array());
    v.lod[1] = crate::cluster_vr::pixels_per_unit(sv, height_px);
    v.lod[3] = (target_px / (1.0 + crate::cluster_vr::lod_slack(crate::cluster_vr::eye_separation(sv), 1.0))).max(0.01);
    v
}

/// What a recorded frame offers the draw calls.
#[derive(Clone, Copy)]
struct Drawable {
    cmds: vk::Buffer,
    set: vk::DescriptorSet,
    batches: u32,
    visible_half: u32,
    /// The late pass ran (its commands are valid).
    late: bool,
}

pub(crate) struct ClusterSystem {
    pub(crate) store: ClusterStore,
    pub(crate) pipes: ClusterPipes,
    /// The skinned-cluster stage (GEO-5); idle when no skinned instance is drawn.
    pub(crate) skin: crate::cluster_skin::SkinStage,
    /// Vertices the frame's skinned instances could need at most (pool sizing).
    /// Pool cap each frame slot's skin pool was recorded with (the retired frame's cap for `pressure`).
    skin_slot_cap: [u32; CLUSTER_SETS],
    /// Which skinned instances may draw from clusters while the pool is under pressure.
    pub(crate) admission: SkinAdmission,
    /// Pool capacity (vertices) the last upload sized.
    pub(crate) skin_pool_cap: u32,
    /// Triangle budget (GEO-5): the controller on the GPU frame time, the fair
    /// share per instance, and what they did last frame.
    pub(crate) budget: crate::cluster_budget::BudgetController,
    /// Global error scale applied this frame (the controller's output).
    pub(crate) global_scale: f32,
    pub(crate) max_triangles: u32,
    classes: HashMap<usize, crate::cluster_budget::ShareClass>,
    inst_class: Vec<crate::cluster_budget::ShareClass>,
    est_scratch: Vec<f32>,
    weight_scratch: Vec<f32>,
    share_scratch: Vec<f32>,
    pub(crate) budget_stats: BudgetStats,
    pool: vk::DescriptorPool,
    frames: Vec<ClusterFrame>,
    /// This frame's CPU lists.
    pub(crate) instances: Vec<InstRec>,
    /// `FrameInput::draws` index of each instance (to hand them back on failure).
    draw_ids: Vec<usize>,
    pub(crate) batches: Vec<ClusterBatch>,
    pub(crate) key_map: HashMap<crate::gpu_cull::SceneBatchKey, usize>,
    table: Vec<MeshRec>,
    per_instance: Vec<(u32, u32, u32)>,
    plan: CullPlan,
    batch_recs: Vec<BatchRec>,
    bounds_scratch: Vec<u64>,
    regions_scratch: Vec<(u32, u32)>,
    drawable: Option<Drawable>,
    /// Shadow views of this frame (cluster shadows), after the main camera's: (layer, `views` index).
    shadow_views: Vec<GpuView>,
    /// Next upload: the right eye's pyramid view and view record (VR Hi-Z).
    pub(crate) stereo_hiz: Option<StereoHiz>,
    /// Index of the right eye's view in the last upload's list.
    stereo_partner: Option<u32>,
    /// The VR eye-render slot that recorded the last cull (its counters stay readable).
    pub(crate) xr_last_slot: Option<usize>,
    pub(crate) shadow_layers: Vec<(usize, u32)>,
    /// Per shadow layer, the draws (indices of `FrameInput::draws`, ascending)
    /// whose shadow comes from clusters this frame; the vectors are reused.
    shadow_handled: Vec<(usize, Vec<u32>)>,
    shadow_handled_used: usize,
    /// Draws drawn from clusters in the camera pass this frame.
    main_clustered: Vec<bool>,
    /// Skinned meshes whose ordinary skin dispatch was skipped last frame (stats).
    pub(crate) skin_skipped: u32,
    skip_scratch: Vec<usize>,
    need_scratch: Vec<bool>,
    views_scratch: Vec<GpuView>,
    /// Counters of the last retired frame.
    pub(crate) counts: ClusterCounts,
    /// Frames (serial) before which the path stays off after an overflow.
    pub(crate) disabled_until: u64,
    pub(crate) overflow_frames: u32,
    pub(crate) warned: bool,
    warned_skin: bool,
    /// Test knob: clamp the queue and list capacities to force an overflow.
    pub(crate) cap_limit: Option<u32>,
    /// Instances / batches submitted by the last recorded frame.
    pub(crate) last_instances: u32,
    pub(crate) last_batches: u32,
}

impl ClusterSystem {
    pub(crate) fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        set_layout: vk::DescriptorSetLayout,
        sizes: PoolSizes,
    ) -> Result<Self> {
        let store = ClusterStore::new(device, allocator, sizes)?;
        let pipes = ClusterPipes::new(device, set_layout)?;
        let skin = crate::cluster_skin::SkinStage::new(device, allocator, set_layout)?;
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count((CLUSTER_SETS * STORAGE_BINDINGS.len()) as u32),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(2 * CLUSTER_SETS as u32),
        ];
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(CLUSTER_SETS as u32).pool_sizes(&pool_sizes),
                None,
            )?
        };
        let layouts = vec![set_layout; CLUSTER_SETS];
        let sets = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))? };
        let frames = sets.into_iter().map(|s| ClusterFrame::new(device, allocator, s)).collect::<Result<Vec<_>>>()?;
        Ok(Self {
            store,
            pipes,
            skin,
            skin_slot_cap: [0; CLUSTER_SETS],
            admission: SkinAdmission::default(),
            skin_pool_cap: 0,
            budget: crate::cluster_budget::BudgetController::new(Default::default()),
            global_scale: 1.0,
            max_triangles: 0,
            classes: HashMap::new(),
            inst_class: Vec::new(),
            est_scratch: Vec::new(),
            weight_scratch: Vec::new(),
            share_scratch: Vec::new(),
            budget_stats: BudgetStats::default(),
            pool,
            frames,
            instances: Vec::new(),
            draw_ids: Vec::new(),
            batches: Vec::new(),
            key_map: HashMap::new(),
            table: Vec::new(),
            per_instance: Vec::new(),
            plan: CullPlan::default(),
            batch_recs: Vec::new(),
            bounds_scratch: Vec::new(),
            regions_scratch: Vec::new(),
            drawable: None,
            shadow_views: Vec::new(),
            stereo_hiz: None,
            stereo_partner: None,
            xr_last_slot: None,
            shadow_layers: Vec::new(),
            shadow_handled: Vec::new(),
            shadow_handled_used: 0,
            main_clustered: Vec::new(),
            skin_skipped: 0,
            skip_scratch: Vec::new(),
            need_scratch: Vec::new(),
            views_scratch: Vec::new(),
            counts: ClusterCounts::default(),
            disabled_until: 0,
            overflow_frames: 0,
            warned: false,
            warned_skin: false,
            cap_limit: None,
            last_instances: 0,
            last_batches: 0,
        })
    }

    pub(crate) fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for f in &mut self.frames {
            f.destroy(device, allocator);
        }
        self.store.destroy(device, allocator);
        self.skin.destroy(device, allocator);
        self.pipes.destroy(device);
        unsafe { device.destroy_descriptor_pool(self.pool, None) };
    }

    pub(crate) fn begin_build(&mut self) {
        self.draw_ids.clear();
        self.instances.clear();
        self.batches.clear();
        self.key_map.clear();
        self.drawable = None;
        self.shadow_views.clear();
        self.shadow_layers.clear();
        self.forget_shadow_handled();
        self.main_clustered.clear();
        self.skin.begin_build();
        self.admission.begin_frame();
        self.inst_class.clear();
    }

    /// `add_instance` for a skinned / blendshape mesh: queues its pose for the
    /// skin stage and grows the cull's bounds by the pose margin. The cone test
    /// is off for it (`eye.w = 0`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_skinned_instance(
        &mut self,
        draw: usize,
        id: u32,
        batch: u32,
        d: &DrawCmd,
        color: [f32; 4],
        param: [f32; 4],
        pose: &crate::cluster_skin::SkinPose,
    ) {
        self.add_instance(draw, id, batch, d, color, param, Vec3::ZERO);
        let index = self.skin.add(pose);
        let rec = self.instances.last_mut().expect("add_instance pushed a record");
        rec.mesh_batch[2] = index;
        rec.mesh_batch[3] = pose.margin.max(0.0).to_bits();
        rec.eye[3] = 0.0;
        self.admission.note_root_vertices(d.mesh.0, self.store.root_vertices(id));
    }

    /// Note that draw `draw` is drawn from clusters in the camera pass.
    pub(crate) fn mark_main(&mut self, draw: usize) {
        if self.main_clustered.len() <= draw {
            self.main_clustered.resize(draw + 1, false);
        }
        self.main_clustered[draw] = true;
    }

    fn is_main(&self, draw: usize) -> bool {
        self.main_clustered.get(draw).copied().unwrap_or(false)
    }

    fn forget_shadow_handled(&mut self) {
        for (_, v) in &mut self.shadow_handled {
            v.clear();
        }
        self.shadow_handled_used = 0;
    }

    /// Note that draw `draw` casts into layer `layer` from clusters (call with
    /// ascending draw indices per layer).
    pub(crate) fn mark_shadow_handled(&mut self, layer: usize, draw: usize) {
        let used = self.shadow_handled_used;
        let at = match self.shadow_handled[..used].iter().position(|(l, _)| *l == layer) {
            Some(at) => at,
            None => {
                if used == self.shadow_handled.len() {
                    self.shadow_handled.push((layer, Vec::new()));
                }
                self.shadow_handled[used].0 = layer;
                self.shadow_handled_used += 1;
                used
            }
        };
        self.shadow_handled[at].1.push(draw as u32);
    }

    /// Whether `draw`'s shadow in `layer` is drawn from clusters this frame.
    pub(crate) fn shadow_is_handled(&self, layer: usize, draw: usize) -> bool {
        self.shadow_handled[..self.shadow_handled_used]
            .iter()
            .find(|(l, _)| *l == layer)
            .is_some_and(|(_, v)| v.binary_search(&(draw as u32)).is_ok())
    }

    /// A shadow view (cluster shadows) for atlas layer `layer`; returns its
    /// index in the views array (the camera is 0).
    pub(crate) fn add_shadow_view(&mut self, layer: usize, view: GpuView) -> u32 {
        self.shadow_views.push(view);
        let index = self.shadow_views.len() as u32;
        self.shadow_layers.push((layer, index));
        index
    }

    /// The batch that collects `layer`'s shadow casters (one per layer: the
    /// shadow pass ignores materials).
    pub(crate) fn add_shadow_batch(&mut self, layer: usize) -> u32 {
        self.batches.push(ClusterBatch {
            material: 0,
            push: PushData::zeroed(),
            mask: [u32::MAX; 4],
            bounds: None,
            bound: 0,
            early_base: 0,
            shadow_layer: Some(layer),
        });
        (self.batches.len() - 1) as u32
    }

    /// Add `d` as a caster of shadow view `view` (its batch from `add_shadow_batch`):
    /// the cut is judged against that view with the error scale `scale`, never
    /// by the cone test, and (skinned) posed by the same skin stage as the camera's.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_shadow_instance(
        &mut self,
        draw: usize,
        id: u32,
        batch: u32,
        view: u32,
        scale: f32,
        d: &DrawCmd,
        pose: Option<&crate::cluster_skin::SkinPose>,
    ) {
        match pose {
            Some(p) => self.add_skinned_instance(draw, id, batch, d, [1.0; 4], [0.0; 4], p),
            None => self.add_instance(draw, id, batch, d, [1.0; 4], [0.0; 4], Vec3::ZERO),
        }
        let rec = self.instances.last_mut().expect("add_instance pushed a record");
        rec.eye[3] = 0.0;
        rec.extra = [scale.max(1.0), view as f32, 0.0, 0.0];
    }

    /// Add one instance of renderer mesh `mesh` (has a resident DAG) to batch `batch`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_instance(&mut self, draw: usize, id: u32, batch: u32, d: &DrawCmd, color: [f32; 4], param: [f32; 4], cam: Vec3) {
        self.draw_ids.push(draw);
        self.inst_class.push(self.classes.get(&d.mesh.0).copied().unwrap_or_default());
        let mut model = d.transform.to_cols_array_2d();
        model[0][3] = 0.0;
        self.instances.push(InstRec {
            model,
            color,
            param,
            sphere: d.mesh_center.extend(d.bound_radius).to_array(),
            mesh_batch: [id, batch, 0, 0],
            eye: crate::cluster_ref::object_space_eye(d.transform, cam),
            extra: [1.0, 0.0, 0.0, 0.0],
        });
        let (clusters, _, _) = self.store.dims(id);
        self.batches[batch as usize].bound += clusters as u64;
    }

    /// Set every instance's LOD error scale for `view`: the controller's global
    /// scale, raised for instances over their fair share of `max_triangles`.
    fn apply_budget(&mut self, view: &GpuView) {
        use crate::cluster_budget as cb;
        let global = self.global_scale.max(1.0);
        self.budget_stats = BudgetStats { max_scale: global, ..Default::default() };
        let n = self.instances.len();
        if n == 0 {
            return;
        }
        let (ppu, tpx) = (view.lod[1], (view.lod[3] * global).max(1e-6));
        let cam = glam::Vec3::new(view.cam[0], view.cam[1], view.cam[2]);
        self.est_scratch.clear();
        self.weight_scratch.clear();
        self.inst_class.resize(n, cb::ShareClass::default());
        for (i, inst) in self.instances.iter().enumerate() {
            let m = glam::Mat4::from_cols_array_2d(&inst.model);
            let scale = m.x_axis.truncate().length().max(m.y_axis.truncate().length()).max(m.z_axis.truncate().length());
            let centre = m.transform_point3(glam::Vec3::new(inst.sphere[0], inst.sphere[1], inst.sphere[2]));
            if inst.extra[1] > 0.5 {
                // A shadow caster: its own (coarser) cut, outside the camera's budget.
                self.est_scratch.push(0.0);
                self.weight_scratch.push(0.0);
                continue;
            }
            let margin = if inst.mesh_batch[2] != 0 { f32::from_bits(inst.mesh_batch[3]).max(0.0) } else { 0.0 };
            let r = ((inst.sphere[3] + margin) * scale).max(1e-6);
            let rpx = r * ppu / centre.distance(cam).max(r);
            let leaf = self.store.resident(inst.mesh_batch[0]).map_or(0, |r| r.leaf_triangles);
            self.est_scratch.push(cb::estimate_triangles(leaf, rpx, tpx));
            self.weight_scratch.push(self.inst_class[i].weight());
        }
        let total: f32 = self.est_scratch.iter().sum();
        self.budget_stats.estimated = total;
        let limited = self.max_triangles > 0 && total > self.max_triangles as f32;
        if limited {
            self.share_scratch.clear();
            self.share_scratch.resize(n, 0.0);
            cb::fair_shares(self.max_triangles as f32, &self.est_scratch, &self.weight_scratch, &mut self.share_scratch);
        }
        for i in 0..n {
            if self.instances[i].extra[1] > 0.5 {
                continue;
            }
            let mut k = global;
            if limited {
                let f = cb::instance_scale(self.est_scratch[i], self.share_scratch[i], 16.0);
                if f > 1.0 {
                    self.budget_stats.clamped += 1;
                    k = global * f;
                }
            }
            let e = &mut self.instances[i].extra[0];
            *e = e.max(1.0) * k;
            self.budget_stats.max_scale = self.budget_stats.max_scale.max(*e);
        }
    }

    /// Admission error scale: while the skin pool is at its largest and still under pressure, every skinned
    /// instance (camera and shadow casters alike) is judged against `extra[0] * scale^f`, `f` rising with the
    /// instance's camera-distance rank from 0 (nearest) to 1 (the far half). Runs after `apply_budget`.
    fn apply_skin_admission(&mut self, view: &GpuView) {
        let s = self.admission.scale();
        if s <= 1.0 {
            return;
        }
        let cam = glam::Vec3::new(view.cam[0], view.cam[1], view.cam[2]);
        let mut by_dist: Vec<(f32, usize)> = self
            .instances
            .iter()
            .enumerate()
            .filter(|(_, i)| i.mesh_batch[2] != 0)
            .map(|(k, i)| (glam::Mat4::from_cols_array_2d(&i.model).transform_point3(glam::Vec3::new(i.sphere[0], i.sphere[1], i.sphere[2])).distance(cam), k))
            .collect();
        by_dist.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let n = by_dist.len();
        for (rank, &(_, k)) in by_dist.iter().enumerate() {
            let f = if n <= 1 { 1.0 } else { (2.0 * rank as f32 / (n - 1) as f32).min(1.0) };
            let e = &mut self.instances[k].extra[0];
            *e = e.max(1.0) * s.powf(f);
            self.budget_stats.max_scale = self.budget_stats.max_scale.max(*e);
        }
    }

    /// This frame's cull recorded draw commands (tests).
    #[cfg(test)]
    pub(crate) fn drawable_armed(&self) -> bool {
        self.drawable.is_some()
    }

    pub(crate) fn has_instances(&self) -> bool {
        !self.instances.is_empty()
    }

    /// Size the queues and lists for the CPU lists, upload them into frame
    /// slot `slot` and fix the batches' regions. Returns the plan to record.
    pub(crate) fn upload_frame(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        slot: usize,
        view: &GpuView,
        hiz: (vk::Sampler, vk::ImageView),
    ) -> Result<CullPlan> {
        self.apply_budget(view);
        if self.store.stream_active() && self.frames[slot].stream_readback.is_none() {
            self.frames[slot].stream_readback = Some(Buffer::new(
                device,
                allocator,
                stream_readback_bytes(stream_max_pages()),
                vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
                "cluster-stream-readback",
            )?);
        }
        let mut table = std::mem::take(&mut self.table);
        self.store.table(&mut table);
        self.per_instance.clear();
        for i in &self.instances {
            self.per_instance.push(self.store.dims(i.mesh_batch[0]));
        }
        self.bounds_scratch.clear();
        self.bounds_scratch.extend(self.batches.iter().map(|b| b.bound));
        let mut regions = std::mem::take(&mut self.regions_scratch);
        let mut plan = plan_cull_into(&self.per_instance, &self.bounds_scratch, &mut regions);
        if let Some(limit) = self.cap_limit {
            plan.node_cap = plan.node_cap.min(limit);
            plan.occ_cap = plan.occ_cap.min(limit);
            let mut base = 0;
            for r in &mut regions {
                *r = (base, r.1.min(limit));
                base += r.1;
            }
            plan.visible_half = plan.visible_half.min(base.max(1)).max(1);
        }
        let mut batch_recs = std::mem::take(&mut self.batch_recs);
        batch_recs.clear();
        batch_recs.extend(regions.iter().map(|&(base, cap)| [base, cap, plan.visible_half + base, 0]));
        let pools = [
            self.store.clusters.buffer.handle,
            self.store.vertices.buffer.handle,
            self.store.triangles.buffer.handle,
            self.store.nodes.buffer.handle,
        ];
        self.apply_skin_admission(view);
        // The pool follows the selected cut (about 0.8 vertices per triangle), not the DAGs' full resolution.
        let skin_vertices = if self.skin.insts_cpu.is_empty() {
            0
        } else {
            let est: f32 = self.instances.iter().zip(&self.est_scratch).filter(|(i, _)| i.mesh_batch[2] != 0 && i.extra[1] < 0.5).map(|(_, e)| *e).sum();
            self.admission.frame_cap((est * 0.8) as u64, self.skin.pool_vertices)
        };
        self.skin_slot_cap[slot] = skin_vertices;
        let mut views = std::mem::take(&mut self.views_scratch);
        views.clear();
        views.push(*view);
        views.extend_from_slice(&self.shadow_views);
        // A VR stereo cull with Hi-Z: the right eye's view rides at the end of the list and
        // the camera's view names it (`hiz.w` = 2 + its index).
        let stereo = self.stereo_hiz.take();
        let mut hiz_right = hiz.1;
        if let Some(st) = stereo {
            views[0].hiz[3] = 2.0 + views.len() as f32;
            self.stereo_partner = Some(views.len() as u32);
            views.push(st.partner);
            hiz_right = st.right;
        } else {
            self.stereo_partner = None;
        }
        let stream_bufs = self.store.stream_buffers();
        let uploaded = self.frames[slot].upload(device, allocator, pools, stream_bufs, skin_vertices, &plan, &self.instances, &batch_recs, &views, &table, hiz, hiz_right);
        self.views_scratch = views;
        let uploaded = uploaded.and_then(|()| {
            let bytes = (plan.visible_half as usize * 2) as u32;
            self.skin.upload(
                device,
                allocator,
                slot,
                [self.store.skin.buffer.handle, self.store.bones.buffer.handle, self.store.morphs.buffer.handle],
                plan.batches,
                bytes,
            )
        });
        self.skin_pool_cap = skin_vertices;
        self.table = table;
        self.batch_recs = batch_recs;
        uploaded?;
        for (b, &(base, _)) in self.batches.iter_mut().zip(&regions) {
            b.early_base = base;
        }
        self.regions_scratch = regions;
        self.plan = plan;
        self.last_instances = self.instances.len() as u32;
        self.last_batches = plan.batches;
        Ok(plan)
    }

    pub(crate) fn targets(&self, slot: usize) -> CullTargets {
        let f = &self.frames[slot];
        let skin = (!self.skin.insts_cpu.is_empty()).then(|| {
            (
                self.skin.targets(slot, f.set, self.plan.batches, self.plan.visible_half * 2, self.skin_pool_cap),
                self.skin.pipes.pipeline,
                self.skin.pipes.layout,
            )
        });
        CullTargets {
            skin,
            skin_stamps: None,
            set: f.set,
            counters: f.counters.handle,
            dispatch: f.dispatch.handle,
            batch_counts: f.batch_counts.handle(),
            batch_count_bytes: self.plan.batches as u64 * 8,
            readback: Some(f.readback.handle),
            stream: self.store.stream_active().then_some(f.stream_readback.as_ref()).flatten().map(|rb| {
                let (_, io) = self.store.stream_buffers();
                StreamTargets { io, readback: rb.handle, touch_words: self.store.stream_high(), max_pages: stream_max_pages() as u32 }
            }),
        }
    }

    /// Frame slot buffers a test inspects: (handle, allocated bytes) of the
    /// visible list, the indirect commands, the two node queues and the occluded list.
    pub(crate) fn frame_outputs(&self, slot: usize) -> [(vk::Buffer, usize); 7] {
        let f = &self.frames[slot];
        [
            (f.visible.handle(), f.visible.bytes()),
            (f.cmds.handle(), f.cmds.bytes()),
            (f.queue_a.handle(), f.queue_a.bytes()),
            (f.queue_b.handle(), f.queue_b.bytes()),
            (f.occluded.handle(), f.occluded.bytes()),
            (f.skin_pool.handle(), f.skin_pool.bytes()),
            (f.skin_slots.handle(), f.skin_slots.bytes()),
        ]
    }

    /// The counters the last cull of `slot` copied out (once it has completed).
    pub(crate) fn read_counters(&self, slot: usize) -> Vec<u32> {
        let m = self.frames[slot].readback.mapped().unwrap_or(&[]);
        bytemuck::pod_collect_to_vec(&m[..m.len().min(COUNTER_BYTES as usize) & !3])
    }

    /// Region record (early base, capacity, late base) of batch `b`.
    pub(crate) fn batch_rec(&self, b: usize) -> BatchRec {
        self.batch_recs[b]
    }

    /// `add_raw_instance` for a skinned instance (headless tests).
    pub(crate) fn add_raw_skinned_instance(&mut self, mut rec: InstRec, batch: u32, draw: usize, pose: &crate::cluster_skin::SkinPose) {
        rec.mesh_batch[2] = self.skin.add(pose);
        rec.mesh_batch[3] = pose.margin.max(0.0).to_bits();
        rec.eye[3] = 0.0;
        self.add_raw_instance(rec, batch, draw);
    }

    /// Add an instance record directly (headless tests).
    pub(crate) fn add_raw_instance(&mut self, rec: InstRec, batch: u32, draw: usize) {
        self.draw_ids.push(draw);
        self.inst_class.push(crate::cluster_budget::ShareClass::default());
        let (clusters, _, _) = self.store.dims(rec.mesh_batch[0]);
        self.batches[batch as usize].bound += clusters as u64;
        self.instances.push(rec);
    }
}

// ---------------------------------------------------------------------------
// Renderer integration

impl Renderer {
    /// Set the cluster-rendering options (takes effect next frame).
    pub fn set_cluster_settings(&mut self, settings: ClusterSettings) {
        // The environment override is fixed at startup.
        let env = self.cluster_settings.env_override;
        self.cluster_settings = ClusterSettings { env_override: env, ..settings };
    }

    pub fn cluster_settings(&self) -> ClusterSettings {
        self.cluster_settings
    }

    /// Whether the cluster path is requested and can run on this device: meshes
    /// with a DAG may be installed and are worth building.
    pub fn cluster_rendering_enabled(&self) -> bool {
        self.cluster_settings.enabled() && !self.cluster_failed && !self.gpu_cull_failed && self.gpu_cull_settings.enabled
    }

    /// Record the scene's cluster coverage (meshes on / off the cluster path and why);
    /// shown in the render statistics.
    pub fn set_cluster_coverage(&mut self, coverage: crate::ClusterCoverage) {
        self.cluster_coverage = coverage;
    }

    /// Whether renderer mesh `mesh` has a DAG installed.
    pub fn has_cluster_dag(&self, mesh: MeshHandle) -> bool {
        self.cluster.as_ref().is_some_and(|c| c.store.record_of(mesh.0).is_some())
    }

    fn cluster_ensure(&mut self) -> Result<()> {
        if self.cluster.is_some() {
            return Ok(());
        }
        let mut sizes = PoolSizes::default();
        // `ZESTENGINE_CLUSTER_POOL_SCALE=<n>` (1..=8) multiplies every pool (and so its growth limit):
        // for diagnostics and for tests that hold a whole large avatar resident as a reference.
        if let Some(n) = std::env::var("ZESTENGINE_CLUSTER_POOL_SCALE").ok().and_then(|v| v.parse::<u32>().ok()) {
            let n = n.clamp(1, 8);
            sizes = PoolSizes {
                clusters: sizes.clusters.saturating_mul(n),
                vertices: sizes.vertices.saturating_mul(n),
                tri_words: sizes.tri_words.saturating_mul(n),
                nodes: sizes.nodes.saturating_mul(n),
                skin: sizes.skin.saturating_mul(n),
                bones: sizes.bones.saturating_mul(n),
                morphs: sizes.morphs.saturating_mul(n),
                range: sizes.range,
            };
        }
        let sizes = sizes.clamped(self.ctx.caps.max_storage_buffer_range.max(1 << 20));
        let allocator = self.allocator();
        let sys = ClusterSystem::new(&self.ctx.device, &mut allocator.lock().unwrap(), self.pipeline_cache.cluster_set_layout, sizes)?;
        self.cluster = Some(sys);
        Ok(())
    }

    /// Install the DAG of an already uploaded static mesh so it can be drawn
    /// from clusters. Fails (leaving the mesh on the ordinary path) when the
    /// DAG is invalid or the pool is full; the caller logs once.
    pub fn install_cluster_dag(&mut self, mesh: MeshHandle, dag: &ClusterMeshUpload) -> Result<()> {
        if !self.cluster_rendering_enabled() {
            bail!("cluster rendering is off");
        }
        if mesh.0 >= self.meshes.len() || self.freed_meshes.contains(&mesh.0) {
            bail!("mesh {} does not exist", mesh.0);
        }
        if let Err(e) = self.cluster_ensure() {
            self.cluster_failed = true;
            tracing::warn!("Zestlets unavailable, using the ordinary path: {e:#}");
            return Err(e);
        }
        if dag.stream.is_some() {
            self.cluster_stream_apply();
        }
        let allocator = self.allocator();
        let mut alloc = allocator.lock().unwrap();
        let sys = self.cluster.as_mut().unwrap();
        sys.store.install(&self.ctx.device, &mut alloc, self.command_pool, self.ctx.queue, mesh.0, dag).map(|_| ())
    }

    /// Install a DAG staged on a loader thread (`Uploader::stage_cluster_dag`)
    /// without any queue submit or wait: the pool ranges are reserved now and the
    /// next rendered frame records the copy into its own command buffer, after
    /// which `has_cluster_dag(mesh)` is true and the mesh draws from clusters.
    /// A mesh freed before then drops the copy.
    pub fn install_staged_cluster_dag(&mut self, mesh: MeshHandle, mut staged: StagedClusterDag) -> Result<()> {
        let refuse = |staged: &mut StagedClusterDag, this: &Self, why: String| -> anyhow::Error {
            staged.destroy(&this.ctx.device, &mut this.allocator().lock().unwrap());
            anyhow::anyhow!(why)
        };
        if !self.cluster_rendering_enabled() {
            return Err(refuse(&mut staged, self, "cluster rendering is off".into()));
        }
        if mesh.0 >= self.meshes.len() || self.freed_meshes.contains(&mesh.0) {
            return Err(refuse(&mut staged, self, format!("mesh {} does not exist", mesh.0)));
        }
        if let Err(e) = self.cluster_ensure() {
            self.cluster_failed = true;
            tracing::warn!("Zestlets unavailable, using the ordinary path: {e:#}");
            staged.destroy(&self.ctx.device, &mut self.allocator().lock().unwrap());
            return Err(e);
        }
        if staged.stream.is_some() {
            self.cluster_stream_apply();
        }
        let allocator = self.allocator();
        let mut alloc = allocator.lock().unwrap();
        let sys = self.cluster.as_mut().unwrap();
        sys.store.install_staged(&self.ctx.device, &mut alloc, self.command_pool, self.ctx.queue, mesh.0, staged).map(|_| ())
    }

    /// `install_cluster_dag` without the wait: stages on the calling thread and
    /// queues the copy for the next frame (see `install_staged_cluster_dag`).
    pub fn install_cluster_dag_deferred(&mut self, mesh: MeshHandle, dag: &ClusterMeshUpload) -> Result<()> {
        let staged = StagedClusterDag::new(&self.ctx.device, &mut self.allocator().lock().unwrap(), dag)?;
        self.install_staged_cluster_dag(mesh, staged)
    }

    /// Free the staging buffer of a DAG that will not be installed.
    pub fn discard_staged_cluster_dag(&mut self, mut staged: StagedClusterDag) {
        // No frame has used it: nothing to wait for.
        staged.destroy(&self.ctx.device, &mut self.allocator().lock().unwrap());
    }

    /// Whether meshes installed now should be streamed (the setting is on and the path runs).
    pub fn cluster_stream_enabled(&self) -> bool {
        self.cluster_rendering_enabled() && self.cluster_settings.stream
    }

    /// Stages page payloads from a loader thread (no queue needed).
    pub fn cluster_page_stager(&self) -> PageStager {
        PageStager { device: self.ctx.device.clone(), allocator: self.allocator() }
    }

    /// Apply the streaming settings (budget) to the store: creates the page
    /// blocks once, then only moves the budget within them. Called every frame
    /// from `cluster_begin` and before a streamed install.
    pub(crate) fn cluster_stream_apply(&mut self) {
        let want = self.cluster_settings;
        let Some(c) = self.cluster.as_mut() else { return };
        if !want.stream && !c.store.stream_active() {
            return;
        }
        let policy = StreamPolicy { budget_bytes: (want.stream_budget_mb as u64) << 20, ..c.store.stream_policy() };
        // Compare with what was ASKED for: the pools may have clamped the effective budget below
        // it (then the effective value never equals the request and this would reconfigure, free and
        // re-create the page blocks and log every frame).
        if c.store.stream_configured() && c.store.stream_requested_budget() == policy.budget_bytes {
            return;
        }
        let policy = StreamPolicy { budget_bytes: policy.budget_bytes, ..StreamPolicy::default() };
        let allocator = self.allocator.clone().expect("allocator");
        match c.store.stream_configure(&self.ctx.device, &mut allocator.lock().unwrap(), self.command_pool, self.ctx.queue, policy) {
            Ok(p) => tracing::info!("Zestlet streaming: budget {} MB", p.budget_bytes >> 20),
            Err(e) => tracing::warn!("Zestlet streaming could not be configured: {e:#}"),
        }
    }

    /// Pages the streamer should read and stage now (parents before children).
    /// Each is answered with `cluster_stream_deliver` or `cluster_stream_fail`.
    pub fn cluster_stream_orders(&mut self) -> Vec<PageLoad> {
        self.cluster.as_mut().map_or_else(Vec::new, |c| c.store.take_orders())
    }

    /// A staged page for an order. Installed at once when its parents are
    /// resident (the next frame records the copy), otherwise held until they are.
    pub fn cluster_stream_deliver(&mut self, mesh: MeshHandle, page: u32, mut staged: StagedPage) {
        let allocator = self.allocator();
        let mut alloc = allocator.lock().unwrap();
        match self.cluster.as_mut() {
            Some(c) => c.store.stream_deliver(&self.ctx.device, &mut alloc, mesh.0, page, staged),
            None => staged.destroy(&self.ctx.device, &mut alloc),
        }
    }

    /// The streamer could not read an ordered page; it is asked for again later.
    pub fn cluster_stream_fail(&mut self, mesh: MeshHandle, page: u32) {
        let allocator = self.allocator();
        let mut alloc = allocator.lock().unwrap();
        if let Some(c) = self.cluster.as_mut() {
            c.store.stream_fail(&self.ctx.device, &mut alloc, mesh.0, page);
        }
    }

    /// Streaming counters (zero while no streamed mesh is installed).
    pub fn cluster_stream_stats(&self) -> StreamStats {
        self.cluster.as_ref().map_or_else(StreamStats::default, |c| c.store.stream_stats())
    }

    /// Drop every streamed page now (tests, a changed pool).
    pub fn cluster_stream_evict_all(&mut self) {
        let serial = self.frame_serial;
        if let Some(c) = self.cluster.as_mut() {
            c.store.stream_evict_all(serial);
        }
    }

    /// DAGs installed but whose copy no frame has recorded yet (tests).
    pub fn cluster_uploads_queued(&self) -> usize {
        self.cluster.as_ref().map_or(0, |c| c.store.queued_uploads())
    }

    /// Record the queued DAG copies into this frame's command buffer, ahead of
    /// every pass that reads the pools.
    pub(crate) fn cluster_record_uploads(&mut self, cb: vk::CommandBuffer) {
        let serial = self.frame_serial;
        if let Some(c) = self.cluster.as_mut() {
            c.store.record_uploads(&self.ctx.device, cb, serial);
        }
    }

    /// Draw skinned copy `mesh` from the DAG installed for `source` (the copies
    /// of one avatar share one DAG; each keeps its own pose). Fails when
    /// `source` has no DAG.
    pub fn alias_cluster_dag(&mut self, mesh: MeshHandle, source: MeshHandle) -> Result<()> {
        if !self.cluster_rendering_enabled() {
            bail!("cluster rendering is off");
        }
        if mesh.0 >= self.meshes.len() || self.freed_meshes.contains(&mesh.0) {
            bail!("mesh {} does not exist", mesh.0);
        }
        if self.cluster.as_mut().is_some_and(|c| c.store.alias(mesh.0, source.0)) {
            Ok(())
        } else {
            bail!("mesh {} has no cluster DAG to share", source.0)
        }
    }

    /// Weight class of a mesh's draws in the triangle budget's fair share (the
    /// local avatar outweighs other players' untrusted ones). Meshes default to
    /// `Trusted`. Takes effect next frame.
    pub fn set_cluster_share_class(&mut self, mesh: MeshHandle, class: crate::cluster_budget::ShareClass) {
        if let Some(c) = self.cluster.as_mut() {
            if class == crate::cluster_budget::ShareClass::default() {
                c.classes.remove(&mesh.0);
            } else {
                c.classes.insert(mesh.0, class);
            }
        }
    }

    /// Frame slot `slot` has retired: read back its counters, apply the overflow
    /// policy, release retired pool ranges.
    pub(crate) fn cluster_begin(&mut self, slot: usize) {
        self.cluster_stream_apply();
        let serial = self.frame_serial;
        let (target_ms, max_tris) = (self.cluster_settings.target_frame_ms, self.cluster_settings.max_triangles);
        let gpu_ms = self.gpu_zone_ms[gpu_zone::FRAME];
        let allocator = self.allocator.clone();
        let Some(c) = self.cluster.as_mut() else { return };
        c.store.reclaim(serial);
        if c.store.retired_buffers() > 0
            && let Some(allocator) = allocator
        {
            c.store.reclaim_buffers(&self.ctx.device, &mut allocator.lock().unwrap());
        }
        c.max_triangles = max_tris;
        // Budget: the always-on GPU frame time against the target coarsens the
        // LOD error scale (one frame behind, smoothed, slew limited).
        if target_ms > 0.0 {
            c.global_scale = c.budget.update(gpu_ms, target_ms, c.counts.triangles);
        } else if c.global_scale != 1.0 {
            c.budget.reset();
            c.global_scale = 1.0;
        }
        if std::mem::take(&mut c.skin.frames[slot].recorded) {
            c.skin.counts = crate::cluster_skin::SkinCounts::from_words(&c.skin.read_ctl(slot));
            let overflow = c.skin.counts.flags & crate::cluster_skin::ctl::FLAG_POOL_FULL != 0 || c.skin.counts.used > c.skin_slot_cap[slot];
            if overflow {
                c.skin.pool_overflows += 1;
                if !c.warned_skin {
                    c.warned_skin = true;
                    tracing::warn!(
                        "cluster path: the skinned-vertex pool overflowed ({} of {} vertices); the pool grows, then the farthest skinned instances \
                         coarsen (raise ZESTENGINE_CLUSTER_SKIN_VERTICES for more room)",
                        c.skin.counts.used,
                        c.skin_slot_cap[slot]
                    );
                }
            }
            let (used, cap) = (c.skin.counts.used, c.skin_slot_cap[slot]);
            c.admission.pressure(used, cap, c.skin.pool_vertices, overflow, serial);
        }
        let f = &mut c.frames[slot];
        let recorded = std::mem::take(&mut f.recorded);
        if !recorded {
            // The path did not run in this slot's last frame: no stale numbers.
            c.counts = ClusterCounts::default();
            c.skin.counts = Default::default();
        }
        // Streaming: the retired frame's page requests and touched pages feed the policy.
        if std::mem::take(&mut f.stream_recorded)
            && let Some(bytes) = f.stream_readback.as_ref().and_then(|b| b.mapped())
            && bytes.len() as u64 >= stream_readback_bytes(stream_max_pages())
        {
            let words: &[u32] = bytemuck::cast_slice(&bytes[..stream_readback_bytes(stream_max_pages()) as usize & !3]);
            let n = (words[0] as usize).min(STREAM_REQ_CAP);
            let reqs: Vec<(u32, u32)> = (0..n).map(|i| (words[STREAM_REQ_BASE + 2 * i], words[STREAM_REQ_BASE + 2 * i + 1])).collect();
            let touch = &words[STREAM_PRIO_BASE..];
            let high = (c.store.stream_high() as usize).min(touch.len());
            c.store.stream_ingest(&reqs, touch[..high].iter().enumerate().filter(|(_, t)| **t != 0).map(|(i, _)| i as u32), serial);
        }
        if c.store.stream_configured() {
            c.store.stream_plan(serial);
            c.store.stream_retry_held(&self.ctx.device, &mut self.allocator.as_ref().expect("allocator").lock().unwrap());
        }
        if recorded
            && let Some(bytes) = f.readback.mapped()
            && bytes.len() >= COUNTER_BYTES as usize
        {
            let words: Vec<u32> = bytemuck::pod_collect_to_vec(&bytes[..COUNTER_BYTES as usize]);
            c.counts = ClusterCounts::from_words(&words);
            if c.counts.overflowed() {
                c.overflow_frames += 1;
                c.disabled_until = serial + OVERFLOW_COOLDOWN;
                if !c.warned {
                    c.warned = true;
                    tracing::warn!(
                        "cluster path: a queue or list overflowed (flags {:#x}); drawing the original meshes for {} frames. \
                         Counted in the 'queue overflows' profiler counter",
                        c.counts.flags,
                        OVERFLOW_COOLDOWN
                    );
                }
            }
        }
    }

    /// Whether a VR stereo frame can use the cluster path (`ZESTENGINE_CLUSTERS_XR=0` turns it off).
    pub(crate) fn cluster_xr_ok(&self, input: &FrameInput) -> bool {
        static XR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *XR.get_or_init(|| !std::env::var("ZESTENGINE_CLUSTERS_XR").is_ok_and(|v| v == "0"))
            && self.cluster_frame_ok(input)
    }

    /// Whether this frame's main camera can use the cluster path.
    pub(crate) fn cluster_frame_ok(&self, input: &FrameInput) -> bool {
        self.cluster_rendering_enabled()
            && input.camera.proj.w_axis.w == 0.0
            && input.camera.proj.y_axis.y.abs() > 0.0
            && self.cluster.as_ref().is_some_and(|c| c.store.mesh_count() > 0 && c.disabled_until <= self.frame_serial)
    }

    /// Pipeline key of a cluster batch's material.
    fn cluster_pipeline_key(&self, material: usize, gbuf: bool) -> crate::pipeline::PipelineKey {
        let m = &self.materials[material];
        let mut key = scene_pipeline_key(m, gbuf, self.velocity_active, self.capturing_refl, self.draw_mode, false);
        key.cluster = true;
        key
    }

    /// Upload the frame's lists and record the early cull (main camera).
    /// False when the upload failed (the caller hands the draws back).
    pub(crate) fn cluster_record_early(&mut self, cb: vk::CommandBuffer, view: &GpuView, hiz: (vk::Sampler, vk::ImageView)) -> bool {
        self.cluster_record_early_in(cb, self.frame_index, view, hiz)
    }

    /// `cluster_record_early` into frame set `slot` (a frame in flight, or
    /// `FRAMES_IN_FLIGHT + eye slot` for a VR stereo pass: no counters are read
    /// back from those and no timestamps are written).
    pub(crate) fn cluster_record_early_in(&mut self, cb: vk::CommandBuffer, slot: usize, view: &GpuView, hiz: (vk::Sampler, vk::ImageView)) -> bool {
        let xr = slot >= FRAMES_IN_FLIGHT;
        let allocator = self.allocator();
        let Some(mut c) = self.cluster.take() else { return false };
        c.drawable = None;
        if c.instances.is_empty() {
            self.cluster = Some(c);
            return true;
        }
        let uploaded = c.upload_frame(&self.ctx.device, &mut allocator.lock().unwrap(), slot, view, hiz);
        let ok = uploaded.is_ok();
        match uploaded {
            Ok(plan) => {
                if !xr {
                    self.gpu_zone_begin(cb, gpu_zone::CLUSTER_CULL);
                }
                let mut targets = c.targets(slot);
                targets.skin_stamps = if xr { None } else { self.cluster_skin_stamps() };
                if xr {
                    // Counters are copied out for tests and statistics only (no ingest: `recorded` stays false).
                    targets.stream = None;
                    c.xr_last_slot = Some(slot);
                }
                record_early(&self.ctx.device, cb, &c.pipes, &targets, &plan);
                c.frames[slot].recorded = !xr;
                c.frames[slot].stream_recorded = targets.stream.is_some();
                c.skin.frames[slot].recorded = targets.skin.is_some() && !xr;
                if !xr {
                    self.gpu_zone_end(cb, gpu_zone::CLUSTER_CULL);
                }
                c.drawable = Some(Drawable {
                    cmds: c.frames[slot].cmds.handle(),
                    set: c.frames[slot].set,
                    batches: plan.batches,
                    visible_half: plan.visible_half,
                    late: false,
                });
            }
            Err(e) => {
                tracing::warn!("Zestlet upload failed, using the ordinary path: {e:#}");
                c.disabled_until = self.frame_serial + OVERFLOW_COOLDOWN;
                // The caller hands the draws back to the ordinary path.
                c.instances.clear();
                c.batches.clear();
                c.forget_shadow_handled();
            }
        }
        self.cluster = Some(c);
        ok
    }

    /// The skin zones' timestamp queries when zone timing is on.
    fn cluster_skin_stamps(&self) -> Option<(vk::QueryPool, [u32; 2])> {
        self.cluster_skin_stamps_for(self.frame_index)
    }

    fn cluster_skin_stamps_for(&self, slot: usize) -> Option<(vk::QueryPool, [u32; 2])> {
        self.ts_active.then(|| {
            (
                self.frames[slot].timestamps,
                [(2 * gpu_zone::CLUSTER_SKIN) as u32, (2 * gpu_zone::CLUSTER_SKIN_LATE) as u32],
            )
        })
    }

    /// Skinned meshes whose ordinary skin dispatch this frame can be dropped:
    /// the cluster path draws every draw of them that reaches the camera pass,
    /// and every shadow layer they cast into, from its own posed pool (shadows
    /// included), and nothing else reads their skinned buffer: no secondary
    /// camera (planar, render texture, preview, probe capture, XR), no ray
    /// tracing, no selection outline. Empty when the cluster path is idle.
    /// `order` = the camera's visible draws, `work` = the shadow layers that re-render.
    pub(crate) fn cluster_skin_skip(&mut self, order: &[usize], input: &FrameInput, work: &[(usize, Mat4, Vec<usize>)]) -> Vec<usize> {
        let mut skip = self.cluster.as_mut().map(|c| std::mem::take(&mut c.skip_scratch)).unwrap_or_default();
        skip.clear();
        let Some(mut c) = self.cluster.take() else { return skip };
        'plan: {
            if c.drawable.is_none()
                || c.main_clustered.is_empty()
                || self.skinner.is_none()
                // Ray-traced reflections refit their BLAS from the skinned buffer.
                || (self.rt_reflect.is_some() && self.flux_settings.reflection_mode == 2)
                || !input.water.is_empty()
                || self.refl_job.is_some()
                || self.xr_render.is_some()
                || !input.planar.is_empty()
                || !input.camera_targets.is_empty()
                || input.camera_preview.is_some()
            {
                break 'plan;
            }
            let draws = input.draws;
            let mut need = std::mem::take(&mut c.need_scratch);
            need.clear();
            need.resize(draws.len(), false);
            for &i in order {
                need[i] = !c.is_main(i);
            }
            for (layer, _, casters) in work {
                for &i in casters {
                    need[i] |= !c.shadow_is_handled(*layer, i);
                }
            }
            // Blocked meshes: some draw still needs the ordinary skinned buffer.
            let mut blocked: Vec<usize> = Vec::new();
            for (i, d) in draws.iter().enumerate() {
                if need[i] || d.highlight > 0.0 {
                    blocked.push(d.mesh.0);
                }
            }
            blocked.sort_unstable();
            blocked.dedup();
            for (i, d) in draws.iter().enumerate() {
                if c.is_main(i) && blocked.binary_search(&d.mesh.0).is_err() && !skip.contains(&d.mesh.0) && self.skinner.as_ref().is_some_and(|s| s.pose_of(d.mesh.0).is_some()) {
                    skip.push(d.mesh.0);
                }
            }
            c.need_scratch = need;
        }
        c.skin_skipped = skip.len() as u32;
        self.cluster = Some(c);
        skip
    }

    /// Give the skip list's storage back for reuse next frame.
    pub(crate) fn cluster_skin_skip_done(&mut self, skip: Vec<usize>) {
        if let Some(c) = self.cluster.as_mut() {
            c.skip_scratch = skip;
        }
    }

    /// Forget last frame's cluster draws (the path is not used this frame).
    pub(crate) fn cluster_reset(&mut self) {
        if let Some(c) = self.cluster.as_mut() {
            c.drawable = None;
            c.batches.clear();
            c.instances.clear();
            c.forget_shadow_handled();
        }
    }

    /// The draws a failed cluster cull took over, to be drawn the ordinary way.
    pub(crate) fn cluster_take_draws(&mut self) -> Vec<usize> {
        self.cluster.as_mut().map(|c| std::mem::take(&mut c.draw_ids)).unwrap_or_default()
    }

    /// The late pass (called by `gpu_cull_late` after the pyramid was built).
    pub(crate) fn cluster_record_late(&mut self, cb: vk::CommandBuffer, late_vp: Mat4) -> bool {
        let slot = self.frame_index;
        let stamps = self.cluster_skin_stamps_for(slot);
        let Some(c) = self.cluster.as_mut() else { return false };
        let Some(d) = c.drawable.as_mut() else { return false };
        d.late = true;
        let mut targets = c.targets(slot);
        targets.skin_stamps = stamps;
        let plan = c.plan;
        record_late(&self.ctx.device, cb, &c.pipes, &targets, &plan, late_vp);
        true
    }

    /// The late cull of a VR stereo pass: first point the two eyes' views at THIS frame's
    /// view-projections (the pyramids were just built from this frame's depth), then re-test
    /// what the early pass found occluded. `vps` = (left, right) view-projection.
    pub(crate) fn cluster_record_late_xr(&mut self, cb: vk::CommandBuffer, slot: usize, vps: [Mat4; 2]) -> bool {
        let Some(c) = self.cluster.as_mut() else { return false };
        let (Some(partner), Some(_)) = (c.stereo_partner, c.drawable.as_ref()) else { return false };
        let views = c.frames[slot].views.handle();
        let stride = size_of::<GpuView>() as u64;
        let hiz_vp_at = 6 * 16;
        let device = &self.ctx.device;
        // The early cull's reads of the views come before these writes, the writes before the late reads.
        barrier(device, cb, (CS, vk::AccessFlags2::SHADER_STORAGE_READ), (vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE));
        for (index, vp) in [(0u64, vps[0]), (partner as u64, vps[1])] {
            let m = vp.to_cols_array();
            unsafe { device.cmd_update_buffer(cb, views, index * stride + hiz_vp_at, bytemuck::cast_slice(&m)) };
        }
        let mut targets = c.targets(slot);
        targets.readback = Some(c.frames[slot].readback.handle);
        let plan = c.plan;
        if let Some(d) = c.drawable.as_mut() {
            d.late = true;
        }
        record_late(device, cb, &c.pipes, &targets, &plan, vps[0]);
        true
    }

    /// The late cull of a per-eye VR pass: re-test what the early pass found occluded against the
    /// pyramid just built from this eye's depth (`vp` is this frame's view-projection).
    pub(crate) fn cluster_record_late_eye(&mut self, cb: vk::CommandBuffer, slot: usize, vp: Mat4) -> bool {
        let Some(c) = self.cluster.as_mut() else { return false };
        if c.drawable.is_none() {
            return false;
        }
        let mut targets = c.targets(slot);
        targets.readback = Some(c.frames[slot].readback.handle);
        let plan = c.plan;
        if let Some(d) = c.drawable.as_mut() {
            d.late = true;
        }
        record_late(&self.ctx.device, cb, &c.pipes, &targets, &plan, vp);
        true
    }

    /// The counters of the last VR eye-render cull (read after that frame completed).
    #[cfg(test)]
    pub(crate) fn cluster_xr_counts(&self) -> Option<ClusterCounts> {
        let c = self.cluster.as_ref()?;
        let words = c.read_counters(c.xr_last_slot?);
        (words.len() >= 24).then(|| ClusterCounts::from_words(&words))
    }

    /// Whether this frame has cluster batches to draw.
    pub(crate) fn cluster_armed(&self) -> bool {
        self.cluster.as_ref().is_some_and(|c| c.drawable.is_some())
    }

    /// Draw the cluster batches of the current pass (early commands, or the
    /// late ones when `late`). One indirect draw per batch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draw_cluster_batches(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        frame_set: vk::DescriptorSet,
        gbuf: bool,
        late: bool,
        bound: &mut BoundDraw,
        stats: &mut RenderStats,
        materials_seen: &mut [bool],
    ) -> Result<()> {
        // Multiview (VR): both eyes draw the one visible list the centre-eye cut
        // produced (`standard.vert -DCLUSTER -DMULTIVIEW`, picked by the pipeline cache).
        let Some(c) = self.cluster.take() else { return Ok(()) };
        let Some(d) = c.drawable.filter(|d| !late || d.late) else {
            self.cluster = Some(c);
            return Ok(());
        };
        let zone = [gpu_zone::CLUSTER_DRAW, gpu_zone::CLUSTER_DRAW_LATE, gpu_zone::CLUSTER_PRE, gpu_zone::CLUSTER_PRE_LATE]
            [2 * (self.draw_mode == DrawMode::Prepass) as usize + late as usize];
        self.gpu_zone_begin(cb, zone);
        let mut result = Ok(());
        let mut drawn = 0;
        for (i, b) in c.batches.iter().enumerate() {
            if b.shadow_layer.is_some() {
                continue;
            }
            let key = self.cluster_pipeline_key(b.material, gbuf);
            let pipeline = match self.pipeline_cache.get(device, key) {
                Ok(p) => p,
                Err(e) => {
                    result = Err(e);
                    break;
                }
            };
            let material = &self.materials[b.material];
            if let Some(seen) = materials_seen.get_mut(b.material)
                && !*seen
            {
                *seen = true;
                stats.materials_drawn += 1;
            }
            let mut push = b.push;
            push.model[0] = if self.cull_active { b.mask } else { [u32::MAX; 4] }.map(f32::from_bits);
            let region = if late { d.visible_half + b.early_base } else { b.early_base };
            push.model[1][0] = f32::from_bits(region);
            unsafe {
                if pipeline != bound.pipeline {
                    stats.pipeline_binds += 1;
                    device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
                    bound.pipeline = pipeline;
                    device.cmd_bind_descriptor_sets(
                        cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_cache.cluster_layout,
                        0,
                        &[frame_set],
                        &[],
                    );
                    device.cmd_bind_descriptor_sets(
                        cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_cache.cluster_layout,
                        2,
                        &[d.set],
                        &[],
                    );
                    bound.material = vk::DescriptorSet::null();
                }
                if material.set != bound.material {
                    device.cmd_bind_descriptor_sets(
                        cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_cache.cluster_layout,
                        1,
                        &[material.set],
                        &[],
                    );
                    bound.material = material.set;
                }
                device.cmd_push_constants(
                    cb,
                    self.pipeline_cache.cluster_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&push),
                );
                let cmd_offset = (if late { d.batches as u64 } else { 0 } + i as u64) * 16;
                device.cmd_draw_indirect(cb, d.cmds, cmd_offset, 1, 16);
            }
            stats.draw_calls += 1;
            drawn += 1;
        }
        // The next batch of the pass binds its own mesh and pipeline again.
        *bound = BoundDraw::default();
        stats.indirect_draws += drawn;
        self.gpu_zone_end(cb, zone);
        self.cluster = Some(c);
        result
    }

    /// Draw the cluster casters of shadow layer `layer` (depth-only pipeline,
    /// vertex pulling from the visible list the cull wrote for that layer's
    /// view). Call outside the layer's ordinary shadow pipeline binding: this
    /// binds its own, and the caller rebinds its pipeline afterwards. False
    /// when the layer has no cluster casters this frame.
    pub(crate) fn draw_cluster_shadow_layer(&mut self, device: &ash::Device, cb: vk::CommandBuffer, layer: usize, light_vp: Mat4) -> Result<bool> {
        let Some(c) = self.cluster.take() else { return Ok(false) };
        let Some(d) = c.drawable else {
            self.cluster = Some(c);
            return Ok(false);
        };
        let mut drawn = false;
        let mut result = Ok(());
        for (i, b) in c.batches.iter().enumerate() {
            if b.shadow_layer != Some(layer) {
                continue;
            }
            let pipeline = match self.pipeline_cache.get(device, crate::pipeline::PipelineKey::shadow_cluster()) {
                Ok(p) => p,
                Err(e) => {
                    result = Err(e);
                    break;
                }
            };
            let mut push = crate::ShadowPush { light_vp: light_vp.to_cols_array_2d(), model: Mat4::IDENTITY.to_cols_array_2d() };
            push.model[1][0] = f32::from_bits(b.early_base);
            unsafe {
                if !drawn {
                    device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
                    device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline_cache.cluster_layout, 2, &[d.set], &[]);
                }
                device.cmd_push_constants(
                    cb,
                    self.pipeline_cache.cluster_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&push),
                );
                device.cmd_draw_indirect(cb, d.cmds, i as u64 * 16, 1, 16);
            }
            drawn = true;
        }
        self.cluster = Some(c);
        result.map(|()| drawn)
    }

    /// Fold the cluster counters into the frame's statistics.
    pub(crate) fn cluster_stats(&self, stats: &mut RenderStats) {
        let s = &mut stats.cluster;
        s.enabled = self.cluster_rendering_enabled();
        s.coverage = self.cluster_coverage;
        s.cull_ms = self.gpu_zone_ms[gpu_zone::CLUSTER_CULL];
        s.draw_ms = self.gpu_zone_ms[gpu_zone::CLUSTER_DRAW] + self.gpu_zone_ms[gpu_zone::CLUSTER_DRAW_LATE];
        s.prepass_ms = self.gpu_zone_ms[gpu_zone::CLUSTER_PRE] + self.gpu_zone_ms[gpu_zone::CLUSTER_PRE_LATE];
        s.skin_ms = self.gpu_zone_ms[gpu_zone::CLUSTER_SKIN] + self.gpu_zone_ms[gpu_zone::CLUSTER_SKIN_LATE];
        // One-time notice when the skin pass alone eats a large part of the frame (it used to be 4+ ms
        // for a 1.7M-entry blendshape face), so a silent slowdown is visible in the log.
        static WARNED_SLOW_SKIN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if s.skin_ms > 2.0 && !WARNED_SLOW_SKIN.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::warn!("zestlet skin pass takes {:.2} ms of GPU time per frame (skinned clusters / blendshape entries are heavy); logged once", s.skin_ms);
        }
        let Some(c) = self.cluster.as_ref() else { return };
        s.meshes = c.store.mesh_count() as u32;
        s.instances = c.counts.instances;
        s.instances_culled = c.counts.instances_culled;
        s.nodes_visited = c.counts.nodes;
        s.clusters_tested = c.counts.tested;
        s.culled_lod = c.counts.lod_rejected;
        s.culled_frustum = c.counts.frustum;
        s.culled_cone = c.counts.cone;
        s.culled_hiz = c.counts.hiz.saturating_sub(c.counts.late_recovered);
        s.recovered = c.counts.late_recovered;
        s.visible_clusters = c.counts.visible;
        s.triangles = c.counts.triangles;
        s.queue_overflows = c.overflow_frames;
        s.error_scale = c.global_scale;
        s.target_ms = self.cluster_settings.target_frame_ms;
        s.budget_estimated = c.budget_stats.estimated as u32;
        s.budget_limit = c.max_triangles;
        s.fair_clamped = c.budget_stats.clamped;
        s.max_instance_scale = c.budget_stats.max_scale;
        s.skin_skipped = c.skin_skipped;
        s.skin_fallbacks = c.admission.last_fallbacks;
        s.skin_denied = c.admission.denied_count() as u32;
        s.shadow_instances = c.counts.shadow_instances;
        s.shadow_clusters = c.counts.shadow_visible;
        s.shadow_triangles = c.counts.shadow_triangles;
        s.skinned_clusters = c.skin.counts.clusters;
        s.skin_vertices = c.skin.counts.used;
        s.skin_pool_overflows = c.skin.pool_overflows;
        s.batches = c.last_batches;
        s.paused = c.disabled_until > self.frame_serial;
        if c.store.stream_active() {
            let st = c.store.stream_stats();
            s.stream_on = true;
            s.stream_pages = st.resident_pages;
            s.stream_loading = st.loading_pages;
            s.stream_resident_mb = st.resident_bytes as f32 / 1048576.0;
            s.stream_budget_mb = st.budget_bytes as f32 / 1048576.0;
            s.stream_requests = st.requests;
            s.stream_shadow_requests = st.shadow_requests;
            s.stream_loads = st.loads;
            s.stream_evictions = st.evictions;
            s.stream_mb_streamed = st.bytes_streamed as f32 / 1048576.0;
            s.stream_stall_frames = st.stall_frames;
            s.stream_pending = st.pending;
            s.stream_skipped = c.counts.stream_skipped;
            s.stream_standins = c.counts.stream_standin;
            s.stream_thrash = st.thrash_reloads;
        }
        s.pool_percent = c
            .store
            .usage()
            .iter()
            .map(|&(used, cap)| (used as u64 * 100 / cap.max(1) as u64) as u32)
            .max()
            .unwrap_or(0);
    }
}

// ---------------------------------------------------------------------------

/// Statistics of the cluster path (read back from the frame slot's previous
/// submission, no CPU wait).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClusterStats {
    /// The path is requested and available.
    pub enabled: bool,
    /// Off for a few frames after an overflow.
    pub paused: bool,
    pub meshes: u32,
    pub batches: u32,
    /// Instances entering the pass / rejected by the instance frustum test.
    pub instances: u32,
    pub instances_culled: u32,
    pub nodes_visited: u32,
    pub clusters_tested: u32,
    /// Tested clusters that are not in the LOD cut.
    pub culled_lod: u32,
    pub culled_frustum: u32,
    pub culled_cone: u32,
    pub culled_hiz: u32,
    /// Hi-Z rejects of the early pass that the late pass drew after all.
    pub recovered: u32,
    pub visible_clusters: u32,
    /// Triangles submitted to the vertex stage (before the hardware culls).
    pub triangles: u32,
    /// Frames in which a queue or list overflowed (and the path paused).
    pub queue_overflows: u32,
    /// Fullest of the four pool arrays (percent).
    pub pool_percent: u32,
    /// Cluster shadows: caster instances culled against the shadow views, and
    /// the clusters / triangles their (coarser) cuts drew into the shadow maps.
    /// Skinned meshes whose ordinary GPU skin pass was skipped this frame (cluster shadows + camera).
    pub skin_skipped: u32,
    /// Skinned instances that drew the ordinary skinned mesh last frame because the skinned-vertex pool was under pressure, and how many are held back.
    pub skin_fallbacks: u32,
    pub skin_denied: u32,
    pub shadow_instances: u32,
    pub shadow_clusters: u32,
    pub shadow_triangles: u32,
    /// Cluster cull plus the skin pass (one GPU zone).
    pub cull_ms: f32,
    /// The main pass's cluster draws (early + late batches).
    pub draw_ms: f32,
    /// The depth prepass's cluster draws.
    pub prepass_ms: f32,
    /// Posing the visible skinned clusters (early + late halves; inside `cull_ms`).
    pub skin_ms: f32,
    /// Skinned clusters posed last frame, and posed-pool vertices reserved.
    pub skinned_clusters: u32,
    pub skin_vertices: u32,
    /// Frames in which the skinned-vertex pool was full.
    pub skin_pool_overflows: u32,
    /// Global LOD error scale the budget controller applies (1 = none), and
    /// the frame time it aims for (ms; 0 = controller off).
    pub error_scale: f32,
    pub target_ms: f32,
    /// Estimated triangles of the instances at that scale, the limit they are
    /// shared out of (0 = none) and how many instances were coarsened further
    /// by the fair share, and the largest scale any instance got.
    pub budget_estimated: u32,
    pub budget_limit: u32,
    pub fair_clamped: u32,
    pub max_instance_scale: f32,
    /// Geometry streaming (GEO-6): on, resident / loading pages (resident bytes
    /// count the pages outside the root pages), the budget, the page requests of
    /// the policy, loads, evictions, bytes streamed, frames a wanted page could
    /// not be admitted, requests waiting, and, from the last cull, clusters on
    /// missing pages and clusters standing in for missing children.
    pub stream_on: bool,
    pub stream_pages: u32,
    pub stream_loading: u32,
    pub stream_resident_mb: f32,
    pub stream_budget_mb: f32,
    pub stream_requests: u64,
    /// Of `stream_requests`, the ones only shadow views asked for.
    pub stream_shadow_requests: u64,
    pub stream_loads: u64,
    pub stream_evictions: u64,
    pub stream_mb_streamed: f32,
    pub stream_stall_frames: u64,
    pub stream_pending: u32,
    pub stream_skipped: u32,
    pub stream_standins: u32,
    /// Pages loaded again soon after an eviction.
    pub stream_thrash: u64,
    /// Which meshes draw from clusters and why the rest do not (set by the engine).
    pub coverage: crate::ClusterCoverage,
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn zestengine_zestlets_is_an_alias_of_zestengine_clusters() {
        assert_eq!(zestlets_env_override(None, None), None);
        assert_eq!(zestlets_env_override(Some("1"), None), Some(true));
        assert_eq!(zestlets_env_override(Some("0"), None), Some(false));
        assert_eq!(zestlets_env_override(None, Some("1")), Some(true));
        assert_eq!(zestlets_env_override(None, Some("0")), Some(false));
        assert_eq!(zestlets_env_override(None, Some("")), Some(false), "empty is off, like the old variable");
        // The new name wins when both are set.
        assert_eq!(zestlets_env_override(Some("1"), Some("0")), Some(true));
        assert_eq!(zestlets_env_override(Some("0"), Some("1")), Some(false));
    }

    /// `n` avatars at 1..n m, each with a coarsest cut of `root` vertices, one frame completed.
    fn admission_with(n: usize, root: u32) -> SkinAdmission {
        let mut a = SkinAdmission::default();
        a.begin_frame();
        for i in 0..n {
            a.note_root_vertices(i, root);
            assert!(a.allowed(i, 1.0 + i as f32, 1));
        }
        a.begin_frame(); // the frame above is now the last completed one
        a
    }

    const POOL: u32 = 1 << 20;

    #[test]
    fn the_cap_follows_the_selected_cut_not_the_pool() {
        let mut a = SkinAdmission::default();
        assert_eq!(a.frame_cap(0, POOL), SKIN_MIN_CAP);
        assert_eq!(a.frame_cap(100_000, POOL), 131_072, "125 000 vertices with headroom, rounded up");
        assert_eq!(a.frame_cap(10, POOL), 131_072, "never shrinks");
        assert_eq!(a.frame_cap(u64::MAX, POOL), POOL, "never above the pool");
        let mut tiny = SkinAdmission::default();
        assert_eq!(tiny.frame_cap(5, 1000), 1000, "a pool smaller than the minimum cap is the cap");
    }

    #[test]
    fn overflow_below_the_max_cap_grows_the_cap_and_denies_nobody() {
        let mut a = admission_with(10, 1000);
        let cap = a.frame_cap(10_000, POOL);
        assert_eq!(cap, SKIN_MIN_CAP);
        // The shader counted 300 000 vertices of demand against 65 536: jump straight past it.
        a.pressure(300_000, cap, POOL, true, 100);
        assert_eq!(a.denied_count(), 0);
        assert_eq!(a.scale(), 1.0, "no coarsening while the pool can still grow");
        assert_eq!(a.frame_cap(10_000, POOL), 524_288, "next power of two over 300 000 x 1.25");
        // Over 60% without an overflow grows too, doubling at least.
        let mut b = admission_with(2, 1000);
        let cap = b.frame_cap(0, POOL);
        b.pressure(cap * 7 / 10, cap, POOL, false, 100);
        assert_eq!(b.frame_cap(0, POOL), cap * 2);
        assert_eq!(b.denied_count(), 0);
    }

    #[test]
    fn at_the_max_cap_the_scale_rises_and_a_pressure_signal_from_before_it_is_ignored() {
        let mut a = admission_with(10, 1000);
        assert_eq!(a.frame_cap(u64::MAX, POOL), POOL);
        a.pressure(POOL, POOL, POOL, true, 100);
        let s = a.scale();
        assert!((1.25..=4.0).contains(&s), "one step is x1.25..4, got {s}");
        assert_eq!(a.denied_count(), 0, "coarsening, not denying");
        a.pressure(POOL, POOL, POOL, true, 101);
        a.pressure(POOL, POOL, POOL, true, 102);
        assert_eq!(a.scale(), s, "frames in flight when the scale rose reported the same overflow");
        a.pressure(POOL, POOL, POOL, true, 200);
        assert!(a.scale() > s, "sustained pressure keeps raising it");
        for k in 0..40 {
            a.pressure(POOL * 2, POOL, POOL, true, 300 + 10 * k);
        }
        assert_eq!(a.scale(), SKIN_MAX_SCALE, "capped");
    }

    /// The factor `apply_skin_admission` gives an instance: scale^f, f from the distance rank.
    fn factor(scale: f32, rank: usize, n: usize) -> f32 {
        scale.powf(if n <= 1 { 1.0 } else { (2.0 * rank as f32 / (n - 1) as f32).min(1.0) })
    }

    #[test]
    fn the_farthest_gets_the_largest_factor_and_the_nearest_none() {
        let mut a = admission_with(10, 1000);
        a.frame_cap(u64::MAX, POOL);
        a.pressure(POOL, POOL, POOL, true, 100);
        let s = a.scale();
        assert!(s > 1.0);
        let f: Vec<f32> = (0..10).map(|r| factor(s, r, 10)).collect();
        assert_eq!(f[0], 1.0, "the nearest is untouched");
        assert!(f.windows(2).all(|w| w[0] <= w[1]), "never finer with distance: {f:?}");
        assert_eq!(f[9], s, "the farthest gets the whole scale");
        assert_eq!(f[5], s, "the far half gets it all");
    }

    #[test]
    fn denial_happens_only_when_the_root_cuts_exceed_the_pool() {
        // 10 avatars with a 100 000 vertex root cut each: 1 000 000 > 0.9 x 1 048 576.
        let mut a = admission_with(10, 100_000);
        a.frame_cap(u64::MAX, POOL);
        let mut serial = 100;
        while a.scale() < SKIN_MAX_SCALE {
            assert_eq!(a.denied_count(), 0, "denied before the scale was exhausted");
            a.pressure(POOL, POOL, POOL, true, serial);
            serial += 10;
        }
        // The step that reached the largest scale denied too: 9 roots fit (900 000 < 943 718), the 10th does not: only the farthest is denied.
        assert_eq!(a.denied_count(), 1);
        assert!(!a.allowed(9, 10.0, serial + 1));
        assert!(a.allowed(0, 1.0, serial + 1));
        // Roots that fit never deny, however hot the pool.
        let mut b = admission_with(10, 5_000);
        b.frame_cap(u64::MAX, POOL);
        for k in 0..60 {
            b.pressure(POOL, POOL, POOL, true, 100 + 10 * k);
        }
        assert_eq!(b.scale(), SKIN_MAX_SCALE);
        assert_eq!(b.denied_count(), 0);
        // A lone instance whose own root cut does not fit goes to the ordinary draw.
        let mut c = admission_with(1, POOL);
        c.frame_cap(u64::MAX, POOL);
        for k in 0..60 {
            c.pressure(POOL, POOL, POOL, true, 100 + 10 * k);
        }
        assert_eq!(c.denied_count(), 1);
    }

    #[test]
    fn the_scale_relaxes_after_calm() {
        let mut a = admission_with(10, 1000);
        a.frame_cap(u64::MAX, POOL);
        a.pressure(POOL, POOL, POOL, true, 100);
        let hot = a.scale();
        // 29 calm frames change nothing, the 30th relaxes by 1.2.
        let mut serial = 110;
        for _ in 0..29 {
            a.pressure(POOL / 4, POOL, POOL, false, serial);
            serial += 1;
        }
        assert_eq!(a.scale(), hot);
        a.pressure(POOL / 4, POOL, POOL, false, serial);
        assert!((a.scale() - (hot / 1.2).max(1.0)).abs() < 1e-5, "{} vs {}", a.scale(), hot / 1.2);
        // A warm frame (between 50% and 85%) resets the calm count without raising the scale.
        let before = a.scale();
        serial += 10;
        a.pressure(POOL * 3 / 5, POOL, POOL, false, serial);
        assert_eq!(a.scale(), before);
        for _ in 0..200 {
            serial += 10;
            a.pressure(POOL / 8, POOL, POOL, false, serial);
        }
        assert_eq!(a.scale(), 1.0, "fully relaxed");
    }

    #[test]
    fn a_denied_instance_returns_when_its_root_cut_fits_again() {
        let mut a = admission_with(10, 100_000);
        a.frame_cap(u64::MAX, POOL);
        let mut serial = 100;
        for _ in 0..60 {
            a.pressure(POOL, POOL, POOL, true, serial);
            serial += 10;
        }
        assert_eq!(a.denied_count(), 1);
        // The root cut of a near instance shrinks (it was freed): the denied one fits again once calm.
        a.note_root_vertices(0, 1000);
        serial += SKIN_DENY_FRAMES + 10;
        a.begin_frame();
        for i in 0..9 {
            assert!(a.allowed(i, 1.0 + i as f32, serial));
        }
        assert!(!a.allowed(9, 10.0, serial));
        a.begin_frame(); // the nine admitted instances are the last completed frame
        for _ in 0..40 {
            a.pressure(POOL / 8, POOL, POOL, false, serial);
            serial += 1;
        }
        assert_eq!(a.denied_count(), 0);
        assert!(a.allowed(9, 10.0, serial));
    }

    #[test]
    fn shadow_view_texel_math_orthographic_and_perspective() {
        // Orthographic cascade 20 m wide, 2048 texels: ~102 texels per metre, constant distance rule.
        let proj = Mat4::orthographic_rh(-10.0, 10.0, -10.0, 10.0, -50.0, 50.0);
        let view = Mat4::look_at_rh(Vec3::new(3.0, 20.0, 5.0), Vec3::new(3.0, 0.0, 5.0), Vec3::Z);
        let v = shadow_cull_view(proj * view, 2048, 2.0);
        assert!((v.lod[1] - 2048.0 / 20.0).abs() < 1e-2, "texels per unit {}", v.lod[1]);
        assert_eq!(v.lod[0], 1.0);
        assert_eq!(v.lod[3], 2.0);
        assert_eq!(v.lod[2], 0.0, "no near-plane test for shadow views");
        // Perspective spot: 90 degree cone, 1024 texels: 512 texels per unit at distance 1; the apex is the light.
        let light = Vec3::new(-2.0, 7.0, 4.0);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 50.0);
        let view = Mat4::look_at_rh(light, Vec3::new(1.0, 0.0, 1.0), Vec3::Y);
        let v = shadow_cull_view(proj * view, 1024, 3.0);
        assert!((v.lod[1] - 512.0).abs() < 1e-2, "texels per unit {}", v.lod[1]);
        assert_eq!(v.lod[0], 0.0);
        for (a, b) in v.cam[..3].iter().zip(light.to_array()) {
            assert!((a - b).abs() < 1e-3, "apex {:?} vs {:?}", v.cam, light);
        }
    }
    use crate::cluster_headless::{HeadlessGpu, HeadlessHiz, HeadlessInstance, HeadlessOptions, HeadlessView};
    use crate::cluster_ref::{self, RefInstance};

    #[test]
    fn range_alloc_grows_and_merges_the_tail() {
        let mut a = RangeAlloc::new(100);
        let x = a.alloc(60).unwrap();
        assert!(a.alloc(70).is_none(), "full");
        a.grow(200);
        assert_eq!(a.capacity(), 200);
        // The free 40 at the end of the old range joins the new 100: one run of 140.
        assert_eq!(a.largest_free(), 140);
        let y = a.alloc(140).unwrap();
        assert_eq!(y, x + 60);
        assert!(a.alloc(1).is_none());
        a.grow(150); // never shrinks
        assert_eq!(a.capacity(), 200);
        a.free(x, 60);
        a.free(y, 140);
        assert_eq!(a.largest_free(), 200);
    }

    #[test]
    fn pool_growth_limits_follow_the_start_size_and_the_device_range() {
        let sizes = PoolSizes::default().clamped(1 << 28);
        let lim = pool_limits(sizes);
        assert_eq!(lim[0], (sizes.clusters * GROW_LIMIT).min((1u32 << 28) / size_of::<ClusterRec>() as u32));
        // 4M vertices x 32 B = 128 MB at the start; four times that is over a 256 MB range.
        assert!(lim[1] as u64 * size_of::<ClusterVertexRec>() as u64 <= 1 << 28);
        assert!(lim[1] >= sizes.vertices);
        // An unclamped size keeps the plain multiple.
        assert_eq!(pool_limits(PoolSizes::default())[2], PoolSizes::default().tri_words * GROW_LIMIT);
    }

    #[test]
    fn range_alloc_reuses_and_coalesces() {
        let mut a = RangeAlloc::new(100);
        let x = a.alloc(30).unwrap();
        let y = a.alloc(30).unwrap();
        let z = a.alloc(30).unwrap();
        assert_eq!((x, y, z), (0, 30, 60));
        assert_eq!(a.alloc(20), None, "only 10 left");
        a.free(y, 30);
        assert_eq!(a.alloc(31), None, "the hole is 30 long");
        a.free(x, 30);
        assert_eq!(a.largest_free(), 60, "x and y merged");
        a.free(z, 30);
        assert_eq!(a.largest_free(), 100);
        assert_eq!(a.used(), 0);
        assert_eq!(a.alloc(100), Some(0));
        assert_eq!(a.alloc(1), None);
    }

    #[test]
    fn range_alloc_random_never_overlaps() {
        let mut a = RangeAlloc::new(4096);
        let mut live: Vec<(u32, u32)> = Vec::new();
        let mut seed = 0x1234_5678u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for _ in 0..5000 {
            if live.len() > 40 || (next() % 3 == 0 && !live.is_empty()) {
                let (s, n) = live.swap_remove(next() as usize % live.len());
                a.free(s, n);
            } else if let Some(s) = a.alloc(1 + next() % 200) {
                let n = a.used() - live.iter().map(|r| r.1).sum::<u32>();
                live.push((s, n));
            }
            let mut sorted = live.clone();
            sorted.sort();
            assert!(sorted.windows(2).all(|w| w[0].0 + w[0].1 <= w[1].0), "ranges overlap");
            assert_eq!(a.used(), live.iter().map(|r| r.1).sum::<u32>());
        }
        for (s, n) in live {
            a.free(s, n);
        }
        assert_eq!(a.largest_free(), 4096, "everything coalesced back");
    }

    #[test]
    fn plan_bounds_regions_by_cluster_counts() {
        let per_instance = [(100, 30, 3), (100, 30, 3), (50, 10, 5)];
        let (plan, regions) = plan_cull(&per_instance, &[200, 50]);
        assert_eq!(plan.levels, 6);
        assert_eq!(regions, vec![(0, 200), (200, 50)]);
        assert!(plan.visible_half >= 250 && plan.node_cap as usize >= MIN_ENTRIES);
        // Past the list budget a region shrinks instead of overrunning it.
        let (_, regions) = plan_cull(&per_instance, &[MAX_VISIBLE_ENTRIES as u64, 10]);
        assert_eq!(regions[0].1 as usize, MAX_VISIBLE_ENTRIES);
        assert_eq!(regions[1].1, 0);
    }

    /// The Rust constants and the shader's agree.
    #[test]
    fn shader_constants_match_rust() {
        let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/cluster_cull.comp")).unwrap();
        let get = |name: &str| -> u32 {
            let line = src.lines().find(|l| l.contains(&format!("const uint {name} ="))).unwrap_or_else(|| panic!("no {name}"));
            line.split('=').nth(1).unwrap().split(';').next().unwrap().trim().trim_end_matches('u').parse().unwrap()
        };
        assert_eq!(get("C_FLAGS") as usize, counter::FLAGS);
        assert_eq!(get("C_QUEUE") as usize, counter::QUEUE);
        assert_eq!(get("C_OCCLUDED") as usize, counter::OCCLUDED);
        assert_eq!(get("C_INSTANCES") as usize, counter::INSTANCES);
        assert_eq!(get("C_INSTANCES_CULLED") as usize, counter::INSTANCES_CULLED);
        assert_eq!(get("C_NODES") as usize, counter::NODES);
        assert_eq!(get("C_TESTED") as usize, counter::TESTED);
        assert_eq!(get("C_LOD") as usize, counter::LOD);
        assert_eq!(get("C_FRUSTUM") as usize, counter::FRUSTUM);
        assert_eq!(get("C_CONE") as usize, counter::CONE);
        assert_eq!(get("C_HIZ") as usize, counter::HIZ);
        assert_eq!(get("C_VISIBLE") as usize, counter::VISIBLE);
        assert_eq!(get("C_TRIANGLES") as usize, counter::TRIANGLES);
        assert_eq!(get("C_OVERFLOW") as usize, counter::OVERFLOW);
        assert_eq!(get("C_LATE_TESTED") as usize, counter::LATE_TESTED);
        assert_eq!(get("C_LATE_RECOVERED") as usize, counter::LATE_RECOVERED);
        assert_eq!(get("C_LATE_OCCLUDED") as usize, counter::LATE_OCCLUDED);
        assert_eq!(get("C_STREAM_SKIPPED") as usize, counter::STREAM_SKIPPED);
        assert_eq!(get("C_STREAM_STANDIN") as usize, counter::STREAM_STANDIN);
        assert_eq!(get("MODE_STREAM"), mode::STREAM);
        assert_eq!(get("STREAM_REQ_CAP") as usize, STREAM_REQ_CAP);
        assert_eq!(get("STREAM_REQ_BASE") as usize, STREAM_REQ_BASE);
        assert_eq!(STREAM_PRIO_BASE, STREAM_REQ_BASE + 2 * STREAM_REQ_CAP);
        assert_eq!(stream_touch_base(1000), STREAM_PRIO_BASE + 1000);
        assert_eq!(get("FLAG_QUEUE"), counter::FLAG_QUEUE);
        assert_eq!(get("FLAG_VISIBLE"), counter::FLAG_VISIBLE);
        assert_eq!(get("FLAG_OCCLUDED"), counter::FLAG_OCCLUDED);
        assert_eq!(get("FLAG_BAD"), counter::FLAG_BAD);
        assert_eq!(get("MODE_SEED"), mode::SEED);
        assert_eq!(get("MODE_NODES"), mode::NODES);
        assert_eq!(get("MODE_LATE"), mode::LATE);
        assert_eq!(get("MODE_FINALIZE"), mode::FINALIZE);
        assert_eq!(get("MODE_ADVANCE"), mode::ADVANCE);
        assert_eq!(get("MODE_ADVANCE_OCC"), mode::ADVANCE_OCC);
        let common = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/cluster_common.glsl")).unwrap();
        assert!(common.contains(&format!("CLUSTER_DRAW_VERTICES = {CLUSTER_DRAW_VERTICES}u")));
        assert!(common.contains(&format!("CLUSTER_MAX_TRIS = {CLUSTER_MAX_TRIS}u")));
        assert!(common.contains("BVH_LEAF = 0x80000000u"));
        assert_eq!(BVH_LEAF, 0x8000_0000);
    }

    /// std430 layouts: field offsets of the records against what
    /// cluster_common.glsl declares.
    #[test]
    fn record_layouts_match_std430() {
        use std::mem::offset_of;
        assert_eq!(offset_of!(ClusterRec, lod_error), 48);
        assert_eq!(offset_of!(ClusterRec, cone), 56);
        assert_eq!(offset_of!(ClusterRec, vertex_offset), 60);
        assert_eq!(offset_of!(ClusterRec, triangle_offset), 64);
        assert_eq!(offset_of!(ClusterRec, vertex_count), 68);
        assert_eq!(offset_of!(ClusterRec, triangle_count), 70);
        assert_eq!(offset_of!(ClusterRec, level), 92);
        assert_eq!(offset_of!(BvhChildRec, max_parent_error), 32);
        assert_eq!(offset_of!(BvhChildRec, child), 36);
        assert_eq!(offset_of!(BvhChildRec, count), 40);
        assert_eq!(offset_of!(MeshRec, b), 16);
        assert_eq!(offset_of!(MeshRec, grid), 32);
        assert_eq!(offset_of!(MeshRec, c), 48);
        assert_eq!(offset_of!(InstRec, color), 64);
        assert_eq!(offset_of!(InstRec, sphere), 96);
        assert_eq!(offset_of!(InstRec, mesh_batch), 112);
        assert_eq!(offset_of!(InstRec, eye), 128);
        assert_eq!(offset_of!(CullPush, mode), 64);
        assert_eq!(offset_of!(ClusterVertexRec, normal), 8);
        assert_eq!(offset_of!(ClusterVertexRec, uv), 16);
    }

    /// The clustered vertex shader links against every fragment variant and
    /// the cluster pipeline layout on a real driver.
    #[test]
    fn cluster_pipelines_compile_on_a_device() {
        let Some(gpu) = HeadlessGpu::new().unwrap() else {
            eprintln!("no Vulkan device: skipping");
            return;
        };
        let made = gpu.check_cluster_pipelines().unwrap();
        eprintln!("{}: {made} clustered pipeline variants created", gpu.device_name());
        assert!(made >= 60);
    }

    // --- a hand-made two-level hierarchy -------------------------------

    /// `n` single-quad clusters on a grid in the XY plane facing +Z, all finest
    /// level (error 0, root parent), under a two-level BVH.
    pub(crate) fn quad_grid(n: usize) -> ClusterMeshUpload {
        let mut up = ClusterMeshUpload { step: 1.0 / 64.0, mesh_min: [-1.0, -1.0, -1.0], ..Default::default() };
        for k in 0..n {
            let (gx, gy) = ((k % 8) as f32, (k / 8) as f32);
            let base = up.vertices.len() as u32;
            let (x0, y0) = (gx * 0.5 - 2.0, gy * 0.5 - 2.0);
            for (dx, dy) in [(0.0, 0.0), (0.4, 0.0), (0.4, 0.4), (0.0, 0.4)] {
                up.vertices.push(ClusterVertexRec {
                    pos: [(((x0 + dx + 2.0) * 64.0).round()) as u16, (((y0 + dy + 2.0) * 64.0).round()) as u16, 64],
                    flags: 0,
                    normal: crate::cluster_ref::oct_encode_z(),
                    tangent: [32767, 0],
                    uv: [dx, dy],
                    color: [255; 4],
                });
            }
            let toff = up.triangles.len() as u32;
            up.triangles.extend([0, 1, 2, 0, 2, 3, 0, 0]);
            up.clusters.push(ClusterRec {
                bounds: [x0 + 0.2, y0 + 0.2, 0.0, 0.3],
                lod_sphere: [x0 + 0.2, y0 + 0.2, 0.0, 0.3],
                parent_lod_sphere: [x0 + 0.2, y0 + 0.2, 0.0, 0.3],
                lod_error: 0.0,
                parent_lod_error: f32::MAX,
                cone: [0, 0, 127, 255],
                vertex_offset: base,
                triangle_offset: toff,
                vertex_count: 4,
                triangle_count: 2,
                ..Default::default()
            });
        }
        up.mesh_min = [-2.0, -2.0, -1.0];
        // Sphere bounding all clusters (a real BVH child bound).
        let all = |first: usize, count: usize| {
            let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
            for c in &up.clusters[first..first + count] {
                for k in 0..2 {
                    lo[k] = lo[k].min(c.bounds[k] - c.bounds[3]);
                    hi[k] = hi[k].max(c.bounds[k] + c.bounds[3]);
                }
            }
            let c = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, 0.0];
            let r = ((hi[0] - lo[0]).hypot(hi[1] - lo[1]) * 0.5) + 0.01;
            [c[0], c[1], c[2], r]
        };
        let leaf = |first: usize, count: usize| BvhChildRec {
            bounds: all(first, count),
            lod_sphere: all(first, count),
            max_parent_error: f32::MAX,
            child: BVH_LEAF | first as u32,
            count: count as u32,
            pad: 0,
        };
        let empty = BvhChildRec { child: BVH_EMPTY, ..Default::default() };
        let inner = |node: u32, first: usize, count: usize| BvhChildRec {
            bounds: all(first, count),
            lod_sphere: all(first, count),
            max_parent_error: f32::MAX,
            child: node,
            count: 0,
            pad: 0,
        };
        let half = n / 2;
        let (a, b) = (half.min(8), half.saturating_sub(8));
        let mut node1 = [empty; 4];
        node1[0] = leaf(0, a);
        if b > 0 {
            node1[1] = leaf(8, b);
        }
        let mut node2 = [empty; 4];
        let rest = n - half;
        node2[0] = leaf(half, rest.min(8));
        if rest > 8 {
            node2[1] = leaf(half + 8, rest - 8);
        }
        up.nodes = vec![
            BvhNodeRec { children: [inner(1, 0, half), inner(2, half, rest), empty, empty] },
            BvhNodeRec { children: node1 },
            BvhNodeRec { children: node2 },
        ];
        up
    }

    #[test]
    fn quad_grid_validates_and_has_depth_one() {
        let up = quad_grid(20);
        up.validate().unwrap();
        assert_eq!(up.max_depth(), 1);
        let mut bad = up.clone();
        bad.nodes[0].children[0].child = 0;
        assert!(bad.validate().is_err(), "a child that is not forward must be rejected");
        let mut bad = up.clone();
        bad.clusters[3].triangle_count = 200;
        assert!(bad.validate().is_err());
        let mut bad = up.clone();
        bad.triangles[1] = 9;
        assert!(bad.validate().is_err(), "triangle index past the vertex count");
        let mut bad = up.clone();
        bad.nodes[1].children[0].count = 99;
        assert!(bad.validate().is_err());
        let mut bad = up;
        bad.skinned = true;
        bad.clusters[0].bone_count = 5;
        assert!(bad.validate().is_err(), "a bone list past the end of the bone array");
    }

    fn quad_view(eye: Vec3, target: Vec3) -> (HeadlessView, crate::cluster_ref::RefView) {
        let rv = cluster_ref::make_view(eye, target, 1.0, 1.5, 1080.0, 0.1, 500.0, 1.0);
        let proj = Mat4::perspective_rh(1.0, 1.5, 0.1, 500.0);
        let vp = proj * Mat4::look_at_rh(eye, target, Vec3::Y);
        (HeadlessView { view_proj: vp, cam: eye, pixels_per_unit: rv.pixels_per_unit, target_px: 1.0 }, rv)
    }

    /// The real shader on a real device against the Rust reference, over a
    /// hand-made DAG, many cameras and instance transforms. Skipped (passes) with
    /// no Vulkan device.
    #[test]
    fn cull_matches_reference_on_a_device() {
        let Some(gpu) = HeadlessGpu::new().unwrap() else {
            eprintln!("no Vulkan device: skipping");
            return;
        };
        eprintln!("device: {}", gpu.device_name());
        let mesh = quad_grid(20);
        let mut seed = 99u32;
        let mut rnd = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        for _ in 0..40 {
            let eye = Vec3::new(rnd() * 8.0 - 4.0, rnd() * 8.0 - 4.0, 2.0 + rnd() * 10.0);
            let target = Vec3::new(rnd() * 4.0 - 2.0, rnd() * 4.0 - 2.0, 0.0);
            let (hv, rv) = quad_view(eye, target);
            let mut insts = Vec::new();
            let mut model_refs = Vec::new();
            for b in 0..3 {
                let yaw = rnd() * std::f32::consts::TAU;
                let model = Mat4::from_translation(Vec3::new(rnd() * 4.0 - 2.0, rnd() * 2.0 - 1.0, rnd() * 3.0 - 1.5))
                    * Mat4::from_rotation_y(yaw)
                    * Mat4::from_scale(Vec3::splat(0.5 + rnd()));
                insts.push(HeadlessInstance { mesh: 0, model, batch: b % 2, skin: None, lod_scale: 1.0 });
                model_refs.push(RefInstance { model: model.to_cols_array_2d(), eye: cluster_ref::object_space_eye(model, eye) });
            }
            let run = gpu.run(std::slice::from_ref(&mesh), &insts, 2, &hv, HeadlessHiz::Off, &HeadlessOptions::default()).unwrap();
            assert!(run.out_of_range_writes.is_empty(), "{:?}", run.out_of_range_writes);
            assert_eq!(run.counters[counter::FLAGS], 0, "no overflow / bad index");
            assert_eq!(run.validation_errors, 0);
            // Expected per (instance, cluster) from the reference (with the instance sphere test).
            let sphere = crate::cluster_headless::mesh_sphere(&mesh);
            let mut want: Vec<Vec<(u32, u32)>> = vec![Vec::new(), Vec::new()];
            for (i, inst) in insts.iter().enumerate() {
                let s = cluster_ref::max_scale(&model_refs[i].model);
                let c = cluster_ref::transform_point(&model_refs[i].model, [sphere[0], sphere[1], sphere[2]]);
                if cluster_ref::outside_frustum(&rv, c, sphere[3] * s) {
                    continue;
                }
                let (cut, _) = cluster_ref::cut(&mesh.clusters, &mesh.nodes, &rv, &model_refs[i], true);
                want[inst.batch as usize].extend(cut.into_iter().map(|c| (i as u32, c)));
            }
            for (b, w) in want.iter_mut().enumerate() {
                w.sort_unstable();
                let mut got = run.early[b].clone();
                got.sort_unstable();
                assert_eq!(&got, w, "batch {b}, eye {eye:?}");
            }
        }
    }
}
