//! zestengine-render-lite: the render-free stand-in for `zestengine-render`.
//!
//! Compiles `zestengine-render/src/types.rs` itself (one source of truth for
//! every data type) next to hand-kept mirrors of the few pure helpers that
//! live beside the Vulkan code there. See each mirror's note.

#[path = "../../zestengine-render/src/types.rs"]
mod types;
pub use types::*;

#[path = "../../zestengine-render/src/cluster_coverage.rs"]
mod cluster_coverage;
pub use cluster_coverage::{
    COVERAGE_OTHER, COVERAGE_ZESTLET, ClusterCoverage, ClusterRefusal, LegendRow, ORDINARY_GREY, REFUSAL_REASONS, VIEW_ZESTLET_COVERAGE, VIEW_ZESTLET_LOD, VIEW_ZESTLET_TRIANGLES,
    VIEW_ZESTLETS, ZESTLET_MIN_TRIANGLES, ZESTLET_VIEW_MODES, is_zestlet_view, zestlet_hash_colour, zestlet_id, zestlet_level_colour,
};

/// Opaque stand-ins for the egui types `types.rs` carries in `EguiDraw`.
pub(crate) mod egui_compat {
    pub struct ClippedPrimitive;
    pub struct TexturesDelta;
}

// Mirror of `zestengine-render/src/cluster.rs::ClusterStats` (a plain data struct
// whose home module is GPU code); `RenderStats` in `types.rs` embeds it.
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

use std::sync::atomic::{AtomicBool, Ordering};
use serde::{Deserialize, Serialize};

pub mod hdr_format;
pub use hdr_format::*;

// ---- mirrors of small pure items in `zestengine-render/src/lib.rs` / `cluster.rs` ----
// (parity: tests/parity.rs)

/// Whether the active device enabled `textureCompressionBC`. Set once at device
/// creation; the texture import cache reads it to decide between BC and an
/// uncompressed fallback. Defaults true (assume supported until told otherwise).
static BC_SUPPORTED: AtomicBool = AtomicBool::new(true);

/// Record BC compression support (called from device creation).
pub fn set_bc_supported(v: bool) {
    BC_SUPPORTED.store(v, Ordering::Relaxed);
}

/// True if the device supports BC compressed textures (see [`set_bc_supported`]).
pub fn bc_supported() -> bool {
    BC_SUPPORTED.load(Ordering::Relaxed)
}

/// `textureCompressionASTC_LDR` enabled on the active device (mobile GPUs).
static ASTC_LDR_SUPPORTED: AtomicBool = AtomicBool::new(false);
/// `textureCompressionASTC_HDR` enabled (VK_EXT_texture_compression_astc_hdr or core 1.3).
static ASTC_HDR_SUPPORTED: AtomicBool = AtomicBool::new(false);

/// Record ASTC support (called from device creation).
pub fn set_astc_supported(ldr: bool, hdr: bool) {
    ASTC_LDR_SUPPORTED.store(ldr, Ordering::Relaxed);
    ASTC_HDR_SUPPORTED.store(hdr, Ordering::Relaxed);
}

/// True if the device samples ASTC LDR blocks (see [`set_astc_supported`]).
pub fn astc_supported() -> bool {
    ASTC_LDR_SUPPORTED.load(Ordering::Relaxed)
}

/// True if the device samples ASTC HDR (float) blocks.
pub fn astc_hdr_supported() -> bool {
    ASTC_HDR_SUPPORTED.load(Ordering::Relaxed)
}

/// Full mip count (down to 1x1) of a `size` cube face.
pub fn cube_mip_count(size: u32) -> u32 {
    32 - size.max(1).leading_zeros()
}

/// Common layer size of a lightmap array holding layers of `sizes`.
pub fn lightmap_array_size(sizes: impl Iterator<Item = u32>) -> u32 {
    sizes.max().unwrap_or(1).clamp(1, 4096)
}

/// Bilinear resample of a square RGBA32F image (`src_size`²) to `dst`². Used to
/// pack per-object lightmaps (varying resolutions) into one uniform texture
/// array. Same-size (or malformed) input is copied/padded.
/// Bilinear resample of a square RGBA32F image (the lightmap array's common
/// layer size). Also used by the bake encoder so stored layers match.
pub fn resample_rgba32f(src: &[f32], src_size: u32, dst: u32) -> Vec<f32> {
    let s = src_size.max(1) as usize;
    let d = dst.max(1) as usize;
    if s == d || src.len() < s * s * 4 {
        let mut out = vec![0.0; d * d * 4];
        let k = out.len().min(src.len());
        out[..k].copy_from_slice(&src[..k]);
        return out;
    }
    let mut out = vec![0.0f32; d * d * 4];
    let smax = (s - 1) as f32;
    for y in 0..d {
        let fy = (y as f32 + 0.5) * s as f32 / d as f32 - 0.5;
        let y0 = fy.floor().clamp(0.0, smax) as usize;
        let y1 = (y0 + 1).min(s - 1);
        let ty = (fy - y0 as f32).clamp(0.0, 1.0);
        for x in 0..d {
            let fx = (x as f32 + 0.5) * s as f32 / d as f32 - 0.5;
            let x0 = fx.floor().clamp(0.0, smax) as usize;
            let x1 = (x0 + 1).min(s - 1);
            let tx = (fx - x0 as f32).clamp(0.0, 1.0);
            for c in 0..4 {
                let p00 = src[(y0 * s + x0) * 4 + c];
                let p10 = src[(y0 * s + x1) * 4 + c];
                let p01 = src[(y1 * s + x0) * 4 + c];
                let p11 = src[(y1 * s + x1) * 4 + c];
                let top = p00 + (p10 - p00) * tx;
                let bot = p01 + (p11 - p01) * tx;
                out[(y * d + x) * 4 + c] = top + (bot - top) * ty;
            }
        }
    }
    out
}

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


// ---- self-contained modules compiled straight from the render crate ----
#[path = "../../zestengine-render/src/tier.rs"]
mod tier;
pub use tier::RenderTier;
#[path = "../../zestengine-render/src/morph.rs"]
mod morph;
pub use morph::{MorphDeltas, MorphTables, morph_reference};
#[path = "../../zestengine-render/src/sdf.rs"]
pub mod sdf;
#[path = "../../zestengine-render/src/cluster_budget.rs"]
pub mod cluster_budget;

#[allow(dead_code)]
mod mirror;
pub use mirror::*;


/// No-op stand-ins for the frame profiler hooks the engine calls from shared code.
pub mod hitch {
    pub struct Scope;
    pub fn scope(_name: &'static str) -> Scope {
        Scope
    }
    pub fn event(_name: &'static str, _tag: &'static str, _val: u64) {}
    pub fn phase(_name: &'static str) {}
}

mod renderer;
pub use renderer::*;

pub use mirror::{Filter as SamplerFilter, Wrap as SamplerWrap};
mod renderer_methods;
