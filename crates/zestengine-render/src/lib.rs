//! zestengine-render: ash-based Vulkan renderer.
//!
//! M2 scope: depth-tested mesh rendering with the zestengine standard shader
//! (phase 1, variant cache via specialization constants), texture/material
//! system, and an egui overlay pass for the in-engine inspector.

mod aa;
mod alloc;
mod ao;
pub use ao::{AoQuality, AoSettings, VIEW_AO};
mod area_light;
mod bloom;
mod exposure;
mod screen_pass;
pub mod cluster;
mod cluster_coverage;
pub use cluster_coverage::{
    COVERAGE_OTHER, COVERAGE_ZESTLET, ClusterCoverage, ClusterRefusal, LegendRow, ORDINARY_GREY, REFUSAL_REASONS, VIEW_ZESTLET_COVERAGE, VIEW_ZESTLET_LOD, VIEW_ZESTLET_TRIANGLES,
    VIEW_ZESTLETS, ZESTLET_MIN_TRIANGLES, ZESTLET_VIEW_MODES, is_zestlet_view, zestlet_hash_colour, zestlet_id, zestlet_level_colour,
};
pub mod cluster_budget;
pub mod cluster_headless;
pub mod cluster_ref;
pub mod cluster_skin;
pub mod cluster_stream;
pub mod cluster_vr;
pub mod impostor;
pub mod ltc;
pub mod offscreen;
pub mod skin_headless;
#[cfg(test)]
mod gpu_tests;
mod bake;
mod context;
mod decal;
mod egui_window;
mod frame;
pub mod frame_limiter;
pub mod hitch;
mod fsr3;
mod gpu_cull;
mod gpu_gi;
mod mesh_sdf;
mod flux_surface;
mod flux_depth;
mod flux_radiance_cache;
mod flux_screen_history;
mod flux_cards;
mod hdr_format;
mod pipeline;
mod vertex_hook;
mod post;
mod resize_debounce;
mod resize_gate;
mod sampler_cache;
mod shadow_debug;
pub use sampler_cache::{Filter as SamplerFilter, SamplerKey, Wrap as SamplerWrap};
mod profiler_window;
pub mod sdf;
mod morph;
mod skin;
// Sky pass support: mipmapped equirect upload, sky sampler, CPU procedural sky.
mod sky;
pub use skin::{CLOTH_PATCH_FLOATS, pack_cloth_patch};
mod rt_reflect;
mod reflection_denoise;
mod rt_water;
mod ssr;
mod surface_lifecycle;
mod shared_bind;
use shared_bind::{SharedTarget, SharedWrite};
mod swapchain;
mod text;
pub mod sprite2d;
mod sprite_gpu;
pub use sprite2d::SPRITE_MATERIAL_XZ;
pub use sprite_gpu::{ANIM_CAP as SPRITE_ANIM_CAP, DEFAULT_SPRITE_MATERIAL, SpriteBlend, SpritePart, SpriteFrameInput, SpriteMaterialDesc};
mod vr_overlay;
mod texture;
mod tier;
mod types;
mod selection;
pub use selection::SelectionOutlineStyle;
mod grid;
pub use grid::{EditorGridStyle, GridLod, GridPlane, grid_lod};
mod sync;
pub use sync::{Access, ImageDesc, Lifetime, Mode, Range, Stages, SyncBatch, Tracker, info, is_write, mode_from_env, resolve_layout};
mod underwater;
mod water_sim;
mod vsm;

/// FidelityFX FSR 3.0.4 FFI (bindgen-generated from the vendored public
/// headers in `third_party/ffx/include`).
#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, unused, clippy::all)]
pub(crate) mod ffx {
    include!(concat!(env!("OUT_DIR"), "/ffx_bindings.rs"));

    // The Vulkan backend entry points live in `host/backends/vk/ffx_vk.h`, which
    // includes `<vulkan/vulkan.h>`; instead of feeding the whole Vulkan C header
    // to bindgen, declare them by hand — Vulkan handles are opaque `void*` here.
    #[repr(C)]
    pub struct VkDeviceContext {
        pub vk_device: *mut ::std::os::raw::c_void,
        pub vk_physical_device: *mut ::std::os::raw::c_void,
        pub vk_device_proc_addr: *mut ::std::os::raw::c_void,
    }

    unsafe extern "C" {
        pub fn ffxGetScratchMemorySizeVK(
            physical_device: *mut ::std::os::raw::c_void,
            max_contexts: usize,
        ) -> usize;
        pub fn ffxGetInterfaceVK(
            backend_interface: *mut FfxInterface,
            device: *mut ::std::os::raw::c_void,
            scratch_buffer: *mut ::std::os::raw::c_void,
            scratch_buffer_size: usize,
            max_contexts: usize,
        ) -> FfxErrorCode;
        pub fn ffxGetDeviceVK(device_context: *mut VkDeviceContext) -> *mut ::std::os::raw::c_void;
        pub fn ffxGetCommandListVK(command_buffer: *mut ::std::os::raw::c_void) -> *mut ::std::os::raw::c_void;
        pub fn ffxGetResourceVK(
            vk_resource: *mut ::std::os::raw::c_void,
            description: FfxResourceDescription,
            name: *mut u32,
            state: FfxResourceStates,
        ) -> FfxResource;
    }
}

/// GPU timestamp zones (a begin/end pair each) for the per-pass profiler
/// breakdown. Zone Z occupies query slots `2*Z` (begin) and `2*Z+1` (end).
/// `FRAME` wraps the whole command buffer; the rest are the major passes of the
/// active path (editor viewport or game). Mutually exclusive passes (e.g. the
/// editor GI vs the game GI) share a zone — only whichever ran writes it.
pub(crate) mod gpu_zone {
    pub const FRAME: usize = 0;
    pub const SHADOWS: usize = 1;
    pub const GI: usize = 2;
    pub const SCENE: usize = 3;
    pub const REFLECT: usize = 4;
    pub const POST: usize = 5;
    pub const EGUI: usize = 6;
    pub const CAM_PREVIEW: usize = 7;
    /// Flux trace sub-passes (inside GI): screen-probe gather, denoise, integrate.
    pub const GI_GATHER: usize = 8;
    pub const GI_DENOISE: usize = 9;
    pub const GI_INTEGRATE: usize = 10;
    /// GPU-driven culling dispatch, and the Hi-Z pyramid build after the scene.
    pub const CULL: usize = 11;
    pub const HIZ: usize = 12;
    /// The late occlusion pass inside the first main-camera depth pass: its
    /// pyramid build + re-test dispatch.
    pub const CULL_LATE: usize = 13;
    /// GPU skinning dispatches and blendshape (morph) dispatches (GEO-1
    /// measurement); `skin::Skinner::record` stamps them directly.
    pub const SKIN: usize = 14;
    pub const MORPH: usize = 15;
    /// Cluster cull dispatches (GEO-4) and the cluster indirect draws (the last
    /// pass that drew them: the main pass).
    pub const CLUSTER_CULL: usize = 16;
    pub const CLUSTER_DRAW: usize = 17;
    /// Posing the visible skinned clusters (inside `CLUSTER_CULL`): the early
    /// half (after the early cull) and the late half (after the late cull).
    pub const CLUSTER_SKIN: usize = 18;
    pub const CLUSTER_SKIN_LATE: usize = 19;
    /// The cluster indirect draws per pass: `CLUSTER_DRAW` / `CLUSTER_DRAW_LATE`
    /// are the main (colour) pass's early and late batches, `CLUSTER_PRE*` the
    /// depth prepass's. (A zone's query pair is written once per frame.)
    pub const CLUSTER_DRAW_LATE: usize = 20;
    pub const CLUSTER_PRE: usize = 21;
    pub const CLUSTER_PRE_LATE: usize = 22;
    /// Screen-space ambient occlusion (`ao.rs`, before the scene pass) and the
    /// anti-aliasing resolve (`aa.rs`: TAA before post, FXAA inside it).
    pub const AO: usize = 23;
    pub const AA: usize = 24;
    /// GPU checkpoint names per zone (see `GpuContext::checkpoint`).
    pub const GI_SURFACE: usize = 25;
    pub const CAMERA_GI: usize = 26;
    pub const CAMERA_GI_GATHER: usize = 27;
    pub const CAMERA_GI_DENOISE: usize = 28;
    pub const CAMERA_GI_INTEGRATE: usize = 29;
    pub const CAMERA_GI_SURFACE: usize = 30;
    pub const BEGIN: [&std::ffi::CStr; 31] = [
        c"frame begin", c"shadows begin", c"GI begin", c"scene begin", c"reflections begin", c"post begin",
        c"UI begin", c"camera preview begin", c"GI gather begin", c"GI denoise begin", c"GI integrate begin",
        c"GPU cull begin", c"Hi-Z build begin", c"late cull begin", c"skinning begin", c"morph begin",
        c"cluster cull begin", c"cluster draw begin", c"cluster skin begin", c"cluster skin late begin",
        c"cluster draw late begin", c"cluster prepass begin", c"cluster prepass late begin",
        c"ambient occlusion begin", c"anti-aliasing begin",
        c"GI surface lighting begin", c"Camera GI begin", c"Camera GI gather begin",
        c"Camera GI denoise begin", c"Camera GI integrate begin", c"Camera GI surface lighting begin",
    ];
    pub const END: [&std::ffi::CStr; 31] = [
        c"frame end", c"shadows end", c"GI end", c"scene end", c"reflections end", c"post end",
        c"UI end", c"camera preview end", c"GI gather end", c"GI denoise end", c"GI integrate end",
        c"GPU cull end", c"Hi-Z build end", c"late cull end", c"skinning end", c"morph end",
        c"cluster cull end", c"cluster draw end", c"cluster skin end", c"cluster skin late end",
        c"cluster draw late end", c"cluster prepass end", c"cluster prepass late end",
        c"ambient occlusion end", c"anti-aliasing end",
        c"GI surface lighting end", c"Camera GI end", c"Camera GI gather end",
        c"Camera GI denoise end", c"Camera GI integrate end", c"Camera GI surface lighting end",
    ];
    pub const COUNT: usize = 31;
    const _: () = assert!(crate::frame::TS_COUNT as usize == 2 * COUNT);
}

/// One GPU timestamp query read back with availability (TYPE_64 layout: the
/// value followed by a non-zero flag when the query was written this frame).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct TsResult {
    pub(crate) value: u64,
    pub(crate) available: u64,
}

/// Milliseconds between two timestamp results, or None when either query was
/// not written. `mask` covers the queue's `timestampValidBits` (the counter wraps
/// at that width and the upper bits are undefined), `period_ns` is ns per tick.
fn ts_zone_ms(begin: TsResult, end: TsResult, period_ns: f64, mask: u64) -> Option<f32> {
    if begin.available == 0 || end.available == 0 {
        return None;
    }
    let ticks = (end.value & mask).wrapping_sub(begin.value & mask) & mask;
    Some((ticks as f64 * period_ns / 1.0e6) as f32)
}

/// `timestampValidBits` -> value mask (64 bits = all ones).
fn ts_valid_mask(valid_bits: u32) -> u64 {
    if valid_bits >= 64 { u64::MAX } else { (1u64 << valid_bits) - 1 }
}

/// Queries to read back for a slot: everything while zone timing is on,
/// otherwise just the always-on FRAME begin/end pair.
fn ts_readback_count(zones_on: bool) -> u32 {
    if zones_on { frame::TS_COUNT } else { 2 }
}

pub use bake::{BakeJob, BakeStep};
pub use egui_window::EguiWindow;
pub use cluster::{stream_max_pages, ClusterMeshUpload, StagedClusterDag, ClusterMode, ClusterSettings, ClusterStats, NO_PAGE, PageLayout, PageLoad, PageStager, StagedPage, StreamLayout};
pub use impostor::{ImpostorHandle, ImpostorUpload};
pub use offscreen::{CaptureTicket, CapturedImpostor, ImpostorCaptureRequest, ModelOnlyPreview, ModelPreviewDepth, neutral_light};
pub use gpu_cull::GpuCullSettings;
pub use gpu_gi::{GpuGiEmitter, GpuGiMarch, GpuGiMaterial};
pub use mesh_sdf::GiMeshSdf;
pub use hdr_format::*;
pub use types::*;
/// The egui types `types.rs` names, behind one alias (see `zestengine-render-lite`).
pub(crate) mod egui_compat {
    pub use egui::{ClippedPrimitive, TexturesDelta};
}
pub use water_sim::{MAX_WATER_DISPLACERS, WaterSimBody, WaterSimEwave, WaterSimReadback};
pub use morph::{MorphDeltas, MorphTables, morph_reference};
pub use pipeline::{CUSTOM_KEYWORDS, CUSTOM_TEXTURES};
pub use tier::RenderTier;

use crate::hitch::WaitNamed as _;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

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

use anyhow::{Context as _, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::Window;

use alloc::Buffer;
use context::GpuContext;
pub use context::GpuCaps;
use frame::{FRAMES_IN_FLIGHT, Frame};
use pipeline::{INSTANCE_STRIDE, InstanceData, PipelineCache, PipelineKey};
use swapchain::Swapchain;
use texture::GpuTexture;

/// Source directory of the engine's built-in shaders, baked in at compile time.
/// Used for dev-time hot-reload so edits to e.g. standard.frag take effect live.
pub const ENGINE_SHADER_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders");

/// The depth-only format: shadow atlases, the opaque depth copy (a copy of the
/// depth aspect only) and every pass that never needs stencil.
const DEPTH_ONLY_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

/// Scene depth(-stencil) format, picked once at device creation
/// (`context::pick_depth_format`): `D32_SFLOAT_S8_UINT`, else `D24_UNORM_S8_UINT`,
/// else `D16_UNORM_S8_UINT`; plain `D32_SFLOAT` (the no-stencil fast path) when
/// no stencil format can be a depth attachment + sampled image, or when
/// `ZESTENGINE_STENCIL=0` forces it. Raw `vk::Format` so it fits an atomic.
static SCENE_DEPTH_FORMAT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(vk::Format::D32_SFLOAT.as_raw());

pub(crate) fn set_depth_format(f: vk::Format) {
    SCENE_DEPTH_FORMAT.store(f.as_raw(), Ordering::Relaxed);
}

/// The scene depth format (see [`SCENE_DEPTH_FORMAT`]).
pub(crate) fn depth_format() -> vk::Format {
    vk::Format::from_raw(SCENE_DEPTH_FORMAT.load(Ordering::Relaxed))
}

/// Whether the scene depth has a stencil aspect (the opt-in stencil path).
pub fn depth_has_stencil() -> bool {
    sprite_gpu::has_stencil(depth_format())
}

/// Aspect of a scene-depth image for barriers, attachment views and clears:
/// DEPTH, plus STENCIL when the format has it. Samplers use `DEPTH` alone.
pub(crate) fn depth_aspect() -> vk::ImageAspectFlags {
    if depth_has_stencil() { vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL } else { vk::ImageAspectFlags::DEPTH }
}

/// Attachment layout of the scene depth. A combined depth-stencil image moves
/// both aspects together (`DEPTH_ATTACHMENT_OPTIMAL` is invalid when the
/// barrier's aspect mask includes stencil).
pub(crate) fn depth_attach_layout() -> vk::ImageLayout {
    if depth_has_stencil() { vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL } else { vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL }
}

/// Format of a copy of the scene depth's depth aspect (the opaque-copy / grab
/// depth): the scene depth format itself. `vkCmdCopyImage` between depth /
/// stencil formats needs identical formats; copying a D32_SFLOAT_S8_UINT depth
/// aspect into a D32_SFLOAT image is invalid and on NVIDIA lands interleaved
/// (every other column held stencil / padding), which striped the water's
/// refraction depth. Only the depth aspect is copied and viewed; layout
/// transitions of the copy cover [`depth_aspect`].
pub(crate) fn depth_copy_format() -> vk::Format {
    depth_format()
}

/// The stencil attachment that pairs with a scene-depth attachment: the same
/// view, layout, load / store ops (stencil cleared to 0), or "no attachment"
/// (null view) when the depth format has no stencil. Call as
/// `.depth_attachment(&d).stencil_attachment(&stencil_attachment(&d))`.
pub(crate) fn stencil_attachment<'a>(depth: &vk::RenderingAttachmentInfo<'a>) -> vk::RenderingAttachmentInfo<'a> {
    stencil_attachment_for(depth, depth_has_stencil())
}

/// [`stencil_attachment`] for a depth target whose format is known to have stencil (`has`) or not.
pub(crate) fn stencil_attachment_for<'a>(depth: &vk::RenderingAttachmentInfo<'a>, has: bool) -> vk::RenderingAttachmentInfo<'a> {
    if !has {
        return vk::RenderingAttachmentInfo::default();
    }
    let mut s = *depth;
    s.clear_value = vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 } };
    s
}
/// Deferred-SSR G-buffer attachment: reflectance.rgb (Fresnel*ao*spec weight)
/// in the colour lanes, surface roughness in alpha. RGBA16F so the resolve pass
/// reads HDR-safe reflectance with no banding.
/// Reflection G-buffer: 10-bit octahedral normal (rg) + roughness / reflectivity
/// packed 6+6 bits into b (10) and a (2) (see `gbuf_pack` in standard.frag).
const GBUF_FORMAT: vk::Format = vk::Format::A2B10G10R10_UNORM_PACK32;
const MAX_MATERIALS: u32 = 1024;
/// set-1 sampler binding indices in texture-slot order (binding 4 is the FX UBO).
/// Set-1 binding of the area-light emission atlas (shared by every material set).
const AREA_ATLAS_BINDING: u32 = 22;
/// Set-1 binding of the LTC matrix table (24 x 24 RGBA32F, `ltc.rs`), shared by every material set.
/// A texture, not a GLSL constant array: a dynamically indexed constant array is copied into
/// per-invocation local memory at shader entry, which cost every pixel of every material.
const LTC_LUT_BINDING: u32 = 23;

const TEXTURE_BINDINGS: [u32; MATERIAL_SLOTS] =
    [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];
/// Index into `default_textures` of the neutral 1x1 for texture slot `i`.
/// Slots 0..16 map to themselves (16 / 17 are the custom-shader black / gray
/// defaults); the detail slots follow those.
fn default_texture_index(slot: usize) -> usize {
    if slot < 16 { slot } else { 18 + (slot - 16) }
}
/// Per-material custom-shader std140 block size (matches
/// zestengine_assets::SHADER_UBO_MAX_BYTES).
const CUSTOM_UBO_BYTES: usize = 4096;

/// Maximum scene lights evaluated per frame by the standard shader. Matches
/// `MAX_LIGHTS` in standard.frag; the array lives in the frame UBO.
// Forward-pass light cap. Lives in the frame UBO, so it's bounded by the 16 KB
// guaranteed uniform-buffer size: 128 GpuLights = 8 KB, leaving headroom. Truly
// unbounded would need a storage buffer (like the screen-GI path already uses).
const MAX_LIGHTS: usize = 128;

/// One light in std140 layout (four vec4s). Mirrors `Light` in standard.frag.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct GpuLight {
    /// xyz world position, w = kind (0 directional, 1 point, 2 spot).
    pos_kind: [f32; 4],
    /// xyz travel direction (normalized), w = range.
    dir_range: [f32; 4],
    /// rgb color premultiplied by intensity, w = cos(outer half-angle).
    color: [f32; 4],
    /// x = cos(inner half-angle); yzw reserved.
    spot: [f32; 4],
}

/// Pack a scene light into its std140 GPU representation.
fn gpu_light(l: &LightInstance) -> GpuLight {
    let (kind, cos_outer, cos_inner) = match l.kind {
        LightKind::Directional => (0.0, -1.0, 1.0),
        LightKind::Point => (1.0, -1.0, 1.0),
        LightKind::Spot => (
            2.0,
            (l.spot_outer_deg.to_radians() * 0.5).cos(),
            (l.spot_inner_deg.to_radians() * 0.5).cos(),
        ),
    };
    let d = l.direction.normalize_or_zero();
    GpuLight {
        pos_kind: [l.position.x, l.position.y, l.position.z, kind],
        dir_range: [d.x, d.y, d.z, l.range],
        color: [
            l.color[0] * l.intensity,
            l.color[1] * l.intensity,
            l.color[2] * l.intensity,
            cos_outer,
        ],
        // spot = [cos_inner, shadow_base_layer (-1 = none), bias, view_count].
        // The shadow planner patches y/z/w when this light casts shadows. For a
        // light that does NOT cast shadows (y stays -1) we reuse w as a "baked"
        // flag (this light's contribution is in lightmaps; the shader skips it for
        // lightmapped objects). Detected as `spot.y < 0 && spot.w > 0`, so it can
        // never be confused with a shadow light's view_count (which has y >= 0).
        spot: [cos_inner, -1.0, 0.0, if l.baked { 1.0 } else { 0.0 }],
    }
}

/// Shadow view-projection matrices, one per shadow-map layer.
const MAX_SHADOW_VIEWS: usize = MAX_SHADOW_MAPS;

/// Max probe volumes in the FrameUbo: the baked cascades first (at most `MAX_BAKED_VOLUMES`),
/// then the Light Volumes.
const MAX_PROBE_VOLUMES: usize = 8;
const MAX_BAKED_VOLUMES: usize = 4;

/// One probe volume's runtime metadata in std140 layout. Mirrors
/// `ProbeVolume` in standard.frag.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct GpuProbeVolume {
    /// World → volume-local; local grid spans -size/2..+size/2.
    world_to_local: [[f32; 4]; 4],
    /// xyz = local box size, w = first probe index (sh_base) as float.
    size_base: [f32; 4],
    /// xyz = probe counts per axis (as floats), w unused.
    counts: [f32; 4],
    /// Light Volume mode: x = edge fade distance (m; 0 = a classic cascade volume),
    /// y = blend weight, zw unused.
    params: [f32; 4],
}

/// One probe's SH-L1 in the GPU storage buffer (std430): 4 coefficients, each
/// an RGB in xyz. The `w` of each carries the SH-L1 of the directional
/// distance-to-geometry (`ProbeSh::dist`) for visibility weighting.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct GpuProbe {
    // c[0..3].xyz = RGB SH-L1; c[0..3].w = distance SH-L1 (mean, visibility);
    // c[4].xyzw = squared-distance SH-L1 (second moment, two-moment Chebyshev).
    // c[5].xyz = world offset from the grid position (relocated probes).
    coeffs: [[f32; 4]; 6],
}

/// Pack a `ProbeSh` into its GPU form: RGB SH in xyz, distance SH in the w lanes,
/// squared-distance SH in the 5th vec4 (for two-moment Chebyshev visibility).
fn gpu_probe(p: &ProbeSh) -> GpuProbe {
    GpuProbe {
        coeffs: [
            [p.coeffs[0][0], p.coeffs[0][1], p.coeffs[0][2], p.dist[0]],
            [p.coeffs[1][0], p.coeffs[1][1], p.coeffs[1][2], p.dist[1]],
            [p.coeffs[2][0], p.coeffs[2][1], p.coeffs[2][2], p.dist[2]],
            [p.coeffs[3][0], p.coeffs[3][1], p.coeffs[3][2], p.dist[3]],
            [p.dist2[0], p.dist2[1], p.dist2[2], p.dist2[3]],
            [p.offset[0], p.offset[1], p.offset[2], 0.0],
        ],
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUbo {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    light_dir: [f32; 4],
    light_color: [f32; 4],
    ambient: [f32; 4],
    misc: [f32; 4], // x = time, y = light count, z = shadow spacing, w = probe-volume count
    // Post-processing (per-pixel, in the surface shaders). After `misc` so the
    // skybox's truncated FrameData prefix can read them too.
    postfx0: [f32; 4], // x tonemap mode, y exposure EV, z grade exposure, w contrast
    postfx1: [f32; 4], // x saturation, y temperature, z tint, w grading enabled
    postfx2: [f32; 4], // x vignette enabled, y intensity, z smoothness, w screen w
    postfx3: [f32; 4], // xyz vignette color, w screen h
    /// Far view-space distance of each directional cascade (xyzw = up to 4).
    cascade_splits: [f32; 4],
    lights: [GpuLight; MAX_LIGHTS],
    shadow_vp: [[[f32; 4]; 4]; MAX_SHADOW_VIEWS],
    /// Baked probe volumes (count in `misc.w`); SH coefficients live in the
    /// set-0 binding-2 storage buffer, indexed via each volume's `size_base.w`.
    probe_volumes: [GpuProbeVolume; MAX_PROBE_VOLUMES],
    /// x = lightmap-UV-checker preview flag (>0.5 on); yzw reserved.
    debug: [f32; 4],
    /// Inverse view / projection — for reconstructing world position from a
    /// depth sample (screen-space GI). Appended last so existing shaders that
    /// read a prefix of FrameData are unaffected.
    inv_view: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    /// Screen-space reflections: x = enabled (>0.5), y = intensity,
    /// z = max ray distance (view-space units), w = roughness cutoff (no SSR above).
    ssr: [f32; 4],
    /// Reflection-probe zone: xyz = CAPTURE position (world, the parallax
    /// re-vector anchor), w = intensity (0 = no probe).
    refl_center: [f32; 4],
    /// Reflection-probe zone: xyz = box half-extents (world), w = box-projection on.
    refl_extents: [f32; 4],
    /// Fog: xyz = colour, w = density (0 = no fog).
    fog_color: [f32; 4],
    /// Fog: x = height falloff, y = height ref (world Y), z = start distance, w unused.
    fog_params: [f32; 4],
    /// Reflection-probe zone: xyz = box CENTER (world; capture + box offset),
    /// w unused. Appended last so shaders that don't read it stay compatible.
    refl_box: [f32; 4],
    /// Second reflection probe (slot 1, binding 9), same layout as refl_*;
    /// refl2_center.w = 0 means no second probe.
    refl2_center: [f32; 4],
    refl2_extents: [f32; 4],
    refl2_box: [f32; 4],
    /// x = slot 0 blend distance, y = slot 1 blend distance, z = slot that wins
    /// where both boxes overlap.
    refl_blend: [f32; 4],
    /// Third/fourth reflection probes (bindings 11/12), same layout as refl_*.
    refl3_center: [f32; 4],
    refl3_extents: [f32; 4],
    refl3_box: [f32; 4],
    refl4_center: [f32; 4],
    refl4_extents: [f32; 4],
    refl4_box: [f32; 4],
    /// x = slot 2 blend distance, y = slot 3 blend distance.
    refl_blend2: [f32; 4],
    /// x = water clock, y = the opaque scene copies (bindings 5/6) are valid
    /// for this pass, z = water body count.
    water: [f32; 4],
    water_bodies: [GpuWater; MAX_WATER_BODIES],
    /// Previous frame's jittered view-projection (motion-vector reprojection).
    prev_view_proj: [[f32; 4]; 4],
    /// Per-eye cameras for a multiview XR pass (`MULTIVIEW` shaders pick one by
    /// gl_ViewIndex); zero elsewhere.
    eye_view: [[[f32; 4]; 4]; 2],
    eye_view_proj: [[[f32; 4]; 4]; 2],
    eye_camera_pos: [[f32; 4]; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PushData {
    model: [[f32; 4]; 4],
    base_color: [f32; 4],
    emission: [f32; 4],
    params0: [f32; 4],
    params1: [f32; 4],
}

const _: () = assert!(size_of::<PushData>() == pipeline::PUSH_CONSTANT_SIZE as usize);

/// A run of consecutive scene draws recorded as one instanced draw.
struct SceneBatch {
    pipeline: vk::Pipeline,
    material: usize,
    mesh: usize,
    /// Push constants of the first draw (model carries the union light mask).
    push: PushData,
    /// Instance index of the first draw (its position in the pass order).
    first: u32,
    count: u32,
    batchable: bool,
    /// GPU-culled batch: (indirect command buffer, command offset, compacted
    /// instance buffer, instance region offset).
    indirect: Option<(vk::Buffer, u64, vk::Buffer, u64)>,
}

/// State last bound while recording scene draws.
struct BoundDraw {
    pipeline: vk::Pipeline,
    material: vk::DescriptorSet,
    mesh: usize,
}

impl Default for BoundDraw {
    fn default() -> Self {
        Self { pipeline: vk::Pipeline::null(), material: vk::DescriptorSet::null(), mesh: usize::MAX }
    }
}
// The frame UBO must fit the Vulkan-guaranteed 16 KB uniform-buffer range (it's
// the floor across all GPUs). Mainly guards MAX_LIGHTS / future field growth.
const _: () = assert!(size_of::<FrameUbo>() <= 16384);

/// Push constants for the shadow depth pass: the light's view-projection and
/// the object's model matrix (same 128-byte block, different interpretation).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShadowPush {
    light_vp: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
}

const _: () = assert!(size_of::<ShadowPush>() == pipeline::PUSH_CONSTANT_SIZE as usize);

/// One shadow-map render: which atlas layer + the light view-projection.
struct ShadowView {
    layer: usize,
    view_proj: Mat4,
}

/// Allocate shadow-map layers to shadow-casting lights, compute their
/// view-projections, and patch each light's shadow info into `lights[*].spot`.
/// Returns the per-view render list and the packed `shadow_vp` matrices.
fn plan_shadows(
    input: &FrameInput,
    lights: &mut [GpuLight; MAX_LIGHTS],
    count: usize,
    cascades: CascadeCfg,
) -> (Vec<ShadowView>, [[[f32; 4]; 4]; MAX_SHADOW_VIEWS], [f32; 4]) {
    let mut views = Vec::new();
    let mut shadow_vp = [Mat4::IDENTITY.to_cols_array_2d(); MAX_SHADOW_VIEWS];
    let mut cascade_splits = [0.0f32; 4];
    let mut next_layer = 0usize;
    let mut cascaded_done = false;
    let mut skipped = 0usize;
    let cam_center = input.camera.position;

    // Allocation is independent of camera position. Changing the view must
    // never switch a local light between shadowed and completely unshadowed.
    // Reserve directional lights first, then stable brightness/position order.
    let mut order: Vec<(usize, u8, f32, [u32; 3])> = (0..count.min(input.lights.len()))
        .filter(|&li| input.lights[li].cast_shadows)
        .map(|li| {
            let l = &input.lights[li];
            (li, if l.kind == LightKind::Directional { 0 } else { 1 },
             l.intensity, l.position.to_array().map(f32::to_bits))
        })
        .collect();
    order.sort_by(|a, b| a.1.cmp(&b.1).then(b.2.total_cmp(&a.2)).then(a.3.cmp(&b.3)));

    for (li, ..) in order {
        let l = &input.lights[li];
        let kind = lights[li].pos_kind[3]; // 0 dir, 1 point, 2 spot
        let dir = l.direction.normalize_or(Vec3::NEG_Y);
        let up = if dir.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
        let range = l.range.max(1.0);

        // The first shadow-casting directional light is cascaded.
        let cascaded = kind == 0.0 && !cascaded_done;
        let needed = if kind == 1.0 {
            6
        } else if cascaded {
            cascades.count
        } else {
            1
        };
        if next_layer + needed > MAX_SHADOW_MAPS {
            skipped += 1;
            continue; // out of shadow slots
        }

        let mats: Vec<Mat4> = if cascaded {
            let (mats, splits) = cascade_matrices(input, dir, up, cascades);
            cascade_splits = splits;
            cascaded_done = true;
            mats
        } else if kind == 0.0 {
            // Extra directional (non-cascaded): one tight box ahead of camera.
            let dist = input.shadow_distance.clamp(2.0, 500.0);
            let s = dist * 0.5;
            let cam_fwd = input
                .camera
                .view
                .inverse()
                .transform_vector3(Vec3::NEG_Z)
                .normalize_or(Vec3::NEG_Z);
            let center = cam_center + cam_fwd * s;
            let eye = center - dir * dist;
            let view = Mat4::look_to_rh(eye, dir, up);
            let proj = Mat4::orthographic_rh(-s, s, -s, s, 0.05, dist * 2.0);
            vec![proj * view]
        } else if kind == 2.0 {
            // Spot: perspective along the cone.
            let fov = l
                .spot_outer_deg
                .to_radians()
                .clamp(0.1, std::f32::consts::PI - 0.05);
            let view = Mat4::look_to_rh(l.position, dir, up);
            let proj = Mat4::perspective_rh(fov, 1.0, 0.1, range);
            vec![proj * view]
        } else {
            // Point: 6 cube faces (+X,-X,+Y,-Y,+Z,-Z), matching the shader's
            // face pick. The FOV is a touch wider than 90° so neighbouring
            // faces overlap: a fragment near a face boundary then still has
            // valid depth in its selected face's map. Without the overlap the
            // shader's PCF taps spill into the atlas border and show a bright
            // seam between faces.
            let faces = [
                (Vec3::X, Vec3::NEG_Y),
                (Vec3::NEG_X, Vec3::NEG_Y),
                (Vec3::Y, Vec3::Z),
                (Vec3::NEG_Y, Vec3::NEG_Z),
                (Vec3::Z, Vec3::NEG_Y),
                (Vec3::NEG_Z, Vec3::NEG_Y),
            ];
            let fov = std::f32::consts::FRAC_PI_2 * 1.12; // ~101°, ~11° overlap
            let proj = Mat4::perspective_rh(fov, 1.0, 0.1, range);
            faces
                .iter()
                .map(|(fwd, up)| proj * Mat4::look_to_rh(l.position, *fwd, *up))
                .collect()
        };

        let base = next_layer;
        for (k, m) in mats.iter().enumerate() {
            let layer = base + k;
            shadow_vp[layer] = m.to_cols_array_2d();
            views.push(ShadowView {
                layer,
                view_proj: *m,
            });
        }
        lights[li].spot[1] = base as f32;
        lights[li].spot[2] = l.shadow_bias;
        // View count in spot.w; its sign carries the filter mode (positive =
        // soft/PCF, negative = hard/single-tap), so no extra GPU field is
        // needed. The shader reads abs() for the count.
        lights[li].spot[3] = if l.soft_shadows { needed as f32 } else { -(needed as f32) };
        next_layer += needed;
    }
    // Lights past the atlas budget draw WITHOUT shadows: their light passes
    // straight through walls. Warn once per budget change so the scene author
    // can spot it (the fix: fewer shadow-casting lights, or Spot/Directional
    // instead of Point - each point light costs 6 atlas slots).
    static WARNED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let prev = WARNED.swap(skipped, std::sync::atomic::Ordering::Relaxed);
    if skipped != prev {
        if skipped > 0 {
            tracing::warn!(
                "{skipped} shadow-casting light(s) exceed the {MAX_SHADOW_MAPS}-slot shadow atlas and draw WITHOUT shadows (their light goes through walls)"
            );
        } else {
            tracing::info!("all shadow-casting lights fit the shadow atlas again");
        }
    }
    (views, shadow_vp, cascade_splits)
}

/// Default blend of the cascade split scheme (0 uniform .. 1 logarithmic).
pub(crate) const DEFAULT_CASCADE_LAMBDA: f32 = 0.7;

/// How the main directional light's cascades are fit (Environment
/// `shadow_cascades` / `shadow_cascade_lambda`, and the shadow map size).
#[derive(Clone, Copy, Debug)]
pub(crate) struct CascadeCfg {
    /// Cascade count, 1..=`NUM_CASCADES`.
    pub count: usize,
    /// Split blend, 0 uniform .. 1 logarithmic.
    pub lambda: f32,
    /// Shadow map size (texels per side); cascade centres snap to its texel grid.
    pub dim: u32,
}

impl CascadeCfg {
    /// The scene's request resolved against the tier's cascade ceiling:
    /// `shadow_cascades` 0 = the ceiling, `shadow_cascade_lambda` 0 = the default.
    pub(crate) fn resolve(input: &FrameInput, tier_max: usize, dim: u32) -> Self {
        let max = tier_max.clamp(1, NUM_CASCADES);
        let count = if input.shadow_cascades == 0 { max } else { (input.shadow_cascades as usize).clamp(1, max) };
        let l = input.shadow_cascade_lambda;
        let lambda = if l.is_finite() && l > 0.0 { l.min(1.0) } else { DEFAULT_CASCADE_LAMBDA };
        Self { count, lambda, dim: dim.max(1) }
    }
}

/// World size of one shadow texel at `p` for the light view-projection `vp`
/// (`dim` texels per side), measured the way `shadow_factor` in standard.frag
/// sizes its offsets (a 5 cm step across the light direction `to_light`'s
/// perpendicular, in shadow uv). The cluster shadow cut (`shadow_cull_view`)
/// must agree with it: its `shadow_px` target is in these texels.
#[cfg(test)]
pub(crate) fn shadow_texel_metres(vp: Mat4, p: Vec3, to_light: Vec3, dim: u32) -> f32 {
    let side = to_light.cross(if to_light.y.abs() < 0.99 { Vec3::Y } else { Vec3::X }).normalize();
    let ndc = |q: Vec3| {
        let c = vp * q.extend(1.0);
        glam::Vec2::new(c.x, c.y) / c.w.max(1e-4)
    };
    let uv_per_m = (ndc(p + side * 0.05) - ndc(p)).length() * 0.5 / 0.05;
    (1.0 / dim as f32) / uv_per_m.max(1e-6)
}

/// Farthest view-space depth any draw's bounding sphere reaches, i.e. how far
/// shadows can be received at all; `INFINITY` when nothing bounds it (no
/// draws, or a draw with no bounds). Cascades never reach past this, so a
/// small scene under a big `shadow_distance` keeps its texel density.
fn receiver_depth_limit(input: &FrameInput) -> f32 {
    if input.draws.is_empty() {
        return f32::INFINITY;
    }
    let view = input.camera.view;
    let mut far = 0.0f32;
    for d in input.draws {
        if d.bound_radius <= 0.0 {
            return f32::INFINITY;
        }
        let max_scale = [d.transform.x_axis, d.transform.y_axis, d.transform.z_axis]
            .iter()
            .map(|c| c.truncate().length())
            .fold(0.0f32, f32::max);
        let c = view.transform_point3(d.transform.transform_point3(d.mesh_center));
        far = far.max(-c.z + d.bound_radius * max_scale);
    }
    far
}

/// Round `d` up onto a 1.25x geometric ladder (2, 2.5, 3.1, ...), so the
/// cascade extent changes only when the scene grows past a rung instead of on
/// every movement of its farthest object (a change rescales the maps once).
fn quantize_up(d: f32) -> f32 {
    let steps = (d.max(2.0) / 2.0).ln() / 1.25f32.ln();
    2.0 * 1.25f32.powf((steps - 1e-4).ceil().max(0.0))
}

/// Fit `cfg.count` (at most `NUM_CASCADES`) directional shadow view-projections
/// to depth slices of the camera frustum (near->`shadow_distance`, or the
/// scene's receiver depth when that is closer), each as a bounding sphere for
/// rotation-stable, tight coverage. The sphere's centre snaps to the shadow
/// map's texel grid in light space, so a moving camera slides the maps by
/// whole texels and edges do not shimmer. Returns the matrices and the far
/// view-space distance of each cascade (for selection in the shader).
/// Off-axis projections (an XR view enclosing both eyes) shift the slices
/// with the frustum.
fn cascade_matrices(input: &FrameInput, dir: Vec3, up: Vec3, cfg: CascadeCfg) -> (Vec<Mat4>, [f32; 4]) {
    let cascades = cfg.count.clamp(1, NUM_CASCADES);
    let near = 0.5f32;
    let far = quantize_up(receiver_depth_limit(input)).min(input.shadow_distance.clamp(2.0, 500.0)).max(near + 0.5);
    let lambda = cfg.lambda;
    let inv_view = input.camera.view.inverse();
    let p = input.camera.proj;
    let tan_v = 1.0 / p.y_axis.y; // tan(fovy/2)
    let tan_h = 1.0 / p.x_axis.x; // tan(fovx/2)
    // Frustum centre offset (zero for a symmetric projection).
    let off_x = p.z_axis.x / p.x_axis.x;
    let off_y = p.z_axis.y / p.y_axis.y;
    // Slice corners in VIEW space: the sphere's radius is then a function of the
    // projection and the split alone (bit-identical every frame), the texel size
    // constant while the camera moves and turns.
    let corners_at = |d: f32| -> [Vec3; 4] {
        let h = d * tan_v;
        let w = d * tan_h;
        let (cx, cy) = (d * off_x, d * off_y);
        [
            Vec3::new(cx - w, cy + h, -d),
            Vec3::new(cx + w, cy + h, -d),
            Vec3::new(cx + w, cy - h, -d),
            Vec3::new(cx - w, cy - h, -d),
        ]
    };
    // Light-space axes through the origin: centres snap in this basis.
    let basis = Mat4::look_to_rh(Vec3::ZERO, dir, up);

    let mut splits = [far; 4];
    let mut mats = Vec::with_capacity(cascades);
    let mut prev = near;
    for (i, slot) in splits.iter_mut().enumerate().take(cascades) {
        let si = (i + 1) as f32 / cascades as f32;
        let uniform = near + (far - near) * si;
        let logd = near * (far / near).powf(si);
        let split = uniform * (1.0 - lambda) + logd * lambda;
        *slot = split;
        // Bounding sphere of this slice's 8 frustum corners.
        let near_c = corners_at(prev);
        let far_c = corners_at(split);
        let mut center_v = Vec3::ZERO;
        for c in near_c.iter().chain(far_c.iter()) {
            center_v += *c;
        }
        center_v /= 8.0;
        let mut radius = 0.0f32;
        for c in near_c.iter().chain(far_c.iter()) {
            radius = radius.max((*c - center_v).length());
        }
        // One texel of slack, so a snapped centre still encloses the slice.
        radius = radius.max(0.01) * (1.0 + 2.0 / cfg.dim as f32);
        let mut center = inv_view.transform_point3(center_v);
        // Snap the centre's light-space x / y to whole texels (2 r / dim each).
        let texel = 2.0 * radius / cfg.dim as f32;
        let mut ls = basis.transform_point3(center);
        ls.x = (ls.x / texel).round() * texel;
        ls.y = (ls.y / texel).round() * texel;
        center = basis.inverse().transform_point3(ls);
        let eye = center - dir * (radius * 2.0);
        let view = Mat4::look_to_rh(eye, dir, up);
        let proj = Mat4::orthographic_rh(-radius, radius, -radius, radius, 0.01, radius * 4.0);
        mats.push(proj * view);
        prev = split;
    }
    (mats, splits)
}

struct GpuMesh {
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    /// Width of the indices in `index_buffer` (16 bit when the mesh fits and
    /// ray tracing, which reads 32 bit indices from the buffer, is off).
    index_type: vk::IndexType,
    index_count: u32,
    vertex_count: u32,
    /// Local bounds used for camera-independent Flux surface captures.
    bounds: [Vec3; 2],
    surface_geometry: flux_cards::MeshCards,
}

/// The index buffer bytes for `data`, and their type: 16 bit when every index
/// fits and nothing (the RT bake and reflections) reads the buffer as 32 bit.
fn index_bytes(data: &MeshData, rt: bool) -> (std::borrow::Cow<'_, [u8]>, vk::IndexType) {
    if !rt && data.vertices.len() <= 1 << 16 {
        let narrow: Vec<u16> = data.indices.iter().map(|&i| i as u16).collect();
        (std::borrow::Cow::Owned(bytemuck::cast_slice(&narrow).to_vec()), vk::IndexType::UINT16)
    } else {
        (std::borrow::Cow::Borrowed(bytemuck::cast_slice(&data.indices)), vk::IndexType::UINT32)
    }
}

/// Pre-uploaded GPU resources for a scene, produced off-thread by [`Uploader`]
/// and consumed by [`Renderer::install_prepared`]. Opaque to callers, and `Send`
/// (Vulkan handles + `Allocation`) so it can cross the thread boundary.
pub struct PreparedScene {
    meshes: Vec<GpuMesh>,
    textures: Vec<GpuTexture>,
    skinned: Vec<bool>,
}

/// Background asset uploader bound to the transfer queue + its own command pool.
/// `Send` — move it to a loader thread, `prepare` there, then `install_prepared`
/// on the main thread. A second VkQueue means transfers run concurrently with
/// the render thread; same queue family means no resource ownership transfer.
pub struct Uploader {
    device: ash::Device,
    allocator: Arc<Mutex<Allocator>>,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    rt: bool,
}

impl Uploader {
    /// Upload meshes + (already compressed) textures on the transfer queue.
    /// `has_skeleton` mirrors the main-thread path: a mesh with skin weights +
    /// a skeleton becomes a host-visible skinned mesh.
    pub fn prepare(
        &self,
        meshes: &[MeshData],
        textures: &[CompressedTexture],
        has_skeleton: bool,
    ) -> Result<PreparedScene> {
        let gpu_textures = {
            let mut alloc = self.allocator.lock().unwrap();
            textures
                .iter()
                .map(|t| GpuTexture::upload_compressed(&self.device, &mut alloc, self.command_pool, self.queue, t))
                .collect::<Result<Vec<_>>>()?
        };
        let mut gpu_meshes = Vec::with_capacity(meshes.len());
        let mut skinned = Vec::with_capacity(meshes.len());
        for mesh in meshes {
            let is_skinned = has_skeleton
                && mesh.vertices.iter().any(|v| v.weights.iter().sum::<f32>() > 0.0);
            gpu_meshes.push(if is_skinned {
                self.upload_skinned(mesh)?
            } else {
                self.upload_static(mesh)?
            });
            skinned.push(is_skinned);
        }
        Ok(PreparedScene {
            meshes: gpu_meshes,
            textures: gpu_textures,
            skinned,
        })
    }

    /// Upload pre-compressed (BC) textures on the transfer queue — the material
    /// texture path, where decode + block compression already ran on the loader
    /// thread (or were read straight from the import cache). Returns a
    /// textures-only `PreparedScene` to be installed on the main thread.
    pub fn prepare_compressed(&self, textures: &[CompressedTexture]) -> Result<PreparedScene> {
        let gpu_textures = {
            let mut alloc = self.allocator.lock().unwrap();
            textures
                .iter()
                .map(|t| {
                    GpuTexture::upload_compressed(
                        &self.device,
                        &mut alloc,
                        self.command_pool,
                        self.queue,
                        t,
                    )
                })
                .collect::<Result<Vec<_>>>()?
        };
        Ok(PreparedScene {
            meshes: Vec::new(),
            textures: gpu_textures,
            skinned: Vec::new(),
        })
    }

    /// Validate a cluster DAG and copy it into a host-visible staging buffer on
    /// this (loader) thread; the main thread installs it with
    /// `Renderer::install_staged_cluster_dag`, which only reserves pool ranges
    /// and lets the next frame record the copy (no submit, no wait).
    pub fn stage_cluster_dag(&self, dag: &ClusterMeshUpload) -> Result<StagedClusterDag> {
        StagedClusterDag::new(&self.device, &mut self.allocator.lock().unwrap(), dag)
    }

    fn upload_static(&self, data: &MeshData) -> Result<GpuMesh> {
        let rt = if self.rt {
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::STORAGE_BUFFER
        } else {
            vk::BufferUsageFlags::empty()
        };
        let mut alloc = self.allocator.lock().unwrap();
        let vertex_buffer = alloc::upload_buffer(
            &self.device, &mut alloc, self.command_pool, self.queue,
            bytemuck::cast_slice(&data.vertices),
            vk::BufferUsageFlags::VERTEX_BUFFER | rt, "vertices",
        )?;
        let (index_data, index_type) = index_bytes(data, self.rt);
        let index_buffer = alloc::upload_buffer(
            &self.device, &mut alloc, self.command_pool, self.queue,
            &index_data,
            vk::BufferUsageFlags::INDEX_BUFFER | rt, "indices",
        )?;
        Ok(GpuMesh {
            vertex_buffer,
            index_buffer,
            index_type,
            index_count: data.indices.len() as u32,
            vertex_count: data.vertices.len() as u32,
            bounds: flux_surface::vertex_bounds(&data.vertices),
            surface_geometry: flux_cards::MeshCards::new(data, false),
        })
    }

    fn upload_skinned(&self, data: &MeshData) -> Result<GpuMesh> {
        let mut alloc = self.allocator.lock().unwrap();
        let mut vertex_buffer = Buffer::new(
            &self.device, &mut alloc,
            std::mem::size_of_val(&data.vertices[..]) as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER, MemoryLocation::CpuToGpu, "skinned vertices",
        )?;
        vertex_buffer.write(0, bytemuck::cast_slice(&data.vertices));
        let index_buffer = alloc::upload_buffer(
            &self.device, &mut alloc, self.command_pool, self.queue,
            bytemuck::cast_slice(&data.indices),
            vk::BufferUsageFlags::INDEX_BUFFER, "skinned indices",
        )?;
        Ok(GpuMesh {
            vertex_buffer,
            index_buffer,
            index_type: vk::IndexType::UINT32,
            index_count: data.indices.len() as u32,
            vertex_count: data.vertices.len() as u32,
            bounds: flux_surface::vertex_bounds(&data.vertices),
            surface_geometry: flux_cards::MeshCards::new(data, true),
        })
    }
}

impl Drop for Uploader {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

/// Unity-style render queue: draws are ordered by this number, with named
/// breakpoints and gaps for fine-tuning. Transparent (≥ TRANSPARENT) sorts
/// back-to-front within the queue.
pub const RENDER_QUEUE_GEOMETRY: i32 = 2000;
pub const RENDER_QUEUE_ALPHA_TEST: i32 = 2450;
pub const RENDER_QUEUE_TRANSPARENT: i32 = 3000;

/// Default render queue for a material given its alpha mode.
pub fn default_render_queue(alpha: AlphaMode) -> i32 {
    match alpha {
        AlphaMode::Opaque => RENDER_QUEUE_GEOMETRY,
        AlphaMode::Cutout => RENDER_QUEUE_ALPHA_TEST,
        AlphaMode::Blend | AlphaMode::Transclipping => RENDER_QUEUE_TRANSPARENT,
    }
}

/// Frustum- and occlusion-cull a draw order against the camera. Returns the
/// subset of `order` not marked `occluded` whose world bounding sphere
/// intersects the view frustum, preserving the input
/// ordering (so the opaque→transparent sort is kept). Draws with `bound_radius`
/// 0 are always kept; looking at empty sky correctly draws nothing.
fn frustum_cull(order: &[usize], draws: &[DrawCmd], view_proj: Mat4) -> Vec<usize> {
    let planes = frustum_planes(view_proj);
    order
        .iter()
        .copied()
        .filter(|&i| !draws[i].occluded && sphere_in_planes(&planes, &draws[i], true))
        .collect()
}

/// Gribb-Hartmann plane extraction from view-proj (clip depth 0..1, glam
/// column-major: `row(i)` is the i-th row), normalized so `plane.xyz·c +
/// plane.w` is a signed distance. Order: left, right, bottom, top, near, far.
fn frustum_planes(vp: Mat4) -> [glam::Vec4; 6] {
    let (r0, r1, r2, r3) = (vp.row(0), vp.row(1), vp.row(2), vp.row(3));
    let raw = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2];
    std::array::from_fn(|i| {
        let p = raw[i];
        let n = p.truncate().length();
        if n > 1e-6 { p / n } else { p }
    })
}

/// Whether a draw's world bounding sphere touches the frustum. `near` off keeps
/// casters behind a shadow view's near plane (they still cast into it).
/// Bits of the lights in `lights` (xyz position, w range, < 0 = everywhere)
/// whose range reaches draw `d`'s bounding sphere, as the four 32-bit words the
/// standard shader reads from `pc.model[0]`.
fn draw_light_mask(lights: &[[f32; 4]], d: &DrawCmd) -> [u32; 4] {
    let sphere = (d.bound_radius > 0.0).then(|| {
        let max_scale = [d.transform.x_axis, d.transform.y_axis, d.transform.z_axis]
            .iter()
            .map(|c| c.truncate().length())
            .fold(0.0f32, f32::max);
        (d.transform.transform_point3(d.mesh_center), d.bound_radius * max_scale)
    });
    sphere_light_mask(lights, sphere)
}

/// Light mask of a world sphere (None = unbounded: every light).
fn sphere_light_mask(lights: &[[f32; 4]], sphere: Option<(glam::Vec3, f32)>) -> [u32; 4] {
    let mut mask = [0u32; 4];
    for (i, l) in lights.iter().enumerate().take(MAX_LIGHTS) {
        let reaches = match sphere {
            Some((c, r)) if l[3] >= 0.0 => c.distance(glam::Vec3::new(l[0], l[1], l[2])) <= r + l[3],
            _ => true,
        };
        if reaches {
            mask[i / 32] |= 1 << (i % 32);
        }
    }
    mask
}

/// Whether draw `d` renders into shadow maps: its Cast Shadows isn't Off and
/// its material is solid (blended and water surfaces cast no shadow).
fn shadow_caster(d: &DrawCmd, f: &MaterialFeatures) -> bool {
    d.shadow_casting.casts() && !matches!(f.alpha_mode, AlphaMode::Blend | AlphaMode::Transclipping) && !f.water
}

fn sphere_in_planes(planes: &[glam::Vec4; 6], d: &DrawCmd, near: bool) -> bool {
    if d.bound_radius <= 0.0 {
        return true;
    }
    // World sphere: object-space center transformed, radius scaled by the
    // largest axis (covers non-uniform scale conservatively).
    let center = d.transform.transform_point3(d.mesh_center);
    let max_scale = [d.transform.x_axis, d.transform.y_axis, d.transform.z_axis]
        .iter()
        .map(|c| c.truncate().length())
        .fold(0.0f32, f32::max);
    let r = d.bound_radius * max_scale;
    // Outside iff fully beyond any one plane.
    !planes
        .iter()
        .enumerate()
        .any(|(k, p)| (near || k != 4) && p.truncate().dot(center) + p.w < -r)
}

/// Pipeline variant of a scene draw of `material` in a pass of `mode`.
fn scene_pipeline_key(material: &Material, gbuf: bool, velocity: bool, mirrored: bool, mode: DrawMode, transparent: bool) -> PipelineKey {
    let mut key = if material.error {
        PipelineKey::error()
    } else {
        let mut key = PipelineKey::from_features(&material.features);
        if let Some(shader) = material.custom_shader {
            key.shader = shader.0 as u32 + 1;
            key.spec = material.custom_keywords;
        }
        key
    };
    // Displaced material: tessellation stages in every colour and depth pass that draws it
    // (the pipeline cache drops the bit again on a device without tessellation).
    key.tess = !material.error && material.params.tess.enabled;
    key.gbuf = gbuf;
    key.velocity = velocity;
    key.mirrored = mirrored;
    key.prepass = mode == DrawMode::Prepass;
    key.depth_equal = mode == DrawMode::DepthEqual && !transparent;
    if key.prepass {
        key.gbuf = false;
        // Stencil is a main-pass effect: the prepass only lays depth.
        key.stencil = StencilState::default();
    }
    key
}

/// Pipeline selection for `record_scene_draws`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DrawMode {
    Normal,
    /// Depth-only prepass (transparent draws skipped).
    Prepass,
    /// Main pass over the prepass depth: non-transparent draws test EQUAL.
    DepthEqual,
}

struct Material {
    #[allow(dead_code)] // used by the editor's material list
    name: String,
    params: MaterialParams,
    features: MaterialFeatures,
    /// Draw-order priority (Unity render queue).
    render_queue: i32,
    has_normal_texture: bool,
    error: bool,
    /// Custom fragment shader; None = standard shader.
    custom_shader: Option<ShaderId>,
    /// Custom-shader push data (15 props + reserved lightmap layer).
    custom_data: [f32; 16],
    /// Custom-shader keyword values (pipeline specialization).
    custom_keywords: [u8; pipeline::CUSTOM_KEYWORDS],
    /// Set 2 (custom material block + texture properties); null until the
    /// material first gets a custom shader.
    custom_set: vk::DescriptorSet,
    custom_ubo: Option<Buffer>,
    /// Bound custom texture views per binding (skip redundant rebinds).
    custom_views: [vk::ImageView; pipeline::CUSTOM_TEXTURES],
    /// Per-material inverted-hull outline (linear colour, width), drawn right
    /// after the object in colour passes.
    outline: Option<([f32; 4], f32)>,
    /// Fur shell pass drawn after the surface (UNITY-12).
    fur: Option<FurParams>,
    set: vk::DescriptorSet,
    /// Extended "FX" uniform block (set 1, binding 4); host-visible, rewritten on
    /// param change. Holds rim + animated-emission params for the built-in shaders.
    fx_ubo: Buffer,
    /// Mean linear colour of the albedo texture ([1,1,1] without one). Ray-
    /// traced hits fall back to base colour x this when the albedo texture
    /// isn't in the RT texture table.
    albedo_mean: [f32; 3],
    /// Mean ORM metalness channel; GI cannot sample the map at each hit.
    metallic_mean: f32,
    /// Bound texture per slot (create_material order: albedo, normal, orm,
    /// emission, ...), so ray-traced hits can sample the same textures.
    slots: [Option<TextureHandle>; MATERIAL_SLOTS],
    /// Water materials: which `FrameInput::water` body they draw.
    water_slot: u32,
}

/// Most cameras that can render into textures at once.
const MAX_CAMERA_TARGETS: usize = 8;

/// A camera's render-to-texture target: linear HDR colour registered in the
/// texture table (materials sample it as a screen), its depth, and per-frame
/// set-0 copies with their own UBO.
struct CameraTarget {
    id: u128,
    texture: TextureHandle,
    extent: vk::Extent2D,
    depth: DepthTarget,
    sets: Vec<vk::DescriptorSet>,
    ubos: Vec<Buffer>,
}

struct DepthTarget {
    image: vk::Image,
    /// Depth-aspect view: what shaders sample (Hi-Z, SSR, fog, RT).
    view: vk::ImageView,
    /// Attachment view (depth + stencil aspects when the format has stencil;
    /// the same view as `view` otherwise).
    att_view: vk::ImageView,
    /// R32 color-aspect view of the same depth image — the temporal upscaler's
    /// depth input (FSR samples a 32-bit float depth, not the depth attachment).
    sample_view: vk::ImageView,
    allocation: Option<Allocation>,
}

impl DepthTarget {
    fn new(ctx: &GpuContext, allocator: &mut Allocator, extent: vk::Extent2D) -> Result<Self> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(depth_format())
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { ctx.device.create_image(&info, None)? };
        let requirements = unsafe { ctx.device.get_image_memory_requirements(image) };
        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "depth",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .with_context(|| "allocating GPU memory for depth")?;
        unsafe {
            ctx.device
                .bind_image_memory(image, allocation.memory(), allocation.offset())?
        };
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(depth_format())
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .level_count(1)
                    .layer_count(1),
            );
        let view = unsafe { ctx.device.create_image_view(&view_info, None)? };
        // The attachment view carries every aspect (depth + stencil); `view`
        // above is the depth-only sampling view (a descriptor view may name
        // exactly one aspect).
        let att_view = if depth_has_stencil() {
            unsafe { ctx.device.create_image_view(&view_info.subresource_range(
                vk::ImageSubresourceRange::default().aspect_mask(depth_aspect()).level_count(1).layer_count(1),
            ), None)? }
        } else {
            view
        };
        // D32_SFLOAT is view-compatible with R32_SFLOAT: a color-aspect view
        // exposes the depth as a sampleable float texture for the upscaler. A
        // depth-stencil image is not R32-compatible, so there the upscaler is
        // handed the depth-aspect view itself.
        let sample_view = if depth_has_stencil() {
            view
        } else {
            let sample_view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R32_SFLOAT)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            unsafe { ctx.device.create_image_view(&sample_view_info, None)? }
        };
        Ok(Self {
            image,
            view,
            att_view,
            sample_view,
            allocation: Some(allocation),
        })
    }

    #[allow(dead_code)] // consumed by the FSR upscaler dispatch (wired next).
    fn sample_view(&self) -> vk::ImageView {
        self.sample_view
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            if self.att_view != self.view {
                device.destroy_image_view(self.att_view, None);
            }
            if self.sample_view != self.view {
                device.destroy_image_view(self.sample_view, None);
            }
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }
        if let Some(allocation) = self.allocation.take() {
            let _ = allocator.free(allocation);
        }
    }
}

/// Motion-vector attachment (RG16F) the main scene pass writes, sampled by the
/// temporal upscaler for reprojection. One per frame in flight.
struct VelocityTarget {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<Allocation>,
}

impl VelocityTarget {
    fn new(device: &ash::Device, allocator: &mut Allocator, extent: vk::Extent2D) -> Result<Self> {
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
        let (image, allocation) = create_image(device, allocator, pipeline::VELOCITY_FORMAT, usage, extent, "motion vectors")?;
        let view = create_view(device, image, pipeline::VELOCITY_FORMAT, vk::ImageAspectFlags::COLOR)?;
        Ok(Self { image, view, allocation: Some(allocation) })
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }
        if let Some(allocation) = self.allocation.take() {
            let _ = allocator.free(allocation);
        }
    }
}

/// Reactive-mask attachment (R8) marking transparent pixels, sampled by the
/// temporal upscaler so it resets history on reactive geometry. One per frame.
struct ReactiveTarget {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<Allocation>,
}

impl ReactiveTarget {
    fn new(device: &ash::Device, allocator: &mut Allocator, extent: vk::Extent2D) -> Result<Self> {
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
        let (image, allocation) = create_image(device, allocator, pipeline::REACTIVE_FORMAT, usage, extent, "reactive mask")?;
        let view = create_view(device, image, pipeline::REACTIVE_FORMAT, vk::ImageAspectFlags::COLOR)?;
        Ok(Self { image, view, allocation: Some(allocation) })
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }
        if let Some(allocation) = self.allocation.take() {
            let _ = allocator.free(allocation);
        }
    }
}

/// Full-resolution HDR target the FSR upscaler writes (UAV) and the post pass
/// then samples for tonemap. One per frame in flight.
struct UpscaleOutput {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<Allocation>,
}

impl UpscaleOutput {
    fn new(device: &ash::Device, allocator: &mut Allocator, extent: vk::Extent2D, format: vk::Format) -> Result<Self> {
        let usage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT;
        let (image, allocation) = create_image(device, allocator, format, usage, extent, "fsr upscale output")?;
        let view = create_view(device, image, format, vk::ImageAspectFlags::COLOR)?;
        Ok(Self { image, view, allocation: Some(allocation) })
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }
        if let Some(allocation) = self.allocation.take() {
            let _ = allocator.free(allocation);
        }
    }
}

/// Copies of the main pass's opaque result (HDR colour + depth), taken
/// before its transparent draws so water can refract and fade by depth.
struct OpaqueCopy {
    extent: vk::Extent2D,
    color: vk::Image,
    color_view: vk::ImageView,
    color_alloc: Option<Allocation>,
    depth: vk::Image,
    depth_view: vk::ImageView,
    depth_alloc: Option<Allocation>,
    /// Still UNDEFINED: the next frame moves both to shader-read first.
    fresh: bool,
    /// Mip levels of `color`: level 0 is the copy, the rest a box-filtered
    /// chain built (only when a transmission material is drawn) for rough
    /// transmission's blur.
    mips: u32,
}

/// Cap on the opaque colour copy's mip chain (1/128 of the target is blur enough).
const OPAQUE_MAX_MIPS: u32 = 8;

/// Mip count of an opaque colour copy of `extent`.
fn opaque_mip_count(extent: vk::Extent2D) -> u32 {
    (32 - extent.width.max(extent.height).max(1).leading_zeros()).clamp(1, OPAQUE_MAX_MIPS)
}

impl OpaqueCopy {
    fn new(device: &ash::Device, allocator: &mut Allocator, extent: vk::Extent2D) -> Result<Self> {
        let usage = vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED;
        let mips = opaque_mip_count(extent);
        let (color, color_alloc) = create_image_mips(
            device, allocator, post::HDR_FORMAT, usage | vk::ImageUsageFlags::TRANSFER_SRC, extent, mips, "opaque colour copy",
        )?;
        let color_view = create_view_mips(device, color, post::HDR_FORMAT, vk::ImageAspectFlags::COLOR, mips)?;
        let (depth, depth_alloc) = create_image(device, allocator, depth_copy_format(), usage, extent, "opaque depth copy")?;
        let depth_view = create_view(device, depth, depth_copy_format(), vk::ImageAspectFlags::DEPTH)?;
        Ok(Self {
            extent,
            color,
            color_view,
            color_alloc: Some(color_alloc),
            depth,
            depth_view,
            depth_alloc: Some(depth_alloc),
            fresh: true,
            mips,
        })
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_image_view(self.color_view, None);
            device.destroy_image(self.color, None);
            device.destroy_image_view(self.depth_view, None);
            device.destroy_image(self.depth, None);
        }
        for a in [self.color_alloc.take(), self.depth_alloc.take()].into_iter().flatten() {
            let _ = allocator.free(a);
        }
    }
}

/// Tiles per row / column of the area-light emission atlas, and a tile's side in texels.
const AREA_COLS: u32 = 4;
const AREA_ROWS: u32 = 2;
const AREA_TILE: u32 = 256;
/// Mip levels of the atlas (a tile's chain ends at one texel).
const AREA_MIPS: u32 = 9;

/// The area lights' emission textures: one RGBA16F atlas of `AREA_COLS x AREA_ROWS`
/// tiles (`MAX_AREA_LIGHTS` of them) with a mip chain, bound at set-1 binding 22 of
/// every material set. Each frame a textured light's texture is sampled into its
/// tile and the tile's chain is rebuilt, so a camera / video texture is live.
struct AreaAtlas {
    image: vk::Image,
    view: vk::ImageView,
    base_view: vk::ImageView,
    copy_layout: vk::DescriptorSetLayout,
    copy_pipeline_layout: vk::PipelineLayout,
    copy_pipeline: vk::Pipeline,
    copy_pool: vk::DescriptorPool,
    copy_sets: Vec<vk::DescriptorSet>,
    alloc: Option<Allocation>,
    /// Still UNDEFINED: the first frame moves it to shader-read.
    fresh: bool,
}

impl AreaAtlas {
    fn new(device: &ash::Device, allocator: &mut Allocator) -> Result<Self> {
        let extent = vk::Extent2D { width: AREA_COLS * AREA_TILE, height: AREA_ROWS * AREA_TILE };
        let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC;
        let (image, alloc) = create_image_mips(device, allocator, post::HDR_FORMAT, usage, extent, AREA_MIPS, "area light atlas")?;
        let view = create_view_mips(device, image, post::HDR_FORMAT, vk::ImageAspectFlags::COLOR, AREA_MIPS)?;
        let base_view = create_view_mips(device, image, post::HDR_FORMAT, vk::ImageAspectFlags::COLOR, 1)?;
        let stage=vk::ShaderStageFlags::COMPUTE;
        let bindings=[vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_count(1).stage_flags(stage)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_count(1).stage_flags(stage)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)];
        let copy_layout=unsafe {device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),None)?};
        let copy_pipeline_layout=unsafe {device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[copy_layout])
            .push_constant_ranges(&[vk::PushConstantRange::default().stage_flags(stage).size(8)]),None)?};
        let module=screen_pass::shader_module(device,include_bytes!(concat!(env!("OUT_DIR"),"/area_atlas.comp.spv")))?;
        let pipeline=unsafe {device.create_compute_pipelines(vk::PipelineCache::null(),&[vk::ComputePipelineCreateInfo::default()
            .layout(copy_pipeline_layout).stage(vk::PipelineShaderStageCreateInfo::default().stage(stage).module(module).name(c"main"))],None)};
        unsafe {device.destroy_shader_module(module,None)};
        let copy_pipeline=pipeline.map_err(|(_,e)|e)?[0];
        let n=(FRAMES_IN_FLIGHT*MAX_AREA_LIGHTS) as u32;
        let sizes=[vk::DescriptorPoolSize::default().ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(n),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(n)];
        let copy_pool=unsafe {device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(n).pool_sizes(&sizes),None)?};
        let copy_sets=unsafe {device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(copy_pool)
            .set_layouts(&vec![copy_layout;n as usize]))?};
        Ok(Self { image, view, base_view, copy_layout, copy_pipeline_layout, copy_pipeline, copy_pool, copy_sets, alloc: Some(alloc), fresh: true })
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_descriptor_pool(self.copy_pool,None);
            device.destroy_pipeline(self.copy_pipeline,None);
            device.destroy_pipeline_layout(self.copy_pipeline_layout,None);
            device.destroy_descriptor_set_layout(self.copy_layout,None);
            device.destroy_image_view(self.base_view,None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }
        if let Some(a) = self.alloc.take() {
            let _ = allocator.free(a);
        }
    }
}

/// The LTC matrix table as a 24 x 24 RGBA32F texture (x = view angle, y = roughness, rgba =
/// a, b, c, d of `area_light.glsl`), read with `texelFetch` through a nearest sampler.
struct LtcLut {
    tex: GpuTexture,
    sampler: vk::Sampler,
}

impl LtcLut {
    fn new(device: &ash::Device, allocator: &mut Allocator, pool: vk::CommandPool, queue: vk::Queue) -> Result<Self> {
        let table = ltc::embedded_table();
        anyhow::ensure!(table.len() == ltc::LUT_N * ltc::LUT_N, "ltc_lut.glsl holds {} cells", table.len());
        let texels: Vec<[f32; 4]> = table.iter().map(|c| [c.a, c.b, c.c, c.d]).collect();
        let tex = GpuTexture::upload_pixels(
            device,
            allocator,
            pool,
            queue,
            ltc::LUT_N as u32,
            ltc::LUT_N as u32,
            vk::Format::R32G32B32A32_SFLOAT,
            bytemuck::cast_slice(&texels),
        )?;
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::NEAREST)
                    .min_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?
        };
        Ok(Self { tex, sampler })
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe { device.destroy_sampler(self.sampler, None) };
        self.tex.destroy(device, allocator);
    }
}

/// Targets of a main camera pass, to resume it after the opaque copy.
struct MainPass {
    color_image: vk::Image,
    color_view: vk::ImageView,
    gbuf_view: Option<vk::ImageView>,
    velocity_view: Option<vk::ImageView>,
    reactive_view: Option<vk::ImageView>,
    depth_image: vk::Image,
    /// Attachment view (depth + stencil); `depth_read_view` is the sampled one.
    depth_view: vk::ImageView,
    depth_read_view: vk::ImageView,
    extent: vk::Extent2D,
}

fn add_draw_stats(a: &mut RenderStats, b: &RenderStats) {
    a.draw_calls += b.draw_calls;
    a.instanced_draws += b.instanced_draws;
    a.opaque_draws += b.opaque_draws;
    a.transparent_draws += b.transparent_draws;
    a.outline_draws += b.outline_draws;
    a.error_draws += b.error_draws;
    a.pipeline_binds += b.pipeline_binds;
    a.materials_drawn += b.materials_drawn;
}

/// Create a GPU-only 2D image + its backing allocation.
fn create_image(
    device: &ash::Device,
    allocator: &mut Allocator,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    extent: vk::Extent2D,
    name: &str,
) -> Result<(vk::Image, Allocation)> {
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.create_image(&info, None)? };
    tracing::debug!(?image, name, "allocated render image");
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let allocation = allocator.allocate(&AllocationCreateDesc {
        name,
        requirements,
        location: MemoryLocation::GpuOnly,
        linear: false,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    })
    .with_context(|| format!("allocating GPU memory for {name}"))?;
    unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };
    Ok((image, allocation))
}

/// [`create_image`] with `layers` array layers.
fn create_image_layers(
    device: &ash::Device,
    allocator: &mut Allocator,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    extent: vk::Extent2D,
    layers: u32,
    name: &str,
) -> Result<(vk::Image, Allocation)> {
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 })
        .mip_levels(1)
        .array_layers(layers.max(1))
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.create_image(&info, None)? };
    tracing::debug!(?image, name, "allocated render image");
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let allocation = allocator.allocate(&AllocationCreateDesc {
        name,
        requirements,
        location: MemoryLocation::GpuOnly,
        linear: false,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    })
    .with_context(|| format!("allocating GPU memory for {name}"))?;
    unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };
    Ok((image, allocation))
}

/// A view of layers `base..base + count` (2D for one layer, else 2D array).
fn create_view_layers(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
    base: u32,
    count: u32,
) -> Result<vk::ImageView> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(if count > 1 { vk::ImageViewType::TYPE_2D_ARRAY } else { vk::ImageViewType::TYPE_2D })
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .level_count(1)
                .base_array_layer(base)
                .layer_count(count),
        );
    Ok(unsafe { device.create_image_view(&info, None)? })
}

/// [`create_image`] with `mip_levels` mip levels.
fn create_image_mips(
    device: &ash::Device,
    allocator: &mut Allocator,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    extent: vk::Extent2D,
    mip_levels: u32,
    name: &str,
) -> Result<(vk::Image, Allocation)> {
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 })
        .mip_levels(mip_levels)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.create_image(&info, None)? };
    tracing::debug!(?image, name, "allocated render image");
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let allocation = allocator.allocate(&AllocationCreateDesc {
        name,
        requirements,
        location: MemoryLocation::GpuOnly,
        linear: false,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    })
    .with_context(|| format!("allocating GPU memory for {name}"))?;
    unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };
    Ok((image, allocation))
}

/// [`create_view`] over `levels` mip levels.
fn create_view_mips(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
    levels: u32,
) -> Result<vk::ImageView> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(aspect).level_count(levels).layer_count(1));
    Ok(unsafe { device.create_image_view(&info, None)? })
}

/// Image layout transition over mip levels `base..base + levels`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn image_barrier_mips(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    base: u32,
    levels: u32,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
) {
    let barrier = vk::ImageMemoryBarrier2::default()
        .image(image)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .subresource_range(
            vk::ImageSubresourceRange::default().aspect_mask(aspect).base_mip_level(base).level_count(levels).layer_count(1),
        );
    let barriers = [barrier];
    let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
    unsafe { device.cmd_pipeline_barrier2(cb, &dependency) };
}

fn create_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
) -> Result<vk::ImageView> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .level_count(1)
                .layer_count(1),
        );
    Ok(unsafe { device.create_image_view(&info, None)? })
}

/// Number of mip levels for a square cubemap of `size` (for roughness prefilter:
/// mip 0 = mirror, higher mips = rougher).
#[allow(dead_code)] // reflection-probe capture (in progress) — foundation landed
/// Full mip count (down to 1x1) of a `size` cube face.
pub fn cube_mip_count(size: u32) -> u32 {
    32 - size.max(1).leading_zeros()
}

/// Create a cube-compatible color image (6 layers) with a roughness mip chain.
/// Usage covers rendering the faces (COLOR_ATTACHMENT), prefiltering by blit
/// (TRANSFER_SRC|DST), and sampling at runtime (SAMPLED).
#[allow(dead_code)] // reflection-probe capture (in progress) — foundation landed
fn create_cube_image(
    device: &ash::Device,
    allocator: &mut Allocator,
    format: vk::Format,
    size: u32,
    name: &str,
) -> Result<(vk::Image, Allocation, u32)> {
    let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
        | vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST;
    create_cube_image_usage(device, allocator, format, size, usage, name)
}

/// [`create_cube_image`] with explicit usage: block-compressed / shared-exponent
/// baked cubes are upload + sample only (they can't be render targets).
fn create_cube_image_usage(
    device: &ash::Device,
    allocator: &mut Allocator,
    format: vk::Format,
    size: u32,
    usage: vk::ImageUsageFlags,
    name: &str,
) -> Result<(vk::Image, Allocation, u32)> {
    let mips = cube_mip_count(size);
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: size,
            height: size,
            depth: 1,
        })
        .mip_levels(mips)
        .array_layers(6)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.create_image(&info, None)? };
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let allocation = allocator.allocate(&AllocationCreateDesc {
        name,
        requirements,
        location: MemoryLocation::GpuOnly,
        linear: false,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    })
    .with_context(|| format!("allocating GPU memory for {name}"))?;
    unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };
    Ok((image, allocation, mips))
}

/// A `samplerCube` view over all 6 faces + all mips (runtime sampling).
#[allow(dead_code)] // reflection-probe capture (in progress) — foundation landed
fn create_cube_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    mips: u32,
) -> Result<vk::ImageView> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::CUBE)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(mips)
                .layer_count(6),
        );
    Ok(unsafe { device.create_image_view(&info, None)? })
}

/// A single-face, single-mip 2D view used as a render target when capturing or
/// prefiltering one cube face.
#[allow(dead_code)] // reflection-probe capture (in progress) — foundation landed
fn create_cube_face_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    face: u32,
    mip: u32,
) -> Result<vk::ImageView> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(mip)
                .level_count(1)
                .base_array_layer(face)
                .layer_count(1),
        );
    Ok(unsafe { device.create_image_view(&info, None)? })
}

/// An in-progress reflection cubemap capture, advanced one face per frame (each
/// face recorded into that frame's command buffer), then one more step builds
/// the mip chain and copies the result into the live environment cube.
struct ReflCaptureJob {
    center: glam::Vec3,
    size: u32,
    /// Faces rendered so far (0..6); at 6 the next step finalizes.
    next_face: usize,
    /// Baked-probe capture: draw only static geometry and no realtime GI, so
    /// dynamic objects and their light bounces never bake into the cube.
    static_only: bool,
    /// Reflection-probe slot the finished cube is copied into (0 or 1).
    slot: u8,
    /// One face per frame; false renders all six faces and the mip chain in
    /// the frame the capture starts.
    sliced: bool,
    /// Draw non-static objects (live captures that opted in).
    dynamic: bool,
}

/// A pending reflection capture, consumed into a [`ReflCaptureJob`].
#[derive(Clone, Copy)]
struct ReflCaptureRequest {
    center: glam::Vec3,
    size: u32,
    static_only: bool,
    slot: u8,
    /// Seconds since the previous capture finished before this one may start.
    min_interval: f32,
    sliced: bool,
    dynamic: bool,
}

/// Persistent capture targets for one probe resolution (kept across captures).
struct ReflCaptureTarget {
    size: u32,
    mips: u32,
    cube: vk::Image,
    cube_alloc: Option<Allocation>,
    face_views: [vk::ImageView; 6],
    depth_img: vk::Image,
    depth_alloc: Option<Allocation>,
    depth_view: vk::ImageView,
}

impl ReflCaptureTarget {
    fn new(device: &ash::Device, allocator: &mut Allocator, size: u32) -> Result<Self> {
        let (cube, cube_alloc, mips) = create_cube_image(device, allocator, ENV_CUBE_FORMAT, size, "reflection capture cube")?;
        let mut face_views = [vk::ImageView::null(); 6];
        for (i, v) in face_views.iter_mut().enumerate() {
            *v = create_cube_face_view(device, cube, ENV_CUBE_FORMAT, i as u32, 0)?;
        }
        let (depth_img, depth_alloc) = create_image(
            device,
            allocator,
            depth_format(),
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::Extent2D { width: size, height: size },
            "reflection capture depth",
        )?;
        let depth_view = create_view(device, depth_img, depth_format(), depth_aspect())?;
        Ok(Self { size, mips, cube, cube_alloc: Some(cube_alloc), face_views, depth_img, depth_alloc: Some(depth_alloc), depth_view })
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            for v in self.face_views {
                device.destroy_image_view(v, None);
            }
            device.destroy_image(self.cube, None);
            device.destroy_image_view(self.depth_view, None);
            device.destroy_image(self.depth_img, None);
        }
        if let Some(a) = self.cube_alloc.take() {
            let _ = allocator.free(a);
        }
        if let Some(a) = self.depth_alloc.take() {
            let _ = allocator.free(a);
        }
    }
}

/// Most planar-reflection surfaces rendered per frame (each costs a full
/// mirrored scene pass); extras fall back to the reflection probe.
pub const MAX_PLANAR: usize = 4;

/// Render targets for planar reflections: a 2D array (one layer per active
/// plane) rendered by mirrored scene passes and sampled by the main pass at
/// screen UV (set 0, binding 8). Recreated when the main target extent changes.
/// Uses the pipeline cache's color format (same pipelines as the main pass);
/// rendered with the inline tonemap and sampled as environment radiance, like
/// the probe cube.
struct PlanarTargets {
    extent: vk::Extent2D,
    image: vk::Image,
    alloc: Option<Allocation>,
    /// Sampled view over all layers.
    array_view: vk::ImageView,
    /// Per-layer attachment views for the mirrored passes.
    layer_views: Vec<vk::ImageView>,
    depth_img: vk::Image,
    depth_alloc: Option<Allocation>,
    depth_view: vk::ImageView,
}

impl PlanarTargets {
    fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> Result<Self> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 })
            .mip_levels(1)
            .array_layers(MAX_PLANAR as u32)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&info, None)? };
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let alloc = allocator
            .allocate(&AllocationCreateDesc {
                name: "planar reflection array",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .with_context(|| "allocating GPU memory for planar reflection array")?;
        unsafe { device.bind_image_memory(image, alloc.memory(), alloc.offset())? };
        let array_view = {
            let info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
                .format(format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(MAX_PLANAR as u32),
                );
            unsafe { device.create_image_view(&info, None)? }
        };
        let mut layer_views = Vec::with_capacity(MAX_PLANAR);
        for layer in 0..MAX_PLANAR as u32 {
            let info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_array_layer(layer)
                        .level_count(1)
                        .layer_count(1),
                );
            layer_views.push(unsafe { device.create_image_view(&info, None)? });
        }
        let (depth_img, depth_alloc) = create_image(
            device,
            allocator,
            depth_format(),
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            extent,
            "planar reflection depth",
        )?;
        let depth_view = create_view(device, depth_img, depth_format(), depth_aspect())?;
        Ok(Self {
            extent,
            image,
            alloc: Some(alloc),
            array_view,
            layer_views,
            depth_img,
            depth_alloc: Some(depth_alloc),
            depth_view,
        })
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_image_view(self.array_view, None);
            for v in self.layer_views.drain(..) {
                device.destroy_image_view(v, None);
            }
            device.destroy_image_view(self.depth_view, None);
            device.destroy_image(self.image, None);
            device.destroy_image(self.depth_img, None);
        }
        if let Some(a) = self.alloc.take() {
            let _ = allocator.free(a);
        }
        if let Some(a) = self.depth_alloc.take() {
            let _ = allocator.free(a);
        }
    }
}

/// Householder reflection about the plane through `point` with unit `normal`:
/// mirrors world space across the plane (applied to the view matrix for planar
/// reflection rendering).
fn mirror_matrix(point: glam::Vec3, normal: glam::Vec3) -> glam::Mat4 {
    let n = normal.normalize_or(glam::Vec3::Y);
    let d = n.dot(point);
    glam::Mat4::from_cols(
        glam::vec4(1.0 - 2.0 * n.x * n.x, -2.0 * n.x * n.y, -2.0 * n.x * n.z, 0.0),
        glam::vec4(-2.0 * n.y * n.x, 1.0 - 2.0 * n.y * n.y, -2.0 * n.y * n.z, 0.0),
        glam::vec4(-2.0 * n.z * n.x, -2.0 * n.z * n.y, 1.0 - 2.0 * n.z * n.z, 0.0),
        glam::vec4(2.0 * d * n.x, 2.0 * d * n.y, 2.0 * d * n.z, 1.0),
    )
}

/// Oblique near-plane projection (Lengyel), [0,1] depth: replace the z row so
/// the near plane IS `clip_view` (a plane in view space, normal toward the
/// camera side). Clips everything behind the mirror plane, which a plain near
/// plane can't do for a mirrored camera.
fn oblique_projection(proj: glam::Mat4, clip_view: glam::Vec4) -> glam::Mat4 {
    // Far-corner point opposite the plane normal, in view space.
    let q = proj.inverse()
        * glam::vec4(clip_view.x.signum(), clip_view.y.signum(), 1.0, 1.0);
    // Scale the plane so at q: z' = w' (depth 1 at the far corner); on the
    // plane itself z' = 0 (the new near plane).
    let row3 = proj.row(3);
    let c = clip_view * (row3.dot(q) / clip_view.dot(q));
    let mut m = proj;
    // glam is column-major: write row 2 across the columns.
    m.x_axis.z = c.x;
    m.y_axis.z = c.y;
    m.z_axis.z = c.z;
    m.w_axis.z = c.w;
    m
}

/// A prefiltered environment reflection cubemap (the runtime side of a
/// reflection probe). Built once from the scene's skybox: a mirror cube at mip 0
/// that box-blurs toward rougher mips, sampled by reflection direction +
/// roughness→mip in the forward shader. A 1×1 neutral cube until a skybox loads,
/// so the `samplerCube` binding is always valid.
struct ReflectionEnv {
    image: vk::Image,
    view: vk::ImageView,
    alloc: Option<Allocation>,
    /// Mip-0 face size (pixels). Stored so the cube can be read back for baking.
    size: u32,
    /// Image format: ENV_CUBE_FORMAT for captures, or a baked cube's
    /// compressed / compact format (never copied into or read back).
    format: vk::Format,
}

/// Linear HDR: captures and the skybox environment keep radiance above 1.
const ENV_CUBE_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
/// Planar mirror targets: linear HDR like the main pass.
const PLANAR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// Per-face `(look_direction, up)` for the reflection-cube capture, ordered
/// +X,-X,+Y,-Y,+Z,-Z. Each face is rendered with `look_at_rh(center, center+dir,
/// up)` + `perspective_rh(90°)` into a POSITIVE-height (non-Y-flipped) viewport.
/// These up-vectors are chosen so that round-trip equals the `face_dir` sampling
/// convention exactly — see `cube_capture_matches_sampling` test, which proves it.
const CUBE_CAPTURE_FACES: [(glam::Vec3, glam::Vec3); 6] = [
    (glam::Vec3::X, glam::Vec3::NEG_Y),
    (glam::Vec3::NEG_X, glam::Vec3::NEG_Y),
    (glam::Vec3::Y, glam::Vec3::Z),
    (glam::Vec3::NEG_Y, glam::Vec3::NEG_Z),
    (glam::Vec3::Z, glam::Vec3::NEG_Y),
    (glam::Vec3::NEG_Z, glam::Vec3::NEG_Y),
];

impl ReflectionEnv {
    /// Direction for cube `face` at face-plane coords (`fx`,`fy`) in [-1,1].
    fn face_dir(face: usize, fx: f32, fy: f32) -> [f32; 3] {
        let d = match face {
            0 => [1.0, -fy, -fx],
            1 => [-1.0, -fy, fx],
            2 => [fx, 1.0, fy],
            3 => [fx, -1.0, -fy],
            4 => [fx, -fy, 1.0],
            _ => [-fx, -fy, -1.0],
        };
        let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
        [d[0] / l, d[1] / l, d[2] / l]
    }

    /// Linear RGBA of texel `i` of `src`: f16 for HDR sources, sRGB-decoded or
    /// raw UNORM for 8-bit ones.
    fn texel(src: &TextureData, i: usize) -> [f32; 4] {
        if src.hdr {
            let o = i * 8;
            let Some(b) = src.pixels.get(o..o + 8) else { return [0.0, 0.0, 0.0, 1.0] };
            let h = |k: usize| half::f16::from_le_bytes([b[k], b[k + 1]]).to_f32();
            [h(0), h(2), h(4), h(6)]
        } else {
            let o = i * 4;
            let Some(b) = src.pixels.get(o..o + 4) else { return [0.0, 0.0, 0.0, 1.0] };
            let c = |v: u8| {
                let x = v as f32 / 255.0;
                if !src.srgb {
                    x
                } else if x <= 0.04045 {
                    x / 12.92
                } else {
                    ((x + 0.055) / 1.055).powf(2.4)
                }
            };
            [c(b[0]), c(b[1]), c(b[2]), b[3] as f32 / 255.0]
        }
    }

    /// Sample an equirectangular image in world direction `dir` (linear RGBA).
    fn sample_equirect(eq: &TextureData, dir: [f32; 3]) -> [f32; 4] {
        let u = dir[2].atan2(dir[0]) * std::f32::consts::FRAC_1_PI * 0.5 + 0.5;
        let v = dir[1].clamp(-1.0, 1.0).acos() * std::f32::consts::FRAC_1_PI;
        let px = ((u * eq.width as f32) as i64).rem_euclid(eq.width.max(1) as i64) as usize;
        let py = ((v * eq.height as f32) as usize).min(eq.height.max(1) as usize - 1);
        Self::texel(eq, py * eq.width as usize + px)
    }

    /// Mip-0 face (linear RGBA) for `size` from the equirect skybox.
    fn build_face_mip0(eq: &TextureData, face: usize, size: u32) -> Vec<[f32; 4]> {
        let mut out = Vec::with_capacity((size * size) as usize);
        for y in 0..size {
            for x in 0..size {
                let fx = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
                let fy = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;
                out.push(Self::sample_equirect(eq, Self::face_dir(face, fx, fy)));
            }
        }
        out
    }

    /// 2x2 box downsample of one face.
    fn downsample(src: &[[f32; 4]], src_size: u32) -> Vec<[f32; 4]> {
        let dst_size = (src_size / 2).max(1);
        let mut out = Vec::with_capacity((dst_size * dst_size) as usize);
        for y in 0..dst_size {
            for x in 0..dst_size {
                let mut sum = [0.0f32; 4];
                for dy in 0..2 {
                    for dx in 0..2 {
                        let sx = (x * 2 + dx).min(src_size - 1);
                        let sy = (y * 2 + dy).min(src_size - 1);
                        let p = src[(sy * src_size + sx) as usize];
                        for c in 0..4 {
                            sum[c] += p[c];
                        }
                    }
                }
                out.push(sum.map(|v| v * 0.25));
            }
        }
        out
    }

    /// Build the reflection cube from an equirectangular skybox (CPU prefilter:
    /// mirror at mip 0, progressively box-blurred mips ≈ roughness).
    fn from_equirect(
        device: &ash::Device,
        allocator: &mut Allocator,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        eq: &TextureData,
    ) -> Result<Self> {
        Self::build_cube(device, allocator, command_pool, queue, 64, |face, size| {
            Self::build_face_mip0(eq, face, size)
        })
    }

    /// Resample a source image to a `size×size` cube face (nearest, linear RGBA).
    fn resample_face(src: &TextureData, size: u32) -> Vec<[f32; 4]> {
        let mut out = Vec::with_capacity((size * size) as usize);
        for y in 0..size {
            for x in 0..size {
                let sx = (x * src.width / size).min(src.width.saturating_sub(1));
                let sy = (y * src.height / size).min(src.height.saturating_sub(1));
                out.push(Self::texel(src, (sy * src.width + sx) as usize));
            }
        }
        out
    }

    /// Build the cube from 6 explicit face images (+X,-X,+Y,-Y,+Z,-Z order) — a
    /// native cubemap / 6-texture skybox or a baked probe. Mip 0 is the faces at
    /// their own resolution; rougher mips blur.
    fn from_faces(
        device: &ash::Device,
        allocator: &mut Allocator,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        faces: &[&TextureData; 6],
    ) -> Result<Self> {
        let size = faces[0].width.clamp(1, MAX_BAKED_CUBE_SIZE);
        let chain = Self::cpu_chain(size, |face, size| Self::resample_face(faces[face], size));
        Self::upload_chain(device, allocator, command_pool, queue, &chain)
    }

    /// Shared cube builder: `face_mip0(face, size)` produces linear mip-0 texels
    /// for each face; higher mips are box-downsampled; stored as RGBA16F.
    /// Uploads + leaves SHADER_READ.
    fn build_cube(
        device: &ash::Device,
        allocator: &mut Allocator,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        size: u32,
        face_mip0: impl Fn(usize, u32) -> Vec<[f32; 4]>,
    ) -> Result<Self> {
        Self::upload_chain(device, allocator, command_pool, queue, &Self::cpu_chain(size, face_mip0))
    }

    /// CPU prefilter: RGBA16F mip chain per face (mip 0 from `face_mip0`, then
    /// 2x2 box downsamples to 1x1).
    fn cpu_chain(size: u32, face_mip0: impl Fn(usize, u32) -> Vec<[f32; 4]>) -> HdrCube {
        let mips = cube_mip_count(size);
        let faces = std::array::from_fn(|face| {
            let mut cur = face_mip0(face, size);
            let mut s = size;
            let mut levels = Vec::with_capacity(mips as usize);
            for _ in 0..mips {
                let mut bytes = Vec::with_capacity(cur.len() * 8);
                for p in &cur {
                    for c in p {
                        bytes.extend_from_slice(&half::f16::from_f32(*c).to_le_bytes());
                    }
                }
                levels.push(bytes);
                if s > 1 {
                    cur = Self::downsample(&cur, s);
                    s /= 2;
                }
            }
            levels
        });
        HdrCube { format: HdrTextureFormat::Rgba16f, size, faces }
    }

    /// Upload a prebuilt mip chain in any [`HdrTextureFormat`]. Only the
    /// capture format gets render-target usage; others are upload + sample.
    fn upload_chain(
        device: &ash::Device,
        allocator: &mut Allocator,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
        cube: &HdrCube,
    ) -> Result<Self> {
        anyhow::ensure!(cube.is_well_formed(), "malformed reflection cube ({} {}px)", cube.format.label(), cube.size);
        let size = cube.size;
        let format = cube.format.vk_format();
        let usage = if format == ENV_CUBE_FORMAT {
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
        } else {
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST
        };
        let (image, alloc, mips) =
            create_cube_image_usage(device, allocator, format, size, usage, "reflection env cube")?;

        // Packed face-major, then mip; regions[(face,mip)] = (byte offset, mip size).
        let mut data: Vec<u8> = Vec::with_capacity(cube.byte_size() as usize);
        let mut regions: Vec<(u64, u32)> = Vec::new();
        for face in &cube.faces {
            for (mip, level) in face.iter().enumerate() {
                regions.push((data.len() as u64, (size >> mip).max(1)));
                data.extend_from_slice(level);
            }
        }

        let mut staging = Buffer::new(
            device,
            allocator,
            data.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "reflection env staging",
        )?;
        staging.write(0, &data);

        let full = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(mips)
            .layer_count(6);
        crate::alloc::one_time_submit(device, command_pool, queue, |cb| unsafe {
            let to_dst = vk::ImageMemoryBarrier::default()
                .image(image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .subresource_range(full);
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_dst],
            );
            let mut copies = Vec::new();
            let mut idx = 0;
            for face in 0..6u32 {
                let mut s = size;
                for mip in 0..mips {
                    let (offset, msize) = regions[idx];
                    idx += 1;
                    copies.push(
                        vk::BufferImageCopy::default()
                            .buffer_offset(offset)
                            .image_subresource(
                                vk::ImageSubresourceLayers::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .mip_level(mip)
                                    .base_array_layer(face)
                                    .layer_count(1),
                            )
                            .image_extent(vk::Extent3D {
                                width: msize,
                                height: msize,
                                depth: 1,
                            }),
                    );
                    let _ = s;
                    s = (s / 2).max(1);
                }
            }
            device.cmd_copy_buffer_to_image(
                cb,
                staging.handle,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copies,
            );
            let to_read = vk::ImageMemoryBarrier::default()
                .image(image)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .subresource_range(full);
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_read],
            );
        })?;
        staging.destroy(device, allocator);

        let view = create_cube_view(device, image, format, mips)?;
        Ok(Self {
            image,
            view,
            alloc: Some(alloc),
            size,
            format,
        })
    }

    /// A 1×1 neutral (black) cube so the binding is valid before a skybox loads.
    /// The default reflection environment when no skybox texture is set: the
    /// procedural sky gradient (matching `skybox.frag`), so reflective surfaces
    /// always reflect the sky instead of black.
    fn neutral(
        device: &ash::Device,
        allocator: &mut Allocator,
        command_pool: vk::CommandPool,
        queue: vk::Queue,
    ) -> Result<Self> {
        Self::build_cube(device, allocator, command_pool, queue, 64, |face, size| {
            let mut out = Vec::with_capacity((size * size) as usize);
            // The procedural sky of skybox.frag (sky.rs copy), lit by the default sun.
            let sun = sky::default_sun();
            for y in 0..size {
                for x in 0..size {
                    let fx = (x as f32 + 0.5) / size as f32 * 2.0 - 1.0;
                    let fy = (y as f32 + 0.5) / size as f32 * 2.0 - 1.0;
                    let d = Self::face_dir(face, fx, fy);
                    let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-9);
                    let c = sky::procedural_radiance([d[0] / l, d[1] / l, d[2] / l], sun);
                    out.push([c[0], c[1], c[2], 1.0]);
                }
            }
            out
        })
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }
        if let Some(a) = self.alloc.take() {
            let _ = allocator.free(a);
        }
    }
}

/// Common layer size of a lightmap array holding layers of `sizes`.
pub fn lightmap_array_size(sizes: impl Iterator<Item = u32>) -> u32 {
    sizes.max().unwrap_or(1).clamp(1, 4096)
}

/// See [`reflection_cube_chain`].
fn reflection_chain_from_faces(faces: &[TextureData; 6]) -> HdrCube {
    let size = faces[0].width.clamp(1, MAX_BAKED_CUBE_SIZE);
    ReflectionEnv::cpu_chain(size, |face, size| ReflectionEnv::resample_face(&faces[face], size))
}

/// Screen-probe tile size: one screen-space GI probe is traced per
/// `SCREEN_PROBE_DIV × SCREEN_PROBE_DIV` block of pixels, then bilinearly
/// upsampled to full resolution in the forward shader (screen-probe
/// scheme). Larger is cheaper and softer; smaller is sharper and uses more rays.
const SCREEN_PROBE_DIV: u32 = 4;

/// Runtime-tunable Flux (screen-space GI) parameters, set from the Environment
/// tab via [`Renderer::set_flux_settings`]. These drive the per-frame trace that
/// was previously hardcoded.
#[derive(Clone, Copy, PartialEq)]
pub struct FluxSettings {
    pub ray_lighting_mode: RayLightingMode,
    /// Rays per screen probe per frame (temporal accumulation smooths the rest).
    pub samples: u32,
    /// 1 uses single-bounce diffuse lighting; 2..4 enable iterative cached
    /// diffuse transport and bound the fallback path when a card is missing.
    pub bounces: u32,
    /// Max trace distance in world units; 0 = auto from the GDF bounds.
    pub march_distance: f32,
    /// Per-sample firefly clamp on the bounce term (caps bright outliers).
    pub firefly_clamp: f32,
    /// Temporal smoothing 0..1: higher is smoother with more lag, lower is sharper and noisier.
    pub smoothing: f32,
    /// Indirect strength multiplier (applied before temporal accumulation).
    pub intensity: f32,
    /// Scale material emission only when it enters diffuse GI transport.
    /// Visible/reflected source emission remains the material's own radiance.
    pub emissive_boost: f32,
    /// Screen-space reflections: trace specular rays against the depth prepass +
    /// last frame's colour. Only active while Flux runs (it owns the depth prepass).
    pub ssr_enabled: bool,
    /// SSR reflection strength multiplier.
    pub ssr_intensity: f32,
    /// SSR max ray distance in view-space units.
    pub ssr_max_distance: f32,
    /// SSR roughness cutoff: surfaces rougher than this skip the march (the cheap
    /// ambient-specular env approximation carries them instead).
    pub ssr_roughness_cutoff: f32,
    /// Reflection model: 0 = environment cube only, 1 = screen-space (SSR),
    /// 2 = ray-traced (1 bounce, needs GPU ray-query support). Ray-traced reflects
    /// real geometry so it has none of SSR's screen-edge/silhouette artifacts.
    pub reflection_mode: u32,
    /// Ray-traced reflections: rays per pixel (1..16).
    pub rt_samples: u32,
    /// Ray-traced reflections: bounces (1..8).
    pub rt_bounces: u32,
    /// Mixed lighting: a bake is loaded, so the Flux gather carries only what the
    /// bake can't (moving emitters, realtime lights) and adds on the lightmaps.
    pub mixed_lighting: bool,
    /// GI trace backend: false = software (GDF march, screen_gi.comp), true =
    /// hardware ray-query against the scene TLAS (screen_gi_rt.comp) — exact
    /// triangle hits, no SDF thickness/leak. Ignored (stays software) when the
    /// device lacks ray-query. Off by default so the SDF path is unaffected.
    pub rt_trace: bool,
}

impl FluxSettings {
    fn for_view(mut self, post: FluxPostFx) -> Self {
        fn finite(value: Option<f32>, default: f32) -> f32 {
            value.filter(|v| v.is_finite()).unwrap_or(default)
        }
        self.ray_lighting_mode = match post.ray_lighting_mode {
            None | Some(RayLightingMode::Default) => self.ray_lighting_mode,
            Some(mode) => mode,
        };
        if self.ray_lighting_mode == RayLightingMode::Default { self.ray_lighting_mode = RayLightingMode::SurfaceCache; }
        let quality = finite(post.final_gather_quality, 1.0).clamp(0.25, 4.0);
        self.samples = ((self.samples as f32 * quality).round() as u32).clamp(16, 256);
        self.march_distance = finite(post.max_trace_distance, self.march_distance).max(0.0);
        self.rt_samples = ((self.rt_samples as f32 * finite(post.reflection_quality, 1.0).clamp(0.25, 4.0)).round() as u32).clamp(1, 16);
        self.ssr_roughness_cutoff = finite(post.max_roughness_to_trace, self.ssr_roughness_cutoff).clamp(0.0, 1.0);
        self.rt_bounces = (finite(post.max_reflection_bounces, self.rt_bounces as f32).round() as u32).clamp(1, 8);
        self
    }
}

/// RT reflection quality lane: rays per pixel, bounces, max ray distance.
fn rt_quality(fs: &FluxSettings, frame: u64) -> [f32; 4] {
    [
        fs.rt_samples.clamp(1, 16) as f32,
        fs.rt_bounces.clamp(1, 8) as f32,
        if fs.ssr_max_distance > 0.0 { fs.ssr_max_distance } else { 1e16 },
        // Continuous frame serial, not the repeating in-flight slot. Wrapping
        // before conversion preserves fractional sampling precision in long runs.
        (frame & 4095) as f32,
    ]
}

/// Build/refresh the shared scene TLAS for the Flux-RT GI gather (free function so
/// it borrows only the disjoint renderer fields it needs — a `&mut self` method
/// would clash with the `device` borrow held across the GI-trace call sites).
/// Returns the TLAS + instance SSBO, or None when ray-query/instances are absent.
#[allow(clippy::too_many_arguments)]
fn build_gi_rt_tlas(
    ctx: &GpuContext,
    allocator: Option<Arc<Mutex<Allocator>>>,
    command_pool: vk::CommandPool,
    meshes: &[GpuMesh],
    rt_reflect: &mut Option<rt_reflect::RtReflect>,
    instances: &[rt_reflect::RtInstance],
    water: &rt_water::RtWaterFrame,
    cb: vk::CommandBuffer,
    frame: usize,
) -> Option<(vk::AccelerationStructureKHR, vk::Buffer)> {
    let allocator = allocator?;
    let rt = rt_reflect.as_mut()?;
    rt.ensure_tlas(ctx, &allocator, command_pool, meshes, instances, water, cb, frame)
        .ok()
        .flatten()
}

impl Default for FluxSettings {
    fn default() -> Self {
        // Mirrors the old hardcoded trace (Balanced preset).
        Self {
            samples: 16,
            bounces: 2,
            march_distance: 0.0,
            firefly_clamp: 2.0,
            smoothing: 0.5,
            intensity: 1.0,
            emissive_boost: 1.0,
            ssr_enabled: true,
            ssr_intensity: 1.0,
            ssr_max_distance: 40.0,
            ssr_roughness_cutoff: 0.6,
            reflection_mode: 1,
            rt_samples: 1,
            rt_bounces: 1,
            ray_lighting_mode: RayLightingMode::SurfaceCache,
            mixed_lighting: false,
            rt_trace: false,
        }
    }
}

/// Screen-space GI targets: a full-res sampleable camera depth prepass + a
/// reduced-resolution screen-probe radiance buffer (one probe per
/// `SCREEN_PROBE_DIV²` pixels), bilinearly upsampled by the forward shader.
struct ScreenGiTargets {
    /// Probe-grid resolution (screen extent / SCREEN_PROBE_DIV).
    probe_extent: vk::Extent2D,
    depth_image: vk::Image,
    /// Depth-aspect view (sampled by the screen-probe passes).
    depth_view: vk::ImageView,
    /// Attachment view (depth + stencil) the prepass renders through.
    depth_att_view: vk::ImageView,
    depth_alloc: Option<Allocation>,
    /// Ping-pong screen-probe buffers: temporal reprojection reads the previous
    /// frame's image and writes the other (in-place would race since reprojection
    /// reads neighbour texels). `parity` selects which is current.
    gi_image: [vk::Image; 2],
    gi_view: [vk::ImageView; 2],
    gi_alloc: [Option<Allocation>; 2],
    /// Directional radiance, hit distances and valid history counts, ping-ponged
    /// per view. The spatial filter writes separate SH lanes after filtering.
    sh_buf: [Buffer; 2],
    /// Current traced samples, separate from accumulated receiver histories.
    raw_rays: Buffer,
    /// Probe-res denoised gather (flux_denoise.comp output). The forward shader
    /// samples THIS, not the raw gather — so the wide edge-aware filter runs once
    /// per probe (16× cheaper than per screen pixel). Not ping-ponged: it's a
    /// display-only product regenerated each traced frame; temporal history still
    /// feeds back through the raw `gi_image` ping-pong (avoids over-blur).
    gi_filtered: vk::Image,
    gi_filtered_view: vk::ImageView,
    gi_filtered_alloc: Option<Allocation>,
    /// FULL-resolution screen-probe integrate output (flux_integrate.comp): the
    /// denoised probe GI re-pointed to each pixel's own normal via the per-probe
    /// SH-L1. The forward samples THIS (per-pixel-normal resolve fixes curved-
    /// surface grain). Regenerated each traced frame; persists across skip frames.
    gi_final: vk::Image,
    gi_final_view: vk::ImageView,
    gi_final_alloc: Option<Allocation>,
    /// Full-resolution radiance/count and normal/distance history, per view.
    full_history: [screen_pass::ScreenImage; 4],
}

impl ScreenGiTargets {
    fn new(ctx: &GpuContext, allocator: &mut Allocator, extent: vk::Extent2D) -> Result<Self> {
        let (depth_image, depth_alloc) = create_image(
            &ctx.device,
            allocator,
            depth_format(),
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            extent,
            "sgi depth prepass",
        )?;
        let depth_view = create_view(
            &ctx.device,
            depth_image,
            depth_format(),
            vk::ImageAspectFlags::DEPTH,
        )?;
        let depth_att_view = create_view(&ctx.device, depth_image, depth_format(), depth_aspect())?;
        // The probe buffer is the screen-probe grid: one texel per tile.
        let probe_extent = vk::Extent2D {
            width: extent.width.div_ceil(SCREEN_PROBE_DIV).max(1),
            height: extent.height.div_ceil(SCREEN_PROBE_DIV).max(1),
        };
        let mut gi_image = [vk::Image::null(); 2];
        let mut gi_view = [vk::ImageView::null(); 2];
        let mut gi_alloc: [Option<Allocation>; 2] = [None, None];
        for k in 0..2 {
            let (img, alloc) = create_image(
                &ctx.device,
                allocator,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                probe_extent,
                "sgi screen probes",
            )?;
            gi_view[k] = create_view(
                &ctx.device,
                img,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageAspectFlags::COLOR,
            )?;
            gi_image[k] = img;
            gi_alloc[k] = Some(alloc);
        }
        // Directional history and filtered SH, GPU-only (see flux_probe.glsl).
        let probe_count = (probe_extent.width as u64) * (probe_extent.height as u64);
        let probe_usage = vk::BufferUsageFlags::STORAGE_BUFFER | if cfg!(test) {
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST
        } else { vk::BufferUsageFlags::empty() };
        let sh_buf = [
            Buffer::new(
                &ctx.device,
                allocator,
                probe_count.max(1) * gpu_gi::SCREEN_PROBE_BYTES,
                probe_usage,
                MemoryLocation::GpuOnly,
                "sgi probe sh 0",
            )?,
            Buffer::new(
                &ctx.device,
                allocator,
                probe_count.max(1) * gpu_gi::SCREEN_PROBE_BYTES,
                probe_usage,
                MemoryLocation::GpuOnly,
                "sgi probe sh 1",
            )?,
        ];
        let raw_rays = Buffer::new(&ctx.device, allocator, 16 + probe_count.max(1) * 17 * 16,
            probe_usage | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, MemoryLocation::GpuOnly, "sgi adaptive raw rays")?;
        // Probe-res denoised gather (storage write by flux_denoise, sampled by the
        // forward). Same format/extent as gi_image.
        let (gi_filtered, gi_filtered_alloc) = create_image(
            &ctx.device,
            allocator,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            probe_extent,
            "sgi denoised",
        )?;
        let gi_filtered_view = create_view(
            &ctx.device,
            gi_filtered,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageAspectFlags::COLOR,
        )?;
        // Full-res integrate output (storage write by flux_integrate, sampled by
        // the forward). Full screen extent, not probe extent.
        let (gi_final, gi_final_alloc) = create_image(
            &ctx.device,
            allocator,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
            extent,
            "sgi integrated",
        )?;
        let gi_final_view = create_view(
            &ctx.device,
            gi_final,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageAspectFlags::COLOR,
        )?;
        let mut full_history = Vec::new();
        for name in ["Flux full history 0", "Flux full history 1", "Flux geometry history 0", "Flux geometry history 1"] {
            full_history.push(screen_pass::ScreenImage::new(&ctx.device, allocator,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                extent, 1, name)?);
        }
        Ok(Self {
            full_history: full_history.try_into().ok().unwrap(),
            probe_extent,
            depth_image,
            depth_view,
            depth_att_view,
            depth_alloc: Some(depth_alloc),
            gi_image,
            gi_view,
            gi_alloc,
            sh_buf,
            raw_rays,
            gi_filtered,
            gi_filtered_view,
            gi_filtered_alloc: Some(gi_filtered_alloc),
            gi_final,
            gi_final_view,
            gi_final_alloc: Some(gi_final_alloc),
        })
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_image_view(self.depth_view, None);
            device.destroy_image_view(self.depth_att_view, None);
            device.destroy_image(self.depth_image, None);
            for k in 0..2 {
                device.destroy_image_view(self.gi_view[k], None);
                device.destroy_image(self.gi_image[k], None);
            }
        }
        if let Some(a) = self.depth_alloc.take() {
            let _ = allocator.free(a);
        }
        for k in 0..2 {
            if let Some(a) = self.gi_alloc[k].take() {
                let _ = allocator.free(a);
            }
        }
        unsafe {
            device.destroy_image_view(self.gi_filtered_view, None);
            device.destroy_image(self.gi_filtered, None);
            device.destroy_image_view(self.gi_final_view, None);
            device.destroy_image(self.gi_final, None);
        }
        if let Some(a) = self.gi_filtered_alloc.take() {
            let _ = allocator.free(a);
        }
        if let Some(a) = self.gi_final_alloc.take() {
            let _ = allocator.free(a);
        }
        self.raw_rays.destroy(device, allocator);
        for b in &mut self.sh_buf {
            b.destroy(device, allocator);
        }
        for image in &mut self.full_history { image.destroy(device,allocator); }
    }

    fn temporal(&self, cur: usize) -> gpu_gi::TemporalImages {
        let prev=cur^1;
        gpu_gi::TemporalImages {
            current: [self.full_history[cur].image,self.full_history[cur+2].image],
            current_views: [self.full_history[cur].view,self.full_history[cur+2].view],
            previous: [self.full_history[prev].image,self.full_history[prev+2].image],
            previous_views: [self.full_history[prev].view,self.full_history[prev+2].view],
        }
    }
}


/// Eye renders in flight before a slot is reused (2 eyes x frames in flight).
const XR_SLOTS: usize = 2 * FRAMES_IN_FLIGHT;

/// One in-flight VR eye render: its command buffer + fence, the eye camera UBO
/// and set-0 descriptor set (shared bindings copied from the frame set), and
/// the XR image views (alive until the fence signals).
struct XrSlot {
    cb: vk::CommandBuffer,
    fence: vk::Fence,
    ubo: Buffer,
    set: vk::DescriptorSet,
    color_views: Vec<vk::ImageView>,
    instances: InstanceRing,
}

/// Per-submission store of per-instance model matrices: the instance-rate
/// vertex stream scene draws read (`standard.vert` / `outline.vert`). Owned by
/// a frame slot or a VR eye slot and reset once that submission's fence has
/// been waited, so a region is never rewritten while the GPU may read it.
#[derive(Default)]
struct InstanceRing {
    buf: Option<Buffer>,
    /// Capacity in matrices.
    capacity: usize,
    used: usize,
    /// Buffers outgrown during the current submission (still referenced by
    /// its commands); freed on the next reset.
    retired: Vec<Buffer>,
}

impl InstanceRing {
    fn reset(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for mut b in self.retired.drain(..) {
            b.destroy(device, allocator);
        }
        self.used = 0;
    }

    /// Append `data` and return the buffer and byte offset it starts at.
    fn push(&mut self, device: &ash::Device, allocator: &mut Allocator, data: &[InstanceData]) -> Result<(vk::Buffer, u64)> {
        if self.buf.is_none() || self.used + data.len() > self.capacity {
            let capacity = ((self.used + data.len()) * 2).max(1024);
            let buf = Buffer::new(
                device,
                allocator,
                (capacity * INSTANCE_STRIDE as usize) as u64,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                gpu_allocator::MemoryLocation::CpuToGpu,
                "scene-instances",
            )?;
            if let Some(old) = self.buf.replace(buf) {
                self.retired.push(old);
            }
            self.capacity = capacity;
            self.used = 0;
        }
        let offset = (self.used * INSTANCE_STRIDE as usize) as u64;
        let buf = self.buf.as_mut().unwrap();
        buf.write(offset as usize, bytemuck::cast_slice(data));
        self.used += data.len();
        Ok((buf.handle, offset))
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        self.reset(device, allocator);
        if let Some(mut b) = self.buf.take() {
            b.destroy(device, allocator);
        }
    }
}

/// VR eye render state: a ring of slots (submits never wait on the CPU) plus
/// the eye-sized HDR target, depth, post source sets and bloom chain they
/// share (queue order serializes them). `layers` is 1 for the per-eye path and
/// 2 for the multiview path (one layer per eye); `samples` > 1 adds
/// multisampled attachments resolved into the HDR / depth targets in-pass.
struct XrRender {
    pool: vk::DescriptorPool,
    slots: Vec<XrSlot>,
    next: usize,
    depth_image: vk::Image,
    /// All layers (2D for one layer, 2D array for multiview).
    depth_view: vk::ImageView,
    /// One 2D view per layer (sampled per eye).
    depth_layer_views: Vec<vk::ImageView>,
    depth_alloc: Option<Allocation>,
    hdr_image: vk::Image,
    hdr_view: vk::ImageView,
    hdr_layer_views: Vec<vk::ImageView>,
    hdr_alloc: Option<Allocation>,
    msaa: Option<XrMsaa>,
    post_pool: vk::DescriptorPool,
    /// Post source set per layer (samples `hdr_layer_views[i]`).
    post_sets: Vec<vk::DescriptorSet>,
    bloom: Option<bloom::BloomChain>,
    extent: vk::Extent2D,
    layers: u32,
    samples: u8,
}

/// Multisampled colour + depth for the eye scene pass; never stored, only
/// resolved, so on a tiler they live in tile memory.
struct XrMsaa {
    color: TransientImage,
    depth: TransientImage,
}

/// An attachment that is only ever cleared, drawn and resolved: transient
/// usage, lazily allocated memory when the device has such a memory type (it
/// then never gets physical backing on a tiled GPU).
struct TransientImage {
    image: vk::Image,
    view: vk::ImageView,
    memory: TransientMemory,
}

enum TransientMemory {
    Lazy(vk::DeviceMemory),
    Pooled(Option<Allocation>),
}

impl TransientImage {
    #[allow(clippy::too_many_arguments)]
    fn new(
        ctx: &GpuContext,
        allocator: &mut Allocator,
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
        usage: vk::ImageUsageFlags,
        extent: vk::Extent2D,
        layers: u32,
        samples: vk::SampleCountFlags,
        name: &str,
    ) -> Result<Self> {
        let device = &ctx.device;
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 })
            .mip_levels(1)
            .array_layers(layers)
            .samples(samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&info, None)? };
        let req = unsafe { device.get_image_memory_requirements(image) };
        let props = unsafe { ctx.instance.get_physical_device_memory_properties(ctx.physical_device) };
        let lazy = (0..props.memory_type_count).find(|&i| {
            req.memory_type_bits & (1 << i) != 0
                && props.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::LAZILY_ALLOCATED)
        });
        let memory = match lazy {
            Some(ty) => {
                let mem = unsafe {
                    device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty), None)?
                };
                unsafe { device.bind_image_memory(image, mem, 0)? };
                TransientMemory::Lazy(mem)
            }
            None => {
                let a = allocator
                    .allocate(&AllocationCreateDesc {
                        name,
                        requirements: req,
                        location: MemoryLocation::GpuOnly,
                        linear: false,
                        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                    })
                    .with_context(|| format!("allocating GPU memory for {name}"))?;
                unsafe { device.bind_image_memory(image, a.memory(), a.offset())? };
                TransientMemory::Pooled(Some(a))
            }
        };
        let view = create_view_layers(device, image, format, aspect, 0, layers)?;
        Ok(Self { image, view, memory })
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }
        match &mut self.memory {
            TransientMemory::Lazy(m) => unsafe { device.free_memory(*m, None) },
            TransientMemory::Pooled(a) => {
                if let Some(a) = a.take() {
                    let _ = allocator.free(a);
                }
            }
        }
    }
}

impl XrRender {
    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator, command_pool: vk::CommandPool) {
        unsafe {
            let fences: Vec<vk::Fence> = self.slots.iter().map(|s| s.fence).collect();
            let _ = device.wait_fences("wait:fence@destroy", &fences, true, u64::MAX);
        }
        for s in &mut self.slots {
            s.ubo.destroy(device, allocator);
            s.instances.destroy(device, allocator);
            unsafe {
                device.destroy_fence(s.fence, None);
                device.free_command_buffers(command_pool, &[s.cb]);
                for v in s.color_views.drain(..) {
                    device.destroy_image_view(v, None);
                }
            }
        }
        if let Some(mut b) = self.bloom.take() {
            b.destroy(device, allocator);
        }
        if let Some(mut m) = self.msaa.take() {
            m.color.destroy(device, allocator);
            m.depth.destroy(device, allocator);
        }
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_pool(self.post_pool, None);
            for v in self.depth_layer_views.drain(..).chain(self.hdr_layer_views.drain(..)) {
                device.destroy_image_view(v, None);
            }
            device.destroy_image_view(self.depth_view, None);
            device.destroy_image(self.depth_image, None);
            device.destroy_image_view(self.hdr_view, None);
            device.destroy_image(self.hdr_image, None);
        }
        for a in [self.depth_alloc.take(), self.hdr_alloc.take()].into_iter().flatten() {
            let _ = allocator.free(a);
        }
    }
}

/// One eye of a multiview XR frame ([`Renderer::render_xr_stereo`]).
#[derive(Clone, Copy, Debug)]
pub struct XrEyeCamera {
    /// World to eye.
    pub view: Mat4,
    pub proj: Mat4,
    /// This eye's post effects (the underwater state can differ per eye).
    pub postfx: PostFx,
}

/// Max VR UI panels drawn per eye.
pub const VR_MAX_PANELS: usize = 4;

/// A tracked hand (XR_EXT_hand_tracking) in world space for the VR overlay:
/// joint positions + validity mask, indexed by the OpenXR joint order (palm,
/// wrist, thumb, index, middle, ring, little).
#[derive(Clone, Copy, Debug, Default)]
pub struct HandSkeleton {
    pub positions: [glam::Vec3; 26],
    pub valid: [bool; 26],
}

/// Hand bones as joint-index pairs (OpenXR `HandJoint` order): palm→wrist, the
/// thumb chain, and the palm→finger chains, four segments per finger.
pub const HAND_BONES: [(usize, usize); 25] = [
    (1, 0),
    (2, 3),
    (3, 4),
    (4, 5),
    (0, 2),
    (0, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (0, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (0, 16),
    (16, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 21),
    (21, 22),
    (22, 23),
    (23, 24),
    (24, 25),
];

#[cfg(test)]
mod hand_bones_tests {
    use super::HAND_BONES;

    #[test]
    fn bones_are_valid_joint_pairs() {
        for &(a, b) in HAND_BONES.iter() {
            assert!(a < 26 && b < 26, "bone {a}-{b} outside the 26 hand joints");
            assert_ne!(a, b, "bone {a}-{b} joins a joint to itself");
        }
    }
}

/// One VR UI panel: a world-space quad showing a sub-rect of the UI texture.
#[derive(Clone, Copy, Debug)]
pub struct VrPanel {
    pub model: Mat4,
    pub uv_rect: [f32; 4],
}

/// VR overlay for an eye render: the hand-menu panels, tracked hands and both
/// hand lasers.
#[derive(Clone, Debug)]
pub struct VrOverlayDraw {
    pub panels: [Option<VrPanel>; VR_MAX_PANELS],
    pub left_hand: Option<Mat4>,
    pub right_hand: Option<Mat4>,
    /// Tracked hand skeletons (world space), drawn as bones + joint dots.
    pub left_hand_skeleton: Option<HandSkeleton>,
    pub right_hand_skeleton: Option<HandSkeleton>,
    /// Laser end points (UI or surface hit, or far along the ray).
    pub left_laser_end: Option<glam::Vec3>,
    pub right_laser_end: Option<glam::Vec3>,
    /// Remote co-edit peers: a hand skeleton + its color each.
    pub remote_hands: Vec<(HandSkeleton, [f32; 3])>,
    /// Remote co-edit heads: (world position, radius, color).
    pub remote_heads: Vec<([f32; 3], f32, [f32; 3])>,
}

/// A camera-facing solid quad for one hand bone (a→b), thickness = width.
fn hand_bone_quad(
    vp: glam::Mat4,
    cam: glam::Vec3,
    a: glam::Vec3,
    b: glam::Vec3,
    thickness: f32,
    rgb: [f32; 3],
) -> Option<vr_overlay::QuadPush> {
    let seg = b - a;
    let len = seg.length();
    if len <= 1e-4 {
        return None;
    }
    let d = seg / len;
    let mid = (a + b) * 0.5;
    let to_cam = (cam - mid).normalize_or(glam::Vec3::Z);
    let thin = d.cross(to_cam).normalize_or(glam::Vec3::Y);
    let nrm = thin.cross(d).normalize_or(to_cam);
    let rot = glam::Mat3::from_cols(d, thin, nrm);
    let model = glam::Mat4::from_translation(mid)
        * glam::Mat4::from_mat3(rot)
        * glam::Mat4::from_scale(glam::Vec3::new(len, thickness, 1.0));
    Some(vr_overlay::QuadPush {
        mvp: (vp * model).to_cols_array_2d(),
        params: [1.0, rgb[0], rgb[1], rgb[2]],
        uv_rect: [0.0; 4],
    })
}

/// A camera-facing square quad for one hand joint.
fn hand_dot_quad(vp: glam::Mat4, cam: glam::Vec3, p: glam::Vec3, size: f32, rgb: [f32; 3]) -> vr_overlay::QuadPush {
    let to_cam = (cam - p).normalize_or(glam::Vec3::Z);
    let right = to_cam.cross(glam::Vec3::Y).normalize_or(glam::Vec3::X);
    let up = right.cross(to_cam);
    let rot = glam::Mat3::from_cols(right, up, to_cam);
    let model = glam::Mat4::from_translation(p)
        * glam::Mat4::from_mat3(rot)
        * glam::Mat4::from_scale(glam::Vec3::new(size, size, 1.0));
    vr_overlay::QuadPush {
        mvp: (vp * model).to_cols_array_2d(),
        params: [1.0, rgb[0], rgb[1], rgb[2]],
        uv_rect: [0.0; 4],
    }
}

/// Fixed-size offscreen render target for the scene's main camera, shown in
/// the editor's Camera tab. Rendered each frame the tab needs it, then sampled
/// by egui as a registered user texture.
struct CameraPreview {
    extent: vk::Extent2D,
    color: vk::Image,
    color_view: vk::ImageView,
    color_alloc: Option<Allocation>,
    depth: vk::Image,
    /// Depth-aspect view (sampled by fog / Hi-Z).
    depth_view: vk::ImageView,
    /// Attachment view (depth + stencil aspects).
    depth_att_view: vk::ImageView,
    depth_alloc: Option<Allocation>,
    /// Linear HDR scene target; the post pass (tonemap + bloom + grading)
    /// resolves it into `color`, like the viewport.
    hdr_image: vk::Image,
    hdr_view: vk::ImageView,
    hdr_alloc: Option<Allocation>,
    post_pool: vk::DescriptorPool,
    post_set: vk::DescriptorSet,
    bloom: Option<bloom::BloomChain>,
    /// Per-frame reflection targets; Camera must not overwrite Scene bindings.
    reflections: Vec<ssr::SsrTarget>,
    /// Per-camera Flux targets (sampleable depth prepass + ping-pong gather), so
    /// the preview gets its own screen-space GI traced from the game camera.
    sgi: ScreenGiTargets,
    /// Per-frame-in-flight camera UBOs + their set-0 descriptor sets.
    ubos: Vec<Buffer>,
    sets: Vec<vk::DescriptorSet>,
    ubo_pool: vk::DescriptorPool,
    /// egui user-texture plumbing (combined image sampler) for the color image.
    egui_layout: vk::DescriptorSetLayout,
    egui_pool: vk::DescriptorPool,
    texture_id: egui::TextureId,
}

impl CameraPreview {
    #[allow(clippy::too_many_arguments)]
    fn new(
        ctx: &GpuContext,
        allocator: &mut Allocator,
        set0_layout: vk::DescriptorSetLayout,
        color_format: vk::Format,
        sampler: vk::Sampler,
        shadow_view: vk::ImageView,
        shadow_sampler: vk::Sampler,
        probe_buffer: vk::Buffer,
        lightmap_view: vk::ImageView,
        lightmap_sampler: vk::Sampler,
        env_cube_view: vk::ImageView,
        env_b_view: vk::ImageView,
        env_c_view: vk::ImageView,
        env_d_view: vk::ImageView,
        ripple_view: vk::ImageView,
        neutral_view: vk::ImageView,
        ao_view: vk::ImageView,
        post_layout: vk::DescriptorSetLayout,
        post_sampler: vk::Sampler,
        egui: &mut egui_ash_renderer::Renderer,
        extent: vk::Extent2D,
    ) -> Result<Self> {
        let device = &ctx.device;
        let (hdr_image, hdr_alloc) = create_image(
            device,
            allocator,
            post::HDR_FORMAT,
            // TRANSFER_SRC: the impostor capture reads the linear HDR texels back.
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
            extent,
            "camera preview hdr",
        )?;
        let hdr_view = create_view(device, hdr_image, post::HDR_FORMAT, vk::ImageAspectFlags::COLOR)?;
        let post_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: 1,
                    },
                ]),
                None,
            )?
        };
        let post_set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(post_pool)
                    .set_layouts(&[post_layout]),
            )?[0]
        };
        let post_info = [vk::DescriptorImageInfo::default()
            .sampler(post_sampler)
            .image_view(hdr_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        unsafe {
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(post_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&post_info)],
                &[],
            )
        };

        let (color, color_alloc) = create_image(
            device,
            allocator,
            color_format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
            extent,
            "camera preview color",
        )?;
        let color_view = create_view(device, color, color_format, vk::ImageAspectFlags::COLOR)?;
        let (depth, depth_alloc) = create_image(
            device,
            allocator,
            depth_format(),
            // Sampled by the underwater fog; copied out for the model preview / impostor capture.
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
            extent,
            "camera preview depth",
        )?;
        let depth_view = create_view(device, depth, depth_format(), vk::ImageAspectFlags::DEPTH)?;
        let depth_att_view = create_view(device, depth, depth_format(), depth_aspect())?;

        // Own pool for the per-frame camera view UBOs (binding 0) + shadow
        // sampler (binding 1).
        let ubo_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(FRAMES_IN_FLIGHT as u32)
                    .pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: FRAMES_IN_FLIGHT as u32,
                        },
                        vk::DescriptorPoolSize {
                            // shadow(1)+lightmap(3)+GI(4)+SSR depth(5)+color(6)
                            // +env cube(7)+planar(8)
                            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: SET0_SAMPLERS * FRAMES_IN_FLIGHT as u32,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: FRAMES_IN_FLIGHT as u32,
                        },
                    ]),
                None,
            )?
        };
        let mut ubos = Vec::new();
        let mut sets = Vec::new();
        for i in 0..FRAMES_IN_FLIGHT {
            let buffer = Buffer::new(
                device,
                allocator,
                size_of::<FrameUbo>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
                &format!("camera ubo {i}"),
            )?;
            let layouts = [set0_layout];
            let set = unsafe {
                device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(ubo_pool)
                        .set_layouts(&layouts),
                )?[0]
            };
            let buffer_info = [vk::DescriptorBufferInfo::default()
                .buffer(buffer.handle)
                .range(size_of::<FrameUbo>() as u64)];
            let shadow_info = [vk::DescriptorImageInfo::default()
                .sampler(shadow_sampler)
                .image_view(shadow_view)
                .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)];
            let probe_info = [vk::DescriptorBufferInfo::default()
                .buffer(probe_buffer)
                .range(vk::WHOLE_SIZE)];
            let lightmap_info = [vk::DescriptorImageInfo::default()
                .sampler(lightmap_sampler)
                .image_view(lightmap_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let env_info = [vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(env_cube_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let env_b_info = [vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(env_b_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let env_c_info = [vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(env_c_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let env_d_info = [vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(env_d_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let neutral = [vk::DescriptorImageInfo::default().sampler(sampler)
                .image_view(neutral_view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let ao = [vk::DescriptorImageInfo::default().sampler(sampler)
                .image_view(ao_view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let ripples = ripple_infos(sampler, [ripple_view; MAX_WATER_BODIES]);
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&buffer_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&shadow_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&probe_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&lightmap_info),
                // Binding 4 (screen-GI) — the preview doesn't run the gather, so
                // just point it at a valid image; the shader gates its use on the
                // screen-GI-active flag (off for the preview).
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&neutral),
                // Bindings 5/6 (SSR depth + colour) — preview never runs SSR
                // (frame.ssr.x stays 0), so point them at a valid image for set
                // completeness only.
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(5)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&neutral),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(6)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&neutral),
                // Binding 7: the environment reflection cube (shared with the
                // main renderer; re-pointed by set_skybox on rebuild).
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(7)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&env_info),
                // Binding 8 (planar reflections) — never sampled by the preview
                // (its UBO leaves the planar flag 0); valid placeholder only.
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(8)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&lightmap_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(9)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&env_b_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(11)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&env_c_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(12)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&env_d_info),
                vk::WriteDescriptorSet::default().dst_set(set).dst_binding(13)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&ao),
                ripple_write(set, &ripples),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
            ubos.push(buffer);
            sets.push(set);
        }

        // Register the color image with egui as a user texture. The set layout
        // is structurally identical to egui's internal one, so the set is
        // compatible with its pipeline.
        let egui_layout = egui_ash_renderer::vulkan::create_vulkan_descriptor_set_layout(device)
            .map_err(|e| anyhow::anyhow!("camera preview descriptor layout: {e}"))?;
        let egui_pool = egui_ash_renderer::vulkan::create_vulkan_descriptor_pool(device, 1)
            .map_err(|e| anyhow::anyhow!("camera preview descriptor pool: {e}"))?;
        let egui_set = egui_ash_renderer::vulkan::create_vulkan_descriptor_set(
            device,
            egui_layout,
            egui_pool,
            color_view,
            sampler,
        )
        .map_err(|e| anyhow::anyhow!("camera preview descriptor set: {e}"))?;
        let texture_id = egui.add_user_texture(egui_set);

        let sgi = ScreenGiTargets::new(ctx, allocator, extent)?;

        Ok(Self {
            extent,
            color,
            color_view,
            color_alloc: Some(color_alloc),
            depth,
            depth_view,
            depth_att_view,
            depth_alloc: Some(depth_alloc),
            hdr_image,
            hdr_view,
            hdr_alloc: Some(hdr_alloc),
            post_pool,
            post_set,
            bloom: None,
            reflections: Vec::new(),
            sgi,
            ubos,
            sets,
            ubo_pool,
            egui_layout,
            egui_pool,
            texture_id,
        })
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        self.sgi.destroy(device, allocator);
        for target in &mut self.reflections { target.destroy(device, allocator); }
        for b in &mut self.ubos {
            b.destroy(device, allocator);
        }
        unsafe {
            device.destroy_descriptor_pool(self.ubo_pool, None);
            device.destroy_descriptor_pool(self.egui_pool, None);
            device.destroy_descriptor_set_layout(self.egui_layout, None);
            device.destroy_image_view(self.color_view, None);
            device.destroy_image(self.color, None);
            device.destroy_image_view(self.depth_view, None);
            device.destroy_image_view(self.depth_att_view, None);
            device.destroy_image(self.depth, None);
            device.destroy_descriptor_pool(self.post_pool, None);
            device.destroy_image_view(self.hdr_view, None);
            device.destroy_image(self.hdr_image, None);
        }
        if let Some(mut b) = self.bloom.take() {
            b.destroy(device, allocator);
        }
        if let Some(a) = self.color_alloc.take() {
            let _ = allocator.free(a);
        }
        if let Some(a) = self.depth_alloc.take() {
            let _ = allocator.free(a);
        }
        if let Some(a) = self.hdr_alloc.take() {
            let _ = allocator.free(a);
        }
    }
}

/// Offscreen target for the EDITOR viewport. The 3D scene renders here sized to
/// the viewport dock rect (so it isn't rasterized at full-window resolution under
/// the panels) and egui shows it as an image. Editor-only: the game/runtime has
/// no `ViewportTarget` and renders straight to the swapchain unchanged. Reuses
/// the main camera descriptor set, so it only holds the color/depth surfaces +
/// the egui texture.
#[allow(dead_code)] // wired into render() in stage 2 of the viewport-RTT refactor
struct ViewportTarget {
    extent: vk::Extent2D,
    color: vk::Image,
    color_view: vk::ImageView,
    color_alloc: Option<Allocation>,
    depth: vk::Image,
    /// Depth-aspect view (sampled by fog / Hi-Z).
    depth_view: vk::ImageView,
    /// Attachment view (depth + stencil aspects).
    depth_att_view: vk::ImageView,
    depth_alloc: Option<Allocation>,
    /// Rect-sized Flux targets for the editor main-camera trace (the game keeps
    /// the renderer's own `self.sgi`).
    sgi: ScreenGiTargets,
    /// The `sgi` images hold no history yet (just created). The temporal
    /// state is shared with the main path, which can re-validate it before the
    /// viewport's first trace, so the target tracks its own freshness.
    sgi_fresh: bool,
    /// Previous-frame colour for SSR in the editor viewport.
    /// HDR scene target + post descriptor set: the viewport scene renders linear
    /// HDR here, then the post pass (tonemap + bloom) writes `color`.
    hdr_image: vk::Image,
    hdr_view: vk::ImageView,
    hdr_alloc: Option<Allocation>,
    post_pool: vk::DescriptorPool,
    post_set: vk::DescriptorSet,
    bloom: Option<bloom::BloomChain>,
    egui_layout: vk::DescriptorSetLayout,
    egui_pool: vk::DescriptorPool,
    texture_id: egui::TextureId,
}

#[allow(dead_code)] // wired in stage 2 of the viewport-RTT refactor
impl ViewportTarget {
    #[allow(clippy::too_many_arguments)]
    fn new(
        ctx: &GpuContext,
        allocator: &mut Allocator,
        color_format: vk::Format,
        sampler: vk::Sampler,
        post_layout: vk::DescriptorSetLayout,
        post_sampler: vk::Sampler,
        extent: vk::Extent2D,
        egui: &mut egui_ash_renderer::Renderer,
    ) -> Result<Self> {
        let device = &ctx.device;
        let (color, color_alloc) = create_image(
            device,
            allocator,
            color_format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
            extent,
            "viewport color",
        )?;
        // HDR scene target + a post descriptor set pointing at it.
        let (hdr_image, hdr_alloc) = create_image(
            device,
            allocator,
            post::HDR_FORMAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
            extent,
            "viewport hdr",
        )?;
        let hdr_view = create_view(device, hdr_image, post::HDR_FORMAT, vk::ImageAspectFlags::COLOR)?;
        let post_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: 1,
                    },
                ]),
                None,
            )?
        };
        let post_set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(post_pool)
                    .set_layouts(&[post_layout]),
            )?[0]
        };
        let post_info = [vk::DescriptorImageInfo::default()
            .sampler(post_sampler)
            .image_view(hdr_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        unsafe {
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(post_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&post_info)],
                &[],
            )
        };
        let color_view = create_view(device, color, color_format, vk::ImageAspectFlags::COLOR)?;
        let (depth, depth_alloc) = create_image(
            device,
            allocator,
            depth_format(),
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
            extent,
            "viewport depth",
        )?;
        let depth_view = create_view(device, depth, depth_format(), vk::ImageAspectFlags::DEPTH)?;
        let depth_att_view = create_view(device, depth, depth_format(), depth_aspect())?;
        let egui_layout = egui_ash_renderer::vulkan::create_vulkan_descriptor_set_layout(device)
            .map_err(|e| anyhow::anyhow!("viewport descriptor layout: {e}"))?;
        let egui_pool = egui_ash_renderer::vulkan::create_vulkan_descriptor_pool(device, 1)
            .map_err(|e| anyhow::anyhow!("viewport descriptor pool: {e}"))?;
        let egui_set = egui_ash_renderer::vulkan::create_vulkan_descriptor_set(
            device, egui_layout, egui_pool, color_view, sampler,
        )
        .map_err(|e| anyhow::anyhow!("viewport descriptor set: {e}"))?;
        let texture_id = egui.add_user_texture(egui_set);
        let sgi = ScreenGiTargets::new(ctx, allocator, extent)?;
        // SSR samples the viewport's previous HDR scene colour.
        Ok(Self {
            extent,
            color,
            color_view,
            color_alloc: Some(color_alloc),
            depth,
            depth_view,
            depth_att_view,
            depth_alloc: Some(depth_alloc),
            sgi,
            sgi_fresh: true,
            hdr_image,
            hdr_view,
            hdr_alloc: Some(hdr_alloc),
            post_pool,
            post_set,
            bloom: None,
            egui_layout,
            egui_pool,
            texture_id,
        })
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        self.sgi.destroy(device, allocator);
        if let Some(mut b) = self.bloom.take() {
            b.destroy(device, allocator);
        }
        unsafe {
            device.destroy_descriptor_pool(self.post_pool, None);
            device.destroy_image_view(self.hdr_view, None);
            device.destroy_image(self.hdr_image, None);
            device.destroy_descriptor_pool(self.egui_pool, None);
            device.destroy_descriptor_set_layout(self.egui_layout, None);
            device.destroy_image_view(self.color_view, None);
            device.destroy_image(self.color, None);
            device.destroy_image_view(self.depth_view, None);
            device.destroy_image_view(self.depth_att_view, None);
            device.destroy_image(self.depth, None);
        }
        if let Some(a) = self.hdr_alloc.take() {
            let _ = allocator.free(a);
        }
        if let Some(a) = self.color_alloc.take() {
            let _ = allocator.free(a);
        }
        if let Some(a) = self.depth_alloc.take() {
            let _ = allocator.free(a);
        }
    }
}

/// Shadow-map array: each shadow-casting light reserves layers (1 for
/// directional/spot, 6 for point). Sampled as a `sampler2DArrayShadow`.
/// Metadata capacity. Physical layers are allocated on demand, with their
/// resolution reduced to keep the previous 12-map texel budget. Seven point
/// lights need 42 maps; silently assigning only the nearest two lost shadows.
const MAX_SHADOW_MAPS: usize = 64;
const SHADOW_BUDGET_LAYERS: usize = 12;

fn shadow_atlas_dim(requested: u32, layers: usize) -> u32 {
    let budget = requested as u64 * requested as u64 * SHADOW_BUDGET_LAYERS as u64;
    let limit = ((budget / layers.max(1) as u64) as f64).sqrt() as u32;
    // Small quantization keeps resolution stable when the number of lights is
    // unchanged, without throwing away a whole power of two of useful detail.
    requested.min((limit / 64).max(1) * 64)
}

fn shadow_layer_demand(input: &FrameInput, cascade_count: usize) -> usize {
    let mut directional = false;
    input.lights.iter().take(MAX_LIGHTS).filter(|l| l.cast_shadows).map(|l| {
        match l.kind {
            LightKind::Point => 6,
            LightKind::Directional if !directional => { directional = true; cascade_count },
            _ => 1,
        }
    }).sum::<usize>().clamp(SHADOW_BUDGET_LAYERS, MAX_SHADOW_MAPS)
}
/// Default shadow-map resolution per layer (runtime-configurable).
const SHADOW_DIM: u32 = 2048;
/// Cascade count for the directional (sun) shadow.
const NUM_CASCADES: usize = 4;

struct ShadowAtlas {
    layers: usize,
    /// Resolution per layer (pixels); runtime-configurable.
    dim: u32,
    image: vk::Image,
    alloc: Option<Allocation>,
    /// Whole-array view used for sampling (set 0 binding 1).
    array_view: vk::ImageView,
    /// One single-layer view per array layer, used as a render target.
    layer_views: Vec<vk::ImageView>,
    /// Depth-compare sampler (PCF).
    sampler: vk::Sampler,
}

impl ShadowAtlas {
    fn new(ctx: &GpuContext, allocator: &mut Allocator, requested: u32, layers: usize) -> Result<Self> {
        let device = &ctx.device;
        let dim = shadow_atlas_dim(requested, layers);
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(DEPTH_ONLY_FORMAT)
            .extent(vk::Extent3D {
                width: dim,
                height: dim,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(layers as u32)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&info, None)? };
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let alloc = allocator
            .allocate(&AllocationCreateDesc {
                name: "shadow atlas",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .with_context(|| "allocating GPU memory for shadow atlas")?;
        unsafe { device.bind_image_memory(image, alloc.memory(), alloc.offset())? };

        let array_view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
                    .format(DEPTH_ONLY_FORMAT)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .level_count(1)
                            .layer_count(layers as u32),
                    ),
                None,
            )?
        };
        let mut layer_views = Vec::with_capacity(layers);
        for layer in 0..layers as u32 {
            layer_views.push(unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(DEPTH_ONLY_FORMAT)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                .level_count(1)
                                .base_array_layer(layer)
                                .layer_count(1),
                        ),
                    None,
                )?
            });
        }

        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
                    .compare_enable(true)
                    .compare_op(vk::CompareOp::LESS_OR_EQUAL),
                None,
            )?
        };

        Ok(Self {
            layers,
            dim,
            image,
            alloc: Some(alloc),
            array_view,
            layer_views,
            sampler,
        })
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.array_view, None);
            for view in self.layer_views.drain(..) {
                device.destroy_image_view(view, None);
            }
            device.destroy_image(self.image, None);
        }
        if let Some(a) = self.alloc.take() {
            let _ = allocator.free(a);
        }
    }
}

/// `ZESTENGINE_FRAME_STATS` set: record the GPU timestamp zones without the
/// profiler window (frame-time logging in game builds).
fn gpu_timing_forced() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // The avatar benchmark needs the skin / morph zones too.
    *ON.get_or_init(|| {
        std::env::var_os("ZESTENGINE_FRAME_STATS").is_some() || std::env::var_os("ZESTENGINE_BENCH_AVATARS").is_some()
    })
}

pub struct Renderer {
    ctx: GpuContext,
    allocator: Option<Arc<Mutex<Allocator>>>,
    swapchain: Swapchain,
    depth: DepthTarget,
    /// Screen-space GI: depth prepass + gather output (recreated on resize).
    sgi: ScreenGiTargets,
    /// Screen-space ambient occlusion (pass, per-path targets): `ao.rs`.
    ao: ao::AoState,
    /// Anti-aliasing (FXAA flags, TAA targets + history): `aa.rs`.
    aa: aa::AaState,
    /// Previous-frame lit colour for SSR (swapchain path; recreated on resize).
    /// HDR scene target + fullscreen post pass (tonemap + bloom) for the game
    /// swapchain path. The editor viewport keeps inline tonemap for now.
    post_pass: post::PostPass,
    /// Resolution scale (fraction of the output the scene renders at; 1.0 =
    /// native). The upscaler reconstructs the full frame from this.
    render_scale: f32,
    /// Active upscaler (Off = native, Fsr3 = temporal upscaling).
    upscaler: Upscaler,
    /// Reduced-resolution render extent for the current frame (`render_scale`
    /// of the output size); equals the output extent at native.
    render_extent: [u32; 2],
    /// Sub-pixel jitter for the current frame (Halton(2, 3), in [0, 1)²) — the
    /// temporal upscaler's per-frame camera offset.
    jitter: [f32; 2],
    /// Previous frame's jittered view-projection (motion-vector reprojection).
    prev_view_proj: [[f32; 4]; 4],
    /// True while recording the main scene pass with the motion-vector
    /// attachment bound (the upscaler is active); gates the `velocity` variant.
    velocity_active: bool,
    /// Deferred screen-space-reflection resolve pass (shared pipeline) + the game
    /// swapchain path's per-frame G-buffer/resolved targets.
    ssr_resolve: ssr::SsrResolve,
    ssr_targets: Vec<ssr::SsrTarget>,
    /// Motion-vector attachments (RG16F), one per frame in flight, written by
    /// the main scene pass when the upscaler is active.
    velocity_targets: Vec<VelocityTarget>,
    /// Reactive-mask attachments (R8), one per frame in flight.
    reactive_targets: Vec<ReactiveTarget>,
    /// The FSR3 upscaler (backend interface + context); `Some` only when the
    /// active upscaler is `Fsr3` and the reduced-resolution targets exist.
    fsr3: Option<fsr3::Fsr3>,
    /// The combined FSR3 context (upscaling + frame interpolation); `Some`
    /// when frame generation is active (replaces `fsr3`).
    fsr3_fg: Option<fsr3::Fsr3FrameGen>,
    /// Frame generation (FSR3 interpolation), gated behind the upscaler.
    frame_gen: bool,
    /// Display-space present target (tonemapped scene, pre-UI) the frame gen
    /// consumes, plus its frame-interpolation output.
    fg_present: Vec<UpscaleOutput>,
    fg_output: Vec<UpscaleOutput>,
    /// Full-resolution upscale outputs, one per frame in flight.
    upscale_outputs: Vec<UpscaleOutput>,
    /// Descriptor sets binding each upscale output (binding 0) for the post pass.
    upscale_pool: vk::DescriptorPool,
    upscale_sets: Vec<vk::DescriptorSet>,
    /// Reset FSR temporal history next frame (first frame / resize / toggle).
    fsr3_reset: bool,
    /// Previous frame's `input.time` (frame-time delta for the upscaler).
    last_time: f32,
    /// Underwater view fog over the HDR scene (every view kind).
    underwater: underwater::UnderwaterPass,
    /// Screen-space selection outline of the editor viewport.
    selection: selection::SelectionPass,
    /// Editor ground grid of the viewport (off unless the editor turns it on).
    grid: grid::GridPass,
    /// Deferred projected-decal pass (shared pipeline + frame sets).
    decal: decal::DecalPass,
    /// Editor-viewport deferred-SSR target (recreated with `viewport_target`).
    viewport_ssr: Option<ssr::SsrTarget>,
    /// Ray-traced reflections (1 bounce); `None` when the GPU lacks ray query.
    rt_reflect: Option<rt_reflect::RtReflect>,
    /// Eye the next un-indexed `render_xr_eye` call is taken for (alternates).
    xr_eye_auto: usize,
    /// VR stereo render resources (lazily created on the first `render_xr_eye`).
    xr_render: Option<XrRender>,
    /// VR UI overlay: egui→texture + the hand-panel/pointer quads (lazy).
    vr_overlay: Option<vr_overlay::VrOverlay>,
    /// The most recent desktop frame UBO (lights/shadows/probes), reused per VR
    /// eye with only the camera matrices swapped.
    last_frame_ubo: Option<FrameUbo>,
    /// Prefiltered environment reflection cubemap (reflection probe), rebuilt
    /// from the skybox. Sampled in the forward shader's specular term.
    reflection_env: ReflectionEnv,
    /// Extra reflection-probe slots (bindings 9/11/12), blended per pixel.
    reflection_env_b: ReflectionEnv,
    reflection_env_c: ReflectionEnv,
    reflection_env_d: ReflectionEnv,
    /// Dedicated set-0 + UBO for the precomputed scene reflection capture (6-face
    /// render from a probe position into the env cube). Set once, reused per face.
    refl_capture_pool: vk::DescriptorPool,
    /// Capture set + UBO per frame in flight (faces record into frame buffers).
    refl_capture_sets: Vec<vk::DescriptorSet>,
    refl_capture_ubos: Vec<Buffer>,
    refl_target: Option<ReflCaptureTarget>,
    /// Depth-prepass set-0 copies (main + preview camera, per frame): the frame
    /// sets are rewritten after the prepass records, which would invalidate a
    /// command buffer that had bound them.
    prepass_sets: Vec<vk::DescriptorSet>,
    refl_last_finish: Option<std::time::Instant>,
    /// Planar reflections: RT array (lazily sized to the main target) + one
    /// pre-allocated set-0/UBO per plane pass (all recorded in the frame cb).
    /// The sets live in `refl_capture_pool`.
    planar_targets: Option<PlanarTargets>,
    /// Render-to-texture cameras (see `camera_target`).
    camera_targets: Vec<CameraTarget>,
    /// Pool for the camera targets' set-0 copies (created with the first one).
    camera_target_pool: vk::DescriptorPool,
    planar_sets: Vec<vk::DescriptorSet>,
    planar_ubos: Vec<Buffer>,
    /// World-space center for the next scene reflection capture; `Some` requests a
    /// (re)capture on the next frame (the scene reflects its surroundings, not just
    /// the sky). Consumed once into `refl_job`, then cleared.
    recapture_reflection: Option<ReflCaptureRequest>,
    /// In-progress reflection capture, stepped one cube face per frame so the
    /// 6-face render (plus first-time pipeline compile) never stalls a single
    /// frame. `None` when idle.
    refl_job: Option<ReflCaptureJob>,
    /// What the last frame's views were, for the scene's animator culling
    /// (see [`Renderer::view_cull_info`]).
    view_cull: ViewCullInfo,
    /// True while rendering the 6 faces of a reflection-probe cube. The cube
    /// capture uses a positive-height (non-Y-flipped) viewport, which mirrors
    /// triangle winding relative to the main passes, so the standard CCW/BACK
    /// pipelines would cull *front* faces and render geometry inside-out (floors
    /// vanish, spheres show their backs). Forcing no-cull (`double_sided`) during
    /// capture sidesteps that — the same trick the bake gbuffer pass uses.
    capturing_refl: bool,
    /// Flip Y in the screen-GI depth→world reconstruction (runtime toggle so the
    /// Y-flipped-viewport convention can be corrected without a rebuild).
    screen_gi_flip_y: bool,
    /// Whether the screen-GI image holds a valid previous-frame result (for
    /// temporal accumulation). False on the first frame + after a resize.
    sgi_history_valid: bool,
    gi_hit_lighting: [bool; 3],
    /// Emissive sphere area-lights for the screen-GI gather (analytic NEE, so
    /// the emitter bounce stays smooth without the coarse-GDF footprint). Set by
    /// the realtime-GI driver alongside the GDF.
    gi_emitters: Vec<gpu_gi::GpuGiEmitter>,
    /// Runtime Flux trace parameters (Environment tab → Flux GI).
    flux_settings: FluxSettings,
    /// Previous depth prepass view-projection, including temporal AA jitter.
    sgi_last_view_proj: Option<glam::Mat4>,
    /// Main camera pass lights (position + range), for per-draw light masks.
    cull_lights: Vec<[f32; 4]>,
    /// True while recording the main camera pass: built-in draws get a mask of
    /// the lights that can reach them. Other passes light with every light.
    cull_active: bool,
    /// Ping-pong parity: which gather image is current (written this trace); the
    /// other holds the previous frame's result for temporal reprojection.
    sgi_parity: usize,
    /// Monotonic trace counter feeding the trace RNG seed. Without per-frame
    /// variation every frame samples the identical random directions, so the
    /// Monte Carlo noise is fixed-pattern and temporal accumulation can never
    /// average it out; that was the source of the static blotches.
    sgi_frame: u32,
    /// Per-frame-in-flight Flux trace resources (descriptor pool + host buffers)
    /// kept alive after the trace is recorded into the main cb, freed one frame
    /// later once that frame's fence has signalled.
    sgi_transients: Vec<Option<gpu_gi::ScreenGiTransient>>,
    /// Retired trace resources per frame slot, reused by that slot's next trace.
    sgi_spare: Vec<Option<gpu_gi::ScreenGiTransient>>,
    preview_sgi_spare: Vec<Option<gpu_gi::ScreenGiTransient>>,
    /// Temporal state + transient ring for the in-game camera preview's own Flux
    /// trace (mirrors the editor-viewport state above, but for the game camera).
    preview_sgi_parity: usize,
    preview_sgi_last_view_proj: Option<glam::Mat4>,
    preview_sgi_last_cam: glam::Vec3,
    preview_sgi_history_valid: bool,
    preview_sgi_frame: u32,
    preview_sgi_transients: Vec<Option<gpu_gi::ScreenGiTransient>>,
    /// Camera position the previous gather was traced from (reprojection).
    sgi_last_cam: glam::Vec3,
    command_pool: vk::CommandPool,
    frames: Vec<Frame>,
    frame_index: usize,
    /// Swapchain rebuild decision: latest window size once per frame, forced
    /// on out-of-date / suboptimal / vsync change (resize_gate.rs).
    resize_gate: resize_gate::ResizeGate,

    pipeline_cache: PipelineCache,
    descriptor_pool: vk::DescriptorPool,
    material_pool: vk::DescriptorPool,
    /// Set-2 pool for custom-shader materials; reset with `material_pool`.
    custom_pool: vk::DescriptorPool,
    /// Dedicated pool for the skybox set-1 (kept out of `material_pool` so the
    /// scene-reset path can reset the material pools without invalidating the
    /// skybox set - a sky draw with a set from a reset pool crashes the driver).
    skybox_pool: vk::DescriptorPool,
    /// Shared zero FX block bound to non-material set-1 users (e.g. the skybox
    /// set) so set-1 binding 4 is always valid.
    default_fx_ubo: Buffer,
    frame_ubos: Vec<Buffer>,
    frame_sets: Vec<vk::DescriptorSet>,
    sampler: vk::Sampler,
    /// Per-texture sampler state (wrap/filter/aniso); the default key is `sampler`.
    sampler_cache: sampler_cache::SamplerCache,
    /// Linear CLAMP_TO_EDGE sampler for baked lightmaps. The material `sampler`
    /// is REPEAT (tiling), which wraps at the atlas border and bleeds chart
    /// edges into the opposite side.
    lightmap_sampler: vk::Sampler,
    /// Zeroed 1x1 field bound in set-0 binding 10 for body slots without
    /// a ripple texture.
    ripple_dummy: GpuTexture,
    default_textures: Vec<GpuTexture>, // [albedo, normal, orm, emission]
    /// Fullscreen skybox: descriptor set (set 1) + optional equirect texture.
    /// When `skybox_has_texture` is false the shader draws a procedural sky.
    skybox_set: vk::DescriptorSet,
    skybox_texture: Option<GpuTexture>,
    skybox_has_texture: bool,
    /// True when the skybox is a cubemap (rendered from the env cube, binding 7)
    /// rather than the equirect texture.
    skybox_is_cube: bool,
    /// Shadow-map array (set 0 binding 1) + per-layer render targets.
    shadow_atlas: ShadowAtlas,
    shadow_requested_dim: u32,
    /// Signature of what each shadow layer last rendered (see `record_shadows`).
    shadow_layer_sig: [u64; MAX_SHADOW_MAPS],
    /// The atlas layers hold valid depth in DEPTH_READ_ONLY (false after a resize).
    shadow_layers_valid: bool,
    /// Per-mesh vertex-update counter (skinning / CPU deformation).
    mesh_versions: Vec<u32>,

    meshes: Vec<GpuMesh>,
    textures: Vec<GpuTexture>,
    materials: Vec<Material>,
    /// Renderer resource indices freed by `free_resources` but still present
    /// as tombstones (indices stay stable, so live handles keep working).
    freed_meshes: HashSet<usize>,
    freed_textures: HashSet<usize>,
    freed_materials: HashSet<usize>,

    egui: Option<egui_ash_renderer::Renderer>,
    /// A second egui renderer dedicated to the VR UI panel (its own in-flight
    /// buffers, so re-drawing the UI for VR can't race the desktop egui draw).
    /// Kept texture-synced with `egui`. None on the game runtime.
    vr_egui: Option<egui_ash_renderer::Renderer>,
    /// egui texture ids freed by a frame, parked on that frame's slot
    /// (`egui_free_slots[frame_index]`) and released once the slot's fence
    /// retires again (two frames later), so no in-flight frame can still sample
    /// them and no `device_wait_idle` is needed. If a frame's `set_textures`
    /// names an id that is still parked (a recycled id), `egui_flush_frees`
    /// falls back to one full idle + free first: deferring past the `set`
    /// destroyed the live reused texture -> use-after-free -> SIGSEGV in
    /// egui's next draw (seen on window resize / thumbnail churn).
    egui_free_slots: Vec<Vec<egui::TextureId>>,
    /// GPU skinning pass (created on the first GPU-skinned mesh).
    skinner: Option<skin::GpuSkinner>,
    /// egui renderer for world-space UI canvases (own texture set: the game
    /// UI's egui context is separate from the editor's).
    world_egui: HashMap<usize, egui_ash_renderer::Renderer>,
    /// World-UI frees seen this frame (before its fence wait); moved onto the
    /// frame slot after the wait, released when the slot retires again.
    world_egui_pending_free: Vec<(usize, Vec<egui::TextureId>)>,
    /// SDF/MTSDF text: GPU atlas + pipeline (created lazily once a font is
    /// registered) and the handle UI systems lay text out through.
    text_gpu: Option<text::TextGpu>,
    /// Octahedral impostor billboards (GEO-5); created by the first upload.
    impostor_gpu: Option<impostor::ImpostorGpu>,
    /// 2D sprite pass (2D-3): GPU half, created by the first texture
    /// registration or sprite frame.
    sprite_gpu: Option<sprite_gpu::SpriteGpu>,
    /// CPU side of the pooled sprite meshes (uploaded by the next frame).
    sprite_pool: sprite2d::SpriteMeshPool,
    /// This frame's sprites and masks, filled by the engine, consumed by the frame.
    sprite_input: SpriteFrameInput,
    text_handle: zestengine_text::TextHandle,
    text_failed: bool,
    /// World-UI texture uploads waiting for a frame command buffer (target
    /// texture index, egui deltas). Kept until recorded, so a frame that
    /// aborts before recording (swapchain out of date) loses nothing.
    world_egui_uploads: Vec<(usize, Vec<(egui::TextureId, egui::epaint::ImageDelta)>)>,
    world_egui_free_slots: Vec<Vec<(usize, Vec<egui::TextureId>)>>,
    /// Built-in unlit custom shader, registered on first use.
    unlit_shader: Option<ShaderId>,
    /// Offscreen main-camera target, created lazily the first frame the Camera
    /// tab asks for it.
    camera_preview: Option<CameraPreview>,
    /// Offscreen model preview (the rig Configure window's 3D figure): a
    /// second camera-preview target of its own size, alive only while the
    /// editor asks for it each frame. Replaced / dropped targets wait
    /// `FRAMES_IN_FLIGHT + 1` frames before they are freed.
    model_preview: Option<CameraPreview>,
    retired_model_previews: Vec<(CameraPreview, usize)>,
    /// Model-only preview asked for this frame (`set_model_only_preview`), its
    /// spare draw/order buffers and the last depth readback.
    model_only: Option<offscreen::ModelOnlyState>,
    model_only_spare: Vec<DrawCmd>,
    model_only_order: Vec<usize>,
    model_only_depth: Option<ModelPreviewDepth>,
    /// Textured impostor captures (`start_impostor_capture`): the queue (the
    /// front is worked on, one pass per frame), their target and finished results.
    capture_jobs: std::collections::VecDeque<offscreen::CaptureJob>,
    capture_target: Option<CameraPreview>,
    capture_done: Vec<(u64, Result<CapturedImpostor, String>)>,
    capture_next_id: u64,
    /// Device->host copies of offscreen targets, complete once their frame slot's fence is.
    readbacks: Vec<offscreen::PendingReadback>,
    /// Store the preview depth (the model-only preview's depth readback needs it).
    preview_keep_depth: bool,
    /// Editor-only offscreen target for the viewport 3D (see `ViewportTarget`).
    /// `None` for the game runtime (renders straight to the swapchain).
    #[allow(dead_code)] // read by the RTT render branch (stage 2, in progress)
    viewport_target: Option<ViewportTarget>,
    /// Replaced viewport targets kept alive until no frame can still draw
    /// them: this frame's egui list was built against the old texture id, and
    /// freeing its descriptor set before the draw is recorded (and the GPU is
    /// done with it) segfaults in the driver. Countdown in rendered frames.
    retired_viewport_targets: Vec<(ViewportTarget, usize)>,
    /// Replaced viewport SSR targets, retired on the same countdown.
    retired_viewport_ssr: Vec<(ssr::SsrTarget, usize)>,
    /// GPU frees of meshes / textures / materials released by `free_resources`,
    /// each with a countdown in rendered frames (see `defer_free`).
    deferred_frees: Vec<(usize, DeferredFree)>,
    /// Set-0 binding swaps owed to frame slots whose fence has not retired, the
    /// serial of the last one queued, and the frees of the resources they
    /// replaced (`shared_bind.rs`).
    pending_shared: Vec<shared_bind::PendingShared>,
    shared_seq: u64,
    shared_frees: Vec<(usize, u64, DeferredFree)>,
    /// Debounces the viewport target size while a panel border is dragged.
    viewport_resize: resize_debounce::Debounce,
    /// A pointer button is down (set by the editor each frame).
    viewport_resize_hold: bool,
    /// Clock for the debounce.
    started: std::time::Instant,
    /// Opaque scene copies sampled by water (set 0 bindings 5/6); created the
    /// first frame that draws water, sized to the main target.
    opaque_copy: Option<OpaqueCopy>,
    area_atlas: AreaAtlas,
    /// The LTC matrix table (set 1 binding 23) and its nearest sampler.
    ltc_lut: LtcLut,
    /// Area lights submitted by the scene ([`Renderer::set_area_lights`]); the nearest
    /// `MAX_AREA_LIGHTS` are packed each frame.
    area_lights: Vec<AreaLightInstance>,
    /// This frame's textured area lights: (atlas tile, sampled source view).
    area_sources: Vec<(u32, vk::ImageView)>,
    packed_area_lights: Vec<GpuLight>,
    /// Build the colour copy's mip chain this frame (a transmission material is drawn).
    opaque_mips: bool,
    /// Sampler of the opaque colour copy (binding 6): trilinear, clamped, mips.
    grab_sampler: vk::Sampler,
    /// This frame's main pass copies its opaque result for water.
    water_copy: bool,
    /// Baked light-probe SH (set-0 binding 2). A 1-probe zero buffer until a
    /// bake is loaded, so the binding is always valid.
    probe_buffer: Buffer,
    /// Incremented per `render`; keys the per-frame caches below.
    frame_serial: u64,
    /// Quality preset (`RenderTier::for_platform` at creation).
    tier: RenderTier,
    /// MSAA samples for the multiview XR scene pass (the tier's ask when the
    /// device supports it for colour and depth, else 1).
    xr_msaa: u8,
    rt_scene_cache: Option<(u64, Vec<rt_reflect::RtInstance>, Vec<rt_reflect::RtLight>)>,
    tex_view_cache: Vec<vk::ImageView>,
    /// Reused per `record_scene_draws` call (it runs several times a frame).
    materials_seen_scratch: Vec<bool>,
    /// How `record_scene_draws` keys its pipelines (prepass / EQUAL over it).
    draw_mode: DrawMode,
    /// Probe SH queued by `update_probe_sh`; copied into `probe_buffer` inside
    /// the next frame's command buffer (ordered after in-flight reads).
    probe_pending: Option<Vec<GpuProbe>>,
    /// Per-frame-in-flight staging for that copy.
    probe_staging: Vec<Option<(Buffer, u64)>>,
    /// Texel rewrites queued by `update_ripple_texture` (texture index, RGBA16F
    /// bytes), copied in the next frame's command buffer like the probe SH.
    tex_pending: Vec<(usize, Vec<u8>)>,
    tex_staging: Vec<Option<(Buffer, u64)>>,
    /// GPU water simulation (created on first use; None after a failed
    /// start, see `water_sim_failed`) and the bodies queued for this frame.
    water_sim: Option<water_sim::GpuWaterSim>,
    water_sim_failed: bool,
    /// Ripple textures (by texture index): their mip-0 view, which the
    /// simulation writes as a storage image, and their mip count.
    ripple_mips: HashMap<usize, (vk::ImageView, u32)>,
    water_sim_queue: Vec<WaterSimBody>,
    /// Number of probes the `probe_buffer` is sized for; lets `update_probe_sh`
    /// rewrite it in place (cheap, no realloc/stall) when the count is unchanged.
    probe_count: usize,
    /// Probe-volume metadata mirrored into each frame's FrameUbo (count drives
    /// `misc.w`). Empty = fall back to flat ambient.
    probe_volumes: Vec<GpuProbeVolume>,
    /// What the frame UBO carries: the baked cascades then the Light Volumes (sh_base already offset).
    frame_probe_volumes: Vec<GpuProbeVolume>,
    /// The baked / realtime probes last uploaded (the first part of the SH buffer).
    baked_cache: Vec<GpuProbe>,
    /// Light Volumes ([`Renderer::set_light_volumes`]): their metadata (sh_base within `lv_probes`) and probes.
    light_volumes: Vec<GpuProbeVolume>,
    lv_probes: Vec<GpuProbe>,
    /// Baked lightmap array (set-0 binding 3), one layer per static object. A
    /// 1x1 black single layer until a bake is loaded.
    lightmaps: GpuTexture,
    last_stats: RenderStats,
    /// Draw stats from the editor viewport scene pass (`record_viewport_scene`),
    /// captured because the editor renders the scene there (not the swapchain
    /// path), so otherwise the reported draw counts would be 0.
    viewport_draw_stats: RenderStats,
    /// Test hook ([`Renderer::debug_capture_viewport_frames`]): when `Some`, every
    /// frame copies the viewport colour into a host buffer at the end of its own
    /// command buffer, so frames in flight can be inspected without a stall.
    viewport_captures: Option<Vec<(alloc::Buffer, u32, u32)>>,
    /// Same, for the composed swapchain image (egui over the viewport), copied before present.
    swapchain_captures: Vec<(alloc::Buffer, u32, u32)>,
    vsync: bool,
    /// Nanoseconds per GPU timestamp tick (`limits.timestamp_period`), or 0 when
    /// the device/queue cannot timestamp — disables GPU pass profiling.
    gpu_ts_period: f32,
    /// Mask of the queue's `timestampValidBits` (see `ts_zone_ms`).
    gpu_ts_mask: u64,
    /// Per-frame-slot flag: whether this slot's timestamp pool has been written
    /// at least once, so the readback after the fence wait has valid data.
    gpu_ts_primed: Vec<bool>,
    /// Whether GPU timestamp zones are recorded this frame (period supported AND
    /// the profiler window is open). Set once per frame, read by `gpu_zone_*`.
    ts_active: bool,
    /// Per-zone GPU timestamps requested through [`Renderer::set_gpu_zone_timing`].
    zone_timing: bool,
    /// Latest per-zone GPU timings (ms), indexed by `gpu_zone::*`, read back from
    /// the retired frame slot each frame.
    gpu_zone_ms: [f32; gpu_zone::COUNT],
    /// Per-frame-slot instance matrices for scene draws.
    instance_rings: Vec<InstanceRing>,
    /// GPU-driven culling (created on first use; None while disabled or after
    /// it failed to start, see `gpu_cull_failed`).
    gpu_cull: Option<gpu_cull::GpuCull>,
    gpu_cull_settings: GpuCullSettings,
    gpu_cull_failed: bool,
    /// GPU cluster culling + cluster draws (GEO-4); created when the first DAG
    /// is installed. `cluster_failed`: it could not be created (logged once).
    cluster: Option<cluster::ClusterSystem>,
    cluster_settings: ClusterSettings,
    cluster_failed: bool,
    /// Which meshes the engine put on / kept off the cluster path (refusal report).
    cluster_coverage: ClusterCoverage,
    /// Why each classified mesh is on the ordinary path (the Zestlet coverage view's colours); meshes
    /// drawn as zestlets are absent. Only read while a zestlet viewport mode is active.
    mesh_refusals: HashMap<usize, ClusterRefusal>,
    /// Draw-time fallbacks of the last main-view batch build, per [`ClusterRefusal`] (instances; diagnostics).
    pub(crate) draw_fallbacks: std::cell::Cell<[u32; REFUSAL_REASONS]>,
    /// Which ring `record_scene_draws` appends to: None = this frame slot's,
    /// Some(i) = VR eye slot i (its own fence).
    instance_ring_sel: Option<usize>,
    /// Whether realtime GI tracing is enabled. Gates `gi_has_gdf` so toggling
    /// GI off stops the screen trace while keeping the GDF cached.
    gi_enabled: bool,
    /// Optional second OS window rendering the profiler egui UI (its own
    /// surface/swapchain/egui, sharing this context). None until opened.
    profiler_window: Option<profiler_window::ProfilerWindow>,
    /// GPU software-GI march pipeline (None if compute init failed → CPU march).
    gpu_gi: Option<gpu_gi::GpuGi>,
    flux_surface: Option<flux_surface::SurfaceCache>,
    flux_screen_history: Option<flux_screen_history::ScreenHistory>,

    /// Declared last so it drops last: `GpuContext::drop` destroys the Vulkan
    /// surface, which must happen while the underlying winit window still
    /// exists. EngineApp drops its own `Arc<Window>` clone before the renderer,
    /// so this field can be the final strong reference. Keeping it last means
    /// the window outlives `ctx`/`surface` teardown; otherwise NVIDIA/Wayland
    /// segfaults on exit.
    window: Arc<Window>,
}

/// A window's raw display + window handles, read on the main thread and
/// carried to the renderer-build worker.
#[derive(Clone, Copy)]
pub struct WindowHandles {
    display: raw_window_handle::RawDisplayHandle,
    window: raw_window_handle::RawWindowHandle,
}

// The handles are plain identifiers; only reading them from the window is
// thread-restricted (Windows), and the surface they create is used from one
// thread at a time.
unsafe impl Send for WindowHandles {}

impl WindowHandles {
    /// Read on the thread that owns the window (the main thread).
    pub fn of(window: &Window) -> Result<Self> {
        Ok(Self { display: window.display_handle()?.as_raw(), window: window.window_handle()?.as_raw() })
    }
}

impl Renderer {
    pub fn new(window: Arc<Window>) -> Result<Self> {
        Self::new_with_xr(window, None)
    }

    /// Like [`Renderer::new`], but routes Vulkan instance/device creation through
    /// OpenXR when a headset is present, so a VR session can share this device.
    /// Pass `Some((&xr_instance, system))` from `zestengine_xr::XrContext`.
    pub fn new_with_xr(
        window: Arc<Window>,
        xr: Option<(&openxr::Instance, openxr::SystemId)>,
    ) -> Result<Self> {
        let handles = WindowHandles::of(&window)?;
        Self::new_with_handles(window, handles, xr)
    }

    /// Probed GPU-driven-rendering capabilities (multi-draw-indirect, indirect
    /// count, int64, BDA, limits); see [`GpuCaps`] for the fallbacks.
    pub fn gpu_caps(&self) -> &GpuCaps {
        &self.ctx.caps
    }

    /// Build from handles taken on the main thread (see [`WindowHandles`]), so
    /// the build itself can run on a worker: Windows only allows the window
    /// handle to be read on the thread that owns the window.
    pub fn new_with_handles(
        window: Arc<Window>,
        handles: WindowHandles,
        xr: Option<(&openxr::Instance, openxr::SystemId)>,
    ) -> Result<Self> {
        let (display, handle) = (handles.display, handles.window);
        let xr = xr.map(|(instance, system)| context::XrBootstrap { instance, system });
        let ctx = GpuContext::new(display, handle, c"zestengine", xr)?;

        // GPU timestamp profiling support: `timestamp_period` is ns/tick (0 if
        // the device can't timestamp); the queue family must also report
        // non-zero valid bits. Either missing → GPU pass timings stay at 0.
        let gpu_ts_valid_bits = unsafe {
            ctx.instance
                .get_physical_device_queue_family_properties(ctx.physical_device)
                .get(ctx.queue_family as usize)
                .map(|q| q.timestamp_valid_bits)
                .unwrap_or(0)
        };
        let gpu_ts_period = unsafe {
            let limits = ctx
                .instance
                .get_physical_device_properties(ctx.physical_device)
                .limits;
            let valid_bits = ctx
                .instance
                .get_physical_device_queue_family_properties(ctx.physical_device)
                .get(ctx.queue_family as usize)
                .map(|q| q.timestamp_valid_bits)
                .unwrap_or(0);
            if valid_bits > 0 {
                limits.timestamp_period
            } else {
                0.0
            }
        };

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: ctx.instance.clone(),
            device: ctx.device.clone(),
            physical_device: ctx.physical_device,
            debug_settings: {
                // Debug builds capture an allocation backtrace so a leak
                // report at teardown names the allocating site.
                let mut settings = gpu_allocator::AllocatorDebugSettings::default();
                settings.store_stack_traces = cfg!(debug_assertions);
                settings
            },
            // Must match the device: buffer device address is enabled only
            // when ray tracing (the lighting bake) is available.
            buffer_device_address: ctx.caps.buffer_device_address,
            allocation_sizes: Default::default(),
        })
        .context("creating GPU allocator")?;
        let allocator = Arc::new(Mutex::new(allocator));

        let size = window.inner_size();
        let swapchain = Swapchain::new(&ctx, size.width, size.height, true)?;
        let depth = DepthTarget::new(&ctx, &mut allocator.lock().unwrap(), swapchain.extent)?;
        let sgi =
            ScreenGiTargets::new(&ctx, &mut allocator.lock().unwrap(), swapchain.extent)?;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(ctx.queue_family);
        let command_pool = unsafe { ctx.device.create_command_pool(&pool_info, None)? };
        let post_pass = post::PostPass::new(
            &ctx.device,
            &mut allocator.lock().unwrap(),
            swapchain.format.format,
            swapchain.extent,
            FRAMES_IN_FLIGHT,
            RenderTier::for_platform().bloom_divisor(),
        )?;
        // Deferred SSR: one resolve pipeline + per-frame G-buffer/resolved targets
        // sized to the swapchain (the game path). The viewport owns its own.
        let ssr_resolve = ssr::SsrResolve::new(&ctx.device)?;
        let underwater = underwater::UnderwaterPass::new(&ctx.device, &mut allocator.lock().unwrap())?;
        let decal = decal::DecalPass::new(&ctx.device, &mut allocator.lock().unwrap())?;
        let selection = selection::SelectionPass::new(&ctx.device, depth_format())?;
        let grid = grid::GridPass::new(&ctx.device)?;
        // Ray-traced reflections (only on GPUs with ray query / the bake's accel
        // loader). Falls back to SSR when absent.
        let rt_reflect = if ctx.accel.is_some() && RenderTier::for_platform().reflections() {
            match rt_reflect::RtReflect::new(&ctx.device) {
                Ok(rt) => Some(rt),
                Err(e) => {
                    tracing::warn!("ray-traced reflections unavailable: {e:#}");
                    None
                }
            }
        } else {
            None
        };
        let ssr_targets = {
            let mut alloc = allocator.lock().unwrap();
            (0..FRAMES_IN_FLIGHT)
                .map(|_| {
                    ssr_resolve.create_target(
                        &ctx.device,
                        &mut alloc,
                        post_pass.set_layout(),
                        swapchain.extent,
                    )
                })
                .collect::<Result<Vec<_>>>()?
        };
        let velocity_targets = {
            let mut alloc = allocator.lock().unwrap();
            (0..FRAMES_IN_FLIGHT)
                .map(|_| VelocityTarget::new(&ctx.device, &mut alloc, swapchain.extent))
                .collect::<Result<Vec<_>>>()?
        };
        let reactive_targets = {
            let mut alloc = allocator.lock().unwrap();
            (0..FRAMES_IN_FLIGHT)
                .map(|_| ReactiveTarget::new(&ctx.device, &mut alloc, swapchain.extent))
                .collect::<Result<Vec<_>>>()?
        };
        let reflection_env = ReflectionEnv::neutral(
            &ctx.device,
            &mut allocator.lock().unwrap(),
            command_pool,
            ctx.queue,
        )?;
        // Second probe slot (two-probe blending); neutral until a probe fills it.
        let reflection_env_b = ReflectionEnv::neutral(
            &ctx.device,
            &mut allocator.lock().unwrap(),
            command_pool,
            ctx.queue,
        )?;
        let reflection_env_c = ReflectionEnv::neutral(
            &ctx.device,
            &mut allocator.lock().unwrap(),
            command_pool,
            ctx.queue,
        )?;
        let reflection_env_d = ReflectionEnv::neutral(
            &ctx.device,
            &mut allocator.lock().unwrap(),
            command_pool,
            ctx.queue,
        )?;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(FRAMES_IN_FLIGHT as u32);
        let command_buffers = unsafe { ctx.device.allocate_command_buffers(&alloc_info)? };
        let frames = command_buffers
            .into_iter()
            .map(|cb| Frame::new(&ctx.device, cb))
            .collect::<Result<Vec<_>>>()?;

        let mut pipeline_cache =
            PipelineCache::new(&ctx.device, swapchain.format.format, depth_format(), GBUF_FORMAT)?;
        // Displaced materials need the tessellation stages (and, in XR, multiview tessellation).
        if ctx.tessellation {
            pipeline_cache.enable_tessellation(&ctx.device, ctx.multiview_tessellation)?;
        }

        // Frame UBOs live in their own pool; the material pool is reset
        // wholesale when a scene is unloaded. Each frame set 0 also carries the
        // shadow-map sampler at binding 1.
        let frame_pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: FRAMES_IN_FLIGHT as u32,
            },
            vk::DescriptorPoolSize {
                // shadow(1)+lightmap(3)+screen-GI(4)+SSR depth(5)+SSR color(6)
                // +env cube(7)+planar array(8)
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: SET0_SAMPLERS * FRAMES_IN_FLIGHT as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: FRAMES_IN_FLIGHT as u32,
            },
        ];
        let descriptor_pool = unsafe {
            ctx.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(FRAMES_IN_FLIGHT as u32)
                    .pool_sizes(&frame_pool_sizes),
                None,
            )?
        };
        // +1 set for the skybox, which also draws from this pool (it shares the
        // set-1 layout: 4 samplers + the FX uniform buffer).
        let material_sets = MAX_MATERIALS + 1;
        let material_pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: material_sets * MATERIAL_SLOTS as u32,
            },
            // One FX uniform buffer per material set (+ the skybox set).
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: material_sets,
            },
        ];
        let material_pool = unsafe {
            ctx.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(material_sets)
                    .pool_sizes(&material_pool_sizes),
                None,
            )?
        };
        // The skybox set-1 lives in its own pool: `reset_scene` resets
        // `material_pool`, and a skybox set surviving that reset as a stale
        // handle is a crash on the next sky draw. A few extra sets cover
        // skybox swaps before their deferred frees return the old ones.
        let skybox_pool = unsafe {
            ctx.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(8)
                    .pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 8 * MATERIAL_SLOTS as u32,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: 8,
                        },
                    ]),
                None,
            )?
        };
        let custom_pool = unsafe {
            ctx.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets(MAX_MATERIALS)
                    .pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: MAX_MATERIALS,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: MAX_MATERIALS * pipeline::CUSTOM_TEXTURES as u32,
                        },
                    ]),
                None,
            )?
        };

        // Per-frame UBOs + descriptor sets.
        let mut frame_ubos = Vec::new();
        let mut frame_sets = Vec::new();
        {
            let mut alloc = allocator.lock().unwrap();
            for i in 0..FRAMES_IN_FLIGHT {
                let buffer = Buffer::new(
                    &ctx.device,
                    &mut alloc,
                    size_of::<FrameUbo>() as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    MemoryLocation::CpuToGpu,
                    &format!("frame ubo {i}"),
                )?;
                let layouts = [pipeline_cache.set0_layout];
                let set = unsafe {
                    ctx.device.allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(descriptor_pool)
                            .set_layouts(&layouts),
                    )?[0]
                };
                let buffer_info = [vk::DescriptorBufferInfo::default()
                    .buffer(buffer.handle)
                    .range(size_of::<FrameUbo>() as u64)];
                let write = vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&buffer_info);
                unsafe { ctx.device.update_descriptor_sets(&[write], &[]) };
                frame_ubos.push(buffer);
                frame_sets.push(set);
            }
        }

        // Material sampler: anisotropic (default 8x, `ZESTENGINE_ANISO` overrides)
        // clamped to the device limit; 1.0 when the feature isn't enabled.
        let aniso = {
            let props = unsafe { ctx.instance.get_physical_device_properties(ctx.physical_device) };
            let has = unsafe { ctx.instance.get_physical_device_features(ctx.physical_device) }.sampler_anisotropy == vk::TRUE;
            let want = std::env::var("ZESTENGINE_ANISO").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(8.0);
            if has { want.clamp(1.0, props.limits.max_sampler_anisotropy.max(1.0)) } else { 1.0 }
        };
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(aniso > 1.0)
            .max_anisotropy(aniso)
            .max_lod(vk::LOD_CLAMP_NONE);
        let sampler = unsafe { ctx.device.create_sampler(&sampler_info, None)? };
        let sampler_cache = {
            let props = unsafe { ctx.instance.get_physical_device_properties(ctx.physical_device) };
            let has = unsafe { ctx.instance.get_physical_device_features(ctx.physical_device) }.sampler_anisotropy == vk::TRUE;
            sampler_cache::SamplerCache::new(sampler, if has { props.limits.max_sampler_anisotropy.max(1.0) } else { 1.0 })
        };
        let ripple_dummy = GpuTexture::cleared(
            &ctx.device,
            &mut allocator.lock().unwrap(),
            command_pool,
            ctx.queue,
            1,
            1,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::empty(),
        )?;
        let dummy_ripples = ripple_infos(sampler, [ripple_dummy.view; MAX_WATER_BODIES]);

        // Lightmaps are a packed UV atlas — clamp (not tile) so bilinear filtering
        // at the atlas border doesn't wrap to the opposite edge (a seam line).
        let lightmap_sampler = unsafe {
            ctx.device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?
        };

        // The opaque colour copy's sampler: trilinear over its mip chain (rough
        // transmission picks a level by roughness), clamped at the border.
        let grab_sampler = unsafe {
            ctx.device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(vk::LOD_CLAMP_NONE),
                None,
            )?
        };

        // Shadow atlas + wire its array view into every frame set (binding 1).
        let tier = RenderTier::for_platform();
        tracing::info!(?tier, "render tier");
        let xr_msaa = {
            let limits = unsafe { ctx.instance.get_physical_device_properties(ctx.physical_device) }.limits;
            let want = vk::SampleCountFlags::from_raw(tier.xr_msaa() as u32);
            let ok = limits.framebuffer_color_sample_counts.contains(want) && limits.framebuffer_depth_sample_counts.contains(want);
            if ok { tier.xr_msaa() } else { 1 }
        };
        let shadow_atlas = ShadowAtlas::new(&ctx, &mut allocator.lock().unwrap(), SHADOW_DIM.min(tier.max_shadow_resolution()), SHADOW_BUDGET_LAYERS)?;
        for &set in &frame_sets {
            let info = [vk::DescriptorImageInfo::default()
                .sampler(shadow_atlas.sampler)
                .image_view(shadow_atlas.array_view)
                .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&info);
            unsafe { ctx.device.update_descriptor_sets(&[write], &[]) };
        }

        // Baked-probe SH buffer (set 0, binding 2): a single zero probe until a
        // bake is loaded, so the binding is always valid. Wire it into every
        // frame set.
        let probe_buffer = {
            let mut alloc = allocator.lock().unwrap();
            let mut buf = Buffer::new(
                &ctx.device,
                &mut alloc,
                size_of::<GpuProbe>() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::CpuToGpu,
                "probe sh (default)",
            )?;
            buf.write(0, bytemuck::bytes_of(&GpuProbe::default()));
            buf
        };
        for &set in &frame_sets {
            let info = [vk::DescriptorBufferInfo::default()
                .buffer(probe_buffer.handle)
                .range(vk::WHOLE_SIZE)];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&info);
            unsafe { ctx.device.update_descriptor_sets(&[write], &[]) };
        }

        // Baked-lightmap array (set 0, binding 3): a 1x1 black single layer
        // until a bake loads, wired into every frame set.
        let lightmaps = GpuTexture::upload_lightmap_array(
            &ctx.device,
            &mut allocator.lock().unwrap(),
            command_pool,
            ctx.queue,
            &HdrTextureArray { format: HdrTextureFormat::Rgba16f, size: 1, layers: vec![vec![0; 8]] },
        )?;
        for &set in &frame_sets {
            let info = [vk::DescriptorImageInfo::default()
                .sampler(lightmap_sampler)
                .image_view(lightmaps.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&info);
            unsafe { ctx.device.update_descriptor_sets(&[write], &[]) };
        }

        // Screen-space GI gather result (set 0, binding 4). Wired into every
        // frame set; re-pointed after a resize (the target is recreated).
        for &set in &frame_sets {
            let info = [vk::DescriptorImageInfo::default()
                .sampler(lightmap_sampler)
                .image_view(sgi.gi_view[0])
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&info);
            unsafe { ctx.device.update_descriptor_sets(&[write], &[]) };
        }

        // SSR inputs (set 0, bindings 5 = scene depth, 6 = last-frame colour).
        // Pointed at valid images so the sets are complete; re-pointed per-pass in
        // render()/record_viewport_scene to the active camera's depth + colour
        // history. The shader gates sampling on frame.ssr.x, so stale content here
        // is never read until a pass wires the real targets.
        for &set in &frame_sets {
            let depth_info = [vk::DescriptorImageInfo::default()
                .sampler(lightmap_sampler)
                .image_view(sgi.depth_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let color_info = [vk::DescriptorImageInfo::default()
                .sampler(lightmap_sampler)
                .image_view(lightmaps.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let env_info = [vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(reflection_env.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let env_b_info = [vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(reflection_env_b.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let env_c_info = [vk::DescriptorImageInfo::default()
                .sampler(sampler).image_view(reflection_env_c.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let env_d_info = [vk::DescriptorImageInfo::default()
                .sampler(sampler).image_view(reflection_env_d.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(5)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&depth_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(6)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&color_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(7)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&env_info),
                // Binding 8 (planar reflections): black lightmap placeholder;
                // re-pointed at the planar RT array on frames that render one.
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(8)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&color_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(9)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&env_b_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set).dst_binding(11)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&env_c_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(set).dst_binding(12)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&env_d_info),
                ripple_write(set, &dummy_ripples),
            ];
            unsafe { ctx.device.update_descriptor_sets(&writes, &[]) };
        }

        // Dedicated set-0 + UBO for the scene reflection capture, bound with the
        // same shadow/probe/lightmap/SGI/env resources as the frame sets (the
        // capture reuses the live lighting). Its UBO is rewritten per cube face.
        let (refl_capture_pool, refl_capture_sets, refl_capture_ubos, planar_sets, planar_ubos, prepass_sets) = {
            // 6 capture sets (one per cube face) + MAX_PLANAR planar-pass sets per
            // frame in flight (a UBO must not be rewritten while the previous
            // frame reads it). (+2 per frame: depth-prepass sets for the main and
            // preview cameras.)
            let total_sets = ((8 + MAX_PLANAR) * FRAMES_IN_FLIGHT) as u32;
            let pool = unsafe {
                ctx.device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default().max_sets(total_sets).pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: total_sets,
                        },
                        vk::DescriptorPoolSize {
                            // bindings 1,3,4,5,6,7,8 per set
                            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: SET0_SAMPLERS * total_sets,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: total_sets,
                        },
                    ]),
                    None,
                )?
            };
            let shadow_info = [vk::DescriptorImageInfo::default()
                .sampler(shadow_atlas.sampler)
                .image_view(shadow_atlas.array_view)
                .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)];
            let probe_info = [vk::DescriptorBufferInfo::default()
                .buffer(probe_buffer.handle)
                .range(vk::WHOLE_SIZE)];
            let lm_info = [vk::DescriptorImageInfo::default()
                .sampler(lightmap_sampler)
                .image_view(lightmaps.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let depth_info = [vk::DescriptorImageInfo::default()
                .sampler(lightmap_sampler)
                .image_view(sgi.depth_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let env_info = [vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(reflection_env.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let env_b_info = [vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(reflection_env_b.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let alloc_set_ubo = |name: &str| -> Result<(vk::DescriptorSet, Buffer)> {
                let layouts = [pipeline_cache.set0_layout];
                let set = unsafe {
                    ctx.device.allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(pool)
                            .set_layouts(&layouts),
                    )?[0]
                };
                let ubo = Buffer::new(
                    &ctx.device,
                    &mut allocator.lock().unwrap(),
                    size_of::<FrameUbo>() as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    MemoryLocation::CpuToGpu,
                    name,
                )?;
                let buffer_info = [vk::DescriptorBufferInfo::default()
                    .buffer(ubo.handle)
                    .range(size_of::<FrameUbo>() as u64)];
                let mk = |binding: u32, ty: vk::DescriptorType| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(binding)
                        .descriptor_type(ty)
                };
                let env_c_info = [vk::DescriptorImageInfo::default()
                    .sampler(sampler).image_view(reflection_env_c.view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                let env_d_info = [vk::DescriptorImageInfo::default()
                    .sampler(sampler).image_view(reflection_env_d.view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                let writes = [
                    mk(0, vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&buffer_info),
                    mk(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&shadow_info),
                    mk(2, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&probe_info),
                    mk(3, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&lm_info),
                    mk(4, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&lm_info),
                    mk(5, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&depth_info),
                    mk(6, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&lm_info),
                    mk(7, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&env_info),
                    mk(9, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&env_b_info),
                    // The shared material shader statically uses all four
                    // reflection slots, even when capture disables sampling.
                    // Initialize every slot before the first offscreen draw;
                    // later cube swaps update these sets via SharedTarget::All.
                    mk(11, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&env_c_info),
                    mk(12, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&env_d_info),
                    // Binding 8 (planar array): valid placeholder; a pass that
                    // reads it (main frame sets) gets the real array re-pointed
                    // per frame. These offscreen sets never sample it.
                    mk(8, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&lm_info),
                    ripple_write(set, &dummy_ripples),
                ];
                unsafe { ctx.device.update_descriptor_sets(&writes, &[]) };
                Ok((set, ubo))
            };
            let mut capture_sets = Vec::with_capacity(6 * FRAMES_IN_FLIGHT);
            let mut capture_ubos = Vec::with_capacity(6 * FRAMES_IN_FLIGHT);
            for i in 0..6 * FRAMES_IN_FLIGHT {
                let (s, u) = alloc_set_ubo(&format!("refl capture ubo {i}"))?;
                capture_sets.push(s);
                capture_ubos.push(u);
            }
            let mut planar_sets = Vec::with_capacity(MAX_PLANAR);
            let mut planar_ubos = Vec::with_capacity(MAX_PLANAR);
            for i in 0..MAX_PLANAR * FRAMES_IN_FLIGHT {
                let (s, u) = alloc_set_ubo(&format!("planar ubo {i}"))?;
                planar_sets.push(s);
                planar_ubos.push(u);
            }
            let prepass_sets = unsafe {
                ctx.device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&[pipeline_cache.set0_layout; 2 * FRAMES_IN_FLIGHT]),
                )?
            };
            (pool, capture_sets, capture_ubos, planar_sets, planar_ubos, prepass_sets)
        };

        // 1x1 defaults, indexed by texture slot (see create_material's slot list):
        // albedo white, flat normal, white ORM, white emission, white opacity,
        // white emission-mask, then 3×(black matcap, white matcap-mask). Black
        // matcaps contribute nothing; white masks pass through.
        let defaults = [
            ([255u8, 255, 255, 255], true),  // 0 albedo
            ([128, 128, 255, 255], false),   // 1 normal
            ([255, 255, 255, 255], false),   // 2 orm
            ([255, 255, 255, 255], true),    // 3 emission
            ([255, 255, 255, 255], false),   // 4 opacity
            ([255, 255, 255, 255], false),   // 5 emission mask
            ([0, 0, 0, 255], true),          // 6 matcap 1
            ([255, 255, 255, 255], false),   // 7 matcap 1 mask
            ([0, 0, 0, 255], true),          // 8 matcap 2
            ([255, 255, 255, 255], false),   // 9 matcap 2 mask
            ([0, 0, 0, 255], true),          // 10 matcap 3
            ([255, 255, 255, 255], false),   // 11 matcap 3 mask
            ([255, 255, 255, 255], false),   // 12 ao (white = no occlusion)
            ([255, 255, 255, 255], false),   // 13 roughness (white, slider scales)
            ([255, 255, 255, 255], false),   // 14 metallic (white, slider scales)
            ([255, 255, 255, 255], false),   // 15 displacement (constant = flat)
            ([0, 0, 0, 255], false),         // 16 custom-shader "black"
            ([128, 128, 128, 255], false),   // 17 custom-shader "gray"
            ([255, 255, 255, 255], true),    // 18 detail albedo (white = neutral)
            ([128, 128, 255, 255], false),   // 19 detail normal (flat)
            ([255, 255, 255, 255], false),   // 20 detail ORM (white = neutral)
            ([255, 255, 255, 255], false),   // 21 detail mask (white = everywhere)
            ([255, 255, 255, 255], false),   // 22 translucency / thickness map
        ];
        let mut default_textures = Vec::new();
        {
            let mut alloc = allocator.lock().unwrap();
            for (pixel, srgb) in defaults {
                default_textures.push(GpuTexture::upload(
                    &ctx.device,
                    &mut alloc,
                    command_pool,
                    ctx.queue,
                    &TextureData {
                        width: 1,
                        height: 1,
                        pixels: pixel.to_vec(),
                        srgb,
                        hdr: false,
                    },
                )?);
            }
        }

        // Screen-space AO (set 0, binding 13): white = no occlusion until a
        // frame computes AO and re-points its frame set (ao.rs).
        {
            let info = [vk::DescriptorImageInfo::default()
                .sampler(lightmap_sampler)
                .image_view(default_textures[ao::AO_DEFAULT_TEXTURE].view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let black = [vk::DescriptorImageInfo::default()
                .sampler(lightmap_sampler).image_view(default_textures[3].view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let writes: Vec<vk::WriteDescriptorSet> = frame_sets
                .iter()
                .chain(&refl_capture_sets)
                .chain(&planar_sets)
                .chain(&prepass_sets)
                .flat_map(|&set| [4, 5, 6, 13].map(|binding| {
                    vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(if binding == 13 { &info } else { &black })
                }))
                .collect();
            unsafe { ctx.device.update_descriptor_sets(&writes, &[]) };
        }

        // Shared zero FX block for set-1 users that aren't materials (skybox).
        let default_fx_ubo = {
            let mut alloc = allocator.lock().unwrap();
            let fx = MaterialFx::from_params(&MaterialParams::default());
            make_fx_ubo(&ctx.device, &mut alloc, fx, "default fx ubo")?
        };

        let area_atlas = AreaAtlas::new(&ctx.device, &mut allocator.lock().unwrap())?;
        let ltc_lut = LtcLut::new(&ctx.device, &mut allocator.lock().unwrap(), command_pool, ctx.queue)?;

        // Skybox descriptor set (set 1 layout): the equirect texture sits in
        // slot 0; default white fills the unused slots. Bound for the
        // fullscreen skybox pass.
        let skybox_set = unsafe {
            let layouts = [pipeline_cache.set1_layout];
            let set = ctx.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(skybox_pool)
                    .set_layouts(&layouts),
            )?[0];
            // Fill all 15 sampler slots with defaults (the skybox shader only
            // reads slot 0, but every binding must be valid). set_skybox writes
            // the real equirect into binding 0 later.
            let infos: Vec<[vk::DescriptorImageInfo; 1]> = (0..MATERIAL_SLOTS)
                .map(|i| {
                    [vk::DescriptorImageInfo::default()
                        .sampler(sampler)
                        .image_view(default_textures[default_texture_index(i)].view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
                })
                .collect();
            let writes: Vec<_> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(TEXTURE_BINDINGS[i])
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(info)
                })
                .collect();
            ctx.device.update_descriptor_sets(&writes, &[]);
            // Binding 4: the shared zero FX block (skybox ignores it).
            let fx_info = [vk::DescriptorBufferInfo::default()
                .buffer(default_fx_ubo.handle)
                .range(vk::WHOLE_SIZE)];
            let fx_write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&fx_info);
            let atlas_info = [vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(area_atlas.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let atlas_write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(AREA_ATLAS_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&atlas_info);
            let ltc_info = [vk::DescriptorImageInfo::default()
                .sampler(ltc_lut.sampler)
                .image_view(ltc_lut.tex.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let ltc_write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(LTC_LUT_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&ltc_info);
            ctx.device.update_descriptor_sets(&[fx_write, atlas_write, ltc_write], &[]);
            set
        };

        let egui = egui_ash_renderer::Renderer::with_gpu_allocator(
            allocator.clone(),
            ctx.device.clone(),
            egui_ash_renderer::DynamicRendering {
                color_attachment_format: swapchain.format.format,
                // egui draws in passes with no depth (the post pass, the editor's
                // swapchain pass), so its pipeline declares none.
                depth_attachment_format: None,
            },
            egui_ash_renderer::Options {
                in_flight_frames: FRAMES_IN_FLIGHT,
                enable_depth_test: false,
                enable_depth_write: false,
                srgb_framebuffer: true,
            },
        )
        .map_err(|e| anyhow::anyhow!("creating egui renderer: {e}"))?;
        // Dedicated VR-UI egui renderer (renders the editor UI into the overlay
        // texture; no depth attachment there). Texture-synced with `egui`.
        let vr_egui = egui_ash_renderer::Renderer::with_gpu_allocator(
            allocator.clone(),
            ctx.device.clone(),
            egui_ash_renderer::DynamicRendering {
                color_attachment_format: swapchain.format.format,
                depth_attachment_format: None,
            },
            egui_ash_renderer::Options {
                in_flight_frames: FRAMES_IN_FLIGHT,
                enable_depth_test: false,
                enable_depth_write: false,
                srgb_framebuffer: true,
            },
        )
        .map_err(|e| anyhow::anyhow!("creating VR egui renderer: {e}"))?;

        // GPU software-GI march pipeline; None falls back to the CPU march.
        let gpu_gi = gpu_gi::GpuGi::new(&ctx)
            .map_err(|e| tracing::warn!("GPU GI init failed, using CPU march: {e:#}"))
            .ok();

        Ok(Self {
            window,
            ctx,
            allocator: Some(allocator),
            swapchain,
            depth,
            sgi,
            ao: ao::AoState::default(),
            aa: aa::AaState::new(AntiAliasing::None),
            post_pass,
            render_scale: 1.0,
            upscaler: Upscaler::Off,
            render_extent: [0; 2],
            jitter: [0.5; 2],
            prev_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            velocity_active: false,
            ssr_resolve,
            ssr_targets,
            velocity_targets,
            reactive_targets,
            fsr3: None,
            fsr3_fg: None,
            frame_gen: false,
            fg_present: Vec::new(),
            fg_output: Vec::new(),
            upscale_outputs: Vec::new(),
            upscale_pool: vk::DescriptorPool::null(),
            upscale_sets: Vec::new(),
            fsr3_reset: true,
            last_time: 0.0,
            underwater,
            selection,
            grid,
            decal,
            viewport_ssr: None,
            rt_reflect,
            xr_render: None,
            xr_eye_auto: 0,
            vr_overlay: None,
            vr_egui: Some(vr_egui),
            last_frame_ubo: None,
            reflection_env,
            reflection_env_b,
            reflection_env_c,
            reflection_env_d,
            refl_capture_pool,
            refl_capture_sets,
            refl_capture_ubos,
            refl_target: None,
            prepass_sets,
            refl_last_finish: None,
            planar_targets: None,
            planar_sets,
            planar_ubos,
            recapture_reflection: None,
            view_cull: ViewCullInfo::default(),
            refl_job: None,
            capturing_refl: false,
            screen_gi_flip_y: true,
            sgi_history_valid: false,
            gi_hit_lighting: [false; 3],
            gi_emitters: Vec::new(),
            flux_settings: FluxSettings::default(),
            sgi_last_view_proj: None,
            cull_lights: Vec::new(),
            cull_active: false,
            sgi_parity: 0,
            sgi_frame: 0,
            sgi_transients: (0..FRAMES_IN_FLIGHT).map(|_| None).collect(),
            sgi_spare: (0..FRAMES_IN_FLIGHT).map(|_| None).collect(),
            preview_sgi_spare: (0..FRAMES_IN_FLIGHT).map(|_| None).collect(),
            preview_sgi_parity: 0,
            preview_sgi_last_view_proj: None,
            preview_sgi_last_cam: glam::Vec3::ZERO,
            preview_sgi_history_valid: false,
            preview_sgi_frame: 0,
            preview_sgi_transients: (0..FRAMES_IN_FLIGHT).map(|_| None).collect(),
            sgi_last_cam: glam::Vec3::ZERO,
            command_pool,
            frames,
            frame_index: 0,
            resize_gate: resize_gate::ResizeGate::new((size.width, size.height)),
            pipeline_cache,
            descriptor_pool,
            material_pool,
            custom_pool,
            skybox_pool,
            default_fx_ubo,
            frame_ubos,
            frame_sets,
            sampler,
            sampler_cache,
            lightmap_sampler,
            ripple_dummy,
            default_textures,
            skybox_set,
            skybox_texture: None,
            skybox_has_texture: false,
            skybox_is_cube: false,
            shadow_atlas,
            shadow_requested_dim: SHADOW_DIM.min(tier.max_shadow_resolution()),
            shadow_layer_sig: [0; MAX_SHADOW_MAPS],
            shadow_layers_valid: false,
            mesh_versions: Vec::new(),
            meshes: Vec::new(),
            textures: Vec::new(),
            materials: Vec::new(),
            freed_meshes: HashSet::new(),
            freed_textures: HashSet::new(),
            freed_materials: HashSet::new(),
            egui: Some(egui),
            egui_free_slots: vec![Vec::new(); FRAMES_IN_FLIGHT],
            skinner: None,
            world_egui: HashMap::new(),
            camera_targets: Vec::new(),
            camera_target_pool: vk::DescriptorPool::null(),
            world_egui_pending_free: Vec::new(),
            text_gpu: None,
            impostor_gpu: None,
            sprite_gpu: None,
            sprite_pool: sprite2d::SpriteMeshPool::new(sprite_gpu::POOL_VERTICES, sprite_gpu::POOL_INDICES),
            sprite_input: SpriteFrameInput::default(),
            text_handle: zestengine_text::TextShared::new(zestengine_text::TextSystem::new(Default::default())),
            text_failed: false,
            world_egui_uploads: Vec::new(),
            world_egui_free_slots: vec![Vec::new(); FRAMES_IN_FLIGHT],
            unlit_shader: None,
            camera_preview: None,
            model_preview: None,
            retired_model_previews: Vec::new(),
            model_only: None,
            model_only_spare: Vec::new(),
            model_only_order: Vec::new(),
            model_only_depth: None,
            capture_jobs: std::collections::VecDeque::new(),
            capture_target: None,
            capture_done: Vec::new(),
            capture_next_id: 0,
            readbacks: Vec::new(),
            preview_keep_depth: false,
            viewport_target: None,
            retired_viewport_targets: Vec::new(),
            retired_viewport_ssr: Vec::new(),
            deferred_frees: Vec::new(),
            pending_shared: Vec::new(),
            shared_seq: 0,
            shared_frees: Vec::new(),
            viewport_resize: Default::default(),
            viewport_resize_hold: false,
            started: std::time::Instant::now(),
            opaque_copy: None,
            area_atlas,
            ltc_lut,
            area_lights: Vec::new(),
            area_sources: Vec::new(),
            packed_area_lights: Vec::new(),
            opaque_mips: false,
            grab_sampler,
            water_copy: false,
            probe_buffer,
            probe_pending: None,
            frame_serial: 0,
            tier,
            xr_msaa,
            rt_scene_cache: None,
            tex_view_cache: Vec::new(),
            materials_seen_scratch: Vec::new(),
            draw_mode: DrawMode::Normal,
            probe_staging: (0..FRAMES_IN_FLIGHT).map(|_| None).collect(),
            tex_pending: Vec::new(),
            tex_staging: (0..FRAMES_IN_FLIGHT).map(|_| None).collect(),
            water_sim: None,
            water_sim_failed: false,
            ripple_mips: HashMap::new(),
            water_sim_queue: Vec::new(),
            probe_count: 1, // default buffer holds one probe
            probe_volumes: Vec::new(),
            frame_probe_volumes: Vec::new(),
            baked_cache: vec![GpuProbe::default()],
            light_volumes: Vec::new(),
            lv_probes: Vec::new(),
            lightmaps,
            last_stats: RenderStats::default(),
            viewport_draw_stats: RenderStats::default(),
            viewport_captures: None,
            swapchain_captures: Vec::new(),
            vsync: true,
            gpu_ts_period,
            gpu_ts_mask: ts_valid_mask(gpu_ts_valid_bits),
            gpu_ts_primed: vec![false; FRAMES_IN_FLIGHT],
            ts_active: false,
            zone_timing: false,
            gpu_zone_ms: [0.0; gpu_zone::COUNT],
            instance_rings: (0..FRAMES_IN_FLIGHT).map(|_| InstanceRing::default()).collect(),
            gpu_cull: None,
            gpu_cull_settings: GpuCullSettings::from_env(),
            gpu_cull_failed: false,
            cluster: None,
            cluster_settings: ClusterSettings::from_env(),
            cluster_failed: false,
            cluster_coverage: ClusterCoverage::default(),
            mesh_refusals: HashMap::new(),
            draw_fallbacks: std::cell::Cell::new([0; REFUSAL_REASONS]),
            instance_ring_sel: None,
            gi_enabled: true,
            profiler_window: None,
            gpu_gi,
            flux_surface: None,
            flux_screen_history: None,
        })
    }

    /// Upload baked light-probe SH (set-0 binding 2) and the per-volume
    /// metadata sampled in the standard shader. Empty `volumes` reverts to the
    /// 1-probe zero buffer so fragments fall back to flat ambient. Call after a
    /// bake or when loading a scene's `.lightdata`.
    pub fn set_baked_probes(
        &mut self,
        probes: &[ProbeSh],
        volumes: &[(Mat4, [f32; 3], [u32; 3], u32)],
    ) {
        // Pack probes into std430 GpuProbe (4 vec4 each); at least one entry so
        // the buffer is never zero-sized.
        let gpu: Vec<GpuProbe> = if probes.is_empty() {
            vec![GpuProbe::default()]
        } else {
            probes.iter().map(gpu_probe).collect()
        };
        self.probe_volumes = volumes
            .iter()
            .map(|(w2l, size, counts, base)| GpuProbeVolume {
                world_to_local: w2l.to_cols_array_2d(),
                size_base: [size[0], size[1], size[2], *base as f32],
                counts: [counts[0] as f32, counts[1] as f32, counts[2] as f32, 0.0],
                params: [0.0; 4],
            })
            .collect();
        self.baked_cache = gpu;
        self.upload_probe_buffer();
    }

    /// Upload the SH storage buffer: the baked / realtime probes, then the Light Volumes'.
    fn upload_probe_buffer(&mut self) {
        self.rebuild_frame_volumes();
        let mut gpu: Vec<GpuProbe> = Vec::with_capacity(self.baked_cache.len() + self.lv_probes.len());
        gpu.extend_from_slice(&self.baked_cache);
        gpu.extend_from_slice(&self.lv_probes);
        let alloc = self.allocator();
        let bytes = std::mem::size_of_val(gpu.as_slice()) as u64;
        let mut buf = match Buffer::new(
            &self.ctx.device,
            &mut alloc.lock().unwrap(),
            bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::CpuToGpu,
            "probe sh",
        ) {
            Ok(b) => b,
            Err(e) => {
                tracing::error!("probe buffer alloc: {e:#}");
                return;
            }
        };
        buf.write(0, bytemuck::cast_slice(&gpu));
        // Re-point binding 2 on every set that renders the scene (frame sets,
        // camera preview, reflection capture): now in the slots that have
        // retired, at their next frame in the others. No device idle.
        let seq = self.queue_shared_writes(
            vec![SharedWrite::buffer(SharedTarget::All, 2, buf.handle)],
            false,
        );
        let old = std::mem::replace(&mut self.probe_buffer, buf);
        self.defer_shared_free(seq, move |device, alloc| {
            let mut old = old;
            old.destroy(device, alloc);
        });
        self.probe_count = self.baked_cache.len();
        self.probe_pending = None;
    }

    /// Cheap per-frame update of the probe SH SSBO: no realloc, no
    /// `device_wait_idle`, no descriptor rewrite. Used to smoothly ease the
    /// realtime-GI probes toward each new trace every frame. Returns false when
    /// the probe count differs from the current buffer (caller must then use
    /// `set_baked_probes` to resize). The data is copied in the next frame's
    /// command buffer, so frames still in flight never read a half-written buffer.
    pub fn update_probe_sh(&mut self, probes: &[ProbeSh]) -> bool {
        if probes.is_empty() || probes.len() != self.probe_count {
            return false;
        }
        let packed: Vec<GpuProbe> = probes.iter().map(gpu_probe).collect();
        self.baked_cache.clone_from(&packed);
        self.probe_pending = Some(packed);
        true
    }

    /// The frame UBO's probe volumes: the baked cascades (at most `MAX_BAKED_VOLUMES`), then the
    /// Light Volumes with their probes placed after the baked ones in the SH buffer.
    fn rebuild_frame_volumes(&mut self) {
        let baked_len = self.baked_cache.len() as f32;
        self.frame_probe_volumes.clear();
        self.frame_probe_volumes.extend(self.probe_volumes.iter().take(MAX_BAKED_VOLUMES).copied());
        let room = MAX_PROBE_VOLUMES - self.frame_probe_volumes.len();
        for v in self.light_volumes.iter().take(room) {
            let mut v = *v;
            v.size_base[3] += baked_len;
            self.frame_probe_volumes.push(v);
        }
    }

    /// Set the scene's Light Volumes (replaces the previous set): baked SH probe grids for
    /// objects without a lightmap, blended by weight with edge fade, independent of the baked
    /// cascades and the realtime GI. At most `MAX_PROBE_VOLUMES - 4` are used. Re-uploads the
    /// probe buffer (no device idle), so call it when the volumes change, not every frame.
    pub fn set_light_volumes(&mut self, volumes: &[LightVolumeData]) {
        self.light_volumes.clear();
        self.lv_probes.clear();
        for v in volumes.iter().take(MAX_PROBE_VOLUMES - MAX_BAKED_VOLUMES) {
            let expected = (v.counts[0] as usize) * (v.counts[1] as usize) * (v.counts[2] as usize);
            if v.blend_distance <= 0.0 || expected == 0 || v.probes.len() != expected {
                tracing::warn!("ignoring a light volume: blend {}, {} probes for a {:?} grid", v.blend_distance, v.probes.len(), v.counts);
                continue;
            }
            self.light_volumes.push(GpuProbeVolume {
                world_to_local: v.world_to_local.to_cols_array_2d(),
                size_base: [v.size[0], v.size[1], v.size[2], self.lv_probes.len() as f32],
                counts: [v.counts[0] as f32, v.counts[1] as f32, v.counts[2] as f32, 0.0],
                params: [v.blend_distance, v.weight.max(0.0), 0.0, 0.0],
            });
            self.lv_probes.extend(v.probes.iter().map(gpu_probe));
        }
        self.upload_probe_buffer();
    }

    /// A water ripple field texture, `n` x `n` RGBA16F (r height, gb its local
    /// XZ gradient), zeroed, with a full mip chain (rebuilt on the GPU after
    /// every write) so the surface mesh and the caustics can read it
    /// band-limited to their own spacing. Bind it as the water material's
    /// slot 15 and feed it with [`update_ripple_texture`].
    pub fn create_ripple_texture(&mut self, n: u32) -> Result<TextureHandle> {
        let n = n.max(1);
        let levels = 32 - n.leading_zeros();
        let allocator = self.allocator();
        // Storage too: the GPU water simulation writes mip 0 directly.
        let texture = GpuTexture::cleared_mips(
            &self.ctx.device,
            &mut allocator.lock().unwrap(),
            self.command_pool,
            self.ctx.queue,
            n,
            n,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            levels,
        )?;
        let base = unsafe {
            self.ctx.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(texture.image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R16G16B16A16_SFLOAT)
                    .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1)),
                None,
            )?
        };
        self.textures.push(texture);
        self.ripple_mips.insert(self.textures.len() - 1, (base, levels));
        Ok(TextureHandle(self.textures.len() - 1))
    }

    /// Queue a ripple field (row-major `n` x `n` heights, `cell` metres apart)
    /// for a [`create_ripple_texture`] texture. It is copied in the next
    /// frame's command buffer, ordered after the frames still reading it.
    pub fn update_ripple_texture(&mut self, handle: TextureHandle, n: usize, heights: &[f32], cell: f32) {
        if heights.len() != n * n || handle.0 >= self.textures.len() {
            return;
        }
        let at = |x: usize, z: usize| heights[z * n + x];
        let inv = 1.0 / (2.0 * cell.max(1e-4));
        let mut bytes = Vec::with_capacity(n * n * 8);
        for z in 0..n {
            for x in 0..n {
                let gx = (at((x + 1).min(n - 1), z) - at(x.saturating_sub(1), z)) * inv;
                let gz = (at(x, (z + 1).min(n - 1)) - at(x, z.saturating_sub(1))) * inv;
                for c in [at(x, z), gx, gz, 1.0] {
                    bytes.extend_from_slice(&half::f16::from_f32(c).to_le_bytes());
                }
            }
        }
        match self.tex_pending.iter_mut().find(|(t, _)| *t == handle.0) {
            Some(p) => p.1 = bytes,
            None => self.tex_pending.push((handle.0, bytes)),
        }
    }

    /// Whether the GPU water simulation can run (it starts on first use).
    pub fn water_sim_supported(&mut self) -> bool {
        if self.water_sim.is_none() && !self.water_sim_failed {
            match water_sim::GpuWaterSim::new(&self.ctx.device) {
                Ok(sim) => self.water_sim = Some(sim),
                Err(e) => {
                    tracing::error!("GPU water simulation unavailable: {e:#}");
                    self.water_sim_failed = true;
                }
            }
        }
        self.water_sim.is_some()
    }

    /// Queue `body` for the next frame's water simulation (replacing what
    /// was queued for the same key).
    pub fn queue_water_sim(&mut self, body: WaterSimBody) {
        match self.water_sim_queue.iter_mut().find(|q| q.key == body.key) {
            Some(q) => *q = body,
            None => self.water_sim_queue.push(body),
        }
    }

    /// The latest finished frame of a body's simulation (a frame or two old).
    pub fn water_sim_readback(&self, key: u64) -> Option<&WaterSimReadback> {
        self.water_sim.as_ref()?.readback(key)
    }

    /// Point this frame slot's set-0 binding 10 (and the camera preview's) at
    /// each water body's ripple field, by the slot its material draws with.
    fn write_frame_ripples(&mut self, input: &FrameInput) {
        let mut views = [self.ripple_dummy.view; MAX_WATER_BODIES];
        for d in input.draws {
            let Some(m) = self.materials.get(d.material.0) else { continue };
            if !m.features.water || m.water_slot as usize >= MAX_WATER_BODIES {
                continue;
            }
            if let Some(t) = m.slots[15].and_then(|t| self.textures.get(t.0)) {
                views[m.water_slot as usize] = t.view;
            }
        }
        let infos = ripple_infos(self.sampler, views);
        let f = self.frame_index;
        let mut writes = vec![ripple_write(self.frame_sets[f], &infos)];
        if let Some(p) = &self.camera_preview {
            writes.push(ripple_write(p.sets[f], &infos));
        }
        unsafe { self.ctx.device.update_descriptor_sets(&writes, &[]) };
    }

    /// Record the queued water simulation (after the slot's fence wait).
    fn record_water_sim(&mut self, cb: vk::CommandBuffer) {
        let queued = std::mem::take(&mut self.water_sim_queue);
        let Some(sim) = self.water_sim.as_mut() else { return };
        let allocator = self.allocator.as_ref().unwrap().clone();
        self.ctx.checkpoint(cb, c"water simulation");
        if let Err(e) = sim.record(&self.ctx.device, &allocator, cb, self.frame_index, &queued, &self.textures, &self.ripple_mips) {
            tracing::error!("water simulation: {e:#}");
        }
    }

    /// Record the queued texture rewrites (see `update_ripple_texture`).
    fn flush_texture_updates(&mut self, cb: vk::CommandBuffer) {
        if self.tex_pending.is_empty() {
            return;
        }
        let pending = std::mem::take(&mut self.tex_pending);
        let total: u64 = pending.iter().map(|(_, b)| b.len() as u64).sum();
        let device = self.ctx.device.clone();
        let slot = self.frame_index;
        if self.tex_staging[slot].as_ref().is_none_or(|(_, size)| *size < total) {
            let alloc = self.allocator();
            let mut alloc = alloc.lock().unwrap();
            if let Some((mut old, _)) = self.tex_staging[slot].take() {
                old.destroy(&device, &mut alloc);
            }
            match Buffer::new(&device, &mut alloc, total, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "texture update staging") {
                Ok(b) => self.tex_staging[slot] = Some((b, total)),
                Err(e) => {
                    tracing::error!("texture update staging alloc: {e:#}");
                    return;
                }
            }
        }
        let staging = &mut self.tex_staging[slot].as_mut().unwrap().0;
        let readers = vk::PipelineStageFlags2::VERTEX_SHADER
            | vk::PipelineStageFlags2::FRAGMENT_SHADER
            | vk::PipelineStageFlags2::COMPUTE_SHADER;
        let mut offset = 0u64;
        for (index, bytes) in &pending {
            let Some(tex) = self.textures.get(*index) else { continue };
            let n = ((bytes.len() / 8) as f64).sqrt() as u32;
            staging.write(offset as usize, bytes);
            image_barrier(
                &device, cb, tex.image, vk::ImageAspectFlags::COLOR,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                readers, vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE,
            );
            let region = vk::BufferImageCopy::default()
                .buffer_offset(offset)
                .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                .image_extent(vk::Extent3D { width: n, height: n, depth: 1 });
            unsafe {
                device.cmd_copy_buffer_to_image(cb, staging.handle, tex.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
            }
            let levels = self.ripple_mips.get(index).map_or(1, |m| m.1);
            texture::record_mip_chain(
                &device,
                cb,
                tex.image,
                n,
                levels,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                readers,
            );
            offset += bytes.len() as u64;
        }
    }

    /// Record the queued probe upload (see `update_probe_sh`) at the start of
    /// this frame's command buffer.
    fn flush_probe_upload(&mut self, cb: vk::CommandBuffer) {
        let Some(gpu) = self.probe_pending.take() else { return };
        let bytes = std::mem::size_of_val(gpu.as_slice()) as u64;
        let device = self.ctx.device.clone();
        let slot = self.frame_index;
        let need_new = self.probe_staging[slot].as_ref().is_none_or(|(_, size)| *size < bytes);
        if need_new {
            let alloc = self.allocator();
            let mut alloc = alloc.lock().unwrap();
            if let Some((mut old, _)) = self.probe_staging[slot].take() {
                old.destroy(&device, &mut alloc);
            }
            match Buffer::new(&device, &mut alloc, bytes, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "probe sh staging") {
                Ok(b) => self.probe_staging[slot] = Some((b, bytes)),
                Err(e) => {
                    tracing::error!("probe staging alloc: {e:#}");
                    return;
                }
            }
        }
        let staging = &mut self.probe_staging[slot].as_mut().unwrap().0;
        staging.write(0, bytemuck::cast_slice(&gpu));
        let barrier = |src_s, src_a, dst_s, dst_a| {
            let b = [vk::MemoryBarrier2::default()
                .src_stage_mask(src_s)
                .src_access_mask(src_a)
                .dst_stage_mask(dst_s)
                .dst_access_mask(dst_a)];
            unsafe { device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&b)) };
        };
        let readers = vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER;
        barrier(readers, vk::AccessFlags2::empty(), vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        unsafe {
            device.cmd_copy_buffer(cb, staging.handle, self.probe_buffer.handle, &[vk::BufferCopy::default().size(bytes)]);
        }
        barrier(vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE, readers, vk::AccessFlags2::SHADER_STORAGE_READ);
    }

    /// Upload raw baked lightmaps (set-0 binding 3) as an RGBA32F 2D array, one
    /// layer per object (resampled to a common size). Empty reverts to a 1x1
    /// black layer. Object→layer indices come from `BakedData.object_lightmap`;
    /// the draw path puts the layer in the push constant. Used for fresh bakes
    /// (and the live bake preview) until the encoded array replaces it.
    pub fn set_baked_lightmaps(&mut self, lightmaps: &[BakedLightmap]) {
        // All layers share the largest baked size (texture-array requirement),
        // so a big surface (e.g. the floor) keeps full resolution. Capped at
        // 4096 (matches the max_lightmap dropdown).
        let size = lightmap_array_size(lightmaps.iter().map(|l| l.size));
        let layers: Vec<Vec<u8>> = if lightmaps.is_empty() {
            vec![vec![0; 16]]
        } else {
            lightmaps
                .iter()
                .map(|l| bytemuck::cast_slice(&resample_rgba32f(&l.pixels, l.size, size)).to_vec())
                .collect()
        };
        self.set_lightmap_array(&HdrTextureArray { format: HdrTextureFormat::Rgba32f, size, layers });
    }

    /// Upload a prepared lightmap array (BC6H from a compressed `.lightmap`, or
    /// an uncompressed fallback). Sampled only, so any [`HdrTextureFormat`]
    /// works; nothing renders or stores into it.
    pub fn set_lightmap_array(&mut self, array: &HdrTextureArray) {
        if array.layers.is_empty() {
            return self.set_baked_lightmaps(&[]);
        }
        let alloc = self.allocator();
        let tex = match GpuTexture::upload_lightmap_array(
            &self.ctx.device,
            &mut alloc.lock().unwrap(),
            self.command_pool,
            self.upload_queue(),
            array,
        ) {
            Ok(t) => t,
            Err(e) => {
                tracing::error!("lightmap array upload: {e:#}");
                return;
            }
        };
        // The lightmap and unused planar-array bindings share this array.
        // 2D placeholders have renderer-lifetime textures and need no rebinding.
        let writes = [3, 8].map(|binding|
            SharedWrite::sampled(SharedTarget::All, binding, self.lightmap_sampler, tex.view)).to_vec();
        let seq = self.queue_shared_writes(writes, false);
        let old = std::mem::replace(&mut self.lightmaps, tex);
        self.defer_shared_free(seq, move |device, alloc| {
            let mut old = old;
            old.destroy(device, alloc);
        });
    }

    /// Toggle vsync (FIFO ↔ MAILBOX/IMMEDIATE); takes effect next frame via
    /// swapchain recreation.
    pub fn set_vsync(&mut self, vsync: bool) {
        if self.vsync != vsync {
            self.vsync = vsync;
            self.resize_gate.force();
        }
    }

    /// Tell the renderer a pointer button is down, so the editor viewport
    /// target is not recreated for every size of a panel-border drag.
    pub fn set_viewport_resize_hold(&mut self, held: bool) {
        self.viewport_resize_hold = held;
    }

    pub fn vsync(&self) -> bool {
        self.vsync
    }

    /// Present mode of the main swapchain (the frame limiter's policy input).
    pub fn main_present_mode(&self) -> ash::vk::PresentModeKHR {
        self.swapchain.present_mode
    }

    /// Set the resolution scale (fraction of the output the scene renders at,
    /// clamped 0.25..=1.0). The upscaler reconstructs the full frame; at 1.0
    /// the scene renders native.
    pub fn set_render_scale(&mut self, scale: f32) {
        let scale = scale.clamp(0.25, 1.0);
        if (self.render_scale - scale).abs() > 1e-3 {
            self.render_scale = scale;
            // Recreate the reduced-resolution targets at the new scale (handled
            // at the top of the next frame, like a window resize).
            self.resize_gate.force();
            tracing::debug!(scale, "render scale");
        }
    }

    pub fn render_scale(&self) -> f32 {
        self.render_scale
    }

    /// The reduced-resolution extent the scene rendered at last frame
    /// (`render_scale` of the output size; equals the output at native).
    pub fn render_extent(&self) -> [u32; 2] {
        self.render_extent
    }

    /// The sub-pixel jitter for the current frame (Halton(2, 3), in [0, 1)²).
    pub fn jitter(&self) -> [f32; 2] {
        self.jitter
    }

    /// Set the active upscaler (Off = native, Fsr3 = temporal upscaling).
    pub fn set_upscaler(&mut self, upscaler: Upscaler) {
        if self.upscaler != upscaler {
            self.upscaler = upscaler;
            // Recreate the upscaler context + full-res output (or tear them
            // down) at the top of the next frame, like a window resize.
            self.resize_gate.force();
            self.fsr3_reset = true;
            tracing::debug!(?upscaler, "upscaler");
        }
    }

    pub fn upscaler(&self) -> Upscaler {
        self.upscaler
    }

    /// Set the upscaler quality preset in one step: `Native` turns the
    /// upscaler off; the rest select FSR3 and its render scale.
    pub fn set_upscale_quality(&mut self, quality: UpscaleQuality) {
        self.set_render_scale(quality.render_scale());
        self.set_upscaler(if quality == UpscaleQuality::Native { Upscaler::Off } else { Upscaler::Fsr3 });
    }

    /// The quality preset matching the current render scale + upscaler.
    pub fn upscale_quality(&self) -> UpscaleQuality {
        if self.upscaler == Upscaler::Off {
            return UpscaleQuality::Native;
        }
        UpscaleQuality::from_render_scale(self.render_scale)
    }

    /// Frame generation is on only when the upscaler is FSR3 and `frame_gen`.
    pub fn frame_gen_active(&self) -> bool {
        self.upscaler == Upscaler::Fsr3 && self.frame_gen
    }

    /// Toggle FSR3 frame generation (gated behind the upscaler).
    pub fn set_frame_gen(&mut self, on: bool) {
        if self.frame_gen != on {
            self.frame_gen = on;
            self.resize_gate.force();
            self.fsr3_reset = true;
            tracing::debug!(on, "frame gen");
        }
    }

    pub fn frame_gen(&self) -> bool {
        self.frame_gen
    }

    /// Statistics of the last rendered frame.
    pub fn stats(&self) -> RenderStats {
        self.last_stats
    }

    /// Append per-instance records to the active instance ring.
    fn push_instances(&mut self, data: &[InstanceData]) -> Result<(vk::Buffer, u64)> {
        let allocator = self.allocator();
        let mut a = allocator.lock().unwrap();
        let device = &self.ctx.device;
        let ring = match self.instance_ring_sel {
            Some(i) => &mut self.xr_render.as_mut().expect("xr render").slots[i].instances,
            None => &mut self.instance_rings[self.frame_index],
        };
        ring.push(device, &mut a, data)
    }

    fn allocator(&self) -> Arc<Mutex<Allocator>> {
        self.allocator.as_ref().unwrap().clone()
    }

    /// The text system shared with UI code: lay text out through
    /// `handle.system`, queue draws with `handle.push`. Register at least one
    /// built-in font on it before the first frame that draws text.
    pub fn text_handle(&self) -> zestengine_text::TextHandle {
        self.text_handle.clone()
    }

    /// Whether the GPU text path is usable (false once creating it failed;
    /// UI code then falls back to egui text).
    pub fn text_available(&self) -> bool {
        !self.text_failed
    }

    fn ensure_sprite_gpu(&mut self) -> Result<()> {
        if self.sprite_gpu.is_some() {
            return Ok(());
        }
        let allocator = self.allocator();
        let mut a = allocator.lock().unwrap();
        self.sprite_gpu = Some(sprite_gpu::SpriteGpu::new(&self.ctx.device, &mut a, self.pipeline_cache.set0_layout)?);
        Ok(())
    }

    /// The pooled sprite mesh buffer's CPU side: add each sprite's cooked mesh
    /// once, keep the returned `MeshRange` in the sprite's `SpriteSource`. New
    /// meshes reach the GPU with the next frame.
    pub fn sprite_mesh_pool(&mut self) -> &mut sprite2d::SpriteMeshPool {
        &mut self.sprite_pool
    }

    /// The sprites and masks to draw this frame. Clear and refill it every frame
    /// before `render`; the frame consumes it.
    pub fn sprite_input_mut(&mut self) -> &mut SpriteFrameInput {
        &mut self.sprite_input
    }

    /// Make an existing texture drawable by sprites; returns the id for
    /// `SpriteSource::texture` / `MaskSource::texture`. `array` = the texture's
    /// view is a 2D-array view (an atlas whose pages are layers). Call
    /// `unregister_sprite_texture` before freeing the texture.
    pub fn register_sprite_texture(&mut self, texture: TextureHandle, array: bool) -> Result<u32> {
        self.ensure_sprite_gpu()?;
        let Some(t) = self.textures.get(texture.0) else {
            anyhow::bail!("register_sprite_texture: no such texture");
        };
        let view = t.view;
        let sampler = self.sampler_cache.for_texture(&self.ctx.device, texture.0);
        self.sprite_gpu.as_mut().unwrap().register_texture(&self.ctx.device, view, sampler, array)
    }

    /// Drop a sprite texture id (waits for the GPU, so no in-flight frame uses it).
    pub fn unregister_sprite_texture(&mut self, id: u32) {
        if self.sprite_gpu.is_none() {
            return;
        }
        unsafe {
            let _ = self.ctx.device.device_wait_idle();
        }
        self.sprite_gpu.as_mut().unwrap().unregister_texture(&self.ctx.device, id);
    }

    /// Register a sprite material (blend mode, alpha cutoff); id 0 is the default.
    pub fn register_sprite_material(&mut self, desc: SpriteMaterialDesc) -> Result<u32> {
        self.ensure_sprite_gpu()?;
        Ok(self.sprite_gpu.as_mut().unwrap().register_material(desc))
    }

    /// Counters of the last built sprite frame.
    pub fn sprite_stats(&self) -> sprite2d::SpriteStats {
        self.sprite_gpu.as_ref().map(|g| g.stats()).unwrap_or_default()
    }

    /// Create the text atlas images and pipelines once a font exists.
    fn ensure_text_gpu(&mut self) {
        if self.text_gpu.is_some() || self.text_failed {
            return;
        }
        if !self.text_handle.system.lock().map(|s| s.has_fonts()).unwrap_or(false) {
            return;
        }
        let allocator = self.allocator();
        let mut a = allocator.lock().unwrap();
        match text::TextGpu::new(
            &self.ctx.device,
            &mut a,
            self.command_pool,
            self.ctx.queue,
            self.text_handle.clone(),
            self.pipeline_cache.set0_layout,
        ) {
            Ok(t) => self.text_gpu = Some(t),
            Err(e) => {
                tracing::error!("text renderer unavailable: {e:#}");
                self.text_failed = true;
            }
        }
    }

    /// Upload a captured octahedral impostor (the fallback for avatars the
    /// safety policy hides or that are far away). Needs a perspective scene pass.
    pub fn upload_impostor(&mut self, data: &ImpostorUpload) -> Result<ImpostorHandle> {
        let allocator = self.allocator();
        let mut alloc = allocator.lock().unwrap();
        if self.impostor_gpu.is_none() {
            self.impostor_gpu = Some(impostor::ImpostorGpu::new(&self.ctx.device, &mut alloc, self.pipeline_cache.set0_layout)?);
        }
        self.impostor_gpu.as_mut().unwrap().upload(&self.ctx.device, &mut alloc, self.command_pool, self.ctx.queue, data)
    }

    /// Free an impostor atlas. The atlas stops being drawn now; its image and
    /// descriptor set are destroyed once the frames in flight have retired.
    pub fn free_impostor(&mut self, handle: ImpostorHandle) {
        let Some(gpu) = self.impostor_gpu.as_mut() else { return };
        let pool = gpu.pool();
        if let Some(mut a) = gpu.take(handle) {
            self.defer_free(move |device, alloc| unsafe { impostor::free_atlas(device, alloc, pool, &mut a) });
        }
    }

    /// Impostor atlases currently alive.
    pub fn impostor_count(&self) -> usize {
        self.impostor_gpu.as_ref().map_or(0, |g| g.live())
    }

    /// Draw a camera-facing impostor billboard this frame (consumed by the next
    /// `render`): `transform` maps the model space the impostor was captured in
    /// (the avatar's root frame) to the world; `tint` = rgb + mix factor towards
    /// that colour (0 = the captured colours). Not drawn into shadow maps, the
    /// depth prepass or reflection captures. A rest-pose impostor looks wrong in motion.
    pub fn draw_impostor(&mut self, handle: ImpostorHandle, transform: Mat4, tint: [f32; 4]) {
        if let Some(i) = self.impostor_gpu.as_mut() {
            i.queue(handle, transform, tint);
        }
    }

    /// Impostor billboards of the scene pass that is currently recording.
    fn record_impostors(&mut self, device: &ash::Device, cb: vk::CommandBuffer, frame_set: vk::DescriptorSet, gbuf: bool) {
        if self.draw_mode == DrawMode::Prepass || self.capturing_refl {
            return;
        }
        let pass = self.pipeline_cache.pass_formats();
        let velocity = self.velocity_active;
        if let Some(i) = self.impostor_gpu.as_mut() {
            i.record(device, cb, self.frame_index, frame_set, pass, gbuf, velocity);
        }
    }

    pub fn upload_mesh(&mut self, data: &MeshData) -> Result<MeshHandle> {
        // When ray tracing is available the bake reads these buffers as
        // acceleration-structure inputs and as SSBOs (via device address),
        // so they need the extra usage flags + device-address storage.
        let rt = if self.ctx.ray_tracing() {
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::STORAGE_BUFFER
        } else {
            vk::BufferUsageFlags::empty()
        };
        let allocator = self.allocator();
        let mut alloc = allocator.lock().unwrap();
        let vertex_buffer = alloc::upload_buffer(
            &self.ctx.device,
            &mut alloc,
            self.command_pool,
            self.ctx.queue,
            bytemuck::cast_slice(&data.vertices),
            vk::BufferUsageFlags::VERTEX_BUFFER | rt,
            "vertices",
        )?;
        let (index_data, index_type) = index_bytes(data, self.ctx.ray_tracing());
        let index_buffer = alloc::upload_buffer(
            &self.ctx.device,
            &mut alloc,
            self.command_pool,
            self.ctx.queue,
            &index_data,
            vk::BufferUsageFlags::INDEX_BUFFER | rt,
            "indices",
        )?;
        self.meshes.push(GpuMesh {
            vertex_buffer,
            index_buffer,
            index_type,
            index_count: data.indices.len() as u32,
            vertex_count: data.vertices.len() as u32,
            bounds: flux_surface::vertex_bounds(&data.vertices),
            surface_geometry: flux_cards::MeshCards::new(data, false),
        });
        Ok(MeshHandle(self.meshes.len() - 1))
    }

    /// Upload a mesh whose vertices are updated every frame (CPU skinning). The
    /// vertex buffer is host-visible so [`update_mesh_vertices`] can rewrite it
    /// cheaply; the index buffer is static/device-local. (No double-buffering
    /// yet, so a fast-moving skinned mesh can tear by a frame — acceptable until
    /// GPU skinning lands.)
    pub fn upload_mesh_skinned(&mut self, data: &MeshData) -> Result<MeshHandle> {
        // Ray tracing builds BLASes from these too (and reads them by address).
        let rt = if self.ctx.ray_tracing() {
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::STORAGE_BUFFER
        } else {
            vk::BufferUsageFlags::empty()
        };
        let allocator = self.allocator();
        let mut alloc = allocator.lock().unwrap();
        let mut vertex_buffer = Buffer::new(
            &self.ctx.device,
            &mut alloc,
            std::mem::size_of_val(&data.vertices[..]) as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER | rt,
            MemoryLocation::CpuToGpu,
            "skinned vertices",
        )?;
        vertex_buffer.write(0, bytemuck::cast_slice(&data.vertices));
        let index_buffer = alloc::upload_buffer(
            &self.ctx.device,
            &mut alloc,
            self.command_pool,
            self.ctx.queue,
            bytemuck::cast_slice(&data.indices),
            vk::BufferUsageFlags::INDEX_BUFFER | rt,
            "skinned indices",
        )?;
        self.meshes.push(GpuMesh {
            vertex_buffer,
            index_buffer,
            index_type: vk::IndexType::UINT32,
            index_count: data.indices.len() as u32,
            vertex_count: data.vertices.len() as u32,
            bounds: flux_surface::vertex_bounds(&data.vertices),
            surface_geometry: flux_cards::MeshCards::new(data, true),
        });
        Ok(MeshHandle(self.meshes.len() - 1))
    }

    /// Rewrite a skinned mesh's vertices (host-visible buffer from
    /// [`upload_mesh_skinned`]). Call with the CPU-skinned vertices each frame.
    pub fn update_mesh_vertices(&mut self, handle: MeshHandle, vertices: &[Vertex]) {
        self.bump_mesh_version(handle);
        if let Some(mesh) = self.meshes.get_mut(handle.0) {
            mesh.vertex_buffer.write(0, bytemuck::cast_slice(vertices));
            mesh.bounds = flux_surface::vertex_bounds(vertices);
            mesh.surface_geometry.update(vertices);
        }
    }

    /// Upload a mesh skinned on the GPU: its vertex buffer is rewritten each
    /// frame by the skinning compute pass from a bind-pose copy, using the
    /// palette given to [`skin_mesh_gpu`]. `joint_count` sizes the palette.
    pub fn upload_mesh_gpu_skinned(&mut self, data: &MeshData, joint_count: usize) -> Result<MeshHandle> {
        self.upload_mesh_gpu_skinned_morphed(data, joint_count, &[])
    }

    /// [`upload_mesh_gpu_skinned`](Self::upload_mesh_gpu_skinned) with
    /// blendshapes: `shapes` holds one sparse delta list per shape (weight
    /// order). A compute pass applies them before skinning, driven by
    /// [`set_mesh_morph_weights`](Self::set_mesh_morph_weights), so morphed
    /// meshes stay on the GPU path.
    pub fn upload_mesh_gpu_skinned_morphed(
        &mut self,
        data: &MeshData,
        joint_count: usize,
        shapes: &[MorphDeltas<'_>],
    ) -> Result<MeshHandle> {
        let morph = (!shapes.is_empty()).then(|| MorphTables::build(&data.vertices, shapes)).filter(|m| !m.is_empty());
        if self.skinner.is_none() {
            self.skinner = Some(skin::GpuSkinner::new(&self.ctx.device)?);
        }
        let allocator = self.allocator();
        let mut alloc = allocator.lock().unwrap();
        let bytes: &[u8] = bytemuck::cast_slice(&data.vertices);
        let vertex_buffer = alloc::upload_buffer(
            &self.ctx.device,
            &mut alloc,
            self.command_pool,
            self.ctx.queue,
            bytes,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
            "gpu skinned vertices",
        )?;
        let base = alloc::upload_buffer(
            &self.ctx.device,
            &mut alloc,
            self.command_pool,
            self.ctx.queue,
            bytes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            "skin bind pose",
        )?;
        let index_buffer = alloc::upload_buffer(
            &self.ctx.device,
            &mut alloc,
            self.command_pool,
            self.ctx.queue,
            bytemuck::cast_slice(&data.indices),
            vk::BufferUsageFlags::INDEX_BUFFER,
            "gpu skinned indices",
        )?;
        let out = vertex_buffer.handle;
        let mesh = self.meshes.len();
        self.skinner.as_mut().unwrap().add(
            &self.ctx.device,
            &mut alloc,
            mesh,
            out,
            base,
            data.vertices.len() as u32,
            joint_count,
            morph,
        )?;
        self.meshes.push(GpuMesh {
            vertex_buffer,
            index_buffer,
            index_type: vk::IndexType::UINT32,
            index_count: data.indices.len() as u32,
            vertex_count: data.vertices.len() as u32,
            bounds: flux_surface::vertex_bounds(&data.vertices),
            surface_geometry: flux_cards::MeshCards::new(data, false),
        });
        Ok(MeshHandle(mesh))
    }

    /// Set this frame's blendshape weights for a mesh uploaded with
    /// [`upload_mesh_gpu_skinned_morphed`](Self::upload_mesh_gpu_skinned_morphed).
    /// Unchanged weights cost nothing; changed ones run the morph pass.
    pub fn set_mesh_morph_weights(&mut self, handle: MeshHandle, weights: &[f32]) {
        if let Some(skinner) = self.skinner.as_mut() {
            skinner.queue_morph(handle.0, weights);
        }
    }

    /// Set this frame's Mesh Cloth overrides for a GPU-skinned mesh: `packed` is
    /// [`pack_cloth_patch`]'s output (`count` entries). Those vertices bypass
    /// skinning; the rest of the mesh keeps skinning on the GPU. Call every
    /// frame the cloth runs (`count` 0 = plain skinning).
    pub fn set_mesh_cloth_patch(&mut self, handle: MeshHandle, packed: &[f32], count: u32) {
        self.bump_mesh_version(handle);
        let allocator = self.allocator();
        if let Some(skinner) = self.skinner.as_mut()
            && let Err(e) = skinner.queue_cloth(&self.ctx.device, &mut allocator.lock().unwrap(), handle.0, packed, count)
        {
            tracing::error!("cloth patch buffers: {e:#}");
        }
    }

    /// Set this frame's joint palette for a GPU-skinned mesh.
    /// Mark a mesh's vertices as changed (shadow caching keys on this).
    fn bump_mesh_version(&mut self, handle: MeshHandle) {
        if self.mesh_versions.len() <= handle.0 {
            self.mesh_versions.resize(handle.0 + 1, 0);
        }
        self.mesh_versions[handle.0] = self.mesh_versions[handle.0].wrapping_add(1);
    }

    pub fn skin_mesh_gpu(&mut self, handle: MeshHandle, palette: &[glam::Mat4]) {
        self.bump_mesh_version(handle);
        if let Some(skinner) = self.skinner.as_mut() {
            skinner.queue(handle.0, palette);
        }
    }

    /// Give a mesh the skinned cluster path draws a pose without a GPU skin of
    /// its own: a CPU-skinned shared mesh (its vertex buffer is written with
    /// [`update_mesh_vertices`](Self::update_mesh_vertices)) whose cluster DAG
    /// is skinned from `palette`. Call when the palette changes (and once more,
    /// so the previous-frame pose catches up).
    pub fn set_mesh_pose(&mut self, handle: MeshHandle, palette: &[glam::Mat4]) {
        if self.skinner.is_none() {
            match skin::GpuSkinner::new(&self.ctx.device) {
                Ok(s) => self.skinner = Some(s),
                Err(e) => {
                    tracing::warn!("GPU skinner: {e:#}");
                    return;
                }
            }
        }
        if let Some(skinner) = self.skinner.as_mut() {
            skinner.set_pose(handle.0, palette);
        }
    }

    /// What the last rendered frame's views were (see [`ViewCullInfo`]).
    pub fn view_cull_info(&self) -> &ViewCullInfo {
        &self.view_cull
    }

    /// The built-in unlit shader (albedo * vertex color, no lighting), for
    /// world-space UI and particles. Pair with `set_material_shader`.
    pub fn unlit_shader(&mut self) -> Result<ShaderId> {
        if let Some(id) = self.unlit_shader {
            return Ok(id);
        }
        const UNLIT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/unlit.frag.spv"));
        let id = self.register_shader(UNLIT_SPV)?;
        self.unlit_shader = Some(id);
        Ok(id)
    }

    /// The texture a render-to-texture camera draws into (created on first
    /// use, `width` x `height`, linear HDR). Materials sample it like any
    /// texture; pass the camera in `FrameInput::camera_targets` each frame it
    /// should update. Cleared by `reset_scene`.
    pub fn camera_target(&mut self, id: u128, width: u32, height: u32) -> Result<TextureHandle> {
        if let Some(t) = self.camera_targets.iter().find(|t| t.id == id) {
            let texture = t.texture;
            self.resize_camera_target(id, width, height)?;
            return Ok(texture);
        }
        if self.camera_targets.len() >= MAX_CAMERA_TARGETS {
            anyhow::bail!("at most {MAX_CAMERA_TARGETS} cameras can render into textures");
        }
        let extent = vk::Extent2D { width: width.clamp(16, 4096), height: height.clamp(16, 4096) };
        let device = self.ctx.device.clone();
        if self.camera_target_pool == vk::DescriptorPool::null() {
            let n = (MAX_CAMERA_TARGETS * FRAMES_IN_FLIGHT) as u32;
            self.camera_target_pool = unsafe {
                device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default().max_sets(n).pool_sizes(&[
                        vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: n },
                        vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: n },
                        vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: SET0_SAMPLERS * n },
                    ]),
                    None,
                )?
            };
        }
        let allocator = self.allocator();
        let mut a = allocator.lock().unwrap();
        let color = GpuTexture::render_target(
            &device,
            &mut a,
            self.command_pool,
            self.ctx.queue,
            extent.width,
            extent.height,
            post::HDR_FORMAT,
        )?;
        let depth = DepthTarget::new(&self.ctx, &mut a, extent)?;
        let layouts = vec![self.pipeline_cache.set0_layout; FRAMES_IN_FLIGHT];
        let sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.camera_target_pool).set_layouts(&layouts),
            )?
        };
        let mut ubos = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for &set in &sets {
            let ubo = Buffer::new(
                &device,
                &mut a,
                size_of::<FrameUbo>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
                "camera target ubo",
            )?;
            let info = [vk::DescriptorBufferInfo::default().buffer(ubo.handle).range(size_of::<FrameUbo>() as u64)];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&info);
            unsafe { device.update_descriptor_sets(&[write], &[]) };
            ubos.push(ubo);
        }
        drop(a);
        self.textures.push(color);
        let texture = TextureHandle(self.textures.len() - 1);
        self.camera_targets.push(CameraTarget { id, texture, extent, depth, sets, ubos });
        Ok(texture)
    }

    /// Resize an existing camera target (no-op when the size is unchanged or the
    /// camera has none). Same texture handle: the image is replaced and every
    /// material sampling it is re-pointed.
    pub fn resize_camera_target(&mut self, id: u128, width: u32, height: u32) -> Result<()> {
        let extent = vk::Extent2D { width: width.clamp(16, 4096), height: height.clamp(16, 4096) };
        let Some(k) = self.camera_targets.iter().position(|t| t.id == id) else {
            return Ok(());
        };
        if self.camera_targets[k].extent == extent {
            return Ok(());
        }
        let device = self.ctx.device.clone();
        let handle = self.camera_targets[k].texture;
        {
            let allocator = self.allocator();
            let mut a = allocator.lock().unwrap();
            let color = GpuTexture::render_target(
                &device,
                &mut a,
                self.command_pool,
                self.ctx.queue,
                extent.width,
                extent.height,
                post::HDR_FORMAT,
            )?;
            let depth = DepthTarget::new(&self.ctx, &mut a, extent)?;
            // Frames in flight still render into / sample the old pair: parked, not
            // destroyed (the materials sampling it are rebound below, also deferred).
            let mut old_color = std::mem::replace(&mut self.textures[handle.0], color);
            let t = &mut self.camera_targets[k];
            let mut old_depth = std::mem::replace(&mut t.depth, depth);
            t.extent = extent;
            self.deferred_frees.push((
                FRAMES_IN_FLIGHT + 1,
                Box::new(move |device, alloc| {
                    old_color.destroy(device, alloc);
                    old_depth.destroy(device, alloc);
                }),
            ));
        }
        let users: Vec<(usize, [Option<TextureHandle>; MATERIAL_SLOTS])> = self
            .materials
            .iter()
            .enumerate()
            .filter(|(_, m)| m.slots.contains(&Some(handle)))
            .map(|(i, m)| (i, m.slots))
            .collect();
        for (i, slots) in users {
            self.bind_material_slots(MaterialHandle(i), &slots);
        }
        Ok(())
    }

    /// Render every camera in `input.camera_targets` into its texture, before
    /// the passes that sample them. Materials sampling any camera target are
    /// left out of these passes, so a screen never draws itself.
    fn camera_target_render_order(ids: &[u128], deps: &HashMap<u128, Vec<u128>>) -> Vec<u128> {
        let mut out = Vec::with_capacity(ids.len());
        let mut done: HashSet<u128> = HashSet::with_capacity(ids.len());
        let mut pending = ids.to_vec();
        while !pending.is_empty() {
            let mut progressed = false;
            let mut i = 0;
            while i < pending.len() {
                let id = pending[i];
                let ready = deps
                    .get(&id)
                    .is_none_or(|need| need.iter().all(|d| done.contains(d)));
                if ready {
                    out.push(id);
                    done.insert(id);
                    pending.remove(i);
                    progressed = true;
                } else {
                    i += 1;
                }
            }
            if !progressed {
                // Dependency cycle: keep deterministic order and accept one-frame
                // latency inside the loop.
                out.extend_from_slice(&pending);
                break;
            }
        }
        out
    }

    /// Render every camera in `input.camera_targets` into its texture, before
    /// the passes that sample them. Materials sampling a camera's OWN target are
    /// left out of that camera pass to avoid feedback; other screens remain so
    /// nested portals/screens can render.
    #[allow(clippy::too_many_arguments)]
    fn record_camera_targets(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        input: &FrameInput,
        lights: [GpuLight; MAX_LIGHTS],
        count: usize,
        shadow_vp: [[[f32; 4]; 4]; MAX_SHADOW_VIEWS],
        cascade_splits: [f32; 4],
        order: &[usize],
    ) -> Result<()> {
        let target_textures: Vec<usize> = self.camera_targets.iter().map(|t| t.texture.0).collect();
        let target_by_texture: HashMap<usize, u128> =
            self.camera_targets.iter().map(|t| (t.texture.0, t.id)).collect();
        let mut target_index: HashMap<u128, usize> = HashMap::new();
        for (k, t) in self.camera_targets.iter().enumerate() {
            target_index.insert(t.id, k);
        }
        let screen_material: Vec<bool> = self
            .materials
            .iter()
            .map(|m| m.slots.iter().flatten().any(|h| target_textures.contains(&h.0)))
            .collect();
        // Seed with targets shown on a screen the main camera can see.
        let main_planes = frustum_planes(input.camera.proj * input.camera.view);
        let mut seed_textures: Vec<usize> = Vec::new();
        for &d in order {
            let draw = &input.draws[d];
            if screen_material.get(draw.material.0).copied().unwrap_or(false) && sphere_in_planes(&main_planes, draw, true) {
                seed_textures.extend(self.materials[draw.material.0].slots.iter().flatten().map(|h| h.0));
            }
        }

        // Follow camera-target dependencies from visible screens so nested
        // portals/screens get rendered when they are reachable through another
        // screen, not just directly visible in the main view.
        let view_by_id: HashMap<u128, &CameraTargetView> =
            input.camera_targets.iter().map(|v| (v.id, v)).collect();
        let mut wanted: HashSet<u128> = HashSet::new();
        let mut queue: VecDeque<usize> = seed_textures.into_iter().collect();
        while let Some(tex) = queue.pop_front() {
            let Some(&cam_id) = target_by_texture.get(&tex) else {
                continue;
            };
            if !wanted.insert(cam_id) {
                continue;
            }
            let Some(view) = view_by_id.get(&cam_id).copied() else {
                continue;
            };
            let planes = frustum_planes(view.camera.proj * view.camera.view);
            for &d in order {
                let draw = &input.draws[d];
                if !screen_material.get(draw.material.0).copied().unwrap_or(false)
                    || !sphere_in_planes(&planes, draw, true)
                {
                    continue;
                }
                let mat = &self.materials[draw.material.0];
                for h in mat.slots.iter().flatten() {
                    if h.0 != tex && target_by_texture.contains_key(&h.0) {
                        queue.push_back(h.0);
                    }
                }
            }
        }

        let wanted_ids: Vec<u128> = input
            .camera_targets
            .iter()
            .map(|v| v.id)
            .filter(|id| wanted.contains(id))
            .collect();
        if wanted_ids.is_empty() {
            return Ok(());
        }

        // Build a dependency graph so cameras that feed another screen render first.
        let mut deps: HashMap<u128, Vec<u128>> = HashMap::new();
        for &id in &wanted_ids {
            let Some(&idx) = target_index.get(&id) else {
                continue;
            };
            let self_tex = self.camera_targets[idx].texture.0;
            let Some(view) = view_by_id.get(&id).copied() else {
                continue;
            };
            let planes = frustum_planes(view.camera.proj * view.camera.view);
            let mut need: HashSet<u128> = HashSet::new();
            for &d in order {
                let draw = &input.draws[d];
                if !screen_material.get(draw.material.0).copied().unwrap_or(false)
                    || !sphere_in_planes(&planes, draw, true)
                {
                    continue;
                }
                let mat = &self.materials[draw.material.0];
                for h in mat.slots.iter().flatten() {
                    if h.0 == self_tex {
                        continue;
                    }
                    // Only cameras actually rendered this frame can order this
                    // pass; a throttled/absent camera's screen keeps showing its
                    // last frame and must not poison the topological sort.
                    if let Some(dep) = target_by_texture.get(&h.0)
                        && wanted_ids.contains(dep)
                        && *dep != id
                    {
                        need.insert(*dep);
                    }
                }
            }
            deps.insert(id, need.into_iter().collect());
        }
        let render_order = Self::camera_target_render_order(&wanted_ids, &deps);

        for id in render_order {
            let Some(view) = view_by_id.get(&id).copied() else {
                continue;
            };
            let Some(&tidx) = target_index.get(&id) else {
                continue;
            };
            let (ext, set, self_tex, depth_img, depth_view) = {
                let t = &self.camera_targets[tidx];
                (
                    t.extent,
                    t.sets[self.frame_index],
                    t.texture.0,
                    t.depth.image,
                    t.depth.att_view,
                )
            };
            let image = self.textures[self_tex].image;
            let color_view = self.textures[self_tex].view;
            let mut ubo = frame_ubo(
                &view.camera, input, lights, count, shadow_vp, cascade_splits,
                &self.frame_probe_volumes, [0.5; 2], [1.0, 1.0], [ext.width as f32, ext.height as f32],
                glam::Mat4::IDENTITY.to_cols_array_2d(),
            );
            ubo.misc[2] *= self.shadow_requested_dim as f32 / self.shadow_atlas.dim as f32;
            ubo.debug[1] = 0.0;
            ubo.debug[2] = 0.0; // no screen-GI from the main camera's buffers
            ubo.debug[3] = 1.0; // linear HDR, sampled as radiance
            ubo.ssr = [0.0; 4];
            if let Some(p) = input.reflection_probe {
                ubo.refl_center = [p.capture[0], p.capture[1], p.capture[2], p.intensity.max(0.0)];
                ubo.refl_extents = [
                    p.half_extents[0],
                    p.half_extents[1],
                    p.half_extents[2],
                    if p.box_projection { 1.0 } else { 0.0 },
                ];
                ubo.refl_box = [p.center[0], p.center[1], p.center[2], 0.0];
            }
            self.camera_targets[tidx].ubos[self.frame_index].write(0, bytemuck::bytes_of(&ubo));
            // Shared lighting resources (shadows, probes, lightmaps, env) come
            // from this frame's set; binding 0 is the target's own UBO.
            let src = self.frame_sets[self.frame_index];
            let copies: Vec<vk::CopyDescriptorSet> = (1..=9u32)
                .map(|b| {
                    vk::CopyDescriptorSet::default()
                        .src_set(src)
                        .src_binding(b)
                        .dst_set(set)
                        .dst_binding(b)
                        .descriptor_count(1)
                })
                .chain([ripple_copy(src, set)])
                .collect();
            let color_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            unsafe {
                device.update_descriptor_sets(&[], &copies);
                let to_color = vk::ImageMemoryBarrier::default()
                    .image(image)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::SHADER_READ)
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .subresource_range(color_range);
                let to_depth = vk::ImageMemoryBarrier::default()
                    .image(depth_img)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(depth_attach_layout())
                    .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(depth_aspect())
                            .level_count(1)
                            .layer_count(1),
                    );
                device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_color, to_depth],
                );
                let color_att = [vk::RenderingAttachmentInfo::default()
                    .image_view(color_view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: input.clear_color } })];
                let depth_att = vk::RenderingAttachmentInfo::default()
                    .image_view(depth_view)
                    .image_layout(depth_attach_layout())
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } });
                let stencil_att = stencil_attachment(&depth_att);
                let area = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: ext };
                device.cmd_begin_rendering(
                    cb,
                    &vk::RenderingInfo::default()
                        .render_area(area)
                        .layer_count(1)
                        .color_attachments(&color_att)
                        .depth_attachment(&depth_att)
                        .stencil_attachment(&stencil_att),
                );
                // Same negative-height viewport convention as the main pass.
                let vp = vk::Viewport {
                    x: 0.0,
                    y: ext.height as f32,
                    width: ext.width as f32,
                    height: -(ext.height as f32),
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                device.cmd_set_viewport(cb, 0, &[vp]);
                device.cmd_set_scissor(cb, 0, &[area]);
            }
            self.pipeline_cache.set_target_format(post::HDR_FORMAT);
            if input.draw_skybox {
                let _ = self.record_skybox(device, cb, set, true, false);
            }
            let planes = frustum_planes(view.camera.proj * view.camera.view);
            let pass_order: Vec<usize> = order
                .iter()
                .copied()
                .filter(|&d| {
                    if !sphere_in_planes(&planes, &input.draws[d], true) {
                        return false;
                    }
                    if !screen_material.get(input.draws[d].material.0).copied().unwrap_or(false) {
                        return true;
                    }
                    let mat = &self.materials[input.draws[d].material.0];
                    // Keep nested screens, but block direct self-feedback.
                    !mat.slots.contains(&Some(TextureHandle(self_tex)))
                })
                .collect();
            let drawn = self.record_scene_draws(device, cb, &pass_order, input, set, false);
            unsafe {
                device.cmd_end_rendering(cb);
                let to_read = vk::ImageMemoryBarrier::default()
                    .image(image)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .subresource_range(color_range);
                device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_read],
                );
            }
            drawn?;
        }
        Ok(())
    }

    /// A sampled color target for a world-space UI canvas.
    pub fn create_ui_target(&mut self, width: u32, height: u32) -> Result<TextureHandle> {
        let allocator = self.allocator();
        let texture = GpuTexture::render_target(
            &self.ctx.device,
            &mut allocator.lock().unwrap(),
            self.command_pool,
            self.ctx.queue,
            width.max(1),
            height.max(1),
            self.swapchain.format.format,
        )?;
        self.textures.push(texture);
        Ok(TextureHandle(self.textures.len() - 1))
    }

    /// Apply texture deltas for this frame's world-UI draws (before recording).
    fn prepare_world_ui(&mut self, draws: &[WorldUiDraw]) -> Result<()> {
        // Frees are deferred to the frame fences (see the drain after the fence
        // wait in render). Only a `set` that reuses a still-parked id needs an
        // idle + immediate free (rare; egui does not recycle managed ids).
        for d in draws {
            let reuses = |ids: &[egui::TextureId]| d.draw.textures_delta.set.iter().any(|(id, _)| ids.contains(id));
            let parked = self
                .world_egui_pending_free
                .iter()
                .chain(self.world_egui_free_slots.iter().flatten())
                .any(|(t, ids)| *t == d.target.0 && reuses(ids));
            if parked {
                unsafe {
                    let _ = self.ctx.device.wait_idle("wait:device_wait_idle@prepare_world_ui");
                }
                let all = std::mem::take(&mut self.world_egui_pending_free)
                    .into_iter()
                    .chain(self.world_egui_free_slots.iter_mut().flat_map(std::mem::take));
                for (target, ids) in all {
                    if let Some(r) = self.world_egui.get_mut(&target) {
                        let _ = r.free_textures(&ids);
                    }
                }
            }
        }
        for d in draws {
            if !self.world_egui.contains_key(&d.target.0) {
                let r = egui_ash_renderer::Renderer::with_gpu_allocator(
                    self.allocator(),
                    self.ctx.device.clone(),
                    egui_ash_renderer::DynamicRendering {
                        color_attachment_format: self.swapchain.format.format,
                        depth_attachment_format: None,
                    },
                    egui_ash_renderer::Options {
                        in_flight_frames: FRAMES_IN_FLIGHT,
                        enable_depth_test: false,
                        enable_depth_write: false,
                        srgb_framebuffer: true,
                    },
                )
                .map_err(|e| anyhow::anyhow!("creating world UI egui renderer: {e}"))?;
                self.world_egui.insert(d.target.0, r);
            }
            // Uploads are recorded into the frame's command buffer (`record_textures`
            // in render_frame), not submitted here with a queue wait.
            if !d.draw.textures_delta.set.is_empty() {
                self.world_egui_uploads.push((d.target.0, d.draw.textures_delta.set.clone()));
            }
            if !d.draw.textures_delta.free.is_empty() {
                self.world_egui_pending_free
                    .push((d.target.0, d.draw.textures_delta.free.clone()));
            }
        }
        Ok(())
    }

    /// Paint each world-UI canvas into its target (inside the frame's cb).
    fn record_world_ui(&mut self, device: &ash::Device, cb: vk::CommandBuffer, draws: &[WorldUiDraw]) {
        for d in draws {
            let Some(tex) = self.textures.get(d.target.0) else {
                continue;
            };
            let (image, view) = (tex.image, tex.view);
            let Some(r) = self.world_egui.get_mut(&d.target.0) else {
                continue;
            };
            let extent = vk::Extent2D {
                width: d.extent[0].max(1),
                height: d.extent[1].max(1),
            };
            image_barrier(
                device, cb, image, vk::ImageAspectFlags::COLOR,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            );
            let atts = [vk::RenderingAttachmentInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0; 4] },
                })];
            let info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
                .layer_count(1)
                .color_attachments(&atts);
            unsafe {
                device.cmd_begin_rendering(cb, &info);
                device.cmd_set_viewport(cb, 0, &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: extent.width as f32,
                    height: extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }]);
                device.cmd_set_scissor(cb, 0, &[vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent }]);
            }
            if let Err(e) = r.cmd_draw(cb, extent, d.draw.pixels_per_point, &d.draw.primitives) {
                tracing::warn!("world UI draw: {e}");
            }
            unsafe { device.cmd_end_rendering(cb) };
            image_barrier(
                device, cb, image, vk::ImageAspectFlags::COLOR,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ,
            );
        }
    }

    /// Whether the GPU lighting bake can run (ray query support).
    pub fn supports_baking(&self) -> bool {
        self.ctx.ray_tracing()
    }

    /// Run the GPU lighting bake (lightmaps + probe SH). Blocks until done;
    /// invoked from an explicit editor action, never the frame loop.
    pub fn bake_lighting(&self, input: &BakeInput<'_>) -> Result<BakeOutput> {
        let allocator = self.allocator();
        bake::bake(
            &self.ctx,
            &allocator,
            self.command_pool,
            &self.meshes,
            input,
        )
    }

    /// Begin a chunked bake: builds the acceleration structure + pipelines once;
    /// step it one unit per frame with `bake_step`, then `bake_finish`.
    pub fn bake_begin(&self, input: &BakeInput<'_>) -> Result<bake::BakeJob> {
        let allocator = self.allocator();
        bake::BakeJob::begin(&self.ctx, &allocator, self.command_pool, &self.meshes, input)
    }

    /// Advance a chunked bake by one unit (one lightmap, or the probe batch).
    pub fn bake_step(&self, job: &mut bake::BakeJob) -> Result<bake::BakeStep> {
        let allocator = self.allocator();
        job.step(&self.ctx, &allocator, self.command_pool, &self.meshes)
    }

    /// Tear down a chunked bake and collect its (possibly partial) output.
    pub fn bake_finish(&self, job: bake::BakeJob) -> BakeOutput {
        let allocator = self.allocator();
        job.finish(&self.ctx, &allocator)
    }

    /// (Re)upload the cached Global Distance Field for the GPU GI march. Call only
    /// when geometry changes; `gi_march` then reuses it every trace. No-op if the
    /// GPU pipeline is unavailable.
    #[allow(clippy::too_many_arguments)]
    pub fn gi_set_gdf(
        &mut self,
        dist: &[f32],
        index: &[u32],
        dims: [u32; 3],
        min: [f32; 3],
        max: [f32; 3],
        materials: &[GpuGiMaterial],
    ) {
        if let Err(e)=self.gi_set_gdf_identified(dist,index,dims,min,max,materials,&[]) {
            tracing::warn!("GPU GI set_gdf failed: {e:#}");
        }
    }

    /// Upload a global field with exact scene sources parallel to its material
    /// table. Source metadata is published only after the upload succeeds.
    #[allow(clippy::too_many_arguments)]
    pub fn gi_set_gdf_identified(&mut self, dist: &[f32], index: &[u32], dims: [u32;3],
        min: [f32;3], max: [f32;3], materials: &[GpuGiMaterial], sources: &[Option<GiSource>]) -> Result<()> {
        let allocator = self.allocator();
        if let Some(gpu) = self.gpu_gi.as_mut() {
            gpu.set_gdf(&self.ctx, &allocator, self.command_pool, dist, index, dims, min, max, materials, sources)?;
            self.invalidate_screen_gi();
        }
        Ok(())
    }

    /// Complete mesh-field snapshot for realtime software tracing. Mesh voxels
    /// are shared across instances/transform updates; in-flight frames retain
    /// their immutable snapshot until their fences retire.
    pub fn gi_set_mesh_sdfs(&mut self, inputs: &[GiMeshSdf]) -> Result<()> {
        let allocator=self.allocator();
        if let Some(gpu)=self.gpu_gi.as_mut() {
            if gpu.mesh_sdfs.set(&self.ctx.device,&mut allocator.lock().unwrap(),self.command_pool,self.ctx.queue,inputs)? {
                gpu.gdf_revision=gpu.gdf_revision.wrapping_add(1);
                self.invalidate_screen_gi();
            }
        }
        Ok(())
    }

    /// Whether the GPU GI compute pipeline initialised. When false the realtime-GI
    /// driver skips the GDF build/upload entirely and marches on the CPU instead.
    pub fn gi_gpu_available(&self) -> bool {
        self.gpu_gi.is_some()
    }

    /// True while an async GPU march is submitted but not yet read back.
    pub fn gi_marching(&self) -> bool {
        self.gpu_gi.as_ref().is_some_and(|g| g.is_marching())
    }

    /// True once a GDF is uploaded AND realtime GI is enabled, so the
    /// screen-space gather should run. The GDF stays cached when GI is toggled
    /// off (so re-enabling is instant), so the `gi_enabled` gate is what
    /// actually stops the per-frame trace — without it, disabling GI in the UI
    /// left the camera-gated screen trace running (the static/moving FPS cliff).
    pub fn gi_has_gdf(&self) -> bool {
        self.gi_enabled && self.gpu_gi.as_ref().is_some_and(|g| g.has_gdf())
    }

    /// Enable/disable the realtime screen-space GI trace without dropping the
    /// cached GDF. Called by the GI driver each frame from the scene's GI state.
    pub fn set_gi_enabled(&mut self, enabled: bool) {
        if self.gi_enabled != enabled {
            self.invalidate_screen_gi();
        }
        self.gi_enabled = enabled;
    }

    /// Discard both views' temporal lighting after a discontinuous input change.
    fn invalidate_screen_gi(&mut self) {
        self.sgi_history_valid = false;
        self.preview_sgi_history_valid = false;
        self.sgi_last_view_proj = None;
        self.preview_sgi_last_view_proj = None;
    }

    /// Write a GPU timestamp into `slot` of this frame's pool (no-op when timing
    /// is inactive). Stage = ALL_COMMANDS so the stamp lands after everything
    /// recorded before it has completed — clean zone brackets.
    fn gpu_ts(&self, cb: vk::CommandBuffer, slot: u32) {
        // Slots 0/1 (the FRAME zone) are always stamped when the device has
        // timestamps: a minimal always-on GPU frame time for the hitch log and
        // frame log. The other zones need `ts_active` (profiler window).
        if self.ts_active || (slot < 2 && self.gpu_ts_period > 0.0) {
            unsafe {
                self.ctx.device.cmd_write_timestamp2(
                    cb,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    self.frames[self.frame_index].timestamps,
                    slot,
                );
            }
        }
    }

    fn gpu_zone_begin(&self, cb: vk::CommandBuffer, zone: usize) {
        self.gpu_ts(cb, (2 * zone) as u32);
        self.ctx.checkpoint(cb, gpu_zone::BEGIN[zone.min(gpu_zone::BEGIN.len() - 1)]);
        // The main Flux trace also stamps its gather / denoise / integrate.
        if (zone == gpu_zone::GI || zone == gpu_zone::CAMERA_GI)
            && self.ts_active
            && let Some(g) = self.gpu_gi.as_ref()
        {
            g.timestamps.set(Some((self.frames[self.frame_index].timestamps, zone == gpu_zone::CAMERA_GI)));
        }
    }

    fn gpu_zone_end(&self, cb: vk::CommandBuffer, zone: usize) {
        self.gpu_ts(cb, (2 * zone + 1) as u32);
        self.ctx.checkpoint(cb, gpu_zone::END[zone.min(gpu_zone::END.len() - 1)]);
        if (zone == gpu_zone::GI || zone == gpu_zone::CAMERA_GI)
            && let Some(g) = self.gpu_gi.as_ref()
        {
            g.timestamps.set(None);
        }
    }

    /// Open the profiler as a second OS window (raw handles from the editor's
    /// winit window). No-op if one is already open.
    pub fn open_profiler_window(
        &mut self,
        display: raw_window_handle::RawDisplayHandle,
        window: raw_window_handle::RawWindowHandle,
        width: u32,
        height: u32,
    ) -> Result<()> {
        if self.profiler_window.is_some() {
            return Ok(());
        }
        let allocator = self.allocator();
        let pw = profiler_window::ProfilerWindow::new(
            &self.ctx, allocator, display, window, width.max(1), height.max(1), self.vsync,
        )?;
        self.profiler_window = Some(pw);
        // Timestamps were disabled while closed; discard any stale primed flags
        // so the first readback waits for fresh writes (no garbage frame).
        for p in self.gpu_ts_primed.iter_mut() {
            *p = false;
        }
        Ok(())
    }

    pub fn profiler_window_open(&self) -> bool {
        self.profiler_window.is_some()
    }

    pub fn resize_profiler_window(&mut self, width: u32, height: u32) {
        if let Some(pw) = self.profiler_window.as_mut() {
            pw.request_resize(width, height);
        }
    }

    pub fn close_profiler_window(&mut self) {
        if let Some(mut pw) = self.profiler_window.take() {
            pw.destroy(&self.ctx);
        }
    }

    /// Render the profiler egui UI to its window. Caller passes the tessellated
    /// output from the profiler's own egui context. No-op if not open.
    /// False while the (throttled, secondary) profiler window is not due for a
    /// redraw; callers skip running its egui frame so no texture delta is lost.
    pub fn profiler_due(&self) -> bool {
        self.profiler_window.as_ref().is_none_or(|p| p.due())
    }

    pub fn render_profiler(&mut self, draw: &EguiDraw) -> Result<()> {
        if let Some(pw) = self.profiler_window.as_mut() {
            pw.render(&self.ctx, draw)?;
        }
        Ok(())
    }

    /// Cache the emissive sphere area-lights the screen-GI gather samples (NEE).
    /// Changed emitters invalidate both views' temporal lighting.
    pub fn gi_set_emitters(&mut self, emitters: &[GpuGiEmitter]) {
        // Keep accumulating while inputs are unchanged; a moved or disabled
        // emitter must not leave its old lighting in either camera's history.
        if self.gi_emitters.as_slice() != emitters {
            self.gi_emitters.clear();
            self.gi_emitters.extend_from_slice(emitters);
            self.invalidate_screen_gi();
        }
    }

    /// Set the runtime Flux trace parameters (Environment tab → Flux GI). A
    /// changed setting re-traces so the new quality/intensity takes effect.
    pub fn set_flux_settings(&mut self, s: FluxSettings) {
        // Gate the re-trace on an actual change: the driver pushes settings every
        // frame, so resetting unconditionally would (like gi_set_emitters) stop the
        // GI ever converging — endless shimmer on a static scene.
        if self.flux_settings != s {
            self.flux_settings = s;
            self.invalidate_screen_gi();
        }
    }

    /// Toggle the screen-GI depth→world Y-flip (corrects a flipped gather without
    /// a rebuild).
    pub fn set_screen_gi_flip_y(&mut self, flip: bool) {
        if self.screen_gi_flip_y != flip {
            self.invalidate_screen_gi();
        }
        self.screen_gi_flip_y = flip;
    }

    pub fn screen_gi_flip_y(&self) -> bool {
        self.screen_gi_flip_y
    }

    /// Begin an async GPU march (submit, don't wait). Returns true if a march was
    /// started. The result is collected later via [`gi_march_poll`] so the trace
    /// never blocks the frame.
    pub fn gi_march_begin(&mut self, m: &GpuGiMarch<'_>) -> bool {
        let allocator = self.allocator();
        let Some(gpu) = self.gpu_gi.as_mut() else {
            return false;
        };
        match gpu.march_begin(&self.ctx, &allocator, m) {
            Ok(started) => started,
            Err(e) => {
                tracing::warn!("GPU GI march failed to start: {e:#}");
                false
            }
        }
    }

    /// Collect a finished async GPU march, or None if still running / none in
    /// flight. One `ProbeSh` per probe.
    pub fn gi_march_poll(&mut self) -> Option<Vec<ProbeSh>> {
        let allocator = self.allocator();
        let gpu = self.gpu_gi.as_mut()?;
        gpu.march_poll(&self.ctx, &allocator)
    }

    /// Raw Vulkan handles (as `u64`) + queue family, for sharing this device with
    /// OpenXR (`XR_KHR_vulkan_enable2`): (instance, physical_device, device,
    /// queue_family_index). The single graphics queue is index 0.
    pub fn vulkan_raw_handles(&self) -> (u64, u64, u64, u32) {
        use ash::vk::Handle as _;
        (
            self.ctx.instance.handle().as_raw(),
            self.ctx.physical_device.as_raw(),
            self.ctx.device.handle().as_raw(),
            self.ctx.queue_family,
        )
    }

    /// The swapchain colour format as a raw `VkFormat` (u32). The VR (OpenXR)
    /// swapchains must use this so the scene pipelines match; pass it to
    /// `XrSession::setup_stereo` and `render_xr_eye`.
    pub fn color_format_raw(&self) -> u32 {
        self.swapchain.format.format.as_raw() as u32
    }

    /// Set the sampler state (wrap, filter, anisotropy, mip bias) used when
    /// texture `handle` is bound in a material slot. Materials pick it up when
    /// they are created or their textures are set (`set_material_textures`), so
    /// call this right after the upload. The default key costs nothing.
    pub fn set_texture_sampler(&mut self, handle: TextureHandle, key: SamplerKey) {
        self.sampler_cache.set_texture(handle.0, key);
    }

    /// Change a texture's sampler state while it is in use: every material
    /// that binds `handle` gets its set rebound (a fresh set swapped in, no
    /// device idle). Returns how many materials were rebound; 0 when the key
    /// did not change.
    pub fn update_texture_sampler(&mut self, handle: TextureHandle, key: SamplerKey) -> usize {
        if !self.sampler_cache.set_texture(handle.0, key) {
            return 0;
        }
        let users: Vec<(usize, [Option<TextureHandle>; MATERIAL_SLOTS])> = self
            .materials
            .iter()
            .enumerate()
            .filter(|(i, m)| !self.freed_materials.contains(i) && m.slots.contains(&Some(handle)))
            .map(|(i, m)| (i, m.slots))
            .collect();
        for (i, slots) in &users {
            self.bind_material_slots(MaterialHandle(*i), slots);
        }
        users.len()
    }

    pub fn upload_texture(&mut self, data: &TextureData) -> Result<TextureHandle> {
        crate::hitch::count("tex_upload", 1);
        let _h = crate::hitch::scope("upload_texture");
        let allocator = self.allocator();
        let texture = GpuTexture::upload(
            &self.ctx.device,
            &mut allocator.lock().unwrap(),
            self.command_pool,
            self.ctx.queue,
            data,
        )?;
        self.textures.push(texture);
        Ok(TextureHandle(self.textures.len() - 1))
    }

    /// Main-thread upload of a pre-compressed (BC) texture — the single-queue
    /// fallback for the material texture path (no transfer queue available).
    pub fn upload_compressed_texture(
        &mut self,
        tex: &CompressedTexture,
    ) -> Result<TextureHandle> {
        let allocator = self.allocator();
        let texture = GpuTexture::upload_compressed(
            &self.ctx.device,
            &mut allocator.lock().unwrap(),
            self.command_pool,
            self.ctx.queue,
            tex,
        )?;
        self.textures.push(texture);
        Ok(TextureHandle(self.textures.len() - 1))
    }

    /// Upload same-sized compressed pages as the layers of one 2D-array
    /// texture (a cooked sprite atlas). Register it with
    /// `register_sprite_texture(handle, true)`; the page index is the layer.
    pub fn upload_compressed_texture_array(&mut self, pages: &[&CompressedTexture]) -> Result<TextureHandle> {
        let allocator = self.allocator();
        let texture = GpuTexture::upload_compressed_layers(
            &self.ctx.device,
            &mut allocator.lock().unwrap(),
            self.command_pool,
            self.ctx.queue,
            pages,
            true,
        )?;
        self.textures.push(texture);
        Ok(TextureHandle(self.textures.len() - 1))
    }

    /// The queue one-time uploads run on: the second queue of the family (shared
    /// with the background `Uploader` through `alloc::queue_lock`) so a big
    /// upload does not queue behind the frames in flight on the graphics queue;
    /// the graphics queue when the device has only one. `ZESTENGINE_UPLOAD_QUEUE=graphics`
    /// forces the old behaviour (measurement).
    fn upload_queue(&self) -> vk::Queue {
        static FORCE_GRAPHICS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let graphics = *FORCE_GRAPHICS.get_or_init(|| std::env::var("ZESTENGINE_UPLOAD_QUEUE").is_ok_and(|v| v == "graphics"));
        if graphics { self.ctx.queue } else { self.ctx.transfer_queue.unwrap_or(self.ctx.queue) }
    }

    /// Create a background asset uploader (loader thread). Returns None when the
    /// device exposes no second queue (uploads then stay on the main thread).
    /// The uploader is `Send`: move it to a worker, call `prepare`, then hand the
    /// `PreparedScene` back to `install_prepared` on the main thread.
    pub fn uploader(&self) -> Option<Uploader> {
        let queue = self.ctx.transfer_queue?;
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(self.ctx.queue_family);
        let command_pool = unsafe { self.ctx.device.create_command_pool(&pool_info, None).ok()? };
        Some(Uploader {
            device: self.ctx.device.clone(),
            allocator: self.allocator(),
            queue,
            command_pool,
            rt: self.ctx.ray_tracing(),
        })
    }

    /// Install pre-uploaded resources from a worker into the renderer's arrays,
    /// returning the per-resource handles (in upload order) + per-mesh skinned
    /// flags so the scene can reference them. Main thread only.
    pub fn install_prepared(
        &mut self,
        prepared: PreparedScene,
    ) -> (Vec<MeshHandle>, Vec<TextureHandle>, Vec<bool>) {
        let meshes = prepared
            .meshes
            .into_iter()
            .map(|m| {
                self.meshes.push(m);
                MeshHandle(self.meshes.len() - 1)
            })
            .collect();
        let textures = prepared
            .textures
            .into_iter()
            .map(|t| {
                self.textures.push(t);
                TextureHandle(self.textures.len() - 1)
            })
            .collect();
        (meshes, textures, prepared.skinned)
    }

    /// Current renderer-side resource counts (used to attribute resources
    /// installed by a content origin).
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    pub fn texture_count(&self) -> usize {
        self.textures.len()
    }

    pub fn material_count(&self) -> usize {
        self.materials.len()
    }

    /// Destroy a `PreparedScene` that was never installed (its GPU allocations
    /// are owned by the prepared buffers, not the renderer arrays).
    pub fn free_prepared(&mut self, mut prepared: PreparedScene) {
        // No idle: a prepared scene was never installed, so no renderer frame
        // has referenced it, and the worker's uploads waited on their own fences.
        let allocator = self.allocator();
        let mut alloc = allocator.lock().unwrap();
        for mut mesh in prepared.meshes.drain(..) {
            mesh.vertex_buffer.destroy(&self.ctx.device, &mut alloc);
            mesh.index_buffer.destroy(&self.ctx.device, &mut alloc);
        }
        for mut texture in prepared.textures.drain(..) {
            texture.destroy(&self.ctx.device, &mut alloc);
        }
    }

    /// Free scene resources by renderer index, in place: the slot stays as a
    /// tombstone so every surviving handle keeps its index. Only resources the
    /// caller has proven unreferenced by the scene may be passed here.
    ///
    /// No device idle: in-flight frames may still draw the meshes, sample the
    /// textures and bind the material sets, so the GPU objects are parked on the
    /// deferred-free queue and destroyed `FRAMES_IN_FLIGHT + 1` rendered frames
    /// later (`run_deferred_frees`), when the fences of every submission that
    /// could have used them have retired. Everything else (tombstoning, cluster
    /// and skinner bookkeeping) happens now.
    pub fn free_resources(
        &mut self,
        meshes: &HashSet<usize>,
        textures: &HashSet<usize>,
        materials: &HashSet<usize>,
    ) -> Result<()> {
        let fallback_view = self.default_textures[0].view;
        for &mesh in meshes {
            if mesh >= self.meshes.len() || self.freed_meshes.contains(&mesh) {
                continue;
            }
            let skin = self.skinner.as_mut().and_then(|s| s.take(mesh));
            if let Some(c) = self.cluster.as_mut() {
                c.store.remove(mesh, self.frame_serial);
            }
            let m = &mut self.meshes[mesh];
            let (mut vb, mut ib) = (m.vertex_buffer.take(), m.index_buffer.take());
            self.freed_meshes.insert(mesh);
            self.defer_free(move |device, alloc| {
                vb.destroy(device, alloc);
                ib.destroy(device, alloc);
                if let Some(skin) = skin {
                    skin.destroy(device, alloc);
                }
            });
        }
        for &texture in textures {
            if texture >= self.textures.len() || self.freed_textures.contains(&texture) {
                continue;
            }
            let t = &mut self.textures[texture];
            let mut old = t.take();
            t.view = fallback_view;
            let ripple = self.ripple_mips.remove(&texture).map(|(view, _)| view);
            self.freed_textures.insert(texture);
            self.defer_free(move |device, alloc| {
                old.destroy(device, alloc);
                if let Some(view) = ripple {
                    unsafe { device.destroy_image_view(view, None) };
                }
            });
        }
        self.tex_pending.retain(|(t, _)| !self.freed_textures.contains(t));
        let material_pool = self.material_pool;
        let custom_pool = self.custom_pool;
        for &material in materials {
            if material >= self.materials.len() || self.freed_materials.contains(&material) {
                continue;
            }
            let m = &mut self.materials[material];
            let mut fx = m.fx_ubo.take();
            let mut custom = m.custom_ubo.take();
            let (set, custom_set) = (m.set, m.custom_set);
            m.set = vk::DescriptorSet::null();
            m.custom_set = vk::DescriptorSet::null();
            self.freed_materials.insert(material);
            self.defer_free(move |device, alloc| {
                fx.destroy(device, alloc);
                if let Some(ubo) = custom.as_mut() {
                    ubo.destroy(device, alloc);
                }
                if set != vk::DescriptorSet::null() {
                    unsafe { let _ = device.free_descriptor_sets(material_pool, &[set]); };
                }
                if custom_set != vk::DescriptorSet::null() {
                    unsafe { let _ = device.free_descriptor_sets(custom_pool, &[custom_set]); };
                }
            });
        }
        Ok(())
    }

    /// Park a GPU destroy until no in-flight frame can still use the objects.
    fn defer_free(&mut self, f: impl FnOnce(&ash::Device, &mut Allocator) + Send + 'static) {
        self.deferred_frees.push((FRAMES_IN_FLIGHT + 1, Box::new(f)));
    }

    /// Count the deferred frees down one rendered frame and run the expired
    /// ones; `all` runs everything (the caller guarantees the device is idle).
    fn run_deferred_frees(&mut self, all: bool) {
        if all {
            self.apply_all_shared();
        }
        self.run_shared_frees(all);
        if self.deferred_frees.is_empty() {
            return;
        }
        for (frames, _) in &mut self.deferred_frees {
            *frames = frames.saturating_sub(1);
        }
        let Some(allocator) = self.allocator.clone() else { return };
        let mut alloc = allocator.lock().unwrap();
        let mut i = 0;
        while i < self.deferred_frees.len() {
            if all || self.deferred_frees[i].0 == 0 {
                let (_, f) = self.deferred_frees.remove(i);
                f(&self.ctx.device, &mut alloc);
            } else {
                i += 1;
            }
        }
    }

    /// Wait for the device and run every parked free now (shutdown, tests).
    /// This is the synchronous path: it stalls like the old `free_resources`.
    pub fn flush_deferred_frees(&mut self) -> Result<()> {
        unsafe { self.ctx.device.wait_idle("wait:device_wait_idle@flush_deferred_frees")? };
        self.run_deferred_frees(true);
        Ok(())
    }

    /// Bytes currently allocated from the device allocator (tests: leaks, growth).
    pub fn gpu_allocated_bytes(&self) -> u64 {
        self.allocator().lock().unwrap().generate_report().total_allocated_bytes
    }

    /// GPU frees still parked (for tests and the stats overlay).
    pub fn deferred_free_count(&self) -> usize {
        self.deferred_frees.len() + self.shared_frees.len()
    }

    pub fn create_material(&mut self, desc: &MaterialDesc) -> Result<MaterialHandle> {
        let layouts = [self.pipeline_cache.set1_layout];
        let set = unsafe {
            self.ctx.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.material_pool)
                    .set_layouts(&layouts),
            )?[0]
        };

        // 12 texture slots in slot order; bindings skip 4 (the FX UBO).
        let slots = [
            desc.albedo,
            desc.normal,
            desc.orm,
            desc.emission,
            desc.opacity,
            desc.emission_mask,
            desc.matcap[0],
            desc.matcap_mask[0],
            desc.matcap[1],
            desc.matcap_mask[1],
            desc.matcap[2],
            desc.matcap_mask[2],
            desc.ao,
            desc.roughness,
            desc.metallic,
            desc.displacement,
            desc.detail[0],
            desc.detail[1],
            desc.detail[2],
            desc.detail[3],
            desc.detail[4],
        ];
        let bindings = TEXTURE_BINDINGS;
        let image_infos: Vec<[vk::DescriptorImageInfo; 1]> = slots
            .iter()
            .enumerate()
            .map(|(i, slot)| {
                let (view, sampler) = match slot {
                    Some(handle) => (self.textures[handle.0].view, self.sampler_cache.for_texture(&self.ctx.device, handle.0)),
                    None => (self.default_textures[default_texture_index(i)].view, self.sampler),
                };
                [vk::DescriptorImageInfo::default()
                    .sampler(sampler)
                    .image_view(view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
            })
            .collect();
        let writes: Vec<_> = image_infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(bindings[i])
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(info)
            })
            .collect();
        unsafe { self.ctx.device.update_descriptor_sets(&writes, &[]) };
        self.write_area_atlas(set);

        // Per-material FX uniform block (set 1, binding 4).
        let fx_ubo = {
            let allocator = self.allocator();
            let mut alloc = allocator.lock().unwrap();
            make_fx_ubo(
                &self.ctx.device,
                &mut alloc,
                MaterialFx::from_params(&desc.params),
                "material fx ubo",
            )?
        };
        let fx_info = [vk::DescriptorBufferInfo::default()
            .buffer(fx_ubo.handle)
            .range(vk::WHOLE_SIZE)];
        let fx_write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&fx_info);
        unsafe { self.ctx.device.update_descriptor_sets(&[fx_write], &[]) };

        let mut features = desc.features;
        features.normal_map &= desc.normal.is_some();
        self.materials.push(Material {
            name: desc.name.clone(),
            params: desc.params,
            features,
            render_queue: default_render_queue(features.alpha_mode),
            has_normal_texture: desc.normal.is_some(),
            error: desc.error,
            custom_shader: None,
            custom_data: [0.0; 16],
            custom_keywords: [0; pipeline::CUSTOM_KEYWORDS],
            custom_set: vk::DescriptorSet::null(),
            custom_ubo: None,
            custom_views: [vk::ImageView::null(); pipeline::CUSTOM_TEXTURES],
            outline: None,
            fur: None,
            set,
            fx_ubo,
            albedo_mean: [1.0; 3],
            metallic_mean: 1.0,
            slots,
            water_slot: 0,
        });
        Ok(MaterialHandle(self.materials.len() - 1))
    }

    /// Rewrite a material's FX uniform block from its current params (set 1,
    /// binding 4). Called after a param edit so rim / animated emission update
    /// live. Host-visible write; materials are static at runtime so there's no
    /// in-flight hazard there, and an editor edit at worst tears for one frame.
    pub fn upload_material_fx(&mut self, handle: MaterialHandle) {
        let Some(material) = self.materials.get_mut(handle.0) else {
            return;
        };
        let mut fx = MaterialFx::from_params(&material.params);
        fx.water[0] = material.water_slot as f32;
        material.fx_ubo.write(0, bytemuck::bytes_of(&fx));
    }

    /// Point a water material at its body in `FrameInput::water`.
    pub fn set_material_water_slot(&mut self, handle: MaterialHandle, slot: u32) {
        let Some(material) = self.materials.get_mut(handle.0) else {
            return;
        };
        if material.water_slot != slot {
            material.water_slot = slot;
            self.upload_material_fx(handle);
        }
    }

    /// Rebind a material's 12 texture slots on its existing descriptor set
    /// (binding order = create_material's slot list; binding 4 is the FX UBO and
    /// is untouched). `None` in a slot binds the 1×1 default for that slot.
    /// The set may be in use by an in-flight frame, so a fresh set is written
    /// and swapped in and the old one freed `FRAMES_IN_FLIGHT + 1` frames later
    /// (`rebind_set`); only a full pool falls back to a device idle.
    pub fn set_material_textures(&mut self, handle: MaterialHandle, slots: &[Option<TextureHandle>; 16]) {
        let Some(m) = self.materials.get(handle.0) else {
            return;
        };
        let mut full = m.slots;
        full[..16].copy_from_slice(slots);
        self.bind_material_slots(handle, &full);
    }

    /// Rebind all [`MATERIAL_SLOTS`] texture slots at once: the 16 base slots, then
    /// the detail albedo / normal / ORM / mask and the translucency (thickness) map
    /// (set 1 bindings 17-21).
    pub fn bind_material_slots(&mut self, handle: MaterialHandle, slots: &[Option<TextureHandle>; MATERIAL_SLOTS]) {
        if let Some(m) = self.materials.get_mut(handle.0) {
            m.slots = *slots;
        }
        let Some(material) = self.materials.get(handle.0) else {
            return;
        };
        let old_set = material.set;
        let views: Vec<(vk::ImageView, vk::Sampler)> = slots
            .iter()
            .enumerate()
            .map(|(i, slot)| match slot {
                Some(h) => (self.textures[h.0].view, self.sampler_cache.for_texture(&self.ctx.device, h.0)),
                None => (self.default_textures[default_texture_index(i)].view, self.sampler),
            })
            .collect();
        // Binding 4 (the FX UBO) is copied over from the old set; the texture bindings are rewritten.
        let set = self.rebind_set(self.material_pool, self.pipeline_cache.set1_layout, old_set, &[4, AREA_ATLAS_BINDING, LTC_LUT_BINDING], "wait:device_wait_idle@set_material_textures");
        self.materials[handle.0].set = set;
        let image_infos: Vec<[vk::DescriptorImageInfo; 1]> = views
            .iter()
            .map(|(view, sampler)| {
                [vk::DescriptorImageInfo::default()
                    .sampler(*sampler)
                    .image_view(*view)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
            })
            .collect();
        let writes: Vec<_> = image_infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(TEXTURE_BINDINGS[i])
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(info)
            })
            .collect();
        unsafe { self.ctx.device.update_descriptor_sets(&writes, &[]) };
        // Slot 1 is the normal map; keep the feature gate's source in sync.
        self.materials[handle.0].has_normal_texture = slots[1].is_some();
    }

    /// Log where the GPU was if `err` is a lost device (once per run).
    /// The quality preset in effect.
    pub fn tier(&self) -> RenderTier {
        self.tier
    }

    /// Negotiated Vulkan device version (major, minor): 1.3 on desktop, 1.1 /
    /// 1.2 plus the KHR dynamic-rendering / sync2 extensions on older drivers.
    pub fn vulkan_version(&self) -> (u32, u32) {
        (vk::api_version_major(self.ctx.api_version), vk::api_version_minor(self.ctx.api_version))
    }

    pub fn report_device_lost(&self, err: &anyhow::Error) {
        self.ctx.report_device_lost(err);
    }

    /// Set (or clear) the equirectangular skybox texture. `None` reverts to
    /// the procedural gradient sky. The old texture and set 1 are freed once
    /// the frames in flight have retired (no device idle).
    pub fn set_skybox(&mut self, data: Option<&TextureData>) -> Result<()> {
        self.set_skybox_texture(data)?;
        self.rebuild_reflection_env(data)
    }

    /// Point the sky pass (set 1, slot 0) at `data` (an equirect, or the procedural
    /// sky for `None`) without touching the reflection environment.
    fn set_skybox_texture(&mut self, data: Option<&TextureData>) -> Result<()> {
        let texture = match data {
            Some(data) => {
                // Full mip chain: a large HDRI minified without mips aliases while the camera moves.
                let allocator = self.allocator();
                Some(sky::upload_equirect(&self.ctx.device, &mut allocator.lock().unwrap(), self.command_pool, self.upload_queue(), data)?)
            }
            None => None,
        };
        // A fresh set 1 (the old one may be bound by frames in flight) with the
        // new equirect in slot 0; the old set and texture are freed once those
        // frames have retired. No device idle.
        let view = texture.as_ref().map_or(self.default_textures[0].view, |t| t.view);
        let keep: Vec<u32> = [4, AREA_ATLAS_BINDING, LTC_LUT_BINDING].into_iter().chain(TEXTURE_BINDINGS[1..].iter().copied()).collect();
        let set = self.rebind_set(self.skybox_pool, self.pipeline_cache.set1_layout, self.skybox_set, &keep, "wait:device_wait_idle@set_skybox");
        self.skybox_set = set;
        // Repeat in U (the -X seam), clamp in V (the poles), trilinear + anisotropic.
        let sky_sampler = self.sampler_cache.get(&self.ctx.device, sky::SAMPLER_KEY);
        let info = [vk::DescriptorImageInfo::default()
            .sampler(sky_sampler)
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(TEXTURE_BINDINGS[0])
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&info);
        unsafe { self.ctx.device.update_descriptor_sets(&[write], &[]) };
        if let Some(mut old) = std::mem::replace(&mut self.skybox_texture, texture) {
            self.defer_free(move |device, alloc| old.destroy(device, alloc));
        }
        self.skybox_has_texture = data.is_some();
        self.skybox_is_cube = false;
        Ok(())
    }

    /// Rebuild the prefiltered environment reflection cube from the skybox (or a
    /// neutral cube when there's no skybox) and re-point the `samplerCube`
    /// binding (set 0, binding 7) on every set that samples it. Called from
    /// `set_skybox` (no device idle; the cube's upload is a one-time submit on the upload queue).
    fn rebuild_reflection_env(&mut self, data: Option<&TextureData>) -> Result<()> {
        let queue = self.upload_queue();
        let allocator = self.allocator();
        let new_env = {
            let mut alloc = allocator.lock().unwrap();
            match data {
                Some(eq) => ReflectionEnv::from_equirect(
                    &self.ctx.device,
                    &mut alloc,
                    self.command_pool,
                    queue,
                    eq,
                )?,
                None => ReflectionEnv::neutral(
                    &self.ctx.device,
                    &mut alloc,
                    self.command_pool,
                    queue,
                )?,
            }
        };
        self.swap_reflection_env(new_env);
        Ok(())
    }

    /// Replace the env reflection cube + re-point the `samplerCube` binding
    /// (set 0, binding 7) on every set that samples it (`shared_bind.rs`: frame
    /// sets, preview sets, and the reflection-capture and planar sets, which
    /// render the scene with the same layout; one that is skipped samples a
    /// stale or destroyed resource and captures come out black).
    fn swap_reflection_env(&mut self, new_env: ReflectionEnv) {
        self.swap_reflection_env_slot(0, new_env);
    }

    /// Exchange the two reflection-probe slots' cubes (the camera crossed from
    /// one probe's box into the other's), without re-uploading either.
    pub fn swap_reflection_slots(&mut self) {
        std::mem::swap(&mut self.reflection_env, &mut self.reflection_env_b);
        let sampler = self.sampler;
        let writes = [(7u32, self.reflection_env.view), (9, self.reflection_env_b.view)]
            .map(|(binding, view)| SharedWrite::sampled(SharedTarget::All, binding, sampler, view))
            .to_vec();
        // The cubes stay alive (only exchanged); no device idle.
        self.queue_shared_writes(writes, false);
    }

    /// The reflection-cube holder for a probe slot (0..`MAX_REFLECTION_PROBES`).
    fn reflection_slot_env(&mut self, slot: u8) -> &mut ReflectionEnv {
        match slot {
            1 => &mut self.reflection_env_b,
            2 => &mut self.reflection_env_c,
            3 => &mut self.reflection_env_d,
            _ => &mut self.reflection_env,
        }
    }

    /// Replace reflection-probe slot `slot`'s cube (binding 7/9/11/12 by slot)
    /// and re-point every scene descriptor set at it.
    fn swap_reflection_env_slot(&mut self, slot: u8, new_env: ReflectionEnv) {
        // In-flight frames may still sample the old cube: the sets are re-pointed
        // as their slots retire and the cube is destroyed after the last has. A
        // swap is infrequent (a probe-boundary crossing re-uploads a baked
        // probe, or the first capture changes the cube size), no device idle.
        let slot = (slot as usize).min(MAX_REFLECTION_PROBES - 1);
        let target = self.reflection_slot_env(slot as u8);
        let old = std::mem::replace(target, new_env);
        let view = self.reflection_slot_env(slot as u8).view;
        let binding = REFLECTION_BINDINGS[slot];
        let seq = self.queue_shared_writes(vec![SharedWrite::sampled(SharedTarget::All, binding, self.sampler, view)], false);
        self.defer_shared_free(seq, move |device, alloc| {
            let mut old = old;
            old.destroy(device, alloc);
        });
    }

    /// Set a cubemap / 6-texture skybox (face order +X,-X,+Y,-Y,+Z,-Z). The env
    /// reflection cube is built from the faces (sharp at mip 0 for the skybox,
    /// blurred mips for reflections), and the skybox renders from it (binding 7).
    pub fn set_skybox_cube(&mut self, faces: [&TextureData; 6]) -> Result<()> {
        let size = faces[0].width.clamp(1, MAX_BAKED_CUBE_SIZE);
        let chain = ReflectionEnv::cpu_chain(size, |face, size| ReflectionEnv::resample_face(faces[face], size));
        let env = {
            let allocator = self.allocator();
            let mut alloc = allocator.lock().unwrap();
            ReflectionEnv::upload_chain(&self.ctx.device, &mut alloc, self.command_pool, self.upload_queue(), &chain)?
        };
        self.swap_reflection_env(env);
        // The sky itself is drawn from its own equirect resampled from the faces, not from
        // the shared reflection environment: a live reflection capture overwrites that
        // cube, which made a cube-map sky change (resolution, ground, objects) whenever a
        // capture finished.
        let equirect = cube_chain_to_equirect(&chain);
        self.set_skybox_texture(Some(&equirect))?;
        Ok(())
    }

    /// Request a precomputed scene reflection capture from `center` on the next
    /// frame: the scene's surroundings are rendered into the reflection cube (not
    /// just the sky), so reflective surfaces show real geometry. Call after a
    /// scene loads / lighting settles.
    pub fn request_reflection_capture(&mut self, center: glam::Vec3, resolution: u32) {
        self.recapture_reflection = Some(ReflCaptureRequest {
            center,
            size: resolution.clamp(32, 8192),
            static_only: false,
            slot: 0,
            min_interval: 0.2,
            sliced: true,
            dynamic: true,
        });
    }

    /// Like `request_reflection_capture`, but draws ONLY static geometry and no
    /// realtime GI (Unity baked-probe semantics): dynamic objects, their GI
    /// bounces, and their emission never end up in a cube that's persisted to
    /// disk. Used by the Bake Reflections path.
    pub fn request_reflection_capture_static(&mut self, center: glam::Vec3, resolution: u32) {
        self.recapture_reflection = Some(ReflCaptureRequest {
            center,
            size: resolution.clamp(32, 8192),
            static_only: true,
            slot: 0,
            min_interval: 0.0,
            sliced: true,
            dynamic: false,
        });
    }

    /// Live capture into reflection-probe slot `slot` (0 = binding 7, 1 = the
    /// blended second probe at binding 9). One capture runs at a time;
    /// `min_interval` (seconds) paces it after the previous one finished and
    /// `sliced` spreads it over one face per frame. `dynamic` also draws
    /// non-static objects (lighting is live either way).
    pub fn request_reflection_capture_slot(
        &mut self,
        center: glam::Vec3,
        resolution: u32,
        slot: u8,
        min_interval: f32,
        sliced: bool,
        dynamic: bool,
    ) {
        self.recapture_reflection = Some(ReflCaptureRequest {
            center,
            size: resolution.clamp(32, 8192),
            static_only: false,
            slot: slot.min((MAX_REFLECTION_PROBES - 1) as u8),
            min_interval,
            sliced,
            dynamic,
        });
    }

    /// True while a requested reflection capture hasn't finished all 6 faces +
    /// the final mip-gen/swap (the capture runs one face per frame). The bake
    /// readback must wait for this to clear, or it reads the pre-capture cube.
    pub fn reflection_capture_in_flight(&self) -> bool {
        self.refl_job.is_some() || self.recapture_reflection.is_some()
    }

    /// Read back the current reflection-env cube (mip 0, all 6 faces) to CPU as
    /// RGBA16F (`hdr` TextureData), in (+X,-X,+Y,-Y,+Z,-Z) order — for BAKING the probe to disk. Uses
    /// `one_time_submit` (full GPU sync, never the in-frame command buffer), so a
    /// failure is contained (returns None) and can't hang the render loop. Call a
    /// frame or two after `request_reflection_capture` so the cube holds the
    /// captured scene, not the sky.
    pub fn read_reflection_faces(&self) -> Option<Vec<TextureData>> {
        if self.reflection_env.format != ENV_CUBE_FORMAT {
            return None; // a loaded baked cube, not a capture
        }
        let device = self.ctx.device.clone();
        let queue = self.ctx.queue;
        let pool = self.command_pool;
        let size = self.reflection_env.size;
        let image = self.reflection_env.image;
        let bytes_per_face = (size * size * 8) as u64;
        let allocator = self.allocator();
        let mut faces = Vec::with_capacity(6);
        for layer in 0..6u32 {
            let mut readback = {
                let mut alloc = allocator.lock().unwrap();
                crate::alloc::Buffer::new(
                    &device,
                    &mut alloc,
                    bytes_per_face,
                    vk::BufferUsageFlags::TRANSFER_DST,
                    gpu_allocator::MemoryLocation::GpuToCpu,
                    "reflprobe-readback",
                )
                .ok()?
            };
            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(layer)
                .layer_count(1);
            let copy = vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(layer)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D { width: size, height: size, depth: 1 });
            let submit = crate::alloc::one_time_submit(&device, pool, queue, |cb| unsafe {
                let to_src = vk::ImageMemoryBarrier::default()
                    .image(image)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::SHADER_READ)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .subresource_range(range);
                device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_src],
                );
                device.cmd_copy_image_to_buffer(
                    cb,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    readback.handle,
                    &[copy],
                );
                let back = vk::ImageMemoryBarrier::default()
                    .image(image)
                    .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .subresource_range(range);
                device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[back],
                );
            });
            let pixels = readback.read();
            {
                let mut alloc = allocator.lock().unwrap();
                readback.destroy(&device, &mut alloc);
            }
            if submit.is_err() || pixels.len() < bytes_per_face as usize {
                return None;
            }
            faces.push(TextureData {
                width: size,
                height: size,
                pixels: pixels[..bytes_per_face as usize].to_vec(),
                srgb: false,
                hdr: true,
            });
        }
        Some(faces)
    }

    /// Upload 6 baked face images (RGBA8 or RGBA16F, +X,-X,+Y,-Y,+Z,-Z) as the reflection
    /// env — the LOAD path for a disk-baked reflection probe (skips re-rendering
    /// the capture). Rebuilds the prefiltered roughness mip chain via `from_faces`.
    pub fn set_reflection_faces(&mut self, faces: &[TextureData; 6]) -> Result<()> {
        self.set_reflection_faces_slot(0, faces)
    }

    /// Upload a prebuilt baked cube (full mip chain, any [`HdrTextureFormat`],
    /// e.g. BC6H from a compressed `.reflprobe`) into slot `slot`.
    pub fn set_reflection_cube_slot(&mut self, slot: u8, cube: &HdrCube) -> Result<()> {
        let device = self.ctx.device.clone();
        let (queue, pool) = (self.ctx.queue, self.command_pool);
        let env = {
            let allocator = self.allocator();
            let mut alloc = allocator.lock().unwrap();
            ReflectionEnv::upload_chain(&device, &mut alloc, pool, queue, cube)?
        };
        self.swap_reflection_env_slot(slot.min((MAX_REFLECTION_PROBES - 1) as u8), env);
        Ok(())
    }

    /// Upload baked faces into reflection-probe slot `slot`.
    pub fn set_reflection_faces_slot(&mut self, slot: u8, faces: &[TextureData; 6]) -> Result<()> {
        let device = self.ctx.device.clone();
        let queue = self.ctx.queue;
        let pool = self.command_pool;
        let env = {
            let allocator = self.allocator();
            let mut alloc = allocator.lock().unwrap();
            let refs: [&TextureData; 6] = [
                &faces[0], &faces[1], &faces[2], &faces[3], &faces[4], &faces[5],
            ];
            ReflectionEnv::from_faces(&device, &mut alloc, pool, queue, &refs)?
        };
        self.swap_reflection_env_slot(slot.min((MAX_REFLECTION_PROBES - 1) as u8), env);
        Ok(())
    }

    /// Render the scene into the reflection cube from `center` (6 faces, 90° FOV)
    /// and box-blur a roughness mip chain, then swap it in as the reflection env.
    /// Reuses the frame's lights/shadows. NOTE: face orientation + prefilter want
    /// on-device verification.
    #[allow(clippy::too_many_arguments)]
    /// Allocate the capture cube + depth target for a new per-frame reflection
    /// capture job; faces are rendered one per frame by `step_refl_capture`.
    fn begin_refl_capture(
        &mut self,
        center: glam::Vec3,
        resolution: u32,
        static_only: bool,
        slot: u8,
        sliced: bool,
        dynamic: bool,
    ) -> Result<ReflCaptureJob> {
        let size = resolution.clamp(32, 8192);
        // Capture targets persist across captures; only a resolution change
        // reallocates them (rare; the old one is destroyed after the frames in
        // flight that may still use it).
        if self.refl_target.as_ref().is_none_or(|t| t.size != size) {
            let device = self.ctx.device.clone();
            let allocator = self.allocator();
            let mut alloc = allocator.lock().unwrap();
            if let Some(mut old) = self.refl_target.take() {
                self.deferred_frees.push((FRAMES_IN_FLIGHT + 1, Box::new(move |device, alloc| old.destroy(device, alloc))));
            }
            self.refl_target = Some(ReflCaptureTarget::new(&device, &mut alloc, size)?);
        }
        Ok(ReflCaptureJob { center, size, next_face: 0, static_only, slot, sliced, dynamic })
    }

    /// Advance the active reflection capture by one cube face, recorded into the
    /// frame's command buffer; after the 6th face the next call finalizes (mip
    /// chain + copy into the live env). One face/frame keeps any frame cheap.
    /// An unsliced job records all six faces and the finish in this frame.
    #[allow(clippy::too_many_arguments)]
    fn step_refl_capture(
        &mut self,
        cb: vk::CommandBuffer,
        lights: [GpuLight; MAX_LIGHTS],
        count: usize,
        shadow_vp: [[[f32; 4]; 4]; MAX_SHADOW_VIEWS],
        cascade_splits: [f32; 4],
        input: &FrameInput,
        order: &[usize],
    ) -> Result<()> {
        let whole = self.refl_job.as_ref().is_some_and(|j| !j.sliced);
        loop {
            self.record_refl_face(cb, lights, count, shadow_vp, cascade_splits, input, order)?;
            if !whole || self.refl_job.is_none() {
                return Ok(());
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record_refl_face(
        &mut self,
        cb: vk::CommandBuffer,
        lights: [GpuLight; MAX_LIGHTS],
        count: usize,
        shadow_vp: [[[f32; 4]; 4]; MAX_SHADOW_VIEWS],
        cascade_splits: [f32; 4],
        input: &FrameInput,
        order: &[usize],
    ) -> Result<()> {
        use glam::Mat4;
        let Some(job) = self.refl_job.as_ref() else {
            return Ok(());
        };
        let (face_i, center, static_only, size) = (job.next_face, job.center, job.static_only, job.size);
        let static_draws = static_only || !job.dynamic;
        if face_i >= 6 {
            return self.finish_refl_capture(cb);
        }
        let Some(target) = self.refl_target.as_ref() else {
            self.refl_job = None;
            return Ok(());
        };
        let (cube, face_view, depth_img, depth_view) = (target.cube, target.face_views[face_i], target.depth_img, target.depth_view);
        let device = self.ctx.device.clone();
        let ext = vk::Extent2D { width: size, height: size };

        // Face look directions (+X,-X,+Y,-Y,+Z,-Z) matching the cube sampling
        // (shared with the round-trip test that proves the orientation).
        let (dir, up) = CUBE_CAPTURE_FACES[face_i];
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.05, 2000.0);
        let cam = CameraData { view: Mat4::look_at_rh(center, center + dir, up), proj, position: center };
        // Only what this face sees; baked-probe captures take static geometry
        // only (Unity semantics: dynamic objects and their GI glow are transient
        // and must not persist into a `.reflprobe`).
        let planes = frustum_planes(proj * cam.view);
        let face_order: Vec<usize> = order
            .iter()
            .copied()
            .filter(|&i| (!static_draws || input.draws[i].static_geometry) && sphere_in_planes(&planes, &input.draws[i], true))
            .collect();

        let mut ubo = frame_ubo(
            &cam, input, lights, count, shadow_vp, cascade_splits,
            &self.frame_probe_volumes, [0.5; 2], [1.0, 1.0], [size as f32, size as f32],
            glam::Mat4::IDENTITY.to_cols_array_2d(),
        );
        ubo.misc[2] *= self.shadow_requested_dim as f32 / self.shadow_atlas.dim as f32;
        ubo.debug[1] = 0.0; // never render a debug view-mode into the cube
        // debug.z in the capture: 0 = no screen-GI (realtime probe still gets
        // FluxVoxel probe GI); -1 = static bake — the shader ALSO drops the
        // realtime probe-GI term, so dynamic emitters' bounce stays out.
        ubo.debug[2] = if static_only { -1.0 } else { 0.0 };
        ubo.debug[3] = 1.0; // linear HDR into the float cube
        ubo.ssr = [0.0; 4];
        // Capture marker (refl_center.w < 0): surfaces in the cube must not
        // sample the env cube, which here is this probe's own previous capture
        // (that fed reflections back into the probe every recapture).
        ubo.refl_center[3] = -1.0;
        // One UBO + set per face, so a whole-cube frame doesn't overwrite the
        // camera of faces still waiting to execute.
        let slot = self.frame_index * 6 + face_i;
        self.refl_capture_ubos[slot].write(0, bytemuck::bytes_of(&ubo));
        let capture_set = self.refl_capture_sets[slot];

        let face_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .base_array_layer(face_i as u32)
            .layer_count(1);
        unsafe {
            // The previous capture's mip-gen / copy read this face: wait for it.
            let b = [
                vk::ImageMemoryBarrier2::default()
                    .image(cube)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .subresource_range(face_range),
                vk::ImageMemoryBarrier2::default()
                    .image(depth_img)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(depth_attach_layout())
                    .src_stage_mask(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
                    .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
                    .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(depth_aspect())
                            .level_count(1)
                            .layer_count(1),
                    ),
            ];
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&b));
            let color_att = vk::RenderingAttachmentInfo::default()
                .image_view(face_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: input.clear_color } });
            let depth_att = vk::RenderingAttachmentInfo::default()
                .image_view(depth_view)
                .image_layout(depth_attach_layout())
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } });
            let stencil_att = stencil_attachment(&depth_att);
            let atts = [color_att];
            let info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: ext })
                .layer_count(1)
                .color_attachments(&atts)
                .depth_attachment(&depth_att)
                .stencil_attachment(&stencil_att);
            device.cmd_begin_rendering(cb, &info);
            // POSITIVE-height viewport (NOT the main pass's negative-height
            // Y-flip): the face cameras + perspective_rh with a positive viewport
            // reproduce the engine's `face_dir` cube convention exactly. The
            // un-flipped viewport reverses the winding, so scene draws use the
            // mirrored pipelines.
            let vp = vk::Viewport { x: 0.0, y: 0.0, width: size as f32, height: size as f32, min_depth: 0.0, max_depth: 1.0 };
            device.cmd_set_viewport(cb, 0, &[vp]);
            device.cmd_set_scissor(cb, 0, &[vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: ext }]);
        }
        self.capturing_refl = true;
        // Linear HDR into the float cube (no inline tonemap / grading).
        self.pipeline_cache.set_target_format(ENV_CUBE_FORMAT);
        if input.draw_skybox {
            let _ = self.record_skybox(&device, cb, capture_set, true, false);
        }
        let _ = self.record_scene_draws(&device, cb, &face_order, input, capture_set, false);
        self.capturing_refl = false;
        unsafe {
            device.cmd_end_rendering(cb);
            // Face -> transfer source for the mip-gen at the end.
            let b = [vk::ImageMemoryBarrier2::default()
                .image(cube)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .subresource_range(face_range)];
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&b));
        }
        self.refl_job.as_mut().unwrap().next_face += 1;
        Ok(())
    }

    /// Finalize the reflection capture inside the frame command buffer: build
    /// the roughness mip chain on the capture cube, then copy every mip into the
    /// live environment cube (descriptors never change, nothing waits on the CPU).
    fn finish_refl_capture(&mut self, cb: vk::CommandBuffer) -> Result<()> {
        let Some(job) = self.refl_job.take() else {
            return Ok(());
        };
        let slot = job.slot;
        self.refl_last_finish = Some(std::time::Instant::now());
        let Some(target) = self.refl_target.as_ref() else {
            return Ok(());
        };
        let (cube, size, mips) = (target.cube, target.size, target.mips);
        let device = self.ctx.device.clone();
        // A live env of another size (or the skybox env) is replaced once by a
        // capture-sized cube (in-flight frames keep sampling the old one).
        // A baked compressed / compact cube can't take the RGBA16F copy either.
        let cur = if slot == 1 { &self.reflection_env_b } else { &self.reflection_env };
        let (cur_size, cur_format) = (cur.size, cur.format);
        if cur_size != size || cube_mip_count(cur_size) != mips || cur_format != ENV_CUBE_FORMAT {
            let (image, alloc, m) = {
                let allocator = self.allocator();
                let mut a = allocator.lock().unwrap();
                create_cube_image(&device, &mut a, ENV_CUBE_FORMAT, size, "reflection env cube")?
            };
            let view = create_cube_view(&device, image, ENV_CUBE_FORMAT, m)?;
            self.swap_reflection_env_slot(slot, ReflectionEnv { image, view, alloc: Some(alloc), size, format: ENV_CUBE_FORMAT });
            // Fresh image: give it a defined layout before the copy below.
            let b = [vk::ImageMemoryBarrier2::default()
                .image(image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(m).layer_count(6))];
            unsafe { device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&b)) };
        }
        let env = if slot == 1 { self.reflection_env_b.image } else { self.reflection_env.image };
        let mip_range = |mip: u32, count: u32| {
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(mip)
                .level_count(count)
                .layer_count(6)
        };
        let barrier = |img: vk::Image, r: vk::ImageSubresourceRange, old: vk::ImageLayout, new: vk::ImageLayout,
                       ss: vk::PipelineStageFlags2, sa: vk::AccessFlags2, ds: vk::PipelineStageFlags2, da: vk::AccessFlags2| {
            vk::ImageMemoryBarrier2::default()
                .image(img)
                .old_layout(old)
                .new_layout(new)
                .src_stage_mask(ss)
                .src_access_mask(sa)
                .dst_stage_mask(ds)
                .dst_access_mask(da)
                .subresource_range(r)
        };
        let run = |b: &[vk::ImageMemoryBarrier2]| unsafe {
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(b));
        };
        let t = vk::PipelineStageFlags2::TRANSFER;
        unsafe {
            // Mip chain by successive linear blits (mip 0 is TRANSFER_SRC from
            // the face renders).
            let mut src_size = size as i32;
            for mip in 1..mips {
                let dst_size = (src_size / 2).max(1);
                run(&[barrier(cube, mip_range(mip, 1), vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    t, vk::AccessFlags2::empty(), t, vk::AccessFlags2::TRANSFER_WRITE)]);
                let blit = vk::ImageBlit::default()
                    .src_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).mip_level(mip - 1).layer_count(6))
                    .src_offsets([vk::Offset3D { x: 0, y: 0, z: 0 }, vk::Offset3D { x: src_size, y: src_size, z: 1 }])
                    .dst_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).mip_level(mip).layer_count(6))
                    .dst_offsets([vk::Offset3D { x: 0, y: 0, z: 0 }, vk::Offset3D { x: dst_size, y: dst_size, z: 1 }]);
                device.cmd_blit_image(cb, cube, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, cube, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[blit], vk::Filter::LINEAR);
                run(&[barrier(cube, mip_range(mip, 1), vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    t, vk::AccessFlags2::TRANSFER_WRITE, t, vk::AccessFlags2::TRANSFER_READ)]);
                src_size = dst_size;
            }
            // Live env: shader-read -> transfer dst (after this and earlier frames' sampling).
            run(&[barrier(env, mip_range(0, mips), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::empty(),
                t, vk::AccessFlags2::TRANSFER_WRITE)]);
            let regions: Vec<vk::ImageCopy> = (0..mips)
                .map(|mip| {
                    let s = (size >> mip).max(1);
                    let layers = vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).mip_level(mip).layer_count(6);
                    vk::ImageCopy::default()
                        .src_subresource(layers)
                        .dst_subresource(layers)
                        .extent(vk::Extent3D { width: s, height: s, depth: 1 })
                })
                .collect();
            device.cmd_copy_image(cb, cube, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, env, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &regions);
            run(&[barrier(env, mip_range(0, mips), vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                t, vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ)]);
        }
        Ok(())
    }

    /// (Re)create the planar RT array at half the main target size. No-op when
    /// already the right size. `device_wait_idle` guards the in-flight frame
    /// still sampling the old array on resize.
    fn ensure_planar_targets(&mut self, main_extent: [u32; 2]) -> Result<()> {
        let extent = vk::Extent2D {
            width: (main_extent[0] / 2).max(64),
            height: (main_extent[1] / 2).max(64),
        };
        if let Some(t) = &self.planar_targets
            && t.extent.width == extent.width
            && t.extent.height == extent.height
        {
            return Ok(());
        }
        let allocator = self.allocator();
        let mut alloc = allocator.lock().unwrap();
        // The old array may still be rendered into / sampled by frames in flight
        // (the sets that name it are rewritten every frame): destroyed after them.
        if let Some(mut old) = self.planar_targets.take() {
            self.deferred_frees.push((FRAMES_IN_FLIGHT + 1, Box::new(move |device, alloc| old.destroy(device, alloc))));
        }
        self.planar_targets = Some(PlanarTargets::new(
            &self.ctx.device,
            &mut alloc,
            extent,
            PLANAR_FORMAT,
        )?);
        Ok(())
    }

    /// (Re)create the opaque scene copies at the main target size and point
    /// every frame set's bindings 5/6 at them. Waits for idle on a resize only,
    /// like the planar targets.
    fn ensure_opaque_copy(&mut self, main_extent: [u32; 2]) -> Result<()> {
        let extent = vk::Extent2D { width: main_extent[0].max(1), height: main_extent[1].max(1) };
        if self.opaque_copy.as_ref().is_some_and(|c| c.extent == extent) {
            return Ok(());
        }
        let old = {
            let allocator = self.allocator();
            let mut alloc = allocator.lock().unwrap();
            let new = OpaqueCopy::new(&self.ctx.device, &mut alloc, extent)?;
            self.opaque_copy.replace(new)
        };
        let c = self.opaque_copy.as_ref().unwrap();
        let writes = vec![
            SharedWrite::sampled(SharedTarget::FrameSet, 5, self.lightmap_sampler, c.depth_view),
            SharedWrite::sampled(SharedTarget::FrameSet, 6, self.grab_sampler, c.color_view),
        ];
        // Called from inside the frame (after its fence wait): this slot's set is
        // written now, the other slot's when it next starts. No device idle.
        let seq = self.queue_shared_writes(writes, true);
        if let Some(mut old) = old {
            self.defer_shared_free(seq, move |device, alloc| old.destroy(device, alloc));
        }
        Ok(())
    }

    /// Where the transparent part of `order` starts, when it holds a water draw.
    fn water_split(&self, order: &[usize], input: &FrameInput) -> Option<usize> {
        let material = |i: usize| &self.materials[input.draws[i].material.0];
        // Water and grab-pass materials both read the opaque copy.
        let water = order.iter().position(|&i| (material(i).features.water || material(i).features.grab) && !material(i).error)?;
        let transparent = order
            .iter()
            .position(|&i| material(i).render_queue >= RENDER_QUEUE_TRANSPARENT)
            .unwrap_or(water);
        Some(transparent.min(water))
    }

    /// Copy the main pass's colour + depth into the opaque copies. Called
    /// between its two rendering scopes; leaves the sources as attachments.
    fn record_opaque_copy(&self, device: &ash::Device, cb: vk::CommandBuffer, pass: &MainPass) {
        let Some(c) = self.opaque_copy.as_ref() else {
            return;
        };
        let extent = vk::Extent3D {
            width: pass.extent.width.min(c.extent.width),
            height: pass.extent.height.min(c.extent.height),
            depth: 1,
        };
        let color = vk::ImageAspectFlags::COLOR;
        let depth = vk::ImageAspectFlags::DEPTH;
        let depth_all = depth_aspect();
        let tests = vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
        let readers = vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER;
        let copy = vk::PipelineStageFlags2::COPY;
        image_barrier(
            device, cb, pass.color_image, color,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            copy, vk::AccessFlags2::TRANSFER_READ,
        );
        image_barrier(
            device, cb, pass.depth_image, depth_all,
            depth_attach_layout(), vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            tests, vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            copy, vk::AccessFlags2::TRANSFER_READ,
        );
        for (image, aspect) in [(c.color, color), (c.depth, depth_all)] {
            image_barrier(
                device, cb, image, aspect,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                readers, vk::AccessFlags2::empty(),
                copy, vk::AccessFlags2::TRANSFER_WRITE,
            );
        }
        let region = |aspect| {
            let layers = vk::ImageSubresourceLayers::default().aspect_mask(aspect).layer_count(1);
            vk::ImageCopy::default().src_subresource(layers).dst_subresource(layers).extent(extent)
        };
        unsafe {
            device.cmd_copy_image(
                cb, pass.color_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                c.color, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region(color)],
            );
            device.cmd_copy_image(
                cb, pass.depth_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                c.depth, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region(depth)],
            );
        }
        image_barrier(
            device, cb, pass.color_image, color,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            copy, vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        );
        image_barrier(
            device, cb, pass.depth_image, depth_all,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL, depth_attach_layout(),
            copy, vk::AccessFlags2::empty(),
            tests, vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );
        // Rough transmission: the box-filtered mip chain of the colour copy, only
        // on frames that draw a transmission material (see `opaque_mips`).
        let build_mips = self.opaque_mips && c.mips > 1;
        let xfer = vk::PipelineStageFlags2::ALL_TRANSFER;
        if build_mips {
            let blit_layers = vk::ImageSubresourceLayers::default().aspect_mask(color).layer_count(1);
            image_barrier_mips(
                device, cb, c.color, color, 0, 1,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                xfer, vk::AccessFlags2::TRANSFER_WRITE, xfer, vk::AccessFlags2::TRANSFER_READ,
            );
            let (mut w, mut h) = (c.extent.width as i32, c.extent.height as i32);
            for level in 1..c.mips {
                let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
                image_barrier_mips(
                    device, cb, c.color, color, level, 1,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    readers, vk::AccessFlags2::empty(), xfer, vk::AccessFlags2::TRANSFER_WRITE,
                );
                let blit = vk::ImageBlit::default()
                    .src_subresource(blit_layers.mip_level(level - 1))
                    .src_offsets([vk::Offset3D::default(), vk::Offset3D { x: w, y: h, z: 1 }])
                    .dst_subresource(blit_layers.mip_level(level))
                    .dst_offsets([vk::Offset3D::default(), vk::Offset3D { x: nw, y: nh, z: 1 }]);
                unsafe {
                    device.cmd_blit_image(
                        cb, c.color, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        c.color, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[blit], vk::Filter::LINEAR,
                    );
                }
                image_barrier_mips(
                    device, cb, c.color, color, level, 1,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    xfer, vk::AccessFlags2::TRANSFER_WRITE, xfer, vk::AccessFlags2::TRANSFER_READ,
                );
                (w, h) = (nw, nh);
            }
            image_barrier_mips(
                device, cb, c.color, color, 0, c.mips,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                xfer, vk::AccessFlags2::TRANSFER_READ, readers, vk::AccessFlags2::SHADER_SAMPLED_READ,
            );
        }
        for (image, aspect) in [(c.color, color), (c.depth, depth_all)] {
            if build_mips && image == c.color {
                continue;
            }
            image_barrier(
                device, cb, image, aspect,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                copy, vk::AccessFlags2::TRANSFER_WRITE,
                readers, vk::AccessFlags2::SHADER_SAMPLED_READ,
            );
        }
    }

    /// Scene draws of a main camera pass (rendering already begun on `pass`).
    /// With water among them, the pass ends after the opaque queues, its colour
    /// and depth are copied for the water's refraction, and it resumes with
    /// LOAD for the transparent draws.
    #[allow(clippy::too_many_arguments)]
    fn record_main_scene_draws(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        order: &[usize],
        input: &FrameInput,
        frame_set: vk::DescriptorSet,
        gbuf: bool,
        pass: &MainPass,
        screen_view: usize,
    ) -> Result<RenderStats> {
        // The first main-camera depth pass runs the late occlusion pass after
        // its early GPU batches (unless a depth prepass already did).
        self.gpu_late_split(gpu_cull::LateSplit {
            colors: std::iter::once(pass.color_view).chain(pass.gbuf_view).collect(),
            depth: (pass.depth_image, pass.depth_read_view),
            depth_att: pass.depth_view,
            extent: pass.extent,
            xr: None,
            eye: None,
        });
        let capture=input.gi_debug==0 && self.flux_screen_history.as_ref().is_some_and(|h|h.ready(screen_view,self.frame_serial));
        let water_split=if self.water_copy {self.water_split(order,input)} else {None};
        let split=if capture {Some(water_split.unwrap_or_else(||order.iter().position(|&i|
            self.materials[input.draws[i].material.0].render_queue>=RENDER_QUEUE_TRANSPARENT).unwrap_or(order.len())))} else {water_split};
        // Sprite layers marked "behind 3D transparents": cut the draw order where
        // the transparent queue starts and draw those sprites there.
        let behind = self.draw_mode != DrawMode::Prepass
            && !self.capturing_refl
            && self.sprite_gpu.as_ref().is_some_and(|s| s.behind_items() > 0);
        let cut = behind.then(|| {
            order
                .iter()
                .position(|&i| self.materials[input.draws[i].material.0].render_queue >= RENDER_QUEUE_TRANSPARENT)
                .unwrap_or(order.len())
        });
        let front = if behind { SpritePart::Front } else { SpritePart::All };
        let Some(split) = split else {
            let stats = self.record_scene_cut(device, cb, order, cut, input, frame_set, gbuf)?;
            self.record_impostors(device, cb, frame_set, gbuf);
            self.record_world_text(device, cb, frame_set, gbuf, front);
            return Ok(stats);
        };
        // The cut is never before `split` (the split is the first transparent or water draw).
        let mut stats = self.record_scene_draws(device, cb, &order[..split], input, frame_set, gbuf)?;
        unsafe { device.cmd_end_rendering(cb) };
        if water_split.is_some() {self.record_opaque_copy(device, cb, pass);}
        if capture {self.capture_flux_history(device,cb,pass,screen_view);}
        if let Some(g) = self.gpu_cull.as_mut()
            && water_split.is_some() && self.opaque_copy.is_some()
        {
            g.hiz_from_copy = true;
        }
        Self::resume_main_pass(device,cb,pass);
        let rest = self.record_scene_cut(device, cb, &order[split..], cut.map(|c| c.saturating_sub(split)), input, frame_set, gbuf)?;
        add_draw_stats(&mut stats, &rest);
        self.record_impostors(device, cb, frame_set, gbuf);
        self.record_world_text(device, cb, frame_set, gbuf, front);
        Ok(stats)
    }

    fn capture_flux_history(&mut self,device:&ash::Device,cb:vk::CommandBuffer,pass:&MainPass,id:usize) {
        let Some(history)=self.flux_screen_history.as_mut() else {return;};
        if !history.ready(id,self.frame_serial) {return;}
        let color=vk::ImageAspectFlags::COLOR;
        let read=vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        for (image,aspect,layout,stage,access) in [
            (pass.color_image,color,vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,vk::AccessFlags2::COLOR_ATTACHMENT_WRITE),
            (pass.depth_image,depth_aspect(),depth_attach_layout(),vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS|vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)] {
            image_barrier(device,cb,image,aspect,layout,read,stage,access,vk::PipelineStageFlags2::COMPUTE_SHADER,vk::AccessFlags2::SHADER_SAMPLED_READ);
        }
        history.capture(device,cb,id,self.frame_serial,self.frame_index,pass.color_view,pass.depth_read_view);
        for (image,aspect,layout,stage,access) in [
            (pass.color_image,color,vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,vk::AccessFlags2::COLOR_ATTACHMENT_READ|vk::AccessFlags2::COLOR_ATTACHMENT_WRITE),
            (pass.depth_image,depth_aspect(),depth_attach_layout(),vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS|vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ|vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)] {
            image_barrier(device,cb,image,aspect,read,layout,vk::PipelineStageFlags2::COMPUTE_SHADER,vk::AccessFlags2::SHADER_SAMPLED_READ,stage,access);
        }
    }
    fn resume_main_pass(device:&ash::Device,cb:vk::CommandBuffer,pass:&MainPass) {
        let load = |view: vk::ImageView| {
            vk::RenderingAttachmentInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
        };
        let mut colors = vec![load(pass.color_view)];
        if let Some(g) = pass.gbuf_view {
            colors.push(load(g));
        }
        if let Some(v) = pass.velocity_view {
            colors.push(load(v));
        }
        if let Some(r) = pass.reactive_view {
            colors.push(load(r));
        }
        let depth = vk::RenderingAttachmentInfo::default()
            .image_view(pass.depth_view)
            .image_layout(depth_attach_layout())
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE);
        let stencil = stencil_attachment(&depth);
        let area = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: pass.extent };
        let info = vk::RenderingInfo::default()
            .render_area(area)
            .layer_count(1)
            .color_attachments(&colors)
            .depth_attachment(&depth)
            .stencil_attachment(&stencil);
        let viewport = vk::Viewport {
            x: 0.0,
            y: pass.extent.height as f32,
            width: pass.extent.width as f32,
            height: -(pass.extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        unsafe {
            device.cmd_begin_rendering(cb, &info);
            device.cmd_set_viewport(cb, 0, &[viewport]);
            device.cmd_set_scissor(cb, 0, &[area]);
        }
    }

    /// `record_scene_draws` with the behind-3D-transparents sprites recorded at
    /// `cut` (an index into `order`) when there is one. The draws after the sprites
    /// rebind pipeline and sets (`BoundDraw` starts empty), so the sprite layout's
    /// own bindings never leak into them.
    #[allow(clippy::too_many_arguments)]
    fn record_scene_cut(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        order: &[usize],
        cut: Option<usize>,
        input: &FrameInput,
        frame_set: vk::DescriptorSet,
        gbuf: bool,
    ) -> Result<RenderStats> {
        let Some(cut) = cut.map(|c| c.min(order.len())) else {
            return self.record_scene_draws(device, cb, order, input, frame_set, gbuf);
        };
        let mut stats = self.record_scene_draws(device, cb, &order[..cut], input, frame_set, gbuf)?;
        let pass = self.pipeline_cache.pass_formats();
        let velocity = self.velocity_active;
        if let Some(s) = self.sprite_gpu.as_mut() {
            s.record(device, cb, self.frame_index, frame_set, pass, gbuf, velocity, SpritePart::Behind);
        }
        let rest = self.record_scene_draws(device, cb, &order[cut..], input, frame_set, gbuf)?;
        add_draw_stats(&mut stats, &rest);
        Ok(stats)
    }

    /// World-space and 3D text, after the scene's transparent draws, in the
    /// pass that is currently recording (depth tested, multiview aware).
    fn record_world_text(&mut self, device: &ash::Device, cb: vk::CommandBuffer, frame_set: vk::DescriptorSet, gbuf: bool, sprites: SpritePart) {
        if self.draw_mode == DrawMode::Prepass || self.capturing_refl {
            return;
        }
        let pass = self.pipeline_cache.pass_formats();
        let velocity = self.velocity_active;
        if let Some(t) = self.text_gpu.as_mut() {
            t.record_world(device, cb, self.frame_index, frame_set, pass, gbuf, velocity);
        }
        // The 2D sprite pass closes the scene pass: over opaque, 3D transparents
        // and text (see sprite_gpu.rs for the interleave rule).
        if let Some(s) = self.sprite_gpu.as_mut() {
            s.record(device, cb, self.frame_index, frame_set, pass, gbuf, velocity, sprites);
        }
    }

    /// Record one mirrored scene pass per active planar surface into the planar
    /// RT array (inside the frame command buffer, before the main scene pass).
    /// Each pass renders from the main camera reflected about the plane, with an
    /// oblique near plane clipping everything behind the mirror, and excludes
    /// the plane's own draws. The main pass samples the layers at screen UV.
    #[allow(clippy::too_many_arguments)]
    fn record_planar_passes(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        input: &FrameInput,
        lights: [GpuLight; MAX_LIGHTS],
        count: usize,
        shadow_vp: [[[f32; 4]; 4]; MAX_SHADOW_VIEWS],
        cascade_splits: [f32; 4],
        order: &[usize],
        planar_n: usize,
        main_extent: [f32; 2],
    ) -> Result<()> {
        use glam::Vec3;
        let Some(targets) = &self.planar_targets else {
            return Ok(());
        };
        let ext = targets.extent;
        let depth_view = targets.depth_view;
        let image = targets.image;
        let layer_views = targets.layer_views.clone();
        let depth_img = targets.depth_img;

        for (i, plane) in input.planar.iter().take(planar_n).enumerate() {
            // Plane normal toward the camera side, so the mirrored camera keeps
            // (and the oblique near plane clips at) the right half-space.
            let cam_pos = input.camera.position;
            let p0 = Vec3::from(plane.point);
            let mut n = Vec3::from(plane.normal).normalize_or(Vec3::Y);
            if n.dot(cam_pos - p0) < 0.0 {
                n = -n;
            }
            let mirror = mirror_matrix(p0, n);
            let view = input.camera.view * mirror;
            let mirrored_pos = mirror.transform_point3(cam_pos);
            // Clip plane in view space of the mirrored camera; w < 0 puts the
            // camera on the clipped side, as the oblique trick requires.
            let plane_w = glam::vec4(n.x, n.y, n.z, -n.dot(p0));
            let mut plane_v = view.inverse().transpose() * plane_w;
            if plane_v.w > 0.0 {
                plane_v = -plane_v;
            }
            let proj = oblique_projection(input.camera.proj, plane_v);
            let cam = CameraData { view, proj, position: mirrored_pos };
            let mut ubo = frame_ubo(
                &cam, input, lights, count, shadow_vp, cascade_splits,
                &self.frame_probe_volumes, [0.5; 2], [1.0, 1.0], main_extent,
                glam::Mat4::IDENTITY.to_cols_array_2d(),
            );
            ubo.misc[2] *= self.shadow_requested_dim as f32 / self.shadow_atlas.dim as f32;
            ubo.debug[1] = 0.0;
            ubo.debug[2] = 0.0; // no screen-GI in the mirrored pass
            ubo.debug[3] = 1.0; // linear HDR, sampled as radiance
            ubo.ssr = [0.0; 4];
            // Probe fallback stays available inside the mirror; planar sampling
            // itself stays off (refl_box.w = 0) — no recursive mirrors.
            if let Some(p) = input.reflection_probe {
                ubo.refl_center =
                    [p.capture[0], p.capture[1], p.capture[2], p.intensity.max(0.0)];
                ubo.refl_extents = [
                    p.half_extents[0],
                    p.half_extents[1],
                    p.half_extents[2],
                    if p.box_projection { 1.0 } else { 0.0 },
                ];
                ubo.refl_box = [p.center[0], p.center[1], p.center[2], 0.0];
            }
            self.planar_ubos[i + self.frame_index * MAX_PLANAR].write(0, bytemuck::bytes_of(&ubo));

            let face_range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .base_array_layer(i as u32)
                .layer_count(1);
            unsafe {
                let to_color = vk::ImageMemoryBarrier::default()
                    .image(image)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .subresource_range(face_range);
                let dep_b = vk::ImageMemoryBarrier::default()
                    .image(depth_img)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(depth_attach_layout())
                    .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(depth_aspect())
                            .level_count(1)
                            .layer_count(1),
                    );
                device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::FRAGMENT_SHADER
                        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_color, dep_b],
                );
                let color_att = vk::RenderingAttachmentInfo::default()
                    .image_view(layer_views[i])
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue { float32: input.clear_color },
                    });
                let depth_att = vk::RenderingAttachmentInfo::default()
                    .image_view(depth_view)
                    .image_layout(depth_attach_layout())
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
                    });
                let stencil_att = stencil_attachment(&depth_att);
                let atts = [color_att];
                let info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: ext })
                    .layer_count(1)
                    .color_attachments(&atts)
                    .depth_attachment(&depth_att)
                    .stencil_attachment(&stencil_att);
                device.cmd_begin_rendering(cb, &info);
                // Same negative-height viewport as the main pass, so a world
                // point lands at the same normalized framebuffer UV in both and
                // the forward pass can sample by gl_FragCoord / screen.
                let vp = vk::Viewport {
                    x: 0.0,
                    y: ext.height as f32,
                    width: ext.width as f32,
                    height: -(ext.height as f32),
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                device.cmd_set_viewport(cb, 0, &[vp]);
                device.cmd_set_scissor(
                    cb,
                    0,
                    &[vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: ext }],
                );
            }
            // The mirrored view flips winding: scene draws use the mirrored
            // pipelines (clockwise front face), like the cube capture.
            self.capturing_refl = true;
            self.pipeline_cache.set_target_format(PLANAR_FORMAT);
            if input.draw_skybox {
                let _ = self.record_skybox(device, cb, self.planar_sets[i + self.frame_index * MAX_PLANAR], true, false);
            }
            // The plane's own draws are excluded — a mirror can't contain
            // itself (coplanar with the near plane anyway).
            // Culled to the mirrored frustum (not the occlusion PVS: that is for
            // the real camera's cell, and the mirror sees from elsewhere).
            let planes = frustum_planes(proj * view);
            let pass_order: Vec<usize> = order
                .iter()
                .copied()
                .filter(|&d| {
                    input.draws[d].planar_index != i as i32 && sphere_in_planes(&planes, &input.draws[d], true)
                })
                .collect();
            let _ =
                self.record_scene_draws(device, cb, &pass_order, input, self.planar_sets[i + self.frame_index * MAX_PLANAR], false);
            self.capturing_refl = false;
            unsafe {
                device.cmd_end_rendering(cb);
                let to_read = vk::ImageMemoryBarrier::default()
                    .image(image)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .subresource_range(face_range);
                device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[to_read],
                );
            }
        }
        Ok(())
    }

    /// Draw the fullscreen skybox into the active rendering pass (call after
    /// begin_rendering + viewport/scissor, before scene geometry).
    fn record_skybox(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        frame_set: vk::DescriptorSet,
        hdr: bool,
        gbuf: bool,
    ) -> Result<()> {
        let mut sky_key = PipelineKey::skybox();
        sky_key.gbuf = gbuf;
        let pipeline = self.pipeline_cache.get(device, sky_key)?;
        let mut push = PushData {
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            base_color: [0.0; 4],
            emission: [0.0; 4],
            params0: [0.0; 4],
            params1: [0.0; 4],
        };
        push.params0[0] = if self.skybox_has_texture { 1.0 } else { 0.0 };
        // params0.z = cubemap mode: sample the env cube (binding 7) by direction.
        push.params0[2] = if self.skybox_is_cube { 1.0 } else { 0.0 };
        // params0.w = HDR output: skip inline tonemap (the post pass handles it).
        push.params0[3] = if hdr { 1.0 } else { 0.0 };
        if sky_debug() {
            // `ZESTENGINE_SKY_DEBUG=1`: tint the sky by frame parity (red even, blue odd) so
            // a frame drawn with stale data shows as a colour flip, and log the view matrix
            // change since the previous sky draw of the main view.
            let odd = self.frame_serial & 1 == 1;
            push.emission = if odd { [0.0, 0.0, 1.0, 1.0] } else { [1.0, 0.0, 0.0, 1.0] };
            if !self.capturing_refl && !gbuf {
                tracing::info!(target: "sky_debug", serial = self.frame_serial, slot = self.frame_index, skybox_set = ?self.skybox_set, cube = self.skybox_is_cube, tex = self.skybox_has_texture, "sky drawn");
            }
        }
        unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_cache.layout,
                0,
                &[frame_set, self.skybox_set],
                &[],
            );
            device.cmd_push_constants(
                cb,
                self.pipeline_cache.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push),
            );
            device.cmd_draw(cb, 3, 1, 0, 0);
        }
        Ok(())
    }

    /// Re-create the shadow atlas at a new per-layer resolution (clamped to a
    /// sane range). No-op when unchanged. Re-binds the array view on every
    /// frame + camera-preview descriptor set.
    pub fn set_shadow_resolution(&mut self, dim: u32) -> Result<()> {
        let dim = dim.clamp(256, self.tier.max_shadow_resolution());
        self.resize_shadow_atlas(dim, self.shadow_atlas.layers)
    }

    fn resize_shadow_atlas(&mut self, requested: u32, layers: usize) -> Result<()> {
        if requested == self.shadow_requested_dim && layers == self.shadow_atlas.layers {
            return Ok(());
        }
        let allocator = match &self.allocator {
            Some(a) => a.clone(),
            None => return Ok(()),
        };
        let new_atlas = ShadowAtlas::new(&self.ctx, &mut allocator.lock().unwrap(), requested, layers)?;
        self.shadow_requested_dim = requested;
        let old = std::mem::replace(&mut self.shadow_atlas, new_atlas);
        self.shadow_layers_valid = false;

        // Re-point binding 1 on every set that renders the scene: now in the
        // slots that have retired, at their next frame in the others. In-flight
        // frames keep sampling and rendering into the old atlas, which is
        // destroyed after the last of them. No device idle.
        let seq = self.queue_shared_writes(
            vec![SharedWrite {
                target: SharedTarget::All,
                binding: 1,
                resource: shared_bind::SharedResource::Image {
                    sampler: self.shadow_atlas.sampler,
                    view: self.shadow_atlas.array_view,
                    layout: vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL,
                },
            }],
            false,
        );
        self.defer_shared_free(seq, move |device, alloc| {
            let mut old = old;
            old.destroy(device, alloc);
        });
        Ok(())
    }

    /// Register a runtime-compiled custom fragment shader (SPIR-V bytes).
    /// Last main-view camera world position (from the most recent desktop frame),
    /// or None before the first frame. Used by the FluxVoxel camera-clipmap grid to
    /// follow the viewer. One frame stale — fine for a grid snapped to the spacing.
    pub fn last_view_pos(&self) -> Option<glam::Vec3> {
        self.last_frame_ubo.map(|u| {
            glam::Vec3::new(u.camera_pos[0], u.camera_pos[1], u.camera_pos[2])
        })
    }

    pub fn register_shader(&mut self, spirv: &[u8]) -> Result<ShaderId> {
        Ok(ShaderId(
            self.pipeline_cache
                .register_custom(&self.ctx.device, spirv)?,
        ))
    }

    /// [`Self::register_shader`] for a shader with a vertex hook: `hook_glsl` is the chunk
    /// `zestengine_assets::vertex_hook_glsl` assembled from the `// @vertex` section. It is
    /// compiled into the engine's vertex shaders (main, depth prepass, XR, shadow) with `glslc`;
    /// a hook that does not compile is an error here, naming the failing line.
    pub fn register_shader_with_vertex(&mut self, spirv: &[u8], hook_glsl: &str) -> Result<ShaderId> {
        Ok(ShaderId(self.pipeline_cache.register_custom_with_hook(&self.ctx.device, spirv, hook_glsl)?))
    }

    /// Whether displaced (tessellated) materials draw displaced on this device. Without
    /// `tessellationShader` they draw flat.
    pub fn tessellation_supported(&self) -> bool {
        self.pipeline_cache.tessellation()
    }

    /// Recompile and hot-swap the engine's standard fragment shader from its
    /// source on disk (dev convenience). Compiles with `glslc`; on any compile
    /// error the existing pipeline is kept and the error is returned so the
    /// caller can surface it without crashing the editor.
    pub fn reload_standard_shaders(&mut self) -> Result<()> {
        let frag_path = std::path::Path::new(ENGINE_SHADER_DIR).join("standard.frag");
        let output = std::process::Command::new("glslc")
            .arg("-o")
            .arg("-")
            .arg(&frag_path)
            .output()
            .with_context(|| format!("running glslc on {}", frag_path.display()))?;
        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "glslc failed for {}:\n{}",
                frag_path.display(),
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        unsafe {
            self.ctx.device.wait_idle("wait:device_wait_idle@reload_standard_shaders")?;
        }
        self.pipeline_cache
            .reload_standard_frag(&self.ctx.device, &output.stdout)?;
        Ok(())
    }

    /// Switch a material between the standard shader (None) and a custom one.
    pub fn set_material_shader(&mut self, handle: MaterialHandle, shader: Option<ShaderId>) {
        self.materials[handle.0].custom_shader = shader;
        if shader.is_some()
            && let Err(e) = self.ensure_custom_set(handle)
        {
            tracing::error!("custom material set: {e:#}");
        }
    }

    /// Allocate a material's set 2 (custom block + texture properties), with
    /// every binding valid (zero block, default textures).
    /// A descriptor set to write a material's new bindings into while frames
    /// may still be using `old`: allocate a fresh set from `pool`, copy the
    /// bindings in `keep` over (single-descriptor bindings), park `old` on the
    /// deferred-free queue and return the new one. When the pool has no room
    /// (materials are at the cap) fall back to idling the device and rewriting
    /// `old` in place, the previous behaviour.
    fn rebind_set(&mut self, pool: vk::DescriptorPool, layout: vk::DescriptorSetLayout, old: vk::DescriptorSet, keep: &[u32], site: &'static str) -> vk::DescriptorSet {
        let layouts = [layout];
        let new = unsafe {
            self.ctx.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts),
            )
        };
        match new {
            Ok(sets) => {
                let new = sets[0];
                let copies: Vec<vk::CopyDescriptorSet> = keep
                    .iter()
                    .map(|&b| {
                        vk::CopyDescriptorSet::default()
                            .src_set(old)
                            .src_binding(b)
                            .dst_set(new)
                            .dst_binding(b)
                            .descriptor_count(1)
                    })
                    .collect();
                unsafe { self.ctx.device.update_descriptor_sets(&[], &copies) };
                self.defer_free(move |device, _| unsafe {
                    let _ = device.free_descriptor_sets(pool, &[old]);
                });
                new
            }
            Err(_) => {
                unsafe {
                    let _ = self.ctx.device.wait_idle(site);
                }
                old
            }
        }
    }

    fn ensure_custom_set(&mut self, handle: MaterialHandle) -> Result<()> {
        if self.materials[handle.0].custom_set != vk::DescriptorSet::null() {
            return Ok(());
        }
        let layouts = [self.pipeline_cache.set2_layout];
        let set = unsafe {
            self.ctx.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.custom_pool)
                    .set_layouts(&layouts),
            )?[0]
        };
        let mut ubo = {
            let allocator = self.allocator();
            let mut alloc = allocator.lock().unwrap();
            Buffer::new(
                &self.ctx.device,
                &mut alloc,
                CUSTOM_UBO_BYTES as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
                "custom material block",
            )?
        };
        ubo.write(0, &[0u8; CUSTOM_UBO_BYTES]);
        let buffer_info = [vk::DescriptorBufferInfo::default()
            .buffer(ubo.handle)
            .range(CUSTOM_UBO_BYTES as u64)];
        let white = self.default_textures[2].view;
        let image_infos: Vec<[vk::DescriptorImageInfo; 1]> = (0..pipeline::CUSTOM_TEXTURES)
            .map(|_| {
                [vk::DescriptorImageInfo::default()
                    .sampler(self.sampler)
                    .image_view(white)
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
            })
            .collect();
        let mut writes = vec![vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&buffer_info)];
        for (i, info) in image_infos.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32 + 1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(info),
            );
        }
        unsafe { self.ctx.device.update_descriptor_sets(&writes, &[]) };
        let m = &mut self.materials[handle.0];
        m.custom_set = set;
        m.custom_ubo = Some(ubo);
        m.custom_views = [white; pipeline::CUSTOM_TEXTURES];
        Ok(())
    }

    /// Upload a custom shader's std140 material block (set 2, binding 0).
    pub fn set_material_custom_block(&mut self, handle: MaterialHandle, bytes: &[u8]) {
        if let Err(e) = self.ensure_custom_set(handle) {
            tracing::error!("custom material set: {e:#}");
            return;
        }
        if let Some(ubo) = self.materials[handle.0].custom_ubo.as_mut() {
            ubo.write(0, &bytes[..bytes.len().min(CUSTOM_UBO_BYTES)]);
        }
    }

    /// Bind a custom shader's texture properties (set 2, bindings 1..), one
    /// entry per declared texture in binding order. `None` binds the declared
    /// default. Rebinding swaps in a fresh set (`rebind_set`), so unchanged
    /// slots are skipped.
    pub fn set_material_custom_textures(
        &mut self,
        handle: MaterialHandle,
        slots: &[(Option<TextureHandle>, CustomTextureDefault)],
    ) {
        if let Err(e) = self.ensure_custom_set(handle) {
            tracing::error!("custom material set: {e:#}");
            return;
        }
        let mut views = self.materials[handle.0].custom_views;
        for (i, (tex, default)) in slots.iter().take(pipeline::CUSTOM_TEXTURES).enumerate() {
            views[i] = match tex {
                Some(h) if h.0 < self.textures.len() => self.textures[h.0].view,
                _ => self.default_textures[match default {
                    CustomTextureDefault::White => 2,
                    CustomTextureDefault::Normal => 1,
                    CustomTextureDefault::Black => 16,
                    CustomTextureDefault::Gray => 17,
                }]
                .view,
            };
        }
        let material = &self.materials[handle.0];
        if views == material.custom_views {
            return;
        }
        let old_set = material.custom_set;
        let changed: Vec<usize> = (0..pipeline::CUSTOM_TEXTURES)
            .filter(|&i| views[i] != material.custom_views[i])
            .collect();
        // Everything not rewritten (the block UBO and the unchanged textures) is copied over.
        let keep: Vec<u32> = std::iter::once(0).chain((0..pipeline::CUSTOM_TEXTURES).filter(|i| !changed.contains(i)).map(|i| i as u32 + 1)).collect();
        let set = self.rebind_set(self.custom_pool, self.pipeline_cache.set2_layout, old_set, &keep, "wait:device_wait_idle@set_material_custom_textures");
        self.materials[handle.0].custom_set = set;
        let infos: Vec<[vk::DescriptorImageInfo; 1]> = changed
            .iter()
            .map(|&i| {
                [vk::DescriptorImageInfo::default()
                    .sampler(self.sampler)
                    .image_view(views[i])
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
            })
            .collect();
        let writes: Vec<_> = changed
            .iter()
            .zip(&infos)
            .map(|(&i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32 + 1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(info)
            })
            .collect();
        unsafe { self.ctx.device.update_descriptor_sets(&writes, &[]) };
        self.materials[handle.0].custom_views = views;
    }

    /// Per-material inverted-hull outline: linear RGBA colour and width
    /// (roughly screen-constant; 1 = thin). None disables it.
    pub fn set_material_fur(&mut self, handle: MaterialHandle, fur: Option<FurParams>) {
        self.materials[handle.0].fur = fur;
    }

    pub fn set_material_outline(&mut self, handle: MaterialHandle, outline: Option<([f32; 4], f32)>) {
        self.materials[handle.0].outline = outline;
    }

    /// Custom-shader keyword values; each distinct set selects (and lazily
    /// builds) its own specialized pipeline.
    pub fn set_material_custom_keywords(&mut self, handle: MaterialHandle, keywords: [u8; pipeline::CUSTOM_KEYWORDS]) {
        self.materials[handle.0].custom_keywords = keywords;
    }

    /// Custom-shader property values, delivered as push constants per draw.
    pub fn set_material_custom_data(&mut self, handle: MaterialHandle, data: [f32; 16]) {
        self.materials[handle.0].custom_data = data;
    }

    /// Flag a material as broken; it renders with the error swirl shader.
    pub fn set_material_error(&mut self, handle: MaterialHandle, error: bool) {
        self.materials[handle.0].error = error;
    }

    /// Set a material's render-queue priority (draw order).
    pub fn set_material_render_queue(&mut self, handle: MaterialHandle, queue: i32) {
        self.materials[handle.0].render_queue = queue;
    }

    /// Mean albedo-texture colour, for ray-traced reflection hit shading.
    pub fn set_material_albedo_mean(&mut self, handle: MaterialHandle, mean: [f32; 3]) {
        if let Some(m) = self.materials.get_mut(handle.0) {
            m.albedo_mean = mean;
        }
    }

    /// Mean ORM metalness for diffuse GI; reflections retain per-texel sampling.
    pub fn set_material_metallic_mean(&mut self, handle: MaterialHandle, mean: f32) {
        if let Some(m) = self.materials.get_mut(handle.0) {
            m.metallic_mean = mean.clamp(0.0, 1.0);
        }
    }

    pub fn material_params_mut(&mut self, handle: MaterialHandle) -> &mut MaterialParams {
        &mut self.materials[handle.0].params
    }

    pub fn set_material_features(&mut self, handle: MaterialHandle, features: MaterialFeatures) {
        let material = &mut self.materials[handle.0];
        material.features = features;
        material.features.normal_map &= material.has_normal_texture;
    }

    pub fn resize(&mut self) {
        // Nothing to do per event: the frame compares the window's current
        // size with the swapchain's (resize_gate.rs), so a burst of Resized
        // events rebuilds once at the latest size and a same-size one never.
    }

    /// Tear down all scene resources (meshes, textures, materials) so a new
    /// scene can be uploaded from scratch. Existing handles become invalid.
    pub fn reset_scene(&mut self) -> Result<()> {
        if let Some(gi)=self.gpu_gi.as_mut() {gi.radiance_cache.invalidate();}
        unsafe { self.ctx.device.wait_idle("wait:device_wait_idle@reset_scene")? };
        // Parked frees name sets of pools reset below and resources about to be torn down.
        self.run_deferred_frees(true);
        let allocator = self.allocator();
        let mut alloc = allocator.lock().unwrap();
        if let Some(mut surface) = self.flux_surface.take() {
            surface.destroy(&self.ctx.device, &mut alloc);
        }
        if let Some(mut history)=self.flux_screen_history.take() {history.destroy(&self.ctx.device,&mut alloc);}
        if let Some(rt)=&mut self.rt_reflect {rt.invalidate_reflection_history();}
        // Camera targets' colour lives in `textures` (cleared below).
        for mut t in self.camera_targets.drain(..) {
            t.depth.destroy(&self.ctx.device, &mut alloc);
            for u in &mut t.ubos {
                u.destroy(&self.ctx.device, &mut alloc);
            }
        }
        if self.camera_target_pool != vk::DescriptorPool::null() {
            unsafe { self.ctx.device.reset_descriptor_pool(self.camera_target_pool, vk::DescriptorPoolResetFlags::empty())? };
        }
        for (i, mesh) in self.meshes.iter_mut().enumerate() {
            if self.freed_meshes.contains(&i) {
                continue;
            }
            mesh.vertex_buffer.destroy(&self.ctx.device, &mut alloc);
            mesh.index_buffer.destroy(&self.ctx.device, &mut alloc);
        }
        self.meshes.clear();
        self.freed_meshes.clear();
        for (i, texture) in self.textures.iter_mut().enumerate() {
            if self.freed_textures.contains(&i) {
                continue;
            }
            texture.destroy(&self.ctx.device, &mut alloc);
        }
        self.textures.clear();
        self.freed_textures.clear();
        self.tex_pending.clear();
        for (_, (view, _)) in self.ripple_mips.drain() {
            unsafe { self.ctx.device.destroy_image_view(view, None) };
        }
        self.tex_view_cache.clear();
        for (i, material) in self.materials.iter_mut().enumerate() {
            if self.freed_materials.contains(&i) {
                continue;
            }
            material.fx_ubo.destroy(&self.ctx.device, &mut alloc);
            if let Some(mut ubo) = material.custom_ubo.take() {
                ubo.destroy(&self.ctx.device, &mut alloc);
            }
        }
        self.materials.clear();
        self.freed_materials.clear();
        // Impostor atlases belong to the scene being torn down (their handles die with it).
        if let Some(gpu) = self.impostor_gpu.as_mut() {
            let pool = gpu.pool();
            for mut a in gpu.take_all() {
                unsafe { impostor::free_atlas(&self.ctx.device, &mut alloc, pool, &mut a) };
            }
        }
        if let Some(skinner) = self.skinner.as_mut() {
            skinner.clear(&self.ctx.device, &mut alloc);
        }
        drop(alloc);
        self.world_egui.clear();
        self.world_egui_pending_free.clear();
        self.world_egui_free_slots.iter_mut().for_each(Vec::clear);
        unsafe {
            self.ctx
                .device
                .reset_descriptor_pool(self.material_pool, vk::DescriptorPoolResetFlags::empty())?;
            self.ctx
                .device
                .reset_descriptor_pool(self.custom_pool, vk::DescriptorPoolResetFlags::empty())?;
            // `skybox_pool` is deliberately NOT reset: its sets survive the
            // scene reload (they reference only renderer-global resources),
            // and resetting it would leave `skybox_set` a stale handle - the
            // next skybox draw would crash the driver.
        }
        Ok(())
    }

    fn recreate_swapchain(&mut self) -> Result<()> {
        let size = self.window.inner_size();
        crate::hitch::event("swapchain_recreate", "", ((size.width as u64) << 16) | size.height as u64);
        unsafe { self.ctx.device.wait_idle("wait:device_wait_idle@recreate_swapchain")? };
        let allocator = self.allocator();
        let mut alloc = allocator.lock().unwrap();
        self.swapchain.destroy(&self.ctx.device);
        self.depth.destroy(&self.ctx.device, &mut alloc);
        self.sgi.destroy(&self.ctx.device, &mut alloc);
        self.swapchain = Swapchain::new(&self.ctx, size.width, size.height, self.vsync)?;
        self.resize_gate.rebuilt((size.width, size.height));
        // FSR3: rebuild the upscaler context + full-res outputs when active. A
        // backend failure (driver/feature mismatch) falls back to native
        // resolution for the session instead of failing the frame: the render
        // targets below are then built at the full size.
        if let Some(mut f) = self.fsr3.take() {
            f.destroy();
        }
        if let Some(mut f) = self.fsr3_fg.take() {
            f.destroy();
        }
        for mut t in self.upscale_outputs.drain(..) {
            t.destroy(&self.ctx.device, &mut alloc);
        }
        for mut t in self.fg_present.drain(..) {
            t.destroy(&self.ctx.device, &mut alloc);
        }
        for mut t in self.fg_output.drain(..) {
            t.destroy(&self.ctx.device, &mut alloc);
        }
        if self.upscaler == Upscaler::Fsr3 {
            let full = [self.swapchain.extent.width, self.swapchain.extent.height];
            let created: Result<()> = (|| {
                if self.frame_gen_active() {
                    // Frame generation: the combined context + display-space present/
                    // output targets.
                    self.fsr3_fg = Some(fsr3::Fsr3FrameGen::new(
                        &self.ctx.device,
                        self.ctx.physical_device,
                        &self.ctx.instance,
                        full,
                        full,
                        full,
                        self.swapchain.format.format,
                        fsr3::frame_gen_flags(true, false, false),
                    )?);
                    self.fg_present = (0..FRAMES_IN_FLIGHT)
                        .map(|_| UpscaleOutput::new(&self.ctx.device, &mut alloc, self.swapchain.extent, self.swapchain.format.format))
                        .collect::<Result<Vec<_>>>()?;
                    self.fg_output = (0..FRAMES_IN_FLIGHT)
                        .map(|_| UpscaleOutput::new(&self.ctx.device, &mut alloc, self.swapchain.extent, self.swapchain.format.format))
                        .collect::<Result<Vec<_>>>()?;
                } else {
                    self.fsr3 = Some(fsr3::Fsr3::new(
                        &self.ctx.device,
                        self.ctx.physical_device,
                        &self.ctx.instance,
                        full,
                        full,
                        fsr3::upscaler_flags(true, false, false),
                    )?);
                }
                // The full-res HDR upscale output + its post-pass descriptor sets are
                // shared by the standalone and frame-gen paths.
                self.upscale_outputs = (0..FRAMES_IN_FLIGHT)
                    .map(|_| UpscaleOutput::new(&self.ctx.device, &mut alloc, self.swapchain.extent, post::HDR_FORMAT))
                    .collect::<Result<Vec<_>>>()?;
                // Descriptor sets binding each output for the post pass.
                if self.upscale_pool == vk::DescriptorPool::null() {
                    self.upscale_pool = unsafe {
                        self.ctx.device.create_descriptor_pool(
                            &vk::DescriptorPoolCreateInfo::default()
                                .max_sets(FRAMES_IN_FLIGHT as u32)
                                .pool_sizes(&[vk::DescriptorPoolSize {
                                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                                    descriptor_count: FRAMES_IN_FLIGHT as u32,
                                }]),
                            None,
                        )?
                    };
                }
                let layouts = vec![self.post_pass.set_layout(); FRAMES_IN_FLIGHT];
                unsafe {
                    self.ctx
                        .device
                        .reset_descriptor_pool(self.upscale_pool, vk::DescriptorPoolResetFlags::empty())?;
                }
                self.upscale_sets = unsafe {
                    self.ctx.device.allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(self.upscale_pool)
                            .set_layouts(&layouts),
                    )?
                };
                for (i, set) in self.upscale_sets.iter().enumerate() {
                    let info = [vk::DescriptorImageInfo::default()
                        .sampler(self.post_pass.sampler())
                        .image_view(self.upscale_outputs[i].view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                    let write = vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&info);
                    unsafe { self.ctx.device.update_descriptor_sets(&[write], &[]) };
                }
                Ok(())
            })();
            if let Err(e) = created {
                // The context could not be created on this device (backend API
                // error): render at native resolution instead of killing the
                // session. The user can re-select FSR3 (or a quality preset) to
                // retry, e.g. after a driver update.
                tracing::error!("FSR3 unavailable: {e:#}; falling back to native resolution");
                self.upscaler = Upscaler::Off;
                self.frame_gen = false;
                self.render_scale = 1.0;
            }
        }
        // The game-path targets (depth, GI, HDR, SSR) render at `render_scale` of
        // the swapchain; the post pass upscales back to the full swapchain.
        let render_extent = render_extent_for(
            [self.swapchain.extent.width, self.swapchain.extent.height],
            self.render_scale,
        );
        let render_extent_vk = vk::Extent2D {
            width: render_extent[0],
            height: render_extent[1],
        };
        self.depth = DepthTarget::new(&self.ctx, &mut alloc, render_extent_vk)?;
        self.sgi = ScreenGiTargets::new(&self.ctx, &mut alloc, render_extent_vk)?;
        self.post_pass.resize(&self.ctx.device, &mut alloc, render_extent_vk)?;
        // Recreate the deferred-SSR targets at the new size.
        for mut t in self.ssr_targets.drain(..) {
            t.destroy(&self.ctx.device, &mut alloc);
        }
        self.ssr_targets = (0..FRAMES_IN_FLIGHT)
            .map(|_| {
                self.ssr_resolve.create_target(
                    &self.ctx.device,
                    &mut alloc,
                    self.post_pass.set_layout(),
                    render_extent_vk,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        // Recreate the motion-vector attachments at the new (render) size.
        for mut t in self.velocity_targets.drain(..) {
            t.destroy(&self.ctx.device, &mut alloc);
        }
        self.velocity_targets = (0..FRAMES_IN_FLIGHT)
            .map(|_| VelocityTarget::new(&self.ctx.device, &mut alloc, render_extent_vk))
            .collect::<Result<Vec<_>>>()?;
        // Reactive masks likewise.
        for mut t in self.reactive_targets.drain(..) {
            t.destroy(&self.ctx.device, &mut alloc);
        }
        self.reactive_targets = (0..FRAMES_IN_FLIGHT)
            .map(|_| ReactiveTarget::new(&self.ctx.device, &mut alloc, render_extent_vk))
            .collect::<Result<Vec<_>>>()?;
        self.fsr3_reset = true;
        if let Some(overlay) = self.vr_overlay.as_mut() {
            overlay.resize(&self.ctx.device, &mut alloc, self.swapchain.extent)?;
        }
        self.sgi_history_valid = false; // recreated target has no history yet
        self.sgi_parity = 0;
        self.sgi_last_view_proj = None;
        drop(alloc);
        // Re-point the screen-GI sampler (binding 4) at the recreated target.
        for &set in &self.frame_sets {
            let info = [vk::DescriptorImageInfo::default()
                .sampler(self.lightmap_sampler)
                .image_view(self.sgi.gi_view[0])
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&info);
            unsafe { self.ctx.device.update_descriptor_sets(&[write], &[]) };
        }
        Ok(())
    }

    pub fn render(&mut self, input: &FrameInput<'_>) -> Result<()> {
        self.render_frame(input, true)
    }

    /// This frame's shared work for the XR eyes without the window: uploads,
    /// skinning, shadows, reflection captures, render-texture cameras and a
    /// fresh frame UBO, but no swapchain acquire, window scene pass or present.
    /// Call instead of [`Renderer::render`] when the headset is the only view
    /// (a standalone headset, whose window is never shown), then render the
    /// eyes. `input.camera` should enclose both eyes (shadow cascades fit it).
    pub fn render_xr_frame(&mut self, input: &FrameInput<'_>) -> Result<()> {
        self.render_frame(input, false)
    }

    fn render_frame(&mut self, input: &FrameInput<'_>, window: bool) -> Result<()> {
        let cascade_count = CascadeCfg::resolve(input, self.tier.shadow_cascades(), self.shadow_atlas.dim).count;
        let demand = shadow_layer_demand(input, cascade_count);
        if demand > self.shadow_atlas.layers {
            self.resize_shadow_atlas(self.shadow_requested_dim, demand)?;
        }
        crate::hitch::phase("render_prep");
        // An offscreen viewport owns its own depth/GI targets. Never also run
        // the swapchain scene with its extent: the two would share temporal
        // state and descriptor sets but write differently sized images.
        let render_game = input.render_viewport && input.viewport_extent.is_none();
        self.frame_serial = self.frame_serial.wrapping_add(1);
        let size = self.window.inner_size();
        // With VR running the eyes need this frame's lighting even when the
        // window can't show it (minimised, no surface, out of date).
        let xr = self.xr_render.is_some();
        let presentable = window && size.width > 0 && size.height > 0 && self.has_surface();
        if window && !presentable && !xr {
            return Ok(());
        }
        let rebuild = self.resize_gate.begin_frame((size.width, size.height));
        if presentable && rebuild {
            self.recreate_swapchain()?;
        }
        self.free_retired_viewport_targets(false);
        self.run_deferred_frees(false);

        // Editor RTT: (re)create the offscreen viewport target at the requested
        // dock-rect size, before anything this frame reads its history. None (game) leaves it absent → renders to swapchain.
        if let Some([vw, vh]) = input.viewport_extent {
            // While a border is dragged keep the old target (shown stretched)
            // and recreate once the size settles or the pointer is released.
            let [vw, vh] = self.viewport_resize.resolve(
                self.started.elapsed().as_secs_f64(),
                [vw.max(1), vh.max(1)],
                // A swapchain rebuild this frame means the window is being
                // resized (compositor animation): hold the viewport target.
                self.viewport_resize_hold || self.resize_gate.rebuilt_this_frame(),
            );
            let need = self
                .viewport_target
                .as_ref()
                .is_none_or(|t| t.extent.width != vw || t.extent.height != vh);
            if need && let Some(allocator) = self.allocator.clone() {
                // No device_wait_idle: the old target (and its SSR target) are
                // retired below and destroyed `FRAMES_IN_FLIGHT + 1` rendered
                // frames later (`free_retired_viewport_targets`). Every
                // submission that touched them was made before this call, and
                // the per-slot fence wait further down retires them in queue
                // order within `FRAMES_IN_FLIGHT` frames, so the extra frame is
                // slack. Nothing else holds their views: each target owns its
                // descriptor sets and pools, and the egui user texture is
                // removed only when the target is finally freed. New images
                // start UNDEFINED and the frame's barriers transition them.
                // Hazard: a resize drag retires one target per frame, so up to
                // `FRAMES_IN_FLIGHT + 1` old targets (full-size HDR + depth +
                // GI images) live at once.
                let mut alloc = allocator.lock().unwrap();
                tracing::debug!(vw, vh, "viewport target resized");
                crate::hitch::event("viewport_resize", "", ((vw as u64) << 16) | vh as u64);
                if let Some(old) = self.viewport_target.take() {
                    self.retired_viewport_targets.push((old, FRAMES_IN_FLIGHT + 1));
                }
                if let Some(old) = self.viewport_ssr.take() {
                    self.retired_viewport_ssr.push((old, FRAMES_IN_FLIGHT + 1));
                }
                self.viewport_ssr = Some(self.ssr_resolve.create_target(
                    &self.ctx.device,
                    &mut alloc,
                    self.post_pass.set_layout(),
                    vk::Extent2D { width: vw, height: vh },
                )?);
                let fmt = self.swapchain.format.format;
                let post_layout = self.post_pass.set_layout();
                let post_sampler = self.post_pass.sampler();
                let mut target = ViewportTarget::new(
                    &self.ctx,
                    &mut alloc,
                    fmt,
                    self.sampler,
                    post_layout,
                    post_sampler,
                    vk::Extent2D { width: vw, height: vh },
                    self.egui.as_mut().unwrap(),
                )?;
                target.bloom = Some(self.post_pass.new_bloom_chain(
                    &self.ctx.device,
                    &mut alloc,
                    vk::Extent2D { width: vw, height: vh },
                )?);
                self.viewport_target = Some(target);
                // The viewport Flux trace shares the main-camera temporal state
                // (sgi_history_valid/parity/last_view_proj) but writes into the
                // viewport target's OWN sgi images, which we just recreated with
                // undefined contents. Reprojecting from that stale-but-new
                // history reads garbage (flickering colour bars on resize), so
                // reset the temporal state exactly as recreate_swapchain does.
                self.sgi_history_valid = false;
                self.sgi_parity = 0;
                self.sgi_last_view_proj = None;
            }
        }

        // Release last frame's egui texture frees BEFORE applying this frame's
        // uploads. egui recycles TextureIds, so an id freed last frame can be
        // reused by a texture in this frame's `set`; freeing first (behind a
        // full idle so no in-flight frame still binds them) destroys the old id
        // then lets `set_textures` cleanly recreate it. Frees are rare (texture
        // drops / font-atlas rebuilds), so the idle is effectively free.
        // Frees are parked per frame slot and released after that slot's fence
        // retires (below), so there is no per-frame device_wait_idle. Only a
        // `set` that reuses a still-parked id forces one idle + free here.
        if let Some(egui_draw) = &input.egui
            && egui_draw
                .textures_delta
                .set
                .iter()
                .any(|(id, _)| self.egui_free_slots.iter().any(|v| v.contains(id)))
        {
            unsafe {
                let _ = self.ctx.device.wait_idle("wait:device_wait_idle@render_frame");
            }
            let ids: Vec<egui::TextureId> =
                self.egui_free_slots.iter_mut().flat_map(std::mem::take).collect();
            self.egui
                .as_mut()
                .unwrap()
                .free_textures(&ids)
                .map_err(|e| anyhow::anyhow!("freeing egui textures: {e}"))?;
            if let Some(vr) = self.vr_egui.as_mut() {
                let _ = vr.free_textures(&ids);
            }
        }

        self.prepare_world_ui(&input.world_ui)?;
        self.ensure_text_gpu();
        if !self.sprite_input.is_empty()
            && let Err(e) = self.ensure_sprite_gpu()
        {
            tracing::warn!("sprite pass unavailable: {e:#}");
            self.sprite_input.clear();
        }

        // egui texture uploads are recorded into the frame's command buffer
        // below (`record_textures`, after `begin_command_buffer`): no per-upload
        // submit + queue_wait_idle, which drained the queue on every file-browser
        // thumbnail.

        let device = self.ctx.device.clone();
        // Screen-space GI: prefetch the depth-only pipeline (needs &mut self,
        // before the per-frame immutable borrow below) when a GDF is available.
        let sgi_active = self.gi_has_gdf() && self.tier.realtime_gi();
        let depth_pipe = if sgi_active {
            // The shadow pipeline is the color-less depth-only one (matches our
            // standalone depth prepass with no color attachment).
            Some(self.pipeline_cache.get(&device, PipelineKey::shadow())?)
        } else {
            None
        };
        let in_flight = self.frames[self.frame_index].in_flight;
        let image_available = self.frames[self.frame_index].image_available;
        let command_buffer = self.frames[self.frame_index].command_buffer;

        unsafe { device.wait_fences("wait:fence@render_frame", &[in_flight], true, u64::MAX)? };
        // This slot's sets are free again: apply the binding swaps queued while it was in flight.
        self.apply_shared_slot(self.frame_index);
        // A disabled effect must not leave a descriptor pointing at a retired
        // viewport image. Active GI/refraction replace these during recording.
        let neutral = [vk::DescriptorImageInfo::default().sampler(self.lightmap_sampler)
            .image_view(self.default_textures[3].view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let mut defaults = vec![vk::WriteDescriptorSet::default()
            .dst_set(self.frame_sets[self.frame_index]).dst_binding(4)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&neutral)];
        if self.opaque_copy.is_none() {
            defaults.extend([5, 6].map(|binding| vk::WriteDescriptorSet::default()
                .dst_set(self.frame_sets[self.frame_index]).dst_binding(binding)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&neutral)));
        }
        if let Some(preview) = &self.camera_preview {
            defaults.push(vk::WriteDescriptorSet::default().dst_set(preview.sets[self.frame_index]).dst_binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&neutral));
        }
        unsafe { self.ctx.device.update_descriptor_sets(&defaults, &[]); }

        // Offscreen readbacks recorded in this slot two frames ago have retired.
        self.complete_readbacks();

        // This slot's submission from two frames ago (and, queue order, the
        // one in between) has retired: release the egui textures it parked.
        {
            // Staging buffers / replaced textures of this slot's egui uploads.
            if let Some(e) = self.egui.as_mut() {
                let _ = e.release_retired(self.frame_index);
            }
            if let Some(vr) = self.vr_egui.as_mut() {
                let _ = vr.release_retired(self.frame_index);
            }
            let ids = std::mem::take(&mut self.egui_free_slots[self.frame_index]);
            if !ids.is_empty() {
                if let Some(e) = self.egui.as_mut() {
                    let _ = e.free_textures(&ids);
                }
                if let Some(vr) = self.vr_egui.as_mut() {
                    let _ = vr.free_textures(&ids);
                }
            }
            for (target, ids) in std::mem::take(&mut self.world_egui_free_slots[self.frame_index]) {
                if let Some(r) = self.world_egui.get_mut(&target) {
                    let _ = r.free_textures(&ids);
                }
            }
            // Staging buffers / replaced textures of this slot's world-UI uploads.
            for r in self.world_egui.values_mut() {
                let _ = r.release_retired(self.frame_index);
            }
            let incoming = std::mem::take(&mut self.world_egui_pending_free);
            self.world_egui_free_slots[self.frame_index] = incoming;
        }

        // This frame slot's submission has retired, so its GPU timestamps are
        // ready. Read them back (TS_COUNT u64 ticks) and convert to ms. This is
        // the only place GPU pass cost is attributed: the CPU-side "realtime GI"
        // timer only covers the trace's CPU prep, while the actual march runs on
        // the GPU and otherwise surfaces only as a longer fence wait above.
        if self.gpu_ts_period > 0.0 && self.gpu_ts_primed[self.frame_index] {
            // Everything while the profiler is open; otherwise only the FRAME
            // zone's begin/end pair (two queries, always on).
            let query_count = ts_readback_count(self.profiler_window.is_some() || self.zone_timing || gpu_timing_forced());
            let pool = self.frames[self.frame_index].timestamps;
            // WITH_AVAILABILITY so zones a frame didn't record (their queries were
            // reset but never written) report available=0 instead of failing the
            // whole read. Each query yields {value, available} (TsResult).
            //
            // IMPORTANT: when any query in the range is unavailable (e.g. a pass
            // that didn't run this frame), vkGetQueryPoolResults returns NOT_READY
            // *even with WITH_AVAILABILITY* — but it still writes the availability
            // flags + values for the queries that ARE ready. So NOT_READY is
            // expected and fine; only a hard error should skip processing. (WAIT
            // is NOT an option: it would hang forever on a never-written query.)
            let mut res = [TsResult::default(); frame::TS_COUNT as usize];
            let result = unsafe {
                device.get_query_pool_results(
                    pool,
                    0,
                    &mut res[..query_count as usize],
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
                )
            };
            let ok = matches!(result, Ok(()) | Err(vk::Result::NOT_READY));
            if ok {
                let period = self.gpu_ts_period as f64;
                for z in 0..(query_count as usize / 2).min(gpu_zone::COUNT) {
                    let b = res[2 * z];
                    let e = res[2 * z + 1];
                    self.gpu_zone_ms[z] = ts_zone_ms(b, e, period, self.gpu_ts_mask).unwrap_or(0.0);
                }
            }
        }

        // This frame slot's prior submission has retired: its instance
        // matrices can be overwritten.
        let allocator = self.allocator();
        self.instance_rings[self.frame_index].reset(&device, &mut allocator.lock().unwrap());
        self.gpu_cull_begin(self.frame_index);
        if let Some(rt) = self.rt_reflect.as_mut() {
            rt.begin_frame(&self.ctx, &mut allocator.lock().unwrap(), self.frame_index);
        }

        // This frame slot's prior submission has retired, so the Flux trace
        // resources it referenced (descriptor pool + host buffers, folded into
        // that submission's command buffer) are now safe to free.
        // They are kept as this slot's spares and reused by its next trace.
        if let Some(t) = self.sgi_transients[self.frame_index].take()
            && let Some(mut old) = self.sgi_spare[self.frame_index].replace(t) {
                old.destroy(&device, &mut self.allocator().lock().unwrap());
            }
        if let Some(t) = self.preview_sgi_transients[self.frame_index].take()
            && let Some(mut old) = self.preview_sgi_spare[self.frame_index].replace(t) {
                old.destroy(&device, &mut self.allocator().lock().unwrap());
            }

        crate::hitch::phase("render_record");
        let image_index = if presentable {
            match crate::hitch::acquire(|| self.swapchain.acquire(image_available)) {
                Ok((index, suboptimal)) => {
                    if suboptimal {
                        self.resize_gate.force();
                    }
                    Some(index)
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.resize_gate.force();
                    if !xr {
                        return Ok(());
                    }
                    None
                }
                Err(e) => return Err(e).context("acquiring swapchain image"),
            }
        } else {
            None
        };
        let window = image_index.is_some();

        unsafe { device.reset_fences(&[in_flight])? };
        self.write_frame_ripples(input);

        // Per-frame uniforms.
        //
        // Pack the scene lights into the std140 array. When the scene has no
        // light objects, synthesize one directional light from the key-light
        // fallback so existing scenes look unchanged. The key-light fields
        // (light_dir/light_color) stay populated for custom-shader macros.
        let mut lights = [GpuLight::default(); MAX_LIGHTS];
        let mut count = 0usize;
        for l in input.lights.iter().take(MAX_LIGHTS) {
            lights[count] = gpu_light(l);
            count += 1;
        }
        if count == 0 {
            // No scene lights: keep the legacy single directional so existing
            // scenes render identically.
            lights[0] = gpu_light(&LightInstance {
                kind: LightKind::Directional,
                position: Vec3::ZERO,
                direction: input.light.direction,
                color: input.light.color,
                intensity: input.light.intensity,
                range: 0.0,
                spot_inner_deg: 0.0,
                spot_outer_deg: 0.0,
                cast_shadows: false,
                soft_shadows: true, radius: 0.0,
                shadow_bias: 0.0,
                baked: false,
            });
            count = 1;
        }

        // Plan shadow maps (allocates atlas layers, patches lights[*].spot)
        // before building the UBOs that carry the shadow matrices.
        let (shadow_views, shadow_vp, cascade_splits) = plan_shadows(input, &mut lights, count, CascadeCfg::resolve(input, self.tier.shadow_cascades(), self.shadow_atlas.dim));
        self.view_cull.shadow_view_projs.clear();
        self.view_cull.shadow_view_projs.extend(shadow_views.iter().map(|v| v.view_proj));
        self.view_cull.secondary = !input.planar.is_empty()
            || !input.camera_targets.is_empty()
            || input.camera_preview.is_some()
            || self.refl_job.as_ref().is_some_and(|j| j.dynamic)
            || self.recapture_reflection.is_some_and(|r| r.dynamic);
        // FluxVoxel moving emitters (kind 3): unshadowed, indirect-only, after
        // the scene lights so the shadow plan's indices stay put.
        for e in input.voxel_emitters {
            if count >= MAX_LIGHTS {
                break;
            }
            lights[count] = GpuLight {
                pos_kind: [e.position.x, e.position.y, e.position.z, 3.0],
                dir_range: [0.0, -1.0, 0.0, e.range],
                color: [e.color[0], e.color[1], e.color[2], -1.0],
                spot: [1.0, -1.0, 0.0, 0.0],
            };
            count += 1;
        }
        // Area lights (two entries each) after the emitters; the nearest ones only.
        self.pack_area_lights(input.camera.position, &mut lights, &mut count);
        // Directional lights reach everything (range < 0).
        self.cull_lights.clear();
        self.cull_lights.extend(lights[..count].iter().map(|l| {
            let range = if l.pos_kind[3] < 0.5 { -1.0 } else { l.dir_range[3] };
            [l.pos_kind[0], l.pos_kind[1], l.pos_kind[2], range]
        }));

        // Screen size for the frame UBO = the render target the main camera draws
        // into: the viewport rect under editor RTT, else the swapchain. The Flux
        // upsample samples by gl_FragCoord / this size, so it must match or the GI
        // lands in the wrong place.
        let main_extent = if input.viewport_extent.is_some() {
            self.viewport_target.as_ref().map(|t| [t.extent.width, t.extent.height])
        } else { None }.unwrap_or([self.swapchain.extent.width, self.swapchain.extent.height]);
        // Reduced-resolution render extent for the upscaler (equals `main_extent`
        // at native scale). Scene, depth, SSR and the post pass size off this.
        self.render_extent = render_extent_for(main_extent, self.render_scale);
        // Per-frame sub-pixel jitter for the temporal upscaler (a fresh sample
        // every frame; unused — and harmless — at native resolution).
        self.aa_prepare(input, window);
        self.jitter = if self.upscaler != Upscaler::Off {
            halton23(self.frame_serial as u32)
        } else {
            self.aa_jitter().unwrap_or([0.5; 2])
        };
        let mut ubo = frame_ubo(
            &input.camera,
            input,
            lights,
            count,
            shadow_vp,
            cascade_splits,
            &self.frame_probe_volumes,
            self.jitter,
            [self.render_extent[0] as f32, self.render_extent[1] as f32],
            [self.render_extent[0] as f32, self.render_extent[1] as f32],
            self.prev_view_proj,
        );
        ubo.misc[2] *= self.shadow_requested_dim as f32 / self.shadow_atlas.dim as f32;
        // This frame's (jittered) view-projection becomes next frame's
        // reprojection source for the motion-vector pass.
        self.prev_view_proj = ubo.view_proj;
        // debug.z = screen-GI active, so the forward shader samples the gather
        // instead of the coarse world-probe grid.
        // 2 = mixed lighting (the gather adds on the bake).
        ubo.debug[2] = match (sgi_active, self.flux_settings.mixed_lighting) {
            (false, _) => 0.0,
            (true, false) => 1.0,
            (true, true) => 2.0,
        };
        // debug.w = HDR output: both the game swapchain path AND the editor
        // viewport render linear HDR (a post pass tonemaps + blooms). The editor
        // swapchain pass is egui-only; the camera preview keeps inline tonemap via
        // its own UBO. So the main frame UBO is always HDR.
        ubo.debug[3] = 1.0;
        // SSR rides on the Flux depth prepass, so it only runs while screen-GI is
        // active. The per-pass binding writes (below / in record_viewport_scene)
        // point bindings 5/6 at the active camera's depth + colour history.
        let fs = self.flux_settings.for_view(input.postfx.flux);
        // Reflections are UNCOUPLED from the GI backend: they run purely from their
        // own settings (the game path always renders the depth prepass + G-buffer),
        // so changing or disabling GI never turns reflections off. 1 = SSR, 2 = RT.
        let rt_on = fs.reflection_mode == 2 && self.rt_reflect.is_some() && self.tier.reflections();
        let ssr_on = fs.reflection_mode == 1 && fs.ssr_enabled && self.tier.reflections();
        // Volumetric fog runs in the same resolve pass off the depth prepass.
        let fog_on = input.fog.is_some_and(|f| f.density > 0.0);
        ubo.ssr = if ssr_on {
            [1.0, fs.ssr_intensity, fs.ssr_max_distance, fs.ssr_roughness_cutoff]
        } else {
            [0.0; 4]
        };
        if let Some(p) = input.reflection_probe {
            ubo.refl_center = [p.capture[0], p.capture[1], p.capture[2], p.intensity.max(0.0)];
            ubo.refl_extents = [
                p.half_extents[0],
                p.half_extents[1],
                p.half_extents[2],
                if p.box_projection { 1.0 } else { 0.0 },
            ];
            ubo.refl_box = [p.center[0], p.center[1], p.center[2], 0.0];
            ubo.refl_blend[0] = p.blend_distance.max(0.0);
        }
        if let Some(p) = input.reflection_probe_b {
            ubo.refl2_center = [p.capture[0], p.capture[1], p.capture[2], p.intensity.max(0.0)];
            ubo.refl2_extents = [
                p.half_extents[0],
                p.half_extents[1],
                p.half_extents[2],
                if p.box_projection { 1.0 } else { 0.0 },
            ];
            ubo.refl2_box = [p.center[0], p.center[1], p.center[2], 0.0];
            ubo.refl_blend[1] = p.blend_distance.max(0.0);
            ubo.refl_blend[2] = input.reflection_primary.min(1) as f32;
        }
        if let Some(p) = input.reflection_probes.get(2).copied().flatten() {
            ubo.refl3_center = [p.capture[0], p.capture[1], p.capture[2], p.intensity.max(0.0)];
            ubo.refl3_extents = [
                p.half_extents[0],
                p.half_extents[1],
                p.half_extents[2],
                if p.box_projection { 1.0 } else { 0.0 },
            ];
            ubo.refl3_box = [p.center[0], p.center[1], p.center[2], 0.0];
            ubo.refl_blend2[0] = p.blend_distance.max(0.0);
        }
        if let Some(p) = input.reflection_probes.get(3).copied().flatten() {
            ubo.refl4_center = [p.capture[0], p.capture[1], p.capture[2], p.intensity.max(0.0)];
            ubo.refl4_extents = [
                p.half_extents[0],
                p.half_extents[1],
                p.half_extents[2],
                if p.box_projection { 1.0 } else { 0.0 },
            ];
            ubo.refl4_box = [p.center[0], p.center[1], p.center[2], 0.0];
            ubo.refl_blend2[1] = p.blend_distance.max(0.0);
        }
        if let Some(f) = input.fog {
            ubo.fog_color = [f.color[0], f.color[1], f.color[2], f.density.max(0.0)];
            ubo.fog_params = [f.height_falloff.max(0.0), f.height_ref, f.start_distance.max(0.0), 0.0];
        }
        // fog_params.w doubles as the FluxVoxel specular-from-volume flag (set
        // unconditionally so it's independent of whether fog is enabled).
        ubo.fog_params[3] = if input.voxel_specular { 1.0 } else { 0.0 };
        // Planar reflections: size the RT array to the main target and flag the
        // main UBO with the plane count (refl_box.w > 0 = planar layers valid
        // for THIS pass; the capture/preview/VR UBOs leave it 0 and fall back
        // to the probe).
        let planar_n = if input.planar.is_empty() || !window {
            0
        } else {
            match self.ensure_planar_targets(main_extent) {
                Ok(()) => input.planar.len().min(MAX_PLANAR),
                Err(e) => {
                    tracing::warn!("planar reflection targets: {e:#}");
                    0
                }
            }
        };
        ubo.refl_box[3] = planar_n as f32;
        if planar_n > 0 {
            let view = self.planar_targets.as_ref().unwrap().array_view;
            let info = [vk::DescriptorImageInfo::default()
                .sampler(self.lightmap_sampler)
                .image_view(view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(self.frame_sets[self.frame_index])
                .dst_binding(8)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&info);
            unsafe { self.ctx.device.update_descriptor_sets(&[write], &[]) };
        }
        // Water refraction: when water is drawn the main pass copies its opaque
        // colour + depth before the transparent draws (bindings 5/6).
        let has_water = input
            .draws
            .iter()
            .any(|d| self.materials.get(d.material.0).is_some_and(|m| (m.features.water || m.features.grab) && !m.error));
        self.water_copy = window
            && has_water
            && match self.ensure_opaque_copy(main_extent) {
                Ok(()) => true,
                Err(e) => {
                    tracing::warn!("water opaque copy targets: {e:#}");
                    false
                }
            };
        ubo.water[1] = if self.water_copy { 1.0 } else { 0.0 };
        // Rough transmission reads the copy's mip chain: build it only when a
        // visible draw uses a transmission material.
        self.opaque_mips = self.water_copy
            && input.draws.iter().any(|d| {
                self.materials.get(d.material.0).is_some_and(|m| m.params.transmission.transmission > 0.0 && !m.error)
            });
        // Screen-space AO for this frame's camera pass (game or viewport):
        // refl_blend2.w gates the forward pass's binding-13 read (ao.rs).
        ubo.refl_blend2[3] = self.ao_prepare(input, self.frame_sets[self.frame_index], Mat4::from_cols_array_2d(&ubo.proj), window);
        self.frame_ubos[self.frame_index].write(0, bytemuck::bytes_of(&ubo));
        // Keep this frame's lights/shadows/probes so the VR eyes can reuse them
        // (only the camera matrices differ per eye).
        self.last_frame_ubo = Some(ubo);

        // Decide whether to trace the screen-GI gather this frame, then point
        // the forward sampler (binding 4) at the image that will hold the latest
        // result. The actual depth prepass + GDF trace is recorded into the main
        // frame command buffer below (no separate submit/fence stall).
        let mut sgi_record: Option<(vk::Pipeline, bool, usize, glam::Mat4, glam::Vec3, f32)> = None;
        // Skip the viewport-camera Flux trace when the viewport is hidden (e.g.
        // only the Camera tab is shown) — nothing samples its result.
        if let Some(dp) = depth_pipe.filter(|_| render_game && window) {
            // Realtime lighting must keep tracing with a stationary camera:
            // lights, occluders, materials and animated geometry can still move.
            // Both views accumulate continuously with surface-validated history.
            let view_proj = Mat4::from_cols_array_2d(&ubo.view_proj);

            {
                // New seed every trace so each frame samples different directions
                // and temporal accumulation can converge the noise.
                self.sgi_frame = self.sgi_frame.wrapping_add(1);
                let hist_valid = self.sgi_history_valid;
                let cur = self.sgi_parity & 1; // ping-pong slot written this frame
                // Capture the previous camera before advancing it, so the gather
                // reprojects history from where the surface actually was last
                // frame (advancing first would make reprojection identity and smear).
                let prev_vp = self.sgi_last_view_proj.unwrap_or(view_proj);
                let prev_cam = if self.sgi_last_view_proj.is_some() {
                    self.sgi_last_cam
                } else {
                    input.camera.position
                };
                // Validate history against world position and normal in the
                // gather, then use the same blend in every view. Camera motion
                // must not change the estimator's brightness or noise level.
                let s = self.flux_settings.smoothing.clamp(0.0, 1.0);
                let alpha = if hist_valid { 0.08 - 0.04 * s } else { 1.0 };
                sgi_record = Some((dp, hist_valid, cur, prev_vp, prev_cam, alpha));
                // Advance temporal state for next frame; the swap makes this
                // frame's output the next frame's history.
                self.sgi_history_valid = true;
                self.sgi_last_view_proj = Some(view_proj);
                self.sgi_last_cam = input.camera.position;
                self.sgi_parity ^= 1;
            }
            // Forward samples the INTEGRATED full-res GI (flux_integrate output:
            // denoised + per-pixel-normal SH resolve), which persists across skip
            // frames; the raw ping-pong still feeds temporal.
            let _ = (self.sgi_parity ^ 1) & 1;
            let cur_view = self.sgi.temporal((self.sgi_parity^1)&1).current_views[0];
            let info = [vk::DescriptorImageInfo::default()
                .sampler(self.lightmap_sampler)
                .image_view(cur_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(self.frame_sets[self.frame_index])
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&info);
            unsafe { device.update_descriptor_sets(&[write], &[]) };
        }

        // Lazily create the offscreen camera target the first time the Camera
        // tab requests it, then write its view UBO for this frame.
        let render_camera_preview = input.camera_preview.is_some();
        // A model-only preview request takes the model-preview target over `input.model_preview`.
        let model_only = self.model_only.take();
        self.prepare_offscreen(model_only.as_ref())?;
        let render_model_preview = input.model_preview.is_some() && model_only.is_none();
        for pass in 0..2 {
            let (camera, extent) = match (pass, &input.camera_preview, &input.model_preview) {
                (0, Some(c), _) => (c, vk::Extent2D { width: 1280, height: 720 }),
                (1, _, Some(m)) if model_only.is_none() => (&m.camera, vk::Extent2D { width: m.size[0].clamp(64, 2048), height: m.size[1].clamp(64, 2048) }),
                _ => continue,
            };
            if pass == 1 {
                // The model preview reuses the camera-preview code through the main slot.
                if self.model_preview.as_ref().is_some_and(|p| p.extent != extent)
                    && let Some(old) = self.model_preview.take()
                {
                    self.retired_model_previews.push((old, FRAMES_IN_FLIGHT + 1));
                }
                std::mem::swap(&mut self.camera_preview, &mut self.model_preview);
            }
            let res: Result<()> = (|| {
                if self.camera_preview.is_none() {
                    let allocator = self.allocator.clone();
                    if let Some(allocator) = allocator {
                        let preview = CameraPreview::new(
                            &self.ctx,
                            &mut allocator.lock().unwrap(),
                            self.pipeline_cache.set0_layout,
                            self.swapchain.format.format,
                            self.sampler,
                            self.shadow_atlas.array_view,
                            self.shadow_atlas.sampler,
                            self.probe_buffer.handle,
                            self.lightmaps.view,
                            self.lightmap_sampler,
                            self.reflection_env.view,
                            self.reflection_env_b.view,
                            self.reflection_env_c.view,
                            self.reflection_env_d.view,
                            self.ripple_dummy.view,
                            self.default_textures[3].view,
                            self.default_textures[ao::AO_DEFAULT_TEXTURE].view,
                            self.post_pass.set_layout(),
                            self.post_pass.sampler(),
                            self.egui.as_mut().unwrap(),
                            extent,
                        )?;
                        let mut preview = preview;
                        preview.bloom = Some(self.post_pass.new_bloom_chain(
                            &self.ctx.device,
                            &mut allocator.lock().unwrap(),
                            preview.extent,
                        )?);
                        self.camera_preview = Some(preview);
                    }
                }
                if let Some(preview) = &mut self.camera_preview {
                    let mut cam_ubo = frame_ubo(
                        camera,
                        input,
                        lights,
                        count,
                        shadow_vp,
                        cascade_splits,
                        &self.frame_probe_volumes,
                        [0.5; 2],
                        [1.0, 1.0],
                        [preview.extent.width as f32, preview.extent.height as f32],
                        glam::Mat4::IDENTITY.to_cols_array_2d(),
                    );
                    cam_ubo.misc[2] *= self.shadow_requested_dim as f32 / self.shadow_atlas.dim as f32;
                    // The preview runs its own Flux trace when a GDF exists, so flag
                    // its shader to sample the screen-GI (binding 4) like the viewport.
                    cam_ubo.debug[2] = match (sgi_active && pass == 0, self.flux_settings.mixed_lighting) {
                        (false, _) => 0.0,
                        (true, false) => 1.0,
                        (true, true) => 2.0,
                    };
                    // Linear HDR out; the preview's post pass tonemaps.
                    cam_ubo.debug[3] = 1.0;
                    preview.ubos[self.frame_index].write(0, bytemuck::bytes_of(&cam_ubo));
                }
            Ok(())
            })();
            if pass == 1 {
                std::mem::swap(&mut self.camera_preview, &mut self.model_preview);
            }
            res?;
        }
        // The window closed: retire the target (freed once the GPU is done with it).
        if input.model_preview.is_none()
            && model_only.is_none()
            && let Some(old) = self.model_preview.take()
        {
            self.retired_model_previews.push((old, FRAMES_IN_FLIGHT + 1));
        }
        self.free_retired_model_previews(false);


        let cb = command_buffer;
        let target = image_index.map(|i| {
            let i = i as usize;
            (self.swapchain.images[i], self.swapchain.views[i], self.swapchain.render_finished[i])
        });
        let extent = self.swapchain.extent;
        // The scene + its dependent resolves (SSR/RT/underwater/Flux) render at
        // the reduced extent; the post pass + present stay at `extent` (full).
        let render_extent_vk = vk::Extent2D {
            width: self.render_extent[0],
            height: self.render_extent[1],
        };
        let frame_set = self.frame_sets[self.frame_index];

        let order = self.draw_order(input, input.camera.position);

        // Precomputed scene reflection capture: render the scene into the env
        // cube from the requested center, ONE face per frame (plus a final
        // mip-gen step), so the 6-face render + first-time pipeline compile never
        // stall a frame. A new request waits until the current job finishes.
        // Realtime recaptures are paced by the probe's update rate and either
        // spread over 7 frames or rendered whole; bake requests start at once.
        let paced = self.recapture_reflection.is_some_and(|r| {
            r.static_only || self.refl_last_finish.is_none_or(|t| t.elapsed().as_secs_f32() >= r.min_interval)
        });
        if self.refl_job.is_none()
            && paced
            && let Some(r) = self.recapture_reflection.take()
        {
            match self.begin_refl_capture(r.center, r.size, r.static_only, r.slot, r.sliced, r.dynamic) {
                Ok(job) => self.refl_job = Some(job),
                Err(e) => tracing::warn!("reflection capture begin failed: {e:#}"),
            }
        }
        // GPU timestamp zones only while the profiler window is open, so a closed
        // profiler adds zero timestamp/readback overhead per frame.
        self.ts_active = self.gpu_ts_period > 0.0 && (self.profiler_window.is_some() || self.zone_timing || gpu_timing_forced());

        unsafe {
            device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        self.flush_probe_upload(cb);
        self.cluster_record_uploads(cb);
        self.ctx.checkpoint(cb, c"texture uploads");
        self.flush_texture_updates(cb);
        // New/changed egui textures, ahead of every pass that samples them. The
        // VR-UI egui renderer is kept in lockstep so it always has the font
        // atlas etc. (it may start mid-run when VR is entered); its draw is a
        // later submission, ordered after these copies on the same queue.
        if let Some(egui_draw) = &input.egui
            && !egui_draw.textures_delta.set.is_empty()
        {
            crate::hitch::count("egui_tex_upload", egui_draw.textures_delta.set.len() as u32);
            self.egui
                .as_mut()
                .unwrap()
                .record_textures(cb, self.frame_index, &egui_draw.textures_delta.set)
                .map_err(|e| anyhow::anyhow!("uploading egui textures: {e}"))?;
            if let Some(vr) = self.vr_egui.as_mut() {
                let _ = vr.record_textures(cb, self.frame_index, &egui_draw.textures_delta.set);
            }
        }
        // World-space UI textures (same lifetime rules as the egui uploads above:
        // staging parked on this slot, released after its fence wait).
        for (target, deltas) in std::mem::take(&mut self.world_egui_uploads) {
            if let Some(r) = self.world_egui.get_mut(&target)
                && let Err(e) = r.record_textures(cb, self.frame_index, &deltas)
            {
                tracing::warn!("world UI textures: {e}");
            }
        }
        self.record_water_sim(cb);
        self.record_area_atlas(&device, cb);
        if let Some(c) = self.opaque_copy.as_mut()
            && c.fresh
        {
            c.fresh = false;
            for (image, aspect, levels) in [(c.color, vk::ImageAspectFlags::COLOR, c.mips), (c.depth, depth_aspect(), 1)] {
                image_barrier_mips(
                    &device, cb, image, aspect, 0, levels,
                    vk::ImageLayout::UNDEFINED, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty(),
                    vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    vk::AccessFlags2::SHADER_SAMPLED_READ,
                );
            }
        }
        unsafe {

            // GPU timestamps: reset the pool (required before reuse) and stamp
            // the frame start. The pool is read back next time this slot's fence
            // has retired (see above).
            if self.ts_active {
                let ts_pool = self.frames[self.frame_index].timestamps;
                device.cmd_reset_query_pool(cb, ts_pool, 0, frame::TS_COUNT);
            } else if self.gpu_ts_period > 0.0 {
                let ts_pool = self.frames[self.frame_index].timestamps;
                device.cmd_reset_query_pool(cb, ts_pool, 0, 2);
            }
            self.gpu_zone_begin(cb, gpu_zone::FRAME);

            // World-space UI targets, before any pass samples them.
            self.record_world_ui(&device, cb, &input.world_ui);
            // Text: pack this frame's instances and copy new atlas rects.
            if let Some(t) = self.text_gpu.as_mut() {
                t.prepare(&device, cb, self.frame_index);
            }
            // Sprites: gather, sort and batch, then write this frame's instances.
            if let Some(s) = self.sprite_gpu.as_mut() {
                s.prepare(&self.sprite_input, &mut self.sprite_pool, self.frame_index);
            }

            // GPU-driven culling for the main camera (prepass + scene pass)
            // and the shadow layers re-rendering this frame, ahead of both.
            crate::hitch::phase("rr_shadow_work");
            let shadow_work = self.shadow_work(&shadow_views, input.draws);
            crate::hitch::phase("rr_gpu_cull");
            // The Hi-Z pyramid follows the scene depth, so at a reduced render
            // scale it must match the reduced extent (not the full output).
            self.gpu_cull_frame(cb, &order, input, &shadow_work, self.render_extent);

            // GPU skinning before anything draws the skinned meshes (after the
            // cull: meshes the cluster path draws completely, shadows included,
            // skip their ordinary skin dispatch).
            crate::hitch::phase("rr_skin_skip");
            self.ctx.checkpoint(cb, c"GPU skinning");
            let skip = self.cluster_skin_skip(&order, input, &shadow_work);
            if let Some(skinner) = self.skinner.as_mut() {
                let ts = self.ts_active.then(|| self.frames[self.frame_index].timestamps);
                skinner.record(&device, cb, self.frame_index, ts, &skip);
            }
            self.cluster_skin_skip_done(skip);

            // Shadow maps first: both the camera-preview and the main scene
            // pass sample them, so they must be rendered + made readable up
            // front.
            crate::hitch::phase("rr_shadows");
            self.gpu_zone_begin(cb, gpu_zone::SHADOWS);
            self.record_shadows(&device, cb, &shadow_work, input.draws)?;
            self.gpu_zone_end(cb, gpu_zone::SHADOWS);
            crate::hitch::phase("rr_main");

            // Reflection capture face (after shadows, which it samples).
            if self.refl_job.is_some() {
                // The capture lights the scene with the FULL set (incl. baked lights
                // that `lights` drops post-bake) so the cube isn't captured dark. Fall
                // back to the main lights when the engine didn't supply a capture set.
                let (cap_lights, cap_count) = if input.capture_lights.is_empty() {
                    (lights, count)
                } else {
                    let mut cl = [GpuLight::default(); MAX_LIGHTS];
                    let mut cc = 0usize;
                    for l in input.capture_lights.iter().take(MAX_LIGHTS) {
                        cl[cc] = gpu_light(l);
                        cc += 1;
                    }
                    (cl, cc)
                };
                if let Err(e) =
                    self.step_refl_capture(cb, cap_lights, cap_count, shadow_vp, cascade_splits, input, &order)
                {
                    tracing::warn!("reflection capture step failed: {e:#}");
                    self.refl_job = None;
                }
            }


            // Planar reflections: mirrored scene passes into the RT array, one
            // per active plane, before any pass that samples them.
            if planar_n > 0
                && let Err(e) = self.record_planar_passes(
                    &device,
                    cb,
                    input,
                    lights,
                    count,
                    shadow_vp,
                    cascade_splits,
                    &order,
                    planar_n,
                    [main_extent[0] as f32, main_extent[1] as f32],
                ) {
                    tracing::warn!("planar reflection pass: {e:#}");
                }

            // Render-to-texture cameras (TV / CCTV screens), before any pass
            // that samples their textures.
            if !input.camera_targets.is_empty()
                && let Err(e) = self.record_camera_targets(
                    &device, cb, input, lights, count, shadow_vp, cascade_splits, &order,
                )
            {
                tracing::warn!("render-texture camera pass: {e:#}");
            }

            // Flux GI (game/swapchain path): depth prepass + GDF trace, folded
            // into this frame's cb. The editor viewport path records its own GI
            // inside record_viewport_scene; they're mutually exclusive per frame,
            // so the GI zone is written by whichever actually runs (never twice,
            // which would be an invalid double-write of the same query).
            // The trace's depth prepass writes the main depth target, which the
            // scene pass then reuses (EQUAL test, each pixel shaded once).
            let mut prepassed = false;
            if let Some((_, hist_valid, cur, prev_vp, prev_cam, alpha)) = sgi_record {
                self.gpu_zone_begin(cb, gpu_zone::GI);
                match self.record_screen_gi(&device, cb, input, hist_valid, cur, prev_vp, prev_cam, alpha) {
                    Ok(()) => prepassed = true,
                    Err(e) => tracing::warn!("Flux GI record failed: {e:#}"),
                }
                self.gpu_zone_end(cb, gpu_zone::GI);
            }
            // Screen-space AO (ao.rs) needs this camera's depth first; the Flux
            // trace above lays it when it runs, else a depth-only prepass does.
            if self.ao.plan.is_some_and(|p| p.target == ao::TARGET_GAME) {
                if !prepassed {
                    let frame_set = self.frame_sets[self.frame_index];
                    let ext = vk::Extent2D { width: self.render_extent[0], height: self.render_extent[1] };
                    let (img, view, att) = (self.depth.image, self.depth.view, self.depth.att_view);
                    match self.record_depth_prepass(&device, cb, frame_set, input, &input.camera, img, view, att, ext) {
                        Ok(()) => prepassed = true,
                        Err(e) => tracing::warn!("AO depth prepass failed: {e:#}"),
                    }
                }
                if prepassed {
                    let view = self.depth.view;
                    self.record_ao(&device, cb, ao::TARGET_GAME, view);
                }
            }

            // Offscreen main-camera pass next, so the swapchain's egui pass
            // can sample the result this frame.
            if render_camera_preview && self.camera_preview.is_some() {
                self.gpu_zone_begin(cb, gpu_zone::CAM_PREVIEW);
                let pcam = input.camera_preview.unwrap_or(input.camera);
                self.record_camera_preview(&device, cb, &order, input, &pcam, true, input.clear_color, None)?;
                self.gpu_zone_end(cb, gpu_zone::CAM_PREVIEW);
            }
            // The Configure window's model preview: only its own draws, no Flux trace.
            if render_model_preview
                && self.model_preview.is_some()
                && let Some(m) = input.model_preview
            {
                std::mem::swap(&mut self.camera_preview, &mut self.model_preview);
                let res = self.record_camera_preview(&device, cb, m.draws, input, &m.camera, false, [0.045, 0.05, 0.065, 1.0], None);
                std::mem::swap(&mut self.camera_preview, &mut self.model_preview);
                res?;
            }

            // Model-only preview and the impostor capture pass (their own draws, camera and light).
            self.record_offscreen(&device, cb, model_only)?;

            // Editor RTT: render the viewport 3D into its offscreen texture (the
            // swapchain pass below skips the scene because the editor sets
            // render_viewport=false, and egui shows this texture). It records its
            // own GI/scene/reflect/post GPU zones internally.
            if input.viewport_extent.is_some() && self.viewport_target.is_some() {
                self.cull_active = true;
                let r = self.record_viewport_scene(&device, cb, &order, input);
                self.cull_active = false;
                if let Err(e) = r {
                    tracing::warn!("viewport RTT failed: {e:#}");
                }
            }

            // The game path renders the scene into the HDR target (the post pass
            // tonemaps + blooms into the swapchain afterwards); the editor
            // swapchain pass (egui only) renders straight to the swapchain.
            // XR-only frames have no window pass (the eyes render after).
            let game = render_game;
            let mut stats = RenderStats::default();
            if let Some((image, color_view, _)) = target {
                let (scene_image, scene_view) = if game {
                    (
                        self.post_pass.hdr_image(self.frame_index),
                        self.post_pass.hdr_view(self.frame_index),
                    )
                } else {
                    (image, color_view)
                };

                // Color: undefined -> color attachment. The editor's scene image is the
                // acquired swapchain image: the transition must not start before the
                // acquire semaphore's wait stage (COLOR_ATTACHMENT_OUTPUT), or it can
                // run while the presentation engine still owns the image.
                image_barrier(
                    &device,
                    cb,
                    scene_image,
                    vk::ImageAspectFlags::COLOR,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    if game { vk::PipelineStageFlags2::TOP_OF_PIPE } else { vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT },
                    vk::AccessFlags2::empty(),
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                );
                // Depth: the prepass result (kept), or undefined -> attachment.
                image_barrier(
                    &device,
                    cb,
                    self.depth.image,
                    depth_aspect(),
                    if prepassed { vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL } else { vk::ImageLayout::UNDEFINED },
                    depth_attach_layout(),
                    vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
                        | vk::PipelineStageFlags2::FRAGMENT_SHADER
                        | vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::empty(),
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                );

                // Deferred-SSR G-buffer: a second MRT attachment for the game path
                // (the scene draws use gbuf=true pipeline variants). Cleared to zero
                // reflectance so untouched/sky pixels reflect nothing.
                let (gbuf_image, gbuf_view) = if game {
                    let t = &self.ssr_targets[self.frame_index];
                    (t.gbuf_image, t.gbuf_view)
                } else {
                    (vk::Image::null(), vk::ImageView::null())
                };
                if game {
                    image_barrier(
                        &device,
                        cb,
                        gbuf_image,
                        vk::ImageAspectFlags::COLOR,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        vk::PipelineStageFlags2::TOP_OF_PIPE,
                        vk::AccessFlags2::empty(),
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    );
                }

                // Motion vectors: a third MRT attachment when the upscaler is
                // active (scene draws use velocity=true pipeline variants).
                let taa_game = self.aa.plan.taa == Some(aa::TARGET_GAME);
                let velocity_active = game && (self.upscaler != Upscaler::Off || taa_game);
                self.velocity_active = velocity_active;
                let (velocity_image, velocity_view) = if velocity_active {
                    let t = &self.velocity_targets[self.frame_index];
                    (t.image, t.view)
                } else {
                    (vk::Image::null(), vk::ImageView::null())
                };
                let (reactive_image, reactive_view) = if velocity_active {
                    let t = &self.reactive_targets[self.frame_index];
                    (t.image, t.view)
                } else {
                    (vk::Image::null(), vk::ImageView::null())
                };
                if velocity_active {
                    image_barrier(
                        &device,
                        cb,
                        velocity_image,
                        vk::ImageAspectFlags::COLOR,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        vk::PipelineStageFlags2::TOP_OF_PIPE,
                        vk::AccessFlags2::empty(),
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    );
                    image_barrier(
                        &device,
                        cb,
                        reactive_image,
                        vk::ImageAspectFlags::COLOR,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        vk::PipelineStageFlags2::TOP_OF_PIPE,
                        vk::AccessFlags2::empty(),
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    );
                }

                let color_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(scene_view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: input.clear_color,
                        },
                    });
                let gbuf_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(gbuf_view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue { float32: [0.0; 4] },
                    });
                // TAA clears velocity to a sentinel so pixels no pipeline wrote
                // (sky, sprites) reproject from depth; FSR needs plain zero.
                let vclear = if taa_game { aa::VELOCITY_SENTINEL } else { 0.0 };
                let velocity_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(velocity_view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue { float32: [vclear, vclear, 0.0, 0.0] },
                    });
                let reactive_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(reactive_view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue { float32: [0.0; 4] },
                    });
                // Stored: the reflection / fog resolve reads it (the Flux depth
                // prepass only exists while screen GI traces).
                let depth_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(self.depth.att_view)
                    .image_layout(depth_attach_layout())
                    .load_op(if prepassed { vk::AttachmentLoadOp::LOAD } else { vk::AttachmentLoadOp::CLEAR })
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    });
                // Stencil (cleared to 0 with the depth, kept through the prepass).
                let stencil_attachment_info = stencil_attachment(&depth_attachment);
                let color_attachments: Vec<vk::RenderingAttachmentInfo> = if game && velocity_active {
                    vec![color_attachment, gbuf_attachment, velocity_attachment, reactive_attachment]
                } else if game {
                    vec![color_attachment, gbuf_attachment]
                } else {
                    vec![color_attachment]
                };
                // The editor's swapchain pass is egui only (full size); the game
                // scene renders at the reduced extent.
                let scene_extent = if game { render_extent_vk } else { extent };
                let mut rendering_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: scene_extent,
                    })
                    .layer_count(1)
                    .color_attachments(&color_attachments);
                // The editor's swapchain pass is egui only: no depth to clear or store.
                if game {
                    rendering_info = rendering_info.depth_attachment(&depth_attachment).stencil_attachment(&stencil_attachment_info);
                    // The game path's passes get the same zones as the editor
                    // viewport (record_viewport_scene), or they read 0.
                    self.gpu_zone_begin(cb, gpu_zone::SCENE);
                }

                device.cmd_begin_rendering(cb, &rendering_info);

                // Negative-height viewport: Y up in NDC, glTF winding preserved.
                let viewport = vk::Viewport {
                    x: 0.0,
                    y: scene_extent.height as f32,
                    width: scene_extent.width as f32,
                    height: -(scene_extent.height as f32),
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                device.cmd_set_viewport(cb, 0, &[viewport]);
                device.cmd_set_scissor(
                    cb,
                    0,
                    &[vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: scene_extent,
                    }],
                );

                // Skybox + scene only when the viewport is visible; otherwise the
                // swapchain is just cleared (above) and egui draws the dock over it.
                if render_game {
                    self.pipeline_cache.set_target_format(post::HDR_FORMAT);
                    if input.draw_skybox {
                        self.record_skybox(&device, cb, frame_set, true, true)?;
                    }
                    // Frustum-cull the main camera pass: offscreen draws are skipped
                    // (the shared `order` still feeds shadows / the reflection cube,
                    // which need offscreen casters, so only this pass is culled).
                    let main_order =
                        frustum_cull(&self.gpu_cpu_order(&order), input.draws, input.camera.proj * input.camera.view);
                    if prepassed {
                        self.draw_mode = DrawMode::DepthEqual;
                    }
                    self.cull_active = true;
                    let pass = MainPass {
                        color_image: scene_image,
                        color_view: scene_view,
                        gbuf_view: Some(gbuf_view),
                        velocity_view: velocity_active.then_some(velocity_view),
                        reactive_view: velocity_active.then_some(reactive_view),
                        depth_image: self.depth.image,
                        depth_view: self.depth.att_view,
                        depth_read_view: self.depth.view,
                        extent: render_extent_vk,
                    };
                    self.gpu_arm(true);
                    let r = self.record_main_scene_draws(&device, cb, &main_order, input, frame_set, true, &pass, 2);
                    self.gpu_arm(false);
                    self.cull_active = false;
                    self.draw_mode = DrawMode::Normal;
                    self.velocity_active = false;
                    stats = r?;
                }

                // Selection outlines: inverted hull pass over highlighted draws
                // (Shadows Only ones too, so a selected invisible caster shows).
                let outline_draws: Vec<usize> = (0..input.draws.len())
                    .filter(|&i| input.draws[i].highlight > 0.0)
                    .collect();
                if render_game && !outline_draws.is_empty() {
                    stats.outline_draws += outline_draws.len() as u32 * 2;
                    stats.draw_calls += outline_draws.len() as u32 * 2;
                    stats.pipeline_binds += 2;

                    // The outline must never be occluded: wipe the scene's depth
                    // and re-lay only the selected objects' depth (prepass below),
                    // so the hull can only be masked by the object itself. Safe
                    // because nothing after the outline reads scene depth (egui
                    // renders with depth testing off).
                    let clear = vk::ClearAttachment {
                        aspect_mask: depth_aspect(),
                        color_attachment: 0,
                        clear_value: vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        },
                    };
                    let clear_rect = vk::ClearRect {
                        rect: vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: render_extent_vk,
                        },
                        base_array_layer: 0,
                        layer_count: 1,
                    };
                    device.cmd_clear_attachments(cb, &[clear], &[clear_rect]);

                    // Depth-only prepass: re-lays the selected objects' own depth
                    // (also needed for transparent objects, which never write
                    // depth in the main pass — without it the hull's interior
                    // shows through and the whole object turns purple).
                    let instances: Vec<InstanceData> = outline_draws
                        .iter()
                        .map(|&i| InstanceData::from_model(input.draws[i].transform.to_cols_array_2d()))
                        .collect();
                    let (inst_buf, inst_off) = self.push_instances(&instances)?;
                    device.cmd_bind_vertex_buffers(cb, 1, &[inst_buf], &[inst_off]);
                    let mut depth_key = PipelineKey::depth_only();
                    depth_key.gbuf = true;
                    let depth_pipeline = self.pipeline_cache.get(&device, depth_key)?;
                    device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, depth_pipeline);
                    device.cmd_bind_descriptor_sets(
                        cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_cache.layout,
                        0,
                        &[frame_set],
                        &[],
                    );
                    for (k, &i) in outline_draws.iter().enumerate() {
                        let draw = &input.draws[i];
                        let material = &self.materials[draw.material.0];
                        let mesh = &self.meshes[draw.mesh.0];
                        device.cmd_bind_descriptor_sets(
                            cb,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline_cache.layout,
                            1,
                            &[material.set],
                            &[],
                        );
                        let push = PushData {
                            model: draw.transform.to_cols_array_2d(),
                            base_color: [0.0; 4],
                            emission: [0.0; 4],
                            params0: [0.0; 4],
                            params1: [0.0; 4],
                        };
                        device.cmd_push_constants(
                            cb,
                            self.pipeline_cache.layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            0,
                            bytemuck::bytes_of(&push),
                        );
                        device.cmd_bind_vertex_buffers(cb, 0, &[mesh.vertex_buffer.handle], &[0]);
                        device.cmd_bind_index_buffer(
                            cb,
                            mesh.index_buffer.handle,
                            0,
                            mesh.index_type,
                        );
                        device.cmd_draw_indexed(cb, mesh.index_count, 1, 0, 0, k as u32);
                    }

                    let mut outline_key = PipelineKey::outline();
                    outline_key.gbuf = true;
                    let pipeline = self.pipeline_cache.get(&device, outline_key)?;
                    device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
                    device.cmd_bind_descriptor_sets(
                        cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_cache.layout,
                        0,
                        &[frame_set],
                        &[],
                    );
                    for (k, i) in outline_draws.into_iter().enumerate() {
                        let draw = &input.draws[i];
                        let material = &self.materials[draw.material.0];
                        let mesh = &self.meshes[draw.mesh.0];
                        device.cmd_bind_descriptor_sets(
                            cb,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline_cache.layout,
                            1,
                            &[material.set],
                            &[],
                        );
                        let p = &material.params;
                        let push = PushData {
                            model: draw.transform.to_cols_array_2d(),
                            base_color: p.base_color,
                            emission: [0.0; 4],
                            params0: draw.mesh_center.extend(0.0).to_array(),
                            params1: [0.0, 0.0, 0.0, draw.highlight],
                        };
                        device.cmd_push_constants(
                            cb,
                            self.pipeline_cache.layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            0,
                            bytemuck::bytes_of(&push),
                        );
                        device.cmd_bind_vertex_buffers(cb, 0, &[mesh.vertex_buffer.handle], &[0]);
                        device.cmd_bind_index_buffer(
                            cb,
                            mesh.index_buffer.handle,
                            0,
                            mesh.index_type,
                        );
                        device.cmd_draw_indexed(cb, mesh.index_count, 1, 0, 0, k as u32);
                    }
                }

                // Editor swapchain pass draws egui here (no scene/HDR). The game path
                // draws egui in the post pass below (on the tonemapped image).
                if !game
                    && let Some(egui_draw) = &input.egui {
                        self.gpu_zone_begin(cb, gpu_zone::EGUI);
                        self.egui
                            .as_mut()
                            .unwrap()
                            .cmd_draw(cb, extent, egui_draw.pixels_per_point, &egui_draw.primitives)
                            .map_err(|e| anyhow::anyhow!("recording egui draw: {e}"))?;
                        self.gpu_zone_end(cb, gpu_zone::EGUI);
                    }

                device.cmd_end_rendering(cb);

                if game && !input.decals.is_empty() {
                    let decal_frame = self.decal_frame(input);
                    self.record_decals(
                        &device, cb, input, scene_view,
                        self.depth.image, self.depth.view,
                        gbuf_image, gbuf_view, render_extent_vk, &decal_frame,
                    );
                }

                if game {
                    self.gpu_zone_end(cb, gpu_zone::SCENE);
                    // Next frame's occlusion pyramid, unless the selection outline
                    // replaced the scene depth.
                    if render_game && input.draws.iter().any(|d| d.highlight > 0.0) {
                        self.gpu_hiz_invalidate();
                    } else {
                        self.gpu_hiz_build(&device, cb, input, (self.depth.image, self.depth.view), true);
                    }
                    self.gpu_zone_begin(cb, gpu_zone::REFLECT);
                    self.ctx.checkpoint(cb, c"underwater");
                    if let Some(fx) = input.postfx.underwater.as_ref() {
                        self.underwater.record(
                            &device, cb, underwater::SLOT_MAIN + self.frame_index as u32,
                            (scene_image, scene_view), (self.depth.image, self.depth.view), render_extent_vk,
                            &input.camera, fx, &input.light, input.time, 0,
                        );
                    }
                    // HDR scene -> shader-read so the resolve + post passes sample it.
                    image_barrier(
                        &device, cb, scene_image, vk::ImageAspectFlags::COLOR,
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                        vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ,
                    );
                    if rt_on || ssr_on || fog_on {
                        image_barrier(
                            &device, cb, self.depth.image, depth_aspect(),
                            depth_attach_layout(), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS, vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
                            vk::AccessFlags2::SHADER_READ,
                        );
                    }
                    // Reflections + fog write a resolved HDR image the post pass reads.
                    // Ray-traced (mode 2) takes precedence; else SSR/fog resolve.
                    let post_src = if rt_on {
                        // G-buffer -> read for the RT compute pass.
                        image_barrier(
                            &device, cb, gbuf_image, vk::ImageAspectFlags::COLOR,
                            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                            vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_READ,
                        );
                        let (instances, lights) = self.rt_scene(input);
                        let refl2 = refl2_uniforms(input);
                        let rt_uniforms = rt_reflect::RtUniforms {
                            inv_proj: ubo.inv_proj,
                            inv_view: ubo.inv_view,
                            camera_pos: ubo.camera_pos,
                            refl_center: ubo.refl_center,
                            refl_extents: ubo.refl_extents,
                            refl_box: ubo.refl_box,
                            params: [
                                fs.ssr_roughness_cutoff,
                                fs.ssr_intensity,
                                lights.len() as f32,
                                // env IBL intensity (skybox-enabled gate): 0 when the
                                // skybox is off, so RT reflection misses don't pull in
                                // the sky cube (matches the forward env_scale).
                                ubo.ambient[3],
                            ],
                            screen: [render_extent_vk.width as f32, render_extent_vk.height as f32, 0.0, fs.bounces as f32],
                            quality: rt_quality(&fs, self.frame_serial),
                            lighting_mode: [fs.ray_lighting_mode as u32, 0, 0, 0],
                            refl2_center: refl2.refl2_center,
                            refl2_extents: refl2.refl2_extents,
                            refl2_box: refl2.refl2_box,
                            refl3_center: refl2.refl3_center,
                            refl3_extents: refl2.refl3_extents,
                            refl3_box: refl2.refl3_box,
                            refl4_center: refl2.refl4_center,
                            refl4_extents: refl2.refl4_extents,
                            refl4_box: refl2.refl4_box,
                            refl_blend: refl2.refl_blend,
                            refl_blend2: refl2.refl_blend2,
                            water: [input.water_time, input.water.len().min(MAX_WATER_BODIES) as f32, 0.0, 0.0],
                            water_bodies: water_bodies(input),
                        };
                        let env_view = self.reflection_env.view;
                        let depth_view = self.depth.view;
                        let resolved_image = self.ssr_targets[self.frame_index].resolved_image();
                        let resolved_view = self.ssr_targets[self.frame_index].resolved_view();
                        let post_set = self.ssr_targets[self.frame_index].post_set;
                        let allocator = self.allocator.clone();
                        let frame_index = self.frame_index;
                        let command_pool = self.command_pool;
                        self.ctx.checkpoint(cb, c"ray-traced reflections");
                        let rt_water = self.rt_water(input, &instances);
                        let tex_views = self.rt_texture_views();
                        if let (Some(rt), Some(allocator)) = (self.rt_reflect.as_mut(), allocator)
                            && let Err(e) = rt.record(
                                &self.ctx, &allocator, command_pool, cb, frame_index, 0, self.frame_serial, &self.meshes,
                                &instances, &rt_water, &lights, resolved_image, resolved_view, scene_view,
                                depth_view, gbuf_view, env_view, self.reflection_env_b.view, self.reflection_env_c.view, self.reflection_env_d.view, &tex_views, (self.lightmaps.view, self.lightmap_sampler), (self.area_atlas.view,self.sampler),
                                self.flux_surface.as_ref().and_then(|s| s.reflection_resources(self.frame_index, self.frame_serial)), &rt_uniforms, render_extent_vk,
                            ) {
                                tracing::warn!("RT reflection record failed: {e:#}");
                            }
                        self.tex_view_cache = tex_views;
                        // Resolved image GENERAL -> shader read for the post pass.
                        image_barrier(
                            &device, cb, resolved_image, vk::ImageAspectFlags::COLOR,
                            vk::ImageLayout::GENERAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_WRITE,
                            vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ,
                        );
                        Some(post_set)
                    } else if ssr_on || fog_on {
                        image_barrier(
                            &device, cb, gbuf_image, vk::ImageAspectFlags::COLOR,
                            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                            vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ,
                        );
                        let uniforms = self.ssr_uniforms(input, render_extent_vk, ssr_on);
                        let env_view = self.reflection_env.view;
                        let depth_view = self.depth.view;
                        self.ctx.checkpoint(cb, c"screen-space reflections");
                        self.ssr_resolve.record(
                            &device,
                            cb,
                            &mut self.ssr_targets[self.frame_index],
                            scene_view,
                            depth_view,
                            env_view,
                            self.reflection_env_b.view,
                            self.reflection_env_c.view,
                            self.reflection_env_d.view,
                            &uniforms,
                            self.frame_index,
                        );
                        Some(self.ssr_targets[self.frame_index].post_set)
                    } else {
                        None
                    };
                    // FSR3: upscale the reduced HDR (scene, or the SSR/RT resolve)
                    // into the full-resolution output, which the post pass samples.
                    let upscale_set = if (self.fsr3.is_some() || self.fsr3_fg.is_some()) && velocity_active {
                        let frame = self.frame_index;
                        let src_view = if post_src.is_some() {
                            self.ssr_targets[frame].resolved_view()
                        } else {
                            self.post_pass.hdr_view(frame)
                        };
                        let (near, far, fov_y) = perspective_params(input.camera.proj);
                        let dt = (input.time - self.last_time).max(1.0 / 1000.0);
                        let upscale_input = UpscaleInput {
                            color: src_view,
                            depth: self.depth.sample_view(),
                            motion: self.velocity_targets[frame].view,
                            reactive: self.reactive_targets[frame].view,
                            exposure: input.postfx.exposure,
                            jitter: upscaler_jitter(self.jitter),
                            render_size: self.render_extent,
                            output_size: [extent.width, extent.height],
                            prev_view_proj: self.prev_view_proj,
                            view_proj: ubo.view_proj,
                        };
                        // Motion + reactive + depth -> shader read; output -> UAV.
                        image_barrier(&device, cb, self.velocity_targets[frame].image, vk::ImageAspectFlags::COLOR,
                            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                            vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_READ);
                        image_barrier(&device, cb, self.reactive_targets[frame].image, vk::ImageAspectFlags::COLOR,
                            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                            vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_READ);
                        image_barrier(&device, cb, self.depth.image, depth_aspect(),
                            depth_attach_layout(), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                            vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_READ);
                        image_barrier(&device, cb, self.upscale_outputs[frame].image, vk::ImageAspectFlags::COLOR,
                            vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL,
                            vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty(),
                            vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_WRITE);
                        let output_view = self.upscale_outputs[frame].view;
                        let formats = fsr3::Fsr3Formats {
                            color: post::HDR_FORMAT,
                            depth: vk::Format::R32_SFLOAT,
                            motion: pipeline::VELOCITY_FORMAT,
                            reactive: pipeline::REACTIVE_FORMAT,
                            output: post::HDR_FORMAT,
                        };
                        let reset = self.fsr3_reset;
                        self.fsr3_reset = false;
                        self.last_time = input.time;
                        self.ctx.checkpoint(cb, c"fsr3 upscale");
                        let dispatch_res = if let Some(f) = self.fsr3_fg.as_mut() {
                            f.dispatch_upscale_with(cb, &upscale_input, output_view, &formats, 0.0, dt, near, far, fov_y, 1.0, reset)
                        } else if let Some(f) = self.fsr3.as_mut() {
                            f.dispatch(cb, &upscale_input, output_view, &formats, 0.0, dt, near, far, fov_y, 1.0, reset)
                        } else {
                            Ok(())
                        };
                        if let Err(e) = dispatch_res {
                            tracing::warn!("FSR3 dispatch failed: {e:#}");
                        }
                        // Output GENERAL -> shader read for the post pass.
                        image_barrier(&device, cb, self.upscale_outputs[frame].image, vk::ImageAspectFlags::COLOR,
                            vk::ImageLayout::GENERAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_WRITE,
                            vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ);
                        Some(self.upscale_sets[frame])
                    } else {
                        None
                    };
                    // TAA (exclusive with the FSR upscaler): resolve the reflection /
                    // fog result (or the scene) against the history; the post pass
                    // then samples the resolved image.
                    let post_src = if taa_game && upscale_set.is_none() {
                        let frame = self.frame_index;
                        // Depth is already shader-readable after the decals or the
                        // reflection / fog resolve.
                        if !(rt_on || ssr_on || fog_on || !input.decals.is_empty()) {
                            image_barrier(&device, cb, self.depth.image, depth_aspect(),
                                depth_attach_layout(), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS, vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                                vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ);
                        }
                        for img in [self.velocity_targets[frame].image, self.reactive_targets[frame].image] {
                            image_barrier(&device, cb, img, vk::ImageAspectFlags::COLOR,
                                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                                vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ);
                        }
                        let color = if post_src.is_some() { self.ssr_targets[frame].resolved_view() } else { self.post_pass.hdr_view(frame) };
                        let inputs = aa::AaInputs {
                            color,
                            depth: self.depth.view,
                            velocity: self.velocity_targets[frame].view,
                            reactive: self.reactive_targets[frame].view,
                        };
                        self.record_taa(&device, cb, aa::TARGET_GAME, &ubo, inputs).or(post_src)
                    } else {
                        post_src
                    };
                    self.gpu_zone_end(cb, gpu_zone::REFLECT);
                    self.gpu_zone_begin(cb, gpu_zone::POST);
                    let bloom_src = upscale_set.or(post_src).unwrap_or(self.post_pass.frame_set(self.frame_index));
                    self.post_pass.exposure.record(&device,cb,bloom_src,extent,input.postfx.auto_exposure,0,
                        self.frame_serial,self.started.elapsed().as_secs_f64());
                    self.ctx.checkpoint(cb, c"bloom");
                    self.post_pass.record_bloom(&device, cb, self.post_pass.chain(), bloom_src, &input.postfx);
                    // Frame gen routes the tonemap into a display-space present
                    // target (pre-UI); the interpolated output is blitted to the
                    // swapchain and the UI drawn on top. Otherwise straight to it.
                    let fg = self.frame_gen_active();
                    let (post_image, post_view) = if fg {
                        (self.fg_present[self.frame_index].image, self.fg_present[self.frame_index].view)
                    } else {
                        (image, color_view)
                    };
                    image_barrier(
                        &device, cb, post_image, vk::ImageAspectFlags::COLOR,
                    // The source stage must match the acquire semaphore's wait stage
                    // (COLOR_ATTACHMENT_OUTPUT): a TOP_OF_PIPE source lets the layout
                    // transition run before the presentation engine released the image.
                        vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::empty(),
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    );
                    let post_color = vk::RenderingAttachmentInfo::default()
                        .image_view(post_view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::DONT_CARE)
                        .store_op(vk::AttachmentStoreOp::STORE);
                    let post_attachments = [post_color];
                    let post_info = vk::RenderingInfo::default()
                        .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
                        .layer_count(1)
                        .color_attachments(&post_attachments);
                    device.cmd_begin_rendering(cb, &post_info);
                    let wobble = underwater::wobble(input.postfx.underwater.as_ref(), &input.camera, input.time);
                    self.aa_post_begin();
                    match upscale_set.or(post_src) {
                        Some(set) => {
                            let bloom = self.post_pass.chain().output_set();
                            self.post_pass.record_set(&device, cb, set, bloom, extent, &input.postfx, wobble)
                        }
                        None => self.post_pass.record(&device, cb, self.frame_index, extent, &input.postfx, wobble),
                    }
                    self.aa_post_end();
                    if !fg {
                        // egui on top of the tonemapped image (frame gen draws the
                        // UI after the interpolated frame is blitted).
                        if let Some(egui_draw) = &input.egui {
                            self.gpu_zone_begin(cb, gpu_zone::EGUI);
                            self.egui
                                .as_mut()
                                .unwrap()
                                .cmd_draw(cb, extent, egui_draw.pixels_per_point, &egui_draw.primitives)
                                .map_err(|e| anyhow::anyhow!("recording egui draw: {e}"))?;
                            self.gpu_zone_end(cb, gpu_zone::EGUI);
                        }
                        // SDF text over the UI (game path only).
                        if let Some(t) = self.text_gpu.as_mut() {
                            t.record_screen(&device, cb, self.frame_index, extent, self.swapchain.format.format);
                        }
                    }
                    device.cmd_end_rendering(cb);
                    self.gpu_zone_end(cb, gpu_zone::POST);

                    if fg {
                        // Frame generation: presentColor (fg_present) -> outputs[0]
                        // (fg_output), then blit fg_output to the swapchain + UI.
                        let present_img = self.fg_present[self.frame_index].image;
                        let present_view = self.fg_present[self.frame_index].view;
                        let out_img = self.fg_output[self.frame_index].image;
                        let out_view = self.fg_output[self.frame_index].view;
                        image_barrier(&device, cb, present_img, vk::ImageAspectFlags::COLOR,
                            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                            vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ);
                        image_barrier(&device, cb, out_img, vk::ImageAspectFlags::COLOR,
                            vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL,
                            vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty(),
                            vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_WRITE);
                        let reset = self.fsr3_reset;
                        self.fsr3_reset = false;
                        let bb_format = self.swapchain.format.format;
                        if let Some(f) = self.fsr3_fg.as_mut()
                            && let Err(e) = f.frame_gen_dispatch(cb, present_view, out_view, bb_format, [extent.width, extent.height], reset)
                        {
                            tracing::warn!("FSR3 frame gen failed: {e:#}");
                        }
                        // Blit fg_output -> swapchain (nearest: same size).
                        image_barrier(&device, cb, out_img, vk::ImageAspectFlags::COLOR,
                            vk::ImageLayout::GENERAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_WRITE,
                            vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ);
                        image_barrier(&device, cb, image, vk::ImageAspectFlags::COLOR,
                            vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty(),
                            vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
                        let sub = vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        };
                        let region = vk::ImageBlit::default()
                            .src_subresource(sub)
                            .src_offsets([vk::Offset3D { x: 0, y: 0, z: 0 }, vk::Offset3D { x: extent.width as i32, y: extent.height as i32, z: 1 }])
                            .dst_subresource(sub)
                            .dst_offsets([vk::Offset3D { x: 0, y: 0, z: 0 }, vk::Offset3D { x: extent.width as i32, y: extent.height as i32, z: 1 }]);
                        device.cmd_blit_image(cb, out_img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region], vk::Filter::NEAREST);
                        // egui on the swapchain.
                        image_barrier(&device, cb, image, vk::ImageAspectFlags::COLOR,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                            vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE,
                            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE);
                        let ui_color = vk::RenderingAttachmentInfo::default()
                            .image_view(color_view)
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::DONT_CARE)
                            .store_op(vk::AttachmentStoreOp::STORE);
                        let ui_attachments = [ui_color];
                        let ui_info = vk::RenderingInfo::default()
                            .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
                            .layer_count(1)
                            .color_attachments(&ui_attachments);
                        device.cmd_begin_rendering(cb, &ui_info);
                        if let Some(egui_draw) = &input.egui {
                            self.gpu_zone_begin(cb, gpu_zone::EGUI);
                            self.egui
                                .as_mut()
                                .unwrap()
                                .cmd_draw(cb, extent, egui_draw.pixels_per_point, &egui_draw.primitives)
                                .map_err(|e| anyhow::anyhow!("recording egui draw: {e}"))?;
                            self.gpu_zone_end(cb, gpu_zone::EGUI);
                        }
                        if let Some(t) = self.text_gpu.as_mut() {
                            t.record_screen(&device, cb, self.frame_index, extent, self.swapchain.format.format);
                        }
                        device.cmd_end_rendering(cb);
                    }
                }

                self.record_swapchain_capture(&device, cb, image, extent);
                image_barrier(
                    &device,
                    cb,
                    image,
                    vk::ImageAspectFlags::COLOR,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                    vk::AccessFlags2::empty(),
                );

            } else {
                self.gpu_hiz_invalidate();
            }

            // Editor path: the scene was drawn in record_viewport_scene, so fold
            // its draw counts in (the swapchain `stats` here is otherwise empty,
            // which is why draw calls read 0).
            if !game {
                let v = self.viewport_draw_stats;
                stats.draw_calls += v.draw_calls;
                stats.instanced_draws += v.instanced_draws;
                stats.opaque_draws += v.opaque_draws;
                stats.transparent_draws += v.transparent_draws;
                stats.error_draws += v.error_draws;
                stats.pipeline_binds += v.pipeline_binds;
                stats.materials_drawn = stats.materials_drawn.max(v.materials_drawn);
            }

            stats.pipeline_variants = self.pipeline_cache.variant_count();
            // GPU per-pass timings carry over from this slot's previous frame
            // (read back above); attach them for the profiler breakdown.
            stats.gpu_frame_ms = self.gpu_zone_ms[gpu_zone::FRAME];
            stats.gpu_gi_ms = self.gpu_zone_ms[gpu_zone::GI];
            stats.gpu_shadows_ms = self.gpu_zone_ms[gpu_zone::SHADOWS];
            stats.gpu_scene_ms = self.gpu_zone_ms[gpu_zone::SCENE];
            stats.gpu_reflect_ms = self.gpu_zone_ms[gpu_zone::REFLECT];
            stats.gpu_post_ms = self.gpu_zone_ms[gpu_zone::POST];
            stats.gpu_egui_ms = self.gpu_zone_ms[gpu_zone::EGUI];
            stats.gpu_cam_preview_ms = self.gpu_zone_ms[gpu_zone::CAM_PREVIEW];
            stats.gpu_skin_ms = self.gpu_zone_ms[gpu_zone::SKIN];
            stats.gpu_morph_ms = self.gpu_zone_ms[gpu_zone::MORPH];
            stats.gpu_gi_gather_ms = self.gpu_zone_ms[gpu_zone::GI_GATHER];
            stats.gpu_gi_denoise_ms = self.gpu_zone_ms[gpu_zone::GI_DENOISE];
            stats.gpu_gi_integrate_ms = self.gpu_zone_ms[gpu_zone::GI_INTEGRATE];
            stats.gpu_gi_surface_ms = self.gpu_zone_ms[gpu_zone::GI_SURFACE];
            stats.gpu_camera_gi_ms = self.gpu_zone_ms[gpu_zone::CAMERA_GI];
            stats.gpu_camera_gi_gather_ms = self.gpu_zone_ms[gpu_zone::CAMERA_GI_GATHER];
            stats.gpu_camera_gi_denoise_ms = self.gpu_zone_ms[gpu_zone::CAMERA_GI_DENOISE];
            stats.gpu_camera_gi_integrate_ms = self.gpu_zone_ms[gpu_zone::CAMERA_GI_INTEGRATE];
            stats.gpu_camera_gi_surface_ms = self.gpu_zone_ms[gpu_zone::CAMERA_GI_SURFACE];
            stats.gpu_ao_ms = self.gpu_zone_ms[gpu_zone::AO];
            stats.gpu_aa_ms = self.gpu_zone_ms[gpu_zone::AA];
            self.gpu_cull_stats(&mut stats);
            self.cluster_stats(&mut stats);
            self.last_stats = stats;

            self.record_viewport_capture(&device, cb);

            // Frame-end stamp last, then mark this slot primed for readback.
            self.gpu_zone_end(cb, gpu_zone::FRAME);
            if self.gpu_ts_period > 0.0 {
                self.gpu_ts_primed[self.frame_index] = true;
            }

            device.end_command_buffer(cb)?;

            crate::hitch::phase("gpu_submit");
            let cb_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
            match target {
                Some((_, _, render_finished)) => {
                    let wait_infos = [vk::SemaphoreSubmitInfo::default()
                        .semaphore(image_available)
                        .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
                    let signal_infos = [vk::SemaphoreSubmitInfo::default()
                        .semaphore(render_finished)
                        // Everything, including the PRESENT_SRC layout transition (which a
                        // COLOR_ATTACHMENT_OUTPUT-only signal does not strictly cover).
                        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
                    let submit = vk::SubmitInfo2::default()
                        .wait_semaphore_infos(&wait_infos)
                        .command_buffer_infos(&cb_infos)
                        .signal_semaphore_infos(&signal_infos);
                    device.queue_submit2(self.ctx.queue, &[submit], in_flight)?;
                }
                None => {
                    let submit = vk::SubmitInfo2::default().command_buffer_infos(&cb_infos);
                    device.queue_submit2(self.ctx.queue, &[submit], in_flight)?;
                }
            }
        }

        if let (Some(image_index), Some((_, _, render_finished))) = (image_index, target) {
            match crate::hitch::present(|| self.swapchain.present(self.ctx.queue, image_index, render_finished)) {
                Ok(suboptimal) => {
                    if suboptimal {
                        self.resize_gate.force();
                    }
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.resize_gate.force(),
                Err(e) => return Err(e).context("presenting swapchain image"),
            }
        }

        crate::hitch::phase("render_post");
        // Queue this frame's egui frees; they're released at the start of the
        // next frame (behind an idle, before that frame's `set_textures`). The
        // freed ids are no longer in this frame's primitives, but this frame's
        // (and the prior frame's) draw may still be in flight, so we can't free
        // them now — and freeing before the next `set` avoids the id-reuse UAF.
        if let Some(egui_draw) = &input.egui
            && !egui_draw.textures_delta.free.is_empty()
        {
            self.egui_free_slots[self.frame_index]
                .extend_from_slice(&egui_draw.textures_delta.free);
        }

        if let Some(t) = self.text_gpu.as_mut() {
            t.end_frame();
        }
        if let Some(i) = self.impostor_gpu.as_mut() {
            i.end_frame();
        }
        // The sprite input is per frame: the engine refills it.
        self.sprite_input.clear();
        self.frame_index = (self.frame_index + 1) % FRAMES_IN_FLIGHT;
        Ok(())
    }
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

/// One sample of the Halton(base) sequence, in [0, 1). Used for the temporal
/// upscaler's sub-pixel camera jitter: a different, evenly-spread sample each
/// frame so the reduced-resolution image accumulates the full detail over time.
fn halton(index: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut r = 0.0;
    let mut i = index;
    while i > 0 {
        f /= base as f32;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
}

/// A sample of the Halton(2, 3) sequence, in [0, 1)².
pub fn halton23(index: u32) -> [f32; 2] {
    [halton(index, 2), halton(index, 3)]
}

/// Offset a projection matrix by a sub-pixel `jitter` (in [0, 1)²) so the scene
/// renders a different half-texel sample each frame (the temporal upscaler
/// reconstructs the full resolution from the accumulating samples).
pub fn jitter_proj(mut proj: Mat4, jitter: [f32; 2], render_size: [f32; 2]) -> Mat4 {
    let jx = (jitter[0] - 0.5) * 2.0 / render_size[0].max(1.0);
    let jy = (jitter[1] - 0.5) * 2.0 / render_size[1].max(1.0);
    if is_ortho_proj(&proj) {
        // Orthographic: a constant clip-space shift (the translation column).
        proj.w_axis.x += jx;
        proj.w_axis.y += jy;
        return proj;
    }
    // Perspective: clip.x = ... + z_axis.x * z_eye and w_clip = -z_eye, so
    // z_axis.x contributes -z_axis.x to NDC x at every depth. (glam is
    // column-major and the UBO is uploaded with `to_cols_array_2d` into a
    // default column-major GLSL mat4, so the jitter terms live in z_axis.)
    proj.z_axis.x -= jx;
    proj.z_axis.y -= jy;
    proj
}

/// The jitter handed to the temporal upscaler for a `jitter` given to
/// [`jitter_proj`]. FFX `jitterOffset` is in render-target pixels, centred on
/// zero, +y down; the FFX docs apply it to the projection as
/// `clip.x += 2 * jx / w`, `clip.y += -2 * jy / h`. The scene pass uses a
/// negative-height viewport (clip +y = screen up), so `jitter_proj`'s clip shift
/// `(j - 0.5) * 2 / size` equals that with `jx = j0 - 0.5`, `jy = 0.5 - j1`.
pub fn upscaler_jitter(jitter: [f32; 2]) -> [f32; 2] {
    [jitter[0] - 0.5, 0.5 - jitter[1]]
}

/// True for an orthographic projection (last matrix row `(0, 0, 0, 1)`).
pub(crate) fn is_ortho_proj(proj: &Mat4) -> bool {
    proj.w_axis.w.abs() > 0.5
}

/// The reduced-resolution render extent for an output extent at `scale`
/// (fraction of the output the scene renders at; `1.0` = native). Clamped to a
/// sane range so a degenerate scale can't collapse the targets.
pub fn render_extent_for(main_extent: [u32; 2], scale: f32) -> [u32; 2] {
    [
        (main_extent[0] as f32 * scale).round().clamp(16.0, 16384.0) as u32,
        (main_extent[1] as f32 * scale).round().clamp(16.0, 16384.0) as u32,
    ]
}

/// Extracts `(near, far, vertical_fov)` from a `glam::perspective_rh` matrix
/// (0..1 depth, column-major). FSR uses these for depth reconstruction and
/// exposure; the camera stores only the projection, so they are recovered here.
pub fn perspective_params(proj: Mat4) -> (f32, f32, f32) {
    // glam::perspective_rh (0..1 depth, column-major):
    //   col(1)[1] = cot(fov/2), col(2)[2] = far/(near-far), col(3)[2] = near*far/(near-far).
    let m11 = proj.col(1)[1];
    let m22 = proj.col(2)[2];
    let m32 = proj.col(3)[2];
    if is_ortho_proj(&proj) {
        // glam::orthographic_rh: col(2)[2] = 1/(near-far), col(3)[2] = near/(near-far).
        // The upscaler has no orthographic mode, so report a nominal vertical
        // fov for its depth reconstruction.
        let near = m32 / m22;
        let far = near - 1.0 / m22;
        return (near, far, 60f32.to_radians());
    }
    let near = m32 / m22;
    let far = near * m22 / (1.0 + m22);
    let fov_y = 2.0 * (1.0 / m11).atan();
    (near, far, fov_y)
}

#[cfg(test)]
mod jitter_tests {
    use super::{halton23, jitter_proj, perspective_params, render_extent_for};
    use glam::Mat4;

    #[test]
    fn halton23_spreads_and_stays_in_unit_square() {
        let mut seen = std::collections::HashSet::new();
        for i in 0..64 {
            let [x, y] = halton23(i);
            assert!((0.0..1.0).contains(&x) && (0.0..1.0).contains(&y), "{i}: {x}, {y}");
            seen.insert((x.to_bits(), y.to_bits()));
        }
        assert_eq!(seen.len(), 64, "no repeated samples across 64 frames");
    }

    #[test]
    fn jitter_offsets_the_clip_projection_columns() {
        // (A perspective matrix: an identity matrix reads as orthographic.)
        let mut proj = Mat4::IDENTITY;
        proj.z_axis.w = -1.0;
        proj.w_axis.w = 0.0;
        let out = jitter_proj(proj, [0.5, 0.5], [100.0, 100.0]);
        assert_eq!(out.z_axis.x, 0.0, "centre sample = no offset");
        // The jitter must not touch the depth row (x_axis.z / y_axis.z), which
        // would skew depth with screen position.
        assert_eq!((out.x_axis.z, out.y_axis.z), (0.0, 0.0));
        // Behaviour: a real perspective matrix shifts NDC x/y by the same
        // amount at every depth (the jitter is a pure sub-pixel image shift).
        let p = Mat4::perspective_rh(1.0, 1.5, 0.1, 100.0);
        let j = jitter_proj(p, [1.0, 0.0], [100.0, 100.0]);
        for z in [-0.5f32, -10.0, -90.0] {
            let a = p.project_point3(glam::Vec3::new(0.3, -0.2, z));
            let b = j.project_point3(glam::Vec3::new(0.3, -0.2, z));
            assert!((b.x - a.x - 0.01).abs() < 1e-5, "ndc x shift at z={z}: {}", b.x - a.x);
            assert!((b.y - a.y + 0.01).abs() < 1e-5, "ndc y shift at z={z}: {}", b.y - a.y);
            assert!((b.z - a.z).abs() < 1e-6, "depth unchanged at z={z}");
        }
    }

    /// Screen pixels (+x right, +y down; the scene viewport is y-flipped so clip
    /// +y is up) a projected point moved by the jitter.
    fn pixel_shift(base: Mat4, jittered: Mat4, size: [f32; 2]) -> [f32; 2] {
        let p = glam::Vec3::new(0.3, -0.2, -7.0);
        let (a, b) = (base.project_point3(p), jittered.project_point3(p));
        [(b.x - a.x) * 0.5 * size[0], -(b.y - a.y) * 0.5 * size[1]]
    }

    #[test]
    fn upscaler_jitter_matches_the_projection_shift() {
        // The FFX contract: jitterOffset is the pixel shift applied to the image
        // (units: render pixels, +y down). Check the offset handed to the
        // upscaler against the real shift, perspective and orthographic, for
        // the whole Halton sequence and a non-square target.
        let size = [160.0, 90.0];
        for proj in [Mat4::perspective_rh(1.0, 16.0 / 9.0, 0.1, 100.0), Mat4::orthographic_rh(-8.0, 8.0, -4.5, 4.5, 0.1, 50.0)] {
            for i in 0..32 {
                let j = super::halton23(i);
                let shift = pixel_shift(proj, jitter_proj(proj, j, size), size);
                let off = super::upscaler_jitter(j);
                assert!((shift[0] - off[0]).abs() < 1e-3, "x {i}: shift {} vs offset {}", shift[0], off[0]);
                assert!((shift[1] - off[1]).abs() < 1e-3, "y {i}: shift {} vs offset {}", shift[1], off[1]);
                assert!(off[0].abs() <= 0.5 && off[1].abs() <= 0.5, "in pixels, within half a pixel");
            }
        }
        assert_eq!(super::upscaler_jitter([0.5, 0.5]), [0.0, 0.0]);
    }

    #[test]
    fn render_extent_scales_and_clamps() {
        assert_eq!(render_extent_for([1920, 1080], 1.0), [1920, 1080]);
        assert_eq!(render_extent_for([1920, 1080], 0.5), [960, 540]);
        assert_eq!(render_extent_for([1920, 1080], 1.0 / 1.5), [1280, 720]);
        // Tiny output sizes never collapse below 16, and stay even-round.
        assert_eq!(render_extent_for([1, 1], 0.25), [16, 16]);
    }

    #[test]
    fn perspective_params_recovers_near_far_fov() {
        let (near, far, fov_y) = perspective_params(Mat4::perspective_rh(50.0_f32.to_radians(), 1.5, 0.05, 500.0));
        assert!((near - 0.05).abs() < 1e-4, "near: {near}");
        assert!((far - 500.0).abs() < 1.0, "far: {far}");
        assert!((fov_y - 50.0_f32.to_radians()).abs() < 1e-4, "fov: {fov_y}");
    }

    #[test]
    fn orthographic_projection_params_and_jitter() {
        let proj = Mat4::orthographic_rh(-8.0, 8.0, -4.0, 4.0, 0.5, 40.0);
        let (near, far, _) = perspective_params(proj);
        assert!((near - 0.5).abs() < 1e-4 && (far - 40.0).abs() < 1e-3, "{near} {far}");
        // A half-texel jitter shifts every clip-space x by the same amount at any depth.
        let j = jitter_proj(proj, [1.0, 0.5], [100.0, 100.0]);
        for z in [-1.0f32, -30.0] {
            let a = proj.project_point3(glam::Vec3::new(2.0, 1.0, z));
            let b = j.project_point3(glam::Vec3::new(2.0, 1.0, z));
            assert!((b.x - a.x - 0.01).abs() < 1e-6 && (b.y - a.y).abs() < 1e-6);
        }
    }

    #[test]
    fn quality_presets_map_to_monotonic_render_scales() {
        use crate::UpscaleQuality;
        let scales: Vec<f32> = UpscaleQuality::ALL.iter().map(|q| q.render_scale()).collect();
        assert_eq!(UpscaleQuality::Native.render_scale(), 1.0);
        assert!((UpscaleQuality::Quality.render_scale() - 1.0 / 1.5).abs() < 1e-6);
        for w in scales.windows(2) {
            assert!(w[0] > w[1], "quality order lowers the render scale");
        }
    }

    #[test]
    fn quality_presets_round_trip_through_render_scale() {
        use crate::UpscaleQuality;
        for q in UpscaleQuality::ALL {
            assert_eq!(UpscaleQuality::from_render_scale(q.render_scale()), q, "{q:?}");
        }
    }
}

/// Inputs the temporal upscaler consumes each frame: the reduced-resolution
/// scene colour + depth + motion, the reactive mask, tonemap exposure, this
/// frame's sub-pixel jitter, the previous view-projection for reprojection,
/// and the render/output resolutions. The FSR 3.1 pass builds one of these
/// from the reduced-resolution scene targets before tonemap.
pub struct UpscaleInput {
    /// Reduced-resolution linear HDR colour.
    pub color: vk::ImageView,
    /// Reduced-resolution scene depth.
    pub depth: vk::ImageView,
    /// Reduced-resolution motion vectors (RG16F, screen-space).
    pub motion: vk::ImageView,
    /// Reduced-resolution reactive mask (high reactiveness = less history).
    pub reactive: vk::ImageView,
    /// Tonemap exposure (the upscaler works in the same exposure domain).
    pub exposure: f32,
    /// This frame's jitter in FFX convention (see [`upscaler_jitter`]): pixels
    /// of the render target, in [-0.5, 0.5), +x right, +y down.
    pub jitter: [f32; 2],
    /// Reduced-resolution render size.
    pub render_size: [u32; 2],
    /// Full-resolution output size.
    pub output_size: [u32; 2],
    /// Previous frame's view-projection (reprojection).
    pub prev_view_proj: [[f32; 4]; 4],
    /// This frame's (jittered) view-projection.
    pub view_proj: [[f32; 4]; 4],
}

/// Build the per-frame UBO for a given camera, sharing the already-packed
/// scene lights and ambient/key-light fallback.
#[allow(clippy::too_many_arguments)]
fn frame_ubo(
    camera: &CameraData,
    input: &FrameInput,
    lights: [GpuLight; MAX_LIGHTS],
    count: usize,
    shadow_vp: [[[f32; 4]; 4]; MAX_SHADOW_VIEWS],
    cascade_splits: [f32; 4],
    probe_volumes: &[GpuProbeVolume],
    jitter: [f32; 2],
    render_size: [f32; 2],
    screen: [f32; 2],
    prev_view_proj: [[f32; 4]; 4],
) -> FrameUbo {
    let mut volumes = [GpuProbeVolume::default(); MAX_PROBE_VOLUMES];
    let nv = probe_volumes.len().min(MAX_PROBE_VOLUMES);
    volumes[..nv].copy_from_slice(&probe_volumes[..nv]);
    let proj = jitter_proj(camera.proj, jitter, render_size);
    FrameUbo {
        view: camera.view.to_cols_array_2d(),
        proj: proj.to_cols_array_2d(),
        view_proj: (proj * camera.view).to_cols_array_2d(),
        camera_pos: camera.position.extend(1.0).to_array(),
        light_dir: input.light.direction.normalize().extend(0.0).to_array(),
        light_color: [
            input.light.color[0] * input.light.intensity,
            input.light.color[1] * input.light.intensity,
            input.light.color[2] * input.light.intensity,
            1.0,
        ],
        ambient: [
            input.light.ambient[0],
            input.light.ambient[1],
            input.light.ambient[2],
            // .w = skylight/env IBL intensity: gates the fallback skybox reflection
            // so ambient-off → metals don't mirror the sky (scene black, no lights).
            input.light.env_intensity,
        ],
        misc: [input.time, count as f32, input.shadow_pcf_texel, nv as f32],
        cascade_splits,
        lights,
        shadow_vp,
        probe_volumes: volumes,
        debug: [
            if input.lightmap_preview { 1.0 } else { 0.0 },
            input.gi_debug as f32,
            0.0,
            0.0,
        ],
        inv_view: camera.view.inverse().to_cols_array_2d(),
        inv_proj: proj.inverse().to_cols_array_2d(),
        // SSR disabled by default; the renderer sets this per-pass from
        // flux_settings once it knows the depth prepass + colour history are valid.
        ssr: [0.0; 4],
        // Reflection-probe zone; the renderer fills these from input.reflection_probe.
        refl_center: [0.0; 4],
        refl_extents: [0.0; 4],
        refl_box: [0.0; 4],
        refl2_center: [0.0; 4],
        refl2_extents: [0.0; 4],
        refl2_box: [0.0; 4],
        refl_blend: [0.0; 4],
        refl3_center: [0.0; 4],
        refl3_extents: [0.0; 4],
        refl3_box: [0.0; 4],
        refl4_center: [0.0; 4],
        refl4_extents: [0.0; 4],
        refl4_box: [0.0; 4],
        // z = shadow normal-offset scale (texels); x / y are the probe blend distances.
        refl_blend2: [0.0, 0.0, shadow_debug::effective_normal_offset(input.shadow_normal_offset), 0.0],
        // w = shadow receiver-bias scale (world metres per shadow texel at bias 1).
        water: [input.water_time, 0.0, input.water.len().min(MAX_WATER_BODIES) as f32, shadow_debug::effective_bias_scale(input.shadow_bias_scale)],
        water_bodies: water_bodies(input),
        prev_view_proj,
        eye_view: [[[0.0; 4]; 4]; 2],
        eye_view_proj: [[[0.0; 4]; 4]; 2],
        eye_camera_pos: [[0.0; 4]; 2],
        // Fog; the renderer fills these from input.fog.
        fog_color: [0.0; 4],
        fog_params: [0.0; 4],
        postfx0: [
            input.postfx.tonemap as f32,
            input.postfx.exposure,
            input.postfx.grade_exposure,
            input.postfx.contrast,
        ],
        postfx1: [
            input.postfx.saturation,
            input.postfx.temperature,
            input.postfx.tint,
            if input.postfx.grading_enabled { 1.0 } else { 0.0 },
        ],
        postfx2: [
            if input.postfx.vignette_enabled { 1.0 } else { 0.0 },
            input.postfx.vignette_intensity,
            input.postfx.vignette_smoothness,
            screen[0],
        ],
        postfx3: [
            input.postfx.vignette_color[0],
            input.postfx.vignette_color[1],
            input.postfx.vignette_color[2],
            screen[1],
        ],
    }
}

impl Renderer {
    /// Screen-space GI: render a camera depth prepass, then trace the GDF per
    /// pixel into the gather target (sampled by the forward pass). Synchronous
    /// (own submits) for now. `depth_pipe` is the depth-only pipeline; `frame_set`
    /// supplies the camera UBO (binding 0). Returns true if the gather ran.
    /// Record the Flux gather (depth prepass + GDF trace) into the main frame
    /// command buffer `cb`, writing gather image `cur`. No fence wait: the
    /// per-frame trace resources are returned into `sgi_transients[frame_index]`
    /// and freed one frame later, after this frame's fence signals. `cur` is the
    /// ping-pong slot to write; `cur ^ 1` is the history read this frame.
    #[allow(clippy::too_many_arguments)]
    /// Thin wrapper: trace Flux for the editor viewport camera into `self.sgi`,
    /// stashing the per-frame transient for deferred free.
    fn record_screen_gi(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        input: &FrameInput,
        history_valid: bool,
        cur: usize,
        prev_vp: glam::Mat4,
        prev_cam: glam::Vec3,
        alpha: f32,
    ) -> Result<()> {
        let prev = cur ^ 1;
        // Hardware ray-query GI gather when the Ray-Query mode is on AND the device
        // supports it; build/refresh the shared scene TLAS for it (else SDF gather).
        let gi_rt = if self.flux_settings.rt_trace && self.ctx.ray_tracing() {
            let (rt_insts, _) = self.rt_scene(input);
            let rt_water = self.rt_water(input, &rt_insts);
            build_gi_rt_tlas(
                &self.ctx, self.allocator.clone(), self.command_pool, &self.meshes,
                &mut self.rt_reflect, &rt_insts, &rt_water, cb, self.frame_index,
            )
        } else {
            None
        };
        let spare = self.sgi_spare[self.frame_index].take();
        let frame_set = self.frame_sets[self.frame_index];
        let t = self.record_flux_trace(
            device,
            cb,
            frame_set,
            2,
            input,
            &input.camera,
            self.depth.image,
            self.depth.view,
            self.depth.att_view,
            self.sgi.probe_extent,
            self.sgi.gi_image[cur],
            self.sgi.gi_view[cur],
            self.sgi.gi_view[prev],
            self.sgi.gi_filtered,
            self.sgi.gi_filtered_view,
            self.sgi.sh_buf[cur].handle,
            self.sgi.sh_buf[prev].handle,
            self.sgi.raw_rays.handle,
            self.sgi.gi_final,
            self.sgi.gi_final_view,
            self.sgi.temporal(cur),
            gi_rt,
            vk::Extent2D {
                width: self.render_extent[0],
                height: self.render_extent[1],
            },
            history_valid,
            prev_vp,
            prev_cam,
            alpha,
            self.sgi_frame,
            self.screen_gi_flip_y,
            spare,
        )?;
        if let Some(t) = t {
            self.sgi_transients[self.frame_index] = Some(t);
        }
        Ok(())
    }

    /// Depth prepass: render `cam`'s opaque + cutout geometry depth-only into
    /// `depth_image` (cleared), leaving it SHADER_READ_ONLY with the writes
    /// visible to compute and fragment reads. The scene pass then reuses it
    /// (EQUAL test). Used by the Flux trace and the screen-space AO.
    #[allow(clippy::too_many_arguments)]
    fn record_depth_prepass(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        cam_set: vk::DescriptorSet,
        input: &FrameInput,
        cam: &CameraData,
        depth_image: vk::Image,
        depth_view: vk::ImageView,
        depth_att_view: vk::ImageView,
        extent: vk::Extent2D,
    ) -> Result<()> {
        // 1) Depth prepass: render opaque geometry depth-only into depth_image,
        //    recorded directly into the main cb (no separate submit/stall).
        unsafe {
            let to_depth = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(depth_attach_layout())
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .image(depth_image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(depth_aspect())
                        .level_count(1)
                        .layer_count(1),
                );
            // The previous frame read this depth (main pass tests, resolves).
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::FRAGMENT_SHADER
                    | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_depth],
            );
            let depth_att = vk::RenderingAttachmentInfo::default()
                .image_view(depth_att_view)
                .image_layout(depth_attach_layout())
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
                });
            let stencil_att = stencil_attachment(&depth_att);
            let info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
                .layer_count(1)
                .depth_attachment(&depth_att)
                .stencil_attachment(&stencil_att);
            device.cmd_begin_rendering(cb, &info);
            // Match the main pass's negative-height (Y-flip) viewport so the
            // depth buffer aligns with the shaded image.
            let viewport = vk::Viewport {
                x: 0.0,
                y: extent.height as f32,
                width: extent.width as f32,
                height: -(extent.height as f32),
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.cmd_set_viewport(cb, 0, &[viewport]);
            device.cmd_set_scissor(
                cb,
                0,
                &[vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent }],
            );
        }
        // Opaque + cutout geometry this camera can see, through the standard
        // vertex shader (invariant position): the main pass can then reuse this
        // depth with an EQUAL test. Occlusion applies to the main camera only.
        let main_cam = std::ptr::eq(cam, &input.camera);
        let planes = frustum_planes(cam.proj * cam.view);
        // The main camera's GPU-culled batches draw their own share.
        let prepass_order: Vec<usize> = (0..input.draws.len())
            .filter(|&i| {
                let d = &input.draws[i];
                d.shadow_casting.visible()
                    && !(main_cam && (d.occluded || self.gpu_handles(i)))
                    && sphere_in_planes(&planes, d, true)
            })
            .collect();
        // Snapshot the camera's set into a prepass-only copy (see `prepass_sets`).
        let pset = self.prepass_sets[self.frame_index * 2 + if main_cam { 0 } else { 1 }];
        let copies: Vec<vk::CopyDescriptorSet> = (0..=9u32)
            .map(|b| {
                vk::CopyDescriptorSet::default()
                    .src_set(cam_set)
                    .src_binding(b)
                    .dst_set(pset)
                    .dst_binding(b)
                    .descriptor_count(1)
            })
            .chain([ripple_copy(cam_set, pset)])
            .collect();
        unsafe { device.update_descriptor_sets(&[], &copies) };
        self.draw_mode = DrawMode::Prepass;
        if main_cam {
            // The late occlusion pass runs here, so its batches are in the
            // depth the main pass tests EQUAL against.
            self.gpu_late_split(gpu_cull::LateSplit {
                colors: Vec::new(),
                depth: (depth_image, depth_view),
                depth_att: depth_att_view,
                extent,
                xr: None,
                eye: None,
            });
        }
        self.gpu_arm(main_cam);
        let prepass = self.record_scene_draws(device, cb, &prepass_order, input, pset, false);
        self.gpu_arm(false);
        self.draw_mode = DrawMode::Normal;
        prepass?;
        unsafe {
            device.cmd_end_rendering(cb);
            let to_read = vk::ImageMemoryBarrier::default()
                .old_layout(depth_attach_layout())
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .image(depth_image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(depth_aspect())
                        .level_count(1)
                        .layer_count(1),
                );
            // Compute: the Flux gather; fragment: the screen-space AO passes.
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_read],
            );
        }
        Ok(())
    }

    /// Record a Flux trace (depth prepass + GDF gather) for an arbitrary camera
    /// into caller-owned targets, returning the per-frame transient to hold until
    /// the frame fence signals. Shared by the editor viewport and the in-game
    /// camera preview (each passes its own camera + targets).
    #[allow(clippy::too_many_arguments)]
    fn record_flux_trace(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        // Set 0 for this camera (its UBO): the prepass draws with it.
        cam_set: vk::DescriptorSet,
        screen_view: usize,
        input: &FrameInput,
        cam: &CameraData,
        depth_image: vk::Image,
        // Depth-aspect view (sampled by the resolve) and attachment view (the prepass target).
        depth_view: vk::ImageView,
        depth_att_view: vk::ImageView,
        probe_extent: vk::Extent2D,
        gi_out_image: vk::Image,
        gi_out_view: vk::ImageView,
        gi_hist_view: vk::ImageView,
        gi_filtered_image: vk::Image,
        gi_filtered_view: vk::ImageView,
        gi_sh_buf: vk::Buffer,
        gi_sh_prev_buf: vk::Buffer,
        gi_raw_rays: vk::Buffer,
        gi_final_image: vk::Image,
        gi_final_view: vk::ImageView,
        temporal_images: gpu_gi::TemporalImages,
        gi_rt: Option<(vk::AccelerationStructureKHR, vk::Buffer)>,
        full_extent: vk::Extent2D,
        history_valid: bool,
        prev_vp: glam::Mat4,
        prev_cam: glam::Vec3,
        alpha: f32,
        frame_seed: u32,
        flip_y: bool,
        reuse: Option<gpu_gi::ScreenGiTransient>,
    ) -> Result<Option<gpu_gi::ScreenGiTransient>> {
        if self.gpu_gi.as_ref().is_none_or(|g| !g.has_gdf()) {
            return Ok(None);
        }
        // Persistent material captures are shared by Scene and Camera. Only
        // changed geometry/material pages are rasterized, under a fixed budget.
        let surface_allocator = self.allocator();
        {
            let mut alloc = surface_allocator.lock().unwrap();
            if self.flux_surface.is_none() {
                self.flux_surface = Some(flux_surface::SurfaceCache::new(device, &mut alloc, self.pipeline_cache.set1_layout)?);
            }
            let dynamic_textures = self.camera_targets.iter().map(|t| t.texture.0).collect();
            let mesh_sources = self.gpu_gi.as_ref().and_then(|g|g.mesh_sdfs.current.as_ref())
                .map_or(&[][..], |s|s.sources.as_slice());
            let gdf_sources = self.gpu_gi.as_ref().map_or(&[][..],|g|g.gdf_sources());
            self.flux_surface.as_mut().unwrap().record(device, &mut alloc, cb,
                self.frame_index, self.frame_serial, input.time, input.draws,
                &self.meshes, &self.materials, &self.mesh_versions, self.skinner.as_ref(), &dynamic_textures, mesh_sources, gdf_sources)?;
        }
        let extent = full_extent;
        // 1) Depth prepass (shared with the screen-space AO's own prepass).
        self.record_depth_prepass(device, cb, cam_set, input, cam, depth_image, depth_view, depth_att_view, extent)?;

        // 2) Screen-GI gather: trace the GDF per pixel into the gather image.
        let allocator = self.allocator();
        // Reconstruct with the exact matrix used by this depth prepass. The
        // main view UBO includes TAA/upscaler jitter; the camera preview does not.
        let view_proj = if cam_set == self.frame_sets[self.frame_index] {
            self.last_frame_ubo.as_ref().map(|u| Mat4::from_cols_array_2d(&u.view_proj))
                .unwrap_or(cam.proj * cam.view)
        } else {
            cam.proj * cam.view
        };
        let inv_vp = view_proj.inverse();
        let post = if cam_set == self.frame_sets[self.frame_index] {
            input.postfx.flux
        } else {
            input.preview_postfx.flux
        };
        let fs = self.flux_settings.for_view(post);
        let hit_lighting = fs.rt_trace && gi_rt.is_some() && fs.ray_lighting_mode == RayLightingMode::HitLighting;
        let history_valid = history_valid && self.gi_hit_lighting[screen_view] == hit_lighting;
        self.gi_hit_lighting[screen_view] = hit_lighting;
        let screen_resources={
            let mut alloc=allocator.lock().unwrap();
            if self.flux_screen_history.is_none() {self.flux_screen_history=Some(flux_screen_history::ScreenHistory::new(device)?);}
            self.flux_screen_history.as_mut().unwrap().prepare(device,&mut alloc,cb,screen_view,self.frame_serial,self.frame_index,
                full_extent,depth_view,cam.view,view_proj*cam.view.inverse(),flip_y,history_valid && input.gi_debug==0 && !self.flux_settings.mixed_lighting)?
        };

        let amb = input.light.ambient; // dim sky/ambient for escaped rays
        let mixed = self.flux_settings.mixed_lighting;
        let mut lights: Vec<BakeLight> = input
            .lights
            .iter()
            // Mixed lighting: baked lights are already in the lightmaps.
            .filter(|l| !(mixed && l.baked))
            .map(|l| BakeLight {
                kind: l.kind,
                position: l.position,
                direction: l.direction,
                color: [
                    l.color[0] * l.intensity,
                    l.color[1] * l.intensity,
                    l.color[2] * l.intensity,
                ],
                range: l.range,
                spot_inner_deg: l.spot_inner_deg,
                spot_outer_deg: l.spot_outer_deg,
                radius: l.radius.max(0.0),
            })
            .collect();
        // Match the forward pass's legacy key light when no scene lights were
        // supplied. Otherwise it illuminated surfaces without ever bouncing.
        if input.lights.is_empty() && !mixed && input.light.intensity > 0.0 {
            lights.push(BakeLight {
                kind: LightKind::Directional,
                position: Vec3::ZERO,
                direction: input.light.direction,
                color: input.light.color.map(|c| c * input.light.intensity),
                range: 0.0,
                spot_inner_deg: 0.0,
                spot_outer_deg: 0.0,
                radius: 0.0,
            });
        }
        // prev_vp / prev_cam are the PREVIOUS frame's camera (captured before the
        // caller advanced sgi_last_*), so the gather reprojects history correctly;
        // alpha is the motion-aware temporal blend the caller chose.
        // Runtime Flux params (Environment tab → Flux GI). Samples/bounces drive
        // the trace; temporal accumulation smooths low per-frame sample counts.
        // march[1] (max trace distance) of 0 means auto: screen_resolve_record
        // fills it from the GDF diagonal. sky.w carries the indirect intensity;
        // march.w the firefly clamp.
        let speed = post.final_gather_update_speed.filter(|s| s.is_finite()).unwrap_or(1.0).clamp(0.5, 8.0);
        // The final-history-only experiment improves light-change response
        // but increased LongDark motion noise. Keep the accepted defaults;
        // opt into the comparison with ZESTENGINE_FLUX_LEGACY_HISTORY=0.
        // Revisit temporal tuning after core Lumen feature parity.
        static LEGACY_HISTORY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        static PROBE_ALPHA: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
        let legacy = *LEGACY_HISTORY.get_or_init(|| !std::env::var("ZESTENGINE_FLUX_LEGACY_HISTORY").is_ok_and(|v| v == "0"));
        let probe_alpha = *PROBE_ALPHA.get_or_init(|| std::env::var("ZESTENGINE_FLUX_PROBE_HISTORY_ALPHA").ok()
            .and_then(|s| s.parse::<f32>().ok()).filter(|s| s.is_finite()).unwrap_or(1.0).clamp(0.02, 1.0));
        let legacy_alpha = if history_valid { (alpha * speed).clamp(0.02, 1.0) } else { 1.0 };
        // Default smoothing=.5 gives ten frames, scaled by the square root of
        // Final Gather Lighting Update Speed, as in Lumen's final gather.
        let max_frames = ((2.0 + 16.0 * fs.smoothing.clamp(0.0, 1.0)) / speed.sqrt()).round().max(1.0);
        let mut temporal = if legacy { [legacy_alpha, 32.0, legacy_alpha, 0.0] }
            else { [probe_alpha, max_frames, 0.0, 0.0] };
        temporal[3] = if hit_lighting { 3.0 } else { 0.0 };
        let alpha = if history_valid { alpha } else { 1.0 };
        // World SH radiance cache volumes (the realtime/baked probe grid the
        // gather reads at ray hits). Mirror up to GI_MAX_VOLUMES into the params.
        let mut volumes = [gpu_gi::GiVolume {
            world_to_local: [[0.0; 4]; 4],
            size_base: [0.0; 4],
            counts: [0.0; 4],
        }; gpu_gi::GI_MAX_VOLUMES];
        let vcount = self.probe_volumes.len().min(gpu_gi::GI_MAX_VOLUMES);
        for (i, v) in self.probe_volumes.iter().take(gpu_gi::GI_MAX_VOLUMES).enumerate() {
            volumes[i] = gpu_gi::GiVolume {
                world_to_local: v.world_to_local,
                size_base: v.size_base,
                counts: v.counts,
            };
        }
        // Diagnostic gather: green = cached hit, red = hit-shading fallback,
        // blue = escaped ray, yellow = rejected backface. Combine with GI-only
        // view mode 6. Mode2: green=lit, red=missing material card, blue=card
        // without usable lighting; escapes/backfaces are black.
        static CACHE_COVERAGE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
        let cache_coverage = *CACHE_COVERAGE.get_or_init(||
            std::env::var("ZESTENGINE_FLUX_CACHE_COVERAGE").ok().and_then(|s|s.parse::<u32>().ok()).unwrap_or(0).min(2));
        let params = gpu_gi::ScreenGiParams {
            inv_view_proj: inv_vp.to_cols_array_2d(),
            prev_view_proj: prev_vp.to_cols_array_2d(),
            cam: cam.position.extend(1.0).to_array(),
            prev_cam: prev_cam.extend(1.0).to_array(),
            gdf_min: [0.0, 0.0, 0.0, self.frame_serial as f32],
            gdf_max: [0.0; 4],
            // Mixed lighting: the sky is in the bake too.
            sky: if mixed { [0.0, 0.0, 0.0, fs.intensity] } else { [amb[0], amb[1], amb[2], fs.intensity] },
            counts: [fs.samples.max(1), fs.bounces.max(1), 0, 0],
            march: [0.02, fs.march_distance.max(0.0), alpha, fs.firefly_clamp],
            misc: [
                frame_seed,
                full_extent.width,
                full_extent.height,
                if flip_y { 1 } else { 0 },
            ],
            // y = mixed lighting (dynamic-only gather).
            probe_info: [vcount as f32, if mixed { 1.0 } else { 0.0 }, 0.0, cache_coverage as f32],
            volumes,
            temporal,
            screen:screen_resources.params,
            transport:[fs.emissive_boost.max(0.0),0.0,0.0,0.0],
        };
        let lighting_signature = flux_surface::lighting_signature(&lights,&self.gi_emitters,amb,
            fs.rt_trace && gi_rt.is_some(),fs.bounces,mixed,self.gpu_gi.as_ref().unwrap().gdf_revision,fs.emissive_boost,
            &self.packed_area_lights,&self.area_sources);
        let scene_lighting_speed=flux_surface::shared_lighting_speed(input.render_viewport.then_some(input.postfx.flux.scene_lighting_update_speed),
            input.camera_preview.map(|_|input.preview_postfx.flux.scene_lighting_update_speed));
        let lighting_views:Vec<_>=input.render_viewport.then_some(input.camera).into_iter()
            .chain(input.camera_preview).map(flux_surface::LightingView::from).collect();
        let lighting_update = self.flux_surface.as_mut().and_then(|surface| surface.prepare_lighting(
            device,cb,self.frame_index,self.frame_serial,lighting_signature,scene_lighting_speed,&lighting_views));
        // Trace at probe resolution (one probe per SCREEN_PROBE_DIV² pixels);
        // the depth is full-res but sampled by normalized UV, so this is
        // resolution-independent. The forward shader bilinearly upsamples.
        let rt_textures=if gi_rt.is_some() {
            let views=self.rt_texture_views();
            let set=self.gpu_gi.as_mut().unwrap().rt_textures.as_mut().map(|t|t.prepare(device,self.frame_index,&views));
            self.tex_view_cache=views;set
        } else {None};
        let radiance={
            let mut hash=std::collections::hash_map::DefaultHasher::new();
            std::hash::Hash::hash(&(self.flux_surface.as_ref().map_or(0,|s|s.radiance_key()),fs.march_distance.to_bits(),fs.bounces),&mut hash);
            let key=std::hash::Hasher::finish(&hash);
            let mut alloc=allocator.lock().unwrap();
            self.gpu_gi.as_mut().unwrap().radiance_cache.prepare(device,&mut alloc,cb,screen_view,self.frame_index,self.frame_serial,
                cam.position,key,(fs.samples/2).clamp(16,64),!mixed && !hit_lighting && input.gi_debug==0 && cache_coverage==0)?
        };
        let transient = self.gpu_gi.as_ref().unwrap().screen_resolve_record(
            device,
            cb,
            &allocator,
            depth_view,
            self.lightmap_sampler,
            gi_out_image,
            gi_out_view,
            gi_hist_view,
            gi_filtered_image,
            gi_filtered_view,
            gi_sh_buf,
            gi_sh_prev_buf,
            gi_raw_rays,
            gi_final_image,
            gi_final_view,
            temporal_images,
            full_extent,
            probe_extent,
            gi_rt,
            rt_textures,
            &lights,
            &self.packed_area_lights,
            (self.sampler,self.area_atlas.view),
            &self.gi_emitters,
            history_valid,
            self.probe_buffer.handle,
            self.flux_surface.as_ref().and_then(|s| s.resources(self.frame_index)),
            lighting_update,
            screen_resources,
            radiance,
            params,
            reuse,
        )?;
        Ok(transient)
    }

    /// Record the opaque+transparent scene draws (no outline, no egui) for one
    /// pass, binding `frame_set` as set 0. Returns draw statistics.
    /// Effective per-instance material values: the draw's override when set,
    /// else the material's own values. The shader only reads these when the
    /// material opts in (`MaterialFeatures::per_instance`); otherwise the
    /// push-constant lanes stay authoritative.
    fn instance_values(&self, draw: &DrawCmd, material: &Material, input: &FrameInput, fallback: Option<ClusterRefusal>) -> ([f32; 4], [f32; 4]) {
        let p = &material.params;
        let mut color = draw.instance_color.unwrap_or(p.base_color);
        let mut param = draw.instance_param.unwrap_or([p.metallic, p.roughness, p.toon_steps, p.pbr_toon_blend]);
        // Zestlet coverage view only: the fragment shows this colour for a draw on the ordinary path (the
        // draw itself is then not lit, so overriding the lanes changes nothing else). Every other mode skips this.
        if input.gi_debug == VIEW_ZESTLET_COVERAGE {
            let c = self.coverage_colour(draw, fallback);
            color = [c[0], c[1], c[2], 1.0];
            param = [0.0; 4];
        }
        (color, param)
    }

    /// The coverage view's colour of an ordinary draw: the draw-time `fallback` reason (a mesh with a DAG that
    /// the batcher sent to the ordinary path this frame), else the recorded refusal of its mesh, else a
    /// baked-lightmap object of a mesh that has a DAG, else "too small" under the DAG threshold, else
    /// "DAG pending" (big, not built / installed yet), else a per-draw feature (it has a DAG). Grey "other"
    /// is only left for a mesh the renderer does not know.
    fn coverage_colour(&self, draw: &DrawCmd, fallback: Option<ClusterRefusal>) -> [f32; 3] {
        let m = draw.mesh.0;
        if let Some(r) = fallback {
            return r.view_colour();
        }
        if let Some(r) = self.mesh_refusals.get(&m) {
            return r.view_colour();
        }
        let has_dag = self.has_cluster_dag(draw.mesh);
        if draw.lightmap_layer >= 0 && has_dag {
            return ClusterRefusal::LightmapUvs.view_colour();
        }
        match self.meshes.get(m) {
            Some(mesh) if mesh.index_count / 3 < ZESTLET_MIN_TRIANGLES => ClusterRefusal::TooSmall.view_colour(),
            Some(_) if has_dag => ClusterRefusal::DrawFeature.view_colour(),
            Some(_) => ClusterRefusal::Pending.view_colour(),
            None => COVERAGE_OTHER,
        }
    }

    /// Instances sent to the ordinary path this frame by draw-time fallbacks, per reason (index = `ClusterRefusal::index`).
    pub fn draw_fallbacks(&self) -> [u32; REFUSAL_REASONS] {
        self.draw_fallbacks.get()
    }

    /// The refusal recorded for `mesh` (what the coverage view colours it by), if any.
    pub fn mesh_refusal(&self, mesh: MeshHandle) -> Option<ClusterRefusal> {
        self.mesh_refusals.get(&mesh.0).copied()
    }

    /// Record why mesh `mesh` stays on the ordinary path (`None`: it draws as zestlets); the Zestlet
    /// coverage viewport mode colours it by this.
    pub fn set_mesh_refusal(&mut self, mesh: MeshHandle, refusal: Option<ClusterRefusal>) {
        match refusal {
            Some(r) => {
                self.mesh_refusals.insert(mesh.0, r);
            }
            None => {
                self.mesh_refusals.remove(&mesh.0);
            }
        }
    }

    /// Push constants for one scene draw (model + material / per-object lanes).
    fn draw_push(&self, draw: &DrawCmd, material: &Material, input: &FrameInput) -> PushData {
        let p = &material.params;
    if material.custom_shader.is_some() && !material.error {
            let d = &material.custom_data;
            // Custom shaders get 15 prop floats; the 16th lane (d3.w) is
            // reserved for the baked-lightmap layer so custom shaders can
            // sample static GI just like the standard shader (preamble's
            // `zestengine_baked_gi`). -1 = no lightmap → fall back to probe GI.
            let layer = if input.lightmap_preview {
                draw.lightmap_size as f32
            } else {
                draw.lightmap_layer as f32
            };
            PushData {
                model: draw.transform.to_cols_array_2d(),
                base_color: d[0..4].try_into().unwrap(),
                emission: d[4..8].try_into().unwrap(),
                params0: d[8..12].try_into().unwrap(),
                params1: [d[12], d[13], d[14], layer],
            }
        } else {
            // Built-in shaders take the model matrix from the instance stream,
            // so pc.model's first column carries the light mask instead.
            let mask = if self.cull_active { draw_light_mask(&self.cull_lights, draw) } else { [u32::MAX; 4] };
            PushData {
                model: [mask.map(f32::from_bits), [0.0; 4], [0.0; 4], [0.0; 4]],
                base_color: p.base_color,
                emission: [
                    p.emission_color[0] * p.emission_intensity,
                    p.emission_color[1] * p.emission_intensity,
                    p.emission_color[2] * p.emission_intensity,
                    // Planar-reflection lane: 0 = none, else layer+1 in the
                    // integer part and the roughness fade cutoff in the
                    // fraction. Gated in-shader on frame.refl_box.w so the
                    // planar/capture/preview passes never self-sample.
                    if draw.planar_index >= 0
                        && (draw.planar_index as usize) < input.planar.len()
                    {
                        let cutoff = input.planar[draw.planar_index as usize]
                            .max_roughness
                            .clamp(0.05, 0.95);
                        (draw.planar_index + 1) as f32 + cutoff
                    } else {
                        0.0
                    },
                ],
                params0: [p.metallic, p.roughness, p.toon_steps, p.pbr_toon_blend],
                // w = baked-lightmap layer (-1 = none); the standard shader
                // samples it for static GI. (Selection highlight is drawn by
                // the separate outline pass, so w is free here.)
                params1: [
                    p.alpha_cutoff,
                    p.normal_strength,
                    p.occlusion_strength,
                    // Normally the lightmap layer; in UV-checker preview the
                    // object's lightmap resolution (the shader reads it then).
                    if input.lightmap_preview {
                        draw.lightmap_size as f32
                    } else {
                        draw.lightmap_layer as f32
                    },
                ],
            }
        }
    }

    /// Record one (possibly instanced) scene draw, binding only what changed.
    fn flush_scene_batch(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        frame_set: vk::DescriptorSet,
        b: &SceneBatch,
        bound: &mut BoundDraw,
        stats: &mut RenderStats,
    ) {
        let material = &self.materials[b.material];
        let mesh = &self.meshes[b.mesh];
        unsafe {
            if b.pipeline != bound.pipeline {
                stats.pipeline_binds += 1;
                device_bind(device, cb, b.pipeline);
                device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_cache.layout,
                    0,
                    &[frame_set],
                    &[],
                );
                bound.pipeline = b.pipeline;
            }
            if material.set != bound.material {
                device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_cache.layout,
                    1,
                    &[material.set],
                    &[],
                );
                if material.custom_shader.is_some() && !material.error && material.custom_set != vk::DescriptorSet::null() {
                    device.cmd_bind_descriptor_sets(
                        cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.pipeline_cache.layout,
                        2,
                        &[material.custom_set],
                        &[],
                    );
                }
                bound.material = material.set;
            }
            device.cmd_push_constants(
                cb,
                self.pipeline_cache.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&b.push),
            );
            if b.mesh != bound.mesh {
                device.cmd_bind_vertex_buffers(cb, 0, &[mesh.vertex_buffer.handle], &[0]);
                // Velocity variants also read the previous frame's position
                // (binding 2): the skinner's prev output for skinned meshes,
                // else the same buffer (static geometry doesn't move).
                if self.velocity_active {
                    let prev = self
                        .skinner
                        .as_ref()
                        .and_then(|s| s.prev_vertex_buffer(b.mesh))
                        .unwrap_or(mesh.vertex_buffer.handle);
                    device.cmd_bind_vertex_buffers(cb, 2, &[prev], &[0]);
                }
                device.cmd_bind_index_buffer(cb, mesh.index_buffer.handle, 0, mesh.index_type);
                bound.mesh = b.mesh;
            }
            if let Some((cmds, cmd_offset, instances, instance_offset)) = b.indirect {
                device.cmd_bind_vertex_buffers(cb, 1, &[instances], &[instance_offset]);
                device.cmd_draw_indexed_indirect(cb, cmds, cmd_offset, 1, size_of::<vk::DrawIndexedIndirectCommand>() as u32);
            } else {
                device.cmd_draw_indexed(cb, mesh.index_count, b.count, 0, 0, b.first);
            }
        }
        stats.draw_calls += 1;
        if b.count > 1 {
            stats.instanced_draws += 1;
        }
    }

    fn record_scene_draws(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        order: &[usize],
        input: &FrameInput,
        frame_set: vk::DescriptorSet,
        gbuf: bool,
    ) -> Result<RenderStats> {
        let mut stats = RenderStats::default();
        let mut materials_seen = std::mem::take(&mut self.materials_seen_scratch);
        materials_seen.clear();
        materials_seen.resize(self.materials.len(), false);
        let mut bound = BoundDraw::default();
        // Cube-face capture (positive-height viewport) and planar mirrors
        // (reflected camera) reverse the on-screen winding: use the mirrored
        // pipeline variants (clockwise front face). Forcing no-cull instead
        // left `gl_FrontFacing` false on every outward face, so the shader
        // flipped every normal and captured surfaces were lit from behind.
        let mirrored = self.capturing_refl;
        let mode = self.draw_mode;
        // GPU-culled batches armed for this pass draw first (none are
        // transparent, so the CPU order's sorting still holds after them).
        if let Err(e) =
            self.record_gpu_scene_batches(device, cb, frame_set, gbuf, &mut bound, &mut stats, &mut materials_seen)
        {
            self.materials_seen_scratch = materials_seen;
            return Err(e);
        }
        // Instance matrices for every draw in `order`: draw k reads instance k
        // (first_instance), so a run of draws sharing pipeline, material set,
        // mesh and fragment push data becomes one instanced draw.
        if order.is_empty() {
            self.materials_seen_scratch = materials_seen;
            return Ok(stats);
        }
        let instances: Vec<InstanceData> = order
            .iter()
            .map(|&i| {
                let d = &input.draws[i];
                let material = &self.materials[d.material.0];
                let (color, param) = self.instance_values(d, material, input, None);
                InstanceData { model: d.transform.to_cols_array_2d(), color, param }
            })
            .collect();
        let (inst_buf, inst_off) = self.push_instances(&instances)?;
        unsafe { device.cmd_bind_vertex_buffers(cb, 1, &[inst_buf], &[inst_off]) };
        let mut batch: Option<SceneBatch> = None;
        for (k, &i) in order.iter().enumerate() {
            let k = k as u32;
            let draw = &input.draws[i];
            let material = &self.materials[draw.material.0];
            let transparent = !material.error && matches!(material.features.alpha_mode, AlphaMode::Blend | AlphaMode::Transclipping);
            // Transparent draws never write depth; a stencil writer with ZWrite off must not
            // occlude anything either, so neither takes part in the prepass.
            if mode == DrawMode::Prepass && (transparent || !material.features.stencil.depth_write) {
                continue;
            }

            if material.error {
                stats.error_draws += 1;
            } else if matches!(material.features.alpha_mode, AlphaMode::Blend | AlphaMode::Transclipping) {
                stats.transparent_draws += 1;
            } else {
                stats.opaque_draws += 1;
            }
            if let Some(seen) = materials_seen.get_mut(draw.material.0)
                && !*seen
            {
                *seen = true;
                stats.materials_drawn += 1;
            }

            let key = scene_pipeline_key(material, gbuf, self.velocity_active, mirrored, mode, transparent);
            let pipeline = self.pipeline_cache.get(device, key)?;
            let material = &self.materials[draw.material.0];
            let push = self.draw_push(draw, material, input);
            // Custom fragment code may read pc.model, which only carries the
            // first instance's matrix, so custom materials draw one at a time.
            let batchable = material.custom_shader.is_none() || material.error;
            let extend = batch.as_ref().is_some_and(|b| {
                batchable
                    && b.batchable
                    && b.pipeline == pipeline
                    // Cross-material instancing: the material handle may differ,
                    // but the draw can still share one instanced batch when it
                    // resolves to the same bound set + same fragment lanes.
                    && self.materials[b.material].set == material.set
                    && b.mesh == draw.mesh.0
                    && b.first + b.count == k
                    && b.push.base_color == push.base_color
                    && b.push.emission == push.emission
                    && b.push.params0 == push.params0
                    && b.push.params1 == push.params1
            });
            if extend {
                let b = batch.as_mut().unwrap();
                b.count += 1;
                // One push for the whole instanced run: light the union.
                for w in 0..4 {
                    b.push.model[0][w] = f32::from_bits(b.push.model[0][w].to_bits() | push.model[0][w].to_bits());
                }
            } else {
                if let Some(b) = batch.take() {
                    self.flush_scene_batch(device, cb, frame_set, &b, &mut bound, &mut stats);
                }
                batch = Some(SceneBatch {
                    pipeline,
                    material: draw.material.0,
                    mesh: draw.mesh.0,
                    push,
                    first: k,
                    count: 1,
                    batchable,
                    indirect: None,
                });
            }

            let outline = if material.error || transparent || mode == DrawMode::Prepass {
                None
            } else {
                material.outline
            };
            if let Some((color, width)) = outline {
                if let Some(b) = batch.take() {
                    self.flush_scene_batch(device, cb, frame_set, &b, &mut bound, &mut stats);
                }
                let mut okey = PipelineKey::outline();
                okey.gbuf = gbuf;
                okey.mirrored = mirrored;
                let pipeline = self.pipeline_cache.get(device, okey)?;
                device_bind(device, cb, pipeline);
                let push = PushData {
                    model: draw.transform.to_cols_array_2d(),
                    base_color: color,
                    emission: [0.0; 4],
                    params0: draw.mesh_center.extend(0.0).to_array(),
                    // A negative width asks for the red vertex colour to scale it.
                    params1: [1.0, if width < 0.0 { 1.0 } else { 0.0 }, 0.0, width.abs() * 0.2],
                };
                unsafe {
                    device.cmd_push_constants(
                        cb,
                        self.pipeline_cache.layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        bytemuck::bytes_of(&push),
                    );
                    device.cmd_draw_indexed(cb, self.meshes[draw.mesh.0].index_count, 1, 0, 0, k);
                }
                stats.draw_calls += 1;
                bound.pipeline = vk::Pipeline::null();
            }

            // Fur shells: the mesh again per shell (one instanced draw each, the shell index
            // in the push constants), after the surface and its outline.
            let fur = if material.error || transparent || mode == DrawMode::Prepass { None } else { material.fur };
            if let Some(f) = fur {
                if let Some(b) = batch.take() {
                    self.flush_scene_batch(device, cb, frame_set, &b, &mut bound, &mut stats);
                }
                let mut fkey = PipelineKey::fur();
                fkey.gbuf = gbuf;
                fkey.mirrored = mirrored;
                let pipeline = self.pipeline_cache.get(device, fkey)?;
                device_bind(device, cb, pipeline);
                let layers = f.layers.clamp(2, 48);
                for shell in 1..=layers {
                    let push = PushData {
                        model: draw.transform.to_cols_array_2d(),
                        base_color: [f.tip_color[0], f.tip_color[1], f.tip_color[2], 1.0],
                        emission: [f.root_color[0], f.root_color[1], f.root_color[2], 0.0],
                        params0: [shell as f32 / layers as f32, f.length, f.density, 0.0],
                        params1: [0.0; 4],
                    };
                    unsafe {
                        device.cmd_push_constants(
                            cb,
                            self.pipeline_cache.layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            0,
                            bytemuck::bytes_of(&push),
                        );
                        device.cmd_draw_indexed(cb, self.meshes[draw.mesh.0].index_count, 1, 0, 0, k);
                    }
                }
                stats.draw_calls += layers;
                bound.pipeline = vk::Pipeline::null();
            }
        }
        if let Some(b) = batch.take() {
            self.flush_scene_batch(device, cb, frame_set, &b, &mut bound, &mut stats);
        }
        self.materials_seen_scratch = materials_seen;
        Ok(stats)
    }

    /// Record the offscreen main-camera pass into the preview target. Assumes
    /// the preview exists and its UBO for this frame has been written.
    #[allow(clippy::too_many_arguments)]
    fn record_camera_preview(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        order: &[usize],
        input: &FrameInput,
        pcam_view: &CameraData,
        flux: bool,
        clear_color: [f32; 4],
        // `(grid, tile, draws per view)`: draw `order` as `grid^2` consecutive runs, each in its own tile viewport.
        tiles: Option<(u32, u32, usize)>,
    ) -> Result<()> {
        // Flux GI for the in-game camera: trace from the preview (game) camera
        // into the preview's own targets, before its color pass, so the preview
        // shows screen-space GI like the main viewport. Folded into the same cb;
        // its own temporal state + transient ring (freed a frame later).
        if flux
            && self.gi_has_gdf()
            && let Some(pcam) = Some(pcam_view)
        {
            let (
                sgi_depth_image,
                sgi_depth_view,
                sgi_depth_att_view,
                sgi_probe_extent,
                sgi_gi_image,
                sgi_gi_view,
                sgi_filtered_image,
                sgi_filtered_view,
                sgi_sh_buf,
                sgi_raw_rays,
                sgi_final_image,
                sgi_final_view,
                pextent,
            ) = {
                let p = self.camera_preview.as_ref().unwrap();
                (
                    p.sgi.depth_image,
                    p.sgi.depth_view,
                    p.sgi.depth_att_view,
                    p.sgi.probe_extent,
                    p.sgi.gi_image,
                    p.sgi.gi_view,
                    p.sgi.gi_filtered,
                    p.sgi.gi_filtered_view,
                    [p.sgi.sh_buf[0].handle, p.sgi.sh_buf[1].handle],
                    p.sgi.raw_rays.handle,
                    p.sgi.gi_final,
                    p.sgi.gi_final_view,
                    p.extent,
                )
            };
            let preview_set = self.camera_preview.as_ref().unwrap().sets[self.frame_index];
            let vp = pcam.proj * pcam.view;

            let cur = self.preview_sgi_parity & 1;
            let prev = cur ^ 1;
            let prev_vp = self.preview_sgi_last_view_proj.unwrap_or(vp);
            let prev_cam = if self.preview_sgi_last_view_proj.is_some() {
                self.preview_sgi_last_cam
            } else {
                pcam.position
            };
            let hist_valid = self.preview_sgi_history_valid;
            let s = self.flux_settings.smoothing.clamp(0.0, 1.0);
            let alpha = if hist_valid { 0.08 - 0.04 * s } else { 1.0 };
            self.preview_sgi_frame = self.preview_sgi_frame.wrapping_add(1);
            let seed = self.preview_sgi_frame;
            let flip = self.screen_gi_flip_y;
            let spare = self.preview_sgi_spare[self.frame_index].take();
            self.gpu_zone_begin(cb, gpu_zone::CAMERA_GI);
            let gi_rt = if self.flux_settings.rt_trace && self.ctx.ray_tracing() {
                let (rt_insts, _) = self.rt_scene(input);
                let rt_water = self.rt_water(input, &rt_insts);
                build_gi_rt_tlas(
                    &self.ctx, self.allocator.clone(), self.command_pool, &self.meshes,
                    &mut self.rt_reflect, &rt_insts, &rt_water, cb, self.frame_index,
                )
            } else {
                None
            };
            match self.record_flux_trace(
                device,
                cb,
                preview_set,
                1,
                input,
                pcam,
                sgi_depth_image,
                sgi_depth_view,
                sgi_depth_att_view,
                sgi_probe_extent,
                sgi_gi_image[cur],
                sgi_gi_view[cur],
                sgi_gi_view[prev],
                sgi_filtered_image,
                sgi_filtered_view,
                sgi_sh_buf[cur],
                sgi_sh_buf[prev],
                sgi_raw_rays,
                sgi_final_image,
                sgi_final_view,
                self.camera_preview.as_ref().unwrap().sgi.temporal(cur),
                gi_rt,
                pextent,
                hist_valid,
                prev_vp,
                prev_cam,
                alpha,
                seed,
                flip,
                spare,
            ) {
                Ok(t) => {
                    if let Some(t) = t {
                        self.preview_sgi_transients[self.frame_index] = Some(t);
                    }
                    self.preview_sgi_history_valid = true;
                    self.preview_sgi_last_view_proj = Some(vp);
                    self.preview_sgi_last_cam = pcam.position;
                    self.preview_sgi_parity ^= 1;
                    // Point the preview's binding 4 at the INTEGRATED full-res GI.
                    let view = self.camera_preview.as_ref().unwrap().sgi.temporal(cur).current_views[0];
                    let set = self.camera_preview.as_ref().unwrap().sets[self.frame_index];
                    let info = [vk::DescriptorImageInfo::default()
                        .sampler(self.lightmap_sampler)
                        .image_view(view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                    let write = vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(4)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&info);
                    unsafe { device.update_descriptor_sets(&[write], &[]) };
                }
                Err(e) => tracing::warn!("preview Flux trace failed: {e:#}"),
            }
            self.gpu_zone_end(cb, gpu_zone::CAMERA_GI);
        }

        // Model thumbnails use this renderer too; only the real Camera view
        // enables scene reflections. Their settings are independent of GI.
        let fs = self.flux_settings.for_view(input.preview_postfx.flux);
        let rt_on = flux && fs.reflection_mode == 2 && self.rt_reflect.is_some() && self.tier.reflections();
        let ssr_on = flux && fs.reflection_mode == 1 && fs.ssr_enabled && self.tier.reflections();
        let reflect = rt_on || ssr_on;
        if reflect && self.camera_preview.as_ref().unwrap().reflections.is_empty() {
            let extent = self.camera_preview.as_ref().unwrap().extent;
            let allocator = self.allocator();
            let mut alloc = allocator.lock().unwrap();
            let mut targets = Vec::new();
            for _ in 0..FRAMES_IN_FLIGHT {
                targets.push(self.ssr_resolve.create_target(device, &mut alloc, self.post_pass.set_layout(), extent)?);
            }
            self.camera_preview.as_mut().unwrap().reflections = targets;
        }
        let preview = self.camera_preview.as_ref().unwrap();
        let (gbuf_image, gbuf_view) = if reflect {
            let target = &preview.reflections[self.frame_index];
            (target.gbuf_image, target.gbuf_view)
        } else { (vk::Image::null(), vk::ImageView::null()) };
        let extent = preview.extent;
        let color = preview.color;
        let color_view = preview.color_view;
        let hdr_image = preview.hdr_image;
        let hdr_view = preview.hdr_view;
        let post_set = preview.post_set;
        let preview_bloom = preview.bloom.as_ref().unwrap().output_set();
        let depth = preview.depth;
        let depth_view = preview.depth_view;
        let depth_att_view = preview.depth_att_view;
        let frame_set = preview.sets[self.frame_index];

        unsafe {
            image_barrier(
                device,
                cb,
                hdr_image,
                vk::ImageAspectFlags::COLOR,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            );
            image_barrier(
                device,
                cb,
                depth,
                depth_aspect(),
                vk::ImageLayout::UNDEFINED,
                depth_attach_layout(),
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );

            if reflect {
                image_barrier(device, cb, gbuf_image, vk::ImageAspectFlags::COLOR,
                    vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::MEMORY_WRITE,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE);
            }
            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(hdr_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: clear_color,
                    },
                });
            // Screen GI history also needs opaque depth, even with reflections
            // and underwater fog disabled. Discarding it rejects Camera hits.
            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(depth_att_view)
                .image_layout(depth_attach_layout())
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(if reflect || (flux && self.flux_screen_history.as_ref().is_some_and(|h|h.ready(1,self.frame_serial))) || input.preview_postfx.underwater.is_some() || self.preview_keep_depth {
                    vk::AttachmentStoreOp::STORE
                } else {
                    vk::AttachmentStoreOp::DONT_CARE
                })
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                });
            let stencil_attachment_info = stencil_attachment(&depth_attachment);
            let mut color_attachments = vec![color_attachment];
            if reflect {
                color_attachments.push(vk::RenderingAttachmentInfo::default()
                    .image_view(gbuf_view).image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR).store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } }));
            }
            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                })
                .layer_count(1)
                .color_attachments(&color_attachments)
                .depth_attachment(&depth_attachment)
                .stencil_attachment(&stencil_attachment_info);
            device.cmd_begin_rendering(cb, &rendering_info);

            let viewport = vk::Viewport {
                x: 0.0,
                y: extent.height as f32,
                width: extent.width as f32,
                height: -(extent.height as f32),
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.cmd_set_viewport(cb, 0, &[viewport]);
            device.cmd_set_scissor(
                cb,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                }],
            );
        }

        self.pipeline_cache.set_target_format(post::HDR_FORMAT);
        if input.draw_skybox {
            self.record_skybox(device, cb, frame_set, true, reflect)?;
        }
        if let Some((grid, tile, per_view)) = tiles {
            for (k, run) in order.chunks(per_view.max(1)).enumerate() {
                let (tx, ty) = ((k as u32 % grid) * tile, (k as u32 / grid) * tile);
                unsafe {
                    device.cmd_set_viewport(
                        cb,
                        0,
                        &[vk::Viewport { x: tx as f32, y: (ty + tile) as f32, width: tile as f32, height: -(tile as f32), min_depth: 0.0, max_depth: 1.0 }],
                    );
                    device.cmd_set_scissor(cb, 0, &[vk::Rect2D { offset: vk::Offset2D { x: tx as i32, y: ty as i32 }, extent: vk::Extent2D { width: tile, height: tile } }]);
                }
                self.record_scene_draws(device, cb, run, input, frame_set, reflect)?;
            }
        } else {
        // Culled to the preview camera's frustum (occlusion is for the main camera).
        let preview_order: Vec<usize> = {
            let planes = frustum_planes(pcam_view.proj * pcam_view.view);
            order.iter().copied().filter(|&i| sphere_in_planes(&planes, &input.draws[i], true)).collect()
        };
        let capture=flux && input.gi_debug==0 && self.flux_screen_history.as_ref().is_some_and(|h|h.ready(1,self.frame_serial));
        if capture {
            let split=preview_order.iter().position(|&i|self.materials[input.draws[i].material.0].render_queue>=RENDER_QUEUE_TRANSPARENT).unwrap_or(preview_order.len());
            self.record_scene_draws(device,cb,&preview_order[..split],input,frame_set,reflect)?;
            unsafe {device.cmd_end_rendering(cb);}
            let pass=MainPass {color_image:hdr_image,color_view:hdr_view,gbuf_view:reflect.then_some(gbuf_view),velocity_view:None,reactive_view:None,
                depth_image:depth,depth_view:depth_att_view,depth_read_view:depth_view,extent};
            self.capture_flux_history(device,cb,&pass,1);
            Self::resume_main_pass(device,cb,&pass);
            self.record_scene_draws(device,cb,&preview_order[split..],input,frame_set,reflect)?;
        } else {self.record_scene_draws(device, cb, &preview_order, input, frame_set, reflect)?;}
        }

        unsafe { device.cmd_end_rendering(cb) };
        let preview_cam = *pcam_view;
        if let Some(fx) = input.preview_postfx.underwater.as_ref() {
            self.underwater.record(
                device, cb, underwater::SLOT_PREVIEW + self.frame_index as u32,
                (hdr_image, hdr_view), (depth, depth_view), extent,
                &preview_cam, fx, &input.light, input.time, 0,
            );
        }
        // HDR -> shader read, then the post pass resolves it into `color`.
        image_barrier(
            device, cb, hdr_image, vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_SAMPLED_READ,
        );
        let post_set = if reflect {
            image_barrier(device, cb, depth, depth_aspect(),
                depth_attach_layout(), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ);
            image_barrier(device, cb, gbuf_image, vk::ImageAspectFlags::COLOR,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ);
            let uniforms = self.ssr_uniforms_for_view(input, extent, ssr_on, &preview_cam, input.preview_postfx.flux);
            if rt_on {
                let (instances, lights) = self.rt_scene(input);
                let water = self.rt_water(input, &instances);
                let textures = self.rt_texture_views();
                let u = uniforms;
                let rt_uniforms = rt_reflect::RtUniforms {
                    inv_proj: u.inv_proj, inv_view: u.inv_view, camera_pos: u.camera_pos,
                    refl_center: u.refl_center, refl_extents: u.refl_extents, refl_box: u.refl_box,
                    params: [fs.ssr_roughness_cutoff, fs.ssr_intensity, lights.len() as f32, input.light.env_intensity],
                    screen: [extent.width as f32, extent.height as f32, 0.0, fs.bounces as f32],
                    quality: rt_quality(&fs, self.frame_serial),
                    lighting_mode: [fs.ray_lighting_mode as u32, 0, 0, 0],
                    refl2_center: u.refl2_center, refl2_extents: u.refl2_extents, refl2_box: u.refl2_box,
                    refl3_center: u.refl3_center, refl3_extents: u.refl3_extents, refl3_box: u.refl3_box,
                    refl4_center: u.refl4_center, refl4_extents: u.refl4_extents, refl4_box: u.refl4_box,
                    refl_blend: u.refl_blend, refl_blend2: u.refl_blend2,
                    water: [input.water_time, input.water.len().min(MAX_WATER_BODIES) as f32, 0.0, 0.0],
                    water_bodies: water_bodies(input),
                };
                let allocator = self.allocator();
                let target = &self.camera_preview.as_ref().unwrap().reflections[self.frame_index];
                self.rt_reflect.as_mut().unwrap().record(&self.ctx, &allocator, self.command_pool, cb,
                    self.frame_index, 1, self.frame_serial, &self.meshes, &instances, &water, &lights,
                    target.resolved_image(), target.resolved_view(), hdr_view, depth_view, gbuf_view,
                    self.reflection_env.view, self.reflection_env_b.view, self.reflection_env_c.view, self.reflection_env_d.view,
                    &textures, (self.lightmaps.view, self.lightmap_sampler), (self.area_atlas.view,self.sampler),
                    self.flux_surface.as_ref().and_then(|s| s.reflection_resources(self.frame_index, self.frame_serial)),
                    &rt_uniforms, extent)?;
                self.tex_view_cache = textures;
                image_barrier(device, cb, target.resolved_image(), vk::ImageAspectFlags::COLOR,
                    vk::ImageLayout::GENERAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_WRITE,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ);
            } else {
                let target = &mut self.camera_preview.as_mut().unwrap().reflections[self.frame_index];
                self.ssr_resolve.record(device, cb, target, hdr_view, depth_view,
                    self.reflection_env.view, self.reflection_env_b.view, self.reflection_env_c.view, self.reflection_env_d.view,
                    &uniforms, self.frame_index);
            }
            self.camera_preview.as_ref().unwrap().reflections[self.frame_index].post_set
        } else { post_set };
        self.post_pass.exposure.record(device,cb,post_set,extent,input.preview_postfx.auto_exposure,1,
            self.frame_serial,self.started.elapsed().as_secs_f64());
        {
            let chain = self.camera_preview.as_ref().unwrap().bloom.as_ref().unwrap();
            self.post_pass.record_bloom(device, cb, chain, post_set, &input.preview_postfx);
        }
        unsafe {
            image_barrier(
                device, cb, color, vk::ImageAspectFlags::COLOR,
                vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            );
            let post_att = [vk::RenderingAttachmentInfo::default()
                .image_view(color_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)];
            device.cmd_begin_rendering(
                cb,
                &vk::RenderingInfo::default()
                    .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
                    .layer_count(1)
                    .color_attachments(&post_att),
            );
        }
        let wobble = underwater::wobble(input.preview_postfx.underwater.as_ref(), &preview_cam, input.time);
        self.post_pass.record_set(device, cb, post_set, preview_bloom, extent, &input.preview_postfx, wobble);
        unsafe {
            device.cmd_end_rendering(cb);
            // Color attachment -> shader read for egui to sample this frame.
            image_barrier(
                device,
                cb,
                color,
                vk::ImageAspectFlags::COLOR,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
            );
        }
        Ok(())
    }

    /// Render every shadow view into its atlas layer, then transition the
    /// whole atlas to a sampleable depth-read layout. Always leaves the atlas
    /// readable (even with zero casters) so the scene pass can sample it.
    /// The shadow layers to re-render this frame, with their casters.
    fn shadow_work(&mut self, views: &[ShadowView], draws: &[DrawCmd]) -> Vec<(usize, Mat4, Vec<usize>)> {
        let dim = self.shadow_atlas.dim;
        // All eligible shadow casters (transparent objects and Cast Shadows =
        // Off don't cast), shared by every view. The GPU cull (gpu_cull.comp)
        // frustum-culls them per view, so the shadow pass culls on the GPU like
        // the main camera; the CPU fallback in record_shadows culls the same
        // list. The cache signature below is computed over this full caster
        // input rather than a per-view frustum subset, so it stays consistent
        // with what the GPU cull actually tests. A layer whose signature is
        // unchanged keeps last frame's depth (static lights over static
        // geometry stop re-rendering every frame).
        let casters: Vec<usize> = (0..draws.len())
            .filter(|&i| shadow_caster(&draws[i], &self.materials[draws[i].material.0].features))
            .collect();
        if shadow_debug::due() {
            let vps: Vec<(usize, Mat4)> = views.iter().map(|v| (v.layer, v.view_proj)).collect();
            let materials = &self.materials;
            shadow_debug::report(
                &vps,
                draws.len(),
                &casters,
                self.last_stats.gpu_cull_shadow_visible,
                &|i| {
                    let d = &draws[i];
                    let f = &materials[d.material.0].features;
                    if !d.shadow_casting.casts() {
                        Some("Cast Shadows is Off")
                    } else if matches!(f.alpha_mode, AlphaMode::Blend | AlphaMode::Transclipping) {
                        Some("blended material")
                    } else if f.water {
                        Some("water material")
                    } else {
                        None
                    }
                },
                &|i, vp| sphere_in_planes(&frustum_planes(*vp), &draws[i], false),
            );
        }
        let mut work: Vec<(usize, Mat4, Vec<usize>)> = Vec::new();
        // The same hook-mix as the signature below, per caster, hashed once for every view.
        let caster_hash: Vec<u64> = casters
            .iter()
            .map(|&i| {
                let mut h: u64 = 0xcbf29ce484222325;
                let mut mix = |v: u64| {
                    h ^= v;
                    h = h.wrapping_mul(0x100000001b3);
                };
                mix(draws[i].mesh.0 as u64);
                mix(draws[i].shadow_casting as u64);
                // Deformed (skinned / CPU-updated) vertices re-render too.
                mix(self.mesh_versions.get(draws[i].mesh.0).copied().unwrap_or(0) as u64);
                // A vertex hook animates with time and a displaced surface follows the camera's
                // tessellation: those casters re-render their layers every frame.
                if self.shadow_special(&draws[i]) {
                    mix(self.frame_serial);
                }
                for f in draws[i].transform.to_cols_array() {
                    mix(f.to_bits() as u64);
                }
                h
            })
            .collect();
        for view in views {
            let mut sig: u64 = 0xcbf29ce484222325 ^ dim as u64;
            for f in view.view_proj.to_cols_array() {
                sig ^= f.to_bits() as u64;
                sig = sig.wrapping_mul(0x100000001b3);
            }
            // Only casters that can reach this view's volume (the near plane excluded: a caster between
            // the light and the volume still shadows it) are drawn, and only they decide whether the
            // layer's cached depth is still valid: an avatar animating outside every cascade costs
            // nothing here, in the GPU cull, in the pose upload or in the shadow pass.
            let planes = frustum_planes(view.view_proj);
            let mut view_casters = Vec::with_capacity(casters.len());
            for (k, &i) in casters.iter().enumerate() {
                if sphere_in_planes(&planes, &draws[i], false) {
                    sig ^= caster_hash[k];
                    sig = sig.wrapping_mul(0x100000001b3);
                    view_casters.push(i);
                }
            }
            if !self.shadow_layers_valid || self.shadow_layer_sig[view.layer] != sig {
                self.shadow_layer_sig[view.layer] = sig;
                work.push((view.layer, view.view_proj, view_casters));
            }
        }
        work
    }

    /// Whether caster `d` needs its own shadow draw: its material runs a vertex hook (so the
    /// shadow must show the displaced surface) or is displaced by tessellation on a device
    /// that has it. Everything else shares the one shadow pipeline's batches.
    pub(crate) fn shadow_special(&self, d: &DrawCmd) -> bool {
        let m = &self.materials[d.material.0];
        if m.error {
            return false;
        }
        (m.params.tess.enabled && self.pipeline_cache.tessellation())
            || m.custom_shader.is_some_and(|s| self.pipeline_cache.has_hook(s.0))
    }

    /// The shadow pipeline variant of a special caster's material: the hooked vertex module
    /// of its shader and / or the tessellation stages, with the material's keywords.
    fn shadow_key_for(&self, m: &Material) -> PipelineKey {
        let mut key = PipelineKey::shadow();
        key.tess = m.params.tess.enabled;
        if let Some(s) = m.custom_shader
            && self.pipeline_cache.has_hook(s.0)
        {
            key.shader = s.0 as u32 + 1;
            key.spec = m.custom_keywords;
        }
        key
    }

    /// Draw the special casters of `casters` into the shadow layer being recorded
    /// (rendering begun): grouped by material and mesh, with the material's descriptor sets
    /// bound (the vertex hook reads set 2, the tessellation stages set 0 and 1).
    fn record_special_shadow_casters(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        light_vp: Mat4,
        casters: &[usize],
        draws: &[DrawCmd],
    ) -> Result<()> {
        let mut special: Vec<usize> = casters.iter().copied().filter(|&i| self.shadow_special(&draws[i])).collect();
        if special.is_empty() {
            return Ok(());
        }
        special.sort_unstable_by_key(|&i| (draws[i].material.0, draws[i].mesh.0));
        let instances: Vec<InstanceData> =
            special.iter().map(|&i| InstanceData::from_model(draws[i].transform.to_cols_array_2d())).collect();
        let (inst_buf, inst_off) = self.push_instances(&instances)?;
        let frame_set = self.frame_sets[self.frame_index];
        let push = ShadowPush { light_vp: light_vp.to_cols_array_2d(), model: Mat4::IDENTITY.to_cols_array_2d() };
        unsafe {
            device.cmd_push_constants(
                cb,
                self.pipeline_cache.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push),
            );
            device.cmd_bind_vertex_buffers(cb, 1, &[inst_buf], &[inst_off]);
        }
        let mut first = 0usize;
        while first < special.len() {
            let (material_index, mesh_index) = (draws[special[first]].material.0, draws[special[first]].mesh.0);
            let run = special[first..]
                .iter()
                .take_while(|&&i| draws[i].material.0 == material_index && draws[i].mesh.0 == mesh_index)
                .count();
            let key = self.shadow_key_for(&self.materials[material_index]);
            let pipeline = self.pipeline_cache.get(device, key)?;
            let material = &self.materials[material_index];
            let mesh = &self.meshes[mesh_index];
            unsafe {
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
                device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline_cache.layout, 0, &[frame_set], &[]);
                device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline_cache.layout, 1, &[material.set], &[]);
                if material.custom_shader.is_some() && material.custom_set != vk::DescriptorSet::null() {
                    device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline_cache.layout, 2, &[material.custom_set], &[]);
                }
                device.cmd_bind_vertex_buffers(cb, 0, &[mesh.vertex_buffer.handle], &[0]);
                device.cmd_bind_index_buffer(cb, mesh.index_buffer.handle, 0, mesh.index_type);
                device.cmd_draw_indexed(cb, mesh.index_count, run as u32, 0, 0, first as u32);
            }
            first += run;
        }
        Ok(())
    }

    /// Render the shadow layers in `work` (see `shadow_work`): GPU-culled
    /// indirect batches when this frame's cull covered the layer, else the
    /// casters grouped by mesh into instanced draws.
    fn record_shadows(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        work: &[(usize, Mat4, Vec<usize>)],
        draws: &[DrawCmd],
    ) -> Result<()> {
        let image = self.shadow_atlas.image;
        let dim = self.shadow_atlas.dim;
        let range = |layer: u32, count: u32| {
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                .level_count(1)
                .base_array_layer(layer)
                .layer_count(count)
        };
        let to_attach = |r: vk::ImageSubresourceRange, old: vk::ImageLayout| {
            vk::ImageMemoryBarrier2::default()
                .image(image)
                .old_layout(old)
                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                // Previous frames sample these layers: wait for those reads.
                .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags2::empty())
                .dst_stage_mask(
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                )
                .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .subresource_range(r)
        };
        let to_read = |r: vk::ImageSubresourceRange| {
            vk::ImageMemoryBarrier2::default()
                .image(image)
                .old_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
                .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .subresource_range(r)
        };
        let barrier = |b: &[vk::ImageMemoryBarrier2]| unsafe {
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(b));
        };
        // First use (or a resized atlas): bring every layer into a known state.
        let first = !self.shadow_layers_valid;
        if first {
            barrier(&[to_attach(range(0, self.shadow_atlas.layers as u32), vk::ImageLayout::UNDEFINED)]);
        } else if !work.is_empty() {
            let b: Vec<_> = work
                .iter()
                .map(|(layer, _, _)| to_attach(range(*layer as u32, 1), vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL))
                .collect();
            barrier(&b);
        }

        if !work.is_empty() {
            let pipeline = self.pipeline_cache.get(device, PipelineKey::shadow())?;
            let extent = vk::Extent2D { width: dim, height: dim };
            for (layer, view_proj, casters) in work {
                let depth_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(self.shadow_atlas.layer_views[*layer])
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
                    });
                let rendering_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
                    .layer_count(1)
                    .depth_attachment(&depth_attachment);
                unsafe {
                    device.cmd_begin_rendering(cb, &rendering_info);
                    // Positive-height viewport: shadow uv = ndc*0.5+0.5 matches.
                    let viewport = vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: dim as f32,
                        height: dim as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    };
                    device.cmd_set_viewport(cb, 0, &[viewport]);
                    device.cmd_set_scissor(cb, 0, &[vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent }]);
                }
                // Cluster casters first (own pipeline), then the ordinary one.
                self.draw_cluster_shadow_layer(device, cb, *layer, *view_proj)?;
                unsafe { device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline) };
                if !self.record_gpu_shadow_layer(device, cb, *layer, *view_proj) {
                    // CPU fallback: the GPU cull path frustum-culls casters in
                    // gpu_cull.comp; do the same here before grouping by mesh
                    // (each run is one instanced draw).
                    let planes = frustum_planes(*view_proj);
                    let mut sorted: Vec<usize> = casters
                        .iter()
                        .copied()
                        .filter(|&i| sphere_in_planes(&planes, &draws[i], false))
                        .filter(|&i| !self.cluster.as_ref().is_some_and(|c| c.shadow_is_handled(*layer, i)))
                        .filter(|&i| !self.shadow_special(&draws[i]))
                        .collect();
                    sorted.sort_unstable_by_key(|&i| draws[i].mesh.0);
                    let instances: Vec<InstanceData> = sorted
                        .iter()
                        .map(|&i| InstanceData::from_model(draws[i].transform.to_cols_array_2d()))
                        .collect();
                    let (inst_buf, inst_off) = self.push_instances(&instances)?;
                    let push = ShadowPush {
                        light_vp: view_proj.to_cols_array_2d(),
                        model: Mat4::IDENTITY.to_cols_array_2d(),
                    };
                    unsafe {
                        device.cmd_push_constants(
                            cb,
                            self.pipeline_cache.layout,
                            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                            0,
                            bytemuck::bytes_of(&push),
                        );
                        device.cmd_bind_vertex_buffers(cb, 1, &[inst_buf], &[inst_off]);
                        let mut first = 0usize;
                        while first < sorted.len() {
                            let mesh_index = draws[sorted[first]].mesh.0;
                            let run = sorted[first..].iter().take_while(|&&i| draws[i].mesh.0 == mesh_index).count();
                            let mesh = &self.meshes[mesh_index];
                            device.cmd_bind_vertex_buffers(cb, 0, &[mesh.vertex_buffer.handle], &[0]);
                            device.cmd_bind_index_buffer(cb, mesh.index_buffer.handle, 0, mesh.index_type);
                            device.cmd_draw_indexed(cb, mesh.index_count, run as u32, 0, 0, first as u32);
                            first += run;
                        }
                    }
                }
                // Casters with a vertex hook or a displaced surface: their own draws (the GPU
                // batches and the cluster shadows leave them out).
                self.record_special_shadow_casters(device, cb, *view_proj, casters, draws)?;
                unsafe { device.cmd_end_rendering(cb) };
            }
        }

        // Rendered layers (or the whole atlas the first time) -> depth-read.
        if first {
            barrier(&[to_read(range(0, self.shadow_atlas.layers as u32))]);
            self.shadow_layers_valid = true;
        } else if !work.is_empty() {
            let b: Vec<_> = work.iter().map(|(layer, _, _)| to_read(range(*layer as u32, 1))).collect();
            barrier(&b);
        }
        Ok(())
    }

    /// egui texture id + pixel size of the model preview target, once a frame
    /// with `FrameInput::model_preview` has created it.
    pub fn model_preview_texture(&self) -> Option<(egui::TextureId, [f32; 2])> {
        self.model_preview.as_ref().map(|p| (p.texture_id, [p.extent.width as f32, p.extent.height as f32]))
    }

    /// Count down retired model previews (one step per frame) and free the
    /// expired ones; `all` frees everything (shutdown, after idle).
    fn free_retired_model_previews(&mut self, all: bool) {
        for (_, frames) in &mut self.retired_model_previews {
            *frames = frames.saturating_sub(1);
        }
        if self.retired_model_previews.iter().all(|(_, f)| *f > 0 && !all) {
            return;
        }
        let Some(allocator) = self.allocator.clone() else {
            return;
        };
        let mut alloc = allocator.lock().unwrap();
        let mut i = 0;
        while i < self.retired_model_previews.len() {
            if all || self.retired_model_previews[i].1 == 0 {
                let (mut p, _) = self.retired_model_previews.swap_remove(i);
                if let Some(egui) = self.egui.as_mut() {
                    egui.remove_user_texture(p.texture_id);
                }
                p.destroy(&self.ctx.device, &mut alloc);
            } else {
                i += 1;
            }
        }
    }

    /// egui texture id + pixel size of the main-camera preview, if it exists.
    pub fn camera_preview_texture(&self) -> Option<(egui::TextureId, [f32; 2])> {
        self.camera_preview.as_ref().map(|p| {
            (
                p.texture_id,
                [p.extent.width as f32, p.extent.height as f32],
            )
        })
    }

    /// egui texture for the editor viewport's offscreen render (RTT), if active.
    /// Count down retired viewport targets (one step per rendered frame) and
    /// free the expired ones; `all` frees everything (shutdown, after idle).
    fn free_retired_viewport_targets(&mut self, all: bool) {
        if self.retired_viewport_targets.is_empty() && self.retired_viewport_ssr.is_empty() {
            return;
        }
        for (_, frames) in &mut self.retired_viewport_targets {
            *frames = frames.saturating_sub(1);
        }
        for (_, frames) in &mut self.retired_viewport_ssr {
            *frames = frames.saturating_sub(1);
        }
        let Some(allocator) = self.allocator.clone() else {
            return;
        };
        let mut alloc = allocator.lock().unwrap();
        let mut i = 0;
        while i < self.retired_viewport_targets.len() {
            if all || self.retired_viewport_targets[i].1 == 0 {
                let (mut t, _) = self.retired_viewport_targets.swap_remove(i);
                if let Some(egui) = self.egui.as_mut() {
                    egui.remove_user_texture(t.texture_id);
                }
                t.destroy(&self.ctx.device, &mut alloc);
            } else {
                i += 1;
            }
        }
        let mut i = 0;
        while i < self.retired_viewport_ssr.len() {
            if all || self.retired_viewport_ssr[i].1 == 0 {
                let (mut t, _) = self.retired_viewport_ssr.swap_remove(i);
                t.destroy(&self.ctx.device, &mut alloc);
            } else {
                i += 1;
            }
        }
    }

    /// Record the per-pass GPU timestamp zones (`RenderStats::gpu_*_ms`,
    /// `ClusterStats::*_ms`) without the profiler window; costs a few
    /// timestamp writes and a larger readback per frame. For benchmarks.
    pub fn set_gpu_zone_timing(&mut self, on: bool) {
        self.zone_timing = on;
    }

    /// How the editor viewport draws the selection (colour, width in pixels,
    /// on/off). Takes effect next frame; draws with `highlight > 0` are outlined.
    pub fn set_selection_outline(&mut self, style: SelectionOutlineStyle) {
        self.selection.style = style;
    }

    /// The editor viewport's ground grid (on/off, plane, unit, colours). Only
    /// the editor viewport target draws it, never the game, the Camera tab
    /// preview, captures or XR. Takes effect next frame.
    pub fn set_editor_grid(&mut self, style: EditorGridStyle) {
        self.grid.style = style;
    }

    /// Wait for the GPU and read back the editor viewport's final colour image
    /// (the target `FrameInput::viewport_extent` renders into, after the post
    /// pass) as tightly packed RGBA8 in the target's encoding (sRGB when the
    /// swapchain format is). For tests and captures: it stalls the pipeline.
    /// Returns (width, height, pixels); None without a viewport target.
    pub fn debug_read_viewport(&mut self) -> Result<Option<(u32, u32, Vec<u8>)>> {
        let Some(vt) = self.viewport_target.as_ref() else { return Ok(None) };
        let (w, h, image) = (vt.extent.width, vt.extent.height, vt.color);
        self.read_back_color(w, h, image).map(Some)
    }

    /// [`Self::debug_read_viewport`] for the Camera tab preview target
    /// (`FrameInput::camera_preview`); None while it does not exist.
    pub fn debug_read_camera_preview(&mut self) -> Result<Option<(u32, u32, Vec<u8>)>> {
        let Some(p) = self.camera_preview.as_ref() else { return Ok(None) };
        let (w, h, image) = (p.extent.width, p.extent.height, p.color);
        self.read_back_color(w, h, image).map(Some)
    }

    /// [`Self::debug_read_viewport`] for the model preview target
    /// (`FrameInput::model_preview`); None while it does not exist.
    pub fn debug_read_model_preview(&mut self) -> Result<Option<(u32, u32, Vec<u8>)>> {
        let Some(p) = self.model_preview.as_ref() else { return Ok(None) };
        let (w, h, image) = (p.extent.width, p.extent.height, p.color);
        self.read_back_color(w, h, image).map(Some)
    }

    /// Test hook: start (`true`) / stop (`false`, drops pending captures) copying
    /// every frame's final viewport colour into its own host buffer, recorded at
    /// the end of the frame's command buffer (no stall, so frames overlap exactly
    /// as in the editor). Collect them with [`Self::debug_take_viewport_frames`].
    pub fn debug_capture_viewport_frames(&mut self, on: bool) {
        if on {
            self.viewport_captures.get_or_insert_with(Vec::new);
        } else if let Some(old) = self.viewport_captures.take() {
            let _ = unsafe { self.ctx.device.wait_idle("wait:device_wait_idle@debug_capture") };
            let allocator = self.allocator();
            let mut alloc = allocator.lock().unwrap();
            for (mut b, ..) in old {
                b.destroy(&self.ctx.device, &mut alloc);
            }
        }
    }

    /// Wait for the GPU and return the frames captured since the last call, oldest
    /// first, as tightly packed RGBA8 (see [`Self::debug_read_viewport`]).
    pub fn debug_take_viewport_frames(&mut self) -> Result<Vec<(u32, u32, Vec<u8>)>> {
        let Some(captures) = self.viewport_captures.as_mut() else { return Ok(Vec::new()) };
        let captures = std::mem::take(captures);
        let bgra = matches!(self.swapchain.format.format, vk::Format::B8G8R8A8_SRGB | vk::Format::B8G8R8A8_UNORM);
        unsafe { self.ctx.device.wait_idle("wait:device_wait_idle@debug_take_viewport_frames")? };
        let allocator = self.allocator();
        let mut alloc = allocator.lock().unwrap();
        let mut out = Vec::new();
        for (mut b, w, h) in captures {
            let mut px = b.mapped().map(|m| m[..(w * h * 4) as usize].to_vec()).unwrap_or_default();
            if bgra {
                for p in px.as_chunks_mut::<4>().0 {
                    p.swap(0, 2);
                }
            }
            out.push((w, h, px));
            b.destroy(&self.ctx.device, &mut alloc);
        }
        Ok(out)
    }

    /// Probe hook: copy the composed swapchain image (egui over the viewport, as
    /// presented) while viewport capture is on. Needs TRANSFER_SRC on the swapchain.
    fn record_swapchain_capture(&mut self, device: &ash::Device, cb: vk::CommandBuffer, image: vk::Image, extent: vk::Extent2D) {
        if self.viewport_captures.is_none() || !self.swapchain.can_copy_from {
            return;
        }
        let (w, h) = (extent.width, extent.height);
        let allocator = self.allocator();
        let mut alloc = allocator.lock().unwrap();
        let Ok(buf) = alloc::Buffer::new(
            device, &mut alloc, w as u64 * h as u64 * 4, vk::BufferUsageFlags::TRANSFER_DST, gpu_allocator::MemoryLocation::GpuToCpu, "swapchain capture",
        ) else {
            return;
        };
        image_barrier(
            device, cb, image, vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::COPY, vk::AccessFlags2::TRANSFER_READ,
        );
        let region = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
            .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
        unsafe { device.cmd_copy_image_to_buffer(cb, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buf.handle, &[region]) };
        image_barrier(
            device, cb, image, vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::COPY, vk::AccessFlags2::TRANSFER_READ,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        );
        self.swapchain_captures.push((buf, w, h));
    }

    /// Wait for the GPU and return the composed swapchain frames captured so far (RGBA8).
    pub fn debug_take_swapchain_frames(&mut self) -> Result<Vec<(u32, u32, Vec<u8>)>> {
        let captures = std::mem::take(&mut self.swapchain_captures);
        let bgra = matches!(self.swapchain.format.format, vk::Format::B8G8R8A8_SRGB | vk::Format::B8G8R8A8_UNORM);
        unsafe { self.ctx.device.wait_idle("wait:device_wait_idle@debug_take_swapchain_frames")? };
        let allocator = self.allocator();
        let mut alloc = allocator.lock().unwrap();
        let mut out = Vec::new();
        for (mut b, w, h) in captures {
            let mut px = b.mapped().map(|m| m[..(w * h * 4) as usize].to_vec()).unwrap_or_default();
            if bgra {
                for p in px.as_chunks_mut::<4>().0 {
                    p.swap(0, 2);
                }
            }
            out.push((w, h, px));
            b.destroy(&self.ctx.device, &mut alloc);
        }
        Ok(out)
    }

    fn record_viewport_capture(&mut self, device: &ash::Device, cb: vk::CommandBuffer) {
        if self.viewport_captures.is_none() {
            return;
        }
        let Some(vt) = self.viewport_target.as_ref() else { return };
        let (w, h, image) = (vt.extent.width, vt.extent.height, vt.color);
        let allocator = self.allocator();
        let mut alloc = allocator.lock().unwrap();
        let Ok(buf) = alloc::Buffer::new(
            device, &mut alloc, w as u64 * h as u64 * 4, vk::BufferUsageFlags::TRANSFER_DST, gpu_allocator::MemoryLocation::GpuToCpu, "viewport capture",
        ) else {
            return;
        };
        image_barrier(
            device, cb, image, vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::MEMORY_WRITE,
            vk::PipelineStageFlags2::COPY, vk::AccessFlags2::TRANSFER_READ,
        );
        let region = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
            .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
        unsafe { device.cmd_copy_image_to_buffer(cb, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buf.handle, &[region]) };
        image_barrier(
            device, cb, image, vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::COPY, vk::AccessFlags2::TRANSFER_READ,
            vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ,
        );
        if let Some(v) = self.viewport_captures.as_mut() {
            v.push((buf, w, h));
        }
    }

    fn read_back_color(&mut self, w: u32, h: u32, image: vk::Image) -> Result<(u32, u32, Vec<u8>)> {
        let bgra = matches!(self.swapchain.format.format, vk::Format::B8G8R8A8_SRGB | vk::Format::B8G8R8A8_UNORM);
        let device = &self.ctx.device;
        unsafe { device.wait_idle("wait:device_wait_idle@debug_read_viewport")? };
        let allocator = self.allocator();
        let mut alloc = allocator.lock().unwrap();
        let bytes = w as u64 * h as u64 * 4;
        let mut buf = alloc::Buffer::new(
            device,
            &mut alloc,
            bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::GpuToCpu,
            "viewport readback",
        )?;
        let barrier = |cb: vk::CommandBuffer, from: vk::ImageLayout, to: vk::ImageLayout| {
            image_barrier(
                device, cb, image, vk::ImageAspectFlags::COLOR, from, to,
                vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::MEMORY_WRITE,
                vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            )
        };
        let done = alloc::one_time_submit(device, self.command_pool, self.ctx.queue, |cb| {
            barrier(cb, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                .image_extent(vk::Extent3D { width: w, height: h, depth: 1 });
            unsafe { device.cmd_copy_image_to_buffer(cb, image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, buf.handle, &[region]) };
            barrier(cb, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        });
        let out = done.map(|()| {
            let mut px = buf.mapped().map(|m| m[..bytes as usize].to_vec()).unwrap_or_default();
            if bgra {
                for p in px.as_chunks_mut::<4>().0 {
                    p.swap(0, 2);
                }
            }
            (w, h, px)
        });
        buf.destroy(device, &mut alloc);
        out
    }

    pub fn viewport_texture(&self) -> Option<(egui::TextureId, [f32; 2])> {
        self.viewport_target.as_ref().map(|t| {
            (t.texture_id, [t.extent.width as f32, t.extent.height as f32])
        })
    }

    /// Build the deferred-SSR resolve uniforms for a camera + frame. Matches the
    /// values `frame_ubo` writes (plain matrix inverses), so the resolve re-derives
    /// the exact env reflection the forward pass added.
    fn ssr_uniforms(
        &self,
        input: &FrameInput,
        ext: vk::Extent2D,
        ssr_on: bool,
    ) -> ssr::SsrUniforms {
        self.ssr_uniforms_for_view(input, ext, ssr_on, &input.camera, input.postfx.flux)
    }

    fn ssr_uniforms_for_view(&self, input: &FrameInput, ext: vk::Extent2D, ssr_on: bool,
        cam: &CameraData, post: FluxPostFx) -> ssr::SsrUniforms {
        let fs = self.flux_settings.for_view(post);
        let ssr = if ssr_on {
            [1.0, fs.ssr_intensity, fs.ssr_max_distance, fs.ssr_roughness_cutoff]
        } else {
            [0.0; 4]
        };
        let mut refl_center = [0.0; 4];
        let mut refl_extents = [0.0; 4];
        let mut refl_box = [0.0; 4];
        if let Some(p) = input.reflection_probe {
            refl_center = [p.capture[0], p.capture[1], p.capture[2], p.intensity.max(0.0)];
            refl_extents = [
                p.half_extents[0],
                p.half_extents[1],
                p.half_extents[2],
                if p.box_projection { 1.0 } else { 0.0 },
            ];
            refl_box = [p.center[0], p.center[1], p.center[2], 0.0];
        }
        let mut fog_color = [0.0; 4];
        let mut fog_params = [0.0; 4];
        if let Some(f) = input.fog {
            fog_color = [f.color[0], f.color[1], f.color[2], f.density.max(0.0)];
            fog_params = [f.height_falloff.max(0.0), f.height_ref, f.start_distance.max(0.0), 0.0];
        }
        let l = &input.light;
        let sun = (-l.direction).normalize_or(Vec3::Y);
        let refl2 = refl2_uniforms(input);
        ssr::SsrUniforms {
            proj: cam.proj.to_cols_array_2d(),
            inv_proj: cam.proj.inverse().to_cols_array_2d(),
            view: cam.view.to_cols_array_2d(),
            inv_view: cam.view.inverse().to_cols_array_2d(),
            camera_pos: cam.position.extend(0.0).to_array(),
            ssr,
            refl_center,
            refl_extents,
            refl_box,
            fog_color,
            fog_params,
            // Sun colour*intensity + forward-scatter anisotropy; sun dir + time.
            fog_light: [
                l.color[0] * l.intensity,
                l.color[1] * l.intensity,
                l.color[2] * l.intensity,
                0.6,
            ],
            fog_sun: [sun.x, sun.y, sun.z, input.time],
            screen: [ext.width as f32, ext.height as f32, 0.0, 0.0],
            refl2_center: refl2.refl2_center,
            refl2_extents: refl2.refl2_extents,
            refl2_box: refl2.refl2_box,
            refl3_center: refl2.refl3_center,
            refl3_extents: refl2.refl3_extents,
            refl3_box: refl2.refl3_box,
            refl4_center: refl2.refl4_center,
            refl4_extents: refl2.refl4_extents,
            refl4_box: refl2.refl4_box,
            refl_blend: refl2.refl_blend,
            refl_blend2: refl2.refl_blend2,
        }
    }

    /// Draw order for a camera at `cam_pos`.
    fn draw_order(&self, input: &FrameInput, cam_pos: glam::Vec3) -> Vec<usize> {
        // Sort by one precomputed key per draw. Primary: render queue (Unity-
        // style). Opaque queues then group by pipeline state, material and mesh
        // (fewer binds); transparent queues sort back-to-front so blends composite.
        // Shadows Only draws are left out here, so no camera pass built from
        // this order (main, captures, mirrors, previews, VR) ever draws them.
        let mut keyed: Vec<(u64, u64, usize)> = input
            .draws
            .iter()
            .enumerate()
            .filter(|(_, d)| d.shadow_casting.visible())
            .map(|(i, d)| {
                let m = &self.materials[d.material.0];
                let queue = m.render_queue as u64;
                let secondary = if m.render_queue >= RENDER_QUEUE_TRANSPARENT {
                    // The bounds' world centre: draws baked in world space
                    // (identity transform) would all sort as the origin.
                    let dist2 = (d.transform.transform_point3(d.mesh_center) - cam_pos).length_squared();
                    u64::MAX - dist2.to_bits() as u64
                } else {
                    let f = &m.features;
                    let state = (f.toon as u64)
                        | (f.normal_map as u64) << 1
                        | (f.emission as u64) << 2
                        | (f.double_sided as u64) << 3
                        | (m.error as u64) << 4
                        | (f.water as u64) << 5
                        | m.custom_shader.map_or(0, |c| c.0 as u64 + 1) << 6;
                    (state << 40) | ((d.material.0 as u64 & 0xF_FFFF) << 20) | (d.mesh.0 as u64 & 0xF_FFFF)
                };
                (queue, secondary, i)
            })
            .collect();
        keyed.sort_unstable();
        keyed.into_iter().map(|(_, _, i)| i).collect()
    }

    /// The ray-traced scene data for this frame, built once and shared by the
    /// Flux-RT GI gather and the reflection dispatch.
    fn rt_scene(&mut self, input: &FrameInput) -> (Vec<rt_reflect::RtInstance>, Vec<rt_reflect::RtLight>) {
        if let Some((serial, insts, lights)) = &self.rt_scene_cache
            && *serial == self.frame_serial
        {
            return (insts.clone(), lights.clone());
        }
        let built = self.build_rt_scene(input);
        self.rt_scene_cache = Some((self.frame_serial, built.0.clone(), built.1.clone()));
        built
    }

    /// This frame's water for the RT scene: the clock, bodies and each body's
    /// ripple field (from its water instance). Water geometry is displaced
    /// only while RT reflections are on (nothing else traces it).
    fn rt_water(&self, input: &FrameInput, instances: &[rt_reflect::RtInstance]) -> rt_water::RtWaterFrame {
        let mut ripples = [vk::ImageView::null(); MAX_WATER_BODIES];
        for inst in instances.iter().filter(|i| i.water >= 0) {
            let slot = (inst.water as usize).min(MAX_WATER_BODIES - 1);
            if let Some(t) = usize::try_from(inst.textures[0]).ok().and_then(|t| self.textures.get(t)) {
                ripples[slot] = t.view;
            }
        }
        // The shore-depth texture is bound to the water material (slot 14) and
        // indexed by the material's water slot; the RT instance table only
        // carries the ripple field (slot 15), so pull the shore view from the
        // material itself.
        let mut shores = [vk::ImageView::null(); MAX_WATER_BODIES];
        for d in input.draws {
            let Some(m) = self.materials.get(d.material.0) else { continue };
            if !m.features.water {
                continue;
            }
            let slot = (m.water_slot as usize).min(MAX_WATER_BODIES - 1);
            if let Some(t) = m.slots[14].and_then(|t| self.textures.get(t.0)) {
                shores[slot] = t.view;
            }
        }
        rt_water::RtWaterFrame {
            animate: self.flux_settings.reflection_mode == 2,
            time: input.water_time,
            bodies: water_bodies(input),
            ripples,
            shores,
        }
    }

    /// Texture views for the RT texture table (refreshed when textures are added).
    fn rt_texture_views(&mut self) -> Vec<vk::ImageView> {
        if self.tex_view_cache.len() != self.textures.len() {
            self.tex_view_cache = self.textures.iter().map(|t| t.view).collect();
        }
        std::mem::take(&mut self.tex_view_cache)
    }

    /// Build the ray-traced-reflection scene data (instances + lights) for a frame.
    fn build_rt_scene(&self, input: &FrameInput) -> (Vec<rt_reflect::RtInstance>, Vec<rt_reflect::RtLight>) {
        let instances = input
            .draws
            .iter()
            .enumerate()
            .filter_map(|(source_draw,d)| {
                let m = self.materials.get(d.material.0)?;
                let p = &m.params;
                if m.features.water {
                    // Shaded from its body in the RT pass; the ripple field rides
                    // in the albedo texture lane. The CPU rewrites its vertices,
                    // so its BLAS follows the mesh version.
                    return Some(rt_reflect::RtInstance {
                        mesh: d.mesh.0,
                        source_draw: source_draw as u32,
                        transform: d.transform,
                        albedo: [1.0; 3],
                        albedo_mean: [1.0; 3],
                        metallic_mean: 1.0,
                        emission: [0.0; 3],
                        emission_raw: [0.0; 3],
                        textures: [m.slots[15].map_or(-1, |t| t.0 as i32), -1, -1, -1],
                        albedo_st: [1.0, 1.0, 0.0, 0.0],
                        normal_st: [1.0, 1.0, 0.0, 0.0],
                        orm_st: [1.0, 1.0, 0.0, 0.0],
                        emission_st: [1.0, 1.0, 0.0, 0.0],
                        base_alpha: 1.0,
                        alpha_cutoff: -1.0,
                        opacity_texture: -1,
                        extra_textures: [-1;4], detail_textures: [-1;4], detail: [0.0;4],
                        detail_st: [1.0,1.0,0.0,0.0], orm_invert: [0.0;4], emission_mask_st: [1.0,1.0,0.0,0.0],
                        metallic: 0.0,
                        roughness: 0.05,
                        normal_strength: 1.0,
                        lightmap_layer: -1,
                        static_geometry: false,
                        water: m.water_slot.min(MAX_WATER_BODIES as u32 - 1) as i32,
                        dynamic: Some(self.mesh_versions.get(d.mesh.0).copied().unwrap_or(0)),
                        shadow_casting: d.shadow_casting,
                    });
                }
                let base=if m.features.per_instance {d.instance_color.unwrap_or(p.base_color)} else {p.base_color};
                let param=if m.features.per_instance {d.instance_param.unwrap_or([p.metallic,p.roughness,0.0,0.0])} else {[p.metallic,p.roughness,0.0,0.0]};
                let fx=MaterialFx::from_params(p);
                let emission_scale=if !m.features.emission {0.0} else {p.emission_intensity*
                    if p.emission_pulse>0.0 {0.5+0.5*(input.time*p.emission_pulse).sin()} else {1.0}};
                Some(rt_reflect::RtInstance {
                    mesh: d.mesh.0,
                    source_draw: source_draw as u32,
                    transform: d.transform,
                    albedo: [base[0], base[1], base[2]],
                    albedo_mean: m.albedo_mean,
                    metallic_mean: m.metallic_mean,
                    // Scale by the emission-map mean so a sparse map (e.g. only a
                    // glowing visor) doesn't bounce as a uniform full-mesh light
                    // in the RT GI gather, which shares this instance buffer.
                    emission: [
                        p.emission_color[0] * emission_scale * p.emission_map_mean[0],
                        p.emission_color[1] * emission_scale * p.emission_map_mean[1],
                        p.emission_color[2] * emission_scale * p.emission_map_mean[2],
                    ],
                    emission_raw: [
                        p.emission_color[0] * emission_scale,
                        p.emission_color[1] * emission_scale,
                        p.emission_color[2] * emission_scale,
                    ],
                    textures: [
                        m.slots[0].map_or(-1, |t| t.0 as i32),
                        if m.features.normal_map { m.slots[1].map_or(-1, |t| t.0 as i32) } else { -1 },
                        m.slots[2].map_or(-1, |t| t.0 as i32),
                        if m.features.emission { m.slots[3].map_or(-1, |t| t.0 as i32) } else { -1 },
                    ],
                    albedo_st: [p.albedo_tiling[0], p.albedo_tiling[1], p.albedo_offset[0]+p.base_scroll[0]*input.time, p.albedo_offset[1]+p.base_scroll[1]*input.time],
                    normal_st: [p.normal_tiling[0], p.normal_tiling[1], p.normal_offset[0], p.normal_offset[1]],
                    orm_st: [p.orm_tiling[0], p.orm_tiling[1], p.orm_offset[0], p.orm_offset[1]],
                    emission_st: [p.emission_tiling[0], p.emission_tiling[1], p.emission_offset[0]+p.emission_scroll[0]*input.time, p.emission_offset[1]+p.emission_scroll[1]*input.time],
                    base_alpha: base[3],
                    alpha_cutoff: if matches!(m.features.alpha_mode, AlphaMode::Cutout|AlphaMode::Transclipping) {p.alpha_cutoff.max(0.0)} else {-1.0},
                    opacity_texture: m.slots[4].map_or(-1,|t|t.0 as i32),
                    extra_textures: [m.slots[13].map_or(-1,|t|t.0 as i32),m.slots[14].map_or(-1,|t|t.0 as i32),m.slots[5].map_or(-1,|t|t.0 as i32),-1],
                    detail_textures: [16,17,18,19].map(|i|m.slots[i].map_or(-1,|t|t.0 as i32)),
                    detail: fx.detail, detail_st: fx.detail_st, orm_invert: fx.orm_invert, emission_mask_st: fx.emission_st,
                    metallic: param[0],
                    roughness: param[1],
                    normal_strength: p.normal_strength,
                    lightmap_layer: d.lightmap_layer,
                    static_geometry: d.static_geometry,
                    water: -1,
                    dynamic: None,
                    shadow_casting: d.shadow_casting,
                })
            })
            .collect();
        let mut lights: Vec<rt_reflect::RtLight> = input
            .lights
            .iter()
            .map(|l| {
                let kind = match l.kind {
                    LightKind::Directional => 0.0,
                    LightKind::Point => 1.0,
                    LightKind::Spot => 2.0,
                };
                rt_reflect::RtLight {
                    position: [l.position.x, l.position.y, l.position.z, kind],
                    direction: [l.direction.x, l.direction.y, l.direction.z, l.range],
                    color: [
                        l.color[0] * l.intensity,
                        l.color[1] * l.intensity,
                        l.color[2] * l.intensity,
                        0.0,
                    ],
                    spot: [
                        (l.spot_inner_deg.to_radians() * 0.5).cos(),
                        (l.spot_outer_deg.to_radians() * 0.5).cos(),
                        if l.baked { 1.0 } else { 0.0 },
                        0.0,
                    ],
                }
            })
            .collect();
        lights.extend(self.packed_area_lights.iter().map(|l|rt_reflect::RtLight {
            position:l.pos_kind,direction:l.dir_range,color:l.color,spot:l.spot,
        }));
        (instances, lights)
    }

    fn ensure_vr_overlay(&mut self) -> Result<()> {
        if self.vr_overlay.is_some() {
            return Ok(());
        }
        let allocator = self
            .allocator
            .clone()
            .context("renderer has no allocator for VR overlay")?;
        let mut alloc = allocator.lock().unwrap();
        let ov = vr_overlay::VrOverlay::new(
            &self.ctx.device,
            &mut alloc,
            self.swapchain.format.format,
            self.swapchain.extent,
        )?;
        self.vr_overlay = Some(ov);
        Ok(())
    }

    /// Render the editor's egui UI into the VR overlay texture (the left-hand
    /// panel samples it). Uses the dedicated VR egui renderer so it can't race the
    /// desktop egui draw. Call after `render` (textures are uploaded there), then
    /// pass a [`VrOverlayDraw`] to `render_xr_eye` to show it. Synchronous.
    pub fn render_vr_ui(&mut self, draw: &EguiDraw) -> Result<()> {
        self.ensure_vr_overlay()?;
        let (ui_image, ui_view, ui_extent) = {
            let ov = self.vr_overlay.as_ref().unwrap();
            (ov.ui_image(), ov.ui_view(), ov.ui_extent())
        };
        let device = self.ctx.device.clone();
        let Some(vr_egui) = self.vr_egui.as_mut() else {
            return Ok(());
        };
        let cb = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0]
        };
        let record = (|| -> Result<()> {
            unsafe {
                device.begin_command_buffer(
                    cb,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
            }
            image_barrier(
                &device, cb, ui_image, vk::ImageAspectFlags::COLOR,
                vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            );
            let att = vk::RenderingAttachmentInfo::default()
                .image_view(ui_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] },
                });
            let atts = [att];
            let info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: ui_extent })
                .layer_count(1)
                .color_attachments(&atts);
            unsafe {
                device.cmd_begin_rendering(cb, &info);
                // Standard (top-left) viewport: egui's own coordinate convention.
                let viewport = vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: ui_extent.width as f32,
                    height: ui_extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                device.cmd_set_viewport(cb, 0, &[viewport]);
                device.cmd_set_scissor(cb, 0, &[vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: ui_extent }]);
            }
            // The VR egui renderer only holds the managed (font/shape) textures,
            // not the editor's USER textures — the embedded 3D viewport image, the
            // camera preview, file thumbnails — which are registered with the main
            // renderer. A primitive referencing one fails the whole draw ("Bad
            // texture ID: User(..)"), blanking the panel. Drop those (the embedded
            // 3D view is redundant in VR) and any paint callbacks (can't run here).
            let primitives: Vec<egui::ClippedPrimitive> = draw
                .primitives
                .iter()
                .filter(|p| match &p.primitive {
                    egui::epaint::Primitive::Mesh(m) => {
                        !matches!(m.texture_id, egui::TextureId::User(_))
                    }
                    _ => false,
                })
                .cloned()
                .collect();
            vr_egui
                .cmd_draw(cb, ui_extent, draw.pixels_per_point, &primitives)
                .map_err(|e| anyhow::anyhow!("VR egui draw: {e}"))?;
            unsafe { device.cmd_end_rendering(cb) };
            image_barrier(
                &device, cb, ui_image, vk::ImageAspectFlags::COLOR,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ,
            );
            unsafe { device.end_command_buffer(cb)? };
            Ok(())
        })();
        let result = record.and_then(|()| unsafe {
            let cbs = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let r = device
                .queue_submit(self.ctx.queue, &[submit], fence)
                .and_then(|()| device.wait_fences("wait:fence@render_vr_ui", &[fence], true, u64::MAX));
            device.destroy_fence(fence, None);
            r.map_err(Into::into)
        });
        unsafe { device.free_command_buffers(self.command_pool, &[cb]) };
        result
    }

    /// (Re)create the VR eye render resources for `extent`: the slot ring
    /// (set-0 + camera UBO + command buffer), `layers` HDR / depth layers (2 =
    /// multiview, one per eye) and, with `samples` > 1, the multisampled
    /// attachments. Cheap no-op when already matching.
    fn ensure_xr_render(&mut self, extent: vk::Extent2D, layers: u32, samples: u8) -> Result<()> {
        if self.xr_render.as_ref().is_some_and(|x| x.extent == extent && x.layers == layers && x.samples == samples) {
            return Ok(());
        }
        let allocator = self
            .allocator
            .clone()
            .context("renderer has no allocator for XR")?;
        let mut alloc = allocator.lock().unwrap();
        if let Some(mut old) = self.xr_render.take() {
            old.destroy(&self.ctx.device, &mut alloc, self.command_pool);
        }
        let device = &self.ctx.device;
        let n = XR_SLOTS as u32;
        // set-0 per slot: 1 UBO (b0), 1 storage buffer (b2), 7 combined samplers (b1,3-8).
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(n).pool_sizes(&[
                    vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: n },
                    vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: n },
                    vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: SET0_SAMPLERS * n },
                ]),
                None,
            )?
        };
        let cbs = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(n),
            )?
        };
        let mut slots = Vec::with_capacity(XR_SLOTS);
        for (i, cb) in cbs.into_iter().enumerate() {
            let set = unsafe {
                device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&[self.pipeline_cache.set0_layout]),
                )?[0]
            };
            let ubo = Buffer::new(
                device,
                &mut alloc,
                size_of::<FrameUbo>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                MemoryLocation::CpuToGpu,
                &format!("xr eye ubo {i}"),
            )?;
            let fence = unsafe {
                device.create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)?
            };
            slots.push(XrSlot { cb, fence, ubo, set, color_views: Vec::new(), instances: InstanceRing::default() });
        }
        let (depth_image, depth_alloc) = create_image_layers(
            device,
            &mut alloc,
            depth_format(),
            // Sampled by the underwater fog.
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            extent,
            layers,
            "xr eye depth",
        )?;
        // Attachment view (depth + stencil); the per-layer views below are the sampled ones.
        let depth_view = create_view_layers(device, depth_image, depth_format(), depth_aspect(), 0, layers)?;
        let (hdr_image, hdr_alloc) = create_image_layers(
            device,
            &mut alloc,
            post::HDR_FORMAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            extent,
            layers,
            "xr eye hdr",
        )?;
        let hdr_view = create_view_layers(device, hdr_image, post::HDR_FORMAT, vk::ImageAspectFlags::COLOR, 0, layers)?;
        let mut depth_layer_views = Vec::with_capacity(layers as usize);
        let mut hdr_layer_views = Vec::with_capacity(layers as usize);
        for l in 0..layers {
            depth_layer_views.push(create_view_layers(device, depth_image, depth_format(), vk::ImageAspectFlags::DEPTH, l, 1)?);
            hdr_layer_views.push(create_view_layers(device, hdr_image, post::HDR_FORMAT, vk::ImageAspectFlags::COLOR, l, 1)?);
        }
        let msaa = if samples > 1 {
            let count = vk::SampleCountFlags::from_raw(samples as u32);
            Some(XrMsaa {
                color: TransientImage::new(
                    &self.ctx, &mut alloc, post::HDR_FORMAT, vk::ImageAspectFlags::COLOR,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT, extent, layers, count, "xr eye msaa colour",
                )?,
                depth: TransientImage::new(
                    &self.ctx, &mut alloc, depth_format(), depth_aspect(),
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, extent, layers, count, "xr eye msaa depth",
                )?,
            })
        } else {
            None
        };
        let post_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(layers).pool_sizes(&[vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: layers,
                }]),
                None,
            )?
        };
        let post_layouts = vec![self.post_pass.set_layout(); layers as usize];
        let post_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(post_pool)
                    .set_layouts(&post_layouts),
            )?
        };
        for (set, &view) in post_sets.iter().zip(&hdr_layer_views) {
            let post_info = [vk::DescriptorImageInfo::default()
                .sampler(self.post_pass.sampler())
                .image_view(view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            unsafe {
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&post_info)],
                    &[],
                )
            };
        }
        let bloom = Some(self.post_pass.new_bloom_chain(device, &mut alloc, extent)?);
        self.xr_render = Some(XrRender {
            pool,
            slots,
            next: 0,
            depth_image,
            depth_view,
            depth_layer_views,
            depth_alloc: Some(depth_alloc),
            hdr_image,
            hdr_view,
            hdr_layer_views,
            hdr_alloc: Some(hdr_alloc),
            msaa,
            post_pool,
            post_sets,
            bloom,
            extent,
            layers,
            samples,
        });
        Ok(())
    }

    /// The eye UBO: this frame's lights / shadows / probes (written by
    /// `render` or `render_xr_frame`) seen from `view` / `proj`, with the paths
    /// that only exist for the window camera turned off.
    fn xr_eye_ubo(base: &FrameUbo, view: glam::Mat4, proj: glam::Mat4, extent: vk::Extent2D) -> FrameUbo {
        let mut ubo = *base;
        let inv_view = view.inverse();
        ubo.view = view.to_cols_array_2d();
        ubo.proj = proj.to_cols_array_2d();
        ubo.view_proj = (proj * view).to_cols_array_2d();
        ubo.inv_view = inv_view.to_cols_array_2d();
        ubo.inv_proj = proj.inverse().to_cols_array_2d();
        ubo.camera_pos = [inv_view.w_axis.x, inv_view.w_axis.y, inv_view.w_axis.z, 1.0];
        ubo.ssr = [0.0; 4];
        ubo.debug[2] = 0.0; // world-probe GI (screen-GI gather is desktop-camera-space)
        ubo.debug[3] = 1.0; // linear HDR; the eye post pass tonemaps
        // Planar reflections are rendered from the DESKTOP camera; their screen
        // UVs are meaningless per-eye, so the eyes fall back to the probe.
        ubo.refl_box[3] = 0.0;
        // The opaque copies are the desktop camera's: no refraction per eye.
        ubo.water[1] = 0.0;
        // Screen-space AO is computed for the desktop camera only (the eye
        // passes have no per-eye depth prepass): off in VR.
        ubo.refl_blend2[3] = 0.0;
        ubo.postfx2[3] = extent.width as f32;
        ubo.postfx3[3] = extent.height as f32;
        ubo
    }

    /// Claim the next XR slot: wait for its previous submission (two frames
    /// ago, normally long finished), release that submission's image views and
    /// instance data, upload `ubo`, and point the slot's set 0 at it with the
    /// shared bindings copied from the frame just recorded.
    fn xr_claim_slot(&mut self, ubo: &FrameUbo) -> Result<(usize, vk::CommandBuffer, vk::Fence, vk::DescriptorSet)> {
        let device = self.ctx.device.clone();
        let frame_set = self.frame_sets[(self.frame_index + FRAMES_IN_FLIGHT - 1) % FRAMES_IN_FLIGHT];
        let (slot_i, cb, fence, xr_set, ubo_buf) = {
            let xr = self.xr_render.as_mut().unwrap();
            let i = xr.next;
            xr.next = (i + 1) % XR_SLOTS;
            let slot = &mut xr.slots[i];
            unsafe {
                device.wait_fences("wait:fence@xr_claim_slot", &[slot.fence], true, u64::MAX)?;
                device.reset_fences(&[slot.fence])?;
                for v in slot.color_views.drain(..) {
                    device.destroy_image_view(v, None);
                }
            }
            slot.instances.reset(&device, &mut self.allocator.as_ref().unwrap().lock().unwrap());
            slot.ubo.write(0, bytemuck::bytes_of(ubo));
            (i, slot.cb, slot.fence, slot.set, slot.ubo.handle)
        };
        // Bind the eye UBO at 0; share the shadow/probe/lightmap/sgi/depth/env/
        // planar bindings (1-9) with the frame set via descriptor copies.
        let ubo_info = [vk::DescriptorBufferInfo::default()
            .buffer(ubo_buf)
            .offset(0)
            .range(size_of::<FrameUbo>() as u64)];
        let write0 = vk::WriteDescriptorSet::default()
            .dst_set(xr_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&ubo_info);
        let copies: Vec<vk::CopyDescriptorSet> = (1..=9u32)
            .map(|b| {
                vk::CopyDescriptorSet::default()
                    .src_set(frame_set)
                    .src_binding(b)
                    .src_array_element(0)
                    .dst_set(xr_set)
                    .dst_binding(b)
                    .dst_array_element(0)
                    .descriptor_count(1)
            })
            .chain([ripple_copy(frame_set, xr_set)])
            .collect();
        unsafe { device.update_descriptor_sets(&[write0], &copies) };
        Ok((slot_i, cb, fence, xr_set))
    }

    /// A 2D view of one layer of an XR swapchain image, kept until the slot's
    /// fence signals.
    fn xr_color_view(&mut self, slot: usize, image: vk::Image, format: vk::Format, layer: u32) -> Result<vk::ImageView> {
        let view = create_view_layers(&self.ctx.device, image, format, vk::ImageAspectFlags::COLOR, layer, 1)?;
        self.xr_render.as_mut().unwrap().slots[slot].color_views.push(view);
        Ok(view)
    }

    /// Both XR eyes can render in one multiview pass ([`Self::render_xr_stereo`]):
    /// the device has multiview and the tier asks for it.
    pub fn xr_multiview(&self) -> bool {
        self.ctx.multiview && self.tier.xr_multiview()
    }

    /// Render both eyes of an XR frame in one multiview pass (view mask 0b11)
    /// into the two layers of `image` (a 2-layer XR swapchain image), then bloom
    /// and post per eye. `cull` is a view enclosing both eyes (it culls the draws
    /// and stands in for the camera in shaders without per-eye variants). Uses
    /// this frame's lighting from `render` / `render_xr_frame`; on the mobile
    /// tier the scene pass is multisampled and resolved in-pass. No VR UI
    /// overlay (use [`Self::render_xr_eye`] for that). Submitted without a CPU
    /// wait, like the per-eye path.
    #[allow(clippy::too_many_arguments)]
    pub fn render_xr_stereo(
        &mut self,
        image: u64,
        color_format: u32,
        width: u32,
        height: u32,
        eyes: &[XrEyeCamera; 2],
        cull: (glam::Mat4, glam::Mat4),
        input: &FrameInput,
    ) -> Result<()> {
        let extent = vk::Extent2D { width: width.max(1), height: height.max(1) };
        let format = vk::Format::from_raw(color_format as i32);
        let device = self.ctx.device.clone();
        let queue = self.ctx.queue;
        let samples = self.xr_msaa;
        self.ensure_xr_render(extent, 2, samples)?;
        self.post_pass.ensure_format(&device, format)?;
        let Some(base) = self.last_frame_ubo else {
            return Ok(());
        };
        let (cull_view, cull_proj) = cull;
        let mut ubo = Self::xr_eye_ubo(&base, cull_view, cull_proj, extent);
        for (i, e) in eyes.iter().enumerate() {
            ubo.eye_view[i] = e.view.to_cols_array_2d();
            ubo.eye_view_proj[i] = (e.proj * e.view).to_cols_array_2d();
            let p = e.view.inverse().w_axis;
            ubo.eye_camera_pos[i] = [p.x, p.y, p.z, 1.0];
        }
        let (slot_i, cb, fence, xr_set) = self.xr_claim_slot(&ubo)?;
        let color_image = {
            use ash::vk::Handle as _;
            vk::Image::from_raw(image)
        };
        let color_views = [
            self.xr_color_view(slot_i, color_image, format, 0)?,
            self.xr_color_view(slot_i, color_image, format, 1)?,
        ];
        let (hdr_image, hdr_view, hdr_layers, depth_image, depth_view, depth_layers, post_sets, msaa) = {
            let xr = self.xr_render.as_ref().unwrap();
            (
                xr.hdr_image,
                xr.hdr_view,
                [xr.hdr_layer_views[0], xr.hdr_layer_views[1]],
                xr.depth_image,
                xr.depth_view,
                [xr.depth_layer_views[0], xr.depth_layer_views[1]],
                [xr.post_sets[0], xr.post_sets[1]],
                xr.msaa.as_ref().map(|m| (m.color.image, m.color.view, m.depth.image, m.depth.view)),
            )
        };

        let cull_vp = cull_proj * cull_view;
        let cull_pos = cull_view.inverse().w_axis.truncate();
        let order = self.draw_order(input, cull_pos);
        let area = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent };
        unsafe {
            device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        self.gpu_cull_begin(gpu_cull::eye_slot(slot_i));
        let stereo = cluster_vr::StereoView { view: [eyes[0].view, eyes[1].view], proj: [eyes[0].proj, eyes[1].proj] };
        self.gpu_cull_eye(cb, slot_i, &order, input, cull_vp, cull_pos, Some((&stereo, extent.height as f32)), [extent.width, extent.height], None);
        let eye_order = frustum_cull(&self.gpu_cpu_order(&order), input.draws, cull_vp);

        // 1) Both eyes' scene in one pass (linear HDR), multisampled + resolved
        //    in-pass when MSAA is on. Depth is only kept when an eye needs it
        //    (underwater fog samples it).
        let keep_depth = eyes.iter().any(|e| e.postfx.underwater.is_some());
        // Hi-Z for both eyes (two-phase): the pass is split after its early draws, so its colour and
        // depth are stored here, and a multisampled depth is resolved for the pyramids.
        let xr_split = self.gpu_xr_split_pending();
        let color_targets = [Some(hdr_image), msaa.map(|m| m.0)];
        for img in color_targets.into_iter().flatten() {
            image_barrier_layers(
                &device, cb, img, vk::ImageAspectFlags::COLOR, 0, 2,
                vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags2::FRAGMENT_SHADER
                    | vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            );
        }
        let depth_targets = [Some(depth_image), msaa.map(|m| m.2)];
        for img in depth_targets.into_iter().flatten() {
            image_barrier_layers(
                &device, cb, img, depth_aspect(), 0, 2,
                vk::ImageLayout::UNDEFINED, depth_attach_layout(),
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::empty(),
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );
        }
        let flipped = vk::Viewport {
            x: 0.0,
            y: extent.height as f32,
            width: extent.width as f32,
            height: -(extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let clear_color = vk::ClearValue { color: vk::ClearColorValue { float32: input.clear_color } };
        let clear_depth = vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } };
        let (color_att, depth_att) = match msaa {
            Some((_, msaa_color, _, msaa_depth)) => {
                let color = vk::RenderingAttachmentInfo::default()
                    .image_view(msaa_color)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(if xr_split { vk::AttachmentStoreOp::STORE } else { vk::AttachmentStoreOp::DONT_CARE })
                    .clear_value(clear_color);
                let color = if xr_split {
                    color
                } else {
                    color.resolve_mode(vk::ResolveModeFlags::AVERAGE).resolve_image_view(hdr_view).resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                };
                let mut depth = vk::RenderingAttachmentInfo::default()
                    .image_view(msaa_depth)
                    .image_layout(depth_attach_layout())
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(if xr_split { vk::AttachmentStoreOp::STORE } else { vk::AttachmentStoreOp::DONT_CARE })
                    .clear_value(clear_depth);
                if keep_depth || xr_split {
                    depth = depth
                        .resolve_mode(vk::ResolveModeFlags::SAMPLE_ZERO)
                        .resolve_image_view(depth_view)
                        .resolve_image_layout(depth_attach_layout());
                }
                (color, depth)
            }
            None => (
                vk::RenderingAttachmentInfo::default()
                    .image_view(hdr_view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(clear_color),
                vk::RenderingAttachmentInfo::default()
                    .image_view(depth_view)
                    .image_layout(depth_attach_layout())
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(if keep_depth || xr_split { vk::AttachmentStoreOp::STORE } else { vk::AttachmentStoreOp::DONT_CARE })
                    .clear_value(clear_depth),
            ),
        };
        unsafe {
            let color_atts = [color_att];
            let stencil_att = stencil_attachment(&depth_att);
            device.cmd_begin_rendering(
                cb,
                &vk::RenderingInfo::default()
                    .render_area(area)
                    .layer_count(1)
                    .view_mask(0b11)
                    .color_attachments(&color_atts)
                    .depth_attachment(&depth_att)
                    .stencil_attachment(&stencil_att),
            );
            device.cmd_set_viewport(cb, 0, &[flipped]);
            device.cmd_set_scissor(cb, 0, &[area]);
        }
        self.pipeline_cache.set_target_format(post::HDR_FORMAT);
        self.pipeline_cache.set_target_views(true, samples);
        let sky = if input.draw_skybox { self.record_skybox(&device, cb, xr_set, true, false) } else { Ok(()) };
        self.instance_ring_sel = Some(slot_i);
        self.gpu_arm(true);
        if xr_split {
            self.gpu_late_split(gpu_cull::LateSplit {
                colors: Vec::new(),
                depth: (depth_image, vk::ImageView::null()),
                depth_att: msaa.map_or(depth_view, |m| m.3),
                extent,
                xr: Some(gpu_cull::XrSplit {
                    depth_image,
                    depth_layers,
                    color: msaa.map_or(hdr_view, |m| m.1),
                    resolve_color: msaa.map(|_| hdr_view),
                    depth_att: msaa.map_or(depth_view, |m| m.3),
                    resolve_depth: msaa.map(|_| depth_view),
                    keep_depth,
                    slot: slot_i,
                    view_proj: [eyes[0].proj * eyes[0].view, eyes[1].proj * eyes[1].view],
                    position: cull_pos,
                    forward: -eyes[0].view.row(2).truncate().normalize_or_zero(),
                }),
                eye: None,
            });
        }
        let scene = sky.and_then(|()| self.record_scene_draws(&device, cb, &eye_order, input, xr_set, false).map(|_| ()));
        if xr_split && let Some(unused) = self.gpu_take_split() {
            // Nothing drew from the cull (so nothing split the pass): split here anyway, the
            // pass's colour (multisampled: its resolve) is only finished by the second half.
            self.gpu_cull_late_xr(&device, cb, &unused);
        }
        if scene.is_ok() {
            self.record_world_text(&device, cb, xr_set, false, SpritePart::All);
        }
        self.gpu_arm(false);
        self.instance_ring_sel = None;
        self.pipeline_cache.set_target_views(false, 1);
        unsafe { device.cmd_end_rendering(cb) };
        scene?;

        // 2) Per eye: underwater fog on its layer, then bloom + post into its
        //    swapchain layer. Resolves (depth included) write in the colour
        //    output stage, which the fog pass's own depth barrier doesn't cover.
        if msaa.is_some() && keep_depth {
            let b = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ)];
            unsafe { device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&b)) };
        }
        for (i, eye) in eyes.iter().enumerate() {
            let cam = CameraData { view: eye.view, proj: eye.proj, position: eye.view.inverse().w_axis.truncate() };
            if let Some(fx) = eye.postfx.underwater.as_ref() {
                let slot = if i == 0 { underwater::SLOT_XR } else { underwater::SLOT_XR_RIGHT } + slot_i as u32;
                self.underwater.record(
                    &device, cb, slot,
                    (hdr_image, hdr_layers[i]), (depth_image, depth_layers[i]), extent,
                    &cam, fx, &input.light, input.time, i as u32,
                );
            }
        }
        image_barrier_layers(
            &device, cb, hdr_image, vk::ImageAspectFlags::COLOR, 0, 2,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_SAMPLED_READ,
        );
        image_barrier_layers(
            &device, cb, color_image, vk::ImageAspectFlags::COLOR, 0, 2,
            vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        );
        for (i, eye) in eyes.iter().enumerate() {
            self.post_pass.exposure.record(&device,cb,post_sets[i],extent,eye.postfx.auto_exposure,2,
                self.frame_serial,self.started.elapsed().as_secs_f64());
            let bloom_set = {
                let chain = self.xr_render.as_ref().unwrap().bloom.as_ref().unwrap();
                self.post_pass.record_bloom(&device, cb, chain, post_sets[i], &eye.postfx);
                chain.output_set()
            };
            unsafe {
                let att = [vk::RenderingAttachmentInfo::default()
                    .image_view(color_views[i])
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .store_op(vk::AttachmentStoreOp::STORE)];
                device.cmd_begin_rendering(cb, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&att));
            }
            let cam = CameraData { view: eye.view, proj: eye.proj, position: eye.view.inverse().w_axis.truncate() };
            let wobble = underwater::wobble(eye.postfx.underwater.as_ref(), &cam, input.time);
            self.post_pass.record_set_as(&device, cb, post_sets[i], bloom_set, extent, &eye.postfx, wobble, format);
            unsafe { device.cmd_end_rendering(cb) };
        }
        unsafe {
            device.end_command_buffer(cb)?;
            let cbs = [cb];
            device.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&cbs)], fence)?;
        }
        Ok(())
    }

    /// Render the scene into one VR eye's XR swapchain image (raw `VkImage`) using
    /// the per-eye `view`/`proj`. Uses this frame's lights / shadows / probes
    /// (call after `render` or `render_xr_frame`). The eye renders linear HDR with world-probe
    /// GI + env reflection (no SSR / screen-GI), then bloom + the post pass into
    /// the XR image, then the VR UI overlay. Submitted without a CPU wait (the
    /// commands only need to be submitted before `xrReleaseSwapchainImage`); a
    /// ring of slots keeps in-flight eye renders from sharing buffers. Gated by
    /// the caller behind an active XR session; the desktop path is untouched.
    #[allow(clippy::too_many_arguments)]
    pub fn render_xr_eye(
        &mut self,
        image: u64,
        color_format: u32,
        width: u32,
        height: u32,
        view: glam::Mat4,
        proj: glam::Mat4,
        input: &FrameInput,
        overlay: Option<VrOverlayDraw>,
    ) -> Result<()> {
        // Callers that do not say which eye alternate left / right (only the Hi-Z pyramid choice depends on it).
        let eye = self.xr_eye_auto;
        self.xr_eye_auto = (eye + 1) % 2;
        self.render_xr_eye_indexed(eye, image, color_format, width, height, view, proj, input, overlay)
    }

    /// [`render_xr_eye`](Self::render_xr_eye) for eye `eye` (0 = left, 1 = right): each eye keeps its own Hi-Z
    /// pyramid of the last frame (the cluster cull of this pass tests it, then re-tests after this
    /// frame's depth exists).
    #[allow(clippy::too_many_arguments)]
    pub fn render_xr_eye_indexed(
        &mut self,
        eye: usize,
        image: u64,
        color_format: u32,
        width: u32,
        height: u32,
        view: glam::Mat4,
        proj: glam::Mat4,
        input: &FrameInput,
        overlay: Option<VrOverlayDraw>,
    ) -> Result<()> {
        let extent = vk::Extent2D {
            width: width.max(1),
            height: height.max(1),
        };
        let format = vk::Format::from_raw(color_format as i32);
        let device = self.ctx.device.clone();
        let queue = self.ctx.queue;
        self.ensure_xr_render(extent, 1, 1)?;
        self.post_pass.ensure_format(&device, format)?;

        // Eye UBO: this frame's lighting with the eye camera.
        let Some(base) = self.last_frame_ubo else {
            return Ok(()); // no frame recorded yet this run
        };
        let ubo = Self::xr_eye_ubo(&base, view, proj, extent);
        let inv_view = view.inverse();
        let (slot_i, cb, fence, xr_set) = self.xr_claim_slot(&ubo)?;
        // `depth_view`: attachment (depth + stencil); `depth_read_view`: the
        // depth-aspect view layer 0 the underwater fog samples.
        let (depth_view, depth_read_view, depth_image, hdr_image, hdr_view, post_set) = {
            let xr = self.xr_render.as_ref().unwrap();
            (xr.depth_view, xr.depth_layer_views[0], xr.depth_image, xr.hdr_image, xr.hdr_view, xr.post_sets[0])
        };
        let color_image = {
            use ash::vk::Handle as _;
            vk::Image::from_raw(image)
        };
        let color_view = self.xr_color_view(slot_i, color_image, format, 0)?;

        // Sorted like the desktop pass (transparents back-to-front), then culled
        // to this eye's frustum.
        let order = self.draw_order(input, inv_view.w_axis.truncate());
        let area = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent };

        unsafe {
            device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        // GPU-culled eye batches (this eye's frustum); the rest on the CPU.
        self.gpu_cull_begin(gpu_cull::eye_slot(slot_i));
        let eye_forward = -view.row(2).truncate().normalize_or_zero();
        self.gpu_cull_eye(cb, slot_i, &order, input, proj * view, inv_view.w_axis.truncate(), None, [extent.width, extent.height], Some((eye, eye_forward)));
        let eye_order = frustum_cull(&self.gpu_cpu_order(&order), input.draws, proj * view);
        // 1) Scene, linear HDR, into the eye HDR target (shared by the slots:
        //    wait for the previous eye's post read before overwriting it).
        image_barrier(
            &device, cb, hdr_image, vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        );
        image_barrier(
            &device, cb, depth_image, depth_aspect(),
            vk::ImageLayout::UNDEFINED, depth_attach_layout(),
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );
        // Negative-height viewport to match the desktop pipelines' winding.
        let flipped = vk::Viewport {
            x: 0.0,
            y: extent.height as f32,
            width: extent.width as f32,
            height: -(extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        unsafe {
            let color_att = [vk::RenderingAttachmentInfo::default()
                .image_view(hdr_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: input.clear_color } })];
            // Kept: the overlay pass below depth-tests the lasers / panels.
            let depth_att = vk::RenderingAttachmentInfo::default()
                .image_view(depth_view)
                .image_layout(depth_attach_layout())
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } });
            let stencil_att = stencil_attachment(&depth_att);
            device.cmd_begin_rendering(
                cb,
                &vk::RenderingInfo::default()
                    .render_area(area)
                    .layer_count(1)
                    .color_attachments(&color_att)
                    .depth_attachment(&depth_att)
                    .stencil_attachment(&stencil_att),
            );
            device.cmd_set_viewport(cb, 0, &[flipped]);
            device.cmd_set_scissor(cb, 0, &[area]);
        }
        self.pipeline_cache.set_target_format(post::HDR_FORMAT);
        if input.draw_skybox {
            self.record_skybox(&device, cb, xr_set, true, false)?;
        }
        self.instance_ring_sel = Some(slot_i);
        self.gpu_arm(true);
        // A Hi-Z pending for this eye's cluster cull splits the pass after the early batches (late cull).
        self.gpu_late_split(gpu_cull::LateSplit {
            colors: vec![hdr_view],
            depth: (depth_image, depth_read_view),
            depth_att: depth_view,
            extent,
            xr: None,
            eye: Some(gpu_cull::EyeSplit { eye, slot: slot_i, view_proj: proj * view, position: inv_view.w_axis.truncate(), forward: eye_forward }),
        });
        let drawn = self.record_scene_draws(&device, cb, &eye_order, input, xr_set, false);
        // Nothing drew from the cull, so nothing split the pass: the late half is skipped.
        let _ = self.gpu_take_split();
        if drawn.is_ok() {
            self.record_world_text(&device, cb, xr_set, false, SpritePart::All);
        }
        self.gpu_arm(false);
        self.instance_ring_sel = None;
        drawn?;
        unsafe { device.cmd_end_rendering(cb) };
        let eye_cam = CameraData { view, proj, position: inv_view.w_axis.truncate() };
        if let Some(fx) = input.postfx.underwater.as_ref() {
            self.underwater.record(
                &device, cb, underwater::SLOT_XR + slot_i as u32,
                (hdr_image, hdr_view), (depth_image, depth_read_view), extent,
                &eye_cam, fx, &input.light, input.time, 0,
            );
        }

        // 2) HDR -> shader read, bloom, then the post pass into the XR image.
        image_barrier(
            &device, cb, hdr_image, vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ,
        );
        self.post_pass.exposure.record(&device,cb,post_set,extent,input.postfx.auto_exposure,2,
            self.frame_serial,self.started.elapsed().as_secs_f64());
        let bloom_set = {
            let chain = self.xr_render.as_ref().unwrap().bloom.as_ref().unwrap();
            self.post_pass.record_bloom(&device, cb, chain, post_set, &input.postfx);
            chain.output_set()
        };
        image_barrier(
            &device, cb, color_image, vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        );
        unsafe {
            let att = [vk::RenderingAttachmentInfo::default()
                .image_view(color_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)];
            device.cmd_begin_rendering(cb, &vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&att));
        }
        let wobble = underwater::wobble(input.postfx.underwater.as_ref(), &eye_cam, input.time);
        self.post_pass.record_set_as(&device, cb, post_set, bloom_set, extent, &input.postfx, wobble, format);
        unsafe { device.cmd_end_rendering(cb) };

        // 3) VR UI panel + hand lasers on top of the tonemapped image (UI is
        //    already display-referred), depth-tested against the scene. The panel
        //    only shows when the hand menu is up; the lasers always show so you
        //    can see what each controller points at.
        if let (Some(ov), Some(panel)) = (overlay, self.vr_overlay.as_ref()) {
            unsafe {
                device.cmd_pipeline_barrier2(
                    cb,
                    &vk::DependencyInfo::default().memory_barriers(&[vk::MemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
                        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
                        .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ)]),
                );
                let color_att = [vk::RenderingAttachmentInfo::default()
                    .image_view(color_view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE)];
                let depth_att = vk::RenderingAttachmentInfo::default()
                    .image_view(depth_view)
                    .image_layout(depth_attach_layout())
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::DONT_CARE);
                let stencil_att = stencil_attachment(&depth_att);
                device.cmd_begin_rendering(
                    cb,
                    &vk::RenderingInfo::default()
                        .render_area(area)
                        .layer_count(1)
                        .color_attachments(&color_att)
                        .depth_attachment(&depth_att)
                        .stencil_attachment(&stencil_att),
                );
                device.cmd_set_viewport(cb, 0, &[flipped]);
                device.cmd_set_scissor(cb, 0, &[area]);
            }
            let vp = proj * view;
            for p in ov.panels.iter().flatten() {
                let panel_push = vr_overlay::QuadPush {
                    mvp: (vp * p.model).to_cols_array_2d(),
                    params: [0.0; 4],
                    uv_rect: p.uv_rect,
                };
                panel.record_quad(&device, cb, &panel_push);
            }
            // Hand lasers: a thin camera-facing quad from each controller to its
            // precomputed hit point (UI/object, or far). Left = cyan, right =
            // orange. params = [mode=1, r, g, b].
            let cam = inv_view.w_axis.truncate();
            for (hand, end, rgb) in [
                (ov.left_hand, ov.left_laser_end, [0.15, 0.8, 1.0]),
                (ov.right_hand, ov.right_laser_end, [1.0, 0.5, 0.1]),
            ] {
                if let (Some(m), Some(b)) = (hand, end) {
                    let a = m.w_axis.truncate();
                    let seg = b - a;
                    let len = seg.length();
                    if len > 1e-3 {
                        let d = seg / len;
                        let mid = (a + b) * 0.5;
                        let to_cam = (cam - mid).normalize_or(glam::Vec3::Z);
                        let thin = d.cross(to_cam).normalize_or(glam::Vec3::Y);
                        let nrm = thin.cross(d).normalize_or(to_cam);
                        let rot = glam::Mat3::from_cols(d, thin, nrm);
                        let model = glam::Mat4::from_translation(mid)
                            * glam::Mat4::from_mat3(rot)
                            * glam::Mat4::from_scale(glam::Vec3::new(len, 0.004, 1.0));
                        let push = vr_overlay::QuadPush {
                            mvp: (vp * model).to_cols_array_2d(),
                            params: [1.0, rgb[0], rgb[1], rgb[2]],
                            uv_rect: [0.0, 0.0, 1.0, 1.0],
                        };
                        panel.record_quad(&device, cb, &push);
                    }
                }
            }
            // Tracked hands (edit mode): camera-facing bone quads + joint dots,
            // drawn over the lasers so the hand reads on top of its pointer.
            for (skeleton, rgb) in [
                (ov.left_hand_skeleton, [0.55, 0.85, 1.0]),
                (ov.right_hand_skeleton, [1.0, 0.75, 0.4]),
            ] {
                let Some(s) = skeleton else { continue };
                for &(ja, jb) in HAND_BONES.iter() {
                    if !s.valid[ja] || !s.valid[jb] {
                        continue;
                    }
                    if let Some(push) =
                        hand_bone_quad(vp, cam, s.positions[ja], s.positions[jb], 0.011, rgb)
                    {
                        panel.record_quad(&device, cb, &push);
                    }
                }
                for j in 0..s.positions.len() {
                    if s.valid[j] {
                        let push = hand_dot_quad(vp, cam, s.positions[j], 0.02, rgb);
                        panel.record_quad(&device, cb, &push);
                    }
                }
            }
            // Remote co-edit peers: their tracked hands at true scale, one
            // color each, drawn after the local hands so peers read on top.
            for (skeleton, rgb) in ov.remote_hands.iter() {
                for &(ja, jb) in HAND_BONES.iter() {
                    if !skeleton.valid[ja] || !skeleton.valid[jb] {
                        continue;
                    }
                    if let Some(push) = hand_bone_quad(
                        vp,
                        cam,
                        skeleton.positions[ja],
                        skeleton.positions[jb],
                        0.011,
                        *rgb,
                    ) {
                        panel.record_quad(&device, cb, &push);
                    }
                }
                for j in 0..skeleton.positions.len() {
                    if skeleton.valid[j] {
                        let push = hand_dot_quad(vp, cam, skeleton.positions[j], 0.02, *rgb);
                        panel.record_quad(&device, cb, &push);
                    }
                }
            }
            // Remote heads: a camera-facing disc at the peer's eye height.
            for (pos, radius, rgb) in ov.remote_heads.iter() {
                let push = hand_dot_quad(vp, cam, glam::Vec3::from(*pos), *radius, *rgb);
                panel.record_quad(&device, cb, &push);
            }
            unsafe { device.cmd_end_rendering(cb) };
        }
        unsafe {
            device.end_command_buffer(cb)?;
            // Submitted, not waited: OpenXR only needs the commands submitted
            // before the image is released; the slot fence guards its reuse.
            let cbs = [cb];
            device.queue_submit(queue, &[vk::SubmitInfo::default().command_buffers(&cbs)], fence)?;
        }
        Ok(())
    }

    /// Editor RTT: render the viewport 3D (Flux trace + skybox + scene) into the
    /// viewport target's offscreen color/depth at the dock-rect resolution, then
    /// transition the color to SHADER_READ so egui can show it. Uses the editor
    /// (main) camera + its sgi temporal state, but the target's own rect-sized
    /// sgi/color/depth. Only called by the editor; the game renders to swapchain.
    /// Rasterise the selected draws into the selection mask (see `selection.rs`).
    /// The scene depth (shader-read layout on entry and exit) is only tested
    /// against, never written. Returns false when the pass could not run.
    #[allow(clippy::too_many_arguments)]
    fn record_selection_mask(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        input: &FrameInput,
        draws: &[usize],
        dep_img: vk::Image,
        dep_att_view: vk::ImageView,
        ext: vk::Extent2D,
    ) -> Result<bool> {
        {
            let allocator = self.allocator();
            let mut alloc = allocator.lock().unwrap();
            self.selection.tick(device, &mut alloc);
            self.selection.ensure_target(device, &mut alloc, ext)?;
        }
        let Some((mask_img, mask_view)) = self.selection.mask_image() else { return Ok(false) };
        let instances: Vec<InstanceData> =
            draws.iter().map(|&i| InstanceData::from_model(input.draws[i].transform.to_cols_array_2d())).collect();
        let (inst_buf, inst_off) = self.push_instances(&instances)?;
        let read_only = vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        image_barrier(
            device, cb, mask_img, vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::SHADER_READ,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        );
        image_barrier(
            device, cb, dep_img, depth_aspect(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, read_only,
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_READ,
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
        );
        let color = vk::RenderingAttachmentInfo::default()
            .image_view(mask_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } });
        let depth = vk::RenderingAttachmentInfo::default()
            .image_view(dep_att_view)
            .image_layout(read_only)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::NONE);
        let stencil = stencil_attachment(&depth);
        let colors = [color];
        let info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: ext })
            .layer_count(1)
            .color_attachments(&colors)
            .depth_attachment(&depth)
            .stencil_attachment(&stencil);
        let (all, visible, layout) = self.selection.mask_pipelines();
        let view_proj = input.camera.proj * input.camera.view;
        unsafe {
            device.cmd_begin_rendering(cb, &info);
            // Same Y-flipped viewport as the scene, so mask rows line up with the colour target.
            device.cmd_set_viewport(
                cb,
                0,
                &[vk::Viewport { x: 0.0, y: ext.height as f32, width: ext.width as f32, height: -(ext.height as f32), min_depth: 0.0, max_depth: 1.0 }],
            );
            device.cmd_set_scissor(cb, 0, &[vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: ext }]);
            device.cmd_bind_vertex_buffers(cb, 1, &[inst_buf], &[inst_off]);
            for pipeline in [all, visible] {
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
                for (k, &i) in draws.iter().enumerate() {
                    let d = &input.draws[i];
                    let mesh = &self.meshes[d.mesh.0];
                    let strength = if d.highlight >= selection::PRIMARY_HIGHLIGHT { 1.0 } else { 0.5 };
                    device.cmd_push_constants(
                        cb,
                        layout,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        0,
                        &selection::SelectionPass::mask_push(view_proj, strength),
                    );
                    device.cmd_bind_vertex_buffers(cb, 0, &[mesh.vertex_buffer.handle], &[0]);
                    device.cmd_bind_index_buffer(cb, mesh.index_buffer.handle, 0, mesh.index_type);
                    device.cmd_draw_indexed(cb, mesh.index_count, 1, 0, 0, k as u32);
                }
            }
            device.cmd_end_rendering(cb);
        }
        image_barrier(
            device, cb, dep_img, depth_aspect(),
            read_only, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS, vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_READ,
        );
        image_barrier(
            device, cb, mask_img, vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ,
        );
        self.viewport_draw_stats.outline_draws += draws.len() as u32 * 2;
        Ok(true)
    }

    fn record_viewport_scene(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        order: &[usize],
        input: &FrameInput,
    ) -> Result<()> {
        let (col_img, col_view, dep_img, dep_view, dep_att_view, ext, _, _, s_probe, s_img, s_view, s_filtered_img, s_filtered_view, s_sh_buf, s_raw_rays, s_final_img, s_final_view, _, hdr_img, hdr_view, post_set) = {
            let vt = self.viewport_target.as_ref().unwrap();
            (
                vt.color, vt.color_view, vt.depth, vt.depth_view, vt.depth_att_view, vt.extent,
                vt.sgi.depth_image, vt.sgi.depth_view, vt.sgi.probe_extent,
                vt.sgi.gi_image, vt.sgi.gi_view, vt.sgi.gi_filtered, vt.sgi.gi_filtered_view,
                [vt.sgi.sh_buf[0].handle, vt.sgi.sh_buf[1].handle], vt.sgi.raw_rays.handle, vt.sgi.gi_final, vt.sgi.gi_final_view,
                vt.depth_view,
                vt.hdr_image, vt.hdr_view, vt.post_set,
            )
        };
        let frame_set = self.frame_sets[self.frame_index];
        let sgi_fresh = self.viewport_target.as_ref().is_some_and(|t| t.sgi_fresh);

        // Deferred resolve for the viewport: render the reflectance G-buffer
        // (gbuf=true variants) and resolve SSR + volumetric fog. UNCOUPLED from the
        // GI backend — reflections/fog run from their own settings off the
        // viewport's own depth + G-buffer, so changing/disabling GI never turns
        // them off. (The screen-GI Flux trace below stays GI-gated.)
        let rt_on = self.flux_settings.reflection_mode == 2 && self.rt_reflect.is_some();
        let ssr_on =
            self.flux_settings.reflection_mode == 1 && self.flux_settings.ssr_enabled;
        let fog_on = input.fog.is_some_and(|f| f.density > 0.0);
        let vt_gbuf = (rt_on || ssr_on || fog_on || !input.decals.is_empty()) && self.viewport_ssr.is_some();
        // TAA needs the G-buffer attachment too (velocity is the third MRT).
        let taa_vp = self.aa.plan.taa == Some(aa::TARGET_VIEWPORT)
            && self.viewport_ssr.is_some()
            && self.aa.targets[aa::TARGET_VIEWPORT].as_ref().is_some_and(|t| t.motion.is_some());
        let vt_gbuf = vt_gbuf || taa_vp;
        let (vp_gbuf_image, vp_gbuf_view) = self
            .viewport_ssr
            .as_ref()
            .map_or((vk::Image::null(), vk::ImageView::null()), |t| {
                (t.gbuf_image, t.gbuf_view)
            });

        // 1) Flux trace (editor camera -> the viewport target's sgi), reusing the
        //    main-camera temporal state. Its depth prepass writes the viewport's
        //    own depth target, which the scene pass then reuses (EQUAL test, each
        //    pixel shaded once).
        let mut prepassed = false;
        if self.gi_has_gdf() {
            let view_proj = self.last_frame_ubo.as_ref()
                .map(|u| Mat4::from_cols_array_2d(&u.view_proj))
                .unwrap_or(input.camera.proj * input.camera.view);
            // Fresh targets count as a camera move: trace now, no history.

            {
                self.sgi_frame = self.sgi_frame.wrapping_add(1);
                let hist_valid = self.sgi_history_valid && !sgi_fresh;
                let cur = self.sgi_parity & 1;
                let prev_vp = self.sgi_last_view_proj.unwrap_or(view_proj);
                let prev_cam = if self.sgi_last_view_proj.is_some() {
                    self.sgi_last_cam
                } else {
                    input.camera.position
                };
                let s = self.flux_settings.smoothing.clamp(0.0, 1.0);
                let alpha = if hist_valid { 0.08 - 0.04 * s } else { 1.0 };
                let seed = self.sgi_frame;
                let flip = self.screen_gi_flip_y;
                self.gpu_zone_begin(cb, gpu_zone::GI);
                let spare = self.sgi_spare[self.frame_index].take();
                // FluxRT: the hardware gather (the editor view used to always fall
                // back to the software march here).
                let gi_rt = if self.flux_settings.rt_trace && self.ctx.ray_tracing() {
                    let (rt_insts, _) = self.rt_scene(input);
                    let rt_water = self.rt_water(input, &rt_insts);
                    build_gi_rt_tlas(
                        &self.ctx, self.allocator.clone(), self.command_pool, &self.meshes,
                        &mut self.rt_reflect, &rt_insts, &rt_water, cb, self.frame_index,
                    )
                } else {
                    None
                };
                let t = self.record_flux_trace(
                    device, cb, frame_set, 0, input, &input.camera, dep_img, dep_view, dep_att_view, s_probe,
                    s_img[cur], s_view[cur], s_view[cur ^ 1], s_filtered_img, s_filtered_view,
                    s_sh_buf[cur], s_sh_buf[cur ^ 1], s_raw_rays, s_final_img, s_final_view,
                    self.viewport_target.as_ref().unwrap().sgi.temporal(cur), gi_rt, ext, hist_valid, prev_vp, prev_cam,
                    alpha, seed, flip, spare,
                )?;
                prepassed = true;
                self.gpu_zone_end(cb, gpu_zone::GI);
                if let Some(t) = t {
                    self.sgi_transients[self.frame_index] = Some(t);
                }
                if let Some(vt) = self.viewport_target.as_mut() {
                    vt.sgi_fresh = false;
                }
                self.sgi_history_valid = true;
                self.sgi_last_view_proj = Some(view_proj);
                self.sgi_last_cam = input.camera.position;
                self.sgi_parity ^= 1;
            }
            let _ = (self.sgi_parity ^ 1) & 1;
            // Forward samples the INTEGRATED full-res GI (persists across skip frames).
            let info = [vk::DescriptorImageInfo::default()
                .sampler(self.lightmap_sampler)
                .image_view(self.viewport_target.as_ref().unwrap().sgi.temporal((self.sgi_parity^1)&1).current_views[0])
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(frame_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&info);
            unsafe { device.update_descriptor_sets(&[write], &[]) };
        }

        // Screen-space AO (ao.rs): from the Flux prepass's depth, else its own.
        if self.ao.plan.is_some_and(|p| p.target == ao::TARGET_VIEWPORT) {
            if !prepassed {
                match self.record_depth_prepass(device, cb, frame_set, input, &input.camera, dep_img, dep_view, dep_att_view, ext) {
                    Ok(()) => prepassed = true,
                    Err(e) => tracing::warn!("viewport AO depth prepass failed: {e:#}"),
                }
            }
            if prepassed {
                self.record_ao(device, cb, ao::TARGET_VIEWPORT, dep_view);
            }
        }

        // 2) Scene color pass into the viewport's HDR target (post pass tonemaps
        //    + blooms into `color` afterwards).
        self.gpu_zone_begin(cb, gpu_zone::SCENE);
        let (vp_vel_img, vp_vel_view, vp_rea_img, vp_rea_view) = self.aa.targets[aa::TARGET_VIEWPORT]
            .as_ref()
            .and_then(|t| t.motion.as_ref())
            .filter(|_| taa_vp)
            .map_or((vk::Image::null(), vk::ImageView::null(), vk::Image::null(), vk::ImageView::null()), |m| (m.0.image, m.0.view, m.1.image, m.1.view));
        self.velocity_active = taa_vp;
        unsafe {
            // The viewport's HDR / G-buffer / colour targets exist once, not per frame
            // slot: this frame's writes must wait for the previous frame's post / bloom /
            // resolve / egui reads of them (a TOP_OF_PIPE source waits for nothing).
            image_barrier(
                device, cb, hdr_img, vk::ImageAspectFlags::COLOR,
                vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::MEMORY_WRITE,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            );
            if prepassed {
                // Keep the prepass depth (the GI trace sampled it).
                image_barrier(
                    device, cb, dep_img, depth_aspect(),
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, depth_attach_layout(),
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    vk::AccessFlags2::MEMORY_WRITE,
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                );
            } else {
                image_barrier(
                    device, cb, dep_img, depth_aspect(),
                    vk::ImageLayout::UNDEFINED, depth_attach_layout(),
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    vk::AccessFlags2::MEMORY_WRITE,
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                );
            }
            if vt_gbuf {
                image_barrier(
                    device, cb, vp_gbuf_image, vk::ImageAspectFlags::COLOR,
                    vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::MEMORY_WRITE,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                );
            }
            let color_att = vk::RenderingAttachmentInfo::default()
                .image_view(hdr_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.016, 0.016, 0.024, 1.0] },
                });
            let gbuf_att = vk::RenderingAttachmentInfo::default()
                .image_view(vp_gbuf_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0; 4] },
                });
            // Stored: the reflection / fog resolve reads it. (The Flux depth
            // prepass is only drawn while screen GI traces, so it can be stale
            // or, after a resize, uninitialised.)
            let depth_att = vk::RenderingAttachmentInfo::default()
                .image_view(dep_att_view)
                .image_layout(depth_attach_layout())
                .load_op(if prepassed { vk::AttachmentLoadOp::LOAD } else { vk::AttachmentLoadOp::CLEAR })
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
                });
            let stencil_att = stencil_attachment(&depth_att);
            let mut color_atts: Vec<vk::RenderingAttachmentInfo> = if vt_gbuf {
                vec![color_att, gbuf_att]
            } else {
                vec![color_att]
            };
            if taa_vp {
                for img in [vp_vel_img, vp_rea_img] {
                    image_barrier(
                        device, cb, img, vk::ImageAspectFlags::COLOR,
                        vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                        vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::MEMORY_WRITE,
                        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    );
                }
                let clear = |v: f32| vk::ClearValue { color: vk::ClearColorValue { float32: [v, v, 0.0, 0.0] } };
                let att = |view: vk::ImageView, v: f32| {
                    vk::RenderingAttachmentInfo::default()
                        .image_view(view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(clear(v))
                };
                color_atts.push(att(vp_vel_view, aa::VELOCITY_SENTINEL));
                color_atts.push(att(vp_rea_view, 0.0));
            }
            let info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: ext })
                .layer_count(1)
                .color_attachments(&color_atts)
                .depth_attachment(&depth_att)
                .stencil_attachment(&stencil_att);
            device.cmd_begin_rendering(cb, &info);
            let viewport = vk::Viewport {
                x: 0.0,
                y: ext.height as f32,
                width: ext.width as f32,
                height: -(ext.height as f32),
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.cmd_set_viewport(cb, 0, &[viewport]);
            device.cmd_set_scissor(cb, 0, &[vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: ext }]);
        }
        self.pipeline_cache.set_target_format(post::HDR_FORMAT);
        if input.draw_skybox {
            // HDR target: no inline tonemap (the post pass tonemaps once).
            self.record_skybox(device, cb, frame_set, true, vt_gbuf)?;
        }
        // Capture the draw counts: the editor renders the scene HERE, so this is
        // where the profiler's draw-call stats come from.
        let main_order = frustum_cull(&self.gpu_cpu_order(order), input.draws, input.camera.proj * input.camera.view);
        if prepassed {
            self.draw_mode = DrawMode::DepthEqual;
        }
        let pass = MainPass {
            color_image: hdr_img,
            color_view: hdr_view,
            gbuf_view: vt_gbuf.then_some(vp_gbuf_view),
            velocity_view: taa_vp.then_some(vp_vel_view),
            reactive_view: taa_vp.then_some(vp_rea_view),
            depth_image: dep_img,
            depth_view: dep_att_view,
            depth_read_view: dep_view,
            extent: ext,
        };
        self.gpu_arm(true);
        let stats = self.record_main_scene_draws(device, cb, &main_order, input, frame_set, vt_gbuf, &pass, 0);
        self.gpu_arm(false);
        self.draw_mode = DrawMode::Normal;
        self.velocity_active = false;
        self.viewport_draw_stats = stats?;
        unsafe { device.cmd_end_rendering(cb) };
        if !input.decals.is_empty() && vt_gbuf {
            let decal_frame = self.decal_frame(input);
            self.record_decals(
                device, cb, input, hdr_view,
                dep_img, dep_view, vp_gbuf_image, vp_gbuf_view, ext, &decal_frame,
            );
        }
        self.gpu_zone_end(cb, gpu_zone::SCENE);
        if let Some(fx) = input.postfx.underwater.as_ref() {
            self.underwater.record(
                device, cb, underwater::SLOT_VIEWPORT + self.frame_index as u32,
                (hdr_img, hdr_view), (dep_img, dep_view), ext,
                &input.camera, fx, &input.light, input.time, 0,
            );
        }
        // HDR -> shader-read; then the SSR resolve (if active) + post pass into `color`.
        image_barrier(
            device, cb, hdr_img, vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ,
        );
        image_barrier(
            device, cb, dep_img, depth_aspect(),
            depth_attach_layout(), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS, vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ,
        );
        // Selection mask (before the Hi-Z build: the mask pass only reads depth).
        let outline_draws: Vec<usize> = if self.selection.style.enabled {
            // Secondary selections first, the primary last (it wins overlaps).
            let mut v: Vec<usize> = (0..input.draws.len()).filter(|&i| input.draws[i].highlight > 0.0).collect();
            v.sort_by(|&a, &b| input.draws[a].highlight.total_cmp(&input.draws[b].highlight));
            v
        } else {
            Vec::new()
        };
        let outline_on = !outline_draws.is_empty() && self.record_selection_mask(device, cb, input, &outline_draws, dep_img, dep_att_view, ext)?;
        // Next frame's occlusion pyramid from this pass's depth.
        self.gpu_hiz_build(device, cb, input, (dep_img, dep_view), false);
        // Reflection resolve into the viewport's resolved HDR target: ray-traced
        // (mode 2) takes precedence, else the SSR/fog resolve.
        self.gpu_zone_begin(cb, gpu_zone::REFLECT);
        let vp_post_set = if rt_on {
            image_barrier(
                device, cb, vp_gbuf_image, vk::ImageAspectFlags::COLOR,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_READ,
            );
            let (instances, lights) = self.rt_scene(input);
            let cam = &input.camera;
            let mut refl_center = [0.0; 4];
            let mut refl_extents = [0.0; 4];
            let mut refl_box = [0.0; 4];
            if let Some(p) = input.reflection_probe {
                refl_center = [p.capture[0], p.capture[1], p.capture[2], p.intensity.max(0.0)];
                refl_extents = [
                    p.half_extents[0], p.half_extents[1], p.half_extents[2],
                    if p.box_projection { 1.0 } else { 0.0 },
                ];
                refl_box = [p.center[0], p.center[1], p.center[2], 0.0];
            }
            let fs = self.flux_settings.for_view(input.postfx.flux);
            let refl2 = refl2_uniforms(input);
            let rt_uniforms = rt_reflect::RtUniforms {
                // Match the jittered projection that produced depth/G-buffer.
                inv_proj: self.last_frame_ubo.as_ref().map_or_else(|| cam.proj.inverse().to_cols_array_2d(),|u|u.inv_proj),
                inv_view: cam.view.inverse().to_cols_array_2d(),
                camera_pos: cam.position.extend(0.0).to_array(),
                refl_center,
                refl_extents,
                refl_box,
                // w: env IBL intensity, so misses fall back to the sky like the game path.
                params: [fs.ssr_roughness_cutoff, fs.ssr_intensity, lights.len() as f32, input.light.env_intensity],
                screen: [ext.width as f32, ext.height as f32, 0.0, fs.bounces as f32],
                quality: rt_quality(&fs, self.frame_serial),
                lighting_mode: [fs.ray_lighting_mode as u32, 0, 0, 0],
                refl2_center: refl2.refl2_center,
                refl2_extents: refl2.refl2_extents,
                refl2_box: refl2.refl2_box,
                refl3_center: refl2.refl3_center,
                refl3_extents: refl2.refl3_extents,
                refl3_box: refl2.refl3_box,
                refl4_center: refl2.refl4_center,
                refl4_extents: refl2.refl4_extents,
                refl4_box: refl2.refl4_box,
                refl_blend: refl2.refl_blend,
                refl_blend2: refl2.refl_blend2,
                water: [input.water_time, input.water.len().min(MAX_WATER_BODIES) as f32, 0.0, 0.0],
                water_bodies: water_bodies(input),
            };
            let env_view = self.reflection_env.view;
            let target = self.viewport_ssr.take().unwrap();
            let resolved_image = target.resolved_image();
            let resolved_view = target.resolved_view();
            let post_set = target.post_set;
            let allocator = self.allocator.clone();
            let frame_index = self.frame_index;
            let command_pool = self.command_pool;
            let rt_water = self.rt_water(input, &instances);
            let tex_views = self.rt_texture_views();
            if let (Some(rt), Some(allocator)) = (self.rt_reflect.as_mut(), allocator)
                && let Err(e) = rt.record(
                    &self.ctx, &allocator, command_pool, cb, frame_index, 0, self.frame_serial, &self.meshes,
                    &instances, &rt_water, &lights, resolved_image, resolved_view, hdr_view, dep_view,
                    vp_gbuf_view, env_view, self.reflection_env_b.view, self.reflection_env_c.view, self.reflection_env_d.view, &tex_views, (self.lightmaps.view, self.lightmap_sampler), (self.area_atlas.view,self.sampler),
                                self.flux_surface.as_ref().and_then(|s| s.reflection_resources(self.frame_index, self.frame_serial)), &rt_uniforms, ext,
                ) {
                    tracing::warn!("viewport RT reflection failed: {e:#}");
                }
            self.tex_view_cache = tex_views;
            image_barrier(
                device, cb, resolved_image, vk::ImageAspectFlags::COLOR,
                vk::ImageLayout::GENERAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ,
            );
            self.viewport_ssr = Some(target);
            post_set
        } else if ssr_on || fog_on {
            image_barrier(
                device, cb, vp_gbuf_image, vk::ImageAspectFlags::COLOR,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ,
            );
            let uniforms = self.ssr_uniforms(input, ext, ssr_on);
            let env_view = self.reflection_env.view;
            let mut target = self.viewport_ssr.take().unwrap();
            self.ssr_resolve
                .record(device, cb, &mut target, hdr_view, dep_view, env_view, self.reflection_env_b.view, self.reflection_env_c.view, self.reflection_env_d.view, &uniforms, self.frame_index);
            let set = target.post_set;
            self.viewport_ssr = Some(target);
            set
        } else {
            post_set
        };
        // TAA before bloom + post (exclusive with nothing here: the editor
        // viewport has no FSR): the post pass samples the resolved image.
        let vp_post_set = if taa_vp {
            for img in [vp_vel_img, vp_rea_img] {
                image_barrier(
                    device, cb, img, vk::ImageAspectFlags::COLOR,
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ,
                );
            }
            let color = if rt_on || ssr_on || fog_on { self.viewport_ssr.as_ref().map_or(hdr_view, |t| t.resolved_view()) } else { hdr_view };
            let inputs = aa::AaInputs { color, depth: dep_view, velocity: vp_vel_view, reactive: vp_rea_view };
            match self.last_frame_ubo {
                Some(ubo) => self.record_taa(device, cb, aa::TARGET_VIEWPORT, &ubo, inputs).unwrap_or(vp_post_set),
                None => vp_post_set,
            }
        } else {
            vp_post_set
        };
        self.gpu_zone_end(cb, gpu_zone::REFLECT);
        self.gpu_zone_begin(cb, gpu_zone::POST);
        let vp_bloom = {
            self.post_pass.exposure.record(device,cb,vp_post_set,ext,input.postfx.auto_exposure,0,
                self.frame_serial,self.started.elapsed().as_secs_f64());
            let chain = self.viewport_target.as_ref().unwrap().bloom.as_ref().unwrap();
            self.post_pass.record_bloom(device, cb, chain, vp_post_set, &input.postfx);
            chain.output_set()
        };
        image_barrier(
            device, cb, col_img, vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::MEMORY_WRITE,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        );
        unsafe {
            let post_att = vk::RenderingAttachmentInfo::default()
                .image_view(col_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE);
            let atts = [post_att];
            let info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: ext })
                .layer_count(1)
                .color_attachments(&atts);
            device.cmd_begin_rendering(cb, &info);
            let wobble = underwater::wobble(input.postfx.underwater.as_ref(), &input.camera, input.time);
            self.aa_post_begin();
            self.post_pass.record_set(device, cb, vp_post_set, vp_bloom, ext, &input.postfx, wobble);
            self.aa_post_end();
            // Editor ground grid over the tonemapped scene, under the outline.
            if self.grid.style.enabled {
                let fmt = self.swapchain.format.format;
                let (view, proj) = (input.camera.view, input.camera.proj);
                if let Err(e) = self.grid.record(device, cb, self.frame_index, fmt, ext, dep_view, view, proj) {
                    tracing::warn!("editor grid failed: {e:#}");
                }
            }
            if outline_on {
                let fmt = self.swapchain.format.format;
                if let Err(e) = self.selection.record_composite(device, cb, fmt, ext) {
                    tracing::warn!("selection outline composite failed: {e:#}");
                }
            }
            device.cmd_end_rendering(cb);
        }
        image_barrier(
            device, cb, col_img, vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ,
        );
        self.gpu_zone_end(cb, gpu_zone::POST);
        Ok(())
    }

    fn decal_frame(&self, input: &FrameInput) -> decal::DecalFrame {
        let proj = jitter_proj(
            input.camera.proj,
            self.jitter,
            [self.render_extent[0] as f32, self.render_extent[1] as f32],
        );
        let inv_view = input.camera.view.inverse();
        decal::DecalFrame {
            inv_view_proj: (inv_view * proj.inverse()).to_cols_array_2d(),
            inv_view: inv_view.to_cols_array_2d(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record_decals(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        input: &FrameInput,
        scene_view: vk::ImageView,
        depth_image: vk::Image,
        depth_view: vk::ImageView,
        gbuf_image: vk::Image,
        gbuf_view: vk::ImageView,
        extent: vk::Extent2D,
        frame: &decal::DecalFrame,
    ) {
        if input.decals.is_empty() {
            return;
        }
        image_barrier(
            device, cb, depth_image, depth_aspect(),
            depth_attach_layout(), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS, vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ,
        );
        image_barrier(
            device, cb, gbuf_image, vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ,
        );
        self.decal.record(
            device, cb, self.frame_index, scene_view, depth_view, gbuf_view, extent,
            frame, input.decals, &self.textures, self.default_textures[0].view,
        );
        image_barrier(
            device, cb, depth_image, depth_aspect(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, depth_attach_layout(),
            vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ,
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );
        image_barrier(
            device, cb, gbuf_image, vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        );
    }
}

/// Bind a graphics pipeline (small helper to keep the draw loop tidy).
/// Second reflection probe for the passes that composite over the forward
/// reflection (SSR, RT): center, extents, box, blend, matching the frame UBO.
/// This frame's water bodies, padded to the UBO array.
fn water_bodies(input: &FrameInput) -> [GpuWater; MAX_WATER_BODIES] {
    let mut bodies = [GpuWater::default(); MAX_WATER_BODIES];
    let n = input.water.len().min(MAX_WATER_BODIES);
    bodies[..n].copy_from_slice(&input.water[..n]);
    bodies
}

/// Reflection-probe data for the passes that composite over the forward
/// reflection (SSR, RT), matching the frame UBO's probe slots 1-3.
struct ReflProbeUniforms {
    refl2_center: [f32; 4],
    refl2_extents: [f32; 4],
    refl2_box: [f32; 4],
    refl3_center: [f32; 4],
    refl3_extents: [f32; 4],
    refl3_box: [f32; 4],
    refl4_center: [f32; 4],
    refl4_extents: [f32; 4],
    refl4_box: [f32; 4],
    refl_blend: [f32; 4],
    refl_blend2: [f32; 4],
}

fn refl_slot_uniforms(p: &ReflectionProbeBox) -> ([f32; 4], [f32; 4], [f32; 4]) {
    (
        [p.capture[0], p.capture[1], p.capture[2], p.intensity.max(0.0)],
        [
            p.half_extents[0],
            p.half_extents[1],
            p.half_extents[2],
            if p.box_projection { 1.0 } else { 0.0 },
        ],
        [p.center[0], p.center[1], p.center[2], 0.0],
    )
}

/// Probe slots 1-3 for the SSR/RT reflection resolve, plus the per-slot blend
/// distances. The third/fourth slots come straight from `reflection_probes`, so
/// those passes can blend the same 3+ probes the forward pass already blends.
fn refl2_uniforms(input: &FrameInput) -> ReflProbeUniforms {
    let mut refl_blend = [
        input
            .reflection_probe
            .map_or(0.0, |p| p.blend_distance.max(0.0)),
        0.0,
        0.0,
        0.0,
    ];
    let (refl2_center, refl2_extents, refl2_box) = input
        .reflection_probe_b
        .as_ref()
        .map(refl_slot_uniforms)
        .unwrap_or(([0.0; 4], [0.0; 4], [0.0; 4]));
    if let Some(p) = input.reflection_probe_b {
        refl_blend[1] = p.blend_distance.max(0.0);
        refl_blend[2] = input.reflection_primary.min(1) as f32;
    }
    let (refl3_center, refl3_extents, refl3_box) = input
        .reflection_probes
        .get(2)
        .copied()
        .flatten()
        .as_ref()
        .map(refl_slot_uniforms)
        .unwrap_or(([0.0; 4], [0.0; 4], [0.0; 4]));
    let (refl4_center, refl4_extents, refl4_box) = input
        .reflection_probes
        .get(3)
        .copied()
        .flatten()
        .as_ref()
        .map(refl_slot_uniforms)
        .unwrap_or(([0.0; 4], [0.0; 4], [0.0; 4]));
    let mut refl_blend2 = [0.0; 4];
    if let Some(p) = input.reflection_probes.get(2).copied().flatten() {
        refl_blend2[0] = p.blend_distance.max(0.0);
    }
    if let Some(p) = input.reflection_probes.get(3).copied().flatten() {
        refl_blend2[1] = p.blend_distance.max(0.0);
    }
    ReflProbeUniforms {
        refl2_center,
        refl2_extents,
        refl2_box,
        refl3_center,
        refl3_extents,
        refl3_box,
        refl4_center,
        refl4_extents,
        refl4_box,
        refl_blend,
        refl_blend2,
    }
}

fn device_bind(device: &ash::Device, cb: vk::CommandBuffer, pipeline: vk::Pipeline) {
    unsafe {
        device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
    }
}

/// Create a host-visible uniform buffer holding one [`MaterialFx`] block.
fn make_fx_ubo(
    device: &ash::Device,
    alloc: &mut Allocator,
    fx: MaterialFx,
    name: &str,
) -> Result<Buffer> {
    let mut buf = Buffer::new(
        device,
        alloc,
        size_of::<MaterialFx>() as u64,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        MemoryLocation::CpuToGpu,
        name,
    )?;
    buf.write(0, bytemuck::bytes_of(&fx));
    Ok(buf)
}

/// A parked GPU destroy (see `Renderer::defer_free`).
type DeferredFree = Box<dyn FnOnce(&ash::Device, &mut Allocator) + Send>;

impl Drop for Renderer {
    fn drop(&mut self) {
        // Profiler window (its own surface/swapchain/egui) frees through the
        // shared allocator, so tear it down first while everything is still
        // alive. Done before binding `device` to avoid a borrow conflict.
        self.close_profiler_window();

        let device = &self.ctx.device;
        unsafe {
            let _ = device.wait_idle("wait:device_wait_idle@drop");
        }

        // egui renderers first: they free allocations through the shared
        // allocator, which must still be alive.
        drop(self.egui.take());
        drop(self.vr_egui.take());
        self.world_egui.clear();

        let allocator = self.allocator.as_ref().unwrap().clone();
        {
            let mut alloc = allocator.lock().unwrap();
            // Frees parked by `free_resources`: the device is idle (above).
            for (_, f) in self.deferred_frees.drain(..) {
                f(device, &mut alloc);
            }
            for (_, _, f) in self.shared_frees.drain(..) {
                f(device, &mut alloc);
            }
            for (i, mesh) in self.meshes.iter_mut().enumerate() {
                if self.freed_meshes.contains(&i) {
                    continue;
                }
                mesh.vertex_buffer.destroy(device, &mut alloc);
                mesh.index_buffer.destroy(device, &mut alloc);
            }
            if let Some(t) = &mut self.text_gpu {
                t.destroy(device, &mut alloc);
            }
            if let Some(s) = &mut self.sprite_gpu {
                s.destroy(device, &mut alloc);
            }
            if let Some(i) = &mut self.impostor_gpu {
                i.destroy(device, &mut alloc);
            }
            for ring in &mut self.instance_rings {
                ring.destroy(device, &mut alloc);
            }
            if let Some(g) = &mut self.gpu_cull {
                g.destroy(device, &mut alloc);
            }
            if let Some(mut c) = self.cluster.take() {
                c.destroy(device, &mut alloc);
            }
            for t in &mut self.camera_targets {
                t.depth.destroy(device, &mut alloc);
                for u in &mut t.ubos {
                    u.destroy(device, &mut alloc);
                }
            }
            if self.camera_target_pool != vk::DescriptorPool::null() {
                unsafe { device.destroy_descriptor_pool(self.camera_target_pool, None) };
            }
            for (i, texture) in self.textures.iter_mut().enumerate() {
                if self.freed_textures.contains(&i) {
                    continue;
                }
                texture.destroy(device, &mut alloc);
            }
            if let Some(skinner) = &mut self.skinner {
                skinner.destroy(device, &mut alloc);
            }
            for texture in &mut self.default_textures {
                texture.destroy(device, &mut alloc);
            }
            if let Some(texture) = &mut self.skybox_texture {
                texture.destroy(device, &mut alloc);
            }
            self.shadow_atlas.destroy(device, &mut alloc);
            self.lightmaps.destroy(device, &mut alloc);
            self.probe_buffer.destroy(device, &mut alloc);
            for (mut b, _) in self.probe_staging.drain(..).chain(self.tex_staging.drain(..)).flatten() {
                b.destroy(device, &mut alloc);
            }
            if let Some(surface) = &mut self.flux_surface {
                surface.destroy(device, &mut alloc);
            }
            if let Some(history)=&mut self.flux_screen_history {history.destroy(device,&mut alloc);}
            // Free any deferred Flux trace resources still held (the device idle
            // wait above guarantees the GPU is done with them).
            for slot in self.sgi_spare.iter_mut().chain(self.preview_sgi_spare.iter_mut()) {
                if let Some(mut t) = slot.take() {
                    t.destroy(device, &mut alloc);
                }
            }
            for slot in &mut self.sgi_transients {
                if let Some(mut t) = slot.take() {
                    t.destroy(device, &mut alloc);
                }
            }
            for slot in &mut self.preview_sgi_transients {
                if let Some(mut t) = slot.take() {
                    t.destroy(device, &mut alloc);
                }
            }
            if let Some(gpu_gi) = &mut self.gpu_gi {
                gpu_gi.destroy(device, &mut alloc);
            }
            for ubo in &mut self.frame_ubos {
                ubo.destroy(device, &mut alloc);
            }
            for (i, material) in self.materials.iter_mut().enumerate() {
                if self.freed_materials.contains(&i) {
                    continue;
                }
                material.fx_ubo.destroy(device, &mut alloc);
                if let Some(mut ubo) = material.custom_ubo.take() {
                    ubo.destroy(device, &mut alloc);
                }
            }
            self.default_fx_ubo.destroy(device, &mut alloc);
            if let Some(mut preview) = self.camera_preview.take() {
                preview.destroy(device, &mut alloc);
            }
            if let Some(mut preview) = self.model_preview.take() {
                preview.destroy(device, &mut alloc);
            }
            if let Some(mut preview) = self.capture_target.take() {
                preview.destroy(device, &mut alloc);
            }
            for mut r in self.readbacks.drain(..) {
                r.destroy(device, &mut alloc);
            }
            for (mut preview, _) in self.retired_model_previews.drain(..) {
                preview.destroy(device, &mut alloc);
            }
            if let Some(mut vt) = self.viewport_target.take() {
                vt.destroy(device, &mut alloc);
            }
            for (mut vt, _) in self.retired_viewport_targets.drain(..) {
                vt.destroy(device, &mut alloc);
            }
            for (mut t, _) in self.retired_viewport_ssr.drain(..) {
                t.destroy(device, &mut alloc);
            }
            if let Some(mut t) = self.viewport_ssr.take() {
                t.destroy(device, &mut alloc);
            }
            if let Some(mut x) = self.xr_render.take() {
                x.destroy(device, &mut alloc, self.command_pool);
            }
            if let Some(mut o) = self.vr_overlay.take() {
                o.destroy(device, &mut alloc);
            }
            self.depth.destroy(device, &mut alloc);
            if let Some(mut c) = self.opaque_copy.take() {
                c.destroy(device, &mut alloc);
            }
            self.sgi.destroy(device, &mut alloc);
            self.ao.destroy(device, &mut alloc);
            self.aa.destroy(device, &mut alloc);
            for t in &mut self.ssr_targets {
                t.destroy(device, &mut alloc);
            }
            for t in &mut self.velocity_targets {
                t.destroy(device, &mut alloc);
            }
            for t in &mut self.reactive_targets {
                t.destroy(device, &mut alloc);
            }
            if let Some(mut f) = self.fsr3.take() {
                f.destroy();
            }
            if let Some(mut f) = self.fsr3_fg.take() {
                f.destroy();
            }
            for t in &mut self.upscale_outputs {
                t.destroy(device, &mut alloc);
            }
            for t in &mut self.fg_present {
                t.destroy(device, &mut alloc);
            }
            for t in &mut self.fg_output {
                t.destroy(device, &mut alloc);
            }
            if self.upscale_pool != vk::DescriptorPool::null() {
                unsafe { device.destroy_descriptor_pool(self.upscale_pool, None) };
            }
            self.ssr_resolve.destroy(device);
            self.underwater.destroy(device, &mut alloc);
            self.selection.destroy(device, &mut alloc);
            self.grid.destroy(device);
            self.decal.destroy(device, &mut alloc);
            if let Some(mut rt) = self.rt_reflect.take() {
                rt.destroy(&self.ctx, &mut alloc);
            }
            if let Some(mut sim) = self.water_sim.take() {
                sim.destroy(device, &mut alloc);
            }
            for (_, (view, _)) in self.ripple_mips.drain() {
                unsafe { device.destroy_image_view(view, None) };
            }
            self.post_pass.destroy(device, &mut alloc);
            self.reflection_env.destroy(device, &mut alloc);
            self.reflection_env_b.destroy(device, &mut alloc);
            // Probe slots 2 and 3 (bindings 11/12) were never freed: the two
            // 'reflection env cube' allocations gpu_allocator reported at exit.
            self.reflection_env_c.destroy(device, &mut alloc);
            self.reflection_env_d.destroy(device, &mut alloc);
            for ubo in &mut self.refl_capture_ubos {
                ubo.destroy(device, &mut alloc);
            }
            if let Some(mut t) = self.refl_target.take() {
                t.destroy(device, &mut alloc);
            }
            for ubo in &mut self.planar_ubos {
                ubo.destroy(device, &mut alloc);
            }
            if let Some(mut t) = self.planar_targets.take() {
                t.destroy(device, &mut alloc);
            }
        }
        unsafe {
            device.destroy_descriptor_pool(self.refl_capture_pool, None);
        }

        unsafe {
            self.sampler_cache.destroy(device);
            device.destroy_sampler(self.sampler, None);
            device.destroy_sampler(self.lightmap_sampler, None);
            device.destroy_sampler(self.grab_sampler, None);
            self.area_atlas.destroy(device, &mut self.allocator.as_ref().unwrap().lock().unwrap());
            self.ltc_lut.destroy(device, &mut self.allocator.as_ref().unwrap().lock().unwrap());
            self.ripple_dummy.destroy(device, &mut self.allocator.as_ref().unwrap().lock().unwrap());
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_pool(self.material_pool, None);
            device.destroy_descriptor_pool(self.custom_pool, None);
            device.destroy_descriptor_pool(self.skybox_pool, None);
        }
        self.pipeline_cache.destroy(device);
        for frame in &mut self.frames {
            frame.destroy(device);
        }
        unsafe {
            device.destroy_command_pool(self.command_pool, None);
        }
        self.swapchain.destroy(device);

        drop(allocator);
        self.allocator = None; // Allocator drops here, before GpuContext's device
    }
}

#[allow(clippy::too_many_arguments)]
/// Combined samplers in one set-0 descriptor set: bindings 1,3-9,11,12 and
/// binding 10's ripple field per water body.
// Bindings 1, 3-9, 11-13 and the water ripple array (13 = screen-space AO).
const SET0_SAMPLERS: u32 = 11 + MAX_WATER_BODIES as u32;
/// Set-0 binding of the water bodies' ripple fields (by body slot).
const RIPPLE_BINDING: u32 = 10;
/// Set-0 bindings of reflection probe cubes (slot order).
const REFLECTION_BINDINGS: [u32; MAX_REFLECTION_PROBES] = [7, 9, 11, 12];

/// Binding 10's descriptors: `views` by water body slot, sampled with `sampler`.
fn ripple_infos(sampler: vk::Sampler, views: [vk::ImageView; MAX_WATER_BODIES]) -> [vk::DescriptorImageInfo; MAX_WATER_BODIES] {
    views.map(|v| vk::DescriptorImageInfo::default().sampler(sampler).image_view(v).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL))
}

fn ripple_write(set: vk::DescriptorSet, infos: &[vk::DescriptorImageInfo]) -> vk::WriteDescriptorSet<'_> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(RIPPLE_BINDING)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(infos)
}

/// Copy of binding 10 (every body slot) from `src` to `dst`.
fn ripple_copy(src: vk::DescriptorSet, dst: vk::DescriptorSet) -> vk::CopyDescriptorSet<'static> {
    vk::CopyDescriptorSet::default()
        .src_set(src)
        .src_binding(RIPPLE_BINDING)
        .dst_set(dst)
        .dst_binding(RIPPLE_BINDING)
        .descriptor_count(MAX_WATER_BODIES as u32)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn image_barrier(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
) {
    image_barrier_layers(device, cb, image, aspect, 0, 1, old_layout, new_layout, src_stage, src_access, dst_stage, dst_access);
}

/// [`image_barrier`] over array layers `base..base + count`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn image_barrier_layers(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    base: u32,
    count: u32,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
) {
    let image_barrier = vk::ImageMemoryBarrier2::default()
        .image(image)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .level_count(1)
                .base_array_layer(base)
                .layer_count(count),
        );
    let barriers = [image_barrier];
    let dependency = vk::DependencyInfo::default().image_memory_barriers(&barriers);
    unsafe {
        device.cmd_pipeline_barrier2(cb, &dependency);
    }
}

#[cfg(test)]
mod refl_capture_tests {
    use super::{ReflectionEnv, CUBE_CAPTURE_FACES};
    use glam::{Mat4, Vec3, Vec4};

    // Simulate EXACTLY what the GPU does to capture one cube-face texel: render the
    // scene through `look_at_rh(center, center+dir, up)` + `perspective_rh(90deg)`
    // into a POSITIVE-height (non-Y-flipped) viewport, then read the texel at
    // face-plane coords (fx, fy). With a positive viewport the framebuffer row maps
    // straight to NDC y (ndc_y = fy) and the column to NDC x (ndc_x = fx), so the
    // world direction seen at that texel is the unprojection of NDC (fx, fy).
    fn capture_ray(face: usize, fx: f32, fy: f32) -> Vec3 {
        let (dir, up) = CUBE_CAPTURE_FACES[face];
        let center = Vec3::ZERO; // direction is translation-invariant
        let view = Mat4::look_at_rh(center, center + dir, up);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.05, 2000.0);
        let inv = (proj * view).inverse();
        let unproject = |d: f32| {
            let p = inv * Vec4::new(fx, fy, d, 1.0);
            p.truncate() / p.w
        };
        // Vulkan clip-space depth is [0,1]: near plane d=0, far plane d=1.
        (unproject(1.0) - unproject(0.0)).normalize()
    }

    // THE invariant: the captured cube must equal the sampling convention, or every
    // reflection is mis-oriented. For all 6 faces and a spread of face-plane points,
    // the world direction the capture RENDERS at (fx,fy) must equal the direction
    // the sampler READS at (fx,fy) (`ReflectionEnv::face_dir`). If any face/axis is
    // flipped, this fails and names it.
    /// Vulkan polygon orientation (spec 28.7.1): a = -1/2 * sum(x_i*y_{i+1} -
    /// x_{i+1}*y_i) in framebuffer coordinates; with COUNTER_CLOCKWISE front
    /// face, a > 0 is front-facing.
    fn framebuffer_area(view_proj: Mat4, flip_y: bool, tri: [Vec3; 3]) -> f32 {
        let fb: Vec<(f32, f32)> = tri
            .iter()
            .map(|p| {
                let c = view_proj * p.extend(1.0);
                let (x, y) = (c.x / c.w, c.y / c.w);
                // Viewport: positive height maps ndc y -1..1 to fb 0..h; the
                // main pass's negative height maps it to h..0.
                let fy = if flip_y { (1.0 - y) * 0.5 } else { (y + 1.0) * 0.5 };
                ((x + 1.0) * 0.5, fy)
            })
            .collect();
        let mut sum = 0.0;
        for i in 0..3 {
            let (x0, y0) = fb[i];
            let (x1, y1) = fb[(i + 1) % 3];
            sum += x0 * y1 - x1 * y0;
        }
        -0.5 * sum
    }

    // Mirrored passes need the clockwise front face: an outward-facing (CCW
    // from outside, like the engine's meshes) triangle in front of the camera
    // is CCW-front in the main pass but reversed in the cube capture and in
    // a planar mirror. If this fails, `PipelineKey::mirrored` is wrong.
    #[test]
    fn mirrored_passes_reverse_winding() {
        // +Z face of a unit cube at z = -3, seen by a camera at the origin
        // looking down -Z: its outward normal points at the camera.
        let tri = [Vec3::new(-0.5, -0.5, -2.5), Vec3::new(0.5, -0.5, -2.5), Vec3::new(0.5, 0.5, -2.5)];
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.05, 100.0);
        let main_view = Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y);
        assert!(framebuffer_area(proj * main_view, true, tri) > 0.0, "main pass: CCW front");

        // Cube capture -Z face (index 5): positive-height viewport.
        let (dir, up) = CUBE_CAPTURE_FACES[5];
        let cap_view = Mat4::look_at_rh(Vec3::ZERO, dir, up);
        assert!(framebuffer_area(proj * cap_view, false, tri) < 0.0, "cube capture: winding reversed");

        // Planar mirror about y = -1 (a floor): reflected view, negative-height
        // viewport.
        let mirror = Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0))
            * Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0))
            * Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0));
        // The pass draws the real geometry through the reflected camera.
        let mirrored_view = main_view * mirror;
        assert!(framebuffer_area(proj * mirrored_view, true, tri) < 0.0, "planar mirror: winding reversed");
    }

    #[test]
    fn cube_capture_matches_sampling() {
        let samples = [
            (0.0f32, 0.0f32),
            (0.5, 0.0),
            (-0.5, 0.0),
            (0.0, 0.5),
            (0.0, -0.5),
            (0.6, -0.3),
            (-0.4, 0.7),
        ];
        for face in 0..6 {
            for &(fx, fy) in &samples {
                let got = capture_ray(face, fx, fy);
                let want = ReflectionEnv::face_dir(face, fx, fy);
                let want = Vec3::from_array(want);
                let err = (got - want).length();
                assert!(
                    err < 1e-4,
                    "face {face} at ({fx},{fy}): capture renders {got:?} but sampler reads \
                     {want:?} (err {err}) — cube is mis-oriented on this face/axis",
                );
            }
        }
    }

    // Center-of-a-mirror-sphere sanity (REFLECTION_PROBES_MATH.md 2.3): a surface
    // whose normal faces the camera must reflect the direction BEHIND the camera
    // (R == V), not in front. Guards against a flipped incident-vector sign.
    #[test]
    fn mirror_center_reflects_behind_camera() {
        // Camera looks down -Z at a surface facing it (N = +Z toward camera).
        let n = Vec3::Z;
        let v = Vec3::Z; // surface->camera, camera is on +Z side
        // GLSL reflect(-V, N) = 2*(N·V)*N - V
        let r = 2.0 * n.dot(v) * n - v;
        assert!(
            (r - v).length() < 1e-6,
            "center of a mirror facing the camera must reflect behind it (R==V), got {r:?}",
        );
    }
}

#[cfg(test)]
mod shadow_casting_tests {
    use super::{shadow_caster, bake, rt_reflect};
    use crate::{AlphaMode, DrawCmd, MaterialFeatures, MaterialHandle, MeshHandle, ShadowCasting};
    use glam::{Mat4, Vec3};

    fn draw(shadow_casting: ShadowCasting) -> DrawCmd {
        DrawCmd {
            gi_source: None,
            mesh: MeshHandle(0),
            material: MaterialHandle(0),
            transform: Mat4::IDENTITY,
            highlight: 0.0,
            mesh_center: Vec3::ZERO,
            bound_radius: 1.0,
            lightmap_layer: -1,
            lightmap_size: 0,
            static_geometry: false,
            planar_index: -1,
            occluded: false,
            shadow_casting,
            instance_color: None,
            instance_param: None,
        }
    }

    #[test]
    fn shadow_pass_skips_off_only() {
        let solid = MaterialFeatures::default();
        assert!(shadow_caster(&draw(ShadowCasting::On), &solid));
        assert!(shadow_caster(&draw(ShadowCasting::TwoSided), &solid));
        assert!(shadow_caster(&draw(ShadowCasting::ShadowsOnly), &solid));
        assert!(!shadow_caster(&draw(ShadowCasting::Off), &solid));
    }

    #[test]
    fn blended_and_water_never_cast() {
        let blend = MaterialFeatures { alpha_mode: AlphaMode::Blend, ..Default::default() };
        let water = MaterialFeatures { water: true, ..Default::default() };
        for mode in [ShadowCasting::On, ShadowCasting::TwoSided, ShadowCasting::ShadowsOnly] {
            assert!(!shadow_caster(&draw(mode), &blend));
            assert!(!shadow_caster(&draw(mode), &water));
        }
    }

    #[test]
    fn camera_passes_skip_shadows_only() {
        assert!(ShadowCasting::On.visible());
        assert!(ShadowCasting::Off.visible());
        assert!(ShadowCasting::TwoSided.visible());
        assert!(!ShadowCasting::ShadowsOnly.visible());
    }

    #[test]
    fn bake_masks_split_bounce_and_shadow_rays() {
        // Bit 0: bounce rays (visible), bit 1: shadow rays (caster).
        assert_eq!(bake::instance_mask(ShadowCasting::On), 0x03);
        assert_eq!(bake::instance_mask(ShadowCasting::TwoSided), 0x03);
        assert_eq!(bake::instance_mask(ShadowCasting::Off), 0x01);
        assert_eq!(bake::instance_mask(ShadowCasting::ShadowsOnly), 0x02);
    }

    #[test]
    fn rt_masks_split_reflection_and_shadow_rays() {
        // Bit 0: reflections + GI gather (solid), bit 2: shadow rays (caster).
        assert_eq!(rt_reflect::solid_mask(ShadowCasting::On), 0x05);
        assert_eq!(rt_reflect::solid_mask(ShadowCasting::TwoSided), 0x05);
        assert_eq!(rt_reflect::solid_mask(ShadowCasting::Off), 0x01);
        assert_eq!(rt_reflect::solid_mask(ShadowCasting::ShadowsOnly), 0x04);
    }
}

#[cfg(test)]
pub(crate) mod cascade_tests {
    use super::*;

    /// A default test frame (also used by the device tests in `gpu_tests`).
    pub(crate) fn input<'a>(camera: CameraData) -> FrameInput<'a> {
        FrameInput {
            clear_color: [0.0; 4],
            camera,
            light: LightData::default(),
            lights: &[],
            capture_lights: &[],
            voxel_emitters: &[],
            camera_preview: None,
            model_preview: None,
            draw_skybox: false,
            shadow_pcf_texel: 0.0,
            shadow_bias_scale: 0.0,
            shadow_normal_offset: 0.0,
            shadow_distance: 60.0,
            shadow_cascades: 0,
            shadow_cascade_lambda: 0.0,
            time: 0.0,
            water_time: 0.0,
            water: &[],
            draws: &[],
            decals: &[],
            lightmap_preview: false,
            gi_debug: 0,
            render_viewport: true,
            viewport_extent: None,
            postfx: PostFx::default(),
            preview_postfx: PostFx::default(),
            reflection_probe: None,
            reflection_probes: [None; MAX_REFLECTION_PROBES],
            reflection_probe_b: None,
            reflection_primary: 0,
            fog: None,
            voxel_specular: false,
            planar: &[],
            camera_targets: &[],
            egui: None,
            world_ui: Vec::new(),
        }
    }

    #[test]
    fn seven_point_lights_keep_shadows_when_the_camera_moves() {
        let lights: Vec<_> = (0..7).map(|i| LightInstance {
            kind: LightKind::Point, position: Vec3::new(i as f32 * 3.0, 2.0, 0.0),
            direction: Vec3::NEG_Y, color: [1.0; 3], intensity: 15.0, range: 10.0,
            spot_inner_deg: 30.0, spot_outer_deg: 45.0, cast_shadows: true,
            soft_shadows: true, shadow_bias: 0.005, radius: 0.25, baked: false,
        }).collect();
        let mut previous = None;
        for x in [-100.0, 0.0, 10.0, 100.0] {
            let cam = CameraData { position: Vec3::new(x, 2.0, 5.0), view: Mat4::IDENTITY, proj: Mat4::IDENTITY };
            let mut frame = input(cam);
            frame.lights = &lights;
            let mut packed = [GpuLight::zeroed(); MAX_LIGHTS];
            for (p, light) in packed.iter_mut().zip(&lights) { *p = gpu_light(light); }
            let (views, _, _) = plan_shadows(&frame, &mut packed, lights.len(), CascadeCfg { count: 4, lambda: 0.7, dim: 2048 });
            assert_eq!(views.len(), 42);
            assert_eq!(shadow_layer_demand(&frame, 4), 42);
            let assigned: Vec<_> = packed[..7].iter().map(|l| l.spot[1] as i32).collect();
            assert!(assigned.iter().all(|&slot| slot >= 0));
            if let Some(old) = previous { assert_eq!(assigned, old); }
            previous = Some(assigned);
        }
    }

    #[test]
    fn more_shadow_lights_stay_within_the_existing_texel_budget() {
        for requested in [256, 512, 1024, 2048, 4096] {
            for layers in SHADOW_BUDGET_LAYERS..=MAX_SHADOW_MAPS {
                let dim = shadow_atlas_dim(requested, layers) as u64;
                assert!(dim >= 64 && dim <= requested as u64);
                assert!(dim * dim * layers as u64 <= requested as u64 * requested as u64 * 12);
            }
        }
    }

    /// Off-axis perspective (an XR view): tangents of the four half-angles.
    fn off_axis(l: f32, r: f32, u: f32, d: f32, near: f32, far: f32) -> Mat4 {
        let (w, h) = (r - l, u - d);
        Mat4::from_cols_array(&[
            2.0 / w, 0.0, 0.0, 0.0,
            0.0, 2.0 / h, 0.0, 0.0,
            (r + l) / w, (u + d) / h, -far / (far - near), -1.0,
            0.0, 0.0, -(far * near) / (far - near), 0.0,
        ])
    }

    #[test]
    fn cascades_cover_an_off_axis_frustum() {
        let view = Mat4::look_at_rh(Vec3::new(1.0, 1.6, 2.0), Vec3::new(0.0, 1.0, -3.0), Vec3::Y);
        // Strongly off-axis so a centred fit would miss the frustum.
        let (l, r, u, d) = (0.2, 1.4, 1.2, 0.1);
        let camera = CameraData { view, proj: off_axis(l, r, u, d, 0.05, 500.0), position: view.inverse().w_axis.truncate() };
        let frame = input(camera);
        let dir = Vec3::new(-0.3, -1.0, -0.2).normalize();
        for cascades in [2usize, 4] {
            let (mats, splits) = cascade_matrices(&frame, dir, Vec3::Y, CascadeCfg { count: cascades, lambda: DEFAULT_CASCADE_LAMBDA, dim: 2048 });
            assert_eq!(mats.len(), cascades);
            assert!((splits[cascades - 1] - 60.0).abs() < 1e-3);
            let inv_view = view.inverse();
            for depth in [0.6f32, 3.0, 10.0, 30.0, 59.0] {
                let c = (0..cascades).find(|&i| depth <= splits[i]).unwrap();
                for (tx, ty) in [(l, u), (r, u), (l, d), (r, d)] {
                    let p = inv_view.transform_point3(Vec3::new(tx * depth, ty * depth, -depth) * 0.999);
                    let n = mats[c].project_point3(p);
                    assert!(n.x.abs() <= 1.0 && n.y.abs() <= 1.0, "cascade {c} misses {p} (ndc {n})");
                }
            }
        }
    }

    fn persp_camera(eye: Vec3, target: Vec3) -> CameraData {
        let view = Mat4::look_at_rh(eye, target, Vec3::Y);
        CameraData { view, proj: Mat4::perspective_rh(1.0, 16.0 / 9.0, 0.1, 300.0), position: eye }
    }

    fn cfg(count: usize, dim: u32) -> CascadeCfg {
        CascadeCfg { count, lambda: DEFAULT_CASCADE_LAMBDA, dim }
    }

    #[test]
    fn cascades_slide_by_whole_texels_when_the_camera_moves() {
        // Shimmer test: translating the camera (here by sub-texel, one-texel and
        // large steps, along every axis) must move each cascade's map over the
        // world by a whole number of texels, with an unchanged texel size.
        let dir = Vec3::new(-0.4, -1.0, -0.3).normalize();
        let dim = 2048u32;
        let eye = Vec3::new(3.0, 1.7, 8.0);
        let target = Vec3::new(0.0, 1.0, 0.0);
        let base = input(persp_camera(eye, target));
        let (m0, _) = cascade_matrices(&base, dir, Vec3::Y, cfg(4, dim));
        let probes = [Vec3::new(0.0, 0.0, 0.0), Vec3::new(5.0, 1.0, -3.0), Vec3::new(-7.0, 2.0, 11.0)];
        for step in [Vec3::new(0.0004, 0.0, 0.0), Vec3::new(0.0, 0.003, 0.0), Vec3::new(0.0, 0.0, -0.0123), Vec3::new(0.021, 0.0157, 0.0333), Vec3::new(1.37, -0.2, 2.9)] {
            // Same orientation, translated eye and target.
            let moved = input(persp_camera(eye + step, target + step));
            let (m1, _) = cascade_matrices(&moved, dir, Vec3::Y, cfg(4, dim));
            for (c, (a, b)) in m0.iter().zip(&m1).enumerate() {
                // The texel size (clip units per metre) must not change.
                let scale = a.row(0).truncate().length();
                assert!((scale - b.row(0).truncate().length()).abs() < 1e-5 * scale, "cascade {c} rescaled by a {step} move");
                for p in probes {
                    let (na, nb) = (a.project_point3(p), b.project_point3(p));
                    for (axis, d) in [("x", nb.x - na.x), ("y", nb.y - na.y)] {
                        let texels = d * 0.5 * dim as f32;
                        assert!((texels - texels.round()).abs() < 0.02, "cascade {c} moved {texels} texels along {axis} for {step} (not whole)");
                    }
                }
            }
        }
    }

    #[test]
    fn cascade_cfg_resolves_against_the_tier_ceiling() {
        let mut f = input(persp_camera(Vec3::new(0.0, 2.0, 5.0), Vec3::ZERO));
        let c = CascadeCfg::resolve(&f, 4, 2048);
        assert_eq!((c.count, c.lambda, c.dim), (4, DEFAULT_CASCADE_LAMBDA, 2048));
        f.shadow_cascades = 3;
        f.shadow_cascade_lambda = 0.5;
        let c = CascadeCfg::resolve(&f, 4, 1024);
        assert_eq!((c.count, c.lambda), (3, 0.5));
        // A mobile tier caps the request; garbage lambdas fall back to the default.
        assert_eq!(CascadeCfg::resolve(&f, 2, 1024).count, 2);
        f.shadow_cascade_lambda = f32::NAN;
        assert_eq!(CascadeCfg::resolve(&f, 4, 1024).lambda, DEFAULT_CASCADE_LAMBDA);
        f.shadow_cascade_lambda = 7.0;
        assert_eq!(CascadeCfg::resolve(&f, 4, 1024).lambda, 1.0);
    }

    #[test]
    fn receiver_depth_limit_tightens_the_far_cascade() {
        let camera = persp_camera(Vec3::new(0.0, 2.0, 0.0), Vec3::new(0.0, 2.0, -10.0));
        let dir = Vec3::new(-0.4, -1.0, -0.3).normalize();
        let far_of = |f: &FrameInput| cascade_matrices(f, dir, Vec3::Y, cfg(4, 2048)).1[3];
        let mut f = input(camera);
        f.shadow_distance = 150.0;
        assert!((far_of(&f) - 150.0).abs() < 1e-3, "no draws: the full shadow distance");
        // One draw whose sphere ends 28 m in front of the camera: far rounds up the 1.25x ladder (29.1 -> 36.4).
        let mut d = DrawCmd {
            gi_source: None,
            mesh: MeshHandle(0),
            material: MaterialHandle(0),
            transform: Mat4::from_translation(Vec3::new(0.0, 2.0, -27.0)),
            highlight: 0.0,
            mesh_center: Vec3::ZERO,
            bound_radius: 1.0,
            lightmap_layer: -1,
            lightmap_size: 0,
            static_geometry: false,
            planar_index: -1,
            occluded: false,
            shadow_casting: ShadowCasting::On,
            instance_color: None,
            instance_param: None,
        };
        let draws = [d];
        f.draws = &draws;
        let far = far_of(&f);
        assert!((28.0..=28.0 * 1.25 + 0.01).contains(&far), "far {far}");
        // The ladder: a small move does not change the extent.
        d.transform = Mat4::from_translation(Vec3::new(0.0, 2.0, -27.2));
        let draws2 = [d];
        f.draws = &draws2;
        assert_eq!(far_of(&f), far);
        // A draw with no bounds, or one farther than the shadow distance, keeps the distance.
        d.bound_radius = 0.0;
        let draws3 = [d];
        f.draws = &draws3;
        assert!((far_of(&f) - 150.0).abs() < 1e-3);
        d.bound_radius = 1.0;
        d.transform = Mat4::from_translation(Vec3::new(0.0, 2.0, -400.0));
        let draws4 = [d];
        f.draws = &draws4;
        assert!((far_of(&f) - 150.0).abs() < 1e-3);
    }

    #[test]
    fn cluster_shadow_cut_uses_the_shaders_texel_for_every_cascade_setup() {
        // The cluster shadow LOD target (`shadow_px` texels) is measured with
        // `shadow_cull_view`'s texels-per-metre; the shader sizes its bias and
        // normal offset with a finite-difference texel. They are the same
        // length for any cascade count / split blend / map size, so a change
        // to the cascade fit cannot silently detune the cluster cut.
        let camera = persp_camera(Vec3::new(2.0, 1.7, 9.0), Vec3::new(0.0, 1.0, 0.0));
        let dir = Vec3::new(-0.5, -1.0, -0.2).normalize();
        for dim in [1024u32, 2048, 4096] {
            for count in 1..=4usize {
                for lambda in [0.2f32, DEFAULT_CASCADE_LAMBDA, 1.0] {
                    let (mats, _) = cascade_matrices(&input(camera), dir, Vec3::Y, CascadeCfg { count, lambda, dim });
                    for (c, m) in mats.iter().enumerate() {
                        let cut = cluster::shadow_cull_view(*m, dim, cluster::DEFAULT_SHADOW_PX);
                        let cut_texel_m = 1.0 / cut.lod[1];
                        let shader_texel_m = shadow_texel_metres(*m, Vec3::new(0.5, 0.0, 0.5), -dir, dim);
                        assert!(
                            (cut_texel_m - shader_texel_m).abs() < 1e-3 * shader_texel_m,
                            "dim {dim}, {count} cascades, lambda {lambda}, cascade {c}: cut texel {cut_texel_m} vs shader {shader_texel_m}"
                        );
                        // The error target in metres is `shadow_px` of those texels.
                        assert_eq!(cut.lod[3], cluster::DEFAULT_SHADOW_PX);
                    }
                }
            }
        }
        // A perspective spot / cube face: texels per metre at distance 1, divided by the distance.
        let light = Vec3::new(-2.0, 7.0, 4.0);
        let vp = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 50.0) * Mat4::look_at_rh(light, Vec3::new(1.0, 0.0, 1.0), Vec3::Y);
        let cut = cluster::shadow_cull_view(vp, 2048, cluster::DEFAULT_SHADOW_PX);
        let p = Vec3::new(1.0, 0.0, 1.0);
        let dist = (p - light).length();
        let shader = shadow_texel_metres(vp, p, (light - p).normalize(), 2048);
        let cut_m = dist / cut.lod[1];
        assert!((cut_m - shader).abs() < 0.02 * shader, "spot: cut {cut_m} vs shader {shader}");
    }
}

#[cfg(test)]
mod camera_target_tests {
    use super::Renderer;
    use std::collections::HashMap;

    #[test]
    fn camera_target_render_order_respects_dependencies() {
        let ids = vec![10u128, 20, 30];
        let mut deps: HashMap<u128, Vec<u128>> = HashMap::new();
        deps.insert(10, vec![20]);
        deps.insert(20, vec![30]);
        let out = Renderer::camera_target_render_order(&ids, &deps);
        assert_eq!(out, vec![30, 20, 10]);
    }

    #[test]
    fn camera_target_render_order_keeps_stable_order_on_cycle() {
        let ids = vec![1u128, 2, 3];
        let mut deps: HashMap<u128, Vec<u128>> = HashMap::new();
        deps.insert(1, vec![2]);
        deps.insert(2, vec![1]); // cycle
        let out = Renderer::camera_target_render_order(&ids, &deps);
        assert_eq!(out, vec![3, 1, 2]);
    }
}

/// Shadow depth bias scale. `standard.frag` biases the receiver by a world
/// distance toward the light, sized in shadow texels; these mirror its math to
/// pin down why the former clip-depth bias ate the contact end of thin casters
/// (an avatar far from a lamp lost its legs, then torso, from the ground shadow).
#[cfg(test)]
mod shadow_bias_tests {
    use super::*;

    /// World metres the old clip-depth bias (`bias + 0.0008`, times 1 + slope)
    /// moved a receiver at `p`: the depth gap between `p` and the point on the
    /// ray to the light whose clip depth is that much smaller.
    fn clip_bias_metres(vp: Mat4, light: Vec3, p: Vec3, slope: f32, bias: f32) -> f32 {
        let z = |q: Vec3| {
            let c = vp * q.extend(1.0);
            c.z / c.w
        };
        let want = z(p) - (bias + 0.0008) * (1.0 + slope);
        let to_light = (light - p).normalize();
        let (mut lo, mut hi) = (0.0f32, (light - p).length() * 0.99);
        for _ in 0..40 {
            let mid = (lo + hi) * 0.5;
            if z(p + to_light * mid) > want { lo = mid } else { hi = mid }
        }
        lo
    }

    /// The new bias in metres: texels of `spacing` uv at the receiver.
    fn texel_bias_metres(vp: Mat4, p: Vec3, to_light: Vec3, slope: f32, bias: f32, spacing: f32) -> f32 {
        let axis = if to_light.y.abs() < 0.99 { Vec3::Y } else { Vec3::X };
        let side = to_light.cross(axis).normalize();
        let a = vp * p.extend(1.0);
        let b = vp * (p + side * 0.05).extend(1.0);
        let uv_per_m = ((b.truncate().truncate() / b.w) - (a.truncate().truncate() / a.w)).length() * 0.5 / 0.05;
        spacing.max(0.0005) / uv_per_m * (bias + 0.0008) * shadow_debug::DEFAULT_BIAS_SCALE * (1.0 + slope)
    }

    #[test]
    fn lamp_clip_bias_swallowed_the_avatar_and_texel_bias_does_not() {
        // A lamp 5 m up looking straight down, range 15, the avatar stood
        // under it and then further out; the ground below its feet is the receiver.
        let light = Vec3::new(0.0, 5.0, 0.0);
        let fov = 90f32.to_radians();
        let vp = Mat4::perspective_rh(fov, 1.0, 0.1, 15.0) * Mat4::look_to_rh(light, Vec3::NEG_Y, Vec3::Z);
        let bias = 0.003;
        let mut prev = 0.0;
        for out in [0.0f32, 2.0, 4.0] {
            let ground = Vec3::new(out, 0.0, 0.0);
            let to_light = (light - ground).normalize();
            let slope = {
                let n = to_light.y.clamp(0.0, 1.0);
                (1.0 - n * n).max(0.0).sqrt() / n.max(0.2)
            };
            let old = clip_bias_metres(vp, light, ground, slope, bias);
            let new = texel_bias_metres(vp, ground, to_light, slope, bias, 1.0 / 2048.0);
            assert!(old > prev, "clip bias should grow with distance ({old} <= {prev})");
            prev = old;
            assert!(old > 0.4, "old bias {old} m at {out} m out should exceed leg height contact");
            assert!(new < 0.25 && new < old * 0.5, "texel bias {new} m too large (old {old})");
        }
    }

    /// Bias in shadow texels, as `shadow_factor` computes it (before the
    /// texel -> metre conversion): `(bias + 0.0008) * scale * (1 + slope)`,
    /// with the shader's `slope = tan(theta)` clamped at NdotL 0.2.
    fn bias_texels(bias: f32, ndotl: f32, scale: f32) -> f32 {
        let slope = (1.0 - ndotl * ndotl).max(0.0).sqrt() / ndotl.max(0.2);
        (bias + 0.0008) * scale * (1.0 + slope)
    }

    /// Texels of receiver depth change across the PCF kernel (radius 2) on a
    /// plane at `ndotl`: what the bias has to cover to avoid acne.
    fn pcf_kernel_depth_texels(ndotl: f32) -> f32 {
        2.0 * (1.0 - ndotl * ndotl).max(0.0).sqrt() / ndotl.max(1e-3)
    }

    #[test]
    fn default_light_bias_is_acne_free_down_to_the_slope_clamp() {
        // Per-light bias 0.003 (the editor default) at the default scale.
        for step in 0..=16 {
            let ndotl = 0.2 + 0.05 * step as f32;
            let ndotl = ndotl.min(1.0);
            let have = bias_texels(0.003, ndotl, shadow_debug::DEFAULT_BIAS_SCALE);
            assert!(have >= pcf_kernel_depth_texels(ndotl), "acne risk at NdotL {ndotl}: {have} texels");
        }
    }

    #[test]
    fn zero_light_bias_floor_covers_moderate_angles_only() {
        // With the light's bias at 0 the 0.0008 floor alone holds until ~57 deg;
        // steeper receivers need the per-light bias (documented in Features.md).
        for ndotl in [1.0f32, 0.9, 0.8, 0.7, 0.6] {
            assert!(bias_texels(0.0, ndotl, shadow_debug::DEFAULT_BIAS_SCALE) >= pcf_kernel_depth_texels(ndotl), "NdotL {ndotl}");
        }
        assert!(bias_texels(0.0, 0.25, shadow_debug::DEFAULT_BIAS_SCALE) < pcf_kernel_depth_texels(0.25));
    }

    #[test]
    fn bias_grows_with_slope_and_scales_linearly() {
        let mut prev = 0.0;
        for ndotl in [1.0f32, 0.8, 0.6, 0.4, 0.2] {
            let b = bias_texels(0.003, ndotl, 1500.0);
            assert!(b > prev, "bias must grow toward grazing");
            prev = b;
        }
        // Below the clamp the bias stops growing (no runaway at 90 deg).
        assert!(bias_texels(0.003, 0.01, 1500.0) < bias_texels(0.003, 0.2, 1500.0) * 1.1);
        assert!((bias_texels(0.003, 0.5, 3000.0) / bias_texels(0.003, 0.5, 1500.0) - 2.0).abs() < 1e-4);
    }

    #[test]
    fn shader_default_matches_the_rust_constant() {
        let frag = include_str!("../shaders/standard.frag");
        assert!(frag.contains(&format!("frame.water.w : {:.1};", shadow_debug::DEFAULT_BIAS_SCALE)));
    }

    #[test]
    fn shader_normal_offset_default_matches_the_rust_constant() {
        let frag = include_str!("../shaders/standard.frag");
        assert!(frag.contains(&format!("frame.refl_blend2.z : {:.1};", shadow_debug::DEFAULT_NORMAL_OFFSET)));
        assert_eq!(shadow_debug::effective_normal_offset(0.0), shadow_debug::DEFAULT_NORMAL_OFFSET);
        assert_eq!(shadow_debug::effective_normal_offset(f32::NAN), shadow_debug::DEFAULT_NORMAL_OFFSET);
        assert_eq!(shadow_debug::effective_normal_offset(3.5), 3.5);
    }

    #[test]
    fn normal_offset_depth_margin_covers_the_pcf_kernel_at_every_angle() {
        // Lifting a receiver by `k` texels x sin(theta) along its normal clears
        // the shadow-map surface under it by k x tan(theta) texels of light
        // depth; the 5x5 PCF kernel (radius 2) spans 2 x tan(theta). So the
        // default offset alone covers the kernel from face-on to grazing,
        // where the clamped slope bias (NdotL 0.2) runs out.
        for step in 0..=89 {
            let theta = (step as f32).to_radians();
            let margin = shadow_debug::DEFAULT_NORMAL_OFFSET * theta.sin() / theta.cos().max(1e-3);
            let kernel = 2.0 * theta.sin() / theta.cos().max(1e-3);
            assert!(margin >= kernel * 0.999, "theta {step}: margin {margin} < kernel {kernel}");
        }
    }

    #[test]
    fn far_cascade_texel_bias_is_bounded() {
        // Ortho cascade of radius 30 (shadow distance 60), sun 45 deg up.
        let r = 30.0;
        let dir = Vec3::new(1.0, -1.0, 0.0).normalize();
        let vp = Mat4::orthographic_rh(-r, r, -r, r, 0.01, r * 4.0) * Mat4::look_to_rh(-dir * r * 2.0, dir, Vec3::Y);
        let slope = 1.0;
        let m = texel_bias_metres(vp, Vec3::ZERO, -dir, slope, 0.003, 1.0 / 2048.0);
        assert!(m < 0.45, "far cascade bias {m} m");
        // The old clip bias there is more than twice that.
        let old = (0.003 + 0.0008) * (1.0 + slope) * (r * 4.0);
        assert!(old > 1.5 * m);
    }
}

#[cfg(test)]
mod ts_tests {
    use super::*;

    fn r(value: u64, available: u64) -> TsResult {
        TsResult { value, available }
    }

    #[test]
    fn zone_ms_converts_ticks_with_the_period() {
        // 2 000 000 ticks at 1 ns/tick = 2 ms; at 52 ns/tick (Adreno-ish) 104 ms.
        assert!((ts_zone_ms(r(100, 1), r(2_000_100, 1), 1.0, u64::MAX).unwrap() - 2.0).abs() < 1e-4);
        assert!((ts_zone_ms(r(0, 1), r(2_000_000, 1), 52.0, u64::MAX).unwrap() - 104.0).abs() < 1e-3);
    }

    #[test]
    fn zone_ms_needs_both_queries_written() {
        assert_eq!(ts_zone_ms(r(1, 0), r(9, 1), 1.0, u64::MAX), None);
        assert_eq!(ts_zone_ms(r(1, 1), r(9, 0), 1.0, u64::MAX), None);
    }

    #[test]
    fn zone_ms_handles_counter_wrap_at_valid_bits() {
        let mask = ts_valid_mask(36);
        assert_eq!(mask, (1u64 << 36) - 1);
        // Begin just below the 36-bit wrap, end just after (upper bits garbage).
        let begin = r(mask - 999_999, 1);
        let end = r(0xABC0_0000_0000_0000 | 1_000_000, 1);
        let ms = ts_zone_ms(begin, end, 1.0, mask).unwrap();
        assert!((ms - 2.0).abs() < 1e-4, "{ms}");
        assert_eq!(ts_valid_mask(64), u64::MAX);
    }

    #[test]
    fn readback_is_two_queries_unless_zones_are_on() {
        assert_eq!(ts_readback_count(false), 2);
        assert_eq!(ts_readback_count(true), frame::TS_COUNT);
    }
}

/// `ZESTENGINE_SKY_DEBUG=1` (read once): see `Renderer::record_skybox`.
fn sky_debug() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ZESTENGINE_SKY_DEBUG").is_some_and(|v| v != "0"))
}


/// Resample a cube's mip-0 faces (RGBA16F bytes, the layout the GPU samples with the Vulkan
/// cube-map face table) into a linear RGBA16F equirect matching `skybox.frag`'s lookup
/// (`u = atan(z, x) / 2pi + 0.5`, `v = acos(y) / pi`). Bilinear within a face.
pub(crate) fn cube_chain_to_equirect(cube: &HdrCube) -> TextureData {
    let size = cube.size.max(1) as usize;
    let w = (size * 4).clamp(64, 4096);
    let h = w / 2;
    let texel = |face: usize, x: usize, y: usize| -> [f32; 4] {
        let o = (y * size + x) * 8;
        let b = &cube.faces[face][0];
        std::array::from_fn(|c| half::f16::from_le_bytes([b[o + c * 2], b[o + c * 2 + 1]]).to_f32())
    };
    let sample = |d: Vec3| -> [f32; 4] {
        let a = d.abs();
        // Vulkan cube-map face selection (sc, tc, major axis).
        let (face, sc, tc, ma) = if a.x >= a.y && a.x >= a.z {
            if d.x > 0.0 { (0, -d.z, -d.y, a.x) } else { (1, d.z, -d.y, a.x) }
        } else if a.y >= a.z {
            if d.y > 0.0 { (2, d.x, d.z, a.y) } else { (3, d.x, -d.z, a.y) }
        } else if d.z > 0.0 {
            (4, d.x, -d.y, a.z)
        } else {
            (5, -d.x, -d.y, a.z)
        };
        let u = ((sc / ma + 1.0) * 0.5 * size as f32 - 0.5).clamp(0.0, size as f32 - 1.0);
        let v = ((tc / ma + 1.0) * 0.5 * size as f32 - 0.5).clamp(0.0, size as f32 - 1.0);
        let (x0, y0) = (u.floor() as usize, v.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(size - 1), (y0 + 1).min(size - 1));
        let (fx, fy) = (u - x0 as f32, v - y0 as f32);
        let (p00, p10, p01, p11) = (texel(face, x0, y0), texel(face, x1, y0), texel(face, x0, y1), texel(face, x1, y1));
        std::array::from_fn(|c| (p00[c] * (1.0 - fx) + p10[c] * fx) * (1.0 - fy) + (p01[c] * (1.0 - fx) + p11[c] * fx) * fy)
    };
    let mut pixels = Vec::with_capacity(w * h * 8);
    for j in 0..h {
        let theta = (j as f32 + 0.5) / h as f32 * std::f32::consts::PI;
        for i in 0..w {
            let phi = ((i as f32 + 0.5) / w as f32 - 0.5) * std::f32::consts::TAU;
            let d = Vec3::new(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin());
            let c = sample(d);
            for (k, v) in c.iter().enumerate() {
                let v = if k == 3 { 1.0 } else { *v };
                pixels.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
        }
    }
    TextureData { width: w as u32, height: h as u32, pixels, srgb: false, hdr: true }
}
