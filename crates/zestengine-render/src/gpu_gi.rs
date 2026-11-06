//! GPU software-GI probe march. Runs `sw_gi.comp` over the merged Global
//! Distance Field (a 3D distance texture plus a nearest-instance index texture)
//! to produce SH-L1 radiance + directional-distance probes, the same packed
//! layout the standard shader samples.
//!
//! The GDF (distance + index textures + per-instance materials) is cached and
//! only rebuilt/re-uploaded via `set_gdf` when geometry moves; `march` then runs
//! every trace against the cached GDF, creating only the small per-trace buffers
//! (lights/probes/output). That lets a static scene keep a high-resolution GDF
//! for free (sharp, un-faceted occlusion) instead of re-uploading it each frame.

use crate::hitch::WaitNamed as _;
use std::io::Cursor;
use std::mem::size_of;
use std::sync::{Arc, Mutex};

use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;

use crate::alloc::Buffer;
use crate::context::GpuContext;
use crate::texture::GpuTexture;
use crate::types::{BakeLight, LightKind, ProbeSh};

const SW_GI_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/sw_gi.comp.spv"));
const SCREEN_GI_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/screen_gi.comp.spv"));
// flux_probe.glsl: 5 metadata vec4s, 16 directional radiance/state records,
// and 4 filtered SH vec4s. Keep both backends and the resolve on this layout.
pub(crate) const SCREEN_PROBE_BYTES: u64 = (5 + 16 + 4 + 3) * 16;

const FLUX_DENOISE_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/flux_denoise.comp.spv"));
const FLUX_INTEGRATE_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/flux_integrate.comp.spv"));
const FLUX_INTEGRATE_RT_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/flux_integrate.rt.comp.spv"));
const SCREEN_GI_RT_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/screen_gi_rt.comp.spv"));

/// One SH probe volume's layout (std140 mirror of `GiProbeVolume` in the shader
/// / `GpuProbeVolume` on the renderer): the world SH radiance cache the gather
/// reads at ray hits.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GiVolume {
    pub world_to_local: [[f32; 4]; 4],
    pub size_base: [f32; 4], // xyz local box size, w = first probe index (sh_base)
    pub counts: [f32; 4],    // xyz probe counts per axis
}

pub const GI_MAX_VOLUMES: usize = 4;

/// Screen-GI compute params (std140 UBO mirror of `screen_gi.comp`'s Params).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ScreenGiParams {
    pub inv_view_proj: [[f32; 4]; 4],
    pub prev_view_proj: [[f32; 4]; 4], // reproject world pos → previous-frame UV
    pub cam: [f32; 4],
    pub prev_cam: [f32; 4],            // previous frame camera pos (disocclusion)
    pub gdf_min: [f32; 4],
    pub gdf_max: [f32; 4],
    pub sky: [f32; 4],
    pub counts: [u32; 4], // samples, bounces, light_count, emitter_count
    pub march: [f32; 4],  // eps, max_dist, temporal_alpha, _
    pub misc: [u32; 4],   // seed, screen_w, screen_h, flip_y
    pub probe_info: [f32; 4], // x = probe volume count
    pub volumes: [GiVolume; GI_MAX_VOLUMES],
    pub temporal: [f32; 4], // probe alpha, final max frames, final minimum alpha, hardware ray lighting mode
    pub(crate) screen: crate::flux_screen_history::Params,
    pub transport: [f32;4], // x material emissive transport strength
}

/// Push constants for flux_denoise.comp (probe-space à-trous denoise).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DenoisePush {
    size: [i32; 2], // probe-grid resolution
    step: i32,      // base à-trous tap spacing (probe texels)
    _pad: i32,
}

/// UBO for screen_gi_rt.comp (scalar layout, reordered from ScreenGiParams).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RtGiParams {
    inv_view_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    cam: [f32; 4],
    prev_cam: [f32; 4],
    sky: [f32; 4],
    counts: [u32; 4],
    march: [f32; 4],
    misc: [u32; 4],
    // Optional world-probe irradiance fallback, shared with the software path.
    gdf_min: [f32; 4],
    gdf_max: [f32; 4],
    probe_info: [f32; 4], // x = probe volume count
    volumes: [GiVolume; GI_MAX_VOLUMES],
    temporal: [f32; 4],
    screen: crate::flux_screen_history::Params,
    transport: [f32;4],
}

impl RtGiParams {
    fn from_screen(p: &ScreenGiParams) -> Self {
        Self {
            inv_view_proj: p.inv_view_proj,
            prev_view_proj: p.prev_view_proj,
            cam: p.cam,
            prev_cam: p.prev_cam,
            sky: p.sky,
            counts: p.counts,
            march: p.march,
            misc: p.misc,
            gdf_min: p.gdf_min,
            gdf_max: p.gdf_max,
            probe_info: p.probe_info,
            volumes: p.volumes,
            temporal: p.temporal,
            screen: p.screen,
            transport:p.transport,
        }
    }
}

/// Push constants for flux_integrate.comp (full-res per-pixel SH resolve).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct IntegratePush {
    inv_view_proj: [[f32; 4]; 4],
    cam: [f32; 4],
    dims: [i32; 4], // screen_w, screen_h, probe_w, probe_h
    misc: [f32; 4], // flip_y, indirect intensity, emitter count, pad
}

/// Per-frame Flux trace resources that must outlive the command-buffer recording
/// (the descriptor pool + the host SSBOs/UBO the dispatch reads). When the trace
/// is folded into the main frame command buffer there is no fence wait to bound
/// their lifetime, so the caller holds these and frees them one frame later,
/// after the frame fence that consumed them has signalled.
pub struct ScreenGiTransient {
    desc_pool: vk::DescriptorPool,
    light_buf: Buffer,
    emitter_buf: Buffer,
    /// Element capacities of the light / emitter buffers (reused while they fit).
    light_cap: usize,
    emitter_cap: usize,
    param_buf: Buffer,
    /// RtGiParams UBO for the hardware gather (only when the RT backend ran).
    rt_param_buf: Option<Buffer>,
    mesh_snapshot: Option<Arc<crate::mesh_sdf::Snapshot>>,
}

impl ScreenGiTransient {
    pub fn destroy(&mut self, device: &ash::Device, alloc: &mut Allocator) {
        unsafe { device.destroy_descriptor_pool(self.desc_pool, None) };
        self.light_buf.destroy(device, alloc);
        self.emitter_buf.destroy(device, alloc);
        self.param_buf.destroy(device, alloc);
        self.mesh_snapshot = None;
        if let Some(mut b) = self.rt_param_buf.take() {
            b.destroy(device, alloc);
        }
    }
}

/// Per-instance material the march shades GDF hits with (looked up by the index
/// texture). Albedo drives the diffuse bounce; emission makes the surface a
/// light source.
#[derive(Clone, Copy)]
pub struct GpuGiMaterial {
    pub albedo: [f32; 3],
    pub emission: [f32; 3],
    /// Metalness [0,1]; the diffuse bounce uses albedo·(1−metallic).
    pub metallic: f32,
    /// Roughness [0,1]; reserved for the specular GI trace.
    pub roughness: f32,
}

/// An emissive instance reduced to a sphere area-light, sampled directly (NEE) by
/// the march so emitter fill is smooth instead of blotchy Monte-Carlo noise.
#[derive(Clone, Copy, PartialEq)]
pub struct GpuGiEmitter {
    pub center: [f32; 3],
    pub radius: f32,
    pub emission: [f32; 3],
}

/// Per-trace march inputs (lights/probes/params that change every trace; the GDF
/// is cached separately via `set_gdf`).
pub struct GpuGiMarch<'a> {
    pub lights: &'a [BakeLight],
    pub emitters: &'a [GpuGiEmitter],
    pub probes: &'a [Vec3],
    pub samples: u32,
    pub bounces: u32,
    pub sky: [f32; 3],
    pub eps: f32,
    pub max_dist: f32,
    pub seed: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GiInst {
    albedo: [f32; 4],
    emission: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GiLight {
    pos_kind: [f32; 4],
    dir_range: [f32; 4],
    color: [f32; 4],
    spot: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GiEmitter {
    center_radius: [f32; 4],
    emission: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GiPush {
    gdf_min: [f32; 4],
    gdf_max: [f32; 4],
    sky: [f32; 4],
    counts: [u32; 4], // samples, bounces, light_count, probe_count
    march: [f32; 4],  // eps, max_dist, _, _
    misc: [u32; 4],   // seed, _, _, _
}

fn pack_light(l: &BakeLight) -> GiLight {
    let kind = match l.kind {
        LightKind::Directional => 0.0,
        LightKind::Point => 1.0,
        LightKind::Spot => 2.0,
    };
    GiLight {
        pos_kind: [l.position.x, l.position.y, l.position.z, kind],
        dir_range: [l.direction.x, l.direction.y, l.direction.z, l.range],
        color: [l.color[0], l.color[1], l.color[2], 0.0],
        spot: [l.spot_inner_deg, l.spot_outer_deg, l.radius, 0.0],
    }
}

/// Cached GPU-side GDF: the two 3D textures + per-instance materials + bounds.
/// Persists across traces; replaced only when geometry changes.
struct CachedGdf {
    sources: Vec<Option<crate::GiSource>>,
    dist_tex: GpuTexture,
    index_tex: GpuTexture,
    inst_buf: Buffer,
    min: [f32; 3],
    max: [f32; 3],
}

impl CachedGdf {
    fn destroy(&mut self, device: &ash::Device, alloc: &mut Allocator) {
        self.dist_tex.destroy(device, alloc);
        self.index_tex.destroy(device, alloc);
        self.inst_buf.destroy(device, alloc);
    }
}

/// An async march submitted to the GPU but not yet read back. Its buffers must
/// stay alive until the fence signals; polled each frame so the trace never
/// blocks the main thread (GI work decoupled from the frame).
struct InFlight {
    fence: vk::Fence,
    cb: vk::CommandBuffer,
    desc_pool: vk::DescriptorPool,
    light_buf: Buffer,
    probe_buf: Buffer,
    emitter_buf: Buffer,
    out_buf: Buffer,
    count: usize,
}

#[derive(Clone,Copy)]
pub(crate) struct TemporalImages {
    pub current: [vk::Image;2], pub current_views: [vk::ImageView;2],
    pub previous: [vk::Image;2], pub previous_views: [vk::ImageView;2],
}

/// Persistent compute pipeline + samplers + cached GDF for the GI march.
pub struct GpuGi {
    #[cfg(test)]
    pub(crate) probe_history_alpha: Option<f32>,
    #[cfg(test)]
    pub(crate) temporal_override: Option<[f32; 4]>,
    #[cfg(test)]
    pub(crate) screen_trace_override: Option<bool>,
    #[cfg(test)]
    pub(crate) importance_sampling: Option<bool>,
    #[cfg(test)]
    pub(crate) balanced_gather: Option<bool>,
    #[cfg(test)]
    pub(crate) gather_seed: Option<u32>,
    #[cfg(test)]
    pub(crate) area_gather_variant: Option<bool>,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    dist_sampler: vk::Sampler,  // trilinear, clamp
    index_sampler: vk::Sampler, // nearest, clamp
    gdf: Option<CachedGdf>,
    pub(crate) mesh_sdfs: crate::mesh_sdf::Cache,
    pub(crate) gdf_revision: u64,
    /// Dedicated pool for async march command buffers (isolated from the
    /// renderer's per-frame pool).
    cmd_pool: vk::CommandPool,
    in_flight: Option<InFlight>,
    /// Screen-space GI final-gather compute (per-pixel GDF trace).
    screen_set_layout: vk::DescriptorSetLayout,
    screen_pipeline_layout: vk::PipelineLayout,
    screen_pipeline: [vk::Pipeline;6],
    screen_detail_pipeline: [vk::Pipeline;6],
    screen_receiver_pipeline: vk::Pipeline,
    screen_placement_pipeline: vk::Pipeline,
    screen_reduce_pipeline: vk::Pipeline,
    surface_pipeline: vk::Pipeline,
    surface_rt_pipeline: Option<vk::Pipeline>,
    radiance_pipeline: vk::Pipeline,
    radiance_rt_pipeline: Option<vk::Pipeline>,
    pub(crate) radiance_cache: crate::flux_radiance_cache::Cache,
    full_temporal: (vk::DescriptorSetLayout,vk::PipelineLayout,vk::Pipeline),
    /// Probe-space GI denoise compute (à-trous filter on the gather, at probe res).
    denoise_set_layout: vk::DescriptorSetLayout,
    denoise_pipeline_layout: vk::PipelineLayout,
    denoise_pipeline: vk::Pipeline,
    /// Full-res screen-probe integrate compute (per-pixel-normal SH resolve).
    integrate_set_layout: vk::DescriptorSetLayout,
    integrate_pipeline_layout: vk::PipelineLayout,
    integrate_pipeline: vk::Pipeline,
    /// Hardware emitter resolve: descriptor layout, pipeline layout, pipeline.
    pub(crate) rt_textures: Option<crate::rt_reflect::MaterialTextures>,
    rt_integrate: Option<(vk::DescriptorSetLayout, vk::PipelineLayout, vk::Pipeline)>,
    /// HARDWARE ray-query screen-probe gather (RT-core backend; ray-query devices).
    /// `None` when the device lacks ray-query — the SDF gather is then the only path.
    rt_set_layout: Option<vk::DescriptorSetLayout>,
    rt_pipeline_layout: Option<vk::PipelineLayout>,
    rt_pipeline: Option<[vk::Pipeline;6]>,
    /// Timestamp pool for per-pass GPU zones (gather / denoise / integrate)
    /// while profiling; set by the renderer around `screen_resolve_record`.
    pub timestamps: std::cell::Cell<Option<(vk::QueryPool, bool)>>,
}

impl GpuGi {
    fn ts(&self, device: &ash::Device, cb: vk::CommandBuffer, slot: u32) {
        if let Some((pool,camera)) = self.timestamps.get() {
            let zone = slot / 2;
            let zone = if camera { match zone as usize {
                crate::gpu_zone::GI_GATHER => crate::gpu_zone::CAMERA_GI_GATHER as u32,
                crate::gpu_zone::GI_DENOISE => crate::gpu_zone::CAMERA_GI_DENOISE as u32,
                crate::gpu_zone::GI_INTEGRATE => crate::gpu_zone::CAMERA_GI_INTEGRATE as u32,
                crate::gpu_zone::GI_SURFACE => crate::gpu_zone::CAMERA_GI_SURFACE as u32,
                _ => zone,
            }} else {zone};
            let slot=zone*2+slot%2;
            unsafe { device.cmd_write_timestamp2(cb, vk::PipelineStageFlags2::ALL_COMMANDS, pool, slot) };
        }
    }

    pub fn new(ctx: &GpuContext) -> Result<Self> {
        let device = &ctx.device;
        let stage = vk::ShaderStageFlags::COMPUTE;
        let sampled = |b: u32| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(stage)
        };
        let ssbo = |b: u32| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(stage)
        };
        let bindings = [sampled(0), sampled(1), ssbo(2), ssbo(3), ssbo(4), ssbo(5), ssbo(6)];
        let set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?
        };
        let set_layouts = [set_layout];
        let push = [vk::PushConstantRange::default()
            .stage_flags(stage)
            .size(size_of::<GiPush>() as u32)];
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push),
                None,
            )?
        };
        let code = ash::util::read_spv(&mut Cursor::new(SW_GI_COMP))?;
        let module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?
        };
        let pipeline = unsafe {
            device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(stage)
                                .module(module)
                                .name(c"main"),
                        )
                        .layout(pipeline_layout)],
                    None,
                )
                .map_err(|(_, e)| e)?[0]
        };
        unsafe { device.destroy_shader_module(module, None) };

        let mk_sampler = |filter: vk::Filter| -> Result<vk::Sampler> {
            let info = vk::SamplerCreateInfo::default()
                .mag_filter(filter)
                .min_filter(filter)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
            Ok(unsafe { device.create_sampler(&info, None)? })
        };
        let dist_sampler = mk_sampler(vk::Filter::LINEAR)?;
        let index_sampler = mk_sampler(vk::Filter::NEAREST)?;

        let cmd_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(ctx.queue_family),
                None,
            )?
        };

        // Screen-space GI pipeline: GDF (0,1) + insts(2) + lights(3) + emitters(4)
        // + depth sampler(5) + output storage image(6) + params UBO(7).
        let storage_img = |b: u32| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(stage)
        };
        let ubo = |b: u32| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(stage)
        };
        let screen_bindings = [
            sampled(0),
            sampled(1),
            ssbo(2),
            ssbo(3),
            ssbo(4),
            sampled(5),
            storage_img(6),
            ubo(7),
            sampled(8), // history (previous frame's result, for temporal accum)
            ssbo(9),    // world SH probe radiance cache (read at ray hits)
            ssbo(10),   // directional history + filtered SH (current)
            ssbo(11),   // directional history (previous)
            ssbo(12), ssbo(13), sampled(14), sampled(15), sampled(16), sampled(17), // surface material cards
            sampled(18), sampled(19), storage_img(20), storage_img(21), // cached surface radiance
            ssbo(22), ssbo(23), sampled(24), // mesh distance-field hierarchy
            ssbo(25), // current adaptive trace samples
            ssbo(26), sampled(27), // current depth hierarchy and previous physical HDR/depth
            ssbo(28),ssbo(29),ubo(30), // automatic directional radiance cache
            sampled(31), // shared area-light emission atlas
        ];
        let screen_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&screen_bindings),
                None,
            )?
        };
        let screen_layouts = [screen_set_layout];
        let screen_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&screen_layouts),
                None,
            )?
        };
        let screen_pipeline=make_gather_pipelines(device,screen_pipeline_layout,SCREEN_GI_COMP)?;

        // Probe-space denoise: input gather (sampled), output filtered (storage).
        let denoise_bindings = [sampled(0), storage_img(1), ssbo(2)];
        let denoise_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&denoise_bindings),
                None,
            )?
        };
        let denoise_layouts = [denoise_set_layout];
        let denoise_pc = [vk::PushConstantRange::default()
            .stage_flags(stage)
            .offset(0)
            .size(size_of::<DenoisePush>() as u32)];
        let denoise_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&denoise_layouts)
                    .push_constant_ranges(&denoise_pc),
                None,
            )?
        };
        let denoise_code = ash::util::read_spv(&mut Cursor::new(FLUX_DENOISE_COMP))?;
        let denoise_module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&denoise_code),
                None,
            )?
        };
        let denoise_pipeline = unsafe {
            device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(stage)
                                .module(denoise_module)
                                .name(c"main"),
                        )
                        .layout(denoise_pipeline_layout)],
                    None,
                )
                .map_err(|(_, e)| e)?[0]
        };
        unsafe { device.destroy_shader_module(denoise_module, None) };

        // Full-res integrate: scalar GI (sampled) + depth (sampled) + screen-probe
        // SH (ssbo) → integrated GI (storage).
        // Emitters (ssbo 4): their direct light is shaded per pixel here.
        let integrate_bindings = [sampled(0), sampled(1), ssbo(2), storage_img(3), ssbo(4), sampled(5), ubo(6), ssbo(8), ssbo(9), sampled(10)];
        let integrate_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&integrate_bindings),
                None,
            )?
        };
        let integrate_layouts = [integrate_set_layout];
        let integrate_pc = [vk::PushConstantRange::default()
            .stage_flags(stage)
            .offset(0)
            .size(size_of::<IntegratePush>() as u32)];
        let integrate_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&integrate_layouts)
                    .push_constant_ranges(&integrate_pc),
                None,
            )?
        };
        let integrate_code = ash::util::read_spv(&mut Cursor::new(FLUX_INTEGRATE_COMP))?;
        let integrate_module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&integrate_code),
                None,
            )?
        };
        let integrate_pipeline = unsafe {
            device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(stage)
                                .module(integrate_module)
                                .name(c"main"),
                        )
                        .layout(integrate_pipeline_layout)],
                    None,
                )
                .map_err(|(_, e)| e)?[0]
        };
        unsafe { device.destroy_shader_module(integrate_module, None) };

        let rt_textures=ctx.ray_tracing().then(||crate::rt_reflect::MaterialTextures::new(device)).transpose()?;
        let rt_integrate = if ctx.ray_tracing() {
            let mut bindings = integrate_bindings.to_vec();
            bindings.push(vk::DescriptorSetLayoutBinding::default()
                .binding(7).descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1).stage_flags(stage));
            bindings.push(ssbo(11));
            let set_layout = unsafe { device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)? };
            let layout = unsafe { device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout,rt_textures.as_ref().unwrap().layout])
                    .push_constant_ranges(&integrate_pc), None)? };
            let code = ash::util::read_spv(&mut Cursor::new(FLUX_INTEGRATE_RT_COMP))?;
            let module = unsafe { device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&code), None)? };
            let result = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default().layout(layout).stage(
                    vk::PipelineShaderStageCreateInfo::default().stage(stage).module(module).name(c"main"))], None) };
            unsafe { device.destroy_shader_module(module, None) };
            Some((set_layout, layout, result.map_err(|(_, e)| e)?[0]))
        } else { None };

        // Hardware ray-query gather pipeline — only when the device exposes ray-query
        // (the shader references accelerationStructureEXT). Bindings mirror
        // screen_gi_rt.comp: depth, out, history, TLAS, instances, lights, emitters,
        // sh-out, sh-prev, params UBO.
        let (rt_set_layout, rt_pipeline_layout, rt_pipeline) = if ctx.ray_tracing() {
            let accel = |b: u32| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .descriptor_count(1)
                    .stage_flags(stage)
            };
            let rt_bindings = [
                sampled(0), storage_img(1), sampled(2), accel(3),
                ssbo(4), ssbo(5), ssbo(6), ssbo(7), ssbo(8), ubo(9),
                ssbo(10), // world SH radiance cache (probe SSBO)
                ssbo(11), ssbo(12), sampled(13), sampled(14), sampled(15), sampled(16), // surface material cards
                sampled(17), sampled(18), storage_img(19), storage_img(20), // cached surface radiance
                ssbo(21), sampled(22),
                ssbo(23),ssbo(24),ubo(25), // automatic directional radiance cache
                sampled(26), // shared area-light emission atlas
            ];
            let rt_set_layout = unsafe {
                device.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&rt_bindings),
                    None,
                )?
            };
            let rt_layouts = [rt_set_layout,rt_textures.as_ref().unwrap().layout];
            let rt_pipeline_layout = unsafe {
                device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&rt_layouts),
                    None,
                )?
            };
            let rt_pipeline=make_gather_pipelines(device,rt_pipeline_layout,SCREEN_GI_RT_COMP)?;
            (Some(rt_set_layout), Some(rt_pipeline_layout), Some(rt_pipeline))
        } else {
            (None, None, None)
        };

        let make_surface = |layout,spv: &[u8]| -> Result<vk::Pipeline> {
            let module = crate::screen_pass::shader_module(device,spv)?;
            let info = vk::ComputePipelineCreateInfo::default().layout(layout)
                .stage(vk::PipelineShaderStageCreateInfo::default().stage(stage).module(module).name(c"main"));
            let result = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(),&[info],None) };
            unsafe { device.destroy_shader_module(module,None); }
            Ok(result.map_err(|(_,e)| e)?[0])
        };
        let screen_receiver_pipeline=make_surface(screen_pipeline_layout,include_bytes!(concat!(env!("OUT_DIR"),"/screen_gi.comp.receiver.spv")))?;
        let screen_placement_pipeline=make_surface(screen_pipeline_layout,include_bytes!(concat!(env!("OUT_DIR"),"/screen_gi.comp.placement.spv")))?;
        let screen_reduce_pipeline=make_surface(screen_pipeline_layout,include_bytes!(concat!(env!("OUT_DIR"),"/screen_gi.comp.reduce.spv")))?;
        let screen_detail_pipeline=make_gather_pipelines(device,screen_pipeline_layout,include_bytes!(concat!(env!("OUT_DIR"),"/screen_gi.comp.detail.spv")))?;
        let surface_pipeline = make_surface(screen_pipeline_layout,include_bytes!(concat!(env!("OUT_DIR"),"/screen_gi.comp.surface.spv")))?;
        let surface_rt_pipeline = rt_pipeline_layout.map(|layout| make_surface(layout,
            include_bytes!(concat!(env!("OUT_DIR"),"/screen_gi_rt.comp.surface.spv")))).transpose()?;
        let radiance_pipeline=make_surface(screen_pipeline_layout,include_bytes!(concat!(env!("OUT_DIR"),"/screen_gi.comp.radiance.spv")))?;
        let radiance_rt_pipeline=rt_pipeline_layout.map(|layout|make_surface(layout,include_bytes!(concat!(env!("OUT_DIR"),"/screen_gi_rt.comp.radiance.spv")))).transpose()?;
        let radiance_cache=crate::flux_radiance_cache::Cache::new(device)?;
        let history_bindings=[sampled(0),sampled(1),sampled(2),sampled(3),storage_img(4),storage_img(5),ubo(6)];
        let history_set=unsafe {device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&history_bindings),None)?};
        let history_layout=unsafe {device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[history_set]),None)?};
        let history_pipeline=make_surface(history_layout,include_bytes!(concat!(env!("OUT_DIR"),"/flux_history.comp.spv")))?;
        let full_temporal=(history_set,history_layout,history_pipeline);
        Ok(Self {
            #[cfg(test)]
            probe_history_alpha: None,
            #[cfg(test)]
            temporal_override: None,
            #[cfg(test)]
            screen_trace_override: None,
            #[cfg(test)]
            importance_sampling: None,
            #[cfg(test)]
            balanced_gather: None,
            #[cfg(test)]
            gather_seed: None,
            #[cfg(test)]
            area_gather_variant: None,
            set_layout,
            pipeline_layout,
            pipeline,
            dist_sampler,
            index_sampler,
            gdf: None,
            mesh_sdfs: Default::default(),
            gdf_revision: 0,
            cmd_pool,
            in_flight: None,
            screen_set_layout,
            screen_pipeline_layout,
            screen_pipeline,
            screen_detail_pipeline,
            screen_receiver_pipeline,
            screen_placement_pipeline,
            screen_reduce_pipeline,
            surface_pipeline,
            surface_rt_pipeline,
            radiance_pipeline,radiance_rt_pipeline,radiance_cache,
            full_temporal,
            denoise_set_layout,
            denoise_pipeline_layout,
            denoise_pipeline,
            integrate_set_layout,
            integrate_pipeline_layout,
            integrate_pipeline,
            rt_textures,
            rt_integrate,
            rt_set_layout,
            rt_pipeline_layout,
            rt_pipeline,
            timestamps: std::cell::Cell::new(None),
        })
    }

    /// (Re)upload the GDF (distance + index 3D textures and per-instance
    /// materials), replacing any cached one. Call only when geometry changes.
    #[allow(clippy::too_many_arguments)]
    pub fn set_gdf(
        &mut self,
        ctx: &GpuContext,
        allocator: &Arc<Mutex<Allocator>>,
        command_pool: vk::CommandPool,
        dist: &[f32],
        index: &[u32],
        dims: [u32; 3],
        min: [f32; 3],
        max: [f32; 3],
        materials: &[GpuGiMaterial],
        sources: &[Option<crate::GiSource>],
    ) -> Result<()> {
        anyhow::ensure!(sources.is_empty() || sources.len()==materials.len(), "GDF source count must match materials");
        let device = &ctx.device;
        let mut alloc = allocator.lock().unwrap();
        let dist_tex =
            GpuTexture::upload_volume(device, &mut alloc, command_pool, ctx.queue, dist, dims)?;
        let index_tex =
            GpuTexture::upload_volume_u32(device, &mut alloc, command_pool, ctx.queue, index, dims)?;
        let insts: Vec<GiInst> = if materials.is_empty() {
            vec![GiInst::zeroed()]
        } else {
            materials
                .iter()
                .map(|m| GiInst {
                    // .w lanes carry the PBR scalars the diffuse bounce needs:
                    // albedo.w = metalness (shader uses albedo·(1−metalness)),
                    // emission.w = roughness (reserved for the specular trace).
                    albedo: [m.albedo[0], m.albedo[1], m.albedo[2], m.metallic],
                    emission: [m.emission[0], m.emission[1], m.emission[2], m.roughness],
                })
                .collect()
        };
        let inst_buf = host_ssbo(device, &mut alloc, bytemuck::cast_slice(&insts), "gi insts")?;
        if let Some(mut old) = self.gdf.take() {
            old.destroy(device, &mut alloc);
        }
        self.gdf_revision = self.gdf_revision.wrapping_add(1);
        self.gdf = Some(CachedGdf {
            sources: if sources.is_empty() {vec![None;materials.len().max(1)]} else {sources.to_vec()},
            dist_tex,
            index_tex,
            inst_buf,
            min,
            max,
        });
        Ok(())
    }

    pub(crate) fn gdf_sources(&self) -> &[Option<crate::GiSource>] {
        self.gdf.as_ref().map_or(&[], |g|g.sources.as_slice())
    }

    /// True while an async march is submitted but not yet read back.
    pub fn is_marching(&self) -> bool {
        self.in_flight.is_some()
    }

    /// True once a GDF has been uploaded (screen-GI can trace it).
    pub fn has_gdf(&self) -> bool {
        self.gdf.is_some()
    }

    /// Begin an async march against the cached GDF: record + submit the compute
    /// with a fence but DON'T wait. Returns false if no GDF, no probes, or a
    /// march is already in flight. Read the result later via [`march_poll`].
    pub fn march_begin(
        &mut self,
        ctx: &GpuContext,
        allocator: &Arc<Mutex<Allocator>>,
        m: &GpuGiMarch<'_>,
    ) -> Result<bool> {
        if self.in_flight.is_some() {
            return Ok(false);
        }
        let Some(gdf) = self.gdf.as_ref() else {
            return Ok(false);
        };
        let device = &ctx.device;
        let count = m.probes.len();
        if count == 0 {
            return Ok(false);
        }
        let mut alloc = allocator.lock().unwrap();

        let lights: Vec<GiLight> = if m.lights.is_empty() {
            vec![GiLight::zeroed()]
        } else {
            m.lights.iter().map(pack_light).collect()
        };
        let probes: Vec<[f32; 4]> = m.probes.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
        let emitters: Vec<GiEmitter> = if m.emitters.is_empty() {
            vec![GiEmitter::zeroed()]
        } else {
            m.emitters
                .iter()
                .map(|e| GiEmitter {
                    center_radius: [e.center[0], e.center[1], e.center[2], e.radius],
                    emission: [e.emission[0], e.emission[1], e.emission[2], 0.0],
                })
                .collect()
        };
        let light_buf =
            host_ssbo(device, &mut alloc, bytemuck::cast_slice(&lights), "gi lights")?;
        let probe_buf =
            host_ssbo(device, &mut alloc, bytemuck::cast_slice(&probes), "gi probes")?;
        let emitter_buf =
            host_ssbo(device, &mut alloc, bytemuck::cast_slice(&emitters), "gi emitters")?;
        let out_buf = Buffer::new(
            device,
            &mut alloc,
            (count * size_of::<[f32; 16]>()) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuToCpu,
            "gi out",
        )?;

        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 2,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 5,
            },
        ];
        let desc_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&pool_sizes),
                None,
            )?
        };
        let set_layouts = [self.set_layout];
        let set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(desc_pool)
                    .set_layouts(&set_layouts),
            )?[0]
        };
        self.write_image(device, set, 0, self.dist_sampler, gdf.dist_tex.view);
        self.write_image(device, set, 1, self.index_sampler, gdf.index_tex.view);
        write_ssbo(device, set, 2, gdf.inst_buf.handle);
        write_ssbo(device, set, 3, light_buf.handle);
        write_ssbo(device, set, 4, probe_buf.handle);
        write_ssbo(device, set, 5, out_buf.handle);
        write_ssbo(device, set, 6, emitter_buf.handle);

        let push = GiPush {
            gdf_min: [gdf.min[0], gdf.min[1], gdf.min[2], 0.0],
            gdf_max: [gdf.max[0], gdf.max[1], gdf.max[2], 0.0],
            sky: [m.sky[0], m.sky[1], m.sky[2], 0.0],
            counts: [
                m.samples.max(1),
                m.bounces.max(1),
                m.lights.len() as u32,
                count as u32,
            ],
            march: [m.eps, m.max_dist, 0.0, 0.0],
            misc: [m.seed, m.emitters.len() as u32, 0, 0],
        };
        let groups = (count as u32).div_ceil(64);
        // Record + submit WITHOUT waiting; the fence is polled next frame.
        let cb = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0]
        };
        unsafe {
            device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[set],
                &[],
            );
            device.cmd_push_constants(
                cb,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );
            device.cmd_dispatch(cb, groups, 1, 1);
            device.end_command_buffer(cb)?;
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cbs = [cb];
            let submit = vk::SubmitInfo::default().command_buffers(&cbs);
            device.queue_submit(ctx.queue, &[submit], fence)?;
            self.in_flight = Some(InFlight {
                fence,
                cb,
                desc_pool,
                light_buf,
                probe_buf,
                emitter_buf,
                out_buf,
                count,
            });
        }
        Ok(true)
    }

    /// Poll the in-flight march; when the GPU has finished, read back the probe
    /// SH, free its resources, and return them. `None` while still running or
    /// when nothing is in flight.
    pub fn march_poll(
        &mut self,
        ctx: &GpuContext,
        allocator: &Arc<Mutex<Allocator>>,
    ) -> Option<Vec<ProbeSh>> {
        let f = self.in_flight.as_ref()?;
        let device = &ctx.device;
        // Non-blocking check: NOT_READY → still running.
        let ready = unsafe { device.get_fence_status(f.fence) }.unwrap_or(false);
        if !ready {
            return None;
        }
        let mut f = self.in_flight.take().unwrap();
        let count = f.count;
        let bytes = f.out_buf.read();
        let raw: &[f32] = bytemuck::cast_slice(&bytes);
        let mut out = Vec::with_capacity(count);
        for p in 0..count {
            let b = p * 16;
            let mut sh = ProbeSh::default();
            for k in 0..4 {
                sh.coeffs[k] = [raw[b + k * 4], raw[b + k * 4 + 1], raw[b + k * 4 + 2]];
                sh.dist[k] = raw[b + k * 4 + 3];
            }
            out.push(sh);
        }
        let mut alloc = allocator.lock().unwrap();
        unsafe {
            device.destroy_fence(f.fence, None);
            device.free_command_buffers(self.cmd_pool, &[f.cb]);
            device.destroy_descriptor_pool(f.desc_pool, None);
        }
        f.light_buf.destroy(device, &mut alloc);
        f.probe_buf.destroy(device, &mut alloc);
        f.emitter_buf.destroy(device, &mut alloc);
        f.out_buf.destroy(device, &mut alloc);
        Some(out)
    }

    /// Screen-space GI final gather: per-pixel GDF trace into `out_view` (a
    /// STORAGE image). Reads `depth_view` (must already be SHADER_READ_ONLY).
    /// Synchronous (blocks) for now; async is a follow-up.
    /// Leaves `out` in SHADER_READ_ONLY_OPTIMAL for the forward pass to sample.
    /// Record the Flux gather dispatch into an externally-owned command buffer
    /// (the main frame cb) instead of submitting + blocking on its own fence.
    /// Returns the transient descriptor pool/buffers the caller must keep alive
    /// until that frame's fence signals, then free via [`ScreenGiTransient::destroy`].
    #[allow(clippy::too_many_arguments)]
    pub fn screen_resolve_record(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        allocator: &Arc<Mutex<Allocator>>,
        depth_view: vk::ImageView,
        depth_sampler: vk::Sampler,
        out_image: vk::Image,
        out_view: vk::ImageView,
        history_view: vk::ImageView,
        // Probe-res denoise target: the gather is written to `out_image`, then the
        // à-trous denoise reads it and writes here; the forward samples THIS.
        filtered_image: vk::Image,
        filtered_view: vk::ImageView,
        // Directional probe history buffers. The gather writes radiance/hit distance;
        // the denoiser writes separate filtered SH lanes consumed by integrate.
        sh_buf: vk::Buffer,
        prev_sh_buf: vk::Buffer,
        raw_rays: vk::Buffer,
        // Full-res integrate output (the forward samples THIS) + full screen extent.
        final_image: vk::Image,
        final_view: vk::ImageView,
        temporal_images: TemporalImages,
        full_extent: vk::Extent2D,
        extent: vk::Extent2D,
        // Hardware ray-query backend: when Some + the device has ray-query, the
        // gather is screen_gi_rt (TLAS trace) instead of the SDF march. (tlas, instance SSBO).
        rt: Option<(vk::AccelerationStructureKHR, vk::Buffer)>,
        rt_textures: Option<vk::DescriptorSet>,
        lights: &[BakeLight],
        area_lights: &[crate::GpuLight],
        area_atlas: (vk::Sampler,vk::ImageView),
        emitters: &[GpuGiEmitter],
        history_valid: bool,
        // World SH radiance cache: the renderer's probe SSBO (always a valid
        // buffer; `params.probe_info.x` = volume count, 0 = none → shader skips it).
        probe_buffer: vk::Buffer,
        surface: Option<crate::flux_surface::Resources>,
        lighting_update: Option<crate::flux_surface::LightingUpdate>,
        screen: crate::flux_screen_history::Resources,
        radiance: crate::flux_radiance_cache::Resources,
        mut params: ScreenGiParams,
        // This frame slot's resources from its previous trace (retired): reused
        // (pool reset, buffers rewritten) instead of reallocating every trace.
        reuse: Option<ScreenGiTransient>,
    ) -> Result<Option<ScreenGiTransient>> {
        let Some(gdf) = self.gdf.as_ref() else {
            return Ok(None);
        };
        let mut alloc = allocator.lock().unwrap();
        let mut gi_lights: Vec<GiLight> = lights.iter().map(pack_light).collect();
        gi_lights.extend(area_lights.iter().map(|l|GiLight {pos_kind:l.pos_kind,dir_range:l.dir_range,color:l.color,spot:l.spot}));
        if gi_lights.is_empty() {gi_lights.push(GiLight::zeroed());}
        let gi_emitters: Vec<GiEmitter> = if emitters.is_empty() {
            vec![GiEmitter::zeroed()]
        } else {
            emitters
                .iter()
                .map(|e| GiEmitter {
                    center_radius: [e.center[0], e.center[1], e.center[2], e.radius],
                    emission: [e.emission[0], e.emission[1], e.emission[2], 0.0],
                })
                .collect()
        };
        // Paired diagnostic runs can measure adaptive placement against dense
        // tracing without changing geometry, ray budget or the resolve stages.
        static DENSE_PROBES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let dense = *DENSE_PROBES.get_or_init(|| std::env::var("ZESTENGINE_FLUX_DENSE_PROBES").is_ok_and(|v| v=="1"));
        params.gdf_min = [gdf.min[0], gdf.min[1], gdf.min[2], if dense {1.0} else {0.0}];
        params.gdf_max = [gdf.max[0], gdf.max[1], gdf.max[2], self.mesh_sdfs.current.as_ref().map_or(0,|s|s.node_count) as f32];
        // Profiling reference uses identical rays with the old serial cell
        // schedule; the default distributes those rays evenly across lanes.
        static BALANCED_GATHER: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let balanced=*BALANCED_GATHER.get_or_init(|| !std::env::var("ZESTENGINE_FLUX_BALANCED_GATHER").is_ok_and(|v|v=="0"));
        params.transport[2]=if balanced {0.0} else {1.0};
        // Auto max trace distance (march.y == 0): the GDF box diagonal, so rays
        // can cross the whole scene without an arbitrary fixed cap.
        if params.march[1] <= 0.0 {
            let d = [
                gdf.max[0] - gdf.min[0],
                gdf.max[1] - gdf.min[1],
                gdf.max[2] - gdf.min[2],
            ];
            params.march[1] = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        }
        #[cfg(test)]
        if let Some(temporal) = self.temporal_override { params.temporal[..3].copy_from_slice(&temporal[..3]); }
        #[cfg(test)]
        if let Some(alpha) = self.probe_history_alpha { params.temporal[0] = alpha; }
        #[cfg(test)]
        if self.screen_trace_override==Some(false) {params.screen.options[0]=0.0;}
        #[cfg(test)]
        if self.importance_sampling==Some(false) {params.transport[1]=1.0;}
        #[cfg(test)]
        if let Some(balanced)=self.balanced_gather {params.transport[2]=if balanced {0.0} else {1.0};}
        #[cfg(test)]
        if let Some(seed)=self.gather_seed {params.misc[0]=seed;}
        params.probe_info[2] = surface.map_or(0, |s| s.node_count) as f32;
        params.counts[2] = (lights.len()+area_lights.len()) as u32;
        params.counts[3] = emitters.len() as u32;
        params.misc[1] = extent.width;
        params.misc[2] = extent.height;
        // params.march[2] (temporal alpha) is set by the caller, motion-aware:
        // snap when fresh, trust the new trace more while moving, converge gently
        // when still. (history_valid only gates the descriptor sanity below.)
        let _ = history_valid;
        // Hardware RT gather active only when a TLAS was supplied AND the device
        // built the RT pipeline. Its own RtGiParams UBO (subset of the SDF params).
        let rt_active = rt.is_some() && self.rt_pipeline.is_some();
        let reused = match reuse {
            Some(t) if t.light_cap >= gi_lights.len() && t.emitter_cap >= gi_emitters.len() => {
                unsafe { device.reset_descriptor_pool(t.desc_pool, vk::DescriptorPoolResetFlags::empty())? };
                Some(t)
            }
            Some(mut t) => {
                t.destroy(device, &mut alloc);
                None
            }
            None => None,
        };
        let (desc_pool, mut light_buf, mut emitter_buf, light_cap, emitter_cap, mut param_buf, mut rt_param_buf) =
            match reused {
                Some(t) => (t.desc_pool, t.light_buf, t.emitter_buf, t.light_cap, t.emitter_cap, t.param_buf, t.rt_param_buf),
                None => {
                    let light_cap = gi_lights.len().next_power_of_two().max(16);
                    let emitter_cap = gi_emitters.len().next_power_of_two().max(16);
                    let light_buf = host_ssbo(device, &mut alloc, &vec![0u8; light_cap * size_of::<GiLight>()], "sgi lights")?;
                    let emitter_buf =
                        host_ssbo(device, &mut alloc, &vec![0u8; emitter_cap * size_of::<GiEmitter>()], "sgi emitters")?;
                    let param_buf = Buffer::new(
                        device,
                        &mut alloc,
                        size_of::<ScreenGiParams>() as u64,
                        vk::BufferUsageFlags::UNIFORM_BUFFER,
                        MemoryLocation::CpuToGpu,
                        "sgi params",
                    )?;
                    (self.new_screen_pool(device)?, light_buf, emitter_buf, light_cap, emitter_cap, param_buf, None)
                }
            };
        light_buf.write(0, bytemuck::cast_slice(&gi_lights));
        emitter_buf.write(0, bytemuck::cast_slice(&gi_emitters));
        param_buf.write(0, bytemuck::bytes_of(&params));
        if rt_active {
            if rt_param_buf.is_none() {
                rt_param_buf = Some(Buffer::new(
                    device,
                    &mut alloc,
                    size_of::<RtGiParams>() as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    MemoryLocation::CpuToGpu,
                    "sgi rt params",
                )?);
            }
            rt_param_buf.as_mut().unwrap().write(0, bytemuck::bytes_of(&RtGiParams::from_screen(&params)));
        }
        drop(alloc);

        let layouts = [self.screen_set_layout];
        let set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(desc_pool)
                    .set_layouts(&layouts),
            )?[0]
        };
        self.write_image(device, set, 0, self.dist_sampler, gdf.dist_tex.view);
        self.write_image(device, set, 1, self.index_sampler, gdf.index_tex.view);
        write_ssbo(device, set, 2, gdf.inst_buf.handle);
        write_ssbo(device, set, 3, light_buf.handle);
        write_ssbo(device, set, 4, emitter_buf.handle);
        write_ssbo(device, set, 9, probe_buffer); // world SH radiance cache
        write_ssbo(device, set, 10, sh_buf); // directional history output
        write_ssbo(device, set, 25, raw_rays);
        write_ssbo(device,set,26,screen.hierarchy);
        self.write_image(device,set,27,depth_sampler,screen.image);
        write_ssbo(device, set, 11, prev_sh_buf); // previous directional history
        let bind_surface = |set, base| {
            write_ssbo(device, set, base, surface.map_or(probe_buffer, |s| s.cards));
            write_ssbo(device, set, base+1, surface.map_or(probe_buffer, |s| s.nodes));
            for i in 0..4 {
                self.write_image(device, set, base+2+i, depth_sampler, surface.map_or(depth_view, |s| s.views[i as usize]));
            }
            for i in 0..2 {
                self.write_image(device,set,base+6+i,self.dist_sampler,surface.map_or(depth_view,|s|s.lighting[i as usize]));
                let image = [vk::DescriptorImageInfo::default().image_view(lighting_update.map_or(out_view,|u|u.output[i as usize]))
                    .image_layout(vk::ImageLayout::GENERAL)];
                let write = vk::WriteDescriptorSet::default().dst_set(set).dst_binding(base+8+i)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&image);
                unsafe {device.update_descriptor_sets(&[write],&[]);}
            }
        };
        bind_surface(set, 12);
        let bind_radiance=|set,base| {write_ssbo(device,set,base,radiance.data);write_ssbo(device,set,base+1,radiance.work);write_ubo(device,set,base+2,radiance.params);};
        bind_radiance(set,28);
        self.write_image(device,set,31,area_atlas.0,area_atlas.1);
        let (mesh_buffers,mesh_voxels)=self.mesh_sdfs.current.as_ref().map_or(([gdf.inst_buf.handle;2],gdf.dist_tex.view),|s|s.resources());
        for (i,&buffer) in mesh_buffers.iter().enumerate() { write_ssbo(device,set,22+i as u32,buffer); }
        self.write_image(device,set,24,self.dist_sampler,mesh_voxels);
        self.write_image(device, set, 5, depth_sampler, depth_view);
        // Storage image (binding 6): GENERAL layout for compute write.
        let img_info = [vk::DescriptorImageInfo::default()
            .image_view(out_view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let img_write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(6)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&img_info);
        let buf_info = [vk::DescriptorBufferInfo::default()
            .buffer(param_buf.handle)
            .range(vk::WHOLE_SIZE)];
        let buf_write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(7)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&buf_info);
        // Binding 8: history = the PREVIOUS frame's gather image (ping-pong, in
        // SHADER_READ), reprojected per pixel for temporal accumulation.
        let _ = history_valid;
        let hist_info = [vk::DescriptorImageInfo::default()
            .sampler(depth_sampler)
            .image_view(history_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let hist_write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(8)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&hist_info);
        unsafe { device.update_descriptor_sets(&[img_write, buf_write, hist_write], &[]) };

        // Hardware RT gather descriptor set (when active): mirrors screen_gi_rt.comp
        // bindings — depth, out(storage), history, TLAS, instances, lights, emitters,
        // sh-out, sh-prev, params. Allocated from the same per-frame pool.
        let rt_set = if let (true, Some((tlas, inst_buf)), Some(rtl)) =
            (rt_active, rt, self.rt_set_layout)
        {
            let layouts = [rtl];
            let rs = unsafe {
                device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(desc_pool)
                        .set_layouts(&layouts),
                )?[0]
            };
            write_ssbo(device,rs,21,screen.hierarchy);
            self.write_image(device,rs,22,depth_sampler,screen.image);
            self.write_image(device, rs, 0, depth_sampler, depth_view); // depth (sampled)
            let rt_out_info = [vk::DescriptorImageInfo::default()
                .image_view(out_view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let rt_out_w = vk::WriteDescriptorSet::default()
                .dst_set(rs)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&rt_out_info);
            self.write_image(device, rs, 2, depth_sampler, history_view); // history (sampled)
            // TLAS (binding 3): acceleration-structure descriptor via push_next.
            let tlas_arr = [tlas];
            let mut accel_info = vk::WriteDescriptorSetAccelerationStructureKHR::default()
                .acceleration_structures(&tlas_arr);
            let mut tlas_w = vk::WriteDescriptorSet::default()
                .dst_set(rs)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .push_next(&mut accel_info);
            tlas_w.descriptor_count = 1; // not set by image/buffer_info; required for accel
            unsafe { device.update_descriptor_sets(&[rt_out_w, tlas_w], &[]) };
            write_ssbo(device, rs, 4, inst_buf); // instances (vtx/idx addrs + albedo/emission)
            write_ssbo(device, rs, 5, light_buf.handle);
            self.write_image(device,rs,26,area_atlas.0,area_atlas.1);
            write_ssbo(device, rs, 6, emitter_buf.handle);
            write_ssbo(device, rs, 7, sh_buf);
            write_ssbo(device, rs, 8, prev_sh_buf);
            write_ssbo(device, rs, 10, probe_buffer); // world SH radiance cache
            bind_surface(rs, 11);
            bind_radiance(rs,23);
            let rt_pbuf = rt_param_buf.as_ref().unwrap().handle;
            let rt_buf_info = [vk::DescriptorBufferInfo::default()
                .buffer(rt_pbuf)
                .range(vk::WHOLE_SIZE)];
            let rt_buf_w = vk::WriteDescriptorSet::default()
                .dst_set(rs)
                .dst_binding(9)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&rt_buf_info);
            unsafe { device.update_descriptor_sets(&[rt_buf_w], &[]) };
            Some(rs)
        } else {
            None
        };

        let groups_x = extent.width.div_ceil(8);
        let groups_y = extent.height.div_ceil(8);
        unsafe {
            // Output is a fresh (ping-pong) image fully overwritten this trace,
            // so discard its old contents. History is the other image (read-only).
            let to_general = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .image(out_image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            device.cmd_pipeline_barrier(
                cb,
                // Wait for the previous frame's reads of this image (forward
                // sampling / next-frame history) before overwriting it.
                vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_general],
            );
            if let Some(update) = lighting_update {
                self.ts(device,cb,2*crate::gpu_zone::GI_SURFACE as u32);
                let (source,set_layout,layout,pipeline,base,bindings) = if let (Some(rs),Some(sl),Some(pl),Some(p)) =
                    (rt_set,self.rt_set_layout,self.rt_pipeline_layout,self.surface_rt_pipeline) {
                    (rs,sl,pl,p,11,21)
                } else { (set,self.screen_set_layout,self.screen_pipeline_layout,self.surface_pipeline,12,26) };
                let layouts=[set_layout];
                let light_set = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(desc_pool).set_layouts(&layouts))?[0];
                let copies: Vec<_> = (0..bindings).map(|binding| vk::CopyDescriptorSet::default()
                    .src_set(source).src_binding(binding).dst_set(light_set).dst_binding(binding).descriptor_count(1)).collect();
                device.update_descriptor_sets(&[],&copies);
                self.write_image(device,light_set,if base==11 {26} else {31},area_atlas.0,area_atlas.1);
                for i in 0..2 {
                    self.write_image(device,light_set,base+6+i,self.dist_sampler,update.previous[i as usize]);
                }
                device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,pipeline);
                device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,layout,0,&[light_set],&[]);
                if base==11 {device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,layout,1,&[rt_textures.unwrap()],&[]);}
                device.cmd_dispatch(cb,update.page_extent.width.div_ceil(8),update.page_extent.height.div_ceil(8),update.card_count);
                let barriers: Vec<_> = update.images.iter().map(|&image| vk::ImageMemoryBarrier::default()
                    .image(image).old_layout(vk::ImageLayout::GENERAL).new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1))).collect();
                device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::COMPUTE_SHADER,vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),&[],&[],&barriers);
                self.ts(device,cb,2*crate::gpu_zone::GI_SURFACE as u32+1);
            }
            if radiance.enabled {
                let (set,layout,pipeline)=if let (Some(set),Some(layout),Some(pipeline))=(rt_set,self.rt_pipeline_layout,self.radiance_rt_pipeline) {
                    (set,layout,pipeline)
                } else {(set,self.screen_pipeline_layout,self.radiance_pipeline)};
                device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,pipeline);
                device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,layout,0,&[set],&[]);
                if rt_active {device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,layout,1,&[rt_textures.unwrap()],&[]);}
                device.cmd_dispatch_indirect(cb,radiance.work,0);
                crate::flux_radiance_cache::barrier(device,cb);
            }
            let has_area_lights=!area_lights.is_empty();
            #[cfg(test)]
            let has_area_lights=self.area_gather_variant.unwrap_or(has_area_lights);
            let gather_variant=(if !radiance.enabled {0} else if radiance.full_interpolation {2} else {1})+if has_area_lights {3} else {0};
            // Gather backend: hardware ray-query (TLAS) when active, else SDF march.
            // Both write `out_image` + `sh_buf` with identical layout, so the denoise
            // + integrate below are backend-agnostic.
            if let (Some(rs), Some(rtp), Some(rtl)) =
                (rt_set, self.rt_pipeline, self.rt_pipeline_layout)
            {
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, rtp[gather_variant]);
                device.cmd_bind_descriptor_sets(
                    cb, vk::PipelineBindPoint::COMPUTE, rtl, 0, &[rs,rt_textures.unwrap()], &[],
                );
            } else {
                let pipeline=if self.mesh_sdfs.current.is_some() {self.screen_detail_pipeline[gather_variant]} else {self.screen_pipeline[gather_variant]};
                device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
                device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    self.screen_pipeline_layout,
                    0,
                    &[set],
                    &[],
                );
            }
            self.ts(device, cb, 2 * crate::gpu_zone::GI_GATHER as u32);
            if rt_set.is_some() {
                device.cmd_dispatch(cb, extent.width.div_ceil(2), extent.height, 1);
            } else if self.mesh_sdfs.current.is_some() {
                let barrier=[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ|vk::AccessFlags::SHADER_WRITE)];
                // Scratch is reused by this view on the graphics queue. Order its
                // prior resolve reads before this frame's tracing writes.
                let scratch=[vk::BufferMemoryBarrier::default().buffer(raw_rays).offset(0).size(vk::WHOLE_SIZE)
                    .src_access_mask(vk::AccessFlags::SHADER_READ|vk::AccessFlags::SHADER_WRITE|vk::AccessFlags::INDIRECT_COMMAND_READ)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)];
                device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::COMPUTE_SHADER|vk::PipelineStageFlags::DRAW_INDIRECT,vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),&[],&scratch,&[]);
                device.cmd_update_buffer(cb,raw_rays,0,bytemuck::cast_slice(&[0u32,0,1,16]));
                let init=[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ|vk::AccessFlags::SHADER_WRITE)];
                device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::TRANSFER,vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),&init,&[],&[]);
                device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,self.screen_receiver_pipeline);
                device.cmd_dispatch(cb,groups_x,groups_y,1);
                device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::COMPUTE_SHADER,vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),&barrier,&[],&[]);
                device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,self.screen_placement_pipeline);
                device.cmd_dispatch(cb,extent.width.div_ceil(32),extent.height.div_ceil(32),1);
                let jobs=[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ|vk::AccessFlags::SHADER_WRITE|vk::AccessFlags::INDIRECT_COMMAND_READ)];
                device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::COMPUTE_SHADER,vk::PipelineStageFlags::COMPUTE_SHADER|vk::PipelineStageFlags::DRAW_INDIRECT,
                    vk::DependencyFlags::empty(),&jobs,&[],&[]);
                device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,self.screen_detail_pipeline[gather_variant]);
                device.cmd_dispatch_indirect(cb,raw_rays,4);
                device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::COMPUTE_SHADER,vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),&barrier,&[],&[]);
                device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,self.screen_reduce_pipeline);
                device.cmd_dispatch(cb,groups_x,groups_y,1);
            } else {
                device.cmd_dispatch(cb, groups_x, groups_y, 1);
            }
            self.ts(device, cb, 2 * crate::gpu_zone::GI_GATHER as u32 + 1);
            // Gather out: GENERAL -> SHADER_READ_ONLY so the DENOISE compute can
            // sample it (and the forward as a fallback). Made available to compute.
            let to_read = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .image(out_image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_read],
            );

            // --- Probe-space denoise pass (à-trous spatial filter) ---
            // Allocate the denoise set from the same per-frame pool, point it at the
            // gather (sampled) + the filtered output (storage), dispatch at probe res.
            let dn_layouts = [self.denoise_set_layout];
            let dn_set = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(desc_pool)
                        .set_layouts(&dn_layouts),
                )
                .map(|v| v[0])?;
            self.write_image(device, dn_set, 0, depth_sampler, out_view);
            write_ssbo(device, dn_set, 2, sh_buf);
            let denoise_sh_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(sh_buf).offset(0).size(vk::WHOLE_SIZE);
            device.cmd_pipeline_barrier(cb,
                vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(), &[], &[denoise_sh_barrier], &[]);
            let dn_img_info = [vk::DescriptorImageInfo::default()
                .image_view(filtered_view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let dn_img_write = vk::WriteDescriptorSet::default()
                .dst_set(dn_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&dn_img_info);
            device.update_descriptor_sets(&[dn_img_write], &[]);
            // filtered image: UNDEFINED -> GENERAL (fully overwritten this pass).
            let f_to_general = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .image(filtered_image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            device.cmd_pipeline_barrier(
                cb,
                // Wait for the previous frame's reads of this image (forward
                // sampling / next-frame history) before overwriting it.
                vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[f_to_general],
            );
            let dn_push = DenoisePush {
                size: [extent.width as i32, extent.height as i32],
                step: 1,
                _pad: 0,
            };
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.denoise_pipeline);
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.denoise_pipeline_layout,
                0,
                &[dn_set],
                &[],
            );
            device.cmd_push_constants(
                cb,
                self.denoise_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&dn_push),
            );
            self.ts(device, cb, 2 * crate::gpu_zone::GI_DENOISE as u32);
            device.cmd_dispatch(cb, groups_x, groups_y, 1);
            self.ts(device, cb, 2 * crate::gpu_zone::GI_DENOISE as u32 + 1);
            // filtered image: GENERAL -> SHADER_READ_ONLY so the INTEGRATE compute
            // can sample it (made available to compute, not fragment).
            let f_to_read = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .image(filtered_image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            // The directional filter wrote SH; make those writes visible to integrate.
            let sh_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(sh_buf)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[sh_barrier],
                &[f_to_read],
            );

            // --- Full-res integrate pass (per-pixel-normal SH resolve) ---
            let (in_set_layout, in_pipeline_layout, in_pipeline) = if rt_active {
                self.rt_integrate.expect("RT integrate accompanies the RT gather")
            } else {
                (self.integrate_set_layout, self.integrate_pipeline_layout, self.integrate_pipeline)
            };
            let in_layouts = [in_set_layout];
            let in_set = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(desc_pool)
                        .set_layouts(&in_layouts),
                )
                .map(|v| v[0])?;
            if rt_active {
                write_ssbo(device,in_set,11,rt.unwrap().1);
                let handles = [rt.unwrap().0];
                let mut accel = vk::WriteDescriptorSetAccelerationStructureKHR::default()
                    .acceleration_structures(&handles);
                let mut write = vk::WriteDescriptorSet::default().dst_set(in_set).dst_binding(7)
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR).push_next(&mut accel);
                write.descriptor_count = 1;
                device.update_descriptor_sets(&[write], &[]);
            }
            self.write_image(device, in_set, 0, depth_sampler, filtered_view); // scalar GI
            self.write_image(device, in_set, 1, depth_sampler, depth_view);    // depth prepass
            write_ssbo(device, in_set, 2, sh_buf);                             // per-probe SH
            write_ssbo(device, in_set, 4, emitter_buf.handle);                 // emitters
            self.write_image(device, in_set, 5, self.dist_sampler, gdf.dist_tex.view); // GDF (emitter shadows)
            for (i,&buffer) in mesh_buffers.iter().enumerate() { write_ssbo(device,in_set,8+i as u32,buffer); }
            self.write_image(device,in_set,10,self.dist_sampler,mesh_voxels);
            write_ubo(device, in_set, 6, param_buf.handle);                    // gather params (GDF bounds, eps)
            let in_img_info = [vk::DescriptorImageInfo::default()
                .image_view(final_view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let in_img_write = vk::WriteDescriptorSet::default()
                .dst_set(in_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&in_img_info);
            device.update_descriptor_sets(&[in_img_write], &[]);
            let fin_to_general = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                .image(final_image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            device.cmd_pipeline_barrier(
                cb,
                // Wait for the previous frame's reads of this image (forward
                // sampling / next-frame history) before overwriting it.
                vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[fin_to_general],
            );
            let in_push = IntegratePush {
                inv_view_proj: params.inv_view_proj,
                cam: params.cam,
                dims: [
                    full_extent.width as i32,
                    full_extent.height as i32,
                    extent.width as i32,
                    extent.height as i32,
                ],
                misc: [params.misc[3] as f32, params.sky[3], if params.probe_info[3] > 0.5 { 0.0 } else { emitters.len() as f32 }, 0.0],
            };
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, in_pipeline);
            if rt_active {device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,in_pipeline_layout,1,&[rt_textures.unwrap()],&[]);}
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                in_pipeline_layout,
                0,
                &[in_set],
                &[],
            );
            device.cmd_push_constants(
                cb,
                in_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&in_push),
            );
            self.ts(device, cb, 2 * crate::gpu_zone::GI_INTEGRATE as u32);
            device.cmd_dispatch(cb, full_extent.width.div_ceil(8), full_extent.height.div_ceil(8), 1);
            self.ts(device, cb, 2 * crate::gpu_zone::GI_INTEGRATE as u32 + 1);
            // integrated image: GENERAL -> SHADER_READ_ONLY for the forward pass.
            let fin_to_read = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .image(final_image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[fin_to_read],
            );
        }

        self.record_full_history(device,cb,desc_pool,depth_sampler,final_view,depth_view,
            param_buf.handle,temporal_images,full_extent,history_valid)?;
        crate::flux_surface::finish_feedback(device,cb,surface);

        Ok(Some(ScreenGiTransient {
            desc_pool,
            light_buf,
            emitter_buf,
            light_cap,
            emitter_cap,
            param_buf,
            rt_param_buf,
            mesh_snapshot: self.mesh_sdfs.current.clone(),
        }))
    }

    #[allow(clippy::too_many_arguments)]
    fn record_full_history(&self,device: &ash::Device,cb: vk::CommandBuffer,pool: vk::DescriptorPool,
        sampler: vk::Sampler,current: vk::ImageView,depth: vk::ImageView,params: vk::Buffer,
        images: TemporalImages,extent: vk::Extent2D,valid: bool) -> Result<()> {
        let (set_layout,layout,pipeline)=self.full_temporal;
        let set=unsafe {device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool).set_layouts(&[set_layout]))?[0]};
        for (i,view) in [current,depth,images.previous_views[0],images.previous_views[1]].into_iter().enumerate() {
            self.write_image(device,set,i as u32,sampler,view);
        }
        write_ubo(device,set,6,params);
        for i in 0..2 {
            let info=[vk::DescriptorImageInfo::default().image_view(images.current_views[i]).image_layout(vk::ImageLayout::GENERAL)];
            let write=vk::WriteDescriptorSet::default().dst_set(set).dst_binding(4+i as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&info);
            unsafe {device.update_descriptor_sets(&[write],&[]);}
        }
        let range=vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
        let transition=|image,from,to,src,dst| {
            let barrier=vk::ImageMemoryBarrier::default().image(image).old_layout(from).new_layout(to)
                .src_access_mask(src).dst_access_mask(dst).subresource_range(range);
            unsafe {device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::ALL_COMMANDS,vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),&[],&[],&[barrier]);}
        };
        if !valid {
            for image in images.previous {
                transition(image,vk::ImageLayout::UNDEFINED,vk::ImageLayout::TRANSFER_DST_OPTIMAL,vk::AccessFlags::empty(),vk::AccessFlags::TRANSFER_WRITE);
                unsafe {device.cmd_clear_color_image(cb,image,vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue {float32:[0.0;4]},&[range]);}
                transition(image,vk::ImageLayout::TRANSFER_DST_OPTIMAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    vk::AccessFlags::TRANSFER_WRITE,vk::AccessFlags::SHADER_READ);
            }
        }
        for image in images.current {
            transition(image,vk::ImageLayout::UNDEFINED,vk::ImageLayout::GENERAL,vk::AccessFlags::empty(),vk::AccessFlags::SHADER_WRITE);
        }
        unsafe {
            device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,pipeline);
            device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,layout,0,&[set],&[]);
            device.cmd_dispatch(cb,extent.width.div_ceil(8),extent.height.div_ceil(8),1);
        }
        for image in images.current {
            transition(image,vk::ImageLayout::GENERAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,vk::AccessFlags::SHADER_WRITE,vk::AccessFlags::SHADER_READ);
        }
        Ok(())
    }

    /// Descriptor pool for one trace (gather + denoise + integrate sets).
    fn new_screen_pool(&self, device: &ash::Device) -> Result<vk::DescriptorPool> {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 64, // gather, surface lighting, area atlas, denoise/integrate
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: 56, // gather, surface lighting, mesh fields, radiance cache, integrate
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 16, // gather, surface direct/indirect outputs, denoise/integrate
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 8, // SDF/RT, radiance cache, surface lighting, integrate, full history
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                descriptor_count: 3, // RT gather, surface lighting, emitter resolve
            },
        ];
        Ok(unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(6) // gather, surface lighting, denoise, integrate, full history
                    .pool_sizes(&pool_sizes),
                None,
            )?
        })
    }

    fn write_image(
        &self,
        device: &ash::Device,
        set: vk::DescriptorSet,
        binding: u32,
        sampler: vk::Sampler,
        view: vk::ImageView,
    ) {
        let info = [vk::DescriptorImageInfo::default()
            .sampler(sampler)
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(binding)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&info);
        unsafe { device.update_descriptor_sets(&[write], &[]) };
    }

    pub fn destroy(&mut self, device: &ash::Device, alloc: &mut Allocator) {
        if let Some(mut f) = self.in_flight.take() {
            unsafe {
                let _ = device.wait_fences("wait:fence@destroy", &[f.fence], true, u64::MAX);
                device.destroy_fence(f.fence, None);
                device.free_command_buffers(self.cmd_pool, &[f.cb]);
                device.destroy_descriptor_pool(f.desc_pool, None);
            }
            f.light_buf.destroy(device, alloc);
            f.probe_buf.destroy(device, alloc);
            f.emitter_buf.destroy(device, alloc);
            f.out_buf.destroy(device, alloc);
        }
        self.mesh_sdfs.destroy(device,alloc);
        if let Some(mut gdf) = self.gdf.take() {
            gdf.destroy(device, alloc);
        }
        self.radiance_cache.destroy(device,alloc);
        unsafe {
            device.destroy_command_pool(self.cmd_pool, None);
            device.destroy_sampler(self.dist_sampler, None);
            device.destroy_sampler(self.index_sampler, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            for pipeline in self.screen_pipeline {device.destroy_pipeline(pipeline,None);}
            for pipeline in self.screen_detail_pipeline {device.destroy_pipeline(pipeline,None);}
            device.destroy_pipeline(self.screen_receiver_pipeline, None);
            device.destroy_pipeline(self.screen_placement_pipeline, None);
            device.destroy_pipeline(self.screen_reduce_pipeline, None);
            device.destroy_pipeline(self.surface_pipeline, None);
            device.destroy_pipeline(self.radiance_pipeline,None);
            if let Some(p)=self.radiance_rt_pipeline {device.destroy_pipeline(p,None);}
            device.destroy_pipeline(self.full_temporal.2,None);
            device.destroy_pipeline_layout(self.full_temporal.1,None);
            device.destroy_descriptor_set_layout(self.full_temporal.0,None);
            if let Some(p)=self.surface_rt_pipeline {device.destroy_pipeline(p,None);}
            device.destroy_pipeline_layout(self.screen_pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.screen_set_layout, None);
            device.destroy_pipeline(self.denoise_pipeline, None);
            device.destroy_pipeline_layout(self.denoise_pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.denoise_set_layout, None);
            device.destroy_pipeline(self.integrate_pipeline, None);
            device.destroy_pipeline_layout(self.integrate_pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.integrate_set_layout, None);
            if let Some((set, layout, pipeline)) = self.rt_integrate {
                device.destroy_pipeline(pipeline, None);
                device.destroy_pipeline_layout(layout, None);
                device.destroy_descriptor_set_layout(set, None);
            }
            if let Some(pipelines) = self.rt_pipeline {
                for pipeline in pipelines {device.destroy_pipeline(pipeline,None);}
            }
            if let Some(l) = self.rt_pipeline_layout {
                device.destroy_pipeline_layout(l, None);
            }
            if let Some(textures)=self.rt_textures.take() {textures.destroy(device);}
            if let Some(l) = self.rt_set_layout {
                device.destroy_descriptor_set_layout(l, None);
            }
        }
    }
}

fn host_ssbo(device: &ash::Device, alloc: &mut Allocator, data: &[u8], name: &str) -> Result<Buffer> {
    let mut buf = Buffer::new(
        device,
        alloc,
        data.len().max(4) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::CpuToGpu,
        name,
    )?;
    if !data.is_empty() {
        buf.write(0, data);
    }
    Ok(buf)
}

fn write_ssbo(device: &ash::Device, set: vk::DescriptorSet, binding: u32, buffer: vk::Buffer) {
    let info = [vk::DescriptorBufferInfo::default()
        .buffer(buffer)
        .range(vk::WHOLE_SIZE)];
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&info);
    unsafe { device.update_descriptor_sets(&[write], &[]) };
}

fn write_ubo(device: &ash::Device, set: vk::DescriptorSet, binding: u32, buffer: vk::Buffer) {
    let info = [vk::DescriptorBufferInfo::default().buffer(buffer).range(vk::WHOLE_SIZE)];
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(&info);
    unsafe { device.update_descriptor_sets(&[write], &[]) };
}


/// Disabled, two-sample, and full-reference gather variants. Specialization
/// allows the driver to eliminate unused interpolation and Area Light sampling.
/// The latter otherwise increases register pressure even in scenes without them.
fn make_gather_pipelines(device:&ash::Device,layout:vk::PipelineLayout,spv:&[u8])->Result<[vk::Pipeline;6]> {
    let module=crate::screen_pass::shader_module(device,spv)?;
    let entries=[vk::SpecializationMapEntry {constant_id:0,offset:0,size:4},
        vk::SpecializationMapEntry {constant_id:1,offset:4,size:4},
        vk::SpecializationMapEntry {constant_id:2,offset:8,size:4}];
    let mut pipelines=Vec::new();
    for variant in 0..6 {
        let values=[u32::from(variant%3==2),u32::from(variant%3!=0),u32::from(variant>=3)];
        let specialization=vk::SpecializationInfo::default().map_entries(&entries).data(bytemuck::cast_slice(&values));
        let info=vk::ComputePipelineCreateInfo::default().layout(layout).stage(vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main").specialization_info(&specialization));
        match unsafe {device.create_compute_pipelines(vk::PipelineCache::null(),&[info],None)} {
            Ok(created)=>pipelines.extend(created),
            Err((partial,error))=>{unsafe {
                for pipeline in pipelines.into_iter().chain(partial) {if pipeline!=vk::Pipeline::null() {device.destroy_pipeline(pipeline,None);}}
                device.destroy_shader_module(module,None);
            }return Err(error.into());}
        }
    }
    unsafe {device.destroy_shader_module(module,None);}
    Ok(pipelines.try_into().unwrap())
}
