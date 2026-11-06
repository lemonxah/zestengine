//! Ray-traced reflections via Vulkan ray query. Reuses the GI bake's
//! acceleration-structure builders (`bake::build_blas`/`build_tlas`): a BLAS is
//! cached per mesh, a TLAS is rebuilt each frame from the visible draws, and a
//! compute pass (`rt_reflect.comp`) traces a reflection ray per reflective pixel,
//! shades the hit, then reconstructs its signed environment replacement and
//! composites it into the resolved HDR target the post pass reads. Only used when the GPU supports ray query (`ctx.accel`).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use gpu_allocator::vulkan::Allocator;

use crate::alloc::Buffer;
use crate::bake::{self, Accel};
use crate::context::GpuContext;
use crate::rt_water::{RtWater, RtWaterFrame};
use crate::{GpuMesh, GpuWater, MAX_WATER_BODIES, frame::FRAMES_IN_FLIGHT};

const REFLECTION_VIEWS: usize = 2; // main (game/Scene) and Camera preview
const SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rt_reflect.comp.spv"));

/// Per-instance shading data, mirrors `Instance` in `rt_reflect.comp` and
/// `screen_gi_rt.comp` (scalar layout, 288 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuInstance {
    vtx_addr: u64,
    idx_addr: u64,
    /// rgb = base colour x mean albedo (GI gather, and RT hits without a texture).
    albedo: [f32; 4],
    /// rgb = emission x emission-map mean (GI gather), w = 1 for static geometry.
    emission: [f32; 4],
    /// rgb = base colour factor, multiplied by the sampled albedo texture;
    /// w = water body slot + 1 (0 = not water).
    base: [f32; 4],
    /// rgb = emission colour x intensity, multiplied by the emission texture.
    /// w = source draw index + 1, for instance-specific surface-cache lookup.
    emission_raw: [f32; 4],
    /// RT texture-table indices: albedo, normal, orm, emission (-1 = none).
    textures: [i32; 4],
    albedo_st: [f32; 4],
    normal_st: [f32; 4],
    orm_st: [f32; 4],
    emission_st: [f32; 4],
    /// x metallic, y roughness, z normal strength, w lightmap layer (-1 = none).
    params: [f32; 4],
    /// x base alpha, y cutoff (-1 opaque), z opacity texture index, w reserved.
    opacity: [f32; 4],
    /// Split roughness, metallic and emission mask (w reserved).
    extra_textures: [i32; 4],
    detail_textures: [i32; 4],
    detail: [f32; 4],
    detail_st: [f32; 4],
    orm_invert: [f32; 4],
    emission_mask_st: [f32; 4],
}

const _: () = assert!(std::mem::size_of::<GpuInstance>() == 288);

/// Size of the RT texture table (binding 9). Textures past it fall back to
/// their mean colour.
pub(crate) const MAX_RT_TEXTURES: u32 = 4096;

/// Persistent material texture descriptors shared by FluxRT gather, cache
/// lighting and emitter resolve. A frame slot is rewritten only after its
/// fence has completed, and only when the renderer's texture views changed.
pub(crate) struct MaterialTextures {
    pub layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    sampler: vk::Sampler,
    sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT],
    views: [Vec<vk::ImageView>; FRAMES_IN_FLIGHT],
}
impl MaterialTextures {
    pub fn new(device:&ash::Device)->Result<Self> {
        let bindings=[vk::DescriptorSetLayoutBinding::default().binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(MAX_RT_TEXTURES).stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let flags=[vk::DescriptorBindingFlags::PARTIALLY_BOUND];
        let mut binding_flags=vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);
        unsafe {
            let layout=device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings).push_next(&mut binding_flags),None)?;
            let pool=device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(FRAMES_IN_FLIGHT as u32)
                .pool_sizes(&[vk::DescriptorPoolSize {ty:vk::DescriptorType::COMBINED_IMAGE_SAMPLER,descriptor_count:MAX_RT_TEXTURES*FRAMES_IN_FLIGHT as u32}]),None)?;
            let sets=device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[layout;FRAMES_IN_FLIGHT]))?;
            let sampler=device.create_sampler(&vk::SamplerCreateInfo::default().mag_filter(vk::Filter::LINEAR).min_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR).address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT).address_mode_w(vk::SamplerAddressMode::REPEAT).max_lod(vk::LOD_CLAMP_NONE),None)?;
            Ok(Self {layout,pool,sampler,sets:sets.try_into().unwrap(),views:std::array::from_fn(|_|Vec::new())})
        }
    }
    pub fn prepare(&mut self,device:&ash::Device,frame:usize,views:&[vk::ImageView])->vk::DescriptorSet {
        let views=&views[..views.len().min(MAX_RT_TEXTURES as usize)];
        if self.views[frame]!=views {
            if !views.is_empty() {
                let infos:Vec<_>=views.iter().map(|&view|vk::DescriptorImageInfo::default().sampler(self.sampler)
                    .image_view(view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)).collect();
                unsafe {device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(self.sets[frame]).dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&infos)],&[]);}
            }
            self.views[frame].clear();self.views[frame].extend_from_slice(views);
        }
        self.sets[frame]
    }
    pub fn destroy(&self,device:&ash::Device) {unsafe {
        device.destroy_descriptor_pool(self.pool,None);device.destroy_sampler(self.sampler,None);device.destroy_descriptor_set_layout(self.layout,None);
    }}
}

/// TLAS instance masks: reflection rays see solid + water, the GI gather only
/// solid geometry, shadow rays only casters (water neither shadows nor blocks
/// bounce light; Cast Shadows = Off drops the caster bit, Shadows Only keeps
/// only it).
const MASK_SOLID: u8 = 0x01;
const MASK_WATER: u8 = 0x02;
const MASK_CASTER: u8 = 0x04;

/// TLAS mask for a non-water instance with this "Cast Shadows" setting.
pub(crate) fn solid_mask(shadow_casting: crate::ShadowCasting) -> u8 {
    let mut mask = 0;
    if shadow_casting.visible() {
        mask |= MASK_SOLID;
    }
    if shadow_casting.casts() {
        mask |= MASK_CASTER;
    }
    mask
}

/// One scene instance to reflect: mesh + world transform + surface colour.
struct TlasSlot {
    accel: Accel,
    /// AS build input (VkAccelerationStructureInstanceKHR array), host-visible.
    inst_input: Buffer,
    /// Per-instance shading records (`GpuInstance`), host-visible SSBO.
    shading: Buffer,
    scratch: Buffer,
    scratch_addr: u64,
    capacity: usize,
    /// Signature of the instances last built into this slot.
    sig: u64,
    /// Signature without transforms: when only transforms changed, the TLAS
    /// is refit (UPDATE) instead of rebuilt.
    layout: u64,
    /// Refits since the last full build (rebuilt periodically so BVH quality
    /// doesn't degrade as objects move).
    refits: u32,
}

/// Full TLAS rebuild after this many consecutive refits.
const TLAS_MAX_REFITS: u32 = 120;
const TLAS_FLAGS: vk::BuildAccelerationStructureFlagsKHR = vk::BuildAccelerationStructureFlagsKHR::from_raw(
    vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE.as_raw()
        | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE.as_raw(),
);

impl TlasSlot {
    fn destroy(mut self, ctx: &GpuContext, allocator: &mut Allocator) {
        if let Some(accel) = ctx.accel.as_ref() {
            unsafe { accel.destroy_acceleration_structure(self.accel.handle, None) };
        }
        self.accel.buffer.destroy(&ctx.device, allocator);
        self.inst_input.destroy(&ctx.device, allocator);
        self.shading.destroy(&ctx.device, allocator);
        self.scratch.destroy(&ctx.device, allocator);
    }
}

#[derive(Clone)]
pub(crate) struct RtInstance {
    pub mesh: usize,
    /// Index in the frame's complete draw list (not the compact TLAS index).
    pub source_draw: u32,
    pub transform: Mat4,
    /// Base colour factor.
    pub albedo: [f32; 3],
    /// Mean albedo-texture colour (used when the texture isn't in the table).
    pub albedo_mean: [f32; 3],
    /// Mean ORM .b, multiplied by metallic for the diffuse GI hit record.
    pub metallic_mean: f32,
    pub emission: [f32; 3],
    pub emission_raw: [f32; 3],
    /// Renderer texture indices: albedo, normal, orm, emission (-1 = none).
    pub textures: [i32; 4],
    pub albedo_st: [f32; 4],
    pub normal_st: [f32; 4],
    pub orm_st: [f32; 4],
    pub emission_st: [f32; 4],
    pub base_alpha: f32,
    /// Negative for opaque/blended materials; nonnegative for alpha testing.
    pub alpha_cutoff: f32,
    pub opacity_texture: i32,
    pub extra_textures: [i32; 4],
    pub detail_textures: [i32; 4],
    pub detail: [f32; 4],
    pub detail_st: [f32; 4],
    pub orm_invert: [f32; 4],
    pub emission_mask_st: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub normal_strength: f32,
    /// Baked lightmap array layer (-1 = none).
    pub lightmap_layer: i32,
    /// Static geometry: its emission and lighting are in the bake (mixed GI skips it).
    pub static_geometry: bool,
    /// Water body slot in `RtUniforms::water_bodies` (-1 = not water). Water
    /// hits are shaded from the body; `textures[0]` is its ripple field.
    pub water: i32,
    /// Version of a mesh whose vertices the CPU rewrites (water): its BLAS is
    /// rebuilt inside the frame's command buffer when this changes (a
    /// displaced water BLAS is refit while the version holds). None = built
    /// once.
    pub dynamic: Option<u32>,
    /// "Cast Shadows": picks the TLAS mask (see `solid_mask`).
    pub shadow_casting: crate::ShadowCasting,
}

/// Light for hit shading, mirrors `Light` in the shader (matches `bake`'s layout).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct RtLight {
    pub position: [f32; 4],  // xyz, w = kind (0 dir / 1 point / 2 spot)
    pub direction: [f32; 4], // xyz, w = range
    pub color: [f32; 4],
    pub spot: [f32; 4], // x cos_inner, y cos_outer, z baked (1 = skip on lightmapped hits)
}

/// Resolve uniforms, mirrors `RtData` in the shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct RtUniforms {
    pub inv_proj: [[f32; 4]; 4],
    pub inv_view: [[f32; 4]; 4],
    pub camera_pos: [f32; 4],
    pub refl_center: [f32; 4],
    pub refl_extents: [f32; 4],
    pub params: [f32; 4], // x rough cutoff, y intensity, z light count, w env IBL intensity (skybox gate)
    pub screen: [f32; 4],
    /// xyz = probe box CENTER (world; capture pos + box offset), w unused.
    /// `refl_center` above is the CAPTURE position (parallax re-vector anchor).
    pub refl_box: [f32; 4],
    /// x rays per pixel, y bounces, z max ray distance, w wrapped sampling frame.
    pub quality: [f32; 4],
    pub lighting_mode: [u32; 4],
    /// Second reflection probe (binding 11) + blend weights, as in the frame UBO.
    pub refl2_center: [f32; 4],
    pub refl2_extents: [f32; 4],
    pub refl2_box: [f32; 4],
    pub refl_blend: [f32; 4],
    /// Third/fourth reflection probes (bindings 12/13), as in the frame UBO.
    pub refl3_center: [f32; 4],
    pub refl3_extents: [f32; 4],
    pub refl3_box: [f32; 4],
    pub refl4_center: [f32; 4],
    pub refl4_extents: [f32; 4],
    pub refl4_box: [f32; 4],
    /// x = slot 2 blend distance, y = slot 3 blend distance.
    pub refl_blend2: [f32; 4],
    /// x water clock, y body count.
    pub water: [f32; 4],
    pub water_bodies: [GpuWater; MAX_WATER_BODIES],
}

/// Per-in-flight-slot small buffers (camera uniforms + lights), recreated each
/// frame and freed when the slot is reused.
/// Per-frame-in-flight lights + uniforms, kept mapped across frames (grown when
/// the light count outgrows `light_cap`).
struct RtFrame {
    lights: Buffer,
    light_cap: usize,
    ubo: Buffer,
}

pub(crate) struct RtReflect {
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sampler: vk::Sampler,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    /// BLAS per mesh index (built once, reused across frames), with the mesh
    /// version it was built from for dynamic meshes.
    blas: HashMap<usize, (Accel, Option<u32>)>,
    /// Per frame slot: BLASes replaced while recording it and their build
    /// scratch, freed once that slot's fence has been waited again.
    retired: Vec<(Vec<Accel>, Vec<Buffer>)>,
    /// Per-view, per-in-flight-slot uniforms/lights. Scene and Camera dispatches
    /// can coexist in a command buffer without rewriting each other's data.
    frames: Vec<Option<RtFrame>>,
    /// One TLAS + instance buffers per frame in flight. A changed scene rebuilds
    /// this frame's slot on the GPU inside the frame command buffer (no CPU wait,
    /// no device idle); the other slot may still be in use by the previous frame.
    tlas_slots: Vec<Option<TlasSlot>>,
    /// Repeat + mipmapped sampler for material textures.
    tex_sampler: vk::Sampler,
    /// Texture-table signature written into each frame's set (rewrite on change).
    tex_sig: Vec<u64>,
    /// Shore-depth table signature written into each frame's set (rewrite on change).
    shore_sig: Vec<u64>,
    /// Displaced water BLASes (per frame slot) while RT reflections are on.
    water: RtWater,
    denoiser: crate::reflection_denoise::Denoiser,
    #[cfg(test)] pub(crate) denoise_override: Option<bool>,
}

impl RtReflect {
    pub fn new(device: &ash::Device) -> Result<Self> {
        let sampler_binding = |b: u32| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        };
        let mut bindings = vec![
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            sampler_binding(1),
            sampler_binding(2),
            sampler_binding(3),
            sampler_binding(4),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(7)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(8)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Baked lightmap array (lightmapped hits read their baked lighting).
            vk::DescriptorSetLayoutBinding::default()
                .binding(10)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Second reflection probe cube (two-probe blend on misses).
            vk::DescriptorSetLayoutBinding::default()
                .binding(11)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Third/fourth reflection probe cubes (3+ probe blend).
            vk::DescriptorSetLayoutBinding::default()
                .binding(12)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(13)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Every renderer texture, indexed per hit (descriptor indexing).
            vk::DescriptorSetLayoutBinding::default()
                .binding(9)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(MAX_RT_TEXTURES)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            // Per-body shore-depth fields (binding 14, partially bound), for
            // the water shading normal's shore calm.
            vk::DescriptorSetLayoutBinding::default()
                .binding(14)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(MAX_WATER_BODIES as u32)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        for binding in 15..24 {
            bindings.push(vk::DescriptorSetLayoutBinding::default().binding(binding)
                .descriptor_type(if binding < 17 { vk::DescriptorType::STORAGE_BUFFER }
                    else { vk::DescriptorType::COMBINED_IMAGE_SAMPLER })
                .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE));
        }
        // Flags are parallel to the binding array, not indexed by binding
        // number (the texture table is deliberately listed after the cubes).
        let binding_flags: Vec<_> = bindings.iter().map(|binding| {
            if matches!(binding.binding, 9 | 14) {
                vk::DescriptorBindingFlags::PARTIALLY_BOUND
            } else {
                vk::DescriptorBindingFlags::empty()
            }
        }).collect();
        let mut flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);
        let set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings).push_next(&mut flags_info),
                None,
            )?
        };
        let layouts = [set_layout];
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                None,
            )?
        };
        let module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default()
                    .code(&ash::util::read_spv(&mut std::io::Cursor::new(SPV))?),
                None,
            )?
        };
        let pipeline = unsafe {
            device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .module(module)
                                .name(c"main"),
                        )
                        .layout(pipeline_layout)],
                    None,
                )
                .map_err(|(_, e)| e)?[0]
        };
        unsafe { device.destroy_shader_module(module, None) };
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?
        };
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets((FRAMES_IN_FLIGHT * REFLECTION_VIEWS) as u32)
                    .pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: (FRAMES_IN_FLIGHT * REFLECTION_VIEWS) as u32,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: (FRAMES_IN_FLIGHT * REFLECTION_VIEWS) as u32 * (15 + MAX_RT_TEXTURES + MAX_WATER_BODIES as u32),
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
                            descriptor_count: (FRAMES_IN_FLIGHT * REFLECTION_VIEWS) as u32,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::STORAGE_BUFFER,
                            descriptor_count: (FRAMES_IN_FLIGHT * REFLECTION_VIEWS * 4) as u32,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: (FRAMES_IN_FLIGHT * REFLECTION_VIEWS) as u32,
                        },
                    ]),
                None,
            )?
        };
        let layouts = vec![set_layout; FRAMES_IN_FLIGHT * REFLECTION_VIEWS];
        let sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )?
        };
        let tex_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::REPEAT)
                    .address_mode_w(vk::SamplerAddressMode::REPEAT)
                    .max_lod(vk::LOD_CLAMP_NONE),
                None,
            )?
        };
        let water = RtWater::new(device)?;
        Ok(Self {
            set_layout,
            pipeline_layout,
            pipeline,
            sampler,
            pool,
            sets,
            water,
            denoiser: crate::reflection_denoise::Denoiser::new(device)?,
            #[cfg(test)] denoise_override: None,
            tlas_slots: (0..FRAMES_IN_FLIGHT).map(|_| None).collect(),
            tex_sampler,
            tex_sig: vec![0; FRAMES_IN_FLIGHT * REFLECTION_VIEWS],
            shore_sig: vec![u64::MAX; FRAMES_IN_FLIGHT * REFLECTION_VIEWS],
            blas: HashMap::new(),
            retired: (0..FRAMES_IN_FLIGHT).map(|_| (Vec::new(), Vec::new())).collect(),
            frames: (0..FRAMES_IN_FLIGHT * REFLECTION_VIEWS).map(|_| None).collect(),
        })
    }

    /// Build/refresh the cached BLAS-per-mesh + this frame's scene TLAS +
    /// per-instance shading SSBO (vertex/index device addresses + materials).
    /// When the scene signature changed, the TLAS is rebuilt on the GPU by
    /// commands recorded into `cb` (followed by a barrier for ray-query reads).
    /// Returns the TLAS handle + instance buffer handle, or None when there's no
    /// ray-query device. An empty scene still has a valid empty TLAS so a
    /// reflection resolve can write its background after the last object is removed.
    /// Shared by the reflection dispatch
    /// and the Flux-RT GI gather: the second call in a frame finds the slot
    /// already current. With `water.animate`, water instances trace this
    /// slot's displaced water BLAS (refit here when the waves moved).
    #[allow(clippy::too_many_arguments)]
    pub fn ensure_tlas(
        &mut self,
        ctx: &GpuContext,
        allocator: &Arc<Mutex<Allocator>>,
        command_pool: vk::CommandPool,
        meshes: &[GpuMesh],
        instances: &[RtInstance],
        water: &RtWaterFrame,
        cb: vk::CommandBuffer,
        frame: usize,
    ) -> Result<Option<(vk::AccelerationStructureKHR, vk::Buffer)>> {
        let Some(accel) = ctx.accel.as_ref() else {
            return Ok(None);
        };
        let device = &ctx.device;
        let scratch_align = bake::accel_scratch_alignment(ctx);

        // BLAS per referenced mesh (cached across frames; built once per mesh).
        // A dynamic mesh's is rebuilt in `cb` when its vertices changed (no CPU
        // wait); the old one is retired until no frame in flight can use it.
        let mut recorded = false;
        let displaced = |inst: &RtInstance| water.animate && inst.water >= 0;
        for inst in instances {
            if displaced(inst) {
                if let Some((old, _)) = self.blas.remove(&inst.mesh) {
                    self.retired[frame].0.push(old);
                }
                continue;
            }
            match inst.dynamic {
                None => {
                    if let std::collections::hash_map::Entry::Vacant(e) = self.blas.entry(inst.mesh) {
                        let blas = bake::build_blas(
                            ctx, accel, allocator, command_pool, &meshes[inst.mesh], scratch_align,
                        )?;
                        e.insert((blas, None));
                    }
                }
                Some(v) => {
                    if self.blas.get(&inst.mesh).is_some_and(|(_, built)| *built == Some(v)) {
                        continue;
                    }
                    let (blas, scratch) = record_blas(ctx, accel, allocator, &meshes[inst.mesh], scratch_align, cb)?;
                    if let Some((old, _)) = self.blas.insert(inst.mesh, (blas, Some(v))) {
                        self.retired[frame].0.push(old);
                    }
                    self.retired[frame].1.push(scratch);
                    recorded = true;
                }
            }
        }
        if water.animate {
            recorded |= self.water.update(ctx, accel, allocator, meshes, instances, water, cb, frame, &mut self.retired[frame])?;
        }
        if recorded {
            let barrier = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR)];
            unsafe { device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&barrier)) };
        }

        let mut tlas_instances = Vec::with_capacity(instances.len());
        let mut gpu_instances = Vec::with_capacity(instances.len());
        let mut sig: u64 = 0xcbf29ce484222325;
        let mut layout: u64 = 0xcbf29ce484222325;
        let fnv = |v: u64, s: &mut u64| {
            *s ^= v;
            *s = s.wrapping_mul(0x100000001b3);
        };
        for inst in instances {
            let (blas_address, water_key) = if displaced(inst) {
                match self.water.blas(inst.mesh, frame) {
                    Some(b) => b,
                    None => continue,
                }
            } else {
                match self.blas.get(&inst.mesh) {
                    Some((blas, _)) => (blas.address, 0),
                    None => continue,
                }
            };
            let mesh = &meshes[inst.mesh];
            let in_table = |t: i32| if t >= 0 && (t as u32) < MAX_RT_TEXTURES { t } else { -1 };
            let textures = inst.textures.map(in_table);
            let albedo = [0, 1, 2].map(|c| inst.albedo[c] * inst.albedo_mean[c]);
            let g = GpuInstance {
                vtx_addr: mesh.vertex_buffer.device_address(device),
                idx_addr: mesh.index_buffer.device_address(device),
                albedo: [albedo[0], albedo[1], albedo[2], (inst.metallic * inst.metallic_mean).clamp(0.0, 1.0)],
                emission: [inst.emission[0], inst.emission[1], inst.emission[2], if inst.static_geometry { 1.0 } else { 0.0 }],
                base: [inst.albedo[0], inst.albedo[1], inst.albedo[2], (inst.water.max(-1) + 1) as f32],
                emission_raw: [inst.emission_raw[0], inst.emission_raw[1], inst.emission_raw[2], (inst.source_draw+1) as f32],
                textures,
                albedo_st: inst.albedo_st,
                normal_st: inst.normal_st,
                orm_st: inst.orm_st,
                emission_st: inst.emission_st,
                params: [inst.metallic, inst.roughness, inst.normal_strength, inst.lightmap_layer as f32],
                opacity: [inst.base_alpha, inst.alpha_cutoff, in_table(inst.opacity_texture) as f32, 0.0],
                extra_textures: inst.extra_textures.map(in_table),
                detail_textures: inst.detail_textures.map(in_table),
                detail: inst.detail,
                detail_st: inst.detail_st,
                orm_invert: inst.orm_invert,
                emission_mask_st: inst.emission_mask_st,
            };
            for w in bytemuck::cast_slice::<GpuInstance, u32>(std::slice::from_ref(&g)).iter().skip(4) {
                fnv(*w as u64, &mut sig);
            }
            // The shader indexes the shading records by custom index, so it
            // must be this record's position (instances without a BLAS are
            // skipped, so it is not the input index).
            let custom = gpu_instances.len() as u32;
            let mask = if inst.water >= 0 { MASK_WATER } else { solid_mask(inst.shadow_casting) };
            // Override opacity only for masked instances, retaining fast opaque BLAS traversal.
            let mut flags=vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE;
            if inst.alpha_cutoff>=0.0 {flags|=vk::GeometryInstanceFlagsKHR::FORCE_NO_OPAQUE;}
            gpu_instances.push(g);
            tlas_instances.push(vk::AccelerationStructureInstanceKHR {
                transform: bake::transform_matrix(&inst.transform),
                instance_custom_index_and_mask: vk::Packed24_8::new(custom, mask),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    0,
                    flags.as_raw() as u8,
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: blas_address,
                },
            });
            fnv(inst.mesh as u64, &mut sig);
            fnv(blas_address, &mut sig);
            fnv(mask as u64, &mut sig);
            fnv(mask as u64, &mut layout);
            fnv(flags.as_raw() as u64, &mut sig);
            fnv(flags.as_raw() as u64, &mut layout);
            // A refit water BLAS needs the TLAS refit over its new bounds.
            fnv(water_key, &mut sig);
            fnv(blas_address, &mut layout);
            for f in inst.transform.to_cols_array() {
                fnv(f.to_bits() as u64, &mut sig);
            }
        }

        let n = tlas_instances.len();
        if let Some(slot) = self.tlas_slots[frame].as_ref()
            && slot.sig == sig
        {
            return Ok(Some((slot.accel.handle, slot.shading.handle)));
        }

        // (Re)allocate this frame's slot when it is missing or too small. This
        // frame's fence has been waited, so nothing on the GPU still uses it.
        let geometry = |addr: u64| {
            vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::INSTANCES)
                .geometry(vk::AccelerationStructureGeometryDataKHR {
                    instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                        .data(vk::DeviceOrHostAddressConstKHR { device_address: addr }),
                })
                .flags(vk::GeometryFlagsKHR::OPAQUE)
        };
        if self.tlas_slots[frame].as_ref().is_none_or(|s| s.capacity < n) {
            let capacity = n.next_power_of_two().max(16);
            let mut a = allocator.lock().unwrap();
            if let Some(old) = self.tlas_slots[frame].take() {
                old.destroy(ctx, &mut a);
            }
            let inst_input = Buffer::new(
                device,
                &mut a,
                (capacity * std::mem::size_of::<vk::AccelerationStructureInstanceKHR>()) as u64,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                gpu_allocator::MemoryLocation::CpuToGpu,
                "tlas-instances",
            )?;
            let shading = Buffer::new(
                device,
                &mut a,
                (capacity * std::mem::size_of::<GpuInstance>()) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::CpuToGpu,
                "rt-instances",
            )?;
            let geoms = [geometry(inst_input.device_address(device))];
            let info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
                .flags(TLAS_FLAGS)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .geometries(&geoms);
            let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
            unsafe {
                accel.get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &info,
                    &[capacity as u32],
                    &mut sizes,
                )
            };
            let as_buffer = Buffer::new(
                device,
                &mut a,
                sizes.acceleration_structure_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                gpu_allocator::MemoryLocation::GpuOnly,
                "tlas",
            )?;
            let handle = unsafe {
                accel.create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR::default()
                        .buffer(as_buffer.handle)
                        .size(sizes.acceleration_structure_size)
                        .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL),
                    None,
                )?
            };
            let scratch_size = sizes.build_scratch_size.max(sizes.update_scratch_size);
            let (scratch, scratch_addr) = bake::scratch_buffer(device, &mut a, scratch_size, scratch_align)?;
            let address = unsafe {
                accel.get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(handle),
                )
            };
            self.tlas_slots[frame] = Some(TlasSlot {
                accel: Accel { handle, buffer: as_buffer, address },
                inst_input,
                shading,
                scratch,
                scratch_addr,
                capacity,
                sig: 0,
                layout: 0,
                refits: 0,
            });
        }

        let slot = self.tlas_slots[frame].as_mut().unwrap();
        let inst_bytes = unsafe {
            std::slice::from_raw_parts(
                tlas_instances.as_ptr() as *const u8,
                std::mem::size_of_val(tlas_instances.as_slice()),
            )
        };
        slot.inst_input.write(0, inst_bytes);
        slot.shading.write(0, bytemuck::cast_slice(&gpu_instances));
        let geoms = [geometry(slot.inst_input.device_address(device))];
        // Same instances and BLASes, only transforms moved: refit in place.
        let refit = slot.sig != 0 && slot.layout == layout && slot.refits < TLAS_MAX_REFITS;
        let mut info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(TLAS_FLAGS)
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geoms)
            .dst_acceleration_structure(slot.accel.handle)
            .scratch_data(vk::DeviceOrHostAddressKHR { device_address: slot.scratch_addr });
        if refit {
            info = info
                .mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
                .src_acceleration_structure(slot.accel.handle);
        }
        let ranges = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(n as u32)];
        let barrier = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
            .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR)];
        unsafe {
            accel.cmd_build_acceleration_structures(cb, &[info], &[&ranges]);
            device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&barrier));
        }
        slot.sig = sig;
        slot.layout = layout;
        slot.refits = if refit { slot.refits + 1 } else { 0 };
        Ok(Some((slot.accel.handle, slot.shading.handle)))
    }

    /// Build/refresh the acceleration structures + buffers for `frame` and record
    /// the reflection compute dispatch, writing the composited result into
    /// `out_view` (a STORAGE-usable RGBA16F image, left in GENERAL). The caller
    /// barriers `out_image` to SHADER_READ afterward for the post pass.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        ctx: &GpuContext,
        allocator: &Arc<Mutex<Allocator>>,
        command_pool: vk::CommandPool,
        cb: vk::CommandBuffer,
        frame: usize,
        view: usize,
        serial: u64,
        meshes: &[GpuMesh],
        instances: &[RtInstance],
        water: &RtWaterFrame,
        lights: &[RtLight],
        out_image: vk::Image,
        out_view: vk::ImageView,
        color_view: vk::ImageView,
        depth_view: vk::ImageView,
        gbuf_view: vk::ImageView,
        env_view: vk::ImageView,
        env_b_view: vk::ImageView,
        env_c_view: vk::ImageView,
        env_d_view: vk::ImageView,
        textures: &[vk::ImageView],
        lightmap: (vk::ImageView, vk::Sampler),
        area_atlas: (vk::ImageView, vk::Sampler),
        surface: Option<crate::flux_surface::Resources>,
        uniforms: &RtUniforms,
        extent: vk::Extent2D,
    ) -> Result<()> {
        let Some(_accel) = ctx.accel.as_ref() else {
            return Ok(());
        };
        let device = &ctx.device;
        assert!(view < REFLECTION_VIEWS);
        let slot = view * FRAMES_IN_FLIGHT + frame;

        // BLAS + TLAS + instance SSBO (shared cache via ensure_tlas).
        let Some((tlas_handle, inst_handle)) =
            self.ensure_tlas(ctx, allocator, command_pool, meshes, instances, water, cb, frame)?
        else {
            return Ok(());
        };

        let (_raw_image,raw_view)={let mut alloc=allocator.lock().unwrap();
            self.denoiser.prepare(device,&mut alloc,cb,view,serial,extent)?};
        // This slot's lights + camera uniforms (its GPU work finished when the
        // renderer waited this frame's fence, so rewriting in place is safe).
        let n_lights = lights.len().max(1);
        if self.frames[slot].as_ref().is_none_or(|f| f.light_cap < n_lights) {
            let mut a = allocator.lock().unwrap();
            if let Some(mut old) = self.frames[slot].take() {
                old.lights.destroy(device, &mut a);
                old.ubo.destroy(device, &mut a);
            }
            let light_cap = n_lights.next_power_of_two().max(8);
            let lights_buf = host_buffer(
                device,
                &mut a,
                &vec![0u8; light_cap * std::mem::size_of::<RtLight>()],
                "rt-lights",
            )?;
            let ubo = Buffer::new(
                device,
                &mut a,
                std::mem::size_of::<RtUniforms>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                gpu_allocator::MemoryLocation::CpuToGpu,
                "rt-uniforms",
            )?;
            self.frames[slot] = Some(RtFrame { lights: lights_buf, light_cap, ubo });
        }
        let f = self.frames[slot].as_mut().unwrap();
        if lights.is_empty() {
            f.lights.write(0, bytemuck::bytes_of(&RtLight::zeroed()));
        } else {
            f.lights.write(0, bytemuck::cast_slice(lights));
        }
        let mut uniforms = *uniforms;
        uniforms.screen[2] = surface.map_or(0, |s| s.node_count) as f32;
        f.ubo.write(0, bytemuck::bytes_of(&uniforms));
        let (light_handle, ubo_handle) = (f.lights.handle, f.ubo.handle);

        // Write the descriptor set for this frame.
        let set = self.sets[slot];
        let out_info = [vk::DescriptorImageInfo::default()
            .image_view(raw_view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let img = |v: vk::ImageView| {
            [vk::DescriptorImageInfo::default()
                .sampler(self.sampler)
                .image_view(v)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
        };
        let color_i = img(color_view);
        let depth_i = img(depth_view);
        let gbuf_i = img(gbuf_view);
        let env_i = img(env_view);
        let env_b_i = img(env_b_view);
        let env_c_i = img(env_c_view);
        let env_d_i = img(env_d_view);
        let lm_i = [vk::DescriptorImageInfo::default()
            .sampler(lightmap.1)
            .image_view(lightmap.0)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let tlas_handles = [tlas_handle];
        let mut tlas_write = vk::WriteDescriptorSetAccelerationStructureKHR::default()
            .acceleration_structures(&tlas_handles);
        let inst_info = [vk::DescriptorBufferInfo::default().buffer(inst_handle).range(vk::WHOLE_SIZE)];
        let light_info = [vk::DescriptorBufferInfo::default().buffer(light_handle).range(vk::WHOLE_SIZE)];
        let ubo_info = [vk::DescriptorBufferInfo::default().buffer(ubo_handle).range(vk::WHOLE_SIZE)];
        let w = |b: u32, ty: vk::DescriptorType| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(b).descriptor_type(ty);
        let mut as_w = w(5, vk::DescriptorType::ACCELERATION_STRUCTURE_KHR);
        as_w = as_w.push_next(&mut tlas_write);
        as_w.descriptor_count = 1;
        let area_i=[vk::DescriptorImageInfo::default().image_view(area_atlas.0).sampler(area_atlas.1)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let writes = [
            w(23,vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&area_i),
            w(0, vk::DescriptorType::STORAGE_IMAGE).image_info(&out_info),
            w(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&color_i),
            w(2, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&depth_i),
            w(3, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&gbuf_i),
            w(4, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&env_i),
            as_w,
            w(6, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&inst_info),
            w(7, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&light_info),
            w(8, vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&ubo_info),
            w(10, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&lm_i),
            w(11, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&env_b_i),
            w(12, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&env_c_i),
            w(13, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&env_d_i),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        // Always overwrite these descriptors, including when GI is disabled or
        // its atlas was retired. The zero node count prevents fallback reads;
        // valid typed placeholders also satisfy static descriptor validation.
        let card_info = [vk::DescriptorBufferInfo::default()
            .buffer(surface.map_or(inst_handle, |s| s.cards)).range(vk::WHOLE_SIZE)];
        let node_info = [vk::DescriptorBufferInfo::default()
            .buffer(surface.map_or(inst_handle, |s| s.nodes)).range(vk::WHOLE_SIZE)];
        let surface_views = surface.map_or([color_view; 6], |s|
            [s.views[0], s.views[1], s.views[2], s.views[3], s.lighting[0], s.lighting[1]]);
        let surface_infos = surface_views.map(img);
        let mut surface_writes = vec![
            w(15, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&card_info),
            w(16, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&node_info),
        ];
        for (i, info) in surface_infos.iter().enumerate() {
            surface_writes.push(w(17+i as u32, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(info));
        }
        unsafe { device.update_descriptor_sets(&surface_writes, &[]) };

        // Texture table: rewritten only when the renderer's texture list changed
        // (the views are part of the signature, so a reload is caught too).
        let n = textures.len().min(MAX_RT_TEXTURES as usize);
        let mut tsig: u64 = 0xcbf29ce484222325 ^ n as u64;
        for v in &textures[..n] {
            tsig ^= vk::Handle::as_raw(*v);
            tsig = tsig.wrapping_mul(0x100000001b3);
        }
        if n > 0 && self.tex_sig[slot] != tsig {
            let infos: Vec<vk::DescriptorImageInfo> = textures[..n]
                .iter()
                .map(|v| {
                    vk::DescriptorImageInfo::default()
                        .sampler(self.tex_sampler)
                        .image_view(*v)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                })
                .collect();
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(9)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&infos);
            unsafe { device.update_descriptor_sets(&[write], &[]) };
            self.tex_sig[slot] = tsig;
        }

        // Shore-depth table (binding 14): rewritten only when the water bodies'
        // shore textures change. PARTIALLY_BOUND permits unused elements to be
        // left unwritten; it does not permit writing null image views.
        let mut hsig: u64 = 0xcbf29ce484222325;
        for v in water.shores {
            hsig ^= vk::Handle::as_raw(v);
            hsig = hsig.wrapping_mul(0x100000001b3);
        }
        if self.shore_sig[slot] != hsig {
            let infos: Vec<_> = water.shores.iter().map(|&view| [
                vk::DescriptorImageInfo::default().sampler(self.sampler)
                    .image_view(view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            ]).collect();
            let writes: Vec<_> = infos.iter().enumerate()
                .filter(|(_, info)| info[0].image_view != vk::ImageView::null())
                .map(|(i, info)| vk::WriteDescriptorSet::default()
                    .dst_set(set).dst_binding(14).dst_array_element(i as u32)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(info))
                .collect();
            unsafe { device.update_descriptor_sets(&writes, &[]) };
            self.shore_sig[slot] = hsig;
        }

        unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[set],
                &[],
            );
            device.cmd_dispatch(cb, extent.width.div_ceil(8), extent.height.div_ceil(8), 1);
        }
        crate::flux_surface::finish_feedback(device,cb,surface);
        let enabled=true;
        #[cfg(test)] let enabled=self.denoise_override.unwrap_or(enabled);
        self.denoiser.record(device,cb,view,frame,serial,color_view,depth_view,gbuf_view,out_image,out_view,&uniforms,enabled);

        Ok(())
    }

    #[cfg(test)]
    pub fn reflection_history(&self,view:usize)->(vk::Image,vk::Image,vk::Extent2D,Mat4,u64) {self.denoiser.debug_history(view)}
    pub fn invalidate_reflection_history(&mut self) {self.denoiser.invalidate();}

    /// Free what was retired while `frame` was last recorded; call once that
    /// slot's fence has been waited.
    pub fn begin_frame(&mut self, ctx: &GpuContext, allocator: &mut Allocator, frame: usize) {
        self.water.begin_frame(ctx, allocator);
        let (blas, scratch) = &mut self.retired[frame];
        for a in blas.drain(..) {
            destroy_accel(ctx, allocator, a);
        }
        for mut b in scratch.drain(..) {
            b.destroy(&ctx.device, allocator);
        }
    }

    pub fn destroy(&mut self, ctx: &GpuContext, allocator: &mut Allocator) {
        for frame in 0..self.retired.len() {
            self.begin_frame(ctx, allocator, frame);
        }
        self.water.destroy(ctx, allocator);
        let device = &ctx.device;
        self.denoiser.destroy(device,allocator);
        for slot in self.frames.iter_mut() {
            if let Some(mut f) = slot.take() {
                f.lights.destroy(device, allocator);
                f.ubo.destroy(device, allocator);
            }
        }
        for slot in self.tlas_slots.drain(..).flatten() {
            slot.destroy(ctx, allocator);
        }
        for (_, (a, _)) in self.blas.drain() {
            destroy_accel(ctx, allocator, a);
        }
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_sampler(self.tex_sampler, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}

fn destroy_accel(ctx: &GpuContext, allocator: &mut Allocator, mut a: Accel) {
    if let Some(accel) = ctx.accel.as_ref() {
        unsafe { accel.destroy_acceleration_structure(a.handle, None) };
    }
    a.buffer.destroy(&ctx.device, allocator);
}

/// Record a BLAS build for `mesh` into `cb` (as `bake::build_blas`, without
/// its submit + wait). Returns the BLAS and the scratch buffer the build uses,
/// which must outlive the command buffer.
fn record_blas(
    ctx: &GpuContext,
    accel: &ash::khr::acceleration_structure::Device,
    allocator: &Arc<Mutex<Allocator>>,
    mesh: &GpuMesh,
    scratch_align: u64,
    cb: vk::CommandBuffer,
) -> Result<(Accel, Buffer)> {
    let device = &ctx.device;
    let prim_count = mesh.index_count / 3;
    let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
        .vertex_format(vk::Format::R32G32B32_SFLOAT)
        .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: mesh.vertex_buffer.device_address(device) })
        .vertex_stride(std::mem::size_of::<crate::Vertex>() as u64)
        .max_vertex(mesh.vertex_count.saturating_sub(1))
        .index_type(vk::IndexType::UINT32)
        .index_data(vk::DeviceOrHostAddressConstKHR { device_address: mesh.index_buffer.device_address(device) });
    let geometries = [vk::AccelerationStructureGeometryKHR::default()
        .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
        .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
        .flags(vk::GeometryFlagsKHR::OPAQUE)];
    let mut info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
        .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&geometries);
    let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
    unsafe {
        accel.get_acceleration_structure_build_sizes(
            vk::AccelerationStructureBuildTypeKHR::DEVICE,
            &info,
            &[prim_count],
            &mut sizes,
        )
    };
    let mut a = allocator.lock().unwrap();
    let as_buffer = Buffer::new(
        device,
        &mut a,
        sizes.acceleration_structure_size,
        vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        gpu_allocator::MemoryLocation::GpuOnly,
        "blas (dynamic)",
    )?;
    let handle = unsafe {
        accel.create_acceleration_structure(
            &vk::AccelerationStructureCreateInfoKHR::default()
                .buffer(as_buffer.handle)
                .size(sizes.acceleration_structure_size)
                .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL),
            None,
        )?
    };
    let (scratch, scratch_addr) = bake::scratch_buffer(device, &mut a, sizes.build_scratch_size, scratch_align)?;
    drop(a);
    info = info
        .dst_acceleration_structure(handle)
        .scratch_data(vk::DeviceOrHostAddressKHR { device_address: scratch_addr });
    let ranges = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(prim_count)];
    unsafe { accel.cmd_build_acceleration_structures(cb, &[info], &[&ranges]) };
    let address = unsafe {
        accel.get_acceleration_structure_device_address(
            &vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(handle),
        )
    };
    Ok((Accel { handle, buffer: as_buffer, address }, scratch))
}

/// A host-visible SSBO seeded with `data`.
fn host_buffer(device: &ash::Device, alloc: &mut Allocator, data: &[u8], name: &str) -> Result<Buffer> {
    let mut buf = Buffer::new(
        device,
        alloc,
        data.len().max(16) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        gpu_allocator::MemoryLocation::CpuToGpu,
        name,
    )
    .with_context(|| format!("creating {name}"))?;
    buf.write(0, data);
    Ok(buf)
}
