//! GPU side of the octahedral impostors (GEO-5): the view atlas as an array
//! image (one layer per view, so filtering never crosses a view border), one
//! camera-facing billboard pipeline per scene-pass variant, and the draw
//! recording. The data comes from `zestengine_assets::impostor` (a capture of
//! the model's bind pose); this module knows only the raw atlas layout, so the
//! render crate does not depend on the assets crate.
//!
//! Like the text system it draws straight into the scene pass (depth tested,
//! depth written, multiview aware): no render target of its own. An impostor
//! of an animated model is the model's REST pose turned to face the camera: it
//! looks wrong in motion by design and is only for the fallback chain (full
//! cut, coarser discrete LOD, impostor, rank-tinted capsule).

use std::collections::HashMap;

use anyhow::{Context as _, Result, bail};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

use crate::alloc::{Buffer, one_time_submit};
use crate::frame::FRAMES_IN_FLIGHT;
use crate::pipeline::PassFormats;

const VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/impostor.vert.spv"));
const VERT_MV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/impostor.vert.mv.spv"));
const FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/impostor.frag.spv"));

/// Largest array the upload accepts: `2 * grid * grid` layers, within the
/// 256 layers every Vulkan device offers (an 11 x 11 grid).
pub const MAX_LAYERS: u32 = 256;
/// Atlases alive at once (each has a descriptor set).
pub const MAX_ATLASES: usize = 64;
/// Billboards one frame can hold.
pub const INSTANCE_CAP: usize = 1024;

/// A captured impostor as raw data (`zestengine_assets::impostor::Impostor`).
#[derive(Clone, Copy, Debug)]
pub struct ImpostorUpload<'a> {
    /// Views per side.
    pub grid: u32,
    /// Texels per view side.
    pub tile: u32,
    /// Centre and radius of the bounding sphere every view frames (model space).
    pub centre: [f32; 3],
    pub radius: f32,
    /// `(grid * tile)^2` RGBA8 texels, views laid out row by row: albedo (rgb colour, a coverage).
    pub albedo: &'a [u8],
    /// Same layout: aux (rg octahedral normal, b depth, a coverage).
    pub aux: &'a [u8],
}

impl ImpostorUpload<'_> {
    pub fn validate(&self) -> Result<(), String> {
        if self.grid == 0 || self.tile == 0 || 2 * self.grid * self.grid > MAX_LAYERS {
            return Err(format!("impostor grid {} (tile {}) needs {} layers, at most {MAX_LAYERS}", self.grid, self.tile, 2 * self.grid * self.grid));
        }
        let size = (self.grid * self.tile) as usize;
        let want = size * size * 4;
        if self.albedo.len() != want || self.aux.len() != want {
            return Err(format!("impostor atlas is {} / {} bytes, expected {want}", self.albedo.len(), self.aux.len()));
        }
        if !(self.radius.is_finite() && self.radius > 0.0 && self.centre.iter().all(|c| c.is_finite())) {
            return Err("impostor bounds are not finite".into());
        }
        Ok(())
    }
}

/// Handle of an uploaded impostor atlas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImpostorHandle(pub(crate) usize);

impl ImpostorHandle {
    pub fn index(self) -> usize {
        self.0
    }
}

/// Re-lay an atlas (`grid * tile` square, views row by row) as one tile after
/// another: layer `cell = j * grid + i` holds view `(i, j)`, `tile * tile * 4`
/// bytes each. The layout the array image is filled with.
pub fn atlas_to_layers(grid: u32, tile: u32, atlas: &[u8]) -> Vec<u8> {
    let (g, t) = (grid as usize, tile as usize);
    let size = g * t;
    let mut out = vec![0u8; g * g * t * t * 4];
    for j in 0..g {
        for i in 0..g {
            let cell = j * g + i;
            for row in 0..t {
                let src = (((j * t + row) * size) + i * t) * 4;
                let dst = ((cell * t + row) * t) * 4;
                out[dst..dst + t * 4].copy_from_slice(&atlas[src..src + t * 4]);
            }
        }
    }
    out
}

/// One billboard (std430 `Inst` in impostor.vert). 96 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub(crate) struct InstanceRec {
    pub model: [[f32; 4]; 4],
    pub centre_radius: [f32; 4],
    pub tint: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Push {
    first: u32,
    grid: u32,
    pad: [u32; 2],
}

const PUSH_SIZE: u32 = std::mem::size_of::<Push>() as u32;

pub(crate) struct AtlasGpu {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<Allocation>,
    set: vk::DescriptorSet,
    grid: u32,
    centre: [f32; 3],
    radius: f32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PipeKey {
    color: vk::Format,
    depth: vk::Format,
    gbuf: bool,
    gbuf_format: vk::Format,
    velocity: bool,
    multiview: bool,
    samples: u8,
}

/// A billboard queued for this frame.
#[derive(Clone, Copy)]
struct Queued {
    atlas: usize,
    model: Mat4,
    tint: [f32; 4],
}

pub(crate) struct ImpostorGpu {
    sampler: vk::Sampler,
    inst_layout: vk::DescriptorSetLayout,
    atlas_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    inst_sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT],
    inst: Vec<Buffer>,
    pub(crate) atlases: Vec<Option<AtlasGpu>>,
    layout: vk::PipelineLayout,
    modules: [vk::ShaderModule; 3],
    pipelines: HashMap<PipeKey, vk::Pipeline>,
    queued: Vec<Queued>,
    /// `(atlas, first instance, count)` runs of this frame's upload.
    runs: Vec<(usize, u32, u32)>,
    records: Vec<InstanceRec>,
    warned_overflow: bool,
}

impl ImpostorGpu {
    pub(crate) fn new(device: &ash::Device, allocator: &mut Allocator, frame_set_layout: vk::DescriptorSetLayout) -> Result<Self> {
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?
        };
        // Set 1: the billboards (binding 2, as the text system's set).
        let inst_binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)];
        let inst_layout =
            unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&inst_binding), None)? };
        // Set 2: one atlas.
        let atlas_binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let atlas_layout =
            unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&atlas_binding), None)? };
        // Twice the atlas cap: an atlas freed on the deferred queue holds its
        // set for a few frames while its replacement is already uploaded.
        let sizes = [
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(FRAMES_IN_FLIGHT as u32),
            vk::DescriptorPoolSize::default().ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(2 * MAX_ATLASES as u32),
        ];
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                    .max_sets((FRAMES_IN_FLIGHT + 2 * MAX_ATLASES) as u32)
                    .pool_sizes(&sizes),
                None,
            )?
        };
        let layouts = [inst_layout; FRAMES_IN_FLIGHT];
        let sets = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))? };
        let mut inst_sets = [vk::DescriptorSet::null(); FRAMES_IN_FLIGHT];
        inst_sets.copy_from_slice(&sets);
        let mut inst = Vec::new();
        for (i, set) in inst_sets.iter().enumerate() {
            let buffer = Buffer::new(
                device,
                allocator,
                (INSTANCE_CAP * std::mem::size_of::<InstanceRec>()) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::CpuToGpu,
                "impostor instances",
            )?;
            let info = [vk::DescriptorBufferInfo::default().buffer(buffer.handle).range(vk::WHOLE_SIZE)];
            unsafe {
                device.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(*set)
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&info)],
                    &[],
                )
            };
            let _ = i;
            inst.push(buffer);
        }
        let push_range = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT).offset(0).size(PUSH_SIZE)];
        let set_layouts = [frame_set_layout, inst_layout, atlas_layout];
        let layout = unsafe {
            device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts).push_constant_ranges(&push_range), None)?
        };
        let module = |spv: &[u8]| -> Result<vk::ShaderModule> {
            let code = ash::util::read_spv(&mut std::io::Cursor::new(spv))?;
            Ok(unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)? })
        };
        Ok(Self {
            sampler,
            inst_layout,
            atlas_layout,
            pool,
            inst_sets,
            inst,
            atlases: Vec::new(),
            layout,
            modules: [module(VERT)?, module(VERT_MV)?, module(FRAG)?],
            pipelines: HashMap::new(),
            queued: Vec::new(),
            runs: Vec::new(),
            records: Vec::new(),
            warned_overflow: false,
        })
    }

    pub(crate) fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            for (_, p) in self.pipelines.drain() {
                device.destroy_pipeline(p, None);
            }
            for m in self.modules {
                device.destroy_shader_module(m, None);
            }
            device.destroy_pipeline_layout(self.layout, None);
            for slot in &mut self.atlases {
                if let Some(mut a) = slot.take() {
                    free_atlas(device, allocator, self.pool, &mut a);
                }
            }
            for b in &mut self.inst {
                b.destroy(device, allocator);
            }
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.inst_layout, None);
            device.destroy_descriptor_set_layout(self.atlas_layout, None);
            device.destroy_sampler(self.sampler, None);
        }
    }

    /// Upload an atlas as an array image of `2 * grid^2` layers (albedo views, then aux views).
    pub(crate) fn upload(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        pool: vk::CommandPool,
        queue: vk::Queue,
        data: &ImpostorUpload,
    ) -> Result<ImpostorHandle> {
        data.validate().map_err(anyhow::Error::msg)?;
        let live = self.atlases.iter().flatten().count();
        if live >= MAX_ATLASES {
            bail!("{MAX_ATLASES} impostor atlases are already uploaded");
        }
        let mut texels = atlas_to_layers(data.grid, data.tile, data.albedo);
        texels.extend_from_slice(&atlas_to_layers(data.grid, data.tile, data.aux));
        let layers = 2 * data.grid * data.grid;
        let (image, view, allocation) = create_layered_image(device, allocator, pool, queue, data.tile, layers, &texels)?;
        let layouts = [self.atlas_layout];
        let set = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.pool).set_layouts(&layouts))? }[0];
        let info = [vk::DescriptorImageInfo::default().sampler(self.sampler).image_view(view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        unsafe {
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&info)],
                &[],
            )
        };
        let atlas = AtlasGpu { image, view, allocation: Some(allocation), set, grid: data.grid, centre: data.centre, radius: data.radius };
        let handle = match self.atlases.iter().position(Option::is_none) {
            Some(free) => {
                self.atlases[free] = Some(atlas);
                free
            }
            None => {
                self.atlases.push(Some(atlas));
                self.atlases.len() - 1
            }
        };
        Ok(ImpostorHandle(handle))
    }

    /// Detach an atlas from the table (it is no longer drawn) and hand it back
    /// for a deferred [`free_atlas`]: submitted frames may still sample it.
    pub(crate) fn take(&mut self, handle: ImpostorHandle) -> Option<AtlasGpu> {
        let a = self.atlases.get_mut(handle.0).and_then(Option::take)?;
        self.queued.retain(|q| q.atlas != handle.0);
        Some(a)
    }

    /// Free an atlas now. The caller guarantees no submitted frame still uses it
    /// (headless tests; the renderer defers through [`Self::take`]).
    #[cfg(test)]
    pub(crate) fn free(&mut self, device: &ash::Device, allocator: &mut Allocator, handle: ImpostorHandle) {
        if let Some(mut a) = self.take(handle) {
            unsafe { free_atlas(device, allocator, self.pool, &mut a) };
        }
    }

    /// Detach every atlas (scene reset), in handle order.
    pub(crate) fn take_all(&mut self) -> Vec<AtlasGpu> {
        self.queued.clear();
        self.atlases.iter_mut().filter_map(Option::take).collect()
    }

    pub(crate) fn pool(&self) -> vk::DescriptorPool {
        self.pool
    }

    /// Atlases currently alive.
    pub(crate) fn live(&self) -> usize {
        self.atlases.iter().flatten().count()
    }

    pub(crate) fn queue(&mut self, handle: ImpostorHandle, model: Mat4, tint: [f32; 4]) {
        if self.atlases.get(handle.0).is_some_and(Option::is_some) {
            self.queued.push(Queued { atlas: handle.0, model, tint });
        }
    }

    /// Frame start / end: this frame's queued billboards are consumed.
    pub(crate) fn end_frame(&mut self) {
        self.queued.clear();
    }

    fn pipeline(&mut self, device: &ash::Device, key: PipeKey) -> Result<vk::Pipeline> {
        if let Some(p) = self.pipelines.get(&key) {
            return Ok(*p);
        }
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(if key.multiview { self.modules[1] } else { self.modules[0] })
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(self.modules[2]).name(c"main"),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let samples = match key.samples {
            4 => vk::SampleCountFlags::TYPE_4,
            2 => vk::SampleCountFlags::TYPE_2,
            _ => vk::SampleCountFlags::TYPE_1,
        };
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(samples);
        // Opaque, alpha-tested in the shader: tests and writes depth like a mesh.
        let depth = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
        let colour = vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA);
        let untouched = vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::empty());
        let mut blends = vec![colour];
        let mut formats = vec![key.color];
        if key.gbuf {
            blends.push(untouched);
            formats.push(key.gbuf_format);
        }
        if key.velocity {
            blends.push(untouched);
            blends.push(untouched);
            formats.push(crate::pipeline::VELOCITY_FORMAT);
            formats.push(crate::pipeline::REACTIVE_FORMAT);
        }
        let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blends);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats).depth_attachment_format(key.depth);
        if crate::sprite_gpu::has_stencil(key.depth) {
            rendering = rendering.stencil_attachment_format(key.depth);
        }
        if key.multiview {
            rendering = rendering.view_mask(0b11);
        }
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport)
            .rasterization_state(&raster)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(self.layout)
            .push_next(&mut rendering);
        let pipeline = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None).map_err(|(_, e)| e) }
            .context("creating impostor pipeline")?[0];
        self.pipelines.insert(key, pipeline);
        Ok(pipeline)
    }

    /// Create the pipeline of a pass variant now (tests; the first draw would).
    #[cfg(test)]
    pub(crate) fn ensure_pipeline(&mut self, device: &ash::Device, pass: PassFormats, gbuf: bool, velocity: bool) -> Result<vk::Pipeline> {
        self.pipeline(device, key_of(pass, gbuf, velocity))
    }

    /// Record this frame's billboards into the scene pass being recorded:
    /// upload the instances (sorted by atlas), then one instanced draw per atlas.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        frame: usize,
        frame_set: vk::DescriptorSet,
        pass: PassFormats,
        gbuf: bool,
        velocity: bool,
    ) {
        if self.queued.is_empty() {
            return;
        }
        if self.queued.len() > INSTANCE_CAP && !self.warned_overflow {
            self.warned_overflow = true;
            tracing::warn!("more than {INSTANCE_CAP} impostors in one frame: the rest are not drawn");
        }
        self.queued.sort_by_key(|q| q.atlas);
        self.records.clear();
        self.runs.clear();
        for q in self.queued.iter().take(INSTANCE_CAP) {
            let Some(Some(a)) = self.atlases.get(q.atlas) else { continue };
            if self.runs.last().is_none_or(|r| r.0 != q.atlas) {
                self.runs.push((q.atlas, self.records.len() as u32, 0));
            }
            self.runs.last_mut().unwrap().2 += 1;
            self.records.push(InstanceRec {
                model: q.model.to_cols_array_2d(),
                centre_radius: [a.centre[0], a.centre[1], a.centre[2], a.radius],
                tint: q.tint,
            });
        }
        if self.records.is_empty() {
            return;
        }
        self.inst[frame].write(0, bytemuck::cast_slice(&self.records));
        let pipeline = match self.pipeline(device, key_of(pass, gbuf, velocity)) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("impostor pipeline: {e:#}");
                return;
            }
        };
        unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[frame_set, self.inst_sets[frame]], &[]);
        }
        for &(atlas, first, count) in &self.runs {
            let Some(Some(a)) = self.atlases.get(atlas) else { continue };
            let push = Push { first, grid: a.grid, pad: [0; 2] };
            unsafe {
                device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, self.layout, 2, &[a.set], &[]);
                device.cmd_push_constants(cb, self.layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, bytemuck::bytes_of(&push));
                device.cmd_draw(cb, 6, count, 0, 0);
            }
        }
    }

    /// The array image of an atlas and its layer count (tests read it back).
    #[cfg(test)]
    pub(crate) fn image_of(&self, handle: ImpostorHandle) -> Option<(vk::Image, u32)> {
        self.atlases.get(handle.0)?.as_ref().map(|a| (a.image, 2 * a.grid * a.grid))
    }
}

fn key_of(pass: PassFormats, gbuf: bool, velocity: bool) -> PipeKey {
    PipeKey { color: pass.color, depth: pass.depth, gbuf, gbuf_format: pass.gbuf, velocity, multiview: pass.multiview, samples: pass.samples }
}

pub(crate) unsafe fn free_atlas(device: &ash::Device, allocator: &mut Allocator, pool: vk::DescriptorPool, a: &mut AtlasGpu) {
    unsafe {
        if a.set != vk::DescriptorSet::null() {
            let _ = device.free_descriptor_sets(pool, &[a.set]);
            a.set = vk::DescriptorSet::null();
        }
        device.destroy_image_view(a.view, None);
        device.destroy_image(a.image, None);
    }
    if let Some(alloc) = a.allocation.take() {
        let _ = allocator.free(alloc);
    }
}

/// An RGBA8 array image of `layers` square layers of `tile` texels, filled
/// from `texels` (layer after layer), left in SHADER_READ_ONLY_OPTIMAL.
pub(crate) fn create_layered_image(
    device: &ash::Device,
    allocator: &mut Allocator,
    pool: vk::CommandPool,
    queue: vk::Queue,
    tile: u32,
    layers: u32,
    texels: &[u8],
) -> Result<(vk::Image, vk::ImageView, Allocation)> {
    let format = vk::Format::R8G8B8A8_UNORM;
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D { width: tile, height: tile, depth: 1 })
        .mip_levels(1)
        .array_layers(layers)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.create_image(&info, None)? };
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let allocation = allocator
        .allocate(&AllocationCreateDesc {
            name: "impostor atlas",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .context("allocating the impostor atlas")?;
    unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };
    let view = unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(layers)),
            None,
        )?
    };
    let mut staging = Buffer::new(device, allocator, texels.len() as u64, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu, "impostor staging")?;
    staging.write(0, texels);
    let result = one_time_submit(device, pool, queue, |cb| unsafe {
        crate::image_barrier_layers(
            device, cb, image, vk::ImageAspectFlags::COLOR, 0, layers,
            vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE,
        );
        let region = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(layers))
            .image_extent(vk::Extent3D { width: tile, height: tile, depth: 1 });
        device.cmd_copy_buffer_to_image(cb, staging.handle, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region]);
        crate::image_barrier_layers(
            device, cb, image, vk::ImageAspectFlags::COLOR, 0, layers,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ,
        );
    });
    staging.destroy(device, allocator);
    result?;
    Ok((image, view, allocation))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layers_hold_each_view_contiguously() {
        // 2 x 2 views of 3 x 3 texels; texel value encodes (cell, row, col).
        let (g, t) = (2usize, 3usize);
        let size = g * t;
        let mut atlas = vec![0u8; size * size * 4];
        for j in 0..g {
            for i in 0..g {
                for y in 0..t {
                    for x in 0..t {
                        let at = (((j * t + y) * size) + i * t + x) * 4;
                        atlas[at..at + 4].copy_from_slice(&[(j * g + i) as u8, y as u8, x as u8, 255]);
                    }
                }
            }
        }
        let layers = atlas_to_layers(2, 3, &atlas);
        assert_eq!(layers.len(), atlas.len());
        for cell in 0..g * g {
            for y in 0..t {
                for x in 0..t {
                    let at = ((cell * t + y) * t + x) * 4;
                    assert_eq!(layers[at..at + 4], [cell as u8, y as u8, x as u8, 255], "cell {cell} ({y},{x})");
                }
            }
        }
    }

    #[test]
    fn validation_rejects_bad_atlases() {
        let ok = vec![0u8; 8 * 8 * 4 * 16];
        let up = |grid, tile, a: &'static [u8]| ImpostorUpload { grid, tile, centre: [0.0; 3], radius: 1.0, albedo: a, aux: a };
        assert!(ImpostorUpload { grid: 2, tile: 8, centre: [0.0; 3], radius: 1.0, albedo: &ok[..16 * 16 * 4], aux: &ok[..16 * 16 * 4] }.validate().is_ok());
        assert!(up(12, 4, &[]).validate().is_err(), "12 x 12 x 2 layers exceeds 256");
        assert!(up(2, 8, &[0; 10]).validate().is_err());
        assert!(ImpostorUpload { grid: 2, tile: 8, centre: [f32::NAN; 3], radius: 1.0, albedo: &ok[..16 * 16 * 4], aux: &ok[..16 * 16 * 4] }.validate().is_err());
        assert!(ImpostorUpload { grid: 2, tile: 8, centre: [0.0; 3], radius: 0.0, albedo: &ok[..16 * 16 * 4], aux: &ok[..16 * 16 * 4] }.validate().is_err());
    }
}
