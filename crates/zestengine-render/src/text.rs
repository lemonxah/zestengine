//! GPU side of the text system: glyph atlas images (one 2D array per pixel
//! format, a page per layer), per-rect atlas uploads, one instanced
//! vertex-pulling pipeline, and the draw recording for screen space and the
//! scene pass.
//!
//! The atlas layers are the "bindless" part: every font, page and effect is a
//! layer index inside one array image per format, so all text draws through a
//! single pipeline and descriptor set per frame, with one `vkCmdDraw` per text
//! object. World and 3D text draw straight into the scene pass (depth tested,
//! multiview aware), so no per-canvas render target is needed.

use std::collections::{HashMap, VecDeque};

use anyhow::{Context as _, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec4};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use zestengine_text::atlas::{AtlasFormat, AtlasUpload};
use zestengine_text::{Billboard, GlyphInstance, TextDraw, TextHandle, TextSpace};

use crate::alloc::{Buffer, one_time_submit};
use crate::frame::FRAMES_IN_FLIGHT;
use crate::pipeline::PassFormats;

const SCREEN_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/text_screen.vert.spv"));
const SCREEN_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/text_screen.frag.spv"));
const WORLD_VERT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/text_world.vert.spv"));
const WORLD_FRAG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/text_world.frag.spv"));
const WORLD_VERT_MV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/text_world.vert.mv.spv"));

/// Glyph instances one frame can hold (40 bytes each: 2.5 MiB per frame).
const INSTANCE_CAP: usize = 65_536;
/// Atlas upload staging per frame.
const STAGING_CAP: u64 = 8 << 20;

const PUSH_SIZE: u32 = std::mem::size_of::<Push>() as u32;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Push {
    model: [[f32; 4]; 4],
    tint: [f32; 4],
    clip: [f32; 4],
    /// x billboard mode, y flags (1 supersample, 2 clip), zw local offset.
    params: [f32; 4],
}

struct AtlasImage {
    image: vk::Image,
    view: vk::ImageView,
    allocation: Option<Allocation>,
    layers: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PipeKey {
    world: bool,
    color: vk::Format,
    depth: vk::Format,
    gbuf: bool,
    gbuf_format: vk::Format,
    velocity: bool,
    multiview: bool,
    samples: u8,
}

pub(crate) struct TextGpu {
    handle: TextHandle,
    sampler: vk::Sampler,
    rgba: AtlasImage,
    r8: AtlasImage,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    sets: [vk::DescriptorSet; FRAMES_IN_FLIGHT],
    inst: Vec<Buffer>,
    staging: Vec<Buffer>,
    layout_screen: vk::PipelineLayout,
    layout_world: vk::PipelineLayout,
    modules: [vk::ShaderModule; 5],
    pipelines: HashMap<PipeKey, vk::Pipeline>,
    pending: VecDeque<AtlasUpload>,
    /// This frame's draws, sorted (screen then world, by order).
    draws: Vec<TextDraw>,
    /// `(first_instance, count)` per draw in the instance buffer.
    ranges: Vec<(u32, u32)>,
    warned_overflow: bool,
}

#[allow(clippy::too_many_arguments)]
fn create_atlas_image(
    device: &ash::Device,
    allocator: &mut Allocator,
    pool: vk::CommandPool,
    queue: vk::Queue,
    format: vk::Format,
    size: u32,
    layers: u32,
    name: &str,
) -> Result<AtlasImage> {
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D { width: size, height: size, depth: 1 })
        .mip_levels(1)
        .array_layers(layers)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.create_image(&info, None)? };
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let allocation = allocator
        .allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .with_context(|| format!("allocating {name}"))?;
    unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };
    let view = unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
                .format(format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(layers),
                ),
            None,
        )?
    };
    // Clear to "far outside the glyph" and leave it readable.
    one_time_submit(device, pool, queue, |cb| unsafe {
        crate::image_barrier_layers(
            device, cb, image, vk::ImageAspectFlags::COLOR, 0, layers,
            vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty(),
            vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE,
        );
        device.cmd_clear_color_image(
            cb,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &vk::ClearColorValue { float32: [0.0; 4] },
            &[vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(layers)],
        );
        crate::image_barrier_layers(
            device, cb, image, vk::ImageAspectFlags::COLOR, 0, layers,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ,
        );
    })?;
    Ok(AtlasImage { image, view, allocation: Some(allocation), layers })
}

impl TextGpu {
    pub fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        cmd_pool: vk::CommandPool,
        queue: vk::Queue,
        handle: TextHandle,
        frame_set_layout: vk::DescriptorSetLayout,
    ) -> Result<Self> {
        let cfg = handle.system.lock().unwrap().config();
        let rgba = create_atlas_image(
            device, allocator, cmd_pool, queue, vk::Format::R8G8B8A8_UNORM,
            cfg.atlas_rgba.page_size, cfg.atlas_rgba.max_pages.max(1), "text atlas rgba",
        )?;
        let r8 = create_atlas_image(
            device, allocator, cmd_pool, queue, vk::Format::R8_UNORM,
            cfg.atlas_r8.page_size, cfg.atlas_r8.max_pages.max(1), "text atlas r8",
        )?;
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
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX),
        ];
        let set_layout = unsafe {
            device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?
        };
        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(2 * FRAMES_IN_FLIGHT as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(FRAMES_IN_FLIGHT as u32),
        ];
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(FRAMES_IN_FLIGHT as u32).pool_sizes(&sizes),
                None,
            )?
        };
        let layouts = [set_layout; FRAMES_IN_FLIGHT];
        let sets_vec = unsafe {
            device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))?
        };
        let mut sets = [vk::DescriptorSet::null(); FRAMES_IN_FLIGHT];
        sets.copy_from_slice(&sets_vec);

        let mut inst = Vec::new();
        let mut staging = Vec::new();
        for _ in 0..FRAMES_IN_FLIGHT {
            inst.push(Buffer::new(
                device, allocator, (INSTANCE_CAP * std::mem::size_of::<GlyphInstance>()) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "text instances",
            )?);
            staging.push(Buffer::new(
                device, allocator, STAGING_CAP, vk::BufferUsageFlags::TRANSFER_SRC, MemoryLocation::CpuToGpu,
                "text atlas staging",
            )?);
        }
        for (i, set) in sets.iter().enumerate() {
            let rgba_info = [vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(rgba.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let r8_info = [vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(r8.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let buf_info = [vk::DescriptorBufferInfo::default().buffer(inst[i].handle).offset(0).range(vk::WHOLE_SIZE)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&rgba_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&r8_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(*set)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buf_info),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }

        let push_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(PUSH_SIZE)];
        let screen_sets = [set_layout];
        let world_sets = [frame_set_layout, set_layout];
        let layout_screen = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&screen_sets).push_constant_ranges(&push_range),
                None,
            )?
        };
        let layout_world = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&world_sets).push_constant_ranges(&push_range),
                None,
            )?
        };
        let module = |spv: &[u8]| -> Result<vk::ShaderModule> {
            let code = ash::util::read_spv(&mut std::io::Cursor::new(spv))?;
            Ok(unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)? })
        };
        let modules = [module(SCREEN_VERT)?, module(SCREEN_FRAG)?, module(WORLD_VERT)?, module(WORLD_FRAG)?, module(WORLD_VERT_MV)?];
        Ok(Self {
            handle,
            sampler,
            rgba,
            r8,
            set_layout,
            pool,
            sets,
            inst,
            staging,
            layout_screen,
            layout_world,
            modules,
            pipelines: HashMap::new(),
            pending: VecDeque::new(),
            draws: Vec::new(),
            ranges: Vec::new(),
            warned_overflow: false,
        })
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            for (_, p) in self.pipelines.drain() {
                device.destroy_pipeline(p, None);
            }
            for m in self.modules {
                device.destroy_shader_module(m, None);
            }
            device.destroy_pipeline_layout(self.layout_screen, None);
            device.destroy_pipeline_layout(self.layout_world, None);
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_sampler(self.sampler, None);
            for img in [&mut self.rgba, &mut self.r8] {
                device.destroy_image_view(img.view, None);
                device.destroy_image(img.image, None);
                if let Some(a) = img.allocation.take() {
                    let _ = allocator.free(a);
                }
            }
        }
        for b in self.inst.iter_mut().chain(self.staging.iter_mut()) {
            b.destroy(device, allocator);
        }
    }

    /// Start of the frame's text work, outside any render pass: take the
    /// queued draws, pack their instances into this frame's buffer, and copy
    /// whatever atlas rects fit the staging buffer.
    pub fn prepare(&mut self, device: &ash::Device, cb: vk::CommandBuffer, frame: usize) {
        // Draws: swap with the shared queue so both keep their capacity.
        self.draws.clear();
        if let Ok(mut q) = self.handle.draws.lock() {
            std::mem::swap(&mut *q, &mut self.draws);
        }
        self.draws.sort_by_key(|d| (d.space == TextSpace::World, d.order));
        self.ranges.clear();
        let mut next = 0usize;
        let stride = std::mem::size_of::<GlyphInstance>();
        let buf = &mut self.inst[frame];
        for d in &self.draws {
            let n = d.instances.len();
            if next + n > INSTANCE_CAP {
                if !self.warned_overflow {
                    tracing::warn!("text: more than {INSTANCE_CAP} glyphs in one frame; the rest are skipped");
                    self.warned_overflow = true;
                }
                self.ranges.push((0, 0));
                continue;
            }
            buf.write(next * stride, bytemuck::cast_slice(&d.instances));
            self.ranges.push((next as u32, n as u32));
            next += n;
        }

        // Atlas uploads.
        if let Ok(mut sys) = self.handle.system.lock() {
            self.pending.extend(sys.take_uploads());
        }
        if self.pending.is_empty() {
            return;
        }
        let mut offset = 0u64;
        let mut rgba_regions: Vec<vk::BufferImageCopy> = Vec::new();
        let mut r8_regions: Vec<vk::BufferImageCopy> = Vec::new();
        while let Some(u) = self.pending.front() {
            let size = u.data.len() as u64;
            if size > STAGING_CAP {
                tracing::warn!("text: atlas upload of {size} bytes exceeds staging; dropped");
                self.pending.pop_front();
                continue;
            }
            let at = (offset + 3) & !3;
            if at + size > STAGING_CAP {
                break;
            }
            let u = self.pending.pop_front().unwrap();
            self.staging[frame].write(at as usize, &u.data);
            let region = vk::BufferImageCopy::default()
                .buffer_offset(at)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_array_layer(u.layer)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D { x: u.x as i32, y: u.y as i32, z: 0 })
                .image_extent(vk::Extent3D { width: u.w, height: u.h, depth: 1 });
            match u.format {
                AtlasFormat::Rgba8 => rgba_regions.push(region),
                AtlasFormat::R8 => r8_regions.push(region),
            }
            offset = at + size;
        }
        for (img, regions) in [(&self.rgba, &rgba_regions), (&self.r8, &r8_regions)] {
            if regions.is_empty() {
                continue;
            }
            crate::image_barrier_layers(
                device, cb, img.image, vk::ImageAspectFlags::COLOR, 0, img.layers,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ,
                vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE,
            );
            unsafe {
                device.cmd_copy_buffer_to_image(
                    cb, self.staging[frame].handle, img.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, regions,
                );
            }
            crate::image_barrier_layers(
                device, cb, img.image, vk::ImageAspectFlags::COLOR, 0, img.layers,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_SAMPLED_READ,
            );
        }
    }

    /// Whether any world-space text is queued for this frame.
    pub fn has_world(&self) -> bool {
        self.draws.iter().any(|d| d.space == TextSpace::World && !d.is_empty())
    }

    pub fn has_screen(&self) -> bool {
        self.draws.iter().any(|d| d.space == TextSpace::Screen && !d.is_empty())
    }

    fn pipeline(&mut self, device: &ash::Device, key: PipeKey) -> Result<vk::Pipeline> {
        if let Some(p) = self.pipelines.get(&key) {
            return Ok(*p);
        }
        let (vert, frag) = if key.world {
            (if key.multiview { self.modules[4] } else { self.modules[2] }, self.modules[3])
        } else {
            (self.modules[0], self.modules[1])
        };
        let stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vert).name(c"main"),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(frag).name(c"main"),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly =
            vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let mut raster = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        if key.world {
            // Pull text slightly toward the camera so it wins over the canvas
            // quad it sits on.
            raster = raster.depth_bias_enable(true).depth_bias_constant_factor(-2.0).depth_bias_slope_factor(-1.5);
        }
        let samples = match key.samples {
            4 => vk::SampleCountFlags::TYPE_4,
            2 => vk::SampleCountFlags::TYPE_2,
            _ => vk::SampleCountFlags::TYPE_1,
        };
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(samples);
        let depth = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(key.world)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
        // Premultiplied alpha into attachment 0; the G-buffer / velocity /
        // reactive attachments of the scene pass are left untouched.
        let premult = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA);
        let untouched = vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::empty());
        let mut blends = vec![premult];
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
        let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
        if key.world {
            rendering = rendering.depth_attachment_format(key.depth);
            if crate::sprite_gpu::has_stencil(key.depth) {
                rendering = rendering.stencil_attachment_format(key.depth);
            }
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
            .layout(if key.world { self.layout_world } else { self.layout_screen })
            .push_next(&mut rendering);
        let pipeline = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None).map_err(|(_, e)| e) }
            .context("creating text pipeline")?[0];
        self.pipelines.insert(key, pipeline);
        Ok(pipeline)
    }

    /// Screen-space text, inside a pass that renders to the swapchain image
    /// (no depth, one sample), after egui.
    pub fn record_screen(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        frame: usize,
        extent: vk::Extent2D,
        color: vk::Format,
    ) {
        if !self.has_screen() {
            return;
        }
        let key = PipeKey {
            world: false, color, depth: vk::Format::UNDEFINED, gbuf: false, gbuf_format: vk::Format::UNDEFINED,
            velocity: false, multiview: false, samples: 1,
        };
        let pipeline = match self.pipeline(device, key) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("text screen pipeline: {e:#}");
                return;
            }
        };
        let (w, h) = (extent.width.max(1) as f32, extent.height.max(1) as f32);
        let ortho = Mat4::from_cols(
            Vec4::new(2.0 / w, 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0 / h, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(-1.0, -1.0, 0.0, 1.0),
        );
        unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, self.layout_screen, 0, &[self.sets[frame]], &[]);
            device.cmd_set_viewport(cb, 0, &[vk::Viewport { x: 0.0, y: 0.0, width: w, height: h, min_depth: 0.0, max_depth: 1.0 }]);
            device.cmd_set_scissor(cb, 0, &[vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent }]);
        }
        for (d, &(first, count)) in self.draws.iter().zip(&self.ranges) {
            if d.space != TextSpace::Screen || count == 0 {
                continue;
            }
            let model = ortho * Mat4::from_cols_array_2d(&d.transform);
            self.push_and_draw(device, cb, self.layout_screen, d, model, first, count);
        }
    }

    /// World and 3D text, inside the scene pass after the transparent draws.
    /// `frame_set` is the pass's frame uniform set (set 0).
    #[allow(clippy::too_many_arguments)]
    pub fn record_world(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        frame: usize,
        frame_set: vk::DescriptorSet,
        pass: PassFormats,
        gbuf: bool,
        velocity: bool,
    ) {
        if !self.has_world() {
            return;
        }
        let key = PipeKey {
            world: true,
            color: pass.color,
            depth: pass.depth,
            gbuf,
            gbuf_format: pass.gbuf,
            velocity,
            multiview: pass.multiview,
            samples: pass.samples,
        };
        let pipeline = match self.pipeline(device, key) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("text world pipeline: {e:#}");
                return;
            }
        };
        unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_bind_descriptor_sets(
                cb, vk::PipelineBindPoint::GRAPHICS, self.layout_world, 0, &[frame_set, self.sets[frame]], &[],
            );
        }
        for (d, &(first, count)) in self.draws.iter().zip(&self.ranges) {
            if d.space != TextSpace::World || count == 0 {
                continue;
            }
            let model = Mat4::from_cols_array_2d(&d.transform);
            self.push_and_draw(device, cb, self.layout_world, d, model, first, count);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push_and_draw(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        d: &TextDraw,
        model: Mat4,
        first: u32,
        count: u32,
    ) {
        let billboard = match d.billboard {
            Billboard::None => 0.0,
            Billboard::Y => 1.0,
            Billboard::Full => 2.0,
        };
        let flags = d.supersample as u32 | ((d.clip.is_some() as u32) << 1);
        let push = Push {
            model: model.to_cols_array_2d(),
            tint: d.tint,
            clip: d.clip.unwrap_or([0.0; 4]),
            params: [billboard, flags as f32, d.local_offset[0], d.local_offset[1]],
        };
        unsafe {
            device.cmd_push_constants(
                cb, layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, bytemuck::bytes_of(&push),
            );
            device.cmd_draw(cb, 6, count, 0, first);
        }
    }

    /// End of the frame: the draws recorded are consumed.
    pub fn end_frame(&mut self) {
        self.draws.clear();
        self.ranges.clear();
        if let Ok(mut s) = self.handle.system.lock() {
            s.next_frame();
        }
    }
}

// Device tests (real Vulkan device, skipped without one); they reach into the
// private pipeline cache and atlas images, hence a child module.
#[cfg(test)]
#[path = "gpu_tests/text.rs"]
mod device_tests;
