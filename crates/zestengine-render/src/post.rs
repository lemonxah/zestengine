//! HDR scene target + fullscreen post-processing pass.
//!
//! The forward pass renders linear HDR into [`PostPass`]'s per-frame
//! `R16G16B16A16_SFLOAT` target (no inline tonemap). This pass then samples it
//! and applies bloom + chromatic aberration + exposure + colour grading +
//! tonemap + vignette (`post.frag`), writing display-space colour to the
//! swapchain. One target per frame-in-flight so a frame's write never races the
//! previous frame's read.

use std::io::Cursor;

use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::vulkan::{Allocation, Allocator};

use crate::PostFx;
use crate::bloom::{BloomChain, BloomPass};
use crate::underwater::Wobble;

const FULLSCREEN_VERT_SPV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/fullscreen.vert.spv"));
const POST_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/post.frag.spv"));

/// Linear HDR scene-colour format.
pub(crate) const HDR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PostPush {
    p0: [f32; 4], // tonemap mode, exposure EV, grade exposure, contrast
    p1: [f32; 4], // saturation, temperature, tint, grading enabled
    p2: [f32; 4], // vignette enabled, intensity, smoothness, _
    p3: [f32; 4], // vignette color rgb, _
    p4: [f32; 4], // bloom enabled, threshold, intensity, radius
    p5: [f32; 4], // CA enabled, CA intensity, wobble, time
    p6: [f32; 4], // waterline (see `underwater::Wobble`)
    p7: [f32; 4], // FXAA enabled, sharpen amount, auto exposure enabled, _
}

struct HdrTarget {
    image: vk::Image,
    view: vk::ImageView,
    alloc: Option<Allocation>,
}

pub(crate) struct PostPass {
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    /// Format `pipeline` writes (the window swapchain's).
    format: vk::Format,
    /// Pipelines for other output formats (XR eye images), built on demand.
    extra: Vec<(vk::Format, vk::Pipeline)>,
    vert: vk::ShaderModule,
    frag: vk::ShaderModule,
    sampler: vk::Sampler,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    targets: Vec<HdrTarget>,
    extent: vk::Extent2D,
    bloom: BloomPass,
    /// Bloom chain for the swapchain path (the viewport / preview own theirs).
    chain: Option<BloomChain>,
    /// Every bloom chain is built for the view extent divided by this (the
    /// render tier's; 2 on mobile).
    bloom_divisor: u32,
    /// FXAA and the sharpen amount (aa.rs) for the next post passes; the
    /// renderer sets them around the main camera's post pass only.
    fxaa: bool,
    sharpen: f32,
    pub exposure: crate::exposure::ExposurePass,
}

impl PostPass {
    pub fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        swapchain_format: vk::Format,
        extent: vk::Extent2D,
        frames: usize,
        bloom_divisor: u32,
    ) -> Result<Self> {
        let set_bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            // Compute too: the bloom prefilter reads post source sets.
            .stage_flags(vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE)];
        let set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&set_bindings),
                None,
            )?
        };
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .size(std::mem::size_of::<PostPush>() as u32)];
        let exposure=crate::exposure::ExposurePass::new(device,allocator,set_layout)?;
        // Set 0 = HDR source, set 1 = bloom, set 2 = per-view exposure.
        let layouts = [set_layout, set_layout, exposure.layout];
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&push_ranges),
                None,
            )?
        };

        let vert = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default()
                    .code(&ash::util::read_spv(&mut Cursor::new(FULLSCREEN_VERT_SPV))?),
                None,
            )?
        };
        let frag = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default()
                    .code(&ash::util::read_spv(&mut Cursor::new(POST_FRAG_SPV))?),
                None,
            )?
        };
        let pipeline = Self::build_pipeline(device, pipeline_layout, vert, frag, swapchain_format)?;

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
                    .max_sets(frames as u32)
                    .pool_sizes(&[vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: frames as u32,
                    }]),
                None,
            )?
        };

        let mut pass = Self {
            set_layout,
            pipeline_layout,
            pipeline,
            format: swapchain_format,
            extra: Vec::new(),
            vert,
            frag,
            sampler,
            pool,
            sets: Vec::new(),
            targets: Vec::new(),
            extent,
            bloom: BloomPass::new(device, set_layout, exposure.layout)?,
            chain: None,
            bloom_divisor: bloom_divisor.max(1),
            fxaa: false,
            sharpen: 0.0,
            exposure,
        };
        pass.create_targets(device, allocator, extent, frames)?;
        pass.chain = Some(pass.new_bloom_chain(device, allocator, extent)?);
        Ok(pass)
    }

    fn build_pipeline(
        device: &ash::Device,
        layout: vk::PipelineLayout,
        vert: vk::ShaderModule,
        frag: vk::ShaderModule,
        color_format: vk::Format,
    ) -> Result<vk::Pipeline> {
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag)
                .name(c"main"),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();
        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let color_formats = [color_format];
        let mut rendering =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic)
            .layout(layout)
            .push_next(&mut rendering);
        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
                .map_err(|(_, e)| e)?[0]
        };
        Ok(pipeline)
    }

    fn create_targets(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
        frames: usize,
    ) -> Result<()> {
        let layouts = vec![self.set_layout; frames];
        let sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.pool)
                    .set_layouts(&layouts),
            )?
        };
        for &set in &sets {
            let (image, alloc) = crate::create_image(
                device,
                allocator,
                HDR_FORMAT,
                // TRANSFER_SRC: the water's opaque scene copy.
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
                extent,
                "scene hdr",
            )?;
            let view = crate::create_view(device, image, HDR_FORMAT, vk::ImageAspectFlags::COLOR)?;
            let info = [vk::DescriptorImageInfo::default()
                .sampler(self.sampler)
                .image_view(view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&info);
            unsafe { device.update_descriptor_sets(&[write], &[]) };
            self.targets.push(HdrTarget {
                image,
                view,
                alloc: Some(alloc),
            });
        }
        self.sets = sets;
        self.extent = extent;
        Ok(())
    }

    /// FXAA on/off and the CAS-like sharpen amount (0 = off) for the post
    /// passes recorded after this call.
    pub fn set_aa(&mut self, fxaa: bool, sharpen: f32) {
        self.fxaa = fxaa;
        self.sharpen = sharpen.clamp(0.0, 1.0);
    }

    pub fn hdr_image(&self, frame: usize) -> vk::Image {
        self.targets[frame].image
    }
    pub fn hdr_view(&self, frame: usize) -> vk::ImageView {
        self.targets[frame].view
    }

    /// The descriptor set layout (1 sampled HDR image) + sampler, so other
    /// targets (the editor viewport) can allocate their own post source set.
    pub fn set_layout(&self) -> vk::DescriptorSetLayout {
        self.set_layout
    }
    pub fn sampler(&self) -> vk::Sampler {
        self.sampler
    }

    /// Record the fullscreen post pass: samples the HDR target for `frame` (must
    /// already be in SHADER_READ_ONLY) and writes display colour to the active
    /// dynamic-rendering attachment (the swapchain). Bind nothing else first.
    pub fn record(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        frame: usize,
        extent: vk::Extent2D,
        post: &PostFx,
        wobble: Option<Wobble>,
    ) {
        let bloom = self.chain.as_ref().unwrap().output_set();
        self.record_set(device, cb, self.sets[frame], bloom, extent, post, wobble);
    }

    /// The swapchain path's own source set for `frame` (its HDR target).
    pub fn frame_set(&self, frame: usize) -> vk::DescriptorSet {
        self.sets[frame]
    }

    /// The swapchain path's bloom chain.
    pub fn chain(&self) -> &BloomChain {
        self.chain.as_ref().unwrap()
    }

    pub fn new_bloom_chain(
        &self,
        device: &ash::Device,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
    ) -> Result<BloomChain> {
        let d = self.bloom_divisor;
        let extent = vk::Extent2D { width: (extent.width / d).max(1), height: (extent.height / d).max(1) };
        self.bloom.new_chain(device, allocator, extent, self.set_layout, self.sampler)
    }

    /// Build `chain`'s bloom from `src_set` (a post source set). Record outside a
    /// render pass, before the post pass that samples it.
    pub fn record_bloom(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        chain: &BloomChain,
        src_set: vk::DescriptorSet,
        post: &PostFx,
    ) {
        self.bloom.record(device, cb, chain, src_set, post, self.exposure.set());
    }

    /// Build the pipeline writing `format` if it isn't the swapchain's and
    /// doesn't exist yet (the XR eye images may use another format).
    pub fn ensure_format(&mut self, device: &ash::Device, format: vk::Format) -> Result<()> {
        if format == self.format || self.extra.iter().any(|(f, _)| *f == format) {
            return Ok(());
        }
        let pipeline = Self::build_pipeline(device, self.pipeline_layout, self.vert, self.frag, format)?;
        self.extra.push((format, pipeline));
        Ok(())
    }

    /// Run the post pass sampling an externally-provided descriptor set (binding
    /// 0 = the HDR source). Used by the editor viewport, which owns its own HDR
    /// target + set.
    #[allow(clippy::too_many_arguments)]
    pub fn record_set(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        set: vk::DescriptorSet,
        bloom_set: vk::DescriptorSet,
        extent: vk::Extent2D,
        post: &PostFx,
        wobble: Option<Wobble>,
    ) {
        self.record_set_as(device, cb, set, bloom_set, extent, post, wobble, self.format);
    }

    /// [`Self::record_set`] into an attachment of `format` (see
    /// [`Self::ensure_format`]; an unknown format falls back to the swapchain's).
    #[allow(clippy::too_many_arguments)]
    pub fn record_set_as(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        set: vk::DescriptorSet,
        bloom_set: vk::DescriptorSet,
        extent: vk::Extent2D,
        post: &PostFx,
        wobble: Option<Wobble>,
        format: vk::Format,
    ) {
        let pipeline = self.extra.iter().find(|(f, _)| *f == format).map_or(self.pipeline, |(_, p)| *p);
        let w = wobble.unwrap_or(Wobble { strength: 0.0, blur: 0.0, time: 0.0, line: [0.0; 3] });
        let push = PostPush {
            p0: [post.tonemap as f32, post.exposure, post.grade_exposure, post.contrast],
            p1: [
                post.saturation,
                post.temperature,
                post.tint,
                if post.grading_enabled { 1.0 } else { 0.0 },
            ],
            p2: [
                if post.vignette_enabled { 1.0 } else { 0.0 },
                post.vignette_intensity,
                post.vignette_smoothness,
                0.0,
            ],
            p3: [post.vignette_color[0], post.vignette_color[1], post.vignette_color[2], 0.0],
            p4: [
                if post.bloom_enabled { 1.0 } else { 0.0 },
                post.bloom_threshold,
                post.bloom_intensity,
                post.bloom_radius,
            ],
            p5: [if post.ca_enabled { 1.0 } else { 0.0 }, post.ca_intensity, w.strength, w.time],
            p6: [w.line[0], w.line[1], w.line[2], w.blur],
            p7: [if self.fxaa { 1.0 } else { 0.0 }, self.sharpen, if post.auto_exposure.enabled {1.0} else {0.0}, 0.0],
        };
        unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
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
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[set, bloom_set, self.exposure.set()],
                &[],
            );
            device.cmd_push_constants(
                cb,
                self.pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push),
            );
            device.cmd_draw(cb, 3, 1, 0, 0);
        }
    }

    pub fn resize(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
    ) -> Result<()> {
        let frames = self.targets.len();
        self.destroy_targets(device, allocator);
        if let Some(mut c) = self.chain.take() {
            c.destroy(device, allocator);
        }
        self.chain = Some(self.new_bloom_chain(device, allocator, extent)?);
        unsafe {
            device.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())?;
        }
        self.create_targets(device, allocator, extent, frames)
    }

    fn destroy_targets(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for t in &mut self.targets {
            unsafe {
                device.destroy_image_view(t.view, None);
                device.destroy_image(t.image, None);
            }
            if let Some(a) = t.alloc.take() {
                let _ = allocator.free(a);
            }
        }
        self.targets.clear();
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        self.exposure.destroy(device,allocator);
        self.destroy_targets(device, allocator);
        if let Some(mut c) = self.chain.take() {
            c.destroy(device, allocator);
        }
        self.bloom.destroy(device);
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_pipeline(self.pipeline, None);
            for (_, p) in self.extra.drain(..) {
                device.destroy_pipeline(p, None);
            }
            device.destroy_shader_module(self.vert, None);
            device.destroy_shader_module(self.frag, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
