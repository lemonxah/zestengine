//! VR UI overlay: renders the editor's egui output to an offscreen texture and
//! draws it (plus pointer/cursor markers) as quads in the headset's eye images,
//! so the whole desktop UI is usable in VR. The left-hand panel quad samples the
//! UI texture; solid quads mark the pointer ray + cursor.
//!
//! [`VrOverlay`] owns the UI texture + a small quad pipeline. The renderer fills
//! the texture via egui (`render_vr_ui`) and records the quads inside each eye's
//! scene pass (`record_quad`). Sized to the swapchain so egui's layout matches
//! the desktop exactly; the right-hand pointer maps panel UV → window pixels.

use std::io::Cursor;

use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::vulkan::{Allocation, Allocator};

use crate::{create_image, create_view};

const VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vr_quad.vert.spv"));
const FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vr_quad.frag.spv"));

/// Push constant for one overlay quad: world MVP + a mode/colour vector.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct QuadPush {
    pub mvp: [[f32; 4]; 4],
    /// x = mode (0 textured UI, 1 solid), yzw = solid colour.
    pub params: [f32; 4],
    /// xy = UV min, zw = UV max — the sub-region of the UI texture this quad
    /// shows. `[0,0,1,1]` for the whole texture (and for solid quads, unused).
    pub uv_rect: [f32; 4],
}

pub(crate) struct VrOverlay {
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sampler: vk::Sampler,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    /// Offscreen UI texture (the editor's egui output), sampled by the panel quad.
    ui_image: vk::Image,
    ui_view: vk::ImageView,
    ui_alloc: Option<Allocation>,
    extent: vk::Extent2D,
    color_format: vk::Format,
}

impl VrOverlay {
    pub fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        color_format: vk::Format,
        extent: vk::Extent2D,
    ) -> Result<Self> {
        let set_bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&set_bindings),
                None,
            )?
        };
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .size(std::mem::size_of::<QuadPush>() as u32)];
        let layouts = [set_layout];
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&push),
                None,
            )?
        };
        let pipeline = Self::build_pipeline(device, pipeline_layout, color_format)?;
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
                    .max_sets(1)
                    .pool_sizes(&[vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: 1,
                    }]),
                None,
            )?
        };
        let set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&[set_layout]),
            )?[0]
        };

        let mut me = Self {
            set_layout,
            pipeline_layout,
            pipeline,
            sampler,
            pool,
            set,
            ui_image: vk::Image::null(),
            ui_view: vk::ImageView::null(),
            ui_alloc: None,
            extent,
            color_format,
        };
        me.create_ui_texture(device, allocator, extent)?;
        Ok(me)
    }

    fn build_pipeline(
        device: &ash::Device,
        layout: vk::PipelineLayout,
        color_format: vk::Format,
    ) -> Result<vk::Pipeline> {
        let vert = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default()
                    .code(&ash::util::read_spv(&mut Cursor::new(VERT_SPV))?),
                None,
            )?
        };
        let frag = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default()
                    .code(&ash::util::read_spv(&mut Cursor::new(FRAG_SPV))?),
                None,
            )?
        };
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
        // No depth test/write: the overlay always draws on top of the scene.
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(false)
            .depth_write_enable(false);
        let blend = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend);
        let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);
        let formats = [color_format];
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&formats)
            .depth_attachment_format(crate::depth_format());
        if crate::depth_has_stencil() {
            rendering = rendering.stencil_attachment_format(crate::depth_format());
        }
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
        unsafe {
            device.destroy_shader_module(vert, None);
            device.destroy_shader_module(frag, None);
        }
        Ok(pipeline)
    }

    fn create_ui_texture(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
    ) -> Result<()> {
        let (image, alloc) = create_image(
            device,
            allocator,
            self.color_format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            extent,
            "vr ui texture",
        )?;
        let view = create_view(device, image, self.color_format, vk::ImageAspectFlags::COLOR)?;
        let info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&info);
        unsafe { device.update_descriptor_sets(&[write], &[]) };
        self.ui_image = image;
        self.ui_view = view;
        self.ui_alloc = Some(alloc);
        self.extent = extent;
        Ok(())
    }

    pub fn resize(
        &mut self,
        device: &ash::Device,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
    ) -> Result<()> {
        self.free_ui_texture(device, allocator);
        self.create_ui_texture(device, allocator, extent)
    }

    pub fn ui_image(&self) -> vk::Image {
        self.ui_image
    }
    pub fn ui_view(&self) -> vk::ImageView {
        self.ui_view
    }
    pub fn ui_extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// Record one overlay quad into the (already-begun) eye rendering: bind the
    /// pipeline + UI-texture set, push the MVP + mode/colour, draw 6 verts. The
    /// caller must have set the viewport/scissor (shared with the scene pass).
    pub fn record_quad(&self, device: &ash::Device, cb: vk::CommandBuffer, push: &QuadPush) {
        unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.set],
                &[],
            );
            device.cmd_push_constants(
                cb,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(push),
            );
            device.cmd_draw(cb, 6, 1, 0, 0);
        }
    }

    fn free_ui_texture(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_image_view(self.ui_view, None);
            device.destroy_image(self.ui_image, None);
        }
        if let Some(a) = self.ui_alloc.take() {
            let _ = allocator.free(a);
        }
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        self.free_ui_texture(device, allocator);
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
        }
    }
}
