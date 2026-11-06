//! Screen-space selection outline for the editor viewport.
//!
//! The selected draws are rasterised into a small `R8G8` mask (R = selection
//! strength, 1 for the primary selection and 0.5 for the others, G = "passed
//! the scene depth test"), straight from `FrameInput::draws` with each mesh's
//! own vertex buffer. That is the buffer every mesh kind owns after its
//! compute passes (skinning, morphs, cloth), so static, skinned, instanced,
//! GPU-culled, cluster-path and transparent objects all take the same two
//! draws; none of them needs its own hull. A fullscreen composite then draws a
//! constant-pixel-width ring outside the mask over the tonemapped viewport
//! (`select_outline.frag`). Nothing is recorded when nothing is selected.

use std::io::Cursor;

use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::vulkan::{Allocation, Allocator};

use crate::pipeline::INSTANCE_STRIDE;
use crate::{Vertex, create_image, create_view};

const MASK_FORMAT: vk::Format = vk::Format::R8G8_UNORM;
const MASK_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/select_mask.vert.spv"));
const MASK_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/select_mask.frag.spv"));
const OUTLINE_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/select_outline.frag.spv"));
const FULLSCREEN_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fullscreen.vert.spv"));

/// Draws with `highlight` at or above this are the primary selection.
pub(crate) const PRIMARY_HIGHLIGHT: f32 = 0.75;

/// How the editor wants the selection drawn.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelectionOutlineStyle {
    pub enabled: bool,
    /// sRGB colour of the primary selection; the others use it dimmed.
    pub color: [f32; 3],
    /// Ring width in framebuffer pixels (the caller scales by pixels per point).
    pub width_px: f32,
    /// Strength of the ring where the selected surface is hidden behind other geometry.
    pub occluded: f32,
}

impl Default for SelectionOutlineStyle {
    fn default() -> Self {
        Self { enabled: true, color: [0.72, 0.25, 1.0], width_px: 2.0, occluded: 0.35 }
    }
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

pub(crate) fn is_srgb(format: vk::Format) -> bool {
    matches!(format, vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB | vk::Format::A8B8G8R8_SRGB_PACK32)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaskPush {
    view_proj: [[f32; 4]; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompositePush {
    primary: [f32; 4],
    secondary: [f32; 4],
    params: [f32; 4],
}

/// The mask image plus the descriptor set the composite samples it through.
pub(crate) struct MaskTarget {
    pub extent: vk::Extent2D,
    image: vk::Image,
    pub view: vk::ImageView,
    allocation: Option<Allocation>,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
}

impl MaskTarget {
    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }
        if let Some(a) = self.allocation.take() {
            let _ = allocator.free(a);
        }
    }
}

pub(crate) struct SelectionPass {
    pub style: SelectionOutlineStyle,
    set_layout: vk::DescriptorSetLayout,
    mask_layout: vk::PipelineLayout,
    composite_layout: vk::PipelineLayout,
    sampler: vk::Sampler,
    mask_vert: vk::ShaderModule,
    mask_frag: vk::ShaderModule,
    composite_vert: vk::ShaderModule,
    composite_frag: vk::ShaderModule,
    /// Mask draw with no depth test, writing R (the selection silhouette).
    mask_all: vk::Pipeline,
    /// Mask draw depth-tested against the scene, writing G (the visible part).
    mask_visible: vk::Pipeline,
    /// Composite pipeline and the colour format it was built for.
    composite: Option<(vk::Format, vk::Pipeline)>,
    target: Option<MaskTarget>,
    /// Targets replaced by a resize, kept until the frames that may still read them retire.
    retired: Vec<(u32, MaskTarget)>,
}

impl SelectionPass {
    pub fn new(device: &ash::Device, depth_format: vk::Format) -> Result<Self> {
        let module = |spv: &[u8]| -> Result<vk::ShaderModule> {
            Ok(unsafe {
                device.create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&ash::util::read_spv(&mut Cursor::new(spv))?),
                    None,
                )?
            })
        };
        let mask_vert = module(MASK_VERT_SPV)?;
        let mask_frag = module(MASK_FRAG_SPV)?;
        let composite_vert = module(FULLSCREEN_VERT_SPV)?;
        let composite_frag = module(OUTLINE_FRAG_SPV)?;
        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let set_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)? };
        let mask_push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .size(size_of::<MaskPush>() as u32)];
        let mask_layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&mask_push), None)? };
        let sets = [set_layout];
        let comp_push = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::FRAGMENT).size(size_of::<CompositePush>() as u32)];
        let composite_layout = unsafe {
            device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&sets).push_constant_ranges(&comp_push), None)?
        };
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
        let mut pass = Self {
            style: SelectionOutlineStyle::default(),
            set_layout,
            mask_layout,
            composite_layout,
            sampler,
            mask_vert,
            mask_frag,
            composite_vert,
            composite_frag,
            mask_all: vk::Pipeline::null(),
            mask_visible: vk::Pipeline::null(),
            composite: None,
            target: None,
            retired: Vec::new(),
        };
        pass.mask_all = pass.build_mask(device, depth_format, false)?;
        pass.mask_visible = pass.build_mask(device, depth_format, true)?;
        Ok(pass)
    }

    fn build_mask(&self, device: &ash::Device, depth_format: vk::Format, depth_tested: bool) -> Result<vk::Pipeline> {
        let stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(self.mask_vert).name(c"main"),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(self.mask_frag).name(c"main"),
        ];
        let bindings = [
            vk::VertexInputBindingDescription::default().binding(0).stride(size_of::<Vertex>() as u32).input_rate(vk::VertexInputRate::VERTEX),
            vk::VertexInputBindingDescription::default().binding(1).stride(INSTANCE_STRIDE).input_rate(vk::VertexInputRate::INSTANCE),
        ];
        let mut attributes = vec![vk::VertexInputAttributeDescription { location: 0, binding: 0, format: vk::Format::R32G32B32_SFLOAT, offset: 0 }];
        for c in 0..4 {
            attributes.push(vk::VertexInputAttributeDescription { location: 8 + c, binding: 1, format: vk::Format::R32G32B32A32_SFLOAT, offset: c * 16 });
        }
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default().vertex_binding_descriptions(&bindings).vertex_attribute_descriptions(&attributes);
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        // Double sided (thin planes, open meshes). The visible pass pulls the
        // selection a hair toward the camera so the object's own depth, laid
        // by the scene with a different vertex shader, never hides it.
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(depth_tested)
            .depth_bias_constant_factor(-4.0)
            .depth_bias_slope_factor(-2.0)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(depth_tested)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
        let blend = [vk::PipelineColorBlendAttachmentState::default().color_write_mask(if depth_tested {
            vk::ColorComponentFlags::G
        } else {
            vk::ColorComponentFlags::R
        })];
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let formats = [MASK_FORMAT];
        let stencil_fmt = if crate::sprite_gpu::has_stencil(depth_format) { depth_format } else { vk::Format::UNDEFINED };
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&formats)
            .depth_attachment_format(depth_format)
            .stencil_attachment_format(stencil_fmt);
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
            .layout(self.mask_layout)
            .push_next(&mut rendering);
        Ok(unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None).map_err(|(_, e)| e)?[0] })
    }

    fn build_composite(&self, device: &ash::Device, color_format: vk::Format) -> Result<vk::Pipeline> {
        let stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(self.composite_vert).name(c"main"),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(self.composite_frag).name(c"main"),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();
        let blend = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ZERO)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B)];
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let formats = [color_format];
        let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&formats);
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
            .layout(self.composite_layout)
            .push_next(&mut rendering);
        Ok(unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None).map_err(|(_, e)| e)?[0] })
    }

    /// Age retired targets by one frame and free the ones no frame in flight can still read.
    pub fn tick(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for (age, _) in self.retired.iter_mut() {
            *age += 1;
        }
        let mut keep = Vec::new();
        for (age, mut t) in self.retired.drain(..) {
            if age > 4 {
                t.destroy(device, allocator);
            } else {
                keep.push((age, t));
            }
        }
        self.retired = keep;
    }

    /// The mask target for `extent` (recreated on resize).
    pub fn ensure_target(&mut self, device: &ash::Device, allocator: &mut Allocator, extent: vk::Extent2D) -> Result<&MaskTarget> {
        if self.target.as_ref().is_some_and(|t| t.extent != extent)
            && let Some(old) = self.target.take()
        {
            self.retired.push((0, old));
        }
        if self.target.is_none() {
            let (image, allocation) = create_image(
                device,
                allocator,
                MASK_FORMAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                extent,
                "selection mask",
            )?;
            let view = create_view(device, image, MASK_FORMAT, vk::ImageAspectFlags::COLOR)?;
            let pool = unsafe {
                device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&[vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 1 }]),
                    None,
                )?
            };
            let layouts = [self.set_layout];
            let set = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))?[0] };
            let info = [vk::DescriptorImageInfo::default()
                .sampler(self.sampler)
                .image_view(view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
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
            self.target = Some(MaskTarget { extent, image, view, allocation: Some(allocation), pool, set });
        }
        Ok(self.target.as_ref().unwrap())
    }

    pub fn mask_image(&self) -> Option<(vk::Image, vk::ImageView)> {
        self.target.as_ref().map(|t| (t.image, t.view))
    }

    pub fn mask_pipelines(&self) -> (vk::Pipeline, vk::Pipeline, vk::PipelineLayout) {
        (self.mask_all, self.mask_visible, self.mask_layout)
    }

    pub fn mask_push(view_proj: glam::Mat4, strength: f32) -> Vec<u8> {
        bytemuck::bytes_of(&MaskPush { view_proj: view_proj.to_cols_array_2d(), params: [strength, 0.0, 0.0, 0.0] }).to_vec()
    }

    /// Draw the ring over the colour target currently bound (inside a rendering
    /// instance whose colour format is `color_format`).
    pub fn record_composite(&mut self, device: &ash::Device, cb: vk::CommandBuffer, color_format: vk::Format, extent: vk::Extent2D) -> Result<()> {
        let Some(target) = self.target.as_ref() else { return Ok(()) };
        if self.composite.is_none_or(|(f, _)| f != color_format) {
            if let Some((_, old)) = self.composite.take() {
                // A format change is a swapchain recreate: the device is idle then.
                unsafe { device.destroy_pipeline(old, None) };
            }
            let p = self.build_composite(device, color_format)?;
            self.composite = Some((color_format, p));
        }
        let pipeline = self.composite.unwrap().1;
        let set = target.set;
        let linear = is_srgb(color_format);
        let conv = |c: [f32; 3], k: f32| -> [f32; 4] {
            let f = |v: f32| if linear { srgb_to_linear(v) } else { v };
            [f(c[0]), f(c[1]), f(c[2]), k]
        };
        let s = self.style;
        // Secondary selections: the same hue, softer.
        let push = CompositePush {
            primary: conv(s.color, 1.0),
            secondary: conv(s.color, 0.6),
            params: [s.width_px.clamp(0.5, 12.0), s.occluded.clamp(0.0, 1.0), 0.0, 0.0],
        };
        unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_set_viewport(
                cb,
                0,
                &[vk::Viewport { x: 0.0, y: 0.0, width: extent.width as f32, height: extent.height as f32, min_depth: 0.0, max_depth: 1.0 }],
            );
            device.cmd_set_scissor(cb, 0, &[vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent }]);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, self.composite_layout, 0, &[set], &[]);
            device.cmd_push_constants(cb, self.composite_layout, vk::ShaderStageFlags::FRAGMENT, 0, bytemuck::bytes_of(&push));
            device.cmd_draw(cb, 3, 1, 0, 0);
        }
        Ok(())
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(mut t) = self.target.take() {
            t.destroy(device, allocator);
        }
        for (_, mut t) in self.retired.drain(..) {
            t.destroy(device, allocator);
        }
        unsafe {
            device.destroy_pipeline(self.mask_all, None);
            device.destroy_pipeline(self.mask_visible, None);
            if let Some((_, p)) = self.composite.take() {
                device.destroy_pipeline(p, None);
            }
            device.destroy_sampler(self.sampler, None);
            device.destroy_pipeline_layout(self.mask_layout, None);
            device.destroy_pipeline_layout(self.composite_layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_shader_module(self.mask_vert, None);
            device.destroy_shader_module(self.mask_frag, None);
            device.destroy_shader_module(self.composite_vert, None);
            device.destroy_shader_module(self.composite_frag, None);
        }
    }
}
