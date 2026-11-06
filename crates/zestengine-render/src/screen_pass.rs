//! Small shared helpers for fullscreen (one triangle, `fullscreen.vert`)
//! fragment passes that render into a single colour attachment with dynamic
//! rendering: the screen-space AO passes (`ao.rs`) and the anti-aliasing
//! resolves (`aa.rs`).

use std::io::Cursor;

use anyhow::Result;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, Allocator};

const FULLSCREEN_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fullscreen.vert.spv"));

pub(crate) fn shader_module(device: &ash::Device, spv: &[u8]) -> Result<vk::ShaderModule> {
    let code = ash::util::read_spv(&mut Cursor::new(spv))?;
    Ok(unsafe { device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)? })
}

/// A fullscreen-triangle pipeline writing `format`, no depth, no blending,
/// dynamic viewport + scissor. `spec` = (constant id, value) pairs for the
/// fragment stage.
pub(crate) fn fullscreen_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    frag_spv: &[u8],
    format: vk::Format,
    spec: &[(u32, i32)],
) -> Result<vk::Pipeline> {
    let vert = shader_module(device, FULLSCREEN_VERT_SPV)?;
    let frag = shader_module(device, frag_spv)?;
    let entries: Vec<vk::SpecializationMapEntry> = spec
        .iter()
        .enumerate()
        .map(|(i, (id, _))| vk::SpecializationMapEntry { constant_id: *id, offset: (i * 4) as u32, size: 4 })
        .collect();
    let data: Vec<u8> = spec.iter().flat_map(|(_, v)| v.to_ne_bytes()).collect();
    let spec_info = vk::SpecializationInfo::default().map_entries(&entries).data(&data);
    let mut frag_stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(frag).name(c"main");
    if !spec.is_empty() {
        frag_stage = frag_stage.specialization_info(&spec_info);
    }
    let stages = [vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vert).name(c"main"), frag_stage];
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
    let blend_attachments = [vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA)];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let color_formats = [format];
    let mut rendering = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);
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
    let result = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) };
    unsafe {
        device.destroy_shader_module(vert, None);
        device.destroy_shader_module(frag, None);
    }
    Ok(result.map_err(|(_, e)| e)?[0])
}

/// Record one fullscreen draw into `view` (`extent`), which must already be in
/// COLOR_ATTACHMENT_OPTIMAL. Binds `pipeline`, `sets` at set 0.. and the push
/// constant bytes for the fragment stage.
#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_fullscreen(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    view: vk::ImageView,
    extent: vk::Extent2D,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    sets: &[vk::DescriptorSet],
    push: &[u8],
) {
    let att = [vk::RenderingAttachmentInfo::default()
        .image_view(view)
        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)];
    let area = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent };
    let info = vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&att);
    unsafe {
        device.cmd_begin_rendering(cb, &info);
        device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
        let viewport = vk::Viewport { x: 0.0, y: 0.0, width: extent.width as f32, height: extent.height as f32, min_depth: 0.0, max_depth: 1.0 };
        device.cmd_set_viewport(cb, 0, &[viewport]);
        device.cmd_set_scissor(cb, 0, &[area]);
        if !sets.is_empty() {
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, layout, 0, sets, &[]);
        }
        if !push.is_empty() {
            device.cmd_push_constants(cb, layout, vk::ShaderStageFlags::FRAGMENT, 0, push);
        }
        device.cmd_draw(cb, 3, 1, 0, 0);
        device.cmd_end_rendering(cb);
    }
}

/// A GPU-only 2D image with `mips` levels, its all-levels view and one view
/// per level (for rendering into / sampling a single level).
pub(crate) struct ScreenImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub level_views: Vec<vk::ImageView>,
    pub extent: vk::Extent2D,
    pub mips: u32,
    alloc: Option<Allocation>,
}

impl ScreenImage {
    pub fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        extent: vk::Extent2D,
        mips: u32,
        name: &str,
    ) -> Result<Self> {
        let (image, alloc) = crate::create_image_mips(device, allocator, format, usage, extent, mips, name)?;
        let view = crate::create_view_mips(device, image, format, vk::ImageAspectFlags::COLOR, mips)?;
        let mut level_views = Vec::new();
        if mips > 1 {
            for level in 0..mips {
                let info = vk::ImageViewCreateInfo::default().image(image).view_type(vk::ImageViewType::TYPE_2D).format(format).subresource_range(
                    vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).base_mip_level(level).level_count(1).layer_count(1),
                );
                level_views.push(unsafe { device.create_image_view(&info, None)? });
            }
        } else {
            level_views.push(view);
        }
        Ok(Self { image, view, level_views, extent, mips, alloc: Some(alloc) })
    }

    /// The view of one mip level (the image's own view when it has one level).
    pub fn level(&self, level: u32) -> vk::ImageView {
        self.level_views[level as usize]
    }

    pub fn level_extent(&self, level: u32) -> vk::Extent2D {
        vk::Extent2D { width: (self.extent.width >> level).max(1), height: (self.extent.height >> level).max(1) }
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            if self.mips > 1 {
                for v in self.level_views.drain(..) {
                    device.destroy_image_view(v, None);
                }
            }
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }
        if let Some(a) = self.alloc.take() {
            let _ = allocator.free(a);
        }
    }
}

/// A combined-image-sampler descriptor write.
pub(crate) fn sampled(set: vk::DescriptorSet, binding: u32, info: &[vk::DescriptorImageInfo; 1]) -> vk::WriteDescriptorSet<'_> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(info)
}

pub(crate) fn read_info(sampler: vk::Sampler, view: vk::ImageView) -> [vk::DescriptorImageInfo; 1] {
    [vk::DescriptorImageInfo::default().sampler(sampler).image_view(view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
}

/// Clamp-to-edge sampler; `linear` filters (and blends mips), else point.
pub(crate) fn clamp_sampler(device: &ash::Device, linear: bool) -> Result<vk::Sampler> {
    let f = if linear { vk::Filter::LINEAR } else { vk::Filter::NEAREST };
    let m = if linear { vk::SamplerMipmapMode::LINEAR } else { vk::SamplerMipmapMode::NEAREST };
    Ok(unsafe {
        device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(f)
                .min_filter(f)
                .mipmap_mode(m)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .max_lod(vk::LOD_CLAMP_NONE),
            None,
        )?
    })
}

/// A descriptor set layout of `n` fragment-stage combined image samplers
/// (bindings 0..n).
pub(crate) fn samplers_layout(device: &ash::Device, n: u32) -> Result<vk::DescriptorSetLayout> {
    let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..n)
        .map(|b| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        })
        .collect();
    Ok(unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)? })
}
