//! Deferred projected-decal pass.
//!
//! Runs fullscreen after the forward pass, one draw per decal. Each decal
//! reconstructs the world position of the opaque surface under it from the
//! scene depth and the forward pass's deferred G-buffer (octahedral normal),
//! projects that position into the decal's local box, and alpha-blends the
//! decal texture onto the lit HDR colour. See `decal.frag`.

use std::collections::HashMap;
use std::io::Cursor;
use std::mem::size_of;

use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::vulkan::Allocator;

use crate::alloc::Buffer;
use crate::frame::FRAMES_IN_FLIGHT;
use crate::post::HDR_FORMAT;
use crate::texture::GpuTexture;
use crate::types::Decal;

const FULLSCREEN_VERT_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/fullscreen.vert.spv"));
const DECAL_FRAG_SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/decal.frag.spv"));

/// Push-constant block the decal fragment shader reads. Sized to the Vulkan
/// guaranteed minimum (128 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct DecalPush {
    pub world_to_local: [[f32; 4]; 4],
    pub size_depth: [f32; 4],  // x width, y height, z depth
    pub fade_tiling: [f32; 4], // x fade, y tiling.x, z tiling.y
    pub tint: [f32; 4],
    pub forward: [f32; 4], // xyz world-space projector +Z
}

const _: () = assert!(size_of::<DecalPush>() == 128);

/// Per-frame uniforms (std140). Mirrors `DecalFrame` in `decal.frag`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct DecalFrame {
    pub inv_view_proj: [[f32; 4]; 4],
    pub inv_view: [[f32; 4]; 4],
}

/// Per-frame-in-flight decal resources: the frame UBO + its descriptor set
/// (depth + gbuffer views are rewritten each time the pass is recorded).
struct FrameRes {
    set: vk::DescriptorSet,
    ubo: Buffer,
}

pub(crate) struct DecalPass {
    frame_set_layout: vk::DescriptorSetLayout,
    decal_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    frame_sampler: vk::Sampler,
    decal_sampler: vk::Sampler,
    frames: Vec<FrameRes>,
    frame_pool: vk::DescriptorPool,
    decal_pool: vk::DescriptorPool,
    decal_sets: HashMap<usize, vk::DescriptorSet>,
}

impl DecalPass {
    pub fn new(device: &ash::Device, allocator: &mut Allocator) -> Result<Self> {
        let sampler_binding = |b: u32| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        };
        let frame_bindings = [
            sampler_binding(0),
            sampler_binding(1),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let frame_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&frame_bindings),
                None,
            )?
        };
        let decal_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[sampler_binding(0)]),
                None,
            )?
        };

        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .size(size_of::<DecalPush>() as u32)];
        let set_layouts = [frame_set_layout, decal_set_layout];
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
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
                    .code(&ash::util::read_spv(&mut Cursor::new(DECAL_FRAG_SPV))?),
                None,
            )?
        };
        let pipeline = Self::build_pipeline(device, pipeline_layout, vert, frag)?;
        unsafe {
            device.destroy_shader_module(vert, None);
            device.destroy_shader_module(frag, None);
        }

        let frame_sampler = unsafe {
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
        let decal_sampler = unsafe {
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

        let decal_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(256).pool_sizes(&[
                    vk::DescriptorPoolSize {
                        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: 256,
                    },
                ]),
                None,
            )?
        };

        let frame_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(FRAMES_IN_FLIGHT as u32)
                    .pool_sizes(&[
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 2 * FRAMES_IN_FLIGHT as u32,
                        },
                        vk::DescriptorPoolSize {
                            ty: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: FRAMES_IN_FLIGHT as u32,
                        },
                    ]),
                None,
            )?
        };

        let mut frames = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for _ in 0..FRAMES_IN_FLIGHT {
            let ubo = Buffer::new(
                device,
                allocator,
                size_of::<DecalFrame>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                gpu_allocator::MemoryLocation::CpuToGpu,
                "decal uniforms",
            )?;
            let set = unsafe {
                device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(frame_pool)
                        .set_layouts(&[frame_set_layout]),
                )?[0]
            };
            let ubo_info = [vk::DescriptorBufferInfo::default()
                .buffer(ubo.handle)
                .offset(0)
                .range(size_of::<DecalFrame>() as u64)];
            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&ubo_info);
            unsafe { device.update_descriptor_sets(&[write], &[]) };
            frames.push(FrameRes { set, ubo });
        }

        // The frame pool stays alive: it owns the sets allocated above.
        Ok(Self {
            frame_set_layout,
            decal_set_layout,
            pipeline_layout,
            pipeline,
            frame_sampler,
            decal_sampler,
            frames,
            frame_pool,
            decal_pool,
            decal_sets: HashMap::new(),
        })
    }

    fn build_pipeline(
        device: &ash::Device,
        layout: vk::PipelineLayout,
        vert: vk::ShaderModule,
        frag: vk::ShaderModule,
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
        // Alpha-blend the decal onto the already-lit scene colour.
        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let color_formats = [HDR_FORMAT];
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

    fn decal_set(
        &mut self,
        device: &ash::Device,
        texture: usize,
        textures: &[GpuTexture],
        default_view: vk::ImageView,
    ) -> vk::DescriptorSet {
        if let Some(&set) = self.decal_sets.get(&texture) {
            return set;
        }
        let set = unsafe {
            device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.decal_pool)
                        .set_layouts(&[self.decal_set_layout]),
                )
                .map(|mut s| s.remove(0))
                .unwrap_or(vk::DescriptorSet::null())
        };
        if set == vk::DescriptorSet::null() {
            return vk::DescriptorSet::null();
        }
        let view = textures.get(texture).map(|t| t.view).unwrap_or(default_view);
        let info = [vk::DescriptorImageInfo::default()
            .sampler(self.decal_sampler)
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&info);
        unsafe { device.update_descriptor_sets(&[write], &[]) };
        self.decal_sets.insert(texture, set);
        set
    }

    /// Record one blended draw per decal into `scene_view`. Inputs `depth_view`
    /// and `gbuf_view` must already be in `SHADER_READ_ONLY_OPTIMAL`; the scene
    /// colour attachment must be in `COLOR_ATTACHMENT_OPTIMAL` (it is loaded).
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        frame_index: usize,
        scene_view: vk::ImageView,
        depth_view: vk::ImageView,
        gbuf_view: vk::ImageView,
        extent: vk::Extent2D,
        frame: &DecalFrame,
        decals: &[Decal],
        textures: &[GpuTexture],
        default_view: vk::ImageView,
    ) {
        if decals.is_empty() {
            return;
        }
        let res = &mut self.frames[frame_index];
        res.ubo.write(0, bytemuck::bytes_of(frame));

        let depth_info = [vk::DescriptorImageInfo::default()
            .sampler(self.frame_sampler)
            .image_view(depth_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let gbuf_info = [vk::DescriptorImageInfo::default()
            .sampler(self.frame_sampler)
            .image_view(gbuf_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(res.set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&depth_info),
            vk::WriteDescriptorSet::default()
                .dst_set(res.set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&gbuf_info),
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };

        unsafe {
            let att = vk::RenderingAttachmentInfo::default()
                .image_view(scene_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE);
            let atts = [att];
            let info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                })
                .layer_count(1)
                .color_attachments(&atts);
            device.cmd_begin_rendering(cb, &info);
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
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
                &[res.set],
                &[],
            );

            for decal in decals {
                let set = self.decal_set(device, decal.texture.0, textures, default_view);
                if set == vk::DescriptorSet::null() {
                    continue;
                }
                let push = DecalPush {
                    world_to_local: decal.world_to_local.to_cols_array_2d(),
                    size_depth: [decal.size[0], decal.size[1], decal.depth, 0.0],
                    fade_tiling: [decal.fade, decal.tiling[0], decal.tiling[1], 0.0],
                    tint: decal.tint,
                    forward: decal.forward.extend(0.0).to_array(),
                };
                device.cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline_layout,
                    1,
                    &[set],
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
            device.cmd_end_rendering(cb);
        }
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            for res in &mut self.frames {
                res.ubo.destroy(device, allocator);
            }
            device.destroy_descriptor_pool(self.frame_pool, None);
            device.destroy_descriptor_pool(self.decal_pool, None);
            device.destroy_sampler(self.frame_sampler, None);
            device.destroy_sampler(self.decal_sampler, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.frame_set_layout, None);
            device.destroy_descriptor_set_layout(self.decal_set_layout, None);
        }
    }
}
