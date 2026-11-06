//! Mip-chain bloom (`bloom.comp`): prefilter the HDR source into a chain of
//! half-resolution levels, downsample to the bottom, then upsample back up
//! blending each level in. The post pass samples level 0 (its set 1).
//!
//! Every target that runs the post pass (game swapchain path, editor viewport,
//! camera preview) owns a [`BloomChain`] sized to it; [`BloomPass`] is shared.

use std::io::Cursor;

use anyhow::Result;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::vulkan::{Allocation, Allocator};

use crate::{PostFx, post::HDR_FORMAT};

const SPV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bloom.comp.spv"));
const MAX_LEVELS: usize = 7;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BloomPush {
    a: [f32; 4],
    b: [f32; 4],
}

pub(crate) struct BloomPass {
    storage_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

struct Level {
    image: vk::Image,
    view: vk::ImageView,
    alloc: Option<Allocation>,
    extent: vk::Extent2D,
    /// Set-0 layout (one sampler): this level as a source.
    sample_set: vk::DescriptorSet,
    /// Storage image: this level as a destination.
    storage_set: vk::DescriptorSet,
}

pub(crate) struct BloomChain {
    levels: Vec<Level>,
    pool: vk::DescriptorPool,
}

impl BloomPass {
    /// `sample_layout` is the post pass's one-sampler set layout (it must be
    /// visible to the compute stage).
    pub fn new(device: &ash::Device, sample_layout: vk::DescriptorSetLayout, exposure_layout: vk::DescriptorSetLayout) -> Result<Self> {
        let binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let storage_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding),
                None,
            )?
        };
        let layouts = [sample_layout, storage_layout, exposure_layout];
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(std::mem::size_of::<BloomPush>() as u32)];
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&push),
                None,
            )?
        };
        let module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default()
                    .code(&ash::util::read_spv(&mut Cursor::new(SPV))?),
                None,
            )?
        };
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let pipeline = unsafe {
            device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(pipeline_layout)],
                    None,
                )
                .map_err(|(_, e)| e)?[0]
        };
        unsafe { device.destroy_shader_module(module, None) };
        Ok(Self { storage_layout, pipeline_layout, pipeline })
    }

    /// A chain for a target of `extent` (level 0 is half resolution).
    pub fn new_chain(
        &self,
        device: &ash::Device,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
        sample_layout: vk::DescriptorSetLayout,
        sampler: vk::Sampler,
    ) -> Result<BloomChain> {
        let mut extents = Vec::new();
        let mut e = vk::Extent2D { width: (extent.width / 2).max(1), height: (extent.height / 2).max(1) };
        while extents.len() < MAX_LEVELS {
            extents.push(e);
            if e.width.min(e.height) < 16 {
                break;
            }
            e = vk::Extent2D { width: (e.width / 2).max(1), height: (e.height / 2).max(1) };
        }
        let n = extents.len() as u32;
        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(n * 2).pool_sizes(&[
                    vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: n },
                    vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: n },
                ]),
                None,
            )?
        };
        let mut levels = Vec::new();
        for (i, ext) in extents.into_iter().enumerate() {
            let (image, alloc) = crate::create_image(
                device,
                allocator,
                HDR_FORMAT,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                ext,
                &format!("bloom level {i}"),
            )?;
            let view = crate::create_view(device, image, HDR_FORMAT, vk::ImageAspectFlags::COLOR)?;
            let sets = unsafe {
                device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&[sample_layout, self.storage_layout]),
                )?
            };
            let sample_info = [vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let storage_info = [vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::GENERAL)];
            unsafe {
                device.update_descriptor_sets(
                    &[
                        vk::WriteDescriptorSet::default()
                            .dst_set(sets[0])
                            .dst_binding(0)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(&sample_info),
                        vk::WriteDescriptorSet::default()
                            .dst_set(sets[1])
                            .dst_binding(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .image_info(&storage_info),
                    ],
                    &[],
                )
            };
            levels.push(Level { image, view, alloc: Some(alloc), extent: ext, sample_set: sets[0], storage_set: sets[1] });
        }
        Ok(BloomChain { levels, pool })
    }

    /// Run the chain from `src_set` (a post source set: binding 0 = the HDR
    /// image in SHADER_READ_ONLY). Always leaves every level in GENERAL so the
    /// post pass's set 1 is valid; the dispatches only run with bloom enabled.
    pub fn record(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        chain: &BloomChain,
        src_set: vk::DescriptorSet,
        post: &PostFx,
        exposure_set: vk::DescriptorSet,
    ) {
        // One batched transition for every level.
        let barriers: Vec<vk::ImageMemoryBarrier2> = chain
            .levels
            .iter()
            .map(|l| {
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(l.image)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    )
            })
            .collect();
        unsafe { device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().image_memory_barriers(&barriers)) };
        if !post.bloom_enabled {
            return;
        }
        // The HDR source was just written as a colour attachment (or by the
        // reflection resolve) and moved to shader-read for the fragment stage.
        memory_barrier(
            device, cb,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags2::FRAGMENT_SHADER
                | vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE | vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_SAMPLED_READ,
        );
        let thr = post.bloom_threshold.max(0.0);
        let exposure = 2f32.powf(post.exposure);
        let scatter = 0.35 + 0.6 * post.bloom_radius.clamp(0.0, 1.0);
        unsafe { device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline) };
        let dispatch = |src: vk::DescriptorSet, dst: &Level, mode: f32| unsafe {
            device.cmd_bind_descriptor_sets(
                cb, vk::PipelineBindPoint::COMPUTE, self.pipeline_layout, 0, &[src, dst.storage_set, exposure_set], &[],
            );
            let push = BloomPush { a: [mode, thr, thr * 0.5, exposure], b: [scatter, if post.auto_exposure.enabled {1.0} else {0.0}, 0.0, 0.0] };
            device.cmd_push_constants(cb, self.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(&push));
            device.cmd_dispatch(cb, dst.extent.width.div_ceil(8), dst.extent.height.div_ceil(8), 1);
            memory_barrier(
                device, cb,
                vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_SAMPLED_READ | vk::AccessFlags2::SHADER_STORAGE_READ
                    | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            );
        };
        let lv = &chain.levels;
        dispatch(src_set, &lv[0], 0.0);
        for i in 0..lv.len() - 1 {
            dispatch(lv[i].sample_set, &lv[i + 1], 1.0);
        }
        for i in (0..lv.len() - 1).rev() {
            dispatch(lv[i + 1].sample_set, &lv[i], 2.0);
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.storage_layout, None);
        }
    }
}

impl BloomChain {
    /// The set the post pass binds as set 1 (level 0, sampled).
    pub fn output_set(&self) -> vk::DescriptorSet {
        self.levels[0].sample_set
    }

    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for l in &mut self.levels {
            unsafe {
                device.destroy_image_view(l.view, None);
                device.destroy_image(l.image, None);
            }
            if let Some(a) = l.alloc.take() {
                let _ = allocator.free(a);
            }
        }
        self.levels.clear();
        unsafe { device.destroy_descriptor_pool(self.pool, None) };
    }
}

fn memory_barrier(
    device: &ash::Device,
    cb: vk::CommandBuffer,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
) {
    let b = [vk::MemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)];
    unsafe { device.cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&b)) };
}
