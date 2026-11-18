use crate::pipeline_store::PipelineDeviceExt as _;
use crate::FRAMES_IN_FLIGHT;
use anyhow::Result;
use ash::vk;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct FsrMotionPush {
    pub reproj: [[f32; 4]; 4],
    pub jit: [f32; 4],
    pub size: [i32; 4],
}

pub(crate) struct FsrMotion {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    sampler: vk::Sampler,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
}

impl FsrMotion {
    pub fn new(device: &ash::Device) -> Result<Self> {
        unsafe {
            let binding = |index: u32, kind: vk::DescriptorType| {
                vk::DescriptorSetLayoutBinding::default().binding(index).descriptor_type(kind).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)
            };
            let bindings = [
                binding(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                binding(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
                binding(2, vk::DescriptorType::STORAGE_IMAGE),
            ];
            let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?;
            let push = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(std::mem::size_of::<FsrMotionPush>() as u32)];
            let layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]).push_constant_ranges(&push), None)?;
            let module = crate::screen_pass::shader_module(device, include_bytes!(concat!(env!("OUT_DIR"), "/fsr_motion.comp.spv")))?;
            let built = device.create_cached_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .layout(layout)
                    .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main"))],
                None,
            );
            device.destroy_shader_module(module, None);
            let pipeline = built.map_err(|(_, e)| e)?[0];
            let sampler = device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .min_filter(vk::Filter::NEAREST)
                    .mag_filter(vk::Filter::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?;
            let pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default().max_sets(FRAMES_IN_FLIGHT as u32).pool_sizes(&[
                    vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 2 * FRAMES_IN_FLIGHT as u32 },
                    vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: FRAMES_IN_FLIGHT as u32 },
                ]),
                None,
            )?;
            let sets = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[set_layout; FRAMES_IN_FLIGHT]))?;
            Ok(Self { set_layout, layout, pipeline, sampler, pool, sets })
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record(&self, device: &ash::Device, cb: vk::CommandBuffer, slot: usize, velocity: vk::ImageView, depth: vk::ImageView, output: vk::ImageView, push: &FsrMotionPush) {
        let set = self.sets[slot];
        let sampled = |view: vk::ImageView| [vk::DescriptorImageInfo::default().sampler(self.sampler).image_view(view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        let (velocity_info, depth_info) = (sampled(velocity), sampled(depth));
        let output_info = [vk::DescriptorImageInfo::default().image_view(output).image_layout(vk::ImageLayout::GENERAL)];
        let writes = [
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&velocity_info),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&depth_info),
            vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&output_info),
        ];
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[set], &[]);
            device.cmd_push_constants(cb, self.layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::bytes_of(push));
            device.cmd_dispatch(cb, (push.size[0] as u32).div_ceil(8), (push.size[1] as u32).div_ceil(8), 1);
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_sampler(self.sampler, None);
        }
    }
}
