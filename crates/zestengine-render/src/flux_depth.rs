//! Current-view min/max depth hierarchy, shared by screen ray consumers.
//! Half-resolution and coarser levels preserve odd edge texels; leaf planes
//! are reconstructed directly from depth, avoiding a full-resolution copy. Each view owns its buffer;
//! frame-slot descriptor sets and queue barriers protect overlapping submissions.
use anyhow::Result;
use ash::vk;
use gpu_allocator::{MemoryLocation,vulkan::Allocator};
use crate::{alloc::Buffer,FRAMES_IN_FLIGHT};

pub(crate) struct DepthPyramid {
    pub buffer:Buffer,
    levels:[[u32;4];32],
    sets:Vec<vk::DescriptorSet>,
    pool:vk::DescriptorPool,
}
pub(crate) struct DepthReduce {
    layout:vk::PipelineLayout,
    set_layout:vk::DescriptorSetLayout,
    pipeline:vk::Pipeline,
    sampler:vk::Sampler,
}
impl DepthReduce {
    pub fn new(device:&ash::Device)->Result<Self> {
        unsafe {
            let bindings=[vk::DescriptorSetLayoutBinding::default().binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)];
            let set_layout=device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),None)?;
            let push=[vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(32)];
            let layout=device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]).push_constant_ranges(&push),None)?;
            let module=crate::screen_pass::shader_module(device,include_bytes!(concat!(env!("OUT_DIR"),"/flux_depth_reduce.comp.spv")))?;
            let built=device.create_compute_pipelines(vk::PipelineCache::null(),&[vk::ComputePipelineCreateInfo::default().layout(layout)
                .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main"))],None);
            device.destroy_shader_module(module,None);
            let pipeline=built.map_err(|(_,e)|e)?[0];
            let sampler=device.create_sampler(&vk::SamplerCreateInfo::default().min_filter(vk::Filter::NEAREST).mag_filter(vk::Filter::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE).address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),None)?;
            Ok(Self {layout,set_layout,pipeline,sampler})
        }
    }
    pub fn target(&self,device:&ash::Device,alloc:&mut Allocator,extent:vk::Extent2D)->Result<DepthPyramid> {
        let mut levels=[[0u32;4];32];let mut size=[extent.width.max(1),extent.height.max(1)];
        let mut offset=0u32;let mut count=0;
        loop {
            levels[count]=[size[0],size[1],offset,0];count+=1;
            if count>1 {offset+=size[0]*size[1];}
            if size==[1,1] {break;}
            size=size.map(|v|v.div_ceil(2));
        }
        levels[0][3]=count as u32;
        let buffer=Buffer::new(device,alloc,512+offset as u64*8,
            vk::BufferUsageFlags::STORAGE_BUFFER|vk::BufferUsageFlags::TRANSFER_DST|vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuOnly,"Flux depth hierarchy")?;
        unsafe {
            let pool=device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(FRAMES_IN_FLIGHT as u32).pool_sizes(&[
                vk::DescriptorPoolSize {ty:vk::DescriptorType::COMBINED_IMAGE_SAMPLER,descriptor_count:FRAMES_IN_FLIGHT as u32},
                vk::DescriptorPoolSize {ty:vk::DescriptorType::STORAGE_BUFFER,descriptor_count:FRAMES_IN_FLIGHT as u32}]),None)?;
            let sets=device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&vec![self.set_layout;FRAMES_IN_FLIGHT]))?;
            for &set in &sets {
                let info=[vk::DescriptorBufferInfo::default().buffer(buffer.handle).range(vk::WHOLE_SIZE)];
                device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info)],&[]);
            }
            Ok(DepthPyramid {buffer,levels,sets,pool})
        }
    }
    pub fn record(&self,device:&ash::Device,cb:vk::CommandBuffer,target:&DepthPyramid,depth:vk::ImageView,inverse_projection:[[f32;4];4],frame:usize) {
        unsafe {
            let info=[vk::DescriptorImageInfo::default().sampler(self.sampler).image_view(depth).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let set=target.sets[frame];
            device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&info)],&[]);
            let reuse=[vk::BufferMemoryBarrier::default().buffer(target.buffer.handle).size(vk::WHOLE_SIZE)
                .src_access_mask(vk::AccessFlags::SHADER_READ|vk::AccessFlags::SHADER_WRITE|vk::AccessFlags::TRANSFER_READ|vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE|vk::AccessFlags::SHADER_WRITE)];
            device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::COMPUTE_SHADER|vk::PipelineStageFlags::FRAGMENT_SHADER|vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER|vk::PipelineStageFlags::COMPUTE_SHADER,vk::DependencyFlags::empty(),&[],&reuse,&[]);
            device.cmd_update_buffer(cb,target.buffer.handle,0,bytemuck::cast_slice(&target.levels));
            let init=[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ)];
            device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::TRANSFER,vk::PipelineStageFlags::COMPUTE_SHADER|vk::PipelineStageFlags::FRAGMENT_SHADER,vk::DependencyFlags::empty(),&init,&[],&[]);
            device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,self.pipeline);
            device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,self.layout,0,&[set],&[]);
            let ready=[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ)];
            for level in 1..target.levels[0][3] {
                device.cmd_push_constants(cb,self.layout,vk::ShaderStageFlags::COMPUTE,0,bytemuck::cast_slice(&[level,0,0,0,
                    inverse_projection[2][2].to_bits(),inverse_projection[3][2].to_bits(),inverse_projection[2][3].to_bits(),inverse_projection[3][3].to_bits()]));
                let size=target.levels[level as usize];
                device.cmd_dispatch(cb,size[0].div_ceil(8),size[1].div_ceil(8),1);
                device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER|vk::PipelineStageFlags::FRAGMENT_SHADER,vk::DependencyFlags::empty(),&ready,&[],&[]);
            }
        }
    }
    pub fn destroy(&mut self,device:&ash::Device) {unsafe {
        device.destroy_pipeline(self.pipeline,None);device.destroy_pipeline_layout(self.layout,None);
        device.destroy_descriptor_set_layout(self.set_layout,None);device.destroy_sampler(self.sampler,None);
    }}
}
impl DepthPyramid {
    #[cfg(test)]
    pub fn byte_len(&self)->u64 {if self.levels[0][3]==1 {512} else {512+(self.levels[self.levels[0][3] as usize-1][2] as u64+1)*8}}

    pub fn destroy(&mut self,device:&ash::Device,alloc:&mut Allocator) {
        unsafe {device.destroy_descriptor_pool(self.pool,None);}
        self.buffer.destroy(device,alloc);
    }
}
