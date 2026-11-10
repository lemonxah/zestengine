//! Exact ray queries against the renderer's actual frame TLAS, independent of
//! surface-cache capture/reconstruction so stale geometry cannot hide there.
use ash::vk;
use crate::{Renderer, alloc::{Buffer, one_time_submit}};
use gpu_allocator::MemoryLocation;

pub(super) struct Probe {
    input: Buffer,
    output: Buffer,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pool: vk::DescriptorPool,
    pipeline: vk::Pipeline,
    set: vk::DescriptorSet,
}

impl Probe {
    pub fn new(r: &Renderer) -> anyhow::Result<Self> {
        let d=&r.ctx.device;let allocator=r.allocator();let mut a=allocator.lock().unwrap();
        let input=Buffer::new(d,&mut a,32*32,vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::CpuToGpu,"ray probes")?;
        let output=Buffer::new(d,&mut a,32*16,vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::GpuToCpu,"ray results")?;
        let types=[vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,vk::DescriptorType::STORAGE_BUFFER,vk::DescriptorType::STORAGE_BUFFER];
        let bindings:Vec<_>=types.iter().enumerate().map(|(i,ty)|vk::DescriptorSetLayoutBinding::default()
            .binding(i as u32).descriptor_type(*ty).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
        unsafe {
            let set_layout=d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),None)?;
            let layout=d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]),None)?;
            let code=ash::util::read_spv(&mut std::io::Cursor::new(include_bytes!(concat!(env!("OUT_DIR"),"/flux_geometry_test.comp.spv"))))?;
            let module=d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code),None)?;
            let pipeline=d.create_compute_pipelines(vk::PipelineCache::null(),&[vk::ComputePipelineCreateInfo::default().layout(layout)
                .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main"))],None).map_err(|(_,e)|e)?[0];
            d.destroy_shader_module(module,None);
            let pool=d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
                vk::DescriptorPoolSize {ty:types[0],descriptor_count:1},vk::DescriptorPoolSize {ty:types[1],descriptor_count:2}]),None)?;
            let set=d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[set_layout]))?[0];
            for (binding,buffer) in [(1,input.handle),(2,output.handle)] {
                let info=[vk::DescriptorBufferInfo::default().buffer(buffer).range(vk::WHOLE_SIZE)];
                d.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding)
                    .descriptor_type(types[1]).buffer_info(&info)],&[]);
            }
            Ok(Self {input,output,set_layout,layout,pool,pipeline,set})
        }
    }

    pub fn trace(&mut self,r:&Renderer,slot:usize,rays:&[[[f32;4];2]])->anyhow::Result<Vec<[f32;4]>> {
        assert!(rays.len()<=32);
        let d=&r.ctx.device;
        self.input.write(0,bytemuck::cast_slice(rays));
        let handles=[r.rt_reflect.as_ref().unwrap().debug_tlas(slot)];
        let mut accel=vk::WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(&handles);
        let write=vk::WriteDescriptorSet::default().dst_set(self.set).dst_binding(0).descriptor_count(1)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR).push_next(&mut accel);
        unsafe {d.update_descriptor_sets(&[write],&[]);}
        one_time_submit(d,r.command_pool,r.ctx.queue,|cb|unsafe {
            let ready=[vk::MemoryBarrier2::default().src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE).dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR|vk::AccessFlags2::SHADER_READ)];
            d.cmd_pipeline_barrier2(cb,&vk::DependencyInfo::default().memory_barriers(&ready));
            d.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,self.pipeline);
            d.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,self.layout,0,&[self.set],&[]);
            d.cmd_dispatch(cb,rays.len() as u32,1,1);
            let host=[vk::MemoryBarrier2::default().src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE).dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ)];
            d.cmd_pipeline_barrier2(cb,&vk::DependencyInfo::default().memory_barriers(&host));
        })?;
        Ok(bytemuck::pod_collect_to_vec(&self.output.read()[..rays.len()*16]))
    }

    pub fn destroy(mut self,r:&Renderer) {
        let d=&r.ctx.device;let allocator=r.allocator();let mut a=allocator.lock().unwrap();
        self.input.destroy(d,&mut a);self.output.destroy(d,&mut a);
        unsafe {d.destroy_pipeline(self.pipeline,None);d.destroy_pipeline_layout(self.layout,None);
            d.destroy_descriptor_pool(self.pool,None);d.destroy_descriptor_set_layout(self.set_layout,None);}
    }
}
