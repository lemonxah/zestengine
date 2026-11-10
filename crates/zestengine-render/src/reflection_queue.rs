//! Bounded per-view reflection worklists. Reused on the graphics queue with
//! explicit barriers; resized allocations outlive every submitted reader.
use anyhow::Result;
use ash::vk;
use gpu_allocator::{MemoryLocation,vulkan::Allocator};
use crate::{alloc::Buffer,FRAMES_IN_FLIGHT};
const MAX_VIEW_BYTES:u64=128*1024*1024;
struct Target {buffer:Buffer,bytes:u64}
#[derive(Default)]
pub(crate) struct Queue {
    views:[Option<Target>;2],retired:Vec<(u64,Target)>,
}
impl Queue {
    pub fn prepare(&mut self,device:&ash::Device,allocator:&mut Allocator,view:usize,
        serial:u64,extent:vk::Extent2D,samples:u32)->Result<Option<vk::Buffer>> {
        for i in (0..self.retired.len()).rev() {
            if self.retired[i].0<=serial {self.retired.swap_remove(i).1.buffer.destroy(device,allocator);}
        }
        let words=5+samples.clamp(1,16).div_ceil(2);
        let bytes=16+u64::from(extent.width)*u64::from(extent.height)*u64::from(words)*4;
        // Large high-sample views retain the identical inline tracer rather
        // than allocating an unbounded full-resolution ray payload buffer.
        if bytes>MAX_VIEW_BYTES {return Ok(None);}
        if self.views[view].as_ref().is_none_or(|v|v.bytes<bytes) {
            let next=Target {buffer:Buffer::new(device,allocator,bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER|vk::BufferUsageFlags::INDIRECT_BUFFER
                    |vk::BufferUsageFlags::TRANSFER_DST|vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::GpuOnly,"Flux reflection world jobs")?,bytes};
            if let Some(old)=self.views[view].replace(next) {self.retired.push((serial+FRAMES_IN_FLIGHT as u64+1,old));}
        }
        Ok(Some(self.views[view].as_ref().unwrap().buffer.handle))
    }
    pub fn reset(device:&ash::Device,cb:vk::CommandBuffer,buffer:vk::Buffer) {unsafe {
        let barrier=[vk::BufferMemoryBarrier::default().buffer(buffer).size(vk::WHOLE_SIZE)
            .src_access_mask(vk::AccessFlags::SHADER_READ|vk::AccessFlags::SHADER_WRITE|vk::AccessFlags::INDIRECT_COMMAND_READ|vk::AccessFlags::TRANSFER_READ|vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)];
        device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::COMPUTE_SHADER|vk::PipelineStageFlags::DRAW_INDIRECT|vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,vk::DependencyFlags::empty(),&[],&barrier,&[]);
        device.cmd_update_buffer(cb,buffer,0,bytemuck::cast_slice(&[0u32,1,1,0]));
        let ready=[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ|vk::AccessFlags::SHADER_WRITE|vk::AccessFlags::INDIRECT_COMMAND_READ)];
        device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER|vk::PipelineStageFlags::DRAW_INDIRECT,vk::DependencyFlags::empty(),&ready,&[],&[]);
    }}
    #[cfg(test)] pub fn debug_buffer(&self,view:usize)->Option<(vk::Buffer,u64)> {
        self.views[view].as_ref().map(|v|(v.buffer.handle,v.bytes))
    }
    pub fn destroy(&mut self,device:&ash::Device,allocator:&mut Allocator) {
        for v in &mut self.views {if let Some(mut v)=v.take() {v.buffer.destroy(device,allocator);}}
        for (_,mut v) in self.retired.drain(..) {v.buffer.destroy(device,allocator);}
    }
}
