//! Automatic directional radiance clipmaps. GPU demand/age selection and fixed
//! trace budgets keep CPU work independent of scene size; world tags preserve
//! overlapping cells when views move. Scene, Camera and game own their histories.
use anyhow::Result;
use ash::vk;
use bytemuck::{Pod,Zeroable};
use glam::Vec3;
use gpu_allocator::{MemoryLocation,vulkan::Allocator};
use crate::{alloc::Buffer,FRAMES_IN_FLIGHT};
pub(crate) const PROBES:u32=3*16*16*16;
pub(crate) const PROBE_BYTES:u64=32+256*8;
pub(crate) const MAX_UPDATES:u32=512;
const WORK_BYTES:u64=(48+MAX_UPDATES as u64+PROBES as u64)*4;
#[repr(C)]
#[derive(Clone,Copy,Pod,Zeroable)]
struct Params {origin:[[i32;4];3],spacing:[f32;4],config:[u32;4],camera:[f32;4]}
struct View {
    data:Buffer,work:Buffer,params:Vec<Buffer>,sets:Vec<vk::DescriptorSet>,pool:vk::DescriptorPool,
    key:Option<u64>,epoch:u32,last_serial:u64,initialized:bool,
}
#[derive(Clone,Copy)]
pub(crate) struct Resources {pub data:vk::Buffer,pub work:vk::Buffer,pub params:vk::Buffer,pub enabled:bool,pub full_interpolation:bool}
pub(crate) struct Cache {
    layout:vk::PipelineLayout,set_layout:vk::DescriptorSetLayout,pipeline:vk::Pipeline,views:[Option<View>;3],
    #[cfg(test)] pub enabled_override:Option<bool>,
    #[cfg(test)] pub full_interpolation_override:Option<bool>,
    #[cfg(test)] pub adaptive_resolution_override:Option<bool>,
    #[cfg(test)] pub clipmap_blending_override:Option<bool>,
}
pub(crate) fn barrier(device:&ash::Device,cb:vk::CommandBuffer) {unsafe {
    device.cmd_pipeline_barrier2(cb,&vk::DependencyInfo::default().memory_barriers(&[vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS).src_access_mask(vk::AccessFlags2::MEMORY_READ|vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS).dst_access_mask(vk::AccessFlags2::MEMORY_READ|vk::AccessFlags2::MEMORY_WRITE)]));
}}
impl Cache {
    pub fn new(device:&ash::Device)->Result<Self> {unsafe {
        let bindings=[vk::DescriptorType::STORAGE_BUFFER,vk::DescriptorType::STORAGE_BUFFER,vk::DescriptorType::UNIFORM_BUFFER]
            .into_iter().enumerate().map(|(i,t)|vk::DescriptorSetLayoutBinding::default().binding(i as u32).descriptor_type(t)
                .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect::<Vec<_>>();
        let set_layout=device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),None)?;
        let layout=device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout])
            .push_constant_ranges(&[vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(4)]),None)?;
        let module=crate::screen_pass::shader_module(device,include_bytes!(concat!(env!("OUT_DIR"),"/flux_radiance_manage.comp.spv")))?;
        let result=device.create_compute_pipelines(vk::PipelineCache::null(),&[vk::ComputePipelineCreateInfo::default().layout(layout)
            .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main"))],None);
        device.destroy_shader_module(module,None);
        Ok(Self {layout,set_layout,pipeline:result.map_err(|(_,e)|e)?[0],views:std::array::from_fn(|_|None),
            #[cfg(test)] enabled_override:None,
            #[cfg(test)] full_interpolation_override:None,
            #[cfg(test)] adaptive_resolution_override:None,
            #[cfg(test)] clipmap_blending_override:None})
    }}
    fn view(&self,device:&ash::Device,alloc:&mut Allocator)->Result<View> {
        let data=Buffer::new(device,alloc,PROBES as u64*PROBE_BYTES,vk::BufferUsageFlags::STORAGE_BUFFER|vk::BufferUsageFlags::TRANSFER_SRC|vk::BufferUsageFlags::TRANSFER_DST,MemoryLocation::GpuOnly,"Flux directional world radiance")?;
        let work=Buffer::new(device,alloc,WORK_BYTES,vk::BufferUsageFlags::STORAGE_BUFFER|vk::BufferUsageFlags::INDIRECT_BUFFER|vk::BufferUsageFlags::TRANSFER_SRC|vk::BufferUsageFlags::TRANSFER_DST,MemoryLocation::GpuOnly,"Flux radiance update queue")?;
        let mut params=Vec::new();for _ in 0..FRAMES_IN_FLIGHT {params.push(Buffer::new(device,alloc,std::mem::size_of::<Params>() as u64,vk::BufferUsageFlags::UNIFORM_BUFFER,MemoryLocation::CpuToGpu,"Flux radiance clipmaps")?);}
        unsafe {
            let pool=device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(FRAMES_IN_FLIGHT as u32).pool_sizes(&[
                vk::DescriptorPoolSize {ty:vk::DescriptorType::STORAGE_BUFFER,descriptor_count:(2*FRAMES_IN_FLIGHT) as u32},
                vk::DescriptorPoolSize {ty:vk::DescriptorType::UNIFORM_BUFFER,descriptor_count:FRAMES_IN_FLIGHT as u32}]),None)?;
            let sets=device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&vec![self.set_layout;FRAMES_IN_FLIGHT]))?;
            for (i,&set) in sets.iter().enumerate() {
                for (binding,buffer) in [data.handle,work.handle,params[i].handle].into_iter().enumerate() {
                    let info=[vk::DescriptorBufferInfo::default().buffer(buffer).range(vk::WHOLE_SIZE)];
                    device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding as u32)
                        .descriptor_type(if binding==2 {vk::DescriptorType::UNIFORM_BUFFER} else {vk::DescriptorType::STORAGE_BUFFER}).buffer_info(&info)],&[]);
                }
            }
            Ok(View {data,work,params,sets,pool,key:None,epoch:0,last_serial:0,initialized:false})
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(&mut self,device:&ash::Device,alloc:&mut Allocator,cb:vk::CommandBuffer,view:usize,frame:usize,serial:u64,
        camera:Vec3,key:u64,budget:u32,enabled:bool)->Result<Resources> {
        static ENABLED:std::sync::OnceLock<bool>=std::sync::OnceLock::new();
        let enabled=enabled && *ENABLED.get_or_init(||!std::env::var("ZESTENGINE_FLUX_RADIANCE_CACHE").is_ok_and(|v|v=="0"));
        #[cfg(test)] let enabled=enabled && self.enabled_override.unwrap_or(true);
        static FULL_INTERPOLATION:std::sync::OnceLock<bool>=std::sync::OnceLock::new();
        let full=*FULL_INTERPOLATION.get_or_init(||std::env::var("ZESTENGINE_FLUX_RADIANCE_INTERPOLATION").is_ok_and(|v|v=="full"));
        #[cfg(test)] let full=self.full_interpolation_override.unwrap_or(full);
        let adaptive=true;
        #[cfg(test)] let adaptive=self.adaptive_resolution_override.unwrap_or(adaptive);
        let blend=true;
        #[cfg(test)] let blend=self.clipmap_blending_override.unwrap_or(blend);
        if self.views[view].is_none() {self.views[view]=Some(self.view(device,alloc)?);}
        let target=self.views[view].as_mut().unwrap();
        if target.key!=Some(key) || target.last_serial+1!=serial {
            target.epoch=target.epoch.wrapping_add(1).max(1);target.key=Some(key);
        }
        target.last_serial=serial;
        let spacing=[1.0,4.0,16.0,3.0f32.sqrt()];
        let origin=std::array::from_fn(|i|{let p=(camera/spacing[i]).floor().as_ivec3()-glam::IVec3::splat(8);[p.x,p.y,p.z,0]});
        target.params[frame].write(0,bytemuck::bytes_of(&Params {origin,spacing,config:[serial as u32,target.epoch,budget.clamp(16,128),if enabled {1|((cfg!(test) as u32)<<1)|((adaptive as u32)<<2)|((blend as u32)<<3)} else {0}],camera:[camera.x,camera.y,camera.z,40.0]}));
        barrier(device,cb);
        unsafe {
            if !target.initialized {device.cmd_fill_buffer(cb,target.data.handle,0,vk::WHOLE_SIZE,0);target.initialized=true;}
            device.cmd_fill_buffer(cb,target.work.handle,0,WORK_BYTES,0);
        }
        barrier(device,cb);
        if enabled {unsafe {
            device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,self.pipeline);
            device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,self.layout,0,&[target.sets[frame]],&[]);
            for phase in 0u32..3 {
                device.cmd_push_constants(cb,self.layout,vk::ShaderStageFlags::COMPUTE,0,bytemuck::bytes_of(&phase));
                device.cmd_dispatch(cb,if phase==1 {1} else {PROBES.div_ceil(64)},1,1);barrier(device,cb);
            }
        }}
        Ok(Resources {data:target.data.handle,work:target.work.handle,params:target.params[frame].handle,enabled,full_interpolation:full})
    }
    #[cfg(test)] pub fn debug(&self,view:usize)->(vk::Buffer,vk::Buffer,u32) {
        let t=self.views[view].as_ref().unwrap();(t.data.handle,t.work.handle,t.epoch)
    }
    pub fn invalidate(&mut self) {for view in self.views.iter_mut().flatten() {view.key=None;}}
    pub fn destroy(&mut self,device:&ash::Device,alloc:&mut Allocator) {unsafe {
        for target in &mut self.views {if let Some(mut t)=target.take() {device.destroy_descriptor_pool(t.pool,None);
            t.data.destroy(device,alloc);t.work.destroy(device,alloc);for p in &mut t.params {p.destroy(device,alloc);}}}
        device.destroy_pipeline(self.pipeline,None);device.destroy_pipeline_layout(self.layout,None);device.destroy_descriptor_set_layout(self.set_layout,None);
    }}
}
