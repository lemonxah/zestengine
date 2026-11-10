//! Software specular world tracing. Persistent per-view/frame descriptors and
//! light uploads reuse GI geometry and surface lighting without a CPU readback.
use crate::pipeline_store::PipelineDeviceExt as _;
use std::sync::Arc;
use anyhow::Result;
use ash::vk;
use bytemuck::{Pod,Zeroable};
use gpu_allocator::{vulkan::Allocator,MemoryLocation};
use crate::{alloc::Buffer,ssr::SsrUniforms,FluxSettings,GpuLight,FRAMES_IN_FLIGHT};
pub(crate) struct Scene {
    pub distance:vk::ImageView,pub materials:Arc<Buffer>,
    pub clipmaps:crate::gpu_gi::clipmaps::Resources,
    pub mesh:Option<Arc<crate::mesh_sdf::Snapshot>>,pub min:[f32;3],pub max:[f32;3],
}
pub(crate) struct Inputs {
    pub scene:Scene,pub surface:Option<crate::flux_surface::Resources>,
    pub lights:Vec<GpuLight>,pub area:vk::ImageView,pub settings:FluxSettings,
}
#[repr(C)]
#[derive(Clone,Copy,Pod,Zeroable)]
struct Params {min:[f32;4],max:[f32;4],counts:[u32;4],march:[f32;4],probe:[f32;4],transport:[f32;4],quality:[f32;4]}
struct Slot {
    pool:vk::DescriptorPool,set:vk::DescriptorSet,trace:Buffer,view:Buffer,lights:Buffer,
    scene:Option<Scene>,light_cap:u64,
}
pub(crate) struct Tracer {
    set_layout:vk::DescriptorSetLayout,layout:vk::PipelineLayout,pipeline:vk::Pipeline,
    nearest:vk::Sampler,linear:vk::Sampler,slots:Vec<Option<Slot>>,
}
fn descriptor_type(b:u32)->vk::DescriptorType {match b {
    2|3|4|12|13|22|23|40=>vk::DescriptorType::STORAGE_BUFFER,
    6=>vk::DescriptorType::STORAGE_IMAGE,7|32|41=>vk::DescriptorType::UNIFORM_BUFFER,
    _=>vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
}}
const BINDINGS:[u32;29]=[0,1,2,3,4,6,7,12,13,14,15,16,17,18,19,22,23,24,31,32,33,34,35,36,37,38,39,40,41];
impl Tracer {
    pub fn new(device:&ash::Device)->Result<Self> {unsafe {
        let bindings:Vec<_>=BINDINGS.into_iter().map(|b|vk::DescriptorSetLayoutBinding::default().binding(b)
            .descriptor_type(descriptor_type(b)).descriptor_count(if b<2 {4} else {1}).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
        let set_layout=device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),None)?;
        let layout=device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]),None)?;
        let module=crate::screen_pass::shader_module(device,include_bytes!(concat!(env!("OUT_DIR"),"/software_reflect.comp.spv")))?;
        let result=device.create_cached_compute_pipelines(vk::PipelineCache::null(),&[vk::ComputePipelineCreateInfo::default().layout(layout)
            .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main"))],None);
        device.destroy_shader_module(module,None);let pipeline=result.map_err(|(_,e)|e)?[0];
        let sampler=|filter|device.create_sampler(&vk::SamplerCreateInfo::default().mag_filter(filter).min_filter(filter)
            .mipmap_mode(if filter==vk::Filter::NEAREST {vk::SamplerMipmapMode::NEAREST} else {vk::SamplerMipmapMode::LINEAR}).max_lod(vk::LOD_CLAMP_NONE)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE).address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),None);
        Ok(Self {set_layout,layout,pipeline,nearest:sampler(vk::Filter::NEAREST)?,linear:sampler(vk::Filter::LINEAR)?,
            slots:(0..2*FRAMES_IN_FLIGHT).map(|_|None).collect()})
    }}
    fn slot(&self,device:&ash::Device,alloc:&mut Allocator,light_bytes:u64)->Result<Slot> {
        let buffer=|alloc:&mut Allocator,size,usage,name|Buffer::new(device,alloc,size,usage,MemoryLocation::CpuToGpu,name);
        let trace=buffer(alloc,std::mem::size_of::<Params>() as u64,vk::BufferUsageFlags::UNIFORM_BUFFER,"Software reflection trace")?;
        let view=buffer(alloc,std::mem::size_of::<SsrUniforms>() as u64,vk::BufferUsageFlags::UNIFORM_BUFFER,"Software reflection view")?;
        let lights=buffer(alloc,light_bytes.max(1024).next_power_of_two(),vk::BufferUsageFlags::STORAGE_BUFFER,"Software reflection lights")?;
        unsafe {
            let sizes=[vk::DescriptorType::COMBINED_IMAGE_SAMPLER,vk::DescriptorType::STORAGE_BUFFER,vk::DescriptorType::STORAGE_IMAGE,vk::DescriptorType::UNIFORM_BUFFER]
                .map(|ty|vk::DescriptorPoolSize {ty,descriptor_count:BINDINGS.iter().filter(|&&b|descriptor_type(b)==ty).map(|&b|if b<2 {4} else {1}).sum()});
            let pool=device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes),None)?;
            let set=device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[self.set_layout]))?[0];
            Ok(Slot {pool,set,trace,view,lights,scene:None,light_cap:light_bytes.max(1024).next_power_of_two()})
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn record(&mut self,device:&ash::Device,alloc:&mut Allocator,cb:vk::CommandBuffer,view:usize,frame:usize,serial:u64,
        inputs:Inputs,u:&SsrUniforms,color:vk::ImageView,gbuf:vk::ImageView,depth:vk::ImageView,
        env:[vk::ImageView;4],hierarchy:vk::Buffer,output:vk::ImageView,extent:vk::Extent2D)->Result<()> {
        let index=view*FRAMES_IN_FLIGHT+frame;
        let light_bytes=(inputs.lights.len().max(1)*std::mem::size_of::<GpuLight>()) as u64;
        if self.slots[index].as_ref().is_none_or(|s|s.light_cap<light_bytes) {
            let new=self.slot(device,alloc,light_bytes)?;
            // The renderer has retired this frame slot's fence before recording.
            if let Some(mut old)=self.slots[index].replace(new) {old.destroy(device,alloc);}
        }
        let slot=self.slots[index].as_mut().unwrap();let scene=&inputs.scene;let fs=inputs.settings;
        let params=Params {min:[scene.min[0],scene.min[1],scene.min[2],0.0],max:[scene.max[0],scene.max[1],scene.max[2],scene.mesh.as_ref().map_or(0,|s|s.node_count) as f32],
            counts:[0,fs.bounces,inputs.lights.len() as u32,fs.reflection_downsample],march:[0.005,fs.ssr_max_distance.max(0.01),0.0,0.0],
            probe:[0.0,0.0,inputs.surface.map_or(0,|s|s.node_count) as f32,0.0],transport:[1.0,if matches!(fs.ray_lighting_mode,crate::RayLightingMode::HitLightingForReflections|crate::RayLightingMode::HitLighting) {1.0} else {0.0},0.0,0.0],
            quality:[fs.rt_samples.clamp(1,16) as f32,fs.rt_bounces.clamp(1,8) as f32,u32::from(fs.reflection_screen_traces) as f32,(serial%4096) as f32]};
        slot.trace.write(0,bytemuck::bytes_of(&params));slot.view.write(0,bytemuck::bytes_of(u));
        if inputs.lights.is_empty() {slot.lights.write(0,&[0;64]);} else {slot.lights.write(0,bytemuck::cast_slice(&inputs.lights));}
        let set=slot.set;
        let buffer=|binding,handle|unsafe {let info=[vk::DescriptorBufferInfo::default().buffer(handle).range(vk::WHOLE_SIZE)];
            device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding).descriptor_type(descriptor_type(binding)).buffer_info(&info)],&[]);};
        let image=|binding,handle,sampler|unsafe {let info=[vk::DescriptorImageInfo::default().image_view(handle).sampler(sampler)
            .image_layout(if binding==6 {vk::ImageLayout::GENERAL} else {vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL})];
            device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding).descriptor_type(descriptor_type(binding)).image_info(&info)],&[]);};
        scene.clipmaps.bind(device,set,41,self.linear,self.nearest);
        buffer(2,scene.materials.handle);buffer(3,slot.lights.handle);buffer(4,scene.materials.handle);
        buffer(7,slot.trace.handle);buffer(32,slot.view.handle);buffer(40,hierarchy);
        let (mesh,voxels)=scene.mesh.as_ref().map_or(([scene.materials.handle;2],scene.distance),|m|m.resources());
        buffer(22,mesh[0]);buffer(23,mesh[1]);image(24,voxels,self.linear);
        buffer(12,inputs.surface.map_or(scene.materials.handle,|s|s.cards));buffer(13,inputs.surface.map_or(scene.materials.handle,|s|s.nodes));
        for i in 0..4 {image(14+i,inputs.surface.map_or(color,|s|s.views[i as usize]),self.nearest);}
        for i in 0..2 {image(18+i,inputs.surface.map_or(color,|s|s.lighting[i as usize]),self.linear);}
        image(31,inputs.area,self.linear);image(33,color,self.linear);image(34,gbuf,self.nearest);image(35,depth,self.nearest);
        for i in 0..4 {image(36+i,env[i as usize],self.linear);}
        image(6,output,self.nearest);
        unsafe {device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,self.pipeline);
            device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,self.layout,0,&[set],&[]);
            device.cmd_dispatch(cb,extent.width.div_ceil(8),extent.height.div_ceil(8),1);}
        crate::flux_surface::finish_feedback(device,cb,inputs.surface);
        slot.scene=Some(inputs.scene);
        Ok(())
    }
    pub fn destroy(&mut self,device:&ash::Device,alloc:&mut Allocator) {
        for s in self.slots.iter_mut().flatten() {s.destroy(device,alloc);}
        unsafe {device.destroy_pipeline(self.pipeline,None);device.destroy_pipeline_layout(self.layout,None);device.destroy_descriptor_set_layout(self.set_layout,None);
            device.destroy_sampler(self.nearest,None);device.destroy_sampler(self.linear,None);}
    }
}
impl Slot {fn destroy(&mut self,device:&ash::Device,alloc:&mut Allocator) {
    self.scene=None;self.trace.destroy(device,alloc);self.view.destroy(device,alloc);self.lights.destroy(device,alloc);
    unsafe {device.destroy_descriptor_pool(self.pool,None);}
}}
