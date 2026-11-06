//! Per-view reflection reconstruction. Histories contain only the signed
//! replacement for forward environment specular, never the scene's diffuse HDR.
use anyhow::Result;
use ash::vk;
use bytemuck::{Pod,Zeroable};
use glam::Mat4;
use gpu_allocator::{vulkan::Allocator,MemoryLocation};
use crate::{alloc::Buffer,screen_pass::ScreenImage,rt_reflect::RtUniforms,FRAMES_IN_FLIGHT};
#[repr(C)]
#[derive(Clone,Copy,Pod,Zeroable)]
struct Params {
    inverse_view_projection:[[f32;4];4],inverse_view:[[f32;4];4],
    previous_view_projection:[[f32;4];4],previous_inverse_view_projection:[[f32;4];4],
    camera:[f32;4],options:[f32;4],
}
#[derive(Clone,Copy)]
struct Capture {serial:u64,vp:Mat4,key:[f32;8]}
struct Target {
    raw:ScreenImage,color:[ScreenImage;2],geometry:[ScreenImage;2],
    pool:vk::DescriptorPool,sets:Vec<vk::DescriptorSet>,uniforms:Vec<Buffer>,last:Option<Capture>,initialized:bool,
}
pub(crate) struct Denoiser {
    views:[Option<Target>;2],retired:Vec<(u64,Target)>,set_layout:vk::DescriptorSetLayout,
    layout:vk::PipelineLayout,pipeline:vk::Pipeline,sampler:vk::Sampler,
}
impl Denoiser {
    pub fn new(device:&ash::Device)->Result<Self> {unsafe {
        let bindings:Vec<_>=(0..10).map(|binding|vk::DescriptorSetLayoutBinding::default().binding(binding)
            .descriptor_type(if binding<6 {vk::DescriptorType::COMBINED_IMAGE_SAMPLER} else if binding<9 {vk::DescriptorType::STORAGE_IMAGE} else {vk::DescriptorType::UNIFORM_BUFFER})
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
        let set_layout=device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),None)?;
        let layout=device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]),None)?;
        let shader=crate::screen_pass::shader_module(device,include_bytes!(concat!(env!("OUT_DIR"),"/reflection_denoise.comp.spv")))?;
        let built=device.create_compute_pipelines(vk::PipelineCache::null(),&[vk::ComputePipelineCreateInfo::default().layout(layout)
            .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(shader).name(c"main"))],None);
        device.destroy_shader_module(shader,None);let pipeline=built.map_err(|(_,e)|e)?[0];
        let sampler=device.create_sampler(&vk::SamplerCreateInfo::default().mag_filter(vk::Filter::NEAREST).min_filter(vk::Filter::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE).address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),None)?;
        Ok(Self {views:[None,None],retired:Vec::new(),set_layout,layout,pipeline,sampler})
    }}
    fn target(&self,device:&ash::Device,alloc:&mut Allocator,extent:vk::Extent2D)->Result<Target> {
        let mut image=|format,name|ScreenImage::new(device,alloc,format,
            vk::ImageUsageFlags::SAMPLED|vk::ImageUsageFlags::STORAGE|vk::ImageUsageFlags::TRANSFER_DST|vk::ImageUsageFlags::TRANSFER_SRC,extent,1,name);
        let raw=image(vk::Format::R16G16B16A16_SFLOAT,"RT reflection samples")?;
        let color=[image(vk::Format::R16G16B16A16_SFLOAT,"RT reflection history A")?,image(vk::Format::R16G16B16A16_SFLOAT,"RT reflection history B")?];
        let geometry=[image(vk::Format::R32G32B32A32_SFLOAT,"RT reflection geometry A")?,image(vk::Format::R32G32B32A32_SFLOAT,"RT reflection geometry B")?];
        let mut uniforms=Vec::new();for _ in 0..FRAMES_IN_FLIGHT {uniforms.push(Buffer::new(device,alloc,std::mem::size_of::<Params>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,MemoryLocation::CpuToGpu,"RT denoise parameters")?);}
        unsafe {
            let pool=device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(FRAMES_IN_FLIGHT as u32).pool_sizes(&[
                vk::DescriptorPoolSize {ty:vk::DescriptorType::COMBINED_IMAGE_SAMPLER,descriptor_count:6*FRAMES_IN_FLIGHT as u32},
                vk::DescriptorPoolSize {ty:vk::DescriptorType::STORAGE_IMAGE,descriptor_count:3*FRAMES_IN_FLIGHT as u32},
                vk::DescriptorPoolSize {ty:vk::DescriptorType::UNIFORM_BUFFER,descriptor_count:FRAMES_IN_FLIGHT as u32}]),None)?;
            let sets=device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&vec![self.set_layout;FRAMES_IN_FLIGHT]))?;
            Ok(Target {raw,color,geometry,pool,sets,uniforms,last:None,initialized:false})
        }
    }
    pub fn prepare(&mut self,device:&ash::Device,alloc:&mut Allocator,cb:vk::CommandBuffer,view:usize,serial:u64,extent:vk::Extent2D)->Result<(vk::Image,vk::ImageView)> {
        for i in (0..self.retired.len()).rev() {if self.retired[i].0<=serial {self.retired.swap_remove(i).1.destroy(device,alloc);}}
        if self.views[view].as_ref().is_none_or(|v|v.raw.extent!=extent) {
            let next=self.target(device,alloc,extent)?;
            if let Some(old)=self.views[view].replace(next) {self.retired.push((serial+FRAMES_IN_FLIGHT as u64+1,old));}
        }
        let v=self.views[view].as_mut().unwrap();
        if !v.initialized {
            let range=vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
            for image in v.color.iter().chain(v.geometry.iter()) {
                barrier(device,cb,image.image,vk::ImageLayout::UNDEFINED,vk::ImageLayout::TRANSFER_DST_OPTIMAL);
                unsafe {device.cmd_clear_color_image(cb,image.image,vk::ImageLayout::TRANSFER_DST_OPTIMAL,&vk::ClearColorValue {float32:[0.0;4]},&[range]);}
                barrier(device,cb,image.image,vk::ImageLayout::TRANSFER_DST_OPTIMAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            }
            v.initialized=true;
        }
        // The ray pass overwrites this image completely. Previous-frame reads
        // finish before the next write on the same graphics queue.
        barrier(device,cb,v.raw.image,vk::ImageLayout::UNDEFINED,vk::ImageLayout::GENERAL);
        Ok((v.raw.image,v.raw.view))
    }
    #[allow(clippy::too_many_arguments)]
    pub fn record(&mut self,device:&ash::Device,cb:vk::CommandBuffer,view:usize,frame:usize,serial:u64,
        color:vk::ImageView,depth:vk::ImageView,gbuf:vk::ImageView,out_image:vk::Image,out_view:vk::ImageView,u:&RtUniforms,enabled:bool) {
        let v=self.views[view].as_mut().unwrap();let cur=(serial&1) as usize;let prev=cur^1;
        let inverse_vp=Mat4::from_cols_array_2d(&u.inv_view)*Mat4::from_cols_array_2d(&u.inv_proj);let vp=inverse_vp.inverse();
        let key=[u.params[0],u.params[1],u.params[3],u.quality[0],u.quality[1],u.quality[2],u32::from(enabled) as f32,u.lighting_mode[0] as f32];
        let valid=v.last.is_some_and(|old|old.serial+1==serial && old.key==key);
        let previous=v.last.map_or(vp,|old|old.vp);
        let params=Params {inverse_view_projection:inverse_vp.to_cols_array_2d(),inverse_view:u.inv_view,
            previous_view_projection:previous.to_cols_array_2d(),previous_inverse_view_projection:previous.inverse().to_cols_array_2d(),
            camera:u.camera_pos,options:[u32::from(valid) as f32,u32::from(enabled) as f32,16.0,0.002]};
        v.uniforms[frame].write(0,bytemuck::bytes_of(&params));let set=v.sets[frame];
        unsafe {
            for (binding,image) in [v.raw.view,color,depth,gbuf,v.color[prev].view,v.geometry[prev].view,out_view,v.color[cur].view,v.geometry[cur].view].into_iter().enumerate() {
                let storage=binding>=6;let info=[vk::DescriptorImageInfo::default().sampler(self.sampler).image_view(image)
                    .image_layout(if storage {vk::ImageLayout::GENERAL} else {vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL})];
                device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding as u32)
                    .descriptor_type(if storage {vk::DescriptorType::STORAGE_IMAGE} else {vk::DescriptorType::COMBINED_IMAGE_SAMPLER}).image_info(&info)],&[]);
            }
            let info=[vk::DescriptorBufferInfo::default().buffer(v.uniforms[frame].handle).range(vk::WHOLE_SIZE)];
            device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(9).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&info)],&[]);
        }
        barrier(device,cb,v.raw.image,vk::ImageLayout::GENERAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        barrier(device,cb,out_image,vk::ImageLayout::UNDEFINED,vk::ImageLayout::GENERAL);
        for image in [v.color[cur].image,v.geometry[cur].image] {barrier(device,cb,image,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,vk::ImageLayout::GENERAL);}
        unsafe {
            device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,self.pipeline);
            device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,self.layout,0,&[set],&[]);
            device.cmd_dispatch(cb,v.raw.extent.width.div_ceil(8),v.raw.extent.height.div_ceil(8),1);
        }
        for image in [v.color[cur].image,v.geometry[cur].image] {barrier(device,cb,image,vk::ImageLayout::GENERAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);}
        v.last=Some(Capture {serial,vp,key});
    }
    #[cfg(test)]
    pub fn debug_history(&self,view:usize)->(vk::Image,vk::Image,vk::Extent2D,Mat4,u64) {
        let v=self.views[view].as_ref().unwrap();let last=v.last.unwrap();let cur=(last.serial&1) as usize;
        (v.color[cur].image,v.geometry[cur].image,v.raw.extent,last.vp,last.serial)
    }
    pub fn invalidate(&mut self) {for v in self.views.iter_mut().flatten() {v.last=None;}}
    pub fn destroy(&mut self,device:&ash::Device,alloc:&mut Allocator) {
        for v in &mut self.views {if let Some(mut v)=v.take() {v.destroy(device,alloc);}}
        for (_,mut v) in self.retired.drain(..) {v.destroy(device,alloc);}
        unsafe {device.destroy_descriptor_set_layout(self.set_layout,None);device.destroy_pipeline_layout(self.layout,None);device.destroy_pipeline(self.pipeline,None);device.destroy_sampler(self.sampler,None);}
    }
}
fn barrier(device:&ash::Device,cb:vk::CommandBuffer,image:vk::Image,from:vk::ImageLayout,to:vk::ImageLayout) {
    crate::image_barrier(device,cb,image,vk::ImageAspectFlags::COLOR,from,to,
        vk::PipelineStageFlags2::ALL_COMMANDS,vk::AccessFlags2::MEMORY_READ|vk::AccessFlags2::MEMORY_WRITE,
        vk::PipelineStageFlags2::ALL_COMMANDS,vk::AccessFlags2::MEMORY_READ|vk::AccessFlags2::MEMORY_WRITE);
}
impl Target {fn destroy(&mut self,device:&ash::Device,alloc:&mut Allocator) {
    unsafe {device.destroy_descriptor_pool(self.pool,None);}
    self.raw.destroy(device,alloc);for image in self.color.iter_mut().chain(self.geometry.iter_mut()) {image.destroy(device,alloc);}
    for uniform in &mut self.uniforms {uniform.destroy(device,alloc);}
}}
