//! Per-view reflection reconstruction. Histories contain only the signed
//! replacement for forward environment specular, never the scene's diffuse HDR.
use crate::pipeline_store::PipelineDeviceExt as _;
#[cfg(test)] mod tests;
use anyhow::Result;
use ash::vk;
use bytemuck::{Pod,Zeroable};
use glam::Mat4;
use gpu_allocator::{vulkan::Allocator,MemoryLocation};
use crate::{alloc::Buffer,screen_pass::ScreenImage,FRAMES_IN_FLIGHT};
#[repr(C)]
#[derive(Clone,Copy,Pod,Zeroable)]
struct Params {
    inverse_view_projection:[[f32;4];4],inverse_view:[[f32;4];4],
    previous_view_projection:[[f32;4];4],previous_inverse_view_projection:[[f32;4];4],
    camera:[f32;4],fog:Fog,options:[f32;4],motion:[f32;4],
}
/// Only final composition consumes fog; it must never affect history validity or radiance.
#[repr(C)]
#[derive(Clone,Copy,Default,Pod,Zeroable)]
pub(crate) struct Fog {
    pub color:[f32;4],pub params:[f32;4],pub light:[f32;4],pub sun:[f32;4],
}
/// Backend-independent reconstruction inputs. Keys describe tracing settings,
/// not exposure, animated fog or the frame's stochastic sample index.
pub(crate) struct Reconstruction {
    pub inv_proj:[[f32;4];4],pub inv_view:[[f32;4];4],pub camera:[f32;4],
    pub key:[f32;10],pub enabled:bool,pub fog:Fog,
}
#[derive(Clone,Copy)]
struct Capture {serial:u64,vp:Mat4,key:[f32;10]}
struct Target {
    raw:ScreenImage,color:[ScreenImage;2],geometry:[ScreenImage;2],moments:[ScreenImage;2],
    pool:vk::DescriptorPool,sets:Vec<vk::DescriptorSet>,uniforms:Vec<Buffer>,last:Option<Capture>,initialized:bool,
}
pub(crate) struct Denoiser {
    motion:[Option<(u64,vk::ImageView)>;2],
    views:[Option<Target>;2],retired:Vec<(u64,Target)>,set_layout:vk::DescriptorSetLayout,
    layout:vk::PipelineLayout,pipeline:vk::Pipeline,spatial:vk::Pipeline,sampler:vk::Sampler,
}
impl Denoiser {
    pub fn new(device:&ash::Device)->Result<Self> {unsafe {
        let bindings:Vec<_>=(0..13).map(|binding|vk::DescriptorSetLayoutBinding::default().binding(binding)
            .descriptor_type(if binding<6 || binding==10 || binding==11 {vk::DescriptorType::COMBINED_IMAGE_SAMPLER} else if binding<9 || binding==12 {vk::DescriptorType::STORAGE_IMAGE} else {vk::DescriptorType::UNIFORM_BUFFER})
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
        let set_layout=device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),None)?;
        let layout=device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]),None)?;
        let shader=crate::screen_pass::shader_module(device,include_bytes!(concat!(env!("OUT_DIR"),"/reflection_denoise.comp.spv")))?;
        let built=device.create_cached_compute_pipelines(vk::PipelineCache::null(),&[vk::ComputePipelineCreateInfo::default().layout(layout)
            .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(shader).name(c"main"))],None);
        device.destroy_shader_module(shader,None);let pipeline=built.map_err(|(_,e)|e)?[0];
        let shader=crate::screen_pass::shader_module(device,include_bytes!(concat!(env!("OUT_DIR"),"/reflection_spatial.comp.spv")))?;
        let built=device.create_cached_compute_pipelines(vk::PipelineCache::null(),&[vk::ComputePipelineCreateInfo::default().layout(layout)
            .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(shader).name(c"main"))],None);
        device.destroy_shader_module(shader,None);let spatial=built.map_err(|(_,e)|e)?[0];
        let sampler=device.create_sampler(&vk::SamplerCreateInfo::default().mag_filter(vk::Filter::NEAREST).min_filter(vk::Filter::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE).address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),None)?;
        Ok(Self {motion:[None,None],views:[None,None],retired:Vec::new(),set_layout,layout,pipeline,spatial,sampler})
    }}
    pub fn set_motion(&mut self,view:usize,serial:u64,image:vk::ImageView) {self.motion[view]=Some((serial,image));}
    fn target(&self,device:&ash::Device,alloc:&mut Allocator,extent:vk::Extent2D)->Result<Target> {
        let raw=ScreenImage::new(device,alloc,vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageUsageFlags::SAMPLED|vk::ImageUsageFlags::STORAGE|vk::ImageUsageFlags::COLOR_ATTACHMENT|vk::ImageUsageFlags::TRANSFER_SRC|vk::ImageUsageFlags::TRANSFER_DST,
            extent,1,"Reflection samples")?;
        let mut image=|format,name|ScreenImage::new(device,alloc,format,
            vk::ImageUsageFlags::SAMPLED|vk::ImageUsageFlags::STORAGE|vk::ImageUsageFlags::TRANSFER_DST|vk::ImageUsageFlags::TRANSFER_SRC,extent,1,name);
        let color=[image(vk::Format::R16G16B16A16_SFLOAT,"RT reflection history A")?,image(vk::Format::R16G16B16A16_SFLOAT,"RT reflection history B")?];
        let geometry=[image(vk::Format::R32G32B32A32_SFLOAT,"RT reflection geometry A")?,image(vk::Format::R32G32B32A32_SFLOAT,"RT reflection geometry B")?];
        let moments=[image(vk::Format::R32G32B32A32_SFLOAT,"Reflection moments A")?,image(vk::Format::R32G32B32A32_SFLOAT,"Reflection moments B")?];
        let mut uniforms=Vec::new();for _ in 0..FRAMES_IN_FLIGHT {uniforms.push(Buffer::new(device,alloc,std::mem::size_of::<Params>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,MemoryLocation::CpuToGpu,"RT denoise parameters")?);}
        unsafe {
            let pool=device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(FRAMES_IN_FLIGHT as u32).pool_sizes(&[
                vk::DescriptorPoolSize {ty:vk::DescriptorType::COMBINED_IMAGE_SAMPLER,descriptor_count:8*FRAMES_IN_FLIGHT as u32},
                vk::DescriptorPoolSize {ty:vk::DescriptorType::STORAGE_IMAGE,descriptor_count:4*FRAMES_IN_FLIGHT as u32},
                vk::DescriptorPoolSize {ty:vk::DescriptorType::UNIFORM_BUFFER,descriptor_count:FRAMES_IN_FLIGHT as u32}]),None)?;
            let sets=device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[self.set_layout; FRAMES_IN_FLIGHT]))?;
            Ok(Target {raw,color,geometry,moments,pool,sets,uniforms,last:None,initialized:false})
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
            for image in v.color.iter().chain(v.geometry.iter()).chain(v.moments.iter()) {
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
        color:vk::ImageView,depth:vk::ImageView,gbuf:vk::ImageView,out_image:vk::Image,out_view:vk::ImageView,u:&Reconstruction) {
        let motion=self.motion[view].filter(|m|m.0==serial).map(|m|m.1);
        let v=self.views[view].as_mut().unwrap();let cur=(serial&1) as usize;let prev=cur^1;
        let inverse_vp=Mat4::from_cols_array_2d(&u.inv_view)*Mat4::from_cols_array_2d(&u.inv_proj);let vp=inverse_vp.inverse();
        let key=u.key;
        let valid=v.last.is_some_and(|old|old.serial+1==serial && old.key==key);
        let previous=v.last.map_or(vp,|old|old.vp);
        let params=Params {inverse_view_projection:inverse_vp.to_cols_array_2d(),inverse_view:u.inv_view,
            previous_view_projection:previous.to_cols_array_2d(),previous_inverse_view_projection:previous.inverse().to_cols_array_2d(),
            camera:u.camera,fog:u.fog,options:[u32::from(valid) as f32,u32::from(u.enabled) as f32,16.0,0.002],motion:[f32::from(motion.is_some()),0.0,0.0,0.0]};
        v.uniforms[frame].write(0,bytemuck::bytes_of(&params));let set=v.sets[frame];
        unsafe {
            for (binding,image) in [v.raw.view,color,depth,gbuf,v.color[prev].view,v.geometry[prev].view,out_view,v.color[cur].view,v.geometry[cur].view].into_iter().enumerate() {
                let storage=binding>=6;let info=[vk::DescriptorImageInfo::default().sampler(self.sampler).image_view(image)
                    .image_layout(if storage {vk::ImageLayout::GENERAL} else {vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL})];
                device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding as u32)
                    .descriptor_type(if storage {vk::DescriptorType::STORAGE_IMAGE} else {vk::DescriptorType::COMBINED_IMAGE_SAMPLER}).image_info(&info)],&[]);
            }
            let motion_info=[vk::DescriptorImageInfo::default().sampler(self.sampler).image_view(motion.unwrap_or(depth))
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(10)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&motion_info)],&[]);
            for (binding,image,storage) in [(11,v.moments[prev].view,false),(12,v.moments[cur].view,true)] {
                let info=[vk::DescriptorImageInfo::default().sampler(self.sampler).image_view(image)
                    .image_layout(if storage {vk::ImageLayout::GENERAL} else {vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL})];
                device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding)
                    .descriptor_type(if storage {vk::DescriptorType::STORAGE_IMAGE} else {vk::DescriptorType::COMBINED_IMAGE_SAMPLER}).image_info(&info)],&[]);
            }
            let info=[vk::DescriptorBufferInfo::default().buffer(v.uniforms[frame].handle).range(vk::WHOLE_SIZE)];
            device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(9).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&info)],&[]);
        }
        barrier(device,cb,v.raw.image,vk::ImageLayout::GENERAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        barrier(device,cb,out_image,vk::ImageLayout::UNDEFINED,vk::ImageLayout::GENERAL);
        for image in [v.color[cur].image,v.geometry[cur].image,v.moments[cur].image] {barrier(device,cb,image,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,vk::ImageLayout::GENERAL);}
        unsafe {
            device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,self.pipeline);
            device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,self.layout,0,&[set],&[]);
            device.cmd_dispatch(cb,v.raw.extent.width.div_ceil(8),v.raw.extent.height.div_ceil(8),1);
        }
        // The spatial pass reads the completed temporal fields; it never feeds
        // its wider blur back into the next frame's persistent history.
        for image in [v.color[cur].image,v.geometry[cur].image,v.moments[cur].image] {
            barrier(device,cb,image,vk::ImageLayout::GENERAL,vk::ImageLayout::GENERAL);
        }
        unsafe {
            device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,self.spatial);
            device.cmd_dispatch(cb,v.raw.extent.width.div_ceil(8),v.raw.extent.height.div_ceil(8),1);
        }
        for image in [v.color[cur].image,v.geometry[cur].image,v.moments[cur].image] {barrier(device,cb,image,vk::ImageLayout::GENERAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);}
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
        unsafe {device.destroy_descriptor_set_layout(self.set_layout,None);device.destroy_pipeline_layout(self.layout,None);device.destroy_pipeline(self.pipeline,None);device.destroy_pipeline(self.spatial,None);device.destroy_sampler(self.sampler,None);}
    }
}
fn barrier(device:&ash::Device,cb:vk::CommandBuffer,image:vk::Image,from:vk::ImageLayout,to:vk::ImageLayout) {
    crate::image_barrier(device,cb,image,vk::ImageAspectFlags::COLOR,from,to,
        vk::PipelineStageFlags2::ALL_COMMANDS,vk::AccessFlags2::MEMORY_READ|vk::AccessFlags2::MEMORY_WRITE,
        vk::PipelineStageFlags2::ALL_COMMANDS,vk::AccessFlags2::MEMORY_READ|vk::AccessFlags2::MEMORY_WRITE);
}
impl Target {fn destroy(&mut self,device:&ash::Device,alloc:&mut Allocator) {
    unsafe {device.destroy_descriptor_pool(self.pool,None);}
    self.raw.destroy(device,alloc);for image in self.color.iter_mut().chain(self.geometry.iter_mut()).chain(self.moments.iter_mut()) {image.destroy(device,alloc);}
    for uniform in &mut self.uniforms {uniform.destroy(device,alloc);}
}}
