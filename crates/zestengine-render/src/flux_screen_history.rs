//! Per-view opaque HDR/depth history captured between opaque and transparent
//! rendering. Scene, Camera and game views have independent geometry, camera
//! metadata and lifetimes. Display exposure, bloom and UI never enter it.
use anyhow::Result;
use ash::vk;
use bytemuck::{Pod,Zeroable};
use glam::Mat4;
use gpu_allocator::vulkan::Allocator;
use crate::{flux_depth::{DepthReduce,DepthPyramid},screen_pass::ScreenImage,FRAMES_IN_FLIGHT};

#[repr(C)]
#[derive(Clone,Copy,Pod,Zeroable)]
pub(crate) struct Params {
    pub view:[[f32;4];4],
    pub projection:[[f32;4];4],
    pub inverse_projection:[[f32;4];4],
    pub previous_view_projection:[[f32;4];4],
    pub previous_inverse_view_projection:[[f32;4];4],
    pub previous_camera:[f32;4],
    pub options:[f32;4], // valid, current flip, previous flip, relative thickness
}
#[derive(Clone,Copy)]
struct Camera {serial:u64,view:Mat4,projection:Mat4,flip:bool}
#[derive(Clone,Copy)]
pub(crate) struct Resources {pub image:vk::ImageView,pub hierarchy:vk::Buffer,pub params:Params}
struct View {
    history:ScreenImage,pyramid:DepthPyramid,pool:vk::DescriptorPool,sets:Vec<vk::DescriptorSet>,
    initialized:bool,captured:Option<Camera>,prepared:Option<Camera>,
    #[cfg(test)] last_valid:bool,
}
pub(crate) struct ScreenHistory {
    views:[Option<View>;3],retired:Vec<(u64,View)>,reduce:DepthReduce,
    set_layout:vk::DescriptorSetLayout,layout:vk::PipelineLayout,pipeline:vk::Pipeline,sampler:vk::Sampler,
}
impl ScreenHistory {
    pub fn new(device:&ash::Device)->Result<Self> {unsafe {
        let bindings:Vec<_>=(0..3).map(|binding|vk::DescriptorSetLayoutBinding::default().binding(binding)
            .descriptor_type(if binding==2 {vk::DescriptorType::STORAGE_IMAGE} else {vk::DescriptorType::COMBINED_IMAGE_SAMPLER})
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
        let set_layout=device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),None)?;
        let layout=device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]),None)?;
        let module=crate::screen_pass::shader_module(device,include_bytes!(concat!(env!("OUT_DIR"),"/flux_screen_capture.comp.spv")))?;
        let built=device.create_compute_pipelines(vk::PipelineCache::null(),&[vk::ComputePipelineCreateInfo::default().layout(layout)
            .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main"))],None);
        device.destroy_shader_module(module,None);
        let pipeline=built.map_err(|(_,e)|e)?[0];
        let sampler=device.create_sampler(&vk::SamplerCreateInfo::default().min_filter(vk::Filter::NEAREST).mag_filter(vk::Filter::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE).address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),None)?;
        Ok(Self {views:[None,None,None],retired:Vec::new(),reduce:DepthReduce::new(device)?,set_layout,layout,pipeline,sampler})
    }}
    fn target(&self,device:&ash::Device,alloc:&mut Allocator,extent:vk::Extent2D)->Result<View> {
        let history=ScreenImage::new(device,alloc,vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageUsageFlags::SAMPLED|vk::ImageUsageFlags::STORAGE|vk::ImageUsageFlags::TRANSFER_DST|vk::ImageUsageFlags::TRANSFER_SRC,
            extent,1,"Flux opaque screen history")?;
        let pyramid=self.reduce.target(device,alloc,extent)?;
        unsafe {
            let pool=device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(FRAMES_IN_FLIGHT as u32).pool_sizes(&[
                vk::DescriptorPoolSize {ty:vk::DescriptorType::COMBINED_IMAGE_SAMPLER,descriptor_count:2*FRAMES_IN_FLIGHT as u32},
                vk::DescriptorPoolSize {ty:vk::DescriptorType::STORAGE_IMAGE,descriptor_count:FRAMES_IN_FLIGHT as u32}]),None)?;
            let sets=device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&vec![self.set_layout;FRAMES_IN_FLIGHT]))?;
            for &set in &sets {
                let info=[vk::DescriptorImageInfo::default().image_view(history.view).image_layout(vk::ImageLayout::GENERAL)];
                device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&info)],&[]);
            }
            Ok(View {history,pyramid,pool,sets,initialized:false,captured:None,prepared:None,#[cfg(test)] last_valid:false})
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(&mut self,device:&ash::Device,alloc:&mut Allocator,cb:vk::CommandBuffer,
        id:usize,serial:u64,frame:usize,extent:vk::Extent2D,depth:vk::ImageView,
        view:Mat4,projection:Mat4,flip:bool,valid:bool)->Result<Resources> {
        for i in (0..self.retired.len()).rev() {if self.retired[i].0<=serial {self.retired.swap_remove(i).1.destroy(device,alloc);}}
        if self.views[id].as_ref().is_none_or(|v|v.history.extent!=extent) {
            let next=self.target(device,alloc,extent)?;
            if let Some(old)=self.views[id].replace(next) {self.retired.push((serial+FRAMES_IN_FLIGHT as u64+1,old));}
        }
        let target=self.views[id].as_mut().unwrap();
        let current=Camera {serial,view,projection,flip};
        target.prepared=Some(current);
        let previous=target.captured.unwrap_or(current);
        let valid=valid && target.captured.is_some_and(|c|c.serial+1==serial);
        #[cfg(test)] {target.last_valid=valid;}
        unsafe {
            if !target.initialized {
                let range=vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
                let barrier=[vk::ImageMemoryBarrier::default().image(target.history.image).subresource_range(range)
                    .old_layout(vk::ImageLayout::UNDEFINED).new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)];
                device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::TOP_OF_PIPE,vk::PipelineStageFlags::TRANSFER,vk::DependencyFlags::empty(),&[],&[],&barrier);
                device.cmd_clear_color_image(cb,target.history.image,vk::ImageLayout::TRANSFER_DST_OPTIMAL,&vk::ClearColorValue {float32:[0.0,0.0,0.0,1.0]},&[range]);
                let barrier=[vk::ImageMemoryBarrier::default().image(target.history.image).subresource_range(range)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL).new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ)];
                device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::TRANSFER,vk::PipelineStageFlags::COMPUTE_SHADER,vk::DependencyFlags::empty(),&[],&[],&barrier);
                target.initialized=true;
            }
        }
        if valid {self.reduce.record(device,cb,&target.pyramid,depth,projection.inverse().to_cols_array_2d(),frame);}
        Ok(Resources {image:target.history.view,hierarchy:target.pyramid.buffer.handle,params:Params {
            view:view.to_cols_array_2d(),projection:projection.to_cols_array_2d(),inverse_projection:projection.inverse().to_cols_array_2d(),
            previous_view_projection:(previous.projection*previous.view).to_cols_array_2d(),
            previous_inverse_view_projection:(previous.projection*previous.view).inverse().to_cols_array_2d(),
            previous_camera:previous.view.inverse().w_axis.to_array(),
            options:[u32::from(valid) as f32,u32::from(flip) as f32,u32::from(previous.flip) as f32,0.01]}})
    }
    pub fn ready(&self,id:usize,serial:u64)->bool {self.views[id].as_ref().is_some_and(|v|v.prepared.is_some_and(|c|c.serial==serial))}
    /// Sources are shader-read-only; consumes the prepared camera metadata.
    pub fn capture(&mut self,device:&ash::Device,cb:vk::CommandBuffer,id:usize,serial:u64,frame:usize,hdr:vk::ImageView,depth:vk::ImageView) {
        let Some(v)=self.views[id].as_mut().filter(|v|v.prepared.is_some_and(|c|c.serial==serial)) else {return;};
        unsafe {
            for (binding,image) in [(0,hdr),(1,depth)] {
                let info=[vk::DescriptorImageInfo::default().sampler(self.sampler).image_view(image).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(v.sets[frame]).dst_binding(binding)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&info)],&[]);
            }
            let range=vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
            let barrier=[vk::ImageMemoryBarrier::default().image(v.history.image).subresource_range(range)
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::SHADER_READ|vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_WRITE)];
            device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::COMPUTE_SHADER|vk::PipelineStageFlags::TRANSFER,vk::PipelineStageFlags::COMPUTE_SHADER,vk::DependencyFlags::empty(),&[],&[],&barrier);
            device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,self.pipeline);
            device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,self.layout,0,&[v.sets[frame]],&[]);
            device.cmd_dispatch(cb,v.history.extent.width.div_ceil(8),v.history.extent.height.div_ceil(8),1);
            let barrier=[vk::ImageMemoryBarrier::default().image(v.history.image).subresource_range(range)
                .old_layout(vk::ImageLayout::GENERAL).new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ)];
            device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::COMPUTE_SHADER,vk::PipelineStageFlags::COMPUTE_SHADER,vk::DependencyFlags::empty(),&[],&[],&barrier);
        }
        v.captured=v.prepared.take();
    }
    #[cfg(test)]
    pub fn debug_history(&self,id:usize)->(vk::Image,vk::Extent2D,Mat4,u64,bool) {let v=self.views[id].as_ref().unwrap();let c=v.captured.unwrap();
        (v.history.image,v.history.extent,c.projection*c.view,c.serial,v.last_valid)}
    pub fn destroy(&mut self,device:&ash::Device,alloc:&mut Allocator) {
        for v in &mut self.views {if let Some(mut v)=v.take() {v.destroy(device,alloc);}}
        for (_,mut v) in self.retired.drain(..) {v.destroy(device,alloc);}
        self.reduce.destroy(device);
        unsafe {device.destroy_pipeline(self.pipeline,None);device.destroy_pipeline_layout(self.layout,None);device.destroy_descriptor_set_layout(self.set_layout,None);device.destroy_sampler(self.sampler,None);}
    }
}
impl View {fn destroy(&mut self,device:&ash::Device,alloc:&mut Allocator) {
    unsafe {device.destroy_descriptor_pool(self.pool,None);}
    self.history.destroy(device,alloc);self.pyramid.destroy(device,alloc);
}}
