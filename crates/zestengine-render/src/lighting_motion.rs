//! Previous object transforms and per-view world displacement for lighting.
//! Independent of temporal AA; missing correspondence rejects history.
use std::collections::HashMap;
use ash::vk;
use glam::Mat4;
use crate::{DrawCmd, GiSource, screen_pass::ScreenImage, FRAMES_IN_FLIGHT};

pub(crate) const FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;
#[derive(Clone,Copy,PartialEq,Eq,Hash)]
enum Key { Source(GiSource), Anonymous(usize,usize,usize) }
#[derive(Clone,Copy)]
struct Transform { model: Mat4, mesh: usize }
#[derive(Default)]
pub(crate) struct Motion {
    last: Option<u64>,
    transforms: HashMap<Key,Option<Transform>>,
    pub previous: Vec<[[f32;4];4]>,
    views: [Option<ScreenImage>;3],
    retired: Vec<(u64,ScreenImage)>,
    serials: [Option<u64>;3],
}
impl Motion {
    pub fn advance(&mut self,draws:&[DrawCmd],serial:u64) {
        let key=|i:usize,d:&DrawCmd|d.gi_source.map_or(Key::Anonymous(i,d.mesh.0,d.material.0),Key::Source);
        let mut current=HashMap::new();
        for (i,d) in draws.iter().enumerate() {
            current.entry(key(i,d)).and_modify(|v|*v=None)
                .or_insert(Some(Transform {model:d.transform,mesh:d.mesh.0}));
        }
        self.previous.clear();
        for (i,d) in draws.iter().enumerate() {
            let k=key(i,d);
            let old=self.transforms.get(&k).copied().flatten().filter(|old|
                self.last.is_some_and(|s|s+1==serial) && old.mesh==d.mesh.0
                && current.get(&k).is_some_and(Option::is_some)
                && old.model.is_finite() && d.transform.is_finite()
                // No stable object ID: only unchanged geometry is safe to reuse.
                && (d.gi_source.is_some() || old.model==d.transform));
            let mut model=old.map_or(d.transform,|old|old.model).to_cols_array_2d();
            model[3][3]=u32::from(old.is_some()) as f32;
            self.previous.push(model);
        }
        self.transforms=current; self.last=Some(serial);
    }
    pub fn previous_model(&self,draw:usize,current:Mat4)->[[f32;4];4] {
        self.previous.get(draw).copied().unwrap_or_else(||{
            let mut model=current.to_cols_array_2d();
            model[3][3]=0.0;
            model
        })
    }
    pub fn prepare(&mut self,device:&ash::Device,alloc:&mut gpu_allocator::vulkan::Allocator,
        cb:vk::CommandBuffer,view:usize,serial:u64,extent:vk::Extent2D)->anyhow::Result<(vk::Image,vk::ImageView)> {
        for i in (0..self.retired.len()).rev() {
            if self.retired[i].0<=serial {self.retired.swap_remove(i).1.destroy(device,alloc);}
        }
        if self.views[view].as_ref().is_none_or(|v|v.extent!=extent) {
            let next=ScreenImage::new(device,alloc,FORMAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT|vk::ImageUsageFlags::SAMPLED|vk::ImageUsageFlags::TRANSFER_SRC,
                extent,1,"Lighting object motion")?;
            if let Some(old)=self.views[view].replace(next) {self.retired.push((serial+FRAMES_IN_FLIGHT as u64+1,old));}
        }
        let target=self.views[view].as_ref().unwrap();
        crate::image_barrier(device,cb,target.image,vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::UNDEFINED,vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::ALL_COMMANDS,vk::AccessFlags2::MEMORY_READ|vk::AccessFlags2::MEMORY_WRITE,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,vk::AccessFlags2::COLOR_ATTACHMENT_WRITE);
        self.serials[view]=Some(serial);
        Ok((target.image,target.view))
    }
    pub fn finish(&self,device:&ash::Device,cb:vk::CommandBuffer,view:usize) {
        let target=self.views[view].as_ref().unwrap();
        crate::image_barrier(device,cb,target.image,vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER,vk::AccessFlags2::SHADER_READ);
    }
    pub fn view(&self,view:usize,serial:u64)->Option<vk::ImageView> {
        (self.serials[view]==Some(serial)).then(||self.views[view].as_ref().unwrap().view)
    }
    #[cfg(test)]
    pub fn debug(&self,view:usize)->(vk::Image,vk::Extent2D) {let v=self.views[view].as_ref().unwrap();(v.image,v.extent)}
    pub fn destroy(&mut self,device:&ash::Device,alloc:&mut gpu_allocator::vulkan::Allocator) {
        for v in self.views.iter_mut().flatten() {v.destroy(device,alloc);}
        for (_,mut v) in self.retired.drain(..) {v.destroy(device,alloc);}
    }
}
