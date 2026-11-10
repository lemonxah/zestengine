//! Immutable descriptor snapshots for four independently updated geometry grids.
use super::*;

pub(crate) const LEVELS:usize=4;
#[repr(C)]
#[derive(Clone,Copy,Pod,Zeroable)]
struct Params {min:[[f32;4];LEVELS],max:[[f32;4];LEVELS],meta:[u32;4]}
#[derive(Clone)]
pub(crate) struct Resources {
    pub distance:[vk::ImageView;LEVELS],
    pub index:[vk::ImageView;LEVELS],
    pub params:Arc<Buffer>,
}
impl Resources {
    pub(super) fn new(device:&ash::Device,alloc:&mut Allocator,fields:[Option<&CachedGdf>;LEVELS],detail_only:bool)->Result<Self> {
        let base=fields[0].ok_or_else(||anyhow::anyhow!("geometry clipmaps require a base field"))?;
        let mut distance=[base.dist_tex.view;LEVELS];let mut index=[base.index_tex.view;LEVELS];
        let mut params=Params::zeroed();params.meta[1]=u32::from(detail_only);
        for (level,field) in fields.into_iter().enumerate() {
            let Some(field)=field else {continue};
            // Old coarse levels are disabled immediately when geometry/source
            // order changes. The current mesh hierarchy covers their intervals
            // until each level finishes rebuilding against the new snapshot.
            let compatible=level==0 || base.geometry.as_ref().zip(field.geometry.as_ref())
                .is_some_and(|(a,b)|a.same_sources(b));
            if !compatible {continue;}
            params.meta[0]|=1<<level;
            params.min[level][..3].copy_from_slice(&field.min);
            params.max[level][..3].copy_from_slice(&field.max);
            distance[level]=field.dist_tex.view;index[level]=field.index_tex.view;
        }
        let mut buffer=Buffer::new(device,alloc,size_of::<Params>() as u64,vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,"Flux geometry clipmap domains")?;
        buffer.write(0,bytemuck::bytes_of(&params));
        Ok(Self {distance,index,params:Arc::new(buffer)})
    }
    pub fn bind(&self,device:&ash::Device,set:vk::DescriptorSet,uniform_binding:u32,distance_sampler:vk::Sampler,index_sampler:vk::Sampler) {
        for (binding,views,sampler) in [(0,&self.distance,distance_sampler),(1,&self.index,index_sampler)] {
            let images=views.map(|view|vk::DescriptorImageInfo::default().sampler(sampler).image_view(view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL));
            unsafe {device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&images)],&[]);}
        }
        write_ubo(device,set,uniform_binding,self.params.handle);
    }
}
