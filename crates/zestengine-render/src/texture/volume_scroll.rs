//! Immutable scrolling volumes: retain overlap by GPU copy and stage only
//! exposed slabs. The old image stays usable until publication succeeds.
use super::*;

#[derive(Clone,Copy,Debug)]
pub(crate) struct Region {pub origin:[u32;3],pub size:[u32;3]}
#[derive(Debug)]
pub(crate) struct VolumeScroll {
    pub dims:[u32;3],pub source:[u32;3],pub destination:[u32;3],pub overlap:[u32;3],
    pub exposed:Vec<Region>,
}
impl VolumeScroll {
    /// `offset` is the new world-grid origin minus the old one, in voxels.
    pub fn new(dims:[u32;3],offset:[i32;3])->Option<Self> {
        if dims.iter().zip(offset).any(|(&d,o)|d==0 || o.unsigned_abs()>=d) {return None;}
        let source=offset.map(|v|v.max(0) as u32);
        let destination=offset.map(|v|v.saturating_neg().max(0) as u32);
        let overlap=std::array::from_fn(|a|dims[a]-offset[a].unsigned_abs());
        let mut lo=[0;3];let mut hi=dims;let mut exposed=Vec::new();
        // Split the complement into disjoint slabs, including diagonal moves.
        for a in 0..3 {
            if lo[a]<destination[a] {
                let mut end=hi;end[a]=destination[a];
                exposed.push(Region {origin:lo,size:std::array::from_fn(|i|end[i]-lo[i])});lo[a]=end[a];
            }
            let end=destination[a]+overlap[a];
            if hi[a]>end {
                let mut begin=lo;begin[a]=end;
                exposed.push(Region {origin:begin,size:std::array::from_fn(|i|hi[i]-begin[i])});hi[a]=end;
            }
        }
        Some(Self {dims,source,destination,overlap,exposed})
    }
    pub fn upload_voxels(&self)->usize {self.exposed.iter().map(|r|r.size.iter().map(|&n|n as usize).product::<usize>()).sum()}
    fn pack(&self,bytes:&[u8])->(Vec<u8>,Vec<vk::BufferImageCopy>) {
        let mut data=Vec::with_capacity(self.upload_voxels()*4);let mut copies=Vec::new();
        for r in &self.exposed {
            copies.push(vk::BufferImageCopy::default().buffer_offset(data.len() as u64)
                .image_subresource(layers()).image_offset(offset(r.origin)).image_extent(extent(r.size)));
            for z in r.origin[2]..r.origin[2]+r.size[2] {for y in r.origin[1]..r.origin[1]+r.size[1] {
                let begin=((z as usize*self.dims[1] as usize+y as usize)*self.dims[0] as usize+r.origin[0] as usize)*4;
                data.extend_from_slice(&bytes[begin..begin+r.size[0] as usize*4]);
            }}
        }
        (data,copies)
    }
}
fn extent(d:[u32;3])->vk::Extent3D {vk::Extent3D {width:d[0],height:d[1],depth:d[2]}}
fn offset(d:[u32;3])->vk::Offset3D {vk::Offset3D {x:d[0] as i32,y:d[1] as i32,z:d[2] as i32}}
fn layers()->vk::ImageSubresourceLayers {vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1)}

impl GpuTexture {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn upload_volume_scrolled(device:&ash::Device,allocator:&mut Allocator,pool:vk::CommandPool,queue:vk::Queue,
        bytes:&[u8],format:vk::Format,previous:&Self,plan:&VolumeScroll)->Result<Self> {
        anyhow::ensure!(matches!(format,vk::Format::R32_SFLOAT|vk::Format::R32_UINT),"scroll requires a 32-bit scalar volume");
        anyhow::ensure!(plan.dims.iter().try_fold(4usize,|n,&d|n.checked_mul(d as usize))==Some(bytes.len()),"invalid scroll volume length");
        let (data,copies)=plan.pack(bytes);
        let mut staging=Buffer::new(device,allocator,data.len().max(4) as u64,vk::BufferUsageFlags::TRANSFER_SRC,MemoryLocation::CpuToGpu,"GDF exposed slabs")?;
        staging.write(0,&data);
        let image=match unsafe {device.create_image(&vk::ImageCreateInfo::default().image_type(vk::ImageType::TYPE_3D)
            .format(format).extent(extent(plan.dims)).mip_levels(1).array_layers(1).samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL).usage(vk::ImageUsageFlags::TRANSFER_SRC|vk::ImageUsageFlags::TRANSFER_DST|vk::ImageUsageFlags::SAMPLED),None)} {
            Ok(image)=>image,Err(e)=>{staging.destroy(device,allocator);return Err(e.into());}
        };
        let allocation=match allocator.allocate(&AllocationCreateDesc {name:"scrolled GDF",requirements:unsafe {device.get_image_memory_requirements(image)},
            location:MemoryLocation::GpuOnly,linear:false,allocation_scheme:AllocationScheme::GpuAllocatorManaged}) {
            Ok(a)=>a,Err(e)=>{unsafe {device.destroy_image(image,None)};staging.destroy(device,allocator);return Err(e.into());}
        };
        let bound=unsafe {device.bind_image_memory(image,allocation.memory(),allocation.offset())};
        let mut texture=Self {image,view:vk::ImageView::null(),allocation:Some(allocation)};
        let result=(||->Result<()> {
            bound?;
            let range=vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
            texture.view=unsafe {device.create_image_view(&vk::ImageViewCreateInfo::default().image(image).view_type(vk::ImageViewType::TYPE_3D)
                .format(format).subresource_range(range),None)?};
            one_time_submit(device,pool,queue,|cb|unsafe {
                let src=vk::ImageMemoryBarrier2::default().image(previous.image).subresource_range(range)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS).src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER).dst_access_mask(vk::AccessFlags2::TRANSFER_READ);
                let dst=vk::ImageMemoryBarrier2::default().image(image).subresource_range(range)
                    .old_layout(vk::ImageLayout::UNDEFINED).new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER).dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE);
                device.cmd_pipeline_barrier2(cb,&vk::DependencyInfo::default().image_memory_barriers(&[src,dst]));
                let copy=vk::ImageCopy::default().src_subresource(layers()).dst_subresource(layers())
                    .src_offset(offset(plan.source)).dst_offset(offset(plan.destination)).extent(extent(plan.overlap));
                device.cmd_copy_image(cb,previous.image,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,image,vk::ImageLayout::TRANSFER_DST_OPTIMAL,&[copy]);
                if !copies.is_empty() {device.cmd_copy_buffer_to_image(cb,staging.handle,image,vk::ImageLayout::TRANSFER_DST_OPTIMAL,&copies);}
                let sampled=|image,layout,access|vk::ImageMemoryBarrier2::default().image(image).subresource_range(range)
                    .old_layout(layout).new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER).src_access_mask(access)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS).dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ);
                device.cmd_pipeline_barrier2(cb,&vk::DependencyInfo::default().image_memory_barriers(&[
                    sampled(previous.image,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,vk::AccessFlags2::TRANSFER_READ),
                    sampled(image,vk::ImageLayout::TRANSFER_DST_OPTIMAL,vk::AccessFlags2::TRANSFER_WRITE)]));
            })
        })();
        staging.destroy(device,allocator);
        if let Err(e)=result {texture.destroy(device,allocator);return Err(e);}
        Ok(texture)
    }
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn scroll_slabs_partition_new_volume_without_overlap_or_gaps() {
        let dims=[7,5,9];let count=7*5*9;
        for x in -6..=6 {for y in -4..=4 {for z in -8..=8 {
            let p=VolumeScroll::new(dims,[x,y,z]).unwrap();let mut touched=vec![0;count];
            for region in p.exposed.iter().copied().chain([Region {origin:p.destination,size:p.overlap}]) {
                for k in region.origin[2]..region.origin[2]+region.size[2] {for j in region.origin[1]..region.origin[1]+region.size[1] {for i in region.origin[0]..region.origin[0]+region.size[0] {
                    touched[(k*5*7+j*7+i) as usize]+=1;
                }}}
            }
            assert!(touched.iter().all(|&n|n==1),"overlap/gap for {x},{y},{z}");
            assert_eq!(p.upload_voxels()+p.overlap.iter().map(|&n|n as usize).product::<usize>(),count);
        }}}
        assert!(VolumeScroll::new(dims,[7,0,0]).is_none());assert!(VolumeScroll::new(dims,[i32::MIN,0,0]).is_none());
        let p=VolumeScroll::new([128;3],[8,0,0]).unwrap();
        assert_eq!(p.upload_voxels(),8*128*128); // one newly exposed slab, not the full grid
    }
}
