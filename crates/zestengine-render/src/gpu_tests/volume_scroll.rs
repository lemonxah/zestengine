//! Actual R32 distance/index transfers, including old-image lifetime/layout.
use ash::vk;
use crate::{alloc::{Buffer,one_time_submit},texture::{GpuTexture,VolumeScroll}};
use gpu_allocator::MemoryLocation;

fn values(dims:[u32;3],origin:[i32;3],float:bool)->Vec<u32> {
    let mut result=Vec::new();
    for z in 0..dims[2] {for y in 0..dims[1] {for x in 0..dims[0] {
        let p=[x as i32+origin[0],y as i32+origin[1],z as i32+origin[2]];
        result.push(if float {((p[0]+p[1]*100+p[2]*10000) as f32).to_bits()}
            else {(p[0] as u32).wrapping_mul(0x9e3779b9)^(p[1] as u32).wrapping_mul(0x85ebca6b)^(p[2] as u32).wrapping_mul(0xc2b2ae35)});
    }}}
    result
}

#[test]
fn scrolled_distance_and_index_volumes_match_full_uploads() {
    let Some(g)=super::gpu() else {return};let device=&g.device;let mut alloc=g.allocator().unwrap();
    let before=crate::context::validation_error_count();let dims=[9,7,11];let count=9*7*11;
    let mut readback=Buffer::new(device,&mut alloc,(count*4) as u64,vk::BufferUsageFlags::TRANSFER_DST,MemoryLocation::GpuToCpu,"scroll validation").unwrap();
    let read=|texture:&GpuTexture|->Vec<u32> {
        one_time_submit(device,g.pool,g.queue,|cb|unsafe {
            let range=vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
            let begin=vk::ImageMemoryBarrier2::default().image(texture.image).subresource_range(range)
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS).src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER).dst_access_mask(vk::AccessFlags2::TRANSFER_READ);
            device.cmd_pipeline_barrier2(cb,&vk::DependencyInfo::default().image_memory_barriers(&[begin]));
            let copy=vk::BufferImageCopy::default().image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                .image_extent(vk::Extent3D {width:dims[0],height:dims[1],depth:dims[2]});
            device.cmd_copy_image_to_buffer(cb,texture.image,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,readback.handle,&[copy]);
            let end=vk::ImageMemoryBarrier2::default().image(texture.image).subresource_range(range)
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL).new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER).src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS).dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ);
            let host=vk::MemoryBarrier2::default().src_stage_mask(vk::PipelineStageFlags2::TRANSFER).src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST).dst_access_mask(vk::AccessFlags2::HOST_READ);
            device.cmd_pipeline_barrier2(cb,&vk::DependencyInfo::default().image_memory_barriers(&[end]).memory_barriers(&[host]));
        }).unwrap();
        let bytes=readback.read();bytemuck::pod_collect_to_vec(&bytes[..count*4])
    };
    let mut failures=Vec::new();let mut checks=0;
    for float in [true,false] {
        let mut origin=[0;3];let mut expected=values(dims,origin,float);
        let format=if float {vk::Format::R32_SFLOAT} else {vk::Format::R32_UINT};
        let mut previous=if float {GpuTexture::upload_volume(device,&mut alloc,g.pool,g.queue,&expected.iter().map(|&x|f32::from_bits(x)).collect::<Vec<_>>(),dims).unwrap()}
            else {GpuTexture::upload_volume_u32(device,&mut alloc,g.pool,g.queue,&expected,dims).unwrap()};
        for delta in [[0,0,0],[1,0,0],[-2,0,0],[0,2,0],[1,-1,-3],[0,0,4],[6,-4,8]] {
            origin=std::array::from_fn(|a|origin[a]+delta[a]);
            let next=values(dims,origin,float);let plan=VolumeScroll::new(dims,delta).unwrap();
            let texture=GpuTexture::upload_volume_scrolled(device,&mut alloc,g.pool,g.queue,bytemuck::cast_slice(&next),format,&previous,&plan).unwrap();
            if read(&texture)!=next {failures.push(format!("destination mismatch float={float} delta={delta:?}"));}
            if read(&previous)!=expected {failures.push(format!("old image modified float={float} delta={delta:?}"));}
            checks+=2*count;previous.destroy(device,&mut alloc);previous=texture;expected=next;
        }
        previous.destroy(device,&mut alloc);
    }
    readback.destroy(device,&mut alloc);drop(alloc);
    eprintln!("[Flux volume scroll] {checks} exact 32-bit voxel comparisons, failures={}",failures.len());
    assert!(failures.is_empty(),"{}",failures.join("\n"));
    assert_eq!(crate::context::validation_error_count(),before,"scroll transfers produced Vulkan errors");
}
