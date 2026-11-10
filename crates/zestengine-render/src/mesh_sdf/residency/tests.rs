use super::*;
use crate::alloc::one_time_submit;

fn field(n:u32,tag:f32)->Arc<SdfVolume> {
    let mut data=Vec::new();
    for z in 0..n {for y in 0..n {for x in 0..n {
        let p=Vec3::splat(-0.6)+Vec3::new(x as f32,y as f32,z as f32)*(1.2/(n-1) as f32);
        let q=p.abs()-Vec3::splat(0.5);
        data.push(q.max(Vec3::ZERO).length()+q.max_element().min(0.0)+tag);
    }}}
    Arc::new(SdfVolume {dims:[n;3],min:Vec3::splat(-0.6),max:Vec3::splat(0.6),geometry_bounds:[Vec3::splat(-0.5),Vec3::splat(0.5)],data})
}

fn read_pages(g:&crate::cluster_headless::HeadlessGpu,cache:&Cache,snapshot:&Snapshot,readback:&Buffer)->Vec<f32> {
    let image=cache.residency.as_ref().unwrap().pages.texture().unwrap().image;
    let copies:Vec<_>=snapshot.voxels.offsets.iter().enumerate().map(|(i,&p)|vk::BufferImageCopy::default()
        .buffer_offset(i as u64*4)
        .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
        .image_offset(vk::Offset3D {x:(p&1023) as i32+1,y:((p>>10)&1023) as i32+1,z:(p>>20) as i32+1})
        .image_extent(vk::Extent3D {width:1,height:1,depth:1})).collect();
    one_time_submit(&g.device,g.pool,g.queue,|cb|unsafe {
        let range=vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
        let begin=vk::ImageMemoryBarrier2::default().image(image).subresource_range(range)
            .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS).src_access_mask(vk::AccessFlags2::MEMORY_READ|vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER).dst_access_mask(vk::AccessFlags2::TRANSFER_READ);
        g.device.cmd_pipeline_barrier2(cb,&vk::DependencyInfo::default().image_memory_barriers(&[begin]));
        g.device.cmd_copy_image_to_buffer(cb,image,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,readback.handle,&copies);
        let end=vk::ImageMemoryBarrier2::default().image(image).subresource_range(range)
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL).new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER).src_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS).dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ);
        let host=vk::MemoryBarrier2::default().src_stage_mask(vk::PipelineStageFlags2::TRANSFER).src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST).dst_access_mask(vk::AccessFlags2::HOST_READ);
        g.device.cmd_pipeline_barrier2(cb,&vk::DependencyInfo::default().image_memory_barriers(&[end]).memory_barriers(&[host]));
    }).unwrap();
    bytemuck::pod_collect_to_vec(&readback.read()[..snapshot.voxels.offsets.len()*4])
}
fn expected(snapshot:&Snapshot)->Vec<f32> {
    snapshot.voxels.fields.iter().map(|f|f.data[(1+f.dims[0]*(1+f.dims[1])) as usize]).collect()
}
fn fine(snapshot:&Snapshot,volume:&Arc<SdfVolume>)->bool {
    let i=snapshot.voxels.volumes.iter().position(|v|Arc::ptr_eq(v,volume)).unwrap();
    Arc::ptr_eq(volume,&snapshot.voxels.fields[i])
}

#[test]
fn mesh_residency_keeps_coarse_coverage_both_views_and_inflight_pages() {
    let Some(g)=crate::gpu_tests::gpu() else {return};let device=&g.device;let mut alloc=g.allocator().unwrap();
    let before=g.validation_error_count();
    let family=unsafe {g.instance.get_physical_device_queue_family_properties(g.pd)}.iter()
        .position(|q|q.queue_flags.contains(vk::QueueFlags::COMPUTE|vk::QueueFlags::TRANSFER)).unwrap() as u32;
    let levels:Vec<_>=(0..40).map(|i|GiMeshSdfLevels {detail:field(8,i as f32*0.0001),coarse:field(4,i as f32*0.0001)}).collect();
    let inputs:Vec<_>=levels.iter().enumerate().map(|(i,l)|GiMeshSdf {
        gi_source:Some(crate::GiSource {object:i as u128+1,slot:0}),world_to_local:Mat4::from_translation(Vec3::X*(-4.0*i as f32)),
        volume:l.detail.clone(),material:GpuGiMaterial {albedo:[0.5;3],emission:[0.;3],metallic:0.,roughness:1.},in_global_field:true,
    }).collect();
    let mut cache=Cache::default();cache.test_streamed_extent(device,&mut alloc,g.queue,family,[32;3],16).unwrap();
    let mut readback=Buffer::new(device,&mut alloc,160,vk::BufferUsageFlags::TRANSFER_DST,MemoryLocation::GpuToCpu,"mesh residency readback").unwrap();
    let settle=|cache:&mut Cache,alloc:&mut Allocator,views:&[Vec3]| {
        for frame in 0..60 {
            let update=cache.set_streamed(device,alloc,g.queue,family,&inputs,&levels,views).unwrap();
            one_time_submit(device,g.pool,g.queue,|_|{}).unwrap();
            if !update.pending && !update.streaming {return frame;}
        }
        panic!("mesh residency did not settle under a feasible fixed budget");
    };
    let first=settle(&mut cache,&mut alloc,&[Vec3::ZERO,Vec3::X*156.]);
    let held=cache.current.as_ref().unwrap().clone();
    assert_eq!(held.sources.len(),40);
    assert!(fine(&held,&levels[0].detail) && fine(&held,&levels[39].detail),"one active view starved");
    assert!(!fine(&held,&levels[20].detail),"fixture failed to apply detail pressure");
    let image=held.resources().1;let original=read_pages(&g,&cache,&held,&readback);
    assert_eq!(original,expected(&held));
    let views=[Vec3::X*76.,Vec3::X*84.];
    for _ in 0..8 {
        let update=cache.set_streamed(device,&mut alloc,g.queue,family,&inputs,&levels,&views).unwrap();
        assert!(!update.pending && !update.geometry_changed,"residency changed physical geometry or dropped base coverage");
        assert_eq!(cache.current.as_ref().unwrap().sources.len(),40);
        assert_eq!(cache.current.as_ref().unwrap().resources().1,image);
        assert_eq!(read_pages(&g,&cache,&held,&readback),original,"old frame sampled an overwritten page");
    }
    drop(held);
    let moved=settle(&mut cache,&mut alloc,&views);
    let current=cache.current.as_ref().unwrap();
    assert!(fine(current,&levels[19].detail) && fine(current,&levels[21].detail));
    assert!(!fine(current,&levels[0].detail));
    assert_eq!(read_pages(&g,&cache,current,&readback),expected(current));
    let mut changed=inputs.clone();changed[20].material.emission=[3.,1.,0.5];
    let update=cache.set_streamed(device,&mut alloc,g.queue,family,&changed,&levels,&views).unwrap();
    assert!(!update.pending && !update.streaming && !update.geometry_changed);
    assert!(update.material_changes.meta[0]>0 && update.occlusion_changes.meta[0]==0,
        "material-only edits must refresh radiance without changing mesh geometry");
    assert_eq!(cache.current.as_ref().unwrap().resources().1,image);
    cache.set(device,&mut alloc,g.pool,g.queue,&[]).unwrap();
    assert!(cache.current.is_none());
    let update=cache.set_streamed(device,&mut alloc,g.queue,family,&inputs[..3],&levels,&[Vec3::ZERO]).unwrap();
    assert!(!update.pending && !update.streaming);
    assert_eq!(cache.current.as_ref().unwrap().sources.len(),3);
    assert_eq!(read_pages(&g,&cache,cache.current.as_ref().unwrap(),&readback),expected(cache.current.as_ref().unwrap()));
    cache.destroy(device,&mut alloc);readback.destroy(device,&mut alloc);drop(alloc);
    assert_eq!(g.validation_error_count(),before,"mesh residency produced Vulkan errors");
    eprintln!("[Flux mesh residency] 40 unique fields, 2 views, fixed128KiB atlas, initial={first} updates, reprioritized={moved} updates; retained old pages unchanged");
}
