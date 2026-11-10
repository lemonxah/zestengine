use super::*;

#[test]
fn adaptive_reflection_trace_rate_preserves_boundaries_and_fills_every_phase() -> Result<()> {
    let Some(g)=crate::gpu_tests::gpu() else {return Ok(())};
    let device=&g.device;let mut alloc=g.allocator()?;let errors=g.validation_error_count();
    let size=vk::Extent2D {width:17,height:9};let pixels=153usize;
    let mut images=Vec::new();
    for name in ["rate depth","rate material","rate raw","rate diagnostics"] {
        images.push(ScreenImage::new(device,&mut alloc,vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageUsageFlags::SAMPLED|vk::ImageUsageFlags::STORAGE|vk::ImageUsageFlags::TRANSFER_DST|vk::ImageUsageFlags::TRANSFER_SRC,size,1,name)?);
    }
    let mut upload=Buffer::new(device,&mut alloc,(pixels*16) as u64,vk::BufferUsageFlags::TRANSFER_SRC,MemoryLocation::CpuToGpu,"rate upload")?;
    let mut readback=Buffer::new(device,&mut alloc,(pixels*16) as u64,vk::BufferUsageFlags::TRANSFER_DST,MemoryLocation::GpuToCpu,"rate readback")?;
    let region=vk::BufferImageCopy::default().image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
        .image_extent(vk::Extent3D {width:17,height:9,depth:1});
    let sampler=unsafe {device.create_sampler(&vk::SamplerCreateInfo::default(),None)?};
    let bindings:Vec<_>=(0..4).map(|b|vk::DescriptorSetLayoutBinding::default().binding(b)
        .descriptor_type(if b==3 {vk::DescriptorType::STORAGE_IMAGE} else {vk::DescriptorType::COMBINED_IMAGE_SAMPLER})
        .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
    let (set_layout,layout,pool,set,pipeline)=unsafe {
        let set_layout=device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),None)?;
        let layout=device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout])
            .push_constant_ranges(&[vk::PushConstantRange {stage_flags:vk::ShaderStageFlags::COMPUTE,offset:0,size:16}]),None)?;
        let pool=device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
            vk::DescriptorPoolSize {ty:vk::DescriptorType::COMBINED_IMAGE_SAMPLER,descriptor_count:3},
            vk::DescriptorPoolSize {ty:vk::DescriptorType::STORAGE_IMAGE,descriptor_count:1}]),None)?;
        let set=device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[set_layout]))?[0];
        let shader=crate::screen_pass::shader_module(device,include_bytes!(concat!(env!("OUT_DIR"),"/reflection_trace_rate_test.comp.spv")))?;
        let pipeline=device.create_compute_pipelines(vk::PipelineCache::null(),&[vk::ComputePipelineCreateInfo::default().layout(layout)
            .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(shader).name(c"main"))],None).map_err(|(_,e)|e)?[0];
        device.destroy_shader_module(shader,None);
        (set_layout,layout,pool,set,pipeline)
    };
    for (b,image) in images.iter().enumerate() {unsafe {
        let info=[vk::DescriptorImageInfo::default().sampler(sampler).image_view(image.view)
            .image_layout(if b==3 {vk::ImageLayout::GENERAL} else {vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL})];
        device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(b as u32)
            .descriptor_type(bindings[b].descriptor_type).image_info(&info)],&[]);
    }}
    // Separate 2x2 regions contain mirrors, material changes, depth edges,
    // normal edges and background. Last row/column exercise odd dimensions.
    let coherent=|x:usize,y:usize|y<8 && (x<2 || (12..16).contains(&x));
    let mut counts=Vec::new();
    for factor in [1u32,2] {for phase in 0..4u32 {
        for channel in 0..3 {
            let data:Vec<[f32;4]>=(0..pixels).map(|i| {
                let (x,y)=(i%17,i/17);
                match channel {
                    0=>[if x==7 {0.7} else if x==11 {1.0} else {0.5},0.0,0.0,0.0],
                    1=>[if x==9 {0.6} else {0.5},0.5,
                        ((if (2..4).contains(&x) {1} else {25})*16+if x==5 {1} else {0}) as f32/1023.0,1.0],
                    _=>if factor==2 && coherent(x,y) && ((x&1)+2*(y&1))!=phase as usize {[0.0,0.0,0.0,-2.0-phase as f32]} else {[1.0,1.0,1.0,4.0]},
                }
            }).collect();
            upload.write(0,bytemuck::cast_slice(&data));
            one_time_submit(device,g.pool,g.queue,|cb|unsafe {
                barrier(device,cb,images[channel].image,vk::ImageLayout::UNDEFINED,vk::ImageLayout::TRANSFER_DST_OPTIMAL);
                device.cmd_copy_buffer_to_image(cb,upload.handle,images[channel].image,vk::ImageLayout::TRANSFER_DST_OPTIMAL,&[region]);
                barrier(device,cb,images[channel].image,vk::ImageLayout::TRANSFER_DST_OPTIMAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            })?;
        }
        one_time_submit(device,g.pool,g.queue,|cb|unsafe {
            barrier(device,cb,images[3].image,vk::ImageLayout::UNDEFINED,vk::ImageLayout::GENERAL);
            device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,pipeline);
            device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,layout,0,&[set],&[]);
            device.cmd_push_constants(cb,layout,vk::ShaderStageFlags::COMPUTE,0,bytemuck::cast_slice(&[factor,phase,0u32,0]));
            device.cmd_dispatch(cb,3,2,1);
            barrier(device,cb,images[3].image,vk::ImageLayout::GENERAL,vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            device.cmd_copy_image_to_buffer(cb,images[3].image,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,readback.handle,&[region]);
            device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::TRANSFER,vk::PipelineStageFlags::HOST,vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ)],&[],&[]);
        })?;
        let bytes=readback.read();let mut traced=0;
        for i in 0..pixels {
            let v:Vec<f32>=bytes[i*16..i*16+16].chunks_exact(4).map(|b|f32::from_ne_bytes(b.try_into().unwrap())).collect();
            let reduced=factor==2 && coherent(i%17,i/17);
            assert_eq!(v[0],if reduced {phase as f32} else {-1.0},"wrong adaptive coverage factor={factor} phase={phase} pixel={i}");
            assert_eq!(&v[2..],&[1.0,4.0],"untraced receiver became a hole or used a stale phase");
            traced+=v[1] as usize;
        }
        assert_eq!(traced,if factor==1 {153} else {117});counts.push(traced);
    }}
    unsafe {device.destroy_pipeline(pipeline,None);device.destroy_pipeline_layout(layout,None);device.destroy_descriptor_pool(pool,None);
        device.destroy_descriptor_set_layout(set_layout,None);device.destroy_sampler(sampler,None);}
    for image in &mut images {image.destroy(device,&mut alloc);}
    upload.destroy(device,&mut alloc);readback.destroy(device,&mut alloc);drop(alloc);
    assert_eq!(g.validation_error_count(),errors);
    eprintln!("[Reflection trace rate] traced counts={counts:?}; 75% fewer in coherent regions, mirrors/material/depth/normal/background/odd edges full resolution; all phases reconstructed");
    Ok(())
}
