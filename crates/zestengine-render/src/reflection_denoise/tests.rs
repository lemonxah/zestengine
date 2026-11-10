mod rate;
use super::*;
use crate::alloc::one_time_submit;

#[test]
fn reflection_variance_filter_preserves_edges_mirrors_and_signed_energy() -> Result<()> {
    let Some(g)=crate::gpu_tests::gpu() else {return Ok(())};
    let device=&g.device;let mut alloc=g.allocator()?;let errors=g.validation_error_count();
    let size=vk::Extent2D {width:64,height:32};let pixels=2048usize;
    let layers=vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1);
    let region=vk::BufferImageCopy::default().image_subresource(layers)
        .image_extent(vk::Extent3D {width:64,height:32,depth:1});
    let mut inputs=Vec::new();
    for name in ["variance base","variance depth","variance material"] {
        inputs.push(ScreenImage::new(device,&mut alloc,vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageUsageFlags::SAMPLED|vk::ImageUsageFlags::TRANSFER_DST,size,1,name)?);
    }
    let mut output=ScreenImage::new(device,&mut alloc,vk::Format::R16G16B16A16_SFLOAT,
        vk::ImageUsageFlags::STORAGE|vk::ImageUsageFlags::TRANSFER_SRC,size,1,"variance output")?;
    let mut upload=Buffer::new(device,&mut alloc,(pixels*16) as u64,vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,"variance upload")?;
    let mut readback=Buffer::new(device,&mut alloc,(pixels*16) as u64,vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuToCpu,"variance output and history")?;
    for (index,image) in inputs.iter().enumerate() {
        let data:Vec<[f32;4]>=(0..pixels).map(|i|match index {
            0=>[0.25,0.25,0.25,1.0],
            1=>[if i%64<32 {0.5} else {0.7},0.0,0.0,0.0],
            _=>[0.5,0.5,((if i/64<8 {1} else {25})*16) as f32/1023.0,0.0],
        }).collect();
        upload.write(0,bytemuck::cast_slice(&data));
        one_time_submit(device,g.pool,g.queue,|cb|unsafe {
            barrier(device,cb,image.image,vk::ImageLayout::UNDEFINED,vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            device.cmd_copy_buffer_to_image(cb,upload.handle,image.image,vk::ImageLayout::TRANSFER_DST_OPTIMAL,&[region]);
            barrier(device,cb,image.image,vk::ImageLayout::TRANSFER_DST_OPTIMAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        })?;
    }
    let mut denoiser=Denoiser::new(device)?;
    let mut samples=[Vec::<Vec<f32>>::new(),Vec::new()];
    let mut histories=[Vec::<Vec<f32>>::new(),Vec::new()];
    for serial in 1..=44u64 {
        let light_on=serial<=32;
        let values:Vec<f32>=(0..pixels).map(|i| {
            if i%64>=32 {-0.2} else if serial>=37 {
                let left=if serial<41 {8} else {22};
                if (left..left+4).contains(&(i%64)) {1.0} else {0.0}
            } else if !light_on {0.0} else {
                let mut x=(i as u32).wrapping_mul(747796405)^(serial as u32).wrapping_mul(2891336453);
                x=((x>>((x>>28)+4))^x).wrapping_mul(277803737);x=(x>>22)^x;
                1.0+((x&65535) as f32/65535.0-0.5)*1.6
            }
        }).collect();
        let raw:Vec<[u16;4]>=values.iter().map(|&v|[half::f16::from_f32(v).to_bits();4])
            .map(|mut v|{v[3]=half::f16::from_f32(4.0).to_bits();v}).collect();
        upload.write(0,bytemuck::cast_slice(&raw));
        for view in 0..2 {
            one_time_submit(device,g.pool,g.queue,|cb|unsafe {
                let (image,_)=denoiser.prepare(device,&mut alloc,cb,view,serial,size).unwrap();
                barrier(device,cb,image,vk::ImageLayout::GENERAL,vk::ImageLayout::TRANSFER_DST_OPTIMAL);
                device.cmd_copy_buffer_to_image(cb,upload.handle,image,vk::ImageLayout::TRANSFER_DST_OPTIMAL,&[region]);
                barrier(device,cb,image,vk::ImageLayout::TRANSFER_DST_OPTIMAL,vk::ImageLayout::GENERAL);
                denoiser.record(device,cb,view,0,serial,inputs[0].view,inputs[1].view,inputs[2].view,output.image,output.view,
                    &Reconstruction {inv_proj:Mat4::IDENTITY.to_cols_array_2d(),inv_view:Mat4::IDENTITY.to_cols_array_2d(),
                        camera:[0.0,0.0,3.0,0.0],key:[0.0;10],enabled:true,fog:Fog::default()});
                let history=denoiser.debug_history(view).0;
                for (image,layout,offset) in [(output.image,vk::ImageLayout::GENERAL,0),(history,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,(pixels*8) as u64)] {
                    barrier(device,cb,image,layout,vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
                    device.cmd_copy_image_to_buffer(cb,image,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,readback.handle,&[region.buffer_offset(offset)]);
                    barrier(device,cb,image,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,layout);
                }
                device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::TRANSFER,vk::PipelineStageFlags::HOST,vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ)],&[],&[]);
            })?;
            let data=readback.read();
            let red=|offset:usize,i:usize|half::f16::from_bits(u16::from_ne_bytes(data[offset+i*8..offset+i*8+2].try_into().unwrap())).to_f32();
            for y in 0..8 {for x in 4..28 {let i=y*64+x;
                assert!((red(0,i)-(values[i]+0.25)).abs()<0.003,"sharp mirror was blurred");
            }}
            for y in 10..30 {
                assert!((red(0,y*64+32)-0.05).abs()<0.002,"filter leaked across receiver depth or clamped signed reflection");
            }
            if serial==36 {for y in 10..30 {for x in 4..28 {
                assert!((red(0,y*64+x)-0.25).abs()<0.003,"switched-off reflection retained bright history");
            }}}
            if serial==44 {for y in 12..28 {
                assert!((red(0,y*64+9)-0.25).abs()<0.003,"moving reflected feature left a trail");
                assert!(red(0,y*64+23)>0.85,"moving reflected feature lost its new position");
            }}
            if (9..=32).contains(&serial) {
                let points:Vec<_>=(12..28).flat_map(|y|(6..26).map(move |x|y*64+x)).collect();
                samples[view].push(points.iter().map(|&i|red(0,i)-0.25).collect());
                histories[view].push(points.iter().map(|&i|red(pixels*8,i)).collect());
            }
        }
    }
    let stats=|frames:&Vec<Vec<f32>>| {
        let count=frames.len() as f32;let mut mean=0.0;let mut variance=0.0;
        for i in 0..frames[0].len() {
            let m=frames.iter().map(|f|f[i]).sum::<f32>()/count;mean+=m;
            variance+=frames.iter().map(|f|(f[i]-m).powi(2)).sum::<f32>()/count;
        }
        (mean/frames[0].len() as f32,variance/frames[0].len() as f32)
    };
    let mut results=Vec::new();
    for view in 0..2 {
        let a=stats(&histories[view]);let b=stats(&samples[view]);results.push((a,b));
        assert!(b.1<a.1*0.9,"variance pass did not improve temporal-only noise: {a:?} -> {b:?}");
        assert!((b.0-1.0).abs()<0.05 && (b.0-a.0).abs()<0.02,"reflection energy changed: {a:?} -> {b:?}");
    }
    denoiser.destroy(device,&mut alloc);for image in &mut inputs {image.destroy(device,&mut alloc);}
    output.destroy(device,&mut alloc);upload.destroy(device,&mut alloc);readback.destroy(device,&mut alloc);drop(alloc);
    assert_eq!(g.validation_error_count(),errors,"reflection filter Vulkan validation error");
    eprintln!("[Reflection variance] Scene/Camera temporal -> spatial: {results:?}; mirror, depth edge, signed energy, light-off and moving reflected feature passed");
    Ok(())
}

#[test]
fn reflection_quality_selects_tracing_density_per_view() {
    let base=crate::FluxSettings {rt_samples:4,..Default::default()};
    let low=base.for_view(crate::FluxPostFx {reflection_quality:Some(0.25),..Default::default()});
    let full=base.for_view(crate::FluxPostFx {reflection_quality:Some(1.0),..Default::default()});
    assert_eq!((low.rt_samples,low.reflection_downsample),(1,2));
    assert_eq!((full.rt_samples,full.reflection_downsample),(4,1));
    assert_eq!(base.for_view(crate::FluxPostFx {reflection_quality:Some(0.5),..Default::default()}).reflection_downsample,1);
    assert_eq!(base.reflection_downsample,1);
    assert_eq!(base.for_view(crate::FluxPostFx {reflection_quality:Some(f32::NAN),..Default::default()}).reflection_downsample,1);
}
