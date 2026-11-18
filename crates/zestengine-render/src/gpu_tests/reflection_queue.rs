//! Compare actual compacted and inline reflection HDR, including glossy rays,
//! screen-edge blends, multiple bounces and independent resized views. Audit
//! GPU-generated indirect commands and every queued receiver/weight as well.
use glam::{Mat4,Quat,Vec3};
use ash::vk;
use crate::{CameraData,FluxSettings,Renderer};
use super::planar::{cube,draw,lit};
fn hdr(r:&Renderer,view:usize)->anyhow::Result<(vk::Extent2D,Vec<f32>)> {
    use crate::alloc::{Buffer,one_time_submit};
    let (image,_,extent,_,_)=r.rt_reflect.as_ref().unwrap().reflection_history(view);
    let device=&r.ctx.device;unsafe {device.device_wait_idle()?;}
    let allocator=r.allocator();let mut alloc=allocator.lock().unwrap();
    let size=extent.width as usize*extent.height as usize*8;
    let mut host=Buffer::new(device,&mut alloc,size as u64,vk::BufferUsageFlags::TRANSFER_DST,
        gpu_allocator::MemoryLocation::GpuToCpu,"reflection queue HDR comparison")?;
    one_time_submit(device,r.command_pool,r.ctx.queue,|cb|unsafe {
        crate::image_barrier(device,cb,image,vk::ImageAspectFlags::COLOR,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::PipelineStageFlags2::COMPUTE_SHADER,vk::AccessFlags2::SHADER_READ|vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::TRANSFER,vk::AccessFlags2::TRANSFER_READ);
        device.cmd_copy_image_to_buffer(cb,image,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,host.handle,&[vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
            .image_extent(vk::Extent3D{width:extent.width,height:extent.height,depth:1})]);
        device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::TRANSFER,vk::PipelineStageFlags::HOST,vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ)],&[],&[]);
        crate::image_barrier(device,cb,image,vk::ImageAspectFlags::COLOR,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::TRANSFER,vk::AccessFlags2::TRANSFER_READ,vk::PipelineStageFlags2::COMPUTE_SHADER,vk::AccessFlags2::SHADER_READ);
    })?;
    let data=host.read()[..size].as_chunks::<2>().0.iter().map(|b|half::f16::from_bits(u16::from_ne_bytes([b[0],b[1]])).to_f32()).collect();
    host.destroy(device,&mut alloc);Ok((extent,data))
}
fn audit(r:&Renderer,view:usize,extent:vk::Extent2D,samples:u32,center_pending:Option<bool>)->anyhow::Result<u32> {
    let (buffer,bytes)=r.rt_reflect.as_ref().unwrap().reflection_queue(view).unwrap();
    let header=super::flux_radiance::read(r,buffer,0,16)?;
    let h:Vec<u32>=header.as_chunks::<4>().0.iter().map(|b|u32::from_ne_bytes(*b)).collect();
    let count=h[3];let pixels=extent.width*extent.height;let stride=5+samples.div_ceil(2);
    assert!(count<=pixels && 16+u64::from(count)*u64::from(stride)*4<=bytes,"queue exceeded allocation");
    assert_eq!(h[..3],[count.div_ceil(64).min(65535),count.div_ceil(65535*64).max(1),1]);
    if count==0 {return Ok(0);}
    let data=super::flux_radiance::read(r,buffer,16,(count*stride*4) as usize)?;
    let mut seen=vec![false;pixels as usize];
    for record in data.chunks_exact(stride as usize*4) {
        let word=|i:usize|u32::from_ne_bytes(record[i*4..i*4+4].try_into().unwrap());
        let pixel=word(0);assert!(pixel<pixels && !seen[pixel as usize],"duplicate/out-of-range world job");seen[pixel as usize]=true;
        for i in 1..5 {assert!(f32::from_bits(word(i)).is_finite(),"non-finite screen lighting entered queue");}
        let mut unresolved=false;
        for k in 0..samples {let bits=(word(5+(k/2) as usize)>>(16*(k&1))) as u16;
            let weight=half::f16::from_bits(bits).to_f32();assert!((0.0..=1.0).contains(&weight));unresolved|=weight>0.0;}
        assert!(unresolved,"fully resolved receiver entered world queue");
    }
    if let Some(expected)=center_pending {assert_eq!(seen[(extent.height/2*extent.width+extent.width/2) as usize],expected,"screen-resolved center was queued, or a world miss was lost");}
    Ok(count)
}
pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    if !r.ctx.ray_tracing() {return Ok(());}
    r.set_gi_enabled(false);r.gi_set_emitters(&[]);
    let mesh=r.upload_mesh(&cube())?;
    let mirror=lit(r,[0.9;4],[0.0;3],0.25,1.0);
    let emitter=lit(r,[0.0,0.0,0.0,1.0],[2.0,0.4,0.1],1.0,0.0);
    let draws=[draw(mesh,mirror,Mat4::from_scale_rotation_translation(Vec3::new(8.0,0.2,8.0),Quat::IDENTITY,Vec3::new(0.0,-0.1,0.0)),-1),
        draw(mesh,emitter,Mat4::from_scale_rotation_translation(Vec3::new(8.0,4.0,0.2),Quat::IDENTITY,Vec3::new(0.0,2.0,-2.5)),-1)];
    let eye=Vec3::new(0.0,2.0,4.0);
    let mut cam=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let mut input=crate::cascade_tests::input(cam);input.camera_preview=Some(cam);input.viewport_extent=Some([323,181]);input.draws=&draws;
    input.light.intensity=0.0;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;input.draw_skybox=false;
    input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
    r.rt_reflect.as_mut().unwrap().denoise_override=Some(false);
    let mut compared=0usize;let mut queued_jobs=0u64;
    let mut world_density=[[0.0f32;2];2];
    for (case,(samples,screen,fov,bounces,downsample)) in [(1,true,1.0,1,1),(4,true,0.7,1,1),(16,true,1.0,3,1),
        (4,false,1.0,1,1),(4,true,0.4,3,1),(4,false,1.0,1,2),(4,true,0.7,1,2)].into_iter().enumerate() {
        r.set_flux_settings(FluxSettings {rt_trace:true,reflection_mode:2,ssr_enabled:false,reflection_screen_traces:screen,rt_samples:samples,rt_bounces:bounces,reflection_downsample:downsample,..Default::default()});
        cam.proj=Mat4::perspective_rh(fov,16.0/9.0,0.1,100.0);input.camera=cam;
        input.camera_preview=Some(CameraData{proj:Mat4::perspective_rh(0.6,16.0/9.0,0.1,100.0),..cam});
        input.viewport_extent=Some([323-case as u32*2,181-case as u32*2]);
        let mut reference=None;
        for queued in [false,true] {
            let rt=r.rt_reflect.as_mut().unwrap();rt.queue_override=Some(queued);
            rt.sampling_frame_override=Some(13.0+case as f32);rt.invalidate_reflection_history();
            for _ in 0..3 {r.render(&input)?;}
            let current=[hdr(r,0)?,hdr(r,1)?];
            if let Some(ref expected)=reference {
                let expected:&[(vk::Extent2D,Vec<f32>);2]=expected;
                let mut maximum=0.0f32;
                for view in 0..2 {
                    assert_eq!(current[view].0,expected[view].0);
                    for (&a,&b) in current[view].1.iter().zip(&expected[view].1) {
                        assert!(a.is_finite() && b.is_finite());
                        let error=(a-b).abs();maximum=maximum.max(error);
                        // FP16 queue fractions and changed addition order may
                        // differ by a few final HDR half-float ULPs.
                        assert!(error<=0.003+0.003*a.abs().max(b.abs()),"queue/reference error case={case} view={view}: {a} versus {b}");
                        compared+=1;
                    }
                    // Glossy samples can miss the screen even at one ray/pixel.
                    // Only world-only, full-density tracing guarantees this job.
                    let count=audit(r,view,current[view].0,samples,if !screen && downsample==1 {Some(true)} else {None})?;
                    queued_jobs+=u64::from(count);
                    if !screen {world_density[(downsample-1) as usize][view]=count as f32/(current[view].0.width*current[view].0.height) as f32;}
                }
                eprintln!("[Flux reflection queue] samples={samples} screen={screen} fov={fov} bounces={bounces} downsample={downsample} max_error={maximum}");
            } else {reference=Some(current);}
        }
    }
    for view in 0..2 {
        assert!(world_density[0][view]>0.05,"fixture lacks world reflection jobs");
        assert!(world_density[1][view]<world_density[0][view]*0.5,"adaptive tracing did not reduce world jobs: {world_density:?}");
    }
    eprintln!("[Flux reflection density] full/adaptive Scene/Camera job fractions: {world_density:?}");
    if std::env::var_os("ZESTENGINE_REFLECTION_QUEUE_BENCH").is_some() {
        input.camera_preview=None;input.viewport_extent=Some([1280,720]);
        for (samples,screen) in [(1,true),(4,true),(16,true),(4,false)] {
            r.set_flux_settings(FluxSettings {rt_trace:true,reflection_mode:2,ssr_enabled:false,reflection_screen_traces:screen,rt_samples:samples,rt_bounces:3,..Default::default()});
            let mut timings=[Vec::new(),Vec::new()];
            for queued in [false,true,true,false] {
                r.rt_reflect.as_mut().unwrap().queue_override=Some(queued);
                for _ in 0..16 {r.render(&input)?;}
                for _ in 0..48 {r.render(&input)?;timings[usize::from(queued)].push(r.stats().gpu_reflect_ms);}
            }
            let medians=timings.map(|mut values|{values.sort_by(f32::total_cmp);values[values.len()/2]});
            eprintln!("[Flux reflection queue timing] samples={samples} screen={screen} inline={}ms queued={}ms delta={}%, 1280x720 alternating blocks",medians[0],medians[1],100.0*(medians[1]/medians[0]-1.0));
        }
        r.set_flux_settings(FluxSettings {rt_trace:true,reflection_mode:2,rt_samples:4,..Default::default()});
        r.rt_reflect.as_mut().unwrap().queue_override=Some(true);
    }
    let saved=r.flux_settings;
    r.set_flux_settings(FluxSettings {ssr_intensity:0.0,..saved});
    input.camera_preview=Some(cam);
    r.render(&input)?;
    for view in 0..2 {let (extent,data)=hdr(r,view)?;assert!(data.iter().all(|v|*v==0.0));assert_eq!(audit(r,view,extent,saved.rt_samples,None)?,0);}
    r.set_flux_settings(saved);
    // A hidden view retains its last allocation; resuming with no reflectors
    // must reset the indirect count and overwrite every previous raw pixel.
    input.camera_preview=None;for _ in 0..3 {r.render(&input)?;}
    input.camera_preview=Some(cam);input.draws=&[];r.render(&input)?;
    for view in 0..2 {let (extent,data)=hdr(r,view)?;assert!(data.iter().all(|v|*v==0.0));assert_eq!(audit(r,view,extent,4,None)?,0);}
    eprintln!("[Flux reflection queue audit] compared={compared} components, jobs={queued_jobs}, empty queues cleared");
    let rt=r.rt_reflect.as_mut().unwrap();rt.queue_override=None;rt.sampling_frame_override=None;rt.denoise_override=None;
    Ok(())
}
