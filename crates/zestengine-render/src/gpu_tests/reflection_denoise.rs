//! Actual glossy reflections of a high-contrast emitter, with diffuse GI off.
//! Compare temporal pixel variance and mean energy; reject stale light after off.
use glam::{Mat4,Quat,Vec3};
use crate::{CameraData,FluxSettings,Renderer};
use super::planar::{cube,draw,lit};
fn crop(image:&(u32,u32,Vec<u8>),cam:CameraData)->Vec<f32> {
    let (w,h,data)=image;let p=(cam.proj*cam.view).project_point3(Vec3::ZERO);
    let x=((p.x*0.5+0.5)* *w as f32) as i32;let y=((0.5-p.y*0.5)* *h as f32) as i32;
    let mut out=Vec::new();for j in -10..=10 {for i in -20..=20 {out.push(data[(((y+j) as u32*w+(x+i) as u32)*4) as usize] as f32);}}out
}
fn history(r:&Renderer,view:usize,point:Vec3,geometry:bool)->anyhow::Result<Vec<f32>> {
    use ash::vk;use gpu_allocator::MemoryLocation;use crate::alloc::{Buffer,one_time_submit};
    let (color,meta,size,vp,_)=r.rt_reflect.as_ref().unwrap().reflection_history(view);
    let image=if geometry {meta} else {color};let q=vp.project_point3(point);
    let center=[((q.x*0.5+0.5)*size.width as f32) as i32,((0.5-q.y*0.5)*size.height as f32) as i32];
    let (width,height,offset)=if geometry {(1,1,vk::Offset3D {x:center[0],y:center[1],z:0})}
        else {(41,21,vk::Offset3D {x:center[0]-20,y:center[1]-10,z:0})};
    let bytes=width*height*if geometry {16} else {8};let device=&r.ctx.device;unsafe {device.device_wait_idle()?;}
    let allocator=r.allocator();let mut alloc=allocator.lock().unwrap();
    let mut output=Buffer::new(device,&mut alloc,bytes as u64,vk::BufferUsageFlags::TRANSFER_DST,MemoryLocation::GpuToCpu,"reflection history readback")?;
    one_time_submit(device,r.command_pool,r.ctx.queue,|cb| {
        crate::image_barrier(device,cb,image,vk::ImageAspectFlags::COLOR,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::PipelineStageFlags2::COMPUTE_SHADER,vk::AccessFlags2::SHADER_READ|vk::AccessFlags2::SHADER_WRITE,vk::PipelineStageFlags2::TRANSFER,vk::AccessFlags2::TRANSFER_READ);
        unsafe {device.cmd_copy_image_to_buffer(cb,image,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,output.handle,&[vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
            .image_offset(offset).image_extent(vk::Extent3D {width,height,depth:1})]);
            device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::TRANSFER,vk::PipelineStageFlags::HOST,vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ)],&[],&[]);}
        crate::image_barrier(device,cb,image,vk::ImageAspectFlags::COLOR,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::TRANSFER,vk::AccessFlags2::TRANSFER_READ,vk::PipelineStageFlags2::COMPUTE_SHADER,vk::AccessFlags2::SHADER_READ);
    })?;
    let data=output.read();let values=if geometry {data[..16].chunks_exact(4).map(|b|f32::from_ne_bytes(b.try_into().unwrap())).collect()}
        else {data[..bytes as usize].chunks_exact(8).map(|b|half::f16::from_bits(u16::from_ne_bytes([b[0],b[1]])).to_f32()).collect()};
    output.destroy(device,&mut alloc);Ok(values)
}
fn frames(r:&Renderer,view:usize,point:Vec3)->anyhow::Result<u32> {Ok((history(r,view,point,true)?[3].round() as u32)>>12)}
fn measure(frames:&[Vec<f32>])->(f32,f32) {
    let n=frames.len() as f32;let pixels=frames[0].len();let mut mean=0.0;let mut variance=0.0;
    for i in 0..pixels {let m=frames.iter().map(|f|f[i]).sum::<f32>()/n;mean+=m;
        variance+=frames.iter().map(|f|(f[i]-m).powi(2)).sum::<f32>()/n;}
    (mean/pixels as f32,variance/pixels as f32)
}
pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    if !r.ctx.ray_tracing() {return Ok(());}
    r.set_gi_enabled(false);r.gi_set_emitters(&[]);
    r.set_flux_settings(FluxSettings {reflection_mode:2,ssr_enabled:false,rt_samples:2,rt_bounces:1,..Default::default()});
    let mesh=r.upload_mesh(&cube())?;
    let mirror=lit(r,[0.95;4],[0.0;3],0.3,1.0);let emitter=lit(r,[0.0,0.0,0.0,1.0],[2.0;3],1.0,0.0);
    let pixels:Vec<u8>=(0..32*32).flat_map(|i| {let value=if ((i%32)/4+(i/32)/4)%2==0 {255} else {0};[value,value,value,255]}).collect();
    let texture=r.upload_texture(&crate::TextureData {width:32,height:32,pixels,srgb:false,hdr:false})?;
    let mut slots=[None;16];slots[3]=Some(texture);r.set_material_textures(emitter,&slots);
    let draws=[draw(mesh,mirror,Mat4::from_scale_rotation_translation(Vec3::new(8.0,0.2,8.0),Quat::IDENTITY,Vec3::new(0.0,-0.1,0.0)),-1),
        draw(mesh,emitter,Mat4::from_scale_rotation_translation(Vec3::new(8.0,4.0,0.2),Quat::IDENTITY,Vec3::new(0.0,2.0,-2.5)),-1)];
    let eye=Vec3::new(0.0,2.0,4.0);let cam=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let mut input=crate::cascade_tests::input(cam);input.camera_preview=Some(cam);input.viewport_extent=Some([323,181]);input.draws=&draws;
    input.light.intensity=0.0;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;input.draw_skybox=false;
    input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
    let mut results=Vec::new();let mut hdr_results=Vec::new();let mut accumulated=[0u32;2];
    for enabled in [false,true] {
        r.rt_reflect.as_mut().unwrap().denoise_override=Some(enabled);
        r.rt_reflect.as_mut().unwrap().invalidate_reflection_history();
        for _ in 0..24 {r.render(&input)?;}
        let mut scene=Vec::new();let mut camera=Vec::new();let mut scene_hdr=Vec::new();let mut camera_hdr=Vec::new();
        for _ in 0..24 {r.render(&input)?;scene.push(crop(&r.debug_read_viewport()?.unwrap(),cam));camera.push(crop(&r.debug_read_camera_preview()?.unwrap(),cam));scene_hdr.push(history(r,0,Vec3::ZERO,false)?);camera_hdr.push(history(r,1,Vec3::ZERO,false)?);
            // A two-ray glossy sample can legitimately reject the centre
            // pixel's history on any one frame. Verify accumulation over the
            // measured sequence, not only its arbitrary final random phase.
            if enabled {for view in 0..2 {accumulated[view]=accumulated[view].max(frames(r,view,Vec3::ZERO)?);}}
        }
        results.push([measure(&scene),measure(&camera)]);hdr_results.push([measure(&scene_hdr),measure(&camera_hdr)]);
    }
    eprintln!("[Flux reflection denoise] raw={:?}, reconstructed={:?}",results[0],results[1]);
    eprintln!("[Flux reflection denoise HDR] raw={:?}, reconstructed={:?}",hdr_results[0],hdr_results[1]);
    for view in 0..2 {
        assert!(accumulated[view]>2,"temporal history did not accumulate during measurement");
        let (mean,variance)=hdr_results[1][view];let (raw_mean,raw_variance)=hdr_results[0][view];
        assert!(variance<raw_variance*0.65 && (mean-raw_mean).abs()<raw_mean*0.1,"HDR variance/energy regression: {hdr_results:?}");
        let (raw_mean,raw_variance)=results[0][view];let (mean,variance)=results[1][view];
        assert!(raw_variance>1.0,"fixture lacks stochastic reflection variance");
        assert!(variance<raw_variance*0.65,"reflection reconstruction did not reduce variance: {results:?}");
        assert!((mean-raw_mean).abs()<raw_mean*0.15,"reflection reconstruction changed mean energy: {results:?}");
    }
    // No stale geometry after a moving reflector, viewport resize, or hidden view.
    let mut raised_draws=draws;raised_draws[0].transform=Mat4::from_scale_rotation_translation(Vec3::new(8.0,0.2,8.0),Quat::IDENTITY,Vec3::new(0.0,0.5,0.0));
    input.draws=&raised_draws;r.render(&input)?;let raised=Vec3::new(0.0,0.6,0.0);
    for view in 0..2 {assert_eq!(frames(r,view,raised)?,1,"old receiver depth survived geometry motion");}
    for _ in 0..24 {r.render(&input)?;}
    let camera_serial=r.rt_reflect.as_ref().unwrap().reflection_history(1).4;input.camera_preview=None;
    for _ in 0..3 {r.render(&input)?;}
    assert_eq!(r.rt_reflect.as_ref().unwrap().reflection_history(1).4,camera_serial);
    input.camera_preview=Some(cam);r.render(&input)?;assert_eq!(frames(r,1,raised)?,1,"hidden Camera reused history");
    input.viewport_extent=Some([321,179]);r.render(&input)?;assert_eq!(frames(r,0,raised)?,1,"resized Scene reused history");
    // A two-ray sample can legitimately reject history at this one pixel when
    // its hit distance changes. Establish the light-off precondition in both
    // views within a bounded warmup instead of depending on one random phase.
    let mut populated=false;
    for _ in 0..48 {
        r.render(&input)?;
        if frames(r,0,raised)?>2 && frames(r,1,raised)?>2 {populated=true;break;}
    }
    assert!(populated,"glossy history was not populated before light-off");
    // A lighting change must not retain old bright reflection history.
    r.material_params_mut(emitter).emission_intensity=0.0;
    for _ in 0..4 {r.render(&input)?;}
    for image in [r.debug_read_viewport()?.unwrap(),r.debug_read_camera_preview()?.unwrap()] {
        let values=crop(&image,cam);assert!(values.iter().all(|v|*v<2.0),"reflection history retained switched-off emission");
    }
    r.material_params_mut(emitter).emission_intensity=1.0;
    r.material_params_mut(mirror).roughness=0.02;
    for _ in 0..3 {r.render(&input)?;}
    for view in 0..2 {assert_eq!(frames(r,view,raised)?,1,"sharp mirror was temporally blurred");}
    r.set_anti_aliasing(crate::AntiAliasing::Taa);
    for _ in 0..4 {
        r.render(&input)?;
        let actual=r.rt_reflect.as_ref().unwrap().reflection_history(0).3.to_cols_array();
        let depth_vp=glam::Mat4::from_cols_array_2d(&r.last_frame_ubo.as_ref().unwrap().view_proj).to_cols_array();
        assert!(actual.iter().zip(depth_vp).all(|(a,b)|(a-b).abs()<1e-4),"reflection history diverged from jittered depth projection");
    }
    r.set_anti_aliasing(crate::AntiAliasing::None);
    r.rt_reflect.as_mut().unwrap().denoise_override=None;
    Ok(())
}
