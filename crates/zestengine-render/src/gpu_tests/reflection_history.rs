//! A moving view can retain surface history while reflected-hit history is
//! disoccluded, or vice versa. Never sample a fictitious point between them.
use ash::vk;
use glam::{Mat4,Vec2,Vec3};
use gpu_allocator::MemoryLocation;
use crate::{Renderer,alloc::{Buffer,one_time_submit},screen_pass::ScreenImage};
use crate::reflection_denoise::{Denoiser,Reconstruction,Fog};

fn transition(device:&ash::Device,cb:vk::CommandBuffer,image:vk::Image,from:vk::ImageLayout,to:vk::ImageLayout) {
    crate::image_barrier(device,cb,image,vk::ImageAspectFlags::COLOR,from,to,
        vk::PipelineStageFlags2::ALL_COMMANDS,vk::AccessFlags2::MEMORY_READ|vk::AccessFlags2::MEMORY_WRITE,
        vk::PipelineStageFlags2::ALL_COMMANDS,vk::AccessFlags2::MEMORY_READ|vk::AccessFlags2::MEMORY_WRITE);
}
pub(super) fn run(r:&Renderer)->anyhow::Result<()> {
    let device=&r.ctx.device;let allocator=r.allocator();let mut alloc=allocator.lock().unwrap();
    let size=vk::Extent2D {width:128,height:64};let pixels=(size.width*size.height) as usize;
    let projection=Mat4::perspective_rh(1.0,2.0,0.1,100.0);
    let old_eye=Vec3::new(0.0,0.0,3.0);let eye=Vec3::new(1.0,0.0,3.0);
    let old_view=Mat4::look_at_rh(old_eye,Vec3::ZERO,Vec3::Y);
    let view=Mat4::look_at_rh(eye,Vec3::X,Vec3::Y);
    let old_vp=projection*old_view;let vp=projection*view;
    let depth=old_vp.project_point3(Vec3::ZERO).z;
    let center=[64,32];let uv=Vec2::new(64.5/128.0,32.5/64.0);
    let world=vp.inverse().project_point3(Vec3::new(uv.x*2.0-1.0,1.0-uv.y*2.0,depth));
    let pixel=|p:Vec3| {let q=old_vp.project_point3(p);Vec2::new((q.x*0.5+0.5)*128.0-0.5,(0.5-q.y*0.5)*64.0-0.5)};
    let surface=pixel(world);let hit=pixel(world-(eye-world).normalize()*4.0);
    assert!((surface.x-hit.x).abs()>6.0,"fixture lacks reflection parallax");
    let mut denoiser=Denoiser::new(device)?;
    let mut inputs=Vec::new();
    for name in ["history test base","history test depth","history test gbuffer"] {
        inputs.push(ScreenImage::new(device,&mut alloc,vk::Format::R32G32B32A32_SFLOAT,
            vk::ImageUsageFlags::SAMPLED|vk::ImageUsageFlags::TRANSFER_DST,size,1,name)?);
    }
    let mut output=ScreenImage::new(device,&mut alloc,vk::Format::R16G16B16A16_SFLOAT,
        vk::ImageUsageFlags::STORAGE,size,1,"history test result")?;
    let mut upload=Buffer::new(device,&mut alloc,(pixels*16) as u64,vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,"history test disocclusion")?;
    let mut readback=Buffer::new(device,&mut alloc,16,vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuToCpu,"history test count")?;
    let range=vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
    one_time_submit(device,r.command_pool,r.ctx.queue,|cb|unsafe {
        for (image,value) in inputs.iter().zip([[0.2,0.2,0.2,1.0],[depth,0.0,0.0,0.0],[0.5,0.5,(19*16) as f32/1023.0,0.0]]) {
            transition(device,cb,image.image,vk::ImageLayout::UNDEFINED,vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            device.cmd_clear_color_image(cb,image.image,vk::ImageLayout::TRANSFER_DST_OPTIMAL,&vk::ClearColorValue {float32:value},&[range]);
            transition(device,cb,image.image,vk::ImageLayout::TRANSFER_DST_OPTIMAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        }
    })?;
    let mut counts=Vec::new();
    for kept in 0..3 {
        let serial=1+kept*2;
        // Seed real production histories from an unfiltered constant plane.
        one_time_submit(device,r.command_pool,r.ctx.queue,|cb| {
            let (raw,raw_view)=denoiser.prepare(device,&mut alloc,cb,0,serial,size).unwrap();
            clear_raw(device,cb,raw,raw_view,size);
            denoiser.record(device,cb,0,0,serial,inputs[0].view,inputs[1].view,inputs[2].view,output.image,output.view,
                &Reconstruction {inv_proj:projection.inverse().to_cols_array_2d(),inv_view:old_view.inverse().to_cols_array_2d(),
                    camera:old_eye.extend(0.0).to_array(),key:[0.0;10],enabled:false,fog:Fog::default()});
        })?;
        let (_,meta,_,_,_)=denoiser.debug_history(0);
        let mut geometry=vec![[depth,0.5,0.5,0.0];pixels];
        if kept<2 {
            let first=if kept==0 {surface.floor()} else {hit.floor()};
            for y in 0..2 {for x in 0..2 {
                let index=(first.y as usize+y)*128+first.x as usize+x;
                geometry[index][3]=((8u32<<12)|(19u32<<6)) as f32;
            }}
        }
        upload.write(0,bytemuck::cast_slice(&geometry));
        one_time_submit(device,r.command_pool,r.ctx.queue,|cb|unsafe {
            transition(device,cb,meta,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            device.cmd_copy_buffer_to_image(cb,upload.handle,meta,vk::ImageLayout::TRANSFER_DST_OPTIMAL,&[vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                .image_extent(vk::Extent3D {width:128,height:64,depth:1})]);
            transition(device,cb,meta,vk::ImageLayout::TRANSFER_DST_OPTIMAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        })?;
        one_time_submit(device,r.command_pool,r.ctx.queue,|cb| {
            let (raw,raw_view)=denoiser.prepare(device,&mut alloc,cb,0,serial+1,size).unwrap();
            clear_raw(device,cb,raw,raw_view,size);
            denoiser.record(device,cb,0,0,serial+1,inputs[0].view,inputs[1].view,inputs[2].view,output.image,output.view,
                &Reconstruction {inv_proj:projection.inverse().to_cols_array_2d(),inv_view:view.inverse().to_cols_array_2d(),
                    camera:eye.extend(0.0).to_array(),key:[0.0;10],enabled:true,fog:Fog::default()});
        })?;
        let (_,meta,_,_,_)=denoiser.debug_history(0);
        one_time_submit(device,r.command_pool,r.ctx.queue,|cb|unsafe {
            transition(device,cb,meta,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
            device.cmd_copy_image_to_buffer(cb,meta,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,readback.handle,&[vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                .image_offset(vk::Offset3D {x:center[0],y:center[1],z:0}).image_extent(vk::Extent3D {width:1,height:1,depth:1})]);
            transition(device,cb,meta,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::TRANSFER,vk::PipelineStageFlags::HOST,vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ)],&[],&[]);
        })?;
        let bytes=readback.read();let packed=f32::from_ne_bytes(bytes[12..16].try_into().unwrap()).round() as u32;
        let count=packed>>12;
        eprintln!("[Reflection history hypotheses] kept={kept} surface={surface:?} hit={hit:?} accumulated={count}");
        counts.push(count);
    }
    denoiser.destroy(device,&mut alloc);for image in &mut inputs {image.destroy(device,&mut alloc);}
    output.destroy(device,&mut alloc);upload.destroy(device,&mut alloc);readback.destroy(device,&mut alloc);
    drop(alloc);
    assert!(counts[..2].iter().all(|count|*count>=5),
        "valid reflection history was discarded when the other hypothesis was disoccluded: {counts:?}");
    assert_eq!(counts[2],1,"disoccluded reflection retained unrelated history");
    Ok(())
}
fn clear_raw(device:&ash::Device,cb:vk::CommandBuffer,image:vk::Image,view:vk::ImageView,size:vk::Extent2D) {
    transition(device,cb,image,vk::ImageLayout::GENERAL,vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let attachment=[vk::RenderingAttachmentInfo::default().image_view(view).image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .load_op(vk::AttachmentLoadOp::CLEAR).store_op(vk::AttachmentStoreOp::STORE)
        .clear_value(vk::ClearValue {color:vk::ClearColorValue {float32:[1.0,1.0,1.0,4.0]}})];
    unsafe {device.cmd_begin_rendering(cb,&vk::RenderingInfo::default().render_area(vk::Rect2D {offset:vk::Offset2D::default(),extent:size})
        .layer_count(1).color_attachments(&attachment));device.cmd_end_rendering(cb);}
    transition(device,cb,image,vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,vk::ImageLayout::GENERAL);
}
