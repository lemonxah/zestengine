//! A visible custom emitter is black to both world tracers and has no surface
//! cards. Only validated screen history can carry its colour onto the floor.
use ash::vk;
use glam::{Mat4,Quat,Vec3};
use gpu_allocator::MemoryLocation;
use crate::{alloc::{Buffer,one_time_submit},CameraData,GiSource,GpuGiMaterial,Renderer};
use super::planar::{cube,draw,lit};
fn patch(image:&(u32,u32,Vec<u8>),camera:CameraData,p:Vec3)->f32 {
    let (w,h,data)=image;let p=(camera.proj*camera.view).project_point3(p);
    let x=((p.x*0.5+0.5)* *w as f32) as i32;let y=((0.5-p.y*0.5)* *h as f32) as i32;
    let mut sum=0.0;for j in -2..=2 {for i in -2..=2 {sum+=data[(((y+j) as u32*w+(x+i) as u32)*4) as usize] as f32;}}sum/25.0
}
fn history_pixel(r:&Renderer,id:usize,p:Vec3)->anyhow::Result<[f32;4]> {
    let (image,size,vp,_,_)=r.flux_screen_history.as_ref().unwrap().debug_history(id);
    let q=vp.project_point3(p);let x=((q.x*0.5+0.5)*size.width as f32) as i32;let y=((0.5-q.y*0.5)*size.height as f32) as i32;
    let device=&r.ctx.device;unsafe {device.device_wait_idle()?;}
    let allocator=r.allocator();let mut alloc=allocator.lock().unwrap();
    let mut output=Buffer::new(device,&mut alloc,16,vk::BufferUsageFlags::TRANSFER_DST,MemoryLocation::GpuToCpu,"screen history texel")?;
    one_time_submit(device,r.command_pool,r.ctx.queue,|cb|unsafe {
        let range=vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
        let barrier=|from,to,src,dst|vk::ImageMemoryBarrier::default().image(image).subresource_range(range)
            .old_layout(from).new_layout(to).src_access_mask(src).dst_access_mask(dst);
        device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::COMPUTE_SHADER,vk::PipelineStageFlags::TRANSFER,vk::DependencyFlags::empty(),&[],&[],
            &[barrier(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,vk::AccessFlags::SHADER_READ|vk::AccessFlags::SHADER_WRITE,vk::AccessFlags::TRANSFER_READ)]);
        let copy=vk::BufferImageCopy::default().image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
            .image_offset(vk::Offset3D{x,y,z:0}).image_extent(vk::Extent3D{width:1,height:1,depth:1});
        device.cmd_copy_image_to_buffer(cb,image,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,output.handle,&[copy]);
        device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::TRANSFER,vk::PipelineStageFlags::COMPUTE_SHADER|vk::PipelineStageFlags::HOST,vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ)],&[],
            &[barrier(vk::ImageLayout::TRANSFER_SRC_OPTIMAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,vk::AccessFlags::TRANSFER_READ,vk::AccessFlags::SHADER_READ)]);
    })?;
    let bytes=output.read();let result=*bytemuck::from_bytes(&bytes[..16]);output.destroy(device,&mut alloc);Ok(result)
}
pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    let shader=r.register_shader(include_bytes!(concat!(env!("OUT_DIR"),"/flux_screen_source_test.frag.spv")))?;
    let mesh=r.upload_mesh(&cube())?;let volume=super::flux::cube_distance_field();
    let receiver=lit(r,[0.8;4],[0.0;3],1.0,0.0);let source=lit(r,[0.0,0.0,0.0,1.0],[0.0;3],1.0,0.0);
    r.set_material_shader(source,Some(shader));
    let mut objects=[draw(mesh,receiver,Mat4::from_scale_rotation_translation(Vec3::new(6.0,0.2,6.0),Quat::IDENTITY,Vec3::new(0.0,-0.1,0.0)),-1),
        draw(mesh,source,Mat4::from_scale_rotation_translation(Vec3::new(1.8,1.8,0.2),Quat::IDENTITY,Vec3::new(0.0,1.2,-1.3)),-1)];
    for (i,d) in objects.iter_mut().enumerate() {d.gi_source=Some(GiSource {object:0x9000+i as u128,slot:0});}
    let fields:Vec<_>=objects.iter().map(|d|crate::GiMeshSdf {gi_source:d.gi_source,world_to_local:d.transform.inverse(),volume:volume.clone(),in_global_field:false,
        material:GpuGiMaterial {albedo:if d.material==source {[0.0;3]} else {[0.8;3]},emission:[0.0;3],metallic:0.0,roughness:1.0}}).collect();
    r.gi_set_mesh_sdfs(&fields)?;r.gi_set_emitters(&[]);
    r.gi_set_gdf(&vec![100.0;512],&vec![0;512],[8;3],[-10.0;3],[10.0;3],&[fields[0].material]);
    let eye=Vec3::new(0.0,2.3,5.0);let camera=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let mut input=crate::cascade_tests::input(camera);input.camera_preview=Some(camera);input.viewport_extent=Some([323,181]);input.draws=&objects;
    input.light.ambient=[0.0;3];input.light.intensity=0.0;input.light.env_intensity=0.0;input.draw_skybox=false;
    input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
    let floor=Vec3::new(0.0,0.0,0.4);let emitter=Vec3::new(0.0,1.2,-1.19);
    // Back-facing to both cameras: raster depth cannot see this opaque wall.
    // It must still block a screen ray between the receiver and visible source.
    let positions=[Vec3::new(-10.0,0.0,-0.4),Vec3::new(10.0,0.0,-0.4),Vec3::new(10.0,8.0,-0.4),Vec3::new(-10.0,8.0,-0.4)];
    let indices=vec![0,2,1,0,3,2];
    let pane_mesh=r.upload_mesh(&crate::MeshData {vertices:positions.iter().map(|p|crate::Vertex {position:p.to_array(),normal:[0.0,0.0,-1.0],color:[1.0;4],..Default::default()}).collect(),indices:indices.clone(),has_lightmap_uv:false})?;
    let black=lit(r,[0.0,0.0,0.0,1.0],[0.0;3],1.0,0.0);
    r.set_material_features(black,crate::MaterialFeatures {double_sided:false,..Default::default()});
    let mut pane=draw(pane_mesh,black,Mat4::IDENTITY,-1);pane.gi_source=Some(GiSource {object:0x9002,slot:0});
    let covered=[objects[0],objects[1],pane];let mut covered_fields=fields.clone();
    covered_fields.push(crate::GiMeshSdf {gi_source:pane.gi_source,world_to_local:Mat4::IDENTITY,
        volume:std::sync::Arc::new(crate::sdf::generate_sdf(&positions,&indices,32,0.1)),in_global_field:false,
        material:GpuGiMaterial {albedo:[0.0;3],emission:[0.0;3],metallic:0.0,roughness:1.0}});
    for rt in [false,true] {
        if rt && !r.ctx.ray_tracing() {continue;}
        r.set_flux_settings(crate::FluxSettings {rt_trace:rt,samples:64,bounces:2,reflection_mode:0,ssr_enabled:false,..Default::default()});r.set_gi_enabled(true);
        r.gpu_gi.as_mut().unwrap().screen_trace_override=Some(false);r.invalidate_screen_gi();
        for _ in 0..24 {r.render(&input)?;}
        let dark=[patch(&r.debug_read_viewport()?.unwrap(),camera,floor),patch(&r.debug_read_camera_preview()?.unwrap(),camera,floor)];
        assert!(dark.iter().all(|v|*v<2.0),"screen-only fixture was lit by world fallback: {dark:?}");
        r.gpu_gi.as_mut().unwrap().screen_trace_override=None;
        for _ in 0..48 {r.render(&input)?;}
        let lit=[patch(&r.debug_read_viewport()?.unwrap(),camera,floor),patch(&r.debug_read_camera_preview()?.unwrap(),camera,floor)];
        eprintln!("[Flux screen history] RT={rt}: world only={dark:?}, screen+world={lit:?}");
        assert!(lit.iter().all(|v|*v>20.0),"visible custom emission did not enter screen GI: {lit:?}");
        for id in 0..2 {
            let value=history_pixel(r,id,emitter)?;
            assert!((value[0]-4.0).abs()<0.001 && (value[1]-1.0).abs()<0.001 && value[3]<1.0,"invalid raw HDR source: {value:?}");
        }
        // Display exposure/bloom do not change captured radiance.
        input.postfx.exposure=4.0;input.preview_postfx.exposure=-4.0;
        input.postfx.bloom_enabled=true;input.preview_postfx.bloom_enabled=true;
        for _ in 0..3 {r.render(&input)?;}
        for id in 0..2 {assert!((history_pixel(r,id,emitter)?[0]-4.0).abs()<0.001);}
        input.postfx.exposure=0.0;input.preview_postfx.exposure=0.0;input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
        // Independent moving views must reproject to the same world source.
        for frame in 0..24 {
            let dx=(frame as f32*0.4).sin()*0.25;
            let moved=|x:f32| {let position=eye+Vec3::X*x;CameraData {position,view:Mat4::look_at_rh(position,Vec3::ZERO,Vec3::Y),..camera}};
            input.camera=moved(dx);input.camera_preview=Some(moved(-dx));r.render(&input)?;
            if frame%6==5 {
                let moving=[patch(&r.debug_read_viewport()?.unwrap(),input.camera,floor),patch(&r.debug_read_camera_preview()?.unwrap(),input.camera_preview.unwrap(),floor)];
                assert!(moving.iter().all(|v|*v>20.0 && *v<180.0),"screen GI lost its world position while moving: {moving:?}");
            }
        }
        input.camera=camera;input.camera_preview=Some(camera);
        let serial=r.flux_screen_history.as_ref().unwrap().debug_history(0).3;
        input.render_viewport=false;input.viewport_extent=None;
        for _ in 0..3 {r.render(&input)?;}
        assert_eq!(r.flux_screen_history.as_ref().unwrap().debug_history(0).3,serial,"Camera overwrote hidden Scene history");
        input.render_viewport=true;input.viewport_extent=Some([323,181]);r.render(&input)?;
        assert!(!r.flux_screen_history.as_ref().unwrap().debug_history(0).4,"hidden Scene reused stale history");
        r.render(&input)?;assert!(r.flux_screen_history.as_ref().unwrap().debug_history(0).4);
        input.viewport_extent=Some([321,179]);r.render(&input)?;
        assert!(!r.flux_screen_history.as_ref().unwrap().debug_history(0).4,"resize reused old history");
        r.render(&input)?;assert!(r.flux_screen_history.as_ref().unwrap().debug_history(0).4);
        input.viewport_extent=Some([323,181]);
        input.draws=&covered;r.gi_set_mesh_sdfs(&covered_fields)?;
        for _ in 0..64 {r.render(&input)?;}
        let blocked=[patch(&r.debug_read_viewport()?.unwrap(),camera,floor),patch(&r.debug_read_camera_preview()?.unwrap(),camera,floor)];
        eprintln!("[Flux screen backface] RT={rt}: unobstructed={lit:?}, blocked={blocked:?}");
        assert!(blocked.iter().all(|v|*v<2.0),"screen tracing bypassed invisible opaque back face: {blocked:?}");
        for _ in 0..96 {r.render(&input)?;}
        let blocked=[patch(&r.debug_read_viewport()?.unwrap(),camera,floor),patch(&r.debug_read_camera_preview()?.unwrap(),camera,floor)];
        eprintln!("[Flux screen backface] RT={rt}: after160 blocked={blocked:?}");
        assert!(blocked.iter().all(|v|*v<2.0),"screen tracing bypassed invisible opaque back face: {blocked:?}");
        for id in 0..2 {assert!((history_pixel(r,id,emitter)?[0]-4.0).abs()<0.001,"pane must be culled from the camera");}
        if rt {
            r.set_material_features(black,crate::MaterialFeatures {alpha_mode:crate::AlphaMode::Cutout,..Default::default()});
            r.material_params_mut(black).alpha_cutoff=0.5;
            for alpha in [0.0,1.0] {
                r.material_params_mut(black).base_color[3]=alpha;
                // Match the preceding opaque-backface steady-state check;
                // changing coverage intentionally retains legacy GI history.
                for _ in 0..if alpha==0.0 {64} else {160} {r.render(&input)?;}
                let masked=[patch(&r.debug_read_viewport()?.unwrap(),camera,floor),patch(&r.debug_read_camera_preview()?.unwrap(),camera,floor)];
                eprintln!("[Flux screen mask] alpha={alpha}: {masked:?}");
                assert!(masked.iter().all(|v|if alpha==0.0 {*v>20.0} else {*v<2.0}),"screen visibility ignored cutout coverage: {masked:?}");
            }
            r.set_material_features(black,crate::MaterialFeatures::default());
        }
        input.draws=&objects;r.gi_set_mesh_sdfs(&fields)?;
    }
    r.gpu_gi.as_mut().unwrap().screen_trace_override=None;
    Ok(())
}
