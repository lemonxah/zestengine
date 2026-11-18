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
fn history_words(r:&Renderer,id:usize,p:Vec3)->anyhow::Result<[u32;4]> {
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
    let bytes=output.read();let words=*bytemuck::from_bytes(&bytes[..16]);
    output.destroy(device,&mut alloc);Ok(words)
}
fn source_parts(words:[u32;4])->([f32;3],[f32;3],f32,bool) {
    let half=|v:u32|half::f16::from_bits(v as u16).to_f32();
    ([half(words[0]),half(words[0]>>16),half(words[1])],
     [half(words[1]>>16),half(words[2]),half(words[2]>>16)],
     f32::from_bits(words[3]&0x7fffffff),words[3]&0x80000000==0)
}
fn history_pixel(r:&Renderer,id:usize,p:Vec3)->anyhow::Result<[f32;4]> {
    let (d,e,z,_)=source_parts(history_words(r,id,p)?);Ok([d[0]+e[0],d[1]+e[1],d[2]+e[2],z])
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
        r.set_flux_settings(crate::FluxSettings {rt_trace:rt,gi_screen_traces:false,samples:64,bounces:2,reflection_mode:0,ssr_enabled:false,..Default::default()});r.set_gi_enabled(true);
        for _ in 0..24 {r.render(&input)?;}
        let vp=r.debug_read_viewport()?.unwrap();
        let pv=r.debug_read_camera_preview()?.unwrap();
        let dark=[patch(&vp,camera,floor),patch(&pv,camera,floor)];
        let emitter_px=[patch(&vp,camera,emitter),patch(&pv,camera,emitter)];
        let stats=|image:&(u32,u32,Vec<u8>)| {
            let (w,h,data)=image;
            let mut max=0u8; let mut above=0u32; let mut zero=0u32;
            for px in data.chunks(4) { max=max.max(px[0]); if px[0]>10 { above+=1; }
            if px[0]==0 { zero+=1; } }
            (max,above,zero,*w,*h)
        };
        let sample=|image:&(u32,u32,Vec<u8>),x:u32,y:u32| {
            let i=((y*image.0+x)*4) as usize; image.2[i..i+4].to_vec()
        };
        eprintln!("[Flux screen history] traces={} serials={:?}/{:?} floor={dark:?} emitter={emitter_px:?} vp_stats={:?} pv_stats={:?} vp_px={:?}/{:?}/{:?}",
            r.flux_settings.gi_screen_traces,
            r.flux_screen_history.as_ref().unwrap().debug_capture_serial(0),
            r.flux_screen_history.as_ref().unwrap().debug_capture_serial(1),
            stats(&vp),stats(&pv),
            sample(&vp,10,10),sample(&vp,160,90),sample(&vp,300,170));
        assert!(dark.iter().all(|v|*v<2.0),"screen-only fixture was lit by world fallback: {dark:?}");
        assert_eq!(r.flux_screen_history.as_ref().unwrap().debug_capture_serial(0),None);
        assert_eq!(r.flux_screen_history.as_ref().unwrap().debug_capture_serial(1),None);
        r.set_flux_settings(crate::FluxSettings {gi_screen_traces:true,..r.flux_settings});
        for _ in 0..48 {r.render(&input)?;}
        let lit=[patch(&r.debug_read_viewport()?.unwrap(),camera,floor),patch(&r.debug_read_camera_preview()?.unwrap(),camera,floor)];
        eprintln!("[Flux screen history] RT={rt}: world only={dark:?}, screen+world={lit:?}");
        assert!(lit.iter().all(|v|*v>20.0),"visible custom emission did not enter screen GI: {lit:?}");
        for id in 0..2 {
            let value=history_pixel(r,id,emitter)?;
            assert!((value[0]-4.0).abs()<0.001 && (value[1]-1.0).abs()<0.001 && value[3]<1.0,"invalid raw HDR source: {value:?}");
        }
        // Real post-process controls, independently blended per view. A
        // disabled view must not retain screen-only lighting or capture work.
        let epochs: [_;2]=std::array::from_fn(|v|r.gpu_gi.as_ref().unwrap().radiance_cache.debug(v).2);
        for scene_enabled in [false,true] {
            input.postfx.flux.gi_screen_traces=Some(scene_enabled);
            input.preview_postfx.flux.gi_screen_traces=Some(!scene_enabled);
            for frame in 0..48 {
                r.render(&input)?;
                for (id,enabled) in [scene_enabled,!scene_enabled].into_iter().enumerate() {
                    assert_eq!(r.flux_screen_history.as_ref().unwrap().debug_capture_serial(id),enabled.then_some(r.frame_serial),"disabled view captured screen-source history");
                    assert_eq!(r.gpu_gi.as_ref().unwrap().radiance_cache.debug(id).2,epochs[id],"screen toggle invalidated world radiance");
                }
                if frame==0 {
                    let off=if scene_enabled {r.debug_read_camera_preview()?.unwrap()} else {r.debug_read_viewport()?.unwrap()};
                    assert!(patch(&off,camera,floor)<2.0,"screen-only lighting lingered after disabling");
                }
            }
            let levels=[patch(&r.debug_read_viewport()?.unwrap(),camera,floor),patch(&r.debug_read_camera_preview()?.unwrap(),camera,floor)];
            for (value,enabled) in levels.into_iter().zip([scene_enabled,!scene_enabled]) {
                assert!(if enabled {value>20.0} else {value<2.0},"per-view screen override ignored: {levels:?}");
            }
            eprintln!("[Flux GI screen control] hardware={rt}, Scene enabled={scene_enabled}, Scene/Camera={levels:?}");
        }
        // A per-view true also overrides an environment-wide false.
        r.set_flux_settings(crate::FluxSettings {gi_screen_traces:false,..r.flux_settings});
        input.postfx.flux.gi_screen_traces=Some(true);input.preview_postfx.flux.gi_screen_traces=Some(true);
        for _ in 0..48 {r.render(&input)?;}
        for image in [r.debug_read_viewport()?.unwrap(),r.debug_read_camera_preview()?.unwrap()] {
            assert!(patch(&image,camera,floor)>20.0,"explicit true did not override environment false");
        }
        input.postfx.flux.gi_screen_traces=None;input.preview_postfx.flux.gi_screen_traces=None;
        r.set_flux_settings(crate::FluxSettings {gi_screen_traces:true,..r.flux_settings});
        for _ in 0..48 {r.render(&input)?;}
        // Display exposure/bloom do not change captured radiance.
        input.postfx.exposure=4.0;input.preview_postfx.exposure = -4.0;
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
        // Built-in emission must continue to light the receiver through world
        // tracing when both views disable screen traces.
        r.set_material_shader(source,None);
        r.material_params_mut(source).emission_color=[4.0,1.0,0.25];r.material_params_mut(source).emission_intensity=1.0;
        r.set_material_features(source,crate::MaterialFeatures {emission:true,..Default::default()});
        let mut world_fields=fields.clone();world_fields[1].material.emission=[4.0,1.0,0.25];
        r.gi_set_mesh_sdfs(&world_fields)?;
        input.postfx.flux.gi_screen_traces=Some(false);input.preview_postfx.flux.gi_screen_traces=Some(false);
        for _ in 0..64 {r.render(&input)?;}
        let world=[patch(&r.debug_read_viewport()?.unwrap(),camera,floor),patch(&r.debug_read_camera_preview()?.unwrap(),camera,floor)];
        assert!(world.iter().all(|v|*v>20.0),"screen toggle disabled world emissive transport: {world:?}");
        eprintln!("[Flux GI screen control] hardware={rt}, screen disabled world emission Scene/Camera={world:?}");
        r.material_params_mut(source).emission_color=[0.0;3];r.set_material_features(source,crate::MaterialFeatures::default());
        r.set_material_shader(source,Some(shader));r.gi_set_mesh_sdfs(&fields)?;
        input.postfx.flux.gi_screen_traces=None;input.preview_postfx.flux.gi_screen_traces=None;
    }
    r.gpu_gi.as_mut().unwrap().screen_trace_override=None;
    physical_sources(r)?;
    Ok(())
}

// Standard surfaces carry independent physical diffuse/emission. Presentation
// gain (including zero), specular environment and exposure cannot enter this
// source; non-unit emission boost must still use the screen-first path.
fn physical_sources(r:&mut Renderer)->anyhow::Result<()> {
    let mesh=r.upload_mesh(&cube())?;
    let receiver=lit(r,[0.8,0.8,0.8,1.0],[0.0;3],1.0,0.0);
    let source=lit(r,[0.0,0.0,0.0,1.0],[4.0,1.0,0.25],0.05,0.0);
    let texture=r.upload_texture(&crate::TextureData {width:1,height:1,pixels:vec![128,255,64,255],srgb:false,hdr:false})?;
    let mut slots=[None;16];slots[3]=Some(texture);r.set_material_textures(source,&slots);
    let mut draws=[draw(mesh,receiver,Mat4::from_scale_rotation_translation(Vec3::new(6.0,0.2,6.0),Quat::IDENTITY,Vec3::new(0.0,-0.1,0.0)),-1),
        draw(mesh,source,Mat4::from_scale_rotation_translation(Vec3::new(1.8,1.8,0.2),Quat::IDENTITY,Vec3::new(0.0,1.2,-1.3)),-1)];
    for (i,d) in draws.iter_mut().enumerate() {d.gi_source=Some(GiSource {object:0x9100+i as u128,slot:0});}
    let fields:Vec<_>=draws.iter().enumerate().map(|(i,d)|crate::GiMeshSdf {gi_source:d.gi_source,world_to_local:d.transform.inverse(),volume:super::flux::cube_distance_field(),in_global_field:false,
        material:GpuGiMaterial {albedo:if i==0 {[0.8;3]} else {[0.0;3]},emission:if i==0 {[0.0;3]} else {[4.0,1.0,0.25]},metallic:0.0,roughness:1.0}}).collect();
    r.gi_set_mesh_sdfs(&fields)?;r.gi_set_emitters(&[]);
    r.gi_set_gdf(&vec![100.0;512],&vec![0;512],[8;3],[-10.0;3],[10.0;3],&[fields[0].material]);
    let eye=Vec3::new(0.0,2.3,5.0);let cam=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let mut input=crate::cascade_tests::input(cam);input.camera_preview=Some(cam);input.viewport_extent=Some([323,181]);input.draws=&draws;
    input.light.ambient=[0.0;3];input.light.intensity=0.0;input.light.env_intensity=0.0;input.draw_skybox=false;
    input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
    let emitter=Vec3::new(0.0,1.2,-1.19);let floor=Vec3::new(0.0,0.0,0.4);
    for rt in [false,true] {
        if rt && !r.ctx.ray_tracing() {continue;}
        r.set_flux_settings(crate::FluxSettings {rt_trace:rt,samples:64,bounces:4,firefly_clamp:16.0,reflection_mode:0,ssr_enabled:false,..Default::default()});r.set_gi_enabled(true);
        r.gpu_gi.as_mut().unwrap().verify_queued_gather=true;
        let settings=r.flux_settings;let mut energies=Vec::new();
        for gain in [1.0,0.0,3.0] {
            r.set_flux_settings(crate::FluxSettings {intensity:gain,..settings});
            for _ in 0..96 {r.render(&input)?;}
            let mut energy=[0.0;2];
            for (id, id_value) in energy.iter_mut().enumerate() {
                let (diffuse,emission,depth,physical)=source_parts(history_words(r,id,emitter)?);
                assert!(physical && depth<1.0 && diffuse.iter().all(|v|v.abs()<0.001),"source was not physical emission: {diffuse:?}/{emission:?}");
                for (v,want) in emission.into_iter().zip([4.0*128.0/255.0,1.0,0.25*64.0/255.0]) {assert!((v-want).abs()<0.002);}
                let (diffuse,_,_,physical)=source_parts(history_words(r,id,floor)?);assert!(physical);*id_value=diffuse[0];
                assert!(*id_value>0.01,"display gain zero removed physical transport");
            }
            if gain==0.0 {assert!(patch(&r.debug_read_viewport()?.unwrap(),cam,floor)<2.0,"zero display GI gain did not hide GI");}
            energies.push(energy);
        }
        eprintln!("[Flux physical source gain] RT={rt}: {energies:?}");
        for energy in energies.iter().skip(1) {for (value, baseline) in energy.iter().zip(&energies[0]) {assert!((*value-*baseline).abs()<*baseline*0.2,"display gain fed back into physical diffuse");}}
        r.set_flux_settings(settings);
        input.postfx.exposure=4.0;input.preview_postfx.exposure = -4.0;input.light.env_intensity=4.0;
        input.light.intensity=60.0;input.light.direction=Vec3::new(0.0,1.1,-6.2).normalize();
        for _ in 0..32 {r.render(&input)?;}
        for id in 0..2 {let (d,e,_,physical)=source_parts(history_words(r,id,emitter)?);assert!(physical && d[0]<0.001 && (e[0]-4.0*128.0/255.0).abs()<0.002,"specular/exposure entered source");}
        input.postfx.exposure=0.0;input.preview_postfx.exposure=0.0;input.light.env_intensity=0.0;input.light.intensity=0.0;
        r.set_flux_settings(crate::FluxSettings {emissive_boost:2.0,..settings});
        for _ in 0..128 {r.render(&input)?;}
        for (id, id_value) in energies[0].iter().enumerate().take(2) {
            let (d,_,_,_)=source_parts(history_words(r,id,floor)?);
            assert!(d[0]>*id_value*1.6 && d[0]<*id_value*2.4,"emission boost was lost/doubled in screen history: {d:?}");
            let (_,work,_)=r.gpu_gi.as_ref().unwrap().radiance_cache.debug(id);
            let bytes=super::flux_radiance::read(r,work,12*4,4)?;let hits=u32::from_ne_bytes(bytes[..4].try_into().unwrap());
            eprintln!("[Flux physical screen first] RT={rt} view={id} hits={hits} boosted={d:?}");
            assert!(hits>0,"standard physical source never used screen-first tracing");
            {
                let bytes=super::flux_radiance::read(r,work,13*4,8)?;
                let mismatch=u32::from_ne_bytes(bytes[..4].try_into().unwrap());
                let checked=u32::from_ne_bytes(bytes[4..8].try_into().unwrap());
                eprintln!("[Flux queued gather reference] RT={rt} view={id} checked={checked} mismatch={mismatch}");
                assert!(checked>100 && mismatch==0,"queued tracing changed the same-frame inline reference");
            }
        }
        r.material_params_mut(source).emission_intensity=0.0;
        for _ in 0..160 {r.render(&input)?;}
        for id in 0..2 {
            let (d,_,_,_)=source_parts(history_words(r,id,floor)?);
            assert!(d[0]<0.002,"screen feedback retained a switched-off emitter: {d:?}");
        }
        r.material_params_mut(source).emission_intensity=1.0;
        r.set_flux_settings(settings);
        {
            // Exercise minimum/rounded/high ray budgets and scratch growth.
            // The per-ray reference also checks the nonuniform cell allocation.
            for samples in [1,17,256] {
                r.set_flux_settings(crate::FluxSettings {samples,..settings});
                for _ in 0..8 {r.render(&input)?;}
                for id in 0..2 {
                    let (_,work,_)=r.gpu_gi.as_ref().unwrap().radiance_cache.debug(id);
                    let bytes=super::flux_radiance::read(r,work,13*4,8)?;
                    let mismatch=u32::from_ne_bytes(bytes[..4].try_into().unwrap());
                    let checked=u32::from_ne_bytes(bytes[4..8].try_into().unwrap());
                    eprintln!("[Flux queued sample budget] RT={rt} samples={samples} view={id} checked={checked} mismatch={mismatch}");
                    assert!(checked>100 && mismatch==0,"queue sample budget changed ray results");
                }
            }
            r.set_flux_settings(settings);
        }
        if !rt {
            // Streaming may temporarily leave only the global field. Keep the
            // same receiver/screen/world schedule instead of switching samplers.
            r.gi_set_mesh_sdfs(&[])?;
            for _ in 0..16 {r.render(&input)?;}
            for id in 0..2 {
                let (_,work,_)=r.gpu_gi.as_ref().unwrap().radiance_cache.debug(id);
                let bytes=super::flux_radiance::read(r,work,13*4,8)?;
                let mismatch=u32::from_ne_bytes(bytes[..4].try_into().unwrap());
                let checked=u32::from_ne_bytes(bytes[4..8].try_into().unwrap());
                eprintln!("[Flux global-field queued reference] view={id} checked={checked} mismatch={mismatch}");
                assert!(checked>100 && mismatch==0,"global-field fallback lost queued gather");
            }
            r.gi_set_mesh_sdfs(&fields)?;
        }
    }
    // The standalone game pass owns a third history and a different attachment
    // set. Exercise it after both editor views have already populated theirs.
    input.viewport_extent=None;input.camera_preview=None;
    for _ in 0..32 {r.render(&input)?;}
    let (d,e,z,physical)=source_parts(history_words(r,2,emitter)?);
    assert!(physical && z<1.0 && d[0]<0.001 && (e[0]-4.0*128.0/255.0).abs()<0.002,"game physical source was not captured: {d:?}/{e:?}");
    r.gpu_gi.as_mut().unwrap().verify_queued_gather=false;
    Ok(())
}
