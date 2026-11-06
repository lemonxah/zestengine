//! Automatic directional cache: actual ray demand, bounded work, world reuse,
//! physical HDR energy, independent views and lighting invalidation.
use glam::{Mat4,Quat,Vec3};
use ash::vk;
use crate::{CameraData,FluxSettings,GiSource,GpuGiMaterial,Renderer};
use super::planar::{cube,draw,lit};
fn read(r:&Renderer,source:vk::Buffer,offset:u64,size:usize)->anyhow::Result<Vec<u8>> {
    let device=&r.ctx.device;let allocator=r.allocator();let mut alloc=allocator.lock().unwrap();
    let mut host=crate::alloc::Buffer::new(device,&mut alloc,size as u64,vk::BufferUsageFlags::TRANSFER_DST,gpu_allocator::MemoryLocation::GpuToCpu,"radiance cache readback")?;
    crate::alloc::one_time_submit(device,r.command_pool,r.ctx.queue,|cb|unsafe {
        crate::flux_radiance_cache::barrier(device,cb);
        device.cmd_copy_buffer(cb,source,host.handle,&[vk::BufferCopy {src_offset:offset,dst_offset:0,size:size as u64}]);
        crate::flux_radiance_cache::barrier(device,cb);
    })?;let bytes=host.read();host.destroy(device,&mut alloc);Ok(bytes)
}
fn stats(r:&Renderer,view:usize)->anyhow::Result<Vec<u32>> {
    let (_,work,_)=r.gpu_gi.as_ref().unwrap().radiance_cache.debug(view);
    Ok(read(r,work,0,48*4)?.chunks_exact(4).map(|b|u32::from_ne_bytes(b.try_into().unwrap())).collect())
}
fn probe(r:&Renderer,view:usize)->anyhow::Result<(Vec<u32>,f32)> {
    let (data,_,_)=r.gpu_gi.as_ref().unwrap().radiance_cache.debug(view);
    let bytes=read(r,data,3855*crate::flux_radiance_cache::PROBE_BYTES,crate::flux_radiance_cache::PROBE_BYTES as usize)?;
    let meta=bytes[..32].chunks_exact(4).map(|b|u32::from_ne_bytes(b.try_into().unwrap())).collect();
    let energy=bytes[32..].chunks_exact(8).map(|b|half::f16::from_bits(u16::from_ne_bytes([b[0],b[1]])).to_f32()).sum::<f32>()/256.0;
    Ok((meta,energy))
}
pub(super) fn hdr(r:&Renderer,view:usize,cam:CameraData)->anyhow::Result<f32> {
    let t=if view==0 {&r.viewport_target.as_ref().unwrap().sgi} else {&r.camera_preview.as_ref().unwrap().sgi};
    let size=if view==0 {r.viewport_target.as_ref().unwrap().extent} else {r.camera_preview.as_ref().unwrap().extent};
    let q=(cam.proj*cam.view).project_point3(Vec3::ZERO);
    let offset=vk::Offset3D {x:((q.x*0.5+0.5)*size.width as f32) as i32-2,y:((0.5-q.y*0.5)*size.height as f32) as i32-2,z:0};
    let device=&r.ctx.device;let allocator=r.allocator();let mut alloc=allocator.lock().unwrap();
    let mut host=crate::alloc::Buffer::new(device,&mut alloc,25*8,vk::BufferUsageFlags::TRANSFER_DST,gpu_allocator::MemoryLocation::GpuToCpu,"radiance cache HDR readback")?;
    crate::alloc::one_time_submit(device,r.command_pool,r.ctx.queue,|cb|unsafe {
        crate::image_barrier(device,cb,t.gi_final,vk::ImageAspectFlags::COLOR,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::PipelineStageFlags2::ALL_COMMANDS,vk::AccessFlags2::MEMORY_READ|vk::AccessFlags2::MEMORY_WRITE,vk::PipelineStageFlags2::TRANSFER,vk::AccessFlags2::TRANSFER_READ);
        device.cmd_copy_image_to_buffer(cb,t.gi_final,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,host.handle,&[vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1)).image_offset(offset)
            .image_extent(vk::Extent3D {width:5,height:5,depth:1})]);
        crate::image_barrier(device,cb,t.gi_final,vk::ImageAspectFlags::COLOR,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::TRANSFER,vk::AccessFlags2::TRANSFER_READ,vk::PipelineStageFlags2::ALL_COMMANDS,vk::AccessFlags2::MEMORY_READ);
        crate::flux_radiance_cache::barrier(device,cb);
    })?;
    let bytes=host.read();let mean=bytes[..200].chunks_exact(8).map(|b|half::f16::from_bits(u16::from_ne_bytes([b[0],b[1]])).to_f32()).sum::<f32>()/25.0;
    host.destroy(device,&mut alloc);Ok(mean)
}
pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    let mesh=r.upload_mesh(&cube())?;let floor=lit(r,[0.8;4],[0.0;3],1.0,0.0);
    let source=lit(r,[0.0,0.0,0.0,1.0],[4.0,0.4,0.1],1.0,0.0);
    r.set_material_features(source,crate::MaterialFeatures {emission:true,..Default::default()});
    let mut draws=[draw(mesh,floor,Mat4::from_scale_rotation_translation(Vec3::new(12.0,0.2,12.0),Quat::IDENTITY,Vec3::new(0.0,-0.1,0.0)),-1),
        draw(mesh,source,Mat4::from_scale_rotation_translation(Vec3::new(24.0,16.0,0.2),Quat::IDENTITY,Vec3::new(0.0,8.0,-12.0)),-1)];
    for (i,d) in draws.iter_mut().enumerate() {d.gi_source=Some(GiSource {object:0xbc00+i as u128,slot:0});}
    let volume=super::flux::cube_distance_field();
    let fields:Vec<_>=draws.iter().enumerate().map(|(i,d)|crate::GiMeshSdf {gi_source:d.gi_source,world_to_local:d.transform.inverse(),volume:volume.clone(),in_global_field:false,
        material:GpuGiMaterial {albedo:if i==0 {[0.8;3]} else {[0.0;3]},emission:if i==0 {[0.0;3]} else {[4.0,0.4,0.1]},metallic:0.0,roughness:1.0}}).collect();
    r.gi_set_mesh_sdfs(&fields)?;r.gi_set_gdf(&vec![100.0;512],&vec![0;512],[8;3],[-30.0;3],[30.0;3],&[fields[0].material]);
    r.gi_set_emitters(&[]);r.set_gi_enabled(true);
    let eye=Vec3::new(0.0,2.0,4.0);let cam=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let mut input=crate::cascade_tests::input(cam);input.camera_preview=Some(cam);input.viewport_extent=Some([323,181]);input.draws=&draws;
    input.light.ambient=[0.0;3];input.light.intensity=0.0;input.light.env_intensity=0.0;input.draw_skybox=false;
    input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
    for hardware in [false,true] {
        if hardware && !r.ctx.ray_tracing() {continue;}
        r.set_flux_settings(FluxSettings {rt_trace:hardware,samples:64,bounces:4,firefly_clamp:16.0,reflection_mode:0,ssr_enabled:false,..Default::default()});
        r.material_params_mut(source).emission_intensity=1.0;
        let mut energy=Vec::new();
        for mode in 0..3 {
            let enabled=mode>0;r.gpu_gi.as_mut().unwrap().radiance_cache.full_interpolation_override=Some(mode==1);
            r.gpu_gi.as_mut().unwrap().radiance_cache.enabled_override=Some(enabled);r.invalidate_screen_gi();
            for _ in 0..128 {r.render(&input)?;}
            let mut sum=[0.0;2];for _ in 0..16 {r.render(&input)?;for view in 0..2 {sum[view]+=hdr(r,view,cam)?/16.0;}}
            energy.push(sum);
        }
        eprintln!("[Flux radiance HDR] hardware={hardware} uncached={:?} full={:?} sampled={:?}",energy[0],energy[1],energy[2]);
        for view in 0..2 {
            let state=stats(r,view)?;let (meta,value)=probe(r,view)?;
            eprintln!("[Flux radiance demand] hardware={hardware} view={view} work={:?} meta={meta:?} energy={value}",&state[..9]);
            assert!(state[0]<=32 && state[3]>0 && state[5]>0,"radiance cache was not used or exceeded trace budget");
            for mode in 1..3 {assert!(energy[0][view]>0.01 && (energy[mode][view]-energy[0][view]).abs()<energy[0][view]*0.2,"radiance cache changed HDR energy: {energy:?}");}
            assert!((energy[2][view]-energy[1][view]).abs()<energy[1][view]*0.08,"sampled interpolation changed full-reference energy: {energy:?}");
        }
        if hardware {
            let tex=r.upload_texture(&crate::TextureData {width:1,height:1,pixels:vec![128,255,255,255],srgb:false,hdr:false})?;
            let mut slots=[None;16];slots[3]=Some(tex);r.set_material_textures(source,&slots);
            input.preview_postfx.flux.ray_lighting_mode=Some(crate::RayLightingMode::HitLighting);
            for _ in 0..128 {r.render(&input)?;}
            let mut result=[0.0;2];for _ in 0..16 {r.render(&input)?;for view in 0..2 {result[view]+=hdr(r,view,cam)?/16.0;}}
            eprintln!("[Flux GI hit textured emission] cached Scene={}, hit Camera={}",result[0],result[1]);
            assert!(result[1]>energy[0][1]*0.35 && result[1]<energy[0][1]*0.65,"GI hit material ignored or doubled emission texture: {result:?}");
            assert!((result[0]-result[1]).abs()<result[1]*0.2,"cached/hit material energy differs: {result:?}");
            input.preview_postfx.flux.ray_lighting_mode=None;r.set_material_textures(source,&[None;16]);
        }
        // A low limiter must not cause a lighting ramp as distant tracing is
        // replaced by cached interpolation. This does not clamp cache storage.
        let physical_settings=r.flux_settings;
        r.set_flux_settings(FluxSettings {firefly_clamp:2.0,..physical_settings});
        let mut limited=Vec::new();
        for enabled in [false,true] {
            r.gpu_gi.as_mut().unwrap().radiance_cache.enabled_override=Some(enabled);r.invalidate_screen_gi();
            for _ in 0..128 {r.render(&input)?;}
            let mut sum=[0.0;2];for _ in 0..16 {r.render(&input)?;for view in 0..2 {sum[view]+=hdr(r,view,cam)?/16.0;}}
            limited.push(sum);
        }
        eprintln!("[Flux radiance limited HDR] hardware={hardware}: {limited:?}");
        for view in 0..2 {assert!(limited[0][view]>0.01 && (limited[1][view]-limited[0][view]).abs()<limited[0][view]*0.2,
            "firefly limiter changed brightness as cache warmed up: {limited:?}");}
        r.set_flux_settings(physical_settings);
        // A distant, zero-thickness opaque pane must block transport from either
        // winding, including rays that reach the automatic far-field cache.
        let black=lit(r,[0.0,0.0,0.0,1.0],[0.0;3],1.0,0.0);
        r.set_material_features(black,crate::MaterialFeatures {double_sided:false,..Default::default()});
        for facing in [1.0f32,-1.0] {
            let positions=[Vec3::new(-20.0,-20.0,-5.0),Vec3::new(20.0,-20.0,-5.0),Vec3::new(20.0,20.0,-5.0),Vec3::new(-20.0,20.0,-5.0)];
            let indices=if facing>0.0 {vec![0,1,2,0,2,3]} else {vec![0,2,1,0,3,2]};
            let data=crate::MeshData {vertices:positions.iter().map(|p|crate::Vertex {position:p.to_array(),normal:[0.0,0.0,facing],color:[1.0;4],..Default::default()}).collect(),indices:indices.clone(),has_lightmap_uv:false};
            let pane_mesh=r.upload_mesh(&data)?;let mut pane=draw(pane_mesh,black,Mat4::IDENTITY,-1);
            pane.gi_source=Some(GiSource {object:0xbc02,slot:0});
            let covered=[draws[0],draws[1],pane];let mut occluders=fields.clone();
            occluders.push(crate::GiMeshSdf {gi_source:pane.gi_source,world_to_local:Mat4::IDENTITY,
                volume:std::sync::Arc::new(crate::sdf::generate_sdf(&positions,&indices,32,0.1)),in_global_field:false,
                material:GpuGiMaterial {albedo:[0.0;3],emission:[0.0;3],metallic:0.0,roughness:1.0}});
            r.gi_set_mesh_sdfs(&occluders)?;
            let mut blocked_input=crate::cascade_tests::input(cam);blocked_input.camera_preview=Some(cam);
            blocked_input.viewport_extent=input.viewport_extent;blocked_input.draws=&covered;blocked_input.light=input.light;
            blocked_input.postfx=input.postfx;blocked_input.preview_postfx=input.preview_postfx;
            for _ in 0..160 {r.render(&blocked_input)?;}
            let blocked=[hdr(r,0,cam)?,hdr(r,1,cam)?];
            eprintln!("[Flux radiance opaque pane] hardware={hardware} facing={facing}: {blocked:?}");
            assert!(blocked.iter().all(|v|*v<0.005),"automatic cache leaked through opaque pane: {blocked:?}");
            if hardware {
                blocked_input.preview_postfx.flux.ray_lighting_mode=Some(crate::RayLightingMode::HitLighting);
                for _ in 0..32 {r.render(&blocked_input)?;}
                let hit=hdr(r,1,cam)?;assert!(hit<0.005,"hit lighting leaked through opaque pane, facing={facing}: {hit}");
            }
        }
        r.gi_set_mesh_sdfs(&fields)?;
        for _ in 0..128 {r.render(&input)?;}
        let before=probe(r,0)?;assert!(before.1>0.01,"physical probe was never lit");let epoch=r.gpu_gi.as_ref().unwrap().radiance_cache.debug(0).2;
        let moved_eye=eye+Vec3::X*1.2;input.camera=CameraData {position:moved_eye,view:Mat4::look_at_rh(moved_eye,Vec3::ZERO,Vec3::Y),..cam};
        for _ in 0..8 {r.render(&input)?;}
        let after=probe(r,0)?;assert_eq!(&before.0[..4],&after.0[..4],"world cell moved with camera");
        assert_eq!(r.gpu_gi.as_ref().unwrap().radiance_cache.debug(0).2,epoch,"camera movement invalidated overlapping world cache");
        input.camera=cam;input.postfx.exposure=3.0;input.preview_postfx.exposure=-3.0;
        let settings=r.flux_settings;r.set_flux_settings(FluxSettings {intensity:3.0,..settings});
        for _ in 0..16 {r.render(&input)?;}
        let display=probe(r,0)?;assert_eq!(r.gpu_gi.as_ref().unwrap().radiance_cache.debug(0).2,epoch);
        assert!((display.1-before.1).abs()<before.1.max(0.01)*0.05,"display controls fed back into physical cache");
        input.postfx.exposure=0.0;input.preview_postfx.exposure=0.0;r.set_flux_settings(settings);
        r.material_params_mut(source).emission_intensity=0.0;
        for _ in 0..160 {r.render(&input)?;}
        assert_ne!(r.gpu_gi.as_ref().unwrap().radiance_cache.debug(0).2,epoch,"material edit left old cache epoch valid");
        let dark=[hdr(r,0,cam)?,hdr(r,1,cam)?];eprintln!("[Flux radiance light off] hardware={hardware}: {dark:?}");
        assert!(dark.iter().all(|v|*v<0.005),"cache retained switched-off source");
        let camera_epoch=r.gpu_gi.as_ref().unwrap().radiance_cache.debug(1).2;input.camera_preview=None;
        for _ in 0..3 {r.render(&input)?;}
        assert_eq!(r.gpu_gi.as_ref().unwrap().radiance_cache.debug(1).2,camera_epoch);
        input.camera_preview=Some(cam);r.render(&input)?;
        assert_ne!(r.gpu_gi.as_ref().unwrap().radiance_cache.debug(1).2,camera_epoch,"hidden Camera reused stale cache");
    }
    r.gpu_gi.as_mut().unwrap().radiance_cache.enabled_override=None;
    r.gpu_gi.as_mut().unwrap().radiance_cache.full_interpolation_override=None;Ok(())
}
