//! Automatic directional cache: actual ray demand, bounded work, world reuse,
//! physical HDR energy, independent views and lighting invalidation.
use glam::{Mat4,Quat,Vec3};
use ash::vk;
use crate::{CameraData,FluxSettings,GiSource,GpuGiMaterial,Renderer};
use super::planar::{cube,draw,lit};
pub(super) fn read(r:&Renderer,source:vk::Buffer,offset:u64,size:usize)->anyhow::Result<Vec<u8>> {
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
    let meta:Vec<u32>=bytes[..32].chunks_exact(4).map(|b|u32::from_ne_bytes(b.try_into().unwrap())).collect();
    let count=(meta[7]*meta[7]) as usize;
    let energy=bytes[32..32+count*8].chunks_exact(8).map(|b|half::f16::from_bits(u16::from_ne_bytes([b[0],b[1]])).to_f32()).sum::<f32>()/count.max(1) as f32;
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
    adaptive_schedule(r)?;
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
            assert!(state[9]<=32*256 && state[0]<=128 && state[3]>0 && state[5]>0,"radiance cache was not used or exceeded trace budget");
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
        // Move one world-space receiver through the first grid's overlap. Keep
        // history across camera steps; compare physical GI to uncached tracing.
        let poses=[5.9f32,6.1,6.9,7.1,7.9,8.1];
        let mut boundary_energy=vec![vec![[0.0f32;2];poses.len()];3];
        for mode in 0..3 {
            let cache=&mut r.gpu_gi.as_mut().unwrap().radiance_cache;
            cache.enabled_override=Some(mode!=0);cache.clipmap_blending_override=Some(mode==2);
            r.invalidate_screen_gi();
            for (pose,&z) in poses.iter().enumerate() {
                let position=Vec3::new(0.0,2.0,z);
                let moving=CameraData {position,view:Mat4::look_at_rh(position,Vec3::ZERO,Vec3::Y),..cam};
                input.camera=moving;input.camera_preview=Some(moving);
                for _ in 0..24 {r.render(&input)?;}
                for _ in 0..8 {r.render(&input)?;for view in 0..2 {boundary_energy[mode][pose][view]+=hdr(r,view,moving)?/8.0;}}
            }
        }
        eprintln!("[Flux clipmap camera sweep] hardware={hardware}: uncached={:?} hard={:?} blended={:?}",boundary_energy[0],boundary_energy[1],boundary_energy[2]);
        for pose in 0..poses.len() {for view in 0..2 {
            let reference=boundary_energy[0][pose][view];
            assert!(reference>0.1);
            assert!((boundary_energy[2][pose][view]-reference).abs()<reference*0.2,"clipmap transition changed physical light: {boundary_energy:?}");
        }}
        input.camera=cam;input.camera_preview=Some(cam);
        r.gpu_gi.as_mut().unwrap().radiance_cache.clipmap_blending_override=None;
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
            for z in [6.9f32,7.1,7.9,8.1] {
                let position=Vec3::new(0.0,2.0,z);
                let moving=CameraData {position,view:Mat4::look_at_rh(position,Vec3::ZERO,Vec3::Y),..cam};
                blocked_input.camera=moving;blocked_input.camera_preview=Some(moving);
                for _ in 0..24 {r.render(&blocked_input)?;}
                let value=[hdr(r,0,moving)?,hdr(r,1,moving)?];
                assert!(value.iter().all(|v|*v<0.005),"clipmap blend skipped the pane: hardware={hardware}, facing={facing}, z={z}: {value:?}");
            }
            blocked_input.camera=cam;blocked_input.camera_preview=Some(cam);
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
    r.gpu_gi.as_mut().unwrap().radiance_cache.full_interpolation_override=None;
    adaptive_transport(r)?;Ok(())
}

// Exercise the real GPU scheduler with more requested cells than fit. Count
// directions in its queue, including the maximum all-distant queue capacity.
fn adaptive_schedule(r:&mut Renderer)->anyhow::Result<()> {
    use crate::flux_radiance_cache::{Cache,PROBES,PROBE_BYTES,MAX_UPDATES};
    let device=&r.ctx.device;
    let mut cache=Cache::new(device)?;
    let allocator=r.allocator();
    for (case,budget) in [(0u32,32u32),(1,32),(2,32),(1,128)] {
        let mut result=Ok(());
        {
            let mut alloc=allocator.lock().unwrap();
            crate::alloc::one_time_submit(device,r.command_pool,r.ctx.queue,|cb| {
                result=cache.prepare(device,&mut alloc,cb,0,0,1,Vec3::ZERO,case as u64,budget,true).map(|_|());
            })?;
        }
        result?;
        let mut result=Ok(());
        let (data,work,_)=cache.debug(0);
        let mut candidates=std::collections::HashMap::new();
        for index in 0..PROBES {
            let level=index/4096;let slot=index%4096;
            let wrapped=[slot%16,(slot/16)%16,slot/256];
            let cell=wrapped.map(|v|if v>=8 {v as i32-16} else {v as i32});
            let spacing=[1.0,4.0,16.0][level as usize];
            let position=Vec3::from_array(cell.map(|v|(v as f32+0.5)*spacing));
            let near=position.length()<10.0;let far=position.length()>50.0;
            if (case==0 && near) || (case==1 && far) || (case==2 && (near||far)) {
                candidates.insert(index,if near {256} else {64});
            }
        }
        crate::alloc::one_time_submit(device,r.command_pool,r.ctx.queue,|cb|unsafe {
            crate::flux_radiance_cache::barrier(device,cb);
            // Clear old requests as well: each case is a fresh, bounded queue.
            for index in 0..PROBES {
                let state=[0u32,0,if candidates.contains_key(&index) {1} else {0},0];
                device.cmd_update_buffer(cb,data,index as u64*PROBE_BYTES+16,bytemuck::cast_slice(&state));
            }
            crate::flux_radiance_cache::barrier(device,cb);
        })?;
        {
            let mut alloc=allocator.lock().unwrap();
            crate::alloc::one_time_submit(device,r.command_pool,r.ctx.queue,|cb| {
                result=cache.prepare(device,&mut alloc,cb,0,0,2,Vec3::ZERO,case as u64,budget,true).map(|_|());
            })?;
        }
        result?;
        let bytes=read(r,work,0,(48+MAX_UPDATES as usize)*4)?;
        let words:Vec<_>=bytes.chunks_exact(4).map(|v|u32::from_ne_bytes(v.try_into().unwrap())).collect();
        assert!(words[0]<=MAX_UPDATES);
        let queue=&words[48..48+words[0] as usize];
        let unique:std::collections::HashSet<_>=queue.iter().copied().collect();
        assert_eq!(unique.len(),queue.len(),"duplicate radiance update");
        let rays:u32=queue.iter().map(|i|*candidates.get(i).expect("unrequested probe scheduled")).sum();
        assert!(rays<=budget*256 && rays>=budget*256-192,"ray budget overrun or idle scheduler: {rays}");
        assert_eq!(words[9],rays);assert_eq!(words[10]+words[11],words[0]);
        if case==1 {assert_eq!(words[0],budget*4,"distant coverage did not increase");}
        if case==0 {assert_eq!(words[0],budget);}
        eprintln!("[Flux adaptive schedule] case={case} budget={budget} probes={} rays={rays}",words[0]);
    }
    // The same resident cell can move between distance bands without losing
    // its old data or its world tag. Hysteresis avoids resampling at every step.
    for (resolution,camera_x,expected_rays) in [(8u32,0.0,0u32),(16,0.0,0),(8,10.0,256),(16,-10.0,64)] {
        let (data,_,_)=cache.debug(0);
        let mut result=Ok(());
        {
            let mut alloc=allocator.lock().unwrap();
            crate::alloc::one_time_submit(device,r.command_pool,r.ctx.queue,|cb|unsafe {
                crate::flux_radiance_cache::barrier(device,cb);
                device.cmd_fill_buffer(cb,data,0,vk::WHOLE_SIZE,0);
                crate::flux_radiance_cache::barrier(device,cb);
                result=cache.prepare(device,&mut alloc,cb,0,0,1,Vec3::ZERO,99,32,true).map(|_|());
            })?;
        }
        result?;
        let (data,work,epoch)=cache.debug(0);let index=8194u32;
        crate::alloc::one_time_submit(device,r.command_pool,r.ctx.queue,|cb|unsafe {
            crate::flux_radiance_cache::barrier(device,cb);
            device.cmd_update_buffer(cb,data,index as u64*PROBE_BYTES+16,bytemuck::cast_slice(&[epoch,1,1,resolution]));
            crate::flux_radiance_cache::barrier(device,cb);
        })?;
        let mut result=Ok(());
        {
            let mut alloc=allocator.lock().unwrap();
            crate::alloc::one_time_submit(device,r.command_pool,r.ctx.queue,|cb| {
                result=cache.prepare(device,&mut alloc,cb,0,0,2,Vec3::new(camera_x,0.0,0.0),99,32,true).map(|_|());
            })?;
        }
        result?;
        let bytes=read(r,work,0,48*4)?;
        let words:Vec<_>=bytes.chunks_exact(4).map(|v|u32::from_ne_bytes(v.try_into().unwrap())).collect();
        assert_eq!(words[9],expected_rays,"resolution hysteresis or upgrade cost is incorrect");
        let bytes=read(r,data,index as u64*PROBE_BYTES,32)?;
        let header:Vec<_>=bytes.chunks_exact(4).map(|v|u32::from_ne_bytes(v.try_into().unwrap())).collect();
        assert_eq!(&header[..4],&[2,0,0,2]);assert_eq!(header[4],epoch);assert_eq!(header[7],resolution,
            "queued resolution change discarded old lighting before replacement");
        assert_eq!(cache.debug(0).2,epoch,"camera movement invalidated the cache");
        eprintln!("[Flux adaptive transition] resolution={resolution} camera_x={camera_x} rays={expected_rays} history retained");
    }
    cache.destroy(device,&mut allocator.lock().unwrap());Ok(())
}

fn adaptive_transport(r:&mut Renderer)->anyhow::Result<()> {
    let mesh=r.upload_mesh(&cube())?;
    let material=lit(r,[0.8;4],[0.0;3],1.0,0.0);
    let transform=Mat4::from_scale_rotation_translation(Vec3::new(40.0,0.2,40.0),Quat::IDENTITY,Vec3::new(0.0,-0.1,0.0));
    let mut floor=draw(mesh,material,transform,-1);floor.gi_source=Some(GiSource {object:0xadab,slot:0});
    let draws=[floor];
    r.gi_set_mesh_sdfs(&[crate::GiMeshSdf {gi_source:floor.gi_source,world_to_local:transform.inverse(),volume:super::flux::cube_distance_field(),
        in_global_field:false,material:GpuGiMaterial {albedo:[0.8;3],emission:[0.0;3],metallic:0.0,roughness:1.0}}])?;
    r.gi_set_gdf(&vec![100.0;512],&vec![0;512],[8;3],[-200.0;3],[200.0;3],&[GpuGiMaterial {albedo:[0.8;3],emission:[0.0;3],metallic:0.0,roughness:1.0}]);
    r.gi_set_emitters(&[]);r.set_area_lights(&[]);r.set_gi_enabled(true);
    let eye=Vec3::new(0.0,20.0,60.0);
    let cam=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(0.8,16.0/9.0,0.1,250.0)};
    let mut input=crate::cascade_tests::input(cam);input.camera_preview=Some(cam);input.viewport_extent=Some([323,181]);input.draws=&draws;
    input.light.ambient=[0.3;3];input.light.intensity=0.0;input.light.env_intensity=0.0;input.draw_skybox=false;
    input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
    for hardware in [false,true] {
        if hardware && !r.ctx.ray_tracing() {continue;}
        r.set_flux_settings(FluxSettings {rt_trace:hardware,samples:64,bounces:1,march_distance:180.0,firefly_clamp:16.0,reflection_mode:0,ssr_enabled:false,..Default::default()});
        let mut energy=Vec::new();
        for mode in 0..3 {
            let cache=&mut r.gpu_gi.as_mut().unwrap().radiance_cache;
            cache.enabled_override=Some(mode!=0);cache.adaptive_resolution_override=Some(mode==2);
            r.invalidate_screen_gi();
            for _ in 0..96 {r.render(&input)?;}
            let mut sum=[0.0;2];
            for _ in 0..8 {r.render(&input)?;for view in 0..2 {sum[view]+=hdr(r,view,cam)?/8.0;}}
            energy.push(sum);
            if mode==2 {
                for view in 0..2 {
                    let state=stats(r,view)?;
                    assert!(state[3]>0 && state[9]<=32*256,"distant cache unused or over budget: {state:?}");
                    let (data,_,epoch)=r.gpu_gi.as_ref().unwrap().radiance_cache.debug(view);
                    let bytes=read(r,data,0,(crate::flux_radiance_cache::PROBES as u64*crate::flux_radiance_cache::PROBE_BYTES) as usize)?;
                    let mut distant=0;
                    for probe in bytes.chunks_exact(crate::flux_radiance_cache::PROBE_BYTES as usize) {
                        let epoch_value=u32::from_ne_bytes(probe[16..20].try_into().unwrap());
                        let resolution=u32::from_ne_bytes(probe[28..32].try_into().unwrap());
                        if epoch_value==epoch && resolution==8 {
                            distant+=1;
                            for ray in probe[32..32+64*8].chunks_exact(8) {
                                for channel in ray.chunks_exact(2) {assert!(half::f16::from_bits(u16::from_ne_bytes(channel.try_into().unwrap())).is_finite());}
                            }
                        }
                    }
                    assert!(distant>8,"distant angular resolution was never traced");
                    eprintln!("[Flux adaptive transport] hardware={hardware} view={view} distant={distant} work={:?}",&state[..12]);
                }
            }
        }
        eprintln!("[Flux adaptive HDR] hardware={hardware}: {energy:?}");
        for view in 0..2 {
            assert!(energy[0][view]>0.1);
            for mode in 1..3 {assert!((energy[mode][view]-energy[0][view]).abs()<energy[0][view]*0.08,"distant cache changed sky irradiance: {energy:?}");}
        }
        input.viewport_extent=Some([1280,720]);input.camera_preview=None;
        let mut timings=[Vec::new(),Vec::new()];
        for adaptive in [false,true,true,false] {
            r.gpu_gi.as_mut().unwrap().radiance_cache.adaptive_resolution_override=Some(adaptive);
            for _ in 0..24 {r.render(&input)?;}
            for _ in 0..32 {
                r.render(&input)?;let ms=r.stats().gpu_gi_ms;
                if ms>0.0 {timings[usize::from(adaptive)].push(ms);}
            }
        }
        for values in &mut timings {values.sort_by(f32::total_cmp);assert!(!values.is_empty());}
        eprintln!("[Flux adaptive timing] hardware={hardware} fixed={}ms adaptive={}ms (total GI,1280x720,alternating blocks)",
            timings[0][timings[0].len()/2],timings[1][timings[1].len()/2]);
        input.viewport_extent=Some([323,181]);input.camera_preview=Some(cam);
    }
    let cache=&mut r.gpu_gi.as_mut().unwrap().radiance_cache;
    cache.enabled_override=None;cache.adaptive_resolution_override=None;
    Ok(())
}
