//! Receiver motion is independent of TAA, draw ordering and GI backend.
use ash::vk;
use glam::{Mat4,Quat,Vec3};
use crate::{CameraData,FluxSettings,GiSource,GpuGiMaterial,Renderer};
use super::planar::{cube,draw,lit};

fn pixel(r:&Renderer,image:vk::Image,extent:vk::Extent2D,camera:CameraData,point:Vec3,half:bool)->anyhow::Result<[f32;4]> {
    use crate::alloc::{Buffer,one_time_submit};
    let device=&r.ctx.device;unsafe {device.device_wait_idle()?;}
    let q=(camera.proj*camera.view).project_point3(point);
    let offset=vk::Offset3D {x:((q.x*0.5+0.5)*extent.width as f32) as i32,y:((0.5-q.y*0.5)*extent.height as f32) as i32,z:0};
    let allocator=r.allocator();let mut alloc=allocator.lock().unwrap();
    let mut host=Buffer::new(device,&mut alloc,16,vk::BufferUsageFlags::TRANSFER_DST,gpu_allocator::MemoryLocation::GpuToCpu,"Motion history test")?;
    one_time_submit(device,r.command_pool,r.ctx.queue,|cb|unsafe {
        crate::image_barrier(device,cb,image,vk::ImageAspectFlags::COLOR,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::PipelineStageFlags2::ALL_COMMANDS,vk::AccessFlags2::MEMORY_READ|vk::AccessFlags2::MEMORY_WRITE,vk::PipelineStageFlags2::TRANSFER,vk::AccessFlags2::TRANSFER_READ);
        device.cmd_copy_image_to_buffer(cb,image,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,host.handle,&[vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
            .image_offset(offset).image_extent(vk::Extent3D {width:1,height:1,depth:1})]);
        crate::image_barrier(device,cb,image,vk::ImageAspectFlags::COLOR,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::TRANSFER,vk::AccessFlags2::TRANSFER_READ,vk::PipelineStageFlags2::ALL_COMMANDS,vk::AccessFlags2::MEMORY_READ);
        device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::TRANSFER,vk::PipelineStageFlags::HOST,vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ)],&[],&[]);
    })?;
    let bytes=host.read();let value=std::array::from_fn(|i|if half {half::f16::from_bits(u16::from_ne_bytes(bytes[i*2..i*2+2].try_into().unwrap())).to_f32()}
        else {f32::from_ne_bytes(bytes[i*4..i*4+4].try_into().unwrap())});
    host.destroy(device,&mut alloc);Ok(value)
}
fn motion(r:&Renderer,view:usize,cam:CameraData,p:Vec3)->anyhow::Result<[f32;4]> {
    let (image,extent)=r.lighting_motion.debug(view);pixel(r,image,extent,cam,p,false)
}
fn gi_frames(r:&Renderer,view:usize,cam:CameraData,p:Vec3)->anyhow::Result<f32> {
    let (t,extent,parity)=match view {
        0=>{let v=r.viewport_target.as_ref().unwrap();(&v.sgi,v.extent,r.sgi_parity)},
        1=>{let v=r.camera_preview.as_ref().unwrap();(&v.sgi,v.extent,r.preview_sgi_parity)},
        _=>(&r.sgi,vk::Extent2D {width:r.render_extent[0],height:r.render_extent[1]},r.sgi_parity),
    };
    Ok(pixel(r,t.temporal((parity^1)&1).current[0],extent,cam,p,true)?[3])
}
fn directional_frames(r:&Renderer,view:usize,cam:CameraData,p:Vec3)->anyhow::Result<f32> {
    let (t,parity)=if view==0 {(&r.viewport_target.as_ref().unwrap().sgi,r.sgi_parity)} else {(&r.camera_preview.as_ref().unwrap().sgi,r.preview_sgi_parity)};
    let q=(cam.proj*cam.view).project_point3(p);let size=t.probe_extent;
    let x=((q.x*0.5+0.5)*size.width as f32) as u32;let y=((0.5-q.y*0.5)*size.height as f32) as u32;
    let data=super::flux_radiance::read(r,t.sh_buf[(parity^1)&1].handle,(y*size.width+x) as u64*crate::gpu_gi::SCREEN_PROBE_BYTES,crate::gpu_gi::SCREEN_PROBE_BYTES as usize)?;
    Ok((0..16).map(|i|{let at=80+i*16+14;half::f16::from_bits(u16::from_ne_bytes([data[at],data[at+1]])).to_f32()}).sum::<f32>()/16.0)
}
pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    r.set_anti_aliasing(crate::AntiAliasing::None);r.set_area_lights(&[]);r.gi_set_emitters(&[]);r.gi_set_mesh_sdfs(&[])?;
    r.gi_set_gdf(&[100.0;512],&[0;512],[8;3],[-20.0;3],[20.0;3],&[GpuGiMaterial {albedo:[0.0;3],emission:[0.0;3],metallic:0.0,roughness:1.0}]);
    let mesh=r.upload_mesh(&cube())?;let material=lit(r,[0.8;4],[0.0;3],0.7,0.0);
    let base=Mat4::from_scale(Vec3::new(6.0,6.0,0.2));
    let mut d=draw(mesh,material,base,-1);d.gi_source=Some(GiSource {object:0x60710,slot:0});
    let eye=Vec3::new(0.0,0.0,5.0);let camera=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),
        proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let mut input=crate::cascade_tests::input(camera);input.viewport_extent=Some([323,181]);input.camera_preview=Some(camera);
    input.light.intensity=0.0;input.light.ambient=[0.1;3];input.light.env_intensity=0.0;input.draw_skybox=false;
    for hardware in [false,true] {
        if hardware && !r.ctx.ray_tracing() {continue;}
        r.set_gi_enabled(true);r.set_flux_settings(FluxSettings {rt_trace:hardware,gi_screen_traces:false,reflection_mode:0,samples:16,..Default::default()});
        for gpu in [false,true] {
            r.set_gpu_cull_settings(crate::GpuCullSettings {enabled:gpu,..Default::default()});
            let mut previous=base;
            for frame in 0..12 {
                // Translation normal to the receiving plane would reject all
                // history if reprojection used today's world position.
                d.transform=Mat4::from_translation(Vec3::new(0.05*frame as f32,0.0,0.08*frame as f32))*base;
                let draws=[d];r.render(&crate::FrameInput {draws:&draws,egui:None,world_ui:Vec::new(),..input})?;
                assert_eq!(r.gpu_handles(0),gpu,"fixture missed requested draw path");
                let point=d.transform.transform_point3(Vec3::new(0.0,0.0,0.5));
                if frame>1 {for view in 0..2 {
                    let got=motion(r,view,camera,point)?;let expected=previous.w_axis.truncate()-d.transform.w_axis.truncate();
                    assert!(got[3]>=0.0 && Vec3::from_slice(&got[..3]).distance(expected)<0.0002,"motion view={view} gpu={gpu}: {got:?} expected {expected}");
                    assert!(gi_frames(r,view,camera,point)?>1.5,"moving receiver lost all GI history view={view} gpu={gpu} hardware={hardware}");
                    if frame>4 {assert!(directional_frames(r,view,camera,point)?>1.1,"moving receiver lost directional history view={view} gpu={gpu} hardware={hardware}");}
                }}
                previous=d.transform;
            }
            // Replacement at the same screen location must not borrow lighting.
            d.gi_source.as_mut().unwrap().object+=1;
            let draws=[d];r.render(&crate::FrameInput {draws:&draws,egui:None,world_ui:Vec::new(),..input})?;
            let p=d.transform.transform_point3(Vec3::new(0.0,0.0,0.5));
            for view in 0..2 {assert!(motion(r,view,camera,p)?[3]<0.0);assert_eq!(gi_frames(r,view,camera,p)?,1.0);}
            eprintln!("[Flux motion] hardware={hardware} GPU draws={gpu}: Scene/Camera translated receiver history and replacement rejection passed");
        }
        // Reflections still receive motion with both GI and AO disabled.
        r.set_gi_enabled(false);r.set_flux_settings(FluxSettings {reflection_mode:2,reflection_screen_traces:false,ssr_roughness_cutoff:0.9,..r.flux_settings});
        for frame in 0..5 {
            d.transform=Mat4::from_translation(Vec3::new(0.0,0.0,frame as f32*0.1))*base;
            let draws=[d];r.render(&crate::FrameInput {draws:&draws,egui:None,world_ui:Vec::new(),..input})?;
            if frame>0 {let p=Vec3::new(0.0,0.0,frame as f32*0.1+0.1);for view in 0..2 {
                let m=motion(r,view,camera,p)?;assert!((m[2]+0.1).abs()<0.0002 && m[3]>=0.0,"GI-off reflection motion {m:?}");
                if frame>1 {assert!(super::reflection_denoise::frames(r,view,p)?>1,"moving reflection receiver lost history hardware={hardware} view={view}");}
            }}
        }
    }
    // Affine motion: read a projected sample, reconstruct its actual world
    // position, and compare with previousModel * inverse(currentModel).
    let previous=d.transform;
    d.transform=Mat4::from_scale_rotation_translation(Vec3::new(5.0,4.0,0.3),Quat::from_rotation_y(0.2),Vec3::new(0.1,0.0,0.0));
    let draws=[d];r.render(&crate::FrameInput {draws:&draws,egui:None,world_ui:Vec::new(),..input})?;
    for view in 0..2 {
        let (image,extent)=r.lighting_motion.debug(view);let p=Vec3::new(0.5,0.0,0.0);
        let m=pixel(r,image,extent,camera,p,false)?;assert!(m[3]>=0.0);
        let q=(camera.proj*camera.view).project_point3(p);
        let px=((q.x*0.5+0.5)*extent.width as f32).floor();let py=((0.5-q.y*0.5)*extent.height as f32).floor();
        let world=(camera.proj*camera.view).inverse().project_point3(Vec3::new((px+0.5)/extent.width as f32*2.0-1.0,1.0-(py+0.5)/extent.height as f32*2.0,m[3]));
        let wanted=(previous*d.transform.inverse()).transform_point3(world)-world;
        assert!(Vec3::from_slice(&m[..3]).distance(wanted)<0.001,"affine motion: {m:?} expected {wanted}");
    }
    // Standalone Game uses a separate motion target and no editor viewport.
    input.draws=&draws;input.viewport_extent=None;input.camera_preview=None;r.set_gi_enabled(true);
    for _ in 0..4 {r.render(&input)?;}
    assert!(motion(r,2,camera,Vec3::new(0.1,0.0,0.0))?[3]>=0.0);
    assert!(gi_frames(r,2,camera,Vec3::new(0.1,0.0,0.0))?>1.5);
    // Previous deformed vertices must remain correct when animation pauses.
    let mut geometry=cube();
    for v in &mut geometry.vertices {v.joints=[0;4];v.weights=[1.0,0.0,0.0,0.0];}
    let skin=r.upload_mesh_gpu_skinned(&geometry,1)?;
    let mut actor=draw(skin,material,Mat4::from_scale(Vec3::new(6.0,6.0,1.0)),-1);
    actor.gi_source=Some(GiSource {object:0x5b010,slot:0});
    let actors=[actor];input.draws=&actors;input.viewport_extent=Some([323,181]);input.camera_preview=Some(camera);
    for frame in 0..9 {
        let pose=frame.min(5) as f32*0.1;
        if frame<=5 {r.skin_mesh_gpu(skin,&[Mat4::from_translation(Vec3::new(0.0,0.0,pose))]);}
        r.render(&input)?;
        if frame>1 {for view in 0..2 {
            let m=motion(r,view,camera,Vec3::new(0.0,0.0,0.5+pose))?;
            let expected=if frame<=5 {-0.1} else {0.0};
            assert!(m[3]>=0.0 && (m[2]-expected).abs()<0.0002,"deformed/paused receiver frame={frame} view={view}: {m:?}, expected {expected}");
        }}
    }
    // Cluster vertex pulling has a different instance ABI from ordinary
    // indirect draws. Read its rendered motion rather than only its CPU record.
    let cluster=super::xr_cluster::Scene::new(r)?;
    let mut clustered=draw(cluster.mesh,cluster.material,Mat4::IDENTITY,-1);
    clustered.gi_source=Some(GiSource {object:0xc10710,slot:0});
    for frame in 0..4 {
        clustered.transform=Mat4::from_translation(Vec3::new(1.8,1.8,frame as f32*0.1));
        let geometry=[clustered];
        r.render(&crate::FrameInput {draws:&geometry,egui:None,world_ui:Vec::new(),..input})?;
        assert!(r.cluster.as_ref().is_some_and(|c|!c.instances.is_empty()),"fixture missed cluster vertex pulling");
        if frame>0 {let p=Vec3::new(0.0,0.0,frame as f32*0.1);for view in 0..2 {
            let m=motion(r,view,camera,p)?;
            assert!(m[3]>=0.0 && (m[2]+0.1).abs()<0.0002,"cluster receiver motion view={view}: {m:?}");
        }}
    }
    eprintln!("[Flux motion] GI-off reflection history, rotation/scale, standalone Game history, paused skin and cluster vertex pulling passed; TAA disabled throughout");
    Ok(())
}

#[test]
fn lighting_motion_tracks_identity_and_rejects_ambiguous_correspondence() {
    let mut m=crate::lighting_motion::Motion::default();
    let mut a=draw(crate::MeshHandle(0),crate::MaterialHandle(0),Mat4::IDENTITY,-1);
    a.gi_source=Some(GiSource {object:1,slot:0});let mut b=a;b.gi_source.as_mut().unwrap().object=2;b.transform=Mat4::from_translation(Vec3::X);
    m.advance(&[a,b],1);assert!(m.previous.iter().all(|v|v[3][3]==0.0));
    let mut moved=a;moved.transform=Mat4::from_translation(Vec3::Y);
    m.advance(&[b,moved],2);assert_eq!(m.previous[1],a.transform.to_cols_array_2d());
    m.advance(&[moved,moved],3);assert!(m.previous.iter().all(|v|v[3][3]==0.0));
    m.advance(&[a],4);assert_eq!(m.previous[0][3][3],0.0);
    m.advance(&[],5);m.advance(&[a],6);assert_eq!(m.previous[0][3][3],0.0);
    m.advance(&[a],8);assert_eq!(m.previous[0][3][3],0.0);
    a.gi_source=None;m.advance(&[a],9);m.advance(&[a],10);assert_eq!(m.previous[0][3][3],1.0);
    a.transform=Mat4::from_translation(Vec3::X);m.advance(&[a],11);assert_eq!(m.previous[0][3][3],0.0);
}
