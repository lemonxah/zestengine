//! Real off-screen camera textures must drive emission and Area Light GI.
use glam::{Mat4,Quat,Vec3};
use crate::{CameraData,CameraTargetView,FluxSettings,GiSource,GpuGiMaterial,Renderer};
use super::planar::{cube,draw,lit};

fn atlas_rgb(r:&Renderer)->anyhow::Result<[f32;3]> {
    use ash::vk;
    let device=&r.ctx.device;let allocator=r.allocator();let mut alloc=allocator.lock().unwrap();
    let mut host=crate::alloc::Buffer::new(device,&mut alloc,8,vk::BufferUsageFlags::TRANSFER_DST,
        gpu_allocator::MemoryLocation::GpuToCpu,"live Area Light atlas readback")?;
    crate::alloc::one_time_submit(device,r.command_pool,r.ctx.queue,|cb|unsafe {
        crate::image_barrier(device,cb,r.area_atlas.image,vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::PipelineStageFlags2::ALL_COMMANDS,vk::AccessFlags2::MEMORY_READ|vk::AccessFlags2::MEMORY_WRITE,
            vk::PipelineStageFlags2::TRANSFER,vk::AccessFlags2::TRANSFER_READ);
        device.cmd_copy_image_to_buffer(cb,r.area_atlas.image,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,host.handle,&[
            vk::BufferImageCopy::default().image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                .image_offset(vk::Offset3D {x:128,y:128,z:0}).image_extent(vk::Extent3D {width:1,height:1,depth:1})]);
        crate::image_barrier(device,cb,r.area_atlas.image,vk::ImageAspectFlags::COLOR,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::TRANSFER,vk::AccessFlags2::TRANSFER_READ,
            vk::PipelineStageFlags2::ALL_COMMANDS,vk::AccessFlags2::MEMORY_READ);
    })?;
    let bytes=host.read();let rgb=std::array::from_fn(|c|half::f16::from_bits(u16::from_ne_bytes([bytes[c*2],bytes[c*2+1]])).to_f32());
    host.destroy(device,&mut alloc);Ok(rgb)
}

pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    let mesh=r.upload_mesh(&cube())?;
    let floor=lit(r,[0.8;4],[0.0;3],1.0,0.0);
    let source=lit(r,[0.0,0.0,0.0,1.0],[8.0;3],1.0,0.0);
    let far_material=lit(r,[0.0,0.0,0.0,1.0],[8.0;3],1.0,0.0);
    for m in [source,far_material] {r.set_material_features(m,crate::MaterialFeatures {emission:true,..Default::default()});}
    let movie=r.camera_target(0x11ce,16,16)?;let far_movie=r.camera_target(0x11cf,16,16)?;
    let mut slots=[None;16];slots[3]=Some(far_movie);r.set_material_textures(far_material,&slots);
    let mut draws=[
        draw(mesh,floor,Mat4::from_scale_rotation_translation(Vec3::new(12.0,0.2,12.0),Quat::IDENTITY,Vec3::new(0.0,-0.1,0.0)),-1),
        draw(mesh,source,Mat4::from_scale_rotation_translation(Vec3::new(8.0,8.0,0.2),Quat::IDENTITY,Vec3::new(0.0,4.0,12.0)),-1),
        draw(mesh,far_material,Mat4::from_translation(Vec3::splat(10000.0)),-1)];
    for (i,d) in draws.iter_mut().enumerate() {d.gi_source=Some(GiSource {object:0x11ce0+i as u128,slot:0});d.bound_radius=0.8661;}
    let fields:Vec<_>=draws.iter().take(2).map(|d|crate::GiMeshSdf {gi_source:d.gi_source,world_to_local:d.transform.inverse(),
        volume:super::flux::cube_distance_field(),in_global_field:false,
        material:GpuGiMaterial {albedo:[0.8;3],emission:[0.0;3],metallic:0.0,roughness:1.0}}).collect();
    r.gi_set_mesh_sdfs(&fields)?;
    r.gi_set_gdf(&vec![100.0;512],&vec![0;512],[8;3],[-30.0;3],[30.0;3],&[fields[0].material]);
    r.gi_set_emitters(&[]);r.set_area_lights(&[]);r.set_gi_enabled(true);
    let eye=Vec3::new(0.0,2.0,4.0);
    let cam=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    assert!(!crate::sphere_in_planes(&crate::frustum_planes(cam.proj*cam.view),&draws[1],true),"test emitter must be off-screen");
    let remote=Vec3::new(1000.0,1000.0,1000.0);
    let feeds=[0x11ce,0x11cf].map(|id|CameraTargetView {id,size:[16,16],camera:CameraData {position:remote,
        view:Mat4::look_at_rh(remote,remote+Vec3::Z,Vec3::Y),..cam}});
    let point_lights=[crate::LightInstance {kind:crate::LightKind::Point,position:Vec3::new(0.0,4.0,10.0),direction:Vec3::Z,
        color:[1.0;3],intensity:80.0,range:20.0,spot_inner_deg:30.0,spot_outer_deg:45.0,
        cast_shadows:false,soft_shadows:false,radius:0.0,shadow_bias:0.0,baked:false}];
    let mut input=crate::cascade_tests::input(cam);input.camera_preview=Some(cam);input.viewport_extent=Some([323,181]);
    input.draws=&draws;input.draw_skybox=false;input.light.ambient=[0.0;3];input.light.intensity=0.0;input.light.env_intensity=0.0;
    input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
    let hdr=|r:&Renderer|->anyhow::Result<[f32;2]> {Ok([super::flux_radiance::hdr(r,0,cam)?,super::flux_radiance::hdr(r,1,cam)?])};
    for hardware in [false,true] {
        if hardware && !r.ctx.ray_tracing() {continue;}
        r.set_flux_settings(FluxSettings {rt_trace:hardware,samples:32,bounces:4,march_distance:40.0,reflection_mode:0,ssr_enabled:false,..Default::default()});
        r.set_area_lights(&[]);slots=[None;16];slots[3]=Some(movie);r.set_material_textures(source,&slots);
        r.material_params_mut(source).base_color=[0.0,0.0,0.0,1.0];r.material_params_mut(source).emission_color=[8.0;3];
        input.camera_targets=&feeds;input.clear_color=[1.0,0.25,0.125,1.0];
        let far_revision=r.texture_revisions[&far_movie.0];let before=r.texture_revisions[&movie.0];
        for _ in 0..96 {r.render(&input)?;}
        let bright=hdr(r)?;
        assert!(bright.iter().all(|v|*v>0.01),"off-screen TV did not illuminate either view: {bright:?}");
        assert_eq!(r.texture_revisions[&movie.0],before+96,"off-screen GI source was not rendered each requested frame");
        assert_eq!(r.texture_revisions[&far_movie.0],far_revision,"unrelated distant source was rendered");
        let epochs=std::array::from_fn::<_,2,_>(|v|r.gpu_gi.as_ref().unwrap().radiance_cache.debug(v).2);
        input.clear_color=[0.0,0.0,0.0,1.0];
        for _ in 0..128 {r.render(&input)?;}
        let dark=hdr(r)?;
        for v in 0..2 {assert!(dark[v]<bright[v]*0.05,"off-screen TV retained stale light: {bright:?}->{dark:?}");
            assert_eq!(r.gpu_gi.as_ref().unwrap().radiance_cache.debug(v).2,epochs[v],"live texture reset global radiance");}
        input.camera_targets=&[];let revision=r.texture_revisions[&movie.0];
        // Live content has priority over resolution migration. Once paused,
        // a deferred resize may capture the same content at its desired mip;
        // only those migrations are permitted without a content revision.
        for frame in 0..16 {r.render(&input)?;if frame>=4 {
            let cache=r.flux_surface.as_ref().unwrap();
            assert_eq!(cache.captures,cache.migration_count,"paused TV invalidated material without a content change");
        }}
        assert_eq!(r.texture_revisions[&movie.0],revision);
        eprintln!("[Flux live off-screen TV] hardware={hardware} bright={bright:?} dark={dark:?}; unrelated and paused textures idle");

        // Same feed, consumed exclusively by an Area Light: no visible material
        // samples it. The light illuminates the wall, whose bounce reaches zero.
        r.set_material_textures(source,&[None;16]);r.material_params_mut(source).emission_color=[0.0;3];
        r.material_params_mut(source).base_color=[0.8,0.8,0.8,1.0];
        r.set_area_lights(&[crate::AreaLightInstance {shape:crate::AreaShape::Rect,position:Vec3::new(0.0,4.0,10.0),
            right:Vec3::X,up:Vec3::Y,half_size:[2.0;2],color:[0.5,1.0,0.25],intensity:8.0,range:20.0,two_sided:false,texture:Some(movie)}]);
        input.camera_targets=&feeds;input.clear_color=[1.0;4];let before=r.texture_revisions[&movie.0];
        r.request_reflection_capture_slot(Vec3::new(0.0,4.0,8.0),32,0,0.0,false,true);
        for frame in 0..96 {r.render(&input)?;assert_eq!(r.area_atlas.copies,1,"live Area Light tile was not updated");
            if frame==0 {
                assert_eq!(atlas_rgb(r)?,[1.0;3],"atlas copied the previous camera image");
                assert!(!r.reflection_capture_in_flight());
                let faces=r.read_reflection_faces().expect("live reflection capture missing");
                let face=&faces[4];let at=((face.height/2*face.width+face.width/2)*8) as usize;
                let rgb:[f32;3]=std::array::from_fn(|c|half::f16::from_bits(u16::from_ne_bytes([face.pixels[at+c*2],face.pixels[at+c*2+1]])).to_f32());
                // Unequal channel ratios also distinguish the lit wall from
                // the white clear color if a broken capture omitted geometry.
                assert!(rgb[0]>0.001 && (rgb[1]/rgb[0]-2.0).abs()<0.1 && (rgb[2]/rgb[0]-0.5).abs()<0.05,
                    "reflection capture missed current textured Area Light shading: {rgb:?}");
            }}
        assert_eq!(r.texture_revisions[&movie.0],before+96,"Area Light-only feed was not rendered");
        let bright=hdr(r)?;assert!(bright.iter().all(|v|*v>0.005),"live Area Light produced no indirect light: {bright:?}");
        let epochs=std::array::from_fn::<_,2,_>(|v|r.gpu_gi.as_ref().unwrap().radiance_cache.debug(v).2);
        let direct={let allocator=r.allocator();let mut alloc=allocator.lock().unwrap();
            r.flux_surface.as_ref().unwrap().debug_lighting(&r.ctx.device,&mut alloc,r.command_pool,r.ctx.queue,source.0,5)?[0]};
        assert!(direct[0]>0.001 && (direct[1]/direct[0]-2.0).abs()<0.02 && (direct[2]/direct[0]-0.5).abs()<0.01,
            "live Area Light lost its color multiplier: {direct:?}");
        input.clear_color=[0.0,0.0,0.0,1.0];
        for frame in 0..128 {r.render(&input)?;if frame==0 {assert_eq!(atlas_rgb(r)?,[0.0;3],"Area Light turn-off lagged the source camera");}}
        let dark=hdr(r)?;assert!(dark.iter().zip(bright).all(|(d,b)|*d<b*0.05),"Area Light ignored changed pixels: {bright:?}->{dark:?}");
        for v in 0..2 {assert_eq!(r.gpu_gi.as_ref().unwrap().radiance_cache.debug(v).2,epochs[v],"live Area Light reset global radiance");}
        input.camera_targets=&[];
        for frame in 0..16 {r.render(&input)?;assert_eq!(r.area_atlas.copies,0,"unchanged Area Light atlas was recopied");
            if frame>=4 {let cache=r.flux_surface.as_ref().unwrap();
                assert_eq!(cache.captures,cache.migration_count,"paused Area Light invalidated captured materials");}}
        eprintln!("[Flux live Area Light] hardware={hardware} bright={bright:?} dark={dark:?}; unchanged atlas idle");

        // Off-screen reflectance also matters: a lit surface whose live albedo
        // changes to black must stop bouncing light, even with emission zero.
        r.set_area_lights(&[]);slots=[None;16];slots[0]=Some(movie);r.set_material_textures(source,&slots);
        input.lights=&point_lights;input.camera_targets=&feeds;input.clear_color=[1.0;4];
        let before=r.texture_revisions[&movie.0];
        for _ in 0..96 {r.render(&input)?;}
        assert_eq!(r.texture_revisions[&movie.0],before+96,"off-screen reflectance feed was not rendered");
        let bright=hdr(r)?;assert!(bright.iter().all(|v|*v>0.005),"live reflectance produced no bounce: {bright:?}");
        input.clear_color=[0.0,0.0,0.0,1.0];
        for _ in 0..128 {r.render(&input)?;}
        let dark=hdr(r)?;assert!(dark.iter().zip(bright).all(|(d,b)|*d<b*0.05),"off-screen black albedo retained bounce: {bright:?}->{dark:?}");
        input.lights=&[];input.camera_targets=&[];
        eprintln!("[Flux live off-screen reflectance] hardware={hardware} bright={bright:?} dark={dark:?}");
    }
    r.set_area_lights(&[]);Ok(())
}
