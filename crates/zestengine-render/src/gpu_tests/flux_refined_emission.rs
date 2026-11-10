//! Live off-screen emission must refresh every resident virtual mip within budget.
use glam::{Mat4,Quat,Vec3};
use crate::{CameraData,CameraTargetView,FluxSettings,GiSource,GpuGiMaterial,Renderer};
use super::planar::{cube,draw,lit};

pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    r.set_area_lights(&[]);r.set_gi_enabled(true);r.set_anti_aliasing(crate::AntiAliasing::None);
    let mesh=r.upload_mesh(&cube())?;
    let floor=lit(r,[0.8;4],[0.0;3],1.0,0.0);
    let source=lit(r,[0.0,0.0,0.0,1.0],[8.0,4.0,2.0],1.0,0.0);
    r.set_material_features(source,crate::MaterialFeatures {emission:true,..Default::default()});
    let movie=r.camera_target(0xf1ee,16,16)?;
    let mut slots=[None;16];slots[3]=Some(movie);r.set_material_textures(source,&slots);
    let mut draws=[
        draw(mesh,floor,Mat4::from_scale_rotation_translation(Vec3::new(12.0,0.2,12.0),Quat::IDENTITY,Vec3::new(0.0,-0.1,0.0)),-1),
        draw(mesh,source,Mat4::from_scale_rotation_translation(Vec3::new(32.0,0.2,32.0),Quat::IDENTITY,Vec3::new(0.0,4.0,0.0)),-1)];
    for (i,d) in draws.iter_mut().enumerate() {d.gi_source=Some(GiSource {object:0xf1ee00+i as u128,slot:0});}
    let fields:Vec<_>=draws.iter().map(|d|crate::GiMeshSdf {gi_source:d.gi_source,world_to_local:d.transform.inverse(),
        volume:super::flux::cube_distance_field(),in_global_field:false,
        material:GpuGiMaterial {albedo:[0.0;3],emission:[0.0;3],metallic:0.0,roughness:1.0}}).collect();
    r.gi_set_mesh_sdfs(&fields)?;
    r.gi_set_gdf(&[100.0;512],&[0;512],[8;3],[-40.0;3],[40.0;3],&[fields[0].material]);r.gi_set_emitters(&[]);
    let eye=Vec3::new(0.0,2.0,4.0);
    let cam=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    // Every emitter corner is outside the upper view plane. Demand for its
    // fine pages must come from indirect hits, not raster visibility.
    let plane=crate::frustum_planes(cam.proj*cam.view).into_iter().any(|p| {
        let centre=Vec3::new(0.0,4.0,0.0);let half=Vec3::new(16.0,0.1,16.0);
        p.truncate().dot(centre)+p.w+half.dot(p.truncate().abs())<0.0
    });
    assert!(plane,"refined emitter must be outside the view");
    let remote=Vec3::splat(1000.0);
    let feeds=[CameraTargetView {id:0xf1ee,size:[16,16],camera:CameraData {position:remote,
        view:Mat4::look_at_rh(remote,remote+Vec3::Z,Vec3::Y),..cam}}];
    let mut input=crate::cascade_tests::input(cam);input.camera_preview=Some(cam);input.viewport_extent=Some([323,181]);
    input.draws=&draws;input.draw_skybox=false;input.light.ambient=[0.0;3];input.light.intensity=0.0;input.light.env_intensity=0.0;
    input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
    let hdr=|r:&Renderer|->anyhow::Result<[f32;2]> {Ok([super::flux_radiance::hdr(r,0,cam)?,super::flux_radiance::hdr(r,1,cam)?])};
    for hardware in [false,true] {
        if hardware && !r.ctx.ray_tracing() {continue;}
        r.set_flux_settings(FluxSettings {rt_trace:hardware,samples:32,bounces:4,march_distance:40.0,
            reflection_mode:0,ssr_enabled:false,..Default::default()});
        input.camera_targets=&feeds;input.clear_color=[0.5,0.25,0.25,1.0];
        let mut colours=std::collections::BTreeMap::new();
        let step=|r:&mut Renderer,input:&crate::FrameInput,colours:&mut std::collections::BTreeMap<u64,[f32;4]>,n:usize|->anyhow::Result<()> {
            for _ in 0..n {
                r.render(input)?;colours.insert(r.frame_serial,input.clear_color);
                let cache=r.flux_surface.as_ref().unwrap();
                assert!(cache.captures<=12 && cache.capture_texels<=12*64*64,"live source exceeded capture budget");
                assert!(cache.lighting_texels[0]<=24*64*64 && cache.lighting_texels[1]<=12*64*64,"live source exceeded lighting budget");
            }Ok(())
        };
        let check=|r:&Renderer,colours:&std::collections::BTreeMap<u64,[f32;4]>,max_age:u64|->anyhow::Result<()> {
            let cache=r.flux_surface.as_ref().unwrap();let pages=cache.debug_virtual_cards(source.0,3);
            assert_eq!(pages.len(),21,"off-screen hits must request coarse+256+512 emission captures");
            let mut levels=[0;3];
            for (id,card) in pages {
                levels[card.virtual_page[1] as usize]+=1;
                let frame=cache.debug_capture_frame(id);
                assert!(r.frame_serial-frame<=max_age,"live fine page starved for {} frames",r.frame_serial-frame);
                let expected=colours[&frame];let allocator=r.allocator();
                let value=cache.debug_page_material(&r.ctx.device,&mut allocator.lock().unwrap(),r.command_pool,r.ctx.queue,card,0)?;
                assert!(value[0][3]>0.5,"missing material capture");
                for c in 0..3 {assert!((value[3][c]-expected[c]*[8.0,4.0,2.0][c]).abs()<0.015,
                    "page{:?} frame{frame}: wrong live texture times emission colour: {:?}, expected{expected:?}",card.virtual_page,value[3]);}
            }
            assert_eq!(levels,[1,4,16]);Ok(())
        };
        step(r,&input,&mut colours,96)?;check(r,&colours,24)?;
        let bright=hdr(r)?;assert!(bright.iter().all(|v|*v>0.01),"refined live emitter did not light either view: {bright:?}");
        let epochs=std::array::from_fn::<_,2,_>(|v|r.gpu_gi.as_ref().unwrap().radiance_cache.debug(v).2);
        // Change pixels every frame while retaining texture/material/geometry
        // identity. Each page must contain the pixels from its own capture frame.
        for frame in 0..48 {
            input.clear_color=match frame%3 {0=>[0.25,0.5,0.125,1.0],1=>[0.125,0.25,0.5,1.0],_=>[0.5,0.125,0.25,1.0]};
            step(r,&input,&mut colours,1)?;
            if frame%8==7 {check(r,&colours,24)?;}
        }
        input.clear_color=[0.0,0.0,0.0,1.0];step(r,&input,&mut colours,128)?;check(r,&colours,24)?;
        let dark=hdr(r)?;
        for v in 0..2 {assert!(dark[v]<bright[v]*0.05,"refined emitter retained stale energy: {bright:?}->{dark:?}");
            assert_eq!(r.gpu_gi.as_ref().unwrap().radiance_cache.debug(v).2,epochs[v],"refined video reset unrelated radiance history");}
        input.camera_targets=&[];let revision=r.texture_revisions[&movie.0];
        step(r,&input,&mut colours,32)?;
        assert_eq!(r.texture_revisions[&movie.0],revision,"paused refined feed continued rendering");
        assert_eq!(r.flux_surface.as_ref().unwrap().captures,0,"paused refined material continued capturing");
        eprintln!("[Flux refined live emission] hardware={hardware}:21 off-screen pages, budgeted per-frame video refresh, bright={bright:?} dark={dark:?}, unrelated history retained and paused feed idle");
        input.draws=&[];r.render(&input)?;input.draws=&draws;
    }
    Ok(())
}
