//! Standalone reflection material captures, live feeds and cached lighting, with no diffuse gather.
use glam::{Mat4,Quat,Vec3};
use crate::{CameraData,CameraTargetView,FluxSettings,GiMeshSdf,GiSource,GpuGiMaterial,RayLightingMode,Renderer};
use super::planar::{cube,draw,lit};
fn patch(image:&(u32,u32,Vec<u8>),cam:CameraData,x:f32)->[f32;3] {
    let (w,h,p)=image;let q=(cam.proj*cam.view).project_point3(Vec3::new(x,0.0,0.0));
    let x=((q.x*0.5+0.5)* *w as f32) as i32;let y=((0.5-q.y*0.5)* *h as f32) as i32;
    let mut sum=[0.0;3];for j in -2..=2 {for i in -2..=2 {for k in 0..3 {sum[k]+=p[(((y+j) as u32*w+(x+i) as u32)*4) as usize+k] as f32/25.0;}}}sum
}
fn samples(r:&mut Renderer,input:&crate::FrameInput,frames:usize)->anyhow::Result<[[[f32;3];2];2]> {
    for _ in 0..frames {r.render(input)?;}
    Ok([r.debug_read_viewport()?.unwrap(),r.debug_read_camera_preview()?.unwrap()].map(|im|[-0.2,0.2].map(|x|patch(&im,input.camera,x))))
}
fn colored(values:[[[f32;3];2];2],channels:[usize;2],label:&str) {
    for view in values {for side in 0..2 {for (c, c_value) in view[side].iter().enumerate() {
        assert!(if c==channels[side] {*c_value>100.0} else {*c_value<8.0},"{label}: {values:?}");
    }}}
}
pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    let old_gi=r.gi_enabled;let old_settings=r.flux_settings;
    r.set_gi_enabled(false);r.set_anti_aliasing(crate::AntiAliasing::None);r.set_area_lights(&[]);
    let mut shape=cube();for v in &mut shape.vertices {v.uv=[v.position[0]+0.5,v.position[1]+0.5];}
    let mesh=r.upload_mesh(&shape)?;
    let mirror=lit(r,[1.0;4],[0.0;3],0.02,1.0);
    let emitter=lit(r,[0.0,0.0,0.0,1.0],[2.0;3],1.0,0.0);
    let far_material=lit(r,[0.0,0.0,0.0,1.0],[2.0;3],1.0,0.0);
    let floor=Mat4::from_scale_rotation_translation(Vec3::new(8.0,0.2,8.0),Quat::IDENTITY,Vec3::new(0.0,-0.1,0.0));
    let wall=Mat4::from_scale_rotation_translation(Vec3::new(1.5,1.0,0.2),Quat::IDENTITY,Vec3::new(0.0,1.2,-2.5));
    let mut draws=[draw(mesh,mirror,floor,-1),draw(mesh,emitter,wall,-1),draw(mesh,far_material,Mat4::from_translation(Vec3::splat(10000.0)),-1)];
    for (i,d) in draws.iter_mut().enumerate() {d.gi_source=Some(GiSource {object:0xface00+i as u128,slot:0});d.bound_radius=0.8661;}
    let neutral=GpuGiMaterial {albedo:[0.5;3],emission:[0.0;3],metallic:0.0,roughness:1.0};
    let fields:Vec<_>=draws[..2].iter().map(|d|GiMeshSdf {gi_source:d.gi_source,world_to_local:d.transform.inverse(),
        volume:super::flux::cube_distance_field(),material:neutral,in_global_field:false}).collect();
    let texture=r.upload_texture(&crate::TextureData {width:32,height:1,pixels:(0..32).flat_map(|i|if i<16 {[255,0,0,255]} else {[0,255,0,255]}).collect(),srgb:false,hdr:false})?;
    let movie=r.camera_target(0xface01,16,16)?;let far_movie=r.camera_target(0xface02,16,16)?;
    let remote=Vec3::splat(1000.0);let eye=Vec3::new(0.0,2.0,4.0);
    let cam=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(0.2,16.0/9.0,0.1,100.0)};
    assert!(!crate::sphere_in_planes(&crate::frustum_planes(cam.proj*cam.view),&draws[1],true),"live source must be offscreen");
    let feeds=[0xface01,0xface02].map(|id|CameraTargetView {id,size:[16,16],camera:CameraData {position:remote,view:Mat4::look_at_rh(remote,remote+Vec3::Z,Vec3::Y),..cam}});
    let light=[crate::LightInstance {kind:crate::LightKind::Directional,position:Vec3::ZERO,direction:Vec3::NEG_Z,color:[1.0;3],intensity:8.0,
        range:0.0,spot_inner_deg:0.0,spot_outer_deg:0.0,cast_shadows:true,soft_shadows:false,radius:0.0,shadow_bias:0.0,shadow_clip:None,baked:false}];
    let mut input=crate::cascade_tests::input(cam);input.camera_preview=Some(cam);input.viewport_extent=Some([323,181]);input.draws=&draws;
    input.light.intensity=0.0;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;input.draw_skybox=false;
    input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
    let mut far_slots=[None;16];far_slots[3]=Some(far_movie);r.set_material_textures(far_material,&far_slots);
    for hardware in [true,false] {
        if hardware && !r.ctx.ray_tracing() {continue;}
        if !hardware {r.gi_set_gdf(&[100.0;512],&[0;512],[8;3],[-10.0;3],[10.0;3],&[neutral]);r.gi_set_mesh_sdfs(&fields)?;}
        let had_gdf=r.gpu_gi.as_ref().unwrap().has_gdf();
        r.set_flux_settings(FluxSettings {rt_trace:hardware,reflection_mode:2,reflection_screen_traces:false,bounces:2,..Default::default()});
        r.material_params_mut(emitter).base_color=[0.0,0.0,0.0,1.0];r.material_params_mut(emitter).emission_intensity=1.0;
        let mut slots=[None;16];slots[3]=Some(texture);r.set_material_textures(emitter,&slots);
        // Stale analytic proxies from a formerly enabled GI feed must not mask captured emission.
        r.gi_set_emitters(&[crate::GpuGiEmitter {center:[0.0;3],radius:100.0,emission:[0.0,10.0,0.0]}]);
        let gather=[r.sgi_frame,r.preview_sgi_frame];let updates=r.reflection_surface.recorded;
        let split=samples(r,&input,12)?;colored(split,[0,1],"offscreen textured emission");
        assert_eq!([r.sgi_frame,r.preview_sgi_frame],gather,"standalone reflections ran diffuse gathering");
        assert_eq!(r.reflection_surface.recorded-updates,12,"surface lighting updated more than once per frame for two views");
        if hardware && !had_gdf {assert!(!r.gpu_gi.as_ref().unwrap().has_gdf(),"hardware reflection-only path constructed a GDF");}
        assert!(r.flux_surface.as_ref().unwrap().debug_lighting_coverage().iter().all(|n|*n>0));
        assert_eq!(r.flux_surface.as_ref().unwrap().captures,0,"static reflection materials repeatedly captured");
        r.gi_set_emitters(&[]);
        // Evaluate actual lit texture response, including the shared cached mode and hit-lighting mode.
        slots[3]=None;slots[0]=Some(texture);r.set_material_textures(emitter,&slots);
        r.material_params_mut(emitter).base_color=[1.0;4];r.material_params_mut(emitter).emission_intensity=0.0;input.lights=&light;
        for mode in [RayLightingMode::SurfaceCache,RayLightingMode::HitLightingForReflections,RayLightingMode::HitLighting] {
            r.set_flux_settings(FluxSettings {ray_lighting_mode:mode,..r.flux_settings});
            let lit=samples(r,&input,16)?;
            if mode==RayLightingMode::SurfaceCache {colored(lit,[0,1],"offscreen cached texture lighting");}
            else {
                // Hit lighting adds a neutral dielectric specular lobe to the colored diffuse lobe.
                // Material-average fallback would make both channels bright, unlike this result.
                for view in lit {for (side, side_value) in view.iter().enumerate() {for (c, c_value) in side_value.iter().enumerate() {
                    assert!(if c==side {*c_value>180.0} else {*c_value<60.0},"hit BRDF texture response {mode:?}: {lit:?}");
                }}}
            }
        }
        input.lights=&[];r.material_params_mut(emitter).base_color=[0.0,0.0,0.0,1.0];r.material_params_mut(emitter).emission_intensity=1.0;
        slots[0]=None;slots[3]=Some(movie);r.set_material_textures(emitter,&slots);input.camera_targets=&feeds;
        let far_revision=r.texture_revisions[&far_movie.0];
        for (color,channel) in [([1.0,0.0,0.0,1.0],0),([0.0,1.0,0.0,1.0],1)] {
            input.clear_color=color;colored(samples(r,&input,12)?,[channel;2],"live offscreen reflection texture");
            assert_eq!(r.texture_revisions[&far_movie.0],far_revision,"unrelated far video feed updated");
        }
        input.clear_color=[0.0;4];let dark=samples(r,&input,12)?;
        assert!(dark.iter().flatten().flatten().all(|v|*v<3.0),"stale emissive reflection after video light-off: {dark:?}");
        // Keep the last frame paused; it must stop recapturing when its content stops changing.
        input.camera_targets=&[];samples(r,&input,8)?;assert_eq!(r.flux_surface.as_ref().unwrap().captures,0);
        // Reflection and GI controls both off: no surface update or offscreen feed work.
        let updates=r.reflection_surface.recorded;let revision=r.texture_revisions[&movie.0];
        r.set_flux_settings(FluxSettings {reflection_mode:0,..r.flux_settings});input.camera_targets=&feeds;samples(r,&input,4)?;
        assert_eq!(r.reflection_surface.recorded,updates);assert_eq!(r.texture_revisions[&movie.0],revision);
        input.camera_targets=&[];slots[3]=Some(texture);r.set_material_textures(emitter,&slots);
        r.set_flux_settings(FluxSettings {reflection_mode:2,ray_lighting_mode:RayLightingMode::SurfaceCache,..r.flux_settings});
        colored(samples(r,&input,6)?,[0,1],"reactivated reflection captures");
        // GI toggle changes the owner of the shared surface pass without stale radiance or double updates.
        if !r.gpu_gi.as_ref().unwrap().has_gdf() {r.gi_set_gdf(&[100.0;512],&[0;512],[8;3],[-10.0;3],[10.0;3],&[neutral]);}
        r.gi_set_mesh_sdfs(&fields)?;
        let updates=r.reflection_surface.recorded;r.set_gi_enabled(true);
        colored(samples(r,&input,8)?,[0,1],"GI owns surface pass");assert_eq!(r.reflection_surface.recorded,updates);
        r.set_gi_enabled(false);colored(samples(r,&input,8)?,[0,1],"standalone takes surface pass back");
        eprintln!("[Standalone reflection surface] hardware={hardware}, initialGDF={had_gdf}, textured={split:?}; live/paused/distant feeds, hit lighting, GI toggle and one-update-per-frame passed");
    }
    r.gi_set_mesh_sdfs(&[])?;r.gi_set_emitters(&[]);r.set_gi_enabled(old_gi);r.set_flux_settings(old_settings);
    Ok(())
}
