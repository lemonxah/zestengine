//! Exact reflected direct light must survive below the surface-lighting texel
//! spacing. Scene and Camera choose opposite modes without changing the scene.
use glam::{Mat4,Quat,Vec3};
use crate::{CameraData,FluxSettings,GpuGiMaterial,RayLightingMode as Mode,Renderer};
use super::planar::{cube,draw,lit};
pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    if !r.ctx.ray_tracing() {return Ok(());}
    let mesh=r.upload_mesh(&cube())?;
    let mirror=lit(r,[0.95;4],[0.0;3],0.02,1.0);
    let wall=lit(r,[0.8;4],[0.0;3],1.0,0.0);
    let mut draws=[draw(mesh,mirror,Mat4::from_scale_rotation_translation(Vec3::new(8.0,0.2,8.0),Quat::IDENTITY,Vec3::new(0.0,-0.1,0.0)),-1),
        draw(mesh,wall,Mat4::from_scale_rotation_translation(Vec3::new(2.0,3.0,0.2),Quat::IDENTITY,Vec3::new(0.0,1.5,-2.0)),-1)];
    for (i,d) in draws.iter_mut().enumerate() {d.gi_source=Some(crate::GiSource{object:0xdd00+i as u128,slot:0});}
    let volume=super::flux::cube_distance_field();
    let fields:Vec<_>=draws.iter().enumerate().map(|(i,d)|crate::GiMeshSdf {gi_source:d.gi_source,world_to_local:d.transform.inverse(),volume:volume.clone(),in_global_field:false,
        material:GpuGiMaterial {albedo:if i==0 {[0.95;3]} else {[0.8;3]},emission:[0.0;3],metallic:if i==0 {1.0} else {0.0},roughness:if i==0 {0.02} else {1.0}}}).collect();
    r.gi_set_mesh_sdfs(&fields)?;r.gi_set_gdf(&vec![100.0;512],&vec![0;512],[8;3],[-10.0;3],[10.0;3],&[fields[0].material]);
    r.gi_set_emitters(&[]);r.set_gi_enabled(true);
    r.set_flux_settings(FluxSettings {rt_trace:true,samples:16,bounces:1,reflection_mode:2,rt_samples:1,rt_bounces:1,ssr_enabled:false,..Default::default()});
    let eye=Vec3::new(0.0,2.0,4.0);let cam=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let lights=[crate::LightInstance {kind:crate::LightKind::Point,position:Vec3::new(0.0,0.95,-1.88),direction:Vec3::NEG_Z,color:[1.0;3],intensity:8.0,
        range:0.08,spot_inner_deg:30.0,spot_outer_deg:45.0,cast_shadows:true,soft_shadows:false,radius:0.0,shadow_bias:0.0,baked:false}];
    let mut input=crate::cascade_tests::input(cam);input.camera_preview=Some(cam);input.viewport_extent=Some([640,360]);input.draws=&draws;input.lights=&lights;
    input.light.intensity=0.0;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;input.draw_skybox=false;
    input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
    let patch=|image:&(u32,u32,Vec<u8>)| {
        let (w,h,pixels)=image;let p=(cam.proj*cam.view).project_point3(Vec3::ZERO);
        let x=((p.x*0.5+0.5)*(*w as f32)) as usize;let y=((0.5-p.y*0.5)*(*h as f32)) as usize;
        let mut sum=0.0;for py in y-1..=y+1 {for px in x-1..=x+1 {sum+=pixels[(py*(*w as usize)+px)*4] as f32;}}sum/9.0
    };
    for swap in [false,true] {
        input.postfx.flux.ray_lighting_mode=Some(if swap {Mode::HitLightingForReflections} else {Mode::SurfaceCache});
        input.preview_postfx.flux.ray_lighting_mode=Some(if swap {Mode::SurfaceCache} else {Mode::HitLightingForReflections});
        for _ in 0..32 {r.render(&input)?;}
        let scene=patch(&r.debug_read_viewport()?.unwrap());let camera=patch(&r.debug_read_camera_preview()?.unwrap());
        let (hit,cached)=if swap {(scene,camera)} else {(camera,scene)};
        eprintln!("[Flux ray lighting direct] swapped={swap}: hit={hit}, cached={cached}");
        anyhow::ensure!(hit>cached+15.0,"hit lighting did not resolve sub-cache direct light independently: hit={hit}, cache={cached}");
    }
    // A metal has no diffuse response: a lit reflection here specifically
    // proves that hit lighting evaluates direct specular, not only Lambertian.
    r.material_params_mut(wall).metallic=1.0;
    r.material_params_mut(wall).roughness=0.5;
    input.postfx.flux.ray_lighting_mode=Some(Mode::SurfaceCache);
    input.preview_postfx.flux.ray_lighting_mode=Some(Mode::HitLightingForReflections);
    for _ in 0..32 {r.render(&input)?;}
    let metal_hit=patch(&r.debug_read_camera_preview()?.unwrap());
    let metal_cache=patch(&r.debug_read_viewport()?.unwrap());
    eprintln!("[Flux ray lighting specular] hit={metal_hit}, cached={metal_cache}");
    anyhow::ensure!(metal_hit>metal_cache+10.0,"hit lighting lost direct metallic highlight: {metal_hit}, {metal_cache}");
    Ok(())
}
