//! Alpha-tested reflection hits and reflected-hit shadow rays, through real
//! Scene/Camera dispatches. Opaque alpha-zero materials must still occlude.
use glam::{Mat4,Quat,Vec3};
use crate::{AlphaMode,CameraData,FluxSettings,MaterialFeatures,Renderer,ShadowCasting};
use super::planar::{cube,draw,lit};
fn patch(image:&(u32,u32,Vec<u8>),cam:CameraData,point:Vec3)->f32 {
    let (w,h,data)=image;let p=(cam.proj*cam.view).project_point3(point);
    let x=((p.x*0.5+0.5)* *w as f32) as i32;let y=((0.5-p.y*0.5)* *h as f32) as i32;
    let mut sum=0.0;for j in -2..=2 {for i in -2..=2 {sum+=data[(((y+j) as u32*w+(x+i) as u32)*4) as usize] as f32;}}sum/25.0
}
fn values(r:&mut Renderer,cam:CameraData)->anyhow::Result<[[f32;2];2]> {
    let mut result=[[0.0;2];2];
    for (view,image) in [r.debug_read_viewport()?.unwrap(),r.debug_read_camera_preview()?.unwrap()].iter().enumerate() {
        for (side,x) in [-0.7,0.7].iter().enumerate() {result[view][side]=patch(image,cam,Vec3::new(*x,0.0,0.0));}
    }Ok(result)
}
pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    if !r.ctx.ray_tracing() {return Ok(());}
    r.set_gi_enabled(false);r.gi_set_emitters(&[]);
    r.set_flux_settings(FluxSettings {reflection_mode:2,rt_samples:1,rt_bounces:1,ssr_enabled:false,..Default::default()});
    let mesh=r.upload_mesh(&cube())?;
    let mirror=lit(r,[0.95;4],[0.0;3],0.02,1.0);
    let emitter=lit(r,[1.0;4],[2.0;3],1.0,0.0);
    let mask=lit(r,[0.0,0.0,0.0,1.0],[0.0;3],1.0,0.0);
    let mut vertices=Vec::new();for (x,y,u,v) in [(-4.0,0.0,0.0,0.0),(4.0,0.0,1.0,0.0),(4.0,4.0,1.0,1.0),(-4.0,4.0,0.0,1.0)] {
        vertices.push(crate::Vertex {position:[x,y,-1.5],normal:[0.0,0.0,1.0],uv:[u,v],color:[1.0;4],..Default::default()});
    }
    let panel=r.upload_mesh(&crate::MeshData {vertices:vertices.clone(),indices:vec![0,1,2,0,2,3],has_lightmap_uv:false})?;
    let reverse=r.upload_mesh(&crate::MeshData {vertices:vertices.clone(),indices:vec![0,2,1,0,3,2],has_lightmap_uv:false})?;
    for vertex in &mut vertices {vertex.color[3]=0.5;}
    let translucent_vertex=r.upload_mesh(&crate::MeshData {vertices,indices:vec![0,1,2,0,2,3],has_lightmap_uv:false})?;
    let texture=r.upload_texture(&crate::TextureData {width:32,height:1,
        pixels:(0..32).flat_map(|x|[255,255,255,if x<16 {0} else {255}]).collect(),srgb:false,hdr:false})?;
    let opacity=r.upload_texture(&crate::TextureData {width:32,height:1,
        pixels:(0..32).flat_map(|x|[if x<16 {0} else {255},255,255,255]).collect(),srgb:false,hdr:false})?;
    let eye=Vec3::new(0.0,2.0,4.0);let cam=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let mut draws=[draw(mesh,mirror,Mat4::from_scale_rotation_translation(Vec3::new(8.0,0.2,8.0),Quat::IDENTITY,Vec3::new(0.0,-0.1,0.0)),-1),
        draw(mesh,emitter,Mat4::from_scale_rotation_translation(Vec3::new(8.0,4.0,0.2),Quat::IDENTITY,Vec3::new(0.0,2.0,-3.0)),-1),
        draw(panel,mask,Mat4::IDENTITY,-1)];
    let mut slots=[None;16];slots[0]=Some(texture);r.set_material_textures(mask,&slots);
    let make_features=|alpha_mode|MaterialFeatures {alpha_mode,..Default::default()};
    r.set_material_features(mask,make_features(AlphaMode::Cutout));r.material_params_mut(mask).alpha_cutoff=0.5;
    let light=[crate::LightInstance {kind:crate::LightKind::Directional,position:Vec3::ZERO,direction:Vec3::NEG_Z,color:[1.0;3],intensity:8.0,
        range:0.0,spot_inner_deg:30.0,spot_outer_deg:45.0,cast_shadows:true,soft_shadows:false,radius:0.0,shadow_bias:0.0,baked:false}];
    for case in 0..11 {
        match case {
            1=>draws[2].mesh=reverse,
            2=>{r.set_material_features(mask,make_features(AlphaMode::Opaque));r.material_params_mut(mask).base_color[3]=0.0;},
            3=>{r.set_material_features(mask,make_features(AlphaMode::Cutout));r.material_params_mut(mask).base_color[3]=1.0;
                slots[0]=None;slots[4]=Some(opacity);r.set_material_textures(mask,&slots);},
            4=>{r.material_params_mut(mask).albedo_offset=[0.5,0.0];},
            5=>{r.material_params_mut(mask).albedo_offset=[0.0;2];r.material_params_mut(mask).base_scroll=[0.5,0.0];},
            6=>{r.material_params_mut(mask).base_scroll=[0.0;2];draws[2].mesh=translucent_vertex;
                r.material_params_mut(mask).base_color[3]=0.8;r.material_params_mut(mask).alpha_cutoff=0.3;},
            7=>{r.material_params_mut(mask).base_color[3]=0.4;},
            8=>{r.set_material_features(mask,MaterialFeatures {per_instance:true,..make_features(AlphaMode::Transclipping)});
                draws[2].instance_color=Some([0.0,0.0,0.0,0.8]);},
            9=>{draws[2].mesh=panel;draws[2].shadow_casting=ShadowCasting::ShadowsOnly;
                draws[2].instance_color=None;r.set_material_features(mask,make_features(AlphaMode::Cutout));
                r.material_params_mut(mask).base_color[3]=1.0;r.material_params_mut(mask).alpha_cutoff=0.5;
                r.material_params_mut(emitter).emission_intensity=0.0;},
            10=>{draws[2].mesh=reverse;},
            _=>{},
        }
        let mut input=crate::cascade_tests::input(cam);input.camera_preview=Some(cam);input.viewport_extent=Some([640,360]);input.draws=&draws;
        input.light.intensity=0.0;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;input.draw_skybox=false;
        input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;input.time=1.0;
        if case>=9 {input.lights=&light;}
        for _ in 0..4 {r.render(&input)?;}
        let result=values(r,cam)?;eprintln!("[Flux reflection alpha] case={case}, Scene/Camera={result:?}");
        let expected=match case {2=>[false,false],4|5=>[false,true],7=>[true,true],_=>[true,false]};
        for view in result {for side in 0..2 {
            assert!(if expected[side] {view[side]>200.0} else {view[side]<5.0},"alpha reflection case {case}: {result:?}, expected bright {expected:?}");
        }}
    }
    Ok(())
}
