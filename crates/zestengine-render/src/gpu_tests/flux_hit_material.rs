//! Compare standard material features with equivalent flattened materials.
//! Both the visible source and its real ray-traced reflection must agree;
//! atlas readback checks that software GI captures the same material response.
use glam::{Mat4,Quat,Vec3};
use crate::{CameraData,FluxSettings,GpuGiMaterial,MaterialDesc,MaterialParams,RayLightingMode,Renderer,TextureData,TextureHandle};
use super::planar::{cube,draw,lit};
fn base()->MaterialDesc {
    MaterialDesc {params:MaterialParams {base_color:[0.4,0.3,0.2,1.0],metallic:0.2,roughness:0.6,
        emission_intensity:1.0,emission_color:[0.0;3],..Default::default()},
        features:crate::MaterialFeatures {normal_map:true,emission:true,..Default::default()},..Default::default()}
}
fn texture(r:&mut Renderer,rgba:[u8;4])->anyhow::Result<TextureHandle> {
    r.upload_texture(&TextureData {width:1,height:1,pixels:rgba.to_vec(),srgb:false,hdr:false})
}
struct Case {name:&'static str,variant:MaterialDesc,reference:MaterialDesc,time:f32,instance:Option<[f32;4]>}
impl Case {fn new(name:&'static str)->Self {Self {name,variant:base(),reference:base(),time:0.0,instance:None}}}
fn patch(image:&(u32,u32,Vec<u8>),cam:CameraData,p:Vec3)->[f32;3] {
    let (w,h,pixels)=image;let p=(cam.proj*cam.view).project_point3(p);
    let x=((p.x*0.5+0.5)*(*w as f32)) as usize;let y=((0.5-p.y*0.5)*(*h as f32)) as usize;
    let mut sum=[0.0;3];for py in y-2..=y+2 {for px in x-2..=x+2 {for c in 0..3 {sum[c]+=pixels[(py*(*w as usize)+px)*4+c] as f32/25.0;}}}sum
}
pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    if !r.ctx.ray_tracing() {return Ok(());}
    let mut cases=Vec::new();
    let mut c=Case::new("split ORM");c.variant.params.metallic=1.0;c.variant.params.roughness=1.0;
    c.variant.orm=Some(texture(r,[255,128,255,255])?);c.variant.metallic=Some(texture(r,[0,0,0,255])?);
    c.variant.roughness=Some(texture(r,[128,128,128,255])?);
    c.reference.params.metallic=0.0;c.reference.params.roughness=(128.0f32/255.0).powi(2);cases.push(c);
    let mut c=Case::new("inverted split maps");c.variant.params.metallic=0.8;c.variant.params.roughness=0.9;
    c.variant.params.metallic_invert=true;c.variant.params.roughness_invert=true;
    c.variant.orm=Some(texture(r,[255,200,180,255])?);c.variant.metallic=Some(texture(r,[128;4])?);c.variant.roughness=Some(texture(r,[64;4])?);
    c.reference.params.metallic=0.8*(180.0/255.0)*(1.0-128.0/255.0);c.reference.params.roughness=0.9*(200.0/255.0)*(1.0-64.0/255.0);cases.push(c);
    let mut c=Case::new("instance surface parameters");c.variant.features.per_instance=true;c.variant.params.metallic=1.0;c.variant.params.roughness=1.0;
    c.instance=Some([0.0,0.3,0.0,0.0]);c.reference.params.metallic=0.0;c.reference.params.roughness=0.3;cases.push(c);
    for mode in 0..4 {
        let mut c=Case::new(["detail multiply","detail overlay","detail add","detail replace"][mode]);
        c.variant.detail[0]=Some(texture(r,[30,120,200,255])?);c.variant.detail[3]=Some(texture(r,[128;4])?);
        c.variant.params.detail.albedo_strength=0.7;c.variant.params.detail.blend=mode as f32;
        for (i,d) in [30.0/255.0,120.0/255.0,200.0/255.0].iter().enumerate() {
            let a=c.reference.params.base_color[i];let b=match mode {0=>a*d*4.594794,1=>if a<0.5 {2.0*a*d} else {1.0-2.0*(1.0-a)*(1.0-d)},2=>a+d-0.2140,_=>*d};
            c.reference.params.base_color[i]=(a+(b-a)*0.7*(128.0/255.0)).max(0.0);
        }
        cases.push(c);
    }
    let mut c=Case::new("detail ORM");c.variant.detail[2]=Some(texture(r,[255,64,128,255])?);
    c.variant.detail[3]=Some(texture(r,[128;4])?);c.variant.params.detail.orm_strength=0.8;
    let strength=0.8*128.0/255.0;c.reference.params.roughness*=1.0+(64.0/255.0-1.0)*strength;
    c.reference.params.metallic*=1.0+(128.0/255.0-1.0)*strength;cases.push(c);
    let mut c=Case::new("BC5 normal reconstruction");c.variant.normal=Some(texture(r,[180,128,0,255])?);
    let x=180.0f32/255.0*2.0-1.0;let y=128.0f32/255.0*2.0-1.0;
    let z=((1.0-x*x-y*y).sqrt()*0.5+0.5)*255.0;
    c.reference.normal=Some(texture(r,[180,128,z.round() as u8,255])?);cases.push(c);
    let mut c=Case::new("detail normal slopes");c.variant.detail[1]=Some(texture(r,[180,128,255,255])?);
    c.variant.params.detail.normal_strength=1.0;c.reference.normal=c.variant.detail[1];cases.push(c);
    let mut c=Case::new("masked emission");c.variant.params.emission_color=[1.5,0.5,0.25];
    c.variant.emission_mask=Some(texture(r,[64;4])?);c.reference.params.emission_color=c.variant.params.emission_color.map(|v|v*64.0/255.0);cases.push(c);
    let mut c=Case::new("disabled emission");c.variant.params.emission_color=[4.0;3];c.variant.features.emission=false;cases.push(c);
    for (name,time,value) in [("emission pulse peak",std::f32::consts::FRAC_PI_2,1.0),("emission pulse trough",3.0*std::f32::consts::FRAC_PI_2,0.0)] {
        let mut c=Case::new(name);c.time=time;c.variant.params.emission_color=[0.2,0.5,1.0];c.variant.params.emission_pulse=1.0;
        c.reference.params.emission_color=c.variant.params.emission_color.map(|v|v*value);cases.push(c);
    }
    let mut c=Case::new("scrolling emission with stationary mask");c.time=0.5;c.variant.params.emission_color=[1.0;3];
    c.variant.emission=Some(r.upload_texture(&TextureData {width:2,height:1,pixels:vec![255,0,0,255,0,255,0,255],srgb:false,hdr:false})?);
    c.variant.emission_mask=Some(r.upload_texture(&TextureData {width:2,height:1,pixels:vec![255;4].into_iter().chain([0,0,0,255]).collect(),srgb:false,hdr:false})?);
    c.variant.params.emission_offset=[0.25,0.0];c.variant.params.emission_scroll=[1.0,0.0];c.reference.params.emission_color=[0.0,1.0,0.0];cases.push(c);

    // Constant UVs eliminate mip/filter uncertainty in this material-contract
    // fixture. The scrolling case samples exact texel centres with its offset.
    let mesh=r.upload_mesh(&cube())?;let mirror=lit(r,[0.95;4],[0.0;3],0.02,1.0);
    let transforms=[Mat4::from_scale_rotation_translation(Vec3::new(8.0,0.2,8.0),Quat::IDENTITY,Vec3::new(0.0,-0.1,0.0)),
        Mat4::from_scale_rotation_translation(Vec3::new(2.0,3.0,0.2),Quat::IDENTITY,Vec3::new(0.0,1.5,-2.0))];
    let fields:Vec<_>=transforms.iter().enumerate().map(|(i,t)|crate::GiMeshSdf {gi_source:Some(crate::GiSource {object:0xdf00+i as u128,slot:0}),
        world_to_local:t.inverse(),volume:super::flux::cube_distance_field(),in_global_field:false,
        material:GpuGiMaterial {albedo:[0.4;3],emission:[0.0;3],metallic:if i==0 {1.0} else {0.2},roughness:0.6}}).collect();
    r.gi_set_mesh_sdfs(&fields)?;r.gi_set_gdf(&vec![100.0;512],&vec![0;512],[8;3],[-10.0;3],[10.0;3],&[fields[0].material]);
    r.gi_set_emitters(&[]);r.set_gi_enabled(true);
    let eye=Vec3::new(0.0,2.0,4.0);let cam=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let lights=[crate::LightInstance {kind:crate::LightKind::Point,position:Vec3::new(0.0,1.5,-1.0),direction:Vec3::NEG_Z,
        color:[1.0;3],intensity:12.0,range:5.0,spot_inner_deg:30.0,spot_outer_deg:45.0,cast_shadows:true,soft_shadows:false,radius:0.0,shadow_bias:0.0,baked:false}];
    for case in cases {
        let handles=[r.create_material(&case.variant)?,r.create_material(&case.reference)?];
        for hardware in [false,true] {
            r.set_flux_settings(FluxSettings {rt_trace:hardware,samples:16,bounces:1,reflection_mode:2,rt_samples:1,rt_bounces:1,
                ray_lighting_mode:RayLightingMode::HitLighting,ssr_enabled:false,..Default::default()});
            let mut results=Vec::new();let mut materials=Vec::new();
            for (i,handle) in handles.iter().enumerate() {
                let mut draws=[draw(mesh,mirror,transforms[0],-1),draw(mesh,*handle,transforms[1],-1)];
                for (j,d) in draws.iter_mut().enumerate() {d.gi_source=fields[j].gi_source;}
                if i==0 {draws[1].instance_param=case.instance;}
                let mut input=crate::cascade_tests::input(cam);input.camera_preview=Some(cam);input.viewport_extent=Some([640,360]);input.draws=&draws;input.lights=&lights;
                input.time=case.time;input.light.intensity=0.0;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;input.draw_skybox=false;
                input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
                for _ in 0..24 {r.render(&input)?;}
                let scene=r.debug_read_viewport()?.unwrap();let camera=r.debug_read_camera_preview()?.unwrap();
                results.push([patch(&scene,cam,Vec3::new(0.0,1.5,-1.9)),patch(&scene,cam,Vec3::ZERO),
                    patch(&camera,cam,Vec3::new(0.0,1.5,-1.9)),patch(&camera,cam,Vec3::ZERO)]);
                materials.push({let allocator=r.allocator();let mut alloc=allocator.lock().unwrap();
                    r.flux_surface.as_ref().unwrap().debug_lighting_receiver(&r.ctx.device,&mut alloc,r.command_pool,r.ctx.queue,handle.0,4)?});
            }
            let error=results[0].iter().flatten().zip(results[1].iter().flatten()).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
            let cache_error=materials[0][1..].iter().flatten().zip(materials[1][1..].iter().flatten()).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
            eprintln!("[Flux hit material] {} hardware={hardware}: render error={error}, cache error={cache_error}, variant={:?}, reference={:?}",case.name,results[0],results[1]);
            anyhow::ensure!(error<4.0 && cache_error<0.012,"material response mismatch: {} hardware={hardware}, render={error}, cache={cache_error}",case.name);
            anyhow::ensure!(results[1][1].iter().copied().fold(0.0f32,f32::max)>10.0,"reflection fixture has no signal");
        }
    }
    Ok(())
}
