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
    run_backend(r,true,false)
}
pub(super) fn software(r:&mut Renderer)->anyhow::Result<()> {run_backend(r,false,false)}
pub(super) fn mixed_field(r:&mut Renderer)->anyhow::Result<()> {run_backend(r,false,true)}
fn run_backend(r:&mut Renderer,hardware:bool,mixed_field:bool)->anyhow::Result<()> {
    r.set_gi_enabled(false);r.gi_set_emitters(&[]);
    r.set_flux_settings(FluxSettings {rt_trace:hardware,reflection_mode:2,rt_samples:1,rt_bounces:1,ssr_enabled:false,..Default::default()});
    let mesh=r.upload_mesh(&cube())?;
    let mirror=lit(r,[0.95;4],[0.0;3],0.02,1.0);
    let emitter=lit(r,[1.0;4],[2.0;3],1.0,0.0);
    let mask=lit(r,[0.0,0.0,0.0,1.0],[0.0;3],1.0,0.0);
    let mut vertices=Vec::new();for (x,y,u,v) in [(-4.0,0.0,0.0,0.0),(4.0,0.0,1.0,0.0),(4.0,4.0,1.0,1.0),(-4.0,4.0,0.0,1.0)] {
        vertices.push(crate::Vertex {position:[x,y,-1.5],normal:[0.0,0.0,1.0],uv:[u,v],color:[1.0;4],..Default::default()});
    }
    let panel=r.upload_mesh(&crate::MeshData {vertices:vertices.clone(),indices:vec![0,1,2,0,2,3],has_lightmap_uv:false})?;
    let reverse=r.upload_mesh(&crate::MeshData {vertices:vertices.clone(),indices:vec![0,2,1,0,3,2],has_lightmap_uv:false})?;
    let software_fields=(!hardware).then(|| {
        let points:Vec<_>=vertices.iter().map(|v|Vec3::from_array(v.position)).collect();
        [super::flux::cube_distance_field(),
            std::sync::Arc::new(crate::sdf::generate_sdf(&points,&[0,1,2,0,2,3],32,0.1)),
            std::sync::Arc::new(crate::sdf::generate_sdf(&points,&[0,2,1,0,3,2],32,0.1))]
    });
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
    if !hardware {
        for (i,d) in draws.iter_mut().enumerate() {d.gi_source=Some(crate::GiSource {object:0xc170+i as u128,slot:0});}
        r.gi_set_gdf_identified(&[100.0;8],&[0;8],[2;3],[-10.0;3],[10.0;3],&[],&[])?;
    }
    let mut slots=[None;16];slots[0]=Some(texture);r.set_material_textures(mask,&slots);
    let make_features=|alpha_mode|MaterialFeatures {alpha_mode,..Default::default()};
    r.set_material_features(mask,make_features(AlphaMode::Cutout));r.material_params_mut(mask).alpha_cutoff=0.5;
    let light=[crate::LightInstance {kind:crate::LightKind::Directional,position:Vec3::ZERO,direction:Vec3::NEG_Z,color:[1.0;3],intensity:8.0,
        range:0.0,spot_inner_deg:30.0,spot_outer_deg:45.0,cast_shadows:true,soft_shadows:false,radius:0.0,shadow_bias:0.0,baked:false}];
    for case in 0..if mixed_field {2} else {15} {
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
            // Hidden casters must not block the reflected emissive source.
            11=>{draws[2].mesh=panel;r.material_params_mut(emitter).emission_intensity=1.0;},
            // Raise the panel above the primary reflection ray while keeping
            // it across the wall's directional-light shadow ray. This isolates
            // casting from primary visibility instead of letting either one
            // produce the same dark half of the image.
            12=>{draws[2].transform=Mat4::from_translation(Vec3::new(0.0,1.2,0.0));
                r.material_params_mut(emitter).emission_intensity=0.0;},
            13=>{draws[2].shadow_casting=ShadowCasting::Off;},
            14=>{draws[2].shadow_casting=ShadowCasting::On;},
            _=>{},
        }
        // UV transforms/scroll live in the raster material UBO. Follow the
        // same edit+upload contract as the editor; RT instance records alone
        // read the CPU params and would hide a stale capture-side fixture.
        r.upload_material_fx(mask);
        if mixed_field && case==1 {draws[2].transform=Mat4::from_translation(Vec3::new(0.0,0.0,-0.8));}
        if mixed_field {
            // A captured masked panel shares a coarse field with an unmapped
            // occluder. Neither "trust every coarse hit" nor "drop the entire
            // coarse field" is correct for a partially streamed mesh snapshot.
            let dims=[81,49,81];let lo=Vec3::new(-5.0,-1.0,-4.0);let hi=Vec3::new(5.0,5.0,1.0);
            let mut distances=Vec::new();let mut indices=Vec::new();
            for z in 0..dims[2] {for y in 0..dims[1] {for x in 0..dims[0] {
                let p=lo+(hi-lo)*Vec3::new(x as f32,y as f32,z as f32)/Vec3::from_array(dims.map(|n|(n-1) as f32));
                let distance=|center:Vec3,half:Vec3| {let q=(p-center).abs()-half;q.max(Vec3::ZERO).length()+q.max_element().min(0.0)};
                let pane=distance(draws[2].transform.transform_point3(Vec3::new(0.0,2.0,-1.5)),Vec3::new(4.0,2.0,0.02));
                let unknown=distance(Vec3::new(-2.65,1.0,-2.25),Vec3::new(0.35,0.8,0.08));
                distances.push(pane.min(unknown));indices.push(if pane<unknown {0} else {1});
            }}}
            let dark=crate::GpuGiMaterial {albedo:[0.0;3],emission:[0.0;3],metallic:0.0,roughness:1.0};
            r.gi_set_gdf_identified(&distances,&indices,dims,lo.to_array(),hi.to_array(),&[dark,dark],
                &[draws[2].gi_source,Some(crate::GiSource {object:0xc173,slot:0})])?;
        }
        if let Some(volumes)=&software_fields {
            let fields:Vec<_>=draws.iter().enumerate().map(|(i,d)|crate::GiMeshSdf {
                gi_source:d.gi_source,world_to_local:d.transform.inverse(),in_global_field:mixed_field && i==2,
                volume:volumes[if i==2 {if d.mesh==reverse {2} else {1}} else {0}].clone(),
                material:crate::GpuGiMaterial {albedo:match i {0=>[0.95;3],1=>[1.0;3],_=>[0.0;3]},
                    emission:if i==1 && (case<9 || case==11) {[2.0;3]} else {[0.0;3]},metallic:if i==0 {1.0} else {0.0},roughness:if i==0 {0.02} else {1.0}},
            }).collect();
            r.gi_set_mesh_sdfs(&fields)?;
            r.set_gi_enabled(false);
        }
        let mut input=crate::cascade_tests::input(cam);input.camera_preview=Some(cam);input.viewport_extent=Some([640,360]);input.draws=&draws;
        input.light.intensity=0.0;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;input.draw_skybox=false;
        input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;input.time=1.0;
        if case>=9 && case!=11 {input.lights=&light;}
        for screen in [false,true] {
            // The unmapped procedural occluder intentionally has no raster
            // draw; this test isolates world tracing, not screen visibility.
            if mixed_field && screen {continue;}
            // Both raster screen lighting and world-hit lighting must respect cutouts.
            input.postfx.flux.reflection_screen_traces=Some(screen);
            input.preview_postfx.flux.reflection_screen_traces=Some(screen);
        for _ in 0..if hardware {4} else {48} {r.render(&input)?;}
        let result=values(r,cam)?;eprintln!("[Flux reflection alpha] hardware={hardware} screen={screen} case={case}, Scene/Camera={result:?}");
        let expected=match case {2=>[false,false],4|5=>[false,true],7|11|13=>[true,true],_=>[true,false]};
        for view in result {for side in 0..2 {
            assert!(if expected[side] {view[side]>200.0} else {view[side]<5.0},"alpha reflection case {case}: {result:?}, expected bright {expected:?}");
        }}
        if mixed_field {
            for image in [r.debug_read_viewport()?.unwrap(),r.debug_read_camera_preview()?.unwrap()] {
                let unknown=patch(&image,cam,Vec3::new(-1.7,0.0,0.0));
                assert!(unknown<5.0,"unmapped coarse-field occluder disappeared: {unknown}");
            }
        }
        }
    }
    Ok(())
}
