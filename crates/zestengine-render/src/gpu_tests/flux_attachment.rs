//! Identified moving objects keep their captures, while lookup and lighting
//! operate at the current world position and orientation in both GI backends.
use glam::{Mat4,Quat,Vec3};
use crate::{CameraData,GiSource,GpuGiMaterial,Renderer};
use super::planar::{cube,draw,lit};

fn patch(image:&(u32,u32,Vec<u8>),camera:CameraData,point:Vec3)->f32 {
    let (w,h,pixels)=image;let p=(camera.proj*camera.view).project_point3(point);
    let x=((p.x*0.5+0.5)* *w as f32) as i32;let y=((0.5-p.y*0.5)* *h as f32) as i32;
    let mut sum=0.0;
    for j in -2..=2 {for i in -2..=2 {sum+=pixels[(((y+j) as u32*w+(x+i) as u32)*4) as usize] as f32;}}
    sum/25.0
}
pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    let mesh=r.upload_mesh(&cube())?;
    let volume=super::flux::cube_distance_field();
    let receiver=lit(r,[0.7;4],[0.0;3],1.0,0.0);
    let source=lit(r,[0.0,0.0,0.0,1.0],[6.0,1.0,0.2],1.0,0.0);
    r.set_material_features(source,crate::MaterialFeatures {emission:true,per_instance:true,..Default::default()});
    r.material_params_mut(source).emission_map_mean=[0.0;3];
    let identities=[GiSource {object:0x1000,slot:0},GiSource {object:0x2000,slot:0}];
    let floor=draw(mesh,receiver,Mat4::from_scale_rotation_translation(Vec3::new(6.0,0.2,6.0),Quat::IDENTITY,Vec3::new(0.0,-0.1,0.0)),-1);
    let initial=Mat4::from_scale_rotation_translation(Vec3::new(1.2,1.0,0.3),Quat::IDENTITY,Vec3::new(-1.0,1.5,0.0));
    let mut objects=[floor,draw(mesh,source,initial,-1)];
    for i in 0..2 {objects[i].gi_source=Some(identities[i]);}
    let sync_fields=|r:&mut Renderer,objects:&[crate::DrawCmd]|->anyhow::Result<()> {
        let fields:Vec<_>=objects.iter().map(|d|crate::GiMeshSdf {gi_source:d.gi_source,world_to_local:d.transform.inverse(),
            volume:volume.clone(),in_global_field:false,
            // Black emitter fallback proves that emission comes from its
            // attached material capture, not mean-material hit shading.
            material:GpuGiMaterial {albedo:if d.material==source {[0.0;3]} else {[0.7;3]},emission:[0.0;3],metallic:0.0,roughness:1.0}}).collect();
        r.gi_set_mesh_sdfs(&fields)
    };
    let eye=Vec3::new(0.0,2.5,5.0);
    let camera=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    for rt in [false,true] {
        if rt && !r.ctx.ray_tracing() {continue;}
        r.material_params_mut(source).emission_color=[6.0,1.0,0.2];r.material_params_mut(source).base_color=[0.0,0.0,0.0,1.0];
        objects[1].transform=initial;objects[1].instance_color=None;
        let mut input=crate::cascade_tests::input(camera);input.camera_preview=Some(camera);input.viewport_extent=Some([323,181]);
        input.light.ambient=[0.0;3];input.light.intensity=0.0;input.light.env_intensity=0.0;input.gi_debug=6;
        input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;

        r.gi_set_gdf(&vec![100.0;512],&vec![0;512],[8;3],[-30.0;3],[30.0;3],
            &[GpuGiMaterial {albedo:[0.0;3],emission:[0.0;3],metallic:0.0,roughness:1.0}]);
        r.gi_set_emitters(&[]);sync_fields(r,&objects)?;
        r.set_flux_settings(crate::FluxSettings {rt_trace:rt,samples:64,bounces:2,reflection_mode:0,ssr_enabled:false,..Default::default()});r.set_gi_enabled(true);
        input.draws=&objects;
        for _ in 0..24 {r.render(&input)?;}
        let first=r.flux_surface.as_ref().unwrap().debug_card(source.0,4);
        let mut minimum=[255.0f32;2];
        let motion_frames:Vec<_>=(0..24).map(|frame| {
            let mut moved=objects;
            let angle=frame as f32*0.17;
            moved[1].transform=Mat4::from_translation(Vec3::new(angle.sin()*0.4,0.0,0.0))
                *Mat4::from_translation(Vec3::new(-1.0,1.5,0.0))
                *Mat4::from_quat(Quat::from_rotation_y(angle))
                *Mat4::from_scale(Vec3::splat(1.0+0.1*angle.sin()))
                *Mat4::from_scale(Vec3::new(1.2,1.0,0.3));
            moved
        }).collect();
        for moved in &motion_frames {
            sync_fields(r,moved)?;input.draws=moved;r.render(&input)?;
            let cache=r.flux_surface.as_ref().unwrap();let now=cache.debug_card(source.0,4);
            assert_eq!(cache.captures,0,"rigid motion recaptured material");
            assert_eq!((now.0,now.1),(first.0,first.1),"motion replaced card identity/capture");
            assert_eq!(now.2.rect[..3],first.2.rect[..3],"motion moved atlas slot");
            let point=Vec3::new(-0.7,0.0,1.0);
            let scene=patch(&r.debug_read_viewport()?.unwrap(),camera,point);
            let preview=patch(&r.debug_read_camera_preview()?.unwrap(),camera,point);
            minimum[0]=minimum[0].min(scene);minimum[1]=minimum[1].min(preview);
        }
        assert!(minimum.iter().all(|v|*v>8.0),"moving cached emitter lost GI: {minimum:?}");
        // Reorder draws without changing the SDF snapshot's source order.
        let last=motion_frames.last().unwrap();
        let reversed=[last[1],last[0]];input.draws=&reversed;r.render(&input)?;
        assert_eq!(r.flux_surface.as_ref().unwrap().captures,0);
        assert_eq!(r.flux_surface.as_ref().unwrap().debug_card(source.0,4).0,first.0);
        eprintln!("[Flux attached emitter] RT={rt}: minimum Scene/Camera={minimum:?}, 24 moving frames without recapture");
        // A nonuniform shape change needs a recapture but retains its pages.
        let mut stretched=objects;stretched[1].transform=Mat4::from_translation(Vec3::new(0.0,1.0,0.0));
        sync_fields(r,&stretched)?;input.draws=&stretched;r.render(&input)?;
        let changed=r.flux_surface.as_ref().unwrap().debug_card(source.0,4);
        assert_eq!(changed.0,first.0);assert!(changed.1>first.1);
        assert_eq!(r.flux_surface.as_ref().unwrap().captures,6,"nonuniform stretch failed to refresh tangent frame");
        // Per-instance edits refresh content without replacing page identity.
        let mut tinted=stretched;tinted[1].instance_color=Some([0.7,0.7,0.7,1.0]);input.draws=&tinted;
        r.material_params_mut(source).emission_color=[0.0;3];
        let light=[crate::LightInstance {kind:crate::LightKind::Point,position:Vec3::new(0.0,1.0,3.0),direction:Vec3::NEG_Z,
            color:[1.0;3],intensity:10.0,range:6.0,spot_inner_deg:30.0,spot_outer_deg:45.0,cast_shadows:false,soft_shadows:false,
            radius:0.0,shadow_bias:0.0,baked:false}];input.lights=&light;
        for _ in 0..8 {r.render(&input)?;}
        let cache=r.flux_surface.as_ref().unwrap();let refreshed=cache.debug_card(source.0,4);
        assert_eq!(refreshed.0,first.0);assert!(refreshed.1>changed.1);
        let direct=|r:&Renderer|->anyhow::Result<f32> {
            let alloc=r.allocator();Ok(r.flux_surface.as_ref().unwrap().debug_lighting(&r.ctx.device,&mut alloc.lock().unwrap(),r.command_pool,r.ctx.queue,source.0,4)?[0][0])
        };
        let lit=direct(r)?;assert!(lit>0.01);
        let mut rotated_objects=tinted;rotated_objects[1].transform=Mat4::from_translation(Vec3::new(0.0,1.0,0.0))*Mat4::from_rotation_y(std::f32::consts::PI);
        sync_fields(r,&rotated_objects)?;input.draws=&rotated_objects;r.render(&input)?;
        let rotated=direct(r)?;assert!(rotated<lit*0.01,"lighting ignored rotated capture normal: {lit}->{rotated}");
        assert_eq!(r.flux_surface.as_ref().unwrap().captures,0);
        let mut translated_objects=tinted;translated_objects[1].transform=Mat4::from_translation(Vec3::new(20.0,1.0,0.0));
        sync_fields(r,&translated_objects)?;input.draws=&translated_objects;r.render(&input)?;
        let translated=direct(r)?;assert!(translated<lit*0.01,"lighting used old capture position: {lit}->{translated}");
        assert_eq!(r.flux_surface.as_ref().unwrap().captures,0);
        eprintln!("[Flux attached lighting] RT={rt}: facing={lit}, rotated={rotated}, translated={translated}");
        input.draws=&[];input.lights=&[];r.render(&input)?;
    }
    Ok(())
}
