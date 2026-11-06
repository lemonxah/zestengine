//! Physical surface-cache readback: realtime point/spot irradiance must use the
//! visible light's range and cone, in both software and hardware tracing.
use glam::{Mat4,Vec3};
use crate::{CameraData,FluxSettings,GiSource,GpuGiMaterial,LightInstance,LightKind,Renderer};
use super::planar::{cube,draw,lit};
pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    let mesh=r.upload_mesh(&cube())?;let material=lit(r,[0.8;4],[0.0;3],1.0,0.0);
    let mut panel=draw(mesh,material,Mat4::from_scale(Vec3::new(4.0,4.0,0.1)),-1);
    panel.gi_source=Some(GiSource {object:0xde10,slot:0});let draws=[panel];
    let field=crate::GiMeshSdf {gi_source:panel.gi_source,world_to_local:panel.transform.inverse(),
        volume:super::flux::cube_distance_field(),in_global_field:false,
        material:GpuGiMaterial {albedo:[0.8;3],emission:[0.0;3],metallic:0.0,roughness:1.0}};
    r.gi_set_gdf(&vec![100.0;512],&vec![0;512],[8;3],[-10.0;3],[10.0;3],&[field.material]);
    r.gi_set_mesh_sdfs(&[field])?;r.gi_set_emitters(&[]);r.set_gi_enabled(true);
    let eye=Vec3::new(0.0,0.0,5.0);let cam=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),
        proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let mut light=LightInstance {kind:LightKind::Point,position:Vec3::Z,direction:Vec3::NEG_Z,color:[1.0;3],intensity:4.0,
        range:4.0,spot_inner_deg:30.0,spot_outer_deg:60.0,cast_shadows:true,soft_shadows:false,radius:0.0,shadow_bias:0.0,baked:false};
    let make_input=|| {let mut input=crate::cascade_tests::input(cam);input.camera_preview=Some(cam);input.viewport_extent=Some([320,180]);input.draws=&draws;
    input.light.intensity=0.0;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;input.draw_skybox=false;
    input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;input};
    for hardware in [false,true] {
        if hardware && !r.ctx.ray_tracing() {continue;}
        r.set_flux_settings(FluxSettings {rt_trace:hardware,samples:16,bounces:1,reflection_mode:0,ssr_enabled:false,..Default::default()});
        {let mut input=make_input();input.lights=std::slice::from_ref(&light);for _ in 0..48 {r.render(&input)?;}}
        let sample={let allocator=r.allocator();let mut alloc=allocator.lock().unwrap();
            r.flux_surface.as_ref().unwrap().debug_lighting_receiver(&r.ctx.device,&mut alloc,r.command_pool,r.ctx.queue,material.0,4)?};
        let p=Vec3::new(sample[0][0],sample[0][1],sample[0][2]);
        let n=Vec3::new(sample[1][0],sample[1][1],sample[1][2]).normalize();
        assert!(n.z>0.99);let receiver=p+n*0.005;
        let diffuse=sample[2][0]*(1.0-sample[2][3])/std::f32::consts::PI;
        // The card stores physical radiance, independent of display exposure.
        for case in 0..12 {
            let (distance,spot_weight)=if case<5 {([0.1f32,0.5,1.0,3.0,5.0][case],1.0)} else {(0.5,0.0)};
            light.kind=if case<5 {LightKind::Point} else {LightKind::Spot};
            light.position=receiver+n*distance;light.spot_inner_deg=30.0;
            let mut weight=spot_weight;
            if case>=5 {
                let ci=15.0f32.to_radians().cos();let co=30.0f32.to_radians().cos();
                let cd=if case<10 {
                    let t=(case-5) as f32*0.25;weight=t*t*(3.0-2.0*t);co+(ci-co)*t
                } else {light.spot_inner_deg=60.0;weight=if case==10 {1.0} else {0.0};co+if case==10 {0.01} else {-0.01}};
                light.direction=Vec3::new((1.0-cd*cd).sqrt(),0.0,-cd);
            }
            {let mut input=make_input();input.lights=std::slice::from_ref(&light);for _ in 0..16 {r.render(&input)?;}}
            let actual={let allocator=r.allocator();let mut alloc=allocator.lock().unwrap();
                r.flux_surface.as_ref().unwrap().debug_lighting(&r.ctx.device,&mut alloc,r.command_pool,r.ctx.queue,material.0,4)?[0][0]};
            let expected=diffuse*4.0*(1.0-distance/4.0).max(0.0).powi(2)/(1.0+distance*distance)*weight;
            eprintln!("[Flux realtime falloff] hardware={hardware}, case={case}, actual={actual}, expected={expected}");
            anyhow::ensure!(actual.is_finite() && (actual-expected).abs()<expected*0.025+0.0001,
                "realtime cache light falloff differs from visible light: hardware={hardware}, case={case}, actual={actual}, expected={expected}");
        }
    }
    Ok(())
}
