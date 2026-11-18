//! A reflected emitter moves while the glossy receiver and camera stay fixed.
//! Compare short-term reconstruction with the settled new-position lighting.
use glam::{Mat4,Quat,Vec3};
use crate::{CameraData,FluxSettings,GiMeshSdf,GiSource,GpuGiMaterial,Renderer};
use super::planar::{cube,draw,lit};

fn patches(r:&Renderer,points:[Vec3;2])->anyhow::Result<[[f32;2];2]> {
    let mut result=[[0.0;2];2];
    for (view, view_value) in result.iter_mut().enumerate() {for (i,point) in points.into_iter().enumerate() {
        let values=super::reflection_denoise::history(r,view,point,false)?;
        view_value[i]=values.iter().sum::<f32>()/values.len() as f32;
    }}
    Ok(result)
}
pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    r.set_gi_enabled(false);r.gi_set_emitters(&[]);r.set_anti_aliasing(crate::AntiAliasing::None);
    let mesh=r.upload_mesh(&cube())?;let volume=super::flux::cube_distance_field();
    let reflector=lit(r,[0.95;4],[0.0;3],0.3,1.0);
    let emitter=lit(r,[0.0,0.0,0.0,1.0],[1.0,0.2,0.1],1.0,0.0);
    let floor=Mat4::from_scale_rotation_translation(Vec3::new(8.0,0.2,8.0),Quat::IDENTITY,Vec3::new(0.0,-0.1,0.0));
    let floor_source=Some(GiSource {object:0x7ef1ec7,slot:0});
    let actor_source=Some(GiSource {object:0x7ef1ec8,slot:0});
    let material=GpuGiMaterial {albedo:[0.95;3],emission:[0.0;3],metallic:1.0,roughness:0.3};
    let emission=GpuGiMaterial {albedo:[0.0;3],emission:[1.0,0.2,0.1],metallic:0.0,roughness:1.0};
    r.gi_set_gdf(&[10.0;512],&[0;512],[8;3],[-10.0;3],[10.0;3],&[material]);
    let eye=Vec3::new(0.0,2.0,4.0);
    let camera=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),
        proj:Mat4::perspective_rh(0.5,16.0/9.0,0.1,100.0)};
    // Mirror virtual emitter centers across y=0 and intersect their camera
    // rays with the unchanged floor. Neither sample follows receiver motion.
    let points=[-1.5,1.5].map(|x| {
        let virtual_center=Vec3::new(x,-1.5,-2.0);
        eye+(virtual_center-eye)*(eye.y/(eye.y-virtual_center.y))
    });
    for hardware in [false,true] {for downsample in [1,2] {
        if hardware && !r.ctx.ray_tracing() {continue;}
        r.set_flux_settings(FluxSettings {rt_trace:hardware,reflection_mode:2,ssr_enabled:false,
            reflection_screen_traces:false,rt_samples:16,rt_bounces:1,reflection_downsample:downsample,..Default::default()});
        let mut old=[[0.0;2];2];let mut moved=[[0.0;2];2];let mut settled=[[0.0;2];2];
        for (stage,x) in [-1.5,1.5].into_iter().enumerate() {
            let actor=Mat4::from_translation(Vec3::new(x,1.5,-2.0));
            let mut draws=[draw(mesh,reflector,floor,-1),draw(mesh,emitter,actor,-1)];
            draws[0].gi_source=floor_source;draws[1].gi_source=actor_source;
            r.gi_set_mesh_sdfs(&[
                GiMeshSdf {gi_source:floor_source,world_to_local:floor.inverse(),volume:volume.clone(),material,in_global_field:false},
                GiMeshSdf {gi_source:actor_source,world_to_local:actor.inverse(),volume:volume.clone(),material:emission,in_global_field:false},
            ])?;
            let mut input=crate::cascade_tests::input(camera);
            input.camera_preview=Some(camera);input.viewport_extent=Some([323,181]);input.draws=&draws;
            input.light.intensity=0.0;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;input.draw_skybox=false;
            input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
            for frame in 0..36 {
                r.render(&input)?;
                if stage==1 && frame==5 {moved=patches(r,points)?;}
                if frame>=28 {
                    let current=patches(r,points)?;
                    let target=if stage==0 {&mut old} else {&mut settled};
                    for view in 0..2 {for p in 0..2 {target[view][p]+=current[view][p]/8.0;}}
                }
            }
        }
        eprintln!("[Moving reflected object] hardware={hardware} downsample={downsample} old={old:?} after6={moved:?} settled={settled:?}");
        for view in 0..2 {
            assert!(old[view][0]>0.1 && settled[view][1]>0.1,"fixture lacks actual reflected emission");
            assert!(settled[view][0]<old[view][0]*0.3,"fixture does not separate old/new reflected positions");
            assert!((moved[view][0]-settled[view][0]).abs()<0.02+old[view][0]*0.1,
                "moving reflected object left stale lighting at its old position");
            assert!((moved[view][1]-settled[view][1]).abs()<0.02+settled[view][1]*0.25,
                "new reflected position did not recover promptly");
        }
    }}
    r.gi_set_mesh_sdfs(&[])?;
    Ok(())
}
