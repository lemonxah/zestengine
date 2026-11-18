//! Software world reflections must work without a TLAS and without screen hits.
use glam::{Mat4,Quat,Vec3};
use crate::{CameraData,FluxSettings,GpuGiMaterial,GiMeshSdf,Renderer};
use super::planar::{cube,draw,lit};
fn patch(image:&(u32,u32,Vec<u8>))->f32 {
    let (w,h,p)=image;let mut sum=0.0;
    for y in h/2-2..=h/2+2 {for x in w/2-2..=w/2+2 {sum+=p[((y*w+x)*4) as usize] as f32;}}
    sum/25.0
}
fn sample(r:&mut Renderer,input:&crate::FrameInput<'_>)->anyhow::Result<[f32;2]> {
    for _ in 0..4 {r.render(input)?;}
    Ok([patch(&r.debug_read_viewport()?.unwrap()),patch(&r.debug_read_camera_preview()?.unwrap())])
}
pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    r.set_gi_enabled(false);r.gi_set_emitters(&[]);
    r.set_anti_aliasing(crate::AntiAliasing::None);
    let mesh=r.upload_mesh(&cube())?;
    let mirror=lit(r,[1.0;4],[0.0;3],0.02,1.0);
    let emitter=lit(r,[0.0,0.0,0.0,1.0],[1.0,0.0,0.0],1.0,0.0);
    let floor=Mat4::from_scale_rotation_translation(Vec3::new(8.0,0.2,8.0),Quat::IDENTITY,Vec3::new(0.0,-0.1,0.0));
    let wall=Mat4::from_scale_rotation_translation(Vec3::new(8.0,4.0,0.2),Quat::IDENTITY,Vec3::new(0.0,2.0,-2.5));
    let draws=[draw(mesh,mirror,floor,-1),draw(mesh,emitter,wall,-1)];
    let material=GpuGiMaterial {albedo:[1.0;3],emission:[0.0;3],metallic:1.0,roughness:0.02};
    let emission=GpuGiMaterial {albedo:[0.0;3],emission:[1.0,0.0,0.0],metallic:0.0,roughness:1.0};
    let volume=super::flux::cube_distance_field();
    let mut fields=vec![GiMeshSdf {gi_source:None,world_to_local:floor.inverse(),volume:volume.clone(),material,in_global_field:false},
        GiMeshSdf {gi_source:None,world_to_local:wall.inverse(),volume:volume.clone(),material:emission,in_global_field:false}];
    r.gi_set_gdf(&[10.0;512],&[0;512],[8;3],[-10.0;3],[10.0;3],&[material]);
    r.gi_set_mesh_sdfs(&fields)?;
    let eye=Vec3::new(0.0,2.0,4.0);
    let cam=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(0.4,16.0/9.0,0.1,100.0)};
    let wide=CameraData {proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0),..cam};
    let mut input=crate::cascade_tests::input(cam);input.camera_preview=Some(cam);input.viewport_extent=Some([323,181]);input.draws=&draws;
    input.light.intensity=0.0;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;input.draw_skybox=false;
    input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
    r.set_flux_settings(FluxSettings {rt_trace:false,reflection_mode:1,..Default::default()});
    let screen=sample(r,&input)?;
    r.set_flux_settings(FluxSettings {rt_trace:false,reflection_mode:2,reflection_screen_traces:false,..Default::default()});
    // The Software backend must never silently dispatch hardware rays while
    // waiting for its geometry field. With screen tracing disabled it keeps
    // only the probe/environment base until the field becomes available.
    let geometry=r.gpu_gi.take();
    let unavailable=sample(r,&input)?;
    assert!(unavailable.iter().all(|v|*v<2.0),"Software backend silently used hardware tracing");
    r.gpu_gi=geometry;
    let world=sample(r,&input)?;
    assert!(r.software_reflections(&r.flux_settings));
    for v in 0..2 {assert!(screen[v]<2.0 && world[v]>100.0,"offscreen software reflection missing: screen={screen:?} world={world:?}");}
    // Separate view projections, with screen tracing enabled in only one view.
    input.camera_preview=Some(wide);input.preview_postfx.flux.reflection_screen_traces=Some(true);
    let split=sample(r,&input)?;
    for v in 0..2 {assert!((split[v]-world[v]).abs()<8.0,"software screen/world handoff disagrees: {split:?} {world:?}");}
    input.preview_postfx.flux.reflection_screen_traces=None;input.camera_preview=Some(cam);
    // Stop tracing before the wall, then restore the full range.
    r.set_flux_settings(FluxSettings {ssr_max_distance:1.0,..r.flux_settings});
    let short=sample(r,&input)?;assert!(short.iter().all(|v|*v<2.0),"software reflection distance control ignored");
    r.set_flux_settings(FluxSettings {ssr_max_distance:40.0,..r.flux_settings});
    // An offscreen pane occludes the reflected emitter. The raster image is
    // unchanged; only software geometry may answer this visibility query.
    let panel=Mat4::from_scale_rotation_translation(Vec3::new(8.0,4.0,0.05),Quat::IDENTITY,Vec3::new(0.0,2.0,-1.0));
    fields.push(GiMeshSdf {gi_source:None,world_to_local:panel.inverse(),volume:volume.clone(),material:GpuGiMaterial {albedo:[0.0;3],emission:[0.0;3],metallic:0.0,roughness:1.0},in_global_field:false});
    r.gi_set_mesh_sdfs(&fields)?;let blocked=sample(r,&input)?;
    assert!(blocked.iter().all(|v|*v<2.0),"software reflection crossed opaque pane: {blocked:?}");
    fields.pop();r.gi_set_mesh_sdfs(&fields)?;let reopened=sample(r,&input)?;
    assert!(reopened.iter().all(|v|*v>100.0));
    // Material-only mesh update changes actual offscreen emission immediately.
    fields[1].material.emission=[0.0;3];r.material_params_mut(emitter).emission_intensity=0.0;
    r.gi_set_mesh_sdfs(&fields)?;
    let dark=sample(r,&input)?;assert!(dark.iter().all(|v|*v<2.0),"software reflection retained stale emissive material");
    fields[1].material=emission;r.material_params_mut(emitter).emission_intensity=1.0;r.gi_set_mesh_sdfs(&fields)?;
    input.viewport_extent=Some([321,179]);r.set_anti_aliasing(crate::AntiAliasing::Taa);
    let jittered=sample(r,&input)?;assert!(jittered.iter().all(|v|*v>100.0),"software reflection failed jitter/resize");
    r.set_anti_aliasing(crate::AntiAliasing::None);
    // Force the coarse global-field path too; it must shade the offscreen hit
    // from its material table when there are no mesh snapshots or cards.
    r.gi_set_mesh_sdfs(&[])?;
    let min=Vec3::new(-5.0,-1.0,-4.0);let max=Vec3::new(5.0,5.0,5.0);let dims=[64u32;3];
    let mut distances=Vec::new();let mut indices=Vec::new();
    let sdf=|p:Vec3,center:Vec3,half:Vec3| {let q=(p-center).abs()-half;q.max(Vec3::ZERO).length()+q.max_element().min(0.0)};
    for z in 0..64 {for y in 0..64 {for x in 0..64 {
        let p=min+(max-min)*Vec3::new(x as f32,y as f32,z as f32)/63.0;
        let a=sdf(p,Vec3::new(0.0,-0.1,0.0),Vec3::new(4.0,0.1,4.0));
        let b=sdf(p,Vec3::new(0.0,2.0,-2.5),Vec3::new(4.0,2.0,0.1));
        distances.push(a.min(b));indices.push(u32::from(b<a));
    }}}
    r.gi_set_gdf(&distances,&indices,dims,min.to_array(),max.to_array(),&[material,emission]);
    input.draws=&draws[..1]; // No offscreen material capture: exercise the GDF fallback.
    let global=sample(r,&input)?;
    assert!(global.iter().all(|v|*v>100.0),"global-field software reflection missing: {global:?}");
    r.gi_set_gdf(&[10.0;512],&[0;512],[8;3],[-10.0;3],[10.0;3],&[material]);
    // Two specular bounces: floor -> offscreen mirror -> ceiling emitter.
    let ceiling=Mat4::from_scale_rotation_translation(Vec3::new(8.0,0.2,8.0),Quat::IDENTITY,Vec3::new(0.0,3.1,0.0));
    let nested=[draws[0],draw(mesh,mirror,wall,-1),draw(mesh,emitter,ceiling,-1)];
    fields[1].material=material;
    fields.push(GiMeshSdf {gi_source:None,world_to_local:ceiling.inverse(),volume:volume.clone(),material:emission,in_global_field:false});
    r.gi_set_mesh_sdfs(&fields)?;input.draws=&nested;
    let single=sample(r,&input)?;
    r.set_flux_settings(FluxSettings {rt_bounces:2,..r.flux_settings});let recursive=sample(r,&input)?;
    assert!(single.iter().all(|v|*v<2.0) && recursive.iter().all(|v|*v>100.0),"software recursive reflection missing: {single:?} {recursive:?}");
    fields.pop();fields[1].material=emission;r.gi_set_mesh_sdfs(&fields)?;input.draws=&draws;
    r.set_flux_settings(FluxSettings {rt_samples:4,rt_bounces:1,..r.flux_settings});r.material_params_mut(mirror).roughness=0.3;
    let mut accumulated=[0;2];
    for _ in 0..24 {r.render(&input)?;for (v, v_value) in accumulated.iter_mut().enumerate() {*v_value=(*v_value).max(super::reflection_denoise::frames(r,v,Vec3::ZERO)?);}}
    assert!(accumulated.iter().all(|n|*n>2),"software glossy reflections did not accumulate: {accumulated:?}");
    let mut density_energy=Vec::new();
    for downsample in [1,2] {
        r.set_flux_settings(FluxSettings {reflection_downsample:downsample,..r.flux_settings});
        let mut energy=[0.0;2];
        for frame in 0..24 {
            r.render(&input)?;
            if frame>=8 {
                energy[0]+=patch(&r.debug_read_viewport()?.unwrap())/16.0;
                energy[1]+=patch(&r.debug_read_camera_preview()?.unwrap())/16.0;
            }
        }
        density_energy.push(energy);
    }
    for view in 0..2 {
        let full=density_energy[0][view];let adaptive=density_energy[1][view];
        assert!(full>50.0 && (adaptive-full).abs()<full*0.12,
            "software adaptive reflection lost energy or left holes: {density_energy:?}");
    }
    eprintln!("[Flux software reflection density] full/adaptive Scene/Camera energy={density_energy:?}");
    r.material_params_mut(mirror).roughness=0.02;r.set_flux_settings(FluxSettings {rt_samples:1,reflection_downsample:1,..r.flux_settings});
    // Hardware is a comparison, not a prerequisite for this test.
    if r.ctx.ray_tracing() {
        r.set_flux_settings(FluxSettings {rt_trace:true,..r.flux_settings});
        let hardware=sample(r,&input)?;
        for v in 0..2 {assert!((hardware[v]-world[v]).abs()<8.0,"software/hardware emission disagree: {hardware:?} {world:?}");}
    }
    if std::env::var_os("ZESTENGINE_SOFTWARE_REFLECTION_BENCH").is_some() {
        input.camera_preview=None;input.viewport_extent=Some([1280,720]);
        r.material_params_mut(mirror).roughness=0.3;
        for samples in [1,4] {
            for hardware in [false,true,false,true] {
                if hardware && !r.ctx.ray_tracing() {continue;}
                r.set_flux_settings(FluxSettings {rt_trace:hardware,rt_samples:samples,reflection_screen_traces:false,..r.flux_settings});
                for _ in 0..16 {r.render(&input)?;}
                let mut timings=Vec::new();for _ in 0..48 {r.render(&input)?;timings.push(r.stats().gpu_reflect_ms);}
                timings.sort_by(f32::total_cmp);
                eprintln!("[Flux software reflection timing] hardware={hardware} samples={samples} median_ms={} 1280x720, 48 frames",timings[24]);
            }
        }
    }
    r.gi_set_mesh_sdfs(&[])?;
    eprintln!("[Flux software reflections] offscreen screen={screen:?} world={world:?}; range, pane occlusion, material refresh, per-view screen continuation and jitter passed");
    Ok(())
}
