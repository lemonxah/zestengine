//! Shared Scene Lighting Quality must change angular work, retain energy, and
//! continue driving cached reflection lighting when diffuse GI is disabled.
use glam::{Mat4,Vec3};
use crate::{CameraData,FluxSettings,GiSource,GpuGiMaterial,Renderer};
use super::planar::{cube,draw,lit};

pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    r.set_area_lights(&[]);r.set_anti_aliasing(crate::AntiAliasing::None);
    let mesh=r.upload_mesh(&cube())?;
    let color=[0.6,0.3,0.2,1.0];let sky=[0.4,0.2,0.1];
    let material=lit(r,color,[0.0;3],1.0,0.0);
    let mut object=draw(mesh,material,Mat4::from_scale(Vec3::new(8.0,8.0,0.2)),-1);
    object.gi_source=Some(GiSource {object:0xc017a,slot:0});let draws=[object];
    let eye=Vec3::new(0.0,0.0,5.0);let camera=CameraData {position:eye,
        view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let mut input=crate::cascade_tests::input(camera);input.camera_preview=Some(camera);
    input.draws=&draws;input.light.ambient=sky;input.light.intensity=0.0;input.light.env_intensity=0.0;
    r.gi_set_mesh_sdfs(&[])?;r.gi_set_emitters(&[]);
    r.gi_set_gdf(&[100.0;512],&[0;512],[8;3],[-20.0;3],[20.0;3],&[GpuGiMaterial {
        albedo:[0.0;3],emission:[0.0;3],metallic:0.0,roughness:1.0}]);
    for gi in [true,false] {for hardware in [false,true] {
        if hardware && !r.ctx.ray_tracing() {continue;}
        r.set_gi_enabled(gi);r.set_flux_settings(FluxSettings {rt_trace:hardware,gi_screen_traces:false,
            reflection_mode:2,reflection_screen_traces:false,..Default::default()});
        for (scene,camera,visible,resolution) in [(Some(0.5),None,true,4),(Some(0.5),Some(0.5),true,2),
            (Some(2.0),Some(0.5),true,5),(Some(0.5),Some(4.0),true,8),(Some(4.0),Some(0.5),false,2),(Some(6.0),Some(0.5),true,8),
            (Some(0.5),Some(6.0),true,8),(Some(6.0),Some(0.5),false,2),(None,None,true,4)] {
            input.postfx.flux.scene_lighting_quality=scene;input.preview_postfx.flux.scene_lighting_quality=camera;
            input.render_viewport=visible;input.viewport_extent=visible.then_some([323,181]);
            for _ in 0..40 {r.render(&input)?;}
            let cache=r.flux_surface.as_ref().unwrap();
            assert_eq!(cache.lighting_resolution,resolution,"shared quality ignored active-view demand");
            let high=(visible && scene.is_some_and(|q|q>=6.0)) || camera.is_some_and(|q|q>=6.0);
            assert_eq!(cache.debug_lighting_spacing(),if high {2} else {4},"radiosity receiver grid ignored active-view quality");
            let allocator=r.allocator();let (size,pixels)=cache.debug_lighting_page(&r.ctx.device,
                &mut allocator.lock().unwrap(),r.command_pool,r.ctx.queue,material.0,4)?;
            for y in 1..size-1 {for x in 1..size-1 {
                let bounce=pixels[(y*size+x) as usize][1];assert!(bounce[3]>0.0);
                for c in 0..3 {assert!((bounce[c]-color[c]*sky[c]).abs()<0.003,
                    "quality changed expected bounce energy: GI={gi} RT={hardware} grid={resolution} {bounce:?}");}
            }}
        }
        eprintln!("[Flux Scene Lighting Quality] GI={gi} hardware={hardware}: angular grids2/4/5/8 and spatial spacing2/4, shared/hidden views, every interior lighting texel preserves constant-sky bounce energy");
    }}
    r.set_gi_enabled(true);
    Ok(())
}

/// Changing spatial quality must replace only that view's gather resources,
/// preserve constant-environment energy and safely retire submitted images.
pub(super) fn density(r:&mut Renderer)->anyhow::Result<()> {
    r.set_area_lights(&[]);r.gi_set_emitters(&[]);r.set_gi_enabled(true);
    r.set_anti_aliasing(crate::AntiAliasing::None);
    let mesh=r.upload_mesh(&cube())?;
    let material=lit(r,[0.7;4],[0.0;3],1.0,0.0);
    let draws=[draw(mesh,material,Mat4::from_scale(Vec3::new(8.0,8.0,0.2)),-1)];
    let eye=Vec3::new(0.0,0.0,5.0);let camera=CameraData {position:eye,
        view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let mut input=crate::cascade_tests::input(camera);input.camera_preview=Some(camera);
    input.viewport_extent=Some([323,181]);input.draws=&draws;
    input.light.ambient=[0.4,0.2,0.1];input.light.intensity=0.0;input.light.env_intensity=0.0;
    input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
    r.gi_set_mesh_sdfs(&[])?;
    r.gi_set_gdf(&[100.0;512],&[0;512],[8;3],[-20.0;3],[20.0;3],&[GpuGiMaterial {
        albedo:[0.0;3],emission:[0.0;3],metallic:0.0,roughness:1.0}]);
    for hardware in [false,true] {
        if hardware && !r.ctx.ray_tracing() {continue;}
        r.set_flux_settings(FluxSettings {rt_trace:hardware,gi_screen_traces:false,samples:16,
            reflection_mode:0,ssr_enabled:false,..Default::default()});
        let mut baseline=[0.0;2];let mut previous=None;
        for (case,(scene,preview)) in [(None,None),(Some(6.0),None),(Some(6.0),Some(6.0)),
            (Some(8.0),Some(6.0)),(None,Some(6.0)),(None,None)].into_iter().enumerate() {
            input.postfx.flux.final_gather_quality=scene;input.preview_postfx.flux.final_gather_quality=preview;
            for _ in 0..24 {r.render(&input)?;}
            let grids=[r.viewport_target.as_ref().unwrap().sgi.probe_extent,r.camera_preview.as_ref().unwrap().sgi.probe_extent];
            let handles=[r.viewport_target.as_ref().unwrap().sgi.gi_final,r.camera_preview.as_ref().unwrap().sgi.gi_final];
            let dense=[scene.is_some_and(|q|q>=6.0),preview.is_some_and(|q|q>=6.0)];
            let extents=[r.viewport_target.as_ref().unwrap().extent,r.camera_preview.as_ref().unwrap().extent];
            for view in 0..2 {
                let div=if dense[view] {2} else {4};
                assert_eq!(grids[view],ash::vk::Extent2D {width:extents[view].width.div_ceil(div),height:extents[view].height.div_ceil(div)});
                if let Some((old_dense,old_handles))=previous {
                    let old_dense:[bool;2]=old_dense;let old_handles:[ash::vk::Image;2]=old_handles;
                    if old_dense[view]==dense[view] {assert_eq!(handles[view],old_handles[view],"unaffected view lost its resources");}
                }
                let energy=super::flux_radiance::hdr(r,view,camera)?;
                if case==0 {assert!(energy>0.1);baseline[view]=energy;}
                assert!((energy/baseline[view]-1.0).abs()<0.1,"spatial quality changed energy: RT={hardware} case={case} view={view} baseline={} energy={energy}",baseline[view]);
            }
            assert!(r.retired_gi_targets.is_empty(),"old targets did not retire");
            previous=Some((dense,handles));
            eprintln!("[Flux gather density] hardware={hardware} case={case}: {grids:?}, constant-sky energy retained");
        }
        // Cross the threshold every submitted frame, before previous image
        // users retire. Validation must see no destroyed/in-flight resources.
        for i in 0..12 {
            input.postfx.flux.final_gather_quality=Some(if i%2==0 {6.0} else {1.0});
            r.render(&input)?;
            assert!(r.retired_gi_targets.len()<=crate::FRAMES_IN_FLIGHT+1);
        }
        // Game output uses a distinct target; it must honor the same control.
        input.viewport_extent=None;input.camera_preview=None;
        input.postfx.flux.final_gather_quality=Some(6.0);
        for _ in 0..8 {r.render(&input)?;}
        assert_eq!(r.sgi.probe_extent,ash::vk::Extent2D {width:r.render_extent[0].div_ceil(2),height:r.render_extent[1].div_ceil(2)});
        assert!(r.sgi_history_valid);
        input.viewport_extent=Some([323,181]);input.camera_preview=Some(camera);
    }
    Ok(())
}

/// Ray limits are display-relative, while physical cache transport is not.
pub(super) fn exposure_limit(r:&mut Renderer)->anyhow::Result<()> {
    r.set_area_lights(&[]);r.gi_set_emitters(&[]);r.set_gi_enabled(true);
    r.set_anti_aliasing(crate::AntiAliasing::None);
    let mesh=r.upload_mesh(&cube())?;let material=lit(r,[0.7;4],[0.0;3],1.0,0.0);
    let draws=[draw(mesh,material,Mat4::from_scale(Vec3::new(8.0,8.0,0.2)),-1)];
    let eye=Vec3::new(0.0,0.0,5.0);let camera=CameraData {position:eye,
        view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let mut input=crate::cascade_tests::input(camera);input.camera_preview=Some(camera);
    input.viewport_extent=Some([323,181]);input.draws=&draws;
    input.light.ambient=[4.0,2.0,1.0];input.light.intensity=0.0;input.light.env_intensity=0.0;
    input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
    r.gi_set_mesh_sdfs(&[])?;
    r.gi_set_gdf(&[100.0;512],&[0;512],[8;3],[-20.0;3],[20.0;3],&[GpuGiMaterial {
        albedo:[0.0;3],emission:[0.0;3],metallic:0.0,roughness:1.0}]);
    r.gpu_gi.as_mut().unwrap().temporal_override=Some([1.0,1.0,1.0,0.0]);
    for hardware in [false,true] {
        if hardware && !r.ctx.ray_tracing() {continue;}
        r.set_flux_settings(FluxSettings {rt_trace:hardware,gi_screen_traces:false,samples:16,
            firefly_clamp:6.0,reflection_mode:0,ssr_enabled:false,..Default::default()});
        super::flux_lowlight::await_pipelines(r,&input)?;
        let mut baseline=[0.0;2];
        for (case,(main,preview,auto)) in [(0.0,0.0,false),(8.0,0.0,false),(0.0,8.0,false),
            (-8.0,8.0,false),(2.0,0.0,true),(0.0,0.0,false)].into_iter().enumerate() {
            input.postfx.exposure=main;input.preview_postfx.exposure=preview;
            input.postfx.auto_exposure=crate::AutoExposure {enabled:auto,min_ev100:-6.0,max_ev100:-6.0,..Default::default()};
            for _ in 0..32 {r.render(&input)?;}
            let adapted=if auto {r.post_pass.exposure.debug_state(&r.ctx.device,0)[0]} else {0.0};
            let mut energies=[0.0;2];
            for view in 0..2 {
                let value=super::flux_radiance::hdr(r,view,camera)?;energies[view]=value;
                if case==0 {assert!(value>1.0);baseline[view]=value;}
                let ev=if view==0 {main+adapted} else {preview};
                let expected=(6.0*2.0f32.powf(-ev)/4.0).min(1.0);
                assert!((value/baseline[view]/expected-1.0).abs()<0.04,
                    "exposure limit mismatch: RT={hardware} case={case} view={view} energy={value} baseline={} expected_ratio={expected}",baseline[view]);
            }
            let allocator=r.allocator();
            let lighting=r.flux_surface.as_ref().unwrap().debug_lighting(&r.ctx.device,
                &mut allocator.lock().unwrap(),r.command_pool,r.ctx.queue,material.0,4)?;
            assert!((lighting[1][0]-2.8).abs()<0.02,"display exposure contaminated physical surface bounce: {lighting:?}");
            eprintln!("[Flux exposure limit] hardware={hardware} case={case} manual={main}/{preview} auto={adapted}: GI={energies:?}, physical cache={}",lighting[1][0]);
        }
        // The disabled meter still contains the earlier dark-room adaptation.
        // Re-enabling it with a new range must not flash a stale, heavily
        // clamped gather before the new HDR frame is metered.
        input.postfx.auto_exposure=crate::AutoExposure {enabled:true,min_ev100:0.0,max_ev100:0.0,..Default::default()};
        r.render(&input)?;
        let first=super::flux_radiance::hdr(r,0,camera)?;
        assert!((first/baseline[0]-1.0).abs()<0.04,
            "stale disabled exposure affected first gather: RT={hardware} first={first} baseline={}",baseline[0]);
    }
    r.gpu_gi.as_mut().unwrap().temporal_override=None;
    Ok(())
}
