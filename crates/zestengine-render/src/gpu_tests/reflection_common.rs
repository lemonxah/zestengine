//! SSR uses the same signed-radiance reconstruction as hardware reflections.
//! Read physical HDR histories as well as displayed Scene/Camera pixels; fog and
//! exposure must affect only the final composite, never feedback into history.
use glam::{Mat4,Quat,Vec3};
use crate::{CameraData,FluxSettings,Renderer};
use super::planar::{cube,draw,lit};
use super::reflection_denoise::{history,frames};
fn image_mean(image:&(u32,u32,Vec<u8>))->f32 {
    image.2.chunks_exact(4).map(|p|p[0] as f32).sum::<f32>()/(image.0*image.1) as f32
}
fn mean(values:&[f32])->f32 {values.iter().sum::<f32>()/values.len() as f32}
fn pictures(r:&mut Renderer)->anyhow::Result<[(u32,u32,Vec<u8>);2]> {
    Ok([r.debug_read_viewport()?.unwrap(),r.debug_read_camera_preview()?.unwrap()])
}
pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    r.set_gi_enabled(false);r.gi_set_emitters(&[]);
    r.set_anti_aliasing(crate::AntiAliasing::None);
    r.set_flux_settings(FluxSettings {reflection_mode:1,ssr_enabled:true,ssr_roughness_cutoff:0.8,..Default::default()});
    let mesh=r.upload_mesh(&cube())?;
    let reflector=lit(r,[0.8;4],[0.0;3],0.3,1.0);
    let emitter=lit(r,[0.0,0.0,0.0,1.0],[1.0,0.1,0.0],1.0,0.0);
    let draws=[draw(mesh,reflector,Mat4::from_scale_rotation_translation(Vec3::new(8.0,0.2,8.0),Quat::IDENTITY,Vec3::new(0.0,-0.1,0.0)),-1),
        draw(mesh,emitter,Mat4::from_scale_rotation_translation(Vec3::new(8.0,4.0,0.2),Quat::IDENTITY,Vec3::new(0.0,2.0,-2.5)),-1)];
    let eye=Vec3::new(0.0,2.0,4.0);
    let cam=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let narrow=CameraData {proj:Mat4::perspective_rh(0.4,16.0/9.0,0.1,100.0),..cam};
    let mut input=crate::cascade_tests::input(cam);
    input.camera_preview=Some(cam);input.viewport_extent=Some([323,181]);input.draws=&draws;
    input.light.intensity=0.0;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;input.draw_skybox=false;
    input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
    for _ in 0..24 {r.render(&input)?;}
    let baseline=[history(r,0,Vec3::ZERO,false)?,history(r,1,Vec3::ZERO,false)?];
    for v in 0..2 {
        assert!(mean(&baseline[v])>0.2,"SSR fixture lacks reflected emission in view {v}");
        assert!(frames(r,v,Vec3::ZERO)?>2,"SSR temporal history never accumulated in view {v}");
    }
    // Different projection in Camera: emitter leaves the screen; its history
    // must reject the old nonzero hit while Scene retains the reflection.
    input.camera_preview=Some(narrow);
    for _ in 0..4 {r.render(&input)?;}
    assert!(mean(&history(r,0,Vec3::ZERO,false)?)>0.2);
    assert!(mean(&history(r,1,Vec3::ZERO,false)?).abs()<0.002,"Camera SSR reused Scene or offscreen history");
    input.camera_preview=Some(cam);
    for _ in 0..24 {r.render(&input)?;}
    let clear=pictures(r)?;
    input.fog=Some(crate::FogParams {color:[0.02,0.03,0.05],density:0.4,height_falloff:0.0,height_ref:0.0,start_distance:0.0});
    input.time=5.0;
    for _ in 0..3 {r.render(&input)?;}
    let fogged=pictures(r)?;
    for v in 0..2 {
        assert!((image_mean(&clear[v])-image_mean(&fogged[v])).abs()>5.0,"fog did not affect resolved view {v}");
        assert!(frames(r,v,Vec3::ZERO)?>2,"animated fog discarded reflection history");
    }
    // Compare unfiltered physical samples exactly: a spatial/temporal kernel
    // can still converge at a few reflection boundaries in a static frame.
    // Its independent history-continuity check above remains enabled.
    r.ssr_resolve.denoise_override=Some(false);
    r.render(&input)?;
    let baseline=[history(r,0,Vec3::ZERO,false)?,history(r,1,Vec3::ZERO,false)?];
    let fog=input.fog;input.fog=None;r.render(&input)?;
    for v in 0..2 {assert_eq!(history(r,v,Vec3::ZERO,false)?,baseline[v],"fog contaminated SSR physical samples");}
    input.fog=fog;
    input.postfx.exposure=0.25;input.preview_postfx.exposure=4.0;
    r.render(&input)?;
    for v in 0..2 {assert!(history(r,v,Vec3::ZERO,false)?.iter().zip(&baseline[v]).all(|(a,b)|(a-b).abs()<0.002),"exposure contaminated SSR history");}
    input.postfx.exposure=1.0;input.preview_postfx.exposure=1.0;
    input.fog=None;
    r.ssr_resolve.denoise_override=None;
    for _ in 0..12 {r.render(&input)?;}
    // Receiver motion rejects old geometry; hidden views and resizes reject
    // incompatible histories without affecting the other live view.
    let mut raised=draws;raised[0].transform=Mat4::from_scale_rotation_translation(Vec3::new(8.0,0.2,8.0),Quat::IDENTITY,Vec3::new(0.0,0.5,0.0));
    input.draws=&raised;let point=Vec3::new(0.0,0.6,0.0);r.render(&input)?;
    for v in 0..2 {assert_eq!(frames(r,v,point)?,1,"SSR reused moved receiver history");}
    for _ in 0..12 {r.render(&input)?;}
    let old=r.ssr_resolve.denoiser.debug_history(1).4;
    input.camera_preview=None;for _ in 0..3 {r.render(&input)?;}
    assert_eq!(r.ssr_resolve.denoiser.debug_history(1).4,old);
    input.camera_preview=Some(cam);r.render(&input)?;assert_eq!(frames(r,1,point)?,1);
    input.viewport_extent=Some([321,179]);r.render(&input)?;assert_eq!(frames(r,0,point)?,1);
    for _ in 0..12 {r.render(&input)?;}
    r.material_params_mut(emitter).emission_intensity=0.0;
    for _ in 0..3 {r.render(&input)?;}
    for v in 0..2 {assert!(history(r,v,point,false)?.iter().all(|v|v.abs()<0.002),"SSR retained switched-off light");}
    r.material_params_mut(emitter).emission_intensity=1.0;
    r.material_params_mut(reflector).roughness=0.02;
    for _ in 0..3 {r.render(&input)?;}
    for v in 0..2 {assert_eq!(frames(r,v,point)?,1,"SSR blurred a sharp mirror");}
    r.set_anti_aliasing(crate::AntiAliasing::Taa);
    for _ in 0..4 {
        r.render(&input)?;
        let actual=r.ssr_resolve.denoiser.debug_history(0).3.to_cols_array();
        let depth=Mat4::from_cols_array_2d(&r.last_frame_ubo.as_ref().unwrap().view_proj).to_cols_array();
        assert!(actual.iter().zip(depth).all(|(a,b)|(a-b).abs()<1e-4),"SSR reconstruction did not use jittered depth projection");
    }
    r.set_anti_aliasing(crate::AntiAliasing::None);
    // With reflection replacement disabled, fog must be numerically the same
    // in the legacy fog-only, SSR and RT composites, including sky pixels.
    input.draws=&draws;
    input.fog=Some(crate::FogParams {color:[0.02,0.03,0.05],density:0.4,height_falloff:0.0,height_ref:0.0,start_distance:0.0});
    let mut reference=None;
    for mode in [0,1,2] {
        if mode==2 && !r.ctx.ray_tracing() {continue;}
        r.set_flux_settings(FluxSettings {rt_trace:true,reflection_mode:mode,ssr_intensity:0.0,..r.flux_settings});
        for _ in 0..3 {r.render(&input)?;}
        let images=pictures(r)?;
        if let Some(old)=&reference {
            let old:&[(u32,u32,Vec<u8>);2]=old;
            for v in 0..2 {
                assert_eq!((old[v].0,old[v].1),(images[v].0,images[v].1));
                let max_error=old[v].2.iter().zip(&images[v].2).map(|(a,b)|a.abs_diff(*b)).max().unwrap();
                assert!(max_error<=1,"fog composition disagrees in mode {mode} view {v}: max error {max_error}");
            }
        } else {reference=Some(images);}
    }
    if std::env::var_os("ZESTENGINE_COMMON_REFLECTION_BENCH").is_some() {
        input.camera_preview=None;input.viewport_extent=Some([1280,720]);
        r.material_params_mut(reflector).roughness=0.3;
        r.set_flux_settings(FluxSettings {reflection_mode:1,ssr_enabled:true,ssr_intensity:1.0,..Default::default()});
        for fog in [None,input.fog] {
            input.fog=fog;
            let mut timings=[Vec::new(),Vec::new()];
            for reconstruct in [false,true,true,false] {
                r.ssr_resolve.reconstruction_override=Some(reconstruct);
                for _ in 0..16 {r.render(&input)?;}
                for _ in 0..48 {r.render(&input)?;timings[usize::from(reconstruct)].push(r.stats().gpu_reflect_ms);}
            }
            let medians=timings.map(|mut v| {v.sort_by(f32::total_cmp);v[v.len()/2]});
            eprintln!("[Flux common reflection timing] fog={} legacy={}ms reconstructed={}ms delta={}ms; 1280x720, 96 samples/mode, alternating blocks",fog.is_some(),medians[0],medians[1],medians[1]-medians[0]);
        }
        r.ssr_resolve.reconstruction_override=None;
    }
    eprintln!("[Flux common reflections] SSR histories, independent views, light-off, motion, resize, jitter, fog/exposure isolation and fog-only/SSR/RT equivalence passed");
    Ok(())
}
