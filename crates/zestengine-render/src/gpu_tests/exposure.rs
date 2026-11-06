//! Real HDR -> histogram -> adaptation -> bloom/post in both editor views.
use glam::{Mat4,Vec3};
use crate::{AutoExposure,CameraData,Renderer};
use super::planar::{cube,draw,lit};

fn center(image:&(u32,u32,Vec<u8>))->f32 {
    let (w,h,p)=image;let mut sum=0.0;
    for y in h/2-3..h/2+3 {for x in w/2-3..w/2+3 {sum+=p[((y*w+x)*4) as usize] as f32;}}
    sum/36.0
}
pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    r.set_gi_enabled(false);
    r.set_flux_settings(crate::FluxSettings {reflection_mode:0,ssr_enabled:false,..Default::default()});
    let mesh=r.upload_mesh(&cube())?;
    let scene_mat=lit(r,[0.0,0.0,0.0,1.0],[11.52;3],1.0,0.0);
    let camera_mat=lit(r,[0.0,0.0,0.0,1.0],[0.045;3],1.0,0.0);
    for material in [scene_mat,camera_mat] {r.set_material_features(material,crate::MaterialFeatures {emission:true,..Default::default()});}
    let draws=[draw(mesh,scene_mat,Mat4::from_scale_rotation_translation(Vec3::new(12.0,8.0,1.0),glam::Quat::IDENTITY,Vec3::new(-20.0,0.0,0.0)),-1),
        draw(mesh,camera_mat,Mat4::from_scale_rotation_translation(Vec3::new(12.0,8.0,1.0),glam::Quat::IDENTITY,Vec3::new(20.0,0.0,0.0)),-1)];
    let camera=|x|CameraData {position:Vec3::new(x,0.0,3.0),view:Mat4::look_at_rh(Vec3::new(x,0.0,3.0),Vec3::new(x,0.0,0.0),Vec3::Y),
        proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let mut input=crate::cascade_tests::input(camera(-20.0));input.camera_preview=Some(camera(20.0));
    input.viewport_extent=Some([323,181]);input.draws=&draws;
    input.light.ambient=[0.0;3];input.light.intensity=0.0;input.light.env_intensity=0.0;
    input.postfx.tonemap=0;input.preview_postfx.tonemap=0;
    input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
    r.render(&input)?;
    let manual=[center(&r.debug_read_viewport()?.unwrap()),center(&r.debug_read_camera_preview()?.unwrap())];
    assert!(manual[0]>manual[1]+50.0,"fixture did not produce distinct HDR views: {manual:?}");
    let auto=AutoExposure {enabled:true,low_percent:0.0,high_percent:100.0,..Default::default()};
    input.postfx.auto_exposure=auto;input.preview_postfx.auto_exposure=auto;
    r.post_pass.exposure.time_override=Some(0.0);r.render(&input)?;
    let read=|r:&Renderer,view|r.post_pass.exposure.debug_state(&r.ctx.device,view);
    let scene=read(r,0);let preview=read(r,1);
    assert!((scene[0]+6.0).abs()<0.03 && (preview[0]-2.0).abs()<0.03,
        "independent HDR metering failed: {scene:?}, {preview:?}");
    let metered=[center(&r.debug_read_viewport()?.unwrap()),center(&r.debug_read_camera_preview()?.unwrap())];
    assert!((metered[0]-metered[1]).abs()<2.0 && (20.0..240.0).contains(&metered[0]),"post did not apply view exposure: {metered:?}");
    // Bloom must see the same exposure, and must not enter the histogram.
    input.postfx.bloom_enabled=true;input.preview_postfx.bloom_enabled=true;
    input.postfx.bloom_threshold=0.1;input.preview_postfx.bloom_threshold=0.1;
    for i in 1..=6 {r.post_pass.exposure.time_override=Some(i as f64/60.0);r.render(&input)?;}
    let bloom=[center(&r.debug_read_viewport()?.unwrap()),center(&r.debug_read_camera_preview()?.unwrap())];
    assert!((bloom[0]-bloom[1]).abs()<2.0 && bloom[0]>metered[0],"bloom exposure differs: {bloom:?}");
    assert!((read(r,0)[1]-scene[1]).abs()<0.001,"bloom fed back into metering");
    input.postfx.exposure=1.0;r.render(&input)?;
    assert!(center(&r.debug_read_viewport()?.unwrap())>bloom[0]+10.0,"compensation ignored");
    assert!((read(r,0)[1]-scene[1]).abs()<0.001,"display compensation fed back into metering");
    eprintln!("[Flux auto exposure] manual={manual:?}, auto={metered:?}, bloom={bloom:?}, states={scene:?}/{preview:?}");
    input.postfx.exposure=0.0;input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
    input.render_viewport=false;
    r.material_params_mut(camera_mat).emission_color=[0.18;3];
    r.post_pass.exposure.time_override=Some(3.0);r.render(&input)?;
    assert_eq!(read(r,0),scene,"hidden Scene exposure was updated by Camera");
    assert!(read(r,1)[0].abs()<0.03,"Camera did not meter while Scene hidden");
    input.render_viewport=true;
    // Bright outliers must be trimmed using histogram percentiles, rather
    // than averaging the display image or letting a small TV dominate it.
    let hot=lit(r,[0.0,0.0,0.0,1.0],[10000.0;3],1.0,0.0);
    r.set_material_features(hot,crate::MaterialFeatures {emission:true,..Default::default()});
    let mut with_hot=draws.to_vec();
    with_hot.push(draw(mesh,hot,Mat4::from_scale_rotation_translation(Vec3::new(0.8,0.6,0.1),glam::Quat::IDENTITY,Vec3::new(-20.0,0.0,1.0)),-1));
    input.draws=&with_hot;
    r.material_params_mut(scene_mat).emission_color=[0.18;3];
    input.postfx.auto_exposure.histogram_min_ev100=-8.0-1.2f32.log2();
    input.postfx.auto_exposure.histogram_max_ev100=7.75-1.2f32.log2();
    r.post_pass.exposure.time_override=Some(5.0);r.render(&input)?;
    let untrimmed=read(r,0)[1];
    input.postfx.auto_exposure.high_percent=80.0;
    r.post_pass.exposure.time_override=Some(7.0);r.render(&input)?;
    let trimmed=read(r,0)[1];
    assert!(untrimmed< -0.15 && trimmed.abs()<0.03,"histogram failed to reject bright outlier: {untrimmed}->{trimmed}");
    eprintln!("[Flux auto exposure outlier] untrimmed={untrimmed}, trimmed={trimmed}");
    input.draws=&draws;input.postfx.auto_exposure=auto;
    // Frame-rate independence and asymmetric speeds, using deterministic
    // timestamps while all HDR work/adaptation still runs on the GPU.
    let mut adapted=Vec::new();
    for (pass,fps) in [15,60].into_iter().enumerate() {
        let start=10.0+pass as f64*10.0;
        r.material_params_mut(scene_mat).emission_color=[0.18;3];
        r.post_pass.exposure.time_override=Some(start);r.render(&input)?;
        let old=read(r,0)[0];
        r.material_params_mut(scene_mat).emission_color=[11.52;3];
        for f in 1..=fps {r.post_pass.exposure.time_override=Some(start+f as f64/fps as f64);r.render(&input)?;}
        let bright=read(r,0)[0];
        assert!((bright-old+3.0).abs()<0.03,"bright adaptation is not 3 stops/s: {old}->{bright}");
        for f in 1..=fps*2 {r.post_pass.exposure.time_override=Some(start+1.0+f as f64/fps as f64);r.render(&input)?;}
        let near=read(r,0)[0];
        let expected=-6.0+1.5*(-3.0f32).exp();
        assert!((near-expected).abs()<0.03,"linear/exponential crossing failed: {near}, expected={expected}");
        r.material_params_mut(scene_mat).emission_color=[0.18/256.0;3];
        for f in 1..=fps {r.post_pass.exposure.time_override=Some(start+3.0+f as f64/fps as f64);r.render(&input)?;}
        let dark=read(r,0)[0];
        assert!((dark-near-1.0).abs()<0.03,"dark adaptation is not 1 stop/s: {near}->{dark}");
        adapted.push([bright,near,dark]);
    }
    assert!(adapted[0].iter().zip(adapted[1]).all(|(a,b)|(*a-b).abs()<0.01));
    // Locked EV100 and malformed serialized controls remain finite.
    input.postfx.auto_exposure.min_ev100=2.0;input.postfx.auto_exposure.max_ev100=2.0;
    r.render(&input)?;
    assert!((read(r,0)[0]-(-2.0-1.2f32.log2())).abs()<0.001,"EV100 calibration/clamp incorrect");
    input.postfx.auto_exposure=AutoExposure {min_ev100:f32::NAN,max_ev100:f32::NAN,low_percent:100.0,
        high_percent:0.0,histogram_min_ev100:24.0,histogram_max_ev100:-20.0,..auto};
    r.render(&input)?;assert!(read(r,0).iter().all(|v|v.is_finite()));
    input.postfx.auto_exposure.min_ev100=24.0;input.postfx.auto_exposure.max_ev100=f32::NAN;
    r.render(&input)?;assert!((read(r,0)[0]-(-24.0-1.2f32.log2())).abs()<0.001);
    // Flux-only illumination must be metered, without scaling physical GI.
    for rt in [false,true] {
        if rt && !r.ctx.ray_tracing() {continue;}
        r.material_params_mut(scene_mat).emission_color=[0.0;3];r.material_params_mut(scene_mat).base_color=[0.7;4];
        r.gi_set_gdf(&vec![10.0;512],&vec![0;512],[8;3],[-100.0;3],[100.0;3],
            &[crate::GpuGiMaterial {albedo:[0.7;3],emission:[0.0;3],metallic:0.0,roughness:1.0}]);
        r.gi_set_emitters(&[]);r.gi_set_mesh_sdfs(&[])?;r.set_gi_enabled(true);
        r.set_flux_settings(crate::FluxSettings {rt_trace:rt,samples:16,reflection_mode:0,ssr_enabled:false,..Default::default()});
        input.light.ambient=[0.1;3];input.gi_debug=6;input.postfx.auto_exposure=auto;
        for f in 0..8 {r.post_pass.exposure.time_override=Some(40.0+f as f64/60.0);r.render(&input)?;}
        let before=read(r,0);assert!(before[2]>0.001,"Flux HDR was absent from histogram");
        input.postfx.exposure=2.0;
        for f in 8..12 {r.post_pass.exposure.time_override=Some(40.0+f as f64/60.0);r.render(&input)?;}
        let after=read(r,0);
        assert!((before[2]-after[2]).abs()<0.001,"exposure changed physical Flux radiance: {before:?}->{after:?}");
        input.postfx.exposure=0.0;
        eprintln!("[Flux auto exposure GI] RT={rt}: {before:?}->{after:?}");
    }
    eprintln!("[Flux auto exposure adaptation] 15/60fps={adapted:?}");
    r.post_pass.exposure.time_override=None;
    Ok(())
}
