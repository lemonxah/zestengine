//! Production screen hits and hardware fallback, isolated in actual Scene and
//! Camera renders. The screen-only diagnostic proves the accepted path is used;
//! the narrow view excludes the emitter while its floor reflection remains.
use glam::{Mat4, Quat, Vec3};
use crate::{CameraData, FluxSettings, Renderer};
use super::planar::{cube, draw, lit};

fn patch(image: &(u32,u32,Vec<u8>)) -> f32 {
    let (w,h,pixels)=image;
    let mut sum=0.0;
    for y in h/2-2..=h/2+2 { for x in w/2-2..=w/2+2 {
        sum+=pixels[((y*w+x)*4) as usize] as f32;
    }}
    sum/25.0
}
fn sample(r: &mut Renderer, input: &crate::FrameInput<'_>, screen: bool, only: bool) -> anyhow::Result<[f32;2]> {
    r.set_flux_settings(FluxSettings {reflection_screen_traces:screen,..r.flux_settings});
    let rt=r.rt_reflect.as_mut().unwrap();
    rt.screen_trace_only=only;
    rt.invalidate_reflection_history();
    for _ in 0..4 { r.render(input)?; }
    Ok([patch(&r.debug_read_viewport()?.unwrap()),patch(&r.debug_read_camera_preview()?.unwrap())])
}
pub(super) fn run(r: &mut Renderer) -> anyhow::Result<()> {
    if !r.ctx.ray_tracing() {return Ok(());}
    r.set_gi_enabled(false); r.gi_set_emitters(&[]);
    r.set_flux_settings(FluxSettings {rt_trace:true,reflection_mode:2,ssr_enabled:false,rt_samples:1,rt_bounces:1,..Default::default()});
    let mesh=r.upload_mesh(&cube())?;
    let mirror=lit(r,[1.0;4],[0.0;3],0.02,1.0);
    let emitter=lit(r,[0.0,0.0,0.0,1.0],[1.0,0.0,0.0],1.0,0.0);
    let draws=[draw(mesh,mirror,Mat4::from_scale_rotation_translation(Vec3::new(8.0,0.2,8.0),Quat::IDENTITY,Vec3::new(0.0,-0.1,0.0)),-1),
        draw(mesh,emitter,Mat4::from_scale_rotation_translation(Vec3::new(8.0,4.0,0.2),Quat::IDENTITY,Vec3::new(0.0,2.0,-2.5)),-1)];
    let eye=Vec3::new(0.0,2.0,4.0);
    let mut cam=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let mut input=crate::cascade_tests::input(cam);
    input.camera_preview=Some(cam); input.viewport_extent=Some([323,181]); input.draws=&draws;
    input.light.intensity=0.0; input.light.ambient=[0.0;3];input.light.env_intensity=0.0;input.draw_skybox=false;
    input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
    let world=sample(r,&input,false,false)?;
    let screen=sample(r,&input,true,true)?;
    let combined=sample(r,&input,true,false)?;
    eprintln!("[Flux reflection screen] world={world:?} screen={screen:?} combined={combined:?}");
    for v in 0..2 {
        assert!(screen[v]>100.0,"screen tracing never resolved visible emitter: {screen:?}");
        assert!((combined[v]-world[v]).abs()<8.0,"screen/world physical emission disagrees");
        assert!((screen[v]-combined[v]).abs()<8.0,"visible screen hit relied on hardware fallback");
    }
    // The two views must sample their own hierarchy and current HDR.
    let narrow=CameraData {proj:Mat4::perspective_rh(0.4,16.0/9.0,0.1,100.0),..cam};
    input.camera_preview=Some(narrow);
    let separate=sample(r,&input,true,true)?;
    assert!(separate[0]>100.0 && separate[1]<2.0,"reflection screen inputs crossed views: {separate:?}");
    input.camera_preview=Some(cam);
    sample(r,&input,true,false)?;
    input.postfx.flux.reflection_screen_traces=Some(false);
    input.preview_postfx.flux.reflection_screen_traces=Some(true);
    let overrides=sample(r,&input,true,true)?;
    assert!(overrides[0]<2.0 && overrides[1]>100.0,"per-view screen trace control was ignored: {overrides:?}");
    input.postfx.flux.reflection_screen_traces=None;
    input.preview_postfx.flux.reflection_screen_traces=None;
    sample(r,&input,true,false)?;
    // Toggle emission without a camera change: current HDR must have no frame
    // of stale screen lighting. No external cache/history invalidation needed.
    r.material_params_mut(emitter).emission_intensity=0.0;
    for _ in 0..2 {r.render(&input)?;}
    for image in [r.debug_read_viewport()?.unwrap(),r.debug_read_camera_preview()?.unwrap()] {
        assert!(patch(&image)<2.0,"screen reflection retained switched-off emission");
    }
    r.material_params_mut(emitter).emission_intensity=1.0;
    cam.proj=Mat4::perspective_rh(0.4,16.0/9.0,0.1,100.0);
    input.camera=cam;input.camera_preview=Some(cam);
    let world=sample(r,&input,false,false)?;
    let screen=sample(r,&input,true,true)?;
    let combined=sample(r,&input,true,false)?;
    eprintln!("[Flux reflection off-screen] world={world:?} screen={screen:?} combined={combined:?}");
    for v in 0..2 {
        assert!(screen[v]<2.0,"narrow view did not isolate a screen miss");
        assert!(combined[v]>100.0 && (combined[v]-world[v]).abs()<2.0,"off-screen reflection lost world fallback");
    }
    // Moving through the screen boundary must hand the ray to the world
    // tracer without a dark seam or energy jump.
    for fov in [0.45,0.55,0.65,0.75,0.85,0.95] {
        cam.proj=Mat4::perspective_rh(fov,16.0/9.0,0.1,100.0);
        input.camera=cam;input.camera_preview=Some(cam);
        let reference=sample(r,&input,false,false)?;
        let combined=sample(r,&input,true,false)?;
        for v in 0..2 {assert!((reference[v]-combined[v]).abs()<8.0,"reflection edge handoff changed emission: fov={fov} world={reference:?} combined={combined:?}");}
    }
    // TAA uses the projection that generated depth, and resizing independently
    // retires each view's hierarchy while earlier submissions may be in flight.
    r.set_anti_aliasing(crate::AntiAliasing::Taa);
    input.viewport_extent=Some([321,179]);
    for _ in 0..5 {r.render(&input)?;}
    for image in [r.debug_read_viewport()?.unwrap(),r.debug_read_camera_preview()?.unwrap()] {
        assert!(patch(&image)>100.0,"reflection disappeared after jitter/resize");
    }
    r.set_anti_aliasing(crate::AntiAliasing::None);
    r.rt_reflect.as_mut().unwrap().screen_trace_only=false;
    // A screen-visible mirror has only environment specular in pre-resolve HDR.
    // Preserve the second world bounce to an off-screen ceiling emitter.
    let nested=[draws[0],
        draw(mesh,mirror,Mat4::from_scale_rotation_translation(Vec3::new(8.0,4.0,0.2),Quat::IDENTITY,Vec3::new(0.0,2.0,-2.5)),-1),
        draw(mesh,emitter,Mat4::from_scale_rotation_translation(Vec3::new(8.0,0.2,8.0),Quat::IDENTITY,Vec3::new(0.0,3.1,0.0)),-1)];
    input.draws=&nested;
    cam.proj=Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0);
    input.camera=cam;input.camera_preview=Some(cam);
    r.set_flux_settings(FluxSettings {rt_bounces:1,..r.flux_settings});
    let single=sample(r,&input,true,false)?;
    r.set_flux_settings(FluxSettings {rt_bounces:2,..r.flux_settings});
    let world=sample(r,&input,false,false)?;
    let combined=sample(r,&input,true,false)?;
    eprintln!("[Flux reflection nested] single={single:?} world={world:?} combined={combined:?}");
    for v in 0..2 {
        assert!(single[v]<5.0 && world[v]>100.0,"nested reflection fixture lacks a second-bounce signal");
        assert!((world[v]-combined[v]).abs()<5.0,"screen hit terminated a recursive reflection");
    }
    Ok(())
}
