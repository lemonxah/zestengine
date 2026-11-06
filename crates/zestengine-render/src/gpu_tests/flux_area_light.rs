//! Area textures must affect physical cached radiance and both visible views.
use glam::{Mat4,Vec3};
use crate::{AreaLightInstance,AreaShape,CameraData,CompressedTexture,FluxSettings,GiSource,GpuGiMaterial,Renderer,TexFormat};
use super::planar::{cube,draw,lit};

// Constant BC7 mode 6 block; even endpoints share p=0. Exercises the actual
// compressed upload route that was absent from the old blit-source registry.
fn bc7(color:[u8;4])->CompressedTexture {
    let mut block=[0u8;16];let mut bit=0usize;
    let mut write=|value:u32,count:usize| {for b in 0..count {block[(bit+b)/8]|=(((value>>b)&1) as u8)<<((bit+b)%8);}bit+=count;};
    write(64,7);
    for c in color {write((c/2) as u32,7);write((c/2) as u32,7);}
    write(0,2); // endpoint p-bits; all indices are zero
    CompressedTexture {format:TexFormat::Bc7Srgb,width:4,height:4,mips:vec![block.to_vec()]}
}
fn center(image:&(u32,u32,Vec<u8>))->[f32;3] {
    let (w,h,p)=image;let mut sum=[0.0;3];
    for y in h/2-2..h/2+3 {for x in w/2-2..w/2+3 {for c in 0..3 {sum[c]+=p[((y*w+x)*4) as usize+c] as f32/25.0;}}}sum
}
pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    let mesh=r.upload_mesh(&cube())?;let material=lit(r,[0.8;4],[0.0;3],1.0,0.0);
    let mut panel=draw(mesh,material,Mat4::from_scale(Vec3::new(4.0,4.0,0.1)),-1);
    panel.gi_source=Some(GiSource {object:0xaea0,slot:0});let mut draws=vec![panel];
    let make_field=|d:crate::DrawCmd| crate::GiMeshSdf {gi_source:d.gi_source,world_to_local:d.transform.inverse(),
        volume:super::flux::cube_distance_field(),in_global_field:false,
        material:GpuGiMaterial {albedo:[0.8;3],emission:[0.0;3],metallic:0.0,roughness:1.0}};
    r.gi_set_gdf(&vec![100.0;512],&vec![0;512],[8;3],[-10.0;3],[10.0;3],&[make_field(panel).material]);
    r.gi_set_mesh_sdfs(&[make_field(panel)])?;r.gi_set_emitters(&[]);r.set_gi_enabled(true);
    let red=r.upload_compressed_texture(&bc7([254,0,0,254]))?;
    let grey=r.upload_compressed_texture(&bc7([128,128,128,254]))?;
    let black=r.upload_compressed_texture(&bc7([0,0,0,254]))?;
    let eye=Vec3::new(0.0,0.0,5.0);let cam=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),
        proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let mut light=AreaLightInstance {shape:AreaShape::Rect,position:Vec3::new(0.0,0.0,1.55),right:Vec3::X,up:Vec3::NEG_Y,
        half_size:[0.5,0.5],color:[0.5,0.25,1.0],intensity:4.0,range:6.0,two_sided:false,texture:None};
    let render=|r:&mut Renderer,ds:&[crate::DrawCmd],frames|->anyhow::Result<()> {
        let mut input=crate::cascade_tests::input(cam);input.camera_preview=Some(cam);input.viewport_extent=Some([320,180]);input.draws=ds;
        input.light.intensity=0.0;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;input.draw_skybox=false;
        input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
        for _ in 0..frames {r.render(&input)?;}Ok(())
    };
    let cache=|r:&Renderer|->anyhow::Result<[f32;4]> {let allocator=r.allocator();let mut alloc=allocator.lock().unwrap();
        Ok(r.flux_surface.as_ref().unwrap().debug_lighting(&r.ctx.device,&mut alloc,r.command_pool,r.ctx.queue,material.0,4)?[0])};
    for hardware in [false,true] {
        if hardware && !r.ctx.ray_tracing() {continue;}
        r.set_flux_settings(FluxSettings {rt_trace:hardware,samples:16,bounces:1,reflection_mode:0,ssr_enabled:false,..Default::default()});
        light.up=Vec3::NEG_Y;light.two_sided=false;light.texture=None;light.shape=AreaShape::Rect;
        r.set_area_lights(&[light]);render(r,&draws,64)?;
        let reference=cache(r)?;
        anyhow::ensure!(reference[0]>0.03,"area did not reach surface cache: {reference:?}");
        anyhow::ensure!((reference[1]/reference[0]-0.5).abs()<0.01 && (reference[2]/reference[0]-2.0).abs()<0.02,
            "area color not multiplied: {reference:?}");
        // Independent numerical area integral at the actual captured receiver.
        let sample={let allocator=r.allocator();let mut alloc=allocator.lock().unwrap();
            r.flux_surface.as_ref().unwrap().debug_lighting_receiver(&r.ctx.device,&mut alloc,r.command_pool,r.ctx.queue,material.0,4)?};
        let p=Vec3::from_slice(&sample[0][..3])+Vec3::Z*0.005;
        let mut integral=0.0f32;
        for y in 0..128 {for x in 0..128 {
            let delta=light.position+Vec3::new((x as f32+0.5)/128.0-0.5,(y as f32+0.5)/128.0-0.5,0.0)-p;
            integral+=delta.z*delta.z/delta.length_squared().powi(2)/16384.0;
        }}
        let expected=integral*(1.0-light.position.distance(p)/light.range).powi(2)*4.0*0.5*sample[2][0]/std::f32::consts::PI;
        anyhow::ensure!((reference[0]-expected).abs()<expected*0.12,"area energy: hardware={hardware} actual={reference:?} reference={expected}");
        for (name,texture,factor) in [("red",red,0.9911),("grey",grey,0.21586),("black",black,0.0)] {
            light.texture=Some(texture);r.set_area_lights(&[light]);render(r,&draws,64)?;
            let actual=cache(r)?;let scene=center(&r.debug_read_viewport()?.unwrap());let camera=center(&r.debug_read_camera_preview()?.unwrap());
            eprintln!("[Flux area] hardware={hardware} texture={name} cache={actual:?} reference={reference:?} scene={scene:?} camera={camera:?}");
            anyhow::ensure!((actual[0]-reference[0]*factor).abs()<reference[0]*0.12+0.0001,"area texture missing from GI");
            if name=="red" {anyhow::ensure!(actual[1]<0.0001 && actual[2]<0.0001 && scene[0]>scene[1]+10.0 && camera[0]>camera[1]+10.0,"compressed red texture ignored");}
            if name=="black" {anyhow::ensure!(actual[..3].iter().all(|&v|v<0.0001) && scene.iter().all(|&v|v<1.0) && camera.iter().all(|&v|v<1.0),"black texture still emits");}
        }
        let mut multi=[light;8];
        for (i,l) in multi.iter_mut().enumerate() {l.texture=Some(if i==0 {red} else {black});}
        r.set_area_lights(&multi);render(r,&draws,64)?;
        let tiled=cache(r)?;
        anyhow::ensure!(tiled[0]>reference[0]*0.8 && tiled[0]<reference[0]*1.2 && tiled[1]<0.0001 && tiled[2]<0.0001,
            "area atlas tile leakage: {tiled:?}");
        light.texture=None;light.up=Vec3::Y;r.set_area_lights(&[light]);render(r,&draws,32)?;
        anyhow::ensure!(cache(r)?[0]<0.0001,"one-sided area emits backwards");
        light.two_sided=true;r.set_area_lights(&[light]);render(r,&draws,48)?;
        anyhow::ensure!(cache(r)?[0]>reference[0]*0.8,"two-sided area failed");
        light.shape=AreaShape::Disc;r.set_area_lights(&[light]);render(r,&draws,48)?;
        let disc=cache(r)?[0];anyhow::ensure!(disc>reference[0]*0.65 && disc<reference[0]*0.98,"disc energy {disc} vs rect {}",reference[0]);
        light.shape=AreaShape::Rect;light.up=Vec3::NEG_Y;light.two_sided=false;
        let mut blocker=draw(mesh,material,Mat4::from_scale_rotation_translation(Vec3::new(3.0,3.0,0.05),glam::Quat::IDENTITY,Vec3::new(0.0,0.0,0.8)),-1);
        // Different material identity so the debug read still names the receiver.
        blocker.material=lit(r,[0.0;4],[0.0;3],1.0,0.0);blocker.gi_source=Some(GiSource {object:0xaea1,slot:0});
        blocker.shadow_casting=crate::ShadowCasting::On;
        draws.push(blocker);r.gi_set_mesh_sdfs(&[make_field(panel),make_field(blocker)])?;
        // Change light energy to invalidate old direct histories as well.
        light.intensity=4.1;r.set_area_lights(&[light]);render(r,&draws,48)?;
        let blocked=cache(r)?;anyhow::ensure!(blocked[0]<reference[0]*0.01,"area shadows leak: {blocked:?}");
        draws.pop();r.gi_set_mesh_sdfs(&[make_field(panel)])?;light.intensity=4.0;
    }
    if r.ctx.ray_tracing() { reflection(r,red,black)?; }
    r.set_area_lights(&[]);r.gi_set_mesh_sdfs(&[])?;Ok(())
}

fn reflection(r:&mut Renderer,red:crate::TextureHandle,black:crate::TextureHandle)->anyhow::Result<()> {
    let mesh=r.upload_mesh(&cube())?;
    let mirror=lit(r,[0.95;4],[0.0;3],0.02,1.0);let wall=lit(r,[0.8;4],[0.0;3],1.0,0.0);
    let draws=[draw(mesh,mirror,Mat4::from_scale_rotation_translation(Vec3::new(8.0,0.2,8.0),glam::Quat::IDENTITY,Vec3::new(0.0,-0.1,0.0)),-1),
        draw(mesh,wall,Mat4::from_scale_rotation_translation(Vec3::new(2.0,3.0,0.2),glam::Quat::IDENTITY,Vec3::new(0.0,1.5,-2.0)),-1)];
    r.set_gi_enabled(false);
    r.set_flux_settings(FluxSettings {rt_trace:true,reflection_mode:2,rt_samples:1,rt_bounces:1,ssr_enabled:false,
        ray_lighting_mode:crate::RayLightingMode::HitLightingForReflections,..Default::default()});
    let eye=Vec3::new(0.0,2.0,4.0);let cam=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),
        proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let mut input=crate::cascade_tests::input(cam);input.camera_preview=Some(cam);input.viewport_extent=Some([640,360]);input.draws=&draws;
    input.light.intensity=0.0;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;input.draw_skybox=false;
    input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
    let mut light=AreaLightInstance {shape:AreaShape::Rect,position:Vec3::new(0.0,1.0,-1.0),right:Vec3::X,up:Vec3::NEG_Y,
        half_size:[0.4,0.4],color:[1.0;3],intensity:8.0,range:5.0,two_sided:false,texture:Some(red)};
    for dark in [false,true] {
        light.texture=Some(if dark {black} else {red});r.set_area_lights(&[light]);
        for _ in 0..40 {r.render(&input)?;}
        let scene=center(&r.debug_read_viewport()?.unwrap());let camera=center(&r.debug_read_camera_preview()?.unwrap());
        eprintln!("[Flux area reflected hit] dark={dark} Scene={scene:?} Camera={camera:?}");
        for sample in [scene,camera] {
            if dark {anyhow::ensure!(sample.iter().all(|&v|v<1.0),"black area texture lights reflection: {sample:?}");}
            else {anyhow::ensure!(sample[0]>sample[1]+10.0 && sample[0]>sample[2]+10.0,"area missing from reflected hit: {sample:?}");}
        }
    }
    Ok(())
}
