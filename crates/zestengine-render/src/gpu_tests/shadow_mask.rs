//! Raster shadow coverage through the real Renderer, including cached layers and both views.
use glam::{Mat4, Vec3};
use anyhow::Context as _;
use crate::{AlphaMode, CameraData, FluxSettings, MaterialFeatures, Renderer, ShadowCasting};
use super::planar::{draw, lit};

fn panel(z: f32, alpha: f32, reverse: bool) -> crate::MeshData {
    let vertices = [(-2.0,-2.0,0.0,0.0),(2.0,-2.0,1.0,0.0),(2.0,2.0,1.0,1.0),(-2.0,2.0,0.0,1.0)]
        .map(|(x,y,u,v)| crate::Vertex { position:[x,y,z],normal:[0.0,0.0,1.0],uv:[u,v],
            color:[1.0,1.0,1.0,alpha],tangent:[1.0,0.0,0.0,1.0],..Default::default() }).to_vec();
    crate::MeshData {vertices,indices:if reverse {vec![0,2,1,0,3,2]} else {vec![0,1,2,0,2,3]},has_lightmap_uv:false}
}
fn patch(image: &(u32,u32,Vec<u8>), cam: CameraData, x: f32) -> f32 {
    let (w,h,data)=image; let p=(cam.proj*cam.view).project_point3(Vec3::new(x,0.0,0.0));
    let x=((p.x*0.5+0.5)* *w as f32) as i32;let y=((0.5-p.y*0.5)* *h as f32) as i32;
    let mut sum=0.0;for j in -2..=2 {for i in -2..=2 {sum+=data[(((y+j) as u32*w+(x+i) as u32)*4) as usize] as f32;}}sum/25.0
}
fn clear_texture(r: &mut Renderer, texture: crate::TextureHandle, color: [f32;4]) -> anyhow::Result<()> {
    use ash::vk;
    let t=&r.textures[texture.0];let device=&r.ctx.device;
    crate::alloc::one_time_submit(device,r.command_pool,r.ctx.queue,|cb|unsafe {
        let range=vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(vk::REMAINING_MIP_LEVELS).layer_count(1);
        let barrier=|old,new,src,dst| [vk::ImageMemoryBarrier2::default().image(t.image).subresource_range(range)
            .old_layout(old).new_layout(new).src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS).src_access_mask(src).dst_access_mask(dst)];
        device.cmd_pipeline_barrier2(cb,&vk::DependencyInfo::default().image_memory_barriers(&barrier(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,vk::AccessFlags2::MEMORY_READ|vk::AccessFlags2::MEMORY_WRITE,vk::AccessFlags2::TRANSFER_WRITE)));
        device.cmd_clear_color_image(cb,t.image,vk::ImageLayout::TRANSFER_DST_OPTIMAL,&vk::ClearColorValue {float32:color},&[range]);
        device.cmd_pipeline_barrier2(cb,&vk::DependencyInfo::default().image_memory_barriers(&barrier(vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,vk::AccessFlags2::TRANSFER_WRITE,vk::AccessFlags2::SHADER_SAMPLED_READ)));
    })?;
    r.notify_texture_updated(texture);Ok(())
}
pub(super) fn run(r: &mut Renderer) -> anyhow::Result<()> {
    eprintln!("[Raster shadow mask] Renderer ready; uploading fixture");
    let old_gi=r.gi_enabled;let old_flux=r.flux_settings;let old_cull=r.gpu_cull_settings();
    r.set_gi_enabled(false);r.set_flux_settings(FluxSettings {reflection_mode:0,ssr_enabled:false,..Default::default()});
    let receiver=r.upload_mesh(&panel(0.0,1.0,false))?;
    let normal=r.upload_mesh(&panel(2.0,1.0,false))?;
    let reverse=r.upload_mesh(&panel(2.0,1.0,true))?;
    let half=r.upload_mesh(&panel(2.0,0.5,false))?;
    let white=lit(r,[1.0;4],[0.0;3],1.0,0.0);
    let mask=lit(r,[1.0;4],[0.0;3],1.0,0.0);
    let texture=r.upload_texture(&crate::TextureData {width:32,height:1,
        pixels:(0..32).flat_map(|x|[255,255,255,if x<16 {0} else {255}]).collect(),srgb:false,hdr:false})?;
    let opacity=r.upload_texture(&crate::TextureData {width:32,height:1,
        pixels:(0..32).flat_map(|x|[if x<16 {0} else {255},255,255,255]).collect(),srgb:false,hdr:false})?;
    let eye=Vec3::new(0.0,0.0,5.0);let cam=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),
        proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,30.0)};
    let light=[crate::LightInstance {kind:crate::LightKind::Directional,direction:Vec3::NEG_Z,position:Vec3::ZERO,
        color:[1.0;3],intensity:4.0,range:0.0,spot_inner_deg:30.0,spot_outer_deg:45.0,cast_shadows:true,
        soft_shadows:false,radius:0.0,shadow_bias:0.0,baked:false}];
    let mut draws=[draw(receiver,white,Mat4::IDENTITY,-1),draw(normal,mask,Mat4::IDENTITY,-1)];
    draws[1].shadow_casting=ShadowCasting::ShadowsOnly;
    let mut slots=[None;16];slots[0]=Some(texture);r.set_material_textures(mask,&slots);
    let features=|alpha_mode|MaterialFeatures {alpha_mode,..Default::default()};
    let mut baseline=[[0.0;2];2];
    let frag=super::vertex_tess::glslc_fragment("#version 450\nlayout(location=0) out vec4 c; void main(){c=vec4(1);}");
    let hook=r.register_shader_with_vertex(&frag,r#"
struct VertexData {
    vec3 position; vec3 normal; vec4 tangent; vec2 uv; vec2 uv1; vec4 color;
    mat4 model; vec3 world_position; vec3 world_offset; float time; int vertex_index;
};
void zestengine_vertex(inout VertexData v) { v.uv.x += 0.5; v.color.a *= clamp(2.0-v.time,0.0,1.0); }
"#)?;
    for case in 0..18 {
        let expected=match case {
            0=>{draws[1].shadow_casting=ShadowCasting::Off;[true,true]},
            1=>{draws[1].shadow_casting=ShadowCasting::ShadowsOnly;[false,false]},
            2=>{r.set_material_features(mask,features(AlphaMode::Cutout));r.material_params_mut(mask).alpha_cutoff=0.5;[true,false]},
            3=>{draws[1].mesh=reverse;[true,false]},
            4=>{slots[0]=None;slots[4]=Some(opacity);r.set_material_textures(mask,&slots);[true,false]},
            5=>{r.material_params_mut(mask).albedo_offset=[0.5,0.0];[false,true]},
            6=>{r.material_params_mut(mask).albedo_offset=[0.0;2];r.material_params_mut(mask).base_scroll=[0.5,0.0];[false,true]},
            7=>{r.material_params_mut(mask).base_scroll=[0.0;2];draws[1].mesh=half;r.material_params_mut(mask).base_color[3]=0.8;
                r.material_params_mut(mask).alpha_cutoff=0.3;[true,false]},
            8=>{r.material_params_mut(mask).base_color[3]=0.4;[true,true]},
            9=>{r.set_material_features(mask,MaterialFeatures {per_instance:true,..features(AlphaMode::Cutout)});
                draws[1].instance_color=Some([1.0,1.0,1.0,0.8]);[true,false]},
            10=>{draws[1].instance_color=Some([1.0,1.0,1.0,0.4]);[true,true]},
            11=>{r.set_material_features(mask,features(AlphaMode::Cutout));r.material_params_mut(mask).base_color[3]=1.0;
                r.material_params_mut(mask).alpha_cutoff=0.5;draws[1].mesh=normal;draws[1].instance_color=None;
                r.material_params_mut(mask).tess=crate::TessParams {enabled:true,scale:0.0,phong:0.0,..Default::default()};[true,false]},
            12=>{r.material_params_mut(mask).tess.enabled=false;clear_texture(r,opacity,[1.0;4])?;[false,false]},
            13=>{clear_texture(r,opacity,[0.0;4])?;[true,true]},
            14=>{r.set_material_features(mask,features(AlphaMode::Opaque));r.material_params_mut(mask).base_color[3]=0.0;[false,false]},
            15=>{slots[0]=Some(texture);slots[4]=None;r.set_material_textures(mask,&slots);
                r.set_material_features(mask,features(AlphaMode::Cutout));r.material_params_mut(mask).base_color[3]=1.0;
                r.set_material_shader(mask,Some(hook));[false,true]},
            16=>[true,true],
            _=>{r.material_params_mut(mask).tess=crate::TessParams {enabled:true,scale:0.2,phong:0.0,..Default::default()};[false,true]},
        };
        eprintln!("[Raster shadow mask] starting case={case}");
        r.upload_material_fx(mask);
        let mut input=crate::cascade_tests::input(cam);input.viewport_extent=Some([640,360]);input.camera_preview=Some(cam);
        input.draws=&draws;input.lights=&light;input.light.intensity=0.0;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;
        input.draw_skybox=false;input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;input.time=if case==16 {2.0} else {1.0};
        for gpu in [false,true] {
            r.set_gpu_cull_settings(crate::GpuCullSettings {enabled:gpu,..Default::default()});
            for frame in 0..2 {r.render(&input).with_context(||format!("raster mask case={case} GPU={gpu} frame={frame}"))?;}
            let signatures=r.shadow_layer_sig;
            r.render(&input)?;
            if case!=6 && case!=11 && case<15 {assert_eq!(signatures,r.shadow_layer_sig,"static mask invalidates cached shadows case {case}");}
            let images=[r.debug_read_viewport()?.unwrap(),r.debug_read_camera_preview()?.unwrap()];
            let values=images.map(|im|[-0.7,0.7].map(|x|patch(&im,cam,x)));
            eprintln!("[Raster shadow mask] case={case} GPUcull={gpu} Scene/Camera={values:?}");
            if case==0 {baseline=values;assert!(values.iter().flatten().all(|v|*v>100.0));}
            for v in 0..2 {for side in 0..2 {let ratio=values[v][side]/baseline[v][side].max(1.0);
                assert!(if expected[side] {ratio>0.9} else {ratio<0.1},"mask case{case} GPU={gpu}: {values:?} vs baseline{baseline:?}, expected {expected:?}");}}
        }
    }
    r.set_gpu_cull_settings(old_cull);r.set_gi_enabled(old_gi);r.set_flux_settings(old_flux);
    eprintln!("[Raster shadow mask] coverage, both windings/views, material/vertex/instance alpha, opacity maps, UV animation, tessellation, live texture content and static caching passed");
    Ok(())
}
