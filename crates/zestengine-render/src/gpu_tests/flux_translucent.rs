//! Emissive blend transport: opacity weights the source and admits light behind
//! it. Read physical cache radiance and both reflection views, without exposure.
use glam::{Mat4,Vec3};
use crate::{AlphaMode,CameraData,FluxSettings,Renderer};
use super::planar::{cube,draw,lit};
pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    for hardware in [false,true] {
        if hardware && !r.ctx.ray_tracing() {continue;}
        cases(r,hardware,None,AlphaMode::Blend)?;
    }
    Ok(())
}
pub(super) fn gather_energy(r:&mut Renderer)->anyhow::Result<()> {
    let mut failures=Vec::new();
    for hardware in [false,true] {
        if hardware && !r.ctx.ray_tracing() {continue;}
        let cached=cases(r,hardware,None,AlphaMode::Blend)?;
        let traced=cases(r,hardware,Some(false),AlphaMode::Blend)?;
        for view in 0..2 {
            for case in [0,2,3,8] {
                let error=(cached[case].1[view]-traced[case].1[view])/traced[0].1[view];
                eprintln!("[Flux gather absolute energy] hardware={hardware} view={view} case={case} cached={} traced={} baseline_relative_error={error}",cached[case].1[view],traced[case].1[view]);
                if error.abs()>0.08 {failures.push((hardware,view,cached[case].1[view],traced[case].1[view]));}
            }
            let cached_ratio=cached[3].1[view]/cached[0].1[view];
            let traced_ratio=traced[3].1[view]/traced[0].1[view];
            eprintln!("[Flux blended gather energy] hardware={hardware} view={view} cached_ratio={cached_ratio} traced_ratio={traced_ratio}");
            if (traced_ratio-0.5).abs()>0.08 || (cached_ratio-traced_ratio).abs()>0.08 {
                failures.push((hardware,view,cached_ratio,traced_ratio));
            }
        }
    }
    r.gpu_gi.as_mut().unwrap().radiance_cache.enabled_override=None;
    assert!(failures.is_empty(),"radiance cache changes partial-opacity gather energy: {failures:?}");
    Ok(())
}
pub(super) fn masked(r:&mut Renderer)->anyhow::Result<()> {
    for hardware in [false,true] {
        if hardware && !r.ctx.ray_tracing() {continue;}
        for mode in [AlphaMode::Cutout,AlphaMode::Transclipping] {
            cases(r,hardware,None,mode)?;
        }
    }
    Ok(())
}
type TransportSamples=Vec<([f32;3],[f32;2],[f32;2])>;
fn cases(r:&mut Renderer,hardware:bool,radiance_cache:Option<bool>,mode:AlphaMode)->anyhow::Result<TransportSamples> {
    r.set_anti_aliasing(crate::AntiAliasing::None);r.set_area_lights(&[]);r.gi_set_emitters(&[]);
    let mesh=r.upload_mesh(&cube())?;
    let floor=lit(r,[0.8;4],[0.0;3],1.0,0.0);
    let red=lit(r,[0.0,0.0,0.0,1.0],[4.0,0.0,0.0],1.0,0.0);
    let green=lit(r,[0.0,0.0,0.0,1.0],[0.0,4.0,0.0],1.0,0.0);
    r.set_material_features(green,crate::MaterialFeatures {alpha_mode:mode,emission:true,..Default::default()});
    r.material_params_mut(green).alpha_cutoff=0.5;
    // A single front-facing pane avoids multiplying opacity by a second face.
    let positions=[[-20.0,2.0,-20.0],[20.0,2.0,-20.0],[20.0,2.0,20.0],[-20.0,2.0,20.0]];
    let pane_data=crate::MeshData {vertices:positions.into_iter().enumerate().map(|(i,p)|crate::Vertex {
        position:p,normal:[0.0,-1.0,0.0],uv:[[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]][i],color:[1.0;4],..Default::default()
    }).collect(),indices:vec![0,1,2,0,2,3],has_lightmap_uv:false};
    let panel=r.upload_mesh(&pane_data)?;
    let mut layered_data=crate::MeshData {vertices:pane_data.vertices.clone(),indices:pane_data.indices.clone(),has_lightmap_uv:false};
    layered_data.vertices.extend(pane_data.vertices.iter().map(|v|{let mut v=*v;v.position=v.position.map(|x|x*1.5);v}));
    layered_data.indices.extend(pane_data.indices.iter().map(|i|i+4));
    let layered_panel=r.upload_mesh(&layered_data)?;
    let mut draws=[draw(mesh,floor,Mat4::from_scale_rotation_translation(Vec3::new(8.0,0.1,8.0),glam::Quat::IDENTITY,Vec3::new(0.0,-0.1,0.0)),-1),
        draw(mesh,red,Mat4::from_scale_rotation_translation(Vec3::new(80.0,0.02,80.0),glam::Quat::IDENTITY,Vec3::new(0.0,4.0,0.0)),-1),
        draw(panel,green,Mat4::IDENTITY,-1)];
    for (i,d) in draws.iter_mut().enumerate() {d.gi_source=Some(crate::GiSource {object:0xb1e0+i as u128,slot:0});}
    let cube_field=super::flux::cube_distance_field();
    let positions:Vec<_>=positions.into_iter().map(Vec3::from_array).collect();
    let pane_field=std::sync::Arc::new(crate::sdf::generate_sdf(&positions,&[0,1,2,0,2,3],32,0.1));
    let layered_positions:Vec<_>=layered_data.vertices.iter().map(|v|Vec3::from_array(v.position)).collect();
    // The two planes are only one metre apart across a sixty-metre mesh.
    // A32-voxel field erases them before opacity traversal even starts.
    let layered_field=std::sync::Arc::new(crate::sdf::generate_sdf(&layered_positions,&layered_data.indices,128,0.1));
    for y in [2.0,3.0] {
        assert!(layered_field.sample(Vec3::new(0.0,y-0.1,0.0))>0.0
            && layered_field.sample(Vec3::new(0.0,y+0.1,0.0))<0.0,
            "layered opacity fixture must represent both emitting planes in its distance field");
    }
    r.gi_set_gdf_identified(&[100.0;8],&[0;8],[2;3],[-24.0;3],[24.0;3],&[],&[])?;r.gi_set_mesh_sdfs(&[])?;
    r.gpu_gi.as_mut().unwrap().radiance_cache.enabled_override=radiance_cache;
    r.invalidate_screen_gi();
    let camera=|eye|CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let cams=[camera(Vec3::new(0.0,1.0,3.0)),camera(Vec3::new(1.0,1.1,3.0))];
    let settings=FluxSettings {rt_trace:hardware,gi_screen_traces:false,reflection_screen_traces:false,samples:64,bounces:2,
        reflection_mode:0,ssr_enabled:false,..Default::default()};
    // Complementary maps: only the transparent half emits. It must contribute
    // no green light, which a separately averaged opacity/emission pair cannot reproduce.
    let opacity=r.upload_texture(&crate::TextureData {width:128,height:1,
        pixels:(0..128).flat_map(|x|[if x<64 {0} else {255},255,255,255]).collect(),srgb:false,hdr:false})?;
    let emission=r.upload_texture(&crate::TextureData {width:128,height:1,
        pixels:(0..128).flat_map(|x|[if x<64 {255} else {0};4]).collect(),srgb:false,hdr:false})?;
    let cases=[(0.0,false,false,false,false),(1.0,false,false,false,false),(0.25,false,false,false,false),(0.5,false,false,false,false),
        (0.0,false,false,false,false),(1.0,true,false,false,false),(1.0,false,true,false,false),(1.0,false,true,true,false),
        (0.5,false,false,false,true)];
    let cases=&cases[..if mode==AlphaMode::Blend {9} else {6}];
    let mut results=Vec::new();
    for &(alpha,textured,mirrored,cull_front,layered) in cases {
        draws[2].mesh=if layered {layered_panel} else {panel};
        draws[2].transform=Mat4::from_scale(Vec3::new(if mirrored {-1.0} else {1.0},1.0,1.0));
        r.set_material_features(green,crate::MaterialFeatures {alpha_mode:mode,emission:true,cull_front,..Default::default()});
        let fields:Vec<_>=draws.iter().enumerate().map(|(i,d)|crate::GiMeshSdf {
            gi_source:d.gi_source,world_to_local:d.transform.inverse(),volume:if i==2 {if layered {layered_field.clone()} else {pane_field.clone()}} else {cube_field.clone()},
            material:crate::GpuGiMaterial {albedo:if i==0 {[0.8;3]} else {[0.0;3]},
                emission:match i {1=>[4.0,0.0,0.0],2=>[0.0,4.0,0.0],_=>[0.0;3]},metallic:0.0,roughness:1.0},in_global_field:false,
        }).collect();
        r.gi_set_mesh_sdfs(&fields)?;
    let mut input=crate::cascade_tests::input(cams[0]);input.camera_preview=Some(cams[1]);input.viewport_extent=Some([323,181]);input.draws=&draws;
    input.light.intensity=0.0;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;input.draw_skybox=false;
    input.postfx.flux.scene_lighting_quality=Some(4.0);input.preview_postfx.flux.scene_lighting_quality=Some(4.0);

        let mut slots=[None;16];if textured {slots[3]=Some(emission);slots[4]=Some(opacity);}
        r.set_material_textures(green,&slots);
        r.material_params_mut(green).base_color[3]=alpha;
        r.material_params_mut(floor).metallic=0.0;r.material_params_mut(floor).roughness=1.0;
        r.set_gi_enabled(true);r.set_flux_settings(settings);
        let mut cache=[0.0f32;3];let mut gi=[0.0;2];
        for frame in 0..96 {
            r.render(&input)?;
            if frame>=64 && frame%4==3 {
                let allocator=r.allocator();let sample=r.flux_surface.as_ref().unwrap().debug_lighting(&r.ctx.device,
                    &mut allocator.lock().unwrap(),r.command_pool,r.ctx.queue,floor.0,2)?;
                for c in 0..3 {cache[c]+=sample[1][c]/8.0;}
                for view in 0..2 {gi[view]+=super::flux_radiance::hdr(r,view,cams[view])?/8.0;}
            }
        }
        if mode==AlphaMode::Blend && alpha==0.0 && radiance_cache.is_none() {
            for view in 0..2 {
                let (_,work,_)=r.gpu_gi.as_ref().unwrap().radiance_cache.debug(view);
                let bytes=super::flux_radiance::read(r,work,3*4,4)?;
                let hits=u32::from_ne_bytes(bytes[..4].try_into().unwrap());
                assert!(hits>0,"opaque reference must actually exercise the radiance cache in view {view}");
            }
        }
        if !hardware && alpha==1.0 && !textured && !mirrored && !layered {
            let points:Vec<_>=(-19..=19).flat_map(|x|(-19..=19).map(move |z|[x as f32,1.998,z as f32,0.0])).collect();
            let allocator=r.allocator();let hits=r.flux_surface.as_ref().unwrap().debug_opacity_samples(&r.ctx.device,
                &mut allocator.lock().unwrap(),r.command_pool,r.ctx.queue,green.0,&points);
            let missing:Vec<_>=points.iter().zip(&hits).filter(|(_,v)|v[0]<0.999).collect();
            eprintln!("[Flux opacity coverage] tested={} missing={} examples={:?}",points.len(),missing.len(),&missing[..missing.len().min(12)]);
        }
        r.set_gi_enabled(false);r.material_params_mut(floor).metallic=1.0;r.material_params_mut(floor).roughness=0.02;
        r.set_flux_settings(FluxSettings {reflection_mode:2,rt_samples:4,rt_bounces:1,..settings});
        let mut reflection=[0.0;2];
        for frame in 0..48 {r.render(&input)?;if frame>=32 && frame%4==3 {
            for (view, view_value) in reflection.iter_mut().enumerate() {let pixels=super::reflection_denoise::history(r,view,Vec3::ZERO,false)?;
                *view_value+=pixels.iter().sum::<f32>()/(pixels.len() as f32*4.0);}
        }}
        eprintln!("[Flux blended transport] hardware={hardware} mode={mode:?} radiance_cache={radiance_cache:?} alpha={alpha} textured={textured} mirrored={mirrored} cull_front={cull_front} layered={layered} cache={cache:?} GI={gi:?} reflection={reflection:?}");
        results.push((cache,gi,reflection));
    }
    assert!(results[0].2.iter().all(|v|*v>1.0),"reflection fixture misses background source: {results:?}");
    assert!(results[0].0[0]>1.0 && results[1].0[1]>1.0,"missing reference emitter: {results:?}");
    for (i,&(alpha,textured,mirrored,cull_front,layered)) in cases.iter().enumerate() {
        let alpha=if mirrored && !cull_front {0.0} else if textured {0.5} else if layered {1.0-(1.0-alpha)*(1.0-alpha)} else if mode!=AlphaMode::Blend {if alpha>=0.5 {1.0} else {0.0}} else {alpha};
        let emitted=if textured {0.0} else {alpha};
        let red=results[i].0[0]/results[0].0[0];let green=results[i].0[1]/results[1].0[1];
        if mode!=AlphaMode::Blend && !textured && alpha>=0.5 {
            assert!(red<0.002 && green>0.98,"a closed scalar cutout must not leak, including alpha exactly at cutoff: hardware={hardware} mode={mode:?} case={i} red={red} green={green}");
            for view in 0..2 {
                assert!(results[i].1[view]<results[0].1[view]*0.002 && results[i].2[view]<results[0].2[view]*0.002,
                    "closed cutout leaked in GI/reflection: hardware={hardware} mode={mode:?} case={i} view={view} result={:?}",results[i]);
            }
        }
        assert!((red-(1.0-alpha)).abs()<0.12 && (green-emitted).abs()<if textured {0.02} else {0.12},"incorrect alpha energy: alpha={alpha}, {results:?}");
        for view in 0..2 {
            assert!((results[i].1[view]/results[0].1[view]-(1.0-alpha)).abs()<0.18,"GI failed to transmit source behind blend: {results:?}");
            // The offset Camera's mirror ray lands wholly in the clear half;
            // Scene straddles the map boundary. Diffuse GI averages both halves.
            let reflected=if textured && view==1 {1.0} else {1.0-alpha};
            assert!((results[i].2[view]/results[0].2[view]-reflected).abs()<0.15,"reflection failed to transmit source behind blend: {results:?}");
        }
    }
    Ok(results)
}
