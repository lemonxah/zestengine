//! Opaque/masked emitters block from either side, but single-sided materials
//! only emit into their world-winding front hemisphere, including mirrored
//! instances. No analytic light/proxy is used.
use std::sync::Arc;
use glam::{Mat4, Vec3};
use crate::{CameraData, FluxSettings, GiSource, GpuGiMaterial, Renderer};
use super::planar::{cube, draw, lit};

pub(super) fn run(r: &mut Renderer) -> anyhow::Result<()> {
    run_fields(r,false)
}
pub(super) fn coarse(r: &mut Renderer) -> anyhow::Result<()> {
    run_fields(r,true)
}
fn run_fields(r: &mut Renderer, coarse: bool) -> anyhow::Result<()> {
    r.set_area_lights(&[]);
    r.gi_set_emitters(&[]);
    r.set_gi_enabled(true);
    r.set_anti_aliasing(crate::AntiAliasing::None);
    let ground = r.upload_mesh(&cube())?;
    let receiver = lit(r, [0.8;4], [0.0;3], 1.0, 0.0);
    let emitter = lit(r, [0.0,0.0,0.0,1.0], [4.0,1.0,0.5], 1.0, 0.0);
    r.set_material_features(emitter, crate::MaterialFeatures {emission:true, ..Default::default()});
    let floor = draw(ground, receiver, Mat4::from_scale_rotation_translation(
        Vec3::new(8.0,0.2,8.0), glam::Quat::IDENTITY, Vec3::new(0.0,-0.1,0.0)), -1);
    let eye = Vec3::new(0.0,0.6,3.0);
    let camera = CameraData {position:eye, view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),
        proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let positions=[Vec3::new(-8.0,1.0,-8.0),Vec3::new(8.0,1.0,-8.0),
        Vec3::new(8.0,1.0,8.0),Vec3::new(-8.0,1.0,8.0)];
    for hardware in [false,true] {
        if coarse && hardware {continue;}
        if hardware && !r.ctx.ray_tracing() {continue;}
        let settings=FluxSettings {rt_trace:hardware, gi_screen_traces:false,
            samples:64,bounces:2,reflection_mode:0,ssr_enabled:false,..Default::default()};
        for alpha_mode in [crate::AlphaMode::Opaque,crate::AlphaMode::Cutout] {
        if coarse && alpha_mode!=crate::AlphaMode::Opaque {continue;}
        let mut readings=Vec::new();let mut expected=Vec::new();
        for mirrored in [false,true] {
        for (facing,double_sided,cull_front) in [(-1.0,false,false),(1.0,false,false),(1.0,true,false),(1.0,false,true),(-1.0,false,true)] {
            r.set_gi_enabled(true);r.set_flux_settings(settings);
            r.material_params_mut(receiver).metallic=0.0;r.material_params_mut(receiver).roughness=1.0;
            r.set_material_features(emitter,crate::MaterialFeatures {emission:true,double_sided,cull_front,alpha_mode,..Default::default()});
            let indices=if facing>0.0 {vec![0,2,1,0,3,2]} else {vec![0,1,2,0,2,3]};
            let data=crate::MeshData {vertices:positions.iter().map(|p|crate::Vertex {
                position:p.to_array(),normal:[0.0,facing,0.0],color:[1.0;4],..Default::default()
            }).collect(),indices:indices.clone(),has_lightmap_uv:false};
            let mesh=r.upload_mesh(&data)?;
            let mut panel=draw(mesh,emitter,Mat4::from_scale(Vec3::new(if mirrored {-1.0} else {1.0},1.0,1.0)),-1);
            panel.gi_source=Some(GiSource {object:0xe515+u128::from(facing>0.0),slot:0});
            let material=GpuGiMaterial {albedo:[0.0;3],emission:[4.0,1.0,0.5],metallic:0.0,roughness:1.0};
            let volume=Arc::new(crate::sdf::generate_sdf(&positions,&indices,if coarse {128} else {32},0.1));
            if coarse {
                // Keep several voxels between the receiver and panel. A
                // coarse origin bias larger than this one-metre gap would
                // start rays beyond the emitter and invalidate a facing test.
                let cell=(volume.max-volume.min)/Vec3::from_array(volume.dims.map(|n|(n-1) as f32));
                assert!(cell.max_element()<0.2,"coarse facing fixture cannot resolve its receiver/emitter gap");
                // This panel is symmetric about X, so mirroring X leaves its
                // distance samples unchanged but reverses its world winding.
                // No detail field may silently repair the coarse hit normal.
                r.gi_set_mesh_sdfs(&[])?;
                r.gi_set_gdf_identified(&volume.data,&vec![0;volume.data.len()],volume.dims,
                    volume.min.to_array(),volume.max.to_array(),&[material],&[panel.gi_source])?;
            } else {
                r.gi_set_gdf(&[100.0;512],&[0;512],[8;3],[-10.0;3],[10.0;3],&[material]);
                r.gi_set_mesh_sdfs(&[crate::GiMeshSdf {gi_source:panel.gi_source,
                world_to_local:panel.transform.inverse(),
                volume,
                material,in_global_field:false}])?;
            }
            let draws=[floor,panel];
            let mut input = crate::cascade_tests::input(camera);
            input.viewport_extent=Some([323,181]); input.camera_preview=Some(camera);
            input.light.intensity=0.0; input.light.ambient=[0.0;3]; input.light.env_intensity=0.0;
            input.draw_skybox=false; input.gi_debug=6;
            input.postfx.bloom_enabled=false; input.preview_postfx.bloom_enabled=false;
            input.draws=&draws;
            for _ in 0..64 {r.render(&input)?;}
            let allocator=r.allocator();
            let cache=r.flux_surface.as_ref().unwrap().debug_lighting(&r.ctx.device,
                &mut allocator.lock().unwrap(),r.command_pool,r.ctx.queue,receiver.0,2)?;
            let mut energy=[super::flux_radiance::hdr(r,0,camera)?,super::flux_radiance::hdr(r,1,camera)?,cache[1][0],0.0,0.0];
            r.set_gi_enabled(false);input.gi_debug=0;
            r.material_params_mut(receiver).metallic=1.0;r.material_params_mut(receiver).roughness=0.02;
            r.set_flux_settings(FluxSettings {reflection_mode:2,reflection_screen_traces:false,rt_samples:1,rt_bounces:1,..settings});
            for _ in 0..12 {r.render(&input)?;}
            for view in 0..2 {
                let pixels=super::reflection_denoise::history(r,view,Vec3::ZERO,false)?;
                energy[3+view]=pixels.iter().sum::<f32>()/pixels.len() as f32;
            }
            eprintln!("[Flux emitter side] hardware={hardware} coarse={coarse} mode={alpha_mode:?} mirrored={mirrored} facing={facing} double={double_sided} cull_front={cull_front}: Scene/Camera/cache/reflections={energy:?}");
            readings.push(energy);
            expected.push(double_sided || ((facing*(if mirrored {-1.0} else {1.0})<0.0)^cull_front));
        }
        }
        for view in 0..5 {
            anyhow::ensure!(readings[0][view]>0.1,"front-facing emitter failed: {readings:?}");
            for (case,lit) in expected.iter().copied().enumerate() {
                anyhow::ensure!(if lit {readings[case][view]>readings[0][view]*0.7} else {readings[case][view]<readings[0][view]*0.02},
                    "emission hemisphere mismatch: hardware={hardware} mode={alpha_mode:?} case={case} view={view} expected_lit={lit} {readings:?}");
            }
        }
        }
    }
    r.gi_set_mesh_sdfs(&[])?;
    Ok(())
}
