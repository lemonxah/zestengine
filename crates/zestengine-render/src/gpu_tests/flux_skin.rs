//! Actual posed/morphed GPU vertices in budgeted Flux captures.
use glam::{Mat4,Vec3};
use crate::{CameraData,FluxSettings,GiSource,GpuGiMaterial,Renderer};
use super::planar::{cube,draw,lit};

pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    r.set_area_lights(&[]);r.set_gi_enabled(true);r.set_anti_aliasing(crate::AntiAliasing::None);
    let mut data=cube();
    for v in &mut data.vertices {v.joints=[u32::from(v.position[0]>0.0),0,0,0];v.weights=[1.0,0.0,0.0,0.0];}
    let morph:Vec<_>=(0..data.vertices.len()).map(|i|(i as u32,[0.0,0.0,0.4],[0.0;3])).collect();
    let mesh=r.upload_mesh_gpu_skinned_morphed(&data,2,&[morph.as_slice()])?;
    let material=lit(r,[0.6,0.3,0.2,1.0],[2.0,1.0,0.5],1.0,0.0);
    r.set_material_features(material,crate::MaterialFeatures {emission:true,..Default::default()});
    let mut receiver=draw(mesh,material,Mat4::from_scale(Vec3::new(4.0,4.0,1.0)),-1);
    receiver.gi_source=Some(GiSource {object:0x5c1a,slot:0});let draws=[receiver];
    let eye=Vec3::new(0.0,0.0,5.0);
    let camera=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let mut input=crate::cascade_tests::input(camera);input.camera_preview=Some(camera);input.viewport_extent=Some([323,181]);
    input.draws=&draws;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;input.light.intensity=0.0;
    r.gi_set_mesh_sdfs(&[])?;r.gi_set_emitters(&[]);
    r.gi_set_gdf(&[100.0;512],&[0;512],[8;3],[-20.0;3],[20.0;3],&[GpuGiMaterial {albedo:[0.0;3],emission:[0.0;3],metallic:0.0,roughness:1.0}]);
    let mut probe=if r.ctx.ray_tracing() {Some(super::flux_geometry::Probe::new(r)?)} else {None};
    for hardware in [false,true] {
        if hardware && !r.ctx.ray_tracing() {continue;}
        r.set_flux_settings(FluxSettings {rt_trace:hardware,reflection_mode:0,..Default::default()});
        let mut expected=std::collections::BTreeMap::new();let mut generation=None;let mut epochs=None;
        let mut storage=None;let mut last_refits=0;let mut rebuilt=false;
        for frame in 0..144 {
            let pose_frame=if frame<64 {frame.min(39)} else {frame};
            let z=if frame<16 {0.0} else {1.0+(pose_frame as f32*0.17).sin()*0.3};
            let slope=if frame<16 {0.0} else {0.5*(pose_frame as f32*0.13).sin()};
            let weight=if frame<16 {0.0} else {(frame as f32*0.21).cos()};
            // Two bones shear the front face; the whole face also moves beyond
            // its original bind-pose depth. A signed morph adds another offset.
            if !(40..64).contains(&frame) {r.skin_mesh_gpu(mesh,&[Mat4::from_translation(Vec3::new(0.0,0.0,z-slope*0.5)),
                Mat4::from_translation(Vec3::new(0.0,0.0,z+slope*0.5))]);}
            r.set_mesh_morph_weights(mesh,&[weight]);let slot=r.frame_index;r.render(&input)?;
            if hardware {assert_eq!(r.rt_reflect.as_ref().unwrap().debug_blas_version(mesh.0),
                Some(r.mesh_versions[mesh.0]),"hardware tracing kept the bind-pose acceleration structure");
                let (address,scratch,refits)=r.rt_reflect.as_ref().unwrap().debug_blas_storage(mesh.0).unwrap();
                assert_eq!(*storage.get_or_insert((address,scratch)),(address,scratch),"animation reallocated BLAS or scratch");
                rebuilt|=last_refits>0 && refits==0;last_refits=refits;
                // Trace both sides, outside the mesh, and finite shadow rays
                // ending just before/after the moving front face. This also
                // catches stale TLAS bounds after in-place BLAS refits.
                if frame%8==0 || frame>=119 {
                    let mut rays=Vec::new();let mut distances=Vec::new();
                    for x in [-1.5,0.0,1.5] {
                        let front=0.5+z+slope*x/4.0+0.4*weight;
                        let back=front-1.0;
                        rays.push([[x,0.0,5.0,0.001],[0.0,0.0,-1.0,10.0]]);distances.push(5.0-front);
                        rays.push([[x,0.0,-5.0,0.001],[0.0,0.0,1.0,10.0]]);distances.push(back+5.0);
                        rays.push([[x,0.0,5.0,0.001],[0.0,0.0,-1.0,5.0-front-0.05]]);distances.push(-1.0);
                        rays.push([[x,0.0,5.0,0.001],[0.0,0.0,-1.0,5.0-front+0.05]]);distances.push(5.0-front);
                    }
                    rays.push([[3.0,0.0,5.0,0.001],[0.0,0.0,-1.0,10.0]]);distances.push(-1.0);
                    let hits=probe.as_mut().unwrap().trace(r,slot,&rays)?;
                    for (i,(hit,want)) in hits.iter().zip(distances).enumerate() {
                        assert!((hit[0]-want).abs()<0.002,"frame {frame}, ray {i}: stale intersection {hit:?}, expected {want}");
                    }
                }
            }
            expected.insert(r.frame_serial,(z,slope,weight));
            if frame>=16 {
                let current=std::array::from_fn::<_,2,_>(|v|r.gpu_gi.as_ref().unwrap().radiance_cache.debug(v).2);
                assert_eq!(*epochs.get_or_insert(current),current,"skin animation reset global radiance history");
            }
            let cache=r.flux_surface.as_ref().unwrap();
            assert!(cache.captures<=12 && cache.capture_texels<=12*64*64);
            let pages=cache.debug_virtual_cards(material.0,4);
            if pages.is_empty() {assert!(frame<8,"skinned surface was never captured");continue;}
            assert_eq!(pages.len(),1,"deforming domains must not publish static virtual neighbour links");
            let (id,card)=pages[0];assert_eq!(*generation.get_or_insert(id),id,"animation replaced logical card identity");
            let captured=cache.debug_capture_frame(id);assert!(r.frame_serial-captured<=8,"deforming capture starved");
            let (z,slope,weight)=expected[&captured];
            let allocator=r.allocator();let sample=cache.debug_page_material(&r.ctx.device,&mut allocator.lock().unwrap(),r.command_pool,r.ctx.queue,card,0)?;
            assert!(sample[0][3]>0.5,"deformation clipped the captured face");
            let p=Vec3::from_slice(&sample[0][..3]);
            let wanted_z=0.5+z+slope*p.x/4.0+0.4*weight;
            assert!((p.z-wanted_z).abs()<0.005,"captured stale/bind-pose geometry: {p:?}, wanted z={wanted_z}, frame={captured}");
            for c in 0..3 {assert!((sample[3][c]-[2.0,1.0,0.5][c]).abs()<0.01,"skinned capture lost emission");}
        }
        // Stop changing the pose: capture revisions must settle as they do for
        // static geometry, without a permanent per-frame refresh tax.
        if hardware {assert!(rebuilt,"periodic in-place BLAS rebuild was not exercised");}
        let paused=r.rt_reflect.as_ref().and_then(|rt|rt.debug_blas_storage(mesh.0));
        for _ in 0..24 {r.render(&input)?;}
        if hardware {assert_eq!(r.rt_reflect.as_ref().unwrap().debug_blas_storage(mesh.0),paused,"paused mesh kept rebuilding");}
        assert_eq!(r.flux_surface.as_ref().unwrap().captures,0,"stationary skin kept recapturing");
        eprintln!("[Flux skinned capture] hardware={hardware}: GPU bone shear, movement beyond bind bounds, signed morphs, actual captured geometry/emission, stable identity, paused refresh, exact ray hits/occlusion and BLAS storage reuse passed");
        input.draws=&[];r.render(&input)?;input.draws=&draws;
    }
    if let Some(probe)=probe {probe.destroy(r);}
    Ok(())
}
