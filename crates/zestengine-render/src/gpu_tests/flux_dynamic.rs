//! Animated crowd: bounded quality rebuilds without stale ray geometry, plus
//! alternating same-workload GPU measurements of refits versus full builds.
use ash::vk;
use glam::{Mat4,Vec3};
use crate::{CameraData,FluxSettings,GiSource,GpuGiMaterial,MeshData,Renderer,Vertex};
use super::planar::{draw,lit};

fn grid(n:u32)->MeshData {
    let mut vertices=Vec::new();let mut indices=Vec::new();
    for y in 0..=n {for x in 0..=n {
        let u=x as f32/n as f32;let v=y as f32/n as f32;
        vertices.push(Vertex {position:[4.0*u-2.0,4.0*v-2.0,0.0],normal:[0.0,0.0,1.0],
            uv:[u,v],joints:[0,1,0,0],weights:[1.0-u,u,0.0,0.0],..Default::default()});
    }}
    for y in 0..n {for x in 0..n {
        let a=y*(n+1)+x;let b=a+1;let c=a+n+1;let d=c+1;
        indices.extend([a,b,c,b,d,c]);
    }}
    MeshData {vertices,indices,has_lightmap_uv:false}
}

pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    if !r.ctx.ray_tracing() {return Ok(());}
    r.set_area_lights(&[]);r.set_gi_enabled(true);r.set_anti_aliasing(crate::AntiAliasing::None);
    r.set_flux_settings(FluxSettings {rt_trace:true,samples:1,reflection_mode:0,..Default::default()});
    r.gi_set_mesh_sdfs(&[])?;r.gi_set_emitters(&[]);
    r.gi_set_gdf(&[100.0;512],&[0;512],[8;3],[-30.0;3],[30.0;3],&[GpuGiMaterial {albedo:[0.0;3],
        emission:[0.0;3],metallic:0.0,roughness:1.0}]);
    let geometry=grid(128);let triangles=geometry.indices.len() as u64/3;
    let material=lit(r,[0.6,0.3,0.2,1.0],[0.5,0.25,0.1],1.0,0.0);
    r.set_material_features(material,crate::MaterialFeatures {emission:true,..Default::default()});
    let mut meshes=Vec::new();let mut draws=Vec::new();
    for i in 0..6 {
        let mesh=r.upload_mesh_gpu_skinned(&geometry,2)?;meshes.push(mesh);
        let mut d=draw(mesh,material,Mat4::from_translation(Vec3::new(i as f32*5.0-12.5,0.0,0.0)),-1);
        d.gi_source=Some(GiSource {object:0xbd600+i as u128,slot:0});draws.push(d);
    }
    // One mesh has two draw instances; it must be rebuilt only once.
    draws.push(draw(meshes[0],material,Mat4::from_translation(Vec3::new(20.0,0.0,0.0)),-1));
    let eye=Vec3::new(0.0,0.0,8.0);
    let camera=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),
        proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let mut input=crate::cascade_tests::input(camera);input.camera_preview=Some(camera);
    input.viewport_extent=Some([320,180]);input.draws=&draws;
    input.light.ambient=[0.0;3];input.light.env_intensity=0.0;input.light.intensity=0.0;
    let mut probe=super::flux_geometry::Probe::new(r)?;
    let mut storage=[None;6];let mut last_refits=[0;6];let mut rebuilt=[false;6];
    let animate=|r:&mut Renderer,phase:f32| {
        let mut expected=Vec::new();
        for (i,mesh) in meshes.iter().enumerate() {
            let a=phase+i as f32*0.3;
            let z=0.7+0.4*a.sin();let slope=0.25*(a*1.3).cos();
            r.skin_mesh_gpu(*mesh,&[Mat4::from_translation(Vec3::new(0.0,0.0,z-2.0*slope)),
                Mat4::from_translation(Vec3::new(0.0,0.0,z+2.0*slope))]);
            expected.push((z,slope));
        }
        expected
    };
    for frame in 0..132 {
        let expected=animate(r,frame as f32*0.11);let slot=r.frame_index;r.render(&input)?;
        let rt=r.rt_reflect.as_ref().unwrap();let (count,cost)=rt.debug_rebuilds();
        assert!(cost<=u64::from(crate::rt_dynamic::REBUILD_TRIANGLES));
        assert_eq!(cost,u64::from(count)*triangles,"duplicate instance spent rebuild budget");
        for (i,mesh) in meshes.iter().enumerate() {
            assert_eq!(rt.debug_blas_version(mesh.0),Some(r.mesh_versions[mesh.0]),"deferred quality rebuild left stale vertices");
            let (address,scratch,refits)=rt.debug_blas_storage(mesh.0).unwrap();
            assert_eq!(*storage[i].get_or_insert((address,scratch)),(address,scratch));
            if last_refits[i]>0 && refits==0 {rebuilt[i]=true;}
            last_refits[i]=refits;
        }
        if frame%16==0 || frame>=120 {
            let mut rays=Vec::new();let mut distances=Vec::new();
            for (i,(z,slope)) in expected.into_iter().enumerate() {for x in [-1.0,1.0] {
                rays.push([[i as f32*5.0-12.5+x,0.1,5.0,0.001],[0.0,0.0,-1.0,10.0]]);
                distances.push(5.0-z-slope*x);
            }}
            let hits=probe.trace(r,slot,&rays)?;
            for (hit,want) in hits.iter().zip(distances) {assert!((hit[0]-want).abs()<0.002,"budgeted animation ray {hit:?} expected {want}");}
        }
    }
    assert!(rebuilt.into_iter().all(|v|v),"oldest-first rebuilds starved a mesh");
    eprintln!("[Flux dynamic budget] six meshes, {} triangles, shared Scene/Camera budget, aliases, every-frame current ray hits, retained storage and eventual rebuilds passed",6*triangles);

    if r.gpu_ts_period>0.0 {
        let pool=unsafe {r.ctx.device.create_query_pool(&vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP).query_count(2),None)?};
        r.rt_reflect.as_mut().unwrap().blas_timing=Some(pool);
        let mut times=[Vec::new(),Vec::new()];
        for block in 0..8 {
            let rebuild=block%2==1;r.rt_reflect.as_mut().unwrap().force_blas_rebuild=rebuild;
            for frame in 0..10 {
                // Identical pose sequence in each mode; exclude two warmups.
                animate(r,frame as f32*0.19);r.render(&input)?;
                let mut results=[crate::TsResult::default();2];
                unsafe {r.ctx.device.get_query_pool_results(pool,0,&mut results,vk::QueryResultFlags::TYPE_64
                    |vk::QueryResultFlags::WITH_AVAILABILITY|vk::QueryResultFlags::WAIT)?;}
                let ms=crate::ts_zone_ms(results[0],results[1],r.gpu_ts_period as f64,r.gpu_ts_mask).unwrap();
                assert!(ms.is_finite() && ms>0.0);
                if frame>=2 {times[usize::from(rebuild)].push(ms);}
            }
        }
        r.rt_reflect.as_mut().unwrap().blas_timing=None;
        r.rt_reflect.as_mut().unwrap().force_blas_rebuild=false;
        // Query completion precedes the final GI passes that still reference
        // these structures. Retire the frame before destroying test resources.
        unsafe {r.ctx.device.device_wait_idle()?;r.ctx.device.destroy_query_pool(pool,None);}
        for t in &mut times {t.sort_by(f32::total_cmp);}
        eprintln!("[Flux dynamic timing] actual GPU skinning,6 meshes/{} triangles,32 samples/mode, alternating identical-pose blocks: refit median={}ms p95={}ms; full build median={}ms p95={}ms. BLAS GPU work only; both reuse storage, excludes CPU allocation and frame total.",
            6*triangles,times[0][16],times[0][30],times[1][16],times[1][30]);
    }
    let frozen:Vec<_>=meshes.iter().map(|m|r.rt_reflect.as_ref().unwrap().debug_blas_storage(m.0)).collect();
    for _ in 0..8 {r.render(&input)?;assert_eq!(r.rt_reflect.as_ref().unwrap().debug_rebuilds(),(0,0));}
    assert_eq!(meshes.iter().map(|m|r.rt_reflect.as_ref().unwrap().debug_blas_storage(m.0)).collect::<Vec<_>>(),frozen);
    unsafe {r.ctx.device.device_wait_idle()?;}
    probe.destroy(r);
    Ok(())
}
