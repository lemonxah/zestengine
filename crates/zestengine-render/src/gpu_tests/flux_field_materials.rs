//! Distance-field material publication must preserve geometry and live frames,
//! while uncaptured fallback emitters still turn on/off in both views.
use glam::{Mat4,Quat,Vec3};
use crate::{CameraData,FluxSettings,GiMeshSdf,GiSource,GpuGiMaterial,GpuGiMarch,Renderer};
use super::{planar::{cube,draw,lit},flux_radiance::hdr};

pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    let mesh=r.upload_mesh(&cube())?;
    let grey=lit(r,[0.8;4],[0.0;3],1.0,0.0);
    let floor=Mat4::from_scale_rotation_translation(Vec3::new(12.0,0.2,12.0),Quat::IDENTITY,Vec3::new(0.0,-0.1,0.0));
    // Only the receiver is rasterized. The distant emitter has no card or
    // screen history; all its illumination must come from the fallback table.
    let draws=[draw(mesh,grey,floor,-1)];
    let eye=Vec3::new(0.0,2.0,4.0);
    let cam=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),
        proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let mut input=crate::cascade_tests::input(cam);
    input.viewport_extent=Some([323,181]);input.camera_preview=Some(cam);input.draws=&draws;
    input.light.intensity=0.0;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;input.draw_skybox=false;
    input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
    r.set_gi_enabled(true);r.gi_set_emitters(&[]);r.set_area_lights(&[]);
    r.set_flux_settings(FluxSettings {rt_trace:false,samples:64,bounces:2,reflection_mode:0,ssr_enabled:false,..Default::default()});
    let dark=GpuGiMaterial {albedo:[0.0;3],emission:[0.0;3],metallic:0.0,roughness:1.0};
    let bright=GpuGiMaterial {emission:[4.0,0.4,0.1],..dark};
    let source=Some(GiSource {object:0xfd00,slot:0});
    let lo=Vec3::new(-12.0,0.0,-12.2);let hi=Vec3::new(12.0,16.0,-11.8);
    let center=(lo+hi)*0.5;let half=(hi-lo)*0.5;
    let transform=Mat4::from_scale_rotation_translation(half*2.0,Quat::IDENTITY,center);
    let volume=super::flux::cube_distance_field();
    let mut fields=[GiMeshSdf {gi_source:source,world_to_local:transform.inverse(),volume,
        material:dark,in_global_field:false}];
    let min=Vec3::new(-16.0,-2.0,-16.0);let max=Vec3::new(16.0,18.0,8.0);
    let dims=[48,32,48];let mut distances=Vec::new();
    for z in 0..dims[2] {for y in 0..dims[1] {for x in 0..dims[0] {
        let p=min+(max-min)*Vec3::new(x as f32/(dims[0]-1) as f32,y as f32/(dims[1]-1) as f32,z as f32/(dims[2]-1) as f32);
        let q=(p-center).abs()-half;
        distances.push(q.max(Vec3::ZERO).length()+q.max_element().min(0.0));
    }}}
    for detail in [false,true] {
        if detail {
            r.gi_set_gdf_identified(&[100.0; 8],&[0; 8],[2;3],min.to_array(),max.to_array(),&[dark],&[source])?;
            fields[0].material=dark;r.gi_set_mesh_sdfs(&fields)?;
        } else {
            r.gi_set_mesh_sdfs(&[])?;
            r.gi_set_gdf_identified(&distances,&vec![0;distances.len()],dims,min.to_array(),max.to_array(),&[dark],&[source])?;
        }
        for _ in 0..64 {r.render(&input)?;}
        let unlit=[hdr(r,0,cam)?,hdr(r,1,cam)?];
        assert!(unlit.iter().all(|x|*x<0.0001),"dark fallback emitted light: {unlit:?}");
        let epochs=std::array::from_fn::<_,2,_>(|v|r.gpu_gi.as_ref().unwrap().radiance_cache.debug(v).2);
        let geometry=r.gpu_gi.as_ref().unwrap().gdf_revision;
        let old_gdf=r.gpu_gi.as_ref().unwrap().debug_gdf_resources();
        let old_mesh=r.gpu_gi.as_ref().unwrap().mesh_sdfs.current.clone();
        // Also retain a submitted legacy march across the material publication.
        assert!(r.gi_march_begin(&GpuGiMarch {lights:&[],emitters:&[],probes:&[Vec3::ZERO],samples:64,bounces:2,
            sky:[0.0;3],eps:0.02,max_dist:40.0,seed:5}));
        if detail {fields[0].material=bright;r.gi_set_mesh_sdfs(&fields)?;}
        else {r.gi_set_gdf_materials(&[bright],&[source],&[[lo,hi]])?;}
        assert_eq!(r.gpu_gi.as_ref().unwrap().gdf_revision,geometry);
        assert!(r.sgi_history_valid && r.preview_sgi_history_valid,"material publication discarded gather history");
        if let Some(old)=&old_mesh {
            let current=r.gpu_gi.as_ref().unwrap().mesh_sdfs.current.as_ref().unwrap();
            let a=old.resources();let b=current.resources();
            assert_eq!(a.1,b.1,"material edit reuploaded mesh voxels");
            assert_eq!(a.0[1],b.0[1],"material edit rebuilt mesh BVH");
            assert_ne!(a.0[0],b.0[0],"material edit overwrote a live instance buffer");
        } else {
            let new=r.gpu_gi.as_ref().unwrap().debug_gdf_resources();
            assert_eq!((old_gdf.0,old_gdf.1),(new.0,new.1));assert_ne!(old_gdf.2,new.2);
        }
        for _ in 0..128 {r.render(&input)?;}
        assert!(r.gi_march_poll().is_some(),"legacy march did not retire after material update");
        drop(old_mesh);
        let lit=[hdr(r,0,cam)?,hdr(r,1,cam)?];
        assert!(lit.iter().all(|x|*x>0.005),"fallback material update failed to light receivers: {lit:?}");
        for (v, v_value) in epochs.iter().enumerate() {assert_eq!(r.gpu_gi.as_ref().unwrap().radiance_cache.debug(v).2,*v_value,"fallback material update reset global cache");}
        let revision=r.gpu_gi.as_ref().unwrap().material_revision;
        if detail {r.gi_set_mesh_sdfs(&fields)?;}
        else {r.gi_set_gdf_materials(&[bright],&[source],&[[lo,hi]])?;}
        assert_eq!(r.gpu_gi.as_ref().unwrap().material_revision,revision,"unchanged table produced lighting work");
        // Rapid changes without synchronous readback exercise immutable tables
        // and retirement with multiple frames still using previous snapshots.
        for frame in 0..12 {
            let material=if frame%2==0 {dark} else {bright};
            if detail {fields[0].material=material;r.gi_set_mesh_sdfs(&fields)?;}
            else {r.gi_set_gdf_materials(&[material],&[source],&[[lo,hi]])?;}
            r.render(&input)?;
        }
        if detail {fields[0].material=dark;r.gi_set_mesh_sdfs(&fields)?;}
        else {r.gi_set_gdf_materials(&[dark],&[source],&[[lo,hi]])?;}
        for _ in 0..160 {r.render(&input)?;}
        let off=[hdr(r,0,cam)?,hdr(r,1,cam)?];
        for v in 0..2 {
            assert!(off[v]<lit[v]*0.05,"fallback retained switched-off emission: {lit:?}->{off:?}");
            assert_eq!(r.gpu_gi.as_ref().unwrap().radiance_cache.debug(v).2,epochs[v]);
        }
        assert_eq!(r.gpu_gi.as_ref().unwrap().gdf_revision,geometry);
        eprintln!("[Flux fallback material] detail={detail} on={lit:?} off={off:?}; geometry and epochs retained");
        // Identity/order changes must go through a geometric publication.
        let before=r.gpu_gi.as_ref().unwrap().debug_gdf_resources();
        assert!(r.gi_set_gdf_materials(&[dark],&[None],&[]).is_err());
        assert!(r.gi_set_gdf_materials(&[],&[],&[]).is_err());
        assert_eq!(r.gpu_gi.as_ref().unwrap().debug_gdf_resources(),before);
        if detail {
            fields[0].world_to_local=Mat4::from_translation(Vec3::X)*fields[0].world_to_local;
            r.gi_set_mesh_sdfs(&fields)?;
            assert_ne!(r.gpu_gi.as_ref().unwrap().gdf_revision,geometry);
            // A moved instance costs neither view its gather history. It is reported as moved
            // geometry with a place, and every history follows it there (flux_changes).
            assert!(r.sgi_history_valid && r.preview_sgi_history_valid,"moving an instance discarded the gather history");
            r.render(&input)?;
            let packed=r.flux_changes.packed();
            let located=(0..packed.info[0] as usize).any(|k| {
                let (low,high)=(packed.bounds[k*2],packed.bounds[k*2+1]);
                low[3]>0.0 && low[3]<1.0e8 && Vec3::from_slice(&low).cmple(center).all() && Vec3::from_slice(&high).cmpge(center).all()
            });
            assert!(located,"the moved instance was not reported as local moved geometry: {:?}",&packed.bounds[..packed.info[0] as usize*2]);
        }
    }
    r.gi_set_mesh_sdfs(&[])?;
    Ok(())
}
