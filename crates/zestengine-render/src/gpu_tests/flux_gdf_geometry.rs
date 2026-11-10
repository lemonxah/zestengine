//! Coarse-field replacements retain unrelated radiance while moved/removed
//! occluders take effect. Software has no mesh fields or rasterized sources.
use glam::{Mat4,Quat,Vec3};
use crate::{CameraData,DrawCmd,FluxSettings,GiGdfGeometry,GiSource,GpuGiMaterial,MaterialHandle,MeshHandle,Renderer};
use super::{planar::{cube,draw,lit},flux_radiance::{hdr,read}};

#[derive(Clone,Copy)]
struct Body {source:GiSource,center:Vec3,half:Vec3,revision:u64,emission:[f32;3]}
impl Body {
    fn transform(self)->Mat4 {Mat4::from_scale_rotation_translation(self.half*2.0,Quat::IDENTITY,self.center)}
    fn metadata(self)->GiGdfGeometry {GiGdfGeometry {source:self.source,revision:self.revision,bounds:[self.center-self.half,self.center+self.half]}}
    fn material(self)->GpuGiMaterial {GpuGiMaterial {albedo:[0.0;3],emission:self.emission,metallic:0.0,roughness:1.0}}
}
struct Field {dist:Vec<f32>,index:Vec<u32>,dims:[u32;3],min:Vec3,max:Vec3,materials:Vec<GpuGiMaterial>,geometry:Vec<GiGdfGeometry>}
impl Field {
    fn build(bodies:&[Body],dims:[u32;3],min:Vec3,max:Vec3)->Self {
        let mut dist=Vec::new();let mut index=Vec::new();
        for z in 0..dims[2] {for y in 0..dims[1] {for x in 0..dims[0] {
            let p=min+(max-min)*Vec3::new(x as f32/(dims[0]-1) as f32,y as f32/(dims[1]-1) as f32,z as f32/(dims[2]-1) as f32);
            let mut best=f32::MAX;let mut who=0;
            for (i,b) in bodies.iter().enumerate() {
                let q=(p-b.center).abs()-b.half;let d=q.max(Vec3::ZERO).length()+q.max_element().min(0.0);
                if d<best {best=d;who=i as u32;}
            }
            dist.push(best);index.push(who);
        }}}
        Self {dist,index,dims,min,max,materials:bodies.iter().map(|b|b.material()).collect(),geometry:bodies.iter().map(|b|b.metadata()).collect()}
    }
    fn publish(&self,r:&mut Renderer)->anyhow::Result<()> {
        r.gi_set_gdf_tracked(&self.dist,&self.index,self.dims,self.min.to_array(),self.max.to_array(),&self.materials,&self.geometry)
    }
}
const FAR:u64=8192+7+16*7+256*7;
fn epochs(r:&Renderer)->[u32;2] {std::array::from_fn(|v|r.gpu_gi.as_ref().unwrap().radiance_cache.debug(v).2)}
fn sentinel(r:&Renderer)->anyhow::Result<[Vec<u8>;2]> {
    let size=crate::flux_radiance_cache::PROBE_BYTES as usize;
    let mut expected=std::array::from_fn(|_|vec![0;size]);
    for (view,bytes) in expected.iter_mut().enumerate() {
        let (data,_,epoch)=r.gpu_gi.as_ref().unwrap().radiance_cache.debug(view);
        bytes[..32].copy_from_slice(bytemuck::cast_slice(&[7u32,7,7,2,epoch,r.frame_serial as u32,0,8]));
        bytes[32..40].copy_from_slice(bytemuck::cast_slice(&[0x3c003c00u32;2]));
        crate::alloc::one_time_submit(&r.ctx.device,r.command_pool,r.ctx.queue,|cb|unsafe {
            crate::flux_radiance_cache::barrier(&r.ctx.device,cb);
            r.ctx.device.cmd_update_buffer(cb,data,FAR*size as u64,bytes);
            crate::flux_radiance_cache::barrier(&r.ctx.device,cb);
        })?;
    }
    Ok(expected)
}
fn retained(r:&Renderer,epoch:[u32;2],expected:&[Vec<u8>;2])->anyhow::Result<()> {
    assert_eq!(epochs(r),epoch,"local GDF replacement reset global radiance");
    for view in 0..2 {
        let (data,work,_)=r.gpu_gi.as_ref().unwrap().radiance_cache.debug(view);
        let bytes=read(r,data,FAR*crate::flux_radiance_cache::PROBE_BYTES,expected[view].len())?;
        assert_eq!(&bytes[..expected[view].len()],&expected[view],"unrelated world probe was modified");
        let bytes=read(r,work,0,48*4)?;let words:&[u32]=bytemuck::cast_slice(&bytes[..48*4]);
        assert!(words[9]<=32*256,"coarse-field update exceeded radiance ray budget");
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn render(r:&mut Renderer,cam:CameraData,receiver:DrawCmd,mesh:MeshHandle,glow:MaterialHandle,black:MaterialHandle,bodies:&[Body],hardware:bool,frames:usize)->anyhow::Result<[f32;2]> {
    let mut draws=vec![receiver];
    if hardware {for b in bodies {
        let mut d=draw(mesh,if b.emission[0]>0.0 {glow} else {black},b.transform(),-1);d.gi_source=Some(b.source);draws.push(d);
    }}
    let mut input=crate::cascade_tests::input(cam);input.viewport_extent=Some([323,181]);input.camera_preview=Some(cam);input.draws=&draws;
    input.light.intensity=0.0;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;input.draw_skybox=false;
    input.postfx.bloom_enabled=false;input.preview_postfx.bloom_enabled=false;
    for _ in 0..frames {r.render(&input)?;}
    Ok([hdr(r,0,cam)?,hdr(r,1,cam)?])
}
pub(super) fn run(r:&mut Renderer)->anyhow::Result<()> {
    scroll_history(r)?;
    let mesh=r.upload_mesh(&cube())?;let floor=lit(r,[0.8;4],[0.0;3],1.0,0.0);
    let glow=lit(r,[0.0,0.0,0.0,1.0],[4.0,0.4,0.1],1.0,0.0);
    r.set_material_features(glow,crate::MaterialFeatures {emission:true,..Default::default()});
    let black=lit(r,[0.0,0.0,0.0,1.0],[0.0;3],1.0,0.0);
    let receiver=draw(mesh,floor,Mat4::from_scale_rotation_translation(Vec3::new(12.0,0.2,12.0),Quat::IDENTITY,Vec3::new(0.0,-0.1,0.0)),-1);
    let eye=Vec3::new(0.0,2.0,4.0);let cam=CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let min=Vec3::new(-80.0,-4.0,-20.0);let max=Vec3::new(80.0,20.0,16.0);let dims=[33,33,73];
    r.set_gi_enabled(true);r.gi_set_emitters(&[]);r.gi_set_mesh_sdfs(&[])?;r.set_area_lights(&[]);
    r.gpu_gi.as_mut().unwrap().screen_trace_override=Some(false);
    for hardware in [false,true] {
        if hardware && !r.ctx.ray_tracing() {continue;}
        r.set_flux_settings(FluxSettings {rt_trace:hardware,samples:64,bounces:2,march_distance:40.0,reflection_mode:0,ssr_enabled:false,..Default::default()});
        let mut bodies=vec![Body {source:GiSource {object:0xfc01,slot:0},center:Vec3::new(0.0,8.0,-12.0),half:Vec3::new(12.0,8.0,0.1),revision:1,emission:[4.0,0.4,0.1]},
            Body {source:GiSource {object:0xfc02,slot:0},center:Vec3::new(50.0,0.0,-5.0),half:Vec3::new(20.0,20.0,0.01),revision:1,emission:[0.0;3]}];
        Field::build(&bodies,dims,min,max).publish(r)?;
        let opened=render(r,cam,receiver,mesh,glow,black,&bodies,hardware,128)?;
        assert!(opened.iter().all(|v|*v>0.03),"GDF receiver was not lit: {opened:?}");
        let epoch=epochs(r);let saved=sentinel(r)?;
        let global_revision=r.gpu_gi.as_ref().unwrap().global_field_revision;
        bodies[1].center.x=0.0;bodies[1].revision+=1;
        Field::build(&bodies,dims,min,max).publish(r)?;
        let blocked=render(r,cam,receiver,mesh,glow,black,&bodies,hardware,16)?;
        assert!(blocked.iter().zip(opened).all(|(b,o)|*b<o*0.02),"moved coarse-field wall reused unoccluded lighting: {opened:?}->{blocked:?}");
        retained(r,epoch,&saved)?;
        bodies[1].center.x=50.0;bodies[1].revision+=1;
        Field::build(&bodies,dims,min,max).publish(r)?;
        let reopened=render(r,cam,receiver,mesh,glow,black,&bodies,hardware,96)?;
        assert!(reopened.iter().zip(opened).all(|(v,o)|*v>o*0.8),"GDF movement retained old occlusion: {reopened:?}");
        retained(r,epoch,&saved)?;
        bodies.swap(0,1);Field::build(&bodies,dims,min,max).publish(r)?;
        let reordered=render(r,cam,receiver,mesh,glow,black,&bodies,hardware,64)?;
        assert!(reordered.iter().zip(reopened).all(|(v,o)|(*v-o).abs()<o*0.2),"index-table reorder changed physical lighting: {reopened:?}->{reordered:?}");
        retained(r,epoch,&saved)?;
        bodies[0].center.x=0.0;bodies[0].revision+=1;Field::build(&bodies,dims,min,max).publish(r)?;
        render(r,cam,receiver,mesh,glow,black,&bodies,hardware,16)?;
        bodies.remove(0);Field::build(&bodies,dims,min,max).publish(r)?;
        let removed=render(r,cam,receiver,mesh,glow,black,&bodies,hardware,96)?;
        assert!(removed.iter().zip(opened).all(|(v,o)|*v>o*0.8),"deleted GDF wall retained its shadow: {removed:?}");
        retained(r,epoch,&saved)?;
        bodies.clear();Field::build(&bodies,dims,min,max).publish(r)?;
        let dark=render(r,cam,receiver,mesh,glow,black,&bodies,hardware,128)?;
        assert!(dark.iter().all(|v|*v<0.005),"deleted GDF emitter retained light: {dark:?}");
        retained(r,epoch,&saved)?;
        assert_eq!(r.gpu_gi.as_ref().unwrap().global_field_revision,global_revision);
        // Invalid publications cannot replace the last good field or metadata.
        let mut field=Field::build(&bodies,dims,min,max);field.dist[0]=f32::NAN;
        let before=r.gpu_gi.as_ref().unwrap().debug_gdf_resources();
        assert!(field.publish(r).is_err());assert_eq!(r.gpu_gi.as_ref().unwrap().debug_gdf_resources(),before);
        for (resolution,upper) in [([65,33,73],max),([65,33,73],max+Vec3::X*10.0)] {
            let previous=epochs(r);Field::build(&bodies,resolution,min,upper).publish(r)?;
            render(r,cam,receiver,mesh,glow,black,&bodies,hardware,2)?;
            assert!(epochs(r).iter().zip(previous).all(|(a,b)|*a!=b),"GDF domain/resolution change retained invalid history");
        }
        let field=Field::build(&bodies,dims,min,max);let previous=epochs(r);
        r.gi_set_gdf_identified(&field.dist,&field.index,dims,min.to_array(),max.to_array(),&field.materials,&[])?;
        render(r,cam,receiver,mesh,glow,black,&bodies,hardware,2)?;
        assert!(epochs(r).iter().zip(previous).all(|(a,b)|*a!=b),"untracked GDF publication failed to invalidate");
        let previous=epochs(r);field.publish(r)?;render(r,cam,receiver,mesh,glow,black,&bodies,hardware,2)?;
        assert!(epochs(r).iter().zip(previous).all(|(a,b)|*a!=b),"tracked field trusted an unknown predecessor");
        eprintln!("[Flux local GDF] hardware={hardware} open={opened:?} blocked={blocked:?} reopened={reopened:?} reordered={reordered:?} removed={removed:?} deleted={dark:?}; unrelated probes retained, global changes rejected");
    }
    r.gpu_gi.as_mut().unwrap().screen_trace_override=None;
    Ok(())
}

fn scroll_history(r:&mut Renderer)->anyhow::Result<()> {
    let body=Body {source:GiSource {object:0xfc10,slot:0},center:Vec3::ZERO,half:Vec3::splat(0.5),revision:1,emission:[0.0;3]};
    let mesh=cube();let positions:Vec<_>=mesh.vertices.iter().map(|v|Vec3::from_array(v.position)).collect();
    let levels=[crate::GiMeshSdfLevels {
        detail:std::sync::Arc::new(crate::sdf::generate_mesh_sdf(&positions,&mesh.indices,32,0.1)),
        coarse:std::sync::Arc::new(crate::sdf::generate_mesh_sdf(&positions,&mesh.indices,8,0.1)),
    }];
    let meshes=[crate::GiMeshSdf {gi_source:Some(body.source),world_to_local:body.transform().inverse(),
        volume:levels[0].detail.clone(),material:body.material(),in_global_field:true}];
    assert_eq!(r.gi_poll_mesh_sdfs(&meshes,&levels)?,(true,false));
    let lo=Vec3::splat(-8.0);let hi=Vec3::splat(8.0);let dims=[33;3];
    Field::build(&[body],dims,lo,hi).publish(r)?;
    r.sgi_history_valid=true;r.preview_sgi_history_valid=true;
    assert_eq!(r.gi_poll_mesh_sdfs(&meshes,&levels)?,(true,false));
    assert!(r.sgi_history_valid && r.preview_sgi_history_valid,"unchanged streamed mesh fields reset either view");
    r.sgi_last_view_proj=Some(Mat4::IDENTITY);r.preview_sgi_last_view_proj=Some(Mat4::IDENTITY);
    let gpu=r.gpu_gi.as_ref().unwrap();
    let revisions=(gpu.gdf_revision,gpu.global_field_revision);
    let ranges=(gpu.gdf_trace_distance(Vec3::new(0.0,1.0,2.0)),gpu.surface_trace_distance());
    let shift=Vec3::X*4.0;
    Field::build(&[body],dims,lo+shift,hi+shift).publish(r)?;
    assert!(r.sgi_history_valid && r.preview_sgi_history_valid,"field scroll reset Scene/Camera temporal history");
    assert_eq!(r.sgi_last_view_proj,Some(Mat4::IDENTITY));assert_eq!(r.preview_sgi_last_view_proj,Some(Mat4::IDENTITY));
    let gpu=r.gpu_gi.as_ref().unwrap();
    assert_eq!((gpu.gdf_revision,gpu.global_field_revision),revisions,"unchanged geometry reset world/surface lighting");
    assert_eq!((gpu.gdf_trace_distance(Vec3::new(0.0,1.0,2.0)),gpu.surface_trace_distance()),ranges,"camera grid changed physical trace range");
    for level in 1..4 {
        let field=Field::build(&[body],dims,lo*(1u32<<level) as f32,hi*(1u32<<level) as f32);
        r.gi_set_gdf_clipmap(level,&field.dist,&field.index,dims,field.min.to_array(),field.max.to_array(),&field.materials,&field.geometry)?;
        let gpu=r.gpu_gi.as_ref().unwrap();
        assert_eq!(gpu.debug_gdf_clipmap_mask(),(1<<(level+1))-1);
        assert_eq!((gpu.gdf_revision,gpu.global_field_revision),revisions,"adding an acceleration level invalidated physical lighting");
        assert!(r.sgi_history_valid && r.preview_sgi_history_valid,"outer field reset Scene/Camera history");
    }
    r.gi_disable_gdf_clipmaps()?;
    assert_eq!(r.gpu_gi.as_ref().unwrap().debug_gdf_clipmap_mask(),1);
    assert!(r.sgi_history_valid && r.preview_sgi_history_valid);
    let outer=Field::build(&[body],dims,lo*2.0,hi*2.0);
    r.gi_set_gdf_clipmap(1,&outer.dist,&outer.index,dims,outer.min.to_array(),outer.max.to_array(),&outer.materials,&outer.geometry)?;
    let mut revised=body;revised.revision+=1;
    Field::build(&[revised],dims,lo+shift,hi+shift).publish(r)?;
    assert_eq!(r.gpu_gi.as_ref().unwrap().debug_gdf_clipmap_mask(),1,"old source snapshot remained active");
    assert!(r.gi_set_gdf_clipmap(1,&outer.dist,&outer.index,dims,outer.min.to_array(),outer.max.to_array(),&outer.materials,&outer.geometry).is_err(),
        "worker from a previous geometry revision was published");
    Field::build(&[body],dims,lo+shift,hi+shift).publish(r)?;
    r.gi_disable_gdf_clipmaps()?;
    r.sgi_history_valid=true;r.preview_sgi_history_valid=true;
    // A pending field is an acceleration change only; exact mesh visibility
    // remains live, and committing its completed grid must not restart history.
    let pending_revisions={let g=r.gpu_gi.as_ref().unwrap();(g.gdf_revision,g.global_field_revision)};
    r.gi_prepare_gdf_build((lo+shift).to_array(),(hi+shift).to_array(),&[body.material()],&[body.metadata()])?;
    assert!(r.sgi_history_valid && r.preview_sgi_history_valid,"preparing unchanged geometry reset history");
    Field::build(&[body],dims,lo+shift,hi+shift).publish(r)?;
    assert!(r.sgi_history_valid && r.preview_sgi_history_valid,"completed background field reset history");
    let g=r.gpu_gi.as_ref().unwrap();
    assert_eq!((g.gdf_revision,g.global_field_revision),pending_revisions);
    let mut edited=body;edited.emission=[1.0;3];
    Field::build(&[edited],dims,lo+shift,hi+shift).publish(r)?;
    assert!(!r.sgi_history_valid && !r.preview_sgi_history_valid,"lighting edit was mistaken for a pure scroll");
    r.gi_set_mesh_sdfs(&[])?;
    assert!(r.gi_prepare_gdf_build(lo.to_array(),hi.to_array(),&[body.material()],&[body.metadata()]).is_err(),
        "missing mesh geometry was allowed to replace the coarse occluder");
    r.sgi_history_valid=true;r.preview_sgi_history_valid=true;
    Field::build(&[edited],dims,lo,hi).publish(r)?;
    assert!(!r.sgi_history_valid && !r.preview_sgi_history_valid,"unmapped coarse geometry retained invalid history");
    eprintln!("[Flux field scroll] Scene/Camera and world history retained, physical range stable; lighting edits and missing detail reject history");
    Ok(())
}
