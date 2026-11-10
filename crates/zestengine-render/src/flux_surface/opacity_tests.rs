//! Thin emissive coverage through the production software distance-field trace.
use super::*;
use std::sync::Arc;
use crate::{texture::GpuTexture,GiMeshSdf,GpuGiMaterial};

#[test]
fn software_opaque_emissive_plane_preserves_ray_coverage() {
    let Some(g)=crate::gpu_tests::gpu() else {return;};
    let device=&g.device;let mut alloc=g.allocator().unwrap();
    let before=crate::context::validation_error_count();
    let positions=[Vec3::new(-20.0,2.0,-20.0),Vec3::new(20.0,2.0,-20.0),
        Vec3::new(20.0,2.0,20.0),Vec3::new(-20.0,2.0,20.0)];
    let plane=Arc::new(crate::sdf::generate_sdf(&positions,&[0,1,2,0,2,3],32,0.1));
    // Match the atlas brick dimensions of the real floor/source/pane fixture.
    let dummy=Arc::new(crate::sdf::SdfVolume {dims:[24;3],min:Vec3::splat(-0.6),max:Vec3::splat(0.6),
        geometry_bounds:[Vec3::splat(-0.5),Vec3::splat(0.5)],data:vec![100.0;24*24*24]});
    let fields:Vec<_>=(0..3).map(|i|GiMeshSdf {gi_source:Some(crate::GiSource {object:i as u128,slot:0}),
        world_to_local:if i==2 {Mat4::IDENTITY} else {Mat4::from_translation(Vec3::new(-100.0-i as f32*10.0,0.0,0.0))},
        volume:if i==2 {plane.clone()} else {dummy.clone()},in_global_field:false,
        material:GpuGiMaterial {albedo:[0.0;3],emission:[0.0,4.0,0.0],metallic:0.0,roughness:1.0}}).collect();
    let mut cache=crate::mesh_sdf::Cache::default();cache.set(device,&mut alloc,g.pool,g.queue,&fields).unwrap();
    let mesh=cache.current.as_ref().unwrap();
    const W:usize=64;
    let positions:Vec<_>=(0..W*W).map(|i|[-20.0+(i%W) as f32*40.0/W as f32+20.0/W as f32,
        2.0,-20.0+(i/W) as f32*40.0/W as f32+20.0/W as f32,1.0]).collect();
    let normals=vec![[0.0,-1.0,0.0,1.0];W*W];let albedo=vec![[0.0;4];W*W];
    let emission=vec![[0.0,4.0,0.0,-2.0];W*W];let light=vec![[0.0;4];W*W];
    let mut textures=Vec::new();
    for data in [&positions,&normals,&albedo,&emission,&light,&light] {
        textures.push(GpuTexture::upload_pixels(device,&mut alloc,g.pool,g.queue,W as u32,W as u32,
            vk::Format::R32G32B32A32_SFLOAT,bytemuck::cast_slice(data)).unwrap());
    }
    let projection=Mat4::from_cols(glam::Vec4::new(0.05,0.0,0.0,0.0),glam::Vec4::new(0.0,0.0,100.0,0.0),
        glam::Vec4::new(0.0,0.05,0.0,0.0),glam::Vec4::new(0.0,0.0,-199.5,1.0));
    let cards=[Card {view_proj:projection.to_cols_array_2d(),normal_texel:[0.0,-1.0,0.0,40.0/W as f32],rect:[0,0,W as u32,55],
        capture_to_world:Mat4::IDENTITY.to_cols_array_2d(),virtual_page:[1,0,0,0],neighbours:[0;4]}];
    let mut nodes=vec![Node {left:1,right:3,..Node::zeroed()},Node::zeroed(),Node::zeroed(),
        Node {left:5,right:1,min:[9.0,1.0,0.0],..Node::zeroed()},
        Node {left:0x80000000,right:5,min:[-20.0,2.0,-20.0],max:[20.0,2.0,20.0]}];
    let points:Vec<_>=(-120..=120).flat_map(|x|(-120..=120).map(move |z|
        [x as f32*0.15,2.0,z as f32*0.15,mesh.node_count as f32])).collect();
    let views:Vec<_>=textures.iter().map(|t|t.view).collect();
    let mut failures=Vec::new();
    for policy in [1.0,9.0] {
        nodes[3].min[0]=policy;
        let results=lookup_tests::sample_with_mesh(device,&mut alloc,g.pool,g.queue,&cards,&nodes,&views,&points,
            include_bytes!(concat!(env!("OUT_DIR"),"/flux_opacity_trace_test.comp.spv")),9,Some(mesh));
        let misses:Vec<_>=points.iter().zip(&results).filter(|(_,r)|r[0]<0.0 || (r[1]-2.0).abs()>0.01 || r[2]>-0.99 || r[3]<0.999).collect();
        eprintln!("[Flux plane trace] policy={policy} rays={} misses={} examples={:?}",points.len(),misses.len(),&misses[..misses.len().min(12)]);
        if !misses.is_empty() {failures.push((policy,misses.len()));}
    }
    for texture in &mut textures {texture.destroy(device,&mut alloc);}
    cache.destroy(device,&mut alloc);
    assert!(failures.is_empty(),"opaque pane lost software ray coverage: {failures:?}");
    assert_eq!(crate::context::validation_error_count(),before);
}
