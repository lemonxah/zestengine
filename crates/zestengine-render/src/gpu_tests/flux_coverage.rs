//! Production software trace queries across the mesh/global-field handoff.
use ash::vk;
use glam::{Mat4,Vec3};
use gpu_allocator::MemoryLocation;
use crate::{alloc::{Buffer,one_time_submit},texture::GpuTexture,GiMeshSdf,GpuGiMaterial};
#[derive(Clone,Copy)]
struct BoxField {center:Vec3,half:Vec3}
impl BoxField {
    fn distance(self,p:Vec3)->f32 {let q=(p-self.center).abs()-self.half;q.max(Vec3::ZERO).length()+q.max_element().min(0.0)}
    fn transform(self)->Mat4 {Mat4::from_scale_rotation_translation(self.half*2.0,glam::Quat::IDENTITY,self.center)}
}
fn body(x:f32,y:f32,z:f32,hx:f32,hy:f32,hz:f32)->BoxField {BoxField {center:Vec3::new(x,y,z),half:Vec3::new(hx,hy,hz)}}
#[test]
fn flux_software_coverage_preserves_thin_walls_and_openings_at_distance() {coverage(false,false,false)}
#[test]
fn flux_geometry_clipmaps_preserve_coverage_across_nested_and_separated_grids() {coverage(true,false,false)}
#[test]
fn flux_pending_geometry_uses_current_meshes_when_all_coarse_grids_are_empty() {coverage(true,true,false)}
#[test]
fn flux_resident_coarse_meshes_preserve_walls_and_openings_without_fine_pages() {coverage(true,true,true)}
fn coverage(multilevel:bool,pending:bool,resident_coarse:bool) {
    let Some(g)=super::gpu() else {return};let device=&g.device;let mut alloc=g.allocator().unwrap();
    let before=g.validation_error_count();
    let mut input=Buffer::new(device,&mut alloc,64*32,vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::CpuToGpu,"Coverage rays").unwrap();
    let mut output=Buffer::new(device,&mut alloc,64*16,vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::GpuToCpu,"Coverage hits").unwrap();
    let mut uniform=Buffer::new(device,&mut alloc,48,vk::BufferUsageFlags::UNIFORM_BUFFER,MemoryLocation::CpuToGpu,"Coverage params").unwrap();
    let mut clip_uniform=Buffer::new(device,&mut alloc,144,vk::BufferUsageFlags::UNIFORM_BUFFER,MemoryLocation::CpuToGpu,"Coverage clipmap domains").unwrap();
    let sampled=vk::DescriptorType::COMBINED_IMAGE_SAMPLER;let storage=vk::DescriptorType::STORAGE_BUFFER;
    let types=[sampled,sampled,storage,storage,sampled,storage,storage,vk::DescriptorType::UNIFORM_BUFFER,vk::DescriptorType::UNIFORM_BUFFER];
    let bindings:Vec<_>=types.iter().enumerate().map(|(i,t)|vk::DescriptorSetLayoutBinding::default().binding(i as u32).descriptor_type(*t).descriptor_count(if multilevel && i<2 {4} else {1}).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
    let set_layout=unsafe {device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),None).unwrap()};
    let layout=unsafe {device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]),None).unwrap()};
    let shader=crate::screen_pass::shader_module(device,if multilevel {include_bytes!(concat!(env!("OUT_DIR"),"/flux_clipmap_test.comp.spv"))} else {include_bytes!(concat!(env!("OUT_DIR"),"/flux_coverage_test.comp.spv"))}).unwrap();
    let pipeline=unsafe {device.create_compute_pipelines(vk::PipelineCache::null(),&[vk::ComputePipelineCreateInfo::default().layout(layout)
        .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(shader).name(c"main"))],None).unwrap()[0]};
    let pool=unsafe {device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
        vk::DescriptorPoolSize {ty:sampled,descriptor_count:if multilevel {9} else {3}},vk::DescriptorPoolSize {ty:storage,descriptor_count:4},
        vk::DescriptorPoolSize {ty:types[7],descriptor_count:2}]),None).unwrap()};
    let set=unsafe {device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[set_layout])).unwrap()[0]};
    let sampler=|filter|unsafe {device.create_sampler(&vk::SamplerCreateInfo::default().mag_filter(filter).min_filter(filter)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE).address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE).address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),None).unwrap()};
    let linear=sampler(vk::Filter::LINEAR);let nearest=sampler(vk::Filter::NEAREST);
    let clip_info=[vk::DescriptorBufferInfo::default().buffer(clip_uniform.handle).range(vk::WHOLE_SIZE)];
    unsafe {device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(8)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&clip_info)],&[]);}
    let cube=super::flux::cube_distance_field();let mut cache=crate::mesh_sdf::Cache::default();
    let family=unsafe {g.instance.get_physical_device_queue_family_properties(g.pd)}.iter()
        .position(|q|q.queue_flags.contains(vk::QueueFlags::COMPUTE|vk::QueueFlags::TRANSFER)).unwrap() as u32;
    let mut coarse=(*cube).clone();coarse.dims=[8;3];coarse.data.clear();
    for z in 0..8 {for y in 0..8 {for x in 0..8 {
        let p=Vec3::splat(-0.6)+Vec3::new(x as f32,y as f32,z as f32)*(1.2/7.0);
        let q=p.abs()-Vec3::splat(0.5);coarse.data.push(q.max(Vec3::ZERO).length()+q.max_element().min(0.0));
    }}}
    let levels=[crate::GiMeshSdfLevels {detail:cube.clone(),coarse:std::sync::Arc::new(coarse)}];
    if resident_coarse {
        // Fine24^3 plus borders cannot fit this16-deep detail partition. Every
        // ray must therefore hit the independently sampled coarse mesh field.
        cache.test_streamed_extent(device,&mut alloc,g.queue,family,[64,64,32],16).unwrap();
    }
    let mut failures=Vec::new();let mut checks=0;
    let cases=[
        ("thin-wall",vec![body(0.0,0.0,3.0,4.0,4.0,0.02)],true),
        ("outside-domain",vec![body(0.0,0.0,20.0,4.0,4.0,0.02)],true),
        ("outside-parallel",vec![body(30.0,0.0,3.0,4.0,4.0,0.02)],true),
        ("opening",vec![body(-2.5,0.0,5.0,1.5,4.0,0.02),body(2.5,0.0,5.0,1.5,4.0,0.02),body(0.0,-2.5,5.0,1.0,1.5,0.02),body(0.0,2.5,5.0,1.0,1.5,0.02)],true),
        ("opening-with-wall",vec![body(-2.5,0.0,5.0,1.5,4.0,0.02),body(2.5,0.0,5.0,1.5,4.0,0.02),body(0.0,0.0,9.0,4.0,4.0,0.02)],true),
        ("partial-mesh-snapshot",vec![body(0.0,0.0,1.25,4.0,4.0,0.5)],false),
    ];
    for (name,bodies,complete) in cases {
      if pending && !complete {continue;}
      // Hold the queried geometry and both ray directions fixed while the
      // primary view's field scrolls past it or teleports to another room.
      // A secondary camera and off-screen occluders must still hit mesh detail.
      for shift in [Vec3::ZERO,Vec3::Z*32.0,-Vec3::Z*32.0,Vec3::X*4096.0] {
        if !complete && shift!=Vec3::ZERO {continue;}
        let fields:Vec<_>=if complete {bodies.clone()} else {vec![body(20.0,0.0,0.0,1.0,1.0,1.0)]}.iter().enumerate().map(|(i,b)|GiMeshSdf {
            gi_source:Some(crate::GiSource {object:i as u128+1,slot:0}),world_to_local:b.transform().inverse(),volume:cube.clone(),
            material:GpuGiMaterial {albedo:[0.0;3],emission:[0.0;3],metallic:0.0,roughness:1.0},in_global_field:true}).collect();
        if resident_coarse {
            let update=cache.set_streamed(device,&mut alloc,g.queue,family,&fields,&levels,&[shift]).unwrap();
            assert!(!update.pending && !update.streaming);
        } else {cache.set(device,&mut alloc,g.pool,g.queue,&fields).unwrap();}
        let snapshot=cache.current.as_ref().unwrap();
        let dims=if complete {[9;3]} else {[65;3]};let lo=Vec3::splat(-16.0)+shift;let hi=Vec3::splat(16.0)+shift;
        let mut distances=Vec::new();let mut indices=Vec::new();
        for z in 0..dims[2] {for y in 0..dims[1] {for x in 0..dims[0] {
            let p=lo+(hi-lo)*Vec3::new(x as f32,y as f32,z as f32)/Vec3::from_array(dims.map(|n|(n-1) as f32));
            let (index,distance)=bodies.iter().enumerate().map(|(i,b)|(i,b.distance(p))).min_by(|a,b|a.1.total_cmp(&b.1)).unwrap();
            distances.push(distance);indices.push(index as u32);
        }}}
        if pending {distances.fill(f32::MAX);}
        let mut distance=GpuTexture::upload_volume(device,&mut alloc,g.pool,g.queue,&distances,dims,None).unwrap();
        let mut index=GpuTexture::upload_volume_u32(device,&mut alloc,g.pool,g.queue,&indices,dims,None).unwrap();
        let mut outer=Vec::new();
        let mut clip_data=[[0.0f32;4];9];clip_data[0]=lo.extend(0.0).to_array();clip_data[4]=hi.extend(0.0).to_array();
        clip_data[8][0]=f32::from_bits(if multilevel && complete {15} else {1});
        clip_data[8][1]=f32::from_bits(u32::from(pending));
        if multilevel && complete {
            for level in 1..4 {
                // Nested grids at rest; deliberately non-nested old/new grids
                // during travel exercise uncovered intervals within their union.
                let center=if shift==Vec3::ZERO {shift} else {shift+[Vec3::X,Vec3::Z,-Vec3::X][level-1]*512.0};
                let half=Vec3::splat(16.0*(1u32<<level) as f32);let a=center-half;let b=center+half;
                clip_data[level]=a.extend(0.0).to_array();clip_data[level+4]=b.extend(0.0).to_array();
                let mut ds=Vec::new();let mut ids=Vec::new();
                for z in 0..9 {for y in 0..9 {for x in 0..9 {
                    let p=a+(b-a)*Vec3::new(x as f32,y as f32,z as f32)/8.0;
                    let (id,d)=bodies.iter().enumerate().map(|(i,b)|(i,b.distance(p))).min_by(|a,b|a.1.total_cmp(&b.1)).unwrap();
                    ds.push(d);ids.push(id as u32);
                }}}
                if pending {ds.fill(f32::MAX);}
                outer.push((GpuTexture::upload_volume(device,&mut alloc,g.pool,g.queue,&ds,[9;3],None).unwrap(),
                    GpuTexture::upload_volume_u32(device,&mut alloc,g.pool,g.queue,&ids,[9;3],None).unwrap()));
            }
        }
        clip_uniform.write(0,bytemuck::cast_slice(&clip_data));
        let (buffers,atlas)=snapshot.resources();
        for (binding,buffer) in [(2,buffers[0]),(3,buffers[1]),(5,input.handle),(6,output.handle),(7,uniform.handle)] {
            let info=[vk::DescriptorBufferInfo::default().buffer(buffer).range(vk::WHOLE_SIZE)];unsafe {device.update_descriptor_sets(&[vk::WriteDescriptorSet::default()
                .dst_set(set).dst_binding(binding).descriptor_type(types[binding as usize]).buffer_info(&info)],&[]);}
        }
        for (binding,view,sampler) in [(0,distance.view,linear),(1,index.view,nearest),(4,atlas,linear)] {
            let mut info=vec![vk::DescriptorImageInfo::default().sampler(sampler).image_view(view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            if multilevel && binding<2 {for level in 0..3 {
                let v=outer.get(level).map(|(d,i)|if binding==0 {d.view} else {i.view}).unwrap_or(view);
                info.push(vk::DescriptorImageInfo::default().sampler(sampler).image_view(v).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL));
            }}
            unsafe {device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding).descriptor_type(sampled).image_info(&info)],&[]);}
        }
        uniform.write(0,bytemuck::cast_slice(&[lo.extend(0.0).to_array(),hi.extend(snapshot.node_count as f32).to_array(),[f32::from(complete),0.0,0.0,0.0]]));
        let mut rays=Vec::new();let mut expected=Vec::new();
        for x in [0.0,2.5,4.5] {for (z,dir) in [(-8.0,1.0),(0.0,1.0),(1.5,1.0),(24.0,-1.0)] {for max in [1.0,40.0] {for min in [0.0,6.0] {
            if min>max {continue;}
            let x=x+if name=="outside-parallel" {30.0} else {0.0};
            let origin=Vec3::new(x,0.0,z);let direction=Vec3::Z*dir;
            let hit=bodies.iter().filter_map(|b| {
                if (x-b.center.x).abs()>b.half.x || b.center.y.abs()>b.half.y {return None;}
                let a=(b.center.z-b.half.z-z)/dir;let c=(b.center.z+b.half.z-z)/dir;
                let t=if a.min(c)>=0.002 {a.min(c)} else {a.max(c)};
                (t>=0.002 && t>=min && t<=max).then_some(t)
            }).min_by(f32::total_cmp).unwrap_or(-1.0);
            // Partial field uses the legacy signed-volume convention; avoid
            // starting inside it in this correspondence test.
            if !complete && (z-1.25_f32).abs()<0.5 {continue;}
            rays.push([origin.extend(min).to_array(),direction.extend(max).to_array()]);expected.push(hit);
        }}}}
        input.write(0,bytemuck::cast_slice(&rays));
        one_time_submit(device,g.pool,g.queue,|cb|unsafe {
            device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,pipeline);device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,layout,0,&[set],&[]);
            device.cmd_dispatch(cb,rays.len() as u32,1,1);
            device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::COMPUTE_SHADER,vk::PipelineStageFlags::HOST,vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ)],&[],&[]);
        }).unwrap();
        let data=output.read();let actual:Vec<[f32;4]>=bytemuck::pod_collect_to_vec(&data[..rays.len()*16]);
        for (i,(got,want)) in actual.iter().zip(expected).enumerate() {
            checks+=1;let valid=if want<0.0 {got[0]<0.0} else {(got[0]-want).abs()<if complete {0.008} else {0.04}};
            if !valid || (rays[i][0][3]==0.0 && (got[3]>0.5)!=(want>=0.0)) {failures.push(format!("{name} domain_shift={shift} ray={:?} got={got:?} expected={want}",rays[i]));}
        }
        for (mut d,mut i) in outer {d.destroy(device,&mut alloc);i.destroy(device,&mut alloc);}
        distance.destroy(device,&mut alloc);index.destroy(device,&mut alloc);
      }
    }
    // A signed field may reach its clip-domain boundary. Normal reconstruction
    // must differentiate that field, not the artificial positive extension
    // used to march outside the domain. Test both gradient orientations.
    for sign in [-1.0f32,1.0] {
        let data:Vec<_>=(0..125).map(|i|sign*((i/5%5) as f32*0.5-1.0)).collect();
        let mut distance=GpuTexture::upload_volume(device,&mut alloc,g.pool,g.queue,&data,[5;3],None).unwrap();
        let mut index=GpuTexture::upload_volume_u32(device,&mut alloc,g.pool,g.queue,&[0;125],[5;3],None).unwrap();
        for (binding,view,sampler) in [(0,distance.view,linear),(1,index.view,nearest)] {
            let info=vec![vk::DescriptorImageInfo::default().sampler(sampler).image_view(view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);if multilevel {4} else {1}];
            unsafe {device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding)
                .descriptor_type(sampled).image_info(&info)],&[]);}
        }
        let mut normal_domains=[[0.0f32;4];9];normal_domains[0]=[-1.0,-1.0,-1.0,0.0];normal_domains[4]=[1.0,1.0,1.0,0.0];normal_domains[8][0]=f32::from_bits(1);
        clip_uniform.write(0,bytemuck::cast_slice(&normal_domains));
        uniform.write(0,bytemuck::cast_slice(&[[-1.0f32,-1.0,-1.0,0.0],[1.0,1.0,1.0,0.0],[0.0,1.0,0.0,0.0]]));
        let queries=[-1.0f32,0.0,1.0].map(|y|[[0.0,y,0.0,0.0],[0.0;4]]);
        input.write(0,bytemuck::cast_slice(&queries));
        one_time_submit(device,g.pool,g.queue,|cb|unsafe {
            device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,pipeline);
            device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,layout,0,&[set],&[]);
            device.cmd_dispatch(cb,3,1,1);
            device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::COMPUTE_SHADER,vk::PipelineStageFlags::HOST,vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ)],&[],&[]);
        }).unwrap();
        let bytes=output.read();let normals:Vec<[f32;4]>=bytemuck::pod_collect_to_vec(&bytes[..48]);
        for (i,n) in normals.iter().enumerate() {
            checks+=1;
            if (Vec3::from_slice(n)-Vec3::Y*sign).length()>1e-5 {
                failures.push(format!("field-boundary normal: sign={sign} point={:?} normal={n:?}",queries[i][0]));
            }
        }
        distance.destroy(device,&mut alloc);index.destroy(device,&mut alloc);
    }
    if multilevel && !pending {
        // Distinct data in every level verifies selection itself: mesh tracing
        // must not accidentally hide an always-base descriptor/lookup bug.
        let mut textures=Vec::new();let mut domains=[[0.0f32;4];9];
        for level in 0..4 {
            let extent=(1u32<<level) as f32;
            domains[level]=[-extent,-extent,-extent,0.0];domains[level+4]=[extent,extent,extent,0.0];
            textures.push((GpuTexture::upload_volume(device,&mut alloc,g.pool,g.queue,&vec![100.0+level as f32;125],[5;3],None).unwrap(),
                GpuTexture::upload_volume_u32(device,&mut alloc,g.pool,g.queue,&vec![70+level as u32;125],[5;3],None).unwrap()));
        }
        for (binding,sampler) in [(0,linear),(1,nearest)] {
            let infos:Vec<_>=textures.iter().map(|(d,i)|vk::DescriptorImageInfo::default().sampler(sampler)
                .image_view(if binding==0 {d.view} else {i.view}).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)).collect();
            unsafe {device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding)
                .descriptor_type(sampled).image_info(&infos)],&[]);}
        }
        uniform.write(0,bytemuck::cast_slice(&[domains[0],domains[4],[0.0,0.0,1.0,0.0]]));
        let queries=[0.0f32,1.0,1.5,3.0,6.0].map(|x|[[x,0.0,0.0,0.0],[0.0;4]]);
        input.write(0,bytemuck::cast_slice(&queries));
        for (mask,expected) in [(15,[0,0,1,2,3]),(13,[0,0,2,2,3])] {
            domains[8][0]=f32::from_bits(mask);clip_uniform.write(0,bytemuck::cast_slice(&domains));
            one_time_submit(device,g.pool,g.queue,|cb|unsafe {
                device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,pipeline);
                device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,layout,0,&[set],&[]);
                device.cmd_dispatch(cb,queries.len() as u32,1,1);
                device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::COMPUTE_SHADER,vk::PipelineStageFlags::HOST,vk::DependencyFlags::empty(),
                    &[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ)],&[],&[]);
            }).unwrap();
            let bytes=output.read();let results:Vec<[f32;4]>=bytemuck::pod_collect_to_vec(&bytes[..queries.len()*16]);
            for (i,(value,level)) in results.iter().zip(expected).enumerate() {
                checks+=1;
                if (value[0]-(100+level) as f32).abs()>0.001 || value[1]!=(70+level) as f32 || value[2]!=level as f32 {
                    failures.push(format!("clipmap selection mask={mask} query={:?} got={value:?} wanted level={level}",queries[i]));
                }
            }
        }
        // A deliberately empty coarse grid must not step over a fine resident
        // occluder. Exercise entry in both directions along all three axes.
        let mut fine=Vec::new();let wall=body(0.0,0.0,0.0,0.25,0.25,0.25);
        for z in 0..9 {for y in 0..9 {for x in 0..9 {fine.push(wall.distance(Vec3::new(x as f32,y as f32,z as f32)*0.25-Vec3::ONE));}}}
        let mut fine_texture=GpuTexture::upload_volume(device,&mut alloc,g.pool,g.queue,&fine,[9;3],None).unwrap();
        let info=[vk::DescriptorImageInfo::default().sampler(linear).image_view(fine_texture.view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
        unsafe {device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(sampled).image_info(&info)],&[]);}
        domains[8][0]=f32::from_bits(15);clip_uniform.write(0,bytemuck::cast_slice(&domains));
        uniform.write(0,bytemuck::cast_slice(&[domains[0],domains[4],[0.0;4]]));
        let queries:Vec<_>=[Vec3::X,Vec3::Y,Vec3::Z].into_iter().flat_map(|a|[-a,a]).map(|axis|[(axis*4.0).extend(0.0).to_array(),(-axis).extend(8.0).to_array()]).collect();
        input.write(0,bytemuck::cast_slice(&queries));
        one_time_submit(device,g.pool,g.queue,|cb|unsafe {
            device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,pipeline);
            device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,layout,0,&[set],&[]);
            device.cmd_dispatch(cb,queries.len() as u32,1,1);
            device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::COMPUTE_SHADER,vk::PipelineStageFlags::HOST,vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ)],&[],&[]);
        }).unwrap();
        let bytes=output.read();let results:Vec<[f32;4]>=bytemuck::pod_collect_to_vec(&bytes[..queries.len()*16]);
        for (i,value) in results.iter().enumerate() {
            checks+=1;
            if (value[0]-3.75).abs()>0.025 || value[3]<0.5 {
                failures.push(format!("coarse-to-fine entry query={:?} got={value:?}",queries[i]));
            }
        }
        fine_texture.destroy(device,&mut alloc);
        for (mut d,mut i) in textures {d.destroy(device,&mut alloc);i.destroy(device,&mut alloc);}
    }
    clip_uniform.destroy(device,&mut alloc);
    cache.destroy(device,&mut alloc);input.destroy(device,&mut alloc);output.destroy(device,&mut alloc);uniform.destroy(device,&mut alloc);
    unsafe {device.destroy_sampler(linear,None);device.destroy_sampler(nearest,None);device.destroy_pipeline(pipeline,None);device.destroy_shader_module(shader,None);
        device.destroy_descriptor_pool(pool,None);device.destroy_pipeline_layout(layout,None);device.destroy_descriptor_set_layout(set_layout,None);}
    eprintln!("[Flux software coverage] multilevel={multilevel} pending={pending} {checks} production geometry queries; failures={}",failures.len());
    assert!(failures.is_empty(),"{}",failures.join("\n"));
    assert_eq!(g.validation_error_count(),before,"coverage test produced Vulkan errors");
}

/// An offscreen emitter above a thin ceiling: neither screen traces nor an
/// analytic light can hide a world-trace leak. Both cameras sample below it.
pub(super) fn scene(r:&mut crate::Renderer)->anyhow::Result<()> {scene_cases(r,false,false)}
pub(super) fn blended_scene(r:&mut crate::Renderer)->anyhow::Result<()> {scene_cases(r,true,false)}
pub(super) fn radiosity_range(r:&mut crate::Renderer)->anyhow::Result<()> {scene_cases(r,false,true)}
fn scene_cases(r:&mut crate::Renderer,blending:bool,ranges:bool)->anyhow::Result<()> {
    use crate::{CameraData,FluxSettings,GiSource};
    use super::planar::{cube,draw,lit};
    r.set_anti_aliasing(crate::AntiAliasing::None);r.set_area_lights(&[]);r.gi_set_emitters(&[]);
    let mesh=r.upload_mesh(&cube())?;
    let floor=lit(r,[0.8;4],[0.0;3],1.0,0.0);
    let source=lit(r,[0.0,0.0,0.0,1.0],[4.0,1.0,0.5],1.0,0.0);
    r.set_material_features(source,crate::MaterialFeatures {emission:true,
        alpha_mode:if blending {crate::AlphaMode::Blend} else {crate::AlphaMode::Opaque},..Default::default()});
    let black=lit(r,[0.0,0.0,0.0,1.0],[0.0;3],1.0,0.0);
    let floor_body=body(0.0,-0.1,0.0,8.0,0.1,8.0);
    let source_body=body(0.0,8.0,0.0,20.0,0.02,20.0);
    let wall=body(0.0,3.0,0.0,20.0,0.02,20.0);
    let volume=super::flux::cube_distance_field();
    let camera=|eye|CameraData {position:eye,view:Mat4::look_at_rh(eye,Vec3::ZERO,Vec3::Y),proj:Mat4::perspective_rh(1.0,16.0/9.0,0.1,100.0)};
    let cams=[camera(Vec3::new(0.0,2.5,3.0)),camera(Vec3::new(1.0,2.8,3.5))];
    let lo=Vec3::new(-24.0,-4.0,-24.0);let hi=Vec3::new(24.0,20.0,24.0);let dims=[9;3];
    for hardware in [false,true] {
        if hardware && !r.ctx.ray_tracing() {continue;}
        let settings=FluxSettings {rt_trace:hardware,gi_screen_traces:false,reflection_screen_traces:false,samples:64,bounces:2,
            march_distance:0.0,reflection_mode:0,ssr_enabled:false,..Default::default()};
        let mut results=Vec::new();
        let states=if ranges {vec![(false,false)]} else if blending {vec![(false,false),(true,false),(true,true),(true,false),(false,false)]}
            else {vec![(false,false),(true,false),(false,false)]};
        for &(present,blended) in &states {
            let blocked=present && !blended;
            r.set_material_features(black,crate::MaterialFeatures {alpha_mode:if blended {crate::AlphaMode::Blend} else {crate::AlphaMode::Opaque},..Default::default()});
            r.material_params_mut(black).base_color[3]=if blended {0.25} else {1.0};
            r.set_gi_enabled(true);r.set_flux_settings(settings);
            r.material_params_mut(floor).metallic=0.0;r.material_params_mut(floor).roughness=1.0;
            let mut geometry=vec![(floor_body,floor),(source_body,source)];if present {geometry.push((wall,black));}
            let draws:Vec<_>=geometry.iter().enumerate().map(|(i,(b,m))| {
                let mut d=draw(mesh,*m,b.transform(),-1);d.gi_source=Some(GiSource {object:0xc0b0+i as u128,slot:0});d
            }).collect();
            // Mirrors gather_realtime_gi's production input policy, separately
            // tested in the engine with multi-slot objects and live mode edits.
            let fields:Vec<_>=geometry.iter().zip(&draws).enumerate().filter(|(i,_)|!blended || *i!=2).map(|(i,((b,_),d))|GiMeshSdf {
                gi_source:d.gi_source,world_to_local:b.transform().inverse(),volume:volume.clone(),in_global_field:i==2,
                material:GpuGiMaterial {albedo:if i==0 {[0.8;3]} else {[0.0;3]},emission:if i==1 {[4.0,1.0,0.5]} else {[0.0;3]},metallic:0.0,roughness:1.0}
            }).collect();
            let (lo,hi)=if blocked {(lo,hi)} else {(-Vec3::ONE,Vec3::ONE)};
            let mut distance=Vec::new();for z in 0..9 {for y in 0..9 {for x in 0..9 {
                let p=lo+(hi-lo)*Vec3::new(x as f32,y as f32,z as f32)/8.0;
                distance.push(if blocked {wall.distance(p)} else {100.0});
            }}}
            let material=GpuGiMaterial {albedo:[0.0;3],emission:[0.0;3],metallic:0.0,roughness:1.0};
            let ids=if blocked {vec![draws[2].gi_source]} else {Vec::new()};
            r.gi_set_gdf_identified(&distance,&[0;729],dims,lo.to_array(),hi.to_array(),&[material],&ids)?;
            r.gi_set_mesh_sdfs(&fields)?;
            let mut input=crate::cascade_tests::input(cams[0]);input.camera_preview=Some(cams[1]);input.viewport_extent=Some([323,181]);input.draws=&draws;
            input.light.intensity=0.0;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;input.draw_skybox=false;
            if ranges {
                // Read the world-space bounce cache itself: a lit screen can
                // otherwise hide broken radiosity by tracing the source directly.
                for (scene_range,camera_range,lit) in [(0.0,0.0,true),(2.0,2.0,false),
                    (2.0,40.0,true),(40.0,2.0,true),(2.0,2.0,false),(0.0,0.0,true)] {
                    input.postfx.flux.max_trace_distance=Some(scene_range);
                    input.preview_postfx.flux.max_trace_distance=Some(camera_range);
                    for _ in 0..64 {r.render(&input)?;}
                    let cached={let allocator=r.allocator();let mut alloc=allocator.lock().unwrap();
                        r.flux_surface.as_ref().unwrap().debug_lighting(&r.ctx.device,&mut alloc,r.command_pool,r.ctx.queue,floor.0,2)?};
                    eprintln!("[Flux radiosity range] hardware={hardware} ranges=({scene_range},{camera_range}) cache={cached:?}");
                    assert!(cached[0][0].abs()<0.001,"fixture unexpectedly has direct light");
                    assert!(cached[1][0].is_finite() && if lit {cached[1][0]>0.5} else {cached[1][0]<0.001},
                        "surface bounce range ignored or stale: hardware={hardware} ranges=({scene_range},{camera_range}) cache={cached:?}");
                }
                // Reflections own the same cache when diffuse GI is disabled.
                // Their bounce transport must also respect explicit/auto limits.
                r.set_gi_enabled(false);
                for distance in [2.0,40.0,2.0,0.0] {
                    r.set_flux_settings(FluxSettings {reflection_mode:2,ssr_max_distance:distance,..settings});
                    for _ in 0..64 {r.render(&input)?;}
                    let cached={let allocator=r.allocator();let mut alloc=allocator.lock().unwrap();
                        r.flux_surface.as_ref().unwrap().debug_lighting(&r.ctx.device,&mut alloc,r.command_pool,r.ctx.queue,floor.0,2)?};
                    eprintln!("[Flux reflection radiosity range] hardware={hardware} distance={distance} cache={cached:?}");
                    assert!(cached[1][0].is_finite() && if distance==2.0 {cached[1][0]<0.001} else {cached[1][0]>0.5},
                        "standalone reflection bounce range ignored or stale: {cached:?}");
                }
                continue;
            }
            for _ in 0..64 {r.render(&input)?;}
            let gi=[super::flux_radiance::hdr(r,0,cams[0])?,super::flux_radiance::hdr(r,1,cams[1])?];
            r.set_gi_enabled(false);r.material_params_mut(floor).metallic=1.0;r.material_params_mut(floor).roughness=0.02;
            r.set_flux_settings(FluxSettings {reflection_mode:2,rt_samples:1,rt_bounces:1,..settings});
            for _ in 0..12 {r.render(&input)?;}
            let mut reflection=[0.0;2];for (view, view_value) in reflection.iter_mut().enumerate() {
                let pixels=super::reflection_denoise::history(r,view,Vec3::ZERO,false)?;
                *view_value=pixels.iter().sum::<f32>()/pixels.len() as f32;
            }
            eprintln!("[Flux coverage scene] hardware={hardware} blocked={blocked} blended={blended}: GI={gi:?} reflection={reflection:?}");
            results.push([gi[0],gi[1],reflection[0],reflection[1]]);
        }
        if ranges {continue;}
        for channel in 0..4 {
            assert!(results[0][channel]>0.05,"offscreen emitter did not light fixture: {results:?}");
            for (i,&(present,blended)) in states.iter().enumerate().skip(1) {
                if present && !blended {
                    assert!(results[i][channel]<results[0][channel]*0.02,"opaque ceiling leaked: {results:?}");
                } else {
                    assert!(results[i][channel]>results[0][channel]*0.8,"removed/alpha-blended ceiling retained opaque occlusion: {results:?}");
                }
            }
        }
        if blending {
            // Publish the same empty-field snapshot as the engine when its
            // last opaque input disappears. A raster receiver remains only
            // so stale world radiance can be measured, with no emitter draw.
            r.gi_set_gdf_identified(&[1e6;8],&[0;8],[2;3],[-1.0;3],[1.0;3],&[],&[])?;
            r.gi_set_mesh_sdfs(&[])?;r.gi_set_emitters(&[]);
            r.set_gi_enabled(true);r.set_flux_settings(settings);
            r.material_params_mut(floor).metallic=0.0;r.material_params_mut(floor).roughness=1.0;
            let draws=[draw(mesh,floor,floor_body.transform(),-1)];
            let mut input=crate::cascade_tests::input(cams[0]);input.camera_preview=Some(cams[1]);input.viewport_extent=Some([323,181]);input.draws=&draws;
            input.light.intensity=0.0;input.light.ambient=[0.0;3];input.light.env_intensity=0.0;input.draw_skybox=false;
            for _ in 0..64 {r.render(&input)?;}
            let empty=[super::flux_radiance::hdr(r,0,cams[0])?,super::flux_radiance::hdr(r,1,cams[1])?];
            eprintln!("[Flux empty scene] hardware={hardware} GI={empty:?}");
            assert!(empty.iter().all(|v|*v<0.001),"empty fields retained old emissive lighting: {empty:?}");
        }
    }
    Ok(())
}
