//! Production GPU lookup: mip selection must not replace geometric identity.
use super::*;
use crate::{alloc::one_time_submit,texture::GpuTexture};

#[test]
fn virtual_surface_lookup_prefers_detail_without_crossing_geometry() {
    let Some(g)=crate::gpu_tests::gpu() else {return;};
    let validation_before=crate::cluster_headless::VALIDATION_ERRORS.load(std::sync::atomic::Ordering::Relaxed);
    let device=&g.device;let mut alloc=g.allocator().unwrap();
    const W:usize=512;const H:usize=128;
    let mut positions=vec![[0.0f32;4];W*H];let mut normals=positions.clone();
    let albedo=vec![[0.5f32,0.5,0.5,0.0];W*H];let emission=vec![[0.0f32;4];W*H];
    for y in 0..H {for x in 0..W {
        positions[y*W+x]=[((x%128) as f32+0.5)/128.0,(y as f32+0.5)/128.0,0.25,if x>=384 {0.0} else {1.0}];
        normals[y*W+x]=if (256..384).contains(&x) {[0.0,1.0,0.0,1.0]} else {[0.0,0.0,1.0,1.0]};
    }}
    // Coarse is bright; fine is valid black. A max-brightness selection or
    // invalid-as-black fallback cannot satisfy both lookup and radiance checks.
    let direct:Vec<_>=(0..W/4*H/4).map(|i|[if i%(W/4)<32 {0.5} else {0.0},0.0,0.0,1.0]).collect();
    let indirect=vec![[0.0f32,0.0,0.0,1.0];W/4*H/4];
    let mut textures=Vec::new();
    for (w,h,data) in [(W,H,&positions),(W,H,&normals),(W,H,&albedo),(W,H,&emission),(W/4,H/4,&direct),(W/4,H/4,&indirect)] {
        textures.push(GpuTexture::upload_pixels(device,&mut alloc,g.pool,g.queue,w as u32,h as u32,
            vk::Format::R32G32B32A32_SFLOAT,bytemuck::cast_slice(data)).unwrap());
    }
    let projection=Mat4::from_scale_rotation_translation(Vec3::new(2.0,2.0,1.0),glam::Quat::IDENTITY,Vec3::new(-1.0,-1.0,0.0));
    let coarse=Card {view_proj:projection.to_cols_array_2d(),normal_texel:[0.0,0.0,1.0,1.0/128.0],
        rect:[0,0,128,55],capture_to_world:Mat4::IDENTITY.to_cols_array_2d(),virtual_page:[1,0,0,0],neighbours:[0;4]};
    let mut fine=coarse;fine.rect[0]=128;fine.rect[3]|=64;fine.virtual_page=[1,1,0,0];
    fine.capture_to_world=Mat4::from_translation(Vec3::new(0.003,0.0,0.0)).to_cols_array_2d();
    let mut unrelated=fine;unrelated.virtual_page[0]=3;
    let mut other_layer=fine;other_layer.capture_to_world=Mat4::from_translation(Vec3::new(0.003,0.0,0.01)).to_cols_array_2d();
    let mut other_normal=fine;other_normal.rect[0]=256;
    let mut hole=fine;hole.rect[0]=384;
    let mut finest=fine;finest.virtual_page[1]=2;
    let data=[coarse,fine,unrelated,other_layer,other_normal,hole,finest];
    let mut nodes_data=Vec::new();let mut queries=Vec::new();let mut expected=Vec::new();
    let middle=64.5/128.0;
    for (pair,p,winner) in [([0,1],[middle,middle,0.25],1),([1,0],[middle,middle,0.25],1),
        ([0,2],[middle,middle,0.25],0),([2,0],[middle,middle,0.25],0),
        ([0,3],[middle,middle,0.25],0),([3,0],[middle,middle,0.25],0),
        ([0,4],[middle,middle,0.25],0),([0,5],[middle,middle,0.25],0),
        ([0,1],[0.02,middle,0.25],0),([1,6],[middle,middle,0.25],6),([6,1],[middle,middle,0.25],6)] {
        let start=nodes_data.len() as u32;
        for (offset,card) in pair.into_iter().enumerate() {nodes_data.push(Node {min:[-1.0;3],max:[2.0;3],left:card|0x80000000,right:start+offset as u32+1});}
        queries.push([p[0],p[1],p[2],start as f32]);
        expected.push([winner as f32,data[winner].virtual_page[1] as f32,if winner==0 {0.5} else {0.0},1.0]);
    }
    let views:Vec<_>=textures.iter().map(|t|t.view).collect();
    let values=sample(device,&mut alloc,g.pool,g.queue,&data,&nodes_data,&views,&queries,
        include_bytes!(concat!(env!("OUT_DIR"),"/flux_surface_lookup_test.comp.spv")),9);
    for (i,(actual,expected)) in values.iter().zip(&expected).enumerate() {
        for c in 0..4 {assert!((actual[c]-expected[c]).abs()<1e-4,"virtual lookup case{i}: {actual:?}, expected{expected:?}");}
    }
    eprintln!("[Flux virtual lookup]{} production GPU queries passed: fine mip preference, traversal order, unrelated geometry, missing coverage, valid black",queries.len());
    for texture in &mut textures {texture.destroy(device,&mut alloc);}
    assert_eq!(crate::cluster_headless::VALIDATION_ERRORS.load(std::sync::atomic::Ordering::Relaxed),validation_before);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sample(device:&ash::Device,alloc:&mut Allocator,command_pool:vk::CommandPool,gpu_queue:vk::Queue,
    data:&[Card],nodes_data:&[Node],views:&[vk::ImageView],queries:&[[f32;4]],shader:&[u8],query_binding:u32)->Vec<[f32;4]> {
    sample_with_mesh(device,alloc,command_pool,gpu_queue,data,nodes_data,views,queries,shader,query_binding,None)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn sample_with_mesh(device:&ash::Device,alloc:&mut Allocator,command_pool:vk::CommandPool,gpu_queue:vk::Queue,
    data:&[Card],nodes_data:&[Node],views:&[vk::ImageView],queries:&[[f32;4]],shader:&[u8],query_binding:u32,
    mesh:Option<&crate::mesh_sdf::Snapshot>)->Vec<[f32;4]> {
    let mut cards=Buffer::new(device,alloc,size_of_val(data) as u64,vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::CpuToGpu,"Virtual lookup cards").unwrap();
    cards.write(0,bytemuck::cast_slice(data));
    let result;
    let mut nodes=Buffer::new(device,alloc,(16+size_of_val(nodes_data)) as u64,vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::CpuToGpu,"Virtual lookup tree").unwrap();
    nodes.write(0,bytemuck::cast_slice(&[0u32,0,0,nodes_data.len() as u32]));nodes.write(16,bytemuck::cast_slice(nodes_data));
    let mut query=Buffer::new(device,alloc,(queries.len()*16) as u64,vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::CpuToGpu,"Virtual lookup queries").unwrap();
    query.write(0,bytemuck::cast_slice(queries));
    let mut output=Buffer::new(device,alloc,(queries.len()*16) as u64,vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::GpuToCpu,"Virtual lookup results").unwrap();
    unsafe {
        let bindings:Vec<_>=[0,1,2,3,4,5,6,7,8,query_binding].into_iter().chain(mesh.into_iter().flat_map(|_|[10,11,12])).map(|binding|vk::DescriptorSetLayoutBinding::default().binding(binding)
            .descriptor_type(if [0,1,2,query_binding,10,11].contains(&binding) {vk::DescriptorType::STORAGE_BUFFER} else {vk::DescriptorType::COMBINED_IMAGE_SAMPLER})
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
        let set_layout=device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),None).unwrap();
        let layout=device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]),None).unwrap();
        let module=crate::screen_pass::shader_module(device,shader).unwrap();
        let pipeline=device.create_compute_pipelines(vk::PipelineCache::null(),&[vk::ComputePipelineCreateInfo::default().layout(layout)
            .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main"))],None).unwrap()[0];
        let pool=device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
            vk::DescriptorPoolSize {ty:vk::DescriptorType::STORAGE_BUFFER,descriptor_count:6},
            vk::DescriptorPoolSize {ty:vk::DescriptorType::COMBINED_IMAGE_SAMPLER,descriptor_count:7}]),None).unwrap();
        let set=device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[set_layout]),).unwrap()[0];
        let sampler=device.create_sampler(&vk::SamplerCreateInfo::default().min_filter(vk::Filter::NEAREST).mag_filter(vk::Filter::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE).address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),None).unwrap();
        for (binding,buffer) in [(0,output.handle),(1,cards.handle),(2,nodes.handle),(query_binding,query.handle)] {
            let info=[vk::DescriptorBufferInfo::default().buffer(buffer).range(vk::WHOLE_SIZE)];
            device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info)],&[]);
        }
        for (index,&view) in views.iter().enumerate() {
            let info=[vk::DescriptorImageInfo::default().sampler(sampler).image_view(view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(index as u32+3).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&info)],&[]);
        }
        let mesh_sampler=mesh.map(|mesh| {
            let sampler=device.create_sampler(&vk::SamplerCreateInfo::default().min_filter(vk::Filter::LINEAR).mag_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE).address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),None).unwrap();
            let (buffers,view)=mesh.resources();
            for (i,buffer) in buffers.into_iter().enumerate() {
                let info=[vk::DescriptorBufferInfo::default().buffer(buffer).range(vk::WHOLE_SIZE)];
                device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(10+i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info)],&[]);
            }
            let info=[vk::DescriptorImageInfo::default().sampler(sampler).image_view(view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(12)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&info)],&[]);
            sampler
        });
        one_time_submit(device,command_pool,gpu_queue,|cb| {
            device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,pipeline);
            device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,layout,0,&[set],&[]);
            device.cmd_dispatch(cb,queries.len() as u32,1,1);
            device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::COMPUTE_SHADER,vk::PipelineStageFlags::HOST,vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ)],&[],&[]);
        }).unwrap();
        let bytes=output.read();let values:&[[f32;4]]=bytemuck::cast_slice(&bytes[..queries.len()*16]);
        result=values.to_vec();
        if let Some(sampler)=mesh_sampler {device.destroy_sampler(sampler,None);}
        device.destroy_sampler(sampler,None);device.destroy_descriptor_pool(pool,None);device.destroy_pipeline(pipeline,None);
        device.destroy_shader_module(module,None);device.destroy_pipeline_layout(layout,None);device.destroy_descriptor_set_layout(set_layout,None);
    }
    for buffer in [&mut cards,&mut nodes,&mut query,&mut output] {buffer.destroy(device,alloc);}
    result
}
