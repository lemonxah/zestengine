//! Exercise the production cache interpolation against invalid geometry and
//! bright neighbouring pages using real sampled textures and a compute dispatch.
use ash::vk;
use gpu_allocator::MemoryLocation;
use crate::{alloc::{Buffer,one_time_submit},texture::GpuTexture};

#[test]
fn flux_surface_filter_rejects_unrelated_geometry_and_preserves_dark_samples() {
    let Some(g)=super::gpu() else {return};
    let device=&g.device; let mut alloc=g.allocator().unwrap();
    const W:usize=128; const H:usize=64; const LW:usize=32; const LH:usize=16;
    let geometry_z=f32::from_bits(0x3fc00000|1023|(1023<<11));
    let geometry_y=f32::from_bits(0x3fc00000|1023|(2046<<11));
    let mut positions=vec![[0.0f32;4];W*H];
    let mut normals=vec![[0.0,0.0,1.0,1.0];W*H];
    for y in 0..H {for x in 0..W {positions[y*W+x]=[(x%64) as f32*0.01,y as f32*0.01,0.0,geometry_z];}}
    for y in 0..H {for x in 64..W { normals[y*W+x]=[0.0,1.0,0.0,1.0]; }}
    let mut direct=vec![[0.2f32,0.4,0.6,1.0];LW*LH];
    let mut indirect=vec![[0.0f32;4];LW*LH];
    for y in 0..LH {for x in 16..LW {direct[y*LW+x]=[10.0,0.0,0.0,1.0];}}
    let bases=[(1,1),(5,1),(9,1),(1,5),(5,5),(9,5),(1,9),(5,9),(9,9),(13,1)];
    let mut queries=Vec::new();
    for (case,&(x,y)) in bases.iter().enumerate() {
        queries.push([4.0*(x as f32+1.0),4.0*(y as f32+1.0),0.0,0.0]);
        let material=(y*4+2)*W+x*4+2;
        match case {
            1=>{positions[material][3]=0.0;direct[y*LW+x]=[10.0,0.0,0.0,1.0];}
            2=>{positions[material][2]=1.0;direct[y*LW+x]=[10.0,0.0,0.0,1.0];}
            3=>{positions[material][3]=geometry_y;normals[material]=[0.0,1.0,0.0,1.0];direct[y*LW+x]=[10.0,0.0,0.0,1.0];}
            4=>for j in 0..2 {for i in 0..2 {positions[((y+j)*4+2)*W+(x+i)*4+2][3]=0.0;}},
            5=>direct[y*LW+x]=[0.0,0.0,0.0,1.0],
            6=>queries[case]=[63.9,56.0,0.0,0.0], // no leakage from neighbouring card
            7=>for j in 0..2 {for i in 0..2 {direct[(y+j)*LW+x+i]=[0.0,0.0,0.0,1.0];}},
            8=>for j in 0..2 {for i in 0..2 {indirect[(y+j)*LW+x+i]=[0.1,0.1,0.1,1.0];}},
            9=>for j in 0..2 {for i in 0..2 {normals[((y+j)*4+2)*W+(x+i)*4+2]=[0.8,0.0,0.6,1.0];}},
            _=>{}
        }
    }
    let axes=[[1.0f32,0.0,0.0],[-1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,-1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,-1.0]];
    queries.extend(axes.map(|n|[n[0],n[1],n[2],2.0]));
    // Global lookup sees two overlapping surfaces. Instance lookup must select
    // the exact object, even when a nearer/identical-position neighbour exists.
    // Draw0 is uncached, draw1 selects card0, draw2 selects card1. Index3 is OOB.
    queries.extend((3..=8).map(|mode|[0.24,0.48,0.005,mode as f32]));
    queries.push([0.24,0.48,0.2,7.0]); // keep the geometric distance rejection
    // A direct-only page must not expose stale indirect atlas contents.
    queries.push([40.0,40.0,0.0,1.0]);
    // SDF snapshot order differs from draw order: card1, known missing,
    // card0, unknown legacy identity, then an out-of-range index.
    queries.extend((10..=14).map(|mode|[0.24,0.48,0.005,mode as f32]));
    queries.extend((20..=24).map(|mode|[0.24,0.48,0.005,mode as f32]));
    let moved=glam::Mat4::from_scale_rotation_translation(glam::Vec3::splat(2.0),
        glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),glam::Vec3::new(2.0,-1.0,3.0));
    for mode in 30..=33 {
        let p=moved.transform_point3(glam::Vec3::new(0.24,0.48,if mode==33 {0.2} else {0.005}));
        queries.push(p.extend(mode as f32).to_array());
    }
    let hit_cases=queries.len();
    queries.extend([[40.0,40.0,0.0,40.0],[40.0,40.0,0.0,41.0],[8.0,8.0,0.0,40.0]]);
    let albedo=vec![[0.5f32,0.25,1.0,0.5];W*H];
    let mut textures=Vec::new();
    for (w,h,data) in [(W,H,&positions),(W,H,&normals),(LW,LH,&direct),(LW,LH,&indirect),(W,H,&albedo)] {
        textures.push(GpuTexture::upload_pixels(device,&mut alloc,g.pool,g.queue,w as u32,h as u32,
            vk::Format::R32G32B32A32_SFLOAT,bytemuck::cast_slice(data)).unwrap());
    }
    // Exercise the production nearest-receiver mip path as well as the
    // interpolation itself. Packed normal bits must survive this exact copy.
    let position_mips=GpuTexture::cleared_mips(device,&mut alloc,g.pool,g.queue,W as u32,H as u32,
        vk::Format::R32G32B32A32_SFLOAT,vk::ImageUsageFlags::TRANSFER_SRC,3).unwrap();
    one_time_submit(device,g.pool,g.queue,|cb|unsafe {
        let barrier=|image,from,to|vk::ImageMemoryBarrier::default().image(image).old_layout(from).new_layout(to)
            .src_access_mask(vk::AccessFlags::MEMORY_READ|vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ|vk::AccessFlags::MEMORY_WRITE)
            .subresource_range(vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1));
        let read=vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;let src=vk::ImageLayout::TRANSFER_SRC_OPTIMAL;let dst=vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::ALL_COMMANDS,vk::PipelineStageFlags::TRANSFER,vk::DependencyFlags::empty(),&[],&[],
            &[barrier(textures[0].image,read,src),barrier(position_mips.image,read,dst)]);
        let sub=vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1);
        let copy=vk::ImageCopy::default().src_subresource(sub).dst_subresource(sub).extent(vk::Extent3D{width:W as u32,height:H as u32,depth:1});
        device.cmd_copy_image(cb,textures[0].image,src,position_mips.image,dst,&[copy]);
        device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::TRANSFER,vk::PipelineStageFlags::ALL_COMMANDS,vk::DependencyFlags::empty(),&[],&[],
            &[barrier(textures[0].image,src,read),barrier(position_mips.image,dst,read)]);
        crate::flux_surface::prepare_lighting_geometry(device,cb,position_mips.image,vk::Extent2D{width:W as u32,height:H as u32});
    }).unwrap();
    textures[0].destroy(device,&mut alloc);textures[0]=position_mips;
    let mut card=[0u32;40];
    for i in [24,29,34,39] {card[i]=1.0f32.to_bits();}
    for (i,value) in [(0,3.125f32),(5,3.125),(10,500.0),(12,-1.0),(13,-1.0),(14,0.5),(15,1.0),(18,1.0),(19,0.01)] {card[i]=value.to_bits();}
    card[22]=64;card[23]=4|32;
    let mut second=card;second[20]=64;
    let mut direct_only=card;direct_only[23]=4;
    let mut attached=card;attached[23]|=64;
    let projection=glam::Mat4::from_cols_array(&std::array::from_fn(|i|f32::from_bits(card[i])))*moved.inverse();
    attached[..16].copy_from_slice(&projection.to_cols_array().map(f32::to_bits));
    attached[16]=1.0f32.to_bits();attached[18]=0.0f32.to_bits();attached[19]=0.02f32.to_bits();
    attached[24..].copy_from_slice(&moved.to_cols_array().map(f32::to_bits));
    let mut cards=Buffer::new(device,&mut alloc,640,vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::CpuToGpu,"filter cards").unwrap();
    cards.write(0,bytemuck::cast_slice(&[card,second,direct_only,attached]));
    let node=|left,right| [(-1.0f32).to_bits(),(-1.0f32).to_bits(),(-0.1f32).to_bits(),left,
        1.0f32.to_bits(),1.0f32.to_bits(),0.1f32.to_bits(),right];
    let tree=[node(0,3),node(0x80000000,2),node(0x80000001,3),node(0,0),node(2,0),node(3,0),
        node(7,4),node(3,1),node(0,1),node(2,1),node(0,0),
        node(12,4),node(2,1),node(0,1),node(3,1),node(0,0),
        [1.9f32.to_bits(),(-1.1f32).to_bits(),1.6f32.to_bits(),0x80000003,2.1f32.to_bits(),0.4f32.to_bits(),3.1f32.to_bits(),17]];
    let mut nodes=Buffer::new(device,&mut alloc,16+tree.len() as u64*32,vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::CpuToGpu,"filter instance tree").unwrap();
    nodes.write(0,bytemuck::cast_slice(&[3u32,3,0,0]));nodes.write(16,bytemuck::cast_slice(&tree));
    let mut query=Buffer::new(device,&mut alloc,(queries.len()*16) as u64,vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::CpuToGpu,"filter queries").unwrap();
    query.write(0,bytemuck::cast_slice(&queries));
    let mut output=Buffer::new(device,&mut alloc,(queries.len()*16) as u64,vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::GpuToCpu,"filter results").unwrap();
    let ty=|binding| if [0,1,2,9].contains(&binding) {vk::DescriptorType::STORAGE_BUFFER} else {vk::DescriptorType::COMBINED_IMAGE_SAMPLER};
    let bindings:Vec<_>=(0..10).map(|binding|vk::DescriptorSetLayoutBinding::default().binding(binding)
        .descriptor_type(ty(binding)).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
    unsafe {
        let set_layout=device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),None).unwrap();
        let layout=device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]),None).unwrap();
        let code=ash::util::read_spv(&mut std::io::Cursor::new(include_bytes!(concat!(env!("OUT_DIR"),"/flux_surface_filter_test.comp.spv")))).unwrap();
        let module=device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code),None).unwrap();
        let pipeline=device.create_compute_pipelines(vk::PipelineCache::null(),&[vk::ComputePipelineCreateInfo::default().layout(layout)
            .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main"))],None).unwrap()[0];
        let pool=device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
            vk::DescriptorPoolSize {ty:vk::DescriptorType::STORAGE_BUFFER,descriptor_count:4},
            vk::DescriptorPoolSize {ty:vk::DescriptorType::COMBINED_IMAGE_SAMPLER,descriptor_count:6}]),None).unwrap();
        let set=device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[set_layout])).unwrap()[0];
        let sampler=device.create_sampler(&vk::SamplerCreateInfo::default().min_filter(vk::Filter::LINEAR).mag_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE).address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),None).unwrap();
        for (binding,handle) in [(0,output.handle),(1,cards.handle),(2,nodes.handle),(9,query.handle)] {
            let info=[vk::DescriptorBufferInfo::default().buffer(handle).range(vk::WHOLE_SIZE)];
            device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info)],&[]);
        }
        for (binding,index) in [(3,0),(4,1),(5,4),(6,0),(7,2),(8,3)] {
            let info=[vk::DescriptorImageInfo::default().sampler(sampler).image_view(textures[index].view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&info)],&[]);
        }
        one_time_submit(device,g.pool,g.queue,|cb| {
            device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,pipeline);
            device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,layout,0,&[set],&[]);
            device.cmd_dispatch(cb,queries.len() as u32,1,1);
            device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::COMPUTE_SHADER,vk::PipelineStageFlags::HOST,vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ)],&[],&[]);
        }).unwrap();
        let bytes=output.read();let values:&[f32]=bytemuck::cast_slice(&bytes[..queries.len()*16]);
        for case in 0..queries.len() {
            let expected=if case>=hit_cases {if case==hit_cases {[0.2,0.4,0.1,1.0]} else {[0.0;4]}} else if case>=34 {match case {34=>[1.0,1.0,2.0,1.0],35=>[0.2,0.4,0.6,1.0],_=>[0.0;4]}} else if case>=29 {match case {29|32=>[0.0,0.0,1.0,1.0],31=>[1.0,1.0,1.0,1.0],_=>[0.0;4]}} else if case>=24 {match case {24=>[1.0,1.0,1.0,1.0],26|27=>[0.0,0.0,1.0,1.0],_=>[0.0;4]}} else if case==23 {[0.2,0.4,0.6,1.0]} else if case>=16 {match case {16|19=>[0.0,0.0,1.0,1.0],20=>[1.0,1.0,1.0,1.0],_=>[0.0;4]}} else if case>=10 {let n=axes[case-10];[n[0],n[1],n[2],1.0]} else {match case {4=>[0.0;4],5=>[0.15,0.3,0.45,1.0],7=>[0.0,0.0,0.0,1.0],8=>[0.3,0.5,0.7,1.0],_=>[0.2,0.4,0.6,1.0]}};
            let actual=&values[case*4..case*4+4];
            eprintln!("[Flux surface filter] case {case}: {actual:?}");
            for i in 0..4 {assert!((actual[i]-expected[i]).abs()<if case>=10 {0.002} else {1e-5},"case {case}: {actual:?} != {expected:?}");}
        }
        device.destroy_sampler(sampler,None);device.destroy_descriptor_pool(pool,None);device.destroy_pipeline(pipeline,None);
        device.destroy_shader_module(module,None);device.destroy_pipeline_layout(layout,None);device.destroy_descriptor_set_layout(set_layout,None);
    }
    for t in &mut textures {t.destroy(device,&mut alloc);}
    for b in [&mut cards,&mut nodes,&mut query,&mut output] {b.destroy(device,&mut alloc);}
}
