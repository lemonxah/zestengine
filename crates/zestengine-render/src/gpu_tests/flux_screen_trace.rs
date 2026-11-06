//! Analytic ray/plane fixtures and independent min/max reductions, including
//! odd edge pixels, both viewport orientations and a one-pixel distant target.
use ash::vk;
use glam::{Mat4,Vec2,Vec3,Vec4};
use gpu_allocator::MemoryLocation;
use crate::{alloc::{Buffer,one_time_submit},texture::GpuTexture,flux_depth::DepthReduce};
#[repr(C)]
#[derive(Clone,Copy,bytemuck::Pod,bytemuck::Zeroable)]
struct Query {projection:[[f32;4];4],inverse_projection:[[f32;4];4],origin:[f32;4],direction:[f32;4]}

#[test]
fn flux_screen_hierarchy_traces_thin_edges_without_silhouette_bridges() {
    let Some(g)=super::gpu() else {return};let device=&g.device;let mut alloc=g.allocator().unwrap();
    let mut reduce=DepthReduce::new(device).unwrap();
    unsafe {
        let bindings:Vec<_>=(0..4).map(|binding|vk::DescriptorSetLayoutBinding::default().binding(binding)
            .descriptor_type(if binding==3 {vk::DescriptorType::COMBINED_IMAGE_SAMPLER} else {vk::DescriptorType::STORAGE_BUFFER}).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
        let set_layout=device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),None).unwrap();
        let layout=device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]),None).unwrap();
        let module=crate::screen_pass::shader_module(device,include_bytes!(concat!(env!("OUT_DIR"),"/flux_screen_trace_test.comp.spv"))).unwrap();
        let pipeline=device.create_compute_pipelines(vk::PipelineCache::null(),&[vk::ComputePipelineCreateInfo::default().layout(layout)
            .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main"))],None).unwrap()[0];
        device.destroy_shader_module(module,None);
        let pool=device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
            vk::DescriptorPoolSize {ty:vk::DescriptorType::STORAGE_BUFFER,descriptor_count:3},vk::DescriptorPoolSize {ty:vk::DescriptorType::COMBINED_IMAGE_SAMPLER,descriptor_count:1}]),None).unwrap();
        let sampler=device.create_sampler(&vk::SamplerCreateInfo::default(),None).unwrap();
        let set=device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[set_layout])).unwrap()[0];
        for (w,h) in [(323u32,181u32),(1537,37),(1,1)] {
            let mut hierarchy=reduce.target(device,&mut alloc,vk::Extent2D{width:w,height:h}).unwrap();
            let mut readback=Buffer::new(device,&mut alloc,hierarchy.byte_len(),vk::BufferUsageFlags::TRANSFER_DST,MemoryLocation::GpuToCpu,"depth readback").unwrap();
            for ortho in [false,true] {for flip in [false,true] {for scenario in 0..7 {
                let projection=if ortho {Mat4::orthographic_rh(-4.0,4.0,-3.0,3.0,0.1,100.0)}
                    else {Mat4::perspective_rh(1.1,w as f32/h as f32,0.1,100.0)};
                let inverse=projection.inverse();
                let at=|uv:Vec2,z:f32| {
                    let mut xy=uv*2.0-Vec2::ONE;if flip {xy.y=-xy.y;}
                    let depth=projection.project_point3(Vec3::new(0.0,0.0,-z)).z;
                    let p=inverse*Vec4::new(xy.x,xy.y,depth,1.0);p.truncate()/p.w
                };
                let z=if scenario==3 {7.0} else {5.0};
                let d=projection.project_point3(Vec3::new(0.0,0.0,-z)).z;
                let mut pixels=vec![d;(w*h) as usize];
                let target=Vec2::new((w as f32-0.5)/w as f32,(h as f32-0.5)/h as f32);
                if scenario==1 {pixels.fill(1.0);pixels[(w*h-1) as usize]=d;}
                if scenario==2 || scenario==6 {
                    pixels.fill(if scenario==6 {projection.project_point3(Vec3::new(0.0,0.0,-9.0)).z} else {1.0});
                    let near=projection.project_point3(Vec3::new(0.0,0.0,-2.0)).z;
                    for y in 0..h {pixels[(y*w+w/2) as usize]=near;}
                }
                if scenario==4 {pixels.fill(1.0);}
                let hit_uv=if scenario==1 {target} else {Vec2::new(0.8,0.7)};
                let start=at(Vec2::new(0.1,0.15),if scenario==2 || scenario==6 {4.0} else {1.0});
                let mut hit=at(hit_uv,z);
                let slope=if ortho {0.02} else {0.001};
                if scenario==5 {
                    for y in 0..h {for x in 0..w {pixels[(y*w+x) as usize]=d+slope*((x as f32+0.5)/w as f32-0.5)+slope*0.7*((y as f32+0.5)/h as f32-0.5);}}
                    let mut xy=hit_uv*2.0-Vec2::ONE;if flip {xy.y=-xy.y;}
                    let depth=d+slope*(hit_uv.x-0.5)+slope*0.7*(hit_uv.y-0.5);
                    let p=inverse*Vec4::new(xy.x,xy.y,depth,1.0);hit=p.truncate()/p.w;
                }let dir=(hit-start).normalize();let distance=hit.distance(start);
                let mut queries=vec![Query {projection:projection.to_cols_array_2d(),inverse_projection:inverse.to_cols_array_2d(),
                    origin:start.extend(distance+1.0).to_array(),direction:dir.extend(if flip {385.0} else {384.0}).to_array()}];
                // The same ray must miss when its requested range ends early.
                let mut short=queries[0];short.origin[3]=distance*0.5;queries.push(short);
                // Start behind the plane and head toward the near plane; the
                // trace kernel handles it (consumer performs normal rejection).
                let from=at(Vec2::new(0.1,0.1),9.0);let toward=(hit-from).normalize();
                queries.push(Query {origin:from.extend(hit.distance(from)+20.0).to_array(),direction:toward.extend(if flip {385.0} else {384.0}).to_array(),..queries[0]});
                // Exhausted traces must report a miss for the caller's world
                // fallback; increasing a budget may recover a grazing hit.
                let mut limited=queries[0];limited.direction[3]=if flip {3.0} else {2.0};queries.push(limited);
                let mut production=queries[0];production.direction[3]=if flip {193.0} else {192.0};queries.push(production);
                let level_start=at(Vec2::new(0.1,0.15),-hit.z);let level_dir=(hit-level_start).normalize();
                queries.push(Query {origin:level_start.extend(hit.distance(level_start)+1.0).to_array(),direction:level_dir.extend(if flip {385.0} else {384.0}).to_array(),..queries[0]});
                let mut input=Buffer::new(device,&mut alloc,(queries.len()*160) as u64,vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::CpuToGpu,"screen queries").unwrap();
                input.write(0,bytemuck::cast_slice(&queries));
                let mut output=Buffer::new(device,&mut alloc,(queries.len()*16) as u64,vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::GpuToCpu,"screen hits").unwrap();
                for (binding,buffer) in [(0,hierarchy.buffer.handle),(1,input.handle),(2,output.handle)] {
                    let info=[vk::DescriptorBufferInfo::default().buffer(buffer).range(vk::WHOLE_SIZE)];
                    device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info)],&[]);
                }
                let mut texture=GpuTexture::upload_pixels(device,&mut alloc,g.pool,g.queue,w,h,vk::Format::R32_SFLOAT,bytemuck::cast_slice(&pixels)).unwrap();
                let info=[vk::DescriptorImageInfo::default().sampler(sampler).image_view(texture.view).image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&info)],&[]);
                one_time_submit(device,g.pool,g.queue,|cb| {
                    reduce.record(device,cb,&hierarchy,texture.view,inverse.to_cols_array_2d(),0);
                    device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,pipeline);
                    device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,layout,0,&[set],&[]);
                    device.cmd_dispatch(cb,queries.len() as u32,1,1);
                    let barrier=[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::TRANSFER_READ|vk::AccessFlags::HOST_READ)];
                    device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::COMPUTE_SHADER,vk::PipelineStageFlags::TRANSFER|vk::PipelineStageFlags::HOST,vk::DependencyFlags::empty(),&barrier,&[],&[]);
                    device.cmd_copy_buffer(cb,hierarchy.buffer.handle,readback.handle,&[vk::BufferCopy::default().size(hierarchy.byte_len())]);
                    let host=[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ)];
                    device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::TRANSFER,vk::PipelineStageFlags::HOST,vk::DependencyFlags::empty(),&host,&[],&[]);
                }).unwrap();
                let bytes=readback.read();let header:&[u32]=bytemuck::cast_slice(&bytes[..512]);let data:&[f32]=bytemuck::cast_slice(&bytes[512..]);
                for level in 1..header[3] as usize {
                    let meta=&header[level*4..];let side=1u32<<level;
                    for y in 0..meta[1] {for x in 0..meta[0] {
                        let mut expected=[1.0f32,0.0f32];
                        for j in y*side..((y+1)*side).min(h) {for i in x*side..((x+1)*side).min(w) {
                            let value=pixels[(j*w+i) as usize];expected[0]=expected[0].min(value);expected[1]=expected[1].max(value);
                        }}
                        let k=((meta[2]+y*meta[0]+x)*2) as usize;
                        if scenario==5 && w>1 {
                            let padding=0.5*(slope/w as f32+slope*0.7/h as f32);
                            expected[0]-=padding;expected[1]+=padding;
                            assert!((data[k]-expected[0]).abs()<2e-7 && (data[k+1]-expected[1]).abs()<2e-7,"sloped depth bounds level={level}");
                        } else {assert_eq!(&data[k..k+2],&expected,"odd depth reduction level={level} texel={x},{y}");}
                    }}
                }
                let bytes=output.read();let hits:&[[f32;4]]=bytemuck::cast_slice(&bytes);
                let expect_hit=w>1 && scenario!=2 && scenario!=4 && scenario!=6;
                eprintln!("[Flux screen trace] {w}x{h} ortho={ortho} flip={flip} scenario={scenario}: {:?}",hits);
                assert_eq!(hits[0][3]>0.0,expect_hit,"screen trace missed or bridged a silhouette");
                assert!(hits[1][3]<0.0,"ignored finite ray range");
                assert!(hits[3][3]<0.0 && hits[3][3]>=-2.0,"ignored traversal budget");
                if hits[4][3]<0.0 && expect_hit {assert_eq!(hits[4][3],-97.0,"96-step fallback must be budget exhaustion");}
                if scenario==5 && w>1 {assert!(hits[5][3]>0.0,"constant-depth ray missed sloped receiver");}
                if expect_hit {
                    assert!((hits[0][2]-distance).abs()<distance*0.001+0.002);
                    assert!((hits[0][0]-hit_uv.x).abs()*(w as f32)<0.55 && (hits[0][1]-hit_uv.y).abs()*(h as f32)<0.55);
                    assert!(hits[2][3]>0.0,"near-plane clipping lost an intersection");
                    if scenario==1 {assert!(hits[0][3]<80.0,"empty space was not skipped hierarchically");}
                }
                input.destroy(device,&mut alloc);output.destroy(device,&mut alloc);texture.destroy(device,&mut alloc);
            }}}
            readback.destroy(device,&mut alloc);hierarchy.destroy(device,&mut alloc);
        }
        device.destroy_sampler(sampler,None);device.destroy_descriptor_pool(pool,None);device.destroy_pipeline(pipeline,None);
        device.destroy_pipeline_layout(layout,None);device.destroy_descriptor_set_layout(set_layout,None);
    }
    reduce.destroy(device);
}
