//! Run the production clipmap selector and directional lookup across every
//! face. Different constant radiance in each grid exposes discontinuous mixing.
use ash::vk;
use gpu_allocator::MemoryLocation;
use crate::alloc::{Buffer,one_time_submit};

#[test]
fn radiance_clipmap_blending_preserves_coverage_and_local_occlusion_distance() {
    let Some(g)=super::gpu() else {return};
    let device=&g.device;let mut alloc=g.allocator().unwrap();
    let before=crate::context::validation_error_count();
    let mut queries=Vec::<[f32;4]>::new();let mut expected=Vec::new();
    const SAMPLES:usize=32;
    for level in 0..3 {for axis in 0..3 {for sign in [-1.0,1.0] {for inset in [0.0f32,0.25,0.5,1.0,1.5,2.0,3.0] {
        for sample in 0..SAMPLES {
            let random=(sample as f32+0.5)/SAMPLES as f32;
            let mut query=[0.0;4];query[axis]=sign*(7.5-inset)*[1.0,4.0,16.0][level];query[3]=random;
            let selected=if random<(inset*0.5).min(1.0) {level as i32} else if level==2 {-1} else {level as i32+1};
            queries.push(query);expected.push(selected);
        }
    }}}}
    let bytes_per_probe=crate::flux_radiance_cache::PROBE_BYTES as usize;
    let mut probes=vec![0u8;crate::flux_radiance_cache::PROBES as usize*bytes_per_probe];
    for (index,probe) in probes.chunks_exact_mut(bytes_per_probe).enumerate() {
        let level=index/4096;let slot=index%4096;
        let cell=[slot%16,(slot/16)%16,slot/256].map(|v|if v>=8 {v as i32-16} else {v as i32});
        let header=[cell[0] as u32,cell[1] as u32,cell[2] as u32,level as u32,7,100,0,16];
        probe[..32].copy_from_slice(bytemuck::cast_slice(&header));
        let radiance=[0.25f32,0.5,0.75][level];
        let packed=[radiance,radiance,radiance,1000.0].map(|v|half::f16::from_f32(v).to_bits());
        for ray in probe[32..].chunks_exact_mut(8) {ray.copy_from_slice(bytemuck::cast_slice(&packed));}
    }
    let params:[[u32;4];6]=[
        [(-8i32) as u32,(-8i32) as u32,(-8i32) as u32,0],
        [(-8i32) as u32,(-8i32) as u32,(-8i32) as u32,0],
        [(-8i32) as u32,(-8i32) as u32,(-8i32) as u32,0],
        [1.0f32.to_bits(),4.0f32.to_bits(),16.0f32.to_bits(),3.0f32.sqrt().to_bits()],
        [123,7,32,1|2|8],[0,0,0,40.0f32.to_bits()]];
    let mut data=Buffer::new(device,&mut alloc,probes.len() as u64,vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::CpuToGpu,"clipmap test probes").unwrap();
    data.write(0,&probes);
    let mut work=Buffer::new(device,&mut alloc,(48+512+12288)*4,vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::GpuToCpu,"clipmap test requests").unwrap();
    work.write(0,&vec![0u8;(48+512+12288)*4]);
    let mut uniform=Buffer::new(device,&mut alloc,96,vk::BufferUsageFlags::UNIFORM_BUFFER,MemoryLocation::CpuToGpu,"clipmap test parameters").unwrap();
    uniform.write(0,bytemuck::cast_slice(&params));
    let mut input=Buffer::new(device,&mut alloc,(queries.len()*16) as u64,vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::CpuToGpu,"clipmap boundary queries").unwrap();
    input.write(0,bytemuck::cast_slice(&queries));
    let mut output=Buffer::new(device,&mut alloc,(queries.len()*32) as u64,vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::GpuToCpu,"clipmap boundary results").unwrap();
    unsafe {
        let bindings:Vec<_>=(0..5).map(|binding|vk::DescriptorSetLayoutBinding::default().binding(binding)
            .descriptor_type(if binding==2 {vk::DescriptorType::UNIFORM_BUFFER} else {vk::DescriptorType::STORAGE_BUFFER})
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
        let set_layout=device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),None).unwrap();
        let layout=device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]),None).unwrap();
        let module=crate::screen_pass::shader_module(device,include_bytes!(concat!(env!("OUT_DIR"),"/flux_radiance_blend_test.comp.spv"))).unwrap();
        let pipeline=device.create_compute_pipelines(vk::PipelineCache::null(),&[vk::ComputePipelineCreateInfo::default().layout(layout)
            .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main"))],None).unwrap()[0];
        let pool=device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
            vk::DescriptorPoolSize {ty:vk::DescriptorType::STORAGE_BUFFER,descriptor_count:4},
            vk::DescriptorPoolSize {ty:vk::DescriptorType::UNIFORM_BUFFER,descriptor_count:1}]),None).unwrap();
        let set=device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[set_layout])).unwrap()[0];
        for (binding,buffer) in [data.handle,work.handle,uniform.handle,input.handle,output.handle].into_iter().enumerate() {
            let info=[vk::DescriptorBufferInfo::default().buffer(buffer).range(vk::WHOLE_SIZE)];
            device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding as u32)
                .descriptor_type(if binding==2 {vk::DescriptorType::UNIFORM_BUFFER} else {vk::DescriptorType::STORAGE_BUFFER}).buffer_info(&info)],&[]);
        }
        one_time_submit(device,g.pool,g.queue,|cb| {
            device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,pipeline);
            device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,layout,0,&[set],&[]);
            device.cmd_dispatch(cb,queries.len() as u32,1,1);
            device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::COMPUTE_SHADER,vk::PipelineStageFlags::HOST,vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ)],&[],&[]);
        }).unwrap();
        let bytes=output.read();let results:&[[f32;8]]=bytemuck::cast_slice(&bytes[..queries.len()*32]);
        for (i,(&level,value)) in expected.iter().zip(results).enumerate() {
            assert!(value.iter().all(|v|v.is_finite()));assert_eq!(value[0],level as f32,"query {:?}",queries[i]);
            let distance=if level<0 {1000.0} else {[1.0,4.0,16.0][level as usize]*2.0*3.0f32.sqrt()};
            assert!((value[1]-distance).abs()<1e-4,"selected grid used the wrong occlusion distance");
            assert_eq!(value[2],if level<0 {0.0} else {1.0});
            if queries[i][0]==7.25 && queries[i][1]==0.0 && queries[i][2]==0.0 {
                assert_eq!(value[7],3.0,"one invocation did not request both overlap levels");
            }
            let energy=if level<0 {0.0} else {[0.25,0.5,0.75][level as usize]};
            for v in &value[4..7] {assert!((*v-energy).abs()<1e-6,"sampled another grid's radiance: {value:?}");}
        }
        // Within each analytic face sweep, the expected average is the linear
        // combination of the two deliberately different grids (or full trace).
        for (group,values) in results.chunks_exact(SAMPLES).enumerate() {
            let mean=values.iter().map(|v|v[4]).sum::<f32>()/SAMPLES as f32;
            let expected_mean=expected[group*SAMPLES..(group+1)*SAMPLES].iter().map(|&l|if l<0 {0.0} else {[0.25,0.5,0.75][l as usize]}).sum::<f32>()/SAMPLES as f32;
            assert!((mean-expected_mean).abs()<1e-6);
        }
        let requests=work.read();let requests:&[u32]=bytemuck::cast_slice(&requests);
        // At p=(7.25,0,0), a single invocation asks both the fine and coarse
        // cells. Both scratch slots must be set even if it sampled the fine one.
        for (level,cell) in [(0usize,[6i32,-1,-1]),(1,[1,-1,-1])] {
            let wrapped=cell.map(|v|(v as u32&15) as usize);let index=level*4096+wrapped[0]+16*wrapped[1]+256*wrapped[2];
            assert_eq!(requests[560+index],1,"overlap grid was never requested");
        }
        eprintln!("[Flux clipmap blend] {} GPU queries: six faces, three levels, analytic mixing and matching occlusion distances",queries.len());
        device.destroy_descriptor_pool(pool,None);device.destroy_pipeline(pipeline,None);device.destroy_shader_module(module,None);
        device.destroy_pipeline_layout(layout,None);device.destroy_descriptor_set_layout(set_layout,None);
    }
    for buffer in [&mut data,&mut work,&mut uniform,&mut input,&mut output] {buffer.destroy(device,&mut alloc);}
    assert_eq!(crate::context::validation_error_count(),before,"clipmap test produced Vulkan validation errors");
}
