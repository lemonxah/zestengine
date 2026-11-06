//! Read back the actual shared GLSL used by the water surface.
use ash::vk;
use glam::Vec3;
use gpu_allocator::MemoryLocation;
use zestengine_core::WaterOptics;
use crate::alloc::{Buffer, one_time_submit};

#[test]
fn medium_matches_cpu_and_conserves_energy_on_gpu() {
    let Some(g) = super::gpu() else { return };
    let device = &g.device;
    let mut alloc = g.allocator().unwrap();
    let mut queries: Vec<[[f32; 4]; 3]> = Vec::new();
    let mut expected = Vec::new();
    for o in [WaterOptics::default(), WaterOptics { absorption: [0.0; 3], scattering: [0.0; 3], ior: 1.0, anisotropy: 0.0 },
        WaterOptics { absorption: [2.0, 0.1, 0.5], scattering: [0.2, 3.0, 1.0], ior: 2.5, anisotropy: -0.95 }] {
        for distance in [0.0, 0.01, 0.3, 1.0, 10.0, 1000.0] {
            for cosine in [-1.0, -0.5, 0.0, 0.5, 0.7, 0.9, 1.0] {
                queries.push([
                    [o.absorption[0], o.absorption[1], o.absorption[2], o.ior],
                    [o.scattering[0], o.scattering[1], o.scattering[2], o.anisotropy],
                    [distance, cosine, 0.0, 0.0],
                ]);
                expected.push((o.transmittance(distance), o.integrated_scattering(distance), o.normal_reflectance(), o.phase(cosine), o.surface_reflectance(cosine, false), o.surface_reflectance(cosine, true)));
            }
        }
    }
    let mut input = Buffer::new(device, &mut alloc, (queries.len() * 48) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "water optical inputs").unwrap();
    input.write(0, bytemuck::cast_slice(&queries));
    let mut output = Buffer::new(device, &mut alloc, (queries.len() * 48) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuToCpu, "water optical readback").unwrap();
    unsafe {
        let bindings: Vec<_> = (0..2).map(|binding| vk::DescriptorSetLayoutBinding::default()
            .binding(binding).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
        let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).unwrap();
        let layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]), None).unwrap();
        let module = crate::screen_pass::shader_module(device, include_bytes!(concat!(env!("OUT_DIR"), "/water_optics_test.comp.spv"))).unwrap();
        let pipeline = device.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().layout(layout)
            .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main"))], None).unwrap()[0];
        let pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1)
            .pool_sizes(&[vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 2 }]), None).unwrap();
        let set = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[set_layout])).unwrap()[0];
        for (binding, buffer) in [(0, input.handle), (1, output.handle)] {
            let info = [vk::DescriptorBufferInfo::default().buffer(buffer).range(vk::WHOLE_SIZE)];
            device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info)], &[]);
        }
        one_time_submit(device, g.pool, g.queue, |cb| {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
            device.cmd_dispatch(cb, queries.len() as u32, 1, 1);
            device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(), &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ)], &[], &[]);
        }).unwrap();
        let bytes = output.read();
        let values: &[[f32; 12]] = bytemuck::cast_slice(&bytes[..queries.len() * 48]);
        for (result, (t, s, f0, phase, entering, exiting)) in values.iter().zip(expected) {
            assert!(result.iter().all(|v| v.is_finite()));
            let gt = Vec3::from_slice(result);
            let gs = Vec3::from_slice(&result[4..]);
            assert!((gt - t).abs().max_element() < 2e-6);
            assert!((gs - s).abs().max_element() < 2e-6);
            assert!((gt + gs).cmple(Vec3::splat(1.000002)).all());
            assert!((result[3] - f0).abs() < 1e-6);
            assert!((result[7] - phase).abs() < phase.abs() * 0.001 + 1e-6);
            assert!((result[8] - entering).abs() < 2e-6);
            assert!((result[9] - exiting).abs() < 2e-6);
        }
        eprintln!("[water optics] {} GPU medium queries matched CPU and conserved energy", queries.len());
        device.destroy_descriptor_pool(pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_shader_module(module, None);
        device.destroy_pipeline_layout(layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
    }
    input.destroy(device, &mut alloc);
    output.destroy(device, &mut alloc);
}
