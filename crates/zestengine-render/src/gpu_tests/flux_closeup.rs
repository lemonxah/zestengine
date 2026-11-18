//! Exercise the production temporal pass with sub-millimetre pixel footprints.
//! Near surfaces must retain history just as distant ones do, in either viewport orientation.
use ash::vk;
use bytemuck::Zeroable;
use glam::{Mat4, Vec3};
use gpu_allocator::MemoryLocation;
use crate::{alloc::{Buffer, one_time_submit}, gpu_gi::ScreenGiParams, texture::GpuTexture};
use super::{Img, read_image, transition, NONE};

#[test]
fn flux_closeup_keeps_history_and_rejects_other_surfaces() {
    let Some(g) = super::gpu() else { return };
    let device = &g.device;
    let mut alloc = g.allocator().unwrap();
    const SIDE: u32 = 64;
    let pixels = (SIDE * SIDE) as usize;
    let projection = Mat4::perspective_rh(1.0, 1.0, 0.001, 100.0);
    let inverse = projection.inverse();
    let mut params = Buffer::new(device, &mut alloc, std::mem::size_of::<ScreenGiParams>() as u64,
        vk::BufferUsageFlags::UNIFORM_BUFFER, MemoryLocation::CpuToGpu, "close-up GI params").unwrap();
    let kinds = [vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        vk::DescriptorType::STORAGE_IMAGE, vk::DescriptorType::STORAGE_IMAGE,
        vk::DescriptorType::UNIFORM_BUFFER, vk::DescriptorType::COMBINED_IMAGE_SAMPLER];
    let mut results = Vec::new();
    unsafe {
        let bindings: Vec<_> = kinds.iter().enumerate().map(|(i, kind)| vk::DescriptorSetLayoutBinding::default()
            .binding(i as u32).descriptor_type(*kind).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
        let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).unwrap();
        let layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]), None).unwrap();
        let module = crate::screen_pass::shader_module(device, include_bytes!(concat!(env!("OUT_DIR"), "/flux_history.comp.spv"))).unwrap();
        let pipeline = device.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().layout(layout)
            .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main"))], None).unwrap()[0];
        device.destroy_shader_module(module, None);
        let pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
            vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 5 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_IMAGE, descriptor_count: 2 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 1 }]), None).unwrap();
        let set = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[set_layout])).unwrap()[0];
        let sampler = device.create_sampler(&vk::SamplerCreateInfo::default()
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE).address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE), None).unwrap();
        let info = [vk::DescriptorBufferInfo::default().buffer(params.handle).range(vk::WHOLE_SIZE)];
        device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(6)
            .descriptor_type(kinds[6]).buffer_info(&info)], &[]);
        for distance in [2.0, 0.02] {
            for flip in [false, true] {
                // Valid history, a different normal, and a different surface depth.
                for rejected in 0..3 {
                    let depth = projection.project_point3(Vec3::new(0.0, 0.0, -distance)).z;
                    let mut geometry = Vec::with_capacity(pixels);
                    for y in 0..SIDE { for x in 0..SIDE {
                        let nx = (x as f32 + 0.5) / SIDE as f32 * 2.0 - 1.0;
                        let ny = ((y as f32 + 0.5) / SIDE as f32 * 2.0 - 1.0) * if flip { -1.0 } else { 1.0 };
                        let p = inverse.project_point3(Vec3::new(nx, ny, depth));
                        geometry.push([0.0, 0.0, if rejected == 1 { -1.0 } else { 1.0 },
                            p.length() + if rejected == 2 { 0.1 } else { 0.0 }]);
                    }}
                    let data = [vec![[0.6f32, 0.6, 0.6, 1.0]; pixels], vec![[depth, 0.0, 0.0, 0.0]; pixels],
                        vec![[0.5, 0.5, 0.5, 8.0]; pixels], geometry];
                    let mut inputs: Vec<_> = data.iter().map(|values| GpuTexture::upload_pixels(device, &mut alloc, g.pool, g.queue,
                        SIDE, SIDE, vk::Format::R32G32B32A32_SFLOAT, bytemuck::cast_slice(values)).unwrap()).collect();
                    let mut outputs: Vec<_> = (0..2).map(|_| Img::new(&g, &mut alloc, SIDE, SIDE, 1, vk::Format::R16G16B16A16_SFLOAT,
                        vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC, vk::SampleCountFlags::TYPE_1).unwrap()).collect();
                    let mut values = ScreenGiParams::zeroed();
                    values.inv_view_proj = inverse.to_cols_array_2d();
                    values.prev_view_proj = projection.to_cols_array_2d();
                    values.misc[3] = u32::from(flip);
                    values.temporal = [0.02, 32.0, 0.02, 0.0];
                    params.write(0, bytemuck::bytes_of(&values));
                    for (binding, texture) in inputs.iter().enumerate().chain(std::iter::once((7, &inputs[1]))) {
                        let image = [vk::DescriptorImageInfo::default().sampler(sampler).image_view(texture.view)
                            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
                        device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding as u32)
                            .descriptor_type(kinds[binding]).image_info(&image)], &[]);
                    }
                    for (i, output) in outputs.iter().enumerate() {
                        let image = [vk::DescriptorImageInfo::default().image_view(output.view).image_layout(vk::ImageLayout::GENERAL)];
                        device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(4 + i as u32)
                            .descriptor_type(kinds[4 + i]).image_info(&image)], &[]);
                    }
                    let compute = (vk::PipelineStageFlags2::COMPUTE_SHADER, vk::AccessFlags2::SHADER_WRITE);
                    one_time_submit(device, g.pool, g.queue, |cb| {
                        for output in &outputs { transition(&g, cb, output, vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL, NONE, compute); }
                        device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipeline);
                        device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, layout, 0, &[set], &[]);
                        device.cmd_dispatch(cb, SIDE / 8, SIDE / 8, 1);
                    }).unwrap();
                    let lighting = read_image(&g, &mut alloc, &outputs[0], vk::ImageLayout::GENERAL, compute, 8).unwrap();
                    let normals = read_image(&g, &mut alloc, &outputs[1], vk::ImageLayout::GENERAL, compute, 8).unwrap();
                    // Inspect the centre and corners: viewport orientation and boundary
                    // reconstruction must not silently throw away valid history either.
                    for pixel in [0, (SIDE - 1) as usize, pixels / 2 + SIDE as usize / 2, pixels - 1] {
                        let at = |bytes: &[u8], channel: usize| {
                            let offset = pixel * 8 + channel * 2;
                            half::f16::from_bits(u16::from_ne_bytes(bytes[offset..offset + 2].try_into().unwrap())).to_f32()
                        };
                        results.push((distance, flip, rejected, pixel, at(&lighting, 0), at(&lighting, 3), at(&normals, 2)));
                    }
                    for input in &mut inputs { input.destroy(device, &mut alloc); }
                    for output in &mut outputs { output.destroy(&g, &mut alloc); }
                }
            }
        }
        device.destroy_sampler(sampler, None);
        device.destroy_descriptor_pool(pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
    }
    params.destroy(device, &mut alloc);
    for result @ (_, _, rejected, _, light, frames, nz) in results {
        let (expected_light, expected_frames) = if rejected == 0 { (0.5 + 0.1 / 9.0, 9.0) } else { (0.6, 1.0) };
        assert!((light - expected_light).abs() < 0.001 && frames == expected_frames && nz > 0.99,
            "close-up history (distance, flip, rejected, pixel, light, frames, normal.z): {result:?}");
    }
}
