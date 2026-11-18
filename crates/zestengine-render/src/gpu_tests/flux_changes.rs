//! The production change-region weights (flux_changes.glsl), fed by the real region list.
use ash::vk;
use glam::Vec3;
use gpu_allocator::MemoryLocation;
use crate::{alloc::{Buffer, one_time_submit}, flux_changes::{Packed, Recent}, flux_radiance_cache::Changes};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Query { point: [f32; 4], source: [f32; 4], body: [f32; 4], extra: [f32; 4] }
/// Region weights for any / geometry / direct changes and the screen blend, then the direct and bounce shadow weights.
#[derive(Clone, Copy)]
struct Weights { any: f32, geometry: f32, direct: f32, blend: f32, shadow: f32, bounce: f32 }

fn evaluate(g: &crate::cluster_headless::HeadlessGpu, packed: &Packed, ignored: bool, queries: &[Query]) -> Vec<Weights> {
    let device = &g.device;
    let mut alloc = g.allocator().unwrap();
    let mut input = Buffer::new(device, &mut alloc, std::mem::size_of_val(queries) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "Flux change queries").unwrap();
    input.write(0, bytemuck::cast_slice(queries));
    let mut output = Buffer::new(device, &mut alloc, (queries.len() * 32) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuToCpu, "Flux change weights").unwrap();
    // The tail of every gather parameter block, after the exposure vector that carries the test switch.
    let mut params = Buffer::new(device, &mut alloc, (16 + std::mem::size_of::<Packed>()) as u64,
        vk::BufferUsageFlags::UNIFORM_BUFFER, MemoryLocation::CpuToGpu, "Flux change regions").unwrap();
    params.write(0, bytemuck::cast_slice(&[0.0f32, 0.0, 0.0, if ignored { 2.0 } else { 0.0 }]));
    params.write(16, bytemuck::bytes_of(packed));
    let values = unsafe {
        let kinds = [vk::DescriptorType::STORAGE_BUFFER, vk::DescriptorType::STORAGE_BUFFER, vk::DescriptorType::UNIFORM_BUFFER];
        let bindings: Vec<_> = kinds.iter().enumerate().map(|(binding, kind)| vk::DescriptorSetLayoutBinding::default()
            .binding(binding as u32).descriptor_type(*kind).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
        let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).unwrap();
        let layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]), None).unwrap();
        let module = crate::screen_pass::shader_module(device, include_bytes!(concat!(env!("OUT_DIR"), "/flux_changes_test.comp.spv"))).unwrap();
        let pipeline = device.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().layout(layout)
            .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main"))], None).unwrap()[0];
        let pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&[
            vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 2 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::UNIFORM_BUFFER, descriptor_count: 1 }]), None).unwrap();
        let set = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[set_layout])).unwrap()[0];
        for (binding, buffer) in [input.handle, output.handle, params.handle].into_iter().enumerate() {
            let info = [vk::DescriptorBufferInfo::default().buffer(buffer).range(vk::WHOLE_SIZE)];
            device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding as u32)
                .descriptor_type(kinds[binding]).buffer_info(&info)], &[]);
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
        let values: Vec<Weights> = bytemuck::cast_slice::<u8, [f32; 8]>(&bytes[..queries.len() * 32]).iter()
            .map(|v| Weights { any: v[0], geometry: v[1], direct: v[2], blend: v[3], shadow: v[4], bounce: v[5] }).collect();
        device.destroy_descriptor_pool(pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_shader_module(module, None);
        device.destroy_pipeline_layout(layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
        values
    };
    input.destroy(device, &mut alloc);
    output.destroy(device, &mut alloc);
    params.destroy(device, &mut alloc);
    assert!(values.iter().all(|w| [w.any, w.geometry, w.direct, w.blend, w.shadow, w.bounce].iter().all(|v| v.is_finite())));
    values
}

fn at(point: Vec3, newer_than: f32) -> Query {
    Query { point: point.extend(newer_than).to_array(), source: [0.0, 1.0, 0.0, 1.0], body: [0.0, 1.0e6, 0.0, 1.0], extra: [0.0, 0.06, 0.0, 0.0] }
}
fn close(a: f32, b: f32) -> bool { (a - b).abs() <= 1e-4 * b.abs().max(1.0) }

#[test]
fn flux_change_regions_weigh_histories_by_place_kind_and_age() {
    let Some(g) = super::gpu() else { return };
    // A half-metre cube moved at the origin, a light changed 50 m away, both three frames ago.
    let mut recent = Recent::default();
    let mut changes = Changes::default();
    changes.moved([Vec3::splat(-0.25), Vec3::splat(0.25)], 0.0);
    changes.light([Vec3::X * 50.0 - Vec3::splat(4.0), Vec3::X * 50.0 + Vec3::splat(4.0)]);
    recent.advance(10, &changes);
    recent.advance(13, &Changes::default());
    let packed = recent.packed();
    // Two bounding radii beyond the cube; a quarter of the light's box diagonal beyond its range.
    let (reach, light_reach) = (0.5 * 3.0f32.sqrt(), 2.0 * 3.0f32.sqrt());
    let queries = [
        at(Vec3::ZERO, 32.0),                                  // 0 inside the moved cube
        at(Vec3::X * (0.25 + reach * 0.99), 32.0),             // 1 just within its reach
        at(Vec3::X * (0.25 + 2.0 * reach), 32.0),              // 2 two reaches away
        at(Vec3::X * (0.25 + 10.0 * reach), 32.0),             // 3 ten reaches away
        at(Vec3::ZERO, 2.0),                                   // 4 history refreshed after the change
        at(Vec3::ZERO, 3.0),                                   // 5 history as old as the change
        at(Vec3::X * 50.0, 32.0),                              // 6 inside the light's range
        at(Vec3::X * (54.0 + 2.0 * light_reach), 32.0),        // 7 two reaches outside it
    ];
    let w = evaluate(&g, &packed, false, &queries);
    assert!(w[0].any == 1.0 && w[0].geometry == 1.0 && w[0].direct < 0.01, "moved geometry is not a changed light: {}", w[0].direct);
    assert!(w[1].any == 1.0, "a change holds fully out to its reach: {}", w[1].any);
    assert!(close(w[2].any, 0.25) && close(w[3].any, 0.01), "beyond the reach the weight falls with the solid angle: {} {}", w[2].any, w[3].any);
    assert!(w[4].any == 0.0 && w[5].any == 1.0, "a history refreshed since the change already contains it: {} {}", w[4].any, w[5].any);
    assert!(w[6].any == 1.0 && w[6].direct == 1.0 && w[6].geometry < 0.001, "a changed light is not moved geometry: {}", w[6].geometry);
    assert!(close(w[7].direct, 0.25), "light outside the range falls like any other change: {}", w[7].direct);
    // Screen histories follow moved geometry only, for a few frames, and by its weight.
    assert!(close(w[0].blend, 0.10) && close(w[2].blend, 0.06) && close(w[6].blend, 0.06), "screen blend: {} {} {}", w[0].blend, w[2].blend, w[6].blend);
    recent.advance(19, &Changes::default());
    let later = evaluate(&g, &recent.packed(), false, &queries[..1]);
    assert!(close(later[0].blend, 0.06) && later[0].any == 1.0, "screen histories settle before bounce light does: {}", later[0].blend);
    // A screen that keeps playing and a light that keeps flickering are animation: after their
    // first frames bounce light stops chasing them, direct light never does.
    let mut playing = Recent::default();
    for frame in 0..90 {
        let mut changes = Changes::default();
        changes.surface([Vec3::splat(-0.25), Vec3::splat(0.25)], 4.0);
        changes.light_level([Vec3::X * 50.0 - Vec3::splat(4.0), Vec3::X * 50.0 + Vec3::splat(4.0)]);
        playing.advance(frame, &changes);
        if frame == 8 || frame == 89 {
            let w = evaluate(&g, &playing.packed(), false, &[queries[0], queries[6]]);
            let follows = if frame == 8 { 1.0 } else { 0.0 };
            assert!(w[0].any == follows && w[1].any == follows && w[1].direct == 1.0,
                "frame {frame}: screen {} / light {} bounce weight, light direct weight {}", w[0].any, w[1].any, w[1].direct);
        }
    }
    // The test switch silences every consumer.
    let silent = evaluate(&g, &packed, true, &queries);
    assert!(silent.iter().all(|w| w.any == 0.0 && w.direct == 0.0 && close(w.blend, 0.06)));
    eprintln!("[Flux changes] region weights by place, kind and age pass on the GPU");
}

#[test]
fn flux_moved_bodies_change_light_along_their_shadows() {
    let Some(g) = super::gpu() else { return };
    let none = Packed { info: [0.0; 4], bounds: [[0.0; 4]; crate::flux_changes::REGIONS * 2] };
    // A body of 0.05 m radius one metre below a point light at y = 4; floors two and four metres below the light.
    let (light, body, radius) = (Vec3::Y * 4.0, Vec3::Y * 3.0, 0.05f32);
    let point = |p: Vec3| Query { point: p.extend(0.0).to_array(), source: light.extend(0.0).to_array(), body: body.extend(radius).to_array(), extra: [0.0; 4] };
    let sun = |p: Vec3, tangent: f32| Query { point: p.extend(0.0).to_array(), source: [0.0, 1.0, 0.0, 1.0], body: body.extend(0.25).to_array(), extra: [tangent, 0.0, 0.0, 0.0] };
    let offsets = [0.02f32, 0.1, 0.25, 0.5, 1.0, 2.0];
    let mut queries = vec![
        point(Vec3::ZERO),                 // 0 in the shadow, far behind the body
        point(Vec3::new(0.4, 0.0, 0.0)),   // 1 two shadow radii from its centre (the shadow is 0.2 m wide here)
        point(Vec3::new(0.8, 0.0, 0.0)),   // 2 four shadow radii
        point(Vec3::new(0.3, 3.8, 0.0)),   // 3 nearer the light than the body: nothing is cast there
        point(Vec3::new(0.0, 6.0, 0.0)),   // 4 beyond the light
        sun(Vec3::ZERO, 0.0),              // 5 below the body under a directional light
        sun(Vec3::X * 0.5, 0.0),           // 6 two body radii from the axis: (0.25 / 0.5)^2
        sun(Vec3::X * 1.0, 0.0),           // 7 four radii: (0.25 / 1.0)^2
        sun(Vec3::Y * 5.0, 0.0),           // 8 above the body
        sun(Vec3::X * 0.5, 0.1),           // 9 a wide sun: its penumbra reaches 0.25 + 0.1 * 3 m here
    ];
    let pairs = queries.len();
    // The same directions seen from the light, on the near and the far floor.
    for x in offsets { queries.push(point(Vec3::new(x, 2.0, 0.0))); queries.push(point(Vec3::new(2.0 * x, 0.0, 0.0))); }
    let w = evaluate(&g, &none, false, &queries);
    assert!(w[0].shadow == 1.0 && w[0].bounce == 1.0, "inside the shadow: {} {}", w[0].shadow, w[0].bounce);
    // Seen from the light the offsets are small angles: the weights are those of a flat shadow.
    assert!(w[1].shadow == 0.0 && (w[1].bounce - 0.25).abs() < 0.02, "beside the shadow direct light is unchanged, bounce light in part: {} {}", w[1].shadow, w[1].bounce);
    assert!((w[2].bounce - 0.0625).abs() < 0.01, "the bounce weight falls with the solid angle of the shadow: {}", w[2].bounce);
    assert!(w[3].bounce == 0.0 && w[4].bounce == 0.0, "no shadow falls towards or behind the light: {} {}", w[3].bounce, w[4].bounce);
    assert!(w[5].shadow == 1.0 && w[5].bounce == 1.0 && w[6].shadow == 0.0 && close(w[6].bounce, 0.25), "directional shadow: {} {}", w[6].shadow, w[6].bounce);
    assert!(close(w[7].bounce, 0.0625) && w[8].bounce == 0.0 && w[8].shadow == 0.0, "directional shadow: {} {}", w[7].bounce, w[8].bounce);
    assert!(w[9].shadow == 1.0, "an extended source widens the shadow: {}", w[9].shadow);
    // A shadow twice as far behind the body is twice as wide, and so is everything measured from it.
    for (k, pair) in w[pairs..].as_chunks::<2>().0.iter().enumerate() {
        assert!(close(pair[0].bounce, pair[1].bounce) && close(pair[0].shadow, pair[1].shadow),
            "offset {}: near floor {} / {}, far floor {} / {}", offsets[k], pair[0].shadow, pair[0].bounce, pair[1].shadow, pair[1].bounce);
    }
    let falling: Vec<f32> = w[pairs..].as_chunks::<2>().0.iter().map(|pair| pair[0].bounce).collect();
    assert!(falling.windows(2).all(|pair| pair[1] <= pair[0]) && falling[0] == 1.0 && *falling.last().unwrap() < 0.05, "{falling:?}");
    eprintln!("[Flux changes] shadow weights for direct and bounce light pass on the GPU");
}
