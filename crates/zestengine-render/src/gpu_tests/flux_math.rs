//! Check production GLSL against analytic lighting integrals and sampling moments.
use ash::vk;
use glam::Vec3;
use gpu_allocator::MemoryLocation;
use crate::alloc::{Buffer, one_time_submit};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Query {
    normal: [f32; 4],
    sample_uv: [f32; 4],
    coefficients: [[f32; 4]; 4],
}

#[test]
fn flux_math_matches_analytic_irradiance_and_cosine_distribution() {
    let Some(g) = super::gpu() else { return };
    let device = &g.device;
    let mut alloc = g.allocator().unwrap();
    // Both poles, floors/walls, tilted surfaces, and either side of the chart seam.
    let normals: Vec<_> = [Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z,
        Vec3::new(1.0, 2.0, 3.0).normalize(), Vec3::new(-2.0, 1.0, -3.0).normalize()]
        .into_iter().chain([-0.50001f32, -0.49999, -1e-7, 1e-7].map(|z|
            Vec3::new((1.0 - z*z).sqrt(), 0.0, z))).collect();
    const SIDE: usize = 32;
    let pi = std::f64::consts::PI;
    let y0 = (1.0 / (4.0*pi)).sqrt() as f32;
    let y1 = (3.0 / (4.0*pi)).sqrt() as f32;
    let ambient = Vec3::new(0.8, 1.2, 2.0);
    // L(w) = ambient + (0.2*w.x, -0.3*w.y, 0.4*w.z).
    let coefficients = [
        (ambient / y0).extend(0.0).to_array(),
        [0.0, 0.3/y1, 0.0, 0.0],
        [0.0, 0.0, 0.4/y1, 0.0],
        [-0.2/y1, 0.0, 0.0, 0.0],
    ];
    let mut queries = Vec::new();
    for &normal in &normals {
        for y in 0..SIDE { for x in 0..SIDE {
            queries.push(Query { normal: normal.extend(0.0).to_array(), coefficients,
                sample_uv: [(x as f32 + 0.5)/SIDE as f32, (y as f32 + 0.5)/SIDE as f32, 2.0, 0.0] });
        }}
    }
    let sample_count = queries.len();
    // Exact sampler endpoints and black / constant environments.
    for &normal in &normals {
        for edge in [0.0, 1.0] {
            queries.push(Query { normal: normal.extend(0.0).to_array(),
                coefficients: if edge == 0.0 { [[0.0; 4]; 4] } else {
                    [coefficients[0], [0.0; 4], [0.0; 4], [0.0; 4]]
                }, sample_uv: [edge, edge, edge*16.0, 0.0] });
        }
    }
    let radiosity_start=queries.len();
    for frame in 0..64 {for ray in 0..16 {
        queries.push(Query {normal:[0.0,0.0,1.0,0.0],sample_uv:[ray as f32,frame as f32,0.0,1.0],coefficients:[[0.0;4];4]});
    }}
    let quality_start=queries.len();
    for resolution in 2..=8 {for frame in 0..64 {for ray in 0..resolution*resolution {
        queries.push(Query {normal:[0.0,0.0,1.0,0.0],sample_uv:[ray as f32,frame as f32,resolution as f32,1.0],coefficients:[[0.0;4];4]});
    }}}
    let mut input = Buffer::new(device, &mut alloc, (queries.len()*96) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "Flux math inputs").unwrap();
    input.write(0, bytemuck::cast_slice(&queries));
    let mut output = Buffer::new(device, &mut alloc, (queries.len()*64) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuToCpu, "Flux math results").unwrap();
    unsafe {
        let bindings: Vec<_> = (0..2).map(|binding| vk::DescriptorSetLayoutBinding::default()
            .binding(binding).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
        let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).unwrap();
        let layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]), None).unwrap();
        let module = crate::screen_pass::shader_module(device, include_bytes!(concat!(env!("OUT_DIR"), "/flux_math_test.comp.spv"))).unwrap();
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
        let values: &[[f32; 16]] = bytemuck::cast_slice(&bytes[..queries.len()*64]);
        let mut means = vec![Vec3::ZERO; normals.len()];
        let mut second_moments = vec![0.0f32; normals.len()];
        for (i, (query, result)) in queries.iter().zip(values).enumerate() {
            assert!(result.iter().all(|v| v.is_finite()), "nonfinite result {i}: {result:?}");
            if i>=radiosity_start {continue;}
            let n = Vec3::from_slice(&query.normal);
            let expected = if i < sample_count {
                ambient * pi as f32 + Vec3::new(0.2*n.x, -0.3*n.y, 0.4*n.z) * (2.0*pi/3.0) as f32
            } else { ambient * (pi as f32 * query.sample_uv[0]) };
            let irradiance = Vec3::from_slice(&result[4..]);
            assert!((irradiance - expected).abs().max_element() < 3e-5, "irradiance {i}: {irradiance:?} != {expected:?}");
            // The L0+L1 addition theorem: sum(Y_i(n)^2) = 1/pi for every n.
            let squared: f32 = result[..4].iter().map(|v| v*v).sum();
            assert!((squared - 1.0/pi as f32).abs() < 1e-6);
            let direction = Vec3::from_slice(&result[8..]);
            assert!((direction.length() - 1.0).abs() < 2e-6);
            let cosine = direction.dot(n);
            assert!((cosine - (1.0 - query.sample_uv[0]).sqrt()).abs() < 2e-6, "hemisphere {i}: {cosine}");
            let peak = query.sample_uv[2].min(8.0);
            assert!((Vec3::from_slice(&result[12..]) - Vec3::new(0.25, 0.5, 1.0)*peak).abs().max_element() < 1e-6);
            if i < sample_count {
                means[i/(SIDE*SIDE)] += direction / (SIDE*SIDE) as f32;
                second_moments[i/(SIDE*SIDE)] += cosine*cosine / (SIDE*SIDE) as f32;
            }
        }
        let radiosity=&values[radiosity_start..];
        for frame in 0..64 {
            let mut x=[0;16];let mut y=[0;16];let mut cells=[0;16];let mut cone=0;
            for value in &radiosity[frame*16..frame*16+16] {
                let u=value[8];let v=value[9];assert!((0.0..1.0).contains(&u) && (0.0..1.0).contains(&v));
                x[(u*16.0) as usize]+=1;y[(v*16.0) as usize]+=1;cells[(u*4.0) as usize+4*(v*4.0) as usize]+=1;
                cone+=usize::from(u<0.2);
            }
            assert_eq!(x,[1;16]);assert_eq!(y,[1;16]);assert_eq!(cells,[1;16]);
            assert!((3..=4).contains(&cone),"small sky cone lost correlated rays: {cone}");
        }
        for ray in 0..16 {
            let mut subcells=std::collections::BTreeSet::new();
            for frame in 0..16 {let value=radiosity[frame*16+ray];subcells.insert(((value[8]*16.0) as u32%4,(value[9]*16.0) as u32%4));}
            assert_eq!(subcells.len(),16,"radiosity sampling left parts of the hemisphere unexplored");
        }
        let mut offset=quality_start;
        for resolution in 2..=8 {
            let count=resolution*resolution;
            for _ in 0..64 {
                let mut x=vec![0;count];let mut y=vec![0;count];let mut cells=vec![0;count];
                for v in &values[offset..offset+count] {
                    let u=v[8];let v=v[9];assert!((0.0..1.0).contains(&u) && (0.0..1.0).contains(&v));
                    x[(u*count as f32) as usize]+=1;y[(v*count as f32) as usize]+=1;
                    cells[(u*resolution as f32) as usize+resolution*(v*resolution as f32) as usize]+=1;
                }
                assert!(x.iter().chain(&y).chain(&cells).all(|v|*v==1),"quality grid left correlated holes: {resolution}");
                offset+=count;
            }
        }
        for (i, &normal) in normals.iter().enumerate() {
            assert!((means[i] - normal*(2.0/3.0)).length() < 0.001, "cosine mean {i}: {:?}", means[i]);
            assert!((second_moments[i] - 0.5).abs() < 1e-5);
        }
        // Tiny normal noise around a wall must not rotate directional bins.
        for i in 0..SIDE*SIDE {
            let a = Vec3::from_slice(&values[10*SIDE*SIDE+i][8..]);
            let b = Vec3::from_slice(&values[11*SIDE*SIDE+i][8..]);
            assert!((a-b).length() < 2e-6);
        }
        eprintln!("[Flux math] {} GPU queries passed: SH integrals, sampler moments/endpoints, chart continuity, RGB clamp", queries.len());
        device.destroy_descriptor_pool(pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_shader_module(module, None);
        device.destroy_pipeline_layout(layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
    }
    input.destroy(device, &mut alloc);
    output.destroy(device, &mut alloc);
}
