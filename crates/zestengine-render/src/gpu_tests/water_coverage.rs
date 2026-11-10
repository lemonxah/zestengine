//! Compare the production shader's authored coverage against gameplay's
//! Water::column, including narrow/smoothed channels and spatially reordered
//! overlapping segments (the original segment must win an exact tie).
use ash::vk;
use glam::Vec2;
use gpu_allocator::MemoryLocation;
use zestengine_core::{Water, WaterBody, RiverPoint};
use crate::{GpuWater, WaterCoverage, MAX_WATER_BODIES, alloc::{Buffer, one_time_submit}};

#[test]
fn authored_coverage_matches_gameplay_on_gpu() {
    let Some(g) = super::gpu() else { return };
    let device = &g.device;
    let mut alloc = g.allocator().unwrap();
    let concave = vec![[2.0,-3.0],[8.0,-3.0],[8.0,4.0],[6.0,4.0],[6.0,-1.0],[4.0,-1.0],[4.0,4.0],[2.0,4.0]];
    let mut shapes = vec![
        Water { size: [4.0, 6.0], depth: 2.75, ..Default::default() },
        Water { size: [0.0, 4.0], depth: 0.3, ..Default::default() },
        Water { body: WaterBody::Ocean, depth: 150.0, ..Default::default() },
        Water { body: WaterBody::Pool, shape: concave.clone(), ..Default::default() },
        Water { shape: concave, smooth_shape: true, ..Default::default() },
        Water { body: WaterBody::River, current: [0.3, 0.0, -0.2], river: vec![
            RiverPoint { position: [-7.0, 8.0, -4.0], width: 0.06, depth: 0.1, speed: 0.2, ..Default::default() },
            RiverPoint { position: [-2.0, 3.0, 3.0], width: 1.4, depth: 2.0, speed: 3.0,
                arrive_tangent: Some([7.0, -1.0, 0.0]), leave_tangent: Some([4.0, -2.0, -1.0]),linear:false },
            RiverPoint { position: [7.0, -2.0, -3.0], width: 0.03, depth: 0.3, speed: 1.0, ..Default::default() },
        ], ..Default::default() },
        Water { body: WaterBody::River, river: vec![], ..Default::default() },
        Water { shape: (0..600).map(|i| { let a = i as f32 * std::f32::consts::TAU / 600.0;
            let r = 5.0 + (a * 13.0).sin(); [a.cos() * r, a.sin() * r] }).collect(), ..Default::default() },
    ];
    // Two coincident channel runs at different heights. Sorting a BVH must
    // not change gameplay's first-segment ownership rule.
    shapes.push(Water { body: WaterBody::River, river: vec![
        RiverPoint { position: [-4.0, 4.0, 0.0], leave_tangent: Some([8.0,0.0,0.0]), ..Default::default() },
        RiverPoint { position: [4.0, 4.0, 0.0], arrive_tangent: Some([8.0,0.0,0.0]), leave_tangent: Some([-8.0,-3.0,0.0]), ..Default::default() },
        RiverPoint { position: [-4.0, 1.0, 0.0], arrive_tangent: Some([-8.0,-3.0,0.0]), ..Default::default() },
    ], ..Default::default() });
    let mut bodies = [GpuWater::default(); MAX_WATER_BODIES];
    let mut coverage = Vec::new();
    let mut queries: Vec<[f32; 4]> = Vec::new();
    let mut expected = Vec::new();
    for (s, water) in shapes.iter().enumerate() {
        let slot = if s == shapes.len() - 1 { MAX_WATER_BODIES - 1 } else { s };
        let body = &mut bodies[slot];
        body.reflect = [0.0, water.current[0], water.current[2], 0.0];
        body.bounds = if water.body == WaterBody::Ocean { [0.0; 4] } else { [water.size[0]*0.5, water.size[1]*0.5, 0.0, 1.0] };
        WaterCoverage::new(water).append(water, body, &mut coverage);
        let mut add = |p: Vec2| {
            queries.push([p.x, p.y, slot as f32, 0.0]);
            expected.push(water.column(p));
        };
        for z in -30..=30 { for x in -40..=40 { add(Vec2::new(x as f32 * 0.261, z as f32 * 0.253)); } }
        for sample in water.river_path().iter() {
            let p = Vec2::new(sample.pos.x, sample.pos.z);
            let side = Vec2::new(-sample.dir.z, sample.dir.x).normalize_or(Vec2::Y);
            for k in [-0.51, -0.49, 0.0, 0.49, 0.51] { add(p + side * sample.width * k); }
        }
    }
    let mut ocean_expected=std::collections::HashMap::new();
    for(slot,t)in[(12,1.25),(13,8.75)] {
        let frame=zestengine_core::OceanCache::default().frame(&zestengine_core::OceanSpectrum{patch_size:32.0,significant_height:2.0,..Default::default()},t);
        bodies[slot].ocean=[coverage.len()as u32,frame.n as u32,frame.size.to_bits(),0];coverage.extend_from_slice(&frame.fields);
        for z in -20..=20{for x in -25..=25{let p=Vec2::new(x as f32*1.271,z as f32*1.653);
            ocean_expected.insert(queries.len(),frame.sample(p));queries.push([p.x,p.y,slot as f32,1.0]);expected.push(None);
        }}
    }
    let recorded=zestengine_core::WaterSimulationCache{n:16,origin:[-2.0,1.0],cell:0.5,frame_rate:20.0,
        frames:(0..10).map(|f|(0..256).map(|i|[((i%16)as f32*0.4).sin()*0.2+f as f32*0.01,0.5,-0.2,0.3]).collect()).collect()};
    let settings=zestengine_core::WaterCacheSettings{looping:false,gain:0.7,..Default::default()};
    bodies[14].recording=[coverage.len()as u32,recorded.n,0,0];bodies[14].recording_window=[recorded.origin[0],recorded.origin[1],recorded.cell,0.0];
    coverage.extend(recorded.gpu_field(0.125,&settings));
    for z in -5..40{for x in -20..40{
        let p=Vec2::new(x as f32*0.271,z as f32*0.293);let sample=recorded.sample(p,0.125,&settings);
        ocean_expected.insert(queries.len(),(glam::Vec3::new(0.0,sample.height,0.0),glam::Vec3::new(-sample.slope.x,1.0,-sample.slope.y).normalize(),sample.foam));
        queries.push([p.x,p.y,14.0,2.0]);expected.push(None);
    }}
    // A transformed receiving surface must contribute its wave motion in
    // river-local space, with the same endpoint weight as gameplay.
    let target=std::sync::Arc::new(Water{size:[100.0;2],..Default::default()});
    let target_world=glam::Mat4::from_scale_rotation_translation(glam::Vec3::splat(1.4),glam::Quat::from_rotation_y(0.7),glam::Vec3::new(2.0,3.0,1.0));
    let mut river=Water{body:WaterBody::River,river:vec![RiverPoint{position:[-10.0,5.0,0.0],..Default::default()},RiverPoint{position:[0.0,3.0,0.0],..Default::default()}],..Default::default()};
    river.resolved_connections[1]=Some(zestengine_core::ResolvedWaterConnection{plane:[0.0,1.0,0.0,-3.0],flow:glam::Vec3::new(0.4,0.0,0.8),depth:4.0,length:5.0});
    river.connected_surfaces[1]=Some(zestengine_core::water_connection::WaterSurfaceConnection{to_target:target_world.inverse(),water:target.clone()});
    for(slot,w)in[(9,&river),(10,target.as_ref())]{
        bodies[slot].to_local=if slot==9{glam::Mat4::IDENTITY}else{target_world.inverse()}.to_cols_array_2d();
        bodies[slot].bounds=[50.0,50.0,0.0,0.0];WaterCoverage::new(w).append(w,&mut bodies[slot],&mut coverage);
        let waves=w.resolved_waves();bodies[slot].look1[2]=waves.len()as f32;
        for(i,w)in waves.iter().enumerate(){bodies[slot].wave_a[i]=[w.dir.x,w.dir.y,w.k,w.omega];bodies[slot].wave_b[i]=[w.qa,w.amplitude,w.qa*w.k,0.0];}
    }
    bodies[9].connections[1]=11;bodies[9].connection_lengths=[0.0,5.0,river.river_path().last().unwrap().dist,0.0];
    for z in -4..=4{for x in 0..=100{let p=Vec2::new(-10.0+x as f32*0.1,z as f32*0.2);let(d,n)=river.displacement(p,0.0);
        ocean_expected.insert(queries.len(),(d,n,0.0));queries.push([p.x,p.y,9.0,3.0]);expected.push(None);
        queries.push([p.x,p.y,9.0,0.0]);expected.push(river.column(p));
    }}
    let mut extra_expected=std::collections::HashMap::new();
    bodies[9].shallow=[0.1,0.2,0.3,1.0];bodies[10].shallow=[0.3,0.7,0.6,2.0];
    bodies[9].look0=[0.1,0.2,0.3,0.4];bodies[10].look0=[0.5,0.6,0.7,0.8];
    bodies[9].ownership=[1.0,1.0,0.0,0.0];bodies[10].ownership=[2.0,1.0,0.0,0.0];
    let overlap=zestengine_core::water_connection::WaterOverlap{to_target:target_world.inverse(),water:target.clone(),wave_padding:0.0};
    for x in -110..=100{let p=Vec2::new(x as f32*0.5,0.0);let weight=if river.covers(p){river.connection_at(p).map_or(0.0,|(_,w)|w)}else{0.0};
        let want=std::array::from_fn(|k|{let(a,b)=if k<4{(bodies[9].shallow[k],bodies[10].shallow[k])}else{(bodies[9].look0[k-4],bodies[10].look0[k-4])};a+(b-a)*weight});
        extra_expected.insert(queries.len(),want);queries.push([p.x,p.y,9.0,4.0]);expected.push(None);
        let mut want=[0.0;8];want[0]=if overlap.contains(glam::Vec3::new(p.x,3.0,p.y)){0.0}else{1.0};extra_expected.insert(queries.len(),want);queries.push([p.x,p.y,9.0,5.0]);expected.push(None);
    }
    let mesh=zestengine_core::WaterMesh::new(&[glam::Vec3::new(-5.0,1.0,-4.0),glam::Vec3::new(-1.0,3.0,-4.0),glam::Vec3::new(-5.0,1.0,4.0),
        glam::Vec3::new(1.0,5.0,-4.0),glam::Vec3::new(5.0,3.0,-4.0),glam::Vec3::new(5.0,3.0,4.0)],&[0,1,2,3,4,5]);
    let custom=Water{body:WaterBody::Custom,custom_mesh:Some(std::sync::Arc::new(mesh)),depth:2.0,current:[0.2,0.0,0.4],..Default::default()};
    bodies[11].reflect=[0.0,0.2,0.4,0.0];WaterCoverage::new(&custom).append(&custom,&mut bodies[11],&mut coverage);
    for z in -30..=30{for x in -30..=30{let p=Vec2::new(x as f32*0.2,z as f32*0.2);queries.push([p.x,p.y,11.0,0.0]);expected.push(custom.column(p));}}
    let shore=Water{body:WaterBody::Ocean,shape:vec![[-2.0,-3.0],[2.0,-3.0],[2.0,3.0],[-2.0,3.0]],depth:20.0,..Default::default()};
    WaterCoverage::new(&shore).append(&shore,&mut bodies[8],&mut coverage);
    for z in -20..=20{for x in -20..=20{let p=Vec2::new(x as f32*0.4,z as f32*0.4);queries.push([p.x,p.y,8.0,0.0]);expected.push(shore.column(p));}}
    // Independent parent triangle interpolation: fine samples must converge
    // to the actual coarse fan, with untouched close-up waves. Include a
    // rotated/scaled zone so the lattice cannot accidentally be world-aligned.
    let mapping=glam::Mat4::from_scale_rotation_translation(glam::Vec3::new(1.4,1.0,0.7),glam::Quat::from_rotation_y(0.6),glam::Vec3::new(0.3,0.0,-0.2));
    bodies[10].zone_to_water=mapping.to_cols_array_2d();bodies[10].zone_lod=[8.0,2.0,0.125,1.0];
    let sample=|p:Vec2,foot:f32|{
        let l=mapping.transform_point3(glam::Vec3::new(p.x,0.0,p.y));let mut out=l;
        for w in target.resolved_waves().iter(){let t=((std::f32::consts::TAU/(w.k*foot)-2.0)/2.0).clamp(0.0,1.0);let band=t*t*(3.0-2.0*t);
            let phase=w.k*w.dir.dot(Vec2::new(l.x,l.z));out+=glam::Vec3::new(w.qa*w.dir.x*phase.cos(),w.amplitude*phase.sin(),w.qa*w.dir.y*phase.cos())*band;
        }out
    };
    for base in [Vec2::ZERO,Vec2::new(8.0,-9.0),Vec2::new(-8.0,9.0)]{for z in 0..=8{for x in 0..=8{
        let p=base+Vec2::new(x as f32,z as f32)*0.03125;let local=mapping.transform_point3(glam::Vec3::new(p.x,0.0,p.y));
        let want=if base==Vec2::ZERO{sample(p,0.125)}else{
            let lo=(p/0.25).floor()*0.25;let center=lo+Vec2::splat(0.125);
            let corners=[lo,lo+Vec2::new(0.25,0.0),lo+Vec2::splat(0.25),lo+Vec2::new(0.0,0.25)];
            (0..4).find_map(|i|{let a=corners[i];let b=corners[(i+1)%4];let ab=a-center;let ac=b-center;let q=p-center;
                let u=q.perp_dot(ac)/ab.perp_dot(ac);let v=ab.perp_dot(q)/ab.perp_dot(ac);
                (u>=-1e-5&&v>=-1e-5&&u+v<=1.00001).then(||sample(center,0.25)*(1.0-u-v)+sample(a,0.25)*u+sample(b,0.25)*v)
            }).unwrap()
        };
        extra_expected.insert(queries.len(),[want.x,want.y,want.z,0.0,0.0,0.0,0.0,0.0]);queries.push([local.x,local.z,10.0,6.125]);expected.push(None);
    }}}
    let mut shore_target=target.as_ref().clone();shore_target.shore_depth=2.0;
    shore_target.shore=Some(std::sync::Arc::new(zestengine_core::ShoreDepth::new(4,6.0,(0..16).map(|i|i as f32*0.15).collect())));
    WaterCoverage::new(&shore_target).append(&shore_target,&mut bodies[10],&mut coverage);
    let mut shore_river=river.clone();shore_river.connected_surfaces[1]=Some(zestengine_core::water_connection::WaterSurfaceConnection{to_target:target_world.inverse(),water:std::sync::Arc::new(shore_target.clone())});
    for(slot,w)in[(9,&shore_river),(10,&shore_target)]{for z in -16..=16{for x in -20..=20{
        let p=Vec2::new(x as f32*0.4,z as f32*0.4);let gain=if slot==9&&!river.covers(p){1.0}else{w.shore_damping(p)};
        extra_expected.insert(queries.len(),[gain,0.0,0.0,0.0,0.0,0.0,0.0,0.0]);queries.push([p.x,p.y,slot as f32,5.5]);expected.push(None);
    }}}
    let header = bytemuck::cast_slice(&bodies);
    let mut scene = Buffer::new(device, &mut alloc, (header.len() + coverage.len() * 16) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "water coverage scene").unwrap();
    scene.write(0, header);
    scene.write(header.len(), bytemuck::cast_slice(&coverage));
    let mut input = Buffer::new(device, &mut alloc, (queries.len() * 16) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu, "water coverage queries").unwrap();
    input.write(0, bytemuck::cast_slice(&queries));
    let mut output = Buffer::new(device, &mut alloc, (queries.len() * 32) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuToCpu, "water coverage results").unwrap();
    unsafe {
        let bindings: Vec<_> = (0..3).map(|binding| vk::DescriptorSetLayoutBinding::default()
            .binding(binding).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
        let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).unwrap();
        let layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]), None).unwrap();
        let module = crate::screen_pass::shader_module(device, include_bytes!(concat!(env!("OUT_DIR"), "/water_coverage_test.comp.spv"))).unwrap();
        let pipeline = device.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().layout(layout)
            .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main"))], None).unwrap()[0];
        let pool = device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1)
            .pool_sizes(&[vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 3 }]), None).unwrap();
        let set = device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[set_layout])).unwrap()[0];
        for (binding, buffer) in [(0, scene.handle), (1, input.handle), (2, output.handle)] {
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
        let values: &[[f32; 8]] = bytemuck::cast_slice(&bytes[..queries.len() * 32]);
        for (i, (got, expected)) in values.iter().zip(&expected).enumerate() {
            if let Some(want)=extra_expected.get(&i){for(a,b)in got.iter().zip(want){assert!((a-b).abs()<0.0001,"connection/ownership mismatch {:?}: {got:?} {want:?}",queries[i]);}continue;}
            if let Some((d,n,foam))=ocean_expected.get(&i){
                let want=[d.x,d.y,d.z,*foam,n.x,n.y,n.z,0.0];
                for(a,b)in got.iter().zip(want){assert!((a-b).abs()<0.00003,"FFT GPU/CPU mismatch {:?}: {got:?} {want:?}",queries[i]);}continue;
            }
            assert_eq!(got[4] > 0.5, expected.is_some(), "coverage mismatch at {:?}: {:?} vs {:?}", queries[i], got, expected);
            if let Some(col) = expected {
                for (a, b) in got[..4].iter().zip([col.surface, col.depth, col.flow.x, col.flow.z]) {
                    assert!((a-b).abs() < 0.00015, "column mismatch at {:?}: {:?} vs {:?}", queries[i], got, col);
                }
                assert!((got[5]-col.edge).abs() < 0.00015, "bank mismatch at {:?}", queries[i]);
            }
        }
        eprintln!("[water coverage] {} GPU queries matched gameplay across {} bodies; {} BVH records", queries.len(), shapes.len(), coverage.len());
        device.destroy_descriptor_pool(pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_shader_module(module, None);
        device.destroy_pipeline_layout(layout, None);
        device.destroy_descriptor_set_layout(set_layout, None);
    }
    scene.destroy(device, &mut alloc);
    input.destroy(device, &mut alloc);
    output.destroy(device, &mut alloc);
}
