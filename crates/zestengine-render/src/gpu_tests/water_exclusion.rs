//! GPU evaluation of dry regions and optical basin paths. Analytic unions of
//! boxes provide an independent reference for concave polygon ray lengths.
use ash::vk;
use glam::{Mat4, Vec3, Quat};
use gpu_allocator::MemoryLocation;
use zestengine_core::{Water, WaterExclusionVolume, WaterBody};
use crate::{GpuWater, WaterCoverage, MAX_WATER_BODIES, alloc::{Buffer, one_time_submit}};

#[test]
fn dry_volume_queries_and_union_lengths_match_cpu() {
    let Some(g) = super::gpu() else { return };
    let device = &g.device;
    let mut alloc = g.allocator().unwrap();
    let mut waters = vec![Water {size:[6.0,8.0],depth:2.0,..Default::default()}; 5];
    let cube = WaterExclusionVolume::default();
    waters[1].exclusions = vec![cube.region(Mat4::IDENTITY).unwrap()];
    waters[2].exclusions = vec![cube.region(Mat4::IDENTITY).unwrap(),
        cube.region(Mat4::from_translation(Vec3::new(1.0,0.0,0.0))).unwrap(),
        cube.region(Mat4::from_scale(Vec3::splat(0.25))).unwrap()];
    waters[3].exclusions = vec![cube.region(Mat4::from_scale_rotation_translation(Vec3::new(2.0,0.5,1.2),
        Quat::from_euler(glam::EulerRot::XYZ,0.4,0.7,0.2),Vec3::new(-1.0,0.3,2.0))).unwrap()];
    waters[4].exclusions = (0..40).map(|i| cube.region(Mat4::from_scale_rotation_translation(
        Vec3::new(0.0001,1.0,1.0),Quat::IDENTITY,Vec3::new(-4.0+i as f32*0.2,0.0,0.0))).unwrap()).collect();
    waters.push(Water {size:[0.0,4.0],depth:2.0,..Default::default()});
    waters.push(Water {body:WaterBody::Pool,depth:2.0,shape:vec![[-4.0,-3.0],[4.0,-3.0],[4.0,3.0],[2.0,3.0],
        [2.0,-1.0],[-2.0,-1.0],[-2.0,3.0],[-4.0,3.0]],exclusions:waters[2].exclusions.clone(),..Default::default()});
    waters.push(Water {body:WaterBody::Pool,depth:2.0,shape:vec![[1.0,-4.0],[1.0002,-4.0],[1.0002,4.0],[1.0,4.0]],..Default::default()});
    waters.push(Water {body:WaterBody::Ocean,size:[0.0,0.0],depth:2.0,shape:vec![[-1.0,-1.0],[1.0,-1.0],[1.0,1.0],[-1.0,1.0]],..Default::default()});
    // These rectangles exactly tile each authored footprint. Their prism
    // intervals are solved in f64 and unioned before subtracting dry boxes.
    let rectangles: Vec<Vec<[f32;4]>> = (0..waters.len()).map(|slot|match slot {
        5 => vec![[-1e6,-2.0,1e6,2.0]],
        6 => vec![[-4.0,-3.0,-2.0,3.0],[-2.0,-3.0,2.0,-1.0],[2.0,-3.0,4.0,3.0]],
        7 => vec![[1.0,-4.0,1.0002,4.0]],
        8 => vec![[-1e6,-1e6,-1.0,1e6],[1.0,-1e6,1e6,1e6],[-1.0,-1e6,1.0,-1.0],[-1.0,1.0,1.0,1e6]],
        _ => vec![[-3.0,-4.0,3.0,4.0]],
    }).collect();
    let mut bodies = [GpuWater::default(); MAX_WATER_BODIES];
    let mut coverage = Vec::new();
    let mut queries: Vec<[[f32;4];3]> = Vec::new();
    let mut expected = Vec::new();
    for (slot, water) in waters.iter().enumerate() {
        bodies[slot].bounds=[water.size[0]*0.5,water.size[1]*0.5,0.0,0.0];
        WaterCoverage::new(water).append(water, &mut bodies[slot], &mut coverage);
        for x in -8..=8 { for y in -3..=3 { for z in -4..=4 {
            let p = Vec3::new(x as f32*0.6,y as f32*0.4,z as f32*0.6);
            for d in [Vec3::X, Vec3::Y, Vec3::NEG_Z, Vec3::new(0.8,0.3,-0.5).normalize(), Vec3::new(0.2,-0.5,1.2), Vec3::new(0.0,-2.0,0.0)] {
                let range = [0.0,12.0];
                queries.push([p.extend(slot as f32).to_array(),d.extend(0.0).to_array(),[range[0],range[1],0.0,0.0]]);
                let mut spans: Vec<_> = water.exclusions.iter().filter_map(|r| r.interval(p,d,range)).collect();
                spans.sort_by(|a,b|a[0].total_cmp(&b[0]));
                let (mut end,mut dry)=(0.0f32,0.0f32);
                for [a,b] in spans {dry+=(b-end.max(a)).max(0.0);end=end.max(b);}
                let mut wet_spans=Vec::new();
                for rect in &rectangles[slot] {
                    let lo=[rect[0],-water.depth,rect[1]];let hi=[rect[2],1e6,rect[3]];
                    let (mut begin,mut end)=(range[0] as f64,range[1] as f64);let mut hit=true;
                    for axis in 0..3 {
                        if d[axis].abs()<1e-12 {hit &= p[axis]>=lo[axis] && p[axis]<=hi[axis];}
                        else {let a=(lo[axis] as f64-p[axis] as f64)/d[axis] as f64;let b=(hi[axis] as f64-p[axis] as f64)/d[axis] as f64;
                            begin=begin.max(a.min(b));end=end.min(a.max(b));}
                    }
                    if hit && end>begin {wet_spans.push([begin as f32,end as f32]);}
                }
                wet_spans.sort_by(|a,b|a[0].total_cmp(&b[0]));
                let mut merged:Vec<[f32;2]>=Vec::new();
                for span in wet_spans {if let Some(last)=merged.last_mut().filter(|last|last[1]>=span[0]) {last[1]=last[1].max(span[1]);}else{merged.push(span);}}
                let mut wet=0.0;
                for [a,b] in merged {
                    let mut spans:Vec<_>=water.exclusions.iter().filter_map(|r|r.interval(p,d,[a,b])).collect();
                    spans.sort_by(|a,b|a[0].total_cmp(&b[0]));let(mut end,mut dry)=(a,0.0);
                    for [a,b] in spans {dry+=(b-end.max(a)).max(0.0);end=end.max(b);}
                    wet+=b-a-dry;
                }
                expected.push((water.excluded(p),dry,wet));
            }
        }}}
    }
    use zestengine_core::RiverPoint;
    let point=|position,width,depth|RiverPoint {position,width,depth,..Default::default()};
    let rivers=[
        Water {body:WaterBody::River,river:vec![point([-4.0,2.0,0.0],0.5,0.4),point([4.0,-2.0,0.0],3.0,2.0)],..Default::default()},
        Water {body:WaterBody::River,river:vec![point([-4.0,1.0,-2.0],2.0,0.8),point([3.0,-0.5,-2.0],1.0,2.0),point([3.0,0.0,2.0],1.5,1.0),point([-4.0,2.0,2.0],2.0,0.5)],
            exclusions:vec![cube.region(Mat4::from_translation(Vec3::new(2.0,0.0,0.0))).unwrap()],..Default::default()},
        Water {body:WaterBody::River,river:vec![point([-4.0,1.0,-2.0],2.0,1.0),point([4.0,1.0,2.0],2.0,1.0),point([-4.0,-1.0,2.0],2.0,0.5),point([4.0,-1.0,-2.0],2.0,0.5)],..Default::default()},
        Water {body:WaterBody::River,river:vec![point([-4.0,1.0,0.0],0.01,1.0),point([4.0,1.0,0.0],0.01,1.0)],..Default::default()},
    ];
    let mesh=zestengine_core::WaterMesh::new(&[Vec3::new(-5.0,1.0,-4.0),Vec3::new(-1.0,3.0,-4.0),Vec3::new(-5.0,1.0,4.0),
        Vec3::new(1.0,5.0,-4.0),Vec3::new(5.0,3.0,-4.0),Vec3::new(5.0,3.0,4.0)],&[0,1,2,3,4,5]);
    let mut rivers=rivers.to_vec();rivers.push(Water{body:WaterBody::Custom,custom_mesh:Some(std::sync::Arc::new(mesh)),depth:2.0,..Default::default()});
    for (j,water) in rivers.iter().enumerate() {
        let slot=9+j;
        WaterCoverage::new(water).append(water,&mut bodies[slot],&mut coverage);
        for i in 0..32 {
            let a=i as f32*2.399963;
            let p=Vec3::new(a.cos()*5.0,(i%7) as f32*0.5-1.5,a.sin()*4.0);
            let d=if i%4==0 {Vec3::NEG_Y} else {Vec3::new(-p.x,0.15*(i%3) as f32-0.15,-p.z).normalize()};
            let range=[0.0,12.0];
            queries.push([p.extend(slot as f32).to_array(),d.extend(0.0).to_array(),[0.0,12.0,0.0,0.0]]);
            let mut spans:Vec<_>=water.exclusions.iter().filter_map(|r|r.interval(p,d,range)).collect();
            spans.sort_by(|a,b|a[0].total_cmp(&b[0]));let(mut end,mut dry)=(0.0f32,0.0);
            for [a,b] in spans {dry+=(b-end.max(a)).max(0.0);end=end.max(b);}
            // Dense gameplay queries are an independent integration oracle.
            // Their maximum edge quantization error is bounded by this step.
            let samples=65536;let step=12.0/samples as f32;let mut wet=0usize;
            for sample in 0..samples {
                let q=p+d*((sample as f32+0.5)*step);
                if !water.excluded(q) && water.column(glam::Vec2::new(q.x,q.z)).is_some_and(|c|q.y<=c.surface && q.y>=c.surface-c.depth) {wet+=1;}
            }
            expected.push((water.excluded(p),dry,wet as f32*step));
        }
    }
    let mut media_expected=std::collections::HashMap::new();
    let media=[Water{size:[6.0,6.0],depth:3.0,..Default::default()},Water{body:WaterBody::River,river:vec![point([-4.0,0.5,-2.0],3.0,2.0),point([4.0,-0.5,2.0],2.0,2.0)],
        exclusions:vec![cube.region(Mat4::from_scale(Vec3::new(0.4,3.0,0.8))).unwrap()],..Default::default()}];
    let transforms=[Mat4::from_translation(Vec3::new(-1.0,0.0,0.0)),Mat4::from_scale_rotation_translation(Vec3::new(1.3,0.7,0.8),Quat::from_rotation_y(0.4),Vec3::new(1.0,0.0,0.5))];
    let extinction=[Vec3::new(0.2,0.4,0.1),Vec3::new(0.6,0.1,0.3)];let sigma_s=[Vec3::new(0.1,0.3,0.05),Vec3::new(0.2,0.03,0.15)];
    for(j,w)in media.iter().enumerate(){let body=&mut bodies[14+j];body.to_local=transforms[j].inverse().to_cols_array_2d();
        body.bounds=[w.size[0]*0.5,w.size[1]*0.5,0.0,0.0];body.ownership=[j as f32+1.0,1.0,0.0,0.0];
        body.set_optics((extinction[j]-sigma_s[j]).extend(1.333).to_array(),sigma_s[j].extend(0.0).to_array());WaterCoverage::new(w).append(w,body,&mut coverage);
    }
    for i in 0..24{let a=i as f32*2.399963;let p=Vec3::new(a.cos()*6.0,(i%5)as f32*0.4-1.5,a.sin()*6.0);let d=Vec3::new(-p.x,0.1*(i%3)as f32-0.1,-p.z).normalize();
        let(mut trans,mut scatter,mut length)=(Vec3::ONE,Vec3::ZERO,0.0);let step=12.0/65536.0;
        for k in 0..65536{let q=p+d*((k as f32+0.5)*step);let owner=(0..2).rev().find(|&j|{let l=transforms[j].inverse().transform_point3(q);
            !media[j].excluded(l)&&media[j].column(glam::Vec2::new(l.x,l.z)).is_some_and(|c|l.y<=c.surface&&l.y>=c.surface-c.depth)});
            if let Some(j)=owner{let t=(-extinction[j]*step).exp();scatter+=trans*(Vec3::ONE-t)*sigma_s[j]/extinction[j];trans*=t;length+=step;}
        }
        for(mode,v)in[(1.0,trans),(2.0,scatter)]{media_expected.insert(queries.len(),v.extend(length).to_array());
            queries.push([p.extend(14.0).to_array(),d.extend(0.0).to_array(),[0.0,12.0,mode,0.0]]);expected.push((false,0.0,0.0));}
    }
    let header = bytemuck::cast_slice(&bodies);
    let mut scene = Buffer::new(device, &mut alloc, (header.len()+coverage.len()*16) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::CpuToGpu,"dry scene").unwrap();
    scene.write(0,header);scene.write(header.len(),bytemuck::cast_slice(&coverage));
    let mut input = Buffer::new(device,&mut alloc,(queries.len()*48) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::CpuToGpu,"dry queries").unwrap();
    input.write(0,bytemuck::cast_slice(&queries));
    let mut output = Buffer::new(device,&mut alloc,(queries.len()*16) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::GpuToCpu,"dry results").unwrap();
    unsafe {
        let bindings: Vec<_> = (0..3).map(|binding| vk::DescriptorSetLayoutBinding::default()
            .binding(binding).descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
        let set_layout = device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None).unwrap();
        let layout = device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]), None).unwrap();
        let module = crate::screen_pass::shader_module(device, include_bytes!(concat!(env!("OUT_DIR"), "/water_exclusion_test.comp.spv"))).unwrap();
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
        let bytes=output.read();
        let values: &[[f32;4]]=bytemuck::cast_slice(&bytes[..queries.len()*16]);
        for (i,(got,(inside,dry,wet))) in values.iter().zip(expected).enumerate() {
            if let Some(want)=media_expected.get(&i){for(a,b)in got.iter().zip(want){assert!((a-b).abs()<0.003,"multi-body optics {:?}: {got:?} vs {want:?}",queries[i]);}continue;}
            assert_eq!(got[0]>0.5,inside,"containment {:?}",queries[i]);
            assert!((got[1]-dry).abs()<0.00005,"union {:?}: {} vs {dry}",queries[i],got[1]);
            let tolerance=if queries[i][0][3]>=9.0 {0.0015} else {0.00008};
            assert!((got[2]-wet).abs()<tolerance,"wet length {:?}: {} vs {wet}",queries[i],got[2]);
        }
        eprintln!("[water exclusions] {} GPU point/ray queries matched CPU, including sloping/self-crossing/thin rivers, concave basins and 40 thin dry volumes",queries.len());
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
