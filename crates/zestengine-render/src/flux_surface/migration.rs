//! GPU lighting transfer for stable card resolution changes; no readback.
use super::*;
#[repr(C)]
#[derive(Clone,Copy,Pod,Zeroable)]
struct Push { source_to_world:[[f32;4];4],source:[u32;4],target:[u32;4],params:[f32;4],uv:[f32;4] }
const _: () = assert!(size_of::<Push>()==128);
pub struct Transfer { pub source:Card, pub target:Card, pub valid:[bool;2],pub uv:[f32;4] }
pub struct Migrator { layout:vk::PipelineLayout,set_layout:vk::DescriptorSetLayout,pipeline:vk::Pipeline,
    pool:vk::DescriptorPool,sets:Vec<vk::DescriptorSet>,sampler:vk::Sampler }
impl Migrator {
    pub fn new(device:&ash::Device)->Result<Self> {unsafe {
        let bindings:Vec<_>=(0..5).map(|b|vk::DescriptorSetLayoutBinding::default().binding(b)
            .descriptor_type(if b<3 {vk::DescriptorType::COMBINED_IMAGE_SAMPLER} else {vk::DescriptorType::STORAGE_IMAGE})
            .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect();
        let set_layout=device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),None)?;
        let layout=device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout])
            .push_constant_ranges(&[vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(size_of::<Push>() as u32)]),None)?;
        let shader=crate::screen_pass::shader_module(device,include_bytes!(concat!(env!("OUT_DIR"),"/flux_surface_migrate.comp.spv")))?;
        let pipeline=device.create_cached_compute_pipelines(vk::PipelineCache::null(),&[vk::ComputePipelineCreateInfo::default().layout(layout)
            .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(shader).name(c"main"))],None);
        device.destroy_shader_module(shader,None);let pipeline=pipeline.map_err(|(_,e)|e)?[0];
        let sizes=[vk::DescriptorPoolSize {ty:vk::DescriptorType::COMBINED_IMAGE_SAMPLER,descriptor_count:3*FRAMES_IN_FLIGHT as u32},
            vk::DescriptorPoolSize {ty:vk::DescriptorType::STORAGE_IMAGE,descriptor_count:2*FRAMES_IN_FLIGHT as u32}];
        let pool=device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(FRAMES_IN_FLIGHT as u32).pool_sizes(&sizes),None)?;
        let sets=device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[set_layout; FRAMES_IN_FLIGHT]))?;
        let sampler=device.create_sampler(&vk::SamplerCreateInfo::default().mag_filter(vk::Filter::NEAREST).min_filter(vk::Filter::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE).address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE),None)?;
        Ok(Self {layout,set_layout,pipeline,pool,sets,sampler})
    }}
    pub fn record(&self,device:&ash::Device,cb:vk::CommandBuffer,frame:usize,atlas:&Atlas,transfers:&[Transfer]) {
        if transfers.is_empty() || !atlas.light_initialized {return;}
        let lights=&atlas.lighting[atlas.light_index];let set=self.sets[frame];
        for light in lights {light_barrier(device,cb,light.image,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,vk::ImageLayout::GENERAL);}
        for b in 0..5 {unsafe {
            let info=[vk::DescriptorImageInfo::default().image_view(if b<3 {atlas.images[b].level(0)} else {lights[b-3].view})
                .sampler(if b<3 {self.sampler} else {vk::Sampler::null()})
                .image_layout(if b<3 {vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL} else {vk::ImageLayout::GENERAL})];
            device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(b as u32)
                .descriptor_type(if b<3 {vk::DescriptorType::COMBINED_IMAGE_SAMPLER} else {vk::DescriptorType::STORAGE_IMAGE}).image_info(&info)],&[]);
        }}
        unsafe {
            device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,self.pipeline);
            device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,self.layout,0,&[set],&[]);
            for transfer in transfers {
                let s=transfer.source;let t=transfer.target;
                let push=Push {source_to_world:s.capture_to_world,source:s.rect,target:t.rect,
                    params:[s.normal_texel[3],t.normal_texel[3],(u32::from(transfer.valid[0]) | (u32::from(transfer.valid[1])*2)) as f32,0.0],uv:transfer.uv};
                device.cmd_push_constants(cb,self.layout,vk::ShaderStageFlags::COMPUTE,0,bytemuck::bytes_of(&push));
                device.cmd_dispatch(cb,(t.rect[2]/atlas.lighting_div).div_ceil(8),(t.rect[2]/atlas.lighting_div).div_ceil(8),1);
                // Validation tracks storage access at image granularity. Pages
                // are disjoint, but make the dispatch ordering explicit too.
                let memory=vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE|vk::AccessFlags::SHADER_READ)
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE|vk::AccessFlags::SHADER_READ);
                crate::sync_check::barrier1(&device, cb,vk::PipelineStageFlags::COMPUTE_SHADER,vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),&[memory],&[],&[]);
            }
        }
        for light in lights {light_barrier(device,cb,light.image,vk::ImageLayout::GENERAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);}
    }
    pub fn destroy(&self,device:&ash::Device) {unsafe {
        device.destroy_sampler(self.sampler,None);device.destroy_descriptor_pool(self.pool,None);
        device.destroy_pipeline(self.pipeline,None);device.destroy_pipeline_layout(self.layout,None);device.destroy_descriptor_set_layout(self.set_layout,None);
    }}
}

#[cfg(test)] mod tests {
    use super::*;
    #[test]
    fn surface_migration_rejects_invalid_history_and_preserves_energy() {
        for spacing in [4usize,2] {check_migration(spacing);}
    }
    fn check_migration(spacing:usize) {
        let edge=64/spacing;let center=(spacing/2) as f32+0.5;
        let Some(g)=crate::gpu_tests::gpu() else {return;};
        let validation_before=crate::cluster_headless::VALIDATION_ERRORS.load(std::sync::atomic::Ordering::Relaxed);
        let device=&g.device;let mut alloc=g.allocator().unwrap();
        let mut atlas=Atlas::with_lighting_div(device,&mut alloc,2,spacing as u32).unwrap();
        let migrator=Migrator::new(device).unwrap();
        let mut output=Buffer::new(device,&mut alloc,(2*edge*edge*8) as u64,vk::BufferUsageFlags::TRANSFER_DST,MemoryLocation::GpuToCpu,"Migrated lighting test").unwrap();
        const W:usize=128;
        // All four256-chart tiles and all sixteen512-chart tiles use a
        // spatial gradient, so an incorrect equal-UV copy cannot pass.
        let tiles:Vec<_>=(1..=2).flat_map(|level|(0..1<<level).flat_map(move |y|
            (0..1<<level).map(move |x|virtual_pages::Tile {level,x,y}))).collect();
        for case in 0..31 {
            let uv=if (9..29).contains(&case) {tiles[case-9].parent_uv()} else {[1.0,1.0,0.0,0.0]};
            let transform=if case==7 {Mat4::from_translation(Vec3::new(1.0,0.0,0.0))*Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2)} else {Mat4::IDENTITY};
            let mut position=vec![[0.0f32;4];W*W];let mut normal=position.clone();let mut albedo=vec![[128u8,128,128,0];W*W];
            for (x0,size,target) in [(0usize,32usize,false),(64,64,true)] {
                for y in 0..size {for x in 0..size {
                    let index=y*W+x0+x;
                    let mut p=Vec3::new(x as f32/size as f32,y as f32/size as f32,if case==2 && !target {1.0} else {0.0});
                    if target {p.x=p.x*uv[0]+uv[2];p.y=p.y*uv[1]+uv[3];}
                    let n=if case==3 && !target {Vec3::Y} else {Vec3::Z};
                    let p=if target {transform.transform_point3(p)} else {p};let n=if target {transform.transform_vector3(n)} else {n};
                    position[index]=p.extend(if (case==1 && !target)||(case==8 && target) {0.0} else {1.0}).to_array();
                    normal[index]=n.extend(1.0).to_array();
                    if case==5 && !target {albedo[index]=[0;4];}
                    if case==6 && target {albedo[index]=[64,64,64,0];}
                }}
            }
            let half=|values:&[[f32;4]]|->Vec<u8> {values.iter().flatten().flat_map(|v|half::f16::from_f32(*v).to_bits().to_ne_bytes()).collect()};
            let light=if case==4 {[0.0,0.0,0.0,1.0]} else {[0.2,0.4,0.6,32.0]};
            let mut light_values=vec![light;(W/spacing)*(W/spacing)];
            if case>=9 {
                light_values.fill([10.0,0.0,10.0,1.0]);
                for y in 0..32/spacing {for x in 0..32/spacing {
                    light_values[y*(W/spacing)+x]=if case==29 {[f32::NAN,0.0,0.0,1.0]}
                        else if case==30 {[f32::INFINITY,0.0,0.0,1.0]}
                        else {[((x*spacing) as f32+center)/32.0,((y*spacing) as f32+center)/32.0,0.25,32.0]};
                }}
            }
            let light_data=half(&light_values);
            let material_data=[bytemuck::cast_slice(&position).to_vec(),half(&normal),bytemuck::cast_slice(&albedo).to_vec(),half(&vec![[0.0;4];W*W])];
            let mut uploads=Vec::new();
            for (im,bytes) in atlas.images.iter().zip(material_data.iter()).chain(atlas.lighting.iter().flatten().map(|im|(im,&light_data))) {
                let mut staging=Buffer::new(device,&mut alloc,bytes.len() as u64,vk::BufferUsageFlags::TRANSFER_SRC,MemoryLocation::CpuToGpu,"Migration test input").unwrap();
                staging.write(0,bytes);uploads.push((im.image,im.extent,im.mips,staging));
            }
            let mut source=Card::zeroed();source.rect=[0,0,32,1];source.normal_texel=[0.0,0.0,1.0,1.0/32.0];source.capture_to_world=transform.to_cols_array_2d();
            let mut target=source;target.rect=[64,0,64,1];target.normal_texel[3]=1.0/64.0;
            crate::alloc::one_time_submit(device,g.pool,g.queue,|cb|unsafe {
                for (image,extent,mips,staging) in &uploads {
                    let range=range(vk::ImageAspectFlags::COLOR).level_count(*mips);
                    let before=vk::ImageMemoryBarrier::default().image(*image).subresource_range(range)
                        .old_layout(if case==0 {vk::ImageLayout::UNDEFINED} else {vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL}).new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_access_mask(if case==0 {vk::AccessFlags::empty()} else {vk::AccessFlags::MEMORY_READ|vk::AccessFlags::MEMORY_WRITE}).dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                    crate::sync_check::barrier1(&device, cb,vk::PipelineStageFlags::ALL_COMMANDS,vk::PipelineStageFlags::TRANSFER,vk::DependencyFlags::empty(),&[],&[],&[before]);
                    let copy=vk::BufferImageCopy::default().image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                        .image_extent(vk::Extent3D {width:extent.width,height:extent.height,depth:1});
                    device.cmd_copy_buffer_to_image(cb,staging.handle,*image,vk::ImageLayout::TRANSFER_DST_OPTIMAL,&[copy]);
                    let after=vk::ImageMemoryBarrier::default().image(*image).subresource_range(range).old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
                    crate::sync_check::barrier1(&device, cb,vk::PipelineStageFlags::TRANSFER,vk::PipelineStageFlags::COMPUTE_SHADER,vk::DependencyFlags::empty(),&[],&[],&[after]);
                }
                atlas.light_initialized=true;
                migrator.record(device,cb,0,&atlas,&[Transfer {source,target,valid:[true;2],uv}]);
                for (i,im) in atlas.lighting[0].iter().enumerate() {
                    light_barrier(device,cb,im.image,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
                    let copy=vk::BufferImageCopy::default().buffer_offset((i*edge*edge*8) as u64)
                        .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                        .image_offset(vk::Offset3D {x:edge as i32,y:0,z:0}).image_extent(vk::Extent3D {width:edge as u32,height:edge as u32,depth:1});
                    device.cmd_copy_image_to_buffer(cb,im.image,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,output.handle,&[copy]);
                    light_barrier(device,cb,im.image,vk::ImageLayout::TRANSFER_SRC_OPTIMAL,vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
                }
            }).unwrap();
            let bytes=output.read();let actual:Vec<f32>=bytes[..2*edge*edge*8].as_chunks::<2>().0.iter().map(|v|half::f16::from_bits(u16::from_ne_bytes(*v)).to_f32()).collect();
            for (i,actual) in actual.as_chunks::<4>().0.iter().enumerate() {
                let x=i%edge;let y=(i/edge)%edge;
                let expected=match case {1|2|3|5|8|29|30=>[0.0;4],4=>[0.0,0.0,0.0,1.0],6=>[0.1,0.2,0.3,4.0],
                    9..=28=>[(((x*spacing) as f32+center)/64.0*uv[0]+uv[2]).clamp(center/32.0,(32.0-spacing as f32+center)/32.0),
                        (((y*spacing) as f32+center)/64.0*uv[1]+uv[3]).clamp(center/32.0,(32.0-spacing as f32+center)/32.0),0.25,4.0],
                    _=>[0.2,0.4,0.6,4.0]};
                for c in 0..4 {assert!((actual[c]-expected[c]).abs()<0.001,
                    "migration spacing={spacing} case {case}, texel({x},{y}), component{}: {actual:?} expected {expected:?}",i/(edge*edge));}
            }
            for (_,_,_,mut staging) in uploads {staging.destroy(device,&mut alloc);}
        }
        eprintln!("[Flux migration]spacing={spacing} 31 cases, both lighting components and every destination texel: cropped256/512 charts, geometry/material rejection, valid black, nonfinite history");
        output.destroy(device,&mut alloc);migrator.destroy(device);atlas.destroy(device,&mut alloc);
        assert_eq!(crate::cluster_headless::VALIDATION_ERRORS.load(std::sync::atomic::Ordering::Relaxed),validation_before);
    }
}
