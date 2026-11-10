//! Surface lighting independent of screen probes. Reuses the diffuse GI surface
//! kernels and budgets, with persistent fence-owned uploads and no gather targets.
use super::*;
use crate::flux_surface::{LightingUpdate, Resources};
use crate::software_reflect::Scene;

struct Slot {
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    params: Buffer,
    lights: Buffer,
    emitters: Buffer,
    light_capacity: u64,
    scene: Option<Scene>,
}
pub(crate) struct Recorder {
    slots: Vec<Option<Slot>>,
    #[cfg(test)] pub recorded: u64,
}
impl Default for Recorder {
    fn default() -> Self { Self { slots: (0..crate::FRAMES_IN_FLIGHT).map(|_|None).collect(), #[cfg(test)] recorded:0 } }
}
impl Recorder {
    #[allow(clippy::too_many_arguments)]
    pub fn record(&mut self, gi: &GpuGi, device: &ash::Device, alloc: &mut Allocator, cb: vk::CommandBuffer,
        frame: usize, scene: Option<Scene>, rt: Option<(vk::AccelerationStructureKHR,vk::Buffer,vk::DescriptorSet)>,
        lights: &[crate::GpuLight], area: vk::ImageView, surface: Resources, update: LightingUpdate, mut params: ScreenGiParams) -> Result<()> {
        let light_bytes=(lights.len().max(1)*size_of::<crate::GpuLight>()) as u64;
        if self.slots[frame].as_ref().is_none_or(|s|s.light_capacity<light_bytes) {
            let new=Slot::new(gi,device,alloc,light_bytes)?;
            // The renderer retired this slot's fence before recording.
            if let Some(mut old)=self.slots[frame].replace(new) { old.destroy(device,alloc); }
        }
        let slot=self.slots[frame].as_mut().unwrap();
        let hardware=rt.is_some();
        let (set,layout,pipeline,base)=if hardware {
            (slot.sets[1],gi.rt_pipeline_layout.unwrap(),gi.surface_rt_pipeline.unwrap(),11)
        } else { (slot.sets[0],gi.screen_pipeline_layout,gi.surface_pipeline,12) };
        params.probe_info[2]=surface.node_count as f32;
        params.counts[2]=lights.len() as u32;
        // Standalone reflections sample actual captured emissive surfaces. They
        // must not consume stale analytic emitters left by a disabled GI feed.
        params.counts[3]=0;
        if let Some(s)=scene.as_ref().filter(|_|!hardware) {
            params.gdf_min[..3].copy_from_slice(&s.min);
            params.gdf_max[..3].copy_from_slice(&s.max);
            params.gdf_max[3]=s.mesh.as_ref().map_or(0,|m|m.node_count) as f32;
        }
        if hardware { slot.params.write(0,bytemuck::bytes_of(&RtGiParams::from_screen(&params))); }
        else { slot.params.write(0,bytemuck::bytes_of(&params)); }
        if lights.is_empty() { slot.lights.write(0,&[0;64]); }
        else { slot.lights.write(0,bytemuck::cast_slice(lights)); }
        let buffer=|binding,handle,kind|unsafe {
            let info=[vk::DescriptorBufferInfo::default().buffer(handle).range(vk::WHOLE_SIZE)];
            device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding)
                .descriptor_type(kind).buffer_info(&info)],&[]);
        };
        let image=|binding,handle,sampler,storage|unsafe {
            let info=[vk::DescriptorImageInfo::default().image_view(handle).sampler(sampler)
                .image_layout(if storage {vk::ImageLayout::GENERAL} else {vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL})];
            device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding)
                .descriptor_type(if storage {vk::DescriptorType::STORAGE_IMAGE} else {vk::DescriptorType::COMBINED_IMAGE_SAMPLER}).image_info(&info)],&[]);
        };
        buffer(if hardware {9} else {7},slot.params.handle,vk::DescriptorType::UNIFORM_BUFFER);
        buffer(if hardware {5} else {3},slot.lights.handle,vk::DescriptorType::STORAGE_BUFFER);
        buffer(if hardware {6} else {4},slot.emitters.handle,vk::DescriptorType::STORAGE_BUFFER);
        if let Some((tlas,instances,_))=rt {
            buffer(4,instances,vk::DescriptorType::STORAGE_BUFFER);
            let acceleration=[tlas];
            let mut info=vk::WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(&acceleration);
            let mut write=vk::WriteDescriptorSet::default().dst_set(set).dst_binding(3)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR).push_next(&mut info);
            write.descriptor_count=1;
            unsafe {device.update_descriptor_sets(&[write],&[]);}
        } else {
            let s=scene.as_ref().expect("software surface lighting requires distance fields");
            s.clipmaps.bind(device,set,36,gi.dist_sampler,gi.index_sampler);
            buffer(2,s.materials.handle,vk::DescriptorType::STORAGE_BUFFER);
            let (mesh,voxels)=s.mesh.as_ref().map_or(([s.materials.handle;2],s.distance),|m|m.resources());
            buffer(22,mesh[0],vk::DescriptorType::STORAGE_BUFFER);buffer(23,mesh[1],vk::DescriptorType::STORAGE_BUFFER);
            image(24,voxels,gi.dist_sampler,false);
        }
        buffer(base,surface.cards,vk::DescriptorType::STORAGE_BUFFER);
        buffer(base+1,surface.nodes,vk::DescriptorType::STORAGE_BUFFER);
        for i in 0..4 {image(base+2+i,surface.views[i as usize],gi.index_sampler,false);}
        for i in 0..2 {
            image(base+6+i,update.previous[i as usize],gi.dist_sampler,false);
            image(base+8+i,update.output[i as usize],vk::Sampler::null(),true);
        }
        image(if hardware {26} else {31},area,gi.dist_sampler,false);
        unsafe {
            device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,pipeline);
            device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,layout,0,&[set],&[]);
            if let Some((_,_,textures))=rt {device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,layout,1,&[textures],&[]);}
            device.cmd_dispatch(cb,update.dispatch[0],update.dispatch[1],update.dispatch[2]);
        }
        crate::flux_surface::finish_lighting(device,cb,update);
        slot.scene=scene;
        #[cfg(test)] {self.recorded+=1;}
        Ok(())
    }
    pub fn destroy(&mut self, device: &ash::Device, alloc: &mut Allocator) {
        for slot in self.slots.iter_mut().flatten() {slot.destroy(device,alloc);}
    }
}
impl Slot {
    fn new(gi:&GpuGi,device:&ash::Device,alloc:&mut Allocator,light_bytes:u64)->Result<Self> {
        let capacity=light_bytes.max(1024).next_power_of_two();
        let params=Buffer::new(device,alloc,size_of::<ScreenGiParams>() as u64,vk::BufferUsageFlags::UNIFORM_BUFFER,MemoryLocation::CpuToGpu,"Standalone surface parameters")?;
        let lights=Buffer::new(device,alloc,capacity,vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::CpuToGpu,"Standalone surface lights")?;
        let mut emitters=Buffer::new(device,alloc,32,vk::BufferUsageFlags::STORAGE_BUFFER,MemoryLocation::CpuToGpu,"Standalone surface emitter placeholder")?;
        emitters.write(0,&[0;32]);
        let mut sizes=vec![vk::DescriptorPoolSize {ty:vk::DescriptorType::STORAGE_BUFFER,descriptor_count:64},
            vk::DescriptorPoolSize {ty:vk::DescriptorType::UNIFORM_BUFFER,descriptor_count:6},
            vk::DescriptorPoolSize {ty:vk::DescriptorType::STORAGE_IMAGE,descriptor_count:16},
            vk::DescriptorPoolSize {ty:vk::DescriptorType::COMBINED_IMAGE_SAMPLER,descriptor_count:64}];
        let mut layouts=vec![gi.screen_set_layout];
        if let Some(layout)=gi.rt_set_layout {layouts.push(layout);sizes.push(vk::DescriptorPoolSize {ty:vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,descriptor_count:1});}
        unsafe {
            let pool=device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(2).pool_sizes(&sizes),None)?;
            let sets=device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))?;
            Ok(Self {pool,sets,params,lights,emitters,light_capacity:capacity,scene:None})
        }
    }
    fn destroy(&mut self,device:&ash::Device,alloc:&mut Allocator) {
        self.scene=None;self.params.destroy(device,alloc);self.lights.destroy(device,alloc);self.emitters.destroy(device,alloc);
        unsafe {device.destroy_descriptor_pool(self.pool,None);}
    }
}
