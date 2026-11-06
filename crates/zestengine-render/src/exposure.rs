//! GPU histogram exposure. Independent view histories; no display-exposed data
//! enters Flux transport, material captures, reflection probes or lightmaps.
use ash::vk;
use anyhow::Result;
use bytemuck::{Pod,Zeroable};
use gpu_allocator::{vulkan::Allocator,MemoryLocation};
use crate::{alloc::Buffer,AutoExposure};

const VIEWS:usize=3; // Scene/game, Camera preview, shared binocular XR exposure.
const SIZE:u64=64*4+16;
#[repr(C)]
#[derive(Clone,Copy,Pod,Zeroable)]
struct Params { histogram:[f32;4],range_percent:[f32;4],adaptation:[f32;4] }
struct View { buffer:Buffer,set:vk::DescriptorSet,last:Option<(u64,f64)>,enabled:bool }
pub(crate) struct ExposurePass {
    pub layout:vk::DescriptorSetLayout,
    pipeline_layout:vk::PipelineLayout,
    histogram:vk::Pipeline,adapt:vk::Pipeline,pool:vk::DescriptorPool,
    views:Vec<View>,active:usize,
    #[cfg(test)]
    pub time_override:Option<f64>,
}
fn finite(value:f32,default:f32,lo:f32,hi:f32)->f32 {
    (if value.is_finite() {value} else {default}).clamp(lo,hi)
}
impl ExposurePass {
    pub fn new(device:&ash::Device,alloc:&mut Allocator,source:vk::DescriptorSetLayout)->Result<Self> {
        let binding=[vk::DescriptorSetLayoutBinding::default().binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE|vk::ShaderStageFlags::FRAGMENT)];
        let layout=unsafe {device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding),None)?};
        let layouts=[source,layout];
        let push=[vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(size_of::<Params>() as u32)];
        let pipeline_layout=unsafe {device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&push),None)?};
        let mut pipelines=Vec::new();
        for bytes in [include_bytes!(concat!(env!("OUT_DIR"),"/exposure_histogram.comp.spv")).as_slice(),
            include_bytes!(concat!(env!("OUT_DIR"),"/exposure_adapt.comp.spv")).as_slice()] {
            let code=ash::util::read_spv(&mut std::io::Cursor::new(bytes))?;
            let module=unsafe {device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code),None)?};
            let info=vk::ComputePipelineCreateInfo::default().layout(pipeline_layout).stage(
                vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main"));
            let result=unsafe {device.create_compute_pipelines(vk::PipelineCache::null(),&[info],None)};
            unsafe {device.destroy_shader_module(module,None);}
            pipelines.push(result.map_err(|(_,e)|e)?[0]);
        }
        let sizes=[vk::DescriptorPoolSize {ty:vk::DescriptorType::STORAGE_BUFFER,descriptor_count:VIEWS as u32}];
        let pool=unsafe {device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(VIEWS as u32).pool_sizes(&sizes),None)?};
        let layouts=vec![layout;VIEWS];
        let sets=unsafe {device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&layouts))?};
        let mut views=Vec::new();
        for set in sets {
            let mut buffer=Buffer::new(device,alloc,SIZE,vk::BufferUsageFlags::STORAGE_BUFFER|vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::CpuToGpu,"view exposure histogram")?;
            buffer.write(0,&[0;SIZE as usize]);
            let info=[vk::DescriptorBufferInfo::default().buffer(buffer.handle).range(SIZE)];
            let writes=[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info)];
            unsafe {device.update_descriptor_sets(&writes,&[]);}
            views.push(View {buffer,set,last:None,enabled:false});
        }
        Ok(Self {layout,pipeline_layout,histogram:pipelines[0],adapt:pipelines[1],pool,views,active:0,
            #[cfg(test)] time_override:None})
    }
    pub fn set(&self)->vk::DescriptorSet {self.views[self.active].set}
    /// Record outside rendering, after the final HDR resolve. XR meters the
    /// first eye once per frame and shares that result with the other eye.
    #[allow(clippy::too_many_arguments)]
    pub fn record(&mut self,device:&ash::Device,cb:vk::CommandBuffer,source:vk::DescriptorSet,
        extent:vk::Extent2D,settings:AutoExposure,view:usize,serial:u64,time:f64) {
        self.active=view;
        #[cfg(test)]
        let time=self.time_override.unwrap_or(time);
        let state=&mut self.views[view];
        if !settings.enabled {state.enabled=false;state.last=None;return;}
        if state.last.is_some_and(|(frame,_)|frame==serial) {return;}
        let dt=state.last.map_or(0.0,|(_,last)|(time-last) as f32);
        let reset=!state.enabled || state.last.is_none() || !(0.0..=1.0).contains(&dt);
        state.enabled=true;state.last=Some((serial,time));
        let min=finite(settings.min_ev100,-10.0,-20.0,24.0);
        let max=finite(settings.max_ev100,20.0,min,24.0);
        let hmin=finite(settings.histogram_min_ev100,-10.0,-20.0,23.0);
        let hmax=finite(settings.histogram_max_ev100,20.0,hmin+0.01,24.0);
        let low=finite(settings.low_percent,70.0,0.0,99.9)*0.01;
        let high=finite(settings.high_percent,90.0,(low*100.0+0.1).min(100.0),100.0)*0.01;
        let calibration=(0.18f32*1.2).log2();
        let grid=[extent.width.clamp(1,256),extent.height.clamp(1,256)];
        let params=Params {histogram:[hmin+calibration,hmax+calibration,grid[0] as f32,grid[1] as f32],
            range_percent:[min+calibration,max+calibration,low,high],
            adaptation:[finite(settings.speed_up,3.0,0.0,20.0),finite(settings.speed_down,1.0,0.0,20.0),dt.clamp(0.0,1.0),if reset {1.0} else {0.0}]};
        unsafe {
            // Covers HDR produced by raster, RT/SSR, TAA or an upscaler, plus
            // the preceding frame's fragment reads of this persistent state.
            let before=[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::MEMORY_READ|vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE|vk::AccessFlags::SHADER_READ|vk::AccessFlags::SHADER_WRITE)];
            device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::ALL_COMMANDS,vk::PipelineStageFlags::TRANSFER|vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),&before,&[],&[]);
            device.cmd_fill_buffer(cb,state.buffer.handle,0,256,0);
            let clear=[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ|vk::AccessFlags::SHADER_WRITE)];
            device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::TRANSFER,vk::PipelineStageFlags::COMPUTE_SHADER,vk::DependencyFlags::empty(),&clear,&[],&[]);
            device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,self.pipeline_layout,0,&[source,state.set],&[]);
            device.cmd_push_constants(cb,self.pipeline_layout,vk::ShaderStageFlags::COMPUTE,0,bytemuck::bytes_of(&params));
            device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,self.histogram);
            device.cmd_dispatch(cb,grid[0].div_ceil(16),grid[1].div_ceil(16),1);
            let compute=[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ|vk::AccessFlags::SHADER_WRITE)];
            device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::COMPUTE_SHADER,vk::PipelineStageFlags::COMPUTE_SHADER,vk::DependencyFlags::empty(),&compute,&[],&[]);
            device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,self.adapt);
            device.cmd_dispatch(cb,1,1,1);
            let done=[vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ|vk::AccessFlags::HOST_READ)];
            device.cmd_pipeline_barrier(cb,vk::PipelineStageFlags::COMPUTE_SHADER,vk::PipelineStageFlags::FRAGMENT_SHADER|vk::PipelineStageFlags::COMPUTE_SHADER|vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),&done,&[],&[]);
        }
    }
    #[cfg(test)]
    pub fn debug_state(&self,device:&ash::Device,view:usize)->[f32;4] {
        unsafe {device.device_wait_idle().unwrap();}
        bytemuck::pod_read_unaligned(&self.views[view].buffer.mapped().unwrap()[256..272])
    }
    pub fn destroy(&mut self,device:&ash::Device,alloc:&mut Allocator) {
        for view in &mut self.views {view.buffer.destroy(device,alloc);}
        unsafe {device.destroy_pipeline(self.histogram,None);device.destroy_pipeline(self.adapt,None);
            device.destroy_pipeline_layout(self.pipeline_layout,None);device.destroy_descriptor_pool(self.pool,None);
            device.destroy_descriptor_set_layout(self.layout,None);}
    }
}
