//! Automatic directional radiance clipmaps. GPU demand/age selection and fixed
//! trace budgets keep CPU work independent of scene size; world tags preserve
//! overlapping cells when views move. Scene, Camera and game own their histories.
use crate::pipeline_store::PipelineDeviceExt as _;
use anyhow::Result;
use ash::vk;
use bytemuck::{Pod,Zeroable};
use glam::Vec3;
use gpu_allocator::{MemoryLocation,vulkan::Allocator};
use crate::{alloc::Buffer,FRAMES_IN_FLIGHT};
pub(crate) const PROBES:u32=3*16*16*16;
pub(crate) const PROBE_BYTES:u64=32+256*8+16;
pub(crate) const MAX_UPDATES:u32=512;
const WORK_BYTES:u64=(48+MAX_UPDATES as u64+PROBES as u64)*4;
#[repr(C)]
#[derive(Clone,Copy,Pod,Zeroable)]
struct Params {origin:[[i32;4];3],spacing:[f32;4],config:[u32;4],camera:[f32;4],changes:Changes}
/// A fixed-size, conservative union of local source changes. Overflow merges
/// into the last region instead of dropping changes or growing GPU work.
/// `regions[i][0].w` is 1 for changed geometry, -1 for a surface or light that changed in place
/// (the same thing at the same place with other light: animation, or an edit), 0 otherwise.
/// `regions[i][1].w` is how far beyond its bounds the change reaches into lighting histories, in
/// metres, negative when it also makes other surfaces' direct light stale (flux_changes.rs).
/// The radiance cache rejects the probes within the reach of changed geometry and refreshes
/// those within four reaches of any change (flux_radiance_manage.comp).
#[repr(C)]
#[derive(Clone,Copy,Pod,Zeroable)]
pub(crate) struct Changes {pub meta:[u32;4],pub regions:[[[f32;4];2];32]}
impl Default for Changes {fn default()->Self {Self::zeroed()}}
/// Larger than any scene.
pub(crate) const UNBOUNDED_REACH:f32=1.0e9;
/// A changed emitter makes surfaces follow it at once only where it delivers
/// at least this irradiance; its weight falls with the inverse square beyond.
const EMITTED_IRRADIANCE_FLOOR:f32=0.01;
/// Reach of a surface that moved or changed, emitting `emission` at most.
/// A body covering under a percent of a receiver's hemisphere cannot change
/// its light measurably: two bounding radii, wherever emitted light is weaker.
fn surface_reach(bounds:[Vec3;2],emission:f32)->f32 {
    let size=bounds[1]-bounds[0];
    let local=size.length();
    if emission.is_nan() || emission>=f32::MAX {return UNBOUNDED_REACH;}
    if emission<=0.0 {return local;}
    let area=(size.x*size.y).max(size.y*size.z).max(size.x*size.z);
    local.max((emission*area/EMITTED_IRRADIANCE_FLOOR).sqrt()).min(UNBOUNDED_REACH)
}
const GEOMETRY:f32=1.0;
const IN_PLACE:f32=-1.0;
const ELSEWHERE:f32=0.0;
impl Changes {
    pub fn extend(&mut self, other:&Self) {
        for region in &other.regions[..other.meta[0] as usize] {
            self.push_kind(region.map(|p|Vec3::from_slice(&p)),region[0][3],region[1][3]);
        }
    }
    /// An unclassified lighting change: assume it reaches everything.
    #[cfg(test)] pub fn push(&mut self,bounds:[Vec3;2]) {self.push_kind(bounds,ELSEWHERE,-UNBOUNDED_REACH);}
    /// Geometry without a body to measure (coarse-field coverage gained or lost): assume it reaches everything.
    pub fn occlusion(&mut self,bounds:[Vec3;2]) {self.push_kind(bounds,GEOMETRY,-UNBOUNDED_REACH);}
    /// Geometry moved, appeared or left, emitting radiance `emission` at most.
    /// It changes light where it is seen from, where it stands between a
    /// surface and a light (found per texel), and as far as its own light carries.
    pub fn moved(&mut self,bounds:[Vec3;2],emission:f32) {self.push_kind(bounds,GEOMETRY,surface_reach(bounds,emission));}
    /// A surface's material changed in place, by `emission` in emitted radiance at most.
    #[cfg(test)] pub fn surface(&mut self,bounds:[Vec3;2],emission:f32) {self.push_kind(bounds,IN_PLACE,surface_reach(bounds,emission));}
    /// A surface's material changed in place: its emission went from `old` to `new` (peak
    /// radiance), by `step` at most. Going dark or lighting up is an event however long the
    /// surface has been animating: light that is gone must not linger as if it were one more
    /// frame of a flicker.
    pub fn emission(&mut self,bounds:[Vec3;2],step:f32,old:f32,new:f32) {
        let switched=step>0.0 && (old<=0.0)!=(new<=0.0);
        self.push_kind(bounds,if switched {ELSEWHERE} else {IN_PLACE},surface_reach(bounds,step));
    }
    /// A finite light moved, turned, appeared or left; `bounds` covers its range, so only its bounce leaves them.
    pub fn light(&mut self,bounds:[Vec3;2]) {self.push_kind(bounds,ELSEWHERE,Self::light_reach(bounds));}
    /// A finite light changed its colour or intensity only.
    pub fn light_level(&mut self,bounds:[Vec3;2]) {self.push_kind(bounds,IN_PLACE,Self::light_reach(bounds));}
    fn light_reach(bounds:[Vec3;2])->f32 {-(0.25*(bounds[1]-bounds[0]).length()).max(1e-3)}
    fn push_kind(&mut self,bounds:[Vec3;2],kind:f32,reach:f32) {
        if !bounds[0].is_finite() || !bounds[1].is_finite() {return;}
        let mut slot=self.meta[0] as usize;
        // Only changes of one kind join: a body passing an animated surface must not lend it
        // its kind, nor take over its reach.
        for i in 0..slot {
            let lo=Vec3::from_slice(&self.regions[i][0]);let hi=Vec3::from_slice(&self.regions[i][1]);
            if self.regions[i][0][3]==kind && bounds[0].cmple(hi+Vec3::splat(0.01)).all() && bounds[1].cmpge(lo-Vec3::splat(0.01)).all() {slot=i;break;}
        }
        if slot>=32 {
            // Full: join the region this one extends least, of its own kind while there is one.
            let grown=|i:usize| {
                let lo=Vec3::from_slice(&self.regions[i][0]);let hi=Vec3::from_slice(&self.regions[i][1]);
                (self.regions[i][0][3]!=kind,(hi.max(bounds[1])-lo.min(bounds[0])).length()-(hi-lo).length())
            };
            slot=(0..32).min_by(|a,b|grown(*a).partial_cmp(&grown(*b)).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
        }
        if slot<self.meta[0] as usize {
            // Unlike kinds that had to join: geometry wins, then any change that is not in place.
            let flag=self.regions[slot][0][3].max(kind);
            let old=self.regions[slot][1][3];
            let widest=old.abs().max(reach.abs());
            let reach=if old<0.0 || reach<0.0 {-widest} else {widest};
            self.regions[slot][0]=Vec3::from_slice(&self.regions[slot][0]).min(bounds[0]).extend(flag).to_array();
            self.regions[slot][1]=Vec3::from_slice(&self.regions[slot][1]).max(bounds[1]).extend(reach).to_array();
        } else {
            self.regions[slot]=[bounds[0].extend(kind).to_array(),bounds[1].extend(reach).to_array()];self.meta[0]+=1;
        }
    }
}

#[cfg(test)]
mod change_tests {
    use super::*;
    #[test]
    fn occlusion_regions_preserve_rejection_through_merging_and_overflow() {
        let mut changes=Changes::default();
        for i in 0..32 {
            let p=Vec3::X*(i as f32*10.0);
            changes.push([p,p+Vec3::ONE]);
        }
        changes.occlusion([Vec3::X*400.0,Vec3::X*400.0+Vec3::ONE]);
        changes.push([Vec3::X*405.0,Vec3::X*405.0+Vec3::ONE]);
        assert_eq!(changes.meta[0],32);
        let mut merged=Changes::default();merged.extend(&changes);
        let regions=&merged.regions[..merged.meta[0] as usize];
        assert!(regions.iter().any(|r|r[0][3]==1.0 && r[0][0]<=400.0 && r[1][0]>=401.0),"overflow/extension discarded occlusion rejection");
        assert!(regions.iter().any(|r|r[0][0]<=405.0 && r[1][0]>=406.0),"overflow dropped a change");
        // Each joined its nearest neighbour, not one fixed slot.
        assert!(regions.iter().all(|r|r[1][0]-r[0][0]<=120.0),"overflow joined distant changes");
        assert_eq!(merged.regions[0][0][3],0.0,"unrelated lighting acquired hard rejection");
    }
}
struct View {
    data:Buffer,work:Buffer,params:Vec<Buffer>,sets:Vec<vk::DescriptorSet>,pool:vk::DescriptorPool,
    key:Option<u64>,epoch:u32,last_serial:u64,initialized:bool,
}
#[derive(Clone,Copy)]
pub(crate) struct Resources {pub data:vk::Buffer,pub work:vk::Buffer,pub params:vk::Buffer,pub enabled:bool,pub full_interpolation:bool}
pub(crate) struct Cache {
    layout:vk::PipelineLayout,set_layout:vk::DescriptorSetLayout,pipeline:vk::Pipeline,views:[Option<View>;3],
    #[cfg(test)] pub enabled_override:Option<bool>,
    #[cfg(test)] pub full_interpolation_override:Option<bool>,
    #[cfg(test)] pub adaptive_resolution_override:Option<bool>,
    #[cfg(test)] pub clipmap_blending_override:Option<bool>,
    /// Diagnostic ablation: local source/occluder changes neither age nor reject probes.
    #[cfg(test)] pub ignore_changes:bool,
}
pub(crate) fn barrier(device:&ash::Device,cb:vk::CommandBuffer) {unsafe {
    crate::sync_check::barrier2(&device, cb,&vk::DependencyInfo::default().memory_barriers(&[vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS).src_access_mask(vk::AccessFlags2::MEMORY_READ|vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS).dst_access_mask(vk::AccessFlags2::MEMORY_READ|vk::AccessFlags2::MEMORY_WRITE)]));
}}
impl Cache {
    pub fn new(device:&ash::Device)->Result<Self> {unsafe {
        let bindings=[vk::DescriptorType::STORAGE_BUFFER,vk::DescriptorType::STORAGE_BUFFER,vk::DescriptorType::UNIFORM_BUFFER]
            .into_iter().enumerate().map(|(i,t)|vk::DescriptorSetLayoutBinding::default().binding(i as u32).descriptor_type(t)
                .descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE)).collect::<Vec<_>>();
        let set_layout=device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),None)?;
        let layout=device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout])
            .push_constant_ranges(&[vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).size(4)]),None)?;
        let module=crate::screen_pass::shader_module(device,include_bytes!(concat!(env!("OUT_DIR"),"/flux_radiance_manage.comp.spv")))?;
        let result=device.create_cached_compute_pipelines(vk::PipelineCache::null(),&[vk::ComputePipelineCreateInfo::default().layout(layout)
            .stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main"))],None);
        device.destroy_shader_module(module,None);
        Ok(Self {layout,set_layout,pipeline:result.map_err(|(_,e)|e)?[0],views:std::array::from_fn(|_|None),
            #[cfg(test)] enabled_override:None,
            #[cfg(test)] full_interpolation_override:None,
            #[cfg(test)] adaptive_resolution_override:None,
            #[cfg(test)] clipmap_blending_override:None,
            #[cfg(test)] ignore_changes:false})
    }}
    fn view(&self,device:&ash::Device,alloc:&mut Allocator)->Result<View> {
        let data=Buffer::new(device,alloc,PROBES as u64*PROBE_BYTES,vk::BufferUsageFlags::STORAGE_BUFFER|vk::BufferUsageFlags::TRANSFER_SRC|vk::BufferUsageFlags::TRANSFER_DST,MemoryLocation::GpuOnly,"Flux directional world radiance")?;
        let work=Buffer::new(device,alloc,WORK_BYTES,vk::BufferUsageFlags::STORAGE_BUFFER|vk::BufferUsageFlags::INDIRECT_BUFFER|vk::BufferUsageFlags::TRANSFER_SRC|vk::BufferUsageFlags::TRANSFER_DST,MemoryLocation::GpuOnly,"Flux radiance update queue")?;
        let mut params=Vec::new();for _ in 0..FRAMES_IN_FLIGHT {params.push(Buffer::new(device,alloc,std::mem::size_of::<Params>() as u64,vk::BufferUsageFlags::UNIFORM_BUFFER,MemoryLocation::CpuToGpu,"Flux radiance clipmaps")?);}
        unsafe {
            let pool=device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(FRAMES_IN_FLIGHT as u32).pool_sizes(&[
                vk::DescriptorPoolSize {ty:vk::DescriptorType::STORAGE_BUFFER,descriptor_count:(2*FRAMES_IN_FLIGHT) as u32},
                vk::DescriptorPoolSize {ty:vk::DescriptorType::UNIFORM_BUFFER,descriptor_count:FRAMES_IN_FLIGHT as u32}]),None)?;
            let sets=device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&[self.set_layout; FRAMES_IN_FLIGHT]))?;
            for (i,&set) in sets.iter().enumerate() {
                for (binding,buffer) in [data.handle,work.handle,params[i].handle].into_iter().enumerate() {
                    let info=[vk::DescriptorBufferInfo::default().buffer(buffer).range(vk::WHOLE_SIZE)];
                    device.update_descriptor_sets(&[vk::WriteDescriptorSet::default().dst_set(set).dst_binding(binding as u32)
                        .descriptor_type(if binding==2 {vk::DescriptorType::UNIFORM_BUFFER} else {vk::DescriptorType::STORAGE_BUFFER}).buffer_info(&info)],&[]);
                }
            }
            Ok(View {data,work,params,sets,pool,key:None,epoch:0,last_serial:0,initialized:false})
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(&mut self,device:&ash::Device,alloc:&mut Allocator,cb:vk::CommandBuffer,view:usize,frame:usize,serial:u64,
        camera:Vec3,key:u64,changes:&Changes,trace_distance:f32,budget:u32,enabled:bool)->Result<Resources> {
        static ENABLED:std::sync::OnceLock<bool>=std::sync::OnceLock::new();
        let enabled=enabled && *ENABLED.get_or_init(||!std::env::var("ZESTENGINE_FLUX_RADIANCE_CACHE").is_ok_and(|v|v=="0"));
        #[cfg(test)] let enabled=enabled && self.enabled_override.unwrap_or(true);
        static FULL_INTERPOLATION:std::sync::OnceLock<bool>=std::sync::OnceLock::new();
        let full=*FULL_INTERPOLATION.get_or_init(||std::env::var("ZESTENGINE_FLUX_RADIANCE_INTERPOLATION").is_ok_and(|v|v=="full"));
        #[cfg(test)] let full=self.full_interpolation_override.unwrap_or(full);
        let adaptive=true;
        #[cfg(test)] let adaptive=self.adaptive_resolution_override.unwrap_or(adaptive);
        let blend=true;
        #[cfg(test)] let blend=self.clipmap_blending_override.unwrap_or(blend);
        #[cfg(test)] let ignore_changes=self.ignore_changes;
        if self.views[view].is_none() {self.views[view]=Some(self.view(device,alloc)?);}
        let target=self.views[view].as_mut().unwrap();
        if target.key!=Some(key) || target.last_serial+1!=serial {
            target.epoch=target.epoch.wrapping_add(1).max(1);target.key=Some(key);
        }
        target.last_serial=serial;
        let spacing=[1.0,4.0,16.0,3.0f32.sqrt()];
        let origin=std::array::from_fn(|i|{let p=(camera/spacing[i]).floor().as_ivec3()-glam::IVec3::splat(8);[p.x,p.y,p.z,0]});
        let mut changes=*changes;
        #[cfg(test)] if ignore_changes {changes.meta[0]=0;}
        changes.meta[1]=8; // bounded stale-history reuse while reactive work is queued
        changes.meta[2]=trace_distance.max(0.0).to_bits();
        target.params[frame].write(0,bytemuck::bytes_of(&Params {origin,spacing,config:[serial as u32,target.epoch,budget.clamp(16,128),if enabled {1|((cfg!(test) as u32)<<1)|((adaptive as u32)<<2)|((blend as u32)<<3)} else {0}],camera:[camera.x,camera.y,camera.z,40.0],changes}));
        barrier(device,cb);
        unsafe {
            if !target.initialized {device.cmd_fill_buffer(cb,target.data.handle,0,vk::WHOLE_SIZE,0);target.initialized=true;}
            device.cmd_fill_buffer(cb,target.work.handle,0,WORK_BYTES,0);
        }
        barrier(device,cb);
        if enabled {unsafe {
            device.cmd_bind_pipeline(cb,vk::PipelineBindPoint::COMPUTE,self.pipeline);
            device.cmd_bind_descriptor_sets(cb,vk::PipelineBindPoint::COMPUTE,self.layout,0,&[target.sets[frame]],&[]);
            for phase in 0u32..3 {
                device.cmd_push_constants(cb,self.layout,vk::ShaderStageFlags::COMPUTE,0,bytemuck::bytes_of(&phase));
                device.cmd_dispatch(cb,if phase==1 {1} else {PROBES.div_ceil(64)},1,1);barrier(device,cb);
            }
        }}
        Ok(Resources {data:target.data.handle,work:target.work.handle,params:target.params[frame].handle,enabled,full_interpolation:full})
    }
    #[cfg(test)] pub fn debug(&self,view:usize)->(vk::Buffer,vk::Buffer,u32) {
        let t=self.views[view].as_ref().unwrap();(t.data.handle,t.work.handle,t.epoch)
    }
    pub fn invalidate(&mut self) {for view in self.views.iter_mut().flatten() {view.key=None;}}
    pub fn destroy(&mut self,device:&ash::Device,alloc:&mut Allocator) {unsafe {
        for target in &mut self.views {if let Some(mut t)=target.take() {device.destroy_descriptor_pool(t.pool,None);
            t.data.destroy(device,alloc);t.work.destroy(device,alloc);for p in &mut t.params {p.destroy(device,alloc);}}}
        device.destroy_pipeline(self.pipeline,None);device.destroy_pipeline_layout(self.layout,None);device.destroy_descriptor_set_layout(self.set_layout,None);
    }}
}
