//! Immutable GPU snapshots of mesh-local distance fields. Voxel storage is
//! shared across transform updates; frame transients retain snapshots until
//! their fences retire, so moving objects never require a device-idle wait.
use std::{collections::BTreeMap, hash::{Hash, Hasher}, sync::Arc};
use anyhow::{Result, ensure};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};
use gpu_allocator::{MemoryLocation, vulkan::Allocator};
use crate::{alloc::Buffer, sdf::{SdfVolume,GiMeshSdfLevels}, GpuGiMaterial};
mod residency;

#[derive(Clone)]
pub struct GiMeshSdf {
    pub gi_source: Option<crate::GiSource>,
    pub world_to_local: Mat4,
    pub volume: Arc<SdfVolume>,
    pub material: GpuGiMaterial,
    /// Static meshes use detail tracing near the receiver, then the GDF.
    /// Dynamic/emissive meshes absent from the GDF trace to the full ray limit.
    pub in_global_field: bool,
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Instance {
    inverse: [[f32;4];4],
    lo_scale: [f32;4], hi_dynamic: [f32;4], dims_offset: [u32;4],
    albedo: [f32;4], emission: [f32;4],
    geometry_lo: [f32;4], geometry_hi: [f32;4],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Node { lo: [f32;3], instance: u32, hi: [f32;3], escape: u32 }
const _: () = assert!(size_of::<Instance>() == 176 && size_of::<Node>() == 32);
enum Image {
    Owned(crate::texture::GpuTexture),
    Paged { view: vk::ImageView, _owner: Arc<()> },
}
impl Image {
    fn view(&self)->vk::ImageView { match self {Self::Owned(t)=>t.view,Self::Paged {view,..}=>*view} }
    fn destroy(&mut self,device:&ash::Device,alloc:&mut Allocator) {if let Self::Owned(t)=self {t.destroy(device,alloc);}}
}
struct Voxels {
    texture: Image, volumes: Vec<Arc<SdfVolume>>, fields: Vec<Arc<SdfVolume>>, offsets: Vec<u32>,
    _leases: Vec<Arc<crate::texture::VolumeSlot>>,
}
pub(crate) struct Snapshot {
    instances: Buffer, nodes: Arc<Buffer>, voxels: Arc<Voxels>,
    pub node_count: u32,
    pub bounds: [Vec3;2],
    pub sources: Vec<Option<crate::GiSource>>,
}
impl Snapshot {
    pub fn resources(&self) -> ([vk::Buffer;2],vk::ImageView) { ([self.instances.handle,self.nodes.handle],self.voxels.texture.view()) }
    fn destroy(mut self, device: &ash::Device, alloc: &mut Allocator) {
        self.instances.destroy(device,alloc);
        if let Ok(mut nodes)=Arc::try_unwrap(self.nodes) {nodes.destroy(device,alloc);}
        if let Ok(mut voxels)=Arc::try_unwrap(self.voxels) { voxels.texture.destroy(device,alloc); }
    }
}
/// Instance (source, placed volume) and material stamp, with world bounds and peak emitted radiance.
type MaterialSources=BTreeMap<(Option<crate::GiSource>,u64,[u32;8]),([Vec3;2],f32)>;
/// Report every instance whose material was edited in place between two snapshots: the change
/// reaches as far as the light it adds or removes. An instance without a counterpart moved,
/// appeared or left; its body is reported with all it emits (`geometry_sources`).
fn material_changes(previous:&MaterialSources,next:&MaterialSources,changes:&mut crate::flux_radiance_cache::Changes) {
    for (key,(bounds,_)) in next {
        if previous.contains_key(key) {continue;}
        let edited=previous.range((key.0,key.1,[0;8])..=(key.0,key.1,[u32::MAX;8]))
            .map(|(same,_)|(GpuGiMaterial::stamp_emission_step(&key.2,&same.2),GpuGiMaterial::stamp_emission(&same.2)))
            .reduce(|a,b|(a.0.max(b.0),a.1.max(b.1)));
        if let Some((step,old))=edited {changes.emission(*bounds,step,old,GpuGiMaterial::stamp_emission(&key.2));}
    }
}
#[cfg(test)]
mod material_change_tests {
    use super::*;
    #[test]
    fn edits_in_place_reach_as_far_as_the_light_they_change() {
        let stamp=|emission:f32|GpuGiMaterial {albedo:[0.5;3],emission:[emission,0.0,0.0],metallic:0.0,roughness:1.0}.bits();
        let bounds=[Vec3::ZERO,Vec3::ONE];
        let reach=|previous:&MaterialSources,next:&MaterialSources| {
            let mut changes=crate::flux_radiance_cache::Changes::default();material_changes(previous,next,&mut changes);
            (changes.meta[0],changes.regions[0][1][3],changes.regions[0][0][3])
        };
        let source=|emission:f32,placed:u64|MaterialSources::from([((None,placed,stamp(emission)),(bounds,emission))]);
        let local=3.0f32.sqrt();
        assert_eq!(reach(&source(100.0,1),&source(100.0,1)).0,0,"an unchanged snapshot reported a change");
        // 100 -> 100.25 on a 1 m face: sqrt(0.25 / 0.01) = 5 m, not the 100 m of the whole emitter.
        let (count,flicker,kind)=reach(&source(100.0,1),&source(100.25,1));
        assert!(count==1 && (flicker-5.0).abs()<1e-3 && kind<0.0,"a small step of a bright emitter, in place: {flicker} {kind}");
        // Switched off: everything it lit, and an event rather than animation in place.
        let (_,off,kind)=reach(&source(100.0,1),&source(0.0,1));
        assert!((off-100.0).abs()<1e-2 && kind==0.0,"switched off: {off} {kind}");
        // A reflectance edit moves no emitted light: the surface's own surroundings.
        let recoloured=MaterialSources::from([((None,1,GpuGiMaterial {albedo:[0.1;3],emission:[100.0,0.0,0.0],metallic:0.0,roughness:1.0}.bits()),(bounds,100.0))]);
        assert!((reach(&source(100.0,1),&recoloured).1-local).abs()<1e-4);
        // Another placement is another instance: its body reports the move, with all of its light.
        assert_eq!(reach(&source(100.0,1),&source(100.0,2)).0,0);
    }
}
#[derive(Default)]
pub(crate) struct Cache {
    pub current: Option<Arc<Snapshot>>,
    retired: Vec<Arc<Snapshot>>,
    signature: Option<u64>,
    geometry_signature: Option<u64>,
    /// Instance identities (volume, source, field membership) without their transforms.
    structure_signature: Option<u64>,
    /// World bounds and the peak radiance the instance emits: a moved emitter relights as far as its light carries.
    geometry_sources: BTreeMap<(Option<crate::GiSource>,u64),([Vec3;2],f32)>,
    material_sources: MaterialSources,
    residency: Option<residency::Atlas>,
}
#[derive(Default)]
pub(crate) struct Update {
    pub pending: bool,
    pub streaming: bool,
    pub geometry_changed: bool,
    /// The geometry change moved existing instances only: no instance was added, removed or re-shaped.
    pub transform_only: bool,
    pub material_changes: crate::flux_radiance_cache::Changes,
    pub occlusion_changes: crate::flux_radiance_cache::Changes,
}
fn buffer(device: &ash::Device, alloc: &mut Allocator, bytes: &[u8], name: &str) -> Result<Buffer> {
    let mut b=Buffer::new(device,alloc,bytes.len().max(16) as u64,vk::BufferUsageFlags::STORAGE_BUFFER,
        MemoryLocation::CpuToGpu,name)?;
    if !bytes.is_empty() { b.write(0,bytes); }
    Ok(b)
}
fn bvh(nodes: &mut Vec<Node>, items: &mut [(u32,Vec3,Vec3)]) {
    let at=nodes.len();
    let lo=items.iter().fold(Vec3::splat(f32::INFINITY),|a,x| a.min(x.1));
    let hi=items.iter().fold(Vec3::splat(f32::NEG_INFINITY),|a,x| a.max(x.2));
    nodes.push(Node {lo:lo.to_array(),hi:hi.to_array(),instance:u32::MAX,escape:0});
    if items.len()==1 { nodes[at].instance=items[0].0; }
    else {
        let d=hi-lo; let axis=if d.x>=d.y && d.x>=d.z {0} else if d.y>=d.z {1} else {2};
        items.sort_by(|a,b| (a.1[axis]+a.2[axis]).total_cmp(&(b.1[axis]+b.2[axis])));
        let mid=items.len()/2; let (a,b)=items.split_at_mut(mid); bvh(nodes,a); bvh(nodes,b);
    }
    nodes[at].escape=nodes.len() as u32;
}
impl Cache {
    #[cfg(test)]
    pub(crate) fn test_streamed_extent(&mut self,device:&ash::Device,alloc:&mut Allocator,queue:vk::Queue,family:u32,dims:[u32;3],base_depth:u32)->Result<()> {
        ensure!(self.residency.is_none(),"test atlas already initialized");
        self.residency=Some(residency::Atlas::with_extent(device,alloc,queue,family,dims,base_depth)?);Ok(())
    }
    fn collect(&mut self,device: &ash::Device,alloc: &mut Allocator) {
        let mut i=0;
        while i<self.retired.len() {
            if Arc::strong_count(&self.retired[i])==1 {
                let old=self.retired.swap_remove(i);
                if let Ok(old)=Arc::try_unwrap(old) { old.destroy(device,alloc); }
            } else { i+=1; }
        }
    }
    pub fn set(&mut self,device: &ash::Device,alloc: &mut Allocator,pool: vk::CommandPool,queue: vk::Queue,inputs: &[GiMeshSdf]) -> Result<Update> {
        self.set_inner(device,alloc,pool,queue,inputs,None)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn set_streamed(&mut self,device:&ash::Device,alloc:&mut Allocator,queue:vk::Queue,family:u32,
        inputs:&[GiMeshSdf],levels:&[GiMeshSdfLevels],views:&[Vec3])->Result<Update> {
        self.set_inner(device,alloc,vk::CommandPool::null(),queue,inputs,Some((family,levels,views)))
    }
    #[allow(clippy::too_many_arguments)]
    fn set_inner(&mut self,device: &ash::Device,alloc: &mut Allocator,pool: vk::CommandPool,queue: vk::Queue,
        inputs: &[GiMeshSdf],stream:Option<(u32,&[GiMeshSdfLevels],&[Vec3])>) -> Result<Update> {
        self.collect(device,alloc);
        if stream.is_none()&& let Some(atlas)=self.residency.as_mut() {atlas.clear();}
        let mut hash=std::collections::hash_map::DefaultHasher::new();
        let mut structure=std::collections::hash_map::DefaultHasher::new();
        inputs.len().hash(&mut hash);
        let mut materials=Vec::with_capacity(inputs.len());
        let mut unique=BTreeMap::new();
        let mut geometry_sources=BTreeMap::new();
        let mut material_sources=BTreeMap::new();
        for i in inputs {
            let v=&i.volume;
            ensure!(i.world_to_local.is_finite() && i.world_to_local.determinant().abs()>1e-15,"invalid mesh SDF transform");
            ensure!(v.min.is_finite() && v.max.is_finite() && v.max.cmpgt(v.min).all()
                && v.dims.iter().all(|&d| d>=2) && v.data.len()==v.dims.iter().map(|&d|d as usize).product::<usize>(),"invalid mesh SDF volume");
            ensure!(v.geometry_bounds.iter().all(|b|b.is_finite())
                && v.geometry_bounds[1].cmpge(v.geometry_bounds[0]).all(),"invalid mesh geometry bounds");
            let id=Arc::as_ptr(v) as usize; id.hash(&mut hash); unique.insert(id,v.clone());
            id.hash(&mut structure); i.in_global_field.hash(&mut structure); i.gi_source.hash(&mut structure);
            for x in i.world_to_local.to_cols_array() {x.to_bits().hash(&mut hash);}
            materials.push(i.material.bits());
            i.in_global_field.hash(&mut hash);
            i.gi_source.hash(&mut hash);
            let mut instance_hash=std::collections::hash_map::DefaultHasher::new();
            id.hash(&mut instance_hash);i.world_to_local.to_cols_array().map(f32::to_bits).hash(&mut instance_hash);
            i.in_global_field.hash(&mut instance_hash);
            let world=i.world_to_local.inverse();let [lo,hi]=v.geometry_bounds;
            let c=world.transform_point3((lo+hi)*0.5);let h=(hi-lo)*0.5;
            let h=world.x_axis.truncate().abs()*h.x+world.y_axis.truncate().abs()*h.y+world.z_axis.truncate().abs()*h.z+Vec3::splat(0.002);
            let emission=i.material.peak_emission();
            geometry_sources.insert((i.gi_source,instance_hash.finish()),([c-h,c+h],emission));
            material_sources.insert((i.gi_source,instance_hash.finish(),i.material.bits()),([c-h,c+h],emission));
        }
        let geometry_signature=hash.finish();
        let structure_signature=structure.finish();
        materials.hash(&mut hash);
        let mut streaming=false;
        let paged=if let Some((family,levels,views))=stream {
            if self.residency.is_none() && !inputs.is_empty() {
                self.residency=Some(residency::Atlas::new(device,alloc,queue,family)?);
            }
            if let Some(atlas)=self.residency.as_mut() {
                let prepared=atlas.prepare(device,alloc,inputs,&unique,levels,views,geometry_signature)?;
                streaming=prepared.streaming;
                if prepared.pending {return Ok(Update {pending:true,streaming:true,..Default::default()});}
                prepared.voxels
            } else {None}
        } else {None};
        // Residency changes replace instance sampling metadata, not physical
        // geometry. They must not globally reset either view's lighting history.
        if let Some(v)=&paged {
            for (field,offset) in v.fields.iter().zip(&v.offsets) {(Arc::as_ptr(field) as usize).hash(&mut hash);offset.hash(&mut hash);}
        }
        let signature=hash.finish();
        if self.signature==Some(signature) { return Ok(Update {streaming,..Default::default()}); }
        let geometry_changed=self.geometry_signature!=Some(geometry_signature);
        let transform_only=geometry_changed && self.structure_signature==Some(structure_signature);
        let mut update=Update {geometry_changed,transform_only,streaming,..Default::default()};
        if geometry_changed {
            for (key,(bounds,emission)) in &self.geometry_sources {if !geometry_sources.contains_key(key) {update.occlusion_changes.moved(*bounds,*emission);}}
            for (key,(bounds,emission)) in &geometry_sources {if !self.geometry_sources.contains_key(key) {update.occlusion_changes.moved(*bounds,*emission);}}
        }
        // Simultaneous motion and material edits can affect different objects.
        // Match immutable source/geometry/material identities, not array order.
        material_changes(&self.material_sources,&material_sources,&mut update.material_changes);
        let next=if inputs.is_empty() { None } else {
            let volumes:Vec<_>=unique.into_values().collect();
            let existing=self.current.as_ref().map(|s|s.voxels.clone()).filter(|v| matches!(v.texture,Image::Owned(_)) &&
                v.volumes.len()==volumes.len() && v.volumes.iter().zip(&volumes).all(|(a,b)|Arc::ptr_eq(a,b)));
            let voxels=if let Some(v)=paged.or(existing) {v} else {
                let mut side=1u32;
                while (side*side*side) < volumes.len() as u32 {side+=1;}
                let brick: [u32;3]=std::array::from_fn(|a|volumes.iter().map(|v|v.dims[a]).max().unwrap()+2);
                let dims=brick.map(|n|n*side);
                ensure!(dims.iter().all(|&n|n<=1024),"mesh SDF atlas exceeds 3D texture extent");
                let count=dims.iter().map(|&d|d as usize).product::<usize>();
                ensure!(count<=32*1024*1024,"mesh SDF atlas exceeds 128 MiB budget");
                let mut data=vec![0.0;count]; let mut offsets=Vec::new();
                for (i,v) in volumes.iter().enumerate() {
                    let i=i as u32;
                    let base=[(i%side)*brick[0],((i/side)%side)*brick[1],(i/(side*side))*brick[2]];
                    // Three ten-bit texel origins avoid integer division in every ray step.
                    offsets.push((base[0]+1)|((base[1]+1)<<10)|((base[2]+1)<<20));
                    for z in 0..v.dims[2]+2 {for y in 0..v.dims[1]+2 {for x in 0..v.dims[0]+2 {
                        let q=[x.saturating_sub(1).min(v.dims[0]-1),y.saturating_sub(1).min(v.dims[1]-1),z.saturating_sub(1).min(v.dims[2]-1)];
                        let src=(q[0]+v.dims[0]*(q[1]+v.dims[1]*q[2])) as usize;
                        let dst=(base[0]+x+dims[0]*(base[1]+y+dims[1]*(base[2]+z))) as usize;
                        data[dst]=v.data[src];
                    }}}
                }
                Arc::new(Voxels {texture:Image::Owned(crate::texture::GpuTexture::upload_volume(device,alloc,pool,queue,&data,dims,None)?),fields:volumes.clone(),volumes,offsets,_leases:Vec::new()})
            };
            let mut instances=Vec::new(); let mut bounds=Vec::new();
            for (index,i) in inputs.iter().enumerate() {
                let inverse=i.world_to_local; let world=inverse.inverse();
                let b=Mat3::from_mat4(inverse); let g=b*b.transpose();
                let scale=(g.x_axis.abs()+g.y_axis.abs()+g.z_axis.abs()).max_element().sqrt().recip();
                // Unique volumes use pointer-key order in both atlas paths.
                let id=Arc::as_ptr(&i.volume) as usize;
                let vi=voxels.volumes.binary_search_by_key(&id,|v|Arc::as_ptr(v) as usize).unwrap();
                let v=&voxels.fields[vi];
                if let Some(old)=&self.current
                    && let Ok(oi)=old.voxels.volumes.binary_search_by_key(&id,|v|Arc::as_ptr(v) as usize)
                        && !Arc::ptr_eq(&old.voxels.fields[oi],v) {
                            let [lo,hi]=i.volume.geometry_bounds;let c=world.transform_point3((lo+hi)*0.5);let h=(hi-lo)*0.5;
                            let h=world.x_axis.truncate().abs()*h.x+world.y_axis.truncate().abs()*h.y+world.z_axis.truncate().abs()*h.z;
                            update.occlusion_changes.moved([c-h,c+h],i.material.peak_emission());
                        }
                instances.push(Instance {inverse:inverse.to_cols_array_2d(),lo_scale:v.min.extend(scale).to_array(),
                    hi_dynamic:v.max.extend(if i.in_global_field {0.0} else {1.0}).to_array(),
                    dims_offset:[v.dims[0],v.dims[1],v.dims[2],voxels.offsets[vi]],
                    albedo:[i.material.albedo[0],i.material.albedo[1],i.material.albedo[2],i.material.metallic],
                    emission:[i.material.emission[0],i.material.emission[1],i.material.emission[2],i.material.roughness],
                    geometry_lo:v.geometry_bounds[0].extend(0.0).to_array(),geometry_hi:v.geometry_bounds[1].extend(0.0).to_array()});
                let c=world.transform_point3((v.geometry_bounds[0]+v.geometry_bounds[1])*0.5);
                let h=(v.geometry_bounds[1]-v.geometry_bounds[0])*0.5;
                let h=world.x_axis.truncate().abs()*h.x+world.y_axis.truncate().abs()*h.y+world.z_axis.truncate().abs()*h.z;
                bounds.push((index as u32,c-h-Vec3::splat(0.002),c+h+Vec3::splat(0.002)));
            }
            let world_bounds=[bounds.iter().fold(Vec3::splat(f32::INFINITY),|v,b|v.min(b.1)),
                bounds.iter().fold(Vec3::splat(f32::NEG_INFINITY),|v,b|v.max(b.2))];
            let mut nodes=Vec::new();
            if geometry_changed {bvh(&mut nodes,&mut bounds);}
            let mut instance_buffer=match buffer(device,alloc,bytemuck::cast_slice(&instances),"Flux mesh SDF instances") {
                Ok(b)=>b,Err(e)=>{if let Ok(mut v)=Arc::try_unwrap(voxels){v.texture.destroy(device,alloc);}return Err(e)} };
            let node_count=if geometry_changed {nodes.len() as u32} else {self.current.as_ref().unwrap().node_count};
            let node_buffer=if !geometry_changed {self.current.as_ref().unwrap().nodes.clone()} else {Arc::new(match buffer(device,alloc,bytemuck::cast_slice(&nodes),"Flux mesh SDF BVH") {
                Ok(b)=>b,Err(e)=>{instance_buffer.destroy(device,alloc);if let Ok(mut v)=Arc::try_unwrap(voxels){v.texture.destroy(device,alloc);}return Err(e)} })};
            Some(Arc::new(Snapshot {sources:inputs.iter().map(|i|i.gi_source).collect(),instances:instance_buffer,nodes:node_buffer,voxels,node_count,bounds:world_bounds}))
        };
        if let Some(old)=self.current.take() {self.retired.push(old);}
        self.current=next; self.signature=Some(signature);self.geometry_signature=Some(geometry_signature);self.structure_signature=Some(structure_signature);self.geometry_sources=geometry_sources;self.material_sources=material_sources;
        self.collect(device,alloc);
        Ok(update)
    }
    pub fn destroy(&mut self,device: &ash::Device,alloc: &mut Allocator) {
        if let Some(current)=self.current.take() {self.retired.push(current);}
        self.collect(device,alloc);
        debug_assert!(self.retired.is_empty(),"mesh SDF frames must retire before cache destruction");
        if let Some(atlas)=self.residency.take()&& let Err(e)=atlas.destroy(device,alloc) {tracing::warn!("Flux atlas teardown: {e:#}");}
    }
}
