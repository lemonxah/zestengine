//! Immutable GPU snapshots of mesh-local distance fields. Voxel storage is
//! shared across transform updates; frame transients retain snapshots until
//! their fences retire, so moving objects never require a device-idle wait.
use std::{collections::BTreeMap, hash::{Hash, Hasher}, sync::Arc};
use anyhow::{Result, ensure};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Vec3};
use gpu_allocator::{MemoryLocation, vulkan::Allocator};
use crate::{alloc::Buffer, sdf::SdfVolume, GpuGiMaterial};

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
struct Voxels { texture: crate::texture::GpuTexture, volumes: Vec<Arc<SdfVolume>>, offsets: Vec<u32> }
pub(crate) struct Snapshot {
    instances: Buffer, nodes: Buffer, voxels: Arc<Voxels>,
    pub node_count: u32,
    pub sources: Vec<Option<crate::GiSource>>,
}
impl Snapshot {
    pub fn resources(&self) -> ([vk::Buffer;2],vk::ImageView) { ([self.instances.handle,self.nodes.handle],self.voxels.texture.view) }
    fn destroy(mut self, device: &ash::Device, alloc: &mut Allocator) {
        self.instances.destroy(device,alloc); self.nodes.destroy(device,alloc);
        if let Ok(mut voxels)=Arc::try_unwrap(self.voxels) { voxels.texture.destroy(device,alloc); }
    }
}
#[derive(Default)]
pub(crate) struct Cache {
    pub current: Option<Arc<Snapshot>>,
    retired: Vec<Arc<Snapshot>>,
    signature: Option<u64>,
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
    fn collect(&mut self,device: &ash::Device,alloc: &mut Allocator) {
        let mut i=0;
        while i<self.retired.len() {
            if Arc::strong_count(&self.retired[i])==1 {
                let old=self.retired.swap_remove(i);
                if let Ok(old)=Arc::try_unwrap(old) { old.destroy(device,alloc); }
            } else { i+=1; }
        }
    }
    pub fn set(&mut self,device: &ash::Device,alloc: &mut Allocator,pool: vk::CommandPool,queue: vk::Queue,inputs: &[GiMeshSdf]) -> Result<bool> {
        self.collect(device,alloc);
        let mut hash=std::collections::hash_map::DefaultHasher::new();
        let mut unique=BTreeMap::new();
        for i in inputs {
            let v=&i.volume;
            ensure!(i.world_to_local.is_finite() && i.world_to_local.determinant().abs()>1e-15,"invalid mesh SDF transform");
            ensure!(v.min.is_finite() && v.max.is_finite() && v.max.cmpgt(v.min).all()
                && v.dims.iter().all(|&d| d>=2) && v.data.len()==v.dims.iter().map(|&d|d as usize).product::<usize>(),"invalid mesh SDF volume");
            ensure!(v.geometry_bounds.iter().all(|b|b.is_finite())
                && v.geometry_bounds[1].cmpge(v.geometry_bounds[0]).all(),"invalid mesh geometry bounds");
            let id=Arc::as_ptr(v) as usize; id.hash(&mut hash); unique.insert(id,v.clone());
            for x in i.world_to_local.to_cols_array().into_iter().chain(i.material.albedo).chain(i.material.emission)
                .chain([i.material.metallic,i.material.roughness]) { x.to_bits().hash(&mut hash); }
            i.in_global_field.hash(&mut hash);
            i.gi_source.hash(&mut hash);
        }
        let signature=hash.finish();
        if self.signature==Some(signature) { return Ok(false); }
        let next=if inputs.is_empty() { None } else {
            let volumes:Vec<_>=unique.into_values().collect();
            let existing=self.current.as_ref().map(|s|s.voxels.clone()).filter(|v|
                v.volumes.len()==volumes.len() && v.volumes.iter().zip(&volumes).all(|(a,b)|Arc::ptr_eq(a,b)));
            let voxels=if let Some(v)=existing {v} else {
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
                Arc::new(Voxels {texture:crate::texture::GpuTexture::upload_volume(device,alloc,pool,queue,&data,dims)?,volumes,offsets})
            };
            let mut instances=Vec::new(); let mut bounds=Vec::new();
            for (index,i) in inputs.iter().enumerate() {
                let v=&i.volume; let inverse=i.world_to_local; let world=inverse.inverse();
                let b=Mat3::from_mat4(inverse); let g=b*b.transpose();
                let scale=(g.x_axis.abs()+g.y_axis.abs()+g.z_axis.abs()).max_element().sqrt().recip();
                let vi=voxels.volumes.iter().position(|a|Arc::ptr_eq(a,v)).unwrap();
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
            let mut nodes=Vec::new(); bvh(&mut nodes,&mut bounds);
            let mut instance_buffer=match buffer(device,alloc,bytemuck::cast_slice(&instances),"Flux mesh SDF instances") {
                Ok(b)=>b,Err(e)=>{if let Ok(mut v)=Arc::try_unwrap(voxels){v.texture.destroy(device,alloc);}return Err(e)} };
            let node_buffer=match buffer(device,alloc,bytemuck::cast_slice(&nodes),"Flux mesh SDF BVH") {
                Ok(b)=>b,Err(e)=>{instance_buffer.destroy(device,alloc);if let Ok(mut v)=Arc::try_unwrap(voxels){v.texture.destroy(device,alloc);}return Err(e)} };
            Some(Arc::new(Snapshot {sources:inputs.iter().map(|i|i.gi_source).collect(),instances:instance_buffer,nodes:node_buffer,voxels,node_count:nodes.len() as u32}))
        };
        if let Some(old)=self.current.take() {self.retired.push(old);}
        self.current=next; self.signature=Some(signature);
        self.collect(device,alloc);
        Ok(true)
    }
    pub fn destroy(&mut self,device: &ash::Device,alloc: &mut Allocator) {
        if let Some(current)=self.current.take() {self.retired.push(current);}
        self.collect(device,alloc);
        debug_assert!(self.retired.is_empty(),"mesh SDF frames must retire before cache destruction");
    }
}
