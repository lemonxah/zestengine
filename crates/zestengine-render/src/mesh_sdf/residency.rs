//! Persistent coarse coverage plus view-prioritized mesh detail. Scene and
//! Camera snapshots retain their page reservations until frame retirement.
use super::*;
use crate::texture::{VolumePage,VolumePages,VolumeSlot,VolumeSlots};
use std::collections::BTreeSet;
#[cfg(test)] mod tests;

const DIMS:[u32;3]=[320;3]; // 125 MiB R32F; no full-size CPU mirror.
const BASE_DEPTH:u32=64;
const STAGING:u64=4*1024*1024;
const UPDATE_BYTES:usize=1024*1024;
const UPDATE_PAGES:usize=32;

struct Page {field:Arc<SdfVolume>,slot:Arc<VolumeSlot>,offset:u32}
struct Resident {detail:Arc<SdfVolume>,base:Page,fine:Option<Page>}
pub(super) struct Atlas {
    pages:VolumePages,base_slots:VolumeSlots,fine_slots:VolumeSlots,
    owner:Arc<()>,resident:BTreeMap<usize,Resident>,
    dims:[u32;3],base_depth:u32,repacking:bool,
    plan_key:Option<u64>,wanted:Vec<usize>,wanted_set:BTreeSet<usize>,
}
pub(super) struct Prepared {pub voxels:Option<Arc<Voxels>>,pub pending:bool,pub streaming:bool}

fn padded_size(v:&SdfVolume)->Result<[u32;3]> {
    ensure!(v.dims.iter().all(|&d|(2..=1022).contains(&d)),"mesh field cannot fit a packed atlas coordinate");
    Ok(v.dims.map(|d|d+2))
}
fn padded(v:&SdfVolume,size:[u32;3])->Vec<f32> {
    let mut data=Vec::with_capacity(size.iter().map(|&n|n as usize).product());
    for z in 0..size[2] {for y in 0..size[1] {for x in 0..size[0] {
        let q=[x.saturating_sub(1).min(v.dims[0]-1),y.saturating_sub(1).min(v.dims[1]-1),z.saturating_sub(1).min(v.dims[2]-1)];
        data.push(v.data[(q[0]+v.dims[0]*(q[1]+v.dims[1]*q[2])) as usize]);
    }}}
    data
}
fn priority(inputs:&[GiMeshSdf],views:&[Vec3])->BTreeMap<usize,f32> {
    let mut ranks=BTreeMap::new();
    for i in inputs {
        let world=i.world_to_local.inverse();let [lo,hi]=i.volume.geometry_bounds;
        let c=world.transform_point3((lo+hi)*0.5);let h=(hi-lo)*0.5;
        let h=world.x_axis.truncate().abs()*h.x+world.y_axis.truncate().abs()*h.y+world.z_axis.truncate().abs()*h.z;
        let distance=views.iter().filter(|v|v.is_finite()).map(|&v|((v-c).abs()-h).max(Vec3::ZERO).length_squared())
            .fold(f32::INFINITY,f32::min);
        ranks.entry(Arc::as_ptr(&i.volume) as usize).and_modify(|r:&mut f32|*r=r.min(distance)).or_insert(distance);
    }
    ranks
}

impl Atlas {
    pub fn clear(&mut self) {
        self.resident.clear();self.wanted.clear();self.wanted_set.clear();self.plan_key=None;self.repacking=false;
    }
    pub fn new(device:&ash::Device,alloc:&mut Allocator,queue:vk::Queue,family:u32)->Result<Self> {
        Self::with_extent(device,alloc,queue,family,DIMS,BASE_DEPTH)
    }
    pub(super) fn with_extent(device:&ash::Device,alloc:&mut Allocator,queue:vk::Queue,family:u32,dims:[u32;3],base_depth:u32)->Result<Self> {
        ensure!(base_depth>0 && base_depth<dims[2],"invalid mesh atlas partitions");
        Ok(Self {pages:VolumePages::new(device,alloc,queue,family,dims,STAGING)?,
            base_slots:VolumeSlots::new([dims[0],dims[1],base_depth])?,
            fine_slots:VolumeSlots::new([dims[0],dims[1],dims[2]-base_depth])?,
            owner:Arc::new(()),resident:BTreeMap::new(),dims,base_depth,repacking:false,
            plan_key:None,wanted:Vec::new(),wanted_set:BTreeSet::new()})
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare(&mut self,device:&ash::Device,alloc:&mut Allocator,inputs:&[GiMeshSdf],
        unique:&BTreeMap<usize,Arc<SdfVolume>>,levels:&[GiMeshSdfLevels],views:&[Vec3],geometry_signature:u64)->Result<Prepared> {
        self.pages.collect(device,alloc)?;
        debug_assert!(self.pages.staging_bytes()<=STAGING);
        self.resident.retain(|id,_|unique.contains_key(id));
        if unique.is_empty() {return Ok(Prepared {voxels:None,pending:false,streaming:false});}
        let coarse:BTreeMap<_,_>=levels.iter().map(|m|(Arc::as_ptr(&m.detail) as usize,&m.coarse)).collect();
        let mut requests=Vec::new();let mut base_voxels=0usize;
        for (&id,detail) in unique {
            let base=coarse.get(&id).copied().unwrap_or(detail);
            ensure!(base.min==detail.min && base.max==detail.max && base.geometry_bounds==detail.geometry_bounds,
                "coarse and detail fields must describe identical mesh bounds");
            let size=padded_size(base)?;
            ensure!(base.data.len()==base.dims.iter().map(|&n|n as usize).product::<usize>(),"invalid coarse mesh field");
            ensure!(size[0]<=self.dims[0] && size[1]<=self.dims[1] && size[2]<=self.base_depth,"coarse mesh field exceeds resident partition");
            base_voxels+=size.iter().map(|&n|n as usize).product::<usize>();
            // A different base is a new reservation; never overwrite a page an
            // old frame may sample, including a fine-to-coarse transition.
            if self.resident.get(&id).is_some_and(|r|!Arc::ptr_eq(&r.base.field,base)) {self.resident.remove(&id);}
            if !self.resident.contains_key(&id) {requests.push((id,base.clone()));}
        }
        ensure!(base_voxels<=self.dims[0] as usize*self.dims[1] as usize*self.base_depth as usize,"coarse mesh residency budget exhausted");
        let mut budget=UPDATE_BYTES;
        let (loaded,base_capacity)=self.load(device,alloc,&requests,false,&mut budget)?;
        for (id,page) in loaded {
            self.resident.insert(id,Resident {detail:unique[&id].clone(),base:page,fine:None});
        }
        // Returning an error lets the caller retain whole-world GDF coverage
        // and retire the previous mesh snapshot. Waiting on it here could never
        // free reservations owned by that same current snapshot.
        ensure!(!base_capacity,"coarse mesh atlas fragmented or held by previous snapshots");
        if unique.keys().any(|id|!self.resident.contains_key(id)) {
            return Ok(Prepared {voxels:None,pending:true,streaming:true});
        }

        let fine_dims=[self.dims[0],self.dims[1],self.dims[2]-self.base_depth];
        let mut key=std::collections::hash_map::DefaultHasher::new();geometry_signature.hash(&mut key);
        views.iter().filter(|v|v.is_finite()).map(|&v|(v/4.0).floor().to_array().map(f32::to_bits)).collect::<Vec<_>>().hash(&mut key);
        for r in self.resident.values() {(Arc::as_ptr(&r.base.field) as usize).hash(&mut key);}
        let key=key.finish();
        if self.plan_key!=Some(key) {
            let ranks=priority(inputs,views);
            let mut candidates:Vec<_>=unique.keys().filter(|id|!Arc::ptr_eq(&self.resident[id].detail,&self.resident[id].base.field)).copied().collect();
            candidates.sort_by(|a,b|ranks[a].total_cmp(&ranks[b]).then_with(||a.cmp(b)));
            // Verify a feasible layout once per geometry/view-cell change, not
            // on every streaming poll or settled lighting update.
            let mut remaining=fine_dims.iter().map(|&d|d as usize).product::<usize>()*4/5;
            let mut layout=VolumeSlots::new(fine_dims)?;let mut planned=Vec::new();
            self.wanted.clear();self.wanted_set.clear();
            for id in candidates {
                let size=padded_size(&unique[&id])?;
                let voxels=size.iter().map(|&n|n as usize).product::<usize>();
                if voxels<=remaining && voxels*4<=UPDATE_BYTES
                    && let Some(slot)=layout.reserve(size) {
                        remaining-=voxels;self.wanted.push(id);self.wanted_set.insert(id);planned.push(slot);
                    }
            }
            self.plan_key=Some(key);
        }
        for (&id,r) in &mut self.resident {if !self.wanted_set.contains(&id) {r.fine=None;}}
        let requests:Vec<_>=self.wanted.iter().filter(|id|self.resident[id].fine.is_none()).map(|&id|(id,unique[&id].clone())).collect();
        if self.repacking && self.fine_slots.reserve(fine_dims).is_some() {self.repacking=false;}
        if !self.repacking {
            let (loaded,capacity)=self.load(device,alloc,&requests,true,&mut budget)?;
            if capacity && loaded.is_empty() {
                // Publish coarse coverage, retire all old fine leases, then
                // recreate the feasible packing in priority order. Never spin
                // forever on an impossible fragmented set of live rectangles.
                for r in self.resident.values_mut() {r.fine=None;}
                self.repacking=true;
            } else {for (id,page) in loaded {self.resident.get_mut(&id).unwrap().fine=Some(page);}}
        }
        let streaming=self.wanted.iter().any(|id|self.resident[id].fine.is_none());
        let mut fields=Vec::with_capacity(unique.len());let mut offsets=Vec::with_capacity(unique.len());let mut leases=Vec::with_capacity(unique.len());
        for id in unique.keys() {
            let r=&self.resident[id];let page=r.fine.as_ref().unwrap_or(&r.base);
            fields.push(page.field.clone());offsets.push(page.offset);leases.push(page.slot.clone());
        }
        Ok(Prepared {voxels:Some(Arc::new(Voxels {
            texture:Image::Paged {view:self.pages.texture().unwrap().view,_owner:self.owner.clone()},
            volumes:unique.values().cloned().collect(),fields,offsets,_leases:leases,
        })),pending:false,streaming})
    }

    /// At most one bounded batch per level. Accepted uploads precede all new
    /// snapshots on the graphics queue; no per-page fence wait or CPU atlas copy.
    fn load(&mut self,device:&ash::Device,alloc:&mut Allocator,requests:&[(usize,Arc<SdfVolume>)],fine:bool,
        budget:&mut usize)->Result<(Vec<(usize,Page)>,bool)> {
        let mut staged=Vec::new();let mut bytes=0usize;let mut capacity=false;
        for (id,field) in requests.iter().take(UPDATE_PAGES) {
            let size=padded_size(field)?;let count=size.iter().map(|&n|n as usize).product::<usize>();
            // Large custom fields need slab streaming; generated fields are
            // <=33 texels/axis. Do not silently substitute an empty detail page.
            ensure!(count*4<=UPDATE_BYTES,"mesh field requires slab streaming beyond the per-update budget");
            if bytes+count*4>*budget {break;}
            let slots=if fine {&mut self.fine_slots} else {&mut self.base_slots};
            let Some(slot)=slots.reserve(size) else {capacity=true;continue;};
            let mut origin=slot.origin();if fine {origin[2]+=self.base_depth;}
            let offset=(origin[0]+1)|((origin[1]+1)<<10)|((origin[2]+1)<<20);
            let data=padded(field,size);bytes+=data.len()*4;
            staged.push((*id,Page {field:field.clone(),slot,offset},origin,data));
        }
        if staged.is_empty() {return Ok((Vec::new(),capacity));}
        let pages:Vec<_>=staged.iter().map(|(_,p,origin,data)|VolumePage {origin:*origin,size:p.slot.size(),data}).collect();
        if !self.pages.upload(device,alloc,&pages)? {return Ok((Vec::new(),capacity));}
        *budget-=bytes;
        Ok((staged.into_iter().map(|(id,p,_,_)|(id,p)).collect(),capacity))
    }

    pub fn destroy(self,device:&ash::Device,alloc:&mut Allocator)->Result<()> {
        ensure!(Arc::strong_count(&self.owner)==1,"mesh atlas snapshots must retire before destruction");
        self.pages.destroy(device,alloc)
    }
}
