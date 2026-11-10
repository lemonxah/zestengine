//! Persistent global distance samples. Unlike a local dirty bounding box,
//! nearest-source ownership remains correct outside an edited mesh's bounds:
//! removing the nearest source must reveal the next one even in empty space.
//! Grid-aligned domain moves reuse overlap and evaluate only exposed samples.
use super::{Gdf,SdfInstance};
use glam::Vec3;
use std::sync::Arc;

#[derive(Default,Clone,Copy,Debug)]
pub struct UpdateStats {
    pub reused:usize,
    pub rebuilt:usize,
    pub source_samples:usize,
}

#[derive(Default)]
pub struct Cache {
    field:Option<Gdf>,
    spare:Option<Gdf>,
    // Keep Arcs alive: pointer reuse after a mesh unload must never identify
    // an unrelated field as unchanged. Materials do not affect geometry.
    sources:Vec<SdfInstance>,
    pub stats:UpdateStats,
}

fn same_geometry(a:&SdfInstance,b:&SdfInstance)->bool {
    a.gi_source==b.gi_source && Arc::ptr_eq(&a.sdf,&b.sdf)
        && a.inv.to_cols_array().map(f32::to_bits)==b.inv.to_cols_array().map(f32::to_bits)
        && a.scale.to_bits()==b.scale.to_bits()
}
fn step(min:Vec3,max:Vec3,dims:[u32;3])->Vec3 {
    (max-min).max(Vec3::splat(1e-4))/Vec3::from_array(dims.map(|n|(n-1) as f32))
}
fn point(min:Vec3,step:Vec3,xyz:[u32;3])->Vec3 {
    min+Vec3::from_array(xyz.map(|n|n as f32))*step
}

impl Cache {
    pub fn field(&self)->Option<&Gdf> {self.field.as_ref()}
    pub fn update(&mut self,insts:&[SdfInstance],min:Vec3,max:Vec3,dims:[u32;3])->&Gdf {
        let dims=dims.map(|n|n.max(2));let count=dims.iter().map(|n|*n as usize).product();
        let previous=self.field.take();
        let mut next=self.spare.take().unwrap_or_else(||Gdf {dims,min,max,dist:Vec::new(),index:Vec::new()});
        next.dims=dims;next.min=min;next.max=max;
        next.dist.resize(count,f32::MAX);next.index.resize(count,0);
        let grid_step=step(min,max,dims);
        let unchanged:Vec<_>=insts.iter().enumerate()
            .map(|(i,s)|self.sources.get(i).is_some_and(|old|same_geometry(old,s))).collect();
        let changed:Vec<_>=insts.iter().enumerate().filter(|(i,_)|!unchanged[*i]).collect();
        let overlap=previous.as_ref().and_then(|old| {
            let old_step=step(old.min,old.max,old.dims);
            if old_step!=grid_step {return None;}
            let offset=(min-old.min)/grid_step;
            offset.is_finite().then_some((old,old_step,offset.to_array().map(|v|v.round() as i64)))
        });
        let slice=dims[0] as usize*dims[1] as usize;
        let chunk=(dims[2] as usize).div_ceil(super::bake_worker_count()).max(1)*slice;
        self.stats=std::thread::scope(|scope| {
            let mut workers=Vec::new();
            for (ci,(dist,index)) in next.dist.chunks_mut(chunk).zip(next.index.chunks_mut(chunk)).enumerate() {
                let unchanged=&unchanged;let changed=&changed;
                workers.push(scope.spawn(move || {
                    super::lower_compute_priority();
                    let mut stats=UpdateStats::default();
                    for (local,(value,winner)) in dist.iter_mut().zip(index.iter_mut()).enumerate() {
                        let at=ci*chunk+local;
                        let xyz=[(at%dims[0] as usize) as u32,((at/dims[0] as usize)%dims[1] as usize) as u32,(at/slice) as u32];
                        let p=point(min,grid_step,xyz);
                        let old_sample=overlap.and_then(|(old,old_step,offset)| {
                            let mut q=[0u32;3];
                            for axis in 0..3 {
                                let v=offset[axis].checked_add(i64::from(xyz[axis]))?;
                                if v<0 || v>=i64::from(old.dims[axis]) {return None;}
                                q[axis]=v as u32;
                            }
                            // Float rounding can make nominally equal grids
                            // differ at large origins. Reuse only exact sample
                            // positions, so thin geometry never shifts silently.
                            if point(old.min,old_step,q).to_array().map(f32::to_bits)!=p.to_array().map(f32::to_bits) {return None;}
                            let i=(q[2] as usize*old.dims[1] as usize+q[1] as usize)*old.dims[0] as usize+q[0] as usize;
                            unchanged.get(old.index[i] as usize).copied().unwrap_or(false)
                                .then_some((old.dist[i],old.index[i]))
                        });
                        let (mut best,mut who)=old_sample.unwrap_or((f32::INFINITY,0));
                        let mut sample=|i:usize,s:&SdfInstance| {
                            let d=s.sdf.sample(s.inv.transform_point3(p))*s.scale;stats.source_samples+=1;
                            // Full builds choose the first source on a tie.
                            if d<best || (d==best && (i as u32)<who) {best=d;who=i as u32;}
                        };
                        if old_sample.is_some() {
                            stats.reused+=1;
                            for &(i,s) in changed {sample(i,s);}
                        } else {
                            stats.rebuilt+=1;
                            for (i,s) in insts.iter().enumerate() {sample(i,s);}
                        }
                        *value=if insts.is_empty() {f32::MAX} else {best};*winner=who;
                    }
                    stats
                }));
            }
            let mut total=UpdateStats::default();
            for worker in workers {let s=worker.join().unwrap();total.reused+=s.reused;total.rebuilt+=s.rebuilt;total.source_samples+=s.source_samples;}
            total
        });
        self.sources=insts.to_vec();self.spare=previous;self.field=Some(next);
        self.field.as_ref().unwrap()
    }
}

#[cfg(test)] mod tests {
    use super::*;
    use glam::Mat4;
    fn source(id:u128,position:Vec3)->SdfInstance {
        let min=Vec3::splat(-1.0);let max=Vec3::ONE;let dims=[9;3];let mut data=Vec::new();
        for z in 0..9 {for y in 0..9 {for x in 0..9 {
            let p=min+Vec3::new(x as f32,y as f32,z as f32)*0.25;
            let q=p.abs()-Vec3::splat(0.5);
            data.push(q.max(Vec3::ZERO).length()+q.max_element().min(0.0));
        }}}
        SdfInstance {gi_source:Some(zestengine_render::GiSource {object:id,slot:0}),
            inv:Mat4::from_translation(-position),scale:1.0,
            sdf:Arc::new(zestengine_render::sdf::SdfVolume {dims,min,max,geometry_bounds:[Vec3::splat(-0.5),Vec3::splat(0.5)],data}),
            albedo:[0.5;3],emission:[0.0;3],emissive_surface:false,metallic:0.0,roughness:1.0,static_geometry:true}
    }
    fn compare(cache:&mut Cache,sources:&[SdfInstance],min:Vec3,max:Vec3,dims:[u32;3]) {
        let expected=super::super::build_gdf(sources,min,max,dims);
        let actual=cache.update(sources,min,max,dims);
        assert_eq!(actual.index,expected.index,"nearest-source identity changed");
        assert_eq!(actual.dist.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.dist.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"distance changed");
    }
    #[test] fn edits_removal_reordering_and_material_changes_match_full_rebuild() {
        let mut cache=Cache::default();let min=Vec3::new(-4.0,-3.0,-3.5);let max=-min;let dims=[17,13,15];
        let count=17*13*15;
        let mut sources=vec![source(1,Vec3::new(-2.0,0.0,0.0)),source(2,Vec3::new(2.0,0.0,0.0)),source(3,Vec3::new(0.0,2.0,0.0))];
        compare(&mut cache,&sources,min,max,dims);assert_eq!(cache.stats.source_samples,count*3);
        sources[0].albedo=[0.2;3];sources[1].emission=[8.0;3];
        compare(&mut cache,&sources,min,max,dims);assert_eq!(cache.stats.source_samples,0);assert_eq!(cache.stats.reused,count);
        sources[1].inv=Mat4::from_translation(Vec3::new(-1.0,0.0,-1.0));
        compare(&mut cache,&sources,min,max,dims);assert!(cache.stats.reused>0 && cache.stats.rebuilt>0);
        assert!(cache.stats.source_samples<count*3);
        sources.push(source(4,Vec3::ZERO));compare(&mut cache,&sources,min,max,dims);
        assert_eq!(cache.stats.source_samples,count); // new source can win anywhere
        sources.remove(1);compare(&mut cache,&sources,min,max,dims);
        sources.swap(0,2);compare(&mut cache,&sources,min,max,dims);
        sources.push(sources[0].clone());compare(&mut cache,&sources,min,max,dims);
        sources.rotate_right(1);compare(&mut cache,&sources,min,max,dims);
        sources[0].inv=Mat4::from_scale(Vec3::new(-1.0,0.5,2.0));sources[0].scale=0.5;
        compare(&mut cache,&sources,min,max,dims);
        sources[1].sdf=Arc::new((*sources[1].sdf).clone());Arc::make_mut(&mut sources[1].sdf).data.fill(0.2);
        compare(&mut cache,&sources,min,max,dims);
        compare(&mut cache,&[],min,max,dims);
        compare(&mut cache,&sources,min,max,dims);
    }
    #[test] fn scrolling_reuses_overlap_and_refreshes_exposed_slabs() {
        let mut cache=Cache::default();let mut min=Vec3::new(-4.0,-3.0,-3.5);let mut max=-min;let dims=[17,13,15];
        let sources=[source(1,Vec3::ZERO),source(2,Vec3::new(2.0,1.0,0.0))];
        compare(&mut cache,&sources,min,max,dims);
        for offset in [Vec3::X*0.5,-Vec3::Y,Vec3::Z*1.5,-Vec3::X*0.5] {
            min+=offset;max+=offset;compare(&mut cache,&sources,min,max,dims);
            assert!(cache.stats.reused>0 && cache.stats.rebuilt>0);
            assert_eq!(cache.stats.source_samples,cache.stats.rebuilt*2);
        }
        min+=Vec3::splat(0.1);max+=Vec3::splat(0.1);compare(&mut cache,&sources,min,max,dims);
        assert_eq!(cache.stats.reused,0);
        compare(&mut cache,&sources,min,max,[19,17,13]);assert_eq!(cache.stats.reused,0);
        compare(&mut cache,&sources,Vec3::splat(1e6),Vec3::splat(1e6+8.0),dims);
        assert_eq!(cache.stats.reused,0);
    }
}
