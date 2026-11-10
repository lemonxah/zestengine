//! Source-aware global-field publication. Unknown fields and incompatible grids
//! invalidate globally; source edits and aligned scrolling invalidate locally.
use std::collections::BTreeMap;
use anyhow::{Result,ensure};
use glam::Vec3;
use crate::{GiSource,flux_radiance_cache::Changes};

/// Metadata parallel to a GDF's material/index table. Bounds must enclose the
/// source's world-space distance-field support. Revision changes whenever its
/// mesh field, transform or distance scale changes; lighting is tracked apart.
#[derive(Clone,Copy,Debug)]
pub struct GiGdfGeometry {
    pub source:GiSource,
    pub revision:u64,
    pub bounds:[Vec3;2],
}

pub(crate) struct Metadata {
    domain:([u32;3],[u32;3],[u32;3]),
    padding:Vec3,
    entries:BTreeMap<GiSource,(usize,GiGdfGeometry)>,
}
impl Metadata {
    pub fn same_sources(&self,previous:&Self)->bool {
        self.entries.len()==previous.entries.len() && self.entries.iter().zip(&previous.entries).all(|((a,(i,x)),(b,(j,y)))|
            a==b && i==j && x.revision==y.revision && x.bounds==y.bounds)
    }
    pub fn represented_bounds(&self,sources:&[Option<GiSource>])->Option<[Vec3;2]> {
        let represented:std::collections::BTreeSet<_>=sources.iter().flatten().copied().collect();
        if self.entries.is_empty() || self.entries.keys().any(|s|!represented.contains(s)) {return None;}
        Some(self.entries.values().fold([Vec3::splat(f32::INFINITY),Vec3::splat(f32::NEG_INFINITY)],|b,(_,s)|
            [b[0].min(s.bounds[0]),b[1].max(s.bounds[1])]))
    }
    pub fn new(dims:[u32;3],min:[f32;3],max:[f32;3],geometry:&[GiGdfGeometry],count:usize)->Result<Self> {
        ensure!(geometry.len()==count,"GDF geometry count must match materials");
        ensure!(dims.iter().all(|&n|n>=2),"GDF dimensions must be at least two");
        let lo=Vec3::from_array(min);let hi=Vec3::from_array(max);
        ensure!(lo.is_finite() && hi.is_finite() && hi.cmpgt(lo).all(),"invalid GDF domain");
        let mut entries=BTreeMap::new();
        for (index,g) in geometry.iter().enumerate() {
            ensure!(g.bounds[0].is_finite() && g.bounds[1].is_finite() && g.bounds[1].cmpge(g.bounds[0]).all(),"invalid GDF source bounds");
            ensure!(entries.insert(g.source,(index,*g)).is_none(),"duplicate GDF source identity");
        }
        // Trilinear distance reconstruction can affect hits outside a source's
        // own SDF support; include neighbouring voxel footprints conservatively.
        let cell=(hi-lo)/Vec3::from_array(dims.map(|d|(d-1) as f32));
        Ok(Self {domain:(dims,min.map(f32::to_bits),max.map(f32::to_bits)),padding:Vec3::splat(cell.length()*2.0),entries})
    }

    /// Exact sample correspondence for immutable GPU overlap copies.
    pub fn scroll(&self,previous:&Self)->Option<crate::texture::VolumeScroll> {
        let (dims,min,max)=self.domain;let (old_dims,old_min,old_max)=previous.domain;
        if dims!=old_dims {return None;}
        let lo=Vec3::from_array(min.map(f32::from_bits));let hi=Vec3::from_array(max.map(f32::from_bits));
        let old_lo=Vec3::from_array(old_min.map(f32::from_bits));let old_hi=Vec3::from_array(old_max.map(f32::from_bits));
        let cell=(hi-lo)/Vec3::from_array(dims.map(|d|(d-1) as f32));
        let offset=(lo-old_lo)/cell;
        // Only exact lattice translations preserve the overlapping samples.
        // A resolution/scale/subcell change still requires full rejection.
        let lattice=lo/cell;let old_lattice=old_lo/cell;
        if cell.to_array().iter().any(|v|v.to_bits()&0x007f_ffff!=0)
            || lattice!=lattice.round() || old_lattice!=old_lattice.round()
            || (hi/cell).abs().max((old_hi/cell).abs()).max(lattice.abs()).max(old_lattice.abs()).max_element()>8_388_608.0
            || hi-lo!=old_hi-old_lo || offset!=offset.round()
            || old_lo+offset*cell!=lo || old_hi+offset*cell!=hi
            || !hi.min(old_hi).cmpgt(lo.max(old_lo)).all() {return None;}
        crate::texture::VolumeScroll::new(dims,offset.to_array().map(|v|v as i32))
    }

    pub fn changes(&self,previous:&Self,old_materials:&[[u32;8]],new_materials:&[[u32;8]])->Option<Changes> {
        let mut changes=Changes::default();
        if self.domain!=previous.domain {
            self.scroll(previous)?;
            let (_,min,max)=self.domain;let (_,old_min,old_max)=previous.domain;
            let lo=Vec3::from_array(min.map(f32::from_bits));let hi=Vec3::from_array(max.map(f32::from_bits));
            let old_lo=Vec3::from_array(old_min.map(f32::from_bits));let old_hi=Vec3::from_array(old_max.map(f32::from_bits));
            // Old-only space lost coarse coverage; new-only space gained it.
            // Both can affect rays. Do not invalidate the entire overlapping
            // world cache merely because the camera crossed a grid boundary.
            for (outer,inner,padding) in [([old_lo,old_hi],[lo,hi],previous.padding),([lo,hi],[old_lo,old_hi],self.padding)] {
                let mut rest=outer;
                for axis in 0..3 {
                    if rest[0][axis]<inner[0][axis] {
                        let mut end=rest[1];end[axis]=inner[0][axis];
                        changes.occlusion([rest[0]-padding,end+padding]);rest[0][axis]=inner[0][axis];
                    }
                    if rest[1][axis]>inner[1][axis] {
                        let mut begin=rest[0];begin[axis]=inner[1][axis];
                        changes.occlusion([begin-padding,rest[1]+padding]);rest[1][axis]=inner[1][axis];
                    }
                }
            }
        }
        self.append_source_changes(previous,old_materials,new_materials,previous.padding,self.padding,&mut changes);
        Some(changes)
    }
    /// During a background build, exact mesh geometry owns visibility; the
    /// placeholder grid's cell size/domain must not expand local invalidation.
    pub fn mesh_changes(&self,previous:&Self,old_materials:&[[u32;8]],new_materials:&[[u32;8]])->Changes {
        let mut changes=Changes::default();
        self.append_source_changes(previous,old_materials,new_materials,Vec3::splat(0.002),Vec3::splat(0.002),&mut changes);
        changes
    }
    fn append_source_changes(&self,previous:&Self,old_materials:&[[u32;8]],new_materials:&[[u32;8]],
        old_padding:Vec3,new_padding:Vec3,changes:&mut Changes) {
        for (source,(_,old)) in &previous.entries {
            if self.entries.get(source).is_none_or(|(_,new)|new.revision!=old.revision || new.bounds!=old.bounds) {
                changes.occlusion([old.bounds[0]-old_padding,old.bounds[1]+old_padding]);
            }
        }
        for (source,(index,new)) in &self.entries {
            let bounds=[new.bounds[0]-new_padding,new.bounds[1]+new_padding];
            match previous.entries.get(source) {
                Some((old_index,old)) if new.revision==old.revision && new.bounds==old.bounds => {
                    // Reordering can change nearest-source tie breaking for
                    // coincident fields. Refresh shading locally in that case.
                    if index!=old_index || new_materials[*index]!=old_materials[*old_index] {changes.push(bounds);}
                }
                _=>changes.occlusion(bounds),
            }
        }
    }

}

/// Automatic trace range covers all represented geometry, not only the coarse
/// field. Keep the old diagonal for cameras inside the bounds so moving the
/// view there does not continuously change the ray limit.
pub(crate) fn trace_distance(mut bounds:[Vec3;2],detail:Option<[Vec3;2]>,camera:Vec3)->f32 {
    if let Some(detail)=detail {bounds=[bounds[0].min(detail[0]),bounds[1].max(detail[1])];}
    (bounds[1]-bounds[0]).length().max((camera-bounds[0]).abs().max((camera-bounds[1]).abs()).length())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn automatic_range_covers_detail_and_external_views_without_camera_churn_inside() {
        let coarse=[-Vec3::ONE,Vec3::ONE];let detail=[Vec3::new(-2.0,-2.0,18.0),Vec3::new(2.0,2.0,22.0)];
        assert_eq!(trace_distance(coarse,None,Vec3::ZERO),Vec3::splat(2.0).length());
        let range=trace_distance(coarse,Some(detail),Vec3::ZERO);
        assert!(range>22.0);
        assert_eq!(range,trace_distance(coarse,Some(detail),Vec3::new(0.5,0.5,1.0)));
        for eye in [Vec3::new(50.0,40.0,-30.0),Vec3::new(-50.0,-40.0,30.0)] {
            let distance=trace_distance(coarse,Some(detail),eye);
            for x in [-2.0,2.0] {for y in [-2.0,2.0] {for z in [-1.0,22.0] {
                assert!(distance+1e-5>=eye.distance(Vec3::new(x,y,z)));
            }}}
        }
    }
    fn source(id:u128,x:f32)->GiGdfGeometry {
        GiGdfGeometry {source:GiSource {object:id,slot:0},revision:1,bounds:[Vec3::new(x,0.0,0.0),Vec3::new(x+1.0,1.0,1.0)]}
    }
    fn metadata(g:&[GiGdfGeometry])->Metadata {Metadata::new([128;3],[-100.0;3],[100.0;3],g,g.len()).unwrap()}
    #[test]
    fn pending_mesh_geometry_ignores_placeholder_grid_size_but_tracks_edits() {
        let source=source(1,0.0);let materials=[[0u32;8]];
        let full=metadata(&[source]);
        let pending=Metadata::new([2;3],[-10000.0;3],[10000.0;3],&[source],1).unwrap();
        assert_eq!(pending.mesh_changes(&full,&materials,&materials).meta[0],0);
        assert_eq!(full.mesh_changes(&pending,&materials,&materials).meta[0],0);
        let mut edited=source;edited.revision+=1;edited.bounds=edited.bounds.map(|p|p+Vec3::X*4.0);
        let changed=Metadata::new([2;3],[-10000.0;3],[10000.0;3],&[edited],1).unwrap();
        let changes=changed.mesh_changes(&pending,&materials,&materials);
        assert!(covers(&changes,Vec3::splat(0.5),true));
        assert!(covers(&changes,Vec3::new(4.5,0.5,0.5),true));
        assert!(!covers(&changes,Vec3::splat(100.0),true),"placeholder spacing invalidated distant lighting");
    }
    fn covers(c:&Changes,p:Vec3,hard:bool)->bool {
        c.regions[..c.meta[0] as usize].iter().any(|r|p.cmpge(Vec3::from_slice(&r[0])).all()
            && p.cmple(Vec3::from_slice(&r[1])).all() && (r[0][3]>0.0)==hard)
    }
    #[test]
    fn aligned_scroll_invalidates_changed_slabs_and_keeps_overlap() {
        let geometry=[source(1,0.0),source(2,12.0)];let materials=[[0;8];2];
        let lo=Vec3::splat(-32.0);let hi=Vec3::splat(31.5);
        let old=Metadata::new([128;3],lo.to_array(),hi.to_array(),&geometry,2).unwrap();
        for shift in [Vec3::X*4.0,-Vec3::X*4.0,Vec3::Y*8.0,Vec3::Z*12.0] {
            let next=Metadata::new([128;3],(lo+shift).to_array(),(hi+shift).to_array(),&geometry,2).unwrap();
            let changes=next.changes(&old,&materials,&materials).expect("integer scroll should preserve overlap");
            assert!(!covers(&changes,Vec3::ZERO,true),"camera movement reset unchanged central radiance");
            let axis=if shift.x!=0.0 {0} else if shift.y!=0.0 {1} else {2};
            let mut removed=Vec3::ZERO;let mut added=Vec3::ZERO;
            if shift[axis]>0.0 {removed[axis]=lo[axis]+0.5;added[axis]=hi[axis]+shift[axis]-0.5;}
            else {removed[axis]=hi[axis]-0.5;added[axis]=lo[axis]+shift[axis]+0.5;}
            assert!(covers(&changes,removed,true));assert!(covers(&changes,added,true));
        }
        for shift in [Vec3::X*0.125,Vec3::X*100.0] {
            let next=Metadata::new([128;3],(lo+shift).to_array(),(hi+shift).to_array(),&geometry,2).unwrap();
            assert!(next.changes(&old,&materials,&materials).is_none(),"incompatible/disjoint fields must reset");
        }
        let mut edited=geometry;edited[0].revision+=1;
        let next=Metadata::new([128;3],(lo+Vec3::X*4.0).to_array(),(hi+Vec3::X*4.0).to_array(),&edited,2).unwrap();
        let changes=next.changes(&old,&materials,&materials).unwrap();
        assert!(covers(&changes,Vec3::splat(0.5),true),"scroll hid a simultaneous geometry edit");
        assert!(!covers(&changes,Vec3::new(12.5,0.5,0.5),true));
    }
    #[test]
    fn tracked_gdf_changes_preserve_unrelated_geometry_and_validate_domains() {
        let initial=[source(1,-60.0),source(2,0.0),source(3,60.0)];
        let old=metadata(&initial);let material=[[0;8];3];
        assert_eq!(metadata(&initial).changes(&old,&material,&material).unwrap().meta[0],0);
        let mut next=initial;next[1]=source(2,12.0);next[1].revision=2;
        let mut new_material=material;new_material[0]=[7;8];
        let c=metadata(&next).changes(&old,&material,&new_material).unwrap();
        assert!(covers(&c,Vec3::new(0.5,0.5,0.5),true));
        assert!(covers(&c,Vec3::new(12.5,0.5,0.5),true));
        assert!(covers(&c,Vec3::new(-59.5,0.5,0.5),false));
        assert!(!covers(&c,Vec3::new(60.5,0.5,0.5),true));
        assert!(!covers(&c,Vec3::new(60.5,0.5,0.5),false));
        let removed=metadata(&[initial[0],initial[2]]).changes(&old,&material,&[[0;8];2]).unwrap();
        assert!(covers(&removed,Vec3::new(0.5,0.5,0.5),true));
        let reordered=metadata(&[initial[2],initial[0],initial[1]]).changes(&old,&material,&material).unwrap();
        assert!(reordered.regions[..reordered.meta[0] as usize].iter().all(|r|r[0][3]==0.0));
        assert!(metadata(&[]).changes(&old,&material,&[]).unwrap().meta[0]>0);
        for (dims,max) in [([64;3],[100.0;3]),([128;3],[200.0;3])] {
            assert!(Metadata::new(dims,[-100.0;3],max,&initial,3).unwrap().changes(&old,&material,&material).is_none());
        }
        assert!(Metadata::new([128;3],[-100.0;3],[100.0;3],&[initial[0],initial[0]],2).is_err());
        let mut invalid=initial;invalid[0].bounds[0].x=f32::NAN;
        assert!(Metadata::new([128;3],[-100.0;3],[100.0;3],&invalid,3).is_err());
    }
}
