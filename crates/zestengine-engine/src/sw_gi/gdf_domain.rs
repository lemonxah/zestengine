//! Geometry-field coverage independent of total world extent. The near domain
//! is an acceleration region: mesh fields must still cover rays outside it.
use glam::Vec3;

#[derive(Clone,Copy,Debug,PartialEq)]
pub(crate) struct Domain {pub min:Vec3,pub max:Vec3,pub dims:[u32;3]}

impl Domain {
    /// Coarser independently snapped regions. A scene that already fits in the
    /// base field, or a whole-world fallback, needs no additional allocations.
    pub fn outer(scene_min:Vec3,scene_max:Vec3,resolution:u32,eye:Option<Vec3>,previous:[Option<Self>;3],enabled:bool)->[Option<Self>;3] {
        let resolution=resolution.clamp(16,256);
        let cell=2.0f32.powf((32.0/(resolution-1) as f32).log2().ceil());
        let base_extent=cell*(resolution-1) as f32;
        let Some(eye)=eye.filter(|p|p.is_finite()) else {return [None;3]};
        if !enabled {return [None;3];}
        std::array::from_fn(|i| {
            let scale=(1u32<<(i+1)) as f32;
            // Stop once the preceding level covers the world's maximum span.
            if (scene_max-scene_min).max_element()<=base_extent*scale*0.5 {return None;}
            Self::snapped(resolution,cell*scale,eye,previous[i])
        })
    }
    fn snapped(resolution:u32,cell:f32,eye:Vec3,previous:Option<Self>)->Option<Self> {
        let extent=cell*(resolution-1) as f32;
        if let Some(old)=previous.filter(|d|d.dims==[resolution;3] && d.max-d.min==Vec3::splat(extent)) {
            let guard=Vec3::splat(extent*0.25);
            if eye.cmpge(old.min+guard).all() && eye.cmple(old.max-guard).all() {return Some(old);}
        }
        let snap=cell*(resolution/8).clamp(1,8) as f32;
        let min=((eye-Vec3::splat(extent*0.5))/snap).floor()*snap;
        let max=min+Vec3::splat(extent);
        (max.is_finite() && max.cmpgt(min).all()).then_some(Self {min,max,dims:[resolution;3]})
    }
    pub fn select(scene_min:Vec3,scene_max:Vec3,resolution:u32,eye:Option<Vec3>,previous:Option<Self>,world_probes:bool)->Self {
        let resolution=resolution.clamp(16,256);
        let size=scene_max-scene_min;
        // Powers of two preserve the exact lattice under scrolling. A default
        // 128-node field covers 63.5m at 0.5m/node, regardless of world size.
        let cell=2.0f32.powf((32.0/(resolution-1) as f32).log2().ceil());
        let extent=cell*(resolution-1) as f32;
        if !world_probes && size.max_element()>extent && let Some(eye)=eye.filter(|p|p.is_finite()) {
            if let Some(domain)=Self::snapped(resolution,cell,eye,previous) {return domain;}
        }
        // Preserve small-scene and legacy whole-world probe behaviour.
        let voxel=(size.max_element()/resolution as f32).max(0.03);
        Self {min:scene_min,max:scene_max,dims:size.to_array().map(|s|((s/voxel).round() as u32+1).clamp(8,resolution))}
    }
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn outer_levels_scroll_independently_and_stop_when_not_needed() {
        let eye=Vec3::new(5000.0,2.0,-5000.0);
        let lo=Vec3::splat(-10000.0);let hi=-lo;
        let levels=Domain::outer(lo,hi,128,Some(eye),[None;3],true);
        for (i,domain) in levels.iter().enumerate() {
            let d=domain.unwrap();
            assert_eq!(d.max-d.min,Vec3::splat(63.5*(1u32<<(i+1)) as f32));
            assert!(eye.cmpge(d.min).all() && eye.cmple(d.max).all());
        }
        assert_eq!(Domain::outer(lo,hi,128,Some(eye+Vec3::X),levels,true),levels);
        let moved=Domain::outer(lo,hi,128,Some(eye+Vec3::X*40.0),levels,true);
        assert_ne!(moved[0],levels[0]);assert_eq!(moved[2],levels[2]);
        for (i,(a,b)) in levels.iter().zip(moved).enumerate() {
            let delta=(b.unwrap().min-a.unwrap().min)/(1u32<<i) as f32;
            assert_eq!(delta,delta.round());
        }
        assert_eq!(Domain::outer(lo,hi,128,Some(eye),levels,false),[None;3]);
        assert_eq!(Domain::outer(lo,hi,128,None,levels,true),[None;3]);
        assert_eq!(Domain::outer(-Vec3::ONE,Vec3::ONE,128,Some(eye),levels,true),[None;3]);
        let small=Domain::outer(Vec3::splat(-50.0),Vec3::splat(50.0),128,Some(Vec3::ZERO),[None;3],true);
        assert!(small[0].is_some());assert_eq!(&small[1..],&[None;2]);
    }
    #[test] fn large_world_keeps_near_detail_and_scrolls_on_an_exact_lattice() {
        let eye=Vec3::new(5000.0,2.0,-5000.0);
        let field=Domain::select(Vec3::splat(-10000.0),Vec3::splat(10000.0),128,Some(eye),None,false);
        assert_eq!(field.max-field.min,Vec3::splat(63.5));
        assert!(eye.cmpge(field.min).all() && eye.cmple(field.max).all());
        let larger=Domain::select(Vec3::splat(-100000.0),Vec3::splat(100000.0),128,Some(eye),Some(field),false);
        assert_eq!(larger,field,"distant geometry must not reduce near detail");
        assert_eq!(Domain::select(Vec3::splat(-10000.0),Vec3::splat(10000.0),128,Some(eye+Vec3::X),Some(field),false),field);
        let moved=Domain::select(Vec3::splat(-10000.0),Vec3::splat(10000.0),128,Some(eye+Vec3::X*20.0),Some(field),false);
        assert_ne!(moved,field);
        assert_eq!((moved.min-field.min)/0.5,((moved.min-field.min)/0.5).round());
        assert!(moved.min.x<field.max.x,"ordinary scroll should reuse overlap");
        let other_view=Vec3::new(-5000.0,2.0,5000.0);
        let teleported=Domain::select(Vec3::splat(-10000.0),Vec3::splat(10000.0),128,Some(other_view),Some(field),false);
        assert!(other_view.cmpge(teleported.min).all() && other_view.cmple(teleported.max).all());
    }
    #[test] fn small_scenes_world_probes_and_invalid_views_keep_full_coverage() {
        let lo=Vec3::new(-10.0,-2.0,-10.0);let hi=-lo;
        let small=Domain::select(lo,hi,128,Some(Vec3::ZERO),None,false);
        assert_eq!(small.min,lo);assert_eq!(small.max,hi);
        for eye in [None,Some(Vec3::splat(f32::NAN))] {
            let full=Domain::select(lo*1000.0,hi*1000.0,128,eye,None,false);
            assert_eq!(full.min,lo*1000.0);assert_eq!(full.max,hi*1000.0);
        }
        let probes=Domain::select(lo*1000.0,hi*1000.0,128,Some(Vec3::ZERO),None,true);
        assert_eq!(probes.min,lo*1000.0);assert_eq!(probes.max,hi*1000.0);
        for resolution in [16,32,64,128,256] {
            let field=Domain::select(lo*1000.0,hi*1000.0,resolution,Some(Vec3::ZERO),None,false);
            let cell=(field.max-field.min).x/(resolution-1) as f32;
            assert_eq!(cell.log2(),cell.log2().round());
            assert!(field.min.cmplt(Vec3::ZERO).all() && field.max.cmpgt(Vec3::ZERO).all());
        }
    }
}
