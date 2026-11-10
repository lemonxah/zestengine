//! Stable physical pages, independent of logical card identity.
use std::collections::BTreeSet;
pub const MIN: u32 = 32;
pub const MAX: u32 = 128;
pub const ATLAS_MAX: u32 = 2048;
pub const CAPTURE_TEXELS: u32 = 12 * 64 * 64;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Page { pub x:u32, pub y:u32, pub size:u32 }
impl Page {
    pub fn rect(self)->[u32;4] {[self.x,self.y,self.size,1]}
    pub fn from_rect(r:[u32;4])->Self {Self {x:r[0],y:r[1],size:r[2]}}
}
pub struct Allocator { pub extent:u32, free:Vec<BTreeSet<[u32;2]>> }
fn level(size:u32)->usize {(size/MIN).ilog2() as usize}
impl Allocator {
    pub fn new(extent:u32)->Self {
        assert!(extent.is_power_of_two() && (MIN..=ATLAS_MAX).contains(&extent));
        let mut free=vec![BTreeSet::new();level(ATLAS_MAX)+1];
        free[level(extent)].insert([0,0]);Self {extent,free}
    }
    pub fn allocate(&mut self,size:u32)->Option<Page> {
        assert!(size.is_power_of_two() && (MIN..=MAX).contains(&size));
        let target=level(size);
        let source=(target..self.free.len()).find(|&l|!self.free[l].is_empty())?;
        let [x,y]=self.free[source].pop_first()?;
        for l in (target..source).rev() {
            let half=MIN<<l;
            self.free[l].extend([[x+half,y],[x,y+half],[x+half,y+half]]);
        }
        Some(Page {x,y,size})
    }
    pub fn release(&mut self,page:Page) {
        let (mut x,mut y,mut size)=(page.x,page.y,page.size);
        loop {
            let l=level(size);assert!(self.free[l].insert([x,y]),"page released twice");
            if size==self.extent {break;}
            let parent=size*2;let px=x&!(parent-1);let py=y&!(parent-1);
            let children=[[px,py],[px+size,py],[px,py+size],[px+size,py+size]];
            if !children.iter().all(|p|self.free[l].contains(p)) {break;}
            for child in children {self.free[l].remove(&child);}
            x=px;y=py;size=parent;
        }
    }
    pub fn grow(&mut self)->bool {
        if self.extent==ATLAS_MAX {return false;}
        let n=self.extent;self.free[level(n)].extend([[n,0],[0,n],[n,n]]);
        self.extent*=2;true
    }
}
/// Keep hysteresis in continuous projected size, before power-of-two snapping.
/// Off-screen ray hits retain useful detail without using a visibility cutoff.
pub fn desired(bounds:Option<[glam::Vec3;2]>,views:impl Iterator<Item=glam::Vec3>,current:u32,hit:bool,resolution:f32)->u32 {
    let Some([lo,hi])=bounds else {return MIN;};
    let span=(hi-lo).max_element().max(0.001);
    let distance=views.map(|p|(p-p.clamp(lo,hi)).length().max(1.0)).reduce(f32::min).unwrap_or(2.0);
    let resolution=if resolution.is_finite() {resolution.clamp(0.5,1.0)} else {1.0};
    let max_size=((MAX as f32*resolution) as u32).next_power_of_two().clamp(MIN,MAX);
    let mut projected=(span*32.0/distance).min(span*64.0)*resolution;
    if hit {projected=projected.max(48.0*resolution);}
    if current>0 && current<=max_size && projected>=current as f32*0.4 && projected<=current as f32*1.25 {return current;}
    (projected.ceil().max(1.0) as u32).min(max_size).next_power_of_two().clamp(MIN,max_size)
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn allocation_growth_and_coalescing_preserve_live_pages() {
        let mut a=Allocator::new(128);let first=a.allocate(64).unwrap();
        let mut small=Vec::new();while let Some(p)=a.allocate(32) {small.push(p);}
        assert_eq!(small.len(),12);assert!(a.grow());
        let large=a.allocate(128).unwrap();assert!(large.x>=128 || large.y>=128);
        let all:Vec<_>=small.iter().copied().chain([first,large]).collect();
        for (i,p) in all.iter().enumerate() {for q in &all[i+1..] {
            assert!(p.x+p.size<=q.x || q.x+q.size<=p.x || p.y+p.size<=q.y || q.y+q.size<=p.y);
        }}
        for p in all {a.release(p);}
        let mut blocks=Vec::new();while let Some(p)=a.allocate(128) {blocks.push(p);}
        assert_eq!(blocks.len(),4);
    }
    #[test] fn projected_detail_has_hysteresis_and_combines_views() {
        use glam::Vec3;
        let bounds=Some([Vec3::splat(-1.0),Vec3::splat(1.0)]);
        let far=Vec3::new(0.0,0.0,100.0);let near=Vec3::new(0.0,0.0,2.0);
        assert_eq!(desired(bounds,[far].into_iter(),0,false,1.0),32);
        assert_eq!(desired(bounds,[far,near].into_iter(),0,false,1.0),64);
        assert_eq!(desired(bounds,[near,far].into_iter(),0,false,1.0),64);
        for z in [2.0,2.01,1.99] {assert_eq!(desired(bounds,[Vec3::new(0.0,0.0,z)].into_iter(),64,false,1.0),64);}
        assert_eq!(desired(bounds,[far].into_iter(),32,true,1.0),64);
        let large=Some([Vec3::splat(-4.0),Vec3::splat(4.0)]);
        assert_eq!(desired(large,[Vec3::ZERO].into_iter(),128,false,0.5),64);
        assert_eq!(desired(large,[Vec3::ZERO].into_iter(),64,false,1.0),128);
        assert_eq!(desired(large,[Vec3::ZERO].into_iter(),0,false,f32::NAN),128);
    }
}
