//! Where the scene recently changed, for every consumer of lighting history.
//! Surface-cache lighting and the screen histories keep their accumulated light
//! except near these regions: an object moving in one corner must not cost the
//! rest of the scene its variance reduction. Regions outlive the frame of the
//! change, because a surface page may next be relit many frames later.
//!
//! A surface or light that keeps changing in place (a video, a pulse, a
//! flicker) is animation, not an event. After its first frames its bounce
//! light is averaged over the settled history like any other light instead of
//! being chased every frame: that would leave everything it lights with the
//! noise of a single update. Its direct light still follows at once.
use bytemuck::{Pod,Zeroable};
use glam::Vec3;
use crate::flux_radiance_cache::{Changes,UNBOUNDED_REACH as UNBOUNDED};

pub(crate) const REGIONS:usize=16;
/// A region keeps its place for at least this long after its last change.
const MIN_RETAINED_FRAMES:u64=64;
const MAX_RETAINED_FRAMES:u64=4000;
/// Added to a packed age; ages stay below it.
const DIRECT:f32=4096.0;
/// Added to a packed age: the region has been changing in place for a while.
const SUSTAINED:f32=8192.0;
/// A change in place that has gone on this long is animation.
const SUSTAINED_FRAMES:u64=16;
/// Animation may skip this many frames (a video below the display rate) and remain one run.
const PAUSE_FRAMES:u64=8;

/// Tail of the gather parameter block (`change_info`, `change_bounds` in GLSL).
/// `bounds[2i]` = min.xyz and the reach in metres, positive where geometry moved
/// and negative for a light or material change; `bounds[2i+1]` = max.xyz and the
/// frames since the region last changed, plus `DIRECT` where a light changed
/// (other surfaces' direct light is stale there), plus `SUSTAINED` for animation
/// in place (bounce light no longer chases it). `info.x` = region count.
#[repr(C)]
#[derive(Clone,Copy,Pod,Zeroable)]
pub struct Packed {pub info:[f32;4],pub bounds:[[f32;4];REGIONS*2]}

/// `since` is when the current run of changes in place began.
#[derive(Clone,Copy,Debug,PartialEq)]
struct Region {lo:Vec3,hi:Vec3,reach:f32,geometry:bool,direct:bool,in_place:bool,since:u64,last:u64}
impl Region {
    fn absorb(&mut self,other:&Region) {
        self.lo=self.lo.min(other.lo);self.hi=self.hi.max(other.hi);
        self.reach=self.reach.max(other.reach);self.last=self.last.max(other.last);
        self.geometry|=other.geometry;self.direct|=other.direct;
        self.in_place&=other.in_place;self.since=self.since.min(other.since);
    }
    fn same_kind(&self,other:&Region)->bool {self.geometry==other.geometry && self.in_place==other.in_place}
    fn touches(&self,other:&Region)->bool {
        other.lo.cmple(self.hi+Vec3::splat(0.01)).all() && other.hi.cmpge(self.lo-Vec3::splat(0.01)).all()
    }
}

/// One renderer-wide list, advanced once per frame serial and read by every view.
pub(crate) struct Recent {
    serial:Option<u64>,
    retain:u64,
    /// Regions grouped by when they started. A long path stays several short
    /// boxes with their own ages instead of one box that never expires.
    slices:Vec<(u64,Vec<Region>)>,
    /// A change without a place: the sun, the sky, unsupported occluders.
    everywhere:Option<u64>,
}
impl Default for Recent {
    fn default()->Self {Self {serial:None,retain:MIN_RETAINED_FRAMES,slices:Vec::new(),everywhere:None}}
}
impl Recent {
    /// Fold this frame's changes in. Repeated calls for one serial do nothing.
    pub fn advance(&mut self,serial:u64,changes:&Changes) {
        if self.serial==Some(serial) {return;}
        self.serial=Some(serial);
        let retain=self.retain;
        self.slices.retain_mut(|(_,regions)| {
            regions.retain(|r|serial.saturating_sub(r.last)<=retain);
            !regions.is_empty()
        });
        if self.everywhere.is_some_and(|last|serial.saturating_sub(last)>retain) {self.everywhere=None;}
        let count=(changes.meta[0] as usize).min(changes.regions.len());
        if count==0 {return;}
        let slice=(retain/4).max(8);
        if self.slices.last().is_none_or(|(start,_)|serial.saturating_sub(*start)>=slice) {self.slices.push((serial,Vec::new()));}
        let newest=self.slices.len()-1;
        let (earlier,current)=self.slices.split_at_mut(newest);
        let regions=&mut current[0].1;
        for region in &changes.regions[..count] {
            let (lo,hi)=(Vec3::from_slice(&region[0]),Vec3::from_slice(&region[1]));
            // The reporter measured the reach on the single object or light;
            // these bounds may already be a union of several.
            let mut new=Region {lo,hi,reach:region[1][3].abs(),geometry:region[0][3]>0.0,direct:region[1][3]<0.0,
                in_place:region[0][3]<0.0,since:serial,last:serial};
            match regions.iter_mut().find(|r|r.same_kind(&new) && r.touches(&new)) {
                Some(existing)=>{
                    // After a pause the next change in place is an event again.
                    let resumed=serial.saturating_sub(existing.last)>PAUSE_FRAMES;
                    existing.absorb(&new);
                    if resumed {existing.since=serial;}
                },
                None=>{
                    // A run of changes in place continues across slices.
                    if let Some(run)=earlier.iter().rev().flat_map(|(_,regions)|regions.iter())
                        .find(|r|r.same_kind(&new) && r.touches(&new) && serial.saturating_sub(r.last)<=PAUSE_FRAMES) {new.since=run.since;}
                    regions.push(new);
                },
            }
        }
    }
    /// Every surface is affected, as of this frame.
    pub fn everywhere(&mut self,serial:u64) {self.everywhere=Some(serial);}
    /// Keep regions until every page that waited through the change was relit.
    pub fn retain_for(&mut self,frames:u64) {self.retain=frames.clamp(MIN_RETAINED_FRAMES,MAX_RETAINED_FRAMES);}
    /// One line per region for the change log (ZESTENGINE_FLUX_CHANGES=1).
    pub fn describe(&self)->Vec<String> {
        let packed=self.packed();
        (0..packed.info[0] as usize).map(|i| {
            let (lo,hi)=(packed.bounds[i*2],packed.bounds[i*2+1]);
            let size=Vec3::from_slice(&hi)-Vec3::from_slice(&lo);
            let what=if lo[0]<=-UNBOUNDED {"everywhere"} else if lo[3]>0.0 {"moved geometry"}
                else if hi[3]%SUSTAINED>=DIRECT {"light"} else {"surface"};
            format!("{what}{}: centre {:.1?}, size {:.1?}, reach {:.1} m, {} frames ago",if hi[3]>=SUSTAINED {" (animation in place)"} else {""},
                (Vec3::from_slice(&lo)+size*0.5).to_array(),size.to_array(),lo[3].abs().min(9999.0),hi[3]%DIRECT)
        }).collect()
    }
    pub fn packed(&self)->Packed {
        let serial=self.serial.unwrap_or(0);
        let mut regions:Vec<Region>=self.slices.iter().rev().flat_map(|(_,r)|r.iter().copied()).collect();
        if let Some(last)=self.everywhere {
            regions.insert(0,Region {lo:Vec3::splat(-UNBOUNDED),hi:Vec3::splat(UNBOUNDED),reach:UNBOUNDED,geometry:false,direct:true,in_place:false,since:last,last});
        }
        // Over capacity, join the two regions whose union grows least: the
        // slices of one path, or anything a larger region already contains,
        // before objects that move in different places.
        while regions.len()>REGIONS {
            let size=|r:&Region|(r.hi-r.lo).length();
            let mut best=(f32::MAX,regions.len()-2,regions.len()-1);
            for a in 0..regions.len() {for b in a+1..regions.len() {
                if !regions[a].same_kind(&regions[b]) {continue;}
                let joined=(regions[a].hi.max(regions[b].hi)-regions[a].lo.min(regions[b].lo)).length();
                let growth=joined-size(&regions[a]).max(size(&regions[b]));
                if growth<best.0 {best=(growth,a,b);}
            }}
            let extra=regions.remove(best.2);
            regions[best.1].absorb(&extra);
        }
        let mut packed=Packed::zeroed();
        packed.info[0]=regions.len() as f32;
        for (i,r) in regions.iter().enumerate() {
            packed.bounds[i*2]=r.lo.extend(if r.geometry {r.reach} else {-r.reach}).to_array();
            let age=serial.saturating_sub(r.last).min(MAX_RETAINED_FRAMES) as f32;
            let sustained=r.in_place && r.last.saturating_sub(r.since)>=SUSTAINED_FRAMES;
            packed.bounds[i*2+1]=r.hi.extend(age+if r.direct {DIRECT} else {0.0}+if sustained {SUSTAINED} else {0.0}).to_array();
        }
        packed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cube(at:Vec3,half:f32)->[Vec3;2] {[at-Vec3::splat(half),at+Vec3::splat(half)]}
    fn regions(packed:&Packed)->Vec<([f32;4],[f32;4])> {
        (0..packed.info[0] as usize).map(|i|(packed.bounds[i*2],packed.bounds[i*2+1])).collect()
    }

    #[test]
    fn regions_carry_kind_reach_and_age_and_expire() {
        let mut recent=Recent::default();
        let mut changes=Changes::default();
        changes.moved(cube(Vec3::ZERO,0.5),0.0);
        changes.light(cube(Vec3::X*50.0,4.0));
        changes.surface(cube(Vec3::Z*50.0,1.0),8.0);
        // Too dim to matter beyond what it covers: a glowing detail on a prop.
        changes.moved(cube(Vec3::Y*50.0,1.0),0.001);
        recent.advance(10,&changes);
        // A second call in the same frame (another view) must not double count.
        recent.advance(10,&changes);
        let got=regions(&recent.packed());
        assert_eq!(got.len(),4);
        let radius=0.5*3.0f32.sqrt();
        assert!((got[0].0[3]-2.0*radius).abs()<1e-5,"moved geometry: positive reach of two radii, got {:?}",got[0]);
        assert!((got[1].0[3]+0.5*8.0*radius).abs()<1e-4,"light: negative reach of half its bounds radius, got {:?}",got[1]);
        // Radiance 8 over a 2 m face delivers the 0.01 irradiance floor out to sqrt(8*4/0.01) m.
        assert!((got[2].0[3]+(8.0f32*4.0/0.01).sqrt()).abs()<1e-2,"emitted light reaches as far as it is felt, got {:?}",got[2]);
        assert!((got[3].0[3]-2.0*3.0f32.sqrt()).abs()<1e-4,"a dim emitter stays as local as its body, got {:?}",got[3]);
        // Only the light makes other surfaces' direct light stale.
        assert_eq!(got.iter().map(|r|r.1[3]).collect::<Vec<_>>(),[0.0,DIRECT,0.0,0.0]);
        recent.advance(13,&Changes::default());
        assert_eq!(regions(&recent.packed()).iter().map(|r|r.1[3]).collect::<Vec<_>>(),[3.0,DIRECT+3.0,3.0,3.0],"age is frames since the last change");
        recent.advance(10+MIN_RETAINED_FRAMES,&Changes::default());
        assert_eq!(recent.packed().info[0],4.0,"still within retention");
        recent.advance(11+MIN_RETAINED_FRAMES,&Changes::default());
        assert_eq!(recent.packed().info[0],0.0,"expired regions must not linger");
    }

    #[test]
    fn a_long_path_stays_local_and_old_parts_expire() {
        let mut recent=Recent::default();
        // One metre per frame for 200 frames: the union of the whole path is 200 m long.
        for frame in 0..200u64 {
            let mut changes=Changes::default();
            changes.moved(cube(Vec3::X*frame as f32,0.5),0.0);
            changes.moved(cube(Vec3::X*(frame as f32+1.0),0.5),0.0);
            recent.advance(frame,&changes);
        }
        let got=regions(&recent.packed());
        assert!(got.len()<=REGIONS && got.len()>=2);
        let longest=got.iter().map(|(lo,hi)|hi[0]-lo[0]).fold(0.0,f32::max);
        assert!(longest<=MIN_RETAINED_FRAMES as f32/4.0+3.0,"one slice of the path, not all of it: {longest}");
        let oldest=got.iter().map(|(lo,_)|lo[0]).fold(f32::MAX,f32::min);
        assert!(oldest>=199.0-MIN_RETAINED_FRAMES as f32-17.0,"path older than the retention is forgotten: {oldest}");
        // The reach stays that of the object, not of the path.
        assert!(got.iter().all(|(lo,_)|lo[3]>0.0 && lo[3]<3.0),"{got:?}");
    }

    #[test]
    fn overflow_joins_neighbours_and_everywhere_covers_all() {
        let mut recent=Recent::default();
        let mut changes=Changes::default();
        // Twelve pairs of neighbours, the pairs far apart: more than the list holds.
        for i in 0..24 {changes.moved(cube(Vec3::X*((i/2) as f32*100.0+(i%2) as f32*3.0),0.5),0.0);}
        recent.advance(1,&changes);
        let got=regions(&recent.packed());
        assert_eq!(got.len(),REGIONS);
        let (lo,hi)=got.iter().fold((f32::MAX,f32::MIN),|(lo,hi),r|(lo.min(r.0[0]),hi.max(r.1[0])));
        assert!(lo<=-0.5 && hi>=1103.5,"overflow dropped a region: {lo}..{hi}");
        let widest=got.iter().map(|(lo,hi)|hi[0]-lo[0]).fold(0.0,f32::max);
        assert!(widest<=4.0+1e-3,"overflow joined distant movers instead of neighbours: {widest}");
        recent.everywhere(1);
        let got=regions(&recent.packed());
        assert_eq!(got.len(),REGIONS);
        assert!(got[0].0[0]<=-UNBOUNDED && got[0].1[0]>=UNBOUNDED && got[0].0[3]<=-UNBOUNDED && got[0].1[3]==DIRECT);
        // An unclassified change is treated as a light that reaches everything.
        let mut unknown=Recent::default();
        let mut changes=Changes::default();changes.occlusion(cube(Vec3::ZERO,1.0));
        unknown.advance(1,&changes);
        let got=regions(&unknown.packed());
        assert!(got[0].0[3]>=UNBOUNDED && got[0].1[3]>=DIRECT,"{got:?}");
        // A longer page wait extends how long regions are kept.
        recent.retain_for(500);
        recent.advance(400,&Changes::default());
        assert_eq!(recent.packed().info[0],REGIONS as f32);
        recent.advance(502,&Changes::default());
        assert_eq!(recent.packed().info[0],0.0);
    }

    #[test]
    fn animation_in_place_is_followed_as_an_event_only_when_it_starts() {
        let screen=cube(Vec3::ZERO,1.0);
        let play=|recent:&mut Recent,frames:std::ops::Range<u64>,step:u64| for frame in frames {
            let mut changes=Changes::default();
            if frame%step==0 {changes.surface(screen,8.0);changes.light_level(cube(Vec3::X*20.0,3.0));}
            recent.advance(frame,&changes);
        };
        // Flags of the regions covering the screen and the light, newest first.
        let flags=|recent:&Recent,at:f32|regions(&recent.packed()).iter().filter(|r|r.0[0]<=at && r.1[0]>=at)
            .map(|r|(r.1[3]>=SUSTAINED,r.1[3]%SUSTAINED>=DIRECT,r.1[3]%DIRECT)).collect::<Vec<_>>();
        let mut recent=Recent::default();
        play(&mut recent,0..SUSTAINED_FRAMES,1);
        assert_eq!(flags(&recent,0.0),[(false,false,0.0)],"the first frames of a video are an event");
        // A video at a third of the display rate, for two seconds.
        play(&mut recent,SUSTAINED_FRAMES..120,3);
        let (screen_flags,light_flags)=(flags(&recent,0.0),flags(&recent,20.0));
        assert!(screen_flags[0].0 && screen_flags[0].2<=2.0,"the newest region of a playing video must be sustained: {screen_flags:?}");
        assert!(screen_flags.iter().all(|f|f.0 || f.2>=32.0),"only the start of the run may still be followed, and it is old by now: {screen_flags:?}");
        assert!(light_flags[0].0 && light_flags[0].1,"a flickering light stays a direct-light change: {light_flags:?}");
        // Geometry moving through the same place is never animation in place, and does not join it.
        let mut changes=Changes::default();changes.surface(screen,8.0);changes.moved(screen,0.0);
        recent.advance(120,&changes);
        let got=regions(&recent.packed());
        assert!(got.iter().any(|r|r.0[3]>0.0 && r.1[3]==0.0) && got.iter().any(|r|r.0[3]<0.0 && r.1[3]>=SUSTAINED),"{got:?}");
        // Paused and switched on again: an event.
        play(&mut recent,121..121+PAUSE_FRAMES+2,u64::MAX);
        play(&mut recent,300..301,1);
        assert_eq!(flags(&recent,0.0)[0],(false,false,0.0),"a change after a pause is an event again");
    }
}
