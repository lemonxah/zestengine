//! Portable mesh-cloth poses and fixed-rate capture. Filesystem IO belongs to
//! assets; scene scheduling and render uploads belong to the engine.
use glam::Vec3;
use serde::{Deserialize,Serialize};

pub const CLOTH_CACHE_MAX_BYTES:usize=128*1024*1024;
pub const CLOTH_CACHE_MAX_VERTICES:usize=1_000_000;
pub const CLOTH_CACHE_MAX_FRAMES:usize=7201;

#[derive(Clone,Debug,PartialEq,Serialize,Deserialize)]
#[serde(default)]
pub struct ClothCacheSettings {
    pub asset:String,
    pub playback:bool,
    pub looping:bool,
    pub speed:f32,
    pub offset:f32,
    pub frame_rate:u32,
    pub record_seconds:f32,
}
impl Default for ClothCacheSettings {
    fn default()->Self{Self{asset:String::new(),playback:false,looping:false,speed:1.0,offset:0.0,frame_rate:30,record_seconds:5.0}}
}

/// Final object-space render poses, captured after all garment/body contacts.
/// Store normals too: imported smoothing and split vertices must replay exactly.
#[derive(Clone,Debug,PartialEq)]
pub struct ClothCacheFrame {pub positions:Vec<[f32;3]>,pub normals:Vec<[f32;3]>}
impl ClothCacheFrame {
    pub fn validate(&self,count:usize)->Result<(),String>{
        if self.positions.len()!=count||self.normals.len()!=count{return Err("cloth cache vertex count changed".into());}
        if self.positions.iter().chain(&self.normals).flatten().any(|v|!v.is_finite()){return Err("cloth cache contains a non-finite pose".into());}Ok(())
    }
    fn interpolate(a:&Self,b:&Self,t:f32)->Self{
        if t<=0.0{return a.clone();}if t>=1.0{return b.clone();}
        Self{positions:a.positions.iter().zip(&b.positions).map(|(&a,&b)|Vec3::from(a).lerp(Vec3::from(b),t).to_array()).collect(),
            normals:a.normals.iter().zip(&b.normals).map(|(&a,&b)|{
                let n=Vec3::from(a).lerp(Vec3::from(b),t);
                if n.length_squared()>1e-20{n.normalize().to_array()}else{if t<0.5{a}else{b}}
            }).collect()}
    }
}

#[derive(Clone,Debug,PartialEq)]
pub struct ClothSimulationCache {
    /// Stable fingerprint of bind positions and ordered topology. Equal vertex
    /// counts alone are not enough to safely replay a reimported garment.
    pub mesh_signature:u64,
    pub vertex_count:u32,
    pub frame_rate:f32,
    pub frames:Vec<ClothCacheFrame>,
}
impl ClothSimulationCache {
    pub fn validate(&self)->Result<(),String>{
        validate_dimensions(self.vertex_count as usize,self.frames.len(),self.frame_rate)?;
        for frame in &self.frames{frame.validate(self.vertex_count as usize)?;}Ok(())
    }
    pub fn duration(&self)->f64{self.frames.len().saturating_sub(1)as f64/self.frame_rate as f64}
    pub fn matches(&self,signature:u64,count:usize)->bool{self.mesh_signature==signature&&self.vertex_count as usize==count}
    pub fn sample(&self,seconds:f64,looping:bool)->Option<ClothCacheFrame>{
        validate_dimensions(self.vertex_count as usize,self.frames.len(),self.frame_rate).ok()?;
        if !seconds.is_finite(){return None;}
        let count=self.frames.len();
        let time=if looping{seconds.rem_euclid(count as f64/self.frame_rate as f64)}else{seconds.clamp(0.0,self.duration())};
        let frame=(time*self.frame_rate as f64).min(count as f64-f64::EPSILON);
        let i=(frame.floor()as usize).min(count-1);let j=if looping{(i+1)%count}else{(i+1).min(count-1)};
        self.frames[i].validate(self.vertex_count as usize).ok()?;self.frames[j].validate(self.vertex_count as usize).ok()?;
        Some(ClothCacheFrame::interpolate(&self.frames[i],&self.frames[j],frame.fract()as f32))
    }
}

pub fn validate_dimensions(vertices:usize,frames:usize,rate:f32)->Result<(),String>{
    if !(1..=CLOTH_CACHE_MAX_VERTICES).contains(&vertices)||!(1..=CLOTH_CACHE_MAX_FRAMES).contains(&frames)||!rate.is_finite()||!(1.0..=240.0).contains(&rate){return Err("invalid cloth cache dimensions or frame rate".into());}
    if vertices.checked_mul(frames).and_then(|n|n.checked_mul(24)).is_none_or(|n|n>CLOTH_CACHE_MAX_BYTES){return Err("cloth cache exceeds the 128 MiB budget".into());}Ok(())
}

/// Stable across processes and platforms; includes ordering and bind geometry.
pub fn cloth_cache_signature(positions:&[[f32;3]],indices:&[u32])->u64{
    let mut hash=0xcbf29ce484222325u64;
    let mut append=|bytes:&[u8]|{for &b in bytes{hash^=b as u64;hash=hash.wrapping_mul(0x100000001b3);}};
    append(&(positions.len()as u64).to_le_bytes());append(&(indices.len()as u64).to_le_bytes());
    for p in positions{for v in p{append(&v.to_bits().to_le_bytes());}}for i in indices{append(&i.to_le_bytes());}hash
}

/// Transient scene state. Cloning a cloth component deliberately drops this.
#[derive(Default)]
pub struct ClothCacheRuntime {
    pub data:Option<std::sync::Arc<ClothSimulationCache>>,
    pub recorder:Option<ClothCacheRecorder>,
    pub recorded:Option<std::sync::Arc<ClothSimulationCache>>,
    pub record_requested:bool,
    pub stop_requested:bool,
    pub save_requested:bool,
    pub unsaved:bool,
    pub playing:bool,
    pub playback_time:f64,
    pub recording_time:f64,
    pub error:String,
    pub signature:u64,
}

/// Resample irregular engine frames without changing the simulation timestep.
/// First capture defines time zero; subsequent timestamps must be monotonic.
pub struct ClothCacheRecorder {
    cache:ClothSimulationCache,
    max_frames:usize,
    start:Option<f64>,
    previous:Option<(f64,ClothCacheFrame)>,
}
impl ClothCacheRecorder {
    pub fn new(signature:u64,vertices:usize,frame_rate:u32,seconds:f32)->Result<Self,String>{
        if !seconds.is_finite()||seconds<=0.0{return Err("recording duration must be positive".into());}
        let frames=(seconds as f64*frame_rate as f64+1e-6).floor()+1.0;
        if frames>CLOTH_CACHE_MAX_FRAMES as f64{return Err("recording duration exceeds the frame budget".into());}
        let max_frames=frames as usize;validate_dimensions(vertices,max_frames,frame_rate as f32)?;
        Ok(Self{cache:ClothSimulationCache{mesh_signature:signature,vertex_count:vertices as u32,frame_rate:frame_rate as f32,frames:Vec::new()},max_frames,start:None,previous:None})
    }
    pub fn frame_count(&self)->usize{self.cache.frames.len()}
    pub fn matches(&self,signature:u64,vertices:usize)->bool{self.cache.matches(signature,vertices)}
    pub fn is_finished(&self)->bool{self.cache.frames.len()>=self.max_frames}
    /// Returns true once the requested duration is fully captured. Invalid
    /// input leaves the recording unchanged so it can still be saved/retried.
    pub fn capture(&mut self,time:f64,pose:ClothCacheFrame)->Result<bool,String>{
        if !time.is_finite(){return Err("invalid cloth recording timestamp".into());}
        pose.validate(self.cache.vertex_count as usize)?;
        if self.previous.as_ref().is_some_and(|(previous,_)|time<*previous){return Err("cloth recording clock moved backwards".into());}
        if self.is_finished(){return Ok(true);}
        let start=*self.start.get_or_insert(time);
        if let Some((previous,frame))=&self.previous{
            while self.cache.frames.len()<self.max_frames{
                let sample=start+self.cache.frames.len()as f64/self.cache.frame_rate as f64;
                if sample>time{break;}
                let t=if time>*previous{((sample-*previous)/(time-*previous)).clamp(0.0,1.0)as f32}else{0.0};
                self.cache.frames.push(ClothCacheFrame::interpolate(frame,&pose,t));
            }
        }else{self.cache.frames.push(pose.clone());}
        self.previous=Some((time,pose));Ok(self.is_finished())
    }
    pub fn finish(self)->Result<ClothSimulationCache,String>{self.cache.validate()?;Ok(self.cache)}
}

#[cfg(test)]
mod tests {
    use super::*;
    fn pose(x:f32)->ClothCacheFrame{ClothCacheFrame{positions:vec![[x,0.0,0.0];3],normals:vec![[0.0,1.0,0.0];3]}}
    #[test]
    fn irregular_capture_replays_the_same_timeline_and_clamps_or_loops(){
        let mut recorder=ClothCacheRecorder::new(17,3,10,0.5).unwrap();
        for t in [0.0,0.07,0.24,0.3,0.57]{recorder.capture(12.0+t,pose(t as f32*2.0)).unwrap();}
        assert!(recorder.is_finished());let cache=recorder.finish().unwrap();assert_eq!(cache.frames.len(),6);
        for(i,f)in cache.frames.iter().enumerate(){assert!((f.positions[0][0]-i as f32*0.2).abs()<1e-5);}
        assert!((cache.sample(0.35,false).unwrap().positions[0][0]-0.7).abs()<1e-5);
        assert_eq!(cache.sample(-5.0,false).unwrap(),cache.frames[0]);assert_eq!(cache.sample(7.0,false).unwrap(),cache.frames[5]);
        assert!((cache.sample(-0.05,true).unwrap().positions[0][0]-0.5).abs()<1e-5);
        assert_eq!(cache.sample(0.6,true).unwrap(),cache.frames[0]);assert!(cache.sample(f64::NAN,false).is_none());
    }
    #[test]
    fn capture_rejects_invalid_input_without_losing_recorded_frames(){
        assert!(ClothCacheRecorder::new(0,1_000_000,240,30.0).is_err());
        assert!(ClothCacheRecorder::new(0,3,0,1.0).is_err());
        let mut r=ClothCacheRecorder::new(0,3,10,1.0).unwrap();assert!(r.capture(0.0,pose(0.0)).is_ok());
        assert!(r.capture(-1.0,pose(1.0)).is_err());assert!(r.capture(0.1,pose(f32::NAN)).is_err());
        assert!(r.capture(0.1,ClothCacheFrame{positions:Vec::new(),normals:Vec::new()}).is_err());assert_eq!(r.frame_count(),1);
        r.capture(0.2,pose(0.2)).unwrap();assert_eq!(r.finish().unwrap().frames.len(),3);
    }
    #[test]
    fn geometry_identity_detects_same_count_reimports(){
        let positions=vec![[0.0,0.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0]];let a=cloth_cache_signature(&positions,&[0,1,2]);
        assert_ne!(a,cloth_cache_signature(&positions,&[0,2,1]));let mut moved=positions.clone();moved[0][0]=0.1;
        assert_ne!(a,cloth_cache_signature(&moved,&[0,1,2]));assert_eq!(a,cloth_cache_signature(&positions,&[0,1,2]));
    }
    #[test]
    fn interpolation_keeps_normals_finite_at_opposite_directions(){
        let a=pose(0.0);let mut b=pose(1.0);b.normals.fill([0.0,-1.0,0.0]);
        let cache=ClothSimulationCache{mesh_signature:0,vertex_count:3,frame_rate:1.0,frames:vec![a,b]};
        let sample=cache.sample(0.5,false).unwrap();assert_eq!(sample.positions[0],[0.5,0.0,0.0]);assert_eq!(Vec3::from(sample.normals[0]).length(),1.0);
    }
}
