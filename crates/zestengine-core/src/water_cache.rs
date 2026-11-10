//! Portable shallow-water recordings. Asset IO and scene obstacle capture
//! live outside core; playback is identical in client and headless worlds.
use glam::{Vec2,Vec4};
use serde::{Deserialize,Serialize};

pub const WATER_CACHE_MAX_BYTES:usize=128*1024*1024;
#[derive(Clone,Debug,PartialEq,Serialize,Deserialize)]
#[serde(default)]
pub struct WaterCacheSettings {
    pub asset:String,
    pub looping:bool,
    pub speed:f32,
    pub offset:f32,
    pub gain:f32,
    pub spray:f32,
    pub loop_blend:f32,
    pub bake_seconds:f32,
    pub warmup_seconds:f32,
    pub frame_rate:u32,
    pub resolution:u32,
}
impl Default for WaterCacheSettings {
    fn default()->Self {Self{asset:String::new(),looping:true,speed:1.0,offset:0.0,gain:1.0,spray:0.5,loop_blend:0.5,bake_seconds:5.0,warmup_seconds:2.0,frame_rate:20,resolution:64}}
}
#[derive(Clone,Debug)]
pub struct WaterSimulationCache {
    pub n:u32,
    /// First cell centre in body-local metres.
    pub origin:[f32;2],
    pub cell:f32,
    pub frame_rate:f32,
    /// Time-major, row-major. Height, velocity X/Z perturbation, foam.
    pub frames:Vec<Vec<[f32;4]>>,
}
#[derive(Clone,Copy,Debug,Default)]
pub struct WaterCacheSample {pub height:f32,pub slope:Vec2,pub flow:Vec2,pub foam:f32}
impl WaterSimulationCache {
    pub fn validate(&self)->Result<(),String> {
        if !(8..=512).contains(&self.n) || !self.cell.is_finite() || self.cell<=0.0 || !self.frame_rate.is_finite() || !(1.0..=120.0).contains(&self.frame_rate)
            || !self.origin.iter().all(|v|v.is_finite()) || self.frames.is_empty() || self.frames.len()>7200 {return Err("invalid water cache dimensions or timebase".into());}
        let cells=self.n as usize*self.n as usize;
        if self.frames.len()*cells*16>WATER_CACHE_MAX_BYTES{return Err("water cache exceeds 128 MiB playback budget".into());}
        if self.frames.iter().any(|f|f.len()!=cells || f.iter().flatten().any(|v|!v.is_finite())){return Err("invalid water cache samples".into());}
        Ok(())
    }
    fn frame_weights(&self,time:f32,settings:&WaterCacheSettings)->[(usize,f32);4] {
        let count=self.frames.len();if count==0{return [(0,0.0);4];}
        let finite=|v:f32,d:f32|if v.is_finite(){v}else{d};
        let seconds=finite(time,0.0)*finite(settings.speed,1.0).clamp(-16.0,16.0)+finite(settings.offset,0.0);
        let duration=(count.saturating_sub(1))as f32/self.frame_rate;
        let blend=finite(settings.loop_blend,0.0).clamp(0.0,duration*0.5);
        let period=(duration-blend).max(1.0/self.frame_rate);
        let t=if settings.looping {seconds.rem_euclid(period)}else{seconds.clamp(0.0,duration)};
        let pair=|seconds:f32,weight:f32|{let f=(seconds*self.frame_rate).clamp(0.0,(count-1)as f32);let i=f.floor()as usize;[(i,weight*(1.0-f.fract())),((i+1).min(count-1),weight*f.fract())]};
        if settings.looping && blend>0.0 && t<blend {
            let w=t/blend;let w=w*w*(3.0-2.0*w);let a=pair(t,w);let b=pair(t+period,1.0-w);[a[0],a[1],b[0],b[1]]
        }else{let a=pair(t,1.0);[a[0],a[1],(0,0.0),(0,0.0)]}
    }
    pub fn frame(&self,time:f32,settings:&WaterCacheSettings)->Vec<[f32;4]> {
        let weights=self.frame_weights(time,settings);let gain=if settings.gain.is_finite(){settings.gain.clamp(0.0,4.0)}else{1.0};
        (0..self.n as usize*self.n as usize).map(|i|{let mut v=Vec4::ZERO;for(f,w)in weights{if w>0.0{v+=Vec4::from(self.frames[f][i])*w;}}(v*gain).to_array()}).collect()
    }
    pub fn sample(&self,p:Vec2,time:f32,settings:&WaterCacheSettings)->WaterCacheSample {
        if !p.is_finite() || self.frames.is_empty(){return Default::default();}
        let weights=self.frame_weights(time,settings);let n=self.n as i32;
        let at=|x:i32,z:i32|{let i=(z.clamp(0,n-1)*n+x.clamp(0,n-1))as usize;let mut v=Vec4::ZERO;for(f,w)in weights{if w>0.0{v+=Vec4::from(self.frames[f][i])*w;}}v};
        let q=(p-Vec2::from(self.origin))/self.cell;
        if q.min_element()<0.0 || q.max_element()>(n-1)as f32{return Default::default();}
        let b=q.floor().as_ivec2();let f=q.fract();
        let sample=|dx,dz|at(b.x+dx,b.y+dz).lerp(at(b.x+1+dx,b.y+dz),f.x).lerp(at(b.x+dx,b.y+1+dz).lerp(at(b.x+1+dx,b.y+1+dz),f.x),f.y);
        let v=sample(0,0);let slope=Vec2::new(sample(1,0).x-sample(-1,0).x,sample(0,1).x-sample(0,-1).x)/(2.0*self.cell);
        let edge=q.min((n-1)as f32-q).min_element();let fade=(edge/2.0).clamp(0.0,1.0);let fade=fade*fade*(3.0-2.0*fade);
        let gain=if settings.gain.is_finite(){settings.gain.clamp(0.0,4.0)}else{1.0};let gain=gain*fade;
        WaterCacheSample{height:v.x*gain,slope:slope*gain,flow:Vec2::new(v.y,v.z)*gain,foam:v.w.clamp(0.0,1.0)*gain}
    }
    /// Packed mips matching the shared water shader's cached field: three
    /// vec4s per texel, displacement/foam, gradient/variance, then flow.
    pub fn gpu_field(&self,time:f32,settings:&WaterCacheSettings)->Vec<[f32;4]> {
        let values=self.frame(time,settings);let n=self.n as usize;let mut fields=Vec::with_capacity(n*n*4);
        let at=|x:usize,z:usize|values[z*n+x][0];
        for z in 0..n{for x in 0..n{
            let v=values[z*n+x];let slope=Vec2::new(at((x+1).min(n-1),z)-at(x.saturating_sub(1),z),at(x,(z+1).min(n-1))-at(x,z.saturating_sub(1)))/(2.0*self.cell);
            fields.push([0.0,v[0],0.0,v[3].clamp(0.0,1.0)]);fields.push([slope.x,slope.y,0.0,slope.length_squared()]);fields.push([v[1],v[2],0.0,0.0]);
        }}
        crate::water_field::append_mips(&mut fields,n,3);fields
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cache_interpolates_space_time_and_loops_without_a_jump(){
        let frames=(0..21).map(|i|(0..64).map(|p|[(p%8)as f32*0.1+i as f32,0.2,0.3,0.0]).collect()).collect();
        let cache=WaterSimulationCache{n:8,origin:[0.0;2],cell:1.0,frame_rate:10.0,frames};cache.validate().unwrap();
        let settings=WaterCacheSettings{looping:false,..Default::default()};let sample=cache.sample(Vec2::splat(3.5),0.25,&settings);
        assert!((sample.height-2.85).abs()<1e-5);assert!((sample.slope.x-0.1).abs()<1e-5);
        assert_eq!(cache.sample(Vec2::splat(-1.0),1.0,&settings).height,0.0);
        let looping=WaterCacheSettings{looping:true,loop_blend:0.5,..settings};
        let a=cache.sample(Vec2::splat(3.5),1.5-1e-5,&looping);let b=cache.sample(Vec2::splat(3.5),1.5+1e-5,&looping);
        assert!((a.height-b.height).abs()<0.001);
        assert!((cache.sample(Vec2::splat(3.5),-0.25,&looping).height-cache.sample(Vec2::splat(3.5),1.25,&looping).height).abs()<1e-5);
    }
}
