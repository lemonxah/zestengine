//! Deterministic periodic ocean spectrum. The same FFT fields drive visible
//! displacement, normals, whitecaps and headless surface queries.
use glam::{Vec2,Vec3};
use serde::{Deserialize,Serialize};
use std::sync::Arc;

#[derive(Clone,Debug,PartialEq,Serialize,Deserialize)]
#[serde(default)]
pub struct OceanSpectrum {
    pub resolution:u32,
    pub patch_size:f32,
    pub wind_speed:f32,
    pub wind_direction:f32,
    pub significant_height:f32,
    pub choppiness:f32,
    pub depth:f32,
    pub seed:u32,
    /// Quantize dispersion to this repeat period; 0 uses free dispersion.
    pub loop_seconds:f32,
    pub foam_threshold:f32,
    pub foam_gain:f32,
}
impl Default for OceanSpectrum {
    fn default()->Self{Self{resolution:64,patch_size:128.0,wind_speed:10.0,wind_direction:20.0,significant_height:1.0,choppiness:0.8,depth:100.0,seed:17,loop_seconds:120.0,foam_threshold:0.65,foam_gain:2.0}}
}
impl OceanSpectrum {
    pub fn sanitized(&self)->Self {
        let f=|v:f32,d:f32,lo:f32,hi:f32|if v.is_finite(){v.clamp(lo,hi)}else{d};
        Self{resolution:self.resolution.clamp(16,256).next_power_of_two(),patch_size:f(self.patch_size,128.0,4.0,100000.0),
            wind_speed:f(self.wind_speed,10.0,0.0,100.0),wind_direction:f(self.wind_direction,20.0,-36000.0,36000.0),
            significant_height:f(self.significant_height,1.0,0.0,50.0),choppiness:f(self.choppiness,0.8,0.0,2.0),depth:f(self.depth,100.0,0.05,10000.0),
            loop_seconds:f(self.loop_seconds,120.0,0.0,3600.0),foam_threshold:f(self.foam_threshold,0.65,0.0,1.0),foam_gain:f(self.foam_gain,2.0,0.0,10.0),seed:self.seed}
    }
}

#[derive(Clone,Debug)]
pub struct OceanFrame {
    pub n:usize,
    pub size:f32,
    pub time:f32,
    /// Mips, fine to coarse. Two vec4s per texel: displacement xyz/foam,
    /// then height slopes x/z and horizontal Jacobian. Periodic at edges.
    pub fields:Vec<[f32;4]>,
}
impl OceanFrame {
    pub fn sample(&self,p:Vec2)->(Vec3,Vec3,f32) {
        let q=p/self.size*self.n as f32;let b=q.floor();let f=q-b;
        let at=|dx:i32,dz:i32,lane:usize|{let x=(b.x as i32+dx).rem_euclid(self.n as i32)as usize;let z=(b.y as i32+dz).rem_euclid(self.n as i32)as usize;glam::Vec4::from(self.fields[(z*self.n+x)*2+lane])};
        let sample=|lane|at(0,0,lane).lerp(at(1,0,lane),f.x).lerp(at(0,1,lane).lerp(at(1,1,lane),f.x),f.y);
        let d=sample(0);let s=sample(1);
        (d.truncate(),Vec3::new(-s.x,1.0,-s.y).normalize_or(Vec3::Y),d.w)
    }
}

#[derive(Debug)]
struct Spectrum {settings:OceanSpectrum,h0:Vec<[f32;2]>,omega:Vec<f32>}
impl Spectrum {
    fn new(settings:OceanSpectrum)->Self {
        let n=settings.resolution as usize;let mut h0=vec![[0.0;2];n*n];let mut omega=vec![0.0;n*n];
        let mut rng=settings.seed;let mut rand=||{rng=rng.wrapping_mul(1664525).wrapping_add(1013904223);((rng>>8)as f32+0.5)/16777216.0};
        let angle=settings.wind_direction.to_radians();let wind=Vec2::new(angle.cos(),angle.sin());
        let length=settings.wind_speed.powi(2)/9.81;
        let freq=|i:usize|if i<n/2{i as f32}else{i as f32-n as f32};
        for z in 0..n {for x in 0..n {
            if x==n/2 || z==n/2 || x==0 && z==0 {continue;}
            let k=Vec2::new(freq(x),freq(z))*(std::f32::consts::TAU/settings.patch_size);let k2=k.length_squared();let km=k2.sqrt();
            let directional=k.dot(wind)/km;
            let p=if length>0.0 {(-1.0/(k2*length*length)).exp()*(-k2*(settings.patch_size/n as f32*0.35).powi(2)).exp()*directional.powi(2)/k2.powi(2)*if directional<0.0{0.15}else{1.0}}else{0.0};
            let radius=(-2.0*rand().ln()).sqrt();let angle=std::f32::consts::TAU*rand();let(s,c)=angle.sin_cos();
            h0[z*n+x]=[radius*c*(p*0.5).sqrt(),radius*s*(p*0.5).sqrt()];
            let w=(9.81*km*(km*settings.depth).tanh()).sqrt();
            omega[z*n+x]=if settings.loop_seconds>0.0{let base=std::f32::consts::TAU/settings.loop_seconds;(w/base).round()*base}else{w};
        }}
        let energy=h0.iter().map(|h|h[0]as f64*h[0]as f64+h[1]as f64*h[1]as f64).sum::<f64>();
        let gain=if energy>0.0{settings.significant_height as f64*(n*n)as f64/(4.0*(2.0*energy).sqrt())}else{0.0};
        for h in &mut h0{h[0]=(h[0]as f64*gain)as f32;h[1]=(h[1]as f64*gain)as f32;}
        Self{settings,h0,omega}
    }
    fn frame(&self,time:f32)->OceanFrame {
        let n=self.settings.resolution as usize;let mut height=vec![[0.0;2];n*n];let mut dx=height.clone();let mut dz=height.clone();
        let t=if self.settings.loop_seconds>0.0{time.rem_euclid(self.settings.loop_seconds)}else{time};
        let freq=|i:usize|if i<n/2{i as f32}else{i as f32-n as f32};
        for z in 0..n {for x in 0..n {
            let i=z*n+x;let a=self.h0[i];let b=self.h0[((n-z)%n)*n+(n-x)%n];let(s,c)=(self.omega[i]*t).sin_cos();
            let h=[(a[0]+b[0])*c-(a[1]+b[1])*s,(a[0]-b[0])*s+(a[1]-b[1])*c];height[i]=h;
            let k=Vec2::new(freq(x),freq(z)).normalize_or_zero()*self.settings.choppiness;
            dx[i]=[-h[1]*k.x,h[0]*k.x];dz[i]=[-h[1]*k.y,h[0]*k.y];
        }}
        for field in [&mut height,&mut dx,&mut dz]{crate::water_ewave::fft2(field,n,true);}
        let mut fields=Vec::with_capacity(n*n*8/3);let inv=0.5*n as f32/self.settings.patch_size;
        let derivative=|field:&[[f32;2]],x:usize,z:usize|Vec2::new(field[z*n+(x+1)%n][0]-field[z*n+(x+n-1)%n][0],field[((z+1)%n)*n+x][0]-field[((z+n-1)%n)*n+x][0])*inv;
        for z in 0..n{for x in 0..n{let i=z*n+x;let h=derivative(&height,x,z);let a=derivative(&dx,x,z);let b=derivative(&dz,x,z);
            let jac=(1.0+a.x)*(1.0+b.y)-a.y*b.x;
            // Surface gradient after horizontal displacement (inverse Jacobian).
            let denom=jac.max(0.1);let slope=Vec2::new((h.x*(1.0+b.y)-h.y*b.x)/denom,((1.0+a.x)*h.y-a.y*h.x)/denom);
            let foam=((self.settings.foam_threshold-jac)*self.settings.foam_gain).clamp(0.0,1.0);
            fields.push([dx[i][0],height[i][0],dz[i][0],foam]);fields.push([slope.x,slope.y,jac,slope.length_squared()]);
        }}
        crate::water_field::append_mips(&mut fields,n,2);
        OceanFrame{n,size:self.settings.patch_size,time,fields}
    }
}

#[derive(Debug,Default)]
pub struct OceanCache {spectrum:Option<Spectrum>,frame:Option<Arc<OceanFrame>>}
impl OceanCache {
    pub fn frame(&mut self,settings:&OceanSpectrum,time:f32)->Arc<OceanFrame> {
        let settings=settings.sanitized();let time=if time.is_finite(){time}else{0.0};
        if self.spectrum.as_ref().is_none_or(|s|s.settings!=settings){self.spectrum=Some(Spectrum::new(settings));self.frame=None;}
        if self.frame.as_ref().is_none_or(|f|f.time!=time){self.frame=Some(Arc::new(self.spectrum.as_ref().unwrap().frame(time)));}
        self.frame.as_ref().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn spectrum_is_periodic_repeatable_and_has_no_mean_drift(){
        let s=OceanSpectrum::default();let mut a=OceanCache::default();let mut b=OceanCache::default();
        let f=a.frame(&s,2.5);assert_eq!(f.fields,b.frame(&s,2.5).fields);
        assert!(Arc::ptr_eq(&f,&a.frame(&s,2.5)));
        let repeated=a.frame(&s,122.5);assert_eq!(f.fields,repeated.fields);
        let h=f.fields[..f.n*f.n*2].as_chunks::<2>().0.iter().map(|v|v[0][1]).collect::<Vec<_>>();
        assert!(h.iter().sum::<f32>().abs()<0.001);assert!(h.iter().any(|h|h.abs()>0.1));
        let p=Vec2::new(7.25,-1.75);let x=f.sample(p);let y=f.sample(p+Vec2::splat(s.patch_size));assert!(x.0.distance(y.0)<1e-5);
        assert!(f.fields.iter().flatten().all(|v|v.is_finite()));
        let quiet=a.frame(&OceanSpectrum{significant_height:0.0,..s},0.0);assert!(quiet.fields.iter().step_by(2).all(|v|v[..3]==[0.0;3]));
    }
    #[test]
    #[ignore="explicit CPU ocean frame budget measurement"]
    fn profile_water_ocean_frame_cost(){
        for n in [16,32,64,128,256]{let settings=OceanSpectrum{resolution:n,..Default::default()};let mut cache=OceanCache::default();cache.frame(&settings,0.0);
            let mut timings=Vec::new();let mut bytes=0;
            for i in 1..=16{let start=std::time::Instant::now();let frame=cache.frame(&settings,i as f32/90.0);timings.push(start.elapsed().as_secs_f64()*1000.0);bytes=frame.fields.len()*16;std::hint::black_box(frame);}
            timings.sort_by(f64::total_cmp);eprintln!("[water ocean CPU] {n}x{n}: median {:.3} ms; upload {:.1} KiB/frame",timings[8],bytes as f64/1024.0);
        }
    }

}
