//! Cloth simulation resolution, independent of the visible mesh topology.
use serde::{Deserialize,Serialize};

/// A distance tier. The near tier uses the Mesh Cloth's ordinary settings.
#[derive(Clone,Debug,Serialize,Deserialize,PartialEq)]
#[serde(default)]
pub struct ClothLodLevel {
    /// Distance from viewer to object origin in world metres.
    pub distance:f32,
    /// Minimum particle spacing in object metres (never finer than the near tier).
    pub particle_spacing:f32,
    pub frequency:f32,
    pub iterations:u32,
}
impl Default for ClothLodLevel {
    fn default()->Self{Self{distance:10.0,particle_spacing:0.04,frequency:60.0,iterations:3}}
}

#[derive(Clone,Debug,Serialize,Deserialize,PartialEq)]
#[serde(default)]
pub struct ClothLod {
    pub enabled:bool,
    pub levels:Vec<ClothLodLevel>,
    /// Metres on each side of a threshold, to avoid switching as the viewer jitters.
    pub hysteresis:f32,
    /// Seconds to fade the render detail that cannot fit the new particle layout.
    pub transition:f32,
}
impl Default for ClothLod {
    fn default()->Self{Self{enabled:false,levels:vec![ClothLodLevel::default(),ClothLodLevel{distance:25.0,particle_spacing:0.1,frequency:30.0,iterations:2}],hysteresis:1.0,transition:0.25}}
}
impl ClothLod {
    fn valid(&self,index:usize)->Option<&ClothLodLevel>{
        index.checked_sub(1).and_then(|i|self.levels.get(i)).filter(|l|l.distance.is_finite()&&l.distance>0.0&&l.particle_spacing.is_finite()&&l.particle_spacing>=0.0)
    }
    /// Zero is the near tier; other values index the authored array plus one.
    /// Without a viewer, headless simulation uses the near tier deterministically.
    pub fn select(&self,distance:Option<f32>,current:usize)->usize{
        let Some(distance)=distance.filter(|d|d.is_finite())else{return 0};
        if !self.enabled{return 0;}
        let current_distance=self.valid(current).map_or(0.0,|l|l.distance);
        let h=if self.hysteresis.is_finite(){self.hysteresis.max(0.0)}else{0.0};
        let mut result=0;let mut threshold=0.0;
        for i in 1..=self.levels.len(){
            let Some(l)=self.valid(i)else{continue};
            let boundary=if l.distance<=current_distance{l.distance-h}else{l.distance+h};
            if distance>=boundary && l.distance>threshold{result=i;threshold=l.distance;}
        }
        result
    }
    pub fn settings(&self,level:usize,spacing:f32,frequency:f32,iterations:u32)->(f32,f32,u32){
        match self.valid(level).filter(|_|self.enabled){
            Some(l)=>(l.particle_spacing.max(spacing),if l.frequency.is_finite(){l.frequency.clamp(10.0,480.0)}else{frequency},l.iterations.clamp(1,32)),
            None=>(spacing,frequency,iterations),
        }
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn cloth_lod_threshold_hysteresis_unsorted_and_headless(){
        let mut lod=ClothLod{enabled:true,..Default::default()};
        assert_eq!(lod.select(Some(10.5),0),0);assert_eq!(lod.select(Some(11.0),0),1);
        assert_eq!(lod.select(Some(9.5),1),1);assert_eq!(lod.select(Some(8.9),1),0);
        assert_eq!(lod.select(Some(100.0),0),2);assert_eq!(lod.select(Some(0.0),2),0);
        assert_eq!(lod.select(None,2),0);
        lod.levels.swap(0,1);assert_eq!(lod.select(Some(15.0),0),2);assert_eq!(lod.select(Some(30.0),0),1);
        lod.levels[0].distance=f32::NAN;assert_eq!(lod.select(Some(30.0),1),2);
        lod.enabled=false;assert_eq!(lod.select(Some(100.0),2),0);
    }
}
