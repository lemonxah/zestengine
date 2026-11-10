//! Non-destructive water and island terrain layers. Data and sampling only;
//! scene composition belongs to engine and mesh/collider uploads to their hosts.
use glam::Vec2;
use serde::{Deserialize, Serialize};
use crate::{ObjectRef, TypedComponent, Water, WaterBody, WaterColumn};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaterTerrainBlend { #[default] Lower, Raise, Set }

/// Add beside Water to carve terrain from that body's authored footprint.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct WaterTerrainBrush {
    /// None applies to every terrain with water layers enabled.
    pub terrain: ObjectRef,
    pub blend: WaterTerrainBlend,
    pub falloff: f32,
    pub depth_scale: f32,
    pub bed_offset: f32,
    pub order: i32,
}
impl Default for WaterTerrainBrush {
    fn default()->Self {Self {terrain:ObjectRef::NONE,blend:WaterTerrainBlend::Lower,falloff:3.0,depth_scale:1.0,bed_offset:0.0,order:0}}
}
impl TypedComponent for WaterTerrainBrush {const NAME:&'static str="Water Terrain Brush";}

/// An island is a terrain layer, applied after water carving. Its object's
/// local Y=0 is the plateau; the outline/size and falloff define its banks.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct WaterIsland {
    pub terrain: ObjectRef,
    pub size:[f32;2],
    pub shape:Vec<[f32;2]>,
    pub smooth_shape:bool,
    pub falloff:f32,
    pub order:i32,
}
impl Default for WaterIsland {
    fn default()->Self {Self {terrain:ObjectRef::NONE,size:[10.0;2],shape:Vec::new(),smooth_shape:false,falloff:5.0,order:0}}
}
impl TypedComponent for WaterIsland {const NAME:&'static str="Water Island";}
impl WaterIsland {
    pub fn footprint(&self)->Water {Water {body:WaterBody::Lake,size:self.size,shape:self.shape.clone(),smooth_shape:self.smooth_shape,waves:Vec::new(),..Default::default()}}
}

#[derive(Clone, Debug, PartialEq)]
pub struct TerrainWaterLayer {pub key:u64,pub heights:Vec<f32>}

pub fn terrain_brush_weight(shore_distance:f32, falloff:f32)->f32 {
    if shore_distance>=0.0 {return 1.0;}
    if !falloff.is_finite() || falloff<=0.0 {return 0.0;}
    let t=(1.0+shore_distance/falloff).clamp(0.0,1.0);
    t*t*(3.0-2.0*t)
}
pub fn blend_terrain_height(base:f32,target:f32,weight:f32,mode:WaterTerrainBlend)->f32 {
    let target=match mode {WaterTerrainBlend::Lower=>base.min(target),WaterTerrainBlend::Raise=>base.max(target),WaterTerrainBlend::Set=>target};
    base+(target-base)*weight.clamp(0.0,1.0)
}

impl Water {
    /// Nearest authored column and signed distance to its bank (positive
    /// inside). Also samples just outside for terrain falloff and transitions;
    /// gameplay containment continues to use column()/height_at().
    pub fn shore_sample(&self,p:Vec2)->Option<(WaterColumn,f32)> {
        let still=|distance|Some((WaterColumn {surface:0.0,depth:self.depth.max(0.0),flow:self.current.into(),edge:0.0},distance));
        match self.body {
            WaterBody::Custom=>self.custom_mesh.as_ref()?.shore(p).map(|(surface,distance)|(WaterColumn{surface,depth:self.depth.max(0.0),flow:self.current.into(),edge:0.0},distance)),
            WaterBody::Ocean if self.shape.len()<3=>still(f32::INFINITY),
            WaterBody::River=>{
                let path=self.river_path();
                let mut inside:Option<(f32,WaterColumn,f32)>=None;
                let mut outside:Option<(WaterColumn,f32)>=None;
                for pair in path.windows(2) {
                    let(a,b)=(pair[0],pair[1]);let a2=Vec2::new(a.pos.x,a.pos.z);let delta=Vec2::new(b.pos.x,b.pos.z)-a2;
                    let t=if delta.length_squared()>1e-8 {((p-a2).dot(delta)/delta.length_squared()).clamp(0.0,1.0)}else{0.0};
                    let d=(a2+delta*t).distance(p);let half=(a.width+(b.width-a.width)*t).max(0.01)*0.5;
                    let bank=half-d;let rank=(d*65536.0+0.5).floor();
                    let column=WaterColumn {surface:a.pos.y+(b.pos.y-a.pos.y)*t,depth:(a.depth+(b.depth-a.depth)*t).max(0.0),
                        flow:a.dir.lerp(b.dir,t).normalize_or(glam::Vec3::X)*(a.speed+(b.speed-a.speed)*t)+glam::Vec3::from(self.current)+a.flow_offset.lerp(b.flow_offset,t),edge:d/half};
                    if bank>=0.0 && inside.as_ref().is_none_or(|(old,_,_)|rank<*old) {inside=Some((rank,column,bank));}
                    if outside.as_ref().is_none_or(|(_,old)|bank>*old) {outside=Some((column,bank));}
                }
                inside.map(|(_,c,d)|(c,d)).or(outside)
            }
            _ if self.shape.len()>=3=>{
                let outline=self.outline();let mut distance=f32::INFINITY;
                for(i,a)in outline.iter().enumerate(){let a=Vec2::from(*a);let b=Vec2::from(outline[(i+1)%outline.len()]);let ab=b-a;
                    let t=if ab.length_squared()>1e-12 {((p-a).dot(ab)/ab.length_squared()).clamp(0.0,1.0)}else{0.0};distance=distance.min(p.distance(a+ab*t));}
                still(if self.column(p).is_some(){distance}else{-distance})
            }
            _=>{
                let half=Vec2::from(self.size)*0.5;
                let q=Vec2::new(if half.x>0.0 {p.x.abs()-half.x}else{f32::NEG_INFINITY},if half.y>0.0 {p.y.abs()-half.y}else{f32::NEG_INFINITY});
                still(-(q.max(Vec2::ZERO).length()+q.max_element().min(0.0)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn banks_and_brush_blends_preserve_authored_water_semantics(){
        let w=Water {size:[4.0,6.0],..Default::default()};
        assert_eq!(w.shore_sample(Vec2::ZERO).unwrap().1,2.0);
        assert!((w.shore_sample(Vec2::new(3.0,4.0)).unwrap().1+2.0f32.sqrt()).abs()<1e-6);
        let river=Water {body:WaterBody::River,river:vec![crate::RiverPoint {position:[0.0,2.0,0.0],width:2.0,..Default::default()},crate::RiverPoint {position:[8.0,0.0,0.0],width:4.0,..Default::default()}],..Default::default()};
        for x in 0..=8 {let p=Vec2::new(x as f32,0.5);let(c,bank)=river.shore_sample(p).unwrap();let actual=river.column(p).unwrap();assert_eq!(c.surface,actual.surface);assert_eq!(c.depth,actual.depth);assert!(bank>0.0);}
        assert_eq!(terrain_brush_weight(-2.0,4.0),0.5);
        assert_eq!(blend_terrain_height(2.0,4.0,1.0,WaterTerrainBlend::Lower),2.0);
        assert_eq!(blend_terrain_height(2.0,-4.0,1.0,WaterTerrainBlend::Raise),2.0);
        assert_eq!(blend_terrain_height(2.0,-4.0,0.5,WaterTerrainBlend::Set),-1.0);
    }
}
