//! Conservative animated capture bounds without GPU readback or per-frame vertex walks.
use std::collections::BTreeMap;
use glam::{Mat4,Vec3};
use crate::{Vertex,MorphDeltas};

pub(crate) struct Bounds {
    joints:BTreeMap<u32,[Vec3;2]>,
    unweighted:Option<[Vec3;2]>,
    morph:Vec<[Vec3;2]>,
    // Malformed negative skin weights can extrapolate beyond the convex hull.
    max_scale:f32,
}
fn add(bounds:&mut Option<[Vec3;2]>,p:Vec3) {
    *bounds=Some(bounds.map_or([p,p],|b|[b[0].min(p),b[1].max(p)]));
}
impl Bounds {
    pub fn new(vertices:&[Vertex],shapes:&[MorphDeltas<'_>])->Self {
        let mut joints:BTreeMap<u32,[Vec3;2]>=BTreeMap::new();let mut unweighted=None;let mut max_scale=1.0f32;
        for v in vertices {
            let p=Vec3::from(v.position);let sum:f32=v.weights.iter().sum();
            if sum<=1e-5 {add(&mut unweighted,p);continue;}
            let positive:f32=v.weights.iter().filter(|w|**w>0.0).sum();max_scale=max_scale.max(positive/sum);
            for (&joint,&weight) in v.joints.iter().zip(&v.weights) {if weight>0.0 {
                joints.entry(joint).and_modify(|b|{b[0]=b[0].min(p);b[1]=b[1].max(p);}).or_insert([p,p]);
            }}
        }
        let morph=shapes.iter().map(|shape| {
            // Duplicate sparse entries for one vertex add in the morph shader.
            let mut deltas:BTreeMap<u32,Vec3>=BTreeMap::new();
            for &(v,p,_) in shape.iter() {if (v as usize)<vertices.len() {*deltas.entry(v).or_default()+=Vec3::from(p);}}
            deltas.values().fold([Vec3::ZERO;2],|b,p|[b[0].min(*p),b[1].max(*p)])
        }).collect();
        Self {joints,unweighted,morph,max_scale}
    }
    pub fn posed(&self,palette:&[Mat4],weights:&[f32])->Option<[Vec3;2]> {
        let mut delta=[Vec3::ZERO;2];
        for (b,&w) in self.morph.iter().zip(weights) {
            if !w.is_finite() {return None;}
            if w.abs()<=crate::morph::WEIGHT_EPS {continue;}
            delta[0]+=(b[0]*w).min(b[1]*w);delta[1]+=(b[0]*w).max(b[1]*w);
        }
        // Out-of-range influences contribute zero, but remain in the skin
        // shader's normalization sum. Include the origin in their convex hull.
        let mut result=Some([Vec3::ZERO;2]);
        if let Some(b)=self.unweighted {add(&mut result,b[0]+delta[0]);add(&mut result,b[1]+delta[1]);}
        for (&joint,b) in &self.joints {
            let Some(m)=palette.get(joint as usize) else {continue;};
            if !m.is_finite() {return None;}
            let lo=b[0]+delta[0];let hi=b[1]+delta[1];let c=(lo+hi)*0.5;let h=(hi-lo)*0.5;
            let centre=m.transform_point3(c);
            let half=m.x_axis.truncate().abs()*h.x+m.y_axis.truncate().abs()*h.y+m.z_axis.truncate().abs()*h.z;
            add(&mut result,centre-half);add(&mut result,centre+half);
        }
        result.map(|b|[b[0]*self.max_scale,b[1]*self.max_scale]).filter(|b|b.iter().all(|p|p.is_finite()))
    }
}

#[cfg(test)] mod tests {
    use super::*;
    #[test] fn animated_bounds_enclose_skinning_and_signed_duplicate_morphs() {
        let mut vertices=Vec::new();
        for x in [-2.0,0.0,3.0] {for y in [-1.0,2.0] {
            vertices.push(Vertex {position:[x,y,0.5],normal:[0.0,0.0,1.0],joints:[0,1,19,0],weights:[0.25,0.5,0.25,0.0],..Default::default()});
        }}
        vertices[0].weights=[0.0;4];vertices[1].weights=[-0.25,0.75,0.0,0.0];
        let shape0=[(0,[2.0,-1.0,0.5],[0.0;3]),(0,[1.0,0.0,1.0],[0.0;3]),(3,[-3.0,2.0,0.0],[0.0;3])];
        let shape1=[(1,[-2.0,3.0,1.0],[0.0;3]),(5,[1.0,-2.0,-4.0],[0.0;3])];
        let shapes=[shape0.as_slice(),shape1.as_slice()];let bounds=Bounds::new(&vertices,&shapes);
        for weights in [[0.0,0.0],[1.0,-2.0],[-1.0,0.5]] {for angle in [0.0,0.8,2.5] {
            let palette=[Mat4::from_translation(Vec3::new(3.0,2.0,-1.0))*Mat4::from_rotation_z(angle),
                Mat4::from_translation(Vec3::new(-2.0,1.0,2.0))*Mat4::from_rotation_x(-angle)*Mat4::from_scale(Vec3::new(1.0,2.0,0.5))];
            let tables=crate::morph::MorphTables::build(&vertices,&shapes);
            let morphed=crate::morph::morph_reference(&vertices,&tables,&weights);
            let posed=crate::skin_headless::skin_reference(&morphed,&palette);
            let b=bounds.posed(&palette,&weights).unwrap();
            for v in posed {let p=Vec3::from(v.position);assert!(p.cmpge(b[0]-Vec3::splat(1e-4)).all() && p.cmple(b[1]+Vec3::splat(1e-4)).all(),"{p:?} outside {b:?}");}
        }}
    }
}
