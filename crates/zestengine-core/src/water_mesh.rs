//! Authoritative imported water surfaces. Triangle footprints preserve holes
//! and changing heights; vertical/degenerate faces do not invent water columns.
use glam::{Vec2,Vec3};
use std::{collections::HashMap,hash::{Hash,Hasher}};
#[derive(Clone,Debug)]
pub struct WaterMeshNode {pub bounds:[f32;4],pub escape:usize,pub triangle:Option<usize>}
#[derive(Clone,Debug,Default)]
pub struct WaterMesh {
    pub triangles:Vec<[Vec3;3]>,
    pub nodes:Vec<WaterMeshNode>,
    pub boundary:Vec<[Vec3;2]>,
    pub bounds:[Vec3;2],
    pub key:u64,
}
fn xz(p:Vec3)->Vec2{Vec2::new(p.x,p.z)}
fn cross(a:Vec2,b:Vec2)->f32{a.x*b.y-a.y*b.x}
impl WaterMesh {
    pub fn new(vertices:&[Vec3],indices:&[u32])->Self{
        let mut mesh=Self{bounds:[Vec3::splat(f32::INFINITY),Vec3::splat(f32::NEG_INFINITY)],..Default::default()};
        let mut hash=std::collections::hash_map::DefaultHasher::new();
        for i in indices.chunks_exact(3){
            let(Some(a),Some(b),Some(c))=(vertices.get(i[0]as usize),vertices.get(i[1]as usize),vertices.get(i[2]as usize))else{continue};
            if !a.is_finite()||!b.is_finite()||!c.is_finite()||cross(xz(*b-*a),xz(*c-*a)).abs()<1e-10{continue;}
            let triangle=[*a,*b,*c];for p in triangle{mesh.bounds[0]=mesh.bounds[0].min(p);mesh.bounds[1]=mesh.bounds[1].max(p);for v in p.to_array(){v.to_bits().hash(&mut hash);}}
            mesh.triangles.push(triangle);
        }
        mesh.key=hash.finish();
        let mut edges=HashMap::new();
        for t in &mesh.triangles{for(a,b)in[(t[0],t[1]),(t[1],t[2]),(t[2],t[0])]{
            let key=|p:Vec3|p.to_array().map(|v|if v==0.0{0}else{v.to_bits()});let(ka,kb)=(key(a),key(b));
            let e=edges.entry(if ka<kb{(ka,kb)}else{(kb,ka)}).or_insert((0,[a,b]));e.0+=1;
        }}
        mesh.boundary=edges.into_values().filter_map(|(n,e)|(n==1).then_some(e)).collect();
        let mut order:Vec<_>=(0..mesh.triangles.len()).collect();if !order.is_empty(){mesh.build(&mut order);}
        mesh
    }
    fn build(&mut self,order:&mut[usize]){
        let index=self.nodes.len();let(mut lo,mut hi)=(Vec2::splat(f32::INFINITY),Vec2::splat(f32::NEG_INFINITY));
        for &i in order.iter(){for p in self.triangles[i]{lo=lo.min(xz(p));hi=hi.max(xz(p));}}
        self.nodes.push(WaterMeshNode{bounds:[lo.x,lo.y,hi.x,hi.y],escape:0,triangle:None});
        if order.len()==1{self.nodes[index].triangle=Some(order[0]);}else{
            let axis=usize::from(hi.y-lo.y>hi.x-lo.x);let mid=order.len()/2;
            order.select_nth_unstable_by(mid,|&a,&b|{let center=|i:usize|self.triangles[i].iter().map(|p|xz(*p)[axis]).sum::<f32>();center(a).total_cmp(&center(b)).then(a.cmp(&b))});
            let(a,b)=order.split_at_mut(mid);self.build(a);self.build(b);
        }
        self.nodes[index].escape=self.nodes.len();
    }
    pub fn column(&self,p:Vec2)->Option<f32>{
        let mut best=None::<f32>;let mut i=0;
        while i<self.nodes.len(){let n=&self.nodes[i];let b=n.bounds;
            if p.x<b[0]||p.y<b[1]||p.x>b[2]||p.y>b[3]{i=n.escape;continue;}
            if let Some(t)=n.triangle{let[a,b,c]=self.triangles[t];let ab=xz(b-a);let ac=xz(c-a);let d=xz(a)-p;let denom=cross(ab,ac);
                let u=cross(-d,ac)/denom;let v=cross(ab,-d)/denom;
                if u>=-1e-6&&v>=-1e-6&&u+v<=1.000001{let height=a.y+(b.y-a.y)*u+(c.y-a.y)*v;best=Some(best.map_or(height,|old|old.max(height)));}
            }i+=1;
        }best
    }
    /// Height at the nearest bank and signed distance for terrain falloff.
    pub fn shore(&self,p:Vec2)->Option<(f32,f32)>{
        let inside=self.column(p);let mut nearest=None::<(f32,f32)>;
        for &[a,b]in &self.boundary{let ab=xz(b-a);let t=((p-xz(a)).dot(ab)/ab.length_squared().max(1e-12)).clamp(0.0,1.0);let d=p.distance(xz(a)+ab*t);
            if nearest.is_none_or(|(_,old)|d<old){nearest=Some((a.y+(b.y-a.y)*t,d));}}
        match(inside,nearest){(Some(h),Some((_,d)))=>Some((h,d)),(Some(h),None)=>Some((h,f32::INFINITY)),(None,Some((h,d)))=>Some((h,-d)),_=>None}
    }
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn disconnected_sloped_triangles_preserve_gaps_and_ignore_bad_faces(){
        let v=[Vec3::new(-2.0,1.0,0.0),Vec3::new(-1.0,2.0,0.0),Vec3::new(-2.0,1.0,2.0),Vec3::new(1.0,4.0,0.0),Vec3::new(2.0,4.0,0.0),Vec3::new(2.0,4.0,2.0)];
        let m=WaterMesh::new(&v,&[0,1,2,3,4,5,0,0,1,0,1,99]);assert_eq!(m.triangles.len(),2);assert_eq!(m.boundary.len(),6);
        assert!((m.column(Vec2::new(-1.5,0.5)).unwrap()-1.5).abs()<1e-6);assert_eq!(m.column(Vec2::new(1.75,0.5)),Some(4.0));assert!(m.column(Vec2::new(0.0,0.5)).is_none());assert!(m.shore(Vec2::new(0.0,0.5)).unwrap().1<0.0);
        assert_eq!(m.key,WaterMesh::new(&v,&[0,1,2,3,4,5]).key);
    }
}
